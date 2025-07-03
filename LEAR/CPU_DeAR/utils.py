# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os
import time
import uuid
from functools import partial
import sys

import boto3
import botocore.exceptions
import torch
from torch.distributions.categorical import Categorical as TorchCategorical
from torch.distributions.utils import lazy_property
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau


class CloudWatchLogsHandler(logging.Handler):
    """
    A minimal, thread-unsafe (only one process at a time) handler that ships each
    log record to a dedicated CloudWatch Logs stream.
    """
    def __init__(self, log_group: str, log_stream: str, session: boto3.Session):
        super().__init__()
        self.log_group  = log_group
        self.log_stream = log_stream
        self.client     = session.client("logs")
        self.sequence_token = None          # updated after every put_log_events

        try:
            self.client.create_log_stream(
                logGroupName=log_group, logStreamName=log_stream
            )
        except self.client.exceptions.ResourceAlreadyExistsException:
            # fetch existing upload token
            resp = self.client.describe_log_streams(
                logGroupName=log_group, logStreamNamePrefix=log_stream
            )
            if resp["logStreams"]:
                self.sequence_token = resp["logStreams"][0].get("uploadSequenceToken")


    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            event = {"timestamp": int(time.time() * 1000), "message": msg}

            kwargs = dict(
                logGroupName = self.log_group,
                logStreamName = self.log_stream,
                logEvents = [event],
            )
            if self.sequence_token:
                kwargs["sequenceToken"] = self.sequence_token

            resp = self.client.put_log_events(**kwargs)
            self.sequence_token = resp["nextSequenceToken"]
        except botocore.exceptions.ClientError as e:
            # CloudWatch rejected the event; fall back to console so nothing is lost
            print("CloudWatchLogsHandler error:", e, file=sys.stderr)
        except botocore.exceptions.EndpointConnectionError as e:
            # CloudWatch is not available, e.g. when running on a local machine
            print("CloudWatchLogsHandler error:", e, file=sys.stderr)


class AverageMeter:
    def __init__(self):
        self.value = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


class Logger():
    """ Create a logger for training and testing.
    
    The logger has three handlers:
    1. Console handler: prints log messages to the console.
    2. File handler: writes log messages to a file.
    3. File handler: write predictions to a file.
    """

    def __init__(self, log_dir: str, checkpoint_name: str,
                 *, cw_group: str = "thesis-mt-runs") -> None:
        print(f"Creating logger at {log_dir} with name {checkpoint_name}.")
        self.logger = logging.getLogger("general_logger")
        self.logger.setLevel(logging.INFO)
        self.checkpoint_name = checkpoint_name
        self.log_dir = log_dir

        # Create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create file handler and set level to debug
        fh = logging.FileHandler(os.path.join(log_dir, f"{checkpoint_name}.log"), mode='w')
        fh.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(message)s", "%d-%m-%Y %H:%M:%S")

        # Add formatter to handlers
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

        creds_file = os.path.join(os.getcwd(), "aws.creds")

        if os.path.isfile(creds_file):
            # Local laptop → use the special credentials file + profile thesis_logger
            os.environ["AWS_SHARED_CREDENTIALS_FILE"] = creds_file
            session = boto3.Session(profile_name="thesis_logger")
        else:
            # Remote cluster → fall back to ~/.aws credentials, profile default
            session = boto3.Session(profile_name="default")

        # One stream per run → easy retrieval later
        cw_stream = f"{checkpoint_name}-{uuid.uuid4().hex[:8]}"
        cw_handler = CloudWatchLogsHandler(cw_group, cw_stream, session)
        cw_handler.setLevel(logging.INFO)
        cw_handler.setFormatter(formatter)
        self.logger.addHandler(cw_handler)

        # Record where the CloudWatch data lives so you can print / store it
        self.logger.info(f"CloudWatch stream: {cw_group}/{cw_stream}")

    
        self.logger.info("Logger initialized.")

    def info(self, message):
        self.logger.info(message)

    def log_state(self, state):
        with open(f"{self.log_dir}{self.checkpoint_name}.tsv", 'a') as f:
            f.write(f"{state}\n")


def get_logger(file_name):
    logger = logging.getLogger("general_logger")
    handler = logging.FileHandler(file_name, mode='w')
    formatter = logging.Formatter("%(asctime)s - %(message)s", "%d-%m-%Y %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def get_lr_scheduler(logger, optimizer, mode='max', factor=0.5, patience=10, threshold=1e-4, threshold_mode='rel'):
    def reduce_lr(self, epoch):
        ReduceLROnPlateau._reduce_lr(self, epoch)
        logger.info(f"learning rate is reduced by factor {factor}!")

    lr_scheduler = ReduceLROnPlateau(optimizer, mode, factor, patience, False, threshold, threshold_mode)
    lr_scheduler._reduce_lr = partial(reduce_lr, lr_scheduler)
    return lr_scheduler


def clamp_grad(v, min_val, max_val):
    if v.requires_grad:
        v_tmp = v.expand_as(v)
        v_tmp.register_hook(lambda g: g.clamp(min_val, max_val))
        return v_tmp
    return v


def length_to_mask(length):
    with torch.no_grad():
        batch_size = length.shape[0]
        max_length = length.data.max()
        range = torch.arange(max_length, dtype=torch.int64, device=length.device)
        range_expanded = range[None, :].expand(batch_size, max_length)
        length_expanded = length[:, None].expand_as(range_expanded)
        return (range_expanded < length_expanded).float()


class Categorical:
    def __init__(self, scores, mask=None):
        self.mask = mask
        if mask is None:
            self.cat_distr = TorchCategorical(F.softmax(scores, dim=-1))
            self.n = scores.shape[0]
            self.log_n = math.log(self.n)
        else:
            self.n = self.mask.sum(dim=-1)
            self.log_n = (self.n + 1e-17).log()
            self.cat_distr = TorchCategorical(Categorical.masked_softmax(scores, self.mask))

    @lazy_property
    def probs(self):
        return self.cat_distr.probs

    @lazy_property
    def logits(self):
        return self.cat_distr.logits

    @lazy_property
    def entropy(self):
        if self.mask is None:
            return self.cat_distr.entropy() * (self.n != 1)
        else:
            entropy = - torch.sum(self.cat_distr.logits * self.cat_distr.probs * self.mask, dim=-1)
            does_not_have_one_category = (self.n != 1.0).to(dtype=torch.float32)
            # to make sure that the entropy is precisely zero when there is only one category
            return entropy * does_not_have_one_category

    @lazy_property
    def normalized_entropy(self):
        return self.entropy / (self.log_n + 1e-17)

    def sample(self):
        return self.cat_distr.sample()

    def rsample(self, temperature=None, gumbel_noise=None, eps=1e-5):
        if gumbel_noise is None:
            with torch.no_grad():
                uniforms = torch.empty_like(self.probs).uniform_()
                uniforms = uniforms.clamp(min=eps, max=1 - eps)
                gumbel_noise = -(-uniforms.log()).log()

        elif gumbel_noise.shape != self.probs.shape:
            raise ValueError

        if temperature is None:
            with torch.no_grad():
                scores = (self.logits + gumbel_noise)
                scores = Categorical.masked_softmax(scores, self.mask)
                sample = torch.zeros_like(scores)
                sample.scatter_(-1, scores.argmax(dim=-1, keepdim=True), 1.0)
                return sample, gumbel_noise
        else:
            scores = (self.logits + gumbel_noise) / temperature
            sample = Categorical.masked_softmax(scores, self.mask)
            return sample, gumbel_noise

    def log_prob(self, value):
        if value.dtype == torch.long:
            if self.mask is None:
                return self.cat_distr.log_prob(value)
            else:
                return self.cat_distr.log_prob(value) * (self.n != 0.).to(dtype=torch.float32)
        else:
            max_values, mv_idxs = value.max(dim=-1)
            relaxed = (max_values - torch.ones_like(max_values)).sum().item() != 0.0
            if relaxed:
                raise ValueError("The log_prob can't be calculated for the relaxed sample!")
            return self.cat_distr.log_prob(mv_idxs) * (self.n != 0.).to(dtype=torch.float32)

    @staticmethod
    def masked_softmax(logits, mask):
        """
        This method will return valid probability distribution for the particular instance if its corresponding row
        in the `mask` matrix is not a zero vector. Otherwise, a uniform distribution will be returned.
        This is just a technical workaround that allows `Categorical` class usage.
        If probs doesn't sum to one there will be an exception during sampling.
        """
        if mask is not None:
            probs = F.softmax(logits, dim=-1) * mask
            probs = probs + (mask.sum(dim=-1, keepdim=True) == 0.).to(dtype=torch.float32)
            Z = probs.sum(dim=-1, keepdim=True)
            return probs / Z
        else:
            return F.softmax(logits, dim=-1)
