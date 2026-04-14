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
        """
        scores: Tensor[..., K] of raw logits
        mask:   same shape as scores, 1.0 for valid, 0.0 for invalid
        """
        if mask is None:
            # no masking → just pass logits straight through
            logits = scores
            self.num_cat = scores.shape[-1]
            self.log_n   = math.log(self.num_cat)
        else:
            # mask out invalid logits with -inf
            logits = scores.masked_fill(mask == 0, float('-inf'))
            # detect rows where everything was masked → give uniform logits
            all_masked = (mask.sum(dim=-1, keepdim=True) == 0)
            if all_masked.any():
                logits = torch.where(
                    all_masked.expand_as(logits),
                    torch.zeros_like(logits),
                    logits
                )
            # record actual number of valid categories per row
            self.num_cat = mask.sum(dim=-1)            # Tensor[...,]
            # log of count, for normalization
            self.log_n   = (self.num_cat + 1e-17).log()

        # build the distribution with *logits* (internally uses log‐softmax)
        self.cat_distr = TorchCategorical(logits=logits)

    @property
    def probs(self):
        return self.cat_distr.probs

    @property
    def logits(self):
        return self.cat_distr.logits

    @property
    def entropy(self):
        # use the built-in entropy if no mask
        if not hasattr(self, 'mask') or self.mask is None:
            ent = self.cat_distr.entropy()
            # zero‐out entropy if only one class
            return ent * (self.num_cat != 1)
        # otherwise compute masked entropy manually
        p     = self.probs                 # [..., K]
        logp  = self.logits
        valid = p > 0                      # mask away the 0 × (-inf) terms
        ent   = -(logp[valid] * p[valid]).sum(dim=-1)
        return ent * (self.num_cat != 1).float()

    @property
    def normalized_entropy(self):
        # divide by log(number of valid classes)
        return self.entropy / (self.log_n + 1e-17)

    def sample(self):
        return self.cat_distr.sample()

    def rsample(self, temperature=None, gumbel_noise=None, eps=1e-6):
        # Gumbel‐softmax sampling (as before), but using logits & mask safely
        if gumbel_noise is None:
            with torch.no_grad():
                u = torch.empty_like(self.probs).uniform_(eps, 1 - eps)
                gumbel_noise = -(-u.log()).log()

        logits = self.logits if temperature is None else self.logits / temperature
        noisy = logits + gumbel_noise
        # mask out again (since adding gumbel may disturb -inf)
        if hasattr(self, 'mask') and self.mask is not None:
            noisy = noisy.masked_fill(self.mask == 0, float('-inf'))
        y = F.softmax(noisy, dim=-1)
        if temperature is None:
            # hard sample
            idx = y.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(y).scatter_(-1, idx, 1.0)
            return y_hard, gumbel_noise
        else:
            return y, gumbel_noise

    def log_prob(self, value):
        return self.cat_distr.log_prob(value)
