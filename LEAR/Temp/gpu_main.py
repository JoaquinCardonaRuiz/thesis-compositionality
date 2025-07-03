import argparse
from collections import defaultdict
import os
import random
import time
import unicodedata
from functools import partial

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import HRLModel, PAD_token, EOS_token
from utils import AverageMeter
from utils import Logger

import re
import json

from functools import partial

try:
    from torch.amp import autocast
    _AMP_HAS_DEVICE_TYPE = True
except ImportError:
    from torch.cuda.amp import autocast
    _AMP_HAS_DEVICE_TYPE = False

USE_CUDA = torch.cuda.is_available()
device   = torch.device("cuda" if USE_CUDA else "cpu")



global_step = 0


class Lang:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {"x1": 3, "x2": 4, "x3": 5, "x4": 6}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "x1", 4: "x2", 5: "x3", 6: "x4"}
        self.n_words = 7  # Count default tokens

    def vocab_size(self):
        return len(self.word2index.keys())

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count default tokens

        for word in keep_words:
            self.index_word(word)


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def data_file_process(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip().lower().split('\t') for line in lines]
        lines_norm = []
        for line in lines:
            input = line[0].strip(' .')
            output = line[1].strip('"').replace('?', input.split(' ')[0].lower())
            input_tokens = input.split()
            if len(input_tokens) <= 1:
                continue
            line_norm = [input, output]
            lines_norm.append(line_norm)
    return lines_norm

def read_data(lang1, lang2, task_name):
    print("Reading dataset from task {}...".format(task_name))

    file_train = f'{task_name}_data/train.tsv'
    file_dev = f'{task_name}_data/dev.tsv'
    file_test = f'{task_name}_data/test.tsv'
    file_gen = f'{task_name}_data/gen.tsv'

    pairs_train = data_file_process(file_train)
    pairs_dev = data_file_process(file_dev)
    pairs_test = data_file_process(file_test)
    pairs_gen = data_file_process(file_gen)

    _input_lang = Lang(lang1)
    _output_lang = Lang(lang2)

    return _input_lang, _output_lang, pairs_train, pairs_dev, pairs_test, pairs_gen

def prepare_dataset(lang1, lang2, task_name):
    global input_lang
    global output_lang
    assert task_name == "cogs" or task_name == "slog"
    print(f"Preparing dataset for task {task_name}...")
    input_lang, output_lang, pairs_train, pairs_dev, pairs_test, pairs_gen = read_data(lang1, lang2, task_name)

    # Dynamically build vocab from the actual dataset (SLOG or COGS)
    for pair in pairs_train + pairs_dev + pairs_test + pairs_gen:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])


    encode_token_filename = f'./{task_name}_data/preprocess/encode_tokens.txt'
    with open(encode_token_filename, 'r') as f:
        encode_tokens = f.readlines()
        for encode_token in encode_tokens:
            input_lang.index_word(encode_token.strip("\n"))

    decode_entity_filename = f'./{task_name}_data/preprocess/entity'
    with open(decode_entity_filename, 'r') as f:
        decode_tokens = f.readlines()
        for decode_token in decode_tokens:
            output_lang.index_word(decode_token.strip("\n"))
    decode_caus_predicate_filename = f'./{task_name}_data/preprocess/caus_predicate'
    with open(decode_caus_predicate_filename, 'r') as f:
        decode_tokens = f.readlines()
        for decode_token in decode_tokens:
            output_lang.index_word(decode_token.strip("\n"))
    decode_unac_predicate_filename = f'./{task_name}_data/preprocess/unac_predicate'
    with open(decode_unac_predicate_filename, 'r') as f:
        decode_tokens = f.readlines()
        for decode_token in decode_tokens:
            output_lang.index_word(decode_token.strip("\n"))

    return input_lang, output_lang, pairs_train, pairs_dev, pairs_test, pairs_gen

def get_bound_idx(pairs, length):
    # Assume that pairs are already sorted.
    # Return the max index in pairs under certain length.
    # Warning: Will return empty when length is greater than max length in pairs.
    index = 0
    for i, pair in enumerate(pairs):
        if len(pair[0].split()) <= length:
            index = i
        else:
            return index + 1

def indexes_from_sentence(lang, sentence, type):
    if type == 'input':
        return [lang.word2index[word] for word in sentence.split(' ')]
    if type == 'output':
        return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def make_path_preparations(args, run_mode):
    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    if run_mode == 'train':

        if not os.path.exists(args.train_logs_path):
            os.makedirs(args.train_logs_path)

        # "thesis-bucket-jcardonaruiz"
        _logger = Logger(args.train_logs_path, args.checkpoint, cw_group = "thesis-train-logs")
        _logger.info(f"random seed: {seed}")

        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        _logger.info(f"checkpoint's dir is: {args.model_dir}")
    else:
        run_name = args.checkpoint.split('/')[-2]
        checkpoint_name = args.checkpoint.split('/')[-1].replace('.mdl', '')
        log_dir = args.test_logs_path + run_name
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        _logger = Logger(log_dir, checkpoint_name, cw_group = "thesis-test-logs")
    return _logger

def prepare_optimisers(args, logger, high_parameters, low_parameters):
    if args.high_optimizer == "adam":
        high_opt_class = torch.optim.Adam
    elif args.high_optimizer == "amsgrad":
        high_opt_class = partial(torch.optim.Adam, amsgrad=True)
    elif args.high_optimizer == "adadelta":
        high_opt_class = torch.optim.Adadelta
    else:
        high_opt_class = torch.optim.SGD

    if args.low_optimizer == "adam":
        low_opt_class = torch.optim.Adam
    elif args.low_optimizer == "amsgrad":
        low_opt_class = partial(torch.optim.Adam, amsgrad=True)
    elif args.low_optimizer == "adadelta":
        low_opt_class = torch.optim.Adadelta
    else:
        low_opt_class = torch.optim.SGD

    optimizer = {"high": high_opt_class(params=high_parameters, lr=args.high_lr, weight_decay=args.l2_weight),
                 "low": low_opt_class(params=low_parameters, lr=args.low_lr, weight_decay=args.l2_weight)}

    return optimizer


def test(test_data, model, example2type, args, log_file=None):
    acc, type_right_count = run_eval(test_data, model, args, "Test",
                                     logger, example2type=example2type)
    return acc, type_right_count

def validate(valid_data, model, epoch, args, logger):
    acc, _ = run_eval(valid_data, model, args, f"Val E{epoch+1}", logger)
    return acc


def collate_batch(batch, *, lang):
    sents   = [pair[0] for pair in batch]
    token_l = [indexes_from_sentence(lang, s, "input") for s in sents]
    lens    = [len(t) for t in token_l]
    max_L   = max(lens)
    padded  = [tl + [PAD_token]*(max_L-len(tl)) for tl in token_l]
    tokens_batch = torch.as_tensor(padded, dtype=torch.long)
    lengths      = torch.as_tensor(lens,   dtype=torch.long)
    return batch, tokens_batch, lengths


def train(train_data, valid_data, model, optimizer, epoch, args, logger,
          total_batch_num, data_len, regular_weight):
    
    logger.info(f"Using device {device.type} for training")
    batch_time_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    comp_reward_meter = AverageMeter()
    sem_reward_meter = AverageMeter()
    comp_loss_meter = AverageMeter()
    sem_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    prob_ratio_meter = AverageMeter()
    reward_std_meter = AverageMeter()


    model.train()
    start = time.time()

    # change for debug
    #train_data = train_data[5204:5208]

    collate_fn = partial(collate_batch, lang=input_lang)
    loader = DataLoader(train_data,
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=collate_fn,
                        pin_memory=(device.type=="cuda"),
                        num_workers=4)

    val_accuracy = 0.0

    for batch_idx, (pair_batch, tokens_batch, lengths) in enumerate(loader):
        # Move to GPU if available
        
        if device.type == "cuda":
            tokens_batch = tokens_batch.to(device, non_blocking=True)
            lengths      = lengths.to(device,       non_blocking=True)

        batch_start = time.time()
        B = tokens_batch.size(0)          # true batch
        # change for debug
        N = sample_num = args.sample_num               # Monte-Carlo clones

        tokens_batch = tokens_batch.to(device, non_blocking=(device.type=="cuda"))
        
        batch_fwd, _, _, _ = model(pair_batch,            # list of length B
                                    tokens_batch,          # (B,L)
                                    sample_num,
                                    is_test=False,
                                    epoch=epoch)

        # batch_fwd is List[ Tuple[composer_rl_dict, solver_rl_dict] ] of length B*N
        composer_log_probs  = torch.stack([comp["log_prob"] for comp, _ in batch_fwd]).squeeze(1)   # (B·N,)
        semantic_log_probs  = torch.stack([sol ["log_prob"] for _,  sol in batch_fwd]).squeeze(1)   # (B·N,)
        
        #print(f"composer_log_probs shape: {composer_log_probs.shape}")
        #print(f"composer_log_probs: {composer_log_probs}")
        #print(f"semantic_log_probs shape: {semantic_log_probs.shape}")
        #print(f"semantic_log_probs: {semantic_log_probs}")

        composer_rewards    = torch.stack([comp["reward"] for comp, _ in batch_fwd]).squeeze(1)     # (B·N,)
        solver_rewards      = torch.stack([sol ["reward"] for _,  sol in batch_fwd]).squeeze(1)     # (B·N,)
        #print(f"composer_rewards shape: {composer_rewards.shape}")
        #print(f"composer_rewards: {composer_rewards}")
        #print(f"solver_rewards shape: {solver_rewards.shape}")
        #print(f"solver_rewards: {solver_rewards}")


        # accuracy / entropy the same way
        good_solver  = torch.isclose(solver_rewards,  torch.tensor(1.0, device=solver_rewards.device), atol=1e-4)
        good_composer = torch.isclose(composer_rewards, torch.tensor(1.0, device=composer_rewards.device), atol=1e-4)
        accuracy_samples = (good_solver & good_composer).float()
        #print(f"accuracy_samples shape: {accuracy_samples.shape}")
        #print(f"accuracy_samples: {accuracy_samples}")

        rewards_all        = solver_rewards + composer_rewards
        #print(f"rewards_all shape: {rewards_all.shape}")
        #print(f"rewards_all: {rewards_all}")

        # Optionally normalize (baseline)
        composer_advantage = composer_rewards - composer_rewards.mean()
        solver_advantage = solver_rewards - solver_rewards.mean()
        #print(f"composer_advantage shape: {composer_advantage.shape}")
        #print(f"composer_advantage: {composer_advantage}")
        #print(f"solver_advantage shape: {solver_advantage.shape}")
        #print(f"solver_advantage: {solver_advantage}")  


        # ── compute the two losses exactly as before ──
        loss_composer = -(composer_advantage.detach() * composer_log_probs).mean()
        loss_solver   = -(solver_advantage.detach()   * semantic_log_probs).mean()
        #print(f"loss_composer: {loss_composer}")
        #print(f"loss_solver: {loss_solver}")
        loss = 0.5 * (loss_solver + loss_composer)     # scalar for logging
        #print(f"loss: {loss}")

        # —— Composer step —— 
        # temporarily freeze solver weights
        for p in model.get_low_parameters(): p.requires_grad = False

        optimizer["high"].zero_grad()
        loss_composer.backward(retain_graph=True)
        if args.clip_grad_norm>0:
            torch.nn.utils.clip_grad_norm_(model.get_high_parameters(),
                                        args.clip_grad_norm,
                                        norm_type=float("inf"))
        optimizer["high"].step()

        # unfreeze solver
        for p in model.get_low_parameters(): p.requires_grad = True

        # —— Solver step —— 
        # temporarily freeze composer weights
        for p in model.get_high_parameters(): p.requires_grad = False

        optimizer["low"].zero_grad()
        loss_solver.backward()   # no need for retain_graph now
        if args.clip_grad_norm>0:
            torch.nn.utils.clip_grad_norm_(model.get_low_parameters(),
                                        args.clip_grad_norm,
                                        norm_type=float("inf"))
        optimizer["low"].step()

        # unfreeze composer
        for p in model.get_high_parameters(): p.requires_grad = True


        total_batch_num += B
        n_tokens = B * N
        
        prob_ratio_meter.update(abs(1.0 - loss.detach().item()), n_tokens)
        
        with torch.no_grad():              # values already detached
            ce_loss_meter.update(rewards_all.mean().item(), n_tokens)
            comp_loss_meter.update(loss_composer.item(), n_tokens)
            sem_loss_meter.update(loss_solver.item(), n_tokens)
            accuracy_meter.update(accuracy_samples[0].item(), 1)
            reward_std_meter.update(rewards_all.std().item(), n_tokens)
            comp_reward_meter.update(composer_rewards.mean().item(), n_tokens)
            sem_reward_meter.update(solver_rewards.mean().item(), n_tokens)

        # time meters -----------------------------------------------------
        batch_time_meter.update(time.time() - batch_start)

        # periodic console / file log  -----------------------------------
        if (batch_idx + 1) % args.log_every == 0:
            logger.info(
                f"[E{epoch+1:02d}] "
                f"[{batch_idx+1:05d}/{len(loader):05d}] "
                f"CE Loss{ce_loss_meter.avg: .4f} | "
                f"Comp Loss{comp_loss_meter.avg: .4f} | "
                f"Sem Loss{sem_loss_meter.avg: .4f} | "
                f"Comp R {comp_reward_meter.avg: .4f} | "
                f"Sem R {sem_reward_meter.avg: .4f} | "
                f"Acc {accuracy_meter.avg: .4f} | "
                f"R σ {reward_std_meter.avg: .3f} | "
                f"ProbΔ {prob_ratio_meter.avg: .3f} | "
                f"Batch Time {batch_time_meter.avg:5.2f} sec |"
                f"Est. Epoch Time {batch_time_meter.avg * len(loader) / 60:5.2f} min"
            )
            # reset short-horizon meters so the next block is fresh
            batch_time_meter.reset()
            ce_loss_meter.reset()
            accuracy_meter.reset()
            prob_ratio_meter.reset()
            reward_std_meter.reset()

        del semantic_log_probs, composer_log_probs
        # uncomment if hit CUDA OOM
        #torch.cuda.empty_cache()

        # periodic validation  -------------------------------------------
        if (batch_idx + 1) % args.val_every == 0:
            val_accuracy = validate(valid_data, model, epoch, args, logger)
            logger.info(f"[E{epoch+1:02d}]  [{batch_idx+1:05d}/{len(loader):05d}]  Val accuracy: {val_accuracy:.4f}")
            model.train()                      # make sure we are back to train mode

    return val_accuracy, total_batch_num

def run_eval(dataset, model, args, desc, logger, example2type=None):
    collate_fn = partial(collate_batch, lang=input_lang)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,          # same knob as training
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        num_workers=4,
    )

    model.eval()
    acc_meter = AverageMeter()
    type_right = defaultdict(list)        # cat → list[0/1]

    with torch.no_grad(), tqdm(total=len(loader), desc=desc) as bar:
        for pair_batch, tokens_batch, _ in loader:
            tokens_batch = tokens_batch.to(device, non_blocking=True)

            fwd, _, _, state = model(pair_batch, tokens_batch, sample_num=1, is_test=True)

            comp_r = torch.stack([fi[0]["reward"] for fi in fwd]).squeeze(-1)
            solv_r = torch.stack([fi[1]["reward"] for fi in fwd]).squeeze(-1)
            per_ex_acc = ((comp_r == 1) & (solv_r == 1)).float()
            batch_acc = per_ex_acc.mean()

            acc_meter.update(batch_acc.item(), n=len(pair_batch))

            bar.set_postfix(acc=acc_meter.avg * 100)

            if example2type:
                for pair, acc_val, st in zip(pair_batch, per_ex_acc, state):
                    cat = example2type[pair[0]]
                    type_right[cat].append(acc_val.item())

                    logger.log_state({
                        "pair": pair,
                        "category": cat,
                        "acc": acc_val.item(),
                        "pred_edges": st["pred"],
                        "gold_edges": st["gold"],
                        "comp_reward": st["comp_reward"],
                        "sol_reward": st["sol_reward"]
                    })
            bar.update()
    return acc_meter.avg, dict(type_right)


def get_alignment(alignment_filename, input_lang, output_lang):
    with open(alignment_filename, 'r') as f:
        alignments = json.load(f)

    alignments_idx = {}
    for enct in alignments:
        assert alignments[enct] != ''
        dects = alignments[enct].split()
        dects = [" ".join(dect.split('-')) for dect in dects]
        enct_idx = input_lang.word2index[enct]
        dects_idx = [output_lang.word2index[dect] for dect in dects]
        alignments_idx[enct_idx] = dects_idx
    # pdb.set_trace()

    return alignments_idx

def train_model(args, task_name, logger):
    global input_lang
    global output_lang

    input_lang, output_lang, pairs_train, pairs_dev, _, pairs_gen = prepare_dataset('nl', 'sparql', task_name)

    train_data, dev_data, gen_data = pairs_train, pairs_dev, pairs_gen

    train_data.sort(key=lambda p: len(p[0].split()))

    dev_data = list(set([tuple(item) for item in dev_data]))
    dev_data.sort(key=lambda p: len(p[0].split()))
    dev_data = [list(item) for item in dev_data]

    args.vocab_size = input_lang.n_words
    args.label_size = output_lang.n_words

    # read pre-alignment file
    alignment_filename = f'./{task_name}_data/preprocess/enct2dect'
    alignments_idx = get_alignment(alignment_filename, input_lang, output_lang)

    entity_list = []
    entity_filename = f'./{task_name}_data/preprocess/entity'
    with open(entity_filename, 'r') as f:
        decode_tokens = f.readlines()
        for decode_token in decode_tokens:
            entity_list.append(decode_token.strip("\n"))

    caus_predicate_list = []
    caus_predicate_filename = f'./{task_name}_data/preprocess/caus_predicate'
    with open(caus_predicate_filename, 'r') as f:
        decode_tokens = f.readlines()
        for decode_token in decode_tokens:
            caus_predicate_list.append(decode_token.strip("\n"))

    unac_predicate_list = []
    unac_predicate_filename = f'./{task_name}_data/preprocess/unac_predicate'
    with open(unac_predicate_filename, 'r') as f:
        decode_tokens = f.readlines()
        for decode_token in decode_tokens:
            unac_predicate_list.append(decode_token.strip("\n"))

    model = HRLModel(vocab_size=args.vocab_size,
                     word_dim=args.word_dim,
                     hidden_dim=args.hidden_dim,
                     label_dim=args.label_size,
                     composer_trans_hidden=args.composer_trans_hidden,
                     var_normalization=args.var_normalization,
                     input_lang=input_lang,
                     output_lang=output_lang,
                     alignments_idx=alignments_idx,
                     entity_list=entity_list,
                     caus_predicate_list=caus_predicate_list,
                     unac_predicate_list=unac_predicate_list)

    if USE_CUDA:
        model = model.cuda(args.gpu_id)

    optimizer = prepare_optimisers(args, logger,
                                   high_parameters=model.get_high_parameters(),
                                   low_parameters=model.get_low_parameters())

    data_len = 'all'   # 7, 21 is the max length
    epoch_count = 0
    cir_epoch_dict = {
        data_len: args.max_epoch
    }

    regular_weight = args.regular_weight
    logger.info(f"Start lesson {data_len} with batch size {args.batch_size}, low_lr {args.low_lr}, high_lr {args.high_lr}, optimisers {args.high_optimizer} and {args.low_optimizer}, l2_weight {args.l2_weight}, regular_weight {regular_weight}, hidden size {args.hidden_dim}")
    total_batch_num = 0
    for epoch in range(args.max_epoch):
        logger.info(f"Start epoch {epoch + 1}")
        if data_len in cir_epoch_dict:
            # training epochs
            cir_epoch_num = cir_epoch_dict[data_len]
        else:
            cir_epoch_num = 1

        if data_len == 'all':
            val_accuracy, total_batch_num = train(train_data,
                                                  dev_data, model, optimizer,
                                                  epoch, args, logger,
                                                  total_batch_num, data_len, regular_weight)
        else:
            train_lesson_idx = get_bound_idx(train_data, data_len)  # get the max index under the max_length
            dev_lesson_idx = get_bound_idx(dev_data, data_len)      # get the max index under the max_length

            val_accuracy, total_batch_num = train(train_data[:train_lesson_idx],
                                                  dev_data[:dev_lesson_idx], model, optimizer,
                                                  epoch, args, logger,
                                                  total_batch_num, data_len, regular_weight)
        # end of epoch
        logger.info(f"End epoch {epoch + 1}")
        logger.info(f"Epoch {epoch + 1} Dev accuracy: {val_accuracy:.4f}")

        final_dev_acc = validate(train_data+dev_data, model, epoch, args, logger)
        logger.info(f"Epoch {epoch + 1} Train+Dev accuracy: {final_dev_acc:.4f}")

        final_gen_acc = validate(gen_data, model, epoch, args, logger)
        logger.info(f"Epoch {epoch + 1} Gen accuracy: {final_gen_acc:.4f}")

        logger.info("Saving model...")
        best_model_path = f"{args.model_dir}/epoch-{epoch+1}.mdl"
        torch.save({"epoch": epoch, "batch_idx": "final", "state_dict": model.state_dict()}, best_model_path)

        if val_accuracy == 1.:
            logger.info(r"Training reached 100% accuracy, stopping training.")
            break


def test_model(args, task_name, logger):
    global input_lang
    global output_lang

    input_lang, output_lang, pairs_train, pairs_dev, pairs_test, pairs_gen = prepare_dataset('nl', 'sparql', task_name)

    test_data = pairs_gen
    test_data.sort(key=lambda p: len(p[0].split()))

    args.vocab_size = input_lang.n_words
    args.label_size = output_lang.n_words

    # read pre-alignment file
    alignment_filename = f'./{task_name}_data/preprocess/enct2dect'
    alignments_idx = get_alignment(alignment_filename, input_lang, output_lang)

    entity_list = []
    entity_filename = f'./{task_name}_data/preprocess/entity'
    with open(entity_filename, 'r') as f:
        decode_tokens = f.readlines()
        for decode_token in decode_tokens:
            entity_list.append(decode_token.strip("\n"))

    caus_predicate_list = []
    caus_predicate_filename = f'./{task_name}_data/preprocess/caus_predicate'
    with open(caus_predicate_filename, 'r') as f:
        decode_tokens = f.readlines()
        for decode_token in decode_tokens:
            caus_predicate_list.append(decode_token.strip("\n"))

    unac_predicate_list = []
    unac_predicate_filename = f'./{task_name}_data/preprocess/unac_predicate'
    with open(unac_predicate_filename, 'r') as f:
        decode_tokens = f.readlines()
        for decode_token in decode_tokens:
            unac_predicate_list.append(decode_token.strip("\n"))

    example2type_file = f'./{task_name}_data/preprocess/example2type'
    with open(example2type_file, 'r') as f:
        example2type = json.load(f)

    model = HRLModel(vocab_size=args.vocab_size,
                     word_dim=args.word_dim,
                     hidden_dim=args.hidden_dim,
                     label_dim=args.label_size,
                     composer_trans_hidden=args.composer_trans_hidden,
                     var_normalization=args.var_normalization,
                     input_lang=input_lang,
                     output_lang=output_lang,
                     alignments_idx=alignments_idx,
                     entity_list=entity_list,
                     caus_predicate_list=caus_predicate_list,
                     unac_predicate_list=unac_predicate_list)

    if USE_CUDA:
        model = model.cuda(args.gpu_id)

    checkpoint_file = args.checkpoint
    logger.info(f"loading {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])
    logger.info("loading finished...")
    max_length = 'all'
    if max_length == 'all':
        test_data = test_data
    else:
        test_lesson_idx = get_bound_idx(test_data, max_length)  # get the max index under the max_length
        # test_lesson_idx = -1
        test_data = test_data[:test_lesson_idx]
    random.shuffle(test_data)
    logger.info("Start testing ..")
    print("Start testing ..")
    log_file = './log/' + re.split(r'/|\.', checkpoint_file)[-3] + "_" + re.split(r'/|\.', checkpoint_file)[-2] + '.txt'
    # pdb.set_trace()
    test_acc, type_right_count = test(test_data, model, example2type, args, log_file)
    logger.info("Test Acc: {} %".format(test_acc * 100))
    for type in type_right_count:
        type_acc = sum(type_right_count[type]) / len(type_right_count[type])
        logger.info(f"{type} acc: {type_acc}")


def prepare_arguments(checkpoint_folder: str, parser: argparse.ArgumentParser):
    # ---------- hard-coded defaults ----------
    hidden_size = 128

    default_args = dict(
        sample_num         = 15,  # number of Monte-Carlo clones
        batch_size         = 8,
        high_lr            = 0.5,
        low_lr             = 0.05,
        clip_grad_norm     = 0.5,
        l2_weight          = 1e-5,
        high_optimizer     = "adadelta",
        low_optimizer      = "adadelta",
        
        word_dim           = hidden_size,
        hidden_dim         = hidden_size,
        composer_leaf      = "no_transformation",
        composer_trans_hidden = hidden_size,
        var_normalization  = True,
        regular_weight     = 1e-1,
        max_epoch          = 20,
        gpu_id             = 0,
        model_dir          = f"checkpoint/models/{checkpoint_folder}",
        train_logs_path    = f"checkpoint/logs/train/{checkpoint_folder}",
        test_logs_path     = "checkpoint/logs/test/",
        regular_decay_rate = 0.5,
        x_ratio_rate       = 0.5,
        log_every          = 500,
        val_every          = 1000,
    )

    # ---------- register only the options you really need to tune ----------
    for name, default in default_args.items():
        arg_type = type(default) if not isinstance(default, bool) else lambda x: x.lower()=='true'
        parser.add_argument(f"--{name.replace('_','-')}", default=default, type=arg_type)

    return parser.parse_args()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--mode", required=True, default='train',
                            choices=['train', 'test'], type=str,
                            help="Determine whether to train a model or test using a trained weight file")
    arg_parser.add_argument("--checkpoint", required=True, type=str,
                            help="When training, it is the folder to store model weights; "
                                 "Otherwise it is the weight path to be loaded.")
    arg_parser.add_argument("--task", required=True, type=str,
                            choices=["addjump", "around_right", "simple", "length",
                                     "extend", "mcd1", "mcd2", "mcd3", "cfq", "cogs", "slog"],
                            help="All tasks on SCAN, the task name is used to load train or test file")
    arg_parser.add_argument("--random-seed", required=False, default=2, type=int)

    parsed_args = arg_parser.parse_args()
    if parsed_args.mode == 'train':
        args = prepare_arguments(parsed_args.checkpoint, arg_parser)
        logger = make_path_preparations(args, parsed_args.mode)
        train_model(args, parsed_args.task, logger)
    else:
        args = prepare_arguments(parsed_args.checkpoint, arg_parser)
        logger = make_path_preparations(args, parsed_args.mode)
        test_model(args, parsed_args.task, logger)

