#!/usr/bin/env python3
"""
train_transformer_seq2seq.py

Usage:
    python train_transformer_seq2seq.py --train train.tsv --test test.tsv --gen gen.tsv

TSV format:
    input\toutput\tcategory

The script trains a vanilla Transformer seq2seq model (encoder-decoder),
validates on test, then evaluates on gen and prints overall + per-category exact-match accuracies.
"""

import argparse
import math
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# -----------------------
# Config / hyperparams
# -----------------------
DEFAULTS = {
    "batch_size": 64,
    "emb_size": 256,
    "nhead": 8,
    "ffn_hidden": 1024,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "dropout": 0.1,
    "lr": 1e-4,
    "epochs": 10,
    "max_len": 128,
    "seed": 42,
    "patience": 5,  # not used heavily, simple placeholder
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "print_every": 100,
}

# -----------------------
# Tokenizer / vocab
# -----------------------
# We'll build a simple tokenizer:
# - split on whitespace to get tokens
# - further split tokens into characters if unseen/rare (fallback)
# This is simple but robust for many small seq2seq tasks.
# Special tokens: <pad>, <bos>, <eos>, <unk>
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def build_vocab_from_texts(texts, min_freq=1, max_size=None):
    """
    Build token vocabulary from a list of strings.
    Tokenization: whitespace tokens. We include individual characters
    for tokens of length 1 or if token frequency low, but for simplicity
    we just include all whitespace tokens.
    """
    counter = Counter()
    for s in texts:
        if not isinstance(s, str):
            continue
        toks = s.strip().split()
        if not toks:
            # preserve empty as explicit token (use empty string)
            counter[""] += 1
        else:
            for t in toks:
                counter[t] += 1

    # filter by min_freq
    items = [(tok, cnt) for tok, cnt in counter.items() if cnt >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    if max_size:
        items = items[:max_size - len(SPECIAL_TOKENS)]
    vocab = SPECIAL_TOKENS + [tok for tok, _ in items]
    idx = {tok: i for i, tok in enumerate(vocab)}
    return vocab, idx


def tokenize_text(s):
    # simple whitespace tokenizer returning list of tokens
    if s is None:
        return []
    return s.strip().split()


def encode_tokens(tokens, idx, max_len, bos=True, eos=True):
    """
    tokens: list of token strings (whitespace tokens)
    returns: list of token ids (with bos/eos and truncated/padded externally)
    """
    ids = []
    if bos:
        ids.append(idx["<bos>"])
    for t in tokens:
        ids.append(idx.get(t, idx["<unk>"]))
        if len(ids) >= max_len - 1:  # keep room for eos
            break
    if eos:
        ids.append(idx["<eos>"])
    return ids


# -----------------------
# Dataset
# -----------------------
class Seq2SeqTsvDataset(Dataset):
    def __init__(self, df, idx, max_len):
        """
        df: pandas DataFrame with columns input, output, category (category optional)
        idx: token->id mapping
        """
        self.srcs = df["input"].astype(str).tolist()
        self.tgts = df["output"].astype(str).tolist()
        self.cats = df["category"].astype(str).tolist() if "category" in df.columns else ["" for _ in self.srcs]
        self.idx = idx
        self.max_len = max_len

    def __len__(self):
        return len(self.srcs)

    def __getitem__(self, i):
        src = self.srcs[i]
        tgt = self.tgts[i]
        cat = self.cats[i]
        src_toks = tokenize_text(src)
        tgt_toks = tokenize_text(tgt)
        src_ids = encode_tokens(src_toks, self.idx, self.max_len, bos=True, eos=True)
        tgt_ids = encode_tokens(tgt_toks, self.idx, self.max_len, bos=True, eos=True)
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "category": cat,
            "raw_src": src,
            "raw_tgt": tgt,
        }


def collate_batch(batch, pad_idx):
    # pad sequences
    srcs = [b["src_ids"] for b in batch]
    tgts = [b["tgt_ids"] for b in batch]
    cats = [b["category"] for b in batch]
    raw_src = [b["raw_src"] for b in batch]
    raw_tgt = [b["raw_tgt"] for b in batch]

    src_lens = [len(x) for x in srcs]
    tgt_lens = [len(x) for x in tgts]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    src_padded = torch.full((len(srcs), max_src), pad_idx, dtype=torch.long)
    tgt_padded = torch.full((len(tgts), max_tgt), pad_idx, dtype=torch.long)

    for i, s in enumerate(srcs):
        src_padded[i, : len(s)] = s
    for i, t in enumerate(tgts):
        tgt_padded[i, : len(t)] = t

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "src_lens": torch.tensor(src_lens, dtype=torch.long),
        "tgt_lens": torch.tensor(tgt_lens, dtype=torch.long),
        "category": cats,
        "raw_src": raw_src,
        "raw_tgt": raw_tgt,
    }


# -----------------------
# Positional encoding
# -----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(maxlen, emb_size)
        position = torch.arange(0, maxlen, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        if emb_size % 2 == 1:
            # odd embedding size: last column zero for cos
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, maxlen, emb_size)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, emb_size)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# -----------------------
# Model
# -----------------------
class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_size=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3, ffn_hidden=1024, dropout=0.1, max_len=512, pad_idx=0):
        super().__init__()
        self.emb_size = emb_size
        self.src_tok_emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.tgt_tok_emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, maxlen=max_len)
        transformer = nn.Transformer(d_model=emb_size, nhead=nhead,
                                     num_encoder_layers=num_encoder_layers,
                                     num_decoder_layers=num_decoder_layers,
                                     dim_feedforward=ffn_hidden,
                                     dropout=dropout,
                                     batch_first=True)  # batch_first for easier handling
        self.transformer = transformer
        self.generator = nn.Linear(emb_size, vocab_size)
        self.pad_idx = pad_idx

        # initialize
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_src_mask(self, src):
        # src: (batch, src_len)
        src_mask = (src == self.pad_idx)  # True where pad
        return src_mask  # used as key_padding_mask for encoder

    def create_tgt_mask(self, tgt):
        # tgt: (batch, tgt_len)
        seq_len = tgt.size(1)
        # subsequent mask (causal)
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        return mask.to(tgt.device)  # (tgt_len, tgt_len)

    def forward(self, src, tgt_input, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # src: (batch, src_len)
        # tgt_input: (batch, tgt_len)  -- tokens shifted right (starts with BOS)
        src_emb = self.src_tok_emb(src) * math.sqrt(self.emb_size)
        tgt_emb = self.tgt_tok_emb(tgt_input) * math.sqrt(self.emb_size)
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.positional_encoding(tgt_emb)
        tgt_mask = self.create_tgt_mask(tgt_input)
        # transformer expects src_key_padding_mask shape: (batch, src_len) True where pad
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=src_key_padding_mask)
        logits = self.generator(out)  # (batch, tgt_len, vocab_size)
        return logits


# -----------------------
# Utility functions
# -----------------------
def make_shifted_targets(tgt_batch, pad_idx):
    """
    For training, we take the target sequence and create:
      - decoder input = target tokens excluding last token (i.e. shifted right already includes BOS)
      - gold labels = target tokens excluding first token
    Both are padded to same length.
    """
    # tgt_batch: (batch, tgt_len)
    decoder_input = tgt_batch[:, :-1]
    gold = tgt_batch[:, 1:]
    return decoder_input, gold


def greedy_decode(model, src, src_mask, idx_to_tok, tok_to_idx, max_len, device):
    """
    Greedy decode one batch of src:
    src: (batch, src_len) tensor
    returns list of predicted token strings
    """
    model.eval()
    batch_size = src.size(0)
    pad_idx = tok_to_idx["<pad>"]
    bos_idx = tok_to_idx["<bos>"]
    eos_idx = tok_to_idx["<eos>"]

    src_key_padding_mask = (src == pad_idx).to(device)
    # encoder memory produced implicitly by transformer forward through full model: we need to call encoder
    # But nn.Transformer's API is wrapped in forward with both; to decode incrementally we can repeatedly call model.transformer
    # We'll use the following approach:
    with torch.no_grad():
        src_emb = model.src_tok_emb(src) * math.sqrt(model.emb_size)
        src_emb = model.positional_encoding(src_emb)
        memory = model.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        # start tokens
        ys = torch.ones((batch_size, 1), dtype=torch.long, device=device) * bos_idx

        finished = [False] * batch_size
        outputs = [[] for _ in range(batch_size)]

        for i in range(max_len - 1):
            tgt_emb = model.tgt_tok_emb(ys) * math.sqrt(model.emb_size)
            tgt_emb = model.positional_encoding(tgt_emb)
            tgt_mask = torch.triu(torch.full((ys.size(1), ys.size(1)), float('-inf')), diagonal=1).to(device)
            out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                                            memory_key_padding_mask=src_key_padding_mask)
            logits = model.generator(out)  # (batch, seq_len, vocab)
            next_token_logits = logits[:, -1, :]  # (batch, vocab)
            next_tokens = next_token_logits.argmax(dim=-1)  # greedy
            ys = torch.cat([ys, next_tokens.unsqueeze(1)], dim=1)

            for b in range(batch_size):
                tok = next_tokens[b].item()
                if (not finished[b]) and tok == eos_idx:
                    finished[b] = True
                if not finished[b]:
                    outputs[b].append(tok)
            if all(finished):
                break

    # convert token ids to raw strings (space-joined tokens)
    pred_texts = []
    for b in range(batch_size):
        toks = outputs[b]
        # convert ids to tokens ignoring special tokens in text (but we will map <unk> to "<unk>")
        tokens = []
        for tid in toks:
            tok = idx_to_tok[tid]
            if tok in SPECIAL_TOKENS:
                # skip pad/bos/eos, but keep unknown marker
                if tok == "<unk>":
                    tokens.append("<unk>")
                else:
                    continue
            else:
                tokens.append(tok)
        # join by single space
        pred_texts.append(" ".join(tokens).strip())
    return pred_texts


# -----------------------
# Training / evaluation
# -----------------------
def train_epoch(model, dataloader, optimizer, criterion, pad_idx, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="train", leave=False)
    for i, batch in pbar:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        src_key_padding_mask = (src == pad_idx).to(device)
        tgt_key_padding_mask = (tgt[:, :-1] == pad_idx).to(device)  # decoder input pad mask

        decoder_input, gold = make_shifted_targets(tgt, pad_idx)
        optimizer.zero_grad()
        logits = model(src, decoder_input, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        # logits: (batch, tgt_len-1, vocab)
        logits_flat = logits.view(-1, logits.size(-1))
        gold_flat = gold.contiguous().view(-1)
        loss = criterion(logits_flat, gold_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % 10 == 0:
            pbar.set_postfix({"loss": total_loss / (i + 1)})
    return total_loss / len(dataloader)


def evaluate_exact_match(model, dataloader, tok_to_idx, idx_to_tok, pad_idx, device, max_len=128):
    model.eval()
    all_preds = []
    all_trues = []
    all_cats = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="eval", leave=False):
            src = batch["src"].to(device)
            raw_tgt = batch["raw_tgt"]
            cats = batch["category"]
            preds = greedy_decode(model, src, None, idx_to_tok, tok_to_idx, max_len=max_len, device=device)
            all_preds.extend(preds)
            all_trues.extend(raw_tgt)
            all_cats.extend(cats)
    # exact match counting
    total = len(all_preds)
    correct = sum(1 for p, t in zip(all_preds, all_trues) if p.strip() == t.strip())
    overall_acc = correct / total if total > 0 else 0.0

    # per-category
    per_cat_counts = defaultdict(int)
    per_cat_correct = defaultdict(int)
    for p, t, c in zip(all_preds, all_trues, all_cats):
        per_cat_counts[c] += 1
        if p.strip() == t.strip():
            per_cat_correct[c] += 1
    per_cat_acc = {c: per_cat_correct[c] / per_cat_counts[c] for c in per_cat_counts}
    return overall_acc, per_cat_acc, list(zip(all_preds, all_trues, all_cats))


# -----------------------
# Main
# -----------------------
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    # load TSVs
    train_df = pd.read_csv(args.train, sep="\t", header=0, usecols=[0, 1, 2], names=["input", "output", "category"], dtype=str, keep_default_na=False)
    test_df = pd.read_csv(args.test, sep="\t", header=0, usecols=[0, 1, 2], names=["input", "output", "category"], dtype=str, keep_default_na=False)
    gen_df = pd.read_csv(args.gen, sep="\t", header=0, usecols=[0, 1, 2], names=["input", "output", "category"], dtype=str, keep_default_na=False)

    # build vocab across all text (train+test+gen) to avoid OOV during evaulation
    all_texts = pd.concat([train_df["input"], train_df["output"], test_df["input"], test_df["output"], gen_df["input"], gen_df["output"]]).astype(str).tolist()
    vocab, tok_to_idx = build_vocab_from_texts(all_texts)
    idx_to_tok = {i: t for t, i in tok_to_idx.items()}

    pad_idx = tok_to_idx["<pad>"]
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}, device: {device}, training samples: {len(train_df)}")

    # datasets + dataloaders
    train_ds = Seq2SeqTsvDataset(train_df, tok_to_idx, max_len=args.max_len)
    test_ds = Seq2SeqTsvDataset(test_df, tok_to_idx, max_len=args.max_len)
    gen_ds = Seq2SeqTsvDataset(gen_df, tok_to_idx, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_batch(b, pad_idx))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=lambda b: collate_batch(b, pad_idx))
    gen_loader = DataLoader(gen_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate_batch(b, pad_idx))

    # model
    model = TransformerSeq2Seq(vocab_size=vocab_size, emb_size=args.emb_size, nhead=args.nhead,
                               num_encoder_layers=args.num_encoder_layers, num_decoder_layers=args.num_decoder_layers,
                               ffn_hidden=args.ffn_hidden, dropout=args.dropout, max_len=args.max_len, pad_idx=pad_idx)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    if not args.eval_mode:
        # training loop
        best_test_acc = -1.0
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            train_loss = train_epoch(model, train_loader, optimizer, criterion, pad_idx, device)
            print(f"Train loss: {train_loss:.4f}")

            # quick evaluate on test set (exact match)
            test_acc, _, _ = evaluate_exact_match(model, test_loader, tok_to_idx, idx_to_tok, pad_idx, device, max_len=args.max_len)
            print(f"Test exact-match acc: {test_acc:.4f}")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                # save checkpoint
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "tok_to_idx": tok_to_idx,
                    "idx_to_tok": idx_to_tok,
                }, args.checkpoint)
                print(f"Saved best model to {args.checkpoint}")

    # load best
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"\nLoaded best model from {args.checkpoint} (test acc {best_test_acc:.4f})")

    # Evaluate on gen dataset
    overall_acc, per_cat_acc, detailed = evaluate_exact_match(model, gen_loader, tok_to_idx, idx_to_tok, pad_idx, device, max_len=args.max_len)
    print(f"\nGEN overall exact-match accuracy: {overall_acc:.4f}\n")
    print("Per-category accuracy:")
    for c, acc in sorted(per_cat_acc.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {c}: {acc:.4f} (n={sum(1 for _c in gen_df['category'] if _c == c)})")

    # Optionally print some examples
    print("\nSome predictions (predicted ||| target ||| category):")
    for pred, tgt, cat in random.sample(detailed, min(10, len(detailed))):
        print(f"  {pred or '[EMPTY]'}  |||  {tgt or '[EMPTY]'}  |||  {cat}")

    # done
    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="train.tsv path")
    parser.add_argument("--test", type=str, required=True, help="test.tsv path")
    parser.add_argument("--gen", type=str, required=True, help="gen.tsv path to evaluate per-category")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--emb_size", type=int, default=DEFAULTS["emb_size"])
    parser.add_argument("--nhead", type=int, default=DEFAULTS["nhead"])
    parser.add_argument("--ffn_hidden", type=int, default=DEFAULTS["ffn_hidden"])
    parser.add_argument("--num_encoder_layers", type=int, default=DEFAULTS["num_encoder_layers"])
    parser.add_argument("--num_decoder_layers", type=int, default=DEFAULTS["num_decoder_layers"])
    parser.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--max_len", type=int, default=DEFAULTS["max_len"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--checkpoint", type=str, default="best_transformer.pt")
    parser.add_argument("--device", type=str, default=DEFAULTS["device"])
    parser.add_argument("--eval_mode", action="store_true", help="if set, only run evaluation using --checkpoint")
    args = parser.parse_args()
    main(args)
