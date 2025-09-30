import argparse
import json
import os
import re
from collections import defaultdict, Counter
from transformers import BitsAndBytesConfig

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


# ---------- Prompt (must match training) ----------
PROMPT_TEMPLATE = """### Instruction
You are given an INPUT sentence and its CATEGORY. Produce the correct OUTPUT.

CATEGORY: {category}
INPUT: {inp}

### Output
"""

RESPONSE_PREFIX = "### Output"


def build_prompt(inp: str, cat: str) -> str:
    return PROMPT_TEMPLATE.format(category=cat, inp=inp)


# ---------- Normalization for exact match ----------
SPACE_RE = re.compile(r"\s+")

def normalize(s: str, lowercase: bool = False) -> str:
    s = s.strip()
    s = SPACE_RE.sub(" ", s)
    if lowercase:
        s = s.lower()
    return s


# ---------- Data loader ----------
def load_tsv(path: str):
    ds = load_dataset(
        "csv",
        data_files={"test": path},
        delimiter="\t",
        column_names=["input", "output", "category"],
        quoting=3  # csv.QUOTE_NONE
    )["test"]
    return ds


# ---------- Inference ----------
@torch.inference_mode()
def generate_batch(model, tok, prompts, max_new_tokens=128):
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,  # safeguard; does not affect new tokens
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    input_lens = enc["input_ids"].ne(tok.pad_token_id).sum(dim=1)

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,            # set >1 if you want beam search, e.g. 3 or 5
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    preds = []
    for i in range(out.size(0)):
        gen_tokens = out[i, input_lens[i]:]
        text = tok.decode(gen_tokens, skip_special_tokens=True)
        # Stop at a safety cutoff if the model keeps talking after the answer
        cutoff = text.split("\n###", 1)[0]
        preds.append(cutoff.strip())
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True,
                    help="HF model id (e.g. meta-llama/Meta-Llama-3-8B-Instruct)")
    ap.add_argument("--adapter_dir", default=None,
                    help="Optional LoRA adapter directory (from training). If omitted, evaluates the base model.")
    ap.add_argument("--test_tsv", required=True, help="Path to TSV with 3 columns: input, output, category")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--lowercase_norm", action="store_true",
                    help="If set, compare predictions and gold in lowercase.")
    ap.add_argument("--metrics_out", default=None, help="Optional JSON path to save metrics.")
    ap.add_argument("--preds_out", default=None,
                    help="Optional TSV path to save (input, gold, pred, category).")
    args = ap.parse_args()

    # Load data
    ds = load_tsv(args.test_tsv)

    # Model + tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Use bfloat16 on GPU if available; CPU falls back to float32
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        quantization_config=bnb,
        device_map="auto",
    )
    if args.adapter_dir:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft is not installed but adapter_dir was provided.")
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    # Iterate in batches
    n = len(ds)
    bs = max(1, args.batch_size)
    correct = 0
    by_cat_correct = Counter()
    by_cat_total = Counter()
    preds_records = []

    for start in range(0, n, bs):
        end = min(start + bs, n)
        batch = ds.select(range(start, end))

        if (start // bs) % 10 == 0:
            print(f"Processed {start}/{n} examples")


        prompts = [build_prompt(inp, cat) for inp, cat in zip(batch["input"], batch["category"])]
        batch_preds = generate_batch(model, tok, prompts, max_new_tokens=args.max_new_tokens)

        # Evaluate this batch
        for inp, gold, cat, pred in zip(batch["input"], batch["output"], batch["category"], batch_preds):
            gold_n = normalize(gold, args.lowercase_norm)
            pred_n = normalize(pred, args.lowercase_norm)
            is_correct = (gold_n == pred_n)
            correct += int(is_correct)
            by_cat_total[cat] += 1
            by_cat_correct[cat] += int(is_correct)
            if args.preds_out:
                preds_records.append((inp, gold, pred, cat, int(is_correct)))

    # Metrics
    overall_acc = correct / n if n else 0.0
    per_cat = {cat: (by_cat_correct[cat] / by_cat_total[cat]) for cat in by_cat_total}
    macro_acc = sum(per_cat.values()) / len(per_cat) if per_cat else 0.0

    # Print summary (no per-example outputs)
    print(f"Examples: {n}")
    print(f"Overall accuracy (micro): {overall_acc*100:.2f}% ({correct}/{n})")
    print(f"Macro accuracy (mean over categories): {macro_acc*100:.2f}%")
    print("Per-category accuracy:")
    for cat in sorted(per_cat.keys()):
        c = by_cat_correct[cat]
        t = by_cat_total[cat]
        print(f"  {cat}: {100*c/t:.2f}% ({c}/{t})")

    # Optional artifacts
    if args.metrics_out:
        metrics = {
            "num_examples": n,
            "overall_accuracy": overall_acc,
            "macro_accuracy": macro_acc,
            "per_category": {k: float(v) for k, v in per_cat.items()},
        }
        with open(args.metrics_out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {args.metrics_out}")

    if args.preds_out and preds_records:
        os.makedirs(os.path.dirname(args.preds_out) or ".", exist_ok=True)
        with open(args.preds_out, "w", encoding="utf-8") as f:
            f.write("input\tgold\tpred\tcategory\tcorrect\n")
            for rec in preds_records:
                # Escape internal tabs/newlines just in case
                safe = [str(x).replace("\t", " ").replace("\n", " ") for x in rec]
                f.write("\t".join(safe) + "\n")
        print(f"Saved predictions to: {args.preds_out}")


if __name__ == "__main__":
    main()
