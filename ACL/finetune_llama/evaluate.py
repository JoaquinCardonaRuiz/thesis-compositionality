import argparse
import json
import os
import re
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# ---------- Prompt (must match training) ----------
PROMPT_TEMPLATE = """### Instruction
You are given an INPUT sentence. Produce the correct OUTPUT.

INPUT: {inp}

### Output
"""

RESPONSE_PREFIX = "### Output"

def build_prompt(inp: str) -> str:
    return PROMPT_TEMPLATE.format(inp=inp)

# ---------- Normalization ----------
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
        quoting=3
    )["test"]
    return ds

# ---------- Batched generation ----------
@torch.inference_mode()
def generate_batch(model, tok, prompts, max_new_tokens=128):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    input_lens = enc["input_ids"].ne(tok.pad_token_id).sum(dim=1)

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    preds = []
    for i in range(out.size(0)):
        gen_tokens = out[i, input_lens[i]:]
        text = tok.decode(gen_tokens, skip_special_tokens=True)
        # Extract text after "### Output" if present
        if RESPONSE_PREFIX in text:
            text = text.split(RESPONSE_PREFIX, 1)[-1].strip()
        preds.append(text.strip())
    return preds

# ---------- Main evaluation ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--adapter_dir", default=None)
    ap.add_argument("--test_tsv", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--lowercase_norm", action="store_true")
    ap.add_argument("--metrics_out", default=None)
    ap.add_argument("--preds_out", default=None)
    args = ap.parse_args()

    # Load dataset
    ds = load_tsv(args.test_tsv)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Model
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
        device_map={"": 0},
        load_in_8bit=False
    )
    model.gradient_checkpointing_disable()
    model.config.use_cache = True

    if args.adapter_dir:
        if not PEFT_AVAILABLE:
            raise RuntimeError("PEFT not installed but adapter_dir provided")
        model = PeftModel.from_pretrained(model, args.adapter_dir)

    model = model.to("cuda")
    model.eval()

    # Quick test prediction
    test_prompt = build_prompt("Avery froze a girl on a bed beside a table in the room.")
    print("Example generation:")
    print(generate_batch(model, tok, [test_prompt], max_new_tokens=64)[0])
    print("\nBegin evaluation...\n")

    # Metrics tracking
    n = len(ds)
    bs = max(1, args.batch_size)
    correct = 0
    by_cat_correct = Counter()
    by_cat_total = Counter()
    preds_records = []

    for start in range(0, n, bs):
        end = min(start + bs, n)
        batch = ds.select(range(start, end))
        if (start // bs) % 5 == 0:
            print(f"Processed {start}/{n} examples")

        prompts = [build_prompt(inp) for inp in batch["input"]]
        batch_preds = generate_batch(model, tok, prompts, max_new_tokens=args.max_new_tokens)

        for inp, gold, cat, pred in zip(batch["input"], batch["output"], batch["category"], batch_preds):
            gold_n = normalize(gold, args.lowercase_norm)
            pred_n = normalize(pred, args.lowercase_norm)
            is_correct = int(gold_n == pred_n)
            correct += is_correct
            by_cat_total[cat] += 1
            by_cat_correct[cat] += is_correct
            if args.preds_out:
                preds_records.append((inp, gold, pred, cat, is_correct))

    # Metrics
    overall_acc = correct / n if n else 0.0
    per_cat = {cat: (by_cat_correct[cat] / by_cat_total[cat]) for cat in by_cat_total}
    macro_acc = sum(per_cat.values()) / len(per_cat) if per_cat else 0.0

    # Print summary
    print(f"Examples: {n}")
    print(f"Overall accuracy (micro): {overall_acc*100:.2f}% ({correct}/{n})")
    print(f"Macro accuracy (mean over categories): {macro_acc*100:.2f}%")
    print("Per-category accuracy:")
    for cat in sorted(per_cat.keys()):
        c = by_cat_correct[cat]
        t = by_cat_total[cat]
        print(f"  {cat}: {100*c/t:.2f}% ({c}/{t})")

    # Save metrics
    if args.metrics_out:
        metrics = {
            "num_examples": n,
            "overall_accuracy": overall_acc,
            "macro_accuracy": macro_acc,
            "per_category": {k: float(v) for k, v in per_cat.items()},
        }
        os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
        with open(args.metrics_out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {args.metrics_out}")

    # Save predictions
    if args.preds_out and preds_records:
        os.makedirs(os.path.dirname(args.preds_out) or ".", exist_ok=True)
        with open(args.preds_out, "w", encoding="utf-8") as f:
            f.write("input\tgold\tpred\tcategory\tcorrect\n")
            for rec in preds_records:
                safe = [str(x).replace("\t", " ").replace("\n", " ") for x in rec]
                f.write("\t".join(safe) + "\n")
        print(f"Saved predictions to: {args.preds_out}")

if __name__ == "__main__":
    main()
