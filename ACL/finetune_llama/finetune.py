import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from datasets import load_dataset, DatasetDict
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import argparse


# ========= USER CONFIG =========
# Choose ONE model id (examples):
# - Original LLaMA 7B (converted HF weights; you must have access):
#   model_id = "decapoda-research/llama-7b-hf"
# - Llama 2 (requires HF auth): "meta-llama/Llama-2-7b-hf"
# - Llama 3 (requires HF auth): "meta-llama/Meta-Llama-3-8B-Instruct"
# - Mistral 7B: "mistralai/Mistral-7B-Instruct-v0.3"

# ========= PROMPT & MASKING =========
# We’ll train only on the "### Output:" part using response_template masking
PROMPT_TEMPLATE = """### Instruction
You are given an INPUT sentence and its CATEGORY. Produce the correct OUTPUT.

CATEGORY: {category}
INPUT: {inp}

### Output
"""

RESPONSE_PREFIX = "### Output"

# ========= LOAD DATA =========
def load_tsv(train_path: str, valid_path: Optional[str]) -> DatasetDict:
    if valid_path and os.path.exists(valid_path):
        dd = load_dataset(
            "csv",
            data_files={"train": train_path, "validation": valid_path},
            delimiter="\t",
            column_names=["input", "output", "category"],
            quoting=3
        )
    else:
        ds = load_dataset(
            "csv",
            data_files={"train": train_path},
            delimiter="\t",
            column_names=["input", "output", "category"],
            quoting=3
        )["train"]
        dd = DatasetDict({
            "train": ds.shuffle(seed=42).select(range(int(0.95 * len(ds)))),
            "validation": ds.shuffle(seed=42).select(range(int(0.95 * len(ds)), len(ds)))
        })
    return dd

# ========= MODEL / TOKENIZER =========
def get_model_tokenizer(model_id: str):
    # select compute dtype sensibly
    if torch.cuda.is_available():
        # prefer bfloat16 on GPUs that support it (A100/H100)
        compute_dtype = torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda : False)() else torch.float16
        model_dtype = torch.bfloat16 if compute_dtype is torch.bfloat16 else torch.float16
    else:
        compute_dtype = torch.float16
        model_dtype = torch.float32  # CPU -> keep full precision (will not use bnb 4-bit anyway)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=model_dtype
        )
    except Exception as e:
        raise RuntimeError(
            f"Error loading model '{model_id}': {e}\n"
            "If this is a gated HF model (e.g. Meta LLaMA family), run `huggingface-cli login` "
            "and ensure your account has access to the model weights."
        )

    return model, tok


# ========= LoRA =========
def get_lora_cfg(model_id: str) -> LoraConfig:
    # Works well for LLaMA/Mistral/Qwen families
    target_modules = ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"]
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )

def main(args):
    data = load_tsv(args.TRAIN_TSV, args.VALID_TSV if os.path.exists(args.VALID_TSV) else None)
    model, tok = get_model_tokenizer(args.MODEL_ID)
    lora_cfg = get_lora_cfg(args.MODEL_ID)

    # Map rows → single training strings
    def formatting_func(examples):
        # examples is a dict whose values are lists (for batched) OR strings (for map with batched=False)
        # we normalize to lists
        inputs = examples["input"] if isinstance(examples["input"], list) else [examples["input"]]
        outputs = examples["output"] if isinstance(examples["output"], list) else [examples["output"]]
        categories = examples["category"] if isinstance(examples["category"], list) else [examples["category"]]

        formatted = []
        for inp, out, cat in zip(inputs, outputs, categories):
            formatted.append(PROMPT_TEMPLATE.format(category=cat, inp=inp) + out)
        return formatted

    # Trainer config
    training_cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        learning_rate=2e-4,        # a good QLoRA starting LR
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=False,
        fp16=True,
        max_seq_length=max_seq_len,
        packing=False,             # set True if you want packing for speed on long corpora
        gradient_checkpointing=True,
        dataset_num_proc=4,
        optim="paged_adamw_8bit",
        report_to="none",
        dataset_text_field=None,   # we'll use formatting_func instead
        response_template=RESPONSE_PREFIX,  # masks loss before this tag
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        peft_config=lora_cfg,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        formatting_func=formatting_func,
        args=training_cfg,
    )

    print("Model dtype:", next(model.parameters()).dtype)
    print("Device map:", model.hf_device_map if hasattr(model, "hf_device_map") else "none")
    print("Tokenizer pad token id:", tok.pad_token_id)
    
    trainer.train()
    trainer.save_model()  # saves LoRA adapter
    tok.save_pretrained(output_dir)

    print("\nTraining complete. Adapter saved to:", output_dir)
    print("To run inference with the adapter, see the snippet below.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--MODEL_ID", type=str, default=os.environ.get("MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--TRAIN_TSV", type=str, default=os.environ.get("TRAIN_TSV", "data/train.tsv"))
    parser.add_argument("--VALID_TSV", type=str, default=os.environ.get("VALID_TSV", "data/valid.tsv"))
    parser.add_argument("--OUT_DIR", type=str, default=os.environ.get("OUT_DIR", "outputs/default-qlora"))
    parser.add_argument("--MAX_SEQ_LEN", type=int, default=int(os.environ.get("MAX_SEQ_LEN", "512")))
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
    