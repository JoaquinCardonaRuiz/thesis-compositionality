import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from datasets import load_dataset, DatasetDict
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# ========= USER CONFIG =========
# Choose ONE model id (examples):
# - Original LLaMA 7B (converted HF weights; you must have access):
#   model_id = "decapoda-research/llama-7b-hf"
# - Llama 2 (requires HF auth): "meta-llama/Llama-2-7b-hf"
# - Llama 3 (requires HF auth): "meta-llama/Meta-Llama-3-8B-Instruct"
# - Mistral 7B: "mistralai/Mistral-7B-Instruct-v0.3"
model_id = os.environ.get("MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")

train_path = os.environ.get("TRAIN_TSV", "data/train.tsv")
valid_path = os.environ.get("VALID_TSV", "data/valid.tsv")  # optional; if missing we split train
output_dir = os.environ.get("OUT_DIR", f"outputs/{model_id.split('/')[-1]}-qlora-tsv")

# Max sequence length for packing/truncation
max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "1024"))

# ========= PROMPT & MASKING =========
# We’ll train only on the "### Output:" part using response_template masking
PROMPT_TEMPLATE = """### Instruction
You are given an INPUT sentence and its CATEGORY. Produce the correct OUTPUT.

CATEGORY: {category}
INPUT: {inp}

### Output
"""

RESPONSE_PREFIX = "### Output"

def format_example(example: Dict[str, str]) -> str:
    inp, out, cat = example["input"], example["output"], example["category"]
    return PROMPT_TEMPLATE.format(category=cat, inp=inp) + out

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
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # Handle missing pad token safely for causal LMs
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
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

def main():
    data = load_tsv(train_path, valid_path if os.path.exists(valid_path) else None)
    model, tok = get_model_tokenizer(model_id)
    lora_cfg = get_lora_cfg(model_id)

    # Map rows → single training strings
    def formatting_func(examples):
        # batched dict of lists
        if isinstance(examples["input"], list):
            out = []
            for i in range(len(examples["input"])):
                ex = {
                    "input": examples["input"][i],
                    "output": examples["output"][i],
                    "category": examples["category"][i],
                }
                out.append(format_example(ex))
            return out
        # single example dict
        return format_example(examples)


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

    trainer.train()
    trainer.save_model()  # saves LoRA adapter
    tok.save_pretrained(output_dir)

    print("\nTraining complete. Adapter saved to:", output_dir)
    print("To run inference with the adapter, see the snippet below.")

if __name__ == "__main__":
    main()
