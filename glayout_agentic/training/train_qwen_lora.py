from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


SYSTEM_PROMPT = (
    "You are a coding agent specialized in gLayout. "
    "You write Python files that keep sizing parameters callable at runtime, "
    "while placement and routing choices stay fixed in the source."
)


class RepairTraceDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]], tokenizer, max_length: int):
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.items[index]
        user_content = self._build_user_content(item)
        assistant_content = item["fixed_code"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_text = self.tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = f"{SYSTEM_PROMPT}\n\n{user_content}\n\n"
            full_text = prompt_text + assistant_content

        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_tokens = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_length = int(prompt_tokens["attention_mask"].sum().item())
        labels[:prompt_length] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    @staticmethod
    def _build_user_content(item: dict[str, Any]) -> str:
        attempt_chunks = []
        for attempt in item.get("attempts", []):
            if attempt.get("success"):
                continue
            attempt_chunks.append(
                "\n".join(
                    [
                        f"Attempt {attempt.get('attempt')}",
                        f"Stage: {attempt.get('stage')}",
                        f"Summary: {attempt.get('summary')}",
                        f"STDERR:\n{attempt.get('stderr', '')}",
                    ]
                )
            )
        failures = "\n\n".join(chunk for chunk in attempt_chunks if chunk.strip())
        return (
            f"Task:\n{item.get('task', '')}\n\n"
            f"Observed failures:\n{failures or 'No prior failures.'}\n\n"
            "Produce a corrected Python generator that fits this repository."
        )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA / QLoRA training for Qwen-based gLayout repair.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("glayout_agentic/data/training/repair_traces.jsonl"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset_items = load_jsonl(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    load_in_4bit = not args.no_4bit
    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[part.strip() for part in args.target_modules.split(",") if part.strip()],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = RepairTraceDataset(dataset_items, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit" if load_in_4bit else "adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
