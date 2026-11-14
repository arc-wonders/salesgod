#!/usr/bin/env python
"""LoRA SFT for the Expression Engine (Layer 6).

Plain transformers + peft (no trl, so chat-template / masking behaviour is explicit
and stable across library versions). Loss is computed ONLY on the final assistant
turn - earlier agent turns in the history contain filled-in digits (that is what
the runtime stores) and must not be learned as outputs.

Default: Qwen/Qwen2.5-7B-Instruct, bf16 LoRA r=16 on all linear layers, one A100 80GB.
  python training/expression/train_lora.py \
      --data data/sft/train.jsonl --seed-data data/sft/seed.jsonl --out runs/acie-expr-qwen7b

40GB card (test3-ocr): bf16 LoRA still fits with --batch 4 --grad-accum 4 (gradient
checkpointing is on by default); use --qlora for 4-bit NF4 base weights if you need
headroom (larger model, longer --max-len, or bigger batches).

Then merge for serving:  python training/expression/merge_lora.py --adapter runs/acie-expr-qwen7b/adapter
"""

from __future__ import annotations

import argparse
import json
import os
import random

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def encode(tok, messages: list[dict], max_len: int) -> dict | None:
    """input_ids/labels with labels = -100 everywhere except the final assistant message."""
    assert messages[-1]["role"] == "assistant", "last message must be the target"
    prompt_text = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tok.apply_chat_template(messages, tokenize=False)
    if not full_text.startswith(prompt_text):
        raise ValueError("chat template: prompt is not a prefix of prompt+answer; cannot mask safely")
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    f_ids = tok(full_text, add_special_tokens=False)["input_ids"]
    if f_ids[:len(p_ids)] != p_ids:  # tokenizer merged across the boundary - tokenize the parts separately
        f_ids = p_ids + tok(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"]
    if len(f_ids) > max_len:
        return None  # never truncate a target; drop the (rare) over-long example
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "labels": labels}


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def make_collator(pad_id: int):
    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            labels[i, :k] = torch.tensor(b["labels"])
            mask[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}
    return collate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", nargs="+", default=["data/sft/train.jsonl"], help="teacher-generated, filtered rows")
    ap.add_argument("--seed-data", default="data/sft/seed.jsonl", help="hand-written gold rows ('' to skip)")
    ap.add_argument("--seed-upsample", type=int, default=4, help="repeat gold rows N times")
    ap.add_argument("--eval-frac", type=float, default=0.03)
    ap.add_argument("--out", default="runs/acie-expr-lora")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--batch", type=int, default=8, help="per-device micro batch")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--qlora", action="store_true", help="4-bit NF4 base weights (bitsandbytes) - for 40GB cards")
    ap.add_argument("--no-gradient-checkpointing", action="store_true")
    ap.add_argument("--attn", default="sdpa", help="sdpa | flash_attention_2 (if flash-attn is installed)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    set_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = [r for p in a.data if os.path.exists(p) for r in read_jsonl(p)]
    gold = read_jsonl(a.seed_data) if a.seed_data and os.path.exists(a.seed_data) else []
    if not rows and not gold:
        raise SystemExit("no training data found")
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    n_eval = int(len(rows) * a.eval_frac) if rows else 0
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:] + gold * a.seed_upsample
    rng.shuffle(train_rows)

    def enc_all(rs):
        out, dropped = [], 0
        for r in rs:
            e = encode(tok, r["messages"], a.max_len)
            if e is None:
                dropped += 1
            else:
                out.append(e)
        return out, dropped

    train_ds, d1 = enc_all(train_rows)
    eval_ds, d2 = enc_all(eval_rows)
    lens = sorted(len(x["input_ids"]) for x in train_ds)
    print(f"train={len(train_ds)} (gold x{a.seed_upsample}={len(gold) * a.seed_upsample}) eval={len(eval_ds)} "
          f"dropped(>max_len)={d1 + d2}; tokens p50={lens[len(lens) // 2]} max={lens[-1]}")

    quant = None
    if a.qlora:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, quantization_config=quant,
                                                 attn_implementation=a.attn, device_map={"": 0})
    model.config.use_cache = False
    gc = not a.no_gradient_checkpointing
    if a.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gc)
    elif gc:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none",
                                             task_type="CAUSAL_LM", target_modules=TARGET_MODULES))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_ratio=0.03, weight_decay=0.0, per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum, bf16=True,
        logging_steps=10, save_strategy="epoch", eval_strategy="epoch" if eval_ds else "no",
        save_total_limit=2, report_to="none", remove_unused_columns=False, dataloader_num_workers=2,
        gradient_checkpointing=False,  # handled above (kwargs differ for QLoRA)
        seed=a.seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ListDataset(train_ds),
                      eval_dataset=ListDataset(eval_ds) if eval_ds else None,
                      data_collator=make_collator(tok.pad_token_id))
    trainer.train()

    adapter_dir = os.path.join(a.out, "adapter")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    with open(os.path.join(a.out, "train_config.json"), "w") as f:
        json.dump({**vars(a), "n_train": len(train_ds), "n_eval": len(eval_ds)}, f, indent=1)
    print(f"adapter -> {adapter_dir}")


if __name__ == "__main__":
    main()
