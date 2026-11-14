#!/usr/bin/env python
"""Merge a LoRA adapter into its base model (bf16) for vLLM serving without --enable-lora.

  python training/expression/merge_lora.py --adapter runs/acie-expr-qwen7b/adapter \
      --out models/acie-expression-qwen7b

QLoRA adapters merge fine into the bf16 base (the 4-bit quantisation was only used
during training). Merging on CPU works (~32GB RAM for 7-8B) with --device cpu.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default=None, help="defaults to base_model_name_or_path from the adapter config")
    ap.add_argument("--out", default="models/acie-expression")
    ap.add_argument("--device", default="auto", help="auto | cpu")
    a = ap.parse_args()

    base = a.base or PeftConfig.from_pretrained(a.adapter).base_model_name_or_path
    print(f"base={base}\nadapter={a.adapter}")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                                 device_map=None if a.device == "cpu" else "auto")
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True, max_shard_size="5GB")
    tok_src = a.adapter if os.path.exists(os.path.join(a.adapter, "tokenizer_config.json")) else base
    AutoTokenizer.from_pretrained(tok_src).save_pretrained(a.out)
    with open(os.path.join(a.out, "acie_merge.json"), "w") as f:
        json.dump({"base": base, "adapter": os.path.abspath(a.adapter)}, f, indent=1)
    print(f"merged model -> {a.out}")


if __name__ == "__main__":
    main()
