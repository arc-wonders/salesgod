"""Multi-task head on top of a multilingual encoder (MuRIL / XLM-R)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel

from data import EXPLICIT_LABELS, INTENTS, STATE_DIMS


class TurnClassifier(nn.Module):
    def __init__(self, encoder_name: str, dropout: float = 0.1, encoder: nn.Module | None = None):
        super().__init__()
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(encoder_name)
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.intent = nn.Linear(h, len(INTENTS))
        self.states = nn.ModuleDict({d: nn.Linear(h, 3) for d in STATE_DIMS})
        self.explicit = nn.Linear(h, len(EXPLICIT_LABELS))

    def forward(self, input_ids, attention_mask, token_type_ids=None) -> dict[str, torch.Tensor]:
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and getattr(self.encoder.config, "type_vocab_size", 1) > 1:
            kw["token_type_ids"] = token_type_ids
        hs = self.encoder(**kw).last_hidden_state
        # mean pooling over real tokens: more stable than CLS for short, noisy ASR turns
        m = attention_mask.unsqueeze(-1).to(hs.dtype)
        pooled = self.drop((hs * m).sum(1) / m.sum(1).clamp(min=1.0))
        return {"intent": self.intent(pooled),
                "states": {d: head(pooled) for d, head in self.states.items()},
                "explicit": self.explicit(pooled)}

    # ------------------------------------------------------------------ io

    def save(self, out: Path, tokenizer, meta: dict) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(out / "encoder")
        tokenizer.save_pretrained(out / "encoder")
        heads = {k: v for k, v in self.state_dict().items() if not k.startswith("encoder.")}
        torch.save(heads, out / "heads.pt")
        (out / "acie_meta.json").write_text(json.dumps(
            {"intents": INTENTS, "state_dims": STATE_DIMS, "explicit_labels": EXPLICIT_LABELS, **meta}, indent=2))

    @classmethod
    def load(cls, model_dir: Path, map_location="cpu") -> "TurnClassifier":
        meta = json.loads((model_dir / "acie_meta.json").read_text())
        if meta["intents"] != INTENTS or meta["state_dims"] != STATE_DIMS or meta["explicit_labels"] != EXPLICIT_LABELS:
            raise ValueError("label maps in checkpoint differ from data.py; retrain or pin the old data.py")
        m = cls(str(model_dir / "encoder"))  # fine-tuned encoder weights
        missing, unexpected = m.load_state_dict(torch.load(model_dir / "heads.pt", map_location=map_location), strict=False)
        bad = [k for k in missing if not k.startswith("encoder.")]
        if bad or unexpected:
            raise ValueError(f"head mismatch: missing={bad} unexpected={unexpected}")
        return m
