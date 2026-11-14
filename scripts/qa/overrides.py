"""Harness-only product overrides.

data/voice/products.json in the repo gives `fitzone-gym` the CLAUDE PRO pitch_hi/pitch_hinglish/pitch_en lines (the
live server's copy has the gym wording: see the persona transcripts). Until that data is fixed the harness swaps in the
public gym wording so the conversations test behaviour, not that data bug. The override only applies while the bug is
present. `repo_data_issues()` reports it so it is not forgotten.
"""

PITCH_OVERRIDES = {
    "fitzone-gym": {
        "pitch_hi": "FitZone एक जिम है जहाँ ट्रेनर, योगा और ज़ुम्बा क्लासेस मिलती हैं, और ये सुबह पाँच से रात ग्यारह बजे तक खुला रहता है।",
        "pitch_hinglish": "FitZone ek gym hai jahan trainer, yoga aur zumba classes milti hain, aur ye subah paanch se raat gyarah baje tak khula rehta hai.",
        "pitch_en": "FitZone is a gym with trainers and yoga and zumba classes, open from 5 in the morning to 11 at night.",
    },
}


def has_wrong_pitch(v: dict) -> bool:
    return v.get("product_name") != "Claude Pro" and any("Claude Pro" in v.get(f, "") for f in ("pitch_hi", "pitch_hinglish", "pitch_en"))


def repo_data_issues(raw: dict) -> list[str]:
    out = []
    for k, v in raw.items():
        if has_wrong_pitch(v):
            out.append(f"products.json: campaign {k!r} pitch_* lines name 'Claude Pro'")
    return out
