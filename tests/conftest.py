import pytest

# The crisis/helpline guardrail is OFF by default (ACIE_SAFETY, owner 15 Nov 2026). The tests that exercise it
# (names mention crisis/distress/safety) run with it switched on, so the guardrail keeps working if it is re-enabled.
_GUARD_WORDS = ("crisis", "distress", "safety")


@pytest.fixture(autouse=True)
def _safety_switch(request, monkeypatch):
    if any(w in request.node.name.lower() for w in _GUARD_WORDS) or "safety" in request.node.fspath.basename:
        monkeypatch.setenv("ACIE_SAFETY", "1")
