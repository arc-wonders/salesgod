import pytest

# The crisis/helpline guardrail is ON by default (ACIE_SAFETY=0 turns it off). The tests that exercise it
# (names mention crisis/distress/safety) pin it on explicitly, whatever the environment says.
_GUARD_WORDS = ("crisis", "distress", "safety")


@pytest.fixture(autouse=True)
def _safety_switch(request, monkeypatch):
    if any(w in request.node.name.lower() for w in _GUARD_WORDS) or "safety" in request.node.fspath.basename:
        monkeypatch.setenv("ACIE_SAFETY", "1")
