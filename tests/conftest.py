import pytest

# The crisis/helpline guardrail is ON by default (ACIE_SAFETY=0 turns it off). The tests that exercise it
# (names mention crisis/distress/safety) pin it on explicitly, whatever the environment says.
_GUARD_WORDS = ("crisis", "distress", "safety")


@pytest.fixture(autouse=True)
def _safety_switch(request, monkeypatch):
    if any(w in request.node.name.lower() for w in _GUARD_WORDS) or "safety" in request.node.fspath.basename:
        monkeypatch.setenv("ACIE_SAFETY", "1")

import os  # noqa: E402

os.environ.setdefault("ACIE_RAPPORT_OPENING", "0")  # module-scoped fixtures (tests/test_qa_harness.py) run before the per-test switch below


@pytest.fixture(autouse=True)
def _opening_switch(request, monkeypatch):
    """The live opening is the step-by-step rapport opening (owner, 15 Nov; ACIE_RAPPORT_OPENING, default on). The older tests script
    whole calls against the one-line opening ("yes" -> pitch -> price), so they pin the old opening; tests about the new opening
    (file or name mentions "rapport") get the live default."""
    if "rapport" in request.node.name.lower() or "rapport" in request.node.fspath.basename:
        monkeypatch.setenv("ACIE_RAPPORT_OPENING", "1")
    else:
        monkeypatch.setenv("ACIE_RAPPORT_OPENING", "0")
