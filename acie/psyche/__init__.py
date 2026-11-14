"""Psyche layer: the caller's need, the move that serves it, and what has already been said. All deterministic."""

import os


def enabled() -> bool:
    """ACIE_PSYCHE=1 turns on need/move selection and the non-crisis emotion rows. Default off."""
    return os.getenv("ACIE_PSYCHE", "") in ("1", "true", "yes", "on")
