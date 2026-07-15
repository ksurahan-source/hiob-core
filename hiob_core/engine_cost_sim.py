"""BF5-1: static engine×resolution×duration cost table (env-overridable rates)."""
from __future__ import annotations

import os
from typing import Any


# USD cents per second at 720p baseline
_BASE_CENTS_PER_SEC = {
    "seedance": 1.2,
    "veo": 2.5,
    "kling": 1.8,
    "runway": 2.0,
    "luma": 1.5,
}

_RES_MULT = {"480p": 0.7, "720p": 1.0, "1080p": 1.6, "4k": 3.0}


def estimate_render_cents(engine: str, resolution: str, duration_s: float) -> dict[str, Any]:
    eng = str(engine or "seedance").lower()
    res = str(resolution or "720p").lower()
    base = float(os.environ.get(f"COST_{eng.upper()}_CPS", _BASE_CENTS_PER_SEC.get(eng, 1.0)))
    mult = _RES_MULT.get(res, 1.0)
    dur = max(0.0, float(duration_s or 0))
    cents = base * mult * dur
    return {
        "engine": eng,
        "resolution": res,
        "duration_s": dur,
        "est_cost_cents": round(cents, 4),
        "table_version": "bf5-1-v1",
    }
