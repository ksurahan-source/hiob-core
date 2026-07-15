"""BF5-1: static engine × resolution × duration cost simulator.

Pure table + env overrides. No network. Used for pre-flight estimates.
"""
from __future__ import annotations

import os
from typing import Any

# cents per second (video) or cents per image (image) — static defaults.
# Override any cell via env ENGINE_COST_<ENGINE>_<RES> e.g. ENGINE_COST_SEEDANCE_FAST_720P=18
COST_TABLE: dict[str, dict[str, float]] = {
    # video engines: values are cents-per-second at listed resolution
    "seedance_fast": {"480p": 12.0, "720p": 16.0, "1080p": 22.0},
    "seedance_hi": {"480p": 16.0, "720p": 20.0, "1080p": 28.0},
    "kling": {"480p": 20.0, "720p": 28.0, "1080p": 36.0},
    "hailuo": {"480p": 18.0, "720p": 24.0, "1080p": 32.0},
    "veo": {"480p": 40.0, "720p": 50.0, "1080p": 70.0},
    # image engines: cents per image (duration ignored)
    "openai_image": {"1k": 4.0, "2k": 8.0, "default": 4.0},
    "gemini_image": {"1k": 3.0, "2k": 6.0, "default": 3.0},
    "qwen-image-2.0": {"1k": 2.0, "2k": 4.0, "default": 2.0},
    "qwen-image-edit-max": {"1k": 3.0, "2k": 5.0, "default": 3.0},
}

_IMAGE_ENGINES = frozenset(
    {"openai_image", "gemini_image", "qwen-image-2.0", "qwen-image-edit-max"}
)

_ENGINE_ALIASES = {
    "seedance": "seedance_fast",
    "openai": "openai_image",
    "gpt-image": "openai_image",
    "gpt_image": "openai_image",
    "gemini": "gemini_image",
}


def normalize_engine(engine: str | None) -> str:
    e = str(engine or "").strip().lower().replace(" ", "_").replace("-", "_")
    # keep dots for qwen-image-2.0 style after re-hyphenating common forms
    raw = str(engine or "").strip()
    if raw in COST_TABLE:
        return raw
    if e in COST_TABLE:
        return e
    # try aliases
    if e in _ENGINE_ALIASES:
        return _ENGINE_ALIASES[e]
    # restore hyphens for qwen ids
    hy = str(engine or "").strip()
    if hy in COST_TABLE:
        return hy
    return raw or "openai_image"


def normalize_resolution(res: str | None, *, image: bool = False) -> str:
    r = str(res or "").strip().lower()
    if not r:
        return "default" if image else "720p"
    r = r.replace(" ", "")
    aliases = {
        "hd": "720p",
        "fhd": "1080p",
        "fullhd": "1080p",
        "sd": "480p",
        "1024": "1k",
        "1024x1024": "1k",
        "2048": "2k",
    }
    return aliases.get(r, r)


def _env_override(engine: str, resolution: str) -> float | None:
    key = f"ENGINE_COST_{engine}_{resolution}".upper().replace(".", "_").replace("-", "_")
    raw = (os.environ.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def unit_cost_cents(engine: str, resolution: str | None = None) -> float:
    """Return unit cost (cents/sec for video, cents/image for image)."""
    eng = normalize_engine(engine)
    image = eng in _IMAGE_ENGINES or eng not in {
        "seedance_fast", "seedance_hi", "kling", "hailuo", "veo",
    } and eng in COST_TABLE and "720p" not in COST_TABLE.get(eng, {})
    # classify by table shape
    table = COST_TABLE.get(eng) or COST_TABLE["openai_image"]
    is_image = "720p" not in table
    res = normalize_resolution(resolution, image=is_image)
    env = _env_override(eng, res)
    if env is not None:
        return env
    if res in table:
        return float(table[res])
    # fallback first value
    return float(next(iter(table.values())))


def estimate_engine_cost_cents(
    engine: str,
    *,
    resolution: str | None = None,
    duration_s: float = 0.0,
    n_units: int = 1,
) -> dict[str, Any]:
    """Estimate total cost in cents for engine×res×duration (or n images).

    Returns dict: {engine, resolution, duration_s, n_units, unit_cents, total_cents, kind}.
    """
    eng = normalize_engine(engine)
    table = COST_TABLE.get(eng) or COST_TABLE["openai_image"]
    is_image = "720p" not in table
    res = normalize_resolution(resolution, image=is_image)
    unit = unit_cost_cents(eng, res)
    n = max(1, int(n_units))
    if is_image:
        total = unit * n
        kind = "image"
        dur = 0.0
    else:
        dur = max(0.0, float(duration_s))
        total = unit * dur * n
        kind = "video"
    return {
        "engine": eng,
        "resolution": res,
        "duration_s": dur,
        "n_units": n,
        "unit_cents": unit,
        "total_cents": round(total, 4),
        "kind": kind,
    }


def list_engines() -> list[str]:
    return sorted(COST_TABLE.keys())
