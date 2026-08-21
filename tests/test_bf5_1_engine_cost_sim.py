"""BF5-1: engine cost simulator static table."""
import os
from unittest.mock import patch

from hiob_core.engine_cost_sim import (
    COST_TABLE,
    estimate_engine_cost_cents,
    list_engines,
    normalize_engine,
    normalize_resolution,
    unit_cost_cents,
)


def test_table_has_video_and_image():
    assert "seedance_fast" in COST_TABLE
    assert "openai_image" in COST_TABLE
    assert len(list_engines()) >= 5


def test_video_cost_scales_with_duration():
    a = estimate_engine_cost_cents("seedance_fast", resolution="720p", duration_s=5)
    b = estimate_engine_cost_cents("seedance_fast", resolution="720p", duration_s=10)
    assert a["kind"] == "video"
    assert b["total_cents"] == a["total_cents"] * 2
    assert a["unit_cents"] == unit_cost_cents("seedance_fast", "720p")


def test_image_cost_ignores_duration():
    a = estimate_engine_cost_cents("openai_image", resolution="1k", duration_s=99, n_units=2)
    assert a["kind"] == "image"
    assert a["total_cents"] == unit_cost_cents("openai_image", "1k") * 2


def test_env_override():
    with patch.dict(os.environ, {"ENGINE_COST_SEEDANCE_FAST_720P": "99.5"}):
        assert unit_cost_cents("seedance_fast", "720p") == 99.5


def test_normalization_aliases_defaults_and_unknowns():
    assert normalize_engine("seedance_fast") == "seedance_fast"
    assert normalize_engine("Seedance Fast") == "seedance_fast"
    assert normalize_engine("seedance") == "seedance_fast"
    assert normalize_engine("custom") == "custom"
    assert normalize_engine(None) == "openai_image"
    assert normalize_resolution(None) == "720p"
    assert normalize_resolution(None, image=True) == "default"
    assert normalize_resolution(" Full HD ") == "1080p"
    assert normalize_resolution("custom") == "custom"


def test_invalid_override_and_unknown_resolution_fall_back(monkeypatch):
    monkeypatch.setenv("ENGINE_COST_SEEDANCE_FAST_720P", "not-a-number")
    assert unit_cost_cents("seedance_fast", "720p") == 16.0
    assert unit_cost_cents("openai_image", "unknown") == 4.0
    assert unit_cost_cents("unknown", "unknown") == 4.0


def test_cost_clamps_units_and_duration():
    video = estimate_engine_cost_cents(
        "seedance_fast", resolution="sd", duration_s=-1, n_units=0
    )
    assert video["duration_s"] == 0.0
    assert video["n_units"] == 1
