"""BF5-1: engine cost simulator static table."""
import os
from unittest.mock import patch

from hiob_core.engine_cost_sim import (
    COST_TABLE,
    estimate_engine_cost_cents,
    list_engines,
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
