from hiob_core.engine_cost_sim import estimate_render_cents


def test_cost_table():
    a = estimate_render_cents("seedance", "720p", 10)
    b = estimate_render_cents("seedance", "1080p", 10)
    assert a["est_cost_cents"] > 0
    assert b["est_cost_cents"] > a["est_cost_cents"]
    assert a["table_version"] == "bf5-1-v1"
