from hiob_core.llm_runtime import cost_line_item, estimate_cost_cents


def test_cost_line_item_fields():
    item = cost_line_item("gpt-4o-mini", 1000, 500)
    assert set(item.keys()) == {"model", "tokens_in", "tokens_out", "est_cost"}
    assert item["model"] == "gpt-4o-mini"
    assert item["tokens_in"] == 1000
    assert item["tokens_out"] == 500
    assert item["est_cost"] == estimate_cost_cents("gpt-4o-mini", 1000, 500)
    assert item["est_cost"] >= 0
