from __future__ import annotations

import pytest

import hiob_core
from hiob_core import llm_runtime


def test_lazy_runtime_exports_and_unknown_attribute() -> None:
    assert hiob_core.__getattr__("llm_json") is llm_runtime.llm_json
    with pytest.raises(AttributeError, match="missing"):
        hiob_core.__getattr__("missing")
