from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from hiob_core import llm_runtime


def _star_deps_path() -> Path:
    source_root = (os.environ.get("HIOB_STAR_SOURCE") or "").strip()
    if source_root:
        path = (
            Path(source_root)
            / "hiob_star"
            / "script_candidates"
            / "deps.py"
        )
        if path.is_file():
            return path
        pytest.fail(f"HIOB_STAR_SOURCE has no production deps module: {path}")

    spec = importlib.util.find_spec("hiob_star")
    if spec and spec.submodule_search_locations:
        path = (
            Path(next(iter(spec.submodule_search_locations)))
            / "script_candidates"
            / "deps.py"
        )
        if path.is_file():
            return path
    pytest.skip("hiob-star source is unavailable for the cross-package contract")


def _load_star_production_deps_module(path: Path):
    module_name = "_hiob_star_production_deps_contract"
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


def test_production_deps_forwards_reserved_writer_options(monkeypatch):
    star_deps_path = _star_deps_path()
    clients = []
    requests = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            clients.append(kwargs)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create),
            )

        def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"ok": true}'),
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=1,
                    completion_tokens=2,
                ),
            )

    monkeypatch.setattr(llm_runtime, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    hiob_ares = _package("hiob_ares")
    hiob_ares_script_write = _package("hiob_ares.script_write")
    hiob_ares_insert = ModuleType("hiob_ares.script_write.insert")
    hiob_ares_insert.make_insert_script_candidate = (
        lambda **kwargs: object()
    )
    monkeypatch.setitem(sys.modules, "hiob_ares", hiob_ares)
    monkeypatch.setitem(
        sys.modules,
        "hiob_ares.script_write",
        hiob_ares_script_write,
    )
    monkeypatch.setitem(
        sys.modules,
        "hiob_ares.script_write.insert",
        hiob_ares_insert,
    )

    hiob_platform = ModuleType("hiob_platform")
    hiob_platform.end_span = lambda *args, **kwargs: None
    hiob_platform.get_service_client = lambda: object()
    hiob_platform.start_span = lambda *args, **kwargs: {"id": "span"}
    monkeypatch.setitem(sys.modules, "hiob_platform", hiob_platform)

    hiob_star = _package("hiob_star")
    hiob_star_orchestration = ModuleType("hiob_star.orchestration")
    hiob_star_orchestration._beat_row_with_sfx = lambda row: row
    hiob_star_orchestration._segments = lambda row: []
    hiob_star_orchestration._update_production_job_safe = (
        lambda *args, **kwargs: None
    )
    hiob_star_orchestration.emit_production_event = (
        lambda *args, **kwargs: None
    )
    hiob_star_sheet = ModuleType("hiob_star.production_sheet")
    hiob_star_sheet.build_production_sheet = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "hiob_star", hiob_star)
    monkeypatch.setitem(
        sys.modules,
        "hiob_star.orchestration",
        hiob_star_orchestration,
    )
    monkeypatch.setitem(
        sys.modules,
        "hiob_star.production_sheet",
        hiob_star_sheet,
    )

    module = _load_star_production_deps_module(star_deps_path)
    deps = module.production_deps()
    monkeypatch.delenv("HIOB_ARES_XL_SCRIPT_MODEL", raising=False)
    assert deps.script_model_id(
        {"ares_xl_jkpa_authority": {"status": "sealed"}}
    ) == "gpt-4o"
    common = {
        "system": "system",
        "user": "user",
        "model": "gpt-4o",
        "idempotency_key": "ares-xl-writer:cross-package",
        "max_completion_tokens": 32000,
    }

    assert deps.llm_json(**common) == ({"ok": True}, 1, 2)
    assert deps.llm_vision_json(
        **common,
        image_urls=["https://example.test/proof.jpg"],
    ) == ({"ok": True}, 1, 2)

    assert clients == [
        {"api_key": "test-openai-key", "max_retries": 0},
        {"api_key": "test-openai-key", "max_retries": 0},
    ]
    assert [request["max_completion_tokens"] for request in requests] == [
        32000,
        32000,
    ]
    assert [request["extra_headers"] for request in requests] == [
        {"X-Client-Request-Id": "ares-xl-writer:cross-package"},
        {"X-Client-Request-Id": "ares-xl-writer:cross-package"},
    ]
