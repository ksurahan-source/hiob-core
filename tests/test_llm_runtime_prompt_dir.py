"""Regression: prompt dir resolution must not crash on unreadable /root (GHA runners)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hiob_core import llm_runtime


def test_is_usable_dir_true(tmp_path: Path) -> None:
    assert llm_runtime._is_usable_dir(tmp_path) is True


def test_is_usable_dir_false_for_missing(tmp_path: Path) -> None:
    assert llm_runtime._is_usable_dir(tmp_path / "nope") is False


def test_is_usable_dir_permission_error_is_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    """Path.is_dir() may raise PermissionError on /root under GitHub Actions."""

    def boom(self: Path) -> bool:  # noqa: ARG001
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "is_dir", boom)
    assert llm_runtime._is_usable_dir(Path("/root/prompts")) is False


def test_resolve_prompt_dir_survives_root_permission_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("HIOB_PROMPT_DIR", raising=False)

    real_is_dir = Path.is_dir

    def guarded(self: Path) -> bool:
        if str(self) == "/root/prompts" or str(self).startswith("/root/"):
            raise PermissionError(13, "Permission denied", str(self))
        return real_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", guarded)
    # Should not raise — may fall through to default /root/prompts path value
    resolved = llm_runtime._resolve_prompt_dir()
    assert isinstance(resolved, Path)


def test_resolve_prompt_dir_honors_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    monkeypatch.setenv("HIOB_PROMPT_DIR", str(prompts))
    assert llm_runtime._resolve_prompt_dir() == prompts


def test_load_prompt_permission_error_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(llm_runtime, "_PROMPT_DIR", tmp_path)
    llm_runtime._PROMPT_CACHE.clear()

    def boom(*_a: object, **_k: object) -> str:
        raise PermissionError(13, "Permission denied", "x")

    monkeypatch.setattr(Path, "read_text", boom)
    assert llm_runtime.load_prompt("any_role") == ""
