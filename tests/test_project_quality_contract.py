from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_python_floor_matches_the_pinned_platform_dependency() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["requires-python"] == ">=3.11"


def test_reproducible_dependency_lock_is_committed() -> None:
    assert (ROOT / "uv.lock").is_file()
