"""LP10-1: hiob_core.platform.* thin re-exports stay import-compatible with hiob_platform."""
from __future__ import annotations

import importlib

import pytest

# IDENTICAL dual-SSOT modules converted to re-exports (pronunciation was LP5-7).
# Note: conftest skips item names containing the substring "shim" outside monorepo.
_PLATFORM_REEXPORT_MODULES = (
    "client",
    "notify",
    "brand_kit",
    "role_artifacts",
    "storage",
    "pronunciation",
)


@pytest.mark.parametrize("name", _PLATFORM_REEXPORT_MODULES)
def test_platform_module_reexports_hiob_platform(name: str) -> None:
    # importlib.import_module avoids package-attribute shadowing
    # (e.g. hiob_platform.notify function vs submodule).
    core = importlib.import_module(f"hiob_core.platform.{name}")
    plat = importlib.import_module(f"hiob_platform.{name}")
    assert isinstance(core, type(plat))
    # Every public/underscore name from platform must be present on the re-export.
    for key in dir(plat):
        if key.startswith("__"):
            continue
        assert hasattr(core, key), f"missing re-export: hiob_core.platform.{name}.{key}"
        assert getattr(core, key) is getattr(plat, key), (
            f"re-export diverged: hiob_core.platform.{name}.{key} is not "
            f"hiob_platform.{name}.{key}"
        )


def test_client_get_service_client_identity() -> None:
    core_fn = importlib.import_module("hiob_core.platform.client").get_service_client
    plat_fn = importlib.import_module("hiob_platform.client").get_service_client
    assert core_fn is plat_fn


def test_storage_public_symbols_identity() -> None:
    core_storage = importlib.import_module("hiob_core.platform.storage")
    plat_storage = importlib.import_module("hiob_platform.storage")
    for sym in ("public_url", "upload_artifact", "register_asset_library_item"):
        assert getattr(core_storage, sym) is getattr(plat_storage, sym)


def test_notify_function_identity() -> None:
    core = importlib.import_module("hiob_core.platform.notify")
    plat = importlib.import_module("hiob_platform.notify")
    assert core.notify is plat.notify
    assert core.notify_run_done is plat.notify_run_done
