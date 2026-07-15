"""Single source of truth = hiob_platform.role_artifacts (LP10-1 dual-SSOT fix).

This module was an IDENTICAL copy of hiob_platform.role_artifacts. Kept as a
thin re-export shim so `from hiob_core.platform import role_artifacts` and
`from hiob_core.platform.role_artifacts import …` keep working.

Edit hiob_platform.role_artifacts, never this file.
"""
from __future__ import annotations

import importlib

_src = importlib.import_module("hiob_platform.role_artifacts")

# Re-export every public + single-underscore name (workers may import privates).
globals().update({k: getattr(_src, k) for k in dir(_src) if not k.startswith("__")})
