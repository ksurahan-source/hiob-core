"""Single source of truth = hiob_platform.notify (LP10-1 dual-SSOT fix).

This module was an IDENTICAL copy of hiob_platform.notify. Kept as a thin
re-export shim so `from hiob_core.platform import notify` and
`from hiob_core.platform.notify import …` keep working.

Uses importlib (not `import hiob_platform.notify`) because hiob_platform.__init__
binds the `notify` *function*, which would shadow the submodule attribute.

Edit hiob_platform.notify, never this file.
"""
from __future__ import annotations

import importlib

_src = importlib.import_module("hiob_platform.notify")

# Re-export every public + single-underscore name (workers may import privates).
globals().update({k: getattr(_src, k) for k in dir(_src) if not k.startswith("__")})
