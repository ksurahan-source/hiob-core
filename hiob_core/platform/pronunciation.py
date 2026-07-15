"""Single source of truth = hiob_platform.pronunciation (LP5-7 dual-SSOT fix).

This module was a diverged ~76-line copy of hiob_platform.pronunciation (core
lagged platform: missing OECD/ISO/IEC/KC terms, standard-code sino reading,
tts_numeral_reading / caption_numerals_to_digits). Kept as a thin re-export
shim so `from hiob_core.platform import pronunciation` and
`from hiob_core.platform.pronunciation import …` keep working.

Edit hiob_platform.pronunciation, never this file.
"""
from __future__ import annotations

import hiob_platform.pronunciation as _src

# Re-export every public + single-underscore name (workers may import privates).
globals().update({k: getattr(_src, k) for k in dir(_src) if not k.startswith("__")})
