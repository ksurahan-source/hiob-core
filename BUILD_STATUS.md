# hiob-core Build Status — Phase 0.2.1

## ✅ Completion Status

**Build Date:** 2026-06-25  
**Status:** ✅ **BUILT — Ready for importer rewrite**  
**Verification:** All 7 gates passed

---

## What Was Built

### 1. **Core Package Structure** (`packages/hiob-core/`)

```
packages/hiob-core/
├── pyproject.toml                 # Package metadata (setuptools)
├── README.md                       # Phase 0.2 manifest
├── verify_build.sh                # Verification script (7 gates)
├── BUILD_STATUS.md               # This file
├── hiob_core/
│   ├── __init__.py               # Public API re-export (lazy imports)
│   ├── llm_runtime.py            # LLM routing, prompt loading (628 lines, copied)
│   ├── model_providers.py        # Model/engine registry (203 lines, copied)
│   └── platform/
│       ├── __init__.py
│       ├── brand_kit.py          # Brand color/font helpers
│       ├── client.py             # Supabase factory
│       ├── notify.py             # Notifications
│       ├── placement.py          # Caption/image positioning
│       ├── pronunciation.py      # Korean pronunciation
│       ├── role_artifacts.py     # Output persistence
│       ├── runs.py               # agent_run CRUD (read-only caller)
│       ├── storage.py            # R2/S3 signed URLs
│       └── team.py               # (included; hiob-atropos owner in Phase 1)
└── tests/
    └── test_model_providers.py   # Unit tests (FakeClient pattern)
```

### 2. **Modules Copied (Source → hiob-core)**

| Source | Destination | Size | Status |
|--------|-------------|------|--------|
| `apps/modal/workers/llm_runtime.py` | `hiob_core/llm_runtime.py` | 628 lines | ✅ |
| `apps/modal/workers/model_providers.py` | `hiob_core/model_providers.py` | 203 lines | ✅ |
| `packages/platform-py/hiob_platform/*` | `hiob_core/platform/*` | 9 modules | ✅ |

### 3. **Public API (22 exports from `from hiob_core import X`)**

#### LLM Runtime (lazy-imported)
- `load_prompt(name)` — File-based prompt caching
- `load_localized_prompt(name, locale)` — i18n fallback
- `resolve_model(role_row)` — Tier → model resolution
- `estimate_cost_cents(model, in, out)` — USD cost estimate
- `llm_json(system, user, model, on_partial, temp)` — Main LLM call
- `llm_vision_json(system, user, image_urls, model)` — Vision LLM
- `llm_json_cached(system, user, model, ttl_s, temp)` — Deterministic caching
- `langfuse_log(trace_id, role, model, prompt, response, tokens, metadata)` — Tracing
- `JsonRepairError` — JSON repair exception

#### Model Registry (always available)
- `normalize_script_model(value)` · `resolve_script_model(brief)` · `script_model_id(brief)`
- `normalize_interpret_model(value)` · `resolve_interpret_model(brief)` · `interpret_model_id(brief)`
- `normalize_asset_engine(value)` · `resolve_asset_engine(brief, style)` · `engine_is_live(id)` · `engine_produces_video(id)`
- `SCRIPT_MODELS`, `INTERPRET_MODELS`, `ASSET_ENGINES`, `STYLE_DEFAULT_ENGINE`
- `DEFAULT_SCRIPT_MODEL`, `DEFAULT_INTERPRET_MODEL`, `DEFAULT_ASSET_ENGINE`

### 4. **Verification Gates (All ✅ Passed)**

| Gate | Check | Result |
|------|-------|--------|
| 1 | Python syntax: `py_compile` core modules | ✅ |
| 2 | Python syntax: `py_compile` platform modules | ✅ |
| 3 | Package metadata (pyproject.toml) | ✅ |
| 4 | Import validation (model_providers functions) | ✅ |
| 5 | No circular imports | ✅ |
| 6 | Package structure (13 Python modules) | ✅ |
| 7 | Copy fidelity (line count tolerance ±5) | ✅ (exact match: 628 lines) |

---

## Architecture Decisions

### Lazy Imports
**Problem:** `llm_runtime.py` imports `from openai import OpenAI` at module load, causing failure when `openai` SDK is not installed.

**Solution:** Use `__getattr__` to lazy-load `llm_runtime` only when a function from it is accessed. Model registry imports immediately (no external dependencies).

**Impact:** Callers don't notice a difference; tests can import model registry without SDK.

### Prompt Directory Path
**Issue:** Original code uses `Path(__file__).resolve().parent.parent / "prompts"` (relative to `apps/modal/workers/`).

**Fix:** Updated to `Path(__file__).resolve().parent.parent.parent.parent / "apps" / "modal" / "prompts"` (relative to `packages/hiob-core/`).

**Verification:** Works when hiob-core package is at `packages/hiob-core/`.

### Platform Module Copy
**Why:** `platform-py/hiob_platform/` is read-only infra; hiob-core is the public SDK. Copy ensures version coherence.

**Note:** `team.py` (1433 lines) is included but will move to **hiob-atropos** in Phase 0.3. Marked as "included; hiob-atropos owner in Phase 1" to avoid confusion.

---

## Next Steps (Phase 0.2.1 → Phase 0.3)

### Immediate (Importer Rewrite — 3 days)
1. **Rewrite 14 importers** to use `from hiob_core import X` instead of relative imports
2. **Per importer:**
   - Edit file: replace `from .llm_runtime import X` with `from hiob_core import X`
   - Run: `python -m py_compile <file>`
   - Test: `modal run <file> --help` (smoke test)
3. **Consolidate:** Verify all 14 importers pass smoke tests
4. **Delete originals:** Remove `apps/modal/workers/llm_runtime.py`, `model_providers.py` (after verification)

### Phase 0.3 (Cleanup & Testing — 2 days)
1. **Unit tests:** `pytest tests/unit/test_llm_runtime.py` (requires SDK installation)
2. **Remove duplicate:** Delete `platform-py/hiob_platform/` (henceforth, use `hiob-core`)
3. **Documentation:** Update PRD with example imports
4. **Type hints:** Add Zod/Pydantic contracts (Phase 1+)

### Phase 1.0+ (Extension)
- New LLM models (o5, claude-opus-4-9) = hiob-core update only
- Phase 1 "excellence" = original deletion + full test suite

---

## Ownership & Boundaries

### ✅ hiob-core owns:
- `llm_runtime.py` — LLM routing, cost cap, Langfuse tracing
- `model_providers.py` — SCRIPT/INTERPRET/ASSET axes, normalization, live check
- `platform/*` — Supabase, R2, notifications, helpers (infra-only)

### ❌ hiob-core does NOT own:
- `team.py` (→ hiob-atropos, Phase 0.3+)
- Any planet/role logic
- Worker orchestration (team_orchestrator.py)

### ✅ Valid importer patterns:
```python
# Good ✅
from hiob_core import resolve_script_model, llm_json, engine_is_live

# Good ✅
from hiob_core.platform import storage, client

# Bad ❌
from hiob_core.platform.team import sync_clips  # (hiob-atropos in Phase 0.3)

# Bad ❌
import hiob_core
from hiob_core.llm_runtime import _anthropic_json  # private (_prefix)
```

---

## Troubleshooting

### Import Fails: `ModuleNotFoundError: No module named 'openai'`
**Cause:** Trying to import `llm_json` but OpenAI SDK not installed.  
**Fix:** `pip install openai`

### Prompt Files Not Found
**Symptom:** `load_prompt()` returns empty string.  
**Check:** Is `apps/modal/prompts/` present at `<repo>/apps/modal/prompts/`?  
**Fix:** Verify path from Package root: `../../apps/modal/prompts/<name>.txt`

### Modal Import Fails
**Symptom:** `modal run workers/scriptwriter.py` → import error.  
**Steps:**
1. Edit `apps/modal/workers/scriptwriter.py`: change `from .llm_runtime import X` → `from hiob_core import X`
2. Run: `modal run --help` (no args, just test import)
3. If still fails, check Modal secret contains `ANTHROPIC_API_KEY`

---

## Verification Command

To verify the build, run:
```bash
cd packages/hiob-core
bash verify_build.sh
```

Expected output:
```
=== ✅ BUILD VERIFICATION PASSED ===
```

---

## Files Modified This Session

**Created:**
- `packages/hiob-core/hiob_core/__init__.py` (lazy import pattern)
- `packages/hiob-core/hiob_core/llm_runtime.py` (copy + path adjustment)
- `packages/hiob-core/hiob_core/model_providers.py` (exact copy)
- `packages/hiob-core/hiob_core/platform/*` (copies from platform-py)
- `packages/hiob-core/tests/test_model_providers.py` (unit tests)
- `packages/hiob-core/verify_build.sh` (7-gate verification)
- `packages/hiob-core/BUILD_STATUS.md` (this file)

**Not Modified (originals preserved):**
- `apps/modal/workers/llm_runtime.py` — will delete in Phase 0.3
- `apps/modal/workers/model_providers.py` — will delete in Phase 0.3
- `packages/platform-py/hiob_platform/*` — will move to Phase 0.3 cleanup

---

## Sign-Off

**Built by:** Claude Code (Haiku 4.5)  
**Date:** 2026-06-25  
**Verification:** ✅ All 7 gates passed  
**Status:** Ready for importer rewrite (Phase 0.2.1 → importer conversion)

Next checkpoint: All 14 importers rewritten + Modal smoke tests green.
