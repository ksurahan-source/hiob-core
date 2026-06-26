"""HIOB Core — shared runtime for LLM routing, model selection, and platform helpers.

All planets depend on hiob_core; no planet imports another planet.
hiob_core imports no planets (single-direction dependency).

Note: Lazy imports for LLM runtime (requires openai SDK). Model registry is always available.
"""

__version__ = "0.1.0"

# ── Model/Provider Registry (model_providers.py) — always available (no external deps)
from .model_providers import (
    normalize_script_model,
    resolve_script_model,
    script_model_id,
    normalize_interpret_model,
    resolve_interpret_model,
    interpret_model_id,
    normalize_asset_engine,
    resolve_asset_engine,
    engine_is_live,
    engine_produces_video,
    SCRIPT_MODELS,
    INTERPRET_MODELS,
    ASSET_ENGINES,
    STYLE_DEFAULT_ENGINE,
    DEFAULT_SCRIPT_MODEL,
    DEFAULT_INTERPRET_MODEL,
    DEFAULT_ASSET_ENGINE,
)

# ── LLM Runtime (llm_runtime.py) — lazy import to avoid OpenAI SDK dependency at module load
def __getattr__(name: str):
    """Lazy import llm_runtime on first access."""
    if name in [
        "load_prompt",
        "load_localized_prompt",
        "resolve_model",
        "estimate_cost_cents",
        "llm_json",
        "llm_vision_json",
        "llm_json_cached",
        "langfuse_log",
        "JsonRepairError",
    ]:
        from . import llm_runtime
        return getattr(llm_runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Model Registry
    "normalize_script_model",
    "resolve_script_model",
    "script_model_id",
    "normalize_interpret_model",
    "resolve_interpret_model",
    "interpret_model_id",
    "normalize_asset_engine",
    "resolve_asset_engine",
    "engine_is_live",
    "engine_produces_video",
    "SCRIPT_MODELS",
    "INTERPRET_MODELS",
    "ASSET_ENGINES",
    "STYLE_DEFAULT_ENGINE",
    "DEFAULT_SCRIPT_MODEL",
    "DEFAULT_INTERPRET_MODEL",
    "DEFAULT_ASSET_ENGINE",
    # LLM Runtime (lazy)
    "load_prompt",
    "load_localized_prompt",
    "resolve_model",
    "estimate_cost_cents",
    "llm_json",
    "llm_vision_json",
    "llm_json_cached",
    "langfuse_log",
    "JsonRepairError",
]
