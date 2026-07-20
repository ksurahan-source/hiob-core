"""Model/provider registry — the model-selection axis (founder 2026-06-14).

Two INDEPENDENT choices per reel, both user-selectable in the studio 옵션 and
plumbed via the brief (per-reel override → style_mode default):

  SCRIPT model  — which LLM writes the 기획서:  claude | gpt
  ASSET engine  — what generates the per-beat visuals (image OR video):
        openai_image                         (image, GPT)         — live
        seedance_fast / seedance_hi          (video, PiAPI)       — live
        kling / hailuo                       (video, PiAPI)       — live*  (*same
                                              PiAPI gateway/key as Seedance; needs the
                                              PiAPI account to have the model enabled)
        gemini_image                         (image, Google)      — needs GEMINI_API_KEY
        veo                                  (video, Google Veo)  — needs VEO_API_KEY

PiAPI (`PIAPI_KEY`) is a multi-model gateway: Seedance, Kling, Hailuo all ride the
SAME key — the request body's `model` field selects which. So adding Kling/Hailuo
is an adapter over the existing piapi_video client, not a new account.

Resolution order: brief.script_model / brief.asset_engine (explicit per-reel) →
style_mode default → global default. Unknown/blank ⇒ default (legacy image reel
for branded). Script model SSOT is resolve_script_model() / script_model_id()
(LP7-7): live default registry id is qwen → provider model qwen3.7-max (cheap);
frontier flip is §C-9 only. Asset default remains OpenAI-image for branded.
"""
from __future__ import annotations

import os
from typing import Any

# ── SCRIPT models (the 기획서 LLM) ───────────────────────────────────────────
# SSOT: call resolve_script_model() / script_model_id() — never fork model choice
# via ad-hoc os.environ.get("CLAUDE_SCRIPT_MODEL") at call sites (LP7-7).
# Provider model *strings* may still be env-overridden (below); the *registry id*
# default is the cheap live path (qwen). Frontier flip is founder §C-9 only.
def _env_model(name: str, default: str) -> str:
    raw = (os.environ.get(name) or "").strip()
    return raw or default


SCRIPT_MODELS: dict[str, dict[str, Any]] = {
    "claude": {
        "id": "claude", "label": "Claude (Opus 4.8)", "provider": "anthropic",
        # Historical env name CLAUDE_SCRIPT_MODEL = Anthropic model string override.
        "model": _env_model("CLAUDE_SCRIPT_MODEL", "claude-opus-4-8"),
        "env": "ANTHROPIC_API_KEY", "status": "live",
    },
    "gpt": {
        "id": "gpt", "label": "GPT-4o", "provider": "openai",
        "model": "gpt-4o", "env": "OPENAI_API_KEY", "status": "live",
    },
    "qwen": {
        # 도쿄 워크스페이스 실존 모델(콘솔 실측 2026-07-02): qwen3.7-max/plus·qwen3.6-plus/flash.
        # qwen3-max는 이 리전에 없음 — 기본은 최상급 qwen3.7-max, env로 오버라이드.
        "id": "qwen", "label": "Qwen (3.7-max · Tokyo)", "provider": "qwen",
        "model": _env_model("HIOB_QWEN_SCRIPT_MODEL", "qwen3.7-max"),
        "env": "DASHSCOPE_API_KEY", "status": "live",
        "base_url_env": "QWEN_OPENAI_BASE", "base_url_default": "https://ws-15myo7yelloeewav.ap-northeast-1.maas.aliyuncs.com/compatible-mode/v1",
    },
}
# Live production default = qwen (cheap · qwen3.7-max path). Do NOT flip to
# opus/gpt here — that is §C-9 cost decision. DEFAULT_SCRIPT_MODEL is a registry
# id, not a provider model string (provider string comes from script_model_id).
DEFAULT_SCRIPT_MODEL = "qwen"

_SCRIPT_ALIASES = {
    "claude-opus-4-8": "claude", "opus": "claude", "anthropic": "claude",
    "gpt-4o": "gpt", "gpt4o": "gpt", "openai": "gpt", "chatgpt": "gpt",
    "qwen": "qwen", "qwen3": "qwen", "qwen3-max": "qwen", "qwen3.7-max": "qwen", "qwen3.7-plus": "qwen",
}

# ── INTERPRET models (the 요청 해석 / "talk" LLM) ─────────────────────────────
# A SEPARATE axis from SCRIPT (founder 2026-06-14): the conversational request
# interpretation runs on a fast/cheap chat model, while the 기획서/대본 strategy
# stays on Opus 4.8. "Talk with Sonnet 4.6, make the strategy with Opus 4.8."
# Keeping this independent of SCRIPT_MODELS means choosing GPT for the brief does
# NOT change who reads the customer request (default: always Sonnet 4.6).
INTERPRET_MODELS: dict[str, dict[str, Any]] = {
    "qwen": {
        "id": "qwen", "label": "Qwen 3.7 Plus", "provider": "qwen",
        "model": "qwen3.7-plus", "env": "DASHSCOPE_API_KEY", "status": "live",
    },
    "sonnet": {
        "id": "sonnet", "label": "Claude (Sonnet 4.6)", "provider": "anthropic",
        "model": "claude-sonnet-4-6", "env": "ANTHROPIC_API_KEY", "status": "live",
    },
    "gpt_mini": {
        "id": "gpt_mini", "label": "GPT-4o mini", "provider": "openai",
        "model": "gpt-4o-mini", "env": "OPENAI_API_KEY", "status": "live",
    },
}
# D-46 (2026-07-03 fact-check): interpret은 dead code가 아니라 hiob_atropos.interpret이
# 실사용 중이었고 기본이 sonnet이었다 — 텍스트 전부 Qwen 결정에 맞춰 기본 교체.
# sonnet/gpt는 brief.interpret_model 명시 오버라이드로만.
DEFAULT_INTERPRET_MODEL = "qwen"

_INTERPRET_ALIASES = {
    "claude-sonnet-4-6": "sonnet", "sonnet": "sonnet", "claude": "sonnet", "anthropic": "sonnet",
    "gpt-4o-mini": "gpt_mini", "gpt4o-mini": "gpt_mini", "gpt_mini": "gpt_mini", "mini": "gpt_mini",
    "qwen3.7-plus": "qwen", "qwen": "qwen",
}

# ── ASSET engines (per-beat visuals: image or video) ────────────────────────
# `cost_cps` = cents/second of generated video (PiAPI 2026 docs), for the produce
# pre-flight cost gate. Image engines bill per image, tracked elsewhere.
ASSET_ENGINES: dict[str, dict[str, Any]] = {
    "openai_image": {
        "id": "openai_image", "label": "GPT 이미지", "kind": "image",
        "provider": "openai", "env": "OPENAI_API_KEY", "status": "live",
    },
    "seedance_fast": {
        "id": "seedance_fast", "label": "Seedance (빠름·저가)", "kind": "video",
        "provider": "piapi", "piapi_model": "seedance", "task_type": "seedance-2-fast",
        "env": "PIAPI_KEY", "status": "live", "cost_cps": 16,
    },
    "seedance_hi": {
        "id": "seedance_hi", "label": "Seedance (고품질)", "kind": "video",
        "provider": "piapi", "piapi_model": "seedance", "task_type": "seedance-2",
        "env": "PIAPI_KEY", "status": "live", "cost_cps": 20,
    },
    "kling": {
        "id": "kling", "label": "Kling", "kind": "video",
        "provider": "piapi", "piapi_model": "kling", "task_type": "kling-video",
        "env": "PIAPI_KEY", "status": "live", "cost_cps": 28,
        "note": "PiAPI gateway — verify the account has Kling enabled.",
    },
    "hailuo": {
        "id": "hailuo", "label": "Hailuo (MiniMax)", "kind": "video",
        "provider": "piapi", "piapi_model": "hailuo", "task_type": "hailuo-video",
        "env": "PIAPI_KEY", "status": "live", "cost_cps": 24,
        "note": "PiAPI gateway — verify the account has Hailuo enabled.",
    },
    "gemini_image": {
        "id": "gemini_image", "label": "Gemini 이미지", "kind": "image",
        "provider": "google", "env": "GEMINI_API_KEY", "status": "needs_key",
    },
    "veo": {
        "id": "veo", "label": "Google Veo", "kind": "video",
        "provider": "google", "env": "VEO_API_KEY", "status": "needs_key", "cost_cps": 50,
    },
}
DEFAULT_ASSET_ENGINE = "openai_image"

# Per style_mode default engine (founder 2026-06-14): UGC=cheap video, cine=hi video,
# branded=image. Overridable per reel via brief.asset_engine.
STYLE_DEFAULT_ENGINE: dict[str, str] = {
    "branded": "openai_image",
    "ugc": "seedance_fast",
    "cine": "seedance_hi",
}

# Keys are in POST-normalize form (lowercased, spaces stripped, '-'→'_').
_ENGINE_ALIASES = {
    "seedance": "seedance_fast", "seedance_2_fast": "seedance_fast",
    "seedance_2": "seedance_hi", "seedance_high": "seedance_hi",
    "openai": "openai_image", "gpt": "openai_image", "gpt_image": "openai_image",
    "image": "openai_image", "gemini": "gemini_image", "google_veo": "veo",
    # Provider-name routing (brief.video_provider): a bare provider name maps to
    # its live engine so "video_provider: kling" routes the PiAPI model field.
    "kling_video": "kling", "hailuo_video": "hailuo", "minimax": "hailuo",
}


def normalize_script_model(value: Any) -> str | None:
    raw = str(value or "").strip().lower().replace(" ", "")
    if not raw:
        return None
    raw = _SCRIPT_ALIASES.get(raw, raw)
    return raw if raw in SCRIPT_MODELS else None


def resolve_script_model(brief: dict[str, Any] | None) -> str:
    """Sole SSOT for SCRIPT registry id (never None — defaults to DEFAULT_SCRIPT_MODEL).

    LP7-7: call sites must use this (or script_model_id) instead of reading
    DEFAULT_SCRIPT_MODEL / CLAUDE_SCRIPT_MODEL independently — those two used to
    diverge (registry key "claude" vs env model string "claude-opus-4-8" /
    live qwen3.7-max). Resolution order: brief.script_model|script_llm → default.
    """
    brief = brief if isinstance(brief, dict) else {}
    return normalize_script_model(brief.get("script_model") or brief.get("script_llm")) or DEFAULT_SCRIPT_MODEL


def script_model_id(brief: dict[str, Any] | None) -> str:
    """Provider model string for llm_json(model=…) — always via resolve_script_model.

    Re-reads env so CLAUDE_SCRIPT_MODEL / HIOB_QWEN_SCRIPT_MODEL stay aligned with
    the registry entry even if env changed after import (tests / Modal secrets).
    """
    mid = resolve_script_model(brief)
    if mid == "claude":
        return _env_model("CLAUDE_SCRIPT_MODEL", "claude-opus-4-8")
    if mid == "qwen":
        return _env_model("HIOB_QWEN_SCRIPT_MODEL", "qwen3.7-max")
    return SCRIPT_MODELS[mid]["model"]


def normalize_interpret_model(value: Any) -> str | None:
    raw = str(value or "").strip().lower().replace(" ", "")
    if not raw:
        return None
    raw = _INTERPRET_ALIASES.get(raw, raw)
    return raw if raw in INTERPRET_MODELS else None


def resolve_interpret_model(brief: dict[str, Any] | None) -> str:
    """Return an INTERPRET_MODELS id (never None — defaults to sonnet).

    Independent of SCRIPT: reads brief.interpret_model only, so choosing GPT for
    the 기획서 does NOT change who reads the customer request (stays Sonnet 4.6).
    """
    brief = brief if isinstance(brief, dict) else {}
    return normalize_interpret_model(brief.get("interpret_model") or brief.get("interpret_llm")) or DEFAULT_INTERPRET_MODEL


def interpret_model_id(brief: dict[str, Any] | None) -> str:
    """The provider model string (e.g. 'claude-sonnet-4-6') for llm_json(model=…)."""
    return INTERPRET_MODELS[resolve_interpret_model(brief)]["model"]


def normalize_asset_engine(value: Any) -> str | None:
    raw = str(value or "").strip().lower().replace(" ", "").replace("-", "_")
    if not raw:
        return None
    raw = _ENGINE_ALIASES.get(raw, raw)
    return raw if raw in ASSET_ENGINES else None


def resolve_asset_engine(brief: dict[str, Any] | None, style_mode: str | None = None) -> str:
    """Return an ASSET_ENGINES id. Explicit brief override → style_mode default → global."""
    brief = brief if isinstance(brief, dict) else {}
    explicit = normalize_asset_engine(
        brief.get("asset_engine") or brief.get("visual_engine") or brief.get("video_provider")
    )
    if explicit:
        return explicit
    return STYLE_DEFAULT_ENGINE.get(str(style_mode or ""), DEFAULT_ASSET_ENGINE)


def engine_is_live(engine_id: str) -> bool:
    """An engine is usable only if it's marked live AND its API key is present."""
    eng = ASSET_ENGINES.get(str(engine_id or ""))
    if not eng or eng.get("status") != "live":
        return False
    env = eng.get("env")
    return bool(env and os.environ.get(env))


def engine_produces_video(engine_id: str) -> bool:
    return ASSET_ENGINES.get(str(engine_id or ""), {}).get("kind") == "video"


# ── Fail-loud LLM credential checks (script_candidates / llm_runtime) ─────────
# Empty env (OPENAI_API_KEY="", DASHSCOPE_API_KEY missing) previously produced a
# late OpenAI-SDK 401 "No API-key provided." after ~20s of work. Check BEFORE
# paid/network work; never put secret values in the raised message.


def env_names_for_llm_model(model: str) -> tuple[str, ...]:
    """Env var name(s) required for the llm_json / llm_vision_json route of `model`.

    First present non-empty wins (e.g. GEMINI_API_KEY or GOOGLE_API_KEY).
    """
    m = str(model or "").strip().lower()
    if m.startswith("claude"):
        return ("ANTHROPIC_API_KEY",)
    if m.startswith("qwen"):
        return ("DASHSCOPE_API_KEY",)
    if m.startswith("gemini"):
        return ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    # OpenAI path (gpt-*, gpt-5.6-sol, and any unrouted default)
    return ("OPENAI_API_KEY",)


def llm_api_key_present(model: str) -> bool:
    """True if at least one required env for this model is non-empty (after strip)."""
    for name in env_names_for_llm_model(model):
        if (os.environ.get(name) or "").strip():
            return True
    return False


def require_llm_api_key(model: str, *, purpose: str = "script_candidates") -> str:
    """Return the API key for `model`, or raise RuntimeError with a bilingual operator message.

    Does NOT include secret values in the exception. Empty string counts as missing
    (Modal secrets often ship KEY= which shadows and causes opaque 401s).
    """
    names = env_names_for_llm_model(model)
    for name in names:
        val = (os.environ.get(name) or "").strip()
        if val:
            return val
    primary = names[0]
    alts = " / ".join(names)
    raise RuntimeError(
        f"LLM_API_KEY_MISSING: {primary} is unset or empty "
        f"(model={str(model or '')[:64] or '?'}, purpose={purpose}). "
        f"Modal secret `hiob-env` must set {alts} (non-empty). "
        f"{primary} 미설정 또는 빈 값 — Modal 시크릿 hiob-env에 실제 키를 넣으세요. "
        f"(No API key value is printed.)"
    )


def require_script_llm_credentials(brief: dict[str, Any] | None = None) -> str:
    """Resolve script model from brief and require its API key (fail-loud).

    Returns the provider model string that will be used for llm_json.
    """
    mid = script_model_id(brief)
    require_llm_api_key(mid, purpose="script_candidates")
    return mid
