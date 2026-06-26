"""Shared LLM runtime — model routing, prompt loading, critic loop,
streaming previews, cost cap enforcement, and Langfuse tracing.

This is the substrate every role call goes through. Pulling it out of
team_orchestrator keeps the orchestrator readable and makes the SV-bar
upgrades (P1–P3) testable in one place.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

# --------------------------------------------------------------------
# Prompt loading — prompts live in git, not in DB.
# --------------------------------------------------------------------

_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "apps" / "modal" / "prompts"
_PROMPT_CACHE: dict[str, str] = {}


def load_prompt(name: str) -> str:
    """Read `apps/modal/prompts/<name>.txt`. Falls back to empty string
    so a missing prompt doesn't kill a run."""
    if name in _PROMPT_CACHE:
        return _PROMPT_CACHE[name]
    path = _PROMPT_DIR / f"{name}.txt"
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        text = ""
    _PROMPT_CACHE[name] = text
    return text


def load_localized_prompt(name: str, locale: str | None) -> str:
    """Locale-aware prompt load (i18n Phase 0 extension point): prefer
    `<name>_<locale>.txt`, fall back to the base `<name>.txt`.

    Phase 0 ships no localized prompt files, so every locale falls back to the
    base prompt and stays byte-identical. Phase 1+ drops in `scriptwriter_zh.txt`
    etc. and existing `load_prompt` callers opt in by switching to this helper."""
    code = str(locale or "").strip().lower()
    if code:
        localized = load_prompt(f"{name}_{code}")
        if localized:
            return localized
    return load_prompt(name)


# --------------------------------------------------------------------
# Model routing — tier → concrete model id via ENV.
# --------------------------------------------------------------------

# Tier → (env var, default model). Provider-agnostic: llm_json() dispatches to
# Anthropic vs OpenAI by model-name prefix, so a tier can point at any model.
# Defaults are Claude-first, routed by seniority of judgment:
#   premium = Opus 4.8 (creative brain) · default = Sonnet 4.6 · cheap = Haiku 4.5.
# Set HIOB_*_MODEL in the Modal secret to override without a code change.
TIER_ENV = {
    "cheap":   ("HIOB_CHEAP_MODEL",   "claude-haiku-4-5"),
    "default": ("HIOB_DEFAULT_MODEL", "claude-sonnet-4-6"),
    "premium": ("HIOB_PREMIUM_MODEL", "claude-opus-4-8"),
}


def resolve_model(role_row: dict | None) -> tuple[str, str]:
    """Pick the model for this role. Returns (model_id, tier_label).

    Resolution order:
      1. agent_role.attributes.model_override (explicit pin)
      2. agent_role.model_tier → ENV
      3. agent_role.default_model (legacy column)
      4. fallback to HIOB_DEFAULT_MODEL (Sonnet 4.6)
    """
    if not role_row:
        return os.environ.get("HIOB_DEFAULT_MODEL", "claude-sonnet-4-6"), "default"

    attrs = role_row.get("attributes") or {}
    if isinstance(attrs, dict) and attrs.get("model_override"):
        return str(attrs["model_override"]), "override"

    tier = (role_row.get("model_tier") or "default").lower()
    env_key, default_model = TIER_ENV.get(tier, TIER_ENV["default"])
    return os.environ.get(env_key, default_model), tier


def estimate_cost_cents(model: str, tokens_in: int, tokens_out: int) -> float:
    """Rough USD-cent cost estimate for budget enforcement.

    Returns FLOAT cents (not int) so that sub-cent calls don't truncate
    to zero — without this, every short role call rounds to 0¢ and the
    per-call cost badge shows $0.0000 even though tokens are non-zero.

    Real billing happens at the provider. This is intentionally cheap
    arithmetic — the goal is to stop runaway loops before they cost
    real money, not perfect accounting.
    """
    # Prices per 1M tokens (USD), approximate as of May 2026.
    rates = {
        "gpt-4o-mini":       (0.15, 0.60),
        "gpt-4o":            (2.50, 10.00),
        "gpt-4.1-mini":      (0.40, 1.60),
        "gpt-4.1":           (2.00, 8.00),
        "gpt-5-mini":        (0.25, 1.00),
        "gpt-5":             (3.00, 12.00),
        "gpt-5.5":           (1.50, 6.00),
        "o4-mini":           (0.30, 1.20),
        "claude-haiku-4-5":  (0.80, 4.00),
        "claude-sonnet-4-6": (3.00, 15.00),
        "claude-opus-4-7":   (5.00, 25.00),
        "claude-opus-4-8":   (5.00, 25.00),
    }
    in_rate, out_rate = rates.get(model, (1.0, 4.0))
    cents = (
        (tokens_in / 1_000_000) * in_rate + (tokens_out / 1_000_000) * out_rate
    ) * 100.0
    return max(0.0, float(cents))


def _is_claude_model(model: str) -> bool:
    return str(model or "").lower().startswith("claude")


class JsonRepairError(ValueError):
    def __init__(self, message: str, *, tokens_in: int = 0, tokens_out: int = 0):
        super().__init__(message)
        self.tokens_in = int(tokens_in or 0)
        self.tokens_out = int(tokens_out or 0)


def _parse_json_text(raw: str) -> dict:
    raw = (raw or "{}").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return json.loads(raw, strict=False)
        except json.JSONDecodeError:
            pass
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            snippet = raw[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return json.loads(snippet, strict=False)
        raise


def _require_json_object(value: Any, *, source: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{source} returned JSON, but not a JSON object")
    return value


def _anthropic_repair_json_text(client: Any, *, model: str, raw: str) -> tuple[dict, int, int]:
    """Repair a malformed Claude JSON response once.

    Claude usually obeys the raw-JSON instruction, but long script payloads can
    occasionally miss a comma or trailing brace. A single deterministic repair
    pass is cheaper and safer than failing a live production job.
    """
    resp = client.messages.create(
        model=model,
        max_tokens=int(os.environ.get("ANTHROPIC_MAX_TOKENS", "16000")),
        temperature=0,
        system=(
            "You repair malformed JSON. Return only one syntactically valid JSON object. "
            "Do not add markdown, explanation, comments, or fields that are not implied by the input."
        ),
        messages=[{
            "role": "user",
            "content": (
                "Repair this malformed JSON object so json.loads can parse it. "
                "Preserve Korean text and field names exactly where possible.\n\n"
                + (raw or "")
            ),
        }],
    )
    repaired_parts: list[str] = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            repaired_parts.append(getattr(block, "text", ""))
    usage = getattr(resp, "usage", None)
    repair_in = int(getattr(usage, "input_tokens", 0) or 0)
    repair_out = int(getattr(usage, "output_tokens", 0) or 0)
    repaired_raw = "".join(repaired_parts)
    try:
        parsed = _require_json_object(_parse_json_text(repaired_raw), source="Claude JSON repair")
    except (json.JSONDecodeError, ValueError) as exc:
        raise JsonRepairError(
            "Claude JSON repair failed to return a valid JSON object",
            tokens_in=repair_in,
            tokens_out=repair_out,
        ) from exc
    return (
        parsed,
        repair_in,
        repair_out,
    )


# --------------------------------------------------------------------
# Langfuse tracing — graceful no-op when keys missing.
# --------------------------------------------------------------------

_LANGFUSE_CLIENT = None
_LANGFUSE_INIT_TRIED = False


def _langfuse():
    global _LANGFUSE_CLIENT, _LANGFUSE_INIT_TRIED
    if _LANGFUSE_INIT_TRIED:
        return _LANGFUSE_CLIENT
    _LANGFUSE_INIT_TRIED = True
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
    sk = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    if not pk or not sk:
        return None
    try:
        from langfuse import Langfuse  # type: ignore
        _LANGFUSE_CLIENT = Langfuse(public_key=pk, secret_key=sk, host=host)
    except Exception:
        _LANGFUSE_CLIENT = None
    return _LANGFUSE_CLIENT


def langfuse_log(
    *,
    trace_id: str | None,
    role: str,
    model: str,
    prompt: str,
    response: str,
    tokens_in: int,
    tokens_out: int,
    metadata: dict | None = None,
) -> str | None:
    """Best-effort send to Langfuse. Returns the trace id (provided or new)."""
    client = _langfuse()
    if not client:
        return trace_id
    try:
        trace = client.trace(
            id=trace_id,
            name=f"role:{role}",
            metadata=metadata or {},
        )
        trace.generation(
            name=role,
            model=model,
            input=prompt[:4000],
            output=response[:4000],
            usage={"input": tokens_in, "output": tokens_out},
        )
        return trace.id
    except Exception:
        return trace_id


# --------------------------------------------------------------------
# LLM call (sync) — used by the role runner.
# --------------------------------------------------------------------

def llm_json(
    *,
    system: str,
    user: str,
    model: str,
    on_partial: callable | None = None,
    temperature: float | None = None,
) -> tuple[dict, int, int]:
    """Call OpenAI or Anthropic; return (parsed_json, tokens_in, tokens_out).

    on_partial(text_so_far) is called with the cumulative stream so the
    orchestrator can incrementally persist `agent_call.output_preview`
    for the UI. When None or streaming unavailable, behaves as a normal
    blocking call.

    temperature: optional temperature for sampling (0.0-2.0). If None, uses model default.
    """
    if _is_claude_model(model):
        return _anthropic_json(system=system, user=user, model=model, temperature=temperature)

    # Gemini via its OpenAI-compatible endpoint (founder-provided key). Used by url_ingest
    # for cheap long-context extraction; falls through to OpenAI for everything else.
    if model.startswith("gemini"):
        gclient = OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", ""),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        gkwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        if temperature is not None:
            gkwargs["temperature"] = temperature
        gresp = gclient.chat.completions.create(**gkwargs)
        graw = gresp.choices[0].message.content or "{}"
        gusage = gresp.usage
        return (
            _parse_json_text(graw),
            gusage.prompt_tokens if gusage else 0,
            gusage.completion_tokens if gusage else 0,
        )

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    if on_partial is None:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or "{}"
        usage = resp.usage
        return (
            _parse_json_text(raw),
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )

    # Streaming path — emit periodic partials so the editor reflects
    # progress. We still parse JSON at the end; partial JSON inside a
    # `json_object` response is not always valid mid-stream.
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    stream = client.chat.completions.create(**kwargs)
    raw_parts: list[str] = []
    tokens_in = 0
    tokens_out = 0
    last_emit = 0.0
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            raw_parts.append(chunk.choices[0].delta.content)
            now = time.monotonic()
            if now - last_emit >= 0.6:
                last_emit = now
                try:
                    on_partial("".join(raw_parts)[-2000:])
                except Exception:
                    pass
        if getattr(chunk, "usage", None):
            tokens_in = chunk.usage.prompt_tokens or 0
            tokens_out = chunk.usage.completion_tokens or 0
    raw = "".join(raw_parts) or "{}"
    return _parse_json_text(raw), tokens_in, tokens_out


def llm_vision_json(*, system: str, user: str, image_urls: list[str], model: str = "gpt-4o") -> tuple[dict, int, int]:
    """VISION — let a vision model SEE image_urls and return JSON. Used so the agent reads the
    brand's UPLOADED images to understand the brand's own content. Caps at 8 images. Routes to
    Claude vision when `model` is a claude-* id (LOOP_STUDIO: Opus is the only brain — no gpt-4o
    analyst); otherwise OpenAI vision (detail='low' keeps cost down)."""
    urls = [u for u in (image_urls or []) if u][:8]
    if _is_claude_model(model):
        return _anthropic_vision_json(system=system, user=user, image_urls=urls, model=model)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    content: list[dict] = [{"type": "text", "text": user}]
    for url in urls:
        content.append({"type": "image_url", "image_url": {"url": url, "detail": "low"}})
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    usage = resp.usage
    return (_parse_json_text(raw), usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0)


def _anthropic_vision_json(*, system: str, user: str, image_urls: list[str], model: str) -> tuple[dict, int, int]:
    """Claude VISION via URL image sources (the bucket is public-read). Lets Opus SEE the brand's
    real images and return JSON — so the brain itself reads the folder (LOOP_STUDIO APPLY#3)."""
    try:
        from anthropic import Anthropic  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("anthropic package is required for Claude models") from exc
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    content: list[dict] = [{"type": "text", "text": user}]
    for url in image_urls:
        content.append({"type": "image", "source": {"type": "url", "url": url}})
    client = client.with_options(
        timeout=float(os.environ.get("ANTHROPIC_TIMEOUT_S", "300")),
        max_retries=int(os.environ.get("ANTHROPIC_MAX_RETRIES", "1")),
    )
    resp = client.messages.create(
        model=model,
        max_tokens=int(os.environ.get("ANTHROPIC_MAX_TOKENS", "16000")),
        system=f"{system}\n\nReturn only one valid JSON object. Do not wrap it in markdown.",
        messages=[{"role": "user", "content": content}],
    )
    raw_parts: list[str] = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            raw_parts.append(getattr(block, "text", ""))
    usage = getattr(resp, "usage", None)
    raw = "".join(raw_parts)
    tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
    tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
    try:
        parsed = _parse_json_text(raw)
    except json.JSONDecodeError:
        try:
            parsed, repair_in, repair_out = _anthropic_repair_json_text(client, model=model, raw=raw)
        except JsonRepairError as exc:
            raise JsonRepairError(
                str(exc),
                tokens_in=tokens_in + exc.tokens_in,
                tokens_out=tokens_out + exc.tokens_out,
            ) from exc
        tokens_in += repair_in
        tokens_out += repair_out
    return (parsed, tokens_in, tokens_out)


def _anthropic_cli_json(*, system: str, user: str, model: str) -> tuple[dict, int, int] | None:
    """Attempt to call the local 'claude' CLI in print mode to use the user's
    Claude Pro/Max subscription session and avoid API billing.

    OPT-IN ONLY (HIOB_LLM_CLI=1, local dev): a Modal container has no
    subscription session, and probing `npx @anthropic-ai/claude-code` there
    would add up to 120s dead latency per LLM call before the API fallback.
    """
    if os.environ.get("HIOB_LLM_CLI", "").strip().lower() not in {"1", "true", "yes"}:
        return None
    import subprocess
    import shutil

    # Check if 'claude' or 'npx' is available locally
    claude_path = shutil.which("claude")
    npx_path = shutil.which("npx")

    if not claude_path and not npx_path:
        return None

    cmd = [claude_path or "claude"]
    if not claude_path:
        cmd = ["npx", "@anthropic-ai/claude-code"]

    model_arg = model
    if "opus" in model.lower():
        model_arg = "opus"
    elif "sonnet" in model.lower():
        model_arg = "sonnet"
    elif "haiku" in model.lower():
        model_arg = "haiku"

    full_prompt = (
        f"System Prompt:\n{system}\n\n"
        f"User Prompt:\n{user}\n\n"
        "Return ONLY a single valid raw JSON object. Do not wrap it in markdown. No explanation."
    )

    run_cmd = [
        cmd[0],
        *cmd[1:],
        "--print",
        full_prompt,
        "--tools", "",
        "--permission-mode", "dontAsk",
        "--model", model_arg
    ]

    try:
        env = os.environ.copy()
        if "ANTHROPIC_API_KEY" in env:
            del env["ANTHROPIC_API_KEY"]

        result = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )

        if result.returncode == 0:
            raw_output = result.stdout.strip()
            # Clean up markdown JSON wrapper if present
            if raw_output.startswith("```json"):
                raw_output = raw_output[7:]
            elif raw_output.startswith("```"):
                raw_output = raw_output[3:]
            if raw_output.endswith("```"):
                raw_output = raw_output[:-3]
            raw_output = raw_output.strip()

            parsed = _parse_json_text(raw_output)
            tokens_in = len(system + user) // 4
            tokens_out = len(raw_output) // 4
            return parsed, tokens_in, tokens_out
        else:
            return None
    except Exception:
        return None


def _anthropic_json(*, system: str, user: str, model: str, temperature: float | None = None) -> tuple[dict, int, int]:
    # Try using local subscription-based Claude CLI first to avoid API billing
    cli_res = _anthropic_cli_json(system=system, user=user, model=model)
    if cli_res is not None:
        return cli_res

    # Fallback to standard Anthropic SDK API billing
    try:
        from anthropic import Anthropic  # type: ignore
    except Exception as exc:
        raise RuntimeError("anthropic package is required for Claude models") from exc
    # Bound a stuck call. An untimed messages.create() (SDK default ~10min ×
    # retries) was the "기획서 작성 중… neverends" hang: a hung Opus draft blocked
    # the whole script-candidates job, so the studio spun until its poll timeout.
    # Opus drafts can legitimately take 2-3 min, so default 300s with a single
    # retry — a genuine hang fails fast enough for the studio to surface it.
    client = Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        timeout=float(os.environ.get("ANTHROPIC_TIMEOUT_S", "300")),
        max_retries=int(os.environ.get("ANTHROPIC_MAX_RETRIES", "1")),
    )
    kwargs = {
        "model": model,
        "max_tokens": int(os.environ.get("ANTHROPIC_MAX_TOKENS", "16000")),
        "system": (
            f"{system}\n\n"
            "Return only one valid JSON object. Do not wrap it in markdown."
        ),
        "messages": [{"role": "user", "content": user}],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = client.messages.create(**kwargs)
    raw_parts: list[str] = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            raw_parts.append(getattr(block, "text", ""))
    usage = getattr(resp, "usage", None)
    raw = "".join(raw_parts)
    tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
    tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
    try:
        parsed = _parse_json_text(raw)
    except json.JSONDecodeError:
        try:
            parsed, repair_in, repair_out = _anthropic_repair_json_text(client, model=model, raw=raw)
        except JsonRepairError as exc:
            raise JsonRepairError(
                str(exc),
                tokens_in=tokens_in + exc.tokens_in,
                tokens_out=tokens_out + exc.tokens_out,
            ) from exc
        tokens_in += repair_in
        tokens_out += repair_out
    return (parsed, tokens_in, tokens_out)


# ── Deterministic LLM call cache (classification / guardrail / OCR callers only) ──

_CACHE_TTL_DEFAULT = 3600  # 1 hour


def llm_json_cached(
    *,
    system: str,
    user: str,
    model: str,
    ttl_s: int = _CACHE_TTL_DEFAULT,
    temperature: float | None = None,
) -> tuple[dict, int, int]:
    """Cached wrapper for llm_json using sha256(model+system+user) as key.

    ONLY for deterministic callers (classification, intent interpretation, OCR/doc
    parsing, URL extraction). NEVER use for creative calls (scriptwriter, visuals).
    Fail-open: any cache error falls through to a direct LLM call.
    Logs [cache] HIT/MISS for observability.
    """
    from infra import redis_client  # lazy — avoids circular import at module load

    raw_key = f"{model}:{system}:{user}"
    cache_key = "llm:" + hashlib.sha256(raw_key.encode()).hexdigest()

    try:
        cached_str = redis_client.cache_get(cache_key)
        if cached_str is not None:
            hit = json.loads(cached_str)
            saved = hit.get("_tok_in", 0)
            print(f"[cache] HIT key={cache_key[4:20]}... saved~{saved} tok (model={model})")
            return hit["result"], 0, 0
    except Exception as exc:
        print(f"[cache] GET error (fail-open): {type(exc).__name__}: {str(exc)[:120]}")

    result, tin, tout = llm_json(system=system, user=user, model=model, temperature=temperature)
    print(f"[cache] MISS key={cache_key[4:20]}... stored {tin} tok (model={model})")

    try:
        payload = json.dumps({"result": result, "_tok_in": tin, "_tok_out": tout})
        redis_client.cache_set(cache_key, payload, ttl_s=ttl_s)
    except Exception as exc:
        print(f"[cache] SET error (non-fatal): {type(exc).__name__}: {str(exc)[:120]}")

    return result, tin, tout
