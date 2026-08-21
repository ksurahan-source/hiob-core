from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from hiob_core import llm_runtime as runtime


def completion(raw: str | None, tokens: tuple[int, int] | None = (3, 4)) -> SimpleNamespace:
    usage = None if tokens is None else SimpleNamespace(
        prompt_tokens=tokens[0], completion_tokens=tokens[1]
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=raw))], usage=usage
    )


def anthropic_response(
    raw: str,
    tokens: tuple[int, int] | None = (5, 6),
    *,
    include_noise: bool = False,
) -> SimpleNamespace:
    blocks = [SimpleNamespace(type="text", text=raw)]
    if include_noise:
        blocks.insert(0, SimpleNamespace(type="tool", text="ignored"))
    usage = None if tokens is None else SimpleNamespace(
        input_tokens=tokens[0], output_tokens=tokens[1]
    )
    return SimpleNamespace(content=blocks, usage=usage)


class Creator:
    def __init__(self, *responses: Any):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class ProviderClient:
    def __init__(self, *responses: Any):
        self.creator = Creator(*responses)
        self.chat = SimpleNamespace(completions=self.creator)
        self.messages = self.creator
        self.option_calls: list[dict[str, Any]] = []

    def with_options(self, **kwargs: Any) -> "ProviderClient":
        self.option_calls.append(kwargs)
        return self


def install_anthropic(monkeypatch: pytest.MonkeyPatch, client: ProviderClient) -> list[dict[str, Any]]:
    constructor_calls: list[dict[str, Any]] = []
    module = ModuleType("anthropic")

    def constructor(**kwargs: Any) -> ProviderClient:
        constructor_calls.append(kwargs)
        return client

    module.Anthropic = constructor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-real")
    return constructor_calls


def install_infra(monkeypatch: pytest.MonkeyPatch, redis: Any) -> None:
    module = ModuleType("infra")
    module.redis_client = redis  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "infra", module)


def test_prompt_resolution_loading_cache_and_locales(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prompt = tmp_path / "base.txt"
    prompt.write_text("  base prompt  ", encoding="utf-8")
    localized = tmp_path / "base_ko.txt"
    localized.write_text("한국어", encoding="utf-8")
    monkeypatch.setattr(runtime, "_PROMPT_DIR", tmp_path)
    runtime._PROMPT_CACHE.clear()

    assert runtime.load_prompt("base") == "base prompt"
    prompt.write_text("changed", encoding="utf-8")
    assert runtime.load_prompt("base") == "base prompt"
    assert runtime.load_localized_prompt("base", " KO ") == "한국어"
    assert runtime.load_localized_prompt("base", "fr") == "base prompt"
    assert runtime.load_localized_prompt("base", None) == "base prompt"

    monkeypatch.setenv("HIOB_PROMPT_DIR", str(tmp_path / "missing"))
    monkeypatch.setattr(runtime, "_is_usable_dir", lambda path: str(path) == "/root/prompts")
    assert runtime._resolve_prompt_dir() == Path("/root/prompts")
    monkeypatch.delenv("HIOB_PROMPT_DIR")
    monkeypatch.setattr(
        runtime,
        "_is_usable_dir",
        lambda path: str(path).endswith("apps/modal/prompts"),
    )
    assert str(runtime._resolve_prompt_dir()).endswith("apps/modal/prompts")


def test_model_resolution_costs_and_error_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIOB_DEFAULT_MODEL", "default-env")
    monkeypatch.setenv("HIOB_PREMIUM_MODEL", "premium-env")
    assert runtime.resolve_model(None) == ("default-env", "default")
    assert runtime.resolve_model({"attributes": {"model_override": 42}}) == ("42", "override")
    assert runtime.resolve_model({"default_model": "legacy-model"}) == ("legacy-model", "legacy")
    assert runtime.resolve_model({}) == ("default-env", "default")
    assert runtime.resolve_model({"model_tier": "premium"}) == ("premium-env", "premium")
    assert runtime.resolve_model({"model_tier": "unknown"}) == ("default-env", "unknown")
    assert runtime.estimate_cost_cents("gpt-4o", 1000, 1000) > 0
    assert runtime.estimate_cost_cents("unknown", -100, -100) == 0
    assert runtime.cost_line_item("", 0, None) == {
        "model": "",
        "tokens_in": 0,
        "tokens_out": 0,
        "est_cost": 0.0,
    }
    assert runtime._is_claude_model("Claude-Opus") is True
    assert runtime._is_claude_model(None) is False
    error = runtime.JsonRepairError("bad", tokens_in=None, tokens_out=2)
    assert (error.tokens_in, error.tokens_out) == (0, 2)


def test_json_parser_and_object_guard() -> None:
    assert runtime._parse_json_text('{"a": 1}') == {"a": 1}
    assert runtime._parse_json_text('{"a": "x\u0001y"}') == {"a": "x\x01y"}
    assert runtime._parse_json_text('prefix {"a": 2} suffix') == {"a": 2}
    assert runtime._parse_json_text('prefix {"a": "x\u0001y"} suffix') == {"a": "x\x01y"}
    with pytest.raises(json.JSONDecodeError):
        runtime._parse_json_text("not json")
    with pytest.raises(json.JSONDecodeError):
        runtime._parse_json_text("prefix {broken} suffix")
    assert runtime._require_json_object({"ok": True}, source="test") == {"ok": True}
    with pytest.raises(ValueError, match="not a JSON object"):
        runtime._require_json_object([], source="test")


def test_anthropic_repair_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "123")
    good = ProviderClient(anthropic_response('{"fixed": true}', None, include_noise=True))
    assert runtime._anthropic_repair_json_text(
        good, model="claude-opus", raw="broken"
    ) == ({"fixed": True}, 0, 0)
    assert good.creator.calls[0]["max_tokens"] == 123

    for bad_raw in ("broken", "[]"):
        bad = ProviderClient(anthropic_response(bad_raw, (2, 3)))
        with pytest.raises(runtime.JsonRepairError) as exc:
            runtime._anthropic_repair_json_text(bad, model="claude-opus", raw="bad")
        assert (exc.value.tokens_in, exc.value.tokens_out) == (2, 3)


def test_langfuse_initialization_and_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    original_langfuse = runtime._langfuse
    runtime._LANGFUSE_INIT_TRIED = False
    runtime._LANGFUSE_CLIENT = None
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert runtime._langfuse() is None
    assert runtime._langfuse() is None

    trace = SimpleNamespace(id="trace-new", generation=lambda **_kwargs: None)
    client = SimpleNamespace(trace=lambda **_kwargs: trace)
    module = ModuleType("langfuse")
    module.Langfuse = lambda **_kwargs: client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", module)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    runtime._LANGFUSE_INIT_TRIED = False
    runtime._LANGFUSE_CLIENT = None
    assert runtime._langfuse() is client

    monkeypatch.setattr(runtime, "_langfuse", lambda: client)
    assert runtime.langfuse_log(
        trace_id=None,
        role="writer",
        model="m",
        prompt="p" * 5000,
        response="r" * 5000,
        tokens_in=1,
        tokens_out=2,
    ) == "trace-new"
    monkeypatch.setattr(runtime, "_langfuse", lambda: None)
    assert runtime.langfuse_log(
        trace_id="old", role="r", model="m", prompt="", response="", tokens_in=0, tokens_out=0
    ) == "old"
    monkeypatch.setattr(
        runtime,
        "_langfuse",
        lambda: SimpleNamespace(trace=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("x"))),
    )
    assert runtime.langfuse_log(
        trace_id="old", role="r", model="m", prompt="", response="", tokens_in=0, tokens_out=0
    ) == "old"

    module.Langfuse = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("init"))  # type: ignore[attr-defined]
    monkeypatch.setattr(runtime, "_langfuse", original_langfuse)
    runtime._LANGFUSE_INIT_TRIED = False
    runtime._LANGFUSE_CLIENT = None
    assert runtime._langfuse() is None


def test_chat_helpers_provider_client_and_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert runtime._chat_messages("s", "u")[1]["content"] == "u"
    bare = runtime._chat_kwargs(system="s", user="u", model="m", temperature=None)
    assert "temperature" not in bare and "stream" not in bare and "extra_body" not in bare
    full = runtime._chat_kwargs(
        system="s",
        user="u",
        model="m",
        temperature=0.2,
        extra_body={"x": 1},
        stream=True,
    )
    assert full["temperature"] == 0.2 and full["stream_options"]["include_usage"] is True
    assert runtime._usage_tokens(None) == (0, 0)
    assert runtime._completion_result(completion(None, None)) == ({}, 0, 0)

    constructors: list[dict[str, Any]] = []
    fake = ProviderClient()
    monkeypatch.setenv("OPENAI_API_KEY", "not-real")
    monkeypatch.setattr(
        runtime,
        "OpenAI",
        lambda **kwargs: constructors.append(kwargs) or fake,
    )
    assert runtime._provider_client("gpt-4o", purpose="test") is fake
    assert runtime._provider_client("gpt-4o", purpose="test", base_url="https://example.test") is fake
    assert constructors == [
        {"api_key": "not-real"},
        {"api_key": "not-real", "base_url": "https://example.test"},
    ]


def test_gemini_qwen_and_openai_nonstream_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    gemini = ProviderClient(completion('{"gemini": true}', (1, 2)))
    monkeypatch.setattr(runtime, "_provider_client", lambda *_a, **_k: gemini)
    assert runtime._gemini_json(system="s", user="u", model="gemini-x", temperature=None) == (
        {"gemini": True}, 1, 2
    )

    qwen = ProviderClient(
        completion('{"qwen": true}', (3, 4)),
        completion('{"fixed": true}', (5, 6)),
    )
    monkeypatch.setattr(runtime, "_provider_client", lambda *_a, **_k: qwen)
    assert runtime._qwen_json(system="s", user="u", model="qwen-x", temperature=0.1) == (
        {"qwen": True}, 3, 4
    )
    qwen.creator.responses.insert(0, completion("broken", (3, 4)))
    assert runtime._qwen_json(system="s", user="u", model="qwen-x", temperature=None) == (
        {"fixed": True}, 8, 10
    )

    invalid = ProviderClient(completion("[]", (1, 2)))
    with pytest.raises(runtime.JsonRepairError) as exc:
        runtime._qwen_repair_result(
            invalid, model="qwen-x", raw="bad", tokens_in=10, tokens_out=20
        )
    assert (exc.value.tokens_in, exc.value.tokens_out) == (11, 22)

    openai = ProviderClient(completion('{"openai": true}', (7, 8)))
    monkeypatch.setattr(runtime, "_provider_client", lambda *_a, **_k: openai)
    assert runtime._openai_json(
        system="s", user="u", model="gpt-x", on_partial=None, temperature=0
    ) == ({"openai": True}, 7, 8)


def test_provider_routes_reject_non_object_json(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="not a JSON object"):
        runtime._completion_result(completion("[]"))

    qwen = ProviderClient(completion("[]"))
    monkeypatch.setattr(runtime, "_provider_client", lambda *_a, **_k: qwen)
    with pytest.raises(ValueError, match="not a JSON object"):
        runtime._qwen_json(system="s", user="u", model="qwen-x", temperature=None)

    stream = ProviderClient(
        [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="[]"))], usage=None)]
    )
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 1.0)
    with pytest.raises(ValueError, match="not a JSON object"):
        runtime._openai_stream_json(
            stream,
            system="s",
            user="u",
            model="gpt",
            on_partial=lambda _x: None,
            temperature=None,
        )


def test_openai_streaming_success_callback_failure_and_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content='{"a":'))], usage=None
        ),
        SimpleNamespace(choices=[], usage=None),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="1}"))],
            usage=SimpleNamespace(prompt_tokens=9, completion_tokens=10),
        ),
    ]
    client = ProviderClient(chunks)
    times = iter([1.0, 1.1])
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(times))
    partials: list[str] = []
    assert runtime._openai_stream_json(
        client, system="s", user="u", model="gpt", on_partial=partials.append, temperature=None
    ) == ({"a": 1}, 9, 10)
    assert partials == ['{"a":']

    failing_client = ProviderClient(
        [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="{}"))], usage=None)]
    )
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 2.0)

    def fail_callback(_text: str) -> None:
        raise RuntimeError("ignore")

    assert runtime._openai_stream_json(
        failing_client,
        system="s",
        user="u",
        model="gpt",
        on_partial=fail_callback,
        temperature=None,
    ) == ({}, 0, 0)
    empty_client = ProviderClient([])
    assert runtime._openai_stream_json(
        empty_client,
        system="s",
        user="u",
        model="gpt",
        on_partial=lambda _x: None,
        temperature=None,
    ) == ({}, 0, 0)


def test_openai_stream_dispatch_and_top_level_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ProviderClient([])
    monkeypatch.setattr(runtime, "_provider_client", lambda *_a, **_k: client)
    monkeypatch.setattr(runtime, "_openai_stream_json", lambda *_a, **_k: ({"stream": 1}, 1, 2))
    assert runtime._openai_json(
        system="s", user="u", model="gpt", on_partial=lambda _x: None, temperature=None
    ) == ({"stream": 1}, 1, 2)

    monkeypatch.setattr(runtime, "_anthropic_json", lambda **_k: ({"route": "a"}, 0, 0))
    monkeypatch.setattr(runtime, "_gemini_json", lambda **_k: ({"route": "g"}, 0, 0))
    monkeypatch.setattr(runtime, "_qwen_json", lambda **_k: ({"route": "q"}, 0, 0))
    monkeypatch.setattr(runtime, "_openai_json", lambda **_k: ({"route": "o"}, 0, 0))
    assert runtime.llm_json(system="s", user="u", model="claude-x")[0]["route"] == "a"
    assert runtime.llm_json(system="s", user="u", model="gemini-x")[0]["route"] == "g"
    assert runtime.llm_json(system="s", user="u", model="qwen-x")[0]["route"] == "q"
    assert runtime.llm_json(system="s", user="u", model="gpt-x")[0]["route"] == "o"


def test_vision_openai_qwen_and_claude_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "_anthropic_vision_json", lambda **_k: ({"claude": 1}, 1, 1))
    assert runtime.llm_vision_json(
        system="s", user="u", image_urls=["x"], model="claude-opus"
    )[0] == {"claude": 1}

    clients = [
        ProviderClient(completion('{"q": 1}', (2, 3))),
        ProviderClient(completion('{"o": 1}', None)),
    ]
    constructor_calls: list[dict[str, Any]] = []
    monkeypatch.setenv("DASHSCOPE_API_KEY", "q-key")
    monkeypatch.setenv("OPENAI_API_KEY", "o-key")

    def constructor(**kwargs: Any) -> ProviderClient:
        constructor_calls.append(kwargs)
        return clients.pop(0)

    monkeypatch.setattr(runtime, "OpenAI", constructor)
    urls = ["", *[f"u{i}" for i in range(10)]]
    assert runtime.llm_vision_json(system="s", user="u", image_urls=urls, model="qwen-x") == (
        {"q": 1}, 2, 3
    )
    assert runtime.llm_vision_json(system="s", user="u", image_urls=["u"], model="gpt-x") == (
        {"o": 1}, 0, 0
    )
    assert "base_url" in constructor_calls[0] and "base_url" not in constructor_calls[1]


def test_anthropic_vision_parse_repair_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    normal = ProviderClient(anthropic_response('{"seen": true}', None, include_noise=True))
    install_anthropic(monkeypatch, normal)
    assert runtime._anthropic_vision_json(
        system="s", user="u", image_urls=["one", "two"], model="claude-opus"
    ) == ({"seen": True}, 0, 0)
    assert normal.option_calls[0] == {"timeout": 300.0, "max_retries": 1}

    repaired = ProviderClient(
        anthropic_response("broken", (5, 6)), anthropic_response('{"fixed": 1}', (2, 3))
    )
    install_anthropic(monkeypatch, repaired)
    assert runtime._anthropic_vision_json(
        system="s", user="u", image_urls=[], model="claude-opus"
    ) == ({"fixed": 1}, 7, 9)

    failed = ProviderClient(
        anthropic_response("broken", (5, 6)), anthropic_response("still broken", (2, 3))
    )
    install_anthropic(monkeypatch, failed)
    with pytest.raises(runtime.JsonRepairError) as exc:
        runtime._anthropic_vision_json(
            system="s", user="u", image_urls=[], model="claude-opus"
        )
    assert (exc.value.tokens_in, exc.value.tokens_out) == (7, 9)


def test_claude_cli_discovery_model_fences_and_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shutil
    import subprocess

    monkeypatch.setattr(shutil, "which", lambda name: "/bin/claude" if name == "claude" else None)
    assert runtime._claude_cli_command() == ["/bin/claude"]
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/npx" if name == "npx" else None)
    assert runtime._claude_cli_command() == ["/bin/npx", "@anthropic-ai/claude-code"]
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert runtime._claude_cli_command() is None
    assert runtime._claude_cli_model("CLAUDE-OPUS-4") == "opus"
    assert runtime._claude_cli_model("sonnet-x") == "sonnet"
    assert runtime._claude_cli_model("haiku-x") == "haiku"
    assert runtime._claude_cli_model("custom") == "custom"
    assert runtime._strip_json_fence("```json\n{}\n```") == "{}"
    assert runtime._strip_json_fence("```\n{}\n```") == "{}"
    assert runtime._strip_json_fence(" {} ") == "{}"

    calls: list[Any] = []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")

    def success(command: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((command, kwargs))
        assert "ANTHROPIC_API_KEY" not in kwargs["env"]
        return SimpleNamespace(returncode=0, stdout='```json\n{"ok": 1}\n```')

    monkeypatch.setattr(subprocess, "run", success)
    result = runtime._run_claude_cli(["claude"], system="system", user="user", model="opus-x")
    assert result[0] == {"ok": 1} and result[1:] == (2, 2)
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=1, stdout="")
    )
    assert runtime._run_claude_cli(["claude"], system="s", user="u", model="m") is None
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")))
    assert runtime._run_claude_cli(["claude"], system="s", user="u", model="m") is None


def test_claude_cli_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIOB_LLM_CLI", raising=False)
    assert runtime._anthropic_cli_json(system="s", user="u", model="m") is None
    monkeypatch.setenv("HIOB_LLM_CLI", "YES")
    monkeypatch.setattr(runtime, "_claude_cli_command", lambda: None)
    assert runtime._anthropic_cli_json(system="s", user="u", model="m") is None
    monkeypatch.setattr(runtime, "_claude_cli_command", lambda: ["claude"])
    monkeypatch.setattr(runtime, "_run_claude_cli", lambda *_a, **_k: ({"ok": 1}, 1, 2))
    assert runtime._anthropic_cli_json(system="s", user="u", model="m") == ({"ok": 1}, 1, 2)


def test_anthropic_json_cli_api_repair_and_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_anthropic_cli_json", lambda **_k: ({"cli": 1}, 1, 2))
    assert runtime._anthropic_json(system="s", user="u", model="claude") == ({"cli": 1}, 1, 2)

    monkeypatch.setattr(runtime, "_anthropic_cli_json", lambda **_k: None)
    real_import = builtins.__import__

    def fail_anthropic(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "anthropic":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_anthropic)
    with pytest.raises(RuntimeError, match="anthropic package"):
        runtime._anthropic_json(system="s", user="u", model="claude")
    monkeypatch.setattr(builtins, "__import__", real_import)

    normal = ProviderClient(anthropic_response('{"api": 1}', None, include_noise=True))
    calls = install_anthropic(monkeypatch, normal)
    assert runtime._anthropic_json(
        system="s", user="u", model="claude-opus", temperature=0.3
    ) == ({"api": 1}, 0, 0)
    assert calls[0]["timeout"] == 300.0
    assert normal.creator.calls[0]["temperature"] == 0.3

    repaired = ProviderClient(
        anthropic_response("broken", (2, 3)), anthropic_response('{"fixed": 1}', (4, 5))
    )
    install_anthropic(monkeypatch, repaired)
    assert runtime._anthropic_json(
        system="s", user="u", model="claude-opus", temperature=None
    ) == ({"fixed": 1}, 6, 8)
    assert "temperature" not in repaired.creator.calls[0]

    failed = ProviderClient(
        anthropic_response("broken", (2, 3)), anthropic_response("bad", (4, 5))
    )
    install_anthropic(monkeypatch, failed)
    with pytest.raises(runtime.JsonRepairError) as exc:
        runtime._anthropic_json(system="s", user="u", model="claude-opus")
    assert (exc.value.tokens_in, exc.value.tokens_out) == (6, 8)


def test_llm_cache_hit_miss_and_fail_open(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hit = SimpleNamespace(
        cache_get=lambda _key: json.dumps({"result": {"hit": 1}, "_tok_in": 9}),
        cache_set=lambda *_a, **_k: None,
    )
    install_infra(monkeypatch, hit)
    assert runtime.llm_json_cached(system="s", user="u", model="m") == ({"hit": 1}, 0, 0)

    stored: list[tuple[Any, ...]] = []
    miss = SimpleNamespace(
        cache_get=lambda _key: None,
        cache_set=lambda *args, **kwargs: stored.append((*args, kwargs)),
    )
    install_infra(monkeypatch, miss)
    monkeypatch.setattr(runtime, "llm_json", lambda **_k: ({"fresh": 1}, 3, 4))
    assert runtime.llm_json_cached(
        system="s", user="u", model="m", ttl_s=7, temperature=0
    ) == ({"fresh": 1}, 3, 4)
    assert stored[0][-1] == {"ttl_s": 7}

    class BrokenRedis:
        def cache_get(self, _key: str) -> None:
            raise RuntimeError("get")

        def cache_set(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("set")

    install_infra(monkeypatch, BrokenRedis())
    assert runtime.llm_json_cached(system="s", user="u", model="m") == ({"fresh": 1}, 3, 4)
    output = capsys.readouterr().out
    assert "HIT" in output and "MISS" in output and "GET error" in output and "SET error" in output


def test_llm_cache_missing_infra_is_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "infra", raising=False)
    real_import = builtins.__import__

    def fail_infra(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "infra":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_infra)
    monkeypatch.setattr(runtime, "llm_json", lambda **_k: ({"direct": 1}, 1, 2))
    assert runtime.llm_json_cached(system="s", user="u", model="m") == ({"direct": 1}, 1, 2)
