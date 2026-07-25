from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from openai import OpenAI as RealOpenAI

from hiob_core import llm_runtime
from hiob_core.model_providers import script_model_id


def _completion_response(payload: str = '{"ok": true}') -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=payload),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )


class _FakeOpenAI:
    clients: list[dict] = []
    requests: list[dict] = []
    closed = 0

    def __init__(self, **kwargs):
        self.clients.append(kwargs)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )

    @classmethod
    def reset(cls) -> None:
        cls.clients = []
        cls.requests = []
        cls.closed = 0

    def close(self) -> None:
        type(self).closed += 1

    @classmethod
    def _create(cls, **kwargs):
        cls.requests.append(kwargs)
        return _completion_response()


@pytest.fixture
def fake_openai(monkeypatch):
    _FakeOpenAI.reset()
    monkeypatch.setattr(llm_runtime, "OpenAI", _FakeOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    return _FakeOpenAI


def _install_real_openai_mock_transport(monkeypatch, responder):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return responder(request)

    transport = httpx.MockTransport(handler)

    def openai_factory(**kwargs):
        return RealOpenAI(
            **kwargs,
            http_client=httpx.Client(transport=transport),
        )

    monkeypatch.setattr(llm_runtime, "OpenAI", openai_factory)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    return requests


def test_llm_json_passes_key_to_direct_openai_http_adapter(fake_openai):
    result, tokens_in, tokens_out = llm_runtime.llm_json(
        system="system",
        user="user",
        model="gpt-4o",
        idempotency_key="ares-xl-writer:run-1:round-1",
        max_completion_tokens=32000,
    )

    assert result == {"ok": True}
    assert (tokens_in, tokens_out) == (11, 7)
    assert fake_openai.clients == [
        {"api_key": "test-openai-key", "max_retries": 0}
    ]
    assert fake_openai.requests[0]["extra_headers"] == {
        "X-Client-Request-Id": "ares-xl-writer:run-1:round-1"
    }
    assert fake_openai.requests[0]["max_completion_tokens"] == 32000


def test_provider_preflight_constructs_and_closes_client_without_http(
    fake_openai,
):
    assert llm_runtime.preflight_llm_request(
        model="gpt-4o",
        purpose="llm_json",
        require_idempotency=True,
        max_completion_tokens=32000,
    ) is None
    assert fake_openai.clients == [
        {"api_key": "test-openai-key", "max_retries": 0}
    ]
    assert fake_openai.closed == 1
    assert fake_openai.requests == []


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("qwen3.7-max", "qwen"),
        ("gemini-2.0-flash", "gemini"),
        ("claude-sonnet-4-6", "anthropic"),
    ],
)
def test_provider_preflight_rejects_unsupported_idempotency_before_credentials(
    monkeypatch,
    model,
    provider,
):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(
        llm_runtime.ProviderIdempotencyUnsupportedError,
        match=f"{provider} model",
    ) as exc_info:
        llm_runtime.preflight_llm_request(
            model=model,
            purpose="llm_json",
            require_idempotency=True,
            max_completion_tokens=32000,
        )

    assert exc_info.value.provider_request_sent is False


def test_provider_preflight_marks_missing_credentials_as_pre_send(monkeypatch):
    from hiob_core.model_providers import ProviderCredentialUnavailableError

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ProviderCredentialUnavailableError) as exc_info:
        llm_runtime.preflight_llm_request(
            model="gpt-4o",
            purpose="llm_json",
            require_idempotency=True,
            max_completion_tokens=32000,
        )

    assert exc_info.value.provider_request_sent is False


def test_provider_preflight_marks_invalid_proxy_config_as_pre_send(
    monkeypatch,
):
    monkeypatch.setattr(llm_runtime, "OpenAI", RealOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("HTTP_PROXY", "://bad")

    with pytest.raises(
        llm_runtime.ProviderClientConfigurationError,
        match="client construction failed",
    ) as exc_info:
        llm_runtime.preflight_llm_request(
            model="gpt-4o",
            purpose="llm_json",
            require_idempotency=True,
            max_completion_tokens=32000,
        )

    assert exc_info.value.provider_request_sent is False


def test_direct_openai_key_reaches_actual_http_header(monkeypatch):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"ok": true}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            },
        )

    transport = httpx.MockTransport(handler)

    def openai_factory(**kwargs):
        return RealOpenAI(
            **kwargs,
            http_client=httpx.Client(transport=transport),
        )

    monkeypatch.setattr(llm_runtime, "OpenAI", openai_factory)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    result, _, _ = llm_runtime.llm_json(
        system="system",
        user="user",
        model="gpt-4o",
        idempotency_key="ares-xl-writer:run-http",
        max_completion_tokens=32000,
    )

    assert result == {"ok": True}
    assert len(requests) == 1
    assert requests[0].headers["x-client-request-id"] == (
        "ares-xl-writer:run-http"
    )
    assert requests[0].read()
    assert json.loads(requests[0].content)["max_completion_tokens"] == 32000


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_definitive_openai_rejection_is_non_unknown(
    monkeypatch,
    status_code,
):
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            request=request,
            json={
                "error": {
                    "message": "definitive request rejection",
                    "type": "invalid_request_error",
                }
            },
        )

    requests = _install_real_openai_mock_transport(
        monkeypatch,
        responder,
    )

    with pytest.raises(
        llm_runtime.ProviderRequestRejectedError,
        match=f"HTTP {status_code}",
    ) as exc_info:
        llm_runtime.llm_json(
            system="system",
            user="user",
            model="gpt-4o",
            idempotency_key=f"ares-xl-writer:status-{status_code}",
            max_completion_tokens=32000,
        )

    assert exc_info.value.provider_request_sent is False
    assert exc_info.value.status_code == status_code
    assert getattr(exc_info.value.__cause__, "status_code", None) == status_code
    assert len(requests) == 1


def test_unreserved_openai_rejection_preserves_sdk_error(monkeypatch):
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            request=request,
            json={
                "error": {
                    "message": "invalid request",
                    "type": "invalid_request_error",
                }
            },
        )

    requests = _install_real_openai_mock_transport(
        monkeypatch,
        responder,
    )

    with pytest.raises(Exception) as exc_info:
        llm_runtime.llm_json(
            system="system",
            user="user",
            model="gpt-4o",
        )

    assert not isinstance(
        exc_info.value,
        llm_runtime.ProviderRequestRejectedError,
    )
    assert getattr(exc_info.value, "status_code", None) == 400
    assert len(requests) == 1


@pytest.mark.parametrize("failure", ["timeout", "connection", "500"])
def test_ambiguous_openai_failure_remains_unknown(monkeypatch, failure):
    def responder(request: httpx.Request) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout("read timed out", request=request)
        if failure == "connection":
            raise httpx.ConnectError("connection lost", request=request)
        return httpx.Response(
            500,
            request=request,
            json={
                "error": {
                    "message": "provider internal error",
                    "type": "server_error",
                }
            },
        )

    requests = _install_real_openai_mock_transport(
        monkeypatch,
        responder,
    )

    with pytest.raises(Exception) as exc_info:
        llm_runtime.llm_json(
            system="system",
            user="user",
            model="gpt-4o",
            idempotency_key=f"ares-xl-writer:{failure}",
            max_completion_tokens=32000,
        )

    assert not isinstance(
        exc_info.value,
        llm_runtime.ProviderRequestRejectedError,
    )
    assert getattr(exc_info.value, "provider_request_sent", None) is not False
    assert len(requests) == 1


def test_definitive_openai_vision_rejection_is_non_unknown(monkeypatch):
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            request=request,
            json={
                "error": {
                    "message": "invalid image request",
                    "type": "invalid_request_error",
                }
            },
        )

    requests = _install_real_openai_mock_transport(
        monkeypatch,
        responder,
    )

    with pytest.raises(llm_runtime.ProviderRequestRejectedError) as exc_info:
        llm_runtime.llm_vision_json(
            system="system",
            user="inspect",
            image_urls=["https://example.test/proof.jpg"],
            model="gpt-4o",
            idempotency_key="ares-xl-writer:vision-422",
            max_completion_tokens=24000,
        )

    assert exc_info.value.provider_request_sent is False
    assert exc_info.value.status_code == 422
    assert len(requests) == 1


def test_llm_vision_json_passes_key_to_direct_openai_http_adapter(fake_openai):
    result, tokens_in, tokens_out = llm_runtime.llm_vision_json(
        system="system",
        user="inspect",
        image_urls=["https://example.test/proof.jpg"],
        model="gpt-4o",
        idempotency_key="ares-xl-writer:run-1:vision-1",
        max_completion_tokens=24000,
    )

    assert result == {"ok": True}
    assert (tokens_in, tokens_out) == (11, 7)
    assert fake_openai.clients == [
        {"api_key": "test-openai-key", "max_retries": 0}
    ]
    assert fake_openai.requests[0]["extra_headers"] == {
        "X-Client-Request-Id": "ares-xl-writer:run-1:vision-1"
    }
    assert fake_openai.requests[0]["max_completion_tokens"] == 24000


def test_llm_json_forwards_key_to_anthropic_adapter(monkeypatch):
    captured = {}

    def fake_anthropic_json(**kwargs):
        captured.update(kwargs)
        return {"ok": True}, 1, 2

    monkeypatch.setattr(llm_runtime, "_anthropic_json", fake_anthropic_json)

    assert llm_runtime.llm_json(
        system="system",
        user="user",
        model="claude-sonnet-4-6",
        idempotency_key="ares-xl-writer:run-2",
        max_completion_tokens=16000,
    ) == ({"ok": True}, 1, 2)
    assert captured["idempotency_key"] == "ares-xl-writer:run-2"
    assert captured["max_completion_tokens"] == 16000


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("qwen3.7-max", "qwen"),
        ("gemini-2.0-flash", "gemini"),
        ("claude-sonnet-4-6", "anthropic"),
    ],
)
def test_llm_json_fails_closed_for_unsupported_provider_key(model, provider):
    with pytest.raises(
        llm_runtime.ProviderIdempotencyUnsupportedError,
        match=f"{provider} model",
    ):
        llm_runtime.llm_json(
            system="system",
            user="user",
            model=model,
            idempotency_key="ares-xl-writer:run-3",
        )


@pytest.mark.parametrize(
    "model",
    ["qwen3.7-plus", "gemini-2.0-flash", "claude-sonnet-4-6"],
)
def test_llm_vision_json_fails_closed_for_unsupported_provider_key(model):
    with pytest.raises(llm_runtime.ProviderIdempotencyUnsupportedError):
        llm_runtime.llm_vision_json(
            system="system",
            user="user",
            image_urls=[],
            model=model,
            idempotency_key="ares-xl-writer:run-4",
            max_completion_tokens=32000,
        )


@pytest.mark.parametrize(
    "idempotency_key",
    ["", " surrounding-space ", "한글-key", "x" * 513],
)
def test_invalid_provider_idempotency_key_is_rejected(idempotency_key):
    with pytest.raises(ValueError):
        llm_runtime.llm_json(
            system="system",
            user="user",
            model="gpt-4o",
            idempotency_key=idempotency_key,
        )


def test_llm_json_without_key_preserves_existing_openai_retry_behavior(fake_openai):
    llm_runtime.llm_json(
        system="system",
        user="user",
        model="gpt-4o",
    )

    assert fake_openai.clients == [{"api_key": "test-openai-key"}]
    assert "extra_headers" not in fake_openai.requests[0]


@pytest.mark.parametrize(
    "base_url",
    [
        None,
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "https://workspace.example/compatible-mode/v1",
    ],
)
def test_reserved_openai_sdk_constructor_always_disables_retries(
    fake_openai,
    base_url,
):
    llm_runtime._openai_client(
        api_key="test-key",
        idempotency_key="reserved:constructor",
        base_url=base_url,
    )

    expected = {
        "api_key": "test-key",
        "max_retries": 0,
    }
    if base_url is not None:
        expected["base_url"] = base_url
    assert fake_openai.clients == [expected]


@pytest.mark.parametrize("vision", [False, True])
def test_qwen_maps_completion_limit_to_provider_max_tokens(
    fake_openai,
    monkeypatch,
    vision,
):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-qwen-key")

    if vision:
        llm_runtime.llm_vision_json(
            system="system",
            user="user",
            image_urls=[],
            model="qwen3.7-plus",
            max_completion_tokens=12000,
        )
    else:
        llm_runtime.llm_json(
            system="system",
            user="user",
            model="qwen3.7-max",
            max_completion_tokens=12000,
        )

    assert fake_openai.requests[0]["max_tokens"] == 12000
    assert "max_completion_tokens" not in fake_openai.requests[0]
    assert "max_retries" not in fake_openai.clients[0]


def test_mixed_case_qwen_uses_one_normalized_route_for_preflight_and_call(
    fake_openai,
    monkeypatch,
):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-qwen-key")
    monkeypatch.delenv("QWEN_OPENAI_BASE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")

    llm_runtime.preflight_llm_request(
        model=" Qwen3.7-Max ",
        purpose="llm_json",
        max_completion_tokens=12000,
    )
    result, _, _ = llm_runtime.llm_json(
        system="system",
        user="user",
        model=" Qwen3.7-Max ",
        max_completion_tokens=12000,
    )

    assert result == {"ok": True}
    assert fake_openai.clients == [
        {
            "api_key": "test-qwen-key",
            "base_url": (
                "https://ws-15myo7yelloeewav.ap-northeast-1.maas."
                "aliyuncs.com/compatible-mode/v1"
            ),
        },
        {
            "api_key": "test-qwen-key",
            "base_url": (
                "https://ws-15myo7yelloeewav.ap-northeast-1.maas."
                "aliyuncs.com/compatible-mode/v1"
            ),
        },
    ]
    assert fake_openai.closed == 1
    assert fake_openai.requests == [
        {
            "model": "qwen3.7-max",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            "response_format": {"type": "json_object"},
            "extra_body": {"enable_thinking": False},
            "max_tokens": 12000,
        }
    ]


def test_anthropic_maps_completion_limit_to_provider_max_tokens(
    monkeypatch,
):
    requests = []

    class FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text='{"ok": true}')],
                usage=SimpleNamespace(input_tokens=2, output_tokens=3),
            )

    anthropic_module = ModuleType("anthropic")
    anthropic_module.Anthropic = FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", anthropic_module)
    monkeypatch.delenv("HIOB_LLM_CLI", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    result, _, _ = llm_runtime.llm_json(
        system="system",
        user="user",
        model="claude-sonnet-4-6",
        max_completion_tokens=9000,
    )

    assert result == {"ok": True}
    assert requests[0]["max_tokens"] == 9000


@pytest.mark.parametrize(
    "call",
    [
        lambda: llm_runtime.llm_json(
            system="system",
            user="user",
            model="gemini-2.0-flash",
            max_completion_tokens=1000,
        ),
        lambda: llm_runtime.llm_vision_json(
            system="system",
            user="user",
            image_urls=[],
            model="gemini-2.0-flash",
            max_completion_tokens=1000,
        ),
    ],
)
def test_gemini_completion_limit_fails_before_openai_sdk_construction(
    fake_openai,
    call,
):
    with pytest.raises(llm_runtime.ProviderRequestOptionUnsupportedError):
        call()
    assert fake_openai.clients == []


@pytest.mark.parametrize("vision", [False, True])
def test_keyed_unsupported_sdk_provider_never_constructs_retrying_client(
    fake_openai,
    vision,
):
    with pytest.raises(llm_runtime.ProviderIdempotencyUnsupportedError):
        if vision:
            llm_runtime.llm_vision_json(
                system="system",
                user="user",
                image_urls=[],
                model="qwen3.7-plus",
                idempotency_key="reserved:qwen:vision",
                max_completion_tokens=1000,
            )
        else:
            llm_runtime.llm_json(
                system="system",
                user="user",
                model="qwen3.7-max",
                idempotency_key="reserved:qwen:text",
                max_completion_tokens=1000,
            )
    assert fake_openai.clients == []


@pytest.mark.parametrize(
    ("override", "provider"),
    [
        ("qwen3.7-max", "qwen"),
        ("claude-opus-4-8", "anthropic"),
        ("gemini-2.0-flash", "gemini"),
    ],
)
def test_sealed_ares_xl_unsupported_override_fails_before_http(
    fake_openai,
    monkeypatch,
    override,
    provider,
):
    monkeypatch.setenv("HIOB_ARES_XL_SCRIPT_MODEL", override)
    model = script_model_id(
        {"ares_xl_jkpa_authority": {"status": "sealed"}}
    )

    with pytest.raises(
        llm_runtime.ProviderIdempotencyUnsupportedError,
        match=f"{provider} model",
    ):
        llm_runtime.llm_json(
            system="system",
            user="user",
            model=model,
            idempotency_key="reserved:sealed-override",
            max_completion_tokens=32000,
        )
    assert fake_openai.clients == []


@pytest.mark.parametrize("max_completion_tokens", [0, -1, True, 1.5, "100"])
def test_invalid_max_completion_tokens_is_rejected(max_completion_tokens):
    expected = TypeError if not isinstance(max_completion_tokens, int) or isinstance(
        max_completion_tokens,
        bool,
    ) else ValueError
    with pytest.raises(expected):
        llm_runtime.llm_json(
            system="system",
            user="user",
            model="gpt-4o",
            max_completion_tokens=max_completion_tokens,
        )
