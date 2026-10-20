import copy
import json
import sys
from unittest import mock

import pytest
import requests

from mlflow.exceptions import MlflowException
from mlflow.genai.utils.gateway_utils import GatewayConfig
from mlflow.metrics.genai import model_utils
from mlflow.metrics.genai.model_utils import (
    _MODELS_WITHOUT_OUTPUT_CONFIG,
    _call_llm_provider_api,
    _parse_model_uri,
    _send_request,
    call_deployments_api,
    get_endpoint_type,
    score_model_on_payload,
)


@pytest.fixture
def set_envs(monkeypatch):
    monkeypatch.setenv("OPENAI_API_TYPE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test")


@pytest.fixture
def set_deployment_envs(monkeypatch):
    monkeypatch.setenv("MLFLOW_DEPLOYMENTS_TARGET", "databricks")


@pytest.fixture
def set_azure_envs(monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", "test")
    monkeypatch.setenv("AZURE_API_BASE", "https://openai-for.openai.azure.com/")
    monkeypatch.setenv("AZURE_API_VERSION", "2023-05-15")


@pytest.fixture(autouse=True)
def force_reload_openai():
    # Force reloading OpenAI module in the next test case. This is because they store
    # configuration like api_key, api_version, at the global variable, which is not
    # updated once set. Even if we reset the environment variable, it will retain the
    # old value and cause unexpected side effects.
    # https://github.com/openai/openai-python/blob/ea049cd0c42e115b90f1b9c7db80b2659a0bb92a/src/openai/__init__.py#L134
    sys.modules.pop("openai", None)


@pytest.mark.parametrize(
    ("model_uri", "expected_prefix", "expected_suffix"),
    [
        ("openai:/gpt-4o-mini", "openai", "gpt-4o-mini"),
        ("model:/123", "model", "123"),
        ("gateway:/my-route", "gateway", "my-route"),
        ("endpoints:/my-endpoint", "endpoints", "my-endpoint"),
        ("vertex_ai:/gemini-2.0", "vertex_ai", "gemini-2.0"),
        ("azure_ai:/gpt-4", "azure_ai", "gpt-4"),
    ],
)
def test_parse_model_uri(model_uri: str, expected_prefix: str, expected_suffix: str):
    prefix, suffix = _parse_model_uri(model_uri)
    assert prefix == expected_prefix
    assert suffix == expected_suffix


def test_parse_model_uri_throws_for_malformed():
    with pytest.raises(MlflowException, match="Malformed model uri"):
        _parse_model_uri("gpt-4o-mini")


def test_score_model_on_payload_throws_for_invalid():
    with pytest.raises(MlflowException, match="Unknown model uri prefix"):
        score_model_on_payload("myprovider:/gpt-4o-mini", "")


def test_score_model_openai_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MlflowException, match="OPENAI_API_KEY environment variable must be set"):
        score_model_on_payload("openai:/gpt-4o-mini", "")


_OAI_RESPONSE = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1677858242,
    "model": "gpt-4o-mini",
    "usage": {
        "prompt_tokens": 13,
        "completion_tokens": 7,
        "total_tokens": 20,
    },
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "\n\nThis is a test!",
            },
            "finish_reason": "stop",
            "index": 0,
        }
    ],
    "headers": {"Content-Type": "application/json"},
}


def test_score_model_openai(set_envs):
    with mock.patch(
        "mlflow.metrics.genai.model_utils._send_request", return_value=_OAI_RESPONSE
    ) as mock_post:
        resp = score_model_on_payload("openai:/gpt-4o-mini", "my prompt", {"temperature": 0.1})

        assert resp == "\n\nThis is a test!"
        mock_post.assert_called_once_with(
            endpoint="https://api.openai.com/v1/chat/completions",
            headers={"authorization": "Bearer test"},
            payload={
                "messages": [{"role": "user", "content": "my prompt"}],
                "model": "gpt-4o-mini",
                "temperature": 0.1,
            },
        )


def test_score_model_openai_with_custom_header_and_proxy_url(set_envs):
    with mock.patch(
        "mlflow.metrics.genai.model_utils._send_request", return_value=_OAI_RESPONSE
    ) as mock_post:
        resp = score_model_on_payload(
            model_uri="openai:/gpt-4o-mini",
            payload="my prompt",
            eval_parameters={"temperature": 0.1},
            extra_headers={"foo": "bar"},
            proxy_url="https://my-proxy.com/chat",
        )

        assert resp == "\n\nThis is a test!"
        mock_post.assert_called_once_with(
            endpoint="https://my-proxy.com/chat",
            headers={"authorization": "Bearer test", "foo": "bar"},
            payload={
                "messages": [{"role": "user", "content": "my prompt"}],
                "model": "gpt-4o-mini",
                "temperature": 0.1,
            },
        )


def test_score_model_openai_honors_openai_base_url(set_envs, monkeypatch):
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://my-host/serving-endpoints/v1")
    with mock.patch(
        "mlflow.metrics.genai.model_utils._send_request", return_value=_OAI_RESPONSE
    ) as mock_post:
        score_model_on_payload("openai:/gpt-4o-mini", "my prompt", {"temperature": 0.1})

        mock_post.assert_called_once_with(
            endpoint="https://my-host/serving-endpoints/v1/chat/completions",
            headers={"authorization": "Bearer test"},
            payload={
                "messages": [{"role": "user", "content": "my prompt"}],
                "model": "gpt-4o-mini",
                "temperature": 0.1,
            },
        )


def test_score_model_openai_api_base_takes_precedence_over_base_url(set_envs, monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "https://api-base/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://base-url/v1")
    with mock.patch(
        "mlflow.metrics.genai.model_utils._send_request", return_value=_OAI_RESPONSE
    ) as mock_post:
        score_model_on_payload("openai:/gpt-4o-mini", "my prompt", {"temperature": 0.1})

        mock_post.assert_called_once_with(
            endpoint="https://api-base/v1/chat/completions",
            headers={"authorization": "Bearer test"},
            payload={
                "messages": [{"role": "user", "content": "my prompt"}],
                "model": "gpt-4o-mini",
                "temperature": 0.1,
            },
        )


def test_openai_other_error(set_envs):
    with mock.patch(
        "mlflow.metrics.genai.model_utils._send_request",
        side_effect=Exception("foo"),
    ):
        with pytest.raises(Exception, match="foo"):
            score_model_on_payload("openai:/gpt-4o-mini", "my prompt", {"temperature": 0.1})


def test_score_model_azure_openai(set_azure_envs):
    with mock.patch(
        "mlflow.metrics.genai.model_utils._send_request", return_value=_OAI_RESPONSE
    ) as mock_post:
        resp = score_model_on_payload("azure:/test-openai", "my prompt", {"temperature": 0.1})

        assert resp == "\n\nThis is a test!"
        mock_post.assert_called_once_with(
            endpoint="https://openai-for.openai.azure.com/openai/deployments/test-openai/chat/completions?api-version=2023-05-15",
            headers={"api-key": "test"},
            payload={
                "messages": [{"role": "user", "content": "my prompt"}],
                "temperature": 0.1,
            },
        )


def test_score_model_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    resp = {
        "content": [
            {
                "text": "This is a test!",
                "type": "text",
            }
        ],
        "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
        "model": "claude-3-5-sonnet-20241022",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
        "usage": {"input_tokens": 2095, "output_tokens": 503},
    }

    with mock.patch(
        "mlflow.metrics.genai.model_utils._send_request", return_value=resp
    ) as mock_request:
        response = score_model_on_payload(
            model_uri="anthropic:/claude-3-5-sonnet-20241022",
            payload="input prompt",
            eval_parameters={"max_tokens": 1000, "top_p": 1},
            extra_headers={"anthropic-version": "2024-10-22"},
        )

    assert response == "This is a test!"
    mock_request.assert_called_once_with(
        endpoint="https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2024-10-22",
        },
        payload={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "input prompt"}],
            "max_tokens": 1000,
            "top_p": 1,
        },
    )


def _bedrock_converse_response(text="This is a test!"):
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 11, "outputTokens": 5, "totalTokens": 16},
    }


def test_score_model_bedrock(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "test-session-token")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

    mock_bedrock = mock.MagicMock()
    with mock.patch("boto3.Session.client", return_value=mock_bedrock) as mock_session:
        mock_bedrock.converse.return_value = _bedrock_converse_response()

        response = score_model_on_payload(
            model_uri="bedrock:/anthropic.claude-3-5-sonnet-20241022-v2:0",
            payload="input prompt",
            eval_parameters={
                "temperature": 0,
                "max_tokens": 1000,
                "anthropic_version": "2023-06-01",
            },
        )

    assert response == "This is a test!"
    mock_session.assert_called_once_with(
        service_name="bedrock-runtime",
        aws_access_key_id="test-access-key",
        aws_secret_access_key="test-secret-key",
        aws_session_token="test-session-token",
    )
    # Converse takes inference parameters in inferenceConfig. Invoke-only fields
    # such as anthropic_version are not part of the Converse request.
    mock_bedrock.converse.assert_called_once_with(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": [{"text": "input prompt"}]}],
        inferenceConfig={"maxTokens": 1000, "temperature": 0},
    )
    mock_bedrock.invoke_model.assert_not_called()


def test_score_model_bedrock_nova_uses_converse(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

    mock_bedrock = mock.MagicMock()
    mock_bedrock.converse.return_value = _bedrock_converse_response("nova reply")
    with mock.patch("boto3.Session.client", return_value=mock_bedrock) as mock_session:
        response = _call_llm_provider_api(
            "bedrock",
            "amazon.nova-lite-v1:0",
            messages=[
                {"role": "system", "content": "You are a judge."},
                {"role": "user", "content": "Grade this answer."},
            ],
            eval_parameters={"temperature": 0, "top_p": 0.9, "stop": ["END"]},
        )

    assert response == "nova reply"
    mock_session.assert_called_once()
    mock_bedrock.converse.assert_called_once_with(
        modelId="amazon.nova-lite-v1:0",
        system=[{"text": "You are a judge."}],
        messages=[{"role": "user", "content": [{"text": "Grade this answer."}]}],
        inferenceConfig={"temperature": 0, "topP": 0.9, "stopSequences": ["END"]},
    )
    mock_bedrock.invoke_model.assert_not_called()


def test_score_model_bedrock_retries_without_structured_output(monkeypatch):
    from botocore.exceptions import ClientError

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

    error = ClientError(
        {
            "Error": {
                "Code": "ValidationException",
                "Message": "This model does not support outputConfig",
            },
            "ResponseMetadata": {"HTTPStatusCode": 400},
        },
        "Converse",
    )
    mock_bedrock = mock.MagicMock()
    mock_bedrock.converse.side_effect = [error, _bedrock_converse_response('{"result": "yes"}')]
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "JudgeEvaluation",
            "schema": {"type": "object", "properties": {"result": {"type": "string"}}},
        },
    }
    with mock.patch("boto3.Session.client", return_value=mock_bedrock):
        response = _call_llm_provider_api(
            "bedrock",
            "amazon.nova-micro-v1:0",
            messages=[{"role": "user", "content": "Grade this."}],
            response_format=response_format,
        )

    assert response == '{"result": "yes"}'
    assert mock_bedrock.converse.call_count == 2
    first_kwargs = mock_bedrock.converse.call_args_list[0].kwargs
    second_kwargs = mock_bedrock.converse.call_args_list[1].kwargs
    assert first_kwargs["outputConfig"]["textFormat"]["type"] == "json_schema"
    schema = json.loads(
        first_kwargs["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"]
    )
    assert schema["properties"]["result"]["type"] == "string"
    assert first_kwargs["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["name"] == (
        "JudgeEvaluation"
    )
    assert "outputConfig" not in second_kwargs


def test_score_model_bedrock_bearer_token_uses_http(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    http_response = mock.Mock(status_code=200)
    http_response.json.return_value = _bedrock_converse_response("bearer reply")
    with (
        mock.patch("requests.post", return_value=http_response) as mock_post,
        mock.patch("boto3.Session.client") as mock_client,
    ):
        response = score_model_on_payload(
            model_uri="bedrock:/amazon.nova-micro-v1:0",
            payload="hello",
        )

    assert response == "bearer reply"
    mock_client.assert_not_called()
    mock_post.assert_called_once()
    assert mock_post.call_args.args[0] == (
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.nova-micro-v1:0/converse"
    )
    assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer bedrock-api-key"
    assert mock_post.call_args.kwargs["json"]["messages"] == [
        {"role": "user", "content": [{"text": "hello"}]}
    ]


def test_score_model_bedrock_access_denied(monkeypatch):
    from botocore.exceptions import ClientError

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

    error = ClientError(
        {
            "Error": {"Code": "AccessDeniedException", "Message": "not authorized"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        },
        "Converse",
    )
    mock_bedrock = mock.MagicMock()
    mock_bedrock.converse.side_effect = error
    with mock.patch("boto3.Session.client", return_value=mock_bedrock):
        with pytest.raises(MlflowException, match="AccessDeniedException: not authorized"):
            score_model_on_payload(
                model_uri="bedrock:/meta.llama3-8b-instruct-v1:0",
                payload="hello",
            )

    mock_bedrock.converse.assert_called_once()


def test_score_model_mistral(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    # Mistral AI API is compatible with OpenAI format
    with mock.patch(
        "mlflow.metrics.genai.model_utils._send_request", return_value=_OAI_RESPONSE
    ) as mock_request:
        response = score_model_on_payload(
            model_uri="mistral:/mistral-small-latest",
            payload="input prompt",
            eval_parameters={"temperature": 0.1},
        )

    assert response == "\n\nThis is a test!"
    mock_request.assert_called_once_with(
        endpoint="https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        payload={
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "input prompt"}],
            "temperature": 0.1,
        },
    )


def test_score_model_togetherai(monkeypatch):
    monkeypatch.setenv("TOGETHERAI_API_KEY", "test-key")

    resp = {
        "id": "8448080b880415ea-SJC",
        "choices": [{"message": {"role": "assistant", "content": "This is a test!"}}],
        "usage": {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        "created": 1705090115,
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "object": "chat.completion",
    }

    with mock.patch(
        "mlflow.metrics.genai.model_utils._send_request", return_value=resp
    ) as mock_request:
        response = score_model_on_payload(
            model_uri="togetherai:/mistralai/Mixtral-8x7B-Instruct-v0.1",
            payload="input prompt",
            eval_parameters={"temperature": 0, "max_tokens": 1000},
        )

    assert response == "This is a test!"
    mock_request.assert_called_once_with(
        endpoint="https://api.together.xyz/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        payload={
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "messages": [{"role": "user", "content": "input prompt"}],
            "temperature": 0,
            "max_tokens": 1000,
        },
    )


def test_score_model_gateway_completions():
    gw_config = GatewayConfig(
        api_base="http://localhost:5000/gateway/mlflow/v1/",
        endpoint_name="my-route",
        extra_headers=None,
    )

    with (
        mock.patch(
            "mlflow.metrics.genai.model_utils.get_gateway_config", return_value=gw_config
        ) as mock_get_config,
        mock.patch(
            "mlflow.metrics.genai.model_utils._send_request", return_value=_OAI_RESPONSE
        ) as mock_send,
    ):
        response = score_model_on_payload("gateway:/my-route", "my prompt")
        assert response == "\n\nThis is a test!"
        mock_get_config.assert_called_once_with("my-route")
        mock_send.assert_called_once_with(
            endpoint="http://localhost:5000/gateway/mlflow/v1/chat/completions",
            headers={},
            payload={"model": "my-route", "messages": [{"role": "user", "content": "my prompt"}]},
        )


@pytest.mark.parametrize(
    ("get_endpoint_response", "expected"),
    [
        ({"task": "llm/v1/completions"}, "llm/v1/completions"),
        ({"endpoint_type": "llm/v1/chat"}, "llm/v1/chat"),
        ({}, None),
    ],
)
def test_get_endpoint_type(get_endpoint_response, expected):
    with mock.patch("mlflow.deployments.get_deploy_client") as mock_get_deploy_client:
        mock_client = mock_get_deploy_client.return_value
        mock_client.get_endpoint.return_value = get_endpoint_response
        assert get_endpoint_type("endpoints:/my-endpoint") == expected


_TEST_CHAT_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4o-mini",
    "system_fingerprint": "fp_44709d6fcb",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "\n\nHello there, how may I assist you today?",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
}


def test_score_model_endpoints_chat(set_deployment_envs):
    with mock.patch("mlflow.deployments.get_deploy_client") as mock_get_deploy_client:
        mock_get_deploy_client().predict.return_value = _TEST_CHAT_RESPONSE
        response = score_model_on_payload(
            model_uri="endpoints:/my-endpoint",
            payload="my prompt",
            eval_parameters={"temperature": 0.1},
            endpoint_type="llm/v1/chat",
        )
        assert response == "\n\nHello there, how may I assist you today?"


_TEST_COMPLETION_RESPONSE = {
    "id": "cmpl-8PgdiXapPWBN3pyUuHcELH766QgqK",
    "object": "text_completion",
    "created": 1701132798,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "text": "\n\nHi there! How can I assist you today?",
            "index": 0,
            "finish_reason": "stop",
        },
    ],
    "usage": {"prompt_tokens": 2, "completion_tokens": 106, "total_tokens": 108},
}


def test_score_model_endpoints_completions(set_deployment_envs):
    with mock.patch("mlflow.deployments.get_deploy_client") as mock_get_deploy_client:
        mock_get_deploy_client().predict.return_value = _TEST_COMPLETION_RESPONSE
        response = score_model_on_payload(
            model_uri="endpoints:/my-endpoint",
            payload="my prompt",
            eval_parameters={"temperature": 0.1},
            endpoint_type="llm/v1/completions",
        )
        assert response == "\n\nHi there! How can I assist you today?"


@pytest.mark.parametrize(
    "input_data",
    [
        "my prompt",
        {"messages": [{"role": "user", "content": "my prompt"}]},
    ],
)
def test_call_deployments_api_chat(input_data, set_deployment_envs):
    with mock.patch("mlflow.deployments.get_deploy_client") as mock_get_deploy_client:
        mock_get_deploy_client().predict.return_value = _TEST_CHAT_RESPONSE
        response = call_deployments_api(
            deployment_uri="my-endpoint",
            input_data=input_data,
            eval_parameters={},
            endpoint_type="llm/v1/chat",
        )
        assert response == "\n\nHello there, how may I assist you today?"


@pytest.mark.parametrize(
    "input_data",
    [
        "my prompt",
        {"prompt": "my prompt"},
    ],
)
def test_call_deployments_api_completion(input_data, set_deployment_envs):
    with mock.patch("mlflow.deployments.get_deploy_client") as mock_get_deploy_client:
        mock_get_deploy_client().predict.return_value = _TEST_COMPLETION_RESPONSE
        response = call_deployments_api(
            deployment_uri="my-endpoint",
            input_data=input_data,
            eval_parameters={"temperature": 0.1},
            endpoint_type="llm/v1/completions",
        )
        assert response == "\n\nHi there! How can I assist you today?"


def test_call_deployments_api_no_endpoint_type(set_deployment_envs):
    with mock.patch("mlflow.deployments.get_deploy_client") as mock_get_deploy_client:
        mock_get_deploy_client().predict.return_value = {"result": "ok"}
        response = call_deployments_api(
            deployment_uri="my-endpoint",
            input_data={"foo": {"bar": "baz"}},
            eval_parameters={},
            endpoint_type=None,
        )
        assert response == {"result": "ok"}


def test_call_deployments_api_str_input_requires_endpoint_type(set_deployment_envs):
    with pytest.raises(MlflowException, match="If string input is provided,"):
        call_deployments_api("my-endpoint", "my prompt", endpoint_type=None)


def test_send_request_includes_response_body_in_error():
    resp = requests.Response()
    resp.status_code = 400
    resp._content = b'{"error": "bad request details"}'

    with mock.patch("requests.post", return_value=resp):
        with pytest.raises(MlflowException, match="bad request details") as exc_info:
            _send_request("http://example.com", {}, {})

        # Verify exception chaining preserves the original HTTPError
        assert isinstance(exc_info.value.__cause__, requests.exceptions.HTTPError)


def test_score_model_retries_without_output_config_on_unsupported(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    anthropic_resp = {
        "content": [{"text": "result text", "type": "text"}],
        "id": "msg_test",
        "model": "claude-sonnet-4-20250514",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    # First call raises 400 with "does not support output format", second call succeeds
    error_body = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "claude-sonnet-4-20250514 does not support output format",
        },
    }
    mock_400_response = requests.Response()
    mock_400_response.status_code = 400
    mock_400_response._content = json.dumps(error_body).encode()

    mock_ok_response = mock.MagicMock()
    mock_ok_response.status_code = 200
    mock_ok_response.raise_for_status.return_value = None
    mock_ok_response.json.return_value = anthropic_resp

    # Capture payloads before they are mutated in-place by the retry logic
    captured_payloads = []
    original_send = None

    def capture_send(endpoint, headers, payload):
        captured_payloads.append(copy.deepcopy(payload))
        return original_send(endpoint=endpoint, headers=headers, payload=payload)

    original_send = model_utils._send_request

    with (
        mock.patch("requests.post", side_effect=[mock_400_response, mock_ok_response]),
        mock.patch("mlflow.metrics.genai.model_utils._send_request", side_effect=capture_send),
    ):
        response = score_model_on_payload(
            model_uri="anthropic:/claude-sonnet-4-20250514",
            payload="test prompt",
            eval_parameters={
                "max_tokens": 100,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "result",
                        "schema": {"type": "object", "properties": {"x": {"type": "string"}}},
                    },
                },
            },
        )

    assert response == "result text"
    # Verify two calls were made: first with output_config, second without
    assert len(captured_payloads) == 2
    assert "output_config" in captured_payloads[0]
    assert "output_config" not in captured_payloads[1]
    assert "response_format" not in captured_payloads[1]


def test_score_model_caches_unsupported_output_config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    model_name = "claude-sonnet-4-20250514-cache-test"
    _MODELS_WITHOUT_OUTPUT_CONFIG.discard(("anthropic", model_name))

    anthropic_resp = {
        "content": [{"text": "result", "type": "text"}],
        "id": "msg_test",
        "model": model_name,
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    error_body = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": f"{model_name} does not support output format",
        },
    }
    mock_400_response = requests.Response()
    mock_400_response.status_code = 400
    mock_400_response._content = json.dumps(error_body).encode()

    mock_ok_response = mock.MagicMock()
    mock_ok_response.status_code = 200
    mock_ok_response.raise_for_status.return_value = None
    mock_ok_response.json.return_value = anthropic_resp

    eval_params = {
        "max_tokens": 100,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "result",
                "schema": {"type": "object", "properties": {"x": {"type": "string"}}},
            },
        },
    }

    # First call: triggers retry (2 requests.post calls)
    with mock.patch("requests.post", side_effect=[mock_400_response, mock_ok_response]) as m:
        score_model_on_payload(
            model_uri=f"anthropic:/{model_name}",
            payload="test prompt",
            eval_parameters=eval_params,
        )
        assert m.call_count == 2

    assert ("anthropic", model_name) in _MODELS_WITHOUT_OUTPUT_CONFIG

    # Second call: skips output_config upfront (only 1 requests.post call)
    with mock.patch("requests.post", return_value=mock_ok_response) as m:
        score_model_on_payload(
            model_uri=f"anthropic:/{model_name}",
            payload="test prompt",
            eval_parameters=eval_params,
        )
        assert m.call_count == 1

    _MODELS_WITHOUT_OUTPUT_CONFIG.discard(("anthropic", model_name))


@pytest.mark.parametrize(
    ("provider", "env_var", "api_key", "expected_endpoint"),
    [
        ("groq", "GROQ_API_KEY", "groq-key", "https://api.groq.com/openai/v1/chat/completions"),
        (
            "deepseek",
            "DEEPSEEK_API_KEY",
            "ds-key",
            "https://api.deepseek.com/v1/chat/completions",
        ),
        ("xai", "XAI_API_KEY", "xai-key", "https://api.x.ai/v1/chat/completions"),
        (
            "openrouter",
            "OPENROUTER_API_KEY",
            "or-key",
            "https://openrouter.ai/api/v1/chat/completions",
        ),
    ],
)
def test_score_model_openai_compatible_providers(
    monkeypatch, provider, env_var, api_key, expected_endpoint
):
    monkeypatch.setenv(env_var, api_key)

    with mock.patch(
        "mlflow.metrics.genai.model_utils._send_request", return_value=_OAI_RESPONSE
    ) as mock_request:
        response = score_model_on_payload(
            model_uri=f"{provider}:/some-model",
            payload="input prompt",
        )

    assert response == "\n\nThis is a test!"
    mock_request.assert_called_once_with(
        endpoint=expected_endpoint,
        headers={"Authorization": f"Bearer {api_key}"},
        payload={
            "model": "some-model",
            "messages": [{"role": "user", "content": "input prompt"}],
        },
    )


def test_score_model_ollama(monkeypatch):
    with mock.patch(
        "mlflow.metrics.genai.model_utils._send_request", return_value=_OAI_RESPONSE
    ) as mock_request:
        response = score_model_on_payload(
            model_uri="ollama:/llama3",
            payload="input prompt",
        )

    assert response == "\n\nThis is a test!"
    # Ollama runs locally; no auth header is sent when using the default "ollama" key
    mock_request.assert_called_once_with(
        endpoint="http://localhost:11434/v1/chat/completions",
        headers={},
        payload={
            "model": "llama3",
            "messages": [{"role": "user", "content": "input prompt"}],
        },
    )


def test_score_model_databricks(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://my-workspace.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-test-token")

    with mock.patch(
        "mlflow.metrics.genai.model_utils._send_request", return_value=_OAI_RESPONSE
    ) as mock_request:
        response = score_model_on_payload(
            model_uri="databricks:/databricks-meta-llama-3-3-70b-instruct",
            payload="input prompt",
        )

    assert response == "\n\nThis is a test!"
    call_kwargs = mock_request.call_args[1]
    assert (
        call_kwargs["endpoint"]
        == "https://my-workspace.databricks.com/serving-endpoints/chat/completions"
    )


def test_score_model_vertex_ai(monkeypatch):
    monkeypatch.setenv("VERTEX_PROJECT", "my-gcp-project")
    monkeypatch.setenv("VERTEX_LOCATION", "us-central1")

    # VertexAI response uses Gemini format (content list), not OpenAI format
    vertex_resp = {
        "candidates": [
            {
                "content": {"parts": [{"text": "\n\nThis is a test!"}], "role": "model"},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 7},
    }

    mock_token = mock.MagicMock()
    mock_token.token = "fake-gcp-token"
    mock_token.valid = True

    with (
        mock.patch(
            "mlflow.gateway.providers.vertex_ai.VertexAIProvider._get_credentials",
            return_value=mock_token,
        ),
        mock.patch(
            "mlflow.metrics.genai.model_utils._send_request", return_value=vertex_resp
        ) as mock_request,
    ):
        response = score_model_on_payload(
            model_uri="vertex_ai:/gemini-2.0-flash",
            payload="input prompt",
        )

    assert response == "\n\nThis is a test!"
    call_kwargs = mock_request.call_args[1]
    assert "my-gcp-project" in call_kwargs["endpoint"]
    assert "gemini-2.0-flash" in call_kwargs["endpoint"]


def test_score_model_does_not_retry_on_other_400_errors(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    error_body = {
        "type": "error",
        "error": {"type": "authentication_error", "message": "invalid api key"},
    }
    mock_400_response = requests.Response()
    mock_400_response.status_code = 400
    mock_400_response._content = json.dumps(error_body).encode()

    with mock.patch("requests.post", return_value=mock_400_response):
        with pytest.raises(MlflowException, match="invalid api key"):
            score_model_on_payload(
                model_uri="anthropic:/claude-sonnet-4-20250514",
                payload="test prompt",
                eval_parameters={
                    "max_tokens": 100,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "result",
                            "schema": {"type": "object", "properties": {"x": {"type": "string"}}},
                        },
                    },
                },
            )


def test_send_request_uses_timeout_from_env_var(monkeypatch):
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_LLM_TIMEOUT", "2")

    with mock.patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {}
        mock_post.return_value.raise_for_status.return_value = None

        _send_request("", {}, {})

        _, kwargs = mock_post.call_args
        assert kwargs["timeout"] == 2
