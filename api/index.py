import asyncio
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gemini_webapi import GeminiClient  # noqa: E402
from gemini_webapi.exceptions import (  # noqa: E402
    APIError,
    AuthError,
    GeminiError,
    TemporarilyBlocked,
    TimeoutError,
    UsageLimitExceeded,
)


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt for Gemini")
    model: str | None = Field(default=None, description="Optional model name")
    temporary: bool = Field(default=False)
    deep_research: bool = Field(default=False)


class OpenAIChatMessage(BaseModel):
    role: str
    content: Any
    name: str | None = None


class OpenAIChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[OpenAIChatMessage] = Field(..., min_length=1)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None

    model_config = ConfigDict(extra="allow")


def _jsonify(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    return value


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    raise HTTPException(
        status_code=500, detail=f"Missing required environment variable: {name}"
    )


def _openai_error(
    status_code: int,
    message: str,
    error_type: str = "invalid_request_error",
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": code,
            }
        },
    )


def _resolve_openai_api_key() -> str | None:
    return os.getenv("OPENAI_COMPAT_API_KEY") or os.getenv("OPENAI_API_KEY")


def _extract_bearer_token(request: Request) -> str | None:
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return None


def _check_openai_auth(request: Request) -> JSONResponse | None:
    expected_key = _resolve_openai_api_key()
    if not expected_key:
        return None

    provided_key = _extract_bearer_token(request) or request.headers.get("x-api-key")
    if provided_key == expected_key:
        return None

    return _openai_error(
        status_code=401,
        message="Invalid API key provided.",
        error_type="authentication_error",
        code="invalid_api_key",
    )


def _normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue

            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
        return "\n".join(parts).strip()

    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            return text_value

    return ""


def _messages_to_prompt(messages: list[OpenAIChatMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        role = (message.role or "user").strip().lower()
        content = _normalize_message_content(message.content).strip()
        if not content:
            continue
        lines.append(f"{role.upper()}: {content}")

    if not lines:
        raise ValueError("messages must contain at least one non-empty text content")

    # A trailing assistant role hint improves compatibility with chat-style prompts.
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Lightweight approximation for OpenAI-compatible usage fields.
    return max(1, len(text) // 4)


app = FastAPI(title="Gemini-API Serverless", version="1.0.0")
_client: GeminiClient | None = None
_client_lock = asyncio.Lock()


async def _get_client() -> GeminiClient:
    global _client

    if _client is not None:
        return _client

    async with _client_lock:
        if _client is not None:
            return _client

        secure_1psid = _required_env("GEMINI_SECURE_1PSID")
        secure_1psidts = os.getenv("GEMINI_SECURE_1PSIDTS", "")
        proxy = os.getenv("GEMINI_PROXY")
        timeout = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "180"))

        client = GeminiClient(
            secure_1psid=secure_1psid,
            secure_1psidts=secure_1psidts,
            proxy=proxy,
        )

        await client.init(
            timeout=timeout,
            auto_refresh=False,
            auto_close=True,
            close_delay=120,
        )
        _client = client
        return _client


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "name": "Gemini-API Serverless",
        "status": "ok",
        "endpoints": [
            "GET /health",
            "GET /models",
            "POST /chat",
            "GET /v1/models",
            "POST /v1/chat/completions",
        ],
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "has_secure_1psid": bool(os.getenv("GEMINI_SECURE_1PSID")),
        "has_secure_1psidts": bool(os.getenv("GEMINI_SECURE_1PSIDTS")),
        "client_initialized": _client is not None,
    }


@app.get("/models")
async def models() -> dict[str, Any]:
    try:
        client = await _get_client()
        available = client.list_models() or []
        return {"models": _jsonify(available)}
    except AuthError as exc:
        raise HTTPException(
            status_code=401, detail=f"Authentication failed: {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat")
async def chat(req: ChatRequest) -> dict[str, Any]:
    try:
        client = await _get_client()
        model = req.model or os.getenv("GEMINI_DEFAULT_MODEL")

        request_kwargs: dict[str, Any] = {
            "prompt": req.prompt,
            "temporary": req.temporary,
            "deep_research": req.deep_research,
        }
        if model:
            request_kwargs["model"] = model

        output = await client.generate_content(**request_kwargs)

        return {
            "text": output.text,
            "metadata": output.metadata,
            "thoughts": output.thoughts,
            "images": _jsonify(output.images),
            "videos": _jsonify(output.videos),
            "media": _jsonify(output.media),
            "candidates": _jsonify(output.candidates),
        }
    except AuthError as exc:
        raise HTTPException(
            status_code=401, detail=f"Authentication failed: {exc}"
        ) from exc
    except (UsageLimitExceeded, TemporarilyBlocked) as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except (GeminiError, APIError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v1/models")
async def openai_models(request: Request) -> JSONResponse:
    auth_error = _check_openai_auth(request)
    if auth_error:
        return auth_error

    try:
        client = await _get_client()
        available = client.list_models() or []

        data: list[dict[str, Any]] = []
        seen: set[str] = set()
        for model in available:
            model_id = (
                getattr(model, "model_name", None)
                or getattr(model, "display_name", None)
                or str(model)
            )
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)

            data.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "google",
                }
            )

        default_model = os.getenv("GEMINI_DEFAULT_MODEL")
        if default_model and default_model not in seen:
            data.append(
                {
                    "id": default_model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "google",
                }
            )

        return JSONResponse(content={"object": "list", "data": data})
    except AuthError as exc:
        return _openai_error(
            401, f"Authentication failed: {exc}", "authentication_error"
        )
    except Exception as exc:
        return _openai_error(500, str(exc), "server_error")


@app.post("/v1/chat/completions")
async def openai_chat_completions(
    request: Request, req: OpenAIChatCompletionRequest
) -> JSONResponse:
    auth_error = _check_openai_auth(request)
    if auth_error:
        return auth_error

    if req.stream:
        return _openai_error(
            400,
            "stream=true is not supported by this endpoint yet.",
            code="unsupported_stream",
        )

    try:
        prompt = _messages_to_prompt(req.messages)
    except ValueError as exc:
        return _openai_error(400, str(exc), code="invalid_messages")

    try:
        client = await _get_client()
        chosen_model = req.model or os.getenv("GEMINI_DEFAULT_MODEL")

        request_kwargs: dict[str, Any] = {"prompt": prompt}
        if chosen_model:
            request_kwargs["model"] = chosen_model

        output = await client.generate_content(**request_kwargs)
        answer = output.text or ""

        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(answer)

        return JSONResponse(
            content={
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": chosen_model or "gemini-web",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )
    except AuthError as exc:
        return _openai_error(
            401, f"Authentication failed: {exc}", "authentication_error"
        )
    except (UsageLimitExceeded, TemporarilyBlocked) as exc:
        return _openai_error(429, str(exc), "rate_limit_error")
    except TimeoutError as exc:
        return _openai_error(504, str(exc), "server_error", code="upstream_timeout")
    except (GeminiError, APIError) as exc:
        return _openai_error(502, str(exc), "server_error")
    except Exception as exc:
        return _openai_error(500, str(exc), "server_error")


@app.on_event("shutdown")
async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None
