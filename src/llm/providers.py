"""LLM provider implementations — Gemini and Groq.

Each provider wraps its respective SDK and returns a unified LLMResponse.
Providers handle structured output differently:
  - Gemini: native `response_schema` + `response_mime_type: "application/json"`
  - Groq: OpenAI-compatible function/tool calling

Both providers accept a Pydantic BaseModel subclass as `response_schema`;
the JSON schema is derived automatically via `model_json_schema()`.

References: PRD §1b (model allocation), §3 (structured output).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol
from groq import AsyncGroq
from pydantic import BaseModel
from google import genai
import structlog

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Unified response type
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""
    content: str                    # Raw text or JSON string
    tokens_used: int = 0           # Total tokens (input + output)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    provider: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_schema(schema: type[BaseModel] | dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert a Pydantic model class to a JSON Schema dict, or pass through."""
    if schema is None:
        return None
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_json_schema()
    return schema  # already a dict


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------

class LLMProvider(Protocol):
    """Protocol for LLM providers — Gemini, Groq, etc."""

    @property
    def name(self) -> str: ...

    async def invoke(
        self,
        messages: list[dict[str, str]],
        response_schema: type[BaseModel] | dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# Gemini Provider (google-genai SDK)
# ---------------------------------------------------------------------------

class GeminiProvider:
    """Google Gemini provider using the google-genai SDK.

    Uses structured output via response_schema when provided.
    """

    def __init__(self, api_key: str, model: str = "gemini-3.8-flash"):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return "gemini"

    async def invoke(
        self,
        messages: list[dict[str, str]],
        response_schema: type[BaseModel] | dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """Call Gemini with optional structured JSON output.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            response_schema: Optional JSON Schema dict for structured output.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.

        Returns:
            LLMResponse with content (JSON string if schema provided).
        """
        from google.genai import types

        # Resolve Pydantic model → JSON Schema dict if needed
        resolved_schema = _resolve_schema(response_schema)

        # Build config
        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        if resolved_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = resolved_schema

        config = types.GenerateContentConfig(**config_kwargs)

        # Convert messages to Gemini format
        # Gemini uses "user" and "model" roles; map "system" → system_instruction
        system_instruction = None
        contents = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_instruction = content
            else:
                gemini_role = "model" if role == "assistant" else "user"
                contents.append(types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=content)],
                ))

        if system_instruction:
            config.system_instruction = system_instruction

        # Make the API call
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        # Extract token usage
        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        content_text = response.text or ""

        logger.debug(
            "llm.gemini.response",
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return LLMResponse(
            content=content_text,
            tokens_used=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._model,
            provider="gemini",
        )


def _loads_or_raw(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text


def _recover_failed_tool_generation(error: Exception) -> str | None:
    """Groq rejects a reply with ``tool_use_failed`` when the model emits a tool
    call it wasn't offered, but includes the generation in the error body. That
    text is a perfectly good tool call for our parser, so use it."""
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("code") == "tool_use_failed":
            gen = err.get("failed_generation")
            if isinstance(gen, str) and gen.strip():
                return gen
    return None


class GroqProvider:
    supports_native_tools = True

    """Groq provider using the official groq SDK.

    Uses function/tool calling for structured output.
    """

    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b"):


        self._client = AsyncGroq(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return "groq"

    async def invoke(
        self,
        messages: list[dict[str, str]],
        response_schema: type[BaseModel] | dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Call Groq with optional structured output via JSON mode.

        Args:
            messages: List of message dicts with "role", "content" keys.
            response_schema: Optional JSON Schema dict — when provided, uses
                JSON mode and injects the schema into the system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.

        Returns:
            LLMResponse with content.
        """
        # Resolve Pydantic model → JSON Schema dict if needed
        resolved_schema = _resolve_schema(response_schema)

        # Build kwargs
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # gpt-oss models are reasoning models: keep reasoning short so the
        # output budget goes to the actual answer (tool calls / JSON).
        if self._model.startswith("openai/gpt-oss"):
            kwargs["reasoning_effort"] = "low"

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # Groq supports JSON mode — inject schema into system prompt
        if resolved_schema is not None and not tools:
            kwargs["response_format"] = {"type": "json_object"}
            # Prepend schema instruction to system message
            schema_instruction = (
                f"\n\nYou MUST respond with valid JSON matching this schema:\n"
                f"```json\n{json.dumps(resolved_schema, indent=2)}\n```"
            )
            messages = list(messages)  # Don't mutate caller's list
            if messages and messages[0]["role"] == "system":
                messages[0] = {
                    "role": "system",
                    "content": messages[0]["content"] + schema_instruction,
                }
            else:
                messages.insert(0, {"role": "system", "content": schema_instruction})
            kwargs["messages"] = messages

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            recovered = _recover_failed_tool_generation(e)
            if recovered is None:
                raise
            logger.info("llm.groq.recovered_tool_generation", model=self._model)
            return LLMResponse(
                content=recovered, tokens_used=0, input_tokens=0, output_tokens=0,
                model=self._model, provider="groq",
            )

        # Extract content and tokens. Native tool calls are serialised into the
        # same JSON text shape the agent loop already parses.
        content_text = ""
        message = response.choices[0].message if response.choices else None
        if message is not None and getattr(message, "tool_calls", None):
            content_text = json.dumps({"tool_calls": [
                {"name": tc.function.name, "arguments": _loads_or_raw(tc.function.arguments)}
                for tc in message.tool_calls
            ]})
        elif message is not None and message.content:
            content_text = message.content

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        logger.debug(
            "llm.groq.response",
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return LLMResponse(
            content=content_text,
            tokens_used=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._model,
            provider="groq",
        )
