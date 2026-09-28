"""LLM provider implementations — Gemini and Groq.

Each provider wraps its respective SDK and returns a unified LLMResponse.
Providers handle structured output differently:
  - Gemini: native `response_schema` + `response_mime_type: "application/json"`
  - Groq: OpenAI-compatible function/tool calling

References: PRD §1b (model allocation), §3 (structured output).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

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
# Provider protocol
# ---------------------------------------------------------------------------

class LLMProvider(Protocol):
    """Protocol for LLM providers — Gemini, Groq, etc."""

    @property
    def name(self) -> str: ...

    async def invoke(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
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

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return "gemini"

    async def invoke(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
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

        # Build config
        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

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


# ---------------------------------------------------------------------------
# Groq Provider (official groq SDK — OpenAI-compatible)
# ---------------------------------------------------------------------------

class GroqProvider:
    """Groq provider using the official groq SDK.

    Uses function/tool calling for structured output.
    """

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        from groq import AsyncGroq

        self._client = AsyncGroq(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return "groq"

    async def invoke(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
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
        # Build kwargs
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Groq supports JSON mode — inject schema into system prompt
        if response_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
            # Prepend schema instruction to system message
            schema_instruction = (
                f"\n\nYou MUST respond with valid JSON matching this schema:\n"
                f"```json\n{json.dumps(response_schema, indent=2)}\n```"
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

        response = await self._client.chat.completions.create(**kwargs)

        # Extract content and tokens
        content_text = ""
        if response.choices and response.choices[0].message.content:
            content_text = response.choices[0].message.content

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
