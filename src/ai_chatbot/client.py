"""
client.py - The local Ollama client

This module talks to Ollama running on the user's own computer, so no paid API
keys are needed and all chat requests stay local.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import ollama

from ai_chatbot.logger import get_logger
from ai_chatbot.models import AppConfig, ChatMessage, OllamaChatResponse
from ai_chatbot.ui import print_retry_message

logger = get_logger()


class OllamaError(Exception):
    """Base class for Ollama-related problems."""


class OllamaAPIError(OllamaError):
    """Raised when Ollama returns an error response or invalid data."""

    def __init__(self, status_code: int, message: str, details: str = ""):
        self.status_code = status_code
        self.message = message
        self.details = details
        super().__init__(f"Ollama error {status_code}: {message}")


class OllamaTimeoutError(OllamaError):
    """Raised when Ollama takes too long to answer."""


class OllamaConnectionError(OllamaError):
    """Raised when the app cannot reach the local Ollama server."""


class OllamaConfigError(OllamaError):
    """Raised when the Ollama client has not been prepared correctly."""


class OllamaModelNotFoundError(OllamaAPIError):
    """Raised when the requested model is not installed locally."""


class OllamaClient:
    """Async helper that sends chat requests to a local Ollama server."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._client: Optional[ollama.AsyncClient] = None
        logger.debug(
            f"OllamaClient initialized with host={config.ollama_host} and model={config.model_name}"
        )

    async def __aenter__(self) -> "OllamaClient":
        """Create the async Ollama client when the chatbot starts."""
        self._client = ollama.AsyncClient(host=self.config.ollama_host)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the HTTP client cleanly when the chatbot exits."""
        if self._client is not None:
            await self._client.close()
            logger.debug("Ollama AsyncClient closed")

    def _require_client(self) -> ollama.AsyncClient:
        """Make sure the async client exists before we try to use it."""
        if self._client is None:
            raise OllamaConfigError("Ollama client is not ready. Use 'async with OllamaClient(...)'.")
        return self._client

    def _build_messages(self, user_message: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Turn the current chat session into the message list Ollama expects."""
        messages: List[Dict[str, str]] = []

        # A system prompt gives the model a simple role before the chat begins.
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})

        # We keep the previous conversation so the model remembers the discussion.
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _build_options(self) -> Dict[str, Any]:
        """Build the Ollama options that control output length and style."""
        return {
            "temperature": self.config.temperature,
            "num_predict": self.config.max_tokens,
        }

    @staticmethod
    def _extract_message_content(response: Any) -> str:
        """Read assistant text from either a dict response or a response object."""
        try:
            validated_response = OllamaChatResponse.model_validate(response)
        except Exception as exc:
            raise OllamaAPIError(
                500,
                "Invalid Ollama response format.",
                str(exc),
            ) from exc

        return validated_response.message.content or ""

    async def verify_ready(self) -> None:
        """Check that Ollama is running and that the requested model exists."""
        client = self._require_client()

        try:
            # Listing local models proves the server is reachable.
            await client.list()
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Ollama did not respond within {self.config.timeout}s."
            ) from exc
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                "Cannot connect to Ollama at http://localhost:11434. Start Ollama first and try again."
            ) from exc
        except Exception as exc:
            raise OllamaConnectionError(f"Could not reach Ollama: {exc}") from exc

        try:
            # show(model) confirms that the exact model is available locally.
            await client.show(self.config.model_name)
        except ollama.ResponseError as exc:
            if getattr(exc, "status_code", None) == 404:
                raise OllamaModelNotFoundError(
                    404,
                    f"The model '{self.config.model_name}' is not installed locally.",
                    "Run: ollama pull gemma3:4b",
                ) from exc
            raise OllamaAPIError(
                getattr(exc, "status_code", 500),
                f"Could not load model '{self.config.model_name}'.",
                getattr(exc, "error", str(exc)),
            ) from exc

        logger.info(
            f"Verified Ollama server and model '{self.config.model_name}' successfully"
        )

    async def send_message(self, message: str, history: List[Dict[str, str]]) -> str:
        """Send one chat message and return the full assistant reply."""
        validated_message = ChatMessage(role="user", content=message)
        messages = self._build_messages(validated_message.content, history)
        options = self._build_options()

        logger.info(
            f"Sending message to Ollama: '{validated_message.content[:50]}...' (history: {len(history)} msgs)"
        )

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = await self._chat_once(messages, options)
                response_text = self._extract_message_content(response).strip()

                if not response_text:
                    raise OllamaAPIError(
                        500,
                        "Ollama returned an empty response.",
                        "The model replied without any text.",
                    )

                logger.info(
                    f"Received Ollama response: '{response_text[:50]}...' ({len(response_text)} chars)"
                )
                return response_text

            except (OllamaTimeoutError, OllamaConnectionError, OllamaAPIError) as exc:
                if isinstance(exc, OllamaAPIError) and exc.status_code in {400, 404}:
                    raise

                if attempt >= self.config.max_retries:
                    raise

                wait_seconds = 2 ** (attempt - 1)
                logger.warning(
                    f"Request attempt {attempt} failed, retrying in {wait_seconds}s: {exc}"
                )
                print_retry_message(attempt, self.config.max_retries, wait_seconds)
                await asyncio.sleep(wait_seconds)

        raise OllamaAPIError(500, "All retry attempts failed.")

    async def _chat_once(self, messages: List[Dict[str, str]], options: Dict[str, Any]) -> Any:
        """Make one Ollama chat request and turn transport errors into simple messages."""
        client = self._require_client()

        try:
            return await client.chat(
                model=self.config.model_name,
                messages=messages,
                options=options,
                stream=False,
            )
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Ollama did not answer within {self.config.timeout}s."
            ) from exc
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                "Cannot connect to Ollama at http://localhost:11434. Start Ollama first and try again."
            ) from exc
        except ollama.ResponseError as exc:
            status_code = getattr(exc, "status_code", 500)
            error_text = getattr(exc, "error", str(exc))

            if status_code == 404:
                raise OllamaModelNotFoundError(
                    404,
                    f"The model '{self.config.model_name}' is not installed locally.",
                    "Run: ollama pull gemma3:4b",
                ) from exc

            if status_code == 503:
                raise OllamaConnectionError(
                    "Ollama is running, but it looks busy or unavailable right now. Try again in a moment."
                ) from exc

            raise OllamaAPIError(status_code, "Ollama returned an error.", error_text) from exc
        except Exception as exc:
            raise OllamaAPIError(500, f"Unexpected Ollama error: {exc}") from exc

    async def send_message_streaming(
        self,
        message: str,
        history: List[Dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """Send one chat message and stream the assistant reply chunk by chunk."""
        validated_message = ChatMessage(role="user", content=message)
        messages = self._build_messages(validated_message.content, history)
        options = self._build_options()
        client = self._require_client()

        logger.info(f"Starting Ollama streaming request: '{validated_message.content[:50]}...'")

        try:
            stream = await client.chat(
                model=self.config.model_name,
                messages=messages,
                options=options,
                stream=True,
            )

            full_text_parts: List[str] = []
            async for chunk in stream:
                chunk_text = self._extract_message_content(chunk)
                if not chunk_text:
                    continue

                full_text_parts.append(chunk_text)
                yield chunk_text

            full_text = "".join(full_text_parts).strip()
            if not full_text:
                raise OllamaAPIError(
                    500,
                    "Ollama returned an empty streamed response.",
                    "The model finished without sending text.",
                )

            logger.info(f"Streaming complete: {len(full_text)} chars received")

        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError("Ollama streaming request timed out.") from exc
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                "Cannot connect to Ollama at http://localhost:11434. Start Ollama first and try again."
            ) from exc
        except ollama.ResponseError as exc:
            status_code = getattr(exc, "status_code", 500)
            error_text = getattr(exc, "error", str(exc))
            if status_code == 404:
                raise OllamaModelNotFoundError(
                    404,
                    f"The model '{self.config.model_name}' is not installed locally.",
                    "Run: ollama pull gemma3:4b",
                ) from exc
            raise OllamaAPIError(status_code, "Ollama returned an error during streaming.", error_text) from exc
        except OllamaError:
            raise
        except Exception as exc:
            raise OllamaAPIError(500, f"Unexpected streaming error: {exc}") from exc
