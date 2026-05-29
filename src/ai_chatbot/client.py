"""
client.py - The local Ollama client

This module talks to Ollama running on the user's own computer, so no paid API
keys are needed and all chat requests stay local.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Protocol, runtime_checkable

import httpx
import ollama

from ai_chatbot.logger import get_logger
from ai_chatbot.models import AppConfig, ChatMessage, OllamaChatResponse
from ai_chatbot.ui import print_retry_message

logger = get_logger()


@runtime_checkable
class ChatClientProtocol(Protocol):
    provider_name: str
    provider_label: str
    supports_streaming: bool

    async def __aenter__(self) -> "ChatClientProtocol":
        ...

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        ...

    async def verify_ready(self) -> None:
        ...

    async def send_message(self, message: str, history: List[Dict[str, str]]) -> str:
        ...

    async def send_message_streaming(
        self,
        message: str,
        history: List[Dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        ...


class ChatClientError(Exception):
    """Base class for provider-related problems."""


class ChatClientAPIError(ChatClientError):
    """Raised when a provider returns an error response or invalid data."""

    def __init__(self, status_code: int, message: str, details: str = ""):
        self.status_code = status_code
        self.message = message
        self.details = details
        super().__init__(f"API error {status_code}: {message}")


class ChatClientTimeoutError(ChatClientError):
    """Raised when a provider takes too long to answer."""


class ChatClientConnectionError(ChatClientError):
    """Raised when the app cannot reach a provider."""


class ChatClientConfigError(ChatClientError):
    """Raised when the client has not been prepared correctly."""


class ChatClientModelNotFoundError(ChatClientAPIError):
    """Raised when the requested model is not available."""


class OllamaError(ChatClientError):
    """Base class for Ollama-related problems."""


class OllamaAPIError(ChatClientAPIError, OllamaError):
    """Raised when Ollama returns an error response or invalid data."""


class OllamaTimeoutError(ChatClientTimeoutError, OllamaError):
    """Raised when Ollama takes too long to answer."""


class OllamaConnectionError(ChatClientConnectionError, OllamaError):
    """Raised when the app cannot reach the local Ollama server."""


class OllamaConfigError(ChatClientConfigError, OllamaError):
    """Raised when the Ollama client has not been prepared correctly."""


class OllamaModelNotFoundError(ChatClientModelNotFoundError, OllamaAPIError):
    """Raised when the requested model is not installed locally."""


class GeminiError(ChatClientError):
    """Base class for Gemini-related problems."""


class GeminiAPIError(ChatClientAPIError, GeminiError):
    """Raised when Gemini returns an error response or invalid data."""


class GeminiTimeoutError(ChatClientTimeoutError, GeminiError):
    """Raised when Gemini takes too long to answer."""


class GeminiConnectionError(ChatClientConnectionError, GeminiError):
    """Raised when the app cannot reach the Gemini API."""


class GeminiConfigError(ChatClientConfigError, GeminiError):
    """Raised when the Gemini client has not been prepared correctly."""


class GeminiModelNotFoundError(ChatClientModelNotFoundError, GeminiAPIError):
    """Raised when the requested Gemini model is not available."""


class OllamaClient:
    """Async helper that sends chat requests to a local Ollama server."""

    provider_name = "ollama"
    provider_label = "Ollama"
    supports_streaming = True

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


class GeminiClient:
    """Async helper that sends chat requests to the Gemini API."""

    provider_name = "gemini"
    provider_label = "Gemini"
    supports_streaming = True

    def __init__(self, config: AppConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._base_url = "https://generativelanguage.googleapis.com/v1beta"
        logger.debug(f"GeminiClient initialized with model={config.model_name}")

    async def __aenter__(self) -> "GeminiClient":
        """Create the async Gemini client when the chatbot starts."""
        self._client = httpx.AsyncClient(timeout=self.config.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the HTTP client cleanly when the chatbot exits."""
        if self._client is not None:
            await self._client.aclose()
            logger.debug("Gemini AsyncClient closed")

    def _require_client(self) -> httpx.AsyncClient:
        """Make sure the async client exists before we try to use it."""
        if self._client is None:
            raise GeminiConfigError("Gemini client is not ready. Use 'async with GeminiClient(...)'.")
        return self._client

    def _auth_params(self) -> Dict[str, str]:
        return {"key": self.config.gemini_api_key.strip()}

    def _build_contents(self, user_message: str, history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        contents: List[Dict[str, Any]] = []

        for message in history:
            role = message.get("role", "user")
            content = message.get("content", "")
            if not content:
                continue

            contents.append(
                {
                    "role": "model" if role == "assistant" else "user",
                    "parts": [{"text": content}],
                }
            )

        contents.append({"role": "user", "parts": [{"text": user_message}]})
        return contents

    def _build_generation_config(self) -> Dict[str, Any]:
        return {
            "temperature": self.config.temperature,
            "maxOutputTokens": self.config.max_tokens,
        }

    def _build_payload(self, user_message: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "contents": self._build_contents(user_message, history),
            "generationConfig": self._build_generation_config(),
        }

        if self.config.system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": self.config.system_prompt}]}

        return payload

    @staticmethod
    def _extract_text(response: Any) -> str:
        if not isinstance(response, dict):
            raise GeminiAPIError(500, "Invalid Gemini response format.", f"Unexpected type: {type(response).__name__}")

        if isinstance(response.get("text"), str) and response["text"]:
            return response["text"]

        candidates = response.get("candidates") or []
        text_parts: List[str] = []

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            content = candidate.get("content") or {}
            parts = content.get("parts") or []
            for part in parts:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)

        return "".join(text_parts)

    @staticmethod
    def _extract_stream_delta(accumulated_text: str, chunk_text: str) -> str:
        if not accumulated_text:
            return chunk_text

        if chunk_text.startswith(accumulated_text):
            return chunk_text[len(accumulated_text) :]

        return chunk_text

    @staticmethod
    async def _read_error_text(response: httpx.Response) -> str:
        """Safely read the body from a response, including streaming responses."""
        try:
            content = await response.aread()
        except Exception:
            return ""

        if not content:
            return ""

        encoding = response.encoding or "utf-8"
        return content.decode(encoding, errors="replace")

    async def verify_ready(self) -> None:
        """Check that the Gemini API key works and the requested model exists."""
        if not self.config.gemini_api_key.strip():
            raise GeminiConfigError("GEMINI_API_KEY is missing.")

        client = self._require_client()
        model_url = f"{self._base_url}/models/{self.config.model_name}"

        try:
            response = await client.get(model_url, params=self._auth_params())
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise GeminiTimeoutError(f"Gemini did not respond within {self.config.timeout}s.") from exc
        except httpx.ConnectError as exc:
            raise GeminiConnectionError("Cannot connect to the Gemini API. Check your network connection.") from exc
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            error_text = await self._read_error_text(exc.response)

            if status_code == 404:
                raise GeminiModelNotFoundError(
                    404,
                    f"The model '{self.config.model_name}' is not available in Gemini.",
                    "Check GEMINI_MODEL and make sure the model name is valid.",
                ) from exc

            if status_code in {401, 403}:
                raise GeminiConfigError(
                    "Gemini rejected the API key or the project does not have access to the model.",
                ) from exc

            raise GeminiAPIError(status_code, "Gemini returned an error while verifying the model.", error_text) from exc
        except Exception as exc:
            raise GeminiConnectionError(f"Could not reach Gemini: {exc}") from exc

        logger.info(f"Verified Gemini model '{self.config.model_name}' successfully")

    async def _generate_once(self, user_message: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        client = self._require_client()
        payload = self._build_payload(user_message, history)
        url = f"{self._base_url}/models/{self.config.model_name}:generateContent"

        try:
            response = await client.post(url, params=self._auth_params(), json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            raise GeminiTimeoutError(f"Gemini did not answer within {self.config.timeout}s.") from exc
        except httpx.ConnectError as exc:
            raise GeminiConnectionError("Cannot connect to the Gemini API. Check your network connection.") from exc
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            error_text = await self._read_error_text(exc.response)

            if status_code == 404:
                raise GeminiModelNotFoundError(
                    404,
                    f"The model '{self.config.model_name}' is not available in Gemini.",
                    "Check GEMINI_MODEL and make sure the model name is valid.",
                ) from exc

            if status_code in {401, 403}:
                raise GeminiConfigError(
                    "Gemini rejected the API key or the project does not have access to the model.",
                ) from exc

            raise GeminiAPIError(status_code, "Gemini returned an error.", error_text) from exc
        except Exception as exc:
            raise GeminiAPIError(500, f"Unexpected Gemini error: {exc}") from exc

    async def send_message(self, message: str, history: List[Dict[str, str]]) -> str:
        """Send one chat message and return the full assistant reply."""
        validated_message = ChatMessage(role="user", content=message)

        logger.info(
            f"Sending message to Gemini: '{validated_message.content[:50]}...' (history: {len(history)} msgs)"
        )

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = await self._generate_once(validated_message.content, history)
                response_text = self._extract_text(response).strip()

                if not response_text:
                    raise GeminiAPIError(
                        500,
                        "Gemini returned an empty response.",
                        "The model replied without any text.",
                    )

                logger.info(
                    f"Received Gemini response: '{response_text[:50]}...' ({len(response_text)} chars)"
                )
                return response_text

            except (GeminiTimeoutError, GeminiConnectionError, GeminiAPIError) as exc:
                if isinstance(exc, GeminiAPIError) and exc.status_code in {400, 404}:
                    raise

                if attempt >= self.config.max_retries:
                    raise

                wait_seconds = 2 ** (attempt - 1)
                logger.warning(
                    f"Request attempt {attempt} failed, retrying in {wait_seconds}s: {exc}"
                )
                print_retry_message(attempt, self.config.max_retries, wait_seconds)
                await asyncio.sleep(wait_seconds)

        raise GeminiAPIError(500, "All retry attempts failed.")

    async def send_message_streaming(
        self,
        message: str,
        history: List[Dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """Send one chat message and stream the assistant reply chunk by chunk."""
        validated_message = ChatMessage(role="user", content=message)
        payload = self._build_payload(validated_message.content, history)
        client = self._require_client()
        url = f"{self._base_url}/models/{self.config.model_name}:streamGenerateContent"

        logger.info(
            f"Starting Gemini streaming request: '{validated_message.content[:50]}...'"
        )

        try:
            async with client.stream(
                "POST",
                url,
                params={**self._auth_params(), "alt": "sse"},
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    error_text = await self._read_error_text(response)

                    if response.status_code == 404:
                        raise GeminiModelNotFoundError(
                            404,
                            f"The model '{self.config.model_name}' is not available in Gemini.",
                            "Check GEMINI_MODEL and make sure the model name is valid.",
                        )

                    if response.status_code in {401, 403}:
                        raise GeminiConfigError(
                            "Gemini rejected the API key or the project does not have access to the model.",
                        )

                    raise GeminiAPIError(
                        response.status_code,
                        "Gemini returned an error during streaming.",
                        error_text,
                    )

                accumulated_text = ""
                full_text_parts: List[str] = []

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue

                    data = line.removeprefix("data:").strip()
                    if not data or data == "[DONE]":
                        continue

                    chunk = json.loads(data)
                    chunk_text = self._extract_text(chunk)
                    if not chunk_text:
                        continue

                    delta = self._extract_stream_delta(accumulated_text, chunk_text)
                    if not delta:
                        continue

                    accumulated_text = chunk_text if chunk_text.startswith(accumulated_text) else accumulated_text + delta
                    full_text_parts.append(delta)
                    yield delta

                full_text = "".join(full_text_parts).strip()
                if not full_text:
                    raise GeminiAPIError(
                        500,
                        "Gemini returned an empty streamed response.",
                        "The model finished without sending text.",
                    )

                logger.info(f"Gemini streaming complete: {len(full_text)} chars received")

        except json.JSONDecodeError as exc:
            raise GeminiAPIError(500, "Gemini returned invalid stream data.", str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise GeminiTimeoutError("Gemini streaming request timed out.") from exc
        except httpx.ConnectError as exc:
            raise GeminiConnectionError("Cannot connect to the Gemini API. Check your network connection.") from exc
        except GeminiError:
            raise
        except Exception as exc:
            raise GeminiAPIError(500, f"Unexpected streaming error: {exc}") from exc
