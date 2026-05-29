"""
chat.py - The chatbot conversation loop

This module keeps the original chat flow, but it can now talk to either
Ollama or Gemini depending on the selected provider.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List

from ai_chatbot.client import (
    ChatClientAPIError,
    ChatClientConnectionError,
    ChatClientModelNotFoundError,
    ChatClientProtocol,
    ChatClientTimeoutError,
    GeminiClient,
    OllamaClient,
)
from ai_chatbot.config import load_config
from ai_chatbot.logger import get_logger, setup_logger
from ai_chatbot.models import AppConfig
from ai_chatbot.ui import (
    clean_markdown,
    console,
    get_user_input,
    print_ai_message,
    print_chat_history,
    print_divider,
    print_error,
    print_goodbye,
    print_help_text,
    print_system_message,
    print_success,
    print_thinking_start,
    print_user_message,
    print_welcome_banner,
)

logger = get_logger()
EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye", "q"}


class ChatSession:
    """Keep the current conversation state in memory."""

    def __init__(self, config: AppConfig, client: ChatClientProtocol):
        """Store the app config, selected client, and conversation history."""
        self.config = config
        self.client = client
        self.history: List[Dict[str, str]] = []
        self.message_count: int = 0
        self.streaming_enabled: bool = config.thinking and client.supports_streaming
        logger.info("New chat session started")

    def add_to_history(self, role: str, text: str) -> None:
        """Save one message so the next request remembers the conversation."""
        self.history.append({"role": role, "content": text})
        logger.debug(f"Added to history: role={role}, len={len(text)}")

    def reset_history(self) -> None:
        """Clear the saved conversation so the user can start fresh."""
        self.history = []
        self.message_count = 0
        logger.info("Conversation history cleared")

    async def process_message(self, user_input: str) -> bool:
        """Process one user message and return False when the user wants to exit."""
        if user_input.lower() in EXIT_COMMANDS:
            logger.info("User requested exit")
            return False

        command = user_input.lower().strip()

        if command == "clear":
            console.clear()
            print_welcome_banner()
            print_help_text()
            return True

        if command == "history":
            print_chat_history(self.history)
            return True

        if command == "reset":
            self.reset_history()
            print_success("Conversation history cleared! Starting fresh.")
            return True

        if command == "help":
            print_help_text()
            return True

        if not user_input.strip():
            print_system_message("Please type something! (or type 'exit' to quit)")
            return True

        logger.info(f"User message #{self.message_count + 1}: '{user_input[:80]}...'")
        print_user_message(user_input)

        try:
            if self.streaming_enabled:
                await self._process_streaming(user_input)
            else:
                await self._process_standard(user_input)

            self.message_count += 1
            return True

        except ChatClientModelNotFoundError as exc:
            logger.error(f"Model not found: {exc}")
            if self.config.provider == "gemini":
                print_error(
                    "Model not available",
                    f"Check GEMINI_MODEL and the Gemini API access for this model.\nDetails: {exc.message}\nThen start the chatbot again.",
                )
            else:
                print_error(
                    "Model not installed",
                    "Run: ollama pull gemma3:4b\n"
                    f"Details: {exc.message}\n"
                    "Then start the chatbot again.",
                )
            return True

        except ChatClientConnectionError as exc:
            logger.error(f"Connection error: {exc}")
            if self.config.provider == "gemini":
                print_error(
                    "Cannot connect to Gemini",
                    f"Make sure your network can reach the Gemini API and GEMINI_API_KEY is valid.\nDetails: {exc}",
                )
            else:
                print_error(
                    "Cannot connect to Ollama",
                    f"Make sure Ollama is running locally at {self.config.ollama_host}.\n"
                    f"Details: {exc}",
                )
            return True

        except ChatClientTimeoutError as exc:
            logger.error(f"Timeout error: {exc}")
            if self.config.provider == "gemini":
                print_error(
                    "Request timed out",
                    f"Gemini took too long to answer. Your timeout is set to {self.config.timeout}s.\n"
                    "Try again or increase REQUEST_TIMEOUT in .env.\n"
                    f"Details: {exc}",
                )
            else:
                print_error(
                    "Request timed out",
                    f"Ollama took too long to answer. Your timeout is set to {self.config.timeout}s.\n"
                    "Try again or increase REQUEST_TIMEOUT in .env.\n"
                    f"Details: {exc}",
                )
            return True

        except ChatClientAPIError as exc:
            logger.error(f"API error for message #{self.message_count}: {exc}")

            if exc.status_code == 404:
                if self.config.provider == "gemini":
                    print_error(
                        "Model not available",
                        f"Check GEMINI_MODEL and make sure Gemini can access the model.\nDetails: {exc.message}",
                    )
                else:
                    print_error(
                        "Model not installed",
                        "Run: ollama pull gemma3:4b\n"
                        f"Details: {exc.message}",
                    )
            elif exc.status_code == 429:
                if self.config.provider == "gemini":
                    print_error(
                        "Gemini rate limited",
                        "The Gemini API is busy or rate limited right now. Wait a moment and try again.",
                    )
                else:
                    print_error(
                        "Ollama is busy",
                        "The local model is busy. Wait a moment and try again.",
                    )
            elif exc.status_code >= 500:
                if self.config.provider == "gemini":
                    print_error(
                        "Gemini server error",
                        f"The Gemini API returned an error. ({exc.message})",
                    )
                else:
                    print_error(
                        "Ollama server error",
                        f"The local Ollama server returned an error. ({exc.message})",
                    )
            else:
                provider_name = self.client.provider_label
                print_error(f"{provider_name} API Error ({exc.status_code})", exc.message)

            return True

        except ValueError as exc:
            logger.warning(f"Validation error: {exc}")
            print_error("Invalid input", str(exc))
            return True

        except KeyboardInterrupt:
            print_system_message("\nInterrupted! Type 'exit' to quit or continue chatting.")
            return True

        except Exception as exc:
            logger.error(f"Unexpected error: {type(exc).__name__}: {exc}", exc_info=True)
            print_error(
                "Unexpected error",
                f"{type(exc).__name__}: {str(exc)}\n"
                "Check the log file for details.",
            )
            return True

    async def _process_streaming(self, user_input: str) -> None:
        """Show the Ollama response as it arrives so the UI feels live."""
        print()
        full_response_chunks: List[str] = []
        is_first_chunk = True

        async for chunk in self.client.send_message_streaming(user_input, self.history):
            if is_first_chunk:
                console.print("AI is writing...", style="info")
                console.print()
                is_first_chunk = False

            clean_chunk = clean_markdown(chunk)
            print(clean_chunk, end="", flush=True)
            full_response_chunks.append(clean_chunk)

        print("\n")
        full_response = "".join(full_response_chunks).strip()

        if full_response:
            self.add_to_history("user", user_input)
            self.add_to_history("assistant", full_response)
            logger.info(
                f"AI response #{self.message_count + 1} received (streaming): {len(full_response)} chars"
            )
        else:
            print_error("Empty response", "The model returned no text. Please try again.")

    async def _process_standard(self, user_input: str) -> None:
        """Show a simple thinking message, wait for the answer, then print it."""
        print_thinking_start()
        response_text = await self.client.send_message(user_input, self.history)
        cleaned = clean_markdown(response_text)

        self.add_to_history("user", user_input)
        self.add_to_history("assistant", cleaned)
        print_ai_message(cleaned, f"{self.client.provider_label} / {self.config.model_name}")
        logger.info(f"AI response #{self.message_count + 1} received: {len(response_text)} chars")


async def run_chatbot(provider: str | None = None, thinking: bool | None = None) -> None:
    """Start the chatbot, verify the selected provider, and then enter the conversation loop."""
    config = load_config(provider=provider, thinking=thinking)
    setup_logger(name="ai_chatbot", log_file=config.log_file, level=config.log_level)
    logger_configured = get_logger()
    logger_configured.info("Starting AI Chatbot application")

    try:
        client = GeminiClient(config) if config.provider == "gemini" else OllamaClient(config)

        async with client as active_client:
            # We verify the selected provider before showing the normal chat UI.
            await active_client.verify_ready()

            print_welcome_banner()
            print_help_text()
            print_system_message(f"Using provider: [bold]{config.provider.title()}[/]")
            print_system_message(f"Using model: [bold]{config.model_name}[/]")
            print_divider()
            if config.provider == "gemini":
                print_system_message("Connected to Gemini API")
            else:
                print_system_message(f"Connected to Ollama at {config.ollama_host}")

            if config.thinking and not active_client.supports_streaming:
                print_system_message(
                    "Thinking mode is enabled, but this provider does not support streaming."
                )

            session = ChatSession(config, active_client)
            logger_configured.info("Chat session started, entering main loop")
            print_system_message("Ready to chat! Type your message below.")

            while True:
                try:
                    user_input = get_user_input()
                    should_continue = await session.process_message(user_input)

                    if not should_continue:
                        break

                except KeyboardInterrupt:
                    print()
                    print_system_message(
                        "Ctrl+C detected. Type 'exit' to quit properly, or press Ctrl+C again."
                    )
                    try:
                        await asyncio.sleep(0.5)
                    except KeyboardInterrupt:
                        break

                except EOFError:
                    break

            logger_configured.info(
                f"Chat session ended. Total messages: {session.message_count}"
            )
            print_goodbye()

    except ChatClientModelNotFoundError as exc:
        if config.provider == "gemini":
            print_error(
                "Model not available",
                f"{exc.message}\n\nDetails: {exc.details}\n\nCheck GEMINI_MODEL and your API access.",
            )
        else:
            print_error(
                "Model not installed",
                f"{exc.message}\n\nDetails: {exc.details}\n\nRun: ollama pull gemma3:4b",
            )
    except ChatClientConnectionError as exc:
        if config.provider == "gemini":
            print_error(
                "Cannot connect to Gemini",
                f"Make sure your network can reach the Gemini API.\nDetails: {exc}",
            )
        else:
            print_error(
                "Cannot connect to Ollama",
                f"Make sure Ollama is running locally at {config.ollama_host if 'config' in locals() else 'http://localhost:11434'}.\n"
                f"Details: {exc}",
            )
    except ChatClientTimeoutError as exc:
        if config.provider == "gemini":
            print_error(
                "Gemini timed out",
                f"Gemini did not answer in time.\nDetails: {exc}",
            )
        else:
            print_error(
                "Ollama timed out",
                f"Ollama did not answer in time.\nDetails: {exc}",
            )
    except ChatClientAPIError as exc:
        if config.provider == "gemini":
            print_error(
                "Gemini error",
                f"{exc.message}\n\nDetails: {exc.details}",
            )
        else:
            print_error(
                "Ollama error",
                f"{exc.message}\n\nDetails: {exc.details}",
            )
