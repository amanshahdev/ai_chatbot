"""
config.py - Loads chatbot settings from .env and CLI overrides

This module reads the environment file, turns the values into a validated
AppConfig object, and keeps the rest of the app free from direct env parsing.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

from ai_chatbot.logger import get_logger
from ai_chatbot.models import AppConfig

logger = get_logger()


def _parse_bool(value: Optional[str], default: bool = True) -> bool:
    """Parse a string boolean from the environment."""
    if value is None:
        return default

    cleaned_value = value.strip().lower()

    if cleaned_value in {"1", "true", "yes", "on"}:
        return True

    if cleaned_value in {"0", "false", "no", "off"}:
        return False

    logger.warning(f"Invalid THINKING value '{value}', using default {default}")
    return default


def load_config(provider: str | None = None, thinking: bool | None = None) -> AppConfig:
    """Load and validate the app settings used by the chatbot."""
    # Load values from .env so users can change settings without touching code.
    load_dotenv(override=False)

    provider_name = (provider or os.getenv("LLM_PROVIDER", "ollama")).strip().lower()
    thinking_enabled = thinking if thinking is not None else _parse_bool(os.getenv("THINKING"), default=True)

    logger.debug(f"Loading {provider_name.title()} configuration from .env file...")

    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "gemma3:4b")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    timeout_str = os.getenv("REQUEST_TIMEOUT", "30")
    max_tokens_str = os.getenv("MAX_TOKENS", "2048")
    max_retries_str = os.getenv("MAX_RETRIES", "3")
    temperature_str = os.getenv("TEMPERATURE", "0.7")
    log_file = os.getenv("LOG_FILE", "logs/chatbot.log")
    log_level = os.getenv("LOG_LEVEL", "INFO")
    system_prompt = os.getenv(
        "SYSTEM_PROMPT",
        "You are a helpful, concise AI assistant.",
    )

    model_name = gemini_model if provider_name == "gemini" else ollama_model

    # Environment variables arrive as text, so we safely convert the numeric ones.
    try:
        timeout = int(timeout_str)
    except ValueError:
        logger.warning(f"Invalid REQUEST_TIMEOUT value '{timeout_str}', using default 30")
        timeout = 30

    try:
        max_tokens = int(max_tokens_str)
    except ValueError:
        logger.warning(f"Invalid MAX_TOKENS value '{max_tokens_str}', using default 2048")
        max_tokens = 2048

    try:
        max_retries = int(max_retries_str)
    except ValueError:
        logger.warning(f"Invalid MAX_RETRIES value '{max_retries_str}', using default 3")
        max_retries = 3

    try:
        temperature = float(temperature_str)
    except ValueError:
        logger.warning(f"Invalid TEMPERATURE value '{temperature_str}', using default 0.7")
        temperature = 0.7

    try:
        config = AppConfig(
            provider=provider_name,
            thinking=thinking_enabled,
            ollama_host=ollama_host,
            model_name=model_name,
            gemini_api_key=gemini_api_key,
            timeout=timeout,
            max_tokens=max_tokens,
            max_retries=max_retries,
            temperature=temperature,
            log_file=log_file,
            log_level=log_level,
            system_prompt=system_prompt,
        )

        logger.info("Configuration loaded successfully")
        logger.info(f"   Provider: {config.provider}")
        logger.info(f"   Thinking mode: {'enabled' if config.thinking else 'disabled'}")
        logger.info(f"   Ollama host: {config.ollama_host}")
        logger.info(f"   Model: {config.model_name}")
        logger.info(f"   Timeout: {config.timeout}s")
        logger.info(f"   Max retries: {config.max_retries}")
        logger.info(f"   Temperature: {config.temperature}")

        return config

    except Exception as e:
        # This message is written in simple language so beginners know what to fix.
        logger.error(f"Configuration error: {e}")
        print(f"\nCONFIGURATION ERROR:\n{e}")
        print("\nPlease check your .env file:")
        print("   1. Make sure LLM_PROVIDER is set to ollama or gemini")
        if provider_name == "gemini":
            print("   2. Make sure GEMINI_API_KEY is set")
            print("   3. Make sure GEMINI_MODEL is set to a valid Gemini model")
        else:
            print("   2. Make sure OLLAMA_HOST is set to the local server address")
            print("   3. Make sure OLLAMA_MODEL is set to gemma3:4b")
        print("   4. See .env.example for the correct settings\n")

        import sys

        sys.exit(1)
