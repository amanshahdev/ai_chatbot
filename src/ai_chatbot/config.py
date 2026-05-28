"""
config.py - Loads local Ollama settings from .env

This module reads the environment file, turns the values into a validated
AppConfig object, and keeps the rest of the app free from direct env parsing.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from ai_chatbot.logger import get_logger
from ai_chatbot.models import AppConfig

logger = get_logger()


def load_config() -> AppConfig:
    """Load and validate the app settings used by the chatbot."""
    # Load values from .env so users can change settings without touching code.
    load_dotenv(override=False)

    logger.debug("Loading Ollama configuration from .env file...")

    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "gemma3:4b")
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
            ollama_host=ollama_host,
            model_name=model_name,
            timeout=timeout,
            max_tokens=max_tokens,
            max_retries=max_retries,
            temperature=temperature,
            log_file=log_file,
            log_level=log_level,
            system_prompt=system_prompt,
        )

        logger.info("Configuration loaded successfully")
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
        print("   1. Make sure OLLAMA_HOST is set to the local server address")
        print("   2. Make sure OLLAMA_MODEL is set to gemma3:4b")
        print("   3. See .env.example for the correct local Ollama settings\n")

        import sys

        sys.exit(1)
