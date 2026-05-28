"""
models.py - Data Blueprints for the Ollama chatbot

These models describe the main data the app works with.
They keep the configuration and chat messages neat and easy to validate.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AppConfig(BaseModel):
    """Stores the app settings loaded from the .env file."""

    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Local Ollama server URL",
    )
    model_name: str = Field(
        default="gemma3:4b",
        description="Local Ollama model name",
    )
    timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Request timeout in seconds",
    )
    max_tokens: int = Field(
        default=2048,
        ge=1,
        le=8192,
        description="Maximum number of tokens to generate",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=5,
        description="How many times to retry a failed request",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the model",
    )
    log_file: str = Field(default="logs/chatbot.log", description="Path to log file")
    log_level: str = Field(default="INFO", description="Logging level")
    system_prompt: str = Field(
        default="You are a helpful, concise AI assistant.",
        description="Optional system prompt sent before each chat",
    )

    @field_validator("ollama_host")
    @classmethod
    def validate_ollama_host(cls, value: str) -> str:
        """Make sure the app only talks to a local Ollama server."""
        cleaned_value = value.strip()
        parsed = urlparse(cleaned_value)

        if parsed.scheme != "http":
            raise ValueError("OLLAMA_HOST must use http:// and point to a local server")

        if parsed.hostname not in {"localhost", "127.0.0.1"}:
            raise ValueError(
                "OLLAMA_HOST must point to the local Ollama server, for example http://localhost:11434"
            )

        return cleaned_value

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        """Keep the model name tidy and reject empty values."""
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("model_name cannot be empty")
        return cleaned_value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Make sure the logging level is one Python understands."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        cleaned_value = value.strip().upper()
        if cleaned_value not in valid_levels:
            raise ValueError(f"log_level must be one of: {sorted(valid_levels)}")
        return cleaned_value

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, value: str) -> str:
        """Trim the system prompt so the chat does not carry extra spaces."""
        return value.strip()


class ChatMessage(BaseModel):
    """Represents one chat message in Ollama's format."""

    role: Literal["user", "assistant"] = Field(
        ..., description="Who sent the message"
    )
    content: str = Field(..., min_length=1, description="Message text")

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Remove extra spaces and reject empty messages."""
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("Message content cannot be empty or just whitespace")
        return cleaned_value


class OllamaResponseMessage(BaseModel):
    """Represents the message block returned by Ollama."""

    model_config = ConfigDict(from_attributes=True)

    role: str = Field(default="assistant", description="Role of the response message")
    content: str = Field(default="", description="Text returned by the model")


class OllamaChatResponse(BaseModel):
    """Validates the main chat response shape returned by Ollama."""

    model_config = ConfigDict(from_attributes=True)

    message: OllamaResponseMessage = Field(..., description="Assistant message")
    done: bool = Field(default=True, description="Whether the response is finished")
