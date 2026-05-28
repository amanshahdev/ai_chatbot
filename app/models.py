"""
models.py - Data Blueprints (Pydantic Models)

Think of Pydantic models like FORMS that must be filled out correctly.
If you try to submit a form with wrong information, Pydantic rejects it immediately.

This file defines the shape/structure of all data in our application:
- What a user message looks like
- What we send to the Gemini API
- What we expect to receive back
- What valid configuration looks like
"""

# pydantic provides the BaseModel class — the foundation for all our models
# Field() lets us add rules and descriptions to each field
# field_validator lets us write custom validation logic
from pydantic import BaseModel, Field, field_validator

# Optional means the value can be None (missing/empty)
# List means a list of items, Dict means a dictionary (key-value pairs)
from typing import Optional, List, Dict, Any


# ============================================================
# CONFIGURATION MODEL
# Validates all settings loaded from the .env file
# ============================================================
class AppConfig(BaseModel):
    """
    This model validates our application configuration.
    
    When we load settings from .env, we pass them through this model.
    If anything is wrong (like a missing API key), we get a clear error.
    """
    
    # The Gemini API key — required, must not be empty
    # min_length=10 means the key must be at least 10 characters long
    gemini_api_key: str = Field(..., min_length=10, description="Google Gemini API Key")
    
    # Which Gemini model to use — has a default value if not set in .env
    model_name: str = Field(
        default="gemini-1.5-flash",  # free tier model
        description="Gemini model to use"
    )
    
    # How many seconds to wait before giving up on an API call
    # Must be between 5 and 120 seconds
    timeout: int = Field(default=30, ge=5, le=120, description="Request timeout in seconds")
    
    # Maximum number of tokens (words roughly) in the AI's response
    max_tokens: int = Field(default=2048, ge=100, le=8192, description="Max response tokens")
    
    # How many times to retry if the API fails
    max_retries: int = Field(default=3, ge=1, le=5, description="Max retry attempts")
    
    # The log file path
    log_file: str = Field(default="logs/chatbot.log", description="Path to log file")
    
    # Log level (DEBUG, INFO, WARNING, ERROR)
    log_level: str = Field(default="INFO", description="Logging level")
    
    @field_validator("gemini_api_key")
    @classmethod
    def validate_api_key(cls, value: str) -> str:
        """
        Custom validator: checks that the API key isn't the placeholder value.
        'cls' refers to the class itself (AppConfig).
        'value' is what was passed in for gemini_api_key.
        """
        # Strip whitespace from both ends (in case of copy-paste accidents)
        value = value.strip()
        
        # Make sure it's not still the example placeholder
        if value == "your_gemini_api_key_here" or value == "":
            raise ValueError(
                "❌ Please set a real Gemini API key in your .env file!\n"
                "   Get one free at: https://aistudio.google.com/app/apikey"
            )
        return value
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Makes sure log_level is one of the valid Python logging levels."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        value = value.upper()  # Convert to uppercase (so 'info' → 'INFO')
        if value not in valid_levels:
            raise ValueError(f"log_level must be one of: {valid_levels}")
        return value


# ============================================================
# CHAT MESSAGE MODEL
# Represents a single message in the conversation
# ============================================================
class ChatMessage(BaseModel):
    """
    A single message in the conversation.
    
    role = who sent it: "user" (you) or "model" (the AI)
    content = what was said
    
    Example:
        ChatMessage(role="user", content="What is Python?")
        ChatMessage(role="model", content="Python is a programming language...")
    """
    
    # Who sent the message - must be "user" or "model"
    role: str = Field(..., description="Who sent this message: 'user' or 'model'")
    
    # The actual message text - must not be empty
    content: str = Field(..., min_length=1, description="The message text")
    
    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        """Makes sure role is either 'user' or 'model'."""
        valid_roles = ["user", "model"]
        if value not in valid_roles:
            raise ValueError(f"role must be one of: {valid_roles}")
        return value
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Removes leading/trailing whitespace and checks it's not empty."""
        value = value.strip()
        if not value:
            raise ValueError("Message content cannot be empty or just whitespace")
        return value


# ============================================================
# API REQUEST MODEL
# What we send to the Gemini API
# ============================================================
class GeminiRequest(BaseModel):
    """
    This represents the data we send to Gemini's API.
    
    Gemini expects messages in a specific format.
    This model ensures we always send the right format.
    """
    
    # The conversation history — list of messages
    # Each message is a dict like: {"role": "user", "parts": [{"text": "Hello"}]}
    contents: List[Dict[str, Any]] = Field(..., description="Conversation history for Gemini")
    
    # Settings for how Gemini should respond
    generation_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "maxOutputTokens": 2048,
            "temperature": 0.7,     # 0 = very precise, 1 = more creative
            "topP": 0.8,            # Controls diversity of responses
        },
        description="Gemini generation settings"
    )


# ============================================================
# API RESPONSE MODELS
# What we receive back from Gemini
# ============================================================
class GeminiResponsePart(BaseModel):
    """
    A single 'part' of a Gemini response.
    Usually just contains text.
    """
    text: str = Field(default="", description="The text content of this part")


class GeminiResponseContent(BaseModel):
    """
    The content block inside a Gemini response candidate.
    Contains the actual message parts.
    """
    parts: List[GeminiResponsePart] = Field(
        default_factory=list,
        description="List of content parts"
    )
    role: str = Field(default="model", description="Who generated this content")


class GeminiResponseCandidate(BaseModel):
    """
    A single response candidate from Gemini.
    Gemini can generate multiple candidates (different possible answers),
    but we usually only ask for one.
    """
    content: GeminiResponseContent = Field(..., description="The response content")
    
    # finishReason tells us why Gemini stopped generating
    # "STOP" = normal completion, "MAX_TOKENS" = hit the token limit
    finishReason: Optional[str] = Field(default="STOP", description="Why generation stopped")


class GeminiResponse(BaseModel):
    """
    The full response from the Gemini API.
    This is what we parse when Gemini sends us back an answer.
    """
    candidates: List[GeminiResponseCandidate] = Field(
        ...,
        min_length=1,
        description="List of response candidates"
    )
    
    def get_text(self) -> str:
        """
        Helper method to easily extract the response text.
        
        Instead of navigating the nested structure every time:
            response.candidates[0].content.parts[0].text
        
        We can just call:
            response.get_text()
        """
        try:
            # Navigate to the first candidate's first part's text
            return self.candidates[0].content.parts[0].text
        except (IndexError, AttributeError):
            # If anything goes wrong in navigation, return empty string
            return ""


# ============================================================
# ERROR RESPONSE MODEL
# Represents an error from the API
# ============================================================
class APIError(BaseModel):
    """
    Represents an error response from the API.
    When the API returns an error, it usually sends a JSON object
    with error details. This model captures that.
    """
    
    # Error code (like 400, 401, 429, 500)
    status_code: int = Field(..., description="HTTP status code")
    
    # Human-readable error message
    message: str = Field(..., description="Error description")
    
    # Optional: more details about the error
    details: Optional[str] = Field(default=None, description="Additional error details")
