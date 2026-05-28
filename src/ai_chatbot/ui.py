"""
ui.py - User Interface Helpers

These helpers keep the terminal output tidy and beginner-friendly.
They are unchanged in spirit, but now they talk about local Ollama instead of Gemini.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme

CHATBOT_THEME = Theme(
    {
        "user.label": "bold",
        "ai.label": "bold",
        "error": "bold red",
        "dim": "dim",
        "success": "green",
        "info": "cyan",
    }
)

console = Console(theme=CHATBOT_THEME)


def clean_markdown(text: str) -> str:
    """Remove a few simple Markdown markers so terminal output stays readable."""
    if not isinstance(text, str):
        return str(text)

    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\*(.*?)\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"`(.*?)`", r"\1", text, flags=re.DOTALL)
    return text


def print_welcome_banner() -> None:
    """Show the app banner with the local Ollama branding."""
    console.clear()
    banner = Text()
    banner.append("AI Chatbot\n", style="bold")
    banner.append("Powered by local Ollama\n", style="dim")
    panel = Panel(Align.center(banner), padding=(1, 2))
    console.print(panel)


def print_help_text() -> None:
    """Show the main commands a beginner can use."""
    lines = [
        "How to use:",
        "  - Type your message and press Enter to chat",
        "  - Type exit, quit, or bye to stop",
        "  - Type clear to clear the screen",
        "  - Type history to see past messages",
        "  - Type reset to start a new conversation",
    ]
    console.print(Panel(Text("\n".join(lines)), title="Commands"))


def print_user_message(message: str) -> None:
    """Print the user's message with a small timestamp."""
    timestamp = datetime.now().strftime("%H:%M")
    content = Text(message)
    console.print(Panel(content, title=f"You {timestamp}", title_align="right"))


def print_ai_message(message: str, model_name: str = "Ollama") -> None:
    """Print the assistant response in a clean panel."""
    timestamp = datetime.now().strftime("%H:%M")
    content = Text(clean_markdown(message))
    console.print(Panel(content, title=f"AI - {model_name} {timestamp}", title_align="left"))
    console.print()


def print_error(message: str, details: Optional[str] = None) -> None:
    """Print an error panel with an optional explanation."""
    error_text = Text()
    error_text.append("Error: ", style="error")
    error_text.append(message)
    if details:
        error_text.append(f"\n\nDetails: {details}", style="dim")
    console.print(Panel(error_text, border_style="red"))


def print_system_message(message: str) -> None:
    """Print a small info line for normal status updates."""
    console.print(f"INFO: {message}", style="info")


def print_success(message: str) -> None:
    """Print a green success line."""
    console.print(f"OK: {message}", style="success")


def print_divider(title: str = "") -> None:
    """Print a horizontal divider line."""
    if title:
        console.print(Rule(title=title))
    else:
        console.print(Rule())


def print_chat_history(history: list) -> None:
    """Print the conversation history from the current session."""
    if not history:
        console.print("No conversation history yet.")
        return

    console.print()
    console.print(Panel(Text("Conversation History")))
    console.print()

    for index, message in enumerate(history, 1):
        role = message.get("role", "assistant")
        content = message.get("content")

        # This fallback keeps older history formats readable if they ever appear.
        if content is None:
            try:
                content = message["parts"][0]["text"]
            except Exception:
                content = str(message)

        cleaned_content = clean_markdown(content)
        prefix = f"#{index} You:" if role == "user" else f"#{index} AI:"
        console.print(f"{prefix} {cleaned_content[:200]}{'...' if len(cleaned_content) > 200 else ''}")

    console.print()


def get_user_input() -> str:
    """Read a line from the terminal and turn it into a clean string."""
    try:
        user_input = console.input("You> ")
        return user_input.strip()
    except (KeyboardInterrupt, EOFError):
        return "exit"


def print_thinking_start() -> None:
    """Show a short message while the model is generating text."""
    console.print("Thinking...", style="dim")


def print_goodbye() -> None:
    """Print a friendly goodbye line."""
    console.print("Goodbye. See you next time.")


def print_retry_message(attempt: int, max_retries: int, wait_seconds: float) -> None:
    """Tell the user that the app is trying the request again."""
    console.print(f"Retry {attempt}/{max_retries} - waiting {wait_seconds:.1f}s...", style="info")
