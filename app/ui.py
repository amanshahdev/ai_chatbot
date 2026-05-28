"""
ui.py - User Interface Helpers

Simple command-line presentation helpers. This module provides
plain, minimal terminal output utilities used by the CLI.
"""

from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.align import Align
from rich.text import Text
from rich.theme import Theme
import re


# Minimal theme for readable output
CHATBOT_THEME = Theme({
    "user.label": "bold",
    "ai.label": "bold",
    "error": "bold red",
    "dim": "dim",
    "success": "green",
    "info": "cyan",
})

console = Console(theme=CHATBOT_THEME)


def clean_markdown(text: str) -> str:
    """Remove simple Markdown markers (**, *, `) from text."""
    if not isinstance(text, str):
        return str(text)

    # Remove bold **text**
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.DOTALL)
    # Remove italic *text*
    text = re.sub(r"\*(.*?)\*", r"\1", text, flags=re.DOTALL)
    # Remove inline code `code`
    text = re.sub(r"`(.*?)`", r"\1", text, flags=re.DOTALL)
    return text


def print_welcome_banner() -> None:
    """Print a minimal welcome banner."""
    console.clear()
    banner = Text()
    banner.append("AI Chatbot\n", style="bold")
    banner.append("Powered by Google Gemini\n", style="dim")
    panel = Panel(Align.center(banner), padding=(1, 2))
    console.print(panel)


def print_help_text() -> None:
    """Print simple usage instructions."""
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
    """Print the user's message with a timestamp."""
    timestamp = datetime.now().strftime("%H:%M")
    content = Text(message)
    console.print(Panel(content, title=f"You {timestamp}", title_align="right"))


def print_ai_message(message: str, model_name: str = "Gemini") -> None:
    """Print the AI response as plain text (no markdown)."""
    timestamp = datetime.now().strftime("%H:%M")
    content = Text(clean_markdown(message))
    console.print(Panel(content, title=f"AI - {model_name} {timestamp}", title_align="left"))
    console.print()


def print_error(message: str, details: Optional[str] = None) -> None:
    """Print an error message."""
    error_text = Text()
    error_text.append("Error: ", style="error")
    error_text.append(message)
    if details:
        error_text.append(f"\n\nDetails: {details}", style="dim")
    console.print(Panel(error_text, border_style="red"))


def print_system_message(message: str) -> None:
    """Print a system/info message."""
    console.print(f"INFO: {message}", style="info")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"OK: {message}", style="success")


def print_divider(title: str = "") -> None:
    """Print a horizontal rule divider."""
    if title:
        console.print(Rule(title=title))
    else:
        console.print(Rule())


def print_chat_history(history: list) -> None:
    """Print conversation history in a simple format."""
    if not history:
        console.print("No conversation history yet.")
        return

    console.print()
    console.print(Panel(Text("Conversation History")))
    console.print()

    for i, message in enumerate(history, 1):
        is_user = message.get("role") == "user"
        try:
            text = message["parts"][0]["text"]
        except Exception:
            text = str(message)
        text = clean_markdown(text)
        prefix = f"#{i} You:" if is_user else f"#{i} AI:"
        console.print(f"{prefix} {text[:200]}{'...' if len(text) > 200 else ''}")

    console.print()


def get_user_input() -> str:
    """Get user input from the terminal with a simple prompt."""
    try:
        user_input = console.input("You> ")
        return user_input.strip()
    except (KeyboardInterrupt, EOFError):
        return "exit"


def print_thinking_start() -> None:
    """Show a simple thinking indicator."""
    console.print("Thinking...", style="dim")


def print_goodbye() -> None:
    """Print a simple goodbye message."""
    console.print("Goodbye. See you next time.")


def print_retry_message(attempt: int, max_retries: int, wait_seconds: float) -> None:
    console.print(f"Retry {attempt}/{max_retries} — waiting {wait_seconds:.1f}s...", style="info")
