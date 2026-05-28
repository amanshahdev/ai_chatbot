"""
main.py - The Entry Point

This is the FIRST file that runs when you type: python -m app.main

Think of it as the "front door" of the application.
Its only job is to:
1. Set up basic logging (before config loads)
2. Call the main async function
3. Handle top-level exceptions
4. Exit cleanly

Why is this separate from chat.py?
- Separation of concerns: entry point logic ≠ chat logic
- Makes testing easier (we can import chat.py without "running" the app)
- Clean architecture practice
"""

# asyncio is Python's async framework
# asyncio.run() is the way to start an async program from regular Python code
import asyncio

# sys for system exit and error handling
import sys

# os for path operations
import os

# Import our main chat runner function
from app.chat import run_chatbot

# Import logger setup (we set up a basic logger before config loads)
from app.logger import setup_logger


def main() -> None:
    """
    The main entry point.
    
    This is a regular (not async) function.
    It uses asyncio.run() to start the async world.
    
    asyncio.run() does three things:
    1. Creates an event loop (the engine that runs async code)
    2. Runs our async function (run_chatbot) to completion
    3. Shuts down the event loop cleanly
    """
    
    # ── Set up a basic logger before configuration loads ──
    # This catches any startup errors before the full logger is configured
    # We use default settings here; the full config will replace this
    os.makedirs("logs", exist_ok=True)  # Create logs folder if missing
    setup_logger(
        name="ai_chatbot",
        log_file="logs/chatbot.log",
        level="INFO"
    )
    
    try:
        # ── Start the async event loop and run the chatbot ──
        # asyncio.run() is the standard way to run async code from a sync context
        asyncio.run(run_chatbot())
    
    except KeyboardInterrupt:
        # User pressed Ctrl+C at the very start (before the chat loop)
        print("\n\nInterrupted. Goodbye!\n")
        sys.exit(0)  # Exit successfully (code 0)
    
    except SystemExit as e:
        # sys.exit() was called somewhere (e.g., in load_config on error)
        # We just re-raise it to let Python handle the exit
        raise
    
    except Exception as e:
        # Truly unexpected top-level errors
        print(f"\nFatal error: {type(e).__name__}: {e}")
        print("Check logs/chatbot.log for details.")
        sys.exit(1)  # Exit with error code 1


# ── This block only runs when the file is executed directly ──
# It does NOT run when another file imports this file
# Example:
#   python main.py    → __name__ == "__main__" → main() IS called
#   import main       → __name__ == "main"      → main() is NOT called
#
# This is a Python best practice for all entry-point files.
if __name__ == "__main__":
    main()
