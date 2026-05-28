"""
chat.py - The Chat Loop (The Heart of the Application)

This file manages the entire conversation experience:
1. Displays the welcome screen
2. Waits for user input
3. Sends messages to the AI
4. Displays responses
5. Maintains conversation history
6. Handles commands (exit, clear, history, reset)
7. Catches and displays errors nicely

Think of this as the "conductor" of the orchestra — it coordinates
all the other pieces (client, UI, logger) to work together.
"""

# asyncio is Python's async framework — we need it to run async code
import asyncio

# sys for system operations (like exiting the program)
import sys

# For type hints
from typing import List, Dict, Optional

# Import our custom classes and functions
from app.config import load_config
from app.client import GeminiClient, GeminiAPIError, GeminiTimeoutError, GeminiNetworkError
from app.logger import setup_logger, get_logger
from app.models import AppConfig
from app.ui import (
    print_welcome_banner,
    print_help_text,
    print_user_message,
    print_ai_message,
    print_error,
    print_system_message,
    print_success,
    print_divider,
    print_chat_history,
    get_user_input,
    print_goodbye,
    print_thinking_start,
    print_retry_message,
    console,
)
from app.ui import clean_markdown

# Get our logger
logger = get_logger()

# Commands that will exit the chatbot
EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye", "q"}

# Commands that do special things
SPECIAL_COMMANDS = {
    "clear": "Clear the terminal screen",
    "history": "Show conversation history",
    "reset": "Start a new conversation (clear history)",
    "help": "Show this help message",
}


class ChatSession:
    """
    Manages a single chat session.
    
    A "session" = one conversation from start to finish.
    This class keeps track of:
    - The conversation history (all messages so far)
    - The AI client (for sending messages)
    - The configuration
    
    Why a class? Because we need to maintain "state" (data that persists
    between function calls). A class stores this state in attributes (self.x).
    """
    
    def __init__(self, config: AppConfig, client: GeminiClient):
        """
        Initialize a new chat session.
        
        Args:
            config: Validated application configuration
            client: The GeminiClient for API communication
        """
        self.config = config
        self.client = client
        
        # Conversation history — stores all messages
        # Format: [{"role": "user", "parts": [{"text": "..."}]}, ...]
        # We use Gemini's native format directly
        self.history: List[Dict] = []
        
        # Counter for how many messages have been exchanged
        self.message_count: int = 0
        
        # Whether streaming mode is enabled
        self.streaming_enabled: bool = True
        
        logger.info("New chat session started")
    
    def add_to_history(self, role: str, text: str) -> None:
        """
        Adds a message to the conversation history.
        
        Args:
            role: "user" or "model"
            text: The message content
        """
        self.history.append({
            "role": role,
            "parts": [{"text": text}]
        })
        logger.debug(f"Added to history: role={role}, len={len(text)}")
    
    def reset_history(self) -> None:
        """Clears all conversation history to start fresh."""
        self.history = []
        self.message_count = 0
        logger.info("Conversation history cleared")
    
    async def process_message(self, user_input: str) -> bool:
        """
        Processes a single user message.
        
        Returns:
            True if we should continue chatting
            False if the user wants to exit
        """
        # ── Handle exit commands ──
        if user_input.lower() in EXIT_COMMANDS:
            logger.info("User requested exit")
            return False  # Signal to stop the loop
        
        # ── Handle special commands ──
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
        
        # ── Reject empty input ──
        if not user_input.strip():
            print_system_message("Please type something! (or type 'exit' to quit)")
            return True
        
        # ── Log the user's message ──
        logger.info(f"User message #{self.message_count + 1}: '{user_input[:80]}...'")
        
        # Print the user's message in the UI
        print_user_message(user_input)
        
        # ── Send to AI and get response ──
        try:
            if self.streaming_enabled:
                # STREAMING: print text as it arrives (word by word effect)
                await self._process_streaming(user_input)
            else:
                # NON-STREAMING: wait for full response then print it
                await self._process_standard(user_input)
            
            self.message_count += 1
            return True
        
        except GeminiAPIError as e:
            # API-level errors (bad key, rate limit, server error, etc.)
            logger.error(f"API error for message #{self.message_count}: {e}")
            
            if e.status_code == 401:
                print_error(
                    "Invalid API Key",
                    "Please check GEMINI_API_KEY in your .env file.\n"
                    "Get a key at: https://aistudio.google.com/app/apikey"
                )
            elif e.status_code == 429:
                print_error(
                    "Rate limit reached",
                    "Too many requests. Please wait a moment and try again."
                )
            elif e.status_code >= 500:
                print_error(
                    "Gemini server error",
                    f"Google's servers may be temporarily down. ({e.message})"
                )
            else:
                print_error(f"API Error ({e.status_code})", e.message)
            
            return True  # Continue chatting (error was handled)
        
        except GeminiTimeoutError as e:
            logger.error(f"Timeout error: {e}")
            print_error(
                "Request timed out",
                f"The AI took too long to respond. Your timeout is set to {self.config.timeout}s.\n"
                "Try again or increase REQUEST_TIMEOUT in .env"
            )
            return True
        
        except GeminiNetworkError as e:
            logger.error(f"Network error: {e}")
            print_error(
                "Network error",
                "Cannot reach the Gemini API. Please check your internet connection."
            )
            return True
        
        except ValueError as e:
            # Input validation errors
            logger.warning(f"Validation error: {e}")
            print_error("Invalid input", str(e))
            return True
        
        except KeyboardInterrupt:
            # User pressed Ctrl+C while waiting for response
            print_system_message("\nInterrupted! Type 'exit' to quit or continue chatting.")
            return True
        
        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"Unexpected error: {type(e).__name__}: {e}", exc_info=True)
            print_error(
                "Unexpected error",
                f"{type(e).__name__}: {str(e)}\n"
                "Check the log file for details."
            )
            return True
    
    async def _process_streaming(self, user_input: str) -> None:
        """
        Handles streaming response — prints text as it arrives.
        
        This gives a "typewriter" effect where you see the AI writing
        in real-time instead of waiting for the full response.
        """
        print()  # Add some space
        
        # Collect all chunks for adding to history later
        full_response_chunks = []
        
        # show_cursor=False hides the blinking cursor during streaming
        # (looks cleaner)
        is_first_chunk = True
        
        async for chunk in self.client.send_message_streaming(user_input, self.history):
            if is_first_chunk:
                # Print the AI label before the first chunk
                console.print("AI is writing...", style="info")
                console.print()
                is_first_chunk = False
            
            # Clean the chunk to remove markdown markers then print
            clean_chunk = clean_markdown(chunk)
            print(clean_chunk, end="", flush=True)
            full_response_chunks.append(clean_chunk)
        
        # Print a newline after streaming is done
        print("\n")
        
        # Combine all chunks into the full response
        full_response = "".join(full_response_chunks)
        
        if full_response:
            # Add both messages to history
            self.add_to_history("user", user_input)
            self.add_to_history("model", full_response)
            logger.info(f"AI response #{self.message_count + 1} received (streaming): {len(full_response)} chars")
        else:
            print_error("Empty response", "The AI returned an empty response. Please try again.")
    
    async def _process_standard(self, user_input: str) -> None:
        """
        Handles standard (non-streaming) response.
        
        Shows a "thinking" indicator, waits for the full response,
        then displays it all at once.
        """
        # Show the user we're waiting for the AI
        print_thinking_start()
        
        # Make the API call and wait for the full response
        response_text = await self.client.send_message(user_input, self.history)
        
        # Clean response for display and history
        cleaned = clean_markdown(response_text)

        # Add both messages to history
        self.add_to_history("user", user_input)
        self.add_to_history("model", cleaned)
        
        # Display the response
        print_ai_message(cleaned, self.config.model_name)
        
        logger.info(f"AI response #{self.message_count + 1} received: {len(response_text)} chars")


async def run_chatbot() -> None:
    """
    The main async function that runs the entire chatbot.
    
    This is an async function (notice 'async def' instead of just 'def').
    It must be called with 'await' or run with asyncio.run().
    
    The flow:
    1. Load configuration
    2. Set up logger
    3. Show welcome screen
    4. Create AI client
    5. Start chat loop
    6. Handle exit gracefully
    """
    
    # ── STEP 1: Load and validate configuration ──
    # This reads .env and validates everything
    # If configuration is invalid, load_config() will exit the program
    config = load_config()
    
    # ── STEP 2: Set up the proper logger with config settings ──
    # Now that we have config, we can set up logging properly
    setup_logger(
        name="ai_chatbot",
        log_file=config.log_file,
        level=config.log_level
    )
    logger_configured = get_logger()
    logger_configured.info("Starting AI Chatbot application")
    
    # ── STEP 3: Show welcome screen ──
    print_welcome_banner()
    print_help_text()
    
    # Show which model we're using
    print_system_message(f"Using model: [bold]{config.model_name}[/]")
    print_divider()
    
    # ── STEP 4: Create the AI client and start chatting ──
    # 'async with' creates the client, and automatically closes it when done
    # This is the "context manager" pattern we explained earlier
    async with GeminiClient(config) as client:
        
        # Create a new chat session
        session = ChatSession(config, client)
        
        logger_configured.info("Chat session started, entering main loop")
        print_system_message("Ready to chat! Type your message below.")
        
        # ── STEP 5: The main chat loop ──
        # This loop keeps running until the user types 'exit'
        # 'while True' = keep looping forever (until we break out of it)
        while True:
            try:
                # Get input from the user
                user_input = get_user_input()
                
                # Process the message
                # process_message returns False when user wants to exit
                should_continue = await session.process_message(user_input)
                
                if not should_continue:
                    # User typed 'exit' — break out of the loop
                    break
            
            except KeyboardInterrupt:
                # User pressed Ctrl+C
                print()
                print_system_message("Ctrl+C detected. Type 'exit' to quit properly, or press Ctrl+C again.")
                try:
                    # Give them a chance to continue
                    # Wait a moment to see if they press Ctrl+C again
                    await asyncio.sleep(0.5)
                except KeyboardInterrupt:
                    # Pressed Ctrl+C twice — force exit
                    break
            
            except EOFError:
                # This happens when stdin is closed (e.g., piped input ended)
                break
        
        # ── STEP 6: Goodbye ──
        logger_configured.info(
            f"Chat session ended. Total messages: {session.message_count}"
        )
        print_goodbye()
