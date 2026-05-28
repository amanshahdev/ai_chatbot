"""
logger.py - The Application's Diary System

Think of logging like keeping a diary for your application.
Every important event gets written down with a timestamp.

Why logging matters:
- When something breaks at 3 AM, the log file tells you exactly what went wrong
- You can see every request and response that happened
- You can track errors without the app crashing
- It's professional — real apps always have logs

Log Levels (from least to most severe):
DEBUG   → Very detailed info (for developers debugging code)
INFO    → Normal operation messages ("User sent message", "Got response")
WARNING → Something unexpected but not breaking ("Retrying request")
ERROR   → Something broke but app continues ("API call failed, retrying")
CRITICAL → App is about to crash ("Cannot connect to API at all")
"""

# Python's built-in logging module — no installation needed
import logging

# For creating folder paths that work on both Windows and Mac/Linux
import os

# For getting current date/time (for log timestamps)
from datetime import datetime


def setup_logger(
    name: str = "ai_chatbot",
    log_file: str = "logs/chatbot.log",
    level: str = "INFO"
) -> logging.Logger:
    """
    Creates and configures our application logger.
    
    This function sets up TWO handlers:
    1. Console Handler: prints colored logs to your terminal
    2. File Handler: writes all logs to a file for later review
    
    Args:
        name: The name of the logger (used to identify it)
        log_file: Where to save the log file
        level: Minimum severity level to log (DEBUG/INFO/WARNING/ERROR)
    
    Returns:
        A configured Logger object ready to use
    """
    
    # ── STEP 1: Create the logs directory if it doesn't exist ──
    # os.path.dirname gets the folder part of the path
    # e.g., "logs/chatbot.log" → "logs"
    log_dir = os.path.dirname(log_file)
    if log_dir:  # Only create directory if there is one in the path
        os.makedirs(log_dir, exist_ok=True)
        # exist_ok=True means "don't error if folder already exists"
    
    # ── STEP 2: Get (or create) a logger with our name ──
    # logging.getLogger() returns the same logger if you call it twice with the same name
    # This means we can call setup_logger from different files and always get the same one
    logger = logging.getLogger(name)
    
    # Convert the level string to Python's internal level number
    # "INFO" → 20, "DEBUG" → 10, "ERROR" → 40, etc.
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Prevent duplicate log messages if setup_logger is called multiple times
    # handlers = the destinations where logs are sent (console, file, etc.)
    if logger.handlers:
        logger.handlers.clear()  # Remove any existing handlers
    
    # ── STEP 3: Define the format for log messages ──
    # Each log message will look like:
    # 2024-01-15 14:30:25 | INFO     | app.client:42 | Sending request to Gemini
    #
    # %(asctime)s    = timestamp
    # %(levelname)-8s = log level, padded to 8 characters
    # %(name)s       = logger name
    # %(filename)s   = which Python file this came from
    # %(lineno)d     = which line number
    # %(message)s    = the actual log message
    
    file_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"  # How to format the date/time
    )
    
    # Console format is simpler (no file/line info — we don't need it in terminal)
    console_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"  # Just time for console (shorter)
    )
    
    # ── STEP 4: Create FILE handler ──
    # This writes logs to the log file
    # mode="a" means "append" — add to existing file instead of overwriting
    # encoding="utf-8" means support international characters
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(log_level)       # Log everything at or above our level
    file_handler.setFormatter(file_format)  # Use the detailed format
    
    # ── STEP 5: Create CONSOLE handler ──
    # This prints logs to the terminal
    # StreamHandler with no args defaults to sys.stderr
    # but we'll direct it to sys.stdout so it shows up normally
    import sys
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # Only show WARNING and above in console
    # Why WARNING? INFO messages would spam the terminal during normal chat
    # Errors and warnings are important enough to always show
    console_handler.setFormatter(console_format)
    
    # ── STEP 6: Attach handlers to the logger ──
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log a startup message so we know the logger is working
    logger.info("=" * 60)
    logger.info(f"Logger initialized | Level: {level} | File: {log_file}")
    logger.info(f"Session started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    return logger


def get_logger(name: str = "ai_chatbot") -> logging.Logger:
    """
    Gets the existing logger by name.
    
    Call this in every file that needs logging:
        logger = get_logger()
    
    Because we call setup_logger once at startup (in main.py),
    every subsequent get_logger() call returns the already-configured logger.
    """
    return logging.getLogger(name)
