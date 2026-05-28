"""
logger.py - The application's diary system

The logger keeps a record of what the chatbot is doing so problems are easier
when something goes wrong.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime


def setup_logger(
    name: str = "ai_chatbot",
    log_file: str = "logs/chatbot.log",
    level: str = "INFO",
) -> logging.Logger:
    """Create the shared application logger."""
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    if logger.handlers:
        logger.handlers.clear()

    file_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_format)

    import sys

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(console_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 60)
    logger.info(f"Logger initialized | Level: {level} | File: {log_file}")
    logger.info(f"Session started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    return logger


def get_logger(name: str = "ai_chatbot") -> logging.Logger:
    """Get the shared application logger by name."""
    return logging.getLogger(name)
