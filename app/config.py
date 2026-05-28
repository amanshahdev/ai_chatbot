"""
config.py - Configuration Manager

This file handles loading and validating all application settings.

The flow is:
1. Read the .env file using python-dotenv
2. Pass the values through our AppConfig Pydantic model for validation
3. Return a single config object used throughout the app

Think of it as the "settings page" of our app.
"""

# os module lets us read environment variables (the values loaded from .env)
import os

# load_dotenv reads the .env file and puts the values into the environment
# After calling load_dotenv(), os.getenv("GEMINI_API_KEY") works!
from dotenv import load_dotenv

# Import our Pydantic validation model
from app.models import AppConfig

# Import our logger (for logging config events)
from app.logger import get_logger

# Get the logger — if setup_logger() was called earlier, this returns that logger
# If not, it returns a basic unconfigured logger
logger = get_logger()


def load_config() -> AppConfig:
    """
    Loads configuration from the .env file and validates it.
    
    Steps:
    1. Find and load the .env file
    2. Read each environment variable with os.getenv()
    3. Create AppConfig (which validates everything automatically)
    4. Return the validated config
    
    Returns:
        AppConfig: A validated configuration object
    
    Raises:
        SystemExit: If configuration is invalid (we exit immediately — can't run without config)
    """
    
    # ── STEP 1: Load the .env file ──
    # load_dotenv() looks for a file called ".env" in the current directory
    # and parent directories. It reads the file and sets environment variables.
    # override=False means: don't override variables that are already set
    # (useful for production environments where vars are set directly)
    load_dotenv(override=False)
    
    logger.debug("Loading configuration from .env file...")
    
    # ── STEP 2: Read environment variables ──
    # os.getenv("KEY") reads the value of environment variable "KEY"
    # os.getenv("KEY", "default") returns "default" if KEY is not set
    
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    timeout_str = os.getenv("REQUEST_TIMEOUT", "30")
    max_tokens_str = os.getenv("MAX_TOKENS", "2048")
    max_retries_str = os.getenv("MAX_RETRIES", "3")
    log_file = os.getenv("LOG_FILE", "logs/chatbot.log")
    log_level = os.getenv("LOG_LEVEL", "INFO")
    
    # ── STEP 3: Convert string values to proper types ──
    # Environment variables are ALWAYS strings (text)
    # We need to convert "30" (string) to 30 (integer)
    # int() converts a string to an integer
    # We use try/except in case someone writes "thirty" instead of "30"
    
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
    
    # ── STEP 4: Create and validate the config using Pydantic ──
    # AppConfig() will immediately validate all the values
    # If anything is wrong, it raises a ValidationError
    
    try:
        config = AppConfig(
            gemini_api_key=gemini_api_key,
            model_name=model_name,
            timeout=timeout,
            max_tokens=max_tokens,
            max_retries=max_retries,
            log_file=log_file,
            log_level=log_level,
        )
        
        # Log that config was loaded successfully
        # We NEVER log the actual API key (security!)
        # Instead we show the first 8 characters + "..."
        key_preview = config.gemini_api_key[:8] + "..." if len(config.gemini_api_key) > 8 else "***"
        logger.info("Configuration loaded successfully")
        logger.info(f"   Model: {config.model_name}")
        logger.info(f"   API Key: {key_preview}")
        logger.info(f"   Timeout: {config.timeout}s")
        logger.info(f"   Max Retries: {config.max_retries}")
        
        return config
    
    except Exception as e:
        # If validation fails, log the error and exit
        # We can't continue without valid configuration
        logger.error(f"Configuration error: {e}")
        
        # Print to console too (in case logging isn't working yet)
        print(f"\nCONFIGURATION ERROR:\n{e}")
        print("\nPlease check your .env file:")
        print("   1. Make sure .env file exists in the project folder")
        print("   2. Make sure GEMINI_API_KEY is set with your real API key")
        print("   3. Get a free API key at: https://aistudio.google.com/app/apikey")
        print("   4. See .env.example for the correct format\n")
        
        # Exit the program with error code 1
        # exit(1) = exit with an error (0 = success, 1 = error)
        import sys
        sys.exit(1)
