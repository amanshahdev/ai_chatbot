"""
client.py - The Gemini API Client

This is the most important file — it handles ALL communication with the Gemini AI.

Responsibilities:
1. Build the correct request format for Gemini
2. Send requests asynchronously using httpx.AsyncClient
3. Parse and validate the response
4. Retry on failure (with exponential backoff)
5. Handle all types of errors gracefully
6. Support streaming responses (getting text word-by-word)

Think of this as the "translator + messenger" between your code and Google's AI.
"""

# httpx is like the 'requests' library but supports async/await
# AsyncClient is the async version — it doesn't block while waiting for responses
import httpx

# asyncio is Python's async framework — it manages async operations
import asyncio

# json module parses JSON text into Python dictionaries
import json

# For type hints — makes code easier to understand
from typing import List, Dict, Any, Optional, AsyncGenerator

# tenacity is a retry library — it handles retrying failed operations
# retry = decorator to make a function retry on failure
# stop_after_attempt = stop after N attempts
# wait_exponential = wait longer and longer between retries (1s, 2s, 4s...)
# retry_if_exception_type = only retry for specific exception types
# before_sleep = do something before sleeping between retries
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

# Import our custom models for request/response validation
from app.models import (
    AppConfig,
    ChatMessage,
    GeminiRequest,
    GeminiResponse,
    GeminiResponseCandidate,
    GeminiResponseContent,
    GeminiResponsePart,
    APIError,
)

# Import our logger
from app.logger import get_logger

# Import UI helpers for showing retry messages
from app.ui import print_retry_message, print_system_message

# Get the logger instance
logger = get_logger()


# ============================================================
# CUSTOM EXCEPTION CLASSES
# Custom exceptions help us distinguish different error types
# ============================================================
class GeminiAPIError(Exception):
    """Raised when the Gemini API returns an error response."""
    def __init__(self, status_code: int, message: str, details: str = ""):
        self.status_code = status_code
        self.message = message
        self.details = details
        super().__init__(f"API Error {status_code}: {message}")


class GeminiTimeoutError(Exception):
    """Raised when a request to Gemini times out."""
    pass


class GeminiNetworkError(Exception):
    """Raised when there's a network connectivity issue."""
    pass


class GeminiConfigError(Exception):
    """Raised when configuration is invalid."""
    pass


# ============================================================
# MAIN CLIENT CLASS
# ============================================================
class GeminiClient:
    """
    An async client for communicating with the Google Gemini API.
    
    Usage:
        # Create client
        client = GeminiClient(config)
        
        # Use as async context manager (handles setup and cleanup)
        async with client:
            response = await client.send_message("Hello!", history)
    
    What is a context manager?
    It's like a shop that opens when you enter (async with client:) and
    closes when you leave (end of the 'async with' block).
    The __aenter__ method runs on entry, __aexit__ runs on exit.
    This ensures the httpx client is always properly closed.
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize the client with configuration.
        
        Args:
            config: Validated AppConfig object with all settings
        """
        self.config = config
        
        # The base URL for Gemini API
        # We'll append the model name and action to this
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        # Build the full URL for generating content (non-streaming)
        # f-string: like a template where {variable} gets replaced with the value
        self.generate_url = (
            f"{self.base_url}/models/{config.model_name}:generateContent"
            f"?key={config.gemini_api_key}"
        )
        
        # Build the URL for streaming responses (gets text word-by-word)
        self.stream_url = (
            f"{self.base_url}/models/{config.model_name}:streamGenerateContent"
            f"?key={config.gemini_api_key}&alt=sse"
        )
        
        # httpx.AsyncClient will be created in __aenter__
        # We declare it as None here so the type system knows about it
        self._http_client: Optional[httpx.AsyncClient] = None
        
        logger.debug(f"GeminiClient initialized with model: {config.model_name}")
    
    async def __aenter__(self):
        """
        Called when entering 'async with GeminiClient(...) as client:'
        
        Creates the httpx.AsyncClient with our configuration.
        AsyncClient maintains a connection pool (reuses connections for speed).
        """
        self._http_client = httpx.AsyncClient(
            # timeout: how long to wait for the API to respond
            timeout=httpx.Timeout(
                connect=10.0,           # Time to establish connection
                read=self.config.timeout,  # Time to read the response
                write=10.0,             # Time to send the request
                pool=5.0,               # Time to get a connection from the pool
            ),
            # headers: extra info sent with every request
            # Content-Type tells the API we're sending JSON
            headers={
                "Content-Type": "application/json",
                "User-Agent": "ai-chatbot/1.0.0",
            },
        )
        logger.debug("httpx.AsyncClient created and ready")
        return self  # Return self so we can use it as 'client' in 'async with ... as client'
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Called when leaving the 'async with' block.
        
        Properly closes the httpx client and releases network connections.
        exc_type, exc_val, exc_tb = exception info if an error occurred
        (we don't handle exceptions here, just cleanup)
        """
        if self._http_client:
            await self._http_client.aclose()  # aclose = async close
            logger.debug("httpx.AsyncClient closed")
    
    def _build_contents(self, message: str, history: List[Dict]) -> List[Dict]:
        """
        Formats the conversation history + new message for Gemini's API.
        
        Gemini expects messages in this exact format:
        [
            {"role": "user", "parts": [{"text": "Hello!"}]},
            {"role": "model", "parts": [{"text": "Hi there!"}]},
            {"role": "user", "parts": [{"text": "What is Python?"}]},
        ]
        
        The history already stores messages in this format.
        We just add the new message to the end.
        
        Args:
            message: The new user message to send
            history: Previous conversation messages
        
        Returns:
            List of message dicts in Gemini format
        """
        # Start with the existing history (copy to avoid modifying original)
        contents = list(history)
        
        # Add the new user message
        contents.append({
            "role": "user",
            "parts": [{"text": message}]
        })
        
        return contents
    
    def _build_request_body(self, contents: List[Dict]) -> Dict:
        """
        Builds the complete request body JSON to send to Gemini.
        
        Args:
            contents: The formatted conversation history
        
        Returns:
            Dictionary that will be converted to JSON
        """
        return {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": self.config.max_tokens,
                "temperature": 0.7,    # 0 = precise, 1 = creative
                "topP": 0.8,           # Controls response diversity
                "topK": 40,            # Consider top 40 token choices
            },
            # Safety settings — we'll use the defaults (BLOCK_MEDIUM_AND_ABOVE)
            # This filters out harmful content
        }
    
    async def send_message(
        self,
        message: str,
        history: List[Dict],
    ) -> str:
        """
        Sends a message to Gemini and returns the AI's response.
        
        This is the main method you'll call to chat with the AI.
        It includes automatic retry logic.
        
        Args:
            message: The user's message
            history: List of previous messages in the conversation
        
        Returns:
            The AI's response text
        
        Raises:
            GeminiAPIError: If the API returns an error
            GeminiTimeoutError: If the request times out
            GeminiNetworkError: If there's a network problem
        """
        # Validate input using ChatMessage Pydantic model
        # This ensures message is not empty
        try:
            validated_msg = ChatMessage(role="user", content=message)
        except Exception as e:
            raise ValueError(f"Invalid message: {e}")
        
        logger.info(f"Sending message: '{validated_msg.content[:50]}...' " 
                   f"(history: {len(history)} msgs)")
        
        # Build the request data
        contents = self._build_contents(validated_msg.content, history)
        request_body = self._build_request_body(contents)
        
        # Validate request using Pydantic before sending
        try:
            GeminiRequest(contents=contents)
        except Exception as e:
            logger.error(f"Request validation failed: {e}")
            raise ValueError(f"Request validation failed: {e}")
        
        logger.debug(f"Request URL: {self.generate_url[:50]}...")
        logger.debug(f"Request body keys: {list(request_body.keys())}")
        
        # Use retry logic — try up to max_retries times
        return await self._send_with_retry(request_body)
    
    async def _send_with_retry(self, request_body: Dict) -> str:
        """
        Sends the request with exponential backoff retry logic.
        
        Exponential backoff means:
        - First retry: wait 1 second
        - Second retry: wait 2 seconds
        - Third retry: wait 4 seconds
        - etc.
        
        This is important because if the API is overloaded, hammering it
        with rapid retries makes things worse. Waiting a bit helps.
        """
        last_error = None
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return await self._make_request(request_body)
                
            except GeminiTimeoutError as e:
                last_error = e
                if attempt < self.config.max_retries:
                    wait_time = 2 ** (attempt - 1)  # 1s, 2s, 4s...
                    logger.warning(f"Timeout on attempt {attempt}, retrying in {wait_time}s")
                    print_retry_message(attempt, self.config.max_retries, wait_time)
                    await asyncio.sleep(wait_time)
                    
            except GeminiNetworkError as e:
                last_error = e
                if attempt < self.config.max_retries:
                    wait_time = 2 ** attempt  # 2s, 4s, 8s...
                    logger.warning(f"Network error on attempt {attempt}, retrying in {wait_time}s")
                    print_retry_message(attempt, self.config.max_retries, wait_time)
                    await asyncio.sleep(wait_time)
                    
            except GeminiAPIError as e:
                # For certain API errors, don't retry (they won't get better)
                # 400 = bad request (our fault), 401 = invalid key, 403 = forbidden
                if e.status_code in [400, 401, 403]:
                    logger.error(f"Non-retryable API error {e.status_code}: {e.message}")
                    raise  # Re-raise immediately without retrying
                
                # For 429 (rate limit) and 500s (server errors), retry
                last_error = e
                if attempt < self.config.max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(f"API error {e.status_code} on attempt {attempt}, retrying in {wait_time}s")
                    print_retry_message(attempt, self.config.max_retries, wait_time)
                    await asyncio.sleep(wait_time)
        
        # If we get here, all retries failed
        logger.error(f"All {self.config.max_retries} attempts failed. Last error: {last_error}")
        raise last_error or GeminiAPIError(500, "All retry attempts failed")
    
    async def _make_request(self, request_body: Dict) -> str:
        """
        Makes a single HTTP request to the Gemini API.
        
        This is the actual network call. It:
        1. Sends the request body as JSON
        2. Checks the response status code
        3. Parses the JSON response
        4. Extracts and returns the text
        
        Args:
            request_body: The JSON body to send
        
        Returns:
            The AI's response text
        """
        if not self._http_client:
            raise GeminiConfigError("HTTP client not initialized. Use 'async with GeminiClient(...)'")
        
        try:
            logger.debug("Making HTTP POST request to Gemini...")
            
            # Make the async HTTP POST request
            # await means "wait for this to complete, but let other things run while waiting"
            # json=request_body automatically converts the dict to JSON and sets Content-Type
            response = await self._http_client.post(
                url=self.generate_url,
                json=request_body,
            )
            
            logger.debug(f"Response status code: {response.status_code}")
            
            # ── Handle HTTP error status codes ──
            # 2xx = success, 4xx = client error, 5xx = server error
            if response.status_code != 200:
                await self._handle_error_response(response)
            
            # ── Parse the successful JSON response ──
            try:
                response_data = response.json()
                logger.debug(f"Response keys: {list(response_data.keys())}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse response JSON: {e}")
                logger.error(f"Raw response: {response.text[:500]}")
                raise GeminiAPIError(500, "Invalid JSON in API response", str(e))
            
            # ── Validate response using Pydantic ──
            try:
                gemini_response = GeminiResponse(**response_data)
            except Exception as e:
                logger.error(f"Response validation failed: {e}")
                logger.error(f"Response data: {str(response_data)[:500]}")
                raise GeminiAPIError(500, "Response format validation failed", str(e))
            
            # ── Extract the text from the response ──
            response_text = gemini_response.get_text()
            
            if not response_text:
                logger.warning("Received empty response from Gemini")
                return "I'm sorry, I couldn't generate a response. Please try again."
            
            logger.info(f"Received response: '{response_text[:50]}...' ({len(response_text)} chars)")
            return response_text
        
        # ── Handle specific exception types ──
        
        except httpx.TimeoutException as e:
            # This happens when the API takes too long to respond
            logger.error(f"Request timed out: {e}")
            raise GeminiTimeoutError(
                f"Request timed out after {self.config.timeout}s. "
                "The Gemini API may be slow. Try again."
            )
        
        except httpx.ConnectError as e:
            # This happens when we can't reach the API at all
            logger.error(f"Connection error: {e}")
            raise GeminiNetworkError(
                "Cannot connect to Gemini API. "
                "Please check your internet connection."
            )
        
        except httpx.NetworkError as e:
            # General network errors (DNS failures, connection resets, etc.)
            logger.error(f"Network error: {e}")
            raise GeminiNetworkError(f"Network error: {str(e)}")
        
        except (GeminiAPIError, GeminiTimeoutError, GeminiNetworkError, GeminiConfigError):
            # Re-raise our custom exceptions unchanged
            raise
        
        except Exception as e:
            # Catch-all for any unexpected errors
            logger.error(f"Unexpected error during API request: {type(e).__name__}: {e}")
            raise GeminiAPIError(500, f"Unexpected error: {str(e)}")
    
    async def _handle_error_response(self, response: httpx.Response) -> None:
        """
        Parses and raises appropriate exceptions for HTTP error responses.
        
        Common HTTP status codes we handle:
        400 = Bad Request — our request was malformed
        401 = Unauthorized — API key is invalid or missing
        403 = Forbidden — API key doesn't have permission
        404 = Not Found — wrong URL or model name
        429 = Too Many Requests — rate limit exceeded
        500 = Internal Server Error — Google's servers crashed
        503 = Service Unavailable — Google's servers are down
        """
        status_code = response.status_code
        
        # Try to get error details from the response body
        try:
            error_data = response.json()
            error_message = (
                error_data.get("error", {}).get("message", "Unknown error")
                if isinstance(error_data, dict) else str(error_data)
            )
        except Exception:
            error_message = response.text[:200] or f"HTTP {status_code}"
        
        logger.error(f"API error response: {status_code} - {error_message}")
        
        # Create helpful user-facing error messages based on status code
        if status_code == 400:
            raise GeminiAPIError(
                status_code,
                "Bad request — the message format was invalid.",
                error_message
            )
        elif status_code == 401:
            raise GeminiAPIError(
                status_code,
                "Invalid API key! Please check your GEMINI_API_KEY in .env file.\n"
                "Get a valid key at: https://aistudio.google.com/app/apikey",
                error_message
            )
        elif status_code == 403:
            raise GeminiAPIError(
                status_code,
                "Access forbidden. Your API key may not have the right permissions.",
                error_message
            )
        elif status_code == 404:
            raise GeminiAPIError(
                status_code,
                f"Model '{self.config.model_name}' not found. "
                "Check your GEMINI_MODEL setting in .env",
                error_message
            )
        elif status_code == 429:
            raise GeminiAPIError(
                status_code,
                "Rate limit exceeded! Too many requests. Wait a moment and try again.",
                error_message
            )
        elif status_code >= 500:
            raise GeminiAPIError(
                status_code,
                f"Gemini API server error ({status_code}). Google's servers may be down.",
                error_message
            )
        else:
            raise GeminiAPIError(
                status_code,
                f"HTTP error {status_code}",
                error_message
            )
    
    async def send_message_streaming(
        self,
        message: str,
        history: List[Dict],
    ) -> AsyncGenerator[str, None]:
        """
        BONUS FEATURE: Sends a message and streams the response word-by-word.
        
        Instead of waiting for the full response, we yield (produce) each
        chunk of text as it arrives. This makes the chatbot feel more
        responsive — like watching someone type.
        
        This is an "async generator" function:
        - 'async' because it does network calls
        - 'generator' because it uses 'yield' to produce values one at a time
        
        Usage:
            async for chunk in client.send_message_streaming(msg, history):
                print(chunk, end="", flush=True)  # Print without newline, flush immediately
        
        Args:
            message: The user's message
            history: Previous conversation messages
        
        Yields:
            String chunks of the response as they arrive
        """
        if not self._http_client:
            raise GeminiConfigError("HTTP client not initialized.")
        
        # Validate input
        try:
            validated_msg = ChatMessage(role="user", content=message)
        except Exception as e:
            raise ValueError(f"Invalid message: {e}")
        
        logger.info(f"Starting streaming request: '{validated_msg.content[:50]}...'")
        
        # Build request body
        contents = self._build_contents(validated_msg.content, history)
        request_body = self._build_request_body(contents)
        
        try:
            # stream=True tells httpx to stream the response byte-by-byte
            # instead of downloading the whole response at once
            async with self._http_client.stream(
                method="POST",
                url=self.stream_url,
                json=request_body,
            ) as response:
                
                if response.status_code != 200:
                    # For streaming, we need to read the full error response
                    await response.aread()
                    await self._handle_error_response(response)
                
                # Read the stream line by line
                # SSE (Server-Sent Events) format sends:
                # "data: {json}\n\n" for each chunk
                # "data: [DONE]\n\n" when finished
                
                full_response = []
                
                async for line in response.aiter_lines():
                    # Skip empty lines
                    if not line.strip():
                        continue
                    
                    # SSE lines start with "data: "
                    if not line.startswith("data: "):
                        continue
                    
                    # Extract the JSON part after "data: "
                    json_str = line[6:]  # Remove "data: " prefix (6 characters)
                    
                    # "[DONE]" signals the end of the stream
                    if json_str.strip() == "[DONE]":
                        break
                    
                    # Parse the JSON chunk
                    try:
                        chunk_data = json.loads(json_str)
                        
                        # Extract text from the chunk
                        # Each chunk has the same structure as a full response
                        text = (
                            chunk_data
                            .get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "")
                        )
                        
                        if text:
                            full_response.append(text)
                            yield text  # 'yield' = produce this value now, continue later
                    
                    except json.JSONDecodeError:
                        # Some chunks might not be valid JSON — skip them
                        continue
                
                # Log the complete response
                complete_text = "".join(full_response)
                logger.info(f"Streaming complete: {len(complete_text)} chars received")
        
        except httpx.TimeoutException:
            logger.error("Streaming request timed out")
            raise GeminiTimeoutError("Streaming request timed out")
        
        except httpx.ConnectError:
            logger.error("Cannot connect to Gemini for streaming")
            raise GeminiNetworkError("Cannot connect to Gemini API")
        
        except (GeminiAPIError, GeminiTimeoutError, GeminiNetworkError):
            raise
        
        except Exception as e:
            logger.error(f"Unexpected streaming error: {e}")
            raise GeminiAPIError(500, f"Streaming error: {str(e)}")
