# Ollama AI Chatbot

A local, asynchronous command-line chatbot built with Python, Rich, Pydantic, and the Ollama Python client. It runs fully on your PC with the `gemma3:4b` model, so no paid API keys or cloud services are needed.

## Features

- Local inference through Ollama
- Async chat loop with streaming output
- Conversation memory for multi-turn chats
- Rich terminal UI
- Input validation and friendly error messages
- Clean project layout with `src/`

## Project Structure

```text
ai_chatbot/
├── src/
│   └── ai_chatbot/
│       ├── __init__.py
│       ├── main.py
│       ├── chat.py
│       ├── client.py
│       ├── config.py
│       ├── models.py
│       ├── logger.py
│       └── ui.py
├── tests/
├── docs/
├── notebooks/
├── logs/
├── pyproject.toml
├── uv.lock
├── CHANGELOG.md
├── .env
└── .env.example
```

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/download) installed locally
- The `gemma3:4b` model pulled into Ollama

## Setup

### 1. Install dependencies

From the project root, run:

```powershell
uv sync
```

This installs the Python packages into `.venv` and uses the versions locked in `uv.lock`.

### 2. Make sure Ollama is running

Start Ollama from the app or the terminal, then confirm it is available at:

```text
http://localhost:11434
```

### 3. Pull the model if it is missing

If `gemma3:4b` is not installed yet, run:

```powershell
ollama pull gemma3:4b
```

### 4. Run the chatbot

```powershell
uv run ai-chatbot
```

You can also run the module directly:

```powershell
uv run python -m ai_chatbot.main
```

## How it works

- The app loads local settings from `.env`.
- The chat client connects to Ollama on `http://localhost:11434`.
- Before the UI starts, the app checks that the server is reachable and that `gemma3:4b` exists locally.
- Conversation history is kept in memory so the model can answer with context.
- Streaming mode prints the response as it arrives, so the UI feels live.

## Environment Variables

Use `.env` for local settings:

| Variable          | Default                  | Purpose                               |
| ----------------- | ------------------------ | ------------------------------------- |
| `OLLAMA_HOST`     | `http://localhost:11434` | Local Ollama server address           |
| `OLLAMA_MODEL`    | `gemma3:4b`              | Model to chat with                    |
| `REQUEST_TIMEOUT` | `30`                     | Seconds to wait before timing out     |
| `MAX_TOKENS`      | `2048`                   | Maximum tokens to generate            |
| `MAX_RETRIES`     | `3`                      | Retry attempts for temporary failures |
| `TEMPERATURE`     | `0.7`                    | Controls how creative the model is    |
| `SYSTEM_PROMPT`   | helpful assistant prompt | Initial instruction for the model     |
| `LOG_FILE`        | `logs/chatbot.log`       | Log file path                         |
| `LOG_LEVEL`       | `INFO`                   | Logging level                         |

## Commands in Chat

- Type anything to chat.
- Type `exit`, `quit`, or `bye` to leave.
- Type `clear` to clear the screen.
- Type `history` to view the current conversation.
- Type `reset` to start over with a fresh history.
- Type `help` to show the command list again.

## Troubleshooting

### Ollama is not running

If you see a connection error, start Ollama first and make sure the server is available at `http://localhost:11434`.

### The model is missing

If you see `Run: ollama pull gemma3:4b`, install the model with:

```powershell
ollama pull gemma3:4b
```

### The chatbot is slow or times out

Try increasing `REQUEST_TIMEOUT` in `.env`.

### The terminal looks blank or weird

Make sure your terminal supports ANSI colors and Rich formatting.

## Notes

- No API key is required.
- No Gemini or OpenAI cloud service is used.
- Everything runs locally through Ollama.
