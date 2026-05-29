# AI Chatbot

An asynchronous command-line chatbot built with Python, Rich, and Pydantic. It can talk to either local Ollama or the Gemini API, and you can choose the provider from the command line.

## Features

- Provider selection for Ollama or Gemini
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
- For Ollama: [Ollama](https://ollama.com/download) installed locally and a pulled model such as `gemma3:4b`
- For Gemini: a valid Gemini API key and access to a Gemini model

## Setup

### 1. Install dependencies

From the project root, run:

```powershell
uv sync
```

This installs the Python packages into `.venv` and uses the versions locked in `uv.lock`.

### 2. Choose a provider

Set `LLM_PROVIDER=ollama` for local inference or `LLM_PROVIDER=gemini` for the Gemini API.

If you use Gemini, also set `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env`.

If you use Ollama, make sure it is running and available at:

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

To select a provider explicitly from the command line:

```powershell
uv run ai-chatbot --provider gemini --thinking
uv run ai-chatbot --provider ollama --no-thinking
```

You can also run the module directly:

```powershell
uv run python -m ai_chatbot.main
```

## How it works

- The app loads settings from `.env` and CLI overrides.
- The chat client connects to the provider selected by `--provider` or `LLM_PROVIDER`.
- Before the UI starts, the app checks that the selected provider is reachable and that the chosen model is available.
- Conversation history is kept in memory so the model can answer with context.
- Streaming mode prints the response as it arrives, so the UI feels live.

## Environment Variables

Use `.env` for local settings:

| Variable          | Default                  | Purpose                                 |
| ----------------- | ------------------------ | --------------------------------------- |
| `LLM_PROVIDER`    | `ollama`                 | Default provider (`ollama` or `gemini`) |
| `THINKING`        | `true`                   | Stream responses as they generate       |
| `OLLAMA_HOST`     | `http://localhost:11434` | Local Ollama server address             |
| `OLLAMA_MODEL`    | `gemma3:4b`              | Model to chat with                      |
| `GEMINI_API_KEY`  | empty                    | Gemini API key                          |
| `GEMINI_MODEL`    | `gemini-2.0-flash`       | Gemini model to chat with               |
| `REQUEST_TIMEOUT` | `30`                     | Seconds to wait before timing out       |
| `MAX_TOKENS`      | `2048`                   | Maximum tokens to generate              |
| `MAX_RETRIES`     | `3`                      | Retry attempts for temporary failures   |
| `TEMPERATURE`     | `0.7`                    | Controls how creative the model is      |
| `SYSTEM_PROMPT`   | helpful assistant prompt | Initial instruction for the model       |
| `LOG_FILE`        | `logs/chatbot.log`       | Log file path                           |
| `LOG_LEVEL`       | `INFO`                   | Logging level                           |

## Commands in Chat

- Type anything to chat.
- Type `exit`, `quit`, or `bye` to leave.
- Type `clear` to clear the screen.
- Type `history` to view the current conversation.
- Type `reset` to start over with a fresh history.
- Type `help` to show the command list again.

## Troubleshooting

### Ollama is not running

If you see a connection error while using Ollama, start it first and make sure the server is available at `http://localhost:11434`.

### Gemini access errors

If you use Gemini, confirm that `GEMINI_API_KEY` is set and that `GEMINI_MODEL` is a valid model your key can access.

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

- Ollama keeps everything local.
- Gemini requires an API key and network access.
- You can switch providers without changing the code.
