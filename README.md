# 🤖 Async AI Chatbot

> A professional, asynchronous command-line AI chatbot powered by **Google Gemini**, built with Python using modern async/await patterns, Pydantic validation, structured logging, and a beautiful Rich CLI interface.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🚀 **Async Architecture** | Non-blocking I/O using `async/await` and `httpx.AsyncClient` |
| 🔒 **Validated Config** | Pydantic models validate all settings before startup |
| 💬 **Chat History** | Full conversation memory throughout the session |
| 🎨 **Rich CLI UI** | Colorful, formatted terminal interface with panels and Markdown |
| 📝 **Structured Logging** | File + console logging with timestamps and levels |
| 🔄 **Auto Retry** | Exponential backoff retry for failed API calls |
| 📡 **Streaming** | Real-time word-by-word response streaming |
| 🛡️ **Error Handling** | Graceful handling of all error types |
| ✅ **Type Safe** | Full Python type hints throughout |

---

## 🛠️ Technologies Used

- **Python 3.10+** — Main language
- **httpx** — Async HTTP client for API requests
- **Pydantic v2** — Data validation with Python type hints
- **python-dotenv** — Environment variable management
- **Rich** — Beautiful terminal formatting and colors
- **tenacity** — Retry logic with exponential backoff
- **Google Gemini API** — Free-tier LLM provider

---

## 📁 Project Structure

```
ai_chatbot/
│
├── app/                     # Main application package
│   ├── __init__.py          # Package marker
│   ├── main.py              # Entry point
│   ├── chat.py              # Chat loop and session management
│   ├── client.py            # Gemini API client (async)
│   ├── config.py            # Configuration loading and validation
│   ├── models.py            # Pydantic data models
│   ├── logger.py            # Logging setup
│   └── ui.py                # CLI interface and formatting
│
├── logs/                    # Log files (auto-created)
│   └── chatbot.log
│
├── .env                     # 🔒 Your secrets (NOT in Git)
├── .env.example             # Safe template for .env
├── .gitignore               # Files Git ignores
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## ⚡ Quick Start

### Step 1: Prerequisites

- Python 3.10 or newer
- A free Google Gemini API key ([get one here](https://aistudio.google.com/app/apikey))

### Step 2: Clone or Download

```bash
# If you have git:
git clone https://github.com/yourusername/ai-chatbot.git
cd ai-chatbot

# Or just download and extract the ZIP, then open a terminal in the folder
```

### Step 3: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

You'll see `(venv)` at the start of your terminal prompt.

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Set Up Your API Key

```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```

Open the `.env` file in any text editor and replace `your_gemini_api_key_here` with your actual API key:

```env
GEMINI_API_KEY=AIzaSyABCDEF1234567890abcdef
```

### Step 6: Run the Chatbot

```bash
python -m app.main
```

---

## 💬 Usage

Once running, you can:

| Command | Action |
|---|---|
| Type anything | Send a message to the AI |
| `exit` / `quit` / `bye` | Exit the chatbot |
| `clear` | Clear the terminal screen |
| `history` | Show conversation history |
| `reset` | Clear history and start a new conversation |
| `help` | Show all commands |
| `Ctrl+C` | Force interrupt |

### Example Session

```
┌─ You ───────────────────────────────────┐
└─ ❯ What is a black hole?

🤖 Gemini is writing...

A black hole is a region in space where gravity is so strong
that nothing — not even light — can escape from it...

┌─ You ───────────────────────────────────┐
└─ ❯ How are they formed?

🤖 Gemini is writing...

Black holes form when massive stars run out of fuel...
```

---

## ⚙️ Configuration

All settings live in your `.env` file:

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Your Gemini API key |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Model to use |
| `REQUEST_TIMEOUT` | `30` | Seconds before timeout |
| `MAX_TOKENS` | `2048` | Max response length |
| `MAX_RETRIES` | `3` | Retry attempts on failure |
| `LOG_FILE` | `logs/chatbot.log` | Log file location |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## 📋 Logging

Logs are saved to `logs/chatbot.log`. Each line includes:

```
2024-01-15 14:30:25 | INFO     | ai_chatbot:client.py:87 | Sending message to Gemini
2024-01-15 14:30:26 | INFO     | ai_chatbot:client.py:124| Response received: 245 chars
2024-01-15 14:30:45 | ERROR    | ai_chatbot:client.py:156| Timeout error: exceeded 30s
```

---

## 🔄 Architecture: How Data Flows

```
User Input
    │
    ▼
chat.py (ChatSession.process_message)
    │  Validates input, handles commands
    ▼
client.py (GeminiClient.send_message)
    │  Validates with Pydantic models
    │  Builds request body
    ▼
httpx.AsyncClient.post()
    │  Async HTTP request
    ▼
Google Gemini API
    │  AI generates response
    ▼
client.py (parse + validate response)
    │  GeminiResponse Pydantic model
    ▼
chat.py (add to history, display)
    │
    ▼
ui.py (Rich formatted output)
    │
    ▼
User sees the response
```

---

## 🚨 Troubleshooting

### "Configuration Error: Please set a real Gemini API key"
- Open `.env` file
- Replace `your_gemini_api_key_here` with your actual API key
- Get a free key at: https://aistudio.google.com/app/apikey

### "Cannot connect to Gemini API"
- Check your internet connection
- Try opening https://google.com in a browser
- Check if a firewall/VPN might be blocking the connection

### "Request timed out"
- Increase `REQUEST_TIMEOUT` in `.env` (try `60`)
- Check your internet speed

### "Rate limit exceeded (429)"
- Wait 1-2 minutes before trying again
- Gemini free tier has usage limits

### "ModuleNotFoundError"
- Make sure your virtual environment is activated: `venv\Scripts\activate`
- Run: `pip install -r requirements.txt`

### Logs show errors
- Check `logs/chatbot.log` for detailed error messages
- Set `LOG_LEVEL=DEBUG` in `.env` for maximum verbosity

---

## 🔮 Future Improvements

- [ ] Multiple LLM provider support (OpenAI, Claude, Ollama)
- [ ] Save/load conversation history to/from files
- [ ] Web interface using FastAPI
- [ ] Voice input/output
- [ ] Custom system prompts
- [ ] Token usage tracking
- [ ] Cost estimation display
- [ ] Multi-turn conversation templates

---

## 📄 License

This project is for educational purposes. Feel free to use, modify, and share it!

---

## 🙏 Acknowledgments

- [Google Gemini API](https://ai.google.dev/) — Free tier AI access
- [Rich](https://github.com/Textualize/rich) — Beautiful terminal output
- [Pydantic](https://docs.pydantic.dev/) — Data validation
- [httpx](https://www.python-httpx.org/) — Async HTTP
