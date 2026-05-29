# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- Added CLI provider selection for Ollama or Gemini.
- Added `--thinking` / `--no-thinking` to control streaming-style replies.
- Added Gemini API support through `GEMINI_API_KEY` and `GEMINI_MODEL`.

## [0.1.0] - 2026-05-28

- Migrated project layout to src-based packaging (`src/ai_chatbot`).
- Added `pyproject.toml` for `uv` dependency and project management.
- Added standard project directories: `tests`, `docs`, `notebooks`.
- Consolidated virtual environment usage to `.venv`.

## [0.2.0] - 2026-05-28

- Replaced the Gemini integration with local Ollama chat using `gemma3:4b`.
- Removed API key requirements and Gemini-specific environment variables.
- Added local Ollama verification before the chat UI starts.
