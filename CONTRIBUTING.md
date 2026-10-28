# Contributing to GenieBot

Thanks for your interest in contributing! GenieBot is a Telegram bot (RAG + vision) built with
`python-telegram-bot`, a SQLite-backed retrieval system, and hosted LLM/vision calls via
OpenRouter (with optional local Ollama support). This guide covers how to get set up, make
changes, and send a pull request.

## Dev setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/iamvisheshsrivastava/geniebot.git
cd geniebot
python -m venv .venv
```

Activate it:

- Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # test tooling (pytest, etc.)
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill it in:

```bash
cp .env.example .env   # macOS/Linux
copy .env.example .env # Windows
```

Key variables:

| Variable | Required for | Notes |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | Running the bot at all | Get one from [@BotFather](https://t.me/BotFather) on Telegram — create a bot, it gives you a token. Use a **separate test bot** for development rather than the production `@Hello_Genie_Bot` token. |
| `LLM_PROVIDER` | Chat answers | `openrouter` (default, no local server needed) or `ollama` (local dev, needs `ollama serve` running). |
| `OPENROUTER_API_KEY` | `LLM_PROVIDER=openrouter`, and always for `/image` (vision always goes through OpenRouter) | Free tier works; see [openrouter.ai/models?max_price=0](https://openrouter.ai/models?max_price=0). |
| `OPENROUTER_MODEL` / `OPENROUTER_VISION_MODEL` | Chat / vision model selection | OpenRouter occasionally retires free models — if `/ask` or `/image` start failing, check the models link above before assuming it's a code bug. |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | `LLM_PROVIDER=ollama` | Only used for chat; vision still requires `OPENROUTER_API_KEY`. |
| `LOG_LEVEL`, `PORT` | Logging verbosity / status server port | Defaults are fine for local dev. |

You do not need a real Telegram bot token to run the test suite — only to run `python app.py`
and actually talk to the bot. Never commit your `.env` file or paste real tokens into issues,
PRs, or commit messages.

### 4. (Optional) Prebuild the embedding index

If you're working on `rag/` and want the SQLite vector store populated ahead of time:

```bash
python scripts/build_vector_db.py --data-dir data --db-path data/rag_embeddings.db --chunk-size 200 --chunk-overlap 50
```

Otherwise, `app.py` builds/loads the index automatically on startup.

### 5. Run the bot locally

```bash
python app.py
```

Message your test bot on Telegram to confirm it responds to `/start`, `/ask`, and `/image`.

## Running tests

The test suite is pure pytest, fully mocked (no live Telegram/OpenRouter/Ollama calls needed):

```bash
pytest tests/
```

Run a single file or test while iterating:

```bash
pytest tests/test_rag_system.py -v
pytest tests/test_handlers.py::test_summarize_command_no_llm -v
```

CI (`.github/workflows/tests.yml`) runs the full suite on every push and PR — make sure it
passes locally before opening a PR. If you add new behavior to `bot/handlers.py`, `rag/`,
`vision/`, or `utils/`, add or update a test alongside the code change; see the existing files
in `tests/` for the mocking patterns used for Telegram `Update`/`Context` objects, RAG retrieval,
and LLM fallback chains.

## Code style

- Keep functions small and focused; most handlers/modules in this repo follow a
  "one responsibility per function" style — follow the existing pattern in the file you're
  editing rather than introducing a new style.
- Prefer explicit, descriptive log messages (see `utils/logger.py` usage throughout) over
  silent failures — this bot runs unattended on a free-tier host, so logs are often the only
  way to debug an issue after the fact.
- Don't swallow exceptions silently; log them and return a user-facing error message instead
  (see `_safe_reply_text` / the `try/except` patterns in `bot/handlers.py`).

## Finding something to work on

- Check the [issue tracker](https://github.com/iamvisheshsrivastava/geniebot/issues), especially
  issues labeled [`good first issue`](https://github.com/iamvisheshsrivastava/geniebot/labels/good%20first%20issue)
  for small, well-scoped starting points.
- If you spot a bug or want to propose a feature that isn't already filed, open an issue first
  describing the problem/proposal before sending a large PR, so we can agree on the approach.

## Pull request process

1. Fork the repo and create a branch off `main` (e.g. `fix/summarize-placeholder-message`).
2. Make your change, with a test covering it where practical.
3. Run `pytest tests/` locally and make sure everything passes.
4. Commit with a clear, descriptive message explaining *why*, not just *what*.
5. Open a PR against `main`:
   - Reference the issue it closes (e.g. `Closes #22`) if applicable.
   - Describe what changed and how you tested it.
   - Keep PRs focused — one logical change per PR is much easier to review than a bundle of
     unrelated fixes.
6. Address review feedback via additional commits (no need to force-push/squash until asked).
7. A maintainer will merge once CI passes and the change looks good.

## Reporting bugs / requesting features

Open a [GitHub issue](https://github.com/iamvisheshsrivastava/geniebot/issues/new) with:

- What you expected to happen vs. what actually happened.
- Steps to reproduce (a specific `/ask`, `/image`, or `/summarize` interaction, if relevant).
- Relevant log output from `logs/geniebot_YYYYMMDD.log`, with any tokens/API keys redacted.

Thanks again for contributing to GenieBot!
