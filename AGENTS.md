# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Bilibili UP 字幕 RAG — a Python/FastAPI application that performs hybrid RAG search over Bilibili video subtitles (.srt files). See `README.md` for full architecture and API docs.

### Prerequisites

- Python 3.12+ (system python on the VM)
- Virtual environment at `.venv/`
- `DASHSCOPE_API_KEY` environment secret (Alibaba Cloud 百炼 API key) — **required** for index building and full-feature search

### Running the app

```bash
source .venv/bin/activate
python -m app.main
# Serves on http://127.0.0.1:8000
```

On first startup (or after `rm -rf data/index data/chroma`), the server builds the index automatically, which requires embedding API calls via `DASHSCOPE_API_KEY`. Once the index is built, subsequent startups are fast.

### Important caveats

- The `.env` file must exist at the project root (copy from `.env.example` and fill in `DASHSCOPE_API_KEY`).
- Index building happens synchronously during the FastAPI lifespan startup event — the server won't accept HTTP requests until indexing completes.
- Default `VIDEO_LIMIT=20` limits indexing to 20 videos for quick dev iteration. Set `VIDEO_LIMIT=0` for full index.
- There are no automated tests in this project currently.
- There is no linter configuration; use standard Python tools (e.g. `ruff`, `pyright`) if needed.
- No Docker, no Makefile, no Node.js — pure Python project.

### Lint / Test / Build commands

| Action | Command |
|--------|---------|
| Install deps | `source .venv/bin/activate && pip install -r requirements.txt` |
| Run app | `source .venv/bin/activate && python -m app.main` |
| Lint (if ruff installed) | `source .venv/bin/activate && ruff check app/` |
| No tests | (project has no test suite) |
