# VocalisAI web app + API. Runs on Cloud Run (or any container host); listens on $PORT.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy PYTHONUNBUFFERED=1 HOME=/tmp \
    # no local Ollama in the cloud: the simulated airline rep runs on free-tier models
    VOCALIS_REP_MODELS=groq/openai/gpt-oss-20b,gemini/gemini-flash-lite-latest

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --extra voice --no-dev

COPY web ./web
COPY evals/scenarios ./evals/scenarios
COPY evals/results/summary.json ./evals/results/summary.json
# caches (TTS audio, quota ledger) must be writable by a non-root runtime user
RUN mkdir -p /app/.cache && chmod -R 777 /app

EXPOSE 7860
CMD ["sh", "-c", "uv run --no-sync uvicorn vocalis.web.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
