# Shared image for the api, ui, and mcp services. One build, three commands —
# the services differ only in entrypoint, so a single layer cache serves all of
# them and there is one dependency set to reason about.
#
# uv is used rather than pip: the lockfile makes the evaluator's build
# byte-identical to the one this was developed against.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first, in their own layer: application edits do not invalidate
# the (slow) dependency install.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --all-extras

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras

# Non-root: nothing here needs privilege, and the sandboxed Python tool is one
# more reason not to run as root.
RUN useradd --create-home --uid 10001 atrium \
    && chown -R atrium:atrium /app
USER atrium

EXPOSE 8000 8501 8900

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
