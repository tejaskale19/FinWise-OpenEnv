"""CLI entrypoint for OpenEnv multi-mode server startup."""

import uvicorn


def main() -> None:
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
