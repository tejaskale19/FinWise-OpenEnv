"""Validator-compatible server entrypoint wrapper."""

import uvicorn

from app import app

__all__ = ["app"]


def main() -> None:
	uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
	main()
