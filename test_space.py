"""Simple health test for a deployed FinWise OpenEnv Space.

Checks the required endpoints in order:
1) POST /reset
2) POST /step
3) POST /state
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Tuple

import httpx


def post_json(client: httpx.Client, url: str, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    response = client.post(url, json=payload)
    try:
        data = response.json()
    except ValueError:
        data = {"raw_text": response.text}
    return response.status_code, data


def main() -> int:
    parser = argparse.ArgumentParser(description="Test /reset, /step, /state on an OpenEnv Space")
    parser.add_argument("--base-url", required=True, help="Space base URL, e.g. https://user-space.hf.space")
    parser.add_argument("--task", default="diversify_sector_easy", help="Task name for reset")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    with httpx.Client(timeout=args.timeout, follow_redirects=True) as client:
        reset_status, reset_data = post_json(
            client,
            f"{base_url}/reset",
            {"task_name": args.task},
        )
        print(f"/reset status={reset_status}")
        if reset_status != 200:
            print(json.dumps(reset_data, indent=2))
            return 1

        session_id = reset_data.get("session_id")
        if not session_id:
            print("/reset did not return session_id")
            print(json.dumps(reset_data, indent=2))
            return 1

        step_status, step_data = post_json(
            client,
            f"{base_url}/step",
            {
                "session_id": session_id,
                "action_type": "hold",
            },
        )
        print(f"/step status={step_status}")
        if step_status != 200:
            print(json.dumps(step_data, indent=2))
            return 1

        state_status, state_data = post_json(
            client,
            f"{base_url}/state",
            {"session_id": session_id},
        )
        print(f"/state status={state_status}")
        if state_status != 200:
            print(json.dumps(state_data, indent=2))
            return 1

    print("PASS: /reset, /step, /state all returned HTTP 200")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
