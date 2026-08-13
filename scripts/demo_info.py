#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Print safe TranslateGemma 12B demo commands without exposing secrets")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    key_file = ROOT / "data/api_key.txt"
    image = ROOT / "assets/sample-image-with-text.png"
    python_client = ROOT / "clients/python/translategemma_client.py"
    node_client = ROOT / "clients/node/translategemma-client.mjs"

    print(f"Base URL: {base}")
    print(f"API key file: {key_file} (value intentionally not displayed)")
    print("\nHealth:")
    print(f"  curl -fsS {base}/health/live")
    print("\nAuthenticated runtime info:")
    print(
        f"  python3 clients/python/translategemma_client.py --base-url {base} "
        "--api-key-file data/api_key.txt info"
    )
    print("\nText translation:")
    print(
        f"  python3 clients/python/translategemma_client.py --base-url {base} "
        "--api-key-file data/api_key.txt text 'Good morning! How are you?'"
    )
    print("\nMultipart image translation:")
    print(
        f"  python3 clients/python/translategemma_client.py --base-url {base} "
        f"--api-key-file data/api_key.txt image {image.relative_to(ROOT)} --multipart"
    )
    print("\nNode.js 18+ equivalents:")
    print(
        f"  node {node_client.relative_to(ROOT)} text --base-url {base} "
        "--api-key-file data/api_key.txt --text 'Good morning! How are you?'"
    )
    print(
        f"  node {node_client.relative_to(ROOT)} image --base-url {base} "
        f"--api-key-file data/api_key.txt --image {image.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
