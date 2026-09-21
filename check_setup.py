"""Check the local RAG workshop environment.

Run:
    python check_setup.py
    python check_setup.py --api   # also makes one tiny embedding request
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from dotenv import load_dotenv


PACKAGES = {
    "openai": "openai",
    "langchain": "langchain",
    "langchain-core": "langchain_core",
    "langchain-openai": "langchain_openai",
    "langchain-text-splitters": "langchain_text_splitters",
    "langchain-community": "langchain_community",
    "faiss-cpu": "faiss",
    "numpy": "numpy",
    "pypdf": "pypdf",
    "python-dotenv": "dotenv",
    "jupyter": "jupyter_core",
    "ipykernel": "ipykernel",
}


def package_version(distribution_name: str) -> str:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return "unknown"


def check_imports() -> bool:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}\n")

    all_ok = True
    for distribution_name, import_name in PACKAGES.items():
        try:
            importlib.import_module(import_name)
            print(f"[OK] {distribution_name:<28} {package_version(distribution_name)}")
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            all_ok = False
            print(f"[FAIL] {distribution_name:<26} {type(exc).__name__}: {exc}")
    return all_ok


def check_api() -> bool:
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key or key == "replace_with_your_key":
        print("\n[FAIL] OPENAI_API_KEY is missing. Copy .env.example to .env and add your key.")
        return False

    try:
        from openai import OpenAI

        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        response = OpenAI().embeddings.create(model=model, input="RAG setup test")
        dimensions = len(response.data[0].embedding)
        print(f"\n[OK] OpenAI embedding request succeeded with {model} ({dimensions} dimensions).")
        return True
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print(f"\n[FAIL] OpenAI API test failed: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true", help="Make one small OpenAI embedding request.")
    args = parser.parse_args()

    imports_ok = check_imports()
    api_ok = check_api() if args.api else True

    if imports_ok and api_ok:
        print("\nEnvironment looks ready.")
        return 0

    print("\nOne or more checks failed. Confirm that VS Code selected this same interpreter/kernel.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
