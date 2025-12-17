#!/usr/bin/env python3
"""
Process a tab-separated CSV/TSV file, sending each row's `text` field to
Anthropic Claude with a prompt loaded from prompt.md, and write an
output TSV with an added `text_normalised` column.

- Parallel processing via ThreadPoolExecutor
- Row limit for quick dev runs
- Retries with exponential backoff for transient errors
- Loads API key from a .env file (via python-dotenv)

Prereqs:
  - Python 3.9+
  - requests (pip install requests)
  - python-dotenv (pip install python-dotenv)
  - Create a .env file in the same directory with:
        ANTHROPIC_API_KEY=sk-ant-...

Edit the CONFIG section below to suit your environment.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

# ============ CONFIG ============
# Paths
INPUT_TSV: str = "input.csv"  # Tab-separated input file
OUTPUT_TSV: str = "output.csv"  # Output file path
PROMPT_FILE: str = "normalise_prompt.md"  # Prompt instructions (system prompt)

# Anthropic API settings
ANTHROPIC_API_URL: str = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION: str = "2023-06-01"
MODEL: str = "claude-sonnet-4-5-20250929"
MAX_TOKENS: int = 2048          # required, max response tokens
TEMPERATURE: float = 0.0
TIMEOUT_SECONDS: int = 60
RETRIES: int = 5
BACKOFF_BASE_SECONDS: float = 2.0  # exponential backoff base (2, 4, 8, ...)
MAX_WORKERS: int = 8  # parallel request workers

# Development row limiting
# Set to None or "none"/"all" to process all rows; set to an int (e.g., 10) to process only the first N rows
ROW_LIMIT: Optional[object] = None

# Name of the input column and output column
INPUT_TEXT_COLUMN: str = "text"
OUTPUT_TEXT_COLUMN: str = "text_normalised"

# ========== END CONFIG ==========

API_KEY_ENV: str = "ANTHROPIC_API_KEY"


def parse_row_limit(value: Optional[object]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "all", ""}:
        return None
    try:
        n = int(value)  # type: ignore[arg-type]
        return max(0, n)
    except Exception:
        return None


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_payload(model: str, system_prompt: str, user_text: str) -> Dict:
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_text},
        ],
    }


def extract_text_from_response(data: Dict) -> str:
    # Anthropic messages API returns content as a list of blocks
    content = data.get("content", [])
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()
    # Fallback for any unexpected shapes
    return str(content).strip()


def call_anthropic(session: requests.Session, headers: Dict[str, str], payload: Dict) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(RETRIES):
        try:
            resp = session.post(
                ANTHROPIC_API_URL,
                headers=headers,
                json=payload,
                timeout=TIMEOUT_SECONDS,
            )
            if resp.status_code == 200:
                data = resp.json()
                return extract_text_from_response(data)

            # Transient errors: retry with backoff
            if resp.status_code in {429, 500, 502, 503, 504}:
                delay = (BACKOFF_BASE_SECONDS ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                continue

            # Non-retryable error
            try:
                err_json = resp.json()
            except Exception:
                err_json = {"error": resp.text}
            raise RuntimeError(
                f"Anthropic API error {resp.status_code}: {json.dumps(err_json)[:1000]}"
            )
        except requests.RequestException as e:
            # Network/timeout: retry
            last_err = e
            delay = (BACKOFF_BASE_SECONDS ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
            continue
    # Exhausted retries
    if last_err:
        raise last_err
    raise RuntimeError("Unknown error calling Anthropic API")


def process_text(session: requests.Session, headers: Dict[str, str], system_prompt: str, text: str) -> str:
    if text is None:
        text = ""
    payload = build_payload(MODEL, system_prompt, text)
    return call_anthropic(session, headers, payload)


def main():
    # Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise SystemExit(
            f"Missing API key. Please set {API_KEY_ENV} in a .env file or environment variables."
        )

    system_prompt = load_prompt(PROMPT_FILE)

    # Read input TSV
    with open(INPUT_TSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise SystemExit("Input file appears to have no header row.")
        fieldnames = list(reader.fieldnames)
        if INPUT_TEXT_COLUMN not in fieldnames:
            raise SystemExit(
                f"Input file is missing required column '{INPUT_TEXT_COLUMN}'. Found columns: {fieldnames}"
            )
        rows = list(reader)

    limit = parse_row_limit(ROW_LIMIT)
    if limit is not None:
        rows = rows[:limit]

    print(
        f"Loaded {len(rows)} row(s) from '{INPUT_TSV}'. Processing with {MAX_WORKERS} worker(s); model={MODEL}."
    )

    # Prepare HTTP session and headers
    session = requests.Session()
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    # Submit work in parallel
    results: Dict[int, str] = {}
    errors: Dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {}
        for idx, row in enumerate(rows):
            txt = row.get(INPUT_TEXT_COLUMN, "") or ""
            fut = executor.submit(process_text, session, headers, system_prompt, txt)
            future_to_idx[fut] = idx

        completed = 0
        total = len(future_to_idx)
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                out_text = fut.result()
                results[idx] = out_text
            except Exception as e:
                msg = str(e)
                errors[idx] = msg
                results[idx] = f"[ERROR] {msg}"
            finally:
                completed += 1
                if completed % 5 == 0 or completed == total:
                    print(f"Progress: {completed}/{total} completed")

    # Prepare writer
    out_fieldnames = list(fieldnames)
    if OUTPUT_TEXT_COLUMN not in out_fieldnames:
        out_fieldnames.append(OUTPUT_TEXT_COLUMN)

    # Write output TSV
    with open(OUTPUT_TSV, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, delimiter="\t", fieldnames=out_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for idx, row in enumerate(rows):
            row_out = dict(row)
            row_out[OUTPUT_TEXT_COLUMN] = results.get(idx, "")
            writer.writerow(row_out)

    print(
        f"Done. Wrote {len(rows)} row(s) to '{OUTPUT_TSV}'. Errors: {len(errors)}"
    )
    if errors:
        print("Some rows failed. You can re-run with a smaller ROW_LIMIT or lower MAX_WORKERS, or inspect errors above.")


if __name__ == "__main__":
    main()
