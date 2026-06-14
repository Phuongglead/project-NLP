#!/usr/bin/env python3
"""Test Gemini API keys from the command line.

Examples:
  # Test key from .env
  python scripts/test_gemini_keys.py

  # Test one key
  python scripts/test_gemini_keys.py --key "YOUR_KEY"

  # Test multiple keys
  python scripts/test_gemini_keys.py --keys "key1,key2,key3"

  # With conda env
  conda activate sa-aqg
  GEMINI_API_KEYS="key1,key2" python scripts/test_gemini_keys.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

from dotenv import load_dotenv

load_dotenv()


def parse_keys(args) -> list[str]:
    keys: list[str] = []
    if args.key:
        keys.append(args.key.strip())
    if args.keys:
        for part in re.split(r"[,;\s]+", args.keys.strip()):
            if part.strip():
                keys.append(part.strip())
    if not keys:
        multi = os.getenv("GEMINI_API_KEYS", "")
        if multi.strip():
            keys.extend(p.strip() for p in re.split(r"[,;\s]+", multi) if p.strip())
        for env_name in ("GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY"):
            single = os.getenv(env_name, "").strip()
            if single and single not in keys:
                keys.append(single)
    # de-dupe, keep order
    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def mask_key(key: str) -> str:
    if len(key) <= 10:
        return "***"
    return f"{key[:6]}...{key[-4:]}"


def test_key(key: str, model: str, prompt: str, timeout_hint: float) -> dict:
    from google import genai
    from google.genai import types

    started = time.time()
    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        text = (getattr(response, "text", None) or "").strip()
        elapsed = time.time() - started
        if not text:
            return {
                "ok": False,
                "status": "EMPTY",
                "message": "API responded but returned no text",
                "elapsed_s": round(elapsed, 2),
            }
        return {
            "ok": True,
            "status": "OK",
            "message": text[:120],
            "elapsed_s": round(elapsed, 2),
        }
    except Exception as exc:
        elapsed = time.time() - started
        msg = str(exc)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
            status = "RATE_LIMIT"
        elif "403" in msg or "PERMISSION_DENIED" in msg:
            status = "DENIED"
        elif "401" in msg or "API key not valid" in msg:
            status = "INVALID"
        else:
            status = "ERROR"
        retry = re.search(r"retry in ([\d.]+)s", msg, re.IGNORECASE)
        return {
            "ok": False,
            "status": status,
            "message": msg[:240],
            "retry_in_s": float(retry.group(1)) if retry else None,
            "elapsed_s": round(elapsed, 2),
        }


def test_grok_key(key: str, model: str, prompt: str) -> dict:
    import os
    os.environ["GROK_API_KEY"] = key
    from importlib import reload
    import src.infrastructure.grok.client as grok_mod
    reload(grok_mod)
    started = time.time()
    text = grok_mod.generate_grok(prompt, model_name=model)
    elapsed = time.time() - started
    if text:
        return {"ok": True, "status": "OK", "message": text[:120], "elapsed_s": round(elapsed, 2)}
    return {"ok": False, "status": "ERROR", "message": "No response from Grok", "elapsed_s": round(elapsed, 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Gemini API keys")
    parser.add_argument("--key", help="Single API key to test")
    parser.add_argument("--keys", help="Comma-separated API keys")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--prompt", default="Reply with exactly one word: OK", help="Test prompt")
    parser.add_argument("--test-grok", action="store_true", help="Test GROK_API_KEY from .env")
    parser.add_argument("--grok-model", default="grok-2-latest", help="Grok model name")
    args = parser.parse_args()

    if args.test_grok:
        grok_key = os.getenv("GROK_API_KEY", "").strip()
        if not grok_key:
            print("GROK_API_KEY not set in environment / .env")
            return 1
        print(f"Testing Grok key {mask_key(grok_key)} model={args.grok_model}\n")
        result = test_grok_key(grok_key, args.grok_model, args.prompt)
        print(f"  status : {result['status']}")
        print(f"  reply  : {result.get('message', '')}")
        print(f"  time   : {result['elapsed_s']}s")
        return 0 if result["ok"] else 1

    keys = parse_keys(args)
    if not keys:
        print("No API keys found.")
        print("Set GOOGLE_GEMINI_API_KEY, GEMINI_API_KEYS, or pass --key / --keys")
        return 1

    print(f"Testing {len(keys)} key(s) with model={args.model}\n")

    ok_count = 0
    for i, key in enumerate(keys, start=1):
        print(f"[{i}/{len(keys)}] {mask_key(key)}")
        result = test_key(key, args.model, args.prompt, timeout_hint=30.0)
        if result["ok"]:
            ok_count += 1
            print(f"  status : {result['status']}")
            print(f"  reply  : {result['message']}")
            print(f"  time   : {result['elapsed_s']}s")
        else:
            print(f"  status : {result['status']}")
            print(f"  error  : {result['message']}")
            if result.get("retry_in_s") is not None:
                print(f"  retry  : wait {result['retry_in_s']}s")
            print(f"  time   : {result['elapsed_s']}s")
        print()

    print(f"Summary: {ok_count}/{len(keys)} keys working")
    return 0 if ok_count == len(keys) else 1


if __name__ == "__main__":
    raise SystemExit(main())
