#!/usr/bin/env python3
import os
import sys
import asyncio
from pathlib import Path

def _get_private_key() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1].strip()
    return os.getenv("POLYMARKET_PRIVATE_KEY", "").strip()

async def _derive():
    try:
        from py_clob_client.client import ClobClient
    except ImportError:
        print("Missing dependency: py-clob-client")
        print("Install with: pip install py-clob-client")
        return 1

    private_key = _get_private_key()
    if not private_key:
        print("Provide your wallet private key via:")
        print("  1) POLYMARKET_PRIVATE_KEY env var, or")
        print("  2) first CLI argument.")
        return 1

    host = os.getenv("POLYMARKET_CLOB_HOST", "https://clob.polymarket.com")
    chain_id = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))

    client = ClobClient(host, key=private_key, chain_id=chain_id)
    api_creds = await client.create_or_derive_api_key()

    key = api_creds.get("apiKey") or api_creds.get("key")
    secret = api_creds.get("secret")
    passphrase = api_creds.get("passphrase")

    if not (key and secret and passphrase):
        print("Unexpected API creds response:", api_creds)
        return 1

    print("Export these credentials:")
    print(f"export POLYMARKET_CLOB_API_KEY=\"{key}\"")
    print(f"export POLYMARKET_CLOB_API_SECRET=\"{secret}\"")
    print(f"export POLYMARKET_CLOB_API_PASSPHRASE=\"{passphrase}\"")

    address = None
    try:
        from eth_account import Account
        address = Account.from_key(private_key).address
        print(f"export POLYMARKET_CLOB_ADDRESS=\"{address}\"")
    except Exception:
        print("Set POLYMARKET_CLOB_ADDRESS to your wallet address for trade fetches.")

    if os.getenv("WRITE_DOTENV", "1") == "1":
        dotenv_path = Path(os.getenv("POLYMARKET_DOTENV_PATH", ".env"))
        _write_env_file(dotenv_path, {
            "POLYMARKET_CLOB_API_KEY": key,
            "POLYMARKET_CLOB_API_SECRET": secret,
            "POLYMARKET_CLOB_API_PASSPHRASE": passphrase,
            "POLYMARKET_CLOB_ADDRESS": address or ""
        })
        print(f"Wrote credentials to {dotenv_path}")
    return 0

def _quote_env(value: str) -> str:
    escaped = value.replace('"', '\\"')
    return f"\"{escaped}\""

def _write_env_file(path: Path, updates: dict) -> None:
    lines = []
    used = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                lines.append(line)
                continue
            key, _ = line.split("=", 1)
            key = key.strip()
            if key in updates:
                lines.append(f"{key}={_quote_env(str(updates[key]))}")
                used.add(key)
            else:
                lines.append(line)
    for key, value in updates.items():
        if key not in used:
            lines.append(f"{key}={_quote_env(str(value))}")
    path.write_text("\n".join(lines) + "\n")

def main() -> int:
    return asyncio.run(_derive())

if __name__ == "__main__":
    raise SystemExit(main())
