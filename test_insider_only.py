"""
Minimal script to verify insider endpoint + parsing only.

Run: python test_insider_only.py
Requires QUIVER_API_TOKEN in the environment.
"""

from __future__ import annotations

import os
import sys

from quiver_api import get_insider_trading

API_KEY = os.getenv("QUIVER_API_TOKEN", "").strip()

if not API_KEY:
    print("Missing QUIVER_API_TOKEN. Please set it in your environment before running.")
    sys.exit(1)

df = get_insider_trading(API_KEY)

print(f"Insider rows: {len(df)}")
print(f"Insider columns: {df.columns.tolist()}")
print()
print(df.head(5))
