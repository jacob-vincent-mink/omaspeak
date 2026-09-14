#!/usr/bin/env python3
"""Extract Omaspeak's JSON object when a native plugin writes a stdout prefix."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {Path(sys.argv[0]).name} RAW_JSON OUTPUT_JSON", file=sys.stderr)
        return 2
    raw = Path(sys.argv[1]).read_text()
    start = raw.find("{")
    if start < 0:
        return 1
    payload = json.loads(raw[start:])
    Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
