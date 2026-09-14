#!/usr/bin/env python3
"""Summarize ORT profiles without copying large trace files into the repository."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def summarize(directory: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    providers: Counter[str] = Counter()
    openvino_nodes: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "duration_microseconds": 0}
    )
    event_count = 0
    invalid_files = 0

    for path in sorted(directory.glob("*.json")):
        payload = path.read_bytes()
        entry: dict[str, object] = {
            "name": path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        try:
            events = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            entry["parse_error"] = str(error)
            invalid_files += 1
        else:
            entry["events"] = len(events)
            event_count += len(events)
            for event in events:
                args = event.get("args", {})
                provider = args.get("provider")
                if provider:
                    providers[str(provider)] += 1
                if provider == "OpenVINOExecutionProvider":
                    name = str(event.get("name", ""))
                    node = openvino_nodes[name]
                    node["count"] += 1
                    node["duration_microseconds"] += int(event.get("dur", 0))
        files.append(entry)

    return {
        "profile_files": files,
        "profile_file_count": len(files),
        "invalid_profile_files": invalid_files,
        "event_count": event_count,
        "provider_event_counts": dict(sorted(providers.items())),
        "openvino_nodes": dict(sorted(openvino_nodes.items())),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} PROFILE_ROOT", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    directories = (
        [root]
        if any(root.glob("*.json"))
        else [path for path in sorted(root.iterdir()) if path.is_dir()]
    )
    result = {
        path.name: summarize(path)
        for path in directories
    }
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
