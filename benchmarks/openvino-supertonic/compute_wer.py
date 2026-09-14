#!/usr/bin/env python3
"""Compute normalized word error rate for a single reference/hypothesis pair."""

from __future__ import annotations

import json
import re
import sys


def words(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    normalized: list[str] = []
    index = 0
    while index < len(tokens):
        if tokens[index : index + 2] == ["oma", "speak"]:
            normalized.append("omaspeak")
            index += 2
        else:
            normalized.append(tokens[index])
            index += 1
    return normalized


def distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} REFERENCE HYPOTHESIS", file=sys.stderr)
        return 2
    reference = words(sys.argv[1])
    hypothesis = words(sys.argv[2])
    errors = distance(reference, hypothesis)
    json.dump(
        {
            "reference": reference,
            "hypothesis": hypothesis,
            "word_errors": errors,
            "reference_words": len(reference),
            "wer": errors / len(reference) if reference else None,
        },
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
