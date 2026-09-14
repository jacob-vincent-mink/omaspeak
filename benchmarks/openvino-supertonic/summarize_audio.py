#!/usr/bin/env python3
"""Record compact validity and level metrics for benchmark WAV files."""

from __future__ import annotations

import array
import hashlib
import json
import math
import sys
import wave
from pathlib import Path


def summarize(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        samples = array.array("h", wav.readframes(frames))
    if sample_width != 2:
        raise ValueError(f"unsupported sample width {sample_width}: {path}")
    if sys.byteorder != "little":
        samples.byteswap()
    squares = sum(sample * sample for sample in samples)
    rms = math.sqrt(squares / len(samples)) if samples else 0.0
    peak = max((abs(sample) for sample in samples), default=0)
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_rate": sample_rate,
        "frames": frames,
        "duration_seconds": frames / sample_rate,
        "rms_dbfs": 20 * math.log10(rms / 32767) if rms else None,
        "peak_dbfs": 20 * math.log10(peak / 32767) if peak else None,
        "zero_fraction": sum(sample == 0 for sample in samples) / len(samples),
        "clipped_fraction": sum(abs(sample) == 32767 for sample in samples) / len(samples),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} AUDIO_ROOT", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    json.dump(
        [summarize(path, root) for path in sorted(root.rglob("*.wav"))],
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
