#!/usr/bin/env python3
"""File-only Kokoro/OpenVINO GenAI device probe."""

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path, help="local Intel Kokoro IR snapshot")
    parser.add_argument("--device", default="NPU", choices=("CPU", "GPU", "NPU"))
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeat", type=int, default=2, help="number of synthesis runs (default: 2)")
    parser.add_argument("text", help="text to synthesize")
    args = parser.parse_args()

    if not args.text.strip():
        parser.error("text must not be empty")
    if not args.voice.startswith("a"):
        parser.error("this probe only prepares American English voices (a prefix)")
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    for name in ("openvino_model.xml", "openvino_model.bin", "config.json", f"voices/{args.voice}.bin"):
        if not (args.model_dir / name).is_file():
            parser.error(f"missing model asset: {args.model_dir / name}")

    try:
        import numpy as np
        import openvino as ov
        import openvino_genai as genai
        import soundfile as sf
    except ImportError as exc:
        parser.error(f"missing Python dependency: {exc}")

    if tuple(int(part) for part in ov.__version__.split(".")[:2]) < (2026, 4):
        parser.error(f"OpenVINO 2026.4 or newer is required; found {ov.__version__}")
    available = ov.Core().available_devices
    if not any(item.upper().split(".")[0] == args.device for item in available):
        parser.error(f"{args.device} is unavailable; OpenVINO reports {available}")

    voice = np.fromfile(args.model_dir / "voices" / f"{args.voice}.bin", dtype=np.float32)
    started = time.monotonic()
    pipeline = genai.Text2SpeechPipeline(str(args.model_dir), args.device)
    shape = tuple(pipeline.get_speaker_embedding_shape())
    if voice.size != int(np.prod(shape)):
        raise ValueError(f"voice has {voice.size} floats; pipeline expects {shape}")
    embedding = ov.Tensor(voice.reshape(shape))
    loaded = time.monotonic()

    for run in range(1, args.repeat + 1):
        begun = time.monotonic()
        result = pipeline.generate(args.text, embedding, language="en-us")
        elapsed = time.monotonic() - begun
        if len(result.speeches) != 1:
            raise ValueError(f"expected one speech, got {len(result.speeches)}")
        samples = np.asarray(result.speeches[0].data).squeeze()
        if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
            raise ValueError(f"invalid mono audio: shape={samples.shape}")
        rate = result.output_sample_rate
        if rate != 24_000:
            raise ValueError(f"expected Kokoro 24 kHz, got {rate}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.output, samples, rate)
        print(
            f"run={run} device={args.device} openvino={ov.__version__} genai={genai.__version__} "
            f"load_s={loaded-started:.3f} synth_s={elapsed:.3f} "
            f"samples={samples.size} peak={np.max(np.abs(samples)):.4f} output={args.output}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
