"""Embedded Omaspeak Kokoro/OpenVINO GenAI worker protocol.

One JSON request per line on stdin; one JSON header and raw float32 payload on
stdout. Diagnostics belong on stderr so they cannot corrupt the audio stream.
"""

import argparse
import json
import re
import sys
from pathlib import Path


def version_tuple(value):
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
    if not match:
        raise ValueError(f"unrecognized OpenVINO version: {value}")
    return tuple(int(part or 0) for part in match.groups())


def send_header(value):
    sys.stdout.buffer.write(json.dumps(value, separators=(",", ":")).encode() + b"\n")
    sys.stdout.buffer.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("device", choices=("CPU", "GPU", "NPU"))
    parser.add_argument("min_openvino_version")
    args = parser.parse_args()

    import numpy as np
    import openvino as ov
    import openvino_genai as genai

    minimum = version_tuple(args.min_openvino_version)
    if version_tuple(ov.__version__) < minimum:
        raise RuntimeError(f"Kokoro needs OpenVINO >={args.min_openvino_version}; found {ov.__version__}")
    if version_tuple(genai.__version__) < minimum:
        raise RuntimeError(f"Kokoro needs OpenVINO GenAI >={args.min_openvino_version}; found {genai.__version__}")
    if args.device not in (item.split(".")[0] for item in ov.Core().available_devices):
        raise RuntimeError(f"OpenVINO device {args.device} is unavailable")

    pipeline = genai.Text2SpeechPipeline(str(args.model_dir), args.device)
    shape = tuple(pipeline.get_speaker_embedding_shape())
    embeddings = []
    for voice in ("af_heart", "am_michael"):
        path = args.model_dir / "voices" / f"{voice}.bin"
        data = np.fromfile(path, dtype=np.float32)
        if data.size != int(np.prod(shape)):
            raise RuntimeError(f"{path} has {data.size} floats; expected {shape}")
        embeddings.append(ov.Tensor(data.reshape(shape)))
    send_header({"ready": True, "sample_rate": 24000, "openvino": ov.__version__,
                 "genai": genai.__version__, "device": args.device})

    for line in sys.stdin.buffer:
        try:
            request = json.loads(line)
            voice = request["voice"]
            if type(voice) is not int or not 0 <= voice < len(embeddings):
                raise ValueError(f"invalid Kokoro voice {voice!r}")
            if request["speed"] != 1.0:
                raise ValueError("Kokoro OpenVINO GenAI currently supports speed 1.0")
            result = pipeline.generate(request["text"], embeddings[voice], language="en-us")
            if result.output_sample_rate != 24000 or len(result.speeches) != 1:
                raise RuntimeError("unexpected Kokoro audio format")
            samples = np.asarray(result.speeches[0].data, dtype="<f4").reshape(-1)
            if not samples.size or not np.isfinite(samples).all():
                raise RuntimeError("Kokoro returned empty or non-finite audio")
            data = samples.tobytes()
            send_header({"ok": True, "bytes": len(data)})
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        except Exception as exc:
            send_header({"error": str(exc)})


if __name__ == "__main__":
    main()
