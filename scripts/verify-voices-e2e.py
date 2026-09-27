#!/usr/bin/env python3
"""Synthesize one real WAV for every advertised voice in the active model.

Use an isolated, already activated XDG profile. The app performs inference in
Rust/native providers; this driver records audio and JSON evidence only.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--provider-library", type=Path, help="optional Kokoro native bridge")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    profile = args.profile.resolve(strict=True)
    output = args.artifacts.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, XDG_CONFIG_HOME=str(profile / "config"),
               XDG_DATA_HOME=str(profile / "data"), XDG_CACHE_HOME=str(profile / "cache"),
               XDG_RUNTIME_DIR=str(profile / "run"))
    if args.provider_library:
        env["OMASPEAK_KOKORO_LIBRARY"] = str(args.provider_library.resolve(strict=True))
    schema = subprocess.run([str(binary), "config", "schema", "--json"], env=env,
                            capture_output=True, text=True, check=True, timeout=30)
    voice = next(item for item in json.loads(schema.stdout)["keys"] if item["key"] == "model.voice")
    choices = voice["choices"]
    if not choices:
        raise AssertionError("the active model advertises no voices")
    results = []
    for index, choice in enumerate(choices):
        label = choice["label"] if isinstance(choice, dict) else str(choice)
        token = choice["value"] if isinstance(choice, dict) else choice
        path = output / f"{index:02d}-{label}.wav"
        process = subprocess.run([str(binary), "say", "Hello.", "--voice", str(token),
                                  "--no-play", "--out", str(path)], env=env,
                                 capture_output=True, text=True, timeout=300)
        result = {"voice": token, "label": label, "exit_code": process.returncode,
                  "stderr": process.stderr}
        try:
            if process.returncode:
                raise AssertionError(process.stderr)
            result["report"] = json.loads(process.stdout)
            with wave.open(str(path)) as audio:
                frames = audio.readframes(audio.getnframes())
                if audio.getnchannels() != 1 or len(frames) < 2000 or not any(frames):
                    raise AssertionError("invalid or silent WAV")
                result["sample_rate"] = audio.getframerate()
                result["frames"] = audio.getnframes()
                result["sha256"] = hashlib.sha256(frames).hexdigest()
            result["status"] = "pass"
        except Exception as error:
            result["status"] = "fail"
            result["error"] = str(error)
        results.append(result)
        print(f"{result['status']}: {label}", flush=True)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if len({result.get("sha256") for result in results}) != len(results):
        raise AssertionError("two voice outputs are identical")
    raise SystemExit(any(result["status"] != "pass" for result in results))


if __name__ == "__main__":
    main()
