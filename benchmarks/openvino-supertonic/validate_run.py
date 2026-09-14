#!/usr/bin/env python3
"""Fail a hardware run when a selected lane lacks required proof artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print(f"usage: {Path(sys.argv[0]).name} RESULT_ROOT BACKEND...", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    backends = sys.argv[2:]
    errors: list[str] = []
    for backend in backends:
        expected_runtime = "default" if backend == "default-cpu" else "openvino"
        expected_device = backend.rsplit("-", 1)[-1]
        for phase, warmup, iterations in (("cold", 0, 1), ("hot", 2, 10)):
            case = root / backend / phase
            status_path = case.with_suffix(".exit-status.txt")
            benchmark_path = case.with_suffix(".benchmark.json")
            if not status_path.is_file() or status_path.read_text().strip() != "0":
                errors.append(f"{backend}/{phase}: process did not exit 0")
                continue
            try:
                benchmark = json.loads(benchmark_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f"{backend}/{phase}: invalid benchmark JSON: {error}")
                continue
            if benchmark.get("warmup_iterations") != warmup:
                errors.append(f"{backend}/{phase}: expected {warmup} warmups")
            if benchmark.get("measured_iterations") != iterations:
                errors.append(f"{backend}/{phase}: expected {iterations} iterations")
            if len(benchmark.get("iterations", [])) != iterations:
                errors.append(f"{backend}/{phase}: incomplete iteration results")
            placement = benchmark.get("backend", {})
            if placement.get("effective_runtime") != expected_runtime:
                errors.append(
                    f"{backend}/{phase}: expected effective runtime "
                    f"{expected_runtime}, got {placement.get('effective_runtime')}"
                )
            if placement.get("requested_device") != expected_device:
                errors.append(
                    f"{backend}/{phase}: expected requested device "
                    f"{expected_device}, got {placement.get('requested_device')}"
                )
            if placement.get("fallback_used") is not False:
                errors.append(f"{backend}/{phase}: runtime fallback was used")
            if placement.get("placement_verified") is not True:
                errors.append(f"{backend}/{phase}: placement was not verified")
            if not placement.get("placement_evidence"):
                errors.append(f"{backend}/{phase}: placement evidence is empty")
            for iteration in benchmark.get("iterations", []):
                if not Path(iteration["output"]).is_file():
                    errors.append(f"{backend}/{phase}: missing {iteration['output']}")

        asr_status = root / backend / "cold.asr.exit-status.txt"
        wer_path = root / backend / "cold.wer.json"
        if not asr_status.is_file() or asr_status.read_text().strip() != "0":
            errors.append(f"{backend}: ASR did not exit 0")
        try:
            wer = json.loads(wer_path.read_text()).get("wer")
            if wer != 0.0:
                errors.append(f"{backend}: expected zero normalized WER, got {wer}")
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{backend}: invalid WER JSON: {error}")

    if "openvino-npu" in backends:
        deltas = []
        for phase in ("cold", "hot"):
            path = root / "openvino-npu" / f"{phase}.npu-busy-delta-us.txt"
            try:
                deltas.append(int(path.read_text().strip()))
            except (OSError, ValueError) as error:
                errors.append(f"openvino-npu/{phase}: invalid busy delta: {error}")
        if deltas and sum(deltas) <= 0:
            errors.append("openvino-npu: busy counter did not increase")

    if errors:
        print("FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
