#!/usr/bin/env bash
set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)
run_id=${RUN_ID:-$(date -u +%Y-%m-%dT%H%M%SZ)}
binary=${BINARY:-$repo_dir/target/release/omaspeak}
runtime_lib=${RUNTIME_LIB:-/tmp/oma-native/runtime/lib}
openvino_lib=${OPENVINO_LIB:-/tmp/oma-native/openvino-2026.2.1/runtime/lib/intel64}
tbb_lib=${TBB_LIB:-/tmp/oma-native/openvino-2026.2.1/runtime/3rdparty/tbb/lib}
ort_library=${ORT_LIBRARY:-$runtime_lib/libonnxruntime.so.1.29.0}
openvino_library=${OPENVINO_LIBRARY:-$openvino_lib/libopenvino_c.so}
openvino_plugins=${OPENVINO_PLUGINS:-$openvino_lib/plugins.xml}
model_name=${MODEL_NAME:-supertonic-3-npu}
model_dir=${MODEL_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/omaspeak/models/$model_name}
vector_estimator=${VECTOR_ESTIMATOR:-vector_estimator.onnx}
voxtype=${VOXTYPE:-$HOME/.local/bin/voxtype}
result_root=${RESULT_ROOT:-$script_dir/results/$run_id-npu-accuracy}
work_root=${WORK_ROOT:-/tmp/omaspeak-openvino-supertonic/$run_id-npu-accuracy}
npu_busy_path=${NPU_BUSY_PATH:-/sys/devices/pci0000:00/0000:00:0b.0/npu_busy_time_us}
max_wer_delta=${MAX_WER_DELTA:-0.10}

phrases=(
  'Omaspeak now supports Supertonic.'
  'Turn on the living room lights.'
  'Schedule a meeting for tomorrow morning.'
  'The quick brown fox jumps over the lazy dog.'
)

for required in "$binary" "$ort_library" "$openvino_library" \
  "$openvino_plugins" "$model_dir/tts.json" "$model_dir/unicode_indexer.bin" \
  "$model_dir/voice.bin" "$model_dir/duration_predictor.int8.onnx" \
  "$model_dir/text_encoder.int8.onnx" "$model_dir/$vector_estimator" \
  "$model_dir/vocoder.int8.onnx" "$voxtype" "$npu_busy_path"; do
  if [[ ! -f $required ]]; then
    printf 'Required file is missing: %s\n' "$required" >&2
    exit 2
  fi
done

mkdir -p "$result_root" "$work_root/configs" "$work_root/audio" "$work_root/state"

render_config() {
  local lane=$1
  sed -e "s|@MODEL_DIR@|$model_dir|g" \
    -e "s|@MODEL_NAME@|$model_name|g" \
    -e "s|@VECTOR_ESTIMATOR@|$vector_estimator|g" \
    "$script_dir/configs/$lane.toml.in" >"$work_root/configs/$lane.toml"
  cp "$work_root/configs/$lane.toml" "$result_root/$lane.config.toml"
}

render_config default-cpu
render_config openvino-npu

{
  printf 'run_id=%s\n' "$run_id"
  printf 'binary=%s\n' "$binary"
  printf 'ort_library=%s\n' "$ort_library"
  printf 'openvino_library=%s\n' "$openvino_library"
  printf 'openvino_plugins=%s\n' "$openvino_plugins"
  printf 'model_name=%s\n' "$model_name"
  printf 'model_dir=%s\n' "$model_dir"
  printf 'vector_estimator=%s\n' "$vector_estimator"
  printf 'max_wer_delta=%s\n' "$max_wer_delta"
  printf 'git_head=%s\n' "$(git -C "$repo_dir" rev-parse HEAD)"
  printf 'kernel=%s\n' "$(uname -srmo)"
} >"$result_root/environment.txt"
sha256sum "$binary" "$ort_library" "$openvino_library" \
  "$openvino_plugins" >>"$result_root/environment.txt"

run_phrase() {
  local lane=$1
  local index=$2
  local text=$3
  local case_dir=$result_root/$lane/phrase-$index
  local audio_dir=$work_root/audio/$lane/phrase-$index
  local state_dir=$work_root/state/$lane/phrase-$index
  mkdir -p "$case_dir" "$audio_dir" "$state_dir"
  printf '%s\n' "$text" >"$case_dir/reference.txt"

  if [[ $lane == openvino-npu ]]; then
    cat "$npu_busy_path" >"$case_dir/npu-busy-before-us.txt"
  fi

  (cd "$work_root" && env \
    LD_LIBRARY_PATH="$runtime_lib:$openvino_lib:$tbb_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    OMASPEAK_ONNXRUNTIME_LIBRARY="$ort_library" \
    OMASPEAK_OPENVINO_LIBRARY="$openvino_library" \
    OMASPEAK_OPENVINO_PLUGINS="$openvino_plugins" \
    XDG_STATE_HOME="$state_dir" \
    "$binary" --config "$work_root/configs/$lane.toml" benchmark \
      --text "$text" --out-dir "$audio_dir" --warmup 0 --iterations 1 \
      >"$case_dir/benchmark.raw.log" 2>"$case_dir/stderr.log")
  local status=$?
  printf '%s\n' "$status" >"$case_dir/exit-status.txt"

  if [[ $lane == openvino-npu ]]; then
    cat "$npu_busy_path" >"$case_dir/npu-busy-after-us.txt"
    awk 'NR==FNR { before=$1; next } { print $1-before }' \
      "$case_dir/npu-busy-before-us.txt" "$case_dir/npu-busy-after-us.txt" \
      >"$case_dir/npu-busy-delta-us.txt"
  fi

  python3 "$script_dir/extract_benchmark_json.py" \
    "$case_dir/benchmark.raw.log" "$case_dir/benchmark.json" \
    2>"$case_dir/json-error.log" || return 1
  [[ $status == 0 ]] || return 1

  local wav=$audio_dir/iteration-0001.wav
  [[ -f $wav ]] || return 1
  "$voxtype" transcribe "$wav" >"$case_dir/asr.raw.log" \
    2>"$case_dir/asr.stderr.log"
  local asr_status=$?
  printf '%s\n' "$asr_status" >"$case_dir/asr.exit-status.txt"
  local hypothesis
  hypothesis=$(awk 'NF { line=$0 } END { print line }' "$case_dir/asr.raw.log")
  printf '%s\n' "$hypothesis" >"$case_dir/transcript.txt"
  python3 "$script_dir/compute_wer.py" "$text" "$hypothesis" \
    >"$case_dir/wer.json"
  [[ $asr_status == 0 ]]
}

overall_status=0
for lane in default-cpu openvino-npu; do
  for index in "${!phrases[@]}"; do
    run_phrase "$lane" "$((index + 1))" "${phrases[$index]}" || overall_status=1
  done
done

python3 - "$result_root" "$max_wer_delta" <<'PY' >"$result_root/accuracy-summary.json" || overall_status=1
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
limit = float(sys.argv[2])
summary = {"schema_version": 1, "max_wer_delta": limit, "lanes": {}}
failures = []
for lane in ("default-cpu", "openvino-npu"):
    errors = 0
    words = 0
    placements = []
    npu_busy = 0
    for case in sorted((root / lane).glob("phrase-*")):
        try:
            wer = json.loads((case / "wer.json").read_text())
            benchmark = json.loads((case / "benchmark.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"{lane}/{case.name}: incomplete proof: {error}")
            continue
        errors += int(wer["word_errors"])
        words += int(wer["reference_words"])
        backend = benchmark["backend"]
        placements.append(
            {
                "effective_runtime": backend["effective_runtime"],
                "requested_device": backend["requested_device"],
                "fallback_used": backend["fallback_used"],
                "placement_verified": backend["placement_verified"],
                "placement_evidence": backend["placement_evidence"],
            }
        )
        busy = case / "npu-busy-delta-us.txt"
        if busy.is_file():
            npu_busy += int(busy.read_text().strip())
    summary["lanes"][lane] = {
        "word_errors": errors,
        "reference_words": words,
        "wer": errors / words if words else None,
        "npu_busy_delta_us": npu_busy,
        "placements": placements,
    }

cpu = summary["lanes"]["default-cpu"]
npu = summary["lanes"]["openvino-npu"]
summary["npu_minus_cpu_wer"] = (
    npu["wer"] - cpu["wer"]
    if npu["wer"] is not None and cpu["wer"] is not None
    else None
)
for placement in cpu["placements"]:
    if (
        placement["effective_runtime"] != "default"
        or placement["requested_device"] != "cpu"
        or placement["fallback_used"]
        or not placement["placement_verified"]
        or not placement["placement_evidence"]
    ):
        failures.append("default CPU placement validation failed")
for placement in npu["placements"]:
    if (
        placement["effective_runtime"] != "openvino"
        or placement["requested_device"] != "npu"
        or placement["fallback_used"]
        or not placement["placement_verified"]
        or not placement["placement_evidence"]
    ):
        failures.append("OpenVINO NPU placement validation failed")
if npu["npu_busy_delta_us"] <= 0:
    failures.append("kernel NPU busy-time counter did not increase")
if summary["npu_minus_cpu_wer"] is None:
    failures.append("CPU or NPU accuracy measurements are incomplete")
elif summary["npu_minus_cpu_wer"] > limit:
    failures.append(
        f"NPU WER degraded by {summary['npu_minus_cpu_wer']:.4f}; limit is {limit:.4f}"
    )
summary["passed"] = not failures
summary["failures"] = sorted(set(failures))
json.dump(summary, sys.stdout, indent=2, sort_keys=True)
print()
if failures:
    raise SystemExit(1)
PY

python3 "$script_dir/summarize_audio.py" "$work_root/audio" \
  >"$result_root/audio-summary.json" || overall_status=1
printf '%s\n' "$overall_status" >"$result_root/exit-status.txt"
printf 'Results: %s\n' "$result_root"
exit "$overall_status"
