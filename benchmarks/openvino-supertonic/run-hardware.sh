#!/usr/bin/env bash
set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)
run_id=${RUN_ID:-$(date -u +%Y-%m-%dT%H%M%SZ)}
binary=${BINARY:-$repo_dir/target-openvino/release/omaspeak}
runtime_lib=${RUNTIME_LIB:-/tmp/oma-native/runtime/lib}
openvino_lib=${OPENVINO_LIB:-/tmp/oma-native/openvino-2026.2.1/runtime/lib/intel64}
tbb_lib=${TBB_LIB:-/tmp/oma-native/openvino-2026.2.1/runtime/3rdparty/tbb/lib}
ort_library=${ORT_LIBRARY:-$runtime_lib/libonnxruntime.so.1.29.0}
openvino_library=${OPENVINO_LIBRARY:-$openvino_lib/libopenvino_c.so}
openvino_plugins=${OPENVINO_PLUGINS:-$openvino_lib/plugins.xml}
model_name=${MODEL_NAME:-supertonic-3-npu}
model_dir=${MODEL_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/omaspeak/models/$model_name}
if [[ $model_name == supertonic-3-npu ]]; then
  vector_estimator=${VECTOR_ESTIMATOR:-vector_estimator.onnx}
else
  vector_estimator=${VECTOR_ESTIMATOR:-vector_estimator.int8.onnx}
fi
result_root=${RESULT_ROOT:-$script_dir/results/$run_id}
work_root=${WORK_ROOT:-/tmp/omaspeak-openvino-supertonic/$run_id}
npu_busy_path=${NPU_BUSY_PATH:-/sys/devices/pci0000:00/0000:00:0b.0/npu_busy_time_us}
voxtype=${VOXTYPE:-$HOME/.local/bin/voxtype}
text=${TEXT:-Omaspeak now supports Supertonic.}
read -r -a backends <<<"${BACKENDS:-default-cpu openvino-cpu openvino-gpu openvino-npu}"
overall_status=0

if ((${#backends[@]} == 0)); then
  printf 'BACKENDS selected no benchmark lanes\n' >&2
  exit 2
fi
declare -A seen_backends=()
for backend in "${backends[@]}"; do
  case "$backend" in
    default-cpu|openvino-cpu|openvino-gpu|openvino-npu) ;;
    *)
      printf 'Unknown benchmark lane: %s\n' "$backend" >&2
      exit 2
      ;;
  esac
  if [[ -n ${seen_backends[$backend]:-} ]]; then
    printf 'Duplicate benchmark lane: %s\n' "$backend" >&2
    exit 2
  fi
  seen_backends[$backend]=1
done

for required in "$binary" "$ort_library" "$openvino_library" \
  "$openvino_plugins" "$model_dir/tts.json" "$model_dir/unicode_indexer.bin" \
  "$model_dir/voice.bin" "$model_dir/duration_predictor.int8.onnx" \
  "$model_dir/text_encoder.int8.onnx" "$model_dir/$vector_estimator" \
  "$model_dir/vocoder.int8.onnx" "$voxtype"; do
  if [[ ! -f $required ]]; then
    printf 'Required file is missing: %s\n' "$required" >&2
    exit 2
  fi
done

mkdir -p "$result_root" "$work_root/configs" "$work_root/audio"

{
  printf 'run_id=%s\n' "$run_id"
  printf 'binary=%s\n' "$binary"
  printf 'runtime_lib=%s\n' "$runtime_lib"
  printf 'ort_library=%s\n' "$ort_library"
  printf 'openvino_lib=%s\n' "$openvino_lib"
  printf 'openvino_library=%s\n' "$openvino_library"
  printf 'openvino_plugins=%s\n' "$openvino_plugins"
  printf 'tbb_lib=%s\n' "$tbb_lib"
  printf 'model_name=%s\n' "$model_name"
  printf 'vector_estimator=%s\n' "$vector_estimator"
  printf 'model_dir=%s\n' "$model_dir"
  printf 'work_root=%s\n' "$work_root"
  printf 'voxtype=%s\n' "$voxtype"
  printf 'text=%s\n' "$text"
  printf 'backends=%s\n' "${backends[*]}"
  printf 'git_head=%s\n' "$(git -C "$repo_dir" rev-parse HEAD)"
  printf 'kernel=%s\n' "$(uname -srmo)"
} >"$result_root/environment.txt"
sha256sum "$binary" "$ort_library" "$openvino_library" \
  "$openvino_plugins" >>"$result_root/environment.txt"
lspci -nn | grep -Ei 'vga|display|3d|processing accelerator' \
  >"$result_root/pci-devices.txt" 2>&1 || true

render_config() {
  local backend=$1
  local template=$script_dir/configs/$backend.toml.in
  local config=$work_root/configs/$backend.toml
  sed -e "s|@MODEL_DIR@|$model_dir|g" \
    -e "s|@MODEL_NAME@|$model_name|g" \
    -e "s|@VECTOR_ESTIMATOR@|$vector_estimator|g" \
    "$template" >"$config"
  cp "$config" "$result_root/$backend.config.toml"
}

run_case() {
  local backend=$1
  local phase=$2
  local warmup=$3
  local iterations=$4
  local config=${5:-$work_root/configs/$backend.toml}
  local output_dir=$work_root/audio/$backend-$phase
  local case_dir=$result_root/$backend
  mkdir -p "$output_dir" "$case_dir"

  if [[ $backend == openvino-npu && -r $npu_busy_path ]]; then
    cat "$npu_busy_path" >"$case_dir/$phase.npu-busy-before-us.txt"
  fi

  local started_ns
  local finished_ns
  started_ns=$(date +%s%N)
  (cd "$work_root" && env LD_LIBRARY_PATH="$runtime_lib:$openvino_lib:$tbb_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      OMASPEAK_ONNXRUNTIME_LIBRARY="$ort_library" \
      OMASPEAK_OPENVINO_LIBRARY="$openvino_library" \
      OMASPEAK_OPENVINO_PLUGINS="$openvino_plugins" \
      XDG_STATE_HOME="$work_root/state/$backend-$phase" \
      "$binary" --config "$config" benchmark --text "$text" \
      --out-dir "$output_dir" --warmup "$warmup" --iterations "$iterations" \
      >"$case_dir/$phase.benchmark.raw.log" 2>"$case_dir/$phase.stderr.log")
  local status=$?
  finished_ns=$(date +%s%N)
  printf '%s\n' "$status" >"$case_dir/$phase.exit-status.txt"
  awk -v started="$started_ns" -v finished="$finished_ns" \
    'BEGIN { printf "{\"elapsed_seconds\":%.6f}\n", (finished-started)/1000000000 }' \
    >"$case_dir/$phase.process.json"
  if ! python3 "$script_dir/extract_benchmark_json.py" \
    "$case_dir/$phase.benchmark.raw.log" "$case_dir/$phase.benchmark.json" \
    2>"$case_dir/$phase.json-error.log"; then
    status=1
  fi

  if [[ $backend == openvino-npu && -r $npu_busy_path ]]; then
    cat "$npu_busy_path" >"$case_dir/$phase.npu-busy-after-us.txt"
    awk 'NR==FNR { before=$1; next } { print $1-before }' \
      "$case_dir/$phase.npu-busy-before-us.txt" \
      "$case_dir/$phase.npu-busy-after-us.txt" \
      >"$case_dir/$phase.npu-busy-delta-us.txt"
  fi
  return "$status"
}

for backend in "${backends[@]}"; do
  render_config "$backend"
  mkdir -p "$result_root/$backend"
  run_case "$backend" cold 0 1 || overall_status=1
  run_case "$backend" hot 2 10 || overall_status=1
done

for backend in "${backends[@]}"; do
  case_dir=$result_root/$backend
  wav=$work_root/audio/$backend-cold/iteration-0001.wav
  if [[ ! -f $wav ]]; then
    overall_status=1
    continue
  fi
  (cd "$work_root" && env LD_LIBRARY_PATH="$runtime_lib:$openvino_lib:$tbb_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    OMASPEAK_ONNXRUNTIME_LIBRARY="$ort_library" \
    OMASPEAK_OPENVINO_LIBRARY="$openvino_library" \
    OMASPEAK_OPENVINO_PLUGINS="$openvino_plugins" \
    "$voxtype" transcribe "$wav" >"$case_dir/cold.asr.raw.log" \
    2>"$case_dir/cold.asr.stderr.log")
  asr_status=$?
  printf '%s\n' "$asr_status" >"$case_dir/cold.asr.exit-status.txt"
  hypothesis=$(awk 'NF { line=$0 } END { print line }' "$case_dir/cold.asr.raw.log")
  printf '%s\n' "$hypothesis" >"$case_dir/cold.transcript.txt"
  python3 "$script_dir/compute_wer.py" "$text" "$hypothesis" \
    >"$case_dir/cold.wer.json" || asr_status=1
  [[ $asr_status == 0 ]] || overall_status=1
done

python3 "$script_dir/summarize_audio.py" "$work_root/audio" \
  >"$result_root/audio-summary.json"
if ! python3 "$script_dir/validate_run.py" "$result_root" "${backends[@]}" \
  >"$result_root/validation.txt" 2>&1; then
  overall_status=1
fi

printf 'Results: %s\n' "$result_root"
exit "$overall_status"
