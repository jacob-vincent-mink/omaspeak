#!/usr/bin/env bash
set -u

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)
run_id=${RUN_ID:-$(date -u +%Y-%m-%dT%H%M%SZ)}
binary=${BINARY:-$repo_dir/target-openvino/release/omaspeak}
runtime_lib=${RUNTIME_LIB:-/tmp/oma-native/runtime/lib}
openvino_lib=${OPENVINO_LIB:-/tmp/oma-native/openvino-2026.2.1/runtime/lib/intel64}
tbb_lib=${TBB_LIB:-/tmp/oma-native/openvino-2026.2.1/runtime/3rdparty/tbb/lib}
model_dir=${MODEL_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/omaspeak/models/supertonic-3-int8}
voxtype=${VOXTYPE:-$HOME/.local/bin/voxtype}
result_root=${RESULT_ROOT:-$script_dir/results/$run_id-npu-accuracy}
work_root=${WORK_ROOT:-/tmp/omaspeak-openvino-supertonic/$run_id-npu-accuracy}
npu_busy_path=${NPU_BUSY_PATH:-/sys/devices/pci0000:00/0000:00:0b.0/npu_busy_time_us}
text='Omaspeak now supports Supertonic.'
reference='omaspeak now supports supertonic'

mkdir -p "$result_root" "$work_root"

run_cpu_control() {
  local lane=$1
  local case_dir=$result_root/$lane
  local state_dir=$work_root/$lane/state
  local audio_dir=$work_root/$lane/audio
  local config=$case_dir/config.toml
  mkdir -p "$case_dir" "$state_dir" "$audio_dir"
  sed -e "s|@MODEL_DIR@|$model_dir|g" -e '/^ProfilingFilePrefix = /d' \
    "$script_dir/configs/$lane.toml.in" >"$config"
  (cd "$work_root" && env LD_LIBRARY_PATH="$runtime_lib:$openvino_lib:$tbb_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    XDG_STATE_HOME="$state_dir" \
    "$binary" --config "$config" benchmark --text "$text" \
    --out-dir "$audio_dir" --warmup 0 --iterations 1 \
    >"$case_dir/benchmark.raw.log" 2>"$case_dir/stderr.log")
  local status=$?
  printf '%s\n' "$status" >"$case_dir/exit-status.txt"
  "$script_dir/extract_benchmark_json.py" "$case_dir/benchmark.raw.log" \
    "$case_dir/benchmark.json" 2>"$case_dir/json-error.log" || status=1
  local wav=$audio_dir/iteration-0001.wav
  if [[ $status != 0 || ! -f $wav ]]; then
    return 1
  fi
  "$voxtype" transcribe "$wav" >"$case_dir/asr.raw.log" \
    2>"$case_dir/asr.stderr.log"
  local asr_status=$?
  printf '%s\n' "$asr_status" >"$case_dir/asr.exit-status.txt"
  local hypothesis
  hypothesis=$(awk 'NF { line=$0 } END { print line }' "$case_dir/asr.raw.log")
  printf '%s\n' "$hypothesis" >"$case_dir/transcript.txt"
  "$script_dir/compute_wer.py" "$reference" "$hypothesis" \
    >"$case_dir/wer.json"
  [[ $asr_status == 0 ]] && grep -q '"wer": 0.0' "$case_dir/wer.json"
}

control_status=0
run_cpu_control default-cpu || control_status=1
run_cpu_control openvino-cpu || control_status=1

declare -A load_config
declare -A qdq
load_config[baseline]='{"NPU":{"NPU_PLATFORM":"5010"}}'
load_config[qdq-off]=${load_config[baseline]}
load_config[accuracy-hint]='{"NPU":{"NPU_PLATFORM":"5010","EXECUTION_MODE_HINT":"ACCURACY"}}'
load_config[npu-qdq-off]='{"NPU":{"NPU_PLATFORM":"5010","EXECUTION_MODE_HINT":"ACCURACY","NPU_QDQ_OPTIMIZATION":"NO"}}'
load_config[dynamic-quant-off]='{"NPU":{"NPU_PLATFORM":"5010","EXECUTION_MODE_HINT":"ACCURACY","NPU_QDQ_OPTIMIZATION":"NO","NPU_COMPILER_DYNAMIC_QUANTIZATION":"NO"}}'
qdq[baseline]=True
qdq[qdq-off]=False
qdq[accuracy-hint]=False
qdq[npu-qdq-off]=False
qdq[dynamic-quant-off]=False

npu_acceptable=1
for variant in baseline qdq-off accuracy-hint npu-qdq-off dynamic-quant-off; do
  case_dir=$result_root/$variant
  state_dir=$work_root/$variant/state
  audio_dir=$work_root/$variant/audio
  config=$case_dir/config.toml
  mkdir -p "$case_dir" "$state_dir" "$audio_dir"
  sed -e "s|@MODEL_DIR@|$model_dir|g" \
    -e '/^ProfilingFilePrefix = /d' \
    -e "s|^load_config = .*|load_config = '${load_config[$variant]}'|" \
    "$script_dir/configs/openvino-npu.toml.in" >"$config"
  sed -i "/^\[backend.options\]/a enable_qdq_optimizer = \"${qdq[$variant]}\"" "$config"
  sed -i "/^\[backend.options\]/a SherpaOnnx.SupertonicComponents = \"all\"" "$config"

  cat "$npu_busy_path" >"$case_dir/npu-busy-before-us.txt"
  (cd "$work_root" && env LD_LIBRARY_PATH="$runtime_lib:$openvino_lib:$tbb_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    XDG_STATE_HOME="$state_dir" \
    "$binary" --config "$config" benchmark --text "$text" \
    --out-dir "$audio_dir" --warmup 0 --iterations 1 \
    >"$case_dir/benchmark.raw.log" 2>"$case_dir/stderr.log")
  status=$?
  printf '%s\n' "$status" >"$case_dir/exit-status.txt"
  cat "$npu_busy_path" >"$case_dir/npu-busy-after-us.txt"
  awk 'NR==FNR { before=$1; next } { print $1-before }' \
    "$case_dir/npu-busy-before-us.txt" "$case_dir/npu-busy-after-us.txt" \
    >"$case_dir/npu-busy-delta-us.txt"
  "$script_dir/extract_benchmark_json.py" "$case_dir/benchmark.raw.log" \
    "$case_dir/benchmark.json" 2>"$case_dir/json-error.log" || true
  provider=$(find "$state_dir" -name provider.config -type f -print -quit)
  [[ -z $provider ]] || cp "$provider" "$case_dir/provider.config"

  wav=$audio_dir/iteration-0001.wav
  if [[ $status == 0 && -f $wav ]]; then
    "$voxtype" transcribe "$wav" >"$case_dir/asr.raw.log" \
      2>"$case_dir/asr.stderr.log"
    asr_status=$?
    printf '%s\n' "$asr_status" >"$case_dir/asr.exit-status.txt"
    hypothesis=$(awk 'NF { line=$0 } END { print line }' "$case_dir/asr.raw.log")
    printf '%s\n' "$hypothesis" >"$case_dir/transcript.txt"
    "$script_dir/compute_wer.py" "$reference" "$hypothesis" \
      >"$case_dir/wer.json"
    if [[ $asr_status == 0 ]] && grep -q '"wer": 0.0' "$case_dir/wer.json"; then
      npu_acceptable=0
    fi
  fi
done

"$script_dir/summarize_audio.py" "$work_root" >"$result_root/audio-summary.json"
overall_status=0
[[ $control_status == 0 && $npu_acceptable == 0 ]] || overall_status=1
printf '%s\n' "$overall_status" >"$result_root/acceptable-variant-exit-status.txt"
exit "$overall_status"
