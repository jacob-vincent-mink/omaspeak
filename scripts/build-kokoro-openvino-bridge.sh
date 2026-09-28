#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
  echo "Usage: $0 GENAI_INCLUDE_DIR GENAI_LIB_DIR OUTPUT_LIBRARY" >&2
  exit 2
fi

genai_include=$1
genai_lib=$2
output_library=$3
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

test -f "${genai_include}/openvino/genai/speech_generation/text2speech_pipeline.hpp"
test -f "${genai_lib}/libopenvino_genai.so"
mkdir -p -- "$(dirname -- "${output_library}")"

${CXX:-c++} -std=c++17 -O2 -fPIC -shared \
  -I "${genai_include}" \
  -I "${GENAI_GENERATED_INCLUDE_DIR:-${genai_include}}" \
  -I "${OPENVINO_INCLUDE_DIR:-/usr/include}" \
  "${source_dir}/native/kokoro_openvino_bridge.cpp" \
  -L "${genai_lib}" -lopenvino_genai -lopenvino -ldl \
  -Wl,-rpath,'$ORIGIN' \
  -o "${output_library}"

echo "${output_library}"
