#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
  echo "Usage: $0 ARCH OUTPUT_LIBRARY WORK_DIR" >&2
  exit 2
fi

arch=$1
output_library=$2
work_dir=$3
case "${arch}" in
  x86_64)
    sdk_arch=x86_64
    expected_sha256=aa121de8791d973d79bdba5db0fee6ec511dd85c5924c82346fd7306d04f8d62
    ;;
  aarch64)
    sdk_arch=arm64
    expected_sha256=aaf9a8c45956332e26da4990153c79f3b7a0d2458ecc369710e42c3f99085b90
    ;;
  *)
    echo "Unsupported release architecture: ${arch}" >&2
    exit 2
    ;;
esac

archive_name="openvino_genai_ubuntu22_2026.4.0.0_${sdk_arch}.tar.gz"
url="https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.4/linux/${archive_name}"
mkdir -p -- "${work_dir}"
archive=${OMASPEAK_GENAI_SDK_ARCHIVE:-${work_dir}/${archive_name}}
if [[ ! -f "${archive}" ]]; then
  curl -fL --retry 3 --output "${archive}" "${url}"
fi
printf '%s  %s\n' "${expected_sha256}" "${archive}" | sha256sum -c -
tar -xzf "${archive}" -C "${work_dir}"
sdk="${work_dir}/openvino_genai_ubuntu22_2026.4.0.0_${sdk_arch}/runtime"
include="${sdk}/include"
lib="${sdk}/lib/intel64"
if [[ "${arch}" == aarch64 ]]; then
  lib="${sdk}/lib/aarch64"
fi
test -f "${include}/openvino/genai/speech_generation/text2speech_pipeline.hpp"
test -f "${include}/openvino/genai/version.hpp"
test -f "${lib}/libopenvino_genai.so"

"$(dirname -- "${BASH_SOURCE[0]}")/build-kokoro-openvino-bridge.sh" \
  "${include}" "${lib}" "${output_library}"
strip --strip-unneeded "${output_library}"
readelf -d "${output_library}" | grep -q 'Shared library: \[libopenvino_genai.so.2640\]'
readelf -d "${output_library}" | grep -q 'Shared library: \[libopenvino.so.2640\]'
if LD_LIBRARY_PATH="${lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ldd -r "${output_library}" | grep -E 'not found|undefined symbol'; then
  echo 'Kokoro bridge has unresolved SDK dependencies' >&2
  exit 1
fi
if readelf -d "${output_library}" | grep -E 'RUNPATH.*(/tmp/|/home/|/opt/)'; then
  echo 'Kokoro bridge contains a build-machine library path' >&2
  exit 1
fi
