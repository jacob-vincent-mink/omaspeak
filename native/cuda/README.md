# Native CUDA runtime

Omaspeak's `cuda` feature needs a shared sherpa-onnx build whose ONNX Runtime
contains the CUDA Execution Provider. The ordinary sherpa-onnx release is
CPU-only. [`build.sh`](build.sh) creates that stack from pinned sources:

- ONNX Runtime `v1.29.0` at `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`
- sherpa-onnx `v1.13.8` at `11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf`

The builder applies Omaspeak's checksum-pinned sherpa patch. The patch sends
offline CUDA provider-file settings to ONNX Runtime's CUDA EP V2 interface and
allows Supertonic's four sessions to select the requested provider.

The builder supports Linux x86_64 and aarch64. Install CMake 3.28 or newer, a
CUDA toolkit, and its matching cuDNN development package first. `CUDA_HOME`
defaults to `/usr/local/cuda`; `CUDNN_HOME` defaults to `/usr`.

```bash
OMA_BUILD_JOBS=10 native/cuda/build.sh
source /tmp/oma-native-cuda/runtime/env.sh
cargo build --release --features cuda
```

`OMA_CUDA_ARCHITECTURES` defaults to CMake's `native` detection. Set an
explicit semicolon-separated architecture list for a release build or when the
toolkit no longer supports ONNX Runtime's historical defaults. NVIDIA GB10,
for example, uses `OMA_CUDA_ARCHITECTURES=121` with CUDA 13.

The script downloads sources and writes builds below `${OMA_NATIVE_ROOT}`
(default `/tmp/oma-native-cuda`). It does not install files into the system.
It refuses mismatched or modified source checkouts so reruns remain tied to the
reviewed commits.

Set `backend.runtime = "cuda"` and `backend.device = "gpu"` after loading the
runtime environment. `backend.device_id` selects the CUDA device ordinal. A
successful build proves that CUDA support is linked; use
`omaspeak say "CUDA synthesis test" --no-play --out /tmp/cuda.wav` to verify
synthesis without making sound.

`--features all-runtimes` is also supported when `SHERPA_ONNX_LIB_DIR` points
to one shared ONNX Runtime build containing both provider libraries. This is
the native contract used by the all-features release build.
