# Historical CUDA validation evidence

This directory preserves the first 2026-09-14 GB10 validation run. Its
`identities.txt` mentions a sherpa-onnx library and an obsolete Omawake patch
because that run used an independent Sherpa/Moonshine ASR tool to transcribe
the generated WAV files. Those artifacts were verifier inputs; they were not
loaded by Omaspeak's synthesis runtime.

The Omaspeak executable in this run used the direct Rust Supertonic frontend
with ONNX Runtime and its CUDA execution provider. Current post-stabilization
setup and synthesis evidence is recorded in
`../cuda-gb10-current-results-2026-09-14/`.
