# Supertonic CUDA corpus on GB10, 2026-09-16

Current release-build code `b787e8d` completed the six-case English corpus with
M1/F1 on `jacob@promaxgb10-d666`: one warmup and three measured requests per
case/voice, **36 valid WAVs**. Every request used the native audio.cpp CUDA
adapter, with verified requested placement and fallback `error`.

The host is ARM64 Ubuntu 24.04.5 with NVIDIA GB10, driver 580.173.02 and CUDA
13.0.88. The provider uses audio.cpp `e9ff20042ec85af960a720368c6927cda19ad65f`,
Supertonic-only family selection, CUDA graphs and `121a-real` architecture.
The catalog Supertonic 3 GGUF hash matches its pinned canonical artifact.
[Environment and exact hashes](environment.json), [configuration](speech-cuda.toml)
and [measurements/audio hashes](speech-measurements.json) preserve the evidence.

The application was built natively from the public committed source with
`cargo build --release --locked -j 4`. Existing user-space ALSA linker metadata
was used because the host lacks the development pkg-config file. Existing
pinned model/provider assets were read, and each run used isolated XDG paths.
No microphone/playback, live configuration mutation, daemon request or service
stop occurred. SGLang remained resident on the same GPU throughout.

| Case | Warm synthesis p95 range (M1/F1) |
|---|---:|
| notification | 55–56 ms |
| reply | 61–63 ms |
| numbers | 73–76 ms |
| names/acronyms | 64–65 ms |
| punctuation | 59–68 ms |
| long reply | 218–222 ms |

These are one-round, three-sample p95 summaries on a shared host. The native
GGUF profile uses eight steps, unlike the five-step direct OpenVINO profile in
previous measurements. This is not a matched CPU speedup comparison, a stable
tail-latency estimate or a blanket recommendation for NVIDIA devices.

Maximum benchmark process RSS was 646.8 MiB. One-second `nvidia-smi` samples
observed Omaspeak CUDA processes using 223–749 MiB of reported GPU memory.
GB10 uses unified memory: do not add process RSS and GPU memory as independent
physical allocations. The mixed wake/speech qualification window recorded
10.30–29.05 W whole-GPU power across 78 samples. Those readings include other
resident workloads and are neither application-attributed power nor idle watts.
[GPU telemetry](gpu.csv) and [process observations](processes.csv) are retained.

## Intelligibility and remaining gates

Whisper.cpp 1.9.3 Base.en independently transcribed the first measured WAV for
all twelve pairs after mono/16 kHz resampling. Ordinary text and both full long
replies were recovered; Omaspeak became `Omispeak` for both voices. Number
formatting and punctuation varied. [Expected text/raw transcripts](transcripts.json)
retain these listening-review targets. No human quality rating is claimed.

Repeat with `scripts/measure-model-corpus.py`, the recorded CUDA configuration,
`--runs 1 --iterations 3`, and the pinned provider/model. Large generated WAVs
remain in the local/remote qualification artifacts rather than Git. This
qualifies a broader file-synthesis check on this GB10/provider combination;
more voices/languages, listening, controlled repeats and idle-power attribution
remain open. **HIP remains pending: the maintainer has no HIP hardware.**
