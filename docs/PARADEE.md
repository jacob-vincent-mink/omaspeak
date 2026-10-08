# Paradee native OpenVINO integration

Omaspeak now provides the experimental `paradee-openvino` backend and the pinned
`paradee-8m-openvino` catalog model. It executes the official **FP32** ONNX graph
using Omaspeak's existing OpenVINO C runtime. No ONNX Runtime, Python interpreter,
PyTorch, or Python Misaki package is installed or required by this backend.
The existing default model is unchanged.

## Requirements and setup

- OpenVINO CPU runtime. This integration was exercised with OpenVINO 2026.4.
- Native `espeak-ng` executable **and its English data**, installed separately
  through the system package manager. This is an optional frontend dependency
  for Paradee, not a newly bundled runtime. eSpeak NG is GPL-3.0; its package's
  license/source obligations remain with that separately installed package.
- The catalog downloads the official 36,986,117-byte FP32 graph and 1,774-byte
  vocabulary metadata from immutable Hugging Face revision
  `8f34b01ef8adcb0bec470b89bf2fd62a7b2369e6`, verifies SHA-256, and installs the
  official Apache-2.0 model license with the ordinary provenance manifest.

Install and activate through the existing model setup commands:

```sh
omaspeak setup model --download paradee-8m-openvino
omaspeak say --no-play 'Hello world. Paradee uses the existing OpenVINO runtime.'
```

If eSpeak NG is not on `PATH`, set its absolute path:

```toml
[backend]
kind = "paradee-openvino"
runtime = "openvino"
device = "cpu"

[backend.options]
g2p_executable = "/absolute/path/to/espeak-ng"

[model]
family = "paradee"
name = "paradee-8m-openvino"
file = "paradee.onnx"
tts_json = "config.json"
language = "en-us"
voice = "af_heart"
```

The existing OpenVINO library/plugin settings still apply. CPU must be selected
explicitly; `auto`, GPU, NPU, CUDA, Vulkan and HIP are not offered for this model.
INT8 ONNX is not included because it failed compilation in the tested OpenVINO
version. A future separately qualified export can replace that restriction.

## Frontend and streaming behavior

The native frontend asks eSpeak NG for American English IPA, then maps that IPA
to Misaki spelling using the conversion documented by the model author for the
[browser frontend](https://github.com/sahilmahendrakar/paradee/blob/main/web/misaki.js).
It maps diphthongs, affricates, rhotics, length marks and syllabic consonants;
raw eSpeak IPA is not fed directly to the model. eSpeak NG provides number and
abbreviation expansion. This is the author's fallback frontend approach, not
the complete Python Misaki lexicon, and unusual names/foreign words can differ.
Unsupported phonemes cause an explicit error rather than silent omission.

Text is normalized narrowly for unambiguous US dollar amounts and common
English titles, then segmented at prosody punctuation. Dollar amounts become
whole dollars/cents; Dr./Mr./Mrs./Ms./Prof. become Doctor/Mister/Missus/Miz/Professor.
Other or ambiguous monetary formats are left for eSpeak rather than guessed.
Decimal/thousands separators, clock notation and initials are protected from
clause splitting. Commas, questions, semicolons, dashes and sentence punctuation
are restored to the model's phonemes instead of discarded with native IPA.
Each frontend call is bounded to 240 Unicode characters, and encoded phonemes
are further split into at most 510 symbols with zero pads. Long input is not
truncated. A sentence ending after a lone initial remains inherently ambiguous.

Only the distilled `af_heart` voice, American English, and speeds from 0.5 to 2.0
are exposed. Audio is mono 24 kHz. Each completed segment is delivered to the
streaming sink immediately; the graph itself returns a complete segment,
not incremental audio frames. CPU placement is checked against the compiled model’s `EXECUTION_DEVICES` property.
The warmed OpenVINO model is reused. Sink errors
stop before the next segment, and the existing supervised worker provides
playback cancellation, staged output and wake-pause ownership. Frontend stdin and
stdout use concurrent bounded nonblocking I/O with a five-second deadline, and
failed/expired children are killed and reaped. On Linux the frontend also exits
if its synthesis parent dies, including direct file synthesis.

## Qualification and remaining work

A native test on this machine produced finite, non-silent audio from short
English text, numbers/abbreviations, and a long streamed passage. Warm generation
for a 648-character repeated passage emitted three chunks: first audio at about
0.586 seconds, completion at 1.68 seconds, and about 38.9 seconds of audio.
Cold backend loading took about 0.64 seconds. These are one-run debug-build
measurements, not portable latency promises or perceptual quality scores.
A reproducible [quality corpus and runner](../benchmarks/paradee/README.md) now
cover names, numbers, titles, punctuation and long text. The
[2026-10-08 native pass](../benchmarks/results/2026-10-08-paradee-quality/RESULTS.md)
generated valid non-silent audio for all 15 cases with no clipped PCM samples.
Human listening and independent pronunciation scores remain pending, so the
integration remains experimental.

The author’s published model quality/performance figures are not measurements
of this native frontend. The [support survey](CLOUD-BACKENDS-AND-PARADEE.md)
records the graph conversion and compiler limitations. GPU requires a qualified
4D Resize conversion and FP32 precision; NPU needs bounded/static exports and
compiler investigation. These routes are intentionally unavailable here.

For the future audio.cpp family, see [the pickup note](AUDIOCPP-PARADEE-FOLLOWUP.md).
The workspace's audio.cpp checkout contains the same note under
`docs/models/paradee-integration-followup.md`.
