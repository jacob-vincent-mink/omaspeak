# Kokoro 82M adapter qualification, 2026-09-16

Extends S09 with a functional, pinned Kokoro 82M GGUF path through the
audio.cpp provider (`kokoro_tts` family) and records a first latency
comparison against the incumbent Supertonic path. Human listening, real-room
recording and power gates remain open for Kokoro; none are claimed here.

## What was exercised

The provider is the pinned audio.cpp runtime built with
`-DAUDIOCPP_MODELS="supertonic;kokoro_tts"` and `AUDIOCPP_STATIC_ESPEAK=ON`
(eSpeak-ng 1.52.0 static adapter, all 114 dictionaries packed into
`espeak-ng-data.bin`). Both families probe successfully against the same
library; requesting one through the other's provider is rejected.

A fresh, isolated XDG tree ran the full S02–S04 model flow for Kokoro:

- `setup model` listed `kokoro-82m-gguf` as available (Apache-2.0, no
  acceptance required) and the pinned flow downloaded, size/sha256-verified
  (189,611,360 bytes, `378abf37…c9889e6`), licensed, and installed the GGUF.
- Activation was withheld when the first provider proof failed (missing
  eSpeak data package next to the executable), and succeeded unchanged once
  `espeak-ng-data.bin` was present — the atomic-activation/rollback path
  behaved as designed and no partial state was saved.
- `say` then synthesized through the process-isolated worker with no CLI
  fallback: valid 24 kHz mono WAVs, non-silent (max amplitude 11,985).

Voice routing was verified for named selection (`af_heart`, `ef_dora`) and
the legacy numeric identity (16 → `am_michael`), with the request language
derived per voice prefix; unknown names and out-of-range IDs are rejected
without config writes.

## Paired latency comparison (same host, same text)

Single sentence ("The quick brown fox jumps over the lazy dog. Omaspeak is a
local text to speech daemon."), three runs each, fresh process per run,
seed 42 for Kokoro. This is a coarse functional comparison, not a
regression gate.

| Path | Backend | Load (ms) | Synthesis (ms) | Audio | Rate |
|---|---|---|---|---|---|
| Supertonic 3 | OpenVINO → NPU | 171–230 | 933–1023 | 6.44 s | 44.1 kHz |
| Kokoro 82M q8_0 | audio.cpp CPU | 1401–1456 | 4647–4716 | 6.20 s | 24 kHz |

Supertonic keeps its incumbent performance profile on the NPU path. Kokoro
on CPU is comfortably faster than real time (~0.76 RTF) but loads slower and
synthesizes roughly 4.6× slower than the NPU Supertonic path on this host.
Kokoro remains optional and unpromoted: no default change, no claim of
parity, and no hardware-specific qualification (OpenVINO, CUDA, NPU) is
made for it.

Raw WAVs: `samples/kokoro-af_heart-en.wav` (24 kHz),
`samples/supertonic-F1-en.wav` (44.1 kHz). A blinded listening pack and
human ratings are still owed before any default-family conversation.

## Scope notes

- The Japanese language path needs MeCab/UniDic, which the small release
  GGUF does not bundle; it is documented as unsupported, not worked around.
- The eSpeak data package must sit next to the process executable (or be
  given explicitly via `backend.options.load.espeak_data_path`); missing or
  invalid data fails the provider proof instead of silently substituting.
- No remote catalog, no dynamic refresh, no voice design/conditioning.
