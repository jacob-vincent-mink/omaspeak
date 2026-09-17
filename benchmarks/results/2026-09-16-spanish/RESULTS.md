# Spanish speech profile (Supertonic), 2026-09-16

S10 evidence: the first qualified non-English speech profile for Omaspeak.
Supertonic's pinned catalogs (GGUF and OpenVINO) ship 31 languages including
`es`; this slice proves the configured language reaches inference and status
and records Spanish synthesis evidence. No new model or download is required.

## Language reaching inference and status

- `model.language` already selected among the 31 Supertonic languages; the
  value is a protected request option (`backend.options.request.language`
  cannot override it) that the audio.cpp provider passes to the engine.
- New: the daemon `status --json` and stopped-status payloads now carry the
  configured `language` next to the model name.
- E2E: a stub-worker test (`configured_language_reaches_the_native_worker`)
  proves the configured language is delivered to the native request builder
  — a request configured for `en` is rejected by the native fixture that
  only accepts `es`, while the `es` configuration synthesizes.

## Evidence

The maintainer's installed `supertonic-3-openvino` (Intel NPU path, OpenVINO
2026.3.1) ran with `model.language = "es"` in an isolated XDG tree:

| Sample | Text | Audio | Synthesis |
|---|---|---|---|
| es1 | "La cena está servida en la mesa del comedor." | 3.59 s | ~2.7–2.8 s |
| es2 | "El clima hoy es muy agradable para caminar." | 3.58 s | ~2.7 s |
| es3 | "El informe se retrasó hasta el próximo martes." | 3.83 s | ~2.8 s |
| en1 (en control) | "Dinner is served on the dining room table." | 3.15 s | ~2.7 s |

All WAVs are valid, non-silent 44.1 kHz outputs
([samples/](samples/)). Timing is coarse and contended (a shared host was
running the W08 benchmark arms concurrently); it is not a performance gate.

Kokoro's Spanish path was proven earlier by the S09 qualification
(`ef_dora` produced the Spanish wake-phrase clips used by the Omawake W09
corpus — see `docs/KOKORO.md`).

## Limits

- Evidence covers synthesis with the configured language on one device
  (NPU) and one installed model; per-language pronunciation quality is a
  listening gate that remains open, as do real-room recordings and the
  Spanish device matrix. No blanket multilingual claim is made beyond the
  language list Supertonic itself documents.
