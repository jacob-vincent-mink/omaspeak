# Kokoro 82M (S09)

Kokoro 82M is the second TTS family in Omaspeak, shipped as a pinned
audio.cpp GGUF alongside Supertonic. This is the adapter/qualification slice of
S09: minimal named-voice/adapter support plus a fresh-installable, pinned
Kokoro model. It does **not** replace Supertonic as the default and introduces
no clone/design or arbitrary conditioning fields.

## What is pinned

| Item | Value |
|---|---|
| Catalog ID | `kokoro-82m-gguf` |
| Family / backend | `kokoro` / `audiocpp` |
| Model file | `kokoro-82m-q8_0.gguf` (single self-contained GGUF) |
| Source | `audio-cpp/audio.cpp-gguf` @ `1b13cd58245c74e3ff4ca06925766c5ef7991bd4` |
| File size / sha256 | `189_611_360` / `378abf37…c9889e6` |
| Original weights | `hexgrad/Kokoro-82M` |
| License | Apache-2.0 (embedded in `licenses/KOKORO-82M-MODEL-LICENSE`, sha256-pinned) |
| Output rate | 24 kHz mono |
| Voices | 54 preset voice packs, embedded in the GGUF (`voices.json`) |

The GGUF is self-contained: weights, config, vocabulary, all 54 voice packs,
and the generated `g2p/ja.json` / `g2p/zh.json` tables are embedded. English,
Spanish, French, Hindi, Italian and Portuguese phonemize through the shared
eSpeak-ng runtime (system `libespeak-ng.so.1` or the provider's static build).
Chinese uses the bundled `g2p/zh.json`; Japanese needs MeCab/UniDic, which the
small release GGUF does not bundle.

## Voice selection

`model.voice` and the IPC `say.voice` field accept either a preset name
(`af_heart`, `bf_emma`, …) or a legacy integer index into the 54-voice catalog.
Names are matched case-insensitively. The request language is derived from the
voice's first letter (voice packs are language-scoped): `a`→`en-us`,
`b`→`en-gb`, `e`→`es`, `f`→`fr`, `h`→`hi`, `i`→`it`, `j`→`ja`,
`p`→`pt-br`, `z`→`zh`. Unknown names, out-of-range legacy IDs, and voice IDs
with an unknown language prefix are rejected without touching the existing
config.

```sh
omaspeak config set model.family kokoro
omaspeak config set model.name kokoro-82m-gguf
omaspeak setup model
omaspeak say --voice af_heart "Hello from Kokoro."
omaspeak say --voice 16 "Am I Michael?"   # 16 = am_michael
omaspeak voices --json
```

`model.steps` is unused by Kokoro (Supertonic denoising steps); the schema
reports `min: 0` for the `kokoro` family. `config set model.steps` still works
for Supertonic and is a no-op semantically for Kokoro.

## Fresh install

```sh
omaspeak setup model   # downloads the pinned GGUF, verifies size + sha256,
                       # writes the license + install manifest, atomically
                       # activates the model
```

The download reuses the existing pinned-file machinery (size bound, checksum
verification, license install, atomic activation, rollback on failure).
`requires_acceptance` is `false` (Apache-2.0 needs no acceptance).

## Qualification status

- **Functional:** `say` produces a valid, non-silent 24 kHz mono WAV for
  `af_heart` (and named/legacy selection), verified end-to-end through the
  process-isolated worker. E2E coverage lives in
  `tests/cli.rs::kokoro_voice_selects_engine_id_language_and_24khz_output`.
- **Voice/language routing:** the native ABI receives the correct engine voice
  ID and derived language (stub-fixture test: only `af_heart` accepted;
  `am_michael` resolves to its ID and is rejected by the single-voice stub).
- **Catalog/schema:** 54-voice inventory, family choices, `steps` min, and
  license text are asserted in unit tests.
- **Open (not claimed by this PR):** pronunciation/latency/quality comparison
  vs. Supertonic on a fixed corpus, a listening pack, and fresh-install proof
  on a clean machine. Those are the remaining S09 evidence and are tracked in
  `docs/COMPLETION.md`. No default promotion and no hardware claims.

## Non-goals (declined scope)

- No voice cloning / design / reference-audio conditioning.
- No Japanese UniDic bundling (documented limitation, not a supported path).
- No change to the Supertonic default or its numeric voice identity.
- No remote catalog or dynamic model refresh.
