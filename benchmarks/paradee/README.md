# Native Paradee quality corpus

This fixed corpus covers English baseline speech, names/initials, accented names,
USD amounts, decimals/percentages, clock/date notation, abbreviations/acronyms,
commas/questions/dashes/ellipsis/quotes, a long passage and an unbroken long word.
The Rust test `native_quality_corpus_has_exact_prosody_frontend_plans` checks
normalization and clause punctuation against the corpus's golden plans.

Run from an Omaspeak checkout with its normal native prerequisites:

```sh
cargo build --locked
python3 benchmarks/paradee/qualify.py \
  --binary target/debug/omaspeak \
  --config /absolute/path/to/paradee-cpu.toml \
  --espeak /absolute/path/to/espeak-ng \
  --out /tmp/paradee-quality-run
python3 -m unittest discover -s benchmarks/paradee -p 'test_*.py'
```

The runner uses only Python 3.10+ standard library modules and reads configuration
through the native CLI's `config get --json`, including compiled defaults.

Use the CPU configuration in `docs/PARADEE.md`, with `model.directory` set to
an absolute installed model directory. The runner checks official model and
vocabulary SHA-256 and that `--espeak` matches the configured frontend. It records
binary/corpus/model hashes, frontend version, CPU/thread configuration, waveform
hashes, mono 24 kHz PCM16 shape, duration, peak/RMS and clipping counts. It also
records native eSpeak IPA for the corpus's golden normalized clauses, before
the production Misaki spelling map. Those are pronunciation review artifacts,
not a second phonemizer implementation. Long-text plans are checked by the Rust
corpus test for bounded segmentation; their complete waveform is retained.

The runner sends full original text through actual `omaspeak say --no-play`.
Temporary XDG directories prevent contact with the normal daemon/configuration.
It opens no playback device and writes review WAVs plus `report.json` into a new
output directory. Every invocation includes cold model load; elapsed time is not
streaming time to first audio. Do not compare individual waveform hashes as
quality regressions: the model includes random waveform generation.

Python is benchmark tooling only. Omaspeak still delivers no Python inference
runtime or ONNX Runtime. The app uses native eSpeak NG and existing OpenVINO.

## Listening protocol

Play the generated WAVs deliberately and record a separate review sheet with
case ID, listener, model/frontend versions and findings. Check word order,
missing/repeated words, natural number expansion, title pronunciation and pauses.
For names, record the intended pronunciation before judging the sample; names
such as Nguyen have valid regional alternatives. Inspect long text for lost
clauses, boundary artifacts and unnatural silence. The unbroken long word is a
robustness stress case, not natural English quality evidence.

Review expectations include `Dr.` → Doctor, `Prof.` → Professor, `Ms.` → miz,
`Mrs.` → missus, USD dollars/cents, decimals kept together, and initial/acronym
periods kept inside their words. Unsupported/ambiguous monetary formats are left
for eSpeak rather than silently assigned a guessed value. Clause punctuation
reaches the graph instead of being discarded with IPA output.

Do not label a run perceptually qualified from signal metrics alone. Record
listening findings, and optionally independent ASR/reference-transcript errors,
before making intelligibility or pronunciation score claims. This corpus does
not add an ASR runtime to the shipped application.
