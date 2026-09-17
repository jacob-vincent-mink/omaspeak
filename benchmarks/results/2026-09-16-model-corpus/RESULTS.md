# Supertonic corpus measurements, 2026-09-16

The versioned six-case English corpus was exercised with M1 and F1 through the
existing direct OpenVINO adapter. CPU used three independent rounds; GPU used
one. Each case/voice round loaded one engine, ran one warmup, then three measured
file-only requests. All 144 measured WAVs had the expected sample rate, frame
count and nonempty payload, with fallback disabled. OpenVINO confirmed the
requested execution device for every graph. No playback or daemon request ran.

These are development-build measurements on the shared Intel development host,
with two configured threads. Cache/state/runtime directories were isolated;
compiled caches persisted within each device's run. Other development checks
ran concurrently, so timing is descriptive and cannot establish a 10% regression
gate or an uncontended hardware comparison. Binary, configuration, corpus,
provider, model-manifest and individual audio hashes are recorded in
[measurements.json](measurements.json). Peak RSS uses Linux wait4 per benchmark
process; it is not aggregate system memory or GPU/NPU memory.

The following ranges span both voices and the measured rounds. Each individual
p95 has only three measured samples and is therefore a coarse maximum-like
statistic, not a stable production-tail estimate.

| Case | CPU warm p95 range | GPU warm p95 range |
|---|---:|---:|
| notification | 376–745 ms | 123–126 ms |
| reply | 592–723 ms | 189–192 ms |
| numbers | 974–1261 ms | 322–328 ms |
| names-acronyms | 577–662 ms | 198–203 ms |
| punctuation | 662–1048 ms | 217–218 ms |
| long-reply | 3836–4595 ms | 1587–1589 ms |

Maximum observed process RSS was 2,606 MiB for CPU and 1,658 MiB for GPU; these
include model/runtime loading and synthesis. The long reply is substantially
more demanding than the previous single-sentence smoke check. Keep these values
as baseline observations rather than advertised minimum memory requirements.

## Independent intelligibility spot check

Whisper.cpp 1.9.3 Base.en on CPU independently transcribed the first measured
output for all 12 CPU case/voice combinations (16 kHz resampling via FFmpeg).
[Expected text and raw transcripts](transcripts.json) retain the evidence.
Notifications, replies and the complete long replies were recovered. Number
formatting and punctuation varied. The proper name Omaspeak became “ALMA speak”
for M1 and “Omah speak” for F1; this could reflect synthesis pronunciation,
recognizer vocabulary or both. No voice-fidelity or listening score is claimed.

## Reproduce and remaining gates

```sh
python3 scripts/measure-model-corpus.py --binary /absolute/path/to/omaspeak \
  --config /absolute/path/to/cpu.toml --out /new/evidence/directory \
  --runs 3 --iterations 3
```

Use a reviewed configuration with fallback disabled and the pinned model.
The script creates a new output directory, isolates runtime/state/cache paths,
records each report and WAV hash, and rejects failed/fallback/truncated output.
It does not install models or modify the provided configuration. WAVs remain in
the local evidence directory for listening; large audio files are not committed.

Independent listening, more speakers/languages, a matched release-build
reference/candidate comparison, idle/peak power, aggregate memory, and additional
CUDA/HIP/NPU corpus runs remain uncompleted qualification gates. Existing
single-sentence device evidence is retained, not upgraded by this report.
