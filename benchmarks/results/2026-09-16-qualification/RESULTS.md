# Paired speech and NPU qualification, 2026-09-16

This extends S01/S05 with matched optimized builds, broader Intel NPU file
synthesis, a reusable blinded listening pack and resource measurements. Human
listening and power qualification remain open. No playback or daemon request
ran; live configuration and caches were not modified.

## Matched CPU comparison

Reference `86064f6` and candidate `d6c84cd` used the same Supertonic 3 model,
OpenVINO provider, six English cases, M1/F1 voices, two threads and fallback
`error`. Each case/voice loaded a fresh engine, warmed up once, then performed
five measured syntheses. Three rounds alternated reference/candidate order by
round: 72 engine loads and 360 measured WAVs. Every WAV payload was valid and
OpenVINO verified placement on CPU. [Raw measurements](paired-cpu.json) retain
load time, warm summaries, maximum process RSS and audio hashes.

Both variants shared an isolated cache which persisted through the run; its
first use was not precompiled. Cold compilation and warm-cache loads are not
interchangeable. CPU affinity was 0–1, with other qualification lanes assigned
separate CPUs. This was still a shared host with uncontrolled thermal/memory and
background workload effects. Five samples per p95 is a coarse tail estimate.

[Candidate/reference ratios](paired-ratios.json) show no case/voice exceeding the
predeclared 10% latency or memory threshold in all three rounds. Individual p95
ratios ranged from 0.47 to 3.03; first-round memory increases on some pairs did
not repeat in rounds two and three. Maximum process RSS across all cases was
2,634 MiB reference and 2,636 MiB candidate. These measurements do not establish
an uncontended regression-gate pass. Keep raw outliers; do not average them away
or infer a speedup from this run.

## NPU corpus and resource evidence

All twelve English case/voice combinations ran on Intel NPU with one warmup and
three measured syntheses each. All 36 WAV payloads were valid, placement was
verified for every graph, and fallback remained disabled. [NPU measurements](npu.json)
record input/binary/configuration hashes and per-recording hashes. A copied,
previously prepared static NPU cache was used; this is not cold compilation.

| Case | NPU warm p95 range (M1/F1) |
|---|---:|
| notification | 95–97 ms |
| reply | 123–133 ms |
| numbers | 155–183 ms |
| names-acronyms | 121–127 ms |
| punctuation | 122–126 ms |
| long-reply | 491–507 ms |

Each p95 uses only three samples and is descriptive. Maximum observed process
RSS across this corpus was 1,290 MiB. A separate
long-reply/M1 run reached 1,342 MiB sampled process-group RSS
([measurement](npu-group-memory.json)). This sums resident pages across the
process group every 200 ms; shared pages can be counted twice and device memory
is excluded. It is not unique physical memory or a minimum RAM recommendation.
The selected prepared NPU cache occupies 780,748,781 bytes by `du -sb`, beyond
the 401,305,212-byte installed model tree on this host. These are observations
for these exact artifacts, not general compiled-cache size estimates.

## Independent NPU intelligibility spot check

Whisper.cpp 1.9.3 Base.en independently transcribed the first measured recording
for every NPU case/voice pair, after FFmpeg mono/16 kHz resampling.
[Expected text and raw transcripts](npu-transcripts.json) recover both full long
replies. Variations include `Your`/`You’re`, `build`/`built`, F1 `a.m.`/`A.N.`,
and Omaspeak proper-name spellings. They could arise from synthesis, recognition
or both; they are listening-review targets, not automated proof of naturalness.

## Blinded listening review

`scripts/build-listening-pack.py` copies the first measured recording from each
reference/candidate case/voice pair, verifies the stored corpus/audio hashes,
and creates 24 WAVs, an offline HTML player, a CSV rating sheet and a zip. A/B
assignment is seeded and independently randomized by pair. The answer key stays
outside the pack. Playback is manual; nothing autoplays or contacts the daemon.

Rate intelligibility, naturalness, pronunciation and voice consistency from 1
(poor) to 5 (excellent); record missing words, clipping and questionable sounds.
Leave uncertain ratings blank. Review with the same headphones and volume before
unblinding. The pack is a small English review aid, not a population study.
**No human ratings or voice-fidelity claim have been made.** Audio remains in the
local qualification artifact directory, rather than adding large WAVs to Git.

```sh
# Python 3.11+, reviewed config with fallback=error and pinned provider/model
python3 scripts/measure-model-corpus.py --binary /path/candidate \
  --reference /path/reference --config /path/cpu.toml \
  --out /new/paired --runs 3 --iterations 5
python3 scripts/build-listening-pack.py /new/paired --out /new/listening-review

# For NPU, copy a compatible prepared cache into the isolated run first.
python3 scripts/measure-model-corpus.py --binary /path/candidate \
  --config /path/npu.toml --cache-seed /path/copied-cache \
  --out /new/npu --runs 1 --iterations 3
```

The measurement script rejects fallback, unverified placement, failed synthesis
and truncated WAVs. `scripts/measure-command.py` can wrap a file-only command to
record process CPU time, sampled group RSS, exit status and accessible package
energy counters. Package energy includes all host workloads and must never be
presented as application-attributed power.

## Open gates and execution limits

Human listening, additional voices/languages, controlled repeat performance,
real-time idle/peak power, broader CUDA/HIP device coverage and production memory
qualification remain open. The existing CPU/GPU corpus and earlier device smokes
remain useful evidence, without upgrading hardware recommendations.

RAPL counters were unreadable; noninteractive sudo required a password. The host
was on AC with a full battery. No watts or battery-discharge estimate is claimed.
Configured CUDA test hosts were unavailable (SANDERS DNS failure; BENSON SSH
timeout), and no local NVIDIA/AMD device was available. [Environment](environment.json)
records source commits, binaries, provider/model-manifest hashes and limits.

The first NPU-cache copy failed at the `/tmp` user quota. A subsequent cache-status
process received SIGBUS with a Linux-loader stack; its exact cause is unresolved.
The identical binary and a complete copied cache succeeded on the workspace
filesystem. Only the incomplete test copy was removed; no live data was changed.
