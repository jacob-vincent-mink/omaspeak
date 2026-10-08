# Native Paradee frontend and signal quality pass

2026-10-08 on this workstation, development Omaspeak build based on 0.1.2-rc.1 with this candidate's frontend changes, OpenVINO
2026.4 CPU, two CPU threads, native eSpeak NG 1.52.0. Official pinned FP32 graph
and vocabulary were used. No ORT/Python inference runtime was loaded or added.

All 15 corpus cases produced non-silent mono 24 kHz PCM16 WAVs, totaling 160.425
seconds of audio. No clipped PCM samples were detected. Names, numbers/titles,
punctuation and long input all completed successfully. The long passage produced
60.275 seconds of audio in about 3.76 seconds including cold load; the unbroken
long-word stress case produced 30.45 seconds in about 2.28 seconds. These are
single-run sanity results, not portable latency promises.

The [native report](native-report.json) records hashes, native IPA for golden
clauses, waveform metrics and each case's elapsed time. Review WAVs remain in
`/tmp/paradee-quality-final-2026-10-08/`; they are not committed as model-quality
proof. Reproduce them using `benchmarks/paradee/README.md`.

This pass fixed observable frontend problems: native IPA discarded internal
punctuation; dollar amounts became “dollar … point …”; `Prof.` became “prof”;
and plain `Ms` became letter names. Omaspeak now preserves clause punctuation,
protects numeric separators/clocks and initials, expands unambiguous USD into
whole dollars/cents, and expands common titles, including Ms → Miz.

23 Paradee tests pass, including exact corpus frontend plans, malformed monetary
forms, punctuation-to-token delivery and the existing native C ABI/streaming
contracts. Focused source line coverage is 96.23% (562/584 lines); no exclusions
or threshold changes. Two Python tooling tests verify invalid/silent/wrong-format
WAV rejection and clipping metrics.

Human listening and independent pronunciation/ASR scoring have not been performed.
The model remains experimental. Foreign names and context-dependent pronunciation
still use the native eSpeak fallback rather than the full Misaki lexicon; sentence
boundaries after lone initials can remain ambiguous. GPU/NPU are still unsupported.
