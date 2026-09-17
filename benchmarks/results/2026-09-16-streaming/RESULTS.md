# Supertonic pull-streaming evaluation (S07)

Date: 2026-09-16. Decision: **defer production streaming at the current native
pin**. Keep the offline path. The API gives useful early audio for long replies
and exact export parity, but retains the entire utterance internally. Pull
backpressure alone does not bound that retained PCM to a fixed streaming budget.
No real speaker, microphone or installed service was used for this evaluation.

## Reproduction and provenance

Run [the standalone ctypes probe](../../streaming/probe.py) against audio.cpp
`e9ff20042ec85af960a720368c6927cda19ad65f`, with Supertonic 3 original-dtype GGUF,
M1, English, seed 20260916, eight inference steps, device 0 and two CPU threads.
The probe loads the model once, creates offline/streaming sessions, warms each
with a short request, then compares short and repeated long text. It writes raw
float PCM incrementally and asserts that chunk concatenation, `stream_finish`
and offline synthesis have exactly the same SHA-256 for each measured case.
Borrowed event PCM is copied and the event freed before requesting another.

Host: `promaxgb10-d666`, NVIDIA GB10, shared with an existing SGLang workload.
These are observations, not controlled regression or power measurements.

- Native library SHA-256: `7e0f52a37bafda9f83302eebd279252c89a16a1c863dba85ada716829fa75638`.
- Model SHA-256: `af814486a0bc9513fb36afabd9b1155ad14fb2c36a107ac6ffe62ea9adafb662`.
- Raw outputs/logs: `/tmp/oma-model-qualification-F1DZcy/streaming-{first,paced,cpu}`
  and their adjacent `.log` files on the qualification host.

```sh
python3 benchmarks/streaming/probe.py --library /path/libaudiocpp.so.0.1.0 \
  --model /path/supertonic-3-orig.gguf --backend cuda --out /tmp/streaming-first
python3 benchmarks/streaming/probe.py --library /path/libaudiocpp.so.0.1.0 \
  --model /path/supertonic-3-orig.gguf --backend cuda --streaming-first \
  --repetitions 32 --pull-delay-ms 100 --out /tmp/streaming-paced
```

The output directory must not already exist. No extra Python packages are needed.
For the CPU run, use the first command with `--backend cpu` and a fresh output
directory. Use an external process timeout when probing a new provider. The second run
reverses mode order and waits 100 ms after each consumed chunk; that delay is
included in streaming total time, but not in first-audio time.

## Observations

| Requested backend / case | Offline complete | First streamed audio | Stream complete | PCM duration / chunks |
|---|---:|---:|---:|---|
| CUDA: short, normal pulls | 52.3 ms | 48.8 ms | 49.4 ms | 3.75 s / 1 |
| CUDA: long, 8 repetitions | 561.8 ms | 101.8 ms | 533.1 ms | 74.69 s / 5 |
| CUDA: short, paced pulls | 50.4 ms | 54.0 ms | 155.1 ms | 3.75 s / 1 |
| CUDA: long, 32 repetitions, paced | 1947.1 ms | 113.7 ms | 4084.3 ms | 298.62 s / 20 |
| CPU: short | 1820.0 ms | 1810.7 ms | 1811.2 ms | 3.75 s / 1 |
| CPU: long, 8 repetitions | 35808.3 ms | 6845.2 ms | 35641.8 ms | 74.71 s / 5 |

All three runs passed exact PCM equality within each requested backend for short and long text. CPU and CUDA outputs
are not claimed byte-identical to each other. Short notifications
have only one chunk and no established latency advantage. Long replies deliver
usable PCM earlier. The largest chunk in the paced long run was 2,896,164 bytes
(about 16.4 seconds at 44.1 kHz mono float32), a player with a smaller PCM budget would need to subdivide it. Audio duration
is not synthesis latency: each pull is synchronous, and this probe does not
measure cancellation during that native call.

RSS did not increase during any 100 ms pause between pulls. This is consistent
with synchronous demand-driven synthesis; it does not prove fixed total memory.
The short long-request retained 13,175,892 PCM bytes and the longer request
retained 52,677,364 bytes for its final result. RSS before/after the longer stream
was 580,300,800 / 935,411,712 bytes, including runtime allocations; that delta is
not attributed entirely to retained PCM. JSON includes per-chunk timings and RSS.

## Why production streaming stays deferred

The pinned [Supertonic session implementation](https://github.com/0xShug0/audio.cpp/blob/e9ff20042ec85af960a720368c6927cda19ad65f/src/models/supertonic/session.cpp#L139)
appends every synthesized chunk to `stream_merged_audio_` before returning it.
`finish_stream` moves that full buffer into the final result; `reset` clears it
but also discards all remaining text chunks. There is no retain-final-output
switch in this session. Freeing consumed events cannot release the merged copy.
The application's text-size limit is finite, but it is not a fixed PCM buffering
budget and does not establish a safe duration bound across arbitrary input.

A production streaming path should not be enabled on this evidence alone. Reopen
when the pinned provider can avoid retaining merged PCM, or when a separately
qualified adapter can preserve chunking/RNG/output identity while bounding total
buffering. Then implement an incremental WAV sink and bounded player queue,
measure actual stop latency and underruns, and obtain listening evidence for
chunk joins. Direct OpenVINO streaming needs its own qualification. Exact native
PCM parity is not a listening, acoustic-feedback, or device cancellation pass.
