# Omaspeak demo

Validated on 2026-09-13 with `sherpa-onnx` 1.13.8 and Piper `en_US-lessac-medium`.

## Standalone

```bash
omaspeak --config /tmp/omaspeak-demo.toml say \
  "Hello Omarchy. Omaspeak is now generating this sentence locally in Rust." \
  --no-play --out /tmp/omaspeak-standalone-demo.wav
```

The model loaded in 433 ms and synthesized 93,184 samples in 111 ms. `ffprobe` verified PCM signed 16-bit, 22.05 kHz, mono, 4.226 seconds.

## Hot daemon

One daemon loaded the model in 396 ms. Two sequential requests reused that load:

| Request | Samples | Audio | Synthesis |
|---|---:|---:|---:|
| one | 72,192 | 3.274 s | 80 ms |
| two | 64,768 | 2.937 s | 69 ms |

Two clients launched concurrently were safely serialized through the same model. They generated 55,296 and 63,488 samples in 63 and 66 ms, both reporting the same 413 ms daemon load. `ffprobe` verified both as 22.05 kHz mono PCM WAVs. Shutdown removed the socket and exited 0.

## Capability behavior

The CPU binary rejected `runtime = "openvino"`, `device = "npu"` under `fallback = "error"`. Under `fallback = "cpu"`, it warned, generated a valid 1.486-second WAV in 52 ms, and reported the fallback rather than claiming NPU placement.
