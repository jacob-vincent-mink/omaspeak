# File-only demo

```bash
omaspeak setup all --model supertonic-3-gguf --accept-license OpenRAIL-M
omaspeak voices
omaspeak say --voice M1 --no-play --out /tmp/m1.wav "Testing one, two, three."
omaspeak say --voice F1 --no-play --out /tmp/f1.wav "Omaspeak runs locally."
omaspeak benchmark --text "A warm inference benchmark." --out-dir /tmp/omaspeak-bench
```

No command above opens a playback device. `setup all` performs its own
file-only model-backed proof and leaves systemd untouched.
