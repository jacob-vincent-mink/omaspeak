# Omaspeak demo

Run guided setup, accept the Supertonic model license, and choose a runtime,
device, model, and voice:

```bash
omaspeak setup
```

Generate audio without playing it through speakers:

```bash
omaspeak say \
  "Hello Omarchy. Omaspeak generated this sentence locally." \
  --no-play --out /tmp/omaspeak-demo.wav
```

Inspect the selected runtime and model placement:

```bash
omaspeak status --json
omaspeak setup check
```

For repeated synthesis, start the daemon explicitly and send requests to its
hot model:

```bash
omaspeak daemon
omaspeak say "The model is already warm." --no-play --out /tmp/omaspeak-hot.wav
omaspeak stop
```
