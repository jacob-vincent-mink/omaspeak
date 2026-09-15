# Omaspeak demo

Run guided setup, accept the Supertonic model license, and choose a runtime,
device, model, and voice:

```bash
omaspeak setup
```

Generate audio without playing it through speakers:

```bash
omaspeak say \
  "Omaspeak generated this sentence locally." \
  --no-play --out /tmp/omaspeak-demo.wav
```

Inspect the selected runtime and model readiness:

```bash
omaspeak status --json
omaspeak setup check
```

`say` loads the engine on demand and exits when no daemon is listening. For
repeated synthesis, start the daemon explicitly in one terminal:

```bash
omaspeak daemon
```

Then use a second terminal to send requests to its hot model:

```bash
omaspeak say "The model is already warm." --no-play --out /tmp/omaspeak-hot.wav
omaspeak stop
```

`status` and `setup check` confirm the requested configuration and device
availability. The checked-in benchmark reports contain the separate execution
placement evidence.
