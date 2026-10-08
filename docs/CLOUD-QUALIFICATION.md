# Cloud qualification handoff

Cloud protocol fixtures pass locally. Paid-account access, real provider playback,
pronunciation and real-room behavior still require independent qualification.
No live vendor requests were made during development.

## Configure and discover

Run `omaspeak setup cloud` for guided cloud setup, or configure explicitly:

```sh
omaspeak setup cloud --provider elevenlabs
omaspeak cloud credential install --stdin < /private/elevenlabs-key
omaspeak cloud voices --json
omaspeak cloud voices --save PROVIDER_ID --alias reader
omaspeak cloud credential check
```

Discovery fetches your ElevenLabs or Cartesia account inventory with bounded
pagination. Other providers use configured preset IDs. The regular `voices`
command lists saved aliases offline. `setup cloud --voice ID --model MODEL`
configures an explicit voice without discovery.

The credential installer writes a private, owned regular file with mode 0600.
Config contains its absolute path, never the key. Both CLI and daemon read this
file directly, so no systemd environment import is needed. An environment key
still takes precedence; remove a stale environment key before testing rotation.
Restart an existing daemon after changing configuration. `cloud credential check
--daemon` checks the service process's environment and the saved key file without
printing secrets; its file check uses the saved configuration, not proof that the
daemon has reloaded it. Existing environment-only configurations remain valid.
Neither credential checks nor ordinary `setup check` contact a provider.

## Explicit smoke and repeatable corpus

```sh
omaspeak cloud smoke --out /tmp/voice-smoke.wav
omaspeak cloud smoke --no-play --out /tmp/voice-smoke-file.wav
python3 benchmarks/cloud/qualify.py --binary ./omaspeak --config /path/config.toml \
  --out-dir /tmp/elevenlabs-qualification --accept-charges
```

Smoke sends one synthesis request and may incur charges. It measures first PCM,
total generation and audio duration, publishes a complete WAV without overwriting
an existing output, and plays it unless `--no-play` is supplied. The corpus driver
makes three requests, saves WAVs and `report.json`, and creates a human review
sheet. Use a new output directory for every provider/voice/build. Run it separately
for ElevenLabs, Cartesia, Deepgram and an actual OpenAI-compatible service.

For each provider, review all WAVs for pronunciation, missing/repeated speech,
pauses and PCM continuity; note model/voice, account access, architecture and date.
Also run a long `say` request and cancel it during playback, then speak again.
Check pinned/default output devices, first-audio responsiveness and absence of
replay after cancellation. Keep provider failures/statuses in the report, without
keys or account credentials. Do not equate a successful HTTP call with qualified
audio quality. Paradee has its own [native quality corpus](../benchmarks/paradee/README.md).
