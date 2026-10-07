# Cloud speech adapters

Cloud inference is opt-in. Existing local defaults remain in place. Set
`backend.kind` to `elevenlabs`, `openai-compatible`, `cartesia`, or `deepgram`.
Use `runtime = "default"`, `device = "remote"`, `fallback = "error"` and
`device_id = 0`. Status reports remote placement; no model download or native
provider library is required. Native model/options fields do not configure remote
models: use `backend.cloud.model` and `backend.cloud.voice` instead.

| Kind | Default API root | Model default | Key environment default |
| --- | --- | --- | --- |
| elevenlabs | https://api.elevenlabs.io | eleven_flash_v2_5 | ELEVENLABS_API_KEY |
| openai-compatible | https://api.openai.com/v1 | gpt-4o-mini-tts | OPENAI_API_KEY |
| cartesia | https://api.cartesia.ai | sonic-3.6 | CARTESIA_API_KEY |
| deepgram | https://api.deepgram.com/v1 | aura-2-thalia-en | DEEPGRAM_API_KEY |

ElevenLabs and Cartesia require an explicit provider voice ID. OpenAI-compatible
defaults to `alloy`; Deepgram uses an Aura model as its voice ID. A Deepgram
`cloud.model` override also selects its voice unless `cloud.voice` is explicit.
Cartesia pins `Cartesia-Version: 2026-08-14`.

Example configuration:

```toml
[backend]
kind = "elevenlabs"
runtime = "default"
device = "remote"
fallback = "error"

[backend.cloud]
model = "eleven_flash_v2_5"
voice = "YOUR_PROVIDER_VOICE_ID"
api_key_env = "ELEVENLABS_API_KEY"
timeout_seconds = 60
max_audio_seconds = 300

[model]
voice = 0
language = "en"
```

Export the key in the environment of the process that performs inference.
A running daemon needs its own environment configured and a restart; exporting
in another terminal does not update it. The config stores the environment
variable **name**, never the key. `config set backend.cloud.api_key_env NAME`
and the other scalar cloud settings are supported. `setup check` validates
configuration and credential availability without making a paid synthesis request.
A missing native model/library is not a cloud setup failure.

For selectable named voices, replace `cloud.voice` with aliases:

```toml
[backend.cloud.voices]
reader = "PROVIDER_VOICE_ID_1"
assistant = "PROVIDER_VOICE_ID_2"
```

Use `say --voice reader TEXT` or set `model.voice = "reader"`. Alias indices
follow sorted alias names; use names for durable selections. `voices --json`
shows this configured inventory without a paid discovery call.

All four routes request signed 16-bit little-endian mono PCM at 24 kHz, stream
HTTP audio into the existing playback sink, and record the same samples to WAV.
Chunk boundaries may split individual PCM samples. Playback cancellation,
timeouts, output staging, wake holds and no-replay semantics use the existing
supervised worker. `--no-play` performs buffered file generation.

Speed ranges: ElevenLabs 0.7–1.2, OpenAI-compatible 0.25–4, Cartesia 0.6–1.5,
Deepgram 0.7–1.5. Provider/model-specific restrictions can still return HTTP errors.
ElevenLabs and Cartesia receive `model.language`; OpenAI speech infers language
from input text. Initial Deepgram TTS requires English. This adapter does not
expose SSML, cloning, expressive instructions or incremental text input.

Custom `base_url` is the API root shown above (include `/v1` for compatible
OpenAI/Deepgram servers). HTTPS is required except explicit loopback HTTP for
local servers/tests. URLs cannot contain user-info, query strings or fragments.
Redirects are disabled. Custom OpenAI-compatible servers may run without keys;
set `api_key_env` to an unset variable if the server is keyless and an ambient
OPENAI_API_KEY exists. A compatible server must honor the raw PCM contract;
compatibility is not established by accepting an arbitrary server URL.

Requests have configurable 1–300 second deadlines and at most 1–300 seconds
of returned PCM. HTTP errors report status, not bodies or credential-bearing
URLs. No retry follows partial audio. Loopback tests cover provider routes,
headers/bodies, fragmented PCM, early playback, malformed replies, redirects,
error redaction and preserving existing output. No paid-provider or perceptual
qualification was performed.

Protocol references: [ElevenLabs](https://elevenlabs.io/docs/api-reference/text-to-speech/stream),
[OpenAI](https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create),
[Cartesia](https://docs.cartesia.ai/api-reference/tts/bytes),
[Deepgram](https://developers.deepgram.com/reference/text-to-speech/speak-request).
