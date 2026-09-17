# Persistent request worker and cancellation, 2026-09-16

A release build ran a private daemon using Supertonic 3, direct OpenVINO CPU,
two threads, five steps, fixed seed 20260916 and fallback `error`. Model files
were read from the existing verified installation. Config, cache, state and
runtime directories for this check were isolated under the qualification archive;
no installed service, microphone or audio player was used.

[Raw results](results.json) include the binary/config/corpus hashes, daemon status,
output hashes and timing. [Configuration](speech.toml) identifies exact settings.
The model/provider provenance is unchanged from the
[earlier paired qualification](../2026-09-16-qualification/RESULTS.md).

- All six English cases with M1/F1 produced valid WAVs; all twelve files are
  byte-identical to their earlier fixed-seed candidate outputs.
- The same request-worker PID served all twelve normal requests. Normal operation
  retains the hot model; it does not spawn an engine for every sentence.
- A long request (sixty repetitions of the long-reply case) was cancelled after
  becoming active and a further 100 ms. Cancellation acknowledgement took
  **136.6 ms**. The previous destination bytes remained intact.
- A subsequent request loaded a replacement worker and returned valid audio.
- Explicit shutdown exited zero and removed the private daemon socket.

This is one shared-host CPU observation, not a GPU claim, worst-case cancellation
bound or matched performance comparison. The synthetic native regression proves
cancellation after entering a deliberately blocked native call, under a one-second
acceptance budget. Human quality, real-room acoustic playback and controlled
power checks remain open.

Reproduction: build with `cargo build --release --locked`, use the linked config
with the pinned model directory, and set all five XDG roots to private directories.
Start `omaspeak --config CONFIG daemon`; send the versioned English corpus over
IPC with `no_play=true`, M1/F1 and unique request IDs. Compare output hashes with
the seeded reference pack. For cancellation, preserve known destination bytes,
submit the repeated long case, wait for its ID in `status`, wait 100 ms, then send
`cancel` for that ID. Check the response, preserved output and a recovery request,
then `stop` and verify socket/process cleanup. Exact host-specific driver script
and WAVs remain in `.qualification/20260916-lifecycle` in the maintainer workspace.
