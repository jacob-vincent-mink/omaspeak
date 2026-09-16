# Voice-list preview smoke test

Recorded 2026-09-15 using the installed, catalog-verified Supertonic 3 OpenVINO
model, two CPU threads, fallback disabled, and an 80×24 pseudo-terminal.
Configuration, cache, state and runtime files were isolated under temporary XDG
paths. The model directory was a read-only-use link to the existing installation.

The test opened `setup model`, selected the installed OpenVINO profile, and
pressed Space on F1 in the voice list. Actual synthesis produced 263,397 samples
at 44,100 Hz (about 5.97 seconds). A fake `pw-play` captured the generated WAV
without sending audio to speakers. The menu reported sample completion. The
test then moved to another voice, started preparation, and cancelled the menu.
Both the isolated and live config bytes remained unchanged, all preview
scratch directories were removed, and `last.wav` was not created.

See [machine-readable evidence](evidence.json) for the waveform hash. This is
synthesis, TUI, handoff and cancellation evidence, not a listening-quality,
physical audio-output or accelerator-qualification result. The model's pinned
asset manifest is unchanged by this work. No audio recording is committed.

Automated tests additionally exercise the audio.cpp worker with a native test
provider, ensure that preview bypasses a listening daemon socket, test player
selection and process cleanup, and check terminal viewport bounds at 24×8,
80×24 and very small dimensions. Missing models produce an inline install-first
message; preview never initiates installation.
