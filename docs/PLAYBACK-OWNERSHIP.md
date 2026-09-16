# Playback pause ownership (S06 / W10)

Before playing speech, Omaspeak connects to the Omawake control socket under
`$XDG_RUNTIME_DIR/omawake/control.sock` (or Omawake's existing temporary-runtime
fallback), sends an IPC v1 `hold_pause` request with a fresh ID, and waits for a
matching `paused` acknowledgement. It retains that connection through playback.
File-only synthesis never acquires a pause. An absent/stale Omawake socket is
allowed; an incompatible running daemon, malformed reply or acknowledgement
that takes more than three seconds fails playback explicitly. No manual pause
or resume command is sent, and no service is started or stopped.

The companion Omawake change acknowledges only after capture is released. Each
connection owns its hold independently of other playback and manual pauses.
Omaspeak's player inherits the hold descriptor, so parent death alone cannot
resume wake detection before the player exits. If the wake daemon disappears,
Omaspeak cancels playback. Normal completion, cancellation and player failure
close the hold automatically. The existing parent-death signal still stops the
player when Omaspeak exits unexpectedly.

Voice-list auditions use a separate playback worker for the same handshake.
Negotiation cannot block the TUI; moving rows, Space, Escape and the existing
120-second preview limit still terminate the audition's process group. The
worker owns no model state and does not contact the speech daemon.

Tests cover missing wake daemon, cancellation before acquisition, mismatched
acknowledgements and descriptor inheritance using a silent pipe-blocked child.
A CLI test proves the player cannot start before acknowledgement and that an
older wake daemon's rejection prevents playback. All endpoints and the fake
player are isolated; no microphone or audio device is used.

The [bounded request lifecycle](REQUEST-LIFECYCLE.md) now adds prompt synthesis
stop, bounded admission, active/queued cancellation and frozen worker recovery.
Acoustic playback echo still needs real-room validation. The companion wake
ownership change is required when Omawake is running; no unsafe compatibility
fallback is provided.
