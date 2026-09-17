# Bounded speech requests (S06)

The daemon now handles control requests while one persistent child owns the
synthesis engine. Normal requests reuse that worker and model. Cancelling native
synthesis terminates and reaps its process group; the next request starts a fresh
worker with the daemon's frozen configuration and original path context. Editing
a config file cannot silently change a replacement worker's model or placement.
The direct OpenVINO runtime is initialized only inside that worker. Nested
audio.cpp workers receive a parent-death signal and share its process group.

## User commands and protocol

- `omaspeak cancel` cancels the active request and keeps the daemon available.
- `omaspeak cancel REQUEST_ID` cancels that active or queued request.
- `omaspeak stop` retains its existing shutdown meaning and cancels all work.
- Disconnecting a requesting client cancels its active work or discards its
  queued request. No compensating resume is needed for owned playback pauses.

IPC v1 adds `{"protocol":1,"id":"control-id","type":"cancel","request_id":"speech-id"}`;
a null or omitted `request_id` selects the active request. The response type is
`cancelled`, with the requested ID and a `count` of zero or one. The original
speech connection receives a `cancelled` error. IDs must contain 1–128 bytes;
duplicate active/queued speech IDs are rejected. Failed native requests are not
automatically replayed; a later request can recover with a new native worker.

`status --json` exposes `backend.requests.active`, `queued`, `capacity`, `loading`
and `worker_ready`. Request IDs remain attached to their own replies. Control
requests do not join the speech queue.

## Bounds and output ownership

There is one active request, eight waiting speech requests and sixteen connections
being read. Each incomplete request has a two-second read deadline. Control
frames have a hard cap of 1 MiB + 16 KiB, also constrained by configured text size
plus protocol overhead. Overflow receives `busy`; malformed/oversized requests
receive `invalid_request`. A partial client cannot block status or cancellation.
Requests and worker startup have a five-minute deadline. Polling is every 10 ms;
these are scheduling bounds, not hard real-time guarantees under host contention.

Synthesis writes to an exclusively created, mode-0600 staging WAV beside the
requested destination. Only successful synthesis publishes it by rename.
Cancellation and synthesis failure preserve an existing destination and remove
owned staging. Once synthesis has succeeded, cancellation during playback keeps
the completed export. Playback uses the existing owned-pause worker; cancellation
stops/reaps that process group and releases its own wake hold. File-only requests
never pause wake detection or open an audio output device.

SIGINT/SIGTERM shutdown closes requests and removes the daemon socket. Forced
SIGKILL cannot run file cleanup and can leave a private staging file or stale
socket; existing startup socket checks still apply. Parent-death signals release
native/player resources. This is not a durable job queue: requests do not survive
restart, and cancelling synthesis intentionally loses the hot model until reload.

## Acceptance evidence

The CLI regression uses a real daemon and native ABI fixture stalled inside its
synthesis call. Status remains responsive, eight requests queue, duplicate IDs
and a ninth queued request are rejected, queued and active cancellation stay
separate, cancellation completes within a one-second test budget, original output
survives, and the remaining queued requests finish. It also checks a slow partial
client, invalid input, native crash without replay, recovery and config edits
between cancellation and worker replacement, disconnect recovery, full read admission,
failed publication, and shutdown with active and queued work. The internal worker
protocol rejects control/playback requests and supports clean shutdown. Other tests cover playback client
interruption and process cleanup.

[Real OpenVINO CPU evidence](../benchmarks/results/2026-09-16-request-lifecycle/RESULTS.md)
records twelve byte-identical fixed-seed outputs, warm-worker reuse, long-request
cancellation and recovery. GPU cancellation timing, real-room echo and human
listening are not established by these file-only results.
