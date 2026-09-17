# Named preset selection (S08)

`model.voice` and the IPC v1 `say.voice` field accept either a preset name or a
legacy integer. For example, `voice = "F1"` and `voice = 5` select the same
Supertonic speaker. Numeric identity is unchanged: M1–M5 are 0–4 and F1–F5 are
5–9. Reading and saving an existing numeric config keeps its numeric value.

```sh
omaspeak config set model.voice F1
omaspeak say --voice M3 "Your download is ready."
omaspeak voices --json
```

Names are matched case-insensitively against the selected model's inventory.
Unknown names, negative/out-of-range legacy IDs, and unknown model inventories
are rejected. Setting an invalid voice through `config set` leaves the existing
config untouched. Voice listing, setup validation, picker preselection, previews,
benchmarking and the supervised worker all resolve the same identity. The voice
picker can save the selected preset name. Config schema choices match the active
representation, preserving numeric values for existing schema clients.

A named IPC request stays named through queueing and worker handoff and resolves
against the daemon's frozen model configuration. The change is additive for
numeric IPC clients. Older daemons do not accept string voices: restart/update
the daemon together with the CLI before using the new config/IPC representation.
There is no silent substitution of an unknown voice.

This is the named-selection part of S08. Kokoro's family-specific inference
options, sample rate, language/voice constraints, phonemizer resources and catalog
activation belong to the next adapter/qualification slice. Kokoro is not offered
as runnable by this change, and no clone/design or arbitrary conditioning fields
are added.

Tests cover every Supertonic numeric/name pair, config serialization and malformed
selection types. A native ABI fixture accepts only F1: named/default requests and
legacy 5 succeed through both local speech and daemon IPC; legacy 0 remains M1
and is rejected by that fixture. Existing cancellation, queue and preview tests
remain active. Tests use private paths and no real audio devices or services.
