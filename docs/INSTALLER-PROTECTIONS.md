# Provider discovery and installation boundaries

This implements the family-discovery and installer-protection slice of S03/S04.
The required audio.cpp families are `supertonic`.

The runtime probe now queries the pinned audio.cpp registry before treating a
provider as usable for the application's default. An ABI-compatible library
that omitted a required family is rejected with the missing family name and
its registered alternatives. Listing the bundled catalog remains offline;
registry discovery runs only in the existing native runtime-probe path.
Registration is not a claim about device placement, inference or output quality.

Installers take a nonblocking per-profile filesystem lock before verifying an
existing installation or touching its downloads/staging directory. A second
writer gets a clear busy error and can retry. Different profiles retain
independent locks. Lock files remain in `models/.locks`: removing the inode on
unlock would allow two simultaneous owners. Locks release on normal completion,
error or process exit, without stale PID-based ownership checks.

Before transferring or staging assets, setup checks filesystem free space for
the complete additional staging copy plus unverified download objects and a
16 MiB metadata reserve. Verified cached files need no new download allocation.
The budgets are combined when models and downloads share a filesystem and
checked separately otherwise. Existing models/caches already consume reported
free space and remain in place while the replacement is prepared. Preflight is
a point-in-time check, not a disk reservation; concurrent unrelated writes can
still exhaust storage and follow the existing failure/rollback path.

SIGINT/Ctrl-C and SIGTERM request cancellation during installation. Hashing,
copying and download loops check between 128 KiB chunks. Network requests have
a 10-second connect timeout and a five-second idle read timeout. Cancellation
cleans this install's staging and partial transfer, preserves verified reusable
cache objects and the active model, and releases the lock. The final atomic
publication/rollback sequence is allowed to finish rather than abandoning a
half-published target. Normal termination signals still work after installation.
Forced termination such as SIGKILL cannot run cleanup; the filesystem lock
still releases, but abandoned staging/partial files may require later cleanup.

Existing license acceptance, pinned hashes, provenance manifests, replacement
rollback and separately fingerprinted compiled caches retain their roles. This
change does not estimate unknown device-cache compilation sizes, implement
range-request download resume, or qualify additional hardware. Cache capacity
estimation and power/quality qualification remain explicit follow-ups.

Regression coverage includes competing installers, lock reuse, insufficient
space and overflow, cancellation cleanup, real signals in disposable processes,
registry enumeration failures and missing required families. Existing corruption,
license and activation tests continue to cover the transaction boundaries.

The TUI now uses the same isolated family-aware probe as runtime CLI setup.
A file named `libaudiocpp.so` is no longer treated as a ready provider merely
because it exists.

## Acceptance audit, 2026-09-16

The current failure tests were inspected, not just counted. The lifecycle-branch
full suite also ran these tests successfully. S04's implementation checks map to:

| Boundary | Existing acceptance evidence |
|---|---|
| Fresh setup | `fresh_full_setup_stages_audio_cpp_cpu_and_gguf_together` exercises coherent provider/model selection before activation |
| Failed installed-model proof | `installed_model_with_failed_provider_proof_keeps_active_config_unchanged` checks original config bytes and explicit installed-but-not-activated remediation |
| Failed cache compilation | `setup_precompile_failure_leaves_config_launcher_and_service_untouched` injects an NPU compile failure and asserts unchanged config with no launcher or restart call |
| Later setup failures | `setup_transaction_restores_existing_and_new_configs_on_late_failures` covers config restoration/removal; separate tests cover concurrent rollback errors |
| Corrupt downloads/cache | `tests/unit/setup_model.rs` verifies size/hash limits before publication, partial cleanup and replacement of corrupt reusable cache |
| Competing installs, cancellation, disk budget | `tests/unit/install_guard.rs` checks lock exclusion/reuse, owned staging cleanup, original asset preservation, insufficient space/overflow and real signal cancellation |

This closes the acceptance mapping for implemented S04 protections. Injected
failures establish transaction ordering; they do not simulate every driver or
mid-compilation disk failure. Unknown compiled-cache sizing and forced-kill debris
remain the documented limits above, rather than claims of completed features.
