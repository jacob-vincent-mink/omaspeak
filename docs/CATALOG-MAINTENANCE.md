# Catalog maintenance, imports and URL health (S10)

Reproducible maintenance of the pinned model catalog. This page documents the
tooling and the maintenance procedure for Omaspeak's pinned Supertonic and
Kokoro catalogs.

## Pinned URL health check

```sh
omaspeak setup model --check-urls          # human-readable, exits nonzero on failure
omaspeak setup model --check-urls --json   # machine-readable report
```

Issues one `HEAD` request per pinned asset URL (18 files across the three
catalog models) and compares the reported `Content-Length` against the pinned
byte size. Per-file outcomes are `ok`, `size-mismatch` (origin reports a
different size than the pin) or `unreachable` (network/HTTP error). The
command downloads nothing, writes nothing, and never changes user pins or the
configuration; a nonzero exit means at least one default origin is unhealthy
and should be investigated or re-pinned through a normal catalog change.

Maintainers can exercise the same logic against a mirror or a stub by
rewriting the origin only for the check:

```sh
omaspeak setup model --check-urls --url-prefix http://127.0.0.1:8080
```

`--url-prefix` replaces the `scheme://authority` of each pinned URL, keeping
the path and query. It is a check-only rewrite: downloads always use the
pinned URL.

Latest health snapshot: [2026-09-16 evidence](../benchmarks/results/2026-09-16-catalog-check/url-checks.json)
(18/18 `ok`).

## Offline local import

```sh
omaspeak setup model --download <model> --source /path/to/exact-assets
```

Imports a pre-downloaded asset set without network access. Every file must be
named exactly as the catalog pins it and still passes the pinned size and
SHA-256 verification; mismatched files fail with diagnostics naming the
expected and observed values (`expected {N} bytes, found {M}`,
`expected {sha256}, found {sha256}`). Import is atomic with rollback, and the
installed tree carries a provenance manifest recording the install source,
original model/revision/license, and the exact pinned files and license
notices.

User pins are never replaced silently: an installed-and-verified model is
reported `already-installed` and left untouched; activation happens only after
verification and (where required) license acceptance succeed, with the
provider proof gating activation (see [INSTALLER-PROTECTIONS.md](INSTALLER-PROTECTIONS.md)).

## Reproducible import recipe

1. Obtain the exact artifacts from the pinned origins (URLs live in
   `src/catalog.rs` next to their sizes and SHA-256 hashes).
2. Verify locally (optional): `sha256sum` against the catalog pins.
3. `omaspeak setup model --download <model> --source <dir> --accept-license <LICENSE>`
   (only models whose license requires acceptance).
4. `omaspeak setup model --verify <model>` to re-verify the installed tree.

## Pin bump procedure

A catalog change is a normal reviewed patch: update the file `url`, `size`,
`sha256`, source revision and license notice in `src/catalog.rs`, add or
refresh the license text, run `--check-urls`, then reinstall and re-verify.
The audio.cpp engine additionally enforces exact asset names, sizes and
hashes at load time, so stale or substituted files fail loudly instead of
silently running.
