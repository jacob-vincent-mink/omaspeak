# Third-party notices

The MIT license in `LICENSE` applies to Omaspeak's project-authored source. It
does not relicense the components and model files described below.

## Supertonic reference code

Portions of `src/supertonic.rs` are adapted from the Supertonic reference
implementation, copyright 2025 Supertone Inc., under the MIT License. See
`licenses/SUPERTONIC-CODE-LICENSE`. The Supertonic model weights use different
terms described below.

## ONNX Runtime

Linux release archives may bundle the official CPU ONNX Runtime library under
its MIT License. Each archive carries ONNX Runtime's `LICENSE` and
`ThirdPartyNotices.txt` files alongside the native library.

## Rust dependencies

Release archives contain a `RUST-DEPENDENCIES.txt` file generated from the
locked, shipped dependency graph with cargo-about. CI rejects dependencies
outside the repository's explicit permissive-license allowlist.

## Text-to-speech models

Models are not part of the Omaspeak source license or release archive.

- Supertonic 3 weights are licensed under BigScience OpenRAIL-M. An installer
  must show those terms, require explicit acceptance, and preserve the exact
  license beside the installed weights.
