# Third-party notices

The MIT license in `LICENSE` applies to Omaspeak-authored source. It does not
relicense the components or models below.

## audio.cpp default provider

Linux releases include a pinned, stripped audio.cpp shared library built from
commit `e9ff20042ec85af960a720368c6927cda19ad65f`. audio.cpp is Apache-2.0.
Release archives include its license and the license of every third-party
component statically linked into that provider, including ggml, sentencepiece,
libyaml, cJSON, and cpp-httplib where present in the pinned source build.
The pinned audio.cpp tree also compiles tokenizer sources derived from
llama.cpp into its runtime archive. The Supertonic-only final shared library
does not retain their symbols after static archive extraction, but releases
conservatively include llama.cpp's MIT license as
`LLAMA-CPP-LICENSE`.

## Supertonic reference code

Parts of the direct OpenVINO frontend are derived from Supertone's MIT-licensed
Supertonic reference implementation. The release retains
`SUPERTONIC-CODE-LICENSE`.

## Rust dependencies

Release archives contain `RUST-DEPENDENCIES.txt`, generated from the locked
Rust dependency graph with cargo-about.

## Models

Models are downloaded separately. Supertonic 3 weights and converted GGUF
artifacts remain under BigScience OpenRAIL-M. Setup requires explicit
acceptance, verifies pinned bytes, and stores the license and provenance beside
the installed weights.
