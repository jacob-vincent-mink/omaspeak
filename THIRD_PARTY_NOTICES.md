# Third-party notices

The MIT license in `LICENSE` applies to Omaspeak-authored source. It does not
relicense the components or models below.

## audio.cpp default provider

Linux releases include a pinned, stripped audio.cpp shared library built from
commit `e9ff20042ec85af960a720368c6927cda19ad65f`. audio.cpp is Apache-2.0.
The final Supertonic provider retains audio.cpp, ggml, cJSON, libyaml, and
BSD-3-Clause PocketFFT-derived FFT code. Its exact in-source FFT notice is
included as `POCKETFFT-LICENSE`.

The audio.cpp build graph also contains sentencepiece and its bundled
third-party sources, plus tokenizer sources derived from llama.cpp. Link-map and
symbol inspection of the Supertonic-only provider found no object code retained
from those static archives. The package nevertheless carries the source-tree
licenses collected during the pinned build and conservatively includes
llama.cpp's MIT license as `LLAMA-CPP-LICENSE`. Native model management is
disabled, so cpp-httplib is neither compiled into nor shipped with the provider.

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
