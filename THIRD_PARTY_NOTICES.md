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

## Kokoro OpenVINO bridge

Linux releases include Omaspeak's optional C++ bridge for OpenVINO GenAI
2026.4. The bridge is built against Intel's checksummed 2026.4 SDK and links
to, but does not bundle, the OpenVINO or OpenVINO GenAI runtime libraries.
Those runtimes are separately installed under Intel's Apache-2.0 license.

## Supertonic reference code

Parts of the direct OpenVINO frontend are derived from Supertone's MIT-licensed
Supertonic reference implementation. The release retains
`SUPERTONIC-CODE-LICENSE`.

## Rust dependencies

Release archives contain `RUST-DEPENDENCIES.txt`, generated from the locked
Rust dependency graph with cargo-about.

## Models

Models are downloaded separately. The direct OpenVINO files come from the
official `supertone-oss-archive/supertonic-3` archive; the default GGUF is an
audio.cpp conversion. Both remain under BigScience OpenRAIL-M. Setup requires
explicit acceptance, verifies each pinned file, and atomically stores the model
license and canonical provenance manifest beside the installed weights.

The optional Paradee profile downloads the pinned official `sahilmahendrakar/Paradee-8M-v1.0`
FP32 graph and vocabulary metadata under Apache-2.0. Its model license is retained
in `licenses/PARADEE-8M-MODEL-LICENSE` and beside downloaded model assets. Paradee
uses a separately installed native eSpeak NG executable and English data; eSpeak
NG is GPL-3.0 and is not bundled in these archives. See `docs/PARADEE.md` for
frontend and pronunciation limitations.
