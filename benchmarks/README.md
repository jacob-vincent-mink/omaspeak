# Runtime evidence

The native-provider architecture starts a new evidence line. Results from the
obsolete runtime design were removed because they do not prove placement or
performance for complete audio.cpp providers or the current direct OpenVINO
path.

New proof bundles identify the Omaspeak commit, provider library hash, provider
upstream revision, every model-file hash, device and driver, file-only output,
placement evidence, cold load, hot synthesis, real-time factor, and
cross-provider quality checks.

The earlier direct OpenVINO result used a converted/mixed graph set that is no
longer in the catalog, so it was removed. No accelerator number is a release
claim until the official `supertonic-3-openvino` files are measured again.
