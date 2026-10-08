use super::*;

#[test]
fn misaki_conversion_matches_upstream_frontend_examples() {
    assert_eq!(
        to_misaki("aɪ aʊ dʒ eɪ tʃ ɔɪ oʊ ɜːɹ ɜː ɪə ɾ ʔ"),
        "I W ʤ A ʧ Y O ɜɹ ɜɹ iə T t"
    );
    assert_eq!(to_misaki("ɐ ɐb n̩ əl əlz ʔˌn̩ ɚ"), "ɐ əb ᵊn ᵊl əlz tn əɹ");
    assert_eq!(to_misaki("həlˈoʊ wˈɜːld"), "həlˈO wˈɜɹld");
}

#[test]
fn tokens_are_padded_and_long_input_never_truncates() {
    let vocab = BTreeMap::from([('a', 43), (' ', 16)]);
    let phonemes = format!("{} {}", "a".repeat(509), "a".repeat(700));
    let chunks = encode(&phonemes, &vocab).unwrap();
    assert!(chunks.len() >= 3);
    assert!(
        chunks
            .iter()
            .all(|ids| ids.len() <= 512 && ids.first() == Some(&0) && ids.last() == Some(&0))
    );
    assert_eq!(
        chunks.iter().flatten().filter(|id| **id == 43).count(),
        1209
    );
}

#[test]
fn unknown_phonemes_and_empty_frontend_are_errors() {
    let vocab = BTreeMap::from([('a', 43)]);
    assert!(encode("", &vocab).is_err());
    assert!(
        encode("a☃", &vocab)
            .unwrap_err()
            .to_string()
            .contains("unsupported phoneme")
    );
}

#[test]
fn cpu_only_catalog_and_named_voice_are_explicit() {
    assert_eq!(
        crate::catalog::models()[0].id,
        crate::catalog::DEFAULT_MODEL_ID
    );
    assert_eq!(crate::catalog::backends()[0].kind, "audiocpp");
    let spec = crate::catalog::model(crate::catalog::PARADEE_MODEL_ID).unwrap();
    assert!(spec.compatible_with("paradee-openvino", Runtime::Openvino, "CPU"));
    for device in ["gpu", "npu", "auto"] {
        assert!(!spec.compatible_with("paradee-openvino", Runtime::Openvino, device));
    }
    assert_eq!(spec.voices.len(), 1);
    assert_eq!(spec.voices[0].name, "af_heart");
    assert_eq!(spec.download_size(), 36_987_891);
}

#[test]
fn model_asset_paths_cannot_escape_directory() {
    assert!(model_asset(Path::new("/tmp"), "../file").is_err());
    assert!(model_asset(Path::new("/tmp"), "/file").is_err());
    assert!(model_asset(Path::new("/tmp"), "").is_err());
}

#[test]
fn execution_evidence_requires_only_cpu_devices() {
    assert!(cpu_execution("CPU"));
    assert!(cpu_execution("[CPU.0, CPU.1]"));
    for devices in ["", "GPU", "CPU,GPU", "CPU.UNKNOWN"] {
        assert!(!cpu_execution(devices));
    }
}

#[cfg(target_os = "linux")]
fn frontend_fixture(body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "omaspeak-paradee-frontend-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    fs::write(&path, format!("#!/bin/sh\nexec python3 -c '{body}'\n")).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
    path
}

#[test]
#[cfg(target_os = "linux")]
fn frontend_drains_output_while_writing_input_and_bounds_excess() {
    let frontend = frontend_fixture(
        "import sys; sys.stdout.write(\"a\"*40000); sys.stdout.flush(); sys.stdin.read()",
    );
    let phonemes = phonemize_with_deadline(
        &frontend,
        &"b".repeat(MAX_FRONTEND_INPUT),
        Duration::from_secs(2),
    )
    .unwrap();
    assert_eq!(phonemes.len(), 40000);
    fs::remove_file(frontend).unwrap();
    let frontend = frontend_fixture(
        "import sys; sys.stdout.write(\"a\"*100000); sys.stdout.flush(); sys.stdin.read()",
    );
    let error = phonemize_with_deadline(&frontend, "hello", Duration::from_secs(2)).unwrap_err();
    assert!(
        error.to_string().contains("excessive phonemes"),
        "{error:#}"
    );
    fs::remove_file(frontend).unwrap();
}

#[test]
#[cfg(target_os = "linux")]
fn frontend_deadline_covers_blocked_input_and_reaps_process() {
    let frontend = frontend_fixture("import time; time.sleep(30)");
    let started = Instant::now();
    let error = phonemize_with_deadline(
        &frontend,
        &"b".repeat(MAX_FRONTEND_INPUT),
        Duration::from_millis(100),
    )
    .unwrap_err();
    assert!(error.to_string().contains("timed out"), "{error:#}");
    assert!(started.elapsed() < Duration::from_secs(1));
    fs::remove_file(frontend).unwrap();
}

#[test]
#[cfg(target_os = "linux")]
fn frontend_dies_if_file_synthesis_parent_disappears() {
    const HELPER: &str = "OMASPEAK_TEST_PARADEE_FRONTEND";
    if let Some(frontend) = std::env::var_os(HELPER) {
        let _ = phonemize(Path::new(&frontend), "hello");
        return;
    }
    let marker = std::env::temp_dir().join(format!(
        "omaspeak-paradee-parent-death-{}",
        std::process::id()
    ));
    let _ = fs::remove_file(&marker);
    let body = format!(
        "import os,time; open(\"{}\",\"w\").write(str(os.getpid())); time.sleep(30)",
        marker.display()
    );
    let frontend = frontend_fixture(&body);
    let mut parent = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "paradee::tests::frontend_dies_if_file_synthesis_parent_disappears",
        ])
        .env(HELPER, &frontend)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let started = Instant::now();
    let pid = loop {
        let pid = fs::read_to_string(&marker)
            .ok()
            .and_then(|value| value.parse::<u32>().ok());
        if pid.is_some() || started.elapsed() >= Duration::from_secs(2) {
            break pid;
        }
        thread::sleep(Duration::from_millis(5));
    };
    let _ = parent.kill();
    let _ = parent.wait();
    let pid = pid.expect("frontend PID marker before killing synthesis parent");
    let exited = || {
        fs::read_to_string(format!("/proc/{pid}/stat"))
            .map(|stat| {
                stat.rsplit_once(')')
                    .is_some_and(|(_, fields)| fields.trim_start().starts_with('Z'))
            })
            .unwrap_or(true)
    };
    let started = Instant::now();
    while !exited() && started.elapsed() < Duration::from_secs(1) {
        thread::sleep(Duration::from_millis(5));
    }
    let dead = exited();
    if !dead {
        unsafe {
            libc::kill(pid as i32, libc::SIGKILL);
        }
    }
    fs::remove_file(frontend).unwrap();
    let _ = fs::remove_file(marker);
    assert!(dead, "frontend survived its synthesis parent");
}

struct ModelFixture {
    directory: PathBuf,
    frontend: PathBuf,
    config: Config,
    paths: AppPaths,
}

impl ModelFixture {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let directory = std::env::temp_dir().join(format!(
            "omaspeak-paradee-model-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::create_dir_all(&directory).unwrap();
        fs::write(
            directory.join("paradee.onnx"),
            b"fixture supplied to compiler boundary",
        )
        .unwrap();
        let metadata = serde_json::json!({"model_type":"paradee", "sample_rate":24000, "vocab":{" ":16,"h":50,"ə":83,"l":54,"ˈ":156,"O":31,"a":43,".":4,"!":5,"?":6,";":1,":":2}});
        fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .unwrap();
        let frontend =
            frontend_fixture("import sys; sys.stdin.read(); sys.stdout.write(\"həlˈoʊ\")");
        let mut config = Config::default();
        crate::catalog::model(crate::catalog::PARADEE_MODEL_ID)
            .unwrap()
            .activate(&mut config);
        config.model.directory = directory.to_string_lossy().into_owned();
        config.backend.options.insert(
            "g2p_executable".into(),
            frontend.to_string_lossy().into_owned(),
        );
        let mut paths = AppPaths::discover();
        paths.config_file = directory.join("application.toml");
        Self {
            directory,
            frontend,
            config,
            paths,
        }
    }

    fn write_metadata(&self, value: serde_json::Value) {
        fs::write(
            self.directory.join("config.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
    }

    fn backend(&self, graph: FixtureGraph) -> Result<ParadeeOpenvinoBackend> {
        ParadeeOpenvinoBackend::create_with(&self.config, &self.paths, |path, config, paths| {
            assert_eq!(path, self.directory.join("paradee.onnx"));
            assert_eq!(config.backend.threads, self.config.backend.threads);
            assert_eq!(paths.config_file, self.paths.config_file);
            Ok(Box::new(graph))
        })
    }
}

impl Drop for ModelFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
        let _ = fs::remove_file(&self.frontend);
    }
}

type InferenceCalls = std::rc::Rc<RefCell<Vec<(Vec<i64>, f32)>>>;

struct FixtureGraph {
    calls: InferenceCalls,
    devices: String,
    shape: Vec<i64>,
    samples: Vec<f32>,
    fail: bool,
}

impl FixtureGraph {
    fn new() -> Self {
        Self {
            calls: Default::default(),
            devices: "CPU".into(),
            shape: vec![1, 2],
            samples: vec![0.25, -0.25],
            fail: false,
        }
    }
}

impl ParadeeGraph for FixtureGraph {
    fn execution_devices(&self) -> Result<String> {
        Ok(self.devices.clone())
    }
    fn infer(&mut self, ids: &[i64], speed: f32) -> Result<(Vec<i64>, Vec<f32>)> {
        self.calls.borrow_mut().push((ids.to_vec(), speed));
        ensure!(!self.fail, "fixture graph inference failed");
        Ok((self.shape.clone(), self.samples.clone()))
    }
}

#[test]
fn provider_stream_preserves_phoneme_contract_speed_pcm_and_cancellation() {
    let fixture = ModelFixture::new();
    let graph = FixtureGraph::new();
    let calls = graph.calls.clone();
    let backend = fixture.backend(graph).unwrap();
    assert_eq!(backend.kind(), "paradee-openvino");
    assert_eq!(backend.sample_rate(), 24000);
    assert_eq!(backend.num_voices(), 1);
    assert_eq!(
        backend.generate("Hello!", 0.75, 0).unwrap(),
        vec![0.25, -0.25]
    );
    assert_eq!(
        calls.borrow()[0],
        (vec![0, 50, 83, 54, 156, 31, 5, 0], 0.75)
    );
    calls.borrow_mut().clear();
    let text = "Hello world. ".repeat(60);
    let mut delivered = Vec::new();
    backend
        .generate_stream(&text, 1.25, 0, &mut |chunk| {
            delivered.extend_from_slice(chunk);
            Ok(())
        })
        .unwrap();
    let count = calls.borrow().len();
    assert!(count > 1);
    assert_eq!(delivered, [0.25, -0.25].repeat(count));
    assert!(
        calls
            .borrow()
            .iter()
            .all(|(ids, speed)| ids.first() == Some(&0)
                && ids.last() == Some(&0)
                && *speed == 1.25)
    );
    calls.borrow_mut().clear();
    let mut emitted = 0;
    let error = backend
        .generate_stream(&text, 1.0, 0, &mut |_| {
            emitted += 1;
            bail!("playback cancelled")
        })
        .unwrap_err();
    assert!(error.to_string().contains("playback cancelled"));
    assert_eq!(emitted, 1);
    assert_eq!(
        calls.borrow().len(),
        1,
        "cancellation must not synthesize later segments"
    );
}

#[test]
fn provider_rejects_invalid_requests_before_inference() {
    let fixture = ModelFixture::new();
    let graph = FixtureGraph::new();
    let calls = graph.calls.clone();
    let backend = fixture.backend(graph).unwrap();
    for (text, speed, voice) in [
        ("Hello", 1.0, 1),
        ("Hello", f32::NAN, 0),
        ("Hello", 0.49, 0),
        ("Hello", 2.01, 0),
        ("   ", 1.0, 0),
    ] {
        assert!(backend.generate(text, speed, voice).is_err());
    }
    assert!(backend.infer(&[0, 0], 1.0).is_err());
    assert!(backend.infer(&[0; 513], 1.0).is_err());
    assert!(calls.borrow().is_empty());
    assert!(backend.generate("Hello", 0.5, 0).is_ok());
    assert!(backend.generate("Hello", 2.0, 0).is_ok());
}

#[test]
fn provider_validates_native_audio_shape_finiteness_size_and_errors() {
    let fixture = ModelFixture::new();
    for (shape, samples) in [
        (vec![2, 2], vec![0.1, 0.2]),
        (vec![1], vec![0.1]),
        (vec![1, 3], vec![0.1, 0.2]),
        (vec![1, -1], vec![0.1]),
        (vec![1, 0], vec![]),
        (vec![1, 1], vec![f32::NAN]),
        (vec![1, 1], vec![f32::INFINITY]),
        (
            vec![1, (MAX_SAMPLES + 1) as i64],
            vec![0.0; MAX_SAMPLES + 1],
        ),
    ] {
        let mut graph = FixtureGraph::new();
        graph.shape = shape;
        graph.samples = samples;
        let backend = fixture.backend(graph).unwrap();
        let mut emitted = false;
        assert!(
            backend
                .generate_stream("Hello", 1.0, 0, &mut |_| {
                    emitted = true;
                    Ok(())
                })
                .is_err()
        );
        assert!(!emitted, "invalid native audio must never reach playback");
    }
    let mut graph = FixtureGraph::new();
    graph.fail = true;
    let error = fixture
        .backend(graph)
        .unwrap()
        .generate("Hello", 1.0, 0)
        .unwrap_err();
    assert!(error.to_string().contains("fixture graph inference failed"));
}

#[test]
fn provider_creation_checks_configuration_metadata_frontend_and_placement() {
    let fixture = ModelFixture::new();
    let reject = |config: &Config| {
        ParadeeOpenvinoBackend::create_with(config, &fixture.paths, |_, _, _| {
            panic!("invalid input must not compile native model")
        })
        .err()
        .unwrap()
    };
    let mut config = fixture.config.clone();
    config.backend.runtime = Runtime::Default;
    assert!(reject(&config).to_string().contains("runtime=openvino"));
    let mut config = fixture.config.clone();
    config.backend.device = "gpu".into();
    assert!(reject(&config).to_string().contains("only OpenVINO CPU"));
    let mut config = fixture.config.clone();
    config.model.family = "kokoro".into();
    assert!(reject(&config).to_string().contains("family=paradee"));
    let mut config = fixture.config.clone();
    config.model.language = "fr".into();
    assert!(reject(&config).to_string().contains("American English"));
    let mut config = fixture.config.clone();
    config.model.file = "missing.onnx".into();
    assert!(reject(&config).to_string().contains("missing"));
    let mut config = fixture.config.clone();
    config
        .backend
        .options
        .insert("g2p_executable".into(), "relative/espeak-ng".into());
    assert!(reject(&config).to_string().contains("absolute"));
    for metadata in [
        serde_json::json!({"model_type":"kokoro","sample_rate":24000,"vocab":{" ":16}}),
        serde_json::json!({"model_type":"paradee","sample_rate":44100,"vocab":{" ":16}}),
        serde_json::json!({"model_type":"paradee","sample_rate":24000,"vocab":{"":16}}),
        serde_json::json!({"model_type":"paradee","sample_rate":24000,"vocab":{"ab":16}}),
        serde_json::json!({"model_type":"paradee","sample_rate":24000,"vocab":{" ":178}}),
        serde_json::json!({"model_type":"paradee","sample_rate":24000,"vocab":{"a":43}}),
    ] {
        fixture.write_metadata(metadata);
        assert!(reject(&fixture.config).to_string().contains("Paradee"));
    }
    fs::write(fixture.directory.join("config.json"), "invalid JSON").unwrap();
    assert!(reject(&fixture.config).to_string().contains("vocabulary"));
    assert!(ParadeeOpenvinoBackend::create(&Config::default(), &fixture.paths).is_err());
    let fixture = ModelFixture::new();
    let mut graph = FixtureGraph::new();
    graph.devices = "GPU".into();
    assert!(
        fixture
            .backend(graph)
            .err()
            .unwrap()
            .to_string()
            .contains("unexpected execution devices")
    );
    let error = ParadeeOpenvinoBackend::create_with(&fixture.config, &fixture.paths, |_, _, _| {
        bail!("fixture compiler rejected graph")
    })
    .err()
    .unwrap();
    assert!(
        error
            .to_string()
            .contains("fixture compiler rejected graph")
    );
}

#[test]
fn provider_splits_expanded_phonemes_without_dropping_tokens() {
    let mut fixture = ModelFixture::new();
    let frontend = frontend_fixture("import sys; sys.stdin.read(); sys.stdout.write(\"a\"*1200)");
    fixture.config.backend.options.insert(
        "g2p_executable".into(),
        frontend.to_string_lossy().into_owned(),
    );
    let graph = FixtureGraph::new();
    let calls = graph.calls.clone();
    let backend = fixture.backend(graph).unwrap();
    backend.generate("One short word.", 1.0, 0).unwrap();
    assert_eq!(calls.borrow().len(), 3);
    assert!(calls.borrow().iter().all(|(ids, _)| ids.len() <= 512));
    assert_eq!(
        calls
            .borrow()
            .iter()
            .flat_map(|(ids, _)| ids.iter())
            .filter(|id| **id == 43)
            .count(),
        1200
    );
    assert_eq!(
        calls
            .borrow()
            .last()
            .unwrap()
            .0
            .iter()
            .filter(|id| **id == 4)
            .count(),
        1
    );
    fs::remove_file(frontend).unwrap();
}

#[test]
fn frontend_failures_are_bounded_and_actionable() {
    assert!(nonblocking(-1).is_err());
    assert!(
        phonemize(Path::new("/missing/omaspeak-espeak"), "hello")
            .unwrap_err()
            .to_string()
            .contains("start Paradee")
    );
    assert!(
        phonemize(
            Path::new("/missing/omaspeak-espeak"),
            &"a".repeat(MAX_FRONTEND_INPUT + 1)
        )
        .unwrap_err()
        .to_string()
        .contains("input exceeds")
    );
    for body in [
        "import sys; sys.stdin.read(); sys.exit(7)",
        "import sys; sys.stdin.read(); sys.stdout.buffer.write(bytes([255]))",
    ] {
        let frontend = frontend_fixture(body);
        let error = phonemize(&frontend, "hello").unwrap_err();
        assert!(
            error.to_string().contains("failed") || error.to_string().contains("UTF-8"),
            "{error:#}"
        );
        fs::remove_file(frontend).unwrap();
    }
}

#[test]
fn native_frontend_resolution_uses_exact_override_then_search_path() {
    let fixture = ModelFixture::new();
    assert_eq!(
        frontend_path_in(&fixture.config, None).unwrap(),
        fixture.frontend
    );
    let mut config = fixture.config.clone();
    config.backend.options.remove("g2p_executable");
    assert!(frontend_path_in(&config, None).is_err());
    fs::copy(&fixture.frontend, fixture.directory.join("espeak-ng")).unwrap();
    let search =
        std::env::join_paths([fixture.directory.join("missing"), fixture.directory.clone()])
            .unwrap();
    assert_eq!(
        frontend_path_in(&config, Some(&search)).unwrap(),
        fixture.directory.join("espeak-ng")
    );
    config.backend.options.insert(
        "g2p_executable".into(),
        fixture
            .directory
            .join("missing")
            .to_string_lossy()
            .into_owned(),
    );
    assert!(
        frontend_path_in(&config, Some(&search)).is_err(),
        "broken explicit path must not silently choose PATH binary"
    );
}

#[test]
fn native_compiler_resolution_errors_propagate_after_frontend_proof() {
    let mut fixture = ModelFixture::new();
    fixture.config.backend.openvino_library =
        Some(fixture.directory.join("missing-libopenvino.so"));
    fixture.config.backend.openvino_plugins = Some(fixture.directory.join("missing-plugins.xml"));
    let error = ParadeeOpenvinoBackend::create(&fixture.config, &fixture.paths)
        .err()
        .unwrap();
    assert!(format!("{error:#}").contains("OpenVINO"));
}

#[test]
fn misaki_strips_unattached_or_whitespace_syllable_marks() {
    assert_eq!(to_misaki("̩ ̩a"), " a");
}

#[test]
#[cfg(target_os = "linux")]
fn native_openvino_abi_maps_named_tensors_speed_and_cpu_placement() {
    const LIBRARY: &str = "OMASPEAK_TEST_PARADEE_NATIVE_ABI";
    if let Some(library) = std::env::var_os(LIBRARY) {
        let mut fixture = ModelFixture::new();
        let plugins = fixture.directory.join("plugins.xml");
        fs::write(&plugins, "test-only configuration").unwrap();
        fixture.config.backend.openvino_library = Some(library.into());
        fixture.config.backend.openvino_plugins = Some(plugins);
        let backend = ParadeeOpenvinoBackend::create(&fixture.config, &fixture.paths);
        if std::env::var("OMASPEAK_TEST_PARADEE_DEVICES").as_deref() == Ok("GPU") {
            assert!(
                backend
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("unexpected execution devices")
            );
        } else {
            let backend = backend.unwrap();
            for speed in [0.75, 1.25] {
                assert_eq!(
                    backend.generate("Hello!", speed, 0).unwrap(),
                    vec![speed, 0.5]
                );
            }
        }
        return;
    }
    let directory = ModelFixture::new();
    let library = directory.directory.join("libopenvino_c.so");
    let status = Command::new("cc")
        .args(["-shared", "-fPIC", "-O0", "-Wall", "-Werror"])
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/paradee_openvino_stub.c"))
        .arg("-o")
        .arg(&library)
        .status()
        .unwrap();
    assert!(status.success());
    for device in ["CPU", "GPU"] {
        let status = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "paradee::tests::native_openvino_abi_maps_named_tensors_speed_and_cpu_placement",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(LIBRARY, &library)
            .env("OMASPEAK_TEST_PARADEE_DEVICES", device)
            .status()
            .unwrap();
        assert!(status.success(), "native ABI contract failed for {device}");
    }
}

#[test]
fn english_currency_and_titles_are_normalized_without_guessing_other_forms() {
    for (input, expected) in [
        ("$1", "1 dollar"),
        ("$12.00", "12 dollars"),
        ("$0.01", "0 dollars and 1 cent"),
        ("$1,234.56", "1234 dollars and 56 cents"),
        ("$-3.50", "minus 3 dollars and 50 cents"),
        ("$12, please", "12 dollars, please"),
        ("$0.05", "0 dollars and 5 cents"),
        ("$18446744073709551616", "$18446744073709551616"),
        ("$1,23", "$1,23"),
        ("$1.2", "$1.2"),
        ("$1.234", "$1.234"),
        ("$12M", "$12M"),
        ("$", "$"),
        ("USD 12 and €12", "USD 12 and €12"),
        (
            "Mr. Jones, Mrs. Smith, Ms. Chen and Prof. Díaz",
            "Mister Jones, Missus Smith, Miz Chen and Professor Díaz",
        ),
        (
            "wordDr. Mr.Smith dr. Smith",
            "wordDr. Mr.Smith Doctor Smith",
        ),
    ] {
        assert_eq!(normalize_english(input), expected, "{input}");
    }
    assert!(usd_amount("").is_none());
    assert!(usd_amount("$1,,000").is_none());
}

#[test]
fn native_quality_corpus_has_exact_prosody_frontend_plans() {
    let corpus: serde_json::Value =
        serde_json::from_str(include_str!("../../benchmarks/paradee/corpus.json")).unwrap();
    for case in corpus["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let actual = speech_segments(text);
        assert!(!actual.is_empty(), "{}", case["id"]);
        assert!(actual.iter().all(|(text, _)| text.chars().count() <= 240));
        if let Some(expected) = case["expected_segments"].as_array() {
            let expected = expected
                .iter()
                .map(|segment| {
                    (
                        segment["text"].as_str().unwrap().to_owned(),
                        segment["punctuation"]
                            .as_str()
                            .and_then(|text| text.chars().next()),
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "{}", case["id"]);
        }
    }
}

#[test]
fn punctuation_survives_frontend_encoding_and_clauses_cancel_in_order() {
    let fixture = ModelFixture::new();
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(fixture.directory.join("config.json")).unwrap()).unwrap();
    metadata["vocab"][","] = serde_json::json!(3);
    fixture.write_metadata(metadata);
    let graph = FixtureGraph::new();
    let calls = graph.calls.clone();
    let backend = fixture.backend(graph).unwrap();
    backend.generate("Hello, really? Yes!", 1.0, 0).unwrap();
    assert_eq!(
        calls
            .borrow()
            .iter()
            .map(|(tokens, _)| tokens[tokens.len() - 2])
            .collect::<Vec<_>>(),
        vec![3, 6, 5]
    );
    calls.borrow_mut().clear();
    assert!(backend.generate("“” ... ;!", 1.0, 0).is_err());
    assert!(calls.borrow().is_empty());
}
