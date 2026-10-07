use super::*;
use std::fs;

fn temp(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "omaspeak-audiocpp-test-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn validates_declared_audio_cpp_abi_contract() {
    validate_abi(0x0000_0100).unwrap();
    validate_abi(0x0000_0207).unwrap();
    assert!(validate_abi(0x0000_0009).is_err());
    assert!(validate_abi(0x0001_0000).is_err());
}

#[test]
fn loader_rejects_a_valid_non_audiocpp_library_without_executing_native_inference() {
    let library = [
        "/usr/lib/libm.so.6",
        "/usr/lib64/libm.so.6",
        "/lib/x86_64-linux-gnu/libm.so.6",
    ]
    .into_iter()
    .map(PathBuf::from)
    .find(|path| path.is_file());
    let Some(library) = library else {
        return;
    };
    let error = Api::load(&library).err().expect("libm is not audio.cpp");
    assert!(error.to_string().contains("audiocpp_abi_version"));
}

#[test]
fn maps_all_supertonic_speaker_ids() {
    let names = (0..10)
        .map(|voice| voice_name(voice).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]
    );
    assert!(voice_name(-1).is_err());
    assert!(voice_name(10).is_err());
}

#[test]
fn framed_control_protocol_round_trips_and_is_bounded() {
    let request = WorkerRequest::Generate {
        text: "Testing one two three".into(),
        speed: 1.25,
        voice: 7,
    };
    let mut frame = Vec::new();
    write_json_frame(&mut frame, &request).unwrap();
    let decoded: WorkerRequest = read_json_frame(frame.as_slice()).unwrap();
    match decoded {
        WorkerRequest::Generate { text, speed, voice } => {
            assert_eq!(text, "Testing one two three");
            assert_eq!(speed, 1.25);
            assert_eq!(voice, 7);
        }
        WorkerRequest::Shutdown | WorkerRequest::GenerateStream { .. } => {
            panic!("wrong request variant")
        }
    }

    let oversized = ((MAX_CONTROL_FRAME as u32) + 1).to_le_bytes();
    assert!(read_json_frame::<_, WorkerRequest>(oversized.as_slice()).is_err());
    let too_large = WorkerRequest::Generate {
        text: "x".repeat(MAX_CONTROL_FRAME),
        speed: 1.0,
        voice: 0,
    };
    assert!(write_json_frame(Vec::new(), &too_large).is_err());
}

#[test]
fn bounded_worker_reap_polling_reports_exit_and_timeout() {
    let mut polls = 0;
    assert!(
        poll_until(Instant::now() + Duration::from_secs(1), || {
            polls += 1;
            Ok(polls == 3)
        })
        .unwrap()
    );
    assert_eq!(polls, 3);

    let mut timeout_polls = 0;
    assert!(
        !poll_until(Instant::now(), || {
            timeout_polls += 1;
            Ok(false)
        })
        .unwrap()
    );
    assert_eq!(timeout_polls, 1);
}

#[test]
fn native_diagnostics_retain_a_bounded_sanitized_tail() {
    let mut diagnostics = BoundedDiagnostics::default();
    diagnostics.append(b"discarded prefix\n");
    diagnostics.append(&vec![b'x'; MAX_WORKER_STDERR]);
    diagnostics.append(b"\nuseful tail\x00\x07\n");
    assert!(diagnostics.bytes.len() <= MAX_WORKER_STDERR);
    let rendered = diagnostics.render().unwrap();
    assert!(rendered.starts_with("[earlier output truncated]"));
    assert!(rendered.ends_with("useful tail"));
    assert!(!rendered.contains('\0'));
    assert!(!rendered.contains('\u{7}'));

    assert_eq!(
        render_worker_diagnostics(None, None),
        "",
        "successful silent workers add no diagnostic noise"
    );
}

#[test]
fn worker_pipes_round_trip_without_socket_permissions() {
    let mut child = Command::new("cat")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = TimedWriter {
        writer: child.stdin.take().unwrap(),
        timeout: Duration::from_secs(1),
    };
    let mut output = TimedReader {
        reader: child.stdout.take().unwrap(),
        timeout: Duration::from_secs(1),
    };

    input.write_all(b"worker pipe").unwrap();
    input.flush().unwrap();
    let mut echoed = [0_u8; 11];
    output.read_exact(&mut echoed).unwrap();
    assert_eq!(&echoed, b"worker pipe");

    drop(input);
    drop(output);
    assert!(wait_for_exit(&mut child, Duration::from_secs(1)).unwrap());
}

#[test]
fn worker_pipe_deadline_times_out_and_child_exits_normally() {
    let mut child = Command::new("sleep")
        .arg("0.05")
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut output = TimedReader {
        reader: child.stdout.take().unwrap(),
        timeout: Duration::from_millis(1),
    };
    let error = output.read(&mut [0_u8; 1]).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    assert!(wait_for_exit(&mut child, Duration::from_secs(1)).unwrap());
    assert!(child.wait().unwrap().success());
}

#[test]
fn backend_options_require_a_scope_and_reserve_model_owned_request_values() {
    let mut options = BTreeMap::new();
    options.insert("load.config".into(), "small".into());
    options.insert("session.batch".into(), "1".into());
    options.insert("request.temperature".into(), "0.5".into());
    let parsed = scoped_options(&options).unwrap();
    assert_eq!(parsed.load["config"], "small");
    assert_eq!(parsed.session["batch"], "1");
    assert_eq!(parsed.request["temperature"], "0.5");

    for invalid in [
        "unscoped",
        "load.",
        "unknown.value",
        "request.language",
        "request.num_inference_steps",
    ] {
        let options = BTreeMap::from([(invalid.into(), "value".into())]);
        assert!(scoped_options(&options).is_err(), "accepted {invalid}");
    }
}

#[test]
fn provider_discovery_prefers_soname_then_highest_numeric_version() {
    let root = temp("provider-versions");
    fs::write(root.join("libaudiocpp.so.9"), b"nine").unwrap();
    fs::write(root.join("libaudiocpp.so.10"), b"ten").unwrap();
    fs::write(root.join("libaudiocpp.so.preview"), b"invalid").unwrap();
    assert_eq!(
        crate::runtime::find_versioned_library(std::slice::from_ref(&root), "libaudiocpp.so")
            .unwrap(),
        root.join("libaudiocpp.so.10").canonicalize().unwrap()
    );

    fs::write(root.join("libaudiocpp.so"), b"soname").unwrap();
    assert_eq!(
        crate::runtime::find_versioned_library(std::slice::from_ref(&root), "libaudiocpp.so")
            .unwrap(),
        root.join("libaudiocpp.so").canonicalize().unwrap()
    );
}

#[test]
fn pcm_protocol_round_trips_finite_samples_and_rejects_bad_shapes() {
    let expected = [-1.0, -0.25, 0.0, 0.5, 1.0];
    let mut bytes = Vec::new();
    write_pcm(&mut bytes, &expected).unwrap();
    assert_eq!(
        read_pcm(bytes.as_slice(), expected.len()).unwrap(),
        expected
    );
    assert!(write_pcm(Vec::new(), &[]).is_err());
    assert!(write_pcm(Vec::new(), &[f32::NAN]).is_err());
    assert!(read_pcm([].as_slice(), 0).is_err());
    assert!(read_pcm([].as_slice(), MAX_PCM_SAMPLES + 1).is_err());
    assert!(read_pcm(f32::NAN.to_le_bytes().as_slice(), 1).is_err());

    let sample = 0.0f32;
    assert_eq!(
        validate_audio_shape("supertonic", &sample, 1, SUPERTONIC_SAMPLE_RATE, 1).unwrap(),
        1
    );
    assert!(
        validate_audio_shape("supertonic", std::ptr::null(), 1, SUPERTONIC_SAMPLE_RATE, 1).is_err()
    );
    assert!(validate_audio_shape("supertonic", &sample, 0, SUPERTONIC_SAMPLE_RATE, 1).is_err());
    assert!(validate_audio_shape("supertonic", &sample, 1, 16_000, 1).is_err());
    assert!(validate_audio_shape("supertonic", &sample, 1, SUPERTONIC_SAMPLE_RATE, 2).is_err());
    assert!(
        validate_audio_shape(
            "supertonic",
            &sample,
            MAX_PCM_SAMPLES + 1,
            SUPERTONIC_SAMPLE_RATE,
            1
        )
        .is_err()
    );
    // Kokoro expects 24 kHz mono; its rate is valid and the Supertonic rate is not.
    assert_eq!(
        validate_audio_shape("kokoro", &sample, 1, KOKORO_SAMPLE_RATE, 1).unwrap(),
        1
    );
    assert!(validate_audio_shape("kokoro", &sample, 1, SUPERTONIC_SAMPLE_RATE, 1).is_err());
    assert!(validate_audio_shape("unknown", &sample, 1, SUPERTONIC_SAMPLE_RATE, 1).is_err());
}

#[test]
fn backend_creation_rejects_invalid_shapes_before_native_launch() {
    let root = temp("create-validation");
    let paths = AppPaths {
        config_file: root.join("config.toml"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    };
    let mut config = Config::default();
    config.model.family = "other".into();
    assert!(AudioCppBackend::create(&config, &paths, Runtime::Default).is_err());

    config.model.family = "supertonic".into();
    config.model.steps = 0;
    assert!(AudioCppBackend::create(&config, &paths, Runtime::Default).is_err());
    config.model.steps = 8;
    assert!(AudioCppBackend::create(&config, &paths, Runtime::Openvino).is_err());
    for runtime in [Runtime::Cuda, Runtime::Vulkan, Runtime::Hip] {
        assert!(AudioCppBackend::create(&config, &paths, runtime).is_err());
    }
}

#[test]
fn native_paths_are_explicit_existing_files_and_relative_models_stay_contained() {
    let root = temp("paths");
    let config_dir = root.join("config");
    let dependency_dir = config_dir.join("dependencies");
    let outside_dependency_dir = root.join("outside-dependencies");
    let model_dir = root.join("models/supertonic");
    fs::create_dir_all(&dependency_dir).unwrap();
    fs::create_dir_all(&outside_dependency_dir).unwrap();
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(config_dir.join("libaudiocpp.so"), b"library").unwrap();
    fs::write(model_dir.join("model.gguf"), b"model").unwrap();
    fs::write(root.join("outside.gguf"), b"outside").unwrap();

    let paths = AppPaths {
        config_file: config_dir.join("config.toml"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    };
    let mut config = Config::default();
    config.backend.library = Some("libaudiocpp.so".into());
    config.backend.library_dirs = vec!["dependencies".into()];
    config.model.directory = model_dir.display().to_string();
    config.model.file = "model.gguf".into();
    assert_eq!(
        resolve_provider_library(&config, &paths.config_file).unwrap(),
        config_dir.join("libaudiocpp.so").canonicalize().unwrap()
    );
    assert_eq!(
        resolve_library_dirs(
            &config,
            &paths.config_file,
            &config_dir.join("libaudiocpp.so").canonicalize().unwrap()
        )
        .unwrap(),
        [
            dependency_dir.canonicalize().unwrap(),
            config_dir.canonicalize().unwrap()
        ]
    );
    let environment_dependency = root.join("environment-dependencies");
    fs::create_dir_all(&environment_dependency).unwrap();
    assert_eq!(
        resolve_library_dirs_with(
            &config,
            &paths.config_file,
            &config_dir.join("libaudiocpp.so").canonicalize().unwrap(),
            std::slice::from_ref(&environment_dependency)
        )
        .unwrap(),
        [
            dependency_dir.canonicalize().unwrap(),
            environment_dependency.canonicalize().unwrap(),
            config_dir.canonicalize().unwrap()
        ]
    );
    assert_eq!(
        resolve_model_file(&config, &paths).unwrap(),
        model_dir.join("model.gguf").canonicalize().unwrap()
    );

    config.model.file = "../../outside.gguf".into();
    let error = resolve_model_file(&config, &paths).unwrap_err();
    assert!(error.to_string().contains("escapes its base directory"));
    config.model.file = "missing.gguf".into();
    assert!(resolve_model_file(&config, &paths).is_err());
    config.model.file.clear();
    assert!(resolve_model_file(&config, &paths).is_err());
    config.backend.library_dirs = vec!["../outside-dependencies".into()];
    assert!(
        resolve_library_dirs(
            &config,
            &paths.config_file,
            &config_dir.join("libaudiocpp.so").canonicalize().unwrap()
        )
        .unwrap_err()
        .to_string()
        .contains("escapes the config directory")
    );
    config.backend.library = None;
    assert!(resolve_provider_library(&config, &paths.config_file).is_err());
}

#[test]
fn kokoro_engine_voice_ids_and_request_languages_cover_every_prefix() {
    assert_eq!(engine_voice_id("kokoro", 0).unwrap(), "af_alloy");
    assert_eq!(engine_voice_id("kokoro", 3).unwrap(), "af_heart");
    assert_eq!(engine_voice_id("kokoro", 16).unwrap(), "am_michael");
    assert_eq!(engine_voice_id("kokoro", 53).unwrap(), "zm_yunyang");
    assert_eq!(
        engine_voice_id("kokoro", 54).unwrap_err().to_string(),
        "voice 54 is outside the Kokoro voice range 0..53"
    );
    assert_eq!(
        engine_voice_id("kokoro", -1).unwrap_err().to_string(),
        "voice -1 is outside the Kokoro voice range 0..53"
    );
    assert!(engine_voice_id("unknown", 0).is_err());
    for (prefix, language) in [
        ('a', "en-us"),
        ('b', "en-gb"),
        ('e', "es"),
        ('f', "fr"),
        ('h', "hi"),
        ('i', "it"),
        ('j', "ja"),
        ('p', "pt-br"),
        ('z', "zh"),
    ] {
        assert_eq!(
            kokoro_voice_language(&format!("{prefix}x_voice")),
            Some(language)
        );
    }
    assert_eq!(kokoro_voice_language("q_unknown"), None);
    assert_eq!(kokoro_voice_language(""), None);
}

#[test]
fn native_stream_contract_handles_cancellation_and_provider_failures() {
    let root = temp("native-stream-contract");
    let library = root.join("provider.so");
    assert!(
        std::process::Command::new("cc")
            .args([
                "-shared",
                "-fPIC",
                "-DOMASPEAK_TEST_STREAMING",
                "tests/fixtures/audiocpp_stub.c",
                "-o"
            ])
            .arg(&library)
            .status()
            .unwrap()
            .success()
    );
    let mut spec = WorkerSpec {
        library,
        library_dirs: vec![],
        model: root.join("valid"),
        backend: "cpu".into(),
        family: "supertonic".into(),
        device: 0,
        threads: 1,
        language: "en".into(),
        steps: 4,
        load_options: Default::default(),
        session_options: Default::default(),
        request_options: Default::default(),
    };
    let mut native = NativeEngine::load(&spec).unwrap();
    let mut pcm = Vec::new();
    let audio = native
        .generate_with_sink(
            "hello",
            1.0,
            0,
            Some(&mut |samples| {
                pcm.extend_from_slice(samples);
                Ok(())
            }),
        )
        .unwrap();
    assert!(!pcm.is_empty());
    assert!(audio.pcm.is_empty());
    assert_eq!(
        pcm.len(),
        native.generate("hello", 1.0, 0).unwrap().pcm.len()
    );
    assert_eq!(audio.sample_rate, SUPERTONIC_SAMPLE_RATE);
    let mut calls = 0;
    assert!(
        native
            .generate_with_sink(
                "hello",
                1.0,
                0,
                Some(&mut |_| {
                    calls += 1;
                    bail!("cancel")
                })
            )
            .is_err()
    );
    assert_eq!(calls, 1);
    assert!(
        native
            .generate_with_sink("stream-fail", 1.0, 0, Some(&mut |_| Ok(())))
            .is_err()
    );
    assert!(native.generate("hello", 1.0, 0).is_ok());
    assert!(native.generate("bad\0text", 1.0, 0).is_err());
    assert!(native.generate("hello", 1.0, -1).is_err());
    drop(native);
    for mode in [
        "null-request",
        "fail-voice",
        "fail-steps",
        "null-result",
        "fail-result",
        "null-samples",
        "huge-audio",
    ] {
        spec.model = root.join(mode);
        let mut native = NativeEngine::load(&spec).unwrap();
        assert!(
            native
                .generate_with_sink("hello", 1.0, 0, Some(&mut |_| Ok(())))
                .is_err(),
            "{mode}"
        );
    }
    spec.model = root.join("valid");
    spec.request_options.insert("extra".into(), "value".into());
    let mut native = NativeEngine::load(&spec).unwrap();
    assert!(native.generate("fail-text", 1.0, 0).is_err());
    drop(native);
    for mode in [
        "fail-load",
        "null-model",
        "unsupported",
        "fail-session",
        "null-session",
    ] {
        spec.model = root.join(mode);
        assert!(NativeEngine::load(&spec).is_err(), "{mode}");
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn native_worker_protocol_orders_chunks_reports_errors_and_shuts_down() {
    let root = temp("worker-stream-protocol");
    for streaming in [false, true] {
        let library = root.join(format!("provider-{streaming}.so"));
        let mut compiler = std::process::Command::new("cc");
        compiler.args(["-shared", "-fPIC"]);
        if streaming {
            compiler.arg("-DOMASPEAK_TEST_STREAMING");
        }
        assert!(
            compiler
                .arg("tests/fixtures/audiocpp_stub.c")
                .arg("-o")
                .arg(&library)
                .status()
                .unwrap()
                .success()
        );
        let spec = WorkerSpec {
            library,
            library_dirs: vec![],
            model: root.join("valid"),
            backend: "cpu".into(),
            family: "supertonic".into(),
            device: 0,
            threads: 1,
            language: "en".into(),
            steps: 4,
            load_options: Default::default(),
            session_options: Default::default(),
            request_options: Default::default(),
        };
        let mut input = Vec::new();
        for request in [
            WorkerRequest::GenerateStream {
                text: "hello".into(),
                speed: 1.0,
                voice: 0,
            },
            WorkerRequest::Generate {
                text: "hello".into(),
                speed: 1.0,
                voice: 0,
            },
            WorkerRequest::GenerateStream {
                text: "fail-text".into(),
                speed: 1.0,
                voice: 0,
            },
            WorkerRequest::Generate {
                text: "fail-text".into(),
                speed: 1.0,
                voice: 0,
            },
            WorkerRequest::Shutdown,
        ] {
            write_json_frame(&mut input, &request).unwrap();
        }
        let mut output = Vec::new();
        run_worker_io(
            &serde_json::to_string(&spec).unwrap(),
            input.as_slice(),
            &mut output,
        )
        .unwrap();
        let mut frames = output.as_slice();
        assert!(matches!(
            read_json_frame::<_, WorkerResponse>(&mut frames).unwrap(),
            WorkerResponse::Ready { .. }
        ));
        let mut chunks = 0;
        loop {
            match read_json_frame::<_, WorkerResponse>(&mut frames).unwrap() {
                WorkerResponse::AudioChunk {
                    sample_rate,
                    samples,
                } => {
                    assert_eq!(sample_rate, SUPERTONIC_SAMPLE_RATE);
                    assert!(!read_pcm(&mut frames, samples).unwrap().is_empty());
                    chunks += 1;
                }
                WorkerResponse::StreamEnd => break,
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(chunks, if streaming { 2 } else { 1 });
        match read_json_frame::<_, WorkerResponse>(&mut frames).unwrap() {
            WorkerResponse::Audio { samples, .. } => {
                read_pcm(&mut frames, samples).unwrap();
            }
            other => panic!("unexpected {other:?}"),
        }
        for _ in 0..2 {
            assert!(matches!(
                read_json_frame::<_, WorkerResponse>(&mut frames).unwrap(),
                WorkerResponse::Error { .. }
            ));
        }
        assert!(matches!(
            read_json_frame::<_, WorkerResponse>(&mut frames).unwrap(),
            WorkerResponse::Shutdown
        ));
        assert!(frames.is_empty());
    }
    assert!(run_worker_io("invalid", &b""[..], Vec::new()).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn streaming_client_validates_framing_and_stops_after_sink_error() {
    let root = temp("client-stream-framing");
    for case in ["valid", "rate", "empty", "error", "unexpected", "cancel"] {
        let mut bytes = Vec::new();
        let rate = if case == "rate" {
            16000
        } else {
            SUPERTONIC_SAMPLE_RATE
        };
        match case {
            "empty" => write_json_frame(&mut bytes, &WorkerResponse::StreamEnd).unwrap(),
            "error" => write_json_frame(
                &mut bytes,
                &WorkerResponse::Error {
                    message: "fixture failure".into(),
                },
            )
            .unwrap(),
            "unexpected" => write_json_frame(&mut bytes, &WorkerResponse::Shutdown).unwrap(),
            _ => {
                write_json_frame(
                    &mut bytes,
                    &WorkerResponse::AudioChunk {
                        sample_rate: rate,
                        samples: 2,
                    },
                )
                .unwrap();
                write_pcm(&mut bytes, &[0.25, -0.25]).unwrap();
                write_json_frame(&mut bytes, &WorkerResponse::StreamEnd).unwrap();
            }
        }
        let path = root.join(case);
        fs::write(&path, bytes).unwrap();
        let mut child = Command::new("sh")
            .args(["-c", "cat \"$1\"; cat >/dev/null", "fixture"])
            .arg(&path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let client = WorkerClient {
            input: Some(TimedWriter {
                writer: child.stdin.take().unwrap(),
                timeout: Duration::from_secs(1),
            }),
            output: Some(TimedReader {
                reader: child.stdout.take().unwrap(),
                timeout: Duration::from_secs(1),
            }),
            diagnostics: WorkerDiagnostics::capture(child.stderr.take().unwrap()),
            child,
            stopped: false,
            expected_rate: SUPERTONIC_SAMPLE_RATE,
        };
        let mut pcm = Vec::new();
        let backend = AudioCppBackend {
            spec: WorkerSpec {
                library: root.join("unused"),
                library_dirs: vec![],
                model: root.join("valid"),
                backend: "cpu".into(),
                family: "supertonic".into(),
                device: 0,
                threads: 1,
                language: "en".into(),
                steps: 4,
                load_options: Default::default(),
                session_options: Default::default(),
                request_options: Default::default(),
            },
            worker: Mutex::new(client),
        };
        let result = backend.generate_stream("hello", 1.0, 0, &mut |chunk| {
            if case == "cancel" {
                bail!("cancelled")
            }
            pcm.extend_from_slice(chunk);
            Ok(())
        });
        if case == "valid" {
            result.unwrap();
            assert_eq!(pcm, vec![0.25, -0.25]);
        } else {
            assert!(result.is_err(), "{case}");
        }
        if case != "valid" {
            assert!(backend.worker.lock().unwrap().stopped);
        }
        drop(backend);
    }
    fs::remove_dir_all(root).unwrap();
}
