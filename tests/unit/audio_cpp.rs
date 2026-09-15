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
        WorkerRequest::Shutdown => panic!("wrong request variant"),
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
        find_provider_library(std::slice::from_ref(&root)).unwrap(),
        root.join("libaudiocpp.so.10").canonicalize().unwrap()
    );

    fs::write(root.join("libaudiocpp.so"), b"soname").unwrap();
    assert_eq!(
        find_provider_library(std::slice::from_ref(&root)).unwrap(),
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
        validate_audio_shape(&sample, 1, SUPERTONIC_SAMPLE_RATE, 1).unwrap(),
        1
    );
    assert!(validate_audio_shape(std::ptr::null(), 1, SUPERTONIC_SAMPLE_RATE, 1).is_err());
    assert!(validate_audio_shape(&sample, 0, SUPERTONIC_SAMPLE_RATE, 1).is_err());
    assert!(validate_audio_shape(&sample, 1, 16_000, 1).is_err());
    assert!(validate_audio_shape(&sample, 1, SUPERTONIC_SAMPLE_RATE, 2).is_err());
    assert!(validate_audio_shape(&sample, MAX_PCM_SAMPLES + 1, SUPERTONIC_SAMPLE_RATE, 1).is_err());
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
