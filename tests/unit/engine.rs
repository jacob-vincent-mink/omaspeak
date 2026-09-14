use super::*;
use std::fs;

struct FakeBackend {
    voices: i32,
    samples: std::result::Result<Vec<f32>, &'static str>,
}

impl TtsBackend for FakeBackend {
    fn kind(&self) -> &'static str {
        "fake"
    }

    fn sample_rate(&self) -> i32 {
        16_000
    }

    fn num_voices(&self) -> i32 {
        self.voices
    }

    fn generate(&self, _: &str, _: f32, _: i32) -> Result<Vec<f32>> {
        self.samples.clone().map_err(|error| anyhow::anyhow!(error))
    }
}

fn temp(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "omaspeak-engine-test-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn engine(samples: std::result::Result<Vec<f32>, &'static str>, voices: i32) -> Engine {
    Engine {
        backend: Box::new(FakeBackend { voices, samples }),
        backend_kind: "fake",
        model_name: "test".into(),
        sample_rate: 16_000,
        load_time: Duration::default(),
        effective_runtime: Runtime::Default,
        fallback_used: false,
    }
}

fn paths(root: &Path) -> AppPaths {
    AppPaths {
        config_file: root.join("config.toml"),
        data_dir: root.join("data"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    }
}

fn install_model_assets(config: &Config, paths: &AppPaths) -> PathBuf {
    let directory = config.model_directory(paths);
    fs::create_dir_all(directory.join(&config.model.data_directory)).unwrap();
    fs::write(directory.join(&config.model.model_file), b"fixture").unwrap();
    fs::write(directory.join(&config.model.tokens_file), b"fixture").unwrap();
    directory
}

fn configure_supertonic(config: &mut Config, paths: &AppPaths) -> PathBuf {
    config.model.family = "supertonic".into();
    config.model.name = "supertonic-fixture".into();
    config.model.duration_predictor = "duration_predictor.int8.onnx".into();
    config.model.text_encoder = "text_encoder.int8.onnx".into();
    config.model.vector_estimator = "vector_estimator.int8.onnx".into();
    config.model.vocoder = "vocoder.int8.onnx".into();
    config.model.tts_json = "tts.json".into();
    config.model.unicode_indexer = "unicode_indexer.bin".into();
    config.model.voice_style = "voice.bin".into();
    config.model.language = "fr".into();
    config.model.steps = 8;
    let directory = config.model_directory(paths);
    fs::create_dir_all(&directory).unwrap();
    for asset in [
        &config.model.duration_predictor,
        &config.model.text_encoder,
        &config.model.vector_estimator,
        &config.model.vocoder,
        &config.model.tts_json,
        &config.model.unicode_indexer,
        &config.model.voice_style,
    ] {
        fs::write(directory.join(asset), b"fixture").unwrap();
    }
    directory
}

#[test]
fn fake_backend_synthesizes_a_valid_clamped_wav() {
    let root = temp("success");
    let output = root.join("nested/out.wav");
    let result = engine(Ok(vec![-2.0, -0.5, 0.5, 2.0]), 2)
        .synthesize("hello", 1.0, 1, &output)
        .unwrap();
    assert_eq!(result.output, output);
    assert_eq!(result.sample_rate, 16_000);
    assert_eq!(result.samples, 4);
    assert!(result.synthesis_time <= Duration::from_secs(1));
    let reader = hound::WavReader::open(result.output).unwrap();
    assert_eq!(reader.spec().sample_rate, 16_000);
    assert_eq!(reader.len(), 4);
}

#[test]
fn synthesis_validates_inputs_and_backend_output() {
    let output = temp("validation").join("out.wav");
    let valid = engine(Ok(vec![0.0]), 1);
    assert!(valid.synthesize(" ", 1.0, 0, &output).is_err());
    assert!(valid.synthesize("x", 0.1, 0, &output).is_err());
    assert!(valid.synthesize("x", f32::NAN, 0, &output).is_err());
    assert!(valid.synthesize("x", 1.0, -1, &output).is_err());
    assert!(valid.synthesize("x", 1.0, 1, &output).is_err());
    assert!(
        engine(Err("generation failed"), 1)
            .synthesize("x", 1.0, 0, &output)
            .is_err()
    );
    assert!(
        engine(Ok(vec![]), 1)
            .synthesize("x", 1.0, 0, &output)
            .is_err()
    );
    assert!(
        engine(Ok(vec![f32::NAN]), 1)
            .synthesize("x", 1.0, 0, &output)
            .is_err()
    );
    assert!(save_wav(&output, -1, &[0.0]).is_err());
}

#[test]
fn engine_load_reports_shape_capability_backend_and_model_errors() {
    let root = temp("load-errors");
    let paths = paths(&root);
    let mut config = Config::default();
    config.backend.threads = 0;
    assert!(Engine::load(&config, &paths).is_err());

    config.backend.threads = 2;
    config.backend.runtime = Runtime::Cuda;
    config.backend.device = "gpu".into();
    assert!(Engine::load(&config, &paths).is_err());
    config.backend.fallback = Fallback::Cpu;
    config.backend.kind = "other".into();
    assert!(Engine::load(&config, &paths).is_err());

    config.backend.runtime = Runtime::Default;
    config.backend.device = "auto".into();
    config.backend.kind = "sherpa-onnx".into();
    config.model.family = "unknown".into();
    assert!(Engine::load(&config, &paths).is_err());
    config.model.family = "piper".into();
    assert!(Engine::load(&config, &paths).is_err());
    install_model_assets(&config, &paths);
    assert!(build_sherpa_config(&config, &paths, Runtime::Openvino).is_ok());
}

#[test]
fn sherpa_config_is_built_without_loading_untrusted_native_models() {
    let root = temp("sherpa-config");
    let paths = paths(&root);
    let mut config = Config::default();
    let directory = install_model_assets(&config, &paths);

    let cpu = build_sherpa_config(&config, &paths, Runtime::Default).unwrap();
    assert_eq!(cpu.model.provider.as_deref(), Some("cpu"));
    assert_eq!(cpu.model.num_threads, 2);
    assert_eq!(
        cpu.model.vits.model.as_deref(),
        Some(directory.join(&config.model.model_file).to_str().unwrap())
    );

    let cuda = build_sherpa_config(&config, &paths, Runtime::Cuda).unwrap();
    assert_eq!(cuda.model.provider.as_deref(), Some("cuda"));

    config.model.family = "unknown".into();
    assert!(build_sherpa_config(&config, &paths, Runtime::Default).is_err());
    config.model.family = "piper".into();
    fs::remove_file(directory.join(&config.model.model_file)).unwrap();
    assert!(build_sherpa_config(&config, &paths, Runtime::Default).is_err());
    assert!(build_sherpa_config(&config, &paths, Runtime::Openvino).is_err());
}

#[test]
fn supertonic_config_maps_all_assets_and_generation_settings() {
    let root = temp("supertonic-config");
    let paths = paths(&root);
    let mut config = Config::default();
    let directory = configure_supertonic(&mut config, &paths);

    let native = build_sherpa_config(&config, &paths, Runtime::Default).unwrap();
    let supertonic = native.model.supertonic;
    assert_eq!(
        supertonic.duration_predictor.as_deref(),
        directory.join("duration_predictor.int8.onnx").to_str()
    );
    assert_eq!(
        supertonic.text_encoder.as_deref(),
        directory.join("text_encoder.int8.onnx").to_str()
    );
    assert_eq!(
        supertonic.vector_estimator.as_deref(),
        directory.join("vector_estimator.int8.onnx").to_str()
    );
    assert_eq!(
        supertonic.vocoder.as_deref(),
        directory.join("vocoder.int8.onnx").to_str()
    );
    assert_eq!(
        supertonic.tts_json.as_deref(),
        directory.join("tts.json").to_str()
    );
    assert_eq!(
        supertonic.unicode_indexer.as_deref(),
        directory.join("unicode_indexer.bin").to_str()
    );
    assert_eq!(
        supertonic.voice_style.as_deref(),
        directory.join("voice.bin").to_str()
    );
    assert!(native.model.vits.model.is_none());

    let generation = sherpa_generation_settings(&config).unwrap();
    assert_eq!(generation.language.as_deref(), Some("fr"));
    assert_eq!(generation.num_steps, Some(8));

    config.model.language = "xx".into();
    assert!(build_sherpa_config(&config, &paths, Runtime::Default).is_err());
    config.model.language = "en".into();
    config.model.steps = 0;
    assert!(build_sherpa_config(&config, &paths, Runtime::Default).is_err());
    config.model.steps = 5;
    fs::remove_file(directory.join("vocoder.int8.onnx")).unwrap();
    assert!(build_sherpa_config(&config, &paths, Runtime::Default).is_err());
}

#[test]
fn supertonic_npu_defaults_to_the_validated_mixed_component_allowlist() {
    let root = temp("supertonic-npu-components");
    let paths = paths(&root);
    let mut config = Config::default();
    configure_supertonic(&mut config, &paths);
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "npu".into();

    let native = build_sherpa_config(&config, &paths, Runtime::Openvino).unwrap();
    let provider = native.model.provider.unwrap();
    let provider_path = PathBuf::from(provider.strip_prefix("openvino:").unwrap());
    let contents = fs::read_to_string(&provider_path).unwrap();
    assert!(
        contents
            .contains("SherpaOnnx.SupertonicComponents=duration_predictor,text_encoder,vocoder\n")
    );

    config.backend.options.insert(
        "SherpaOnnx.SupertonicComponents".into(),
        "duration_predictor".into(),
    );
    build_sherpa_config(&config, &paths, Runtime::Openvino).unwrap();
    let contents = fs::read_to_string(provider_path).unwrap();
    assert!(contents.contains("SherpaOnnx.SupertonicComponents=duration_predictor\n"));
}

#[test]
fn supertonic_gpu_defaults_to_the_validated_vector_estimator_placement() {
    let root = temp("supertonic-gpu-components");
    let paths = paths(&root);
    let mut config = Config::default();
    configure_supertonic(&mut config, &paths);
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "gpu".into();
    let native = build_sherpa_config(&config, &paths, Runtime::Openvino).unwrap();
    let provider = native.model.provider.unwrap();
    let provider_path = PathBuf::from(provider.strip_prefix("openvino:").unwrap());
    let contents = fs::read_to_string(provider_path).unwrap();
    assert!(contents.contains("SherpaOnnx.SupertonicComponents=vector_estimator\n"));
    assert!(contents.contains("disable_dynamic_shapes=True\n"));
    assert!(contents.contains("enable_qdq_optimizer=False\n"));
    assert!(contents.contains("precision=FP32\n"));
}

#[test]
fn generates_a_private_openvino_provider_config_with_npu_defaults_and_options() {
    let root = temp("openvino-generated");
    let paths = paths(&root);
    let mut config = Config::default();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "nPu".into();
    config.backend.options.insert(
        "ProfilingFilePrefix".into(),
        root.join("profile").display().to_string(),
    );
    config.backend.options.insert(
        "load_config".into(),
        r#"{"NPU":{"NPU_PLATFORM":"5010"}}"#.into(),
    );
    config
        .backend
        .options
        .insert("device_type".into(), "NPU".into());
    install_model_assets(&config, &paths);

    let native = build_sherpa_config(&config, &paths, Runtime::Openvino).unwrap();
    let provider = native.model.provider.unwrap();
    let provider_path = PathBuf::from(provider.strip_prefix("openvino:").unwrap());
    assert!(provider_path.is_absolute());
    assert_eq!(
        provider_path,
        paths.state_dir.join("cache/openvino/npu/provider.config")
    );
    let contents = fs::read_to_string(&provider_path).unwrap();
    assert!(contents.contains("device_type=NPU\n"));
    assert!(contents.contains("enable_qdq_optimizer=True\n"));
    assert!(contents.contains("disable_dynamic_shapes=True\n"));
    assert!(contents.contains(&format!(
        "cache_dir={}\n",
        paths.state_dir.join("cache/openvino/npu/compiled").display()
    )));
    assert!(contents.contains(&format!(
        "ProfilingFilePrefix={}\n",
        root.join("profile").display()
    )));
    assert!(contents.contains("load_config={\"NPU\":{\"NPU_PLATFORM\":\"5010\"}}\n"));
    assert!(paths.state_dir.join("cache/openvino/npu/compiled").is_dir());
    assert_eq!(
        fs::read_dir(provider_path.parent().unwrap())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
            .count(),
        0
    );

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(&provider_path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            fs::metadata(provider_path.parent().unwrap())
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
    }

    config
        .backend
        .options
        .insert("disable_dynamic_shapes".into(), "False".into());
    let overridden = build_sherpa_config(&config, &paths, Runtime::Openvino)
        .unwrap()
        .model
        .provider
        .unwrap();
    assert!(
        fs::read_to_string(overridden.strip_prefix("openvino:").unwrap())
            .unwrap()
            .contains("disable_dynamic_shapes=False\n")
    );
}

#[test]
fn separates_openvino_device_caches_and_only_enables_qdq_for_npu() {
    let root = temp("openvino-gpu");
    let paths = paths(&root);
    let mut config = Config::default();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "auto:GPU, CPU".into();
    install_model_assets(&config, &paths);

    let provider = build_sherpa_config(&config, &paths, Runtime::Openvino)
        .unwrap()
        .model
        .provider
        .unwrap();
    assert!(provider.ends_with("/cache/openvino/auto-gpu-cpu/provider.config"));
    let contents = fs::read_to_string(provider.strip_prefix("openvino:").unwrap()).unwrap();
    assert!(contents.contains("device_type=AUTO:GPU,CPU\n"));
    assert!(contents.contains("/cache/openvino/auto-gpu-cpu/compiled\n"));
    assert!(!contents.contains("enable_qdq_optimizer"));
}

#[test]
fn honors_an_existing_relative_provider_config_without_rewriting_it() {
    let root = temp("openvino-supplied");
    let paths = paths(&root);
    let mut config = Config::default();
    let supplied = root.join("custom.config");
    fs::write(&supplied, "device_type=GPU\n").unwrap();
    config.backend.provider_config = "custom.config".into();
    install_model_assets(&config, &paths);

    let provider = build_sherpa_config(&config, &paths, Runtime::Openvino)
        .unwrap()
        .model
        .provider
        .unwrap();
    assert_eq!(provider, format!("openvino:{}", supplied.display()));
    assert_eq!(fs::read_to_string(&supplied).unwrap(), "device_type=GPU\n");
    assert!(!paths.state_dir.exists());

    config.backend.provider_config = "missing.config".into();
    assert!(build_sherpa_config(&config, &paths, Runtime::Openvino).is_err());
}

#[test]
fn rejects_unsafe_or_conflicting_generated_provider_options() {
    let root = temp("openvino-invalid-options");
    let paths = paths(&root);
    let mut config = Config::default();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "npu".into();
    install_model_assets(&config, &paths);

    config
        .backend
        .options
        .insert("bad key".into(), "value".into());
    assert!(build_sherpa_config(&config, &paths, Runtime::Openvino).is_err());
    config.backend.options.clear();
    config
        .backend
        .options
        .insert("ProfilingFilePrefix".into(), "one\ntwo".into());
    assert!(build_sherpa_config(&config, &paths, Runtime::Openvino).is_err());
    config.backend.options.clear();
    config
        .backend
        .options
        .insert("device_type".into(), "GPU".into());
    assert!(build_sherpa_config(&config, &paths, Runtime::Openvino).is_err());
    config.backend.options.clear();
    config
        .backend
        .options
        .insert("cache_dir".into(), "/tmp/shared".into());
    assert!(build_sherpa_config(&config, &paths, Runtime::Openvino).is_err());
}

#[test]
fn injected_load_constructs_engine_and_exercises_cpu_fallback() {
    let root = temp("load-success");
    let paths = paths(&root);
    let mut config = Config::default();
    config.model.name = "fixture".into();

    let loaded = Engine::load_with(&config, &paths, |_, _, runtime| {
        assert_eq!(runtime, Runtime::Default);
        Ok(Box::new(FakeBackend {
            voices: 3,
            samples: Ok(vec![0.0]),
        }))
    })
    .unwrap();
    assert_eq!(loaded.backend_kind, "fake");
    assert_eq!(loaded.model_name, "fixture");
    assert_eq!(loaded.sample_rate, 16_000);
    assert_eq!(loaded.effective_runtime, Runtime::Default);
    assert!(!loaded.fallback_used);
    assert!(loaded.load_time <= Duration::from_secs(1));

    config.backend.runtime = Runtime::Cuda;
    config.backend.device = "gpu".into();
    config.backend.fallback = Fallback::Cpu;
    let fallback = Engine::load_with(&config, &paths, |_, _, runtime| {
        assert_eq!(runtime, Runtime::Default);
        Ok(Box::new(FakeBackend {
            voices: 1,
            samples: Ok(vec![0.0]),
        }))
    })
    .unwrap();
    assert!(fallback.fallback_used);
}
