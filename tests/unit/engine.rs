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
    let provider = cuda.model.provider.unwrap();
    assert!(provider.starts_with("cuda:"));
    assert_eq!(
        fs::read_to_string(provider.strip_prefix("cuda:").unwrap()).unwrap(),
        "cudnn_conv_algo_search=HEURISTIC\ndevice_id=0\n"
    );

    config.model.family = "unknown".into();
    assert!(build_sherpa_config(&config, &paths, Runtime::Default).is_err());
    config.model.family = "piper".into();
    fs::remove_file(directory.join(&config.model.model_file)).unwrap();
    assert!(build_sherpa_config(&config, &paths, Runtime::Default).is_err());
    assert!(build_sherpa_config(&config, &paths, Runtime::Openvino).is_err());
}

#[test]
fn cuda_provider_config_selects_device_and_forwards_options() {
    let root = temp("cuda-generated");
    let paths = paths(&root);
    let mut config = Config::default();
    config.backend.runtime = Runtime::Cuda;
    config.backend.device = "gpu".into();
    config.backend.device_id = 2;
    config
        .backend
        .options
        .insert("cudnn_conv_algo_search".into(), "HEURISTIC".into());
    config
        .backend
        .options
        .insert("gpu_mem_limit".into(), "4294967296".into());
    install_model_assets(&config, &paths);

    let native = build_sherpa_config(&config, &paths, Runtime::Cuda).unwrap();
    let provider = native.model.provider.unwrap();
    let provider_path = PathBuf::from(provider.strip_prefix("cuda:").unwrap());
    assert_eq!(
        provider_path,
        paths.state_dir.join("cache/cuda/device-2/provider.config")
    );
    let contents = fs::read_to_string(&provider_path).unwrap();
    assert!(contents.contains("device_id=2\n"));
    assert!(contents.contains("cudnn_conv_algo_search=HEURISTIC\n"));
    assert!(contents.contains("gpu_mem_limit=4294967296\n"));

    config
        .backend
        .options
        .insert("device_id".into(), "3".into());
    assert!(build_sherpa_config(&config, &paths, Runtime::Cuda).is_err());
}

#[test]
fn cuda_accepts_an_explicit_provider_config_relative_to_app_config() {
    let root = temp("cuda-explicit");
    let paths = paths(&root);
    let mut config = Config::default();
    config.backend.runtime = Runtime::Cuda;
    config.backend.device = "gpu".into();
    install_model_assets(&config, &paths);
    fs::create_dir_all(paths.config_file.parent().unwrap()).unwrap();
    let supplied = paths.config_file.parent().unwrap().join("cuda.conf");
    fs::write(&supplied, "device_id=1\n").unwrap();
    config.backend.provider_config = "cuda.conf".into();

    let provider = build_sherpa_config(&config, &paths, Runtime::Cuda)
        .unwrap()
        .model
        .provider
        .unwrap();
    assert_eq!(provider, format!("cuda:{}", supplied.display()));
    config.backend.provider_config = "missing.conf".into();
    assert!(build_sherpa_config(&config, &paths, Runtime::Cuda).is_err());
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
fn npu_catalog_model_submits_all_supertonic_components_and_remains_overridable() {
    let root = temp("supertonic-npu-catalog-components");
    let paths = paths(&root);
    let mut config = Config::default();
    crate::catalog::model("supertonic-3-npu")
        .unwrap()
        .activate(&mut config);
    let directory = config.model_directory(&paths);
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
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "npu".into();

    let native = build_sherpa_config(&config, &paths, Runtime::Openvino).unwrap();
    let provider = native.model.provider.unwrap();
    let provider_path = PathBuf::from(provider.strip_prefix("openvino:").unwrap());
    let contents = fs::read_to_string(&provider_path).unwrap();
    assert!(contents.contains("SherpaOnnx.SupertonicComponents=all\n"));

    fs::write(directory.join("vector_estimator.int8.onnx"), b"fixture").unwrap();
    config.model.vector_estimator = "vector_estimator.int8.onnx".into();
    build_sherpa_config(&config, &paths, Runtime::Openvino).unwrap();
    let contents = fs::read_to_string(&provider_path).unwrap();
    assert!(
        contents
            .contains("SherpaOnnx.SupertonicComponents=duration_predictor,text_encoder,vocoder\n")
    );

    config.model.vector_estimator = "vector_estimator.onnx".into();
    let custom_directory = root.join("custom-model");
    fs::create_dir_all(&custom_directory).unwrap();
    for asset in [
        &config.model.duration_predictor,
        &config.model.text_encoder,
        &config.model.vector_estimator,
        &config.model.vocoder,
        &config.model.tts_json,
        &config.model.unicode_indexer,
        &config.model.voice_style,
    ] {
        fs::write(custom_directory.join(asset), b"fixture").unwrap();
    }
    config.model.directory = custom_directory.display().to_string();
    build_sherpa_config(&config, &paths, Runtime::Openvino).unwrap();
    let contents = fs::read_to_string(&provider_path).unwrap();
    assert!(
        contents
            .contains("SherpaOnnx.SupertonicComponents=duration_predictor,text_encoder,vocoder\n")
    );

    config.backend.options.insert(
        "SherpaOnnx.SupertonicComponents".into(),
        "vector_estimator".into(),
    );
    build_sherpa_config(&config, &paths, Runtime::Openvino).unwrap();
    let contents = fs::read_to_string(provider_path).unwrap();
    assert!(contents.contains("SherpaOnnx.SupertonicComponents=vector_estimator\n"));
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

    let mut attempts = Vec::new();
    let fallback =
        Engine::load_with_capabilities(&config, &paths, &["cpu", "cuda"], |_, _, runtime| {
            attempts.push(runtime);
            if runtime == Runtime::Cuda {
                Err(anyhow::anyhow!("CUDA provider unavailable"))
            } else {
                Ok(Box::new(FakeBackend {
                    voices: 1,
                    samples: Ok(vec![0.0]),
                }))
            }
        })
        .unwrap();
    assert_eq!(attempts, [Runtime::Cuda, Runtime::Default]);
    assert_eq!(fallback.effective_runtime, Runtime::Default);
    assert!(fallback.fallback_used);

    config.backend.fallback = Fallback::Error;
    let error = Engine::load_with_capabilities(&config, &paths, &["cpu", "cuda"], |_, _, _| {
        Err(anyhow::anyhow!("CUDA provider unavailable"))
    })
    .err()
    .unwrap();
    assert_eq!(error.to_string(), "CUDA provider unavailable");

    config.backend.fallback = Fallback::Cpu;
    let mut attempts = 0;
    let error =
        Engine::load_with_capabilities(&config, &paths, &["cpu", "cuda"], |_, _, runtime| {
            attempts += 1;
            Err(anyhow::anyhow!("{runtime:?} initialization failed"))
        })
        .err()
        .unwrap();
    assert_eq!(attempts, 2);
    assert!(error.to_string().contains("CPU fallback also failed"));
}

#[test]
fn model_and_provider_paths_report_actionable_filesystem_errors() {
    let root = temp("filesystem-errors");
    let mut app_paths = paths(&root);
    let mut config = Config::default();

    config.model.model_file.clear();
    let error = build_sherpa_config(&config, &app_paths, Runtime::Default).unwrap_err();
    assert!(error.to_string().contains("path is not configured"));
    config.model.model_file = "model.onnx".into();
    config.model.data_directory.clear();
    install_model_assets(&config, &app_paths);
    let error = build_sherpa_config(&config, &app_paths, Runtime::Default).unwrap_err();
    assert!(error.to_string().contains("directory is not configured"));

    let supplied = root.join("absolute-provider.config");
    fs::write(&supplied, "device_id=0\n").unwrap();
    config.backend.runtime = Runtime::Cuda;
    config.backend.device = "gpu".into();
    config.backend.provider_config = supplied.display().to_string();
    assert_eq!(
        cuda_provider(&config, &app_paths).unwrap(),
        format!("cuda:{}", supplied.canonicalize().unwrap().display())
    );

    config.backend.provider_config.clear();
    fs::write(&app_paths.state_dir, "not a directory").unwrap();
    let error = generate_cuda_provider_config(&config, &app_paths).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("create private runtime directory")
    );

    app_paths.state_dir = root.join("openvino-state");
    fs::write(&app_paths.state_dir, "not a directory").unwrap();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "gpu".into();
    let error = generate_openvino_provider_config(&config, &app_paths).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("create private runtime directory")
    );

    app_paths.state_dir = root.join("install-conflict-state");
    let provider_dir = app_paths.state_dir.join("cache/cuda/device-0");
    fs::create_dir_all(provider_dir.join("provider.config")).unwrap();
    config.backend.runtime = Runtime::Cuda;
    config.backend.device = "gpu".into();
    let error = generate_cuda_provider_config(&config, &app_paths).unwrap_err();
    assert!(error.to_string().contains("install provider config"));
    assert_eq!(
        fs::read_dir(&provider_dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
            .count(),
        0
    );

    assert!(write_private_atomic(Path::new("/"), b"x").is_err());
    assert!(create_private_directory(&root.join("missing/child")).is_ok());
}

#[test]
fn engine_native_boundary_and_output_paths_fail_cleanly() {
    let root = temp("native-and-output-errors");
    let app_paths = paths(&root);
    let mut config = Config::default();
    install_model_assets(&config, &app_paths);

    let error = Engine::load(&config, &app_paths).err().unwrap();
    assert!(!error.to_string().is_empty());

    let directory = config.model_directory(&app_paths);
    config.model.data_directory = "data-file".into();
    fs::write(directory.join("data-file"), b"not a directory").unwrap();
    let error = build_sherpa_config(&config, &app_paths, Runtime::Default).unwrap_err();
    assert!(error.to_string().contains("directory is missing"));

    let blocked_parent = root.join("blocked-parent");
    fs::write(&blocked_parent, b"not a directory").unwrap();
    let error = engine(Ok(vec![0.0]), 1)
        .synthesize("hello", 1.0, 0, &blocked_parent.join("out.wav"))
        .err()
        .unwrap();
    assert!(error.to_string().contains("create output directory"));

    configure_supertonic(&mut config, &app_paths);
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "cpu".into();
    let provider = build_sherpa_config(&config, &app_paths, Runtime::Openvino)
        .unwrap()
        .model
        .provider
        .unwrap();
    let contents = fs::read_to_string(provider.strip_prefix("openvino:").unwrap()).unwrap();
    assert!(contents.contains("disable_dynamic_shapes=True\n"));
}
