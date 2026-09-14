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
    assert!(build_sherpa_config(&config, &paths, Runtime::Openvino).is_err());
}

#[test]
fn sherpa_config_is_built_without_loading_untrusted_native_models() {
    let root = temp("sherpa-config");
    let paths = paths(&root);
    let mut config = Config::default();
    let directory = config.model_directory(&paths);
    fs::create_dir_all(directory.join("espeak-ng-data")).unwrap();
    fs::write(directory.join(&config.model.model_file), b"fixture").unwrap();
    fs::write(directory.join(&config.model.tokens_file), b"fixture").unwrap();

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
