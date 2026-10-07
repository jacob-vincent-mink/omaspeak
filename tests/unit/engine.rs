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
        cache_dir: root.join("cache"),
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
        engine(Ok(vec![0.0]), 0)
            .synthesize("x", 1.0, 0, &output)
            .is_err()
    );
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
    let mut invalid_rate = engine(Ok(vec![0.0]), 1);
    invalid_rate.sample_rate = -1;
    assert!(invalid_rate.synthesize("x", 1.0, 0, &output).is_err());
}

#[test]
fn engine_load_reports_shape_backend_and_runtime_errors() {
    let root = temp("load-errors");
    let paths = paths(&root);
    let mut config = Config::default();
    config.backend.threads = 0;
    assert!(Engine::load(&config, &paths).is_err());

    config.backend.threads = 2;
    config.backend.kind = "other".into();
    assert!(
        Engine::load(&config, &paths)
            .err()
            .unwrap()
            .to_string()
            .contains("unsupported TTS backend")
    );

    config.backend.kind = "supertonic".into();
    config.model.family = "unknown".into();
    assert!(Engine::load(&config, &paths).is_err());

    config.model.family = "supertonic".into();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "cpu".into();
    assert!(Engine::load(&config, &paths).is_err());

    let native = root.join("native");
    fs::create_dir_all(&native).unwrap();
    let library = native.join("libopenvino_c.so");
    let plugins = native.join("plugins.xml");
    fs::write(&library, b"fixture").unwrap();
    fs::write(&plugins, b"fixture").unwrap();
    config.backend.openvino_library = Some(library);
    config.backend.openvino_plugins = Some(plugins);
    assert!(Engine::load(&config, &paths).is_err());
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
    let mut attempts = Vec::new();
    let fallback = Engine::load_with(&config, &paths, |_, _, runtime| {
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
    let error = Engine::load_with(&config, &paths, |_, _, _| {
        Err(anyhow::anyhow!("CUDA provider unavailable"))
    })
    .err()
    .unwrap();
    assert_eq!(error.to_string(), "CUDA provider unavailable");

    config.backend.fallback = Fallback::Cpu;
    let mut attempts = 0;
    let error = Engine::load_with(&config, &paths, |_, _, runtime| {
        attempts += 1;
        Err(anyhow::anyhow!("{runtime:?} initialization failed"))
    })
    .err()
    .unwrap();
    assert_eq!(attempts, 2);
    assert!(error.to_string().contains("CPU fallback also failed"));

    config.backend.kind = "unknown".into();
    let mut attempts = 0;
    let error = Engine::load_with(&config, &paths, |_, _, _| {
        attempts += 1;
        Err(anyhow::anyhow!("unknown provider"))
    })
    .err()
    .unwrap();
    assert_eq!(attempts, 1);
    assert_eq!(error.to_string(), "unknown provider");
}

#[test]
fn direct_openvino_fallback_keeps_provider_and_model_while_selecting_cpu() {
    let root = temp("openvino-fallback");
    let paths = paths(&root);
    let mut config = Config::default();
    crate::catalog::model("supertonic-3-openvino")
        .unwrap()
        .activate(&mut config);
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "npu".into();
    config.backend.fallback = Fallback::Cpu;

    let mut attempts = Vec::new();
    let loaded = Engine::load_with(&config, &paths, |attempt, _, runtime| {
        attempts.push((
            attempt.backend.kind.clone(),
            runtime,
            attempt.backend.device.clone(),
        ));
        if attempt.backend.device == "npu" {
            Err(anyhow::anyhow!("NPU unavailable"))
        } else {
            Ok(Box::new(FakeBackend {
                voices: 1,
                samples: Ok(vec![0.0]),
            }))
        }
    })
    .unwrap();

    assert_eq!(
        attempts,
        [
            ("supertonic".into(), Runtime::Openvino, "npu".into()),
            ("supertonic".into(), Runtime::Openvino, "cpu".into()),
        ]
    );
    assert_eq!(loaded.model_name, "supertonic-3-openvino");
    assert_eq!(loaded.effective_runtime, Runtime::Openvino);
    assert!(loaded.fallback_used);
}

#[test]
fn output_parent_errors_are_actionable() {
    let root = temp("output-errors");
    let blocked_parent = root.join("blocked-parent");
    std::fs::write(&blocked_parent, b"not a directory").unwrap();
    let error = engine(Ok(vec![0.0]), 1)
        .synthesize("hello", 1.0, 0, &blocked_parent.join("out.wav"))
        .err()
        .unwrap();
    assert!(error.to_string().contains("create output directory"));

    let directory_output = root.join("existing-directory.wav");
    std::fs::create_dir(&directory_output).unwrap();
    let error = engine(Ok(vec![0.0]), 1)
        .synthesize("hello", 1.0, 0, &directory_output)
        .err()
        .expect("a directory cannot be replaced with WAV output");
    assert!(error.to_string().contains("create WAV"));
}

#[test]
fn streamed_chunks_reach_the_sink_before_later_generation_and_match_wav() {
    use std::cell::Cell;
    struct Incremental {
        emitted: Cell<bool>,
    }
    impl TtsBackend for Incremental {
        fn kind(&self) -> &'static str {
            "incremental"
        }
        fn sample_rate(&self) -> i32 {
            24_000
        }
        fn num_voices(&self) -> i32 {
            1
        }
        fn generate(&self, _: &str, _: f32, _: i32) -> Result<Vec<f32>> {
            panic!("streaming must not call complete-buffer generation")
        }
        fn generate_stream(
            &self,
            _: &str,
            _: f32,
            _: i32,
            sink: &mut dyn FnMut(&[f32]) -> Result<()>,
        ) -> Result<()> {
            sink(&[0.5, -0.5])?;
            self.emitted.set(true);
            sink(&[1.0, -1.0])
        }
    }
    let root = temp("streaming");
    let engine = Engine {
        backend: Box::new(Incremental {
            emitted: Cell::new(false),
        }),
        backend_kind: "incremental",
        model_name: "stream".into(),
        sample_rate: 24_000,
        load_time: Duration::ZERO,
        effective_runtime: Runtime::Default,
        fallback_used: false,
    };
    let mut chunks = Vec::new();
    let output = root.join("stream.wav");
    let result = engine
        .synthesize_stream("hello", 1.0, 0, &output, &mut |chunk| {
            chunks.push(chunk.to_vec());
            Ok(())
        })
        .unwrap();
    assert_eq!(chunks, vec![vec![0.5, -0.5], vec![1.0, -1.0]]);
    assert_eq!(result.samples, 4);
    let audio = hound::WavReader::open(output)
        .unwrap()
        .into_samples::<i16>()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(audio, vec![16383, -16383, 32767, -32767]);
    let mut calls = 0;
    let error = engine
        .synthesize_stream("hello", 1.0, 0, &root.join("cancelled.wav"), &mut |_| {
            calls += 1;
            anyhow::bail!("sink closed")
        })
        .err()
        .unwrap();
    assert_eq!(calls, 1);
    assert!(error.to_string().contains("sink closed"));
}

#[test]
fn failed_offline_generation_does_not_truncate_existing_output() {
    let path = temp("existing-offline").join("out.wav");
    fs::write(&path, b"original audio").unwrap();
    assert!(
        engine(Err("generation failed"), 1)
            .synthesize("hello", 1.0, 0, &path)
            .is_err()
    );
    assert_eq!(fs::read(&path).unwrap(), b"original audio");
}

#[test]
fn buffered_provider_stream_fallback_validates_pcm_and_propagates_sink_failure() {
    let root = temp("buffered-stream-fallback");
    let output = root.join("out.wav");
    let mut delivered = Vec::new();
    engine(Ok(vec![0.25; 5000]), 1)
        .synthesize_stream("hello", 1.0, 0, &output, &mut |chunk| {
            assert!(chunk.len() <= 4096);
            delivered.extend_from_slice(chunk);
            Ok(())
        })
        .unwrap();
    assert_eq!(delivered, vec![0.25; 5000]);
    for samples in [Ok(vec![]), Ok(vec![f32::NAN]), Err("provider failed")] {
        assert!(
            engine(samples, 1)
                .synthesize_stream("hello", 1.0, 0, &output, &mut |_| Ok(()))
                .is_err()
        );
    }
    assert!(
        engine(Ok(vec![0.25]), 1)
            .synthesize_stream("hello", 1.0, 0, &output, &mut |_| bail!("sink closed"))
            .is_err()
    );
    fs::remove_dir_all(root).unwrap();
}
