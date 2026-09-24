use super::*;
use crate::backend::Runtime;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn temp(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "omaspeak-supertonic-test-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn paths(root: &Path) -> AppPaths {
    AppPaths {
        config_file: root.join("config/config.toml"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    }
}

fn frontend() -> SupertonicFrontend {
    SupertonicFrontend {
        config: TtsFileConfig {
            ae: AeConfig {
                sample_rate: 100,
                base_chunk_size: 2,
            },
            ttl: TtlConfig {
                chunk_compress_factor: 2,
                latent_dim: 2,
            },
        },
        indexer: (0..2048).collect(),
        style: VoiceStyle {
            speakers: 2,
            ttl_shape: [2, 2, 2],
            ttl: (0..8).map(|value| value as f32).collect(),
            dp_shape: [2, 1, 2],
            dp: (0..4).map(|value| value as f32).collect(),
        },
        language: "en".into(),
        steps: 3,
        seed: Some(42),
        silence_seconds: 0.1,
    }
}

#[derive(Default)]
struct FakePipeline {
    calls: Vec<Graph>,
    duration: Vec<f32>,
    fail: Option<Graph>,
    wrong_vector_shape: bool,
    bad_wav: Option<Vec<f32>>,
}

impl FakePipeline {
    fn successful() -> Self {
        Self {
            duration: vec![0.2],
            ..Default::default()
        }
    }
}

impl ModelPipeline for FakePipeline {
    fn run(
        &mut self,
        graph: Graph,
        inputs: Vec<NamedTensor>,
        output: &'static str,
    ) -> Result<NamedTensor> {
        self.calls.push(graph);
        if self.fail == Some(graph) {
            bail!("injected {} failure", graph.name());
        }
        match graph {
            Graph::DurationPredictor => {
                assert_eq!(output, "duration");
                assert_eq!(
                    inputs.iter().map(|item| item.name).collect::<Vec<_>>(),
                    ["text_ids", "style_dp", "text_mask"]
                );
                Ok(NamedTensor {
                    name: output,
                    shape: vec![self.duration.len() as i64],
                    data: TensorData::F32(self.duration.clone()),
                })
            }
            Graph::TextEncoder => {
                assert_eq!(output, "text_emb");
                let text_len = inputs[0].shape[1];
                NamedTensor::f32(
                    output,
                    [1, 2, text_len],
                    vec![0.25; (2 * text_len) as usize],
                )
            }
            Graph::VectorEstimator => {
                assert_eq!(output, "denoised_latent");
                let latent = inputs
                    .into_iter()
                    .find(|item| item.name == "noisy_latent")
                    .unwrap();
                let (mut shape, values) = latent.into_f32()?;
                if self.wrong_vector_shape {
                    shape[2] += 1;
                }
                Ok(NamedTensor {
                    name: output,
                    shape,
                    data: TensorData::F32(values),
                })
            }
            Graph::Vocoder => {
                assert_eq!(output, "wav_tts");
                let values = self.bad_wav.clone().unwrap_or_else(|| {
                    let (_, latent) = inputs.into_iter().next().unwrap().into_f32().unwrap();
                    latent.into_iter().cycle().take(1_000).collect()
                });
                Ok(NamedTensor {
                    name: output,
                    shape: vec![1, 1, values.len() as i64],
                    data: TensorData::F32(values),
                })
            }
        }
    }
}

fn write_compact_assets(root: &Path) -> (Config, AppPaths) {
    let app_paths = paths(root);
    let model = root.join("model");
    fs::create_dir_all(&model).unwrap();
    let mut config = Config::default();
    config.backend.kind = "supertonic".into();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "cpu".into();
    config.model.family = "supertonic".into();
    config.model.name = "test-supertonic".into();
    config.model.directory = model.display().to_string();
    config.model.duration_predictor = "duration.onnx".into();
    config.model.text_encoder = "text.onnx".into();
    config.model.vector_estimator = "vector.onnx".into();
    config.model.vocoder = "vocoder.onnx".into();
    config.model.tts_json = "tts.json".into();
    config.model.unicode_indexer = "unicode.json".into();
    config.model.voice_style = "voice_styles".into();
    config.model.language = "en".into();
    config.model.steps = 3;
    for file in ["duration.onnx", "text.onnx", "vector.onnx", "vocoder.onnx"] {
        fs::write(model.join(file), b"onnx fixture").unwrap();
    }
    fs::write(
        model.join("tts.json"),
        br#"{"ae":{"sample_rate":100,"base_chunk_size":2},"ttl":{"chunk_compress_factor":2,"latent_dim":2}}"#,
    )
    .unwrap();
    fs::write(
        model.join("unicode.json"),
        serde_json::to_vec(&(0_i32..2048).collect::<Vec<_>>()).unwrap(),
    )
    .unwrap();
    write_style_directory(&model.join("voice_styles"));
    (config, app_paths)
}

fn write_style_directory(directory: &Path) {
    fs::create_dir_all(directory).unwrap();
    for (index, name) in crate::catalog::SUPERTONIC_VOICE_NAMES.iter().enumerate() {
        let base = index as f32 * 4.0;
        let style = serde_json::json!({
            "style_ttl": {
                "data": [[[base, base + 1.0], [base + 2.0, base + 3.0]]],
                "dims": [1, 2, 2],
                "type": "float32"
            },
            "style_dp": {
                "data": [[[index as f32 * 10.0, index as f32 * 10.0 + 1.0]]],
                "dims": [1, 1, 2],
                "type": "float32"
            }
        });
        fs::write(
            directory.join(format!("{name}.json")),
            serde_json::to_vec(&style).unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn named_tensors_validate_shape_count_and_type() {
    assert_eq!(Graph::DurationPredictor.name(), "duration_predictor");
    assert_eq!(Graph::TextEncoder.name(), "text_encoder");
    assert_eq!(Graph::VectorEstimator.name(), "vector_estimator");
    assert_eq!(Graph::Vocoder.name(), "vocoder");
    let tensor = NamedTensor::f32("x", [1, 2], vec![1.0, 2.0]).unwrap();
    assert_eq!(
        tensor.clone().into_f32().unwrap(),
        (vec![1, 2], vec![1.0, 2.0])
    );
    assert!(
        NamedTensor::i64("x", [2], vec![1, 2])
            .unwrap()
            .into_f32()
            .is_err()
    );
    assert!(NamedTensor::f32("x", [], vec![]).is_err());
    assert!(NamedTensor::f32("x", [1, 0], vec![]).is_err());
    assert!(NamedTensor::f32("x", [2, 2], vec![0.0]).is_err());
    assert!(NamedTensor::f32("x", [i64::MAX, i64::MAX], Vec::new()).is_err());
    assert!(execution_device_matches("GPU", "GPU.0"));
    assert!(execution_device_matches("NPU", "[CPU, NPU]"));
    assert!(!execution_device_matches("GPU", "CPU"));
}

#[test]
fn unicode_classifiers_cover_every_supported_terminal_and_emoji_range() {
    for character in [
        '😀',
        '🌀',
        '🚀',
        '\u{1F700}',
        '\u{1F780}',
        '\u{1F800}',
        '🤖',
        '\u{1FA00}',
        '☀',
        '🇺',
    ] {
        assert!(is_emoji(character), "expected emoji: {character}");
    }
    for character in [
        '.', '!', '?', ';', ':', ',', '\'', '"', ')', ']', '}', '>', '…', '。', '」', '』', '】',
        '〉', '》', '›', '»', '“', '”', '‘', '’',
    ] {
        assert!(
            is_ending_punctuation(character),
            "expected punctuation: {character}"
        );
    }
    assert!(!is_ending_punctuation('x'));
}

#[test]
fn normalization_is_nfkd_multilingual_and_matches_reference_replacements() {
    let indexer = (0..2048).collect::<Vec<_>>();
    let (ids, mask) = process_text(
        "  Café—test_“” '' [x] @ e.g., i.e., ♥©\\ 😀  ",
        "fr",
        &indexer,
    );
    let normalized = ids
        .iter()
        .filter_map(|&value| char::from_u32(value as u32))
        .collect::<String>();
    assert_eq!(mask, vec![1.0; ids.len()]);
    assert!(normalized.starts_with("<fr>Cafe\u{301}-test \"' x at for example,"));
    assert!(normalized.contains("that is,"));
    assert!(normalized.ends_with("</fr>"));
    assert!(!normalized.contains(['♥', '©', '\\', '😀']));
    let (ids, _) = process_text("already!", "en", &indexer);
    let value = ids
        .iter()
        .filter_map(|&id| char::from_u32(id as u32))
        .collect::<String>();
    assert_eq!(value, "<en>already!</en>");
    assert!(is_emoji('☀'));
    assert!(!is_emoji('A'));
}

#[test]
fn chunking_handles_blank_lines_abbreviations_sentences_and_long_words() {
    assert!(chunk_text(" \n \n", 20).is_empty());
    assert_eq!(chunk_text("one\n  \n two", 20), ["one", "two"]);
    assert_eq!(
        split_sentences("Dr. Smith went. Next!"),
        ["Dr. Smith went.", "Next!"]
    );
    assert_eq!(split_sentences("No terminator"), ["No terminator"]);
    let chunks = chunk_text(
        "Dr. Smith has several words, enough to split. This is another sentence!",
        18,
    );
    assert!(chunks.len() > 2);
    assert!(chunks.iter().all(|chunk| chunk.chars().count() <= 18));
    assert_eq!(split_long_piece("abcdefgh", 3), ["abc", "def", "gh"]);
    assert_eq!(split_long_piece("a bc", 0), ["a", "b", "c"]);
    assert_eq!(split_long_piece("one two", 20), ["one two"]);
    assert_eq!(split_long_piece("one abcdef", 4), ["one", "abcd", "ef"]);
    assert_eq!(split_sentences("Dr."), ["Dr."]);
    assert_eq!(split_paragraphs("one\ntwo"), ["one two"]);
    assert_eq!(
        chunk_text("one two three four", 7),
        ["one two", "three", "four"]
    );
}

#[test]
fn fixed_seed_generation_runs_all_graphs_and_inserts_chunk_silence() {
    let frontend = frontend();
    let mut first = FakePipeline::successful();
    let audio = frontend.generate(&mut first, "hello", 1.0, 1).unwrap();
    assert_eq!(audio.len(), 20);
    assert_eq!(
        first.calls,
        [
            Graph::DurationPredictor,
            Graph::TextEncoder,
            Graph::VectorEstimator,
            Graph::VectorEstimator,
            Graph::VectorEstimator,
            Graph::Vocoder,
        ]
    );
    let mut second = FakePipeline::successful();
    assert_eq!(
        audio,
        frontend.generate(&mut second, "hello", 1.0, 1).unwrap()
    );
    let mut chunks = FakePipeline::successful();
    let audio = frontend
        .generate(&mut chunks, "first\n\nsecond", 1.0, 0)
        .unwrap();
    assert_eq!(audio.len(), 50);
    assert!(audio[20..30].iter().all(|sample| *sample == 0.0));
}

#[derive(Default)]
struct NpuShapePipeline {
    calls: Vec<(Graph, Vec<NamedTensor>)>,
    duration: f32,
}

impl ModelPipeline for NpuShapePipeline {
    fn run(
        &mut self,
        graph: Graph,
        inputs: Vec<NamedTensor>,
        output: &'static str,
    ) -> Result<NamedTensor> {
        self.calls.push((graph, inputs.clone()));
        match graph {
            Graph::DurationPredictor => NamedTensor::f32(output, [1], vec![self.duration]),
            Graph::TextEncoder => {
                NamedTensor::f32(output, [1, 2, NPU_TEXT_BUCKET], vec![0.25; 640])
            }
            Graph::VectorEstimator => {
                let shape = inputs[0].shape.clone();
                let len = inputs[0].clone().into_f32()?.1.len();
                NamedTensor::f32(output, shape, vec![7.0; len])
            }
            Graph::Vocoder => NamedTensor::f32(output, [1, 1, 4096], vec![0.5; 4096]),
        }
    }
}

#[test]
fn npu_generation_uses_only_prepared_shapes_masks_padding_and_trims_audio() {
    assert_eq!(npu_latent_bucket(1).unwrap(), 32);
    assert_eq!(npu_latent_bucket(32).unwrap(), 32);
    assert_eq!(npu_latent_bucket(33).unwrap(), 64);
    assert_eq!(npu_latent_bucket(256).unwrap(), 256);
    assert!(npu_latent_bucket(257).is_err());

    let frontend = frontend();
    let mut pipeline = NpuShapePipeline {
        duration: 0.2,
        ..Default::default()
    };
    let audio = frontend
        .generate_npu(&mut pipeline, "hello", 1.0, 0)
        .unwrap();
    assert_eq!(audio.len(), 20);

    for (graph, inputs) in &pipeline.calls {
        match graph {
            Graph::DurationPredictor | Graph::TextEncoder => {
                assert_eq!(inputs[0].shape, [1, NPU_TEXT_BUCKET]);
                let mask = inputs
                    .iter()
                    .find(|input| input.name == "text_mask")
                    .unwrap();
                assert_eq!(mask.shape, [1, 1, NPU_TEXT_BUCKET]);
                let (_, mask) = mask.clone().into_f32().unwrap();
                assert!(mask.contains(&1.0));
                assert_eq!(mask.last(), Some(&0.0));
            }
            Graph::VectorEstimator => {
                assert_eq!(inputs[0].shape, [1, 4, 32]);
                let (_, latent) = inputs[0].clone().into_f32().unwrap();
                for channel in 0..4 {
                    assert!(
                        latent[channel * 32 + 5..(channel + 1) * 32]
                            .iter()
                            .all(|value| *value == 0.0)
                    );
                }
                let mask = inputs
                    .iter()
                    .find(|input| input.name == "latent_mask")
                    .unwrap()
                    .clone()
                    .into_f32()
                    .unwrap()
                    .1;
                assert_eq!(&mask[..5], &[1.0; 5]);
                assert!(mask[5..].iter().all(|value| *value == 0.0));
            }
            Graph::Vocoder => {
                assert_eq!(inputs[0].shape, [1, 4, 32]);
                let (_, latent) = inputs[0].clone().into_f32().unwrap();
                for channel in 0..4 {
                    assert!(
                        latent[channel * 32 + 5..(channel + 1) * 32]
                            .iter()
                            .all(|value| *value == 0.0)
                    );
                }
            }
        }
    }
    assert_eq!(
        pipeline
            .calls
            .iter()
            .filter(|(graph, _)| *graph == Graph::VectorEstimator)
            .count(),
        frontend.steps as usize
    );
}

#[test]
fn npu_generation_rejects_inputs_outside_the_prepared_shape_plan() {
    let frontend = frontend();
    let mut too_long = NpuShapePipeline {
        duration: 25.0,
        ..Default::default()
    };
    let error = frontend
        .generate_npu(&mut too_long, "hello", 1.0, 0)
        .unwrap_err()
        .to_string();
    assert!(error.contains("prepared Intel NPU limit"), "{error}");
    assert!(
        !too_long
            .calls
            .iter()
            .any(|(graph, _)| *graph == Graph::VectorEstimator)
    );

    let mut expanded = NpuShapePipeline {
        duration: 0.2,
        ..Default::default()
    };
    let error = frontend
        .generate_npu(&mut expanded, &"@".repeat(300), 1.0, 0)
        .unwrap_err()
        .to_string();
    assert!(error.contains("normalized text chunk"), "{error}");
    assert!(expanded.calls.is_empty());
}

#[test]
fn npu_generation_uses_a_shorter_device_safe_chunk_budget() {
    for (language, text) in [("en", "word ".repeat(40)), ("ja", "あ".repeat(100))] {
        let mut frontend = frontend();
        frontend.language = language.into();
        let mut pipeline = NpuShapePipeline {
            duration: 0.2,
            ..Default::default()
        };
        frontend.generate_npu(&mut pipeline, &text, 1.0, 0).unwrap();
        assert!(
            pipeline
                .calls
                .iter()
                .filter(|(graph, _)| *graph == Graph::DurationPredictor)
                .count()
                >= 2
        );
    }
}

#[test]
fn npu_cache_manifest_is_fingerprinted_and_requires_every_recorded_blob() {
    let root = temp("npu-cache-manifest");
    let (mut config, app_paths) = write_compact_assets(&root);
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "npu".into();
    let runtime = root.join("runtime");
    fs::create_dir_all(&runtime).unwrap();
    let library = runtime.join("libopenvino_c.so");
    let plugins = runtime.join("plugins.xml");
    fs::write(&library, b"runtime-a").unwrap();
    fs::write(&plugins, b"<ie/>").unwrap();
    config.backend.openvino_library = Some(library.clone());
    config.backend.openvino_plugins = Some(plugins);
    let original_fingerprint = npu_cache_fingerprint(&config, &app_paths).unwrap();
    config
        .backend
        .options
        .insert("NPU_PLATFORM".into(), "5010".into());
    assert_ne!(
        npu_cache_fingerprint(&config, &app_paths).unwrap(),
        original_fingerprint
    );
    config.backend.options.clear();
    config.model.language = "ja".into();
    config.model.steps = 2;
    assert_eq!(
        npu_cache_fingerprint(&config, &app_paths).unwrap(),
        original_fingerprint,
        "language and diffusion-step values change tensor contents but not compiled shapes"
    );
    config.model.language = "en".into();
    config.model.steps = 5;
    fs::write(&library, b"runtime-b").unwrap();
    assert_ne!(
        npu_cache_fingerprint(&config, &app_paths).unwrap(),
        original_fingerprint
    );
    fs::write(&library, b"runtime-a").unwrap();
    let model_file = config
        .model_directory(&app_paths)
        .join(&config.model.duration_predictor);
    let original_model = fs::read(&model_file).unwrap();
    fs::write(&model_file, vec![b'x'; original_model.len()]).unwrap();
    assert_ne!(
        npu_cache_fingerprint(&config, &app_paths).unwrap(),
        original_fingerprint
    );
    fs::write(&model_file, original_model).unwrap();
    let directory = npu_cache_directory(&config, &app_paths).unwrap();
    assert!(directory.starts_with(app_paths.cache_dir.join("openvino/npu/static-v1")));
    fs::create_dir_all(directory.join("nested")).unwrap();
    for index in 0..NPU_COMPILED_MODELS {
        fs::write(
            directory.join(format!("nested/model-{index}.blob")),
            b"compiled",
        )
        .unwrap();
    }
    let blobs = write_npu_cache_manifest(&config, &app_paths, &directory).unwrap();
    assert_eq!(blobs.len(), NPU_COMPILED_MODELS);
    let state = npu_cache_state(&config, &app_paths);
    assert!(state.required && state.ready, "{}", state.detail);
    assert_eq!(state.blobs.len(), NPU_COMPILED_MODELS);

    fs::remove_file(directory.join("nested/model-0.blob")).unwrap();
    let state = npu_cache_state(&config, &app_paths);
    assert!(state.required && !state.ready);
    assert!(
        state
            .detail
            .contains("missing or empty compiled-model blobs")
    );
}

#[test]
fn npu_cache_manifest_rejects_shape_path_and_directory_integrity_failures() {
    let root = temp("npu-cache-integrity-errors");
    let (mut config, app_paths) = write_compact_assets(&root);
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "npu".into();
    let runtime = root.join("runtime");
    fs::create_dir_all(&runtime).unwrap();
    config.backend.openvino_library = Some(runtime.join("libopenvino_c.so"));
    config.backend.openvino_plugins = Some(runtime.join("plugins.xml"));
    fs::write(
        config.backend.openvino_library.as_ref().unwrap(),
        b"runtime",
    )
    .unwrap();
    fs::write(config.backend.openvino_plugins.as_ref().unwrap(), b"<ie/>").unwrap();
    let directory = npu_cache_directory(&config, &app_paths).unwrap();
    fs::create_dir_all(&directory).unwrap();
    for index in 0..NPU_COMPILED_MODELS {
        fs::write(directory.join(format!("model-{index}.blob")), b"compiled").unwrap();
    }
    write_npu_cache_manifest(&config, &app_paths, &directory).unwrap();
    let manifest_path = directory.join(NPU_MANIFEST);
    let original = fs::read(&manifest_path).unwrap();

    let write_variant = |update: &dyn Fn(&mut serde_json::Value)| {
        let mut value: serde_json::Value = serde_json::from_slice(&original).unwrap();
        update(&mut value);
        fs::write(&manifest_path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        npu_cache_state(&config, &app_paths)
    };
    let state = write_variant(&|value| value["schema"] = 999.into());
    assert!(state.detail.contains("does not match"));
    let state = write_variant(&|value| {
        value["blobs"].as_array_mut().unwrap().pop();
    });
    assert!(
        state
            .detail
            .contains(&format!("records {}", NPU_COMPILED_MODELS - 1))
    );
    let state = write_variant(&|value| value["blobs"][0] = "../escape.blob".into());
    assert!(state.detail.contains("unsafe compiled-model path"));
    let state = write_variant(&|value| value["blobs"][0] = value["blobs"][1].clone());
    assert!(state.detail.contains("duplicate compiled-model blobs"));

    fs::write(&manifest_path, &original).unwrap();
    fs::write(directory.join("model-0.blob"), b"").unwrap();
    let state = npu_cache_state(&config, &app_paths);
    assert!(state.detail.contains("missing or empty"));
    fs::write(directory.join("model-0.blob"), b"compiled").unwrap();
    fs::write(directory.join("extra.blob"), b"compiled").unwrap();
    let state = npu_cache_state(&config, &app_paths);
    assert!(state.detail.contains("does not exactly match"));

    fs::remove_file(directory.join("extra.blob")).unwrap();
    let last_blob = directory.join(format!("model-{}.blob", NPU_COMPILED_MODELS - 1));
    fs::remove_file(&last_blob).unwrap();
    assert!(write_npu_cache_manifest(&config, &app_paths, &directory).is_err());
    fs::write(last_blob, b"").unwrap();
    assert!(write_npu_cache_manifest(&config, &app_paths, &directory).is_err());

    let mut cpu = config;
    cpu.backend.device = "cpu".into();
    assert!(npu_cache_state(&cpu, &app_paths).ready);
    assert!(npu_cache_fingerprint(&cpu, &app_paths).is_err());
}

#[test]
fn npu_backend_refuses_on_demand_compilation_without_setup_manifest() {
    let root = temp("npu-cache-required");
    let (mut config, app_paths) = write_compact_assets(&root);
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "npu".into();
    let error = DirectOpenvinoBackend::create_with(&config, &app_paths, |_, _, _, _| {
        unreachable!("pipeline creation must not run before cache validation")
    })
    .err()
    .unwrap()
    .to_string();
    assert!(error.contains("setup cache --prepare"), "{error}");
}

#[test]
fn generation_rejects_pipeline_duration_shape_and_audio_failures() {
    let frontend = frontend();
    assert!(
        frontend
            .generate(&mut FakePipeline::successful(), "", 1.0, 0)
            .is_err()
    );
    assert!(
        frontend
            .generate(&mut FakePipeline::successful(), "x", 1.0, 2)
            .is_err()
    );
    let mut pipeline = FakePipeline::successful();
    pipeline.duration = vec![0.1, 0.2];
    assert!(frontend.generate(&mut pipeline, "x", 1.0, 0).is_err());
    for duration in [f32::NAN, 700.0] {
        let mut pipeline = FakePipeline::successful();
        pipeline.duration = vec![duration];
        assert!(frontend.generate(&mut pipeline, "x", 1.0, 0).is_err());
    }
    let mut pipeline = FakePipeline::successful();
    pipeline.duration = vec![500.0];
    assert!(frontend.generate(&mut pipeline, "x", 1.0, 0).is_err());
    let mut pipeline = FakePipeline::successful();
    pipeline.duration = vec![-1.0];
    assert!(
        !frontend
            .generate(&mut pipeline, "x", 1.0, 0)
            .unwrap()
            .is_empty()
    );
    let mut pipeline = FakePipeline::successful();
    pipeline.wrong_vector_shape = true;
    assert!(frontend.generate(&mut pipeline, "x", 1.0, 0).is_err());
    for samples in [vec![], vec![f32::NAN]] {
        let mut pipeline = FakePipeline::successful();
        pipeline.bad_wav = Some(samples);
        assert!(frontend.generate(&mut pipeline, "x", 1.0, 0).is_err());
    }
    for graph in [
        Graph::DurationPredictor,
        Graph::TextEncoder,
        Graph::VectorEstimator,
        Graph::Vocoder,
    ] {
        let mut pipeline = FakePipeline::successful();
        pipeline.fail = Some(graph);
        assert!(frontend.generate(&mut pipeline, "x", 1.0, 0).is_err());
    }
}

#[test]
fn generation_covers_multibyte_chunking_and_unseeded_rng() {
    let mut frontend = frontend();
    frontend.language = "ja".into();
    frontend.seed = None;
    let text = "あ".repeat(121);
    let audio = frontend
        .generate(&mut FakePipeline::successful(), &text, 1.0, 0)
        .unwrap();
    assert!(!audio.is_empty());
}

#[test]
fn random_normal_sampler_is_deterministic_finite_and_handles_odd_counts() {
    let mut first = StdRng::seed_from_u64(9);
    let mut second = StdRng::seed_from_u64(9);
    let a = normal_samples(5, &mut first);
    assert_eq!(a, normal_samples(5, &mut second));
    assert_eq!(a.len(), 5);
    assert!(a.iter().all(|value| value.is_finite()));
    assert!(normal_samples(0, &mut first).is_empty());
}

#[test]
fn compact_assets_load_and_preserve_multiple_voice_slices() {
    let root = temp("assets");
    let (mut config, app_paths) = write_compact_assets(&root);
    config.model.options.insert("seed".into(), "7".into());
    config
        .model
        .options
        .insert("silence_duration".into(), "0.5".into());
    let (loaded, graphs) = SupertonicFrontend::load(&config, &app_paths).unwrap();
    assert_eq!(loaded.config.ae.sample_rate, 100);
    assert_eq!(loaded.indexer[100], 100);
    assert_eq!(loaded.seed, Some(7));
    assert_eq!(loaded.silence_seconds, 0.5);
    assert_eq!(graphs.len(), 4);
    let first = loaded.style.slice(0).unwrap();
    let second = loaded.style.slice(1).unwrap();
    assert_eq!(first.ttl, &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(second.ttl, &[4.0, 5.0, 6.0, 7.0]);
    assert_eq!(second.dp, &[10.0, 11.0]);
    assert_eq!(loaded.style.slice(2).unwrap().ttl, &[8.0, 9.0, 10.0, 11.0]);
    assert!(loaded.style.slice(10).is_err());
}

#[test]
fn official_json_asset_parsers_reject_corruption() {
    let root = temp("bad-assets");
    let indexer = root.join("index.json");
    fs::write(&indexer, b"[]").unwrap();
    assert!(read_indexer(&indexer).is_err());
    fs::write(&indexer, b"not json").unwrap();
    assert!(read_indexer(&indexer).is_err());
    fs::write(&indexer, b"[2147483648]").unwrap();
    assert!(read_indexer(&indexer).is_err());

    let voices = root.join("voices");
    assert!(read_voice_styles(&voices).is_err());
    write_style_directory(&voices);
    fs::write(voices.join("M1.json"), b"not json").unwrap();
    assert!(read_voice_styles(&voices).is_err());
    write_style_directory(&voices);
    let invalid = serde_json::json!({
        "style_ttl": {"data": [[[1.0]]], "dims": [2, 1, 1], "type": "float32"},
        "style_dp": {"data": [[[1.0]]], "dims": [1, 1, 1], "type": "float32"}
    });
    fs::write(
        voices.join("M1.json"),
        serde_json::to_vec(&invalid).unwrap(),
    )
    .unwrap();
    assert!(read_voice_styles(&voices).is_err());
    write_style_directory(&voices);
    let mismatch = serde_json::json!({
        "style_ttl": {"data": [[[1.0]]], "dims": [1, 1, 2], "type": "float32"},
        "style_dp": {"data": [[[1.0]]], "dims": [1, 1, 1], "type": "float32"}
    });
    fs::write(
        voices.join("M1.json"),
        serde_json::to_vec(&mismatch).unwrap(),
    )
    .unwrap();
    assert!(read_voice_styles(&voices).is_err());
    write_style_directory(&voices);
    let wrong_type = serde_json::json!({
        "style_ttl": {"data": [[[1.0]]], "dims": [1, 1, 1], "type": "float16"},
        "style_dp": {"data": [[[1.0]]], "dims": [1, 1, 1], "type": "float32"}
    });
    fs::write(
        voices.join("M1.json"),
        serde_json::to_vec(&wrong_type).unwrap(),
    )
    .unwrap();
    assert!(read_voice_styles(&voices).is_err());

    write_style_directory(&voices);
    let wrong_rank = serde_json::json!({
        "style_ttl": {"data": [[[1.0]]], "dims": [1, 1], "type": "float32"},
        "style_dp": {"data": [[[1.0]]], "dims": [1, 1, 1], "type": "float32"}
    });
    fs::write(
        voices.join("M1.json"),
        serde_json::to_vec(&wrong_rank).unwrap(),
    )
    .unwrap();
    assert!(read_voice_styles(&voices).is_err());

    write_style_directory(&voices);
    let wrong_plane_count = serde_json::json!({
        "style_ttl": {"data": [], "dims": [1, 2, 2], "type": "float32"},
        "style_dp": {"data": [[[1.0]]], "dims": [1, 1, 1], "type": "float32"}
    });
    fs::write(
        voices.join("M1.json"),
        serde_json::to_vec(&wrong_plane_count).unwrap(),
    )
    .unwrap();
    assert!(read_voice_styles(&voices).is_err());

    write_style_directory(&voices);
    let wrong_row_count = serde_json::json!({
        "style_ttl": {"data": [[[1.0, 2.0]]], "dims": [1, 2, 2], "type": "float32"},
        "style_dp": {"data": [[[1.0]]], "dims": [1, 1, 1], "type": "float32"}
    });
    fs::write(
        voices.join("M1.json"),
        serde_json::to_vec(&wrong_row_count).unwrap(),
    )
    .unwrap();
    assert!(read_voice_styles(&voices).is_err());

    assert!(
        flatten_style_component(
            Path::new("non-finite.json"),
            "style_ttl",
            StyleComponentFile {
                data: vec![vec![vec![f32::NAN]]],
                dims: vec![1, 1, 1],
                dtype: "float32".into(),
            },
        )
        .is_err()
    );

    write_style_directory(&voices);
    let different_dimensions = serde_json::json!({
        "style_ttl": {"data": [[[1.0, 2.0]]], "dims": [1, 1, 2], "type": "float32"},
        "style_dp": {"data": [[[1.0, 2.0]]], "dims": [1, 1, 2], "type": "float32"}
    });
    fs::write(
        voices.join("M2.json"),
        serde_json::to_vec(&different_dimensions).unwrap(),
    )
    .unwrap();
    assert!(read_voice_styles(&voices).is_err());
}

#[test]
fn frontend_load_reports_model_config_and_option_errors() {
    let root = temp("load-errors");
    let (mut config, app_paths) = write_compact_assets(&root);
    let model = PathBuf::from(&config.model.directory);
    config.model.family = "unknown".into();
    assert!(SupertonicFrontend::load(&config, &app_paths).is_err());
    config.model.family = "supertonic".into();
    config.model.language = "xx".into();
    assert!(SupertonicFrontend::load(&config, &app_paths).is_err());
    config.model.language = "en".into();
    config.model.steps = 0;
    assert!(SupertonicFrontend::load(&config, &app_paths).is_err());
    config.model.steps = 3;
    config.model.options.insert("seed".into(), "bad".into());
    assert!(SupertonicFrontend::load(&config, &app_paths).is_err());
    config.model.options.clear();
    for silence in ["nan", "11"] {
        config
            .model
            .options
            .insert("silence_duration".into(), silence.into());
        assert!(SupertonicFrontend::load(&config, &app_paths).is_err());
    }
    config.model.options.clear();
    fs::write(model.join("tts.json"), b"not json").unwrap();
    assert!(SupertonicFrontend::load(&config, &app_paths).is_err());
    fs::write(
        model.join("tts.json"),
        br#"{"ae":{"sample_rate":0,"base_chunk_size":2},"ttl":{"chunk_compress_factor":2,"latent_dim":2}}"#,
    )
    .unwrap();
    assert!(SupertonicFrontend::load(&config, &app_paths).is_err());
    for json in [
        br#"{"ae":{"sample_rate":1,"base_chunk_size":0},"ttl":{"chunk_compress_factor":2,"latent_dim":2}}"#.as_slice(),
        br#"{"ae":{"sample_rate":1,"base_chunk_size":2},"ttl":{"chunk_compress_factor":0,"latent_dim":2}}"#.as_slice(),
        br#"{"ae":{"sample_rate":1,"base_chunk_size":2},"ttl":{"chunk_compress_factor":2,"latent_dim":0}}"#.as_slice(),
    ] {
        fs::write(model.join("tts.json"), json).unwrap();
        assert!(SupertonicFrontend::load(&config, &app_paths).is_err());
    }
    fs::remove_file(model.join("tts.json")).unwrap();
    assert!(SupertonicFrontend::load(&config, &app_paths).is_err());
    config.model.tts_json.clear();
    assert!(SupertonicFrontend::load(&config, &app_paths).is_err());
}

#[test]
fn direct_openvino_entry_points_fail_cleanly_before_native_execution() {
    let root = temp("native-errors");
    let (config, app_paths) = write_compact_assets(&root);
    let runtime = OpenvinoRuntimePaths {
        library: root.join("missing-libopenvino_c.so"),
        plugins: root.join("missing-plugins.xml"),
    };
    assert!(probe_runtime(runtime.clone(), "cpu").is_err());
    assert!(DirectOpenvinoBackend::create(&config, &app_paths, runtime).is_err());
}

#[test]
fn openvino_runtime_plans_validate_devices_and_properties_without_loading_libraries() {
    let root = temp("native-openvino-runtime-plans");
    let available = vec!["CPU".into(), "GPU".into(), "NPU".into()];
    let cache = root.join("cache");
    let (device, properties) =
        openvino_execution_plan("cpu", &available, &cache, 7, &BTreeMap::new()).unwrap();
    assert_eq!(device, "CPU");
    assert!(matches!(
        &properties[..],
        [OpenvinoProperty::CacheDir(_), OpenvinoProperty::InferenceNumThreads(value)] if value == "7"
    ));
    let (_, properties) =
        openvino_execution_plan("gpu", &available, &cache, 1, &BTreeMap::new()).unwrap();
    assert!(properties.contains(&OpenvinoProperty::HintInferencePrecision("f32".into())));
    let (_, properties) = openvino_execution_plan(
        "gpu",
        &available,
        &cache,
        1,
        &BTreeMap::from([("INFERENCE_PRECISION_HINT".into(), "f16".into())]),
    )
    .unwrap();
    assert!(!properties.contains(&OpenvinoProperty::HintInferencePrecision("f32".into())));
    assert!(properties.contains(&OpenvinoProperty::Other(
        "INFERENCE_PRECISION_HINT".into(),
        "f16".into()
    )));
    openvino_execution_plan("auto", &[], &cache, 1, &BTreeMap::new()).unwrap();
    assert!(openvino_execution_plan("npu", &["CPU".into()], &cache, 1, &BTreeMap::new()).is_err());
    let (_, npu_properties) =
        openvino_execution_plan("npu", &available, &cache, 1, &BTreeMap::new()).unwrap();
    assert!(
        npu_properties
            .iter()
            .any(|property| matches!(property, OpenvinoProperty::CacheDir(value) if value == cache.to_str().unwrap())),
        "NPU persistence uses OpenVINO's device-neutral compiled-model cache"
    );
    for managed in ["CACHE_DIR", "INFERENCE_NUM_THREADS"] {
        assert!(
            openvino_execution_plan(
                "cpu",
                &available,
                &cache,
                1,
                &BTreeMap::from([(managed.into(), "bad".into())]),
            )
            .is_err()
        );
    }

    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        let invalid = PathBuf::from(std::ffi::OsString::from_vec(vec![0xff]));
        assert!(openvino_execution_plan("auto", &[], &invalid, 1, &BTreeMap::new()).is_err());
    }
}

struct FakeOpenvinoCompiler {
    compilations: Arc<AtomicUsize>,
    actual_device: String,
    fail: bool,
    fail_device_query: bool,
}

struct FakeOpenvinoGraph {
    actual_device: String,
    fail_device_query: bool,
}

struct StaticPlanCompiler {
    cache_ids: Arc<Mutex<Vec<u64>>>,
    fail_graph: Option<Graph>,
    wrong_vector_shape: bool,
}

struct StaticPlanGraph {
    graph: Graph,
    input_shape: Vec<i64>,
    fail_graph: Option<Graph>,
    wrong_vector_shape: bool,
}

struct DefaultContractGraph;

impl OpenvinoGraph for DefaultContractGraph {
    fn execution_devices(&self) -> Result<String> {
        Ok("NPU".into())
    }

    fn run(&mut self, _: Graph, _: Vec<NamedTensor>, output: &'static str) -> Result<NamedTensor> {
        NamedTensor::f32(output, [1], vec![0.0])
    }
}

impl OpenvinoCompiler for StaticPlanCompiler {
    fn compile(
        &mut self,
        graph: Graph,
        _: &[u8],
        inputs: &[NamedTensor],
    ) -> Result<Box<dyn OpenvinoGraph>> {
        self.cache_ids
            .lock()
            .unwrap()
            .push(openvino_cache_blob_id(graph, inputs));
        Ok(Box::new(StaticPlanGraph {
            graph,
            input_shape: inputs[0].shape.clone(),
            fail_graph: self.fail_graph,
            wrong_vector_shape: self.wrong_vector_shape,
        }))
    }
}

impl OpenvinoGraph for StaticPlanGraph {
    fn execution_devices(&self) -> Result<String> {
        Ok("NPU".into())
    }

    fn run(&mut self, _: Graph, _: Vec<NamedTensor>, output: &'static str) -> Result<NamedTensor> {
        if self.fail_graph == Some(self.graph) {
            bail!("injected {} static-plan failure", self.graph.name());
        }
        match self.graph {
            Graph::DurationPredictor => NamedTensor::f32(output, [1], vec![0.2]),
            Graph::TextEncoder => NamedTensor::f32(output, [1, 2, NPU_TEXT_BUCKET], vec![0.0; 640]),
            Graph::VectorEstimator => {
                let mut shape = self.input_shape.clone();
                if self.wrong_vector_shape {
                    shape[2] += 1;
                }
                let values = vec![0.0; shape.iter().product::<i64>() as usize];
                NamedTensor::f32(output, shape, values)
            }
            Graph::Vocoder => NamedTensor::f32(output, [1, 1, 1], vec![0.0]),
        }
    }
}

#[test]
fn npu_preparation_compiles_the_exact_static_graph_shape_plan() {
    let root = temp("static-npu-plan");
    let model = root.join("model.onnx");
    fs::write(&model, b"model fixture").unwrap();
    let cache_ids = Arc::new(Mutex::new(Vec::new()));
    let mut pipeline = OpenvinoPipeline {
        device: "NPU".into(),
        graphs: [
            Graph::DurationPredictor,
            Graph::TextEncoder,
            Graph::VectorEstimator,
            Graph::Vocoder,
        ]
        .into_iter()
        .map(|graph| (graph, model.clone()))
        .collect(),
        compiler: Box::new(StaticPlanCompiler {
            cache_ids: cache_ids.clone(),
            fail_graph: None,
            wrong_vector_shape: false,
        }),
        compiled: HashMap::new(),
        require_cache_hits: false,
    };
    pipeline.prepare_npu_static_shapes(&frontend()).unwrap();
    assert_eq!(pipeline.compiled.len(), NPU_COMPILED_MODELS);
    let first_ids = cache_ids.lock().unwrap().clone();
    assert_eq!(first_ids.len(), NPU_COMPILED_MODELS);
    assert_eq!(
        first_ids
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        NPU_COMPILED_MODELS
    );
    let mut graph_shapes = pipeline
        .compiled
        .keys()
        .map(|key| (key.graph, key.input_shapes[0].clone()))
        .collect::<Vec<_>>();
    graph_shapes.sort_by_key(|(graph, shape)| (graph.name(), shape.clone()));
    assert!(graph_shapes.contains(&(Graph::DurationPredictor, vec![1, NPU_TEXT_BUCKET])));
    assert!(graph_shapes.contains(&(Graph::TextEncoder, vec![1, NPU_TEXT_BUCKET])));
    for bucket in NPU_LATENT_BUCKETS {
        assert!(graph_shapes.contains(&(Graph::VectorEstimator, vec![1, 4, *bucket])));
        assert!(graph_shapes.contains(&(Graph::Vocoder, vec![1, 4, *bucket])));
    }

    let repeat_ids = Arc::new(Mutex::new(Vec::new()));
    pipeline.compiler = Box::new(StaticPlanCompiler {
        cache_ids: repeat_ids.clone(),
        fail_graph: None,
        wrong_vector_shape: false,
    });
    pipeline.compiled.clear();
    pipeline.prepare_npu_static_shapes(&frontend()).unwrap();
    assert_eq!(*repeat_ids.lock().unwrap(), first_ids);
}

#[test]
fn npu_static_plan_reports_each_graph_failure_and_wrong_vector_shape() {
    let root = temp("static-npu-plan-errors");
    let model = root.join("model.onnx");
    fs::write(&model, b"model fixture").unwrap();
    let pipeline = |fail_graph, wrong_vector_shape| OpenvinoPipeline {
        device: "NPU".into(),
        graphs: [
            Graph::DurationPredictor,
            Graph::TextEncoder,
            Graph::VectorEstimator,
            Graph::Vocoder,
        ]
        .into_iter()
        .map(|graph| (graph, model.clone()))
        .collect(),
        compiler: Box::new(StaticPlanCompiler {
            cache_ids: Arc::new(Mutex::new(Vec::new())),
            fail_graph,
            wrong_vector_shape,
        }),
        compiled: HashMap::new(),
        require_cache_hits: false,
    };

    for graph in [
        Graph::DurationPredictor,
        Graph::TextEncoder,
        Graph::VectorEstimator,
        Graph::Vocoder,
    ] {
        let error = pipeline(Some(graph), false)
            .prepare_npu_static_shapes(&frontend())
            .unwrap_err();
        assert!(format!("{error:#}").contains(graph.name()));
    }
    let error = pipeline(None, true)
        .prepare_npu_static_shapes(&frontend())
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("vector estimator returned shape")
    );
}

#[test]
fn native_npu_preparation_contract_validates_plan_and_persisted_blobs_without_ffi() {
    let root = temp("native-npu-contract");
    let cache = root.join("cache");
    fs::create_dir_all(&cache).unwrap();
    let mut config = Config::default();
    let skipped = prepare_npu_cache_native_with(&config, &cache, false, || {
        unreachable!("a non-NPU configuration must fail before compilation")
    })
    .unwrap_err();
    assert!(skipped.to_string().contains("non-NPU"));

    config.backend.kind = "supertonic".into();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "npu".into();
    let compile_error = prepare_npu_cache_native_with(&config, &cache, false, || {
        bail!("injected static-plan failure")
    })
    .unwrap_err();
    assert!(compile_error.to_string().contains("static-plan failure"));
    let wrong_count =
        prepare_npu_cache_native_with(&config, &cache, false, || Ok(NPU_COMPILED_MODELS - 1))
            .unwrap_err();
    assert!(
        wrong_count
            .to_string()
            .contains(&format!("expected {NPU_COMPILED_MODELS}"))
    );
    let empty = prepare_npu_cache_native_with(&config, &cache, false, || Ok(NPU_COMPILED_MODELS))
        .unwrap_err();
    assert!(empty.to_string().contains("did not persist"));

    let nested = cache.join("device");
    fs::create_dir_all(&nested).unwrap();
    for index in 0..NPU_COMPILED_MODELS {
        fs::write(nested.join(format!("compiled-{index}.blob")), b"cache").unwrap();
    }
    let result =
        prepare_npu_cache_native_with(&config, &cache, true, || Ok(NPU_COMPILED_MODELS)).unwrap();
    assert_eq!(result.compiled_models, NPU_COMPILED_MODELS);
    assert_eq!(result.cache_blobs.len(), NPU_COMPILED_MODELS);
    assert!(result.loaded_from_cache_required);
}

#[test]
fn openvino_cache_hit_contract_and_static_shape_errors_are_injected_safely() {
    let root = temp("cache-hit-contract");
    let model = root.join("model.onnx");
    fs::write(&model, b"model fixture").unwrap();
    let graph = FakeOpenvinoGraph {
        actual_device: "NPU".into(),
        fail_device_query: false,
    };
    assert!(!graph.loaded_from_cache().unwrap());

    let mut pipeline = OpenvinoPipeline {
        device: "NPU".into(),
        graphs: HashMap::from([(Graph::Vocoder, model)]),
        compiler: Box::new(FakeOpenvinoCompiler {
            compilations: Arc::new(AtomicUsize::new(0)),
            actual_device: "NPU".into(),
            fail: false,
            fail_device_query: false,
        }),
        compiled: HashMap::new(),
        require_cache_hits: true,
    };
    let error = pipeline
        .run(
            Graph::Vocoder,
            vec![NamedTensor::f32("latent", [1, 4, 32], vec![0.0; 128]).unwrap()],
            "wav_tts",
        )
        .unwrap_err();
    assert!(error.to_string().contains("did not load the prepared"));

    assert_eq!(split_sentences(""), vec![String::new()]);
    assert_eq!(split_long_piece("a bb cccc", 4), vec!["a bb", "cccc"]);
}

impl OpenvinoCompiler for FakeOpenvinoCompiler {
    fn compile(
        &mut self,
        _: Graph,
        model: &[u8],
        _: &[NamedTensor],
    ) -> Result<Box<dyn OpenvinoGraph>> {
        self.compilations.fetch_add(1, Ordering::Relaxed);
        assert_eq!(model, b"model fixture");
        if self.fail {
            bail!("injected compiler failure");
        }
        Ok(Box::new(FakeOpenvinoGraph {
            actual_device: self.actual_device.clone(),
            fail_device_query: self.fail_device_query,
        }))
    }
}

impl OpenvinoGraph for FakeOpenvinoGraph {
    fn execution_devices(&self) -> Result<String> {
        if self.fail_device_query {
            bail!("injected placement query failure");
        }
        Ok(self.actual_device.clone())
    }

    fn run(&mut self, _: Graph, _: Vec<NamedTensor>, output: &'static str) -> Result<NamedTensor> {
        NamedTensor::f32(output, [1, 2], vec![0.25, -0.25])
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CompileFailure {
    Read,
    Shape,
    Reshape,
    Compile,
}

struct FakeCompileApi {
    failure: Option<CompileFailure>,
    events: Arc<Mutex<Vec<String>>>,
}

impl OpenvinoCompileApi for FakeCompileApi {
    type Model = Vec<u8>;
    type Shape = Vec<i64>;

    fn read_model(&mut self, bytes: &[u8]) -> Result<Self::Model> {
        self.events.lock().unwrap().push("read".into());
        if self.failure == Some(CompileFailure::Read) {
            bail!("injected read failure");
        }
        Ok(bytes.to_vec())
    }

    fn shape(&self, dimensions: &[i64]) -> Result<Self::Shape> {
        self.events
            .lock()
            .unwrap()
            .push(format!("shape:{dimensions:?}"));
        if self.failure == Some(CompileFailure::Shape) {
            bail!("injected shape failure");
        }
        Ok(dimensions.to_vec())
    }

    fn reshape(&self, model: &mut Self::Model, shapes: Vec<(&str, Self::Shape)>) -> Result<()> {
        assert_eq!(model, b"model fixture");
        self.events.lock().unwrap().push(format!(
            "reshape:{:?}",
            shapes
                .iter()
                .map(|(name, shape)| (*name, shape.clone()))
                .collect::<Vec<_>>()
        ));
        if self.failure == Some(CompileFailure::Reshape) {
            bail!("injected reshape failure");
        }
        Ok(())
    }

    fn compile(&mut self, model: &Self::Model, device: &str) -> Result<Box<dyn OpenvinoGraph>> {
        assert_eq!(model, b"model fixture");
        self.events
            .lock()
            .unwrap()
            .push(format!("compile:{device}"));
        if self.failure == Some(CompileFailure::Compile) {
            bail!("injected compile failure");
        }
        Ok(Box::new(FakeOpenvinoGraph {
            actual_device: device.into(),
            fail_device_query: false,
        }))
    }
}

#[test]
fn openvino_npu_compiler_delegates_cache_management_to_openvino() {
    let inputs = [NamedTensor::f32("latent", [1, 4, 32], vec![0.0; 128]).unwrap()];
    let cache_dir = temp("npu-explicit-export-import");
    fs::create_dir_all(&cache_dir).unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut compiler = OpenvinoCompilerAdapter {
        api: FakeCompileApi {
            failure: None,
            events: events.clone(),
        },
        device: "NPU".into(),
    };

    compiler
        .compile(Graph::Vocoder, b"model fixture", &inputs)
        .unwrap();
    assert!(cache_blobs(&cache_dir).unwrap().is_empty());
    assert!(events.lock().unwrap().contains(&"compile:NPU".into()));

    events.lock().unwrap().clear();
    compiler
        .compile(Graph::Vocoder, b"model fixture", &inputs)
        .unwrap();
    let events = events.lock().unwrap();
    assert!(events.contains(&"compile:NPU".into()));
}

#[test]
fn compiled_model_default_does_not_claim_cache_hit() {
    let graph = DefaultContractGraph;
    assert!(!graph.loaded_from_cache().unwrap());
}

#[test]
fn openvino_compiler_adapter_specializes_every_input_and_contextualizes_api_failures() {
    let inputs = [
        NamedTensor::f32("signal", [1, 2], vec![0.0, 1.0]).unwrap(),
        NamedTensor::i64("length", [1], vec![2]).unwrap(),
    ];
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut compiler = OpenvinoCompilerAdapter {
        api: FakeCompileApi {
            failure: None,
            events: events.clone(),
        },
        device: "CPU".into(),
    };
    let graph = compiler
        .compile(Graph::Vocoder, b"model fixture", &inputs)
        .unwrap();
    assert_eq!(graph.execution_devices().unwrap(), "CPU");
    assert_eq!(
        *events.lock().unwrap(),
        [
            "read",
            "shape:[1, 2]",
            "shape:[1]",
            "reshape:[(\"signal\", [1, 2]), (\"length\", [1])]",
            "compile:CPU",
        ]
    );

    for failure in [
        CompileFailure::Read,
        CompileFailure::Shape,
        CompileFailure::Reshape,
        CompileFailure::Compile,
    ] {
        let mut compiler = OpenvinoCompilerAdapter {
            api: FakeCompileApi {
                failure: Some(failure),
                events: Arc::new(Mutex::new(Vec::new())),
            },
            device: "CPU".into(),
        };
        let error = compiler
            .compile(Graph::Vocoder, b"model fixture", &inputs)
            .err()
            .expect("injected compile step must fail");
        let message = error.to_string();
        assert!(
            message.contains("import vocoder")
                || message.contains("shape failure")
                || message.contains("specialize vocoder")
                || message.contains("compile vocoder for CPU"),
            "failure {failure:?}: {message}"
        );
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuntimeApiFailure {
    Load,
    Devices,
    Property,
}

struct FakeRuntimeApi {
    failure: Option<RuntimeApiFailure>,
    devices: Vec<String>,
    properties: Arc<Mutex<Vec<(String, OpenvinoProperty)>>>,
}

impl OpenvinoRuntimeApi for FakeRuntimeApi {
    type Core = ();

    fn load(&mut self, paths: &OpenvinoRuntimePaths) -> Result<Self::Core> {
        assert!(paths.library.ends_with("libopenvino_c.so"));
        if self.failure == Some(RuntimeApiFailure::Load) {
            bail!("injected runtime load failure");
        }
        Ok(())
    }

    fn available_devices(&mut self, _: &Self::Core) -> Result<Vec<String>> {
        if self.failure == Some(RuntimeApiFailure::Devices) {
            bail!("injected device enumeration failure");
        }
        Ok(self.devices.clone())
    }

    fn set_property(
        &mut self,
        _: &mut Self::Core,
        device: &str,
        property: &OpenvinoProperty,
    ) -> Result<()> {
        self.properties
            .lock()
            .unwrap()
            .push((device.into(), property.clone()));
        if self.failure == Some(RuntimeApiFailure::Property) {
            bail!("injected property failure");
        }
        Ok(())
    }

    fn compiler(&mut self, _: Self::Core, device: String) -> Box<dyn OpenvinoCompiler> {
        Box::new(FakeOpenvinoCompiler {
            compilations: Arc::new(AtomicUsize::new(0)),
            actual_device: device,
            fail: false,
            fail_device_query: false,
        })
    }
}

#[test]
fn openvino_runtime_adapter_applies_all_property_kinds_and_preserves_failures() {
    let root = temp("fake-openvino-runtime-api");
    let paths = OpenvinoRuntimePaths {
        library: root.join("libopenvino_c.so"),
        plugins: root.join("plugins.xml"),
    };
    let properties = Arc::new(Mutex::new(Vec::new()));
    let mut api = FakeRuntimeApi {
        failure: None,
        devices: vec!["CPU".into(), "GPU".into()],
        properties: properties.clone(),
    };
    OpenvinoPipeline::create_with_api(
        &paths,
        "cpu",
        HashMap::new(),
        &root.join("cpu-cache"),
        4,
        &BTreeMap::from([("PERFORMANCE_HINT".into(), "LATENCY".into())]),
        &mut api,
    )
    .unwrap();
    assert!(matches!(
        &properties.lock().unwrap()[..],
        [
            (device, OpenvinoProperty::CacheDir(_)),
            (_, OpenvinoProperty::InferenceNumThreads(_)),
            (_, OpenvinoProperty::Other(_, _)),
        ] if device == "CPU"
    ));

    let properties = Arc::new(Mutex::new(Vec::new()));
    let mut api = FakeRuntimeApi {
        failure: None,
        devices: vec!["GPU".into()],
        properties: properties.clone(),
    };
    OpenvinoPipeline::create_with_api(
        &paths,
        "gpu",
        HashMap::new(),
        &root.join("gpu-cache"),
        1,
        &BTreeMap::new(),
        &mut api,
    )
    .unwrap();
    assert!(
        properties.lock().unwrap().iter().any(|(_, property)| {
            matches!(property, OpenvinoProperty::HintInferencePrecision(_))
        })
    );
    assert_eq!(
        probe_runtime_with_api(&paths, "auto", &mut api).unwrap(),
        ["GPU"]
    );
    assert!(probe_runtime_with_api(&paths, "npu", &mut api).is_err());

    for failure in [
        RuntimeApiFailure::Load,
        RuntimeApiFailure::Devices,
        RuntimeApiFailure::Property,
    ] {
        let mut api = FakeRuntimeApi {
            failure: Some(failure),
            devices: vec!["CPU".into()],
            properties: Arc::new(Mutex::new(Vec::new())),
        };
        assert!(
            OpenvinoPipeline::create_with_api(
                &paths,
                "cpu",
                HashMap::new(),
                &root.join(format!("failure-{failure:?}")),
                1,
                &BTreeMap::new(),
                &mut api,
            )
            .is_err()
        );
    }
}

fn fake_openvino_pipeline(
    path: PathBuf,
    device: &str,
    actual_device: &str,
    compilations: Arc<AtomicUsize>,
    fail: bool,
    fail_device_query: bool,
) -> OpenvinoPipeline {
    OpenvinoPipeline {
        device: device.into(),
        graphs: HashMap::from([(Graph::Vocoder, path)]),
        compiler: Box::new(FakeOpenvinoCompiler {
            compilations,
            actual_device: actual_device.into(),
            fail,
            fail_device_query,
        }),
        compiled: HashMap::new(),
        require_cache_hits: false,
    }
}

#[test]
fn openvino_graph_orchestration_caches_shapes_and_checks_placement_with_fake_native_objects() {
    let root = temp("fake-openvino-orchestration");
    let model = root.join("vocoder.onnx");
    fs::write(&model, b"model fixture").unwrap();
    let compilations = Arc::new(AtomicUsize::new(0));
    let mut pipeline = fake_openvino_pipeline(
        model.clone(),
        "CPU",
        "CPU",
        compilations.clone(),
        false,
        false,
    );
    let inputs = vec![NamedTensor::f32("latent", [1, 2], vec![0.0, 1.0]).unwrap()];
    let output = pipeline
        .run(Graph::Vocoder, inputs.clone(), "wav_tts")
        .unwrap();
    assert_eq!(output.into_f32().unwrap().1, [0.25, -0.25]);
    pipeline.run(Graph::Vocoder, inputs, "wav_tts").unwrap();
    assert_eq!(compilations.load(Ordering::Relaxed), 1);
    pipeline
        .run(
            Graph::Vocoder,
            vec![NamedTensor::f32("latent", [1], vec![0.0]).unwrap()],
            "wav_tts",
        )
        .unwrap();
    assert_eq!(compilations.load(Ordering::Relaxed), 2);
    assert!(
        pipeline
            .run(
                Graph::TextEncoder,
                vec![NamedTensor::f32("text", [1], vec![0.0]).unwrap()],
                "text_emb",
            )
            .is_err()
    );

    let mut missing = fake_openvino_pipeline(
        root.join("missing.onnx"),
        "CPU",
        "CPU",
        Arc::new(AtomicUsize::new(0)),
        false,
        false,
    );
    assert!(
        missing
            .run(
                Graph::Vocoder,
                vec![NamedTensor::f32("latent", [1], vec![0.0]).unwrap()],
                "wav_tts",
            )
            .is_err()
    );
    for (device, actual, fail, fail_query) in [
        ("NPU", "CPU", false, false),
        ("CPU", "CPU", true, false),
        ("CPU", "CPU", false, true),
    ] {
        let mut pipeline = fake_openvino_pipeline(
            model.clone(),
            device,
            actual,
            Arc::new(AtomicUsize::new(0)),
            fail,
            fail_query,
        );
        assert!(
            pipeline
                .run(
                    Graph::Vocoder,
                    vec![NamedTensor::f32("latent", [1], vec![0.0]).unwrap()],
                    "wav_tts",
                )
                .is_err()
        );
    }
    let mut automatic = fake_openvino_pipeline(
        model,
        "AUTO",
        "GPU.0",
        Arc::new(AtomicUsize::new(0)),
        false,
        false,
    );
    automatic
        .run(
            Graph::Vocoder,
            vec![NamedTensor::f32("latent", [1], vec![0.0]).unwrap()],
            "wav_tts",
        )
        .unwrap();
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RequestFailure {
    SetF32,
    SetI64,
    Infer,
    Output,
}

struct FakeRequestFactory {
    failure: Option<RequestFailure>,
    fail_create: bool,
    output: (Vec<i64>, TensorData),
    events: Arc<Mutex<Vec<String>>>,
    drops: Arc<AtomicUsize>,
}

struct FakeOpenvinoRequest {
    failure: Option<RequestFailure>,
    output: (Vec<i64>, TensorData),
    events: Arc<Mutex<Vec<String>>>,
    drops: Arc<AtomicUsize>,
}

struct FakeRequestApi;

impl OpenvinoRequestFactory for FakeRequestFactory {
    fn create_request(&mut self) -> Result<Box<dyn OpenvinoRequest>> {
        if self.fail_create {
            bail!("injected request creation failure");
        }
        Ok(Box::new(OpenvinoRequestAdapter::<FakeRequestApi> {
            request: FakeOpenvinoRequest {
                failure: self.failure,
                output: self.output.clone(),
                events: self.events.clone(),
                drops: self.drops.clone(),
            },
            inputs: Vec::new(),
        }))
    }
}

impl Drop for FakeOpenvinoRequest {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

impl OpenvinoRequestApi for FakeRequestApi {
    type Request = FakeOpenvinoRequest;
    type Tensor = (Vec<i64>, TensorData);

    fn f32_tensor(shape: &[i64], values: &[f32]) -> Result<Self::Tensor> {
        Ok((shape.to_vec(), TensorData::F32(values.to_vec())))
    }

    fn i64_tensor(shape: &[i64], values: &[i64]) -> Result<Self::Tensor> {
        Ok((shape.to_vec(), TensorData::I64(values.to_vec())))
    }

    fn set_tensor(
        request: &mut Self::Request,
        name: &str,
        (shape, data): &Self::Tensor,
    ) -> Result<()> {
        let (kind, values, failure) = match data {
            TensorData::F32(values) => ("f32", format!("{values:?}"), RequestFailure::SetF32),
            TensorData::I64(values) => ("i64", format!("{values:?}"), RequestFailure::SetI64),
        };
        request
            .events
            .lock()
            .unwrap()
            .push(format!("{kind}:{name}:{shape:?}:{values}"));
        if request.failure == Some(failure) {
            bail!("injected {kind} input failure");
        }
        Ok(())
    }

    fn infer(request: &mut Self::Request) -> Result<()> {
        request.events.lock().unwrap().push("infer".into());
        if request.failure == Some(RequestFailure::Infer) {
            bail!("injected inference failure");
        }
        Ok(())
    }

    fn output(request: &Self::Request, name: &str) -> Result<(Vec<i64>, TensorData)> {
        request
            .events
            .lock()
            .unwrap()
            .push(format!("output:{name}"));
        if request.failure == Some(RequestFailure::Output) {
            bail!("injected output failure");
        }
        Ok(request.output.clone())
    }
}

fn fake_request_factory(
    failure: Option<RequestFailure>,
    output: (Vec<i64>, TensorData),
    events: Arc<Mutex<Vec<String>>>,
    drops: Arc<AtomicUsize>,
) -> FakeRequestFactory {
    FakeRequestFactory {
        failure,
        fail_create: false,
        output,
        events,
        drops,
    }
}

#[test]
fn openvino_request_adapter_validates_types_shapes_and_drops_requests_on_every_exit() {
    let inputs = || {
        vec![
            NamedTensor::f32("signal", [1, 2], vec![0.25, -0.25]).unwrap(),
            NamedTensor::i64("length", [1], vec![2]).unwrap(),
        ]
    };
    let events = Arc::new(Mutex::new(Vec::new()));
    let drops = Arc::new(AtomicUsize::new(0));
    let mut factory = fake_request_factory(
        None,
        (vec![1, 2], TensorData::F32(vec![0.5, -0.5])),
        events.clone(),
        drops.clone(),
    );
    let output = run_openvino_request(&mut factory, Graph::Vocoder, inputs(), "wav").unwrap();
    assert_eq!(output.into_f32().unwrap().1, [0.5, -0.5]);
    assert_eq!(
        *events.lock().unwrap(),
        [
            "f32:signal:[1, 2]:[0.25, -0.25]",
            "i64:length:[1]:[2]",
            "infer",
            "output:wav",
        ]
    );
    assert_eq!(drops.load(Ordering::Relaxed), 1);

    for failure in [
        RequestFailure::SetF32,
        RequestFailure::SetI64,
        RequestFailure::Infer,
        RequestFailure::Output,
    ] {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut factory = fake_request_factory(
            Some(failure),
            (vec![1], TensorData::F32(vec![0.0])),
            Arc::new(Mutex::new(Vec::new())),
            drops.clone(),
        );
        assert!(run_openvino_request(&mut factory, Graph::Vocoder, inputs(), "wav").is_err());
        assert_eq!(drops.load(Ordering::Relaxed), 1, "failure: {failure:?}");
    }

    for output in [
        (vec![2], TensorData::F32(vec![0.0])),
        (vec![1], TensorData::I64(vec![0])),
    ] {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut factory = fake_request_factory(
            None,
            output,
            Arc::new(Mutex::new(Vec::new())),
            drops.clone(),
        );
        assert!(run_openvino_request(&mut factory, Graph::Vocoder, inputs(), "wav").is_err());
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }

    let mut factory = FakeRequestFactory {
        fail_create: true,
        ..fake_request_factory(
            None,
            (vec![1], TensorData::F32(vec![0.0])),
            Arc::new(Mutex::new(Vec::new())),
            Arc::new(AtomicUsize::new(0)),
        )
    };
    assert!(run_openvino_request(&mut factory, Graph::Vocoder, inputs(), "wav").is_err());
}

#[test]
fn direct_openvino_backend_uses_injected_pipeline_without_native_libraries() {
    let root = temp("injected-direct-backends");
    let (mut config, app_paths) = write_compact_assets(&root);

    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "npu".into();
    let prepared_cache = npu_cache_directory(&config, &app_paths).unwrap();
    fs::create_dir_all(&prepared_cache).unwrap();
    for index in 0..NPU_COMPILED_MODELS {
        fs::write(
            prepared_cache.join(format!("fixture-{index}.blob")),
            b"compiled",
        )
        .unwrap();
    }
    write_npu_cache_manifest(&config, &app_paths, &prepared_cache).unwrap();
    let openvino = DirectOpenvinoBackend::create_with(
        &config,
        &app_paths,
        |frontend, graphs, device, cache_dir| {
            assert_eq!(graphs.len(), 4);
            assert_eq!(device, "npu");
            assert_eq!(cache_dir, prepared_cache);
            Ok((frontend, Box::new(FakePipeline::successful())))
        },
    )
    .unwrap();
    assert_eq!(openvino.kind(), "openvino");
    assert_eq!(openvino.sample_rate(), 100);
    assert_eq!(openvino.num_voices(), 10);
    assert!(!openvino.generate("hello", 1.0, 1).unwrap().is_empty());

    config.backend.device = "cpu".into();
    let cpu_openvino =
        DirectOpenvinoBackend::create_with(&config, &app_paths, |frontend, _, device, _| {
            assert_eq!(device, "cpu");
            Ok((frontend, Box::new(FakePipeline::successful())))
        })
        .unwrap();
    assert!(!cpu_openvino.generate("hello", 1.0, 0).unwrap().is_empty());
    config.backend.device = "npu".into();

    let poisoned = DirectOpenvinoBackend::create_with(&config, &app_paths, |frontend, _, _, _| {
        Ok((frontend, Box::new(FakePipeline::successful())))
    })
    .unwrap();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = poisoned.pipeline.lock().unwrap();
        panic!("poison test pipeline");
    }));
    assert!(poisoned.generate("hello", 1.0, 0).is_err());
}

#[test]
fn installed_runtime_adapters_initialize_and_reject_invalid_graphs() {
    if std::env::var_os("OMASPEAK_RUN_NATIVE_INTEGRATION").is_none() {
        return;
    }
    let root = temp("installed-runtime-boundaries");
    let (mut config, app_paths) = write_compact_assets(&root);

    let openvino_library = PathBuf::from("/usr/lib/libopenvino_c.so");
    let openvino_plugins = PathBuf::from("/usr/lib/openvino/plugins.xml");
    if openvino_library.is_file() && openvino_plugins.is_file() {
        let runtime = OpenvinoRuntimePaths {
            library: openvino_library,
            plugins: openvino_plugins,
        };
        let devices = probe_runtime(runtime.clone(), "cpu").unwrap();
        assert!(devices.iter().any(|device| device == "CPU"));
        assert!(probe_runtime(runtime.clone(), "definitely-missing").is_err());

        let backend = DirectOpenvinoBackend::create(&config, &app_paths, runtime.clone()).unwrap();
        assert_eq!(backend.kind(), "openvino");
        assert_eq!(backend.sample_rate(), 100);
        assert_eq!(backend.num_voices(), 10);
        assert!(backend.generate("hello", 1.0, 0).is_err());

        let installed_model = std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join(".local/share/omaspeak/models/supertonic-3-openvino"));
        if let Some(installed_model) =
            installed_model.filter(|path| path.join("voice_styles/M1.json").is_file())
        {
            let mut real = Config::default();
            crate::catalog::model("supertonic-3-openvino")
                .unwrap()
                .activate(&mut real);
            real.backend.runtime = Runtime::Openvino;
            real.backend.device = "cpu".into();
            real.model.directory = installed_model.display().to_string();
            let real_backend =
                DirectOpenvinoBackend::create(&real, &app_paths, runtime.clone()).unwrap();
            let audio = real_backend.generate("coverage test", 1.0, 0).unwrap();
            assert!(audio.len() > 1_000);
            assert!(audio.iter().all(|sample| sample.is_finite()));
        }

        config
            .backend
            .options
            .insert("CACHE_DIR".into(), "elsewhere".into());
        assert!(DirectOpenvinoBackend::create(&config, &app_paths, runtime.clone()).is_err());
        config.backend.options.clear();
        fs::create_dir_all(&app_paths.state_dir).unwrap();
        let _ = fs::remove_dir_all(app_paths.state_dir.join("cache"));
        fs::write(app_paths.state_dir.join("cache"), b"blocked").unwrap();
        let backend = DirectOpenvinoBackend::create(&config, &app_paths, runtime).unwrap();
        assert!(backend.generate("hello", 1.0, 0).is_err());
    }
}
