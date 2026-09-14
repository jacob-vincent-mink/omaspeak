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
    config.model.unicode_indexer = "unicode.bin".into();
    config.model.voice_style = "voice.bin".into();
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
        model.join("unicode.bin"),
        (0_i32..2048).flat_map(i32::to_le_bytes).collect::<Vec<_>>(),
    )
    .unwrap();
    let mut voice = [2_i64, 2, 2, 2, 1, 2]
        .into_iter()
        .flat_map(i64::to_le_bytes)
        .collect::<Vec<_>>();
    voice.extend((0..12).flat_map(|value| (value as f32).to_le_bytes()));
    fs::write(model.join("voice.bin"), voice).unwrap();
    (config, app_paths)
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
    assert!(loaded.style.slice(2).is_err());
}

#[test]
fn compact_asset_parsers_reject_corruption() {
    let root = temp("bad-assets");
    let indexer = root.join("index.bin");
    fs::write(&indexer, []).unwrap();
    assert!(read_indexer(&indexer).is_err());
    fs::write(&indexer, [1, 2, 3]).unwrap();
    assert!(read_indexer(&indexer).is_err());
    let voice = root.join("voice.bin");
    fs::write(&voice, [0; 47]).unwrap();
    assert!(read_voice_style(&voice).is_err());
    let mut invalid = [0_i64, 1, 1, 1, 1, 1]
        .into_iter()
        .flat_map(i64::to_le_bytes)
        .collect::<Vec<_>>();
    invalid.extend(0_f32.to_le_bytes());
    fs::write(&voice, &invalid).unwrap();
    assert!(read_voice_style(&voice).is_err());
    let mut mismatch = [1_i64; 6]
        .into_iter()
        .flat_map(i64::to_le_bytes)
        .collect::<Vec<_>>();
    mismatch.extend(0_f32.to_le_bytes());
    fs::write(&voice, &mismatch).unwrap();
    assert!(read_voice_style(&voice).is_err());

    let overflow = [i64::MAX, i64::MAX, 2, i64::MAX, i64::MAX, 2]
        .into_iter()
        .flat_map(i64::to_le_bytes)
        .collect::<Vec<_>>();
    fs::write(&voice, overflow).unwrap();
    assert!(read_voice_style(&voice).is_err());

    let duration_overflow = [1_i64, 1, 1, 1, i64::MAX, i64::MAX]
        .into_iter()
        .flat_map(i64::to_le_bytes)
        .collect::<Vec<_>>();
    fs::write(&voice, duration_overflow).unwrap();
    assert!(read_voice_style(&voice).is_err());
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

    let runtime = OnnxRuntimePaths {
        onnxruntime: root.join("missing-libonnxruntime.so"),
        provider: None,
    };
    assert!(probe_onnx_runtime(&runtime, Runtime::Default).is_err());
    let error = DirectOrtBackend::create(&config, &app_paths, runtime, Runtime::Default)
        .err()
        .expect("missing ONNX Runtime should fail");
    assert!(
        error
            .to_string()
            .contains("ONNX Runtime library is missing")
    );
}

#[test]
fn native_runtime_plans_validate_files_devices_and_provider_options_without_loading_libraries() {
    let root = temp("native-runtime-plans");
    let ort = root.join("libonnxruntime.so");
    let cuda = root.join("libonnxruntime_providers_cuda.so");
    fs::write(&ort, b"fixture").unwrap();
    fs::write(&cuda, b"fixture").unwrap();

    let mut paths = OnnxRuntimePaths {
        onnxruntime: ort,
        provider: None,
    };
    validate_onnx_runtime_paths(&paths, Runtime::Default).unwrap();
    assert!(validate_onnx_runtime_paths(&paths, Runtime::Cuda).is_err());
    assert!(validate_onnx_runtime_paths(&paths, Runtime::Openvino).is_err());
    paths.provider = Some(root.join("missing-provider.so"));
    assert!(validate_onnx_runtime_paths(&paths, Runtime::Cuda).is_err());
    paths.provider = Some(cuda);
    validate_onnx_runtime_paths(&paths, Runtime::Cuda).unwrap();
    paths.onnxruntime = root.join("missing-ort.so");
    assert!(validate_onnx_runtime_paths(&paths, Runtime::Default).is_err());

    let options = BTreeMap::from([("gpu_mem_limit".into(), "1024".into())]);
    let configured = cuda_provider_options(3, &options).unwrap();
    assert_eq!(configured["device_id"], "3");
    assert_eq!(configured["cudnn_conv_algo_search"], "HEURISTIC");
    assert_eq!(configured["gpu_mem_limit"], "1024");
    let explicit = cuda_provider_options(
        0,
        &BTreeMap::from([("cudnn_conv_algo_search".into(), "EXHAUSTIVE".into())]),
    )
    .unwrap();
    assert_eq!(explicit["cudnn_conv_algo_search"], "EXHAUSTIVE");
    assert!(cuda_provider_options(0, &BTreeMap::from([("device_id".into(), "9".into())])).is_err());

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

#[derive(Default)]
struct FakeOrtRuntime {
    initialized: Vec<(Runtime, u32, BTreeMap<String, String>)>,
    built: Vec<(Graph, PathBuf, u16)>,
    runs: Arc<AtomicUsize>,
    fail_initialize: bool,
    fail_build: Option<Graph>,
}

struct FakeOrtGraph {
    graph: Graph,
    runs: Arc<AtomicUsize>,
}

impl OrtRuntimeAdapter for FakeOrtRuntime {
    fn initialize(
        &mut self,
        paths: &OnnxRuntimePaths,
        runtime: Runtime,
        device_id: u32,
        options: &BTreeMap<String, String>,
    ) -> Result<()> {
        assert!(paths.onnxruntime.is_file());
        self.initialized.push((runtime, device_id, options.clone()));
        if self.fail_initialize {
            bail!("injected runtime initialization failure");
        }
        Ok(())
    }

    fn build_session(
        &mut self,
        graph: Graph,
        path: &Path,
        threads: u16,
    ) -> Result<Box<dyn OrtGraph>> {
        self.built.push((graph, path.to_owned(), threads));
        if self.fail_build == Some(graph) {
            bail!("injected graph build failure");
        }
        Ok(Box::new(FakeOrtGraph {
            graph,
            runs: self.runs.clone(),
        }))
    }
}

impl OrtGraph for FakeOrtGraph {
    fn run(
        &mut self,
        graph: Graph,
        inputs: Vec<NamedTensor>,
        output: &'static str,
    ) -> Result<NamedTensor> {
        assert_eq!(graph, self.graph);
        assert_eq!(inputs.len(), 2);
        self.runs.fetch_add(1, Ordering::Relaxed);
        NamedTensor::f32(output, [1], vec![0.5])
    }
}

struct FakeLowOrtApi;

struct FakeLowOrtSession {
    threads: u16,
    provider_options: usize,
}

impl OrtRuntimeApi for FakeLowOrtApi {
    type ExecutionProvider = BTreeMap<String, String>;
    type Session = FakeLowOrtSession;
    type Value = (Vec<i64>, TensorData);

    fn initialize(path: &Path) -> Result<()> {
        if path.file_name().is_some_and(|name| name == "fail-init.so") {
            bail!("injected initialization failure");
        }
        Ok(())
    }

    fn cuda_available(path: &Path) -> Result<bool> {
        Ok(!path.file_name().is_some_and(|name| name == "no-cuda.so"))
    }

    fn cuda_provider(options: &BTreeMap<String, String>) -> Result<Self::ExecutionProvider> {
        if options.contains_key("fail_provider") {
            bail!("injected provider failure");
        }
        Ok(options.clone())
    }

    fn build_session(
        path: &Path,
        threads: u16,
        provider: Option<&Self::ExecutionProvider>,
    ) -> Result<Self::Session> {
        if path
            .file_name()
            .is_some_and(|name| name == "fail-build.onnx")
        {
            bail!("injected session build failure");
        }
        Ok(FakeLowOrtSession {
            threads,
            provider_options: provider.map_or(0, BTreeMap::len),
        })
    }

    fn f32_value(shape: Vec<i64>, values: Vec<f32>) -> Result<Self::Value> {
        Ok((shape, TensorData::F32(values)))
    }

    fn i64_value(shape: Vec<i64>, values: Vec<i64>) -> Result<Self::Value> {
        Ok((shape, TensorData::I64(values)))
    }

    fn run_values(
        session: &mut Self::Session,
        _: Graph,
        inputs: Vec<(&'static str, Self::Value)>,
        output: &'static str,
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        assert!(matches!(&inputs[0].1.1, TensorData::F32(_)));
        assert!(matches!(&inputs[1].1.1, TensorData::I64(_)));
        let _ = output;
        Ok((
            vec![2],
            vec![session.threads as f32, session.provider_options as f32],
        ))
    }
}

#[test]
fn ort_runtime_adapter_shares_initialization_provider_builder_and_session_semantics() {
    let root = temp("fake-low-ort-api");
    let library = root.join("libonnxruntime.so");
    let provider = root.join("libonnxruntime_providers_cuda.so");
    fs::write(&library, b"runtime").unwrap();
    fs::write(&provider, b"provider").unwrap();
    let paths = OnnxRuntimePaths {
        onnxruntime: library,
        provider: Some(provider),
    };
    let mut adapter = OrtRuntimeAdapterImpl::<FakeLowOrtApi> {
        _api: PhantomData,
        execution_provider: None,
        runtime: Runtime::Cuda,
    };
    let mut pipeline = OrtPipeline::create_with_adapter(
        paths.clone(),
        Runtime::Cuda,
        HashMap::from([(Graph::Vocoder, root.join("vocoder.onnx"))]),
        5,
        2,
        &BTreeMap::from([("gpu_mem_limit".into(), "2048".into())]),
        &mut adapter,
    )
    .unwrap();
    let values = pipeline
        .run(
            Graph::Vocoder,
            vec![
                NamedTensor::f32("signal", [1], vec![0.5]).unwrap(),
                NamedTensor::i64("length", [1], vec![1]).unwrap(),
            ],
            "wav",
        )
        .unwrap()
        .into_f32()
        .unwrap()
        .1;
    assert_eq!(values, [5.0, 3.0]);

    let mut adapter = OrtRuntimeAdapterImpl::<FakeLowOrtApi> {
        _api: PhantomData,
        execution_provider: None,
        runtime: Runtime::Cuda,
    };
    assert!(
        OrtPipeline::create_with_adapter(
            paths.clone(),
            Runtime::Cuda,
            HashMap::new(),
            1,
            0,
            &BTreeMap::from([("fail_provider".into(), "1".into())]),
            &mut adapter,
        )
        .is_err()
    );
    let no_cuda = root.join("no-cuda.so");
    fs::write(&no_cuda, b"runtime").unwrap();
    let no_cuda_paths = OnnxRuntimePaths {
        onnxruntime: no_cuda,
        provider: paths.provider,
    };
    probe_onnx_runtime_with_api::<FakeLowOrtApi>(&no_cuda_paths, Runtime::Default).unwrap();
    assert!(probe_onnx_runtime_with_api::<FakeLowOrtApi>(&no_cuda_paths, Runtime::Cuda).is_err());
}

#[test]
fn ort_session_orchestration_initializes_builds_dispatches_and_contextualizes_failures() {
    let root = temp("fake-ort-orchestration");
    let library = root.join("libonnxruntime.so");
    let provider = root.join("libonnxruntime_providers_cuda.so");
    fs::write(&library, b"runtime").unwrap();
    fs::write(&provider, b"provider").unwrap();
    let paths = OnnxRuntimePaths {
        onnxruntime: library,
        provider: Some(provider),
    };
    let graphs = HashMap::from([
        (Graph::TextEncoder, root.join("text.onnx")),
        (Graph::Vocoder, root.join("vocoder.onnx")),
    ]);
    let options = BTreeMap::from([("gpu_mem_limit".into(), "1024".into())]);
    let mut adapter = FakeOrtRuntime::default();
    let mut pipeline = OrtPipeline::create_with_adapter(
        paths.clone(),
        Runtime::Cuda,
        graphs,
        7,
        3,
        &options,
        &mut adapter,
    )
    .unwrap();
    assert_eq!(adapter.initialized, [(Runtime::Cuda, 3, options)]);
    assert_eq!(adapter.built.len(), 2);
    assert!(adapter.built.iter().all(|(_, _, threads)| *threads == 7));
    let output = pipeline
        .run(
            Graph::Vocoder,
            vec![
                NamedTensor::f32("latent", [1], vec![0.0]).unwrap(),
                NamedTensor::i64("length", [1], vec![1]).unwrap(),
            ],
            "wav",
        )
        .unwrap();
    assert_eq!(output.into_f32().unwrap().1, [0.5]);
    assert_eq!(adapter.runs.load(Ordering::Relaxed), 1);
    assert!(
        pipeline
            .run(Graph::DurationPredictor, Vec::new(), "duration")
            .is_err()
    );

    let mut adapter = FakeOrtRuntime {
        fail_initialize: true,
        ..Default::default()
    };
    assert!(
        OrtPipeline::create_with_adapter(
            paths.clone(),
            Runtime::Default,
            HashMap::new(),
            1,
            0,
            &BTreeMap::new(),
            &mut adapter,
        )
        .is_err()
    );
    assert!(adapter.built.is_empty());

    let mut adapter = FakeOrtRuntime {
        fail_build: Some(Graph::Vocoder),
        ..Default::default()
    };
    let error = OrtPipeline::create_with_adapter(
        paths,
        Runtime::Default,
        HashMap::from([(Graph::Vocoder, root.join("broken.onnx"))]),
        1,
        0,
        &BTreeMap::new(),
        &mut adapter,
    )
    .err()
    .expect("injected build must fail");
    assert!(error.to_string().contains("load vocoder graph"));
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
        device: "NPU".into(),
    };
    let graph = compiler
        .compile(Graph::Vocoder, b"model fixture", &inputs)
        .unwrap();
    assert_eq!(graph.execution_devices().unwrap(), "NPU");
    assert_eq!(
        *events.lock().unwrap(),
        [
            "read",
            "shape:[1, 2]",
            "shape:[1]",
            "reshape:[(\"signal\", [1, 2]), (\"length\", [1])]",
            "compile:NPU",
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
fn direct_backends_use_injected_runtime_pipelines_without_native_libraries() {
    let root = temp("injected-direct-backends");
    let (mut config, app_paths) = write_compact_assets(&root);

    config.backend.runtime = Runtime::Default;
    config.backend.device = "cpu".into();
    let ort =
        DirectOrtBackend::create_with(&config, &app_paths, Runtime::Default, |frontend, graphs| {
            assert_eq!(graphs.len(), 4);
            Ok((frontend, Box::new(FakePipeline::successful())))
        })
        .unwrap();
    assert_eq!(ort.kind(), "onnxruntime");
    assert_eq!(ort.sample_rate(), 100);
    assert_eq!(ort.num_voices(), 2);
    assert!(!ort.generate("hello", 1.0, 0).unwrap().is_empty());

    let poisoned_ort =
        DirectOrtBackend::create_with(&config, &app_paths, Runtime::Default, |frontend, _| {
            Ok((frontend, Box::new(FakePipeline::successful())))
        })
        .unwrap();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = poisoned_ort.pipeline.lock().unwrap();
        panic!("poison test ONNX Runtime pipeline");
    }));
    assert!(poisoned_ort.generate("hello", 1.0, 0).is_err());

    let cuda = DirectOrtBackend::create_with(&config, &app_paths, Runtime::Cuda, |frontend, _| {
        Ok((frontend, Box::new(FakePipeline::successful())))
    })
    .unwrap();
    assert_eq!(cuda.kind(), "cuda");

    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "npu".into();
    let openvino = DirectOpenvinoBackend::create_with(
        &config,
        &app_paths,
        |frontend, graphs, device, cache_dir| {
            assert_eq!(graphs.len(), 4);
            assert_eq!(device, "npu");
            assert!(cache_dir.ends_with("cache/openvino/npu"));
            Ok((frontend, Box::new(FakePipeline::successful())))
        },
    )
    .unwrap();
    assert_eq!(openvino.kind(), "openvino");
    assert_eq!(openvino.sample_rate(), 100);
    assert_eq!(openvino.num_voices(), 2);
    assert!(!openvino.generate("hello", 1.0, 1).unwrap().is_empty());

    let poisoned = DirectOpenvinoBackend::create_with(&config, &app_paths, |frontend, _, _, _| {
        Ok((frontend, Box::new(FakePipeline::successful())))
    })
    .unwrap();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = poisoned.pipeline.lock().unwrap();
        panic!("poison test pipeline");
    }));
    assert!(poisoned.generate("hello", 1.0, 0).is_err());

    assert!(
        DirectOrtBackend::create_with(&config, &app_paths, Runtime::Openvino, |_, _| bail!(
            "pipeline rejected runtime"
        ),)
        .is_err()
    );
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
        assert_eq!(backend.num_voices(), 2);
        assert!(backend.generate("hello", 1.0, 0).is_err());

        let installed_model = std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join(".local/share/omaspeak/models/supertonic-3-int8"));
        if let Some(installed_model) =
            installed_model.filter(|path| path.join("voice.bin").is_file())
        {
            let mut real = Config::default();
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
        assert!(DirectOpenvinoBackend::create(&config, &app_paths, runtime).is_err());
    }

    let ort = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/debug/libonnxruntime.so");
    if ort.is_file() {
        let runtime = OnnxRuntimePaths {
            onnxruntime: ort,
            provider: None,
        };
        probe_onnx_runtime(&runtime, Runtime::Default).unwrap();
        let error = probe_onnx_runtime(&runtime, Runtime::Cuda).unwrap_err();
        assert!(error.to_string().contains("provider is not configured"));
        let missing_provider = OnnxRuntimePaths {
            onnxruntime: runtime.onnxruntime.clone(),
            provider: Some(root.join("missing-cuda-provider.so")),
        };
        assert!(probe_onnx_runtime(&missing_provider, Runtime::Cuda).is_err());
        assert!(
            DirectOrtBackend::create(&config, &app_paths, runtime.clone(), Runtime::Default)
                .is_err()
        );
        assert!(DirectOrtBackend::create(&config, &app_paths, runtime, Runtime::Openvino).is_err());
    }
}
