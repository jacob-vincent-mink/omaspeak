//! Runtime-independent Supertonic frontend and direct OpenVINO execution.
//!
//! Portions adapted from Supertonic, Copyright (c) 2025 Supertone Inc.,
//! licensed under MIT. See `THIRD_PARTY_NOTICES.md`.
//!
//! The text and synthesis flow is based on the MIT-licensed Supertonic 3
//! reference frontend. Model execution is kept behind [`ModelPipeline`] so a
//! future runtime can reuse the frontend without duplicating TTS behavior.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Read;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow, bail};
use openvino::{
    CompiledModel, Core, DeviceType, ElementType, InferRequest, Model, PartialShape, PropertyKey,
    RwPropertyKey, Shape, Tensor,
};
use ort::ep::{
    ArbitrarilyConfigurableExecutionProvider, ExecutionProvider, ExecutionProviderDispatch,
};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::{DynValue, Tensor as OrtTensor};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use unicode_normalization::UnicodeNormalization;

use crate::backend::Runtime;
use crate::config::Config;
use crate::engine::TtsBackend;
use crate::paths::AppPaths;
use crate::runtime::{OnnxRuntimePaths, OpenvinoRuntimePaths};

const LANGUAGES: &[&str] = &[
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu",
    "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi",
];
const MAX_LATENT_LENGTH: i64 = 10_000;
const MIN_DURATION_SECONDS: f32 = 0.1;
pub const NPU_TEXT_BUCKET: i64 = 320;
pub const NPU_LATENT_BUCKETS: &[i64] = &[32, 64, 128, 256, 512];
pub const NPU_COMPILED_MODELS: usize = 2 + 2 * NPU_LATENT_BUCKETS.len();
const NPU_CACHE_SCHEMA: u32 = 1;
const NPU_MANIFEST: &str = "omaspeak-npu-cache.json";

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Graph {
    DurationPredictor,
    TextEncoder,
    VectorEstimator,
    Vocoder,
}

impl Graph {
    const fn name(self) -> &'static str {
        match self {
            Self::DurationPredictor => "duration_predictor",
            Self::TextEncoder => "text_encoder",
            Self::VectorEstimator => "vector_estimator",
            Self::Vocoder => "vocoder",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

#[derive(Clone, Debug, PartialEq)]
struct NamedTensor {
    name: &'static str,
    shape: Vec<i64>,
    data: TensorData,
}

impl NamedTensor {
    fn f32(name: &'static str, shape: impl Into<Vec<i64>>, data: Vec<f32>) -> Result<Self> {
        Self::new(name, shape.into(), TensorData::F32(data))
    }

    fn i64(name: &'static str, shape: impl Into<Vec<i64>>, data: Vec<i64>) -> Result<Self> {
        Self::new(name, shape.into(), TensorData::I64(data))
    }

    fn new(name: &'static str, shape: Vec<i64>, data: TensorData) -> Result<Self> {
        if shape.is_empty() || shape.iter().any(|&dimension| dimension <= 0) {
            bail!("tensor {name} has invalid shape {shape:?}");
        }
        let count = shape.iter().try_fold(1_usize, |count, &dimension| {
            count.checked_mul(dimension as usize)
        });
        let actual = match &data {
            TensorData::F32(values) => values.len(),
            TensorData::I64(values) => values.len(),
        };
        if count != Some(actual) {
            bail!("tensor {name} shape {shape:?} needs {count:?} values, got {actual}");
        }
        Ok(Self { name, shape, data })
    }

    fn into_f32(self) -> Result<(Vec<i64>, Vec<f32>)> {
        match self.data {
            TensorData::F32(data) => Ok((self.shape, data)),
            TensorData::I64(_) => bail!("tensor {} is not f32", self.name),
        }
    }
}

trait ModelPipeline {
    fn run(
        &mut self,
        graph: Graph,
        inputs: Vec<NamedTensor>,
        output: &'static str,
    ) -> Result<NamedTensor>;
}

struct OrtPipeline {
    sessions: HashMap<Graph, Box<dyn OrtGraph>>,
}

trait OrtRuntimeAdapter {
    fn initialize(
        &mut self,
        paths: &OnnxRuntimePaths,
        runtime: Runtime,
        device_id: u32,
        options: &BTreeMap<String, String>,
    ) -> Result<()>;

    fn build_session(
        &mut self,
        graph: Graph,
        path: &Path,
        threads: u16,
    ) -> Result<Box<dyn OrtGraph>>;
}

trait OrtGraph: Send {
    fn run(
        &mut self,
        graph: Graph,
        inputs: Vec<NamedTensor>,
        output: &'static str,
    ) -> Result<NamedTensor>;
}

struct OrtRuntimeAdapterImpl<A: OrtRuntimeApi> {
    _api: PhantomData<A>,
    execution_provider: Option<A::ExecutionProvider>,
    runtime: Runtime,
}

trait OrtRuntimeApi: Send + 'static {
    type ExecutionProvider: Clone + Send;
    type Session: Send;
    type Value;

    fn initialize(path: &Path) -> Result<()>;
    fn register_cuda(_path: &Path) -> Result<()> {
        Ok(())
    }
    fn cuda_available(path: &Path) -> Result<bool>;
    fn cuda_provider(options: &BTreeMap<String, String>) -> Result<Self::ExecutionProvider>;
    fn build_session(
        path: &Path,
        threads: u16,
        provider: Option<&Self::ExecutionProvider>,
    ) -> Result<Self::Session>;
    fn f32_value(shape: Vec<i64>, values: Vec<f32>) -> Result<Self::Value>;
    fn i64_value(shape: Vec<i64>, values: Vec<i64>) -> Result<Self::Value>;
    fn run_values(
        session: &mut Self::Session,
        graph: Graph,
        inputs: Vec<(&'static str, Self::Value)>,
        output: &'static str,
    ) -> Result<(Vec<i64>, Vec<f32>)>;
}

struct NativeOrtApi;

struct OrtGraphAdapter<A: OrtRuntimeApi> {
    session: A::Session,
    api: PhantomData<A>,
}

pub(crate) fn probe_onnx_runtime(paths: &OnnxRuntimePaths, runtime: Runtime) -> Result<()> {
    probe_onnx_runtime_with_api::<NativeOrtApi>(paths, runtime)
}

fn probe_onnx_runtime_with_api<A: OrtRuntimeApi>(
    paths: &OnnxRuntimePaths,
    runtime: Runtime,
) -> Result<()> {
    validate_onnx_runtime_paths(paths, runtime)?;
    A::initialize(&paths.onnxruntime)?;
    if runtime == Runtime::Cuda {
        let provider = paths
            .provider
            .as_deref()
            .context("CUDA provider is not configured")?;
        A::register_cuda(provider)?;
        if !A::cuda_available(&paths.onnxruntime)? {
            bail!("selected ONNX Runtime does not expose the CUDA execution provider");
        }
    }
    Ok(())
}

fn validate_onnx_runtime_paths(paths: &OnnxRuntimePaths, runtime: Runtime) -> Result<()> {
    if !paths.onnxruntime.is_file() {
        bail!(
            "ONNX Runtime library is missing: {}",
            paths.onnxruntime.display()
        );
    }
    match runtime {
        Runtime::Default => Ok(()),
        Runtime::Cuda => {
            let provider = paths
                .provider
                .as_ref()
                .context("CUDA provider is not configured")?;
            if !provider.is_file() {
                bail!("CUDA provider library is missing: {}", provider.display());
            }
            Ok(())
        }
        Runtime::Openvino => bail!("OpenVINO does not use the ONNX Runtime pipeline"),
    }
}

fn cuda_provider_options(
    device_id: u32,
    options: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>> {
    let mut configured = BTreeMap::from([("device_id".into(), device_id.to_string())]);
    if !options.contains_key("cudnn_conv_algo_search") {
        configured.insert("cudnn_conv_algo_search".into(), "HEURISTIC".into());
    }
    for (key, value) in options {
        if key == "device_id" {
            bail!("backend.options.device_id is managed by backend.device_id");
        }
        configured.insert(key.clone(), value.clone());
    }
    Ok(configured)
}

impl OrtPipeline {
    fn create(
        paths: OnnxRuntimePaths,
        runtime: Runtime,
        graphs: HashMap<Graph, PathBuf>,
        threads: u16,
        device_id: u32,
        options: &std::collections::BTreeMap<String, String>,
    ) -> Result<Self> {
        Self::create_with_adapter(
            paths,
            runtime,
            graphs,
            threads,
            device_id,
            options,
            &mut OrtRuntimeAdapterImpl::<NativeOrtApi> {
                _api: PhantomData,
                execution_provider: None,
                runtime,
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create_with_adapter(
        paths: OnnxRuntimePaths,
        runtime: Runtime,
        graphs: HashMap<Graph, PathBuf>,
        threads: u16,
        device_id: u32,
        options: &BTreeMap<String, String>,
        adapter: &mut dyn OrtRuntimeAdapter,
    ) -> Result<Self> {
        validate_onnx_runtime_paths(&paths, runtime)?;
        adapter.initialize(&paths, runtime, device_id, options)?;
        let mut sessions = HashMap::new();
        for (graph, path) in graphs {
            let session = adapter
                .build_session(graph, &path, threads)
                .with_context(|| format!("load {} graph {}", graph.name(), path.display()))?;
            sessions.insert(graph, session);
        }
        Ok(Self { sessions })
    }
}

impl<A: OrtRuntimeApi> OrtRuntimeAdapter for OrtRuntimeAdapterImpl<A> {
    fn initialize(
        &mut self,
        paths: &OnnxRuntimePaths,
        runtime: Runtime,
        device_id: u32,
        options: &BTreeMap<String, String>,
    ) -> Result<()> {
        A::initialize(&paths.onnxruntime)?;
        self.execution_provider = match runtime {
            Runtime::Default => None,
            Runtime::Cuda => {
                A::register_cuda(
                    paths
                        .provider
                        .as_deref()
                        .context("CUDA provider is not configured")?,
                )?;
                let configured = cuda_provider_options(device_id, options)?;
                Some(A::cuda_provider(&configured)?)
            }
            Runtime::Openvino => bail!("OpenVINO does not use the ONNX Runtime pipeline"),
        };
        Ok(())
    }

    fn build_session(
        &mut self,
        _graph: Graph,
        path: &Path,
        threads: u16,
    ) -> Result<Box<dyn OrtGraph>> {
        let session = A::build_session(path, threads, self.execution_provider.as_ref())
            .with_context(|| format!("configure {} session", self.runtime.capability()))?;
        Ok(Box::new(OrtGraphAdapter::<A> {
            session,
            api: PhantomData,
        }))
    }
}

impl OrtRuntimeApi for NativeOrtApi {
    type ExecutionProvider = ExecutionProviderDispatch;
    type Session = Session;
    type Value = DynValue;

    fn initialize(path: &Path) -> Result<()> {
        crate::runtime_inventory::initialize_ort(path)
    }

    fn register_cuda(path: &Path) -> Result<()> {
        static PROVIDER: Mutex<Option<PathBuf>> = Mutex::new(None);
        let exact = path.canonicalize().context("resolve CUDA provider")?;
        let mut registered = PROVIDER
            .lock()
            .map_err(|_| anyhow!("CUDA provider registration lock poisoned"))?;
        if let Some(previous) = registered.as_ref() {
            if previous != &exact {
                bail!(
                    "CUDA provider changed from {} to {}; restart the process before changing native runtimes",
                    previous.display(),
                    exact.display()
                );
            }
            return Ok(());
        }
        ort::environment::Environment::current()?
            .register_ep_library("omaspeak-cuda", &exact)
            .context("register selected CUDA provider library")?;
        *registered = Some(exact);
        Ok(())
    }

    fn cuda_available(_path: &Path) -> Result<bool> {
        ort::ep::CUDA::default()
            .is_available()
            .context("query ONNX Runtime CUDA execution provider")
    }

    fn cuda_provider(options: &BTreeMap<String, String>) -> Result<Self::ExecutionProvider> {
        let mut cuda = ort::ep::CUDA::default();
        for (key, value) in options {
            cuda = cuda.with_arbitrary_config(key, value);
        }
        Ok(cuda.build().error_on_failure())
    }

    fn build_session(
        path: &Path,
        threads: u16,
        provider: Option<&Self::ExecutionProvider>,
    ) -> Result<Self::Session> {
        let mut builder = Session::builder()
            .context("create ONNX Runtime session builder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| anyhow!("configure ONNX Runtime graph optimization: {error}"))?
            .with_intra_threads(threads.into())
            .map_err(|error| anyhow!("configure ONNX Runtime thread count: {error}"))?;
        if let Some(provider) = provider {
            builder = builder
                .with_execution_providers([provider.clone()])
                .map_err(|error| anyhow!("configure execution provider: {error}"))?;
        }
        builder.commit_from_file(path).map_err(Into::into)
    }

    fn f32_value(shape: Vec<i64>, values: Vec<f32>) -> Result<Self::Value> {
        Ok(OrtTensor::from_array((shape, values))?.into_dyn())
    }

    fn i64_value(shape: Vec<i64>, values: Vec<i64>) -> Result<Self::Value> {
        Ok(OrtTensor::from_array((shape, values))?.into_dyn())
    }

    fn run_values(
        session: &mut Self::Session,
        graph: Graph,
        inputs: Vec<(&'static str, Self::Value)>,
        output: &'static str,
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        let outputs = session
            .run(inputs)
            .with_context(|| format!("run {} with ONNX Runtime", graph.name()))?;
        let value = outputs
            .get(output)
            .with_context(|| format!("get {} output {output}", graph.name()))?;
        let (shape, data) = value
            .try_extract_tensor::<f32>()
            .with_context(|| format!("read {} output {output}", graph.name()))?;
        Ok((shape.iter().copied().collect(), data.to_vec()))
    }
}

impl<A: OrtRuntimeApi> OrtGraph for OrtGraphAdapter<A> {
    fn run(
        &mut self,
        graph: Graph,
        inputs: Vec<NamedTensor>,
        output: &'static str,
    ) -> Result<NamedTensor> {
        let inputs = inputs
            .into_iter()
            .map(|input| {
                let value = match input.data {
                    TensorData::F32(values) => A::f32_value(input.shape, values),
                    TensorData::I64(values) => A::i64_value(input.shape, values),
                }?;
                Ok((input.name, value))
            })
            .collect::<Result<Vec<_>>>()?;
        let (shape, values) = A::run_values(&mut self.session, graph, inputs, output)?;
        NamedTensor::f32(output, shape, values)
    }
}

impl ModelPipeline for OrtPipeline {
    fn run(
        &mut self,
        graph: Graph,
        inputs: Vec<NamedTensor>,
        output: &'static str,
    ) -> Result<NamedTensor> {
        self.sessions
            .get_mut(&graph)
            .with_context(|| format!("{} ONNX graph is not loaded", graph.name()))?
            .run(graph, inputs, output)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct GraphKey {
    graph: Graph,
    input_shapes: Vec<Vec<i64>>,
}

struct OpenvinoPipeline {
    device: String,
    graphs: HashMap<Graph, PathBuf>,
    compiler: Box<dyn OpenvinoCompiler>,
    compiled: HashMap<GraphKey, Box<dyn OpenvinoGraph>>,
    require_cache_hits: bool,
}

trait OpenvinoCompiler: Send {
    fn compile(
        &mut self,
        graph: Graph,
        model: &[u8],
        inputs: &[NamedTensor],
    ) -> Result<Box<dyn OpenvinoGraph>>;
}

trait OpenvinoGraph: Send {
    fn execution_devices(&self) -> Result<String>;
    fn loaded_from_cache(&self) -> Result<bool> {
        Ok(false)
    }
    fn run(
        &mut self,
        graph: Graph,
        inputs: Vec<NamedTensor>,
        output: &'static str,
    ) -> Result<NamedTensor>;
}

struct OpenvinoCompilerAdapter<A> {
    api: A,
    device: String,
}

trait OpenvinoCompileApi: Send {
    type Model;
    type Shape;

    fn read_model(&mut self, bytes: &[u8]) -> Result<Self::Model>;
    fn shape(&self, dimensions: &[i64]) -> Result<Self::Shape>;
    fn reshape(&self, model: &mut Self::Model, shapes: Vec<(&str, Self::Shape)>) -> Result<()>;
    fn compile(&mut self, model: &Self::Model, device: &str) -> Result<Box<dyn OpenvinoGraph>>;
}

struct NativeOpenvinoCompileApi(Core);

struct NativeOpenvinoGraph(CompiledModel);

trait OpenvinoRequestFactory {
    fn create_request(&mut self) -> Result<Box<dyn OpenvinoRequest>>;
}

trait OpenvinoRequest {
    fn set_f32(&mut self, name: &str, shape: &[i64], values: &[f32]) -> Result<()>;
    fn set_i64(&mut self, name: &str, shape: &[i64], values: &[i64]) -> Result<()>;
    fn infer(&mut self) -> Result<()>;
    fn output(&self, name: &str) -> Result<(Vec<i64>, TensorData)>;
}

struct OpenvinoRequestAdapter<A: OpenvinoRequestApi> {
    request: A::Request,
    inputs: Vec<A::Tensor>,
}

trait OpenvinoRequestApi {
    type Request;
    type Tensor;

    fn f32_tensor(shape: &[i64], values: &[f32]) -> Result<Self::Tensor>;
    fn i64_tensor(shape: &[i64], values: &[i64]) -> Result<Self::Tensor>;
    fn set_tensor(request: &mut Self::Request, name: &str, tensor: &Self::Tensor) -> Result<()>;
    fn infer(request: &mut Self::Request) -> Result<()>;
    fn output(request: &Self::Request, name: &str) -> Result<(Vec<i64>, TensorData)>;
}

struct NativeOpenvinoRequestApi;

trait OpenvinoRuntimeApi {
    type Core;

    fn load(&mut self, paths: &OpenvinoRuntimePaths) -> Result<Self::Core>;
    fn available_devices(&mut self, core: &Self::Core) -> Result<Vec<String>>;
    fn set_property(
        &mut self,
        core: &mut Self::Core,
        device: &str,
        property: &OpenvinoProperty,
    ) -> Result<()>;
    fn compiler(&mut self, core: Self::Core, device: String) -> Box<dyn OpenvinoCompiler>;
}

struct NativeOpenvinoRuntimeApi;

#[derive(Clone, Debug, Eq, PartialEq)]
enum OpenvinoProperty {
    CacheDir(String),
    InferenceNumThreads(String),
    HintInferencePrecision(String),
    Other(String, String),
}

fn openvino_execution_plan(
    device: &str,
    available: &[String],
    cache_dir: &Path,
    threads: u16,
    options: &BTreeMap<String, String>,
) -> Result<(String, Vec<OpenvinoProperty>)> {
    let device = device.to_ascii_uppercase();
    if device != "AUTO" && !available.iter().any(|item| item == &device) {
        bail!(
            "OpenVINO runtime is installed, but device {device} is not accessible; available devices: {}",
            available.join(", ")
        );
    }
    let cache = cache_dir
        .to_str()
        .context("OpenVINO cache path is not valid UTF-8")?;
    let mut properties = vec![OpenvinoProperty::CacheDir(cache.into())];
    if device == "CPU" {
        properties.push(OpenvinoProperty::InferenceNumThreads(threads.to_string()));
    }
    if device == "GPU" && !options.contains_key("INFERENCE_PRECISION_HINT") {
        properties.push(OpenvinoProperty::HintInferencePrecision("f32".into()));
    }
    for (key, value) in options {
        if key == "CACHE_DIR" || key == "INFERENCE_NUM_THREADS" {
            bail!("backend.options.{key} is managed by Omaspeak");
        }
        properties.push(OpenvinoProperty::Other(key.clone(), value.clone()));
    }
    Ok((device, properties))
}

impl OpenvinoPipeline {
    fn create(
        paths: OpenvinoRuntimePaths,
        device: &str,
        graphs: HashMap<Graph, PathBuf>,
        cache_dir: &Path,
        threads: u16,
        options: &std::collections::BTreeMap<String, String>,
    ) -> Result<Self> {
        Self::create_with_api(
            &paths,
            device,
            graphs,
            cache_dir,
            threads,
            options,
            &mut NativeOpenvinoRuntimeApi,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create_with_api<A: OpenvinoRuntimeApi>(
        paths: &OpenvinoRuntimePaths,
        device: &str,
        graphs: HashMap<Graph, PathBuf>,
        cache_dir: &Path,
        threads: u16,
        options: &BTreeMap<String, String>,
        api: &mut A,
    ) -> Result<Self> {
        let mut core = api.load(paths)?;
        let available = api
            .available_devices(&core)
            .context("enumerate OpenVINO devices")?;
        fs::create_dir_all(cache_dir)
            .with_context(|| format!("create OpenVINO cache {}", cache_dir.display()))?;
        let (device, properties) =
            openvino_execution_plan(device, &available, cache_dir, threads, options)?;
        for property in properties {
            let context = match &property {
                OpenvinoProperty::CacheDir(_) => "configure OpenVINO compiled-model cache".into(),
                OpenvinoProperty::InferenceNumThreads(_) => {
                    "configure OpenVINO CPU thread count".into()
                }
                OpenvinoProperty::HintInferencePrecision(_) => {
                    "configure OpenVINO GPU inference precision".into()
                }
                OpenvinoProperty::Other(key, _) => format!("set OpenVINO property {key}"),
            };
            api.set_property(&mut core, &device, &property)
                .with_context(|| context)?;
        }
        Ok(Self {
            compiler: api.compiler(core, device.clone()),
            device,
            graphs,
            compiled: HashMap::new(),
            require_cache_hits: false,
        })
    }

    fn compile(
        &mut self,
        graph: Graph,
        inputs: &[NamedTensor],
    ) -> Result<&mut Box<dyn OpenvinoGraph>> {
        let key = GraphKey {
            graph,
            input_shapes: inputs.iter().map(|input| input.shape.clone()).collect(),
        };
        if !self.compiled.contains_key(&key) {
            let path = self
                .graphs
                .get(&graph)
                .with_context(|| format!("{} graph is not configured", graph.name()))?;
            let bytes = fs::read(path).with_context(|| format!("read model {}", path.display()))?;
            let compiled = self.compiler.compile(graph, &bytes, inputs)?;
            let actual = compiled
                .execution_devices()
                .with_context(|| format!("query {} execution device", graph.name()))?;
            if self.device != "AUTO" && !execution_device_matches(&self.device, &actual) {
                bail!(
                    "OpenVINO compiled {} for unexpected device {actual}; requested {}",
                    graph.name(),
                    self.device
                );
            }
            if self.require_cache_hits
                && !compiled
                    .loaded_from_cache()
                    .with_context(|| format!("query {} compiled-model cache state", graph.name()))?
            {
                bail!(
                    "OpenVINO did not load the prepared {} model from cache",
                    graph.name()
                );
            }
            self.compiled.insert(key.clone(), compiled);
        }
        self.compiled
            .get_mut(&key)
            .context("compiled OpenVINO graph cache insertion failed")
    }
}

impl OpenvinoRuntimeApi for NativeOpenvinoRuntimeApi {
    type Core = Core;

    fn load(&mut self, paths: &OpenvinoRuntimePaths) -> Result<Self::Core> {
        openvino_sys::library::load_from(&paths.library).map_err(|error| {
            anyhow!(
                "load OpenVINO C library {}: {error}",
                paths.library.display()
            )
        })?;
        Core::new_with_config(
            paths
                .plugins
                .to_str()
                .context("OpenVINO plugins path is not valid UTF-8")?,
        )
        .with_context(|| format!("load OpenVINO plugins {}", paths.plugins.display()))
    }

    fn available_devices(&mut self, core: &Self::Core) -> Result<Vec<String>> {
        Ok(core
            .available_devices()?
            .into_iter()
            .map(|item| item.to_string())
            .collect())
    }

    fn set_property(
        &mut self,
        core: &mut Self::Core,
        device: &str,
        property: &OpenvinoProperty,
    ) -> Result<()> {
        let (key, value) = match property {
            OpenvinoProperty::CacheDir(value) => (RwPropertyKey::CacheDir, value),
            OpenvinoProperty::InferenceNumThreads(value) => {
                (RwPropertyKey::InferenceNumThreads, value)
            }
            OpenvinoProperty::HintInferencePrecision(value) => {
                (RwPropertyKey::HintInferencePrecision, value)
            }
            OpenvinoProperty::Other(key, value) => {
                (RwPropertyKey::Other(Cow::Owned(key.clone())), value)
            }
        };
        core.set_property(&DeviceType::from(device), &key, value)
            .map_err(Into::into)
    }

    fn compiler(&mut self, core: Self::Core, device: String) -> Box<dyn OpenvinoCompiler> {
        Box::new(OpenvinoCompilerAdapter {
            api: NativeOpenvinoCompileApi(core),
            device,
        })
    }
}

impl<A: OpenvinoCompileApi> OpenvinoCompiler for OpenvinoCompilerAdapter<A> {
    fn compile(
        &mut self,
        graph: Graph,
        model: &[u8],
        inputs: &[NamedTensor],
    ) -> Result<Box<dyn OpenvinoGraph>> {
        let mut model = self
            .api
            .read_model(model)
            .with_context(|| format!("import {} into OpenVINO", graph.name()))?;
        let shapes = inputs
            .iter()
            .map(|input| Ok((input.name, self.api.shape(&input.shape)?)))
            .collect::<Result<Vec<_>>>()?;
        self.api
            .reshape(&mut model, shapes)
            .with_context(|| format!("specialize {} input shapes", graph.name()))?;
        self.api
            .compile(&model, &self.device)
            .with_context(|| format!("compile {} for {}", graph.name(), self.device))
    }
}

impl OpenvinoCompileApi for NativeOpenvinoCompileApi {
    type Model = Model;
    type Shape = PartialShape;

    fn read_model(&mut self, bytes: &[u8]) -> Result<Self::Model> {
        self.0
            .read_model_from_buffer(bytes, None)
            .map_err(Into::into)
    }

    fn shape(&self, dimensions: &[i64]) -> Result<Self::Shape> {
        PartialShape::new_static(dimensions.len() as i64, dimensions).map_err(Into::into)
    }

    fn reshape(&self, model: &mut Self::Model, shapes: Vec<(&str, Self::Shape)>) -> Result<()> {
        let shapes = shapes
            .iter()
            .map(|(name, shape)| (*name, shape))
            .collect::<Vec<_>>();
        model.reshape(&shapes).map_err(Into::into)
    }

    fn compile(&mut self, model: &Self::Model, device: &str) -> Result<Box<dyn OpenvinoGraph>> {
        Ok(Box::new(NativeOpenvinoGraph(
            self.0.compile_model(model, DeviceType::from(device))?,
        )))
    }
}

impl OpenvinoGraph for NativeOpenvinoGraph {
    fn execution_devices(&self) -> Result<String> {
        self.0
            .get_property(&PropertyKey::Other(Cow::Borrowed("EXECUTION_DEVICES")))
            .context("query OpenVINO execution devices")
            .map(Cow::into_owned)
    }

    fn loaded_from_cache(&self) -> Result<bool> {
        let value = self
            .0
            .get_property(&PropertyKey::Other(Cow::Borrowed("LOADED_FROM_CACHE")))
            .context("query OpenVINO compiled-model cache state")?;
        Ok(matches!(
            value.trim().to_ascii_uppercase().as_str(),
            "YES" | "TRUE" | "1"
        ))
    }

    fn run(
        &mut self,
        graph: Graph,
        inputs: Vec<NamedTensor>,
        output: &'static str,
    ) -> Result<NamedTensor> {
        run_openvino_request(self, graph, inputs, output)
    }
}

impl OpenvinoRequestFactory for NativeOpenvinoGraph {
    fn create_request(&mut self) -> Result<Box<dyn OpenvinoRequest>> {
        Ok(Box::new(
            OpenvinoRequestAdapter::<NativeOpenvinoRequestApi> {
                request: self.0.create_infer_request()?,
                inputs: Vec::new(),
            },
        ))
    }
}

impl<A: OpenvinoRequestApi> OpenvinoRequest for OpenvinoRequestAdapter<A> {
    fn set_f32(&mut self, name: &str, shape: &[i64], values: &[f32]) -> Result<()> {
        let tensor = A::f32_tensor(shape, values)?;
        A::set_tensor(&mut self.request, name, &tensor)?;
        self.inputs.push(tensor);
        Ok(())
    }

    fn set_i64(&mut self, name: &str, shape: &[i64], values: &[i64]) -> Result<()> {
        let tensor = A::i64_tensor(shape, values)?;
        A::set_tensor(&mut self.request, name, &tensor)?;
        self.inputs.push(tensor);
        Ok(())
    }

    fn infer(&mut self) -> Result<()> {
        A::infer(&mut self.request)
    }

    fn output(&self, name: &str) -> Result<(Vec<i64>, TensorData)> {
        A::output(&self.request, name)
    }
}

impl OpenvinoRequestApi for NativeOpenvinoRequestApi {
    type Request = InferRequest;
    type Tensor = Tensor;

    fn f32_tensor(shape: &[i64], values: &[f32]) -> Result<Self::Tensor> {
        let mut tensor = Tensor::new(ElementType::F32, &Shape::new(shape)?)?;
        tensor.get_data_mut::<f32>()?.copy_from_slice(values);
        Ok(tensor)
    }

    fn i64_tensor(shape: &[i64], values: &[i64]) -> Result<Self::Tensor> {
        let mut tensor = Tensor::new(ElementType::I64, &Shape::new(shape)?)?;
        tensor.get_data_mut::<i64>()?.copy_from_slice(values);
        Ok(tensor)
    }

    fn set_tensor(request: &mut Self::Request, name: &str, tensor: &Self::Tensor) -> Result<()> {
        request.set_tensor(name, tensor).map_err(Into::into)
    }

    fn infer(request: &mut Self::Request) -> Result<()> {
        request.infer().map_err(Into::into)
    }

    fn output(request: &Self::Request, name: &str) -> Result<(Vec<i64>, TensorData)> {
        let tensor = request.get_tensor(name)?;
        Ok((
            tensor.get_shape()?.get_dimensions().to_vec(),
            TensorData::F32(tensor.get_data::<f32>()?.to_vec()),
        ))
    }
}

fn run_openvino_request(
    factory: &mut dyn OpenvinoRequestFactory,
    graph: Graph,
    inputs: Vec<NamedTensor>,
    output: &'static str,
) -> Result<NamedTensor> {
    let mut request = factory.create_request()?;
    for input in &inputs {
        match &input.data {
            TensorData::F32(values) => request.set_f32(input.name, &input.shape, values),
            TensorData::I64(values) => request.set_i64(input.name, &input.shape, values),
        }
        .with_context(|| format!("set {} input {}", graph.name(), input.name))?;
    }
    request
        .infer()
        .with_context(|| format!("run {} with OpenVINO", graph.name()))?;
    let (shape, data) = request
        .output(output)
        .with_context(|| format!("get {} output {output}", graph.name()))?;
    let (shape, values) = NamedTensor::new(output, shape, data)?.into_f32()?;
    NamedTensor::f32(output, shape, values)
}

fn execution_device_matches(requested: &str, actual: &str) -> bool {
    actual
        .split(|character: char| character.is_whitespace() || matches!(character, ',' | '[' | ']'))
        .any(|item| item == requested || item.starts_with(&format!("{requested}.")))
}

pub(crate) fn probe_runtime(paths: OpenvinoRuntimePaths, device: &str) -> Result<Vec<String>> {
    probe_runtime_with_api(&paths, device, &mut NativeOpenvinoRuntimeApi)
}

fn probe_runtime_with_api<A: OpenvinoRuntimeApi>(
    paths: &OpenvinoRuntimePaths,
    device: &str,
    api: &mut A,
) -> Result<Vec<String>> {
    let core = api.load(paths)?;
    let available = api
        .available_devices(&core)
        .context("enumerate OpenVINO devices")?;
    let requested = device.to_ascii_uppercase();
    if requested != "AUTO" && !available.iter().any(|item| item == &requested) {
        bail!(
            "OpenVINO runtime is installed, but device {requested} is not accessible; available devices: {}",
            available.join(", ")
        );
    }
    Ok(available)
}

impl ModelPipeline for OpenvinoPipeline {
    fn run(
        &mut self,
        graph: Graph,
        inputs: Vec<NamedTensor>,
        output: &'static str,
    ) -> Result<NamedTensor> {
        self.compile(graph, &inputs)?.run(graph, inputs, output)
    }
}

impl OpenvinoPipeline {
    fn prepare_npu_static_shapes(&mut self, frontend: &SupertonicFrontend) -> Result<()> {
        let style = frontend.style.slice(0)?;
        let (mut text_ids, mut text_mask) = process_text(
            "Omaspeak NPU cache preparation.",
            &frontend.language,
            &frontend.indexer,
        );
        text_ids.resize(NPU_TEXT_BUCKET as usize, 0);
        text_mask.resize(NPU_TEXT_BUCKET as usize, 0.0);
        self.run(
            Graph::DurationPredictor,
            vec![
                NamedTensor::i64("text_ids", [1, NPU_TEXT_BUCKET], text_ids.clone())?,
                NamedTensor::f32("style_dp", style.dp_shape, style.dp.to_vec())?,
                NamedTensor::f32("text_mask", [1, 1, NPU_TEXT_BUCKET], text_mask.clone())?,
            ],
            "duration",
        )?;
        let (text_embedding_shape, text_embedding) = self
            .run(
                Graph::TextEncoder,
                vec![
                    NamedTensor::i64("text_ids", [1, NPU_TEXT_BUCKET], text_ids)?,
                    NamedTensor::f32("style_ttl", style.ttl_shape, style.ttl.to_vec())?,
                    NamedTensor::f32("text_mask", [1, 1, NPU_TEXT_BUCKET], text_mask.clone())?,
                ],
                "text_emb",
            )?
            .into_f32()?;
        let latent_dimension =
            frontend.config.ttl.latent_dim * frontend.config.ttl.chunk_compress_factor;
        for &bucket in NPU_LATENT_BUCKETS {
            let latent_shape = [1, latent_dimension, bucket];
            let latent = vec![0.0; (latent_dimension * bucket) as usize];
            let denoised = self.run(
                Graph::VectorEstimator,
                vec![
                    NamedTensor::f32("noisy_latent", latent_shape, latent)?,
                    NamedTensor::f32(
                        "text_emb",
                        text_embedding_shape.clone(),
                        text_embedding.clone(),
                    )?,
                    NamedTensor::f32("style_ttl", style.ttl_shape, style.ttl.to_vec())?,
                    NamedTensor::f32("latent_mask", [1, 1, bucket], vec![1.0; bucket as usize])?,
                    NamedTensor::f32("text_mask", [1, 1, NPU_TEXT_BUCKET], text_mask.clone())?,
                    NamedTensor::f32("current_step", [1], vec![0.0])?,
                    NamedTensor::f32("total_step", [1], vec![frontend.steps as f32])?,
                ],
                "denoised_latent",
            )?;
            let (shape, latent) = denoised.into_f32()?;
            if shape != latent_shape {
                bail!("vector estimator returned shape {shape:?} while preparing {latent_shape:?}");
            }
            self.run(
                Graph::Vocoder,
                vec![NamedTensor::f32("latent", latent_shape, latent)?],
                "wav_tts",
            )?;
        }
        Ok(())
    }
}

#[derive(Deserialize)]
struct TtsFileConfig {
    ae: AeConfig,
    ttl: TtlConfig,
}

#[derive(Deserialize)]
struct AeConfig {
    sample_rate: i32,
    base_chunk_size: i64,
}

#[derive(Deserialize)]
struct TtlConfig {
    chunk_compress_factor: i64,
    latent_dim: i64,
}

struct VoiceStyle {
    speakers: usize,
    ttl_shape: [i64; 3],
    ttl: Vec<f32>,
    dp_shape: [i64; 3],
    dp: Vec<f32>,
}

impl VoiceStyle {
    fn slice(&self, speaker: usize) -> Result<StyleSlice<'_>> {
        if speaker >= self.speakers {
            bail!(
                "voice {speaker} is outside the Supertonic speaker range 0..{}",
                self.speakers
            );
        }
        let ttl_size = (self.ttl_shape[1] * self.ttl_shape[2]) as usize;
        let dp_size = (self.dp_shape[1] * self.dp_shape[2]) as usize;
        Ok(StyleSlice {
            ttl_shape: [1, self.ttl_shape[1], self.ttl_shape[2]],
            ttl: &self.ttl[speaker * ttl_size..(speaker + 1) * ttl_size],
            dp_shape: [1, self.dp_shape[1], self.dp_shape[2]],
            dp: &self.dp[speaker * dp_size..(speaker + 1) * dp_size],
        })
    }
}

struct StyleSlice<'a> {
    ttl_shape: [i64; 3],
    ttl: &'a [f32],
    dp_shape: [i64; 3],
    dp: &'a [f32],
}

struct SupertonicFrontend {
    config: TtsFileConfig,
    indexer: Vec<i32>,
    style: VoiceStyle,
    language: String,
    steps: i32,
    seed: Option<u64>,
    silence_seconds: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct NpuCacheManifest {
    schema: u32,
    fingerprint: String,
    text_bucket: i64,
    latent_buckets: Vec<i64>,
    compiled_models: usize,
    blobs: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NpuCacheState {
    pub required: bool,
    pub ready: bool,
    pub fingerprint: Option<String>,
    pub directory: Option<PathBuf>,
    pub blobs: Vec<PathBuf>,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NpuNativePreparation {
    pub compiled_models: usize,
    pub cache_blobs: Vec<PathBuf>,
    pub loaded_from_cache_required: bool,
}

pub fn uses_static_npu_shapes(config: &Config) -> bool {
    config.backend.runtime == Runtime::Openvino
        && config.backend.device.trim().eq_ignore_ascii_case("npu")
}

fn fingerprint_contents(hasher: &mut Sha256, path: &Path) -> Result<()> {
    hasher.update(path.to_string_lossy().as_bytes());
    let mut input = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        let count = input
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(())
}

pub fn npu_cache_fingerprint(config: &Config, paths: &AppPaths) -> Result<String> {
    if !uses_static_npu_shapes(config) {
        bail!("Intel NPU cache preparation requires runtime=openvino and device=npu");
    }
    let mut hasher = Sha256::new();
    hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
    hasher.update(NPU_CACHE_SCHEMA.to_le_bytes());
    hasher.update(NPU_TEXT_BUCKET.to_le_bytes());
    for bucket in NPU_LATENT_BUCKETS {
        hasher.update(bucket.to_le_bytes());
    }
    hasher.update(config.model.name.as_bytes());
    hasher.update(serde_json::to_vec(&config.backend.options)?);
    let directory = config.model_directory(paths);
    let configured_files = [
        &config.model.duration_predictor,
        &config.model.text_encoder,
        &config.model.vector_estimator,
        &config.model.vocoder,
        &config.model.tts_json,
        &config.model.unicode_indexer,
        &config.model.voice_style,
    ];
    // Cache identity must follow the bytes OpenVINO will compile. File size and
    // timestamps are insufficient: a model can be replaced in place while
    // preserving both, which would otherwise allow an incompatible blob to be
    // accepted as ready.
    for name in configured_files {
        fingerprint_contents(&mut hasher, &directory.join(name))?;
    }
    if let Some(spec) = crate::catalog::model(&config.model.name) {
        hasher.update(spec.source_revision.as_bytes());
        for file in spec.required_files {
            hasher.update(file.path.as_bytes());
            hasher.update(file.size.to_le_bytes());
            hasher.update(file.sha256.as_bytes());
        }
    }
    let runtime = crate::runtime::discover(&config.backend, &paths.config_file);
    for path in [
        runtime.openvino_library.as_deref(),
        runtime.openvino_plugins.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        fingerprint_contents(&mut hasher, path)?;
        if path.file_name().is_some_and(|name| name == "plugins.xml")
            && let Some(parent) = path.parent()
        {
            let npu_plugin = parent.join("libopenvino_intel_npu_plugin.so");
            if npu_plugin.is_file() {
                fingerprint_contents(&mut hasher, &npu_plugin)?;
            }
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn npu_cache_directory(config: &Config, paths: &AppPaths) -> Result<PathBuf> {
    Ok(paths
        .cache_dir
        .join("openvino/npu/static-v1")
        .join(npu_cache_fingerprint(config, paths)?))
}

pub(crate) fn cache_blobs(directory: &Path) -> Result<Vec<PathBuf>> {
    fn walk(root: &Path, directory: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
        for entry in fs::read_dir(directory)
            .with_context(|| format!("read OpenVINO cache {}", directory.display()))?
        {
            let path = entry?.path();
            if path.is_dir() {
                walk(root, &path, output)?;
            } else if path
                .extension()
                .is_some_and(|extension| extension == "blob")
            {
                output.push(path.strip_prefix(root)?.to_owned());
            }
        }
        Ok(())
    }
    let mut blobs = Vec::new();
    if directory.is_dir() {
        walk(directory, directory, &mut blobs)?;
    }
    blobs.sort();
    Ok(blobs)
}

pub fn npu_cache_state(config: &Config, paths: &AppPaths) -> NpuCacheState {
    if !uses_static_npu_shapes(config) {
        return NpuCacheState {
            required: false,
            ready: true,
            fingerprint: None,
            directory: None,
            blobs: Vec::new(),
            detail: "static NPU cache is not required for this runtime/device".into(),
        };
    }
    let fingerprint = match npu_cache_fingerprint(config, paths) {
        Ok(fingerprint) => fingerprint,
        Err(error) => {
            return NpuCacheState {
                required: true,
                ready: false,
                fingerprint: None,
                directory: None,
                blobs: Vec::new(),
                detail: format!("{error:#}"),
            };
        }
    };
    let directory = paths
        .cache_dir
        .join("openvino/npu/static-v1")
        .join(&fingerprint);
    let attempted = (|| -> Result<NpuCacheState> {
        let manifest_path = directory.join(NPU_MANIFEST);
        let manifest: NpuCacheManifest =
            serde_json::from_slice(&fs::read(&manifest_path).with_context(|| {
                format!("read NPU cache manifest {}", manifest_path.display())
            })?)?;
        if manifest.schema != NPU_CACHE_SCHEMA
            || manifest.fingerprint != fingerprint
            || manifest.text_bucket != NPU_TEXT_BUCKET
            || manifest.latent_buckets != NPU_LATENT_BUCKETS
            || manifest.compiled_models != NPU_COMPILED_MODELS
        {
            bail!("NPU cache manifest does not match the active runtime, model, and shape plan");
        }
        if manifest.blobs.len() != NPU_COMPILED_MODELS {
            bail!(
                "NPU cache manifest records {} compiled-model blobs; expected {}",
                manifest.blobs.len(),
                NPU_COMPILED_MODELS
            );
        }
        let relative_blobs = manifest.blobs.iter().map(PathBuf::from).collect::<Vec<_>>();
        if relative_blobs.iter().any(|path| {
            path.is_absolute()
                || path
                    .components()
                    .any(|part| matches!(part, std::path::Component::ParentDir))
        }) {
            bail!("NPU cache manifest contains an unsafe compiled-model path");
        }
        let mut unique = relative_blobs.clone();
        unique.sort();
        unique.dedup();
        if unique.len() != NPU_COMPILED_MODELS {
            bail!("NPU cache manifest contains duplicate compiled-model blobs");
        }
        let blobs = relative_blobs
            .iter()
            .map(|relative| directory.join(relative))
            .collect::<Vec<_>>();
        if blobs.iter().any(|path| {
            !path.is_file() || fs::metadata(path).is_ok_and(|metadata| metadata.len() == 0)
        }) {
            bail!("NPU cache manifest references missing or empty compiled-model blobs");
        }
        if cache_blobs(&directory)? != unique {
            bail!("NPU cache directory does not exactly match its compiled-model manifest");
        }
        Ok(NpuCacheState {
            required: true,
            ready: true,
            fingerprint: Some(fingerprint.clone()),
            directory: Some(directory.clone()),
            detail: format!("{} prepared OpenVINO cache blobs", blobs.len()),
            blobs,
        })
    })();
    attempted.unwrap_or_else(|error| NpuCacheState {
        required: true,
        ready: false,
        fingerprint: Some(fingerprint),
        directory: Some(directory),
        blobs: Vec::new(),
        detail: format!("{error:#}"),
    })
}

pub fn write_npu_cache_manifest(
    config: &Config,
    paths: &AppPaths,
    directory: &Path,
) -> Result<Vec<PathBuf>> {
    let fingerprint = npu_cache_fingerprint(config, paths)?;
    let blobs = cache_blobs(directory)?;
    if blobs.len() != NPU_COMPILED_MODELS {
        bail!(
            "OpenVINO NPU preparation created {} .blob files in {}; expected {}",
            blobs.len(),
            directory.display(),
            NPU_COMPILED_MODELS
        );
    }
    if blobs
        .iter()
        .any(|path| fs::metadata(directory.join(path)).is_ok_and(|metadata| metadata.len() == 0))
    {
        bail!("OpenVINO NPU preparation created an empty compiled-model blob");
    }
    let manifest = NpuCacheManifest {
        schema: NPU_CACHE_SCHEMA,
        fingerprint,
        text_bucket: NPU_TEXT_BUCKET,
        latent_buckets: NPU_LATENT_BUCKETS.to_vec(),
        compiled_models: NPU_COMPILED_MODELS,
        blobs: blobs
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect(),
    };
    let temporary = directory.join(format!("{NPU_MANIFEST}.tmp"));
    fs::write(&temporary, serde_json::to_vec_pretty(&manifest)?)?;
    fs::rename(temporary, directory.join(NPU_MANIFEST))?;
    Ok(blobs)
}

pub(crate) fn prepare_npu_cache_native(
    config: &Config,
    paths: &AppPaths,
    runtime: OpenvinoRuntimePaths,
    cache_dir: &Path,
    require_cache_hits: bool,
) -> Result<NpuNativePreparation> {
    prepare_npu_cache_native_with(config, cache_dir, require_cache_hits, || {
        let (frontend, graphs) = SupertonicFrontend::load(config, paths)?;
        let mut pipeline = OpenvinoPipeline::create(
            runtime,
            "NPU",
            graphs,
            cache_dir,
            config.backend.threads,
            &config.backend.options,
        )?;
        pipeline.require_cache_hits = require_cache_hits;
        pipeline.prepare_npu_static_shapes(&frontend)?;
        let compiled_models = pipeline.compiled.len();
        // OpenVINO may finish serializing a compiled model when its last handle is
        // released. Drop every compiled-model handle before inspecting the cache.
        drop(pipeline);
        Ok(compiled_models)
    })
}

fn prepare_npu_cache_native_with(
    config: &Config,
    cache_dir: &Path,
    require_cache_hits: bool,
    compile_static_plan: impl FnOnce() -> Result<usize>,
) -> Result<NpuNativePreparation> {
    if !uses_static_npu_shapes(config) {
        bail!("refusing NPU cache preparation for a non-NPU configuration");
    }
    let compiled_models = compile_static_plan()?;
    if compiled_models != NPU_COMPILED_MODELS {
        bail!(
            "NPU shape preparation compiled {} models; expected {}",
            compiled_models,
            NPU_COMPILED_MODELS
        );
    }
    let blobs = cache_blobs(cache_dir)?;
    if blobs.is_empty() {
        bail!("OpenVINO did not persist any compiled NPU cache blobs");
    }
    Ok(NpuNativePreparation {
        compiled_models,
        cache_blobs: blobs,
        loaded_from_cache_required: require_cache_hits,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SynthesisShapes {
    Exact,
    NpuStatic,
}

fn npu_latent_bucket(length: i64) -> Result<i64> {
    NPU_LATENT_BUCKETS
        .iter()
        .copied()
        .find(|bucket| length <= *bucket)
        .with_context(|| {
            format!(
                "predicted latent length {length} exceeds the prepared Intel NPU limit of {}",
                NPU_LATENT_BUCKETS.last().copied().unwrap_or_default()
            )
        })
}

fn zero_latent_padding(values: &mut [f32], channels: i64, used: i64, bucket: i64) {
    for channel in 0..channels as usize {
        let start = channel * bucket as usize + used as usize;
        let end = (channel + 1) * bucket as usize;
        values[start..end].fill(0.0);
    }
}

impl SupertonicFrontend {
    fn load(config: &Config, paths: &AppPaths) -> Result<(Self, HashMap<Graph, PathBuf>)> {
        if config.model.family != "supertonic" {
            bail!(
                "direct OpenVINO supports Supertonic models; configured family is {:?}",
                config.model.family
            );
        }
        if !LANGUAGES.contains(&config.model.language.as_str()) {
            bail!(
                "Supertonic language {:?} is unsupported; use one of {}",
                config.model.language,
                LANGUAGES.join(", ")
            );
        }
        if config.model.steps <= 0 {
            bail!("Supertonic generation steps must be greater than zero");
        }
        let directory = config.model_directory(paths);
        let required = |name: &str| -> Result<PathBuf> {
            if name.trim().is_empty() {
                bail!("required Supertonic asset path is not configured");
            }
            let path = directory.join(name);
            if !path.is_file() {
                bail!("required Supertonic asset is missing: {}", path.display());
            }
            Ok(path)
        };
        let tts_json = required(&config.model.tts_json)?;
        let file_config: TtsFileConfig = serde_json::from_slice(
            &fs::read(&tts_json).with_context(|| format!("read {}", tts_json.display()))?,
        )
        .with_context(|| format!("parse {}", tts_json.display()))?;
        if file_config.ae.sample_rate <= 0
            || file_config.ae.base_chunk_size <= 0
            || file_config.ttl.chunk_compress_factor <= 0
            || file_config.ttl.latent_dim <= 0
        {
            bail!("Supertonic tts.json contains non-positive synthesis dimensions");
        }
        let indexer = read_indexer(&required(&config.model.unicode_indexer)?)?;
        let style = read_voice_style(&required(&config.model.voice_style)?)?;
        let seed = config
            .model
            .options
            .get("seed")
            .map(|value| {
                value
                    .parse()
                    .context("model.options.seed must be an unsigned integer")
            })
            .transpose()?;
        let silence_seconds: f32 = config
            .model
            .options
            .get("silence_duration")
            .map(|value| {
                value
                    .parse()
                    .context("model.options.silence_duration must be a number")
            })
            .transpose()?
            .unwrap_or(0.3);
        if !silence_seconds.is_finite() || !(0.0..=10.0).contains(&silence_seconds) {
            bail!("model.options.silence_duration must be between 0 and 10 seconds");
        }
        let graphs = HashMap::from([
            (
                Graph::DurationPredictor,
                required(&config.model.duration_predictor)?,
            ),
            (Graph::TextEncoder, required(&config.model.text_encoder)?),
            (
                Graph::VectorEstimator,
                required(&config.model.vector_estimator)?,
            ),
            (Graph::Vocoder, required(&config.model.vocoder)?),
        ]);
        Ok((
            Self {
                config: file_config,
                indexer,
                style,
                language: config.model.language.clone(),
                steps: config.model.steps,
                seed,
                silence_seconds,
            },
            graphs,
        ))
    }

    fn generate(
        &self,
        pipeline: &mut (impl ModelPipeline + ?Sized),
        text: &str,
        speed: f32,
        voice: i32,
    ) -> Result<Vec<f32>> {
        self.generate_with_shapes(pipeline, text, speed, voice, SynthesisShapes::Exact)
    }

    fn generate_npu(
        &self,
        pipeline: &mut (impl ModelPipeline + ?Sized),
        text: &str,
        speed: f32,
        voice: i32,
    ) -> Result<Vec<f32>> {
        self.generate_with_shapes(pipeline, text, speed, voice, SynthesisShapes::NpuStatic)
    }

    fn generate_with_shapes(
        &self,
        pipeline: &mut (impl ModelPipeline + ?Sized),
        text: &str,
        speed: f32,
        voice: i32,
        shapes: SynthesisShapes,
    ) -> Result<Vec<f32>> {
        let max_len = if matches!(self.language.as_str(), "ko" | "ja") {
            120
        } else {
            300
        };
        let chunks = chunk_text(text, max_len);
        if chunks.is_empty() {
            bail!("text must not be empty");
        }
        let seed = self.seed.unwrap_or_else(|| rand::thread_rng().r#gen());
        let mut rng = StdRng::seed_from_u64(seed);
        let mut output = Vec::new();
        for (index, chunk) in chunks.iter().enumerate() {
            let audio =
                self.generate_chunk(pipeline, chunk, speed, voice as usize, &mut rng, shapes)?;
            if index > 0 {
                output.resize(
                    output.len()
                        + (self.silence_seconds * self.config.ae.sample_rate as f32) as usize,
                    0.0,
                );
            }
            output.extend(audio);
        }
        Ok(output)
    }

    fn generate_chunk(
        &self,
        pipeline: &mut (impl ModelPipeline + ?Sized),
        text: &str,
        speed: f32,
        voice: usize,
        rng: &mut StdRng,
        shapes: SynthesisShapes,
    ) -> Result<Vec<f32>> {
        let style = self.style.slice(voice)?;
        let (mut text_ids, mut text_mask) = process_text(text, &self.language, &self.indexer);
        let actual_text_len = text_ids.len() as i64;
        let text_len = match shapes {
            SynthesisShapes::Exact => actual_text_len,
            SynthesisShapes::NpuStatic => {
                if actual_text_len > NPU_TEXT_BUCKET {
                    bail!(
                        "normalized text chunk has {actual_text_len} tokens, exceeding the prepared Intel NPU limit of {NPU_TEXT_BUCKET}; split the input into shorter sentences"
                    );
                }
                text_ids.resize(NPU_TEXT_BUCKET as usize, 0);
                text_mask.resize(NPU_TEXT_BUCKET as usize, 0.0);
                NPU_TEXT_BUCKET
            }
        };
        let (_, duration) = pipeline
            .run(
                Graph::DurationPredictor,
                vec![
                    NamedTensor::i64("text_ids", [1, text_len], text_ids.clone())?,
                    NamedTensor::f32("style_dp", style.dp_shape, style.dp.to_vec())?,
                    NamedTensor::f32("text_mask", [1, 1, text_len], text_mask.clone())?,
                ],
                "duration",
            )?
            .into_f32()?;
        if duration.len() != 1 {
            bail!(
                "duration predictor returned {} values; expected one",
                duration.len()
            );
        }
        let predicted_seconds = duration[0] / speed;
        if !predicted_seconds.is_finite() || predicted_seconds > 600.0 {
            bail!("duration predictor returned invalid duration {predicted_seconds}");
        }
        let seconds = predicted_seconds.max(MIN_DURATION_SECONDS);
        if !seconds.is_finite() {
            bail!("duration predictor returned invalid duration {seconds}");
        }
        let (text_embedding_shape, text_embedding) = pipeline
            .run(
                Graph::TextEncoder,
                vec![
                    NamedTensor::i64("text_ids", [1, text_len], text_ids)?,
                    NamedTensor::f32("style_ttl", style.ttl_shape, style.ttl.to_vec())?,
                    NamedTensor::f32("text_mask", [1, 1, text_len], text_mask.clone())?,
                ],
                "text_emb",
            )?
            .into_f32()?;
        let sample_rate = self.config.ae.sample_rate as i64;
        let wav_length = (seconds * sample_rate as f32) as i64;
        let chunk_size = self.config.ae.base_chunk_size * self.config.ttl.chunk_compress_factor;
        let actual_latent_length = (wav_length + chunk_size - 1) / chunk_size;
        if !(1..=MAX_LATENT_LENGTH).contains(&actual_latent_length) {
            bail!(
                "predicted latent length {actual_latent_length} is outside 1..={MAX_LATENT_LENGTH}"
            );
        }
        let latent_length = match shapes {
            SynthesisShapes::Exact => actual_latent_length,
            SynthesisShapes::NpuStatic => npu_latent_bucket(actual_latent_length)?,
        };
        let latent_dimension = self.config.ttl.latent_dim * self.config.ttl.chunk_compress_factor;
        let latent_shape = [1, latent_dimension, latent_length];
        let mut latent = normal_samples((latent_dimension * latent_length) as usize, rng);
        let mut latent_mask = vec![0.0; latent_length as usize];
        latent_mask[..actual_latent_length as usize].fill(1.0);
        if shapes == SynthesisShapes::NpuStatic {
            zero_latent_padding(
                &mut latent,
                latent_dimension,
                actual_latent_length,
                latent_length,
            );
        }
        for step in 0..self.steps {
            let (shape, next) = pipeline
                .run(
                    Graph::VectorEstimator,
                    vec![
                        NamedTensor::f32("noisy_latent", latent_shape, latent)?,
                        NamedTensor::f32(
                            "text_emb",
                            text_embedding_shape.clone(),
                            text_embedding.clone(),
                        )?,
                        NamedTensor::f32("style_ttl", style.ttl_shape, style.ttl.to_vec())?,
                        NamedTensor::f32(
                            "latent_mask",
                            [1, 1, latent_length],
                            latent_mask.clone(),
                        )?,
                        NamedTensor::f32("text_mask", [1, 1, text_len], text_mask.clone())?,
                        NamedTensor::f32("current_step", [1], vec![step as f32])?,
                        NamedTensor::f32("total_step", [1], vec![self.steps as f32])?,
                    ],
                    "denoised_latent",
                )?
                .into_f32()?;
            if shape != latent_shape {
                bail!("vector estimator returned shape {shape:?}; expected {latent_shape:?}");
            }
            latent = next;
            if shapes == SynthesisShapes::NpuStatic {
                zero_latent_padding(
                    &mut latent,
                    latent_dimension,
                    actual_latent_length,
                    latent_length,
                );
            }
        }
        let (wav_shape, wav) = pipeline
            .run(
                Graph::Vocoder,
                vec![NamedTensor::f32("latent", latent_shape, latent)?],
                "wav_tts",
            )?
            .into_f32()?;
        if wav_shape.is_empty() || wav.is_empty() || wav.iter().any(|sample| !sample.is_finite()) {
            bail!("vocoder returned empty or non-finite audio with shape {wav_shape:?}");
        }
        wav.get(..(wav_length as usize).min(wav.len()))
            .map(<[f32]>::to_vec)
            .context("vocoder output could not be trimmed")
    }
}

pub struct DirectOpenvinoBackend {
    frontend: SupertonicFrontend,
    pipeline: Mutex<Box<dyn ModelPipeline>>,
    static_npu_shapes: bool,
}

pub struct DirectOrtBackend {
    frontend: SupertonicFrontend,
    pipeline: Mutex<Box<dyn ModelPipeline>>,
    runtime: Runtime,
}

impl DirectOrtBackend {
    pub(crate) fn create(
        config: &Config,
        paths: &AppPaths,
        libraries: OnnxRuntimePaths,
        runtime: Runtime,
    ) -> Result<Self> {
        Self::create_with(config, paths, runtime, |frontend, graphs| {
            Ok((
                frontend,
                Box::new(OrtPipeline::create(
                    libraries,
                    runtime,
                    graphs,
                    config.backend.threads,
                    config.backend.device_id,
                    &config.backend.options,
                )?) as Box<dyn ModelPipeline>,
            ))
        })
    }

    fn create_with(
        config: &Config,
        paths: &AppPaths,
        runtime: Runtime,
        create_pipeline: impl FnOnce(
            SupertonicFrontend,
            HashMap<Graph, PathBuf>,
        ) -> Result<(SupertonicFrontend, Box<dyn ModelPipeline>)>,
    ) -> Result<Self> {
        let (frontend, graphs) = SupertonicFrontend::load(config, paths)?;
        let (frontend, pipeline) = create_pipeline(frontend, graphs)?;
        Ok(Self {
            frontend,
            pipeline: Mutex::new(pipeline),
            runtime,
        })
    }
}

impl TtsBackend for DirectOrtBackend {
    fn kind(&self) -> &'static str {
        match self.runtime {
            Runtime::Default => "onnxruntime",
            Runtime::Cuda => "cuda",
            Runtime::Openvino => unreachable!("OpenVINO has a separate backend"),
        }
    }

    fn sample_rate(&self) -> i32 {
        self.frontend.config.ae.sample_rate
    }

    fn num_voices(&self) -> i32 {
        self.frontend.style.speakers as i32
    }

    fn generate(&self, text: &str, speed: f32, voice: i32) -> Result<Vec<f32>> {
        let mut pipeline = self
            .pipeline
            .lock()
            .map_err(|_| anyhow!("ONNX Runtime model pipeline lock was poisoned"))?;
        self.frontend.generate(&mut **pipeline, text, speed, voice)
    }
}

impl DirectOpenvinoBackend {
    pub(crate) fn create(
        config: &Config,
        paths: &AppPaths,
        runtime: OpenvinoRuntimePaths,
    ) -> Result<Self> {
        Self::create_with(config, paths, |frontend, graphs, device, cache_dir| {
            let mut pipeline = OpenvinoPipeline::create(
                runtime,
                device,
                graphs,
                cache_dir,
                config.backend.threads,
                &config.backend.options,
            )?;
            pipeline.require_cache_hits = device.eq_ignore_ascii_case("npu");
            Ok((frontend, Box::new(pipeline) as Box<dyn ModelPipeline>))
        })
    }

    fn create_with(
        config: &Config,
        paths: &AppPaths,
        create_pipeline: impl FnOnce(
            SupertonicFrontend,
            HashMap<Graph, PathBuf>,
            &str,
            &Path,
        ) -> Result<(SupertonicFrontend, Box<dyn ModelPipeline>)>,
    ) -> Result<Self> {
        let (frontend, graphs) = SupertonicFrontend::load(config, paths)?;
        let device = config.backend.canonical_device()?;
        let static_npu_shapes = device.eq_ignore_ascii_case("npu");
        let cache_dir = if static_npu_shapes {
            let state = npu_cache_state(config, paths);
            if !state.ready {
                bail!(
                    "Intel NPU compiled-model cache is not prepared: {}; run `omaspeak setup cache --prepare` (or rerun full/model setup)",
                    state.detail,
                );
            }
            state
                .directory
                .context("prepared NPU cache path is missing")?
        } else {
            paths
                .cache_dir
                .join("openvino")
                .join(device.to_ascii_lowercase())
        };
        let (frontend, pipeline) = create_pipeline(frontend, graphs, &device, &cache_dir)?;
        Ok(Self {
            frontend,
            pipeline: Mutex::new(pipeline),
            static_npu_shapes,
        })
    }
}

impl TtsBackend for DirectOpenvinoBackend {
    fn kind(&self) -> &'static str {
        "openvino"
    }

    fn sample_rate(&self) -> i32 {
        self.frontend.config.ae.sample_rate
    }

    fn num_voices(&self) -> i32 {
        self.frontend.style.speakers as i32
    }

    fn generate(&self, text: &str, speed: f32, voice: i32) -> Result<Vec<f32>> {
        let mut pipeline = self
            .pipeline
            .lock()
            .map_err(|_| anyhow!("OpenVINO model pipeline lock was poisoned"))?;
        if self.static_npu_shapes {
            self.frontend
                .generate_npu(&mut **pipeline, text, speed, voice)
        } else {
            self.frontend.generate(&mut **pipeline, text, speed, voice)
        }
    }
}

fn read_indexer(path: &Path) -> Result<Vec<i32>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let (chunks, remainder) = bytes.as_chunks::<4>();
    if chunks.is_empty() || !remainder.is_empty() {
        bail!("{} is not a non-empty raw i32 indexer", path.display());
    }
    Ok(chunks
        .iter()
        .map(|bytes| i32::from_le_bytes(*bytes))
        .collect())
}

fn read_voice_style(path: &Path) -> Result<VoiceStyle> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() < 48 {
        bail!(
            "{} is shorter than the six-i64 voice header",
            path.display()
        );
    }
    let mut dimensions = [0_i64; 6];
    for (index, bytes) in bytes[..48].as_chunks::<8>().0.iter().enumerate() {
        dimensions[index] = i64::from_le_bytes(*bytes);
    }
    if dimensions.iter().any(|&dimension| dimension <= 0) || dimensions[0] != dimensions[3] {
        bail!(
            "{} has invalid voice dimensions {dimensions:?}",
            path.display()
        );
    }
    let ttl_count = dimensions[..3]
        .iter()
        .try_fold(1_i64, |count, dimension| count.checked_mul(*dimension))
        .and_then(|count| usize::try_from(count).ok())
        .context("voice TTL dimensions overflow")?;
    let dp_count = dimensions[3..]
        .iter()
        .try_fold(1_i64, |count, dimension| count.checked_mul(*dimension))
        .and_then(|count| usize::try_from(count).ok())
        .context("voice duration dimensions overflow")?;
    let (chunks, remainder) = bytes[48..].as_chunks::<4>();
    if !remainder.is_empty() || chunks.len() != ttl_count + dp_count {
        bail!(
            "{} payload does not match voice dimensions {dimensions:?}",
            path.display()
        );
    }
    let values = chunks
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    Ok(VoiceStyle {
        speakers: dimensions[0] as usize,
        ttl_shape: [dimensions[0], dimensions[1], dimensions[2]],
        ttl: values[..ttl_count].to_vec(),
        dp_shape: [dimensions[3], dimensions[4], dimensions[5]],
        dp: values[ttl_count..].to_vec(),
    })
}

fn process_text(text: &str, language: &str, indexer: &[i32]) -> (Vec<i64>, Vec<f32>) {
    let mut text: String = text.nfkd().collect();
    for (from, to) in [
        ("–", "-"),
        ("‑", "-"),
        ("—", "-"),
        ("_", " "),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
        ("´", "'"),
        ("`", "'"),
        ("[", " "),
        ("]", " "),
        ("|", " "),
        ("/", " "),
        ("#", " "),
        ("→", " "),
        ("←", " "),
        ("♥", ""),
        ("☆", ""),
        ("♡", ""),
        ("©", ""),
        ("\\", ""),
        ("@", " at "),
        ("e.g.,", "for example, "),
        ("i.e.,", "that is, "),
    ] {
        text = text.replace(from, to);
    }
    text.retain(|character| !is_emoji(character));
    text = remove_spaces_before_punctuation(&text);
    while text.contains("\"\"") {
        text = text.replace("\"\"", "\"");
    }
    while text.contains("''") {
        text = text.replace("''", "'");
    }
    text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if !text.is_empty() && !text.ends_with(is_ending_punctuation) {
        text.push('.');
    }
    let tagged = format!("<{language}>{text}</{language}>");
    let ids = tagged
        .chars()
        .map(|character| indexer.get(character as usize).copied().unwrap_or_default() as i64)
        .collect::<Vec<_>>();
    let mask = vec![1.0; ids.len()];
    (ids, mask)
}

fn is_emoji(character: char) -> bool {
    matches!(
        character as u32,
        0x1F600..=0x1F64F
            | 0x1F300..=0x1F5FF
            | 0x1F680..=0x1F6FF
            | 0x1F700..=0x1F77F
            | 0x1F780..=0x1F7FF
            | 0x1F800..=0x1F8FF
            | 0x1F900..=0x1F9FF
            | 0x1FA00..=0x1FAFF
            | 0x2600..=0x27BF
            | 0x1F1E6..=0x1F1FF
    )
}

fn remove_spaces_before_punctuation(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(character) = chars.next() {
        if character == ' '
            && chars
                .peek()
                .is_some_and(|next| matches!(next, ',' | '.' | '!' | '?' | ';' | ':' | '\''))
        {
            continue;
        }
        result.push(character);
    }
    result
}

fn is_ending_punctuation(character: char) -> bool {
    matches!(
        character,
        '.' | '!'
            | '?'
            | ';'
            | ':'
            | ','
            | '\''
            | '"'
            | ')'
            | ']'
            | '}'
            | '>'
            | '…'
            | '。'
            | '」'
            | '』'
            | '】'
            | '〉'
            | '》'
            | '›'
            | '»'
            | '“'
            | '”'
            | '‘'
            | '’'
    )
}

const ABBREVIATIONS: &[&str] = &[
    "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Jr.", "St.", "Ave.", "Rd.", "Blvd.", "Dept.",
    "Inc.", "Ltd.", "Co.", "Corp.", "etc.", "vs.", "i.e.", "e.g.", "Ph.D.",
];

fn chunk_text(text: &str, max_len: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    for paragraph in split_paragraphs(text) {
        if paragraph.chars().count() <= max_len {
            chunks.push(paragraph.to_owned());
            continue;
        }
        let mut current = String::new();
        for sentence in split_sentences(&paragraph) {
            for piece in split_long_piece(&sentence, max_len) {
                let separator = usize::from(!current.is_empty());
                if current.chars().count() + separator + piece.chars().count() > max_len
                    && !current.is_empty()
                {
                    chunks.push(std::mem::take(&mut current));
                }
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(piece.trim());
            }
        }
        if !current.is_empty() {
            chunks.push(current);
        }
    }
    chunks
}

fn split_paragraphs(text: &str) -> Vec<String> {
    let mut paragraphs = Vec::new();
    let mut current = String::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            if !current.is_empty() {
                paragraphs.push(std::mem::take(&mut current));
            }
        } else {
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(line.trim());
        }
    }
    if !current.is_empty() {
        paragraphs.push(current);
    }
    paragraphs
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let chars = text.char_indices().collect::<Vec<_>>();
    for (index, &(offset, character)) in chars.iter().enumerate() {
        if !matches!(character, '.' | '!' | '?') {
            continue;
        }
        let end = offset + character.len_utf8();
        let candidate = text[start..end].trim();
        let abbreviated = ABBREVIATIONS.iter().any(|item| candidate.ends_with(item));
        let followed_by_space = chars
            .get(index + 1)
            .is_none_or(|(_, next)| next.is_whitespace());
        if !abbreviated && followed_by_space {
            sentences.push(text[start..end].trim().to_owned());
            start = chars.get(index + 1).map_or(text.len(), |(next, _)| *next);
        }
    }
    if start < text.len() {
        sentences.push(text[start..].trim().to_owned());
    }
    if sentences.is_empty() {
        sentences.push(text.to_owned());
    }
    sentences
}

fn split_long_piece(text: &str, max_len: usize) -> Vec<String> {
    let max_len = max_len.max(1);
    if text.chars().count() <= max_len {
        return vec![text.to_owned()];
    }
    let mut pieces = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if word.chars().count() > max_len {
            if !current.is_empty() {
                pieces.push(std::mem::take(&mut current));
            }
            let characters = word.chars().collect::<Vec<_>>();
            pieces.extend(
                characters
                    .chunks(max_len)
                    .map(|chunk| chunk.iter().collect::<String>()),
            );
            continue;
        }
        if current.chars().count() + usize::from(!current.is_empty()) + word.chars().count()
            > max_len
            && !current.is_empty()
        {
            pieces.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        pieces.push(current);
    }
    pieces
}

fn normal_samples(count: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut samples = Vec::with_capacity(count);
    while samples.len() < count {
        let uniform_a = rng.r#gen::<f32>().max(f32::MIN_POSITIVE);
        let uniform_b = rng.r#gen::<f32>();
        let radius = (-2.0 * uniform_a.ln()).sqrt();
        let angle = std::f32::consts::TAU * uniform_b;
        samples.push(radius * angle.cos());
        if samples.len() < count {
            samples.push(radius * angle.sin());
        }
    }
    samples
}

#[cfg(test)]
#[path = "../tests/unit/supertonic.rs"]
mod tests;
