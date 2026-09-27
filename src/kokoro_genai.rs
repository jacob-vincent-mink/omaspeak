//! Optional native C bridge to OpenVINO GenAI's C++ Kokoro pipeline.

use std::ffi::{CStr, CString, c_char, c_void};
use std::fs;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Mutex;

use anyhow::{Context, Result, bail, ensure};
use libloading::Library;

use crate::backend::{BackendConfig, Runtime};
use crate::config::Config;
use crate::engine::TtsBackend;
use crate::paths::AppPaths;
use crate::runtime_inventory::{Evidence, Probe};

const NAME: &str = "libomaspeak_kokoro_openvino.so";
const ERROR_LEN: usize = 4096;
const MAX_SAMPLES: usize = 128 * 1024 * 1024;

type Abi = unsafe extern "C" fn() -> u32;
type ProbeFn = unsafe extern "C" fn(*const c_char, *const c_char, *mut c_char, usize) -> i32;
type OpenFn = unsafe extern "C" fn(
    *const c_char,
    *const c_char,
    *const c_char,
    *mut *mut c_void,
    *mut c_char,
    usize,
) -> i32;
type GenerateFn = unsafe extern "C" fn(
    *mut c_void,
    *const c_char,
    *const f32,
    usize,
    *mut *mut f32,
    *mut usize,
    *mut u32,
    *mut c_char,
    usize,
) -> i32;
type FreeFn = unsafe extern "C" fn(*mut f32);
type CloseFn = unsafe extern "C" fn(*mut c_void);

#[derive(Clone, Copy)]
struct Symbols {
    probe: ProbeFn,
    open: OpenFn,
    generate: GenerateFn,
    free: FreeFn,
    close: CloseFn,
}

struct Native {
    pipeline: *mut c_void,
    symbols: Symbols,
    _library: Library,
}

// The Mutex serializes every call to the C++ pipeline.
unsafe impl Send for Native {}

pub struct KokoroGenAiBackend {
    native: Mutex<Native>,
    voices: [Vec<f32>; 2],
}

fn provider_path(config: &BackendConfig) -> Result<PathBuf> {
    let explicit = config
        .options
        .get("kokoro_library")
        .cloned()
        .or_else(|| std::env::var("OMASPEAK_KOKORO_LIBRARY").ok());
    if let Some(explicit) = explicit {
        let path = PathBuf::from(explicit);
        ensure!(
            path.is_absolute(),
            "Kokoro native provider path must be absolute"
        );
        ensure!(
            path.is_file(),
            "Kokoro native provider is missing: {}",
            path.display()
        );
        return Ok(path);
    }
    let executable = std::env::current_exe().context("locate Omaspeak executable")?;
    let home = std::env::var_os("HOME").map(PathBuf::from);
    provider_candidates(&executable, home.as_deref())?
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| anyhow::anyhow!(
            "native Kokoro OpenVINO provider is unavailable; build {NAME} with scripts/build-kokoro-openvino-bridge.sh and install it beside the Omaspeak provider"
        ))
}

fn provider_candidates(executable: &Path, home: Option<&Path>) -> Result<Vec<PathBuf>> {
    let bin = executable.parent().context("executable has no parent")?;
    let mut candidates = vec![bin.join(NAME), bin.join("lib").join(NAME)];
    if let Some(prefix) = bin.parent() {
        candidates.push(prefix.join("lib/omaspeak").join(NAME));
    }
    if let Some(home) = home {
        candidates.push(home.join(".local/lib/omaspeak").join(NAME));
    }
    candidates.push(PathBuf::from("/usr/lib/omaspeak").join(NAME));
    candidates.push(PathBuf::from("/usr/local/lib/omaspeak").join(NAME));
    candidates.push(PathBuf::from("/usr/lib64/omaspeak").join(NAME));
    Ok(candidates)
}

fn symbols(library: &Library) -> Result<Symbols> {
    unsafe {
        let abi = *library.get::<Abi>(b"omaspeak_kokoro_abi_version\0")?;
        ensure!(abi() == 2, "Kokoro native provider ABI is not version 2");
        Ok(Symbols {
            probe: *library.get(b"omaspeak_kokoro_probe\0")?,
            open: *library.get(b"omaspeak_kokoro_open\0")?,
            generate: *library.get(b"omaspeak_kokoro_generate\0")?,
            free: *library.get(b"omaspeak_kokoro_free_samples\0")?,
            close: *library.get(b"omaspeak_kokoro_close\0")?,
        })
    }
}

fn load(config: &BackendConfig) -> Result<(PathBuf, Library, Symbols)> {
    let path = provider_path(config)?;
    let library = unsafe { Library::new(&path) }
        .with_context(|| format!("load Kokoro native provider {}", path.display()))?;
    let symbols = symbols(&library)?;
    Ok((path, library, symbols))
}

fn message(buffer: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn device(config: &BackendConfig) -> Result<CString> {
    ensure!(
        config.runtime == Runtime::Openvino,
        "Kokoro GenAI requires runtime=openvino"
    );
    let value = config.canonical_device()?.to_ascii_uppercase();
    ensure!(
        matches!(value.as_str(), "CPU" | "GPU" | "NPU"),
        "Kokoro requires CPU, GPU, or NPU"
    );
    CString::new(value).context("encode Kokoro device")
}

fn device_plugin(config: &BackendConfig, device: &CStr) -> Result<Option<CString>> {
    let name = match device.to_str()? {
        "CPU" => "libopenvino_intel_cpu_plugin.so",
        "GPU" => "libopenvino_intel_gpu_plugin.so",
        "NPU" => "libopenvino_intel_npu_plugin.so",
        _ => return Ok(None),
    };
    let mut directories = Vec::new();
    if let Some(plugins) = &config.openvino_plugins
        && let Some(parent) = plugins.parent()
    {
        directories.push(parent.to_path_buf());
    }
    directories.extend(config.library_dirs.iter().cloned());
    directories.extend([
        PathBuf::from("/usr/lib/openvino"),
        PathBuf::from("/usr/local/lib/openvino"),
        PathBuf::from("/usr/lib"),
        PathBuf::from("/usr/local/lib"),
    ]);
    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        directories.push(home.join(".local/lib/openvino"));
        directories.push(home.join(".local/lib"));
    }
    directories
        .into_iter()
        .map(|directory| directory.join(name))
        .find(|path| path.is_file())
        .map(|path| CString::new(path.to_string_lossy().as_bytes()).map_err(Into::into))
        .transpose()
}

pub fn probe(config: &BackendConfig) -> Probe {
    let attempt = (|| -> Result<Probe> {
        let device = device(config)?;
        let plugin = device_plugin(config, &device)?;
        let (path, _library, symbols) = load(config)?;
        let mut error = [0 as c_char; ERROR_LEN];
        let status = unsafe {
            (symbols.probe)(
                device.as_ptr(),
                plugin.as_ref().map_or(ptr::null(), |path| path.as_ptr()),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ensure!(
            status == 0,
            "Kokoro native provider probe failed: {}",
            message(&error)
        );
        let selected = device.to_str()?.to_owned();
        Ok(Probe {
            loadable: true,
            device_accessible: Some(true),
            ready: true,
            evidence: Evidence {
                versions: vec!["Kokoro native bridge ABI 2".into()],
                provider_registration: true,
                available_devices: vec![selected.clone()],
                selected_device: Some(selected),
                provider_path: Some(path),
                ..Default::default()
            },
            errors: Vec::new(),
        })
    })();
    attempt.unwrap_or_else(|error| Probe {
        errors: vec![format!("{error:#}")],
        ..Default::default()
    })
}

fn read_voice(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("read Kokoro voice {}", path.display()))?;
    ensure!(
        !bytes.is_empty() && bytes.len() % 4 == 0,
        "Kokoro voice has invalid float32 size"
    );
    let data = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect::<Vec<_>>();
    ensure!(
        data.iter().all(|sample| sample.is_finite()),
        "Kokoro voice contains non-finite data"
    );
    Ok(data)
}

impl KokoroGenAiBackend {
    pub fn create(config: &Config, paths: &AppPaths) -> Result<Self> {
        ensure!(
            config.model.name == crate::catalog::KOKORO_OPENVINO_MODEL_ID,
            "kokoro-genai requires the pinned Kokoro OpenVINO model"
        );
        let model_dir = config.model_directory(paths);
        for asset in [
            "openvino_model.xml",
            "openvino_model.bin",
            "config.json",
            "voices/af_heart.bin",
            "voices/am_michael.bin",
        ] {
            ensure!(
                model_dir.join(asset).is_file(),
                "Kokoro OpenVINO asset is missing: {}",
                model_dir.join(asset).display()
            );
        }
        let voices = [
            read_voice(&model_dir.join("voices/af_heart.bin"))?,
            read_voice(&model_dir.join("voices/am_michael.bin"))?,
        ];
        let device = device(&config.backend)?;
        let plugin = device_plugin(&config.backend, &device)?;
        let (_path, library, symbols) = load(&config.backend)?;
        let model = CString::new(model_dir.to_string_lossy().as_bytes())?;
        let mut error = [0 as c_char; ERROR_LEN];
        let mut pipeline = ptr::null_mut();
        let status = unsafe {
            (symbols.open)(
                model.as_ptr(),
                device.as_ptr(),
                plugin.as_ref().map_or(ptr::null(), |path| path.as_ptr()),
                &mut pipeline,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ensure!(
            status == 0 && !pipeline.is_null(),
            "load Kokoro pipeline: {}",
            message(&error)
        );
        Ok(Self {
            native: Mutex::new(Native {
                pipeline,
                symbols,
                _library: library,
            }),
            voices,
        })
    }
}

impl TtsBackend for KokoroGenAiBackend {
    fn kind(&self) -> &'static str {
        "kokoro-genai"
    }
    fn sample_rate(&self) -> i32 {
        24_000
    }
    fn num_voices(&self) -> i32 {
        2
    }

    fn generate(&self, text: &str, speed: f32, voice: i32) -> Result<Vec<f32>> {
        ensure!(
            speed == 1.0,
            "Kokoro OpenVINO GenAI currently supports speed 1.0"
        );
        let index = usize::try_from(voice).context("invalid Kokoro voice")?;
        let embedding = self.voices.get(index).context("invalid Kokoro voice")?;
        let text = CString::new(text).context("Kokoro text contains a NUL byte")?;
        let native = self
            .native
            .lock()
            .map_err(|_| anyhow::anyhow!("Kokoro pipeline lock was poisoned"))?;
        let mut samples = ptr::null_mut();
        let mut count = 0usize;
        let mut rate = 0u32;
        let mut error = [0 as c_char; ERROR_LEN];
        let status = unsafe {
            (native.symbols.generate)(
                native.pipeline,
                text.as_ptr(),
                embedding.as_ptr(),
                embedding.len(),
                &mut samples,
                &mut count,
                &mut rate,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ensure!(status == 0, "Kokoro synthesis failed: {}", message(&error));
        if samples.is_null() || count == 0 || count > MAX_SAMPLES || rate != 24_000 {
            if !samples.is_null() {
                unsafe { (native.symbols.free)(samples) };
            }
            bail!("Kokoro native provider returned invalid audio");
        }
        let output = unsafe { std::slice::from_raw_parts(samples, count).to_vec() };
        unsafe { (native.symbols.free)(samples) };
        ensure!(
            output.iter().all(|sample| sample.is_finite()),
            "Kokoro returned non-finite audio"
        );
        Ok(output)
    }
}

impl Drop for KokoroGenAiBackend {
    fn drop(&mut self) {
        if let Ok(native) = self.native.lock() {
            unsafe { (native.symbols.close)(native.pipeline) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn provider_discovery_covers_user_and_install_prefix() {
        let paths = provider_candidates(
            Path::new("/opt/omaspeak/bin/omaspeak"),
            Some(Path::new("/home/example")),
        )
        .unwrap();
        assert!(paths.contains(&PathBuf::from(
            "/home/example/.local/lib/omaspeak/libomaspeak_kokoro_openvino.so"
        )));
        assert!(paths.contains(&PathBuf::from(
            "/opt/omaspeak/lib/omaspeak/libomaspeak_kokoro_openvino.so"
        )));
    }

    #[test]
    fn missing_native_bridge_fails_before_model_download() {
        let mut config = BackendConfig {
            runtime: Runtime::Openvino,
            device: "npu".into(),
            ..Default::default()
        };
        config
            .options
            .insert("kokoro_library".into(), "/missing/kokoro.so".into());
        let result = probe(&config);
        assert!(!result.ready);
        assert!(result.errors[0].contains("native provider is missing"));
        config
            .options
            .insert("kokoro_library".into(), "relative/kokoro.so".into());
        assert!(probe(&config).errors[0].contains("must be absolute"));
    }

    #[test]
    fn rejects_corrupt_voice_embeddings_before_loading_native_code() {
        let root =
            std::env::temp_dir().join(format!("omaspeak-voice-validation-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let voice = root.join("voice.bin");
        fs::write(&voice, []).unwrap();
        assert!(read_voice(&voice).unwrap_err().to_string().contains("size"));
        fs::write(&voice, [0u8; 3]).unwrap();
        assert!(read_voice(&voice).unwrap_err().to_string().contains("size"));
        fs::write(&voice, f32::NAN.to_le_bytes()).unwrap();
        assert!(
            read_voice(&voice)
                .unwrap_err()
                .to_string()
                .contains("non-finite")
        );
        fs::remove_file(&voice).unwrap();
        assert!(
            read_voice(&voice)
                .unwrap_err()
                .to_string()
                .contains("read Kokoro voice")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn native_abi_probes_plugin_and_round_trips_audio_without_genai() {
        let root = std::env::temp_dir().join(format!(
            "omaspeak-kokoro-ffi-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("provider.c");
        let library = root.join("libmock_kokoro.so");
        fs::write(
            &source,
            r#"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
unsigned omaspeak_kokoro_abi_version(void) { return 2; }
int omaspeak_kokoro_probe(const char* device, const char* plugin, char* error, size_t capacity) {
    if (capacity) error[0] = 0;
    if (strcmp(device, "GPU") == 0) {
        if (capacity > 12) strcpy(error, "GPU rejected");
        return 1;
    }
    return strcmp(device, "NPU") == 0 && plugin && strstr(plugin, "npu_plugin.so") ? 0 : 1;
}
int omaspeak_kokoro_open(const char* model, const char* device, const char* plugin,
                         void** pipeline, char* error, size_t capacity) {
    if (capacity) error[0] = 0;
    if (!model || !device || !plugin) return 1;
    if (strstr(model, "reject-open")) {
        if (capacity > 13) strcpy(error, "open rejected");
        return 1;
    }
    *pipeline = malloc(1);
    return *pipeline ? 0 : 1;
}
int omaspeak_kokoro_generate(void* pipeline, const char* text, const float* voice,
                             size_t voice_count, float** output, size_t* count,
                             unsigned* rate, char* error, size_t capacity) {
    if (capacity) error[0] = 0;
    if (!pipeline || !text || !voice || voice_count != 2) return 1;
    if (strcmp(text, "reject") == 0) {
        if (capacity > 18) strcpy(error, "synthesis rejected");
        return 1;
    }
    if (strcmp(text, "empty") == 0) {
        *output = NULL; *count = 0; *rate = 24000;
        return 0;
    }
    *output = malloc(4 * sizeof(float));
    if (!*output) return 1;
    (*output)[0] = voice[0]; (*output)[1] = voice[1];
    (*output)[2] = 0.25f; (*output)[3] = -0.25f;
    if (strcmp(text, "nonfinite") == 0) (*output)[2] = 0.0f / 0.0f;
    *count = 4; *rate = strcmp(text, "wrong-rate") == 0 ? 16000 : 24000;
    return 0;
}
void omaspeak_kokoro_free_samples(float* samples) { free(samples); }
void omaspeak_kokoro_close(void* pipeline) { free(pipeline); }
"#,
        )
        .unwrap();
        assert!(
            Command::new("cc")
                .args(["-shared", "-fPIC", "-o"])
                .arg(&library)
                .arg(&source)
                .status()
                .unwrap()
                .success()
        );
        let plugins = root.join("plugins");
        fs::create_dir_all(&plugins).unwrap();
        fs::write(plugins.join("plugins.xml"), b"<ie/>").unwrap();
        fs::write(plugins.join("libopenvino_intel_npu_plugin.so"), b"fixture").unwrap();
        let mut config = Config::default();
        config.backend.kind = "kokoro-genai".into();
        config.backend.runtime = Runtime::Openvino;
        config.backend.device = "npu".into();
        config.backend.openvino_plugins = Some(plugins.join("plugins.xml"));
        config
            .backend
            .options
            .insert("kokoro_library".into(), library.display().to_string());
        config.model.name = crate::catalog::KOKORO_OPENVINO_MODEL_ID.into();
        let paths = AppPaths {
            config_file: root.join("config.toml"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
            state_dir: root.join("state"),
            runtime_dir: root.join("run"),
        };
        let model = config.model_directory(&paths);
        assert!(
            KokoroGenAiBackend::create(&config, &paths)
                .err()
                .unwrap()
                .to_string()
                .contains("asset is missing")
        );
        fs::create_dir_all(model.join("voices")).unwrap();
        for name in ["openvino_model.xml", "openvino_model.bin", "config.json"] {
            fs::write(model.join(name), b"fixture").unwrap();
        }
        for name in ["af_heart.bin", "am_michael.bin"] {
            fs::write(
                model.join("voices").join(name),
                [0.5f32, -0.5].map(f32::to_le_bytes).concat(),
            )
            .unwrap();
        }
        assert!(probe(&config.backend).ready);
        assert_eq!(
            device_plugin(&config.backend, &CString::new("NPU").unwrap())
                .unwrap()
                .unwrap()
                .to_str()
                .unwrap(),
            plugins
                .join("libopenvino_intel_npu_plugin.so")
                .to_str()
                .unwrap()
        );
        let mut rejected_probe = config.backend.clone();
        rejected_probe.device = "gpu".into();
        let rejected = probe(&rejected_probe);
        assert!(!rejected.ready);
        assert!(rejected.errors[0].contains("GPU rejected"));
        let backend = KokoroGenAiBackend::create(&config, &paths).unwrap();
        assert_eq!(backend.kind(), "kokoro-genai");
        assert_eq!(backend.sample_rate(), 24_000);
        assert_eq!(backend.num_voices(), 2);
        assert_eq!(
            backend.generate("hello", 1.0, 0).unwrap(),
            [0.5, -0.5, 0.25, -0.25]
        );
        assert!(backend.generate("hello", 1.1, 0).is_err());
        assert!(backend.generate("hello", 1.0, 9).is_err());
        assert!(backend.generate("nul\0byte", 1.0, 0).is_err());
        for (input, expected) in [
            ("reject", "synthesis rejected"),
            ("empty", "invalid audio"),
            ("wrong-rate", "invalid audio"),
            ("nonfinite", "non-finite audio"),
        ] {
            let error = backend.generate(input, 1.0, 0).unwrap_err();
            assert!(
                format!("{error:#}").contains(expected),
                "{input}: {error:#}"
            );
        }
        drop(backend);
        let rejected_model = root.join("reject-open");
        fs::rename(&model, &rejected_model).unwrap();
        // A valid asset set can still be rejected by the native pipeline.
        config.model.directory = rejected_model.display().to_string();
        assert!(
            format!(
                "{:#}",
                KokoroGenAiBackend::create(&config, &paths).err().unwrap()
            )
            .contains("open rejected")
        );
        fs::remove_dir_all(root).unwrap();
    }
}
