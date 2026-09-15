//! Supervised audio.cpp provider.
//!
//! The application process never loads optional native inference code. A
//! private worker dynamically loads one complete audio.cpp installation and
//! keeps its model and session alive across synthesis requests.

use std::collections::BTreeMap;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::io::{self, Read, Write};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use libloading::Library;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::backend::Runtime;
use crate::config::Config;
use crate::engine::TtsBackend;
use crate::paths::AppPaths;

const AUDIOCPP_ABI_MAJOR: u32 = 0;
const AUDIOCPP_ABI_MIN_MINOR: u32 = 1;
const SUPERTONIC_SAMPLE_RATE: i32 = 44_100;
const SUPERTONIC_VOICES: i32 = 10;
const MAX_CONTROL_FRAME: usize = 1024 * 1024;
const MAX_PCM_SAMPLES: usize = 64 * 1024 * 1024;
const WORKER_TIMEOUT: Duration = Duration::from_secs(300);
const WORKER_STOP_TIMEOUT: Duration = Duration::from_secs(1);
const PROVIDER_PROBE_TIMEOUT: Duration = Duration::from_secs(10);

type Status = c_int;
type Handle = *mut c_void;

#[repr(C)]
struct ModelConfig {
    family_hint: *const c_char,
    config_id: *const c_char,
    weight_id: *const c_char,
    model_spec_override: *const c_char,
}

#[repr(C)]
struct NativeBackendConfig {
    backend: *const c_char,
    device: c_int,
    threads: c_int,
}

type LastError = unsafe extern "C" fn() -> *const c_char;
type AbiVersion = unsafe extern "C" fn() -> u32;
type OptionsCreate = unsafe extern "C" fn() -> Handle;
type OptionsSet = unsafe extern "C" fn(Handle, *const c_char, *const c_char) -> Status;
type OptionsFree = unsafe extern "C" fn(Handle);
type RegistryCreate = unsafe extern "C" fn(*const c_char, *mut Handle) -> Status;
type RegistryFree = unsafe extern "C" fn(Handle);
type ModelLoad =
    unsafe extern "C" fn(Handle, *const c_char, *const ModelConfig, Handle, *mut Handle) -> Status;
type ModelFree = unsafe extern "C" fn(Handle);
type ModelSupports = unsafe extern "C" fn(Handle, *const c_char, *const c_char) -> c_int;
type SessionCreate = unsafe extern "C" fn(
    Handle,
    *const c_char,
    *const c_char,
    *const NativeBackendConfig,
    Handle,
    *mut Handle,
) -> Status;
type SessionFree = unsafe extern "C" fn(Handle);
type SessionRun = unsafe extern "C" fn(Handle, Handle, *mut Handle) -> Status;
type RequestCreate = unsafe extern "C" fn() -> Handle;
type RequestFree = unsafe extern "C" fn(Handle);
type RequestSetText = unsafe extern "C" fn(Handle, *const c_char, *const c_char) -> Status;
type RequestSetVoiceId = unsafe extern "C" fn(Handle, *const c_char) -> Status;
type RequestSetSpeakingRate = unsafe extern "C" fn(Handle, f32) -> Status;
type RequestSetOption = unsafe extern "C" fn(Handle, *const c_char, *const c_char) -> Status;
type ResultAudio =
    unsafe extern "C" fn(Handle, *mut *const f32, *mut usize, *mut c_int, *mut c_int) -> Status;
type ResultFree = unsafe extern "C" fn(Handle);

struct Api {
    abi_version: AbiVersion,
    last_error: LastError,
    options_create: OptionsCreate,
    options_set: OptionsSet,
    options_free: OptionsFree,
    registry_create: RegistryCreate,
    registry_free: RegistryFree,
    model_load: ModelLoad,
    model_free: ModelFree,
    model_supports: ModelSupports,
    session_create: SessionCreate,
    session_free: SessionFree,
    session_run: SessionRun,
    request_create: RequestCreate,
    request_free: RequestFree,
    request_set_text: RequestSetText,
    request_set_voice_id: RequestSetVoiceId,
    request_set_speaking_rate: RequestSetSpeakingRate,
    request_set_option: RequestSetOption,
    result_audio: ResultAudio,
    result_free: ResultFree,
    // Keep the library loaded until every copied function pointer is dropped.
    _library: Library,
}

impl Api {
    fn load(path: &Path) -> Result<Self> {
        // The library is loaded only in the disposable worker process. The
        // parent validates the path and supervises worker failure.
        let library = unsafe { Library::new(path) }
            .with_context(|| format!("load audio.cpp provider {}", path.display()))?;
        unsafe {
            let api = Self {
                abi_version: load_symbol(&library, b"audiocpp_abi_version\0")?,
                last_error: load_symbol(&library, b"audiocpp_last_error\0")?,
                options_create: load_symbol(&library, b"audiocpp_options_create\0")?,
                options_set: load_symbol(&library, b"audiocpp_options_set\0")?,
                options_free: load_symbol(&library, b"audiocpp_options_free\0")?,
                registry_create: load_symbol(&library, b"audiocpp_registry_create\0")?,
                registry_free: load_symbol(&library, b"audiocpp_registry_free\0")?,
                model_load: load_symbol(&library, b"audiocpp_model_load\0")?,
                model_free: load_symbol(&library, b"audiocpp_model_free\0")?,
                model_supports: load_symbol(&library, b"audiocpp_model_supports\0")?,
                session_create: load_symbol(&library, b"audiocpp_session_create\0")?,
                session_free: load_symbol(&library, b"audiocpp_session_free\0")?,
                session_run: load_symbol(&library, b"audiocpp_session_run\0")?,
                request_create: load_symbol(&library, b"audiocpp_request_create\0")?,
                request_free: load_symbol(&library, b"audiocpp_request_free\0")?,
                request_set_text: load_symbol(&library, b"audiocpp_request_set_text\0")?,
                request_set_voice_id: load_symbol(&library, b"audiocpp_request_set_voice_id\0")?,
                request_set_speaking_rate: load_symbol(
                    &library,
                    b"audiocpp_request_set_speaking_rate\0",
                )?,
                request_set_option: load_symbol(&library, b"audiocpp_request_set_option\0")?,
                result_audio: load_symbol(&library, b"audiocpp_result_audio\0")?,
                result_free: load_symbol(&library, b"audiocpp_result_free\0")?,
                _library: library,
            };
            validate_abi((api.abi_version)())?;
            Ok(api)
        }
    }

    fn check(&self, status: Status, operation: &str) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        let pointer = unsafe { (self.last_error)() };
        let detail = if pointer.is_null() {
            "provider returned no error detail".into()
        } else {
            // audio.cpp documents this pointer as thread-local and valid until
            // the next ABI call. Copy it immediately.
            unsafe { CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned()
        };
        bail!("audio.cpp {operation} failed ({status}): {detail}")
    }
}

fn create_option_map(api: &Api, entries: &BTreeMap<String, String>, scope: &str) -> Result<Handle> {
    if entries.is_empty() {
        return Ok(std::ptr::null_mut());
    }
    let entries = entries
        .iter()
        .map(|(key, value)| {
            Ok((
                CString::new(key.as_str()).context("audio.cpp option name contains a NUL byte")?,
                CString::new(value.as_str())
                    .context("audio.cpp option value contains a NUL byte")?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let options = unsafe { (api.options_create)() };
    if options.is_null() {
        bail!("audio.cpp {scope} option-map creation returned a null handle");
    }
    for (key, value) in entries {
        if let Err(error) = api.check(
            unsafe { (api.options_set)(options, key.as_ptr(), value.as_ptr()) },
            &format!("{scope} option configuration"),
        ) {
            unsafe { (api.options_free)(options) };
            return Err(error);
        }
    }
    Ok(options)
}

#[derive(Default)]
struct ScopedOptions {
    load: BTreeMap<String, String>,
    session: BTreeMap<String, String>,
    request: BTreeMap<String, String>,
}

fn scoped_options(options: &BTreeMap<String, String>) -> Result<ScopedOptions> {
    let mut scoped = ScopedOptions::default();
    for (key, value) in options {
        let (scope, name) = key.split_once('.').with_context(|| {
            format!(
                "audio.cpp backend option {key:?} has no scope; use load.NAME, session.NAME, or request.NAME"
            )
        })?;
        if name.is_empty() {
            bail!("audio.cpp backend option {key:?} has an empty option name");
        }
        let target = match scope {
            "load" => &mut scoped.load,
            "session" => &mut scoped.session,
            "request" => &mut scoped.request,
            _ => bail!(
                "audio.cpp backend option {key:?} has unknown scope {scope:?}; use load, session, or request"
            ),
        };
        if scope == "request" && matches!(name, "num_inference_steps" | "language") {
            bail!(
                "audio.cpp request option {name:?} is managed by model.steps/model.language and cannot be overridden through backend.options"
            );
        }
        target.insert(name.to_owned(), value.to_owned());
    }
    Ok(scoped)
}

unsafe fn load_symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    // Every symbol is copied while `library` remains owned by Api.
    let symbol = unsafe { library.get::<T>(name) }.with_context(|| {
        format!(
            "resolve native symbol {}",
            String::from_utf8_lossy(name).trim_end_matches('\0')
        )
    })?;
    Ok(*symbol)
}

fn validate_abi(version: u32) -> Result<()> {
    let major = version >> 16;
    let minor = (version >> 8) & 0xff;
    if major != AUDIOCPP_ABI_MAJOR || minor < AUDIOCPP_ABI_MIN_MINOR {
        bail!(
            "audio.cpp ABI {major}.{minor}.{} is incompatible; expected {}.{} or newer with the same major",
            version & 0xff,
            AUDIOCPP_ABI_MAJOR,
            AUDIOCPP_ABI_MIN_MINOR
        );
    }
    Ok(())
}

struct NativeEngine {
    api: Api,
    registry: Handle,
    model: Handle,
    session: Handle,
    steps: i32,
    language: CString,
    request_options: Vec<(CString, CString)>,
}

impl NativeEngine {
    fn load(spec: &WorkerSpec) -> Result<Self> {
        let api = Api::load(&spec.library)?;
        let family = CString::new("supertonic").expect("static string has no NUL");
        let model_path = path_to_c_string(&spec.model, "model.file")?;
        let task = CString::new("tts").expect("static string has no NUL");
        let mode = CString::new("offline").expect("static string has no NUL");
        let backend = CString::new(spec.backend.as_str()).context("backend contains a NUL byte")?;
        let language =
            CString::new(spec.language.as_str()).context("language contains a NUL byte")?;

        let request_options = spec
            .request_options
            .iter()
            .map(|(key, value)| {
                Ok((
                    CString::new(key.as_str())
                        .context("request option name contains a NUL byte")?,
                    CString::new(value.as_str())
                        .context("request option value contains a NUL byte")?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut registry = std::ptr::null_mut();
        api.check(
            unsafe { (api.registry_create)(std::ptr::null(), &mut registry) },
            "registry creation",
        )?;
        if registry.is_null() {
            bail!("audio.cpp registry creation returned a null handle");
        }

        let model_config = ModelConfig {
            family_hint: family.as_ptr(),
            config_id: std::ptr::null(),
            weight_id: std::ptr::null(),
            model_spec_override: std::ptr::null(),
        };
        let load_options = match create_option_map(&api, &spec.load_options, "load") {
            Ok(options) => options,
            Err(error) => {
                unsafe { (api.registry_free)(registry) };
                return Err(error);
            }
        };
        let mut model = std::ptr::null_mut();
        let model_status = unsafe {
            (api.model_load)(
                registry,
                model_path.as_ptr(),
                &model_config,
                load_options,
                &mut model,
            )
        };
        unsafe { (api.options_free)(load_options) };
        if let Err(error) = api.check(model_status, "model load") {
            unsafe { (api.registry_free)(registry) };
            return Err(error);
        }
        if model.is_null() {
            unsafe { (api.registry_free)(registry) };
            bail!("audio.cpp model load returned a null handle");
        }
        if unsafe { (api.model_supports)(model, task.as_ptr(), mode.as_ptr()) } != 1 {
            unsafe {
                (api.model_free)(model);
                (api.registry_free)(registry);
            }
            bail!("model.file does not support offline TTS through audio.cpp");
        }

        let native_config = NativeBackendConfig {
            backend: backend.as_ptr(),
            device: spec.device,
            threads: spec.threads,
        };
        let session_options = match create_option_map(&api, &spec.session_options, "session") {
            Ok(options) => options,
            Err(error) => {
                unsafe {
                    (api.model_free)(model);
                    (api.registry_free)(registry);
                }
                return Err(error);
            }
        };
        let mut session = std::ptr::null_mut();
        let session_status = unsafe {
            (api.session_create)(
                model,
                task.as_ptr(),
                mode.as_ptr(),
                &native_config,
                session_options,
                &mut session,
            )
        };
        unsafe { (api.options_free)(session_options) };
        if let Err(error) = api.check(session_status, "session creation") {
            unsafe {
                (api.model_free)(model);
                (api.registry_free)(registry);
            }
            return Err(error);
        }
        if session.is_null() {
            unsafe {
                (api.model_free)(model);
                (api.registry_free)(registry);
            }
            bail!("audio.cpp session creation returned a null handle");
        }

        Ok(Self {
            api,
            registry,
            model,
            session,
            steps: spec.steps,
            language,
            request_options,
        })
    }

    fn generate(&mut self, text: &str, speed: f32, voice: i32) -> Result<Audio> {
        let text = CString::new(text).context("synthesis text contains a NUL byte")?;
        let voice = CString::new(voice_name(voice)?).expect("voice name has no NUL");
        let steps = CString::new(self.steps.to_string()).expect("integer has no NUL");
        let steps_key = CString::new("num_inference_steps").expect("static string has no NUL");
        let request = unsafe { (self.api.request_create)() };
        if request.is_null() {
            bail!("audio.cpp request creation returned a null handle");
        }

        let mut result = std::ptr::null_mut();
        let operation = (|| {
            self.api.check(
                unsafe {
                    (self.api.request_set_text)(request, text.as_ptr(), self.language.as_ptr())
                },
                "text configuration",
            )?;
            self.api.check(
                unsafe { (self.api.request_set_voice_id)(request, voice.as_ptr()) },
                "voice configuration",
            )?;
            self.api.check(
                unsafe { (self.api.request_set_speaking_rate)(request, speed) },
                "speaking-rate configuration",
            )?;
            self.api.check(
                unsafe {
                    (self.api.request_set_option)(request, steps_key.as_ptr(), steps.as_ptr())
                },
                "generation-step configuration",
            )?;
            for (key, value) in &self.request_options {
                self.api.check(
                    unsafe { (self.api.request_set_option)(request, key.as_ptr(), value.as_ptr()) },
                    "request option configuration",
                )?;
            }
            self.api.check(
                unsafe { (self.api.session_run)(self.session, request, &mut result) },
                "synthesis",
            )?;
            if result.is_null() {
                bail!("audio.cpp synthesis returned a null result handle");
            }

            let mut samples = std::ptr::null();
            let mut frames = 0usize;
            let mut sample_rate = 0;
            let mut channels = 0;
            self.api.check(
                unsafe {
                    (self.api.result_audio)(
                        result,
                        &mut samples,
                        &mut frames,
                        &mut sample_rate,
                        &mut channels,
                    )
                },
                "audio result",
            )?;
            let sample_count = validate_audio_shape(samples, frames, sample_rate, channels)?;
            let pcm = unsafe { std::slice::from_raw_parts(samples, sample_count) }.to_vec();
            if pcm.iter().any(|sample| !sample.is_finite()) {
                bail!("audio.cpp returned non-finite PCM");
            }
            Ok(Audio { sample_rate, pcm })
        })();

        if !result.is_null() {
            unsafe { (self.api.result_free)(result) };
        }
        unsafe { (self.api.request_free)(request) };
        operation
    }
}

impl Drop for NativeEngine {
    fn drop(&mut self) {
        unsafe {
            (self.api.session_free)(self.session);
            (self.api.model_free)(self.model);
            (self.api.registry_free)(self.registry);
        }
    }
}

fn validate_audio_shape(
    samples: *const f32,
    frames: usize,
    sample_rate: i32,
    channels: i32,
) -> Result<usize> {
    if sample_rate != SUPERTONIC_SAMPLE_RATE {
        bail!(
            "audio.cpp returned sample rate {sample_rate}; expected {SUPERTONIC_SAMPLE_RATE} for Supertonic"
        );
    }
    if channels != 1 {
        bail!("audio.cpp returned {channels} channels; Omaspeak requires mono PCM");
    }
    let samples_len = frames
        .checked_mul(channels as usize)
        .context("audio.cpp PCM length overflow")?;
    if samples_len == 0 {
        bail!("audio.cpp returned empty PCM");
    }
    if samples_len > MAX_PCM_SAMPLES {
        bail!("audio.cpp returned {samples_len} PCM samples; limit is {MAX_PCM_SAMPLES}");
    }
    if samples.is_null() {
        bail!("audio.cpp returned a null PCM pointer");
    }
    Ok(samples_len)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct WorkerSpec {
    library: PathBuf,
    library_dirs: Vec<PathBuf>,
    model: PathBuf,
    backend: String,
    device: c_int,
    threads: c_int,
    language: String,
    steps: i32,
    load_options: BTreeMap<String, String>,
    session_options: BTreeMap<String, String>,
    request_options: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ProviderProbeSpec {
    library: PathBuf,
    library_dirs: Vec<PathBuf>,
}

#[derive(Debug, Deserialize, Serialize)]
enum WorkerRequest {
    Generate {
        text: String,
        speed: f32,
        voice: i32,
    },
    Shutdown,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WorkerResponse {
    Ready { sample_rate: i32, voices: i32 },
    Audio { sample_rate: i32, samples: usize },
    Error { message: String },
    Shutdown,
}

struct Audio {
    sample_rate: i32,
    pcm: Vec<f32>,
}

struct WorkerClient {
    child: Child,
    input: Option<TimedWriter>,
    output: Option<TimedReader>,
    stopped: bool,
}

struct TimedReader {
    reader: ChildStdout,
    timeout: Duration,
}

impl Read for TimedReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        wait_for_io(self.reader.as_raw_fd(), libc::POLLIN, self.timeout).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("wait for audio.cpp worker IPC read: {error}"),
            )
        })?;
        self.reader.read(buffer).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("read audio.cpp worker IPC pipe: {error}"),
            )
        })
    }
}

struct TimedWriter {
    writer: ChildStdin,
    timeout: Duration,
}

impl Write for TimedWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        wait_for_io(self.writer.as_raw_fd(), libc::POLLOUT, self.timeout).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("wait for audio.cpp worker IPC write: {error}"),
            )
        })?;
        self.writer.write(buffer).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("write audio.cpp worker IPC pipe: {error}"),
            )
        })
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

fn wait_for_io(fd: c_int, events: libc::c_short, timeout: Duration) -> io::Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let milliseconds = if remaining.is_zero() {
            0
        } else {
            i32::try_from(remaining.as_millis().max(1)).unwrap_or(i32::MAX)
        };
        let mut descriptor = libc::pollfd {
            fd,
            events,
            revents: 0,
        };
        let status = unsafe { libc::poll(&mut descriptor, 1, milliseconds) };
        if status > 0 {
            if descriptor.revents & libc::POLLNVAL != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "audio.cpp worker IPC descriptor is invalid",
                ));
            }
            return Ok(());
        }
        if status == 0 {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "audio.cpp worker IPC timed out",
            ));
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

impl WorkerClient {
    fn launch(spec: &WorkerSpec) -> Result<Self> {
        let executable = std::env::current_exe().context("locate Omaspeak executable")?;
        let encoded =
            serde_json::to_string(spec).context("encode audio.cpp worker configuration")?;
        let mut command = Command::new(executable);
        command
            .arg("__audiocpp-worker")
            .arg("--spec")
            .arg(encoded)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let loader_path = std::env::join_paths(&spec.library_dirs)
            .context("encode audio.cpp worker native library path")?;
        command.env("LD_LIBRARY_PATH", loader_path);
        let mut child = command
            .spawn()
            .context("start supervised audio.cpp worker")?;
        let input = child
            .stdin
            .take()
            .context("audio.cpp worker stdin is unavailable")?;
        let output = child
            .stdout
            .take()
            .context("audio.cpp worker stdout is unavailable")?;
        let mut client = Self {
            child,
            input: Some(TimedWriter {
                writer: input,
                timeout: WORKER_TIMEOUT,
            }),
            output: Some(TimedReader {
                reader: output,
                timeout: WORKER_TIMEOUT,
            }),
            stopped: false,
        };
        let ready: WorkerResponse = match client.read_response() {
            Ok(ready) => ready,
            Err(error) => {
                let _ = client.stop();
                return Err(error).context("read audio.cpp worker startup response");
            }
        };
        match ready {
            WorkerResponse::Ready {
                sample_rate: SUPERTONIC_SAMPLE_RATE,
                voices: SUPERTONIC_VOICES,
            } => Ok(client),
            WorkerResponse::Error { message } => {
                let _ = client.stop();
                bail!("audio.cpp worker initialization failed: {message}")
            }
            other => {
                let _ = client.stop();
                bail!("audio.cpp worker returned invalid startup response {other:?}")
            }
        }
    }

    fn write_request(&mut self, request: &WorkerRequest) -> Result<()> {
        write_json_frame(
            self.input
                .as_mut()
                .context("audio.cpp worker input is closed")?,
            request,
        )
    }

    fn read_response(&mut self) -> Result<WorkerResponse> {
        read_json_frame(
            self.output
                .as_mut()
                .context("audio.cpp worker output is closed")?,
        )
    }

    fn request(&mut self, request: &WorkerRequest) -> Result<std::result::Result<Audio, String>> {
        self.write_request(request)?;
        let response = self.read_response()?;
        match response {
            WorkerResponse::Audio {
                sample_rate,
                samples,
            } => {
                if sample_rate != SUPERTONIC_SAMPLE_RATE {
                    bail!("audio.cpp worker returned unexpected sample rate {sample_rate}");
                }
                Ok(Ok(Audio {
                    sample_rate,
                    pcm: read_pcm(
                        self.output
                            .as_mut()
                            .context("audio.cpp worker output is closed")?,
                        samples,
                    )?,
                }))
            }
            WorkerResponse::Error { message } => Ok(Err(message)),
            other => bail!("audio.cpp worker returned invalid synthesis response {other:?}"),
        }
    }

    fn stop(&mut self) -> Result<()> {
        if self.stopped {
            return Ok(());
        }
        self.stopped = true;
        if let Some(input) = self.input.as_mut() {
            input.timeout = WORKER_STOP_TIMEOUT;
        }
        if let Some(output) = self.output.as_mut() {
            output.timeout = WORKER_STOP_TIMEOUT;
        }
        let _ = self.write_request(&WorkerRequest::Shutdown);
        let response = self.read_response();
        if matches!(response, Ok(WorkerResponse::Shutdown))
            && wait_for_exit(&mut self.child, WORKER_STOP_TIMEOUT)?
        {
            return Ok(());
        }
        self.input.take();
        self.output.take();
        if wait_for_exit(&mut self.child, WORKER_STOP_TIMEOUT)? {
            return Ok(());
        }
        match self.child.kill() {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::InvalidInput => {}
            Err(error) => return Err(error).context("terminate unresponsive audio.cpp worker"),
        }
        if !wait_for_exit(&mut self.child, WORKER_STOP_TIMEOUT)? {
            let _ = self.child.kill();
            bail!(
                "audio.cpp worker did not exit after termination; refusing to start a replacement"
            );
        }
        Ok(())
    }
}

impl Drop for WorkerClient {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

fn wait_for_exit(child: &mut Child, timeout: Duration) -> Result<bool> {
    let deadline = Instant::now() + timeout;
    poll_until(deadline, || {
        child
            .try_wait()
            .map(|status| status.is_some())
            .context("reap audio.cpp worker")
    })
}

fn poll_until(deadline: Instant, mut poll: impl FnMut() -> Result<bool>) -> Result<bool> {
    loop {
        if poll()? {
            return Ok(true);
        }
        if Instant::now() >= deadline {
            return Ok(false);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// A process-isolated audio.cpp backend. One worker, model, and session stay
/// warm for the lifetime of this value.
pub struct AudioCppBackend {
    spec: WorkerSpec,
    worker: Mutex<WorkerClient>,
}

impl AudioCppBackend {
    pub fn create(config: &Config, paths: &AppPaths, runtime: Runtime) -> Result<Self> {
        if config.model.family != "supertonic" {
            bail!(
                "the audio.cpp backend requires model.family=\"supertonic\"; got {:?}",
                config.model.family
            );
        }
        let backend = match runtime {
            Runtime::Default => "cpu",
            Runtime::Cuda => "cuda",
            Runtime::Vulkan => "vulkan",
            Runtime::Hip => "hip",
            Runtime::Openvino => {
                bail!("runtime=openvino uses Omaspeak's direct OpenVINO provider")
            }
        };
        if config.model.steps <= 0 {
            bail!("model.steps must be positive");
        }
        let options = scoped_options(&config.backend.options)?;
        let library = resolve_provider_library(config, &paths.config_file)?;
        let spec = WorkerSpec {
            library_dirs: resolve_library_dirs(config, &paths.config_file, &library)?,
            library,
            model: resolve_model_file(config, paths)?,
            backend: backend.into(),
            device: config
                .backend
                .device_id
                .try_into()
                .context("device_id is too large")?,
            threads: config.backend.threads.into(),
            language: config.model.language.clone(),
            steps: config.model.steps,
            load_options: options.load,
            session_options: options.session,
            request_options: options.request,
        };
        let worker = WorkerClient::launch(&spec)?;
        Ok(Self {
            spec,
            worker: Mutex::new(worker),
        })
    }
}

impl TtsBackend for AudioCppBackend {
    fn kind(&self) -> &'static str {
        "audiocpp"
    }

    fn sample_rate(&self) -> i32 {
        SUPERTONIC_SAMPLE_RATE
    }

    fn num_voices(&self) -> i32 {
        SUPERTONIC_VOICES
    }

    fn generate(&self, text: &str, speed: f32, voice: i32) -> Result<Vec<f32>> {
        let mut worker = self
            .worker
            .lock()
            .map_err(|_| anyhow!("audio.cpp worker lock is poisoned"))?;
        let request = WorkerRequest::Generate {
            text: text.into(),
            speed,
            voice,
        };
        match worker.request(&request) {
            Ok(Ok(audio)) => Ok(audio.pcm),
            Ok(Err(message)) => bail!("audio.cpp synthesis failed: {message}"),
            Err(first_error) => {
                worker.stop().with_context(|| {
                    format!("stop failed audio.cpp worker before restart: {first_error:#}")
                })?;
                let replacement = WorkerClient::launch(&self.spec).with_context(|| {
                    format!("restart audio.cpp worker after IPC failure: {first_error:#}")
                })?;
                *worker = replacement;
                match worker.request(&request)? {
                    Ok(audio) => Ok(audio.pcm),
                    Err(message) => {
                        bail!("audio.cpp synthesis failed after worker restart: {message}")
                    }
                }
            }
        }
    }
}

pub fn run_worker(spec_json: &str) -> Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    if let Err(error) = disable_core_dumps() {
        let _ = write_json_frame(
            &mut output,
            &WorkerResponse::Error {
                message: format!("{error:#}"),
            },
        );
        return Err(error);
    }
    let spec: WorkerSpec =
        serde_json::from_str(spec_json).context("decode audio.cpp worker spec")?;
    let mut engine = match NativeEngine::load(&spec) {
        Ok(engine) => engine,
        Err(error) => {
            let _ = write_json_frame(
                &mut output,
                &WorkerResponse::Error {
                    message: format!("{error:#}"),
                },
            );
            return Err(error);
        }
    };
    write_json_frame(
        &mut output,
        &WorkerResponse::Ready {
            sample_rate: SUPERTONIC_SAMPLE_RATE,
            voices: SUPERTONIC_VOICES,
        },
    )?;
    loop {
        let request: WorkerRequest = read_json_frame(&mut input)?;
        match request {
            WorkerRequest::Generate { text, speed, voice } => {
                match engine.generate(&text, speed, voice) {
                    Ok(audio) => {
                        write_json_frame(
                            &mut output,
                            &WorkerResponse::Audio {
                                sample_rate: audio.sample_rate,
                                samples: audio.pcm.len(),
                            },
                        )?;
                        write_pcm(&mut output, &audio.pcm)?;
                    }
                    Err(error) => write_json_frame(
                        &mut output,
                        &WorkerResponse::Error {
                            message: format!("{error:#}"),
                        },
                    )?,
                }
            }
            WorkerRequest::Shutdown => {
                write_json_frame(&mut output, &WorkerResponse::Shutdown)?;
                return Ok(());
            }
        }
    }
}

/// Validate the configured provider ABI in a short-lived hardened process.
/// This does not need a model, so runtime-only setup can stage the provider
/// before model setup without loading optional native code into the caller.
pub fn probe_provider(config: &Config, config_file: &Path) -> Result<PathBuf> {
    let library = resolve_provider_library(config, config_file)?;
    let spec = ProviderProbeSpec {
        library_dirs: resolve_library_dirs(config, config_file, &library)?,
        library: library.clone(),
    };
    let executable = std::env::current_exe().context("locate Omaspeak executable")?;
    let encoded = serde_json::to_string(&spec).context("encode audio.cpp provider probe")?;
    let loader_path = std::env::join_paths(&spec.library_dirs)
        .context("encode audio.cpp provider probe library path")?;
    let mut child = Command::new(executable)
        .arg("__audiocpp-probe")
        .arg("--spec")
        .arg(encoded)
        .env("LD_LIBRARY_PATH", loader_path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("start isolated audio.cpp provider probe")?;
    let deadline = Instant::now() + PROVIDER_PROBE_TIMEOUT;
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .context("wait for audio.cpp provider probe")?
        {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            if !wait_for_exit(&mut child, WORKER_STOP_TIMEOUT)? {
                bail!("audio.cpp provider probe did not stop after its timeout");
            }
            bail!("audio.cpp provider probe timed out");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let mut stderr = String::new();
    if let Some(mut stream) = child.stderr.take() {
        stream
            .read_to_string(&mut stderr)
            .context("read audio.cpp provider probe error")?;
    }
    if !status.success() {
        let detail = stderr.trim();
        bail!(
            "audio.cpp provider probe failed{}",
            if detail.is_empty() {
                format!(" with {status}")
            } else {
                format!(": {detail}")
            }
        );
    }
    Ok(library)
}

pub fn run_provider_probe(spec_json: &str) -> Result<()> {
    disable_core_dumps()?;
    let spec: ProviderProbeSpec =
        serde_json::from_str(spec_json).context("decode audio.cpp provider probe spec")?;
    Api::load(&spec.library).map(|_| ())
}

/// Disable core collection before a hidden process touches optional native code.
pub fn disable_core_dumps() -> Result<()> {
    #[cfg(target_os = "linux")]
    let dumpable_result = unsafe { libc::prctl(libc::PR_SET_DUMPABLE, 0) };
    #[cfg(target_os = "linux")]
    let dumpable_error = std::io::Error::last_os_error();

    let limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    let rlimit_result = unsafe { libc::setrlimit(libc::RLIMIT_CORE, &limit) };
    let rlimit_error = std::io::Error::last_os_error();
    #[cfg(target_os = "linux")]
    if dumpable_result == 0 || rlimit_result == 0 {
        return Ok(());
    }
    #[cfg(target_os = "linux")]
    bail!(
        "disable audio.cpp worker core dumps: PR_SET_DUMPABLE failed: {dumpable_error}; RLIMIT_CORE failed: {rlimit_error}"
    );
    #[cfg(not(target_os = "linux"))]
    if rlimit_result == 0 {
        Ok(())
    } else {
        Err(rlimit_error).context("disable audio.cpp worker core dumps")
    }
}

fn resolve_provider_library(config: &Config, config_file: &Path) -> Result<PathBuf> {
    if let Some(configured) = config.backend.library.as_deref() {
        return resolve_file_beneath(
            configured,
            config_file.parent().unwrap_or_else(|| Path::new(".")),
            "backend.library",
        );
    }
    discover_provider_library(config, config_file)?.with_context(
        || "audio.cpp provider was not found; install the packaged default or set backend.library",
    )
}

/// Find an audio.cpp provider only in explicitly configured or package-owned
/// library directories. Ambient loader paths are intentionally excluded.
pub fn discover_provider_library(config: &Config, config_file: &Path) -> Result<Option<PathBuf>> {
    if let Some(configured) = config.backend.library.as_deref() {
        return resolve_file_beneath(
            configured,
            config_file.parent().unwrap_or_else(|| Path::new(".")),
            "backend.library",
        )
        .map(Some);
    }
    let report = crate::runtime::inspect(&config.backend, config_file);
    let mut directories = report.configured_library_dirs;
    directories.extend(report.package_library_dirs);
    let directories = directories
        .into_iter()
        .fold(Vec::new(), |mut unique, path| {
            if !unique.contains(&path) {
                unique.push(path);
            }
            unique
        });
    Ok(find_provider_library(&directories))
}

fn find_provider_library(directories: &[PathBuf]) -> Option<PathBuf> {
    for directory in directories {
        let direct = directory.join("libaudiocpp.so");
        if direct.is_file() {
            return direct.canonicalize().ok();
        }
    }
    let mut versioned = directories
        .iter()
        .flat_map(|directory| std::fs::read_dir(directory).into_iter().flatten().flatten())
        .map(|entry| entry.path())
        .filter_map(|path| {
            let version = path
                .is_file()
                .then(|| path.file_name()?.to_str()?.strip_prefix("libaudiocpp.so."))
                .flatten()?
                .split('.')
                .map(str::parse::<u64>)
                .collect::<std::result::Result<Vec<_>, _>>()
                .ok()?;
            (!version.is_empty()).then_some((version, path))
        })
        .collect::<Vec<_>>();
    versioned.sort_by(|(left, _), (right, _)| left.cmp(right));
    versioned
        .pop()
        .and_then(|(_, path)| path.canonicalize().ok())
}

fn resolve_library_dirs(
    config: &Config,
    config_file: &Path,
    library: &Path,
) -> Result<Vec<PathBuf>> {
    let base = config_file.parent().unwrap_or_else(|| Path::new("."));
    let mut canonical_base = None;
    let mut directories = Vec::with_capacity(config.backend.library_dirs.len() + 1);
    for configured in &config.backend.library_dirs {
        let explicit_absolute = configured.is_absolute();
        let candidate = if explicit_absolute {
            configured.clone()
        } else {
            base.join(configured)
        };
        let resolved = candidate.canonicalize().with_context(|| {
            format!("resolve backend.library_dirs entry {}", candidate.display())
        })?;
        if !resolved.is_dir() {
            bail!(
                "backend.library_dirs entry is not a directory: {}",
                resolved.display()
            );
        }
        if !explicit_absolute {
            if canonical_base.is_none() {
                canonical_base = Some(base.canonicalize().with_context(|| {
                    format!("resolve backend.library_dirs base {}", base.display())
                })?);
            }
            let relative_base = canonical_base.as_ref().expect("base was initialized");
            if !resolved.starts_with(relative_base) {
                bail!(
                    "backend.library_dirs entry escapes the config directory {}",
                    relative_base.display()
                );
            }
        }
        if !directories.contains(&resolved) {
            directories.push(resolved);
        }
    }
    let parent = library
        .parent()
        .context("backend.library has no parent directory")?
        .to_path_buf();
    if !directories.contains(&parent) {
        directories.push(parent);
    }
    Ok(directories)
}

fn resolve_model_file(config: &Config, paths: &AppPaths) -> Result<PathBuf> {
    if config.model.file.trim().is_empty() {
        bail!("model.file must name the audio.cpp GGUF model")
    }
    resolve_file_beneath(
        Path::new(&config.model.file),
        &config.model_directory(paths),
        "model.file",
    )
}

fn resolve_file_beneath(path: &Path, base: &Path, label: &str) -> Result<PathBuf> {
    let explicit_absolute = path.is_absolute();
    let candidate = if explicit_absolute {
        path.to_owned()
    } else {
        base.join(path)
    };
    let resolved = candidate
        .canonicalize()
        .with_context(|| format!("resolve {label} {}", candidate.display()))?;
    if !resolved.is_file() {
        bail!("{label} is not a regular file: {}", resolved.display());
    }
    if !explicit_absolute {
        let base = base
            .canonicalize()
            .with_context(|| format!("resolve {label} base directory {}", base.display()))?;
        if !resolved.starts_with(&base) {
            bail!("{label} escapes its base directory {}", base.display());
        }
    }
    Ok(resolved)
}

fn path_to_c_string(path: &Path, label: &str) -> Result<CString> {
    use std::os::unix::ffi::OsStrExt as _;
    CString::new(path.as_os_str().as_bytes())
        .with_context(|| format!("{label} contains a NUL byte"))
}

pub fn voice_name(voice: i32) -> Result<&'static str> {
    usize::try_from(voice)
        .ok()
        .and_then(|index| crate::voices::SUPERTONIC_PRESET_NAMES.get(index).copied())
        .with_context(|| format!("voice {voice} is outside the Supertonic speaker range 0..9"))
}

fn write_json_frame(mut writer: impl Write, value: &impl Serialize) -> Result<()> {
    let payload = serde_json::to_vec(value).context("encode audio.cpp worker frame")?;
    if payload.len() > MAX_CONTROL_FRAME {
        bail!("audio.cpp worker frame exceeds {MAX_CONTROL_FRAME} bytes");
    }
    writer.write_all(&(payload.len() as u32).to_le_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()?;
    Ok(())
}

fn read_json_frame<R: Read, T: DeserializeOwned>(mut reader: R) -> Result<T> {
    let mut length = [0u8; 4];
    reader.read_exact(&mut length)?;
    let length = u32::from_le_bytes(length) as usize;
    if length > MAX_CONTROL_FRAME {
        bail!("audio.cpp worker frame declares {length} bytes; limit is {MAX_CONTROL_FRAME}");
    }
    let mut payload = vec![0u8; length];
    reader.read_exact(&mut payload)?;
    serde_json::from_slice(&payload).context("decode audio.cpp worker frame")
}

fn write_pcm(mut writer: impl Write, pcm: &[f32]) -> Result<()> {
    if pcm.is_empty() || pcm.len() > MAX_PCM_SAMPLES || pcm.iter().any(|sample| !sample.is_finite())
    {
        bail!("refuse invalid audio.cpp PCM response");
    }
    let mut bytes = vec![0u8; pcm.len().min(4096) * size_of::<f32>()];
    for chunk in pcm.chunks(4096) {
        bytes.resize(std::mem::size_of_val(chunk), 0);
        for (sample, output) in chunk.iter().zip(bytes.as_chunks_mut::<4>().0) {
            output.copy_from_slice(&sample.to_le_bytes());
        }
        writer.write_all(&bytes)?;
    }
    writer.flush()?;
    Ok(())
}

fn read_pcm(mut reader: impl Read, samples: usize) -> Result<Vec<f32>> {
    if samples == 0 || samples > MAX_PCM_SAMPLES {
        bail!("audio.cpp worker declared {samples} PCM samples; limit is {MAX_PCM_SAMPLES}");
    }
    let mut pcm = Vec::with_capacity(samples);
    let mut bytes = vec![0u8; samples.min(4096) * size_of::<f32>()];
    while pcm.len() < samples {
        let count = (samples - pcm.len()).min(4096);
        bytes.resize(count * size_of::<f32>(), 0);
        reader.read_exact(&mut bytes)?;
        for input in bytes.as_chunks::<4>().0 {
            let sample = f32::from_le_bytes(*input);
            if !sample.is_finite() {
                bail!("audio.cpp worker returned non-finite PCM");
            }
            pcm.push(sample);
        }
    }
    Ok(pcm)
}

#[cfg(test)]
#[path = "../tests/unit/audio_cpp.rs"]
mod tests;
