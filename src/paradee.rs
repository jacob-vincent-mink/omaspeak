//! Paradee FP32 ONNX executed by the existing OpenVINO runtime.
//! Native eSpeak NG supplies English G2P; its IPA is mapped to Misaki spelling.
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs;
use std::io::{ErrorKind, Read, Write};
use std::os::fd::AsRawFd;
#[cfg(target_os = "linux")]
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use openvino::{CompiledModel, Core, DeviceType, ElementType, PropertyKey, Shape, Tensor};
use serde::Deserialize;

use crate::{backend::Runtime, config::Config, engine::TtsBackend, paths::AppPaths};

const MAX_PHONEMES: usize = 510;
const MAX_SAMPLES: usize = 24_000 * 120;
const MAX_FRONTEND_OUTPUT: usize = 65_536;
const MAX_FRONTEND_INPUT: usize = 16_384;

#[derive(Deserialize)]
struct ModelMetadata {
    model_type: String,
    sample_rate: u32,
    vocab: BTreeMap<String, i64>,
}

pub struct ParadeeOpenvinoBackend {
    compiled: RefCell<CompiledModel>,
    vocab: BTreeMap<char, i64>,
    frontend: PathBuf,
}

impl ParadeeOpenvinoBackend {
    pub fn create(config: &Config, paths: &AppPaths) -> Result<Self> {
        ensure!(
            config.backend.runtime == Runtime::Openvino,
            "Paradee requires runtime=openvino"
        );
        ensure!(
            config.backend.canonical_device()? == "cpu",
            "Paradee currently supports only OpenVINO CPU; GPU conversion and NPU support are not qualified"
        );
        ensure!(
            config.model.family == "paradee",
            "Paradee backend requires model.family=paradee"
        );
        ensure!(
            matches!(
                config.model.language.to_ascii_lowercase().as_str(),
                "en" | "en-us"
            ),
            "Paradee supports American English only"
        );
        let directory = config.model_directory(paths);
        let graph = model_asset(&directory, &config.model.file)?;
        let metadata_path = model_asset(&directory, &config.model.tts_json)?;
        let metadata: ModelMetadata = serde_json::from_slice(&fs::read(&metadata_path)?)
            .context("read Paradee vocabulary")?;
        ensure!(
            metadata.model_type == "paradee" && metadata.sample_rate == 24_000,
            "invalid Paradee model metadata"
        );
        let mut vocab = BTreeMap::new();
        for (symbol, id) in metadata.vocab {
            let mut chars = symbol.chars();
            let ch = chars.next().context("empty Paradee vocabulary symbol")?;
            ensure!(
                chars.next().is_none() && (1..178).contains(&id),
                "invalid Paradee vocabulary entry"
            );
            vocab.insert(ch, id);
        }
        ensure!(vocab.contains_key(&' '), "Paradee vocabulary lacks a space");
        let frontend = frontend_path(config)?;
        // Exercise G2P data before declaring the backend usable.
        encode(&to_misaki(&phonemize(&frontend, "Hello.")?), &vocab)?;
        let locations = crate::runtime::inspect(&config.backend, &paths.config_file);
        let runtime = crate::runtime::resolve_openvino_runtime(
            &config.backend,
            &paths.config_file,
            &locations,
        )?;
        openvino_sys::library::load_from(&runtime.library)
            .map_err(anyhow::Error::msg)
            .context("load Paradee OpenVINO runtime")?;
        let mut core = Core::new_with_config(
            runtime
                .plugins
                .to_str()
                .context("non-UTF8 OpenVINO plugins path")?,
        )?;
        core.set_property(
            &DeviceType::CPU,
            &openvino::RwPropertyKey::InferenceNumThreads,
            &config.backend.threads.to_string(),
        )?;
        let model = core
            .read_model_from_file(graph.to_str().context("non-UTF8 Paradee model path")?, "")
            .context("import Paradee FP32 graph")?;
        let compiled = core
            .compile_model(&model, DeviceType::CPU)
            .context("compile Paradee on OpenVINO CPU")?;
        let execution = compiled
            .get_property(&PropertyKey::Other(Cow::Borrowed("EXECUTION_DEVICES")))
            .context("verify Paradee OpenVINO execution placement")?;
        ensure!(
            cpu_execution(&execution),
            "Paradee compiled on unexpected execution devices: {execution}"
        );
        Ok(Self {
            compiled: RefCell::new(compiled),
            vocab,
            frontend,
        })
    }

    fn infer(&self, ids: &[i64], speed: f32) -> Result<Vec<f32>> {
        ensure!(
            (3..=512).contains(&ids.len()),
            "invalid Paradee token length"
        );
        let mut tokens = Tensor::new(ElementType::I64, &Shape::new(&[1, ids.len() as i64])?)?;
        tokens.get_data_mut::<i64>()?.copy_from_slice(ids);
        let mut pace = Tensor::new(ElementType::F32, &Shape::new(&[1])?)?;
        pace.get_data_mut::<f32>()?[0] = speed;
        let mut compiled = self.compiled.borrow_mut();
        let mut request = compiled.create_infer_request()?;
        request.set_tensor("input_ids", &tokens)?;
        request.set_tensor("speed", &pace)?;
        request.infer().context("generate Paradee audio")?;
        let waveform = request.get_tensor("waveform")?;
        let shape = waveform.get_shape()?;
        ensure!(
            shape.get_dimensions().len() == 2 && shape.get_dimensions()[0] == 1,
            "Paradee returned non-mono audio"
        );
        let samples = waveform.get_data::<f32>()?;
        ensure!(
            !samples.is_empty()
                && samples.len() <= MAX_SAMPLES
                && samples.iter().all(|sample| sample.is_finite()),
            "Paradee returned invalid or excessive audio"
        );
        Ok(samples.to_vec())
    }
}

impl TtsBackend for ParadeeOpenvinoBackend {
    fn kind(&self) -> &'static str {
        "paradee-openvino"
    }
    fn sample_rate(&self) -> i32 {
        24_000
    }
    fn num_voices(&self) -> i32 {
        1
    }
    fn generate(&self, text: &str, speed: f32, voice: i32) -> Result<Vec<f32>> {
        let mut output = Vec::new();
        self.generate_stream(text, speed, voice, &mut |chunk| {
            output.extend_from_slice(chunk);
            Ok(())
        })?;
        Ok(output)
    }
    fn generate_stream(
        &self,
        text: &str,
        speed: f32,
        voice: i32,
        sink: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<()> {
        ensure!(voice == 0, "Paradee has only the af_heart voice (0)");
        ensure!(
            speed.is_finite() && (0.5..=2.0).contains(&speed),
            "Paradee speed must be between 0.5 and 2.0"
        );
        ensure!(!text.trim().is_empty(), "Paradee text must not be empty");
        for segment in crate::supertonic::chunk_text(text, 240) {
            let mut phonemes = to_misaki(&phonemize(&self.frontend, &segment)?);
            if let Some(last) = segment
                .trim()
                .chars()
                .last()
                .filter(|ch| ".!?;:".contains(*ch))
            {
                phonemes.push(last);
            }
            for ids in encode(&phonemes, &self.vocab)? {
                sink(&self.infer(&ids, speed)?)?;
            }
        }
        Ok(())
    }
}

fn model_asset(directory: &Path, value: &str) -> Result<PathBuf> {
    ensure!(!value.is_empty(), "Paradee model asset path is empty");
    let path = Path::new(value);
    ensure!(
        !path.is_absolute()
            && path
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_))),
        "Paradee assets must resolve inside the model directory"
    );
    let path = directory.join(path);
    ensure!(
        path.is_file(),
        "Paradee model asset is missing: {}",
        path.display()
    );
    Ok(path)
}

fn frontend_path(config: &Config) -> Result<PathBuf> {
    if let Some(value) = config.backend.options.get("g2p_executable") {
        let path = PathBuf::from(value);
        ensure!(
            path.is_absolute() && path.is_file(),
            "Paradee g2p_executable must be an absolute existing eSpeak NG executable"
        );
        return Ok(path);
    }
    std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
        .map(|directory| directory.join("espeak-ng"))
        .find(|path| path.is_file())
        .context("Paradee requires native espeak-ng and its English data; install espeak-ng or set backend.options.g2p_executable to its absolute path")
}

fn cpu_execution(value: &str) -> bool {
    let devices = value
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '.')
        .filter(|device| !device.is_empty())
        .collect::<Vec<_>>();
    !devices.is_empty()
        && devices.iter().all(|device| {
            *device == "CPU"
                || device
                    .strip_prefix("CPU.")
                    .is_some_and(|id| !id.is_empty() && id.chars().all(|ch| ch.is_ascii_digit()))
        })
}

fn nonblocking(fd: i32) -> Result<()> {
    // These descriptors belong exclusively to this frontend invocation.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    ensure!(
        flags >= 0,
        "read Paradee frontend pipe flags: {}",
        std::io::Error::last_os_error()
    );
    ensure!(
        unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } >= 0,
        "set Paradee frontend pipe nonblocking: {}",
        std::io::Error::last_os_error()
    );
    Ok(())
}

fn phonemize(executable: &Path, text: &str) -> Result<String> {
    phonemize_with_deadline(executable, text, Duration::from_secs(5))
}

fn phonemize_with_deadline(executable: &Path, text: &str, deadline: Duration) -> Result<String> {
    ensure!(
        text.len() <= MAX_FRONTEND_INPUT,
        "Paradee frontend input exceeds {MAX_FRONTEND_INPUT} bytes"
    );
    let mut command = Command::new(executable);
    command
        .args(["-q", "--ipa", "-v", "en-us", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    #[cfg(target_os = "linux")]
    {
        let parent = unsafe { libc::getpid() };
        // The frontend stays in its supervisor's process group. Also terminate
        // it if an unsupervised file-synthesis process disappears unexpectedly.
        unsafe {
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() != parent {
                    return Err(std::io::Error::from_raw_os_error(libc::EPIPE));
                }
                Ok(())
            });
        }
    }
    let mut child = command
        .spawn()
        .context("start Paradee eSpeak NG frontend")?;
    let started = Instant::now();
    let result = (|| {
        let mut input = Some(child.stdin.take().context("missing G2P stdin")?);
        let mut output = child.stdout.take().context("missing G2P stdout")?;
        nonblocking(input.as_ref().unwrap().as_raw_fd())?;
        nonblocking(output.as_raw_fd())?;
        let mut written = 0;
        let mut bytes = Vec::new();
        let mut eof = false;
        let mut status = None;
        loop {
            ensure!(
                started.elapsed() < deadline,
                "Paradee eSpeak NG frontend timed out"
            );
            if let Some(stdin) = input.as_mut() {
                match stdin.write(&text.as_bytes()[written..]) {
                    Ok(count) => {
                        written += count;
                        if written == text.len() {
                            input.take();
                        }
                    }
                    Err(error)
                        if matches!(
                            error.kind(),
                            ErrorKind::WouldBlock | ErrorKind::Interrupted
                        ) => {}
                    Err(error) => return Err(error).context("write Paradee frontend input"),
                }
            }
            if !eof {
                let mut buffer = [0_u8; 4096];
                loop {
                    match output.read(&mut buffer) {
                        Ok(0) => {
                            eof = true;
                            break;
                        }
                        Ok(count) => {
                            ensure!(
                                bytes.len() + count <= MAX_FRONTEND_OUTPUT,
                                "Paradee frontend returned excessive phonemes"
                            );
                            bytes.extend_from_slice(&buffer[..count]);
                        }
                        Err(error) if error.kind() == ErrorKind::WouldBlock => break,
                        Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                        Err(error) => return Err(error).context("read Paradee frontend output"),
                    }
                }
            }
            if status.is_none() {
                status = child.try_wait()?;
            }
            if let Some(status) = status {
                ensure!(
                    status.success(),
                    "Paradee eSpeak NG frontend failed ({status})"
                );
                if eof {
                    ensure!(
                        written == text.len(),
                        "Paradee frontend exited before consuming its text"
                    );
                    break;
                }
            }
            thread::sleep(Duration::from_millis(2));
        }
        let phonemes = String::from_utf8(bytes).context("invalid UTF-8 from Paradee frontend")?;
        Ok(phonemes.split_whitespace().collect::<Vec<_>>().join(" "))
    })();
    if result.is_err() {
        let _ = child.kill();
    }
    // Reap on every path, including timeout, invalid output and pipe failure.
    let _ = child.wait();
    result
}

fn boundary(ch: Option<char>) -> bool {
    ch.is_none_or(|ch| ch.is_whitespace() || ";:,.!?—…\"“”()".contains(ch))
}

/// Adapt the upstream Paradee browser frontend's American eSpeak spelling map.
/// This is its fallback frontend, not the full Python Misaki lexicon.
fn to_misaki(ipa: &str) -> String {
    let mut value = ipa.to_owned();
    for (from, to) in [
        ("ʔˌn̩", "ʔn"),
        ("ʔn̩", "ʔn"),
        ("aɪ", "I"),
        ("aʊ", "W"),
        ("dʒ", "ʤ"),
        ("eɪ", "A"),
        ("e", "A"),
        ("tʃ", "ʧ"),
        ("ɔɪ", "Y"),
        ("ʲo", "jo"),
        ("ʲə", "jə"),
        ("ʲ", ""),
        ("ɚ", "əɹ"),
        ("r", "ɹ"),
        ("x", "k"),
        ("ç", "k"),
        ("ɬ", "l"),
        ("̃", ""),
    ] {
        value = value.replace(from, to);
    }
    let mut syllables = String::new();
    for ch in value.chars() {
        if ch == '̩' {
            if let Some(previous) = syllables.pop() {
                if previous.is_whitespace() {
                    syllables.push(previous);
                } else {
                    syllables.push('ᵊ');
                    syllables.push(previous);
                }
            }
        } else {
            syllables.push(ch);
        }
    }
    let chars: Vec<_> = syllables.chars().collect();
    let mut mapped = String::new();
    let mut index = 0;
    while index < chars.len() {
        if chars[index] == 'ə'
            && chars.get(index + 1) == Some(&'l')
            && boundary(chars.get(index + 2).copied())
        {
            mapped.push_str("ᵊl");
            index += 2;
            continue;
        }
        mapped.push(
            if chars[index] == 'ɐ' && !boundary(chars.get(index + 1).copied()) {
                'ə'
            } else {
                chars[index]
            },
        );
        index += 1;
    }
    for (from, to) in [
        ("oʊ", "O"),
        ("ɜːɹ", "ɜɹ"),
        ("ɜː", "ɜɹ"),
        ("ɪə", "iə"),
        ("ː", ""),
        ("o", "ɔ"),
        ("ɾ", "T"),
        ("ʔ", "t"),
    ] {
        mapped = mapped.replace(from, to);
    }
    mapped
}

fn encode(phonemes: &str, vocab: &BTreeMap<char, i64>) -> Result<Vec<Vec<i64>>> {
    let chars: Vec<_> = phonemes.trim().chars().collect();
    ensure!(!chars.is_empty(), "Paradee frontend produced no phonemes");
    let mut result = Vec::new();
    let mut offset = 0;
    while offset < chars.len() {
        let limit = (offset + MAX_PHONEMES).min(chars.len());
        let end = if limit < chars.len() {
            (offset + 1..limit)
                .rev()
                .find(|index| boundary(Some(chars[*index])))
                .unwrap_or(limit)
        } else {
            limit
        };
        let mut ids = vec![0];
        for ch in &chars[offset..end] {
            ids.push(*vocab.get(ch).with_context(|| {
                format!("Paradee frontend produced unsupported phoneme {ch:?}")
            })?);
        }
        ids.push(0);
        result.push(ids);
        offset = end;
        while offset < chars.len() && chars[offset].is_whitespace() {
            offset += 1;
        }
    }
    if result.is_empty() {
        bail!("Paradee frontend produced no tokens");
    }
    Ok(result)
}

#[cfg(test)]
#[path = "../tests/unit/paradee.rs"]
mod tests;
