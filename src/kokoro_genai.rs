//! Kokoro synthesis through a persistent OpenVINO GenAI Python worker.

use std::io::{BufRead, BufReader, Read, Write};
#[cfg(target_os = "linux")]
use std::os::unix::process::CommandExt;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};

use crate::backend::BackendConfig;
use crate::config::Config;
use crate::engine::TtsBackend;
use crate::paths::AppPaths;
use crate::runtime_inventory::{Evidence, Probe};

struct Worker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

pub struct KokoroGenAiBackend {
    worker: Mutex<Worker>,
}

fn python_command(config: &BackendConfig) -> String {
    config
        .options
        .get("python")
        .cloned()
        .or_else(|| std::env::var("OMASPEAK_KOKORO_PYTHON").ok())
        .unwrap_or_else(|| "python3".to_owned())
}

pub fn probe(config: &BackendConfig) -> Probe {
    let attempt = (|| -> Result<Probe> {
        ensure!(
            config.runtime == crate::backend::Runtime::Openvino,
            "Kokoro GenAI requires runtime=openvino"
        );
        let device = config.canonical_device()?.to_ascii_uppercase();
        ensure!(
            matches!(device.as_str(), "CPU" | "GPU" | "NPU"),
            "Kokoro GenAI requires an explicit CPU, GPU, or NPU device"
        );
        let python = python_command(config);
        let minimum = crate::catalog::model(crate::catalog::KOKORO_OPENVINO_MODEL_ID)
            .expect("Kokoro OpenVINO catalog profile exists")
            .min_openvino_version;
        let output = Command::new(&python)
            .arg("-c")
            .arg("import json,re,sys,openvino as ov,openvino_genai as genai; \
                 version=lambda s: tuple(map(int,re.match(r'^(\\d+)\\.(\\d+)\\.(\\d+)',s).groups())); \
                 minimum=version(sys.argv[1]); device=sys.argv[2]; \
                 assert version(ov.__version__)>=minimum, 'OpenVINO runtime is too old'; \
                 assert version(genai.__version__)>=minimum, 'OpenVINO GenAI is too old'; \
                 assert device in [d.split('.')[0] for d in ov.Core().available_devices], 'device is unavailable'; \
                 print(json.dumps({'openvino':ov.__version__,'genai':genai.__version__,'device':device}))")
            .arg(minimum)
            .arg(&device)
            .output()
            .with_context(|| format!("run Kokoro GenAI probe with {python}"))?;
        ensure!(
            output.status.success(),
            "Kokoro GenAI runtime probe failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
        let result: Value =
            serde_json::from_slice(&output.stdout).context("parse Kokoro GenAI runtime probe")?;
        Ok(Probe {
            loadable: true,
            device_accessible: Some(true),
            ready: true,
            evidence: Evidence {
                versions: vec![format!(
                    "OpenVINO {} · GenAI {}",
                    result["openvino"], result["genai"]
                )],
                provider_registration: true,
                available_devices: vec![device.clone()],
                selected_device: Some(device),
                provider_path: Some(python.into()),
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
        let device = config.backend.canonical_device()?.to_ascii_uppercase();
        ensure!(
            matches!(device.as_str(), "CPU" | "GPU" | "NPU"),
            "Kokoro OpenVINO requires an explicit CPU, GPU, or NPU device"
        );
        let python = python_command(&config.backend);
        let mut command = Command::new(&python);
        command
            .arg("-u")
            .arg("-c")
            .arg(include_str!("../scripts/kokoro-openvino-worker.py"))
            .arg(&model_dir)
            .arg(&device)
            .arg(
                crate::catalog::model(crate::catalog::KOKORO_OPENVINO_MODEL_ID)
                    .expect("Kokoro OpenVINO catalog profile exists")
                    .min_openvino_version,
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        #[cfg(target_os = "linux")]
        unsafe {
            command.pre_exec(|| {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut child = command
            .spawn()
            .with_context(|| format!("start Kokoro GenAI Python worker with {python}"))?;
        let stdin = child
            .stdin
            .take()
            .context("Kokoro worker stdin is unavailable")?;
        let stdout = child
            .stdout
            .take()
            .context("Kokoro worker stdout is unavailable")?;
        let mut worker = Worker {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        };
        let ready = worker
            .read_header()
            .context("load Kokoro OpenVINO GenAI worker")?;
        ensure!(
            ready["ready"] == true && ready["sample_rate"] == 24_000 && ready["device"] == device,
            "Kokoro worker did not confirm the requested device: {ready}"
        );
        Ok(Self {
            worker: Mutex::new(worker),
        })
    }
}

impl Worker {
    fn read_header(&mut self) -> Result<Value> {
        let mut line = String::new();
        ensure!(
            self.stdout.read_line(&mut line)? != 0,
            "Kokoro worker exited before responding"
        );
        serde_json::from_str(&line).context("parse Kokoro worker response")
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
        let mut worker = self
            .worker
            .lock()
            .map_err(|_| anyhow::anyhow!("Kokoro worker lock was poisoned"))?;
        let request = json!({"text": text, "speed": speed, "voice": voice});
        writeln!(worker.stdin, "{request}").context("send Kokoro synthesis request")?;
        worker
            .stdin
            .flush()
            .context("flush Kokoro synthesis request")?;
        let header = worker.read_header()?;
        if let Some(error) = header["error"].as_str() {
            bail!("Kokoro synthesis failed: {error}");
        }
        ensure!(
            header["ok"] == true,
            "unexpected Kokoro worker response: {header}"
        );
        let bytes = header["bytes"]
            .as_u64()
            .context("Kokoro response has no byte count")?;
        ensure!(
            bytes > 0 && bytes <= 512 * 1024 * 1024 && bytes % 4 == 0,
            "invalid Kokoro audio byte count {bytes}"
        );
        let mut data = vec![0u8; bytes as usize];
        worker
            .stdout
            .read_exact(&mut data)
            .context("read Kokoro audio payload")?;
        let samples = data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect::<Vec<_>>();
        ensure!(
            samples.iter().all(|sample| sample.is_finite()),
            "Kokoro returned non-finite audio"
        );
        Ok(samples)
    }
}

impl Drop for KokoroGenAiBackend {
    fn drop(&mut self) {
        if let Ok(mut worker) = self.worker.lock() {
            let _ = worker.child.kill();
            let _ = worker.child.wait();
        }
    }
}
