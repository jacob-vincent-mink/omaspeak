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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Runtime;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture() -> (std::path::PathBuf, Config, AppPaths) {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("omaspeak-kokoro-test-{nonce}"));
        let model = directory.join("model");
        fs::create_dir_all(model.join("voices")).unwrap();
        for asset in [
            "openvino_model.xml",
            "openvino_model.bin",
            "config.json",
            "voices/af_heart.bin",
            "voices/am_michael.bin",
        ] {
            fs::write(model.join(asset), []).unwrap();
        }
        let interpreter = directory.join("fake-python");
        fs::write(
            &interpreter,
            r#"#!/usr/bin/env python3
import json, struct, sys
if '-u' not in sys.argv:
    print(json.dumps({'openvino': '2026.4.0', 'genai': '2026.4.0', 'device': 'NPU'}))
else:
    print(json.dumps({'ready': True, 'sample_rate': 24000, 'device': 'NPU'}), flush=True)
    for request in sys.stdin:
        request = json.loads(request)
        if request['text'] == 'error':
            print(json.dumps({'error': 'synthesis rejected'}), flush=True)
        elif request['text'] == 'nan':
            print(json.dumps({'ok': True, 'bytes': 4}), flush=True)
            sys.stdout.buffer.write(struct.pack('<f', float('nan')))
            sys.stdout.buffer.flush()
        else:
            payload = struct.pack('<ff', 0.25, -0.5)
            print(json.dumps({'ok': True, 'bytes': len(payload)}), flush=True)
            sys.stdout.buffer.write(payload)
            sys.stdout.buffer.flush()
"#,
        )
        .unwrap();
        let mut permissions = fs::metadata(&interpreter).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&interpreter, permissions).unwrap();
        let mut config = Config::default();
        config.model.name = crate::catalog::KOKORO_OPENVINO_MODEL_ID.into();
        config.model.directory = model.to_string_lossy().into_owned();
        config.backend.kind = "kokoro-genai".into();
        config.backend.runtime = Runtime::Openvino;
        config.backend.device = "npu".into();
        config
            .backend
            .options
            .insert("python".into(), interpreter.to_string_lossy().into_owned());
        (directory, config, AppPaths::discover())
    }

    #[test]
    fn probes_explicit_npu_and_reports_runtime_errors() {
        let (directory, mut config, _) = fixture();
        let found = probe(&config.backend);
        assert!(found.ready, "{:?}", found.errors);
        assert_eq!(found.evidence.selected_device.as_deref(), Some("NPU"));
        config.backend.runtime = Runtime::Default;
        assert!(probe(&config.backend).errors[0].contains("runtime=openvino"));
        config.backend.runtime = Runtime::Openvino;
        config.backend.options.insert(
            "python".into(),
            directory
                .join("missing-python")
                .to_string_lossy()
                .into_owned(),
        );
        assert!(probe(&config.backend).errors[0].contains("run Kokoro GenAI probe"));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn worker_confirms_device_decodes_audio_and_rejects_bad_samples() {
        let (directory, mut config, paths) = fixture();
        let backend = KokoroGenAiBackend::create(&config, &paths).unwrap();
        assert_eq!(backend.kind(), "kokoro-genai");
        assert_eq!(backend.sample_rate(), 24_000);
        assert_eq!(backend.num_voices(), 2);
        assert_eq!(backend.generate("hello", 1.0, 0).unwrap(), vec![0.25, -0.5]);
        assert!(
            backend
                .generate("error", 1.0, 0)
                .unwrap_err()
                .to_string()
                .contains("synthesis rejected")
        );
        assert!(
            backend
                .generate("nan", 1.0, 0)
                .unwrap_err()
                .to_string()
                .contains("non-finite")
        );
        drop(backend);
        config.backend.device = "gpu".into();
        assert!(
            KokoroGenAiBackend::create(&config, &paths)
                .err()
                .unwrap()
                .to_string()
                .contains("requested device")
        );
        config.model.name = "supertonic-3-gguf".into();
        assert!(
            KokoroGenAiBackend::create(&config, &paths)
                .err()
                .unwrap()
                .to_string()
                .contains("pinned Kokoro")
        );
        fs::remove_dir_all(directory).unwrap();
    }
}
