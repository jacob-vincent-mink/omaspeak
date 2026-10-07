use super::*;

#[test]
fn misaki_conversion_matches_upstream_frontend_examples() {
    assert_eq!(
        to_misaki("aɪ aʊ dʒ eɪ tʃ ɔɪ oʊ ɜːɹ ɜː ɪə ɾ ʔ"),
        "I W ʤ A ʧ Y O ɜɹ ɜɹ iə T t"
    );
    assert_eq!(to_misaki("ɐ ɐb n̩ əl əlz ʔˌn̩ ɚ"), "ɐ əb ᵊn ᵊl əlz tn əɹ");
    assert_eq!(to_misaki("həlˈoʊ wˈɜːld"), "həlˈO wˈɜɹld");
}

#[test]
fn tokens_are_padded_and_long_input_never_truncates() {
    let vocab = BTreeMap::from([('a', 43), (' ', 16)]);
    let phonemes = format!("{} {}", "a".repeat(509), "a".repeat(700));
    let chunks = encode(&phonemes, &vocab).unwrap();
    assert!(chunks.len() >= 3);
    assert!(
        chunks
            .iter()
            .all(|ids| ids.len() <= 512 && ids.first() == Some(&0) && ids.last() == Some(&0))
    );
    assert_eq!(
        chunks.iter().flatten().filter(|id| **id == 43).count(),
        1209
    );
}

#[test]
fn unknown_phonemes_and_empty_frontend_are_errors() {
    let vocab = BTreeMap::from([('a', 43)]);
    assert!(encode("", &vocab).is_err());
    assert!(
        encode("a☃", &vocab)
            .unwrap_err()
            .to_string()
            .contains("unsupported phoneme")
    );
}

#[test]
fn cpu_only_catalog_and_named_voice_are_explicit() {
    assert_eq!(
        crate::catalog::models()[0].id,
        crate::catalog::DEFAULT_MODEL_ID
    );
    assert_eq!(crate::catalog::backends()[0].kind, "audiocpp");
    let spec = crate::catalog::model(crate::catalog::PARADEE_MODEL_ID).unwrap();
    assert!(spec.compatible_with("paradee-openvino", Runtime::Openvino, "CPU"));
    for device in ["gpu", "npu", "auto"] {
        assert!(!spec.compatible_with("paradee-openvino", Runtime::Openvino, device));
    }
    assert_eq!(spec.voices.len(), 1);
    assert_eq!(spec.voices[0].name, "af_heart");
    assert_eq!(spec.download_size(), 36_987_891);
}

#[test]
fn model_asset_paths_cannot_escape_directory() {
    assert!(model_asset(Path::new("/tmp"), "../file").is_err());
    assert!(model_asset(Path::new("/tmp"), "/file").is_err());
    assert!(model_asset(Path::new("/tmp"), "").is_err());
}

#[test]
fn execution_evidence_requires_only_cpu_devices() {
    assert!(cpu_execution("CPU"));
    assert!(cpu_execution("[CPU.0, CPU.1]"));
    for devices in ["", "GPU", "CPU,GPU", "CPU.UNKNOWN"] {
        assert!(!cpu_execution(devices));
    }
}

#[cfg(target_os = "linux")]
fn frontend_fixture(body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "omaspeak-paradee-frontend-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    fs::write(&path, format!("#!/bin/sh\nexec python3 -c '{body}'\n")).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
    path
}

#[test]
#[cfg(target_os = "linux")]
fn frontend_drains_output_while_writing_input_and_bounds_excess() {
    let frontend = frontend_fixture(
        "import sys; sys.stdout.write(\"a\"*40000); sys.stdout.flush(); sys.stdin.read()",
    );
    let phonemes = phonemize_with_deadline(
        &frontend,
        &"b".repeat(MAX_FRONTEND_INPUT),
        Duration::from_secs(2),
    )
    .unwrap();
    assert_eq!(phonemes.len(), 40000);
    fs::remove_file(frontend).unwrap();
    let frontend = frontend_fixture(
        "import sys; sys.stdout.write(\"a\"*100000); sys.stdout.flush(); sys.stdin.read()",
    );
    let error = phonemize_with_deadline(&frontend, "hello", Duration::from_secs(2)).unwrap_err();
    assert!(
        error.to_string().contains("excessive phonemes"),
        "{error:#}"
    );
    fs::remove_file(frontend).unwrap();
}

#[test]
#[cfg(target_os = "linux")]
fn frontend_deadline_covers_blocked_input_and_reaps_process() {
    let frontend = frontend_fixture("import time; time.sleep(30)");
    let started = Instant::now();
    let error = phonemize_with_deadline(
        &frontend,
        &"b".repeat(MAX_FRONTEND_INPUT),
        Duration::from_millis(100),
    )
    .unwrap_err();
    assert!(error.to_string().contains("timed out"), "{error:#}");
    assert!(started.elapsed() < Duration::from_secs(1));
    fs::remove_file(frontend).unwrap();
}

#[test]
#[cfg(target_os = "linux")]
fn frontend_dies_if_file_synthesis_parent_disappears() {
    const HELPER: &str = "OMASPEAK_TEST_PARADEE_FRONTEND";
    if let Some(frontend) = std::env::var_os(HELPER) {
        let _ = phonemize(Path::new(&frontend), "hello");
        return;
    }
    let marker = std::env::temp_dir().join(format!(
        "omaspeak-paradee-parent-death-{}",
        std::process::id()
    ));
    let _ = fs::remove_file(&marker);
    let body = format!(
        "import os,time; open(\"{}\",\"w\").write(str(os.getpid())); time.sleep(30)",
        marker.display()
    );
    let frontend = frontend_fixture(&body);
    let mut parent = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "paradee::tests::frontend_dies_if_file_synthesis_parent_disappears",
        ])
        .env(HELPER, &frontend)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let started = Instant::now();
    let pid = loop {
        let pid = fs::read_to_string(&marker)
            .ok()
            .and_then(|value| value.parse::<u32>().ok());
        if pid.is_some() || started.elapsed() >= Duration::from_secs(2) {
            break pid;
        }
        thread::sleep(Duration::from_millis(5));
    };
    let _ = parent.kill();
    let _ = parent.wait();
    let pid = pid.expect("frontend PID marker before killing synthesis parent");
    let exited = || {
        fs::read_to_string(format!("/proc/{pid}/stat"))
            .map(|stat| {
                stat.rsplit_once(')')
                    .is_some_and(|(_, fields)| fields.trim_start().starts_with('Z'))
            })
            .unwrap_or(true)
    };
    let started = Instant::now();
    while !exited() && started.elapsed() < Duration::from_secs(1) {
        thread::sleep(Duration::from_millis(5));
    }
    let dead = exited();
    if !dead {
        unsafe {
            libc::kill(pid as i32, libc::SIGKILL);
        }
    }
    fs::remove_file(frontend).unwrap();
    let _ = fs::remove_file(marker);
    assert!(dead, "frontend survived its synthesis parent");
}
