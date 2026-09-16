use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture {
    root: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "omaspeak-audio-tests-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(root.join("bin")).unwrap();
        let result = Self { root };
        result.script(
            "pw-dump",
            "#!/bin/sh\ncat \"$AUDIO_FIXTURE/inventory.json\"\n",
        );
        fs::write(result.root.join("inventory.json"), r#"[{"type":"PipeWire:Interface:Node","info":{"props":{"media.class":"Audio/Sink","node.name":"other-device","node.description":"USB device"}}},{"type":"PipeWire:Interface:Node","info":{"props":{"media.class":"Audio/Sink","node.name":"test-device","node.description":"USB device"}}}]"#).unwrap();
        result
    }
    fn script(&self, name: &str, source: &str) {
        let p = self.root.join("bin").join(name);
        fs::write(&p, source).unwrap();
        fs::set_permissions(p, fs::Permissions::from_mode(0o755)).unwrap();
    }
    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_omaspeak"))
            .arg("--config")
            .arg(self.root.join("config.toml"))
            .args(args)
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", self.root.join("bin").display()),
            )
            .env("AUDIO_FIXTURE", &self.root)
            .output()
            .unwrap()
    }
    fn ok(&self, args: &[&str]) -> String {
        let out = self.run(args);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn device_configuration_discovery_and_schema_are_model_free() {
    let f = Fixture::new();
    f.ok(&["config", "set", "audio.device", "pipewire:test-device"]);
    assert_eq!(
        f.ok(&["config", "get", "audio.device"]).trim(),
        "pipewire:test-device"
    );
    let report: serde_json::Value =
        serde_json::from_str(&f.ok(&["audio-devices", "--detailed", "--json"])).unwrap();
    assert_eq!(report["selected"], "pipewire:test-device");
    assert!(
        report["devices"]
            .as_array()
            .unwrap()
            .iter()
            .any(|d| d["selector"] == "pipewire:test-device" && d["available"] == true)
    );
    let schema: serde_json::Value =
        serde_json::from_str(&f.ok(&["config", "schema", "--json"])).unwrap();
    let audio = schema["keys"]
        .as_array()
        .unwrap()
        .iter()
        .find(|k| k["key"] == "audio.device")
        .unwrap();
    assert_eq!(audio["type"], "enum");
    assert_eq!(audio["restart_required"], true);
    assert!(
        audio["choices"]
            .as_array()
            .unwrap()
            .iter()
            .any(|d| d["value"] == "pipewire:test-device")
    );
    f.ok(&["config", "set", "audio.device", "pipewire:offline"]);
    let before = fs::read(f.root.join("config.toml")).unwrap();
    assert!(
        !f.run(&["config", "set", "audio.device", "pipewire:123"])
            .status
            .success()
    );
    assert_eq!(fs::read(f.root.join("config.toml")).unwrap(), before);
    f.ok(&["setup", "audio", "--device", "default"]);
    assert_eq!(fs::read(f.root.join("config.toml")).unwrap(), before);
    f.ok(&["setup", "audio", "--device", "default", "--apply"]);
    assert_eq!(f.ok(&["config", "get", "audio.device"]).trim(), "default");
    f.ok(&["config", "unset", "audio.device"]);
}

#[test]
fn disconnected_and_broken_inventory_remain_readable() {
    let f = Fixture::new();
    f.ok(&["config", "set", "audio.device", "pipewire:offline"]);
    f.script("pw-dump", "#!/bin/sh\nexit 1\n");
    let report: serde_json::Value =
        serde_json::from_str(&f.ok(&["audio-devices", "--detailed", "--json"])).unwrap();
    assert!(
        report["devices"]
            .as_array()
            .unwrap()
            .iter()
            .any(|d| d["selector"] == "pipewire:offline" && d["available"] == false)
    );
    f.ok(&["config", "schema", "--json"]);
    assert!(!f.run(&["setup", "audio", "--test"]).status.success());
    f.script("pw-dump", "#!/bin/sh\nprintf invalid\n");
    f.ok(&["audio-devices", "--detailed", "--json"]);
}

#[test]
fn pinned_playback_targets_one_output_and_never_falls_back() {
    let f = Fixture::new();
    f.script(
        "pw-play",
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$AUDIO_FIXTURE/args\"\nexit 0\n",
    );
    f.script(
        "aplay",
        "#!/bin/sh\ntouch \"$AUDIO_FIXTURE/fallback\"\nexit 0\n",
    );
    f.ok(&[
        "setup",
        "audio",
        "--device",
        "pipewire:test-device",
        "--test",
    ]);
    let args = fs::read_to_string(f.root.join("args")).unwrap();
    assert!(args.starts_with("--target\ntest-device\n--properties\n"));
    assert!(args.contains("node.dont-fallback = true"));
    assert!(args.contains("node.dont-reconnect = true"));
    assert!(args.contains("node.dont-move = true"));
    f.script("pw-play", "#!/bin/sh\nexit 1\n");
    let failed = f.run(&[
        "setup",
        "audio",
        "--device",
        "pipewire:test-device",
        "--test",
    ]);
    assert!(!failed.status.success());
    assert!(!f.root.join("fallback").exists());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("no alternate output"));
    assert!(!f.root.join("config.toml").exists());
}

#[test]
fn pinned_playback_timeout_reaps_the_child() {
    let f = Fixture::new();
    f.script(
        "pw-play",
        "#!/bin/sh\necho $$ > \"$AUDIO_FIXTURE/pid\"\nexec sleep 30\n",
    );
    let now = std::time::Instant::now();
    let out = f.run(&[
        "setup",
        "audio",
        "--device",
        "pipewire:test-device",
        "--test",
    ]);
    assert!(!out.status.success());
    assert!(now.elapsed().as_secs() < 10);
    assert!(String::from_utf8_lossy(&out.stderr).contains("timed out"));
    let pid: i32 = fs::read_to_string(f.root.join("pid"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
}
