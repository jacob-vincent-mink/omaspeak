use super::*;
use std::collections::BTreeMap;
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};

fn captured_shell(script: &str) -> ChildOutput {
    let child = std::process::Command::new("sh")
        .args(["-c", script])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    collect_child_output(child, Duration::from_secs(2)).unwrap()
}

fn framed_npu_report() -> &'static str {
    r#"printf 'native stdout diagnostic\nOMASPEAK_NPU_PREPARATION_V1_BEGIN\n%s\nOMASPEAK_NPU_PREPARATION_V1_END\ntrailing native output' '{"compiled_models":10,"cache_blobs":["model.blob"],"loaded_from_cache_required":true}'"#
}

fn temp(name: &str) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "omaspeak-inventory-{name}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn resolves_one_complete_audio_provider() {
    let root = temp("resolve-audio");
    let library = root.join("libaudiocpp.so");
    fs::write(&library, b"provider").unwrap();
    let config = BackendConfig {
        runtime: Runtime::Vulkan,
        device: "gpu".into(),
        library: Some(library.clone()),
        library_dirs: vec![root.clone()],
        ..Default::default()
    };
    let exact = resolve(&config, &root.join("config.toml"));
    assert_eq!(exact.library, Some(library.canonicalize().unwrap()));
    assert!(exact.openvino_library.is_none());
    assert!(exact.openvino_plugins.is_none());
    assert!(exact.library_dirs.contains(&root.canonicalize().unwrap()));
}

#[test]
fn resolves_openvino_without_audio_provider() {
    let root = temp("resolve-openvino");
    let library = root.join("libopenvino_c.so");
    let plugins = root.join("plugins.xml");
    fs::write(&library, b"runtime").unwrap();
    fs::write(&plugins, b"plugins").unwrap();
    let config = BackendConfig {
        kind: "supertonic".into(),
        runtime: Runtime::Openvino,
        device: "npu".into(),
        library: Some(root.join("libaudiocpp.so")),
        openvino_library: Some(library.clone()),
        openvino_plugins: Some(plugins.clone()),
        ..Default::default()
    };
    let exact = resolve(&config, &root.join("config.toml"));
    assert!(exact.library.is_none());
    assert_eq!(
        exact.openvino_library,
        Some(library.canonicalize().unwrap())
    );
    assert_eq!(
        exact.openvino_plugins,
        Some(plugins.canonicalize().unwrap())
    );
}

#[test]
fn audio_child_reports_abi_only_until_model_proof() {
    let root = temp("audio-child");
    let library = root.join("libaudiocpp.so");
    fs::write(&library, b"fixture").unwrap();
    let config = BackendConfig {
        runtime: Runtime::Cuda,
        device: "gpu".into(),
        library: Some(library.clone()),
        ..Default::default()
    };
    let result = child_with(
        &config,
        |_| unreachable!(),
        |candidate| {
            assert_eq!(candidate.runtime, Runtime::Cuda);
            Ok(library.clone())
        },
    );
    assert!(result.loadable);
    assert!(result.ready);
    assert_eq!(result.device_accessible, None);
    assert!(!result.evidence.model_inference_verified);
    assert_eq!(result.evidence.provider_path, Some(library));
    assert_eq!(result.evidence.selected_device.as_deref(), Some("gpu"));
}

#[test]
fn openvino_child_reports_selected_accessible_device() {
    let root = temp("openvino-child");
    let library = root.join("libopenvino_c.so");
    let plugins = root.join("plugins.xml");
    fs::write(&library, b"runtime").unwrap();
    fs::write(&plugins, b"plugins").unwrap();
    let config = BackendConfig {
        kind: "supertonic".into(),
        runtime: Runtime::Openvino,
        device: "npu".into(),
        openvino_library: Some(library.clone()),
        openvino_plugins: Some(plugins.clone()),
        ..Default::default()
    };
    let result = child_with(
        &config,
        |paths| {
            assert_eq!(paths.library, library);
            assert_eq!(paths.plugins, plugins);
            Ok(("OpenVINO fixture".into(), vec!["CPU".into(), "NPU".into()]))
        },
        |_| unreachable!(),
    );
    assert!(result.ready && result.loadable);
    assert_eq!(result.device_accessible, Some(true));
    assert_eq!(result.evidence.selected_device.as_deref(), Some("NPU"));
    assert_eq!(result.evidence.available_devices, ["CPU", "NPU"]);
}

#[test]
fn openvino_child_rejects_unavailable_device() {
    let config = BackendConfig {
        kind: "supertonic".into(),
        runtime: Runtime::Openvino,
        device: "npu".into(),
        openvino_library: Some("/runtime/libopenvino_c.so".into()),
        openvino_plugins: Some("/runtime/plugins.xml".into()),
        ..Default::default()
    };
    let result = child_with(
        &config,
        |_| Ok(("OpenVINO fixture".into(), vec!["CPU".into()])),
        |_| unreachable!(),
    );
    assert!(!result.ready);
    assert!(result.errors[0].contains("unavailable"));
}

#[test]
fn openvino_auto_rejects_an_empty_device_inventory() {
    let config = BackendConfig {
        kind: "supertonic".into(),
        runtime: Runtime::Openvino,
        device: "auto".into(),
        openvino_library: Some("/runtime/libopenvino_c.so".into()),
        openvino_plugins: Some("/runtime/plugins.xml".into()),
        ..Default::default()
    };
    let result = child_with(
        &config,
        |_| Ok(("OpenVINO fixture".into(), Vec::new())),
        |_| unreachable!(),
    );
    assert!(!result.ready);
    assert!(result.errors[0].contains("no accessible devices"));
}

#[test]
fn probe_reports_each_missing_direct_openvino_component() {
    let root = temp("missing-openvino");
    let mut config = BackendConfig {
        kind: "supertonic".into(),
        runtime: Runtime::Openvino,
        device: "npu".into(),
        openvino_library: Some(root.join("missing-libopenvino_c.so")),
        openvino_plugins: Some(root.join("missing-plugins.xml")),
        ..Default::default()
    };
    let result = probe(&config, &root.join("config.toml"));
    assert!(result.errors[0].contains("OpenVINO C library"));
    let library = root.join("libopenvino_c.so");
    fs::write(&library, b"fixture").unwrap();
    config.openvino_library = Some(library);
    let result = probe(&config, &root.join("config.toml"));
    assert!(result.errors[0].contains("OpenVINO plugins.xml"));
}

#[test]
fn apply_only_saves_ready_candidates() {
    let root = temp("apply");
    let path = root.join("config.toml");
    let config = Config::default();
    let ready = Probe {
        loadable: true,
        ready: true,
        ..Default::default()
    };
    apply_with(&config, &path, true, |_, _| ready.clone()).unwrap();
    assert!(path.is_file());

    fs::remove_file(&path).unwrap();
    let rejected = Probe {
        errors: vec!["no provider".into()],
        ..Default::default()
    };
    assert!(apply_with(&config, &path, true, |_, _| rejected).is_err());
    assert!(!path.exists());
}

#[test]
fn inventory_lists_every_native_runtime_device_pair() {
    let root = temp("matrix");
    let config = BackendConfig {
        library: Some(root.join("missing-libaudiocpp.so")),
        ..Default::default()
    };
    let states = inventory(&config, &root.join("config.toml"));
    let pairs = states
        .iter()
        .map(|state| (state.runtime, state.device.as_str()))
        .collect::<Vec<_>>();
    assert!(pairs.contains(&("default", "cpu")));
    assert!(pairs.contains(&("cuda", "gpu")));
    assert!(pairs.contains(&("vulkan", "gpu")));
    assert!(pairs.contains(&("hip", "gpu")));
    assert!(pairs.contains(&("openvino", "npu")));
}

#[test]
fn candidate_source_precedence_is_deterministic() {
    assert_eq!(candidate_source(true, true, true, true), "configured");
    assert_eq!(candidate_source(false, true, true, true), "environment");
    assert_eq!(candidate_source(false, false, true, true), "package");
    assert_eq!(candidate_source(false, false, false, true), "system");
    assert_eq!(candidate_source(false, false, false, false), "candidate");
}

#[test]
fn probe_rejects_invalid_dependency_directories_before_native_load() {
    let root = temp("invalid-dirs");
    let provider = root.join("libaudiocpp.so");
    fs::write(&provider, b"fixture").unwrap();
    let config = BackendConfig {
        library: Some(provider),
        library_dirs: vec![root.join("missing")],
        ..Default::default()
    };
    let result = probe(&config, &root.join("config.toml"));
    assert!(!result.ready);
    assert!(result.errors[0].contains("invalid dependency directory"));
}

#[test]
fn evidence_defaults_do_not_claim_inference_or_devices() {
    let evidence = Evidence::default();
    assert!(!evidence.model_inference_verified);
    assert!(evidence.provider_path.is_none());
    assert_eq!(BTreeMap::<String, String>::new().len(), 0);
}

#[test]
fn npu_child_capture_preserves_diagnostics_and_bounds_execution() {
    let success = captured_shell(&format!(
        "{}; printf 'benign native diagnostic' >&2",
        framed_npu_report()
    ));
    assert_eq!(success.stderr, b"benign native diagnostic");
    let report = decode_npu_preparation(success).unwrap();
    assert_eq!(report.compiled_models, 10);
    assert_eq!(report.cache_blobs, [PathBuf::from("model.blob")]);
    assert!(report.loaded_from_cache_required);

    let failure = captured_shell("printf 'provider failed' >&2; exit 7");
    let error = decode_npu_preparation(failure).unwrap_err().to_string();
    assert!(error.contains("exit status: 7"));
    assert!(error.contains("native stderr: provider failed"));

    let silent_failure = captured_shell("exit 9");
    let error = decode_npu_preparation(silent_failure)
        .unwrap_err()
        .to_string();
    assert!(error.ends_with("exit status: 9"));

    let malformed = captured_shell("printf 'not-json'; printf 'parse context' >&2");
    let error = decode_npu_preparation(malformed).unwrap_err().to_string();
    assert!(error.contains("returned no result frame"));
    assert!(error.contains("native stdout: not-json"));
    assert!(error.contains("native stderr: parse context"));

    let malformed = captured_shell(
        "printf 'OMASPEAK_NPU_PREPARATION_V1_BEGIN\\nnot-json\\nOMASPEAK_NPU_PREPARATION_V1_END\\n'",
    );
    let error = decode_npu_preparation(malformed).unwrap_err().to_string();
    assert!(error.contains("read isolated NPU preparation evidence"));

    let incomplete =
        captured_shell("printf 'before\\nOMASPEAK_NPU_PREPARATION_V1_BEGIN\\nnot-finished'");
    let error = decode_npu_preparation(incomplete).unwrap_err().to_string();
    assert!(error.contains("incomplete result frame"));
    assert!(error.contains("native stdout: before"));

    let child = std::process::Command::new("sleep")
        .arg("0.1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let error = collect_child_output(child, Duration::from_millis(1)).unwrap_err();
    assert!(error.to_string().contains("timed out"));
}

#[test]
fn production_npu_cache_entry_point_skips_non_npu_configurations() {
    let root = temp("npu-skip");
    let paths = crate::paths::AppPaths {
        config_file: root.join("config.toml"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    };
    let mut config = Config::default();
    assert!(prepare_npu_cache(&mut config, &paths).unwrap().is_none());
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "cpu".into();
    assert!(prepare_npu_cache(&mut config, &paths).unwrap().is_none());
}

#[test]
fn npu_child_orchestration_forwards_resolved_runtime_and_cache_policy() {
    let root = temp("npu-child");
    let paths = crate::paths::AppPaths {
        config_file: root.join("config.toml"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    };
    let cache_dir = root.join("compiled");
    let request = NpuPreparationRequest {
        config: Config::default(),
        paths: paths.clone(),
        cache_dir: cache_dir.clone(),
        require_cache_hits: true,
    };
    let runtime_paths = runtime::OpenvinoRuntimePaths {
        library: root.join("libopenvino_c.so"),
        plugins: root.join("plugins.xml"),
    };
    let result = npu_child_with(
        request,
        |backend, path| {
            assert_eq!(backend.runtime, Runtime::Default);
            assert_eq!(path, paths.config_file);
            Ok(runtime_paths.clone())
        },
        |_, actual_paths, runtime, actual_cache, require_hits| {
            assert_eq!(actual_paths.config_file, paths.config_file);
            assert_eq!(runtime.library, runtime_paths.library);
            assert_eq!(actual_cache, cache_dir);
            assert!(require_hits);
            Ok(crate::supertonic::NpuNativePreparation {
                compiled_models: crate::supertonic::NPU_COMPILED_MODELS,
                cache_blobs: Vec::new(),
                loaded_from_cache_required: true,
            })
        },
    )
    .unwrap();
    assert!(result.loaded_from_cache_required);

    let error = npu_child_with(
        NpuPreparationRequest {
            config: Config::default(),
            paths,
            cache_dir,
            require_cache_hits: false,
        },
        |_, _| bail!("injected resolution failure"),
        |_, _, _, _, _| unreachable!(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("resolution failure"));
}

#[test]
fn production_npu_child_rejects_a_non_npu_request_before_model_execution() {
    let root = temp("production-child");
    let request = NpuPreparationRequest {
        config: Config::default(),
        paths: crate::paths::AppPaths {
            config_file: root.join("config.toml"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
            state_dir: root.join("state"),
            runtime_dir: root.join("run"),
        },
        cache_dir: root.join("compiled"),
        require_cache_hits: false,
    };
    assert!(npu_child(request).is_err());
}

#[test]
fn candidate_transactions_preserve_bytes_until_successful_apply() {
    let root = temp("apply-transaction");
    let path = root.join("config.toml");
    let original = b"# Keep this comment\n[backend]\ndevice = 'cpu'\n";
    fs::write(&path, original).unwrap();
    let mut candidate = Config::load(&path).unwrap();
    candidate.backend.device = "auto".into();
    for error in [
        "missing libraries",
        "wrong ABI",
        "provider registration failed",
        "device unavailable",
        "native probe terminated",
    ] {
        assert!(
            apply_with(&candidate, &path, true, |_, _| Probe {
                errors: vec![error.into()],
                ..Default::default()
            })
            .is_err()
        );
        assert_eq!(fs::read(&path).unwrap(), original);
    }
    let success = |_: &BackendConfig, _: &Path| Probe {
        ready: true,
        loadable: true,
        device_accessible: Some(true),
        ..Default::default()
    };
    apply_with(&candidate, &path, false, success).unwrap();
    assert_eq!(fs::read(&path).unwrap(), original);
    apply_with(&candidate, &path, true, success).unwrap();
    assert_eq!(Config::load(&path).unwrap().backend.device, "auto");
    assert!(!path.with_extension("toml.tmp").exists());
}

fn npu_cache_fixture(name: &str) -> (Config, crate::paths::AppPaths) {
    let root = temp(name);
    let paths = crate::paths::AppPaths {
        config_file: root.join("config/config.toml"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    };
    let mut config = Config::default();
    config.backend.kind = "supertonic".into();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "npu".into();
    config.model.name = format!("custom-{name}");
    config.model.file.clear();
    config.model.duration_predictor = "duration.onnx".into();
    config.model.text_encoder = "text.onnx".into();
    config.model.vector_estimator = "vector.onnx".into();
    config.model.vocoder = "vocoder.onnx".into();
    config.model.tts_json = "tts.json".into();
    config.model.unicode_indexer = "unicode.json".into();
    config.model.voice_style = "voice_styles".into();
    let model = config.model_directory(&paths);
    fs::create_dir_all(&model).unwrap();
    for file in [
        &config.model.duration_predictor,
        &config.model.text_encoder,
        &config.model.vector_estimator,
        &config.model.vocoder,
        &config.model.tts_json,
        &config.model.unicode_indexer,
    ] {
        fs::write(model.join(file), file.as_bytes()).unwrap();
    }
    fs::create_dir_all(model.join("voice_styles")).unwrap();
    for name in crate::catalog::SUPERTONIC_VOICE_NAMES {
        fs::write(
            model.join(format!("voice_styles/{name}.json")),
            name.as_bytes(),
        )
        .unwrap();
    }
    (config, paths)
}

fn fake_npu_blobs(cache_dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    fs::create_dir_all(cache_dir.join("nested"))?;
    let mut blobs = Vec::new();
    for index in 0..crate::supertonic::NPU_COMPILED_MODELS {
        let relative = PathBuf::from(format!("nested/model-{index}.blob"));
        fs::write(cache_dir.join(&relative), b"compiled")?;
        blobs.push(relative);
    }
    Ok(blobs)
}

fn add_fake_npu_blobs(cache_dir: &Path, start: usize, end: usize) -> anyhow::Result<Vec<PathBuf>> {
    fs::create_dir_all(cache_dir.join("nested"))?;
    for index in start..end {
        fs::write(
            cache_dir.join(format!("nested/model-{index}.blob")),
            b"compiled",
        )?;
    }
    Ok((0..end)
        .map(|index| PathBuf::from(format!("nested/model-{index}.blob")))
        .collect())
}

#[test]
fn npu_cache_transaction_verifies_cache_hits_and_restores_previous_data_on_failure() {
    let (config, paths) = npu_cache_fixture("rollback");
    let target = crate::supertonic::npu_cache_directory(&config, &paths).unwrap();
    fs::create_dir_all(&target).unwrap();
    fs::write(target.join("keep"), b"previous cache").unwrap();
    let calls = std::cell::RefCell::new(Vec::new());
    let error = prepare_npu_cache_with(&config, &paths, |request| {
        calls.borrow_mut().push(request.require_cache_hits);
        if request.require_cache_hits {
            bail!("injected cross-process cache miss");
        }
        let cache_blobs = fake_npu_blobs(&request.cache_dir)?;
        Ok(crate::supertonic::NpuNativePreparation {
            compiled_models: crate::supertonic::NPU_COMPILED_MODELS,
            cache_blobs,
            loaded_from_cache_required: false,
        })
    })
    .unwrap_err();
    assert!(format!("{error:#}").contains("cache miss"));
    assert_eq!(&*calls.borrow(), &[false, true]);
    assert_eq!(fs::read(target.join("keep")).unwrap(), b"previous cache");
    assert!(!target.join("omaspeak-npu-cache.json").exists());
}

#[test]
fn npu_cache_transaction_publishes_only_after_cache_hit_verification() {
    let (config, paths) = npu_cache_fixture("success");
    let calls = std::cell::RefCell::new(Vec::new());
    let state = prepare_npu_cache_with(&config, &paths, |request| {
        calls.borrow_mut().push(request.require_cache_hits);
        if !request.require_cache_hits {
            fake_npu_blobs(&request.cache_dir)?;
        }
        Ok(crate::supertonic::NpuNativePreparation {
            compiled_models: crate::supertonic::NPU_COMPILED_MODELS,
            cache_blobs: crate::supertonic::cache_blobs(&request.cache_dir)?,
            loaded_from_cache_required: request.require_cache_hits,
        })
    })
    .unwrap();
    assert!(state.ready, "{}", state.detail);
    assert_eq!(&*calls.borrow(), &[false, true]);
}

#[test]
fn npu_cache_transaction_reenters_a_partial_static_plan_before_publication() {
    let (config, paths) = npu_cache_fixture("multi-pass");
    let calls = std::cell::RefCell::new(Vec::new());
    let state = prepare_npu_cache_with(&config, &paths, |request| {
        calls.borrow_mut().push(request.require_cache_hits);
        let pass = calls.borrow().len();
        let cache_blobs = if request.require_cache_hits {
            crate::supertonic::cache_blobs(&request.cache_dir)?
        } else if pass == 1 {
            add_fake_npu_blobs(&request.cache_dir, 0, 5)?
        } else {
            add_fake_npu_blobs(
                &request.cache_dir,
                5,
                crate::supertonic::NPU_COMPILED_MODELS,
            )?
        };
        Ok(crate::supertonic::NpuNativePreparation {
            compiled_models: crate::supertonic::NPU_COMPILED_MODELS,
            cache_blobs,
            loaded_from_cache_required: request.require_cache_hits,
        })
    })
    .unwrap();
    assert!(state.ready, "{}", state.detail);
    assert_eq!(&*calls.borrow(), &[false, false, true]);
}

#[test]
fn npu_cache_transaction_revalidates_an_existing_ready_cache() {
    let (config, paths) = npu_cache_fixture("ready-idempotent");
    let target = crate::supertonic::npu_cache_directory(&config, &paths).unwrap();
    fake_npu_blobs(&target).unwrap();
    crate::supertonic::write_npu_cache_manifest(&config, &paths, &target).unwrap();
    let calls = std::cell::Cell::new(0);
    let state = prepare_npu_cache_with(&config, &paths, |request| {
        calls.set(calls.get() + 1);
        assert!(request.require_cache_hits);
        Ok(crate::supertonic::NpuNativePreparation {
            compiled_models: crate::supertonic::NPU_COMPILED_MODELS,
            cache_blobs: crate::supertonic::cache_blobs(&request.cache_dir)?,
            loaded_from_cache_required: true,
        })
    })
    .unwrap();
    assert!(state.ready);
    assert_eq!(calls.get(), 1);
}

#[test]
fn npu_cache_transaction_rejects_bad_reports_and_exhausted_persistence() {
    for (name, compiled_models, cache_blobs, expected) in [
        (
            "wrong-keys",
            crate::supertonic::NPU_COMPILED_MODELS - 1,
            vec![PathBuf::from("one.blob")],
            "graph-shape keys",
        ),
        (
            "no-blobs",
            crate::supertonic::NPU_COMPILED_MODELS,
            Vec::new(),
            "returned 0 compiled-model",
        ),
    ] {
        let (config, paths) = npu_cache_fixture(name);
        let error = prepare_npu_cache_with(&config, &paths, |_| {
            Ok(crate::supertonic::NpuNativePreparation {
                compiled_models,
                cache_blobs: cache_blobs.clone(),
                loaded_from_cache_required: false,
            })
        })
        .unwrap_err();
        assert!(format!("{error:#}").contains(expected));
    }

    let (config, paths) = npu_cache_fixture("exhausted");
    let calls = std::cell::Cell::new(0);
    let error = prepare_npu_cache_with(&config, &paths, |request| {
        calls.set(calls.get() + 1);
        let cache_blobs = add_fake_npu_blobs(&request.cache_dir, 0, 1)?;
        Ok(crate::supertonic::NpuNativePreparation {
            compiled_models: crate::supertonic::NPU_COMPILED_MODELS,
            cache_blobs,
            loaded_from_cache_required: false,
        })
    })
    .unwrap_err();
    assert_eq!(calls.get(), 5);
    assert!(format!("{error:#}").contains("after 5 complete static-plan passes"));
}

#[test]
fn npu_cache_transaction_atomically_replaces_an_invalid_previous_directory() {
    let (config, paths) = npu_cache_fixture("replace-invalid");
    let target = crate::supertonic::npu_cache_directory(&config, &paths).unwrap();
    fs::create_dir_all(&target).unwrap();
    fs::write(target.join("stale"), b"old").unwrap();
    let state = prepare_npu_cache_with(&config, &paths, |request| {
        let cache_blobs = if request.require_cache_hits {
            crate::supertonic::cache_blobs(&request.cache_dir)?
        } else {
            fake_npu_blobs(&request.cache_dir)?
        };
        Ok(crate::supertonic::NpuNativePreparation {
            compiled_models: crate::supertonic::NPU_COMPILED_MODELS,
            cache_blobs,
            loaded_from_cache_required: request.require_cache_hits,
        })
    })
    .unwrap();
    assert!(state.ready);
    assert!(!target.join("stale").exists());
    assert!(
        fs::read_dir(target.parent().unwrap())
            .unwrap()
            .all(|entry| !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with('.'))
    );
}
