use super::*;
use std::fs;

fn captured_shell(script: &str) -> ChildOutput {
    let child = Command::new("sh")
        .args(["-c", script])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    collect_child_output(child, Duration::from_secs(2)).unwrap()
}

#[test]
fn npu_child_capture_suppresses_success_diagnostics_and_preserves_failures() {
    let success = captured_shell(
        r#"printf '%s' '{"compiled_models":12,"cache_blobs":["model.blob"],"loaded_from_cache_required":true}'; printf 'benign native diagnostic' >&2"#,
    );
    assert_eq!(success.stderr, b"benign native diagnostic");
    let report = decode_npu_preparation(success).unwrap();
    assert_eq!(report.compiled_models, 12);
    assert_eq!(report.cache_blobs, [PathBuf::from("model.blob")]);
    assert!(report.loaded_from_cache_required);

    let failure = captured_shell("printf 'provider failed' >&2; exit 7");
    let error = decode_npu_preparation(failure).unwrap_err().to_string();
    assert!(error.contains("exit status: 7"));
    assert!(error.contains("provider failed"));

    let malformed = captured_shell("printf 'not-json'; printf 'parse context' >&2");
    let error = decode_npu_preparation(malformed).unwrap_err().to_string();
    assert!(error.contains("read isolated NPU preparation evidence"));
    assert!(error.contains("parse context"));
}

#[test]
fn production_npu_cache_entry_point_skips_non_npu_configurations() {
    let root = env::temp_dir().join(format!("omaspeak-runtime-npu-skip-{}", std::process::id()));
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
    let root = env::temp_dir().join(format!("omaspeak-runtime-npu-child-{}", std::process::id()));
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
    let root = env::temp_dir().join(format!(
        "omaspeak-runtime-production-child-{}",
        std::process::id()
    ));
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
fn initialized_ort_core_cannot_silently_change_during_reload() {
    let core = test_ort_library();
    if !core.is_file() {
        return;
    }
    initialize_ort(&core).unwrap();
    initialize_ort(&core.canonicalize().unwrap()).unwrap();
    let other = env::temp_dir().join(format!("omaspeak-other-core-{}", std::process::id()));
    fs::write(&other, b"different candidate").unwrap();
    let error = initialize_ort(&other).unwrap_err().to_string();
    assert!(error.contains("restart the process"), "{error}");
    fs::remove_file(other).unwrap();
}

#[test]
fn production_cpu_probe_reports_the_pinned_runtime_without_a_subprocess() {
    let core = test_ort_library();
    if !core.is_file() {
        return;
    }
    let result = child(&BackendConfig {
        onnxruntime_library: Some(core),
        ..Default::default()
    });
    assert!(result.ready, "{:?}", result.errors);
    assert!(result.loadable);
    assert!(result.device_accessible);
    assert_eq!(result.evidence.versions, ["ONNX Runtime 1.30.0"]);
    assert_eq!(result.evidence.available_devices, ["cpu"]);
    assert_eq!(result.evidence.selected_device.as_deref(), Some("cpu"));
}

#[test]
fn candidate_transactions_preserve_bytes_until_successful_apply() {
    let root = env::temp_dir().join(format!("omaspeak-runtime-apply-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
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
        device_accessible: true,
        ..Default::default()
    };
    apply_with(&candidate, &path, false, success).unwrap();
    assert_eq!(fs::read(&path).unwrap(), original);
    apply_with(&candidate, &path, true, success).unwrap();
    assert_eq!(Config::load(&path).unwrap().backend.device, "auto");
    assert!(!path.with_extension("toml.tmp").exists());
    let missing = BackendConfig {
        onnxruntime_library: Some(root.join("missing")),
        ..Default::default()
    };
    assert!(!probe(&missing, &root.join("absent/config.toml")).ready);
    assert!(!root.join("absent").exists());
}

fn npu_cache_fixture(name: &str) -> (Config, crate::paths::AppPaths) {
    let root = env::temp_dir().join(format!(
        "omaspeak-npu-cache-transaction-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    let paths = crate::paths::AppPaths {
        config_file: root.join("config/config.toml"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    };
    let mut config = Config::default();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "npu".into();
    let model = config.model_directory(&paths);
    fs::create_dir_all(&model).unwrap();
    for file in [
        &config.model.duration_predictor,
        &config.model.text_encoder,
        &config.model.vector_estimator,
        &config.model.vocoder,
        &config.model.tts_json,
        &config.model.unicode_indexer,
        &config.model.voice_style,
    ] {
        fs::write(model.join(file), file.as_bytes()).unwrap();
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
            compiled_models: 12,
            cache_blobs,
            loaded_from_cache_required: false,
        })
    })
    .unwrap_err();
    assert!(format!("{error:#}").contains("cache miss"));
    assert_eq!(&*calls.borrow(), &[false, true]);
    assert_eq!(fs::read(target.join("keep")).unwrap(), b"previous cache");
    assert!(!target.join("omaspeak-npu-cache.json").exists());
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

#[test]
fn npu_cache_transaction_publishes_only_after_build_and_cache_hit_verification() {
    let (config, paths) = npu_cache_fixture("success");
    let calls = std::cell::RefCell::new(Vec::new());
    let state = prepare_npu_cache_with(&config, &paths, |request| {
        calls.borrow_mut().push(request.require_cache_hits);
        if !request.require_cache_hits {
            let _ = fake_npu_blobs(&request.cache_dir)?;
        } else {
            assert!(request.cache_dir.join("omaspeak-npu-cache.json").is_file());
            assert!(request.cache_dir.join("nested/model-0.blob").is_file());
        }
        Ok(crate::supertonic::NpuNativePreparation {
            compiled_models: 12,
            cache_blobs: (0..crate::supertonic::NPU_COMPILED_MODELS)
                .map(|index| PathBuf::from(format!("nested/model-{index}.blob")))
                .collect(),
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
    assert_eq!(state.blobs.len(), crate::supertonic::NPU_COMPILED_MODELS);
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
fn npu_cache_transaction_rejects_bad_child_reports_and_exhausted_persistence() {
    for (name, compiled_models, cache_blobs, expected) in [
        (
            "wrong-keys",
            11,
            vec![PathBuf::from("one.blob")],
            "graph-shape keys",
        ),
        ("no-blobs", 12, Vec::new(), "returned 0 compiled-model"),
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

#[test]
fn inventory_names_and_absent_optional_adapters_are_stable() {
    let path = env::temp_dir().join(format!("omaspeak-inventory-absent-{}", std::process::id()));
    let config = BackendConfig {
        onnxruntime_library: Some(path.join("missing-ort")),
        openvino_library: Some(path.join("missing-openvino")),
        provider_library: Some(path.join("missing-provider")),
        ..Default::default()
    };
    let states = inventory(&config, &path.join("config.toml"));
    assert_eq!(states.len(), 8);
    for state in states {
        assert!(state.supported && !state.probe.ready && !state.configured);
        let value = serde_json::to_value(state).unwrap();
        for key in [
            "supported",
            "discovered",
            "source",
            "configured",
            "loadable",
            "device_accessible",
            "ready",
            "paths",
            "evidence",
            "errors",
            "remediation",
        ] {
            assert!(value.get(key).is_some(), "{key}");
        }
    }
}

#[test]
fn candidate_sources_follow_the_documented_precedence() {
    assert_eq!(candidate_source(true, true, true, true), "configured");
    assert_eq!(candidate_source(false, true, true, true), "environment");
    assert_eq!(candidate_source(false, false, true, true), "package");
    assert_eq!(candidate_source(false, false, false, true), "system");
    assert_eq!(candidate_source(false, false, false, false), "candidate");
}

#[test]
fn missing_runtime_files_report_the_exact_component_setup_must_supply() {
    let root = env::temp_dir().join(format!(
        "omaspeak-missing-runtime-components-{}",
        std::process::id()
    ));
    fs::create_dir_all(&root).unwrap();
    let config_path = root.join("config.toml");

    let mut config = BackendConfig {
        onnxruntime_library: Some(root.join("missing-libonnxruntime.so")),
        ..Default::default()
    };
    let result = probe(&config, &config_path);
    assert!(result.errors[0].contains("ONNX Runtime 1.30.0 core library"));

    config.runtime = Runtime::Cuda;
    config.device = "gpu".into();
    let result = probe(&config, &config_path);
    assert!(result.errors[0].contains("packaged ONNX Runtime 1.30.0 core library"));

    let core = root.join("libonnxruntime.so");
    fs::write(&core, b"fixture").unwrap();
    config.onnxruntime_library = Some(core);
    config.provider_library = Some(root.join("missing-libonnxruntime_providers_cuda.so"));
    let result = probe(&config, &config_path);
    assert!(result.errors[0].contains("CUDA Plugin EP"));

    config.runtime = Runtime::Openvino;
    config.device = "npu".into();
    config.openvino_library = Some(root.join("missing-libopenvino_c.so"));
    config.openvino_plugins = Some(root.join("missing-plugins.xml"));
    let result = probe(&config, &config_path);
    assert!(result.errors[0].contains("OpenVINO C library"));

    let openvino = root.join("libopenvino_c.so");
    fs::write(&openvino, b"fixture").unwrap();
    config.openvino_library = Some(openvino);
    let result = probe(&config, &config_path);
    assert!(result.errors[0].contains("OpenVINO plugins.xml"));
}

#[test]
fn resolved_candidates_keep_runtime_overlay_dirs_without_ambient_loader_paths() {
    let root = env::temp_dir().join(format!("omaspeak-runtime-overlay-{}", std::process::id()));
    let configured = root.join("configured");
    let overlay = root.join("overlay");
    let ambient = root.join("ambient");
    for directory in [&configured, &overlay, &ambient] {
        fs::create_dir_all(directory).unwrap();
    }
    let core = configured.join("libonnxruntime.so.1.30.0");
    fs::write(&core, b"fixture").unwrap();

    let config = BackendConfig {
        library_dirs: vec![configured.clone()],
        onnxruntime_library: Some(core),
        ..Default::default()
    };
    let report = runtime::inspect_with(
        &config,
        &root.join("config.toml"),
        Some(env::join_paths([&overlay]).unwrap()),
        Some(env::join_paths([&ambient]).unwrap()),
        None,
    );
    let exact = resolve_with_locations(&config, report);
    assert!(
        exact
            .library_dirs
            .contains(&configured.canonicalize().unwrap())
    );
    assert!(
        exact
            .library_dirs
            .contains(&overlay.canonicalize().unwrap())
    );
    assert!(
        !exact
            .library_dirs
            .contains(&ambient.canonicalize().unwrap())
    );
}

#[test]
fn cuda_selection_uses_logical_ordinal_and_retains_hardware_identity() {
    let devices = vec![
        (0, 11794, "CUDA ordinal 0, hardware 11794 (GPU)".to_owned()),
        (1, 42, "CUDA ordinal 1, hardware 42 (GPU)".to_owned()),
    ];
    let (available, selected) = cuda_device_evidence(devices, 0).unwrap();
    assert_eq!(available.len(), 2);
    assert_eq!(selected, "CUDA ordinal 0, hardware 11794 (GPU)");
    assert!(cuda_device_evidence(vec![(0, 11794, "GPU 0".into())], 1).is_err());
}

#[test]
fn invalid_external_cuda_provider_fails_at_the_registration_boundary() {
    let core = test_ort_library();
    if !core.is_file() {
        return;
    }

    let root = env::temp_dir().join(format!(
        "omaspeak-invalid-cuda-provider-{}",
        std::process::id()
    ));
    fs::create_dir_all(&root).unwrap();
    let provider = root.join("libonnxruntime_providers_cuda.so");
    fs::write(&provider, b"not a shared library").unwrap();
    let result = child(&BackendConfig {
        runtime: Runtime::Cuda,
        device: "gpu".into(),
        onnxruntime_library: Some(core),
        provider_library: Some(provider),
        library_dirs: vec![root],
        ..Default::default()
    });

    assert!(!result.ready);
    assert!(!result.evidence.provider_registration);
    assert!(
        result
            .errors
            .iter()
            .any(|error| error.contains("register selected CUDA provider library")),
        "{:?}",
        result.errors
    );
}

fn test_ort_library() -> PathBuf {
    env::var_os("OMASPEAK_TEST_ONNXRUNTIME_LIBRARY")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/debug/libonnxruntime.so")
        })
}

#[test]
fn child_evidence_and_device_validation_are_testable_without_native_runtimes() {
    let root = env::temp_dir().join(format!("omaspeak-child-evidence-{}", std::process::id()));
    let ort = root.join("libonnxruntime.so");
    let provider = root.join("libonnxruntime_providers_cuda.so");
    let openvino = root.join("libopenvino_c.so");
    let plugins = root.join("plugins.xml");

    let mut config = BackendConfig {
        runtime: Runtime::Openvino,
        device: "gpu".into(),
        openvino_library: Some(openvino.clone()),
        openvino_plugins: Some(plugins.clone()),
        ..Default::default()
    };
    let result = child_with(
        &config,
        |paths| {
            assert_eq!(paths.library, openvino);
            assert_eq!(paths.plugins, plugins);
            Ok((
                "OpenVINO fixture".into(),
                vec!["CPU".into(), "GPU.0".into()],
            ))
        },
        |_, _| unreachable!(),
    );
    assert!(result.ready && result.loadable && result.device_accessible);
    assert!(result.evidence.provider_registration);
    assert_eq!(result.evidence.versions, ["OpenVINO fixture"]);
    assert_eq!(result.evidence.selected_device.as_deref(), Some("GPU"));

    config.device = "auto".into();
    let result = child_with(
        &config,
        |_| Ok(("OpenVINO fixture".into(), Vec::new())),
        |_, _| unreachable!(),
    );
    assert!(result.errors[0].contains("no accessible devices"));
    config.device = "npu".into();
    let result = child_with(
        &config,
        |_| Ok(("OpenVINO fixture".into(), vec!["CPU".into()])),
        |_, _| unreachable!(),
    );
    assert!(result.errors[0].contains("requested device NPU is unavailable"));

    config.runtime = Runtime::Default;
    config.device = "cpu".into();
    config.onnxruntime_library = Some(ort.clone());
    let result = child_with(
        &config,
        |_| unreachable!(),
        |actual, provider| {
            assert_eq!(actual, ort);
            assert!(provider.is_none());
            Ok(Vec::new())
        },
    );
    assert!(result.ready && !result.evidence.provider_registration);
    assert_eq!(result.evidence.selected_device.as_deref(), Some("cpu"));

    config.runtime = Runtime::Cuda;
    config.device = "gpu".into();
    config.provider_library = Some(provider.clone());
    config.device_id = 1;
    let result = child_with(
        &config,
        |_| unreachable!(),
        |actual, actual_provider| {
            assert_eq!(actual, ort);
            assert_eq!(actual_provider, Some(provider.as_path()));
            Ok(vec![
                (0, 11794, "CUDA ordinal 0, hardware 11794 (GPU)".into()),
                (1, 42, "CUDA ordinal 1, hardware 42 (GPU)".into()),
            ])
        },
    );
    assert!(result.ready && result.evidence.provider_registration);
    assert_eq!(
        result.evidence.selected_device.as_deref(),
        Some("CUDA ordinal 1, hardware 42 (GPU)")
    );

    config.provider_library = None;
    let result = child_with(&config, |_| unreachable!(), |_, _| unreachable!());
    assert!(result.errors[0].contains("missing CUDA provider"));
}
