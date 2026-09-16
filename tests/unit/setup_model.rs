use std::fs;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use sha2::{Digest, Sha256};

use super::*;

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn temp(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "omaspeak-model-test-{}-{name}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn leak(value: impl Into<String>) -> &'static str {
    Box::leak(value.into().into_boxed_str())
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn paths(root: &Path) -> AppPaths {
    AppPaths {
        config_file: root.join("config/omaspeak.toml"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    }
}

fn spec(id: &str, definitions: &[(&str, &[u8], &str)]) -> &'static ModelSpec {
    let files = definitions
        .iter()
        .map(|(path, bytes, url)| ModelFile {
            path: leak(*path),
            url: leak(*url),
            size: bytes.len() as u64,
            sha256: leak(digest(bytes)),
        })
        .collect::<Vec<_>>();
    let mut model = *crate::catalog::model("supertonic-3-gguf").unwrap();
    model.id = leak(id);
    model.name = model.id;
    model.model_file = files.first().map_or("", |file| file.path);
    model.files = Box::leak(files.into_boxed_slice());
    model.license = "MIT";
    model.license_status = "test fixture";
    model.requires_acceptance = false;
    model.license_file = "";
    model.license_sha256 = "";
    Box::leak(Box::new(model))
}

fn write_source(root: &Path, spec: &ModelSpec, definitions: &[(&str, &[u8], &str)]) {
    for (file, (_, bytes, _)) in spec.files.iter().zip(definitions) {
        let path = root.join(file.path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }
}

#[test]
fn model_path_is_backend_neutral() {
    let root = temp("path");
    let app = paths(&root);
    assert_eq!(
        model_directory(&app, crate::catalog::model("supertonic-3-gguf").unwrap()),
        root.join("data/models/supertonic-3-gguf")
    );
}

#[test]
fn local_directory_install_is_verified_idempotent_and_repairable() {
    let definitions = [
        (
            "onnx/model.bin",
            &b"model"[..],
            "https://unused.invalid/model",
        ),
        (
            "voices/F1.json",
            &b"voice"[..],
            "https://unused.invalid/voice",
        ),
    ];
    let model = spec("local", &definitions);
    let root = temp("local");
    let source = root.join("source");
    write_source(&source, model, &definitions);
    let app = paths(&root);
    let installed = install(&app, model, Some(&source), ProgressFormat::Json, None).unwrap();
    verify(&app, model).unwrap();
    verify_at(&installed, model).unwrap();
    assert_eq!(
        fs::read(installed.join("onnx/model.bin")).unwrap(),
        b"model"
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(installed.join(".omaspeak-model.json")).unwrap()).unwrap();
    assert_eq!(manifest["provenance"]["source"], "user-supplied");
    assert_eq!(manifest["provenance"]["modified"], false);

    fs::write(source.join("onnx/model.bin"), b"wrong").unwrap();
    install(&app, model, Some(&source), ProgressFormat::Human, None).unwrap();
    assert_eq!(
        fs::read(installed.join("onnx/model.bin")).unwrap(),
        b"model"
    );

    fs::write(installed.join("onnx/model.bin"), b"bad!!").unwrap();
    assert!(verify(&app, model).is_err());
    fs::write(source.join("onnx/model.bin"), b"model").unwrap();
    install(&app, model, Some(&source), ProgressFormat::Human, None).unwrap();
    verify(&app, model).unwrap();
}

#[test]
fn missing_or_tampered_manifest_is_never_treated_as_installed() {
    let bytes = b"model";
    let model = spec("manifest", &[("model", bytes, "https://unused.invalid")]);
    let root = temp("manifest");
    let source = root.join("source");
    write_source(
        &source,
        model,
        &[("model", bytes, "https://unused.invalid")],
    );
    let app = paths(&root);
    let target = install(&app, model, Some(&source), ProgressFormat::Human, None).unwrap();
    let manifest_path = target.join(".omaspeak-model.json");
    let original = fs::read(&manifest_path).unwrap();

    fs::remove_file(&manifest_path).unwrap();
    assert!(
        verify(&app, model)
            .unwrap_err()
            .to_string()
            .contains("manifest")
    );
    fs::write(&manifest_path, &original).unwrap();
    let mut manifest: serde_json::Value = serde_json::from_slice(&original).unwrap();
    manifest["provenance"]["artifact_revision"] = serde_json::json!("tampered");
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let error = verify(&app, model).unwrap_err().to_string();
    assert!(error.contains("catalog") || error.contains("provenance"));
}

#[test]
fn manifest_verification_rejects_structural_and_source_tampering() {
    let bytes = b"model";
    let model = spec(
        "manifest-structure",
        &[("model", bytes, "https://unused.invalid")],
    );
    let root = temp("manifest-structure");
    let source = root.join("source");
    write_source(
        &source,
        model,
        &[("model", bytes, "https://unused.invalid")],
    );
    let app = paths(&root);
    let target = install(&app, model, Some(&source), ProgressFormat::Human, None).unwrap();
    let manifest_path = target.join(".omaspeak-model.json");
    let original: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();

    let reject = |mut manifest: serde_json::Value| {
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        assert!(verify(&app, model).is_err());
        manifest = original.clone();
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
    };

    let mut value = original.clone();
    value["unexpected"] = true.into();
    reject(value);
    let mut value = original.clone();
    value["schema"] = 2.into();
    reject(value);
    let mut value = original.clone();
    value["catalog"]["name"] = "tampered".into();
    reject(value);
    let mut value = original.clone();
    value["provenance"]["unexpected"] = true.into();
    reject(value);
    let mut value = original.clone();
    value["provenance"]
        .as_object_mut()
        .unwrap()
        .remove("source");
    value["provenance"]["replacement"] = true.into();
    reject(value);
    let mut value = original.clone();
    value["provenance"]
        .as_object_mut()
        .unwrap()
        .remove("source_path");
    value["provenance"]["replacement"] = true.into();
    reject(value);
    let mut value = original.clone();
    value["provenance"]["source"] = "catalog-download".into();
    reject(value);
    let mut value = original.clone();
    value["license_acceptance"] = serde_json::json!({"unexpected": true});
    reject(value);

    fs::write(&manifest_path, b"not json").unwrap();
    assert!(verify(&app, model).is_err());
}

#[test]
fn interrupted_staging_and_old_directories_are_cleaned_before_publication() {
    let bytes = b"fresh";
    let model = spec("interrupted", &[("model", bytes, "https://unused.invalid")]);
    let root = temp("interrupted");
    let source = root.join("source");
    write_source(
        &source,
        model,
        &[("model", bytes, "https://unused.invalid")],
    );
    let app = paths(&root);
    let models = app.data_dir.join("models");
    let staging = models.join(format!(".interrupted.install-{}", std::process::id()));
    let old = models.join(format!(".interrupted.old-{}", std::process::id()));
    fs::create_dir_all(&staging).unwrap();
    fs::write(staging.join("partial"), b"partial").unwrap();
    fs::create_dir_all(&old).unwrap();
    fs::write(old.join("stale"), b"stale").unwrap();
    let target = install(&app, model, Some(&source), ProgressFormat::Json, None).unwrap();
    verify_at(&target, model).unwrap();
    assert!(!staging.exists());
    assert!(!old.exists());
}

#[test]
fn one_file_model_accepts_file_source_and_multifile_requires_directory() {
    let bytes = b"one";
    let one = spec(
        "one",
        &[("model.gguf", bytes, "https://unused.invalid/one")],
    );
    let root = temp("file-source");
    let file = root.join("model.gguf");
    fs::write(&file, bytes).unwrap();
    install(&paths(&root), one, Some(&file), ProgressFormat::Human, None).unwrap();

    let multi = spec(
        "multi",
        &[
            ("a", &b"a"[..], "https://unused.invalid/a"),
            ("b", &b"b"[..], "https://unused.invalid/b"),
        ],
    );
    let error = install(
        &paths(&temp("multi")),
        multi,
        Some(&file),
        ProgressFormat::Human,
        None,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("must point to a directory"));
}

#[test]
fn local_sources_must_exist_and_match_every_pin() {
    let bytes = b"right";
    let model = spec("pins", &[("nested/model", bytes, "https://unused.invalid")]);
    let root = temp("pins");
    let app = paths(&root);
    assert!(
        install(
            &app,
            model,
            Some(&root.join("missing")),
            ProgressFormat::Human,
            None
        )
        .is_err()
    );
    let source = root.join("source");
    fs::create_dir_all(source.join("nested")).unwrap();
    fs::write(source.join("nested/model"), b"wrong").unwrap();
    assert!(install(&app, model, Some(&source), ProgressFormat::Human, None).is_err());
    assert!(!model_directory(&app, model).exists());
}

#[test]
fn license_acceptance_and_provenance_are_recorded() {
    let bytes = b"licensed";
    let mut licensed = *spec("licensed", &[("model", bytes, "https://unused.invalid")]);
    licensed.license = "OpenRAIL-M";
    licensed.license_status = "acceptance required";
    licensed.requires_acceptance = true;
    licensed.license_file = "MODEL-LICENSE";
    licensed.license_sha256 = "0d944a9110fed9a9602d60e0423a272903e7bd21ab060490774efc77c2275e9f";
    let licensed = Box::leak(Box::new(licensed));
    let root = temp("license");
    let source = root.join("source");
    write_source(
        &source,
        licensed,
        &[("model", bytes, "https://unused.invalid")],
    );
    let app = paths(&root);
    let error = install(&app, licensed, Some(&source), ProgressFormat::Human, None).unwrap_err();
    assert!(error.to_string().contains("requires acceptance"));
    let installed = install(
        &app,
        licensed,
        Some(&source),
        ProgressFormat::Json,
        Some("OpenRAIL-M"),
    )
    .unwrap();
    assert_eq!(
        fs::read(installed.join("MODEL-LICENSE")).unwrap().len(),
        15_007
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(installed.join(".omaspeak-model.json")).unwrap()).unwrap();
    assert_eq!(manifest["license_acceptance"]["license"], "OpenRAIL-M");
    assert!(
        manifest["license_acceptance"]["accepted_at_unix_seconds"]
            .as_u64()
            .is_some()
    );
    let license_path = installed.join("MODEL-LICENSE");
    let original_license = fs::read(&license_path).unwrap();
    let mut tampered_license = original_license.clone();
    tampered_license[0] ^= 1;
    fs::write(&license_path, tampered_license).unwrap();
    assert!(verify(&app, licensed).is_err());
    fs::write(&license_path, original_license).unwrap();
    verify(&app, licensed).unwrap();
    let mut tampered = manifest;
    tampered["license_acceptance"]["license"] = "different".into();
    fs::write(
        installed.join(".omaspeak-model.json"),
        serde_json::to_vec_pretty(&tampered).unwrap(),
    )
    .unwrap();
    assert!(verify(&app, licensed).is_err());
}

#[test]
fn non_downloadable_models_require_a_local_source() {
    let mut restricted = *spec("restricted", &[("model", &b"x"[..], "")]);
    restricted.downloadable = false;
    let restricted = Box::leak(Box::new(restricted));
    let root = temp("restricted");
    let error = install(&paths(&root), restricted, None, ProgressFormat::Human, None).unwrap_err();
    assert!(error.to_string().contains("--source"));
}

fn local_downloads(bodies: Vec<Vec<u8>>) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        for body in bodies {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 2048];
            let _ = stream.read(&mut request);
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .unwrap();
            stream.write_all(&body).unwrap();
        }
    });
    (format!("http://{address}/asset"), server)
}

#[test]
fn catalog_downloads_every_file_and_reuses_verified_cache() {
    let (url, server) = local_downloads(vec![b"alpha".to_vec(), b"beta".to_vec()]);
    let model = spec(
        "download",
        &[
            ("a/model", &b"alpha"[..], &url),
            ("b/style", &b"beta"[..], &url),
        ],
    );
    let root = temp("download");
    let app = paths(&root);
    install(&app, model, None, ProgressFormat::Json, None).unwrap();
    server.join().unwrap();
    verify(&app, model).unwrap();
    assert_eq!(
        fs::read(root.join("data/downloads/download/a/model")).unwrap(),
        b"alpha"
    );
    install(&app, model, None, ProgressFormat::Human, None).unwrap();
    fs::remove_dir_all(model_directory(&app, model)).unwrap();
    install(&app, model, None, ProgressFormat::Json, None).unwrap();
    verify(&app, model).unwrap();
}

#[test]
fn downloads_reject_wrong_lengths_checksums_and_remove_parts() {
    let root = temp("bad-download");
    let app = paths(&root);
    let bytes = b"good";
    let missing_url = spec("missing-url", &[("model", bytes, "")]);
    assert!(
        download_file(
            &app,
            missing_url,
            &missing_url.files[0],
            ProgressFormat::Human
        )
        .is_err()
    );
    let model = spec(
        "bad-download",
        &[("model", bytes, "http://127.0.0.1:9/missing")],
    );
    assert!(download_file(&app, model, &model.files[0], ProgressFormat::Json).is_err());

    let (url, server) = local_downloads(vec![b"too long".to_vec()]);
    let bad_wire = spec("bad-wire", &[("model", bytes, &url)]);
    assert!(download_file(&app, bad_wire, &bad_wire.files[0], ProgressFormat::Human).is_err());
    server.join().unwrap();
    let wire_cache = app.data_dir.join("downloads/bad-wire");
    assert!(fs::read_dir(wire_cache).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".part-")
    }));

    let target = root.join("target");
    let part = root.join("part");
    let mut wrong_size = model.files[0];
    wrong_size.size += 1;
    assert!(
        write_pinned_download(
            &bytes[..],
            &part,
            &target,
            model,
            &wrong_size,
            ProgressFormat::Human
        )
        .is_err()
    );
    assert!(!target.exists());
    let mut wrong_hash = model.files[0];
    wrong_hash.sha256 = leak("0".repeat(64));
    assert!(
        write_pinned_download(
            &bytes[..],
            &part,
            &target,
            model,
            &wrong_hash,
            ProgressFormat::Json
        )
        .is_err()
    );
    let longer = b"longer";
    assert!(
        write_pinned_download(
            &longer[..],
            &part,
            &target,
            model,
            &model.files[0],
            ProgressFormat::Human
        )
        .is_err()
    );

    let copy_target = root.join("copy/model");
    fs::create_dir_all(copy_target.parent().unwrap()).unwrap();
    fs::create_dir_all(
        copy_target
            .parent()
            .unwrap()
            .join(format!(".model.part-{}", std::process::id())),
    )
    .unwrap();
    let source = root.join("copy-source");
    fs::write(&source, bytes).unwrap();
    assert!(copy_verified(&source, &copy_target, &model.files[0]).is_err());
}

#[test]
fn corrupted_cache_is_replaced_before_install() {
    let bytes = b"fresh";
    let (url, server) = local_downloads(vec![bytes.to_vec()]);
    let model = spec("cache-repair", &[("model", bytes, &url)]);
    let root = temp("cache-repair");
    let cache = root.join("data/downloads/cache-repair/model");
    fs::create_dir_all(cache.parent().unwrap()).unwrap();
    fs::write(&cache, b"stale").unwrap();
    install(&paths(&root), model, None, ProgressFormat::Human, None).unwrap();
    server.join().unwrap();
    assert_eq!(fs::read(cache).unwrap(), bytes);
}

#[test]
fn staging_manifest_failure_preserves_previous_model() {
    let bytes = b"new";
    let mut model = *spec(
        "rollback",
        &[(".omaspeak-model.json/file", bytes, "https://unused.invalid")],
    );
    model.model_file = ".omaspeak-model.json/file";
    let model = Box::leak(Box::new(model));
    let root = temp("rollback");
    let app = paths(&root);
    let target = model_directory(&app, model);
    fs::create_dir_all(&target).unwrap();
    fs::write(target.join("previous"), b"keep").unwrap();
    let source = root.join("source");
    write_source(
        &source,
        model,
        &[(".omaspeak-model.json/file", bytes, "https://unused.invalid")],
    );
    assert!(install(&app, model, Some(&source), ProgressFormat::Human, None).is_err());
    assert_eq!(fs::read(target.join("previous")).unwrap(), b"keep");
}

#[test]
fn unsafe_catalog_paths_and_non_regular_assets_are_rejected() {
    for path in ["", "../escape", "/absolute", "a/../b"] {
        assert!(validate_relative_file(path).is_err());
    }
    assert!(validate_relative_file("nested/file").is_ok());
    let root = temp("directory-asset");
    let directory = root.join("asset");
    fs::create_dir_all(&directory).unwrap();
    assert!(verify_pinned_file(&directory, 0, &digest(b"")).is_err());
    assert!(verify_pinned_file(&root.join("missing"), 0, &digest(b"")).is_err());
}

#[test]
fn progress_threshold_and_emit_formats_cover_completion() {
    assert!(!should_report_progress(100, 0, 200));
    assert!(should_report_progress(200, 100, 200));
    assert!(should_report_progress(1024 * 1024, 0, 2 * 1024 * 1024));
    let model = spec(
        "progress",
        &[("model", &b"x"[..], "https://unused.invalid")],
    );
    emit(
        ProgressFormat::Human,
        "downloaded",
        model,
        Some("model"),
        Some(1),
        Some(1),
    )
    .unwrap();
    emit(ProgressFormat::Json, "installed", model, None, None, None).unwrap();
}

#[test]
fn a_second_installer_cannot_replace_an_active_profile() {
    let root = temp("locked-profile");
    let app = paths(&root);
    let definitions = [("model.bin", &b"new-model"[..], "https://unused.invalid")];
    let spec = spec("locked", &definitions);
    let source = root.join("source");
    write_source(&source, spec, &definitions);
    let target = model_directory(&app, spec);
    fs::create_dir_all(&target).unwrap();
    fs::write(target.join("keep"), b"active").unwrap();
    let guard = InstallGuard::acquire(&app.data_dir, spec.id).unwrap();
    assert!(
        install(&app, spec, Some(&source), ProgressFormat::Human, None)
            .unwrap_err()
            .to_string()
            .contains("busy")
    );
    assert_eq!(fs::read(target.join("keep")).unwrap(), b"active");
    drop(guard);
    install(&app, spec, Some(&source), ProgressFormat::Human, None).unwrap();
    verify(&app, spec).unwrap();
}

#[test]
fn url_checks_report_ok_mismatch_and_unreachable_without_writes() {
    let probes: &mut dyn FnMut(&str) -> std::result::Result<u64, String> = &mut |url: &str| {
        if url.contains("supertonic-3-orig.gguf") {
            Ok(454_072_836)
        } else if url.contains("/onnx/") {
            Err("request failed: connection refused".into())
        } else {
            Ok(1)
        }
    };
    let checks = check_urls_with(None, probes);
    let mut seen = 0;
    for spec in crate::catalog::models() {
        seen += spec.files.len();
    }
    assert_eq!(checks.len(), seen);
    let ok = checks.iter().find(|check| check.status == "ok").unwrap();
    assert!(ok.detail.is_none());
    assert!(ok.url.contains("https://"));
    let mismatch = checks
        .iter()
        .find(|check| check.status == "size-mismatch")
        .unwrap();
    assert!(mismatch.detail.as_deref().unwrap().contains("pinned size"));
    let unreachable = checks
        .iter()
        .find(|check| check.status == "unreachable")
        .unwrap();
    assert!(
        unreachable
            .detail
            .as_deref()
            .unwrap()
            .contains("connection refused")
    );
}

#[test]
fn url_prefix_replaces_the_origin_and_keeps_the_path() {
    let checks = check_urls_with(Some("http://127.0.0.1:9"), &mut |_url| Ok(0));
    assert!(!checks.is_empty());
    assert!(
        checks
            .iter()
            .all(|check| check.url.starts_with("http://127.0.0.1:9/"))
    );
    assert!(
        checks
            .iter()
            .all(|check| !check.url.contains("huggingface.co"))
    );
    let checks = check_urls_with(Some("http://127.0.0.1:9"), &mut |_url| Ok(0));
    assert!(checks.iter().all(|check| !check.url.contains("//9")));
}

#[test]
fn verify_pinned_file_diagnostics_name_expected_and_actual_values() {
    let dir = temp("verify-diagnostics");
    let path = dir.join("asset.bin");
    fs::write(&path, b"short").unwrap();
    let size_error = verify_pinned_file(&path, 10, &digest(b"short"))
        .err()
        .unwrap()
        .to_string();
    assert!(
        size_error.contains("expected 10 bytes, found 5"),
        "{size_error}"
    );
    fs::write(&path, [0_u8; 10]).unwrap();
    let digest_error = verify_pinned_file(&path, 10, &digest(b"other"))
        .err()
        .unwrap()
        .to_string();
    assert!(digest_error.contains("checksum mismatch"), "{digest_error}");
    assert!(digest_error.contains("expected") && digest_error.contains("found"));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn url_check_reports_render_json_and_human_rows() {
    let mut buffer: Vec<u8> = Vec::new();
    let checks = check_urls_with(None, &mut |url| {
        if url.ends_with("one") {
            Ok(2)
        } else {
            Err("connection refused".into())
        }
    });
    write_url_checks(&checks, &mut buffer, true).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&buffer).unwrap();
    assert!(parsed.is_array());
    buffer.clear();
    write_url_checks(&checks, &mut buffer, false).unwrap();
    let human = String::from_utf8(buffer).unwrap();
    assert!(human.contains("unreachable: "));
    assert!(human.contains("connection refused"));
}

#[test]
fn rewritten_url_leaves_non_http_urls_untouched() {
    let checks = check_urls_with(Some("http://127.0.0.1:9"), &mut |_url| Ok(0));
    assert!(
        checks
            .iter()
            .any(|check| check.url.starts_with("http://127.0.0.1:9/"))
    );
}
