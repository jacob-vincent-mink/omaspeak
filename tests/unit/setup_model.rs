use super::*;
use bzip2::Compression;
use bzip2::write::BzEncoder;

use crate::catalog::RequiredFile;

fn temp(name: &str) -> PathBuf {
    let path =
        std::env::temp_dir().join(format!("omaspeak-model-test-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn leak(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn archive(root: &str, path: &str, bytes: &[u8]) -> Vec<u8> {
    let encoder = BzEncoder::new(Vec::new(), Compression::best());
    let mut builder = tar::Builder::new(encoder);
    let mut header = tar::Header::new_gnu();
    header.set_size(bytes.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    builder
        .append_data(&mut header, format!("{root}/{path}"), bytes)
        .unwrap();
    builder.into_inner().unwrap().finish().unwrap()
}

fn archive_with_marker_directory(root: &str, path: &str, bytes: &[u8]) -> Vec<u8> {
    let encoder = BzEncoder::new(Vec::new(), Compression::best());
    let mut builder = tar::Builder::new(encoder);

    let mut file = tar::Header::new_gnu();
    file.set_size(bytes.len() as u64);
    file.set_mode(0o644);
    file.set_cksum();
    builder
        .append_data(&mut file, format!("{root}/{path}"), bytes)
        .unwrap();

    let mut directory = tar::Header::new_gnu();
    directory.set_entry_type(tar::EntryType::Directory);
    directory.set_size(0);
    directory.set_mode(0o755);
    directory.set_cksum();
    builder
        .append_data(
            &mut directory,
            format!("{root}/.omaspeak-model.json"),
            &[][..],
        )
        .unwrap();

    builder.into_inner().unwrap().finish().unwrap()
}

fn spec(archive_bytes: &[u8], url: &str) -> &'static ModelSpec {
    let asset = b"tiny model";
    let required: &'static [RequiredFile] = Box::leak(
        vec![RequiredFile {
            path: "model.bin",
            size: asset.len() as u64,
            sha256: leak(digest(asset)),
        }]
        .into_boxed_slice(),
    );
    Box::leak(Box::new(ModelSpec {
        id: "tiny",
        backend: "sherpa-onnx",
        family: "piper",
        name: "tiny",
        description: "test model",
        archive_url: leak(url.to_owned()),
        archive_size: archive_bytes.len() as u64,
        archive_sha256: leak(digest(archive_bytes)),
        archive_root: "tiny-root",
        model_file: "model.bin",
        tokens_file: "model.bin",
        data_directory: "model.bin",
        duration_predictor: "",
        text_encoder: "",
        vector_estimator: "",
        vocoder: "",
        tts_json: "",
        unicode_indexer: "",
        voice_style: "",
        language: "en",
        steps: 5,
        npu_capable: false,
        required_files: required,
    }))
}

fn paths(root: &Path) -> AppPaths {
    AppPaths {
        config_file: root.join("config/config.toml"),
        data_dir: root.join("data"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    }
}

#[test]
fn model_path_is_backend_neutral() {
    let paths = AppPaths {
        config_file: "/tmp/config".into(),
        data_dir: "/tmp/data".into(),
        state_dir: "/tmp/state".into(),
        runtime_dir: "/tmp/run".into(),
    };
    let spec = crate::catalog::models().first().unwrap();
    assert_eq!(
        model_directory(&paths, spec),
        PathBuf::from("/tmp/data/models/en_US-lessac-medium")
    );
}

#[test]
fn progress_is_rate_limited_and_always_reports_completion() {
    assert!(!should_report_progress(128 * 1024, 0, 2 * 1024 * 1024));
    assert!(should_report_progress(1024 * 1024, 0, 2 * 1024 * 1024));
    assert!(should_report_progress(
        2 * 1024 * 1024,
        1024 * 1024,
        2 * 1024 * 1024
    ));
}

#[test]
fn local_archive_install_is_verified_idempotent_and_repairable() {
    let root = temp("install");
    let archive_bytes = archive("tiny-root", "model.bin", b"tiny model");
    let archive_path = root.join("tiny.tar.bz2");
    fs::write(&archive_path, &archive_bytes).unwrap();
    let spec = spec(&archive_bytes, "http://unused.invalid/model");
    let paths = paths(&root);

    let installed = install(&paths, spec, Some(&archive_path), ProgressFormat::Json).unwrap();
    assert_eq!(installed, model_directory(&paths, spec));
    assert!(installed.join(".omaspeak-model.json").is_file());
    verify(&paths, spec).unwrap();
    install(&paths, spec, Some(&archive_path), ProgressFormat::Human).unwrap();

    fs::write(installed.join("model.bin"), b"bad").unwrap();
    assert!(verify(&paths, spec).is_err());
    install(&paths, spec, Some(&archive_path), ProgressFormat::Human).unwrap();
    verify(&paths, spec).unwrap();
}

#[test]
fn archive_and_asset_corruption_are_rejected() {
    let root = temp("corruption");
    let bytes = archive("tiny-root", "model.bin", b"tiny model");
    let path = root.join("archive.tar.bz2");
    fs::write(&path, &bytes).unwrap();
    let spec = spec(&bytes, "http://unused.invalid/model");
    verify_archive(&path, spec).unwrap();

    let mut changed = bytes.clone();
    let middle = changed.len() / 2;
    changed[middle] ^= 1;
    fs::write(&path, &changed).unwrap();
    assert!(verify_archive(&path, spec).is_err());
    fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
    assert!(verify_archive(&path, spec).is_err());
    assert!(verify_archive(&root.join("missing"), spec).is_err());

    let directory = root.join("assets");
    fs::create_dir_all(&directory).unwrap();
    assert!(verify_directory(&directory, spec).is_err());
    fs::write(directory.join("model.bin"), b"bad model!").unwrap();
    assert!(verify_directory(&directory, spec).is_err());
    fs::write(directory.join("model.bin"), b"tiny xodel").unwrap();
    assert!(verify_directory(&directory, spec).is_err());
}

#[test]
fn extraction_rejects_wrong_roots_and_non_files() {
    let root = temp("unsafe");
    let wrong = archive("other-root", "model.bin", b"tiny model");
    let wrong_path = root.join("wrong.tar.bz2");
    fs::write(&wrong_path, wrong).unwrap();
    let expected = archive("tiny-root", "model.bin", b"tiny model");
    let spec = spec(&expected, "http://unused.invalid/model");
    assert!(extract_archive(&wrong_path, &root.join("stage"), spec).is_err());

    let encoder = BzEncoder::new(Vec::new(), Compression::best());
    let mut builder = tar::Builder::new(encoder);
    let mut header = tar::Header::new_gnu();
    header.set_entry_type(tar::EntryType::Symlink);
    header.set_size(0);
    header.set_mode(0o777);
    header.set_link_name("target").unwrap();
    header.set_cksum();
    builder
        .append_data(&mut header, "tiny-root/link", &[][..])
        .unwrap();
    let links = builder.into_inner().unwrap().finish().unwrap();
    let links_path = root.join("links.tar.bz2");
    fs::write(&links_path, links).unwrap();
    assert!(extract_archive(&links_path, &root.join("stage2"), spec).is_err());
}

#[test]
fn downloaded_bytes_succeed_cache_and_reject_bad_lengths() {
    let root = temp("network");
    let archive_bytes = archive("tiny-root", "model.bin", b"tiny model");
    let model_spec = spec(&archive_bytes, "http://unused.invalid/model");
    let app_paths = paths(&root);
    fs::create_dir_all(app_paths.data_dir.join("downloads")).unwrap();
    let downloaded = app_paths.data_dir.join("downloads/tiny.tar.bz2");
    let part = downloaded.with_extension("tar.bz2.part");
    write_download(
        &archive_bytes[..],
        &part,
        &downloaded,
        model_spec,
        ProgressFormat::Json,
    )
    .unwrap();
    verify_archive(&downloaded, model_spec).unwrap();
    assert_eq!(
        download_archive(&app_paths, model_spec, ProgressFormat::Human).unwrap(),
        downloaded
    );

    let short_root = temp("short");
    let short_spec = spec(&archive_bytes, "http://unused.invalid/model");
    let short_paths = paths(&short_root);
    fs::create_dir_all(short_paths.data_dir.join("downloads")).unwrap();
    let target = short_paths.data_dir.join("downloads/tiny.tar.bz2");
    assert!(
        write_download(
            &archive_bytes[..archive_bytes.len() - 1],
            &target.with_extension("tar.bz2.part"),
            &target,
            short_spec,
            ProgressFormat::Human,
        )
        .is_err()
    );

    let overflow_root = temp("overflow");
    let expected = &archive_bytes[..archive_bytes.len() - 1];
    let overflow_spec = spec(expected, "http://unused.invalid/model");
    let overflow_paths = paths(&overflow_root);
    fs::create_dir_all(overflow_paths.data_dir.join("downloads")).unwrap();
    let target = overflow_paths.data_dir.join("downloads/tiny.tar.bz2");
    assert!(
        write_download(
            &archive_bytes[..],
            &target.with_extension("tar.bz2.part"),
            &target,
            overflow_spec,
            ProgressFormat::Human,
        )
        .is_err()
    );

    let wrong_root = temp("wrong-checksum");
    let wrong_spec = spec(&archive_bytes, "http://unused.invalid/model");
    let target = wrong_root.join("tiny.tar.bz2");
    let mut wrong = archive_bytes.clone();
    wrong[0] ^= 1;
    assert!(
        write_download(
            &wrong[..],
            &target.with_extension("tar.bz2.part"),
            &target,
            wrong_spec,
            ProgressFormat::Human,
        )
        .is_err()
    );
}

#[test]
fn install_uses_cached_download_and_cleans_stale_work_directories() {
    let root = temp("cached-install");
    let archive_bytes = archive("tiny-root", "model.bin", b"tiny model");
    let spec = spec(&archive_bytes, "http://unused.invalid/model");
    let paths = paths(&root);
    let downloads = paths.data_dir.join("downloads");
    let models = paths.data_dir.join("models");
    fs::create_dir_all(&downloads).unwrap();
    fs::create_dir_all(&models).unwrap();
    fs::write(downloads.join("tiny.tar.bz2"), &archive_bytes).unwrap();
    fs::create_dir_all(models.join(format!(".tiny.install-{}", std::process::id()))).unwrap();
    fs::create_dir_all(models.join(format!(".tiny.old-{}", std::process::id()))).unwrap();

    let installed = install(&paths, spec, None, ProgressFormat::Human).unwrap();
    assert_eq!(
        fs::read(installed.join("model.bin")).unwrap(),
        b"tiny model"
    );
}

#[test]
fn failed_model_finalization_restores_the_previous_directory() {
    let root = temp("finalize-rollback");
    let archive_bytes = archive_with_marker_directory("tiny-root", "model.bin", b"tiny model");
    let archive_path = root.join("tiny.tar.bz2");
    fs::write(&archive_path, &archive_bytes).unwrap();
    let spec = spec(&archive_bytes, "http://unused.invalid/model");
    let paths = paths(&root);
    let previous = model_directory(&paths, spec);
    fs::create_dir_all(&previous).unwrap();
    fs::write(previous.join("previous"), b"keep me").unwrap();

    let error = install(&paths, spec, Some(&archive_path), ProgressFormat::Human).unwrap_err();
    assert!(error.to_string().contains("finalize installed model"));
    assert_eq!(fs::read(previous.join("previous")).unwrap(), b"keep me");
    assert!(!previous.join("model.bin").exists());

    emit(
        ProgressFormat::Human,
        "downloaded",
        spec,
        Some(spec.archive_size),
        Some(spec.archive_size),
    )
    .unwrap();
}
