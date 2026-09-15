use super::*;
use bzip2::Compression;
use bzip2::write::BzEncoder;
use std::net::TcpListener;
use std::thread;

use crate::catalog::{RequiredFile, SingleFile, SupplementalFile};

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

fn archive_with_superseded(root: &str) -> Vec<u8> {
    let encoder = BzEncoder::new(Vec::new(), Compression::best());
    let mut builder = tar::Builder::new(encoder);
    for (path, bytes) in [
        ("model.bin", &b"tiny model"[..]),
        ("model.int8", &b"superseded"[..]),
    ] {
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, format!("{root}/{path}"), bytes)
            .unwrap();
    }
    builder.into_inner().unwrap().finish().unwrap()
}

fn archive_with_superseded_directory(root: &str) -> Vec<u8> {
    let encoder = BzEncoder::new(Vec::new(), Compression::best());
    let mut builder = tar::Builder::new(encoder);
    let mut model = tar::Header::new_gnu();
    model.set_size(b"tiny model".len() as u64);
    model.set_mode(0o644);
    model.set_cksum();
    builder
        .append_data(&mut model, format!("{root}/model.bin"), &b"tiny model"[..])
        .unwrap();

    let mut directory = tar::Header::new_gnu();
    directory.set_entry_type(tar::EntryType::Directory);
    directory.set_size(0);
    directory.set_mode(0o755);
    directory.set_cksum();
    builder
        .append_data(&mut directory, format!("{root}/model.int8"), &[][..])
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
        backend: "supertonic",
        family: "supertonic",
        name: "tiny",
        description: "test model",
        license: "MIT",
        license_url: "https://example.invalid/license",
        license_status: "verified",
        downloadable: true,
        requires_acceptance: false,
        source_revision: "test-revision",
        single_file: None,
        license_file: "",
        license_sha256: "",
        archive_url: leak(url.to_owned()),
        archive_size: archive_bytes.len() as u64,
        archive_sha256: leak(digest(archive_bytes)),
        archive_root: "tiny-root",
        duration_predictor: "",
        text_encoder: "",
        vector_estimator: "",
        vocoder: "",
        tts_json: "",
        unicode_indexer: "",
        voice_style: "",
        language: "en",
        steps: 5,
        voices: &[],
        openvino_capable: false,
        npu_capable: false,
        required_files: required,
        supplemental_files: &[],
    }))
}

fn spec_with_supplement(archive_bytes: &[u8]) -> &'static ModelSpec {
    let required: &'static [RequiredFile] = Box::leak(
        vec![
            RequiredFile {
                path: "model.bin",
                size: 10,
                sha256: leak(digest(b"tiny model")),
            },
            RequiredFile {
                path: "extra/model.fp32",
                size: 11,
                sha256: leak(digest(b"float model")),
            },
        ]
        .into_boxed_slice(),
    );
    let supplemental_files: &'static [SupplementalFile] = Box::leak(
        vec![SupplementalFile {
            path: "extra/model.fp32",
            supersedes: "model.int8",
            url: "http://unused.invalid/model.fp32",
            size: 11,
            sha256: leak(digest(b"float model")),
        }]
        .into_boxed_slice(),
    );
    Box::leak(Box::new(ModelSpec {
        supplemental_files,
        required_files: required,
        ..*spec(archive_bytes, "http://unused.invalid/model")
    }))
}

fn single_file_spec() -> &'static ModelSpec {
    let bytes = b"tiny model";
    let mut model = *spec(&[], "");
    model.single_file = Some(SingleFile {
        path: "model.bin",
        url: "https://example.invalid/model.bin",
        size: bytes.len() as u64,
        sha256: leak(digest(bytes)),
    });
    model.archive_url = "";
    model.archive_size = 0;
    model.archive_sha256 = "";
    model.archive_root = "";
    Box::leak(Box::new(model))
}

fn paths(root: &Path) -> AppPaths {
    AppPaths {
        config_file: root.join("config/config.toml"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    }
}

#[test]
fn model_path_is_backend_neutral() {
    let paths = AppPaths {
        config_file: "/tmp/config".into(),
        data_dir: "/tmp/data".into(),
        cache_dir: "/tmp/cache".into(),
        state_dir: "/tmp/state".into(),
        runtime_dir: "/tmp/run".into(),
    };
    let spec = crate::catalog::models().first().unwrap();
    assert_eq!(
        model_directory(&paths, spec),
        PathBuf::from("/tmp/data/models/supertonic-3-gguf")
    );
}

#[test]
fn cached_single_file_install_is_verified_and_published_atomically() {
    let root = temp("single-file");
    let app_paths = paths(&root);
    let model = single_file_spec();
    let cache = app_paths.data_dir.join("downloads/tiny-model.bin");
    fs::create_dir_all(cache.parent().unwrap()).unwrap();
    fs::write(&cache, b"tiny model").unwrap();

    let installed = install(&app_paths, model, None, ProgressFormat::Human, None).unwrap();
    assert_eq!(
        fs::read(installed.join("model.bin")).unwrap(),
        b"tiny model"
    );
    verify(&app_paths, model).unwrap();
    assert!(installed.join(".omaspeak-model.json").is_file());
    assert!(
        fs::read_dir(app_paths.data_dir.join("models"))
            .unwrap()
            .all(|entry| !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".tiny.install-"))
    );

    fs::write(installed.join("model.bin"), b"bad").unwrap();
    assert!(verify(&app_paths, model).is_err());
}

#[test]
fn restricted_models_are_user_supplied_only() {
    let root = temp("restricted-license");
    let archive_bytes = archive("tiny-root", "model.bin", b"tiny model");
    let archive_path = root.join("tiny.tar.bz2");
    fs::write(&archive_path, &archive_bytes).unwrap();
    let mut restricted = *spec(&archive_bytes, "https://example.invalid/model");
    restricted.downloadable = false;
    restricted.license = "research-only terms";
    restricted.license_status = "restricted";
    let restricted = Box::leak(Box::new(restricted));
    let app_paths = paths(&root);

    let error = install(&app_paths, restricted, None, ProgressFormat::Human, None).unwrap_err();
    assert!(error.to_string().contains("user-supplied only"));
    assert!(!app_paths.data_dir.exists());

    install(
        &app_paths,
        restricted,
        Some(&archive_path),
        ProgressFormat::Human,
        None,
    )
    .unwrap();
}

#[test]
fn accepted_model_license_and_provenance_are_preserved_beside_weights() {
    let root = temp("accepted-license");
    let archive_bytes = archive("tiny-root", "model.bin", b"tiny model");
    let archive_path = root.join("tiny.tar.bz2");
    fs::write(&archive_path, &archive_bytes).unwrap();
    let mut licensed = *spec(&archive_bytes, "https://example.invalid/model");
    licensed.license = "OpenRAIL-M";
    licensed.license_url = "https://example.invalid/OpenRAIL-M";
    licensed.requires_acceptance = true;
    licensed.source_revision = "pinned-revision";
    licensed.license_file = "MODEL-LICENSE";
    licensed.license_sha256 = "0d944a9110fed9a9602d60e0423a272903e7bd21ab060490774efc77c2275e9f";
    let licensed = Box::leak(Box::new(licensed));
    let app_paths = paths(&root);

    let error = install(
        &app_paths,
        licensed,
        Some(&archive_path),
        ProgressFormat::Human,
        None,
    )
    .unwrap_err();
    assert!(error.to_string().contains("--accept-license OpenRAIL-M"));
    assert!(!app_paths.data_dir.exists());

    let directory = install(
        &app_paths,
        licensed,
        Some(&archive_path),
        ProgressFormat::Human,
        Some("OpenRAIL-M"),
    )
    .unwrap();
    assert_eq!(
        sha256_file(&directory.join("MODEL-LICENSE")).unwrap(),
        licensed.license_sha256
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(directory.join(".omaspeak-model.json")).unwrap()).unwrap();
    assert_eq!(manifest["license_acceptance"]["license"], "OpenRAIL-M");
    assert_eq!(manifest["provenance"]["source"], "user-supplied-archive");
    assert_eq!(manifest["provenance"]["source_revision"], "pinned-revision");
    assert!(
        manifest["license_acceptance"]["accepted_at_unix_seconds"]
            .as_u64()
            .is_some()
    );
    verify(&app_paths, licensed).unwrap();
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

    let installed = install(
        &paths,
        spec,
        Some(&archive_path),
        ProgressFormat::Json,
        None,
    )
    .unwrap();
    assert_eq!(installed, model_directory(&paths, spec));
    assert!(installed.join(".omaspeak-model.json").is_file());
    verify(&paths, spec).unwrap();
    install(
        &paths,
        spec,
        Some(&archive_path),
        ProgressFormat::Human,
        None,
    )
    .unwrap();

    fs::write(installed.join("model.bin"), b"bad").unwrap();
    assert!(verify(&paths, spec).is_err());
    install(
        &paths,
        spec,
        Some(&archive_path),
        ProgressFormat::Human,
        None,
    )
    .unwrap();
    verify(&paths, spec).unwrap();
}

#[test]
fn supplemental_asset_is_cached_verified_and_installed_with_the_archive() {
    let root = temp("supplement");
    let archive_bytes = archive_with_superseded("tiny-root");
    let archive_path = root.join("tiny.tar.bz2");
    fs::write(&archive_path, &archive_bytes).unwrap();
    let spec = spec_with_supplement(&archive_bytes);
    let paths = paths(&root);
    let downloads = paths.data_dir.join("downloads");
    fs::create_dir_all(&downloads).unwrap();
    fs::write(downloads.join("tiny.extra-model.fp32"), b"float model").unwrap();

    let installed = install(
        &paths,
        spec,
        Some(&archive_path),
        ProgressFormat::Json,
        None,
    )
    .unwrap();
    assert_eq!(
        fs::read(installed.join("extra/model.fp32")).unwrap(),
        b"float model"
    );
    assert!(!installed.join("model.int8").exists());
    verify(&paths, spec).unwrap();

    fs::write(installed.join("extra/model.fp32"), b"broken file").unwrap();
    install(
        &paths,
        spec,
        Some(&archive_path),
        ProgressFormat::Human,
        None,
    )
    .unwrap();
    verify(&paths, spec).unwrap();
}

#[test]
fn supplemental_install_rejects_a_directory_at_the_superseded_asset_path() {
    let root = temp("superseded-directory");
    let archive_bytes = archive_with_superseded_directory("tiny-root");
    let archive_path = root.join("tiny.tar.bz2");
    fs::write(&archive_path, &archive_bytes).unwrap();
    let spec = spec_with_supplement(&archive_bytes);
    let app_paths = paths(&root);
    let downloads = app_paths.data_dir.join("downloads");
    fs::create_dir_all(&downloads).unwrap();
    fs::write(downloads.join("tiny.extra-model.fp32"), b"float model").unwrap();

    let error = install(
        &app_paths,
        spec,
        Some(&archive_path),
        ProgressFormat::Human,
        None,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("remove superseded model asset"));
    assert!(!model_directory(&app_paths, spec).exists());
    assert!(
        !app_paths
            .data_dir
            .join("models")
            .join(format!(".tiny.install-{}", std::process::id()))
            .exists()
    );
}

#[test]
fn supplemental_download_and_paths_are_strictly_verified() {
    let root = temp("supplement-verify");
    let archive_bytes = archive("tiny-root", "model.bin", b"tiny model");
    let spec = spec_with_supplement(&archive_bytes);
    let asset = &spec.supplemental_files[0];
    let part = root.join("download.part");
    let target = root.join("download");
    write_pinned_download(
        &b"float model"[..],
        &part,
        &target,
        asset.size,
        asset.sha256,
        spec,
        ProgressFormat::Json,
    )
    .unwrap();
    verify_pinned_file(&target, asset.size, asset.sha256).unwrap();
    assert!(
        write_pinned_download(
            &b"short"[..],
            &part,
            &target,
            asset.size,
            asset.sha256,
            spec,
            ProgressFormat::Human,
        )
        .is_err()
    );
    assert!(validate_relative_file("../escape").is_err());
    assert!(validate_relative_file("/absolute").is_err());
    assert!(validate_relative_file("").is_err());
}

#[test]
fn unavailable_catalog_downloads_fail_without_leaving_partial_files() {
    let root = temp("network-failures");
    let archive_bytes = archive("tiny-root", "model.bin", b"tiny model");
    let mut unavailable = *spec(&archive_bytes, "http://127.0.0.1:9/model.tar.bz2");
    unavailable.id = "unavailable";
    let unavailable = Box::leak(Box::new(unavailable));
    let app_paths = paths(&root);
    fs::create_dir_all(app_paths.data_dir.join("downloads")).unwrap();
    assert!(download_archive(&app_paths, unavailable, ProgressFormat::Json).is_err());
    assert!(
        !app_paths
            .data_dir
            .join("downloads/unavailable.tar.bz2.part")
            .exists()
    );

    let mut with_supplement = *spec_with_supplement(&archive_bytes);
    with_supplement.id = "unavailable-supplement";
    let bad_asset: &'static [SupplementalFile] = Box::leak(
        vec![SupplementalFile {
            url: "http://127.0.0.1:9/model.fp32",
            ..with_supplement.supplemental_files[0]
        }]
        .into_boxed_slice(),
    );
    with_supplement.supplemental_files = bad_asset;
    let with_supplement = Box::leak(Box::new(with_supplement));
    assert!(
        download_supplemental(
            &app_paths,
            with_supplement,
            &bad_asset[0],
            ProgressFormat::Human,
        )
        .is_err()
    );
    assert!(
        !app_paths
            .data_dir
            .join("downloads/unavailable-supplement.extra-model.fp32.part")
            .exists()
    );
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

    let installed = install(&paths, spec, None, ProgressFormat::Human, None).unwrap();
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

    let error = install(
        &paths,
        spec,
        Some(&archive_path),
        ProgressFormat::Human,
        None,
    )
    .unwrap_err();
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

#[test]
fn failed_model_preparation_cleans_staging_and_supplement_parts() {
    let root = temp("prepare-cleanup");
    let archive_bytes = archive("tiny-root", "model.bin", b"tiny model");
    let broken_archive = root.join("broken.tar.bz2");
    fs::write(&broken_archive, b"not a bzip archive").unwrap();
    let model_spec = spec(b"not a bzip archive", "http://unused.invalid/model");
    let app_paths = paths(&root);
    let error = install(
        &app_paths,
        model_spec,
        Some(&broken_archive),
        ProgressFormat::Human,
        None,
    )
    .unwrap_err();
    assert!(
        format!("{error:#}").contains("prepare verified model installation"),
        "{error:#}"
    );
    assert!(
        !app_paths
            .data_dir
            .join("models")
            .join(format!(".tiny.install-{}", std::process::id()))
            .exists()
    );

    let supplemental_spec = spec_with_supplement(&archive_bytes);
    let asset = &supplemental_spec.supplemental_files[0];
    let part = root.join("supplement.part");
    let target = root.join("supplement");
    assert!(
        write_pinned_download(
            &b"float model extra"[..],
            &part,
            &target,
            asset.size,
            asset.sha256,
            supplemental_spec,
            ProgressFormat::Human,
        )
        .is_err()
    );
    assert!(!part.exists());
    assert!(
        write_pinned_download(
            &b"wrong model"[..],
            &part,
            &target,
            asset.size,
            asset.sha256,
            supplemental_spec,
            ProgressFormat::Human,
        )
        .is_err()
    );
    assert!(!part.exists());

    let destination = root.join("installed");
    fs::create_dir_all(&destination).unwrap();
    let wrong_source = root.join("wrong-source");
    fs::write(&wrong_source, b"bad").unwrap();
    assert!(install_supplemental(&wrong_source, &destination, asset).is_err());
    assert!(!destination.join("extra/.model.fp32.part").exists());
}

fn local_download(body: Vec<u8>) -> Option<(String, thread::JoinHandle<()>)> {
    let listener = match TcpListener::bind("127.0.0.1:0") {
        Ok(listener) => listener,
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return None,
        Err(error) => panic!("bind local download fixture: {error}"),
    };
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = [0_u8; 2048];
        let _ = stream.read(&mut request).unwrap();
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .unwrap();
        stream.write_all(&body).unwrap();
    });
    Some((format!("http://{address}/asset"), server))
}

#[test]
fn local_http_downloads_are_pinned_before_entering_the_install_cache() {
    let root = temp("local-downloads");
    let app_paths = paths(&root);
    fs::create_dir_all(app_paths.data_dir.join("downloads")).unwrap();

    let archive_bytes = archive("tiny-root", "model.bin", b"tiny model");
    let Some((archive_url, archive_server)) = local_download(archive_bytes.clone()) else {
        return;
    };
    let archive_spec = spec(&archive_bytes, &archive_url);
    let downloaded = download_archive(&app_paths, archive_spec, ProgressFormat::Json).unwrap();
    archive_server.join().unwrap();
    assert_eq!(fs::read(downloaded).unwrap(), archive_bytes);

    let supplement = b"float model".to_vec();
    let (supplement_url, supplement_server) = local_download(supplement.clone()).unwrap();
    let base = spec_with_supplement(&archive_bytes);
    let supplemental_files: &'static [SupplementalFile] = Box::leak(
        vec![SupplementalFile {
            url: leak(supplement_url),
            ..base.supplemental_files[0]
        }]
        .into_boxed_slice(),
    );
    let with_supplement = Box::leak(Box::new(ModelSpec {
        id: "local-supplement",
        supplemental_files,
        ..*base
    }));
    let downloaded = download_supplemental(
        &app_paths,
        with_supplement,
        &supplemental_files[0],
        ProgressFormat::Json,
    )
    .unwrap();
    supplement_server.join().unwrap();
    assert_eq!(fs::read(downloaded).unwrap(), supplement);

    let large = (0..1_200_000)
        .map(|index| ((index * 31 + 17) % 251) as u8)
        .collect::<Vec<_>>();
    let (large_url, large_server) = local_download(large.clone()).unwrap();
    let large_spec = spec(&large, &large_url);
    let downloaded = download_archive(&app_paths, large_spec, ProgressFormat::Json).unwrap();
    large_server.join().unwrap();
    assert_eq!(fs::read(downloaded).unwrap(), large);

    let (large_supplement_url, large_supplement_server) = local_download(large.clone()).unwrap();
    let large_asset = Box::leak(Box::new(SupplementalFile {
        path: "large/model.bin",
        supersedes: "",
        url: leak(large_supplement_url),
        size: large.len() as u64,
        sha256: leak(digest(&large)),
    }));
    let downloaded =
        download_supplemental(&app_paths, large_spec, large_asset, ProgressFormat::Human).unwrap();
    large_supplement_server.join().unwrap();
    assert_eq!(fs::read(downloaded).unwrap(), large);
}
