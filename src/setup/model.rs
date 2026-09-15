use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::catalog::{ModelFile, ModelSpec};
use crate::paths::AppPaths;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, clap::ValueEnum)]
pub enum ProgressFormat {
    #[default]
    Human,
    Json,
}

#[derive(Serialize)]
struct Event<'a> {
    event: &'a str,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    file: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    current: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total: Option<u64>,
}

pub fn model_directory(paths: &AppPaths, spec: &ModelSpec) -> PathBuf {
    paths.data_dir.join("models").join(spec.id)
}

pub fn verify(paths: &AppPaths, spec: &ModelSpec) -> Result<()> {
    verify_directory(&model_directory(paths, spec), spec)
}

pub fn verify_at(directory: &Path, spec: &ModelSpec) -> Result<()> {
    verify_directory(directory, spec)
}

/// Install every pinned file into a staging directory and publish it atomically.
///
/// `source_override` may be a directory with the catalog layout. A regular
/// file is accepted only for a one-file model.
pub fn install(
    paths: &AppPaths,
    spec: &ModelSpec,
    source_override: Option<&Path>,
    progress: ProgressFormat,
    accepted_license: Option<&str>,
) -> Result<PathBuf> {
    let target = model_directory(paths, spec);
    if verify_directory(&target, spec).is_ok() {
        emit(progress, "already-installed", spec, None, None, None)?;
        return Ok(target);
    }
    validate_install_authorization(spec, source_override, accepted_license)?;

    let models = paths.data_dir.join("models");
    fs::create_dir_all(&models)?;
    fs::create_dir_all(paths.data_dir.join("downloads"))?;
    let staging = models.join(format!(".{}.install-{}", spec.id, std::process::id()));
    if staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    fs::create_dir_all(&staging)?;

    let prepared = (|| -> Result<()> {
        match source_override {
            Some(source) => materialize_local_source(source, &staging, spec, progress),
            None => materialize_downloads(paths, &staging, spec, progress),
        }?;
        install_model_license(&staging, spec)?;
        write_install_manifest(&staging, spec, source_override, accepted_license)?;
        verify_directory(&staging, spec)
    })();
    if let Err(error) = prepared {
        let _ = fs::remove_dir_all(&staging);
        return Err(error).context("prepare verified model installation");
    }

    let old = models.join(format!(".{}.old-{}", spec.id, std::process::id()));
    if old.exists() {
        fs::remove_dir_all(&old)?;
    }
    if target.exists() {
        fs::rename(&target, &old)?;
    }
    if let Err(error) = fs::rename(&staging, &target) {
        if old.exists() {
            let _ = fs::rename(&old, &target);
        }
        return Err(error).context("activate model directory");
    }
    let _ = fs::remove_dir_all(&old);
    emit(progress, "installed", spec, None, None, None)?;
    Ok(target)
}

fn validate_install_authorization(
    spec: &ModelSpec,
    source_override: Option<&Path>,
    accepted_license: Option<&str>,
) -> Result<()> {
    if source_override.is_none()
        && (!spec.downloadable || spec.files.iter().any(|file| file.url.is_empty()))
    {
        bail!(
            "{} is user-supplied only because its model terms are {}; provide its pinned files with --source",
            spec.id,
            spec.license_status
        );
    }
    if spec.requires_acceptance && accepted_license != Some(spec.license) {
        bail!(
            "installing {} requires acceptance of {}; review {} and rerun with --accept-license {}",
            spec.id,
            spec.license,
            spec.license_url,
            spec.license
        );
    }
    Ok(())
}

fn materialize_local_source(
    source: &Path,
    staging: &Path,
    spec: &ModelSpec,
    progress: ProgressFormat,
) -> Result<()> {
    if source.is_file() && spec.files.len() != 1 {
        bail!(
            "{} contains {} files; --source must point to a directory with the catalog layout",
            spec.id,
            spec.files.len()
        );
    }
    if !source.is_dir() && !source.is_file() {
        bail!("model source does not exist: {}", source.display());
    }
    for file in spec.files {
        validate_relative_file(file.path)?;
        let input = if source.is_dir() {
            source.join(file.path)
        } else {
            source.to_owned()
        };
        verify_pinned_file(&input, file.size, file.sha256)?;
        copy_verified(&input, &staging.join(file.path), file)?;
        emit(
            progress,
            "source-file",
            spec,
            Some(file.path),
            Some(file.size),
            Some(file.size),
        )?;
    }
    Ok(())
}

fn materialize_downloads(
    paths: &AppPaths,
    staging: &Path,
    spec: &ModelSpec,
    progress: ProgressFormat,
) -> Result<()> {
    for file in spec.files {
        let cached = download_file(paths, spec, file, progress)?;
        copy_verified(&cached, &staging.join(file.path), file)?;
    }
    Ok(())
}

fn copy_verified(source: &Path, target: &Path, file: &ModelFile) -> Result<()> {
    let parent = target
        .parent()
        .context("model file has no parent directory")?;
    fs::create_dir_all(parent)?;
    let name = target
        .file_name()
        .context("model file has no file name")?
        .to_string_lossy();
    let part = parent.join(format!(".{name}.part-{}", std::process::id()));
    let _ = fs::remove_file(&part);
    let copied = (|| -> Result<()> {
        fs::copy(source, &part)?;
        verify_pinned_file(&part, file.size, file.sha256)?;
        fs::rename(&part, target)?;
        Ok(())
    })();
    if copied.is_err() {
        let _ = fs::remove_file(part);
    }
    copied
}

fn download_file(
    paths: &AppPaths,
    spec: &ModelSpec,
    file: &ModelFile,
    progress: ProgressFormat,
) -> Result<PathBuf> {
    validate_relative_file(file.path)?;
    if file.url.is_empty() {
        bail!("{} has no download URL for {}", spec.id, file.path);
    }
    let target = paths
        .data_dir
        .join("downloads")
        .join(spec.id)
        .join(file.path);
    if verify_pinned_file(&target, file.size, file.sha256).is_ok() {
        emit(
            progress,
            "cached",
            spec,
            Some(file.path),
            Some(file.size),
            Some(file.size),
        )?;
        return Ok(target);
    }
    let parent = target
        .parent()
        .context("download has no parent directory")?;
    fs::create_dir_all(parent)?;
    let part = parent.join(format!(
        ".{}.part-{}",
        target.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id()
    ));
    let _ = fs::remove_file(&part);
    emit(
        progress,
        "download-start",
        spec,
        Some(file.path),
        Some(0),
        Some(file.size),
    )?;
    let response = ureq::get(file.url)
        .call()
        .with_context(|| format!("download {} from {}", file.path, file.url))?;
    let result =
        write_pinned_download(response.into_reader(), &part, &target, spec, file, progress);
    if result.is_err() {
        let _ = fs::remove_file(part);
    }
    result?;
    Ok(target)
}

fn write_pinned_download(
    mut input: impl Read,
    part: &Path,
    target: &Path,
    spec: &ModelSpec,
    file: &ModelFile,
    progress: ProgressFormat,
) -> Result<()> {
    let output_file = File::create(part)?;
    let mut output = BufWriter::new(output_file);
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut reported = 0_u64;
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total += count as u64;
        if total > file.size {
            bail!("download exceeded expected size for {}", file.path);
        }
        hasher.update(&buffer[..count]);
        output.write_all(&buffer[..count])?;
        if should_report_progress(total, reported, file.size) {
            emit(
                progress,
                "download-progress",
                spec,
                Some(file.path),
                Some(total),
                Some(file.size),
            )?;
            reported = total;
        }
    }
    output.flush()?;
    output.get_ref().sync_all()?;
    if total != file.size {
        bail!(
            "downloaded {total} bytes for {}, expected {}",
            file.path,
            file.size
        );
    }
    if format!("{:x}", hasher.finalize()) != file.sha256 {
        bail!("download checksum mismatch for {}", file.path);
    }
    fs::rename(part, target)?;
    emit(
        progress,
        "downloaded",
        spec,
        Some(file.path),
        Some(total),
        Some(total),
    )
}

fn should_report_progress(current: u64, last: u64, total: u64) -> bool {
    current.saturating_sub(last) >= 1024 * 1024 || current == total
}

fn install_model_license(directory: &Path, spec: &ModelSpec) -> Result<()> {
    let Some(text) = crate::catalog::model_license_text(spec) else {
        return Ok(());
    };
    validate_relative_file(spec.license_file)?;
    fs::write(directory.join(spec.license_file), text)
        .with_context(|| format!("write {} model license", spec.id))?;
    verify_pinned_file(
        &directory.join(spec.license_file),
        text.len() as u64,
        spec.license_sha256,
    )
}

fn write_install_manifest(
    directory: &Path,
    spec: &ModelSpec,
    source_override: Option<&Path>,
    accepted_license: Option<&str>,
) -> Result<()> {
    let accepted_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let acceptance = spec.requires_acceptance.then(|| {
        serde_json::json!({
            "license": accepted_license.expect("authorization checked before installation"),
            "license_url": spec.license_url,
            "license_file": spec.license_file,
            "license_sha256": spec.license_sha256,
            "accepted_at_unix_seconds": accepted_at,
        })
    });
    let manifest = serde_json::json!({
        "schema": 1,
        "catalog": spec,
        "provenance": {
            "source": if source_override.is_some() { "user-supplied" } else { "catalog-download" },
            "source_path": source_override.map(|path| path.display().to_string()),
            "source_revision": spec.source_revision,
            "artifact_source": spec.artifact_source,
            "artifact_revision": spec.artifact_revision,
            "original_model_source": spec.original_model_source,
            "original_model_revision": spec.original_model_revision,
            "files": spec.files,
            "modified": false,
        },
        "license_acceptance": acceptance,
    });
    fs::write(
        directory.join(".omaspeak-model.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )
    .context("write model provenance manifest")
}

fn validate_relative_file(path: &str) -> Result<()> {
    let path = Path::new(path);
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        bail!("model asset has unsafe path {}", path.display());
    }
    Ok(())
}

fn verify_pinned_file(path: &Path, expected_size: u64, expected_sha256: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("model asset is missing: {}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.len() != expected_size {
        bail!("model asset has wrong size: {}", path.display());
    }
    if sha256_file(path)? != expected_sha256 {
        bail!("model asset checksum mismatch: {}", path.display());
    }
    Ok(())
}

fn verify_directory(directory: &Path, spec: &ModelSpec) -> Result<()> {
    for file in spec.files {
        validate_relative_file(file.path)?;
        verify_pinned_file(&directory.join(file.path), file.size, file.sha256)?;
    }
    if let Some(text) = crate::catalog::model_license_text(spec) {
        validate_relative_file(spec.license_file)?;
        verify_pinned_file(
            &directory.join(spec.license_file),
            text.len() as u64,
            spec.license_sha256,
        )?;
    }
    verify_manifest(directory, spec)
}

fn verify_manifest(directory: &Path, spec: &ModelSpec) -> Result<()> {
    let path = directory.join(".omaspeak-model.json");
    let manifest: serde_json::Value = serde_json::from_slice(
        &fs::read(&path).with_context(|| format!("read model manifest {}", path.display()))?,
    )
    .with_context(|| format!("parse model manifest {}", path.display()))?;
    if manifest.as_object().is_none_or(|object| object.len() != 4) {
        bail!(
            "model manifest has unexpected top-level fields: {}",
            path.display()
        );
    }
    if manifest.get("schema") != Some(&serde_json::json!(1)) {
        bail!(
            "model manifest has an unsupported schema: {}",
            path.display()
        );
    }
    if manifest.get("catalog") != Some(&serde_json::to_value(spec)?) {
        bail!("model manifest catalog does not match the active pinned catalog");
    }
    let provenance = manifest
        .get("provenance")
        .and_then(serde_json::Value::as_object)
        .context("model manifest is missing provenance")?;
    if provenance.len() != 9 {
        bail!("model manifest provenance has unexpected fields");
    }
    let source = provenance
        .get("source")
        .and_then(serde_json::Value::as_str)
        .context("model manifest is missing its source kind")?;
    let source_path = provenance
        .get("source_path")
        .context("model manifest is missing source_path")?;
    match source {
        "catalog-download" if source_path.is_null() => {}
        "user-supplied" if source_path.as_str().is_some_and(|path| !path.is_empty()) => {}
        _ => bail!("model manifest source and source_path are inconsistent"),
    }
    for (key, expected) in [
        ("source_revision", serde_json::json!(spec.source_revision)),
        ("artifact_source", serde_json::json!(spec.artifact_source)),
        (
            "artifact_revision",
            serde_json::json!(spec.artifact_revision),
        ),
        (
            "original_model_source",
            serde_json::json!(spec.original_model_source),
        ),
        (
            "original_model_revision",
            serde_json::json!(spec.original_model_revision),
        ),
        ("files", serde_json::to_value(spec.files)?),
        ("modified", serde_json::json!(false)),
    ] {
        if provenance.get(key) != Some(&expected) {
            bail!("model manifest provenance field {key} does not match the catalog");
        }
    }
    let acceptance = manifest
        .get("license_acceptance")
        .context("model manifest is missing license_acceptance")?;
    if spec.requires_acceptance {
        let acceptance = acceptance
            .as_object()
            .context("model manifest is missing required license acceptance")?;
        if acceptance.len() != 5
            || acceptance.get("license") != Some(&serde_json::json!(spec.license))
            || acceptance.get("license_url") != Some(&serde_json::json!(spec.license_url))
            || acceptance.get("license_file") != Some(&serde_json::json!(spec.license_file))
            || acceptance.get("license_sha256") != Some(&serde_json::json!(spec.license_sha256))
            || acceptance
                .get("accepted_at_unix_seconds")
                .and_then(serde_json::Value::as_u64)
                .is_none()
        {
            bail!("model manifest license acceptance does not match the catalog");
        }
    } else if !acceptance.is_null() {
        bail!("model manifest records unexpected license acceptance");
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut input = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn emit(
    format: ProgressFormat,
    event: &'static str,
    spec: &ModelSpec,
    file: Option<&str>,
    current: Option<u64>,
    total: Option<u64>,
) -> Result<()> {
    match format {
        ProgressFormat::Human => match (file, current, total) {
            (Some(file), Some(current), Some(total)) if event == "download-progress" => {
                eprint!(
                    "\rDownloading {file}: {:>3}%",
                    current.saturating_mul(100) / total.max(1)
                );
                std::io::stderr().flush()?;
            }
            _ => {
                if event == "downloaded" {
                    eprintln!();
                }
                if let Some(file) = file {
                    eprintln!("{event}: {} ({file})", spec.id);
                } else {
                    eprintln!("{event}: {}", spec.id);
                }
            }
        },
        ProgressFormat::Json => println!(
            "{}",
            serde_json::to_string(&Event {
                event,
                model: spec.id,
                file,
                current,
                total,
            })?
        ),
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/setup_model.rs"]
mod tests;
