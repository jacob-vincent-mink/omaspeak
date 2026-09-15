use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use bzip2::read::BzDecoder;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::catalog::{ModelSpec, SupplementalFile};
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

pub fn install(
    paths: &AppPaths,
    spec: &ModelSpec,
    archive_override: Option<&Path>,
    progress: ProgressFormat,
    accepted_license: Option<&str>,
) -> Result<PathBuf> {
    let target = model_directory(paths, spec);
    if verify_directory(&target, spec).is_ok() {
        emit(progress, "already-installed", spec, None, None)?;
        return Ok(target);
    }

    validate_install_authorization(spec, archive_override, accepted_license)?;

    fs::create_dir_all(paths.data_dir.join("models"))?;
    fs::create_dir_all(paths.data_dir.join("downloads"))?;
    let archive = match archive_override {
        Some(path) => path.to_owned(),
        None => download_archive(paths, spec, progress)?,
    };
    let directory_source = archive.is_dir();
    if !directory_source {
        verify_archive(&archive, spec)?;
    }
    let supplemental_files = if directory_source {
        Vec::new()
    } else {
        spec.supplemental_files
            .iter()
            .map(|asset| download_supplemental(paths, spec, asset, progress))
            .collect::<Result<Vec<_>>>()?
    };

    let staging =
        paths
            .data_dir
            .join("models")
            .join(format!(".{}.install-{}", spec.id, std::process::id()));
    if staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    fs::create_dir_all(&staging)?;
    emit(
        progress,
        if spec.single_file.is_some() {
            "install-file"
        } else {
            "extract"
        },
        spec,
        None,
        None,
    )?;
    let prepare = || -> Result<PathBuf> {
        let extracted = materialize_artifact(&archive, &staging, spec)
            .with_context(|| format!("install model artifact {}", archive.display()))?;
        for (asset, source) in spec.supplemental_files.iter().zip(&supplemental_files) {
            install_supplemental(source, &extracted, asset)?;
            if !asset.supersedes.is_empty() && asset.supersedes != asset.path {
                validate_relative_file(asset.supersedes)?;
                let superseded = extracted.join(asset.supersedes);
                if superseded.exists() {
                    fs::remove_file(&superseded).with_context(|| {
                        format!("remove superseded model asset {}", superseded.display())
                    })?;
                }
            }
        }
        install_model_license(&extracted, spec)?;
        verify_directory(&extracted, spec)?;
        Ok(extracted)
    };
    let extracted = match prepare() {
        Ok(extracted) => extracted,
        Err(error) => {
            let _ = fs::remove_dir_all(&staging);
            return Err(error).context("prepare verified model installation");
        }
    };

    let old =
        paths
            .data_dir
            .join("models")
            .join(format!(".{}.old-{}", spec.id, std::process::id()));
    if old.exists() {
        fs::remove_dir_all(&old)?;
    }
    if target.exists() {
        fs::rename(&target, &old)?;
    }
    if let Err(error) = fs::rename(&extracted, &target) {
        if old.exists() {
            let _ = fs::rename(&old, &target);
        }
        return Err(error).context("activate extracted model");
    }
    let activate = || -> Result<()> {
        write_install_manifest(&target, spec, archive_override, accepted_license)?;
        verify_directory(&target, spec)
    };
    if let Err(error) = activate() {
        let _ = fs::remove_dir_all(&target);
        if old.exists() {
            let _ = fs::rename(&old, &target);
        }
        let _ = fs::remove_dir_all(&staging);
        return Err(error).context("finalize installed model");
    }
    let _ = fs::remove_dir_all(&old);
    let _ = fs::remove_dir_all(&staging);
    emit(progress, "installed", spec, None, None)?;
    Ok(target)
}

fn validate_install_authorization(
    spec: &ModelSpec,
    archive_override: Option<&Path>,
    accepted_license: Option<&str>,
) -> Result<()> {
    if archive_override.is_none() && !spec.downloadable {
        bail!(
            "{} is user-supplied only because its model terms are {}; Omaspeak will not download it; supply a directory or archive you have the right to use with --archive",
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
    archive_override: Option<&Path>,
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
        "catalog": spec,
        "provenance": {
            "source": if let Some(source) = archive_override {
                if source.is_dir() { "user-supplied-directory" } else if spec.single_file.is_some() { "user-supplied-file" } else { "user-supplied-archive" }
            } else { "catalog-download" },
            "source_revision": spec.source_revision,
            "artifact_source": spec.artifact_source,
            "artifact_revision": spec.artifact_revision,
            "original_model_source": spec.original_model_source,
            "original_model_revision": spec.original_model_revision,
            "artifact_url": artifact_url(spec),
            "artifact_sha256": artifact_sha256(spec),
            "supplemental_assets": spec.supplemental_files,
            "modified": !spec.supplemental_files.is_empty(),
        },
        "license_acceptance": acceptance,
    });
    fs::write(
        directory.join(".omaspeak-model.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )
    .context("write model provenance manifest")
}

fn download_supplemental(
    paths: &AppPaths,
    spec: &ModelSpec,
    asset: &SupplementalFile,
    progress: ProgressFormat,
) -> Result<PathBuf> {
    validate_relative_file(asset.path)?;
    let cache_name = format!("{}.{}", spec.id, asset.path.replace(['/', '\\'], "-"));
    let target = paths.data_dir.join("downloads").join(cache_name);
    if verify_pinned_file(&target, asset.size, asset.sha256).is_ok() {
        emit(
            progress,
            "supplement-cached",
            spec,
            Some(asset.size),
            Some(asset.size),
        )?;
        return Ok(target);
    }

    let part = target.with_extension("part");
    let _ = fs::remove_file(&part);
    emit(
        progress,
        "supplement-download-start",
        spec,
        Some(0),
        Some(asset.size),
    )?;
    let response = ureq::get(asset.url)
        .call()
        .with_context(|| format!("download supplemental model asset {}", asset.url))?;
    write_pinned_download(
        response.into_reader(),
        &part,
        &target,
        asset.size,
        asset.sha256,
        spec,
        progress,
    )?;
    Ok(target)
}

fn write_pinned_download(
    mut input: impl Read,
    part: &Path,
    target: &Path,
    expected_size: u64,
    expected_sha256: &str,
    spec: &ModelSpec,
    progress: ProgressFormat,
) -> Result<()> {
    let result = (|| -> Result<()> {
        let file = File::create(part)?;
        let mut output = BufWriter::new(file);
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
            if total > expected_size {
                bail!("download exceeded expected size for {}", spec.id);
            }
            hasher.update(&buffer[..count]);
            output.write_all(&buffer[..count])?;
            if should_report_progress(total, reported, expected_size) {
                emit(
                    progress,
                    "supplement-download-progress",
                    spec,
                    Some(total),
                    Some(expected_size),
                )?;
                reported = total;
            }
        }
        output.flush()?;
        output.get_ref().sync_all()?;
        if total != expected_size {
            bail!(
                "downloaded {} bytes for {}, expected {}",
                total,
                spec.id,
                expected_size
            );
        }
        if format!("{:x}", hasher.finalize()) != expected_sha256 {
            bail!("download checksum mismatch for {}", spec.id);
        }
        fs::rename(part, target)?;
        emit(
            progress,
            "supplement-downloaded",
            spec,
            Some(total),
            Some(total),
        )?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(part);
    }
    result
}

fn install_supplemental(source: &Path, directory: &Path, asset: &SupplementalFile) -> Result<()> {
    validate_relative_file(asset.path)?;
    verify_pinned_file(source, asset.size, asset.sha256)?;
    let target = directory.join(asset.path);
    let parent = target
        .parent()
        .context("supplemental model asset has no parent directory")?;
    fs::create_dir_all(parent)?;
    let file_name = target
        .file_name()
        .context("supplemental model asset has no file name")?
        .to_string_lossy();
    let part = parent.join(format!(".{file_name}.part-{}", std::process::id()));
    let _ = fs::remove_file(&part);
    let result = (|| -> Result<()> {
        fs::copy(source, &part)?;
        verify_pinned_file(&part, asset.size, asset.sha256)?;
        fs::rename(&part, &target)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&part);
    }
    result
}

fn validate_relative_file(path: &str) -> Result<()> {
    let path = Path::new(path);
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        bail!(
            "supplemental model asset has unsafe path {}",
            path.display()
        );
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

fn download_archive(
    paths: &AppPaths,
    spec: &ModelSpec,
    progress: ProgressFormat,
) -> Result<PathBuf> {
    let cache_id = crate::catalog::models()
        .iter()
        .find(|candidate| {
            artifact_size(candidate) == artifact_size(spec)
                && artifact_sha256(candidate) == artifact_sha256(spec)
                && artifact_url(candidate) == artifact_url(spec)
        })
        .map_or(spec.id, |candidate| candidate.id);
    let target = paths
        .data_dir
        .join("downloads")
        .join(spec.single_file.map_or_else(
            || format!("{cache_id}.tar.bz2"),
            |file| format!("{cache_id}-{}", file.path.replace(['/', '\\'], "-")),
        ));
    if verify_archive(&target, spec).is_ok() {
        emit(
            progress,
            "cached",
            spec,
            Some(artifact_size(spec)),
            Some(artifact_size(spec)),
        )?;
        return Ok(target);
    }
    let part = target.with_extension("part");
    let _ = fs::remove_file(&part);
    emit(
        progress,
        "download-start",
        spec,
        Some(0),
        Some(artifact_size(spec)),
    )?;
    let response = ureq::get(artifact_url(spec))
        .call()
        .with_context(|| format!("download {}", artifact_url(spec)))?;
    write_download(response.into_reader(), &part, &target, spec, progress)?;
    Ok(target)
}

fn write_download(
    mut input: impl Read,
    part: &Path,
    target: &Path,
    spec: &ModelSpec,
    progress: ProgressFormat,
) -> Result<()> {
    let file = File::create(part)?;
    let mut output = BufWriter::new(file);
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
        if total > artifact_size(spec) {
            bail!("download exceeded expected size for {}", spec.id);
        }
        hasher.update(&buffer[..count]);
        output.write_all(&buffer[..count])?;
        if should_report_progress(total, reported, artifact_size(spec)) {
            emit(
                progress,
                "download-progress",
                spec,
                Some(total),
                Some(artifact_size(spec)),
            )?;
            reported = total;
        }
    }
    output.flush()?;
    output.get_ref().sync_all()?;
    if total != artifact_size(spec) {
        bail!(
            "downloaded {} bytes for {}, expected {}",
            total,
            spec.id,
            artifact_size(spec)
        );
    }
    let digest = format!("{:x}", hasher.finalize());
    if digest != artifact_sha256(spec) {
        bail!("download checksum mismatch for {}", spec.id);
    }
    fs::rename(part, target)?;
    emit(progress, "downloaded", spec, Some(total), Some(total))?;
    Ok(())
}

fn should_report_progress(current: u64, last: u64, total: u64) -> bool {
    current.saturating_sub(last) >= 1024 * 1024 || current == total
}

fn verify_archive(path: &Path, spec: &ModelSpec) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("model archive is missing: {}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.len() != artifact_size(spec) {
        bail!("model artifact size mismatch for {}", path.display());
    }
    let digest = sha256_file(path)?;
    if digest != artifact_sha256(spec) {
        bail!("model artifact checksum mismatch for {}", path.display());
    }
    Ok(())
}

fn materialize_artifact(archive: &Path, staging: &Path, spec: &ModelSpec) -> Result<PathBuf> {
    if archive.is_dir() {
        let directory = staging.join("model");
        fs::create_dir_all(&directory)?;
        for required in spec.required_files {
            validate_relative_file(required.path)?;
            let source = archive.join(required.path);
            verify_pinned_file(&source, required.size, required.sha256)?;
            let target = directory.join(required.path);
            fs::create_dir_all(target.parent().context("model asset has no parent")?)?;
            fs::copy(&source, &target).with_context(|| {
                format!(
                    "copy model asset {} to {}",
                    source.display(),
                    target.display()
                )
            })?;
        }
        return Ok(directory);
    }
    if let Some(file) = spec.single_file {
        validate_relative_file(file.path)?;
        let directory = staging.join(spec.id);
        let target = directory.join(file.path);
        fs::create_dir_all(target.parent().context("single-file model has no parent")?)?;
        fs::copy(archive, &target).with_context(|| {
            format!(
                "copy single-file model {} to {}",
                archive.display(),
                target.display()
            )
        })?;
        return Ok(directory);
    }
    let decoder = BzDecoder::new(BufReader::new(File::open(archive)?));
    let mut archive = tar::Archive::new(decoder);
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        if path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
        {
            bail!("archive contains unsafe path {}", path.display());
        }
        if path.components().next().and_then(|part| match part {
            Component::Normal(value) => value.to_str(),
            _ => None,
        }) != Some(spec.archive_root)
        {
            bail!("archive entry is outside expected root: {}", path.display());
        }
        let kind = entry.header().entry_type();
        if !kind.is_file() && !kind.is_dir() {
            bail!("archive contains unsupported entry {}", path.display());
        }
        if !entry.unpack_in(staging)? {
            bail!("archive entry escaped destination: {}", path.display());
        }
    }
    Ok(staging.join(spec.archive_root))
}

#[cfg(test)]
fn extract_archive(archive: &Path, staging: &Path, spec: &ModelSpec) -> Result<PathBuf> {
    materialize_artifact(archive, staging, spec)
}

fn artifact_url(spec: &ModelSpec) -> &str {
    spec.single_file.map_or(spec.archive_url, |file| file.url)
}

fn artifact_size(spec: &ModelSpec) -> u64 {
    spec.single_file.map_or(spec.archive_size, |file| file.size)
}

fn artifact_sha256(spec: &ModelSpec) -> &str {
    spec.single_file
        .map_or(spec.archive_sha256, |file| file.sha256)
}

fn verify_directory(directory: &Path, spec: &ModelSpec) -> Result<()> {
    for required in spec.required_files {
        let path = directory.join(required.path);
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("required model asset is missing: {}", path.display()))?;
        if !metadata.file_type().is_file() || metadata.len() != required.size {
            bail!("model asset has wrong size: {}", path.display());
        }
        if sha256_file(&path)? != required.sha256 {
            bail!("model asset checksum mismatch: {}", path.display());
        }
    }
    if let Some(text) = crate::catalog::model_license_text(spec) {
        validate_relative_file(spec.license_file)?;
        verify_pinned_file(
            &directory.join(spec.license_file),
            text.len() as u64,
            spec.license_sha256,
        )?;
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
    current: Option<u64>,
    total: Option<u64>,
) -> Result<()> {
    match format {
        ProgressFormat::Human => match (current, total) {
            (Some(current), Some(total)) if event.ends_with("download-progress") => {
                eprint!(
                    "\rDownloading {}: {:>3}%",
                    spec.id,
                    current.saturating_mul(100) / total.max(1)
                );
                std::io::stderr().flush()?;
            }
            _ => {
                if event.ends_with("downloaded") {
                    eprintln!();
                }
                eprintln!("{}: {}", event, spec.id);
            }
        },
        ProgressFormat::Json => println!(
            "{}",
            serde_json::to_string(&Event {
                event,
                model: spec.id,
                current,
                total
            })?
        ),
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/setup_model.rs"]
mod tests;
