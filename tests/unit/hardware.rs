use super::*;

use std::os::unix::fs::symlink;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn sandbox() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "omaspeak-hardware-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn pci_device(root: &Path, address: &str, vendor: &str, class: &str, driver: &str) {
    let device = root.join(address);
    let driver_path = Path::new("/sys/bus/pci/drivers").join(driver);
    fs::create_dir_all(&device).unwrap();
    fs::write(device.join("vendor"), vendor).unwrap();
    fs::write(device.join("class"), class).unwrap();
    symlink(driver_path, device.join("driver")).unwrap();
}

#[test]
fn sysfs_detection_recognizes_intel_npu_igpu_and_nvidia_gpu_evidence() {
    let root = sandbox();
    pci_device(&root, "0000:00:0b.0", "0x8086", "0x120000", "intel_vpu");
    pci_device(&root, "0000:00:02.0", "0x8086", "0x030000", "xe");
    pci_device(&root, "0000:01:00.0", "0x10de", "0x030200", "nvidia");
    pci_device(&root, "0000:00:00.0", "0x8086", "0x060000", "pcieport");

    let report = detect_at(&root);
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert!(report.cuda_gpu);
    assert!(report.intel_npu);
    assert!(report.intel_gpu);
    assert!(report.vulkan_candidate);
    assert_eq!(report.devices.len(), 3);
    assert_eq!(report.devices[0].driver.as_deref(), Some("xe"));
    assert_eq!(report.devices[1].driver.as_deref(), Some("intel_vpu"));
    assert_eq!(report.devices[2].driver.as_deref(), Some("nvidia"));
}

#[test]
fn recommendation_prefers_the_best_detected_provider_and_never_claims_readiness() {
    let providers = ProviderAvailability {
        packaged_cpu: true,
        cuda: false,
        openvino: true,
        vulkan: true,
    };
    let mut hardware = HardwareReport {
        cuda_gpu: true,
        intel_npu: true,
        intel_gpu: true,
        vulkan_candidate: true,
        ..Default::default()
    };
    let npu = recommend(&hardware, providers);
    assert_eq!(
        (npu.runtime, npu.device.as_str()),
        (Runtime::Openvino, "npu")
    );
    assert!(npu.hardware_detected);
    assert!(npu.provider_detected);
    assert!(!npu.ready);

    let cuda = recommend(
        &hardware,
        ProviderAvailability {
            cuda: true,
            ..providers
        },
    );
    assert_eq!((cuda.runtime, cuda.device.as_str()), (Runtime::Cuda, "gpu"));
    assert!(cuda.provider_detected);
    assert!(!cuda.ready);

    hardware.cuda_gpu = false;
    hardware.intel_npu = false;
    assert_eq!(recommend(&hardware, providers).device, "gpu");
    hardware.intel_gpu = false;
    assert_eq!(recommend(&hardware, providers).runtime, Runtime::Vulkan);
    hardware.vulkan_candidate = false;
    let cpu = recommend(&hardware, providers);
    assert_eq!(cpu.runtime, Runtime::Default);
    assert_eq!(cpu.device, "cpu");
    assert!(!cpu.hardware_detected);
    assert!(cpu.provider_detected);
    assert!(!cpu.ready);

    hardware.cuda_gpu = true;
    let unavailable = recommend(&hardware, ProviderAvailability::default());
    assert_eq!(unavailable.runtime, Runtime::Cuda);
    assert!(unavailable.hardware_detected);
    assert!(!unavailable.provider_detected);
    assert!(unavailable.detail.contains("complete external provider"));

    let available_cpu = recommend(
        &hardware,
        ProviderAvailability {
            packaged_cpu: true,
            ..Default::default()
        },
    );
    assert_eq!(available_cpu.runtime, Runtime::Default);
    assert!(!available_cpu.hardware_detected);
    assert!(available_cpu.provider_detected);
    assert!(available_cpu.detail.contains("incomplete"));
}

#[test]
fn missing_and_malformed_sysfs_are_advisory() {
    let root = sandbox();
    fs::create_dir(root.join("bad")).unwrap();
    fs::write(root.join("bad/vendor"), "not-hex").unwrap();
    fs::write(root.join("bad/class"), "0x030000").unwrap();
    let report = detect_at(&root);
    assert!(report.devices.is_empty());
    assert!(report.errors.is_empty());

    let missing = detect_at(&root.join("missing"));
    assert!(missing.devices.is_empty());
    assert_eq!(missing.errors.len(), 1);
}

#[test]
fn fresh_config_recognizes_an_external_audio_provider_as_an_accelerator_candidate() {
    let root = sandbox();
    let external = root.join("external/libaudiocpp.so.0");
    fs::create_dir_all(external.parent().unwrap()).unwrap();
    fs::write(&external, b"provider fixture").unwrap();
    let locations = LibraryPathReport {
        configured_library_dirs: Vec::new(),
        environment_library_dirs: vec![external.parent().unwrap().to_path_buf()],
        package_library_dirs: vec![root.join("package")],
        effective_library_dirs: vec![external.parent().unwrap().to_path_buf()],
        missing_library_dirs: Vec::new(),
        audiocpp_library: Some(external),
        openvino_library: None,
        openvino_plugins: None,
        runtime_loadable: Default::default(),
        remediation: Vec::new(),
    };
    let providers = provider_availability(&BackendConfig::default(), &locations);
    assert!(!providers.packaged_cpu);
    assert!(providers.cuda);
    assert!(providers.vulkan);
    assert!(!providers.openvino);
}
