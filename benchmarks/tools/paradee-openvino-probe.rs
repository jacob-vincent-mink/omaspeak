use anyhow::{Context, Result};
use openvino::{Core, DeviceType, ElementType, Shape, Tensor};
fn probe() -> Result<()> {
    let device = std::env::args().nth(1).unwrap_or_else(|| "CPU".into());
    let path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/tmp/paradee-survey/paradee_int8.onnx".into());
    openvino_sys::library::load_from("/usr/lib/libopenvino_c.so.2026.4.0")
        .map_err(anyhow::Error::msg)?;
    let mut core = Core::new_with_config("/usr/lib/openvino/plugins.xml")?;
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read("/tmp/paradee-survey/config.json")?)?;
    let mut ids = vec![0_i64];
    for ch in "həlˈoʊ wˈɜɹld".chars() {
        ids.push(config["vocab"][ch.to_string()].as_i64().context("missing phoneme")?);
    }
    ids.push(0);
    let mut model = core.read_model_from_file(&path, "").context("import Paradee ONNX")?;
    println!("imported");
    if std::env::var_os("PARADEE_STATIC").is_some() {
        let shape = openvino::PartialShape::new_static(2, &[1, ids.len() as i64])?;
        model.reshape(&[("input_ids", &shape)])?;
        println!("reshaped static input");
    }
    if device == "GPU" && std::env::var_os("PARADEE_F32").is_some() {
        core.set_property(&DeviceType::from("GPU"), &openvino::RwPropertyKey::HintInferencePrecision, "f32")?;
        println!("GPU f32 hint");
    }
    let mut compiled = core.compile_model(&model, DeviceType::from(device.as_str())).context("compile Paradee")?;
    println!("compiled");
    let mut input = Tensor::new(ElementType::I64, &Shape::new(&[1, ids.len() as i64])?)?;
    input.get_data_mut::<i64>()?.copy_from_slice(&ids);
    let mut speed = Tensor::new(ElementType::F32, &Shape::new(&[1])?)?;
    speed.get_data_mut::<f32>()?[0] = 1.0;
    let mut request = compiled.create_infer_request()?;
    request.set_tensor("input_ids", &input)?;
    request.set_tensor("speed", &speed)?;
    request.infer().context("infer Paradee")?;
    let waveform = request.get_tensor("waveform")?;
    let samples = waveform.get_data::<f32>()?;
    println!(
        "samples={} finite={}",
        samples.len(),
        samples.iter().all(|x| x.is_finite())
    );
    Ok(())
}
fn main() {
    if let Err(error) = probe() {
        eprintln!("{error:#}");
        unsafe {
            let detail = openvino_sys::ov_get_last_err_msg();
            if !detail.is_null() {
                eprintln!("{}", std::ffi::CStr::from_ptr(detail).to_string_lossy());
            }
        }
        std::process::exit(1);
    }
}
