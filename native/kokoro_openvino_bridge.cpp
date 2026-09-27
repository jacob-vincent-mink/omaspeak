// Small C ABI for OpenVINO GenAI's C++ Text2SpeechPipeline.
// The Rust executable loads this optional provider at runtime.
#include <openvino/genai/speech_generation/text2speech_pipeline.hpp>
#include <openvino/genai/version.hpp>
#include <openvino/openvino.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// GenAI's TTS implementation uses this core. Registering on a separate
// ov::Core would leave the pipeline unable to see distribution-split plugins.
namespace ov::genai::utils { ov::Core& singleton_core(); }

namespace {
void require_2026_4(const char* label, const char* build) {
    int year = 0;
    int minor = 0;
    if (!build || std::sscanf(build, "%d.%d", &year, &minor) != 2 ||
        year < 2026 || (year == 2026 && minor < 4))
        throw std::runtime_error(std::string(label) + " 2026.4 or newer is required");
}

void ensure_device(const char* device, const char* plugin) {
    require_2026_4("OpenVINO", ov::get_openvino_version().buildNumber);
    require_2026_4("OpenVINO GenAI", ov::genai::get_version().buildNumber);
    if (!device || !*device) throw std::runtime_error("device is empty");
    const std::string requested(device);
    if (requested != "CPU" && requested != "GPU" && requested != "NPU")
        throw std::runtime_error("Kokoro requires CPU, GPU, or NPU");
    auto& core = ov::genai::utils::singleton_core();
    auto devices = core.get_available_devices();
    auto found = [&] {
        return std::any_of(devices.begin(), devices.end(), [&](const auto& name) {
            return name == requested || name.rfind(requested + ".", 0) == 0;
        });
    };
    if (!found() && plugin && *plugin) {
        core.register_plugin(std::string(plugin), requested);
        devices = core.get_available_devices();
    }
    if (!found()) throw std::runtime_error("requested OpenVINO device is unavailable: " + requested);
}

struct Pipeline {
    explicit Pipeline(const char* model_dir, const char* device)
        : value(model_dir, device), device_name(device) {}
    ov::genai::Text2SpeechPipeline value;
    std::string device_name;
};

void error_text(char* output, size_t capacity, const std::string& message) {
    if (output && capacity) {
        const size_t count = std::min(capacity - 1, message.size());
        std::memcpy(output, message.data(), count);
        output[count] = '\0';
    }
}

template <typename F>
int guarded(F&& action, char* error, size_t capacity) {
    try {
        action();
        error_text(error, capacity, "");
        return 0;
    } catch (const std::exception& failure) {
        error_text(error, capacity, failure.what());
        return 1;
    } catch (...) {
        error_text(error, capacity, "unknown native provider failure");
        return 1;
    }
}
}  // namespace

extern "C" {
unsigned omaspeak_kokoro_abi_version() { return 2; }

int omaspeak_kokoro_probe(const char* device, const char* plugin, char* error, size_t capacity) {
    return guarded([&] {
        ensure_device(device, plugin);
    }, error, capacity);
}

int omaspeak_kokoro_open(const char* model_dir, const char* device, const char* plugin, void** output,
                         char* error, size_t capacity) {
    return guarded([&] {
        if (!output || !model_dir || !device) throw std::runtime_error("invalid pipeline argument");
        ensure_device(device, plugin);
        *output = new Pipeline(model_dir, device);
    }, error, capacity);
}

int omaspeak_kokoro_generate(void* opaque, const char* text, const float* voice,
                             size_t voice_count, float** output, size_t* output_count,
                             unsigned* sample_rate, char* error, size_t capacity) {
    return guarded([&] {
        if (!opaque || !text || !voice || !output || !output_count || !sample_rate)
            throw std::runtime_error("invalid synthesis argument");
        auto& pipeline = static_cast<Pipeline*>(opaque)->value;
        const auto shape = pipeline.get_speaker_embedding_shape();
        size_t expected = 1;
        for (const size_t dimension : shape) expected *= dimension;
        if (voice_count != expected) throw std::runtime_error("speaker embedding shape mismatch");
        ov::Tensor embedding(ov::element::f32, shape);
        std::memcpy(embedding.data<float>(), voice, voice_count * sizeof(float));
        ov::AnyMap properties{{"language", std::string("en-us")}};
        const auto result = pipeline.generate(std::string(text), embedding, properties);
        if (result.speeches.size() != 1 || result.output_sample_rate != 24000)
            throw std::runtime_error("unexpected Kokoro audio format");
        const auto& waveform = result.speeches.front();
        if (waveform.get_element_type() != ov::element::f32)
            throw std::runtime_error("Kokoro audio is not float32");
        const size_t count = waveform.get_size();
        if (!count || count > 128 * 1024 * 1024)
            throw std::runtime_error("Kokoro audio size is invalid");
        const auto* samples = waveform.data<const float>();
        if (!std::all_of(samples, samples + count, [](float sample) { return std::isfinite(sample); }))
            throw std::runtime_error("Kokoro returned non-finite audio");
        auto* copy = static_cast<float*>(std::malloc(count * sizeof(float)));
        if (!copy) throw std::bad_alloc();
        std::memcpy(copy, samples, count * sizeof(float));
        *output = copy;
        *output_count = count;
        *sample_rate = result.output_sample_rate;
    }, error, capacity);
}

void omaspeak_kokoro_free_samples(float* samples) { std::free(samples); }
void omaspeak_kokoro_close(void* pipeline) { delete static_cast<Pipeline*>(pipeline); }
}
