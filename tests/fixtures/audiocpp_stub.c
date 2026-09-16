#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void exit_worker_with_diagnostics(const char *diagnostic) {
    for (int i = 0; i < 4096; ++i)
        fputs("native provider initialization detail padding\n", stderr);
    fprintf(stderr, "%s\n", diagnostic);
    fflush(stderr);
    exit(70);
}

typedef struct {
    const char *family_hint;
    const char *config_id;
    const char *weight_id;
    const char *model_spec_override;
} audiocpp_model_config;

typedef struct {
    const char *backend;
    int device;
    int threads;
} audiocpp_backend_config;

typedef struct { int unsupported; int fail_session; int null_session; } stub_model;
typedef struct { char text[64]; } stub_request;
typedef struct {
    float samples[882];
    size_t frames;
    int sample_rate;
    int channels;
    int null_samples;
} stub_result;

static int stub_mode;

uint32_t audiocpp_abi_version(void) { return 0x00000100u; }
const char *audiocpp_last_error(void) { return "stub provider error"; }

void *audiocpp_options_create(void) { return calloc(1, 1); }
int audiocpp_options_set(void *options, const char *key, const char *value) {
    if (!options || !key || !value) return 1;
    return strcmp(key, "fail") == 0 ? 31 : 0;
}
void audiocpp_options_free(void *options) { free(options); }

int audiocpp_registry_create(const char *path, void **registry) {
    (void)path;
    *registry = calloc(1, 1);
    return *registry ? 0 : 1;
}
size_t audiocpp_registry_family_count(const void *registry) { (void)registry; return 1; }
int audiocpp_registry_family(const void *registry, size_t index, const char **out) {
    (void)registry; static const char *names[] = {"supertonic"};
    if (index >= 1) return 1; *out = names[index]; return 0;
}
void audiocpp_registry_free(void *registry) { free(registry); }

int audiocpp_model_load(void *registry, const char *path,
                         const audiocpp_model_config *config, void *options,
                         void **model) {
    (void)options;
    if (!registry || !path || !config || !config->family_hint) return 1;
    if (strstr(path, "exit-load"))
        exit_worker_with_diagnostics("audio.cpp startup diagnostic from provider");
    if (strstr(path, "fail-load")) return 41;
    if (strstr(path, "null-model")) {
        *model = NULL;
        return 0;
    }
    stub_model *loaded = calloc(1, sizeof(*loaded));
    if (!loaded) return 1;
    loaded->unsupported = strstr(path, "unsupported") != NULL;
    loaded->fail_session = strstr(path, "fail-session") != NULL;
    loaded->null_session = strstr(path, "null-session") != NULL;
    stub_mode = 0;
    if (strstr(path, "null-request")) stub_mode = 1;
    if (strstr(path, "fail-voice")) stub_mode = 2;
    if (strstr(path, "fail-steps")) stub_mode = 3;
    if (strstr(path, "null-result")) stub_mode = 4;
    if (strstr(path, "fail-result")) stub_mode = 5;
    if (strstr(path, "null-samples")) stub_mode = 6;
    if (strstr(path, "huge-audio")) stub_mode = 7;
    if (strstr(path, "require-F1")) stub_mode = 8;
    *model = loaded;
    return *model ? 0 : 1;
}
void audiocpp_model_free(void *model) { free(model); }
int audiocpp_model_supports(void *model, const char *task, const char *mode) {
    return model && task && mode && !((stub_model *)model)->unsupported ? 1 : 0;
}

int audiocpp_session_create(void *model, const char *task, const char *mode,
                             const audiocpp_backend_config *config,
                             void *options, void **session) {
    (void)options;
    if (!model || !task || !mode || !config || !config->backend) return 1;
    stub_model *loaded = model;
    if (loaded->fail_session) return 42;
    if (loaded->null_session) {
        *session = NULL;
        return 0;
    }
    *session = calloc(1, 1);
    return *session ? 0 : 1;
}
void audiocpp_session_free(void *session) { free(session); }

void *audiocpp_request_create(void) {
    return stub_mode == 1 ? NULL : calloc(1, sizeof(stub_request));
}
void audiocpp_request_free(void *request) { free(request); }
int audiocpp_request_set_text(void *request, const char *text,
                              const char *language) {
    if (!request || !text || !language) return 1;
    if (strcmp(text, "fail-text") == 0) return 43;
    strncpy(((stub_request *)request)->text, text,
            sizeof(((stub_request *)request)->text) - 1);
    return 0;
}
int audiocpp_request_set_voice_id(void *request, const char *voice) {
    if (!request || !voice) return 1;
    if (stub_mode == 8 && strcmp(voice, "F1") != 0) return 44;
    return stub_mode == 2 ? 44 : 0;
}
int audiocpp_request_set_speaking_rate(void *request, float rate) {
    if (!request || rate <= 0.0f) return 1;
    return rate > 3.7f ? 45 : 0;
}
int audiocpp_request_set_option(void *request, const char *key,
                                const char *value) {
    if (!request || !key || !value) return 1;
    if (strcmp(key, "fail") == 0) return 46;
    if (stub_mode == 3 && strcmp(key, "num_inference_steps") == 0) return 46;
    return 0;
}

int audiocpp_session_run(void *session, void *request, void **result) {
    if (!session || !request) return 1;
    stub_request *input = request;
    if (strcmp(input->text, "fail-run") == 0) return 47;
    if (strcmp(input->text, "crash-run") == 0) {
        const char *marker = getenv("OMASPEAK_STUB_CRASHES");
        if (marker) {
            FILE *file = fopen(marker, "a");
            if (file) { fputs("crash\n", file); fclose(file); }
        }
        _exit(71);
    }
    if (strcmp(input->text, "stall-run") == 0) {
        const char *marker = getenv("OMASPEAK_STUB_STARTED");
        if (marker) {
            FILE *file = fopen(marker, "w");
            if (file) { fprintf(file, "%d", (int)getpid()); fclose(file); }
        }
        sleep(30);
    }
    if (stub_mode == 4) {
        *result = NULL;
        return 0;
    }
    stub_result *output = calloc(1, sizeof(*output));
    if (!output) return 1;
    for (size_t i = 0; i < 882; ++i)
        output->samples[i] = (i % 2 == 0) ? 0.05f : -0.05f;
    output->frames = strcmp(input->text, "empty-audio") == 0 ? 0 : 441;
    output->sample_rate = strcmp(input->text, "bad-rate") == 0 ? 16000 : 44100;
    output->channels = strcmp(input->text, "bad-channels") == 0 ? 2 : 1;
    output->null_samples = stub_mode == 6;
    if (stub_mode == 7) output->frames = 70000000;
    if (strcmp(input->text, "nan-audio") == 0)
        output->samples[0] = 0.0f / 0.0f;
    *result = output;
    return 0;
}

int audiocpp_result_audio(void *result, const float **samples, size_t *frames,
                           int *sample_rate, int *channels) {
    if (!result || !samples || !frames || !sample_rate || !channels) return 1;
    if (stub_mode == 5) return 48;
    stub_result *output = result;
    *samples = output->null_samples ? NULL : output->samples;
    *frames = output->frames;
    *sample_rate = output->sample_rate;
    *channels = output->channels;
    return 0;
}
void audiocpp_result_free(void *result) { free(result); }
