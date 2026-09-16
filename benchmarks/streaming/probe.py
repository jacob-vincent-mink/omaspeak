#!/usr/bin/env python3
"""File-only probe of pinned audio.cpp pull streaming; never opens audio devices."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--backend', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=8)
    parser.add_argument('--pull-delay-ms', type=float, default=0)
    parser.add_argument('--streaming-first', action='store_true')
    args = parser.parse_args()
    if args.repetitions < 1 or args.pull_delay_ms < 0:
        parser.error('repetitions must be positive and pull delay nonnegative')
    args.out.mkdir(parents=True, exist_ok=False)
    lib = C.CDLL(args.library)
    P, S, I, Z = C.c_void_p, C.c_char_p, C.c_int, C.c_size_t
    FP = C.POINTER(C.c_float)
    class ModelConfig(C.Structure):
        _fields_ = [('family', S), ('config', S), ('weight', S), ('override', S)]
    class Backend(C.Structure):
        _fields_ = [('backend', S), ('device', I), ('threads', I)]
    signatures = {
        'last_error': (S, []), 'abi_version': (C.c_uint32, []),
        'registry_create': (I, [S, C.POINTER(P)]), 'registry_free': (None, [P]),
        'model_load': (I, [P, S, C.POINTER(ModelConfig), P, C.POINTER(P)]),
        'model_free': (None, [P]), 'model_supports': (I, [P, S, S]),
        'session_create': (I, [P, S, S, C.POINTER(Backend), P, C.POINTER(P)]),
        'session_free': (None, [P]), 'session_run': (I, [P, P, C.POINTER(P)]),
        'request_create': (P, []), 'request_free': (None, [P]),
        'request_set_text': (I, [P, S, S]), 'request_set_voice_id': (I, [P, S]),
        'request_set_option': (I, [P, S, S]),
        'result_free': (None, [P]),
        'result_audio': (I, [P, C.POINTER(FP), C.POINTER(Z), C.POINTER(I), C.POINTER(I)]),
        'result_named_audio_count': (Z, [P]),
        'result_named_audio': (I, [P, Z, C.POINTER(S), C.POINTER(FP), C.POINTER(Z), C.POINTER(I), C.POINTER(I)]),
        'stream_policy': (I, [P, C.POINTER(I), C.POINTER(I), C.POINTER(C.c_int64), C.POINTER(C.c_double)]),
        'stream_start': (I, [P, P]), 'stream_next_event': (I, [P, C.POINTER(P)]),
        'stream_finish': (I, [P, C.POINTER(P)]), 'stream_reset': (I, [P]),
        'event_free': (None, [P]), 'event_as_result': (P, [P]),
    }
    api = {}
    for name, (restype, argtypes) in signatures.items():
        fn = getattr(lib, 'audiocpp_' + name)
        fn.restype, fn.argtypes = restype, argtypes
        api[name] = fn
    if api['abi_version']() != 0x000100:
        raise RuntimeError('probe requires the pinned audio.cpp ABI 0.1.0')
    def check(status):
        if status:
            raise RuntimeError(api['last_error']().decode())
    def rss():
        return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    def audio(result, named=False):
        samples, frames, rate, channels = FP(), Z(), I(), I()
        if named:
            assert api['result_named_audio_count'](result) == 1
            name = S()
            check(api['result_named_audio'](result, 0, C.byref(name), C.byref(samples), C.byref(frames), C.byref(rate), C.byref(channels)))
        else:
            check(api['result_audio'](result, C.byref(samples), C.byref(frames), C.byref(rate), C.byref(channels)))
        assert frames.value > 0 and rate.value > 0 and channels.value == 1
        return C.string_at(samples, frames.value * channels.value * 4), frames.value / rate.value, rate.value
    registry, model = P(), P()
    check(api['registry_create'](None, C.byref(registry)))
    config = ModelConfig(b'supertonic', None, None, None)
    check(api['model_load'](registry, os.fsencode(args.model), C.byref(config), None, C.byref(model)))
    assert api['model_supports'](model, b'tts', b'streaming') == 1
    backend = Backend(args.backend.encode(), 0, 2)
    short = 'Your download is complete. You can open the file now.'
    long = ('The next meeting starts in fifteen minutes. Save your work before you leave. '
            'The weather will be clear this afternoon, with light winds from the west. ') * args.repetitions
    report = {'backend': args.backend, 'library': args.library, 'model': args.model,
              'pull_delay_ms': args.pull_delay_ms, 'streaming_first': args.streaming_first,
              'abi': api['abi_version'](), 'seed': 20260916, 'steps': 8, 'voice': 'M1', 'cases': []}
    try:
        for mode in (['streaming', 'offline'] if args.streaming_first else ['offline', 'streaming']):
            session = P()
            check(api['session_create'](model, b'tts', mode.encode(), C.byref(backend), None, C.byref(session)))
            try:
                if mode == 'streaming':
                    inp, out = I(), I()
                    check(api['stream_policy'](session, C.byref(inp), C.byref(out), None, None))
                    report['policy'] = {'input': inp.value, 'output': out.value}
                    assert (inp.value, out.value) == (0, 1)
                for label, text in [('warmup', short), ('short', short), ('long', long)]:
                    request = api['request_create']()
                    check(api['request_set_text'](request, text.encode(), b'en'))
                    check(api['request_set_voice_id'](request, b'M1'))
                    for key, value in [('seed', '20260916'), ('num_inference_steps', '8')]:
                        check(api['request_set_option'](request, key.encode(), value.encode()))
                    record = {'mode': mode, 'case': label, 'text_bytes': len(text.encode()), 'rss_before': rss()}
                    start = time.monotonic()
                    result = P()
                    try:
                        if mode == 'offline':
                            check(api['session_run'](session, request, C.byref(result)))
                            record['first_audio_s'] = time.monotonic() - start
                        else:
                            check(api['stream_start'](session, request))
                            chunks, digest, total_audio = [], hashlib.sha256(), 0
                            with (args.out / f'{label}-chunks.f32').open('wb') as file:
                                while True:
                                    event = P()
                                    check(api['stream_next_event'](session, C.byref(event)))
                                    available = time.monotonic() - start
                                    if not event:
                                        break
                                    try:
                                        pcm, duration, rate = audio(api['event_as_result'](event), True)
                                        file.write(pcm)
                                        digest.update(pcm)
                                        total_audio += duration
                                        chunks.append({'available_s': available, 'audio_s': duration, 'bytes': len(pcm), 'rss': rss()})
                                    finally:
                                        api['event_free'](event)
                                    del pcm
                                    if args.pull_delay_ms:
                                        before = rss()
                                        time.sleep(args.pull_delay_ms / 1000)
                                        chunks[-1]['rss_before_pause'] = before
                                        chunks[-1]['rss_after_pause'] = rss()
                            record.update(chunks=chunks, first_audio_s=chunks[0]['available_s'],
                                          chunks_sha256=digest.hexdigest(), chunks_audio_s=total_audio,
                                          rss_before_finish=rss())
                            check(api['stream_finish'](session, C.byref(result)))
                        record['total_s'] = time.monotonic() - start
                        pcm, duration, rate = audio(result)
                        record.update(audio_s=duration, sample_rate=rate, pcm_bytes=len(pcm), sha256=hashlib.sha256(pcm).hexdigest())
                        (args.out / f'{mode}-{label}.f32').write_bytes(pcm)
                        del pcm
                        if mode == 'streaming':
                            assert record['sha256'] == record['chunks_sha256'], 'finish output differs from incremental audio'
                        report['cases'].append(record)
                        (args.out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
                        print(json.dumps({k: record[k] for k in ['mode', 'case', 'first_audio_s', 'total_s', 'audio_s']}), flush=True)
                    finally:
                        api['result_free'](result)
                        api['request_free'](request)
            finally:
                api['session_free'](session)
        for label in ['short', 'long']:
            pair = [c for c in report['cases'] if c['case'] == label]
            assert pair[0]['sha256'] == pair[1]['sha256'], f'{label}: offline and streaming differ'
        report['offline_streaming_exact_match'] = True
        (args.out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        api['model_free'](model)
        api['registry_free'](registry)


if __name__ == '__main__':
    main()
