#!/usr/bin/env python3
"""Measure pinned corpus inputs through file-only synthesis; never qualifies quality."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import signal
import shutil
import wave


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--reference', type=Path, help='Optional reference binary; alternate pair order each round')
    parser.add_argument('--cache-seed', type=Path, help='Copy prepared caches into the isolated cache before running')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, default=Path(__file__).resolve().parents[1] / 'benchmarks/model-defaults-cases.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1 or args.iterations < 1:
        parser.error('runs and iterations must be positive')
    binary, config, out = args.binary.resolve(), args.config.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    corpus = json.loads(args.corpus.read_text())
    environment = os.environ.copy()
    for key, part in [('XDG_CACHE_HOME', 'cache'), ('XDG_STATE_HOME', 'state'), ('XDG_RUNTIME_DIR', 'run')]:
        environment[key] = str(out / 'xdg' / part)
        Path(environment[key]).mkdir(parents=True, mode=0o700, exist_ok=True)
    if args.cache_seed:
        shutil.copytree(args.cache_seed, out / 'xdg/cache', dirs_exist_ok=True)
    variants = [('candidate', binary)]
    if args.reference:
        variants.insert(0, ('reference', args.reference.resolve()))
    results = {'schema_version': 1, 'status': 'measurement-only; listening and regression comparison required',
               'binary_sha256': digest(binary), 'variants': {name: digest(path) for name, path in variants}, 'config_sha256': digest(config),
               'corpus_sha256': digest(args.corpus), 'corpus_id': corpus['id'], 'measurements': []}
    for run in range(args.runs):
        for index, case in enumerate(corpus['cases']):
            for voice in corpus['voices']:
                for variant, executable in (variants if run % 2 == 0 else list(reversed(variants))):
                    directory = out / f'{variant}-run-{run}-case-{index}-voice-{voice}'
                    directory.mkdir()
                    command = [str(executable), '--config', str(config), 'benchmark', '--text', case['text'],
                               '--voice', voice, '--out-dir', str(directory / 'audio'), '--warmup', '1', '--iterations', str(args.iterations)]
                    with (directory / 'stderr.log').open('w') as stderr, (directory / 'report.json').open('w') as stdout:
                        process = subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
                        def timeout(_signal, _frame):
                            raise TimeoutError('synthesis measurement exceeded 600 seconds')
                        previous = signal.signal(signal.SIGALRM, timeout)
                        signal.alarm(600)
                        try:
                            _, status, usage = os.wait4(process.pid, 0)
                            process.returncode = os.waitstatus_to_exitcode(status)
                        except BaseException:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                            raise
                        finally:
                            signal.alarm(0)
                            signal.signal(signal.SIGALRM, previous)
                    if process.returncode:
                        raise subprocess.CalledProcessError(process.returncode, command)
                    (directory / 'max-rss-kib.txt').write_text(str(usage.ru_maxrss))
                    report = json.loads((directory / 'report.json').read_text())
                    if report['backend']['fallback_used'] or not report['backend'].get('placement_verified', False):
                        raise RuntimeError('fallback or unverified placement invalidates this measurement')
                    samples = []
                    for iteration in report['iterations']:
                        path = Path(iteration['output'])
                        with wave.open(str(path)) as audio:
                            if audio.getnframes() <= 0 or audio.getframerate() != iteration['sample_rate'] or audio.getnframes() != iteration['samples']:
                                raise RuntimeError(f'invalid or truncated WAV: {path}')
                            expected_bytes = audio.getnframes() * audio.getnchannels() * audio.getsampwidth()
                            if len(audio.readframes(audio.getnframes())) != expected_bytes:
                                raise RuntimeError(f'truncated WAV payload: {path}')
                        samples.append({'sha256': digest(path), 'samples': iteration['samples'], 'sample_rate': iteration['sample_rate']})
                    results['measurements'].append({'variant': variant, 'run': run, 'case': case['id'], 'voice': voice,
                        'backend': report['backend'], 'load_ms': report['model_load_milliseconds'],
                        'summary': report['summary'], 'max_process_rss_kib': int((directory / 'max-rss-kib.txt').read_text()), 'audio': samples})
                    (out / 'measurements.json').write_text(json.dumps(results, indent=2) + '\n')
                    print(f"{variant} run {run+1}: {case['id']} / {voice}", flush=True)


if __name__ == '__main__':
    main()
