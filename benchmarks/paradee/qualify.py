#!/usr/bin/env python3
"""Native, file-only Paradee quality corpus. Python is tooling, not an app dependency."""
import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import wave


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def waveform(path):
    with wave.open(str(path), 'rb') as reader:
        if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) != (1, 2, 24000):
            raise ValueError('expected mono PCM16 at 24 kHz')
        samples = array('h', reader.readframes(reader.getnframes()))
    if sys.byteorder != 'little':
        samples.byteswap()
    if not samples:
        raise ValueError('empty waveform')
    peak = max(abs(sample) for sample in samples) / 32768
    if peak == 0:
        raise ValueError('silent waveform')
    return {'samples': len(samples), 'audio_seconds': len(samples) / 24000,
            'peak': peak, 'rms': math.sqrt(sum((sample / 32768) ** 2 for sample in samples) / len(samples)),
            'clipped_samples': sum(abs(sample) >= 32767 for sample in samples),
            'wav_sha256': checksum(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, default=Path(__file__).with_name('corpus.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--espeak', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    config = json.loads(subprocess.check_output(
        [str(args.binary.resolve()), '--config', str(args.config.resolve()), 'config', 'get', '--json'],
        text=True, timeout=args.timeout))
    if (config['backend']['kind'], config['backend']['runtime'], config['backend']['device']) != ('paradee-openvino', 'openvino', 'cpu'):
        parser.error('requires a Paradee OpenVINO CPU config; never qualify a different provider silently')
    configured_frontend = config['backend'].get('options', {}).get('g2p_executable') or shutil.which('espeak-ng')
    if not configured_frontend or Path(configured_frontend).resolve() != args.espeak.resolve():
        parser.error('--espeak must match the configured native frontend')
    corpus = json.loads(args.corpus.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    sandbox = args.out / 'xdg'
    env = os.environ.copy()
    for name, subdirectory in [('XDG_RUNTIME_DIR', 'runtime'), ('XDG_CACHE_HOME', 'cache'),
                               ('XDG_STATE_HOME', 'state'), ('XDG_DATA_HOME', 'data'), ('XDG_CONFIG_HOME', 'config')]:
        directory = sandbox / subdirectory
        directory.mkdir(parents=True, mode=0o700)
        env[name] = str(directory.resolve())
    model_directory = Path(config['model']['directory'])
    graph = model_directory / config['model']['file']
    metadata = model_directory / config['model']['tts_json']
    report = {'schema': 1, 'status': 'native signal/frontend qualification; human listening pending',
              'platform': os.uname()._asdict() if hasattr(os.uname(), '_asdict') else list(os.uname()),
              'binary_sha256': checksum(args.binary), 'corpus_sha256': checksum(args.corpus),
              'model_sha256': checksum(graph), 'metadata_sha256': checksum(metadata),
              'backend': config['backend']['kind'], 'placement': 'CPU', 'threads': config['backend'].get('threads', 2),
              'openvino_library': config['backend'].get('openvino_library'), 'openvino_plugins': config['backend'].get('openvino_plugins'),
              'espeak_version': subprocess.check_output([str(args.espeak.resolve()), '--version'], text=True).strip(),
              'cases': []}
    if report['model_sha256'] != '77b8bb28caf3dddda0cc61d16b700febc187ee6336c695d7865b1e77e452ee11':
        parser.error('model differs from pinned official Paradee FP32 graph')
    if report['metadata_sha256'] != 'f24046974a3a8c747affefb45c7c504263a99d5081787908b16abe8f5ac94fcd':
        parser.error('metadata differs from pinned official Paradee vocabulary')
    for case in corpus['cases']:
        path = args.out / (case['id'] + '.wav')
        started = time.monotonic()
        try:
            result = subprocess.run([str(args.binary.resolve()), '--config', str(args.config.resolve()), 'say', '--no-play',
                                     '--voice', 'af_heart', '--speed', '1', '--out', str(path.resolve())],
                                    input=case['text'], text=True, capture_output=True, timeout=args.timeout, env=env)
        except subprocess.TimeoutExpired:
            result = subprocess.CompletedProcess([], 124, '', 'native synthesis timed out')
        row = {'id': case['id'], 'category': case['category'], 'text': case['text'],
               'total_seconds': time.monotonic() - started, 'wav': path.name,
               'exit_code': result.returncode, 'frontend_ipa': []}
        for segment in case.get('expected_segments', []):
            ipa = subprocess.check_output([str(args.espeak.resolve()), '-q', '--ipa', '-v', 'en-us', '--stdin'],
                                           input=segment['text'], text=True, timeout=5)
            row['frontend_ipa'].append({'normalized_text': segment['text'], 'ipa': ipa.strip(),
                                        'restored_punctuation': segment['punctuation']})
        try:
            if result.returncode:
                raise ValueError(result.stderr.strip())
            row.update(waveform(path))
            row['valid_audio'] = True
        except (ValueError, wave.Error, OSError) as error:
            row['valid_audio'] = False
            row['error'] = str(error)
        report['cases'].append(row)
        print(case['id'], 'ok' if row['valid_audio'] else 'FAILED', flush=True)
        (args.out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return 0 if all(case['valid_audio'] for case in report['cases']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
