#!/usr/bin/env python3
"""Build a blinded first-round A/B listening pack from paired corpus reports."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import random
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('measurements', type=Path, help='Paired measurement directory')
    parser.add_argument('--corpus', type=Path, default=Path(__file__).resolve().parents[1] / 'benchmarks/model-defaults-cases.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260916)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text())
    measurements = json.loads((args.measurements / 'measurements.json').read_text())
    if hashlib.sha256(args.corpus.read_bytes()).hexdigest() != measurements['corpus_sha256']:
        raise ValueError('Corpus does not match the measurements')
    args.out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    pairs, key, rows = [], [], []
    for index, case in enumerate(corpus['cases']):
        for voice in corpus['voices']:
            pair = f'pair-{len(pairs)+1:02}'
            variants = ['reference', 'candidate']
            rng.shuffle(variants)
            controls = []
            for label, variant in zip('AB', variants):
                directory = args.measurements / f'{variant}-run-0-case-{index}-voice-{voice}'
                report = json.loads((directory / 'report.json').read_text())
                source = Path(report['iterations'][0]['output'])
                expected = next(m for m in measurements['measurements'] if m['variant'] == variant and m['run'] == 0 and m['case'] == case['id'] and m['voice'] == voice)
                if hashlib.sha256(source.read_bytes()).hexdigest() != expected['audio'][0]['sha256']:
                    raise ValueError(f'Recording changed since measurement: {source}')
                target = args.out / f'{pair}-{label}.wav'
                shutil.copyfile(source, target)
                key.append({'pair': pair, 'label': label, 'variant': variant, 'case': case['id'], 'voice': voice,
                            'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
                controls.append(f'<label>{label} <audio controls preload="none" src="{target.name}"></audio></label>')
                rows.append([pair,label,'','','','',''])
            pairs.append(f'<section><h2>{pair} · {html.escape(voice)}</h2><p>{html.escape(case["text"])}</p>'+''.join(controls)+'</section>')
    with (args.out/'ratings.csv').open('w', newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['pair','label','intelligibility_1_to_5','naturalness_1_to_5','pronunciation_1_to_5','voice_consistency_1_to_5','notes'])
        writer.writerows(rows)
    (args.out/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Omaspeak listening review</title>
<style>body{font:18px system-ui;max-width:850px;margin:40px auto;padding:0 24px;background:#faf9f6;color:#202423}section{padding:20px 0;border-top:1px solid #ccc}h2{font-size:20px}label{display:flex;align-items:center;gap:20px;margin:12px 0}audio{width:100%}</style>
<h1>Omaspeak listening review</h1><p>Listen to both clips for each pair using the same headphones and volume. Record ratings in <a href="ratings.csv">ratings.csv</a>: 1 = poor, 3 = acceptable, 5 = excellent. Note missing words, pronunciation errors, clipping and unexpected voice changes. Leave uncertain ratings blank. Playback starts only when you press play.</p><p>A/B assignment is randomized separately for each pair. Voices M1 and F1 are identified so consistency can be assessed across pairs. This small English corpus is a review aid, not a population listening study.</p>'''+''.join(pairs)+'</html>')
    # Keep unblinding material outside the review directory/zip.
    args.out.with_name(args.out.name+'-answer-key.json').write_text(json.dumps({'seed':args.seed,'clips':key},indent=2)+'\n')
    shutil.make_archive(str(args.out), 'zip', args.out)
    print(args.out/'index.html')


if __name__ == '__main__':
    main()
