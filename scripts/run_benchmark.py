#!/usr/bin/env python3
"""One inference runner for the gender and main-language benchmarks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from backends import create_backend
from audio_store import resolve_audio
from evaluate_gender import MANIFESTS, evaluate, read_rows

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['id', 'predicted_label', 'gold_label', 'correct', 'backend', 'model',
          'latency_seconds', 'raw_response', 'status', 'error']


def parse_label(text: str, task: str) -> str:
    cleaned = text.strip().strip('`*_ \r\n')
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            value = value.get('answer') or value.get('choice') or value.get('gender') or value.get('prediction')
            if value is not None:
                cleaned = str(value).strip()
    except json.JSONDecodeError:
        pass
    if task == 'gender':
        label = cleaned.lower().strip('.,:;!?\"\'()[] ')
        if label in {'male', 'female'}:
            return label
    else:
        match = re.fullmatch(r'(?:答案|answer|choice)?\s*[:：]?\s*[\(（]?([A-Da-dＡ-Ｄａ-ｄ])[\)）]?\s*[.。]?', cleaned, re.I)
        if match:
            label = match.group(1).upper()
            return chr(ord(label) - ord('Ａ') + ord('A')) if 'Ａ' <= label <= 'Ｄ' else label
    raise ValueError(f'cannot parse one {task} label from response: {text[:160]!r}')


def safe_audio_path(row: dict[str, str], task: str) -> Path:
    return resolve_audio(row, task)


def prompt_for(row: dict[str, str], task: str, template: str) -> str:
    if task == 'gender':
        return template
    options = '\n'.join(f'{letter}. {row[f"option_{letter}"]}' for letter in 'ABCD')
    # The written question and gold answer are deliberately not exposed to the model.
    return template.replace('{options}', options)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: x['id']))
    temp.replace(path)


def infer_one(row: dict[str, str], backend, task: str, template: str,
              timeout: float, retries: int) -> dict[str, str]:
    start = time.perf_counter()
    gold = row['gender'] if task == 'gender' else row['answer']
    result = {'id': row['id'], 'predicted_label': '', 'gold_label': gold, 'correct': '',
              'backend': backend.name, 'model': backend.model_id, 'latency_seconds': '',
              'raw_response': '', 'status': 'failed', 'error': ''}
    allowed_item = {'id': row['id'], 'audio_name': Path(row['audio_path']).name}
    if task == 'main_language':
        allowed_item.update({f'option_{letter}': row[f'option_{letter}'] for letter in 'ABCD'})
    for attempt in range(retries):
        try:
            raw = backend.predict(safe_audio_path(row, task), prompt_for(row, task, template), allowed_item, timeout)
            label = parse_label(raw, task)
            result.update(predicted_label=label, correct=str(label == gold).lower(),
                          raw_response=raw, status='success', error='')
            break
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 8))
    result['latency_seconds'] = f'{time.perf_counter() - start:.3f}'
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description='Run either speech benchmark with one backend interface.')
    parser.add_argument('--task', choices=tuple(MANIFESTS), default='gender')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('task', {}).get('name', 'gender') != args.task:
        parser.error('config task.name does not match --task')
    template = str(config['task']['prompt'])
    if not template or (args.task == 'main_language' and '{options}' not in template):
        parser.error('task.prompt must be nonempty and include {options} for main_language')
    manifest = (args.manifest or MANIFESTS[args.task]).resolve()
    all_rows = read_rows(manifest)
    expected = {'id', 'audio_path', 'gender'} if args.task == 'gender' else {
        'id', 'audio_path', 'answer', 'option_A', 'option_B', 'option_C', 'option_D'}
    if not all_rows or not expected.issubset(all_rows[0]):
        parser.error(f'manifest is empty or missing required fields: {sorted(expected)}')
    if len({r['id'] for r in all_rows}) != len(all_rows):
        parser.error('duplicate IDs in manifest')
    rows = all_rows[:args.limit] if args.limit else all_rows
    runner = config.get('runner', {})
    workers = args.workers or int(runner.get('workers', 1))
    retries = int(runner.get('retries', 1))
    timeout = float(runner.get('timeout_seconds', 120))
    if min(workers, retries, timeout) <= 0:
        parser.error('workers, retries and timeout_seconds must be positive')
    try:
        backend = create_backend(config['backend'], ROOT)
    except Exception as exc:
        parser.error(f'cannot initialize backend: {exc}')
    run_name = re.sub(r'[^A-Za-z0-9._-]+', '_', str(config.get('run_name') or backend.model_id))
    default_name = f'{run_name}_smoke_{args.limit}' if args.limit else run_name
    output_dir = (args.output_dir or ROOT / 'results' / args.task / default_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / 'predictions.csv'
    info_path = output_dir / 'run_info.json'
    fingerprint = hashlib.sha256(config_path.read_bytes() + manifest.read_bytes()).hexdigest()
    if predictions_path.exists() and not args.overwrite:
        if not info_path.exists():
            parser.error('existing predictions lack run_info.json; use --overwrite or a new output directory')
        previous = json.loads(info_path.read_text(encoding='utf-8'))
        if previous.get('fingerprint') != fingerprint:
            parser.error('config or manifest changed since this run; use --overwrite or a new output directory')
        completed = {r['id']: r for r in read_rows(predictions_path) if r.get('status') == 'success'}
    else:
        completed = {}
    info = {'task': args.task, 'backend': backend.name, 'model': backend.model_id,
            'fingerprint': fingerprint, 'config': str(config_path), 'manifest': str(manifest),
            'expected_items': len(all_rows), 'requested_items': len(rows)}
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    results = [completed[r['id']] for r in rows if r['id'] in completed]
    pending = [r for r in rows if r['id'] not in completed]
    print(f'{args.task}: {len(rows)} requested, {len(results)} resumed, {len(pending)} pending')
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(infer_one, r, backend, args.task, template, timeout, retries): r for r in pending}
        for future in as_completed(futures):
            result = future.result()
            with lock:
                results.append(result)
                write_csv(predictions_path, results)
            print(f"[{len(results):03d}/{len(rows):03d}] {result['id']}: {result['predicted_label'] or 'ERROR'}")
    if not pending:
        write_csv(predictions_path, results)
    successful = sum(r['status'] == 'success' for r in results)
    if successful:
        metrics = evaluate(args.task, manifest, predictions_path, allow_partial=successful < len(all_rows))
        (output_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f"Accuracy={metrics['overall']['accuracy']:.4f}; coverage={metrics['coverage']:.1%}")
    info.update(successful_items=successful, failed_items=len(results) - successful,
                finished_at=datetime.now(timezone.utc).isoformat())
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return 0 if successful == len(rows) else 1


if __name__ == '__main__':
    sys.exit(main())
