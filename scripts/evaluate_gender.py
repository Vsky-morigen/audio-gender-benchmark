#!/usr/bin/env python3
"""Score a completed or partial gender/MCQ prediction file."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {'gender': ROOT / 'benchmark_metadata.csv', 'main_language': ROOT / 'data/main_language.csv'}
LABELS = {'gender': ('male', 'female'), 'main_language': ('A', 'B', 'C', 'D')}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def classification_metrics(rows: list[dict[str, str]], labels: tuple[str, ...]) -> dict:
    n = len(rows)
    per_class = {}
    for label in labels:
        tp = sum(x['gold'] == label and x['prediction'] == label for x in rows)
        fp = sum(x['gold'] != label and x['prediction'] == label for x in rows)
        fn = sum(x['gold'] == label and x['prediction'] != label for x in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {'support': tp + fn, 'precision': precision, 'recall': recall, 'f1': f1}
    return {
        'n': n,
        'accuracy': sum(x['gold'] == x['prediction'] for x in rows) / n if n else 0.0,
        'macro_f1': sum(v['f1'] for v in per_class.values()) / len(labels),
        'uar': sum(v['recall'] for v in per_class.values()) / len(labels),
        'per_class': per_class,
    }


def evaluate(task: str, manifest: Path, predictions: Path, allow_partial: bool = False) -> dict:
    gold_rows = read_rows(manifest)
    gold = {row['id']: row for row in gold_rows}
    if len(gold) != len(gold_rows):
        raise ValueError('duplicate IDs in manifest')
    predicted = {}
    for row in read_rows(predictions):
        item_id = row.get('id', '').strip()
        label = (row.get('predicted_label') or row.get('predicted_gender') or '').strip()
        if row.get('status', 'success') != 'success' or not label:
            continue
        if item_id in predicted:
            raise ValueError(f'duplicate prediction ID: {item_id}')
        if item_id not in gold:
            raise ValueError(f'unknown prediction ID: {item_id}')
        if label not in LABELS[task]:
            raise ValueError(f'invalid label for {item_id}: {label}')
        predicted[item_id] = label
    missing = set(gold) - set(predicted)
    if missing and not allow_partial:
        raise ValueError(f'missing {len(missing)} predictions; first IDs: {sorted(missing)[:5]}')
    if not predicted:
        raise ValueError('no successful predictions')
    gold_field = 'gender' if task == 'gender' else 'answer'
    joined = [{**gold[item_id], 'gold': gold[item_id][gold_field], 'prediction': label}
              for item_id, label in sorted(predicted.items())]
    breakdown_keys = ['audio_type', 'language']
    breakdown_keys += ['data_source', 'length_category', 'scene'] if task == 'gender' else ['category', 'source']
    breakdowns = {}
    for field in breakdown_keys:
        groups = defaultdict(list)
        for row in joined:
            groups[row[field]].append(row)
        breakdowns[field] = {key: classification_metrics(value, LABELS[task])
                             for key, value in sorted(groups.items())}
    groups = defaultdict(list)
    for row in joined:
        groups[f"{row['audio_type']}:{row['language']}"].append(row)
    breakdowns['audio_type_language'] = {key: classification_metrics(value, LABELS[task])
                                         for key, value in sorted(groups.items())}
    return {
        'task': task,
        'complete': not missing,
        'coverage': len(predicted) / len(gold),
        'successful': len(predicted),
        'expected': len(gold),
        'overall': classification_metrics(joined, LABELS[task]),
        'breakdowns': breakdowns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Score predictions against a benchmark manifest.')
    parser.add_argument('--task', choices=tuple(MANIFESTS), default='gender')
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    result = evaluate(args.task, args.manifest or MANIFESTS[args.task], args.predictions, args.allow_partial)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered)


if __name__ == '__main__':
    main()
