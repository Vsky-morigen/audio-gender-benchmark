#!/usr/bin/env python3
"""Run both benchmark tasks and write one compact completion summary."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description='Run gender and main-language evaluations in sequence.')
    parser.add_argument('--gender-config', type=Path, default=ROOT / 'configs/qwen_api.json')
    parser.add_argument('--main-language-config', type=Path, default=ROOT / 'configs/main_language_api.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--limit', type=int, help='Smoke-test N samples from each task.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    name = f'combined_smoke_{args.limit}' if args.limit else 'combined'
    output = (args.output_dir or ROOT / 'results' / name).resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs = {'gender': args.gender_config, 'main_language': args.main_language_config}
    summary = {'tasks': {}, 'all_complete': True}
    for task, config in configs.items():
        task_dir = output / task
        command = [sys.executable, str(ROOT / 'scripts/run_benchmark.py'), '--task', task,
                   '--config', str(config), '--output-dir', str(task_dir)]
        if args.limit:
            command += ['--limit', str(args.limit)]
        if args.overwrite:
            command.append('--overwrite')
        print(f'Running {task}...', flush=True)
        code = subprocess.run(command, check=False).returncode
        metrics_path = task_dir / 'metrics.json'
        metrics = json.loads(metrics_path.read_text(encoding='utf-8')) if metrics_path.exists() else None
        summary['tasks'][task] = {
            'exit_code': code,
            'predictions': str(task_dir / 'predictions.csv'),
            'metrics': str(metrics_path) if metrics else None,
            'coverage': metrics['coverage'] if metrics else 0,
            'accuracy_on_completed': metrics['overall']['accuracy'] if metrics else None,
        }
        if code != 0 or metrics is None or not metrics['complete']:
            summary['all_complete'] = False
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"Combined summary: {output / 'summary.json'}")
    return 0 if all(x['exit_code'] == 0 for x in summary['tasks'].values()) else 1


if __name__ == '__main__':
    sys.exit(main())
