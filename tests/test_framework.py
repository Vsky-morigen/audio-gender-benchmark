import csv
import json
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from evaluate_gender import evaluate, read_rows  # noqa: E402
from run_benchmark import parse_label, prompt_for  # noqa: E402


class FrameworkTests(unittest.TestCase):
    def test_strict_answer_parsing(self):
        self.assertEqual(parse_label('答案：Ｃ', 'main_language'), 'C')
        self.assertEqual(parse_label('{"answer":"b"}', 'main_language'), 'B')
        self.assertEqual(parse_label('female', 'gender'), 'female')
        with self.assertRaises(ValueError):
            parse_label('A or B', 'main_language')

    def test_model_prompt_excludes_question_and_gold(self):
        row = read_rows(ROOT / 'data/main_language.csv')[0]
        prompt = prompt_for(row, 'main_language', 'Listen to the question.\n{options}')
        self.assertNotIn(row['question'], prompt)
        self.assertIn('A. ' + row['option_A'], prompt)
        self.assertIn('D. ' + row['option_D'], prompt)
        self.assertEqual(prompt.count('\n'), 4)

    def test_partial_scoring_and_coverage(self):
        manifest = ROOT / 'data/main_language.csv'
        rows = read_rows(manifest)
        predictions = ROOT / 'tests' / f'.predictions-{uuid.uuid4().hex}.csv'
        try:
            with predictions.open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['id', 'predicted_label', 'status'])
                writer.writeheader()
                writer.writerow({'id': rows[0]['id'], 'predicted_label': rows[0]['answer'], 'status': 'success'})
                writer.writerow({'id': rows[1]['id'], 'predicted_label': rows[1]['answer'], 'status': 'success'})
            result = evaluate('main_language', manifest, predictions, allow_partial=True)
            self.assertFalse(result['complete'])
            self.assertEqual(result['successful'], 2)
            self.assertEqual(result['expected'], 100)
            self.assertEqual(result['overall']['accuracy'], 1.0)
            with self.assertRaises(ValueError):
                evaluate('main_language', manifest, predictions, allow_partial=False)
        finally:
            predictions.unlink(missing_ok=True)

    def test_runner_command_backend_and_resume(self):
        root = (ROOT / 'tests').resolve()
        work = (root / f'.runner-{uuid.uuid4().hex}').resolve()
        self.assertTrue(work.is_relative_to(root))
        work.mkdir()
        try:
            for task, label in [('gender', 'male'), ('main_language', 'A')]:
                config = {
                    'run_name': 'test',
                    'task': {'name': task, 'prompt': 'Return one label.\n{options}' if task == 'main_language' else 'Return one label.'},
                    'backend': {'type': 'command', 'model': 'mock', 'command': [sys.executable, '-c', f"print('{label}')"]},
                    'runner': {'workers': 1, 'retries': 1, 'timeout_seconds': 10},
                }
                config_path = work / f'{task}.json'
                config_path.write_text(json.dumps(config), encoding='utf-8')
                output = work / task
                command = [sys.executable, str(ROOT / 'scripts/run_benchmark.py'), '--task', task,
                           '--config', str(config_path), '--output-dir', str(output), '--limit', '2']
                for _ in range(2):
                    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                predictions = read_rows(output / 'predictions.csv')
                metrics = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
                self.assertEqual(len(predictions), 2)
                self.assertEqual(metrics['successful'], 2)
                self.assertFalse(metrics['complete'])
            combined = work / 'combined'
            command = [sys.executable, str(ROOT / 'scripts/run_all.py'),
                       '--gender-config', str(work / 'gender.json'),
                       '--main-language-config', str(work / 'main_language.json'),
                       '--output-dir', str(combined), '--limit', '2']
            result = subprocess.run(command, capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((combined / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(set(summary['tasks']), {'gender', 'main_language'})
            self.assertFalse(summary['all_complete'])
        finally:
            self.assertTrue(work.is_relative_to(root))
            shutil.rmtree(work)


if __name__ == '__main__':
    unittest.main()
