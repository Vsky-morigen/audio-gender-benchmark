#!/usr/bin/env python3
"""Config-driven inference and evaluation for Audio Gender Benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from backends import create_backend


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmark_metadata.csv"
DEFAULT_CONFIG = ROOT / "configs" / "qwen_api.json"
OUTPUT_FIELDS = [
    "id", "audio_name", "predicted_gender", "gold_gender", "correct", "backend", "model",
    "latency_seconds", "raw_response", "status", "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model inference and calculate benchmark metrics.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="JSON experiment configuration.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, help="Default: results/<run_name>.")
    parser.add_argument("--workers", type=int, help="Override runner.workers in the config.")
    parser.add_argument("--limit", type=int, help="Only run the first N items for a smoke test.")
    parser.add_argument("--overwrite", action="store_true", help="Start again instead of resuming successful rows.")
    return parser.parse_args()


def parse_gender(text: str) -> str:
    cleaned = text.strip().lower().strip("`*_ .,:;!?\"'")
    if cleaned in {"male", "female"}:
        return cleaned
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            label = str(value.get("gender") or value.get("prediction") or "").lower()
            if label in {"male", "female"}:
                return label
    except json.JSONDecodeError:
        pass
    labels = set(re.findall(r"\b(?:male|female)\b", cleaned))
    if len(labels) == 1:
        return labels.pop()
    raise ValueError(f"could not parse one gender label from response: {text!r}")


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("config root must be a JSON object")
    return value


def load_manifest(path: Path, limit: int | None) -> tuple[list[dict[str, str]], int]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    required = {"id", "audio_name", "audio_path", "gender"}
    missing = required - set(all_rows[0] if all_rows else {})
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    return (all_rows[:limit] if limit is not None else all_rows), len(all_rows)


def safe_audio_path(row: dict[str, str]) -> Path:
    path = (ROOT / row["audio_path"]).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"audio path escapes benchmark directory: {row['audio_path']}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def make_result(
    row: dict[str, str], backend_name: str, model: str, started: float, prediction: str = "",
    raw: str = "", status: str = "failed", error: str = "",
) -> dict[str, str]:
    return {
        "id": row["id"],
        "audio_name": row["audio_name"],
        "predicted_gender": prediction,
        "gold_gender": row["gender"],
        "correct": str(prediction == row["gender"]).lower() if prediction else "",
        "backend": backend_name,
        "model": model,
        "latency_seconds": f"{time.perf_counter() - started:.3f}",
        "raw_response": raw,
        "status": status,
        "error": error,
    }


def infer_one(row, backend, prompt: str, timeout: float, retries: int) -> dict[str, str]:
    started = time.perf_counter()
    last_error = ""
    safe_item = {"id": row["id"], "audio_name": row["audio_name"]}
    for attempt in range(1, retries + 1):
        try:
            raw = backend.predict(safe_audio_path(row), prompt, safe_item, timeout)
            label = parse_gender(raw)
            return make_result(row, backend.name, backend.model_id, started, label, raw, "success", "")
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    return make_result(row, backend.name, backend.model_id, started, error=last_error)


def load_completed(path: Path, overwrite: bool, backend_name: str, model: str) -> dict[str, dict[str, str]]:
    if overwrite or not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["id"]: row for row in rows
        if row.get("status") == "success"
        and row.get("predicted_gender")
        and row.get("backend") == backend_name
        and row.get("model") == model
    }


def write_results(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["id"]))
    temp.replace(path)


def score(predictions: Path, manifest: Path, metrics: Path, partial: bool) -> dict[str, object]:
    command = [
        sys.executable, str(ROOT / "scripts" / "evaluate_gender.py"),
        "--predictions", str(predictions), "--manifest", str(manifest), "--output", str(metrics),
    ]
    if partial:
        command.append("--allow-partial")
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    config_path = args.config.resolve()
    config = load_json(config_path)
    backend_config = config.get("backend")
    runner_config = config.get("runner", {})
    task_config = config.get("task", {})
    if not isinstance(backend_config, dict) or not isinstance(runner_config, dict) or not isinstance(task_config, dict):
        raise ValueError("config sections backend, runner and task must be JSON objects")
    prompt = str(task_config.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("task.prompt is required")
    workers = args.workers if args.workers is not None else int(runner_config.get("workers", 1))
    retries = int(runner_config.get("retries", 1))
    timeout = float(runner_config.get("timeout_seconds", 120))
    if workers < 1 or retries < 1 or timeout <= 0:
        raise ValueError("workers, retries and timeout_seconds must be positive")

    try:
        backend = create_backend(backend_config, ROOT)
    except Exception as exc:
        print(f"ERROR: cannot initialize backend: {exc}", file=sys.stderr)
        return 2
    run_name = str(config.get("run_name", backend.model_id))
    output_dir = (args.output_dir or ROOT / "results" / run_name).resolve()
    manifest_path = args.manifest.resolve()
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"
    rows, manifest_total = load_manifest(manifest_path, args.limit)
    completed = load_completed(predictions_path, args.overwrite, backend.name, backend.model_id)
    results = [completed[row["id"]] for row in rows if row["id"] in completed]
    pending = [row for row in rows if row["id"] not in completed]

    print(f"Backend: {backend.name}")
    print(f"Model: {backend.model_id}")
    print(f"Items: {len(rows)} total, {len(results)} resumed, {len(pending)} pending")
    print(f"Output: {output_dir}")
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(infer_one, row, backend, prompt, timeout, retries): row for row in pending}
        for future in as_completed(futures):
            result = future.result()
            with lock:
                results.append(result)
                write_results(predictions_path, results)
            print(f"[{len(results):03d}/{len(rows):03d}] {result['id']} -> {result['predicted_gender'] or 'ERROR'}")

    successful = [row for row in results if row["status"] == "success"]
    failed = [row for row in results if row["status"] != "success"]
    evaluation = None
    if successful:
        evaluation = score(
            predictions_path, manifest_path, metrics_path,
            partial=len(successful) != manifest_total,
        )
        overall = evaluation["overall"]
        print(
            f"Metrics: Accuracy={overall['accuracy']:.4f}, "
            f"Macro-F1={overall['macro_f1']:.4f}, UAR={overall['uar']:.4f}"
        )
    run_info = {
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": str(config_path),
        "backend": backend.name,
        "model": backend.model_id,
        "requested_items": len(rows),
        "successful_items": len(successful),
        "failed_items": len(failed),
        "predictions_file": str(predictions_path),
        "metrics_file": str(metrics_path) if evaluation else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if failed:
        print(f"ERROR: {len(failed)} item(s) failed; rerun the same command to resume.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
