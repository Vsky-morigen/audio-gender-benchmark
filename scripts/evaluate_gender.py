#!/usr/bin/env python3
"""Evaluate gender predictions for Paralinguistic Gender Benchmark v1."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def metrics(rows: list[dict[str, str]]) -> dict[str, object]:
    labels = ("male", "female")
    correct = sum(r["gender"] == r["predicted_gender"] for r in rows)
    recalls, f1s, per_class = [], [], {}
    for label in labels:
        tp = sum(r["gender"] == label and r["predicted_gender"] == label for r in rows)
        fp = sum(r["gender"] != label and r["predicted_gender"] == label for r in rows)
        fn = sum(r["gender"] == label and r["predicted_gender"] != label for r in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
        per_class[label] = {"support": tp + fn, "precision": precision, "recall": recall, "f1": f1}
    return {
        "n": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
        "macro_f1": sum(f1s) / len(f1s),
        "uar": sum(recalls) / len(recalls),
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "benchmark_metadata.csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8-sig", newline="") as f:
        gold = {r["id"]: r for r in csv.DictReader(f)}
    with args.predictions.open(encoding="utf-8-sig", newline="") as f:
        prediction_rows = list(csv.DictReader(f))
    predictions = {}
    for row in prediction_rows:
        item_id = row.get("id", "").strip()
        label = row.get("predicted_gender", "").strip().lower()
        if not item_id or not label:
            continue
        if item_id in predictions:
            raise ValueError(f"Duplicate prediction ID: {item_id}")
        if item_id not in gold:
            raise ValueError(f"Unknown prediction ID: {item_id}")
        if label not in {"male", "female"}:
            raise ValueError(f"Invalid label for {item_id}: {label}")
        predictions[item_id] = label
    missing = sorted(set(gold) - set(predictions))
    if missing and not args.allow_partial:
        raise ValueError(f"Missing {len(missing)} predictions; first missing IDs: {missing[:5]}")
    joined = [{**gold[item_id], "predicted_gender": predictions[item_id]} for item_id in sorted(predictions)]
    if not joined:
        raise ValueError("No usable predictions found")

    result: dict[str, object] = {"overall": metrics(joined), "coverage": len(joined) / len(gold), "breakdowns": {}}
    breakdowns = {
        "audio_type": lambda r: r["audio_type"],
        "data_source": lambda r: r["data_source"],
        "language": lambda r: r["language"],
        "audio_type_language": lambda r: f"{r['audio_type']}:{r['language']}",
        "length_category": lambda r: r["length_category"],
        "scene": lambda r: r["scene"],
    }
    for name, key_fn in breakdowns.items():
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in joined:
            groups[key_fn(row)].append(row)
        result["breakdowns"][name] = {key: metrics(value) for key, value in sorted(groups.items())}

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
