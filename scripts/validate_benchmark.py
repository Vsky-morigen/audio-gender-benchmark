#!/usr/bin/env python3
"""Validate the packaged 100-item benchmark."""

from __future__ import annotations

import csv
import hashlib
import re
import sys
import wave
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmark_metadata.csv"
TARGET_SOURCE = {"TTS": 50, "real_human": 50}
TARGET_LANGUAGE = {"zh": 50, "en": 50}
TARGET_GENDER = {"male": 50, "female": 50}
TARGET_CROSS = {("zh", "male"): 25, ("zh", "female"): 25, ("en", "male"): 25, ("en", "female"): 25}
TARGET_LENGTH = {"short": 20, "medium": 60, "long": 20}
TARGET_SCENE = {"daily_chat": 26, "news_sharing": 26, "emotional_support": 24, "task_planning": 24}
TEXT_RANGES = {"zh": {"short": (6, 10), "medium": (11, 18), "long": (19, 26)}, "en": {"short": (4, 6), "medium": (7, 12), "long": (13, 18)}}


def count_text(language: str, text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text)) if language == "zh" else len(re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text))


def main() -> int:
    with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    errors = []
    if len(rows) != 100:
        errors.append(f"expected 100 rows, found {len(rows)}")
    expected_ids = [f"PLB_{i:03d}" for i in range(1, 101)]
    if [r["id"] for r in rows] != expected_ids:
        errors.append("IDs must be exactly PLB_001..PLB_100 in order")
    checks = [
        (Counter(r["audio_type"] for r in rows), TARGET_SOURCE, "audio_type"),
        (Counter(r["language"] for r in rows), TARGET_LANGUAGE, "language"),
        (Counter(r["gender"] for r in rows), TARGET_GENDER, "gender"),
        (Counter((r["language"], r["gender"]) for r in rows), TARGET_CROSS, "language x gender"),
        (Counter(r["length_category"] for r in rows), TARGET_LENGTH, "length"),
        (Counter(r["scene"] for r in rows), TARGET_SCENE, "scene"),
    ]
    for actual, target, name in checks:
        if dict(actual) != target:
            errors.append(f"{name} mismatch: {dict(actual)}")
    hashes, texts = [], []
    for row in rows:
        path = (ROOT / row["audio_path"]).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError:
            errors.append(f"{row['id']}: audio path escapes benchmark root")
            continue
        if not path.exists():
            errors.append(f"{row['id']}: missing {row['audio_path']}")
            continue
        with wave.open(str(path), "rb") as w:
            duration = w.getnframes() / w.getframerate()
            rate, channels = w.getframerate(), w.getnchannels()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if abs(duration - float(row["duration_seconds"])) > 0.001:
            errors.append(f"{row['id']}: duration mismatch")
        if rate != 16000 or channels != 1:
            errors.append(f"{row['id']}: expected 16 kHz mono WAV, got {rate} Hz/{channels} channels")
        actual_count = count_text(row["language"], row["transcript"])
        lo, hi = TEXT_RANGES[row["language"]][row["length_category"]]
        if not lo <= actual_count <= hi:
            errors.append(f"{row['id']}: text length outside category")
        hashes.append(digest)
        texts.append(re.sub(r"[^\w\u4e00-\u9fff]+", "", row["transcript"].lower()))
    if len(set(hashes)) != 100:
        errors.append(f"expected 100 unique audio hashes, found {len(set(hashes))}")
    if len(set(texts)) != 100:
        errors.append(f"expected 100 unique normalized transcripts, found {len(set(texts))}")

    print("BENCHMARK VALIDATION", "PASS" if not errors else "FAIL")
    for actual, _, name in checks:
        print(f"{name}: {dict(actual)}")
    print(f"unique_audio_hashes: {len(set(hashes))}")
    duration_match = Counter()
    duration_ranges = {"short": (2.0, 3.0), "medium": (3.0, 5.0), "long": (5.0, 7.0)}
    for row in rows:
        lo, hi = duration_ranges[row["length_category"]]
        duration_match["yes" if lo <= float(row["duration_seconds"]) <= hi else "no"] += 1
    print(f"duration_target_match: {duration_match}")
    if errors:
        for error in errors:
            print("-", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
