#!/usr/bin/env python3
"""Check the packaged 200 audio files and both gold manifests."""
from __future__ import annotations

import hashlib
import math
import re
import struct
import sys
import wave
import zipfile
from collections import Counter
from pathlib import Path

from evaluate_gender import MANIFESTS, read_rows
from audio_store import resolve_audio

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    errors = []
    hashes = []
    all_ids = []
    for task, manifest in MANIFESTS.items():
        rows = read_rows(manifest)
        if len(rows) != 100:
            errors.append(f'{task}: expected 100 rows, found {len(rows)}')
        if len({r['id'] for r in rows}) != len(rows):
            errors.append(f'{task}: duplicate IDs')
        if Counter(r['language'] for r in rows) != {'zh': 50, 'en': 50}:
            errors.append(f'{task}: language quota mismatch')
        types = Counter(r['audio_type'] for r in rows)
        expected = {'TTS': 50, 'real_human': 50} if task == 'gender' else {'TTS': 50, 'human': 50}
        if types != expected:
            errors.append(f'{task}: audio type quota mismatch: {types}')
        if task == 'gender' and Counter(r['gender'] for r in rows) != {'male': 50, 'female': 50}:
            errors.append('gender: gold label quota mismatch')
        all_ids.extend(r['id'] for r in rows)
        questions = set()
        for row in rows:
            item = row['id']
            try:
                audio = resolve_audio(row, task)
            except (FileNotFoundError, ValueError, OSError, zipfile.BadZipFile) as exc:
                errors.append(f'{item}: {exc}')
                continue
            try:
                with wave.open(str(audio), 'rb') as wav:
                    rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
                    frames = wav.getnframes()
                    duration = frames / rate
                    if task == 'main_language' and row['audio_type'] == 'TTS':
                        pcm = wav.readframes(frames)
                        samples = struct.unpack('<' + 'h' * frames, pcm)
                        rms = math.sqrt(sum(x*x for x in samples) / frames) / 32768
                        if rms <= 0.003:
                            errors.append(f'{item}: near-silent TTS')
                if (rate, channels, width) != (16000, 1, 2):
                    errors.append(f'{item}: expected 16 kHz mono PCM16, got {rate}/{channels}/{width}')
                if abs(duration - float(row['duration_seconds'])) > 0.002:
                    errors.append(f'{item}: duration mismatch')
            except (wave.Error, ValueError, ZeroDivisionError) as exc:
                errors.append(f'{item}: invalid WAV: {exc}')
                continue
            digest = hashlib.sha256(audio.read_bytes()).hexdigest()
            hashes.append(digest)
            if digest != row['sha256']:
                errors.append(f'{item}: SHA-256 mismatch')
            if task == 'main_language':
                answer = row['answer']
                options = [row[f'option_{letter}'] for letter in 'ABCD']
                if answer not in set('ABCD') or len(set(options)) != 4 or row['answer_text'] != row.get(f'option_{answer}'):
                    errors.append(f'{item}: invalid single-choice answer')
                norm = re.sub(r'\W+', '', row['question'].casefold())
                if norm in questions:
                    errors.append(f'{item}: duplicate question')
                questions.add(norm)
    if len(set(all_ids)) != 200:
        errors.append(f'IDs collide across tasks: {len(set(all_ids))} distinct')
    if len(set(hashes)) != 200:
        errors.append(f'audio hashes collide: {len(set(hashes))} distinct')
    print(f"PACKAGE VALIDATION {'PASS' if not errors else 'FAIL'}: {len(all_ids)} rows, {len(set(hashes))} unique WAVs")
    for error in errors:
        print('-', error)
    return 0 if not errors else 1


if __name__ == '__main__':
    sys.exit(main())
