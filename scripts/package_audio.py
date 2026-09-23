#!/usr/bin/env python3
"""Rebuild task ZIPs from verified WAV files in audio/ or .audio_cache/audio/."""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path, PurePosixPath

from audio_store import ARCHIVES, CACHE, ROOT
from evaluate_gender import MANIFESTS, read_rows


def main() -> int:
    for task, manifest in MANIFESTS.items():
        rows = read_rows(manifest)
        destination = ARCHIVES[task]
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.tmp')
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
            for row in rows:
                name = PurePosixPath(row['audio_path'])
                if name.is_absolute() or '..' in name.parts or name.parts[0] != 'audio':
                    raise ValueError(f"unsafe audio path: {name}")
                source = ROOT.joinpath(*name.parts)
                if not source.is_file():
                    source = CACHE.joinpath(*name.parts)
                data = source.read_bytes()
                if hashlib.sha256(data).hexdigest() != row['sha256']:
                    raise ValueError(f"SHA-256 mismatch for {row['id']}")
                bundle.writestr(name.as_posix(), data)
        temporary.replace(destination)
        print(f'{task}: {len(rows)} WAVs -> {destination} ({destination.stat().st_size:,} bytes)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
