"""Resolve WAVs from the task archives into a verified local cache."""
from __future__ import annotations

import hashlib
import threading
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.audio_cache'
ARCHIVES = {
    'gender': ROOT / 'archives' / 'gender.zip',
    'main_language': ROOT / 'archives' / 'main_language.zip',
}
_LOCK = threading.Lock()


def resolve_audio(row: dict[str, str], task: str) -> Path:
    """Extract one WAV as needed and check it against the manifest digest."""
    relative = PurePosixPath(row['audio_path'])
    if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0] != 'audio':
        raise ValueError(f"unsafe audio path: {row['audio_path']}")
    if task not in ARCHIVES:
        raise ValueError(f'unknown task: {task}')
    output = CACHE.joinpath(*relative.parts)
    expected = row['sha256'].lower()
    if len(expected) != 64 or any(ch not in '0123456789abcdef' for ch in expected):
        raise ValueError(f"invalid SHA-256 for {row['id']}")
    with _LOCK:
        if output.is_file() and hashlib.sha256(output.read_bytes()).hexdigest() == expected:
            return output
        archive = ARCHIVES[task]
        if not archive.is_file():
            raise FileNotFoundError(archive)
        with zipfile.ZipFile(archive) as bundle:
            try:
                data = bundle.read(relative.as_posix())
            except KeyError as exc:
                raise FileNotFoundError(f'{relative} missing from {archive}') from exc
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch for {row['id']}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.tmp')
        temporary.write_bytes(data)
        temporary.replace(output)
    return output
