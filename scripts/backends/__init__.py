from __future__ import annotations

from pathlib import Path

from .base import InferenceBackend
from .command_backend import CommandBackend
from .python_backend import PythonBackend
from .responses_api import ResponsesAPIBackend


BACKENDS = {
    "responses_api": ResponsesAPIBackend,
    "command": CommandBackend,
    "python": PythonBackend,
}


def create_backend(config: dict[str, object], root: Path) -> InferenceBackend:
    backend_type = str(config.get("type", ""))
    if backend_type not in BACKENDS:
        raise ValueError(f"unknown backend {backend_type!r}; available: {', '.join(sorted(BACKENDS))}")
    return BACKENDS[backend_type](config, root)
