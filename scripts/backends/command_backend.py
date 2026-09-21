from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .base import InferenceBackend


class CommandBackend(InferenceBackend):
    """Run one local executable per audio file and read its label from stdout."""

    name = "command"

    def __init__(self, config: dict[str, object], root: Path):
        command = config.get("command")
        if not isinstance(command, list) or not command:
            raise ValueError("command backend requires a non-empty command array")
        self.command = [str(value) for value in command]
        self.model_id = str(config.get("model", "local-command-model"))
        cwd = Path(str(config.get("cwd", ".")))
        self.cwd = cwd if cwd.is_absolute() else (root / cwd).resolve()
        self.extra_env = {str(k): str(v) for k, v in dict(config.get("env", {})).items()}

    def predict(self, audio_path: Path, prompt: str, item: dict[str, str], timeout: float) -> str:
        values = {"audio_path": str(audio_path), "id": item["id"], "prompt": prompt}
        command = [part.format_map(values) for part in self.command]
        completed = subprocess.run(
            command,
            cwd=self.cwd,
            env={**os.environ, **self.extra_env},
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"command exited {completed.returncode}: {completed.stderr.strip()[:500]}")
        if not completed.stdout.strip():
            raise ValueError("command produced no stdout")
        return completed.stdout.strip()
