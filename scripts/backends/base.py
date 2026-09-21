from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class InferenceBackend(ABC):
    """Common interface implemented by every API or local model backend."""

    name: str
    model_id: str

    @abstractmethod
    def predict(self, audio_path: Path, prompt: str, item: dict[str, str], timeout: float) -> str:
        """Return the model's raw textual response for one audio file."""
