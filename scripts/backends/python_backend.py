from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from .base import InferenceBackend


class PythonBackend(InferenceBackend):
    """Load a user-provided Python model adapter once and reuse it for inference."""

    name = "python"

    def __init__(self, config: dict[str, object], root: Path):
        module_path = Path(str(config["module_path"]))
        module_path = module_path if module_path.is_absolute() else (root / module_path).resolve()
        spec = importlib.util.spec_from_file_location("benchmark_user_backend", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load Python backend: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        factory_name = str(config.get("factory", "create_backend"))
        factory = getattr(module, factory_name)
        self.instance = factory(config)
        if not hasattr(self.instance, "predict"):
            raise TypeError("custom backend object must define predict(audio_path, prompt, item)")
        self.model_id = str(config.get("model", getattr(self.instance, "model_id", "local-python-model")))

    def predict(self, audio_path: Path, prompt: str, item: dict[str, str], _timeout: float) -> str:
        value = self.instance.predict(str(audio_path), prompt, item)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            prediction = value.get("prediction") or value.get("predicted_gender")
            return str(prediction) if prediction else json.dumps(value, ensure_ascii=False)
        raise TypeError("custom backend predict() must return str or dict")
