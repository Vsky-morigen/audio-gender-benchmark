from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from .base import InferenceBackend


class ResponsesAPIBackend(InferenceBackend):
    """Audio input through an OpenAI-compatible Responses API."""

    name = "responses_api"

    def __init__(self, config: dict[str, object], _root: Path):
        self.model_id = str(config["model"])
        key_env = str(config.get("api_key_env", "DASHSCOPE_API_KEY"))
        self.api_key = os.getenv(key_env, "")
        if not self.api_key:
            raise ValueError(f"environment variable {key_env} is not set")
        base_url = str(config.get("base_url", "")).rstrip("/")
        base_url_env = str(config.get("base_url_env", ""))
        if base_url_env and os.getenv(base_url_env):
            base_url = os.environ[base_url_env].rstrip("/")
        if not base_url:
            raise ValueError("responses_api backend requires base_url")
        self.endpoint = base_url if base_url.endswith("/responses") else base_url + "/responses"
        self.extra_headers = {str(k): str(v) for k, v in dict(config.get("headers", {})).items()}

    @staticmethod
    def _data_uri(path: Path) -> tuple[str, str]:
        suffix = path.suffix.lower().lstrip(".") or "wav"
        mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4", "flac": "audio/flac"}.get(
            suffix, f"audio/{suffix}"
        )
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}", suffix

    @staticmethod
    def _output_text(response: dict[str, object]) -> str:
        if isinstance(response.get("output_text"), str):
            return str(response["output_text"]).strip()
        texts: list[str] = []
        for item in response.get("output", []):
            if not isinstance(item, dict):
                continue
            for part in item.get("content", []):
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"].strip())
        return "\n".join(filter(None, texts))

    def predict(self, audio_path: Path, prompt: str, _item: dict[str, str], timeout: float) -> str:
        data_uri, audio_format = self._data_uri(audio_path)
        payload = {
            "model": self.model_id,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_audio", "data": data_uri, "format": audio_format},
                ],
            }],
            "stream": False,
            "store": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body[:500]}") from exc
        text = self._output_text(parsed)
        if not text:
            raise ValueError("API response did not contain output text")
        return text
