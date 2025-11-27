from __future__ import annotations

import base64
from pathlib import Path
from typing import Sequence

import requests


def _encode_image(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


class VisualQAOllama:
    """VQA adapter for multimodale Ollama-Modelle wie qwen3-vl."""

    def __init__(
        self,
        model: str = "qwen3-vl:4b-instruct",
        endpoint: str = "http://localhost:11434/api/generate",
        timeout: int = 600,
    ) -> None:
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def extract(self, image_paths: Sequence[str], query: str, batch_size: int = 1):
        responses = []
        for image_path in image_paths:
            payload = {
                "model": self.model,
                "prompt": query,
                "stream": False,
                "images": [_encode_image(Path(image_path))],
            }
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            responses.append(data.get("response", "").strip())
        return responses
