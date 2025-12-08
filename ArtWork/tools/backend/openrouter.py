from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import List, Optional, Sequence
import time

import requests


def _encode_image(path: Path) -> str:
    """Return a data URL for the supplied image."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _extract_text_content(content: object) -> str:
    """Normalize OpenRouter response message content to plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return ""


class VisualQAOpenRouter:
    """VQA adapter that calls OpenRouter vision-capable chat models."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        endpoint: str = "https://openrouter.ai/api/v1/chat/completions",
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        timeout: int = 120,
        max_tokens: int = 512,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.site_url = site_url
        self.site_name = site_name
        self.timeout = timeout
        self.max_tokens = max_tokens

    def _headers(self) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
        return headers

    def _payload(self, image_data_url: str, query: str) -> dict:
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
        }

    def _post_with_retry(self, payload: dict, retries: int = 3, backoff: float = 1.0) -> requests.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                # Retry on transient 5xx
                if response.status_code >= 500:
                    raise requests.HTTPError(f"Server error {response.status_code}", response=response)
                response.raise_for_status()
                return response
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < retries - 1:
                    time.sleep(backoff * (2 ** attempt))
                    continue
                raise last_exc

    def extract(self, image_paths: Sequence[str], query: str, batch_size: int = 1) -> List[str]:
        """Run the configured OpenRouter model on each image."""
        responses: List[str] = []
        for image_path in image_paths:
            payload = self._payload(_encode_image(Path(image_path)), query)
            try:
                print(f"[debug:openrouter] sending VQA request model={self.model} image={image_path}")
                response = self._post_with_retry(payload)
                print(f"[debug:openrouter] received status={response.status_code} for image={image_path}")
                data = response.json()
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message", {})
                responses.append(_extract_text_content(message.get("content")))
            except Exception as exc:  # noqa: BLE001
                print(f"[debug:openrouter] VQA request failed for image={image_path}: {exc}")
                responses.append(f"ERROR(OpenRouter VQA failed: {exc})")
        return responses
