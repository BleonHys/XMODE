# reserved for m3ae API client
import base64
import os
from pathlib import Path
from typing import Union

import requests

from src.settings import get_settings


_SETTINGS = get_settings()


def _resolve_endpoint(value: Union[str, None]) -> str:
    fallback = _SETTINGS.ehrxqa.m3ae_endpoint or "http://localhost:8080"
    endpoint = value or fallback
    if not endpoint.startswith("http"):
        endpoint = f"http://{endpoint}"
    return endpoint.rstrip("/")


def _resolve_files_dir() -> Path:
    env_override = os.environ.get("XMODE_FILES_DIR")
    if env_override:
        return Path(env_override)
    return _SETTINGS.base_dir / "files"


ENDPOINT = _resolve_endpoint(os.environ.get("M3AE_ENDPOINT"))
FILES_DIR = _resolve_files_dir()


def base64coode_from_image_id(image_id: str) -> str:
    res = [img for img in FILES_DIR.rglob(f"{image_id}.jpg")][0]
    base64code = "data:image/png;base64," + base64.b64encode(res.read_bytes()).decode("ascii")
    return base64code


def post_vqa_m3ae(question: str, image_id: str):
    url = f"{ENDPOINT}/predict"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "question": question,
        "image_url": base64coode_from_image_id(image_id),
    }
    response = requests.post(url, data=data, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()


def post_vqa_m3ae_with_url(question: str, image_url: str):
    url = f"{ENDPOINT}/predict"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "question": question,
        "image_url": image_url,
    }
    response = requests.post(url, data=data, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()
