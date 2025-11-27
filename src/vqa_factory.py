from __future__ import annotations

from ArtWork.tools.backend.image_qa import VisualQA as VisualQABlip
from ArtWork.tools.backend.openrouter import VisualQAOpenRouter
from ArtWork.tools.backend.qwen_ollama import VisualQAOllama
from src.settings import Settings


def build_vqa(provider: str, settings: Settings):
    provider = (provider or "blip").lower()
    if provider == "ollama":
        return VisualQAOllama(
            model=settings.artwork.ollama_model,
            endpoint=settings.artwork.ollama_endpoint,
        )
    if provider == "openrouter":
        return VisualQAOpenRouter(
            api_key=settings.openrouter_api_key,
            model=settings.artwork.openrouter_model,
            endpoint=settings.artwork.openrouter_endpoint,
            site_url=settings.openrouter_site_url,
            site_name=settings.openrouter_site_name,
            timeout=settings.artwork.openrouter_timeout,
            max_tokens=settings.artwork.openrouter_max_tokens,
        )
    return VisualQABlip()
