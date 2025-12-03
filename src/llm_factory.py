from __future__ import annotations

from typing import Literal, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel
from langchain.chains.openai_functions import create_structured_output_runnable

from src.settings import get_settings

Provider = Literal["openai", "anthropic"]


def build_chat_model(
    provider: Provider | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    """Instantiate a chat model based on the configured provider."""
    settings = get_settings()
    provider = (provider or settings.models.provider).lower()
    temperature = (
        temperature
        if temperature is not None
        else settings.models.default_temperature
    )

    if provider == "anthropic":
        selected_model = model or settings.models.anthropic_model
        _ensure_anthropic_env()
        llm = ChatAnthropic(
            model=selected_model,
            temperature=temperature,
            api_key=settings.anthropic_api_key,
            max_tokens=8000,
        )
        llm._llm_provider = "anthropic"
        return llm

    if provider == "openai":
        selected_model = model or settings.models.default_chat_model
        _ensure_openai_env()
        llm = ChatOpenAI(
            model=selected_model,
            temperature=temperature,
            api_key=settings.openai_api_key,
        )
        llm._llm_provider = "openai"
        return llm

    raise ValueError(f"Unsupported LLM provider '{provider}'.")


def _ensure_openai_env() -> None:
    settings = get_settings()
    settings.openai_api_key  # raises if missing


def _ensure_anthropic_env() -> None:
    settings = get_settings()
    settings.anthropic_api_key  # raises if missing


def build_structured_runnable(
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
    schema: Type[BaseModel],
    **kwargs,
):
    # Encourage models to return full structured payloads by allowing more tokens
    llm = llm.bind(max_tokens=16384)
    provider = getattr(llm, "_llm_provider", "openai")
    if provider == "anthropic":
        return prompt | llm.with_structured_output(schema, max_output_tokens=16384)
    return create_structured_output_runnable(schema, llm, prompt, **kwargs)
