from __future__ import annotations

import ast
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from langchain_core.messages import HumanMessage

from src.build_graph import graph_construction_m3ae
from src.settings import get_settings


def _ensure_file(initial_data: Iterable[Any], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not file_path.exists():
        file_path.write_text(json.dumps(list(initial_data), indent=4), encoding="utf-8")


def _append_json(data: Dict[str, Any], file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} not found.")
    with file_path.open("r+", encoding="utf-8") as handle:
        existing = json.load(handle)
        if isinstance(existing, dict):
            existing = [existing]
        existing.append(data)
        handle.seek(0)
        json.dump(existing, handle, ensure_ascii=False, indent=4)


def _load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"Dataset {dataset_path} must be a JSON list of records.")
        return payload


def _configure_environment() -> None:
    settings = get_settings()
    settings.require_env("OPENAI_API_KEY")

    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    if settings.langchain_api_key:
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)

    os.environ["LANGCHAIN_TRACING_V2"] = "true" if settings.langchain.tracing_v2 else "false"
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.ehrxqa.langchain_project)

    if settings.ehrxqa.m3ae_endpoint:
        os.environ.setdefault("M3AE_ENDPOINT", settings.ehrxqa.m3ae_endpoint)


def run() -> None:
    settings = get_settings()
    _configure_environment()

    model = settings.models.default_chat_model
    chain = graph_construction_m3ae(model, db_path=settings.ehrxqa.db_path)

    dataset = _load_dataset(settings.ehrxqa.dataset_file)

    language = settings.ehrxqa.language
    output_file = settings.ehrxqa.output_dir / f"xmode-vqa-m3ae-{language}.json"
    _ensure_file([], output_file)

    for idx, entry in enumerate(dataset):
        config = {"configurable": {"thread_id": f"{settings.ehrxqa.thread_id}-{entry.get('id', idx)}"}}
        chain_input = {"question": entry.get("question")}
        payload = [HumanMessage(content=[chain_input])]

        executed_chain: List[Any] = []
        for step in chain.stream(payload, config, stream_mode="values"):
            executed_chain = step

        to_json: List[Dict[str, Any]] = []
        prediction: Any
        try:
            for msg in executed_chain:
                to_json.append(msg.to_json()["kwargs"])
            prediction = [ast.literal_eval(executed_chain[-1].content)]
        except Exception:
            prediction = executed_chain[-1].content if executed_chain else None

        entry["xmode"] = to_json
        entry["prediction"] = prediction

        _append_json(entry, output_file)
        time.sleep(1)


if __name__ == "__main__":
    run()
