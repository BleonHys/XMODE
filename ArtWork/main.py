from __future__ import annotations

import ast
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent


def _ensure_pythonpath() -> None:
    """Ensure repository modules can be imported."""
    repo_str = str(REPO_ROOT)
    base_str = str(BASE_DIR)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    if base_str not in sys.path:
        insert_idx = 1 if sys.path and sys.path[0] == repo_str else 0
        sys.path.insert(insert_idx, base_str)

    tools_pkg = importlib.import_module("ArtWork.tools")
    sys.modules.setdefault("tools", tools_pkg)
    sys.modules.setdefault("tools.backend", importlib.import_module("ArtWork.tools.backend"))


_ensure_pythonpath()

from ArtWork.src.build_graph import graph_construction
from src.settings import get_settings


def _ensure_file(initial_data: Iterable[Any], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not file_path.exists():
        file_path.write_text(json.dumps(list(initial_data), indent=4), encoding="utf-8")


def _append_json(data: Any, file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} not found.")
    with file_path.open("r+", encoding="utf-8") as handle:
        existing = json.load(handle)
        if isinstance(data, dict):
            existing.append(data)
        elif isinstance(data, list):
            existing.extend(data)
        else:
            raise ValueError(f"Invalid data type: {type(data)}")
        handle.seek(0)
        json.dump(existing, handle, ensure_ascii=False, indent=4)


def _load_questions(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
        if not isinstance(payload, list):  # defensive check
            raise ValueError(f"Question file {path} is not a JSON list.")
        return [str(question) for question in payload]


def _configure_environment() -> None:
    settings = get_settings()

    # Ensure credentials are present before executing heavy pipelines.
    provider = settings.models.provider.lower()
    if provider == "openai":
        settings.require_env("OPENAI_API_KEY")
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    elif provider == "anthropic":
        settings.require_env("ANTHROPIC_API_KEY")
        os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
    else:
        raise ValueError(f"Unsupported LLM provider '{provider}'.")

    if settings.langchain_api_key:
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)

    tracing_flag = "true" if settings.langchain.tracing_v2 else "false"
    os.environ["LANGCHAIN_TRACING_V2"] = tracing_flag
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.artwork.langchain_project)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.artwork.langchain_project)

    vqa_provider = settings.artwork.vqa_provider.lower()
    if vqa_provider == "openrouter":
        settings.require_env("OPENROUTER_API_KEY")


def run() -> None:
    settings = get_settings()
    _configure_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    questions = _load_questions(settings.artwork.questions_file)

    provider = settings.models.provider.lower()
    model = (
        settings.models.default_chat_model
        if provider == "openai"
        else settings.models.anthropic_model
    )
    temperature = settings.models.default_temperature
    db_path = settings.artwork.db_path
    language = settings.artwork.language

    output_path = settings.artwork.output_dir
    output_file = output_path / f"ceasura_artWork-{language}-test.json"
    _ensure_file([], output_file)

    log_root = settings.artwork.log_dir
    log_root.mkdir(parents=True, exist_ok=True)

    chain = graph_construction(
        model=model,
        temperature=temperature,
        db_path=str(db_path),
        log_path=str(log_root),
        vqa_provider=settings.artwork.vqa_provider,
        settings=settings,
    )

    results: List[Dict[str, Any]] = []
    for idx, question in enumerate(questions):
        use_case_log_path = log_root / str(idx)
        use_case_log_path.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(use_case_log_path / "out.log")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.root.addHandler(file_handler)

        logger.debug("Question: %s", question)

        result: Dict[str, Any] = {"question": question, "id": idx}
        result_steps = []

        recursion_limit = int(os.environ.get("XMODE_RECURSION_LIMIT", "45"))
        config = {
            "recursion_limit": recursion_limit,
            "configurable": {"thread_id": f"{settings.artwork.thread_id}-{idx}"},
        }
        stream_output = []
        for step in chain.stream(question, config, stream_mode="values"):
            stream_output = step
            result_steps.append(step)

        to_json: List[Dict[str, Any]] = []
        prediction: Any
        try:
            for msg in stream_output:
                to_json.append(msg.to_json()["kwargs"])
            prediction = [ast.literal_eval(stream_output[-1].content)]
        except Exception:  # noqa: BLE001 - preserve legacy behaviour while restructuring
            prediction = stream_output[-1].content if stream_output else None

        result["xmode"] = to_json
        result["prediction"] = prediction

        results.append(result)
        with (use_case_log_path / "xmode.json").open("w", encoding="utf-8") as handle:
            json.dump([result], handle, ensure_ascii=False, indent=4)

        steps_log = "\n\n".join(
            f"Step {step_idx + 1}\n {step}" for step_idx, step in enumerate(result_steps)
        )
        with (use_case_log_path / "steps-values.log").open("w", encoding="utf-8") as handle:
            handle.write(steps_log)

        logging.root.removeHandler(file_handler)

    _append_json(results, output_file)


if __name__ == "__main__":
    run()
