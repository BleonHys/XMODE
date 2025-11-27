#!/usr/bin/env python3
"""Batch runner for the Artwork pipeline with LangSmith exports."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from langsmith_utils import extract_and_save_all_child_runs_by_project
except Exception:  # pragma: no cover - defensive import guard
    extract_and_save_all_child_runs_by_project = None


@dataclass
class ModelSpec:
    provider: str
    model: str

    @property
    def slug(self) -> str:
        return _slugify(f"{self.provider}-{self.model}")


def _slugify(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in value.lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")


def _parse_model_spec(raw: str) -> ModelSpec:
    parts = raw.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Model spec '{raw}' must use the format '<provider>:<model_name>'."
        )
    provider, model = parts[0].strip(), parts[1].strip()
    if not provider or not model:
        raise argparse.ArgumentTypeError(f"Invalid model spec '{raw}'.")
    return ModelSpec(provider=provider, model=model)


def _load_base_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_run_config(
    base_cfg: Dict[str, Any],
    model: ModelSpec,
    run_label: str,
    output_root: Path,
    log_root: Path,
    project_prefix: str,
    thread_prefix: str,
    questions_file: Path | None = None,
    vqa_provider: str | None = None,
    openrouter_model: str | None = None,
    project_suffix: str | None = None,
    vqa_slug: str | None = None,
) -> Dict[str, Any]:
    cfg = deepcopy(base_cfg)
    cfg.setdefault("models", {})
    cfg.setdefault("langchain", {})
    cfg.setdefault("artwork", {})

    cfg["models"]["provider"] = model.provider
    if model.provider == "openai":
        cfg["models"]["default_chat_model"] = model.model
    elif model.provider == "anthropic":
        cfg["models"]["anthropic_model"] = model.model
    else:
        cfg["models"]["default_chat_model"] = model.model

    safe_model = model.slug
    suffix = f"-{project_suffix}" if project_suffix else ""
    vqa_part = f"-{vqa_slug}" if vqa_slug else ""
    project_name = f"{project_prefix}-{safe_model}{vqa_part}{suffix}"
    cfg["langchain"]["project"] = project_name
    cfg["langchain"]["tracing_v2"] = True

    output_dir_template = Path(output_root) / "{language}" / safe_model / run_label
    cfg["artwork"]["output_dir_template"] = str(output_dir_template)
    cfg["artwork"]["log_dir"] = str(Path(log_root) / safe_model / run_label)
    cfg["artwork"]["langchain_project"] = project_name
    cfg["artwork"]["thread_id"] = f"{thread_prefix}-{safe_model}-{run_label}"
    if questions_file:
        cfg["artwork"]["questions_file"] = str(questions_file)
    if vqa_provider:
        cfg["artwork"]["vqa_provider"] = vqa_provider
    if openrouter_model:
        cfg["artwork"]["openrouter_model"] = openrouter_model
    return cfg


def _write_config(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _run_single(config_path: Path, project_name: str, env_path: Path | None, dry_run: bool) -> None:
    env = os.environ.copy()
    env["XMODE_CONFIG_PATH"] = str(config_path)
    if env_path:
        env["XMODE_ENV_PATH"] = str(env_path)
    env["LANGCHAIN_PROJECT"] = project_name
    env["LANGSMITH_PROJECT"] = project_name

    cmd = [sys.executable, "ArtWork/main.py"]
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)} (XMODE_CONFIG_PATH={config_path})")
        return

    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)


def _export_langsmith(projects: Iterable[str], export_dir: Path) -> None:
    if extract_and_save_all_child_runs_by_project is None:
        print("LangSmith export helper not available. Skipping export.")
        return
    export_dir.mkdir(parents=True, exist_ok=True)
    for project in projects:
        print(f"Exporting LangSmith runs for project '{project}' into '{export_dir}'.")
        extract_and_save_all_child_runs_by_project(project, data_path=str(export_dir))


def _expand_models(raw_models: List[str], base_cfg: Dict[str, Any]) -> List[ModelSpec]:
    if not raw_models:
        provider = base_cfg.get("models", {}).get("provider", "openai")
        default_model = base_cfg.get("models", {}).get("default_chat_model", "gpt-4o")
        raw_models = [f"{provider}:{default_model}"]

    flattened: List[str] = []
    for raw in raw_models:
        flattened.extend([item for item in raw.split(",") if item.strip()])
    return [_parse_model_spec(item) for item in flattened]


def _prepare_question_slice(
    source: Path, dest_dir: Path, limit: int | None, offset: int
) -> Path:
    """Return a questions file, optionally sliced to offset/limit."""
    with source.open("r", encoding="utf-8") as handle:
        questions = json.load(handle)
    if not isinstance(questions, list):
        raise ValueError(f"Questions file {source} must contain a JSON list.")
    if offset or limit is not None:
        start = max(offset, 0)
        end = start + limit if limit is not None else None
        questions = questions[start:end]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{source.stem}-offset{offset}-limit{limit or 'all'}.json"
    dest.write_text(json.dumps(questions, indent=2), encoding="utf-8")
    return dest


def _write_run_log(meta: Dict[str, Any], log_root: Path) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    project_dir = log_root / meta.get("project_name", "unknown_project")
    project_dir.mkdir(parents=True, exist_ok=True)
    log_path = project_dir / f"{meta.get('run_label', 'run')}.json"
    log_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _append_runs_csv(entries: List[Dict[str, Any]], log_root: Path) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    csv_path = log_root / "runs.csv"
    headers = [
        "project_name",
        "run_label",
        "model_provider",
        "model_name",
        "vqa_provider",
        "vqa_model",
        "status",
        "config_path",
        "output_dir_template",
        "log_dir",
    ]
    import csv

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        for meta in entries:
            writer.writerow([meta.get(h, "") for h in headers])


def _ensure_langsmith_key(env_path: Path | None = None) -> bool:
    """Load .env if present and return True if a LangSmith key is set."""
    key_names = ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY")
    # Choose .env: use provided env_path if present, else default to repo root .env
    candidate_envs = []
    if env_path:
        candidate_envs.append(env_path)
    candidate_envs.append(REPO_ROOT / ".env")
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        load_dotenv = None  # type: ignore
    if load_dotenv:
        for env_file in candidate_envs:
            if env_file and env_file.exists():
                load_dotenv(dotenv_path=env_file, override=False)
    return any(os.environ.get(name) for name in key_names)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Artwork pipeline multiple times across different model backends."
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per model (e.g. 100).")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model specs as '<provider>:<model>'. Separate multiple entries with spaces or commas.",
    )
    parser.add_argument(
        "--model",
        help="Single model spec shorthand (same format as --models but for one entry).",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=REPO_ROOT / "config" / "defaults.json",
        help="Base JSON config to clone for each run.",
    )
    parser.add_argument(
        "--env-path",
        type=Path,
        default=os.environ.get("XMODE_ENV_PATH"),
        help="Optional .env path to pass through via XMODE_ENV_PATH.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "ArtWork" / "experiments" / "batch",
        help="Root folder for batch outputs (per-model/run folders are created inside).",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=REPO_ROOT / "ArtWork" / "experiments" / "log-batch",
        help="Root folder for batch logs.",
    )
    parser.add_argument(
        "--project-prefix",
        default="XMODE-ArtWork-batch",
        help="Prefix for LangSmith project names (project = prefix-modelslug).",
    )
    parser.add_argument(
        "--thread-prefix",
        default="artwork-batch",
        help="Prefix for thread ids passed to the pipeline.",
    )
    parser.add_argument(
        "--batch-id",
        default=datetime.utcnow().strftime("%Y%m%d-%H%M%S"),
        help="Identifier appended to each run folder (default: UTC timestamp).",
    )
    parser.add_argument(
        "--project-suffix",
        help="Optional suffix appended to LangSmith project names (e.g., timestamp or hash).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going even if a run fails. By default the batch stops on the first error.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export LangSmith traces for all batch projects after the runs complete.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=REPO_ROOT / "ArtWork" / "experiments" / "langsmith-exports",
        help="Where to write LangSmith export JSON files.",
    )
    parser.add_argument(
        "--local-run-log-dir",
        type=Path,
        default=REPO_ROOT / "ArtWork" / "experiments" / "langsmith_runs",
        help="Where to write per-run local log JSON files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only use the first N questions (after offset).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N questions before applying limit.",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        help="Override the questions file; defaults to artwork.questions_file in the base config.",
    )
    parser.add_argument(
        "--questions",
        help="Shorthand slice as 'limit:offset' or just 'limit'. Example: '10:0' = first 10 starting at 0.",
    )
    parser.add_argument(
        "--vqa-provider",
        choices=["openrouter", "ollama", "blip", "qwen"],
        help="Override artwork.vqa_provider for this batch. Use 'qwen' as a shorthand for OpenRouter Qwen vision.",
    )
    parser.add_argument(
        "--vqa",
        help="Shorthand for VQA backend; use 'qwen' to select OpenRouter Qwen vision, or pass openrouter/ollama/blip.",
    )
    parser.add_argument(
        "--openrouter-model",
        help="Override artwork.openrouter_model (used when vqa-provider=openrouter).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands/configs without executing the pipeline.",
    )
    args = parser.parse_args()

    # Shorthands/overrides
    if args.model and args.models:
        raise argparse.ArgumentTypeError("Use either --model or --models, not both.")

    raw_models = args.models or ([args.model] if args.model else [])

    # Parse questions shorthand (limit:offset or limit)
    if args.questions:
        parts = args.questions.split(":")
        try:
            if len(parts) == 1:
                args.limit = int(parts[0])
            elif len(parts) == 2:
                args.limit = int(parts[0]) if parts[0] else None
                args.offset = int(parts[1]) if parts[1] else 0
            else:
                raise ValueError
        except ValueError:
            raise argparse.ArgumentTypeError("--questions must be 'limit' or 'limit:offset', e.g. '10:0'")

    # VQA shorthand
    if args.vqa:
        vqa_lower = args.vqa.lower()
        if vqa_lower == "qwen":
            args.vqa_provider = "openrouter"
            args.openrouter_model = args.openrouter_model or "qwen/qwen3-vl-8b-instruct"
        elif vqa_lower in {"openrouter", "ollama", "blip", "qwen"}:
            # If user explicitly passes provider names, honor them
            args.vqa_provider = "openrouter" if vqa_lower == "qwen" else vqa_lower
        else:
            raise argparse.ArgumentTypeError("--vqa must be one of: qwen, openrouter, ollama, blip")

    base_cfg = _load_base_config(args.base_config)
    models = _expand_models(raw_models, base_cfg)
    # Resolve VQA provider/model for naming
    vqa_provider = args.vqa_provider or base_cfg.get("artwork", {}).get("vqa_provider", "blip")
    if vqa_provider == "openrouter":
        vqa_model = args.openrouter_model or base_cfg.get("artwork", {}).get("openrouter_model", "qwen/qwen3-vl-8b-instruct")
    elif vqa_provider == "ollama":
        vqa_model = base_cfg.get("artwork", {}).get("ollama_model", "qwen3-vl:4b-instruct")
    else:
        vqa_model = "blip-vqa-base"
    vqa_slug = _slugify(f"{vqa_provider}-{vqa_model}")

    tmp_config_root = Path(args.output_root) / "tmp-configs"
    # Determine questions file (optionally sliced)
    default_questions = Path(base_cfg.get("artwork", {}).get("questions_file", "ArtWork/questions_en.json"))
    questions_source = args.questions_file or default_questions
    if not questions_source.is_absolute():
        questions_source = REPO_ROOT / questions_source
    sliced_questions = _prepare_question_slice(
        questions_source,
        dest_dir=Path(args.output_root) / "tmp-questions",
        limit=args.limit,
        offset=args.offset,
    )

    projects: Set[str] = set()
    failures: List[Tuple[ModelSpec, int, Exception]] = []
    run_meta_entries: List[Dict[str, Any]] = []

    total = len(models) * args.runs
    print(f"Planned runs: {total} ({args.runs} per model across {len(models)} model(s)).")

    for model in models:
        for run_idx in range(1, args.runs + 1):
            run_label = f"{args.batch_id}-r{run_idx:03d}"
            cfg = _build_run_config(
                base_cfg=base_cfg,
                model=model,
                run_label=run_label,
                output_root=args.output_root,
                log_root=args.log_root,
                project_prefix=args.project_prefix,
                thread_prefix=args.thread_prefix,
                questions_file=sliced_questions,
                vqa_provider=args.vqa_provider,
                openrouter_model=args.openrouter_model,
                project_suffix=args.project_suffix or args.batch_id,
                vqa_slug=vqa_slug,
            )
            project_name = cfg["artwork"]["langchain_project"]
            projects.add(project_name)

            config_path = tmp_config_root / model.slug / f"{run_label}.json"
            _write_config(cfg, config_path)

            print(
                f"[{model.provider}:{model.model}] run {run_idx}/{args.runs} "
                f"-> output_dir_template={cfg['artwork']['output_dir_template']}"
            )
            try:
                _run_single(config_path, project_name, args.env_path, args.dry_run)
                run_meta = {
                    "project_name": project_name,
                    "run_label": run_label,
                    "model_provider": model.provider,
                    "model_name": model.model,
                    "vqa_provider": vqa_provider,
                    "vqa_model": vqa_model,
                    "config_path": str(config_path),
                    "output_dir_template": cfg["artwork"]["output_dir_template"],
                    "log_dir": cfg["artwork"]["log_dir"],
                    "status": "success",
                }
            except Exception as exc:  # noqa: BLE001 - intentional catch to allow continue flag
                failures.append((model, run_idx, exc))
                print(f"Run failed: {model.provider}:{model.model} #{run_idx}: {exc}")
                run_meta = {
                    "project_name": project_name,
                    "run_label": run_label,
                    "model_provider": model.provider,
                    "model_name": model.model,
                    "vqa_provider": vqa_provider,
                    "vqa_model": vqa_model,
                    "config_path": str(config_path),
                    "output_dir_template": cfg["artwork"]["output_dir_template"],
                    "log_dir": cfg["artwork"]["log_dir"],
                    "status": f"failed: {exc}",
                }
                if not args.continue_on_error:
                    _write_run_log(run_meta, args.local_run_log_dir)
                    raise
            _write_run_log(run_meta, args.local_run_log_dir)
            run_meta_entries.append(run_meta)

    # Always write aggregated CSV locally
    _append_runs_csv(run_meta_entries, args.local_run_log_dir)

    if args.export and not args.dry_run:
        if not _ensure_langsmith_key(args.env_path):
            print("LANGSMITH/LANGCHAIN API key not set; skipping LangSmith remote export.")
        else:
            _export_langsmith(sorted(projects), args.export_dir)

    if failures:
        print(f"Completed with {len(failures)} failure(s).")
    else:
        print("Batch completed without errors.")


if __name__ == "__main__":
    main()
