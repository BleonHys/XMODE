#!/usr/bin/env python3
"""Utility to download and materialise datasets required by XMODE."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import requests
from tqdm import tqdm

from src.settings import get_settings


CHUNK_SIZE = 32768
GOOGLE_DRIVE_ENDPOINT = "https://docs.google.com/uc"  # more reliable for large files


class DownloadError(RuntimeError):
    """Raised when an asset cannot be downloaded."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_google_drive(file_id: str, destination: Path) -> None:
    session = requests.Session()
    params = {"id": file_id, "export": "download"}
    response = session.get(GOOGLE_DRIVE_ENDPOINT, params=params, stream=True, timeout=30)
    token = _confirm_token(response)

    if token:
        params["confirm"] = token
        response = session.get(GOOGLE_DRIVE_ENDPOINT, params=params, stream=True, timeout=30)

    _save_stream(response, destination)


def _confirm_token(response: requests.Response) -> Optional[str]:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _save_stream(response: requests.Response, destination: Path) -> None:
    response.raise_for_status()
    total = int(response.headers.get("Content-Length", 0)) or None
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    with temp_path.open("wb") as handle, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=f"Downloading {destination.name}",
    ) as progress:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                handle.write(chunk)
                progress.update(len(chunk))

    temp_path.replace(destination)


def _needs_download(path: Path, expected_sha: Optional[str], force: bool) -> bool:
    if force or not path.exists():
        return True
    if expected_sha is None:
        return False
    return _sha256(path) != expected_sha


def _download_asset(name: str, info: Dict[str, object], force: bool = False) -> None:
    settings = get_settings()
    target = Path(info["path"])  # type: ignore[index]
    if not target.is_absolute():
        target = settings.base_dir / target

    checksum_info = info.get("checksum") if isinstance(info, dict) else None
    expected_sha = None
    if isinstance(checksum_info, dict):
        expected_sha = checksum_info.get("sha256")

    if not _needs_download(target, expected_sha, force):
        print(f"✔ {name}: already present at {target}")
        return

    source = info.get("source") if isinstance(info, dict) else None
    if not isinstance(source, dict):
        raise DownloadError(f"{name}: missing download source configuration")

    source_type = source.get("type")
    if source_type == "gdrive":
        file_id = source.get("id")
        if not file_id:
            raise DownloadError(f"{name}: Google Drive entry missing 'id'")
        _download_google_drive(str(file_id), target)
    elif source_type == "http":
        url = source.get("url")
        if not url:
            raise DownloadError(f"{name}: HTTP entry missing 'url'")
        response = requests.get(str(url), stream=True, timeout=30)
        _save_stream(response, target)
    else:
        raise DownloadError(f"{name}: unsupported source type '{source_type}'")

    if expected_sha and _sha256(target) != expected_sha:
        raise DownloadError(f"{name}: checksum mismatch after download")

    print(f"✔ {name}: downloaded to {target}")


def _iter_assets(requested: Iterable[str]) -> Dict[str, Dict[str, object]]:
    settings = get_settings()
    assets: Dict[str, Dict[str, object]] = {}
    configured = {
        key: value for key, value in settings.downloads.items() if isinstance(value, dict)
    }

    if not requested:
        return configured

    for item in requested:
        if item not in configured:
            raise KeyError(
                f"Unknown asset '{item}'. Use --list to see available options."
            )
        assets[item] = configured[item]
    return assets


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download and prepare dataset artefacts for XMODE.",
    )
    parser.add_argument(
        "--target",
        "-t",
        action="append",
        help="Specific asset key to fetch (can be supplied multiple times). Defaults to all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download assets even if they already exist.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available assets and exit.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    settings = get_settings()
    assets = _iter_assets(args.target or [])

    if args.list:
        if not assets:
            print("No downloadable assets configured.")
        else:
            for key, value in assets.items():
                description = value.get("description", "") if isinstance(value, dict) else ""
                target_path = value.get("path") if isinstance(value, dict) else ""
                print(f"{key}: {description}\n    -> {target_path}")
        return 0

    if not assets:
        print("No assets to download. Check configuration.")
        return 0

    exit_code = 0
    for key, value in assets.items():
        try:
            _download_asset(key, value, force=args.force)
        except Exception as exc:  # noqa: BLE001
            print(f"✖ {key}: {exc}")
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
