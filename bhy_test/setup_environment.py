#!/usr/bin/env python3
"""
Utility to bootstrap a tiny self-contained dataset for the Claude playground.
Creates:
- bhy_test/images/nature_sample.jpg       (placeholder image)
- bhy_test/nature.db                      (SQLite DB with table 'paintings')
- bhy_test/questions_nature.json          (single sample question)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
DB_PATH = BASE_DIR / "nature.db"
QUESTIONS_PATH = BASE_DIR / "questions_nature.json"


def ensure_image() -> str:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_path = IMAGES_DIR / "nature_sample.jpg"
    if image_path.exists():
        return image_path.as_posix()

    img = Image.new("RGB", (640, 480), "#6ca16b")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 320, 640, 480], fill="#4c6b3c")
    draw.ellipse([50, 50, 250, 250], fill="#ffd966")
    draw.polygon([(320, 200), (400, 400), (240, 400)], fill="#8c6239")
    draw.polygon([(420, 220), (520, 420), (320, 420)], fill="#8c6239")
    font = ImageFont.load_default()
    draw.text((20, 440), "Nature Sample", fill="white", font=font)
    img.save(image_path, "JPEG")
    return image_path.as_posix()


def ensure_database(image_path: str) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS paintings (
            title TEXT,
            inception TEXT,
            movement TEXT,
            genre TEXT,
            image_url TEXT,
            img_path TEXT
        )
        """
    )
    cur.execute("DELETE FROM paintings")
    cur.execute(
        """
        INSERT INTO paintings (title, inception, movement, genre, image_url, img_path)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "Serene Valley",
            "1495-01-01 00:00:00",
            "Nature Revival",
            "Landscape",
            "local://nature_sample",
            image_path,
        ),
    )
    conn.commit()
    conn.close()


def ensure_questions() -> None:
    QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if QUESTIONS_PATH.exists():
        return
    QUESTIONS_PATH.write_text(
        json.dumps(
            [
                "Describe the oldest nature painting in the test dataset.",
                "What scenes are depicted in the Serene Valley artwork?",
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    img_path = ensure_image()
    ensure_database(img_path)
    ensure_questions()
    print("Test environment ready:")
    print(" - DB:", DB_PATH)
    print(" - Image:", img_path)
    print(" - Questions:", QUESTIONS_PATH)


if __name__ == "__main__":
    main()
