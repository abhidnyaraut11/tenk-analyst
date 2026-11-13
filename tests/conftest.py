"""Test setup: load .env and give tests a helper to read questions.json."""
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def load_questions():
    return json.loads((ROOT / "test_data" / "questions.json").read_text(encoding="utf-8"))
