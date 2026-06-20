"""Tests for confab.db."""

import json
from pathlib import Path

import pytest

from confab.db import clear_history, get_history, save_check
from confab.engine import Claim


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


class TestDatabase:
    def test_save_and_retrieve(self, tmp_db: Path):
        claims = [Claim(text="Test claim", confidence=0.8, support_count=4, total_samples=5)]
        save_check("check", "test prompt", "gpt-4o-mini", 5, 1.2, claims, db_path=tmp_db)

        history = get_history(limit=10, db_path=tmp_db)
        assert len(history) == 1
        assert history[0]["command"] == "check"
        assert history[0]["prompt"] == "test prompt"
        assert history[0]["model"] == "gpt-4o-mini"
        assert history[0]["samples"] == 5

    def test_save_verify(self, tmp_db: Path):
        save_check("verify", "claim text", "gpt-4o", 1, 0.5, verdict="SUPPORTED", db_path=tmp_db)

        history = get_history(db_path=tmp_db)
        assert history[0]["verdict"] == "SUPPORTED"

    def test_claims_json_serialized(self, tmp_db: Path):
        claims = [
            Claim(text="A", confidence=0.9, support_count=4, total_samples=5),
            Claim(text="B", confidence=0.4, support_count=2, total_samples=5),
        ]
        save_check("check", "p", "m", 5, 1.0, claims, db_path=tmp_db)

        history = get_history(db_path=tmp_db)
        parsed = json.loads(history[0]["claims_json"])
        assert len(parsed) == 2
        assert parsed[0]["text"] == "A"
        assert parsed[0]["confidence"] == 0.9

    def test_clear_history(self, tmp_db: Path):
        save_check("check", "p", "m", 5, 1.0, db_path=tmp_db)
        save_check("check", "p2", "m", 5, 1.0, db_path=tmp_db)
        assert len(get_history(db_path=tmp_db)) == 2

        clear_history(db_path=tmp_db)
        assert len(get_history(db_path=tmp_db)) == 0

    def test_history_order_newest_first(self, tmp_db: Path):
        save_check("check", "first", "m", 5, 1.0, db_path=tmp_db)
        save_check("check", "second", "m", 5, 1.0, db_path=tmp_db)

        history = get_history(db_path=tmp_db)
        assert history[0]["prompt"] == "second"
        assert history[1]["prompt"] == "first"

    def test_limit_respected(self, tmp_db: Path):
        for i in range(10):
            save_check("check", f"prompt {i}", "m", 5, 1.0, db_path=tmp_db)

        assert len(get_history(limit=3, db_path=tmp_db)) == 3

    def test_empty_claims(self, tmp_db: Path):
        save_check("verify", "claim", "m", 1, 0.5, claims=None, verdict="REFUTED", db_path=tmp_db)
        history = get_history(db_path=tmp_db)
        assert json.loads(history[0]["claims_json"]) == []
