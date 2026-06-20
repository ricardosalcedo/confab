"""Tests for confab.cli."""

from io import StringIO
from unittest.mock import patch

import pytest

from confab.cli import _get_verdict, _read_input, main


class TestReadInput:
    def test_passthrough(self):
        assert _read_input("hello") == "hello"

    def test_stdin(self):
        with patch("sys.stdin", StringIO("from stdin")):
            assert _read_input("-") == "from stdin"

    def test_stdin_empty_exits(self):
        with patch("sys.stdin", StringIO("")), pytest.raises(SystemExit):
            _read_input("-")


class TestGetVerdict:
    def test_supported(self):
        assert _get_verdict("This is SUPPORTED by evidence.") == "SUPPORTED"

    def test_refuted(self):
        assert _get_verdict("This claim is REFUTED.") == "REFUTED"

    def test_uncertain(self):
        assert _get_verdict("I'm not sure about this.") == "UNCERTAIN"


class TestCLI:
    def test_no_command_shows_help(self, capsys):
        with patch("sys.argv", ["confab"]), pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0

    def test_version(self, capsys):
        with patch("sys.argv", ["confab", "--version"]), pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        assert "0.5.0" in capsys.readouterr().out

    def test_check_demo(self, capsys):
        with patch("sys.argv", ["confab", "--demo", "check", "Who created Python?"]):
            main()
        output = capsys.readouterr().out
        assert "high" in output or "medium" in output or "low" in output

    def test_check_demo_json(self, capsys):
        with patch("sys.argv", ["confab", "--demo", "--json", "check", "Who created Python?"]):
            main()
        output = capsys.readouterr().out
        assert '"claims"' in output

    def test_verify_demo(self, capsys):
        with patch("sys.argv", ["confab", "--demo", "verify", "Python was created in 1991"]):
            main()
        output = capsys.readouterr().out
        assert "SUPPORTED" in output

    def test_check_no_api_key_exits(self):
        with patch("sys.argv", ["confab", "check", "test"]), \
             patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False), \
             pytest.raises(SystemExit):
            main()

    def test_history_empty(self, capsys, tmp_path):
        with patch("sys.argv", ["confab", "history"]), \
             patch("confab.cli.get_history", return_value=[]):
            main()
        output = capsys.readouterr().out
        assert "No history" in output



class TestCLIHistory:
    def test_history_with_entries(self, capsys):
        mock_rows = [
            {"id": 1, "timestamp": "2024-01-01", "command": "check", "prompt": "test prompt",
             "model": "gpt-4o-mini", "samples": 5, "elapsed": 1.2,
             "claims_json": '[{"level": "high"}, {"level": "low"}]', "verdict": None},
            {"id": 2, "timestamp": "2024-01-02", "command": "verify", "prompt": "a claim",
             "model": "gpt-4o", "samples": 1, "elapsed": 0.5,
             "claims_json": "[]", "verdict": "SUPPORTED"},
        ]
        with patch("sys.argv", ["confab", "history"]), \
             patch("confab.cli.get_history", return_value=mock_rows):
            main()
        output = capsys.readouterr().out
        assert "test prompt" in output
        assert "SUPPORTED" in output

    def test_history_clear(self, capsys):
        with patch("sys.argv", ["confab", "history", "--clear"]), \
             patch("confab.cli.clear_history") as mock_clear:
            main()
        mock_clear.assert_called_once()


class TestCLIProxy:
    def test_proxy_no_key_exits(self):
        with patch("sys.argv", ["confab", "proxy"]), \
             patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False), \
             pytest.raises(SystemExit):
            main()
