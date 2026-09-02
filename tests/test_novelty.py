import json
import re
import tempfile
from pathlib import Path

from src.check_novelty import validate_novelty


def test_offline_novelty_is_fail_closed_for_missing_cache(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = Path(temp_dir) / "mp_novelty_cache.json"
        cache_path.write_text(
            json.dumps(
                {
                    "Cs2AgBiBr6": {
                        "mp_id": "mp-123",
                        "is_synthesized": True,
                    }
                }
            ),
            encoding="utf-8",
        )

        # Keep the test explicitly offline; the temporary path is passed
        # directly so no default cache can influence the result.
        monkeypatch.setenv("MP_API_KEY", "")
        report, exit_code = validate_novelty(
            ["Cs2AgBiBr6", "Xu2ZrCl6"],
            cache_path=cache_path,
            online=False,
        )

        known = report.loc[report["Formula"] == "Cs2AgBiBr6"].iloc[0]
        unresolved = report.loc[report["Formula"] == "Xu2ZrCl6"].iloc[0]
        assert known["status"] == "known"
        assert bool(known["is_synthesized"]) is True
        assert unresolved["status"] == "unknown"
        assert exit_code == 1


def test_cell_14_does_not_contain_hardcoded_api_key_literal():
    cell14_path = "scripts/_cells/cell_14.py"
    source = open(cell14_path, encoding="utf-8").read()
    assert not re.search(r'MP_API_KEY\s*=\s*"[A-Za-z0-9]{20,}"', source)
