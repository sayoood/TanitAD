"""tools/download_receipt.py — a receipt must never forget when the bytes were first fetched.

⛔ The defect these pin (MEASURED 2026-09-26, caught by the Master Mind): re-running the nuScenes fetch to
add S3-ETag verification TRUNCATED the receipt, moving ``utc`` from the real first fetch (13:10:51Z) to
the re-verification (13:47:12Z). The clock is CONTROLLED here, because two calls in the same second
would make "unchanged" pass for the wrong reason.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import download_receipt as R  # noqa: E402

KEY = "public/v1.0/v1.0-trainval_meta.tgz"
MD5 = "537d3954ec34e5bcb89a35d4f6fb0d4a"
ETAG = "fd4ea76d8701fb567a67a65025d0b022-56"
SRC = "https://motional-nuscenes.s3.amazonaws.com"


@pytest.fixture
def clock(monkeypatch):
    t = {"now": "2026-09-26T13:10:51Z"}
    monkeypatch.setattr(R, "_now", lambda: t["now"])
    return t


def _open(p, who="Sayed", tier="meta", **kw):
    argv = ["open", str(p), "--tier", tier, "--who", who, "--source", SRC]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", v]
    return R.main(argv)


def _record(p, md5=MD5, nbytes="461678030"):
    return R.main(["record", str(p), "--key", KEY, "--bytes", nbytes, "--md5", md5,
                   "--etag", ETAG, "--verdict", "MATCH"])


def _doc(p):
    return json.load(open(p, encoding="utf-8"))


def test_REGRESSION_a_re_run_never_moves_first_fetch_utc(clock, tmp_path):
    """⛔ The exact bug: fetch at 13:10:51, re-verify at 13:47:12 — the first time must survive."""
    p = tmp_path / "RECEIPT_meta.json"
    assert _open(p) == 0 and _record(p) == 0
    clock["now"] = "2026-09-26T13:47:12Z"
    assert _open(p) == 0 and _record(p) == 0
    d = _doc(p)
    assert d["first_fetch_utc"] == "2026-09-26T13:10:51Z"
    assert d["files"][0]["first_fetch_utc"] == "2026-09-26T13:10:51Z"
    assert [r["action"] for r in d["runs"]] == ["fetch", "re-verify"]
    assert [v["verified_utc"] for v in d["files"][0]["verifications"]] == \
           ["2026-09-26T13:10:51Z", "2026-09-26T13:47:12Z"]


def test_changed_bytes_on_a_re_fetch_are_REFUSED_and_the_receipt_is_untouched(clock, tmp_path):
    """A different object served must be LOUD, never silently absorbed."""
    p = tmp_path / "r.json"
    _open(p); _record(p)
    before = p.read_bytes()
    clock["now"] = "2026-09-26T14:00:00Z"
    assert _record(p, md5="0" * 32) == 1
    assert p.read_bytes() == before


def test_receipts_are_never_merged_across_tiers(clock, tmp_path):
    p = tmp_path / "r.json"
    _open(p, tier="meta")
    assert _open(p, tier="planning") == 1


def test_the_exact_55aa747_receipt_migrates_without_losing_meaning(clock, tmp_path):
    """The real first receipt, byte-for-byte in shape: its single `utc` becomes first_fetch_utc."""
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"tier": "meta", "accepted_terms_by": "Sayed", "utc": "2026-09-26T13:10:51Z",
                             "source": SRC, "files": [{"key": KEY, "bytes": 461678030, "md5": MD5}]}),
                 encoding="utf-8")
    clock["now"] = "2026-09-26T15:00:00Z"
    assert _open(p) == 0 and _record(p) == 0
    d = _doc(p)
    assert d["first_fetch_utc"] == "2026-09-26T13:10:51Z" and d["accepted_terms_by"] == "Sayed"
    assert d["files"][0]["verifications"][-1]["etag_verdict"] == "MATCH"


def test_a_repair_is_recorded_never_silent(clock, tmp_path):
    """Restoring an overwritten first_fetch_utc leaves an audit trail of what it was and where from."""
    p = tmp_path / "r.json"
    clock["now"] = "2026-09-26T13:47:12Z"                 # the overwritten state
    _open(p)
    assert _open(p, first_fetch_utc="2026-09-26T13:10:51Z", restored_from="git 55aa747") == 0
    d = _doc(p)
    assert d["first_fetch_utc"] == "2026-09-26T13:10:51Z"
    assert d["repairs"] == [{"utc": "2026-09-26T13:47:12Z", "field": "first_fetch_utc",
                             "was": "2026-09-26T13:47:12Z", "now": "2026-09-26T13:10:51Z",
                             "restored_from": "git 55aa747"}]


def test_writes_are_atomic_and_valid_json(clock, tmp_path):
    p = tmp_path / "r.json"
    _open(p); _record(p); R.main(["close", str(p), "--note", "licence=CC BY-NC-SA 4.0"])
    assert _doc(p)["licence"] == "CC BY-NC-SA 4.0"
    assert not (tmp_path / "r.json.tmp").exists()
