"""Tests for log->JSONL recovery. This is the ONLY path by which work survives
a reclaimed VM, so its failure modes must be loud."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import vlm_tac_recover as R  # noqa: E402


def _frame(obj) -> str:
    b = base64.b64encode(json.dumps(obj).encode()).decode()
    return f"@@G{len(b)}@{b}@@"


def test_a_frame_round_trips():
    rec = {"clip_id": "a", "kind": "lon", "raw": "VERDICT: FOLLOW",
           "n_new_tokens": 900, "hit_cap": False}
    recs, short = R.frames_from("noise\n" + _frame(rec) + "\nmore noise")
    assert recs == [rec] and short == 0


def test_a_wrapped_frame_still_decodes():
    """A log viewer or PTY may wrap a long line. Harmless — only a LENGTH
    mismatch means real loss."""
    rec = {"clip_id": "a", "kind": "lane", "raw": "x" * 200}
    f = _frame(rec)
    i = len(f) // 2
    recs, short = R.frames_from(f[:i] + "\n" + f[i:])
    assert recs == [rec] and short == 0


def test_a_truncated_frame_is_COUNTED_not_silently_accepted():
    """⛔ The whole point of the declared length: base64 can decode to garbage
    rather than raising, so a short frame must be caught by arithmetic."""
    rec = {"clip_id": "a", "kind": "lon", "raw": "y" * 100}
    f = _frame(rec)
    recs, short = R.frames_from(f[:40] + f[60:])     # bytes removed
    assert recs == [] and short == 1


def test_merge_is_last_writer_wins_per_clip_and_kind(tmp_path):
    out = tmp_path / "raw.jsonl"
    out.write_text(json.dumps({"clip_id": "a", "kind": "lon", "raw": "OLD"}) + "\n",
                   encoding="utf-8")
    lg = tmp_path / "run.log"
    lg.write_text(_frame({"clip_id": "a", "kind": "lon", "raw": "NEW"}) + "\n"
                  + _frame({"clip_id": "b", "kind": "lon", "raw": "B"}),
                  encoding="utf-8")
    R.main(["--out", str(out), str(lg)])
    got = {(json.loads(l)["clip_id"], json.loads(l)["kind"]): json.loads(l)["raw"]
           for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    assert got == {("a", "lon"): "NEW", ("b", "lon"): "B"}


def test_nul_bytes_and_bom_from_powershell_redirection_survive(tmp_path):
    """⚠️ PowerShell `Out-File -Encoding utf8` writes a BOM, and the captured
    stream carries NULs. Recovery must not choke on either."""
    out = tmp_path / "raw.jsonl"
    lg = tmp_path / "run.log"
    body = _frame({"clip_id": "a", "kind": "lon", "raw": "ok"})
    lg.write_bytes(b"\xef\xbb\xbf" + body.encode().replace(b"@", b"\x00@", 1))
    R.main(["--out", str(out), str(lg)])
    assert "ok" in out.read_text(encoding="utf-8")


def test_missing_log_is_reported_not_fatal(tmp_path):
    out = tmp_path / "raw.jsonl"
    assert R.main(["--out", str(out), str(tmp_path / "nope.log")]) == 0
