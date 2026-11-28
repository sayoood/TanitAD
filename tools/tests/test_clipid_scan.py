"""The identifier guard — grandfather what is banked, refuse what is added.

**PI ruling 2026-09-17:** the sha12 rule is **hygiene and reproducibility**, not
confidentiality. ⇒ the banked record is left alone and the count is stopped from
growing.

⛔ Every refusal below is tested in BOTH directions, and the two that earn the
module are `_MUT_a_NEW_file_carrying_an_id_is_REFUSED` and
`_MUT_an_existing_file_that_GAINS_one_is_REFUSED` — a guard that only checked
totals would pass a file that gained one while another lost one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clipid_scan import (  # noqa: E402
    UUID_RE, LeakGrew, compare_to_baseline, scan_text, scan_tree,
    ClipListMismatch, main,
)
import clipid_scan as CS  # noqa: E402

# ⛔⛔ SYNTHETIC, and this matters more than it looks. My first version used two
# REAL corpus clip ids copied out of `extrinsics141.json` — so the guard's own
# test would have banked two clip ids, which is exactly what the guard exists to
# stop. The staging check caught it. These share no prefix with any corpus clip
# and are obviously fabricated on sight.
U1 = "deadbeef-0000-4000-8000-000000000001"
U2 = "deadbeef-0000-4000-8000-000000000002"


def _tree(tmp_path, files: dict) -> Path:
    for rel, body in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return tmp_path


G = ("Project Steering/**/*.md",)


def test_a_tree_at_its_baseline_passes_and_reports_the_totals(tmp_path):
    _tree(tmp_path, {"Project Steering/a.md": f"clip {U1} was inspected"})
    cur = scan_tree(tmp_path, globs=G)
    rep = compare_to_baseline(cur, cur)
    assert rep["verdict"] == "NO-GROWTH"
    assert rep["total_uuids"] == 1 and rep["files_with_ids"] == 1


def test_MUT_a_NEW_file_carrying_an_id_is_REFUSED(tmp_path):
    """The common case: somebody bankes a fresh note with a raw UUID in it."""
    _tree(tmp_path, {"Project Steering/a.md": f"clip {U1}"})
    base = scan_tree(tmp_path, globs=G)
    _tree(tmp_path, {"Project Steering/b.md": f"and clip {U2}"})
    with pytest.raises(LeakGrew, match="new files"):
        compare_to_baseline(scan_tree(tmp_path, globs=G), base)


def test_MUT_an_existing_file_that_GAINS_one_is_REFUSED(tmp_path):
    """⭐ The case a totals-only guard misses. Here the TOTAL is unchanged — one
    file gains an id while another loses one — and it must still be refused."""
    _tree(tmp_path, {"Project Steering/a.md": f"clip {U1}",
                     "Project Steering/b.md": f"clip {U2}"})
    base = scan_tree(tmp_path, globs=G)
    assert sum(v["uuids"] for v in base.values()) == 2
    _tree(tmp_path, {"Project Steering/a.md": f"clips {U1} and {U2}",
                     "Project Steering/b.md": "cleaned up, now uses sha12"})
    cur = scan_tree(tmp_path, globs=G)
    assert sum(v["uuids"] for v in cur.values()) == 2      # total UNCHANGED
    with pytest.raises(LeakGrew, match="grew"):
        compare_to_baseline(cur, base)


def test_a_file_that_SHRINKS_is_allowed_and_REPORTED(tmp_path):
    """⛔ Cleaning a file up must never be blocked by the guard that protects the
    cleanup — but it is reported, so the drop is visible."""
    _tree(tmp_path, {"Project Steering/a.md": f"{U1} and {U2}"})
    base = scan_tree(tmp_path, globs=G)
    _tree(tmp_path, {"Project Steering/a.md": f"{U1} only"})
    rep = compare_to_baseline(scan_tree(tmp_path, globs=G), base)
    assert rep["verdict"] == "NO-GROWTH"
    assert rep["shrank"] == ["Project Steering/a.md:uuids 2 -> 1"]


def test_MUT_an_UNREADABLE_file_REFUSES_rather_than_reporting_clean(tmp_path):
    """⛔ A zero from a file nobody opened is indistinguishable from a genuine
    absence — the trap this repo has been bitten by on a flaky mount."""
    cur = {"Project Steering/a.md": {"uuids": 1, "prefixes": 0},
           "_unreadable": ["Project Steering/b.md"]}
    with pytest.raises(LeakGrew, match="could not be read"):
        compare_to_baseline(cur, {"Project Steering/a.md":
                                  {"uuids": 1, "prefixes": 0}})


def test_scan_tree_REPORTS_a_file_it_could_not_open(tmp_path, monkeypatch):
    """⭐ The half the hand-built fixture above cannot reach. The audit caught
    this: removing `scan_tree`'s own unreadable branch broke NO test, because
    every other test constructs the `_unreadable` key by hand instead of making
    `scan_tree` produce it."""
    _tree(tmp_path, {"Project Steering/a.md": f"clip {U1}",
                     "Project Steering/bad.md": "whatever"})
    import clipid_scan as cs
    real = cs._read_any
    monkeypatch.setattr(
        cs, "_read_any",
        lambda p: None if p.name == "bad.md" else real(p))
    cur = scan_tree(tmp_path, globs=G)
    assert cur["_unreadable"] == ["Project Steering/bad.md"]
    with pytest.raises(LeakGrew, match="could not be read"):
        compare_to_baseline(cur, {"Project Steering/a.md":
                                  {"uuids": 1, "prefixes": 0}})


def test_a_cp1252_file_is_READ_not_reported_unreadable(tmp_path):
    """⚠️ Real files in this repo are cp1252 (one carries byte 0x97). Refusing
    them would make the guard permanently red, and a permanently red guard gets
    switched off."""
    p = tmp_path / "Project Steering" / "c.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    # ⛔ RAW BYTES. "\x97" in a Python str is U+0097, which cp1252 cannot ENCODE
    # — the cp1252 BYTE 0x97 is what decodes to an em-dash. My first version
    # built the str and died in the TEST, not in the code under test.
    p.write_bytes(b"an em" + bytes([0x97]) + b"dash and clip " + U1.encode())
    cur = scan_tree(tmp_path, globs=G)
    assert "_unreadable" not in cur
    assert cur["Project Steering/c.md"]["uuids"] == 1


def test_the_BASELINE_itself_carries_no_identifiers(tmp_path):
    """⛔⛔ The design property. An allowlist of the leaked ids would BE a list of
    clip ids — the exact thing this guard exists to stop growing."""
    _tree(tmp_path, {"Project Steering/a.md": f"{U1} {U2}"})
    blob = json.dumps(scan_tree(tmp_path, globs=G))
    assert UUID_RE.findall(blob) == []
    assert U1 not in blob and U2 not in blob


def test_the_pattern_does_not_match_a_longer_hex_run():
    """⛔ Word-anchored: an md5 or a sha must not read as a UUID."""
    assert scan_text(f"{U1}")["uuids"] == 1
    assert scan_text(f"ff{U1}")["uuids"] == 0
    assert scan_text("ff0e68fd41e86f6a30180ff6173b0685")["uuids"] == 0


def test_the_prefix_half_is_OPT_IN_and_off_without_a_clip_list():
    """The 8-char half needs external data, so it is optional; the UUID half is
    the one that always runs and cannot rot when a corpus list moves."""
    # ⛔ Synthetic prefix, same reason as U1/U2 above. My first version used a
    # REAL corpus clip prefix, i.e. exactly what this half detects — and the
    # replacement comment QUOTED IT, so the staging guard refused twice. The
    # literal is deliberately not repeated here.
    text = "clip deadbeef is a road bend"
    assert scan_text(text)["prefixes"] == 0
    assert scan_text(text, clip_prefixes={"deadbeef"})["prefixes"] == 1


# ── the NAMED prefix floor (2026-09-19) ─────────────────────────────────────
# A prefix count only means something against the list it was counted with, so the
# floor records the list's DIGEST and a run against any other list is refused.
# ⛔ Synthetic ids only — same reason as U1/U2: a real corpus id here would bank one.
L1 = ["deadbeef-0000-4000-8000-00000000000a", "deadbeef-0000-4000-8000-00000000000b"]
L1_ONE_OFF = ["deadbeef-0000-4000-8000-00000000000a", "deadbeef-0000-4000-8000-00000000000c"]


def _clips(tmp_path, name, ids):
    p = tmp_path / name
    p.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return str(p)


def _run(root, base, clips=None, write=False, name=""):
    argv = ["--root", str(root), "--baseline", str(base)]
    if clips:
        argv += ["--clips", clips]
    if write:
        argv += ["--write-baseline"]
    if name:
        argv += ["--clips-name", name]
    return main(argv)


def _repo(tmp_path, body="clip deadbeef is a road bend"):
    return _tree(tmp_path / "repo", {"Project Steering/a.md": body})


def test_a_prefix_floor_recorded_against_a_NAMED_list_reads_green_on_legacy(tmp_path, capsys):
    root, base, c = _repo(tmp_path), tmp_path / "b.json", _clips(tmp_path, "l1.txt", L1)
    assert _run(root, base, c, write=True, name="synthetic L1") == 0
    rec = json.loads(base.read_text(encoding="utf-8"))["_clips_list"]
    assert rec["n"] == 2 and rec["name"] == "synthetic L1"                      # literal
    capsys.readouterr()
    assert _run(root, base, c) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "NO-GROWTH"


def test_MUT_prefix_GROWTH_against_the_same_list_is_REFUSED(tmp_path):
    root, base, c = _repo(tmp_path), tmp_path / "b.json", _clips(tmp_path, "l1.txt", L1)
    _run(root, base, c, write=True)
    (root / "Project Steering" / "a.md").write_text(
        "clip deadbeef is a road bend; deadbeef again", encoding="utf-8")
    with pytest.raises(LeakGrew, match="prefixes 1 -> 2"):
        _run(root, base, c)


def test_MUT_a_DIFFERENT_list_of_the_SAME_size_is_REFUSED_not_compared(tmp_path):
    """Same n, one id different: a check on the count alone would pass it."""
    root, base = _repo(tmp_path), tmp_path / "b.json"
    _run(root, base, _clips(tmp_path, "l1.txt", L1), write=True)
    with pytest.raises(ClipListMismatch, match="SAME list"):
        _run(root, base, _clips(tmp_path, "l1b.txt", L1_ONE_OFF))


def test_MUT_a_baseline_WITHOUT_a_named_list_refuses_the_prefix_half(tmp_path):
    root, base = _repo(tmp_path, f"{U1}"), tmp_path / "b.json"
    _run(root, base, write=True)                                    # UUID half only
    with pytest.raises(ClipListMismatch, match="NO prefix floor"):
        _run(root, base, _clips(tmp_path, "l1.txt", L1))


def test_the_default_run_does_not_report_the_prefix_floor_as_a_SHRINK(tmp_path, capsys):
    """Without a list the prefix half is not counted; reading its 0 as a cleanup
    would tell an operator somebody removed ids nobody touched."""
    root, base = _repo(tmp_path, f"{U1} and clip deadbeef"), tmp_path / "b.json"
    _run(root, base, _clips(tmp_path, "l1.txt", L1), write=True)
    capsys.readouterr()
    assert _run(root, base) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["verdict"] == "NO-GROWTH" and rep["shrank"] == []                # literal


def test_the_SAME_list_written_differently_is_the_SAME_list(tmp_path, capsys):
    """The list cannot be banked, so every run REBUILDS it — from a directory listing
    or a manifest, in any order, perhaps by PowerShell (BOM, CRLF). Its identity is the
    sorted set; a rebuilt copy must not read as a mismatch."""
    root, base = _repo(tmp_path), tmp_path / "b.json"
    _run(root, base, _clips(tmp_path, "l1.txt", L1), write=True)
    p = tmp_path / "l1_rebuilt.txt"
    p.write_bytes(b"\xef\xbb\xbf" + ("\r\n".join(L1[::-1] + L1[:1]) + "\r\n").encode("utf-8"))
    capsys.readouterr()
    assert _run(root, base, str(p)) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "NO-GROWTH"


def test_the_named_list_is_stored_as_a_DIGEST_never_as_ids(tmp_path):
    root, base = _repo(tmp_path), tmp_path / "b.json"
    _run(root, base, _clips(tmp_path, "l1.txt", L1), write=True)
    text = base.read_text(encoding="utf-8")
    assert all(i not in text for i in L1)
    assert UUID_RE.findall(text) == []
    assert len(json.loads(text)["_clips_list"]["sha256_sorted"]) == 64          # literal


def test_MUT_a_baseline_is_NOT_written_over_UNREADABLE_files(tmp_path, monkeypatch):
    root = _tree(tmp_path / "repo", {"Project Steering/a.md": "clip deadbeef",
                                     "Project Steering/b.md": "anything"})
    base = tmp_path / "b.json"
    real = CS._read_any
    monkeypatch.setattr(CS, "_read_any", lambda p: None if p.name == "b.md" else real(p))
    with pytest.raises(LeakGrew, match="refusing to write a baseline"):
        _run(root, base, _clips(tmp_path, "l1.txt", L1), write=True)
    assert not base.exists()
