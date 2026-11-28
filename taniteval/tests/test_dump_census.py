"""`dump_census` — the C126 institutionalization, pinned by test.

⛔ THE DEFECT THESE PIN (RETRACTION_LOG.md C126). Two banked dump pairs are one
model each; the duplication was documented in prose 2026-07-26 and then
re-counted by every later census for 23 days, because censuses re-derive their
arm list from ``glob("windows_*.pt")`` — a surface prose cannot reach. "27
arms" was 27 dumps over 25 distinct arms. The machine-readable fix is
``taniteval/results/dump_exclusions.json``; ``taniteval.dump_census`` is what
makes census code actually CONSUME it. These tests pin the contract:

* an exclusion is honored AND reported — never a silent drop;
* a MISSING exclusions file is loud, not fatal (tests census scratch dirs);
* a STALE exclusion (file bytes changed under the recorded sha256) FAILS —
  silently excluding unverified content would be worse than double counting;
* the census surfaces (`driving.arms_with_windows`, `driving._load_blocks`,
  `tools/ff_rescore.py`) really route through it;
* the REAL results dir reads dumps_found - 2 = distinct_arms while the two
  recorded exclusions hold (relationship, not a pinned total — new dumps may
  land; regressions of the subtraction may not).

CPU-only, no GPU, no checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from taniteval import dump_census as DC

REPO = Path(__file__).resolve().parents[2]
FF_TOOL = REPO / "taniteval" / "tools" / "ff_rescore.py"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _mk_dump(d: Path, key: str, payload: bytes) -> Path:
    p = d / f"windows_{key}.pt"
    p.write_bytes(payload)
    return p


def _write_exclusions(d: Path, entries) -> Path:
    p = d / DC.EXCLUSIONS_NAME
    p.write_text(json.dumps({"_schema": ["excluded_name", "canonical_name",
                                         "reason", "evidence", "sha256"],
                             "exclusions": entries}), encoding="utf-8")
    return p


def _entry(d: Path, excl_key: str, canon_key: str, reason="same model, "
           "double-banked") -> dict:
    ex, ca = d / f"windows_{excl_key}.pt", d / f"windows_{canon_key}.pt"
    ent = {"excluded_name": ex.name, "canonical_name": ca.name,
           "reason": reason, "evidence": "test fixture"}
    if ex.exists():
        ent["sha256"] = _sha(ex.read_bytes())
    if ca.exists():
        ent["canonical_sha256"] = _sha(ca.read_bytes())
    return ent


# --------------------------------------------------------------------------- #
# the loader contract                                                          #
# --------------------------------------------------------------------------- #
def test_exclusion_honored_and_reported_never_silent(tmp_path):
    for k, payload in (("a", b"AAAA"), ("b", b"AAAA"), ("c", b"CCCC")):
        _mk_dump(tmp_path, k, payload)
    _write_exclusions(tmp_path, [_entry(tmp_path, "b", "a")])

    c = DC.list_dumps(tmp_path)
    assert [p.name for p in c.paths] == ["windows_a.pt", "windows_c.pt"]
    assert c.arm_keys == ["a", "c"]
    # the drop is PART OF THE RETURN VALUE, not an internal detail
    assert set(c.excluded) == {"windows_b.pt"}
    assert "double-banked" in c.excluded["windows_b.pt"]
    assert c.counts == {"dumps_found": 3, "distinct_arms": 2}
    assert c.excluded_arm_keys == {"b"}
    assert c.pairs_by_key == {"b": "a"}
    assert not c.exclusions_missing


def test_summary_renders_dumps_arms_and_the_excluded_pairs(tmp_path):
    for k in ("a", "b", "c"):
        _mk_dump(tmp_path, k, k.encode())
    _mk_dump(tmp_path, "b", b"same")          # rewrite b, then hash it
    _write_exclusions(tmp_path, [_entry(tmp_path, "b", "a")])

    s = DC.list_dumps(tmp_path).summary()
    assert "3 dumps = 2 distinct arms" in s
    assert "1 excluded" in s
    assert "windows_b.pt -> windows_a.pt" in s
    # __str__ is the summary, so a bare f-string interpolation reports right
    assert str(DC.list_dumps(tmp_path)) == s


def test_missing_exclusions_file_is_loud_not_fatal(tmp_path):
    for k in ("a", "b"):
        _mk_dump(tmp_path, k, k.encode())

    c = DC.list_dumps(tmp_path)
    assert c.exclusions_missing is True
    assert [p.name for p in c.paths] == ["windows_a.pt", "windows_b.pt"]
    assert c.counts == {"dumps_found": 2, "distinct_arms": 2}
    s = c.summary()
    assert "NOT DEDUPLICATED" in s and DC.EXCLUSIONS_NAME in s


def test_stale_excluded_sha256_fails_loudly(tmp_path):
    """The exclusion asserts equality about SPECIFIC bytes. If the file
    changed, silently excluding the new content could hide a genuinely
    distinct arm — that would be WORSE than the double count."""
    _mk_dump(tmp_path, "a", b"AAAA")
    _mk_dump(tmp_path, "b", b"AAAA")
    ent = _entry(tmp_path, "b", "a")
    _write_exclusions(tmp_path, [ent])
    (tmp_path / "windows_b.pt").write_bytes(b"REBANKED-DIFFERENT")

    with pytest.raises(DC.StaleExclusionError) as ei:
        DC.list_dumps(tmp_path)
    assert "windows_b.pt" in str(ei.value)
    assert "STALE" in str(ei.value)


def test_stale_canonical_sha256_also_fails(tmp_path):
    _mk_dump(tmp_path, "a", b"AAAA")
    _mk_dump(tmp_path, "b", b"AAAA")
    ent = _entry(tmp_path, "b", "a")
    _write_exclusions(tmp_path, [ent])
    (tmp_path / "windows_a.pt").write_bytes(b"CANONICAL-CHANGED")

    with pytest.raises(DC.StaleExclusionError):
        DC.list_dumps(tmp_path)


def test_unparseable_exclusions_is_fatal_not_silent(tmp_path):
    _mk_dump(tmp_path, "a", b"AAAA")
    (tmp_path / DC.EXCLUSIONS_NAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(DC.ExclusionsError):
        DC.list_dumps(tmp_path)


def test_include_excluded_returns_files_but_still_reports(tmp_path):
    """A tool doing per-dump integrity work counts FILES — and says so."""
    for k in ("a", "b"):
        _mk_dump(tmp_path, k, b"AAAA")
    _write_exclusions(tmp_path, [_entry(tmp_path, "b", "a")])

    c = DC.list_dumps(tmp_path, include_excluded=True)
    assert [p.name for p in c.paths] == ["windows_a.pt", "windows_b.pt"]
    assert set(c.excluded) == {"windows_b.pt"}          # still reported
    assert c.counts == {"dumps_found": 2, "distinct_arms": 1}
    assert "include_excluded=True" in c.summary()


def test_canonical_absent_keeps_the_excluded_dump(tmp_path):
    """Dropping the excluded dump when its canonical is GONE would remove the
    model from the census, not a duplicate of it."""
    _mk_dump(tmp_path, "b", b"AAAA")
    ent = {"excluded_name": "windows_b.pt", "canonical_name": "windows_a.pt",
           "reason": "same model", "evidence": "test",
           "sha256": _sha(b"AAAA")}
    _write_exclusions(tmp_path, [ent])

    c = DC.list_dumps(tmp_path)
    assert [p.name for p in c.paths] == ["windows_b.pt"]
    assert c.excluded == {}
    assert c.counts == {"dumps_found": 1, "distinct_arms": 1}
    assert any("windows_b.pt kept" in n for n in c.notes)
    assert "kept" in c.summary()


def test_check_explicit_classifies_pair_vs_alone(tmp_path):
    _mk_dump(tmp_path, "a", b"AAAA")
    _mk_dump(tmp_path, "b", b"AAAA")
    _write_exclusions(tmp_path, [_entry(tmp_path, "b", "a")])

    both, consulted = DC.check_explicit(
        [str(tmp_path / "windows_a.pt"), str(tmp_path / "windows_b.pt")])
    assert consulted and both and both[0]["kind"] == "pair_present"

    alone, _ = DC.check_explicit([str(tmp_path / "windows_b.pt")])
    assert alone[0]["kind"] == "excluded_passed"

    none_found, consulted2 = DC.check_explicit(
        [str(tmp_path.parent / "nowhere" / "windows_x.pt")])
    assert none_found == [] and consulted2 == []


# --------------------------------------------------------------------------- #
# census surfaces actually consume it                                          #
# --------------------------------------------------------------------------- #
def test_driving_arms_with_windows_consumes_exclusions(tmp_path):
    """The tier-0 population is DISTINCT ARMS, not dump files."""
    from taniteval import driving as D
    for k in ("flagship-30k", "overfit-dup", "not-in-any-registry"):
        _mk_dump(tmp_path, k, b"AAAA")
    _write_exclusions(tmp_path, [_entry(tmp_path, "overfit-dup",
                                        "flagship-30k")])
    assert D.arms_with_windows(tmp_path) == ["flagship-30k",
                                             "not-in-any-registry"]


def test_driving_arms_with_windows_without_exclusions_unchanged(tmp_path):
    """No exclusions file (scratch dirs, pods mid-eval): the old contract
    holds verbatim — everything persisted is censused."""
    from taniteval import driving as D
    for k in ("flagship-30k", "not-in-any-registry"):
        _mk_dump(tmp_path, k, b"")
    assert D.arms_with_windows(tmp_path) == ["flagship-30k",
                                             "not-in-any-registry"]


def test_driving_load_blocks_drops_the_duplicate_row(tmp_path, capsys):
    """Both C126 arms HAVE driving_*.json blocks — the panel/leaderboard read
    must not double-count, and must not drop silently."""
    from taniteval import driving as D
    _mk_dump(tmp_path, "canon", b"AAAA")
    _mk_dump(tmp_path, "dup", b"AAAA")
    _write_exclusions(tmp_path, [_entry(tmp_path, "dup", "canon")])
    for key in ("canon", "dup"):
        (tmp_path / f"driving_{key}.json").write_text(
            json.dumps({"arm": key, "block": D.BLOCK}), encoding="utf-8")

    blocks = D._load_blocks(tmp_path)
    assert set(blocks) == {"canon"}
    out = capsys.readouterr().out
    assert "dropped duplicate" in out and "dup" in out


def test_driving_load_blocks_keeps_row_when_canonical_block_absent(tmp_path):
    from taniteval import driving as D
    _mk_dump(tmp_path, "canon", b"AAAA")
    _mk_dump(tmp_path, "dup", b"AAAA")
    _write_exclusions(tmp_path, [_entry(tmp_path, "dup", "canon")])
    (tmp_path / "driving_dup.json").write_text(
        json.dumps({"arm": "dup", "block": D.BLOCK}), encoding="utf-8")

    assert set(D._load_blocks(tmp_path)) == {"dup"}


@pytest.mark.skipif(not FF_TOOL.exists(), reason="ff_rescore.py missing")
def test_ff_rescore_refuses_a_known_same_model_pair(tmp_path):
    """The line-574 check refuses duplicate NAMES only; this pins that the
    tool now also refuses duplicate VALUES it can know about — BEFORE the
    expensive part (the fixture dumps are not even loadable)."""
    _mk_dump(tmp_path, "canon", b"NOT-A-REAL-TORCH-FILE")
    _mk_dump(tmp_path, "dup", b"NOT-A-REAL-TORCH-FILE")
    _write_exclusions(tmp_path, [_entry(tmp_path, "dup", "canon")])

    env = dict(os.environ, PYTHONUTF8="1")
    r = subprocess.run(
        [sys.executable, str(FF_TOOL),
         "--dump", f"A={tmp_path / 'windows_canon.pt'}",
         "--dump", f"B={tmp_path / 'windows_dup.pt'}",
         "--out-dir", str(tmp_path / "out"), "--strategic-no-label"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=240, env=env)
    assert r.returncode == 2, (r.stdout[-800:], r.stderr[-800:])
    assert "REFUSED" in r.stdout
    assert "windows_dup.pt" in r.stdout and "windows_canon.pt" in r.stdout
    assert "--include-excluded" in r.stdout


# --------------------------------------------------------------------------- #
# the real bank                                                                #
# --------------------------------------------------------------------------- #
def test_real_results_dir_subtracts_the_recorded_exclusions():
    """Relationship pin, not a total pin: new dumps may land (found grows),
    but while the two recorded exclusions hold, distinct = found - 2 and the
    two names are the excluded ones. Running this also re-validates the four
    recorded sha256s against the committed bytes — a re-banked dump fails
    here LOUDLY instead of silently re-entering censuses."""
    res = REPO / "taniteval" / "results"
    if not (res / DC.EXCLUSIONS_NAME).exists():
        pytest.skip("no committed results bank beside this checkout")
    c = DC.list_dumps(res)
    assert not c.exclusions_missing
    assert set(c.excluded) == {"windows_overfit_refa-dynin-30k.pt",
                               "windows_refc-v12-identity.pt"}
    assert c.counts["distinct_arms"] == c.counts["dumps_found"] - 2
    assert c.counts["dumps_found"] >= 27          # the 2026-08-18 bank
    s = c.summary()
    assert "distinct arms" in s and "2 excluded" in s
