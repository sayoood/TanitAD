"""The standing check for MODEL_REGISTRY.md's citations, and its ratchet.

Why these tests exist
--------------------
On 2026-08-03 a sweep of `Project Steering/MODEL_REGISTRY.md` found **21 defective
citations** — 4 malformed and 17 brace/glob forms that name no file — sitting in
the one document the whole programme is required to quote from. None of them were
new. They had accumulated **invisibly**, because nothing ever failed.

A one-shot repair does not fix that. `test_real_registry_has_no_dead_citations`
below is the fix: a new dead or malformed citation now turns `pytest -q` red.

The allowlist tests guard the obvious hole in that idea — an allowlist is only
honest if it cannot be used to make a defect disappear. Two mechanisms, one test
each: every excused token declares HOW MANY sites it is excused at
(`test_reintroducing_an_excused_token_at_a_new_site_fails`), and genuinely
unrepairable citations are ratcheted (`test_unresolved_over_ratchet_is_hard_failure`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import registry_paths as rp                                   # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _write_allow(tmp_path: Path, entries: list[dict], max_unresolved: int = 0):
    p = tmp_path / "allow.json"
    p.write_text(json.dumps({"entries": entries,
                             "max_unresolved": max_unresolved}),
                 encoding="utf-8")
    return p


def _write_registry(tmp_path: Path, body: str) -> Path:
    d = tmp_path / "Project Steering"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "MODEL_REGISTRY.md"
    p.write_text(body, encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# allowlist validation — an excuse with no reason teaches nobody anything
# --------------------------------------------------------------------------

def test_allowlist_entry_without_reason_is_rejected(tmp_path):
    p = _write_allow(tmp_path, [{"citation": "a_*.pt", "status": "pattern",
                                 "occurrences": 1}])
    with pytest.raises(ValueError, match="no reason"):
        rp.load_allow(p)


def test_allowlist_entry_without_occurrences_is_rejected(tmp_path):
    p = _write_allow(tmp_path, [{"citation": "a_*.pt", "status": "pattern",
                                 "reason": "x"}])
    with pytest.raises(ValueError, match="occurrences"):
        rp.load_allow(p)


def test_allowlist_entry_with_unknown_status_is_rejected(tmp_path):
    p = _write_allow(tmp_path, [{"citation": "a_*.pt", "status": "wontfix",
                                 "occurrences": 1, "reason": "x"}])
    with pytest.raises(ValueError, match="status"):
        rp.load_allow(p)


def test_absent_allowlist_is_the_strictest_state_not_a_broken_one(tmp_path):
    """No allowlist must mean 'nothing is excused', never 'skip the check'."""
    allow = rp.load_allow(tmp_path / "does_not_exist.json")
    assert allow["entries"] == []
    assert allow["max_unresolved"] == 0


# --------------------------------------------------------------------------
# relabelling
# --------------------------------------------------------------------------

def test_pattern_entry_relabels_and_is_not_a_defect(tmp_path):
    reg = _write_registry(tmp_path, "Episode contract: `ep_*.pt` per episode.\n")
    allow = _write_allow(tmp_path, [{"citation": "ep_*.pt", "status": "pattern",
                                     "occurrences": 1,
                                     "reason": "filename convention"}])
    res = rp.sweep(registry=reg, repo_root=tmp_path, allow_path=allow)
    rec = res["citations"][0]
    assert rec["status"] == rp.PATTERN
    assert rec["allow_reason"] == "filename convention"
    assert res["counts"].get(rp.NOT_A_PATH, 0) == 0
    assert rp.exit_code(res) == 0


def test_unresolved_entry_is_counted_and_exits_2_not_0(tmp_path):
    """A known-dead citation must never look clean."""
    reg = _write_registry(tmp_path, "cited: `gate_step{1k,5k}.json` on a dead pod\n")
    allow = _write_allow(tmp_path,
                         [{"citation": "gate_step{1k,5k}.json",
                           "status": "unresolved", "occurrences": 1,
                           "reason": "host terminated, no backup"}],
                         max_unresolved=1)
    res = rp.sweep(registry=reg, repo_root=tmp_path, allow_path=allow)
    assert res["counts"][rp.UNRESOLVED] == 1
    assert res["ratchet_exceeded"] is False
    assert rp.exit_code(res) == 2                     # visible, not fatal
    assert rp.exit_code(res, strict=True) == 1        # fatal on demand


# --------------------------------------------------------------------------
# the two anti-rug mechanisms
# --------------------------------------------------------------------------

def test_reintroducing_an_excused_token_at_a_new_site_fails(tmp_path):
    """The allowlist excuses COUNTED SITES, not a token forever.

    Without this, one allowlist line would licence the same defect anywhere in
    the document — which is exactly how 17 of them accumulated.
    """
    reg = _write_registry(tmp_path, "one `eff_*.json`\nand another `eff_*.json`\n")
    allow = _write_allow(tmp_path, [{"citation": "eff_*.json", "status": "pattern",
                                     "occurrences": 1, "reason": "quoted verbatim"}])
    res = rp.sweep(registry=reg, repo_root=tmp_path, allow_path=allow)
    kinds = [i["status"] for i in res["allow_issues"]]
    assert rp.ALLOW_COUNT_MISMATCH in kinds
    assert res["allow_issues"][0]["expected"] == 1
    assert res["allow_issues"][0]["found"] == 2
    assert rp.exit_code(res) == 1


def test_stale_allowlist_entry_fails(tmp_path):
    """An excuse for a citation that no longer exists is rot; delete it."""
    reg = _write_registry(tmp_path, "nothing cited here\n")
    allow = _write_allow(tmp_path, [{"citation": "eff_*.json", "status": "pattern",
                                     "occurrences": 1, "reason": "x"}])
    res = rp.sweep(registry=reg, repo_root=tmp_path, allow_path=allow)
    assert res["allow_issues"][0]["status"] == rp.ALLOW_STALE
    assert rp.exit_code(res) == 1


def test_unresolved_over_ratchet_is_a_hard_failure(tmp_path):
    """Adding a new unresolved defect must fail until someone raises the bar
    deliberately, in a one-line reviewable diff."""
    reg = _write_registry(tmp_path, "`a_{x,y}.json` and `b_{x,y}.json`\n")
    allow = _write_allow(tmp_path,
                         [{"citation": "a_{x,y}.json", "status": "unresolved",
                           "occurrences": 1, "reason": "lost"},
                          {"citation": "b_{x,y}.json", "status": "unresolved",
                           "occurrences": 1, "reason": "lost"}],
                         max_unresolved=1)
    res = rp.sweep(registry=reg, repo_root=tmp_path, allow_path=allow)
    assert res["n_unresolved"] == 2
    assert res["ratchet_exceeded"] is True
    assert rp.exit_code(res) == 1


def test_ratchet_reports_when_it_can_be_tightened(tmp_path):
    reg = _write_registry(tmp_path, "nothing\n")
    allow = _write_allow(tmp_path, [], max_unresolved=2)
    res = rp.sweep(registry=reg, repo_root=tmp_path, allow_path=allow)
    assert res["ratchet_loose"] is True
    assert rp.exit_code(res) == 0          # improvement never blocks a commit


def test_count_occurrences_is_an_exact_backticked_match():
    text = "`foo.json` and `bar/foo.json` and plain foo.json\n"
    assert rp.count_occurrences(text, "foo.json") == 1
    assert rp.count_occurrences(text, "bar/foo.json") == 1
    assert rp.count_occurrences(text, "nope.json") == 0


def test_a_synthetic_registry_never_picks_up_the_real_allowlist(tmp_path):
    """A fixture must not be judged against the real document's excuses —
    a check that cries wolf on its own fixtures gets switched off."""
    reg = _write_registry(tmp_path, "`ep_*.pt`\n")
    res = rp.sweep(registry=reg, repo_root=tmp_path)
    assert res["allow_issues"] == []
    assert res["citations"][0]["status"] == rp.NOT_A_PATH


# --------------------------------------------------------------------------
# THE STANDING CHECK — this is the one that makes the defect class visible
# --------------------------------------------------------------------------

def test_real_allowlist_parses_and_every_entry_carries_its_reason():
    allow = rp.load_allow()          # raises if any entry is malformed
    assert allow["entries"], "the allowlist should not be silently emptied"
    for e in allow["entries"]:
        assert e["reason"].strip()
        assert isinstance(e["occurrences"], int) and e["occurrences"] >= 1


def test_real_registry_has_no_dead_or_malformed_citations():
    """MODEL_REGISTRY.md is the ONLY quotable source for model facts, so a dead
    path in it is a programme-level defect, not a typo.

    If this fails: run `python tools/registry_paths.py --only-bad`. Fix the
    citation by verifying what it was MEANT to point at. Do NOT point it at a
    plausible lookalike, and do NOT add it to the allowlist to make this pass —
    an unrepairable citation goes in as `unresolved`, which needs the ratchet
    raised deliberately.
    """
    res = rp.sweep()
    bad = {k: v for k, v in res["counts"].items()
           if k in (rp.MISSING, rp.NOT_A_PATH)}
    offenders = [c["citation"] for c in res["citations"]
                 if c["status"] in (rp.MISSING, rp.NOT_A_PATH)]
    assert not bad, f"{bad} — {offenders}"
    assert not res["allow_issues"], res["allow_issues"]
    assert not res["ratchet_exceeded"]


def test_real_registry_exit_code_is_not_a_hard_failure():
    assert rp.exit_code(rp.sweep()) != 1
