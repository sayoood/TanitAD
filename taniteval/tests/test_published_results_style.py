"""`published_results.json` keeps its UTF-8 serialisation, and the navtest headline stays in it.

⛔ WHY THIS EXISTS (MEASURED 2026-09-26). I added the DiffusionDrive row with
``json.dumps(..., indent=1)``, whose ``ensure_ascii`` defaults to **True**. That rewrote all 191
non-ASCII characters as ``\\uXXXX`` escapes — content-identical, but it flipped the whole file's style
and cost the Master Mind **four section numbers** that its tokeniser lost behind ``\\u00a7``.

⭐ The mistake underneath is worth more than the fix: I "matched the file's style" by looking at my own
``json.dumps`` output of a row, which escapes by default — so I inferred the file's encoding from my own
printing, not from the file. Same class as reading a truncated ``--help`` as a missing flag: **the
artifact of a probe is not the world.**

Left unpinned this file flips style on every edit, because half the writers default to ``ensure_ascii``
and half do not — and a 191-line escaping diff hides the one row that actually changed.
"""
from __future__ import annotations

import json
import pathlib

import pytest

F = pathlib.Path(__file__).resolve().parents[2] / "products" / "P7-TanitEval" / "benchmarks" / "published_results.json"
ESCAPE = chr(92) + "u"          # a literal backslash-u, written without one
SECTION_SIGN = chr(0xA7)


@pytest.fixture(scope="module")
def raw() -> str:
    if not F.exists():
        pytest.skip(f"not on this checkout: {F}")
    return F.read_text(encoding="utf-8")


def test_it_parses(raw):
    assert isinstance(json.loads(raw)["results"], list)


def test_no_unicode_escapes(raw):
    """The pin: written with ensure_ascii=False. ⛔ Any writer that defaults to True goes RED here."""
    n = raw.count(ESCAPE)
    assert n == 0, (f"{n} unicode escapes: this file is UTF-8 and must be written with "
                    f"ensure_ascii=False, or it flips style on every edit")


def test_control_the_file_really_does_hold_non_ascii(raw):
    """⭐ Without this control the test above passes trivially on a pure-ASCII file — it would then be
    pinning nothing at all, which is how a green test ends up guarding a broken thing."""
    assert sum(1 for ch in raw if ord(ch) > 127) > 0


def test_the_section_sign_is_greppable(raw):
    """The concrete cost that motivated the pin: behind an escape, a search for it finds nothing."""
    assert SECTION_SIGN in raw


def test_the_navtest_community_headline_is_present(raw):
    """Master Mind item 4, pinned as a LITERAL: this file is the leaderboard generator's source, and
    the row's absence is exactly the gap that made the generated board omit the headline."""
    rows = json.loads(raw)["results"]
    hits = [r for r in rows
            if r.get("protocol") == "PDMS_v1_navtest" and "DiffusionDrive" in str(r.get("system", ""))]
    assert len(hits) == 1, "expected exactly one DiffusionDrive row under PDMS_v1_navtest"
    r = hits[0]
    assert r["value"] == 88.1
    assert r["source"]["library_key"] == "2411.15139"      # verified in the banked PDF, p.6 Tab. 1
    assert r["source"]["page"] == 6
    assert r["evidence_class"] == "PUBLISHED"
