"""A hand edit INSIDE the generated section is REFUSED, never overwritten silently.

⛔ WHY (2026-09-26). The RETR-2026-09-26-YAWMASK correction was typed into the generated section of the tip's
LEADERBOARD.md (commit ``da5760e2`` changed the page and no input); the page's data-model digest did not move. The
generator overwrites everything between the markers by design, so the next build would have printed the retracted
yaw-rate value again — silently, on the published page. Each section now records its own SHA-256; a build refuses to
replace a section whose fingerprint no longer matches, or one built before fingerprints existed, unless told to with
``discard_hand_edits=True`` — and the refusal lists the lines it would drop, corrections and retractions first.

All runs are on TEMPORARY copies of the real page; the real page is never written.
"""
from __future__ import annotations

import re

import pytest

from taniteval.leaderboard import render as R
from taniteval.leaderboard import sources as S
from taniteval.leaderboard.build import build as run_build          # the package re-exports the FUNCTION
from taniteval.leaderboard.build import section_state

REPO = S.REPO
CFG = S.load_config(REPO)
PAGE = REPO / CFG["target_md"]
pytestmark = pytest.mark.skipif(not PAGE.exists(), reason=f"{CFG['target_md']} not in this checkout")


@pytest.fixture
def host(tmp_path):
    h = tmp_path / "LEADERBOARD.md"
    h.write_bytes(PAGE.read_bytes())
    return h


def _edit_inside_section(h, old_fragment: str, new_fragment: str):
    text = h.read_bytes().decode("utf-8")
    b = text.index(CFG["markers"]["begin"])
    i = text.index(old_fragment, b)
    assert i < text.index(CFG["markers"]["end"]), "the fragment must be INSIDE the generated section"
    h.write_bytes((text[:i] + new_fragment + text[i + len(old_fragment):]).encode("utf-8"))


def test_a_legacy_section_is_refused_until_acknowledged_then_fingerprinted(host):
    assert section_state(host.read_bytes().decode("utf-8"), CFG)[0] == "legacy-unfingerprinted"
    with pytest.raises(S.SourceError) as e:
        run_build(md_path=host, write_html=False)
    assert "predates section fingerprints" in str(e.value) and "--discard-hand-edits" in str(e.value)
    run_build(md_path=host, write_html=False, discard_hand_edits=True)
    assert section_state(host.read_bytes().decode("utf-8"), CFG)[0] == "fingerprint-ok"
    out = run_build(md_path=host, write_html=False)                   # an untouched section rebuilds, no flag
    assert out["md_changed"] is False


def test_a_hand_edit_after_a_build_is_refused_and_named(host):
    run_build(md_path=host, write_html=False, discard_hand_edits=True)
    marker = "RETR-2099-01-01-TEST — a correction typed into the generated section"
    _edit_inside_section(host, "data-model digest", f"{marker} · data-model digest")
    assert section_state(host.read_bytes().decode("utf-8"), CFG)[0] == "hand-edited"
    assert run_build(md_path=host, check=True)["section_state"] == "hand-edited"   # --check reports, never raises
    before = host.read_bytes()
    with pytest.raises(S.SourceError) as e:
        run_build(md_path=host, write_html=False)
    msg = str(e.value)
    assert "EDITED BY HAND" in msg and marker in msg
    assert msg.index(marker) < msg.index("--discard-hand-edits")
    assert host.read_bytes() == before, "a refused build must leave the page untouched"


def _edit_section_lines(h, fn):
    """Apply ``fn(list_of_section_lines) -> list`` to the lines strictly between the markers; CRLF preserved."""
    text = h.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(nl)
    b = next(i for i, ln in enumerate(lines) if CFG["markers"]["begin"] in ln)
    e = next(i for i, ln in enumerate(lines) if CFG["markers"]["end"] in ln)
    lines[b + 1:e] = fn(lines[b + 1:e])
    h.write_bytes(nl.join(lines).encode("utf-8"))


def test_a_retraction_line_is_listed_before_ordinary_drops(host):
    """TEN ordinary edits early in the section, ONE retraction edit at its end: the message shows 8 dropped lines, so
    only the priority sort can put the retraction into it — and first."""
    run_build(md_path=host, write_html=False, discard_hand_edits=True)

    def edit(sec):
        # ORDINARY rows only: a row that itself carries a correction (e.g. the regenerated RETR-2026-09-26-YAWMASK yaw
        # row, "masked cell … RETR-…") is rightly prioritised too, and would break this test's premise
        risky = re.compile(r"RETR-|RETRACT|CORRECT|[Mm]ask")
        rows = [i for i, ln in enumerate(sec)
                if ln.startswith("| ") and not ln.startswith("| ---") and not risky.search(ln)][:10]
        assert len(rows) == 10
        for i in rows:
            sec[i] = "| ORDINARY-EDIT " + sec[i][2:]
        last = max(i for i, ln in enumerate(sec) if "data-model digest" in ln)
        sec[last] = "RETR-2099-02-02-FIRST · " + sec[last]
        return sec

    _edit_section_lines(host, edit)
    with pytest.raises(S.SourceError) as e:
        run_build(md_path=host, write_html=False)
    msg = str(e.value)
    assert "RETR-2099-02-02-FIRST" in msg and msg.index("RETR-2099-02-02-FIRST") < msg.index("ORDINARY-EDIT")


def test_the_fingerprint_line_is_the_hash_of_the_section_above_it():
    body = "<!-- BENCH:BEGIN -->\nline one\nline two"
    assert R.section_fingerprint(body) == R.section_fingerprint(body.replace("\n", "\r\n"))   # newline-blind
    assert R.section_fingerprint(body) != R.section_fingerprint(body + " ")
