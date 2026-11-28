"""⛔ EVERY NUMBER IN `P3_PREBUILD.md` MUST MATCH THE ARTIFACT IT CAME FROM.

MEASURED 2026-09-20, and this test exists because of it: correcting a truncation-order error in
that very report, I TYPED `p95 15 / max 27` into the summary table where the banked JSON said
`13 / 20`. Plausible numbers, written by hand, **inside the edit that was fixing exactly that
class of mistake**. The class recurs under maximum vigilance, so vigilance is not the fix.

⭐ The fix is mechanical: GENERATE THE SUMMARY, NEVER TYPE IT — and where a table is already
prose, CHECK it against its source. This test is that check. It parses the figures out of the
report and asserts each against `raw/p3_census_*.json`.

⛔ It is a real guard, not a formality: `stack/scripts/mutate_p3_doc_check.py` perturbs a figure
in a scratch copy of the report and this test must go RED.

⚠️ If the report legitimately changes, this test names the exact figure and its JSON key, so the
failure says what to update rather than merely that something moved.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / \
    "2026-09-19-s1-collision-gate"
DOC = PKG / "P3_PREBUILD.md"
sys.path.insert(0, str(ROOT / "stack"))

pytestmark = pytest.mark.skipif(
    not DOC.exists(), reason="NO_TREE: the research package is not present in this checkout")


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def _census(half: str) -> dict:
    return json.loads((PKG / ("raw/p3_census_%s.json" % half)).read_text(encoding="utf-8"))


def test_the_delivered_gate_ROW_matches_halfA_s_census():
    """The row I typed wrong. Every cell is asserted against its JSON key."""
    d = _census("halfA")["delivered_gate_after_pad"]
    m = re.search(r"gate \*delivered\* after the pad \(trainer's order\) \| \*\*([\d.]+)\*\* \| "
                  r"([\d.]+) \| ([\d.]+) \| (\d+) \| ([\d.]+) %", _doc())
    assert m, "the delivered-gate row is missing or reshaped — regenerate it from the JSON"
    mean, median, p95, mx, empty = m.groups()
    assert float(mean) == pytest.approx(d["mean"], abs=5e-4), "mean"
    assert float(median) == pytest.approx(d["median"]), "median"
    assert float(p95) == pytest.approx(d["p95"]), "p95 — the cell I invented as 15"
    assert int(mx) == d["max"], "max — the cell I invented as 27"
    assert float(empty) == pytest.approx(100 * d["frac_windows_empty"], abs=5e-3), "frac empty"


def test_the_halfB_summary_line_matches_halfB_s_census():
    b = _census("halfB")
    m = re.search(r"halfB, same order: \*\*([\d.]+) / ([\d.]+) / ([\d.]+) / ([\d.]+)\*\*", _doc())
    assert m, "the halfB summary line is missing or reshaped"
    raw, gate, deliv360, delivgate = (float(x) for x in m.groups())
    assert raw == pytest.approx(b["population_360"]["mean"], abs=5e-3)
    assert gate == pytest.approx(b["population_gate_relevant"]["mean"], abs=5e-4)
    assert deliv360 == pytest.approx(b["delivered_360_after_pad"]["mean"], abs=5e-3)
    assert delivgate == pytest.approx(b["delivered_gate_after_pad"]["mean"], abs=5e-4)


@pytest.mark.parametrize("half", ["halfA", "halfB"])
def test_the_pad_DROP_ROW_is_recomputable_from_the_census(half):
    """⛔ The figures whose wrong versions (0.05 % / 15.66 %) were published.

    ⚠️ AN EARLIER VERSION OF THIS TEST ASSERTED ONLY THAT THE STRING "13.91" APPEARED SOMEWHERE
    IN THE DOCUMENT. Mutation `M4` reverted the table CELL to the retracted `0.05 %` and the test
    stayed GREEN, because `13.91` still occurred in §7. An existence check cannot tell where a
    number is — so the assertion is now anchored to the ROW, and every cell in it is compared.
    """
    c = _census(half)
    pre = c["population_gate_relevant"]["mean"]
    post = c["delivered_gate_after_pad"]["mean"]
    m = re.search(r"\| %s \| ([\d.]+) \| \*\*([\d.]+)\*\* \| \*\*([\d.]+) %%\*\*" % half, _doc())
    assert m, "the %s drop row is missing or reshaped — regenerate it from the JSON" % half
    d_pre, d_post, d_drop = (float(x) for x in m.groups())
    assert d_pre == pytest.approx(pre, abs=5e-4), "pre-pad cell"
    assert d_post == pytest.approx(post, abs=5e-4), "delivered cell"
    assert d_drop == pytest.approx(100.0 * (pre - post) / pre, abs=5e-3), (
        "the row says %.2f %% but the census implies %.2f %%"
        % (d_drop, 100.0 * (pre - post) / pre))


def test_the_WRONG_figures_survive_only_INSIDE_the_retraction():
    """⛔ The old numbers must remain quoted as wrong — deleting them hides the correction —
    but must not appear anywhere a reader could take them as current."""
    doc = _doc()
    for wrong in ("10,321 of 65,918", "0.05 %", "15.66 %"):
        for line in doc.splitlines():
            if wrong in line:
                assert ("I first" in line or "I said" in line or "published" in line
                        or "wrong" in line.lower()), (
                    "a retracted figure appears outside the retraction: %r" % line[:100])
