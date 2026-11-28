"""RETR-2026-09-26-YAWMASK lives in the leaderboard's INPUTS, not in a hand edit (Master Mind decision, batch 15).

⛔ WHY. The tip's LEADERBOARD.md carried the corrected refcv3 LATERAL yaw row TYPED INTO the generated section
(``da5760e2`` changed the page and no input). A rebuild overwrites that section by design, so it would have printed the
retracted "+0.1700 → floor WINS" again. The decision: DERIVE a new input record — the banked ARM with only the masked
yaw cells replaced by the mask package's ``yaw_fixed`` values — point the leaderboard at it, and keep the banked
record byte-identical. Pinned here:

* the derived record is EXACTLY the derivation of its inputs, and the banked record is untouched;
* a rebuild from inputs renders the corrected row, and its values equal the tip's hand-typed row VALUE FOR VALUE —
  compared by an independent regex parser, against the tip's row copied verbatim as a literal;
* REGRESSION ARM: pointing the input back at the old ARM shows "+0.1700 → ⛔ floor WINS" and FAILS the comparison;
* one ``--discard-hand-edits`` rebuild over a page carrying the hand-typed row drops only lines whose regenerated twin
  is present, and the hand-typed row's twin carries its values.
"""
from __future__ import annotations

import copy
import importlib
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

_TE = Path(__file__).resolve().parents[1]
REPO = _TE.parent
for _p in (str(_TE), str(_TE / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import derive_yawmask_record as DY                          # noqa: E402
from taniteval.leaderboard import render, sources           # noqa: E402
from taniteval.leaderboard.build import build as run_build  # noqa: E402  (the package re-exports the FUNCTION)
from taniteval.leaderboard.build import dropped_lines, section_state  # noqa: E402

BUILD = importlib.import_module("taniteval.leaderboard.build")
CFG = sources.load_config(REPO)
PAGE = REPO / CFG["target_md"]

#: the tip's hand-typed row (``da5760e2``), VERBATIM — the literal the regenerated row is compared against
HAND = ("| LATERAL | `LAT_yaw_rate_mae_radps` | **−0.0100 [−0.0168, −0.0037] → model WINS** (masked cell, n 4,524; "
        "unmasked it read +0.1700 [+0.1006, +0.2534] → floor WINS, `RETR-2026-09-26-YAWMASK`) | **+0.0064 [+0.0023, "
        "+0.0107] → ⛔ floor WINS** (masked, n 4,535; unmasked +0.1849 [+0.1162, +0.2683]) | 4,823 windows; 4,524 / 4,535 "
        "with a valid pair |")
ROW_PREFIX = "| LATERAL | `LAT_yaw_rate_mae_radps` |"
BANKED_ARM_BLOB = "15dd65041f528680723e9eb1672eb949b662ce80"       # MEASURED 2026-09-26; = the mask package's banked_blob

# ------------------------------------------------------------------ an INDEPENDENT reader of a family row's values
_NUM = re.compile(r"[+−-]?\d+\.\d{4}")
_N = re.compile(r"\bn (\d{1,3}(?:,\d{3})*)")
_VERDICT = re.compile(r"model WINS|floor WINS|TIE")


def values(row: str) -> dict:
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    floors = []
    for c in cells[2:-1]:
        floors.append({"nums": [float(x.replace("−", "-")) for x in _NUM.findall(c)],
                       "n": [int(x.replace(",", "")) for x in _N.findall(c)],
                       "verdicts": _VERDICT.findall(c)})
    ncol = re.sub(r"`[^`]*`", "", cells[-1])
    return {"floors": floors, "ncol": [int(x.replace(",", "")) for x in re.findall(r"\d{1,3}(?:,\d{3})+|\d+", ncol)],
            "retr": "RETR-2026-09-26-YAWMASK" in row}


def values_match(mine: str, hand: str) -> bool:
    """Every value the hand-typed row states is present, equal and in place in ``mine``. A verdict the hand row omits
    (it gave none for the unmasked second cell) is not required."""
    a, b = values(mine), values(hand)
    if len(a["floors"]) != len(b["floors"]) or a["ncol"] != b["ncol"] or a["retr"] != b["retr"]:
        return False
    for x, y in zip(a["floors"], b["floors"]):
        if x["nums"] != y["nums"] or x["n"] != y["n"] or x["verdicts"][:len(y["verdicts"])] != y["verdicts"]:
            return False
    return True


SECTION = "#### refcv3 — final step 40,284 (T1 read)"      # other models' sections carry a LATERAL yaw row too


def _yaw_index(lines: list) -> int:
    """The refcv3 @40,284 section's yaw row — never the first yaw row on the page."""
    h = [k for k, ln in enumerate(lines) if ln.startswith(SECTION)]
    assert len(h) == 1, h
    return next(k for k in range(h[0] + 1, len(lines)) if lines[k].startswith(ROW_PREFIX))


def _yaw_row(md: str) -> str:
    lines = md.split("\n")
    return lines[_yaw_index(lines)]


def _page_md(cfg=CFG) -> str:
    return render.render_md(BUILD.load_model(REPO, cfg), cfg)


# ------------------------------------------------------------------ the derived record
def test_the_banked_arm_record_is_untouched():
    assert DY.git_blob_id(REPO / DY.ARM) == BANKED_ARM_BLOB


def test_the_derived_record_is_exactly_the_derivation_of_its_inputs():
    banked = json.loads((REPO / DY.OUT).read_text(encoding="utf-8"))
    assert banked == DY.derive(REPO)


def test_only_the_seven_yaw_cells_and_the_provenance_differ():
    a = json.loads((REPO / DY.ARM).read_text(encoding="utf-8"))
    d = json.loads((REPO / DY.OUT).read_text(encoding="utf-8"))
    diffs = []

    def walk(x, y, p):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if k not in x or k not in y:
                    diffs.append(f"{p}/{k}")
                else:
                    walk(x[k], y[k], f"{p}/{k}")
        elif x != y:
            diffs.append(p)
    walk(a, d, "")
    cells = {"/".join(q.split("/")[:7]) for q in diffs if q.startswith("/refcv3/families_paired/")}
    assert len(cells) == 7 and all(c.endswith("/lateral/LAT_yaw_rate_mae_radps") for c in cells)
    assert [q for q in diffs if not q.startswith("/refcv3/families_paired/")] == ["/_derived"]
    assert sorted(d["_derived"]["changed"]) == sorted(cells)


def test_fields_the_mask_did_not_rederive_are_nulled_not_left_stale():
    d = json.loads((REPO / DY.OUT).read_text(encoding="utf-8"))
    for ptr in d["_derived"]["changed"]:
        cell = d
        for k in ptr.strip("/").split("/"):
            cell = cell[k]
        assert cell["ci95"] is None and cell["p_delta_gt0"] is None
        assert cell["_masked_by"] == "RETR-2026-09-26-YAWMASK" and set(cell["_unmasked"]) == set(DY.FIELDS)
    assert len(d["_derived"]["stale_not_rederived"]) == 18        # 6 arms x (MAE, n_steps, ci component)


def test_the_derivation_refuses_a_record_it_was_not_computed_from(tmp_path):
    for rel in (DY.ARM, DY.MASK):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, tmp_path / rel)
    arm = json.loads((tmp_path / DY.ARM).read_text(encoding="utf-8"))
    arm["refcv3"]["families_paired"]["paired_os_minus_ha0"]["families"]["lateral"]["LAT_yaw_rate_mae_radps"]["delta"] = 0.1701
    (tmp_path / DY.ARM).write_text(json.dumps(arm), encoding="utf-8")
    with pytest.raises(DY.DeriveError):                          # a different blob: refused before any cell is read
        DY.derive(tmp_path)


def test_the_blob_guard_alone_refuses_a_record_changed_outside_the_yaw_cells(tmp_path):
    """Every yaw cell still matches the mask's banked view, but the record is NOT the one the fix was computed on (an
    unrelated field moved). Only the blob guard can see that — the per-cell guard would pass it."""
    for rel in (DY.ARM, DY.MASK):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, tmp_path / rel)
    arm = json.loads((tmp_path / DY.ARM).read_text(encoding="utf-8"))
    arm["n_windows"] = arm["n_windows"] + 1
    (tmp_path / DY.ARM).write_text(json.dumps(arm), encoding="utf-8")
    with pytest.raises(DY.DeriveError, match="different record"):
        DY.derive(tmp_path)


def test_the_derivation_refuses_when_the_mask_packages_banked_view_disagrees(tmp_path):
    """The SAME record (blob matches), but the mask package's own 'banked' view of one cell is wrong: the per-cell
    guard must refuse — the fix is applied to the cell it was computed from, or not at all."""
    for rel in (DY.ARM, DY.MASK):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, tmp_path / rel)
    mask = json.loads((tmp_path / DY.MASK).read_text(encoding="utf-8"))
    k = "paired_os_minus_ha0 | lateral | LAT_yaw_rate_mae_radps"
    mask["yaw_banked"][k][0] = 0.1701
    (tmp_path / DY.MASK).write_text(json.dumps(mask), encoding="utf-8")
    with pytest.raises(DY.DeriveError, match="was not computed from"):
        DY.derive(tmp_path)


# ------------------------------------------------------------------ the page, rebuilt from inputs
def test_the_yaw_row_rebuilt_from_inputs_equals_the_tips_hand_typed_row_value_for_value():
    row = _yaw_row(_page_md())
    assert values_match(row, HAND), (row, values(row), values(HAND))


def test_REGRESSION_the_old_arm_input_shows_the_retracted_value_and_fails_the_comparison():
    cfg = copy.deepcopy(CFG)
    ent = next(e for e in cfg["internal"] if e["id"] == "refcv3")
    ent["path"] = DY.ARM                                         # the banked, unmasked record
    row = _yaw_row(_page_md(cfg))
    assert "+0.1700 [+0.1006, +0.2534] → ⛔ floor WINS" in row
    assert not values_match(row, HAND)


def test_one_discard_rebuild_drops_only_lines_with_regenerated_twins(tmp_path):
    host = tmp_path / "LEADERBOARD.md"
    text = PAGE.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(nl)
    lines[_yaw_index(lines)] = HAND                              # the tip's state: the correction as a HAND EDIT
    host.write_bytes(nl.join(lines).encode("utf-8"))
    old_state, old_lines = section_state(host.read_bytes().decode("utf-8"), CFG)
    assert old_state in ("legacy-unfingerprinted", "hand-edited")
    with pytest.raises(sources.SourceError) as e:
        run_build(md_path=host, write_html=False)
    first = str(e.value).split("first: ")[1].split(" || ")[0]
    assert HAND.startswith(first), first                         # the hand-typed correction is listed FIRST
    run_build(md_path=host, write_html=False, discard_hand_edits=True)
    new = host.read_bytes().decode("utf-8").replace("\r\n", "\n")
    gone = dropped_lines(old_lines, new)

    def key(ln):
        if ln.startswith("|"):
            return tuple(c.strip() for c in ln.strip("|").split("|")[:2])
        return re.split(r"[:\d]", ln, maxsplit=1)[0]
    new_lines = new.split("\n")
    orphans = [ln for ln in gone if not any(key(x) == key(ln) for x in new_lines)]
    assert not orphans, f"dropped with no regenerated twin: {orphans[:3]}"
    assert HAND in gone and values_match(_yaw_row(new), HAND)
