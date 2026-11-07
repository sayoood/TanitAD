"""PREREG_M1_PEDGEFIT's BESIDE rows (edge-class P / R / F @ 0.3 m, ``nf_edge03_*``): the EVAL-side counts, the floor, and the VERDICT's reporting of them.

* EVAL: ``_bworld`` is a 220 x 40 hand raster whose every count is derived below; read at a thresholds file (pedge levels 80 / 60 / 40 / 30 per read band) the
  evaluator must emit exactly the hand [nE, hitE, nXJ, hitX] per band, and the pooled 5-20 m row by COUNTS (never the mean of the two bands);
* MUTATION: an evaluator that IGNORES the pedge level (``MapThresholds.level`` -> 128) must make the guard go RED;
* FLOOR: ``nearfield_rows`` carries the rows only when BOTH sides are thresholded;
* VERDICT: rows present for M1 / M2a (and ONLY those arms), outside clauses 1-3 (not a veto, not ``missing_uncovered`` on a floor without them, not
  informative), a deliberately separated-WORSE beside row does not change a KEEP (nor a separated-better row rescue a DROP), NOT-COMPUTABLE with the
  reason on a read without them.

THE HAND WORLD (``_bworld``; file A with flags, file B = the same GT with an all-zero pedge plane; one episode each):
  band 0-5 m  (rows 0-49,   level 80): GT edge row 20 cols 10-19 (10); GT sidewalk row 26 cols 10-19 (judged, 6 rows off the edge: not within 0.3 m);
        pedge 90 on both rows -> flags 20, judged 20, hits 10;  recalled 10 of 10.                         A: [nE 10, hitE 10, nXJ 20, hitX 10]
  band 5-10 m (rows 50-99,  level 60): GT edge rows 70 and 90, cols 5-14 (20); GT sidewalk row 71 cols 5-14 = pedge 70 (judged, ONE row off the edge: a hit);
        a NOCLS blob (gt 0) row 80 cols 20-24 = 70 (flagged, NOT judged: free).  Row 90 is never recalled.   A: [20, 10, 10, 10]
  band 10-20 m (rows 100-199, level 40): GT edge row 120 cols 0-29 (30); pedge 50 on row 120 cols 0-9 (10 flags ON the edge) and on GT-sidewalk row 150 cols
        0-19 (20 judged flags far from any edge).  Recall needs the DILATION: the 10 flags reach edge cols 0-12 (3 cells right of col 9, left clipped) = 13.   A: [30, 13, 30, 10]
  band 20-40 m (rows 200-399, level 30): no GT edge; pedge 50 on GT-sidewalk row 205 cols 0-4 (5 judged flags) and on an UNSEEN strip (gt 255) row 212
        cols 30-34 (flagged, not judged).                                                                 A: [0, 0, 5, 0]
  B (no flag): [10, 0, 0, 0] [20, 0, 0, 0] [30, 0, 0, 0] [0, 0, 0, 0].
  A + B (the pooled blocks):  0-5 P 1/2 R 1/2 F 1/2;  5-10 P 1 R 1/4 F 2/5;  10-20 P 1/3 R 13/60 F 26/99;  20-40 P 0, R and F undefined (no GT edge: NaN);
  5-20 pooled BY COUNTS [nE 100, hitE 23, nXJ 40, hitX 20] -> P 1/2 R 23/100 F 23/73  (the mean of the two band F's would be 0.3313, of the two P's 2/3)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r9_e1_eval as E                                   # noqa: E402
import r9_e1_floor as FL                                 # noqa: E402
import r9_e1_verdict as V                                # noqa: E402
from test_r9_e1_verdict import base_floor, frow, neutral, with_delta, LV      # noqa: E402,F401  (the verdict suite's literal-row builders)

LEVELS = (80, 60, 40, 30)
HAND_A = np.array([[10, 10, 20, 10], [20, 10, 10, 10], [30, 13, 30, 10], [0, 0, 5, 0]], float)
HAND_B = np.array([[10, 0, 0, 0], [20, 0, 0, 0], [30, 0, 0, 0], [0, 0, 0, 0]], float)
NAN = float("nan")
HAND_PRF = {"0_5": (1 / 2, 1 / 2, 1 / 2), "5_10": (1.0, 1 / 4, 2 / 5), "10_20": (1 / 3, 13 / 60, 26 / 99), "20_40": (0.0, NAN, NAN), "5_20": (1 / 2, 23 / 100, 23 / 73)}
EDGE_KEYS = ["nf_edge03_%s_%s" % (k, b) for b in ("0_5", "5_10", "10_20", "20_40", "5_20") for k in ("prec", "rec", "f")]


def _bworld(path, flags=True):
    gt = np.zeros((220, 40), np.uint8)
    pe = np.zeros((220, 40), np.uint8)
    gt[20, 10:20] = 5
    gt[26, 10:20] = 7
    gt[70, 5:15] = 5
    gt[90, 5:15] = 5
    gt[71, 5:15] = 7
    gt[80, 20:25] = 0
    gt[120, 0:30] = 5
    gt[150, 0:20] = 7
    gt[205, 0:5] = 7
    gt[212, 30:35] = 255
    if flags:
        pe[20, 10:20] = 90
        pe[26, 10:20] = 90
        pe[71, 5:15] = 70
        pe[80, 20:25] = 70
        pe[120, 0:10] = 50
        pe[150, 0:20] = 50
        pe[205, 0:5] = 50
        pe[212, 30:35] = 50
    np.savez_compressed(path, pdrv=np.zeros_like(pe), pedge=pe, psw=np.zeros_like(pe), code=np.zeros_like(pe), gt=gt)


def _bdir(tmp_path, name):
    d = tmp_path / name / "rast"
    d.mkdir(parents=True)
    _bworld(d / "e000_w0007.npz", True)
    _bworld(d / "e001_w0007.npz", False)
    return d


def _bthr(tmp_path, name="thr.json", levels=LEVELS):
    p = tmp_path / name
    p.write_text(json.dumps({"schema": E.THR_SCHEMA, "bands_m": [[0, 5], [5, 10], [10, 20], [20, 40]], "planes": {"pedge": list(levels)}, "provenance": {}}), encoding="utf-8")
    return p


def _same(a, b):
    return (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, abs=1e-12)


def _eval_guard(tmp_path):
    """True iff the evaluator, read at the thresholds file, gives exactly the hand counts per band and the hand pooled P / R / F (what the MUTATION must turn False)."""
    d = _bdir(tmp_path, "g")
    thr = E.load_map_thresholds(_bthr(tmp_path))
    files = sorted(str(p) for p in d.glob("*.npz"))
    names, ea = E.edge_tol_per_file(files, 1, thr)
    if names != ["e000_w0007.npz", "e001_w0007.npz"] or not (np.array_equal(ea[0], HAND_A) and np.array_equal(ea[1], HAND_B)):
        return False
    _, acc, _, info = E.nearfield_per_file_ex_edge(files, 1, "psw", thresholds=thr, edge_tol=True)
    if acc.shape != (2, 4, E.NF_COLS + E.EDGE_TOL_N) or info["edge_tol"]["levels"] != {"0-5": 80, "5-10": 60, "10-20": 40, "20-40": 30}:
        return False
    s = E.nearfield_stats(list(acc))
    return all(_same(s["nf_edge03_%s_%s" % (k, b)], HAND_PRF[b][i]) for b in HAND_PRF for i, k in enumerate(("prec", "rec", "f")))


# ================================================================================================================= EVAL
def test_eval_beside_counts_and_prf_at_a_thresholds_file_equal_the_hand_values(tmp_path):
    assert _eval_guard(tmp_path) is True


def test_pooled_5_20_is_by_counts_not_the_mean_of_the_bands(tmp_path):
    d = _bdir(tmp_path, "p")
    thr = E.load_map_thresholds(_bthr(tmp_path))
    _, acc, _, _ = E.nearfield_per_file_ex_edge(sorted(str(p) for p in d.glob("*.npz")), 1, "psw", thresholds=thr, edge_tol=True)
    s = E.nearfield_stats(list(acc))
    assert s["nf_edge03_f_5_20"] == pytest.approx(23 / 73) and s["nf_edge03_prec_5_20"] == pytest.approx(1 / 2) and s["nf_edge03_rec_5_20"] == pytest.approx(23 / 100)
    assert abs(s["nf_edge03_f_5_20"] - (s["nf_edge03_f_5_10"] + s["nf_edge03_f_10_20"]) / 2) > 0.01
    assert abs(s["nf_edge03_prec_5_20"] - (s["nf_edge03_prec_5_10"] + s["nf_edge03_prec_10_20"]) / 2) > 0.1
    assert math.isnan(s["nf_edge03_rec_20_40"]) and math.isnan(s["nf_edge03_f_20_40"]) and s["nf_edge03_prec_20_40"] == 0.0       # no GT edge: undefined, never 0


def test_unset_read_is_exactly_the_old_read_and_carries_no_edge_row(tmp_path):
    d = _bdir(tmp_path, "u")
    files = sorted(str(p) for p in d.glob("*.npz"))
    old = E.nearfield_per_file_ex(files, 1, "psw")
    new = E.nearfield_per_file_ex_edge(files, 1, "psw", edge_tol=False)
    assert old[0] == new[0] and np.array_equal(old[1], new[1], equal_nan=True) and np.array_equal(old[2], new[2]) and old[3] == new[3] and "edge_tol" not in new[3]
    assert not [k for k in E.nearfield_stats(list(new[1])) if k.startswith("nf_edge03")]


def test_MUTATION_eval_ignores_the_pedge_level_makes_the_guard_red(tmp_path, monkeypatch):
    (tmp_path / "ok").mkdir()
    (tmp_path / "bad").mkdir()
    assert _eval_guard(tmp_path / "ok") is True                                                                      # GREEN on the real evaluator
    monkeypatch.setattr(E.MapThresholds, "level", lambda self, plane, lo, hi: E.FIXED_LEVEL)                          # the level of every plane is 128
    assert _eval_guard(tmp_path / "bad") is False                                                                    # RED: nothing is flagged at 128 (pedge <= 90)


# ================================================================================================================= FLOOR
def test_floor_rows_carry_the_beside_rows_only_when_both_sides_are_thresholded(tmp_path):
    a, b = _bdir(tmp_path, "a"), _bdir(tmp_path, "b")
    t = _bthr(tmp_path)
    rows, meta = FL.nearfield_rows(a, b, boot=20, seed=0, thr0=str(t), thr1=str(t))
    assert [k for k in rows if k.startswith("nf_edge03")] == sorted(EDGE_KEYS) and "thresholds" in meta
    r = rows["nf_edge03_f_5_20"]
    assert r["higher_is_better"] is True and r["family"] == "MAP" and "BESIDE" in r["group"] and r["c0"] == pytest.approx(23 / 73) and r["c1"] == pytest.approx(23 / 73)
    assert rows["nf_edge03_rec_20_40"]["undefined"] is True                                                           # no GT edge in the band: reported undefined, not 0
    p: list = []
    none, _ = FL.nearfield_rows(a, b, boot=20, seed=0, problems=p)
    assert not [k for k in none if k.startswith("nf_edge03")] and p == []                                             # fixed 128: the old rows, none of these
    p2: list = []
    one, _ = FL.nearfield_rows(a, b, boot=20, seed=0, problems=p2, thr0=str(t))                                       # ONE side thresholded is not a floor
    assert one == {} and p2 and "ONE side only" in p2[0]


def test_MUTATION_floor_that_drops_the_edge_counts_makes_the_row_check_red(tmp_path, monkeypatch):
    a, b = _bdir(tmp_path, "a"), _bdir(tmp_path, "b")
    t = str(_bthr(tmp_path))
    ok = lambda: any(k.startswith("nf_edge03") for k in FL.nearfield_rows(a, b, boot=20, seed=0, thr0=t, thr1=t)[0])           # noqa: E731
    assert ok() is True
    real = E.nearfield_per_file_ex_edge
    monkeypatch.setattr(E, "nearfield_per_file_ex_edge", lambda *a_, **k_: real(*a_, **dict(k_, edge_tol=False)))
    assert ok() is False


# ================================================================================================================= VERDICT
def _edge_floor(fl, c0=0.5, c1=0.75):
    """A floor read on the thresholded protocol: ``fl`` plus the 15 nf_edge03 rows (floor |c1 - c0| = 0.25 each)."""
    rows = dict(fl["rows"])
    rows.update({k: frow(k, c0, c1, family="MAP") for k in EDGE_KEYS})
    return dict(fl, rows=rows)


def _edge_pair(pair, delta=0.0, separated=False, keys=EDGE_KEYS):
    p = dict(pair)
    for k in keys:
        p[k] = {"ref": 0.5, "arm": 0.5 + delta, "delta": delta, "ci_lo": delta - 0.01, "ci_hi": delta + 0.01, "separated": separated, "undefined": False}
    return p


def _core(v):
    """Everything a verdict DECIDES on, plus the mapping-free blocks the keep rule fills: must be identical with and without the beside rows."""
    return {k: v[k] for k in ("verdict", "failing_clauses", "not_computable", "reads", "clause3", "alternative_reading", "named_guards")}


def test_beside_rows_are_present_for_m1_and_m2a_and_only_for_them():
    fl = _edge_floor(base_floor(lv=True))
    pair = _edge_pair(with_delta(neutral(fl), LV, +0.6), delta=-0.3, separated=True)
    for arm, ref in (("M1", "C0"), ("M2a", "M1")):
        v = V.judge(arm, ref, fl, pair, ctx={"rasters": True, "thresholds": True})
        b = v["beside_pedgefit"]
        assert b["title"] == "Beside (PREREG_M1_PEDGEFIT rows 2/3 selector)" and b["decides"] is False
        want = ([V.D3_LANE_KEY, V.D3_BOUNDARY_KEY] if arm == "M1" else []) + \
            [k for band in V.EDGE03_BANDS for k in ("nf_edge03_f_" + band, "nf_edge03_prec_" + band, "nf_edge03_rec_" + band)]          # M1: + its 2 design beside rows
        assert [r["key"] for r in b["rows"]] == want and len(b["rows"]) == (17 if arm == "M1" else 15)
        f = next(r for r in b["rows"] if r["key"] == "nf_edge03_f_5_20")
        assert f["state"] == "READ" and f["role"] == "beside" and f["delta"] == pytest.approx(-0.3) and f["separated"] is True and f["direction"] == "worse"
        assert f["ci_lo"] == pytest.approx(-0.31) and f["ci_hi"] == pytest.approx(-0.29) and f["reading"] == "SEPARATED WORSE" and f["floor_context"]["abs_c1_minus_c0"] == 0.25
        assert b["selector"]["separated_worse"] is True and b["selector"]["key"] == "nf_edge03_f_5_20"
        assert [r["id"] for r in v["reads"]][:1] == ["%s.primary" % arm] and not [r for r in v["reads"] if "edge03" in r["id"]]       # reads stays what it was
    assert V.judge("M1", "C0", fl, pair)["beside_pedgefit"]["selector"]["applies"].startswith("PREREG_M1_PEDGEFIT: if M1-vs-C0 is a DROP on clause 1-2")
    assert V.judge("M2a", "M1", fl, pair)["beside_pedgefit"]["selector"]["applies"].startswith("reported for M2a vs M1")
    for arm, s in V.ARMS.items():                                                  # no other arm gains the section
        if s["kind"] != "floor" and arm not in ("M1", "M2a", "M1N20"):
            ref = s["ref"] or (s["allowed"][0] if s["allowed"] else None)
            if ref is not None:
                assert "beside_pedgefit" not in V.judge(arm, ref, fl, neutral(fl)), arm
    assert [m["id"] for m in V.ARMS["M1"]["metrics"] if not m["clause3"]] == [r["id"] for r in V.judge("M1", "C0", fl, pair)["beside_pedgefit"]["rows"]]


def test_beside_rows_are_outside_clause_3_even_when_the_floor_lacks_or_carries_them():
    D3 = {V.D3_LANE_KEY, V.D3_BOUNDARY_KEY}                                       # hold_r9_d3beside's rows: beside only, like the edge rows
    assert V.beside_only_keys() == set(EDGE_KEYS) | D3 and not (V.table_keys() & (set(EDGE_KEYS) | D3)) and not (V.clause3_scope() & (set(EDGE_KEYS) | D3))
    old = base_floor(lv=True)                                                      # a floor read before the rows existed
    v = V.judge("M1", "C0", old, _edge_pair(with_delta(neutral(old), LV, +0.6), -0.3, True))
    assert v["verdict"] == "KEEP" and v["clause3"]["missing_uncovered"] == [] and not [k for k in v["clause3"]["missing_from_floor"] if "edge03" in k]
    e0 = next(r for r in v["beside_pedgefit"]["rows"] if r["key"].startswith("nf_edge03"))         # M1's first EDGE row (its 2 D3 rows come first)
    assert e0["state"] == "READ" and e0["floor_context"] is None        # reported without a floor, never a veto
    new = _edge_floor(old)
    w = V.judge("M1", "C0", new, _edge_pair(with_delta(neutral(new), LV, +0.6), -0.3, True))
    assert w["verdict"] == "KEEP" and w["clause3"]["vetoes"] == [] and not [x for x in w["clause3"]["informative_worse"] if "edge03" in x["metric"]]
    assert w["clause3"]["checked"] == v["clause3"]["checked"] and w["clause3"]["no_floor"] == v["clause3"]["no_floor"]            # the sweep did not even look at them


def _worse_beside_keeps(judge, arm="M1", ref="C0"):
    """Guard: a DELIBERATELY separated-worse (beyond 2 x floor) beside edge row leaves a KEEP a KEEP, with no clause failing."""
    fl = _edge_floor(base_floor(lv=True))
    good = with_delta(neutral(fl), LV, +0.6)
    v = judge(arm, ref, fl, _edge_pair(good, delta=-0.9, separated=True))
    return v["verdict"] == "KEEP" and v["failing_clauses"] == [] and v["clause3"]["vetoes"] == []


def test_a_separated_worse_beside_row_does_not_change_keep_or_drop():
    assert _worse_beside_keeps(V.judge) is True and _worse_beside_keeps(V.judge, "M2a", "M1") is True
    fl = _edge_floor(base_floor(lv=True))
    for arm, ref in (("M1", "C0"), ("M2a", "M1")):
        for pair in (with_delta(neutral(fl), LV, +0.6), with_delta(neutral(fl), LV, +0.1), with_delta(neutral(fl), LV, +0.6, separated=False),
                     with_delta(with_delta(neutral(fl), LV, +0.6), "ap2m_all", -0.625), neutral(fl)):
            base = V.judge(arm, ref, fl, pair)
            for d, sep in ((-0.9, True), (+0.9, True), (-0.9, False)):
                got = V.judge(arm, ref, fl, _edge_pair(pair, d, sep))
                assert _core(got) == _core(base), (arm, ref, d, sep)                  # KEEP / DROP / failing clauses / reads / clause 3 / guards: all unchanged
    assert V.judge("M1", "C0", fl, _edge_pair(with_delta(neutral(fl), LV, +0.1), +0.9, True))["verdict"] == "DROP"                  # a better beside row cannot rescue a DROP


def test_MUTATION_a_beside_row_in_clause_3_makes_the_guard_red(monkeypatch):
    assert _worse_beside_keeps(V.judge) is True
    for a in ("M1", "M2a", "M1N20"):                                                # the rows are re-declared as ordinary clause-3 table keys
        for i, m in enumerate(V.ARMS[a]["metrics"]):
            if not m["clause3"]:
                monkeypatch.setitem(V.ARMS[a]["metrics"][i], "clause3", True)
    assert not (V.beside_only_keys() & set(EDGE_KEYS))
    fl = _edge_floor(base_floor(lv=True))
    v = V.judge("M1", "C0", fl, _edge_pair(with_delta(neutral(fl), LV, +0.6), -0.9, True))
    assert v["verdict"] == "DROP" and 3 in [f["clause"] for f in v["failing_clauses"]]                                          # RED: with them in scope the veto fires


def test_not_computable_with_the_reason_on_a_read_without_the_rows():
    fl = base_floor(lv=True)
    pair = with_delta(neutral(fl), LV, +0.6)
    why = {}
    for tag, ctx in (("none", None), ("norast", {"rasters": False, "thresholds": False}), ("fixed128", {"rasters": True, "thresholds": False}),
                     ("thr", {"rasters": True, "thresholds": True})):
        v = V.judge("M1", "C0", fl, pair, ctx=ctx)
        allr = v["beside_pedgefit"]["rows"]
        rows = [r for r in allr if r["key"].startswith("nf_edge03")]
        assert v["verdict"] == "KEEP" and len(allr) == 17 and len(rows) == 15 and {r["state"] for r in allr} == {"NOT-COMPUTABLE"}
        assert v["beside_pedgefit"]["selector"]["state"] == "NOT-COMPUTABLE" and v["beside_pedgefit"]["selector"]["separated_worse"] is None
        why[tag] = rows[0]["why"]
        d3why = {r["why"] for r in allr if not r["key"].startswith("nf_edge03")}
        assert len(d3why) == 1 and ("no raster dirs" in d3why.pop() if tag in ("none", "norast") else "--map-beside-d3" in [r["why"] for r in allr][0])
    assert "fixed 128" in why["fixed128"] and "no raster dirs" in why["norast"] and "emitted no such row" in why["thr"] and "THRESHOLDED" in why["none"]
    un = _edge_pair(pair)
    un["nf_edge03_f_5_20"] = {"ref": None, "arm": None, "delta": None, "ci_lo": None, "ci_hi": None, "separated": False, "undefined": True}
    r = next(x for x in V.judge("M1", "C0", fl, un)["beside_pedgefit"]["rows"] if x["key"] == "nf_edge03_f_5_20")
    assert r["state"] == "UNDEFINED" and "never read as 0" in r["why"]


def _vdict(arm, ref, floor, pair, ctx):
    v = {"schema": V.SCHEMA, "tier": V.TIER, "arm": arm, "keep_rule": V.KEEP_RULE, "status": "OK", "estimator": {"answers": "x"}, "ref": ref, "mapping": V._mapping_entry(arm),
         "families": V._families(), "floor": {"sha256": "0" * 64, "status": "OK", "n_rows": len(floor["rows"]), "estimator": {"B": 1, "seed": 0}}, "inputs": {"eval_arm": {"sha256": "0" * 64}},
         "n_windows": 1, "n_clips": 1, "gate": {}, "next_lever_on_drop": V.ARMS[arm]["next_lever"], "ambiguities": []}
    v.update(V.judge(arm, ref, floor, pair, ctx=ctx))
    return v


def test_the_markdown_has_its_own_short_beside_section_and_the_rest_is_unchanged():
    fl = _edge_floor(base_floor(lv=True))
    pair = _edge_pair(with_delta(neutral(fl), LV, +0.6), -0.3, True)
    md = V.render_md(_vdict("M1", "C0", fl, pair, {"rasters": True, "thresholds": True}))
    head, _, tail = md.partition("## Beside (PREREG_M1_PEDGEFIT rows 2/3 selector)")
    assert tail and "nf_edge03" not in head and head.count("\n## ") >= 4
    assert tail.count("| edge-class") == 15 and "`nf_edge03_f_5_20`" in tail and "separated worse: **yes**" in tail and "NEVER decisive" in tail
    nc = V.render_md(_vdict("M1", "C0", base_floor(lv=True), with_delta(neutral(base_floor(lv=True)), LV, +0.6), {"rasters": True, "thresholds": False}))
    assert "**NOT-COMPUTABLE** (15 row(s)" in nc and "fixed 128" in nc and "Selector `nf_edge03_f_5_20`: **NOT-COMPUTABLE**" in nc
    plain = V.render_md(_vdict("B1", "C0", base_floor(), with_delta(neutral(base_floor()), "dx_0_40_mean", -0.75), None))
    assert "Beside (PREREG_M1_PEDGEFIT" not in plain                                # an arm without these rows prints no such section


def test_mapping_names_the_beside_rows_with_their_source_line():
    md = V.mapping_md().replace("`", "")
    assert md.count("nf_edge03_f_5_20 (r9_e1_eval.py:697)") == 3 and md.count("nf_edge03_rec_20_40 (r9_e1_eval.py:697)") == 3                # M1, M2a and M1N20 (E1-NEAR20)
    assert [m["clause3"] for m in V._mapping_entry("M1")["metrics"] if "clause3" in m] == [False] * 17 and "clause3" not in V._mapping_entry("B1")["metrics"][0]
    assert md.count("lane_tiou05_0_20 (r9_e1_eval.py:875)") == 2 and md.count("drv_bnd_rec03_0_20 (r9_e1_eval.py:880)") == 2              # M1 and M1N20


def test_d3_rows_are_read_from_the_verdicts_own_paired_rows_and_never_decide():
    """MM 2026-11-07 (hold_r9_d3wire): with the paired D3 rows present, each reads its delta / CI / direction; a SEPARATED WORSE lane row changes
    neither the verdict nor the clauses; without the flag (ctx) the absence names --map-beside-d3."""
    fl = base_floor(lv=True)
    pair = with_delta(neutral(fl), LV, +0.6)
    base = V.judge("M1", "C0", fl, pair, ctx={"rasters": True, "thresholds": True})
    for k, d in ((V.D3_LANE_KEY, -0.2), (V.D3_BOUNDARY_KEY, +0.1)):
        pair[k] = {"ref": 0.5, "arm": 0.5 + d, "delta": d, "ci_lo": d - 0.01, "ci_hi": d + 0.01, "separated": True}
    v = V.judge("M1", "C0", fl, pair, ctx={"rasters": True, "thresholds": True, "beside_d3": True})
    r = {x["key"]: x for x in v["beside_pedgefit"]["rows"]}
    assert r[V.D3_LANE_KEY]["state"] == "READ" and r[V.D3_LANE_KEY]["direction"] == "worse" and r[V.D3_LANE_KEY]["delta"] == pytest.approx(-0.2)
    assert r[V.D3_BOUNDARY_KEY]["state"] == "READ" and r[V.D3_BOUNDARY_KEY]["direction"] == "better"
    assert v["verdict"] == base["verdict"] == "KEEP" and v["reads"] == base["reads"] and v["failing_clauses"] == base["failing_clauses"]
    assert "--map-beside-d3" in {x["key"]: x for x in base["beside_pedgefit"]["rows"]}[V.D3_LANE_KEY]["why"]
