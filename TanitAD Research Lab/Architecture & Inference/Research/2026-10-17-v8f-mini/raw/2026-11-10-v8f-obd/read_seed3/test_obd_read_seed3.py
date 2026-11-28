"""Tests of obd_read_seed3.py on PLANTED per-window arrays (no dumps, no torch, no GPU): the per-seed labels of the new pairs, every outcome of the combination rule (all 81 label
patterns against an independent oracle, the named 4-0 / 3-1 / 2-2 / 2-1-1 cases, the 'at least one per init' clause on a planted unbalanced layout), the triggered recipe, the floor G
cross-check, the existing-result sha256 pin, the beside rows and the rendering.  Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_obd_read_seed3.py
The module under test is read from env OBD_SEED3_SRC when set (the mutation runner), else obd_read_seed3.py beside this file."""
import importlib.util
import itertools
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "read"))
SRC = os.environ.get("OBD_SEED3_SRC", os.path.join(HERE, "obd_read_seed3.py"))
_spec = importlib.util.spec_from_file_location("obd_read_seed3_under_test", SRC)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
R, C = S.OR, S.C
CI = R.load_ci()
NB = 40
N_EP, PER_EP = 40, 20
N = N_EP * PER_EP
EID = np.repeat(np.arange(N_EP), PER_EP)
EIDS = {0: EID, 1: EID}
rng = np.random.default_rng(23)
BASE = {c: (0.8 + rng.normal(0, 0.05, N)) if c in R.ACCURACY else (2.0 + rng.normal(0, 0.3, N)) for c in R.CELLS}
TAC_FDE = f"{R.TAC}:fde_m"
# a planted label: ADDS = operative ADE better by 0.5; COSTS = a tactical FDE worse by 0.5; NEUTRAL = nothing moved
DELTA = {"BAND-ADDS": {R.OP_ADE: -0.5}, "BAND-COSTS": {TAC_FDE: 0.5}, "BAND-NEUTRAL": {}}
LABS = S.LABELS


def cells(delta=None, shift=0.0):
    d = delta or {}
    return {c: BASE[c] + d.get(c, 0.0) + shift for c in R.CELLS}


def build(labels):
    """labels {'s0','s1','s10','s11'} -> (arms, existing).  Floor G reads 0.01 on every cell (BAND_rep = BAND_s0 + 0.01)."""
    arms, ex_cells = {}, {}
    band_s0 = cells(DELTA[labels["s0"]])
    arms["OBD-BAND_s0"] = band_s0
    arms["OBD-BAND_rep"] = {c: v + 0.01 for c, v in band_s0.items()}
    G = R.floors_from(arms["OBD-BAND_rep"], arms["OBD-BAND_s0"])
    for s in ("s0", "s1"):
        ex_cells[s] = R.compare_cells(CI, cells(DELTA[labels[s]]), cells(), EID, G, NB)
    for s in ("s10", "s11"):
        arms[f"OBD-CTL_{s}"] = cells()
        arms[f"OBD-BAND_{s}"] = cells(DELTA[labels[s]])
    existing = {"labels": {s: R.label_band(ex_cells[s]) for s in ("s0", "s1")}, "cells": ex_cells, "floor_G": G, "result": {}}
    return arms, existing


def run(labels):
    arms, existing = build(labels)
    return S.read_seed3(CI, arms, EIDS, existing, NB)


def oracle(items):
    """independent statement of the amendment rule on (label, init) pairs"""
    for lab in LABS:
        held = [i for l, i in items if l == lab]
        if len(held) >= 3 and 0 in held and 1 in held:
            return lab
    return "BAND-UNSETTLED"


# ---------------------------------------------------------------------------------------------------------------------------------------- per-seed labels of the new pairs
def test_planted_labels_are_read_per_new_pair():
    r = run({"s0": "BAND-NEUTRAL", "s1": "BAND-NEUTRAL", "s10": "BAND-ADDS", "s11": "BAND-COSTS"})
    assert r["labels"] == {"s0": "BAND-NEUTRAL", "s1": "BAND-NEUTRAL", "s10": "BAND-ADDS", "s11": "BAND-COSTS"}
    assert [c["cell"] for c in r["per_seed"]["s11"]["named_costs"]] == [TAC_FDE]
    assert [c["cell"] for c in r["per_seed"]["s10"]["named_gains"]] == [R.OP_ADE]
    assert r["per_seed"]["s10"]["n_cells_read"] == 24 and r["per_seed"]["s11"]["n_cells_read"] == 24


def test_floor_g_is_the_replicate_gap_and_gates_the_new_pairs():
    arms, existing = build({k: "BAND-NEUTRAL" for k in S.SEEDS})
    assert all(abs(v - 0.01) < 1e-9 for v in S.OR.floors_from(arms["OBD-BAND_rep"], arms["OBD-BAND_s0"]).values())
    arms["OBD-BAND_s10"] = cells({R.OP_ADE: -0.005})          # separated but below the 0.01 floor: not a gain
    r = S.read_seed3(CI, arms, EIDS, existing, NB)
    assert r["per_seed"]["s10"]["cells"][R.OP_ADE]["separated"] and not r["per_seed"]["s10"]["cells"][R.OP_ADE]["gain"] and r["labels"]["s10"] == "BAND-NEUTRAL"


def test_new_pairs_use_their_own_inits_eids():
    arms, existing = build({k: "BAND-NEUTRAL" for k in S.SEEDS})
    eid1 = np.repeat(np.arange(N_EP), PER_EP)[::-1].copy()          # a different clustering for init 1
    r = S.read_seed3(CI, arms, {0: EID, 1: eid1}, existing, NB)
    assert r["per_seed"]["s11"]["cells"][R.OP_ADE]["n_episodes"] == N_EP


def test_missing_new_arm_or_floor_is_not_yet_not_a_label():
    arms, existing = build({k: "BAND-ADDS" for k in S.SEEDS})
    del arms["OBD-BAND_s11"]
    r = S.read_seed3(CI, arms, EIDS, existing, NB)
    assert r["labels"]["s11"] is None and r["outcome"].startswith("NOT-YET") and r["recipe_of_record"] is None
    arms, existing = build({k: "BAND-ADDS" for k in S.SEEDS})
    del arms["OBD-BAND_rep"]
    r = S.read_seed3(CI, arms, EIDS, existing, NB)
    assert r["labels"]["s10"] is None and r["outcome"].startswith("NOT-YET")


# ---------------------------------------------------------------------------------------------------------------------------------------- the combination rule
def test_every_one_of_the_81_label_patterns_matches_the_oracle():
    for pat in itertools.product(LABS, repeat=4):
        lab = dict(zip(S.SEEDS, pat))
        r = run(lab)
        assert r["labels"] == lab
        assert r["outcome"] == oracle([(lab[s], C.SEED_LABELS[s]) for s in S.SEEDS]), pat


@pytest.mark.parametrize("pat,want", [
    (("BAND-ADDS",) * 4, "BAND-ADDS"), (("BAND-NEUTRAL",) * 4, "BAND-NEUTRAL"), (("BAND-COSTS",) * 4, "BAND-COSTS"),
    # 3-1 with both inits among the three (always the case with two seeds per init)
    (("BAND-ADDS", "BAND-ADDS", "BAND-ADDS", "BAND-COSTS"), "BAND-ADDS"),
    (("BAND-ADDS", "BAND-COSTS", "BAND-ADDS", "BAND-ADDS"), "BAND-ADDS"),
    (("BAND-COSTS", "BAND-ADDS", "BAND-ADDS", "BAND-ADDS"), "BAND-ADDS"),
    (("BAND-ADDS", "BAND-ADDS", "BAND-COSTS", "BAND-ADDS"), "BAND-ADDS"),
    (("BAND-COSTS", "BAND-COSTS", "BAND-COSTS", "BAND-NEUTRAL"), "BAND-COSTS"),
    (("BAND-NEUTRAL", "BAND-NEUTRAL", "BAND-ADDS", "BAND-NEUTRAL"), "BAND-NEUTRAL"),
    # 2-2 splits
    (("BAND-ADDS", "BAND-COSTS", "BAND-ADDS", "BAND-COSTS"), "BAND-UNSETTLED"),         # the existing pair split, repeated: the smoke layout
    (("BAND-ADDS", "BAND-ADDS", "BAND-COSTS", "BAND-COSTS"), "BAND-UNSETTLED"),
    (("BAND-ADDS", "BAND-COSTS", "BAND-COSTS", "BAND-ADDS"), "BAND-UNSETTLED"),
    (("BAND-ADDS", "BAND-NEUTRAL", "BAND-NEUTRAL", "BAND-ADDS"), "BAND-UNSETTLED"),
    # 2-1-1
    (("BAND-ADDS", "BAND-ADDS", "BAND-COSTS", "BAND-NEUTRAL"), "BAND-UNSETTLED"),
    (("BAND-ADDS", "BAND-COSTS", "BAND-NEUTRAL", "BAND-ADDS"), "BAND-UNSETTLED"),
])
def test_named_outcomes(pat, want):
    r = run(dict(zip(S.SEEDS, pat)))
    assert r["outcome"] == want


def test_at_least_one_per_init_clause_on_an_unbalanced_layout():
    """Three equal labels at ONE init is not a majority across inits.  With two seeds per init the clause is met by every 3-of-4 majority (see README), so the clause is exercised
    on the generic rule with a planted unbalanced layout; this is the test that goes RED when the clause is dropped."""
    A, X = "BAND-ADDS", "BAND-COSTS"
    assert S.combine4([(A, 0), (A, 0), (A, 0), (X, 1)]) == "BAND-UNSETTLED"          # 3 of 4 but all at init 0
    assert S.combine4([(X, 0), (A, 1), (A, 1), (A, 1)]) == "BAND-UNSETTLED"          # 3 of 4 but all at init 1
    assert S.combine4([(A, 0), (A, 0), (X, 1), (X, 1)]) == "BAND-UNSETTLED"          # no 3 of 4 at all
    assert S.combine4([(A, 0), (A, 1), (A, 1), (X, 0)]) == "BAND-ADDS"               # 3 of 4 with both inits
    assert S.combine4([(A, 0), (A, 0), (A, 0), (A, 1)]) == "BAND-ADDS"               # 4 of 4, both inits present


def test_the_real_layout_is_two_seeds_per_init():
    assert C.SEED_LABELS == {"s0": 0, "s1": 1, "s10": 0, "s11": 1}
    assert C.init_of_tag("OBD-CTL_s11") == 1 and C.init_of_tag("OBD-BAND_s10") == 0 and C.init_of_tag("OBD-BAND_s1") == 1 and C.init_of_tag("OBD-BAND_rep") == 0


def test_an_unread_label_is_not_yet_never_unsettled():
    assert S.combine4([(None, 0), ("BAND-ADDS", 1), ("BAND-ADDS", 0), ("BAND-ADDS", 1)]).startswith("NOT-YET")
    with pytest.raises(ValueError):
        S.combine4([("BAND-ADDS", 0)] * 3)


# ---------------------------------------------------------------------------------------------------------------------------------------- trigger
@pytest.mark.parametrize("pat,recipe,phrase", [
    (("BAND-ADDS",) * 4, "OBD-BAND", "O/T recipe of record"),
    (("BAND-NEUTRAL",) * 4, "OBD-CTL", "not carried"),
    (("BAND-COSTS",) * 4, "OBD-CTL", "not carried"),
    (("BAND-ADDS", "BAND-COSTS", "BAND-ADDS", "BAND-COSTS"), "OBD-CTL", "conservative branch"),
])
def test_triggered_recipe(pat, recipe, phrase):
    r = run(dict(zip(S.SEEDS, pat)))
    assert r["recipe_of_record"] == recipe and phrase in r["trigger"]


def test_unsettled_keeps_the_operative_gain_beside_with_the_drift_caveat():
    r = run(dict(zip(S.SEEDS, ("BAND-ADDS", "BAND-COSTS", "BAND-ADDS", "BAND-COSTS"))))
    note = r["beside"]["operative_only_gain_note"]
    assert "code-drift caveat" in note and "VERDICT_OBD.md" in note and "s10" in note


# ---------------------------------------------------------------------------------------------------------------------------------------- controls
def test_floor_crosscheck():
    G = {c: 0.01 for c in R.CELLS}
    assert S.floor_crosscheck(G, dict(G)) == []
    other = dict(G)
    other[R.OP_ADE] = 0.02
    assert any(R.OP_ADE in p for p in S.floor_crosscheck(G, other))
    other = dict(G)
    other[R.OP_ADE] = None
    assert S.floor_crosscheck(G, other)
    assert S.floor_crosscheck(None, G)


def test_existing_result_sha_is_pinned(tmp_path):
    p = tmp_path / "RESULT_OBD.json"
    p.write_text(json.dumps({"smoke": False}))
    with pytest.raises(AssertionError, match="sha256"):
        S.load_existing(str(p))
    real = S.EXISTING_RESULT
    if os.path.isfile(real):
        ex, sha = S.load_existing(real)
        assert sha == S.EXISTING_SHA256 and ex["labels"] == {"s0": "BAND-ADDS", "s1": "BAND-COSTS"} and len(ex["floor_G"]) == 24 and ex["cells"]["s0"][R.OP_ADE]["gain"]
        assert S.floor_crosscheck(ex["floor_G"], ex["floor_G"]) == []


def test_existing_result_that_is_smoke_or_invalid_is_refused(tmp_path):
    for body in ({"smoke": True}, {"smoke": False, "outcome": "INVALID"}):
        p = tmp_path / "r.json"
        p.write_text(json.dumps(body))
        with pytest.raises(AssertionError):
            S.load_existing(str(p), want_sha=R.sha_file(str(p)))


def test_identity_control_through_the_same_function():
    assert R.identity_control(CI, EID)["pass"]


# ---------------------------------------------------------------------------------------------------------------------------------------- beside + render
def test_spread_rows():
    r = run(dict(zip(S.SEEDS, ("BAND-ADDS", "BAND-COSTS", "BAND-ADDS", "BAND-COSTS"))))
    sp = r["beside"]["per_cell_spread_BAND_minus_CTL"]
    assert len(sp) == 24 and sp[R.OP_ADE]["n"] == 4
    assert abs(sp[R.OP_ADE]["s10"] + 0.5) < 1e-9 and abs(sp[R.OP_ADE]["range"] - 0.5) < 1e-9 and abs(sp[TAC_FDE]["s11"] - 0.5) < 1e-9


def test_drift_sizes_from_the_existing_result():
    cell = lambda d: {"status": "OK", "delta": d, "lo": d - 1, "hi": d + 1, "separated": abs(d) > 0.5}  # noqa: E731
    res = {"beside": {"code_drift_NULL_minus_DAGOACC": {"s0": {R.OP_ADE: cell(0.9), f"{R.OP_ADE[:-5]}fde_m": cell(-2.0), "TAC-LAT": cell(0.1)}}}}
    d = S.drift_sizes(res)["s0"]
    assert d["largest_by_family"]["PRIMARY"]["delta"] == -2.0 and d["n_cells_separated"] == 2 and "TACTICAL" in d["largest_by_family"]
    assert S.drift_sizes({}) == {}


def test_render_md_carries_the_four_labels_the_families_and_the_pause():
    r = run(dict(zip(S.SEEDS, ("BAND-ADDS", "BAND-COSTS", "BAND-ADDS", "BAND-COSTS"))))
    rep = {"outcome_line": r["outcome"], "card_sha256": R.CARD_SHA256, "code_of_record": R.CODE_OF_RECORD, "existing_result_sha256": S.EXISTING_SHA256, "tier": "T1-kin",
           "evidence_class": "MEASURED", "estimator": "x", "smoke": True, "controls": {"problems": []}, "band": r}
    md = S.render_md(rep)
    for needle in ("SMOKE", "PAUSED", "BAND-UNSETTLED", "## Family PRIMARY", "## Family LONGITUDINAL", "## Family LATERAL", "## Family TACTICAL", "| s10 | M2b s0 |", "| s11 | M2b s1 |",
                   "Recipe of record:** OBD-CTL", "Per-cell spread"):
        assert needle in md, needle
    for banned in ("void", "superseded", "retracted", "formerly", "correction", "withdrawn", "no longer"):
        assert banned not in md.lower()
