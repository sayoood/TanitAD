"""Tests of obd_read.py on PLANTED per-window arrays (no dumps, no torch, no GPU): every committed outcome (DETACH-KEEPS / DETACH-COSTS / BAND-ADDS / BAND-NEUTRAL / BAND-COSTS /
SEEDS-DISAGREE), the NULL-parity fallback and the unknown-parity case, an exact-zero control, sub-floor deltas, accuracy-cell direction, missing floors, the controls' read-backs,
and (tests/run_mutation_floor.py) the floor-comparison mutation.  Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_obd_read.py
The module under test is read from env OBD_READ_SRC when set (the mutation runner), else obd_read.py beside this file."""
import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("OBD_READ_SRC", os.path.join(HERE, "obd_read.py"))
_spec = importlib.util.spec_from_file_location("obd_read_under_test", SRC)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
CI = R.load_ci()
NB = 40
N_EP, PER_EP = 40, 20
N = N_EP * PER_EP
EID = np.repeat(np.arange(N_EP), PER_EP)
EIDS = {0: EID, 1: EID}
rng = np.random.default_rng(11)
BASE = {c: (0.8 + rng.normal(0, 0.05, N)) if c in R.ACCURACY else (2.0 + rng.normal(0, 0.3, N)) for c in R.CELLS}
TAC_ADE, TAC_FDE, TAC_LON = f"{R.TAC}:ade_m", f"{R.TAC}:fde_m", "TAC-LON"


def arm(delta=None, base=None, shift=0.0):
    """BASE (or `base`) moved by delta[cell] on its cells (+ shift on every cell).  A positive delta is a worse error and a better accuracy."""
    b = BASE if base is None else base
    d = delta or {}
    return {c: b[c] + d.get(c, 0.0) + shift for c in R.CELLS}


def world(ctl=(None, None), band=(None, None), parity="holds", null_s0=None, null_s1_present=False, with_rep=True, with_band_rep=True):
    """ctl = (delta_s0, delta_s1) of OBD-CTL against DAG-OACC; band = (delta_s0, delta_s1) of OBD-BAND against OBD-CTL.  Both floors read 0.01 on every cell."""
    A = {"DAGOACC_s0": arm(), "DAGOACC_s1": arm()}
    A["OBD-NULL_s0"] = arm(null_s0)
    if null_s1_present:
        A["OBD-NULL_s1"] = arm(null_s0)
    if with_rep:
        A["OBD-NULL_rep"] = arm(null_s0, shift=0.01)
    for s in (0, 1):
        A[f"OBD-CTL_s{s}"] = arm(ctl[s])
        A[f"OBD-BAND_s{s}"] = arm(ctl[s])
        for c, v in (band[s] or {}).items():
            A[f"OBD-BAND_s{s}"][c] = A[f"OBD-CTL_s{s}"][c] + v
    if with_band_rep:
        A["OBD-BAND_rep"] = {c: A["OBD-BAND_s0"][c] + 0.01 for c in R.CELLS}
    return R.read_all(CI, A, EIDS, parity, NB)


def lab(rep, q, s):
    return rep[q]["per_seed"][s]["label"]


def test_24_cells_and_families():
    assert len(R.CELLS) == 24 and len(set(R.CELLS)) == 24
    assert sum(c.startswith(R.OP) for c in R.CELLS) == 11 and sum(c.startswith(R.TAC) for c in R.CELLS) == 11 and R.ACCURACY == ("TAC-LAT", "TAC-LON")
    assert {R.FAMILY[c.split(":")[-1]] for c in R.CELLS} == {"PRIMARY", "LONGITUDINAL", "LATERAL", "TACTICAL"}


def test_floors_are_the_replicate_gaps_and_read_at_seed0():
    rep = world()
    assert all(abs(v - 0.01) < 1e-9 for v in rep["floors"]["F"].values()) and all(abs(v - 0.01) < 1e-9 for v in rep["floors"]["G"].values())
    # the floor of seed 1's cells is the SAME seed-0 floor
    assert rep["detach"]["per_seed"][1]["cells"][TAC_ADE]["floor"] == rep["detach"]["per_seed"][0]["cells"][TAC_ADE]["floor"]


def test_exact_zero_control():
    rep = world()
    for q in ("detach", "band"):
        for s in (0, 1):
            ps = rep[q]["per_seed"][s]
            assert ps["n_cells_read"] == 24
            for c, v in ps["cells"].items():
                assert v["delta"] == 0.0 and not v["gain"] and not v["cost"], (q, s, c, v)
    assert rep["detach"]["outcome"] == "DETACH-KEEPS" and rep["band"]["outcome"] == "BAND-NEUTRAL"


def test_detach_keeps_and_band_adds_with_committed_next_step():
    rep = world(band=({R.OP_ADE: -0.5}, {R.OP_ADE: -0.5}))
    assert rep["detach"]["outcome"] == "DETACH-KEEPS" and rep["band"]["outcome"] == "BAND-ADDS"
    assert any("recipe of record" in n and "HWM-PLAN" in n for n in rep["next_step"]), rep["next_step"]


def test_detach_keeps_band_neutral_next_step():
    rep = world()
    assert any("OBD-CTL's argv becomes the recipe of record" in n for n in rep["next_step"])


def test_subfloor_delta_is_not_a_cost_or_a_gain():
    rep = world(ctl=({TAC_ADE: 0.005}, {TAC_ADE: 0.005}), band=({R.OP_ADE: -0.005}, {R.OP_ADE: -0.005}))
    c = rep["detach"]["per_seed"][0]["cells"][TAC_ADE]
    assert c["separated"] and c["delta"] > 0 and not c["above_floor"] and not c["cost"], c
    assert rep["detach"]["outcome"] == "DETACH-KEEPS"
    o = rep["band"]["per_seed"][0]["cells"][R.OP_ADE]
    assert o["separated"] and o["delta"] < 0 and not o["gain"], o
    assert rep["band"]["outcome"] == "BAND-NEUTRAL"


def test_detach_costs_names_every_costing_cell_with_delta_ci_floor():
    rep = world(ctl=({TAC_ADE: 0.5, TAC_FDE: 0.4}, {TAC_ADE: 0.5}))
    assert rep["detach"]["outcome"] == "DETACH-COSTS"
    names = {c["cell"] for c in rep["detach"]["per_seed"][0]["named_costs"]}
    assert names == {TAC_ADE, TAC_FDE}
    nc = rep["detach"]["per_seed"][0]["named_costs"][0]
    assert nc["delta"] > 0 and nc["lo"] <= nc["delta"] <= nc["hi"] and nc["floor"] == pytest.approx(0.01)
    assert any("--w-tac" in n for n in rep["next_step"])


def test_accuracy_direction_a_fall_is_a_cost_and_a_rise_is_a_gain():
    rep = world(ctl=({TAC_LON: -0.2}, {TAC_LON: -0.2}))
    assert rep["detach"]["outcome"] == "DETACH-COSTS" and rep["detach"]["per_seed"][0]["cells"][TAC_LON]["cost"]
    rep = world(ctl=({TAC_LON: +0.2}, {TAC_LON: +0.2}))
    assert rep["detach"]["outcome"] == "DETACH-KEEPS" and rep["detach"]["per_seed"][0]["cells"][TAC_LON]["gain"]


def test_an_improvement_of_the_detach_is_not_a_cost():
    rep = world(ctl=({R.OP_ADE: -0.5}, {R.OP_ADE: -0.5}))
    assert rep["detach"]["outcome"] == "DETACH-KEEPS" and rep["detach"]["per_seed"][0]["named_gains"]


def test_detach_seeds_disagree():
    rep = world(ctl=({TAC_ADE: 0.5}, None))
    assert rep["detach"]["outcome"].startswith("SEEDS-DISAGREE") and "s0 DETACH-COSTS" in rep["detach"]["outcome"] and "s1 DETACH-KEEPS" in rep["detach"]["outcome"]
    assert any("third training seed" in n for n in rep["next_step"])


def test_band_outcomes():
    adds = world(band=({R.OP_ADE: -0.5}, {R.OP_ADE: -0.5}))
    neutral = world(band=({TAC_ADE: -0.5}, {TAC_ADE: -0.5}))                       # a tactical gain does not make BAND-ADDS: the card keys it on operative ADE
    costs = world(band=({R.OP_ADE: -0.5, TAC_FDE: 0.5}, {R.OP_ADE: -0.5}))         # operative gain but a cost at seed 0 only
    both_cost = world(band=({R.OP_ADE: -0.5, TAC_FDE: 0.5}, {R.OP_ADE: -0.5, TAC_LON: -0.2}))
    assert adds["band"]["outcome"] == "BAND-ADDS" and neutral["band"]["outcome"] == "BAND-NEUTRAL"
    assert lab(costs, "band", 0) == "BAND-COSTS" and lab(costs, "band", 1) == "BAND-ADDS" and costs["band"]["outcome"].startswith("SEEDS-DISAGREE")
    assert both_cost["band"]["outcome"] == "BAND-COSTS"
    assert {c["cell"] for c in both_cost["band"]["per_seed"][0]["named_costs"]} == {TAC_FDE}


def test_band_seeds_disagree_adds_vs_neutral():
    rep = world(band=({R.OP_ADE: -0.5}, None))
    assert rep["band"]["outcome"] == "SEEDS-DISAGREE (s0 BAND-ADDS; s1 BAND-NEUTRAL)", rep["band"]["outcome"]


def test_one_seed_decides_nothing():
    A_ = {"DAGOACC_s0": arm(), "OBD-NULL_s0": arm(), "OBD-NULL_rep": arm(shift=0.01), "OBD-CTL_s0": arm(), "OBD-BAND_s0": arm(), "OBD-BAND_rep": arm(shift=0.01)}
    rep = R.read_all(CI, A_, EIDS, "holds", NB)
    assert rep["detach"]["outcome"].startswith("NOT-YET") and rep["band"]["outcome"].startswith("NOT-YET")
    assert rep["detach"]["per_seed"][0]["label"] == "DETACH-KEEPS" and rep["detach"]["per_seed"][1]["label"] is None


def test_missing_floor_arm_decides_nothing():
    rep = world(ctl=({TAC_ADE: 0.5}, {TAC_ADE: 0.5}), with_rep=False)
    assert rep["floors"]["F"] is None and rep["detach"]["outcome"].startswith("NOT-YET")        # no floor, no cost is read
    assert rep["band"]["outcome"] == "BAND-NEUTRAL"                                              # G is present: the band question is unaffected
    rep = world(with_band_rep=False)
    assert rep["band"]["outcome"].startswith("NOT-YET") and rep["detach"]["outcome"] == "DETACH-KEEPS"


def test_null_parity_fails_reads_against_the_null_of_the_same_seed():
    drift = {TAC_ADE: 0.5}                      # the code drifted: NULL itself is 0.5 worse than DAG-OACC on tactical ADE; CTL inherits it
    holds = world(ctl=(drift, drift), null_s0=drift, parity="holds")
    fails = world(ctl=(drift, drift), null_s0=drift, parity="fails", null_s1_present=True)
    assert holds["detach"]["outcome"] == "DETACH-COSTS"                  # against DAG-OACC it looks like a detach cost
    assert fails["detach"]["outcome"] == "DETACH-KEEPS"                  # against OBD-NULL it is not
    assert fails["detach"]["per_seed"][0]["reference"].startswith("OBD-NULL_s0") and fails["detach"]["per_seed"][1]["reference"].startswith("OBD-NULL_s1")
    d = fails["beside"]["code_drift_NULL_minus_DAGOACC"]["s0"][TAC_ADE]
    assert d["delta"] == pytest.approx(0.5) and d["separated"]            # NULL minus DAG-OACC is reported as a code-drift row
    assert any("NULL parity FAILED" in n for n in fails["next_step"])
    assert fails["band"]["outcome"] == holds["band"]["outcome"]           # the band question does not use the parity state


def test_null_parity_fails_without_null_s1_is_not_yet_for_seed_1():
    rep = world(parity="fails", null_s1_present=False)
    assert rep["detach"]["per_seed"][1]["label"] is None and rep["detach"]["outcome"].startswith("NOT-YET")
    assert rep["detach"]["per_seed"][0]["label"] == "DETACH-KEEPS"


def test_parity_unknown_blocks_the_detach_question_only():
    rep = world(parity="unknown", band=({R.OP_ADE: -0.5}, {R.OP_ADE: -0.5}))
    assert rep["detach"]["outcome"].startswith("NOT-YET") and rep["band"]["outcome"] == "BAND-ADDS"


def test_nan_windows_are_masked_and_an_all_nan_cell_is_not_applicable():
    a, b = arm(), arm()
    a[TAC_ADE] = a[TAC_ADE].copy()
    a[TAC_ADE][:50] = np.nan
    a["TAC-LAT"] = np.full(N, np.nan)
    cells = R.compare_cells(CI, a, b, EID, None, NB)
    assert cells[TAC_ADE]["n_windows"] == N - 50 and cells[TAC_ADE]["delta"] == 0.0
    assert cells["TAC-LAT"]["status"] == "NOT-APPLICABLE" and not cells["TAC-LAT"]["cost"]


def test_cell_without_a_floor_is_never_a_gain_or_a_cost():
    a, b = arm({TAC_ADE: 0.5}), arm()
    fl = {c: 0.01 for c in R.CELLS}
    fl[TAC_ADE] = None
    cells = R.compare_cells(CI, a, b, EID, fl, NB)
    assert cells[TAC_ADE]["separated"] and not cells[TAC_ADE]["cost"]


def test_identity_control():
    r = R.identity_control(CI, EID)
    assert r["pass"] and r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0


def test_parity_state_from_the_chains_file():
    assert R.parity_state_from({"append_obd_null_s1": False}) == "holds"
    assert R.parity_state_from({"append_obd_null_s1": True, "file_sha_equal": False, "tensor_equal": False}) == "fails"
    assert R.parity_state_from(None) == "unknown" and R.parity_state_from({}) == "unknown" and R.parity_state_from({"append_obd_null_s1": "False"}) == "unknown"


def test_manifest_check():
    man = {"tag": "OBD-CTL_s0", "commit": R.CODE_OF_RECORD, "files": {"ckpt_last.pt": {"sha256": "a" * 64}}}
    assert R.check_manifest("OBD-CTL_s0", man, "a" * 64) == []
    assert R.check_manifest("OBD-CTL_s0", man, "b" * 64)
    assert R.check_manifest("OBD-CTL_s0", dict(man, commit="202849da" + "0" * 32), "a" * 64)
    assert R.check_manifest("OBD-CTL_s1", man, "a" * 64)
    assert R.check_manifest("OBD-CTL_s0", {"files": {}}, "a" * 64)


def _sc(**kw):
    sc = {"declared_vs_built": "PASS", "levers": {"dagger_bank_sha256": "197a68bb" + "0" * 56, "w_op": 0.3333333, "w_tac_traj": 0.6666667}, "otband_probe": {"detach_upper": True, "otband": "on"}}
    sc.update(kw)
    return sc


def test_selfcheck_bank_identity_and_detach_read_back():
    assert R.check_selfcheck("OBD-BAND_s0", _sc(), 0)[0] == []
    assert R.check_selfcheck("OBD-BAND_s0", _sc(declared_vs_built="FAIL"), 0)[0]
    assert R.check_selfcheck("OBD-BAND_s1", _sc(), 1)[0]                                   # s0's DAgger bank on a seed-1 arm (control 4)
    assert R.check_selfcheck("OBD-CTL_s0", _sc(otband_probe={"detach_upper": False}), 0)[0]
    assert R.check_selfcheck("OBD-CTL_s0", _sc(otband_probe=None), 0)[0]                    # a CTL arm that did not build the detach did not test it
    bad = _sc()
    bad["levers"]["w_op"] = 1.0
    assert R.check_selfcheck("OBD-BAND_s0", bad, 0)[0]
    assert R.check_selfcheck("OBD-NULL_s0", {"declared_vs_built": "PASS", "levers": {"dagger_bank_sha256": "197a68bb" + "0" * 56}}, 0)[0] == []
    assert R.check_selfcheck("OBD-NULL_s0", _sc(), 0)[0]                                    # a NULL arm whose probe says detach is on
    assert R.check_selfcheck("OBD-NULL_s0", {"declared_vs_built": "PASS", "levers": {}}, 0)[1]    # layout not confirmed: a note, not a problem


def test_render_md_names_costs_and_pauses_strategic():
    rep = world(ctl=({TAC_ADE: 0.5}, {TAC_ADE: 0.5}))
    rep.update({"outcome": "x", "card_sha256": R.CARD_SHA256, "code_of_record": R.CODE_OF_RECORD, "tier": "T1-kin", "evidence_class": "MEASURED", "estimator": "e",
                "parity_state": "holds", "STRATEGIC": {"ruling": "PI item 25"}, "controls": {"problems": []}})
    md = R.render_md(rep)
    assert "COST tactical_2_6s:ade_m" in md and "PAUSED" in md and "DETACH-COSTS" in md
