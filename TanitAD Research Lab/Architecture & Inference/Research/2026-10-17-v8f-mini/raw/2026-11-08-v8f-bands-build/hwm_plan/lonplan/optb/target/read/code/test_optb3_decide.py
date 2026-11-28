"""Label-logic tests for optb3_decide (no data). Run: venv python -m pytest -q test_optb3_decide.py  (CUDA_VISIBLE_DEVICES=-1)."""
import sys
sys.path.insert(0, "F:/TanitAD-scratch/optb3_read")
import optb3_decide as D


def mk(better=(), worse=(), attr_hi=0.1, mech=True, bm=False, tr=None, ctl=True):
    qs = ("acc_mae", "yaw_mae")
    return {"T15": {q: {"better": q in better, "worse": q in worse} for q in qs}, "attr": {q: {"delta": -0.1, "lo": -0.2, "hi": attr_hi} for q in qs},
            "mech_raised": mech, "band_met_lost_T15": bm, "target_read_within_0p1": tr, "controls_hold": ctl}


def test_bw():
    assert D.bw({"delta": -0.5, "lo": -0.6, "hi": -0.1}, 0.2) == {"better": True, "worse": False}
    assert D.bw({"delta": -0.1, "lo": -0.6, "hi": -0.05}, 0.2)["better"] is False          # under the floor
    assert D.bw({"delta": 0.5, "lo": 0.1, "hi": 0.9}, 0.2) == {"better": False, "worse": True}
    assert D.bw({"delta": 0.5, "lo": -0.1, "hi": 0.9}, 0.2)["worse"] is False               # CI straddles 0


def test_labels():
    assert D.chassis_label(mk(better=("acc_mae",), attr_hi=-0.01))["label"] == "TGT-GAIN"
    assert D.chassis_label(mk(better=("acc_mae",), attr_hi=0.01))["label"] == "TGT-GAIN-UNATTRIBUTED"
    assert D.chassis_label(mk(better=("yaw_mae",), attr_hi=-0.01, mech=False))["label"] == "TGT-GAIN-UNATTRIBUTED"
    assert D.chassis_label(mk())["label"] == "TGT-MECH-ONLY"
    n = D.chassis_label(mk(mech=False, tr=True))
    assert n["label"] == "TGT-NULL" and n["null_variant"] == "(i)"
    assert D.chassis_label(mk(mech=False, tr=False))["null_variant"] == "(ii)"
    assert D.chassis_label(mk(mech=None))["label"] == "MECH-NOT-COMPUTABLE"
    assert D.chassis_label(mk(better=("acc_mae",), worse=("yaw_mae",)))["label"] == "TGT-HARM"
    assert D.chassis_label(mk(better=("acc_mae",), bm=True))["label"] == "TGT-HARM"


def test_overall():
    g = {"label": "TGT-GAIN"}
    assert D.overall_label({0: g, 1: g})["label"] == "TGT-GAIN"
    assert D.overall_label({0: g, 1: {"label": "TGT-HARM"}})["label"] == "TGT-HARM"
    o = D.overall_label({0: g, 1: {"label": "TGT-MECH-ONLY"}})
    assert o["label"] == "CHASSIS-DISAGREE" and o["less_favourable"] == "TGT-MECH-ONLY"
    assert D.overall_label({0: g})["label"] == "INCOMPLETE"


def test_mech():
    a, b = {"point": 1.5, "ci95": [1.4, 1.6]}, {"point": 1.0, "ci95": [0.9, 1.1]}
    assert D.mech_raised(a, b, 0.2)["pass"] is True
    assert D.mech_raised(a, b, 0.6)["pass"] is False                                        # gap under the floor
    assert D.mech_raised({"point": 1.5, "ci95": [1.05, 1.6]}, b, 0.2)["pass"] is False      # CIs overlap
    assert D.mech_raised(b, a, 0.2)["pass"] is False                                        # below the pair


def test_mm_rule_common_quantity():
    import optb3_decide as Dm
    a = {0: {"label": "TGT-GAIN", "attributed_quantities": ["acc_mae"], "notes": []}, 1: {"label": "TGT-GAIN", "attributed_quantities": ["yaw_mae"], "notes": []}}
    r = Dm.apply_mm_rule_common_quantity(a)
    assert r[0]["label"] == r[1]["label"] == "TGT-GAIN-UNATTRIBUTED"
    b = {0: {"label": "TGT-GAIN", "attributed_quantities": ["acc_mae"], "notes": []}, 1: {"label": "TGT-GAIN", "attributed_quantities": ["acc_mae", "yaw_mae"], "notes": []}}
    r = Dm.apply_mm_rule_common_quantity(b)
    assert r[0]["label"] == r[1]["label"] == "TGT-GAIN"
    c = {0: {"label": "TGT-GAIN", "attributed_quantities": ["acc_mae"], "notes": []}, 1: {"label": "TGT-MECH-ONLY", "notes": []}}
    r = Dm.apply_mm_rule_common_quantity(c)
    assert r[0]["label"] == "TGT-GAIN" and r[1]["label"] == "TGT-MECH-ONLY"
