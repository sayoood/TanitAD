"""Tests of tokprobe_read.py (card PREREG_V8F_PROOF_ADD27 with note N1). Synthetic inputs, CPU, fast. Evidence class: MEASURED (synthetic).

Run: CUDA_VISIBLE_DEVICES=-1 python -m pytest -q test_tokprobe_read.py --basetemp F:/TanitAD-scratch/v8proof/tokprobe/tmp/pt/<new name>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import tokprobe_read as R  # noqa: E402

RP, O = R.RP, R.O


@pytest.fixture(scope="module")
def ctx():
    n_ep, per = 40, 10
    rng = np.random.default_rng(0)
    ep = np.repeat(np.arange(n_ep), per)
    n = len(ep)
    return {"valid": np.ones(n, bool), "c_lab": rng.normal(0, 2.0, n), "ep10": ep, "n10": n_ep, "C10": RP.boot_counts(n_ep, 300, O.SEED)}


def test_paired_read_controls_self_swap_and_planted_shift(ctx):
    c = R.paired_read_controls(ctx)
    assert c["paired_self_reads_exactly_0"]["pass"] and c["paired_self_reads_exactly_0"]["diff"] == 0.0
    assert c["paired_swap_reads_the_negative"]["pass"]
    assert c["paired_planted_shift_read_back"]["pass"]


def test_planted_shift_is_read_back_independently_of_the_reader(ctx):
    lab = ctx["c_lab"]
    d = R.paired_m1(ctx, lab.copy(), lab + 1.0)
    exp = 1.0 - (1.0 ** 2 * len(lab)) / ((lab - lab.mean()) ** 2).sum()
    assert abs(d["diff"] - (1.0 - exp)) < 1e-9 and d["separated_pos"]


def test_a_broken_pair_read_would_be_caught(ctx):
    a = ctx["c_lab"] + np.random.default_rng(1).normal(size=len(ctx["c_lab"]))
    b = ctx["c_lab"] + np.random.default_rng(2).normal(size=len(ctx["c_lab"]))
    ab, ba = R.paired_m1(ctx, a, b), R.paired_m1(ctx, b, a)
    assert abs(ab["diff"] + ba["diff"]) < 1e-9 and ab["diff"] != 0.0
    assert R.paired_m1(ctx, a, a)["diff"] == 0.0


def _arms(r2, lo, typ="TK16"):
    return {"%s_%s" % (typ, s): {"r2": r2, "r2_ci95": [lo, r2 + 0.05]} for s in R.SEEDS}


def _pairs(d, lo, typ="TK16"):
    return {"%s_%s" % (typ, s): {"diff": d, "ci95": [lo, d + 0.05]} for s in R.SEEDS}


def test_names_both_outcomes():
    hold = R.name_type("TK16", _arms(0.2, 0.1), _pairs(0.1, 0.03), 0.02, [], True)
    assert hold["name"] == "TOKENS-HOLD-CLOSING"
    flat = R.name_type("TK16", _arms(0.2, -0.01), _pairs(0.1, 0.03), 0.02, [], True)           # M1 interval includes 0 at both seeds
    assert flat["name"] == "TOKENS-FLAT"
    flat2 = R.name_type("TK16", _arms(0.2, 0.1), _pairs(0.1, -0.01), 0.02, [], True)           # paired interval includes 0
    assert flat2["name"] == "TOKENS-FLAT"
    below = R.name_type("TK16", _arms(0.2, 0.1), _pairs(0.01, 0.005), 0.02, [], True)           # mean paired difference not above the floor
    assert below["name"] == "TOKENS-FLAT" and not below["mean_above_floor"]
    one_seed_bad = _arms(0.2, 0.1)
    one_seed_bad["TK16_s1"]["r2_ci95"][0] = -0.1
    assert R.name_type("TK16", one_seed_bad, _pairs(0.1, 0.03), 0.02, [], True)["name"] == "TOKENS-FLAT"


def test_p2b_false_blocks_flat_but_not_hold():
    for p2b in (False, None):
        n = R.name_type("TK16", _arms(0.2, -0.01), _pairs(0.1, 0.03), 0.02, [], p2b)
        assert n["name"] == "not named" and R.NO_READ_TEXT in n["reason"]
    assert R.name_type("TK16", _arms(0.2, 0.1), _pairs(0.1, 0.03), 0.02, [], False)["name"] == "TOKENS-HOLD-CLOSING"


def test_a_failed_control_blocks_the_name_for_either_outcome():
    for arms in (_arms(0.2, 0.1), _arms(0.2, -0.01)):
        n = R.name_type("TK16", arms, _pairs(0.1, 0.03), 0.02, ["k4_no_future_input|TK16_s0"], True)
        assert n["name"] == "not named" and "k4_no_future_input|TK16_s0" in n["reason"]


def test_a_missing_seed_or_floor_blocks_the_name():
    arms, pairs = _arms(0.2, 0.1), _pairs(0.1, 0.03)
    arms["TK16_s1"] = None
    n = R.name_type("TK16", arms, pairs, 0.02, [], True)
    assert n["name"] == "not named" and "one seed" in n["reason"]
    pairs2 = _pairs(0.1, 0.03)
    pairs2["TK16_s0"] = None
    assert R.name_type("TK16", _arms(0.2, 0.1), pairs2, 0.02, [], True)["name"] == "not named"
    assert R.name_type("TK16", _arms(0.2, 0.1), _pairs(0.1, 0.03), None, [], True)["name"] == "not named"


def _ctrl_ok():
    c = {"k1_label_path": {"pass": True}, "a2_pack_counts": {"pass": True}, "paired_self_reads_exactly_0": {"pass": True}}
    for a in R.PROBE_ARMS + R.REF_ARMS:
        c["k2_head_forward_eval|%s" % a] = {"pass": True}
        c["k3_closing|%s" % a] = {"pass": True}
        c["k3_slot_n1|%s" % a] = {"pass": True}
        c["k4_no_future_input|%s" % a] = {"pass": True}
        c["k3_card_form_slot|%s" % a] = {"pass": False}                      # printed, gates nothing
    return c


def test_failed_controls_touch_only_their_arm_type_plus_the_reference_and_the_given_inputs():
    given = {"p1": True, "p2a": True, "p2b": True, "p5": True}
    p3 = {"pass": True}
    c = _ctrl_ok()
    assert R.failed_controls_for_type("TK16", c, given, p3) == []
    c["k4_no_future_input|TK64_s1"] = {"pass": False}
    assert R.failed_controls_for_type("TK16", c, given, p3) == []
    assert R.failed_controls_for_type("TK64", c, given, p3) == ["k4_no_future_input|TK64_s1"]
    c["k3_slot_n1|LSD_s0"] = {"pass": False}                                   # a reference arm touches both types
    assert "k3_slot_n1|LSD_s0" in R.failed_controls_for_type("TK16", c, given, p3)
    assert "k3_slot_n1|LSD_s0" in R.failed_controls_for_type("TK64", c, given, p3)


def test_inputs_p1_p2a_p5_and_p3_gate_and_p2b_does_not():
    c = _ctrl_ok()
    ok = {"p1": True, "p2a": True, "p2b": False, "p5": True}
    assert R.failed_controls_for_type("TK16", c, ok, {"pass": True}) == []
    assert any(x.startswith("p3") for x in R.failed_controls_for_type("TK16", c, ok, {"pass": False}))
    for k in ("p1", "p2a", "p5"):
        bad = dict(ok)
        bad[k] = False
        assert any(x.startswith(k) for x in R.failed_controls_for_type("TK16", c, bad, {"pass": True}))
        miss = {x: v for x, v in ok.items() if x != k}
        assert any(x.startswith(k) and "not given" in x for x in R.failed_controls_for_type("TK16", c, miss, {"pass": True}))
    per_arm = {"p1": True, "p2a": True, "p5": {a: True for a in R.PROBE_ARMS}}
    assert any(x.startswith("p5") for x in R.failed_controls_for_type("TK16", c, per_arm, {"pass": True})) is True       # LSD arms not in the dict
    per_arm["p5"].update({a: True for a in R.REF_ARMS})
    assert R.failed_controls_for_type("TK16", c, per_arm, {"pass": True}) == []


def test_alias_loads_the_files_of_the_other_arm_for_lead_outputs_and_states():
    f = R.arm_files("/in", "/st", "TK16_s0", {"TK16_s0": "LSD_s0"})
    assert f["eval_lead"] == "/in/LSD_s0_eval_lead.npz" and f["train_states"] == "/st/LSD_s0_train.npz"
    g = R.arm_files("/in", None, "LSD_s1")
    assert g["eval_states"] == "/in/LSD_s1_eval.npz"
    assert R.parse_alias("A=B,C=D") == {"A": "B", "C": "D"} and R.parse_flags("p1=true,p2b=false") == {"p1": True, "p2b": False}


BANNED = ["vo" + "id", "super" + "seded", "retr" + "acted", "form" + "erly", "corr" + "ection", "with" + "drawn", "no lon" + "ger", "closed" + " loop"]


def test_files_have_lf_line_endings_and_none_of_the_banned_words():
    for name in ("tokprobe_read.py", "test_tokprobe_read.py", "p2b_run.py"):
        raw = (HERE / name).read_bytes()
        assert b"\r" not in raw, name
        low = raw.decode("utf-8").lower()
        for w in BANNED:
            assert w not in low, (name, w)
