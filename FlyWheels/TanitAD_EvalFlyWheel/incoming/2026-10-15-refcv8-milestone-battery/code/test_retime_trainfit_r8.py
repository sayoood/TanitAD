"""Pins for retime_trainfit_r8.py (train-fitted RETIME gains).  Expectations are LITERALS.

  * the fit-source guard REFUSES clips of eval139 and clips outside train4572 -- and a DELIBERATE-REGRESSION arm: with the guard replaced by a no-op the same
    eval-fit input is accepted AND labelled fit_source='train' (the exact mislabel the guard prevents), so the refusal test goes RED against a no-op guard;
  * a frozen file that is not train-fitted is refused at forward time;
  * a planted gain is recovered through fit_surface; the frozen schema is accepted by RT.frozen_weights;
  * the outcome table of PREREG_TRAINFIT section 5 on synthetic forward dicts (PASS / PARTIAL / FAIL x2);
  * the unpaired train-vs-eval bootstrap on constants; the registered texts / constants of the roll script and remap file.
    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' python -m pytest -q test_retime_trainfit_r8.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a0k0_headroom_retime_r8 as RT  # noqa: E402
import retime_trainfit_r8 as TFM  # noqa: E402


def _planted(n=500, T=6, seed=10):
    rng = np.random.default_rng(seed)
    v, w = rng.uniform(1, 25, n), rng.normal(0, 0.12, n)

    def path(vv, ww):
        x, y, th, out = np.zeros(n), np.zeros(n), np.zeros(n), []
        for _ in range(T):
            x, y, th = x + vv * np.cos(th), y + vv * np.sin(th), th + ww
            out.append(np.stack([x, y], -1))
        return np.stack(out, 1)
    O = path(v, w).astype(np.float32)
    B = path(v * rng.uniform(0.6, 1.1, n), w * rng.uniform(0.3, 1.0, n)).astype(np.float32)
    g_true = np.array([1.2, 0.8, 0.5, 1.1, 0.9, 1.3])
    G, _ = RT.retime_positions(RT.polyline_geometry(O), RT.prior_arc(B), g_true)
    return O, B, G, v.astype(np.float32), g_true


def _per_surf(O, B, G, v):
    return {"S6": {"Os": [O, O], "HP": B, "G": G, "v0": v, "mask": np.ones(len(v), bool), "instants_s": [1, 2, 3, 4, 5, 6], "hz": [10, 20, 30, 40, 50, 60], "n_episodes": 10}}


# ------------------------------------------------------------------ the guard and its mutation
def test_guard_refuses_eval_clips_and_clips_outside_train4572_and_accepts_clean():
    ev, tr = {"e1", "e2", "e3"}, {"t1", "t2", "t3", "t4"}
    with pytest.raises(SystemExit, match="eval139"):
        TFM.assert_train_only({"t1", "e2"}, ev, tr)
    with pytest.raises(SystemExit, match="not in train4572"):
        TFM.assert_train_only({"t1", "zz"}, ev, tr)
    with pytest.raises(SystemExit, match="no fit clips"):
        TFM.assert_train_only(set(), ev, tr)
    g = TFM.assert_train_only({"t1", "t2"}, ev, tr)
    assert g["n_fit_clips"] == 2 and g["n_overlap_with_eval139"] == 0 and g["is_registered_A6_sample"] is False


def test_fitting_on_eval_windows_is_refused_and_a_noop_guard_would_mislabel_it(monkeypatch):
    O, B, G, v, _ = _planted(80)
    per = _per_surf(O, B, G, v)
    ev, tr = {"e1", "e2"}, {"t1", "t2"}

    def refuses() -> bool:
        try:
            TFM.freeze_arrays(per, {"e1", "e2"}, ev, tr)       # the fit clips ARE the eval clips
        except SystemExit:
            return True
        return False
    assert refuses() is True                                    # the real guard refuses an eval fit
    monkeypatch.setattr(TFM, "assert_train_only", lambda *a, **k: {"n_fit_clips": 2, "n_overlap_with_eval139": 0})   # DELIBERATE REGRESSION
    assert refuses() is False                                   # with a no-op guard the same input goes through ...
    out = TFM.freeze_arrays(per, {"e1", "e2"}, ev, tr)
    assert out["fit_source"] == "train"                         # ... and is LABELLED train: this is what the guard prevents


def test_forward_refuses_a_frozen_file_that_is_not_train_fitted():
    ev_file = json.load(open(HERE.parent / "raw" / "a0k0_retime" / "FROZEN_GAINS_RETIME_ema30k.json", encoding="utf-8")) \
        if (HERE.parent / "raw" / "a0k0_retime" / "FROZEN_GAINS_RETIME_ema30k.json").exists() else {"surfaces": {}}
    with pytest.raises(SystemExit, match="not a train-fitted file"):
        TFM.assert_frozen_is_train(ev_file)                    # the eval-fitted file carries no fit_source
    with pytest.raises(SystemExit):
        TFM.assert_frozen_is_train({"fit_source": "train", "fit_guard": {"n_overlap_with_eval139": 1}})
    TFM.assert_frozen_is_train({"fit_source": "train", "fit_guard": {"n_overlap_with_eval139": 0}})


# ------------------------------------------------------------------ the fit
def test_fit_surface_recovers_a_planted_gain_and_schema_feeds_frozen_weights():
    O, B, G, v, g_true = _planted()
    rec = TFM.fit_surface([O, O], B, G, v, np.ones(len(v), bool))
    assert np.allclose(rec["all"]["RETIME"], [1.2, 0.8, 0.5, 1.1, 0.9, 1.3], atol=1e-4)
    assert rec["all"]["n_fit_rows_pooled"] == 1000
    fz = {"surfaces": {"S6": rec}}
    fw = RT.frozen_weights(fz, "S6", v)
    assert fw["RETIME_FROZEN30k"].shape == (6,) and fw["GATED_RETIME_FROZEN30k"].shape == (500, 6)
    assert np.allclose(fw["GATED_RETIME_FROZEN30k"][np.argmax(v)], [1.2, 0.8, 0.5, 1.1, 0.9, 1.3], atol=1e-4)
    assert all(0.0 <= x <= 2.0 for x in rec["all"]["RETIME"])


def test_freeze_arrays_records_the_guard_and_pools_seeds():
    O, B, G, v, _ = _planted(120)
    out = TFM.freeze_arrays(_per_surf(O, B, G, v), {"t1"}, {"e1"}, {"t1", "t2"})
    assert out["fit_source"] == "train" and out["fit_guard"]["n_overlap_with_eval139"] == 0 and out["fallback"] is None
    assert out["surfaces"]["S6"]["all"]["n_fit_rows_pooled"] == 240
    assert TFM.freeze_arrays(_per_surf(O, B, G, v), {"t1"}, {"e1"}, {"t1"}, fallback=True)["fallback"] == "FALLBACK_SINGLE_SEED"


# ------------------------------------------------------------------ the outcome table (PREREG section 5)
def _cell(d, lo, hi):
    return {"delta": d, "lo": lo, "hi": hi, "separated": bool(hi < 0 or lo > 0)}


def _fwd(e1="PASS (single training seed; training-seed floor unmeasured)", e1p_sep=True, lat_bad=(), nr="PASS", s2_os=(-0.02, -0.03, -0.01), s6_os=(-0.2, -0.3, -0.1)):
    arm = "RETIME_TRAINFIT"
    per = {str(s): {"delta": -0.03, "lo": -0.04, "hi": -0.02, "pass_separated": e1p_sep} for s in (0, 1)}
    lat_rows = {k: {"per_seed": {}, "worse_at_seeds": ([0] if k in lat_bad else [])} for k in RT.LATERAL_KEYS}
    lat = {"HOLDS": not lat_bad, "separated_worse_at_any_seed": list(lat_bad), "separated_worse_at_both_seeds": [], "undetermined": [], "rows": lat_rows}
    sd = {str(s): {"surfaces": {"S2": {"sets": {"compliant": {"arms": {arm: {"minus_os": {"ade_m": _cell(*s2_os)}}}}}},
                                "S6": {"sets": {"compliant": {"arms": {arm: {"minus_os": {"ade_m": _cell(*s6_os)}}}}}}}} for s in (0, 1)}
    return {"seeds": sd, "bars": {f"compliant|{arm}": {"FZR-E1": {"verdict": e1}, "FZR-E1P": {"per_inference_seed": per}, "FZR-E3": {}, "FZR-LAT": lat,
                                                      "FZR-NR": {"per_family": {"ADE": nr, "LATERAL": "PASS"}}}}}


def test_outcome_labels_literal():
    assert TFM.outcome(_fwd())["label"] == "DEPLOYABLE-PASS"
    assert TFM.outcome(_fwd(e1="NOT PROVEN (|delta| within 2x the inference-seed floor)"))["label"] == "PARTIAL"
    assert TFM.outcome(_fwd(e1="FAILED"))["label"] == "PARTIAL"                         # lever moves vs os, does not clear the echo
    assert TFM.outcome(_fwd(e1p_sep=False))["label"] == "PARTIAL"
    o = TFM.outcome(_fwd(lat_bad=("LAT_heading_mae_deg",)))
    assert o["label"] == "FAIL" and o["ACC_LAT_label"] == "FAILS"
    assert TFM.outcome(_fwd(nr="FAIL: LAT_x"))["label"] == "FAIL"
    o = TFM.outcome(_fwd(s2_os=(0.001, -0.01, 0.01)))                                   # arm - os not separated better at S2: the lever does not move
    assert o["label"] == "FAIL" and "NOT MOVES" in o["fail_reasons"][0]
    o = TFM.outcome(_fwd(s6_os=(-0.001, -0.05, 0.02)))                                  # moves at S2 but not at S6 -> ACC-LAT only holds vacuously -> not a PASS
    assert o["ACC_LAT_label"].startswith("HOLDS VACUOUSLY") and o["label"] == "PARTIAL"


# ------------------------------------------------------------------ the train-vs-eval gap estimator
def test_unpaired_gap_on_constants_is_exact():
    rng = np.random.default_rng(0)
    ea, eb = np.repeat(np.arange(6), 5), np.repeat(np.arange(8), 4)
    a = {"os": (np.full(30, 2.0), ea), "ha0_ext": (np.full(30, 4.0), ea)}
    b = {"os": (np.full(32, 1.0), eb), "ha0_ext": (np.full(32, 2.0), eb)}
    r = TFM.unpaired_gap(a, b, n_boot=200)
    assert r["os_diff_a_minus_b"] == {"delta": 1.0, "lo": 1.0, "hi": 1.0}
    assert r["ratio_num_over_den"]["a"] == 0.5 and r["ratio_num_over_den"]["b"] == 0.5
    assert r["ratio_num_over_den"]["diff_a_minus_b"] == {"delta": 0.0, "lo": 0.0, "hi": 0.0}
    del rng


# ------------------------------------------------------------------ registered texts and constants
def test_prereg_text_is_the_registered_one_plus_an_append_only_amendment():
    assert TFM.sha256_file(TFM.PREREG) == TFM.PREREG_SHA256 == "648cdee6e68f70c2a91f518dbcce2999e846d5de6759a7deaf8a7d07311096f6"
    assert (TFM.TF / "PREREG_TRAINFIT.sha256").read_text().strip() == TFM.PREREG_SHA256
    assert TFM.sha256_file(TFM.PREREG_V0) == TFM.PREREG_V0_SHA256 == "be16e4dd9ab2c5b45cb571b083f6f7e97217b8134ebbd83c3732b629f0e4cf7e"
    v0, full = TFM.PREREG_V0.read_bytes(), TFM.PREREG.read_bytes()
    assert full.startswith(v0) and len(full) > len(v0)                    # the amendment only APPENDS
    tail = full[len(v0):].decode("utf-8")
    assert "# AMENDMENT 1" in tail and "2026-10-19T03:26:11" in tail and "**No train-clip roll output exists.**" in tail and "total 0" in tail
    for tok in ("CONST_G1", "CONST_G0", "CONST_G05", "INDISTINGUISHABLE", "FITTED-SHAPE-MATTERS", "WITHHELD", "ZZS40STARTZZ"):
        assert tok in tail
    assert TFM.A6_SAMPLE_FINGERPRINT == "6086d9cc0c90281619eb0bb63736a73e012322b48dddf5b4ed0363008a0ee3ed" and TFM.A6_SAMPLE_N == 139


# ------------------------------------------------------------------ AMENDMENT 1: known-value controls and the constant comparator
def test_walker_literal_on_a_straight_path_incl_the_extension():
    O = np.array([[[1, 0], [2, 0], [3, 0], [4, 0]]], np.float32)
    HP = np.array([[[0.5, 0], [1.5, 0], [2.5, 0], [6.0, 0]]], np.float32)       # chord arcs M = 0.5, 1.5, 2.5, 6.0 (the last is past L_S = 4)
    assert np.allclose(TFM.walker_positions(O, HP)[0, :, 0], [0.5, 1.5, 2.5, 6.0], atol=1e-12)
    assert np.allclose(TFM.walker_positions(O, HP)[0, :, 1], 0.0)


def test_const_weights_literals():
    w = TFM.const_weights(4)
    assert w["CONST_G1"].tolist() == [1.0] * 4 and w["CONST_G0"].tolist() == [0.0] * 4 and w["CONST_G05"].tolist() == [0.5] * 4
    assert TFM.CONST_ARM_NAMES == ("CONST_G1", "CONST_G0", "CONST_G05")


def test_known_value_controls_read_exactly_and_withhold_when_mutated():
    O, B, _G, v, _ = _planted(200)
    ok = TFM.known_value_arrays(O, B)
    assert ok["g1_bit_equal_os"] is True and ok["g1_max_abs_m"] == 0.0 and ok["g0_vs_walker_max_abs_m"] <= 1e-9
    TFM.assert_known_values({"S6|s0": ok})                                         # passes
    with pytest.raises(SystemExit, match="WITHHELD"):
        TFM.assert_known_values({"S6|s0": TFM.known_value_arrays(O, B, g1=1.001)})   # MUTATION: CONST_G1 off by 1e-3 must withhold
    with pytest.raises(SystemExit, match="WITHHELD"):
        TFM.assert_known_values({"S6|s0": TFM.known_value_arrays(O, B, g0=0.001)})   # MUTATION: CONST_G0 off by 1e-3 must withhold
    with pytest.raises(SystemExit, match="WITHHELD"):
        TFM.assert_known_values({})                                                  # nothing checked is not a pass


def _g1_res(delta=0.0, lo=0.0, hi=0.0, extra=None):
    cells = {"ade_m": {"delta": delta, "lo": lo, "hi": hi}, "LON_speed_mae_mps": {"delta": 0.0, "lo": 0.0, "hi": 0.0}}
    cells.update(extra or {})
    return {"seeds": {"0": {"surfaces": {"S2": {"sets": {"compliant": {"arms": {"CONST_G1": {"minus_os": cells}}}}}}}}}


def test_const_g1_cells_must_be_exactly_zero():
    assert TFM.assert_const_g1_cells(_g1_res(extra={"TAC_x": {"status": "REFUSED"}})) == 2
    with pytest.raises(SystemExit, match="WITHHELD"):
        TFM.assert_const_g1_cells(_g1_res(delta=1e-3))                               # MUTATION: a 1e-3 ADE difference withholds
    with pytest.raises(SystemExit, match="WITHHELD"):
        TFM.assert_const_g1_cells(_g1_res(lo=-1e-9, hi=1e-9))                        # a non-zero-width CI withholds
    with pytest.raises(SystemExit, match="WITHHELD"):
        TFM.assert_const_g1_cells(_g1_res(extra={"LAT_y": {"foo": 1}}))              # a row with neither an interval nor REFUSED withholds


def _vs(cells):                                                                      # cells = {(surf, seed): (delta, lo, hi)}
    return {f"{sf}|compliant|s{s}": {"RETIME_TRAINFIT": {"ade_m": {"delta": d, "lo": lo, "hi": hi}}} for (sf, s), (d, lo, hi) in cells.items()}


def test_const_g05_reading_labels_literal():
    keys = [(sf, s) for sf in ("S2", "S6") for s in (0, 1)]
    r = TFM.const_reading(_vs({k: (0.001, -0.01, 0.012) for k in keys}))
    assert r["reading"] == "INDISTINGUISHABLE" and "not the evidence" in r["sentence"]
    assert TFM.const_reading(_vs({k: (-0.05, -0.07, -0.02) for k in keys}))["reading"] == "FITTED-SHAPE-MATTERS"
    c = {k: (-0.05, -0.07, -0.02) for k in keys}
    c[("S6", 1)] = (0.05, 0.02, 0.08)
    assert TFM.const_reading(_vs(c))["reading"] == "CONST-BEATS-FIT"
    c = {k: (-0.05, -0.07, -0.02) for k in keys}
    c[("S2", 0)] = (-0.001, -0.01, 0.01)
    assert TFM.const_reading(_vs(c))["reading"] == "MIXED"


def test_roll_script_and_remap_pin_the_train_inputs():
    src = (HERE / "roll_seed_r8_train.py").read_text(encoding="utf-8")
    assert "STRIDE = 5" in src and "% STRIDE == 0" in src
    assert 'os.environ["R8_NO_POSE_SYNC"] = "1"' in src
    assert 'TRAIN_V9_MD5 = "83e30bd3791f0767707eade5f4671759"' in src
    assert "--no-navshuf" in src and "with_navflip=False" in src          # the os-only branch of roll_one (SPEC A3)
    rm = json.load(open(HERE / "refcv8_train_remap_devbox.json", encoding="utf-8"))
    assert rm["--eval-cache"].endswith("refcv6-b1-416x1024-train-a6")
    assert rm["--eval-labels"].endswith("s2_labels_v8_train.jsonl.gz") and rm["--speed-max-sidecar-v6-eval"].endswith("refcv6_speed_max_v8_train.jsonl")
    assert rm["--r8-v9-labels-eval"].endswith("release_train4572/v9_labels_train.npz")
    assert not any("eval139" in rm[k] and k in ("--eval-cache", "--eval-labels", "--speed-max-sidecar-v6-eval", "--r8-v9-labels-eval") for k in rm)
