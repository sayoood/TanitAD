"""Tests of s1_read.py on SYNTHETIC data only (no probe-set file, no trained state, no label of the programme is read).
Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q --basetemp F:/TanitAD-scratch/v8proof/spatial/s1/tmp_test test_s1_read.py
"""
import os as _o
_o.environ["OMP_NUM_THREADS"] = "2"
import hashlib
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import s1_read as R  # noqa: E402

E, S0, RP = R.E, R.S0, R.RP


def make_probe(n_clips=16, frames=50, seed=0):
    rng = np.random.default_rng(seed)
    beid, bg, epi, gap, pres, kin = [], [], [], [], [], []
    for c in range(n_clips):
        g = 30.0 + np.cumsum(rng.uniform(-0.3, 0.3, frames))
        g = np.clip(g, 5, 55)
        p = np.ones(frames)
        if c < 8:
            g[30:] = np.nan; p[30:] = 0.0
        beid += [f"clip{c:03d}"] * frames; bg += list(range(frames)); epi += [c] * frames
        gap.append(g); pres.append(p); kin.append(rng.standard_normal((frames, 3)))
    D = {"beid": np.array(beid), "bg": np.array(bg, np.int64), "epi": np.array(epi, np.int64), "gap": np.concatenate(gap), "pres": np.concatenate(pres),
         "kin": np.concatenate(kin), "uep": [f"clip{c:03d}" for c in range(n_clips)]}
    D["gap_loaded"], D["pres_loaded"] = D["gap"].copy(), D["pres"].copy()
    return D


def run_core(D, Z, n_folds=2, labsets=None):
    D, Pr, _ = R.prepare(D)
    labsets = labsets or R.make_labsets(D, Pr)
    oof, times, infos = R.compute_oof(Z, D, Pr, labsets, n_folds, log_=lambda *a: None)
    return D, Pr, oof


def planted_state(D, rng):
    g = np.where(np.isfinite(D["gap"]), D["gap"], 0.0)
    return np.column_stack([g + 0.01 * rng.standard_normal(len(g)), rng.standard_normal((len(g), 15))]).astype(np.float32)


def test_planted_state_reads_r2_near_one():
    D = make_probe()
    Z = planted_state(D, np.random.default_rng(1))
    D, Pr, oof = run_core(D, Z)
    r2 = RP.r2_pooled(D["gap"], oof["ra|main|s|gap"])
    assert r2 > 0.9, r2
    assert RP.r2_pooled(D["gap"], oof["ra|main|ego|gap"]) < 0.5        # the ego vector is noise here


def test_random_state_reads_at_or_below_bar():
    D = make_probe()
    Z = np.random.default_rng(2).standard_normal((len(D["gap"]), 16)).astype(np.float32)
    D, Pr, oof = run_core(D, Z)
    assert RP.r2_pooled(D["gap"], oof["ra|main|s|gap"]) <= 0.02
    for a in R.ARMS3:
        assert R.S0C.skill(Pr["d"], oof[f"arm|main|{a}"]) <= 0.10, a   # tiny synthetic set: a finite-sample mean of d is not zero


def test_t3_passes_on_permuted_labels_and_goes_red_on_unpermuted(monkeypatch):
    monkeypatch.setattr(R, "E1_BAR", 0.10)      # tiny synthetic set (the real bar is 0.02)
    D = make_probe()
    Z = planted_state(D, np.random.default_rng(3))
    D, Pr, _ = R.prepare(D)
    ls = R.make_labsets(D, Pr)
    oof, _, _ = R.compute_oof(Z, D, Pr, ls, 2, log_=lambda *a: None)
    assert R.t3_values(D, Pr, oof)["pass"] is True
    # deliberate regression: the 'permuted' slot carries the TRUE labels -> the control must fail
    bad = {"main": ls["main"], "e1": ls["main"]}
    oof2, _, _ = R.compute_oof(Z, D, Pr, bad, 2, log_=lambda *a: None)
    D2 = dict(D); D2["pres_e1"] = D["pres"]; Pr2 = dict(Pr); Pr2["d_e1"] = Pr["d"]
    assert R.t3_values(D2, Pr2, oof2)["pass"] is False


def test_equals_stage_0e_code_path():
    """The state-file folds give bit-identical predictions to s0e_read.ra_fold / change_fold when the stage-0e state slots hold the same matrix."""
    D = make_probe(n_clips=16)
    Z = planted_state(D, np.random.default_rng(4))
    D, Pr, _ = R.prepare(D)
    lab = {"main": {"gap": D["gap"], "pres": D["pres"], "d": Pr["d"]}}
    f = 1
    mine = {}
    R.ra_state_fold(f, Z, D["kin"], D["gap"], D["pres"], D["epi"], Pr["folds_b"], mine, "main")
    R.change_state_fold(f, Z, Pr, {"main": Pr["d"]}, mine)
    ref = {}
    E.ra_fold(f, {"g": Z, "m": Z}, D["kin"], D["gap"], D["pres"], D["epi"], Pr["folds_b"], ref, "main")
    E.change_fold(f, {"g": Z, "m": Z}, Pr, {"main": Pr["d"]}, ref, Pr["folds_b"])
    for tg in ("gap", "pres"):
        for s0, s1 in (("s", "g"), ("ego", "ego")):
            assert np.array_equal(mine[f"ra|main|{s0}|{tg}"], ref[f"ra|main|{s1}|{tg}"], equal_nan=True), (s0, tg)
    for a0, a1 in (("H", "H-g"), ("N", "N-g"), ("E", "E")):
        assert np.array_equal(mine[f"arm|main|{a0}"], ref[f"arm|main|{a1}"], equal_nan=True), a0


def write_state(tmp, D, Z, dim_record=None, mode="mean", extra=None):
    p = f"{tmp}/s.npz"
    np.savez(p, Z=Z.astype(np.float16), clip_idx=D["epi"].astype(np.int32), t=D["bg"].astype(np.int16), clips=np.array(D["uep"]))
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    rec = {"ckpt_sha256": "ab" * 32, "wm_state": mode, "D": int(Z.shape[1] if dim_record is None else dim_record), "n_frames": int(len(Z)), "n_clips": len(D["uep"]), "out_sha256": h,
           "split": "train", "lo": 600, "hi": 1800}
    rec.update(extra or {})
    with open(p + ".json", "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
    return p


def test_t1_t4_green_then_red(tmp_path):
    D = make_probe(n_clips=10, frames=30)
    Z = np.random.default_rng(5).standard_normal((len(D["gap"]), 16)).astype(np.float32)
    p = write_state(str(tmp_path), D, Z)
    S = R.load_states(p)
    assert R.check_t1(S, D)["pass"] is True
    assert R.check_t4(S, p, 16, "mean", "ab" * 32, expect_clips=10)["pass"] is True
    # t1 red: clip_idx shifted by one row
    S2 = dict(S); S2["clip_idx"] = np.roll(S["clip_idx"], 1)
    assert R.check_t1(S2, D)["pass"] is False
    S3 = dict(S); S3["t"] = S["t"][::-1].copy()
    assert R.check_t1(S3, D)["pass"] is False
    S4 = dict(S); S4["clips"] = list(reversed(S["clips"]))
    assert R.check_t1(S4, D)["pass"] is False
    # t4 red: wrong declared width, wrong mode, wrong checkpoint, record width mismatch
    assert R.check_t4(S, p, 128, "mean", None, expect_clips=10)["pass"] is False
    assert R.check_t4(S, p, 16, "grid", None, expect_clips=10)["pass"] is False
    assert R.check_t4(S, p, 16, "mean", "cd" * 32, expect_clips=10)["pass"] is False
    p2 = write_state(str(tmp_path), D, Z, dim_record=640)
    assert R.check_t4(R.load_states(p2), p2, 16, "mean", None, expect_clips=10)["pass"] is False


def test_pair_of_a_run_with_itself_is_exactly_zero():
    D = make_probe(n_clips=16)
    Z = planted_state(D, np.random.default_rng(6))
    D, Pr, oof = run_core(D, Z, n_folds=2)
    dummy = {"controls_pass": True}
    td = {}
    import tempfile
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        p = f"{tmp}/o.npz"
        R.save_oof(p, D, Pr, oof)
        z = np.load(p); z = {k: z[k] for k in z.files}
    diffs, n = R.paired_from(z, z, B=60)
    assert len(diffs) == 5
    for k, v in diffs.items():
        assert v["diff"] == 0.0 and v["ci95"] == [0.0, 0.0], (k, v)
    z2 = dict(z); z2["gap"] = z["gap"] + 1.0
    with pytest.raises(SystemExit):
        R.paired_from(z, z2, B=20)
    # a real difference is not zero: A = planted, B = random
    Zr = np.random.default_rng(7).standard_normal(Z.shape).astype(np.float32)
    _, _, oofr = run_core(make_probe(n_clips=16), Zr, n_folds=2)
    zr = dict(z)
    for k in ("ra__main__s__gap", "ra__main__s__pres"):
        zr[k] = oofr[k.replace("__", "|")]
    d2, _ = R.paired_from(z, zr, B=60)
    assert d2["gap R2 (state)"]["diff"] > 0.5 and d2["gap R2 (state)"]["separated_pos"]


def test_t2_logic_and_spread():
    v = {"gap_r2": 0.30, "pres_auc": 0.80}
    assert R.check_t2(v, None)["pass"] is False
    assert R.check_t2(v, {"gap_r2": 0.303, "pres_auc": 0.798, "ref_controls_pass": True})["pass"] is True
    assert R.check_t2(v, {"gap_r2": 0.31, "pres_auc": 0.80, "ref_controls_pass": True})["pass"] is False
    assert R.check_t2(v, {"gap_r2": 0.30, "pres_auc": 0.80, "ref_controls_pass": False})["pass"] is True      # note N9: the reference is an identity value; it does not need the stage 0e controls
    assert R.check_t2(v, {"missing": True, "ref_controls_pass": True})["pass"] is False

    def fake(tag, x):
        ci = [0.0, 1.0]
        return {"meta": {"tag": tag}, "main": {"r_a": {"estimates": {"gap|s": {"r2": x}, "pres|s": {"auc": 0.5 + x}},
                                                       "diffs_gap_r2": {"state minus ego": {"diff": x}}, "diffs_presence_auc": {"state minus ego": {"diff": x}}},
                                               "change": {"arms": {"H": {"skill": x}, "N": {"skill": x}, "E": {"skill": 0.0}},
                                                          "diffs": {"H minus E": {"diff": x, "ci95": ci}, "H minus N": {"diff": 0.0, "ci95": ci}}}}}
    sp = R.spread_of([fake("a", 0.1), fake("b", 0.3), fake("c", 0.2)])
    m = sp["gap R2 (state)"]
    assert m["min"] == 0.1 and m["max"] == 0.3 and abs(m["std_ddof1"] - 0.1) < 1e-12


def test_label_text():
    assert R.label_of("mean", 128, "H1") == "baseline (trained mean state), H1"
    assert R.label_of("grid", 640, "G1") == "trained state, G1"
    assert R.label_of("mean", 640, "W1") == "trained state, W1"
