"""Tests added for E-GAIN-D (PREREG_WSP_GAIN_D.md section 5), beside the banked tests (tests/test_wsp_edit.py, copied unchanged). Run:
    CUDA_VISIBLE_DEVICES=-1 python -I -m pytest -q -p no:cacheprovider --basetemp=<dir on E:> tests/

  * THE NEW TEST OF THE CARD: with g_data := g_model (and the band-mean target := the model's own) the edit reproduces C0 byte for byte -- synthetic fit set AND the real shards on the real C0 s0 / s1 state dicts
  * with the BASE's targets the changed code returns the banked code's dW / db bit for bit (the banked wsp_edit.py is loaded by path and compared), for several alphas
  * a signed band-mean target (negative, zero, positive) is met exactly on the fit windows
  * a data target moves the toy world's measured median gains to within 10 % of it (the same fit / apply / measure path as the real edit); a flipped delta-g sign does not
"""
import os
import sys
import importlib.util

HERE = os.environ.get("WSP_EDIT_SRC") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np
import pytest
import torch

import wsp_common as W
import wsp_edit as E
from test_wsp_edit import ToyWorld, synth_fit_inputs, basis_like, NB, P


def _banked_module():
    spec = importlib.util.spec_from_file_location("wsp_edit_banked", W.BANK + "/wsp_edit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def real_sd():
    return {s: torch.load(W.CKPT_C0[s], map_location="cpu", weights_only=False) for s in (0, 1)}


def test_banked_module_is_the_pinned_one():
    assert W.sha256_file(W.BANK + "/wsp_edit.py") == "ad4cda1ba53973e4fac898eca4f50b0b2ebbfc1f1b4f8dba875e3d428b783c28"


@pytest.mark.parametrize("alpha", [0.5, 1.0, 0.0])
def test_banked_targets_reproduce_banked_fit(alpha):
    B = _banked_module()
    H, kin3, gv, ga, gb = synth_fit_inputs(seed=3)
    r = E.readout(basis_like(), 2.0)
    old = B.fit_edit(H, kin3, gv, ga, gb, r, alpha)
    new_default = E.fit_edit(H, kin3, gv, ga, gb, r, alpha)
    new_explicit = E.fit_edit(H, kin3, gv, ga, gb, r, alpha, shift=0.35, g_target=(float(np.median(gb[:, 0])), float(np.median(gb[:, 1]))))
    for new in (new_default, new_explicit):
        assert np.array_equal(old["dW"], new["dW"]) and np.array_equal(old["db"], new["db"])
        assert old["kappa"] == new["kappa"] and old["m"] == new["m"] and old["dg_v"] == new["dg_v"] and old["dg_a"] == new["dg_a"]


def test_g_data_equals_g_model_reproduces_C0_synthetic(real_sd):
    """THE CARD'S NEW TEST: g_data := g_model, band-mean target := the model's own -> dW = 0, db = 0 -> every tensor of C0 byte for byte."""
    sd = real_sd[0]["model"]
    H, kin3, gv, ga, gb = synth_fit_inputs(seed=1)
    r = E.readout(sd["basis"].double().numpy(), 2.0)
    f = E.fit_edit(H, kin3, gv, ga, gb, r, 1.0, shift=0.0, g_target=(float(np.median(gv)), float(np.median(ga))))
    assert not np.any(f["dW"]) and not np.any(f["db"])
    new = E.apply_edit(sd, f["dW"], f["db"], NB)
    assert E.state_identical(sd, new)


def _real_shards():
    import fit_gdata as G
    R, _ = G.load_shards()
    return R


@pytest.mark.parametrize("s", [0, 1])
def test_g_data_equals_g_model_reproduces_C0_real_shards(real_sd, s):
    """Same on the REAL 20,000-window shards and the REAL C0 state dict of each seed: the null edit is byte-identical to C0."""
    R = _real_shards()
    sd, cfg = real_sd[s]["model"], real_sd[s]["cfg"]
    r = E.readout(sd["basis"].double().numpy(), cfg["res_scale_m"])
    gv, ga = R[f"s{s}.gv"], R[f"s{s}.ga"]
    f = E.fit_edit(R[f"s{s}.h2"], R["kin3"], gv, ga, R["gb"], r, 1.0, shift=0.0, g_target=(float(np.median(gv)), float(np.median(ga))))
    new = E.apply_edit(sd, f["dW"], f["db"], cfg["nb"])
    assert E.state_identical(sd, new)
    # and the same fit with a data target that differs DOES change the x rows only (the null test is not vacuous)
    f2 = E.fit_edit(R[f"s{s}.h2"], R["kin3"], gv, ga, R["gb"], r, 1.0, shift=0.3, g_target=(float(np.median(gv)) + 0.15, float(np.median(ga)) + 0.15))
    new2 = E.apply_edit(sd, f2["dW"], f2["db"], cfg["nb"])
    assert not E.state_identical(sd, new2) and E.lateral_unchanged(sd, new2, cfg["nb"])


@pytest.mark.parametrize("shift", [-0.4, 0.0, 0.25, 0.8])
def test_signed_band_mean_shift_is_met(shift):
    H, kin3, gv, ga, gb = synth_fit_inputs(seed=2)
    r = E.readout(basis_like(), 2.0)
    for alpha in (0.5, 1.0):
        f = E.fit_edit(H, kin3, gv, ga, gb, r, alpha, shift=shift, g_target=(0.93, 1.36))
        per_window = (H @ f["dW"].T + f["db"]) @ r
        assert abs(float(per_window.mean()) - alpha * shift) < 1e-9, (shift, alpha, float(per_window.mean()))


def _toy_edit(wld, g_target, shift, alpha=1.0, flip=False):
    gv, ga = wld.gains(wld.sd)
    gb = np.stack([np.ones(wld.n), np.full(wld.n, 1.9)], 1)
    kin3 = np.stack([wld.v0, wld.a0, np.zeros(wld.n)], 1)
    H = wld.h2(wld.v0, wld.a0)
    f = E.fit_edit(H, kin3, gv, ga, gb, wld.r, alpha, shift=shift, g_target=g_target)
    dW, db = (f["dW"], f["db"]) if not flip else (-f["dW"], f["db"])
    sd = E.apply_edit(wld.sd, dW, db, NB)
    ge = wld.gains(sd)
    return (float(np.median(gv)), float(np.median(ga))), (float(np.median(ge[0])), float(np.median(ge[1]))), f


def test_data_target_reached_within_10pct_on_toy_world():
    wld = ToyWorld()
    for tgt in ((0.93, 1.36), (1.00, 1.60), (0.95, 1.45)):                # each target at least 0.1 from the toy model's own gains (0.81, 1.27)
        (m_v, m_a), (e_v, e_a), f = _toy_edit(wld, tgt, 0.3)
        assert abs(e_v - tgt[0]) / tgt[0] <= 0.10 and abs(e_a - tgt[1]) / tgt[1] <= 0.10, (tgt, e_v, e_a)
        assert abs(e_v - m_v) > 0.05 and abs(e_a - m_a) > 0.05                # the edit moved the gains
        assert f["g_v_target"] == tgt[0] and f["g_a_target"] == tgt[1]


def test_wrong_sign_of_delta_g_misses_the_target_on_toy_world():
    wld = ToyWorld()
    tgt = (0.93, 1.36)
    (m_v, m_a), (e_v, e_a), f = _toy_edit(wld, tgt, 0.3, flip=True)
    assert abs(e_v - tgt[0]) / tgt[0] > 0.10 or abs(e_a - tgt[1]) / tgt[1] > 0.10


def test_target_below_model_lowers_gains_symmetric():
    """The edit is two-sided: a target below the model's gain lowers it (g_data may sit below g_model)."""
    wld = ToyWorld()
    (m_v, m_a), (e_v, e_a), _ = _toy_edit(wld, (0.70, 1.00), 0.0)
    assert e_v < m_v - 0.05 and e_a < m_a - 0.05 and abs(e_v - 0.70) / 0.70 <= 0.10 and abs(e_a - 1.00) / 1.00 <= 0.10
