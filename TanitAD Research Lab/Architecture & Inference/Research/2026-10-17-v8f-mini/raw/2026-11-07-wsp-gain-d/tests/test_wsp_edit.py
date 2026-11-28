"""Tests of the E-GAIN edit (PREREG_WSP_GAIN.md section 5). Run: CUDA_VISIBLE_DEVICES=-1 python -I -m pytest -q tests/test_wsp_edit.py

  * alpha = 0 reproduces C0 bit for bit (real C0 s0 checkpoint state dict; every tensor byte-identical; the forward plan is identical too, on a synthetic head)
  * the lateral coefficients (head_o.2 rows nb:2nb) and every other tensor are byte-identical at alpha = 1; the x rows changed
  * a MUTANT that edits one lateral coefficient turns the lateral test RED (the check function detects it); a mutant that edits another tensor does too
  * a MUTANT with the wrong u sign makes read 1 FAIL, on a synthetic world whose correct edit passes read 1 (the same fit / apply / measure code path)
  * readout / unit-direction identities (r_res . u == 1), the kappa rule (band-mean shift == 0.35 * alpha on the fit windows), ridge limit
"""
import os
import sys

HERE = os.environ.get("WSP_EDIT_SRC") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")     # WSP_EDIT_SRC: a mutated copy (run_mutations.py)
sys.path.insert(0, HERE)
import numpy as np
import pytest
import torch

import wsp_common as W
import wsp_edit as E

NB = 12
P = 512


@pytest.fixture(scope="module")
def real_sd():
    ck = torch.load(W.CKPT_C0[0], map_location="cpu", weights_only=False)
    return ck["model"], ck["cfg"]


def synth_fit_inputs(seed=0, n=3000, noise=0.0):
    """A synthetic fit set: h2 linear in (v0, a0) plus latent features; kin3, gains and base gains in the shape the real data have."""
    rng = np.random.default_rng(seed)
    v0 = rng.uniform(3, 8, n)
    a0 = rng.normal(0.3, 1.0, n)
    lat = rng.normal(size=(n, P - 2))
    Pm = rng.normal(size=(2, P)) * 0.5
    H = np.stack([v0, a0], 1) @ Pm[:2] + np.concatenate([np.zeros((n, 2)), lat], 1) @ np.eye(P) * 0.3 + noise * rng.normal(size=(n, P))
    kin3 = np.stack([v0, a0, rng.normal(0, 0.01, n)], 1)
    gv = 0.81 + 0.01 * rng.normal(size=n)
    ga = 1.27 + 0.02 * rng.normal(size=n)
    gb = np.stack([np.ones(n), np.full(n, 1.9)], 1)
    return H, kin3, gv, ga, gb


def basis_like():
    i = np.arange(60, dtype=np.float64)
    b = np.stack([np.cos(k * np.pi * (i + 0.5) / 60) for k in range(NB)], 1) * ((i + 1) / 60)[:, None]
    return b


def test_readout_unit_direction():
    r = E.readout(basis_like(), 2.0)
    u = E.unit_direction(r)
    assert abs(float(r @ u) - 1.0) < 1e-12 and r.shape == (NB,)


def test_real_basis_readout_matches_buffer(real_sd):
    sd, cfg = real_sd
    r = E.readout(sd["basis"].double().numpy(), cfg["res_scale_m"])
    assert np.isfinite(r).all() and np.linalg.norm(r) > 0


def test_alpha0_bit_for_bit(real_sd):
    sd, _ = real_sd
    H, kin3, gv, ga, gb = synth_fit_inputs()
    f = E.fit_edit(H, kin3, gv, ga, gb, E.readout(sd["basis"].double().numpy(), 2.0), alpha=0.0)
    new = E.apply_edit(sd, f["dW"], f["db"], NB)
    assert E.state_identical(sd, new)


def test_alpha1_changes_only_x_rows(real_sd):
    sd, _ = real_sd
    H, kin3, gv, ga, gb = synth_fit_inputs()
    f = E.fit_edit(H, kin3, gv, ga, gb, E.readout(sd["basis"].double().numpy(), 2.0), alpha=1.0)
    snap = {k: v.clone() for k, v in sd.items()}
    new = E.apply_edit(sd, f["dW"], f["db"], NB)
    assert E.state_identical(snap, sd)                                       # apply_edit did not mutate its input in place
    assert E.lateral_unchanged(sd, new, NB)
    assert not E.state_identical(sd, new)
    assert not E.tensors_equal_bytes(sd[E.KEY_W][:NB], new[E.KEY_W][:NB])


def test_mutant_lateral_edit_turns_lateral_test_red(real_sd):
    sd, _ = real_sd
    H, kin3, gv, ga, gb = synth_fit_inputs()
    f = E.fit_edit(H, kin3, gv, ga, gb, E.readout(sd["basis"].double().numpy(), 2.0), alpha=1.0)
    good = E.apply_edit(sd, f["dW"], f["db"], NB)
    assert E.lateral_unchanged(sd, good, NB)                                 # the clean edit is GREEN
    # mutant 1: one lateral weight coefficient (row NB + 3, column 17) nudged by one ulp-scale amount
    m1 = {k: v.clone() for k, v in good.items()}
    m1[E.KEY_W][NB + 3, 17] += 1e-6
    assert not E.lateral_unchanged(sd, m1, NB)                               # RED
    # mutant 2: one lateral bias coefficient
    m2 = {k: v.clone() for k, v in good.items()}
    m2[E.KEY_B][NB + 5] += 1e-6
    assert not E.lateral_unchanged(sd, m2, NB)
    # mutant 3: the edit written to ALL 2*nb rows (a whole-layer edit instead of the x rows)
    m3 = {k: v.clone() for k, v in sd.items()}
    m3[E.KEY_W][:] = (m3[E.KEY_W].double() + torch.as_tensor(np.vstack([f["dW"], f["dW"]]))).to(torch.float32)
    assert not E.lateral_unchanged(sd, m3, NB)
    # mutant 4: an unrelated tensor touched
    m4 = {k: v.clone() for k, v in good.items()}
    k0 = [k for k in m4 if k.endswith("e_ego.weight")][0]
    m4[k0].view(-1)[0] += 1e-6
    assert not E.lateral_unchanged(sd, m4, NB)


def test_kappa_rule_band_mean_shift():
    H, kin3, gv, ga, gb = synth_fit_inputs()
    r = E.readout(basis_like(), 2.0)
    for alpha in (0.5, 1.0):
        f = E.fit_edit(H, kin3, gv, ga, gb, r, alpha=alpha)
        shift_per_window = (H @ f["dW"].T + f["db"]) @ r                      # plan-speed shift of every fit window
        assert abs(float(shift_per_window.mean()) - alpha * E.SHIFT) < 1e-9


def test_ridge_recovers_known_direction():
    rng = np.random.default_rng(1)
    H = rng.normal(size=(4000, 40))
    w_true = rng.normal(size=40)
    y = H @ w_true + 2.0
    w, lam = E.ridge_direction(H, y, lam_frac=1e-9)
    assert np.allclose(w, w_true, atol=1e-4)


# ---------- synthetic world for the read-1 mutant (same fit / apply / measure path as the real edit; the model is linear in (v0, a0) through h2)
class ToyWorld:
    """s(v0, a0) = base(v0, a0) + learned(v0, a0) where learned = c_v v0 + c_a a0 + b0 + r_res . (W_x h2 + b_x), h2 = A [v0, a0, latent] (frozen features, latent fixed per window).
    Gains are measured by central finite differences through the SAME apply path (state dict -> head -> plan speed)."""

    def __init__(self, seed=0, n=2000):
        rng = np.random.default_rng(seed)
        self.n = n
        self.v0 = rng.uniform(3, 8, n)
        self.a0 = rng.normal(0.3, 1.0, n)
        self.lat = rng.normal(size=(n, P - 2)) * 0.3
        self.A = rng.normal(size=(2, P)) * 0.5
        self.basis = basis_like()
        self.r = E.readout(self.basis, 2.0)
        self.sd = {E.KEY_W: torch.zeros(2 * NB, P), E.KEY_B: torch.zeros(2 * NB)}
        self.shrink_v, self.shrink_a = 0.19, 0.63          # the learned parts shrink the gains (v: 1 - 0.19, a: 1.9 - 0.63); the shrink lives outside head_o in this toy

    def h2(self, v0, a0):
        return np.stack([v0, a0], 1) @ self.A + np.concatenate([np.zeros((len(v0), 2)), self.lat[: len(v0)]], 1)

    def speed(self, sd, v0, a0):
        base = 1.0 * v0 + 1.9 * a0
        learned = -self.shrink_v * v0 - self.shrink_a * a0 - 0.35                              # the 'prior + learned' shrink outside head_o
        h = torch.as_tensor(self.h2(v0, a0)).float()
        out = h @ sd[E.KEY_W].T + sd[E.KEY_B]
        return base + learned + (out[:, :NB].double().numpy() @ self.r)

    def gains(self, sd, eps=0.05):
        gv = (self.speed(sd, self.v0 + eps, self.a0) - self.speed(sd, self.v0 - eps, self.a0)) / (2 * eps)
        ga = (self.speed(sd, self.v0, self.a0 + eps) - self.speed(sd, self.v0, self.a0 - eps)) / (2 * eps)
        return gv, ga


def test_wrong_u_sign_makes_read1_fail():
    wld = ToyWorld()
    gv, ga = wld.gains(wld.sd)
    gb = np.stack([np.ones(wld.n), np.full(wld.n, 1.9)], 1)
    kin3 = np.stack([wld.v0, wld.a0, np.zeros(wld.n)], 1)
    H = wld.h2(wld.v0, wld.a0)
    for alpha in (0.5, 1.0):
        good = E.fit_edit(H, kin3, gv, ga, gb, wld.r, alpha=alpha)
        sd_g = E.apply_edit(wld.sd, good["dW"], good["db"], NB)
        ge = wld.gains(sd_g)
        ok, rows = E.read1_verdict((np.median(ge[0]), np.median(ge[1])), (np.median(gv), np.median(ga)), (1.0, 1.9), alpha)
        assert ok, rows                                                       # the correct edit passes read 1 on the toy world (decoder slope ~ 1)
        bad = E.fit_edit(H, kin3, gv, ga, gb, wld.r, alpha=alpha, u_sign=-1.0)
        sd_b = E.apply_edit(wld.sd, bad["dW"], bad["db"], NB)
        gb_ = wld.gains(sd_b)
        ok_b, rows_b = E.read1_verdict((np.median(gb_[0]), np.median(gb_[1])), (np.median(gv), np.median(ga)), (1.0, 1.9), alpha)
        assert not ok_b, rows_b                                               # wrong sign: read 1 FAILS
        assert all(not r["within"] for r in rows_b)


def test_alpha0_read1_is_trivially_the_model_gains():
    wld = ToyWorld()
    gv, ga = wld.gains(wld.sd)
    gb = np.stack([np.ones(wld.n), np.full(wld.n, 1.9)], 1)
    kin3 = np.stack([wld.v0, wld.a0, np.zeros(wld.n)], 1)
    f = E.fit_edit(wld.h2(wld.v0, wld.a0), kin3, gv, ga, gb, wld.r, alpha=0.0)
    sd0 = E.apply_edit(wld.sd, f["dW"], f["db"], NB)
    assert E.state_identical(wld.sd, sd0)
    g0 = wld.gains(sd0)
    assert np.array_equal(g0[0], gv) and np.array_equal(g0[1], ga)
