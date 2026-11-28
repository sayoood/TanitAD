"""wsp_edit (E-GAIN-D copy): E-GAIN(alpha), the closed-form edit of PREREG_WSP_GAIN.md section 2 (card sha256 92d61dcb...), with the TARGETS made parameters for
PREREG_WSP_GAIN_D.md section 2 (card sha256 d6e691a3...). Pure numpy / torch tensor arithmetic. Zero training.
Copied from wsp_edit/wsp_edit.py (banked, ad4cda1b...). The ONLY change is in ``fit_edit``: two optional keywords, ``g_target`` (the gains to reach, default = the base's median gains = the banked
behaviour) and ``shift`` (the band-mean plan-speed shift to reach, default the banked 0.35; a signed value is allowed and 0.0 is allowed). With the defaults the outputs are the banked outputs bit for bit
(tests/test_wsp_edit_d.py::test_banked_targets_reproduce_banked_fit; scripts/repro_banked.py proves it on the checkpoints' sha256). Nothing else is changed.
E-GAIN-D: g_target = g_data (OLS of the realised plan speed on 1, v0, a0, kappa0 on the fit windows), shift = mean realised speed - mean model plan speed on the fit windows.

The edit touches ONE tensor pair: the x (longitudinal) rows of head_o's LAST linear layer (`head_o.2.weight[0:nb]`, `head_o.2.bias[0:nb]`). Lateral rows nb:2nb and every other tensor stay bit for bit.
  r_res  [nb]   s_res = r_res . out_x : the plan-speed (0.8 s secant at +2 s) readout of head_o's raw x outputs through the fixed basis and res_scale_m
  u      [nb]   = r_res / (r_res . r_res)        so r_res . u = 1.0: it changes plan speed at +2 s by exactly 1 m/s
  w_a,w_v [p]   ridge directions (lambda = 1e-3 * trace Cov(h2)) decoding the measured a0 and v0 from h2 (the post-GELU input of the last layer, p = 2d = 512), fitted on the TRAIN subset
  d_g           = (median base gain - median model gain) on the same TRAIN subset, per coordinate (a0, v0)
  dW = alpha * outer(u, dg_a * w_a + dg_v * w_v)                      [nb, p]
  db = alpha * u * 0.35 * kappa,  kappa = 1 - m / 0.35,  m = mean over the fit windows of h2 . (dg_a w_a + dg_v w_v)
       so the edit's band-mean plan-speed shift over the fit windows is exactly +0.35 * alpha m/s (it undoes the measured -0.35 m/s of the learned parts)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np
import torch

SHIFT = 0.35          # card section 2: the measured mean shift of the learned parts, m/s
LAM_FRAC = 1e-3       # card section 2: lambda = 1e-3 * trace(Cov(h2))
J_HI, J_LO, DT_SEC = 11, 7, 0.8
KEY_W, KEY_B = "head_o.2.weight", "head_o.2.bias"


def readout(basis, res_scale_m, j_hi=J_HI, j_lo=J_LO, dt=DT_SEC):
    """r_res [nb]: d s_res / d (raw head_o x output), s = (x30[11] - x30[7]) / 0.8, x30[j] = ... + res60[2j + 1]."""
    b = np.asarray(basis, np.float64)
    return float(res_scale_m) * (b[2 * j_hi + 1] - b[2 * j_lo + 1]) / dt


def unit_direction(r_res):
    r = np.asarray(r_res, np.float64)
    return r / float(r @ r)


def ridge_direction(H, y, lam_frac=LAM_FRAC):
    """w = (C + lam I)^-1 cov(h, y), C = Cov(h) (1/n), lam = lam_frac * trace(C). H centered inside; the intercept is not part of w."""
    H = np.asarray(H, np.float64)
    y = np.asarray(y, np.float64)
    n = len(H)
    Hc = H - H.mean(0)
    yc = y - y.mean()
    C = Hc.T @ Hc / n
    lam = lam_frac * float(np.trace(C))
    return np.linalg.solve(C + lam * np.eye(C.shape[0]), Hc.T @ yc / n), lam


def fit_edit(H, kin3, gv, ga, gb, r_res, alpha, lam_frac=LAM_FRAC, shift=SHIFT, u_sign=1.0, g_target=None):
    """H [n, p] h2 on the fit windows; kin3 [n, 3] (v0, a0, kappa); gv, ga [n] the model's measured gains; gb [n, 2] the base's gains (v0, a0).
    Returns dW [nb, p], db [nb] and every fitted quantity. ``u_sign`` exists for the wrong-sign mutant test only.
    E-GAIN-D targets: ``g_target`` = (g_v, g_a) the gains to reach (default None = the base's median gains, the banked behaviour); ``shift`` = the band-mean plan-speed shift to reach over the fit
    windows, m/s (default the banked 0.35; signed, and 0.0 is allowed: then db = alpha u (0 - m), the Delta-m of PREREG_WSP_GAIN_D.md section 2)."""
    H = np.asarray(H, np.float64)
    kin3 = np.asarray(kin3, np.float64)
    u = unit_direction(r_res) * float(u_sign)
    w_v, lam_v = ridge_direction(H, kin3[:, 0], lam_frac)
    w_a, lam_a = ridge_direction(H, kin3[:, 1], lam_frac)
    g_v_model, g_a_model = float(np.median(gv)), float(np.median(ga))
    g_v_base, g_a_base = float(np.median(gb[:, 0])), float(np.median(gb[:, 1]))
    g_v_tgt, g_a_tgt = (g_v_base, g_a_base) if g_target is None else (float(g_target[0]), float(g_target[1]))
    dg_v, dg_a = g_v_tgt - g_v_model, g_a_tgt - g_a_model
    d = dg_a * w_a + dg_v * w_v
    m = float(np.mean(H @ d))                                # m/s per unit alpha, per unit (u . r_res = 1)
    if shift != 0.0:
        kappa = 1.0 - m / shift
        db = float(alpha) * u * shift * kappa
    else:                                                    # shift 0 (g_data := g_model leaves nothing to move): db = alpha u (0 - m), kappa undefined
        kappa = float("nan")
        db = float(alpha) * u * 0.0
    dW = float(alpha) * np.outer(u, d)
    return {"dW": dW, "db": db, "u": u, "w_v": w_v, "w_a": w_a, "lam_v": lam_v, "lam_a": lam_a, "dg_v": dg_v, "dg_a": dg_a, "g_v_model": g_v_model, "g_a_model": g_a_model,
            "g_v_base": g_v_base, "g_a_base": g_a_base, "g_v_target": g_v_tgt, "g_a_target": g_a_tgt, "shift": float(shift), "m": m, "kappa": kappa, "alpha": float(alpha),
            "decode_r2_v": float(1 - np.var((H - H.mean(0)) @ w_v - (kin3[:, 0] - kin3[:, 0].mean())) / np.var(kin3[:, 0])),
            "decode_r2_a": float(1 - np.var((H - H.mean(0)) @ w_a - (kin3[:, 1] - kin3[:, 1].mean())) / np.var(kin3[:, 1]))}


def apply_edit(sd, dW, db, nb):
    """New state dict: the x rows 0:nb of head_o.2 get +dW / +db (added in float64, cast back); an entry whose delta is exactly zero is left untouched (so alpha = 0 is bit for bit)."""
    out = {k: v.clone() for k, v in sd.items()}
    W = out[KEY_W]
    B = out[KEY_B]
    assert W.shape[0] == 2 * nb and B.shape[0] == 2 * nb, (W.shape, B.shape)
    dWt = torch.as_tensor(np.asarray(dW, np.float64))
    dbt = torch.as_tensor(np.asarray(db, np.float64))
    assert dWt.shape == (nb, W.shape[1]) and dbt.shape == (nb,)
    Wx = W[:nb].double()
    Bx = B[:nb].double()
    W[:nb] = torch.where(dWt == 0, Wx, Wx + dWt).to(W.dtype)
    B[:nb] = torch.where(dbt == 0, Bx, Bx + dbt).to(B.dtype)
    return out


def tensors_equal_bytes(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.contiguous().numpy().tobytes() == b.contiguous().numpy().tobytes()


def lateral_unchanged(sd0, sd1, nb):
    """True iff every tensor except head_o.2's x rows is byte-identical (the lateral rows nb:2nb of head_o.2 included)."""
    if set(sd0) != set(sd1):
        return False
    for k in sd0:
        if k in (KEY_W, KEY_B):
            if not tensors_equal_bytes(sd0[k][nb:], sd1[k][nb:]):
                return False
        elif not tensors_equal_bytes(sd0[k], sd1[k]):
            return False
    return True


def state_identical(sd0, sd1):
    return set(sd0) == set(sd1) and all(tensors_equal_bytes(sd0[k], sd1[k]) for k in sd0)


def read1_verdict(g_edit, g_model, g_base, alpha, tol=0.10):
    """Read 1 (card section 3): the edited model's median gains within ``tol`` of the alpha-restored targets (1 - alpha) g_model + alpha g_base, per coordinate.
    g_* = (g_v, g_a). Returns (ok, rows)."""
    rows = []
    ok = True
    for i, nm in enumerate(("dS/dv0", "dS/da0")):
        tgt = (1.0 - alpha) * g_model[i] + alpha * g_base[i]
        rel = abs(g_edit[i] - tgt) / abs(tgt)
        rows.append({"gain": nm, "edited": float(g_edit[i]), "target": float(tgt), "model": float(g_model[i]), "base": float(g_base[i]), "rel_err": float(rel), "within": bool(rel <= tol)})
        ok = ok and rel <= tol
    return ok, rows
