"""p5_lib: shared pieces for the P5 read (THEORY_CRAWL_AMEND_3.md, sha256 dcb82ae9...). Dev-box CPU, zero training.

* the model's own a0: ``tanitad.data.v8f_mini_data.yaw_kin_from_track`` (the trainer ran ``--kin-mode yaw``) = the ONE-frame backward speed
  difference of the NATIVE 10 Hz track at raw frame 2t, divided by 0.1 s (``ego_history.ego_channels_from_poses``, dt 0.1);
* ``a0_indep``: an independent numpy float64 re-implementation of the same quantity straight from the track's speed column;
* ``sg_accel``: the label-side acceleration, a local quadratic (Savitzky-Golay order 2) fit of logged speed over +-m native frames, derivative at the centre
  (closed form: sum_k k v_{k0+k} / (dt sum_k k^2)); tested against np.polyfit and scipy.signal.savgol_filter in test_p5.py;
* ``lam_stat``: lambda = 1 - Var(e_perp) / Var(a0_perp), e = a0 - a, perp = residual of an OLS on the other ego inputs.
Import-safe: nothing is read at import time.   Run with  CUDA_VISIBLE_DEVICES=-1  python -I.
"""
import os
import sys
import json
import hashlib

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np

ART = "E:/Projects/TanitAD-artifacts"
TREE = ART + "/dagt1_tree_b43c7a9a"
HERE = ART + "/dagland/motion_c/theory/p5"
THEORY = ART + "/dagland/motion_c/theory"
CARD = THEORY + "/THEORY_CRAWL_AMEND_3.md"
CARD_SHA = "dcb82ae94c22ffbeb1353704ae6951f1fce2143eebe72d7b5544a4a1fce7f394"
TIP = "1f6a0f1f2dc87576c4167d9c2beb96cfa84aaf6a"
TRAIN_CACHE = "F:/TanitAD-scratch/speedread/cache"
TRAIN_MANIFEST = "F:/TanitAD-scratch/c_moved_20261103/box2_ship/logs/v8f_dscale/manifest_train.jsonl"
TRAIN_JOIN = "F:/TanitAD-scratch/aside_from_E/probe_scratch/trigger_probe_scratch/join_train_rebuilt.npz"
TRACKS = ART + "/sfresh_inputs/inputs/native_tracks.pt"
WINDOW_LIST_SHA = "e144adcd9b6273f2b44ad7dfa162931b90e159f7b3911e0893b25777b2ae8168"
BAND = (3.0, 8.0)
DT = 0.1                     # the native track interval (D.DT_NATIVE); MEASURED in the gate (displacement / speed)
SG_M = {"pm0.3s": 3, "pm0.5s": 5, "pm0.8s": 8}          # half-widths in native frames
GAIN_TOTAL = {"s0": 1.265, "s1": 1.112}                  # WSP_READ.md lines 31-44: total ds/da0 (median over held-out windows, 3-8 m/s)
GAIN_BASE = 1.899                                        # the non-learned base's ds/da0
RATIO = {"s0": 0.666, "s1": 0.586}                       # the card's gain ratios (as written there)
TOL = 0.10                                               # the card's tolerance: ratio within lambda +- 0.10
UB_MAX = 0.90                                            # the card's upper-bound limit


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def assert_card_frozen():
    got = sha256_file(CARD)
    if got != CARD_SHA:
        raise SystemExit(f"REFUSED: the card is not the frozen one ({got})")
    return got


def setup_tree():
    """Put the dagt1 tree (the source that trained / read C0) on sys.path; the ego pieces used here are identical on the tip (diff checked: tip adds lead labels + get_kin only)."""
    for p in (TREE + "/stack", TREE + "/taniteval"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import tanitad.data.v8f_mini_data as D
    assert os.path.abspath(D.__file__).replace("\\", "/").startswith(TREE), D.__file__
    return D


def load_ci():
    """The tip's taniteval/ci.py (vendored read copy in ref/, sha256 pinned against ``git show <tip>:taniteval/taniteval/ci.py``)."""
    import importlib.util
    p = HERE + "/ref/ci_tip.py"
    spec = importlib.util.spec_from_file_location("ci_tip", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ci_tip"] = m
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------------------ estimators (pure numpy, tested)
def _take(v, idx):
    """v[idx] with NaN where idx is outside the track (numpy would silently wrap a negative index to the END of the array)."""
    v = np.asarray(v, np.float64)
    idx = np.asarray(idx)
    ok = (idx >= 0) & (idx < len(v))
    return np.where(ok, v[np.clip(idx, 0, len(v) - 1)], np.nan)


def a0_backward(v, k0, lag=1, dt=DT):
    """(v[k0] - v[k0 - lag]) / (lag * dt)  at raw frames k0 (array). lag 1 = the model's a0 (one native frame, 0.1 s); lag 2 = the 0.2 s reading of the card's prose."""
    k0 = np.asarray(k0)
    return (_take(v, k0) - _take(v, k0 - lag)) / (lag * dt)


def a0_forward(v, k0, dt=DT):
    """DELIBERATE-REGRESSION MUTANT: the forward difference (reads the first future frame)."""
    k0 = np.asarray(k0)
    return (_take(v, k0 + 1) - _take(v, k0)) / dt


def sg_accel(v, k0, m, dt=DT):
    """Derivative at raw frame k0 of the least-squares quadratic fit of v over frames k0-m..k0+m: sum_k k v[k0+k] / (dt sum_k k^2). NaN if any support frame is NaN."""
    k0 = np.asarray(k0)
    ks = np.arange(-m, m + 1)
    s2 = float((ks * ks).sum())
    acc = np.zeros(len(k0), np.float64)
    for k in ks:
        acc += k * _take(v, k0 + k)            # k = 0 contributes 0 * v (NaN stays NaN: a NaN anywhere in the support gives NaN)
    return acc / (dt * s2)


def design(v0, kappa, kind="lin"):
    """The nuisance design (the model's other ego inputs at the present tick, a0 excluded).
    lin    : [1, v0, kappa0]                                   (primary: the two other kin3 entries the model reads)
    p10    : [1, v0, kappa0, v0^2, v0*kappa0, v0^2*kappa0]     (every one of the model's 8 p10 features that does NOT contain a0; the other two, a0 and a0*kappa0 / v0*a0, contain it)
    none   : [1]                                               (centering only; the MUTANT 'no residualisation' uses this)"""
    v0 = np.asarray(v0, np.float64)
    kappa = np.asarray(kappa, np.float64)
    one = np.ones_like(v0)
    if kind == "lin":
        cols = [one, v0, kappa]
    elif kind == "p10":
        cols = [one, v0, kappa, v0 * v0, v0 * kappa, v0 * v0 * kappa]
    elif kind == "none":
        cols = [one]
    else:
        raise ValueError(kind)
    return np.stack(cols, 1)


def residual(y, X):
    """OLS residual of y on X (least squares, min-norm)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def lam_parts(a0, a, v0, kappa, kind="lin"):
    """All the pieces of one lambda read on one window set (ddof 0 throughout; the ratio does not depend on it)."""
    a0 = np.asarray(a0, np.float64)
    a = np.asarray(a, np.float64)
    e = a0 - a
    X = design(v0, kappa, kind)
    a0p, ep, ap = residual(a0, X), residual(e, X), residual(a, X)
    va0, ve, va = float(a0p.var()), float(ep.var()), float(ap.var())
    cov = float(np.mean((a0p - a0p.mean()) * (ap - ap.mean())))
    return {"lam": 1.0 - ve / va0, "var_a0_perp": va0, "var_e_perp": ve, "var_a_perp": va, "slope_a_on_a0_perp": cov / va0,
            "corr_e_a_perp": float(np.corrcoef(ep, ap)[0, 1]) if ve > 0 and va > 0 else float("nan"), "n": int(len(a0))}


def lam_stat(a0, a, v0, kappa, kind="lin"):
    return lam_parts(a0, a, v0, kappa, kind)["lam"]


def gram_stats(a0, a, v0, kappa, kind, eid_idx, n_ep):
    """Per-episode sufficient statistics for the FAST bootstrap: for z = [X, a0, e] the Gram matrix sum_w z z^T of each episode. Shape [n_ep, p+2, p+2]."""
    X = design(v0, kappa, kind)
    z = np.concatenate([X, np.asarray(a0, np.float64)[:, None], (np.asarray(a0, np.float64) - np.asarray(a, np.float64))[:, None]], 1)
    p = z.shape[1]
    G = np.zeros((n_ep, p, p))
    for i in range(p):
        for j in range(i, p):
            G[:, i, j] = np.bincount(eid_idx, weights=z[:, i] * z[:, j], minlength=n_ep)
            G[:, j, i] = G[:, i, j]
    return G


def lam_from_gram(G):
    """lambda from a summed Gram [p, p] (last two coordinates = a0, e): residual sums of squares after projecting out the first p-2 columns."""
    p = G.shape[0]
    q = p - 2
    Gxx = G[:q, :q]
    Gxy = G[:q, q:]
    Gyy = G[q:, q:]
    R = Gyy - Gxy.T @ np.linalg.solve(Gxx, Gxy)
    return 1.0 - R[1, 1] / R[0, 0]


def fast_cluster_bootstrap(G_ep, n_boot=2000, seed=0, alpha=0.05):
    """Episode-cluster bootstrap on per-episode Gram matrices. The resampling is ci.py's: rng = default_rng(seed); pick = rng.choice(uniq, size=n_ep, replace=True) -- here the
    uniq episodes are 0..n_ep-1 in np.unique (sorted-string) order, which is the order the caller must give G_ep."""
    n_ep = G_ep.shape[0]
    rng = np.random.default_rng(seed)
    allp = np.arange(n_ep)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(allp, size=n_ep, replace=True)
        vals[b] = lam_from_gram(G_ep[pick].sum(0))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lo": float(lo), "hi": float(hi), "mean": float(vals.mean()), "sd": float(vals.std()), "n_boot": int(n_boot), "n_episodes": int(n_ep)}


def verdict(lam_point, lam_hi, ratios=RATIO, tol=TOL, ub_max=UB_MAX):
    """The card's section 3 rule, verbatim: HOLDS iff hi < 0.90 AND every seed's ratio within lambda +- 0.10; FAILS if hi >= 0.90 or either ratio outside. (Never NOT EVALUABLE here: the gate is read before.)"""
    in_tol = {k: bool(abs(r - lam_point) <= tol + 1e-12) for k, r in ratios.items()}      # inclusive, with a float-noise epsilon (0.666 - 0.566 is 0.10000000000000003)
    ub_ok = bool(lam_hi < ub_max)
    holds = ub_ok and all(in_tol.values())
    return {"word": "HOLDS" if holds else "FAILS", "upper_bound_below_0.90": ub_ok, "ratio_within_lambda_pm_0.10": in_tol,
            "lambda_window_for_both_seeds": [max(r - tol for r in ratios.values()), min(r + tol for r in ratios.values())]}


# ------------------------------------------------------------------------------------------------ gate comparator and the synthetic control world
def gate_compare(a0_model, a0_repro, tol=1e-4, n_min=1000):
    """The card's gate: max |a0_reproduced - a0_model| <= tol on at least n_min windows (NaN pairs are not counted)."""
    d = np.abs(np.asarray(a0_model, np.float64) - np.asarray(a0_repro, np.float64))
    fin = np.isfinite(d)
    mx = float(d[fin].max()) if fin.any() else float("nan")
    return {"n": int(fin.sum()), "max_abs": mx, "pass": bool(fin.sum() >= n_min and mx <= tol)}


SYN_LEN, SYN_K0 = 31, 15             # one window per synthetic episode: frame 15 of a 31-frame series (the +-0.8 s support, 8 frames each side, is inside)


def synth_world(n_ep=20000, sigma=0.06, alpha=0.5, sigma_u=0.8, seed=0):
    """A synthetic series with a KNOWN noise level. Per episode: true acceleration a_c = alpha (v_s - 5.5) + u (u ~ N(0, sigma_u^2)), constant over the episode, v_s ~ U(3, 8);
    speed v(k) = v_s + a_c * 0.1 (k - 15) + n_k with n_k white N(0, sigma^2); a curvature kappa ~ N(0, 0.02^2) independent of everything. ONE window per episode, at frame 15.
    The estimators under test (a0_backward, sg_accel) are then run on the flat series exactly as on the real track. Because the speed is linear, the SG fit is exact on the signal: a = a_c + SG noise."""
    rng = np.random.default_rng(seed)
    vs = rng.uniform(3.0, 8.0, n_ep)
    ac = alpha * (vs - 5.5) + rng.normal(0.0, sigma_u, n_ep)
    k = np.arange(SYN_LEN)
    v = vs[:, None] + ac[:, None] * DT * (k[None, :] - SYN_K0) + rng.normal(0.0, sigma, (n_ep, SYN_LEN))
    kappa = rng.normal(0.0, 0.02, n_ep)
    flat = v.reshape(-1)
    k0 = SYN_LEN * np.arange(n_ep) + SYN_K0
    return {"v_flat": flat, "k0": k0, "v0": flat[k0], "kappa0": kappa, "a_true": ac, "eid": np.arange(n_ep), "params": dict(n_ep=n_ep, sigma=sigma, alpha=alpha, sigma_u=sigma_u, seed=seed)}


def synth_expected_lambda(sigma=0.06, alpha=0.5, sigma_u=0.8, m=5, dt=DT):
    """The population value of lambda for synth_world, from the coefficient vectors of the white noise in a0 and in the SG a (no simulation)."""
    js = np.arange(-8, 9)
    c_a0 = np.zeros(len(js))
    c_a0[js == 0] = 1.0 / dt
    c_a0[js == -1] = -1.0 / dt
    ks = np.arange(-m, m + 1)
    s2 = float((ks * ks).sum())
    c_a = np.where(np.abs(js) <= m, js / (dt * s2), 0.0)
    c_e = c_a0 - c_a
    vs_var = 25.0 / 12.0
    var_v0 = vs_var + sigma ** 2
    var_a0 = alpha ** 2 * vs_var + sigma_u ** 2 + sigma ** 2 * float((c_a0 ** 2).sum())
    cov_a0_v0 = alpha * vs_var + sigma ** 2 * float(c_a0[js == 0][0])
    var_e = sigma ** 2 * float((c_e ** 2).sum())
    cov_e_v0 = sigma ** 2 * float(c_e[js == 0][0])
    va0_perp = var_a0 - cov_a0_v0 ** 2 / var_v0
    ve_perp = var_e - cov_e_v0 ** 2 / var_v0
    lam = 1.0 - ve_perp / va0_perp
    lam_none = 1.0 - var_e / var_a0
    return {"lambda": lam, "lambda_without_residualisation": lam_none, "var_a0_perp": va0_perp, "var_e_perp": ve_perp, "var_a0": var_a0, "var_e": var_e}


def synth_read(w, a0_kind="backward", m=5, design_kind="lin"):
    """lambda on a synthetic world through the production estimators. a0_kind: backward | forward | lag2 | a_itself."""
    a = sg_accel(w["v_flat"], w["k0"], m)
    if a0_kind == "backward":
        a0 = a0_backward(w["v_flat"], w["k0"], 1)
    elif a0_kind == "lag2":
        a0 = a0_backward(w["v_flat"], w["k0"], 2)
    elif a0_kind == "forward":
        a0 = a0_forward(w["v_flat"], w["k0"])
    elif a0_kind == "a_itself":
        a0 = a.copy()
    else:
        raise ValueError(a0_kind)
    return {"lam": lam_stat(a0, a, w["v0"], w["kappa0"], design_kind), "a0": a0, "a": a}
