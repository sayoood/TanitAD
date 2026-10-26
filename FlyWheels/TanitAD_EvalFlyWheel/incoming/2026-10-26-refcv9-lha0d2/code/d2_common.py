"""d2_common.py -- shared, EVAL-FREE helpers for the LHA0-D2 arm (PREREG_LHA0_D2).  No eval139 file is opened here.

Design 2: a binary logistic head  P(decel set | x),  x = standardised [a0, v0/10, psi_dot0]  (3 features).
  * decel set  = LON6 indices G_DECEL = (2, 3, 4) = (BRAKE_TO, CREEP, HOLD)    -- l2_common.G_DECEL, the set K-HEAD's renormalisation uses
  * complement = LON6 indices G_CRUISE = (0, 1)   = (FOLLOW, CRUISE)           -- l2_common.G_CRUISE
  * PRIMARY D2 fits ALL FIT rows (ACCELERATE is a negative); the beside head D2NOACC drops ACCELERATE rows, as K-HEAD's P(decel)/(P(decel)+P(follow+cruise)) renormalisation does.
  * objective = sum_i CE_i + lambda/2 ||w||^2, intercept unpenalised (sklearn LogisticRegression(C = 1/lambda, lbfgs)), LH's own convention.
Score used for ranking = the head's LOGIT (strictly monotone in P, so AUC is identical; no sigmoid saturation ties).
Frozen head = plain-number JSON.  CPU only.  NEW file (LHA0-D2 agent, 2026-10-26).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE = Path("C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58")
LONHEAD = Path("C:/Users/Admin/refcv8_pf/refcv9_lonhead")
LHA0 = Path("C:/Users/Admin/refcv8_pf/refcv9_lha0")
for _p in (str(TREE / "stack"), str(LONHEAD / "code"), "C:/Users/Admin/refcv8_pf/refcv9_l2/code", "C:/Users/Admin/refcv8_pf/refcv9_l1/code"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCHEMA = "tanitad.refcv9_lha0d2.d2/1"
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)               # the task's grid (== lh_fit.LAMBDAS)
G_DECEL, G_CRUISE, G_ACCEL = (2, 3, 4), (0, 1), (5,)         # LON6 indices; asserted against l2_common at fit time
FEATURES = ("a0", "v0_over_10", "psi_dot0")
UNIVERSE = Path("F:/c_overflow_2026-10-26/tmp/lha0/universe.npz")   # the diag's own scratch table (regenerable: lha0_diag.py prep)
BOOT_B, BOOT_SEED = 2000, 0
BAR_DELTA = 0.030


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def pr(x, d=4):
    return None if x is None else round(float(x), d)


def design(a0, v0, psi) -> np.ndarray:
    """raw (un-standardised) design matrix, columns FEATURES: [a0, v0/10, psi_dot0]."""
    return np.column_stack([np.asarray(a0, np.float64), np.asarray(v0, np.float64) / 10.0, np.asarray(psi, np.float64)])


def std_stats(X: np.ndarray) -> tuple:
    """mean / sd (+1e-9, LH's std_of convention) fitted on the TRAINING rows only."""
    return X.mean(0), X.std(0) + 1e-9


def fit_binary(X: np.ndarray, y: np.ndarray, lam: float) -> dict:
    """ridge logistic on already-standardised X; y in {0, 1}; objective sum CE + lam/2 ||w||^2 (sklearn C = 1/lam); intercept unpenalised."""
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1.0 / lam, solver="lbfgs", max_iter=5000, tol=1e-8)
    m.fit(X, np.asarray(y, np.int64))
    assert [int(c) for c in m.classes_] == [0, 1], m.classes_
    return {"w": m.coef_[0].astype(np.float64).copy(), "b": float(m.intercept_[0]), "n_iter": int(np.max(m.n_iter_))}


def logit_of(head: dict, Xstd: np.ndarray) -> np.ndarray:
    return Xstd @ np.asarray(head["w"], np.float64) + float(head["b"])


def bce(logit: np.ndarray, y: np.ndarray) -> np.ndarray:
    """per-row binary cross-entropy from a logit, numerically stable (no clipping)."""
    y = np.asarray(y, np.float64)
    return np.logaddexp(0.0, logit) - y * logit


def freeze(path: Path, fit: dict, mu, sd, lam: float, meta: dict) -> str:
    blob = {"schema": SCHEMA, "features": list(FEATURES), "mu": [float(x) for x in mu], "sd": [float(x) for x in sd], "w": [float(x) for x in fit["w"]], "b": float(fit["b"]),
            "lambda": float(lam), "objective": "sum CE + lambda/2 ||w||^2, intercept unpenalised (sklearn C = 1/lambda)", "n_iter": int(fit["n_iter"]),
            "decel_set_lon6_idx": list(G_DECEL), "complement_lon6_idx": list(G_CRUISE) + (list(G_ACCEL) if meta.get("accel_is_negative") else []),
            "score": "logit = ((x - mu) / sd) @ w + b,  x = [a0, v0/10, psi_dot0]  (a0 in m/s^2, psi_dot0 in rad/s, v0 in m/s)", **meta}
    path.write_text(json.dumps(blob, indent=1, sort_keys=True), encoding="utf-8")        # json floats are repr round-trip exact
    return sha256_file(path)


def load_frozen(path: Path, registered_sha: str | None = None) -> dict:
    got = sha256_file(path)
    if registered_sha is not None and (len(got) != 64 or got != registered_sha):
        raise SystemExit(f"[d2] {path} sha256 {got} != registered {registered_sha!r} -- refusing")
    b = json.loads(Path(path).read_text(encoding="utf-8"))
    assert b["schema"] == SCHEMA and tuple(b["features"]) == FEATURES, b.get("schema")
    b["_sha256"] = got
    return b


def head_logit(blob: dict, a0, v0, psi) -> np.ndarray:
    X = design(a0, v0, psi)
    Xs = (X - np.asarray(blob["mu"], np.float64)) / np.asarray(blob["sd"], np.float64)
    return Xs @ np.asarray(blob["w"], np.float64) + float(blob["b"])


def lh_logit(logits6: np.ndarray) -> np.ndarray:
    """the K-HEAD score on the logit scale, exactly:  log P(decel) - log P(follow+cruise)  of the 6-way LON6 softmax (log-domain, no clipping)."""
    from scipy.special import logsumexp
    return logsumexp(logits6[:, list(G_DECEL)], axis=1) - logsumexp(logits6[:, list(G_CRUISE)], axis=1)


def zscore(s: np.ndarray) -> np.ndarray:
    """label-free standardisation over the scored population (no parameter fitted on a label)."""
    return (s - s.mean()) / s.std()


def permute_within_episode(eid: np.ndarray, cols: list, seed: int = 0) -> list:
    """ONE shared permutation per episode (episodes in sorted order, np.random.default_rng(seed)), applied to every column in `cols`
    (the eval-side image of LHA0_DIAG's C4: lha0_diag.py prep)."""
    rng = np.random.default_rng(seed)
    out = [np.empty_like(c) for c in cols]
    for e in np.unique(eid):
        ix = np.flatnonzero(eid == e)
        pm = rng.permutation(len(ix))
        for o, c in zip(out, cols):
            o[ix] = c[ix][pm]
            assert np.array_equal(np.sort(o[ix]), np.sort(c[ix]))
    return out


def paired_boot(y, eid, scores: dict, B: int = BOOT_B, seed: int = BOOT_SEED) -> dict:
    """K-HEAD's loop verbatim (k_head.py main): episode-resampling bootstrap, B 2000, seed 0, weight = times the episode was drawn,
    every arm scored on the SAME draws; returns point AUCs, bootstrap vectors, WAuc unit-weight self-check."""
    import k_head as KH
    y = np.asarray(y, bool)
    fns = {k: KH.WAuc(y, s) for k, s in scores.items()}
    pt = {k: KH.auc_np(y, s) for k, s in scores.items()}
    unit_ok = {k: bool(abs(f(np.ones(len(y))) - pt[k]) < 1e-9) for k, f in fns.items()}
    ue, inv = np.unique(np.asarray(eid), return_inverse=True)
    rng = np.random.default_rng(seed)
    bt = {k: np.empty(B) for k in scores}
    for b in range(B):
        cnt = np.bincount(rng.integers(0, len(ue), len(ue)), minlength=len(ue)).astype(np.float64)
        w = cnt[inv]
        for k, f in fns.items():
            bt[k][b] = f(w)
    return {"pt": pt, "bt": bt, "unit_ok": unit_ok, "n_episodes": int(len(ue)), "n": int(len(y)), "n_pos": int(y.sum())}


def diff(R: dict, a: str, b: str) -> dict:
    import k_head as KH
    d = R["bt"][a] - R["bt"][b]
    return {"point": pr(R["pt"][a] - R["pt"][b]), "ci95": [pr(x) for x in KH.ci(d)], "boot_se": pr(np.std(d, ddof=1))}


# ------------------------------------------------------------------------------------------------------------------ verdict logic (pure; unit-tested in d2_selftest.py)
def clears_bar(point: float, ci_lo: float) -> bool:
    """the K-HEAD-D2 bar on UNROUNDED numbers: AUC(arm) - AUC(LH) >= +0.030 AND the paired episode-cluster bootstrap 95% CI excludes 0 (lower bound > 0)."""
    return bool(np.isfinite(point) and np.isfinite(ci_lo) and point >= BAR_DELTA and ci_lo > 0.0)


def verdict_word(controls_ok: bool, d2_clears: bool, perm_clears: bool) -> str:
    """STOP/INVALID if a known-value control failed; VOID if the within-episode-permuted control clears the bar; else GO / NO-GO."""
    if not controls_ok:
        return "INVALID (a known-value control failed)"
    if perm_clears:
        return "VOID (the a0/psi_dot0-permuted control clears the bar: the bar is passable without the per-window value)"
    return "GO" if d2_clears else "NO-GO"
