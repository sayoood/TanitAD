"""NC-FASTCAL library: per-stratum Platt scaling of the NC logit, the EP-led composite with the calibrated NC, the registered decision functions, the controls.

REGISTERED SOURCE: PREREG_NC_FASTCAL.md (sha256 PREREG_SHA256 below).  Every constant cites its section.  No stage of this package computes a quantity unless the prereg file (and every amendment)
still hashes to its sealed value (`require_prereg`).  CPU only; numpy plus the pinned landed modules (pickrule_lib, epr1, fastneg_lib).
"""
from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = "nc-fastcal-v1"
PREREG_SHA256 = "4db61ce178c16e1ba862e4087ff55712fb12c03ed6111a886979aeccfef8cc4f"          # PREREG_NC_FASTCAL.md, sealed 2026-10-19T10:24:44+0200 before any real-data stage
AMENDMENTS: dict = {}                                                                         # {"PREREG_NC_FASTCAL_AMENDMENT<k>.md": sha256}; none at sealing
PICKRULE_LIB_SHA256 = "555c0a3468acd94adeeeb10f47527644a2b9d49722f554fed2da1fb29668eaeb"     # prereg s2
EPR1_SHA256 = "a751c63eb8e38343a7ec1286114a158046ce4ec13390b2eea0a4b1d2237c38d8"             # prereg s2
FASTNEG_LIB_SHA256 = "0c0b399b37a47ee3e17f9d5484bc7f9521ff7dbe6bea37e8c7a38042668667ca"      # prereg s2
STRATA_JSON_SHA256 = "67f43e55dd7c693e31cd991efc3d51fedc278331ba4f8f882e6adbdae8203dc6"      # prereg s3
SPEEDS_NPZ_SHA256 = "433fd274abdae0de12b027ca27a4c1178fa64d60c25cdb48eb9acae22ba103de"       # prereg s3
REF = "C:/Users/Admin/par18/ncfastcal/ref"
_RAW = "2026-09-20-refe-plan/raw"                                                              # sibling-dir anchor inside the repo package tree

N_STRATA, N_FOLDS = 5, 5                         # prereg s3 (S, Q1..Q4) / s2 (rank mod 5)
STRATA_NAMES = ("S", "Q1", "Q2", "Q3", "Q4")
SEEDS = (0, 1)                                   # prereg s2
N_LOGS, BOOT_B, BOOT_SEED = 117, 10000, 20260927   # prereg s2
NC_UB = 0.001                                    # prereg s6 (i): the upper bound of dNC must be STRICTLY below this
PDMS_LB = 0.0                                    # prereg s6 (ii): the lower bound of dPDMS x100 must be >= this
NULL_REPS, NULL_SALT, NULL_SEED_BASE = 5, 7919, 20261019      # prereg s7 C-NULL
NULL_VOID_FIRES = 2
MIN_ROWS, MAX_ABS_PARAM = 1000, 50.0             # prereg s4 fit-failure rule
GRAD_TOL, GRAD_ACCEPT, MAX_ITER = 1e-10, 1e-8, 100
ARMIJO_C, MAX_HALVINGS = 1e-4, 60
MIN_AVAIL_GB = 3.0
NC, DAC, EP, TTC, CMF, DDC = 0, 1, 2, 3, 4, 5


class NcFastcalError(RuntimeError):
    """An integrity check, a pin or a precondition failed."""


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# hashing, pins, pinned module loading
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def require_prereg(path: str | None = None) -> str:
    """Refuse unless the prereg file AND every amendment on disk still hash to their sealed values (a change comes only by a written amendment BEFORE the number it governs)."""
    p = path or os.path.join(HERE, "..", "PREREG_NC_FASTCAL.md")
    if not os.path.exists(p):
        raise NcFastcalError(f"the sealed registration file {p} does not exist")
    sha = sha256_file(p)
    if sha != PREREG_SHA256:
        raise NcFastcalError(f"{p} sha256 {sha} != the sealed {PREREG_SHA256}: refuse (write an amendment BEFORE the number it governs)")
    d = os.path.dirname(os.path.abspath(p))
    on_disk = sorted(os.path.basename(x) for x in glob.glob(os.path.join(d, "PREREG_NC_FASTCAL_AMENDMENT*.md")))
    for name in on_disk:
        if name not in AMENDMENTS:
            raise NcFastcalError(f"amendment {name} exists on disk but is not registered in ncfastcal_lib.AMENDMENTS: refuse")
    for name, want in AMENDMENTS.items():
        f = os.path.join(d, name)
        if not os.path.exists(f):
            raise NcFastcalError(f"registered amendment {name} does not exist")
        if sha256_file(f) != want:
            raise NcFastcalError(f"amendment {name} sha256 != the sealed {want}")
    return PREREG_SHA256


def load_pinned(name: str, candidates: list, want_sha: str):
    """Import a landed module by path after checking its sha256 against the pin; a tampered or missing copy is refused."""
    if name in sys.modules and getattr(sys.modules[name], "_NCFASTCAL_PINNED", None) == want_sha:
        return sys.modules[name]
    tried = []
    for c in candidates:
        if c and os.path.exists(c):
            sha = sha256_file(c)
            if sha != want_sha:
                raise NcFastcalError(f"{name} at {c} has sha256 {sha} != the pinned {want_sha}")
            if os.path.dirname(os.path.abspath(c)) not in sys.path:
                sys.path.insert(0, os.path.dirname(os.path.abspath(c)))
            spec = importlib.util.spec_from_file_location(name, c)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            mod._NCFASTCAL_PINNED = want_sha
            mod._PICKRULE_PINNED = want_sha          # the landed pickrule_lib.load_pinned cache key
            return mod
        tried.append(c)
    raise NcFastcalError(f"{name} not found; probed: {tried}")


def _cands(rel: str, fname: str, explicit: str | None) -> list:
    return [explicit, os.path.join(HERE, "..", "..", rel, "code", fname), f"{REF}/{fname}"]


def load_pickrule(explicit: str | None = None):
    return load_pinned("pickrule_lib", _cands("2026-10-19-pickrule-audit-nc", "pickrule_lib.py", explicit), PICKRULE_LIB_SHA256)


def load_epr1(explicit: str | None = None):
    return load_pinned("epr1", _cands("2026-10-18-ep-r1", "epr1.py", explicit), EPR1_SHA256)


def load_fastneg(explicit: str | None = None):
    return load_pinned("fastneg_lib", _cands("2026-10-19-fastneg-census", "fastneg_lib.py", explicit), FASTNEG_LIB_SHA256)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# per-stratum Platt scaling (prereg s4)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _sigmoid(u: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * u))


def _bce_grad_hess(x: np.ndarray, t: np.ndarray, a: float, b: float):
    u = a * x + b
    loss = float(np.mean(np.logaddexp(0.0, u) - t * u))
    p = _sigmoid(u)
    r = p - t
    g = np.array([np.mean(r * x), np.mean(r)])
    w = p * (1.0 - p)
    H = np.array([[np.mean(w * x * x), np.mean(w * x)], [np.mean(w * x), np.mean(w)]])
    return loss, g, H


def platt_fit(x: np.ndarray, t: np.ndarray) -> dict:
    """prereg s4: minimise mean BCE-with-logits of u = a x + b against the SOFT target t, from (1, 0), by deterministic Newton (exact gradient and Hessian) with Armijo backtracking.
    -> {a, b, ok, reason, n, iters, grad_inf, loss_identity, loss_fit}.  On any failure (too few rows, singular Hessian, no convergence to grad_inf < 1e-8, non-finite, |param| > 50) a = 1, b = 0 and ok False."""
    x = np.asarray(x, np.float64).ravel()
    t = np.asarray(t, np.float64).ravel()
    n = int(len(x))
    out = {"a": 1.0, "b": 0.0, "ok": False, "reason": "", "n": n, "iters": 0, "grad_inf": float("nan"), "loss_identity": float("nan"), "loss_fit": float("nan")}
    if n < MIN_ROWS:
        out["reason"] = f"fewer than {MIN_ROWS} training rows"
        return out
    a, b = 1.0, 0.0
    loss, g, H = _bce_grad_hess(x, t, a, b)
    out["loss_identity"] = loss
    it = 0
    while it < MAX_ITER and float(np.max(np.abs(g))) >= GRAD_TOL:
        det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
        if not np.isfinite(det) or det <= 1e-14 * max(1e-300, abs(H[0, 0] * H[1, 1])):
            out["reason"] = "singular Hessian"
            return out
        step = -np.linalg.solve(H, g)
        slope = float(g @ step)
        s, accepted = 1.0, False
        for _ in range(MAX_HALVINGS):
            na, nb = a + s * step[0], b + s * step[1]
            nloss, ng, nH = _bce_grad_hess(x, t, na, nb)
            if np.isfinite(nloss) and nloss <= loss + ARMIJO_C * s * slope:
                a, b, loss, g, H, accepted = na, nb, nloss, ng, nH, True
                break
            s *= 0.5
        if not accepted:
            break
        it += 1
    gi = float(np.max(np.abs(g)))
    out.update({"iters": it, "grad_inf": gi, "loss_fit": loss})
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(gi)):
        out["reason"] = "non-finite"
        return out
    if gi >= GRAD_ACCEPT:
        out["reason"] = f"no convergence (grad_inf {gi:.3e})"
        return out
    if abs(a) > MAX_ABS_PARAM or abs(b) > MAX_ABS_PARAM:
        out["reason"] = f"|param| > {MAX_ABS_PARAM}"
        return out
    out.update({"a": float(a), "b": float(b), "ok": True, "reason": "converged"})
    return out


def identity_params() -> np.ndarray:
    p = np.zeros((N_FOLDS, N_STRATA, 2), np.float64)
    p[..., 0] = 1.0
    return p


def fit_calibration(x: np.ndarray, t: np.ndarray, code: np.ndarray, fold: np.ndarray, n_strata: int = N_STRATA, n_folds: int = N_FOLDS) -> tuple:
    """x [N,64] the NC logit, t [N,64] the NC label (soft target), code [N,64] the stratum, fold [N] the set's fold.  Fold k is fitted on the sets with fold != k ONLY.
    -> (params [n_folds, n_strata, 2] = (a, b), diag list of per-(fold, stratum) dicts)."""
    x = np.asarray(x, np.float64)
    t = np.asarray(t, np.float64)
    params = np.zeros((n_folds, n_strata, 2), np.float64)
    params[..., 0] = 1.0
    diag = []
    for k in range(n_folds):
        tr = fold != k
        xs, ts, cs = x[tr], t[tr], code[tr]
        for s in range(n_strata):
            m = cs == s
            r = platt_fit(xs[m], ts[m])
            params[k, s, 0], params[k, s, 1] = r["a"], r["b"]
            diag.append({"fold": k, "stratum": s, "stratum_name": STRATA_NAMES[s] if n_strata == N_STRATA else str(s), **r})
    return params, diag


def apply_calibration(x: np.ndarray, code: np.ndarray, fold: np.ndarray, params: np.ndarray) -> np.ndarray:
    """z' = a_{fold, stratum} x + b_{fold, stratum} per proposal, float64 [N,64].  The parameters of fold k are applied ONLY to the held-out sets of fold k."""
    ci = np.asarray(code, np.int64)
    fi = np.asarray(fold, np.int64)[:, None]
    A = params[fi, ci, 0]
    B = params[fi, ci, 1]
    return A * np.asarray(x, np.float64) + B


def composite(z6: np.ndarray, ep: np.ndarray | None = None, nc: np.ndarray | None = None) -> np.ndarray:
    """P2's logits as float64 with column EP replaced by `ep` and column NC by `nc` (every other column untouched)."""
    z = np.asarray(z6, np.float64).copy()
    if ep is not None:
        z[..., EP] = ep
    if nc is not None:
        z[..., NC] = nc
    return z


def heldout_bce(x: np.ndarray, t: np.ndarray, code: np.ndarray, fold: np.ndarray, params: np.ndarray, n_strata: int = N_STRATA) -> dict:
    """Pooled held-out BCE per stratum, before (raw logit) and after (calibrated): does Platt improve the held-out NC log-loss at all?  prereg s8."""
    xc = apply_calibration(x, code, fold, params)
    x = np.asarray(x, np.float64)
    t = np.asarray(t, np.float64)
    out = {}
    for s in range(n_strata):
        m = code == s
        if not m.any():
            continue
        bef = float(np.mean(np.logaddexp(0.0, x[m]) - t[m] * x[m]))
        aft = float(np.mean(np.logaddexp(0.0, xc[m]) - t[m] * xc[m]))
        out[STRATA_NAMES[s] if n_strata == N_STRATA else str(s)] = {"rows": int(m.sum()), "bce_before": bef, "bce_after": aft, "improvement": bef - aft}
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# within-token permutation of the stratum codes (prereg s7 C-NULL)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def token_groups(tokens: np.ndarray) -> list:
    """Sets grouped by token, tokens in sorted order, set indices ascending inside a token."""
    _, inv = np.unique(np.asarray(tokens).astype(str), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    cuts = np.nonzero(np.diff(inv[order]))[0] + 1
    return [g for g in np.split(order, cuts)]


def permute_codes_within_token(code: np.ndarray, groups: list, seed: int) -> np.ndarray:
    """prereg s7: for each token (sorted order) pool the codes of all its sets in row-major order and permute them with one rng; a single-set token is permuted within the set."""
    out = np.array(code, copy=True)
    rng = np.random.default_rng(seed)
    for idx in groups:
        pooled = code[idx].reshape(-1)
        perm = rng.permutation(pooled.size)
        out[idx] = pooled[perm].reshape(len(idx), code.shape[1])
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the registered decision functions (pure; `branch_table_failures` is the table every mutation must break)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def cond_i(d_nc_ci: dict) -> bool:
    """prereg s6 (i): the UPPER bound of dNC (calibrated composite minus P2 NC-unsafe pick rate) is strictly below +0.001.  NaN -> False."""
    return bool(d_nc_ci["hi"] < NC_UB)


def cond_ii(d_pdms_ci: dict) -> bool:
    """prereg s6 (ii): the LOWER bound of dPDMS x100 (calibrated composite vs P2) is >= 0.  NaN -> False."""
    return bool(d_pdms_ci["lo"] >= PDMS_LB)


def seed_ok(d_nc_ci: dict, d_pdms_ci: dict) -> bool:
    return cond_i(d_nc_ci) and cond_ii(d_pdms_ci)


def recovers(per_seed: list) -> bool:
    """prereg s6: RECOVERS iff BOTH seeds meet (i) AND (ii).  per_seed = [(d_nc_ci, d_pdms_ci), ...]; an empty list is not a recovery."""
    return bool(len(per_seed) > 0 and all(seed_ok(a, b) for a, b in per_seed))


def cond_i_only(per_seed: list) -> bool:
    """prereg s7 descriptive companion: condition (i) alone holds on every seed."""
    return bool(len(per_seed) > 0 and all(cond_i(a) for a, _ in per_seed))


def null_void(fires: list) -> bool:
    """prereg s7: VOID iff the null fires (reads RECOVERS) on >= 2 of its replicates."""
    return bool(sum(1 for f in fires if f) >= NULL_VOID_FIRES)


def final_label(recovers_flag: bool, void: bool) -> str:
    if void:
        return "VOID"
    return "RECOVERS" if recovers_flag else "NO-RECOVER"


NEXT_ARM = {
    "RECOVERS": "a 0-train pick-rule registration for the dev gate (it reads SEEN scenes, D-L1-G0-1)",
    "NO-RECOVER": "NC calibration closes; the line goes to PRIVGEOM-SAFE (already queued) and the trunk",
    "VOID": "no next arm from this read; the null's firing is reported and the instrument is repaired first",
}


def next_arm(label: str) -> str:
    return NEXT_ARM[label]


def branch_table_failures(mod) -> list:
    """Every decision function on a synthetic branch table.  `mod` is this module or a MUTATED copy; returns the failing cases (empty = every branch right)."""
    bad = []

    def chk(name, got, want):
        if got != want:
            bad.append(f"{name}: got {got!r}, want {want!r}")
    nc = lambda hi: {"hi": hi, "lo": -1.0, "point": 0.0}                                              # noqa: E731
    pd = lambda lo: {"lo": lo, "hi": 5.0, "point": 1.0}                                               # noqa: E731
    nan = float("nan")
    chk("cond_i below", mod.cond_i(nc(0.0009)), True)
    chk("cond_i at bound is not <", mod.cond_i(nc(0.001)), False)
    chk("cond_i above", mod.cond_i(nc(0.0011)), False)
    chk("cond_i negative", mod.cond_i(nc(-0.01)), True)
    chk("cond_i nan", mod.cond_i(nc(nan)), False)
    chk("cond_ii above", mod.cond_ii(pd(0.3)), True)
    chk("cond_ii at bound is >=", mod.cond_ii(pd(0.0)), True)
    chk("cond_ii below", mod.cond_ii(pd(-0.001)), False)
    chk("cond_ii nan", mod.cond_ii(pd(nan)), False)
    chk("seed_ok both", mod.seed_ok(nc(0.0), pd(0.1)), True)
    chk("seed_ok only i", mod.seed_ok(nc(0.0), pd(-0.1)), False)
    chk("seed_ok only ii", mod.seed_ok(nc(0.01), pd(0.1)), False)
    chk("seed_ok neither", mod.seed_ok(nc(0.01), pd(-0.1)), False)
    good, bad_i, bad_ii = (nc(0.0), pd(0.1)), (nc(0.01), pd(0.1)), (nc(0.0), pd(-0.1))
    chk("recovers both seeds", mod.recovers([good, good]), True)
    chk("recovers needs seed 0", mod.recovers([bad_i, good]), False)
    chk("recovers needs seed 1", mod.recovers([good, bad_ii]), False)
    chk("recovers one seed only is not enough (i)", mod.recovers([good, bad_i]), False)
    chk("recovers empty", mod.recovers([]), False)
    chk("recovers nan on a seed", mod.recovers([good, (nc(nan), pd(0.1))]), False)
    chk("cond_i_only both seeds", mod.cond_i_only([good, bad_ii]), True)
    chk("cond_i_only one seed fails", mod.cond_i_only([good, bad_i]), False)
    chk("cond_i_only empty", mod.cond_i_only([]), False)
    chk("void 2 of 5", mod.null_void([True, True, False, False, False]), True)
    chk("void 1 of 5", mod.null_void([True, False, False, False, False]), False)
    chk("void 5 of 5", mod.null_void([True] * 5), True)
    chk("void 0 of 5", mod.null_void([False] * 5), False)
    chk("label void wins", mod.final_label(True, True), "VOID")
    chk("label void wins over no-recover", mod.final_label(False, True), "VOID")
    chk("label recovers", mod.final_label(True, False), "RECOVERS")
    chk("label no-recover", mod.final_label(False, False), "NO-RECOVER")
    chk("next RECOVERS", mod.next_arm("RECOVERS").split(" (")[0], "a 0-train pick-rule registration for the dev gate")
    chk("next NO-RECOVER", mod.next_arm("NO-RECOVER").split(";")[0], "NC calibration closes")
    chk("next VOID", mod.next_arm("VOID").split(";")[0], "no next arm from this read")
    return bad


def reach_table_failures(mod) -> list:
    """C-REACH / C-FOLD on a PLANTED synthetic bank (prereg s7): `mod` is this module or a MUTATED copy; returns the failing checks (empty = the calibration code path reaches the pick)."""
    bad = []

    def chk(name, ok):
        if not ok:
            bad.append(name)
    rng = np.random.default_rng(20261019)
    N = 300
    code = np.zeros((N, 64), np.uint8)
    code[0:100] = rng.integers(1, 4, size=(100, 64))                 # sets 0..99: Q1..Q3 only (no Q4, no S)
    code[100:200] = rng.integers(1, 5, size=(100, 64))               # sets 100..199: Q1..Q4
    code[200:300] = 0                                                  # sets 200..299: S sets (every proposal S)
    fold = (np.arange(N) % N_FOLDS).astype(np.int64)
    z6 = rng.normal(0.0, 2.0, size=(N, 64, 6)).astype(np.float32)
    x = z6[..., NC]
    # (1) the exact formula, per fold and stratum, with a planted table whose every entry is distinct
    pt = np.zeros((N_FOLDS, N_STRATA, 2))
    for k in range(N_FOLDS):
        for s in range(N_STRATA):
            pt[k, s] = (1.0 + 0.1 * k + 0.01 * s, -0.5 * k + 0.1 * s)
    got = mod.apply_calibration(x, code, fold, pt)
    want = np.empty((N, 64))
    for i in range(N):
        for j in range(64):
            k, s = int(fold[i]), int(code[i, j])
            want[i, j] = pt[k, s, 0] * float(x[i, j]) + pt[k, s, 1]
    chk("apply_calibration equals a_{fold,stratum} x + b_{fold,stratum} on every proposal", np.array_equal(got, want))
    # (2) identity is bit-exact
    ident = mod.identity_params()
    chk("identity params reproduce the logit bit for bit", np.array_equal(mod.apply_calibration(x, code, fold, ident), x.astype(np.float64)))
    # (3) a planted shift in ONE stratum moves the logit exactly there, the picks on some sets and not all, and no set without that stratum
    for target, free_sets in ((4, np.r_[0:100, 200:300]), (0, np.r_[0:200])):
        p = mod.identity_params()
        p[:, target, 1] = -3.0
        zc = mod.apply_calibration(x, code, fold, p)
        chk(f"shift in stratum {target} changes the logit exactly on that stratum", bool(((zc != x.astype(np.float64)) == (code == target)).all()))
        zE0 = mod.composite(z6)
        zE1 = mod.composite(z6, nc=zc)
        chk("composite writes only the NC column", bool(np.array_equal(zE1[..., 1:], zE0[..., 1:])) and not np.array_equal(zE1[..., 0], zE0[..., 0]))
        agg0, agg1 = _rule_pick(zE0), _rule_pick(zE1)
        moved = agg0 != agg1
        chk(f"shift in stratum {target} moves the pick on >= 1 set and on fewer than all", 0 < int(moved.sum()) < N)
        chk(f"shift in stratum {target} leaves every set without that stratum unchanged", not bool(moved[free_sets].any()))
    # (4) C-FOLD: fold k's fit does not read fold k's labels
    M = 1500
    cd = rng.integers(0, 5, size=(M, 64)).astype(np.uint8)
    fd = (np.arange(M) % N_FOLDS).astype(np.int64)
    xx = rng.normal(0.0, 2.0, size=(M, 64))
    tt = (rng.random((M, 64)) < 1.0 / (1.0 + np.exp(-(0.7 * xx - 0.3)))).astype(np.float64)
    pa, _ = mod.fit_calibration(xx, tt, cd, fd)
    flip = tt.copy()
    flip[fd == 2] = 1.0 - flip[fd == 2]
    pb, _ = mod.fit_calibration(xx, flip, cd, fd)
    chk("flipping fold 2's labels leaves fold 2's parameters bit-identical", bool(np.array_equal(pa[2], pb[2])))
    chk("flipping fold 2's labels changes the other folds' parameters", all(not np.array_equal(pa[k], pb[k]) for k in (0, 1, 3, 4)))
    return bad


def _rule_pick(zE: np.ndarray) -> np.ndarray:
    """navsim_v1 aggregate (planner.py:558-561, float64, first argmax) for the table checks; the production path uses the pinned pickrule_lib.rule_pick (a test asserts they agree)."""
    p = 1.0 / (1.0 + np.exp(-zE))
    agg = p[..., 0] * p[..., 1] * (p[..., 2] * 5.0 + p[..., 3] * 5.0 + p[..., 4] * 2.0) / 12.0
    return np.argmax(agg, axis=1)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# quantities, the arms, the null
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def calibrated_nc(z6: np.ndarray, lab: np.ndarray, code: np.ndarray, fold: np.ndarray) -> tuple:
    """The OOF per-stratum Platt NC logit (float64 [N,64]) for the given stratum codes -> (nc_cal, params, diag)."""
    x = np.asarray(z6[..., NC], np.float64)
    params, diag = fit_calibration(x, lab[..., NC], code, fold)
    return apply_calibration(x, code, fold, params), params, diag


def arm_decision(P, zE: np.ndarray, lab: np.ndarray, pdms: np.ndarray, pk_p: np.ndarray, S) -> dict:
    """The lean decision pair for one arm and one seed against P2's pick: dNC and dPDMS x100 (paired, same draws)."""
    pk = P.rule_pick(zE)
    f, f0 = P.pick_flags(lab, pdms, pk), P.pick_flags(lab, pdms, pk_p)
    d_nc = S.rate_diff(f["nc0"], f0["nc0"])
    d_pd = P.scale_ci(S.mean(f["pd"] - f0["pd"]), 100.0)
    return {"d_nc": d_nc, "d_pdms_x100": d_pd, "cond_i": cond_i(d_nc), "cond_ii": cond_ii(d_pd), "n_changed_vs_p2": int((pk != pk_p).sum())}


def decide(P, z6: np.ndarray, lab: np.ndarray, eps: dict, nc_cal: np.ndarray, pdms: np.ndarray, pk_p: np.ndarray, S) -> dict:
    """RECOVERS for the calibrated composite on every seed in `eps` ({seed: OOF EP column}); the shared path of the real run and of the null replicates."""
    per = {}
    for sd, ep in sorted(eps.items()):
        per[sd] = arm_decision(P, composite(z6, ep=ep, nc=nc_cal), lab, pdms, pk_p, S)
    pairs = [(per[sd]["d_nc"], per[sd]["d_pdms_x100"]) for sd in sorted(per)]
    return {"per_seed": per, "recovers": recovers(pairs), "cond_i_only": cond_i_only(pairs)}


def run_null(P, z6, lab, eps, code, fold, groups, pdms, pk_p, S, reps: int = NULL_REPS) -> dict:
    """prereg s7 C-NULL: the stratum codes permuted within token, the fit and the application on the permuted codes, the full decision recomputed on both seeds.  A replicate FIRES iff it reads RECOVERS."""
    rows, fires, ci_only = [], [], []
    for rep in range(reps):
        seed = NULL_SEED_BASE + NULL_SALT * rep
        cp = permute_codes_within_token(code, groups, seed)
        ncp, params, diag = calibrated_nc(z6, lab, cp, fold)
        d = decide(P, z6, lab, eps, ncp, pdms, pk_p, S)
        fires.append(bool(d["recovers"]))
        ci_only.append(bool(d["cond_i_only"]))
        rows.append({"rep": rep, "perm_seed": seed, "recovers": bool(d["recovers"]), "cond_i_only_both_seeds": bool(d["cond_i_only"]), "n_fit_failed": int(sum(1 for r in diag if not r["ok"])),
                     "params_mean_a_by_stratum": [float(np.mean(params[:, s, 0])) for s in range(N_STRATA)], "params_mean_b_by_stratum": [float(np.mean(params[:, s, 1])) for s in range(N_STRATA)],
                     "per_seed": {str(sd): {"d_nc": v["d_nc"], "d_pdms_x100": v["d_pdms_x100"], "cond_i": v["cond_i"], "cond_ii": v["cond_ii"]} for sd, v in d["per_seed"].items()}})
    return {"replicates": rows, "fires": int(sum(fires)), "of": reps, "cond_i_only_fires": int(sum(ci_only)), "void": null_void(fires)}


def reliability_at_picks(z_raw_nc: np.ndarray, z_cal_nc: np.ndarray, lab_nc: np.ndarray, pk: np.ndarray, P) -> dict:
    """prereg s8: at the picks, the mean raw NC sigmoid, the mean calibrated NC sigmoid and the empirical mean NC label (a probability of safe: 0 / 0.5 / 1)."""
    return {"n": int(len(pk)), "mean_raw_nc_sigmoid": float(P.at(P.sigmoid64(z_raw_nc), pk).mean()), "mean_calibrated_nc_sigmoid": float(P.at(P.sigmoid64(z_cal_nc), pk).mean()),
            "mean_nc_label": float(P.at(np.asarray(lab_nc, np.float64), pk).mean())}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the gate (controls that must hold before any decision quantity is read)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def strata_join_checks(arr: dict, speeds: dict, strata_json: dict) -> dict:
    """C-JOIN-STRATA (prereg s3), label-free: key_log vs the arrays' log names, pick vs top8[:, 0], the code recomputed from r and the cut points, rows per stratum."""
    names = np.asarray(arr["log_names"]).astype(str)
    key_ok = bool(np.array_equal(np.asarray(speeds["key_log"]).astype(str), names[arr["log_idx"]]))
    pick_ok = bool(np.array_equal(np.asarray(speeds["pick"], np.int64), arr["top8"][:, 0].astype(np.int64)))
    r, code = speeds["r"], speeds["code"]
    cuts = strata_json["cut_points"]
    c25, c50, c75 = cuts["c25"], cuts["c50"], cuts["c75"]
    v_b = speeds["v_b"]
    s_set = v_b < 0.5
    re = np.zeros(code.shape, np.uint8)
    ns = ~s_set
    rn = r[ns]
    re[ns] = (1 + (rn > c25).astype(np.int64) + (rn > c50).astype(np.int64) + (rn > c75).astype(np.int64)).astype(np.uint8)
    code_ok = bool(np.array_equal(re, code))
    counts = {n: int((code == i).sum()) for i, n in enumerate(STRATA_NAMES)}
    counts_ok = counts == {k: int(v) for k, v in strata_json["rows_per_stratum"].items()}
    s_ok = bool((code[s_set] == 0).all() and (code[ns] > 0).all())
    return {"pass": bool(key_ok and pick_ok and code_ok and counts_ok and s_ok and code.shape == arr["z6"].shape[:2]),
            "detail": {"key_log_equals_arrays_log_names": key_ok, "pick_equals_top8_0": pick_ok, "code_recomputed_from_r_and_cuts_equals_stored": code_ok, "rows_per_stratum": counts,
                       "rows_per_stratum_equal_strata_json": counts_ok, "S_sets_are_all_S_and_other_sets_have_no_S": s_ok, "n_S_sets": int(s_set.sum())}}


LANDED_FIELDS = ("nc_unsafe_rate", "dac_unsafe_rate", "pdms_x100", "d_nc_unsafe_rate", "d_dac_unsafe_rate", "d_pdms_x100")


def compare_landed(mine: dict, landed: dict, tol: float = 1e-12) -> dict:
    """Max absolute difference of the registered fields (point, lo, hi) between my U_s sweep row and the landed audit's `sweep.none` row."""
    worst, per = 0.0, {}
    for f in LANDED_FIELDS:
        for k in ("point", "lo", "hi"):
            dlt = abs(float(mine[f][k]) - float(landed[f][k]))
            per[f"{f}.{k}"] = dlt
            worst = max(worst, dlt)
    return {"max_abs_diff": worst, "tolerance": tol, "pass": bool(worst <= tol), "n_picks_changed_vs_p2": [int(mine["n_changed_vs_p2"]), int(landed["n_changed_vs_p2"])]}
