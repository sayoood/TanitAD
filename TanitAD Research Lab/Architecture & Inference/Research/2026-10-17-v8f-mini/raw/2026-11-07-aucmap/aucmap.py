#!/usr/bin/env python3
"""aucmap.py -- PREREG_AUCMAP.md (frozen, see PREREG_AUCMAP.sha256): posterior-GATED banked level-floor levers on the T1-kin reindexed eval-141 windows.
CEILING-TYPE where the posterior is synthetic (built from the true label) or the level is logged; no driving claim.  CPU only, zero training.
    C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I aucmap.py --controls | --run [--workers N]"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from statistics import NormalDist

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
MC = "E:/Projects/TanitAD-artifacts/dagland/motion_c"
RAW = f"{HERE}/raw"
P1P = f"{MC}/theory/wsp_read/P1P_PERWINDOW.npz"
LF_READ_PY = f"{MC}/lf_read/code/lf_read.py"
BASE_DIR = "ri_read"
BASE = "RIC0"
ADMISSIBLE = {"LF050": ("lf_read", "RILF050"), "LF075": ("lf_read", "RILF075"), "LF100": ("lf5_read", "RILF100"), "LFB05": ("lf6_read", "RILFB05"), "LFB10": ("lf6_read", "RILFB10")}
CEILING = {"LFO075": ("lfo_read", "RILFO075"), "LFO100": ("lf5_read", "RILFO100"), "LFOB05": ("lf6_read", "RILFOB05"), "LFOB10": ("lf6_read", "RILFOB10")}
LEVERS = {**ADMISSIBLE, **CEILING}
AUC_LEVELS = (0.50, 0.68, 0.785, 0.85, 0.90, 0.95, 0.99, 1.0)
K_DRAWS = 20
Q_GRID = tuple(round(0.05 * i, 2) for i in range(21))
SEEDS = (0, 1)
NB, BSEED = 2000, 0
M1 = "M1 crawl share"
BASE_SEED = 20261107
SHARE_BAR = 0.5
GO_DV = 1.0

GD = rr = SR = None


def init():
    global GD, rr, SR
    spec = importlib.util.spec_from_file_location("lf_read_for_aucmap", LF_READ_PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.init()
    GD, rr, SR = m.GD, m.rr, m.SR
    return m


# ------------------------------------------------------------------------------------------------ data
def _ld(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def arrays_path(d, name, s):
    return f"{MC}/{d}/raw/arrays/{name}_s{s}.npz"


class Rig:
    """Per-seed stacks: X[lever][s] = (19, N) arm values, R[s] = base RIC0, P[s] = P10; identical windows asserted; fixed ok masks (NaN pattern identical across arms)."""

    def __init__(self, levers=None, n_boot=NB):
        self.n_boot = n_boot
        self.R, self.P, self.X = {}, {}, {}
        self.meta = {}
        for s in SEEDS:
            b = _ld(arrays_path(BASE_DIR, BASE, s))
            assert tuple(str(q) for q in b["labels"]) == tuple(GD.LABELS)
            self.R[s] = np.stack(rr.Aof(b))
            self.P[s] = np.stack(rr.Bof(b))
            if s == 0:
                self.eid, self.v0, self.g = b["eid"], b["v0"], b["g"]
                self.ci, self.ws = b["clip_index"], b["ws"]
            else:
                for k, ref in (("eid", self.eid), ("v0", self.v0), ("g", self.g), ("clip_index", self.ci), ("ws", self.ws)):
                    assert np.array_equal(b[k], ref, equal_nan=True), f"seed 1 base: {k} differs from seed 0's"
        for name in (levers or LEVERS):
            d, pref = LEVERS[name]
            self.X[name] = {}
            for s in SEEDS:
                z = _ld(arrays_path(d, pref, s))
                for k, ref in (("eid", self.eid), ("v0", self.v0), ("g", self.g), ("clip_index", self.ci), ("ws", self.ws)):
                    assert np.array_equal(z[k], ref, equal_nan=True), f"{name} s{s}: {k} differs from the base's"
                a, bb = np.stack(rr.Aof(z)), np.stack(rr.Bof(z))
                assert np.array_equal(np.isnan(bb), np.isnan(self.P[s])), f"{name} s{s}: P10 NaN pattern differs"
                keep = [i for i, l in enumerate(GD.LABELS) if not l.startswith("S-POS")]      # the S-POS b-arrays differ between arms (STRATEGIC PAUSED; no X - P10 contrast on S-POS is read)
                assert np.array_equal(bb[keep], self.P[s][keep], equal_nan=True), f"{name} s{s}: P10 arrays differ on a non-STRATEGIC row"
                self.X[name][s] = a
        self.ep = np.asarray([str(x) for x in self.eid])
        self._prep()

    def _prep(self):
        """Integer episode ids (sorted-string order = np.unique order, as gapdecomp.decompose) and a draw-matrix cache keyed by the set of episodes present in a row's pool."""
        self.uniq_all, self.eidx = np.unique(self.ep, return_inverse=True)
        self._mc = {}

    def _draws(self, present):
        key = present.tobytes()
        if key not in self._mc:
            self._mc[key] = GD.cached_draws(self.uniq_all[present], self.n_boot, BSEED)
        return self._mc[key]

    def row_stat(self, lever, s, i, on, vs="R", dcache=None, boot=True):
        """Gated arm (X where on, R elsewhere) minus R (vs='R') or minus P10 (vs='P') on row i: gapdecomp.decompose's delta / episode-cluster bootstrap (B n_boot, seed 0), pair mask and pool exactly as decompose."""
        Rv, Pv = self.R[s][i], self.P[s][i]
        g = np.where(on > 0, self.X[lever][s][i], Rv)
        ref = Rv if vs == "R" else Pv
        ok = np.isfinite(g) & np.isfinite(ref)
        masked = GD.POOL[GD.LABELS[i]] == "masked"
        nE = len(self.uniq_all)
        present = (np.bincount(self.eidx[ok], minlength=nE) > 0) if masked else np.ones(nE, bool)
        sub = np.cumsum(present) - 1
        e_ok = sub[self.eidx[ok]]
        d = (g - ref)[ok]
        S = np.bincount(e_ok, weights=d, minlength=int(present.sum()))
        N = np.bincount(e_ok, minlength=int(present.sum())).astype(float)
        delta = float(S.sum() / N.sum())
        if not boot:
            return {"delta": delta}
        M = self._draws(present)
        with np.errstate(divide="ignore", invalid="ignore"):
            d_b = (M @ S) / (M @ N)
        lo, hi, _ = GD._pct(d_b)
        return {"delta": delta, "lo": lo, "hi": hi, "separated": bool(lo > 0 or hi < 0)}

    def point(self, lever, s, i, on, dcache=None):
        return self.row_stat(lever, s, i, on, "R", boot=False)["delta"]

    def ex(self, s, i=18):
        return self.row_stat(lever=next(iter(self.X)), s=s, i=i, on=np.zeros(len(self.ep)), vs="P", boot=False)["delta"]


# ------------------------------------------------------------------------------------------------ labels, scores, gates
def labels_from(v0, g):
    v6 = np.linalg.norm(g[:, 29] - g[:, 28], axis=-1) / 0.2
    return {"GO": v6 >= v0 + GO_DV, "KEEP": v6 >= v0 - 1.0}


def auc_mw(score, y):
    """Mann-Whitney AUC of score against boolean y (no ties assumed; ties get average rank via argsort of a jittered-free rank)."""
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def make_score(y, auc, level_index, draw, permute=False):
    rng = np.random.default_rng([BASE_SEED, level_index, draw])
    eps = rng.standard_normal(len(y))
    yy = np.asarray(y, float)
    if permute:
        yy = yy[np.random.default_rng([BASE_SEED, 99, draw]).permutation(len(y))]
    if auc >= 1.0:
        return 1e6 * yy + eps
    return yy * (np.sqrt(2.0) * NormalDist().inv_cdf(auc)) + eps


def select_top(score, scope, q):
    """ON mask: the top ceil-free round(q * n_scope) scoring windows among ``scope``; windows outside scope are OFF."""
    idx = np.flatnonzero(scope)
    k = int(round(q * len(idx)))
    on = np.zeros(len(score), float)
    if k > 0:
        top = idx[np.argsort(-score[idx], kind="mergesort")[:k]]
        on[top] = 1.0
    return on


# ------------------------------------------------------------------------------------------------ corner
def active_nonrow(label):
    return label not in SR.CRAWL and SR.FAMILY[label] in ("LONGITUDINAL", "LATERAL", "TACTICAL")


def evaluate(rig, lever, on, full=False):
    """Corner predicates of one gate (both seeds).  Phase 1: point share_closed; bootstrap rows only when the point share could reach the bar (or ``full``).
    -> dict(share_mean, share_s, ade6_s, M1P, M1FULL, GUARD, GUARD_ade6sw, CORNER_P, CORNER_FULL, rows_xr, rows_xp_m1)."""
    dc = None
    ex = {s: rig.ex(s) for s in SEEDS}
    share_s = {s: ((-rig.point(lever, s, 18, on, dc) / ex[s]) if ex[s] > 0 else 0.0) for s in SEEDS}
    ade_s = {s: rig.point(lever, s, 0, on, dc) for s in SEEDS}
    out = {"share_s": share_s, "share_mean": float(np.mean(list(share_s.values()))), "ade6_s": ade_s, "on_frac": float(on.mean())}
    need = full or out["share_mean"] >= SHARE_BAR
    out.update({"M1P": False, "M1FULL": False, "GUARD": None, "GUARD_ade6sw": None, "CORNER_P": False, "CORNER_FULL": False})
    if not need:
        return out
    xr_m1 = {s: rig.row_stat(lever, s, 18, on, "R", dc) for s in SEEDS}
    cells = {f"s{s}": rr.m1_cell(xr_m1[s], {"delta": ex[s], "separated": True}) for s in SEEDS}
    m1p = bool(all(c["better"] for c in cells.values()) and rr.share_mean(cells) >= SHARE_BAR)
    xp_m1 = {s: rig.row_stat(lever, s, 18, on, "P", dc) for s in SEEDS}
    m1full = bool(m1p and all(not SR.worse(GD, M1, xp_m1[s]) for s in SEEDS))
    out.update({"M1P": m1p, "M1FULL": m1full, "rows_m1_xr": xr_m1, "rows_m1_xp": xp_m1})
    if m1p or full:
        guard, guard2, rows = True, True, {}
        for s in SEEDS:
            rows[s] = {lab: (xr_m1[s] if i == 18 else rig.row_stat(lever, s, i, on, "R", dc)) for i, lab in enumerate(GD.LABELS)}
            sw = sorted(l for l, r in rows[s].items() if active_nonrow(l) and SR.worse(GD, l, r))
            ok_ade = rows[s]["ADE6"]["delta"] <= SR.ADE_GUARD_M
            guard &= bool(ok_ade and not sw)
            guard2 &= bool(ok_ade and not sw and not SR.worse(GD, "ADE6", rows[s]["ADE6"]))
            rows[s]["_active_separated_worse"] = sw
        out.update({"GUARD": bool(guard), "GUARD_ade6sw": bool(guard2), "rows_xr": rows, "CORNER_P": bool(m1p and guard), "CORNER_FULL": bool(m1full and guard)})
    return out


# ------------------------------------------------------------------------------------------------ run one (lever, scope, label)
def run_synthetic(rig, lever, scope_name, label_name, scope, y, permuted=False, levels=AUC_LEVELS, k=K_DRAWS, qs=Q_GRID, log=print):
    res = {"lever": lever, "scope": scope_name, "label": label_name, "permuted": permuted, "levels": {}}
    t0 = time.time()
    for li, auc in enumerate(levels):
        rec = {"auc_nominal": auc, "realized_auc": [], "reach_P": [], "reach_FULL": [], "share_mean": [], "ade6": [], "corner_FULL_q": [], "corner_P_q": [], "M1FULL_q": [], "GUARD_q": [], "share_s": []}
        for dr in range(k):
            sc = make_score(y, auc, li, dr, permute=permuted)
            rec["realized_auc"].append(auc_mw(sc[scope], y[scope]))
            shm, ade, cf, cp, m1f, gd, shs = [], [], [], [], [], [], []
            for q in qs:
                e = evaluate(rig, lever, select_top(sc, scope, q))
                shm.append(e["share_mean"]); shs.append([e["share_s"][0], e["share_s"][1]]); ade.append([e["ade6_s"][0], e["ade6_s"][1]])
                cf.append(bool(e["CORNER_FULL"])); cp.append(bool(e["CORNER_P"])); m1f.append(bool(e["M1FULL"])); gd.append(e["GUARD"])
            rec["share_mean"].append(shm); rec["share_s"].append(shs); rec["ade6"].append(ade); rec["corner_FULL_q"].append(cf); rec["corner_P_q"].append(cp); rec["M1FULL_q"].append(m1f); rec["GUARD_q"].append(gd)
            rec["reach_FULL"].append(bool(any(cf))); rec["reach_P"].append(bool(any(cp)))
        rec["rate_FULL"] = float(np.mean(rec["reach_FULL"])); rec["rate_P"] = float(np.mean(rec["reach_P"]))
        res["levels"][str(auc)] = rec
        log(f"[{lever}|{scope_name}|{label_name}{'|perm' if permuted else ''}] AUC {auc}: realized {np.mean(rec['realized_auc']):.3f} reach FULL {rec['rate_FULL']:.2f} P {rec['rate_P']:.2f} ({time.time() - t0:.0f}s)")
    return res


def a_star(res, key="rate_FULL", thr=0.5):
    lv = sorted((float(k), v[key]) for k, v in res["levels"].items())
    reached = [a for a, r in lv if r >= thr]
    return reached[0] if reached else None


def interval(res, key="rate_FULL"):
    return [a_star(res, key, 0.05), a_star(res, key, 0.95)]


def anchor_curve(rig, lever, post, y_go):
    """Real-posterior anchor: scope = windows with a posterior; full evaluation on the whole q grid."""
    scope = np.isfinite(post)
    pf = np.where(scope, post, -np.inf)
    out = {"lever": lever, "n_scope": int(scope.sum()), "realized_auc": auc_mw(post[scope], y_go[scope]), "q": [], "curve": []}
    for q in Q_GRID:
        e = evaluate(rig, lever, select_top(pf, scope, q), full=True)
        out["q"].append(q)
        out["curve"].append({k: e[k] for k in ("share_s", "share_mean", "ade6_s", "M1P", "M1FULL", "GUARD", "GUARD_ade6sw", "CORNER_P", "CORNER_FULL")})
        if e.get("rows_xr"):
            out["curve"][-1]["active_separated_worse"] = {f"s{s}": e["rows_xr"][s]["_active_separated_worse"] for s in SEEDS}
            out["curve"][-1]["rows_xr"] = {f"s{s}": {l: {kk: r[kk] for kk in ("delta", "lo", "hi")} for l, r in e["rows_xr"][s].items() if not l.startswith("_")} for s in SEEDS}
    return out


def load_post(rig):
    z = np.load(P1P, allow_pickle=True)
    key = {(int(c), int(w)): j for j, (c, w) in enumerate(zip(rig.ci, rig.ws))}
    post = np.full(len(rig.ci), np.nan)
    gop = np.zeros(len(rig.ci), bool)
    v0p = np.full(len(rig.ci), np.nan)
    for j in range(len(z["ci"])):
        k = key[(int(z["ci"][j]), int(z["t0"][j]))]
        post[k] = z["post_primary"][j]; gop[k] = bool(z["go"][j]); v0p[k] = z["v0"][j]
    return post, gop, v0p


if __name__ == "__main__":
    print(__doc__)
