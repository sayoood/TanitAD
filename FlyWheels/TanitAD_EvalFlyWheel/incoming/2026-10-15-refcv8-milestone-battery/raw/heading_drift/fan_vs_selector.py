"""fan_vs_selector.py -- does refcv8's tight-turn under-steering come from its CANDIDATE PLANS (anchor bank + refinement)
or from WHICH CANDIDATE IT SELECTS?   CPU only, existing per-window dumps (S2 panels), no GPU, no git.

    OMP_NUM_THREADS=4 python fan_vs_selector.py --out-dir <dir> --tag FAN_VS_SELECTOR

Decomposition (exact identity, asserted numerically at the point estimate AND on every bootstrap draw):

    gap  = slope(r8 os) - slope(r7 os)
         = [slope(r8 oracle_sel) - slope(r7 oracle_sel)]                                  <- CANDIDATE (anchor coverage + refinement)
         + [(slope(r8 os) - slope(r8 oracle_sel)) - (slope(r7 os) - slope(r7 oracle_sel))] <- SELECTION

`oracle_sel` = the refinement out["anchor_traj"][a_star] of the anchor nearest the GT in position (a T0 ceiling probing anchor coverage
+ refinement; NOT the best of the whole 149-candidate fan).  slope = through-origin slope of the turn-aligned signed heading error on the
turn-aligned GT chord heading (hd.shrinkage definition; -0.10 = the plan turns 10 % less than GT), 2-seed mean of the signed per-segment error,
only segments valid on EVERY arm in the contrast.  Bootstrap = ONE shared episode-cluster resample matrix (139 episodes, B 2000, seed 0+29, the
same W that hd.shrinkage builds), so every term and every difference is paired.  Estimator question answered: 'another draw of EPISODES, models
FIXED' -- NOT another training run (H-ESTIM-SEED-1), NOT another inference run (that is addressed only by per-seed rows).

Rule 6 control (runs BEFORE anything else, asserts): the banked EMA-15k sharp-turn last-segment slope -0.14093672 and refcv7 -0.09212979 must
reproduce through the ORIGINAL heading_drift.shrinkage on the original arm set.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import heading_drift as hd                                              # noqa: E402  (imported, never edited)

TREES = {                                                               # label -> root  (dumps panel_s{0,1}/ + decisions/)
    "EMA-15k": "E:/refcv8_eval_kit/battery/step15000ema_unc",
    "EMA-10.5k": "E:/refcv8_eval_kit/battery/step10500ema_unc",
    "EMA-5k": "C:/Users/Admin/refcv8_eval_archive/battery/step5000ema_unc",
    "RAW-5k": "C:/Users/Admin/refcv8_eval_archive/battery/step5000_unc",
}
R7_ROOT = "D:/refcv7_eval_kit/battery/step50400_a6text"
CKPT = {"r8": "E:/refcv8_eval_kit/ckpt/ckpt_ema_15000.pt", "r7": "D:/refcv7_eval_kit/ckpt/ckpt_50400.pt"}
N_BANK = 117
SEEDS = (0, 1)
PANEL_KEYS = ("g", "v0", "os", "oracle_sel", "ws")
NAV_ARMS = ("os_navzero", "os_navflip", "os_navshuf")
DEC_KEYS = ("nav_cmd", "sel_idx", "sel_idx_base", "a_star", "sel_agrees_oracle")


# ------------------------------------------------------------------------------------------------ memory guard
class _MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_commit_gb() -> float:
    ms = _MS()
    ms.dwLength = ctypes.sizeof(_MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    return ms.ullAvailPageFile / 2 ** 30


def guard(tag: str) -> float:
    g = free_commit_gb()
    if g < 3.0:
        raise SystemExit(f"ABORT: free commit {g:.2f} GB < 3 GB at {tag}")
    return g


# ------------------------------------------------------------------------------------------------ loading
def load(root: str, seed: int, ref_label: str | None = None) -> dict:
    pd = hd.panel_dir(root, "S2", seed)
    man = json.load(open(os.path.join(pd, "manifest.json"), encoding="utf-8"))
    cid = {int(e["file_index"]): str(e["clip_id"]) for e in man["episodes"]}
    files = sorted(glob.glob(os.path.join(pd, "ep*.npz")))
    acc = {k: [] for k in PANEL_KEYS + NAV_ARMS + DEC_KEYS + ("eid", "ref")}
    keys = []
    for fi, f in enumerate(files):
        file_index = int(os.path.basename(f)[2:5])
        with np.load(f) as z:
            names = set(z.files)
            n = z["g"].shape[0]
            for k in PANEL_KEYS:
                acc[k].append(z[k])
            for k in NAV_ARMS:
                acc[k].append(z[k] if k in names else np.full(z["os"].shape, np.nan, np.float32))
            if ref_label is not None:
                acc["ref"].append(z[ref_label])
            wsf = z["ws"]
        acc["eid"].append(np.full(n, fi))
        with np.load(os.path.join(pd, "decisions", os.path.basename(f))) as d:
            if not np.array_equal(d["ws"], wsf):
                raise RuntimeError("decisions ws != panel ws in " + f)
            for k in DEC_KEYS:
                acc[k].append(d[k].astype(np.float64))
        keys += [(cid[file_index], int(w)) for w in wsf]
    out = {k: (np.concatenate(v) if v else None) for k, v in acc.items()}
    out["keys"] = keys
    out["n_files"] = len(files)
    out["model"] = man.get("model") or {}
    out["dt"] = float(man["grid"]["dt_s"])
    return out


def reorder(P: dict, idx: np.ndarray) -> dict:
    out = {}
    for k, v in P.items():
        out[k] = v[idx] if (isinstance(v, np.ndarray) and v.shape[:1] == (len(idx),)) else v
    return out


def align_to(ref: dict, P: dict) -> dict:
    pos = {k: i for i, k in enumerate(P["keys"])}
    if len(pos) != len(P["keys"]) or set(ref["keys"]) != set(pos):
        raise RuntimeError("window sets differ")
    idx = np.asarray([pos[k] for k in ref["keys"]])
    Q = reorder(P, idx)
    d = float(np.abs(ref["g"].astype(np.float64) - Q["g"].astype(np.float64)).max())
    if d != 0.0:
        raise RuntimeError(f"GT differs ({d})")
    Q["_identity_order"] = bool(np.array_equal(idx, np.arange(len(idx))))
    return Q


# ------------------------------------------------------------------------------------------------ bootstrap primitives
def build_W(eid, n_boot: int):
    ue, inv = np.unique(np.asarray([str(e) for e in eid]), return_inverse=True)
    E = len(ue)
    rng = np.random.default_rng(hd.SEED + 29)                          # IDENTICAL to hd.shrinkage -> identical resamples
    pick = rng.integers(0, E, size=(n_boot, E))
    W = np.stack([np.bincount(pick[b], minlength=E) for b in range(n_boot)]).astype(float)
    return inv, E, W


def rstat(num, den, W, keep_bs=False):
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    pt = float(num.sum() / den.sum()) if den.sum() != 0 else float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (W @ num) / (W @ den)
    fin = np.isfinite(bs)
    if fin.sum() < 50:
        r = {"est": pt, "lo": None, "hi": None, "separated": None, "n_boot_finite": int(fin.sum())}
    else:
        lo, hi = np.percentile(bs[fin], [2.5, 97.5])
        r = {"est": pt, "lo": float(lo), "hi": float(hi), "separated": bool(lo > 0 or hi < 0), "n_boot_finite": int(fin.sum())}
    return (r, bs) if keep_bs else r


def bs_diff(est_a, bs_a, est_b, bs_b):
    ok = np.isfinite(bs_a) & np.isfinite(bs_b)
    if ok.sum() < 50:
        return {"est": est_a - est_b, "lo": None, "hi": None, "separated": None}
    lo, hi = np.percentile((bs_a - bs_b)[ok], [2.5, 97.5])
    return {"est": float(est_a - est_b), "lo": float(lo), "hi": float(hi), "separated": bool(lo > 0 or hi < 0)}


def fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.{nd}f}"


def cellmd(r, nd=4):
    if r is None:
        return "n/a"
    s = fmt(r.get("est"), nd)
    if r.get("lo") is not None:
        s += f" [{fmt(r['lo'], nd)}, {fmt(r['hi'], nd)}]"
        s += " SEP" if r.get("separated") else ""
    return s


class Ctx:
    """one (window-group, segment-set) for the through-origin slope; a fixed 'valid' mask (segments finite on EVERY arm of the contrast)."""

    def __init__(self, hg, sw, valid, gm, ks, inv, E):
        m = np.zeros_like(valid)
        m[:, ks] = True
        m &= valid & gm[:, None] & (sw != 0)[:, None]
        self.m, self.sw, self.inv, self.E = m, sw, inv, E
        self.h = hg * sw[:, None]
        self.n_seg, self.n_win = int(m.sum()), int(m.any(1).sum())
        self.Sxx = self._sum(self.h ** 2, m)
        self.n_ep = int((self.Sxx > 0).sum())
        self.mean_h = float(self.h[m].mean()) if self.n_seg else float("nan")

    def _sum(self, vals, mm):
        ii = np.broadcast_to(self.inv[:, None], mm.shape)[mm]
        return np.bincount(ii, weights=vals[mm], minlength=self.E)

    def rows(self, rowmask=None):
        return self.m if rowmask is None else (self.m & np.asarray(rowmask, bool)[:, None])

    def sxy(self, dh, rowmask=None):
        return self._sum((dh * self.sw[:, None]) * self.h, self.rows(rowmask))

    def sxx(self, rowmask=None):
        return self._sum(self.h ** 2, self.rows(rowmask))

    def nwin(self, rowmask=None):
        return int(self.rows(rowmask).any(1).sum())


COMBOS = {
    "slope r8 os": {"r8os": 1}, "slope r8 oracle_sel": {"r8orc": 1}, "slope r7 os": {"r7os": 1}, "slope r7 oracle_sel": {"r7orc": 1},
    "GAP (r8 os - r7 os)": {"r8os": 1, "r7os": -1},
    "CANDIDATE (r8 orc - r7 orc)": {"r8orc": 1, "r7orc": -1},
    "SELECTION ((r8 os-orc) - (r7 os-orc))": {"r8os": 1, "r8orc": -1, "r7os": -1, "r7orc": 1},
    "selection cost r8 (os - orc)": {"r8os": 1, "r8orc": -1},
    "selection cost r7 (os - orc)": {"r7os": 1, "r7orc": -1},
}


def decomposition(ctx: Ctx, dhs: dict, W) -> dict:
    S = {k: ctx.sxy(v) for k, v in dhs.items()}
    res, bs = {}, {}
    for name, c in COMBOS.items():
        num = sum(v * S[k] for k, v in c.items())
        res[name], bs[name] = rstat(num, ctx.Sxx, W, True)
    g, c_, s_ = "GAP (r8 os - r7 os)", "CANDIDATE (r8 orc - r7 orc)", "SELECTION ((r8 os-orc) - (r7 os-orc))"
    id_pt = res[g]["est"] - res[c_]["est"] - res[s_]["est"]
    id_bs = float(np.max(np.abs(bs[g] - bs[c_] - bs[s_])))
    assert abs(id_pt) < 1e-12 and id_bs < 1e-12, f"decomposition identity broken: point {id_pt}, boot {id_bs}"
    out = {"n_seg": ctx.n_seg, "n_win": ctx.n_win, "n_ep": ctx.n_ep, "mean_gt_turn_aligned_heading_deg": ctx.mean_h,
           "terms": res, "identity_max_abs_err_point": abs(id_pt), "identity_max_abs_err_all_boot_draws": id_bs}
    gb = bs[g]
    for nm, key in (("share_candidate", c_), ("share_selection", s_)):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = bs[key] / gb
        ok = np.isfinite(r)
        lo, hi = np.percentile(r[ok], [2.5, 97.5])
        out[nm] = {"est": res[key]["est"] / res[g]["est"], "lo": float(lo), "hi": float(hi),
                   "frac_boot_gap_same_sign_as_point": float((np.sign(gb) == np.sign(res[g]["est"])).mean()),
                   "note": "ratio of two correlated bootstrap terms; read only when GAP is separated from 0"}
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="FAN_VS_SELECTOR")
    ap.add_argument("--n-boot", type=int, default=hd.N_BOOT)
    ap.add_argument("--taniteval-root", default="E:/Projects/TanitAD/taniteval")
    ap.add_argument("--stack-root", default="E:/Projects/TanitAD/stack")
    a = ap.parse_args()
    t0 = time.time()
    g0 = guard("start")
    torch, ci, ff = hd.setup(a.taniteval_root, a.stack_root)
    print(f"[{time.time() - t0:6.1f}s] torch imported; free commit {guard('after torch import'):.2f} GB", flush=True)

    # ---------------- load everything (arrays are tiny: ~1-2 MB per tree x seed)
    P = {}                                                              # (label, seed) -> aligned dict
    ref = None
    for lab, root in TREES.items():
        for s in SEEDS:
            Q = load(root, s, ref_label=f"b_refcv7_50400_s{s}")
            if ref is None:
                ref = Q                                                 # EMA-15k seed 0 defines row order and the episode ids (= banked)
            P[(lab, s)] = Q if (lab, s) == ("EMA-15k", 0) else align_to(ref, Q)
        print(f"[{time.time() - t0:6.1f}s] loaded {lab}", flush=True)
    for s in SEEDS:
        P[("r7", s)] = align_to(ref, load(R7_ROOT, s))
    print(f"[{time.time() - t0:6.1f}s] loaded refcv7 tree; free commit {guard('after loads'):.2f} GB", flush=True)
    G, v0, eid = ref["g"], ref["v0"], ref["eid"]
    dt = ref["dt"]
    N = len(eid)
    inv, E, W = build_W(eid, a.n_boot)
    B = hd.gt_bins(ff, torch, G, v0, ref["nav_cmd"], dt)
    Gg = ff._seq_geometry(torch.as_tensor(G).float(), dt)
    hg = Gg["heading"].numpy().astype(np.float64) * 180.0 / math.pi
    gvalid = Gg["valid"].numpy()
    sw = np.where((B["turn"] >= 1) & (B["turn"] <= 3), np.sign(B["gt_yaw_signed"]), 0.0)
    sharp, curved = B["turn"] == 3, (B["turn"] >= 1) & (B["turn"] <= 3)
    K = gvalid.shape[1]
    R = {"title": "refcv8 tight-turn under-steering: candidate plans (anchor bank + refinement) vs selection",
         "tier": "T1-style open loop on the S2 panel windows for `os` (own action); `oracle_sel` is a T0 ceiling (nearest-GT anchor, refined)",
         "estimator": f"paired episode-cluster bootstrap, ONE shared resample matrix (E={E} episodes, B={a.n_boot}, rng seed {hd.SEED + 29}); "
                      "answers 'another draw of EPISODES, models fixed' only -- NOT another training run (H-ESTIM-SEED-1), NOT another inference run",
         "evidence_class": "MEASURED (ours; per-window dumps listed under `inputs`)", "generated": time.strftime("%FT%T"),
         "n_windows": N, "n_episodes": E, "dt_s": dt, "instants_s": [0.5, 1.0, 1.5, 2.0], "inputs": {"r8": TREES, "r7": R7_ROOT, "ckpts": CKPT},
         "windows": {"sharp (GT |yaw|>=0.15)": int(sharp.sum()), "curved (>=0.02)": int(curved.sum())},
         "row_order_identical_across_trees": {f"{k[0]}|s{k[1]}": v.get("_identity_order", True) for k, v in P.items()}}
    dh = lambda plan: hd.seg_signed_heading(ff, torch, plan, G, dt)     # noqa: E731

    # ---------------- RULE 6: reproduction control (assert before anything else)
    pdh = {}
    for lab in ("EMA-15k", "EMA-10.5k", "EMA-5k"):
        pdh[lab] = 0.5 * (dh(P[(lab, 0)]["os"]) + dh(P[(lab, 1)]["os"]))
    pdh["refcv7-50400"] = 0.5 * (dh(P[("EMA-15k", 0)]["ref"]) + dh(P[("EMA-15k", 1)]["ref"]))
    pdh = {k: pdh[k] for k in ("EMA-15k", "refcv7-50400", "EMA-10.5k", "EMA-5k")}         # banked arm order / arm set
    shr = hd.shrinkage(ff, torch, pdh, G, dt, eid, B, {"sharp (>= 0.15)": sharp}, a.n_boot, "refcv7-50400")
    row = shr["sharp (>= 0.15) | last segment"]
    ctl = {"banked_artifact": "raw/heading_drift/HEADING_DRIFT_step15000ema.json -> S2.extras.curvature_shrinkage['sharp (>= 0.15) | last segment']",
           "banked": {"EMA-15k": -0.14093672306643532, "refcv7-50400": -0.09212979250717415, "EMA-15k_lo": -0.24445423362117089, "EMA-15k_hi": -0.06155907359084041},
           "reproduced": {"EMA-15k": row["slope"]["EMA-15k"]["slope"], "refcv7-50400": row["slope"]["refcv7-50400"]["slope"], "n_seg": row["n_seg"], "n_win": row["n_win"]}}
    ctl["max_abs_diff_slope"] = max(abs(ctl["reproduced"]["EMA-15k"] - ctl["banked"]["EMA-15k"]), abs(ctl["reproduced"]["refcv7-50400"] - ctl["banked"]["refcv7-50400"]))
    ctl["max_abs_diff_ci"] = max(abs(row["slope"]["EMA-15k"]["lo"] - ctl["banked"]["EMA-15k_lo"]), abs(row["slope"]["EMA-15k"]["hi"] - ctl["banked"]["EMA-15k_hi"]))
    ctl["PASS"] = bool(ctl["max_abs_diff_slope"] < 1e-9 and ctl["max_abs_diff_ci"] < 1e-9 and row["n_seg"] == 363)
    R["control_rule6_reproduction"] = ctl
    print(f"[{time.time() - t0:6.1f}s] CONTROL EMA-15k {row['slope']['EMA-15k']['slope']:+.6f} (banked -0.140937)  r7 {row['slope']['refcv7-50400']['slope']:+.6f} "
          f"(banked -0.092130)  PASS={ctl['PASS']}", flush=True)
    assert ctl["PASS"], f"REPRODUCTION CONTROL FAILED: {ctl}"
    # own implementation of the same slope on the same arm set must agree too (guards the new Ctx code)
    c_chk = Ctx(hg, sw, np.all([np.isfinite(pdh[l]) for l in pdh], 0) & gvalid, sharp, [K - 1], inv, E)
    own = c_chk.sxy(pdh["EMA-15k"]).sum() / c_chk.Sxx.sum()
    assert abs(own - row["slope"]["EMA-15k"]["slope"]) < 1e-12, "Ctx slope != hd.shrinkage slope"
    ctl["ctx_implementation_matches_hd_shrinkage"] = True

    # ---------------- cross-tree controls
    ctr = {}
    for s in SEEDS:
        ctr[f"r7 own os vs embedded b_refcv7_50400_s{s} (max abs diff, m)"] = float(np.abs(P[("r7", s)]["os"].astype(np.float64) - P[("EMA-15k", s)]["ref"].astype(np.float64)).max())
        ctr[f"nav_cmd r7 == r8 EMA-15k, s{s} (frac equal)"] = float((P[("r7", s)]["nav_cmd"] == P[("EMA-15k", s)]["nav_cmd"]).mean())
        for lab in TREES:
            ctr[f"os==oracle_sel where sel_idx==a_star (max abs m), {lab}|s{s}"] = float(np.abs(
                (P[(lab, s)]["os"] - P[(lab, s)]["oracle_sel"]).astype(np.float64)[(P[(lab, s)]["sel_idx"] == P[(lab, s)]["a_star"])]).max())
            ctr[f"sel_agrees_oracle == (sel_idx==a_star), {lab}|s{s} (frac equal)"] = float(((P[(lab, s)]["sel_idx"] == P[(lab, s)]["a_star"]) == (P[(lab, s)]["sel_agrees_oracle"] > 0.5)).mean())
        ctr[f"os==oracle_sel where sel_idx==a_star (max abs m), r7|s{s}"] = float(np.abs((P[("r7", s)]["os"] - P[("r7", s)]["oracle_sel"]).astype(np.float64)[P[("r7", s)]["sel_idx"] == P[("r7", s)]["a_star"]]).max())
        ctr[f"r7 max sel_idx s{s} (117 anchors -> <=116)"] = float(P[("r7", s)]["sel_idx"].max())
        ctr[f"r8 EMA-15k max sel_idx s{s} (149 candidates -> <=148)"] = float(P[("EMA-15k", s)]["sel_idx"].max())
        ctr[f"oracle_sel seed difference max abs (EMA-15k s0 vs s1, m)"] = float(np.abs(P[("EMA-15k", 0)]["oracle_sel"].astype(np.float64) - P[("EMA-15k", 1)]["oracle_sel"].astype(np.float64)).max())
    # nav-arm availability per seed
    for m_ in ("EMA-15k", "r7"):
        for s in SEEDS:
            ctr[f"nav arms finite fraction {m_}|s{s}"] = {k: float(np.isfinite(P[(m_, s)][k]).mean()) for k in NAV_ARMS}
    # anchor bank identity across the two checkpoints + manifests
    banks = {}
    for k, p in CKPT.items():
        ck = torch.load(p, mmap=True, weights_only=True, map_location="cpu")["model"]
        banks[k] = (ck["core.decoder.anchors"].clone().numpy(), ck["core.decoder.anchor_controls"].clone().numpy())
        del ck
    ctr["anchor bank: r7 == r8 ckpt (anchors, controls) bitwise"] = bool(np.array_equal(banks["r7"][0], banks["r8"][0]) and np.array_equal(banks["r7"][1], banks["r8"][1]))
    ctr["anchor bank sha256[:16] (anchors bytes | controls bytes), r8 ckpt"] = [hashlib.sha256(banks["r8"][0].tobytes()).hexdigest()[:16], hashlib.sha256(banks["r8"][1].tobytes()).hexdigest()[:16]]
    ctr["manifest anchor_file_vs_ckpt_buffers max_abs_diff (all trees)"] = {
        f"{lab}": {kk: vv.get("max_abs_diff") for kk, vv in ((P[(lab, 0)]["model"].get("refcv6_loader") or {}).get("anchor_file_vs_ckpt_buffers") or {}).items()} for lab in TREES}
    R["controls"] = ctr
    anchors, alat_all = banks["r8"]
    assert alat_all.shape == (N_BANK, 2)
    print(f"[{time.time() - t0:6.1f}s] controls done", flush=True)

    # ---------------- (1) SLOPE DECOMPOSITION
    GROUPS = {"sharp (GT |yaw| >= 0.15)": sharp, "curved (GT |yaw| >= 0.02)": curved}
    SEGS = {"last segment (1.5->2.0 s chord)": [K - 1], "segments 2..K": list(range(1, K))}
    R["decomposition"] = {}
    DH = {}
    for lab in TREES:
        for s in SEEDS:
            DH[(lab, s, "os")] = dh(P[(lab, s)]["os"])
            DH[(lab, s, "orc")] = dh(P[(lab, s)]["oracle_sel"])
    for s in SEEDS:
        DH[("r7", s, "os")] = dh(P[("r7", s)]["os"])
        DH[("r7", s, "orc")] = dh(P[("r7", s)]["oracle_sel"])
    for lab in TREES:
        R["decomposition"][lab] = {}
        views = {"2-seed mean": lambda arm, lb: 0.5 * (DH[(lb, 0, arm)] + DH[(lb, 1, arm)]),
                 "seed 0": lambda arm, lb: DH[(lb, 0, arm)], "seed 1": lambda arm, lb: DH[(lb, 1, arm)]}
        for vname, fn in views.items():
            dhs = {"r8os": fn("os", lab), "r8orc": fn("orc", lab), "r7os": fn("os", "r7"), "r7orc": fn("orc", "r7")}
            valid = np.all([np.isfinite(v) for v in dhs.values()], 0) & gvalid
            for gname, gm in GROUPS.items():
                for sname, ks in SEGS.items():
                    if vname != "2-seed mean" and not (gname.startswith("sharp") and sname.startswith("last")):
                        continue
                    if lab not in ("EMA-15k",) and not (gname.startswith("sharp") and vname == "2-seed mean"):
                        continue
                    ctx = Ctx(hg, sw, valid, gm, ks, inv, E)
                    R["decomposition"][lab][f"{vname} | {gname} | {sname}"] = decomposition(ctx, dhs, W)
        d = R["decomposition"][lab]["2-seed mean | sharp (GT |yaw| >= 0.15) | last segment (1.5->2.0 s chord)"]
        print(f"[{time.time() - t0:6.1f}s] {lab} sharp-last: GAP {cellmd(d['terms']['GAP (r8 os - r7 os)'])}  CAND {cellmd(d['terms']['CANDIDATE (r8 orc - r7 orc)'])}  "
              f"SEL {cellmd(d['terms']['SELECTION ((r8 os-orc) - (r7 os-orc))'])}", flush=True)

    # ---------------- (2) SELECTION DETAIL
    R["selection"] = {}
    # 2c: agreement + a_star + extras fraction, all trees, per seed, all/sharp
    wide = np.ones(N, bool)

    def csub(vals, mask):
        return rstat(np.bincount(inv, weights=np.where(mask, vals, 0.0), minlength=E), np.bincount(inv, weights=mask.astype(float), minlength=E), W, True)

    agree = {}
    for lab in TREES:
        for s in SEEDS:
            Pr, P7 = P[(lab, s)], P[("r7", s)]
            ex8 = (Pr["sel_idx"] >= N_BANK).astype(float)
            ag8, ag7 = (Pr["sel_agrees_oracle"] > 0.5).astype(float), (P7["sel_agrees_oracle"] > 0.5).astype(float)
            ast = (Pr["a_star"] == P7["a_star"]).astype(float)
            row = {}
            for gname, gm in (("all", wide), ("sharp", sharp), ("curved", curved)):
                r_ex, _ = csub(ex8, gm)
                r8a, b8 = csub(ag8, gm)
                r7a, b7 = csub(ag7, gm)
                r_as, _ = csub(ast, gm)
                row[gname] = {"n_win": int(gm.sum()), "frac_r8_selects_allocator_extra (sel_idx>=117)": r_ex, "sel_agrees_oracle r8": r8a, "sel_agrees_oracle r7": r7a,
                              "sel_agrees_oracle r8 - r7": bs_diff(r8a["est"], b8, r7a["est"], b7), "a_star identical r8 vs r7 (frac)": r_as}
            agree[f"{lab}|s{s}"] = row
    R["selection"]["agreement_and_extras"] = agree

    # 2a: selected vs oracle anchor control / implied turn rate (EMA-15k bank; sharp windows; in-bank selections only)
    al = alat_all[:, 1].astype(np.float64)
    out_alat = {}
    a_or = P[("EMA-15k", 0)]["a_star"].astype(int)
    cor = float(np.corrcoef(al[a_or][sharp], B["gt_yaw_signed"][sharp])[0, 1])
    orient = 1.0 if cor >= 0 else -1.0
    gate_v = v0 >= 2.0
    sg = np.sign(B["gt_yaw_signed"])
    yaw_abs = np.abs(B["gt_yaw_signed"])
    gy2 = G[:, -1, 1].astype(np.float64)
    out_alat["anchor_index_faithfulness_check"] = {
        "corr(alat[a_star], GT lateral y@2s), all windows": float(np.corrcoef(al[a_or], gy2)[0, 1]),
        "corr(alat[a_star], GT lateral y@2s), windows with |y@2s|>3 m": float(np.corrcoef(al[a_or][np.abs(gy2) > 3], gy2[np.abs(gy2) > 3])[0, 1]),
        "corr(bank anchor terminal y[a_star], GT lateral y@2s), all windows": float(np.corrcoef(banks['r8'][0][a_or, -1, 1], gy2)[0, 1]),
        "corr(oracle_sel lateral y@2s, GT lateral y@2s), all windows": float(np.corrcoef(P[("EMA-15k", 0)]["oracle_sel"][:, -1, 1], gy2)[0, 1]),
        "n_windows_|y@2s|>3": int((np.abs(gy2) > 3).sum()),
        "reading": "if the refined oracle plan tracks GT (corr~1) while the raw anchor control of a_star does not, the anchor INDEX is not a geometric prototype of the final plan: the refinement, not the bank, carries the turn"}
    out_alat["orientation_check"] = {"corr(alat_col1[a_star], gt_yaw_signed) on sharp windows (EMA-15k s0)": cor, "orientation_sign_used": orient,
                                     "controls_col0_vs_col1_corr_with_anchor_terminal_y": [float(np.corrcoef(banks['r8'][1][:, c], banks['r8'][0][:, -1, 1])[0, 1]) for c in (0, 1)],
                                     "units_note": "controls col1 = lateral acceleration (m/s^2) per the refc anchor builder (alat); verified only by the correlation with the anchors' terminal lateral offset above"}
    for lab in TREES:
        for s in SEEDS:
            if lab != "EMA-15k":
                continue
            for mname, M in (("r8", P[(lab, s)]), ("r7", P[("r7", s)])):
                sel = M["sel_idx"].astype(int)
                inb = sel < N_BANK
                msk = sharp & inb & gate_v
                selal = np.where(inb, al[np.clip(sel, 0, N_BANK - 1)], np.nan)
                orcal = al[M["a_star"].astype(int)]
                row = {"n_sharp_v0>=2": int((sharp & gate_v).sum()), "n_sharp_v0>=2_in_bank_selection": int(msk.sum())}
                for nm, x in (("|alat| selected anchor (m/s^2)", np.abs(selal)), ("|alat| oracle anchor (m/s^2)", np.abs(orcal))):
                    row[nm] = csub(np.nan_to_num(x), msk)[0]
                r_d, _ = csub(np.nan_to_num(np.abs(selal) - np.abs(orcal)), msk)
                row["|alat| selected - oracle"] = r_d
                # implied turn-rate ratio: through-origin slope of (turn-aligned alat/v0) on |GT yaw rate|   [ESTIMATED: constant speed v0 over the horizon]
                for nm, x in (("selected", selal), ("oracle", orcal)):
                    ta = np.nan_to_num(orient * x * sg / np.maximum(v0, 1.0))
                    num = np.bincount(inv, weights=np.where(msk, ta * yaw_abs, 0.0), minlength=E)
                    den = np.bincount(inv, weights=np.where(msk, yaw_abs ** 2, 0.0), minlength=E)
                    row[f"implied-turn-rate / GT-turn-rate slope, {nm} anchor (ESTIMATED)"] = rstat(num, den, W)
                # wrong-way selections: selected anchor curves opposite to the GT turn
                wrong = np.nan_to_num((orient * selal * sg) < 0).astype(float)
                row["frac selected anchor curves the WRONG way (turn-aligned alat<0)"] = csub(wrong, msk)[0]
                row["frac oracle anchor curves the wrong way"] = csub(((orient * orcal * sg) < 0).astype(float), msk)[0]
                out_alat[f"{mname}|s{s}"] = row
    R["selection"]["anchor_alat_sharp"] = out_alat

    # 2b: allocator extras -- slope on extras vs rest, gap / selection-cost contributions, selection-cost class decomposition (per seed; sharp, last segment)
    classes = {}
    for lab in TREES:
        classes[lab] = {}
        for s in SEEDS:
            dhs = {"r8os": DH[(lab, s, "os")], "r8orc": DH[(lab, s, "orc")], "r7os": DH[("r7", s, "os")], "r7orc": DH[("r7", s, "orc")]}
            valid = np.all([np.isfinite(v) for v in dhs.values()], 0) & gvalid
            ctx = Ctx(hg, sw, valid, sharp, [K - 1], inv, E)
            Pr, P7 = P[(lab, s)], P[("r7", s)]
            sel8, sel7 = Pr["sel_idx"], P7["sel_idx"]
            ext = sel8 >= N_BANK
            agr8 = (sel8 == Pr["a_star"]) & ~ext
            dis8 = ~ext & ~agr8
            agr7 = sel7 == P7["a_star"]
            S = {k: ctx.sxy(v) for k, v in dhs.items()}
            full_gap = (S["r8os"] - S["r7os"]).sum() / ctx.Sxx.sum()
            full_sel8 = (S["r8os"] - S["r8orc"]).sum() / ctx.Sxx.sum()
            full_sel7 = (S["r7os"] - S["r7orc"]).sum() / ctx.Sxx.sum()
            rows = {"n_sharp_last_windows_valid_all_arms": ctx.n_win, "frac_extra_among_these": float(ext[ctx.m.any(1)].mean())}
            tot_gap = tot_s8 = 0.0
            cl = {}
            for cname, cm in (("EXTRA (r8 sel_idx>=117)", ext), ("IN-BANK agrees with oracle anchor", agr8), ("IN-BANK disagrees with oracle anchor", dis8), ("REST (all non-extra)", ~ext)):
                sub = {k: ctx.sxy(v, cm) for k, v in dhs.items()}
                den_sub = ctx.sxx(cm)
                row = {"n_win": ctx.nwin(cm), "n_ep": int((den_sub > 0).sum())}
                if den_sub.sum() > 0:
                    r8os, b8os = rstat(sub["r8os"], den_sub, W, True)
                    r8o, _ = rstat(sub["r8orc"], den_sub, W, True)
                    r7os, b7os = rstat(sub["r7os"], den_sub, W, True)
                    r7o, _ = rstat(sub["r7orc"], den_sub, W, True)
                    row.update({"slope r8 os": r8os, "slope r8 oracle_sel": r8o, "slope r7 os": r7os, "slope r7 oracle_sel": r7o,
                                "slope r8 os - r7 os (same windows)": bs_diff(r8os["est"], b8os, r7os["est"], b7os)})
                row["contribution to full-sample GAP (num/full Sxx)"] = rstat(sub["r8os"] - sub["r7os"], ctx.Sxx, W)
                row["contribution to r8 selection cost (os-orc)"] = rstat(sub["r8os"] - sub["r8orc"], ctx.Sxx, W)
                cl[cname] = row
                if cname.startswith(("EXTRA", "IN-BANK")):
                    tot_gap += (sub["r8os"] - sub["r7os"]).sum() / ctx.Sxx.sum()
                    tot_s8 += (sub["r8os"] - sub["r8orc"]).sum() / ctx.Sxx.sum()
            assert abs(tot_gap - full_gap) < 1e-12 and abs(tot_s8 - full_sel8) < 1e-12, "class partition does not sum to the totals"
            # extras-vs-rest slope difference with a shared bootstrap
            sub_e, sub_r = ctx.sxy(dhs["r8os"], ext), ctx.sxy(dhs["r8os"], ~ext)
            de, dr = ctx.sxx(ext), ctx.sxx(~ext)
            if de.sum() > 0:
                re_, be_ = rstat(sub_e, de, W, True)
                rr_, br_ = rstat(sub_r, dr, W, True)
                rows["r8 os slope EXTRA minus REST"] = bs_diff(re_["est"], be_, rr_["est"], br_)
                # does the extras subset look hard for EVERY model (r7 on the same windows)?
                r7e, b7e = rstat(ctx.sxy(dhs["r7os"], ext), de, W, True)
                r7r, b7r = rstat(ctx.sxy(dhs["r7os"], ~ext), dr, W, True)
                rows["r7 os slope EXTRA-windows minus REST (windows defined by r8's selection)"] = bs_diff(r7e["est"], b7e, r7r["est"], b7r)
            # r7 selection-cost classes (r7's own selector): agree / disagree
            c7 = {}
            for cname, cm in (("r7 agrees with oracle anchor", agr7), ("r7 disagrees", ~agr7)):
                c7[cname] = {"n_win": ctx.nwin(cm), "contribution to r7 selection cost (os-orc)": rstat(ctx.sxy(dhs["r7os"], cm) - ctx.sxy(dhs["r7orc"], cm), ctx.Sxx, W)}
            rows.update({"classes_by_r8_selection": cl, "classes_by_r7_selection": c7,
                         "full: GAP": full_gap, "full: r8 selection cost": full_sel8, "full: r7 selection cost": full_sel7})
            classes[lab][f"s{s}"] = rows
    R["selection"]["extras_and_classes_sharp_last"] = classes
    print(f"[{time.time() - t0:6.1f}s] selection detail done", flush=True)

    # ---------------- (3) NAV: does refcv8 use the turn signal less?  (sharp turns, last segment; seed 0 [and seed 1 if finite])
    navres = {}
    for s in SEEDS:
        arms = {}
        for mname, M in (("r8", P[("EMA-15k", s)]), ("r7", P[("r7", s)])):
            arms[(mname, "os")] = dh(M["os"])
            for k in NAV_ARMS:
                arms[(mname, k)] = dh(M[k])
        if not all(np.isfinite(v[gvalid]).any() for v in arms.values()):
            navres[f"seed{s}"] = {"status": "nav arms not present for this seed (all-NaN) -- skipped"}
            continue
        valid = np.all([np.isfinite(v) for v in arms.values()], 0) & gvalid
        nv = P[("EMA-15k", s)]["nav_cmd"]
        subg = {"sharp": sharp, "sharp & nav=left/right": sharp & np.isin(nv, (1, 2)), "sharp & nav=follow": sharp & (nv == 0), "curved": curved}
        navres[f"seed{s}"] = {}
        for gname, gm in subg.items():
            ctx = Ctx(hg, sw, valid, gm, [K - 1], inv, E)
            if ctx.n_win < 15:
                navres[f"seed{s}"][gname] = {"n_win": ctx.n_win, "status": "too few windows"}
                continue
            sl, bsd = {}, {}
            for k, v in arms.items():
                sl[k], bsd[k] = rstat(ctx.sxy(v), ctx.Sxx, W, True)
            res = {"n_win": ctx.n_win, "n_ep": ctx.n_ep, "slopes": {f"{m_} {k}": sl[(m_, k)] for (m_, k) in arms}}

            def comb(c):                                              # linear combination of arm slopes -> est + shared-bootstrap CI
                est = sum(w * sl[k]["est"] for k, w in c.items())
                bs = sum(w * bsd[k] for k, w in c.items())
                lo, hi = np.percentile(bs, [2.5, 97.5])
                return {"est": float(est), "lo": float(lo), "hi": float(hi), "separated": bool(lo > 0 or hi < 0)}
            for m_ in ("r8", "r7"):
                res[f"{m_} nav response: slope(os) - slope(navzero)  [+ = nav makes the plan turn MORE]"] = comb({(m_, "os"): 1, (m_, "os_navzero"): -1})
                res[f"{m_} flip response: slope(os) - slope(navflip)"] = comb({(m_, "os"): 1, (m_, "os_navflip"): -1})
                res[f"{m_} shuffle response: slope(os) - slope(navshuf)"] = comb({(m_, "os"): 1, (m_, "os_navshuf"): -1})
            res["r8 - r7: nav response"] = comb({("r8", "os"): 1, ("r8", "os_navzero"): -1, ("r7", "os"): -1, ("r7", "os_navzero"): 1})
            res["r8 - r7: flip response"] = comb({("r8", "os"): 1, ("r8", "os_navflip"): -1, ("r7", "os"): -1, ("r7", "os_navflip"): 1})
            res["r8 - r7: slope(os) in this valid set"] = comb({("r8", "os"): 1, ("r7", "os"): -1})
            res["r8 - r7: slope(navzero)  [the nav-free plan]"] = comb({("r8", "os_navzero"): 1, ("r7", "os_navzero"): -1})
            navres[f"seed{s}"][gname] = res
    R["nav"] = navres

    R["free_commit_gb_start_end"] = [g0, guard("end")]
    R["runtime_s"] = time.time() - t0
    os.makedirs(a.out_dir, exist_ok=True)
    jp = os.path.join(a.out_dir, a.tag + ".json")
    json.dump(R, open(jp, "w", encoding="utf-8"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    open(os.path.join(a.out_dir, a.tag + ".md"), "w", encoding="utf-8").write(render_md(R))
    print("WROTE", jp, f"({R['runtime_s']:.0f}s)")


# ------------------------------------------------------------------------------------------------ markdown
def render_md(R: dict) -> str:
    L = []
    A = L.append
    A("# FAN vs SELECTOR -- candidate plans or selection? (refcv8 tight-turn under-steering)\n")
    A(f"- **Tier:** {R['tier']}\n- **Evidence class:** {R['evidence_class']}\n- **Estimator:** {R['estimator']}")
    A(f"- **n:** {R['n_windows']} S2 windows / {R['n_episodes']} episodes; sharp (GT mean |yaw rate| >= 0.15 rad/s) = {R['windows']['sharp (GT |yaw|>=0.15)']} windows; curved (>= 0.02) = {R['windows']['curved (>=0.02)']}.")
    A("- **Metric:** through-origin slope of turn-aligned signed heading error on turn-aligned GT chord heading (-0.10 = plan turns 10 % less than GT), "
      "`hd.shrinkage` definition, 2-seed mean of the signed per-segment error, segments valid on every arm of the contrast.")
    A("- **Not answered:** another TRAINING run (one checkpoint per arm; H-ESTIM-SEED-1) and another INFERENCE run (per-seed rows only).\n")
    c = R["control_rule6_reproduction"]
    A("## 0. Reproduction control (rule 6; asserted before anything else)\n")
    A(f"banked EMA-15k sharp last-segment slope {c['banked']['EMA-15k']:+.6f}, refcv7 {c['banked']['refcv7-50400']:+.6f}; reproduced through the ORIGINAL `hd.shrinkage` "
      f"{c['reproduced']['EMA-15k']:+.6f} / {c['reproduced']['refcv7-50400']:+.6f} (n_seg {c['reproduced']['n_seg']}); max |diff| slope {c['max_abs_diff_slope']:.2e}, "
      f"CI {c['max_abs_diff_ci']:.2e}; **PASS = {c['PASS']}**. The new `Ctx` slope code matches `hd.shrinkage` to 1e-12.\n")
    A("Other controls (all MEASURED):\n")
    for k, v in R["controls"].items():
        A(f"- {k}: `{v if not isinstance(v, float) else format(v, '.6g')}`")
    A("\n## 1. Slope decomposition (identity asserted at the point estimate and on all bootstrap draws)\n")
    A("gap = CANDIDATE (anchor coverage + refinement: r8 `oracle_sel` - r7 `oracle_sel`) + SELECTION ((r8 os - r8 orc) - (r7 os - r7 orc)). Negative = more under-rotation.\n")
    for lab, dd in R["decomposition"].items():
        A(f"### {lab} vs refcv7-50,400\n")
        for view, d in dd.items():
            A(f"**{view}** -- n_win {d['n_win']}, n_seg {d['n_seg']}, n_ep {d['n_ep']}, mean GT turn-aligned heading {d['mean_gt_turn_aligned_heading_deg']:.1f} deg; identity err {d['identity_max_abs_err_all_boot_draws']:.1e}\n")
            A("| term | est [95 % CI] |\n|---|---|")
            for k, v in d["terms"].items():
                A(f"| {k} | {cellmd(v)} |")
            sc, ss = d["share_candidate"], d["share_selection"]
            A(f"| share CANDIDATE of GAP | {sc['est']:.2f} [{sc['lo']:.2f}, {sc['hi']:.2f}] |\n| share SELECTION of GAP | {ss['est']:.2f} [{ss['lo']:.2f}, {ss['hi']:.2f}] |")
            A(f"\n(shares are ratios of correlated bootstrap terms; P(boot gap has the point sign) = {sc['frac_boot_gap_same_sign_as_point']:.3f})\n")
    A("\n## 2. Selection detail\n")
    A("### 2c. Agreement with the oracle anchor, allocator-extra selection rate, a_star identity\n")
    A("| tree|seed | group | n | r8 selects extra (117-148) | sel_agrees_oracle r8 | r7 | r8 - r7 | a_star same r8/r7 |\n|---|---|---|---|---|---|---|---|")
    for k, row in R["selection"]["agreement_and_extras"].items():
        for gname, r in row.items():
            A(f"| {k} | {gname} | {r['n_win']} | {cellmd(r['frac_r8_selects_allocator_extra (sel_idx>=117)'], 3)} | {cellmd(r['sel_agrees_oracle r8'], 3)} | {cellmd(r['sel_agrees_oracle r7'], 3)} | {cellmd(r['sel_agrees_oracle r8 - r7'], 3)} | {cellmd(r['a_star identical r8 vs r7 (frac)'], 3)} |")
    A("\n### 2a. Selected vs oracle anchor (EMA-15k bank; sharp windows, v0 >= 2 m/s, in-bank selections)\n")
    oa = R["selection"]["anchor_alat_sharp"]
    A(f"orientation/units check: `{json.dumps(oa['orientation_check'])}`\n")
    A(f"anchor-index faithfulness check: `{json.dumps(oa['anchor_index_faithfulness_check'])}`\n")
    A("**CAUTION: the |alat| / implied-turn-rate rows below are only interpretable if the anchor index is a geometric prototype of the final plan; see the faithfulness check above.**\n")
    for k, row in oa.items():
        if k in ("orientation_check", "anchor_index_faithfulness_check"):
            continue
        A(f"**{k}**\n\n| quantity | est [95 % CI] |\n|---|---|")
        for kk, vv in row.items():
            A(f"| {kk} | {cellmd(vv) if isinstance(vv, dict) else vv} |")
        A("")
    A("\n### 2b. Allocator extras and selection-class decomposition (sharp turns, last segment, per inference seed)\n")
    for lab, d1 in R["selection"]["extras_and_classes_sharp_last"].items():
        for sk, rows in d1.items():
            A(f"**{lab} | {sk}** -- n_win {rows['n_sharp_last_windows_valid_all_arms']}; frac r8-extra {rows['frac_extra_among_these']:.3f}; full GAP {rows['full: GAP']:+.4f}; "
              f"r8 selection cost {rows['full: r8 selection cost']:+.4f}; r7 selection cost {rows['full: r7 selection cost']:+.4f}\n")
            A("| r8 selection class | n_win | slope r8 os | slope r8 orc | slope r7 os | contrib. to GAP | contrib. to r8 sel. cost |\n|---|---|---|---|---|---|---|")
            for cn, r in rows["classes_by_r8_selection"].items():
                A(f"| {cn} | {r['n_win']} | {cellmd(r.get('slope r8 os'))} | {cellmd(r.get('slope r8 oracle_sel'))} | {cellmd(r.get('slope r7 os'))} | {cellmd(r['contribution to full-sample GAP (num/full Sxx)'])} | {cellmd(r['contribution to r8 selection cost (os-orc)'])} |")
            if "r8 os slope EXTRA minus REST" in rows:
                A(f"\nr8 os slope EXTRA - REST: {cellmd(rows['r8 os slope EXTRA minus REST'])}")
            A("\nr7 classes: " + "; ".join(f"{cn}: n {r['n_win']}, contrib. to r7 sel. cost {cellmd(r['contribution to r7 selection cost (os-orc)'])}" for cn, r in rows["classes_by_r7_selection"].items()) + "\n")
    A("\n## 3. Nav: does refcv8 use the turn signal less? (EMA-15k vs refcv7; last segment)\n")
    for sk, gd in R["nav"].items():
        A(f"### {sk}\n")
        if "status" in gd:
            A(gd["status"] + "\n")
            continue
        for gname, res in gd.items():
            if "status" in res:
                A(f"**{gname}**: n_win {res['n_win']} -- {res['status']}\n")
                continue
            A(f"**{gname}** -- n_win {res['n_win']}, n_ep {res['n_ep']}\n\n| quantity | est [95 % CI] |\n|---|---|")
            for k, v in res["slopes"].items():
                A(f"| slope {k} | {cellmd(v)} |")
            for k, v in res.items():
                if isinstance(v, dict) and k not in ("slopes",) and "est" in v:
                    A(f"| {k} | {cellmd(v)} |")
            A("")
    A(f"\nruntime {R['runtime_s']:.0f} s; free commit at start/end {R['free_commit_gb_start_end'][0]:.2f}/{R['free_commit_gb_start_end'][1]:.2f} GB.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
