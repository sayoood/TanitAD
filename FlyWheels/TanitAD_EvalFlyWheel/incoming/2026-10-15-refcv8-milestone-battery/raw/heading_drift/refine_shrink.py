"""refine_shrink.py -- does refcv8's REFINEMENT step (diffusion sampler on the selected anchor) shrink tight turns?
CPU only, existing per-window dumps (S2 panels + decisions/ sidecars), no GPU, no git.   Imports fan_vs_selector / heading_drift, edits neither.

    OMP_NUM_THREADS=4 python refine_shrink.py --out-dir <dir> --tag REFINE_SHRINK

Per arm, on sharp windows (GT mean |yaw| >= 0.15) where the arm's selection is IN-BANK (sel_idx < 117; the sidecar `sel_bank_nav_true` has rows only there):
    a = slope of the UNREFINED selected anchor path  (sel_bank_nav_true[:, :4], model slots 0..3 = 0.5/1.0/1.5/2.0 s)
    b = slope of the REFINED emitted plan            (plan_full_nav_true[:, :4]  == panel `os`, asserted bitwise)
    refinement effect = b - a ;  DIFF_EFFECT = (b8 - a8) - (b7 - a7) ;  exact identity  GAP_refined = GAP_unrefined + DIFF_EFFECT  (asserted on every bootstrap draw)
Slope = hd.shrinkage definition (through-origin slope of turn-aligned signed heading error on turn-aligned GT chord heading, last 1.5->2.0 s chord).
ONE shared episode-cluster resample matrix (139 episodes, B 2000, seed 29) -> every term / difference is paired.  Answers 'another draw of EPISODES,
models FIXED' only -- NOT another training run (H-ESTIM-SEED-1), NOT another inference run (seed 0 / seed 1 rows are reported separately).
Rule 6 control runs BEFORE anything else: the banked EMA-15k / refcv7 `os` and `oracle_sel` slopes and CIs must reproduce.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fan_vs_selector as fvs                                            # noqa: E402  (imported, never edited)

hd = fvs.hd
N_BANK, SEEDS, TREES, R7_ROOT = fvs.N_BANK, fvs.SEEDS, fvs.TREES, fvs.R7_ROOT
EXTRA_KEYS = ("sel_idx_nav_true", "plan_full_nav_true", "fan_term_heading_nav_true", "reach_keep_nav_true",
              "gt_future_ext", "gt_future_valid_ext", "pose_last")
TAU_PRIMARY = 0.1806411302089691          # rad; the nav-compliance tolerance in refcv8 config.json (--nav-compliance-tau-rad)
TAU_SECOND = 0.35                          # rad (20 deg), pre-registered second tolerance
MIN_CHORD_M = 1.0                          # a 1 s chord shorter than this has no meaningful heading (GT or plan)
BANKED_JSON = os.path.join(HERE, "FAN_VS_SELECTOR.json")
KEY_SHARP_LAST = "2-seed mean | sharp (GT |yaw| >= 0.15) | last segment (1.5->2.0 s chord)"


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


# ------------------------------------------------------------------------------------------------ sidecar loading
def load_extra(root: str, seed: int) -> dict:
    """sidecar keys needed here, rows in the SAME file/row order as fvs.load; sel_bank expanded to [n,8,2] with NaN on non-in-bank rows."""
    pd = hd.panel_dir(root, "S2", seed)
    files = sorted(glob.glob(os.path.join(pd, "ep*.npz")))
    acc = {k: [] for k in EXTRA_KEYS + ("sel_bank_full", "sel_idx")}
    for f in files:
        with np.load(os.path.join(pd, "decisions", os.path.basename(f))) as d:
            n = d["ws"].shape[0]
            sel = d["sel_idx"].astype(np.int64)
            inb = sel < N_BANK
            sb = d["sel_bank_nav_true"]
            if sb.shape[0] != int(inb.sum()):
                raise RuntimeError(f"{f}: sel_bank rows {sb.shape[0]} != in-bank windows {int(inb.sum())}")
            if not np.array_equal(d["sel_idx_nav_true"].astype(np.int64), sel):
                raise RuntimeError(f"{f}: sel_idx != sel_idx_nav_true")
            full = np.full((n,) + sb.shape[1:], np.nan, np.float32)
            full[inb] = sb
            acc["sel_bank_full"].append(full)
            acc["sel_idx"].append(sel)
            for k in EXTRA_KEYS:
                acc[k].append(d[k])
    return {k: np.concatenate(v) for k, v in acc.items()}


def reorder_extra(X: dict, keys_src: list, keys_ref: list) -> dict:
    pos = {k: i for i, k in enumerate(keys_src)}
    idx = np.asarray([pos[k] for k in keys_ref])
    return {k: v[idx] for k, v in X.items()}


# ------------------------------------------------------------------------------------------------ statistics
REFINE_COMBOS = {
    "slope r8 UNREFINED selected anchor": {"a8": 1}, "slope r8 REFINED emitted plan": {"b8": 1},
    "slope r7 UNREFINED selected anchor": {"a7": 1}, "slope r7 REFINED emitted plan": {"b7": 1},
    "refinement effect r8 (refined - unrefined)": {"b8": 1, "a8": -1},
    "refinement effect r7 (refined - unrefined)": {"b7": 1, "a7": -1},
    "DIFF_EFFECT (effect r8 - effect r7)": {"b8": 1, "a8": -1, "b7": -1, "a7": 1},
    "GAP_unrefined (r8 - r7, selected anchors)": {"a8": 1, "a7": -1},
    "GAP_refined (r8 - r7, emitted plans)": {"b8": 1, "b7": -1},
}


def refine_block(ctx, dhs: dict, W) -> dict:
    S = {k: ctx.sxy(v) for k, v in dhs.items()}
    res, bs = {}, {}
    for name, c in REFINE_COMBOS.items():
        res[name], bs[name] = fvs.rstat(sum(v * S[k] for k, v in c.items()), ctx.Sxx, W, True)
    g, u, e = "GAP_refined (r8 - r7, emitted plans)", "GAP_unrefined (r8 - r7, selected anchors)", "DIFF_EFFECT (effect r8 - effect r7)"
    id_pt = res[g]["est"] - res[u]["est"] - res[e]["est"]
    id_bs = float(np.max(np.abs(bs[g] - bs[u] - bs[e])))
    assert abs(id_pt) < 1e-12 and id_bs < 1e-12, f"refinement identity broken: {id_pt} {id_bs}"
    out = {"n_seg": ctx.n_seg, "n_win": ctx.n_win, "n_ep": ctx.n_ep, "mean_gt_turn_aligned_heading_deg": ctx.mean_h,
           "terms": res, "identity_max_abs_err_point": abs(id_pt), "identity_max_abs_err_all_boot_draws": id_bs}
    # recovery fraction = share of the unrefined anchor's under-rotation that the refinement gives back: 1 - b/a   (DERIVED ratio; shared bootstrap)
    rec, rbs = {}, {}
    for arm in ("8", "7"):
        ka, kb = f"slope r{arm} UNREFINED selected anchor", f"slope r{arm} REFINED emitted plan"
        with np.errstate(invalid="ignore", divide="ignore"):
            rbs[arm] = 1.0 - bs[kb] / bs[ka]
        okf = np.isfinite(rbs[arm])
        lo, hi = np.percentile(rbs[arm][okf], [2.5, 97.5])
        rec[f"recovery fraction r{arm} (1 - refined/unrefined slope)"] = {"est": 1.0 - res[kb]["est"] / res[ka]["est"], "lo": float(lo), "hi": float(hi), "separated": bool(lo > 0 or hi < 0)}
    dd = rbs["8"] - rbs["7"]
    okf = np.isfinite(dd)
    lo, hi = np.percentile(dd[okf], [2.5, 97.5])
    rec["recovery fraction r8 - r7"] = {"est": rec["recovery fraction r8 (1 - refined/unrefined slope)"]["est"] - rec["recovery fraction r7 (1 - refined/unrefined slope)"]["est"],
                                        "lo": float(lo), "hi": float(hi), "separated": bool(lo > 0 or hi < 0)}
    out["recovery"] = rec
    for nm, key in (("share_DIFF_EFFECT_of_GAP_refined", e), ("share_GAP_unrefined_of_GAP_refined", u)):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = bs[key] / bs[g]
        ok = np.isfinite(r)
        lo, hi = np.percentile(r[ok], [2.5, 97.5])
        out[nm] = {"est": res[key]["est"] / res[g]["est"], "lo": float(lo), "hi": float(hi),
                   "note": "ratio of two correlated bootstrap terms; read only when GAP_refined is separated from 0"}
    return out


def csub(inv, E, W, vals, mask):
    m = np.asarray(mask, bool)
    return fvs.rstat(np.bincount(inv, weights=np.where(m, np.nan_to_num(vals), 0.0), minlength=E),
                     np.bincount(inv, weights=m.astype(float), minlength=E), W, True)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="REFINE_SHRINK")
    ap.add_argument("--n-boot", type=int, default=hd.N_BOOT)
    ap.add_argument("--taniteval-root", default="E:/Projects/TanitAD/taniteval")
    ap.add_argument("--stack-root", default="E:/Projects/TanitAD/stack")
    a = ap.parse_args()
    t0 = time.time()
    g0 = fvs.guard("start")
    torch, ci, ff = hd.setup(a.taniteval_root, a.stack_root)
    print(f"[{time.time() - t0:6.1f}s] torch imported; free commit {fvs.guard('after torch import'):.2f} GB", flush=True)

    labs = ("EMA-15k", "EMA-5k")
    P, X = {}, {}
    ref = None
    for lab in labs:
        for s in SEEDS:
            Q = fvs.load(TREES[lab], s)
            if ref is None:
                ref = Q
            P[(lab, s)] = Q if (lab, s) == ("EMA-15k", 0) else fvs.align_to(ref, Q)
            X[(lab, s)] = reorder_extra(load_extra(TREES[lab], s), Q["keys"], ref["keys"])
    for s in SEEDS:
        Q = fvs.load(R7_ROOT, s)
        P[("r7", s)] = fvs.align_to(ref, Q)
        X[("r7", s)] = reorder_extra(load_extra(R7_ROOT, s), Q["keys"], ref["keys"])
    print(f"[{time.time() - t0:6.1f}s] loaded; free commit {fvs.guard('after loads'):.2f} GB", flush=True)

    G, v0, eid, dt = ref["g"], ref["v0"], ref["eid"], ref["dt"]
    N = len(eid)
    inv, E, W = fvs.build_W(eid, a.n_boot)
    B = hd.gt_bins(ff, torch, G, v0, ref["nav_cmd"], dt)
    Gg = ff._seq_geometry(torch.as_tensor(G).float(), dt)
    hg = Gg["heading"].numpy().astype(np.float64) * 180.0 / math.pi
    gvalid = Gg["valid"].numpy()
    sw = np.where((B["turn"] >= 1) & (B["turn"] <= 3), np.sign(B["gt_yaw_signed"]), 0.0)
    sharp, curved = B["turn"] == 3, (B["turn"] >= 1) & (B["turn"] <= 3)
    K = gvalid.shape[1]
    dh = lambda plan: hd.seg_signed_heading(ff, torch, plan, G, dt)       # noqa: E731
    R = {"title": "refcv8 tight-turn under-steering: does the REFINEMENT step shrink the turn? (unrefined selected anchor vs refined emitted plan)",
         "tier": "T1-style open loop on the S2 panel windows (own action, `os`); the unrefined anchor is the selection surface of the same window",
         "evidence_class": "MEASURED (ours; per-window dumps listed under `inputs`)",
         "estimator": f"paired episode-cluster bootstrap, ONE shared resample matrix (E={E} episodes, B={a.n_boot}, rng seed {hd.SEED + 29}); 'another draw of EPISODES, "
                      "models fixed' only -- NOT another training run (H-ESTIM-SEED-1), NOT another inference run",
         "generated": time.strftime("%FT%T"), "n_windows": N, "n_episodes": E,
         "windows": {"sharp (GT |yaw|>=0.15)": int(sharp.sum()), "curved (>=0.02)": int(curved.sum())},
         "inputs": {"r8": {k: TREES[k] for k in labs}, "r7": R7_ROOT, "banked_reference": BANKED_JSON}}

    # =========================================================================================== RULE 6 CONTROL (first)
    banked = json.load(open(BANKED_JSON, encoding="utf-8"))
    ctl = {"literal_banked": {"EMA-15k os": -0.14093672306643532, "oracle r8": -0.0826, "oracle r7": -0.0479, "r7 os": -0.0921}}
    DH = {}
    for lab in labs:
        for s in SEEDS:
            DH[(lab, s, "os")] = dh(P[(lab, s)]["os"])
            DH[(lab, s, "orc")] = dh(P[(lab, s)]["oracle_sel"])
    for s in SEEDS:
        DH[("r7", s, "os")] = dh(P[("r7", s)]["os"])
        DH[("r7", s, "orc")] = dh(P[("r7", s)]["oracle_sel"])
    two = lambda arm, lb: 0.5 * (DH[(lb, 0, arm)] + DH[(lb, 1, arm)])      # noqa: E731
    ctl["reproduced_vs_FAN_VS_SELECTOR_json"] = {}
    for lab in labs:
        dhs = {"r8os": two("os", lab), "r8orc": two("orc", lab), "r7os": two("os", "r7"), "r7orc": two("orc", "r7")}
        valid = np.all([np.isfinite(v) for v in dhs.values()], 0) & gvalid
        c = fvs.Ctx(hg, sw, valid, sharp, [K - 1], inv, E)
        row, mx = {}, 0.0
        for nm, key in (("slope r8 os", "r8os"), ("slope r8 oracle_sel", "r8orc"), ("slope r7 os", "r7os"), ("slope r7 oracle_sel", "r7orc")):
            r = fvs.rstat(c.sxy(dhs[key]), c.Sxx, W, False)
            b = banked["decomposition"][lab][KEY_SHARP_LAST]["terms"][nm]
            d = max(abs(r["est"] - b["est"]), abs(r["lo"] - b["lo"]), abs(r["hi"] - b["hi"]))
            mx = max(mx, d)
            row[nm] = {"est": r["est"], "lo": r["lo"], "hi": r["hi"], "banked_est": b["est"], "max_abs_diff_est_lo_hi": d}
        row["n_seg"], row["max_abs_diff_all"] = c.n_seg, mx
        ctl["reproduced_vs_FAN_VS_SELECTOR_json"][lab] = row
        assert mx < 1e-9 and (c.n_seg == 363 or lab != "EMA-15k"), f"RULE-6 REPRODUCTION FAILED for {lab}: {row}"   # banked EMA-15k n_seg is 363
    e15 = ctl["reproduced_vs_FAN_VS_SELECTOR_json"]["EMA-15k"]
    assert abs(e15["slope r8 os"]["est"] - ctl["literal_banked"]["EMA-15k os"]) < 1e-9
    assert abs(e15["slope r8 oracle_sel"]["est"] + 0.083) < 5e-4 and abs(e15["slope r7 oracle_sel"]["est"] + 0.048) < 5e-4
    ctl["PASS"] = True
    R["control_rule6_reproduction"] = ctl
    print(f"[{time.time() - t0:6.1f}s] CONTROL PASS: EMA-15k os {e15['slope r8 os']['est']:+.6f} r7 os {e15['slope r7 os']['est']:+.6f} "
          f"oracle r8 {e15['slope r8 oracle_sel']['est']:+.4f} r7 {e15['slope r7 oracle_sel']['est']:+.4f} n_seg {e15['n_seg']}", flush=True)

    # =========================================================================================== NEW CONTROLS (frames / units / pairing)
    nc = {}
    for k, x in X.items():
        nc[f"plan_full[:, :4] == os (max abs m), {k[0]}|s{k[1]}"] = float(np.abs(x["plan_full_nav_true"][:, :4].astype(np.float64) - P[k]["os"].astype(np.float64)).max())
        inb = x["sel_idx"] < N_BANK
        nc[f"sel_bank rows finite iff sel_idx<117, {k[0]}|s{k[1]}"] = bool(np.array_equal(np.isfinite(x["sel_bank_full"][:, 0, 0]), inb))
        nc[f"frac in-bank (all windows / sharp), {k[0]}|s{k[1]}"] = [float(inb.mean()), float(inb[sharp].mean())]
    for s in SEEDS:                                                 # the unrefined bank path is a function of (anchor, v0) only -> arm-independent
        both = np.isfinite(X[("EMA-15k", s)]["sel_bank_full"][:, 0, 0]) & (X[("EMA-15k", s)]["sel_idx"] == X[("r7", s)]["sel_idx"])
        nc[f"same selected idx r8/r7 -> sel_bank identical (max abs m; n windows), s{s}"] = [
            float(np.abs(X[("EMA-15k", s)]["sel_bank_full"][both].astype(np.float64) - X[("r7", s)]["sel_bank_full"][both].astype(np.float64)).max()), int(both.sum())]
    gtd = np.abs(X[("EMA-15k", 0)]["gt_future_ext"].astype(np.float64) - X[("r7", 0)]["gt_future_ext"].astype(np.float64)).max()
    nc["gt_future_ext identical r8 EMA-15k vs r7 (max abs)"] = float(gtd)
    pl = X[("EMA-15k", 0)]["pose_last"].astype(np.float64)

    def ego_xy(pts):
        d = pts - pl[:, :2]
        c_, s_ = np.cos(-pl[:, 2]), np.sin(-pl[:, 2])
        return np.stack([d[:, 0] * c_ - d[:, 1] * s_, d[:, 0] * s_ + d[:, 1] * c_], -1)
    gt = X[("EMA-15k", 0)]["gt_future_ext"].astype(np.float64)
    gtv = X[("EMA-15k", 0)]["gt_future_valid_ext"] > 0.5
    fr = {i: float(np.abs(ego_xy(gt[:, i, :2]) - G[:, 0].astype(np.float64))[gtv[:, i]].max()) for i in (3, 4, 5)}
    nc["GT frame/time control: ego_xy(gt_future_ext[:, i]) vs panel g[:,0] (0.5 s), max abs m for i=3/4/5"] = fr
    assert fr[4] < 0.05 and fr[4] < fr[3] and fr[4] < fr[5], f"gt_future_ext index/time convention failed: {fr}"
    R["controls"] = nc
    assert nc["gt_future_ext identical r8 EMA-15k vs r7 (max abs)"] == 0.0
    print(f"[{time.time() - t0:6.1f}s] new controls: GT idx4 diff {fr[4]:.2e} m; plan_full==os max "
          f"{max(v for k_, v in nc.items() if k_.startswith('plan_full')):.1e}", flush=True)

    # =========================================================================================== (1)/(2) REFINEMENT EFFECT
    A, Bp = {}, {}
    for lab in labs:
        for s in SEEDS:
            A[(lab, s)] = dh(X[(lab, s)]["sel_bank_full"][:, :4])
            Bp[(lab, s)] = dh(X[(lab, s)]["plan_full_nav_true"][:, :4])
    for s in SEEDS:
        A[("r7", s)] = dh(X[("r7", s)]["sel_bank_full"][:, :4])
        Bp[("r7", s)] = dh(X[("r7", s)]["plan_full_nav_true"][:, :4])
    # known-value control: a plan equal to the GT has slope exactly 0
    gz = dh(G.astype(np.float32))
    cz = fvs.Ctx(hg, sw, np.isfinite(gz) & gvalid, sharp, [K - 1], inv, E)
    nc["known-value control: slope of GT-as-plan"] = float(cz.sxy(gz).sum() / cz.Sxx.sum())
    assert abs(nc["known-value control: slope of GT-as-plan"]) < 1e-12
    R["refinement"] = {}
    for lab in labs:
        views = {"2-seed mean": lambda arm, lb: 0.5 * (arm[(lb, 0)] + arm[(lb, 1)]),
                 "seed 0": lambda arm, lb: arm[(lb, 0)], "seed 1": lambda arm, lb: arm[(lb, 1)]}
        R["refinement"][lab] = {}
        for vname, fn in views.items():
            dhs = {"a8": fn(A, lab), "b8": fn(Bp, lab), "a7": fn(A, "r7"), "b7": fn(Bp, "r7")}
            valid = np.all([np.isfinite(v) for v in dhs.values()], 0) & gvalid
            for gname, gm in (("sharp (GT |yaw| >= 0.15)", sharp), ("curved (GT |yaw| >= 0.02)", curved)):
                for sname, ks in (("last segment (1.5->2.0 s chord)", [K - 1]), ("segments 2..K", list(range(1, K)))):
                    if lab != "EMA-15k" and not (gname.startswith("sharp") and sname.startswith("last")):
                        continue
                    if vname != "2-seed mean" and not (gname.startswith("sharp") and sname.startswith("last")):
                        continue
                    ctx = fvs.Ctx(hg, sw, valid, gm, ks, inv, E)
                    R["refinement"][lab][f"{vname} | {gname} | {sname}"] = refine_block(ctx, dhs, W)
        d = R["refinement"][lab]["2-seed mean | sharp (GT |yaw| >= 0.15) | last segment (1.5->2.0 s chord)"]["terms"]
        print(f"[{time.time() - t0:6.1f}s] {lab} sharp-last in-bank: a8 {fvs.cellmd(d['slope r8 UNREFINED selected anchor'])}  b8 {fvs.cellmd(d['slope r8 REFINED emitted plan'])}  "
              f"a7 {fvs.cellmd(d['slope r7 UNREFINED selected anchor'])}  b7 {fvs.cellmd(d['slope r7 REFINED emitted plan'])}  DIFF_EFFECT {fvs.cellmd(d['DIFF_EFFECT (effect r8 - effect r7)'])}", flush=True)
    # same-anchor / different-anchor split: where r8 and r7 pick the SAME bank index the unrefined path is identical, so any slope difference is the refinement FUNCTION
    R["refinement_by_anchor_agreement"] = {}
    for lab in labs:
        R["refinement_by_anchor_agreement"][lab] = {}
        for vname, seeds_ in (("2-seed mean", (0, 1)), ("seed 0", (0,)), ("seed 1", (1,))):
            same = np.ones(N, bool)
            for s in seeds_:
                same &= X[(lab, s)]["sel_idx"] == X[("r7", s)]["sel_idx"]
            fn = (lambda arm, lb: 0.5 * (arm[(lb, 0)] + arm[(lb, 1)])) if len(seeds_) == 2 else (lambda arm, lb, s=seeds_[0]: arm[(lb, s)])
            dhs = {"a8": fn(A, lab), "b8": fn(Bp, lab), "a7": fn(A, "r7"), "b7": fn(Bp, "r7")}
            valid = np.all([np.isfinite(v) for v in dhs.values()], 0) & gvalid
            for sname, mk in (("SAME anchor index in r8 and r7", same), ("DIFFERENT anchor index", ~same)):
                ctx = fvs.Ctx(hg, sw, valid & mk[:, None], sharp, [K - 1], inv, E)
                blk = refine_block(ctx, dhs, W)
                if sname.startswith("SAME"):                 # control: identical unrefined paths -> GAP_unrefined is exactly 0
                    assert abs(blk["terms"]["GAP_unrefined (r8 - r7, selected anchors)"]["est"]) < 1e-12, "same-anchor subset: unrefined paths differ"
                R["refinement_by_anchor_agreement"][lab][f"{vname} | {sname} | sharp | last segment"] = blk
    # mispaired control: the unrefined paths of EMA-15k shifted by one in-bank window -> must NOT reproduce the paired slope
    a8m = 0.5 * (A[("EMA-15k", 0)] + A[("EMA-15k", 1)])
    okr = np.where(np.isfinite(a8m).all(1))[0]
    mis = a8m.copy()
    mis[okr] = a8m[np.roll(okr, 1009)]
    cm = fvs.Ctx(hg, sw, np.isfinite(mis) & gvalid, sharp, [K - 1], inv, E)
    cp = fvs.Ctx(hg, sw, np.isfinite(a8m) & gvalid, sharp, [K - 1], inv, E)
    nc["mispaired control (unrefined r8, shifted by 1009 in-bank windows (stride-1 windows: a 1-window shift hits a near-identical neighbour)): slope vs paired slope"] = [float(cm.sxy(mis).sum() / cm.Sxx.sum()), float(cp.sxy(a8m).sum() / cp.Sxx.sum())]
    print(f"[{time.time() - t0:6.1f}s] mispaired {nc['mispaired control (unrefined r8, shifted by 1009 in-bank windows (stride-1 windows: a 1-window shift hits a near-identical neighbour)): slope vs paired slope']}", flush=True)

    # =========================================================================================== (6 s terminal-chord extension, same estimator)
    pg = ego_xy(gt[:, 59, :2]) - ego_xy(gt[:, 49, :2])
    gclen = np.hypot(pg[:, 0], pg[:, 1])
    gth = np.arctan2(pg[:, 1], pg[:, 0])
    gvalid6 = gtv[:, 59] & gtv[:, 49] & (gclen >= MIN_CHORD_M)

    def term(path):                                                 # chord of model slots 6->7 (5.0 -> 6.0 s) -> heading rad, NaN if stalled/absent
        d_ = path[:, 7].astype(np.float64) - path[:, 6].astype(np.float64)
        ln = np.hypot(d_[:, 0], d_[:, 1])
        th = np.arctan2(d_[:, 1], d_[:, 0])
        th[~(ln >= MIN_CHORD_M)] = np.nan
        return th
    hg6 = (gth * 180.0 / math.pi)[:, None]
    gv6 = gvalid6[:, None]
    dh6 = lambda path: (wrap(term(path) - gth) * 180.0 / math.pi)[:, None]          # noqa: E731
    A6 = {k: dh6(x["sel_bank_full"]) for k, x in X.items()}
    B6 = {k: dh6(x["plan_full_nav_true"]) for k, x in X.items()}
    R["refinement_6s_terminal_chord"] = {"_definition": "heading of the 5.0->6.0 s chord (model slots 6->7) minus GT's 5->6 s chord heading (ego frame at t0), same through-origin turn-aligned slope; "
                                         "needs GT not end-clamped and both chords >= 1 m; SECONDARY (large angles, wrap possible)"}
    for lab in labs:
        for vname, fn in (("2-seed mean", lambda arm, lb: 0.5 * (arm[(lb, 0)] + arm[(lb, 1)])), ):
            dhs = {"a8": fn(A6, lab), "b8": fn(B6, lab), "a7": fn(A6, "r7"), "b7": fn(B6, "r7")}
            valid = np.all([np.isfinite(v) for v in dhs.values()], 0) & gv6
            ctx = fvs.Ctx(hg6, sw, valid, sharp, [0], inv, E)
            R["refinement_6s_terminal_chord"][f"{lab} | {vname} | sharp"] = refine_block(ctx, dhs, W)
    print(f"[{time.time() - t0:6.1f}s] 6 s terminal-chord extension done", flush=True)

    # =========================================================================================== (3) FAN-LEVEL COVERAGE
    wsel = sharp & gvalid6
    fan = {"n_sharp_windows": int(sharp.sum()), "n_sharp_windows_with_valid_6s_GT_chord": int(wsel.sum()),
           "n_episodes": int(len(set(inv[wsel].tolist()))),
           "_definition": "GT terminal heading = heading of the GT 5.0->6.0 s chord (gt_future_ext idx 49->59, ego frame at t0, convention verified against panel g); candidate terminal heading = "
                          "fan_term_heading_nav_true (chord of the REFINED candidate's slots 6->7 = 5.0->6.0 s); error = wrap(candidate - GT).  hit@tau = at least one candidate in the subset has |error| <= tau.  "
                          "subsets: ALL = every emitted candidate (r8 149 = 117 bank + 32 allocator extras; r7 117), BANK117 = first 117 columns, SURV = candidates the S2 reach mask keeps.",
           "tau_primary_rad": TAU_PRIMARY, "tau_second_rad": TAU_SECOND, "rows": {}}
    st = {}
    for k, x in X.items():
        fth = x["fan_term_heading_nav_true"].astype(np.float64)
        rk = x["reach_keep_nav_true"] > 0.5
        err = np.abs(wrap(fth - gth[:, None]))
        for sub in ("ALL", "BANK117", "SURV"):
            e = err[:, :N_BANK] if sub == "BANK117" else (np.where(rk, err, np.inf) if sub == "SURV" else err)
            fsub = fth[:, :N_BANK] if sub == "BANK117" else fth
            best = e.min(1)
            bi = e.argmin(1)
            ar = np.arange(N)
            sg = sw * wrap(fsub[ar, bi] - gth)
            ncand = (np.isfinite(e).sum(1)) if sub == "SURV" else np.full(N, e.shape[1])
            st[(k, sub)] = {"hit1": (best <= TAU_PRIMARY).astype(float), "hit2": (best <= TAU_SECOND).astype(float), "best_deg": np.degrees(np.where(np.isfinite(best), best, np.nan)),
                            "best_signed_turn_aligned_deg": np.degrees(sg), "frac_within": (e <= TAU_PRIMARY).sum(1) / np.maximum(ncand, 1)}
    mets = (("hit@0.1806 rad (10.3 deg)", "hit1"), ("hit@0.35 rad (20.1 deg)", "hit2"), ("mean best |err| (deg)", "best_deg"),
            ("mean best-candidate signed turn-aligned err (deg; <0 = best candidate still under-rotated)", "best_signed_turn_aligned_deg"), ("frac of candidates within 0.1806 rad", "frac_within"))
    for lab in labs:
        fan["rows"][lab] = {}
        for sub in ("ALL", "BANK117", "SURV"):
            fan["rows"][lab][sub] = {}
            for mname, mk in mets:
                v8 = 0.5 * (st[((lab, 0), sub)][mk] + st[((lab, 1), sub)][mk])
                v7 = 0.5 * (st[(("r7", 0), sub)][mk] + st[(("r7", 1), sub)][mk])
                fin = wsel & np.isfinite(v8) & np.isfinite(v7)
                r8, b8_ = csub(inv, E, W, v8, fin)
                r7, b7_ = csub(inv, E, W, v7, fin)
                seeds = {}
                for s in SEEDS:
                    x8, x7 = st[((lab, s), sub)][mk], st[(("r7", s), sub)][mk]
                    f2 = wsel & np.isfinite(x8) & np.isfinite(x7)
                    q8, bq8 = csub(inv, E, W, x8, f2)
                    q7, bq7 = csub(inv, E, W, x7, f2)
                    seeds[f"s{s}"] = {"r8": q8, "r7": q7, "r8 - r7": fvs.bs_diff(q8["est"], bq8, q7["est"], bq7), "n_win": int(f2.sum())}
                fan["rows"][lab][sub][mname] = {"n_win": int(fin.sum()), "r8": r8, "r7": r7, "r8 - r7": fvs.bs_diff(r8["est"], b8_, r7["est"], b7_), "per_seed": seeds}
    R["fan_coverage"] = fan
    print(f"[{time.time() - t0:6.1f}s] fan coverage done (n sharp & valid GT chord = {int(wsel.sum())})", flush=True)

    # =========================================================================================== (4) FEASIBILITY (source facts, nothing run)
    R["n_alloc_feasibility"] = {"status": "FROM SOURCE ONLY; nothing executed (no GPU, no model build)", "see": "REFINE_SHRINK.md section 4"}
    R["free_commit_gb_start_end"] = [g0, fvs.guard("end")]
    R["runtime_s"] = time.time() - t0
    os.makedirs(a.out_dir, exist_ok=True)
    jp = os.path.join(a.out_dir, a.tag + ".json")
    json.dump(R, open(jp, "w", encoding="utf-8"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    open(os.path.join(a.out_dir, a.tag + ".md"), "w", encoding="utf-8").write(render_md(R))
    print("WROTE", jp, f"({R['runtime_s']:.0f}s)")


# ------------------------------------------------------------------------------------------------ markdown
def render_md(R: dict) -> str:
    L, A = [], None
    A = L.append
    A(f"# REFINE SHRINK -- does the refinement step shrink tight turns? (refcv8 vs refcv7)\n")
    A(f"- **Tier:** {R['tier']}\n- **Evidence class:** {R['evidence_class']}\n- **Estimator:** {R['estimator']}")
    A(f"- **n:** {R['n_windows']} S2 windows / {R['n_episodes']} episodes; sharp = {R['windows']['sharp (GT |yaw|>=0.15)']} windows; curved = {R['windows']['curved (>=0.02)']}. "
      "In-bank restriction (sel_idx < 117 in the view's seeds) shrinks the window set; n is printed per table.")
    A("- **Not answered:** another TRAINING run (one checkpoint per arm; H-ESTIM-SEED-1) and another INFERENCE run (seed rows only).\n")
    c = R["control_rule6_reproduction"]
    A("## 0. Controls (asserted before anything else)\n")
    A("**Rule 6 -- banked numbers reproduced through the same Ctx slope code, against `FAN_VS_SELECTOR.json`:**\n")
    A("| tree | slope r8 os | slope r8 oracle_sel | slope r7 os | slope r7 oracle_sel | n_seg | max abs diff (est, lo, hi vs banked) |\n|---|---|---|---|---|---|---|")
    for lab, row in c["reproduced_vs_FAN_VS_SELECTOR_json"].items():
        A(f"| {lab} | {row['slope r8 os']['est']:+.6f} | {row['slope r8 oracle_sel']['est']:+.6f} | {row['slope r7 os']['est']:+.6f} | {row['slope r7 oracle_sel']['est']:+.6f} | {row['n_seg']} | {row['max_abs_diff_all']:.1e} |")
    A(f"\nLiteral task values: EMA-15k os -0.1409, oracle r8 -0.083, r7 -0.048 -> **PASS = {c['PASS']}**.\n")
    A("**Frame / pairing controls (MEASURED):**\n")
    for k, v in R["controls"].items():
        A(f"- {k}: `{v if not isinstance(v, float) else format(v, '.6g')}`")
    A("\n## 1. Refinement effect: unrefined selected anchor (a) vs refined emitted plan (b)\n")
    A("Negative slope = the path turns LESS than GT. effect = b - a (negative = refinement shrinks the turn). `GAP_refined = GAP_unrefined + DIFF_EFFECT` is an exact identity (asserted on all bootstrap draws).\n")
    for lab, dd in R["refinement"].items():
        A(f"### {lab} vs refcv7-50,400\n")
        for view, d in dd.items():
            A(f"**{view}** -- n_win {d['n_win']}, n_seg {d['n_seg']}, n_ep {d['n_ep']}, mean GT turn-aligned heading {d['mean_gt_turn_aligned_heading_deg']:.1f} deg; identity err {d['identity_max_abs_err_all_boot_draws']:.1e}\n")
            A("| term | est [95 % CI] |\n|---|---|")
            for k, v in d["terms"].items():
                A(f"| {k} | {fvs.cellmd(v)} |")
            for nm in ("share_DIFF_EFFECT_of_GAP_refined", "share_GAP_unrefined_of_GAP_refined"):
                s_ = d[nm]
                A(f"| {nm.replace('_', ' ')} | {s_['est']:.2f} [{s_['lo']:.2f}, {s_['hi']:.2f}] |")
            for kk, vv in d["recovery"].items():
                A(f"| {kk} (DERIVED) | {fvs.cellmd(vv, 3)} |")
            A("")
    A("\n### 1a. Same-anchor split (sharp, last segment): where r8 and r7 select the SAME bank index their unrefined paths are bit-identical (GAP_unrefined = 0 asserted), so DIFF_EFFECT there is the refinement FUNCTION on identical prior paths\n")
    for lab, dd in R["refinement_by_anchor_agreement"].items():
        for view, d in dd.items():
            A(f"**{lab} | {view}** -- n_win {d['n_win']}, n_ep {d['n_ep']}\n\n| term | est [95 % CI] |\n|---|---|")
            for k, v in d["terms"].items():
                A(f"| {k} | {fvs.cellmd(v)} |")
            for kk, vv in d["recovery"].items():
                A(f"| {kk} (DERIVED) | {fvs.cellmd(vv, 3)} |")
            A("")
    A("\n### 1b. Same, on the 5.0->6.0 s terminal chord (SECONDARY; large angles)\n")
    t6 = R["refinement_6s_terminal_chord"]
    A(f"_{t6['_definition']}_\n")
    for k, d in t6.items():
        if k.startswith("_"):
            continue
        A(f"**{k}** -- n_win {d['n_win']}, n_ep {d['n_ep']}, mean GT turn-aligned terminal heading {d['mean_gt_turn_aligned_heading_deg']:.1f} deg\n\n| term | est [95 % CI] |\n|---|---|")
        for kk, v in d["terms"].items():
            A(f"| {kk} | {fvs.cellmd(v)} |")
        A("")
    f = R["fan_coverage"]
    A("\n## 3. Fan-level coverage on sharp turns (terminal 5-6 s heading of every refined candidate vs the GT terminal heading)\n")
    A(f"{f['_definition']}\n\nn sharp {f['n_sharp_windows']}; with valid 6 s GT chord (>= {MIN_CHORD_M} m, not end-clamped) **{f['n_sharp_windows_with_valid_6s_GT_chord']} windows / {f['n_episodes']} episodes**. "
      "Values are 2-seed means of per-window indicators; per-seed rows follow.\n")
    for lab, d1 in f["rows"].items():
        A(f"### {lab} vs refcv7\n")
        for sub, d2 in d1.items():
            A(f"**subset {sub}**\n\n| metric | n | r8 | r7 | r8 - r7 | seed 0 r8 - r7 | seed 1 r8 - r7 |\n|---|---|---|---|---|---|---|")
            for mname, r in d2.items():
                A(f"| {mname} | {r['n_win']} | {fvs.cellmd(r['r8'], 3)} | {fvs.cellmd(r['r7'], 3)} | {fvs.cellmd(r['r8 - r7'], 3)} | {fvs.cellmd(r['per_seed']['s0']['r8 - r7'], 3)} | {fvs.cellmd(r['per_seed']['s1']['r8 - r7'], 3)} |")
            A("")
    A(f"\nruntime {R['runtime_s']:.0f} s; free commit at start/end {R['free_commit_gb_start_end'][0]:.2f}/{R['free_commit_gb_start_end'][1]:.2f} GB.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
