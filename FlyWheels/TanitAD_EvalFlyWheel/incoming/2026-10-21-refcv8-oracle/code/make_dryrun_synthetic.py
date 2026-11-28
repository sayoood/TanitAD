"""make_dryrun_synthetic.py -- an END-TO-END KNOWN-VALUE test of r8oracle_analyze.py's full (non-smoke) path, with NO model and NO GPU.

SYNTHETIC, NOT A RESULT.  Builds <out>/dump_s{0,1} (copies of the BANKED battery dumps) and <out>/oracle_s{0,1}.npz in which
  OWN          = the banked plan_full_nav_true (so C1/C2/C7 read 0 against the banked tree);
  GT_BOTH      = OWN + 0.50 (G - OWN), GT_CONS = OWN + 0.45 (G - OWN), GT_HYP = OWN + 0.05 (G - OWN) on every slot where the banked
                 panels give G (S2 slots 0-3, S6 slots 1,3-7), so per-window ADE scales EXACTLY: share_CONS = 0.9, share_HYP = 0.1;
  GT_SPEED_ONLY = OWN + 0.30 (G - OWN) (share 0.6), OWN_ARGMAX = OWN, SWAP arms = GT_HYP / GT_BOTH, SELF_NAN = OWN on clip % 10 == 0;
  spy / GT fields consistent, so C3 / C3b / C4 / C6 pass.
Expected: verdict CONSTRAINT-DOMINANT on S6 and S2, shares 0.9000 / 0.1000 exactly at both seeds; the beside speed rule True.
"""
import glob
import json
import os
import shutil
import sys

import numpy as np

BANK = "C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf"
DEP = "C:/Users/Admin/refcv8_pf/wise_ship/step48852ema_pf/pfpkg/raw/retime_deploy/DEPLOY8_POS_step48852ema_pf_s{seed}.npz"
OUT = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/refcv8_pf/r8oracle/dryrun_SYNTHETIC"
ARMS = ("GT_CONS", "GT_HYP", "GT_BOTH", "GT_SPEED_ONLY", "OWN_ARGMAX", "GT_HYP_SWAP", "GT_BOTH_SWAP", "SELF_NAN")
FRAC = {"GT_BOTH": 0.50, "GT_CONS": 0.45, "GT_HYP": 0.05, "GT_SPEED_ONLY": 0.30, "OWN_ARGMAX": 0.0, "GT_HYP_SWAP": 0.05,
        "GT_BOTH_SWAP": 0.50, "SELF_NAN": 0.0}
os.makedirs(OUT, exist_ok=True)
for s in ("0", "1"):
    dd = f"{OUT}/dump_s{s}"
    if not os.path.isdir(dd):
        shutil.copytree(f"{BANK}/dump_s{s}", dd)
    g8 = {}
    for S, slots, pdir in (("S2", [0, 1, 2, 3], f"panel_s{s}"), ("S6", [1, 3, 4, 5, 6, 7], f"s6_s{s}")):
        for f in sorted(glob.glob(f"{BANK}/{pdir}/ep*.npz")):
            z = np.load(f)
            ci = int(z["clip_index"][0])
            for j, w in enumerate(z["ws"]):
                g = g8.setdefault((ci, int(w)), np.full((8, 2), np.nan, np.float64))
                g[slots] = z["g"][j]
    dep = np.load(DEP.format(seed=s), allow_pickle=True)
    prior = {(int(c), int(w)): dep["S2_prior8"][i] for i, (c, w) in enumerate(zip(dep["S2_clip"], dep["S2_ws"]))}
    recs = []
    for f in sorted(glob.glob(f"{dd}/ep*.npz")):
        z = np.load(f)
        d = np.load(f"{dd}/decisions/{os.path.basename(f)}")
        ci = int(z["clip_index"][0])
        for j, w in enumerate(d["ws"]):
            recs.append((ci, int(w), d["plan_full_nav_true"][j].astype(np.float64), int(d["sel_idx_nav_true"][j])))
    n = len(recs)
    rng = np.random.default_rng(int(s))
    out = {"clip_index": np.array([r[0] for r in recs]), "ws": np.array([r[1] for r in recs]),
           "gt_gl": rng.integers(0, 3, n), "gt_go": rng.integers(0, 6, n)}
    cons = rng.uniform(-0.2, 0.9, (n, 4)).astype(np.float32)
    cons[:, 1][rng.random(n) < 0.8] = np.nan
    out.update(gt_cons_force=cons, gt_cons_raw=np.nan_to_num(cons), gt_valid=np.isfinite(cons))
    own_hl, own_ho = rng.integers(0, 3, n), rng.integers(0, 6, n)
    for arm in ("OWN",) + ARMS:
        fr = 0.0 if arm == "OWN" else FRAC[arm]
        T = []
        for (ci, w, P, sel) in recs:
            G = g8.get((ci, w))
            Q = P.copy()
            if G is not None:
                m = np.isfinite(G[:, 0])
                Q[m] = P[m] + fr * (G[m] - P[m])
            T.append(Q)
        has = np.ones(n, bool) if arm != "SELF_NAN" else (out["clip_index"] % 10 == 0)
        if arm == "GT_HYP_SWAP":                         # C3b: identity where own == GT
            has = np.ones(n, bool)
        out[f"has_{arm}"] = has
        out[f"traj_{arm}"] = np.stack(T).astype(np.float32)
        out[f"sel_{arm}"] = np.array([r[3] for r in recs])
        out[f"forced_{arm}"] = has.copy() if arm != "OWN" else np.zeros(n, bool)
        forced_class = arm in ("GT_HYP", "GT_BOTH", "GT_HYP_SWAP", "GT_BOTH_SWAP")
        out[f"spy_hl_{arm}"] = out["gt_gl"].copy() if forced_class else own_hl.copy()
        out[f"spy_ho_{arm}"] = out["gt_go"].copy() if forced_class else own_ho.copy()
        sc = rng.uniform(-0.2, 0.9, (n, 4)).astype(np.float32)
        if arm in ("GT_CONS", "GT_BOTH", "GT_BOTH_SWAP"):
            sc = np.where(np.isfinite(cons), cons, sc)
        if arm == "GT_SPEED_ONLY":
            sc[:, 3] = np.where(np.isfinite(cons[:, 3]), cons[:, 3], sc[:, 3])
        out[f"spy_cons_{arm}"] = sc
        out[f"spy_fd_{arm}"] = np.full(n, arm in ("GT_CONS", "GT_BOTH", "GT_SPEED_ONLY", "GT_BOTH_SWAP", "SELF_NAN"))
        out[f"spy_fmin_{arm}"] = np.full(n, 2 if out[f"spy_fd_{arm}"][0] else -1)
        out[f"prior_maxdiff_{arm}"] = np.zeros(n)
        out[f"wall_{arm}"] = np.zeros(n)
    # C3b: make the swap arm differ from OWN only where own != GT (here: it equals GT_HYP's synthetic plan; set own==GT nowhere)
    out["spy_hl_OWN"] = (out["gt_gl"] + 1) % 3
    out["prior_OWN"] = np.stack([prior.get((r[0], r[1]), np.full((8, 2), np.nan)) for r in recs]).astype(np.float32)
    meta = {"tool": "make_dryrun_synthetic.py", "SYNTHETIC": True, "final": True, "n_windows": n, "arms": list(ARMS), "wall_by_arm_s": None}
    out["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(f"{OUT}/oracle_s{s}.npz", **out)
    print(f"seed {s}: {n} windows -> {OUT}/oracle_s{s}.npz")
