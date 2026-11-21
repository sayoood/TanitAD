"""nav_brake_r8.py -- H-R9-NAV-BRAKE-1: does the clip-level nav command drive refcv8's cruise false-brake?

    cd <PKG>/code && CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 \
      PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' \
      python -X utf8 nav_brake_r8.py        # -> <PKG>/raw/refcv9_headroom/NAV_BRAKE.{json,md}

CPU only, banked data only (no model forward). Prereg: raw/refcv9_headroom/PREREG_NAV_BRAKE.md (sha256-sealed; refuses if changed).
Intervention: the battery banked refcv8's plan on the SAME windows under four nav conditions (nav_true / nav_zero / nav_flipped /
nav_shuffled); dFD(cond) = false-decel(nav_true) - false-decel(cond) on GT-cruise windows, paired episode-cluster bootstrap.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PKG = HERE.parent
OUT = PKG / "raw" / "refcv9_headroom"
DEP = PKG / "raw" / "retime_deploy"
ARCHIVE = "C:/Users/Admin/refcv8_eval_archive/battery"
TREES = ("step40000ema_unc", "step30000ema_unc")      # first = the label's tree; second printed beside
SEEDS = (0, 1)
DT, THR = 1.0, 1.5                                     # = H-R9-BRAKE-LOCUS-1 / H-R9-L2-HEADROOM-1
S6_FROM_FULL8 = [1, 3, 4, 5, 6, 7]                     # slots 1,2,3,4,5,6 s inside 0.5,1,1.5,2,3,4,5,6 s
STRAIGHT_M = 2.0
CONDS = ("nav_zero", "nav_flipped", "nav_shuffled")
NAV = {0: "FOLLOW", 1: "TURN_L", 2: "TURN_R"}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(tree: str, seed: int) -> dict:
    pdir = f"{ARCHIVE}/{tree}/s6_s{seed}"
    cols: dict = {}
    files = sorted(glob.glob(pdir + "/ep*.npz"))
    if not files:
        raise SystemExit(f"[nav_brake] no panel files under {pdir}")
    for f in files:
        z, d = np.load(f), np.load(f"{pdir}/decisions/{os.path.basename(f)}")
        if not np.array_equal(z["ws"], d["ws"]):
            raise SystemExit(f"[nav_brake] {f}: panel ws != decisions ws")
        rec = {"os": z["os"], "g": z["g"], "v0": z["v0"].astype(np.float64), "b_r7": z[f"b_refcv7_50400_s{seed}"],
               "nav_cmd": d["nav_cmd"].astype(np.int64), "nav_valid": d["nav_valid"].astype(bool)}
        for c in ("nav_true",) + CONDS:
            if "plan_full_" + c in d.files:
                rec["pf_" + c] = d["plan_full_" + c]
        for k, v in rec.items():
            cols.setdefault(k, []).append(v)
    if any(len(v) != len(files) for v in cols.values()):
        raise SystemExit(f"[nav_brake] {tree} s{seed}: a key is present on only some episodes")
    T = {k: np.concatenate(v) for k, v in cols.items()}
    T["has_conds"] = all("pf_" + c in T for c in CONDS)
    zp = DEP / f"DEPLOY8_POS_{tree}_s{seed}.npz"
    zz = np.load(zp, allow_pickle=True)
    T["comp"] = np.asarray(zz["S6_compliant"], bool)
    T["eid"] = [str(x) for x in zz["S6_eid"]]
    if len(T["eid"]) != len(T["v0"]) or len(T["comp"]) != len(T["v0"]):
        raise SystemExit(f"[nav_brake] {tree} s{seed}: DEPLOY8_POS rows {len(T['eid'])} != panel rows {len(T['v0'])}")
    return T


def measure(PR, CB, T: dict) -> dict:
    v0, G = T["v0"], T["g"]
    gtc = CB.plan_class_thr(G, v0, DT, THR)
    m = T["comp"] & (gtc == "cruise")
    turn, follow = np.isin(T["nav_cmd"], (1, 2)), T["nav_cmd"] == 0
    straight = np.abs(np.asarray(G, np.float64)[:, -1, 1]) < STRAIGHT_M
    conds = CONDS if T["has_conds"] else ()
    dec = {c: (CB.plan_class_thr(T["pf_" + c][:, S6_FROM_FULL8], v0, DT, THR) == "decel").astype(np.float64)
           for c in ("nav_true",) + conds}
    dec7 = (CB.plan_class_thr(T["b_r7"], v0, DT, THR) == "decel").astype(np.float64)
    eid = T["eid"]

    def w(mask, x):
        return np.where(mask, x, np.nan)

    def pc(mask, a, b):
        r = PR.paired_cell(w(mask, a), w(mask, b), eid)
        return {k: (round(float(r[k]), 4) if isinstance(r.get(k), (float, np.floating)) else r.get(k))
                for k in ("delta", "lo", "hi", "mean_new", "mean_ref", "n_windows", "n_episodes") if k in r}
    out = {"n_gt_cruise": int(m.sum()), "n_turn_cruise": int((m & turn).sum()), "n_follow_cruise": int((m & follow).sum()),
           "nav_valid_share_on_cruise": float(T["nav_valid"][m].mean())}
    out["dFD"] = {c: {"TURN": pc(m & turn, dec["nav_true"], dec[c]), "FOLLOW": pc(m & follow, dec["nav_true"], dec[c]),
                      "ALL": pc(m, dec["nav_true"], dec[c])} for c in conds}
    out["has_nav_condition_plans"] = bool(T["has_conds"])
    # descriptive: r8 vs r7 per stratum
    desc = {}
    for nm, sm in (("TURN", turn), ("FOLLOW", follow)):
        for sn, ss in (("STRAIGHT6", straight), ("CURVED6", ~straight)):
            mm = m & sm & ss
            desc[f"{nm}|{sn}"] = {"n": int(mm.sum()), "fd_r8": float(dec["nav_true"][mm].mean()) if mm.any() else None,
                                  "fd_r7": float(dec7[mm].mean()) if mm.any() else None,
                                  "fd_r8_nav_zero": float(dec["nav_zero"][mm].mean()) if (mm.any() and "nav_zero" in dec) else None,
                                  "r8_minus_r7": pc(mm, dec["nav_true"], dec7) if mm.sum() >= 20 else None}
    out["strata"] = desc
    fb = m & (dec["nav_true"] > 0)
    out["turn_share_of_r8_false_brake"] = float(turn[fb].mean()) if fb.any() else None
    out["turn_share_of_gt_cruise"] = float(turn[m].mean())
    out["nav_zero_releases_share_of_r8_false_brake"] = ({"TURN": float((dec["nav_zero"][fb & turn] == 0).mean()) if (fb & turn).any() else None,
                                                         "FOLLOW": float((dec["nav_zero"][fb & follow] == 0).mean()) if (fb & follow).any() else None}
                                                        if "nav_zero" in dec else None)
    return out


def controls(T: dict) -> dict:
    os_ = np.asarray(T["os"], np.float64)
    pf = np.asarray(T["pf_nav_true"], np.float64)[:, S6_FROM_FULL8]
    err = float(np.max(np.abs(os_ - pf)))
    navs_differ = {c: float(np.mean(np.any(np.abs(np.asarray(T["pf_" + c], np.float64) - np.asarray(T["pf_nav_true"], np.float64)) > 1e-6, axis=(1, 2))))
                   for c in CONDS if "pf_" + c in T}
    return {"C1_plan_full_nav_true_eq_os_max_abs_m": err, "C1_pass": err <= 1e-4,
            "C2_share_windows_where_cond_plan_differs_from_nav_true": navs_differ,
            "C2_pass": all(v > 0 for v in navs_differ.values()), "has_nav_condition_plans": bool(T["has_conds"])}


def label(cells: list) -> str:
    """AMENDMENT 1: cells = [40k-s0 dFD(nav_zero), 30k-s0 dFD(nav_zero)] (two checkpoints, inference seed 0)."""
    z = cells
    sep = [c.get("lo") is not None and c["lo"] > 0 for c in (zz["TURN"] for zz in z)]
    bigger = [zz["TURN"]["delta"] > zz["FOLLOW"]["delta"] for zz in z]
    if all(sep) and all(bigger):
        return "NAV-DRIVES-BRAKE"
    if not any(sep):
        return "NAV-NOT-THE-LEVER"
    return "MIXED"


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[nav_brake] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    pre, pre_sha = OUT / "PREREG_NAV_BRAKE.md", OUT / "PREREG_NAV_BRAKE.sha256"
    if pre_sha.read_text(encoding="utf-8").strip() != sha(pre):
        raise SystemExit("[nav_brake] REFUSED: PREREG_NAV_BRAKE.md changed after sealing")
    import pair_read_r8 as PR
    import cruise_brake_retime_r8 as CB
    t0 = time.time()
    rep = {"tool": "nav_brake_r8.py", "prereg_sha256": sha(pre), "script_sha256": sha(__file__), "tier": "T1 open loop, banked plans, no forward",
           "estimator": "paired episode-cluster bootstrap B 2000 (pair_read_r8.paired_cell); answers another draw of EPISODES only",
           "trees": {}}
    for tree in TREES:
        res, ctl = {}, {}
        for s in SEEDS:
            T = load(tree, s)
            ctl[s] = controls(T)
            res[s] = measure(PR, CB, T)
        all_pass = all(c["C1_pass"] and c["C2_pass"] for c in ctl.values())
        rep["trees"][tree] = {"controls": ctl, "controls_all_pass": all_pass, "per_seed": res}
        print(f"ZZNAVBRAKEZZ {tree} controls_all_pass={all_pass}", flush=True)
    am, am_sha = OUT / "PREREG_NAV_BRAKE_AMENDMENT1.md", OUT / "PREREG_NAV_BRAKE_AMENDMENT1.sha256"
    if am_sha.read_text(encoding="utf-8").strip() != sha(am):
        raise SystemExit("[nav_brake] REFUSED: AMENDMENT1 changed after sealing")
    rep["amendment1_sha256"] = sha(am)
    ok = all(rep["trees"][t]["controls_all_pass"] for t in TREES)
    rep["label_cells"] = "40k-s0 and 30k-s0 dFD(nav_zero) (AMENDMENT 1; SINGLE-INFERENCE-SEED)"
    rep["label"] = (label([rep["trees"][t]["per_seed"][0]["dFD"]["nav_zero"] for t in TREES]) + " (SINGLE-INFERENCE-SEED)") if ok else "VOID (controls)"
    for t in TREES:
        rep["trees"][t]["label"] = rep["label"]
    rep["wall_s"] = round(time.time() - t0, 1)
    (OUT / "NAV_BRAKE.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    L = [f"# NAV_BRAKE — H-R9-NAV-BRAKE-1: **{rep['label']}** (tree {TREES[0]}; prereg sha256 {rep['prereg_sha256'][:12]}…)", "",
         f"tier: {rep['tier']}; estimator: {rep['estimator']}; GT-cruise = S6 compliant windows with GT class cruise (thr 1.5 m/s, dt 1 s)", ""]
    for tree in TREES:
        t = rep["trees"][tree]
        L.append(f"## {tree} — label {t['label']} (controls {'ALL_PASS' if t['controls_all_pass'] else 'FAIL'})")
        for s in SEEDS:
            r = t["per_seed"][s]
            L.append(f"* seed {s}: GT-cruise {r['n_gt_cruise']} (TURN {r['n_turn_cruise']}, FOLLOW {r['n_follow_cruise']}); "
                     f"TURN share of r8 false-brake {r['turn_share_of_r8_false_brake']:.3f} vs of GT-cruise {r['turn_share_of_gt_cruise']:.3f}; "
                     f"nav_zero releases {r['nav_zero_releases_share_of_r8_false_brake']}")
            for c in r["dFD"]:
                d = r["dFD"][c]
                L.append(f"  * dFD({c}) = FD(nav_true) − FD({c}): " + " | ".join(
                    f"{k} {d[k].get('delta', float('nan')):+.4f} [{d[k].get('lo', float('nan')):+.4f}, {d[k].get('hi', float('nan')):+.4f}] n {d[k].get('n_windows')}"
                    for k in ("TURN", "FOLLOW", "ALL")))
            for k, v in r["strata"].items():
                rr = v["r8_minus_r7"]
                L.append(f"  * {k}: n {v['n']}, FD r8 {v['fd_r8']}, r7 {v['fd_r7']}, r8 nav_zero {v['fd_r8_nav_zero']}"
                         + (f"; r8 − r7 {rr['delta']:+.4f} [{rr['lo']:+.4f}, {rr['hi']:+.4f}]" if rr else ""))
        L.append("")
    (OUT / "NAV_BRAKE.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"ZZNAVBRAKEDONEZZ label={rep['label']} wall {rep['wall_s']} s", flush=True)
    return 0 if all(rep["trees"][t]["controls_all_pass"] for t in TREES) else 2


if __name__ == "__main__":
    sys.exit(main())
