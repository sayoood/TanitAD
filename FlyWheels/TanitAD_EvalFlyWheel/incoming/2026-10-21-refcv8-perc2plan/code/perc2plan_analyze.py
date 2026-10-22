#!/usr/bin/env python3
"""perc2plan_analyze.py -- PERC2PLAN-R8 analysis, read MECHANICALLY per PREREG_PERC2PLAN.md (sha256 d1a2eb53..., incl.
AMENDMENT 1).  Every arm X vs BASE at the SAME inference seed, S2 (dt 0.5) and S6 (dt 1.0, slots [1,3,4,5,6,7]),
components = refav1_arm._components + pair-masked curvature / yaw-rate (as pai_gate_families.py), interval =
refav1_arm._paired_families (paired episode-cluster bootstrap, B 2000, 139 episodes).

    python perc2plan_analyze.py --root <dir with <ARM>_s<seed>/dump> --seeds 0,1 --out-json PERC2PLAN.json \
        [--banked C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf] [--fan-dump <thor fan-roll s1 dump>] \
        [--ref-dump <unmodified-child dump, C1>] [--n-boot 2000]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

LT = os.environ.get("P2P_LAUNCH_TREE", "C:/lgs/Lfb47a58376")
for p in (f"{LT}/stack", f"{LT}/taniteval", f"{LT}/taniteval/tools"):
    if p not in sys.path:
        sys.path.append(p)
ARMS = ("BASE", "AGT_ZERO", "AGT_SHUF", "BEV_ZERO", "BEV_SHUF", "BOTH_ZERO")
S6_SLOTS = [1, 3, 4, 5, 6, 7]
SURF_DT = {"S2": 0.5, "S6": 1.0}
PREREG_SHA = "d1a2eb53d4488dde404b4c30b84ddb9d35bfa0aead8564fa07d729d4e2b68568"


def load_dump(d):
    """Per-window arrays in the dump's own episode-file order."""
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    if not fs:
        raise SystemExit(f"no ep*.npz in {d}")
    cat = {}
    for f in fs:
        z = np.load(f)
        dz = np.load(os.path.join(d, "decisions", os.path.basename(f)))
        n = int(z["ws"].shape[0])
        ci = int(z["clip_index"][0])
        for k, v in (("os", z["os"]), ("g", z["g"]), ("ws", z["ws"]), ("ci", np.full(n, ci, np.int64)),
                     ("plan_full", dz["plan_full_nav_true"]), ("fv60", dz["gt_future_valid_ext"][:, 59] > 0.5),
                     ("sel_idx", dz["sel_idx"]), ("pose_last", dz["pose_last"]), ("gt_future_ext", dz["gt_future_ext"])):
            cat.setdefault(k, []).append(np.asarray(v))
    out = {k: np.concatenate(v) for k, v in cat.items()}
    ex = os.path.join(d, "refcv6_extras.npz")
    if os.path.exists(ex):
        e = np.load(ex, allow_pickle=True)
        out["extras"] = {k: np.asarray(e[k]) for k in e.files if k.startswith("nav_true.tacv6_") or k in ("ws", "clip_index")}
    return out


def key_index(D):
    return {(int(c), int(w)): i for i, (c, w) in enumerate(zip(D["ci"], D["ws"]))}


def s6_view(D, banked_s6_g):
    keep = D["fv60"]
    P = D["plan_full"][keep][:, S6_SLOTS].astype(np.float64)
    keys = list(zip(D["ci"][keep].tolist(), D["ws"][keep].tolist()))
    G = np.stack([banked_s6_g[k] for k in keys]).astype(np.float64)
    return P, G, keep


def load_banked_s6_g(banked, seed):
    m = {}
    for f in sorted(glob.glob(os.path.join(banked, f"s6_s{seed}", "ep*.npz"))):
        z = np.load(f)
        ci = int(z["clip_index"][0])
        for w, g in zip(z["ws"].tolist(), z["g"]):
            m[(ci, int(w))] = np.asarray(g)
    return m


def components(ra, ff, torch, P, G, dt):
    c = ra._components(P, G, dt)
    Pg, Gg = ff._seq_geometry(torch.as_tensor(P).float(), dt), ff._seq_geometry(torch.as_tensor(G).float(), dt)
    bp = Pg["pair_valid"] & Gg["pair_valid"]
    nv = bp.sum(1)
    for nm, key in (("LAT_curvature_mae_1pm", "curvature"), ("LAT_yaw_rate_mae_masked_radps", "yaw_rate")):
        c[nm] = torch.where(nv > 0, ((Pg[key] - Gg[key]).abs() * bp).sum(1) / nv.clamp_min(1), torch.nan).numpy()
    return c


def worse_list(block):
    w, b = [], []
    for fam, blk in block["families"].items():
        for m, r in blk.items():
            if not isinstance(r, dict) or not r.get("separated"):
                continue
            agree = m.startswith("TAC_")
            worse = (agree and r["delta"] < 0) or (not agree and r["delta"] > 0)
            (w if worse else b).append(f"{fam}.{m}")
    return w, b


def disp(P, Q):
    d = np.linalg.norm(P.astype(np.float64) - Q.astype(np.float64), axis=-1).mean(1)
    return {"mean_m": round(float(d.mean()), 4), "p95_m": round(float(np.percentile(d, 95)), 4),
            "max_m": round(float(d.max()), 4), "frac_windows_moved": round(float((d > 0).mean()), 4)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--banked", default="C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf")
    ap.add_argument("--fan-dump", default=None, help="C3: 's1=<dir>,s0=<dir>' -- the fan-roll agent's same-device os-only dumps")
    ap.add_argument("--ref-dump", default=None, help="C1: the unmodified child's dump (smoke)")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:      # noqa: BLE001
        pass
    import torch
    import refav1_arm as ra
    from taniteval import four_families as ff
    ra._FAMILY_OF = {**ra._FAMILY_OF, "LAT_curvature_mae_1pm": "lateral", "LAT_yaw_rate_mae_masked_radps": "lateral"}
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    arms = [x for x in a.arms.split(",") if x]
    res = {"id": "PERC2PLAN-R8", "prereg_sha256": PREREG_SHA,
           "tier": "T1 self-action OPEN LOOP on logged eval139 frames (the SHIP-R8 tier); ONE training seed",
           "interval": "paired episode-cluster bootstrap (refav1_arm._paired_families), B %d -- answers 'another draw of "
                       "EPISODES' only; inference-run variance is read from the second inference seed (floor)" % a.n_boot,
           "strategic_family": {"status": "NOT COMPUTABLE by this instrument", "n": 0,
                                "reason": "_paired_families carries no strategic component; the nav-obedience block exists "
                                          "only in the full-arm-set acceptance stage, not on an os-only roll"},
           "launch_tree": LT, "root": a.root, "seeds": {}, "controls": {}, "floor": {}, "verdicts": {}}
    D, comps, eids, plans = {}, {}, {}, {}
    for s in seeds:
        g6 = load_banked_s6_g(a.banked, s)
        ctl = res["controls"].setdefault(f"s{s}", {})
        for x in arms:
            d = os.path.join(a.root, f"{x}_s{s}", "dump")
            if not os.path.exists(os.path.join(d, "manifest.json")):
                ctl[f"present_{x}"] = False
                continue
            D[(x, s)] = load_dump(d)
            ctl[f"present_{x}"] = True
        if ("BASE", s) not in D:
            continue
        B = D[("BASE", s)]
        for x in arms:
            if (x, s) not in D:
                continue
            X = D[(x, s)]
            # C5 alignment
            ctl[f"C5_aligned_{x}"] = bool(np.array_equal(X["ci"], B["ci"]) and np.array_equal(X["ws"], B["ws"])
                                          and np.array_equal(X["g"], B["g"]) and np.array_equal(X["fv60"], B["fv60"]))
            for surf in ("S2", "S6"):
                if surf == "S2":
                    P, G = X["os"].astype(np.float64), X["g"].astype(np.float64)
                    eid = X["ci"]
                else:
                    P, G, keep = s6_view(X, g6)
                    eid = X["ci"][keep]
                comps[(x, s, surf)] = components(ra, ff, torch, P, G, SURF_DT[surf])
                eids[(s, surf)] = list(eid)
                plans[(x, s, surf)] = P
                if x == "BASE" and surf == "S6":
                    # independent S6 GT: the roll's own banked future through refb_labels (alignment, not a number)
                    try:
                        import refb_labels
                        hz = [5, 10, 15, 20, 30, 40, 50, 60]
                        gi = refb_labels.waypoint_targets(torch.as_tensor(X["pose_last"][keep]).float(),
                                                          torch.as_tensor(X["gt_future_ext"][keep]).float(), hz)[:, S6_SLOTS].numpy()
                        ctl["S6_gt_independent_max_abs_m"] = float(np.abs(gi - G).max())
                    except Exception as e:      # noqa: BLE001
                        ctl["S6_gt_independent_max_abs_m"] = f"unavailable: {e!r}"[:200]
        # C1 (smoke): BASE == the unmodified child, bitwise
        if a.ref_dump:
            R = load_dump(a.ref_dump)
            ctl["C1_identity_os_bitexact"] = bool(np.array_equal(R["os"], B["os"]) and np.array_equal(R["ci"], B["ci"])
                                                 and np.array_equal(R["ws"], B["ws"]))
            ctl["C1_identity_plan_full_bitexact"] = bool(np.array_equal(R["plan_full"], B["plan_full"]))
            ctl["C1_max_abs_os_diff"] = float(np.abs(R["os"].astype(np.float64) - B["os"]).max())
        # C2: banked battery os at the same seed (match by (clip_index, ws))
        bd = os.path.join(a.banked, f"dump_s{s}")
        if os.path.exists(os.path.join(bd, "manifest.json")):
            K = load_dump(bd)
            ki = key_index(K)
            idx = [ki.get((int(c), int(w))) for c, w in zip(B["ci"], B["ws"])]
            if all(i is not None for i in idx):
                Kos = K["os"][idx].astype(np.float64)
                dd = np.abs(Kos - B["os"].astype(np.float64))
                per = dd.reshape(len(dd), -1).max(1)
                ctl["C2_vs_banked"] = {"binding": s == 1, "n_windows": len(idx), "max_abs_m": float(dd.max()),
                                       "frac_windows_bitexact": round(float((per == 0).mean()), 4),
                                       "frac_windows_le_1e-5": round(float((per <= 1e-5).mean()), 4),
                                       "pass": bool(dd.max() <= 1e-5),
                                       "g_equal": bool(np.array_equal(K["g"][idx], B["g"]))}
                if len(idx) == len(K["ws"]):
                    comps[("BANKED", s, "S2")] = components(ra, ff, torch, Kos, B["g"].astype(np.float64), 0.5)
            else:
                ctl["C2_vs_banked"] = {"pass": False, "reason": "window keys missing in banked dump"}
        # C3: same-device fan-roll os-only dump at the same seed
        fdm = dict(kv.split("=", 1) for kv in (a.fan_dump or "").split(",") if "=" in kv)
        fd = fdm.get(f"s{s}")
        if fd and os.path.exists(os.path.join(fd, "manifest.json")):
            F = load_dump(fd)
            ok = np.array_equal(F["ci"], B["ci"]) and np.array_equal(F["ws"], B["ws"])
            ctl["C3_vs_fanroll"] = {"aligned": bool(ok),
                                    "os_bitexact": bool(ok and np.array_equal(F["os"], B["os"])),
                                    "plan_full_bitexact": bool(ok and np.array_equal(F["plan_full"], B["plan_full"])),
                                    "max_abs_m": (float(np.abs(F["os"].astype(np.float64) - B["os"]).max()) if ok else None)}
        else:
            ctl["C3_vs_fanroll"] = {"status": "ABSENT", "fan_dump": fd}
        # C4 from each roll's meta
        for x in arms:
            mj = os.path.join(a.root, f"{x}_s{s}", f"meta_s{s}.json")
            if os.path.exists(mj):
                m = json.load(open(mj, encoding="utf-8"))
                ctl[f"C4_{x}"] = m.get("C4", {}).get("all_pass")
                ctl[f"C4_{x}_donor_same_clip"] = m.get("counters", {}).get("donor_same_clip")
                if "SHUF" in x:
                    ctl[f"C4_{x}_donor_same_clip_zero"] = m.get("counters", {}).get("donor_same_clip") == 0
        # per-seed blocks
        blk = {"n_windows_S2": len(eids[(s, "S2")]), "n_windows_S6": len(eids[(s, "S6")]),
               "n_episodes": len(set(eids[(s, "S2")])), "means": {}, "pairs": {}, "displacement": {}, "sel_idx_changed": {},
               "tacv6_argmax_changed": {}}
        for surf in ("S2", "S6"):
            tiers = {x: "T1" for x in arms}
            tiers["BASE_self"] = "T1"
            cc = {x: comps[(x, s, surf)] for x in arms if (x, s, surf) in comps}
            blk["means"][surf] = {x: {m: round(float(np.nanmean(v)), 6) for m, v in cc[x].items()} for x in cc}
            cc["BASE_self"] = cc["BASE"]
            sp = ra._paired_families(cc, "BASE", "BASE_self", eids[(s, surf)], tiers, 200, 0)
            ctl[f"C6_self_pair_zero_{surf}"] = all(float(r["delta"]) == 0.0 for fam in sp["families"].values()
                                                   for r in fam.values() if isinstance(r, dict) and "delta" in r)
            for x in arms:
                if x == "BASE" or x not in cc:
                    continue
                blk["pairs"][f"{surf}|{x}"] = ra._paired_families(cc, "BASE", x, eids[(s, surf)], tiers, a.n_boot, 0)
                blk["displacement"][f"{surf}|{x}"] = disp(plans[(x, s, surf)], plans[("BASE", s, surf)])
            if ("BANKED", s, surf) in comps:
                cb = {"BANKED": comps[("BANKED", s, surf)], "BASE": cc["BASE"]}
                blk["pairs"][f"{surf}|BASE_minus_BANKED"] = ra._paired_families(
                    cb, "BANKED", "BASE", eids[(s, surf)], {"BANKED": "T1", "BASE": "T1"}, a.n_boot, 0)
        for x in arms:
            if x == "BASE" or (x, s) not in D:
                continue
            blk["sel_idx_changed"][x] = round(float((D[(x, s)]["sel_idx"] != B["sel_idx"]).mean()), 4)
            ex, eb = D[(x, s)].get("extras"), B.get("extras")
            if ex and eb:
                for hk in ("nav_true.tacv6_lat_logits", "nav_true.tacv6_lon_logits", "nav_true.tacv6_goal_logits"):
                    if hk in ex and hk in eb:
                        if hk.endswith("goal_logits"):
                            blk["tacv6_argmax_changed"][f"{x}|{hk}|max_abs_logit_diff"] = float(np.abs(ex[hk] - eb[hk]).max())
                        else:
                            blk["tacv6_argmax_changed"][f"{x}|{hk}"] = round(float((ex[hk].argmax(-1) != eb[hk].argmax(-1)).mean()), 4)
        res["seeds"][f"s{s}"] = blk
    # ---- inference-seed floor + verdicts (PREREG s4) ---------------------------------------------------------------
    have2 = len(seeds) == 2 and all(("BASE", s, "S2") in comps for s in seeds)
    if have2:
        s0, s1 = seeds
        for surf in ("S2", "S6"):
            if eids[(s0, surf)] != eids[(s1, surf)]:
                res["floor"][surf] = {"status": "windows differ across seeds"}
                continue
            cc = {"BASE_s0": comps[("BASE", s0, surf)], "BASE_s1": comps[("BASE", s1, surf)]}
            fb = ra._paired_families(cc, "BASE_s0", "BASE_s1", eids[(s0, surf)], {"BASE_s0": "T1", "BASE_s1": "T1"},
                                     a.n_boot, 0)
            res["floor"][surf] = {"ade_block": fb["families"]["ADE"]["ade_m"],
                                  "floor_m": abs(float(fb["families"]["ADE"]["ade_m"]["delta"])),
                                  "seed_displacement": disp(plans[("BASE", s1, surf)], plans[("BASE", s0, surf)])}
    for x in arms:
        if x == "BASE":
            continue
        per = {}
        for surf in ("S2", "S6"):
            rows = {}
            for s in seeds:
                p = res["seeds"].get(f"s{s}", {}).get("pairs", {}).get(f"{surf}|{x}")
                if p is None:
                    continue
                w, b = worse_list(p)
                ade = p["families"]["ADE"]["ade_m"]
                rows[f"s{s}"] = {"ade_delta": ade["delta"], "ade_lo": ade["lo"], "ade_hi": ade["hi"],
                                 "ade_separated": bool(ade.get("separated")), "separated_worse": w, "separated_better": b}
            if not have2 or len(rows) != 2 or "floor_m" not in res["floor"].get(surf, {}):
                per[surf] = {"verdict": "INCOMPLETE (needs both seeds + floor)", "rows": rows}
                continue
            fl = res["floor"][surf]["floor_m"]
            used = all(r["ade_separated"] and r["ade_delta"] > 0 and r["ade_delta"] > fl for r in rows.values())
            worse_both = sorted(set.intersection(*[set(r["separated_worse"]) for r in rows.values()]))
            better_both = sorted(set.intersection(*[set(r["separated_better"]) for r in rows.values()]))
            v = "USED" if used else ("NOT USED" if not worse_both else "PARTIAL")
            per[surf] = {"verdict": v, "floor_m": fl, "separated_worse_both_seeds": worse_both,
                         "separated_better_both_seeds": better_both, "rows": rows}
        vs = [per[s_]["verdict"] for s_ in ("S2", "S6") if s_ in per]
        if any(v.startswith("INCOMPLETE") for v in vs) or len(vs) < 2:
            head = "INCOMPLETE"
        elif "USED" in vs:
            head = "USED (on %s)" % ",".join(s_ for s_ in ("S2", "S6") if per[s_]["verdict"] == "USED")
        elif all(v == "NOT USED" for v in vs):
            bb = sorted(set(per["S2"]["separated_better_both_seeds"]) | set(per["S6"]["separated_better_both_seeds"]))
            head = "NOT USED" + (f" (+ separated-BETTER: {bb})" if bb else "")
        else:
            head = "PARTIAL"
        res["verdicts"][x] = {"headline": head, "per_surface": per}
    flat = {k: v for c in res["controls"].values() if isinstance(c, dict) for k, v in c.items()}
    void = {k: v for k, v in flat.items() if k.startswith(("C1_identity", "C4_", "C5_", "C6_")) and isinstance(v, bool)}
    for sk, cs in res["controls"].items():
        if isinstance(cs, dict) and isinstance(cs.get("C3_vs_fanroll"), dict) and "os_bitexact" in cs["C3_vs_fanroll"]:
            void[f"C3_{sk}_os_bitexact"] = bool(cs["C3_vs_fanroll"]["os_bitexact"])
    res["controls"]["void_gates_all_pass"] = all(void.values()) if void else None
    res["controls"]["void_gates_failed"] = sorted(k for k, v in void.items() if not v)
    c2 = res["controls"].get("s1", {}).get("C2_vs_banked", {})
    res["controls"]["C2_stamp"] = (None if not c2 else ("PASS" if c2.get("pass") else
                                   "C2-FAIL (cross-device reproduction not established)"))
    json.dump(res, open(a.out_json, "w", encoding="utf-8"), indent=1, default=str)
    print("QQP2P " + json.dumps({"verdicts": {k: v["headline"] for k, v in res["verdicts"].items()},
                                 "void_pass": res["controls"]["void_gates_all_pass"],
                                 "void_failed": res["controls"]["void_gates_failed"], "C2": res["controls"]["C2_stamp"]}) + " QQ")


if __name__ == "__main__":
    main()
