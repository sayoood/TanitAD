"""Dev-box side of P0: evaluate the PRE-REGISTERED parity gates (colab/NAVTEST_ON_COLAB_PLAN.md section f) on what
the Colab VMs produced, against the dev box's own snapshot-016 sub200 artifacts. Thresholds are the plan's, written
before any VM result; nothing here tunes them.

    C:/Users/Admin/venvs/tanitad/Scripts/python.exe colab/navtest/compare_p0.py \
        --runs <evidence>/navtest-p0-cpu/results <evidence>/navtest-p0-l4/results --out <evidence>/gates_p0.json

`--runs` are unpacked p0_results.tar.gz trees (each holds out/...); a gate is read from the FIRST run that has its
inputs, and every run that has them is compared (so a gate reproduced on two VMs is reported twice). A gate with no
inputs is NOT_RUN with the reason -- never PASS by absence.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

DATA = "D:/Projects/TanitAD/data/refe_navtest"
NAME = "sub200_ep016"
DEV_CSV = f"{DATA}/score/refe_{NAME}/refe_{NAME}.csv"
DEV_SEAM = f"{DATA}/seams/refe_{NAME}.npz"
DEV_DUMP = f"{DATA}/proptable/{NAME}/proposals.npz"
DEV_TABLE = f"{DATA}/proptable/{NAME}/table.npz"
W3RAW = "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw"
TOKENS = f"{W3RAW}/A1_sub200_tokens.json"
TERMS = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound",
         "comfort", "driving_direction_compliance", "score")
DISCRETE = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound", "comfort",
            "driving_direction_compliance")
REF_PDMS = 76.5243


def rows(p):
    return {r["token"]: r for r in csv.DictReader(open(p, encoding="utf-8")) if r["token"] != "average"}


def cmp_csv(a_p, b_p, toks=None):
    a, b = rows(a_p), rows(b_p)
    toks = sorted(toks if toks is not None else a)
    missing = [t for t in toks if t not in a or t not in b]
    toks = [t for t in toks if t in a and t in b]
    d = {t: max(abs(float(a[t][k]) - float(b[t][k])) for k in TERMS) for t in toks}
    textual = sum(1 for t in toks for k in TERMS if a[t][k] == b[t][k])
    return {"n": len(toks), "missing": len(missing), "max_abs_diff": max(d.values()) if d else None,
            "tokens_with_any_diff": sum(1 for v in d.values() if v > 0), "cells_textually_equal": textual,
            "cells": len(toks) * len(TERMS), "valid_a": sum(1 for t in toks if a[t]["valid"] == "True"),
            "mean_a_x100": round(100 * float(np.mean([float(a[t]["score"]) for t in toks])), 4) if toks else None,
            "mean_b_x100": round(100 * float(np.mean([float(b[t]["score"]) for t in toks])), 4) if toks else None}


def counts(p):
    try:
        c = json.load(open(p, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return {k: c.get(k) for k in ("status", "log_successful", "log_failed", "csv_valid_rows", "C1_max_abs_delta",
                                  "C3_missing", "C3_unexpected", "summary_x100_4dp", "wall_s", "failures",
                                  "imported_from", "patches", "agent_calls")}


def aggregate(lg):
    s = 1.0 / (1.0 + np.exp(-lg.astype(np.float32)))
    return s[..., 0] * s[..., 1] * (5 * s[..., 2] + 5 * s[..., 3] + 2 * s[..., 4]) / np.float32(12.0)


def first(runs, rel):
    return [os.path.join(r, rel) for r in runs if os.path.exists(os.path.join(r, rel))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hybrid-csv", default=None, help="the VM seam scored by the DEV BOX harness (extra readout)")
    a = ap.parse_args()
    G = {}
    toks = json.load(open(TOKENS, encoding="utf-8"))["tokens"]

    # ---- G-ENV (per VM that ran it)
    for r in a.runs:
        st_p = os.path.join(r, "out/status.json")
        if not os.path.exists(st_p):
            continue
        st = json.load(open(st_p))
        pull = (st.get("relay") or {}).get("pull") or {}
        env = {"relay_pull_all_md5_ok": bool(pull.get("files")) and all(f.get("ok") for f in pull["files"].values()),
               "relay_pull": {k: {x: v.get(x) for x in ("bytes", "seconds", "mb_s", "host", "ok")}
                              for k, v in (pull.get("files") or {}).items()}}
        if st.get("genv"):
            g = st["genv"]
            env.update(selftest=g.get("selftest"), jpeg_equal=g.get("jpeg_equal"), params=g.get("params"),
                       cuda=g.get("cuda"))
            env["pass"] = bool(env["relay_pull_all_md5_ok"] and g.get("selftest") and g.get("jpeg_equal") == "4/4"
                               and g.get("params") and g["params"][:2] == [322070854, 18991426]
                               and (g.get("cuda") or {}).get("conv_ok"))
        key = os.path.basename(r.rstrip("/\\"))
        if key == "results":
            key = os.path.basename(os.path.dirname(r.rstrip("/\\")))
        G.setdefault("G-ENV", {})[key] = env

    # ---- G-H: the dev box's seam + cache, scored on Linux
    for p in first(a.runs, f"out/gh/refe_{NAME}/refe_{NAME}.csv"):
        c = cmp_csv(p, DEV_CSV, toks)
        cn = counts(p.replace(".csv", ".counts.json"))
        c8 = (cn or {}).get("imported_from") or {}
        g = {"csv": p, "vs_devbox": c, "counts": cn,
             "C8_navsim_from_v11_tree": "navsim-3e8291b" in str(c8.get("navsim", "")),
             "C8_nuplan_from_ce3c323": "nuplan-devkit-ce3c323" in str(c8.get("nuplan", "")),
             "pdms_x100": (cn or {}).get("summary_x100_4dp", {}).get("PDMS")}
        g["pass"] = bool(c["max_abs_diff"] == 0.0 and c["n"] == 200 and c["valid_a"] == 200 and cn
                         and cn["status"] == "PASS" and g["pdms_x100"] == REF_PDMS)
        G.setdefault("G-H", []).append(g)

    # ---- G-H' (sub200 form): the VM-BUILT cache; seam CSV == G-H's, floors == W3's banked per-token rows
    for r in a.runs:
        base = os.path.join(r, "out/gh2")
        if not os.path.isdir(base):
            continue
        g = {"cache_counts": counts(os.path.join(base, "refe_cache_vm/refe_cache_vm.counts.json"))}
        sp = os.path.join(base, f"refe_{NAME}_vmcache/refe_{NAME}_vmcache.csv")
        g["seam_vs_devbox"] = cmp_csv(sp, DEV_CSV, toks) if os.path.exists(sp) else "MISSING"
        gh = first(a.runs, f"out/gh/refe_{NAME}/refe_{NAME}.csv")
        g["seam_vs_GH"] = cmp_csv(sp, gh[0], toks) if (gh and os.path.exists(sp)) else "MISSING"
        fl = {}
        for arm in ("CV", "STOP", "HUMAN"):
            fp = os.path.join(base, f"refe_{arm.lower()}_vmcache/refe_{arm.lower()}_vmcache.csv")
            fl[arm] = cmp_csv(fp, f"{W3RAW}/{arm}_navtest/{arm}_navtest.csv", toks) if os.path.exists(fp) else "MISSING"
        g["floors_vs_W3_banked_rows_on_sub200"] = fl
        g["pass_sub200"] = bool(isinstance(g["seam_vs_devbox"], dict) and g["seam_vs_devbox"]["max_abs_diff"] == 0.0
                                and g["seam_vs_devbox"]["n"] == 200
                                and all(isinstance(v, dict) and v["max_abs_diff"] == 0.0 and v["n"] == 200
                                        for v in fl.values()))
        g["scope_note"] = ("section f defines G-H' on ALL 12,146 navtest tokens (final only); this is its sub200 "
                           "form -- the VM cache rebuilt for the 200 tokens, the seam CSV and the three floors' "
                           "per-token rows compared on those 200. The full-set half stays with P2.")
        G.setdefault("G-H'", []).append(g)

    # ---- G-E6: the 64 single-proposal runs from the dev box's dump
    ref = np.load(DEV_TABLE)
    for p in first(a.runs, "out/ge6/table.npz"):
        t = np.load(p)
        same_tok = [str(x) for x in t["token"]] == [str(x) for x in ref["token"]]
        g = {"table": p, "tokens_equal": same_tok, "shape": list(t["pdms"].shape),
             "pdms_max_abs_diff": float(np.nanmax(np.abs(t["pdms"] - ref["pdms"]))) if same_tok else None,
             "sub_max_abs_diff": float(np.nanmax(np.abs(t["sub"] - ref["sub"]))) if same_tok else None,
             "nan_cells_vm": int(np.isnan(t["pdms"]).sum()), "valid_all": bool(t["valid"].all()),
             "pick_equal": bool(np.array_equal(t["pick"], ref["pick"]))}
        gj = os.path.join(os.path.dirname(p), "gates.json")
        if os.path.exists(gj):
            gg = json.load(open(gj))
            g["vm_gates"] = {k: {x: v for x, v in gg["gates"][k].items() if x != "run_seconds"} for k in gg["gates"]}
            g["vm_seconds"] = gg.get("seconds")
            rs = gg["gates"].get("G3", {}).get("run_seconds") or {}
            if rs:
                v = sorted(float(x) for x in rs.values())
                g["per_run_seconds"] = {"min": v[0], "median": v[len(v) // 2], "max": v[-1]}
        g["pass"] = bool(same_tok and g["pdms_max_abs_diff"] == 0.0 and g["sub_max_abs_diff"] == 0.0
                         and g["nan_cells_vm"] == 0)
        G.setdefault("G-E6", []).append(g)

    # ---- G-S: the VM GPU seam + dump against the dev box's
    S0, D0 = np.load(DEV_SEAM), np.load(DEV_DUMP)
    agg0 = aggregate(D0["logits"])
    srt = np.sort(agg0, -1)
    margin0 = (srt[:, -1] - srt[:, -2]).astype(np.float64)
    for r in a.runs:
        sp, dp = os.path.join(r, f"out/seam/refe_{NAME}_vm.npz"), os.path.join(r, "out/seam/proposals_vm.npz")
        if not (os.path.exists(sp) and os.path.exists(dp)):
            continue
        S1, D1 = np.load(sp), np.load(dp)
        rep = json.load(open(sp.replace(".npz", ".report.json")))
        tok_eq = [str(x) for x in S1["token"]] == [str(x) for x in S0["token"]]
        pick0, pick1 = D0["pick"], D1["pick"]
        flip = pick0 != pick1
        same = ~flip
        dpose = np.abs(S1["poses"].astype(np.float64) - S0["poses"].astype(np.float64))      # [N, 8, 3] executed
        dprop = np.abs(D1["proposals"].astype(np.float64) - D0["proposals"].astype(np.float64))
        agg1 = aggregate(D1["logits"])
        dagg = np.abs(agg1.astype(np.float64) - agg0.astype(np.float64))
        dlog = np.abs(D1["logits"].astype(np.float64) - D0["logits"].astype(np.float64))
        big = margin0 >= 1e-3
        g = {"seam": sp, "rows": int(S1["token"].shape[0]), "tokens_equal_order": tok_eq,
             "frame_control_max_m": rep["frame_control"]["max_m"], "misses": rep["misses"],
             "device_seconds": rep.get("seconds"),
             "n_tokens_margin_ge_1e-3": int(big.sum()), "flips_total": int(flip.sum()),
             "flips_at_margin_ge_1e-3": int((flip & big).sum()),
             "flipped_tokens": [{"token": str(S0["token"][i]), "margin_devbox": float(margin0[i]),
                                 "pick_devbox": int(pick0[i]), "pick_vm": int(pick1[i])} for i in np.where(flip)[0]],
             "same_pick_max_abs_pose_xy_m": float(dpose[same][..., :2].max()) if same.any() else None,
             "same_pick_max_abs_pose_heading_rad": float(dpose[same][..., 2].max()) if same.any() else None,
             "same_pick_max_abs_aggregate_all_proposals": float(dagg[same].max()) if same.any() else None,
             "same_pick_max_abs_pose_all_proposals": float(dprop[same].max()) if same.any() else None,
             "max_abs_logit": float(dlog.max()), "bitwise_equal_poses": bool(np.array_equal(S1["poses"], S0["poses"])),
             "tokens_bitwise_equal_executed": int(sum(np.array_equal(S1["poses"][i], S0["poses"][i])
                                                      for i in range(min(len(S1["poses"]), len(S0["poses"])))))}
        g["pass"] = bool(tok_eq and g["rows"] == 200 and g["frame_control_max_m"] <= 1e-3
                         and g["flips_at_margin_ge_1e-3"] == 0
                         and (g["same_pick_max_abs_pose_xy_m"] or 0) <= 1e-3
                         and (g["same_pick_max_abs_pose_heading_rad"] or 0) <= 1e-3
                         and (g["same_pick_max_abs_aggregate_all_proposals"] or 0) <= 1e-3)
        g["bar_reading"] = ("pose bar on the EXECUTED rows (the seam; the plan's 0.53 mm scale -> 1 mm = ~2x), "
                            "aggregate bar over ALL 64 proposals (the plan's 1.1e-4 scale -> 1e-3 = ~9x); the "
                            "all-proposal pose max is reported, not gated")
        G.setdefault("G-S", []).append(g)

        # ---- G-P: the harness on the VM's own seam
        cp = os.path.join(r, f"out/gp/refe_{NAME}_vm/refe_{NAME}_vm.csv")
        if os.path.exists(cp):
            a_, b_ = rows(cp), rows(DEV_CSV)
            fl_t = {str(S0["token"][i]) for i in np.where(flip)[0]}
            nonflip = [t for t in toks if t not in fl_t]
            disc_eq = all(a_[t][k] == b_[t][k] for t in nonflip for k in DISCRETE)
            ds = {t: float(a_[t]["score"]) - float(b_[t]["score"]) for t in toks}
            cn = counts(cp.replace(".csv", ".counts.json"))
            pd = (cn or {}).get("summary_x100_4dp", {}).get("PDMS")
            gp = {"csv": cp, "counts": cn, "pdms_x100": pd, "delta_pdms_x100": None if pd is None else round(pd - REF_PDMS, 4),
                  "nonflip_discrete_equal": disc_eq,
                  "nonflip_max_abs_score_diff": max(abs(ds[t]) for t in nonflip) if nonflip else None,
                  "flip_contribution_x100": round(100 * sum(ds[t] for t in fl_t) / len(toks), 6),
                  "nonflip_contribution_x100": round(100 * sum(ds[t] for t in nonflip) / len(toks), 6)}
            if not fl_t:
                gp["pass"] = bool(pd is not None and abs(pd - REF_PDMS) <= 0.01 and disc_eq and cn["status"] == "PASS")
            else:
                gp["pass"] = bool(disc_eq and cn and cn["status"] == "PASS"
                                  and abs(gp["nonflip_contribution_x100"]) <= 0.01)
            G.setdefault("G-P", []).append(gp)
        # ---- G-REP: K shards vs 1 process on the same VM
        st = json.load(open(os.path.join(r, "out/status.json")))
        if st.get("rep"):
            rr = dict(st["rep"])
            kp = [f for f in os.listdir(os.path.join(r, "out/rep")) if f.startswith(f"refe_{NAME}_k") and f.endswith(".npz")] \
                if os.path.isdir(os.path.join(r, "out/rep")) else []
            if kp:
                K = np.load(os.path.join(r, "out/rep", kp[0]))
                KD = np.load(os.path.join(r, "out/rep", kp[0].replace(f"refe_{NAME}_k", "proposals_k")))
                rr["devbox_recheck"] = {"file": kp[0], "tokens_equal": [str(x) for x in K["token"]] == [str(x) for x in S1["token"]],
                                        "poses_bitwise": bool(np.array_equal(K["poses"], S1["poses"])),
                                        "proposals_bitwise": bool(np.array_equal(KD["proposals"], D1["proposals"])),
                                        "logits_bitwise": bool(np.array_equal(KD["logits"], D1["logits"])),
                                        "pick_equal": bool(np.array_equal(KD["pick"], D1["pick"]))}
                rr["pass"] = all(bool(v) for k, v in rr["devbox_recheck"].items() if k != "file")
            G.setdefault("G-REP", []).append(rr)
    # ---- NOT a pre-registered gate: the hybrid lever (the VM's GPU seam scored by the DEV BOX's own harness)
    if a.hybrid_csv and os.path.exists(a.hybrid_csv):
        A_, B_ = rows(a.hybrid_csv), rows(DEV_CSV)
        cn = counts(a.hybrid_csv.replace(".csv", ".counts.json"))
        G["hybrid_devbox_harness_on_vm_seam"] = {
            "csv": a.hybrid_csv, "counts": cn, "vs_devbox": cmp_csv(a.hybrid_csv, DEV_CSV, toks),
            "discrete_equal_all_tokens": all(A_[t][k] == B_[t][k] for t in toks for k in DISCRETE),
            "note": "an ADDITIONAL readout, not a section-f gate: the Windows harness that produced every banked "
                    "point, applied to the Colab L4 seam"}
    for k in ("G-ENV", "G-H", "G-H'", "G-E6", "G-S", "G-P", "G-REP"):
        G.setdefault(k, "NOT_RUN (no inputs in the given runs)")
    json.dump(G, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    for k, v in G.items():
        vv = v if isinstance(v, (list, str)) else list(v.values())
        print(k, [x.get("pass", x.get("pass_sub200")) if isinstance(x, dict) else x for x in vv]
              if not isinstance(vv, str) else vv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
