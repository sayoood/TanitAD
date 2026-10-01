#!/usr/bin/env python3
"""Measure 5 -- what each arm's PICK does, per metric family (REPORTED, never gating; eval/PREREG_MEASURE5.md §4-§5,
blob c668b5b6, and the binding four-families rule). Runs after `m5_finetune_eval.py eval`, CPU only.

  PART 1  paired, per token: the harness sub-scores of each model's pick -- NC, DAC, EP, TTC, C, DDC and PDMS (x100) --
          on the two routes the prereg names: E3b = the argmax among the 64 alone (the shipped route) and E2 = the
          argmax among the 64 + 9 extras (masked route). Every sub-score is the NAVSIM harness's own, already banked:
          the E-6 table for the 64, the f075_r00..r07 / stopzeros csvs for the extras. Arm value per token = its seed
          mean; pairs V4-V3 (the prereg's contrast), V3-A0, V4-A0, V4all-V4; the SAME estimator as the readout
          (paired log-cluster bootstrap over the 93 logs, 10,000, 95 %, seed 20260927 -- m5_finetune_eval's draws).
          Plus the E2 selection mix (original / 0.75x copy / STOP) -- a tactical read of what the selector chooses.
  PART 2  families6.py (the eval pipeline's step 4, TANITAD venv, unchanged) on a seam of each model's pick, both routes
          -> longitudinal / lateral / tactical / strategic blocks against the logged human future (an imitation view;
          per-model blocks with families6's own intervals, NOT paired). Amendment 6's family guard is applied to each
          arm's E2 pick vs the shipped pick (A0, E3b == the table's pick == the landed seam, asserted bit for bit).
  PART 3  POST HOC, NOT GATING: E1 / E3a / E3b / E2 re-read against the REPAIRED harness truth (Amendment 7, adopted
          after the registered truth table was scored): the banked repaired E-6 table for the 64 + the repaired copies
          scored by eval/m5_repaired_extras.py + STOP (the repair is the identity on it). Same draws, same pairs.

    python eval/m5_families.py [--no-families6]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m5_finetune_eval as FE  # noqa: E402

SEAM0 = "D:/Projects/TanitAD/data/refe_navtest/seams/refe_sub200_ep015.npz"
HEADS = ("NC", "DAC", "EP", "TTC", "C", "DDC")
GUARD = {"progress_ratio_mean_min": 0.90, "speed_mae_up_max": 0.30, "cross_mae_up_max": 0.10,
         "heading_mae_up_max": 0.5, "goal_point_error_up_max": 1.0}


def picks(ev: dict, n: int):
    ar = np.arange(n)
    k64 = FE.v1_agg(ev["pure"]).argmax(1)
    agg_m = FE.v1_agg(ev["masked"])
    k73 = np.nan_to_num(agg_m, nan=-1).argmax(1)
    return ar, k64, k73


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m5", default=FE.M5)
    ap.add_argument("--no-families6", action="store_true")
    a = ap.parse_args()
    evd = os.path.join(a.m5, "evals")
    ev = {os.path.basename(p)[:-4]: dict(np.load(p)) for p in glob.glob(os.path.join(evd, "*.npz"))
          if not os.path.basename(p).startswith("_")}
    HX = np.load(os.path.join(evd, "_harness_extra.npz"))
    toks = [str(t) for t in ev["A0"]["token"]]
    n = len(toks)
    T = np.load(FE.TABLE)
    ti = {str(t): i for i, t in enumerate(T["token"])}
    ri = np.array([ti[t] for t in toks])
    if [str(x) for x in T["sub_names"]] != list(FE.SUB):
        raise SystemExit("table sub-score order != the csv order")
    sub64 = T["sub"][ri].astype(np.float64)                             # [n, 64, 6]
    subx = HX["sub_extra"].astype(np.float64)                           # [n, 9, 6]
    sub_all = np.concatenate([sub64, subx], 1)                          # [n, 73, 6]
    pd_all = HX["pdms"].astype(np.float64)                              # [n, 73] (64 from the table + 9 extras)
    if not np.array_equal(pd_all[:, :64], T["pdms"][ri].astype(np.float64)):
        raise SystemExit("_harness_extra pdms[:, :64] != the table's")
    draws, n_logs = FE.cluster_draws(toks)
    seeds = sorted({int(k.split("_s")[1]) for k in ev if "_s" in k})

    per = {}                                                            # model -> route -> [n, 7] (6 subs + PDMS) x100
    mix = {}
    kp = {}
    for name, e in ev.items():
        ar, k64, k73 = picks(e, n)
        kp[name] = (k64, k73)
        per[name] = {"E3b": np.concatenate([sub64[ar, k64], pd_all[ar, k64][:, None]], 1) * 100.0,
                     "E2": np.concatenate([sub_all[ar, k73], pd_all[ar, k73][:, None]], 1) * 100.0}
        mix[name] = {"original": int((k73 < 64).sum()), "copy_075": int(((k73 >= 64) & (k73 < 72)).sum()),
                     "stop": int((k73 == 72).sum())}

    def arm(name, route):
        if name == "A0":
            return per["A0"][route]
        return np.mean(np.stack([per[f"{name}_s{s}"][route] for s in seeds if f"{name}_s{s}" in per]), 0)

    cols = HEADS + ("PDMS",)
    out = {"_label": "REPORTED, never gating (PREREG_MEASURE5 §5). Harness sub-scores x100 of each arm's pick; arm = "
                     "seed mean per token; paired log-cluster bootstrap over the logs (the readout's draws)",
           "n_tokens": n, "n_logs": n_logs, "seeds": seeds, "families_map": {
               "longitudinal": "EP (ego progress), TTC; + families6 speed/progress/along-track",
               "lateral": "DAC, DDC; + families6 heading/cross-track/curvature/yaw-rate",
               "tactical": "E1/E3a pair concordance (the readout), the E2 selection mix; + families6 decision kappas "
                           "and goal-point error",
               "strategic": "UNAVAILABLE: NAVSIM carries no route/goal decision label and a scorer-only fine-tune does "
                            "not set the route (families6 reports it unavailable by design)"},
           "arm_means": {}, "pairs": {}, "E2_selection_mix": mix}
    for route in ("E3b", "E2"):
        out["arm_means"][route] = {nm: {c: round(float(np.nanmean(arm(nm, route)[:, j])), 3) for j, c in enumerate(cols)}
                                   for nm in ("A0", "V3", "V4", "V4all")}
        for pair, (x, y) in {"V4_minus_V3": ("V4", "V3"), "V3_minus_A0": ("V3", "A0"), "V4_minus_A0": ("V4", "A0"),
                             "V4all_minus_V4": ("V4all", "V4")}.items():
            d = arm(x, route) - arm(y, route)
            res = {}
            for j, c in enumerate(cols):
                lo, hi = FE.boot_ci(d[:, j], draws)
                res[c] = {"diff": round(float(np.nanmean(d[:, j])), 3), "ci95": [round(lo, 3), round(hi, 3)],
                          "separated": bool(lo > 0 or hi < 0)}
            out["pairs"].setdefault(route, {})[pair] = res
    # seed floor per sub-score (the largest |difference| between two seeds of the same arm, token means)
    fl = {}
    for route in ("E3b", "E2"):
        for j, c in enumerate(cols):
            v = 0.0
            for nm in ("V3", "V4"):
                tm = [float(np.nanmean(per[f"{nm}_s{s}"][route][:, j])) for s in seeds if f"{nm}_s{s}" in per]
                v = max([v] + [abs(tm[i] - tm[k]) for i in range(len(tm)) for k in range(i + 1, len(tm))])
            fl.setdefault(route, {})[c] = round(v, 3)
    out["seed_floor"] = fl

    # ------------------------------------------------------------------ PART 2: families6 on seams of the picks
    if not a.no_families6:
        import eval_checkpoint as EC
        import stop_candidate_probe as SCP
        S0 = np.load(SEAM0)
        if [str(t) for t in S0["token"]] != [str(t) for t in T["token"]]:
            raise SystemExit("the landed seam's token order differs from the table's")
        s0i = {str(t): i for i, t in enumerate(S0["token"])}
        si = np.array([s0i[t] for t in toks])
        B = np.load(FE.BUILD)
        RK = np.load(FE.RANKS)
        P = T["proposals"][ri].astype(np.float32)                           # [n, 64, 8, 3] NAVSIM grid
        X = np.zeros((n, 9, 8, 3), np.float32)
        for r in range(FE.N_SRC):
            X[:, r] = B["nav_f075"][ri, RK["top"][ri, r]]
        C8 = np.concatenate([P, X], 1)                                      # [n, 73, 8, 3]; slot 72 = STOP zeros
        ar = np.arange(n)
        fam_dir = os.path.join(a.m5, "families")
        os.makedirs(fam_dir, exist_ok=True)
        ctrl = float(np.abs(C8[ar, kp["A0"][0]].astype(np.float64) - S0["poses"][si].astype(np.float64)).max())
        fam = {"instrument": os.path.join(EC.EV6, "families6.py"), "python": EC.TANITAD_PY, "dir": fam_dir,
               "control_A0_E3b_pick_vs_landed_seam_max_abs_m": ctrl, "blocks": {}}
        if ctrl != 0.0:
            fam["status"] = "FAILED: A0's rebuilt pick differs from the landed seam"
        else:
            for name in sorted(ev):
                for route, k in (("E3b", kp[name][0]), ("E2", kp[name][1])):
                    tag = f"{name}_{route}"
                    sp, op, lg = (os.path.join(fam_dir, f"seam_{tag}.npz"), os.path.join(fam_dir, f"families_{tag}.json"),
                                  os.path.join(fam_dir, f"families_{tag}.log"))
                    if os.path.exists(op):
                        os.remove(op)
                    poses = S0["poses"].copy()
                    poses[si] = C8[ar, k]
                    np.savez(sp, token=S0["token"], fingerprint=S0["fingerprint"], poses=poses.astype(np.float32),
                             sampling=S0["sampling"], arm=np.array(f"REFe_m5_{tag}"))
                    rc, _ = EC.run([EC.TANITAD_PY, "families6.py", "--seam", sp, "--inputs", EC.EXPORT, "--stage", "1",
                                    "--label", f"REFe-m5-{tag}", "--out", op, "--n-boot", "2000"],
                                   EC.EV6, dict(os.environ, PYTHONIOENCODING="utf-8"), lg)
                    fam["blocks"][tag] = (SCP.summarize_families(op) if os.path.exists(op)
                                          else {"status": f"FAILED rc={rc}", "log": lg})
                    print(f"  families {tag}: {'OK' if os.path.exists(op) else 'FAILED'}", flush=True)

            def get(b, fam_, key):
                v = (b.get(fam_) or {}).get(key)
                return float(v) if isinstance(v, (int, float)) else np.nan

            keys = {"longitudinal": ("speed_mae_mps", "progress_ratio_mean", "along_mae_m", "under_progress_rate"),
                    "lateral": ("heading_mae_deg", "cross_mae_m", "curvature_mae_1pm", "yaw_rate_mae_degps"),
                    "tactical": ("goal_point_error_m", "longitudinal_decision_kappa", "lateral_decision_kappa")}
            agg = {}
            for route in ("E3b", "E2"):
                for nm in ("A0", "V3", "V4", "V4all"):
                    tags = [f"A0_{route}"] if nm == "A0" else [f"{nm}_s{s}_{route}" for s in seeds]
                    bl = [fam["blocks"][t] for t in tags if t in fam["blocks"] and "status" not in fam["blocks"][t]]
                    if not bl:
                        continue
                    agg.setdefault(route, {})[nm] = {f: {k: round(float(np.nanmean([get(b, f, k) for b in bl])), 4)
                                                         for k in ks} for f, ks in keys.items()}
                    agg[route][nm]["strategic"] = bl[0].get("strategic")
            fam["arm_seed_means"] = agg
            ship = agg.get("E3b", {}).get("A0")
            guard = {}
            for route in ("E2", "E3b"):
                for nm, g in agg.get(route, {}).items():
                    if ship is None or (route == "E3b" and nm == "A0"):
                        continue
                    chk = {"progress>=0.90": g["longitudinal"]["progress_ratio_mean"] >= GUARD["progress_ratio_mean_min"],
                           "speedMAE<=+0.30": g["longitudinal"]["speed_mae_mps"] - ship["longitudinal"]["speed_mae_mps"]
                           <= GUARD["speed_mae_up_max"],
                           "cross<=+0.10": g["lateral"]["cross_mae_m"] - ship["lateral"]["cross_mae_m"]
                           <= GUARD["cross_mae_up_max"],
                           "heading<=+0.5": g["lateral"]["heading_mae_deg"] - ship["lateral"]["heading_mae_deg"]
                           <= GUARD["heading_mae_up_max"],
                           "goal<=+1.0": g["tactical"]["goal_point_error_m"] - ship["tactical"]["goal_point_error_m"]
                           <= GUARD["goal_point_error_up_max"]}
                    guard[f"{nm}_{route}"] = {"pass": all(chk.values()), "fails": [k for k, v in chk.items() if not v]}
            fam["family_guard_vs_shipped"] = {"rule": "Amendment 6 (SPEC_NAVTEST.md): vs the shipped pick (A0 E3b) on the "
                                                      "same tokens, seed-mean blocks", "bounds": GUARD, "result": guard}
        out["families6"] = fam
    # ------------------------------------------------------------------ PART 3 (POST HOC, NOT GATING): repaired truth
    # The registered truth is the E-6 table (unrepaired, it predates Amendment 7). Re-read E1 / E3a / E3b / E2 against
    # the DEPLOYED planner's harness truth: the 64 originals from the banked repaired E-6 table (like_for_like_016), the
    # 8 copies from eval/m5_repaired_extras.py's harness runs, STOP unchanged (the repair is the identity on it).
    TR = "D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep015_repaired/table.npz"
    rep_csv = [f"{FE.NAV}/score/refe_sub200_ep015_f075rep_r{r:02d}/refe_sub200_ep015_f075rep_r{r:02d}.csv"
               for r in range(FE.N_SRC)]
    if os.path.exists(TR) and all(os.path.exists(p_) for p_ in rep_csv):
        Tr = np.load(TR)
        if [str(t) for t in Tr["token"]] != [str(t) for t in T["token"]] or not np.array_equal(Tr["logits"], T["logits"]):
            raise SystemExit("the repaired table is not the same snapshot/tokens as the E-6 table")
        hr = [FE.read_csv(p_) for p_ in rep_csv]
        stop = FE.read_csv(FE.STOP_CSV)
        pr = np.concatenate([Tr["pdms"][ri].astype(np.float64),
                             np.array([[hr[r][t][1] if hr[r].get(t) and hr[r][t][0] else np.nan for r in range(FE.N_SRC)]
                                       + [stop[t][1] if stop.get(t) and stop[t][0] else np.nan] for t in toks])], 1)
        extra = np.array([False] * 64 + [True] * 9)
        mr = {}
        for name, e in ev.items():
            agg_m, agg_p = FE.v1_agg(e["masked"]), FE.v1_agg(e["pure"])
            ar = np.arange(n)
            mr[name] = {"E1_masked": np.array([FE.concordance(agg_m[t], pr[t], extra) for t in ar]),
                        "E3a": np.array([FE.concordance(agg_p[t], pr[t, :64], None) for t in ar]),
                        "E3b": 100.0 * pr[ar, agg_p.argmax(1)],
                        "E2": 100.0 * pr[ar, np.nan_to_num(agg_m, nan=-1).argmax(1)]}

        def armr(nm, key):
            return mr["A0"][key] if nm == "A0" else np.nanmean(
                np.stack([mr[f"{nm}_s{s}"][key] for s in seeds if f"{nm}_s{s}" in mr]), 0)
        p3 = {"_label": "POST HOC, NOT GATING: the same metrics against the REPAIRED harness truth (Amendment 7, "
                        "adopted after the pre-registration's truth table was scored)",
              "truth": {"originals": TR, "copies": rep_csv, "stop": FE.STOP_CSV},
              "harness_missing": int(np.isnan(pr).sum()), "arm_means": {}, "pairs": {}}
        for key in ("E1_masked", "E3a", "E3b", "E2"):
            p3["arm_means"][key] = {nm: round(float(np.nanmean(armr(nm, key))), 5) for nm in ("A0", "V3", "V4", "V4all")}
            for pair, (x, y) in {"V4_minus_V3": ("V4", "V3"), "V3_minus_A0": ("V3", "A0"),
                                 "V4all_minus_V4": ("V4all", "V4")}.items():
                d = armr(x, key) - armr(y, key)
                lo, hi = FE.boot_ci(d, draws)
                p3["pairs"].setdefault(key, {})[pair] = {"diff": round(float(np.nanmean(d)), 5),
                                                         "ci95": [round(lo, 5), round(hi, 5)]}
        sf = {}
        for key in ("E1_masked", "E3a", "E3b", "E2"):
            v = 0.0
            for nm in ("V3", "V4"):
                tm = [float(np.nanmean(mr[f"{nm}_s{s_}"][key])) for s_ in seeds if f"{nm}_s{s_}" in mr]
                v = max([v] + [abs(tm[i] - tm[k]) for i in range(len(tm)) for k in range(i + 1, len(tm))])
            sf[key] = round(v, 5)
        p3["seed_floor"] = sf
        p3["seed_token_means"] = {f"{nm}_s{s_}": {key: round(float(np.nanmean(mr[f"{nm}_s{s_}"][key])), 5)
                                                  for key in ("E1_masked", "E3a", "E3b", "E2")}
                                  for nm in ("V3", "V4", "V4all") for s_ in seeds if f"{nm}_s{s_}" in mr}
        out["posthoc_repaired_truth"] = p3
    else:
        out["posthoc_repaired_truth"] = {"status": "not computed: the repaired table or the repaired copies' csvs are "
                                                   "missing (eval/m5_repaired_extras.py)"}
    p = os.path.join(FE.OUT if os.path.normcase(os.path.abspath(a.m5)) == os.path.normcase(os.path.abspath(FE.M5))
                     else os.path.join(a.m5, "readout"), "families.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    for route in ("E3b", "E2"):
        print(f"  {route} arm means: " + json.dumps(out["arm_means"][route]))
        print(f"  {route} V4-V3: " + json.dumps({c: out["pairs"][route]["V4_minus_V3"][c] for c in ("EP", "PDMS")}))
    print(f"ZZM5_FAMILIES_OK -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
