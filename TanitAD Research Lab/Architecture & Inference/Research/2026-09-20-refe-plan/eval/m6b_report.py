#!/usr/bin/env python3
"""M6b's full report: the FOUR metric families (binding, 2026-08-02), the manipulation check, the position guard, the
PDMS sub-scores and the declared deviations -- APPENDED to the RESULT_M6B_TANGENT.md that `proxy_eval.py analyze-m6b`
wrote (its verdict block is kept verbatim above the marker; nothing here changes the verdict).

It is a separate module ON PURPOSE: no arm and no decode imports it, so it can be written while the M6b chain runs
without touching a module a running stage has imported. Every number is read from an artifact:
result_m6b.json (analyze-m6b), families/<seam>.json (families6), score/<seam>/<seam>.csv (the NAVSIM harness),
fast_copy.json / chain.log / the arms' config.json, and the selftest records. Writes result_m6b_report.json.

    python eval/m6b_report.py --out raw/2026-09-28-m6b-tangent --arms-root raw/2026-09-28-m6b-tangent/arms
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE))
import proxy_eval as PE  # noqa: E402

MARK = "<!-- m6b_report: everything below is appended by eval/m6b_report.py -->"
SEEDS = ("0", "1", "2")
LON = ("speed_mae_mps", "speed_bias_mps", "along_mae_m", "along_final_bias_m", "accel_mae_mps2")
LAT = ("heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm", "cross_mae_m", "cross_final_mae_m")
TAC = ("lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed")


def jload(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms-root", required=True)
    a = ap.parse_args()
    out, arms = Path(a.out), Path(a.arms_root)
    r = jload(out / "result_m6b.json")
    if r is None:
        print("ZZM6B_REPORT_FAIL no result_m6b.json")
        return 1
    wt = {s: f"refe_m6_Wt{s}_on" for s in SEEDS}
    tt = {s: f"refe_m6_T{s}_off" for s in SEEDS}
    labels = [*wt.values(), *tt.values()]
    fam = {lb: jload(out / "families" / f"{lb}.json") for lb in labels}
    adverse = {s: (PE.fam_adverse(fam[wt[s]], fam[tt[s]]) if fam[wt[s]] and fam[tt[s]] else {"missing": True})
               for s in SEEDS}
    rep = {"families_adverse_separations": adverse, "at_local": time.strftime("%Y-%m-%dT%H:%M:%S")}
    L = [MARK, "", "## Manipulation check (each wrapped arm) and position guard (each seed)", "",
         "| arm | median winner step-19 heading error (rad) | raw step-19 \\|h\\| > pi (%) | stays >= 0.50 |",
         "|---|---|---|---|"]
    hm = r.get("metrics", {})
    for s in SEEDS:
        n = f"Wt{s}"
        v = hm.get(n, {})
        L.append(f"| {n} | {fmt(v.get('median_winner_h19_err_rad'))} | {fmt(v.get('pct_raw_h19_beyond_pi'))} | "
                 f"{'PASS' if (r.get('manipulation_ok') or {}).get(n) else 'FAIL'} |")
    L += ["", "| seed | T - Wt, winner ADE over the 8 NAVSIM poses (m) | <= +0.10 |", "|---|---|---|"]
    for s, v in (r.get("position_guard") or {}).items():
        L.append(f"| {s} | {v['T_minus_Wt_ade_m']:+.4f} | {'PASS' if v['ok'] else 'FAIL'} |")
    p = r.get("pdms", {})
    L += ["", "## PDMS: seams, sub-scores, between-seed spread", ""]
    if "means" in p:
        L.append(f"Estimator: {p.get('estimator')}; {p.get('n_tokens')} tokens. Between-seed SD (reported, never "
                 f"gating): of D {fmt(p.get('between_seed_sd_of_D'), 3)}, of Wt PDMS "
                 f"{fmt(p.get('between_seed_sd_Wt_pdms'), 3)}, of T PDMS {fmt(p.get('between_seed_sd_T_pdms'), 3)}.")
        L += ["", "| seam | PDMS x100 | " + " | ".join(PE.SUBSCORES) + " |", "|---|---|" + "---|" * len(PE.SUBSCORES)]
        subs = {}
        for lb in labels:
            cp = out / "score" / lb / f"{lb}.csv"
            if not cp.exists():
                L.append(f"| {lb} | MISSING | " + " | ".join("" for _ in PE.SUBSCORES) + " |")
                continue
            rows = PE.read_csv(cp)
            subs[lb] = {c: 100.0 * float(np.mean([float(x[c]) for x in rows.values()])) for c in PE.SUBSCORES
                        if c in next(iter(rows.values()))}
            L.append(f"| {lb} | {p['means'].get(lb, float('nan')):.3f} | " +
                     " | ".join(f"{subs[lb].get(c, float('nan')):.2f}" for c in PE.SUBSCORES) + " |")
        rep["subscores_x100"] = subs
    else:
        L.append(f"Gating seams missing: {p.get('missing')}")
    L += ["", "## Four families (families6 on the six gating seams; binding 2026-08-02)", "",
          "Tier: T1-family (stage-1 loop OPEN; the plan is the model's single query), NAVSIM navtest 1,123 tokens. "
          "Intervals: families6's episode-cluster bootstrap (2,000 resamples, 136 episodes).", ""]
    L += ["### LONGITUDINAL", "", "| seam | " + " | ".join(LON) + " | target speed within 0.5 / 1.0 / 2.0 m/s | "
          "progress ratio (mean) |", "|---|" + "---|" * (len(LON) + 2)]
    for lb in labels:
        f = ((fam.get(lb) or {}).get("families") or {}).get("refcv6", {}).get("longitudinal") or {}
        ts = f.get("target_speed_acc") or {}
        L.append(f"| {lb} | " + " | ".join(fmt(f.get(m)) for m in LON) +
                 f" | {fmt(ts.get('within_0.5_mps'))} / {fmt(ts.get('within_1.0_mps'))} / {fmt(ts.get('within_2.0_mps'))}"
                 f" | {fmt((f.get('ego_progress') or {}).get('progress_ratio_mean'))} |")
    L += ["", "### LATERAL", "", "| seam | " + " | ".join(LAT) + " |", "|---|" + "---|" * len(LAT)]
    for lb in labels:
        f = ((fam.get(lb) or {}).get("families") or {}).get("refcv6", {}).get("lateral") or {}
        L.append(f"| {lb} | " + " | ".join(fmt(f.get(m)) for m in LAT) + " |")
    L += ["", "### TACTICAL (trajectory-derived decisions; accuracy / kappa -- kappa is the readable number)", "",
          "| seam | " + " | ".join(TAC) + " | goal point error (m) | goal bearing MAE (deg) |", "|---|" + "---|" * (len(TAC) + 2)]
    for lb in labels:
        f = ((fam.get(lb) or {}).get("families") or {}).get("refcv6", {}).get("tactical") or {}
        g = f.get("goal_setting") or {}
        L.append(f"| {lb} | " + " | ".join(f"{fmt((f.get(k) or {}).get('accuracy'))} / {fmt((f.get(k) or {}).get('kappa'))}"
                                            for k in TAC) +
                 f" | {fmt(g.get('goal_point_error_m'))} | {fmt(g.get('goal_bearing_mae_deg'))} |")
    st = ((fam.get(labels[0]) or {}).get("families") or {}).get("refcv6", {}).get("strategic") or {}
    L += ["", f"### STRATEGIC: **{st.get('status', 'MISSING')}** (n = {st.get('n')}) -- "
          f"{str(st.get('reason', ''))[:400]}", "",
          "### Adverse separations, T_s vs Wt_s (a T interval entirely on the worse side of Wt's; named, NOT gating)", ""]
    for s, v in adverse.items():
        if v.get("missing"):
            L.append(f"- seed {s}: families MISSING")
            continue
        L.append(f"- seed {s}: {len(v['adverse'])} of {v['components_compared']} components -- " +
                 (", ".join(f"{x['family']}.{x['metric']} (Wt {x['W_ci']}, T {x['P_ci']})" for x in v["adverse"])
                  or "none"))
    L += ["", "## Heading profile (reported, not gating)", ""]
    for n, v in hm.items():
        L.append(f"- {n}: winner heading error, median by NAVSIM pose 0.5..4.0 s: "
                 f"{v.get('winner_heading_err_median_rad_by_navsim_pose_0.5_to_4.0s')}; raw |h| > pi (%) by native "
                 f"step 0..19: {v.get('pct_raw_heading_beyond_pi_by_native_step_0_to_19')}; all-slot step-19 vs own "
                 f"tangent (median rad): {fmt(v.get('median_allslot_h19_vs_own_tangent_rad'))}")
    # ---- declared deviations and the run record
    fc = jload(out / "fast_copy.json") or {}
    reused = [v for k, v in sorted(fc.items()) if k.startswith("reused_")]
    deleted = [(k, v) for k, v in sorted(fc.items()) if k.startswith("deleted_")]
    cfgs = {n: jload(arms / n / "config.json") for n in [f"Wt{s}" for s in SEEDS] + [f"T{s}" for s in SEEDS]}
    caches = sorted({c["argv"][c["argv"].index("--cache") + 1] for c in cfgs.values() if c})
    stm = jload(out / "selftest_measures.json") or {}
    spt = jload(out / "selftest_proxy_train.json") or {}
    spe = jload(out / "selftest_proxy_eval.json") or {}
    base = jload(out / "code_baseline.json") or {}
    L += ["", "## Declared implementation choices and the run record (M6b)", "",
          "1. **Micro-batch 64 x accum 4 = effective batch 256**, as M6 (proof: "
          "`raw/2026-09-27-m6-proxy/microbatch_invariance.json`). The scorer loss on covered samples only, the visual "
          "context on labelled frames only, the live calibration table, the harness label prefix: all as M6 (its "
          "RESULT section 'Pre-data implementation deviations').",
          f"2. **All six arms read ONE train-cache path**: {caches} (G5 compares argv).",
          "3. **The NVMe copy**: the epoch-2 copy was KEPT as M6b's one authorised copy (coordinator 00:4x: floor >= 30 "
          "GiB with it present, emergency < 15 GiB, delete after analyze-m6b) and RE-HASHED file by file against the "
          "source manifest before any arm read it: " +
          ("; ".join(f"{len(v.get('files_manifest_copy_sha256', []))} shard files, verify rc {v.get('verify_rc')}, "
                     f"C: free {v.get('c_free_gib')} GiB" for v in reused) or "no reuse record") +
          (". Deleted: " + "; ".join(f"{k[8:]} ({v.get('why')}): C: {v.get('c_free_gib_before')} -> "
                                     f"{v.get('c_free_gib_after')} GiB" for k, v in deleted) if deleted else "") + ".",
          "4. **G3 re-run under the M6b trainer bytes** (`arms/G3_zero_lr`, M6's argv) and **the untouched-snapshot "
          "decode re-run under the final proxy_eval bytes** (`dumps/base.json`) -- every stage ran the SAME code "
          "(the prereg's reason for re-running the wrapped arms).",
          "5. **Code identity**: the chain recorded the sha256 of every module an arm or a decode imports at start "
          f"(`code_baseline.json`, {len(base.get('base_sha256', {}))} modules) and asserted it unchanged before and "
          "after every GPU stage; each arm's own `config.json` code_sha256 was asserted equal to it.",
          f"6. **Selftests on the final bytes**: selftest_measures {stm.get('n_checks', '?')} checks, failed "
          f"{stm.get('failed', '?')}; selftest_proxy_train {spt.get('n_checks', '?')}, failed {spt.get('failed', '?')}; "
          f"selftest_proxy_eval {spe.get('n_checks', '?')}, failed {spe.get('failed', '?')}. Measures and proxy_train "
          "ran in a sibling copy of the package holding the SAME bytes (tested sha256 == the package's, asserted by "
          "the chain) while the epoch-2 analysis still held the package; selftest_measures (~25 min) ran BESIDE the "
          "first arms and the analysis waited for its PASS on the chain's exact measures.py.",
          "7. **Chain v1 -> v2 at 00:51**: v1 admitted two stages on ONE VRAM reading (harmless then: 2,525 + 800 + "
          "2,500 = 5,825 MiB <= 6,960); v2 serialises the gate and counts its own ramping stages. v1's two running "
          "stages (G3, the base decode) were adopted, not repeated.", ""]
    rep["declared"] = {"caches": caches, "fast_copy": fc, "selftests": {"measures": stm.get("failed"),
                                                                        "proxy_train": spt.get("failed"),
                                                                        "proxy_eval": spe.get("failed")}}
    rmd = out / "RESULT_M6B_TANGENT.md"
    head = rmd.read_text(encoding="utf-8").split(MARK)[0].rstrip() if rmd.exists() else "# RESULT: M6b (header missing)"
    rmd.write_text(head + "\n\n" + "\n".join(L) + "\n", encoding="utf-8", newline="\n")
    json.dump(rep, open(out / "result_m6b_report.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    print("ZZM6B_REPORT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
