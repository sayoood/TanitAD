"""EP-ADMISSION-R2-CONFIRM analysis: apply the COMMITTED sealing rule of MM_RULING_EPADM_R1.md section 2 to a FRESH set of replicates (ep_admission_sim.py run --rep-offset 1000000).
Nothing here chooses a bar: the bars are constants below, copied from the ruling, and the decision is a pure function of the rates.
   python r2_confirm.py --workdir D:/epadm2/run [--supp D:/epadm2/run/a2_power_g25_g05.json] [--banked-dir D:/epadm2/banked]
writes <workdir>/r2_confirm.json and <workdir>/result_tables.md.   Also: `--selftest` (the decision function on constructed rates, incl. RED arms)."""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ep_admission_sim as S                                                                  # noqa: E402

# ---- the committed bars (MM_RULING_EPADM_R1.md section 2); do not edit ----
PRIMARY, MARGIN_SEAL, MARGIN_FALLBACK = "G78", 1.05, 1.10
BAR_A2_FR_WILSON_UPPER = 0.05
BAR_DAC_POWER_125_POINT = 0.70
BAR_A1_FR_WILSON_UPPER = 0.05
GEOMS = ("G78", "G25", "G05")
MARGINS = (1.00, 1.05, 1.10)
RHOS = (1.25, 1.5)


def decide(a2_fr_upper: float, dac_power_125: float, a1_fr_upper: float) -> dict:
    """The committed rule, exactly: seal at 1.05 iff ALL THREE bars hold on the primary geometry, else seal at 1.10 and name the bars that missed."""
    bars = {"A2_1.05_false_refusal_wilson_upper_le_5pct": a2_fr_upper <= BAR_A2_FR_WILSON_UPPER,
            "DAC_power_at_x1.25_point_ge_70pct": dac_power_125 >= BAR_DAC_POWER_125_POINT,
            "A1_r1_false_refusal_wilson_upper_le_5pct": a1_fr_upper <= BAR_A1_FR_WILSON_UPPER}
    seal = MARGIN_SEAL if all(bars.values()) else MARGIN_FALLBACK
    return {"seal_margin": seal, "bars": bars, "missed": [k for k, v in bars.items() if not v]}


def fr_a2(a2: np.ndarray, t: float) -> dict:
    """False refusal at margin t: any head, any seed has a lower CI above t.  a2 [n, seed, head, (point, lo, hi)]."""
    return S.rate(int((a2[:, :, :, 1] > t).any(axis=(1, 2)).sum()), len(a2))


def fr_a2_head(a2: np.ndarray, j: int, t: float) -> dict:
    return S.rate(int((a2[:, :, j, 1] > t).any(axis=1).sum()), len(a2))


def power_a2(a2: np.ndarray, j: int, t: float) -> dict:
    """Power at margin t: the PLANTED head's lower CI above t on any seed.  a2 [n, seed, head, (point, lo, hi)]."""
    return S.rate(int((a2[:, :, j, 1] > t).any(axis=1).sum()), len(a2))


def fr_a1(dm: np.ndarray) -> dict:
    """A1-r1 false refusal, any seed, from the raw replicate CIs, the rule restated inline (FAIL iff lo > 0 or point > +0.05 on any seed)."""
    return S.rate(int(((dm[:, :, 1] > 0.0) | (dm[:, :, 0] > 0.05)).any(axis=1).sum()), len(dm))


def pct(r: dict) -> str:
    return f"{100 * r['rate']:.1f} % [{100 * r['wilson95'][0]:.1f}, {100 * r['wilson95'][1]:.1f}] ({r['k']}/{r['n']})"


def selftest() -> int:
    ok = True
    for args, want in (((0.049, 0.70, 0.049), 1.05), ((0.0501, 0.80, 0.03), 1.10), ((0.03, 0.6999, 0.03), 1.10), ((0.03, 0.80, 0.0501), 1.10), ((0.05, 0.70, 0.05), 1.05), ((0.2, 0.1, 0.2), 1.10)):
        d = decide(*args)
        ok &= d["seal_margin"] == want
    d = decide(0.03, 0.6999, 0.0501)
    ok &= d["missed"] == ["DAC_power_at_x1.25_point_ge_70pct", "A1_r1_false_refusal_wilson_upper_le_5pct"]
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=False)
    ap.add_argument("--supp", default=None, help="a2_power_g25_g05.json (supplementary A2 power on G25 / G05; not in the banked condition set)")
    ap.add_argument("--banked-dir", default=None, help="dir holding the banked replicates.npz / sim_results.json for the side-by-side column")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    z = np.load(os.path.join(a.workdir, "replicates.npz"))
    get = lambda n, k: z[f"{n}|{k}".replace("|", "__")]                                       # noqa: E731
    sim = json.load(open(os.path.join(a.workdir, "sim_results.json"), encoding="utf-8"))
    out: dict = {"what": "EP-ADMISSION-R2-CONFIRM: the committed sealing rule applied to FRESH replicates", "rep_offset": sim.get("rep_offset"), "n_reps": sim["n_reps"],
                 "controls_ok_excluding_PC5": sim["controls_ok_excluding_PC5"], "ep_admission_sha256": sim["ep_admission_sha256"], "bars": {"A2_fr_wilson_upper_le": BAR_A2_FR_WILSON_UPPER, "DAC_power_x1.25_point_ge": BAR_DAC_POWER_125_POINT, "A1_fr_wilson_upper_le": BAR_A1_FR_WILSON_UPPER}}
    # A2 false refusal, all margins and geometries
    out["A2_false_refusal"] = {g: {str(t): fr_a2(get(f"{g}|A|CAL", "a2"), t) for t in MARGINS} for g in GEOMS}
    out["A2_false_refusal_per_head_any_seed"] = {g: {h: {str(t): fr_a2_head(get(f"{g}|A|CAL", "a2"), S.HEADS.index(h), t) for t in MARGINS} for h in S.HEADS} for g in GEOMS}
    # A2 power (G78: in the committed condition set)
    out["A2_power_G78"] = {h: {str(rho): {str(t): power_a2(get(f"G78|A|PLANT_A2_{h}_{rho}", "a2"), S.HEADS.index(h), t) for t in MARGINS} for rho in RHOS} for h in S.HEADS}
    # A1-r1 false refusal (module decision functions from sim_results.json, and the inline restatement)
    out["A1_r1_false_refusal"] = {g: {"module_any_seed": sim["conditions"][f"{g}|A|CAL"]["A1"]["r1_any_seed"], "inline_any_seed": fr_a1(get(f"{g}|A|CAL", "dm")),
                                      "seed0": sim["conditions"][f"{g}|A|CAL"]["A1"]["r1_seed0"], "seed1": sim["conditions"][f"{g}|A|CAL"]["A1"]["r1_seed1"]} for g in GEOMS}
    out["A1_r1_inline_equals_module"] = bool(all(out["A1_r1_false_refusal"][g]["module_any_seed"]["k"] == out["A1_r1_false_refusal"][g]["inline_any_seed"]["k"] for g in GEOMS))
    out["A1_r1_gen_B_G78"] = sim["conditions"]["G78|B|CAL"]["A1"]["r1_any_seed"]
    out["A1_r1_power_G78"] = {d: sim["conditions"][f"G78|A|PLANT_EP_{d}"]["A1"]["r1_any_seed"] for d in ("0.05", "0.135")}
    # the decision, on the primary geometry only
    fr105, fr110 = out["A2_false_refusal"][PRIMARY][str(MARGIN_SEAL)], out["A2_false_refusal"][PRIMARY][str(MARGIN_FALLBACK)]
    dac125 = out["A2_power_G78"]["DAC"]["1.25"][str(MARGIN_SEAL)]
    a1 = out["A1_r1_false_refusal"][PRIMARY]["module_any_seed"]
    dec = decide(fr105["wilson95"][1], dac125["rate"], a1["wilson95"][1])
    dec["inputs"] = {"A2_1.05_false_refusal_G78": fr105, "DAC_power_x1.25_margin1.05_G78": dac125, "A1_r1_false_refusal_G78": a1}
    # the fallback's own numbers (reported whichever way the rule goes)
    dec["fallback_1.10_G78"] = {"A2_false_refusal": fr110, "DAC_power_x1.25": out["A2_power_G78"]["DAC"]["1.25"][str(MARGIN_FALLBACK)], "DAC_power_x1.5": out["A2_power_G78"]["DAC"]["1.5"][str(MARGIN_FALLBACK)]}
    out["decision"] = dec
    # side by side with the banked run (the SAME code at offset 0): the margin was CHOSEN on those replicates
    if a.banked_dir:
        bz = np.load(os.path.join(a.banked_dir, "replicates.npz"))
        bget = lambda n, k: bz[f"{n}|{k}".replace("|", "__")]                                  # noqa: E731
        bs = json.load(open(os.path.join(a.banked_dir, "sim_results.json"), encoding="utf-8"))
        out["banked_offset0"] = {"A2_false_refusal": {g: {str(t): fr_a2(bget(f"{g}|A|CAL", "a2"), t) for t in MARGINS} for g in GEOMS},
                                 "A2_power_G78": {h: {str(rho): {str(t): power_a2(bget(f"G78|A|PLANT_A2_{h}_{rho}", "a2"), S.HEADS.index(h), t) for t in MARGINS} for rho in RHOS} for h in S.HEADS},
                                 "A1_r1_false_refusal": {g: bs["conditions"][f"{g}|A|CAL"]["A1"]["r1_any_seed"] for g in GEOMS}}
    if a.supp and os.path.exists(a.supp):
        out["A2_power_supplementary_G25_G05"] = json.load(open(a.supp, encoding="utf-8"))
    json.dump(out, open(os.path.join(a.workdir, "r2_confirm.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)

    L = []
    L.append(f"#### D. THE DECISION (G78, the primary geometry; fresh replicates, rep_offset {out['rep_offset']}, N = {out['n_reps']} per condition)\n")
    L.append("| bar (committed in MM_RULING_EPADM_R1.md s2) | measured | holds |\n|---|---|---|")
    L.append(f"| A2(1.05) false refusal, any head, any seed: Wilson upper <= 5.0 % | {pct(fr105)} -> upper {100 * fr105['wilson95'][1]:.2f} % | **{dec['bars']['A2_1.05_false_refusal_wilson_upper_le_5pct']}** |")
    L.append(f"| DAC power at x1.25 (margin 1.05), point >= 70 % | {pct(dac125)} | **{dec['bars']['DAC_power_at_x1.25_point_ge_70pct']}** |")
    L.append(f"| A1-r1 false refusal, any seed: Wilson upper <= 5.0 % | {pct(a1)} -> upper {100 * a1['wilson95'][1]:.2f} % | **{dec['bars']['A1_r1_false_refusal_wilson_upper_le_5pct']}** |")
    L.append(f"\n**Seal at {dec['seal_margin']:.2f}.** " + ("All three bars hold." if not dec["missed"] else "Bars missed: " + ", ".join(dec["missed"]) + "."))
    L.append("\n#### T1. A2 false refusal on CALIBRATED replicates (any head, any seed), by margin and geometry; Wilson 95 %\n")
    L.append("| geometry | margin 1.00 (r1 as specified) | **margin 1.05** | margin 1.10 (fallback) |\n|---|---|---|---|")
    for g in GEOMS:
        L.append(f"| {g} | " + " | ".join(pct(out["A2_false_refusal"][g][str(t)]) for t in MARGINS) + " |")
    L.append("\n#### T1b. A2 false refusal per head (any seed), G78\n")
    L.append("| head | 1.00 | 1.05 | 1.10 |\n|---|---|---|---|")
    for h in S.HEADS:
        L.append(f"| {h} | " + " | ".join(pct(out["A2_false_refusal_per_head_any_seed"]["G78"][h][str(t)]) for t in MARGINS) + " |")
    L.append("\n#### T2. A2 POWER (G78): refusal of rows that under-predict risk by rho at the picks; any seed; Wilson 95 %\n")
    L.append("| head planted | rho | margin 1.00 | **margin 1.05** | margin 1.10 |\n|---|---|---|---|---|")
    for h in S.HEADS:
        for rho in RHOS:
            L.append(f"| {h} | {rho} | " + " | ".join(pct(out["A2_power_G78"][h][str(rho)][str(t)]) for t in MARGINS) + " |")
    L.append("\n#### T3. A1-r1 false refusal on CALIBRATED replicates (generator A), by geometry; Wilson 95 %\n")
    L.append("| geometry | any seed (module) | any seed (inline restatement) | seed 0 | seed 1 |\n|---|---|---|---|---|")
    for g in GEOMS:
        r = out["A1_r1_false_refusal"][g]
        L.append(f"| {g} | **{pct(r['module_any_seed'])}** | {pct(r['inline_any_seed'])} | {pct(r['seed0'])} | {pct(r['seed1'])} |")
    L.append(f"\nGenerator B (wild sign-flip, G78, the cross-check): {pct(out['A1_r1_gen_B_G78'])}.  A1-r1 power (G78, generator A): +0.05 -> {pct(out['A1_r1_power_G78']['0.05'])}; +0.135 -> {pct(out['A1_r1_power_G78']['0.135'])}.")
    if "banked_offset0" in out:
        b = out["banked_offset0"]
        L.append("\n#### T4. FRESH (offset 1,000,000) beside the BANKED run (offset 0; the replicates the 1.05 was chosen on); point estimates\n")
        L.append("| quantity | banked (offset 0) | fresh (offset 1,000,000) |\n|---|---|---|")
        for g in GEOMS:
            for t in (1.05, 1.10):
                L.append(f"| A2({t:.2f}) false refusal {g} | {pct(b['A2_false_refusal'][g][str(t)])} | {pct(out['A2_false_refusal'][g][str(t)])} |")
        for rho in RHOS:
            for t in (1.05, 1.10):
                L.append(f"| DAC power x{rho} margin {t:.2f} (G78) | {pct(b['A2_power_G78']['DAC'][str(rho)][str(t)])} | {pct(out['A2_power_G78']['DAC'][str(rho)][str(t)])} |")
        for g in GEOMS:
            L.append(f"| A1-r1 false refusal {g} | {pct(b['A1_r1_false_refusal'][g])} | {pct(out['A1_r1_false_refusal'][g]['module_any_seed'])} |")
    if "A2_power_supplementary_G25_G05" in out:
        sp = out["A2_power_supplementary_G25_G05"]
        L.append("\n#### T5. SUPPLEMENTARY: A2 power on G25 and G05 (planted A2 conditions are NOT in the banked condition set; same generator, fresh offset)\n")
        L.append("| geometry | head | rho | margin 1.00 | **margin 1.05** | margin 1.10 |\n|---|---|---|---|---|---|")
        for g in ("G25", "G05"):
            for h in S.HEADS:
                for rho in RHOS:
                    L.append(f"| {g} | {h} | {rho} | " + " | ".join(pct(sp["power"][g][h][str(rho)][str(t)]) for t in MARGINS) + " |")
    open(os.path.join(a.workdir, "result_tables.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nDECISION: seal at {dec['seal_margin']:.2f}; missed: {dec['missed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
