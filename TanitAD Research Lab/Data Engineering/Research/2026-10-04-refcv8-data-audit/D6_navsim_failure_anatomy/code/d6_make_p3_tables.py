"""D6 -- render raw/d6_p3_tables.md from raw/d6_p3_navhard.json and raw/d6_p3_navtest.json (generated, never hand-copied)."""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "raw")


def J(n):
    return json.load(open(os.path.join(RAW, n), encoding="utf-8"))


def pc(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f} %"


def ci(c):
    return "" if not c or c[0] is None else f" [{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def main():
    nh, nt = J("d6_p3_navhard.json"), J("d6_p3_navtest.json")
    L = []
    w = L.append
    sp, sn = nh["speed"], nh["snap"]
    w(f"### P3-a SPEED oracle on navhard NC-zero scenes (n scored {sp['n_scored']} of {sp['n_NC_zero']}; the same geometric path driven at s x the planned speed; exact reactive devkit re-score)\n")
    w("| speed scale | NC cleared n (share [95 % log CI]) | cleared and DAC not worse | median 4-s distance m: cleared plans / all scaled plans (banked plans / PDM-closed reference) | cleared plans that still travel >= half the reference distance |")
    w("|---|---|---|---|---|")
    r = sp["results"]
    for s in ("S0.8", "S0.6", "S0.4"):
        v = r[s]
        w(f"| x{s[1:]} | {v['n_cleared']} of {v['n']} ({pc(v['NC_cleared_share'])}{ci(v['ci95_logs'])}) | {v['cleared_and_DAC_not_worse']} | {v['median_end_dist_of_cleared']:.1f} / {v['median_end_dist_all']:.1f} ({v['median_banked_end_dist']:.1f} / {v['median_reference_end_dist']:.1f}) | {v['cleared_with_end_dist_ge_half_reference']} |")
    a = r["any_scale"]
    w(f"| any of the three (oracle) | {a['n_cleared']} of {a['n']} ({pc(a['NC_cleared_share'])}{ci(a['ci95_logs'])}) | | | |")
    w(f"\nNot fixable by any speed (STOP also NC-zero: the initial-overlap scenes): {r['not_fixable_by_any_speed_STOP_also_NC_zero']}; among the scenes where STOP passes, the oracle clears {r['cleared_any_given_STOP_passes']['cleared']} of {r['cleared_any_given_STOP_passes']['n']} ({pc(r['cleared_any_given_STOP_passes']['share'])}).")
    if "by_nc_class" in r:
        w("\nBy NC class (from the re-score of RESULT s5):\n")
        w("| class | n | cleared by any scale | x0.8 | x0.6 | x0.4 |")
        w("|---|---|---|---|---|---|")
        for c, v in sorted(r["by_nc_class"].items(), key=lambda kv: -kv[1]["n"]):
            w(f"| {c} | {v['n']} | {v['any_cleared']} | {v['S0.8']} | {v['S0.6']} | {v['S0.4']} |")
    e = sp.get("official_epdms_oracle_speed_on_NC0_tokens")
    if e:
        w(f"\nOfficial two-stage EPDMS if the per-token best of {{banked, x0.8, x0.6, x0.4}} were taken on the NC-zero scenes: {e['base']:.4f} -> {e['oracle']:.4f} (delta {e['delta']:+.4f}; ESTIMATED upper bound, EC and stage-2 weights held; {e['n_tokens_edited']} scenes edited).")
    w("\n### P3-b LATERAL / SPEED oracle on navhard DAC-zero scenes (n scored %d of %d; map-only DAC test of the edited plan; centreline = the cached PDM-Closed route centreline)\n" % (sn["n_ok"], sn["n_DAC_zero"]))
    w("| fix | what it does | DAC cleared n (share [95 % log CI]) | in clean-path scenes | in reference-also-fails scenes | stage 1 | stage 2 | cleared and DDC ok |")
    w("|---|---|---|---|---|---|---|---|")
    desc = {"B075": "shift toward the centreline by <= 0.75 m", "B150": "shift toward the centreline by <= 1.5 m", "FULL": "put every point ON the centreline at the plan's own along-route progress (route-follow oracle)",
            "V80": "same path at 0.8 x speed", "V60": "same path at 0.6 x speed", "V40": "same path at 0.4 x speed"}
    for v in ("B075", "B150", "FULL", "V80", "V60", "V40"):
        x = sn["results"][v]
        w(f"| {v} | {desc[v]} | {x['n_cleared']} of {x['n']} ({pc(x['DAC_cleared_share'])}{ci(x['ci95_logs'])}) | {x['cleared_in_clean_path_scenes']['cleared']} of {x['cleared_in_clean_path_scenes']['n']} | {x['cleared_in_ref_fails_scenes']['cleared']} of {x['cleared_in_ref_fails_scenes']['n']} | {x['by_stage']['1']['cleared']} of {x['by_stage']['1']['n']} | {x['by_stage']['2']['cleared']} of {x['by_stage']['2']['n']} | {x['cleared_and_DDC_ok']} |")
    u = sn["union_FULL_or_V60_or_V40"]
    w(f"\nUnion (FULL snap or x0.6 or x0.4 speed): {u['n_cleared']} of {sn['n_scored']} ({pc(u['share'])}). Control: the identity plan stays DAC-zero on all scenes: {sn['control_ID_dac_is_zero_on_all']}.")
    w("\nOfficial two-stage EPDMS if DAC were set to 1 on the scenes a fix clears (other sub-scores and weights held; ESTIMATED upper bound):\n")
    w("| fix | official EPDMS | delta vs A1 0.2269 |")
    w("|---|---|---|")
    for v, x in sn["official_epdms_DAC_cleared_by_variant"].items():
        w(f"| {v} | {x['official_epdms']:.4f} | {x['delta_vs_A1']:+.4f} |")
    # navtest
    w(f"\n### P3-c NAVTEST on the navsim-1.1 devkit (n scored {nt['n_ok']} of the DAC-zero {nt['n_DAC_zero_scored']} + NC-zero {nt['n_NC_zero_scored']} + controls)\n")
    c = nt["controls"]
    w(f"Controls: exact re-score reproduces the banked row on {pc(c['rescore_repro_exact_share'])} of scenes (max abs {c['rescore_repro_max_abs_all']:.2g}); the identity snap reproduces the banked DAC on {pc(c['identity_snap_DAC_equals_banked'])}; PASS = {c['PASS']}.\n")
    w("DAC-zero first non-drivable instant (s): " + ", ".join(f"{k}: {v}" for k, v in nt["DAC_first_nondrivable_time_s"].items()) + f"; t = 0 violations {nt['DAC_t0_violations']}; clean path exists (reference passes) {pc(nt['DAC_clean_path_share'])}; reference also fails {nt['DAC_ref_also_fails']}.\n")
    w("NC-zero first at-fault event classes:\n")
    w("| class | n | share |")
    w("|---|---|---|")
    for k, v in sorted(nt["NC_classes"].items(), key=lambda kv: -kv[1]["n"]):
        w(f"| {k} | {v['n']} | {pc(v['share'])} |")
    w(f"\nFront collisions (active-front + stopped-track): {pc(nt['NC_front_collision_share'])} of NC-zero.\n")
    w("| oracle | clears | share [95 % log CI] |")
    w("|---|---|---|")
    for s in ("S0.8", "S0.6", "S0.4", "any_scale"):
        v = nt["speed_oracle_NC"][s]
        w(f"| speed {s if s != 'any_scale' else 'any of the three'} on NC-zero | {v['n_cleared']} of {v['n']} | {pc(v['NC_cleared_share'])}{ci(v['ci95_logs'])} |")
    for s in ("B075", "B150", "FULL", "V80", "V60", "V40"):
        v = nt["snap_oracle_DAC"][s]
        w(f"| {s} on DAC-zero | {v['n_cleared']} of {v['n']} | {pc(v['DAC_cleared_share'])}{ci(v['ci95_logs'])} |")
    p = nt["PDMS_x100"]
    w(f"\nPDMS x100 (ESTIMATED upper bounds; banked A1 {p['A1_banked']:.2f}, STOP {p['STOP']:.2f}): oracle speed on NC-zero {p['oracle_speed_on_NC0']:.2f}; DAC := 1 where the FULL snap clears {p['DAC_to_1_where_FULL_snap_clears']:.2f}.")
    open(os.path.join(RAW, "d6_p3_tables.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(len(L), "lines")


if __name__ == "__main__":
    main()
