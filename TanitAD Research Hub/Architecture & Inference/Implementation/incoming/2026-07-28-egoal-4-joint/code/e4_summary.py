#!/usr/bin/env python3
"""E-GOAL-4 S2 -- the two backgrounds side by side, and the MECHANICAL verdict.

⛔ C30. The whole point of running both is that a recovery number is a property
of the treatment AND the background. `parent_resampled` is E-GOAL-3's headline
carrier and is CONSERVATIVE in recovery; `sel` is the FUTURE-BLIND one (S0
proved it: `future_blind` max |Δ| == 0.0 on every goal column, vs 5.2e4 under
`parent_resampled`). Where the two disagree, the `sel` cell is the one that
survives the C23 audit -- registered in PRE_REGISTRATION §4 before either ran.

⭐ THE NUMBER THE HEADLINE MUST CARRY IS THE MARGINAL. E-GOAL-3's +46.3 % is
measured against `A0`, the AS-TRAINED selector. The counterfactual for "does v5
need a goal INPUT" is a TRAINED selector WITHOUT one (`S_nogoal`), which is a
different and much stronger baseline.

Run:  python e4_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e4_common import E3_FIXED, STREAM  # noqa: E402

BG = ("parent_resampled", "sel")
ORDER = ["A0", "FIXED_cv", "FIXED_goal", "FIXED_goal_ego", "FIXED_oracle",
         "FIXED_shuf", "S_nogoal", "S_nogoal_z", "S_goal_shuf", "S_goal_cv",
         "S_goal_crossonly", "S_goal_alongonly", "S_goalonly", "S_goal",
         "S_goal_z", "S_goal_ego", "S_goal_coadapt", "S_LEAK",
         "S_goal_oracle", "S_goal_oracle2d",
         "S_nogoal_INSAMPLE", "S_goal_INSAMPLE"]


def main():
    raw = STREAM / "raw"
    J = {}
    for b in BG:
        p = raw / f"e4_select_{b}.json"
        if p.exists():
            J[b] = json.loads(p.read_text())
    if not J:
        raise SystemExit("no e4_select_*.json yet")
    ref = J[BG[0]] if BG[0] in J else J[next(iter(J))]
    out = {"_stream": "2026-07-28-egoal-4-joint", "_stage": "S2 summary",
           "_backgrounds": list(J),
           "deployment": ref["deployment"],
           "_estimator": ref["_estimator"],
           "_coupling": ref["_coupling"],
           "gates": {b: {k: v for k, v in J[b].items() if k.startswith("G_")}
                     for b in J},
           "arms": {}, "contrasts": {}, "verdict": {}}
    hr = ref["deployment"]["headroom"]

    print(f"{'arm':22s} " + "".join(f"| {b[:16]:^34s} " for b in J))
    print("-" * (22 + 36 * len(J)))
    for k in ORDER:
        row = {}
        line = f"{k:22s} "
        for b in J:
            aa = J[b]["arms"].get(k)
            if aa is None:
                line += "| " + " " * 34
                continue
            p = aa["paired_vs_as_trained"]
            row[b] = {"realised": aa["realised_ade_0_2s"]["mean"],
                      "recovery": aa["recovery_of_headroom"],
                      "paired": [p["delta"], p["lo"], p["hi"]],
                      "verdict": aa["verdict"],
                      "ties_mean": aa.get("ties_at_min_mean"),
                      "pick_eq_a0": aa.get("pick_equals_as_trained_frac")}
            line += (f"| {aa['realised_ade_0_2s']['mean']:.4f} "
                     f"{100*aa['recovery_of_headroom']:+8.2f}% "
                     f"{aa['verdict'][:6]:6s}   ")
        if row:
            out["arms"][k] = row
            print(line)

    print()
    keys = set()
    for b in J:
        keys |= set(J[b]["contrasts"])
    for k in sorted(keys):
        row = {}
        line = f"{k[:44]:44s} "
        for b in J:
            c = J[b]["contrasts"].get(k)
            if c is None:
                line += "| " + " " * 30
                continue
            d = c["paired_delta_ade"]
            row[b] = {"delta": d["delta"], "lo": d["lo"], "hi": d["hi"],
                      "separated": c["separated"],
                      "recovery_points": c["recovery_points"],
                      "_what": c["_what"]}
            line += (f"| {d['delta']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}] "
                     f"{'SEP' if c['separated'] else 'null'}  ")
        if row:
            out["contrasts"][k] = row
            print(line)

    print()
    for b in J:
        v = dict(J[b].get("VERDICT", {}))
        v["EGOAL_3_fixed_rule_recovery_raw_json"] = E3_FIXED[b]["H_v0_ax"]
        v["background_future_blind"] = bool(b == "sel")
        out["verdict"][b] = v
        print(f"[{b:17s}] {v.get('verdict')}  total {100*v.get('recovery',0):+.2f}% "
              f"| S_nogoal {100*v.get('S_nogoal_recovery',0):+.2f}% "
              f"| ⭐ GOAL MARGINAL {v.get('goal_marginal_recovery_points')} pts "
              f"{v.get('goal_marginal_recovery_points_ci')} "
              f"| fixed rule {100*E3_FIXED[b]['H_v0_ax']:+.2f}%")

    # ------------------------------------------------------------------------
    # ⭐ EXTRA PAIRED CONTRASTS, computed here from the staged per-window
    # realised arrays. Every arm involved was pre-registered; only the PAIRING
    # is added, because S1's result made these the decisive ones. Stated as
    # post-hoc pairings of pre-registered arms, not as new arms.
    # ------------------------------------------------------------------------
    import numpy as np                                   # noqa: E402
    from e4_common import ci_paired, sep                 # noqa: E402
    EXTRA = [
        ("S_goal", "S_goal_cv",
         "⭐⭐ DOES THE GOAL'S ACCURACY MATTER? learned `v+ax_fd` vs a naive 2*v0"),
        ("S_goal_cv", "S_goal_shuf",
         "a CV goal vs a REAL goal from the WRONG episode"),
        ("S_goal_alongonly", "S_goal_shuf",
         "the ALONG axis, capacity-matched against the shuffled arm"),
        ("S_nogoal_INSAMPLE", "S_nogoal", "in-sample vs OOF, no-goal arm"),
        ("S_goal_oracle", "FIXED_oracle",
         "the SAME true goal through the trained rule vs the fixed rule"),
        ("S_goal_cv", "FIXED_cv",
         "⭐ the SAME CV goal through the trained rule vs the fixed rule"),
        ("S_nogoal", "S_goal_oracle", "the whole span the goal slot can buy"),
    ]
    out["extra_contrasts"] = {}
    for b in J:
        f = raw / f"e4_select_{b}_realised.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        eid = z["eid"]
        for x, yy, why in EXTRA:
            kx, ky = f"r|{x}", f"r|{yy}"
            if kx in z.files and ky in z.files:
                pc = ci_paired(z[kx], z[ky], eid)
                out["extra_contrasts"].setdefault(f"{x}__vs__{yy}", {})[b] = {
                    "_what": why, "delta": pc["delta"], "lo": pc["lo"],
                    "hi": pc["hi"], "separated": sep(pc),
                    "recovery_points": round(-100 * pc["delta"] / hr, 2),
                    "recovery_points_ci": [round(-100 * pc["hi"] / hr, 2),
                                           round(-100 * pc["lo"] / hr, 2)]}
    print("\nEXTRA (post-hoc pairings of pre-registered arms):")
    for k, v in out["extra_contrasts"].items():
        line = f"  {k[:44]:44s} "
        for b in J:
            c = v.get(b)
            line += ("| " + " " * 30) if c is None else (
                f"| {c['delta']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}] "
                f"{'SEP' if c['separated'] else 'null'} {c['recovery_points']:+6.2f}pts ")
        print(line)

    # recovery-point conversion of every §7 contrast, so the report never has
    # to divide by hand
    for k, v in out["contrasts"].items():
        for b, c in v.items():
            c["recovery_points_from_delta"] = round(-100 * c["delta"] / hr, 2)
            c["recovery_points_ci"] = [round(-100 * c["hi"] / hr, 2),
                                       round(-100 * c["lo"] / hr, 2)]

    if "ladder" in ref:
        out["ladder"] = ref["ladder"]
        print("\nladder (primary background):")
        for r in ref["ladder"]:
            print(f"  k={r['k']:<5} rms={r['along_rms_m']:.3f} m  trained "
                  f"{100*r['trained']:+7.2f}% {'sep' if r['trained_sep'] else 'NOT SEP'}"
                  f"   fixed {100*r['fixed']:+7.2f}%")

    (raw / "e4_summary.json").write_text(json.dumps(out, indent=1))
    print(f"\n-> {raw/'e4_summary.json'}")


if __name__ == "__main__":
    main()
