"""Generate every table in E1C_RESULTS.md from the raw JSONs.

Numbers in the report are GENERATED, never re-typed — the rule E1b adopted after
three prose-copied errors propagated for days.

  python summarize_e1c.py <frontier_result.json> <heldout_gate.jsonl> [selftest.json]
"""
import json
import sys
from pathlib import Path


def D(d):
    """paired delta cell"""
    if not isinstance(d, dict) or "delta" not in d:
        return "n/a"
    sep = " **SEP**" if d.get("separated") else ""
    return f"{d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]{sep}"


def A(d, nd=4):
    """single-arm cell"""
    if not isinstance(d, dict) or "mean" not in d:
        return "n/a"
    return f"{d['mean']:.{nd}f} [{d['lo']:.{nd}f}, {d['hi']:.{nd}f}]"


def yn(b):
    return "OK" if b else "**FAIL**"


def main():
    res = json.loads(Path(sys.argv[1]).read_text())
    gate = [json.loads(l) for l in Path(sys.argv[2]).read_text().splitlines() if l.strip()]
    steps = [s for s in res["frontier_steps"] if str(s) in res["points"]]
    P = res["points"]

    print("### FRONTIER — closed-loop gain vs open-loop cost, per checkpoint\n")
    print("| step | CDR@K185 overall Δ | CDR@K185 junction Δ | open-loop ADE@2s Δ | "
          "anchor_acc Δ | anchor_traj_l1 Δ | OOD peak | P1 | P2 | Ga | Gb1 | Gb2 | Gc | **SUCCESS** |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    print(f"| **0 (base)** | — (ref) | — (ref) | — (ref) | — (ref) | — (ref) | "
          f"{res['base_reproduction_check']['K185_ood_peak']:.4f} | — | — | — | — | — | — | — |")
    for s in steps:
        p = P[str(s)]; e = p["EVAL"]
        cl = p["closed_loop_K185"]; ol = p["open_loop"]
        print(f"| {s} | {D(cl['overall']['dep']['paired_delta_ft_minus_base'])} | "
              f"{D(cl['junction']['dep']['paired_delta_ft_minus_base'])} | "
              f"{D(ol['ade2s']['paired_delta_ft_minus_base'])} | "
              f"{D(ol['anchor_acc']['paired_delta_ft_minus_base'])} | "
              f"{D(ol['anchor_traj_l1']['paired_delta_ft_minus_base'])} | "
              f"{e['_ood_peak_ft']:.4f} | "
              f"{yn(e['P1_dep_overall_separated_lower'])} | {yn(e['P2_dep_junction_separated_lower'])} | "
              f"{yn(e['Ga_openloop_ade2s_ok'])} | {yn(e['Gb1_anchor_acc_ok'])} | "
              f"{yn(e['Gb2_anchor_traj_l1_ok'])} | {yn(e['Gc_ood_in_band'])} | "
              f"{'**YES**' if e['SUCCESS_POINT'] else 'no'} |")

    print("\n### FRONTIER — absolute levels (episode-cluster bootstrap per arm)\n")
    b = res["base_reproduction_check"]
    print("| step | CDR@K185 overall | CDR@K185 junction | peak abs XTE (m) | "
          "open-loop ADE@2s (m) | anchor_acc | anchor_traj_l1 |")
    print("|---|---|---|---|---|---|---|")
    print(f"| **0 (base)** | {b['K185_overall_dep']:.4f} | {b['K185_junction_dep']:.4f} | "
          f"{b['K185_peak_xte']:.4f} | {b['openloop_ade2s']:.4f} | "
          f"{b['openloop_anchor_acc']:.4f} | {b['openloop_anchor_traj_l1']:.4f} |")
    for s in steps:
        p = P[str(s)]; cl = p["closed_loop_K185"]; ol = p["open_loop"]
        print(f"| {s} | {A(cl['overall']['dep']['ft'])} | {A(cl['junction']['dep']['ft'])} | "
              f"{A(cl['overall']['peak_xte']['ft'])} | {A(ol['ade2s']['ft'])} | "
              f"{A(ol['anchor_acc']['ft'])} | {A(ol['anchor_traj_l1']['ft'])} |")

    print("\n### IN-TRAINING HELD-OUT GATE (the corrected guard, computed during the run)\n")
    print("| step | ADE@2s base→ft | paired Δ | sep? | anchor_acc | anchor_traj_l1 | "
          "gate_ok | probe s |")
    print("|---|---|---|---|---|---|---|---|")
    for r in gate:
        d = r["ade2s"]["paired_delta"]
        print(f"| {r['step']} | {r['ade2s']['base_mean']:.4f} → {r['ade2s']['ft_mean']:.4f} | "
              f"{d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] | "
              f"{'**SEP**' if d['separated'] else 'no'} | "
              f"{r['anchor_acc']['base_mean']:.4f} → {r['anchor_acc']['ft_mean']:.4f} | "
              f"{r['anchor_traj_l1']['base_mean']:.4f} → {r['anchor_traj_l1']['ft_mean']:.4f} | "
              f"{'OK' if r['gate_ok'] else '**STOP**'} | {r.get('probe_s')} |")

    def strata_block(s, title):
        p = P[str(s)]
        print(f"\n### {title} — closed loop @ K=185, all strata\n")
        print("| stratum | metric | base | step %s | paired Δ |" % s)
        print("|---|---|---|---|---|")
        for nm in ("overall", "junction", "longitudinal"):
            for f in ("dep", "win_dep", "peak_xte", "mean_xte", "peak_dpsi",
                      "ood_peak", "ood_mean", "out_env", "ade2s"):
                c = p["closed_loop_K185"][nm][f]
                print(f"| {nm} | {f} | {A(c['base'])} | {A(c['ft'])} | "
                      f"{D(c['paired_delta_ft_minus_base'])} |")
        print(f"\n**Open-loop guardrails at step {s}**\n")
        print("| metric | base | step %s | paired Δ |" % s)
        print("|---|---|---|---|")
        for f in ("ade2s", "anchor_acc", "anchor_ce", "anchor_traj_l1"):
            c = p["open_loop"][f]
            print(f"| {f} | {A(c['base'])} | {A(c['ft'])} | "
                  f"{D(c['paired_delta_ft_minus_base'])} |")
        if "closed_loop_K20_nondeciding" in p:
            print(f"\n**K=20 (2 s) — reported, NON-DECIDING, at step {s}**\n")
            print("| stratum | metric | base | step %s | paired Δ |" % s)
            print("|---|---|---|---|---|")
            for nm in ("overall", "junction", "longitudinal"):
                for f in ("dep", "win_dep", "peak_xte", "ade2s"):
                    c = p["closed_loop_K20_nondeciding"][nm][f]
                    print(f"| {nm} | {f} | {A(c['base'])} | {A(c['ft'])} | "
                          f"{D(c['paired_delta_ft_minus_base'])} |")

    def m1_block(s, title):
        p = P[str(s)]
        for which, lbl in (("openloop", "open loop, held-out 44"),
                           ("closedloop_K185", "closed loop, 2 s knots inside K=185")):
            m = p["M1_lateral_split"][which]
            print(f"\n### M1 lateral/longitudinal — {title}, {lbl}\n")
            print(f"GT identity check max|Δ| = {m['_gt_identity_max_abs_diff']} · "
                  f"axis convention {m.get('_axis_convention')} · "
                  f"n_windows {m['n_windows']}\n")
            print("| frame | metric | base | step %s | paired Δ |" % s)
            print("|---|---|---|---|---|")
            for mode in ("ego", "frenet"):
                blk = m[mode]
                for k in ("ade_over_knots", "cross_abs@2s", "cross_p90@2s",
                          "along_abs@2s"):
                    print(f"| {mode} | {k} | {A(blk['base'][k])} | {A(blk['ft'][k])} | "
                          f"{D(blk['paired_delta_ft_minus_base'][k])} |")
                print(f"| {mode} | energy share (lon/lat) | "
                      f"{blk['base']['energy_share']} | {blk['ft']['energy_share']} | — |")

    v = res["VERDICT"]
    focus = [s for s in (v.get("winner_step"), steps[-1] if steps else None)
             if s is not None]
    seen = set()
    for s in focus:
        if s in seen:
            continue
        seen.add(s)
        tag = "SELECTED CHECKPOINT" if s == v.get("winner_step") else "ENDPOINT"
        strata_block(s, f"{tag} (step {s})")
        m1_block(s, f"{tag} step {s}")

    print("\n### VERDICT\n")
    print("```")
    print(json.dumps(v, indent=2))
    print("```")
    print("\n### Base reproduction control\n```")
    print(json.dumps(res["base_reproduction_check"], indent=2))
    print("```")


if __name__ == "__main__":
    main()
