#!/usr/bin/env python3
"""Generate the markdown tables of PSEUDOSIM_ARM_PANEL.md FROM the artifact.

Nothing in the report is hand-typed. Run:
    python3 summarize_panel.py artifacts/pseudosim_arm_panel.json
"""
from __future__ import annotations

import json
import sys


def f(x, n=4, sign=False):
    if x is None:
        return "—"
    s = f"{x:+.{n}f}" if sign else f"{x:.{n}f}"
    return s


def ci(d, n=4, sign=False):
    if not d:
        return "—"
    if "delta" in d:
        return (f"**{f(d['delta'], n, sign)}** [{f(d['lo'], n, True)}, "
                f"{f(d['hi'], n, True)}] "
                + ("⭐ SEP" if d.get("separated") else "n.s."))
    return f"{f(d.get('mean'), n)} [{f(d.get('lo'), n)}, {f(d.get('hi'), n)}]"


def main():
    r = json.load(open(sys.argv[1], encoding="utf-8"))
    arms = r["arms"]
    order = sorted(arms, key=lambda k: -(
        ((arms[k].get("composite") or {}).get("ci") or {}).get("mean") or -9))

    print("### Panel gate\n")
    pg = r["PANEL_GATE"]
    print(f"admitted: `{sorted(pg['admitted'])}`  ·  dropped: "
          f"`{sorted(pg['dropped'])}`\n")
    for k, v in pg["dropped"].items():
        print(f"* `{k}` — {v}")
    print()

    print("### Arm scores\n")
    print("| arm | n evals | n eps | goal provenance | `ego_progress` | "
          "`recovery` | **PSS_recovery_progress** |")
    print("|---|---:|---:|---|---|---|---|")
    for k in order:
        a = arms[k]
        m = a.get("_meta", {})
        c = a.get("composite") or {}
        comp = a.get("components", {})
        gp = str(m.get("goal_provenance", "—")).split(" (")[0][:34]
        print(f"| `{k}` | {a.get('n_evaluations')} | {a.get('n_episodes')} | "
              f"{gp} | {ci(comp.get('ego_progress', {}).get('ci'))} | "
              f"{ci(comp.get('recovery', {}).get('ci'))} | "
              f"**{ci(c.get('ci'))}** |")
    print()

    print("### Diagnostics\n")
    print("| arm | recovery defined frac | mean along-track end (m) | "
          "mean cross-track end (m) | wallclock s | planner calls | "
          "rollout steps |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for k in order:
        m = arms[k].get("_meta", {})
        print(f"| `{k}` | {f(r['_recovery_defined_fraction'].get(k))} | "
              f"{f(r['_along_track_end_mean_m'].get(k), 3)} | "
              f"{f(r['_cross_track_end_mean_m'].get(k), 3)} | "
              f"{m.get('wallclock_s')} | {m.get('planner_calls')} | "
              f"{m.get('rollout_steps_executed')} |")
    print()

    def get(x, y, key="PSS_recovery_progress"):
        k, sgn = f"{x}__minus__{y}", 1.0
        if k not in r["paired"]:
            k, sgn = f"{y}__minus__{x}", -1.0
        d = (r["paired"].get(k) or {}).get(key)
        if not d:
            return None
        if sgn < 0:
            return {"delta": -d["delta"], "lo": -d["hi"], "hi": -d["lo"],
                    "separated": d["separated"]}
        return d

    for ref in ("v4_oracle", "cv_holdv0", "v1_tactical_oracle"):
        if ref not in arms:
            continue
        print(f"### Paired vs `{ref}` (episode-cluster bootstrap, B=2000, "
              f"identical rows)\n")
        print("| arm | Δ `ego_progress` | Δ `recovery` | **Δ PSS** |")
        print("|---|---|---|---|")
        for k in order:
            if k == ref:
                continue
            print(f"| `{k}` − `{ref}` | "
                  f"{ci(get(k, ref, 'ego_progress'), 4, True)} | "
                  f"{ci(get(k, ref, 'recovery'), 4, True)} | "
                  f"{ci(get(k, ref), 4, True)} |")
        print()

    print("### Pre-registered gates\n```")
    print(json.dumps(r["PREREGISTERED_GATES"], indent=2, default=str))
    print("```")


if __name__ == "__main__":
    main()
