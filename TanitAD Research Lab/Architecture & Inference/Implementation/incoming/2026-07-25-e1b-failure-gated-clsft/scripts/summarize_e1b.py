"""Render E1B_RESULTS.md tables straight from e1b_eval_result.json.

Every number in the report is generated here, never re-typed — the program has
retracted five claims in one session that were copied from prose.
"""
import json
import sys
from pathlib import Path

J = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
KM = str(max(J["horizons_K"]))
KMIN = str(min(J["horizons_K"]))

# Windows stdout is cp1252 and would die on the delta glyph — write UTF-8 to a
# file and let the caller read it.
_OUT = open(sys.argv[2], "w", encoding="utf-8") if len(sys.argv) > 2 else sys.stdout
_print = print


def print(*a, **k):  # noqa: A001
    k.setdefault("file", _OUT)
    _print(*a, **k)


def iv(d):
    if d is None or "mean" not in d:
        return "n/a"
    return f"{d['mean']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}]"


def dl(d):
    if d is None or "delta" not in d:
        return "n/a"
    sep = "**SEP**" if d.get("separated") else "not sep"
    return f"{d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] {sep}"


def n_of(d):
    if d is None or "n_windows" not in d:
        return "?"
    return f"{d['n_windows']}w/{d['n_episodes']}ep"


print("### PRIMARY — junction corridor-departure @ K=185\n")
p = J["PRIMARY_junction_corridor_departure_K" + KM]
print("| arm | rate | n |")
print("|---|---|---|")
print(f"| base | {iv(p['base'])} | {n_of(p['base'])} |")
print(f"| FT (E1b CL-SFT) | {iv(p['ft'])} | {n_of(p['ft'])} |")
print(f"| **paired Δ(FT−base)** | {dl(p['paired_delta_ft_minus_base'])} | "
      f"{n_of(p['paired_delta_ft_minus_base'])} |")
print(f"\nverdict string: `{J['verdict']}`\n")

FIELDS = [("dep", "corridor-departure rate"), ("win_dep", "window departure rate"),
          ("peak_xte", r"peak \|XTE\| (m)"), ("mean_xte", r"mean \|XTE\| (m)"),
          ("peak_dpsi", r"peak \|Δψ\| (deg)"), ("ade2s", "closed ADE@2s (m)"),
          ("ood_peak", "OOD peak ratio"), ("ood_mean", "OOD mean ratio"),
          ("out_env", "frac windows out of envelope")]

for K in (KM, KMIN):
    blk = J.get("closed_loop_K" + K, {})
    print(f"\n### Closed loop @ K={K} ({int(K) * 0.1:.1f} s)\n")
    print("| stratum | metric | base | FT | paired Δ(FT−base) | n |")
    print("|---|---|---|---|---|---|")
    for st in ("overall", "junction", "longitudinal"):
        s = blk.get(st, {})
        for f, lab in FIELDS:
            if f not in s:
                continue
            print(f"| {st} | {lab} | {iv(s[f]['base'])} | {iv(s[f]['ft'])} | "
                  f"{dl(s[f]['paired_delta_ft_minus_base'])} | "
                  f"{n_of(s[f]['paired_delta_ft_minus_base'])} |")

print("\n### GUARDRAIL (a) — open-loop ADE@2s (PAIRED)\n")
g = J["GUARDRAIL_a_openloop_ade2s"]
print("| arm | ADE@2s (m) |")
print("|---|---|")
print(f"| base | {iv(g['base'])} ({n_of(g['base'])}) |")
print(f"| FT | {iv(g['ft'])} ({n_of(g['ft'])}) |")
print(f"| **paired Δ** | {dl(g['paired_delta_ft_minus_base'])} |")

print("\n### GUARDRAIL (b) — open-loop anchor block (PAIRED)\n")
gb = J["GUARDRAIL_b_openloop_anchor_block"]
print("| metric | base | FT | paired Δ(FT−base) |")
print("|---|---|---|---|")
for f in ("anchor_acc", "anchor_ce", "anchor_traj_l1"):
    print(f"| {f} | {iv(gb[f]['base'])} | {iv(gb[f]['ft'])} | "
          f"{dl(gb[f]['paired_delta_ft_minus_base'])} |")

print("\n### GUARDRAIL SUMMARY\n")
print("```json")
print(json.dumps(J["GUARDRAIL_SUMMARY"], indent=2))
print("```")

for key, name in (("M1_lateral_split_openloop", "OPEN LOOP"),
                  ("M1_lateral_split_closedloop_K" + KM,
                   f"CLOSED LOOP (inside K={KM})")):
    m = J.get(key)
    if not m:
        continue
    print(f"\n### M1 lateral/longitudinal split — {name}\n")
    print(f"windows {m['n_windows']} · GT identity check "
          f"max|Δ| = {m['_gt_identity_max_abs_diff']} · axis convention "
          f"{m['_axis_convention'].get('verified')}\n")
    print("| mode | metric | base | FT | paired Δ(FT−base) |")
    print("|---|---|---|---|---|")
    for mode in ("ego", "frenet"):
        b = m[mode]
        for f in ("ade_over_knots", "cross_abs@2s", "cross_p90@2s", "along_abs@2s"):
            d = b["paired_delta_ft_minus_base"].get(f)
            print(f"| {mode} | {f} | {iv(b['base'][f])} | {iv(b['ft'][f])} | {dl(d)} |")
    for mode in ("ego", "frenet"):
        b = m[mode]
        for nm in ("base", "ft"):
            es = b[nm]["energy_share"]
            print(f"| {mode} | energy share ({nm}) | "
                  f"lon {es['longitudinal_share_of_squared_error']:.4f} / "
                  f"lat {es['lateral_share_of_squared_error']:.4f} | | |")
    for mode in ("ego", "frenet"):
        for nm in ("base", "ft"):
            t = m[mode][nm]["cross_tail@2s"]
            print(f"| {mode} | \\|XTE\\|@2s tail ({nm}) | mean {t['mean']} p50 {t['p50']} "
                  f"p90 {t['p90']} p99 {t['p99']} max {t['max']} | "
                  f"frac>1.75m {t['frac_beyond_m']['1.75']} | |")

print("\n### provenance\n")
for k in ("base_ckpt", "base_step", "ft_ckpt", "ft_step", "val_dir",
          "n_episodes", "corridor_halfwidth_m", "junction_deg", "stride",
          "wall_s", "_estimator"):
    print(f"- `{k}` = {J.get(k)}")
