# -*- coding: utf-8 -*-
"""Render ORACLE_RECHECK.md's tables from raw/oracle_mirror.json (+ WP-A's banked
oracle_s0/t1/t2 for the side-by-side).  Reads only; writes nothing."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
J = r"C:\Users\Admin\wpa-readout\oracle_recheck\raw\oracle_mirror.json"
WPA = (r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Lab"
       r"\Architecture & Inference\Research\2026-09-07-wpa-readout-localisation\raw")
d = json.load(open(J, encoding="utf-8"))
A, B = d["arms"], d["paired_bootstrap"]
RUNGS = ["16x40", "8x20", "16x4", "4x40", "4x8", "4x4", "4x2", "1x1"]

bank = {}
for f, ts in (("oracle_s0.json", 0), ("oracle_s0_t1.json", 1), ("oracle_s0_t2.json", 2)):
    try:
        bank[ts] = json.load(open(f"{WPA}\\{f}", encoding="utf-8"))["arms"]
    except Exception as e:                                    # G: mount can flap
        print(f"<!-- could not read {f}: {e} -->")

print("### n / d / controls\n")
print(f"- rows fit/val/**test** {d['n_fit_rows']}/{d['n_val_rows']}/**{d['n_test_rows']}**"
      f" · **d = {d['n_addressable_cells_both_senses']}** addressable BEV cells"
      f" · scored cells n = **{d['n_scored_cells']:,}**")
c = d["controls"]
print(f"- base rate **{c['base_rate_literal']} = {d['base_rate_test']:.9f}**")
print(f"- tie-group constant control **{c['constant_ap_tiegroup']:.12f}**, "
      f"all-zero **{c['allzero_ap_tiegroup']:.12f}** — "
      f"`equals_base_rate_exactly = {c['equals_base_rate_exactly']}`")
print(f"- s6's NAIVE form on the same all-zero control **{c['allzero_ap_NAIVE_s6_form']:.9f}** "
      f"= **{c['naive_tie_bias_pct']:+.3f} %**")
print(f"- mirror exactness max|colf_prog+colf_wpa-(TW-1)| = **{d['mirror_exactness_max_abs_err']:.1e}**\n")

print("### The ladder, both senses, tie-group AP, train seed 0\n")
print("| rung | az cols | rows | CORRECTED (`prog`) | MIRRORED (`wpa`) | Δ prog−wpa | paired CI95 | separated |")
print("|---|---|---|---|---|---|---|---|")
for r in RUNGS:
    p = A.get(f"orc_{r}@prog|s0"); w = A.get(f"orc_{r}@wpa|s0")
    b = B.get(f"prog_minus_wpa@{r}|s0")
    if not (p and w):
        continue
    ci = f"[{b['lo']:+.4f}, {b['hi']:+.4f}]" if b else "—"
    sep = ("**YES**" if b["separated"] else "no") if b else "—"
    print(f"| {r} | {p['az_cols']} | {p['rows']} | **{p['ap_test']:.4f}** | "
          f"{w['ap_test']:.4f} | {b['delta']:+.4f} | {ci} | {sep} |")
for nm in ("pos_only",):
    p = A.get(f"{nm}@prog|s0"); w = A.get(f"{nm}@wpa|s0")
    if p:
        print(f"| `{nm}` (control) | 40 | 16 | {p['ap_test']:.4f} | {w['ap_test']:.4f} | "
              f"{p['ap_test']-w['ap_test']:+.4f} | — | — |")
print()

print("### Replicate floor (same flags, same sense, TRAINING seed 0 vs 1)\n")
print("| rung | prog s0 | prog s1 | |Δ| | wpa s0 | wpa s1 | |Δ| | sense effect |Δ| |")
print("|---|---|---|---|---|---|---|---|")
for r in RUNGS:
    p0, p1 = A.get(f"orc_{r}@prog|s0"), A.get(f"orc_{r}@prog|s1")
    w0, w1 = A.get(f"orc_{r}@wpa|s0"), A.get(f"orc_{r}@wpa|s1")
    if not (p0 and p1 and w0 and w1):
        continue
    print(f"| {r} | {p0['ap_test']:.4f} | {p1['ap_test']:.4f} | "
          f"{abs(p0['ap_test']-p1['ap_test']):.4f} | {w0['ap_test']:.4f} | "
          f"{w1['ap_test']:.4f} | {abs(w0['ap_test']-w1['ap_test']):.4f} | "
          f"**{abs(p0['ap_test']-w0['ap_test']):.4f}** |")
print()

print("### Mutation control — the real defect deliberately reintroduced\n")
print("| arm | build sense | read sense | AP | vs its matched full arm |")
print("|---|---|---|---|---|")
for k, ref in (("xwire_16x40|s0", "orc_16x40@prog|s0"),
               ("xwire_8x20|s0", "orc_8x20@prog|s0"),
               ("xwire_16x40_r|s0", "orc_16x40@wpa|s0")):
    x = A.get(k)
    if not x:
        continue
    print(f"| `{k.split('|')[0]}` | {x['build']} | {x['read']} | **{x['ap_test']:.4f}** | "
          f"{x['ap_test']-A[ref]['ap_test']:+.4f} vs `{ref.split('|')[0]}` "
          f"({A[ref]['ap_test']:.4f}) |")
print(f"| `pos_only` (floor) | — | — | {A['pos_only@prog|s0']['ap_test']:.4f} | — |")
print(f"| all-zero (base rate) | — | — | {d['base_rate_test']:.4f} | — |")
print()

print("### Paired episode-cluster bootstrap, all pairs\n")
print("| contrast | Δ AP | CI95 | separated | p(Δ>0) |")
print("|---|---|---|---|---|")
for k, v in B.items():
    print(f"| `{k}` | {v['delta']:+.4f} | [{v['lo']:+.4f}, {v['hi']:+.4f}] | "
          f"{'**YES**' if v['separated'] else 'no'} | {v['p_delta_gt0']:.4f} |")
print()

if bank:
    print("### Against WP-A's banked (mirrored-address, NAIVE-AP) ladder\n")
    print("| rung | WP-A banked seed-mean | this run CORRECTED s0 | this run MIRRORED s0 |")
    print("|---|---|---|---|")
    for r in RUNGS:
        vs = [bank[t][f"orc_{r}"]["ap_test"]
              for t in sorted(bank) if f"orc_{r}" in bank[t]]
        if not vs:
            continue
        m = sum(vs) / len(vs)
        p = A.get(f"orc_{r}@prog|s0"); w = A.get(f"orc_{r}@wpa|s0")
        print(f"| {r} | {m:.4f} (n={len(vs)} seeds) | "
              f"{p['ap_test']:.4f} | {w['ap_test']:.4f} |")
    vs = [bank[t]["pos_only"]["ap_test"] for t in sorted(bank) if "pos_only" in bank[t]]
    print(f"| `pos_only` | {sum(vs)/len(vs):.4f} (n={len(vs)}) | "
          f"{A['pos_only@prog|s0']['ap_test']:.4f} | {A['pos_only@wpa|s0']['ap_test']:.4f} |")

print()
print("### Against the full 16x40 grid (CORRECTED address, tie-group AP, seed 0)\n")
full = A["orc_16x40@prog|s0"]["ap_test"]
print("| pooled to | AP | AP retained | AP cost | banked (mirrored, naive-AP) retained |")
print("|---|---|---|---|---|")
BANKED = {"8x20": 0.3374, "4x8": 0.2077, "4x4": 0.1583}
for r, lbl in (("8x20", "8x20 (refcv5)"), ("4x8", "4x8 (v7-tiny)"),
               ("4x4", "4x4 (flagship)")):
    v = A[f"orc_{r}@prog|s0"]["ap_test"]
    old = BANKED[r] / 0.4713
    print(f"| {lbl} | {v:.4f} | **{100*v/full:.1f} %** | {full/v:.2f}x | {100*old:.1f} % |")

print()
print("### Axis attribution (CORRECTED address, seed 0)\n")
c4 = A["orc_16x4@prog|s0"]["ap_test"]
r4 = A["orc_4x40@prog|s0"]["ap_test"]
b2 = A["orc_8x20@prog|s0"]["ap_test"]
print(f"- azimuth only 40 -> 4 (`16x4`): {full:.4f} -> {c4:.4f} = **{100*(c4/full-1):+.1f} %** "
      f"(banked: -57.0 %)")
print(f"- elevation only 16 -> 4 (`4x40`): {full:.4f} -> {r4:.4f} = **{100*(r4/full-1):+.1f} %** "
      f"(banked: -34.6 %)")
print(f"- ratio **{(1-c4/full)/(1-r4/full):.2f}x** (banked: 1.65x)")
print(f"- matched-factor datum: both axes 2x (`8x20`) = {100*(b2/full-1):+.1f} % "
      f"(banked: -28.4 %)")
