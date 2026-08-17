"""P8 — render the banked results as the report's markdown tables.

Transcribing 40 numbers by hand is how a report drifts from its artifacts. This
prints what §6 quotes, straight from the JSON.
"""
import json, sys
from pathlib import Path
raw = Path(sys.argv[1])
ORDER = [("s02000", 2000), ("s09000", 9000), ("s09250", 9250),
         ("s10000", 10000), ("s11250", 11250)]

def d(x):  return "—" if x is None else f"{x:.4g}"
def ci(v): return f"**{v['delta']:+.3f}** [{v['lo']:+.3f}, {v['hi']:+.3f}]{' sep' if v['separated'] else ' **ns**'}" if v else "—"

def load(lbl):
    p = raw / f"results_{lbl}.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else None

print("### TRAJECTORY — `cells`, all five points, identical windows-construction\n")
print("| stamp | clusters | windows | arm err (m) | C-CONST | C-EPMEAN | K1 Δ vs C-CONST | K5 Δ vs C-EPMEAN | K2 Δ vs C-SHUF | Δ vs C-SHUF-XEP | K3 | verdict |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for lbl, step in ORDER:
    r = load(lbl)
    if not r:
        print(f"| `v6F-SW-30k@{step}` | — | — | NOT RUN | | | | | | | | |"); continue
    a = r["per_arm"]["cells"]; e = a["lead_gap_abs_err_m"]
    P = r["paired"]
    print(f"| `{r['run_stamp']}` | {r['n_bootstrap_clusters']} | {r['n_scored_windows']} | "
          f"**{e['mean']:.3f}** [{e['lo']:.3f}, {e['hi']:.3f}] | "
          f"{r['per_arm']['C-CONST']['lead_gap_abs_err_m']['mean']:.3f} | "
          f"{r['per_arm']['C-EPMEAN']['lead_gap_abs_err_m']['mean']:.3f} | "
          f"{ci(P.get('cells vs C-CONST'))} | {ci(P.get('cells vs C-EPMEAN'))} | "
          f"{ci(P.get('cells vs cells__C-SHUF'))} | {ci(P.get('cells vs cells__C-SHUF-XEP'))} | "
          f"{r['verdict']['cells']['K3_recall']:.4f} | {r['verdict']['cells']['admissible_as']} |")

print("\n### DIAGNOSTICS per point\n")
print("| stamp | median err | mean predicted gap | mean GT gap | oracle median (null floor 0.790) | head params |")
print("|---|---|---|---|---|---|")
for lbl, step in ORDER:
    r = load(lbl)
    if not r: continue
    a = r["per_arm"]["cells"]
    print(f"| `{r['run_stamp']}` | {a['median_abs_err_m']:.3f} | {a['mean_pred_gap_m']:.3f} | "
          f"{a['mean_gt_gap_m']:.3f} | {a['_diag_oracle_slot_abs_err_m']['median']:.3f} | "
          f"{r['arms']['cells']['n_params']:,} |")

for lbl, title in [("tok11250", "THE `tokens` CONTROL — matched windows, one cache"),
                   ("s11250_nq32", "n_slot_queries SENSITIVITY — 32 (inherited) vs 74 (fitted)"),
                   ("NULLCONTROL", "PIPELINE NULL — random latents")]:
    r = load(lbl)
    print(f"\n### {title}\n")
    if not r:
        print("NOT RUN\n"); continue
    print(f"clusters {r['n_bootstrap_clusters']} · windows {r['n_scored_windows']} · "
          f"C-CONST {r['per_arm']['C-CONST']['lead_gap_abs_err_m']['mean']:.3f} · "
          f"C-EPMEAN {r['per_arm']['C-EPMEAN']['lead_gap_abs_err_m']['mean']:.3f} · "
          f"D-RULE {r.get('d_rule')}\n")
    print("| arm | memory | err (m) | median | K1 vs C-CONST | K5 vs C-EPMEAN | K2 vs C-SHUF | vs C-SHUF-XEP | K3 | verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for arm in r["verdict"]:
        a = r["per_arm"][arm]; e = a["lead_gap_abs_err_m"]; P = r["paired"]
        print(f"| `{arm}` | {r['arms'][arm]['memory_shape']} | **{e['mean']:.3f}** [{e['lo']:.3f}, {e['hi']:.3f}] | "
              f"{a['median_abs_err_m']:.3f} | {ci(P.get(f'{arm} vs C-CONST'))} | {ci(P.get(f'{arm} vs C-EPMEAN'))} | "
              f"{ci(P.get(f'{arm} vs {arm}__C-SHUF'))} | {ci(P.get(f'{arm} vs {arm}__C-SHUF-XEP'))} | "
              f"{r['verdict'][arm]['K3_recall']:.4f} | {r['verdict'][arm]['admissible_as']} |")
