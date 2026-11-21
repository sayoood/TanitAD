"""Markdown table of a G0 artifact (every value copied from the JSON). usage: python g0_table.py <g0.json>"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
rec = json.load(open(sys.argv[1], encoding="utf-8"))
v = rec["verdict"]
print(f"**G0 = {v['G0']}**; reasons: {v['reasons'] or 'none'}; class counts {v['by_class_counts']}; "
      f"medians {v.get('medians')}")
if "verdict_as_run" in rec:
    va = rec["verdict_as_run"]
    print(f"\n(as run, before the COUNT-rule correction: G0 = {va['G0']}; reasons: {va['reasons'] or 'none'})")
wc = rec.get("wrapper_control") or {}
print(f"\nwrapper control (batch 4, micro 1+3, DDIM eps = 0 in both): max rel diff "
      f"{wc.get('max_rel_nonzero')} over {len(wc.get('terms', {}))} terms -> "
      f"{'PASS' if wc.get('pass_rel_1e-3') else 'FAIL'}")
m = rec["model"]
print(f"\nstrict load: missing {len(m['state_dict']['missing'])}, unexpected "
      f"{len(m['state_dict']['unexpected'])}, {m['state_dict']['n_keys']} keys, step "
      f"{m['state_dict']['step']}; param_breakdown equal to config.json: {m['param_breakdown']['equal']} "
      f"(total {m['param_breakdown']['rebuilt']['total']:,}); anchor file vs ckpt buffers max |Δ| "
      f"{max(x['max_abs_diff'] for x in m['anchor_file_vs_ckpt_buffers'].values())}")
print("\n| term | class | in-run (Thor, step %d) | dev-box mean over %d seeds | sd | tolerance | verdict |"
      % (rec["step"], len(rec["by_seed"])))
print("|---|---|---|---|---|---|---|")
for k, t in sorted(v["terms"].items(), key=lambda kv: (kv[1].get("cls", kv[1]["class"]), kv[0])):
    cls = t.get("cls", t["class"])
    tol = t.get("tol", "")
    if cls == "STOCHASTIC" and "pi_lo" in t:
        tol = f"99% PI [{t['pi_lo']:.5g}, {t['pi_hi']:.5g}]"
    elif "rel_dev" in t:
        tol = f"{tol} (rel dev {t['rel_dev']:.2e})"
    mean = t.get("mean")
    sd = t.get("sd")
    print(f"| `{k}` | {cls} | {t['inrun']} | {'' if mean is None else f'{mean:.6g}'} | "
          f"{'' if sd is None else f'{sd:.3g}'} | {tol} | {t['verdict']} |")
m1 = v.get("m1_terms_out_of_tolerance")
if m1 is not None:
    print(f"\n**M1 (lift bank without equalize_bottom_rows, = refcv3_arm.py:1101)**: {len(m1)} SMOOTH "
          f"term(s) leave tolerance:")
    for x in m1[:20]:
        print(f"* `{x['term']}` in-run {x['inrun']} vs M1 {x['m1']}")
