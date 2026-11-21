"""Markdown table of an A2 wrapper-probe artifact (numbers copied from its JSON).
usage: python a2_table.py <g0_A2_wrapper_probe.json>"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
r = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"**A2 wrapper clause = {r['A2_wrapper_clause']}**; G0-A1 as registered = "
      f"{r.get('G0_A1_as_registered')}; **G0-A2 = {r.get('G0_A2')}** "
      f"(reasons: {r.get('G0_A2_reasons') or 'none'}); step {r['step']}, ckpt md5 `{r['ckpt_md5']}`\n")
print("| condition | settings | max Wrapper rel | max Floor rel | terms over bar | W1 detected | W2 detected |")
print("|---|---|---|---|---|---|---|")
for c, v in r["conditions"].items():
    a = v["analysis"]
    st = v["settings"]
    s = (f"cuDNN TF32 {st['cudnn_tf32']}, det {st['deterministic']}, trunk bf16 {st['trunk_bf16']}"
         + (f", chunk {v['chunk_ckpt_used']}" if v.get("chunk_ckpt_used") not in (None, 8) else ""))
    print(f"| {c} | {s} | {a['max_wrapper_rel']:.3e} | {a['max_floor_rel']:.3e} | "
          f"{len(a['terms_over_bar'])} of {a['n_terms']} | "
          f"{len(a.get('W1_detected_terms', [])) if 'W1_detected' in a else '—'} | "
          f"{len(a.get('W2_detected_terms', [])) if 'W2_detected' in a else '—'} |")
p3 = r["conditions"]["P3_fp32_det"]["analysis"]["terms"]
worst = sorted(p3.items(), key=lambda kv: -kv[1]["wrapper"])[:5]
print("\nP3 worst terms (wrapper / floor / bar): " + "; ".join(
    f"`{k}` {v['wrapper']:.2e} / {v['floor']:.2e} / {v['bar']:.2e}" for k, v in worst))
p1 = r["conditions"]["P1_as_run"]["analysis"]["terms"]
worst1 = sorted(p1.items(), key=lambda kv: -kv[1]["wrapper"])[:5]
print("\nP1 worst terms (wrapper / floor): " + "; ".join(
    f"`{k}` {v['wrapper']:.2e} / {v['floor']:.2e}" for k, v in worst1))
