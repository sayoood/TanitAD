"""Render an early_panel.py output directory as markdown (numbers copied from its JSONs).
usage: python render_early.py <early_dir> <title>"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_battery as SB  # noqa: E402

d = Path(sys.argv[1])
title = sys.argv[2] if len(sys.argv) > 2 else d.name
an = json.load(open(d / "analysis.json", encoding="utf-8"))
print(f"### S2 four families — {title} — {an['n_windows']} windows / {an['n_episodes']} episodes\n")
print(SB.families_table(an))
print("\n#### Distance keeping (LONGITUDINAL, second half)\n")
print(SB.dk_table(an))
print("\n#### Paired cells, S2\n")
print(SB.paired_table(json.load(open(d / "cross_paired.json", encoding="utf-8"))))
print("\n#### TACTICAL: declared heads and the 22-token goal set\n")
print(SB.tactical_tables(json.load(open(d / "tactical_v6.json", encoding="utf-8"))))
a6 = json.load(open(d / "analysis_s6.json", encoding="utf-8"))
print(f"\n### S6 (1–6 s) — {a6['n_windows']} windows / {a6['n_episodes']} episodes\n")
print(SB.families_table(a6, ade_label="ADE 1–6 s"))
print()
print(SB.paired_table(json.load(open(d / "cross_paired_s6.json", encoding="utf-8"))))
acc = json.load(open(d / "acceptance.json", encoding="utf-8"))
vg = json.load(open(d / "void_gates.json", encoding="utf-8"))
pr = json.load(open(d / "panel_record.json", encoding="utf-8"))
print("\n#### VOID gates\n")
print(f"* pairing (g and ha/ha0/ha0_ext bit-identical vs every baseline): {pr['void_gates_3_4']}")
print(f"* STOP: {vg['stop']}")
print(f"* ha0: {vg['ha0']}")
print(f"* profiles: {vg['profiles']}")
print("\n#### Frozen refcv6 acceptance instruments\n")
t = acc.get("tflip", {})
print(f"* T-FLIP: **{t.get('verdict')}** — {json.dumps({k: t.get(k) for k in ('follows_FED_command', 'paired_true_minus_shuffled', 'n_informative_windows', 'n_informative_episodes', 'reason') if k in t}, default=str)[:900]}")
o = acc.get("obedience", {})
print(f"* OBEDIENCE (forced 30 km/h): **{o.get('verdict')}** — obeys {json.dumps(o.get('obeys'), default=str)}; "
      f"n {o.get('n_windows')} / {o.get('n_episodes')} eps; rows with no compliant candidate "
      f"{o.get('rows_with_no_compliant_candidate')}; violation {json.dumps(o.get('violation_ms'))}; "
      f"ADE cost {json.dumps(o.get('ade_cost'), default=str)[:300]}")
z = (acc.get("tzero") or {}).get("summary") or {}
print(f"* T-ZERO (diagnostic, non-turn classes): {json.dumps({k: z.get(k) for k in ('n_classes_summarised', 'median_retained_fraction', 'min_retained_fraction')})}")
