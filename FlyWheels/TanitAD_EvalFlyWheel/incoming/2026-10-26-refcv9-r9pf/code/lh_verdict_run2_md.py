"""lh_verdict_run2_md.py -- renders result/R9PF_LH_VERDICT_RUN2.json (lh_r9pf_run2.py) as result/R9PF_LH_VERDICT_RUN2.md.  Reads only; no number is
computed here except formatting.  NEW file (EvalFlyWheel run 2, 2026-10-26).   python -X utf8 lh_verdict_run2_md.py [json] [md] [--next <file>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9pf_common as RC  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
SRC = Path(args[0]) if args else RC.RESULT / "R9PF_LH_VERDICT_RUN2.json"
DST = Path(args[1]) if len(args) > 1 else RC.RESULT / "R9PF_LH_VERDICT_RUN2.md"
NEXT = Path(sys.argv[sys.argv.index("--next") + 1]) if "--next" in sys.argv else None
DEV = Path(sys.argv[sys.argv.index("--dev") + 1]) if "--dev" in sys.argv else None


def f(c) -> str:
    if not isinstance(c, dict) or c.get("delta") is None:
        return "n/a"
    return (f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" + (" SEP" if c.get("separated") else "")
            + f" ({c['mean_new']:.4f} vs {c['mean_ref']:.4f}, n={c['n_windows']}/{c['n_episodes']} ep)")


v = json.loads(SRC.read_text(encoding="utf-8"))
C, Bs, ctl = v["cells"], v["bars"], v["controls"]
L = [f"# R9PF_LH_VERDICT_RUN2 -- {v['word']}", "",
     f"**label:** {v['label']}. The run-1 record `result/R9PF_LH_VERDICT.{{json,md}}` (VOID precondition) is kept unchanged.", "",
     f"**tier:** {v['tier']}  ", f"**estimator:** {v['estimator']}  ", f"**intervals:** {v['interval_rule']}  ",
     "**questions answered:** another draw of EPISODES (the bootstrap); another TRAINING run = R9 profile fit seeds 0/1/2 (PA-R; one LH fit, one "
     "refcv8 trunk seed -- the LH fit and the trunk are NOT varied); another INFERENCE run = seeds 0 / 1 reported apart.", "",
     f"**prereg:** PREREG_R9PF_LH sha256 {v['prereg']['sha256']} (sealed 16:02:32 Berlin). **Forking-paths disclosure (prereg s0):** designed after "
     "reading R9-PF's provisional bars, its EXPLORATORY ORACLE / SOFT reads and LONHEAD-C's verdict, all on eval139; a further eval139 look in the "
     "progress-decision family; the multiplicity is paid by the 97.5 % separations. No further relabel / decode variant of this family is admissible "
     "on eval139 after this read.", "",
     "**precondition (s2):** R9-PF K-OFF (run 1) = {}; R9-PF K-EQ as written (AMENDMENT1, CUDA vs CPU) = {} (VOID stands for run 1); K-EQ under "
     "AMENDMENT2 (device-matched, post-failure) = {} -> satisfied in the AMENDMENT2 labelled sense = {}.".format(
         ctl["K-PRECONDITION"]["K-OFF_run1"], ctl["K-PRECONDITION"]["K-EQ_run1_as_written"], ctl["K-PRECONDITION"]["K-EQ_AMEND2"],
         ctl["K-PRECONDITION"]["PASS"]), "",
     f"**committed outcome (s5):** {v['committed_outcome_s5']}", "",
     f"**failed controls:** {v['failed_controls'] or 'none'}", "",
     "## Bars per arm (an arm PASSES iff PA-1a, PA-1b, PA-2, PA-G and PA-R hold at both inference seeds)", "",
     "| arm | PA-1a | PA-1b | PA-2 | PA-G (fs0/1/2) | PA-R | PASS | class |", "|---|---|---|---|---|---|---|---|"]
for a, b in Bs.items():
    L.append(f"| {a} | {b['PA-1a']} | {b['PA-1b']} | {b['PA-2']} | {b['PA-G']} (fs0 alone {b['PA-G_fs0_only_beside']}) | {b['PA-R']} | {b['PASS']} | {b['class']} |")
L += ["", "R9DER-ARG is the K-DER control (it must NOT pass PA-1a), not a candidate.", "",
      "### PA-R detail (each cell = holds at BOTH inference seeds)", "", "| arm | fit seed | PA-1a | PA-1b | PA-2 | PA-G | PA-G rows |", "|---|---|---|---|---|---|---|"]
for a, b in Bs.items():
    for fs, x in b["per_fit_seed"].items():
        L.append(f"| {a} | {fs} | {x['PA-1a']} | {x['PA-1b']} | {x['PA-2']} | {x['PA-G']} | {len(x['PA-G_worse_rows'])} |")
for a in ("R9LH-ARG", "R9LH-SOFT", "R9DER-ARG"):
    L += ["", f"### {a}, fit seed 0 -- the bar cells (S6 compliant; paired vs the reference on the same draw)", "",
          "| cell | interval | seed 0 | seed 1 |", "|---|---|---|---|"]
    t = f"{a}_fs0"
    for lab, key in (("PA-1a MD vs H2CN_DEPLOY8 (sep BELOW)", "MD_vs_H8_975"), ("PA-1a FD vs H2CN_DEPLOY8 (NOT sep above)", "FD_vs_H8_95"),
                     ("PA-1b FD vs FACTORED_DEPLOY8 (sep BELOW)", "FD_vs_FX_975"), ("PA-1b MD vs FACTORED_DEPLOY8 (NOT sep above)", "MD_vs_FX_95"),
                     ("PA-2 S2 ADE vs RETIME_DEPLOY8", "ADE_S2_vs_D8_975"), ("PA-2 S6 ADE vs RETIME_DEPLOY8", "ADE_S6_vs_D8_975"),
                     ("PA-2 S2 E1 vs ha0_ext_pose", "ADE_S2_vs_HP_975"), ("PA-2 S6 E3 vs ha0_ext_pose", "ADE_S6_vs_HP_975"),
                     ("beside: MD vs H2CN_DEPLOY8 at 95 %", "MD_vs_H8_95"), ("PA-G exceedance vs H2CN_DEPLOY8 (NOT sep above)", "EXC_vs_H8_95")):
        lvl = "97.5 %" if key.endswith("_975") else "95 %"
        L.append(f"| {lab} | {lvl} | {f(C[t]['s0'].get(key))} | {f(C[t]['s1'].get(key))} |")
    L.append(f"| rates MD / FD / exceedance | -- | {C[t]['s0']['MD_rate']:.4f} / {C[t]['s0']['FD_rate']:.4f} / {C[t]['s0']['EXC_rate']:.4f} | "
             f"{C[t]['s1']['MD_rate']:.4f} / {C[t]['s1']['FD_rate']:.4f} / {C[t]['s1']['EXC_rate']:.4f} |")
    rows = Bs[a]["per_fit_seed"]["fs0"]["PA-G_worse_rows"]
    L += ["", f"PA-G rows separated WORSE than H2CN_DEPLOY8 ({a} fs0; {len(rows)}):"] + ([f"* `{r}`" for r in rows] or ["* none"])
    for fs in ("fs1", "fs2"):
        rr = Bs[a]["per_fit_seed"][fs]["PA-G_worse_rows"]
        if rr:
            L += [f"{fs} ({len(rr)}):"] + [f"* `{r}`" for r in rr]
L += ["", "## K-DER (attribution: the decision information, not the profiles, moves MD)", "",
      f"R9DER-ARG PA-1a at 97.5 %: {ctl['K-DER']['PA-1a_97.5']}; stricter 95 % reading: {ctl['K-DER']['PA-1a_95_stricter']} -> K-DER PASS = {ctl['K-DER']['PASS']}.", "",
      "## Four families -- fit seed 0 minus H2CN_DEPLOY8 (compliant sets; 95 % paired interval; worse = separated in the bad direction)", "",
      "| surface | family | metric | R9LH-ARG s0 | R9LH-ARG s1 | R9LH-SOFT s0 | R9LH-SOFT s1 |", "|---|---|---|---|---|---|---|"]
for sf in ("S2", "S6"):
    for k, c0 in C["R9LH-ARG_fs0"]["s0"][f"fam_{sf}_vs_H8_95"].items():
        cells = []
        for a in ("R9LH-ARG", "R9LH-SOFT"):
            for s in ("s0", "s1"):
                c = C[f"{a}_fs0"][s][f"fam_{sf}_vs_H8_95"][k]
                bad = (c["lower_is_better"] and c.get("separated") and c["lo"] > 0) or ((not c["lower_is_better"]) and c.get("separated") and c["hi"] < 0)
                good = (c["lower_is_better"] and c.get("separated") and c["hi"] < 0) or ((not c["lower_is_better"]) and c.get("separated") and c["lo"] > 0)
                cells.append(f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" + (" **WORSE**" if bad else (" better" if good else "")))
        L.append(f"| {sf} | {c0['family']} | {k} (n={c0['n_windows']}/{c0['n_episodes']} ep) | " + " | ".join(cells) + " |")
L += ["", "**TACTICAL -- the regime decision vs the GT regime (S6 compliant; rows GT decel / cruise / accel, cols decision):**", "",
      "| arm (fs0) | seed | confusion | accuracy | decel recall | false decel on GT cruise | n |", "|---|---|---|---|---|---|---|"]
for a in ("R9LH-ARG", "R9LH-SOFT", "R9DER-ARG"):
    for s in ("s0", "s1"):
        d = C[f"{a}_fs0"][s]["TACTICAL_decision_vs_GT_regime_S6_compliant"]
        L.append(f"| {a} | {s} | {d['rows_GT_regime_cols_decision (0 decel, 1 cruise, 2 accel)']} | {d['accuracy']:.4f} | {d['decel_recall']:.4f} | "
                 f"{d['false_decel_on_GT_cruise']:.4f} | {d['n_S6_compliant']} |")
L += ["", "R9LH-SOFT's row is its posterior's argmax; its plan uses the full posterior. The decision is shared by every fit seed (the profiles differ).", "",
      "**STRATEGIC:** n/a -- refcv8 runs `--no-strategic`; there is no strategic head to score (n = 0). **Distance-keeping (headway / TTC):** n/a -- "
      "no lead-agent join in the eval139 panel (n = 0).", "",
      "## Beside references (read from their own records; not re-scored)", "",
      "| reference | seed 0 MD / FD rate | seed 1 MD / FD rate | source |", "|---|---|---|---|"]
for nm, src in (("R9PF_fs0", "R9PF_PA_BARS.json"), ("R9PF_fs1", "R9PF_PA_BARS.json"), ("R9PF_fs2", "R9PF_PA_BARS.json"),
                ("R9PF-TL_fs0", "R9PF_PA_BARS.json"), ("R9PF-TL_fs1", "R9PF_PA_BARS.json"), ("R9PF-TL_fs2", "R9PF_PA_BARS.json"),
                ("H2CN_DEPLOY8", "R9PF_PA_BARS.json"), ("FX", "R9PF_PA_BARS.json (FACTORED_DEPLOY8)"), ("D8", "R9PF_PA_BARS.json (RETIME_DEPLOY8)")):
    r0, r1 = v["beside_references"]["s0"][nm], v["beside_references"]["s1"][nm]
    L.append(f"| {nm} | {r0['MD_rate']:.4f} / {r0['FD_rate']:.4f} | {r1['MD_rate']:.4f} / {r1['FD_rate']:.4f} | {src} |")
for fs in ("fs0", "fs1", "fs2"):
    o0 = v["beside_references"]["s0"]["ORACLE_EXPLORATORY_inadmissible"][fs]
    o1 = v["beside_references"]["s1"]["ORACLE_EXPLORATORY_inadmissible"][fs]
    L.append(f"| ORACLE (GT regime, R9PF_{fs} profiles; EXPLORATORY, inadmissible) | {o0['MD_rate']:.4f} / {o0['FD_rate']:.4f} | "
             f"{o1['MD_rate']:.4f} / {o1['FD_rate']:.4f} | R9PF_X_ATTRIB.json |")
l0, l1 = v["beside_references"]["s0"]["LHS_FACT8"], v["beside_references"]["s1"]["LHS_FACT8"]
L.append(f"| LHS_FACT8 (LONHEAD-C) | MD vs H8 {l0['MD_vs_H8']}; FD vs H8 {l0['FD_vs_H8']} | MD vs H8 {l1['MD_vs_H8']}; FD vs H8 {l1['FD_vs_H8']} | LONHEAD_VERDICT.json |")
L += ["", "## Controls", "", "| control | PASS | detail |", "|---|---|---|"]
for k, c in ctl.items():
    if isinstance(c, dict) and "PASS" in c:
        det = {x: y for x, y in c.items() if x not in ("PASS", "pins", "detail", "per_fit_seed") and not isinstance(y, (dict, list))}
        L.append(f"| {k} | {c['PASS']} | {json.dumps(det, default=str)[:220]} |")
ds = v["decision_stats"]
L += ["", "## n", "", f"* seed 0: {json.dumps(ds['s0'])}", f"* seed 1: {json.dumps(ds['s1'])}", ""]
if DEV is not None and DEV.is_file():
    L += ["## Deviations (disclosed)", ""] + [f"* **{d['id']}** {d['text']}" for d in json.loads(DEV.read_text(encoding="utf-8"))] + [""]
if NEXT is not None and NEXT.is_file():
    L += ["## Next lever (RULE ZERO)", "", NEXT.read_text(encoding="utf-8").strip(), ""]
DST.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"[lhmd] wrote {DST}", flush=True)
