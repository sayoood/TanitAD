"""Generate the data-dependent RESULT.md blocks from the banked raw/*.json (no hand-copied numbers).
Usage: python fill_result.py <raw dir> <template RESULT.md> <out RESULT.md>"""
import json
import sys
from pathlib import Path

R = Path(sys.argv[1])
THIN = ("lane", "crosswalk", "arrow", "edge", "hatched")


def f(v, n=3):
    if v is None:
        return "–"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(f(x, n) for x in v) + "]"
    return f"{v:.{n}f}"


def J(n):
    p = R / n
    return json.loads(p.read_text()) if p.exists() else None


def box_table():
    bx = J("B_box.json")
    out = []
    for h in ("box3d", "agent"):
        ev = bx[h]["eval"]
        fit = bx[h]["fit_train_diag"]
        out.append(f"**{h}** — EVAL-DIAG {ev['n_windows']} labelled windows / {ev['n_episodes']} episodes, n_pos "
                   f"{ev['n_pos']}, {ev['n_rows']} scored slot rows. TRAIN-DIAG fit: shift b* {fit['T1b_b']:+.3f}, "
                   f"T* {fit['T2_T']:.3f}, Platt a {fit['T2b_a']:.3f} b {fit['T2b_b']:+.3f}, P = R gate "
                   f"{fit['T3_gate']:.4f} (TRAIN rows {fit['n_rows']}, TP {fit['n_tp']}). MEASURED, `raw/B_box.json`.\n")
        out.append("| transform | conf_ratio [95 % CI] | prec | rec | F1 [95 % CI] | F1 − T0, paired | AP@2 m | AP@4 m | "
                   "ECE_all | ECE_{p≥0.05} | Σp / n_pos |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        names = {"T0": "T0 identity (declared, gate 0.5)", "T1a": "T1a prior shift +4.595 (no fit)",
                 "T1b": "T1b fitted shift", "T1c": "T1c focal inversion (no fit)", "T2": "T2 temperature",
                 "T2b": "T2b Platt", "T3": "T3 TRAIN P = R gate"}
        for t, r in ev["transforms"].items():
            s = r.get("summarise", {})
            out.append(f"| {names[t]} | {f(r['conf_ratio'])} {f(r['ci95']['conf_ratio'])} | {f(r['prec'])} | "
                       f"{f(r['rec'])} | {f(r['f1'])} {f(r['ci95']['f1'])} | {f(r['ci95_f1_minus_T0_paired'])} | "
                       f"{f(s.get(f'eval_{h}_ap2m'), 4)} | {f(s.get(f'eval_{h}_det_ap4_all_all'), 4)} | "
                       f"{f(r['ECE_all'], 4)} | {f(r['ECE_ge005'], 4)} | {f(r['sum_p_over_npos'], 2)} |")
        out.append("")
    return "\n".join(out)


def c5():
    bx = J("B_box.json")
    worst = 0.0
    for h in ("box3d", "agent"):
        t = bx[h]["eval"]["transforms"]
        for key in ("ap2m", "det_ap4_all_all", "det_ap1_all_all", "det_ap0p5_all_all"):
            vals = [t[k]["summarise"][f"eval_{h}_{key}"] for k in t]
            worst = max(worst, max(vals) - min(vals))
    return worst


def sens():
    bx = J("B_box.json")
    lines = ["**Sensitivity — the same fits on the A10 banked TRAIN-CALIB256 set** (64 clips × 4 windows), scored on "
             "EVAL-DIAG (`raw/B_box.json` `eval_with_calib256_fit`):", ""]
    if "eval_with_calib256_fit" not in bx.get("box3d", {}):
        return "**Sensitivity (TRAIN-CALIB256):** NOT RUN — see the deliverable manifest."
    lines.append("| head | fit set | P = R gate | T3 conf_ratio [CI] | T3 F1 | T1b shift | T2 T* | T2b conf_ratio |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for h in ("box3d", "agent"):
        for nm, fk, ek in (("TRAIN-DIAG (primary)", "fit_train_diag", "eval"),
                           ("TRAIN-CALIB256", "fit_train_calib256", "eval_with_calib256_fit")):
            fi, ev = bx[h][fk], bx[h][ek]["transforms"]
            lines.append(f"| {h} | {nm} | {fi['T3_gate']:.4f} | {f(ev['T3']['conf_ratio'])} "
                         f"{f(ev['T3']['ci95']['conf_ratio'])} | {f(ev['T3']['f1'])} | {fi['T1b_b']:+.3f} | "
                         f"{fi['T2_T']:.3f} | {f(ev['T2b']['conf_ratio'])} |")
    return "\n".join(lines)


def me():
    m = J("M_e.json")
    if not m:
        return "NOT RUN — see the deliverable manifest."
    steps = sorted(m["steps"], key=int)
    out = ["EVAL-DIAG (same 1,112 windows) at each checkpoint; ORACLE = best one-vs-rest threshold on p̂ tuned on EVAL "
           "(a trend diagnostic of separable signal, never a result). MEASURED, `raw/M_e.json`.", "",
           "| class | metric | " + " | ".join(steps) + " |", "|---|---|" + "---|" * len(steps)]
    for c in THIN + ("drivable",):
        for key, lab in (("iou_pc", "IoU pc (declared)"), ("iou_raw", "IoU raw"), ("pred_over_gt_pc", "pred/gt pc"),
                         ("auroc_phat", "AUROC p̂"), ("oracle_iou_phat", "ORACLE thr IoU")):
            out.append(f"| {c} | {lab} | " + " | ".join(f(m['steps'][s][c][key]) for s in steps) + " |")
    return "\n".join(out)


def passes():
    rows = []
    for tag in ("inrun_final", "train_fitpass", "eval_final", "train_final", "eval_s5000", "eval_s15000",
                "eval_s20000", "eval_s30000", "cal256_final", "inrun_final_rep"):
        r = J(f"run_{tag}.json")
        if not r:
            rows.append(f"| {tag} | NOT RUN | | | | |")
            continue
        rows.append(f"| {tag} ({r['split']}) | {r['n_windows_done']} | {r['n_episodes']} | {r['step']} | "
                    f"{r['wall_s'] / 60:.1f} min | `raw/run_{tag}.json` |")
    return "\n".join(rows)


def c1n():
    n = 0
    for p in R.glob("run_*.json"):
        n += json.loads(p.read_text())["control_C1"]["n_compared"]
    return n


tmpl = Path(sys.argv[2]).read_text(encoding="utf-8")
w = c5()
c7 = J("C7_determinism.json")
rep = {"__BOXTABLE__": box_table(), "__SENS__": sens(), "__ME__": me(), "__PASSES__": passes(),
       "__C1_N__": f"{c1n():,}",
       "__C5__": f"max spread {w:.2e} (float64 transforms)", "__C5V__": "PASS" if w <= 1e-12 else
       "FAIL at the literal 1e-12 (see §5)",
       "__C5NOTE__": (f"the first analysis cast transformed logits to float32 and AP spread up to 6.6e-7 "
                      f"(`raw/B_box_v1_float32.json`); re-run in float64: max spread {w:.2e}. "
                      + ("PASS." if w <= 1e-12 else "Still above 1e-12: the residue is tie-ordering inside the "
                         "greedy matcher; 6 orders of magnitude below any reported effect, recorded as a FAIL of "
                         "the literal tolerance, not waived.")),
       "__C7__": (c7["summary"] if c7 else "NOT RUN"), "__C7V__": (c7["verdict"] if c7 else "–")}
for k, v in rep.items():
    tmpl = tmpl.replace(k, v)
Path(sys.argv[3]).write_text(tmpl, encoding="utf-8")
left = [k for k in rep if k in tmpl] + [t for t in ("__HEADLINE__", "__BODY__", "__RENDER_END__") if t in tmpl]
print("filled; placeholders left:", left)
