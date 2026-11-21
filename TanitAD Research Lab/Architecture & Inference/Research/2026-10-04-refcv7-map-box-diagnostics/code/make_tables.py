"""Print the RESULT.md tables from the banked raw/*.json (read-only). Usage: python make_tables.py <raw dir>"""
import json
import sys
from pathlib import Path

R = Path(sys.argv[1])
CK = ("nocls", "drivable", "lane", "crosswalk", "arrow", "edge", "hatched", "sidewalk")
THIN = ("lane", "crosswalk", "arrow", "edge", "hatched")


def f(v, n=3):
    if v is None:
        return "null"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(f(x, n) for x in v) + "]"
    return f"{v:.{n}f}"


def load(n):
    p = R / n
    return json.loads(p.read_text()) if p.exists() else None


mc = load("M_c.json")
if mc:
    print("\n## M-c EVAL-DIAG, all bands pooled: IoU [95% CI] and N_pred/N_gt per decision")
    decs = [d for d in ("pc", "raw", "thr_phat", "thr_q", "off") if d in mc["eval"]]
    print("| class | n_gt | " + " | ".join(f"{d} IoU" for d in decs) + " | " + " | ".join(f"{d} pred/gt" for d in decs) + " | ORACLE thr IoU |")
    print("|" + "---|" * (2 + 2 * len(decs) + 1))
    for c in CK:
        e = mc["eval"]
        row = [c, f"{e['pc'][c]['all']['n_gt']:.0f}"]
        row += [f"{f(e[d][c]['all']['iou'])} {f(e[d][c]['all']['ci95'])}" for d in decs]
        row += [f(e[d][c]['all']['pred_over_gt']) for d in decs]
        row += [f(mc["oracle_eval_best_threshold"]["phat"][c]["iou"])]
        print("| " + " | ".join(row) + " |")
    print("\n### M-c TRAIN-DIAG (same layout)")
    decs_t = [d for d in ("pc", "raw", "thr_phat", "thr_q", "off") if d in mc["train"]]
    print("| class | " + " | ".join(f"{d} IoU" for d in decs_t) + " | " + " | ".join(f"{d} pred/gt" for d in decs_t) + " |")
    print("|" + "---|" * (1 + 2 * len(decs_t)))
    for c in CK:
        t = mc["train"]
        print("| " + " | ".join([c] + [f(t[d][c]['all']['iou']) for d in decs_t] + [f(t[d][c]['all']['pred_over_gt']) for d in decs_t]) + " |")
    print("\n### per band, EVAL, thin classes: IoU pc / raw / thr_phat / off (n_gt)")
    bks = [k for k in mc["eval"]["pc"]["lane"] if k != "all"]
    print("| class | " + " | ".join(bks) + " |")
    print("|" + "---|" * (1 + len(bks)))
    for c in THIN + ("drivable",):
        e = mc["eval"]
        cells = []
        for b in bks:
            cells.append(" / ".join(f(e[d][c][b]['iou']) for d in decs if d in ("pc", "raw", "thr_phat", "off"))
                         + f" ({e['pc'][c][b]['n_gt']:.0f})")
        print("| " + c + " | " + " | ".join(cells) + " |")
    print("\n### paired EVAL differences vs pc (thin, all bands)")
    for c, v in mc["eval_paired_alt_minus_pc_thin"].items():
        print(c, {a: (f(x['iou_alt']), f(x['ci95_alt_minus_pc_paired'])) for a, x in v.items()})
    print("\nfrac of GT-c cells with score >= 0.5:", json.dumps(mc["frac_gt_cells_score_ge_0p5"]["eval"], indent=0)[:800])
    print("fit:", {k: (f(v) if isinstance(v, list) else v) for k, v in mc["fit"].items()})

ma = load("M_a.json")
if ma:
    print("\n## M-a TRAIN vs EVAL, all bands: train / eval / gap [CI]")
    for d in ("pc", "raw", "thr_phat", "off"):
        if d not in ma:
            continue
        print(f"\n{d}:")
        for c in CK:
            r = ma[d][c]["all"]
            print(f"| {c} | {f(r['train'])} | {f(r['eval'])} | {f(r['gap'])} {f(r['ci95_gap'])} |")

mb = load("M_b.json")
if mb:
    print("\n## M-b EVAL tolerant IoU_k (k = 0,1,2,5,10 cells) all bands")
    for d in [x for x in ("pc", "raw", "thr_phat", "off") if x in mb["eval"]]:
        print(f"\n{d}:")
        for c in THIN + ("drivable",):
            r = mb["eval"][d][c]["all"]
            ks = sorted(int(k[3:]) for k in r if k.startswith("IoU"))
            print(f"| {c} | " + " | ".join(f"{f(r[f'IoU{k}'])}" for k in ks) + " | P/R k0 " + f"{f(r['P0'])}/{f(r['R0'])}"
                  + (f" | P/R k10 {f(r.get('P10'))}/{f(r.get('R10'))}" if 'P10' in r else "") + " |")
    print("\nper band edge thr_phat / off IoU_k:")
    for d in [x for x in ("thr_phat", "off", "raw") if x in mb["eval"]]:
        for b, r in mb["eval"][d]["edge"].items():
            ks = sorted(int(k[3:]) for k in r if k.startswith("IoU"))
            print(d, b, [f(r[f"IoU{k}"]) for k in ks])

md = load("M_d.json")
if md:
    print("\n## M-d AUROC(phat) EVAL / TRAIN, per band")
    for c in CK:
        e, t = md["eval"]["phat"][c], md["train"]["phat"][c]
        print(f"| {c} | " + " | ".join(f"{f(e[b]['auroc'])}/{f(t[b]['auroc'])}" for b in e) + f" | pos p50 logit {f(e['all']['pos_logit_p50'],2)} neg p99 {f(e['all']['neg_logit_p99'],2)} |")

me = load("M_e.json")
if me:
    print("\n## M-e EVAL-DIAG by checkpoint: iou_pc / iou_raw / auroc / oracle / pred_over_gt pc,raw")
    for c in THIN + ("drivable",):
        print(c)
        for s, row in sorted(me["steps"].items(), key=lambda x: int(x[0])):
            r = row[c]
            print(f"  {s}: {f(r['iou_pc'])} / {f(r['iou_raw'])} / {f(r['auroc_phat'])} / {f(r['oracle_iou_phat'])} / "
                  f"{f(r['pred_over_gt_pc'])},{f(r['pred_over_gt_raw'])} / frac>=0.5 {f(r['frac_gt_phat_ge_0p5'])}")

mf = load("M_f.json")
if mf:
    print("\n## M-f")
    for split in ("eval", "train"):
        if not mf.get(split):
            continue
        print(split, "gt_fraction_all", {k: f(v, 4) for k, v in mf[split]["gt_fraction_all"].items()})
        for c, w in mf[split]["line_width"].items():
            print("  width", c, {b: (v.get("median_cells"), v.get("p90_cells"), v.get("n_runs")) for b, v in w.items()})
        for k, v in mf[split]["edge_to_drivable_boundary"].items():
            print("  reg", k, v["buckets"], f(v["all"]), "n", v["n_edge_cells"])

dec = load("DECISIONS_map.json")
if dec:
    print("\n## map decisions")
    for c in THIN:
        print(c, json.dumps({k: v for k, v in dec[c].items() if k != "M_e_trend"}, default=str)[:900])

bx = load("B_box.json")
if bx:
    for h in ("box3d", "agent"):
        if h not in bx:
            continue
        print(f"\n## B-c {h}: fit (TRAIN-DIAG)", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in bx[h]["fit_train_diag"].items() if k != "T3_train"})
        ev = bx[h]["eval"]
        print(f"n_pos {ev['n_pos']} windows {ev['n_windows']} episodes {ev['n_episodes']} rows {ev['n_rows']}")
        print("| T | conf_ratio [CI] | prec | rec | F1 [CI] | F1-T0 paired CI | AP2m | AP4m | ECE_all | ECE>=.05 | sum p/npos |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for t, r in ev["transforms"].items():
            s = r.get("summarise", {})
            print(f"| {t} | {f(r['conf_ratio'])} {f(r['ci95']['conf_ratio'])} | {f(r['prec'])} | {f(r['rec'])} | "
                  f"{f(r['f1'])} {f(r['ci95']['f1'])} | {f(r['ci95_f1_minus_T0_paired'])} | "
                  f"{f(s.get(f'eval_{h}_ap2m'), 4)} | {f(s.get(f'eval_{h}_det_ap4_all_all'), 4)} | {f(r['ECE_all'], 4)} | "
                  f"{f(r['ECE_ge005'], 4)} | {f(r['sum_p_over_npos'])} |")
        if "eval_with_calib256_fit" in bx[h]:
            print("sensitivity (fit on TRAIN-CALIB256):", {t: (f(r['conf_ratio']), f(r['f1'])) for t, r in bx[h]["eval_with_calib256_fit"]["transforms"].items()},
                  {k: (round(v, 4) if isinstance(v, float) else None) for k, v in bx[h]["fit_train_calib256"].items() if isinstance(v, float)})
        print("train_self:", {t: (f(r['conf_ratio']), f(r['f1'])) for t, r in bx[h]["train_self"]["transforms"].items()})
        t0 = ev["transforms"]["T0"]
        print("T0 hist TP", t0["hist_tp"], "\nT0 hist FP", t0["hist_fp"])
        print("T0 reliability", [(round(x["lo"], 2), x["n"], f(x["conf"]), f(x["acc"])) for x in t0["reliability"]])
    print("\nB-a:", bx.get("B_a_inrun_reproduction"))
