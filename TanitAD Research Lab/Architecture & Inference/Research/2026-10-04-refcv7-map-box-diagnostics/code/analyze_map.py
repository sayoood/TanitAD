"""M-a .. M-f (SPEC.md sec. 2 and 5) from the banked accumulators. CPU only.

--eval / --train : the *.acc.pt of EVAL-DIAG and TRAIN-DIAG at the final checkpoint, both passed WITH the
                   TRAIN fit (so every decision has exact per-episode counts on both splits);
--fit            : the TRAIN fit JSON (fit_map.py);
--milestones     : step=path pairs of EVAL-DIAG accumulators (no fit) for M-e.
Writes one JSON per diagnostic into --out-dir: M_c.json, M_a.json, M_b.json, M_d.json, M_e.json, M_f.json,
and DECISIONS_map.json (the SPEC sec. 5 rules applied).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_metrics as dm  # noqa: E402

CK = dm.CLASS_KEYS
ALT = ("raw", "thr_phat", "thr_q", "off")
EDGE0 = int(round((0.0 - dm.HIST_LO) / ((dm.HIST_HI - dm.HIST_LO) / dm.HIST_NB)))   # logit 0 <-> p 0.5
INTERVAL = ("episode-cluster bootstrap over the split's episodes, B=1000, 95 % percentile; answers 'another draw "
            "of EPISODES' only -- blind to training-run variance (one seed, no replicate)")


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def arr(d, dec, stat):
    return d["acc"][dec][stat].numpy()            # [n_ep, 8, nb]


def pooled_iou(d, dec, c, b=None):
    sl = (slice(None), c) if b is None else (slice(None), c, b)
    i, p, g = (arr(d, dec, s)[sl] for s in ("inter", "pred", "gt"))
    if b is None:
        i, p, g = i.sum(-1), p.sum(-1), g.sum(-1)
    return i, p, g                                  # per-episode arrays


def iou_block(d, dec, c, b=None, B=1000):
    i, p, g = pooled_iou(d, dec, c, b)
    v = dm.iou(i.sum(), p.sum(), g.sum())
    return {"iou": v, "ci95": dm.bootstrap_iou(i, p, g, B=B) if v is not None else None,
            "n_pred": float(p.sum()), "n_gt": float(g.sum()),
            "pred_over_gt": float(p.sum() / g.sum()) if g.sum() > 0 else None}


def m_c(ev, tr, fit, bk):
    out = {"definition": "pooled IoU and N_pred/N_gt per decision; thr_* = one-vs-rest at the TRAIN-fitted tau*; "
                         "off = argmax(z - ln w + delta_TRAIN); ORACLE = best threshold on EVAL (ceiling only)",
           "interval": INTERVAL, "eval": {}, "train": {}, "oracle_eval_best_threshold": {},
           "frac_gt_cells_score_ge_0p5": {}, "fit": {k: fit[k] for k in fit if k.startswith(("tau_", "delta",
                                                                                        "subsample_mean"))}}
    for split, d in (("eval", ev), ("train", tr)):
        for dec in d["acc"]:
            out[split][dec] = {}
            for c in range(8):
                row = {"all": iou_block(d, dec, c)}
                for bi, b in enumerate(bk):
                    row[b] = iou_block(d, dec, c, bi, B=300)
                out[split][dec][CK[c]] = row
    for split, d in (("eval", ev), ("train", tr)):
        o, f = {}, {}
        for score in ("phat", "q"):
            h = d["hist"][score].numpy()
            o[score], f[score] = {}, {}
            for c in range(8):
                hc = h[c].sum(0)
                i, best = dm.best_threshold(hc[0], hc[1])
                o[score][CK[c]] = {"iou": best, "tau_prob": float(1 / (1 + np.exp(-dm.hist_edges()[i])))}
                pos = hc[1]
                f[score][CK[c]] = float(pos[EDGE0:].sum() / pos.sum()) if pos.sum() else None
        out["oracle_eval_best_threshold" if split == "eval" else "train_best_threshold"] = o
        out["frac_gt_cells_score_ge_0p5"][split] = f
    # paired eval differences vs the declared rule, thin classes, all bands pooled
    diffs = {}
    for c in dm.THIN:
        i0, p0, g0 = pooled_iou(ev, "pc", c)
        diffs[CK[c]] = {}
        for alt in ALT:
            if alt not in ev["acc"]:
                continue
            i1, p1, g1 = pooled_iou(ev, alt, c)
            diffs[CK[c]][alt] = {"iou_alt": dm.iou(i1.sum(), p1.sum(), g1.sum()),
                                 "iou_pc": dm.iou(i0.sum(), p0.sum(), g0.sum()),
                                 "ci95_alt_minus_pc_paired": dm.bootstrap_iou(i1, p1, g1, other=(i0, p0, g0))}
    out["eval_paired_alt_minus_pc_thin"] = diffs
    return out


def m_a(ev, tr, bk):
    out = {"definition": "pooled IoU TRAIN-DIAG vs EVAL-DIAG (same windows positions, different episodes); gap = "
                         "train - eval, unpaired bootstrap (independent episode resamples)", "interval": INTERVAL}
    for dec in ("pc", "raw", "thr_phat", "off"):
        if dec not in ev["acc"] or dec not in tr["acc"]:
            continue
        out[dec] = {}
        for c in range(8):
            row = {}
            for b in [None] + list(range(len(bk))):
                it, pt, gt = pooled_iou(tr, dec, c, b)
                ie, pe, ge = pooled_iou(ev, dec, c, b)
                vt, ve = dm.iou(it.sum(), pt.sum(), gt.sum()), dm.iou(ie.sum(), pe.sum(), ge.sum())
                row["all" if b is None else bk[b]] = {
                    "train": vt, "eval": ve, "gap": (vt - ve) if (vt is not None and ve is not None) else None,
                    "ci95_gap": (dm.bootstrap_iou_unpaired((it, pt, gt), (ie, pe, ge), B=1000 if b is None else 300)
                                 if (vt is not None and ve is not None) else None),
                    "n_gt_train": float(gt.sum()), "n_gt_eval": float(ge.sum())}
            out[dec][CK[c]] = row
    return out


def m_b(ev, tr, bk):
    out = {"definition": "boundary-tolerant P_k/R_k/F_k and IoU_k = F_k/(2-F_k), Chebyshev k cells (0.1 m each), "
                         "supervised cells; k=0 is the exact IoU", "eval": {}, "train": {}}
    for split, d in (("eval", ev), ("train", tr)):
        for dec in d["acc"]:
            out[split][dec] = {}
            for c in range(8):
                row = {}
                for b in [None] + list(range(len(bk))):
                    sl = (slice(None), c) if b is None else (slice(None), c, b)
                    cnt = {s: float(arr(d, dec, s)[sl].sum()) for s in d["acc"][dec]}
                    row["all" if b is None else bk[b]] = dm.tolerant_summary(cnt)
                out[split][dec][CK[c]] = row
    return out


def m_d(ev, tr, bk):
    out = {"definition": "AUROC of logit(score_c) on GT-c vs GT-not-c supervised cells (binned, 4000 bins over "
                         "[-20, 20], ties half); score phat = softmax(z - ln w), q = softmax(z)", "eval": {},
           "train": {}}
    centers = dm.hist_edges() + (dm.HIST_HI - dm.HIST_LO) / dm.HIST_NB / 2
    for split, d in (("eval", ev), ("train", tr)):
        for score in ("phat", "q"):
            h = d["hist"][score].numpy()
            out[split][score] = {}
            for c in range(8):
                row = {}
                for b in [None] + list(range(len(bk))):
                    hc = h[c].sum(0) if b is None else h[c, b]
                    neg, pos = hc[0], hc[1]

                    def q(hh, qq):
                        cs = np.cumsum(hh)
                        return float(centers[np.searchsorted(cs, qq * cs[-1])]) if cs[-1] else None
                    row["all" if b is None else bk[b]] = {
                        "auroc": dm.auroc_from_hist(neg, pos), "n_pos": float(pos.sum()), "n_neg": float(neg.sum()),
                        "pos_logit_p50": q(pos, 0.5), "pos_logit_p90": q(pos, 0.9),
                        "neg_logit_p50": q(neg, 0.5), "neg_logit_p99": q(neg, 0.99)}
                out[split][score][CK[c]] = row
    return out


def m_e(ms, bk):
    out = {"definition": "EVAL-DIAG at each checkpoint: IoU (pc, raw), N_pred/N_gt (pc, raw), AUROC(phat) and the "
                         "ORACLE best one-vs-rest threshold IoU on phat (tuned on EVAL: a trend diagnostic of "
                         "separable signal, NEVER a result)", "steps": {}}
    for step, d in sorted(ms.items()):
        row = {}
        for c in range(8):
            r = {}
            for dec in ("pc", "raw"):
                i, p, g = pooled_iou(d, dec, c)
                r[f"iou_{dec}"] = dm.iou(i.sum(), p.sum(), g.sum())
                r[f"pred_over_gt_{dec}"] = float(p.sum() / g.sum()) if g.sum() else None
                i0, p0, g0 = pooled_iou(d, dec, c, 0)
                r[f"iou_{dec}_0_20"] = dm.iou(i0.sum(), p0.sum(), g0.sum())
            hc = d["hist"]["phat"].numpy()[c].sum(0)
            r["auroc_phat"] = dm.auroc_from_hist(hc[0], hc[1])
            r["oracle_iou_phat"] = dm.best_threshold(hc[0], hc[1])[1]
            r["frac_gt_phat_ge_0p5"] = float(hc[1][EDGE0:].sum() / hc[1].sum()) if hc[1].sum() else None
            row[CK[c]] = r
        out["steps"][str(step)] = row
    return out


def m_f(ev, tr, bk):
    out = {"definition": "(1) GT class fraction of supervised cells per band; (2) lateral run length (cells, 0.1 m) of "
                         "GT thin-class runs per fine row: median / p90 per band; (3) Chebyshev distance (cells) from "
                         "each GT-edge cell to the nearest drivable-boundary cell of the GT and of the PREDICTED (pc) "
                         "drivable mask", "eval": {}, "train": {}}
    for split, d in (("eval", ev), ("train", tr)):
        if "runs" not in d or not d["runs"]:
            continue
        sup = d["sup_cells"].numpy().sum(0)
        gt = arr(d, "pc", "gt").sum(0)
        frac = {CK[c]: {bk[b]: float(gt[c, b] / sup[b]) if sup[b] else None for b in range(len(bk))}
                for c in range(8)}
        frac_all = {CK[c]: float(gt[c].sum() / sup.sum()) for c in range(8)}
        widths = {}
        for c, h in d["runs"].items():
            h = h.numpy()
            row = {}
            for b in [None] + list(range(len(bk))):
                hh = h.sum(0) if b is None else h[b]
                n = hh.sum()
                if n == 0:
                    row["all" if b is None else bk[b]] = {"n_runs": 0}
                    continue
                cs = np.cumsum(hh)
                row["all" if b is None else bk[b]] = {
                    "n_runs": int(n), "median_cells": int(np.searchsorted(cs, 0.5 * n)),
                    "p90_cells": int(np.searchsorted(cs, 0.9 * n)),
                    "frac_runs_le_2_cells": float(hh[:3].sum() / n)}
            widths[CK[int(c)]] = row
        reg = {}
        for k, v in d["reg"].items():
            v = v.numpy()
            tot = v.sum(0)
            reg[k] = {"buckets": list(dm.REG_BUCKETS), "all": (tot / tot.sum()).tolist() if tot.sum() else None,
                      "n_edge_cells": float(tot.sum()),
                      **{bk[b]: ((v[b] / v[b].sum()).tolist() if v[b].sum() else None) for b in range(len(bk))}}
        out[split] = {"gt_fraction_by_band": frac, "gt_fraction_all": frac_all, "line_width": widths,
                      "edge_to_drivable_boundary": reg}
    return out


def decisions(mc, mb, md, ma, me):
    out = {"rules": "SPEC.md sec. 5 (D1-D4), applied verbatim; 'best alternative' = the TRAIN-fitted decision with "
                    "the highest TRAIN IoU for that class (selected on TRAIN, scored on EVAL)"}
    for c in dm.THIN:
        name = CK[c]
        ev_pc = mc["eval"]["pc"][name]["all"]
        tr_ious = {alt: mc["train"][alt][name]["all"]["iou"] for alt in ALT if alt in mc["train"]}
        best = max((a for a in tr_ious if tr_ious[a] is not None), key=lambda a: tr_ious[a], default=None)
        ev_best = mc["eval"][best][name]["all"] if best else None
        diff = mc["eval_paired_alt_minus_pc_thin"][name].get(best, {}) if best else {}
        auroc = md["eval"]["phat"][name]["all"]["auroc"]
        r = {"pc_pred_over_gt": ev_pc["pred_over_gt"], "pc_iou": ev_pc["iou"], "best_alt": best,
             "best_alt_eval_iou": ev_best["iou"] if ev_best else None, "best_alt_ci_minus_pc": diff.get(
                 "ci95_alt_minus_pc_paired"), "auroc_phat": auroc,
             "all_alt_eval_iou": {a: mc["eval"][a][name]["all"]["iou"] for a in ALT if a in mc["eval"]}}
        ci = diff.get("ci95_alt_minus_pc_paired")
        r["D1_decision_rule_suppression"] = bool(
            ev_pc["pred_over_gt"] is not None and ev_pc["pred_over_gt"] < 0.2 and ev_best and ev_best["iou"] is not None
            and ev_pc["iou"] is not None and ev_best["iou"] >= 2 * ev_pc["iou"] and ci is not None and ci[0] > 0)
        any_alt_005 = any((mc["eval"][a][name]["all"]["iou"] or 0) >= 0.05 for a in ALT if a in mc["eval"])
        r["D2_no_signal"] = bool(auroc is not None and auroc < 0.80 and not any_alt_005)
        tb = mb["eval"][best][name]["all"] if best else None
        r["best_alt_IoU0_IoU1_IoU2"] = [tb.get("IoU0"), tb.get("IoU1"), tb.get("IoU2")] if tb else None
        r["D3_localisation_limited"] = bool(
            auroc is not None and auroc >= 0.90 and tb and tb.get("IoU0") is not None and tb["IoU0"] < 0.15
            and tb.get("IoU2") is not None and tb["IoU2"] > 0 and tb["IoU2"] >= 2 * tb["IoU0"])
        # IMPLEMENTATION NOTE (logged in RESULT.md): IoU2 = IoU0 = 0 satisfies "IoU2 >= 2 IoU0" vacuously; the rule
        # is read as requiring a NON-ZERO tolerant IoU, and the vacuous case is flagged instead.
        r["D3_vacuous_both_zero"] = bool(tb and tb.get("IoU0") == 0 and tb.get("IoU2") == 0)
        g = ma.get("thr_phat", {}).get(name, {}).get("all", {})
        r["thr_phat_train_eval_gap"] = [g.get("train"), g.get("eval"), g.get("gap"), g.get("ci95_gap")]
        r["D4_generalisation_gap"] = bool(g.get("gap") is not None and g["gap"] > 0.10 and g.get("ci95_gap")
                                          and g["ci95_gap"][0] > 0)
        r["D4p_optimisation"] = bool(r["D4_generalisation_gap"] and g.get("train") is not None and g["train"] < 0.15)
        r["D4p_train_thr_iou_below_0p15"] = bool(g.get("train") is not None and g["train"] < 0.15)
        if me:
            steps = sorted(me["steps"], key=int)
            first = me["steps"][steps[0]][name]["iou_pc"]
            fell = [s for s in steps if first and me["steps"][s][name]["iou_pc"] is not None
                    and me["steps"][s][name]["iou_pc"] < 0.5 * first]
            r["M_e_first_step_pc_iou_below_half_of_5000"] = fell[0] if fell else None
            r["M_e_trend"] = {s: [me["steps"][s][name]["iou_pc"], me["steps"][s][name]["iou_raw"],
                                  me["steps"][s][name]["auroc_phat"], me["steps"][s][name]["oracle_iou_phat"]]
                              for s in steps}
        out[name] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--milestones", nargs="*", default=[])
    ap.add_argument("--train-mf", default=None, help="TRAIN acc that carries the M-f census (the fit pass)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    ev, tr = load(a.eval), load(a.train)
    fit = json.load(open(a.fit))
    bk = list(ev["band_keys"])
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    mc = m_c(ev, tr, fit, bk)
    (od / "M_c.json").write_text(json.dumps(mc, indent=1))
    ma = m_a(ev, tr, bk)
    (od / "M_a.json").write_text(json.dumps(ma, indent=1))
    mb = m_b(ev, tr, bk)
    (od / "M_b.json").write_text(json.dumps(mb, indent=1))
    md = m_d(ev, tr, bk)
    (od / "M_d.json").write_text(json.dumps(md, indent=1))
    ms = {}
    for kv in a.milestones:
        s, p = kv.split("=", 1)
        ms[int(s)] = load(p)
    me = m_e(ms, bk) if ms else None
    if me:
        (od / "M_e.json").write_text(json.dumps(me, indent=1))
    mf = m_f(ev, load(a.train_mf) if a.train_mf else tr, bk)
    (od / "M_f.json").write_text(json.dumps(mf, indent=1))
    dec = decisions(mc, mb, md, ma, me)
    (od / "DECISIONS_map.json").write_text(json.dumps(dec, indent=1))
    for c in dm.THIN:
        r = dec[CK[c]]
        print(f"[map] {CK[c]}: pc iou {r['pc_iou']} pred/gt {r['pc_pred_over_gt']} best {r['best_alt']} "
              f"{r['best_alt_eval_iou']} auroc {r['auroc_phat']} D1 {r['D1_decision_rule_suppression']} "
              f"D2 {r['D2_no_signal']} D3 {r['D3_localisation_limited']} D4 {r['D4_generalisation_gap']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
