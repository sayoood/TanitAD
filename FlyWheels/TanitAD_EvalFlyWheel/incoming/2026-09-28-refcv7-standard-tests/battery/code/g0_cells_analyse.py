"""Decompose a `tac_goal_conf_bce` difference into VALIDITY FLIPS + a continuous remainder (2026-10-04).

    python g0_cells_analyse.py --cells D:/refcv7_eval_kit/battery/g0cells_step30000_cpu/cells.json \
        --g0 D:/refcv7_eval_kit/battery/step30000/g0.json --out <analysis.json>

NOT a gate. Inputs: the per-cell capture of a numerics-only arm (`g0_cells_cpu_r7.py`: trunk fp32 on
CPU) and G0's own replay (eager bf16 on the RTX 4060: per-batch values, seed 0) plus the in-run row.

For every batch b it asks: is  T_bf16(b) - T_arm(b)  the sum of single-cell flips of the arm's
NEAR-THRESHOLD supervised cells? A flip of cell i (validity decision crosses 0) changes the batch
value by exactly (2*correct_i - 1) * c_i * w_i / sum(w_b) (softplus(c) - softplus(-c) = c), with
correct_i / c_i read from the arm. Every subset of the |l| <= --max-abs-logit cells is enumerated
(capped), the best-matching subset and its residual are reported next to the SECOND best (a match
that is not unique is labelled AMBIGUOUS), and the residual is compared with the continuous term's
own movement (`tacv6_goal_bce`, which has no threshold). The same is done for the 8-batch MEAN
against the in-run value (rounded to 5 dp: +-5e-6 on the mean).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g0_refcv7 as G  # noqa: E402


def near_cells(call: dict, max_abs: float) -> tuple[list, float]:
    cells, s = G._cells_flat(call)
    out = []
    n_tok = len(call["goal_logits"][0])
    for i, (l, c, y, w) in enumerate(cells):
        if w > 0 and abs(l) <= max_abs and y in (0.0, 1.0):
            corr = G._correct(l, y)
            out.append({"cell": i, "row": i // n_tok, "tok": i % n_tok, "logit": l, "conf": c,
                        "y": y, "correct_arm": corr,
                        "flip_delta_batch": (2.0 * corr - 1.0) * c * w / max(s, 1.0)})
    return out, s


def best_subsets(cands: list, target: float, key: str, cap: int = 18) -> dict:
    cs = sorted(cands, key=lambda d: abs(d["logit"]))[:cap]
    res = []
    for r in range(len(cs) + 1):
        for sub in itertools.combinations(range(len(cs)), r):
            tot = sum(cs[j][key] for j in sub)
            res.append((abs(target - tot), tot, sub))
    res.sort(key=lambda t: t[0])
    b = res[0]
    second = res[1] if len(res) > 1 else None
    return {"target": target, "n_candidates": len(cs), "best_residual": b[0], "best_sum": b[1],
            "best_cells": [cs[j] for j in b[2]],
            "second_residual": None if second is None else second[0],
            "second_cells": None if second is None else [cs[j]["cell"] for j in second[2]],
            "no_flip_residual": abs(target)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--g0", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-abs-logit", type=float, default=0.5)
    a = ap.parse_args()
    cj = json.load(open(a.cells, encoding="utf-8"))
    g0 = json.load(open(a.g0, encoding="utf-8"))
    out = {"tool": "g0_cells_analyse.py", "NOT_A_GATE": True, "cells": a.cells, "g0": a.g0,
           "max_abs_logit": a.max_abs_logit, "step": cj.get("step")}
    bad = []
    if cj.get("ckpt_md5") != g0.get("ckpt_md5"):
        bad.append("checkpoint md5 differs")
    if cj.get("perm_sha256") != g0.get("perm_sha256"):
        bad.append("window permutation differs")
    if bad:
        out["REFUSED"] = bad
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
        print(json.dumps(out), flush=True)
        return 0
    pb_gpu = g0["by_seed"]["0"]["per_batch"]
    cells = cj["cells"]
    nb = len(pb_gpu)
    per = {}
    ctl = []
    t_arm, t_gpu, gbce_arm, gbce_gpu = [], [], [], []
    all_c = []
    for b in range(nb):
        call = cells.get(str(b))
        if call is None:
            continue
        rec_cells, s = near_cells(call, a.max_abs_logit)
        arm_v = float(call["tac_goal_conf_bce"])
        ctl.append(abs(G.conf_bce_from_cells(call) - arm_v))
        gpu_v = float(pb_gpu[b]["tacv6_goal_conf_bce"])
        d = gpu_v - arm_v
        t_arm.append(arm_v)
        t_gpu.append(gpu_v)
        gbce_arm.append(float(call["tac_goal_bce"]))
        gbce_gpu.append(float(pb_gpu[b]["tacv6_goal_bce"]))
        bs = best_subsets(rec_cells, d, "flip_delta_batch")
        per[str(b)] = {"n_supervised": sum(1 for (_l, _c, _y, w) in G._cells_flat(call)[0] if w > 0),
                       "sum_w": s, "arm": arm_v, "bf16_replay": gpu_v, "delta_bf16_minus_arm": d,
                       "goal_bce_arm": gbce_arm[-1], "goal_bce_bf16": gbce_gpu[-1],
                       "goal_bce_rel_move": abs(gbce_gpu[-1] - gbce_arm[-1]) / max(abs(gbce_arm[-1]), 1e-12),
                       "near_threshold_cells": len(rec_cells), "decomposition": bs}
        for c in rec_cells:
            all_c.append({**c, "batch": b, "flip_delta_mean": c["flip_delta_batch"] / nb})
    out["control_cells_vs_trainer_max_abs"] = max(ctl) if ctl else None
    out["per_batch"] = per
    if len(t_arm) == nb:
        m_arm, m_gpu = sum(t_arm) / nb, sum(t_gpu) / nb
        terms = g0["verdict"]["terms"]
        inrun = float(terms["eval_tacv6_goal_conf_bce"]["inrun"])
        out["mean"] = {"arm_fp32_cpu": m_arm, "bf16_replay": m_gpu, "inrun": inrun,
                       "rel_inrun_vs_arm": (inrun - m_arm) / inrun,
                       "rel_inrun_vs_bf16": (inrun - m_gpu) / inrun,
                       "rel_bf16_vs_arm": (m_gpu - m_arm) / m_arm,
                       "goal_bce": {"inrun": terms["eval_tacv6_goal_bce"]["inrun"],
                                    "arm": sum(gbce_arm) / nb, "bf16": sum(gbce_gpu) / nb}}
        out["inrun_vs_arm_decomposition"] = best_subsets(all_c, inrun - m_arm, "flip_delta_mean")
        out["bf16_vs_arm_decomposition_mean"] = best_subsets(all_c, m_gpu - m_arm, "flip_delta_mean")
    row = cj.get("row") or {}
    if row:
        sm = {}
        for k, r in g0["verdict"]["terms"].items():
            if r.get("cls") == "SMOOTH" and k in row and abs(float(r["inrun"])) >= 0.1:
                x = float(r["inrun"])
                sm[k] = {"inrun": x, "bf16": r["mean"], "arm": row[k],
                         "rel_bf16": abs(r["mean"] - x) / abs(x), "rel_arm": abs(row[k] - x) / abs(x),
                         "phi_rel": abs(row[k] - r["mean"]) / abs(x)}
        out["smooth_terms_bf16_vs_arm_vs_inrun"] = sm
        if sm:
            import statistics
            out["smooth_medians"] = {kk: statistics.median(v[kk] for v in sm.values())
                                     for kk in ("rel_bf16", "rel_arm", "phi_rel")}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({k: out.get(k) for k in ("mean", "smooth_medians",
                                              "control_cells_vs_trainer_max_abs")}, indent=1,
                     default=str), flush=True)
    for b, r in per.items():
        dd = r["decomposition"]
        print(f"b{b}: n_sup {r['n_supervised']} near {r['near_threshold_cells']} delta {r['delta_bf16_minus_arm']:+.6f} "
              f"best flips {[(c['cell'], round(c['logit'], 4), round(c['conf'], 3)) for c in dd['best_cells']]} "
              f"resid {dd['best_residual']:.2e} (2nd {dd['second_residual']:.2e}; no-flip {dd['no_flip_residual']:.2e}) "
              f"goal_bce move {r['goal_bce_rel_move']:.1e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
