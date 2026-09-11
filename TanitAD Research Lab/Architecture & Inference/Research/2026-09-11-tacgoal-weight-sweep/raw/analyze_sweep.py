# -*- coding: utf-8 -*-
"""D-TACGOAL-2 / PI queue item 10 -- read the --w-tac-goal weight sweep.

Runs POD-SIDE on Thor, emits ONE json to stdout plus a json file. It answers
four questions and no others:

  1. grad_abs_sum on tac_goal_tok_head per weight, and does the zero-weight
     control read EXACTLY 0.0 (with n_grad_none == n_tensors)?
  2. the tactical-goal loss term's magnitude against the PRIMARY (`traj`) --
     the budget question, reported as w * tac_goal / traj.
  3. does the primary objective DEGRADE -- read against the REPLICATE arm's own
     spread, never against zero.
  4. torch.cuda.max_memory_allocated() per arm (the ONLY admissible device
     memory probe on Thor).

Every number carries its n. No CI is computed and none is implied: these are
single short training arms, and the honest floor here is the replicate's own
run-to-run difference, which is what the A_w0 / A_w0_rep pair exists to measure.
"""
import json
import os
import sys

ROOT = "/home/nvidia/experiments/tacgoal-wsweep"
ARMS = [("A_w0", None), ("A_w0_rep", None), ("B_w0p005", 0.005),
        ("C_w0p05", 0.05), ("D_w0p5", 0.5), ("E_w5p0", 5.0)]
HEAD = "tac_goal_tok_head"
PRIMARY = ["traj", "lat", "lon", "cls", "law", "route", "lat_tac", "lon_tac",
           "goal_tac", "goal2s_err_m", "sel_v3", "u0", "loss"]


def rows(tag):
    p = os.path.join(ROOT, tag, "metrics.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def tail_mean(rs, key, n=5):
    """Mean of the LAST n logged values -- a short arm's endpoint is noisy, and
    a single final row is one sample."""
    vals = [r[key] for r in rs if key in r and isinstance(r[key], (int, float))]
    vals = [v for v in vals if "step" in key or True]
    if not vals:
        return None, 0
    t = vals[-n:] if len(vals) >= n else vals
    return sum(t) / len(t), len(t)


def main():
    res = {"root": ROOT, "arms": {}, "n_arms_found": 0}
    for tag, w in ARMS:
        rs = [r for r in rows(tag) if "step" in r and "loss" in r]
        ev = [r for r in rows(tag) if "eval_error" in r
              or any(k.startswith("eval_") for k in r)]
        if not rs:
            res["arms"][tag] = {"weight": w, "status": "NO_ROWS"}
            continue
        res["n_arms_found"] += 1
        gk = "gp_%s_grad_abs_sum" % HEAD
        nk = "gp_%s_n_grad_none" % HEAD
        tk = "gp_%s_n_tensors" % HEAD
        pk = "gp_%s_n_params" % HEAD
        grads = [r[gk] for r in rs if gk in r]
        a = {
            "weight": w,
            "status": "OK",
            "n_log_rows": len(rs),
            "final_step": rs[-1]["step"],
            "wallclock_s": rs[-1].get("elapsed_s"),
            "s_per_step": (round((rs[-1]["elapsed_s"] - rs[0]["elapsed_s"])
                                 / max(1, rs[-1]["step"] - rs[0]["step"]), 3)
                           if len(rs) > 1 else None),
            # --- 1. the gradient -------------------------------------------
            "grad_abs_sum_first": grads[0] if grads else None,
            "grad_abs_sum_last": grads[-1] if grads else None,
            "grad_abs_sum_mean": (sum(grads) / len(grads)) if grads else None,
            "grad_abs_sum_max": max(grads) if grads else None,
            "grad_n_rows": len(grads),
            "grad_all_exactly_zero": (bool(grads)
                                      and all(g == 0.0 for g in grads)),
            "n_grad_none_last": rs[-1].get(nk),
            "n_tensors": rs[-1].get(tk),
            "n_params": rs[-1].get(pk),
            # --- 4. memory --------------------------------------------------
            "cuda_max_mem_gb": max([r["gp_cuda_max_mem_gb"] for r in rs
                                    if "gp_cuda_max_mem_gb" in r] or [None]),
            "eval_rows": len(ev),
        }
        # --- the two structurally-inert heads, as free extra controls -------
        for other in ("core.decoder.offset_head", "scorer.goal_point"):
            k = "gp_%s_grad_abs_sum" % other
            g2 = [r[k] for r in rs if k in r]
            a["ctrl_%s_all_zero" % other] = (bool(g2)
                                             and all(g == 0.0 for g in g2))
            a["ctrl_%s_n_rows" % other] = len(g2)
        # --- 2. the budget ratio -------------------------------------------
        tg = [r["tac_goal"] for r in rs if "tac_goal" in r]
        a["tac_goal_term_last"] = tg[-1] if tg else None
        a["tac_goal_term_mean_tail5"], a["tac_goal_n_tail"] = tail_mean(
            rs, "tac_goal")
        a["tac_goal_n_supervised_mean"], _ = tail_mean(
            rs, "tac_goal_n_supervised", n=len(rs))
        a["tac_goal_n_pos_mean"], _ = tail_mean(
            rs, "tac_goal_n_pos", n=len(rs))
        traj_tail, n_traj = tail_mean(rs, "traj")
        a["traj_tail5"], a["traj_n_tail"] = traj_tail, n_traj
        if w and tg and traj_tail:
            a["w_times_term_over_traj_tail5"] = (
                w * a["tac_goal_term_mean_tail5"] / traj_tail)
            a["w_times_term_last_over_traj_last"] = (
                w * tg[-1] / rs[-1]["traj"])
        # --- 3. the primary -------------------------------------------------
        a["primary_tail5"] = {}
        for k in PRIMARY:
            v, n = tail_mean(rs, k)
            if v is not None:
                a["primary_tail5"][k] = {"mean": v, "n": n}
        res["arms"][tag] = a

    # --- the noise floor: A_w0 vs A_w0_rep, zero levers moved --------------
    base, rep = res["arms"].get("A_w0"), res["arms"].get("A_w0_rep")
    floor = {}
    if base and rep and base.get("status") == "OK" and rep.get("status") == "OK":
        for k in PRIMARY:
            b = base["primary_tail5"].get(k)
            r = rep["primary_tail5"].get(k)
            if b and r:
                d = r["mean"] - b["mean"]
                floor[k] = {"A_w0": b["mean"], "A_w0_rep": r["mean"],
                            "abs_diff": abs(d),
                            "rel_diff": (abs(d) / abs(b["mean"])
                                         if b["mean"] else None)}
    res["replicate_noise_floor"] = floor

    # --- every weighted arm read AGAINST that floor ------------------------
    deltas = {}
    if base and base.get("status") == "OK":
        for tag, w in ARMS:
            if w is None:
                continue
            a = res["arms"].get(tag)
            if not a or a.get("status") != "OK":
                continue
            row = {}
            for k in PRIMARY:
                b = base["primary_tail5"].get(k)
                v = a["primary_tail5"].get(k)
                if not (b and v):
                    continue
                d = v["mean"] - b["mean"]
                f = floor.get(k, {}).get("abs_diff")
                row[k] = {
                    "A_w0": b["mean"], "arm": v["mean"], "delta": d,
                    "floor_abs_diff": f,
                    "exceeds_floor": (abs(d) > f) if f is not None else None,
                    "x_floor": (abs(d) / f) if f else None,
                }
            deltas[tag] = row
    res["primary_delta_vs_control"] = deltas
    return res


if __name__ == "__main__":
    out = main()
    dest = os.path.join(ROOT, "SWEEP_ANALYSIS.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    sys.stdout.write(json.dumps(out, sort_keys=True))
    sys.stderr.write("\nZZWROTE %s\n" % dest)
