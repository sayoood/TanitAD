# -*- coding: utf-8 -*-
"""D-TACGOAL-2 / PI queue item 10 — the BUDGET reading of the --w-tac-goal sweep.

Runs against the banked ``raw/<arm>_metrics.jsonl`` files in this package, so it
is reproducible without Thor.

⛔ THE BUDGET QUESTION IS NOT "IS 0.05 BIG". A weight is meaningless without the
term it multiplies: ``tac_goal`` is a pos-weighted multi-label BCE over 22 tokens
and ``traj`` is an L1 in metres. What is comparable is the CONTRIBUTION each term
makes to the scalar that is actually differentiated, so this script reports
``w x tac_goal`` against three references:

  * ``TRAJ_WEIGHT x traj``     — the primary objective;
  * the tactical-aux budget    — ``0.025 x (lat + lat_tac + lon + lon_tac)``,
    which is ``MANEUVER_WEIGHT`` = 0.1 spread over the four surfaces the
    trainer supervises (``refc_v3_train.py``'s ``(LAT_WEIGHT / 2.0) * (loss_lat
    + loss_lat_tac) + (LON_WEIGHT / 2.0) * (loss_lon + loss_lon_tac)``);
  * the total ``loss``.

⛔ EVERY CONSTANT BELOW IS A LITERAL COPIED FROM ``refc_train.py``, never
imported or re-derived from the trainer — re-running the producer's own
arithmetic and finding agreement measures determinism, not correctness. If the
trainer's constants change, this file must go stale VISIBLY (the assertions in
``stack/tests/test_grad_probe_tacgoal.py`` do not cover them; the values are
restated here with their file:line so a reader can check them by hand).

  refc_train.py:76  TRAJ_WEIGHT       = 1.0
  refc_train.py:77  ANCHOR_CLS_WEIGHT = 1.0
  refc_train.py:78  LAW_WEIGHT        = 0.5
  refc_train.py:79  ROUTE_WEIGHT      = 0.1
  refc_train.py:80  MANEUVER_WEIGHT   = 0.1
  refc_train.py:90  LAT_WEIGHT        = MANEUVER_WEIGHT / 2.0 = 0.05
  refc_train.py:91  LON_WEIGHT        = MANEUVER_WEIGHT / 2.0 = 0.05
"""
import json
import os
import sys

TRAJ_WEIGHT = 1.0
MANEUVER_WEIGHT = 0.1
LAT_WEIGHT = 0.05
LON_WEIGHT = 0.05
HALF_LAT = LAT_WEIGHT / 2.0            # 0.025, the trainer's own /2.0
HALF_LON = LON_WEIGHT / 2.0            # 0.025

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = [("A_w0", None), ("A_w0_rep", None), ("B_w0p005", 0.005),
        ("C_w0p05", 0.05), ("D_w0p5", 0.5), ("E_w5p0", 5.0)]
GK = "gp_tac_goal_tok_head_grad_abs_sum"
NK = "gp_tac_goal_tok_head_n_grad_none"
PK = "gp_tac_goal_tok_head_n_params"
PRIMARY = ["traj", "cls", "law", "route", "lat", "lon", "lat_tac", "lon_tac",
           "goal_tac", "goal2s_err_m", "sel_v3", "u0", "loss"]


def load(tag):
    p = os.path.join(HERE, "%s_metrics.jsonl" % tag)
    if not os.path.exists(p):
        return []
    rs = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if "step" in d and "loss" in d:
                rs.append(d)
    return rs


def tail(rs, key, n=5):
    v = [r[key] for r in rs if isinstance(r.get(key), (int, float))]
    if not v:
        return None, 0
    t = v[-n:] if len(v) >= n else v
    return sum(t) / len(t), len(t)


def main():
    out = {"_rig": "refc_v3_train.py, 108,257,502 params, 400 steps, seed 0",
           "_constants": {"TRAJ_WEIGHT": TRAJ_WEIGHT,
                          "MANEUVER_WEIGHT": MANEUVER_WEIGHT,
                          "per_tactical_surface": HALF_LAT},
           "arms": {}, "budget": {}, "primary_vs_floor": {}}

    for tag, w in ARMS:
        rs = load(tag)
        if not rs:
            out["arms"][tag] = {"weight": w, "status": "NO_ROWS"}
            continue
        g = [r[GK] for r in rs if GK in r]
        a = {"weight": w, "status": "OK", "n_log_rows": len(rs),
             "final_step": rs[-1]["step"],
             "wallclock_s": rs[-1].get("elapsed_s"),
             "n_params": rs[-1].get(PK),
             "n_grad_none_last": rs[-1].get(NK),
             "grad_rows": len(g),
             "grad_first": g[0] if g else None,
             "grad_last": g[-1] if g else None,
             "grad_mean": (sum(g) / len(g)) if g else None,
             "grad_max": max(g) if g else None,
             "grad_all_exactly_zero": bool(g) and all(x == 0.0 for x in g),
             "cuda_max_mem_gb": max([r["gp_cuda_max_mem_gb"] for r in rs
                                     if "gp_cuda_max_mem_gb" in r] or [None])}
        for k in PRIMARY + ["tac_goal", "tac_goal_n_supervised",
                            "tac_goal_n_pos"]:
            m, n = tail(rs, k)
            if m is not None:
                a["tail5_" + k] = m
                a["n_tail_" + k] = n
        out["arms"][tag] = a

        # --- the budget, at the END of the arm ------------------------------
        if w:
            tg, _ = tail(rs, "tac_goal")
            tj, _ = tail(rs, "traj")
            la, _ = tail(rs, "lat")
            lat_, _ = tail(rs, "lat_tac")
            lo, _ = tail(rs, "lon")
            lon_, _ = tail(rs, "lon_tac")
            ls, _ = tail(rs, "loss")
            if None not in (tg, tj, la, lat_, lo, lon_, ls):
                tac_budget = HALF_LAT * (la + lat_) + HALF_LON * (lo + lon_)
                contrib = w * tg
                out["budget"][tag] = {
                    "w": w,
                    "tac_goal_term_tail5": tg,
                    "w_times_term": contrib,
                    "traj_contrib (TRAJ_WEIGHT x traj)": TRAJ_WEIGHT * tj,
                    "tactical_aux_contrib (MANEUVER_WEIGHT budget)": tac_budget,
                    "total_loss_tail5": ls,
                    "ratio_to_primary": contrib / (TRAJ_WEIGHT * tj),
                    "ratio_to_tactical_budget": contrib / tac_budget,
                    "frac_of_total_loss": contrib / ls,
                }

    # --- the replicate floor, then every arm read against it ---------------
    base, rep = out["arms"].get("A_w0"), out["arms"].get("A_w0_rep")
    floor = {}
    if (base and rep and base.get("status") == "OK"
            and rep.get("status") == "OK"):
        for k in PRIMARY:
            b, r = base.get("tail5_" + k), rep.get("tail5_" + k)
            if b is not None and r is not None:
                floor[k] = {"A_w0": b, "A_w0_rep": r, "abs_diff": abs(r - b)}
    out["replicate_noise_floor"] = floor

    for tag, w in ARMS:
        if w is None:
            continue
        a = out["arms"].get(tag)
        if not a or a.get("status") != "OK" or not base:
            continue
        row = {}
        for k in PRIMARY:
            b, v = base.get("tail5_" + k), a.get("tail5_" + k)
            f = floor.get(k, {}).get("abs_diff")
            if b is None or v is None:
                continue
            d = v - b
            row[k] = {"A_w0": b, "arm": v, "delta": d, "floor": f,
                      "x_floor": (abs(d) / f) if f else None,
                      "exceeds_floor": (abs(d) > f) if f is not None else None}
        out["primary_vs_floor"][tag] = row
    return out


if __name__ == "__main__":
    res = main()
    dest = os.path.join(HERE, "BUDGET_ANALYSIS.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print("wrote", dest)
    for tag, _ in ARMS:
        a = res["arms"].get(tag, {})
        if a.get("status") != "OK":
            print("%-10s %s" % (tag, a.get("status")))
            continue
        print("%-10s w=%-7s grad[first/last/max]=%s / %s / %s  "
              "none=%s  params=%s  traj=%s"
              % (tag, a["weight"], a["grad_first"], a["grad_last"],
                 a["grad_max"], a["n_grad_none_last"], a["n_params"],
                 a.get("tail5_traj")))
    sys.exit(0)
