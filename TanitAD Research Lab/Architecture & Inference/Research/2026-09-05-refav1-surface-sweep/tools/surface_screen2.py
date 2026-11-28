#!/usr/bin/env python3
"""H-REFAV1-SURFACE-1 Stage A (v2) — the cost-weight screen, decomposed, ZERO GPU.

v1 conflated two things and its deliberate-regression control failed for an informative
reason, both fixed here:

* **`sign(kappa_plan) == sign(kappa_GT)` counts a STRAIGHT plan as a wrong turn.** It is
  decomposed into `turns_at_all` and `sign_correct_GIVEN_it_turns`, so "does not turn" is
  never reported as "turns the wrong way".
* **`W_JERK` is INERT on this candidate box**: 24 of the 26 candidates have `jerk_raw` identically
  0 (measured; only `seed0` and `proposal` carry jerk). So raising `W_JERK` cannot force `cv`, and
  the v1 regression arm was mis-designed for this instrument. The regression that DOES bite the box
  is `W_KAPPA` -> very large, which must drive kappa == 0 everywhere.

⭐ It also asks the question the weights cannot answer: **is the decoded GOAL pointing the right
way?** That is `sign(kappa_seed)` vs `sign(kappa_GT)` and involves no weights at all.

⛔ Screen only: 26 injected/seeded candidates vs a 300x30 search. Decides no committed outcome.
⛔ Weights are chosen on the SELECT half only (raw/split.json, banked before this ran).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

TRIVIAL = ("cv", "decel_1.5")
TURN_TOKENS = (6, 7)          # v7.2 lateral goal tokens TURN_L / TURN_R
GT_TURN_KAPPA = 1e-3          # |kappa| above which the GT window counts as turning
PLAN_TURN_KAPPA = 1e-9        # |kappa| above which the selected candidate counts as turning


def main():
    panel_p, dump_d, split_p, out_p, md_p, stack, tanit = sys.argv[1:8]
    sys.path.insert(0, stack)
    sys.path.insert(0, tanit)
    import torch
    from taniteval import four_families as ff

    P = json.load(open(panel_p, encoding="utf-8"))
    names, rows = P["box_names"], P["rows"]
    n = len(rows)
    idx = {nm: i for i, nm in enumerate(names)}
    goal = np.array([r["goal_ccos"] for r in rows], dtype=np.float64)
    jerk = np.array([r["jerk_raw"] for r in rows], dtype=np.float64)
    kapp = np.array([r["kap_raw"] for r in rows], dtype=np.float64)
    kmean = np.array([r["box_kmean"] for r in rows], dtype=np.float64)
    a0 = np.array([r["box_a0"] for r in rows], dtype=np.float64)
    p_ep = np.array([int(r["ep"]) for r in rows])
    p_t = np.array([int(r["t"]) for r in rows])

    files = sorted(glob.glob(os.path.join(dump_d, "ep*.npz")))
    G, EID, WS, CIX, GLAT, GLON = [], [], [], [], [], []
    for f in files:
        with np.load(f) as z:
            m = len(z["ws"])
            G.append(z["g"]); WS.append(z["ws"])
            CIX.append(np.repeat(np.asarray(z["clip_index"]).ravel()[0], m))
            EID.append(np.array([os.path.basename(f)[:-4]] * m))
        with np.load(os.path.join(dump_d, "decisions", os.path.basename(f))) as z:
            GLAT.append(z["goal_lat_cl"]); GLON.append(z["goal_lon_cl"])
    G = np.concatenate(G); EID = np.concatenate(EID); WS = np.concatenate(WS)
    CIX = np.concatenate(CIX); GLAT = np.concatenate(GLAT); GLON = np.concatenate(GLON)
    key = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(CIX, WS))}
    order = np.array([key[(int(e), int(t))] for e, t in zip(p_ep, p_t)])
    if len(set(order.tolist())) != n:
        raise SystemExit("REFUSED: panel<->dump join is not one-to-one")
    G, EID, GLAT, GLON = G[order], EID[order], GLAT[order], GLON[order]

    Gg = ff._seq_geometry(torch.as_tensor(G).float(), 0.2)
    # yaw_rate is one step shorter than speed here, so kappa is MEAN yaw-rate / MEAN speed
    gt_kappa = (Gg["yaw_rate"].mean(1) / Gg["speed"].mean(1).clamp_min(0.5)).numpy()
    gt_accel = Gg["accel"].mean(1).numpy()

    S = json.load(open(split_p, encoding="utf-8"))
    masks = {"SELECT": np.isin(EID, S["SELECT"]), "SCORE": np.isin(EID, S["SCORE"]),
             "ALL": np.ones(n, bool)}

    turn_goal = np.isin(GLAT, TURN_TOKENS)
    seed_k = kmean[:, idx["seed0"]]
    trivial_cols = np.array([idx[t] for t in TRIVIAL])
    base = (12.859430084830139, 32.148575212075350)

    # ---------------------------------------------------------------- #
    # ⭐ GOAL QUALITY — no weights involved at all                       #
    # ---------------------------------------------------------------- #
    gq = {}
    for mn, mk in masks.items():
        gt_turn = np.abs(gt_kappa) > GT_TURN_KAPPA
        gsel, gtt, sk = turn_goal[mk], gt_turn[mk], seed_k[mk]
        gk = gt_kappa[mk]
        goal_turns = np.abs(sk) > PLAN_TURN_KAPPA
        both = gtt & goal_turns
        gq[mn] = {
            "n": int(mk.sum()),
            "n_GT_turning": int(gtt.sum()),
            "n_goal_is_TURN_token": int(gsel.sum()),
            "n_goal_has_nonzero_kappa": int(goal_turns.sum()),
            "goal_turns_when_GT_turns": float(goal_turns[gtt].mean()) if gtt.sum() else None,
            "goal_turns_when_GT_straight": float(goal_turns[~gtt].mean()) if (~gtt).sum() else None,
            "GOAL_SIGN_CORRECT_given_both_turn": (float((np.sign(sk[both]) == np.sign(gk[both])).mean())
                                                  if both.sum() else None),
            "n_both_turn": int(both.sum()),
            "chance": 0.5,
        }

    def readouts(wj, wk, mk):
        tot = goal + wj * jerk + wk * kapp
        win = tot.argmin(1)
        w = win[mk]
        km, am = kmean[mk, w], a0[mk, w]
        gk, ga = gt_kappa[mk], gt_accel[mk]
        gtt = np.abs(gk) > GT_TURN_KAPPA
        plan_turns = np.abs(km) > PLAN_TURN_KAPPA
        both = gtt & plan_turns
        tg = turn_goal[mk]
        sk = seed_k[mk]
        tg_k = tg & (np.abs(sk) > PLAN_TURN_KAPPA)
        return {
            "n": int(mk.sum()),
            "frac_nontrivial": float((~np.isin(w, trivial_cols)).mean()),
            "frac_plan_turns": float(plan_turns.mean()),
            # decomposed, so "did not turn" is never read as "turned wrongly"
            "plan_turns_when_GT_turns": float(plan_turns[gtt].mean()) if gtt.sum() else None,
            "PLAN_SIGN_CORRECT_given_both_turn": (float((np.sign(km[both]) == np.sign(gk[both])).mean())
                                                  if both.sum() else None),
            "n_both_turn": int(both.sum()),
            "plan_follows_goal_sign_on_turn_goals": (float((np.sign(km[tg_k]) == np.sign(sk[tg_k])).mean())
                                                     if tg_k.sum() else None),
            "n_turn_goal_windows": int(tg_k.sum()),
            "mean_abs_accel_err_vs_gt": float(np.abs(am - ga).mean()),
            "share_proposal": float((w == idx["proposal"]).mean()),
            "share_seed0": float((w == idx["seed0"]).mean()),
            "share_cv": float((w == idx["cv"]).mean()),
            "winner_hist": {names[c]: int((w == c).sum()) for c in np.unique(w)},
        }

    grid = [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 1e3, 1e6]
    res = {"tool": "surface_screen2.py", "stage": "A (screen, ZERO GPU)",
           "limit": ("26 injected/seeded candidates vs a 300x30 search: this SCREENS weight "
                     "settings for Stage B and decides no committed outcome"),
           "metric": "ccos", "compensated_base": {"W_JERK": base[0], "W_KAPPA": base[1]},
           "n_windows": n,
           "split": {k: int(v.sum()) for k, v in masks.items()},
           "MEASURED_W_JERK_is_inert_on_this_box": {
               "candidates_with_any_nonzero_jerk": [nm for i, nm in enumerate(names)
                                                    if (jerk[:, i] > 1e-12).any()],
               "n_jerk_free_candidates": int(sum((jerk[:, i] <= 1e-12).all() for i in range(len(names)))),
               "proposal_jerk_raw_median": float(np.median(jerk[:, idx["proposal"]])),
               "seed0_jerk_raw_median": float(np.median(jerk[:, idx["seed0"]])),
               "consequence": ("W_JERK cannot re-rank 24 of the 26 candidates at all. Its ONLY "
                               "function on this surface is to penalise the iCEM PROPOSAL "
                               "(median jerk_raw 2.47) and, weakly, the seed."),
           },
           "GOAL_QUALITY_no_weights_involved": gq,
           "arms": {}}

    def add(nm, wj, wk):
        res["arms"][nm] = {"W_JERK": wj, "W_KAPPA": wk,
                           "SELECT": readouts(wj, wk, masks["SELECT"]),
                           "SCORE": readouts(wj, wk, masks["SCORE"])}

    for m in grid:
        add(f"S-JERK-x{m:g}", base[0] * m, base[1])
    for m in grid:
        add(f"S-KAPPA-x{m:g}", base[0], base[1] * m)
    for mj in (0.0, 1e-4, 1e-2, 1.0, 1e3):
        for mkk in (0.0, 1e-4, 1e-2, 1.0, 1e3):
            add(f"S-BOTH-j{mj:g}-k{mkk:g}", base[0] * mj, base[1] * mkk)
    add("S-SHIPPED", 0.02, 0.05)
    add("S-COMPENSATED", *base)
    add("S-REG (regression: W_KAPPA x1e6)", base[0], base[1] * 1e6)

    reg = res["arms"]["S-REG (regression: W_KAPPA x1e6)"]["SELECT"]
    res["deliberate_regression_control"] = {
        "designed_for_this_instrument": ("W_KAPPA -> huge must drive kappa == 0 on ~100 % of "
                                         "windows. (W_JERK -> huge CANNOT, because 24 of 26 "
                                         "candidates are jerk-free — measured above.)"),
        "frac_plan_turns": reg["frac_plan_turns"],
        "PASS": reg["frac_plan_turns"] < 0.01,
    }
    cands = [(nm, a) for nm, a in res["arms"].items()
             if not nm.startswith("S-REG")
             and a["SELECT"]["frac_nontrivial"] >= 0.10
             and a["SELECT"]["plan_follows_goal_sign_on_turn_goals"] is not None]
    best = max(cands, key=lambda kv: (kv[1]["SELECT"]["plan_follows_goal_sign_on_turn_goals"],
                                      kv[1]["SELECT"]["frac_plan_turns"])) if cands else None
    res["selection"] = {
        "rule": ("SPEC.md §2 as written: maximise turn-goal kappa-sign agreement on SELECT, "
                 "subject to frac_nontrivial >= 0.10"),
        "n_eligible": len(cands), "chosen": best[0] if best else None,
        "chosen_weights": {"W_JERK": best[1]["W_JERK"], "W_KAPPA": best[1]["W_KAPPA"]} if best else None,
        "chosen_SELECT": best[1]["SELECT"] if best else None,
        "chosen_SCORE_held_out": best[1]["SCORE"] if best else None,
    }
    json.dump(res, open(out_p, "w", encoding="utf-8"), indent=1, default=float)

    L = [f"### Stage A (v2) — ccos, n = {n} (SELECT {int(masks['SELECT'].sum())} / "
         f"SCORE {int(masks['SCORE'].sum())})", "",
         "#### GOAL QUALITY — no weights involved", "",
         "| split | n | GT turning | goal turns when GT turns | goal turns when GT straight | "
         "**goal sign correct \\| both turn** | n both | chance |", "|---|---|---|---|---|---|---|---|"]
    for mn in ("SELECT", "SCORE", "ALL"):
        g = gq[mn]
        f = lambda x: "—" if x is None else f"{x:.3f}"
        L.append(f"| {mn} | {g['n']} | {g['n_GT_turning']} | {f(g['goal_turns_when_GT_turns'])} | "
                 f"{f(g['goal_turns_when_GT_straight'])} | **{f(g['GOAL_SIGN_CORRECT_given_both_turn'])}** | "
                 f"{g['n_both_turn']} | 0.500 |")
    L += ["", "#### the weight screen (SELECT)", "",
          "| arm | W_JERK | W_KAPPA | nontriv | plan turns | turns when GT turns | "
          "**plan sign correct \\| both turn** (n) | follows goal sign | prop | seed | cv |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for nm, aa in res["arms"].items():
        s = aa["SELECT"]
        f = lambda x: "—" if x is None else f"{x:.3f}"
        L.append(f"| {nm} | {aa['W_JERK']:.4g} | {aa['W_KAPPA']:.4g} | {s['frac_nontrivial']:.3f} | "
                 f"{s['frac_plan_turns']:.3f} | {f(s['plan_turns_when_GT_turns'])} | "
                 f"**{f(s['PLAN_SIGN_CORRECT_given_both_turn'])}** ({s['n_both_turn']}) | "
                 f"{f(s['plan_follows_goal_sign_on_turn_goals'])} | {s['share_proposal']:.3f} | "
                 f"{s['share_seed0']:.3f} | {s['share_cv']:.3f} |")
    L += ["", f"**selection (SELECT only):** {res['selection']['chosen']} — "
              f"{json.dumps(res['selection']['chosen_weights'])}",
          f"**deliberate regression (W_KAPPA x1e6):** frac_plan_turns "
          f"{reg['frac_plan_turns']:.4f} ⇒ "
          f"{'PASS' if res['deliberate_regression_control']['PASS'] else 'FAIL'}",
          "", f"**W_JERK is inert on this box:** only "
              f"{res['MEASURED_W_JERK_is_inert_on_this_box']['candidates_with_any_nonzero_jerk']} "
              f"carry any jerk; the other "
              f"{res['MEASURED_W_JERK_is_inert_on_this_box']['n_jerk_free_candidates']} are "
              f"jerk-free by construction."]
    md = "\n".join(L)
    print(md)
    open(md_p, "w", encoding="utf-8").write(md + "\n")


if __name__ == "__main__":
    main()
