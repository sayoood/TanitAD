#!/usr/bin/env python3
"""RUNG 2 — attribute the deployable gap across the three causes.  ZERO GPU.

The brief's question: of the gap between the **deployable** ``T_blind`` (0.8 s,
the model's own actions) and the **privileged** one (6.5 s, the expert's
actions), how much survives when you (a) fix the readout horizon, (b) replace
the model's actions with held-last-action, (c) both?

⚠️ Pre-registered (PRE_REGISTRATION.md §5): the attribution runs on **``de_N``
and ``ade_0_2s``**, NOT on ``T_blind``. ``T_blind`` is a non-linear functional of
a *contrast*, and four of the six factorial cells have no readout-matched
frozen-last comparator in the dump — attributing on it would confound the
readout with the comparator. ``de_N`` needs no comparator at all, so the 2 x 3
factorial is exactly matched in every cell.

Estimator: paired episode-cluster bootstrap (``taniteval/ci.py``, B = 2000,
seed 0, resampling unit = val episode), identical 599 windows / 596 clusters.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[6]
for _p in (_REPO / "taniteval", _REPO / "stack"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from taniteval import ci as _ci                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tb_rung0 import (DT, GRID, B_BOOT, SEED, WP4, draws_for,   # noqa: E402
                      paired_at, separated_better_interval, single_at,
                      ade_0_2s, t_blind)

#: the 2 x 3 factorial on arm (a) — action source x readout calibration.
CELLS = {
    ("own", "op"): "a_imagination__own",
    ("own", "str"): "a_imagination__own__roSTR",
    ("hold", "op"): "a_imagination__hold",
    ("hold", "str"): "a_imagination__hold__roSTR",
    ("true", "op"): "a_imagination__true",
    ("true", "str"): "a_imagination__true__roSTR",
}
ACTION_LABEL = {"own": "the model's own actions (DEPLOYABLE)",
                "hold": "last observed action held (DEPLOYABLE, no policy)",
                "true": "the expert's true actions (PRIVILEGED)"}
READOUT_LABEL = {"op": "step['op'] — calibrated over 4 steps (0.4 s)",
                 "str": "step['str'] — calibrated over 20 steps (2.0 s)"}
CV = "d_constant_velocity"


BARS_M = {"lane_half_1m": 1.0, "corridor_1p391m": 1.391, "miss_2m": 2.0}


def t_useful(de_arm, draws, bar):
    """Largest N at which the arm's mean ``de_N`` stays below ``bar`` metres,
    with a bootstrapped interval on that HORIZON. Comparator-free — it answers
    the PI's 'how long' directly, with no control arm to mismatch."""
    K = de_arm.shape[1]
    m = np.empty((len(draws), K))
    for i, s in enumerate(draws):
        m[i] = de_arm[s].mean(axis=0)

    def _t(curve):
        ok = curve < bar
        if not ok[0]:
            return 0
        bad = np.flatnonzero(~ok)
        return int(bad[0]) if bad.size else int(ok.size)

    td = np.array([_t(c) for c in m])
    lo, hi = np.percentile(td, [2.5, 97.5])
    return {"bar_m": bar, "T_steps": _t(de_arm.mean(axis=0)),
            "T_s": round(_t(de_arm.mean(axis=0)) * DT, 2),
            "T_ci95_s": [round(float(lo) * DT, 2), round(float(hi) * DT, 2)]}


def _lvl(de, eid, draws, arm):
    return {"ade_0_2s": ade_0_2s(de[arm], eid),
            "de_2s": single_at(de[arm], eid, draws, 20),
            "de_6s": single_at(de[arm], eid, draws, 60),
            "T_useful": {k: t_useful(de[arm], draws, v)
                         for k, v in BARS_M.items()},
            "beats_cv": separated_better_interval(de[arm], de[CV], draws)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    d = torch.load(a.dump, map_location="cpu", weights_only=False)
    de = {k: v.double().numpy() for k, v in d["dense_de_headline"].items()}
    eid = [str(x) for x in d["eid"]]
    draws, n_ep = draws_for(eid)

    res = {"meta": {"n_windows": int(de[CV].shape[0]), "n_episode_clusters": n_ep,
                    "K_max": int(de[CV].shape[1]), "n_boot": B_BOOT, "seed": SEED,
                    "arm_ckpt": d["meta_sweep"].get("arm_ckpt"),
                    "ckpt_step": d["meta_sweep"].get("ckpt_step"),
                    "estimator": "paired_episode_cluster_bootstrap",
                    "why_de_not_T_blind": (
                        "T_blind is a functional of a CONTRAST and 4 of the 6 "
                        "cells have no readout-matched frozen-last comparator "
                        "in the dump; de_N and ade_0_2s need no comparator, so "
                        "every cell is exactly matched")},
           "cells": {}, "factorial": {}, "gap_attribution": {},
           "action_source_dominance": {}}

    # ---- the six cells, levels ------------------------------------------- #
    for (act, ro), arm in CELLS.items():
        res["cells"][f"{act}|{ro}"] = {
            "arm": arm, "action_source": ACTION_LABEL[act],
            "readout": READOUT_LABEL[ro], **_lvl(de, eid, draws, arm)}
    for nm in (CV, "d2_hold_v0", "b_frozenlast__own", "b_frozenlast__hold",
               "b_frozenlast__true"):
        res["cells"][f"ref:{nm}"] = {"arm": nm,
                                     "ade_0_2s": ade_0_2s(de[nm], eid),
                                     "de_2s": single_at(de[nm], eid, draws, 20),
                                     "de_6s": single_at(de[nm], eid, draws, 60)}

    # ---- main effects + interaction --------------------------------------- #
    # READOUT effect within each action source (positive = str is better)
    res["factorial"]["readout_effect_within_action_source"] = {
        act: {"de_2s": paired_at(de[CELLS[(act, "str")]],
                                 de[CELLS[(act, "op")]], draws, 20),
              "de_6s": paired_at(de[CELLS[(act, "str")]],
                                 de[CELLS[(act, "op")]], draws, 60),
              "ade_0_2s_rel_pct": round(float(
                  de[CELLS[(act, "str")]][:, WP4].mean()
                  / de[CELLS[(act, "op")]][:, WP4].mean() - 1.0) * 100, 2)}
        for act in ("own", "hold", "true")}
    # ACTION-SOURCE effect within each readout (positive = the better source)
    res["factorial"]["action_effect_within_readout"] = {
        f"{ro}: own->{alt}": {
            "de_2s": paired_at(de[CELLS[(alt, ro)]], de[CELLS[("own", ro)]],
                               draws, 20),
            "de_6s": paired_at(de[CELLS[(alt, ro)]], de[CELLS[("own", ro)]],
                               draws, 60)}
        for ro in ("op", "str") for alt in ("hold", "true")}
    # INTERACTION: is the readout lever GATED by the action source?
    res["factorial"]["interaction_de_2s"] = {}
    for act in ("hold", "true"):
        d_own = (de[CELLS[("own", "op")]][:, 19]
                 - de[CELLS[("own", "str")]][:, 19])
        d_alt = (de[CELLS[(act, "op")]][:, 19]
                 - de[CELLS[(act, "str")]][:, 19])
        boots = np.array([d_alt[s].mean() - d_own[s].mean() for s in draws])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        res["factorial"]["interaction_de_2s"][f"{act}_minus_own"] = {
            "delta_of_deltas_m": round(float(d_alt.mean() - d_own.mean()), 4),
            "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "separated": bool(lo > 0 or hi < 0),
            "reads": ("positive = the readout swap buys MORE under this action "
                      "source than under the model's own actions, i.e. the "
                      "horizon lever is GATED by policy quality"),
            "n_boot": B_BOOT, "estimator": "paired_episode_cluster_bootstrap"}

    # ---- the brief's gap decomposition, on de@2s -------------------------- #
    base = float(de[CELLS[("own", "op")]][:, 19].mean())     # deployable, as shipped
    ceil = float(de[CELLS[("true", "op")]][:, 19].mean())    # privileged, as shipped
    gap = base - ceil
    fixes = {"(a) fix the READOUT only": CELLS[("own", "str")],
             "(b) fix the ACTIONS only (held-last)": CELLS[("hold", "op")],
             "(c) BOTH": CELLS[("hold", "str")],
             "(d) privileged actions + fixed readout": CELLS[("true", "str")]}
    res["gap_attribution"]["de_2s"] = {
        "deployable_as_shipped_m": round(base, 4),
        "privileged_as_shipped_m": round(ceil, 4),
        "gap_m": round(gap, 4),
        "fixes": {k: {
            "de_2s_m": round(float(de[v][:, 19].mean()), 4),
            "gap_closed_m": round(base - float(de[v][:, 19].mean()), 4),
            "gap_closed_pct": round((base - float(de[v][:, 19].mean()))
                                    / gap * 100, 1),
            "paired_vs_deployable": paired_at(de[v], de[CELLS[("own", "op")]],
                                              draws, 20)}
            for k, v in fixes.items()}}
    # the same on ade_0_2s, the program's own convention
    b2 = float(de[CELLS[("own", "op")]][:, WP4].mean())
    c2 = float(de[CELLS[("true", "op")]][:, WP4].mean())
    res["gap_attribution"]["ade_0_2s"] = {
        "deployable_as_shipped": round(b2, 4),
        "privileged_as_shipped": round(c2, 4), "gap": round(b2 - c2, 4),
        "fixes": {k: {"ade_0_2s": round(float(de[v][:, WP4].mean()), 4),
                      "gap_closed_pct": round((b2 - float(de[v][:, WP4].mean()))
                                              / (b2 - c2) * 100, 1)}
                  for k, v in fixes.items()}}

    # ---- the horizon-level read: beats-CV over how many steps? ------------ #
    res["gap_attribution"]["beats_cv_steps"] = {
        f"{act}|{ro}": separated_better_interval(de[CELLS[(act, ro)]], de[CV],
                                                 draws)
        for (act, ro) in CELLS}

    # ---- T_blind, ONLY where a readout-matched comparator exists ---------- #
    res["T_blind_where_matched"] = {
        "own|op": t_blind(de["a_imagination__own"], de["b_frozenlast__own"],
                          draws, label_a="a__own", label_b="b__own"),
        "hold|op": t_blind(de["a_imagination__hold"], de["b_frozenlast__hold"],
                           draws, label_a="a__hold", label_b="b__hold"),
        "true|op": t_blind(de["a_imagination__true"], de["b_frozenlast__true"],
                           draws, label_a="a__true", label_b="b__true"),
        "true|str": t_blind(de["a_imagination__true__roSTR"],
                            de["b_frozenlast__true__roSTR"], draws,
                            label_a="a__true__roSTR", label_b="b__true__roSTR"),
        "MISSING": ["own|str", "hold|str"],
        "why_missing": ("b_frozenlast__own__roSTR and b_frozenlast__hold__roSTR "
                        "were never rolled out; building them is GPU work and "
                        "is the first item of the Rung-1 pod job"),
    }

    # ---- how much of the own-action penalty is the POLICY? ---------------- #
    # held-last is a deployable policy with NO model in the control loop; the
    # gap between it and own_kinematic is the compounding cost of closing the
    # loop through the model's own decoded motion.
    for ro in ("op", "str"):
        own = de[CELLS[("own", ro)]][:, 19]
        hold = de[CELLS[("hold", ro)]][:, 19]
        true = de[CELLS[("true", ro)]][:, 19]
        res["action_source_dominance"][ro] = {
            "own_de_2s": round(float(own.mean()), 4),
            "hold_de_2s": round(float(hold.mean()), 4),
            "true_de_2s": round(float(true.mean()), 4),
            "policy_penalty_m (own - hold)": round(float(own.mean() - hold.mean()), 4),
            "residual_privilege_m (hold - true)": round(
                float(hold.mean() - true.mean()), 4),
            "policy_share_of_own_minus_true_pct": round(
                float((own.mean() - hold.mean())
                      / max(own.mean() - true.mean(), 1e-9)) * 100, 1),
            "paired_own_vs_hold": paired_at(de[CELLS[("hold", ro)]],
                                            de[CELLS[("own", ro)]], draws, 20),
        }

    # ---- the headline contrast, with its interval -------------------------- #
    # "the best fully DEPLOYABLE configuration beats the PRIVILEGED as-shipped
    # arm" is a load-bearing claim, so it carries its own paired test rather
    # than being read off two point estimates.
    dep_best, priv_ship = CELLS[("hold", "str")], CELLS[("true", "op")]
    res["headline_contrasts"] = {
        "deployable_best_vs_privileged_as_shipped": {
            "deployable_arm": dep_best, "privileged_arm": priv_ship,
            "de_2s": paired_at(de[dep_best], de[priv_ship], draws, 20),
            "de_6s": paired_at(de[dep_best], de[priv_ship], draws, 60),
            "ade_0_2s": _ci.paired_episode_cluster_bootstrap(
                de[priv_ship][:, WP4].mean(axis=1),
                de[dep_best][:, WP4].mean(axis=1), eid, n_boot=B_BOOT, seed=SEED),
            "reads": ("positive delta = the DEPLOYABLE arm (held-last action + "
                      "20-step readout) has the smaller error than the "
                      "PRIVILEGED as-shipped arm (expert actions + 4-step "
                      "readout)")},
        "deployable_best_vs_cv_floor": {
            "arm": dep_best,
            "de_2s": paired_at(de[dep_best], de[CV], draws, 20),
            "beats_cv": separated_better_interval(de[dep_best], de[CV], draws)},
        "deployable_asshipped_vs_cv_floor": {
            "arm": CELLS[("own", "op")],
            "de_2s": paired_at(de[CELLS[("own", "op")]], de[CV], draws, 20),
            "beats_cv": separated_better_interval(de[CELLS[("own", "op")]],
                                                  de[CV], draws)},
    }

    (out / "rung2_decomposition.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")

    # ---- console summary --------------------------------------------------- #
    print("2x3 FACTORIAL — de@2s (m), arm (a), identical 599 windows")
    print("%-8s %10s %10s %10s" % ("action", "op(k=4)", "str(k=20)", "rel%"))
    for act in ("own", "hold", "true"):
        o = float(de[CELLS[(act, "op")]][:, 19].mean())
        s = float(de[CELLS[(act, "str")]][:, 19].mean())
        print("%-8s %10.4f %10.4f %+9.1f%%" % (act, o, s, (s / o - 1) * 100))
    print("\nT_useful — how long does de_N stay under the bar? (s)")
    print("%-10s %8s %8s %8s" % ("cell", "<1.0m", "<1.391m", "<2.0m"))
    for (act, ro) in CELLS:
        tu = res["cells"][f"{act}|{ro}"]["T_useful"]
        print("%-10s %8.1f %8.1f %8.1f" % (f"{act}|{ro}", tu["lane_half_1m"]["T_s"],
                                           tu["corridor_1p391m"]["T_s"],
                                           tu["miss_2m"]["T_s"]))
    print("\nbeats-CV steps of 185:")
    for k, v in res["gap_attribution"]["beats_cv_steps"].items():
        print("   %-12s %3d steps  (%s .. %s s)" % (k, v["n_steps"],
                                                    v["first_s"], v["last_s"]))
    print("\nGAP ATTRIBUTION on de@2s: deployable %.4f -> privileged %.4f "
          "(gap %.4f m)" % (base, ceil, gap))
    for k, v in res["gap_attribution"]["de_2s"]["fixes"].items():
        print("   %-42s %.4f m   closes %5.1f%%"
              % (k, v["de_2s_m"], v["gap_closed_pct"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
