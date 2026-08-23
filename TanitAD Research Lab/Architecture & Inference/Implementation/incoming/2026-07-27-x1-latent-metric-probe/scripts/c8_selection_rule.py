#!/usr/bin/env python3
"""C8 — the horizon-banked readout SELECTION RULE, evaluated at zero GPU.

Escalation E2 (MANIFOLD_MISMATCH_RESEARCH.md §5 C8, MANIFEST.md §3):

    "The design four independent SOTA forecasting systems converged on is a bank
     of horizon-specialised readouts selected by lead time (GraphCast recommends
     it; FuXi and Pangu-Weather ship it), and WE ALREADY OWN THE BANK.  The
     decision to make is not 'which single readout' but 'WHAT SELECTION RULE',
     and it costs zero GPU."

The bank is ``grounding.step['op'|'tac'|'str']`` — trained at rollout lengths
4 / 16 / 20 steps, all three present in every 4-brain checkpoint.  Every arm below
decodes THE SAME latent rollout with a different head, so a selection rule is a
pure inference-time choice: no retraining, no extra rollout, no extra encode.

⚠️ SELECTION SEMANTICS, stated because it is easy to get wrong.  This selects
WHICH HEAD'S TRAJECTORY IS READ AT LEAD TIME j — exactly the GraphCast/FuXi/Pangu
pattern (each specialist forecasts, you read the one for your lead time).  It is
NOT a splice of per-step Δposes inside one SE(2) accumulation; that would need a
re-decode and is not what the literature does.  Consequence to flag for the
harness owner: the concatenated path can be DISCONTINUOUS at a switch step.  For
a planner consuming waypoints at fixed lead times that is irrelevant; for a
smooth-path consumer it is not, and a blend window would be needed.

Data: the committed per-window dense DE dumps, [599 windows x 185 steps], v1
(`flagship-30k`, step 29999), val `physicalai-val-0c5f7dac3b11`, 600 episodes /
596 episode clusters, stride 8, dt 0.1 s.  No GPU, no pod, no model load.

Estimator: paired episode-cluster bootstrap (taniteval/ci.py, B=2000, seed 0),
unit = episode cluster.  NEVER overlapping_holdout_se.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                    "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "taniteval"))
from taniteval import ci as _ci                                    # noqa: E402

INC = os.path.join(REPO, "TanitAD Research Hub", "Architecture & Inference",
                   "Implementation", "incoming")
BI = os.path.join(INC, "2026-07-26-blind-imagination", "perwindow",
                  "bi_perwindow_compact.pt")
LADDER = os.path.join(INC, "2026-07-26-tblind-ladder", "perwindow",
                      "perwindow_matched_K185.pt")
RUNG1 = os.path.join(INC, "2026-07-26-tblind-rung1", "perwindow",
                     "rung1_perwindow_compact.pt")

WP_IDX = [4, 9, 14, 19]          # 0.5 / 1 / 1.5 / 2 s  (0-based step index)
GRID = {"0.5s": 4, "1s": 9, "1.5s": 14, "2s": 19, "3s": 29, "4.5s": 44,
        "6s": 59, "9s": 89, "12s": 119, "18.5s": 184}
# committed numbers this script must reproduce before quoting anything new
GATE = {"a_imagination__true": 0.3839, "a_imagination__true__roSTR": 0.1950}
GATE_TOL = 5e-4
# committed beats-CV window for v1's `op` readout under true actions
# (BLIND_IMAGINATION.md §2.5 / line 325: "0.4 s … 7.4 s | 71 / 185")
GATE_BEATS_CV = {"arm": "a_imagination__true", "first": 4, "last": 74,
                 "n_steps": 71}


def ade(d):
    """The program's ``ade_0_2s``: per-window mean over the 4 sparse waypoints."""
    return d[:, WP_IDX].mean(axis=1)


def fit_rule(train, banks, kind, kmax):
    """Return r[j] (readout name per step) fitted on ``train`` window indices."""
    names = list(banks)
    m = np.stack([banks[n][train].mean(axis=0) for n in names])       # [R, K]
    if kind == "argmin_per_step":
        return [names[i] for i in m.argmin(axis=0)]
    if kind == "crossover_op_str":
        a, b = banks["op"][train], banks["str"][train]
        best, bj = None, 0
        for j in range(kmax + 1):
            v = np.concatenate([a[:, :j], b[:, j:]], axis=1).mean()
            if best is None or v < best:
                best, bj = v, j
        return ["op"] * bj + ["str"] * (kmax - bj)
    if kind == "crossover_op_tac_str":
        best, bj = None, (0, 0)
        for j1 in range(0, kmax + 1, 2):
            for j2 in range(j1, kmax + 1, 2):
                v = np.concatenate([banks["op"][train][:, :j1],
                                    banks["tac"][train][:, j1:j2],
                                    banks["str"][train][:, j2:]], axis=1).mean()
                if best is None or v < best:
                    best, bj = v, (j1, j2)
        j1, j2 = bj
        return ["op"] * j1 + ["tac"] * (j2 - j1) + ["str"] * (kmax - j2)
    raise ValueError(kind)


def apply_rule(rule, banks, rows):
    out = np.empty((len(rows), len(rule)), dtype=np.float64)
    for j, r in enumerate(rule):
        out[:, j] = banks[r][rows, j]
    return out


def folds(eid, n_folds, seed=0):
    uniq = sorted(set(eid))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    grp = {uniq[perm[i]]: i % n_folds for i in range(len(uniq))}
    f = np.array([grp[e] for e in eid])
    return [(np.nonzero(f != k)[0], np.nonzero(f == k)[0]) for k in range(n_folds)]


def beats_floor_window(arm, floor, eid, n_boot=400, kmax=185):
    """The program's 'beats-CV WINDOW' — [first, last] lead time over which the
    arm is SEPARATED-better than the floor.

    ⚠️ Convention recovered by reproduction, not assumed.  A point-mean rule
    (``arm <= floor``) gives steps 4..88 for v1's ``op`` readout under true
    actions; the committed artifact says **0.4 s … 7.4 s = 71 steps**.  The rule
    that reproduces 4..74 = 71 steps EXACTLY is the paired episode-cluster
    bootstrap with the CI lower bound of ``floor - arm`` > 0.  So "beats CV" in
    this program means SEPARATED-beats, and the point-mean version overstates the
    end of the window by ~1.4 s.  Pinned in :data:`GATE_BEATS_CV`.
    """
    first = last = None
    for j in range(kmax):
        r = _ci.paired_episode_cluster_bootstrap(floor[:, j], arm[:, j], eid,
                                                 n_boot=n_boot, seed=0)
        if r["lo"] > 0:
            if first is None:
                first = j + 1
            last = j + 1
        elif first is not None:
            break
    return (first or 0), (last or 0)


def t_blind(a, b, eid, n_boot=500):
    """Deployable T_blind: largest N such that the paired CI lower bound of
    (b - a) is > 0 contiguously from step 2 (`t_blind.json` rule; step 1 is
    bit-identical across state-source arms by construction)."""
    n = 1
    for j in range(1, a.shape[1]):
        r = _ci.paired_episode_cluster_bootstrap(b[:, j], a[:, j], eid,
                                                 n_boot=n_boot, seed=0)
        if r["lo"] > 0:
            n = j + 1
        else:
            break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "..", "artifacts"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--kmax", type=int, default=185)
    ap.add_argument("--tblind-boot", type=int, default=500)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    bi = torch.load(BI, map_location="cpu", weights_only=False)
    lad = torch.load(LADDER, map_location="cpu", weights_only=False)
    r1 = torch.load(RUNG1, map_location="cpu", weights_only=False)

    # ---- integrity gate: the three dumps must describe the SAME windows ---- #
    align = {"bi_vs_ladder_eid": bi["eid"] == lad["eid"],
             "bi_vs_ladder_t0": bi["t0"] == lad["t0"],
             "bi_vs_rung1_eid": bi["eid"] == r1["eid"],
             "bi_vs_rung1_t0": bi["t0"] == r1["t0"]}
    if not all(align.values()):
        raise SystemExit(f"window sets do not align across dumps: {align}")
    eid = bi["eid"]
    D = {k: v.double().numpy() for k, v in bi["dense_de_headline"].items()}
    D.update({k: v.double().numpy()
              for k, v in lad["dense_de_matched_arms"].items() if k not in D})
    D.update({k: v.double().numpy()
              for k, v in r1["dense_de"].items() if k not in D})

    gate = {}
    for arm, exp in GATE.items():
        got = float(ade(D[arm]).mean())
        gate[arm] = {"expect": exp, "got": round(got, 4),
                     "PASS": abs(got - exp) <= GATE_TOL}
    # the committed crossover contrast, recomputed with the decision-grade estimator
    gate["de_0p5s_paired_str_vs_op_true"] = _ci.paired_episode_cluster_bootstrap(
        D["a_imagination__true__roSTR"][:, 4], D["a_imagination__true"][:, 4],
        eid, n_boot=2000, seed=0)
    gb = GATE_BEATS_CV
    f0, f1 = beats_floor_window(D[gb["arm"]], D["d_constant_velocity"], eid,
                                a.tblind_boot, a.kmax)
    gate["beats_cv_window_convention"] = {
        "arm": gb["arm"], "expect_steps": [gb["first"], gb["last"]],
        "got_steps": [f0, f1], "expect_n_steps": gb["n_steps"],
        "got_n_steps": f1 - f0 + 1,
        "PASS": bool(f0 == gb["first"] and f1 == gb["last"])}
    gate["ALL_PASS"] = all(v["PASS"] for v in gate.values() if isinstance(v, dict)
                           and "PASS" in v)
    print("GATE", json.dumps({k: v for k, v in gate.items()
                              if k != "de_0p5s_paired_str_vs_op_true"}, indent=1))

    regimes = {
        "own_kinematic_DEPLOYABLE": {
            "a": {"op": D["a_imagination__own"],
                  "tac": D["a_imagination__own__roTAC"],
                  "str": D["a_imagination__own__roSTR"]},
            "b": {"op": D["b_frozenlast__own"],
                  "tac": D["b_frozenlast__own__roTAC"],
                  "str": D["b_frozenlast__own__roSTR"]}},
        "true_actions_PRIVILEGED": {
            "a": {"op": D["a_imagination__true"],
                  "tac": D["a_imagination__true__roTAC"],
                  "str": D["a_imagination__true__roSTR"]},
            "b": None},
    }
    cv, hv = D["d_constant_velocity"], D["d2_hold_v0"]
    K = a.kmax
    fl = folds(eid, a.folds)

    out = {
        "experiment": "C8 — horizon-banked readout selection rule (zero GPU)",
        "escalation": "E2 — the open question is the RULE, not the readout",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": "dev box (CPU only, no GPU, no pod, no model load)",
        "sources": {"bi": os.path.relpath(BI, REPO).replace("\\", "/"),
                    "ladder": os.path.relpath(LADDER, REPO).replace("\\", "/"),
                    "rung1": os.path.relpath(RUNG1, REPO).replace("\\", "/")},
        "provenance": {k: bi.get("meta_sweep", {}).get(k) for k in
                       ("arm_ckpt", "ckpt_step", "kmax", "stride", "n_windows",
                        "n_episode_clusters", "dt_s")},
        "trained_rollout_lengths_steps": r1["meta"].get(
            "trained_rollout_lengths_steps"),
        "window_alignment_gate": align,
        "reproduction_gate": gate,
        "selection_semantics": ("selects WHICH HEAD'S TRAJECTORY is read at lead "
                                "time j (the GraphCast/FuXi/Pangu pattern); NOT a "
                                "per-step splice inside one SE(2) accumulation. "
                                "The concatenated path may be DISCONTINUOUS at a "
                                "switch step — flagged for the harness owner."),
        "estimator": ("paired episode-cluster bootstrap, B=2000, seed 0, unit = "
                      "episode cluster (taniteval/ci.py)"),
        "cv_folds": a.folds,
        "regimes": {},
    }

    for rname, R in regimes.items():
        banks = {k: v[:, :K] for k, v in R["a"].items()}
        arms = {f"always_{n}": v for n, v in banks.items()}
        rules_fitted, oof = {}, {}
        for kind in ("argmin_per_step", "crossover_op_str", "crossover_op_tac_str"):
            pred = np.empty((len(eid), K), dtype=np.float64)
            for tr, te in fl:
                rule = fit_rule(tr, banks, kind, K)
                pred[te] = apply_rule(rule, banks, te)
            oof[kind] = pred
            rules_fitted[kind] = fit_rule(np.arange(len(eid)), banks, kind, K)
        arms.update({f"rule_{k}_oof": v for k, v in oof.items()})
        arms["oracle_per_window_NOT_DEPLOYABLE"] = np.min(
            np.stack(list(banks.values())), axis=0)
        arms["floor_constant_velocity"] = cv[:, :K]
        arms["floor_hold_v0"] = hv[:, :K]

        rr = {"arms": {}, "fitted_rules": {}}
        for n, v in arms.items():
            per = ade(v)
            e = {"ade_0_2s": _ci.episode_cluster_bootstrap(per, eid, n_boot=2000,
                                                           seed=0),
                 "de_at_grid": {g: round(float(v[:, i].mean()), 4)
                                for g, i in GRID.items() if i < K}}
            if not n.startswith("floor_"):
                f0, f1 = beats_floor_window(v, cv[:, :K], eid, a.tblind_boot, K)
                h0, h1 = beats_floor_window(v, hv[:, :K], eid, a.tblind_boot, K)
                e["beats_cv_window_steps"] = [f0, f1]
                e["beats_cv_window_s"] = [round(f0 * 0.1, 1), round(f1 * 0.1, 1)]
                e["beats_holdv0_window_steps"] = [h0, h1]
            rr["arms"][n] = e
        for k, rule in rules_fitted.items():
            sw = [(0, rule[0])]
            for j in range(1, K):
                if rule[j] != rule[j - 1]:
                    sw.append((j, rule[j]))
            rr["fitted_rules"][k] = {
                "switch_points_step_readout": [[int(j), r] for j, r in sw],
                "switch_points_s": [[round(j * 0.1, 1), r] for j, r in sw],
                "n_steps_per_readout": {n: int(sum(1 for x in rule if x == n))
                                        for n in banks}}
        # paired contrasts against the two candidate "flat" answers
        base = {"always_op": ade(banks["op"]), "always_str": ade(banks["str"]),
                "always_tac": ade(banks["tac"])}
        rr["paired_vs_flat"] = {}
        for k, v in oof.items():
            rr["paired_vs_flat"][k] = {
                b: _ci.paired_episode_cluster_bootstrap(ade(v), bv, eid,
                                                        n_boot=2000, seed=0)
                for b, bv in base.items()}
        rr["paired_str_minus_op_at_grid"] = {
            g: _ci.paired_episode_cluster_bootstrap(
                banks["str"][:, i], banks["op"][:, i], eid, n_boot=2000, seed=0)
            for g, i in GRID.items() if i < K}
        # T_blind (imagination vs frozen-last), matched readout, when b exists
        if R["b"] is not None:
            bb = {k: v[:, :K] for k, v in R["b"].items()}
            idx = np.arange(len(eid))
            tb = {}
            for n in banks:
                tb[f"always_{n}"] = t_blind(banks[n], bb[n], eid, a.tblind_boot)
            for k, rule in rules_fitted.items():
                tb[f"rule_{k}"] = t_blind(apply_rule(rule, banks, idx),
                                          apply_rule(rule, bb, idx), eid,
                                          a.tblind_boot)
            rr["t_blind_steps_vs_frozen_last"] = tb
            rr["t_blind_s_vs_frozen_last"] = {k: round(v * 0.1, 1)
                                              for k, v in tb.items()}
            # ⭐ THE TWO-OBJECTIVE TABLE.  ADE-optimal and T_blind-optimal switch
            # points need not coincide, and if they do not the "rule" question
            # cannot be answered without naming the objective.
            two = {}
            for j in (0, 1, 2, 3, 5, 8, 10, 13, 20, 30):
                rule = ["op"] * j + ["str"] * (K - j)
                pa, pb = apply_rule(rule, banks, idx), apply_rule(rule, bb, idx)
                two[f"switch_at_step_{j}"] = {
                    "switch_s": round(j * 0.1, 1),
                    "ade_0_2s": round(float(ade(pa).mean()), 4),
                    "t_blind_steps": t_blind(pa, pb, eid, a.tblind_boot),
                    "beats_cv_window_steps": list(
                        beats_floor_window(pa, cv[:, :K], eid, a.tblind_boot, K))}
                two[f"switch_at_step_{j}"]["t_blind_s"] = round(
                    two[f"switch_at_step_{j}"]["t_blind_steps"] * 0.1, 1)
            rr["two_objective_op_to_str_switch_sweep"] = two
        out["regimes"][rname] = rr

        print(f"\n=== {rname} ===")
        for n in sorted(rr["arms"], key=lambda x: rr["arms"][x]["ade_0_2s"]["mean"]):
            e = rr["arms"][n]
            print(f"  {n:<38} ade_0_2s {e['ade_0_2s']['mean']:.4f} "
                  f"[{e['ade_0_2s']['lo']:.4f},{e['ade_0_2s']['hi']:.4f}]  "
                  f"beats-CV {e.get('beats_cv_window_s')}")
        for k, v in rr["fitted_rules"].items():
            print(f"  rule {k:<24} switches {v['switch_points_s']}")
        if "t_blind_s_vs_frozen_last" in rr:
            print("  T_blind(s):", rr["t_blind_s_vs_frozen_last"])
            print("  two-objective op->str switch sweep "
                  "(switch_s: ade_0_2s / T_blind_s):")
            for k, v in rr["two_objective_op_to_str_switch_sweep"].items():
                print(f"    {v['switch_s']:>5.1f}s  ade {v['ade_0_2s']:.4f}  "
                      f"T_blind {v['t_blind_s']:.1f}s  "
                      f"beats-CV {v['beats_cv_window_steps']}")

    fp = os.path.join(a.out, "c8_selection_rule.json")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nelapsed {time.time()-t0:.0f}s -> {fp}")


if __name__ == "__main__":
    main()
