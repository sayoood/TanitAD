#!/usr/bin/env python3
"""RUNG 0 — does the readout swap extend the DEPLOYABLE ``T_blind``?  ZERO GPU.

Recomputes ``T_blind`` under the model's OWN actions with the 20-step-calibrated
``step["str"]`` readout in place of the 4-step ``step["op"]`` one, from the
committed per-window dump
``…/2026-07-26-blind-imagination/perwindow/bi_perwindow_compact.pt``.

⚠️ The ``T_blind`` rule is **RE-DERIVED here from its written specification**, not
imported from ``bi_analyze``. The brief requires it: that agent's own
pre-registration carried a C13 defect (a criterion that could not fire) in this
exact rule. The re-derivation is then held against the committed outputs as a
fidelity gate, AND against deliberately failing inputs.

Estimator everywhere: **paired episode-cluster bootstrap** (``taniteval/ci.py``,
B = 2000, seed 0, resampling unit = val episode), identical windows for every
arm. ``overlapping_holdout_se`` appears nowhere.

Usage:
    python tb_rung0.py --dump <bi_perwindow_compact.pt> --out <artifacts dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# taniteval.ci — the program's estimator. Imported, never re-implemented.
# --------------------------------------------------------------------------- #
_REPO = Path(__file__).resolve().parents[6]
for _p in (_REPO / "taniteval", _REPO / "stack"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from taniteval import ci as _ci                      # noqa: E402

DT = 0.1                       # 10 Hz — the corpus rate (blindimag.DT)
B_BOOT = 2000
SEED = 0
GRID = (5, 10, 20, 30, 45, 60, 90, 120, 185)
WP4 = [4, 9, 14, 19]           # the program's ade_0_2s: 0.5 / 1 / 1.5 / 2 s

#: AMENDMENT A4 (blind-imagination, `bi_analyze.T_CONTIGUITY_START_STEP`).
#: Contiguity is anchored at step 2 because arms (a)/(b)/(c) decode a
#: BIT-IDENTICAL first transition by construction, so a rule anchored at N = 1
#: returns 0 for every arm regardless of the data — a C13 criterion that cannot
#: fire. Re-derived here rather than inherited; §3 of PRE_REGISTRATION.md states
#: the FAILING value this rule can return (1 step) and the result that produces
#: it, and `selftest_failing_inputs()` proves it fires.
T_CONTIGUITY_START_STEP = 2


# --------------------------------------------------------------------------- #
def t_contiguous(delta_lo, start_idx: int = T_CONTIGUITY_START_STEP - 1) -> int:
    """Largest N with ``delta_lo > 0`` contiguously from ``T_CONTIGUITY_START_STEP``.

    ⭐ **The failing value is ``start_idx`` = 1 step (0.1 s), not 0** — returned
    whenever the FIRST evaluable horizon already fails. A rule whose worst case
    were unreachable would be worthless whichever way the world is.
    """
    ok = np.asarray(delta_lo)[start_idx:] > 0
    if ok.size == 0 or not ok[0]:
        return int(start_idx)
    bad = np.flatnonzero(~ok)
    return int(start_idx) + (int(bad[0]) if bad.size else int(ok.size))


def draws_for(eid):
    uniq, idx = _ci.episode_index(eid)
    return list(_ci._draws(uniq, idx, B_BOOT, SEED)), len(uniq)


def t_blind(de_a, de_b, draws, *, label_a="a", label_b="b") -> dict:
    """``T_blind`` = largest horizon at which (a) is separated-better than (b) on
    ``de_N``, contiguously from N = 2, with the interval obtained by re-deriving
    the WHOLE curve inside every episode resample."""
    K = de_a.shape[1]
    d_boot = np.empty((len(draws), K))
    for i, s in enumerate(draws):
        d_boot[i] = de_b[s].mean(axis=0) - de_a[s].mean(axis=0)
    lo = np.percentile(d_boot, 2.5, axis=0)
    hi = np.percentile(d_boot, 97.5, axis=0)
    point_delta = de_b.mean(axis=0) - de_a.mean(axis=0)
    t_point = t_contiguous(lo)
    t_dist = np.array([t_contiguous(d) for d in d_boot])
    t_lo, t_hi = np.percentile(t_dist, [2.5, 97.5])
    return {
        "arm_a": label_a, "arm_b": label_b,
        "T_blind_steps": int(t_point),
        "T_blind_s": round(t_point * DT, 2),
        "T_blind_ci95_steps": [int(round(t_lo)), int(round(t_hi))],
        "T_blind_ci95_s": [round(float(t_lo) * DT, 2), round(float(t_hi) * DT, 2)],
        "T_blind_median_boot_steps": int(float(np.median(t_dist))),
        # ⚠️ `frac_draws_T_blind_is_zero` (printed by the committed artifact) is
        # STRUCTURALLY 0 under A4 — the rule's minimum return is 1. The floor
        # indicator that carries information is this one:
        "frac_draws_T_blind_at_floor_1step": round(float((t_dist == 1).mean()), 4),
        "frac_draws_T_blind_ge_16": round(float((t_dist >= 16).mean()), 4),
        "frac_draws_T_blind_gt_8": round(float((t_dist > 8).mean()), 4),
        "delta_at_step1_m": round(float(point_delta[0]), 6),
        "delta_at_step2_m": round(float(point_delta[1]), 6),
        "lo_at_step2_m": round(float(lo[1]), 6),
        "first_step_where_a_loses_point": (
            int(np.flatnonzero(point_delta[1:] <= 0)[0] + 2)
            if (point_delta[1:] <= 0).any() else None),
        "first_step_where_b_separated_better": (
            int(np.flatnonzero(hi[1:] < 0)[0] + 2) if (hi[1:] < 0).any() else None),
        "C14_saturated_at_grid_terminus": bool(t_point >= K),
        "contiguity_start_step": T_CONTIGUITY_START_STEP,
        "K_max_swept": int(K), "n_boot": B_BOOT,
        "estimator": "paired_episode_cluster_bootstrap",
        "rule": ("largest N with paired CI lower bound > 0 contiguously from "
                 "N=2 (A4); positive delta = the FIRST arm is better; the "
                 "FAILING return is 1 step"),
    }


def paired_at(de_a, de_b, draws, n, cumulative=False) -> dict:
    """``mean(b) - mean(a)`` at horizon ``n``; POSITIVE = arm (a) is better."""
    if cumulative:
        va, vb = de_a[:, :n].mean(axis=1), de_b[:, :n].mean(axis=1)
    else:
        va, vb = de_a[:, n - 1], de_b[:, n - 1]
    d = np.array([vb[s].mean() - va[s].mean() for s in draws])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta_b_minus_a": round(float(vb.mean() - va.mean()), 4),
            "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "separated": bool(lo > 0 or hi < 0), "a_better": bool(lo > 0),
            "n_windows": int(va.size), "n_boot": B_BOOT,
            "estimator": "paired_episode_cluster_bootstrap"}


def single_at(de, eid, draws, n, cumulative=False) -> dict:
    v = de[:, :n].mean(axis=1) if cumulative else de[:, n - 1]
    bs = np.array([v[s].mean() for s in draws])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": round(float(v.mean()), 4), "lo": round(float(lo), 4),
            "hi": round(float(hi), 4), "n_windows": int(v.size),
            "n_episodes": int(len(set(eid))), "n_boot": B_BOOT,
            "estimator": "episode_cluster_bootstrap"}


def ade_0_2s(de, eid) -> dict:
    """The program's own sparse 4-waypoint ade_0_2s, with its interval."""
    return _ci.episode_cluster_bootstrap(de[:, WP4].mean(axis=1), eid,
                                         n_boot=B_BOOT, seed=SEED)


def separated_better_interval(de_a, de_b, draws) -> dict:
    """The horizon INTERVAL over which (a) is separated-better than (b).

    ⚠️ NOT the contiguity rule. Contiguity is right for ``T_blind`` ("how long
    does imagination stay ahead of a frozen percept") and wrong for the floor
    ("over WHICH horizons is it better at all") — the blind-imagination report
    makes that distinction and this reproduces its construction.
    """
    K = de_a.shape[1]
    d_boot = np.empty((len(draws), K))
    for i, s in enumerate(draws):
        d_boot[i] = de_b[s].mean(axis=0) - de_a[s].mean(axis=0)
    lo = np.percentile(d_boot, 2.5, axis=0)
    ok = np.flatnonzero(lo > 0)
    return {"first_s": round(float(ok[0] + 1) * DT, 2) if ok.size else None,
            "last_s": round(float(ok[-1] + 1) * DT, 2) if ok.size else None,
            "n_steps": int(ok.size), "K": int(K),
            "estimator": "paired_episode_cluster_bootstrap"}


# =========================================================================== #
# ⛔ VALIDATION — BOTH DIRECTIONS. Nothing below is read until this passes.
# =========================================================================== #
COMMITTED_T_BLIND = {           # bi/artifacts/t_blind.json, read from the JSON
    ("a_imagination__true", "b_frozenlast__true"): (65, [6.5, 8.9]),
    ("a_imagination__own", "b_frozenlast__own"): (8, [0.8, 1.0]),
    ("a_imagination__hold", "b_frozenlast__hold"): (32, [3.2, 5.0]),
    ("a_imagination__gtkin", "b_frozenlast__gtkin"): (54, [5.4, 7.3]),
    ("a_imagination__true__roSTR", "b_frozenlast__true__roSTR"): (185, [18.5, 18.5]),
}
COMMITTED_ADE = {               # BLIND_IMAGINATION.md §1/§2.5/§3.5, cross-checked
    "a_imagination__true": 0.3839, "a_imagination__true__roTAC": 0.1865,
    "a_imagination__true__roSTR": 0.1950, "c2_observedpair__true": 3.6093,
    "a_imagination__own": 0.9554, "a_imagination__hold": 0.4712,
    "c_fullobs__true": 0.5167, "d_constant_velocity": 0.6083,
}


def fidelity_gate(de, eid, draws) -> dict:
    """Direction 1: my re-derivation must reproduce every committed number."""
    rows, ok_all = [], True
    for (a, b), (steps, ci) in COMMITTED_T_BLIND.items():
        r = t_blind(de[a], de[b], draws, label_a=a, label_b=b)
        ok = (r["T_blind_steps"] == steps and r["T_blind_ci95_s"] == ci)
        ok_all &= ok
        rows.append({"pair": f"{a} vs {b}", "committed_steps": steps,
                     "recomputed_steps": r["T_blind_steps"],
                     "committed_ci95_s": ci,
                     "recomputed_ci95_s": r["T_blind_ci95_s"], "match": ok})
    ade_rows = []
    for arm, val in COMMITTED_ADE.items():
        got = ade_0_2s(de[arm], eid)["mean"]
        ok = abs(got - val) < 5e-4
        ok_all &= ok
        ade_rows.append({"arm": arm, "committed": val, "recomputed": got,
                         "abs_diff": round(abs(got - val), 6), "match": ok})
    return {"t_blind": rows, "ade_0_2s": ade_rows, "GATE_PASS": bool(ok_all)}


def selftest_failing_inputs(de, draws) -> dict:
    """Direction 2 (M3): the rule must FAIL on inputs it should fail on.

    A guard that cannot return a failing value is worth nothing whichever way
    the world is — that is exactly the C13 defect this ladder was told to check
    for before adjudicating.
    """
    a = de["a_imagination__own"]
    out = {}

    # (1) identical arms  =>  delta == 0 everywhere  =>  must return the floor
    r = t_blind(a, a.copy(), draws, label_a="own", label_b="own(copy)")
    out["identical_arms_returns_floor"] = {
        "T_blind_steps": r["T_blind_steps"], "expected": 1,
        "pass": r["T_blind_steps"] == 1}

    # (2) swapped arms: feed a KNOWN-WORSE arm as "imagination"
    r = t_blind(de["b_frozenlast__true"], de["a_imagination__true"], draws,
                label_a="frozen(true) [deliberately wrong]", label_b="imagination(true)")
    out["swapped_arms_returns_floor"] = {
        "T_blind_steps": r["T_blind_steps"], "expected": 1,
        "pass": r["T_blind_steps"] == 1}

    # (3) an ISOLATED LATE WIN must not be promoted to a long horizon:
    #     arm (a) is made worse than (b) for steps 1..39 and much better after.
    b = de["b_frozenlast__own"].copy()
    a2 = b.copy() + 0.5          # (a) uniformly worse ...
    a2[:, 39:] = b[:, 39:] - 0.5  # ... except from step 40 on, where it wins big
    r = t_blind(a2, b, draws, label_a="late-win synthetic", label_b="frozen(own)")
    out["isolated_late_win_not_promoted"] = {
        "T_blind_steps": r["T_blind_steps"], "expected": 1,
        "pass": r["T_blind_steps"] == 1}

    # (4) a POSITIVE control: a uniformly-better arm must saturate, so the rule
    #     is not simply always-failing.
    a3 = de["b_frozenlast__own"] - 0.5
    r = t_blind(a3, de["b_frozenlast__own"], draws,
                label_a="uniformly-better synthetic", label_b="frozen(own)")
    out["uniformly_better_saturates"] = {
        "T_blind_steps": r["T_blind_steps"], "expected": int(a3.shape[1]),
        "pass": r["T_blind_steps"] == a3.shape[1]}

    out["ALL_PASS"] = bool(all(v["pass"] for k, v in out.items() if k != "ALL_PASS"))
    return out


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d = torch.load(args.dump, map_location="cpu", weights_only=False)
    de = {k: v.double().numpy() for k, v in d["dense_de_headline"].items()}
    eid = [str(x) for x in d["eid"]]
    draws, n_ep = draws_for(eid)
    K = next(iter(de.values())).shape[1]
    meta = {"source_dump": str(Path(args.dump).name),
            "n_windows": int(next(iter(de.values())).shape[0]),
            "n_episode_clusters": int(n_ep), "K_max": int(K),
            "n_boot": B_BOOT, "seed": SEED, "dt_s": DT,
            "arm_ckpt": d["meta_sweep"].get("arm_ckpt"),
            "ckpt_step": d["meta_sweep"].get("ckpt_step"),
            "trained_rollout_lengths_steps":
                d["meta_sweep"].get("trained_rollout_lengths_steps"),
            "estimator": "paired_episode_cluster_bootstrap (taniteval/ci.py)",
            "arms_present_dense": sorted(de)}

    # ---- ⛔ VALIDATION FIRST -------------------------------------------- #
    gate = fidelity_gate(de, eid, draws)
    fail = selftest_failing_inputs(de, draws)
    (out / "rung0_validation.json").write_text(json.dumps(
        {"meta": meta, "fidelity_gate_direction_1": gate,
         "deliberately_failing_inputs_direction_2": fail}, indent=2,
        default=float), encoding="utf-8")
    print("FIDELITY GATE :", "PASS" if gate["GATE_PASS"] else "FAIL", flush=True)
    print("FAILING-INPUT :", "PASS" if fail["ALL_PASS"] else "FAIL", flush=True)
    if not (gate["GATE_PASS"] and fail["ALL_PASS"]):
        print("⛔ VALIDATION FAILED — pre-registration §3 voids the run. "
              "No new number is read.", flush=True)
        (out / "rung0_BLOCKED.json").write_text(json.dumps(
            {"blocked": True, "gate": gate, "selftest": fail}, indent=2),
            encoding="utf-8")
        return 2

    # ---- ⭐ RUNG 0 ------------------------------------------------------- #
    A_OWN, A_OWN_STR = "a_imagination__own", "a_imagination__own__roSTR"
    B_OWN, CV = "b_frozenlast__own", "d_constant_velocity"

    res = {"meta": meta, "preregistered": {
        "primary_P1": f"T_blind({A_OWN_STR} vs {B_OWN}) — UNMATCHED comparator, "
                      "declared a LOWER BOUND",
        "secondary_P2": f"separated-better interval vs {CV} — EXACTLY MATCHED "
                        "(the CV floor has no readout)",
        "sensitivity_P1S": "the same comparator mismatch measured in the "
                           "PRIVILEGED regime, where both comparators exist",
        "buckets": {"CONFIRM": ">= 16 steps (1.6 s)",
                    "PARTIAL": "10..15 steps",
                    "REFUTE": "<= 9 steps"},
        "baseline_steps": 8}}

    # P1 — the primary
    res["P1_deployable_T_blind"] = {
        "op_readout_committed": t_blind(de[A_OWN], de[B_OWN], draws,
                                        label_a=A_OWN, label_b=B_OWN),
        "str_readout_NEW": t_blind(de[A_OWN_STR], de[B_OWN], draws,
                                   label_a=A_OWN_STR, label_b=B_OWN),
        "comparator_note": ("b_frozenlast__own carries the op readout; the "
                            "matched b_frozenlast__own__roSTR was never rolled "
                            "out and cannot be built without GPU. In the "
                            "privileged regime the str readout makes the "
                            "frozen-last arm WORSE at every horizon, so the "
                            "matched delta would be LARGER and the matched "
                            "T_blind >= this one. LOWER BOUND."),
    }

    # P2 — the unconfounded floor contrast
    res["P2_vs_cv_floor"] = {
        arm: separated_better_interval(de[arm], de[CV], draws)
        for arm in (A_OWN, A_OWN_STR, "a_imagination__hold",
                    "a_imagination__hold__roSTR", "a_imagination__true",
                    "a_imagination__true__roSTR", "a_imagination__true__roTAC")}
    res["P2_paired_vs_cv_at_grid"] = {
        arm: {f"{n * DT:g}s": paired_at(de[arm], de[CV], draws, n) for n in GRID}
        for arm in (A_OWN, A_OWN_STR)}

    # P1-S — how big IS the comparator mismatch, where it can be measured?
    res["P1S_comparator_mismatch_privileged"] = {
        "matched_str_vs_str": t_blind(de["a_imagination__true__roSTR"],
                                      de["b_frozenlast__true__roSTR"], draws,
                                      label_a="a__true__roSTR",
                                      label_b="b__true__roSTR"),
        "unmatched_str_vs_op": t_blind(de["a_imagination__true__roSTR"],
                                       de["b_frozenlast__true"], draws,
                                       label_a="a__true__roSTR",
                                       label_b="b__true (op)"),
        "b_arm_readout_effect_at_grid": {
            f"{n * DT:g}s": paired_at(de["b_frozenlast__true__roSTR"],
                                      de["b_frozenlast__true"], draws, n)
            for n in GRID},
        "note": ("positive b_arm_readout_effect = the str readout makes the "
                 "FROZEN-LAST arm better; the blind-imagination report measured "
                 "it NEGATIVE at every horizon, which is what makes P1 a lower "
                 "bound"),
    }

    # arm-level levels, so the verdict can be read without a comparator at all
    res["arm_levels"] = {
        arm: {"ade_0_2s": ade_0_2s(de[arm], eid),
              "de": {f"{n * DT:g}s": single_at(de[arm], eid, draws, n)
                     for n in GRID}}
        for arm in (A_OWN, A_OWN_STR, B_OWN, CV, "a_imagination__hold",
                    "a_imagination__hold__roSTR", "a_imagination__true",
                    "a_imagination__true__roSTR", "a_imagination__true__roTAC",
                    "d2_hold_v0")}

    # ---- adjudication, applied mechanically to the pre-registered buckets -- #
    t = res["P1_deployable_T_blind"]["str_readout_NEW"]["T_blind_steps"]
    verdict = ("CONFIRM" if t >= 16 else "PARTIAL" if t >= 10 else "REFUTE")
    res["VERDICT"] = {
        "T_blind_str_steps": t, "T_blind_str_s": round(t * DT, 2),
        "T_blind_op_steps": res["P1_deployable_T_blind"][
            "op_readout_committed"]["T_blind_steps"],
        "bucket": verdict,
        "beats_cv_op": res["P2_vs_cv_floor"][A_OWN]["n_steps"],
        "beats_cv_str": res["P2_vs_cv_floor"][A_OWN_STR]["n_steps"],
        "rule_applied": "PRE_REGISTRATION.md §2, buckets fixed before computing",
    }
    (out / "rung0_tblind_deployable.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")

    print(f"\n*** RUNG 0 VERDICT: {verdict}")
    print(f"   deployable T_blind  op(k=4)  = "
          f"{res['VERDICT']['T_blind_op_steps']} steps "
          f"({res['VERDICT']['T_blind_op_steps'] * DT:.1f} s)")
    print(f"   deployable T_blind str(k=20) = {t} steps ({t * DT:.1f} s)  "
          f"CI {res['P1_deployable_T_blind']['str_readout_NEW']['T_blind_ci95_s']}")
    print(f"   beats CV: op {res['VERDICT']['beats_cv_op']}/185 steps -> "
          f"str {res['VERDICT']['beats_cv_str']}/185 steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
