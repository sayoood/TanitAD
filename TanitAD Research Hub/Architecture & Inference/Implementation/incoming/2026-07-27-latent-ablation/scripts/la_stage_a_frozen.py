#!/usr/bin/env python3
"""STAGE A — the FROZEN-LATENT ablation. **ZERO GPU.** The priority-1 deliverable.

⭐ WHY THIS NEEDS NO NEW COMPUTE. The PI's ablation 1 is *"hold the last real
latent constant through the rollout while the action channel proceeds normally"*.
That is **exactly** ``blindimag.blind_rollout(state_source="frozen_last")``:
``z_frozen = states[:, -1]`` is appended at every step while the ``own_kinematic``
action is still derived from the model's own decoded Δpose (``blindimag.py``
lines 470, 493-495, 553-556). Rung 1 rolled that arm at **every** blend α as the
matched comparator ``b_*`` — so ``b_blend0.25`` IS the frozen-latent arm at the
genuinely model-driven point, and it is already committed.

⛔ AND THE STRUCTURAL FACT THAT REFRAMES THE QUESTION: the program's ``T_blind``
is DEFINED as "largest N at which (a) is separated-better than (b), contiguous
from N=2" with **b = frozen_last** (``tb_rung0.t_blind``; every committed
``T_blind`` pair in ``COMMITTED_T_BLIND``). **``T_blind`` already IS the
imagination-minus-frozen-latent gap.** Therefore ``T_blind(FROZEN vs FROZEN)`` is
1 step by construction and is emitted as VACUOUS, never adjudicated
(``PRE_REGISTRATION.md`` §1.3, §5).

Estimator: paired episode-cluster bootstrap, ``taniteval/ci.py``, B=2000, seed 0,
unit = episode cluster, identical 599 windows. ``overlapping_holdout_se``
appears nowhere.

Usage:
    python la_stage_a_frozen.py --out ../artifacts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_INCOMING = _HERE.parent.parent
sys.path.insert(0, str(_INCOMING / "2026-07-26-tblind-ladder" / "scripts"))
_REPO = _HERE.parents[5]
for _p in (_REPO / "taniteval", _REPO / "stack"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tb_rung0 import (DT, GRID, B_BOOT, ade_0_2s, draws_for,      # noqa: E402
                      paired_at, separated_better_interval, single_at,
                      t_blind)

RUNG1 = (_INCOMING / "2026-07-26-tblind-rung1" / "perwindow"
         / "rung1_perwindow_compact.pt")

#: alpha -> (INTACT arm, FROZEN arm) in the committed Rung-1 dump.
#: alpha=1.0 uses the hold arms: Rung 1's plumbing self-test proved
#: `own_kinematic|blend=1.0` is BIT-IDENTICAL to `hold_last` (max |d| = 0.0).
ALPHAS = {
    "0":    ("a_imagination__own__roSTR", "b_frozenlast__own__roSTR"),
    "0.25": ("a_blend0.25", "b_blend0.25"),
    "0.75": ("a_blend0.75", "b_blend0.75"),
    "1":    ("a_imagination__hold__roSTR", "b_frozenlast__hold__roSTR"),
}
#: TBLIND_RUNG1.md §3 / §2.3 — the fidelity gate. Level agreement is BLOCKING.
COMMITTED = {
    "0":    {"T_blind": 25,  "de2s": 1.8165, "ade": 0.8710, "tu1m": 1.4,
             "beats_cv": 0},
    "0.25": {"T_blind": 85,  "de2s": 1.0736, "ade": 0.5440, "tu1m": 1.9,
             "beats_cv": 43},
    "0.75": {"T_blind": 116, "de2s": 0.6842, "ade": 0.3437, "tu1m": 2.3,
             "beats_cv": 81},
    "1":    {"T_blind": 115, "de2s": 0.6718, "ade": 0.3351, "tu1m": 2.3,
             "beats_cv": 83},
}
BARS = (1.0, 2.0)


def t_useful(de, bar: float) -> float:
    """Largest horizon (s) with mean ``de`` below ``bar``, contiguous from step 1.

    Returns **0.0** when step 1 is already above the bar — a reachable failing
    value, not a structural minimum.
    """
    m = de.mean(axis=0)
    ok = m < bar
    if not ok[0]:
        return 0.0
    bad = np.flatnonzero(~ok)
    return round(float(int(bad[0]) if bad.size else int(ok.size)) * DT, 2)


def _grid_block(de_x, de_i, draws):
    """R_X across the horizon grid, with the paired interval at each horizon."""
    rows = {}
    for n in GRID:
        p = paired_at(de_i, de_x, draws, n)     # positive => INTACT better
        mi, mx = float(de_i[:, n - 1].mean()), float(de_x[:, n - 1].mean())
        rows[f"{n * DT:g}s"] = {
            "step": n,
            "de_intact_m": round(mi, 4), "de_ablated_m": round(mx, 4),
            "R_x": round((mx - mi) / mi, 4) if mi > 0 else None,
            "paired_delta_intact_minus_ablated_m": p["delta_b_minus_a"],
            "ci95": [p["lo"], p["hi"]], "separated": p["separated"],
        }
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", default=str(RUNG1))
    ap.add_argument("--out", default=str(_HERE.parent / "artifacts"))
    a = ap.parse_args()
    d = torch.load(a.dump, map_location="cpu", weights_only=False)
    de = {k: v.double().numpy() for k, v in d["dense_de"].items()}
    eid, t0 = d["eid"], d["t0"]
    draws, n_ep = draws_for(eid)
    cv = de["d_constant_velocity"]

    out = {"stage": "A_frozen_latent", "gpu": "none",
           "source_dump": str(Path(a.dump).name),
           "n_windows": int(cv.shape[0]), "n_episode_clusters": n_ep,
           "K": int(cv.shape[1]), "n_boot": B_BOOT,
           "estimator": "paired_episode_cluster_bootstrap",
           "readout": "str (k=20 calibrated)",
           "note": ("FROZEN = blindimag state_source='frozen_last': the last "
                    "REAL percept held constant through the rollout while the "
                    "own_kinematic action channel proceeds normally. It is the "
                    "PI's ablation 1 and it is ALREADY the comparator arm of the "
                    "program's T_blind.")}

    # ---- G1(partial): window bookkeeping ---------------------------------- #
    out["gate_windows"] = {
        "n_windows": int(cv.shape[0]), "expected": 599,
        "n_episode_clusters": n_ep, "expected_clusters": 596,
        "n_t0_zero": int(sum(1 for x in t0 if int(x) == 0)),
        "PASS": bool(cv.shape[0] == 599 and n_ep == 596)}

    # ---- G3: fidelity to the committed headline ---------------------------- #
    fid, fid_ok = {}, True
    rows = {}
    for al, (ai, bi_) in ALPHAS.items():
        tb = t_blind(de[ai], de[bi_], draws, label_a=ai, label_b=bi_)
        c = COMMITTED[al]
        got = {"T_blind": tb["T_blind_steps"],
               "de2s": round(float(de[ai][:, 19].mean()), 4),
               "ade": round(float(ade_0_2s(de[ai], eid)["mean"]), 4),
               "tu1m": t_useful(de[ai], 1.0),
               "beats_cv": separated_better_interval(de[ai], cv, draws)["n_steps"]}
        lvl_ok = (abs(got["de2s"] - c["de2s"]) < 5e-4
                  and abs(got["ade"] - c["ade"]) < 5e-4
                  and got["tu1m"] == c["tu1m"]
                  and got["beats_cv"] == c["beats_cv"])
        fid[f"alpha={al}"] = {"committed": c, "recomputed": got,
                              "level_agreement": bool(lvl_ok),
                              "T_blind_integer_matches":
                                  bool(got["T_blind"] == c["T_blind"])}
        fid_ok &= lvl_ok
        rows[al] = tb
    fid["LEVEL_FIDELITY_PASS"] = bool(fid_ok)
    out["gate_fidelity"] = fid
    if not fid_ok:
        out["ABORT"] = "G3 level fidelity FAILED — nothing below is quotable"

    # ---- the ablation table ------------------------------------------------ #
    tab = {}
    for al, (ai, bi_) in ALPHAS.items():
        di, dx = de[ai], de[bi_]
        de2s_i, de2s_x = float(di[:, 19].mean()), float(dx[:, 19].mean())
        tb = rows[al]
        blk = {
            "arms": {"INTACT": ai, "FROZEN": bi_},
            "model_share_of_command_pct": round((1 - float(al)) * 100, 1),
            # --- PRIMARY ------------------------------------------------- #
            "de_2s": {"INTACT": single_at(di, eid, draws, 20),
                      "FROZEN": single_at(dx, eid, draws, 20)},
            "R_FROZEN_at_2s": round((de2s_x - de2s_i) / de2s_i, 4),
            "paired_2s_intact_minus_frozen":
                paired_at(di, dx, draws, 20),
            "de_6s": {"INTACT": single_at(di, eid, draws, 60),
                      "FROZEN": single_at(dx, eid, draws, 60)},
            "R_FROZEN_at_6s": round(
                (float(dx[:, 59].mean()) - float(di[:, 59].mean()))
                / float(di[:, 59].mean()), 4),
            # --- CO-PRIMARY ---------------------------------------------- #
            "T_blind_INTACT_vs_FROZEN": {
                "steps": tb["T_blind_steps"], "s": tb["T_blind_s"],
                "ci95_s": tb["T_blind_ci95_s"],
                "frac_draws_at_floor_1step":
                    tb["frac_draws_T_blind_at_floor_1step"],
                "C14_saturated": tb["C14_saturated_at_grid_terminus"]},
            "T_blind_FROZEN_vs_FROZEN": {
                "VACUOUS": True, "structural_value_steps": 1,
                "why": ("t_blind's comparator IS frozen_last; an arm against "
                        "itself returns the rule's failing floor by "
                        "construction. Not adjudicated (PRE_REGISTRATION §1.3)")},
            # --- capability tier (M10 split) ------------------------------ #
            "capability": {
                "beats_cv": {
                    "INTACT": separated_better_interval(di, cv, draws),
                    "FROZEN": separated_better_interval(dx, cv, draws)},
                "T_useful_s": {
                    "INTACT": {f"{b:g}m": t_useful(di, b) for b in BARS},
                    "FROZEN": {f"{b:g}m": t_useful(dx, b) for b in BARS}},
                "ade_0_2s": {"INTACT": ade_0_2s(di, eid),
                             "FROZEN": ade_0_2s(dx, eid)}},
            "grid": _grid_block(dx, di, draws),
        }
        tab[f"alpha={al}"] = blk
        print(f"[A] alpha={al:>4}  INTACT de@2s={de2s_i:.4f}  "
              f"FROZEN de@2s={de2s_x:.4f}  R={blk['R_FROZEN_at_2s']:+.4f}  "
              f"T_blind={tb['T_blind_steps']:3d}  "
              f"beatsCV int/frz="
              f"{blk['capability']['beats_cv']['INTACT']['n_steps']}/"
              f"{blk['capability']['beats_cv']['FROZEN']['n_steps']}  "
              f"Tuseful1m int/frz="
              f"{blk['capability']['T_useful_s']['INTACT']['1m']}/"
              f"{blk['capability']['T_useful_s']['FROZEN']['1m']}", flush=True)
    out["table"] = tab

    # ---- the comparator-free floors, for context --------------------------- #
    out["floors"] = {
        "constant_velocity": {"de_2s": single_at(cv, eid, draws, 20),
                              "ade_0_2s": ade_0_2s(cv, eid),
                              "T_useful_1m_s": t_useful(cv, 1.0)},
        "hold_v0": {"de_2s": single_at(de["d2_hold_v0"], eid, draws, 20),
                    "ade_0_2s": ade_0_2s(de["d2_hold_v0"], eid),
                    "T_useful_1m_s": t_useful(de["d2_hold_v0"], 1.0)}}

    Path(a.out).mkdir(parents=True, exist_ok=True)
    p = Path(a.out) / "la_stage_a_frozen.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[A] wrote {p}")
    return 0 if fid_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
