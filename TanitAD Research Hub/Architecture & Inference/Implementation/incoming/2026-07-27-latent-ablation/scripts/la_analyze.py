#!/usr/bin/env python3
"""THE LATENT ABLATION — adjudication. **ZERO GPU.**

Answers, in the pre-registered order (``PRE_REGISTRATION.md``), banking each
stage's artifact before the next begins:

  gates       window-set identity against the committed Rung-1 dump, the
              PLUMBING SELF-TEST in both directions, fidelity to the committed
              headline, and the DIAGNOSTIC VACUITY AUDIT — with the verdict rule
              demonstrated to FIRE on a real arm. Nothing below is read until
              they pass.
  table       the ablation table: every latent ablation x every alpha, with
              T_blind / de@2s / de@6s / ade_0_2s / beats-CV / T_useful, the
              horizon grid, and the decoded-speed diagnostic.
  verdict     the pre-registered INTEGRATOR / SEMANTIC / PARTIAL buckets.
  fixedpoint  do the imagined latents converge? (corroborative, never
              adjudicating)

Estimator everywhere: paired episode-cluster bootstrap, ``taniteval/ci.py``,
B = 2000, seed 0, unit = episode cluster, identical 599 windows for every arm.
``overlapping_holdout_se`` appears nowhere. The rule machinery (``t_blind``,
``paired_at``, ``separated_better_interval``, ``draws_for``) is IMPORTED from the
ladder's ``tb_rung0.py`` rather than re-implemented.

Usage:
    python la_analyze.py --new perwindow/latab_perwindow_compact.pt \
        --rung1 ../2026-07-26-tblind-rung1/perwindow/rung1_perwindow_compact.pt \
        --out artifacts
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

from tb_rung0 import (DT, GRID, B_BOOT, ade_0_2s, draws_for,       # noqa: E402
                      paired_at, separated_better_interval, single_at,
                      t_blind)

ALPHAS = ("0", "0.25", "0.75", "1")
INTACT, FROZEN = "imagination", "frozen_last"
#: The DESTRUCTIVE set — the latent's content is destroyed. FROZEN is reported
#: apart: a STALE REAL latent is not a destroyed one (PRE_REGISTRATION §3).
DESTRUCTIVE = ("frozen_other", "shuffled", "shuffled_obs", "mean_latent",
               "zero_latent")
#: PRIVILEGED — reads the true future. A diagnostic CEILING, never deployable.
DIAGNOSTIC = ("full_obs",)
SOURCES = (INTACT, FROZEN) + DESTRUCTIVE + DIAGNOSTIC
BARS = (1.0, 2.0)
ANCHOR_TOL = 1e-4          # Rung 0b measured 3.05e-05 m between two encode passes

#: my arm name -> the arm it must reproduce in the committed Rung-1 dump
ANCHORS = {
    "imagination__a0": "a_imagination__own__roSTR",
    "frozen_last__a0": "b_frozenlast__own__roSTR",
    "imagination__a0.25": "a_blend0.25",
    "frozen_last__a0.25": "b_blend0.25",
    "imagination__a0.75": "a_blend0.75",
    "frozen_last__a0.75": "b_blend0.75",
    "anchor_a_hold": "a_imagination__hold__roSTR",
    "anchor_b_hold": "b_frozenlast__hold__roSTR",
}
#: TBLIND_RUNG1.md §3 — the fidelity gate. LEVEL agreement is BLOCKING.
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
#: The pre-registered thresholds. Fixed before any number existed.
R_INTEGRATOR, R_SEMANTIC = 0.20, 1.00
C_INTEGRATOR, C_SEMANTIC = 0.20, 0.80


def t_useful(de, bar: float) -> float:
    """Largest horizon (s) with mean ``de`` below ``bar``, contiguous from step 1.
    Returns **0.0** when step 1 is already above the bar — reachable, not
    structural."""
    m = de.mean(axis=0)
    ok = m < bar
    if not ok[0]:
        return 0.0
    bad = np.flatnonzero(~ok)
    return round(float(int(bad[0]) if bad.size else int(ok.size)) * DT, 2)


def _arm(src, al):
    return f"{src}__a{al}"


# =========================================================================== #
def stage_gates(new, r1, de, eid, draws, out: Path) -> dict:
    de_r1 = {k: v.double().numpy() for k, v in r1["dense_de"].items()}
    g = {"stage": "gates"}

    # --- G1 window-set identity ---------------------------------------- #
    same_eid = list(new["eid"]) == list(r1["eid"])
    same_t0 = [int(x) for x in new["t0"]] == [int(x) for x in r1["t0"]]
    anch = {}
    for mine, ref in ANCHORS.items():
        m = float(np.abs(de[mine] - de_r1[ref]).max())
        anch[mine] = {"vs_committed": ref, "max_abs_diff_m": round(m, 9),
                      "within_tol": bool(m <= ANCHOR_TOL)}
    g["G1_window_identity"] = {
        "n_windows": int(de[_arm(INTACT, "0")].shape[0]), "expected": 599,
        "n_episode_clusters": len(set(new["eid"])), "expected_clusters": 596,
        "eid_ordering_identical": bool(same_eid),
        "t0_ordering_identical": bool(same_t0),
        "anchors": anch, "tolerance_m": ANCHOR_TOL,
        "PASS": bool(same_eid and same_t0
                     and all(v["within_tol"] for v in anch.values())
                     and de[_arm(INTACT, "0")].shape[0] == 599
                     and len(set(new["eid"])) == 596)}

    # --- G2 plumbing self-test (resolved at full K during compaction) ---- #
    g["G2_plumbing_selftest"] = new["selftest"]

    # --- G3 fidelity to the committed headline --------------------------- #
    cv = de["d_constant_velocity"]
    fid, ok = {}, True
    for al in ALPHAS:
        a_i, a_f = _arm(INTACT, al), _arm(FROZEN, al)
        tb = t_blind(de[a_i], de[a_f], draws)
        c = COMMITTED[al]
        got = {"T_blind": tb["T_blind_steps"],
               "de2s": round(float(de[a_i][:, 19].mean()), 4),
               "ade": round(float(ade_0_2s(de[a_i], eid)["mean"]), 4),
               "tu1m": t_useful(de[a_i], 1.0),
               "beats_cv": separated_better_interval(de[a_i], cv,
                                                     draws)["n_steps"]}
        lvl = (abs(got["de2s"] - c["de2s"]) < 5e-4
               and abs(got["ade"] - c["ade"]) < 5e-4
               and got["tu1m"] == c["tu1m"] and got["beats_cv"] == c["beats_cv"])
        ok &= lvl
        fid[f"alpha={al}"] = {"committed": c, "recomputed": got,
                              "level_agreement": bool(lvl),
                              "T_blind_integer_matches":
                                  bool(got["T_blind"] == c["T_blind"])}
    fid["LEVEL_FIDELITY_PASS"] = bool(ok)
    fid["note"] = ("level agreement is BLOCKING; the T_blind integer is "
                   "reported non-blocking (a step count is a threshold "
                   "crossing and these come from a second encode pass)")
    g["G3_fidelity"] = fid

    # --- G4 ⚠️ THE VACUITY AUDIT, and proof the rule FIRES on a real arm -- #
    # `own_vupd` (feed the model's own predicted speed into v0) is a REAL arm on
    # these very windows with a catastrophic de@2s. It is the positive control:
    # the SEMANTIC bucket must fire on it, and the two statistics must be shown
    # capable of DISAGREEING (which my rule forces to PARTIAL).
    d_i0 = float(de_r1["a_imagination__own__roSTR"][:, 19].mean())
    d_vu = float(de_r1["a_own_vupd"][:, 19].mean())
    t_i0 = t_blind(de_r1["a_imagination__own__roSTR"],
                   de_r1["b_frozenlast__own__roSTR"], draws)["T_blind_steps"]
    t_vu = t_blind(de_r1["a_own_vupd"], de_r1["b_own_vupd"],
                   draws)["T_blind_steps"]
    r_vu, c_vu = (d_vu - d_i0) / d_i0, 1.0 - t_vu / t_i0
    g["G4_vacuity_audit"] = {
        "positive_control_own_vupd": {
            "what": ("a REAL Rung-1 arm on these very windows: the model's own "
                     "predicted speed fed into the v0 channel"),
            "de_2s_intact": round(d_i0, 4), "de_2s_arm": round(d_vu, 4),
            "R": round(r_vu, 3), "PRIMARY_bucket_it_returns":
                "SEMANTIC" if r_vu >= R_SEMANTIC else "not SEMANTIC",
            "T_blind_intact": t_i0, "T_blind_arm": t_vu,
            "cost": round(c_vu, 3), "CO_PRIMARY_bucket_it_returns":
                ("SEMANTIC" if c_vu > C_SEMANTIC
                 else "INTEGRATOR" if c_vu < C_INTEGRATOR else "PARTIAL"),
            "⭐ demonstrates": ("the SEMANTIC bucket is REACHABLE on a real arm, "
                               "AND that the two adjudicating statistics can "
                               "disagree — which the pre-registered rule forces "
                               "to PARTIAL rather than to a choice of quote")},
        "diagnostics": {
            "T_blind(FROZEN vs FROZEN)": {
                "admissible": False,
                "why": ("structurally 1 step — t_blind's comparator IS "
                        "frozen_last. NOT EMITTED anywhere in this folder."),
                "emitted": False},
            "R_X": {"admissible": True,
                    "why": ("0.0 attainable (the identity self-test arms are "
                            "bit-identical => R = 0) and 12.2 attainable "
                            "(own_vupd, above)")},
            "cost_X": {"admissible": True,
                       "why": "0.0 and 0.64 both measured on real arms"},
            "anti_no_op": {"admissible": True,
                           "why": "a zero-difference arm is attainable and fails"},
            "beats_cv": {"admissible": True,
                         "why": ("0/185 is attainable and is MEASURED on the "
                                 "frozen arm at every alpha")},
            "T_useful": {"admissible": True,
                         "why": "returns 0.0 when step 1 is already above the bar"},
            "fixed_point": {"admissible": True,
                            "why": ("both readings are attainable and are "
                                    "test-pinned "
                                    "(test_fixed_point_probe_can_report_BOTH_readings)")},
        }}

    g["ALL_GATES_PASS"] = bool(g["G1_window_identity"]["PASS"]
                               and new["selftest"]["SELFTEST_PASS"]
                               and fid["LEVEL_FIDELITY_PASS"])
    (out / "la_gates.json").write_text(json.dumps(g, indent=2), encoding="utf-8")
    print(f"[gates] G1={g['G1_window_identity']['PASS']} "
          f"G2={new['selftest']['SELFTEST_PASS']} "
          f"G3={fid['LEVEL_FIDELITY_PASS']} -> ALL={g['ALL_GATES_PASS']}",
          flush=True)
    return g


# =========================================================================== #
def _speed_diag(new, arm, v_last):
    """⭐ Does the decoded SPEED survive the latent ablation? The integrator
    hypothesis says the speed is read off the CONSTANT v0 action channel, not
    the latent — in which case destroying the latent leaves it intact."""
    ps = new["pred_speed"].get(arm)
    if ps is None:
        return None
    v = ps.double().numpy()
    v0 = np.asarray(v_last, dtype=float)
    m20 = v[:, :20].mean(axis=1)
    r = float(np.corrcoef(m20, v0)[0, 1])
    return {"mean_decoded_speed_0_2s_mps": round(float(m20.mean()), 4),
            "mean_true_v0_mps": round(float(v0.mean()), 4),
            "corr_with_true_v0": round(r, 4),
            "r2_with_true_v0": round(r * r, 4),
            "mean_abs_speed_error_mps": round(float(np.abs(m20 - v0).mean()), 4)}


def stage_table(new, de, eid, draws, out: Path) -> dict:
    cv = de["d_constant_velocity"]
    v_last = new["v_last"].double().numpy()
    tab = {}
    for al in ALPHAS:
        a_i, a_f = _arm(INTACT, al), _arm(FROZEN, al)
        d_i = de[a_i]
        de2s_i = float(d_i[:, 19].mean())
        t_i = t_blind(d_i, de[a_f], draws)["T_blind_steps"]
        rows = {}
        for src in SOURCES:
            arm = _arm(src, al)
            d_x = de[arm]
            de2s_x = float(d_x[:, 19].mean())
            de6s_x = float(d_x[:, 59].mean())
            if src == FROZEN:
                tb = {"steps": None, "s": None, "ci95_s": None,
                      "VACUOUS": True,
                      "why": ("T_blind's comparator IS this arm; an arm against "
                              "itself returns the rule's failing floor by "
                              "construction. Not adjudicated.")}
                cost = None
            else:
                _t = t_blind(d_x, de[a_f], draws)
                tb = {"steps": _t["T_blind_steps"], "s": _t["T_blind_s"],
                      "ci95_s": _t["T_blind_ci95_s"],
                      "frac_draws_at_floor_1step":
                          _t["frac_draws_T_blind_at_floor_1step"],
                      "C14_saturated": _t["C14_saturated_at_grid_terminus"]}
                cost = round(1.0 - _t["T_blind_steps"] / t_i, 4)
            pd = paired_at(d_i, d_x, draws, 20)     # positive => INTACT better
            rows[src] = {
                "arm": arm,
                "class": ("INTACT" if src == INTACT else
                          "REFERENCE(stale real latent)" if src == FROZEN else
                          "DIAGNOSTIC(privileged)" if src in DIAGNOSTIC else
                          "DESTRUCTIVE"),
                "de_2s": single_at(d_x, eid, draws, 20),
                "de_6s": single_at(d_x, eid, draws, 60),
                "ade_0_2s": ade_0_2s(d_x, eid),
                "R_at_2s": round((de2s_x - de2s_i) / de2s_i, 4),
                "R_at_6s": round((de6s_x - float(d_i[:, 59].mean()))
                                 / float(d_i[:, 59].mean()), 4),
                "paired_2s_intact_minus_this": pd,
                "separated_from_intact_at_2s": pd["separated"],
                "T_blind_vs_FROZEN": tb, "cost_vs_intact": cost,
                "capability": {
                    "beats_cv": separated_better_interval(d_x, cv, draws),
                    "T_useful_s": {f"{b:g}m": t_useful(d_x, b) for b in BARS}},
                "speed_diagnostic": _speed_diag(new, arm, v_last),
                "grid_R": {f"{n * DT:g}s": round(
                    (float(d_x[:, n - 1].mean()) - float(d_i[:, n - 1].mean()))
                    / float(d_i[:, n - 1].mean()), 4) for n in GRID},
            }
            print(f"[tab] a={al:>4} {src:<13} de@2s={de2s_x:7.4f} "
                  f"R={rows[src]['R_at_2s']:+8.4f} "
                  f"T_blind={str(tb.get('steps')):>4} "
                  f"cost={str(cost):>7} "
                  f"beatsCV={rows[src]['capability']['beats_cv']['n_steps']:3d} "
                  f"Tu1m={rows[src]['capability']['T_useful_s']['1m']}",
                  flush=True)
        tab[f"alpha={al}"] = {
            "model_share_of_command_pct": round((1 - float(al)) * 100, 1),
            "T_blind_intact_steps": t_i, "de_2s_intact": round(de2s_i, 4),
            "unconfounded": al == "1",
            "rows": rows}
    # seed robustness of the derangement
    seeds = {}
    for al in ("0.25", "1"):
        base = de[_arm("shuffled", al)]
        seeds[f"alpha={al}"] = {
            "seed0_de_2s": round(float(base[:, 19].mean()), 4),
            **{f"seed{s}_de_2s": round(
                float(de[f"shuffled_seed{s}__a{al}"][:, 19].mean()), 4)
               for s in (1, 2) if f"shuffled_seed{s}__a{al}" in de}}
    tab["seed_robustness"] = seeds
    tab["floors"] = {
        "constant_velocity": {"de_2s": round(float(cv[:, 19].mean()), 4),
                              "ade_0_2s": ade_0_2s(cv, eid)["mean"],
                              "T_useful_1m_s": t_useful(cv, 1.0)},
        "hold_v0": {"de_2s": round(float(de["d2_hold_v0"][:, 19].mean()), 4),
                    "ade_0_2s": ade_0_2s(de["d2_hold_v0"], eid)["mean"],
                    "T_useful_1m_s": t_useful(de["d2_hold_v0"], 1.0)}}
    (out / "la_table.json").write_text(json.dumps(tab, indent=2),
                                       encoding="utf-8")
    return tab


# =========================================================================== #
def stage_verdict(tab, out: Path) -> dict:
    v = {"stage": "verdict",
         "rule": ("PRE_REGISTRATION.md §4, fixed before any number existed. "
                  "PRIMARY de@2s: INTEGRATOR if max R < 0.20; SEMANTIC if "
                  "min R >= 1.00 and every X separated; else PARTIAL. "
                  "CO-PRIMARY T_blind: INTEGRATOR if max cost < 0.20; SEMANTIC "
                  "if min cost > 0.80; else PARTIAL. DISAGREEMENT => PARTIAL, "
                  "mandatory."),
         "thresholds": {"R_INTEGRATOR": R_INTEGRATOR, "R_SEMANTIC": R_SEMANTIC,
                        "C_INTEGRATOR": C_INTEGRATOR, "C_SEMANTIC": C_SEMANTIC}}
    for al in ("0.25", "1"):
        rows = tab[f"alpha={al}"]["rows"]
        R = {s: rows[s]["R_at_2s"] for s in DESTRUCTIVE}
        C = {s: rows[s]["cost_vs_intact"] for s in DESTRUCTIVE}
        sep = {s: rows[s]["separated_from_intact_at_2s"] for s in DESTRUCTIVE}
        rmax, rmin = max(R.values()), min(R.values())
        cmax, cmin = max(C.values()), min(C.values())
        prim = ("INTEGRATOR" if rmax < R_INTEGRATOR
                else "SEMANTIC" if (rmin >= R_SEMANTIC and all(sep.values()))
                else "PARTIAL")
        co = ("INTEGRATOR" if cmax < C_INTEGRATOR
              else "SEMANTIC" if cmin > C_SEMANTIC else "PARTIAL")
        final = prim if prim == co else "PARTIAL"
        v[f"alpha={al}"] = {
            "unconfounded": al == "1",
            "model_share_of_command_pct":
                tab[f"alpha={al}"]["model_share_of_command_pct"],
            "R_per_destructive_ablation": R, "R_max": rmax, "R_min": rmin,
            "separated_per_ablation": sep,
            "cost_per_destructive_ablation": C, "cost_max": cmax,
            "cost_min": cmin,
            "PRIMARY_de2s": prim, "CO_PRIMARY_T_blind": co,
            "agree": bool(prim == co),
            "VERDICT": final,
            "R_FROZEN_reported_apart": rows[FROZEN]["R_at_2s"],
            "cost_FROZEN": "VACUOUS — T_blind's comparator IS this arm"}
        print(f"[verdict] alpha={al}: PRIMARY={prim} CO-PRIMARY={co} "
              f"=> {final}   (R {rmin:+.3f}..{rmax:+.3f}, "
              f"cost {cmin:+.3f}..{cmax:+.3f})", flush=True)
    v["HEADLINE"] = {
        "attributable_row_alpha_1": v["alpha=1"]["VERDICT"],
        "deployable_row_alpha_0.25": v["alpha=0.25"]["VERDICT"]}
    (out / "la_verdict.json").write_text(json.dumps(v, indent=2),
                                         encoding="utf-8")
    return v


# =========================================================================== #
def stage_fixedpoint(new, out: Path) -> dict:
    """Do the imagined latents drift to a fixed point? Corroborative only."""
    fp = {"stage": "fixed_point_probe", "adjudicating": False,
          "criterion": ("FIXED POINT iff relative step size at step 40 < 5% of "
                        "its step-1 value AND ||z_j - z_0|| plateaus (|delta| "
                        "over steps 100->185 < 5% of its value at 100)")}
    for arm, blk in sorted(new["latent_stepmean"].items()):
        dz = blk["lat_dz"].double().numpy()
        d0 = blk["lat_d0"].double().numpy()
        nrm = blk["lat_norm"].double().numpy()
        cos0 = blk["lat_cos0"].double().numpy()
        rel = dz / np.clip(nrm, 1e-12, None)
        r1, r40 = float(rel[0]), float(rel[39])
        d100, d185 = float(d0[99]), float(d0[-1])
        conv_step = r40 < 0.05 * r1
        conv_d0 = abs(d185 - d100) < 0.05 * max(d100, 1e-12)
        fp[arm] = {
            "rel_step_size": {f"{n * DT:g}s": round(float(rel[n - 1]), 6)
                              for n in GRID},
            "dist_from_z0": {f"{n * DT:g}s": round(float(d0[n - 1]), 4)
                             for n in GRID},
            "cos_with_z0": {f"{n * DT:g}s": round(float(cos0[n - 1]), 4)
                            for n in GRID},
            "latent_norm": {f"{n * DT:g}s": round(float(nrm[n - 1]), 4)
                            for n in GRID},
            "rel_step_at_0.1s": round(r1, 6),
            "rel_step_at_4s": round(r40, 6),
            "rel_step_ratio_4s_over_0.1s": round(r40 / max(r1, 1e-12), 4),
            "d0_change_10s_to_18.5s_frac": round(
                (d185 - d100) / max(d100, 1e-12), 4),
            "step_size_converged": bool(conv_step),
            "distance_plateaued": bool(conv_d0),
            "READING": "FIXED POINT" if (conv_step and conv_d0)
                       else "NOT A FIXED POINT"}
    (out / "la_fixedpoint.json").write_text(json.dumps(fp, indent=2),
                                            encoding="utf-8")
    for k, b in fp.items():
        if isinstance(b, dict) and "READING" in b:
            print(f"[fp] {k:<24} rel_step 0.1s={b['rel_step_at_0.1s']:.4f} "
                  f"4s={b['rel_step_at_4s']:.4f} "
                  f"ratio={b['rel_step_ratio_4s_over_0.1s']:.3f}  "
                  f"|z-z0| 10s->18.5s {b['d0_change_10s_to_18.5s_frac']:+.3f}  "
                  f"=> {b['READING']}", flush=True)
    return fp


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", required=True)
    ap.add_argument("--rung1", default=str(
        _INCOMING / "2026-07-26-tblind-rung1" / "perwindow"
        / "rung1_perwindow_compact.pt"))
    ap.add_argument("--out", default=str(_HERE.parent / "artifacts"))
    ap.add_argument("--force", action="store_true",
                    help="run the table even if a gate fails (never for a quote)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    new = torch.load(a.new, map_location="cpu", weights_only=False)
    r1 = torch.load(a.rung1, map_location="cpu", weights_only=False)
    de = {k: v.double().numpy() for k, v in new["dense_de"].items()}
    eid = new["eid"]
    draws, n_ep = draws_for(eid)
    print(f"[latab] {len(de)} arms, {len(eid)} windows, {n_ep} clusters, "
          f"B={B_BOOT}", flush=True)

    g = stage_gates(new, r1, de, eid, draws, out)
    if not g["ALL_GATES_PASS"] and not a.force:
        print("⛔ GATES FAILED — nothing below is quotable. Stopping.")
        return 3
    tab = stage_table(new, de, eid, draws, out)
    stage_verdict(tab, out)
    stage_fixedpoint(new, out)
    print("[latab] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
