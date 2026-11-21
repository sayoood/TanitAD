#!/usr/bin/env python3
"""SOTA_SCAN §11 item 2 — EXECUTED. 0 GPU, dev-box CPU.

Four things, in the order the operating standard requires (control FIRST, instrument SECOND):

  D-PROG-2  self-consistency: the module's progress_ratio vs an INDEPENDENT numpy recompute.
  D-PROG-1  discrimination:  GT-as-arm vs hold-v0 CV on progress_error, paired episode-cluster
                             bootstrap. If this fails, nothing below may be quoted.
  E2c       the scan's registered test: Kendall tau between the arm ranking by ade_0_2s and by
                             progress_error, over every banked window dump.
  E2d       the DECISION-GRADE test the scan could not ask for: within-model, per-window Spearman
                             rho between each OPEN-loop metric and the CLOSED-loop error, on the
                             arms that have both, with an episode-cluster bootstrap on rho itself.

Pre-registration: ../PRE_REGISTRATION.md  (written before this file ran).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "taniteval"))

from taniteval import ci as CI                      # noqa: E402
from taniteval import four_families as FF           # noqa: E402
from taniteval import progress as PROG              # noqa: E402

BANK = REPO / "taniteval" / "results"
CLBANK = (REPO / "TanitAD Research Hub" / "Benchmarks & Eval" / "Implementation" / "incoming"
          / "2026-07-26-closedloop-artifact-rerun" / "raw_windows")
OUT = Path(__file__).resolve().parents[1] / "raw"
B, SEED = 2000, 0


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def ade_per_window(pred, gt):
    """The programme's ade_0_2s per window — bench._suite_components, verbatim shape."""
    de = torch.linalg.norm(torch.as_tensor(pred) - torch.as_tensor(gt), dim=-1).numpy()
    return de.mean(axis=1).astype(np.float64)


def rankdata(x):
    """Average ranks, ties shared. Independent of scipy so the estimator has no hidden dependency."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # average tied ranks
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def kendall_tau_b(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    conc = disc = ta = tb = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            s = np.sign(da) * np.sign(db)
            if da == 0 and db == 0:
                ta += 1
                tb += 1
            elif da == 0:
                ta += 1
            elif db == 0:
                tb += 1
            elif s > 0:
                conc += 1
            else:
                disc += 1
    n0 = n * (n - 1) / 2
    den = np.sqrt((n0 - ta) * (n0 - tb))
    return float((conc - disc) / den) if den > 0 else float("nan")


def spearman_ci(a, b, eid, n_boot=B, seed=SEED):
    """Episode-cluster bootstrap on Spearman rho ITSELF.

    The resampling unit is the val EPISODE, exactly as for a mean, because windows inside one clip
    are strongly dependent. Passing a CALLABLE reducer is the documented path in ci.resolve_reducer
    — it is how kappa/AUC are bootstrapped in this suite — so rho gets the same estimator as every
    other number rather than inventing its own interval.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    stacked = np.stack([a, b], axis=1)
    uniq, idx_by_ep = CI.episode_index(eid)
    point = spearman(a, b)
    draws = []
    rng_draws = CI._draws(uniq, idx_by_ep, n_boot, seed)
    for sel in rng_draws:
        s = stacked[sel]
        draws.append(spearman(s[:, 0], s[:, 1]))
    d = np.asarray(draws, float)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"rho": round(point, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "separated_from_0": bool(lo > 0 or hi < 0),
            "n_windows": int(len(a)), "n_episodes": int(len(uniq)),
            "n_boot": n_boot, "estimator": "episode_cluster_bootstrap_on_spearman_rho"}


# --------------------------------------------------------------------------- #
# load the bank                                                                #
# --------------------------------------------------------------------------- #
def load_bank():
    arms = {}
    for p in sorted(BANK.glob("windows_*.pt")):
        d = torch.load(p, map_location="cpu", weights_only=False)
        if not {"pred", "gt", "cv", "eid"} <= set(d):
            continue
        arms[p.stem.replace("windows_", "")] = d
    return arms


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    arms = load_bank()
    print(f"loaded {len(arms)} banked arms from {BANK}")

    # ---- parity: a cross-arm ranking is only valid on IDENTICAL windows ---------------------
    ref_name = "flagship-30k"
    ref = arms[ref_name]
    parity = {}
    for k, d in arms.items():
        same_gt = (d["gt"].shape == ref["gt"].shape
                   and bool(torch.allclose(d["gt"], ref["gt"], atol=1e-6)))
        parity[k] = {"n": int(d["pred"].shape[0]), "n_eps": len(set(d["eid"])),
                     "gt_identical_to_ref": same_gt,
                     "eid_identical_to_ref": list(d["eid"]) == list(ref["eid"])}
    comparable = [k for k, v in parity.items()
                  if v["gt_identical_to_ref"] and v["eid_identical_to_ref"]]
    print(f"comparable arms (identical gt AND eid to {ref_name}): {len(comparable)}")

    eid = list(ref["eid"])
    gt = ref["gt"]
    cv = ref["cv"]

    # ======================================================================== #
    # D-PROG-2 — SELF-CONSISTENCY (component vs family), MANDATORY, FIRST       #
    # ======================================================================== #
    def independent_ratio(pred_t, gt_t):
        """A from-scratch recompute that shares NO code with taniteval.progress."""
        p = pred_t.numpy().astype(np.float64)
        g = gt_t.numpy().astype(np.float64)
        out = np.full(len(p), np.nan)
        for i in range(len(p)):
            gx, gy = g[i, -1, 0], g[i, -1, 1]
            chord = (gx * gx + gy * gy) ** 0.5
            if chord < 0.5:
                continue
            out[i] = (p[i, -1, 0] * gx + p[i, -1, 1] * gy) / (chord * chord)
        return out

    mod = PROG.progress_per_window(ref["pred"].numpy(), gt.numpy())["ratio"]
    ind = independent_ratio(ref["pred"], gt)
    both = ~np.isnan(mod) & ~np.isnan(ind)
    max_abs = float(np.abs(mod[both] - ind[both]).max())
    dprog2 = {"control": "D-PROG-2 self-consistency (module vs independent numpy reducer)",
              "n_compared": int(both.sum()),
              "max_abs_diff": max_abs,
              "nan_masks_agree": bool((np.isnan(mod) == np.isnan(ind)).all()),
              "tolerance": 1e-6,
              "verdict": "PASS" if max_abs < 1e-6 else "INSTRUMENT-FAIL"}
    print("D-PROG-2:", dprog2["verdict"], f"max|diff|={max_abs:.3e}")
    if dprog2["verdict"] != "PASS":
        json.dump({"D_PROG_2": dprog2}, open(OUT / "item2_ABORTED.json", "w"), indent=1)
        raise SystemExit("INSTRUMENT-FAIL: no verdict issued (pre-registered branch)")

    # ======================================================================== #
    # D-PROG-1 — DISCRIMINATION CONTROL: GT-as-arm vs hold-v0 CV                #
    # ======================================================================== #
    err_gt = PROG.progress_per_window(gt.numpy(), gt.numpy())["error"]
    err_cv = PROG.progress_per_window(cv.numpy(), gt.numpy())["error"]
    valid = ~np.isnan(err_gt) & ~np.isnan(err_cv)
    eid_v = [e for e, m in zip(eid, valid) if m]
    dprog1 = CI.paired_episode_cluster_bootstrap(err_cv[valid], err_gt[valid], eid_v,
                                                 n_boot=B, seed=SEED)
    dprog1_block = {
        "control": "D-PROG-1 discrimination (CV progress_error MINUS GT progress_error)",
        "orientation": "positive = CV is worse than GT, the expected sign",
        "delta": dprog1["delta"], "lo": dprog1["lo"], "hi": dprog1["hi"],
        "separated": dprog1["separated"], "p_delta_gt0": dprog1["p_delta_gt0"],
        "n_windows": dprog1["n_windows"], "n_episodes": dprog1["n_episodes"],
        "estimator": dprog1["estimator"],
        "gt_progress_error_mean": round(float(err_gt[valid].mean()), 6),
        "cv_progress_error_mean": round(float(err_cv[valid].mean()), 4),
        "n_excluded_low_gt_progress": int((~valid).sum()),
        "verdict": ("PASS — ADMISSIBLE" if dprog1["separated"] and dprog1["delta"] > 0
                    else "FAIL — the gauge cannot move; do NOT score arms with it"),
    }
    print("D-PROG-1:", dprog1_block["verdict"], dprog1["delta"], [dprog1["lo"], dprog1["hi"]])

    # ======================================================================== #
    # PER-ARM FOUR FAMILIES over every banked dump                             #
    # ======================================================================== #
    per_arm = {}
    for k in sorted(arms):
        d = arms[k]
        win = {"pred": d["pred"], "gt": d["gt"], "wp_steps": d.get("wp_steps"), "dt_s": 0.1}
        fam = FF.all_families(win)
        ade = ade_per_window(d["pred"], d["gt"])
        pe = PROG.progress_per_window(d["pred"].numpy(), d["gt"].numpy())["error"]
        v = ~np.isnan(pe)
        e_ = list(d["eid"])
        per_arm[k] = {
            "n_windows": int(d["pred"].shape[0]), "n_episodes": len(set(e_)),
            "ade_0_2s": CI.episode_cluster_bootstrap(ade, e_, n_boot=B, seed=SEED),
            "progress_error": CI.episode_cluster_bootstrap(
                pe[v], [x for x, m in zip(e_, v) if m], n_boot=B, seed=SEED),
            "families": fam,
        }
    print(f"scored {len(per_arm)} arms on the four families")

    # ======================================================================== #
    # E2c — Kendall tau between the two arm rankings                            #
    # ======================================================================== #
    ranked = [k for k in comparable]
    a_ade = [per_arm[k]["ade_0_2s"]["mean"] for k in ranked]
    a_prog = [per_arm[k]["progress_error"]["mean"] for k in ranked]
    tau = kendall_tau_b(a_ade, a_prog)
    if tau < 0.7:
        e2c = "PASS — progress-ratio is NON-REDUNDANT with ADE on our corpus"
    elif tau > 0.9:
        e2c = "FAIL — progress adds nothing on our corpus; ADE is a sufficient proxy for it"
    else:
        e2c = ("INDETERMINATE on ranking alone (pre-registered band 0.7-0.9) — decided by E2d")
    e2c_block = {"kendall_tau_b": round(tau, 4), "n_arms": len(ranked),
                 "spearman_rho_of_the_same_ranking": round(spearman(a_ade, a_prog), 4),
                 "verdict": e2c,
                 "⚠️_power": ("an arm-level tau over these arms is WEAK: they are not independent "
                              "(many are checkpoints of one lineage). Registered in advance as "
                              "unable to promote a metric on its own.")}
    print("E2c: tau =", round(tau, 4), "->", e2c)

    # ======================================================================== #
    # E2d — WITHIN-MODEL open->closed predictivity, per window, with a CI       #
    # ======================================================================== #
    e2d = {}
    for p in sorted(CLBANK.glob("clwin_*.pt")):
        name = p.stem.replace("clwin_", "")
        d = torch.load(p, map_location="cpu", weights_only=False)
        if "eid" not in d:
            e2d[name] = {"status": "SKIPPED", "reason": "no eid -> no episode-cluster bootstrap"}
            continue
        g, e_ = d["gt"], list(d["eid"])
        closed = ade_per_window(d["closed_bike"], g)          # the OUTCOME
        preds = {"open_grnd": d["open_grnd"], "open_bike": d["open_bike"], "cv": d["cv"]}
        block = {"n_windows": len(e_), "n_episodes": len(set(e_)),
                 "outcome": "closed_bike ADE@2s per window",
                 "closed_ade_mean": round(float(closed.mean()), 4)}
        for src, pr in preds.items():
            ade = ade_per_window(pr, g)
            fam_l = None
            pe = PROG.progress_per_window(pr.numpy(), g.numpy())["error"]
            # per-window LATERAL/LONGITUDINAL components on the same sparse grid
            P_ = FF._seq_geometry(torch.as_tensor(pr).float(), 0.5)
            G_ = FF._seq_geometry(torch.as_tensor(g).float(), 0.5)
            metrics = {
                "ade_0_2s": ade,
                "progress_error": pe,
                "speed_bias_abs": (P_["speed"] - G_["speed"]).mean(1).abs().numpy(),
                "along_bias_abs": (P_["along"] - G_["along"]).mean(1).abs().numpy(),
                "cross_mae": (P_["cross"] - G_["cross"]).abs().mean(1).numpy(),
                "heading_mae": (P_["heading"] - G_["heading"]).abs().mean(1).numpy(),
                "curvature_mae": (P_["curvature"] - G_["curvature"]).abs().mean(1).numpy(),
                "yaw_rate_mae": (P_["yaw_rate"] - G_["yaw_rate"]).abs().mean(1).numpy(),
            }
            sub = {}
            for mname, mv in metrics.items():
                mv = np.asarray(mv, float)
                ok = ~np.isnan(mv) & ~np.isnan(closed)
                sub[mname] = spearman_ci(mv[ok], closed[ok],
                                         [x for x, m in zip(e_, ok) if m])
            block[src] = sub
            _ = fam_l
        e2d[name] = block
        print(f"E2d {name}: open_grnd ade rho={block['open_grnd']['ade_0_2s']['rho']}, "
              f"progress rho={block['open_grnd']['progress_error']['rho']}")

    result = {
        "run": "SOTA_SCAN §11 item 2 — progress-ratio + open->closed predictivity",
        "date": "2026-08-03", "host": "dev-box CPU", "gpu_hours": 0.0,
        "evidence_class": "MEASURED (ours)",
        "estimator": {"interval": "episode_cluster_bootstrap",
                      "delta": "paired_episode_cluster_bootstrap",
                      "rho": "episode_cluster_bootstrap_on_spearman_rho",
                      "n_boot": B, "seed": SEED, "resampling_unit": "val episode",
                      "refused": "overlapping_holdout_se"},
        "bank": str(BANK), "n_arms_loaded": len(arms), "parity": parity,
        "n_arms_comparable": len(comparable),
        "D_PROG_2_self_consistency": dprog2,
        "D_PROG_1_discrimination": dprog1_block,
        "per_arm": per_arm,
        "E2c_ranking": e2c_block,
        "E2d_open_to_closed_predictivity": e2d,
        "wall_clock_s": round(time.time() - t0, 1),
    }
    (OUT / "item2_progress_and_predictivity.json").write_text(
        json.dumps(result, indent=1, default=str), encoding="utf-8")
    print("wrote", OUT / "item2_progress_and_predictivity.json",
          f"in {result['wall_clock_s']}s")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    main()
