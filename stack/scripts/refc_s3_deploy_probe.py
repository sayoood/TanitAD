"""E-SEL-1D — the DEPLOYABLE consequence score, and what it can actually buy.

Pre-registration: ``…/incoming/2026-08-03-s3-deployable/PREREG_S3_DEPLOYABLE.md``
(staged BEFORE any statistic here was computed; its blob id is recorded below).
Parent: ``Project Steering/PREREG_D-SEL_REFC_SELECTION_SURFACE.md`` §6.2/§6.3.

E-SEL-1 measured Spearman **rho = 0.6657 / 0.6212** for

    s_oracle = -||law_head([pooled, fan_i]) - encode_pooled(frame_{t+5})||^2

and verdicted **S3 LIVE**. ``z_{t+5}`` is the FUTURE FRAME, and the thing S3
deploys (``refc_select.consequence_scores``, called at ``refc.py:1246-1251``)
never sees it. ⇒ **0.6657 is an UPPER BOUND on the information present, not S3's
delivered gain** — quoting it as an effect size is the C6 / I-JEPA-leak shape.

This probe reads ``refc_sel_dump_deploy.py``'s bank, which carries the
**deployable** score computed by calling the deployed function itself, and
reports it BESIDE the upper bound on the same windows, paired.

⛔ CONTROLS THAT CAN FIRE. E-SEL found C-shuffled was vacuous for a selection
argmax (permute-then-argmax is uniform for ANY score); for a Spearman it is
vacuous in a second way — ``E[rho] = 0`` analytically for any score — so it
establishes only ``rho != 0``. It is kept for continuity and is **never
load-bearing**. The load-bearing controls are:

  * **C-ctxswap** — ``pooled`` replaced by a Sattolo DERANGEMENT of the windows.
    Is the score reading the SCENE, or only trajectory shape?
  * **C-cv**      — a ZERO-parameter score: negative distance to the banked
    constant-velocity baseline. If a free heuristic matches S3, S3's parameter
    and its per-candidate ``law_head`` evaluation are unjustified.
  * **C-degenerate** — per-window std of the score along the candidate axis.
  * **C-reproduce-esel / C-oracle-reproduce** — the re-decode must land on the
    same fan and reproduce E-SEL's rho, or the paired contrast is void.

⛔ ``overlapping_holdout_se`` is never called: it is not a jackknife, not a valid
SE, and it biases the POINT estimate bidirectionally, up to a sign flip.

Run:
    OMP_NUM_THREADS=6 python stack/scripts/refc_s3_deploy_probe.py \\
        --bank "…/2026-08-03-s3-deployable/raw/fan_deploy_refc-base-30k.pt" \\
        --arm refc-base-30k --out "…/2026-08-03-s3-deployable/raw/"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "6")
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

_HERE = Path(__file__).resolve()
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import refc_sel_probe as P                                       # noqa: E402

PREREG_S3 = ("TanitAD Research Hub/Architecture & Inference/Implementation/"
             "incoming/2026-08-03-s3-deployable/PREREG_S3_DEPLOYABLE.md")

#: A TRANSCRIPTION of PREREG_S3_DEPLOYABLE.md §4/§4.1. No run may edit these.
#: ``rho_min`` is the PARENT pre-registration's own registered S3 bar, carried
#: over unchanged — neither weakened for the harder statistic nor tightened
#: after the fact.
S3_THRESHOLDS = {
    "D_FUND": ("rho_deploy separated from C-ctxswap AND separated from C-cv "
               "AND |rho_deploy| >= 0.10"),
    "D_MARGINAL": ("|rho_deploy| >= 0.10 and separated from shuffled, but NOT "
                   "separated from C-ctxswap or not from C-cv"),
    "D_NULL": ("rho_deploy CI includes the shuffled control, OR "
               "|rho_deploy| < 0.10  =>  drop S3 and say so"),
    "RED_FLAG_AUDIT": "rho_deploy separated BETTER than rho_oracle => audit",
    "S3_PAYS": ("LOEO paired dADE@2s separated in the improving direction AND "
                ">= 0.02 m"),
    "rho_min": 0.10,
    "free_win_m": 0.02,
    #: §4.1, fixed in advance.
    "alphas": [0.0, 0.05, -0.05, 0.1, -0.1, 0.2, -0.2, 0.5, -0.5,
               1.0, -1.0, 2.0, -2.0, 5.0, -5.0],
    "registered_prediction": ("rho_deploy = 0.10-0.35, most likely D-MARGINAL; "
                              "C-cv is predicted to be the control that bites"),
    "axis_rule": ("MEASURED: the selection gap is 89.28 % (base) / 87.60 % (XL) "
                  "LONGITUDINAL. A perfect reranker buys 0.0334 m cross-track "
                  "and moves heading/curvature/yaw-rate NOT AT ALL. A score "
                  "that helps only laterally CANNOT pay for itself."),
}

#: E-SEL-1's published upper bound, for C-oracle-reproduce.
ESEL_RHO = {
    "refc-base-30k": {"rho": 0.6657, "lo": 0.6183, "hi": 0.7157},
    "refc-xl-30k": {"rho": 0.6212, "lo": 0.5650, "hi": 0.6791},
}


# =========================================================================== #
# statistics                                                                  #
# =========================================================================== #

def per_window_spearman(score: np.ndarray, neg_ade: np.ndarray) -> np.ndarray:
    """Per-window Spearman rho between a candidate score and -ADE."""
    from scipy.stats import spearmanr                            # noqa: PLC0415
    return np.array([spearmanr(score[i], neg_ade[i]).statistic
                     for i in range(score.shape[0])], dtype=np.float64)


def rho_block(name: str, score: np.ndarray, neg_ade: np.ndarray, eid,
              *, sees_future: bool, provenance: str) -> dict:
    per = per_window_spearman(score, neg_ade)
    b = P._boot(per, eid)
    b["_per_window"] = per
    b["name"] = name
    b["sees_future_frame"] = sees_future
    b["provenance"] = provenance
    b["n_nan_windows"] = int(np.isnan(per).sum())
    return b


def rho_on_reachable(score: np.ndarray, neg_ade: np.ndarray,
                     mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Spearman restricted to the S2-REACHABLE candidates.

    ⭐ WHY THIS EXISTS. 72-74 % of REF-C's fan is outside a bounded-acceleration
    band around ``v0`` and deleting it is MEASURED exactly inert on ADE — i.e.
    nothing there is ever selected. A rank correlation over the FULL candidate
    axis is therefore dominated by candidates no selector can pick, and a high
    rho can coexist with a useless argmax. Restricting to the survivors asks
    whether the correlation is about the part of the fan that decides anything.
    """
    from scipy.stats import spearmanr                            # noqa: PLC0415
    out, used = [], 0
    for i in range(score.shape[0]):
        m = mask[i]
        if m.sum() < 3:                                # rho undefined/degenerate
            out.append(np.nan)
            continue
        used += 1
        out.append(spearmanr(score[i][m], neg_ade[i][m]).statistic)
    return np.array(out, dtype=np.float64), used


def zscore_rows(x: torch.Tensor) -> torch.Tensor:
    """Per-window standardisation so the gate alpha is scale-free."""
    return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-9)


def rerank_ade(de_all: torch.Tensor, logits: torch.Tensor,
               z: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``rank = anchor_logits + alpha * zscore(s_deploy)`` -> (idx, per-window ADE).

    ⚠️ UNCLAMPED. The real seam (``refc_select.graft_with_seam``) can only
    SHRINK a graft relative to the base norm, so this sweep is an UPPER BOUND on
    what the learned ``cons_gate`` can express — stated, not assumed.
    """
    idx = (logits + alpha * z).argmax(1)
    return idx, de_all.gather(1, idx[:, None]).squeeze(1)


def alpha_sweep(de_all, logits, z, eid, alphas) -> dict:
    out = {}
    for a in alphas:
        idx, ade = rerank_ade(de_all, logits, z, a)
        out[f"{a:+.2f}"] = {"alpha": a, "mean_ade": round(float(ade.mean()), 6),
                            "_idx": idx, "_ade": ade}
    return out


def loeo_alpha(de_all, logits, z, eid, alphas):
    """Leave-one-EPISODE-out gate selection — the honest realized number.

    Choosing alpha on the same 881 windows it is then scored on is selection ON
    TEST and is optimistic by construction. Here alpha is chosen on 39 episodes
    and applied to the 40th, for all 40 folds, so the reported effect is not
    contaminated by its own tuning.
    """
    eid = np.asarray(eid)
    uniq = np.unique(eid)
    cache = {a: rerank_ade(de_all, logits, z, a) for a in alphas}
    n = de_all.shape[0]
    out_ade = torch.zeros(n, dtype=torch.float64)
    out_idx = torch.zeros(n, dtype=torch.long)
    chosen = {}
    for e in uniq:
        held = eid == e
        train = ~held
        best_a, best_v = None, float("inf")
        for a in alphas:
            v = float(cache[a][1][torch.as_tensor(train)].mean())
            if v < best_v - 1e-12:
                best_v, best_a = v, a
        h = torch.as_tensor(held)
        out_ade[h] = cache[best_a][1][h].double()
        out_idx[h] = cache[best_a][0][h]
        chosen[int(e)] = best_a
    return out_idx, out_ade, chosen


# =========================================================================== #
# driver                                                                      #
# =========================================================================== #

def run(bank: str, arm: str, out_dir: str) -> dict:
    d = P.load_fan(bank)
    eid = d["eid"]
    fan, gt = d["fan"], d["gt"]
    de_all = P.candidate_ade(fan, gt)                      # [B, N] per-candidate
    de_or = de_all.min(1).values
    neg = -de_all.numpy()
    logits = d["logits"]

    res: dict = {
        "experiment": "E-SEL-1D — the DEPLOYABLE consequence score",
        "arm": arm,
        "question": ("what Spearman rho does the score S3 can ACTUALLY produce "
                     "at inference reach, with no access to z_{t+5} or any "
                     "other future frame — reported BESIDE E-SEL-1's "
                     "future-frame upper bound on the same windows"),
        "prereg_s3": {
            "path": PREREG_S3,
            "staged_blob": (P._git("ls-files", "-s", "--", PREREG_S3).split() or
                            ["", "<not staged>"])[1],
            "worktree_blob": P._git("hash-object", PREREG_S3),
            "thresholds": S3_THRESHOLDS,
        },
        "prereg_parent": P.prereg_provenance(),
        "estimator": {
            "point_and_interval": "episode_cluster_bootstrap",
            "paired": "paired_episode_cluster_bootstrap",
            "unit": "episode", "n_boot": P.N_BOOT,
            "⛔": "overlapping_holdout_se is NEVER called",
        },
        "bank": {k: d.get(k) for k in
                 ("_path", "_sha256", "host", "ckpt", "ckpt_step", "steps",
                  "nav_mode", "n_anchors", "raster", "grid_shape", "n_tokens",
                  "law_ahead", "episodes_scored", "episodes_dropped",
                  "torch_version", "ctxswap_seed", "ctxswap_is_derangement",
                  "ctxswap_cross_episode_frac", "candidate_axis_guard",
                  "deploy_provenance", "oracle_provenance",
                  "ctxswap_provenance", "wall_s")},
        "n_windows": int(fan.shape[0]),
        "n_episodes": int(len(set(eid))),
    }
    res["prereg_s3"]["thresholds_unmoved_since_staging"] = bool(
        res["prereg_s3"]["staged_blob"] == res["prereg_s3"]["worktree_blob"])

    # ------------------------------------------------------------- controls
    shipped = P.ranker_block(de_all, de_or, logits.argmax(1), eid, tag="shipped")
    ctrl: dict = {}
    # BINDING, not "fan-only NOT-APPLICABLE": this bank came from a real decode
    # and stamped the raster it was fed, so the control is given that raster and
    # will RAISE on a mismatch (R-2026-08-02-a: base returns numbers silently).
    ctrl["C-raster"] = P.control_raster(d, frames_hw=tuple(d["raster"]))
    ctrl["C-raster"]["asserted_again_at_dump_time"] = {
        "grid_shape": list(d["grid_shape"]), "n_tokens": d["n_tokens"]}
    ctrl["C-identity"] = P.control_identity(d, de_all, arm)
    ctrl["C-oracle-floor"] = P.control_oracle_floor(d, de_or, arm)
    ctrl["C-reproduce-esel"] = d.get("c_reproduce_esel", "NOT RUN")
    sd = d["cons_deploy"].std(dim=1)
    ctrl["C-degenerate"] = {
        "median_per_window_std": round(float(sd.median()), 8),
        "min_per_window_std": round(float(sd.min()), 8),
        "guard_verdict": d.get("candidate_axis_guard"),
        "threshold": "< 1e-6 => assert_candidate_axis would raise; rho is noise",
        "pass": bool(float(sd.min()) >= 1e-6),
        "can_fire": True,
    }

    # ------------------------------------------------------------------ rho
    rng = np.random.default_rng(20260803)
    dep = d["cons_deploy"].numpy().astype(np.float64)
    orc = d["cons_oracle"].numpy().astype(np.float64)
    cxs = d["cons_ctxswap"].numpy().astype(np.float64)
    s_cv = (-torch.linalg.norm(fan - d["cv"][:, None], dim=-1).mean(-1)) \
        .numpy().astype(np.float64)
    shf = np.stack([rng.permutation(dep[i]) for i in range(dep.shape[0])])

    R = {
        "deploy": rho_block(
            "rho_deploy", dep, neg, eid, sees_future=False,
            provenance=("conf_head(layer_norm(feat_proj(law_head([pooled, "
                        "fan_i])))) — refc_select.consequence_scores, THE "
                        "DEPLOYED FUNCTION. **THE HEADLINE.**")),
        "oracle": rho_block(
            "rho_oracle", orc, neg, eid, sees_future=True,
            provenance=("-||law_head([pooled, fan_i]) - z_{t+5}||^2 — E-SEL-1's "
                        "statistic. UPPER BOUND ONLY: uses the future frame.")),
        "ctxswap": rho_block(
            "rho_ctxswap", cxs, neg, eid, sees_future=False,
            provenance="s_deploy with `pooled` DERANGED across windows"),
        "cv": rho_block(
            "rho_cv", s_cv, neg, eid, sees_future=False,
            provenance=("-mean||fan_i - constant_velocity||: ZERO parameters, "
                        "zero compute, already in every bank")),
        # ⭐ THE INCUMBENT'S OWN rho. Without it the panel has no scale: S3 is
        # proposed as a GRAFT ONTO this score, so "is rho_deploy large" is the
        # wrong question and "is it large NEXT TO the score it is added to" is
        # the right one. `refc_rescorer.py:27` asserts 0.907 for it in PROSE —
        # measured here so the comparison is not INHERITED.
        "shipped": rho_block(
            "rho_shipped", logits.numpy().astype(np.float64), neg, eid,
            sees_future=False,
            provenance=("anchor_logits — the t=0 classifier score REF-C "
                        "ACTUALLY ships. The incumbent S3 would be grafted onto.")),
        "shuffled": rho_block(
            "rho_shuffled", shf, neg, eid, sees_future=False,
            provenance="s_deploy permuted along the candidate axis"),
    }
    pw = {k: v.pop("_per_window") for k, v in R.items()}
    res["rho"] = R
    res["rho"]["shuffled"]["⛔_vacuous"] = (
        "E[rho] = 0 analytically for ANY score under a candidate-axis "
        "permutation, so this control CANNOT FAIL and establishes only "
        "rho != 0. Reported for continuity with E-SEL-1; NEVER load-bearing. "
        "The load-bearing controls are C-ctxswap and C-cv.")

    pair = {
        "deploy_minus_shuffled": P._paired(pw["deploy"], pw["shuffled"], eid),
        "deploy_minus_ctxswap": P._paired(pw["deploy"], pw["ctxswap"], eid),
        "deploy_minus_cv": P._paired(pw["deploy"], pw["cv"], eid),
        "deploy_minus_oracle": P._paired(pw["deploy"], pw["oracle"], eid),
        "deploy_minus_shipped": P._paired(pw["deploy"], pw["shipped"], eid),
        "oracle_minus_shipped": P._paired(pw["oracle"], pw["shipped"], eid),
        "oracle_minus_shuffled": P._paired(pw["oracle"], pw["shuffled"], eid),
        "ctxswap_minus_shuffled": P._paired(pw["ctxswap"], pw["shuffled"], eid),
    }
    res["paired_rho"] = pair

    ctrl["C-ctxswap"] = {
        "rho_ctxswap": R["ctxswap"]["mean"],
        "paired_deploy_minus_ctxswap": pair["deploy_minus_ctxswap"],
        "derangement": d.get("ctxswap_is_derangement"),
        "cross_episode_frac": d.get("ctxswap_cross_episode_frac"),
        "reading": ("NOT separated => the deployable score is CONTEXT-BLIND: it "
                    "ranks candidates by trajectory shape alone and calling it "
                    "'the consequence of flying it' is not supported"),
        "can_fire": True,
    }
    ctrl["C-cv"] = {
        "rho_cv": R["cv"]["mean"],
        "paired_deploy_minus_cv": pair["deploy_minus_cv"],
        "reading": ("NOT separated in favour of deploy => a ZERO-parameter, "
                    "zero-compute heuristic already carries whatever S3 "
                    "carries, and S3's parameter + per-candidate law_head "
                    "evaluation are unjustified"),
        "can_fire": True,
    }
    ctrl["C-oracle-reproduce"] = {
        "rho_oracle_here": R["oracle"]["mean"],
        "esel_published": ESEL_RHO.get(arm),
        "inside_esel_ci": bool(
            arm in ESEL_RHO
            and ESEL_RHO[arm]["lo"] <= R["oracle"]["mean"] <= ESEL_RHO[arm]["hi"]),
        "reading": ("outside the CI => this pipeline is not the one that "
                    "produced the upper bound and the gap is not attributable"),
        "can_fire": True,
    }
    ctrl["C-shuffled"] = {
        "rho_shuffled": R["shuffled"]["mean"],
        "⛔_vacuous": res["rho"]["shuffled"]["⛔_vacuous"], "can_fire": False,
    }
    res["controls"] = ctrl

    # -------------------------------------------------------------- verdict
    rho = float(R["deploy"]["mean"])
    sep_shuf = bool(pair["deploy_minus_shuffled"]["separated"])
    sep_ctx = bool(pair["deploy_minus_ctxswap"]["separated"])
    sep_cv = bool(pair["deploy_minus_cv"]["separated"])
    fav_ctx = sep_ctx and pair["deploy_minus_ctxswap"]["delta"] > 0
    fav_cv = sep_cv and pair["deploy_minus_cv"]["delta"] > 0
    if (not sep_shuf) or abs(rho) < S3_THRESHOLDS["rho_min"]:
        verdict = "D-NULL"
    elif sep_ctx and sep_cv:
        verdict = "D-FUND"
    else:
        verdict = "D-MARGINAL"
    red = bool(pair["deploy_minus_oracle"]["separated"]
               and pair["deploy_minus_oracle"]["delta"] > 0)
    res["verdict"] = {
        "branch": verdict,
        "rho_deploy": rho,
        "rho_oracle_upper_bound": float(R["oracle"]["mean"]),
        "gap_upper_bound_minus_deployable": round(
            float(R["oracle"]["mean"]) - rho, 4),
        "separated_from_shuffled": sep_shuf,
        "separated_from_ctxswap": sep_ctx,
        "separated_from_ctxswap_FAVOURABLY": fav_ctx,
        "separated_from_cv": sep_cv,
        "separated_from_cv_FAVOURABLY": fav_cv,
        "RED_FLAG_deploy_beats_oracle": red,
        "thresholds": S3_THRESHOLDS,
        "⚠️_branch_order": (
            "D-NULL is evaluated FIRST because it is the conservative branch; "
            "its two disjuncts can in principle co-occur with D-FUND's "
            "conjuncts (rho >= 0.10 with sep_ctx/sep_cv but NOT sep_shuf) and "
            "the registered text does not order them. Recorded so the ordering "
            "is visible rather than silently decisive."),
        "⚠️_direction": (
            "`separated` is the registered predicate and is what the branch "
            "uses. `separated_..._FAVOURABLY` additionally requires the delta "
            "to point the right way — reported because E-SEL-0 hit exactly the "
            "unregistered case of SEPARATED IN THE ADVERSE DIRECTION."),
    }

    # ---------------------------------------------------------- P2: sizing
    z = zscore_rows(d["cons_deploy"])
    sweep = alpha_sweep(de_all, logits, z, eid, S3_THRESHOLDS["alphas"])
    best_key = min(sweep, key=lambda k: sweep[k]["mean_ade"])
    idx_lo, ade_lo, chosen = loeo_alpha(de_all, logits, z, eid,
                                        S3_THRESHOLDS["alphas"])
    idx_ship = logits.argmax(1)
    ade_ship = shipped["_per_window_ade"]
    idx_dep = d["cons_deploy"].argmax(1)
    dep_only = P.ranker_block(de_all, de_or, idx_dep, eid, tag="deploy_argmax")

    p_loeo = P._paired(ade_lo.numpy(), ade_ship.numpy(), eid)
    p_best = P._paired(sweep[best_key]["_ade"].numpy(), ade_ship.numpy(), eid)
    p_dep = P._paired(dep_only["_per_window_ade"].numpy(), ade_ship.numpy(), eid)
    sel_gap = float(shipped["sel_gap"]["mean"])
    pays = bool(p_loeo["separated"] and p_loeo["delta"] <= -S3_THRESHOLDS["free_win_m"])
    res["sizing"] = {
        "shipped_ade_0_2s": shipped["ade_0_2s"],
        "oracle_in_fan_ade_0_2s": P._boot(de_or.numpy(), eid),
        "selection_gap_m": sel_gap,
        "alpha_grid": S3_THRESHOLDS["alphas"],
        "alpha_sweep_mean_ade": {k: v["mean_ade"] for k, v in sweep.items()},
        "alpha_star_on_test": {
            "alpha": sweep[best_key]["alpha"],
            "mean_ade": sweep[best_key]["mean_ade"],
            "paired_vs_shipped": p_best,
            "⚠️": ("OPTIMISTIC BY CONSTRUCTION — alpha chosen on the same 881 "
                   "windows it is scored on. An UPPER BOUND, never the headline."),
        },
        "loeo": {
            # str keys: `_clean` walks dicts and calls `k.startswith`
            "alpha_per_held_out_episode": {str(k): v for k, v in chosen.items()},
            "n_distinct_alphas_chosen": len(set(chosen.values())),
            "mean_ade": round(float(ade_lo.mean()), 6),
            "paired_vs_shipped": p_loeo,
            "frac_of_selection_gap_closed": round(
                float(-p_loeo["delta"] / sel_gap), 4) if sel_gap else None,
            "note": ("alpha chosen on 39 episodes, applied to the held-out one, "
                     "all 40 folds. THE HONEST REALIZED NUMBER."),
        },
        "deploy_argmax_only": {
            **{k: v for k, v in dep_only.items() if not k.startswith("_")},
            "paired_vs_shipped": p_dep,
            "note": "alpha -> infinity: rank by the consequence score ALONE",
        },
        "verdict": "S3 PAYS" if pays else "S3 LOWER-BOUND ONLY",
        "⚠️_lower_bound": (
            "`cons_gate` is ZERO-INIT and UNTRAINED in these weights, and "
            "feat_proj/conf_head do receive gradient in S3 — so a realized null "
            "here BOUNDS how much of S3 is free, exactly as E-SEL-0 bounded S1. "
            "It is NOT an effect size for a trained S3 and is not upgraded to one."),
        "⚠️_unclamped": (
            "the real seam can only SHRINK a graft, so this sweep is an UPPER "
            "BOUND on what a learned scalar cons_gate can express."),
    }

    # ------------------------------------------------------------------- #
    # DIAGNOSTIC — is rho even a proxy for SELECTION quality?               #
    # ------------------------------------------------------------------- #
    # ⚠️ POST-HOC. Not a registered branch and it moves no threshold. It exists
    # because the registered statistic (rho over the WHOLE candidate axis) and
    # the registered decision (fund a SELECTOR) are not obviously the same
    # quantity, and P2 asks to convert rho into an ADE effect. The honest
    # conversion is to MEASURE the selector each score produces, not to model it.
    SCORES = {"deploy": d["cons_deploy"], "oracle": d["cons_oracle"],
              "ctxswap": d["cons_ctxswap"], "shipped": logits,
              "cv": -torch.linalg.norm(fan - d["cv"][:, None], dim=-1).mean(-1)}
    diag: dict = {"argmax_selector": {}, "graft_on_shipped": {}}
    for nm, sc in SCORES.items():
        rb = P.ranker_block(de_all, de_or, sc.argmax(1), eid, tag=f"argmax_{nm}")
        diag["argmax_selector"][nm] = {
            **{k: v for k, v in rb.items() if not k.startswith("_")},
            "paired_vs_shipped": P._paired(rb["_per_window_ade"].numpy(),
                                           ade_ship.numpy(), eid),
            "rho_full_axis": float(R[nm]["mean"]),
        }
        if nm == "shipped":
            # grafting the incumbent onto itself is not a lever; the argmax row
            # above is the identity check and IS meaningful.
            diag["graft_on_shipped"][nm] = "N/A — that is the base, not a graft"
            continue
        zz = zscore_rows(sc)
        _, a_lo, ch = loeo_alpha(de_all, logits, zz, eid, S3_THRESHOLDS["alphas"])
        sw = alpha_sweep(de_all, logits, zz, eid, S3_THRESHOLDS["alphas"])
        bk = min(sw, key=lambda k: sw[k]["mean_ade"])
        diag["graft_on_shipped"][nm] = {
            "loeo_mean_ade": round(float(a_lo.mean()), 6),
            "loeo_paired_vs_shipped": P._paired(a_lo.numpy(), ade_ship.numpy(), eid),
            "alpha_star_on_test": sw[bk]["alpha"],
            "alpha_star_mean_ade": sw[bk]["mean_ade"],
            "n_distinct_alphas_loeo": len(set(ch.values())),
        }
    try:
        from tanitad.refs import refc_select as sl                # noqa: PLC0415
        rmask = sl.reachability_mask(fan, d["v0"].to(fan.dtype),
                                     accel_max=2.5, horizon_s=2.0)
        rmask = (rmask | ~rmask.any(1)[:, None]).numpy()
        reach = {}
        for nm, sc in SCORES.items():
            per, used = rho_on_reachable(sc.numpy().astype(np.float64), neg, rmask)
            ok = ~np.isnan(per)
            reach[nm] = {
                **P._boot(per[ok], np.asarray(eid)[ok]),
                "n_windows_with_>=3_survivors": used,
                "n_windows_dropped": int((~ok).sum()),
                "rho_full_axis": float(R[nm]["mean"]),
            }
        reach["_frac_candidates_removed"] = round(
            float(1 - rmask.mean()), 4)
        diag["rho_on_S2_reachable_only"] = reach
    except Exception as exc:                                      # pragma: no cover
        diag["rho_on_S2_reachable_only"] = {
            "status": f"UNAVAILABLE — {type(exc).__name__}: {exc}"}
    diag["reachability"] = P.reachability_block(d, de_all, de_or, eid)
    diag["⭐_reading"] = (
        "a per-window Spearman over the FULL candidate axis is dominated by the "
        "72-74 % of the fan that is outside the reachable band and is MEASURED "
        "never selected. If a score with a high rho produces a BAD argmax, then "
        "rho — including E-SEL-1's 0.6657 upper bound — is not a proxy for "
        "selection quality and must not be used to size a SELECTOR.")
    res["diagnostic_rho_vs_selection"] = diag

    # ------------------------------------------------- P3: four families
    fam = P.family_paired(d, idx_lo, idx_ship, eid,
                          tag="LOEO-reranked minus shipped")
    fam_oracle = P.family_paired(d, de_all.argmin(1), idx_ship, eid,
                                 tag="oracle-in-fan minus shipped (the CEILING)")
    res["families"] = {
        "paired_loeo_rerank_minus_shipped": fam,
        "paired_oracle_minus_shipped_CEILING": fam_oracle,
        "absolute_shipped": P.families_block(d, idx_ship, eid,
                                             sel_half=shipped, tag="shipped"),
        "absolute_loeo": P.families_block(d, idx_lo, eid, tag="loeo_rerank"),
        "UNAVAILABLE": {
            "LONGITUDINAL_distance_keeping": {
                "status": "UNAVAILABLE", "n": 0,
                "reason": ("headway / time-gap / TTC need a LEAD-AGENT track. A "
                           "fan bank carries none. The reader exists "
                           "(`…/2026-08-03-longitudinal-distance-keeping/"
                           "build_lead_tracks.py` over `obstacle.offline`) and "
                           "joining it to the val windows is a WORK ITEM, not a "
                           "pass.")},
            "TACTICAL_manoeuvre_decision": {
                "status": "UNAVAILABLE", "n": 0,
                "reason": ("selected-vs-executed manoeuvre and the 5-way "
                           "confusion need decoded manoeuvre logits; the fan "
                           "bank stores none. The GOAL/ANCHOR-SELECTION half IS "
                           "measured — see sizing.* rank_acc / sel_gap / "
                           "frac_sel_2x_worse. WORK ITEM.")},
            "STRATEGIC": {
                "status": "UNAVAILABLE", "n": 0,
                "reason": ("no route/goal label in a fan bank, AND the decode "
                           "used nav_mode='follow_constant' so the route input "
                           "was never exercised — the 07-21 C6 confound, "
                           "inherited deliberately so the paired baseline does "
                           "not move. WORK ITEM (the S5 arm needs it).")},
        },
        "⛔": "per family, never pooled. An ADE sweep is ONE ROW OF FOUR.",
    }
    res["rankers"] = {"shipped": {k: v for k, v in shipped.items()
                                  if not k.startswith("_")}}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"s3_deploy_probe_{arm}.json"
    f.write_text(json.dumps(P._clean(res), indent=2, ensure_ascii=False),
                 encoding="utf-8")
    print(f"[s3] {arm}: rho_deploy {rho:.4f} "
          f"[{R['deploy']['lo']}, {R['deploy']['hi']}] vs UPPER BOUND "
          f"rho_oracle {R['oracle']['mean']:.4f} :: {verdict} :: "
          f"sizing {res['sizing']['verdict']} -> {f}", flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run(a.bank, a.arm, a.out)
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
