#!/usr/bin/env python3
"""E-OBJ-1 — is ``refc_train.loss_rcls`` itself the liability, and WHICH HALF of it?

Pre-registration: ``TanitAD Research Hub/Architecture & Inference/Implementation/incoming/
2026-08-04-loss-rcls-objective/PREREG_LOSS_RCLS_OBJECTIVE.md``, staged and content-pinned
**before** the first statistic. The runner re-reads it and refuses to call a result
admissible if the staged blob and the worktree blob disagree.

⭐ WHY THIS EXISTS. E-S1-0 fit every candidate ranker with a **listwise softmax CE against the
one-hot oracle index** — which is exactly what ``loss_rcls`` optimises
(``refc_train.py:408-415``) — and *every* fitted ranker came back separated WORSE than the
incumbent selector, including feature sets containing the incumbent's own score, with a C-leak
gap of −0.001 to −0.003 m (i.e. **not** overfitting). A POST-HOC swap to expected-ADE recovered
−0.0974 m (base) / −0.1670 m (XL). That is a claim about an objective **S1, S3, S5 and S6 all
depend on**, made by a probe that adjudicated no branch. This module re-runs it as a
pre-registered experiment, and adds the half nothing in the programme has separated:

  * **O-ce**      ``CE(s, argmin_survivors fan_err)``            — the INCUMBENT objective
  * **O-softade** ``E_{i~softmax(s)}[fan_err_i]``                — the escalation's candidate
  * **O-softce**  ``CE(s, softmax(-fan_err/tau))``               — ⭐ keeps the CE FORM, replaces
                                                                    only the TARGET SHAPE

Q2 is which of the last two the recovery belongs to, because they are DIFFERENT one-line changes
to ``loss_rcls`` — both at **0 parameters**, both preserving the gradient path ``cons_gate`` and
``route_to_anchor`` depend on (``refc_train.py:377-387``).

⛔ WHAT THIS CANNOT DECIDE, registered in advance (prereg §1): it measures an objective's effect on
**combining frozen banked scores** with 1-7 parameters, not on **learning** a score. A win is
NECESSARY-NOT-SUFFICIENT for funding a retrain; a null does NOT show a retrain would fail. And it
is structurally blind on a single feature, because ``argmax(w.phi)`` is invariant to a positive
rescaling of ``w`` — which is why the 1-D rows are used as CONTROLS, not treatments.

⛔ 0 GPU. 0 inference. No checkpoint, no cache, no pod. Reads two banked fan dumps.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import refc_s1_climbout_probe as S          # survivor mask + features, REUSED not re-implemented
import refc_sel_probe as P

PREREG = ("TanitAD Research Hub/Architecture & Inference/Implementation/incoming/"
          "2026-08-04-loss-rcls-objective/PREREG_LOSS_RCLS_OBJECTIVE.md")

#: prereg §5 — fixed here, read by :func:`adjudicate`, never recomputed from data.
THRESHOLDS = {
    "material_m": 0.02,           # the parent's free_win_m, inherited not re-invented
    "red_flag_m": 0.10,           # separated better than shipped by > this => AUDIT, do not publish
    "tau_headline": 0.25,         # the scale of the published sel_gap (0.2813 base)
    "tau_strip": [0.01, 0.05, 0.10, 0.25, 0.50, 1.00],
    "tau_branch_check": [0.10, 0.25, 0.50],   # branch must agree across these or it is tau-DEPENDENT
    "continuity_tau": 0.01,
    "continuity_tol_m": 1e-3,
    "leak_frac_voids": 0.50,
    "estimator": ("paired_episode_cluster_bootstrap, unit=episode, n_boot=2000; "
                  "overlapping_holdout_se is NEVER called"),
    "primaries": ["PRIMARY-1 B-both O-softade minus O-ce",
                  "PRIMARY-2 best arm minus SHIPPED",
                  "PRIMARY-3 B-both O-softce(0.25) vs both poles"],
    "replication_rule": ("refc-base-30k is primary; refc-xl-30k is a pre-registered "
                         "REPLICATION. A conclusion requires the SAME three-sided "
                         "branch on BOTH arms."),
    "registered_prediction": {
        "PRIMARY-1": "OBJ-LIABILITY-CONFIRMED (~80%)",
        "PRIMARY-2": "DEPLOYABLE-NULL (~85%)",
        "PRIMARY-3": "TARGET-SHAPE (~55%, genuinely uncertain)",
        "Q3": "REDUNDANT (~60%)",
        "C-lon": "NULL-REPLICATES (~85%)",
    },
}

OBJECTIVES = ("O-ce", "O-softade", "O-softce")

#: prereg §3.2. The first four are DEGENERATE (1-D) and therefore CONTROLS: every objective has
#: the identical argmax on them, so they cannot move and they must reproduce their incumbents.
FEATURE_SETS = {
    "A-shipped":    ["logits"],
    "A-refined":    ["refined_logits"],
    "A-t0":         ["emitted_t0_logits"],
    "E-cv":         ["neg_cv_dist_m"],
    "B-both":       ["logits", "refined_logits"],
    "C-lon":        ["along_end_m", "dv_mps", "abs_dv_mps", "outside_band_mps"],
    "D-lon+scores": ["logits", "refined_logits", "along_end_m", "dv_mps",
                     "abs_dv_mps", "outside_band_mps"],
    "F-t0":         ["logits", "emitted_t0_logits"],
    "G-all-t0":     ["logits", "refined_logits", "along_end_m", "dv_mps",
                     "abs_dv_mps", "outside_band_mps", "emitted_t0_logits"],
}
DEGENERATE = {"A-shipped", "A-refined", "A-t0", "E-cv"}
#: prereg §5.2 — the candidate pool PRIMARY-2 reads over. Named in advance so "the best arm"
#: cannot be chosen after seeing the numbers.
DEPLOY_POOL = [("B-both", "O-softade"), ("B-both", "O-softce"),
               ("D-lon+scores", "O-softade"), ("D-lon+scores", "O-softce"),
               ("F-t0", "O-softade"), ("F-t0", "O-softce"),
               ("G-all-t0", "O-softade"), ("G-all-t0", "O-softce")]

NEG = torch.finfo(torch.float64).min / 4


# =========================================================================== #
# provenance                                                                  #
# =========================================================================== #

def prereg_provenance() -> dict:
    staged = P._git("ls-files", "-s", "--", PREREG).split()
    worktree = P._git("hash-object", PREREG)
    blob = staged[1] if len(staged) > 1 else "<not staged>"
    return {
        "path": PREREG,
        "staged_blob": blob,
        "worktree_blob": worktree,
        "thresholds_unmoved_since_staging": bool(blob and blob == worktree),
        "how_to_verify": (f"git ls-files -s -- '{PREREG}' && git hash-object '{PREREG}'"
                          "   # the two MUST match, and must match staged_blob below"),
        "thresholds": THRESHOLDS,
        "parent_s1": S.PREREG,
        "parent_dsel": P.PREREG,
    }


# =========================================================================== #
# the three objectives — the ONLY thing that differs between arms             #
# =========================================================================== #

def _fit_ce(phi, tgt, keep, de=None, tau=None, iters=S.FIT_ITERS, lr=S.FIT_LR):
    """INCUMBENT. Listwise softmax CE vs the one-hot oracle index over survivors.

    Bit-for-bit the shape of ``refc_train.loss_rcls``:
    ``F.cross_entropy(sel_score.masked_fill(~reach_keep, min/4), fan_err.masked_fill(
    ~reach_keep, inf).argmin(1))``. Delegated to :func:`refc_s1_climbout_probe._fit` so the
    E-S1-0 rows are reproduced by the SAME code object, not by a copy that can drift.
    """
    return S._fit(phi, tgt, keep, iters=iters, lr=lr)


def _fit_softade(phi, tgt, keep, de=None, tau=None, iters=S.FIT_ITERS, lr=S.FIT_LR):
    """Expected ADE under the score's own softmax. Delegated to E-S1-0's implementation."""
    return S._fit_softade(phi, de, keep, iters=iters, lr=lr)


def soft_target(de: torch.Tensor, keep: torch.Tensor, tau: float) -> torch.Tensor:
    """``softmax(-fan_err / tau)`` over the survivors — the TARGET SHAPE half of Q2.

    As ``tau -> 0`` this converges to ``onehot(argmin fan_err)``, i.e. EXACTLY the incumbent CE
    target. That is not a remark, it is control **C-continuity**: a tau knob that does not
    reproduce the incumbent at tau = 0.01 is not the knob this module claims it is.
    """
    z = torch.where(keep, -de.double() / float(tau),
                    torch.full_like(de.double(), NEG))
    return torch.softmax(z, dim=-1)


def _fit_softce(phi, tgt, keep, de=None, tau=None, iters=S.FIT_ITERS, lr=S.FIT_LR):
    """CE FORM kept, one-hot target replaced by ``softmax(-fan_err/tau)``.

    ⭐ This is the arm nothing in the programme has run. It separates "the CE is the wrong shape
    of loss" from "one winner among ~128 near-duplicate candidates is the wrong target" — two
    different one-line changes to ``loss_rcls`` with different gradient consequences, both at 0
    parameters.
    """
    q = soft_target(de, keep, tau)
    w = torch.zeros(phi.shape[-1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        s = (phi * w).sum(-1)
        s = torch.where(keep, s, torch.full_like(s, NEG))
        logp = torch.log_softmax(s, dim=-1)
        # ⚠️ masked explicitly: a non-survivor has q = 0 and logp ~ -4.5e307, and 0 * -inf is NaN
        # the moment the mask value stops being finite. Belt and braces, not decoration.
        loss = -(torch.where(keep, q * logp, torch.zeros_like(logp))).sum(-1).mean()
        loss.backward()
        opt.step()
    return w.detach()


FITTERS = {"O-ce": _fit_ce, "O-softade": _fit_softade, "O-softce": _fit_softce}


def objective_values(phi, w, tgt, de, keep, tau) -> dict:
    """All three objectives evaluated at ONE weight vector — the C-optimiser matrix.

    ⚠️ THE CONTROL THAT CAN VOID THE HEADLINE. If the CE fit is beaten UNDER ITS OWN OBJECTIVE by
    the soft-ADE fit, then the difference in selection ADE is an OPTIMISER artifact and there is
    no objective finding in either direction.
    """
    with torch.no_grad():
        s = (phi * w).sum(-1)
        s = torch.where(keep, s, torch.full_like(s, NEG))
        d0 = de.double().masked_fill(~keep, 0.0)
        q = soft_target(de, keep, tau)
        logp = torch.log_softmax(s, dim=-1)
        return {
            "L_ce": float(torch.nn.functional.cross_entropy(s, tgt)),
            "L_softade": float((torch.softmax(s, -1) * d0).sum(-1).mean()),
            "L_softce": float(-(torch.where(keep, q * logp,
                                            torch.zeros_like(logp))).sum(-1).mean()),
        }


def loeo(feats, names, keep, de_all, eid, objective, tau, iters=S.FIT_ITERS):
    """LEAVE-ONE-EPISODE-OUT under ONE objective. Folds asserted disjoint, per fold."""
    phi = S._stack(feats, names, keep)
    tgt = de_all.double().masked_fill(~keep, float("inf")).argmin(1)
    uniq, by_ep = P.ci.episode_index(eid)
    fit = FITTERS[objective]
    out = torch.zeros(phi.shape[0], phi.shape[1], dtype=torch.float64)
    ws = []
    for e in uniq:
        te = torch.zeros(phi.shape[0], dtype=torch.bool)
        te[torch.as_tensor(by_ep[e], dtype=torch.long)] = True
        tr = ~te
        assert bool(te.any()) and bool(tr.any()), f"degenerate fold for episode {e}"
        assert not bool((te & tr).any()), "folds are NOT disjoint"
        w = fit(phi[tr], tgt[tr], keep[tr], de=de_all[tr], tau=tau, iters=iters)
        out[te] = (phi[te] * w).sum(-1)
        ws.append(w.numpy())
    w_full = fit(phi, tgt, keep, de=de_all, tau=tau, iters=iters)
    meta = {"objective": objective, "tau": (tau if objective == "O-softce" else None),
            "n_folds": len(uniq), "features": names, "iters": iters,
            "w_mean": np.round(np.mean(ws, 0), 6).tolist(),
            "w_std": np.round(np.std(ws, 0), 6).tolist(),
            "w_insample": np.round(w_full.numpy(), 6).tolist(),
            "_w_full": w_full, "_phi": phi, "_tgt": tgt}
    return out, meta


# =========================================================================== #
# the run                                                                     #
# =========================================================================== #

def _load(bank: str, t0_bank: str) -> tuple[dict, dict, dict]:
    d = P.load_fan(bank)
    if "refined_logits" not in d:
        raise KeyError(f"{bank} carries no refined_logits — needs the AUGMENTED bank")
    t = torch.load(t0_bank, map_location="cpu", weights_only=False)
    for k in ("emitted_t0_logits", "emitted_logits", "prefinal_logits"):
        if k not in t:
            raise KeyError(f"{t0_bank} carries no {k!r} — needs refc_s1_dump_emitted.py --t 0")
    ctl = {
        "what": ("the two banks must describe the SAME forward. A changed fan would silently "
                 "re-baseline every D-SEL number, and every contrast here is paired against "
                 "the published 0.4728 / 0.4714."),
        "fan_bit_identical": bool(torch.equal(d["fan"], t["fan"])),
        "gt_bit_identical": bool(torch.equal(d["gt"], t["gt"])),
        "logits_bit_identical": bool(torch.equal(d["logits"], t["logits"])),
        "v0_bit_identical": bool(torch.equal(d["v0"], t["v0"])),
        "eid_match": list(d["eid"]) == list(t["eid"]),
        "prefinal_reproduces_refined": bool(torch.equal(d["refined_logits"],
                                                        t["prefinal_logits"])),
        "shipped_sel_is_argmax_logits": float((d["logits"].argmax(1) == d["sel"]).double().mean()),
        "can_fire": True,
    }
    ctl["passes"] = bool(ctl["fan_bit_identical"] and ctl["gt_bit_identical"]
                         and ctl["logits_bit_identical"] and ctl["eid_match"]
                         and ctl["prefinal_reproduces_refined"]
                         and ctl["shipped_sel_is_argmax_logits"] == 1.0)
    return d, t, ctl


def run(bank: str, t0_bank: str, arm: str, out_dir: str, *, quick: bool = False,
        lead_blocks: str | None = None) -> dict:
    t_start = time.time()
    d, t0, c_reproduce = _load(bank, t0_bank)
    eid = list(d["eid"])
    de_all = P.candidate_ade(d["fan"], d["gt"])            # [B, N] — taniteval's own definition
    de_or = de_all.min(1).values
    keep, reach = S.survivor_mask(d)
    feats = S.build_features(d, keep)
    feats["emitted_t0_logits"] = S._zrow(t0["emitted_t0_logits"].double(), keep)
    feats["emitted_logits"] = S._zrow(t0["emitted_logits"].double(), keep)
    tau_h = THRESHOLDS["tau_headline"]

    shipped_idx = d["sel"]
    shipped = P.ranker_block(de_all, de_or, shipped_idx, eid, tag="shipped")
    sets = dict(FEATURE_SETS)
    if quick:
        sets = {k: v for k, v in sets.items() if k in ("A-shipped", "B-both")}

    # ---- the panel: every feature set x every objective, identical folds -----
    arms, idxs = {}, {}
    for fname, names in sets.items():
        for obj in OBJECTIVES:
            sc, meta = loeo(feats, names, keep, de_all, eid, obj, tau_h)
            idx = S.argmax_over_survivors(sc, keep)
            blk = P.ranker_block(de_all, de_or, idx, eid, tag=f"{fname}|{obj}")
            ins = S.argmax_over_survivors((meta["_phi"] * meta["_w_full"]).sum(-1), keep)
            ins_ade = de_all.gather(1, ins[:, None]).squeeze(1)
            key = f"{fname}|{obj}"
            idxs[key] = idx
            arms[key] = {
                "feature_set": fname, "objective": obj, "features": names,
                "degenerate_control": fname in DEGENERATE,
                "n_params": len(names),
                "loeo": {k: v for k, v in blk.items() if not k.startswith("_")},
                "paired_vs_shipped": P._paired(blk["_per_window_ade"].numpy(),
                                               shipped["_per_window_ade"].numpy(), eid),
                "C-leak_in_sample_ade": round(float(ins_ade.mean()), 6),
                "C-leak_gap_m": round(float(ins_ade.mean()
                                            - blk["_per_window_ade"].mean()), 6),
                # ⛔ SECONDARY, on the reachable subset only, and NEVER without the
                # selection ADE above it. rho over the full candidate axis is disconnected
                # from selection for every score (S3_DEPLOYABLE §3).
                "rho_reachable_SECONDARY": S.rho_reachable(sc, de_all, keep, eid),
                "weights": {k: v for k, v in meta.items() if not k.startswith("_")},
                "_meta": meta, "_blk": blk,
            }
        # paired contrasts BETWEEN objectives on the SAME features/folds/survivors
        for obj in ("O-softade", "O-softce"):
            a, b = arms[f"{fname}|{obj}"], arms[f"{fname}|O-ce"]
            a["paired_vs_same_features_under_CE"] = P._paired(
                a["_blk"]["_per_window_ade"].numpy(),
                b["_blk"]["_per_window_ade"].numpy(), eid)
        arms[f"{fname}|O-softce"]["paired_vs_same_features_under_softade"] = P._paired(
            arms[f"{fname}|O-softce"]["_blk"]["_per_window_ade"].numpy(),
            arms[f"{fname}|O-softade"]["_blk"]["_per_window_ade"].numpy(), eid)

    # ---- controls ---------------------------------------------------------
    controls = {
        "C-reproduce-banks": c_reproduce,
        "C-identity": P.control_identity(d, de_all, arm),
        "C-oracle-floor": P.control_oracle_floor(d, de_or, arm),
        "C-incumbent": {
            "what": ("EVERY treatment is scored against the SHIPPED selector, never against a "
                     "shuffled or random comparator. This is the control S3_DEPLOYABLE "
                     "registered against a shuffled comparator and thereby missed."),
            "shipped_ade": round(float(shipped["_per_window_ade"].mean()), 6),
            "published": P.PUBLISHED.get(arm, {}).get("selected_ade2s"),
            "can_fire": True,
        },
        "C-shuffled": {
            "what": "permute-then-argmax is a uniform random pick for ANY score",
            "status": "VACUOUS BY CONSTRUCTION — named, reported, NEVER load-bearing",
            "can_fire": False,
        },
    }

    # C-monotone: the DEGENERATE rows must reproduce their incumbents AND be
    # objective-invariant, because argmax(w.phi) == argmax(c.w.phi) for c > 0.
    mono = {"what": ("a 1-D feature set is a strictly monotone within-window transform, so its "
                     "argmax is identical under ALL THREE objectives and must reproduce its "
                     "incumbent exactly. If the LOEO split, the survivor mask or the argmax is "
                     "wrong, this breaks."), "can_fire": True, "rows": {}}
    incumbent_of = {
        "A-shipped": float(shipped["_per_window_ade"].mean()),
        "A-refined": float(de_all.gather(
            1, S.argmax_over_survivors(d["refined_logits"], keep)[:, None]).squeeze(1).mean()),
        "A-t0": float(de_all.gather(
            1, S.argmax_over_survivors(t0["emitted_t0_logits"], keep)[:, None]).squeeze(1).mean()),
    }
    for fname in [f for f in sets if f in DEGENERATE]:
        per = {o: float(de_all.gather(1, idxs[f"{fname}|{o}"][:, None]).squeeze(1).mean())
               for o in OBJECTIVES}
        same = all(bool(torch.equal(idxs[f"{fname}|{OBJECTIVES[0]}"], idxs[f"{fname}|{o}"]))
                   for o in OBJECTIVES)
        row = {"ade_per_objective": {k: round(v, 6) for k, v in per.items()},
               "argmax_identical_across_objectives": same}
        if fname in incumbent_of:
            row["incumbent_ade"] = round(incumbent_of[fname], 6)
            row["reproduces_incumbent"] = bool(
                max(abs(v - incumbent_of[fname]) for v in per.values()) < P.TOL_IDENTITY)
        mono["rows"][fname] = row
    mono["passes"] = bool(all(r["argmax_identical_across_objectives"] for r in mono["rows"].values())
                          and all(r.get("reproduces_incumbent", True)
                                  for r in mono["rows"].values()))
    controls["C-monotone"] = mono

    # ⭐ C-optimiser — the cross-objective loss matrix. THE control that can VOID the headline.
    opt_rows = {}
    for fname, names in sets.items():
        phi = arms[f"{fname}|O-ce"]["_meta"]["_phi"]
        tgt = arms[f"{fname}|O-ce"]["_meta"]["_tgt"]
        vals = {o: objective_values(phi, arms[f"{fname}|{o}"]["_meta"]["_w_full"],
                                    tgt, de_all, keep, tau_h) for o in OBJECTIVES}
        own = {"L_ce": "O-ce", "L_softade": "O-softade", "L_softce": "O-softce"}
        wins = {L: min(vals, key=lambda o: vals[o][L]) for L in own}
        opt_rows[fname] = {
            "matrix_in_sample": {o: {k: round(v, 6) for k, v in vals[o].items()}
                                 for o in OBJECTIVES},
            "argmin_of_each_objective": wins,
            "each_fit_wins_its_own_objective": bool(all(wins[L] == own[L] for L in own)),
            "cos_w_ce_vs_softade": round(float(_cos(
                arms[f"{fname}|O-ce"]["_meta"]["_w_full"],
                arms[f"{fname}|O-softade"]["_meta"]["_w_full"])), 6),
            "cos_w_ce_vs_softce": round(float(_cos(
                arms[f"{fname}|O-ce"]["_meta"]["_w_full"],
                arms[f"{fname}|O-softce"]["_meta"]["_w_full"])), 6),
        }
    controls["C-optimiser"] = {
        "what": ("each fit must be the BEST under ITS OWN objective, in-sample. If the CE fit is "
                 "beaten under the CE, the selection-ADE difference is an OPTIMISER artifact and "
                 "the finding is VOID — not a result in either direction."),
        "predicate": "each_fit_wins_its_own_objective == True for EVERY non-degenerate row",
        "rows": opt_rows,
        "passes": bool(all(v["each_fit_wins_its_own_objective"]
                           for k, v in opt_rows.items() if k not in DEGENERATE)),
        "can_fire": True,
    }

    # ⭐ C-scale-invariance — argmax is invariant to a positive rescale, so any objective
    # difference MUST be a DIRECTION difference and not a magnitude one.
    scale_ok = True
    for fname in sets:
        m = arms[f"{fname}|O-ce"]["_meta"]
        i1 = S.argmax_over_survivors((m["_phi"] * m["_w_full"]).sum(-1), keep)
        i2 = S.argmax_over_survivors((m["_phi"] * (100.0 * m["_w_full"])).sum(-1), keep)
        scale_ok &= bool(torch.equal(i1, i2))
    controls["C-scale-invariance"] = {
        "what": "argmax(w.phi) == argmax(100 w.phi); a linear score's SCALE cannot change selection",
        "passes": bool(scale_ok), "can_fire": True,
    }

    # C-permuted-features. TWO versions, both reported — see the note below.
    uni = de_all.masked_fill(~keep, float("nan")).nanmean(1)

    def _perm_rows(pf: dict) -> dict:
        out = {}
        for obj in OBJECTIVES:
            psc, _ = loeo(pf, FEATURE_SETS["D-lon+scores"], keep, de_all, eid, obj, tau_h)
            p_ade = de_all.gather(1, S.argmax_over_survivors(psc, keep)[:, None]).squeeze(1)
            out[obj] = {"ade": round(float(p_ade.mean()), 4),
                        "paired_vs_survivor_uniform_floor":
                            P._paired(p_ade.numpy(), uni.numpy(), eid)}
        return out

    # (a) WHOLE-ROW — the version E-S1-0 registered and ran.
    g = torch.Generator().manual_seed(S.RNG_SEED)
    perm_all = torch.stack([torch.randperm(de_all.shape[1], generator=g)
                            for _ in range(de_all.shape[0])])
    rows_whole = _perm_rows({k: torch.gather(v, 1, perm_all) for k, v in feats.items()})

    # (b) SURVIVOR-RESTRICTED — survivors permuted AMONG THEMSELVES, non-survivors left in
    #     place. ⭐ This is the version that is a control. `_zrow` fills every non-survivor
    #     entry with the CONSTANT 0, so a whole-row permutation drags ~74 % zeros INTO survivor
    #     slots and changes the tie structure the argmax resolves — an artefact of the
    #     z-scoring, not a property of the data. Same class as E-S1-0 §3.5's mask-permutation
    #     defect, one level deeper.
    g2 = torch.Generator().manual_seed(S.RNG_SEED + 1)
    perm_surv = torch.arange(de_all.shape[1]).repeat(de_all.shape[0], 1).clone()
    for i in range(de_all.shape[0]):
        s = torch.nonzero(keep[i], as_tuple=False).squeeze(1)
        perm_surv[i, s] = s[torch.randperm(s.numel(), generator=g2)]
    rows_surv = _perm_rows({k: torch.gather(v, 1, perm_surv) for k, v in feats.items()})

    controls["C-permuted-features"] = {
        "what": ("features permuted INDEPENDENTLY per window, then fit and evaluated LOEO under "
                 "each objective. Must be NOT SEPARATED from the survivor-restricted uniform "
                 "floor. If a destroyed target still scores off the floor, the split leaks or "
                 "the protocol is broken."),
        "WARNING_from_E-S1": ("this control FAILED SPURIOUSLY once by permuting the reachability "
                              "MASK along with the features, which made the honest comparator the "
                              "FULL-FAN floor (14.54) instead of the survivor floor (~2.78). The "
                              "mask is NOT permuted here."),
        "⭐_INSTRUMENT_CORRECTION": (
            "the WHOLE-ROW permutation E-S1-0 ran is NOT a valid control on z-scored features: "
            "`_zrow` fills every non-survivor with the CONSTANT 0, so permuting the whole row "
            "moves ~74 % zeros into survivor slots and changes the TIE STRUCTURE the argmax "
            "resolves. That is an artefact of the standardisation, not of the data. The "
            "SURVIVOR-RESTRICTED version destroys exactly the feature<->ADE association and "
            "nothing else, and IT is the load-bearing row. Both are printed."),
        "survivor_uniform_floor": round(float(uni.mean()), 4),
        "rows": rows_surv,
        "rows_whole_row_permutation_NOT_LOAD_BEARING": rows_whole,
        "passes": bool(all(not r["paired_vs_survivor_uniform_floor"]["separated"]
                           for r in rows_surv.values())),
        "passes_whole_row_version": bool(
            all(not r["paired_vs_survivor_uniform_floor"]["separated"]
                for r in rows_whole.values())),
        "can_fire": True,
    }

    # ---- the tau strip + C-continuity (prereg §3.1, §4) ---------------------
    strip = {}
    for tau in THRESHOLDS["tau_strip"]:
        sc, meta = loeo(feats, sets["B-both"], keep, de_all, eid, "O-softce", tau)
        idx = S.argmax_over_survivors(sc, keep)
        blk = P.ranker_block(de_all, de_or, idx, eid, tag=f"B-both|O-softce(tau={tau})")
        strip[str(tau)] = {
            "ade": blk["ade_0_2s"]["mean"],
            "paired_vs_CE": P._paired(blk["_per_window_ade"].numpy(),
                                      arms["B-both|O-ce"]["_blk"]["_per_window_ade"].numpy(), eid),
            "paired_vs_softade": P._paired(
                blk["_per_window_ade"].numpy(),
                arms["B-both|O-softade"]["_blk"]["_per_window_ade"].numpy(), eid),
            "paired_vs_shipped": P._paired(blk["_per_window_ade"].numpy(),
                                           shipped["_per_window_ade"].numpy(), eid),
            "w_insample": meta["w_insample"],
        }
    ct = str(THRESHOLDS["continuity_tau"])
    controls["C-continuity"] = {
        "what": ("softmax(-e/tau) -> onehot(argmin e) as tau -> 0, so O-softce at tau = "
                 f"{ct} must reproduce O-ce. A tau knob that does not converge to the incumbent "
                 "target is not the knob this module claims it is."),
        "ade_at_tau_0.01": strip[ct]["ade"],
        "ade_O-ce": arms["B-both|O-ce"]["loeo"]["ade_0_2s"]["mean"],
        "abs_dev_m": round(abs(strip[ct]["ade"] - arms["B-both|O-ce"]["loeo"]["ade_0_2s"]["mean"]), 6),
        "tol_m": THRESHOLDS["continuity_tol_m"],
        "passes": bool(abs(strip[ct]["ade"]
                           - arms["B-both|O-ce"]["loeo"]["ade_0_2s"]["mean"])
                       < THRESHOLDS["continuity_tol_m"]),
        "can_fire": True,
    }
    # ⚠️ POST-HOC DIAGNOSTIC, labelled as such: it MOVES NO THRESHOLD and adjudicates no branch.
    # It exists only to say WHY the registered C-continuity fired, if it did. The hypothesis it
    # tests: the fan is made of NEAR-DUPLICATES, so the ADE gaps among the top candidates are
    # FINER than tau = 0.01 m and the "one-hot limit" simply has not been reached there. If the
    # deviation collapses at tau = 1e-3 / 1e-4 the knob is verified and the registered tolerance
    # was set at the wrong tau; if it does not, the knob is not what this module claims.
    ext = {}
    for tau in (0.001, 0.0001):
        sc, _m = loeo(feats, sets["B-both"], keep, de_all, eid, "O-softce", tau)
        i2 = S.argmax_over_survivors(sc, keep)
        ext[str(tau)] = {
            "ade": round(float(de_all.gather(1, i2[:, None]).squeeze(1).mean()), 6),
            "abs_dev_from_O-ce_m": round(abs(
                float(de_all.gather(1, i2[:, None]).squeeze(1).mean())
                - arms["B-both|O-ce"]["loeo"]["ade_0_2s"]["mean"]), 6),
            "argmax_identical_to_O-ce": bool(torch.equal(i2, idxs["B-both|O-ce"])),
        }
    controls["C-continuity_POSTHOC_EXTENSION"] = {
        "status": ("POST-HOC. Moves no threshold, adjudicates no branch. Explains WHY the "
                   "registered C-continuity fired."),
        "rows": ext,
        "median_survivor_ADE_gap_to_oracle_m": round(float(
            (de_all.masked_fill(~keep, float("inf")).sort(1).values[:, 1]
             - de_all.masked_fill(~keep, float("inf")).min(1).values).median()), 6),
    }

    # ---- the four families, per family, NEVER pooled ------------------------
    fams = {}
    for key, idx in idxs.items():
        fname, obj = key.split("|")
        row = {"paired_vs_shipped": P.family_paired(d, idx, shipped_idx, eid,
                                                    tag=f"{key}-minus-shipped")}
        if obj != "O-ce":
            row["paired_vs_same_features_under_CE"] = P.family_paired(
                d, idx, idxs[f"{fname}|O-ce"], eid, tag=f"{key}-minus-CE")
        fams[key] = row
    fams["_ceiling_oracle_minus_shipped"] = P.family_paired(
        d, de_all.argmin(1), shipped_idx, eid, tag="oracle-minus-shipped")
    fams["_absolute_shipped"] = P.families_block(d, shipped_idx, eid, sel_half=shipped,
                                                 tag="shipped")
    fams["_LONGITUDINAL_distance_keeping"] = {
        "status": "see the top-level distance_keeping block",
    }

    # ⭐ LONGITUDINAL — distance-keeping. E-S1-0 could not compute this family and reported it
    # n = 0 with its reason. It IS computable here: lead_source reproduces the canonical 881
    # windows exactly, so a banked lead block is row-aligned with a banked pred dump with NO
    # re-inference. This is the family the objective swap is claimed to move.
    if lead_blocks:
        lead = load_lead(lead_blocks, d)
        dk = distance_keeping_block(d, idxs, shipped_idx, lead, eid, de_all)
    else:
        dk = {"status": "NOT COMPUTED",
              "reason": ("no --lead-blocks bundle supplied. This is a RUN, not a work item: "
                         "taniteval/lead_source.py reproduces the canonical 881 windows and a "
                         "banked lead block is row-aligned with this fan dump."),
              "n": 0}

    res = {
        "experiment": ("E-OBJ-1 — is loss_rcls itself the liability, and WHICH HALF of it?"),
        "arm": arm,
        "n_windows": int(d["fan"].shape[0]),
        "n_episodes": len(set(eid)),
        "n_anchors": int(d["logits"].shape[1]),
        "bank": {"path": d["_path"], "sha256": d["_sha256"], "ckpt": d.get("ckpt"),
                 "ckpt_step": d.get("ckpt_step"), "steps": d.get("steps"),
                 "nav_mode": d.get("nav_mode"), "host": d.get("host")},
        "t0_bank": {"path": str(t0_bank), "host": t0.get("host")},
        "published": P.PUBLISHED.get(arm, {}),
        "prereg": prereg_provenance(),
        "protocol": {
            "objectives": {
                "O-ce": "CE(s, argmin_survivors fan_err) — bit-for-bit refc_train.loss_rcls",
                "O-softade": "E_{i~softmax(s)}[fan_err_i] over survivors",
                "O-softce": f"CE(s, softmax(-fan_err/tau)), tau = {tau_h}",
            },
            "held_fixed_across_objectives": ("features, z-scoring, survivor mask, LOEO folds, "
                                             "optimiser, iters, lr, init, dtype"),
            "eval": "LEAVE-ONE-EPISODE-OUT, folds asserted disjoint per fold",
            "headline": "selection ADE@2s over the survivors, PAIRED vs the named comparator",
            "estimator": THRESHOLDS["estimator"],
            "iters": S.FIT_ITERS, "lr": S.FIT_LR, "dtype": "float64",
        },
        "reachability": reach,
        "shipped": {k: v for k, v in shipped.items() if not k.startswith("_")},
        "arms": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                 for k, v in arms.items()},
        "tau_strip": strip,
        "controls": controls,
        "families": fams,
        "distance_keeping_LONGITUDINAL": dk,
        "wall_s": round(time.time() - t_start, 1),
    }
    res["verdict"] = adjudicate(res)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"obj_probe_{arm}.json").write_text(json.dumps(P._clean(res), indent=2),
                                               encoding="utf-8")
    return res


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a @ b) / (a.norm().clamp_min(1e-12) * b.norm().clamp_min(1e-12)))


# =========================================================================== #
# LONGITUDINAL — distance-keeping. The family the lever is supposed to move.   #
# =========================================================================== #

def load_lead(path: str, d: dict) -> dict:
    """Row-align a banked ``lead_source.lead_block`` bundle to a banked fan dump.

    ⭐ ALIGNMENT IS PROVEN BY CONTENT, NOT ASSUMED. The lead block's ``speeds`` are interpolated
    from the raw **egomotion parquet** at each window's registered clip time; the fan bank's
    ``v0`` comes from the **episode cache**. They are two independent paths to the same physical
    quantity, so agreeing to ~1e-3 m/s on all 881 rows is evidence the row orders are the same.
    A window-count match alone would NOT be — two different orderings can have identical counts.
    ⇒ this is control **C-lead-alignment**, and it can fire.
    """
    L = torch.load(path, map_location="cpu", weights_only=False)
    eps = sorted(L)
    cat = lambda k: np.concatenate([np.asarray(L[e][k]) for e in eps])   # noqa: E731
    leads = np.concatenate([np.asarray(L[e]["leads"], float) for e in eps], axis=0)
    speeds = cat("speeds").astype(float)
    state = cat("state").astype(object)
    n = int(d["fan"].shape[0])
    v0 = d["v0"].double().numpy()
    dev = float(np.abs(speeds - v0).max()) if speeds.size == n else float("inf")
    per_ep_match = ([len(L[e]["speeds"]) for e in eps]
                    == [int((np.asarray([str(x) for x in d["eid"]]) == str(i)).sum())
                        for i in range(len(eps))])
    ctl = {
        "what": ("the lead block's speeds come from the raw egomotion parquet via registration; "
                 "the fan bank's v0 comes from the episode cache. Two independent paths to the "
                 "same quantity agreeing on every row is what proves the row ORDER, which a "
                 "window-count match alone does not."),
        "n_lead_rows": int(speeds.size), "n_fan_rows": n,
        "per_episode_window_counts_match": bool(per_ep_match),
        "max_abs_speed_dev_mps": round(dev, 6),
        "tol_mps": 0.01,
        "passes": bool(speeds.size == n and per_ep_match and dev < 0.01),
        "can_fire": True,
    }
    return {"leads": leads, "lead_lens": cat("lead_lens").astype(float),
            "speeds": speeds, "state": state,
            "ts_rel_s": np.asarray(L[eps[0]]["ts_rel_s"], float),
            "counts": {s: int((state == s).sum()) for s in ("LEAD", "NO_LEAD", "NO_LABEL")},
            "control": ctl, "_path": str(path)}


def _dk(pred: torch.Tensor, lead: dict, dt: float) -> dict:
    from taniteval import lead_metrics
    return lead_metrics.distance_keeping(pred.double().numpy(), lead["leads"],
                                         lead["lead_lens"], lead["speeds"], dt)


def distance_keeping_block(d, idxs, shipped_idx, lead, eid, de_all) -> dict:
    """Headway / time-gap / min-TTC, PAIRED, over the windows in state LEAD.

    ⛔ Three window states, never two. ``NO_LABEL`` is NEVER counted as free flow — that would
    manufacture empty road and flatter every arm (``lead_source`` module docstring).
    ⛔ Every paired contrast is on the **intersection** of the two arms' finite windows, and the
    asymmetry (`n_only_a` / `n_only_b`) is reported, because an arm that steers its predicted path
    out of the corridor silently DROPS its own lead and would otherwise look free.
    """
    from taniteval import four_families
    b = torch.arange(d["fan"].shape[0])
    dt, prov = four_families.infer_dt({"wp_steps": list(d["wp_steps"]), "dt_s": 0.1})
    keys = ("headway_min_m", "time_gap_min_s", "min_ttc_s")

    ref = {"shipped": _dk(d["fan"][b, shipped_idx], lead, dt),
           "GT": _dk(d["gt"], lead, dt),
           "CV": _dk(d["cv"], lead, dt),
           "ORACLE-in-fan": _dk(d["fan"][b, de_all.argmin(1)], lead, dt)}

    def _pair(a, s):
        out = {}
        for k in keys:
            va, vs = np.asarray(a[k], float), np.asarray(s[k], float)
            m = np.isfinite(va) & np.isfinite(vs)
            out[k] = {
                "n_both": int(m.sum()),
                "n_only_treatment": int((np.isfinite(va) & ~np.isfinite(vs)).sum()),
                "n_only_shipped": int((~np.isfinite(va) & np.isfinite(vs)).sum()),
            }
            if m.sum() >= 2:
                out[k]["paired"] = P._paired(va[m], vs[m],
                                             [e for e, ok in zip(eid, m) if ok])
                out[k]["mean_treatment"] = round(float(np.mean(va[m])), 4)
                out[k]["mean_shipped"] = round(float(np.mean(vs[m])), 4)
            else:
                out[k]["status"] = "NOT-COMPUTABLE"
                out[k]["reason"] = ("fewer than 2 windows where BOTH arms keep the lead inside "
                                    "the corridor of their own predicted path")
        return out

    arms = {}
    for key, idx in idxs.items():
        a = _dk(d["fan"][b, idx], lead, dt)
        arms[key] = {"n": a["n"], "n_closing": a.get("n_closing"),
                     "mean_headway_min_m": a.get("mean_headway_min_m"),
                     "mean_time_gap_min_s": a.get("mean_time_gap_min_s"),
                     "n_time_gap": a.get("n_time_gap"),
                     "mean_min_ttc_s": a.get("mean_min_ttc_s"),
                     "paired_vs_shipped": _pair(a, ref["shipped"])}
    return {
        "status": "MEASURED",
        "source": lead["_path"],
        "dt_s": dt, "dt_provenance": prov,
        "ts_rel_s": lead["ts_rel_s"].tolist(),
        "window_states": lead["counts"],
        "n_LEAD": lead["counts"]["LEAD"],
        "denominators_note": ("three states, never two. NO_LABEL is not free flow. Every mean "
                              "below is over the windows where the ARM's OWN predicted path "
                              "keeps the lead in its corridor, which is <= n_LEAD and differs "
                              "between arms — hence the paired intersection counts."),
        "C-lead-alignment": lead["control"],
        "C-lead-frame": {
            "what": ("the GT, CV and oracle-in-fan reference rows. If the lead frame or the "
                     "waypoint times were wrong, the GROUND-TRUTH path's headway would be "
                     "implausible — a control that can fire on a frame or clock error."),
            "rows": {k: {"n": v["n"], "mean_headway_min_m": v.get("mean_headway_min_m"),
                         "mean_time_gap_min_s": v.get("mean_time_gap_min_s"),
                         "n_time_gap": v.get("n_time_gap"),
                         "mean_min_ttc_s": v.get("mean_min_ttc_s"),
                         "n_closing": v.get("n_closing")} for k, v in ref.items()},
            "can_fire": True,
        },
        "arms": arms,
    }


# =========================================================================== #
# adjudication — prereg §5, applied MECHANICALLY. No row is chosen by hand.    #
# =========================================================================== #

def _sep_dir(p: dict) -> tuple[bool, bool, float]:
    """(separated, favours_treatment, delta). The DIRECTION PREDICATE §5 requires.

    A trigger that reads only "separated" was satisfied LITERALLY by S3_DEPLOYABLE while BOTH of
    its controls BEAT the score. Every branch below reads both bits.
    """
    m = p.get("delta", p.get("mean"))
    sep = bool(p.get("separated", False)) and not bool(p.get("degenerate", False))
    return sep, bool(m is not None and m < 0), (float(m) if m is not None else float("nan"))


def _three_sided(p: dict, mat: float, names=("BETTER", "MARGINAL", "NOT-SEPARATED", "ADVERSE")):
    sep, fav, m = _sep_dir(p)
    if sep and fav and m <= -mat:
        b = names[0]
    elif sep and fav:
        b = names[1]
    elif sep and not fav:
        b = names[3]
    else:
        b = names[2]
    return {"branch": b, "delta_m": round(m, 6), "separated": sep, "favours_treatment": fav}


def adjudicate(res: dict) -> dict:
    mat = THRESHOLDS["material_m"]
    A, C = res["arms"], res["controls"]
    v = {"_estimator": THRESHOLDS["estimator"],
         "_replication_rule": THRESHOLDS["replication_rule"]}

    # --- gates that can VOID everything ------------------------------------
    voids = []
    if not C["C-optimiser"]["passes"]:
        voids.append("C-optimiser: a fit is beaten under its OWN objective => OPTIMISER ARTIFACT")
    if not C["C-monotone"]["passes"]:
        voids.append("C-monotone: a degenerate row failed to reproduce its incumbent, or the "
                     "argmax was not objective-invariant => the plumbing is wrong")
    if not C["C-permuted-features"]["passes"]:
        voids.append("C-permuted-features: a destroyed target scored off the uniform floor "
                     "=> the split leaks or the protocol is broken")
    if not C["C-reproduce-banks"]["passes"]:
        voids.append("C-reproduce-banks: the two banks are not the same forward")
    if not C["C-scale-invariance"]["passes"]:
        voids.append("C-scale-invariance: a linear score's scale changed its argmax")
    if not res["prereg"]["thresholds_unmoved_since_staging"]:
        voids.append("PREREG BLOB MISMATCH: the thresholds moved after staging => INADMISSIBLE")
    v["VOID"] = {"fired": bool(voids), "reasons": voids,
                 "meaning": ("⛔ a VOID is NOT a result in either direction. Prereg §5.1."),
                 "C-continuity_passes": C["C-continuity"]["passes"]}

    # --- PRIMARY-1 (Q1) — does the objective swap survive? ------------------
    p1 = A["B-both|O-softade"]["paired_vs_same_features_under_CE"]
    t1 = _three_sided(p1, mat, ("OBJ-LIABILITY-CONFIRMED", "OBJ-MARGINAL",
                                "OBJ-NOT-SEPARATED", "OBJ-ADVERSE"))
    leak = abs(A["B-both|O-softade"]["C-leak_gap_m"])
    t1["C-leak_gap_m"] = A["B-both|O-softade"]["C-leak_gap_m"]
    t1["leak_voids_effect"] = bool(abs(t1["delta_m"]) > 0 and
                                   leak > THRESHOLDS["leak_frac_voids"] * abs(t1["delta_m"]))
    t1["_note"] = ("this arm is BASE-primary; the prereg requires the SAME branch on refc-xl-30k "
                   "before the conclusion is drawn.")
    v["PRIMARY-1_Q1_objective_swap"] = t1

    # --- PRIMARY-2 — does ANYTHING beat the SHIPPED selector? ---------------
    pool = [(f"{f}|{o}", A[f"{f}|{o}"]["paired_vs_shipped"])
            for f, o in DEPLOY_POOL if f"{f}|{o}" in A]
    best = min(pool, key=lambda kv: kv[1].get("delta", 0.0)) if pool else (None, {})
    t2 = _three_sided(best[1], mat, ("DEPLOYABLE-WIN", "DEPLOYABLE-MARGINAL",
                                     "DEPLOYABLE-NULL", "DEPLOYABLE-ADVERSE")) if pool else {}
    if pool:
        t2["best_arm"] = best[0]
        t2["pool_declared_in_advance"] = [f"{f}|{o}" for f, o in DEPLOY_POOL]
        t2["all_pool_deltas"] = {k: p.get("delta") for k, p in pool}
        if t2["separated"] and t2["favours_treatment"] and \
                t2["delta_m"] <= -THRESHOLDS["red_flag_m"]:
            t2["RED_FLAG"] = ("separated better than shipped by > 0.10 m — STOP AND AUDIT for a "
                              "protocol leak (non-disjoint split, or a feature reading gt / "
                              "z_{t+5}). Do NOT publish as a win.")
    v["PRIMARY-2_deployment"] = t2

    # --- PRIMARY-3 (Q2) — WHICH HALF of loss_rcls? --------------------------
    sc_vs_ce = A["B-both|O-softce"]["paired_vs_same_features_under_CE"]
    sc_vs_sa = A["B-both|O-softce"]["paired_vs_same_features_under_softade"]
    sa_vs_ce = p1
    b_sc_ce = _three_sided(sc_vs_ce, mat)
    b_sc_sa = _three_sided(sc_vs_sa, mat)
    b_sa_ce = _three_sided(sa_vs_ce, mat)
    if b_sc_ce["branch"] in ("BETTER", "MARGINAL") and b_sc_sa["branch"] == "NOT-SEPARATED":
        attr = "TARGET-SHAPE"
    elif b_sc_ce["branch"] == "NOT-SEPARATED" and b_sa_ce["branch"] in ("BETTER", "MARGINAL"):
        attr = "OBJECTIVE-FORM"
    elif b_sc_ce["branch"] in ("BETTER", "MARGINAL") and b_sc_sa["branch"] == "ADVERSE":
        attr = "BOTH-PARTIAL"
    elif b_sc_ce["branch"] == "NOT-SEPARATED" and b_sa_ce["branch"] == "NOT-SEPARATED":
        attr = "NEITHER"
    else:
        attr = "UNCLASSIFIED — see the three rows; the prereg's four cases do not cover this"
    # tau-dependence check, mechanical
    strip = res["tau_strip"]
    branches = {str(t): _three_sided(strip[str(t)]["paired_vs_CE"], mat)["branch"]
                for t in THRESHOLDS["tau_branch_check"] if str(t) in strip}
    tau_dep = len(set(branches.values())) > 1
    # ⚠️ The registered §5.3 table has FOUR joint cases and they are NOT exhaustive. When the
    # classifier returns UNCLASSIFIED the honest response is S3_DEPLOYABLE's: report the LITERAL
    # registered branch AND the substance, and do NOT rewrite the trigger. This field NAMES the
    # substance; it never overrides `attribution`.
    substance = None
    if attr.startswith("UNCLASSIFIED"):
        if b_sc_ce["branch"] == "ADVERSE" and b_sa_ce["branch"] in ("BETTER", "MARGINAL"):
            substance = ("OBJECTIVE-FORM-STRONG — the metric-aware objective recovers, and "
                         "softening the TARGET without metric-awareness is separated WORSE than "
                         "the incumbent one-hot CE. The registered OBJECTIVE-FORM branch required "
                         "`softce - ce` NOT-SEPARATED; the observed row is ADVERSE, which is the "
                         "same conclusion arrived at more strongly. ⛔ It is reported as "
                         "UNCLASSIFIED because that is what the registered table returns.")
        elif b_sc_ce["branch"] == "ADVERSE" and b_sa_ce["branch"] == "ADVERSE":
            substance = ("BOTH-ADVERSE — every alternative objective is worse than the incumbent "
                         "CE. The escalation's premise is refuted.")
        else:
            substance = ("the three rows do not match any registered joint case; read them "
                         "individually and pre-register the next question rather than "
                         "reclassifying this one.")
    v["PRIMARY-3_Q2_which_half"] = {
        "attribution": ("τ-DEPENDENT — claim nothing" if tau_dep else attr),
        "substance_POSTHOC_never_overrides_attribution": substance,
        "tau_dependent": bool(tau_dep),
        "branch_at_tau": branches,
        "softce_minus_ce": b_sc_ce,
        "softce_minus_softade": b_sc_sa,
        "softade_minus_ce": b_sa_ce,
        "what_it_asks_the_PI_to_change": {
            "TARGET-SHAPE": "r_star -> softmax(-fan_err/tau) inside loss_rcls. 0 params, CE form "
                            "and its gradient path preserved.",
            "OBJECTIVE-FORM": "loss_rcls -> E_{softmax(sel_score)}[fan_err]. 0 params.",
            "BOTH-PARTIAL": "prefer the target change unless the residual is material.",
            "NEITHER": "⛔ recommend NO arm.",
        }.get(attr if not tau_dep else "NEITHER", "⛔ recommend NO arm."),
    }

    # --- SECONDARY Q3 — does the token finding COMPOSE? ---------------------
    if "F-t0|O-softade" in A:
        f_vs_ship = _three_sided(A["F-t0|O-softade"]["paired_vs_shipped"], mat)
        v["SECONDARY_Q3_composition"] = {
            "F-t0|O-softade_vs_shipped": f_vs_ship,
            "F-t0|O-softade_vs_CE": _three_sided(
                A["F-t0|O-softade"]["paired_vs_same_features_under_CE"], mat),
            "G-all-t0|O-softade_vs_shipped": _three_sided(
                A["G-all-t0|O-softade"]["paired_vs_shipped"], mat),
            "A-t0_degenerate_floor_ade": A["A-t0|O-ce"]["loeo"]["ade_0_2s"]["mean"],
            "B-both|O-softade_ade": A["B-both|O-softade"]["loeo"]["ade_0_2s"]["mean"],
            "F-t0|O-softade_ade": A["F-t0|O-softade"]["loeo"]["ade_0_2s"]["mean"],
        }

    # --- SECONDARY — C-lon replication --------------------------------------
    if "C-lon|O-softade" in A:
        v["SECONDARY_C-lon_replication"] = {
            "softade_minus_ce": _three_sided(
                A["C-lon|O-softade"]["paired_vs_same_features_under_CE"], mat,
                ("NULL-FAILS-TO-REPLICATE", "NULL-FAILS-TO-REPLICATE",
                 "NULL-REPLICATES", "ADVERSE")),
            "vs_shipped": _three_sided(A["C-lon|O-softade"]["paired_vs_shipped"], mat),
        }

    v["registered_prediction"] = THRESHOLDS["registered_prediction"]
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, help="fan_refined_<arm>.pt (carries refined_logits)")
    ap.add_argument("--t0-bank", required=True, help="fan_emitted_t0_<arm>.pt")
    ap.add_argument("--arm", required=True, choices=sorted(P.PUBLISHED))
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="smoke: two feature sets only")
    ap.add_argument("--lead-blocks", default=None,
                    help="val40_lead_blocks.pt from lead_source.lead_block — LONGITUDINAL "
                         "distance-keeping (headway / time-gap / TTC), 0 re-inference")
    a = ap.parse_args(argv)
    res = run(a.bank, a.t0_bank, a.arm, a.out, quick=a.quick, lead_blocks=a.lead_blocks)
    print(json.dumps(P._clean(res["verdict"]), indent=2), flush=True)
    return 0


if __name__ == "__main__":                                        # pragma: no cover
    raise SystemExit(main())
