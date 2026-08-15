"""E-WC2 — can the tactical goal head reach σ\\* on our corpus AT ALL?

Pre-registration: `Project Steering/V6F_PLANNER_DESIGN.md` §5.2, verbatim:

    **Question:** can the *tactical goal head* reach σ* on our corpus at all?
    **Method:** fit the smallest admissible predictor of the 6 s endpoint from
    **frozen S-W latents only** (a ridge, the P1/P2 battery's method) on the 40
    val episodes, LOEO; report its 1σ endpoint error at 2 s **and** 6 s beside
    the fan oracle and the incumbent selected ADE, i.e. as the two **ratios**
    σ/ADE and σ/oracle so it composes with §3.1's curve.
    **Cost:** 0 GPU, banked latents.
    **Committed outcomes:** σ/ADE ≤ 1.7 ⇒ SEL-1 is funded and S-T launches with
    it. σ/ADE ≥ 3.0 ⇒ **SEL-1 is refused before launch**, and the work moves to
    `ANCHOR_GOAL` supervision (PH0 + `obstacle.offline`). In between ⇒
    inconclusive; run the capacity control first.

⚠️ **§5.2's "Cost: 0 GPU, banked latents" IS TRUE ABOUT COMPUTE AND MISLEADING ABOUT
READINESS — the correction, MEASURED 2026-08-16.** It was widely believed the banked
latents died with the eval pod. They did not: they are IN THE REPO at
`…/incoming/2026-08-04-lambda-findability/raw/latents_refc-{base,xl}-30k.pt`, at the
pre-registered 881 windows × 40 episodes, with `instrument_fail: []` and the
producer's `fan_bit_identical` gate green (verified by loading them on CPU). What
§5.2 omitted is the **target**: the 6 s endpoint was never dumped. Because that is
GROUND TRUTH FROM POSES, it costs no GPU either — see
`refc_dump_latents.py --backfill-endpoints`. The only real prerequisite is the
val40 pose arrays on a reachable disk.
⇒ **Root-cause class: a cost estimate that priced the expensive input and not the
cheap one.** The latents were correctly identified as costly and correctly noted as
banked; the free-to-compute target was therefore never costed, and a genuinely-0-GPU
experiment sat un-runnable for want of a one-minute step.
⚠️ **A GPU pass IS still needed for the surface §5.2 NAMES** — *"frozen S-W latents"*.
S-W latents have never been dumped; that one pass needs the GPU and a deliberate pause.

THIS FILE IS THE INSTRUMENT ONLY — it runs on CPU against a dump and launches
nothing. The dump contract is `DUMP_CONTRACT` below; the producer is
`scripts/refc_dump_latents.py` (`--endpoint-steps 20,60` on an inference pass, or
`--backfill-endpoints` to add endpoints to an already-banked dump with 0 GPU).


σ IS PER-AXIS, NOT RADIAL — the unit that composes with §3.1
============================================================
§3.1's requirement curve perturbs the goal with
``g = gt_end + rng.normal(0.0, s, size=gt_end.shape)``
(`scripts/sel_winners_curse_law.py:221`), where ``gt_end`` is ``[W, 2]``. So ``s``
is the **per-axis** standard deviation of an isotropic 2-D Gaussian, in metres —
NOT the radial RMS, which is ``√2 ×`` larger. σ\\* ≈ 0.8 m, and 0.8 / 0.4714 =
**1.70** and 0.8 / 0.1639 = **4.88** reproduce §3.1's published ratios exactly,
which pins the unit. ⛔ Reporting a radial RMS against the 1.7 threshold would
inflate σ by 1.414 and could flip FUNDED → INCONCLUSIVE on arithmetic alone.
Both forms are emitted, the headline is per-axis, and the relation is stated in
the JSON so the number cannot be re-read in the wrong unit.


THE RIDGE IS THE P1/P2 BATTERY'S, IMPORTED — NOT A SECOND ONE
=============================================================
Every estimation primitive comes from ``scripts/probe_latent_state.py``:

  * ``RidgeSVD``            (probe_latent_state.py:142-187) — closed-form ridge,
    one economy SVD of the centred design shared across targets and λ.
  * ``RIDGE_LAMBDAS``       (probe_latent_state.py:134) — (1e-2 … 1e3).
  * ``RidgeSVD.best_lambda``/``gcv`` (probe_latent_state.py:172-187) — λ chosen by
    Golub-Heath-Wahba GCV **on the TRAIN fold only**; model selection never
    touches the held-out episode.
  * ``_standardize``        (probe_latent_state.py:269-274) — z-score by **TRAIN**
    mean/sd, sd floor 1e-8 → 1.0.
  * ``episode_disjoint_folds`` (probe_latent_state.py:226-246) — the fold builder.
  * ``r2_score``            (probe_latent_state.py:212-220) — ``None`` when the
    target has no variance, never 0-filled.

``ridge_oof_predict`` here is ``ridge_probe_cv`` (probe_latent_state.py:280-306)
with the out-of-fold PREDICTIONS returned as well as the pooled R² — residuals are
what σ is computed from. ``tests/test_e_wc2_sigma_star.py`` pins its pooled R²
**equal** to ``ridge_probe_cv``'s on the same inputs, so the two cannot drift.


LOEO — LEAVE-ONE-**EPISODE**-OUT, AND WHY IT IS NOT LEAVE-ONE-WINDOW-OUT
=======================================================================
``loeo_folds`` = ``episode_disjoint_folds(eid, n_folds=n_unique_episodes)``, then
ASSERTS one episode per fold. Window-disjoint folds are the REF-A I-JEPA defect
(~80 % of val inside train made its number unusable): adjacent windows on a
stride-8 grid are near-duplicates, and a fold split that puts a window's neighbour
in train reports a σ that is a memorisation artefact. A test builds an
episode-level nuisance feature and shows the leaky scheme reports a **smaller** σ
than LOEO on the same data — the leak is a downward bias on exactly the number
this instrument exists to produce.


REFUSAL, NOT A WEAK VERDICT
===========================
The verdict is emitted ONLY when the pre-registered surface is met:
40 episodes, the canonical 881-window grid, LOEO, and a 6 s endpoint present
(§5.2 requires σ at 2 s **AND** 6 s). Anything short — including *relaxing the
guards on the command line* — yields ``verdict: "NO_VERDICT"`` with the reasons
enumerated. Everything measurable is still written, so a short run is inspectable
rather than invisible. ``NO_VERDICT`` is a distinct token from ``REFUSED``, which
means "SEL-1 is refused"; the two must never be confused.


THE §5.3 REDERIVE CHECK
=======================
§5.3, verbatim: *"a σ\\* re-measured at 6 s exceeds 3× the 2 s value ⇒ the ratio
form does not transfer; the threshold must be re-derived, not scaled."* Implemented
as an explicit flag. When it fires, ``threshold_6s`` is emitted as ``null`` with the
reason — this instrument will not print a scaled 6 s threshold under any branch.
⚠️ The 3× comparison is made on **MATCHED WINDOWS**: the 6 s endpoint does not exist
for windows within 6 s of an episode's end, so σ(2 s) is re-fit on exactly the
6 s-valid subset for the comparison. Comparing a full-grid σ(2 s) against a
truncated-grid σ(6 s) would be the "never compare across different windows" defect.

EVIDENCE / TIER STAMPS (they travel in the JSON, not only here)
  * evidence class: MEASURED (ours) once run on a real dump — re-analysis of a
    banked latent dump, no model, no GPU, no re-inference at analysis time.
  * class: EXPLORATORY when the surface is a REF-C fan (a different arm at a 2 s
    horizon); the RATIOS are the transferable claim, the absolute metres are that
    fan's.
  * tier: **T0-DIAGNOSTIC**. This is a representation-capacity probe on banked
    latents. It is NOT a driving-performance number and no T1 claim may cite it.
  * interval: episode-cluster bootstrap (`taniteval/ci.py`), reducer named in the
    JSON. ``overlapping_holdout_se`` is used nowhere.

USAGE (CPU, ~1 minute)
    python stack/scripts/e_wc2_sigma_star.py --print-contract
    python stack/scripts/e_wc2_sigma_star.py --dump <latents.pt> --validate-only
    python stack/scripts/e_wc2_sigma_star.py --dump <latents.pt> \\
        --features pooled,ctx --out ewc2_sigma_star.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_latent_state import (RIDGE_LAMBDAS, RidgeSVD,  # noqa: E402
                                _standardize, episode_disjoint_folds, r2_score)

DT = 0.1                                  # 10 Hz — the corpus contract
ENDPOINT = -1                             # sel_winners_curse_law.py:65, verbatim

# --------------------------------------------------------------------------- #
# THE PRE-REGISTRATION, as data. Every one of these is quoted from §5.2/§5.3.  #
# --------------------------------------------------------------------------- #
PREREG = {
    "source": "Project Steering/V6F_PLANNER_DESIGN.md §5.2 (E-WC2) + §5.3",
    "min_episodes": 40,          # "on the 40 val episodes, LOEO"
    "min_windows": 881,          # the canonical val40 eval grid (§3.1's surface)
    "fold_scheme": "LOEO (leave-one-EPISODE-out)",
    "verdict_horizon_s": 2.0,    # where §3.1's fan references exist
    "transfer_horizon_s": 6.0,   # "report its 1σ endpoint error at 2 s AND 6 s"
    "sigma_form": "per-axis isotropic 1-sigma, metres "
                  "(sel_winners_curse_law.py:221 — N(0, s) per axis)",
    "fund_at_or_below": 1.7,     # "σ/ADE ≤ 1.7 ⇒ SEL-1 is funded"
    "refuse_at_or_above": 3.0,   # "σ/ADE ≥ 3.0 ⇒ SEL-1 is refused before launch"
    "rederive_multiple": 3.0,    # §5.3 "exceeds 3× the 2 s value"
    "reference_sigma_star_m": 0.8,        # §3.1 B, measured on the REF-C-XL fan
    "reference_ratio_vs_ade": 1.7,        # 0.8 / 0.4714
    "reference_ratio_vs_oracle": 4.9,     # 0.8 / 0.1639
}

VERDICTS = {
    "FUNDED": "σ/ADE ≤ 1.7 — SEL-1 is funded and S-T launches with it (§5.2)",
    "REFUSED": "σ/ADE ≥ 3.0 — SEL-1 is REFUSED before launch; work moves to "
               "ANCHOR_GOAL supervision (PH0 + obstacle.offline) (§5.2)",
    "INCONCLUSIVE": "1.7 < σ/ADE < 3.0 — run the capacity control first (§5.2)",
    "NO_VERDICT": "the pre-registered surface was NOT met — this instrument "
                  "refuses to emit a weak verdict. NOT the same as REFUSED.",
}

# --------------------------------------------------------------------------- #
# ADMISSIBILITY — the binding PI rule, enforced, not documented                #
# --------------------------------------------------------------------------- #
# "LABELS MAY USE EGO; INFERENCE IS VISION-ONLY" (Sayed, 2026-08-03) and "a goal
# input must not carry the situation classifier's output". A goal head that reads
# the ego/nav embedding at inference is NOT deployable, and its σ would be a leak
# magnitude rather than a capability. Blocks are classified, not assumed.
FEATURE_ADMISSIBILITY = {
    # VISION ONLY — refc_dump_latents.py:25-27 states this verbatim for all three
    "pooled": "VISION_ONLY",
    "pooled_seq": "VISION_ONLY",
    "ctx": "VISION_ONLY",
    # ⛔ THE ECHO PATH — refc_dump_latents.py:28 labels `measurement` as the
    # ego+nav embedding "and is the v0-echo path by construction".
    "measurement": "ECHO",
    "v0": "ECHO",
}

DUMP_CONTRACT = {
    "_what": "the latent dump E-WC2 consumes. One torch.save'd dict.",
    "_producer": "stack/scripts/refc_dump_latents.py --endpoint-steps 20,60",
    "_grid": "the canonical val40 eval grid: 40 episodes, WINDOW=8, STRIDE=8, "
             "881 windows. Re-selecting episodes breaks parity and is refused.",
    "required": {
        "eid": "list[int] length n — episode id PER WINDOW. LOEO folds are "
               "built from this and from nothing else.",
        "<feature block>": "torch.Tensor [n, F] (or [n, W, F], flattened) — at "
                           "least one VISION-ONLY block; `pooled` and `ctx` are "
                           "the defaults. See FEATURE_ADMISSIBILITY.",
        "gt_endpoint": "torch.Tensor [n, He, 2] — GT ego-frame displacement to "
                       "each endpoint horizon, axis0 = LONGITUDINAL (forward), "
                       "axis1 = LATERAL (left). driving_diagnostic.gt_ego_"
                       "waypoints(poses, last, wp_steps=endpoint_steps).",
        "endpoint_steps": "list[int] — the horizons of `gt_endpoint`, in 10 Hz "
                          "steps. MUST contain 20 (2.0 s) and 60 (6.0 s).",
        "endpoint_valid": "torch.Tensor [n, He] bool — False where the horizon "
                          "runs past the end of the episode. ⛔ NEVER impute; "
                          "invalid rows are EXCLUDED with n reported.",
    },
    "required_for_the_ratios": {
        "fan": "torch.Tensor [n, C, T, 2] — the candidate fan, ego frame.",
        "gt": "torch.Tensor [n, T, 2] — GT at the FAN's waypoint grid.",
        "sel": "torch.Tensor [n] int — the incumbent selector's chosen index. "
               "`sel_ade` (the ratio denominator) is computed from it.",
        "wp_steps": "list[int] — the fan's waypoint grid (e.g. [5,10,15,20]).",
    },
    "optional": {
        "cv": "torch.Tensor [n, T, 2] — the constant-velocity baseline; adds the "
              "deployable 0-param reference row from §3.1 B.",
        "ckpt": "str — provenance, stamped into the output.",
        "ckpt_step": "int — provenance.",
        "controls_vs_bank": "dict — refc_dump_latents.py's bit-identity gate. "
                            "Carried through so the surface is auditable.",
        "instrument_fail": "list[str] — a NON-EMPTY list makes E-WC2 refuse.",
    },
    "⚠️ the K_MAX conflict, stated": (
        "the producer's window grid is pinned at K_MAX = max(WP_STEPS) = 20 so "
        "the 881-window grid and the fan bit-identity gate are PRESERVED. A 6 s "
        "endpoint therefore does not exist for the last ~5 windows of every "
        "episode. Those rows are masked False in `endpoint_valid`, never "
        "dropped from the grid and never imputed — widening K_MAX to 60 would "
        "silently re-select windows and break parity."),
}


# --------------------------------------------------------------------------- #
# folds                                                                        #
# --------------------------------------------------------------------------- #
def loeo_folds(eid) -> np.ndarray:
    """Leave-one-EPISODE-out fold ids: one fold per unique episode.

    Built by ``episode_disjoint_folds`` (probe_latent_state.py:226) with
    ``n_folds = n_unique_episodes``, then ASSERTED to hold exactly one episode
    per fold. The assertion is the point: a fold builder that silently merged two
    episodes would turn LOEO into 39-fold CV and no output would say so.
    """
    uids = np.asarray([int(x) for x in eid])
    uniq = np.unique(uids)
    if uniq.size < 2:
        raise ValueError(f"LOEO needs >= 2 episodes, got {uniq.size}")
    folds = episode_disjoint_folds(uids, n_folds=int(uniq.size))
    for f in np.unique(folds):
        eps_in_fold = np.unique(uids[folds == f])
        if eps_in_fold.size != 1:
            raise AssertionError(
                f"fold {f} holds {eps_in_fold.size} episodes {eps_in_fold[:5]} "
                f"— that is not LOEO")
    if np.unique(folds).size != uniq.size:
        raise AssertionError(f"{np.unique(folds).size} folds for {uniq.size} "
                             f"episodes — that is not LOEO")
    return folds


def window_random_folds(n: int, n_folds: int, seed: int = 0) -> np.ndarray:
    """⛔ THE LEAKY SCHEME — window-disjoint, NOT episode-disjoint.

    Present ONLY so the test suite can measure how much σ the REF-A I-JEPA defect
    would have bought us. It is never used by :func:`run`.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_folds, size=n).astype(np.int64)


# --------------------------------------------------------------------------- #
# the ridge — the P1/P2 path, with the OOF predictions returned                 #
# --------------------------------------------------------------------------- #
def ridge_oof_predict(X: np.ndarray, y: np.ndarray, fold_ids: np.ndarray,
                      lambdas: tuple[float, ...] = RIDGE_LAMBDAS) -> dict:
    """``ridge_probe_cv`` (probe_latent_state.py:280) returning ``yhat`` as well.

    Per fold: standardize by TRAIN stats, pick λ by GCV on the TRAIN fold,
    closed-form fit, predict the held-out fold. Headline R² is over the POOLED
    out-of-fold predictions — identical construction, pinned equal by
    ``tests/test_e_wc2_sigma_star.py::test_ridge_oof_matches_probe_battery``.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    fold_ids = np.asarray(fold_ids)
    yhat = np.full(y.shape[0], np.nan)
    per_fold, lam_by_fold = [], []
    for f in sorted(set(fold_ids.tolist())):
        te = fold_ids == f
        tr = ~te
        if not te.any() or not tr.any():
            continue
        Xtr, Xte = _standardize(X[tr], X[te])
        solver = RidgeSVD(Xtr)
        lam = solver.best_lambda(y[tr], lambdas)
        w, b = solver.fit(y[tr], lam)
        yhat[te] = Xte @ w + b
        per_fold.append(r2_score(y[te], yhat[te]))
        lam_by_fold.append(lam)
    ok = np.isfinite(yhat)
    return {"yhat": yhat, "ok": ok,
            "r2": r2_score(y[ok], yhat[ok]) if ok.any() else None,
            "per_fold_r2": per_fold, "lambda_by_fold": lam_by_fold,
            "n": int(ok.sum())}


# --------------------------------------------------------------------------- #
# σ                                                                            #
# --------------------------------------------------------------------------- #
def sigma_from_residuals(res: np.ndarray) -> dict:
    """1σ endpoint error from the OOF residual ``res`` [n, 2] (metres, ego frame).

    ``sigma_perax`` is THE headline and the only form the §5.2 thresholds accept:
    the per-axis SD of the isotropic 2-D error, i.e. ``sqrt(mean(|e|²)/2)`` — the
    same ``s`` that §3.1 injects as ``N(0, s)`` per axis. ``sigma_radial_rms`` is
    ``√2 ×`` that and is reported so the two can never be silently swapped.
    ``sigma_long``/``sigma_lat`` are the four-families decomposition (axis 0 is
    forward, axis 1 is left — ``driving_diagnostic._ego``).
    """
    res = np.asarray(res, dtype=np.float64)
    if res.ndim != 2 or res.shape[1] != 2:
        raise ValueError(f"residuals must be [n, 2], got {res.shape}")
    sq = (res ** 2).sum(axis=1)                       # |e|² per window
    radial = np.sqrt(sq)
    return {
        "sigma_perax_m": float(np.sqrt(sq.mean() / 2.0)),
        "sigma_radial_rms_m": float(np.sqrt(sq.mean())),
        "sigma_long_m": float(np.sqrt((res[:, 0] ** 2).mean())),
        "sigma_lat_m": float(np.sqrt((res[:, 1] ** 2).mean())),
        "radial_p50_m": float(np.median(radial)),
        "radial_p90_m": float(np.quantile(radial, 0.90)),
        "n": int(res.shape[0]),
        "_unit_note": "sigma_perax_m is the PER-AXIS isotropic 1-sigma in "
                      "metres — the unit §3.1's requirement curve is stated in "
                      "(sel_winners_curse_law.py:221). sigma_radial_rms_m = "
                      "sqrt(2) x sigma_perax_m; the §5.2 thresholds are NOT "
                      "defined against it.",
    }


def fit_sigma(X: np.ndarray, target: np.ndarray, eid, *,
              lambdas: tuple[float, ...] = RIDGE_LAMBDAS) -> dict:
    """LOEO ridge on both endpoint axes; σ from the pooled OOF residuals."""
    folds = loeo_folds(eid)
    axes, res = {}, np.empty_like(np.asarray(target, dtype=np.float64))
    for j, name in enumerate(("long", "lat")):
        fit = ridge_oof_predict(X, np.asarray(target)[:, j], folds, lambdas)
        if not bool(fit["ok"].all()):
            raise AssertionError("LOEO left windows unpredicted — a fold had no "
                                 "train or no test rows")
        res[:, j] = np.asarray(target)[:, j] - fit["yhat"]
        axes[name] = {"r2_oof": fit["r2"],
                      "lambda_by_fold": [float(x) for x in fit["lambda_by_fold"]],
                      "per_fold_r2": [None if v is None else float(v)
                                      for v in fit["per_fold_r2"]]}
    out = sigma_from_residuals(res)
    out["axes"] = axes
    out["n_folds"] = int(np.unique(folds).size)
    out["fold_scheme"] = PREREG["fold_scheme"]
    out["_residual_sq_per_window"] = (res ** 2).sum(axis=1)   # for the bootstrap
    return out


# --------------------------------------------------------------------------- #
# the §3.1 reference points — computed with sel_winners_curse_law's own algebra #
# --------------------------------------------------------------------------- #
def per_candidate_err(fan: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """[W, C, T, 2] vs [W, T, 2] -> [W, C] mean displacement error.

    Byte-identical to ``sel_winners_curse_law.per_candidate_err`` (:80-82) so the
    ratio denominators are the SAME numbers §3.1 published, not a re-derivation.
    """
    return np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1)


def fan_references(d, mask: np.ndarray | None = None) -> dict:
    """oracle / fan-mean / incumbent-selected ADE from the dump's banked fan."""
    if "fan" not in d or "gt" not in d:
        return {"available": False,
                "reason": "the dump carries no `fan`/`gt`; the ratios σ/ADE and "
                          "σ/oracle are NOT computable — reported n/a, never "
                          "back-filled from a doc"}
    fan = np.asarray(d["fan"], dtype=np.float64)
    gt = np.asarray(d["gt"], dtype=np.float64)
    if mask is not None:
        fan, gt = fan[mask], gt[mask]
    err = per_candidate_err(fan, gt)
    w = err.shape[0]
    out = {"available": True,
           "n_windows": int(w), "n_candidates": int(err.shape[1]),
           "wp_steps": [int(x) for x in d.get("wp_steps", [])],
           "oracle_ade": float(err.min(1).mean()),
           "fan_mean_ade": float(err.mean()),
           "_per_window_oracle": err.min(1)}
    if "sel" in d:
        sel = np.asarray(d["sel"]).astype(int)
        if mask is not None:
            sel = sel[mask]
        chosen = err[np.arange(w), sel]
        out["sel_ade"] = float(chosen.mean())
        out["_per_window_sel"] = chosen
    if "cv" in d:
        cv = np.asarray(d["cv"], dtype=np.float64)
        if mask is not None:
            cv = cv[mask]
        out["cv_ade"] = float(np.linalg.norm(cv - gt, axis=-1).mean(-1).mean())
    out["reference_horizon_s"] = (float(max(out["wp_steps"]) * DT)
                                  if out["wp_steps"] else None)
    return out


# --------------------------------------------------------------------------- #
# dump handling                                                                #
# --------------------------------------------------------------------------- #
def validate_dump(d) -> list[str]:
    """Contract violations, as a list. Empty list = conformant."""
    problems: list[str] = []
    if "eid" not in d:
        problems.append("missing `eid` — LOEO folds cannot be built")
    if "gt_endpoint" not in d or "endpoint_steps" not in d:
        problems.append("missing `gt_endpoint`/`endpoint_steps` — §5.2 requires "
                        "the 6 s endpoint; a 2 s-only dump cannot answer E-WC2")
    else:
        steps = [int(x) for x in d["endpoint_steps"]]
        for need, label in ((20, "2.0 s"), (60, "6.0 s")):
            if need not in steps:
                problems.append(f"`endpoint_steps` {steps} has no {need} "
                                f"({label}) — §5.2 requires 2 s AND 6 s")
        ge = np.asarray(d["gt_endpoint"])
        if ge.ndim != 3 or ge.shape[1] != len(steps) or ge.shape[2] != 2:
            problems.append(f"`gt_endpoint` must be [n, {len(steps)}, 2], got "
                            f"{tuple(ge.shape)}")
        if "endpoint_valid" not in d:
            problems.append("missing `endpoint_valid` [n, He] — without it a "
                            "truncated 6 s endpoint is indistinguishable from a "
                            "real one (⛔ never imputed)")
    if not any(k in d for k in FEATURE_ADMISSIBILITY):
        problems.append("no recognised feature block; expected one of "
                        + ", ".join(sorted(FEATURE_ADMISSIBILITY)))
    if "fan" not in d or "gt" not in d or "sel" not in d:
        problems.append("missing `fan`/`gt`/`sel` — the ratio denominators "
                        "(oracle ADE, incumbent selected ADE) are not computable")
    if d.get("instrument_fail"):
        problems.append("the producer reported instrument_fail="
                        + str(list(d["instrument_fail"])[:5]))
    return problems


def resolve_endpoint(d, step: int) -> tuple[np.ndarray | None, np.ndarray | None,
                                            str]:
    """(target [n, 2], valid [n] bool, source) for a horizon in 10 Hz steps."""
    if "gt_endpoint" in d and "endpoint_steps" in d:
        steps = [int(x) for x in d["endpoint_steps"]]
        if step in steps:
            j = steps.index(step)
            tgt = np.asarray(d["gt_endpoint"], dtype=np.float64)[:, j, :]
            if "endpoint_valid" in d:
                val = np.asarray(d["endpoint_valid"]).astype(bool)[:, j]
            else:                              # legacy: assume all valid, say so
                val = np.ones(tgt.shape[0], dtype=bool)
            return tgt, val, f"gt_endpoint[:, {j}] (endpoint_steps={steps})"
    if "gt" in d and "wp_steps" in d:          # legacy fallback: the fan's grid
        wps = [int(x) for x in d["wp_steps"]]
        if step in wps:
            j = wps.index(step)
            tgt = np.asarray(d["gt"], dtype=np.float64)[:, j, :]
            return tgt, np.ones(tgt.shape[0], dtype=bool), \
                f"gt[:, {j}] (wp_steps={wps}) — LEGACY fallback"
    return None, None, f"NOT PRESENT at step {step} ({step * DT:g} s)"


def build_features(d, names: list[str], *, allow_echo: bool,
                   declared: dict[str, str]) -> tuple[np.ndarray, dict]:
    """Concatenate the requested latent blocks, enforcing the admissibility rule."""
    cols, meta, refused = [], [], []
    for nm in names:
        if nm not in d:
            raise KeyError(f"feature block {nm!r} is not in the dump; present: "
                           + ", ".join(sorted(k for k in d
                                              if hasattr(d[k], "shape"))))
        cls = declared.get(nm) or FEATURE_ADMISSIBILITY.get(nm, "UNDECLARED")
        if cls == "ECHO" and not allow_echo:
            refused.append(f"{nm} is the ECHO path (ego/nav at inference) — "
                           f"⛔ inadmissible under the vision-only rule "
                           f"(Sayed 2026-08-03). Pass --allow-echo-features to "
                           f"run it as a LABELLED-INADMISSIBLE control.")
            continue
        if cls == "UNDECLARED":
            refused.append(f"{nm} has no admissibility class — state it with "
                           f"--declare {nm}=VISION_ONLY (or =ECHO). A goal "
                           f"feature whose provenance is unstated cannot be "
                           f"shown to be vision-only.")
            continue
        a = np.asarray(d[nm], dtype=np.float64)
        a = a.reshape(a.shape[0], -1)
        if a.shape[1] == 0:                    # e.g. ctx on a non-hierarchy arm
            meta.append({"block": nm, "admissibility": cls, "dim": 0,
                         "used": False, "reason": "zero-width block, dropped"})
            continue
        cols.append(a)
        meta.append({"block": nm, "admissibility": cls, "dim": int(a.shape[1]),
                     "used": True})
    if refused:
        raise PermissionError(" | ".join(refused))
    if not cols:
        raise ValueError("no usable feature columns after admissibility "
                         "filtering and zero-width drops")
    return np.concatenate(cols, axis=1), {
        "blocks": meta,
        "total_dim": int(sum(c.shape[1] for c in cols)),
        "any_echo": any(m["admissibility"] == "ECHO" for m in meta if m["used"]),
    }


# --------------------------------------------------------------------------- #
# the interval — episode-cluster bootstrap, never overlapping_holdout_se        #
# --------------------------------------------------------------------------- #
def _load_ci():
    """`taniteval.ci` lives in the sibling `taniteval/` package, not in `stack/`.

    Verbatim the loader in ``sel_winners_curse_law.py:68-77`` so both instruments
    resolve the SAME estimator module.
    """
    here = Path(__file__).resolve()
    for up in here.parents:
        cand = up / "taniteval" / "taniteval" / "ci.py"
        if cand.exists():
            sys.path.insert(0, str(up / "taniteval"))
            from taniteval import ci  # noqa: E402
            return ci
    raise ImportError("taniteval/taniteval/ci.py not found above " + str(here))


def sigma_ci(res_sq: np.ndarray, eid, n_boot: int, seed: int = 0) -> dict:
    """Episode-cluster bootstrap CI on ``sigma_perax``.

    ``per_window = |e|² / 2`` with ``reduce="rms"`` gives
    ``sqrt(mean(|e|²/2)) = sigma_perax`` exactly — the reducer does the algebra,
    so the interval is on the SAME statistic as the point estimate rather than on
    a proxy. The point estimate is the **full-set** value; the bootstrap supplies
    only the interval (taniteval/ci.py:225-234).
    """
    ci = _load_ci()
    return ci.episode_cluster_bootstrap(np.asarray(res_sq) / 2.0, list(eid),
                                        reduce="rms", n_boot=n_boot, seed=seed)


def ratio_ci(res_sq: np.ndarray, denom_per_window: np.ndarray, eid,
             n_boot: int, seed: int = 0) -> dict:
    """CI on ``sigma_perax / mean(denom)`` under ONE shared episode resample.

    Numerator and denominator live on the same windows, so they must move
    together inside a draw — combining two separate intervals in quadrature is
    the defect ``paired_episode_cluster_bootstrap`` exists to avoid, and the same
    argument applies to a ratio. ``ci._draws`` is used directly so the episode
    draws are bit-identical to the canonical estimator's.
    """
    ci = _load_ci()
    v = np.asarray(res_sq, dtype=np.float64) / 2.0
    dd = np.asarray(denom_per_window, dtype=np.float64)
    uniq, idx_by_ep = ci.episode_index(list(eid))
    point = float(np.sqrt(np.nanmean(v)) / np.nanmean(dd))
    boots = np.array([float(np.sqrt(np.nanmean(v[s])) / np.nanmean(dd[s]))
                      for s in ci._draws(uniq, idx_by_ep, n_boot, seed)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"ratio": round(point, 4), "lo": round(float(lo), 4),
            "hi": round(float(hi), 4),
            "se": round(float(boots.std(ddof=1)), 4),
            "n_episodes": int(len(uniq)), "n_boot": int(n_boot),
            "estimator": "episode-cluster bootstrap, ONE shared episode draw "
                         "for numerator and denominator (taniteval/ci.py "
                         "draws); ratio of sqrt(mean(|e|^2/2)) to mean(ADE)"}


# --------------------------------------------------------------------------- #
# the verdict                                                                  #
# --------------------------------------------------------------------------- #
def decide(ratio_vs_ade: float | None, guard_failures: list[str]) -> dict:
    """The pre-registered decision. Guards first — they can only produce
    ``NO_VERDICT``, never a softened FUNDED/REFUSED."""
    if guard_failures:
        return {"verdict": "NO_VERDICT",
                "meaning": VERDICTS["NO_VERDICT"],
                "ratio_sigma_over_ade": ratio_vs_ade,
                "refusal_reasons": list(guard_failures)}
    if ratio_vs_ade is None:
        return {"verdict": "NO_VERDICT", "meaning": VERDICTS["NO_VERDICT"],
                "ratio_sigma_over_ade": None,
                "refusal_reasons": ["σ/ADE is not computable on this surface"]}
    if ratio_vs_ade <= PREREG["fund_at_or_below"]:
        v = "FUNDED"
    elif ratio_vs_ade >= PREREG["refuse_at_or_above"]:
        v = "REFUSED"
    else:
        v = "INCONCLUSIVE"
    return {"verdict": v, "meaning": VERDICTS[v],
            "ratio_sigma_over_ade": round(float(ratio_vs_ade), 4),
            "refusal_reasons": []}


def rederive_check(sigma2_matched: float | None,
                   sigma6_matched: float | None) -> dict:
    """§5.3: *"a σ* re-measured at 6 s exceeds 3× the 2 s value ⇒ the ratio form
    does not transfer; the threshold must be re-derived, not scaled."*

    Emits an explicit flag. ⛔ Under NO branch does this function return a scaled
    6 s threshold — when the ratio form holds, the 6 s threshold is still
    ``None`` with the reason, because "holds" licenses re-measuring on a 6 s
    surface, not multiplying the 2 s number by 3.
    """
    if sigma2_matched is None or sigma6_matched is None:
        return {"rederive_required": None,
                "threshold_transfer": "NOT_TESTABLE",
                "threshold_6s": None,
                "reason": "σ at both horizons is required for the §5.3 check; "
                          "one of them is not computable on this surface",
                "multiple": None,
                "limit": PREREG["rederive_multiple"],
                "sigma_perax_2s_matched_m": (None if sigma2_matched is None
                                             else round(float(sigma2_matched), 4)),
                "sigma_perax_6s_matched_m": (None if sigma6_matched is None
                                             else round(float(sigma6_matched), 4)),
                "matched_windows": False}
    mult = float(sigma6_matched / sigma2_matched) if sigma2_matched > 0 else None
    fired = mult is not None and mult > PREREG["rederive_multiple"]
    return {
        "rederive_required": bool(fired),
        "threshold_transfer": "REDERIVE" if fired else "RATIO_FORM_HOLDS",
        "multiple": None if mult is None else round(mult, 4),
        "limit": PREREG["rederive_multiple"],
        "sigma_perax_2s_matched_m": round(float(sigma2_matched), 4),
        "sigma_perax_6s_matched_m": round(float(sigma6_matched), 4),
        "matched_windows": True,
        "threshold_6s": None,
        "reason": ("σ(6 s) exceeds 3× σ(2 s) — §5.3 REFUTATION: the ratio form "
                   "does NOT transfer. The 1.7 / 3.0 thresholds must be "
                   "RE-DERIVED on a 6 s fan, NOT scaled. No 6 s threshold is "
                   "emitted." if fired else
                   "σ(6 s) is within 3× σ(2 s), so §5.3's refutation does not "
                   "fire. This licenses RE-MEASURING the threshold on a 6 s "
                   "fan; it does NOT license scaling the 2 s threshold, so no "
                   "6 s threshold is emitted here either."),
    }


# --------------------------------------------------------------------------- #
# the run                                                                      #
# --------------------------------------------------------------------------- #
def run(d, *, features: list[str], allow_echo: bool = False,
        declared: dict[str, str] | None = None, n_boot: int = 2000,
        seed: int = 0, min_episodes: int | None = None,
        min_windows: int | None = None,
        lambdas: tuple[float, ...] = RIDGE_LAMBDAS) -> dict:
    """The whole instrument on an in-memory dump dict. Pure CPU, no I/O."""
    declared = dict(declared or {})
    min_ep = PREREG["min_episodes"] if min_episodes is None else int(min_episodes)
    min_win = PREREG["min_windows"] if min_windows is None else int(min_windows)
    relaxed = (min_ep < PREREG["min_episodes"]) or (min_win < PREREG["min_windows"])

    contract_problems = validate_dump(d)
    eid = [int(x) for x in d["eid"]]
    n = len(eid)
    X, feat_meta = build_features(d, features, allow_echo=allow_echo,
                                  declared=declared)
    if X.shape[0] != n:
        raise ValueError(f"features have {X.shape[0]} rows, eid has {n}")

    vstep = int(round(PREREG["verdict_horizon_s"] / DT))     # 20
    tstep = int(round(PREREG["transfer_horizon_s"] / DT))    # 60
    horizons: dict[str, dict] = {}
    sig: dict[int, dict] = {}
    valid: dict[int, np.ndarray] = {}
    for step in (vstep, tstep):
        tgt, val, src = resolve_endpoint(d, step)
        key = f"{step * DT:g}s"
        if tgt is None:
            horizons[key] = {"available": False, "reason": src, "n": 0}
            continue
        # ⛔ Defence in depth: the producer writes NaN where a horizon runs past
        # the end of an episode. A dump with no `endpoint_valid` (legacy) would
        # otherwise let those NaNs into the fit and return a NaN σ that reads
        # like a number. Dropped, counted, and reported — never imputed.
        val = np.asarray(val, dtype=bool)
        finite = np.isfinite(tgt).all(axis=1)
        n_nonfinite = int((val & ~finite).sum())
        val = val & finite
        valid[step] = val
        if int(val.sum()) < 2 or np.unique(np.asarray(eid)[val]).size < 2:
            horizons[key] = {"available": False, "n": int(val.sum()),
                             "reason": f"{int(val.sum())} valid windows / "
                                       f"{np.unique(np.asarray(eid)[val]).size} "
                                       f"episodes — LOEO not constructible",
                             "source": src}
            continue
        s = fit_sigma(X[val], tgt[val], np.asarray(eid)[val], lambdas=lambdas)
        sig[step] = s
        # only the bootstrap payload is stripped — `_unit_note` is load-bearing
        # documentation and MUST travel with the number it qualifies.
        row = {k: v for k, v in s.items() if k != "_residual_sq_per_window"}
        row.update(available=True, source=src, horizon_s=step * DT,
                   n_windows=int(val.sum()),
                   n_episodes=int(np.unique(np.asarray(eid)[val]).size),
                   n_excluded=int((~val).sum()),
                   n_nonfinite_dropped=n_nonfinite,
                   excluded_reason="the horizon runs past the end of the "
                                   "episode; ⛔ excluded, never imputed")
        if n_boot > 0:
            try:
                row["sigma_perax_ci"] = sigma_ci(s["_residual_sq_per_window"],
                                                 np.asarray(eid)[val], n_boot,
                                                 seed)
            except ImportError as exc:
                row["sigma_perax_ci"] = {"unavailable": str(exc)}
        horizons[key] = row

    # ---- the §3.1 references, on the VERDICT horizon's windows ---------------
    vmask = valid.get(vstep)
    refs = fan_references(d, vmask)
    ratios: dict = {"_note": "σ/ADE and σ/oracle at the VERDICT horizon "
                             f"({PREREG['verdict_horizon_s']:g} s), where §3.1's "
                             "fan references exist. σ is PER-AXIS."}
    ratio_vs_ade = None
    if refs.get("available") and vstep in sig:
        rh = refs.get("reference_horizon_s")
        if rh is not None and abs(rh - PREREG["verdict_horizon_s"]) > 1e-6:
            ratios["MISMATCH"] = (
                f"⛔ the fan's last waypoint is {rh:g} s but the verdict horizon "
                f"is {PREREG['verdict_horizon_s']:g} s — the ratios are NOT "
                f"emitted; a σ at one horizon over an ADE at another is not a "
                f"quantity")
        else:
            s2 = sig[vstep]["sigma_perax_m"]
            ratio_vs_ade = (s2 / refs["sel_ade"]) if refs.get("sel_ade") else None
            ratios["sigma_perax_2s_m"] = round(s2, 4)
            ratios["oracle_ade"] = round(refs["oracle_ade"], 4)
            ratios["fan_mean_ade"] = round(refs["fan_mean_ade"], 4)
            if refs.get("sel_ade") is not None:
                ratios["sel_ade_incumbent"] = round(refs["sel_ade"], 4)
                ratios["sigma_over_ade"] = round(float(ratio_vs_ade), 4)
            ratios["sigma_over_oracle"] = round(s2 / refs["oracle_ade"], 4)
            if refs.get("cv_ade") is not None:
                ratios["cv_ade"] = round(refs["cv_ade"], 4)
            ratios["reference_from_3_1"] = {
                "sigma_star_m": PREREG["reference_sigma_star_m"],
                "sigma_over_ade": PREREG["reference_ratio_vs_ade"],
                "sigma_over_oracle": PREREG["reference_ratio_vs_oracle"]}
            if n_boot > 0 and "_per_window_sel" in refs:
                try:
                    ratios["sigma_over_ade_ci"] = ratio_ci(
                        sig[vstep]["_residual_sq_per_window"],
                        refs["_per_window_sel"], np.asarray(eid)[vmask]
                        if vmask is not None else eid, n_boot, seed)
                except ImportError as exc:
                    ratios["sigma_over_ade_ci"] = {"unavailable": str(exc)}
    else:
        ratios["available"] = False
        ratios["reason"] = refs.get("reason", "the verdict-horizon σ is not "
                                              "computable on this dump")

    # ---- §5.3 REDERIVE, on MATCHED windows ----------------------------------
    s2m = s6m = None
    matched_note = "not computable"
    if vstep in sig and tstep in valid and tstep in sig:
        m = valid[vstep] & valid[tstep]
        if int(m.sum()) >= 2 and np.unique(np.asarray(eid)[m]).size >= 2:
            t2, _, _ = resolve_endpoint(d, vstep)
            t6, _, _ = resolve_endpoint(d, tstep)
            s2m = fit_sigma(X[m], t2[m], np.asarray(eid)[m],
                            lambdas=lambdas)["sigma_perax_m"]
            s6m = fit_sigma(X[m], t6[m], np.asarray(eid)[m],
                            lambdas=lambdas)["sigma_perax_m"]
            matched_note = (f"both σ re-fit on the {int(m.sum())} windows valid "
                            f"at BOTH horizons ({int(np.unique(np.asarray(eid)[m]).size)} "
                            f"episodes) — comparing a full-grid σ(2 s) against a "
                            f"truncated σ(6 s) would compare different windows")
    rederive = rederive_check(s2m, s6m)
    rederive["matched_window_note"] = matched_note

    # ---- the guards ---------------------------------------------------------
    guards: list[str] = []
    n_ep = len(set(eid))
    if n_ep < min_ep:
        guards.append(f"{n_ep} episodes < the pre-registered {min_ep}")
    if n < min_win:
        guards.append(f"{n} windows < the pre-registered {min_win}")
    if relaxed:
        guards.append(f"the pre-registered guards were RELAXED on the command "
                      f"line (min_episodes={min_ep} vs {PREREG['min_episodes']}, "
                      f"min_windows={min_win} vs {PREREG['min_windows']}) — a "
                      f"verdict from a relaxed surface is not admissible")
    if not horizons.get(f"{vstep * DT:g}s", {}).get("available"):
        guards.append(f"no σ at the verdict horizon {vstep * DT:g} s")
    if not horizons.get(f"{tstep * DT:g}s", {}).get("available"):
        guards.append(f"no σ at the transfer horizon {tstep * DT:g} s — §5.2 "
                      f"requires 2 s AND 6 s")
    if feat_meta["any_echo"]:
        guards.append("an ECHO (ego/nav-at-inference) feature block is in the "
                      "design matrix — this arm is a LABELLED-INADMISSIBLE "
                      "control and may not produce a deployment verdict")
    if contract_problems:
        guards.append("dump-contract violations: " + "; ".join(contract_problems))
    for step in (vstep, tstep):
        if step in sig and sig[step]["n_folds"] != len(
                set(np.asarray(eid)[valid[step]].tolist())):
            guards.append(f"LOEO folds != episodes at {step * DT:g} s")

    verdict = decide(ratio_vs_ade, guards)

    return {
        "item": "E-WC2 — can the tactical goal head reach σ* on our corpus at all?",
        "_prereg": PREREG,
        "_evidence_class": "MEASURED (ours) — LOEO ridge re-analysis of a banked "
                           "latent dump; no model, no GPU, no re-inference at "
                           "analysis time. The GPU cost is the DUMP, not this.",
        "_class": "EXPLORATORY when the surface is a REF-C fan (a different arm, "
                  "2 s horizon): the RATIOS are the transferable claim, the "
                  "absolute metres are that fan's.",
        "_tier": "T0-DIAGNOSTIC — a representation-capacity probe on banked "
                 "latents. NOT a driving-performance number; no T1 capability "
                 "claim may cite it (EVAL_DOCTRINE.md).",
        "_estimator": "point estimates are full-set; intervals are the "
                      "episode-cluster bootstrap (taniteval/ci.py). "
                      "overlapping_holdout_se is used NOWHERE.",
        "_ridge": "probe_latent_state.RidgeSVD + GCV λ on the TRAIN fold only "
                  "+ TRAIN-stat standardisation (the P1/P2 battery's method, "
                  "imported, not re-implemented)",
        "surface": {
            "n_windows": int(n), "n_episodes": int(n_ep),
            "ckpt": str(d.get("ckpt", "")),
            "ckpt_step": int(d.get("ckpt_step", -1)) if d.get("ckpt_step") is not None else None,
            "nav_mode": str(d.get("nav_mode", "")),
            "controls_vs_bank": d.get("controls_vs_bank"),
            "producer_instrument_fail": list(d.get("instrument_fail", [])),
            "contract_problems": contract_problems,
        },
        "features": feat_meta,
        "sigma": horizons,
        "references_and_ratios": ratios,
        "rederive_check_5_3": rederive,
        "decision": verdict,
        "four_families": {
            "LONGITUDINAL": "sigma_long_m per horizon (endpoint along-track 1σ). "
                            "Target-speed / distance-keeping are NOT produced "
                            "here — this is an endpoint-capacity probe, n/a with "
                            "reason.",
            "LATERAL": "sigma_lat_m per horizon (endpoint cross-track 1σ). "
                       "Heading / curvature / yaw-rate n/a — no trajectory is "
                       "rolled by this instrument.",
            "TACTICAL": "THE headline: σ/ADE and σ/oracle are the goal/anchor "
                        "selection admissibility test (§3.1 B's requirement "
                        "curve, read from the capability side).",
            "STRATEGIC": "n/a with reason — PhysicalAI-AV ships no map, lane "
                         "graph, junction annotation or route signal (§6).",
        },
    }


# --------------------------------------------------------------------------- #
def _parse_declared(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--declare expects NAME=CLASS, got {it!r}")
        k, v = it.split("=", 1)
        if v not in ("VISION_ONLY", "ECHO"):
            raise SystemExit(f"--declare class must be VISION_ONLY or ECHO, "
                             f"got {v!r}")
        out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("e_wc2_sigma_star", description=__doc__[:400])
    ap.add_argument("--dump", help="the latent dump (see --print-contract)")
    ap.add_argument("--out", help="write the result JSON here")
    ap.add_argument("--features", default="pooled,ctx",
                    help="comma-separated latent blocks (default: pooled,ctx)")
    ap.add_argument("--allow-echo-features", action="store_true",
                    help="run an ECHO (ego/nav) block as a LABELLED-INADMISSIBLE "
                         "control; forces NO_VERDICT")
    ap.add_argument("--declare", action="append", default=[],
                    help="NAME=VISION_ONLY|ECHO for a block with no built-in class")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-episodes", type=int, default=PREREG["min_episodes"])
    ap.add_argument("--min-windows", type=int, default=PREREG["min_windows"])
    ap.add_argument("--print-contract", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args(argv)

    if a.print_contract:
        print(json.dumps(DUMP_CONTRACT, indent=2, ensure_ascii=False))
        return 0
    if not a.dump:
        ap.error("--dump is required (or --print-contract)")

    import torch
    d = torch.load(a.dump, map_location="cpu", weights_only=False)

    if a.validate_only:
        problems = validate_dump(d)
        print(json.dumps({"dump": a.dump, "conformant": not problems,
                          "problems": problems}, indent=2, ensure_ascii=False))
        return 0 if not problems else 3

    res = run(d, features=[s for s in a.features.split(",") if s],
              allow_echo=a.allow_echo_features,
              declared=_parse_declared(a.declare),
              n_boot=a.n_boot, seed=a.seed,
              min_episodes=a.min_episodes, min_windows=a.min_windows)
    res["surface"]["dump_path"] = str(a.dump)
    txt = json.dumps(res, indent=2, ensure_ascii=False, default=float)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)
    dec = res["decision"]
    print(f"\n[E-WC2] verdict: {dec['verdict']} — {dec['meaning']}", flush=True)
    if dec["refusal_reasons"]:
        for r in dec["refusal_reasons"]:
            print(f"[E-WC2]   refusal: {r}", flush=True)
    rd = res["rederive_check_5_3"]
    print(f"[E-WC2] §5.3 transfer: {rd['threshold_transfer']} "
          f"(multiple={rd['multiple']}, limit={rd['limit']})", flush=True)
    return 0 if dec["verdict"] != "NO_VERDICT" else 4


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
