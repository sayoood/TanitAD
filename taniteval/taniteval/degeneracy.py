"""K1 DEGENERACY GUARD — is a K1 verdict about the LATENT, or about WHICH CONSTANT?

⛔ WHY THIS EXISTS — TWO DEFECTS THAT ARE MIRROR IMAGES OF EACH OTHER.

**C92 (2026-08-18).** ``pc6_linear_readout.ridge_fit`` put the appended
ones-column INSIDE ``alpha * np.eye(d)``, so the intercept was shrunk like any
other coefficient and predictions collapsed toward **ZERO, not the MEAN**. A
no-signal arm therefore scored WORSE than a constant BY CONSTRUCTION ⇒ the
**floor** was biased ⇒ every ``K1 FAIL`` from that module was suspect.

**C97 (2026-08-18).** The repair (``intercept_col=-1``) opened the mirror defect.
A fully-shrunk repaired ridge **is** "predict the train MEAN", while ``C-CONST``
is the train **MEDIAN**. On a skewed target K1 degenerates into a
**mean-versus-median contest**, and a pure ``torch.randn`` null "PASSES"
``n_agents_all`` at **−1.884** with ``pred_sd`` **0.715** against ``gt_sd``
**46.459** — a flat line beating a constant because it is a *different, luckier*
constant.

⇒ **C92 made no-signal arms FAIL by construction; C97 makes them PASS by
construction. Same criterion, opposite bias, caused by its own repair.** The
lesson is C95's: *when you loosen a criterion, test the direction you were not
trying to fix.* A guard that rejects everything is as useless as one that
rejects nothing, and this programme built one of each inside 24 hours.

===============================================================================
THE MECHANISM — an EXACT decomposition, not a heuristic
===============================================================================

Write ``c_own = mean(pred)``, the arm's **own** constant. Then, identically::

    K1  =  MAE(pred) − MAE(c_const)
        = [MAE(pred) − MAE(c_own)]  +  [MAE(c_own) − MAE(c_const)]
        =        K1B               +          K1C

* **K1B — the LATENT-ATTRIBUTABLE part.** What the readout's *variation* buys
  over its own mean level. This is the quantity a claim about the latent is
  actually about, and it is **invariant to the choice of C-CONST**, so the whole
  mean-versus-median question cannot touch it.
* **K1C — the WHICH-CONSTANT part.** A contest between two constants. On a
  z-scored design the ridge intercept is ``mean(y_train)``, which carries
  **zero** latent information — so K1C is never evidence about the latent, it is
  evidence about the target's skew and the train/eval shift.

⭐ **AND K1B IS BOUNDED, EXACTLY.** By the reverse triangle inequality
``||a|−|b|| ≤ |a−b|`` applied per window with ``a = pred−y``, ``b = c_own−y``::

    |K1B|  ≤  mean|pred − c_own|  =  pred_mad  ≤  pred_sd            (Jensen)

⇒ **if ``pred_sd < |K1|`` then at least ``|K1| − pred_sd`` of that delta is
PROVABLY the constant offset and not the latent.** That screen needs no
threshold, no refit and no bootstrap — ``K1_delta`` and ``pred_sd`` are already
banked in every artifact this module guards (:func:`screen_banked_k1`).

===============================================================================
THE THREE LAYERS, and which one decides
===============================================================================

======  ===================================  ==========================
layer   statistic                            status
======  ===================================  ==========================
1       ``pred_sd < |K1|``                   EXACT (a theorem). Screens
                                             banked JSON at zero compute.
2       **K1B, paired episode-cluster        ⭐ **THE DECISION.** Same
        bootstrap**                          estimator as K1 itself.
3       ``sd_ratio = pred_sd / gt_sd``       DESCRIPTIVE. Readable, and
                                             the only layer with a
                                             threshold — so it never
                                             decides alone.
======  ===================================  ==========================

⚠️ **Layer 3 is the one C97 asked for by name, and it is deliberately the
weakest.** A bare ``pred_sd/gt_sd`` cut is a *threshold nobody registered*,
exactly what ``ci._render_bounds`` refused to introduce. It is reported because
it is readable and because it is computable from banked files, but the verdict
rests on layer 2, which has no free parameter.

===============================================================================
⛔ WHAT THIS MODULE DELIBERATELY DOES **NOT** DO
===============================================================================

**It does not move ``C-CONST`` from the MEDIAN to the MEAN.** That is the
tempting "fix" for the mean-vs-median mismatch and it is wrong twice over:

1. K1 is scored in **MAE**, and the MAE-optimal constant **is the median**.
   Swapping in the mean would replace the strongest honest baseline with a
   demonstrably weaker one under the loss actually used — i.e. it would
   **manufacture PASSes**. That is C97's own failure mode a third time, and
   C95's lesson says a criterion must not be loosened without testing the
   direction you were not trying to fix.
2. It would silently rewrite the meaning of **214 banked verdict rows** whose
   filenames would not change — the same objection that keeps
   ``ridge_fit``'s default at the incumbent behaviour (C92 precedent: make the
   correct path available and explicit, re-read the banked numbers, never mutate
   the code underneath them).

⇒ **The mismatch is dissolved, not adjudicated: C-CONST stays the median, and
K1B — which does not depend on C-CONST at all — is what decides whether a K1 is
about the latent.** ``k1_guard`` additionally reports what the MEAN constant
would have scored (``c_mean_*``), so the gap is *visible* instead of being an
invisible route to a PASS.

===============================================================================
⚠️ WHAT THE GUARD DOES **NOT** ANSWER — read this before quoting an ``OK``
===============================================================================

**The guard qualifies ATTRIBUTION, not MAGNITUDE.** ``OK`` means *"this verdict
is about the readout's variation"*; it does **not** mean the effect is large
enough to matter.

MEASURED 2026-08-18 (``…/2026-08-18-k1-degeneracy-guard/raw/reread/llR_nullmatched.json``):
the random-latent null on ``ego_yawrate`` returns ``K1 = +0.0000
[+0.0000, +0.0000] separated`` with ``K1B`` likewise ``+0.0000`` separated, and
the guard correctly says ``OK`` — the null's minuscule noise variation really
does make it very slightly worse than a constant. The verdict is **attributable
and physically nil** (``pred_sd`` 0.0002 rad/s). ``ci._render_bounds`` catches
only the float64-resolution case below 1e-12; between that and "meaningful"
there is a judgement that needs the target's UNITS, and this module does not
make it.

⇒ **Always read a verdict beside a scale.** ``K1B / gt_sd`` is the scale-free
form and is derivable from the fields emitted here. ⛔ **No threshold on it is
defined on purpose** — that would be a tunable nobody registered, which is the
mistake this module spends its whole design avoiding.

⛔ **T0-DIAGNOSTIC.** Everything guarded here is a frozen-latent readout — a
world-model diagnostic, never driving performance (``EVAL_DOCTRINE.md``).
"""
from __future__ import annotations

import numpy as np

from .ci import paired_episode_cluster_bootstrap

__all__ = [
    "SD_RATIO_FLAT_FLOOR",
    "GUARD_VERDICTS",
    "screen_banked_k1",
    "k1_guard",
]

#: Layer-3 screen only. A readout whose spread is under this fraction of the
#: target's spread is a flat line for reporting purposes. ⚠️ This is the one
#: tunable in the module and it NEVER decides alone — see the layer table.
SD_RATIO_FLAT_FLOOR = 0.05

GUARD_VERDICTS = (
    "OK",                    # the K1 verdict survives removing the constant
    "CONSTANT-OFFSET-ONLY",  # K1 separates, K1B does not ⇒ about WHICH constant
    "DEGENERATE-CONSTANT",   # the readout is a flat line
    "NO-VERDICT-TO-GUARD",   # K1 did not separate; there is nothing to attribute
)


def _mae(a, b) -> float:
    return float(np.abs(np.asarray(a, dtype=np.float64)
                        - np.asarray(b, dtype=np.float64)).mean())


def screen_banked_k1(k1_delta, pred_sd, gt_sd,
                     k1_separated=None,
                     sd_ratio_floor: float = SD_RATIO_FLAT_FLOOR) -> dict:
    """LAYER 1 + 3 from **banked numbers alone** — no refit, no predictions.

    Every artifact this module guards already records ``K1_delta``, ``pred_sd``
    and ``gt_sd``, so 214 banked verdict rows can be screened at zero compute.

    ``k1_exceeds_own_spread`` is a **theorem, not a heuristic**: since
    ``|K1B| ≤ pred_mad ≤ pred_sd``, a ``|K1_delta|`` larger than ``pred_sd``
    cannot be produced by the prediction's variation, so at least
    ``|K1_delta| − pred_sd`` of it is the constant offset.

    ⚠️ The converse does NOT hold. Passing this screen is **not** evidence the
    verdict is latent-attributable — that needs :func:`k1_guard`'s layer 2.
    """
    k1_delta = float(k1_delta)
    pred_sd = float(pred_sd)
    gt_sd = float(gt_sd)
    exceeds = bool(pred_sd < abs(k1_delta))
    ratio = float(pred_sd / gt_sd) if gt_sd > 0 else float("inf")
    return {
        "K1_delta": k1_delta,
        "pred_sd": pred_sd,
        "gt_sd": gt_sd,
        "sd_ratio": round(ratio, 6),
        "flat_line": bool(ratio < sd_ratio_floor),
        "sd_ratio_floor": sd_ratio_floor,
        "k1_exceeds_own_spread": exceeds,
        "min_constant_component": round(max(0.0, abs(k1_delta) - pred_sd), 6),
        "provable_note": (
            "|K1B| <= pred_mad <= pred_sd (reverse triangle inequality, then "
            "Jensen). pred_sd < |K1_delta| therefore PROVES a constant-offset "
            "component of at least |K1_delta| - pred_sd."),
        "screen_verdict": (
            "SUSPECT — constant-offset component proven"
            if exceeds else
            "SUSPECT — readout is a flat line" if ratio < sd_ratio_floor else
            "not screened out (layer 2 still required)"),
        "k1_separated": (None if k1_separated is None else bool(k1_separated)),
        "layers_applied": [1, 3],
        "_evidence_class": "DERIVED (algebra on banked K1_delta/pred_sd/gt_sd)",
    }


def k1_guard(pred, y_eval, eid, c_const, n_boot: int = 2000, seed: int = 0,
             alpha: float = 0.05,
             sd_ratio_floor: float = SD_RATIO_FLAT_FLOOR,
             c_mean=None) -> dict:
    """⭐ THE FULL GUARD — decomposes K1 into its latent and constant parts.

    ``pred``   [N] the readout's eval predictions
    ``y_eval`` [N] the ground truth on the same windows
    ``eid``    [N] episode ids — the bootstrap's clusters
    ``c_const`` the scalar C-CONST the K1 was scored against (the PROBE-TRAIN
                **median**; see the module docstring for why it stays the median)
    ``c_mean``  optional: the PROBE-TRAIN **mean**, reported for visibility of
                the mean-vs-median gap. It is NEVER used as a baseline.

    Returns the exact decomposition ``K1 = K1B + K1C``, both bootstrapped with
    ``paired_episode_cluster_bootstrap`` (⛔ never ``overlapping_holdout_se``),
    plus ``guard_verdict`` and ``K1_quotable_as_latent_evidence``.

    ⚠️ **The guard is direction-symmetric on purpose.** It qualifies a separated
    **FAIL** exactly as it qualifies a **PASS** — C92 biased FAILs, C97 biases
    PASSes, and a guard that only watched one of them would be the next entry in
    this same family.
    """
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y_eval, dtype=np.float64)
    if pred.shape != y.shape:
        raise ValueError(f"pred/y must be aligned: {pred.shape} vs {y.shape}")
    if len(eid) != pred.size:
        raise ValueError(f"eid/pred length mismatch: {len(eid)} vs {pred.size}")
    c_const = float(c_const)

    c_own = float(pred.mean())
    e_arm = np.abs(pred - y)
    e_own = np.abs(c_own - y)
    e_con = np.abs(c_const - y)

    # unrounded points, so the decomposition is EXACT rather than "to 4 dp"
    k1_point = float(e_arm.mean() - e_con.mean())
    k1b_point = float(e_arm.mean() - e_own.mean())
    k1c_point = float(e_own.mean() - e_con.mean())

    k1 = paired_episode_cluster_bootstrap(e_arm, e_con, eid, n_boot=n_boot,
                                          seed=seed, alpha=alpha)
    k1b = paired_episode_cluster_bootstrap(e_arm, e_own, eid, n_boot=n_boot,
                                           seed=seed, alpha=alpha)

    pred_sd = float(pred.std())
    pred_mad = float(np.abs(pred - c_own).mean())
    gt_sd = float(y.std())
    ratio = float(pred_sd / gt_sd) if gt_sd > 0 else float("inf")
    flat = bool(ratio < sd_ratio_floor)

    # ci.py marks an interval whose bounds are at float64 resolution: the arms
    # are effectively identical, so `separated` there is arithmetic not evidence.
    k1b_sep = bool(k1b["separated"] and not k1b.get("degenerate", False))
    same_sign = bool(k1b_point * k1_point > 0)
    k1_sep = bool(k1["separated"] and not k1.get("degenerate", False))

    # ⭐ K1B DECIDES; sd_ratio only DESCRIBES. The order matters and it was wrong
    # in the first draft of this module: letting `flat_line` short-circuit K1B
    # would reject a genuinely-skilled readout on a HEAVY-TAILED target, where
    # `gt_sd` is inflated by a handful of extreme windows while the readout
    # tracks the bulk. `n_agents_all` (gt_sd 46.5 against a median of 34) is
    # exactly that shape. ⇒ a flat-looking arm that nonetheless BEATS ITS OWN
    # MEAN, paired and separated, has demonstrated latent-attributable skill and
    # is passed. `flat_line` then only chooses the LABEL for a failure K1B has
    # already decided — DEGENERATE-CONSTANT is the specific case of
    # CONSTANT-OFFSET-ONLY where the readout is also a flat line.
    if not k1_sep:
        verdict = "NO-VERDICT-TO-GUARD"
    elif k1b_sep and same_sign:
        verdict = "OK"
    elif flat:
        verdict = "DEGENERATE-CONSTANT"
    else:
        verdict = "CONSTANT-OFFSET-ONLY"

    out = {
        "_evidence_class": "MEASURED (ours; K1 degeneracy guard)",
        "eval_tier": "T0-DIAGNOSTIC",
        "estimator": "taniteval.ci.paired_episode_cluster_bootstrap",
        "forbidden": "overlapping_holdout_se",

        # ---- the exact decomposition K1 = K1B + K1C ----------------------
        "K1_delta": round(k1_point, 6),
        "K1_lo": k1["lo"], "K1_hi": k1["hi"],
        "K1_separated": k1_sep,
        "K1_PASSES": bool(k1_sep and k1_point < 0),
        "K1B_delta": round(k1b_point, 6),          # LATENT-ATTRIBUTABLE
        "K1B_lo": k1b["lo"], "K1B_hi": k1b["hi"],
        "K1B_separated": k1b_sep,
        "K1B_PASSES": bool(k1b_sep and k1b_point < 0),
        "K1C_delta": round(k1c_point, 6),          # WHICH-CONSTANT
        "decomposition_residual": round(k1_point - (k1b_point + k1c_point), 12),

        # ---- layer 1: the theorem ---------------------------------------
        "pred_mad": round(pred_mad, 6),
        "pred_sd": round(pred_sd, 6),
        "gt_sd": round(gt_sd, 6),
        "k1_exceeds_own_spread": bool(pred_mad < abs(k1_point)),
        "min_constant_component": round(max(0.0, abs(k1_point) - pred_mad), 6),
        "bound_holds": bool(abs(k1b_point) <= pred_mad + 1e-9),

        # ---- layer 3: the readable screen -------------------------------
        "sd_ratio": round(ratio, 6),
        "sd_ratio_floor": sd_ratio_floor,
        "flat_line": flat,

        # ---- the constants, so the mean-vs-median gap is VISIBLE ---------
        "c_const_value": round(c_const, 6),
        "c_const_err": round(float(e_con.mean()), 6),
        "c_own_value": round(c_own, 6),
        "c_own_err": round(float(e_own.mean()), 6),
        "arm_err": round(float(e_arm.mean()), 6),
        "n_windows": int(pred.size),
        "n_episodes": int(k1["n_episodes"]),

        # ---- the verdict -------------------------------------------------
        "guard_verdict": verdict,
        "K1_quotable_as_latent_evidence": bool(verdict == "OK"),
        "guard_note": {
            "OK": "K1's sign and separation survive removing the arm's own "
                  "constant ⇒ attributable to the latent.",
            "CONSTANT-OFFSET-ONLY":
                "⛔ K1 separates but K1B does not (same sign). This verdict is "
                "about WHICH CONSTANT wins on a skewed target, not about the "
                "latent. C97: a torch.randn null 'PASSES' this way.",
            "DEGENERATE-CONSTANT":
                "⛔ K1B does not carry the verdict AND pred_sd/gt_sd is below "
                "the flat-line floor: the readout emits essentially no "
                "variation, so its K1 is a constant-vs-constant contest by "
                "construction. (The FAILURE is decided by K1B; the flat line "
                "only names it.)",
            "NO-VERDICT-TO-GUARD":
                "K1 did not separate; there is no verdict to attribute.",
        }[verdict],
    }
    if c_mean is not None:
        e_cm = np.abs(float(c_mean) - y)
        out["c_mean_value"] = round(float(c_mean), 6)
        out["c_mean_err"] = round(float(e_cm.mean()), 6)
        out["mean_minus_median_const_gap"] = round(
            float(e_cm.mean() - e_con.mean()), 6)
        out["mean_vs_median_note"] = (
            "NEGATIVE ⇒ the train MEAN is the better MAE constant on these "
            "eval windows, which is the exact route C97 found: a fully-shrunk "
            "repaired ridge IS the train mean. C-CONST deliberately stays the "
            "MEDIAN (MAE-optimal on train); K1B is what decides.")
    return out
