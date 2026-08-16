"""E-AG1 / E-AG2 / E-AG3 — can ``ANCHOR_GOAL`` supervision reach the goal bar?

⛔ **THIS FILE IS THE PRE-REGISTRATION.** Both outcomes of every arm are committed
in :data:`PREREG` and :data:`VERDICTS` **before** the instrument was first run, in
the same shape as ``e_wc2_sigma_star.py`` — the instrument that refused SEL-1 on
2026-08-16. Reading a number and then choosing a threshold is the failure this
pattern exists to prevent.

WHY THIS EXISTS
---------------
E-WC2 measured σ(2 s) = **4.7104 m** per-axis for a ridge on frozen REF-C vision
latents ⇒ σ/ADE **9.9915 [7.4492, 13.5119]** ⇒ **SEL-1 REFUSED** (§5.2's committed
line was σ/ADE ≥ 3.0). Its committed fallback is `ANCHOR_GOAL` supervision. But
E-WC2 also measured that a **0-parameter constant-yaw-rate** goal reaches
**1.1888 m** — 3.96× better — so the estimand survives and the *surface* is the
defect. ⚠️ And even that floor sits at σ/ADE **2.52**, i.e. still not FUNDED.

`ANCHOR_GOAL(anchor_id ∈ fan vocab, t_reach_s)` is a **DIFFERENT ESTIMAND from a
free 2-vector regression**: the goal point is `anchor_table[anchor_id]`, so the
head solves a **K-way classification** over a fixed vocabulary instead of a
regression. That change has a consequence a regression does not have:

  ⭐ **its error is bounded below by the vocabulary's own quantisation, which is
     a property of GEOMETRY ALONE — no model, no surface, no GPU.**

⇒ **The cheapest possible refusal test.** If the quantisation floor of the largest
affordable vocabulary already exceeds the admission bar, `ANCHOR_GOAL` is refused
**with a perfect classifier**, before a single GPU-minute, exactly as E-WC2
refused SEL-1. That is E-AG1.

THE THREE ARMS
--------------
* **E-AG1 — the quantisation floor.** LOEO: build the K-anchor vocabulary from 39
  episodes, assign the held-out episode's windows to their ORACLE anchor, report
  σ of the residual. This is the σ a **perfect** classifier would reach.
* **E-AG2 — does discretising beat regressing, ON THE SAME SURFACE?** A K-way
  linear classifier on the identical VISION_ONLY features (`pooled` + `ctx`),
  identical LOEO folds, against the identical E-WC2 ridge re-fit here as a parity
  control. Paired episode-cluster bootstrap.
* **E-AG3 — the `v0` admissibility contradiction, MEASURED not argued.**
  ⛔ `e_wc2_sigma_star.FEATURE_ADMISSIBILITY["v0"] = "ECHO"` (inadmissible) while
  `V6F_PLANNER_DESIGN.md` §1.4 row 3 declares **`true v0` ✅ admissible** for the
  emitted fan. Both are at HEAD and they disagree. This arm runs ONLY under
  ``--allow-echo-features`` and is stamped **PENDING_PI_ADJUDICATION** — it is
  reported as *the magnitude of the contradiction*, never as a funded result.

⛔ ADMISSIBILITY, ENFORCED NOT DOCUMENTED
----------------------------------------
Features are filtered through ``e_wc2_sigma_star.build_features``, so the
VISION_ONLY rule is the same code path E-WC2 used. Separately, and additionally:

  ⛔ **THE ANCHOR VOCABULARY AND THE LABELS ARE INFORMATION-DISJOINT FROM
     ``tanitad.data.situations``.** The label is `argmin_k ‖endpoint − anchor_k‖`
     over ego-frame FUTURE displacement; no detector output, no class posterior,
     no argmax, no embedding derived from one enters at any point. This module
     does not import `situations` and must never do so — pinned by
     ``tests/test_e_ag1_anchor_floor.py::test_no_situation_classifier_path``.
     (The same load-bearing omission as ``ph0_pilot._fmt_engine_a``, which
     computes `situations` and deliberately does not forward it into the B4 goal
     prompt.)

  ⛔ **THE ECHO TEST ON THIS DESIGN.** At inference the goal point is
     `anchor_table[argmax_k head(z_tac_p, e_g_str)]` — a fixed buffer indexed by
     a vision-derived logit. The label is derived from FUTURE ego poses
     `[t, t+h]`; no input at inference is derived from those. The vocabulary
     itself is built from the TRAIN episodes' futures and is a **frozen buffer**
     shared by every window, so it carries no per-window information — which is
     exactly why the goal-echo control (goal ← corpus marginal) is the right
     null and is reported as ``marginal`` below.

Estimator: point estimates are **full-set**; intervals are the **episode-cluster
bootstrap** over the val episodes (``taniteval/ci.py``); arm-vs-arm comparisons
are **paired** on the same windows. ⛔ ``overlapping_holdout_se`` is used nowhere.

Tier: ⛔ **T0-DIAGNOSTIC.** A representation/geometry capacity probe on banked
latents. **No T1 capability claim may cite it.**

Usage
-----
    python scripts/e_ag1_anchor_floor.py \
        --dump ".../raw/latents_refc-xl-30k-ep.pt" \
        --out  ".../raw/e_ag1_anchor_floor_refc-xl-30k.json"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e_wc2_sigma_star as E                                       # noqa: E402
from probe_latent_state import RIDGE_LAMBDAS, _standardize         # noqa: E402

DT = 0.1

# --------------------------------------------------------------------------- #
# ⛔ THE PRE-REGISTRATION, as data. Written BEFORE the first run.               #
# --------------------------------------------------------------------------- #
PREREG = {
    "source": "TanitAD Research Hub/.../2026-08-16-anchor-goal-supervision/"
              "ANCHOR_GOAL_SUPERVISION.md §4, deriving from "
              "V6F_PLANNER_DESIGN.md §3.1 B / §5.2 and the E-WC2 result.",
    "incumbent_sel_ade_m": 0.4714,        # §3.1's shipped supervised selector
    "fan_oracle_ade_m": 0.1639,           # §3.1's fan oracle
    "fund_ratio_vs_ade": 1.7,             # σ/ADE ≤ 1.7 ⇒ FUNDED  (§5.2)
    "refuse_ratio_vs_ade": 3.0,           # σ/ADE ≥ 3.0 ⇒ REFUSED (§5.2)
    "sigma_funded_m": 1.7 * 0.4714,       # 0.80138 — "σ ≤ 0.80 m"
    "sigma_refused_m": 3.0 * 0.4714,      # 1.41420 — "σ ≤ 1.41 m to leave REFUSED"
    "verdict_horizon_s": 2.0,
    "report_only_horizon_s": 6.0,
    "fold_scheme": "LOEO (leave-one-EPISODE-out); the vocabulary is REBUILT per "
                   "fold from the TRAIN episodes only",
    "sigma_form": "per-axis isotropic 1-sigma, metres — sqrt(mean(|e|^2)/2), "
                  "the unit §3.1's requirement curve is stated in",
    "k_sweep": (8, 16, 32, 64, 128, 256),
    "methods": ("fps", "kmeans"),
    # --------------- the committed outcomes, both directions ---------------- #
    "E-AG1": {
        "question": "Can the ANCHOR_GOAL formulation reach the bar with a "
                    "PERFECT classifier? (a geometry-only bound)",
        "FLOOR_CLEARS": "σ_quant(K) ≤ 0.80 m at some K ≤ 256 ⇒ the formulation "
                        "has headroom; proceed to E-AG2 (the classification "
                        "gate). This does NOT fund ANCHOR_GOAL — it only fails "
                        "to refuse it.",
        "REFUSED": "σ_quant(K) ≥ 1.41 m for EVERY K ≤ 256 ⇒ ⛔ ANCHOR_GOAL as a "
                   "pure K-way selection over a ≤256 vocabulary is REFUSED on "
                   "geometry, with a perfect classifier and 0 GPU. The only "
                   "surviving form is anchor + CONTINUOUS RESIDUAL, which is "
                   "SEL-1's regression again and inherits its refusal.",
        "INCONCLUSIVE": "in between at every K ⇒ the K budget or the vocabulary "
                        "construction is the lever; re-run before any training.",
    },
    "E-AG2": {
        "question": "On the SAME frozen surface and folds, does discretising "
                    "into anchors beat regressing the endpoint freely?",
        "BETTER": "σ_anchor < σ_ridge, paired-separated ⇒ the discretisation is "
                  "a real lever and ANCHOR_GOAL supervision is worth building "
                  "even before S-W latents exist.",
        "WORSE_OR_FLAT": "σ_anchor ≥ σ_ridge, or the paired interval spans 0 ⇒ "
                         "⛔ discretisation buys nothing; the SURFACE is the "
                         "whole problem and ANCHOR_GOAL must be justified by "
                         "something other than σ (e.g. the tokens it unlocks).",
    },
    "E-AG3": {
        "question": "How large is the v0 admissibility contradiction?",
        "status": "⛔ PENDING_PI_ADJUDICATION — reported as the magnitude of a "
                  "contradiction between two live HEAD documents, NEVER as a "
                  "funded arm. e_wc2_sigma_star.py:187 classes v0 ECHO; "
                  "V6F_PLANNER_DESIGN.md §1.4 row 3 declares true v0 ✅.",
    },
    "⛔ the 6 s rule": "V6F §5.3's refutation check FIRED on 2026-08-16 "
                      "(σ(6 s) = 3.75 × σ(2 s) > 3 ⇒ REDERIVE). No 6 s "
                      "threshold exists and none may be scaled from the 2 s "
                      "one. This instrument therefore emits verdict = None at "
                      "6 s, under every branch, by construction.",
}

VERDICTS = {
    "FLOOR_CLEARS": PREREG["E-AG1"]["FLOOR_CLEARS"],
    "REFUSED": PREREG["E-AG1"]["REFUSED"],
    "INCONCLUSIVE": PREREG["E-AG1"]["INCONCLUSIVE"],
    "NO_VERDICT": "the pre-registered surface was not met — this instrument "
                  "refuses to emit a weak verdict. NOT the same as REFUSED.",
}


# --------------------------------------------------------------------------- #
# vocabularies                                                                 #
# --------------------------------------------------------------------------- #
def fps_anchors(pool: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """[M, 2] -> [k, 2] with the programme's OWN greedy FPS.

    ``tanitad.refs.refc.furthest_point_sample`` is imported rather than
    re-implemented: the anchor vocabularies actually shipped (`refc_anchor_vocab.pt`,
    `flagship_v4_anchors_dense.pt`) were built by that function via
    ``build_refc_anchors.py``, and a re-implementation here would measure a
    vocabulary the programme does not use.

    ⚠️ FPS is deliberately NOT density-adaptive — it spreads over the SUPPORT so
    the rare sharp-curve / hard-brake modes survive. For a quantiser judged by an
    RMS over the corpus that is the wrong objective, which is exactly why
    :func:`kmeans_anchors` is measured beside it rather than assumed equivalent.
    """
    from tanitad.refs.refc import furthest_point_sample
    t = torch.as_tensor(np.asarray(pool, dtype=np.float64),
                        dtype=torch.float32).reshape(-1, 1, 2)
    return (furthest_point_sample(t, int(k), seed=seed)
            .reshape(-1, 2).numpy().astype(np.float64))


def kmeans_anchors(pool: np.ndarray, k: int, seed: int = 0,
                   iters: int = 100, tol: float = 1e-9) -> np.ndarray:
    """[M, 2] -> [k, 2] by k-means++ init + Lloyd. Deterministic given ``seed``.

    ⭐ This is the vocabulary that MINIMISES the quantisation σ, because Lloyd's
    algorithm descends exactly ``mean ‖x − anchor(x)‖²`` — the statistic the
    admission bar is stated in. Its cost is the one ``build_refc_anchors.py``
    names: it concentrates on the dense straight mode and starves the turns. Both
    are reported so the trade-off is measured rather than asserted.
    """
    X = np.asarray(pool, dtype=np.float64)
    m = X.shape[0]
    if k > m:
        raise ValueError(f"cannot build {k} anchors from a pool of {m}")
    rng = np.random.default_rng(seed)
    cent = np.empty((k, 2), dtype=np.float64)
    cent[0] = X[rng.integers(m)]
    d2 = ((X - cent[0]) ** 2).sum(1)
    for i in range(1, k):
        tot = d2.sum()
        p = (np.full(m, 1.0 / m) if not np.isfinite(tot) or tot <= 0
             else d2 / tot)
        cent[i] = X[rng.choice(m, p=p)]
        d2 = np.minimum(d2, ((X - cent[i]) ** 2).sum(1))
    prev = np.inf
    for _ in range(int(iters)):
        ids = _assign_ids(X, cent)
        for j in range(k):
            sel = ids == j
            if sel.any():
                cent[j] = X[sel].mean(0)
        inertia = float(((X - cent[ids]) ** 2).sum())
        if prev - inertia <= tol:
            break
        prev = inertia
    return cent


def _assign_ids(points: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    d2 = ((np.asarray(points)[:, None, :]
           - np.asarray(anchors)[None, :, :]) ** 2).sum(-1)
    return d2.argmin(1)


def assign(points: np.ndarray, anchors: np.ndarray) -> dict:
    """Nearest-anchor assignment plus the two miss-cost references.

    ``resid`` is the ORACLE residual (a perfect classifier). ``resid_second`` is
    the runner-up's — a NEAR miss. ``msq_marginal`` is the mean squared distance
    to a UNIFORMLY RANDOM anchor — a FAR miss, and the same shape of null as the
    goal-echo control. Both are needed because the accuracy a scheme requires
    depends entirely on what a mistake costs.
    """
    P = np.asarray(points, dtype=np.float64)
    A = np.asarray(anchors, dtype=np.float64)
    d2 = ((P[:, None, :] - A[None, :, :]) ** 2).sum(-1)             # [M, K]
    order = np.argsort(d2, axis=1)
    ids = order[:, 0]
    second = order[:, 1] if A.shape[0] > 1 else order[:, 0]
    return {"ids": ids, "resid": P - A[ids], "resid_second": P - A[second],
            "msq_marginal": float(d2.mean()), "d2": d2}


# --------------------------------------------------------------------------- #
# the multi-target ridge — one SVD per fold, shared GCV λ                       #
# --------------------------------------------------------------------------- #
def ridge_multi(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray,
                lambdas: tuple[float, ...] = RIDGE_LAMBDAS
                ) -> tuple[np.ndarray, float]:
    """Closed-form multi-target ridge. Returns ``(Yhat_te [n_te, K], lambda)``.

    Identical algebra to ``probe_latent_state.RidgeSVD`` (one economy SVD of the
    centred design; ``w = V diag(s/(s²+λ)) Uᵀ y_c``), extended to K targets so a
    256-way one-hot classifier costs ONE factorisation per fold instead of 256.
    λ is chosen by GCV **summed over the targets, on the TRAIN fold only** —
    model selection never touches the held-out episode. Pinned equal to
    ``RidgeSVD`` on a single target by
    ``tests/test_e_ag1_anchor_floor.py::test_ridge_multi_matches_ridgesvd``.
    """
    Xtr_s, Xte_s = _standardize(np.asarray(Xtr, dtype=np.float64),
                                np.asarray(Xte, dtype=np.float64))
    Y = np.asarray(Ytr, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    n = Xtr_s.shape[0]
    mx = Xtr_s.mean(0)
    U, s, Vt = np.linalg.svd(Xtr_s - mx, full_matrices=False)
    my = Y.mean(0)
    Yc = Y - my
    UtY = U.T @ Yc                                                  # [r, K]
    yy = (Yc ** 2).sum(0)                                           # [K]
    best_lam, best_score = None, np.inf
    for lam in lambdas:
        shrink = s ** 2 / (s ** 2 + lam)
        resid = yy - ((2 * shrink - shrink ** 2)[:, None] * UtY ** 2).sum(0)
        df = float(shrink.sum())
        score = float(n * np.maximum(resid, 0.0).sum() / max(n - df, 1e-9) ** 2)
        if score < best_score:
            best_lam, best_score = float(lam), score
    shrink = s / (s ** 2 + best_lam)
    W = Vt.T @ (shrink[:, None] * UtY)                              # [d, K]
    b = my - mx @ W
    return Xte_s @ W + b, float(best_lam)


# --------------------------------------------------------------------------- #
# the arms                                                                     #
# --------------------------------------------------------------------------- #
def build_vocab(pool: np.ndarray, k: int, method: str, seed: int) -> np.ndarray:
    if method == "fps":
        return fps_anchors(pool, k, seed=seed)
    if method == "kmeans":
        return kmeans_anchors(pool, k, seed=seed)
    raise ValueError(f"unknown vocabulary method {method!r}")


def loeo_anchor_arms(gt: np.ndarray, eid, X: np.ndarray | None, k: int,
                     method: str, seed: int = 0) -> dict:
    """One LOEO pass. The vocabulary is REBUILT from the TRAIN episodes in every
    fold, so no held-out endpoint ever contributed to the anchors it is scored
    against — the anchor-set analogue of the LOEO leak E-WC2 measured at 2.06×.

    Returns per-window residuals for every arm, all on the SAME rows:
      ``oracle``   — E-AG1's floor (a perfect classifier)
      ``second``   — the runner-up anchor (a near miss)
      ``marginal`` — the goal-echo null: the vocabulary's own centroid
      ``clf``      — E-AG2's K-way linear classifier      (needs ``X``)
      ``ridge``    — the E-WC2 free-endpoint regression   (needs ``X``)
      ``snap``     — ``ridge`` snapped to the nearest anchor (needs ``X``)
    """
    folds = E.loeo_folds(eid)
    n = gt.shape[0]
    out = {k_: np.full((n, 2), np.nan) for k_ in
           ("oracle", "second", "marginal", "clf", "ridge", "snap")}
    top1 = np.full(n, np.nan)
    lam_clf, lam_ridge, msq_marg = [], [], []
    for f in sorted(set(folds.tolist())):
        te, tr = folds == f, folds != f
        if not te.any() or not tr.any():
            continue
        A = build_vocab(gt[tr], k, method, seed)
        a_te = assign(gt[te], A)
        out["oracle"][te] = a_te["resid"]
        out["second"][te] = a_te["resid_second"]
        out["marginal"][te] = gt[te] - A.mean(0)[None, :]
        msq_marg.append(a_te["msq_marginal"])
        if X is None:
            continue
        ids_tr = _assign_ids(gt[tr], A)
        Y = np.zeros((int(tr.sum()), A.shape[0]), dtype=np.float64)
        Y[np.arange(Y.shape[0]), ids_tr] = 1.0
        scores, lam = ridge_multi(X[tr], Y, X[te])
        lam_clf.append(lam)
        pred = scores.argmax(1)
        out["clf"][te] = gt[te] - A[pred]
        top1[te] = (pred == a_te["ids"]).astype(np.float64)
        yhat, lam2 = ridge_multi(X[tr], gt[tr], X[te])
        lam_ridge.append(lam2)
        out["ridge"][te] = gt[te] - yhat
        out["snap"][te] = gt[te] - A[_assign_ids(yhat, A)]
    return {"resid": out, "top1": top1, "folds": folds,
            "lambda_clf": lam_clf, "lambda_ridge": lam_ridge,
            "msq_marginal": float(np.mean(msq_marg)) if msq_marg else float("nan")}


def required_top1(sigma_target_m: float, msq_hit: float,
                  msq_miss: float) -> float | None:
    """The top-1 accuracy a K-way goal head needs to reach ``sigma_target_m``.

    ``E‖e‖² = p·msq_hit + (1−p)·msq_miss`` and ``σ² = E‖e‖²/2`` ⇒
    ``p = (2σ² − msq_miss) / (msq_hit − msq_miss)``.
    Returns ``None`` when the target is unreachable at any accuracy (i.e. the
    quantisation floor alone already exceeds it) — a ``None`` here IS the
    refusal, and it must never be rendered as 1.0.
    """
    denom = float(msq_hit) - float(msq_miss)
    if denom == 0.0:
        return None
    p = (2.0 * float(sigma_target_m) ** 2 - float(msq_miss)) / denom
    return None if (p > 1.0 or not np.isfinite(p)) else float(max(p, 0.0))


def decide_ag1(sigma_by_k: dict, prereg: dict = PREREG) -> dict:
    """E-AG1's committed three-sided verdict, over the whole K sweep."""
    vals = [v for v in sigma_by_k.values() if v is not None and np.isfinite(v)]
    if not vals:
        return {"verdict": "NO_VERDICT", "meaning": VERDICTS["NO_VERDICT"]}
    best = float(min(vals))
    if best <= prereg["sigma_funded_m"]:
        v = "FLOOR_CLEARS"
    elif best >= prereg["sigma_refused_m"]:
        v = "REFUSED"
    else:
        v = "INCONCLUSIVE"
    return {"verdict": v, "meaning": VERDICTS[v], "best_sigma_perax_m": best,
            "best_k": min((k for k, s in sigma_by_k.items()
                           if s is not None and float(s) == best), default=None),
            "sigma_funded_m": prereg["sigma_funded_m"],
            "sigma_refused_m": prereg["sigma_refused_m"]}


# --------------------------------------------------------------------------- #
# run                                                                          #
# --------------------------------------------------------------------------- #
def run(d, *, features: list[str], allow_echo: bool = False,
        declared: dict[str, str] | None = None, ks=PREREG["k_sweep"],
        methods=PREREG["methods"], steps=(20, 60), n_boot: int = 2000,
        seed: int = 0) -> dict:
    problems = E.validate_dump(d)
    X, fmeta = E.build_features(d, features, allow_echo=allow_echo,
                                declared=declared or {})
    eid_all = [int(x) for x in d["eid"]]
    rep = {
        "_what": "E-AG1/2/3 — the ANCHOR_GOAL quantisation floor, the "
                 "discretise-vs-regress test, and the v0 contradiction",
        "_tier": "⛔ T0-DIAGNOSTIC — a geometry/representation capacity probe on "
                 "banked latents. NOT a driving-performance number.",
        "_estimator": "full-set point estimates; episode-cluster bootstrap "
                      "intervals (taniteval/ci.py); paired for arm-vs-arm. "
                      "overlapping_holdout_se is used NOWHERE.",
        "_admissibility": {
            "features": fmeta,
            "situation_disjoint": "the label is argmin_k ‖endpoint − anchor_k‖ "
                                  "over FUTURE ego displacement; no situation "
                                  "detector output enters in any form. This "
                                  "module does not import tanitad.data.situations.",
            "echo_test": "at inference the goal point is a FROZEN buffer indexed "
                         "by a vision-derived logit; the label comes from future "
                         "ego poses, which no inference input is derived from. "
                         "The `marginal` arm is the goal-echo null.",
            "v0_status": PREREG["E-AG3"]["status"] if fmeta["any_echo"] else None,
        },
        "prereg": PREREG,
        "dump": {"ckpt": d.get("ckpt"), "ckpt_step": d.get("ckpt_step"),
                 "n_windows": len(eid_all),
                 "n_episodes": len(set(eid_all)),
                 "contract_problems": problems,
                 "instrument_fail": list(d.get("instrument_fail") or [])},
        "horizons": {},
    }
    ci = E._load_ci()
    for step in steps:
        tgt, valid, src = E.resolve_endpoint(d, int(step))
        if tgt is None:
            rep["horizons"][f"{step * DT:g}s"] = {"skipped": src}
            continue
        m = valid & np.isfinite(tgt).all(axis=1)
        gt, eid = tgt[m], [e for e, keep in zip(eid_all, m) if keep]
        Xm = X[m]
        h = {"horizon_s": step * DT, "n": int(m.sum()),
             "n_episodes": len(set(eid)), "target_source": src,
             "endpoint_excluded": int((~m).sum()),
             "endpoint_exclusion_reason":
                 "the horizon runs past the end of the episode "
                 "(endpoint_valid False). Excluded with n reported, NEVER imputed.",
             "vocabularies": {}}
        for method in methods:
            per_k, sigma_by_k = {}, {}
            for k in ks:
                arms = loeo_anchor_arms(gt, eid, Xm, int(k), method, seed=seed)
                row = {"k": int(k), "arms": {}}
                for name, res in arms["resid"].items():
                    if not np.isfinite(res).all():
                        continue
                    st = E.sigma_from_residuals(res)
                    sq = (res ** 2).sum(1)
                    st["ci95"] = E.sigma_ci(sq, eid, n_boot, seed=seed)
                    st["ratio_vs_ade"] = (st["sigma_perax_m"]
                                          / PREREG["incumbent_sel_ade_m"])
                    st["msq"] = float(sq.mean())
                    row["arms"][name] = st
                if np.isfinite(arms["top1"]).all():
                    row["clf_top1_acc"] = float(arms["top1"].mean())
                    row["lambda_clf"] = arms["lambda_clf"]
                    row["lambda_ridge"] = arms["lambda_ridge"]
                o, s2 = row["arms"].get("oracle"), row["arms"].get("second")
                if o is not None:
                    row["required_top1_for_0.80m"] = required_top1(
                        PREREG["sigma_funded_m"], o["msq"],
                        s2["msq"] if s2 else arms["msq_marginal"])
                    row["required_top1_for_1.41m"] = required_top1(
                        PREREG["sigma_refused_m"], o["msq"],
                        s2["msq"] if s2 else arms["msq_marginal"])
                    row["miss_cost_used"] = ("second-nearest anchor (a NEAR "
                                             "miss — the optimistic assumption)")
                    sigma_by_k[int(k)] = o["sigma_perax_m"]
                for a, b in (("clf", "ridge"), ("snap", "ridge"),
                             ("oracle", "ridge")):
                    ra, rb = arms["resid"][a], arms["resid"][b]
                    if np.isfinite(ra).all() and np.isfinite(rb).all():
                        row.setdefault("paired_vs_ridge", {})[a] = \
                            ci.paired_episode_cluster_bootstrap(
                                (ra ** 2).sum(1) / 2.0, (rb ** 2).sum(1) / 2.0,
                                eid, n_boot=n_boot, seed=seed, reduce="rms")
                per_k[int(k)] = row
            h["vocabularies"][method] = {
                "per_k": per_k,
                "verdict": (decide_ag1(sigma_by_k) if step == 20 else {
                    "verdict": None,
                    "meaning": PREREG["⛔ the 6 s rule"]}),
            }
        rep["horizons"][f"{step * DT:g}s"] = h
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", default="pooled,ctx")
    ap.add_argument("--declare", action="append", default=[])
    ap.add_argument("--allow-echo-features", action="store_true",
                    help="⛔ runs the PENDING_PI_ADJUDICATION v0 arm")
    ap.add_argument("--ks", default=",".join(str(k) for k in PREREG["k_sweep"]))
    ap.add_argument("--methods", default=",".join(PREREG["methods"]))
    ap.add_argument("--steps", default="20,60")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    d = torch.load(a.dump, map_location="cpu", weights_only=False)
    rep = run(d, features=[s for s in a.features.split(",") if s],
              allow_echo=a.allow_echo_features,
              declared=E._parse_declared(a.declare),
              ks=tuple(int(x) for x in a.ks.split(",")),
              methods=tuple(a.methods.split(",")),
              steps=tuple(int(x) for x in a.steps.split(",")),
              n_boot=a.n_boot, seed=a.seed)
    rep["dump"]["path"] = str(a.dump)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False),
                           encoding="utf-8")
    for hz, h in rep["horizons"].items():
        if "skipped" in h:
            continue
        for meth, mv in h["vocabularies"].items():
            v = mv["verdict"]
            print(f"{hz} {meth}: verdict={v.get('verdict')} "
                  f"best_sigma={v.get('best_sigma_perax_m')} "
                  f"K={v.get('best_k')} n={h['n']}")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
