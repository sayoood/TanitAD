#!/usr/bin/env python3
"""PER-STRATUM P7 calibration — the read the T3 gate row asks for and pooled P7
cannot give.

⛔ WHY THIS MODULE EXISTS. The F-9 interaction-curriculum cell (catalog row T3)
carries the gate

    "P7 calibration rho >= 0.3 held on **interaction-rich strata, not just
     pooled**"

and its own deliverable records that the row is **NOT COMPUTABLE** with what
existed: `stack/scripts/w7_roll_rerank.py` holds P7 (`P7_GATE_RHO = 0.3`) and has
**no stratification support at all** (MEASURED, two probes -- `grep strat` over
that file, and a repo-wide sweep across `stack/` and `taniteval/`). A T3 arm could
therefore be trained and still not be gradeable. This module is that read.

⭐ WHAT P7 IS, so the strata do not quietly change the question. P7 asks whether a
proposal fan's **spread** (selector entropy, prob-weighted endpoint dispersion)
ranks the **realised error** of the candidate the arm actually selected. It is a
Spearman rho between two per-window scalars. Stratifying means computing that rho
**inside** window subsets, never pooling them.

⛔ THE ADMISSIBILITY RULE THE STRATA MUST SATISFY -- and it is not a formality.
Sayed's 2026-08-03 ruling keeps the goal path and the situation path
information-disjoint, and `stack/tanitad/data/situations.py` derives the situation
labels from **ego dynamics**. So:

  * ⛔ a stratum cut on **ego state** (speed, accel, yaw-rate, curvature) is partly
    a stratum cut on the situation label's OWN SOURCE. Inadmissible by default.
  * ⛔ a stratum cut on the **arm's own occupancy prediction** (T3's training score
    descends from the P8 decoder) would let the thing being graded choose the
    strata that grade it. Inadmissible by default.
  * ✅ a stratum cut on **`obstacle.offline`** -- the dataset's own 3D agent
    cuboids -- is an EXTERNAL annotation of other traffic. It is not computed from
    ego dynamics and not computed from any model output, so it can carry an
    "interaction-rich" claim without circularity. `O4`'s own docstring already
    names the obstacle join *"frozen-probe/eval-strata material"*; this is exactly
    that use.

:func:`assert_stratifier_admissible` makes the declaration mandatory and REFUSES
the two inadmissible classes unless a caller passes an explicit written override,
so an ego-cut stratification cannot arrive by accident.

⚠️ THREE STATES, NEVER TWO (inherited from `taniteval.lead_source`). A window is
``LEAD`` / ``NO_LEAD`` / ``NO_LABEL``. Collapsing ``NO_LABEL`` into ``NO_LEAD``
manufactures free-flow: `obstacle.offline` spans ~20 s while `egomotion` runs
20-140 s, so most of a long clip is unlabelled. ``NO_LABEL`` is reported as its own
row and is **never** part of an interaction-rich verdict.

⚠️ A DISPERSION IS NOT A CONFIDENCE INTERVAL (C109). Every bracket this module
emits carries ``bracket_kind``; the only interval that may decide the gate is
``episode_cluster_bootstrap_percentile_95``.

⚠️ A COUNT IS A CLAIM ABOUT THE FILTER (C110). Every stratum row carries its
``inclusion_rule`` verbatim next to its ``n``.

Estimator: episode-cluster bootstrap over the eval episodes, resampled with
:func:`taniteval.ci._draws` -- the programme's single resampler implementation -- so
this module adds only the Spearman statistic. ⛔ ``overlapping_holdout_se`` is never
used: it biases the point estimate bidirectionally, up to a sign flip.

Tier: this reads a **teacher-forced open-loop fan dump** (`rollout.collect` feeds
the expert's true future actions), so every number here is stamped **T0** -- a WM /
instrument diagnostic, NEVER a driving claim (EVAL_DOCTRINE).
"""
from __future__ import annotations

import numpy as np

from taniteval.ci import _draws, episode_index

__all__ = [
    "P7_GATE_RHO", "MIN_N_WINDOWS", "MIN_N_EPISODES", "DEFAULT_GAP_EDGES_M",
    "STRATIFIER_KIND_LABEL", "STRATIFIER_KIND_EGO", "STRATIFIER_KIND_MODEL",
    "spearman", "cluster_bootstrap_spearman", "assert_stratifier_admissible",
    "lead_state_strata", "p7_per_stratum", "permutation_null", "arm_controls",
    "p7_strata_report",
]

#: P7's pre-registered gate. Duplicated as DATA from ``w7_roll_rerank.P7_GATE_RHO``
#: (taniteval must not import a pod-side script); ``tests/test_p7_strata.py``
#: asserts the two are equal, the same pinning `v6.py` uses for the same constant.
P7_GATE_RHO = 0.3

#: A stratum below either floor gets NO rho. Refusing is the point: a rho on 11
#: windows from 3 episodes is not evidence, and printing it invites a gate verdict
#: the data cannot support. 32 mirrors the programme's existing control floor
#: (`v6.T3_CONTROL_MIN_N`); the episode floor exists because the estimator's
#: resampling unit is the EPISODE, not the window.
MIN_N_WINDOWS = 32
MIN_N_EPISODES = 8

#: Proximity bands for the LEAD strata, metres, rig-origin to the lead's rear face
#: (`lead_source`'s ``gap`` convention). FIXED thresholds, not quantiles, so the
#: same band means the same traffic situation across arms and across corpora.
DEFAULT_GAP_EDGES_M = (20.0, 40.0)

STRATIFIER_KIND_LABEL = "external_label"    # ✅ admissible
STRATIFIER_KIND_EGO = "ego_derived"         # ⛔ refused by default
STRATIFIER_KIND_MODEL = "model_derived"     # ⛔ refused by default

_ADMISSIBLE_KINDS = (STRATIFIER_KIND_LABEL,)
_REFUSED_KINDS = {
    STRATIFIER_KIND_EGO: (
        "an ego-derived stratifier cuts on the SAME quantity the situation labels "
        "are derived from (stack/tanitad/data/situations.py), so the stratum is "
        "partly defined by the label's own source -- Sayed 2026-08-03"),
    STRATIFIER_KIND_MODEL: (
        "a model-derived stratifier lets the arm being graded choose the strata "
        "that grade it; T3's own score descends from the P8 decoder, which is "
        "why O4 admits the obstacle join as EVAL-STRATA material and not as a "
        "training-time selector"),
}


# ============================================================================
# the statistic (rank convention pinned to tools_p7_calibration.py:15-19)
# ============================================================================
def spearman(a, b) -> float:
    """Spearman rho by double argsort -- byte-for-byte the convention of
    ``tools_p7_calibration.py:15-19`` and ``w7_roll_rerank.spearman``, including
    the constant-input guard (a constant vector has no rank order, so it returns
    ``nan`` rather than letting a stable argsort mint ranks out of memory order).
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape or a.size < 2:
        raise ValueError(f"need two equal-length vectors of n >= 2, got "
                         f"{a.shape} vs {b.shape}")
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom == 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def cluster_bootstrap_spearman(x, y, eid, *, n_boot: int = 2000,
                               seed: int = 0) -> dict:
    """Episode-cluster bootstrap percentile-95 interval on Spearman rho.

    Episodes are the resampling unit (windows travel with their episode) and the
    draws come from :func:`taniteval.ci._draws`, so this module does not add a
    second resampler to the programme. Degenerate replicates (constant x or y in
    the draw) yield ``nan`` and are counted, not silently averaged away.

    ⛔ Returns ``rho_ci_cluster = None`` with a stated reason rather than a fake
    interval when fewer than two episodes are present.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    eid = np.asarray([str(e) for e in np.asarray(eid).ravel()])
    if not (x.shape == y.shape == eid.shape):
        raise ValueError(f"x/y/eid must match, got {x.shape}/{y.shape}/"
                         f"{eid.shape}")
    uniq, idx_by_ep = episode_index(eid)
    out = {
        "spearman_rho": round(spearman(x, y), 4),
        "n": int(x.size), "n_episodes": int(uniq.size),
        "n_boot": int(n_boot), "seed": int(seed),
        "estimator": "episode_cluster_bootstrap",
        "bracket_kind": "episode_cluster_bootstrap_percentile_95",
        "rank_convention": ("tools_p7_calibration.py:15-19 (double argsort, no "
                            "tie averaging)"),
        "resampler": "taniteval.ci._draws",
    }
    if uniq.size < 2:
        out["rho_ci_cluster"] = None
        out["ci_note"] = (f"only {uniq.size} episode(s) -- a cluster bootstrap "
                          f"needs >= 2 resampling units; interval NOT computable, "
                          f"stated rather than faked")
        return out
    rhos, n_degen = [], 0
    for rows in _draws(uniq, idx_by_ep, int(n_boot), int(seed)):
        r = spearman(x[rows], y[rows])
        if np.isnan(r):
            n_degen += 1
        else:
            rhos.append(r)
    if not rhos:
        out["rho_ci_cluster"] = None
        out["ci_note"] = "every bootstrap replicate degenerate"
        out["n_boot_degenerate"] = n_degen
        return out
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    out["rho_ci_cluster"] = [round(float(lo), 4), round(float(hi), 4)]
    out["n_boot_degenerate"] = int(n_degen)
    return out


# ============================================================================
# stratifier admissibility -- a declaration, enforced
# ============================================================================
def assert_stratifier_admissible(spec: dict, *, override_reason: str | None = None
                                 ) -> dict:
    """Validate a stratifier declaration and REFUSE the inadmissible classes.

    ``spec`` must carry ``name``, ``kind`` (one of the ``STRATIFIER_KIND_*``),
    ``derived_from`` (the concrete channel, e.g. ``"obstacle.offline"``) and
    ``why_admissible``. The point is that a stratification cannot arrive without
    stating what it is computed from -- which is the question the T3 gate turns on.

    ``override_reason`` is the only way to run an ego- or model-derived cut, and it
    is written into the returned record so the console cannot lose it.
    """
    if not isinstance(spec, dict):
        raise TypeError(f"stratifier spec must be a dict, got {type(spec)!r}")
    missing = [k for k in ("name", "kind", "derived_from", "why_admissible")
               if not spec.get(k)]
    if missing:
        raise ValueError(
            f"stratifier declaration is missing {missing}. A stratum that does "
            f"not say what it is computed from cannot be adjudicated -- state "
            f"the channel and why it is not the label's own source.")
    kind = spec["kind"]
    if kind not in _ADMISSIBLE_KINDS and kind not in _REFUSED_KINDS:
        raise ValueError(f"unknown stratifier kind {kind!r}; expected one of "
                         f"{sorted(set(_ADMISSIBLE_KINDS) | set(_REFUSED_KINDS))}")
    rec = dict(spec)
    rec["admissible"] = kind in _ADMISSIBLE_KINDS
    if not rec["admissible"]:
        if not override_reason:
            raise ValueError(
                f"REFUSING stratifier {spec['name']!r} of kind {kind!r}: "
                f"{_REFUSED_KINDS[kind]}. Pass override_reason=... in writing if "
                f"this is deliberate; it will be stamped into the report.")
        rec["override_reason"] = override_reason
        rec["admissible"] = False
    return rec


# ============================================================================
# the strata themselves
# ============================================================================
def lead_state_strata(state, gap_m, *, edges=DEFAULT_GAP_EDGES_M
                      ) -> tuple[np.ndarray, dict]:
    """`obstacle.offline` lead state + proximity -> per-window stratum labels.

    Strata (exhaustive, mutually exclusive, and every one carries its rule):

      ``NO_LABEL``     no `obstacle.offline` for this clip, or t0 outside the
                       labelled span. ⛔ NOT free-flow -- reported, never counted
                       toward an interaction verdict.
      ``NO_LEAD``      labels present, no causal in-corridor vehicle ahead.
                       This is the free-flow reference.
      ``LEAD_le20m``   a lead at gap < 20 m  -- interaction-rich.
      ``LEAD_20_40m``  a lead at 20 <= gap < 40 m -- interaction-rich.
      ``LEAD_ge40m``   a lead at gap >= 40 m -- interaction-rich (far following).

    ⚠️ ``gap_m`` is finite exactly on ``LEAD`` windows by construction; a ``LEAD``
    window with a non-finite gap is a build defect and raises rather than being
    swept into a band.
    """
    state = np.asarray([str(s) for s in np.asarray(state).ravel()])
    gap = np.asarray(gap_m, dtype=np.float64).ravel()
    if state.shape != gap.shape:
        raise ValueError(f"state/gap_m must match, got {state.shape} vs "
                         f"{gap.shape}")
    lo, hi = (float(edges[0]), float(edges[1]))
    if not (lo < hi):
        raise ValueError(f"gap edges must be increasing, got {edges}")
    known = {"LEAD", "NO_LEAD", "NO_LABEL"}
    bad = sorted(set(state.tolist()) - known)
    if bad:
        raise ValueError(f"unknown lead states {bad}; expected {sorted(known)} "
                         f"(taniteval.lead_source's three states)")
    is_lead = state == "LEAD"
    if is_lead.any() and not np.isfinite(gap[is_lead]).all():
        raise ValueError(
            f"{int((~np.isfinite(gap[is_lead])).sum())} LEAD window(s) carry a "
            f"non-finite gap -- that is a lead-block build defect, not a band")
    labels = np.array(state, dtype=object)
    labels[is_lead & (gap < lo)] = f"LEAD_le{lo:g}m"
    labels[is_lead & (gap >= lo) & (gap < hi)] = f"LEAD_{lo:g}_{hi:g}m"
    labels[is_lead & (gap >= hi)] = f"LEAD_ge{hi:g}m"
    labels = labels.astype(str)
    spec = {
        "stratum_order": ["NO_LABEL", "NO_LEAD", f"LEAD_le{lo:g}m",
                          f"LEAD_{lo:g}_{hi:g}m", f"LEAD_ge{hi:g}m"],
        "inclusion_rule": {
            "NO_LABEL": ("state == NO_LABEL: no obstacle.offline for the clip, or "
                         "t0 outside the ~20 s labelled span. NOT free-flow."),
            "NO_LEAD": ("state == NO_LEAD: obstacle.offline present AND no causal "
                        "in-corridor vehicle ahead within 80 m / +-2 m lateral."),
            f"LEAD_le{lo:g}m": f"state == LEAD AND gap_m < {lo:g}",
            f"LEAD_{lo:g}_{hi:g}m": f"state == LEAD AND {lo:g} <= gap_m < {hi:g}",
            f"LEAD_ge{hi:g}m": f"state == LEAD AND gap_m >= {hi:g}",
        },
        "interaction_rich": {
            "NO_LABEL": None, "NO_LEAD": False,
            f"LEAD_le{lo:g}m": True, f"LEAD_{lo:g}_{hi:g}m": True,
            f"LEAD_ge{hi:g}m": True,
        },
        "gap_convention": ("lead_source: gap = along - size_x/2, rig origin to "
                           "the lead's REAR face; rig frame x-forward/y-left"),
        "edges_m": [lo, hi],
    }
    return labels, spec


# ============================================================================
# the read
# ============================================================================
def p7_per_stratum(spread, err, eid, strata, *, stratifier: dict,
                   stratum_spec: dict | None = None,
                   min_n: int = MIN_N_WINDOWS, min_eps: int = MIN_N_EPISODES,
                   n_boot: int = 2000, seed: int = 0, tier: str = "T0",
                   override_reason: str | None = None) -> dict:
    """P7 rho + episode-cluster interval **inside every stratum**.

    ⛔ There is deliberately no "pooled only" mode. ``pooled`` IS returned, because
    the gate is *"held on interaction-rich strata, **not just pooled**"* and the
    reader needs both -- but it is stamped ``is_gate_read: False`` so it cannot be
    quoted as the T3 row.

    A stratum below ``min_n`` windows or ``min_eps`` episodes returns
    ``status = "REFUSED_MIN_N"`` with its counts and no rho at all.
    """
    spread = np.asarray(spread, dtype=np.float64).ravel()
    err = np.asarray(err, dtype=np.float64).ravel()
    eid = np.asarray([str(e) for e in np.asarray(eid).ravel()])
    strata = np.asarray([str(s) for s in np.asarray(strata).ravel()])
    if not (spread.shape == err.shape == eid.shape == strata.shape):
        raise ValueError(f"spread/err/eid/strata must match, got {spread.shape}/"
                         f"{err.shape}/{eid.shape}/{strata.shape}")
    decl = assert_stratifier_admissible(stratifier,
                                        override_reason=override_reason)
    spec = dict(stratum_spec or {})
    order = spec.get("stratum_order") or sorted(set(strata.tolist()))
    order = [s for s in order if s in set(strata.tolist())] + \
            [s for s in sorted(set(strata.tolist())) if s not in order]
    rich = spec.get("interaction_rich", {})
    rules = spec.get("inclusion_rule", {})

    rows: dict[str, dict] = {}
    for s in order:
        m = strata == s
        n = int(m.sum())
        n_ep = int(np.unique(eid[m]).size) if n else 0
        row = {
            "n": n, "n_episodes": n_ep,
            "inclusion_rule": rules.get(s, f"stratum label == {s!r}"),
            "interaction_rich": rich.get(s, None),
            "frac_of_windows": round(n / max(spread.size, 1), 4),
        }
        if n < min_n or n_ep < min_eps:
            row["status"] = "REFUSED_MIN_N"
            row["reason"] = (
                f"n={n} windows over {n_ep} episode(s) is below the floor "
                f"(min_n={min_n}, min_episodes={min_eps}); a rho here would not "
                f"be evidence, so none is reported")
            rows[s] = row
            continue
        cal = cluster_bootstrap_spearman(spread[m], err[m], eid[m],
                                         n_boot=n_boot, seed=seed)
        ci = cal.get("rho_ci_cluster")
        row.update(cal)
        row["status"] = "OK"
        row["gate_pass"] = bool(
            np.isfinite(cal["spearman_rho"]) and ci is not None
            and cal["spearman_rho"] >= P7_GATE_RHO and ci[0] > 0.0)
        row["gate"] = (f"P7 rho >= {P7_GATE_RHO} with the episode-cluster "
                       f"interval excluding 0")
        rows[s] = row

    pooled = cluster_bootstrap_spearman(spread, err, eid, n_boot=n_boot,
                                        seed=seed)
    pooled_ci = pooled.get("rho_ci_cluster")
    pooled.update({
        "is_gate_read": False,
        "note": ("POOLED -- reported for contrast only. The T3 row is 'held on "
                 "interaction-rich strata, NOT just pooled', so this number "
                 "cannot decide it."),
        "gate_pass_if_it_were_the_gate": bool(
            np.isfinite(pooled["spearman_rho"]) and pooled_ci is not None
            and pooled["spearman_rho"] >= P7_GATE_RHO and pooled_ci[0] > 0.0),
    })

    rich_rows = {s: r for s, r in rows.items() if rich.get(s) is True}
    reportable = {s: r for s, r in rich_rows.items() if r.get("status") == "OK"}
    refused = sorted(s for s, r in rich_rows.items()
                     if r.get("status") == "REFUSED_MIN_N")
    if not rich_rows:
        verdict, why = "NOT_APPLICABLE", (
            "no stratum is declared interaction-rich; the T3 row has nothing to "
            "hold on")
    elif not reportable:
        verdict, why = "NOT_COMPUTABLE", (
            f"every interaction-rich stratum is below the min-n floor "
            f"({refused}); the gate cannot be adjudicated on this data")
    elif all(r["gate_pass"] for r in reportable.values()):
        verdict = "PASS" if not refused else "PASS_PARTIAL"
        why = (f"rho >= {P7_GATE_RHO} with the interval excluding 0 in all "
               f"{len(reportable)} reportable interaction-rich strata"
               + (f"; {len(refused)} further stratum/strata refused for min-n "
                  f"({refused}) so coverage is incomplete" if refused else ""))
    else:
        failed = sorted(s for s, r in reportable.items() if not r["gate_pass"])
        verdict, why = "FAIL", (
            f"interaction-rich stratum/strata {failed} do not hold rho >= "
            f"{P7_GATE_RHO} with the interval excluding 0")

    return {
        "block": "taniteval.p7_strata",
        "version": "1.0.0",
        "tier": tier,
        "tier_note": ("T0 = teacher-forced / true-future-conditioned. A WM and "
                      "instrument diagnostic, NEVER a driving claim "
                      "(EVAL_DOCTRINE)."),
        "gate": (f"T3: P7 calibration rho >= {P7_GATE_RHO} held on "
                 f"interaction-rich strata, not just pooled"),
        "estimator": "episode_cluster_bootstrap",
        "estimator_note": ("overlapping_holdout_se is NEVER used here: it biases "
                           "the point estimate bidirectionally, up to a sign flip"),
        "stratifier": decl,
        "stratum_spec": spec,
        "min_n_windows": int(min_n), "min_n_episodes": int(min_eps),
        "n_windows_total": int(spread.size),
        "n_episodes_total": int(np.unique(eid).size),
        "strata": rows,
        "pooled": pooled,
        "verdict": verdict,
        "verdict_reason": why,
    }


# ============================================================================
# controls -- PER ARM, never per study (C107)
# ============================================================================
def permutation_null(spread, err, eid, strata, *, min_n: int = MIN_N_WINDOWS,
                     min_eps: int = MIN_N_EPISODES, n_perm: int = 500,
                     seed: int = 0) -> dict:
    """Within-stratum permutation NULL for the observed rho.

    ⚠️ THIS REPLACED A CONTROL THAT WAS WRONG, and the mistake is worth keeping
    visible. The obvious permutation control -- shuffle the spread once, then run
    the same cluster bootstrap on it -- puts an interval **around that one
    shuffle's rho**, which for a single draw can sit anywhere in the null's own
    spread. MEASURED on real windows: one shuffle of `refc-xl-30k`'s entropy gave
    rho +0.1998 with a bootstrap interval of [0.0133, 0.4009], i.e. "significant"
    -- from pure noise. The interval was not wrong; the QUESTION was. A null is a
    DISTRIBUTION over shuffles, not one shuffle plus an interval.

    ⚠️ The percentiles returned here are the NULL's own spread. They are labelled
    ``permutation_null_dispersion_not_a_ci`` and may never be read as a confidence
    interval (C109).
    """
    spread = np.asarray(spread, dtype=np.float64).ravel()
    err = np.asarray(err, dtype=np.float64).ravel()
    eid = np.asarray([str(e) for e in np.asarray(eid).ravel()])
    strata = np.asarray([str(s) for s in np.asarray(strata).ravel()])
    rng = np.random.default_rng(seed + 993)
    rows: dict[str, dict] = {}
    for s in np.unique(strata):
        m = strata == s
        n, n_ep = int(m.sum()), int(np.unique(eid[m]).size)
        if n < min_n or n_ep < min_eps:
            rows[str(s)] = {"n": n, "n_episodes": n_ep,
                            "status": "REFUSED_MIN_N"}
            continue
        x, y = spread[m], err[m]
        obs = spearman(x, y)
        null = np.array([spearman(rng.permutation(x), y)
                         for _ in range(int(n_perm))])
        null = null[np.isfinite(null)]
        lo, hi = (np.percentile(null, [2.5, 97.5]) if null.size
                  else (np.nan, np.nan))
        rows[str(s)] = {
            "n": n, "n_episodes": n_ep, "status": "OK",
            "rho_observed": round(float(obs), 4),
            "n_perm": int(null.size),
            "null_median": round(float(np.median(null)), 4) if null.size else None,
            "null_p2p5_p97p5": [round(float(lo), 4), round(float(hi), 4)],
            "bracket_kind": "permutation_null_dispersion_not_a_ci",
            "p_two_sided": (round(float((np.abs(null) >= abs(obs)).mean()), 4)
                            if null.size and np.isfinite(obs) else None),
            "null_centred_on_zero": (bool(abs(float(np.median(null))) < 0.05)
                                     if null.size else None),
        }
    return {
        "what": ("the arm's own spread shuffled WITHIN each stratum, "
                 "n_perm times, against the unshuffled error"),
        "must": ("the null is centred on 0 and the observed rho sits in its "
                 "tail wherever the gate is claimed"),
        "bracket_kind": "permutation_null_dispersion_not_a_ci",
        "seed": int(seed), "n_perm": int(n_perm),
        "strata": rows,
    }


def arm_controls(spread, err, eid, strata, *, stratifier: dict,
                 stratum_spec: dict | None = None, trivial_proxy=None,
                 trivial_proxy_name: str = "trivial_proxy",
                 trivial_proxy_note: str = "", min_n: int = MIN_N_WINDOWS,
                 min_eps: int = MIN_N_EPISODES, n_boot: int = 2000,
                 n_perm: int = 500, seed: int = 0) -> dict:
    """The three controls every P7-per-stratum read must carry, computed **for
    this arm on these windows** -- not once for the study.

    ``positive``      an oracle spread (the realised error plus seeded noise).
                      The instrument MUST detect calibration here in every
                      stratum it reports; a stratum where the positive control
                      fails is a stratum where a negative result means "not
                      enough data", not "not calibrated".
    ``trivial_proxy`` the caller's freely-available scalar -- for P7 the obvious
                      one is EGO SPEED at t0. If it reaches the gate, the fan's
                      spread has demonstrated nothing beyond a number the arm did
                      not have to compute.
    ``constant``      a constant "spread". MUST come back refused/nan; a rho here
                      would mean the rank convention is minting order out of
                      memory layout.
    ``permutation_null`` the null DISTRIBUTION over within-stratum shuffles --
                      see :func:`permutation_null` for why a single shuffle plus
                      an interval is the wrong control and what it produced.

    ⭐ ``positive`` is a function of (err, eid, strata) only, so it is identical
    across spread measures for the same arm. That is intended: it answers "does
    this stratum have the power to show calibration if calibration is there",
    which is a property of the STRATUM, not of the measure.
    """
    err = np.asarray(err, dtype=np.float64).ravel()
    spread = np.asarray(spread, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed + 991)
    scale = float(np.std(err)) or 1.0
    positive = err + rng.normal(0.0, 0.5 * scale, size=err.shape)

    def _run(x, tier="T0"):
        return p7_per_stratum(x, err, eid, strata, stratifier=stratifier,
                              stratum_spec=stratum_spec, min_n=min_n,
                              min_eps=min_eps, n_boot=n_boot, seed=seed,
                              tier=tier)

    out = {
        "scope": "PER ARM, on these windows (C107: a study-level control leaves "
                 "rows uncovered)",
        "positive": {
            "what": "realised error + N(0, 0.5*sd(err)), seeded",
            "must": "detect calibration in every stratum reported for this arm",
            "result": _run(positive),
        },
        "constant": {
            "what": "a constant spread",
            "must": "return nan / no gate pass in every stratum",
            "result": _run(np.zeros_like(err) + 1.0),
        },
        "permutation_null": permutation_null(
            spread, err, eid, strata, min_n=min_n, min_eps=min_eps,
            n_perm=n_perm, seed=seed),
    }
    if trivial_proxy is not None:
        out["trivial_proxy"] = {
            "what": trivial_proxy_name,
            "note": trivial_proxy_note,
            "must": ("NOT reach the gate -- if a freely-available scalar ranks "
                     "the error as well as the fan's spread, the fan's spread is "
                     "not what is being demonstrated"),
            "result": _run(np.asarray(trivial_proxy, dtype=np.float64).ravel()),
        }
    return out


def p7_strata_report(arms: dict, *, provenance: dict) -> dict:
    """Wrap per-arm ``{"read": ..., "controls": ...}`` blocks with provenance.

    Kept trivial on purpose: the arithmetic lives in the two functions above so a
    reader can check the gate read without reading a driver.
    """
    return {
        "block": "taniteval.p7_strata/report",
        "version": "1.0.0",
        "gate": (f"T3: P7 calibration rho >= {P7_GATE_RHO} held on "
                 f"interaction-rich strata, not just pooled"),
        "provenance": dict(provenance),
        "arms": dict(arms),
    }
