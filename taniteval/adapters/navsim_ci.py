#!/usr/bin/env python3
"""NavSim LOG-CLUSTER BOOTSTRAP — the settled interval for NavSim EPDMS/PDMS.

Pre-registered in
``FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-estimator-and-route-leak/SPEC.md``
(E3, 2026-09-19) — read it before changing any constant here. This module is the
implementation of that SPEC and nothing else.

WHY A NEW ESTIMATOR AND NOT ``taniteval.ci`` AS IS
--------------------------------------------------
``taniteval.ci.episode_cluster_bootstrap`` is the right MACHINE (resample
independent clusters, recompute the statistic, percentile interval) and it is
REUSED here verbatim. What it cannot know is NavSim's two facts:

1. **The independent unit is the OpenScene LOG (``log_name``), not the scene
   token.** MEASURED in the E3 census: scenes inside a log are sliding windows at
   a stride of ONE key frame, so consecutive navtest scenes share 13 of their 14
   frames; a scene-token bootstrap treats those as independent draws and
   manufactures a narrow interval — the ``overlapping_holdout_se`` class on a
   borrowed benchmark. Logs share no frame (MEASURED: 0 tokens shared, 0
   time-overlapping segment pairs).
2. **The statistic is not a mean of scene scores.** Two-stage EPDMS
   (navhard / warmup / private_test_hard) is the mean over MAPPING KEYS
   ``(orig_token, prev_token)`` of
   ``( s1(orig)·wavg(stage-2 "now") + s1(prev)·wavg(stage-2 "prev") ) / 2``
   (``run_pdm_score.py::calculate_individual_mapping_scores``, L242-290
   @0a380a9); single-stage (navtest) is the plain ``nanmean`` over tokens
   (``run_pdm_score_one_stage.py`` L293). A bootstrap of the wrong aggregate is
   precise about the wrong thing. So each unit's CONTRIBUTION is computed first,
   exactly as the devkit computes it, and the official aggregate over a set of
   units is then ``nanmean(contributions)`` — which is what makes a resample of
   whole logs reproduce the official aggregation on every draw.

WHAT THE INTERVAL ANSWERS — AND THE TWO QUESTIONS IT DOES NOT
-------------------------------------------------------------
It answers *"would another draw of LOGS like these give this score?"* — the
evaluation-set variance. It is structurally BLIND to training variance
(``H-ESTIM-SEED-1``: a same-flags replicate cleared a paired bootstrap on 14.3 %
of cells) and to inference variance of a sampling planner (the refav1 seed floor
≈0.30 m). A separated paired interval between two TRAINED arms is therefore
NECESSARY, NOT SUFFICIENT; the record says so in ``question_answered``.

⛔ ``overlapping_holdout_se`` is not a fallback here, ever.
"""
from __future__ import annotations

import math
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/taniteval/adapters
_TE_PARENT = os.path.dirname(_HERE)                          # <repo>/taniteval
if _TE_PARENT not in sys.path:
    sys.path.insert(0, _TE_PARENT)

from taniteval import ci as _ci                              # noqa: E402  REUSED, not re-implemented

SPEC_DOC = ("FlyWheels/TanitAD_EvalFlyWheel/incoming/"
            "2026-09-19-navsim-estimator-and-route-leak/SPEC.md")
CENSUS = ("FlyWheels/TanitAD_EvalFlyWheel/incoming/"
          "2026-09-19-navsim-estimator-and-route-leak/raw/split_census.json")
DEVKIT_PIN = "autonomousvision/navsim@0a380a9 (2025-10-27)"

ESTIMATOR = "navsim_log_cluster_bootstrap"
PAIRED_ESTIMATOR = "paired_navsim_log_cluster_bootstrap"
CLUSTER_UNIT = "log_name"
#: The finer units the SPEC forbids as a resampling unit, by name.
FORBIDDEN_UNITS = ("scene_token", "token", "initial_token", "mapping_key", "frame")
#: SPEC §3: the coarser SENSITIVITY unit, computed beside the primary. A segment
#: name is a nuPlan drive + ``_NNNNN_NNNNN``; a name without that suffix is its own drive.
SENSITIVITY_UNIT = "nuplan_drive"
_DRIVE_RE = re.compile(
    r"^(?P<drive>\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}_veh-\d+)_\d{5}_\d{5}$")


def drive_of(log_name: str) -> str:
    """``2021.05.25.14.16.10_veh-35_00083_00485`` -> ``2021.05.25.14.16.10_veh-35``."""
    m = _DRIVE_RE.match(str(log_name))
    return m.group("drive") if m else str(log_name)

#: RG-14's floor (``tools/release_gate.py`` EPISODE_CLUSTER_FLOOR = 8), restated
#: here for the NavSim unit; ``test_navsim_ci.py`` pins the two equal, and the
#: registry's ``GATE_estimator_cluster_unit.min_clusters`` to both.
MIN_CLUSTERS = 8
N_BOOT = 2000
ALPHA = 0.05

AGG_TWO_STAGE = "two_stage_mapping_key_mean"
AGG_SINGLE_STAGE = "single_stage_token_mean"
AGGREGATIONS = {
    AGG_TWO_STAGE: ("run_pdm_score.py::calculate_individual_mapping_scores L242-290 "
                    "@0a380a9 — per key ((s1(orig)*wavg(now)) + (s1(prev)*wavg(prev)))/2, "
                    "then pd.DataFrame(...).mean() (skipna) over keys"),
    AGG_SINGLE_STAGE: ("run_pdm_score_one_stage.py L293 @0a380a9 — "
                       "pdm_score_df[score_cols].mean(skipna=True) over tokens"),
}
#: The protocol -> aggregation map the SPEC registers (closed set of the registry's
#: GATE_no_cross_protocol_comparison).
PROTOCOL_AGGREGATION = {
    "EPDMS_v2_navhard_two_stage": AGG_TWO_STAGE,
    "EPDMS_v2_warmup_two_stage": AGG_TWO_STAGE,
    "EPDMS_v2_private_test_hard_two_stage": AGG_TWO_STAGE,
    "EPDMS_v2_navtest_single_stage": AGG_SINGLE_STAGE,
    "PDMS_v1_navtest": AGG_SINGLE_STAGE,
}

QUESTION_ANSWERED = (
    "would another draw of LOGS (the OpenScene log_name clusters) give this score? "
    "— evaluation-set variance only. ⛔ BLIND to training variance (H-ESTIM-SEED-1) "
    "and to inference variance of a sampling planner: a separated paired interval "
    "between two trained arms is NECESSARY, NOT SUFFICIENT — read it against a "
    "replicate arm's floor.")


class NavSimCIError(ValueError):
    """Refuses rather than guesses. Every message names the defect it prevents."""


# --------------------------------------------------------------------------- #
# 1. Per-unit contributions — the devkit's aggregation, reproduced exactly     #
# --------------------------------------------------------------------------- #
def _wavg(tokens, rows: dict, column: str) -> float:
    """``calculate_weighted_average_score`` (run_pdm_score.py L221-239 @0a380a9)
    for ONE column, with its exact NaN semantics:

    * no row for any of the tokens  -> NaN   (``df.empty``, L228-229)
    * total weight == 0             -> NaN   (L235-236)
    * any NaN score or weight       -> NaN   (a plain ``.sum``, not ``nansum``)

    ⚠️ ``df[df.token.isin(tokens)]`` selects each PRESENT row once, however often
    the token is listed, and silently drops listed tokens with no row — both
    reproduced (``dict.fromkeys`` + the membership filter)."""
    sel = [rows[t] for t in dict.fromkeys(tokens) if t in rows]
    if not sel:
        return float("nan")
    w = np.array([float(r["weight"]) for r in sel], dtype=np.float64)
    s = np.array([float(r[column]) for r in sel], dtype=np.float64)
    tw = w.sum()
    if tw == 0:
        return float("nan")
    return float((s * w).sum() / tw)


def two_stage_key_contributions(rows: dict, mapping, column: str = "score") -> np.ndarray:
    """Per-mapping-key contribution to two-stage EPDMS, exactly as the devkit.

    ``rows``    ``{token: {"score": float, "weight": float, ...}}`` — the per-token
                frame AFTER ``create_scene_aggregators`` + ``compute_final_scores``
                (the pre-CSV ``pdm_score_df``). ⛔ The CSV that run_pdm_score.py
                writes DROPS ``weight`` (L422 keeps token/valid/score_cols only),
                so the stage-2 Gaussian weights cannot be recovered from it — a
                caller must capture the frame before L422.
    ``mapping`` ``[(orig_token, prev_token, [(now_tok, prev_tok), ...]), ...]`` —
                ``reactive_all_mapping`` from the split YAML, in its order.
    """
    out = []
    for entry in mapping:
        orig, prev, pairs = str(entry[0]), str(entry[1]), entry[2]
        first = [str(p[0]) for p in pairs if len(p) > 0]          # L258
        second = [str(p[1]) for p in pairs if len(p) > 1]         # L259
        g1 = _wavg([orig], rows, column) * _wavg(first, rows, column)    # L267-279
        g2 = _wavg([prev], rows, column) * _wavg(second, rows, column)   # L270-281
        out.append((g1 + g2) / 2.0)                                   # L283
    return np.asarray(out, dtype=np.float64)


def single_stage_contributions(scores) -> np.ndarray:
    """Single-stage: each token IS its own contribution (L293, a skipna mean)."""
    return np.asarray(scores, dtype=np.float64).reshape(-1)


def official_aggregate(contributions) -> float:
    """``pd.DataFrame(...).mean()`` / ``.mean(skipna=True)``: NaN units are
    SKIPPED, and an all-NaN set is NaN. ⚠️ Skipping changes the DENOMINATOR: an
    agent that crashes on its hardest scenes is scored on the rest — reported as
    ``n_units_nan`` on every record, never hidden."""
    c = np.asarray(contributions, dtype=np.float64)
    ok = ~np.isnan(c)
    return float(c[ok].mean()) if ok.any() else float("nan")


def _nanmean(v) -> float:
    v = np.asarray(v, dtype=np.float64)
    ok = ~np.isnan(v)
    return float(v[ok].mean()) if ok.any() else float("nan")


_nanmean.__name__ = "nanmean_official_skipna"


# --------------------------------------------------------------------------- #
# 2. Refusals                                                                  #
# --------------------------------------------------------------------------- #
def refusal(reason: str, *, n_clusters: int, n_units: int, aggregation=None,
            **extra) -> dict:
    """``{status: UNAVAILABLE, reason, n}`` — criteria_check reads it as REFUSED
    (a visible work item), never as a pass and never as a silent omission."""
    out = {"status": "UNAVAILABLE", "reason": reason, "n": int(n_clusters),
           "n_clusters": int(n_clusters), "n_units": int(n_units),
           "min_clusters": MIN_CLUSTERS, "cluster_unit": CLUSTER_UNIT,
           "estimator_if_it_could_rule": ESTIMATOR, "spec": SPEC_DOC}
    if aggregation is not None:
        out["aggregation"] = aggregation
    out.update(extra)
    return out


def _validate(contributions, clusters, aggregation, min_clusters):
    c = np.asarray(contributions, dtype=np.float64).reshape(-1)
    if len(clusters) != c.size:
        raise NavSimCIError(f"clusters/contributions length mismatch: {len(clusters)} vs "
                            f"{c.size} — a positional join on mismatched rows assigns "
                            f"scenes to the wrong log")
    if c.size == 0:
        raise NavSimCIError("no units — refusing to bootstrap nothing")
    if aggregation not in AGGREGATIONS:
        raise NavSimCIError(f"aggregation must be one of {sorted(AGGREGATIONS)}, got "
                            f"{aggregation!r}. ⛔ An interval that does not name the "
                            f"aggregate it resamples is precise about an unknown thing.")
    if int(min_clusters) < 2:
        raise NavSimCIError("min_clusters < 2 cannot define a cluster bootstrap")
    cl = [str(x) for x in clusters]
    if any(x in ("", "None", "nan") for x in cl):
        raise NavSimCIError("a unit has no cluster id — refusing to invent one")
    return c, cl


def _check_official(point: float, official_value, tol: float):
    if official_value is None:
        return {"status": "UNCHECKED",
                "reason": ("no official_value supplied — pass the devkit's summary row "
                           "(`extended_pdm_score_combined` / `average_all_frames`) to "
                           "prove the resampled statistic IS the official aggregate")}
    d = abs(float(point) - float(official_value))
    return {"status": "OK" if d <= tol else "MISMATCH", "official_value": float(official_value),
            "point": float(point), "abs_diff": d, "tol": tol}


# --------------------------------------------------------------------------- #
# 3. The estimator                                                             #
# --------------------------------------------------------------------------- #
def log_cluster_bootstrap(contributions, clusters, *, aggregation: str,
                          n_boot: int = N_BOOT, seed: int = 0, alpha: float = ALPHA,
                          min_clusters: int = MIN_CLUSTERS, official_value=None,
                          official_tol: float = 1e-9, dp: int = 4) -> dict:
    """Percentile CI on the OFFICIAL NavSim aggregate, resampling LOGS.

    ``contributions`` [U] per-unit contributions (per mapping key for two-stage,
    per token for single-stage — see §1); ``clusters`` [U] their ``log_name``.
    ⛔ Refuses (``status: UNAVAILABLE``) below ``min_clusters`` distinct logs, when
    the full-sample point does not reproduce ``official_value``, or when a
    resample contains only failed units.
    """
    c, cl = _validate(contributions, clusters, aggregation, min_clusters)
    n_clusters = len(set(cl))
    n_nan = int(np.isnan(c).sum())
    point = official_aggregate(c)
    rep = _check_official(point, official_value, official_tol)
    common = {"aggregation": aggregation, "n_units_nan": n_nan}
    if n_clusters < int(min_clusters):
        return refusal(
            f"n_clusters={n_clusters} distinct {CLUSTER_UNIT}s < the floor of "
            f"{int(min_clusters)} (RG-14). The bootstrap resamples LOGS, so {c.size} "
            f"units are {n_clusters} independent draws — an interval here could not "
            f"rule. ⛔ Not a pass; not a scene-token fallback.",
            n_clusters=n_clusters, n_units=c.size, point_estimate=point,
            reproduces_official=rep, **common)
    if rep["status"] == "MISMATCH":
        return refusal(
            f"the full-sample aggregate {point!r} does not reproduce the official "
            f"value {rep['official_value']!r} (|diff| {rep['abs_diff']:.3g} > "
            f"{official_tol}). A bootstrap of a different aggregate is precise about "
            f"the wrong thing — refused.",
            n_clusters=n_clusters, n_units=c.size, reproduces_official=rep, **common)
    if n_nan == c.size:
        return refusal("every unit is NaN (all scenes failed) — nothing to resample",
                       n_clusters=n_clusters, n_units=c.size, **common)
    boots = boot_statistics(c, cl, n_boot=n_boot, seed=seed)
    if np.isnan(boots).any():
        return refusal(
            f"{int(np.isnan(boots).sum())} of {n_boot} resamples contained ONLY failed "
            f"units (NaN) — the official aggregate is undefined on them",
            n_clusters=n_clusters, n_units=c.size, **common)
    ec = _ci.episode_cluster_bootstrap(c, cl, reduce=_nanmean, n_boot=n_boot,
                                       seed=seed, alpha=alpha, dp=dp)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    sens = _sensitivity_single(c, cl, n_boot=n_boot, seed=seed, alpha=alpha,
                               min_clusters=min_clusters, dp=dp)
    # ⛔ the two paths use the SAME draws (ci._draws, same seed): if they disagree
    # the reuse is broken, and a silently different interval would be published.
    if (abs(round(float(lo), dp) - ec["lo"]) > 10 ** -dp or
            abs(round(float(hi), dp) - ec["hi"]) > 10 ** -dp):
        raise NavSimCIError(f"internal draw mismatch vs taniteval.ci: ({lo}, {hi}) vs "
                            f"({ec['lo']}, {ec['hi']})")
    return {
        "status": "OK",
        "estimator": ESTIMATOR,
        "cluster_unit": CLUSTER_UNIT,
        "resample_unit": CLUSTER_UNIT,
        "aggregation": aggregation,
        "aggregation_source": AGGREGATIONS[aggregation],
        "point": round(point, dp),
        "mean": round(point, dp),
        "lo": round(float(lo), dp),
        "hi": round(float(hi), dp),
        "ci95": round(float((hi - lo) / 2.0), dp),
        "se": round(float(boots.std(ddof=1)), dp),
        "bootstrap_mean": round(float(boots.mean()), dp),
        "n_clusters": n_clusters,
        "n_units": int(c.size),
        "n_units_nan": n_nan,
        "min_clusters": int(min_clusters),
        "n_boot": int(n_boot), "seed": int(seed), "alpha": float(alpha),
        "reproduces_official": rep,
        "sensitivity_" + SENSITIVITY_UNIT: sens,
        "question_answered": QUESTION_ANSWERED,
        "spec": SPEC_DOC,
        "devkit_pin": DEVKIT_PIN,
        "_machine": ("taniteval.ci.episode_cluster_bootstrap(eid=log_name, "
                     "reduce=nanmean_official_skipna) — the SAME draws, cross-checked"),
        "_forbidden": ("⛔ scene-token resampling (overlapping units), the episode-"
                       "cluster bootstrap under any other unit, and overlapping_holdout_se "
                       "(biases the point estimate) are NOT admissible for NavSim."),
    }


def _sensitivity_single(c, cl, *, n_boot, seed, alpha, min_clusters, dp):
    """SPEC §3: the same interval with ``nuplan_drive`` clusters, or CANNOT-RULE."""
    dv = [drive_of(x) for x in cl]
    nd = len(set(dv))
    if nd < int(min_clusters):
        return {"status": "CANNOT-RULE", "unit": SENSITIVITY_UNIT, "n": nd,
                "reason": f"{nd} drives < floor {int(min_clusters)}"}
    b = boot_statistics(c, dv, n_boot=n_boot, seed=seed)
    if np.isnan(b).any():
        return {"status": "CANNOT-RULE", "unit": SENSITIVITY_UNIT, "n": nd,
                "reason": "a drive-level replicate drew only failed units"}
    lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"status": "OK", "unit": SENSITIVITY_UNIT, "n_clusters": nd,
            "lo": round(float(lo), dp), "hi": round(float(hi), dp),
            "se": round(float(b.std(ddof=1)), dp)}


def boot_statistics(contributions, clusters, *, n_boot: int = N_BOOT,
                    seed: int = 0) -> np.ndarray:
    """The ``n_boot`` resampled aggregates, drawn by ``taniteval.ci``'s OWN draw
    generator (same RNG path as ``episode_cluster_bootstrap``). Exposed so tests
    can assert ANALYTIC properties of the draws, not only their percentiles."""
    c = np.asarray(contributions, dtype=np.float64).reshape(-1)
    uniq, idx = _ci.episode_index([str(x) for x in clusters])
    return np.array([_nanmean(c[sel]) for sel in _ci._draws(uniq, idx, n_boot, seed)])


def paired_log_cluster_bootstrap(contrib_a, contrib_b, clusters, *, aggregation: str,
                                 n_boot: int = N_BOOT, seed: int = 0,
                                 alpha: float = ALPHA,
                                 min_clusters: int = MIN_CLUSTERS,
                                 official_a=None, official_b=None,
                                 official_tol: float = 1e-9) -> dict:
    """CI on ``agg(A) - agg(B)`` for two arms on the IDENTICAL units, the SAME
    resampled logs in every draw (``taniteval.ci.paired_episode_cluster_bootstrap``).

    ⛔ Refuses when the two arms' failed-unit (NaN) patterns differ: the official
    aggregate skips failed units, so the arms would be averaged over DIFFERENT
    scene sets and the delta would not be a paired delta (RG-13)."""
    a, cl = _validate(contrib_a, clusters, aggregation, min_clusters)
    b, _ = _validate(contrib_b, clusters, aggregation, min_clusters)
    if a.shape != b.shape:
        raise NavSimCIError(f"paired arms must be aligned: {a.shape} vs {b.shape}")
    n_clusters = len(set(cl))
    na, nb = np.isnan(a), np.isnan(b)
    if (na != nb).any():
        return refusal(
            f"the arms failed on DIFFERENT units ({int((na & ~nb).sum())} only-A, "
            f"{int((nb & ~na).sum())} only-B). The official aggregate skips failed "
            f"units, so the two scores cover different scene sets and their delta is "
            f"not paired — refused (RG-13: comparing on different grids is not a "
            f"comparison).", n_clusters=n_clusters, n_units=a.size,
            aggregation=aggregation)
    if n_clusters < int(min_clusters):
        return refusal(
            f"n_clusters={n_clusters} < floor {int(min_clusters)} (RG-14) — a paired "
            f"interval over {n_clusters} logs cannot rule; report the per-unit paired "
            f"deltas descriptively, with NO interval.",
            n_clusters=n_clusters, n_units=a.size, aggregation=aggregation,
            point_delta=official_aggregate(a) - official_aggregate(b))
    rep_a = _check_official(official_aggregate(a), official_a, official_tol)
    rep_b = _check_official(official_aggregate(b), official_b, official_tol)
    if "MISMATCH" in (rep_a["status"], rep_b["status"]):
        return refusal("an arm's full-sample aggregate does not reproduce its official "
                       "value — refused", n_clusters=n_clusters, n_units=a.size,
                       aggregation=aggregation, reproduces_official_a=rep_a,
                       reproduces_official_b=rep_b)
    if na.all():
        return refusal("every unit failed in both arms", n_clusters=n_clusters,
                       n_units=a.size, aggregation=aggregation)
    pr = _ci.paired_episode_cluster_bootstrap(a, b, cl, n_boot=n_boot, seed=seed,
                                              alpha=alpha, reduce=_nanmean)
    dv = [drive_of(x) for x in cl]
    nd = len(set(dv))
    if nd >= int(min_clusters):
        pd_ = _ci.paired_episode_cluster_bootstrap(a, b, dv, n_boot=n_boot, seed=seed,
                                                   alpha=alpha, reduce=_nanmean)
        drive_blk = {"status": "OK", "unit": SENSITIVITY_UNIT, "n_clusters": nd,
                     "lo": pd_["lo"], "hi": pd_["hi"], "separated": pd_["separated"]}
        sep_scope = "log_name AND nuplan_drive (SPEC §5 conjunction)"
        separated = bool(pr["separated"] and pd_["separated"])
    else:
        drive_blk = {"status": "CANNOT-RULE", "unit": SENSITIVITY_UNIT, "n": nd,
                     "reason": f"{nd} drives < floor {int(min_clusters)}"}
        sep_scope = ("log_name ONLY — the drive level CANNOT-RULE; state any claim as "
                     "'separated at the log level only' (SPEC §5)")
        separated = bool(pr["separated"])
    out = dict(pr)
    out.update({
        "separated": separated,
        "separated_log_name": bool(pr["separated"]),
        "separated_scope": sep_scope,
        "sensitivity_" + SENSITIVITY_UNIT: drive_blk,
        "status": "OK",
        "estimator": PAIRED_ESTIMATOR,
        "cluster_unit": CLUSTER_UNIT, "resample_unit": CLUSTER_UNIT,
        "aggregation": aggregation, "aggregation_source": AGGREGATIONS[aggregation],
        "n_clusters": n_clusters, "n_units": int(a.size),
        "n_units_nan": int(na.sum()), "min_clusters": int(min_clusters),
        "seed": int(seed), "alpha": float(alpha),
        "reproduces_official_a": rep_a, "reproduces_official_b": rep_b,
        "question_answered": QUESTION_ANSWERED,
        "spec": SPEC_DOC, "devkit_pin": DEVKIT_PIN,
        "_machine": ("taniteval.ci.paired_episode_cluster_bootstrap(eid=log_name, "
                     "reduce=nanmean_official_skipna)"),
    })
    out.pop("n_episodes", None)          # the unit is a LOG here; never call it an episode
    out.pop("n_windows", None)
    return out


# --------------------------------------------------------------------------- #
# 4. Convenience: from a devkit run to an interval                             #
# --------------------------------------------------------------------------- #
def interval_from_run(*, protocol: str, clusters_by_unit: dict, rows: dict = None,
                      mapping=None, scores: dict = None, official_value=None,
                      n_boot: int = N_BOOT, seed: int = 0, column: str = "score") -> dict:
    """One call from a scored run to the ``estimator.interval`` block.

    Two-stage protocols: ``rows`` (pre-CSV per-token frame with ``weight``) +
    ``mapping`` (``reactive_all_mapping``); ``clusters_by_unit`` maps each key's
    ORIG token to its ``log_name`` (``raw/cluster_maps/<split>.json`` →
    ``mapping_key_orig_token_to_log_name``).
    Single-stage protocols: ``scores`` ``{token: score}``; ``clusters_by_unit``
    maps token → ``log_name`` (``token_to_log_name``).
    """
    if protocol not in PROTOCOL_AGGREGATION:
        raise NavSimCIError(f"protocol {protocol!r} is not in the registered closed set "
                            f"{sorted(PROTOCOL_AGGREGATION)}")
    agg = PROTOCOL_AGGREGATION[protocol]
    if agg == AGG_TWO_STAGE:
        if rows is None or mapping is None:
            raise NavSimCIError("two-stage needs the per-token rows (with weight) AND the "
                                "mapping")
        keys = [str(e[0]) for e in mapping]
        missing = [k for k in keys if k not in clusters_by_unit]
        if missing:
            raise NavSimCIError(f"{len(missing)} mapping key(s) have no log_name (first: "
                                f"{missing[0]!r}) — refusing to guess a cluster")
        contrib = two_stage_key_contributions(rows, mapping, column=column)
        cl = [clusters_by_unit[k] for k in keys]
    else:
        if scores is None:
            raise NavSimCIError("single-stage needs {token: score}")
        toks = sorted(scores)
        missing = [t for t in toks if t not in clusters_by_unit]
        if missing:
            raise NavSimCIError(f"{len(missing)} token(s) have no log_name (first: "
                                f"{missing[0]!r}) — refusing to guess a cluster")
        contrib = single_stage_contributions([scores[t] for t in toks])
        cl = [clusters_by_unit[t] for t in toks]
    out = log_cluster_bootstrap(contrib, cl, aggregation=agg, n_boot=n_boot, seed=seed,
                                official_value=official_value)
    out["protocol"] = protocol
    return out


def rows_from_score_frame(path, *, score_column: str = "score") -> tuple:
    """Read a devkit PRE-CSV score frame -> ``(rows, token_to_log_name)``.

    The frame is ``pdm_score_df`` AFTER ``create_scene_aggregators`` +
    ``compute_final_scores`` and BEFORE ``run_pdm_score.py:422`` drops ``weight`` and
    ``log_name`` (E1 banks it as ``<arm>_final_scores_frame.csv``). ⛔ The PUBLISHED
    CSV cannot be used: without ``weight`` the stage-2 Gaussian aggregation cannot be
    reproduced, and an interval on a different aggregate is precise about the wrong
    thing. Summary rows (``extended_pdm_score_*``, ``average*``) are skipped.
    """
    import csv as _csv
    rows, tok2log, skipped = {}, {}, []
    with open(path, newline="", encoding="utf-8") as f:
        rd = _csv.DictReader(f)
        for need in ("token", score_column, "weight"):
            if need not in (rd.fieldnames or []):
                raise NavSimCIError(
                    f"{path}: column {need!r} is missing (have {rd.fieldnames}). The "
                    f"PUBLISHED CSV drops `weight`/`log_name` — capture the pre-CSV frame.")
        for r in rd:
            tok = str(r["token"])
            if tok.startswith(("extended_pdm_score", "average")):
                skipped.append(tok)
                continue
            def _f(x):
                x = (x or "").strip()
                return float("nan") if x == "" or x.lower() == "nan" else float(x)
            rows[tok] = {"score": _f(r[score_column]), "weight": _f(r["weight"]),
                         "log_name": r.get("log_name"), "valid": r.get("valid")}
            if r.get("log_name"):
                tok2log[tok] = r["log_name"]
    if not rows:
        raise NavSimCIError(f"{path}: no per-token rows (skipped {skipped}) — refusing "
                            f"to report success over an empty set")
    return rows, tok2log


def is_admissible_interval(block, *, protocol: str | None = None,
                           min_clusters: int = MIN_CLUSTERS,
                           max_clusters: int | None = None) -> tuple:
    """The SPEC's admissibility predicate, usable by any consumer (the registry
    checker implements the same rule from the registry's own fields).

    Returns ``(verdict, why)`` with verdict in {"PASS", "REFUSED", "FAIL"}."""
    if not isinstance(block, dict):
        return "FAIL", "no interval block"
    status = str(block.get("status", "")).upper()
    if status in ("UNAVAILABLE", "REFUSED", "N/A", "NA", "NOT_APPLICABLE", "BLOCKED"):
        return ("REFUSED", block.get("reason", "")) if block.get("reason") else \
            ("FAIL", "a refusal with no reason")
    est = str(block.get("estimator", ""))
    unit = str(block.get("cluster_unit", block.get("resample_unit", "")))
    if est not in (ESTIMATOR, PAIRED_ESTIMATOR):
        return "FAIL", f"estimator {est!r} is not {ESTIMATOR!r}/{PAIRED_ESTIMATOR!r}"
    if unit != CLUSTER_UNIT or str(block.get("resample_unit", unit)) != CLUSTER_UNIT:
        return "FAIL", f"resampling unit {unit!r} is not {CLUSTER_UNIT!r}"
    n = block.get("n_clusters")
    if not isinstance(n, int) or n < int(min_clusters):
        return "FAIL", f"n_clusters={n!r} < floor {int(min_clusters)}"
    if max_clusters is not None and n > int(max_clusters):
        return "FAIL", (f"n_clusters={n} exceeds the split's {max_clusters} logs — a "
                        f"finer unit than log_name was resampled")
    agg = block.get("aggregation")
    if agg not in AGGREGATIONS:
        return "FAIL", f"aggregation {agg!r} not named"
    if protocol is not None and PROTOCOL_AGGREGATION.get(protocol) != agg:
        return "FAIL", (f"aggregation {agg!r} is not the official one for {protocol!r} "
                        f"({PROTOCOL_AGGREGATION.get(protocol)!r})")
    if int(block.get("n_boot", 0)) < N_BOOT:
        return "FAIL", f"n_boot={block.get('n_boot')!r} < {N_BOOT}"
    return "PASS", f"{est} over {n} {CLUSTER_UNIT}s, {agg}"
