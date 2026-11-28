"""The refcv6 tactical ACCEPTANCE INSTRUMENTS — T-FLIP, T-ZERO, speed obedience.

⛔ These are the instruments, frozen BEFORE any arm trains. The GPU runs later;
what ships today is the code, its unit tests, and the bars — so that when the
numbers arrive nobody gets to choose the threshold after seeing them.

    T-FLIP      flip the junction command; does the plan FOLLOW the fed command?
                bar: follows_FED >= 0.50   (refcv5-v2 today: 0.205)
                     true - shuffled >= 0.38   (today: +0.099)
    T-ZERO      a REPORTED DIAGNOSTIC on the NON-TURN classes.
    OBEDIENCE   force 30 km/h on windows whose GT max exceeds 40 km/h;
                the planned maximum must stay <= the limit on >= 99 %.

⭐ WHY T-FLIP AND NOT THE SHUFFLE ALONE. The shuffle feeds ``follow`` on most
windows (nav is constant on ~75-79 % of the corpus — the C6 confound), so a
shuffle delta mixes "was fed a different turn" with "was fed nothing". The flip
feeds the OPPOSITE turn on EVERY informative window, so ``follows_FED`` and
``follows_TRUE`` are mutually exclusive by construction: a follower reads
FED high / TRUE low, a scene-driven arm keeps TRUE at its nav_true rate, and a
blind token-echo reads exactly 1 / 0. ``taniteval.nav_compliance.compliance_arm``
already computes that block; ``refcv3_arm.py --with-navflip`` (``:1888-1898``)
already rolls the arm. ⛔ **IT HAS NEVER BEEN RUN.** This module is the missing
half: the verdict function that applies the frozen bars to that block.

⭐⭐ WHY T-ZERO IS SCOPED TO THE **NON-TURN** CLASSES, and why it is a DIAGNOSTIC
rather than a bar.

⛔ FIRST, THE RULING, so no later reader misreads the turn numbers as a defect.
**PI 2026-09-16, verbatim:** *"its totally fine to process the nav command and
generate from it the turing command, we are not to aim in this stage to generate
a route."* A tactical layer that derives ``TURN_L`` / ``TURN_R`` from the nav
command is therefore **BY DESIGN**. refcv5-v2's tactical turn head collapses to
recall 0.000 with nav zeroed; under this ruling that is an admissible reading,
NOT a failure, and nothing in this module penalises or gates it.

⇒ Which is exactly why the turn classes cannot carry the diagnostic. MEASURED
(``REFCV6_CLARIFICATION.md`` §4.1): **42.7 % of the tactical TURN label's entropy
is already in the nav token**, and *no clip has ``NAV_FOLLOW_ROAD`` together with
a tactical turn (0 of 2,897)*. A layer can satisfy the turn labels from the
command alone — permitted, but uninformative about learning. The classes where
nav explains almost nothing are the longitudinal actions (5.2 %), the speed
bucket (5.5 %) and the non-turn laterals (lateral overall 16.9 %) — lane
keeping, nudges, yielding, gap targets, the speed behaviours. **That** is where
"learned from the scene" has to show, and that is what this instrument reports.

⛔ EVERY NUMBER CARRIES ``n_windows`` AND ``n_episodes``. A rate without both is
not admissible (the estimator is the episode-cluster bootstrap in
``taniteval.ci``; windows inside one episode are not independent).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "TFLIP_BAR_FOLLOWS_FED", "TFLIP_BAR_TRUE_MINUS_SHUFFLED",
    "TFLIP_TODAY_FOLLOWS_FED", "TFLIP_TODAY_TRUE_MINUS_SHUFFLED",
    "OBEDIENCE_BAR", "OBEDIENCE_FORCED_LIMIT_KMH", "OBEDIENCE_GT_FLOOR_KMH",
    "TURN_CLASSES", "tflip_verdict", "tzero_nonturn_report",
    "speed_obedience_report", "acceptance_panel",
]

# --- the frozen bars -------------------------------------------------------
#: ⛔ FROZEN 2026-09-16, before any refcv6 arm trains. Sourced from
#: ``SPEC_REFCV6_V2.md`` §5 and ``REFCV6_CLARIFICATION.md`` §3.3.
TFLIP_BAR_FOLLOWS_FED: float = 0.50
#: Half the 0.76 ceiling a perfect follower would reach on these windows.
TFLIP_BAR_TRUE_MINUS_SHUFFLED: float = 0.38

#: refcv5-v2's MEASURED values on the same instrument (T1, 4,823 windows / 141
#: episodes; 39 windows fed a different turn). Carried here so a result is
#: always read against the thing it must beat, not against zero.
TFLIP_TODAY_FOLLOWS_FED: float = 0.205
TFLIP_TODAY_TRUE_MINUS_SHUFFLED: float = 0.099

OBEDIENCE_BAR: float = 0.99
OBEDIENCE_FORCED_LIMIT_KMH: float = 30.0
OBEDIENCE_GT_FLOOR_KMH: float = 40.0

#: The classes a nav token can satisfy by itself. T-ZERO EXCLUDES them.
TURN_CLASSES: frozenset[str] = frozenset({"TURN_L", "TURN_R"})

_MIN_WINDOWS = 30
_MIN_EPISODES = 5


def _n(block: Mapping[str, Any]) -> tuple[int, int]:
    return (int(block.get("n_windows") or 0), int(block.get("n_episodes") or 0))


# ---------------------------------------------------------------------------
# T-FLIP
# ---------------------------------------------------------------------------

def tflip_verdict(arm: Mapping[str, Any], *,
                  bar_follows_fed: float = TFLIP_BAR_FOLLOWS_FED,
                  bar_delta: float = TFLIP_BAR_TRUE_MINUS_SHUFFLED
                  ) -> dict[str, Any]:
    """Apply the frozen T-FLIP bars to ONE ``compliance_arm`` block.

    ``arm`` is what ``taniteval.nav_compliance.compliance_arm`` returns for the
    PLAN readout — it must carry a ``flipped`` sub-block, which exists only when
    the roll passed ``--with-navflip``.

    ⛔ AN ABSENT ``flipped`` BLOCK IS ``NOT_RUN``, NEVER ``FAIL``. The arm has
    never been rolled; calling that a failure would bank a verdict about a model
    from an absence of data about it.

    ⛔ THE VERDICT IS THE **LOWER CI BOUND** AGAINST THE BAR, not the point
    estimate. ``n = 39`` windows fed a different turn on refcv5-v2's panel, and
    at that n a point estimate of 0.51 has a CI that comfortably contains 0.2.
    Reporting ``PASS`` off the mean would be the winner's-curse read this
    programme has already paid for.
    """
    out: dict[str, Any] = {
        "test": "T-FLIP",
        "bars": {"follows_FED_command": bar_follows_fed,
                 "paired_true_minus_shuffled": bar_delta},
        "today_refcv5_v2": {"follows_FED_command": TFLIP_TODAY_FOLLOWS_FED,
                            "paired_true_minus_shuffled":
                                TFLIP_TODAY_TRUE_MINUS_SHUFFLED},
    }
    flipped = arm.get("flipped")
    if not flipped:
        out.update(verdict="NOT_RUN", reason=(
            "the block carries no `flipped` sub-block: the arm was rolled "
            "without --with-navflip (taniteval/tools/refcv3_arm.py:1888-1898). "
            "T-FLIP has NEVER been run on any REF-C arm — this is an absence "
            "of data, not a failing model."))
        return out
    if not arm.get("powered", False):
        out.update(verdict="UNPOWERED", reason=arm.get("reason"),
                   n_informative_windows=arm.get("n_informative_windows"),
                   n_informative_episodes=arm.get("n_informative_episodes"))
        return out

    fed = flipped.get("follows_FED_command") or {}
    true_f = flipped.get("follows_TRUE_command") or {}
    d_s = arm.get("paired_true_minus_shuffled") or {}
    nw, ne = _n(fed)
    out["measured"] = {
        "follows_FED_command": {"mean": fed.get("mean"),
                                "ci": [fed.get("lo"), fed.get("hi")],
                                "n_windows": nw, "n_episodes": ne},
        "follows_TRUE_command": {"mean": true_f.get("mean"),
                                 "ci": [true_f.get("lo"), true_f.get("hi")]},
        "paired_true_minus_shuffled": {
            "delta": d_s.get("delta"), "ci": [d_s.get("lo"), d_s.get("hi")],
            "separated": d_s.get("separated"),
            "n_windows": d_s.get("n_windows")},
    }
    if nw < _MIN_WINDOWS or ne < _MIN_EPISODES:
        out.update(verdict="UNPOWERED", reason=(
            f"{nw} windows / {ne} episodes fed a flipped turn < "
            f"{_MIN_WINDOWS} / {_MIN_EPISODES}. ⚠️ On refcv5-v2's panel this "
            f"block had n = 39 windows; a bar applied at that n is a coin "
            f"toss dressed as a decision."))
        return out

    lo_fed = fed.get("lo", fed.get("mean"))
    lo_delta = d_s.get("lo", d_s.get("delta"))
    pass_fed = lo_fed is not None and float(lo_fed) >= bar_follows_fed
    pass_delta = (lo_delta is not None and float(lo_delta) >= bar_delta
                  and bool(d_s.get("separated")))
    out["gates"] = {
        "follows_FED_lower_bound_clears_bar": bool(pass_fed),
        "true_minus_shuffled_lower_bound_clears_bar_and_separated":
            bool(pass_delta),
    }
    if pass_fed and pass_delta:
        out.update(verdict="PASS", reason=(
            "with the junction command flipped the plan FOLLOWS the fed "
            "command at a rate whose CI lower bound clears the bar, and the "
            "true-minus-shuffled compliance drop is separated and clears its "
            "bar. Nav is mandatory in the plan, as the PI required."))
    else:
        missed = [k for k, v in out["gates"].items() if not v]
        out.update(verdict="FAIL", reason=(
            f"the plan does not follow a flipped command: {missed} did not "
            f"clear. The operative plan is still vision/ego-driven and only "
            f"weakly steerable by nav — the refcv5-v2 reading (0.205)."))
    return out


# ---------------------------------------------------------------------------
# T-ZERO
# ---------------------------------------------------------------------------

def tzero_nonturn_report(per_class_true: Mapping[str, Mapping[str, Any]],
                         per_class_navzero: Mapping[str, Mapping[str, Any]],
                         *, turn_classes: Sequence[str] = tuple(TURN_CLASSES),
                         min_n: int = 30) -> dict[str, Any]:
    """⭐ A REPORTED DIAGNOSTIC, not a bar — on the classes nav cannot explain.

    ``per_class_*`` are ``refcv6_tactical.per_class_report``-shaped dicts (each
    class carrying ``n``/``n_pos``, ``recall``, ``precision``). The two are the
    SAME windows under ``nav_true`` and under ``nav_cmd=None``.

    Reports, per class: recall true, recall nav-zeroed, and the RETAINED
    fraction ``recall_zero / recall_true``. A class whose recall collapses to
    ~0 under the zero is a **nav echo**; a class that keeps most of its recall
    learned something from the scene.

    ⛔ TURN CLASSES ARE EXCLUDED FROM THE SUMMARY AND REPORTED SEPARATELY,
    because 42.7 % of the TURN label's entropy is already in nav — a turn head
    can satisfy its labels by copying the command, so its collapse under the
    zero is uninformative about learning. They are still PRINTED, because
    hiding them would hide the echo itself.

    ⛔ CLASSES UNDER ``min_n`` ARE REPORTED WITH THEIR n AND EXCLUDED FROM THE
    SUMMARY. A retained fraction over 15 positives is noise.
    """
    turn = set(turn_classes)
    rows: dict[str, Any] = {}
    kept: list[float] = []
    for cls, t in per_class_true.items():
        z = per_class_navzero.get(cls) or {}
        n = int(t.get("n") if t.get("n") is not None else (t.get("n_pos") or 0))
        r_t, r_z = t.get("recall"), z.get("recall")
        retained = (None if (r_t in (None, 0.0) or r_z is None)
                    else float(r_z) / float(r_t))
        row = {"n": n, "recall_nav_true": r_t, "recall_nav_zero": r_z,
               "retained_fraction": retained,
               "is_turn_class": cls in turn,
               "powered": n >= min_n}
        if cls in turn:
            row["_note"] = ("EXCLUDED from the summary. Deriving the turn "
                            "command from nav is BY DESIGN (PI 2026-09-16); "
                            "42.7 % of the TURN label's entropy is already in "
                            "the nav token, so a collapse here is PERMITTED "
                            "and uninformative about learning. ⛔ Do not read "
                            "a low retained fraction on a turn class as a "
                            "defect.")
        elif n < min_n:
            row["_note"] = (f"n = {n} < {min_n}: reported with its n, excluded "
                            f"from the summary")
        elif retained is not None:
            kept.append(retained)
        rows[cls] = row
    summary = {
        "n_classes_summarised": len(kept),
        "median_retained_fraction": (float(np.median(kept)) if kept else None),
        "min_retained_fraction": (float(np.min(kept)) if kept else None),
        "_reads": ("retained ~ 0 on the NON-TURN classes means the tactical "
                   "layer learned nothing from the scene on the behaviours nav "
                   "cannot explain (longitudinal 5.2 %, speed bucket 5.5 %, "
                   "lane keeping / nudges / yielding / gap targets). retained "
                   "~ 1 means the behaviour is read from the scene. ⛔ TURN "
                   "classes are excluded: deriving them from nav is BY DESIGN "
                   "(PI 2026-09-16), so their collapse is permitted. ⛔ This is "
                   "a DIAGNOSTIC: there is no pass/fail bar, because a high "
                   "retained fraction on a class the model never fires is not a "
                   "success."),
    }
    return {"test": "T-ZERO (non-turn diagnostic)", "classes": rows,
            "summary": summary, "turn_classes_excluded": sorted(turn),
            "min_n": min_n}


# ---------------------------------------------------------------------------
# the speed-obedience test
# ---------------------------------------------------------------------------

def speed_obedience_report(planned_max_ms, gt_max_ms, episode_id,
                           *, forced_limit_kmh: float = OBEDIENCE_FORCED_LIMIT_KMH,
                           gt_floor_kmh: float = OBEDIENCE_GT_FLOOR_KMH,
                           ade_forced=None, ade_baseline=None,
                           rows_empty: int = 0, tol_ms: float = 0.0,
                           bar: float = OBEDIENCE_BAR,
                           n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """⛔ Force 30 km/h; the PLANNED maximum must stay <= the limit on >= 99 %.

    ``planned_max_ms`` ``[N]``  the max speed of the EMITTED plan under the
                               FORCED ceiling (``refcv6_selection
                               .planned_max_speed`` on the selected row).
    ``gt_max_ms``      ``[N]``  the window's realised maximum — the population
                               filter, and the reason the test has teeth: on a
                               window the ego never exceeded 40 km/h, obeying
                               30 km/h costs nothing.
    ``ade_forced`` / ``ade_baseline`` ``[N]``  the ADE of the same windows with
                               and without the forced ceiling. ⛔ THE COST IS
                               PART OF THE RESULT: a model can obey any ceiling
                               by planning to stop, and only the ADE delta says
                               whether it did.
    ``rows_empty``             how many of those windows had NO compliant
                               candidate in the fan and therefore kept their
                               whole fan (``SpeedCeilingFilter``). ⚠️ This is
                               the ONLY structural reason the rate can fall
                               below 1.0, so it is in the verdict, not a
                               footnote.
    """
    p = np.asarray(planned_max_ms, dtype=np.float64).reshape(-1)
    g = np.asarray(gt_max_ms, dtype=np.float64).reshape(-1)
    eid = np.asarray(episode_id).reshape(-1)
    if not (p.shape == g.shape == eid.shape):
        raise ValueError(
            f"[obedience] shapes must match: planned {p.shape}, gt {g.shape}, "
            f"episode {eid.shape}")
    sel = g * 3.6 > float(gt_floor_kmh)
    n_w = int(sel.sum())
    n_e = int(len(set(eid[sel].tolist())))
    lim_ms = float(forced_limit_kmh) / 3.6
    out: dict[str, Any] = {
        "test": "speed obedience",
        "forced_limit_kmh": float(forced_limit_kmh),
        "population": f"windows whose GT max speed exceeds {gt_floor_kmh} km/h",
        "n_windows": n_w, "n_episodes": n_e,
        "bar_obeys_fraction": float(bar),
        "tol_ms": float(tol_ms),
        "rows_with_no_compliant_candidate": int(rows_empty),
    }
    if n_w < _MIN_WINDOWS or n_e < _MIN_EPISODES:
        out.update(verdict="UNPOWERED", reason=(
            f"{n_w} windows / {n_e} episodes < {_MIN_WINDOWS} / "
            f"{_MIN_EPISODES}"))
        return out
    obeys = (p[sel] <= lim_ms + float(tol_ms)).astype(np.float64)
    try:
        from taniteval import ci as _ci
        rate = _ci.episode_cluster_bootstrap(obeys, list(eid[sel]),
                                             n_boot=n_boot, seed=seed)
    except Exception as exc:                       # pragma: no cover - env
        rate = {"mean": float(obeys.mean()), "n_windows": n_w,
                "n_episodes": n_e, "lo": None, "hi": None,
                "estimator": f"point estimate only ({type(exc).__name__})"}
    out["obeys"] = rate
    over = p[sel] - lim_ms
    out["violation_ms"] = {
        "max": float(over.max()), "p95": float(np.percentile(over, 95)),
        "mean_over_violators": (float(over[over > 0].mean())
                                if (over > 0).any() else 0.0),
        "n_violating": int((over > float(tol_ms)).sum()),
    }
    if ade_forced is not None and ade_baseline is not None:
        a = np.asarray(ade_forced, dtype=np.float64).reshape(-1)[sel]
        b = np.asarray(ade_baseline, dtype=np.float64).reshape(-1)[sel]
        try:
            from taniteval import ci as _ci
            out["ade_cost"] = _ci.paired_episode_cluster_bootstrap(
                a, b, list(eid[sel]), n_boot=n_boot, seed=seed)
        except Exception as exc:                   # pragma: no cover - env
            out["ade_cost"] = {"delta": float((a - b).mean()),
                               "estimator": f"point only ({type(exc).__name__})"}
        out["ade_cost"]["_reads"] = (
            "forced minus baseline, in metres. A model can obey ANY ceiling by "
            "planning to stop; this delta is what says it did not.")
    else:
        out["ade_cost"] = {"status": "UNAVAILABLE", "reason": (
            "the ADE of the forced and baseline rolls was not supplied. ⛔ The "
            "obedience rate alone is NOT the result — report both or report "
            "neither.")}
    lo = rate.get("lo", rate.get("mean"))
    if lo is None:
        out.update(verdict="INADMISSIBLE",
                   reason="no CI on the obedience rate")
    elif float(lo) >= float(bar):
        out.update(verdict="PASS", reason=(
            f"the planned maximum stays at or below {forced_limit_kmh} km/h on "
            f"a fraction whose CI lower bound clears {bar}"))
    else:
        out.update(verdict="FAIL", reason=(
            f"obedience lower bound {float(lo):.4f} < {bar}. "
            f"{out['violation_ms']['n_violating']} of {n_w} windows exceeded "
            f"the forced ceiling; {rows_empty} of them had no compliant "
            f"candidate in the fan at all, which is a VOCABULARY limit, not a "
            f"model one — separate the two before reading this as disobedience."))
    return out


# ---------------------------------------------------------------------------
# the panel
# ---------------------------------------------------------------------------

def acceptance_panel(*, tflip: Mapping[str, Any] | None = None,
                     tzero: Mapping[str, Any] | None = None,
                     obedience: Mapping[str, Any] | None = None
                     ) -> dict[str, Any]:
    """The three instruments, with the overall reading.

    ⛔ A missing instrument is ``NOT_RUN`` and makes the panel ``INCOMPLETE``.
    It never silently drops out, because a panel that reports two of three
    passes reads as a pass.
    """
    parts = {"T-FLIP": tflip, "T-ZERO": tzero, "OBEDIENCE": obedience}
    missing = [k for k, v in parts.items() if v is None]
    verdicts = {k: (v or {}).get("verdict", "NOT_RUN") for k, v in parts.items()}
    # T-ZERO is a DIAGNOSTIC and carries no verdict; it must be PRESENT, not
    # passing.
    gating = {k: verdicts[k] for k in ("T-FLIP", "OBEDIENCE")}
    if missing:
        overall = "INCOMPLETE"
    elif all(v == "PASS" for v in gating.values()):
        overall = "PASS"
    elif any(v in ("UNPOWERED", "NOT_RUN", "INADMISSIBLE")
             for v in gating.values()):
        overall = "INCONCLUSIVE"
    else:
        overall = "FAIL"
    return {"overall": overall, "missing": missing,
            "gating_tests": gating,
            "diagnostic_tests": ["T-ZERO"],
            "T-FLIP": tflip, "T-ZERO": tzero, "OBEDIENCE": obedience,
            "_reads": ("T-FLIP and OBEDIENCE gate; T-ZERO is a reported "
                       "diagnostic on the non-turn classes and has no bar "
                       "(42.7 % of the TURN label's entropy is already in nav, "
                       "so a turn-class number cannot separate learning from "
                       "echoing).")}
