"""Stage 3 — CURATION (TanitDataSet rev-3 §7.3): decide what the model SEES how
often. Filtering removes the unusable; curation fixes the sampling distribution so
the world model doesn't over-fit the ~74 %-straight-highway majority and stay weak
where v1 is weak.

Four parts, all pure over the per-episode enrichment (goal + scene_tags stub +
lead_state stub + poses):
  1. stratified scene×behavior strata + inverse-frequency CLAMPED weights — the
     exact scheme ``refb_train`` uses for the route-heading aux CE, generalized to
     the full stratum grid.
  2. up-sample the 5 known weakness strata (high-speed longitudinal, stop-and-go,
     cut-ins, merges, curves) via a target-density boost.
  3. safety-event mining → a curated split (``hard_brake`` kinematic now;
     ``near_miss``/``anomaly`` VLM-pending).
  4. a frozen, hash-pinned per-tier eval holdout at ``split_unit_id`` granularity
     (never leak a route's windows; comparable across runs).

The scene axis of the strata comes from VLM ``scene_tags`` (deferred) and reads
``unknown`` for now; the behavior axis is fully kinematic and live today, so
curation already up-weights the kinematic weakness strata without the VLM pass.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field

import torch
from torch import Tensor

from tanitad.lake import goal_labels as GL
from tanitad.lake import vocab as V

# =========================================================================== #
# 1. STRATA + inverse-frequency clamped weights                                #
# =========================================================================== #
WEIGHT_CLAMP = 10.0           # == refb_train.ROUTE_CE_CLAMP (highway-dominated)
SCENE_AXES = ("road_type", "weather", "time_of_day", "traffic_density")
BEHAVIOR_AXES = ("LONMODE", "VTARGET", "LATMANEUVER")


def stratum_key(scene_tags: dict | None, goal: dict) -> tuple:
    """The stratum of one episode: (scene axes × behavior axes). Scene axes come
    from VLM ``scene_tags`` (``unknown`` until the deferred pass); behavior axes are
    the kinematic goal tokens (live now). Returns a hashable tuple used as the
    histogram key."""
    st = scene_tags or {}
    scene = tuple(str(st.get(a, {}).get("token", V.UNKNOWN)) if isinstance(
        st.get(a), dict) else str(st.get(a, V.UNKNOWN)) for a in SCENE_AXES)
    beh = tuple(goal.get(a, V.unknown_slot())["token"] for a in BEHAVIOR_AXES)
    return scene + beh


def inverse_frequency_weights(counts: dict, clamp: float = WEIGHT_CLAMP
                              ) -> dict:
    """Inverse-frequency, CLAMPED stratum weights (the ``refb_train`` scheme):
    ``w[s] = clamp_max(N / (K · count[s]), clamp)``. Average-frequency strata get
    ~1; rare strata are up-weighted up to ``clamp``; the dominant majority is
    down-weighted toward ``N/(K·N_maj)``."""
    N = sum(counts.values())
    K = max(1, len(counts))
    return {s: min(clamp, N / (K * max(1, c))) for s, c in counts.items()}


# =========================================================================== #
# 2. WEAKNESS strata — the 5 up-sampled targets (§7.3.2)                        #
# =========================================================================== #
WEAKNESS_BOOST = 2.5          # multiplies the inverse-freq weight of a weakness
# VTARGET bands at/above 30 m/s = high-speed longitudinal (the flagship lever).
HIGH_SPEED_VTARGET = frozenset(
    t for t in V.VTARGET_TOKENS if t.startswith("v(")
    and float(t[2:].split("-")[0]) >= 30.0)
STOPGO_LONMODE = frozenset({"stop_at_point", "hold_stop", "launch", "creep"})


def weakness_strata(goal: dict, lead_state: dict | None = None) -> list[str]:
    """Which of the 5 weakness categories this episode belongs to (§7.3.2).

    KINEMATIC-now: ``high_speed_longitudinal`` (VTARGET ≥ 30, VSOURCE sign/road),
    ``stop_and_go`` (LONMODE stop/hold/launch/creep), ``curves`` (VSOURCE
    curve_constrained). VLM/map-PARTIAL: ``cut_ins`` (INTERACT=yield_to_k + lead
    closing) and ``merges`` (ROUTE=merge / LATMANEUVER merge_in|yield_merge) fire
    only once the VLM lead/map slots are filled — listed so the boost activates
    automatically after the deferred pass, with no code change."""
    tags: list[str] = []
    vt = goal.get("VTARGET", {}).get("token")
    vs = goal.get("VSOURCE", {}).get("token")
    lon = goal.get("LONMODE", {}).get("token")
    lat = goal.get("LATMANEUVER", {}).get("token")
    route = goal.get("ROUTE", {}).get("token")
    interact = goal.get("INTERACT", {}).get("token")

    if vt in HIGH_SPEED_VTARGET and vs in ("sign_limit", "road_class_default"):
        tags.append("high_speed_longitudinal")
    if lon in STOPGO_LONMODE:
        tags.append("stop_and_go")
    if vs == "curve_constrained":
        tags.append("curves")
    # VLM/map-partial (fire when the deferred pass fills lead_state / map ROUTE):
    closing = bool(lead_state and lead_state.get("closing_speed_ms", 0) and
                   float(lead_state.get("closing_speed_ms", 0) or 0) > 0)
    if interact == "yield_to_k" and closing:
        tags.append("cut_ins")
    if route == "merge" or lat in ("merge_in", "yield_merge"):
        tags.append("merges")
    return tags


def weakness_boost(goal: dict, lead_state: dict | None = None) -> float:
    """Multiplicative target-density boost: ``WEAKNESS_BOOST`` if the episode is in
    any weakness stratum, else 1.0 (a stratum in several categories still boosts
    once — the boost is a density target, not a per-category stack)."""
    return WEAKNESS_BOOST if weakness_strata(goal, lead_state) else 1.0


def episode_weakness(poses: Tensor, lead_state: dict | None = None,
                     stride: int = 10) -> list[str]:
    """Weakness categories the episode CONTAINS anywhere (union over strided
    per-window kinematic goals), not just at its dominant token — an episode that
    is mostly cruise but contains a stop-and-go / high-speed / curve SEGMENT must
    still be up-sampled (the strategy boosts weakness *windows*). This is the
    poses-available path; the summary-goal path falls back to
    :func:`weakness_strata`."""
    tags: set[str] = set()
    for t in range(0, poses.shape[0], max(1, stride)):
        tags.update(weakness_strata(GL.mint_kinematic_goal(poses, t), lead_state))
    return sorted(tags)


# =========================================================================== #
# 3. SAFETY-EVENT mining → a curated split (§7.3.3)                             #
# =========================================================================== #
HARD_BRAKE_ACCEL = -3.5       # m/s^2: a firm/max deceleration
HARD_BRAKE_JERK = 4.0         # m/s^3: the onset spike
DT = 0.1


def _brake_features(poses: Tensor) -> tuple[float, float, float]:
    """(min_accel, peak_neg_jerk, min_v) over the episode — the kinematic brake
    signature. accel = Δv/Δt on the 10 Hz contract; jerk = Δaccel/Δt."""
    v = poses[:, 3]
    if v.shape[0] < 3:
        return 0.0, 0.0, float(v.min()) if v.numel() else 0.0
    a = (v[1:] - v[:-1]) / DT
    j = (a[1:] - a[:-1]) / DT
    return float(a.min()), float((-j).max()), float(v.min())


def safety_event(poses: Tensor, goal: dict | None = None,
                 lead_state: dict | None = None,
                 coc_trace: dict | None = None,
                 notable_events: list | None = None) -> str | None:
    """The safety-event class of an episode, or ``None``.

    ``hard_brake`` (KINEMATIC, cheap, high-precision): strong deceleration +
    firm/max DYN + a jerk spike. ``near_miss`` (kinematic+VLM): a lead ``ttc_s <
    1.5 s`` (VLM lead_state) + an evasive lateral maneuver — fires only with the
    lead slot filled. ``anomaly`` (VLM): a CoC ``physics_flag`` or a rare
    ``notable_events`` item. The two VLM classes stay dormant until the deferred
    pass provides lead_state / CoC — then they light up with no code change."""
    dyn = (goal or {}).get("DYN", {}).get("token")
    a_min, jerk, _ = _brake_features(poses)
    if a_min <= HARD_BRAKE_ACCEL and jerk >= HARD_BRAKE_JERK and \
            dyn in ("firm", "max", None):
        return "hard_brake"
    if lead_state and lead_state.get("ttc_s") is not None:
        ttc = float(lead_state["ttc_s"])
        lat = (goal or {}).get("LATMANEUVER", {}).get("token")
        if ttc < 1.5 and lat in ("nudge_left", "nudge_right", "abort_lc"):
            return "near_miss"
    if (coc_trace and coc_trace.get("physics_flag")) or (notable_events):
        return "anomaly"
    return None


# =========================================================================== #
# 4. FROZEN, hash-pinned per-tier eval holdout (§7.3.4)                         #
# =========================================================================== #
HOLDOUT_FRAC = 0.1
HOLDOUT_SALT = "taniteval-holdout-v1"


def stable_unit_frac(*parts: str, salt: str = HOLDOUT_SALT) -> float:
    """Deterministic ``[0,1)`` from a blake2b digest of the parts — the hash-pin
    that makes a split frozen + reproducible across machines/runs."""
    h = hashlib.blake2b(("\x1f".join([salt, *map(str, parts)])).encode(),
                        digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def is_eval_holdout(split_unit_id: str, tier: str, frac: float = HOLDOUT_FRAC,
                    salt: str = HOLDOUT_SALT) -> bool:
    """Frozen per-tier eval membership, split at ``split_unit_id`` granularity so a
    route's windows never straddle train/eval. Hashing includes ``tier`` → C-eval
    and R-eval are independent frozen holdouts; uniform hashing gives proportional
    coverage of every stratum (a stratified top-up can only ADD, never move a unit —
    the pin stays comparable across runs)."""
    if not split_unit_id:
        return False
    return stable_unit_frac(tier, split_unit_id, salt=salt) < frac


# =========================================================================== #
# The corpus-level curation pass                                               #
# =========================================================================== #
@dataclass
class CurationVerdict:
    strata: tuple
    weight: float
    safety_event: str | None
    is_eval_holdout: bool
    weakness: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"strata": list(self.strata), "weight": round(self.weight, 4),
                "safety_event": self.safety_event,
                "is_eval_holdout": self.is_eval_holdout,
                "weakness": list(self.weakness)}


def curate_corpus(records: list[dict], clamp: float = WEIGHT_CLAMP,
                  holdout_frac: float = HOLDOUT_FRAC) -> dict[object, CurationVerdict]:
    """Corpus-level Stage 3 → ``{id: CurationVerdict}``.

    ``records`` = ``[{id, split_unit_id, tier, goal, scene_tags?, lead_state?,
    poses?}]`` (one per episode; ``poses`` optional — enables ``hard_brake`` mining).
    Two passes: build the stratum histogram, then per-record weight = the
    inverse-frequency stratum weight × the weakness boost, plus safety mining and
    the frozen holdout flag."""
    strata = {r["id"]: stratum_key(r.get("scene_tags"), r["goal"])
              for r in records}
    counts = Counter(strata.values())
    base_w = inverse_frequency_weights(dict(counts), clamp=clamp)

    out: dict[object, CurationVerdict] = {}
    for r in records:
        poses = r.get("poses")
        weak = (episode_weakness(poses, r.get("lead_state")) if poses is not None
                else weakness_strata(r["goal"], r.get("lead_state")))
        boost = WEAKNESS_BOOST if weak else 1.0
        w = min(clamp * WEAKNESS_BOOST, base_w[strata[r["id"]]] * boost)
        sev = safety_event(poses, r.get("goal"), r.get("lead_state"),
                           r.get("coc_trace"), r.get("notable_events")) \
            if poses is not None else None
        out[r["id"]] = CurationVerdict(
            strata=strata[r["id"]], weight=w, safety_event=sev,
            is_eval_holdout=is_eval_holdout(r.get("split_unit_id", ""),
                                            r.get("tier", "ship"), holdout_frac),
            weakness=weak)
    return out
