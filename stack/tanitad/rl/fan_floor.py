"""``tanitad.rl.fan_floor`` — the RL rung's PRIMARY readout: the fan's FLOOR, its
DIVERSITY, and the collision rate of the candidate SET.

⛔ **WHY THIS MODULE EXISTS.** `B2` of the 2026-09-05 RL package
(`…/2026-09-05-rl-posttrain-refcv4b-refcv5/RESULT.md` §7) records that
``fan_floor@k``, fan-collision-vs-replay and diversity **do not exist as
instruments**, and that they are *"a hard blocker on the primary readout [that]
must land before any arm"*. Re-verified ABSENT 2026-09-06 by a content-verified
scan of **1,180 `.py`/`.md` files under `stack/` and `taniteval/` with 0
unreadable** — the absence claim is about content that was READ, not about a
search that returned nothing (`CLAUDE.md`, the "0 hits is a claim about the
SEARCH" rule).

⭐ **WHAT DD-v2 ACTUALLY PUBLISHES, and why the floor is the endpoint.** The
banked primary (`2512.07745`, sha256 `076ce47e…`, `Library/papers/`) reports its
RL stage's effect on the **raw fan before any selector** (Tab. 3, 20
trajectories): PDMS **@1 93.5 → 94.9 (+1.4)**, **@5 84.3 → 91.1**, **@10 75.3 →
84.4 (+9.1)**, with **diversity 42.3 → 30.3 (−28 %)**. ⇒ the published gain is
**an order of magnitude larger at the fan's FLOOR than at its TOP**, and it is
bought partly by narrowing the fan. A readout that quotes only the selected path
— or only ``@1`` — cannot see either half of that.

⛔⛔ **THE DEFECT THIS MODULE IS BUILT TO PREVENT: ``fan_floor@k`` IS MAXIMISED BY
MODE COLLAPSE.** Replace every candidate by the fan's own mean and the floor rises
to equal the top **while the fan stops being a fan**. That is not a hypothesis;
:func:`collapse_fan` constructs it and ``test_rl_fan_floor.py`` asserts it. It is
the same shape as ``rewards.RewardComponent.hackable_alone`` — a term that scores
well under a policy that drives badly — so this module carries the same
discipline: :func:`floor_verdict` **REFUSES** to return a floor verdict unless
diversity is measured on the same windows and reported beside it. A fan-floor
gain quoted without its diversity is inadmissible here by construction, not by
convention.

⛔⛔ **AND ``@k`` IS AN ORDER STATISTIC OVER K SAMPLES — the exact object the
programme retracted a decision over on 2026-09-05** (*"the RL arm's objective is
the 2.11x selection gap"* → **a best-of-N statistic read as a skill gap**;
`RETRACTION_LOG.md`, root-cause class *A BEST-OF-N STATISTIC READ AS A SKILL
GAP*). The rule that retraction leaves behind is binding on this file: **any
min/max-over-N quantity is reported beside the SAME statistic from a RANDOM
selector at the same N.** Hence :func:`random_best_of_k` is not optional
decoration — :func:`summarise_fan` computes it on every call, and the two are
different questions that must never be swapped:

===========================  ==================================================
``fan_floor@k``              the k-th best of **all K** candidates — a QUANTILE
                             of this fan's quality distribution. Valid only at
                             **fixed K**, which :func:`assert_equal_k` enforces.
``rand_best@k``              E[max over a RANDOM k-subset] — what a selector
                             with **no skill** would reach at budget k. The
                             control for "is this headroom?"
===========================  ==================================================

⚠️ **K-INVARIANCE.** ``@k`` moves when K moves, with no change in fan quality at
all. Any comparison of arms whose fans differ in size must use
:func:`fan_quantile` (τ-quantile, K-free) instead, and :func:`assert_equal_k`
raises rather than letting the K-dependent form be used across mismatched fans.

⛔ **WHAT THE QUALITY SIGNAL MAY BE — the echo rule.** ``quality`` is a
per-candidate scalar, higher-better, and it MUST be rule-based: the composed
reward (`rewards.RewardSpec`), a safety term, or a physical feasibility score.
⛔ **It may NEVER be ADE/FDE to the ego's recorded future, a speed-profile match,
or "distance to the logged path"** — those are the imitation loss in a reward
costume (the DDv2 analysis §3.3 echo table), and a floor computed on them
measures how tightly the fan hugs the human, which is the thing RL is *not* for.
:func:`assert_quality_admissible` refuses the named echo keys outright, and
``rewards.FORBIDDEN_REWARD_INPUTS`` covers the selector-derived ones.

⛔ **TIER.** Every number here is **T0 / fan-level**: a property of what the model
EMITS on recorded scenes, never a driving claim. A capability claim is **T1** with
the four families (`EVAL_DOCTRINE.md`). The fan floor is the RL rung's PRIMARY
endpoint and the T1 families are its GUARD — a safety gain bought with driving
quality is a failure, not a split.

Evidence class of everything computed here: **MEASURED (ours)**.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

__all__ = [
    "ECHO_QUALITY_KEYS",
    "DEFAULT_KS",
    "FanFloorError",
    "assert_equal_k",
    "assert_quality_admissible",
    "fan_floor_at_k",
    "fan_quantile",
    "random_best_of_k",
    "fan_diversity",
    "fan_collision_rate",
    "collapse_fan",
    "FanReadout",
    "summarise_fan",
    "floor_verdict",
]


#: ⛔ Quality signals that are the IMITATION TARGET wearing a reward's clothes.
#: A fan floor computed on any of these measures how closely the fan hugs the
#: human's recorded future — the thing the RL stage is explicitly NOT for.
ECHO_QUALITY_KEYS = frozenset({
    "ade", "fde", "minade", "minfde", "ade_m", "gt_similarity",
    "gt_distance", "path_match", "speed_match", "speed_profile",
    "nav_agreement", "route_agreement", "ep_ratio",
})

#: DD-v2 Tab. 3 reports @1 / @5 / @10 of 20. On a 128-candidate fan the same
#: FRACTIONS are 1 / 32 / 64; both sets are computed so the published shape is
#: readable and the fraction-matched one is available.
DEFAULT_KS: tuple[int, ...] = (1, 5, 8, 10, 32, 64)


class FanFloorError(ValueError):
    """Raised when a fan readout is asked for something it cannot honestly give."""


# ---------------------------------------------------------------------------
# Guards — each refuses a specific way this readout has been got wrong before
# ---------------------------------------------------------------------------

def assert_equal_k(*fans: Tensor) -> int:
    """Refuse a K-dependent comparison across fans of different size.

    ``fan_floor@k`` is an order statistic over K candidates: it moves when K
    moves with **no change in fan quality at all**. Two arms whose fans differ in
    size cannot be compared on it, and the failure is silent — both numbers are
    well-formed. Use :func:`fan_quantile` instead when K genuinely differs.
    """
    ks = set()
    for f in fans:
        if f.dim() == 2:
            ks.add(int(f.shape[-1]))
        elif f.dim() == 4:
            ks.add(int(f.shape[1]))
        else:
            raise FanFloorError(
                f"expected [W, K] quality or [W, K, S, 2] fan, got "
                f"{tuple(f.shape)}")
    if len(ks) != 1:
        raise FanFloorError(
            f"fans differ in candidate count {sorted(ks)}: fan_floor@k is an "
            f"order statistic over K and is NOT comparable across K. Use "
            f"fan_quantile(tau) for a K-free comparison.")
    return ks.pop()


def assert_quality_admissible(name: str) -> str:
    """Refuse an ECHO quality signal by name. Returns the name when admissible."""
    key = name.strip().lower()
    if key in ECHO_QUALITY_KEYS:
        raise FanFloorError(
            f"quality={name!r} is an ECHO of the ego's recorded future. A fan "
            f"floor on it measures how tightly the fan hugs the human, which is "
            f"the imitation loss, not the RL endpoint. Admissible quality: the "
            f"composed rule-based reward, a safety term, or feasibility.")
    return name


def _check_fan(fan: Tensor) -> None:
    if fan.dim() != 4 or fan.shape[-1] != 2:
        raise FanFloorError(
            f"fan must be [W, K, S, 2] waypoints, got {tuple(fan.shape)}")


def _check_quality(quality: Tensor) -> None:
    if quality.dim() != 2:
        raise FanFloorError(
            f"quality must be [W, K] (higher is better), got "
            f"{tuple(quality.shape)}")
    if not torch.isfinite(quality).all():
        n = int((~torch.isfinite(quality)).sum())
        raise FanFloorError(
            f"quality carries {n} non-finite entries. A sentinel (-inf from a "
            f"reachability mask) read as a number produced this programme's "
            f"three-way control tie on 2026-09-05; carry the mask as a separate "
            f"binary feature instead of poisoning the score column.")


# ---------------------------------------------------------------------------
# The floor, its K-free form, and its mandatory best-of-k control
# ---------------------------------------------------------------------------

def fan_floor_at_k(quality: Tensor, ks=DEFAULT_KS) -> dict:
    """``fan_floor@k`` — the k-th BEST candidate's quality, per window.

    ``quality`` is ``[W, K]``, higher-better. Returns ``{k: [W]}``.

    This is DD-v2 Tab. 3's PDMS@k with our rule-based quality substituted for
    NAVSIM's PDM score. ``@1`` is the fan's best candidate; larger ``k`` reads
    **deeper into the fan**, so a rise at large ``k`` is the published signature
    of the RL stage (75.3 → 84.4 at @10 of 20).

    ⚠️ Order statistic over K — see :func:`assert_equal_k`. ⚠️ Maximised by mode
    collapse — see :func:`fan_diversity`, which :func:`floor_verdict` requires.
    """
    _check_quality(quality)
    K = int(quality.shape[1])
    srt, _ = torch.sort(quality, dim=1, descending=True)
    out = {}
    for k in ks:
        k = int(k)
        if k < 1:
            raise FanFloorError(f"k must be >= 1, got {k}")
        if k > K:
            continue                      # not defined on this fan; omitted, not faked
        out[k] = srt[:, k - 1]
    if not out:
        raise FanFloorError(f"no requested k in 1..{K}: {tuple(ks)}")
    return out


def fan_quantile(quality: Tensor, taus=(0.0, 0.25, 0.5, 0.75)) -> dict:
    """The K-FREE form: the ``tau`` quantile from the TOP of the fan, per window.

    ``tau = 0`` is the best candidate, ``tau = 0.5`` the median. Unlike ``@k``
    this is invariant to fan size, so it is the correct statistic whenever two
    arms emit different numbers of candidates.
    """
    _check_quality(quality)
    srt, _ = torch.sort(quality, dim=1, descending=True)
    K = int(quality.shape[1])
    out = {}
    for t in taus:
        t = float(t)
        if not 0.0 <= t <= 1.0:
            raise FanFloorError(f"tau must be in [0, 1], got {t}")
        idx = min(K - 1, max(0, int(round(t * (K - 1)))))
        out[t] = srt[:, idx]
    return out


def random_best_of_k(quality: Tensor, ks=DEFAULT_KS, *, draws: int = 40,
                     seed: int = 0) -> dict:
    """⛔ MANDATORY CONTROL — E[max over a RANDOM k-subset], per window.

    What a selector with **no skill at all** reaches at budget ``k``. The
    2026-09-05 retraction (*a best-of-N statistic read as a skill gap*) exists
    because this control was never computed: a random best-of-32 already beat the
    trained selector's argmax over all 128, so the "gap" was substantially a
    property of fan SIZE.

    ⇒ Whenever a ``@k`` number is quoted as headroom, this is the number it must
    be quoted beside. :func:`summarise_fan` computes it unconditionally.
    """
    _check_quality(quality)
    W, K = quality.shape
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    out = {}
    for k in ks:
        k = int(k)
        if k < 1 or k > K:
            continue
        acc = torch.zeros(W, dtype=quality.dtype)
        for _ in range(int(draws)):
            # independent subset per window, without replacement
            perm = torch.argsort(torch.rand(W, K, generator=g), dim=1)[:, :k]
            acc += quality.gather(1, perm.to(quality.device)).max(dim=1).values
        out[k] = acc / float(draws)
    return out


# ---------------------------------------------------------------------------
# Diversity — the half of DD-v2's Tab. 3 that a floor number hides
# ---------------------------------------------------------------------------

def fan_diversity(fan: Tensor, *, mode: str = "pairwise_path") -> Tensor:
    """Per-window fan diversity, in METRES. Higher = more spread.

    ``fan`` is ``[W, K, S, 2]``.

    * ``pairwise_path`` (default) — mean over unordered candidate pairs of the
      RMS waypoint distance between the two paths. Reduces to 0 exactly for a
      collapsed fan.
    * ``endpoint_std`` — the RMS distance of terminal points from their centroid.
      The statistic a two-scalar SCALE policy moves most directly, so it is the
      one to watch on this design.

    ⚠️ **OUR DEFINITION, not a reproduction of DD-v2's.** The primary reports a
    diversity of 42.3 → 30.3 and does not publish its formula in any section we
    have banked; the LEVEL is therefore not comparable to theirs and only the
    DIRECTION and the RELATIVE change are. Stated here so the number is never
    quoted against their table.

    ⚠️ **COMPUTED IN float64 ON PURPOSE.** ``endpoint_std`` subtracts a mean, and
    in float32 a fan of twelve IDENTICAL 19 m endpoints reads **1.9e-06** instead
    of zero — enough to defeat the control that must read a KNOWN value. The
    dtype is part of the instrument, not an implementation detail.
    """
    _check_fan(fan)
    fan = fan.to(torch.float64)
    if mode == "endpoint_std":
        end = fan[:, :, -1, :]                                   # [W, K, 2]
        c = end.mean(dim=1, keepdim=True)
        return (end - c).pow(2).sum(-1).mean(dim=1).sqrt()        # [W]
    if mode != "pairwise_path":
        raise FanFloorError(f"unknown diversity mode {mode!r}")
    W, K, S, _ = fan.shape
    if K < 2:
        raise FanFloorError("diversity needs K >= 2 candidates")
    # RMS-over-waypoints distance between every pair, averaged over pairs.
    a = fan.unsqueeze(2)                                          # [W, K, 1, S, 2]
    b = fan.unsqueeze(1)                                          # [W, 1, K, S, 2]
    d = (a - b).pow(2).sum(-1).mean(-1).sqrt()                    # [W, K, K]
    iu = torch.triu_indices(K, K, offset=1)
    return d[:, iu[0], iu[1]].mean(dim=1)                         # [W]


def collapse_fan(fan: Tensor, alpha: float) -> Tensor:
    """⛔ THE INSTRUMENT'S OWN DELIBERATE REGRESSION.

    Shrink every candidate toward the fan's per-window mean path by ``alpha``
    (``0`` = unchanged, ``1`` = total collapse). A gate that cannot FAIL a
    knowingly-collapsed fan cannot certify an honest one, and the test suite
    asserts that diversity falls monotonically while ``fan_floor@k`` for ``k > 1``
    RISES — i.e. that the floor alone is gameable and the pair is not.
    """
    _check_fan(fan)
    if not 0.0 <= alpha <= 1.0:
        raise FanFloorError(f"alpha must be in [0, 1], got {alpha}")
    mu = fan.mean(dim=1, keepdim=True)
    return fan + (mu - fan) * float(alpha)


# ---------------------------------------------------------------------------
# The candidate SET's collision rate
# ---------------------------------------------------------------------------

def fan_collision_rate(flag: Tensor, *, rank=None, top_k=None) -> Tensor:
    """Per-window fraction of the candidate SET flagged as colliding.

    ``flag`` is ``[W, K]`` in ``{0, 1}``. With ``rank`` (``[W, K]``, higher =
    preferred) and ``top_k``, the rate is restricted to the ``top_k`` candidates
    the model itself would consider.

    ⭐ WHY BOTH POPULATIONS. ``sel_contact`` — the rate of the ONE selected path —
    measures the SELECTOR, not the generator (`D-RL-GEN-COLLIDES-1`). The rate
    over the whole emitted set is the generator's own property and is the
    quantity a truncated advantage acts on.
    """
    if flag.dim() != 2:
        raise FanFloorError(f"flag must be [W, K], got {tuple(flag.shape)}")
    f = flag.to(torch.float64)
    if top_k is None:
        return f.mean(dim=1)
    if rank is None:
        raise FanFloorError("top_k needs a rank tensor")
    if rank.shape != flag.shape:
        raise FanFloorError(f"rank {tuple(rank.shape)} != flag {tuple(flag.shape)}")
    K = int(flag.shape[1])
    k = min(int(top_k), K)
    idx = torch.topk(rank.to(torch.float64), k, dim=1).indices
    return f.gather(1, idx).mean(dim=1)


# ---------------------------------------------------------------------------
# The readout, and the verdict that refuses to be quoted alone
# ---------------------------------------------------------------------------

@dataclass
class FanReadout:
    """Per-window fan readout. Every field is ``[W]`` so ``taniteval.ci``'s paired
    episode-cluster bootstrap can consume it directly."""
    quality_name: str
    n_windows: int
    n_candidates: int
    floor: dict = field(default_factory=dict)
    rand_best: dict = field(default_factory=dict)
    quantile: dict = field(default_factory=dict)
    diversity: object = None
    endpoint_std: object = None
    collision: dict = field(default_factory=dict)

    def flat(self) -> dict:
        """Flatten to ``{metric_name: [W]}`` for the bootstrap."""
        out = {}
        for k, v in self.floor.items():
            out["fan_floor@%d" % k] = v
        for k, v in self.rand_best.items():
            out["rand_best@%d" % k] = v
        for t, v in self.quantile.items():
            out["fan_q%.2f" % t] = v
        if self.diversity is not None:
            out["fan_diversity"] = self.diversity
        if self.endpoint_std is not None:
            out["fan_endpoint_std"] = self.endpoint_std
        for k, v in self.collision.items():
            out["fan_collision_" + k] = v
        return out


def summarise_fan(fan: Tensor, quality: Tensor, *, quality_name: str,
                  ks=DEFAULT_KS, taus=(0.0, 0.25, 0.5, 0.75),
                  collision_flag=None, rank=None, top_ks=(8, 32),
                  rand_draws: int = 40, seed: int = 0) -> FanReadout:
    """The whole primary readout, per window, with every mandatory control.

    ``rand_best@k`` is computed unconditionally — it is not an option, because
    every ``@k`` number is a max-over-k and the programme has already decided a
    campaign on one that had no matched-random control.
    """
    _check_fan(fan)
    _check_quality(quality)
    assert_quality_admissible(quality_name)
    if fan.shape[0] != quality.shape[0] or fan.shape[1] != quality.shape[1]:
        raise FanFloorError(
            f"fan {tuple(fan.shape)} and quality {tuple(quality.shape)} disagree "
            f"on [W, K]")
    r = FanReadout(quality_name=quality_name,
                   n_windows=int(fan.shape[0]),
                   n_candidates=int(fan.shape[1]))
    r.floor = fan_floor_at_k(quality, ks)
    r.rand_best = random_best_of_k(quality, ks, draws=rand_draws, seed=seed)
    r.quantile = fan_quantile(quality, taus)
    r.diversity = fan_diversity(fan, mode="pairwise_path")
    r.endpoint_std = fan_diversity(fan, mode="endpoint_std")
    if collision_flag is not None:
        r.collision["all"] = fan_collision_rate(collision_flag)
        if rank is not None:
            for tk in top_ks:
                r.collision["top%d" % int(tk)] = fan_collision_rate(
                    collision_flag, rank=rank, top_k=int(tk))
    return r


def floor_verdict(arm: FanReadout, base: FanReadout, *, k: int,
                  diversity_tol: float = 0.30) -> dict:
    """⛔ REFUSES to certify a floor gain without its diversity, by construction.

    Returns a dict carrying the raw deltas and a ``verdict`` in
    ``{"FLOOR-GAIN", "COLLAPSE-SUSPECT", "NO-GAIN", "VOID-K"}``.

    * ``VOID-K`` — the two fans differ in K, so ``@k`` is not comparable at all.
    * ``COLLAPSE-SUSPECT`` — the floor rose **and** diversity fell by more than
      ``diversity_tol`` of the base. ⭐ DD-v2's own published stage sits here
      (diversity −28 %), which is exactly why the verdict is a NAMED OUTCOME and
      not a failure: the trade must be *visible and priced*, never invisible.
    * ``FLOOR-GAIN`` — the floor rose and diversity did not collapse.

    ⚠️ This function returns point deltas ONLY. It carries **no interval and
    therefore no separation claim**: the decision-grade statistic is the paired
    episode-cluster bootstrap over episodes (`taniteval.ci`), and under
    `H-ESTIM-SEED-1` a one-seed separated CI is **necessary, not sufficient**.
    The verdict here is a SHAPE, not a result.
    """
    if arm.n_candidates != base.n_candidates:
        return {"verdict": "VOID-K", "k": int(k),
                "reason": f"K {arm.n_candidates} vs {base.n_candidates}; "
                          f"fan_floor@k is not comparable across K"}
    if k not in arm.floor or k not in base.floor:
        raise FanFloorError(f"@{k} missing from a readout: "
                            f"arm={sorted(arm.floor)} base={sorted(base.floor)}")
    if arm.diversity is None or base.diversity is None:
        raise FanFloorError(
            "floor_verdict REFUSES a verdict without diversity on the same "
            "windows: fan_floor@k is MAXIMISED by mode collapse, so the floor "
            "alone cannot distinguish a better fan from a narrower one.")
    d_floor = float(arm.floor[k].mean() - base.floor[k].mean())
    b_div = float(base.diversity.mean())
    d_div = float(arm.diversity.mean()) - b_div
    d_rand = (float(arm.rand_best[k].mean() - base.rand_best[k].mean())
              if k in arm.rand_best and k in base.rand_best else float("nan"))
    rel_div = d_div / b_div if b_div > 0 else float("nan")
    if d_floor <= 0:
        verdict = "NO-GAIN"
    elif rel_div == rel_div and rel_div < -abs(diversity_tol):
        verdict = "COLLAPSE-SUSPECT"
    else:
        verdict = "FLOOR-GAIN"
    return {"verdict": verdict, "k": int(k),
            "d_floor": d_floor, "d_rand_best": d_rand,
            "d_diversity": d_div, "rel_diversity": rel_div,
            "base_floor": float(base.floor[k].mean()),
            "arm_floor": float(arm.floor[k].mean()),
            "base_diversity": b_div,
            "note": "point deltas only; the decision-grade statistic is the "
                    "paired episode-cluster bootstrap, and one seed is "
                    "necessary-not-sufficient (H-ESTIM-SEED-1)"}
