"""refcv6 ``--max-speed-input-v6``: the PI's FOUR-VALUE set-speed, as a one-hot INPUT.

⭐ WHAT THE PI RULED, 2026-09-16, verbatim: *"Let review the max speed logic, it
should be simple and models a max set speed included as input parameter. It
should few discrete values: 30 kph, 50 kph, 100 kph, 120 kph. I correct my
statement saying the tactical layer must learn max speed. Let use it as input in
our next experiment."* — so the tactical layer does **not** predict it in this
experiment, and nothing here may be reached by a loss.

⛔⛔ WHY THIS IS A SECOND MODULE AND NOT AN EDIT TO
:mod:`tanitad.refs.max_speed_input`. That module ships a **CONTINUOUS**
``[v/V_SCALE, over_ceiling, valid]`` block over an **EIGHT**-step ladder
``{20,30,50,70,80,100,120,130}`` km/h, and it is what refcv5-v2's banked arms and
``--max-speed-input`` were built and stamped against. Rewriting its ladder in
place would silently re-define every banked ``speed_max_derivation`` string —
the exact "a derived constant that changes under its readers" failure the
programme has already paid for twice. This module is ADDITIVE, it carries its
OWN stamp, and its own refusal; the 8-step module is untouched.

⭐ THE ENCODING IS A ONE-HOT, NOT A SCALAR, AND THAT IS THE POINT. A scalar
ceiling is an ordered, differentiable proxy for the raw ego-future speed. Four
one-hot slots destroy the ordering the regressor would otherwise exploit, which
is what makes the obedience test (§5) a test of OBEDIENCE rather than of
regression.

MEASURED HERE, on the locally available v8 train blob (``s2_labels_v8.0_train
.jsonl.gz``, md5 ``fa89ea55dfce68403eb30300e57852ab``, n = 4,572 clips; the
census script is banked beside the RESULT), snapping UP to the four steps:

    step        n      share
    30 km/h   1741    38.08 %
    50 km/h   1593    34.84 %
   100 km/h   1014    22.18 %
   120 km/h    224     4.90 %   (of which 86 clips, 1.88 %, are ABOVE 120)

    snap-up slack  p50 12.32 km/h · p95 44.60 km/h · mean 15.25 km/h
    corpus max v_hi 136.08 km/h
    obedience-test population (v_hi > 40 km/h): 1,962 clips = 42.91 %

⚠️ CROSS-CHECKED against the independently banked numbers in
``max_speed_input.py``'s docstring: "224 clips (4.90 %) have ``v_hi`` > 100
km/h", "38.1 % of clips have ``v_hi`` <= 30 km/h", "the corpus's maximum
``v_hi`` is 37.803 m/s = 136.1 km/h". All three reproduce here to the digit, on
a blob that module never read. That agreement is why the ladder census below is
MEASURED and not merely computed.

⛔⛔ THE CHANNEL IS EGO-FUTURE DERIVED AND THE FOUR-VALUE LADDER DOES NOT
LAUNDER IT. The training value is a function of ``SPEED_BAND.v_hi_ms`` = the max
of the ego's OWN realised speed over ``[t0+2 s, +6 s]``. Quantizing to four
steps leaves H = **1.7555 bits** (MEASURED from the shares above) — less than
the 8-step ladder's 2.480 bits, but not zero, and ``(bin, v0)`` still
narrows the raw value. The PI authorised it as a declared oracle INPUT standing
in for a map/nav set-speed service. ⇒ :func:`assert_speed_max_stamp_v6` REFUSES
TO START a run that would not carry the declaration, **and refuses the mirror
case** — a control stamped as conditioned. Both directions are proven by
mutation in ``stack/tests/test_refcv6_tactical.py``.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn

__all__ = [
    "SPEED_MAX_STEPS_KMH_V6", "SPEED_MAX_STEPS_MS_V6", "N_SPEED_MAX_BINS_V6",
    "SPEED_MAX_DERIVATION_V6", "SPEED_MAX_STAMP_REQUIRED_V6",
    "SPEED_MAX_CONTROL_UNITS",
    "speed_max_bin", "speed_max_bin_tensor", "speed_max_onehot",
    "limit_ms_of_bin", "assert_speed_max_stamp_v6", "MaxSpeedOneHotEncoder",
    "SpeedMaxStampError", "ladder_census",
]


class SpeedMaxStampError(SystemExit):
    """Raised (as a ``SystemExit``) when a run's provenance stamp is wrong.

    ⭐ A ``SystemExit`` subclass on purpose: the existing
    ``refc_v3_train._assert_speed_max_stamp`` raises ``SystemExit`` and the
    trainer's ``main`` must die the same way whichever guard fires. It stays a
    distinct CLASS so a test can assert WHICH guard refused instead of matching
    on message text.
    """


#: ⛔ THE PI'S FOUR VALUES, LITERALLY AND IN ORDER. Not a quantile, not a fit —
#: the PI named them. Ascending, and the index into this tuple IS the one-hot
#: slot, so APPEND/REORDER IS A CHECKPOINT BREAK (the ``TACTICAL_LAT_ACTIONS``
#: contract, same reason).
SPEED_MAX_STEPS_KMH_V6: tuple[int, ...] = (30, 50, 100, 120)

#: The same ladder in SI. ⛔ Derived here ONCE; never re-divided at a call site.
SPEED_MAX_STEPS_MS_V6: tuple[float, ...] = tuple(
    s / 3.6 for s in SPEED_MAX_STEPS_KMH_V6)

N_SPEED_MAX_BINS_V6: int = len(SPEED_MAX_STEPS_KMH_V6)

#: What every speed in this module MEANS. Same discipline as
#: ``max_speed_input.CONTROL_UNITS`` and ``anchor_meta.CONTROL_UNITS``: m/s vs
#: km/h is a 3.6x spread and this programme has published a wrong table from
#: exactly that.
SPEED_MAX_CONTROL_UNITS = "m_s"

#: ⛔ THE STAMP. Written into ``config.json``; a run without it is REFUSED.
#: It names the SOURCE FIELD, the WINDOW, the word ``oracle``, the LADDER and
#: the ROLE, so a reader who opens ``config.json`` in isolation learns what fed
#: the channel without going to find the code.
SPEED_MAX_DERIVATION_V6 = (
    "refcv6 4-way one-hot set-speed, CONTAINING-WINDOW quantization over "
    "{30, 50, 100, 120} km/h: (0,30]->30, (30,50]->50, (50,100]->100, "
    "(100,120]->120, and > 120 CLAMPS to 120 (counted and reported). Source "
    "field v7.2/v8 g_tac.goals.SPEED_BAND.v_hi_ms (oracle, provenance "
    "ego-future: max of the ego's OWN REALISED speed over [t0+2 s, +6 s]); the "
    "PI authorised the future-ego derivation explicitly on 2026-09-16 provided "
    "the value lands in the containing window. It models a max SET speed - a "
    "limiter the driver sets - so the realised speed sits INSIDE the window and "
    "never above it. INPUT, never a training signal: no loss reaches this "
    "channel in refcv6.")

#: ⛔ The tokens :func:`assert_speed_max_stamp_v6` requires to be PRESENT.
#: Written as LITERALS, never as an expression over the constant above — a check
#: derived from the value it checks is green forever, which is exactly how a
#: label builder once verified its buckets against its own rounded ladder and
#: passed while 57.5 % of the corpus was wrong.
SPEED_MAX_STAMP_REQUIRED_V6: tuple[str, ...] = (
    "oracle", "ego-future", "SPEED_BAND.v_hi_ms", "[t0+2 s, +6 s]",
    "{30, 50, 100, 120} km/h", "4-way one-hot", "CONTAINING-WINDOW",
)


# --- the ladder ------------------------------------------------------------

def speed_max_bin(v_ms: float) -> tuple[int, bool]:
    """``(bin_index, over_ceiling)`` for ONE realised maximum speed, in m/s.

    ⭐ THE CONTAINING-WINDOW RULE (PI 2026-09-16): the bin is the window that
    CONTAINS ``v`` — ``(0, 30] -> 30``, ``(30, 50] -> 50``, ``(50, 100] -> 100``,
    ``(100, 120] -> 120`` — which is the same arithmetic as *"the smallest step
    that is >= v"* and is stated both ways because the PI stated it as a window.
    ⛔ It is NOT a nearest-value rule: a set speed is a LIMITER, so the realised
    speed must sit INSIDE the window and never above it. 96 km/h binning to 100
    is correct; binning it to the nearer-looking value would hand the model a
    ceiling the ego demonstrably exceeded.

    A value above the top step is CLAMPED to 120 and flagged
    ``over_ceiling=True`` — it is NOT given an invented fifth step.

    ⚠️ ``over_ceiling`` is not cosmetic. MEASURED: 86 of 4,572 clips (1.88 %)
    sit above 120 km/h, so on those windows the fed ceiling is one a real
    vehicle would have EXCEEDED. Reporting it is what stops the obedience test
    from silently scoring 1.0 on a population where obedience was impossible.
    """
    v = float(v_ms)
    if not math.isfinite(v):
        raise ValueError(
            f"[refcv6-vmax] non-finite realised speed {v_ms!r}. A NaN binned "
            f"silently becomes bin 0 = 30 km/h, i.e. the SLOWEST ceiling, and "
            f"a whole batch of them would read as 'the model obeys 30 km/h "
            f"everywhere'.")
    for i, s in enumerate(SPEED_MAX_STEPS_MS_V6):
        if v <= s + 1e-9:
            return i, False
    return N_SPEED_MAX_BINS_V6 - 1, True


def speed_max_bin_tensor(v_ms: Tensor) -> tuple[Tensor, Tensor]:
    """Batched :func:`speed_max_bin`. ``v_ms [..]`` -> ``(bin [..] long,
    over_ceiling [..] bool)``. Vectorised, and byte-identical to the scalar
    function (fuzzed in the test module)."""
    if not torch.is_tensor(v_ms):
        raise TypeError(f"v_ms must be a Tensor, got {type(v_ms).__name__}")
    if not torch.isfinite(v_ms).all():
        raise ValueError(
            "[refcv6-vmax] non-finite realised speed in the batch — see "
            "`speed_max_bin` for why this refuses instead of binning to 0.")
    steps = torch.tensor(SPEED_MAX_STEPS_MS_V6, device=v_ms.device,
                         dtype=v_ms.dtype)
    ge = v_ms.unsqueeze(-1) <= steps + 1e-9                 # [..., K]
    over = ~ge.any(dim=-1)
    idx = torch.where(over,
                      torch.full_like(over, N_SPEED_MAX_BINS_V6 - 1,
                                      dtype=torch.long),
                      ge.to(torch.long).argmax(dim=-1))
    return idx, over


def speed_max_onehot(bin_idx: Tensor, valid: Tensor | None = None,
                     dtype: torch.dtype = torch.float32) -> Tensor:
    """``bin [B] long`` -> ``[B, 4]`` one-hot, zeroed where ``valid`` is false.

    ⛔ AN INVALID ROW IS ALL-ZERO, NOT BIN 0. "No set-speed known" and "the set
    speed is 30 km/h" are DIFFERENT inputs, and collapsing them is the X15
    defect: a withheld block must be exactly zeros, never a plausible value.
    The model can tell them apart because an all-zero row sums to 0 while every
    valid row sums to 1 — no companion flag is needed and none is invented.
    """
    if bin_idx.dtype not in (torch.long, torch.int32, torch.int64):
        raise TypeError(
            f"[refcv6-vmax] bin_idx must be integral, got {bin_idx.dtype}. A "
            f"float index silently floors and 1.999 becomes bin 1.")
    if int(bin_idx.min()) < 0 or int(bin_idx.max()) >= N_SPEED_MAX_BINS_V6:
        raise ValueError(
            f"[refcv6-vmax] bin_idx out of range [0, {N_SPEED_MAX_BINS_V6}): "
            f"min {int(bin_idx.min())} max {int(bin_idx.max())}")
    oh = torch.zeros(*bin_idx.shape, N_SPEED_MAX_BINS_V6,
                     device=bin_idx.device, dtype=dtype)
    oh.scatter_(-1, bin_idx.unsqueeze(-1), 1.0)
    if valid is not None:
        oh = oh * valid.reshape(*bin_idx.shape, 1).to(dtype)
    return oh


def limit_ms_of_bin(bin_idx: Tensor) -> Tensor:
    """``bin [..] long`` -> the SPEED LIMIT that bin means, in m/s.

    This is what the obedience test compares a plan's realised maximum against,
    and what the selection speed mask is built from.
    """
    steps = torch.tensor(SPEED_MAX_STEPS_MS_V6, device=bin_idx.device,
                         dtype=torch.float32)
    return steps[bin_idx.reshape(-1).long()].reshape(bin_idx.shape)


@torch.no_grad()
def ladder_census(v_hi_ms: Sequence[float]) -> dict:
    """The per-bin shares, the snap-up slack and the over-ceiling count.

    Goes into ``config.json`` beside the stamp, so an arm's record says what the
    channel's distribution actually WAS on the split it trained on rather than
    quoting this module's docstring.
    """
    n = len(v_hi_ms)
    counts = [0] * N_SPEED_MAX_BINS_V6
    over = 0
    slack = []
    for v in v_hi_ms:
        i, ov = speed_max_bin(v)
        counts[i] += 1
        over += int(ov)
        slack.append(SPEED_MAX_STEPS_KMH_V6[i] - float(v) * 3.6)
    slack.sort()
    ent = 0.0
    for c in counts:
        if c:
            p = c / n
            ent -= p * math.log2(p)
    return {
        "n": n,
        "steps_kmh": list(SPEED_MAX_STEPS_KMH_V6),
        "counts": counts,
        "shares": [round(c / n, 6) if n else None for c in counts],
        "over_ceiling": over,
        "over_ceiling_share": round(over / n, 6) if n else None,
        "slack_kmh_p50": round(slack[n // 2], 4) if n else None,
        "slack_kmh_p95": round(slack[int(0.95 * n)], 4) if n else None,
        "entropy_bits": round(ent, 4),
        "_units": SPEED_MAX_CONTROL_UNITS,
        "_derivation": SPEED_MAX_DERIVATION_V6,
    }


# --- the refusal -----------------------------------------------------------

def assert_speed_max_stamp_v6(cfg_dict: dict, on: bool) -> None:
    """⛔ REFUSE a refcv6 4-way max-speed run whose config does not DECLARE the
    channel's provenance — **and refuse the mirror**, a control that carries the
    stamp while the channel was off.

    Deliberately the same shape as ``refc_v3_train._assert_speed_max_stamp``
    (the pattern the task names), with three differences that matter:

    1. it keys on ``speed_max_derivation_v6``, so the 8-step channel's stamp
       cannot satisfy the 4-way channel's guard **or vice versa** — two
       different ladders sharing one key is how an arm ends up describing a
       ceiling it never fed;
    2. the required tokens include the LADDER and the word ``4-way one-hot``,
       because the ladder is the thing that changed;
    3. ``on`` is passed EXPLICITLY rather than read off ``args``, so the guard
       is callable from a test with no argparse namespace and its two branches
       are independently mutable.

    ⭐ WHY A REFUSAL AND NOT A DEFAULT: the PI authorised an EGO-FUTURE input on
    the axis that owns most of the oracle gap. That is defensible exactly as
    long as every artifact says so. A stamp that can be silently dropped is not
    a stamp; a stamp that can be silently ADDED manufactures a
    max-speed-conditioned arm out of a control.
    """
    stamp = cfg_dict.get("speed_max_derivation_v6")
    if not on:
        if stamp is not None:
            raise SpeedMaxStampError(
                "[refcv6] ⛔ config carries `speed_max_derivation_v6` but the "
                "4-way max-speed input is OFF. A run that did not feed a set "
                "speed must not be stamped as one — that is the MIRROR of the "
                "missing-stamp failure and it manufactures a "
                "max-speed-conditioned arm out of a control. Drop the stamp or "
                "turn the channel on.")
        return
    if not isinstance(stamp, str) or not stamp.strip():
        raise SpeedMaxStampError(
            "[refcv6] ⛔ the 4-way max-speed input is ON but this run's "
            "config.json would carry no `speed_max_derivation_v6`. The "
            "channel's value is derived from the ego's OWN FUTURE speed over "
            "the horizon being scored; the PI authorised it as a declared "
            "oracle INPUT, and the declaration is the condition. Refusing to "
            "start rather than banking an arm whose record cannot say what fed "
            "it.")
    missing = [t for t in SPEED_MAX_STAMP_REQUIRED_V6 if t not in stamp]
    if missing:
        raise SpeedMaxStampError(
            f"[refcv6] ⛔ 4-way max-speed input: `speed_max_derivation_v6` is "
            f"present but does not declare {missing}. The stamp exists so a "
            f"reader who opens config.json in ISOLATION learns the source "
            f"field, the window, the LADDER and the word `oracle` without "
            f"going to find the code. A stamp missing any of those is a key, "
            f"not a declaration. Got: {stamp!r}")


# --- the conditioner -------------------------------------------------------

class MaxSpeedOneHotEncoder(nn.Module):
    """``[B] realised-max speed (m/s) -> [B, 4] one-hot``, as a module so the
    ladder travels with the checkpoint's config rather than with a call site.

    ⛔ IT HAS NO PARAMETERS. The embedding of this one-hot belongs to whatever
    consumes it (the tactical FiLM, the decoder condition), so the SAME one-hot
    reaches every consumer and two halves of one switch cannot drift.
    """

    def __init__(self) -> None:
        super().__init__()
        self.steps_kmh = SPEED_MAX_STEPS_KMH_V6
        self.n_bins = N_SPEED_MAX_BINS_V6

    @property
    def n_params(self) -> int:
        return 0

    def forward(self, v_max_ms: Tensor, valid: Tensor | None = None
                ) -> tuple[Tensor, Tensor]:
        """``(onehot [B, 4], over_ceiling [B] bool)``."""
        idx, over = speed_max_bin_tensor(v_max_ms.reshape(-1).float())
        return speed_max_onehot(idx, valid, dtype=v_max_ms.dtype), over
