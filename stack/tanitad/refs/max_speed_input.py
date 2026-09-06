"""``--max-speed-input``: a MAP/NAV SPEED-LIMIT channel, quantized to real posted limits.

⭐ WHAT THIS IS AND WHY IT IS ADMISSIBLE. PI 2026-09-01, reaffirmed 2026-09-06:
*"regarding max speed, introduce an input in the models which corresponds the
upper band value of speed band"* and *"what is the problem with max speed
acquisition, take just the upper bound of the speed band"*, then *"good idea with
quantization"*. ``v_hi_ms`` STANDS IN FOR A MAP/NAV SPEED-LIMIT SERVICE exactly as
``nav_command`` stands in for the nav system, and the PI has ruled that class is an
INPUT, not a training signal. We do not have a map; this module is the simulator
for one. The channel therefore lives beside ``v0`` and ``nav_cmd``, never beside a
loss.

⛔⛔ THE DEFECT THIS MODULE EXISTS TO FIX, MEASURED, NOT ARGUED. The source value is
``g_tac.goals.SPEED_BAND.v_hi_ms`` = ``max`` of the ego's ACTUAL speed over
``[anchor+2 s, anchor+6 s]`` (``stack/scripts/s2_geom_emit_v7.py`` :199-200 / :211 /
:309-315; ``poses[:, 3]`` is m/s, ``tanitad.data.egomotion_source``:65). So the
"ceiling" is **OPTIMISTIC BY CONSTRUCTION** — it is the speed the car actually
reached, hence always exactly achievable and NEVER violated. A real posted limit
is not like that: a driver may sit far below it, and may exceed it.

MEASURED on the full v8 train corpus, n = 4,572 (join re-derived from egomotion and
checked against the label: 4,572/4,572 identical, 0 mismatches):
    corr(v_hi, v0)  = 0.941 (n=1,200 panel) / see the package for the n=4,572 refit
    v_hi - v0       : mean +1.08, median +0.42, p95 +6.12 m/s
    v_hi differs from v0 by > 1 m/s on 51.5 % of clips

⭐⭐ THE PINNED STEP SET, AND WHY IT IS THESE VALUES

Two rules, and BOTH are needed. Values come from ROAD LAW; membership comes from
the CORPUS. A ladder fitted to the corpus's speed histogram would re-encode exactly
the information quantization is meant to destroy, so no step here is a quantile.

    RULE 1 (admissibility -- road law).  Every step must be a real posted limit in
    the corpus's countries. The v8 corpus is 24 European countries + the United
    States (measured: ``strata.country``, 25 distinct values, largest share 6.3 %).

    RULE 2 (necessity -- the corpus decides WHICH of those real limits are kept).
    A candidate step is kept only if dropping it either (a) leaves a bin holding an
    unreasonable share of the corpus, or (b) widens the p95 snap-up slack.

MEASURED, snapping UP (``q(v) = min{s in S : s >= v}``), n = 4,572:

    ladder                          k   H(bits)  max bin  slack p50  slack p95  over-ceiling
    {30,50,70,100}      (PI start)  4    1.841    38.1 %   10.4 km/h  25.4 km/h    4.90 %
    {30,50,70,100,130}              5    1.969    38.1 %   11.1 km/h  26.3 km/h    0.61 %
    {20,30,50,70,80,100,120,130}    8    2.480    34.9 %    8.1 km/h  19.4 km/h    0.61 %   <-- PINNED
    {..,+90,+110}                  10    2.559    34.9 %    7.5 km/h  19.2 km/h    0.61 %
    {10,20,...,130} (every 10)     13    3.203    20.4 %    4.9 km/h   9.7 km/h    0.61 %

⇒ THE ANSWER TO "does the corpus need 20/80/120 as well?" is **YES to all three**:
  * **20** is needed because 38.1 % of clips have ``v_hi <= 30 km/h`` (the ego is
    stopped or crawling at intersections; p1 of ``v_hi`` is 0.00 m/s). Adding the
    20 km/h step splits that single 38.1 % mass into 17.7 % / 20.4 %.
  * **80** and **120** are needed because the PI's ladder has 30 km/h-wide gaps
    above 70, where the corpus is thin but real: they cut the p95 slack from
    25.4 to 19.4 km/h.
  * **A step above 100 is mandatory**: 224 clips (4.90 %) have ``v_hi > 100 km/h``.

⇒ AND THE TWO REJECTIONS, ALSO MEASURED:
  * **90 and 110 are excluded.** Adding them moves the recoverability R^2 by
    +0.0009 and the median slack by 0.6 km/h. They buy nothing and cost resolution
    (every extra step makes the bin a better proxy for the raw value).
  * **The every-10 ladder is excluded.** At 13 steps the raw value is recoverable
    from ``(bin, v0)`` at R^2 = 0.9884, residual 0.80 m/s -- at that resolution the
    bin IS the raw value in disguise, i.e. the quantization would be COSMETIC.

⭐ THE TOP STEP IS 130 km/h AND IT IS NOT A CLAMP-TO-COVER. 130 is the highest
generally-posted limit in the corpus's countries (FR/AT/CZ/HU/PL/RO/BG/HR/SI
motorways). The corpus's maximum ``v_hi`` is 37.803 m/s = 136.1 km/h, so **28 clips
(0.61 %) sit ABOVE the ceiling this module reports**. That is deliberate and is the
single most important behavioural difference from the raw value: a real limit CAN
be exceeded, and :func:`quantize_up` reports that as ``over_ceiling=True`` rather
than inventing a 140 km/h sign that exists nowhere. The ego-derived raw value is
violated on 0/4,572 clips by construction; the quantized channel is violated on 28.

⛔ UNITS ARE DECLARED, NEVER INFERRED. Everything here is METRES PER SECOND and says
so in :data:`CONTROL_UNITS`. :func:`read_max_speed_field` REFUSES a payload that
declares no units, for the reason ``tanitad.refs.anchor_meta`` refuses one: m/s vs
km/h vs mph is a 1.61x spread, and this programme published a 396 g anchor table
from exactly this error. A declared non-SI unit is CONVERTED and the conversion is
recorded; an UNDECLARED one raises.

⛔ THE OFF PATH IS BIT-IDENTICAL. :class:`MaxSpeedConditioner` is additive and
zero-init at the output projections, mirroring E13 (nav) and E11' (ego) verbatim, so
a build with the flag ON is bit-identical to one with it OFF at step 0 -- and the
accompanying test proves that assertion can FAIL by mutating the projections
(``stack/tests/test_max_speed_input.py``). An equality that cannot fail proves
nothing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

# --- units -------------------------------------------------------------------

#: What every speed in this module MEANS. Written into every artifact; required
#: from every payload. Deliberately the same spelling discipline as
#: ``tanitad.refs.anchor_meta.CONTROL_UNITS``.
CONTROL_UNITS = "m_s"

#: Accepted unit spellings -> multiplier INTO m/s. A payload may declare km/h or
#: mph; it may not decline to declare.
UNIT_TO_MS: dict[str, float] = {
    "m_s": 1.0, "m/s": 1.0, "ms-1": 1.0, "mps": 1.0, "meters_per_second": 1.0,
    "km_h": 1.0 / 3.6, "km/h": 1.0 / 3.6, "kmh": 1.0 / 3.6, "kph": 1.0 / 3.6,
    "mph": 0.44704, "mi_h": 0.44704, "mi/h": 0.44704,
}

#: One sentence a refusal can quote, so the reader knows what the rule costs.
INCIDENT = (
    "MEASURED 2026-09-04 on refcv4b's live anchors.pt: a controls column whose "
    "units the file did not declare, read as the wrong quantity, produced a "
    "396 g anchor table that looked exactly like an answer. m/s vs km/h vs mph "
    "is a 1.61x spread; a speed ceiling read in the wrong unit is the same "
    "failure with a speedometer on it.")


class MaxSpeedUnitsError(ValueError):
    """Base: the payload's speed units cannot be established."""


class MaxSpeedUnitsMissing(MaxSpeedUnitsError):
    """The payload carries a speed and declares no units."""


class MaxSpeedUnitsUnknown(MaxSpeedUnitsError):
    """The payload declares units this module does not recognise."""


# --- the pinned ladder -------------------------------------------------------

#: ⭐ THE PINNED STEP SET, in km/h. Derived by the two rules in the module
#: docstring: admissible values from road law, membership from the corpus.
#: DO NOT change these without re-running
#: ``.../Research/2026-09-06-max-speed-input/code/quantization_panel.py`` and
#: re-reading the three numbers it prints -- the whole point of pinning is that
#: the ladder is a decision with evidence attached, not a default.
POSTED_LIMIT_STEPS_KMH: tuple[int, ...] = (20, 30, 50, 70, 80, 100, 120, 130)

#: The same ladder in m/s -- the ONLY unit anything downstream sees.
POSTED_LIMIT_STEPS_MS: tuple[float, ...] = tuple(
    k / 3.6 for k in POSTED_LIMIT_STEPS_KMH)

#: Normalisation divisor for the model-facing channel: the TOP step, so the
#: normalised value lands in (0, 1] for every in-ladder speed and slightly above
#: 1 is impossible (an over-ceiling clip is clamped to the top step and flagged).
#: Stated as a constant so an arm that changes it must say that it did.
V_SCALE_MS: float = POSTED_LIMIT_STEPS_MS[-1]

#: The two modes behind the single ``--max-speed-input`` flag.
MODES = ("quantized", "raw")
DEFAULT_MODE = "quantized"

#: Width of the model-facing block: (value_norm, over_ceiling, valid).
MAX_SPEED_DIMS = 3

#: ⛔ THE VALIDITY SLOT IS NOT OPTIONAL, for the reason ``nav_args``' third slot is
#: not optional (``refc_v3.NAV_ARG_DIMS``): a silent 0.0 in the value channel reads
#: as "the limit here is 0 m/s -- stop", which is a LIE, not a missing value.
#: "No limit known" must be a distinct, learnable input.
VALID_SLOT = MAX_SPEED_DIMS - 1


# --- the quantizer -----------------------------------------------------------

def quantize_up(v_ms: float, steps_ms: tuple[float, ...] = POSTED_LIMIT_STEPS_MS
                ) -> tuple[float, bool]:
    """Snap ``v_ms`` UP to the next posted limit. Returns ``(q_ms, over_ceiling)``.

    ``q = min{s in steps : s >= v}``. Above the top step there is no larger real
    sign, so ``q`` is the top step and ``over_ceiling`` is True -- the clip is
    genuinely OVER the limit this channel reports. That case is the whole point:
    it is the property a posted limit has and the ego-derived value does not.
    """
    if not (isinstance(v_ms, (int, float)) and math.isfinite(v_ms)):
        raise ValueError(f"v_ms must be a finite number, got {v_ms!r}")
    for s in steps_ms:
        if v_ms <= s:
            return float(s), False
    return float(steps_ms[-1]), True


def quantize_up_array(v_ms, steps_ms: tuple[float, ...] = POSTED_LIMIT_STEPS_MS):
    """Vectorised :func:`quantize_up`. Returns ``(q, over_ceiling)`` arrays.

    numpy is imported lazily so the scalar path and the nn.Module stay importable
    in a torch-only environment.
    """
    import numpy as np
    v = np.asarray(v_ms, dtype=np.float64)
    if not np.isfinite(v).all():
        raise ValueError("quantize_up_array: non-finite speed in input")
    s = np.asarray(steps_ms, dtype=np.float64)
    idx = np.searchsorted(s, v, side="left")
    over = idx >= len(s)
    return s[np.clip(idx, 0, len(s) - 1)], over


# --- the consumer: units are REQUIRED ---------------------------------------

def read_max_speed_field(payload: dict, *, mode: str = DEFAULT_MODE,
                         units_override: str | None = None) -> dict:
    """Read ``speed_max_input`` (or any dict carrying a max speed) into m/s.

    ⛔ REFUSES a payload that carries a speed and declares no units. There is a
    named override so a genuine legacy artifact can still be loaded, and taking it
    is RECORDED in the result as ``units_source = "caller-override"`` -- the
    ``anchor_meta`` contract verbatim, so the run record says the units came from
    the operator rather than from the artifact.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    v_raw = payload.get("v_max_ms", payload.get("v_max"))
    if v_raw is None:
        return {"value_ms": None, "valid": False, "reason": "no v_max field",
                "control_units": CONTROL_UNITS, "mode": mode}
    declared = payload.get("units", payload.get("control_units"))
    if declared is None:
        if units_override is None:
            raise MaxSpeedUnitsMissing(
                "speed_max_input payload carries a max speed and declares no "
                "`units`. Refusing to guess. Pass units_override=... only for a "
                "genuine legacy artifact, and the choice is recorded. " + INCIDENT)
        source, declared = "caller-override", units_override
    else:
        source = "artifact"
    key = str(declared).strip().lower()
    if key not in UNIT_TO_MS:
        raise MaxSpeedUnitsUnknown(
            f"unrecognised speed units {declared!r}; known: "
            f"{sorted(set(UNIT_TO_MS))}. " + INCIDENT)
    v_ms = float(v_raw) * UNIT_TO_MS[key]
    if mode == "raw":
        return {"value_ms": v_ms, "quantized_ms": None, "over_ceiling": False,
                "valid": True, "control_units": CONTROL_UNITS, "mode": mode,
                "units_declared": declared, "units_source": source}
    q, over = quantize_up(v_ms)
    return {"value_ms": q, "raw_ms": v_ms, "quantized_ms": q,
            "over_ceiling": bool(over), "valid": True,
            "control_units": CONTROL_UNITS, "mode": mode,
            "units_declared": declared, "units_source": source}


def artifact_meta(mode: str = DEFAULT_MODE) -> dict:
    """The block a builder writes into config.json / any emitted artifact.

    ``control_units`` is present and non-empty in every branch: a consumer that
    finds this block may read the number; one that does not must refuse.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return {
        "schema": "tanitad.max_speed_input/1",
        "control_units": CONTROL_UNITS,
        "mode": mode,
        "steps_kmh": list(POSTED_LIMIT_STEPS_KMH) if mode == "quantized" else None,
        "steps_ms": [round(s, 6) for s in POSTED_LIMIT_STEPS_MS]
                    if mode == "quantized" else None,
        "rule": ("snap UP to the next posted limit; above the top step the value "
                 "is the top step and over_ceiling is True"
                 if mode == "quantized" else "unquantized float, comparison only"),
        "v_scale_ms": V_SCALE_MS,
        "dims": MAX_SPEED_DIMS,
        "dim_names": ["value_norm", "over_ceiling", "valid"],
        "source_field": "speed_max_input.v_max_ms (= g_tac.goals.SPEED_BAND.v_hi_ms)",
        "provenance": "ego-future (stands in for a map/nav speed-limit service)",
        "role": "INPUT CHANNEL, user/nav-supplied at inference, like nav_command",
    }


# --- the model-facing block --------------------------------------------------

def encode_block(v_ms, valid=None, *, mode: str = DEFAULT_MODE,
                 device=None, dtype=None) -> Tensor:
    """``[B, 3]`` = ``(value_norm, over_ceiling, valid)``.

    ⛔ The validity bit GATES the value inside this function as well as in the
    conditioner (X15's rule: the consumer re-applies the flag), so a caller that
    forgot to zero an invalid row cannot leak a phantom ceiling.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    v = torch.as_tensor(v_ms, dtype=dtype or torch.float32, device=device).reshape(-1)
    ok = (torch.ones_like(v) if valid is None
          else torch.as_tensor(valid, dtype=v.dtype, device=v.device).reshape(-1))
    if ok.shape != v.shape:
        raise ValueError(f"valid must match v_ms, got {tuple(ok.shape)} vs "
                         f"{tuple(v.shape)}")
    # a row that is not valid must not carry a finite ceiling into the quantizer
    v = torch.where(ok > 0, v, torch.zeros_like(v))
    if mode == "quantized":
        steps = torch.as_tensor(POSTED_LIMIT_STEPS_MS, dtype=v.dtype,
                                device=v.device)
        idx = torch.searchsorted(steps, v.contiguous(), right=False)
        over = (idx >= steps.numel()).to(v.dtype)
        q = steps[idx.clamp(max=steps.numel() - 1)]
    else:
        q, over = v, torch.zeros_like(v)
    return torch.stack([(q / V_SCALE_MS) * ok, over * ok, ok], dim=-1)


@dataclass
class MaxSpeedConfig:
    """⚠️ ``enabled`` DEFAULTS FALSE: no banked arm's recipe changes."""
    enabled: bool = False
    mode: str = DEFAULT_MODE
    d_speed: int = 32          # ~= d_ego's 32; nav uses 64 for 4 tokens


class MaxSpeedConditioner(nn.Module):
    """Additive, zero-init injection of the speed ceiling into {tactical, strategic}.

    Structurally identical to E13's ``nav_to_tac`` / ``nav_to_str`` and E11's
    ``ego_to_tac`` / ``ego_to_str``: the ceiling enters exactly where ``v0`` and
    the nav command already enter, and nowhere else.
    """

    def __init__(self, d_tac: int, d_ctx: int, cfg: MaxSpeedConfig | None = None):
        super().__init__()
        self.cfg = cfg or MaxSpeedConfig(enabled=True)
        if self.cfg.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.cfg.mode!r}")
        self.proj = nn.Linear(MAX_SPEED_DIMS, self.cfg.d_speed)
        self.to_tac = nn.Linear(self.cfg.d_speed, d_tac)
        self.to_str = nn.Linear(self.cfg.d_speed, d_ctx)
        for lin in (self.to_tac, self.to_str):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, v_max_ms: Tensor | None, valid: Tensor | None = None
                ) -> tuple[Tensor | None, Tensor | None]:
        """``(delta_tac, delta_str)``, or ``(None, None)`` when nothing was fed."""
        if v_max_ms is None:
            return None, None
        w = self.proj.weight
        blk = encode_block(v_max_ms, valid, mode=self.cfg.mode,
                           device=w.device, dtype=w.dtype)
        if blk.shape[-1] != MAX_SPEED_DIMS:
            raise ValueError(f"block must be [B, {MAX_SPEED_DIMS}] = "
                             f"(value_norm, over_ceiling, valid), got "
                             f"{tuple(blk.shape)}")
        # re-apply the validity bit AFTER the first projection's bias, for the same
        # reason `ego_inj` re-applies `keep`: a withheld row must be exactly the
        # zero vector next to a valid bit of 0, never a bias-shifted leftover.
        e = self.proj(blk) * blk[:, VALID_SLOT:VALID_SLOT + 1]
        return self.to_tac(e), self.to_str(e)
