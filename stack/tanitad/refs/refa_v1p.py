"""REF-A v1′ — REF-A v1 with the action carried as TOKENS in the attention stream.

⭐ WHAT THIS IS AND IS NOT. It is an **ALTERNATIVE ARM, parked**, not a
replacement: `refa_v1.py` is untouched and remains the reference. Everything here
is a **single-axis delta** — the adapter, the three-rate hierarchy, the brains,
the factored tactical heads, the planner and the whole training objective are
INHERITED unchanged, so a v1 vs v1′ comparison isolates exactly one design
choice and nothing else.

## The one difference

`TokenFieldPredictor` (v1) conditions on the action by **broadcast-and-concat** —
DINO-WM's exact scheme::

    a = act(action)                      # [B, d]
    a = a[:, None].expand(-1, N, -1)     # broadcast over all N tokens
    x = mix(cat([field, a], -1))         # concatenate, project back to d

`ActionStreamPredictor` (v1′) instead makes the action **tokens in the shared
self-attention stream**, which is SimWAM's layout (2608.07468)::

    at = act_split(act(action))          # [B, n_act, d]
    x  = cat([field, at], dim=1)         # ONE stream, N + n_act tokens
    ... attention ...
    x  = x[:, :N]                        # read the FIELD tokens back out

The action is no longer a per-token additive context; it is a participant the
field tokens attend to and which attends back.

## Why it is worth a parked arm — MEASURED, and with its limits stated

`E-ACTSTREAM-1` (`…/2026-08-19-simwam-analysis/E_ACTSTREAM_1.md`) ran both
schemes on banked v6 cell fields, **parameter-matched to 576 params (0.04 %)**,
3 seeds, episode-disjoint split, paired episode-cluster bootstrap:

| width | concat (v1) | token (v1′) | paired Δ | separated |
|---|---|---|---|---|
| d=192 L=4 | 0.000086 | **0.000013** | −0.000073 [−0.000087, −0.000060] | ✅ |
| d=48 L=2 | 0.000443 | **0.000042** | −0.000400 [−0.000476, −0.000328] | ✅ |
| d=32 L=2 | 0.000607 | **0.000166** | −0.000440 [−0.000543, −0.000341] | ✅ |

⛔ **AND THE LIMIT, WHICH TRAVELS WITH THE RESULT: on that probe cache NEITHER
scheme beat C-PERSIST (copy the last field).** The token arm beat the mean-field
baseline and the concat arm did not, but both sat above persistence. So what is
licensed is *"the joint stream extracts more from the same parameters"* — **not**
that either models the dynamics. This arm exists to test whether that relative
advantage survives at real geometry and real data; it is not evidence that it
will.

## What this costs

Attention runs over ``N + n_act_tokens`` instead of ``N``. At v1's real geometry
(640 tokens) with ``n_act_tokens=2`` that is **642 vs 640 — +0.6 % attention
cost** and no change to the rollout's memory profile, so the ``last_only``
requirement inherited from :meth:`TokenFieldPredictor.rollout` still holds
unchanged. ⚠️ ``rollout`` is deliberately NOT overridden: duplicating it would
let the 3.9 GB memory rule drift out of sync between the two arms.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from tanitad.refs.refa_v1 import RefAV1, RefAV1Config, TokenFieldPredictor

__all__ = ["ActionStreamPredictor", "RefAV1PrimeConfig", "RefAV1Prime"]


@dataclass
class RefAV1PrimeConfig(RefAV1Config):
    """v1's config plus the single new knob.

    ⚠️ ``n_act_tokens`` is 2 by DEFAULT because that is the value E-ACTSTREAM-1
    measured at PARAMETER PARITY with the concat arm. Raising it buys capacity
    as well as structure, and a win at n=4 would no longer isolate the design
    choice — which is the only thing this arm exists to test.
    """

    n_act_tokens: int = 2

    def sanity(self) -> None:
        super().sanity()
        if self.n_act_tokens < 1:
            raise ValueError(
                f"n_act_tokens must be >= 1, got {self.n_act_tokens}: zero "
                f"action tokens is not 'v1' — it is a predictor the action "
                f"cannot reach at all, which trains a scene extrapolator "
                f"wearing a world model's name.")


class ActionStreamPredictor(TokenFieldPredictor):
    """v1's predictor with the action as TOKENS rather than broadcast context.

    ⭐ Subclasses rather than forks: ONLY :meth:`step` changes. ``rollout`` —
    and with it the ``last_only`` memory requirement measured at 3.9 GB for a
    300-candidate CEM population — is inherited verbatim, so the two arms can
    never drift apart on the expensive path.
    """

    def __init__(self, cfg: RefAV1Config, d: int, layers: int, heads: int = 8,
                 intent_dim: int | None = None, n_act_tokens: int = 2):
        super().__init__(cfg, d, layers, heads, intent_dim)
        self.n_act_tokens = int(n_act_tokens)
        # ⛔ `mix` IS DELETED, and getting this backwards cost a 2.4 % capacity
        # confound that the module-size report caught. v1's `mix` is
        # Linear(2d, d) = 2,098,176 params at d=1024; this arm's `act_split` is
        # Linear(d, 2d) = 2,099,200. act_split REPLACES mix at almost exactly
        # the same size, so DELETING mix is what keeps the arms matched
        # (+2,048 total, 0.001 %) — KEEPING it made v1′ +4,202,496 (+2.4 %) of
        # dead weight, which is the capacity confound this arm exists to avoid.
        # ⚠️ n_act_tokens != 2 breaks that parity by design; see the config.
        del self.mix
        self.act_split = nn.Linear(d, self.n_act_tokens * d)
        self.act_pos = nn.Parameter(torch.zeros(1, self.n_act_tokens, d))
        nn.init.trunc_normal_(self.act_pos, std=0.02)
        # ZERO-INIT the split so that at initialisation the action tokens carry
        # no signal and the step is near-identity — the same property v1 gets
        # from its residual head, preserved so the 6 s rollout does not drift
        # on the first gradient.
        nn.init.zeros_(self.act_split.weight)
        nn.init.zeros_(self.act_split.bias)

    def step(self, field: Tensor, action: Tensor,
             intent: Tensor | None = None) -> Tensor:
        """One latent step. ``field`` [B,N,d], ``action`` [B,a_dim] -> [B,N,d].

        Residual by construction, exactly as v1: the predictor learns the
        CHANGE, so a zero-action step is near-identity at init.
        """
        n = field.shape[1]
        a = self.act(action)
        if intent is not None and self.intent is not None:
            # the hierarchy port, unchanged from v1: g_tac reaches the
            # operative predictor through the SAME additive intent path
            a = a + self.intent(intent)
        at = self.act_split(a).reshape(a.shape[0], self.n_act_tokens, -1)
        at = at + self.act_pos
        x = torch.cat([field, at], dim=1)          # ONE self-attention stream
        for blk in self.blocks:
            x = blk(x)
        return field + self.head(x[:, :n])          # read the FIELD back out


class RefAV1Prime(RefAV1):
    """REF-A v1 with both field predictors swapped for the action-stream form.

    ⚠️ The strategic predictor is NOT swapped. ``StrategicSubspacePredictor``
    operates on a compact strategy-only subspace rather than a token field, so
    "action as a token in the stream" has no counterpart there; changing it too
    would have made the arm a two-axis edit and destroyed the comparison.
    """

    def __init__(self, cfg: RefAV1PrimeConfig | None = None):
        cfg = cfg or RefAV1PrimeConfig()
        super().__init__(cfg)
        self.cfg = cfg
        intent_dim = (cfg.tactical_cfg.d_intent
                      if cfg.tactical_cfg is not None else None)
        self.operative = ActionStreamPredictor(
            cfg, cfg.d_state, cfg.op_layers, cfg.op_heads, intent_dim,
            n_act_tokens=cfg.n_act_tokens)
        self.tactical = ActionStreamPredictor(
            cfg, cfg.d_state, cfg.tac_layers, cfg.op_heads, intent_dim,
            n_act_tokens=cfg.n_act_tokens)
