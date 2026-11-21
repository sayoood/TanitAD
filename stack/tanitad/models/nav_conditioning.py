"""Nav-command conditioning for ALL THREE layers — operative, tactical, strategic.

PI DIRECTIVE 2026-08-30 (binding): the nav command token is a **MANDATORY input**
to every abstraction layer, *"conditioning the WM like the actions"*. Spec:
``Project Steering/SPEC_NAV_CONDITIONING_ALL_LAYERS.md``.

⭐ IT IS AN INPUT — not a head, not a loss. One embedding of the 3-token vocab
plus its two continuous args, **shared across all three layers so they cannot
drift apart**, injected at each layer's conditioning point.

⛔ THE CHANNEL IS AN ORACLE ON OUR CORPUS, AND ITS CONTROLS SHIP WITH IT.
Every ``nav_command`` in B1 carries ``provenance: "ego-future"`` — computed from
the ego's own future path (4,719/4,719). Flagship v1's route head was an exact
bijection of the nav it was fed (369/369, 81/81) and **scored 1.0000** — an echo
read as skill. ⇒ :func:`apply_nav_control` implements the anti-echo twins of the
hold-action floor, and **no capability claim from a nav arm is admissible without
`shuffled` reported beside it**:

  ``hold``      freeze the token at t0 — is the arm merely tracking a moving oracle?
  ``shuffled``  serve ANOTHER clip's nav — ⛔ THE DECISIVE ONE. No degradation
                means the channel is INERT; degradation to chance may mean the arm
                is reading the FUTURE rather than following a route.
  ``none``      the ablation that sizes what nav actually buys.

⚠️ A MISSING NAV TOKEN RAISES. It never defaults. A silent default is the
``_ensure_ego`` size-threshold trap in a new costume: it would let an arm train
without the channel while its config claims otherwise, and make two arms silently
incomparable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from tanitad.channel_admissibility import ChannelExclusion
from tanitad.models.vocab_v7 import NAV_COMMAND_TOKENS, NAV_PROVENANCE

__all__ = ["NAV_CONTROLS", "NavArgStats", "NavConditioner", "NavTokenMissing",
           "apply_nav_control", "nav_provenance_stamp"]

#: the controls the spec requires; ``real`` is the uncontrolled channel.
NAV_CONTROLS = ("real", "hold", "shuffled", "none")

#: ⛔⛔ `nav_args` MUST NOT BE PLUMBED INTO AN RL ROLLOUT **AS BUILT** -- and the
#: reason is one of its two value slots, not the channel.
#:
#: ⭐ THE TOKEN IS PLUMBED AND THE ARGS ARE NOT, WHICH IS NOT AN INCONSISTENCY.
#: `nav_cmd` carries `provenance: "ego-future"` on 4,719/4,719 records too, and it IS
#: in `refc_adapter.FORWARD_KEYS`. The programme admits it because the QUANTITY --
#: "which way at the next junction" -- is one a real nav system genuinely supplies,
#: and because this module makes the `shuffled` control mandatory beside any claim
#: from it. Apply the same two tests slot by slot and the block splits:
#:
#:   `distance_m`  ⭐ ADMISSIBLE AS A QUANTITY. A real nav system says "turn left in
#:                 300 m". Our value is measured along the ego's driven path
#:                 (`s2_geom_emit_v7._arc_to`), but arc length TO A FIXED ROAD
#:                 FEATURE is a property of the route, and a map knows it without
#:                 knowing how the car will be driven.
#:   `time_s`      ⛔ NOT ADMISSIBLE AT ALL. `s2_geom_emit_v7.py:715` sets it to
#:                 `nxt[0]` -- the time offset at which THE EGO'S OWN FUTURE PATH
#:                 reaches the turn. No nav system can know when you will arrive,
#:                 because that depends on how you drive: `time_s` is the ego's
#:                 future SPEED PROFILE, inverted. Handing it to a rollout hands it
#:                 the along-track answer, on the axis that owns 88.7 % of the oracle
#:                 gap (`Research/2026-09-06-goal-point`).
#:
#: ⚠️ AND THERE IS NO WAY TO SHIP ONE WITHOUT THE OTHER TODAY. `NAV_ARG_SLOTS`
#: is `("distance_m", "time_s")`, `refc_v3.NAV_ARG_DIMS` is 3 and sizes
#: `nn.Linear(NAV_ARG_DIMS, d_nav)`, the loader always emits `[dist_norm,
#: time_norm, valid]`, and the forward REFUSES any other width. `--nav-args` is a
#: boolean: there is no distance-only mode to select. That is why the exclusion is
#: on the whole channel and is TEMPORARY -- the fix is a build option, not a ruling.
FORWARD_EXCLUSIONS = (
    ChannelExclusion(
        channel="nav_args",
        owner="tanitad.models.nav_conditioning (E13b nav-argument seam)",
        reason=(
            "The block ships (distance_norm, time_norm, args_valid) and cannot ship "
            "fewer. `distance_m` is admissible as a quantity -- a real nav system "
            "says 'turn left in 300 m'. `time_s` is not: `s2_geom_emit_v7.py:715` "
            "sets it to the time offset at which the EGO'S OWN FUTURE PATH reaches "
            "the turn, and no nav system can know when you will arrive because that "
            "depends on how you drive. It is the ego's future speed profile "
            "inverted, on the axis that owns 88.7 % of the oracle gap. Since "
            "`NAV_ARG_DIMS = 3` sizes the projection and the forward refuses any "
            "other width, plumbing the channel plumbs the time slot."),
        unblock=(
            "A DISTANCE-ONLY MODE. Ship `--nav-args=distance` with "
            "`NAV_ARG_SLOTS = ('distance_m',)` and `NAV_ARG_DIMS = 2` -- the loader "
            "emitting `[distance_norm, valid]` and the projection resized to match. "
            "At that point every slot is a quantity a nav system supplies and the "
            "channel becomes plumbable, subject to the same `shuffled` control this "
            "module already makes mandatory for the token. ⚠️ The eval-time "
            "`--nav-args` arm is NOT blocked meanwhile: it is a supervised arm, not "
            "a rollout, and it reports its oracle stamp."),
        evidence=("PUBLISHED-CODE: `vocab_v7.NAV_ARG_SLOTS`, `refc_v3.NAV_ARG_DIMS` "
                  "= 3 and its width refusal, `refc_v3_train.py` loader emitting "
                  "`[dn, tn, 1.0]`; `s2_geom_emit_v7.py:714-715` for both slots' "
                  "provenance. MEASURED: 88.7 % longitudinal share of the oracle "
                  "gap, `Research/2026-09-06-goal-point`."),
        permanent=False,
        refs=("Research/2026-09-07-rl-channel-admissibility/",),
    ),
)


class NavTokenMissing(RuntimeError):
    """Raised when a batch reaches the model without a nav token."""


@dataclass(frozen=True)
class NavArgStats:
    """Normalisation statistics for the two continuous args.

    ⛔ FIT-SPLIT ONLY. Computing these per-batch (or over the scored split) is the
    2026-08-22 ridge-probe failure in a new costume: a statistic fitted on the
    data it will later score makes the channel look more informative than it is.
    :meth:`from_fit_split` is the only constructor that reads data, and it takes
    the fit split explicitly rather than "the corpus".
    """

    distance_mean: float
    distance_std: float
    time_mean: float
    time_std: float
    n_fit: int

    @classmethod
    def from_fit_split(cls, distances, times) -> "NavArgStats":
        d = torch.as_tensor(list(distances), dtype=torch.float32)
        t = torch.as_tensor(list(times), dtype=torch.float32)
        if d.numel() == 0:
            raise ValueError("[nav] fit split is empty — refusing to invent stats")
        return cls(float(d.mean()), float(d.std().clamp_min(1e-6)),
                   float(t.mean()), float(t.std().clamp_min(1e-6)), int(d.numel()))

    def normalise(self, distance_m: Tensor, time_s: Tensor) -> Tensor:
        return torch.stack([(distance_m - self.distance_mean) / self.distance_std,
                            (time_s - self.time_mean) / self.time_std], dim=-1)

    def to_dict(self) -> dict[str, Any]:
        return {"distance_mean": self.distance_mean, "distance_std": self.distance_std,
                "time_mean": self.time_mean, "time_std": self.time_std,
                "n_fit": self.n_fit,
                "_read": "FIT-SPLIT ONLY — never per-batch, never on the scored split"}


class NavConditioner(nn.Module):
    """ONE shared nav embedding, injected into all three layers.

    ⭐ SHARED BY CONSTRUCTION, NOT BY CONVENTION. A single :class:`nn.Embedding`
    and a single arg projection serve operative, tactical and strategic. Three
    per-layer copies would drift apart under training and the three layers would
    silently stop meaning the same thing by the same token — the failure this
    design exists to make impossible. ``test_the_embedding_is_shared_across_layers``
    pins it by mutating the embedding once and asserting all three outputs move.

    Per-layer adaptation is a small output projection (``layer_proj``), so each
    layer can receive the conditioning at its own width while the SEMANTICS stay
    shared.
    """

    def __init__(self, d_model: int | None = None, *,
                 widths: dict[str, int] | None = None,
                 layers=("operative", "tactical", "strategic"),
                 d_embed: int = 32, stats: NavArgStats | None = None,
                 gate_init: float = 0.1, zero_init: bool = True):
        """``widths`` gives EACH layer the width of the tensor nav is added to.

        ⛔ THE THREE SITES DIFFER IN SHAPE, AND ONE ``d_model`` IS WRONG FOR TWO
        OF THEM. MEASURED from source, not from a docstring:

        * **operative** — ``predictor.py:199``: ``cond = act_emb(actions)``, an
          ADDITIVE FiLM pathway at the predictor's hidden width.
        * **tactical / strategic** — ``tactical.py:267-272``: ``FTac.forward``
          computes ``in_proj(cat([z_tac, g_flat]))``. It **CONCATENATES**; there
          is NO additive cond pathway at these layers at all. Widening that cat
          would change ``in_proj``'s shape, which the v6 docs say bypasses
          ``STAGE_MAY_INTRODUCE`` — and would make every existing checkpoint
          UNLOADABLE, breaking the comparability the PI ruled on.

        ⭐ So nav follows the precedent already in the tree for this exact
        problem, ``cond_tac_dyn`` (``v6.py:4904-4907``, applied at ``:5555``):
        **project INTO the existing ``g_flat`` width and ADD**, zero-init —
        ``g_cond_tac = e_a_tac + cond_tac_dyn(...)``. No shape change, so the
        strict-subset property of the operative port is preserved here too.

        | layer | width | added to |
        |---|---|---|
        | operative | predictor hidden ``d`` | ``cond`` after ``act_emb`` |
        | tactical | ``2 * d_goal_embed`` | ``e_a_tac`` |
        | strategic | ``d_goal_embed`` | the strategic ``g_flat`` |

        ⚠️ ``d_model`` is retained ONLY for the uniform-width test case. Passing
        it for a real stack gives two of three layers the wrong width and dies at
        the first forward.
        """
        super().__init__()
        if widths is None:
            if d_model is None:
                raise ValueError(
                    "[nav] pass widths={layer: width} — the three conditioning "
                    "sites have DIFFERENT widths (operative=predictor hidden, "
                    "tactical=2*d_goal_embed, strategic=d_goal_embed). d_model "
                    "is only for the uniform-width test case.")
            widths = {ln: int(d_model) for ln in layers}
        self.tokens = tuple(NAV_COMMAND_TOKENS)
        self.tok2id = {t: i for i, t in enumerate(self.tokens)}
        self.layers = tuple(widths)
        self.widths = dict(widths)
        self.stats = stats
        self.embed = nn.Embedding(len(self.tokens), d_embed)     # SHARED
        self.arg_proj = nn.Linear(2, d_embed)                    # SHARED
        self.layer_proj = nn.ModuleDict(
            {ln: nn.Linear(d_embed, w) for ln, w in widths.items()})
        #: ⛔ REZERO GATE, PER LAYER — AND IT IS NOT OPTIONAL POLISH. MEASURED
        #: (H26, ``predictor.py:126``): the UNGATED ``intent_proj`` term reached
        #: norm ~31.4 against ``act_emb`` ~28.3, **diluting the action
        #: conditioning**, and engaging intent was measured NET-HARMFUL to the
        #: operative. The fix there was a ReZero gate at init 0.1 so the term
        #: starts action-dominant and grows only if training earns it.
        #: ⇒ Nav enters the SAME additive ``cond`` pathway as a THIRD term. An
        #: ungated nav term would reproduce H26 with one more competitor. The
        #: PI's "condition the WM like the actions" is right in KIND — nav is an
        #: input, not a head — but "like the actions" must not mean "at equal
        #: magnitude from step 0", which is the configuration H26 measured harmful.
        self.gate = nn.ParameterDict(
            {ln: nn.Parameter(torch.tensor(float(gate_init))) for ln in self.layers})
        if zero_init:
            #: zero-init the OUTPUT projection so introducing nav mid-ladder is
            #: loss-continuous — the same discipline as FiLM's zero-init
            #: (``predictor.py:41-42``) and the tac_goal_cond port.
            for ln in self.layers:
                nn.init.zeros_(self.layer_proj[ln].weight)
                nn.init.zeros_(self.layer_proj[ln].bias)

    def encode(self, token_id: Tensor, args: Tensor) -> Tensor:
        """``[B]`` ids + ``[B, 2]`` NORMALISED args -> ``[B, d_embed]`` shared code."""
        return self.embed(token_id) + self.arg_proj(args)

    def forward(self, token_id: Tensor, args: Tensor, layer: str) -> Tensor:
        """The term to ADD into that layer's existing ``cond``.

        ⛔ ADDITIVE, NEVER CONCATENATED. ``predictor.py:199-207`` builds
        ``cond = act_emb(actions)`` then ``cond = cond + intent_term``; the v6
        docs state that a SHAPE CHANGE bypasses ``STAGE_MAY_INTRODUCE``'s
        adjudication (and ``load_state_dict(strict=False)`` still RAISES on
        shapes, measured). Concatenating nav would therefore change the
        conditioning width and slip past the stage gate that decides what a
        stage is allowed to introduce.
        """
        if layer not in self.layer_proj:
            raise KeyError(f"[nav] unknown layer {layer!r}; have {self.layers}")
        return self.gate[layer] * self.layer_proj[layer](self.encode(token_id, args))

    def ids_from_batch(self, batch: dict) -> Tensor:
        """⛔ RAISES on a missing/unknown token. Never defaults."""
        nav = batch.get("nav_command")
        if nav is None:
            raise NavTokenMissing(
                "[nav] ⛔ batch has no 'nav_command'. The nav token is a MANDATORY "
                "input to all three layers (PI directive 2026-08-30) and must not "
                "be defaulted: a silent default would train an arm without the "
                "channel while its config claims otherwise, and make two arms "
                "silently incomparable.")
        toks = nav["token"] if isinstance(nav, dict) else nav
        if isinstance(toks, Tensor):
            return toks.long()
        out = []
        for t in ([toks] if isinstance(toks, str) else list(toks)):
            if t not in self.tok2id:
                raise NavTokenMissing(
                    f"[nav] ⛔ unknown nav token {t!r}; the released vocab is "
                    f"{self.tokens}. An unrecognised token is refused rather than "
                    f"mapped to a default.")
            out.append(self.tok2id[t])
        return torch.tensor(out, dtype=torch.long)


def apply_nav_control(token_id: Tensor, args: Tensor, control: str,
                      *, generator: torch.Generator | None = None
                      ) -> tuple[Tensor, Tensor, bool]:
    """Apply an anti-echo control. Returns ``(ids, args, channel_active)``.

    ⛔ ``shuffled`` IS THE DECISIVE CONTROL and it must be reported beside any
    capability claim. It serves another clip's nav by permuting along the batch:
      * no degradation ⇒ the channel is INERT, the arm ignores it;
      * degradation to chance ⇒ the arm may be reading the FUTURE, not routing.

    ⚠️ ``shuffled`` uses a DERANGEMENT-style roll rather than a random permutation
    so that no element can keep its own value by chance. A random permutation
    leaves ~1/B items in place, which on small batches silently weakens the
    control — the control would be partly measuring the real channel.
    """
    if control not in NAV_CONTROLS:
        raise ValueError(f"[nav] unknown control {control!r}; have {NAV_CONTROLS}")
    if control == "real":
        return token_id, args, True
    if control == "none":
        # the channel is switched OFF — zeroed ids AND args, and reported inactive
        return (torch.zeros_like(token_id), torch.zeros_like(args), False)
    if control == "hold":
        # freeze at t0: broadcast the first element of the sequence axis if there
        # is one, else the value is already per-window and holding is a no-op that
        # must still be REPORTED so the arm's control is visible.
        if token_id.dim() > 1:
            return (token_id[:, :1].expand_as(token_id),
                    args[:, :1].expand_as(args), True)
        return token_id, args, True
    # shuffled: roll by 1 so nothing keeps its own nav
    if token_id.shape[0] < 2:
        raise ValueError("[nav] ⛔ shuffled-nav needs batch >= 2; with one item "
                         "there is no other clip's nav to serve, and returning "
                         "the same nav would make the control silently inert.")
    return token_id.roll(1, 0), args.roll(1, 0), True


def nav_provenance_stamp(provenance: str, control: str = "real") -> dict[str, Any]:
    """The stamp that travels into ``config.json`` and every eval record.

    ⚠️ An arm trained or evaluated on ``ego-future`` nav is labelled WHEREVER its
    numbers appear. A future ``nav-system`` arm is a DIFFERENT measurement and
    must never be pooled with the oracle one.
    """
    if provenance not in NAV_PROVENANCE:
        raise ValueError(f"[nav] unknown provenance {provenance!r}; "
                         f"the released vocab is {tuple(NAV_PROVENANCE)}")
    oracle = provenance == "ego-future"
    return {"nav_provenance": provenance, "nav_control": control,
            "nav_is_oracle": oracle,
            "_read": ("ORACLE — computed from the ego's own future path. No "
                      "capability claim is admissible without the shuffled-nav "
                      "control reported beside it." if oracle else
                      "real nav-system source — NOT poolable with oracle-nav arms")}
