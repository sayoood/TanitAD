"""Tactical layer, stage 0 (V18 backlog E4.2 + E4.3) — phi_tac, the goal fan,
f_tac, and the tactical selector, composed by :class:`TacticalStage0`.

WHERE THIS SITS (HIERARCHICAL_WM_REDESIGN.md §3.1–§3.5)
-------------------------------------------------------
The redesign's pillar is a per-level {state, predictor, goal fan, selector}
stack. This file is the TACTICAL level at stage 0 — trained on a FROZEN
operative trunk (pure addition, zero risk to the operative stack):

    z_op(t-3..t) [1 Hz samples of the operative latent]
        │  PhiTac (causal TCN pool, ~2 M)                 §3.1
        ▼
    z_tac ∈ R^512 @ 1 Hz
        │  TacticalGoalFan (N=8 anchor queries, shared trunk)   §3.2/§3.3
        ▼
    fan of g_tac candidates [N, K, 4] + logits [N]        K = |{2,4,6 s}|
        │  FTac rolls each candidate: ẑ_tac(t+1s) = f_tac(z_tac, g)  §3.3
        ▼
    TacticalSelector scores the rolled futures — RANKING loss against the
    hindsight-oracle winner, never CE                     §3.3 + E4.3

The tactical "action" IS the goal (§3.3): f_tac has no separate action input.
Down-sampling in TIME is the abstraction mechanism — z_tac (512) is narrower
than z_op by design; the level must be *forced* to abstract.

LABELS (E4.1 — ``stack/scripts/refb_labels.py``, its E4.1 section)
------------------------------------------------------------------
The fan is supervised against ``refb_labels.goal_tac_targets`` /
``goal_tac_labels``: hindsight goals ``[K, 4]`` = (x, y, heading, speed) of
the ego at t+tau (tau ∈ {20, 40, 60} steps = {2, 4, 6} s @ 10 Hz), in the ego
frame of t, with a ``[K]`` bool validity mask (taus beyond the episode end are
valid=False and CLAMP, never NaN). This module reproduces that layout EXACTLY
— (x, y, heading, speed), same tau order — and consumes the mask so invalid
rows contribute exactly zero loss (grads at masked rows are zero, pinned by
tests/test_tactical.py). The optional aux heads use the E4.1 3-axis class
spaces (LAT 5-way / LON 5-way / LANE 3-way — ``LAT3_NAMES``/``LON3_NAMES``/
``LANE3_NAMES``); the counts are re-declared here as constants rather than
imported, because ``scripts/`` is not an importable package from ``tanitad``
(the contract is the E4.1 section; tests pin the counts).

⛔ WHY THE SELECTOR IS RANKING-TRAINED, NOT CE (E4.3)
----------------------------------------------------
"The selector is the known failure point — v5f measures a good fan and a
selector that never closes (sel_gap ~0.3–0.5, no trend)." (redesign §3.3.)
Consequence (1) baked in here: the selector trains with a MARGIN RANKING loss
against the hindsight-oracle outcome — which candidate's future was closest
to what the driver actually achieved — not a softmax CE over candidates.
Consequence (2): ``sel_gap_tac`` (oracle-vs-selected) is a first-class metric
emitted next to the loss. For EVAL-time, CI-carrying use the instrument is
``taniteval.selgap`` (episode-cluster bootstrap; per-level, never pooled) —
referenced, deliberately NOT imported: ``stack`` does not depend on
``taniteval``. The functions here are the train-time counterparts only.

NAMING NOTE: ``tanitad.models.fourbrain`` also exports a ``TacticalSelector``
(the v1 4-brain fallback arbiter — a different object). This module is NOT
re-exported from ``tanitad.models.__init__``; import it by its full path
``tanitad.models.tactical`` to keep the two apart.

Evidence class for every number in this docstring: the param counts are
MEASURED by ``n_params()`` at the reference sizes and asserted in band by
tests/test_tactical.py; design facts cite HIERARCHICAL_WM_REDESIGN.md and
V18_BACKLOG.md E4 (INHERITED design intent, verified against the doc text).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# E4.1 label-contract constants (source of truth: scripts/refb_labels.py, E4.1
# section — LAT3_NAMES / LON3_NAMES / LANE3_NAMES and GOAL_TAC_TAUS_STEPS).
# Re-declared, not imported: scripts/ is not a package reachable from tanitad.
# ---------------------------------------------------------------------------
GOAL_DIMS = 4                    # (x, y, heading, speed) — the E4.1 row layout
GOAL_TAC_TAUS_STEPS = (20, 40, 60)   # 2 / 4 / 6 s @ 10 Hz — tactical horizons
N_LAT3 = 5                       # straight/gentle_l/gentle_r/sharp_l/sharp_r
N_LON3 = 5                       # hard_brake/brake/keep/accel/hard_accel
N_LANE3 = 3                      # keep/change_left/change_right


def _wrap_to_pi(x: Tensor) -> Tensor:
    """Wrap angles to (-pi, pi] — same convention as refb_labels.wrap_to_pi."""
    return torch.remainder(x + math.pi, 2.0 * math.pi) - math.pi


def n_params(module: nn.Module) -> int:
    """Trainable-parameter count (the number tests assert in band)."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ============================================================================
# 1. PhiTac — temporal pool z_op(t-3..t) -> z_tac  (E4.2, redesign §3.1)
# ============================================================================

class PhiTac(nn.Module):
    """Causal TCN pool over the last ``window`` operative latents -> z_tac.

    ``[B, W, d_op] -> [B, d_tac]`` at 1 Hz: the W inputs are 1 Hz SAMPLES of
    the operative latent (t-3 s .. t for W=4), not consecutive 10 Hz frames —
    down-sampling in time IS the abstraction mechanism (redesign §3.1).

    Geometry: input projection d_op->hidden, then residual blocks of DILATED
    causal 1-D convs (kernel 3, left-padded by (k-1)*dilation so position t
    never sees t+1 — causality holds by construction, not by masking), then
    the LAST position (which, with dilations (1, 2, 4), has receptive field
    15 >= any sane window) through a 2-layer MLP to d_tac. A TCN and not
    attention because W is tiny (4) and fixed: convs give the causal pool at
    a fraction of the params, and the ~2 M budget (MEASURED 1.71 M at the
    reference d_op=512, hidden=256, d_tac=512; asserted in [1 M, 4 M] by
    tests/test_tactical.py) is the point — stage 0 is a pure, cheap addition
    on a frozen trunk.
    """

    def __init__(self, d_op: int, d_tac: int = 512, window: int = 4,
                 hidden: int = 256, dilations: tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.d_op, self.d_tac, self.window = d_op, d_tac, window
        self.in_proj = nn.Linear(d_op, hidden)
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for d in dilations:
            self.blocks.append(nn.ModuleDict({
                "conv1": nn.Conv1d(hidden, hidden, 3, dilation=d),
                "conv2": nn.Conv1d(hidden, hidden, 3, dilation=d),
            }))
            # GroupNorm(1, ·) == LayerNorm over channels at each position:
            # works on [B, C, T] without transposes.
            self.norms.append(nn.GroupNorm(1, hidden))
        self.dilations = dilations
        self.out = nn.Sequential(
            nn.Linear(hidden, d_tac), nn.GELU(), nn.Linear(d_tac, d_tac))

    def n_params(self) -> int:
        return n_params(self)

    def forward(self, z_op: Tensor) -> Tensor:
        """``z_op`` [B, W, d_op] (oldest first, index -1 = now) -> [B, d_tac]."""
        if z_op.ndim != 3 or z_op.shape[-1] != self.d_op:
            raise ValueError(
                f"z_op must be [B, W, {self.d_op}], got {tuple(z_op.shape)}")
        h = self.in_proj(z_op).transpose(1, 2)          # [B, hidden, W]
        for blk, norm, d in zip(self.blocks, self.norms, self.dilations):
            r = h
            # causal: pad LEFT only, so conv output at t reads t-k..t.
            h = nn.functional.pad(h, (2 * d, 0))
            h = torch.nn.functional.gelu(blk["conv1"](h))
            h = nn.functional.pad(h, (2 * d, 0))
            h = blk["conv2"](h)
            h = norm(torch.nn.functional.gelu(h) + r)   # residual
        return self.out(h[:, :, -1])                    # last (= current) step


# ============================================================================
# 2. TacticalGoalFan — z_tac -> N candidate g_tac + logits  (E4.2, §3.2/§3.3)
# ============================================================================

class TacticalGoalFan(nn.Module):
    """N candidate tactical goals from z_tac ONLY: ``[B, d_tac] ->
    (goals [B, N, K, 4], logits [B, N])`` with K = len(horizon_taus) and the
    last dim laid out EXACTLY as the E4.1 label: (x, y, heading, speed) at
    tau ∈ horizon_taus (default (20, 40, 60) steps = 2/4/6 s), ego frame of
    now, heading wrapped to (-pi, pi].

    DIVERSITY BY CONSTRUCTION, NOT N HEADS: the N candidates are N learned
    ANCHOR QUERIES (an ``nn.Embedding``) added to the projected z_tac and
    pushed through ONE shared MLP trunk. N independent heads would cost N×
    the trunk params and, worse, have no shared representation to keep the
    fan calibrated; anchor queries give N distinct modes for the price of
    N·d_tac extra params (the same lesson as REF-C's anchor vocabulary —
    modes live in cheap queries, competence lives in one shared trunk).

    ⛔ ADMISSIBILITY — binding rule, Sayed 2026-08-03, quoted verbatim:
    "yes a goal input is admissible, at the same time, we need to be careful
    not to include the result of the situation classification in the goal
    input."  The check to run: *"could this input contain what the situation
    classifier produces?"*  Answer for this module: **No.** The ONLY forward
    input is ``z_tac`` — no situation-classifier posterior/argmax/embedding,
    no ego state, no ``**kwargs`` backdoor through which one could be
    threaded (the signature is pinned by tests/test_tactical.py). ``z_tac``
    is a pure temporal pool (PhiTac) of the operative latents, which come
    from the VISION encoder (vision-only-at-inference rule, 2026-08-03).
    Shared-trunk disclosure, as the rule requires: if a situation classifier
    is ever trained as another READ-ONLY head off the same frozen z_op/z_tac
    trunk, the sharing stays admissible because information flows trunk→
    classifier only — nothing of the classifier's OUTPUT enters this fan's
    input path, and at stage 0 the trunk is frozen so no classifier gradient
    can sculpt it either. Coupling a classifier gradient into a shared
    UNFROZEN trunk (stage 2+) would need a fresh admissibility argument.
    """

    def __init__(self, d_tac: int, n_goals: int = 8,
                 horizon_taus: tuple[int, ...] = GOAL_TAC_TAUS_STEPS,
                 hidden: int = 256):
        super().__init__()
        self.d_tac, self.n_goals = d_tac, n_goals
        self.horizon_taus = tuple(int(t) for t in horizon_taus)
        self.n_taus = len(self.horizon_taus)
        self.anchors = nn.Embedding(n_goals, hidden)    # the N mode queries
        self.z_proj = nn.Linear(d_tac, hidden)
        self.trunk = nn.Sequential(                      # SHARED across N
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU())
        self.goal_head = nn.Linear(hidden, self.n_taus * GOAL_DIMS)
        self.logit_head = nn.Linear(hidden, 1)

    def n_params(self) -> int:
        return n_params(self)

    def forward(self, z_tac: Tensor) -> tuple[Tensor, Tensor]:
        """``z_tac`` [B, d_tac] -> (goals [B, N, K, 4], logits [B, N]).

        z_tac ONLY — see the class docstring's admissibility block."""
        if z_tac.ndim != 2 or z_tac.shape[-1] != self.d_tac:
            raise ValueError(
                f"z_tac must be [B, {self.d_tac}], got {tuple(z_tac.shape)}")
        B = z_tac.shape[0]
        q = self.anchors.weight[None, :, :]              # [1, N, hidden]
        h = self.z_proj(z_tac)[:, None, :] + q           # [B, N, hidden]
        h = self.trunk(h)
        goals = self.goal_head(h).view(B, self.n_goals, self.n_taus, GOAL_DIMS)
        # heading channel wrapped so the emitted layout matches the label's
        # (-pi, pi] convention exactly.
        goals = torch.cat([goals[..., :2],
                           _wrap_to_pi(goals[..., 2:3]),
                           goals[..., 3:]], dim=-1)
        logits = self.logit_head(h).squeeze(-1)          # [B, N]
        return goals, logits


# ============================================================================
# 3. FTac — the 1 Hz tactical latent predictor  (E4.2, redesign §3.3)
# ============================================================================

class FTac(nn.Module):
    """``ẑ_tac(t+1s) = f_tac(z_tac, g_tac)`` — the tactical level's OWN
    predictor, in its OWN space (redesign §3.3: each level rolls itself; a
    tactical roll is cheap because the state is 512-d and the clock is 1 Hz).
    The tactical "action" IS the goal — there is no separate action input.

    ``(z_tac [B, d_tac], g_flat [B, d_goal]) -> [B, d_tac]`` with
    ``d_goal = K*4`` (the flattened (x, y, heading, speed) fan candidate,
    default 3 taus -> 12). Geometry: input projection, 3 residual MLP blocks
    (d_tac -> 2*d_tac -> d_tac), output projection — MEASURED 3.68 M at the
    reference d_tac=512, hidden=512 (the ~4 M budget of V18 E4.2; asserted
    in [2 M, 6 M] by tests/test_tactical.py). The selector rolls this once
    per candidate to score the fan."""

    def __init__(self, d_tac: int, d_goal: int = GOAL_DIMS * 3,
                 hidden: int = 512, n_blocks: int = 3):
        super().__init__()
        self.d_tac, self.d_goal = d_tac, d_goal
        self.in_proj = nn.Linear(d_tac + d_goal, hidden)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(hidden),
                          nn.Linear(hidden, 2 * hidden), nn.GELU(),
                          nn.Linear(2 * hidden, hidden))
            for _ in range(n_blocks))
        self.out = nn.Linear(hidden, d_tac)

    def n_params(self) -> int:
        return n_params(self)

    def forward(self, z_tac: Tensor, g_flat: Tensor) -> Tensor:
        """``z_tac`` [B, d_tac] + flattened goal [B, d_goal] -> ẑ_tac [B, d_tac]."""
        if g_flat.shape[-1] != self.d_goal:
            raise ValueError(
                f"g_flat must be [B, {self.d_goal}], got {tuple(g_flat.shape)}")
        h = self.in_proj(torch.cat([z_tac, g_flat], dim=-1))
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


# ============================================================================
# 4. Losses + sel_gap (E4.3) — module-level so tests/trainers hit them direct
# ============================================================================

def fan_goal_error(goals: Tensor, goal_labels: Tensor, valid: Tensor,
                   w_pos: float = 1.0, w_heading: float = 1.0,
                   w_speed: float = 1.0) -> Tensor:
    """Per-candidate goal error vs the E4.1 hindsight label: ``(goals
    [B, N, K, 4], labels [B, K, 4], valid [B, K] bool) -> [B, N]``.

    Per valid tau: ||Δ(x, y)||_2 + w_h·|wrap(Δheading)| + w_v·|Δspeed|,
    masked-mean over the valid taus (a sample with zero valid taus returns
    0 — callers exclude it via ``valid.any(-1)``). The three terms are
    dimensionally mixed (m, rad, m/s) ON PURPOSE and ONLY for ORDERING —
    winner selection and ranking — never quoted as a metric; quotable goal
    error is per-family (position FDE, heading, speed separately, the
    2026-08-02 four-families rule) and lives in the eval, not here."""
    d = goals - goal_labels[:, None, :, :]              # [B, N, K, 4]
    err = (d[..., :2].norm(dim=-1)
           + w_heading * _wrap_to_pi(d[..., 2]).abs()
           + w_speed * d[..., 3].abs())                 # [B, N, K]
    m = valid[:, None, :].to(err.dtype)                 # [B, 1, K]
    return (err * m).sum(-1) / m.sum(-1).clamp(min=1.0)


def wta_regression_loss(goals: Tensor, goal_labels: Tensor,
                        valid: Tensor) -> Tensor:
    """Winner-takes-all fan regression to the E4.1 labels -> scalar.

    The GT-nearest candidate (argmin :func:`fan_goal_error`, selection
    DETACHED) is regressed to the label with smooth-L1 on (x, y, speed) and
    smooth-L1 on the WRAPPED heading delta; the other N-1 candidates get no
    gradient (that is what keeps the fan diverse — each anchor query only
    learns the modes it wins). Invalid taus are masked BEFORE reduction, so
    their rows contribute exactly zero loss AND zero gradient (pinned by
    tests/test_tactical.py). Samples with no valid tau at all are excluded
    from the mean."""
    with torch.no_grad():
        win = fan_goal_error(goals, goal_labels, valid).argmin(dim=1)  # [B]
    B = goals.shape[0]
    g = goals[torch.arange(B, device=goals.device), win]    # [B, K, 4]
    d = g - goal_labels
    per = (nn.functional.smooth_l1_loss(
               d[..., [0, 1, 3]], torch.zeros_like(d[..., [0, 1, 3]]),
               reduction="none").sum(-1)
           + nn.functional.smooth_l1_loss(
               _wrap_to_pi(d[..., 2]), torch.zeros_like(d[..., 2]),
               reduction="none"))                            # [B, K]
    m = valid.to(per.dtype)
    row = (per * m).sum(-1) / m.sum(-1).clamp(min=1.0)       # [B]
    any_valid = valid.any(-1)
    if not bool(any_valid.any()):
        return goals.sum() * 0.0
    return row[any_valid].mean()


def ranking_loss(logits: Tensor, fan_goal_err: Tensor,
                 margin: float = 0.1) -> Tensor:
    """E4.3's selector loss — MARGIN RANKING against the hindsight-oracle
    winner, NOT CE: ``(logits [B, N], fan_goal_err [B, N]) -> scalar``.

    winner = argmin error (which candidate's outcome was closest to what the
    driver achieved — the hindsight oracle); loss = mean over the N-1 losers
    of relu(margin + logit_loser - logit_winner). Why not CE (redesign
    §3.3): CE on a one-hot winner punishes every non-winner symmetrically
    and saturates once the winner leads — ranking only asks for the ORDER
    with a margin, which is the actual selection contract, and is what the
    'selector never closes' v5f failure motivates. ``fan_goal_err`` is
    consumed detached (targets, not a gradient path)."""
    if logits.shape != fan_goal_err.shape:
        raise ValueError(f"shape mismatch {tuple(logits.shape)} vs "
                         f"{tuple(fan_goal_err.shape)}")
    B, N = logits.shape
    if N < 2:
        return logits.sum() * 0.0
    win = fan_goal_err.detach().argmin(dim=1)               # [B]
    lw = logits[torch.arange(B, device=logits.device), win] # [B]
    viol = (margin + logits - lw[:, None]).clamp(min=0.0)   # [B, N]
    loser = torch.ones_like(viol)
    loser[torch.arange(B, device=logits.device), win] = 0.0
    return (viol * loser).sum() / (loser.sum().clamp(min=1.0))


def sel_gap_tac(fan_goal_err: Tensor, sel_idx: Tensor
                ) -> tuple[Tensor, Tensor, Tensor]:
    """Oracle-vs-selected on the tactical fan: ``(fan_goal_err [B, N],
    sel_idx [B]) -> (selected_err [B], oracle_err [B], gap [B])``.

    ``gap = selected - oracle`` separates "the fan cannot propose it"
    (oracle high) from "the selector cannot find it" (oracle low, gap high)
    — the v5f failure signature this whole stage exists to attack. Returned
    PER-WINDOW so the caller can aggregate correctly. ⛔ For any QUOTED
    number use the CI-carrying instrument ``taniteval.selgap`` (episode-
    cluster bootstrap, per-level never pooled) — this function is the cheap
    train-time monitor only, and ``stack`` deliberately does not import
    ``taniteval``."""
    B = fan_goal_err.shape[0]
    sel = fan_goal_err[torch.arange(B, device=fan_goal_err.device), sel_idx]
    oracle = fan_goal_err.min(dim=1).values
    return sel, oracle, sel - oracle


class TacticalSelector(nn.Module):
    """E4.3 — scores the fan by ROLLING f_tac per candidate (redesign §3.3:
    each level proposes N candidates, rolls ITS OWN predictor, scores).

    ``forward(z_tac, goals, f_tac) -> scores [B, N]``: each candidate's goal
    is flattened, rolled one tactical step through the caller-supplied
    ``f_tac`` (passed as an ARGUMENT, not owned, so the stage composes the
    modules and no parameter is registered twice), and a small MLP scores
    (z_tac, ẑ_rolled, g_flat) jointly. Trained with :func:`ranking_loss`
    against the hindsight-oracle winner — never CE (class docstring above).
    ``sel_gap_tac`` is the first-class monitor; the CI-carrying eval-time
    instrument is ``taniteval.selgap`` (referenced, not imported)."""

    # re-exported as statics so trainers can reach everything through the
    # selector object alone.
    ranking_loss = staticmethod(ranking_loss)
    sel_gap_tac = staticmethod(sel_gap_tac)

    def __init__(self, d_tac: int, d_goal: int = GOAL_DIMS * 3,
                 hidden: int = 256):
        super().__init__()
        self.d_tac, self.d_goal = d_tac, d_goal
        self.scorer = nn.Sequential(
            nn.Linear(2 * d_tac + d_goal, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1))

    def n_params(self) -> int:
        return n_params(self)

    def forward(self, z_tac: Tensor, goals: Tensor, f_tac: FTac) -> Tensor:
        """``z_tac`` [B, d_tac], ``goals`` [B, N, K, 4] -> scores [B, N]."""
        B, N = goals.shape[0], goals.shape[1]
        g_flat = goals.reshape(B, N, -1)                     # [B, N, K*4]
        z_rep = z_tac[:, None, :].expand(B, N, self.d_tac)
        z_hat = f_tac(z_rep.reshape(B * N, -1),
                      g_flat.reshape(B * N, -1)).view(B, N, -1)
        feat = torch.cat([z_rep, z_hat, g_flat], dim=-1)
        return self.scorer(feat).squeeze(-1)                 # [B, N]


# ============================================================================
# 5. TacticalStage0 — the composition (E4.2+E4.3), frozen-trunk stage 0
# ============================================================================

@dataclass
class TacticalStage0Config:
    """Geometry + loss weights. Aux manoeuvre weights DEFAULT 0 (E4.2: the
    3-axis heads are optional at stage 0; flip the weights to enable)."""
    d_op: int = 2048                 # operative latent width (v1 readout)
    d_tac: int = 512                 # §3.1: d_t ≈ 512, information decreases up
    window: int = 4                  # z_op(t-3..t) @ 1 Hz
    n_goals: int = 8                 # fan size N (E4.2)
    horizon_taus: tuple[int, ...] = GOAL_TAC_TAUS_STEPS
    phi_hidden: int = 256
    f_hidden: int = 512
    fan_hidden: int = 256
    sel_hidden: int = 256
    margin: float = 0.1              # ranking margin
    # loss weights
    w_wta: float = 1.0               # fan WTA regression to E4.1 labels
    w_rank_fan: float = 1.0          # ranking on the fan's own logits
    w_rank_sel: float = 1.0          # ranking on the selector's rolled scores
    w_ftac: float = 1.0              # 1-step latent prediction (needs z_op_next)
    w_lat: float = 0.0               # optional 3-axis aux CE (E4.1 classes)
    w_lon: float = 0.0
    w_lane: float = 0.0


class TacticalStage0(nn.Module):
    """PhiTac + TacticalGoalFan + FTac + TacticalSelector, wired for stage-0
    training on a FROZEN operative trunk (redesign §3.5: pure addition; the
    z_op windows arrive precomputed/detached from the v1 trunk — this module
    never touches the encoder).

    BATCH CONTRACT for :meth:`losses` (all tensors on one device):

      ``z_op``       [B, W, d_op]  float — 1 Hz window of operative latents,
                     oldest first, index -1 = now. REQUIRED.
      ``goal``       [B, K, 4]     float — E4.1 hindsight goal labels
                     (``refb_labels.goal_tac_labels`` row at t; layout
                     (x, y, heading, speed), tau order = ``horizon_taus``).
                     REQUIRED.
      ``goal_valid`` [B, K]        bool  — the E4.1 validity mask (False =
                     tau beyond episode end; contributes ZERO loss and zero
                     grad). Optional; missing = all valid.
      ``z_op_next``  [B, W, d_op]  float — the window one tactical step
                     (1 s) later; enables the f_tac latent loss (target
                     ``phi_tac(z_op_next)`` DETACHED — f_tac learns dynamics,
                     not a collapse direction for phi). Optional.
      ``man3``       [B, 3]        long  — E4.1 3-axis labels (LAT, LON,
                     LANE columns of ``refb_labels.maneuver3_labels``).
                     Optional; consumed only where the aux weight > 0.

    Returns a dict of scalar tensors: always ``wta``, ``rank_fan``,
    ``rank_sel``, ``sel_gap``/``sel_err``/``oracle_err`` (monitors, detached)
    and ``total``; plus ``ftac`` / ``aux_lat`` / ``aux_lon`` / ``aux_lane``
    when their inputs are present (and, for aux, the weight is non-zero).
    ``total`` is the weighted sum of the LOSS terms only — the sel_gap
    monitors are never optimised directly (optimising the gap would just
    teach the selector to prefer whatever the fan already emits)."""

    def __init__(self, cfg: TacticalStage0Config | None = None, **kw):
        super().__init__()
        self.cfg = cfg or TacticalStage0Config(**kw)
        c = self.cfg
        d_goal = GOAL_DIMS * len(c.horizon_taus)
        self.phi_tac = PhiTac(c.d_op, c.d_tac, c.window, c.phi_hidden)
        self.goal_fan = TacticalGoalFan(c.d_tac, c.n_goals, c.horizon_taus,
                                        c.fan_hidden)
        self.f_tac = FTac(c.d_tac, d_goal, c.f_hidden)
        self.selector = TacticalSelector(c.d_tac, d_goal, c.sel_hidden)
        self.aux_lat = nn.Linear(c.d_tac, N_LAT3)
        self.aux_lon = nn.Linear(c.d_tac, N_LON3)
        self.aux_lane = nn.Linear(c.d_tac, N_LANE3)

    def n_params(self) -> int:
        return n_params(self)

    def forward(self, z_op: Tensor) -> dict[str, Tensor]:
        """Inference surface: ``z_op`` [B, W, d_op] -> dict with ``z_tac``
        [B, d_tac], ``goals`` [B, N, K, 4], ``logits`` [B, N] (fan prior),
        ``scores`` [B, N] (selector, f_tac-rolled), ``sel_idx`` [B] (argmax
        of scores — the selected g_tac* that conditions the level below)."""
        z_tac = self.phi_tac(z_op)
        goals, logits = self.goal_fan(z_tac)
        scores = self.selector(z_tac, goals, self.f_tac)
        return {"z_tac": z_tac, "goals": goals, "logits": logits,
                "scores": scores, "sel_idx": scores.argmax(dim=1)}

    def losses(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        c = self.cfg
        z_op = batch["z_op"]
        labels = batch["goal"]
        valid = batch.get("goal_valid")
        if valid is None:
            valid = torch.ones(labels.shape[:2], dtype=torch.bool,
                               device=labels.device)

        out = self.forward(z_op)
        goals, logits, scores = out["goals"], out["logits"], out["scores"]
        err = fan_goal_error(goals, labels, valid)             # [B, N]

        losses: dict[str, Tensor] = {}
        losses["wta"] = wta_regression_loss(goals, labels, valid)
        # ranking targets are the hindsight-oracle errors — detached inside
        # ranking_loss; the fan/selector only learn the ORDER.
        losses["rank_fan"] = ranking_loss(logits, err, c.margin)
        losses["rank_sel"] = ranking_loss(scores, err, c.margin)

        total = (c.w_wta * losses["wta"]
                 + c.w_rank_fan * losses["rank_fan"]
                 + c.w_rank_sel * losses["rank_sel"])

        if "z_op_next" in batch and c.w_ftac > 0:
            with torch.no_grad():                    # target: phi of the NEXT
                z_next = self.phi_tac(batch["z_op_next"])      # window, frozen
            B = goals.shape[0]
            g_label = labels.reshape(B, -1)          # teacher-goal = GT label
            z_hat = self.f_tac(out["z_tac"], g_label)
            losses["ftac"] = nn.functional.mse_loss(z_hat, z_next)
            total = total + c.w_ftac * losses["ftac"]

        if "man3" in batch:
            man3 = batch["man3"]
            for i, (name, head, w) in enumerate((
                    ("aux_lat", self.aux_lat, c.w_lat),
                    ("aux_lon", self.aux_lon, c.w_lon),
                    ("aux_lane", self.aux_lane, c.w_lane))):
                if w > 0:
                    losses[name] = nn.functional.cross_entropy(
                        head(out["z_tac"]), man3[:, i])
                    total = total + w * losses[name]

        # monitors (detached — never optimised; see class docstring).
        sel, oracle, gap = sel_gap_tac(err.detach(), out["sel_idx"])
        losses["sel_err"] = sel.mean()
        losses["oracle_err"] = oracle.mean()
        losses["sel_gap"] = gap.mean()
        losses["total"] = total
        return losses
