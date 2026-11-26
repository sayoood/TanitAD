"""ENS-SCORE (PREREG_ENSSCORE.md section 2.1): an equal-weight two-member deep ensemble of REFe's scoring heads. Parameter-free.

Per token: the PRIMARY planner A (the seam's own checkpoint) gives its complete planned set traj_A [64, 20, 3] and its scoring context
sctx_A; the MEMBER model B (the other training seed, its own trunk) runs its own forward on the SAME inputs and gives sctx_B. Both
members score A's complete set with their own heads (`REFe.score_trajectories(traj_A, sctx)`, the modules the forward applies), each
term's logit becomes a probability (sigmoid, as `REFePlanner.aggregate` does), the two probabilities are averaged with EQUAL weights,
and the planner's OWN navsim_v1 aggregate is applied to the averaged probabilities: k_ens = argmax. The SUBMITTED trajectory is A's
PLANNED proposal k_ens (the caller executes it).

Why probabilities (card 2.1): the heads are trained per term with binary cross-entropy on logits (`refe/train.py:1919`), so each
output is a Bernoulli logit; `REFePlanner.aggregate` (`refe/planner.py:537-575`) consumes sigmoid(logit) per term. The average of the
members' per-term probabilities is the ensemble's predictive probability for that term (the deep-ensemble mixture), and the aggregate is
then applied to it unchanged. `aggregate_p` feeds the averaged probabilities through the planner's own `aggregate` via a proxy whose
`.sigmoid()` returns them, so the aggregate's code is never re-typed here (bitwise identity pinned by test_ens_score.py).
"""
from __future__ import annotations

import time

import numpy as np

N_PROP, N_STEP, TRAJ_DIM, N_TERMS = 64, 20, 3, 6
RED_ROLL = 1            # CTRL-RED: member B's probabilities misaligned by ONE proposal index (torch.roll along the set axis)


class _Prob:
    """A stand-in for a logit tensor whose `.sigmoid()` is the given probabilities: `REFePlanner.aggregate`'s first line is
    `p = score.sigmoid()` and everything after reads `p` only (planner.py:551-575), so aggregate(_Prob(p)) is the planner's own rule on p."""
    __slots__ = ("p",)

    def __init__(self, p):
        self.p = p

    def sigmoid(self):
        return self.p


def ens_probs(pa, pb):
    """THE pre-registered combination: the equal-weight mean of the two members' per-term probabilities, (pA + pB) / 2.
    (x + x) / 2 == x exactly in IEEE float, so a member ensembled with itself is the member bit for bit (CTRL-ID (a))."""
    return (pa + pb) / 2


def aggregate_p(planner, p):
    """[M, 6] probabilities -> [M] the planner's own aggregate, through the same [1, M, 6] shape path as REFePlanner._pick."""
    return planner.aggregate(_Prob(p.unsqueeze(0)))[0]


def pick_p(planner, p) -> int:
    return int(aggregate_p(planner, p).argmax())


def pick_logits(planner, s) -> int:
    """The planner's own rule on a [M, 6] logit tensor: aggregate([1, M, 6])[0].argmax(), as REFePlanner._pick."""
    if getattr(planner, "select", "best") != "best":
        raise ValueError(f"ENS-SCORE is defined for select='best' only, got {planner.select!r}")
    return int(planner.aggregate(s.unsqueeze(0))[0].argmax())


def check_set(traj, model_cfg=None):
    if traj.dim() != 3 or tuple(traj.shape[1:]) != (N_STEP, TRAJ_DIM) or traj.shape[0] != N_PROP:
        raise ValueError(f"ENS-SCORE needs the COMPLETE set [{N_PROP}, {N_STEP}, {TRAJ_DIM}], got {tuple(traj.shape)}")
    if model_cfg is not None:
        got = (model_cfg.n_proposals, model_cfg.horizon_steps, model_cfg.traj_dim)
        if got != (N_PROP, N_STEP, TRAJ_DIM):
            raise ValueError(f"model config (n_proposals, horizon_steps, traj_dim) = {got}, ENS-SCORE is registered for {(N_PROP, N_STEP, TRAJ_DIM)}")


def check_scores(s, who: str):
    if s.dim() != 2 or tuple(s.shape) != (N_PROP, N_TERMS):
        raise ValueError(f"ENS-SCORE: {who} scores must be [{N_PROP}, {N_TERMS}] (the 6-term navsim_v1 scorer), got {tuple(s.shape)}")


def check_members(pa, pb):
    """CTRL-CFG: the member must be the SAME recipe as the primary (only the weights may differ) and use the same selection settings."""
    if pa.cfg != pb.cfg:
        diff = {k: (getattr(pa.cfg, k), getattr(pb.cfg, k)) for k in vars(pa.cfg) if getattr(pa.cfg, k) != getattr(pb.cfg, k, None)}
        raise ValueError(f"ENS-SCORE: member config differs from the primary's: {diff}")
    for k in ("rule", "select", "repair_last_heading", "per_sample_calib", "trained"):
        if getattr(pa, k) != getattr(pb, k):
            raise ValueError(f"ENS-SCORE: member {k}={getattr(pb, k)!r} != primary {getattr(pa, k)!r}")
    if pa.rule != "navsim_v1" or pa.select != "best":
        raise ValueError(f"ENS-SCORE is registered for rule navsim_v1 / select best, got {pa.rule!r} / {pa.select!r}")


def ens_step(pa, pb, traj, score, k_ship: int, sctx_a, traj_b, score_b, k_b_own: int, sctx_b, to_navsim) -> dict:
    """ONE token. `traj` [64, 20, 3] / `score` [64, 6]: the primary's forward outputs; `traj_b` / `score_b`: the member's own forward on
    the same inputs; `sctx_*` the scoring contexts those forwards used. rec['k_ens'] is the index to SUBMIT (A's PLANNED proposal)."""
    import torch
    t0 = time.time()
    check_set(traj, getattr(pa.model, "cfg", None))
    check_set(traj_b, getattr(pb.model, "cfg", None))
    with torch.no_grad():
        s_a = pa.model.score_trajectories(traj.unsqueeze(0), sctx_a)[0]          # member A on A's set (== the forward's own scores: CTRL-ID)
        s_b = pb.model.score_trajectories(traj.unsqueeze(0), sctx_b)[0]          # member B on A's COMPLETE set, B's own context
        s_bo = pb.model.score_trajectories(traj_b.unsqueeze(0), sctx_b)[0]       # member B on B's own set (CTRL-ID (b))
        for s, w in ((s_a, "member A"), (s_b, "member B"), (score, "primary forward"), (score_b, "member forward")):
            check_scores(s, w)
        p_a, p_b = s_a.sigmoid(), s_b.sigmoid()
        a_ens = aggregate_p(pa, ens_probs(p_a, p_b))
        k_ens = int(a_ens.argmax())
        rec = {"k_ship": int(k_ship), "k_ens": k_ens,
               "k_id": pick_p(pa, ens_probs(p_a, p_a)),                                    # CTRL-ID (a): A ensembled with ITSELF
               "k_red": pick_p(pa, ens_probs(p_a, torch.roll(p_b, RED_ROLL, 0))),          # CTRL-RED: member B misaligned by one index
               "k_a_only": pick_p(pa, p_a), "k_b_only": pick_p(pa, p_b),                   # descriptive (M5)
               "k_b_own": int(k_b_own), "k_b_own_id": pick_logits(pb, s_bo),
               "id_maxdiff_a": float((s_a.float() - score.float()).abs().max()),
               "id_maxdiff_b": float((s_bo.float() - score_b.float()).abs().max()),
               "s_a": s_a.float().cpu().numpy().astype(np.float32), "s_b": s_b.float().cpu().numpy().astype(np.float32),
               "s_b_fwd": score_b.float().cpu().numpy().astype(np.float32), "a_ens": a_ens.float().cpu().numpy().astype(np.float32),
               "ship8": np.asarray(to_navsim(pa.executed(traj, int(k_ship)).float().cpu().numpy()), dtype=np.float32),
               "ship8_b": np.asarray(to_navsim(pb.executed(traj_b, int(k_b_own)).float().cpu().numpy()), dtype=np.float32),
               "props8_b": np.stack([to_navsim(pb.executed(traj_b, j).float().cpu().numpy()) for j in range(traj_b.shape[0])]).astype(np.float32)}
    rec["t_ens"] = time.time() - t0
    return rec
