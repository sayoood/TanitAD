"""Inference cascade S0-S4 and the decision rule (paper sec:prune, sec:eff). Single-sample (batch 1) reference implementation:
S0 masks -> S1 factored coarse -> S2 compose (+ guard set) -> S3 MaxSim shortlist -> S4 cross-attention re-rank -> decision."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from .model import HiCAP
from .losses import VocabTensors, node_embeddings
from .vocab import FactoredVocab, compose, slots, DT
from . import prune


@dataclass
class CascadeCfg:
    top_lat: int = 2
    top_lon: int = 3
    k_p: int = 20
    k_v: int = 10
    shortlist: int = 32
    deploy_k: int = 16
    escalate_margin: float = 0.5          # S1 margin below which K escalates deploy_k -> shortlist
    w_cost: tuple = (1.0, 1.0, 1.0)       # w_k, fixed a priori by the four-family policy
    lam: float = 1.0                      # prior weight, fitted on held-out TRAIN episodes
    lam_maxsim: float = 0.1
    lam_rerank: float = 1.0
    use_masks: bool = True
    nav_fed: bool = True


def maxsim(cand_tokens: torch.Tensor, scene: torch.Tensor) -> torch.Tensor:
    """Late interaction: sum over candidate tokens of the max cosine to any scene token. cand [K,m,d], scene [N,d] -> [K]."""
    c = F.normalize(cand_tokens, dim=-1); s = F.normalize(scene, dim=-1)
    return (c @ s.t()).max(-1).values.sum(-1)


@torch.no_grad()
def decide(m: HiCAP, V: VocabTensors, vocab: FactoredVocab, batch: dict, v0: float, a0: float, yaw: float,
           cfg: CascadeCfg = CascadeCfg(), prev_traj: np.ndarray | None = None, nav: int = 0) -> dict:
    m.eval()
    s, _, _, scene = m.encode(batch)
    A = node_embeddings(m, V)
    scene = scene[0]
    lat_cls_p = torch.as_tensor(vocab.path_cls); lon_cls_q = torch.as_tensor(vocab.prof_cls)
    # S0 masks
    pm = prune.path_mask(vocab, yaw, v0) if cfg.use_masks else np.ones(vocab.K_P, bool)
    qm = prune.profile_mask(vocab, v0, a0) if cfg.use_masks else np.ones(vocab.K_V, bool)
    # S1 classes first, then members
    zl = m.tau("s2lat") * s["s2lat"][0] @ A["s2lat"].t(); zo = m.tau("s2lon") * s["s2lon"][0] @ A["s2lon"].t()
    ok_lat = torch.as_tensor(prune.route_mask(nav, np.arange(zl.numel())) if cfg.nav_fed else np.ones(zl.numel(), bool))
    zl = zl.masked_fill(~ok_lat, -1e9)
    lp_lat = F.log_softmax(zl, -1); lp_lon = F.log_softmax(zo, -1)
    keep_lat = lp_lat.topk(min(cfg.top_lat, zl.numel())).indices; keep_lon = lp_lon.topk(min(cfg.top_lon, zo.numel())).indices
    zP = m.tau("s3P") * s["s3P"][0] @ A["s3P"].t(); zQ = m.tau("s3V") * s["s3V"][0] @ A["s3V"].t()
    liveP = torch.as_tensor(pm) & torch.isin(lat_cls_p, keep_lat); liveQ = torch.as_tensor(qm) & torch.isin(lon_cls_q, keep_lon)
    if not liveP.any(): liveP = torch.ones_like(liveP)
    if not liveQ.any(): liveQ = torch.ones_like(liveQ)
    zP_live = zP.masked_fill(~liveP, -1e9); zQ_live = zQ.masked_fill(~liveQ, -1e9)
    idxP = zP_live.topk(min(cfg.k_p, int(liveP.sum()))).indices; idxQ = zQ_live.topk(min(cfg.k_v, int(liveQ.sum()))).indices
    # cascade log-probabilities (class mass x member | class)
    def member_lp(z, cls, live):
        out = torch.full_like(z, -1e9)
        for c in cls.unique():
            mk = (cls == c) & live
            if mk.any(): out[mk] = F.log_softmax(z[mk], -1)
        return out
    lpP = member_lp(zP, lat_cls_p, liveP) + lp_lat[lat_cls_p]; lpQ = member_lp(zQ, lon_cls_q, liveQ) + lp_lon[lon_cls_q]
    # S2 compose k_p x k_v, exact feasibility, pair head, guards
    ps, qs, trajs, prior = [], [], [], []
    for p in idxP.tolist():
        for q in idxQ.tolist():
            tr = compose(vocab.paths[p], vocab.profiles[q], v0)
            if cfg.use_masks and not prune.pair_feasible(tr): continue
            ps.append(p); qs.append(q); trajs.append(tr)
            pq = m.pq_head(s["s3P"][0] * s["s3V"][0] * A["s3P"][p] * A["s3V"][q]).squeeze()
            prior.append(float(lpP[p] + lpQ[q] + pq))
    guards = prune.guard_set(v0, yaw, prev_traj)
    for g in guards:
        ps.append(-1); qs.append(-1); trajs.append(g); prior.append(float(min(prior) - 1.0) if prior else -10.0)
    trajs = np.stack(trajs).astype(np.float32); prior = np.asarray(prior, np.float32)
    prior = prior + prune.hysteresis_bonus(trajs, prev_traj)
    # S3 MaxSim shortlist (guards use a zero candidate embedding and are always kept)
    d = A["s3P"].shape[-1]; n = len(trajs)
    a_p = torch.stack([A["s3P"][p] if p >= 0 else torch.zeros(d) for p in ps]); a_q = torch.stack([A["s3V"][q] if q >= 0 else torch.zeros(d) for q in qs])
    cls_p = torch.tensor([int(vocab.path_cls[p]) if p >= 0 else 0 for p in ps]); cls_q = torch.tensor([int(vocab.prof_cls[q]) if q >= 0 else 0 for q in qs])
    a_cell = F.normalize(A["s2lat"][cls_p] + A["s2lon"][cls_q], dim=-1)
    ms = maxsim(torch.stack([a_p, a_q, a_cell], 1), scene)
    s2score = torch.as_tensor(prior) + cfg.lam_maxsim * ms
    is_guard = torch.tensor([p < 0 for p in ps])
    margin = float((s2score.topk(min(2, n)).values[0] - s2score.topk(min(2, n)).values[-1])) if n > 1 else 9.0
    K = cfg.shortlist if margin < cfg.escalate_margin else cfg.deploy_k
    order = torch.argsort(-(s2score + 1e3 * is_guard.float()))
    sel = order[: min(K, n)]
    if not bool(is_guard[sel].any()):                              # guard set is never dropped: keep the best guard
        gbest = torch.where(is_guard)[0][s2score[is_guard].argmax()]; sel = torch.cat([sel[:-1], gbest.view(1)])
    # S4 re-rank
    sl = torch.as_tensor(slots(trajs[sel.numpy()]))
    cost, logit = m.rerank(a_p[sel][None], a_q[sel][None], a_cell[sel][None], sl[None], scene[None], cls_p[sel][None])
    cost, logit = cost[0], logit[0]
    w = torch.tensor(cfg.w_cost)
    obj = (cost * w).sum(-1) - cfg.lam * (s2score[sel] - torch.logsumexp(s2score[sel], 0)) - cfg.lam_rerank * logit
    best = int(obj.argmin())
    chosen = int(sel[best])
    # refinement on the chosen open candidate
    if ps[chosen] >= 0:
        r = m.refine(torch.cat([s["s3P"][0] * A["s3P"][ps[chosen]], s["s3V"][0] * A["s3V"][qs[chosen]]], -1)).view(-1, 2).numpy()
    else:
        r = np.zeros((len(slots(trajs[:1])[0]), 2), np.float32)
    return dict(traj=trajs[chosen], slots=slots(trajs[chosen]) + r, path=ps[chosen], profile=qs[chosen], is_guard=bool(is_guard[chosen]),
                n_composed=n, k_final=len(sel), s1_margin=margin, cell=(int(cls_p[chosen]), int(cls_q[chosen])),
                lat_probs=lp_lat.exp().numpy(), lon_probs=lp_lon.exp().numpy())
