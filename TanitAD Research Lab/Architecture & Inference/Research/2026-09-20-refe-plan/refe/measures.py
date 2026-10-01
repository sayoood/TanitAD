"""Training-side measures M3 / M4a / M4b for REFe -- EVERY ONE DEFAULT OFF.

PI directive 2026-09-27: implement all measures, but a measure is used in the live run ONLY after its
effectiveness AND validity are proven. This file is the implementation; `selftest_measures.py` is the
validity proof; `measures_hooks.patch` is the only change to `model.py` / `train.py` (a few lines each,
all gated). Nothing here reaches the live pod, which runs its own copy of the trainer.

THE FINDING THESE ADDRESS (EXPLORATORY, eval/raw/e6_sub200_ep015/): best of the 64 proposals fell
91.33 -> 84.19 PDMS from snapshot 012 to 015 (paired CI [-10.79, -4.01], oracle_trend.json), while the
mean of 64 stayed flat; every slot's plan got ~4.3 m longer over 4 s at the first on-policy epoch, so
scenes that need a slow plan have none (standing still beats all 64 on 16 of 200 tokens).

M3  SEPARATE GRADIENT CLIPPING. `train.py` clips the norm of ALL trainable parameters together at 1.0,
    so the scorer's gradient norm sets the clip factor the trajectory generator receives. With the
    scorer detached from the proposals AND from the visual context (`model.py` forward), the two
    parameter groups receive gradient from DISJOINT losses; clipping them separately removes the only
    coupling left. ⚠️ Under AdamW a CONSTANT clip factor cancels (m and sqrt(v) scale together); only a
    time-varying factor changes the update, and then only until v re-adapts (beta2 = 0.999). The
    expected size of M3's effect is therefore an EMPIRICAL question -- log the group norms first.
M4a SLOWED-TARGET TWINS: offline, `slow_twins.py` (uses `slow_copies.slow_copy`, the ONE shared copy).
    The trainer-side part is here: the twin row marker and the rank a twin's scorer supervision is
    looked up under (its SOURCE rank -- same image, ego, goal and route, so the same labelled set).
M4b ANCHORED SLOW SLOTS. K of the 64 proposal slots are reserved. Each reserved slot keeps its OWN
    learned path (the raw proposal the decoder emits for it), but its TIMING is pinned to an analytic
    anchor profile: constant deceleration from the ego's measured speed v0 to a target fraction of v0,
    then hold. The slot cannot be pulled faster than its anchor by construction -- its arc length at
    every horizon step IS the anchor's -- which is the property the shift above violated.
    Supervision mirrors DiffusionDrive's anchored decoder (arXiv 2411.15139 p.5-6, banked in
    Library/papers): "We assign the noisy trajectory around the closest anchor to the ground truth
    trajectory as positive sample", L = sum_k [y_k L_rec + lambda BCE]. Here the positive reserved slot
    is the one whose anchor PROFILE is closest to the teacher's, and its regression target is the
    TEACHER'S PATH driven at that anchor's profile (the loss cannot move the timing, so regressing to the
    raw teacher points would only bend the path to cut corners). The 64 - K free slots keep plain
    winner-takes-all, the SAME `model.wta_loss`, over the free slots only.
    DiffusionDriveV2 (arXiv 2512.07745 p.2, banked) names the mechanism behind the shift: imitation
    learning optimises "only the parameters of the single positive mode", so negative modes are
    unconstrained -- 63 of 64 slots here, which moved with the shared weights.
    ⚠️ DEPARTURE FROM DIFFUSIONDRIVE, stated: their anchors are K-means (x, y) shapes with a free learned
    offset; ours anchor the SPEED PROFILE only and leave the path free, because a bounded (x, y) offset
    around a straight or clustered anchor cannot follow a curved road at low speed, and an unbounded one
    can be pulled away -- exactly the failure being fixed.
    NO NEW PARAMETERS AND NO BUFFERS: the reserved slots reuse their existing query, decoder and
    `traj_head`, so a mid-run switch keeps every state-dict key, the optimiser's parameter list and the
    scheduler unchanged; it is one declared identity change (`--declare-change slow_slots`).

MID-RUN SWITCH -- what a resume from the LIVE checkpoint needs (pinned by selftest_measures.py T9):
  code        measures.py + measures_hooks.patch on model.py / train.py. The pod runs ITS OWN copy: compare
              its sha256 with the patch header's BASE first, or re-cut the hunks. With every flag off the
              patched trainer resumes today's checkpoint BIT-IDENTICALLY (T9: max state diff 0.0).
  consumers   measures_consumers.patch on ckpt_io.py for EVERY consumer (planner eval, onpolicy_dump,
              probes) BEFORE the first M4b snapshot is evaluated or dumped -- today's loader ignores the
              measure and runs the RAW reserved slots (T9 HAZARD: 11.26 m past the anchor, tiny rig).
  parameters  none: state-dict keys, optimiser parameter list, scheduler and frozen fingerprint are
              unchanged, so ckpt_io.load_partial and opt.load_state_dict need nothing.
  identity    every flag is a resume-identity key that an old checkpoint reads as OFF. Switch with
              --declare-change clip_split | slow_slots [+ slow_slot_indices / slow_slot_profiles /
              slow_slot_w when not default] | slow_twins. Undeclared -> REFUSING TO RESUME (rc 4);
              declared -> a `declared_change` event in metrics.jsonl and meta["measures"] in every later
              checkpoint. Order: kill the SUPERVISOR first, swap code, restart the supervisor with the flags.
  M4a extra   twin rank files go into the --grow bank dir (slow_twins.py --into-live-bank) only AFTER the
              patched trainer runs: today's trainer absorbs a new targets_rank*.jsonl at its next epoch
              boundary with no flag (T9 HAZARD, MEASURED); the patched one refuses it without --slow-twins.
              Twins join at the NEXT epoch boundary (the resumed epoch keeps its checkpointed snapshot).
              Keep appending twins for rows grow_assemble adds later (the tool is append-only). Never add
              twin ranks to onpolicy_dump --ranks (the labeller would treat the slowed twin as the teacher).
  M4b extra   reserve the LEAST-USED slots (--slow-slot-indices), not the default last K; see from_args.
  M6 extra    --yaw-loss plain --declare-change yaw_loss. The trainer refuses it when any target heading exceeds
              3.0 rad (checked at start AND at every --grow rebuild). The trapped heading row then moves to the
              principal branch, so the SCORER's input (the detached fan, heading included) shifts: re-read its
              selection skill; on-policy sets labelled before the switch carry the trapped headings.
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass

import torch

__all__ = [
    "SCORER_PREFIXES", "clip_groups", "check_clip_premise", "clip_grad_norm_split", "group_grad_norms",
    "DEFAULT_SLOW_PROFILES", "parse_profiles", "SlowSlotConfig", "anchor_profiles", "wrap_angle",
    "retime_to_profile", "progress_profile", "AnchoredSlowSlots", "install_slow_slots",
    "install_from_meta", "anchored_wta_loss", "TWIN_KEY", "is_twin", "scorer_rank", "check_twins",
    "run_meta", "source_sha256", "YAW_MODES", "wta_loss_yaw", "check_plain_yaw_targets",
]


# =============================================================================================== M3
# The scoring branch, by module. Everything else that trains is the trajectory generator (backbone LoRA,
# registers, pos3d MLP, register compression, scene projection, ego/goal encoder, queries, trajectory
# decoder, trajectory head). ⭐ Verified by GRADIENT FLOW, not only by name, in selftest_measures.py:
# the score loss alone leaves every trajectory-group gradient exactly zero and vice versa.
SCORER_PREFIXES = ("score_q_mlp.", "score_dec.", "score_head.")


def clip_groups(model) -> dict:
    """{'traj': [(name, p)], 'score': [(name, p)]} over the TRAINABLE parameters, in module order.

    Every trainable parameter lands in exactly one group (asserted); a scorer group that comes out
    EMPTY is refused, because then 'separate clipping' would silently be the old global clip.
    """
    groups: dict = {"traj": [], "score": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        groups["score" if n.startswith(SCORER_PREFIXES) else "traj"].append((n, p))
    n_train = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    if len(groups["traj"]) + len(groups["score"]) != n_train:
        raise AssertionError("clip groups do not partition the trainable parameters")
    if not groups["score"] or not groups["traj"]:
        raise ValueError(f"clip groups: traj {len(groups['traj'])} / score {len(groups['score'])} "
                         f"parameters -- a split with an empty side is the global clip in disguise")
    return groups


def check_clip_premise(model) -> None:
    """M3 is only a DECOUPLING when the score loss cannot reach the trajectory group at all.

    With `detach_scorer_context=False` (the `--scorer-sees-trunk` ablation) the PDM loss backpropagates
    into `scene_proj` and the backbone LoRA -- trajectory-group parameters -- so a split clip would still
    let the scorer's loss change the trajectory generator's gradient. Refuse rather than pretend.
    """
    if not bool(getattr(model.cfg, "detach_scorer_context", True)):
        raise ValueError("M3 (--clip-split) needs detach_scorer_context=True: with the scorer's visual "
                         "context attached, the score loss reaches scene_proj and the backbone LoRA, which "
                         "are trajectory-group parameters, so no clip split can isolate the two losses")


def clip_grad_norm_split(groups: dict, max_traj: float = 1.0, max_score: float = 1.0):
    """Clip each group to its own max norm, with torch's own `clip_grad_norm_`.

    Returns the two PRE-clip norms as 0-dim tensors (no device sync here; convert when logging).
    """
    nt = torch.nn.utils.clip_grad_norm_([p for _, p in groups["traj"]], float(max_traj))
    ns = torch.nn.utils.clip_grad_norm_([p for _, p in groups["score"]], float(max_score))
    return nt, ns


def group_grad_norms(groups: dict) -> dict:
    """READ-ONLY L2 norms of each group's gradient and of both together (clip_grad_norm_'s definition:
    the norm of the per-parameter norms). Never writes a gradient -- pinned bit-exact by the self-test."""
    out = {}
    per = {}
    for g, lst in groups.items():
        norms = [p.grad.detach().float().norm(2) for _, p in lst if p.grad is not None]
        per[g] = torch.stack(norms) if norms else torch.zeros(0)
        out[g] = torch.linalg.vector_norm(per[g], 2) if norms else torch.zeros(())
    both = [v for v in per.values() if v.numel()]
    out["all"] = torch.linalg.vector_norm(torch.cat([b.to(both[0].device) for b in both]), 2) \
        if both else torch.zeros(())
    return out


# =============================================================================================== M4b
# (decel m/s^2, target speed as a fraction of v0). "inf" = reach the target speed immediately.
# Chosen to span what the lost-mode analysis needs: standing still (the STOP candidate that beat all 64
# on 16/200 tokens), three stops of different firmness, and four slower-cruise plans. A PROPOSAL, not
# a measured optimum -- the effectiveness experiment is what judges it.
DEFAULT_SLOW_PROFILES = "inf:0,6:0,3:0,1.5:0,2:0.25,2:0.5,2:0.75,1:0.875"
EGO_SPEED_INDEX = 6     # ego = [vx, vy, ax, ay, yaw_rate, steering, speed] -- build_teacher_rollouts.py:381-383


def parse_profiles(spec: str) -> tuple:
    """'a:f,a:f,...' -> ((a, f), ...). a > 0 (m/s^2, 'inf' allowed), 0 <= f < 1."""
    out = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        a_s, f_s = item.split(":")
        a = math.inf if a_s.strip().lower() in ("inf", "infinity") else float(a_s)
        f = float(f_s)
        if not (a > 0.0) or math.isnan(a):
            raise ValueError(f"slow-slot profile {item!r}: deceleration must be > 0 (or inf)")
        if not (0.0 <= f < 1.0):
            raise ValueError(f"slow-slot profile {item!r}: target fraction of v0 must be in [0, 1)")
        out.append((a, f))
    if not out:
        raise ValueError("no slow-slot profiles given")
    return tuple(out)


def _fmt_profiles(profiles) -> str:
    return ",".join(f"{'inf' if math.isinf(a) else repr(float(a))}:{float(f)!r}" for a, f in profiles)


@dataclass(frozen=True)
class SlowSlotConfig:
    slots: tuple                 # reserved proposal indices (default: the LAST K)
    profiles: tuple              # ((decel, fraction), ...), one per reserved slot, same order
    w_res: float = 1.0           # weight of the reserved-slot regression term
    yaw_w: float = 0.1           # the same yaw weight as model.wta_loss
    speed_index: int = EGO_SPEED_INDEX
    dt: float = 0.2              # horizon sample spacing (planner.TRAJ_DT_S)

    def validate(self, n_proposals: int, horizon: int, traj_dim: int, ego_dim: int) -> None:
        if len(self.slots) != len(self.profiles):
            raise ValueError(f"{len(self.slots)} reserved slots but {len(self.profiles)} profiles")
        if len(set(self.slots)) != len(self.slots):
            raise ValueError(f"reserved slots repeat: {self.slots}")
        if not all(0 <= int(s) < n_proposals for s in self.slots):
            raise ValueError(f"reserved slots {self.slots} outside [0, {n_proposals})")
        if len(self.slots) >= n_proposals:
            raise ValueError("at least one free slot must remain for winner-takes-all")
        if traj_dim != 3:
            raise ValueError(f"anchored slow slots need (x, y, heading) trajectories, traj_dim={traj_dim}")
        if not (0 <= self.speed_index < ego_dim):
            raise ValueError(f"speed index {self.speed_index} outside the {ego_dim}-D ego vector")
        if horizon < 2:
            raise ValueError("horizon too short")
        parse_profiles(_fmt_profiles(self.profiles))          # re-validate the numbers

    @classmethod
    def from_args(cls, k: int, profiles_spec: str, n_proposals: int, w_res: float = 1.0,
                  indices: str = ""):
        """`indices` '' = the LAST k slots; else k comma-separated slot numbers. ⭐ For a MID-RUN switch name
        the least-used slots: EXPLORATORY at snapshot 015 (sub200), slots 56-63 held the scorer's pick on
        6/200 tokens and the best of 64 on 9/200, while [8, 16, 19, 21, 22, 34, 38, 39] held neither
        (best of 64 84.194 -> 84.173 without 56-63, unchanged without those 8). Choose them on tokens
        DISJOINT from the effectiveness eval, or the selection leaks into the readout."""
        profiles = parse_profiles(profiles_spec)
        if int(k) != len(profiles):
            raise ValueError(f"--slow-slots {k} needs exactly {k} profiles in --slow-slot-profiles, got "
                             f"{len(profiles)} ({profiles_spec!r})")
        if str(indices).strip():
            slots = tuple(int(x) for x in str(indices).split(",") if x.strip())
            if len(slots) != int(k):
                raise ValueError(f"--slow-slot-indices names {len(slots)} slots, --slow-slots is {k}")
        else:
            slots = tuple(range(n_proposals - int(k), n_proposals))
        return cls(slots=slots, profiles=profiles, w_res=float(w_res))

    def to_meta(self) -> dict:
        return {"version": 1, "slots": [int(s) for s in self.slots],
                "profiles": _fmt_profiles(self.profiles), "w_res": float(self.w_res),
                "yaw_w": float(self.yaw_w), "speed_index": int(self.speed_index), "dt": float(self.dt)}

    @classmethod
    def from_meta(cls, d: dict):
        if int(d.get("version", 0)) != 1:
            raise ValueError(f"unknown slow-slot meta version {d.get('version')!r}")
        return cls(slots=tuple(int(s) for s in d["slots"]), profiles=parse_profiles(d["profiles"]),
                   w_res=float(d["w_res"]), yaw_w=float(d["yaw_w"]),
                   speed_index=int(d["speed_index"]), dt=float(d["dt"]))


def anchor_profiles(v0: torch.Tensor, profiles, T: int, dt: float = 0.2) -> torch.Tensor:
    """Arc length S[b, k, i] travelled by t_i = (i + 1) dt under profile k from speed v0[b].

    Constant deceleration a_k from v0 to f_k * v0, then constant speed:
        t1 = (1 - f) v0 / a;  S(t) = v0 t - a t^2 / 2  (t < t1),  (1 + f) v0 t1 / 2 + f v0 (t - t1)  (t >= t1)
    a = inf gives t1 = 0 (the target speed at once; f = 0 is standing still). v0 is clamped at 0.
    Starts at the ego's MEASURED speed, so -- unlike a time-rescaled copy -- it asks for no speed step.
    """
    v0 = v0.clamp(min=0.0)
    dev, dt_ = v0.device, v0.dtype
    t = dt * torch.arange(1, T + 1, device=dev, dtype=dt_)                           # [T]
    a = torch.tensor([p[0] for p in profiles], device=dev, dtype=dt_)                 # [K]
    f = torch.tensor([p[1] for p in profiles], device=dev, dtype=dt_)                 # [K]
    vb = v0[:, None]                                                                  # [B, 1]
    vt = f[None, :] * vb                                                              # [B, K]
    t1 = (vb - vt) / a[None, :]                                                       # [B, K]
    s1 = 0.5 * (vb + vt) * t1                                                         # braking distance
    tt = t[None, None, :]                                                             # [1, 1, T]
    brake = vb[..., None] * tt - 0.5 * a[None, :, None] * tt * tt                     # -inf when a = inf
    cruise = s1[..., None] + vt[..., None] * (tt - t1[..., None])
    return torch.where(tt < t1[..., None], brake, cruise)


def wrap_angle(a):
    """to [-pi, pi) -- the same floor-modulo rule as model.wta_loss and slow_copies._wrap."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def retime_to_profile(traj: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-6,
                      detach_weights: bool = False) -> torch.Tensor:
    """The SAME PATH, re-timed so that pose i sits at arc length sigma[..., i] along it.

    traj [..., T, 3] (x, y, heading; ego frame; the origin (0, 0, 0) at t = 0 is prepended, exactly as
    `slow_copies` does) and sigma [..., T] (>= 0, non-decreasing) -> [..., T, 3]. Positions are linear in
    arc length along the polyline origin -> p_1 -> ... -> p_T; the heading is the start heading plus the
    WRAPPED difference times the weight (the seam converter's rule, as in slow_copies). Beyond the end
    of the path the pose continues straight along the LAST HEADING. Differentiable in `traj`.
    `eps` floors every segment length so a stationary (zero-length) path has a defined parametrisation.

    `detach_weights=True` (what the M4b forward uses): WHERE each pose falls -- segment, weight, the
    extrapolation direction and distance -- is computed from a DETACHED copy, so any gradient reaches the
    raw poses only through convex interpolation weights (each <= 1). Forward values are identical.
    ⚠️ No training loss is taken THROUGH this re-timing (anchored_wta_loss regresses the raw proposal);
    selftest T7 keeps both rejected designs as mutation arms that must fail (see anchored_wta_loss). The
    detach only keeps an accidental consumer's gradient bounded.
    """
    if traj.shape[-1] != 3:
        raise ValueError(f"expected [..., T, 3], got {tuple(traj.shape)}")
    if tuple(sigma.shape) != tuple(traj.shape[:-1]):
        raise ValueError(f"sigma {tuple(sigma.shape)} must be traj's leading shape {tuple(traj.shape[:-1])}")
    T = traj.shape[-2]
    P = torch.cat([traj.new_zeros(traj.shape[:-2] + (1, 3)), traj], dim=-2)          # [..., T+1, 3]
    Pw = P.detach() if detach_weights else P                                          # where poses fall
    d = Pw[..., 1:, :2] - Pw[..., :-1, :2]                                            # [..., T, 2]
    seg = torch.sqrt((d * d).sum(-1) + eps * eps)                                     # [..., T] > 0
    c = torch.cat([seg.new_zeros(seg.shape[:-1] + (1,)), torch.cumsum(seg, dim=-1)], dim=-1)  # [..., T+1]
    sig = sigma.to(c.dtype)
    s = torch.searchsorted(c[..., 1:].contiguous(), sig.contiguous()).clamp(max=T - 1)  # segment index
    c0 = torch.gather(c, -1, s)
    c1 = torch.gather(c, -1, s + 1)
    w = ((sig - c0) / (c1 - c0)).clamp(0.0, 1.0)
    i0 = s.unsqueeze(-1).expand(*s.shape, 3)
    p0 = torch.gather(P, -2, i0)
    p1 = torch.gather(P, -2, i0 + 1)
    xy_in = p0[..., :2] + w.unsqueeze(-1) * (p1[..., :2] - p0[..., :2])
    h_in = wrap_angle(p0[..., 2] + w * wrap_angle(p1[..., 2] - p0[..., 2]))
    cT = c[..., -1:]                                                                  # [..., 1]
    pT = P[..., -1:, :]                                                               # [..., 1, 3]
    hT = Pw[..., -1:, 2]                                                              # direction [..., 1]
    ext = (sig - cT).clamp(min=0.0)                                                   # [..., T]
    xy_out = pT[..., :2] + ext.unsqueeze(-1) * torch.stack([torch.cos(hT), torch.sin(hT)], dim=-1)
    beyond = sig > cT
    xy = torch.where(beyond.unsqueeze(-1), xy_out, xy_in)
    h = torch.where(beyond, pT[..., 2].expand_as(h_in), h_in)
    return torch.cat([xy, h.unsqueeze(-1)], dim=-1)


def progress_profile(traj: torch.Tensor) -> torch.Tensor:
    """Cumulative CHORD length origin -> p_1 -> ... -> p_i: [..., T, 3] -> [..., T]. An independent
    measurement of how far a trajectory travels by each step (chords never exceed the arc they span)."""
    P = torch.cat([traj.new_zeros(traj.shape[:-2] + (1, 3)), traj], dim=-2)
    return torch.cumsum((P[..., 1:, :2] - P[..., :-1, :2]).norm(dim=-1), dim=-1)


class AnchoredSlowSlots(torch.nn.Module):
    """Forward hook of M4b: the reserved slots' raw proposals, re-timed to their anchor profiles.

    ⛔ NO PARAMETERS AND NO BUFFERS, deliberately -- the state dict, the optimiser's parameter list and
    the frozen fingerprint are identical with and without it, which is what makes a mid-run switch a
    declared identity change instead of a checkpoint migration.
    """

    def __init__(self, cfg: SlowSlotConfig):
        super().__init__()
        self.cfg = cfg

    def anchors(self, ego: torch.Tensor, T: int) -> torch.Tensor:
        return anchor_profiles(ego[:, self.cfg.speed_index].float(), self.cfg.profiles, T, self.cfg.dt)

    def forward(self, traj: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        # FP32 regardless of autocast: under bf16 a 60 m coordinate has ~0.25 m resolution
        with torch.autocast(device_type=traj.device.type, enabled=False):
            t32 = traj.float()
            idx = torch.tensor(self.cfg.slots, device=traj.device, dtype=torch.long)
            S = self.anchors(ego, t32.shape[-2])                                     # [B, K, T]
            raw = t32.index_select(1, idx)                                           # [B, K, T, 3]
            # the reserved slots' RAW proposals, WITH their graph: `anchored_wta_loss` trains these
            # (never through the re-timing -- see there). A plain attribute: not state, not a buffer.
            self.last_raw = raw
            anchored = retime_to_profile(raw, S, detach_weights=True)                # [B, K, T, 3]
            return t32.index_copy(1, idx, anchored)

    def extra_repr(self) -> str:
        return f"slots={list(self.cfg.slots)}, profiles={_fmt_profiles(self.cfg.profiles)}"


def install_slow_slots(model, cfg: SlowSlotConfig) -> AnchoredSlowSlots:
    """Attach the hook `model.forward` calls (see measures_hooks.patch). Refuses a second, different one."""
    c = model.cfg
    cfg.validate(c.n_proposals, c.horizon_steps, c.traj_dim, c.ego_dim)
    have = getattr(model, "slow_slots", None)
    if have is not None:
        if have.cfg != cfg:
            raise ValueError(f"model already carries different slow slots: {have.cfg} vs {cfg}")
        return have
    n_state = len(model.state_dict())
    model.slow_slots = AnchoredSlowSlots(cfg)
    if len(model.state_dict()) != n_state:                 # the no-new-state promise, checked
        raise AssertionError("installing the slow-slot hook changed the state dict")
    return model.slow_slots


def install_from_meta(model, meta) -> bool:
    """For every CONSUMER of a checkpoint (planner, on-policy dump, probes): a model trained with M4b
    must be run WITH it -- its reserved slots' raw outputs were never trained as trajectories."""
    ss = ((meta or {}).get("measures") or {}).get("slow_slots") if isinstance(meta, dict) else None
    if not ss:
        return False
    install_slow_slots(model, SlowSlotConfig.from_meta(ss))
    return True


def anchored_wta_loss(traj: torch.Tensor, target: torch.Tensor, ego: torch.Tensor,
                      cfg: SlowSlotConfig, wta_fn, yaw_w: float = 0.1, info: dict | None = None,
                      raw_reserved: torch.Tensor | None = None, yaw_mode: str = "wrapped"):
    """(loss, winner index) -- the M4b replacement for `wta_loss(traj, target)`.

    free slots     `wta_fn` (model.wta_loss, unchanged) over the 64 - K free slots only;
    reserved slots the one whose ANCHOR is closest to the target (DiffusionDrive's positive rule): its
                   RAW proposal (`raw_reserved`, the hook's `last_raw`) regressed to the TARGET's path
                   driven at that anchor's profile, with the same point-L1 + yaw term as `wta_loss`.
    `traj` is the forward's output (reserved slots already anchored). The winner index is the FREE
    winner in global slot numbering (what the trainer's 'winners' log counts).

    ⛔ WHY THE RAW PROPOSAL AND NOT THE ANCHORED OUTPUT -- two designs FAILED the self-test and are kept there
    as mutation arms (T7, tiny model, 200 steps on fast-only targets; best free-slot WTA 31.35 m at init,
    0.50 m with M4b OFF, 0.68 m with this design; raw/2026-09-27-training-measures/selftest_measures.json):
      * regressing the ANCHORED output with the full gradient: that gradient runs through the arc-length
        parametrisation (divisions by short segments, extrapolation levers) and dominates the shared
        decoder -> free slots stalled at 7.81 m;
      * the same with detached re-timing weights: a straight path re-timed is INVARIANT to its poses
        sliding along it, the detached gradient does not know that and pushes them forward forever ->
        the raw reserved paths grew to 1,333 m through the shared head and the free slots stalled at 27.05 m.
    Regressing the raw proposal is an ordinary point regression (no degenerate direction); the forward's
    re-timing still pins the TIMING exactly, whatever the raw proposal does. `raw_reserved=None` uses
    `traj`'s reserved slots (unit tests that hand-build a fan).
    """
    B, M, T, _ = traj.shape
    dev = traj.device
    res = torch.tensor(cfg.slots, device=dev, dtype=torch.long)
    rr = traj.index_select(1, res) if raw_reserved is None else raw_reserved.float()
    if tuple(rr.shape) != (B, len(cfg.slots), T, 3):
        raise ValueError(f"raw reserved proposals {tuple(rr.shape)} do not match the fan "
                         f"({B}, {len(cfg.slots)}, {T}, 3) -- a stale hook output?")
    keep = torch.ones(M, dtype=torch.bool, device=dev)
    keep[res] = False
    free = torch.nonzero(keep).squeeze(1)
    l_free, i_free = wta_fn(traj.index_select(1, free), target, yaw_w=yaw_w)
    with torch.no_grad():
        S = anchor_profiles(ego[:, cfg.speed_index].float(), cfg.profiles, T, cfg.dt)      # [B, K, T]
        K = S.shape[1]
        tk = retime_to_profile(target.float().unsqueeze(1).expand(B, K, T, 3).contiguous(), S)
        d_anc = (tk[..., :2] - target.float()[:, None, :, :2]).abs().sum(-1).mean(-1)      # [B, K]
        k_star = d_anc.argmin(dim=1)                                                       # [B]
    ar = torch.arange(B, device=dev)
    out = rr[ar, k_star]                                                                   # [B, T, 3]
    g = tk[ar, k_star]
    l_pos = (out[..., :2] - g[..., :2]).abs().sum(-1).mean(-1).mean()
    dyaw = out[..., 2] - g[..., 2]
    l_yaw = (wrap_angle(dyaw) if yaw_mode == "wrapped" else dyaw).abs().mean()       # M6: follow --yaw-loss
    l_res = l_pos + yaw_w * l_yaw
    if info is not None:
        info.update({"l_free": l_free.detach(), "l_res": l_res.detach(),
                     "d_anc_min": d_anc.min(dim=1).values.mean(), "k_star": k_star})
    return l_free + cfg.w_res * l_res, free[i_free]


# =============================================================================================== M6 heading loss
# ⛔ THE WRAPPED HEADING L1 TRAPS UNBOUNDED HEADING OUTPUTS ON THE WRONG 2*pi BRANCH (MEASURED 2026-09-27).
# `model.wta_loss` computes |wrap(yaw_pred - yaw_target)|: periodic in the prediction, so a heading output on ANY
# branch k (pred = target + 2*pi*k) scores as correct, and the plain Linear `traj_head` is not bounded to one branch.
# Live run: native step 19's heading is corrupted in EVERY snapshot 005-015 (sub200 E-6 tables: raw -21..+1 rad,
# 60-72 % beyond +-pi, corr -0.94 with the plan length), while steps 0-18 stay within +-2.1 rad; its output row's norm
# grows 0.546 (ep001, init) -> 1.539 (ep015) while every other heading row stays 0.54-0.65. Nothing else is special
# about step 19: the targets are clean there (61,821 local live-bank rows: max wrap error vs the path tangent 0.137
# rad), and the loss gives it the same gradient as every step. ⭐ The tiny rig with the REAL trainer on REAL targets
# traps FIVE other steps (3, 7, 10, 11, 15) within its first epoch and never frees them (the winners' wrapped error
# stays 1.3-2.1 rad for 15 epochs), with the same signature -- which step is caught is dynamics, not index 19. Once a
# heading output varies over several branches with the plan length, the wrapped error of the winners is
# pseudo-random in sign and averages to no corrective gradient.
# ⭐ FIX (`--yaw-loss plain`): the SAME winner and position term; the heading L1 WITHOUT the wrap. Every REFe target
# heading is wrapped at build time and lies far from the +-pi seam (max |target| 2.46 rad over those rows), so on this
# data the wrap bought nothing but the branch ambiguity; unwrapped, a branch-shifted output is penalised in a
# CONSISTENT direction. `check_plain_yaw_targets` refuses the mode on a bank that approaches the seam.
YAW_MODES = ("wrapped", "plain", "plain_tangent")
# M6b (PREREG_M6B_TANGENT): 'plain_tangent' = 'plain' + a TARGET-FREE term on ALL slots pulling each slot's heading toward
# the tangent of its OWN predicted path (stop-gradient target; positions untouched). Under WTA only the winner gets a
# heading gradient, so M6's plain loss left the trapped branch in rarely-winning slots (MEASURED 2026-09-27: P0e2 still
# 1.70 % of all raw step-19 headings beyond +-pi); this term reaches every slot.
TAN_MIN_STEP = 0.2                # m per native step (1 m/s at 5 Hz): below it the tangent is undefined -> masked
TAN_W_DEFAULT = 0.1
PLAIN_YAW_MAX_TARGET = 3.0        # rad: above this a target is near the seam and an unwrapped L1 would fight it


def tangent_targets(traj: torch.Tensor):
    """[..., T, 3] -> (theta [..., T], mask [..., T]), both detached. The tangent of the slot's OWN positions, ego frame,
    p_{-1} = (0, 0): the central difference (p_{t+1} - p_{t-1}) / 2 for t < T-1, the backward difference at T-1.
    Masked where the step is < TAN_MIN_STEP or |theta| > PLAIN_YAW_MAX_TARGET (near +-pi atan2 itself flips branch)."""
    xy = traj[..., :2].detach()
    p = torch.cat([torch.zeros_like(xy[..., :1, :]), xy], dim=-2)            # p[k] = position at native step k-1
    T = xy.shape[-2]
    d = torch.cat([(p[..., 2:T + 1, :] - p[..., 0:T - 1, :]) / 2.0,
                   (p[..., T:T + 1, :] - p[..., T - 1:T, :])], dim=-2)       # [..., T, 2]
    theta = torch.atan2(d[..., 1], d[..., 0])
    mask = (d.norm(dim=-1) >= TAN_MIN_STEP) & (theta.abs() <= PLAIN_YAW_MAX_TARGET)
    return theta, mask


def tangent_loss(traj: torch.Tensor) -> torch.Tensor:
    """mean over the masked (b, m, t) of |heading - stopgrad(own tangent)|, PLAIN (not wrapped): a heading 2*pi off is
    pulled back to the principal branch. The target is detached, so positions get NO gradient from this term."""
    theta, mask = tangent_targets(traj)
    m = mask.to(traj.dtype)
    return ((traj[..., 2] - theta).abs() * m).sum() / m.sum().clamp(min=1.0)


def wta_loss_yaw(traj: torch.Tensor, target: torch.Tensor, yaw_w: float = 0.1, mode: str = "wrapped",
                 wta_fn=None, tan_w: float = TAN_W_DEFAULT):
    """'wrapped' = `wta_fn` (model.wta_loss) EXACTLY -- today's loss; 'plain' = the same winner selection and position
    term (the same ops, so the same bits), heading L1 without the 2*pi wrap."""
    if mode == "wrapped":
        return wta_fn(traj, target, yaw_w=yaw_w)
    if mode not in ("plain", "plain_tangent"):
        raise ValueError(f"yaw loss mode {mode!r} not in {YAW_MODES}")
    pos = (traj[..., :2] - target[..., :2].unsqueeze(1)).abs().sum(-1).mean(-1)       # [B, M] (= wta_loss)
    idx = pos.argmin(dim=1)
    ar = torch.arange(pos.shape[0], device=pos.device)
    l_pos = pos[ar, idx].mean()
    dy = traj[ar, idx, :, 2] - target[..., 2]                                          # NOT wrapped
    loss = l_pos + yaw_w * dy.abs().mean()
    if mode == "plain_tangent":
        loss = loss + tan_w * tangent_loss(traj)                                        # ALL slots, target-free
    return loss, idx


def check_plain_yaw_targets(rows) -> float:
    """max |target heading| over a bank's rows; raises ValueError if any approaches the +-pi seam."""
    mx = 0.0
    for r in rows:
        for p in r["traj"]:
            mx = max(mx, abs(float(p[2])))
    if mx > PLAIN_YAW_MAX_TARGET:
        raise ValueError(f"a target heading reaches {mx:.3f} rad (> {PLAIN_YAW_MAX_TARGET}): near the +-pi seam an "
                         f"unwrapped heading L1 would fight the wrap -- use the wrapped loss or a sin/cos head")
    return mx


# =============================================================================================== M4a
TWIN_KEY = "slow"        # a twin row carries {"factor", "src_rank", ...} under this key (slow_twins.py)


def is_twin(row: dict) -> bool:
    return isinstance(row.get(TWIN_KEY), dict)


def scorer_rank(row: dict) -> int:
    """The rank a row's SCORER supervision is keyed under. A twin shares its source's image, ego, goal
    and route, so its labelled proposal set IS its source's; every other row: its own rank (unchanged)."""
    s = row.get(TWIN_KEY)
    if isinstance(s, dict):
        return int(s["src_rank"])
    return int(row.get("rank", 0))


def check_twins(rows: list, allow: bool) -> dict:
    """Count twin rows; REFUSE them unless the run declared `--slow-twins`.

    ⛔ Why a refusal and not a warning: under `--grow` a new `targets_rank*.jsonl` in the bank directory
    joins the NEXT epoch silently (train.py: a rank file that appears after the snapshot "belongs to a
    later epoch"). Without this guard a twin file dropped into a live bank changes the recipe with no
    flag, no identity change and no event -- the silent-arm class this package keeps finding.
    """
    by_rank: dict = {}
    for r in rows:
        if is_twin(r):
            k = int(r.get("rank", 0))
            by_rank[k] = by_rank.get(k, 0) + 1
    n = sum(by_rank.values())
    if n and not allow:
        raise SystemExit(f"TargetBank: {n:,} SLOWED-TARGET TWIN rows (ranks {sorted(by_rank)}) are in the "
                         f"bank but this run did not pass --slow-twins. Refusing rather than training on "
                         f"them silently (M4a is DEFAULT OFF; a mid-run switch needs --slow-twins "
                         f"--declare-change slow_twins).")
    return {"n_twins": n, "by_rank": dict(sorted(by_rank.items()))}


# =============================================================================================== run meta
def run_meta(clip_split: bool, clip_traj: float, clip_score: float, slow_cfg, slow_twins: bool,
             yaw_loss: str = "wrapped"):
    """What a checkpoint's meta records about the ACTIVE measures (None when all are off, so an OFF run's
    meta is byte-for-byte what it was). Consumers call `install_from_meta` on it."""
    m = {}
    if clip_split:
        m["clip_split"] = {"max_traj": float(clip_traj), "max_score": float(clip_score),
                           "scorer_prefixes": list(SCORER_PREFIXES)}
    if slow_cfg is not None:
        m["slow_slots"] = slow_cfg.to_meta()
    if slow_twins:
        m["slow_twins"] = True
    if yaw_loss != "wrapped":
        m["yaw_loss"] = yaw_loss
    return m or None


def source_sha256(path: str | None = None) -> str:
    p = path or os.path.abspath(__file__)
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
