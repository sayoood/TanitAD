"""``--r9-bev-warp`` / ``--r9-bev-warp-lags`` / ``--r9-bev-warp-shuffle`` -- refcv9 arm T1 (REFCV9_DESIGN 4.3, 6 D3): the temporal BEV warp.
The module (warp, fuse, derangement) lives in ``tanitad/models/bev_warp.py``; this file is the flag's contract with the trainer, in the shape
``box_depth_aux_flag.py`` set: ``pin_args`` (argv refusals, a no-op when OFF), ``attach`` (build-time: the fuse module on ``model._map_hires`` and
the forward hook on its shared lift), ``attach_dataset`` + ``item_fields`` (the DataLoader-worker side: the past frames and the PAST poses),
``stage`` (the step side: the past frames' stride-8 maps and the relative transforms, staged for the hook), ``stamp``, the G-DVB checks.

ADMISSIBILITY (CLAUDE.md goal-input rule; REFCV9_DESIGN D3).  At inference the warp reads ONLY egomotion poses at ``t <= t0`` (IMU-class, causal):
``item_fields`` reads ``ep.poses[row]`` and ``ep.poses[row - lag]`` -- never ``row + 1`` -- and ``stage`` reads ONLY the four ``bw_*`` batch keys.
It reads no label, no ``future_poses*``, no ``goal_tac``, no nav, no situation-classifier output (tests/test_bev_warp.py: the future-pose
perturbation changes the T1 forward by exactly 0.0; an AST audit of the two functions' key reads).

REACHABILITY (the CLAUDE.md class "built, tested and unreachable from its caller"), one hop per line:
  argv ``--r9-bev-warp`` (default False -- NAMED in the arm's argv)  -> ``args.r9_bev_warp``  -> ``pin_args`` (refusals)
  -> ``attach`` (train() after the 10 cm branch: ``model._map_hires.bev_warp`` + a forward hook on ``model._map_hires.lift``)
  -> ``attach_dataset`` (train(): ``ds._r9_bw`` and ``e_ds._r9_bw``) -> V3Dataset.__getitem__ -> ``item_fields`` (IN THE WORKER: ``bw_*``)
  -> default collate -> ``compute_losses_perc_rig`` -> ``stage`` (BEFORE ``model(...)``) -> ``RefCV3Model.forward`` -> ``MapHiresBranch.forward``
  -> ``self.lift(fmap_s8, grid, valid)`` -> ``BevWarpFuse.lift_hook`` -> the FUSED BEV -> the shared encoder / the M3 map encoder / the banded head
  / the planner pool / the box head -> the loss -> ``.backward()`` (gate and fuse).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from tanitad.models import bev_warp as BW

FLAG = "--r9-bev-warp"
DEST = "r9_bev_warp"
LAGS_FLAG = "--r9-bev-warp-lags"
LAGS_DEST = "r9_bev_warp_lags"
LAGS_DEFAULT = "0.5,1.0"
SHUF_FLAG = "--r9-bev-warp-shuffle"
SHUF_DEST = "r9_bev_warp_shuffle"
#: the fuse hangs on the 10 cm branch (the rig's allowed root ``_map_hires``); ``refcv9_perc_rig.bev_warp_exemptions`` NAMES the prefix.
HEAD_ATTR = "bev_warp"
KEY_PREFIX = "_map_hires." + HEAD_ATTR + "."
MODEL_ATTR = "_r9_bev_warp"
#: the dataset item's keys (``item_fields``) and the ONLY batch keys ``stage`` reads.
BATCH_KEYS = ("bw_frames", "bw_pose_now", "bw_pose_past", "bw_has")
SHUFFLE_SEED = 0


@dataclass(frozen=True)
class BevWarpState:
    """What the built model carries about T1 (G-DVB reads it; ``attach_dataset`` builds the worker's source from it, never from argv)."""
    lags_s: tuple
    lag_rows: tuple
    shuffle: bool


def is_on(args) -> bool:
    return bool(getattr(args, DEST, False))


def parse_lags(text) -> tuple:
    try:
        return tuple(float(p) for p in str(text).split(",") if p.strip() != "")
    except ValueError:
        raise SystemExit("[bev-warp] %s %r: not a comma-separated list of seconds" % (LAGS_FLAG, text))


def lag_rows_of(args) -> tuple:
    try:
        return BW.lags_to_rows(parse_lags(getattr(args, LAGS_DEST, LAGS_DEFAULT) or LAGS_DEFAULT))
    except ValueError as e:
        raise SystemExit(str(e))


def _fuse(model):
    mh = getattr(model, "_map_hires", None)
    return None if mh is None else getattr(mh, HEAD_ATTR, None)


def pin_args(args) -> None:
    """The flag's refusals, at argv time. OFF: a non-default ``--r9-bev-warp-lags`` or ``--r9-bev-warp-shuffle`` alone is refused (a dead flag)."""
    if not is_on(args):
        dead = []
        if str(getattr(args, LAGS_DEST, LAGS_DEFAULT) or LAGS_DEFAULT) != LAGS_DEFAULT:
            dead.append(LAGS_FLAG)
        if bool(getattr(args, SHUF_DEST, False)):
            dead.append(SHUF_FLAG)
        if dead:
            raise SystemExit("[bev-warp] %s without %s: nothing would read it (a dead flag)" % (" / ".join(dead), FLAG))
        return
    bad = []
    try:
        lag_rows_of(args)
    except SystemExit as e:
        bad.append(str(e))
    if not bool(getattr(args, "r9_perc_rig", False)):
        bad.append("--r9-perc-rig is off: T1 stages its history in the perception rig's loss function (compute_losses_perc_rig); the full "
                   "model's loss function does not, and the hook would refuse every forward")
    if str(getattr(args, "map_hires", "off") or "off") != "on":
        bad.append("--map-hires is not `on`: the shared lift T1 fuses into exists only in the 10 cm branch")
    if str(getattr(args, "trunk", "") or "") != "timm" or str(getattr(args, "trunk_mode", "shared") or "shared") != "shared":
        bad.append("--trunk timm --trunk-mode shared is required: the past frames' stride-8 maps are read off the shared-weight backbone")
    from tanitad.train import map_only_ft as _mof
    if _mof.is_on(args):
        bad.append("a --r8-train-only-map / --r9-train-only freeze is declared: the NEW fuse would stay at its zero gate")
    if bad:
        raise SystemExit("[bev-warp] %s refused:\n  - " % FLAG + "\n  - ".join(bad))


def attach(model, args) -> None:
    """Build-time wiring, called by the trainer (and the G-EVAL loader) once the 10 cm branch exists. OFF: touches nothing.  The fuse's init draws run
    inside ``fork_rng`` so NO other parameter's draw moves (OFF / ON share every shared init)."""
    if not is_on(args):
        return
    mh = getattr(model, "_map_hires", None)
    if mh is None:
        raise SystemExit("[bev-warp] no 10 cm branch is built (--map-hires on): the fuse has no lift to hook")
    rows = lag_rows_of(args)
    lags_s = parse_lags(getattr(args, LAGS_DEST, LAGS_DEFAULT) or LAGS_DEFAULT)
    dev = next(mh.parameters()).device
    with torch.random.fork_rng(devices=[]):
        fuse = BW.BevWarpFuse(int(mh.cfg.lift_width), len(rows), float(mh.cfg.x_max_m), float(mh.cfg.y_half_m), rows)
    mh.add_module(HEAD_ATTR, fuse.to(dev))
    mh.lift.register_forward_hook(fuse.lift_hook)
    setattr(model, MODEL_ATTR, BevWarpState(tuple(lags_s), tuple(rows), bool(getattr(args, SHUF_DEST, False))))


# ---- the DataLoader-worker side -------------------------------------------------------------------------------------------------- #
class ItemSource:
    """What a DataLoader worker needs to build ONE item's history: the lags in rows and (for the shuffled control) the episode derangement.
    Read-only, picklable (spawn) and fork-safe; stored on the dataset as ``ds._r9_bw``."""

    def __init__(self, lag_rows: tuple, shuffle: bool, n_episodes: int):
        self.lag_rows = tuple(int(r) for r in lag_rows)
        self.shuffle = bool(shuffle)
        self.perm = BW.derangement(int(n_episodes), SHUFFLE_SEED) if self.shuffle else None


def _pose_row(row: int, back: int = 0) -> int:
    """The ONLY place a pose row index is computed: ``row - back``, so ``back >= 0`` is the PAST.  (The RED arm monkeypatches this to ``+ 1``.)"""
    return row - back


def item_fields(episodes, e_i: int, row: int, src: ItemSource) -> dict:
    """ONE window's history, UNbatched.  ``row`` = the window's NOW row (``t + w - 1``).  Reads ``ep.poses`` at ``row`` and ``row - lag`` ONLY
    (``_pose_row``): the NOW pose, the PAST poses, and the newest raw frame of each past row's stack (``ep.frames[row - lag][-3:]``).
    A past row before the clip starts is ``has = False`` with zero frames / the NOW pose.  The shuffled control takes the FRAMES from the donor
    episode ``perm[e_i]`` at the same row (clamped to the donor's length) and keeps the item's OWN poses: only the history CONTENT is wrong."""
    ep = episodes[e_i]
    donor = episodes[src.perm[e_i]] if src.shuffle else ep
    pose_now = ep.poses[_pose_row(row)][:4].float()
    h, w = (int(s) for s in ep.frames.shape[-2:])
    frames, poses, has = [], [], []
    for r in src.lag_rows:
        pr = _pose_row(row, r)
        dr = (min(row, int(donor.poses.shape[0]) - 1) - r) if src.shuffle else pr
        ok = pr >= 0 and dr >= 0
        has.append(ok)
        poses.append(ep.poses[pr][:4].float() if ok else pose_now)
        frames.append(donor.frames[dr][-3:] if ok else torch.zeros(3, h, w, dtype=ep.frames.dtype))      # u8 in production; the dtype of the episode
    return {"bw_frames": torch.stack(frames), "bw_pose_now": pose_now, "bw_pose_past": torch.stack(poses),
            "bw_has": torch.tensor(has, dtype=torch.bool)}


def attach_dataset(ds, model, args) -> dict:
    """Train-time wiring (``train()``, for the TRAIN and the EVAL dataset, BEFORE their DataLoaders): ``V3Dataset.__getitem__`` then adds the
    ``bw_*`` history in the worker.  The source is built from the MODEL (lags, shuffle), never from argv.  OFF: touches nothing."""
    if not is_on(args):
        return {}
    st = getattr(model, MODEL_ATTR, None)
    if st is None or _fuse(model) is None:
        raise SystemExit("[bev-warp] the model carries no temporal BEV warp: attach(model, args) must run before attach_dataset")
    n = len(getattr(ds, "episodes", None) or [])
    if st.shuffle and n < 2:
        raise SystemExit("[bev-warp] %s needs >= 2 episodes in the dataset, got %d" % (SHUF_FLAG, n))
    ds._r9_bw = ItemSource(st.lag_rows, st.shuffle, n)
    return {"lag_rows": list(st.lag_rows), "shuffle": st.shuffle, "n_episodes": n}


# ---- the step side --------------------------------------------------------------------------------------------------------------- #
def stage(model, batch: dict, device) -> None:
    """Called by ``compute_losses_perc_rig`` BEFORE ``model(...)``.  Reads ONLY ``BATCH_KEYS``.  Runs the past frames through the trunk's stride-8
    tap (``no_grad``) and stages ``(s8_past, T, has)`` on the fuse for the lift hook.  OFF (no fuse built): nothing.  A batch without the keys while
    the fuse is built REFUSES (never skips)."""
    fuse = _fuse(model)
    if fuse is None:
        return
    miss = [k for k in BATCH_KEYS if k not in batch]
    if miss:
        raise SystemExit("[bev-warp] the model is built with the temporal BEV warp but the batch carries no %s: the dataset was not attached "
                         "(attach_dataset on the train AND the eval dataset)" % miss)
    enc = model.core.encoder
    k = int(enc._mean.shape[1]) // 3
    if k > 1 and str(enc.cfg.mode) != "shared":
        raise SystemExit("[bev-warp] the trunk is not in shared mode: the past frame's stride-8 map cannot be read off a K-frame stack")
    x = batch["bw_frames"].to(device)                                  # [B, P, 3, H, W] u8
    x = x.float().div_(torch.tensor(255.0, device=x.device)) if x.dtype == torch.uint8 else x.float()
    b, p = int(x.shape[0]), int(x.shape[1])
    rows = torch.arange(b, device=x.device)
    with torch.no_grad():
        s8 = [enc.s8_from_normalised(enc.normalise(x[:, i].repeat(1, k, 1, 1)), rows) for i in range(p)]
    pn = batch["bw_pose_now"].to(device)[:, None, :3]
    T = BW.relative_se2(pn, batch["bw_pose_past"].to(device)[..., :3]).float()
    fuse.stage(torch.stack(s8, dim=1), T, batch["bw_has"].to(device))


def warm_start_prefixes(model) -> tuple:
    """New key prefixes a refcv8 source cannot carry (``refcv9_perc_rig.bev_warp_exemptions``); ``()`` unless the fuse is built."""
    return (KEY_PREFIX,) if _fuse(model) is not None else ()


def stamp(model, args) -> dict | None:
    """``config.json["r9_bev_warp"]`` -- read OFF the built fuse; ``None`` (key absent) when the flag is OFF."""
    fuse, st = _fuse(model), getattr(model, MODEL_ATTR, None)
    if fuse is None or st is None:
        return None
    return {"flag": FLAG, "lags_s": list(st.lags_s), "lag_rows": list(st.lag_rows), "dt_s": BW.DT_S, "shuffle_control": st.shuffle,
            "shuffle_rule": "frames from donor episode perm[e] at the same row (fixed derangement, torch.randperm seed 0); the item's OWN poses",
            "fusion_point": "output of model._map_hires.lift (pre-encoder), via a forward hook on the lift: feeds the shared encoder, the M3 map "
                            "encoder, hence the banded head, the planner pool and the box head",
            "fuse": "bev + gate * conv1x1(cat_p[warped_p, mask_p]); gate per-channel, zero-init (exact: gate 0 = the tip's forward)",
            "warp": "rigid SE(2) on the ground plane, bilinear, zeros outside the grid / unobserved cells + a validity channel",
            "history_grad": "detached (no_grad past lifts; BEVFormer / StreamMapNet convention)",
            "poses": "ep.poses[row] and ep.poses[row - lag] only (never a later row); PAST-only, IMU-class",
            "near_lift_note": "the 0.1 m near-lift skip (NearLiftSkip) still reads the current frame only",
            "d_bev": fuse.d_bev, "n_past": fuse.n_past, "x_max_m": fuse.x_max_m, "y_half_m": fuse.y_half_m,
            "n_params": sum(p.numel() for p in fuse.parameters()), "key_prefix": KEY_PREFIX}


# ---- G-DVB ----------------------------------------------------------------------------------------------------------------------- #
def dvb_check(model, args) -> list:
    """``built`` lever: declared <=> the fuse is built on the 10 cm branch, hooked on ITS lift, at the lift's width, with one input block per lag."""
    from tanitad.train.declared_vs_built import Mismatch
    declared, fuse = is_on(args), _fuse(model)
    out = []
    if declared != (fuse is not None):
        out.append(Mismatch(DEST, declared, fuse is not None, "model._map_hires." + HEAD_ATTR, "argv and the build disagree"))
    if fuse is not None:
        mh = model._map_hires
        if not any(h == fuse.lift_hook for h in mh.lift._forward_hooks.values()):
            out.append(Mismatch(DEST, "fuse hooked on the shared lift", "no hook", "model._map_hires.lift._forward_hooks", "built but unreachable"))
        if fuse.d_bev != int(mh.cfg.lift_width) or int(fuse.fuse.in_channels) != fuse.n_past * (fuse.d_bev + BW.MASK_CH):
            out.append(Mismatch(DEST, "d_bev %d, in %d" % (mh.cfg.lift_width, fuse.n_past * (mh.cfg.lift_width + BW.MASK_CH)),
                                "d_bev %d, in %d" % (fuse.d_bev, fuse.fuse.in_channels), "model._map_hires." + HEAD_ATTR, "widths"))
    return out


def dvb_check_lags(model, args) -> list:
    from tanitad.train.declared_vs_built import Mismatch
    declared = lag_rows_of(args)
    fuse = _fuse(model)
    built = fuse.lag_rows if fuse is not None else BW.lags_to_rows(BW.LAGS_S_DEFAULT)
    if tuple(declared) != tuple(built):
        return [Mismatch(LAGS_DEST, tuple(declared), tuple(built), "model._map_hires." + HEAD_ATTR + ".lag_rows", "argv lags and the built lags disagree")]
    return []


def dvb_check_shuffle(model, args) -> list:
    from tanitad.train.declared_vs_built import Mismatch
    declared = bool(getattr(args, SHUF_DEST, False))
    st = getattr(model, MODEL_ATTR, None)
    built = bool(st.shuffle) if st is not None else False
    if declared != built:
        return [Mismatch(SHUF_DEST, declared, built, "model." + MODEL_ATTR + ".shuffle", "argv and the state the dataset source is built from disagree")]
    return []


def register_dvb_levers(register) -> None:
    """Called by the trainer at import (the registry is pinned two-way against its parser)."""
    register(DEST, "built", dvb_check)
    register(LAGS_DEST, "built", dvb_check_lags)
    register(SHUF_DEST, "built", dvb_check_shuffle)
