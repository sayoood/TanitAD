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
  -> default collate -> ``compute_losses_perc_rig`` (the perception rig) OR ``compute_losses_v3`` (the FULL model; refcv9 T1-FULL) -> ``stage``
  (BEFORE ``model(...)``, in both) -> ``RefCV3Model.forward`` -> ``MapHiresBranch.forward``
  -> ``self.lift(fmap_s8, grid, valid)`` -> ``BevWarpFuse.lift_hook`` -> the FUSED BEV -> the shared encoder / the M3 map encoder / the banded head
  / the planner pool / the box head -> the loss -> ``.backward()`` (gate and fuse).

T1h (``--r9-bev-warp-supervise-hist``, refcv9 E1; REQUIRES ``--r9-bev-warp``): the 10 cm map loss's lift-valid mask becomes ``lift_valid_now OR hist_valid``, where
``hist_valid`` is the UNION of the lags' warp-valid masks (``bev_warp.hist_valid_of``: GEOMETRY ONLY -- poses, the camera's own ``valid`` and ``has``; never the warped
content, so ``--r9-bev-warp-shuffle`` has the SAME supervision set).  WHY: T1's gate never opened (|g| 0.0077) because the loss only sees cells the camera reaches NOW, and the
cells history could fix (0-5 m: 94.7 % unobserved now, 42 % covered only by warped history) were masked out of it.  REACHABILITY, one hop per line:
  argv ``--r9-bev-warp-supervise-hist`` (default False -- NAMED in the arm's argv) -> ``args.r9_bev_warp_supervise_hist`` -> ``pin_args`` (refusals)
  -> ``attach`` (``fuse.emit_hist = True``; ``BevWarpState.supervise_hist``) -> ``BevWarpFuse.lift_hook`` (``self._hist_valid = hist_valid_of(masks)``)
  -> ``MapHiresBranch.forward`` (``take_hist_valid()`` -> ``out["map_hires_hist_valid"]``) -> ``RefCV3Model._bev_hook`` (``pout["map_hires_hist_valid"]``)
  -> ``refc_v3_train._map_hires_loss`` (``lv = _bw.supervised_lift_valid(model, pout, lv, sel)``) -> ``map_hires_loss_row(lift_valid_025=lv)`` -> the CE -> ``.backward()``.
OFF: no key is emitted, ``supervised_lift_valid`` returns ``lv`` itself, and the loss is bit-identical.

T1h + the BANDED map head (K2, ``--r9-bev-warp-supervise-hist`` WITH ``--r9-map-banded``; refcv9 E1 follow-on): the banded loss never reads ``map_hires_lift_valid``; it masks each band by the head's
``unobs`` (``map_banded.unobs_masks(lift_valid)``: near = the 0.25 m cell its 10 cm centre falls in (nearest-exact), mid = the lift cell itself, far = observed only when ALL FOUR of its 0.25 m cells
are), ``supervised = seen & ~unobs``, and the same ``unobs`` also gates the max-entropy term.  ``hist_valid`` is at the SAME 0.25 m lift grid ([B, X4, Y4]), so the OR is taken THERE, exactly where
T1h defines it, and the head's OWN resampling then does the rest: ``unobs_loss = unobs_masks(lift_valid_now | hist_valid)`` (``supervised_banded_unobs``).  Consequences, all by construction: the
near band's supervised cells are the ones T1h's 10 cm loss supervises (the same nearest-exact upsample of the same 0.25 m OR); mid is the OR itself; a far 0.5 m cell is supervised when every one of
its four lift cells is seen NOW or in HISTORY (the head's conservative ALL rule, widened -- never "any one cell"); the supervised set only GROWS and the max-entropy set only SHRINKS by the same
cells (a cell the loss now trusts is no longer also pulled to p = 0.5).  The head's EMITTED ``unobs`` (``perception.map_banded["unobs"]``: the plan wires and the eval dumps read it) is NOT changed --
the OR is loss-side only.  REACHABILITY, one hop per line:
  argv ``--r9-bev-warp-supervise-hist`` + ``--r9-map-banded`` -> ``pin_args`` (no banded refusal any more) -> ``attach`` (``fuse.emit_hist``; ``BevWarpState.supervise_hist``)
  -> ``BevWarpFuse.lift_hook`` (``self._hist_valid``) -> ``MapHiresBranch.forward`` (``take_hist_valid()`` -> ``out_b["map_hires_hist_valid"]`` beside ``map_banded``)
  -> ``RefCV3Model._bev_hook`` (``pout["map_hires_hist_valid"]``) -> ``refc_v3_train._map_banded_loss`` (``_bw.supervised_banded_unobs``) -> ``banded_map_loss`` (``seen & ~unobs``, the max-entropy mask) -> ``.backward()``.
OFF: ``supervised_banded_unobs`` returns the emitted list ITSELF (bit-identical), no counter is logged.

T1h-INIT (``--r9-bev-warp-fuse-init {rezero,identity}``, default ``rezero`` = today, bit-identical; REQUIRES ``--r9-bev-warp`` when not the default): MEASURED by the MM on E1 arm T1 --
the trained 1x1 fuse stayed at its PyTorch default init and the gate random-walked (a ReZero DEAD START: random W -> no consistent gate push -> gate ~ 0 -> W's gradient ~ gate ~ 0).
``identity`` writes the COPY init ``W[c, p*(C+1)+c] = 1/P`` (``BevWarpFuse.init_identity_``: fuse(x) = the mean of the lags' warped past BEV) at FRESH BUILD, inside ``attach``, with
the gate still exactly 0 (the step-0 forward is the baseline's).  ``attach`` runs BEFORE every checkpoint load (train(): resume's strict ``load_state_dict`` / the ``--init-from`` warm
start, which EXEMPTS the fuse keys; the eval loader's strict load), so a trained fuse is never overwritten by the init.  REACHABILITY: argv ``--r9-bev-warp-fuse-init`` ->
``args.r9_bev_warp_fuse_init`` -> ``pin_args`` -> ``attach`` (``fuse.init_identity_()``: it draws NO random number, and the default init's draws stay inside the build's ``fork_rng``, so no other parameter moves) -> ``fuse.fuse.weight`` -> the lift hook's
``mix`` -> the fused BEV -> every loss -> ``.backward()`` (G-DVB reads ``fuse.init_mode``, which only ``init_identity_`` writes).

T1h-FILL (``--r9-bev-warp-fill-only``, default off = today, bit-identical; REQUIRES ``--r9-bev-warp``): MEASURED by the MM post hoc on the gradmask probes' saved per-window gate gradients
(GADD_CONSISTENCY.txt) -- the gate gradient from the HISTORY-ONLY cells alone is sign-consistent (0.383 trained-W / 0.391 identity-W, above the sign-flip null p99 0.30 / 0.34), while the gradient over ALL
supervised cells sits at the null because the ~10x more camera-seen cells, where history is redundant, swamp it.  So the gated history term is applied ONLY where the CURRENT lift is invalid:
``out = bev + gate * fuse(x) * (1 - valid_now)`` (``BevWarpFuse.mix``; ``valid_now`` = the lift's own ``valid.any(dim=1)`` that ``lift_hook`` already holds and passes ONLY when ``fuse.fill_only``).
REACHABILITY: argv ``--r9-bev-warp-fill-only`` -> ``args.r9_bev_warp_fill_only`` -> ``pin_args`` -> ``attach`` (``fuse.fill_only = True``) -> ``BevWarpFuse.lift_hook`` (``self.mix(..., valid_now=src_valid)``)
-> ``mix`` (the ``(1 - valid_now)`` factor) -> the fused BEV -> every loss -> ``.backward()``; G-DVB reads ``fuse.fill_only``; ``mix`` REFUSES a fill-only call without ``valid_now``.

EVERY forward of a T1-built model stages its history or REFUSES (never a silent un-warped forward): the loss functions stage; the eval loader's
``build_eval_dataset`` attaches the history to the eval dataset; a bare ``model(...)`` with nothing staged raises in ``BevWarpFuse.lift_hook``.
``--ablate-frames`` (the information-free control) is REFUSED with T1 built (``refuse_ablate``): it replaces only the CURRENT window, so the
history would still carry the scene and the control would no longer be information-free.
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
FILL_FLAG = "--r9-bev-warp-fill-only"
FILL_DEST = "r9_bev_warp_fill_only"
FUSE_INIT_FLAG = "--r9-bev-warp-fuse-init"
FUSE_INIT_DEST = "r9_bev_warp_fuse_init"
FUSE_INIT_DEFAULT = "rezero"
SUP_FLAG = "--r9-bev-warp-supervise-hist"
SUP_DEST = "r9_bev_warp_supervise_hist"
#: the forward's emitted key (``MapHiresBranch.forward`` -> ``pout``): [B, X4, Y4] bool, ABSENT unless T1h is built.
HIST_KEY = "map_hires_hist_valid"
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
    supervise_hist: bool = False         # T1h: the map loss supervises ``lift_valid_now | hist_valid`` (``supervised_lift_valid``)


def is_on(args) -> bool:
    return bool(getattr(args, DEST, False))


def fuse_init_of(args) -> str:
    """The declared fuse init (``rezero`` | ``identity``); anything else REFUSES."""
    v = str(getattr(args, FUSE_INIT_DEST, FUSE_INIT_DEFAULT) or FUSE_INIT_DEFAULT)
    if v not in BW.FUSE_INITS:
        raise SystemExit("[bev-warp] %s %r: not one of %s" % (FUSE_INIT_FLAG, v, list(BW.FUSE_INITS)))
    return v


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


def is_built(model) -> bool:
    """True iff the temporal BEV warp (the fuse on ``model._map_hires``) is built into ``model``."""
    return _fuse(model) is not None


ABLATE_MSG = ("--ablate-frames replaces only the CURRENT window with a constant; the T1 history (past frames) would still carry the scene, so the "
              "information-free control would no longer be information-free")


def refuse_ablate(model) -> None:
    """The runtime half of the ``--ablate-frames`` refusal (``compute_losses_v3`` calls it when ``ablate_frames`` is set; ``pin_args`` is the argv half --
    the loader-parsed argv of an eval entry point never passes through ``pin_args``).  A model with no fuse: nothing."""
    if is_built(model):
        raise SystemExit("[bev-warp] %s" % ABLATE_MSG)


def pin_args(args) -> None:
    """The flag's refusals, at argv time. OFF: a non-default ``--r9-bev-warp-lags`` or ``--r9-bev-warp-shuffle`` alone is refused (a dead flag).
    ON: the 10 cm branch, the timm shared trunk, no freeze, no ``--ablate-frames``.  The perception RIG is NOT required any more (refcv9 T1-FULL: the full
    model's loss function stages the history too)."""
    if not is_on(args):
        dead = []
        if str(getattr(args, LAGS_DEST, LAGS_DEFAULT) or LAGS_DEFAULT) != LAGS_DEFAULT:
            dead.append(LAGS_FLAG)
        if bool(getattr(args, SHUF_DEST, False)):
            dead.append(SHUF_FLAG)
        if bool(getattr(args, SUP_DEST, False)):
            dead.append(SUP_FLAG)
        if fuse_init_of(args) != FUSE_INIT_DEFAULT:
            dead.append(FUSE_INIT_FLAG)
        if bool(getattr(args, FILL_DEST, False)):
            dead.append(FILL_FLAG)
        if dead:
            raise SystemExit("[bev-warp] %s without %s: nothing would read it (a dead flag)" % (" / ".join(dead), FLAG))
        return
    bad = []
    try:
        lag_rows_of(args)
    except SystemExit as e:
        bad.append(str(e))
    try:
        fuse_init_of(args)
    except SystemExit as e:
        bad.append(str(e))
    if bool(getattr(args, "ablate_frames", False)):
        bad.append(ABLATE_MSG)
    if str(getattr(args, "map_hires", "off") or "off") != "on":
        bad.append("--map-hires is not `on`: the shared lift T1 fuses into exists only in the 10 cm branch")
    if str(getattr(args, "trunk", "") or "") != "timm" or str(getattr(args, "trunk_mode", "shared") or "shared") != "shared":
        bad.append("--trunk timm --trunk-mode shared is required: the past frames' stride-8 maps are read off the shared-weight backbone")
    from tanitad.train import map_only_ft as _mof
    if _mof.is_on(args):
        bad.append("a --r8-train-only-map / --r9-train-only freeze is declared: the NEW fuse would stay at its zero gate")
    if bool(getattr(args, SUP_DEST, False)):
        # K2: `--r9-map-banded` is ADMITTED now -- `supervised_banded_unobs` ORs hist_valid into the banded loss's `unobs` (the refusal that stood here is gone; the other refusals stay)
        if getattr(args, "map_lift_valid_mask", True) is False:
            bad.append("%s with --no-map-lift-valid-mask: every labelled cell is already supervised, the OR has nothing to add (a dead flag)" % SUP_FLAG)
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
    sup = bool(getattr(args, SUP_DEST, False))
    fuse.emit_hist = sup                                     # T1h: the hook records the warp-valid union; OFF: nothing is computed
    fuse.fill_only = bool(getattr(args, FILL_DEST, False))     # T1h-FILL: the history term only where the current lift is invalid; OFF: today's mix
    if fuse_init_of(args) == "identity":
        fuse.init_identity_()                                # T1h-INIT: the COPY init, at FRESH BUILD only (every later checkpoint load overwrites it); default: nothing
    mh.add_module(HEAD_ATTR, fuse.to(dev))
    mh.lift.register_forward_hook(fuse.lift_hook)
    setattr(model, MODEL_ATTR, BevWarpState(tuple(lags_s), tuple(rows), bool(getattr(args, SHUF_DEST, False)), sup))


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
def supervise_hist_on(model) -> bool:
    """True iff the built model carries T1h (``--r9-bev-warp-supervise-hist``): the loss's mask is ``lift_valid_now | hist_valid``."""
    st = getattr(model, MODEL_ATTR, None)
    return bool(st is not None and getattr(st, "supervise_hist", False))


def supervised_lift_valid(model, pout: dict, lv, sel):
    """The 10 cm map loss's lift-valid mask under T1h.  ``lv`` = the loss's CURRENT mask (``pout["map_hires_lift_valid"]`` already ``index_select(0, sel)``; ``None`` when
    ``--no-map-lift-valid-mask`` = every labelled cell is supervised already) -> ``lv | pout["map_hires_hist_valid"].index_select(0, sel)``.  OFF (no T1h built, or ``lv`` None):
    returns ``lv`` ITSELF (the same object: bit-identical).  ON with no ``map_hires_hist_valid`` in the forward's output REFUSES (never silently the un-widened mask)."""
    if lv is None or not supervise_hist_on(model):
        return lv
    hv = pout.get(HIST_KEY)
    if hv is None:
        raise SystemExit("[bev-warp] %s is built but the forward's perception output carries no `%s`: the loss would silently supervise only the cells the camera "
                         "reaches NOW (the lift hook / MapHiresBranch.forward / bev_hook did not emit it)" % (SUP_FLAG, HIST_KEY))
    return lv | hv.index_select(0, sel)


def supervised_banded_unobs(model, pout: dict, unobs: list, sel, geom: tuple) -> list:
    """K2 (T1h + the banded head): the banded loss's per-band ``unobs`` under T1h.  ``unobs`` = the head's EMITTED masks, already ``index_select(0, sel)`` (``[near, mid, far]``, True = NOT
    observed); ``pout`` = the forward's perception output (``map_hires_lift_valid`` and ``map_hires_hist_valid``, both ``[B, X4, Y4]`` bool at the 0.25 m lift grid); ``geom`` = the BUILT head's
    band geometry.  -> ``map_banded.unobs_masks(lift_valid_now | hist_valid, geom)``: the OR at the lift grid (where T1h's own OR is defined), then the head's OWN resampling (near: nearest-exact to
    0.10 m; mid: the cell itself; far: observed only when ALL FOUR 0.25 m cells are).  OFF (no T1h built): returns ``unobs`` ITSELF (the same list: bit-identical).  ON with no
    ``map_hires_hist_valid`` / ``map_hires_lift_valid`` in the forward's output REFUSES (never silently the un-widened mask).  Two runtime guards: the loss-side recomputation WITHOUT the history
    must equal the head's emitted ``unobs`` exactly (else the OR is not 'in addition to what is supervised today'), and the widened mask must be a subset of it (the OR only ever ADDS observed cells)."""
    if not supervise_hist_on(model):
        return unobs
    hv, lv = pout.get(HIST_KEY), pout.get("map_hires_lift_valid")
    if hv is None or lv is None:
        raise SystemExit("[bev-warp] %s is built with the banded head but the forward's perception output carries no `%s` / `map_hires_lift_valid`: the banded loss would silently supervise only the "
                         "cells the camera reaches NOW (MapHiresBranch.forward / bev_hook did not emit them)" % (SUP_FLAG, HIST_KEY))
    lv, hv = lv.index_select(0, sel), hv.index_select(0, sel)
    if tuple(lv.shape) != tuple(hv.shape) or lv.dtype != torch.bool or hv.dtype != torch.bool:
        raise SystemExit("[bev-warp] %s: lift_valid %s %s and hist_valid %s %s must be the same [B, X4, Y4] bool grid" % (SUP_FLAG, tuple(lv.shape), lv.dtype, tuple(hv.shape), hv.dtype))
    from tanitad.models import map_banded as MB
    base, wide = MB.unobs_masks(lv, geom), MB.unobs_masks(lv | hv, geom)
    for g, e, b, w in zip(geom, unobs, base, wide):
        if not torch.equal(e, b):
            raise SystemExit("[bev-warp] %s: band %s -- the head's emitted `unobs` is not unobs_masks(map_hires_lift_valid): the OR would not be 'in addition to what is supervised today'" % (SUP_FLAG, g.name))
        if bool((w & ~e).any()):
            raise SystemExit("[bev-warp] %s: band %s -- the widened mask UN-observes a cell the head observes (the OR may only add cells)" % (SUP_FLAG, g.name))
    return wide


def stage(model, batch: dict, device) -> None:
    """Called by ``compute_losses_perc_rig`` AND by ``compute_losses_v3`` (the FULL model) BEFORE ``model(...)``.  Reads ONLY ``BATCH_KEYS``.  Runs the past frames through the trunk's stride-8
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
            "n_params": sum(p.numel() for p in fuse.parameters()), "key_prefix": KEY_PREFIX,
            **({"supervise_hist": {"flag": SUP_FLAG, "rule": "map loss lift-valid mask = lift_valid_now OR hist_valid (T1h)",
                                   "hist_valid": "union over lags of the warp-valid mask (inb & warped past `valid` & has): GEOMETRY ONLY, content-independent (shuffle keeps it)",
                                   "emitted_as": "pout[%r] from MapHiresBranch.forward" % HIST_KEY,
                                   "sites": "refc_v3_train._map_hires_loss only; the legacy 0.5 m map_valid sites carry no mask under --map-hires (map_hires_pool)",
                                   **({"banded": "K2: refc_v3_train._map_banded_loss -- the per-band `unobs` = unobs_masks(lift_valid_now OR hist_valid) at the 0.25 m lift grid (near nearest-exact, mid identity, "
                                                 "far ALL FOUR cells); supervision grows and the max-entropy set shrinks by the same cells; the emitted `unobs` is unchanged"}
                                      if getattr(getattr(model, "_map_hires", None), "banded", None) is not None else {})}}
               if st.supervise_hist else {}),
            **({"fill_only": {"flag": FILL_FLAG, "rule": "out = bev + gate * fuse(x) * (1 - valid_now); valid_now = the lift's own valid.any(dim=1) at the fuse's resolution",
                              "effect": "a cell the camera sees NOW is exactly the lift's BEV, whatever the gate"}} if fuse.fill_only else {}),
            **({"fuse_init": {"flag": FUSE_INIT_FLAG, "mode": fuse.init_mode, "rule": "W[c, p*(C+1)+c] = 1/P, every other entry 0 (mask columns included): fuse(x) = the mean of the lags' warped past BEV; gate exactly 0",
                                "applied": "at fresh build (attach), before any checkpoint load; a trained fuse is never overwritten"}}
               if fuse.init_mode != FUSE_INIT_DEFAULT else {})}


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


def dvb_check_fuse_init(model, args) -> list:
    """T1h-INIT ``built`` lever: the declared init == ``fuse.init_mode`` (only ``BevWarpFuse.init_identity_`` writes ``identity``, together with the weights).  No fuse: ``rezero``."""
    from tanitad.train.declared_vs_built import Mismatch
    declared, fuse = fuse_init_of(args), _fuse(model)
    built = str(getattr(fuse, "init_mode", FUSE_INIT_DEFAULT)) if fuse is not None else FUSE_INIT_DEFAULT
    if declared != built:
        return [Mismatch(FUSE_INIT_DEST, declared, built, "model._map_hires." + HEAD_ATTR + ".init_mode", "argv and the build disagree on how the fuse was initialised")]
    return []


def dvb_check_fill_only(model, args) -> list:
    """T1h-FILL ``built`` lever: declared <=> ``fuse.fill_only`` (the flag that makes ``mix`` apply the ``(1 - valid_now)`` factor).  No fuse: False."""
    from tanitad.train.declared_vs_built import Mismatch
    declared, fuse = bool(getattr(args, FILL_DEST, False)), _fuse(model)
    built = bool(getattr(fuse, "fill_only", False)) if fuse is not None else False
    if declared != built:
        return [Mismatch(FILL_DEST, declared, built, "model._map_hires." + HEAD_ATTR + ".fill_only", "argv and the build disagree on where the history term is applied")]
    return []


def dvb_check_supervise_hist(model, args) -> list:
    """T1h ``built`` lever: declared <=> the model state carries ``supervise_hist`` AND the fuse emits ``hist_valid`` (the two halves of the hop the loss reads)."""
    from tanitad.train.declared_vs_built import Mismatch
    declared = bool(getattr(args, SUP_DEST, False))
    st, fuse = getattr(model, MODEL_ATTR, None), _fuse(model)
    s_built = bool(st.supervise_hist) if st is not None else False
    e_built = bool(getattr(fuse, "emit_hist", False)) if fuse is not None else False
    out = []
    if declared != s_built:
        out.append(Mismatch(SUP_DEST, declared, s_built, "model." + MODEL_ATTR + ".supervise_hist", "argv and the state the loss reads disagree"))
    if s_built != e_built:
        out.append(Mismatch(SUP_DEST, s_built, e_built, "model._map_hires." + HEAD_ATTR + ".emit_hist", "the loss mask is widened but the fuse emits no hist_valid (or the reverse)"))
    return out


def register_dvb_levers(register) -> None:
    """Called by the trainer at import (the registry is pinned two-way against its parser)."""
    register(DEST, "built", dvb_check)
    register(LAGS_DEST, "built", dvb_check_lags)
    register(SHUF_DEST, "built", dvb_check_shuffle)
    register(SUP_DEST, "built", dvb_check_supervise_hist)
    register(FUSE_INIT_DEST, "built", dvb_check_fuse_init)
    register(FILL_DEST, "built", dvb_check_fill_only)
