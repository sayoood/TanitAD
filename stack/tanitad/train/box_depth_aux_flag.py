"""``--r9-box-depth-aux`` / ``--w-box-depth`` -- refcv9 arm B1 (REFCV9_DESIGN 2.3, 4.3, 6b D4): the training-only box-derived depth
auxiliary head on the trunk's s8 + s16 maps.  The head, targets and loss live in ``tanitad/models/box_depth_aux.py``; the builder
is the pinned byte copy ``tanitad/data/box_depth_targets.py``.  This module is the flag's contract with the trainer, in the shape
``map_banded_flag.py`` set: ``pin_args`` (argv refusals, a no-op when OFF), ``attach`` (build-time: the head on ``model._perception``
+ the weight attribute), ``loss_term`` (the loss block the rig's loss function calls), ``stamp``, the two G-DVB checks and
``register_dvb_levers``.

REACHABILITY (the CLAUDE.md class "built, tested and unreachable from its caller"), one hop per line:
  argv ``--r9-box-depth-aux`` / ``--w-box-depth``  (refc_v3_train.py build_parser, defaults False / 0.0 -- BOTH named in the arm's argv)
  -> ``args.r9_box_depth_aux`` / ``args.w_box_depth``  -> ``pin_args`` (refusals) -> ``attach`` (train(), after the perception branch:
  ``model._perception.box_depth_aux`` + ``model._w_box_depth``) -> ``attach_dataset`` (train(), before the DataLoader: ``ds._r9_bd``)
  -> V3Dataset.__getitem__ -> ``item_targets`` (IN THE DATALOADER WORKER: the cylindrical ray-cast targets of the item's own cuboids)
  -> default collate -> ``compute_losses_perc_rig`` (the rig) OR ``compute_losses_v3`` (the full model, AUX-FULL: the same call, right
  after the 10 cm map term) -> ``loss_term`` (reads ``out["fmap_s8"]``, ``out["fmap_s16"]`` and the batch's ``bd_*``
  targets) -> ``loss = loss + term`` -> the trainer's ``.backward()``.
The head is NEVER called by ``RefCV3Model.forward``: the eval forward is the OFF forward, bit for bit (tests/test_box_depth_aux.py).

WHERE THE TARGETS ARE BUILT (refcv9 C3B, the cost rule: C3 MEASURED 1.02 s serial / 0.41-0.45 s on 4-8 threads per batch of 16 on the
training step, above the 0.2 s budget): in the DataLoader WORKERS, per item -- the item already carries its cuboids (``agent_box``,
``agent_yaw``, ``agent_cz``, ``agent_h``, ``agent_zh_mask``, ``agent_label``, ``agent_ep``: ``V3Dataset._agent_item`` under
``--agent-join --join3d``) and the clip camera is the bank the trainer already built (``model._rig_camera``, keyed by ``agent_ep``).
No sidecar file, no new flag, no G-DVB entry: the targets are a pure function of fields the item and the model already own, so they are
BYTE-identical to the on-step build (tests/test_box_depth_aux.py: the KA). The step only moves the collated tensors to the device.
A batch WITHOUT the ``bd_*`` keys while the head is built REFUSES (never builds on the step, never skips the term).
"""
from __future__ import annotations

import time

import torch

FLAG = "--r9-box-depth-aux"
DEST = "r9_box_depth_aux"
W_FLAG = "--w-box-depth"
W_DEST = "w_box_depth"
#: the head hangs on the PERCEPTION BRANCH, so its state_dict keys sit under the rig's allowed root ``_perception`` (the rig's
#: RIG_BUILT_ROOTS is pinned by a literal test); ``refcv9_perc_rig.box_depth_exemptions`` NAMES the prefix for the warm start.
HEAD_ATTR = "box_depth_aux"
KEY_PREFIX = "_perception." + HEAD_ATTR + "."
WEIGHT_ATTR = "_w_box_depth"
#: the per-item targets a DataLoader worker adds to the item (``item_targets``) and the step reads (``targets_from_batch``).
STRIDES = (8, 16)
BATCH_KEYS = tuple(f"bd_{k}{s}" for s in STRIDES for k in ("t", "fg", "w"))          # target [D,Hs,Ws] f32, fg [Hs,Ws] bool, weight [Hs,Ws] f32
CENSUS_KEYS = ("bd_n_boxes", "bd_n_boxes_no_zh", "bd_build_s")                        # per-item scalars (collated to [B])


def is_on(args) -> bool:
    return bool(getattr(args, DEST, False))


def _head(model):
    br = getattr(model, "_perception", None)
    return None if br is None else getattr(br, HEAD_ATTR, None)


def pin_args(args) -> None:
    """The flag's refusals, at argv time. OFF: nothing is checked (an orphan ``--w-box-depth`` alone is refused too). Admits the perception
    rig (``--r9-perc-rig``) AND the full model (AUX-FULL: ``--w-box3d > 0``, its perception branch) when every input the head reads is present."""
    w = float(getattr(args, W_DEST, 0.0) or 0.0)
    if not is_on(args):
        if w != 0.0:
            raise SystemExit(f"[box-depth-aux] {W_FLAG} {w:g} without {FLAG}: the weight would multiply nothing (a dead flag)")
        return
    bad = []
    if not w > 0.0:
        bad.append(f"{W_FLAG} is {w:g}: the depth term (weight = {W_FLAG}) would not exist; name a positive weight in the argv")
    if not bool(getattr(args, "r9_perc_rig", False)) and not float(getattr(args, "w_box3d", 0.0) or 0.0) > 0.0:
        # AUX-FULL: the full model's loss (compute_losses_v3) calls the head too; the perception branch it hangs on is built there by --w-box3d > 0
        # (the rig builds it by its own pin). Without either, `attach` has nowhere to hang the head.
        bad.append("--r9-perc-rig is off and --w-box3d is not > 0: the head hangs on the perception branch, which the full model builds only "
                   "with --w-box3d > 0 (the perception rig builds it itself); no branch, nowhere to attach the head")
    if str(getattr(args, "map_hires", "off") or "off") != "on":
        bad.append("--map-hires is not `on`: the stride-8 map the head reads exists only with the s8 tap (--map-hires on)")
    if not str(getattr(args, "join3d", "") or ""):
        bad.append("--join3d is not set: the cuboids' centre z / height (agent_cz, agent_h) are what the ray-cast needs")
    if not str(getattr(args, "agent_join", "") or ""):
        bad.append("--agent-join is not set: no `obstacle.offline` cuboids in the batch, nothing to ray-cast")
    if str(getattr(args, "agent_rig_camera", "off") or "off") != "extrinsics":
        bad.append("--agent-rig-camera is not `extrinsics`: the ray-cast needs the clip's own camera (a constant mount pose "
                   "mis-places every pixel's ray)")
    from tanitad.train import map_only_ft as _mof
    if _mof.is_on(args):
        bad.append("a --r8-train-only-map / --r9-train-only freeze is declared: the NEW head would stay at its random init")
    if bad:
        raise SystemExit(f"[box-depth-aux] {FLAG} refused:\n  - " + "\n  - ".join(bad))


def attach(model, args) -> None:
    """Build-time wiring, called by the trainer (and the G-EVAL loader) once the perception branch and the s8 tap exist. OFF: touches
    nothing. The head's init draws run inside ``fork_rng`` so NO other parameter's draw moves (OFF/ON share every shared init)."""
    if not is_on(args):
        return
    import torch
    from tanitad.models.box_depth_aux import BoxDepthAuxHead
    br = getattr(model, "_perception", None)
    if br is None:
        raise SystemExit("[box-depth-aux] no perception branch is built (--w-box3d > 0 builds it): the head has nowhere to hang")
    enc = model.core.encoder
    if not getattr(enc, "s8_tap", False) or getattr(enc, "s8_dim", None) is None or getattr(enc, "s16_dim", None) is None:
        raise SystemExit("[box-depth-aux] the trunk has no stride-8 tap / stride-16 width (--trunk timm --map-hires on)")
    dev = next(br.parameters()).device
    with torch.random.fork_rng(devices=[]):
        head = BoxDepthAuxHead(int(enc.s8_dim), int(enc.s16_dim))
    setattr(br, HEAD_ATTR, head.to(dev))
    setattr(model, WEIGHT_ATTR, float(getattr(args, W_DEST)))


def row_cameras(cam, ep_ids) -> list:
    """The camera object (``RigCameraBank`` | ``RigCamera``) + the rows' episode ids -> one builder ``Camera`` per row. A row with no
    camera REFUSES (RuntimeError: this runs inside a DataLoader worker, where a SystemExit would only kill the worker)."""
    from tanitad.models.box_depth_aux import camera_of
    from tanitad.refs.refc_agents import RigCameraBank
    from tanitad.data.rig_projection import RigCamera
    if isinstance(cam, RigCameraBank):
        rcs = cam.for_episodes(ep_ids)
    elif isinstance(cam, RigCamera):
        n = int(torch.as_tensor(ep_ids).numel())
        rcs = [cam] * n
    else:
        raise RuntimeError("[box-depth-aux] no rig camera (model._rig_camera is None / not a bank): the ray-cast has no ray "
                           "(--agent-rig-camera extrinsics)")
    if any(c is None for c in rcs):
        raise RuntimeError("[box-depth-aux] a row's clip is not in the camera bank (episode ids %s): the ray-cast has no ray -- refuse, "
                           "do not skip" % torch.as_tensor(ep_ids).reshape(-1).tolist())
    return [camera_of(c) for c in rcs]


class TargetSource:
    """What a DataLoader worker needs to build ONE item's targets: the rig-camera object and the trunk's zeroed-row count. Read-only,
    picklable (spawn) and fork-safe. Built once by ``attach_dataset``; stored on the dataset as ``ds._r9_bd``."""

    def __init__(self, cam, equalize_bottom_rows: int):
        self.cam = cam
        self.eq = int(equalize_bottom_rows)


def item_targets(item: dict, src: TargetSource) -> dict:
    """ONE dataset item (UNbatched: ``agent_box`` [M, 4] ...) -> the ``bd_*`` entries the worker adds to it. The very functions the batch
    path used (``boxes_from_batch`` on a batch of one, ``build_targets`` with one row), so row i of a collated batch is the on-step
    batch's row i, byte for byte (tests/test_box_depth_aux.py::test_ka_*). Runs in the DataLoader worker: single-threaded."""
    from tanitad.models import box_depth_aux as BDA
    t0 = time.perf_counter()
    miss = [k for k in BDA.KEYS_NEEDED if k not in item]
    if miss:
        raise RuntimeError("[box-depth-aux] the item carries no %s: the targets read the box3d cuboids of the item "
                           "(--agent-join + --join3d); refuse, do not skip" % miss)
    one = {k: item[k].unsqueeze(0) for k in BDA.KEYS_NEEDED + ("agent_label",) if k in item}
    boxes, st = BDA.boxes_from_batch(one)
    cams = row_cameras(src.cam, item["agent_ep"].reshape(1))
    built = BDA.build_targets(boxes, cams, equalize_bottom_rows=src.eq, threads=1)
    out = {}
    for s in STRIDES:
        t, f, w = built[s]
        out[f"bd_t{s}"], out[f"bd_fg{s}"], out[f"bd_w{s}"] = torch.from_numpy(t[0]), torch.from_numpy(f[0]), torch.from_numpy(w[0])
    out["bd_n_boxes"] = torch.tensor(int(st["n_boxes"]), dtype=torch.long)
    out["bd_n_boxes_no_zh"] = torch.tensor(int(st["n_boxes_no_zh"]), dtype=torch.long)
    out["bd_build_s"] = torch.tensor(time.perf_counter() - t0, dtype=torch.float64)
    return out


def attach_dataset(ds, model, args) -> dict:
    """Train-time wiring (``train()``, BEFORE the DataLoader is built): hand the dataset the camera bank so ``V3Dataset.__getitem__`` adds the
    ``bd_*`` targets in the worker. OFF: touches nothing. Refuses at STARTUP a dataset without the cuboid join, a model without a camera, and
    a bank that does not cover every episode of the run (a mid-run worker failure is the late way to learn that)."""
    if not is_on(args):
        return {}
    from tanitad.refs.refc_agents import RigCameraBank
    from tanitad.data.rig_projection import RigCamera
    if getattr(ds, "agent_join", None) is None or getattr(ds, "join3d", None) is None:
        raise SystemExit("[box-depth-aux] the dataset carries no agent_join / join3d: its items would have no cuboids to ray-cast")
    cam = getattr(model, "_rig_camera", None)
    if not isinstance(cam, (RigCameraBank, RigCamera)):
        raise SystemExit("[box-depth-aux] model._rig_camera is None: no camera to ray-cast from (--agent-rig-camera extrinsics)")
    ids = [int(e.episode_id) for e in (getattr(ds, "episodes", None) or [])]
    if isinstance(cam, RigCameraBank):
        miss = sorted({i for i in ids if cam.get(i) is None})
        if miss:
            raise SystemExit("[box-depth-aux] the camera bank covers no camera for %d of this run's episodes (e.g. %s): their targets "
                             "could not be built -- refuse, do not skip" % (len(miss), miss[:4]))
    eq = int(getattr(getattr(model.core.encoder, "cfg", None), "equalize_bottom_rows", 0) or 0)
    ds._r9_bd = TargetSource(cam, eq)
    nw = int(getattr(args, "workers", 0) or 0)
    if nw < 1:                                                      # not a refusal (a dev-box smoke at --workers 0 is legitimate): a counted, printed cost
        print("[box-depth-aux] WARNING --workers %d: the targets are built in the MAIN process (MEASURED 0.64-1.02 s serial per batch of 16, "
              "hold_r9_c3 evidence/raw: above the 0.2 s/step budget); a real arm names --workers >= 1" % nw, flush=True)
    return {"where": "DataLoader worker (V3Dataset.__getitem__ -> item_targets)", "equalize_bottom_rows": eq, "n_episodes_checked": len(ids),
            "workers": nw}


def targets_from_batch(batch: dict, device):
    """The collated ``bd_*`` entries -> ({stride: (target, fg, weight) tensors on ``device``}, census). The ONLY main-thread work the
    targets cost the step. A batch without them REFUSES: the dataset hook did not run, and building them here would put the cost back."""
    miss = [k for k in BATCH_KEYS + CENSUS_KEYS if k not in batch]
    if miss:
        raise SystemExit("[box-depth-aux] the batch carries no precomputed %s: the head is built but the dataset hook is not (attach_dataset / "
                         "V3Dataset.__getitem__ -> item_targets). Refuse: the term is neither skipped nor rebuilt on the step" % miss)
    tg = {s: (batch[f"bd_t{s}"].to(device), batch[f"bd_fg{s}"].to(device), batch[f"bd_w{s}"].to(device)) for s in STRIDES}
    cen = {"n_boxes": float(batch["bd_n_boxes"].sum()), "n_boxes_no_zh": float(batch["bd_n_boxes_no_zh"].sum()),
           "build_s_worker_mean": float(batch["bd_build_s"].double().mean())}
    return tg, cen


def loss_term(model, out: dict, batch: dict, device, extra: dict):
    """The weighted depth term, or ``None`` (head not built, or the model is in EVAL mode: training-only). Writes ``extra["box_depth"]``
    (the unweighted loss) and its census. Called by ``compute_losses_perc_rig`` and (AUX-FULL) ``compute_losses_v3``; the head is called HERE and nowhere else."""
    head = _head(model)
    if head is None or not model.training:
        return None
    from tanitad.models import box_depth_aux as BDA
    s8, s16 = out.get("fmap_s8"), out.get("fmap_s16")
    if s8 is None or s16 is None:
        raise SystemExit("[box-depth-aux] the forward emitted no fmap_s8 / fmap_s16 (--map-hires on builds the s8 tap): the head has no input")
    t0 = time.perf_counter()
    tg, cen = targets_from_batch(batch, s8.device)                  # built in the DataLoader workers (item_targets); the step only moves them
    dt = time.perf_counter() - t0
    for s, ref in ((8, s8), (16, s16)):                            # the head's grid must be the builder's grid (a silent resize is a different target)
        if tuple(tg[s][1].shape[-2:]) != tuple(ref.shape[-2:]) or int(tg[s][1].shape[0]) != int(ref.shape[0]):
            raise SystemExit("[box-depth-aux] stride-%d map is %s but the target grid is %s" % (s, tuple(ref.shape[:1] + ref.shape[-2:]),
                                                                                            tuple(tg[s][1].shape[:1] + tg[s][1].shape[-2:])))
    loss, met = BDA.depth_loss(head(s8, s16), tg)
    extra["box_depth"] = loss
    extra["box_depth_main_s"] = float(dt)                          # main-thread seconds the targets cost this step (the move-to-device)
    extra["box_depth_build_s"] = cen["build_s_worker_mean"]        # mean per-item build seconds IN THE WORKER (off the step)
    extra["box_depth_n_boxes"] = cen["n_boxes"]
    extra["box_depth_n_boxes_no_zh"] = cen["n_boxes_no_zh"]
    for k, v in met.items():
        extra["box_depth_" + k] = float(v)
    return float(getattr(model, WEIGHT_ATTR)) * loss


def stamp(model, args) -> dict | None:
    """``config.json["r9_box_depth_aux"]`` -- read OFF the built head; ``None`` (key absent) when the flag is OFF."""
    head = _head(model)
    if head is None:
        return None
    from tanitad.models import box_depth_aux as BDA
    return {"flag": FLAG, "weight": float(getattr(model, WEIGHT_ATTR)), "weight_flag": W_FLAG,
            "n_bins": head.n_bins, "d_range_m": [BDA.BD.D_MIN_M, BDA.BD.D_MAX_M], "lid": "CaDDN eq. 2, bin_size = 2 (dmax-dmin)/(D(D+1))",
            "depth_kind": BDA.DEPTH_KIND, "soft_target": "two-hot in the bin-centre coordinate", "strides": list(BDA.STRIDES),
            "alpha_fg": BDA.ALPHA_FG, "alpha_bg": BDA.ALPHA_BG,
            "alpha_bg_note": "INERT: box-only depth gives a background cell no label (all-zero target => CE exactly 0)",
            "loss": "per stride alpha_fg * mean over FG cells of soft-CE (FG = >= 50 % foreground pixels); strides averaged; object cells only",
            "head": {"c8": head.c8, "c16": head.c16, "mid": head.mid, "n_params": sum(p.numel() for p in head.parameters()),
                     "key_prefix": KEY_PREFIX},
            "observed_mask": "bottom equalize_bottom_rows rows are the trunk's zeroed rows -> builder observed=False (cells touching them IGNORE)",
            "targets_from": "the batch's own box3d cuboids (agent_box, agent_yaw, agent_cz, agent_h, agent_valid & agent_zh_mask), clip camera "
                            "from model._rig_camera", "builder": "tanitad/data/box_depth_targets.py (byte copy of e0 e0g_depth_targets.py, "
                            "blob 6fff1d3a15d9e66fa8ef82256654a30ca26f21d6)",
            "targets_built": "in the DataLoader workers, per item, single-threaded (item_targets; byte-identical to the batch build)", "training_only": "head called only by the loss functions (compute_losses_perc_rig, and since AUX-FULL compute_losses_v3); never by RefCV3Model.forward"}


# ---- G-DVB -------------------------------------------------------------------------------------------------------------------- #
def dvb_check(model, args) -> list:
    """``built`` lever: declared <=> the head is built on ``model._perception``, with the trunk's widths and the builder's bins."""
    from tanitad.train.declared_vs_built import Mismatch
    declared = is_on(args)
    head = _head(model)
    built = head is not None
    out = []
    if declared != built:
        out.append(Mismatch(DEST, declared, built, "model._perception." + HEAD_ATTR, "argv and the build disagree"))
    if built:
        from tanitad.data import box_depth_targets as BD
        enc = model.core.encoder
        for what, want, got, where in (("s8 input width", int(getattr(enc, "s8_dim", -1)), int(head.s8[0].in_channels), KEY_PREFIX + "s8.0.in_channels"),
                                       ("s16 input width", int(getattr(enc, "s16_dim", -1)), int(head.s16[0].in_channels), KEY_PREFIX + "s16.0.in_channels"),
                                       ("LID bins", BD.N_BINS, head.n_bins, KEY_PREFIX + "n_bins"),
                                       ("s8 logits", BD.N_BINS, head.s8[4].out_channels, KEY_PREFIX + "s8.4"),
                                       ("s16 logits", BD.N_BINS, head.s16[4].out_channels, KEY_PREFIX + "s16.4")):
            if want != got:
                out.append(Mismatch(DEST, f"{what} {want}", got, "model._perception." + HEAD_ATTR, "the head is not the one the trunk / builder need"))
    return out


def dvb_check_weight(model, args) -> list:
    """``loss`` lever: the argv's weight equals the attribute the loss multiplies by (and a weight needs the head)."""
    from tanitad.train.declared_vs_built import Mismatch
    declared = float(getattr(args, W_DEST, 0.0) or 0.0)
    built = float(getattr(model, WEIGHT_ATTR, 0.0) or 0.0)
    out = []
    if abs(declared - built) > 1e-12:
        out.append(Mismatch(W_DEST, declared, built, "model." + WEIGHT_ATTR, "argv and the weight the LOSS multiplies by disagree"))
    if declared > 0.0 and _head(model) is None:
        out.append(Mismatch(W_DEST, declared, "no head", "model._perception." + HEAD_ATTR, "a depth weight with no head multiplies nothing"))
    return out


def register_dvb_levers(register) -> None:
    """Called by the trainer at import (the registry is pinned two-way against its parser)."""
    register(DEST, "built", dvb_check)
    register(W_DEST, "loss", dvb_check_weight)
