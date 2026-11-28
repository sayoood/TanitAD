"""``--r9-box-persp-aux`` / ``--w-box-persp`` (arm B2) and its control ``--r9-box-bev2-aux`` / ``--w-box-bev2`` (arm B2c) -- refcv9 C4
(REFCV9_DESIGN 2.4 row "perspective 3D aux head (B2)", 4.3 rows B2 / B2c).  The heads, targets and losses live in
``tanitad/models/persp3d_aux.py``.  This module is the flags' contract with the trainer, in the shape ``box_depth_aux_flag`` (C3) set:
``pin_args`` (argv refusals), ``attach`` (build-time), ``attach_dataset`` + ``item_targets`` (the DataLoader-worker target build),
``loss_term``, ``stamp``, the G-DVB checks and ``register_dvb_levers``.

WHICH WEIGHT FLAG: each arm has its OWN (``--w-box-persp`` for B2, ``--w-box-bev2`` for B2c), so every flag has exactly one lever and one
attribute the loss multiplies by; the two arms are run at the SAME numeric weight, and the two flags are MUTUALLY EXCLUSIVE (both -> refused).

REACHABILITY (the CLAUDE.md class "built, tested and unreachable from its caller"), one hop per line, B2:
  argv ``--r9-box-persp-aux`` / ``--w-box-persp`` (refc_v3_train.py build_parser, defaults False / 0.0 -- BOTH named in the arm's argv)
  -> ``args.r9_box_persp_aux`` / ``args.w_box_persp`` -> ``pin_args`` -> ``attach`` (train(), after the perception branch:
  ``model._perception.box_persp_aux`` + ``model._w_box_persp``) -> ``attach_dataset`` (train(), before the DataLoader: ``ds._r9_p3``)
  -> V3Dataset.__getitem__ -> ``item_targets`` (IN THE DATALOADER WORKER) -> default collate -> ``compute_losses_perc_rig`` (the rig) OR
  ``compute_losses_v3`` (the full model, AUX-FULL: the same call after the 10 cm map term) -> ``loss_term``
  (reads ``out["fmap_s8"]``, ``out["fmap_s16"]``, the batch's ``p3_*`` targets and ``agent_*`` fields) -> ``loss = loss + term`` -> ``.backward()``.
B2c is the same chain with ``box_bev2_aux`` / ``_w_box_bev2`` / ``b2_*`` and ``out["perception"]["bev_feats"]`` as the head's input.
Neither head is EVER called by ``RefCV3Model.forward``: the eval forward is the OFF forward, bit for bit (tests/test_persp3d_aux.py).
"""
from __future__ import annotations

import time

import numpy as np
import torch

FLAG_P, DEST_P, W_FLAG_P, W_DEST_P = "--r9-box-persp-aux", "r9_box_persp_aux", "--w-box-persp", "w_box_persp"
FLAG_B, DEST_B, W_FLAG_B, W_DEST_B = "--r9-box-bev2-aux", "r9_box_bev2_aux", "--w-box-bev2", "w_box_bev2"
HEAD_P, HEAD_B = "box_persp_aux", "box_bev2_aux"
#: both heads hang on the PERCEPTION BRANCH (state_dict keys under the rig's allowed root ``_perception``); the warm start NAMES the prefixes.
KEY_PREFIX_P, KEY_PREFIX_B = "_perception." + HEAD_P + ".", "_perception." + HEAD_B + "."
WEIGHT_ATTR_P, WEIGHT_ATTR_B = "_w_box_persp", "_w_box_bev2"
LOSS_KEY_P, LOSS_KEY_B = "box_persp", "box_bev2"          # the G-LIVE keys (extra[...] of compute_losses_perc_rig)
STRIDES = (8, 16)
BATCH_KEYS_P = tuple(f"p3_{k}{s}" for s in STRIDES for k in ("heat", "own", "ctr", "ign")) + ("p3_i0", "p3_w1")
CENSUS_KEYS_P = ("p3_n_boxes", "p3_n_visible", "p3_n_heat_pos", "p3_build_s")
BATCH_KEYS_B = ("b2_heat", "b2_own", "b2_ctr", "b2_ign")
CENSUS_KEYS_B = ("b2_n_boxes", "b2_n_heat_pos", "b2_build_s")


def is_persp(args) -> bool:
    return bool(getattr(args, DEST_P, False))


def is_bev2(args) -> bool:
    return bool(getattr(args, DEST_B, False))


def is_any(args) -> bool:
    return is_persp(args) or is_bev2(args)


def _head_p(model):
    br = getattr(model, "_perception", None)
    return None if br is None else getattr(br, HEAD_P, None)


def _head_b(model):
    br = getattr(model, "_perception", None)
    return None if br is None else getattr(br, HEAD_B, None)


def pin_args(args) -> None:
    """The flags' refusals, at argv time.  OFF (both): nothing is checked except an orphan weight, which is refused.  Admits the perception
    rig AND the full model (AUX-FULL) when every input the head reads is present; both heads at once is refused in either."""
    wp = float(getattr(args, W_DEST_P, 0.0) or 0.0)
    wb = float(getattr(args, W_DEST_B, 0.0) or 0.0)
    if is_persp(args) and is_bev2(args):
        raise SystemExit(f"[persp3d-aux] {FLAG_P} and {FLAG_B} are MUTUALLY EXCLUSIVE: B2c is B2's control (a different head on the same "
                         "targets), not an add-on; running both makes the two arms one non-attributable arm")
    for on, w, flag, wflag in ((is_persp(args), wp, FLAG_P, W_FLAG_P), (is_bev2(args), wb, FLAG_B, W_FLAG_B)):
        if not on and w != 0.0:
            raise SystemExit(f"[persp3d-aux] {wflag} {w:g} without {flag}: the weight would multiply nothing (a dead flag)")
    if not is_any(args):
        return
    flag, wflag, w = (FLAG_P, W_FLAG_P, wp) if is_persp(args) else (FLAG_B, W_FLAG_B, wb)
    bad = []
    if not w > 0.0:
        bad.append(f"{wflag} is {w:g}: the term (weight = {wflag}) would not exist; name a positive weight in the argv")
    # AUX-FULL: no rig bullet. The head's loss is called by the perception rig's loss AND the full model's (compute_losses_v3); the perception
    # branch it hangs on exists in BOTH exactly when --w-box3d > 0 (the box3d branch this auxiliary exists to help).
    if not float(getattr(args, "w_box3d", 0.0) or 0.0) > 0.0:
        bad.append("--w-box3d is not > 0: the box3d branch (the thing this auxiliary exists to help) is not trained, and the perception "
                   "branch the head hangs on is not built")
    if str(getattr(args, "map_hires", "off") or "off") != "on":
        bad.append("--map-hires is not `on`: the stride-8 map and the pooled BEV the heads read exist only with the 10 cm branch")
    if not str(getattr(args, "join3d", "") or ""):
        bad.append("--join3d is not set: the cuboids' centre z / height (agent_cz, agent_h) are what the targets need")
    if not str(getattr(args, "agent_join", "") or ""):
        bad.append("--agent-join is not set: no `obstacle.offline` cuboids in the batch, nothing to project")
    if is_persp(args) and str(getattr(args, "agent_rig_camera", "off") or "off") != "extrinsics":
        bad.append("--agent-rig-camera is not `extrinsics`: the perspective targets need the clip's own camera (a constant mount pose "
                   "mis-places every projected centre)")
    from tanitad.train import map_only_ft as _mof
    if _mof.is_on(args):
        bad.append("a --r8-train-only-map / --r9-train-only freeze is declared: the NEW head would stay at its random init")
    if bad:
        raise SystemExit(f"[persp3d-aux] {flag} refused:\n  - " + "\n  - ".join(bad))


# ---- build -------------------------------------------------------------------------------------------------------------------- #
def _reference_persp_params(enc) -> int:
    from tanitad.models import persp3d_aux as P
    with torch.random.fork_rng(devices=[]):
        return P.n_params(P.Persp3DAuxHead(int(enc.s8_dim), int(enc.s16_dim)))


def attach(model, args) -> None:
    """Build-time wiring (trainer + G-EVAL loader).  OFF: touches nothing.  The init draws run inside ``fork_rng`` so NO other parameter's
    draw moves (OFF / ON share every shared init)."""
    if not is_any(args):
        return
    from tanitad.models import persp3d_aux as P
    br = getattr(model, "_perception", None)
    if br is None:
        raise SystemExit("[persp3d-aux] no perception branch is built (--w-box3d > 0 builds it): the head has nowhere to hang")
    dev = next(br.parameters()).device
    enc = model.core.encoder
    if not getattr(enc, "s8_tap", False) or getattr(enc, "s8_dim", None) is None or getattr(enc, "s16_dim", None) is None:
        raise SystemExit("[persp3d-aux] the trunk has no stride-8 tap / stride-16 width (--trunk timm --map-hires on)")
    if is_persp(args):
        with torch.random.fork_rng(devices=[]):
            head = P.Persp3DAuxHead(int(enc.s8_dim), int(enc.s16_dim))
        setattr(br, HEAD_P, head.to(dev))
        setattr(model, WEIGHT_ATTR_P, float(getattr(args, W_DEST_P)))
        return
    from tanitad.data.bev_raster import GRID_DEFAULT
    cfg = getattr(br, "cfg", None)
    d_bev = int(getattr(getattr(cfg, "bev_cfg", None), "d_out", 0) or 0)
    grid = getattr(cfg, "planner_grid", None)
    if d_bev <= 0 or grid != GRID_DEFAULT:
        raise SystemExit(f"[persp3d-aux] the shared BEV is not the 120 x 64 planner grid with a known width (d_bev {d_bev}, grid {grid}): "
                         "the B2c targets are built on GRID_DEFAULT")
    n_ref = _reference_persp_params(enc)
    mid = P.matched_mid(d_bev, n_ref)
    with torch.random.fork_rng(devices=[]):
        head = P.Bev2AuxHead(d_bev, mid=mid)
    if not P.count_matched(P.n_params(head), n_ref):
        raise SystemExit(f"[persp3d-aux] B2c cannot be matched to B2's {n_ref} parameters within {P.MATCH_TOL:.0%} (best width {mid}: "
                         f"{P.n_params(head)})")
    head.ref_params = int(n_ref)                                   # a plain int attribute: no state_dict key, read by the G-DVB check + stamp
    setattr(br, HEAD_B, head.to(dev))
    setattr(model, WEIGHT_ATTR_B, float(getattr(args, W_DEST_B)))


class TargetSource:
    """What a DataLoader worker needs to build ONE item's targets (read-only, picklable, fork-safe).  ``kind`` = "persp" (camera object + the
    trunk's zeroed-row count) or "bev2" (the BEV grid)."""

    def __init__(self, kind: str, *, cam=None, eq: int = 0, grid=None):
        self.kind, self.cam, self.eq, self.grid = str(kind), cam, int(eq), grid


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def item_targets(item: dict, src: TargetSource) -> dict:
    """ONE dataset item (UNbatched) -> the target entries a DataLoader worker adds to it.  Runs in the worker: single-threaded, torch ops
    only on tiny tensors.  A NO_LABEL row (``agent_label`` False) has NO boxes and must not teach 'empty': its heat IGNORE map is all True."""
    from tanitad.models import persp3d_aux as P
    t0 = time.perf_counter()
    miss = [k for k in P.SLOT_KEYS if k not in item]
    if miss:
        raise RuntimeError("[persp3d-aux] the item carries no %s: the targets read the box3d cuboids of the item "
                           "(--agent-join + --join3d); refuse, do not skip" % miss)
    like = {k: _np(item[k]) for k in P.SLOT_KEYS + ("agent_label",) if k in item}
    slots, boxes, _nozh = P.selected_slots(like)
    A = int(like["agent_valid"].reshape(-1).shape[0])
    labelled = True if "agent_label" not in like else bool(like["agent_label"].reshape(-1)[0])
    out: dict = {}
    if src.kind == "persp":
        from tanitad.train.box_depth_aux_flag import row_cameras
        cams = row_cameras(src.cam, item["agent_ep"].reshape(1))
        r = P.persp_targets(boxes, slots, A, cams[0], equalize_bottom_rows=src.eq)
        for s in STRIDES:
            ign = r[f"ign{s}"] | (not labelled)
            out[f"p3_heat{s}"], out[f"p3_own{s}"] = torch.from_numpy(r[f"heat{s}"]), torch.from_numpy(r[f"own{s}"])
            out[f"p3_ctr{s}"], out[f"p3_ign{s}"] = torch.from_numpy(r[f"ctr{s}"]), torch.from_numpy(np.asarray(ign))
        out["p3_i0"], out["p3_w1"] = torch.from_numpy(r["i0"]), torch.from_numpy(r["w1"])
        c = r["census"]
        out["p3_n_boxes"], out["p3_n_visible"], out["p3_n_heat_pos"] = (torch.tensor(int(c[k]), dtype=torch.long)
                                                                          for k in ("n_boxes", "n_visible", "n_heat_pos"))
        out["p3_build_s"] = torch.tensor(time.perf_counter() - t0, dtype=torch.float64)
        return out
    r = P.bev_targets(boxes, slots, src.grid)
    out["b2_heat"], out["b2_own"], out["b2_ctr"] = torch.from_numpy(r["heat"]), torch.from_numpy(r["own"]), torch.from_numpy(r["ctr"])
    out["b2_ign"] = torch.full(r["heat"].shape, not labelled, dtype=torch.bool)
    out["b2_n_boxes"] = torch.tensor(int(r["census"]["n_boxes"]), dtype=torch.long)
    out["b2_n_heat_pos"] = torch.tensor(int(r["census"]["n_heat_pos"]), dtype=torch.long)
    out["b2_build_s"] = torch.tensor(time.perf_counter() - t0, dtype=torch.float64)
    return out


def attach_dataset(ds, model, args) -> dict:
    """Train-time wiring (``train()``, BEFORE the DataLoader is built).  OFF: touches nothing.  Refuses at STARTUP a dataset without the
    cuboid join, a model without a camera (B2), and a bank that does not cover every episode of the run."""
    if not is_any(args):
        return {}
    if getattr(ds, "agent_join", None) is None or getattr(ds, "join3d", None) is None:
        raise SystemExit("[persp3d-aux] the dataset carries no agent_join / join3d: its items would have no cuboids to project")
    nw = int(getattr(args, "workers", 0) or 0)
    if is_bev2(args):
        from tanitad.data.bev_raster import GRID_DEFAULT
        ds._r9_p3 = TargetSource("bev2", grid=GRID_DEFAULT)
        rep = {"where": "DataLoader worker (V3Dataset.__getitem__ -> item_targets)", "kind": "bev2", "workers": nw}
    else:
        from tanitad.data.rig_projection import RigCamera
        from tanitad.refs.refc_agents import RigCameraBank
        cam = getattr(model, "_rig_camera", None)
        if not isinstance(cam, (RigCameraBank, RigCamera)):
            raise SystemExit("[persp3d-aux] model._rig_camera is None: no camera to project through (--agent-rig-camera extrinsics)")
        ids = [int(e.episode_id) for e in (getattr(ds, "episodes", None) or [])]
        if isinstance(cam, RigCameraBank):
            miss = sorted({i for i in ids if cam.get(i) is None})
            if miss:
                raise SystemExit("[persp3d-aux] the camera bank covers no camera for %d of this run's episodes (e.g. %s): their targets "
                                 "could not be built -- refuse, do not skip" % (len(miss), miss[:4]))
        eq = int(getattr(getattr(model.core.encoder, "cfg", None), "equalize_bottom_rows", 0) or 0)
        ds._r9_p3 = TargetSource("persp", cam=cam, eq=eq)
        rep = {"where": "DataLoader worker (V3Dataset.__getitem__ -> item_targets)", "kind": "persp", "equalize_bottom_rows": eq,
               "n_episodes_checked": len(ids), "workers": nw}
    if nw < 1:
        print("[persp3d-aux] WARNING --workers %d: the targets are built in the MAIN process (a ray-cast per item); a real arm names "
              "--workers >= 1" % nw, flush=True)
    return rep


def _from_batch(batch: dict, device, keys, census, what):
    miss = [k for k in keys + census if k not in batch]
    if miss:
        raise SystemExit(f"[persp3d-aux] the batch carries no precomputed {miss}: the {what} head is built but the dataset hook is not "
                         "(attach_dataset / V3Dataset.__getitem__ -> item_targets). Refuse: the term is neither skipped nor rebuilt on the step")
    return {k: batch[k].to(device) for k in keys}


def persp_from_batch(batch: dict, device):
    raw = _from_batch(batch, device, BATCH_KEYS_P, CENSUS_KEYS_P, "perspective")
    tg = {s: {"heat": raw[f"p3_heat{s}"], "own": raw[f"p3_own{s}"], "ctr": raw[f"p3_ctr{s}"], "ign": raw[f"p3_ign{s}"]} for s in STRIDES}
    tg["i0"], tg["w1"] = raw["p3_i0"], raw["p3_w1"]
    cen = {"n_boxes": float(batch["p3_n_boxes"].sum()), "n_visible": float(batch["p3_n_visible"].sum()),
           "n_heat_pos": float(batch["p3_n_heat_pos"].sum()), "build_s_worker_mean": float(batch["p3_build_s"].double().mean())}
    return tg, cen


def bev2_from_batch(batch: dict, device):
    raw = _from_batch(batch, device, BATCH_KEYS_B, CENSUS_KEYS_B, "BEV2")
    tg = {"heat": raw["b2_heat"], "own": raw["b2_own"], "ctr": raw["b2_ctr"], "ign": raw["b2_ign"]}
    cen = {"n_boxes": float(batch["b2_n_boxes"].sum()), "n_heat_pos": float(batch["b2_n_heat_pos"].sum()),
           "build_s_worker_mean": float(batch["b2_build_s"].double().mean())}
    return tg, cen


def loss_term(model, out: dict, batch: dict, device, extra: dict):
    """The weighted term, or ``None`` (neither head built, or EVAL mode: training-only).  B2 writes ``extra["box_persp"]`` (+ ``_heat`` /
    ``_depth`` / ``_size`` / ``_yaw`` / ``_ctr``), B2c ``extra["box_bev2"]`` (+ ``_heat`` / ``_pos`` / ``_size`` / ``_yaw`` / ``_ctr``).
    Called by ``compute_losses_perc_rig`` and (AUX-FULL) ``compute_losses_v3``; the heads are called HERE and nowhere else."""
    hp, hb = _head_p(model), _head_b(model)
    if (hp is None and hb is None) or not model.training:
        return None
    from tanitad.models import persp3d_aux as P
    if hp is not None and hb is not None:
        raise SystemExit("[persp3d-aux] both heads are built: B2 and B2c are mutually exclusive")
    slots = P.slot_tables(batch, device)
    if hp is not None:
        s8, s16 = out.get("fmap_s8"), out.get("fmap_s16")
        if s8 is None or s16 is None:
            raise SystemExit("[persp3d-aux] the forward emitted no fmap_s8 / fmap_s16 (--map-hires on builds the s8 tap): the head has no input")
        t0 = time.perf_counter()
        tg, cen = persp_from_batch(batch, s8.device)
        dt = time.perf_counter() - t0
        for s, ref in ((8, s8), (16, s16)):
            if tuple(tg[s]["heat"].shape[-2:]) != tuple(ref.shape[-2:]) or int(tg[s]["heat"].shape[0]) != int(ref.shape[0]):
                raise SystemExit("[persp3d-aux] stride-%d map is %s but the target grid is %s" % (s, tuple(ref.shape[:1] + ref.shape[-2:]),
                                                                                              tuple(tg[s]["heat"].shape[:1] + tg[s]["heat"].shape[-2:])))
        loss, parts, met = P.persp_loss(hp(s8, s16), tg, slots)
        extra[LOSS_KEY_P] = loss
        for k, v in parts.items():
            extra[f"{LOSS_KEY_P}_{k}"] = float(v.detach())
        extra[f"{LOSS_KEY_P}_main_s"] = float(dt)
        extra[f"{LOSS_KEY_P}_build_s"] = cen["build_s_worker_mean"]
        for k in ("n_boxes", "n_visible", "n_heat_pos"):
            extra[f"{LOSS_KEY_P}_{k}"] = cen[k]
        for k, v in met.items():
            extra[f"{LOSS_KEY_P}_{k}"] = float(v)
        return float(getattr(model, WEIGHT_ATTR_P)) * loss
    bev = (out.get("perception") or {}).get("bev_feats")
    if bev is None:
        raise SystemExit("[persp3d-aux] the forward emitted no perception.bev_feats (the pooled BEV): B2c has no input")
    tg, cen = bev2_from_batch(batch, bev.device)
    if tuple(tg["heat"].shape[-2:]) != tuple(bev.shape[-2:]) or int(tg["heat"].shape[0]) != int(bev.shape[0]):
        raise SystemExit("[persp3d-aux] the BEV is %s but the target grid is %s" % (tuple(bev.shape), tuple(tg["heat"].shape)))
    from tanitad.data.bev_raster import GRID_DEFAULT
    loss, parts, met = P.bev2_loss(hb(bev), tg, slots, GRID_DEFAULT)
    extra[LOSS_KEY_B] = loss
    for k, v in parts.items():
        extra[f"{LOSS_KEY_B}_{k}"] = float(v.detach())
    extra[f"{LOSS_KEY_B}_build_s"] = cen["build_s_worker_mean"]
    for k in ("n_boxes", "n_heat_pos"):
        extra[f"{LOSS_KEY_B}_{k}"] = cen[k]
    for k, v in met.items():
        extra[f"{LOSS_KEY_B}_{k}"] = float(v)
    return float(getattr(model, WEIGHT_ATTR_B)) * loss


def stamp(model, args) -> dict | None:
    """``config.json["r9_box_persp_aux"]`` / ``["r9_box_bev2_aux"]`` -- read OFF the built heads; ``None`` when the flag is OFF."""
    from tanitad.models import persp3d_aux as P
    hp, hb = _head_p(model), _head_b(model)
    if hp is not None:
        return {"flag": FLAG_P, "weight": float(getattr(model, WEIGHT_ATTR_P)), "weight_flag": W_FLAG_P, "n_bins": P.N_BINS,
                "n_params": P.n_params(hp), "head": {"c8": hp.c8, "c16": hp.c16, "mid": hp.mid, "key_prefix": KEY_PREFIX_P},
                "strides": list(STRIDES), "outputs": dict(P.PERSP_CH), "part_weights": dict(P.PART_W),
                "targets": "compact, built per item in the DataLoader worker: Gaussian centre heat (CenterNet focal), owner map of agent slots, "
                           "FCOS centre-ness, depth two-hot of the CENTRE range (C3's LID bins); size / yaw gathered on the device from agent_*",
                "visibility": f">= {P.VIS_MIN_PIX} nearest-surface pixels (E0g raycast); regression positives = FG cells (>= 50 %) + the centre cell",
                "sigma": f"max({P.SIGMA_MIN}, {P.SIGMA_ALPHA} * extent_cells / 6) per axis (HYPOTHESIS: CenterNet/TTFNet style)",
                "yaw": "rig-frame yaw as (sin, cos)", "training_only": "head called only by the loss functions (compute_losses_perc_rig, and since AUX-FULL compute_losses_v3); never by RefCV3Model.forward"}
    if hb is not None:
        return {"flag": FLAG_B, "weight": float(getattr(model, WEIGHT_ATTR_B)), "weight_flag": W_FLAG_B, "n_params": P.n_params(hb),
                "ref_params_b2": int(hb.ref_params), "match_tol": P.MATCH_TOL,
                "head": {"d_bev": hb.d_bev, "mid": hb.mid, "key_prefix": KEY_PREFIX_B}, "outputs": dict(P.BEV2_CH),
                "part_weights": dict(P.BEV2_PART_W), "grid": "GRID_DEFAULT 120 x 64, 0.5 m",
                "targets": "the same cuboids as B2 (agent_valid & agent_zh_mask, labelled rows) on the BEV grid; no occlusion filter",
                "training_only": "head called only by the loss functions (compute_losses_perc_rig, and since AUX-FULL compute_losses_v3); never by RefCV3Model.forward"}
    return None


# ---- G-DVB -------------------------------------------------------------------------------------------------------------------- #
def dvb_check_persp(model, args) -> list:
    """``built`` lever: declared <=> the head is built on ``model._perception``, with the trunk's widths."""
    from tanitad.train.declared_vs_built import Mismatch
    declared, head = is_persp(args), _head_p(model)
    out = []
    if declared != (head is not None):
        out.append(Mismatch(DEST_P, declared, head is not None, "model._perception." + HEAD_P, "argv and the build disagree"))
    if head is not None:
        enc = model.core.encoder
        for what, want, got in (("s8 input width", int(getattr(enc, "s8_dim", -1)), int(head.s8.tower[0].in_channels)),
                                ("s16 input width", int(getattr(enc, "s16_dim", -1)), int(head.s16.tower[0].in_channels)),
                                ("depth bins", 80, int(head.s8.depth.out_channels)), ("depth bins s16", 80, int(head.s16.depth.out_channels))):
            if want != got:
                out.append(Mismatch(DEST_P, f"{what} {want}", got, "model._perception." + HEAD_P, "the head is not the one the trunk / LID bins need"))
    return out


def dvb_check_bev2(model, args) -> list:
    """``built`` lever: declared <=> the head is built; it reads the shared BEV's width and its parameter count is within 10 % of B2's."""
    from tanitad.models import persp3d_aux as P
    from tanitad.train.declared_vs_built import Mismatch
    declared, head = is_bev2(args), _head_b(model)
    out = []
    if declared != (head is not None):
        out.append(Mismatch(DEST_B, declared, head is not None, "model._perception." + HEAD_B, "argv and the build disagree"))
    if head is not None:
        br = model._perception
        d_bev = int(br.cfg.bev_cfg.d_out)
        if head.d_bev != d_bev:
            out.append(Mismatch(DEST_B, f"BEV width {d_bev}", head.d_bev, "model._perception." + HEAD_B, "the head does not read the shared BEV's width"))
        n_ref = _reference_persp_params(model.core.encoder)
        if not P.count_matched(P.n_params(head), n_ref):
            out.append(Mismatch(DEST_B, f"{n_ref} params +-{P.MATCH_TOL:.0%} (B2's)", P.n_params(head), "model._perception." + HEAD_B,
                                "the control is not capacity-matched to B2"))
    return out


def _dvb_weight(model, args, dest, attr, head_fn, head_name):
    from tanitad.train.declared_vs_built import Mismatch
    declared = float(getattr(args, dest, 0.0) or 0.0)
    built = float(getattr(model, attr, 0.0) or 0.0)
    out = []
    if abs(declared - built) > 1e-12:
        out.append(Mismatch(dest, declared, built, "model." + attr, "argv and the weight the LOSS multiplies by disagree"))
    if declared > 0.0 and head_fn(model) is None:
        out.append(Mismatch(dest, declared, "no head", "model._perception." + head_name, "a weight with no head multiplies nothing"))
    return out


def dvb_check_weight_persp(model, args) -> list:
    return _dvb_weight(model, args, W_DEST_P, WEIGHT_ATTR_P, _head_p, HEAD_P)


def dvb_check_weight_bev2(model, args) -> list:
    return _dvb_weight(model, args, W_DEST_B, WEIGHT_ATTR_B, _head_b, HEAD_B)


def register_dvb_levers(register) -> None:
    """Called by the trainer at import (the registry is pinned two-way against its parser)."""
    register(DEST_P, "built", dvb_check_persp)
    register(W_DEST_P, "loss", dvb_check_weight_persp)
    register(DEST_B, "built", dvb_check_bev2)
    register(W_DEST_B, "loss", dvb_check_weight_bev2)
