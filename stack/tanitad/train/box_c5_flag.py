"""refcv9 C5 -- the box-head levers of arm B4: ``--r9-box-layers`` and ``--r9-box-s8-horizon`` (REFCV9_DESIGN 2.4 rows *depth* / *memory* / *size cap*, 4.2 item C5, 4.3 arm B4, D2).

B4 = "B2 winner + per-frame queries + 6 layers + s8 horizon tokens (needs D2)".  This module is the flags' contract with the trainer, in the
shape ``box_b3_flag.py`` set: ``pin_args`` (refusals at argv time), ``attach`` (build-time wiring on the BUILT perception branch), ``stamp``
(``config.json["r9_box_c5"]``, absent when every flag is OFF), the G-DVB ``built`` checks and ``register_dvb_levers``.

LEVERS IN THIS MODULE (bank order = the brief's priority order; each lever lands with its tests):
  --r9-box-layers N   (lever 2)  the box3d decoder's depth.  Default 0 = "not named" = today's depth (``TIP_DEPTH`` = 3), so every existing argv
                      builds and runs bit-identically.  B4 names 6.  The decoder is EXTENDED IN PLACE (``Box3DSlotDecoder.extend_depth``): the new
                      layers are deep copies of the template layer, which is exactly what ``nn.TransformerDecoder(layer, num_layers=N)`` builds, so
                      the result equals a from-scratch depth-N build and consumes no RNG (a test pins both).
  --r9-box-s8-horizon (lever 3)  B4's s8 horizon tokens: the 16 stride-8 rows around the rig's HORIZON row x the s8 width (128) = 2,048 tokens, appended
                      LAST to the box memory with their own projection (``Box3DMemory.s8_proj``) and embedding (``src_embed_s8``) and their own positional
                      table (``Box3DSlotDecoder.mem_pos_s8``); every existing parameter keeps its shape.  The band's first s8 row (``row0``) is DERIVED from
                      the rig geometry at attach time (``horizon_row0``: the median horizon row of the camera bank, ``row_px = (H-1)/2 + f_ref * v_h``,
                      ``v_h = -R[2,2] / R[2,1]``) and STAMPED in ``config.json["r9_box_c5"]["s8_horizon"]``, so the eval loader rebuilds the identical
                      model WITHOUT the camera bank.
  --r9-box-perframe-queries (lever 4)  B4's per-frame query seeding: 200 per-frame seeds (the perspective head's s8 heat/depth, 3x3-NMS top-200, DETACHED, lifted through the
                      clip's own rig camera by E[LID range] along the cell's ray) + 100 learned GLOBAL anchors (normalised [0,1], a THIRD optimizer group
                      ``box_anchor`` at LR x10) = 300 slots, decoded by ``slot_query_select.anchored_forward`` (the HQS path).  REFUSED without ``--r9-box-persp-aux``.
                      The camera is STAGED before the forward (``stage``: train, eval/dump AND the harness replay).
  (lever 1, the D2 size cap, lives in ``box3d_head.BOX3D_PARAM_CAP`` -- it is a construction-time refusal, not a flag.)

REACHABILITY (CLAUDE.md "built, tested and unreachable from its caller"), one hop per line:
  argv ``--r9-box-layers`` (refc_v3_train.py build_parser, g8 group, default 0) -> ``args.r9_box_layers`` -> ``pin_args`` (``_pin_refcv8``) ->
  ``attach`` (train(), right after ``build_perception_branch``; and the eval loader) -> ``model._perception.box_dec.blocks.layers`` ->
  ``Box3DSlotDecoder.forward`` / ``box_b3.run_decoder`` / ``slot_query_select.anchored_forward`` (each loops over ``dec.blocks.layers``) ->
  ``dec.head`` -> the box3d loss (every layer's ``aux`` decode under ``--slot-deep-supervision``) -> ``.backward()``.
  argv ``--r9-box-s8-horizon`` (build_parser, g8 group, store_true) -> ``args.r9_box_s8_horizon`` -> ``pin_args`` (``_pin_refcv8``) -> train(): ``model._rig_camera``
  (``_build_rig_camera``) -> ``attach`` -> ``Box3DMemory.enable_s8`` + ``Box3DSlotDecoder.enable_s8_memory`` (``model._perception.box_mem.s8_proj`` /
  ``.box_dec.mem_pos_s8``) -> ``RefCV3Model._bev_hook`` hands ``fmap_s8`` to the branch (``br.wants_s8``) -> ``PerceptionBranch.forward`` ->
  ``Box3DMemory.forward(.., s8=fmap_s8)`` -> ``Box3DSlotDecoder.forward`` / ``box_b3.run_decoder`` / ``slot_query_select.anchored_forward``
  (``dec.memory_pos(memory)``) -> the box3d loss -> ``.backward()``.  Eval: ``refcv7_loader`` -> ``attach(.., stamped=config["r9_box_c5"])``.
  argv ``--r9-box-perframe-queries`` (build_parser, g8 group, store_true) -> ``args.r9_box_perframe_queries`` -> ``pin_args`` (``_pin_refcv8``) -> train(): ``attach`` (after
  ``_p3f.attach``) -> ``model._perception.box_pfq`` (``PerFrameQueries``) + ``box_qpos`` -> ``stage(model, batch, device)`` (``compute_losses_v3`` / ``compute_losses_perc_rig``,
  BEFORE the forward) -> ``RefCV3Model._bev_hook`` hands ``fmap_s8`` (``br.wants_s8``) -> ``PerceptionBranch.forward`` -> ``box_pfq(persp_head, fmap_s8, fmap_s16)``
  (seeds under ``no_grad`` + the global anchors) -> ``anchored_forward`` -> the box3d loss -> ``.backward()`` -> the global anchors' grad -> ``build_optimizer``'s
  ``box_anchor`` group (``split_anchor_group``; ``assert_optimizer_matches_recipe`` reads lr ratio ``box_anchor / head`` == 10 after ``apply_lr_schedule``).
"""
from __future__ import annotations

FLAGS = {"layers": "--r9-box-layers", "s8_horizon": "--r9-box-s8-horizon", "perframe_queries": "--r9-box-perframe-queries"}
DESTS = {k: "r9_box_" + k for k in FLAGS}
#: the depth of the tip's box decoder (``Box3DSlotDecoder(depth=3)`` as ``PerceptionBranch`` builds it) -- what an unnamed flag means
TIP_DEPTH: int = 3
#: the key of the stamp in config.json
STAMP_KEY = "r9_box_c5"


def layers_named(args) -> int:
    """The depth the argv NAMES (0 = not named)."""
    return int(getattr(args, DESTS["layers"], 0) or 0)


def declared_depth(args) -> int:
    n = layers_named(args)
    return n if n > 0 else TIP_DEPTH


def s8_named(args) -> bool:
    """``--r9-box-s8-horizon`` is on the argv (a store_true flag; absent from an older config.json's namespace = off)."""
    return bool(getattr(args, DESTS["s8_horizon"], False))


def pfq_named(args) -> bool:
    """``--r9-box-perframe-queries`` is on the argv (a store_true flag; absent from an older config.json's namespace = off)."""
    return bool(getattr(args, DESTS["perframe_queries"], False))


def is_any(args) -> bool:
    return layers_named(args) > 0 or s8_named(args) or pfq_named(args)


def pin_args(args) -> None:
    """The flags' refusals, at argv time.  All OFF: nothing is checked."""
    v = getattr(args, DESTS["layers"], 0)
    v = 0 if v is None else v
    if isinstance(v, bool) or not isinstance(v, int):
        raise SystemExit(f"[box-c5] {FLAGS['layers']} must be an integer, got {v!r}")
    if v < 0:
        raise SystemExit(f"[box-c5] {FLAGS['layers']} {v} < 0: a decoder depth is >= 1 (0 = not named = the tip's {TIP_DEPTH})")
    if not is_any(args):
        return
    from tanitad.train import map_only_ft as _mof
    if layers_named(args) > 0:
        bad = []
        if not float(getattr(args, "w_box3d", 0.0) or 0.0) > 0.0:
            bad.append("--w-box3d is 0: there is no box3d decoder for --r9-box-layers to deepen (a dead flag)")
        if _mof.is_on(args):
            bad.append("a --r8-train-only-map / --r9-train-only freeze is declared: the NEW decoder layers would stay at their template init")
        if bad:
            raise SystemExit("[box-c5] --r9-box-layers refused:\n  - " + "\n  - ".join(bad))
    if pfq_named(args):
        bad = []
        if not float(getattr(args, "w_box3d", 0.0) or 0.0) > 0.0:
            bad.append("--w-box3d is 0: there is no box3d decoder for the per-frame queries to seed (a dead flag)")
        if not bool(getattr(args, "r9_box_persp_aux", False)) or not float(getattr(args, "w_box_persp", 0.0) or 0.0) > 0.0:
            bad.append("--r9-box-persp-aux (with --w-box-persp > 0) is not on the argv: the seeds are the perspective head's s8 heat/depth -- an untrained head would seed nothing")
        if str(getattr(args, "map_hires", "off") or "off") != "on":
            bad.append("--map-hires is not 'on': the stride-8 tap the seed head reads exists only behind it")
        if str(getattr(args, "agent_rig_camera", "off") or "off") != "extrinsics":
            bad.append("--agent-rig-camera is not 'extrinsics': the seeds are lifted through the clip's own camera (a nominal one mis-places every seed)")
        if str(getattr(args, "opt", "adam") or "adam") != "dd":
            bad.append("--opt is not 'dd': the global anchors' LR x10 is a third group split out of `build_optimizer`'s dd groups; `adam` has a single group")
        if str(getattr(args, "slot_query_select", "learned") or "learned") != "learned":
            bad.append("--slot-query-select is not 'learned': heatmap / learned_ref are other anchor suppliers for the SAME slots (two suppliers = a non-attributable arm)")
        if _mof.is_on(args):
            bad.append("a --r8-train-only-map / --r9-train-only freeze is declared: the NEW anchors / position embed would stay at their init")
        if bad:
            raise SystemExit("[box-c5] --r9-box-perframe-queries refused:\n  - " + "\n  - ".join(bad))
    if s8_named(args):
        bad = []
        if not float(getattr(args, "w_box3d", 0.0) or 0.0) > 0.0:
            bad.append("--w-box3d is 0: there is no box memory for the s8 horizon tokens to join (a dead flag)")
        if str(getattr(args, "map_hires", "off") or "off") != "on":
            bad.append("--map-hires is not 'on': the stride-8 tap (`enable_s8_tap`) exists only behind it, so no fmap_s8 would ever reach the branch")
        if str(getattr(args, "agent_rig_camera", "off") or "off") != "extrinsics":
            bad.append("--agent-rig-camera is not 'extrinsics': the horizon row is DERIVED from the per-clip mount pitch of the extrinsics bank "
                       "(a nominal camera has no pitch, so its 'horizon' would be the frame centre by construction)")
        if _mof.is_on(args):
            bad.append("a --r8-train-only-map / --r9-train-only freeze is declared: the NEW s8 keys (s8_proj, src_embed_s8, mem_pos_s8) would stay at their init")
        if bad:
            raise SystemExit("[box-c5] --r9-box-s8-horizon refused:\n  - " + "\n  - ".join(bad))


def _box_dec(model):
    br = getattr(model, "_perception", None)
    return None if br is None else getattr(br, "box_dec", None)


def _mem(model):
    br = getattr(model, "_perception", None)
    return None if br is None else getattr(br, "box_mem", None)


def _band_rows() -> int:
    """The band's height in s8 rows (REFCV9_DESIGN 2.4: "+16 s8 rows around the horizon") -- ``box3d_head.S8_BAND_ROWS``."""
    from tanitad.models.box3d_head import S8_BAND_ROWS
    return int(S8_BAND_ROWS)


def horizon_row0(cam, s8_hw, n_rows: int | None = None) -> dict:
    """The s8 band's first row, DERIVED from the rig geometry (never a literal).

    The canonical frame is cylindrical and principal-point centred (``calib.py``), so the optical axis sits on the centre row and a ray at the centre
    column (u = 0) is ``(0, v, 1)`` in the camera frame (``rig_projection.frame_to_cam_ray``).  It is horizontal in the rig frame (rig z up) when
    ``R_cam_to_rig[2] . (0, v, 1) = 0``, i.e. ``v_h = -R[2,2] / R[2,1]``, and lands on pixel row ``(H-1)/2 + f_ref * v_h``.  ``cam`` is ONE ``RigCamera``
    or a ``RigCameraBank`` (every camera of its table, the bank's fallback only when the table is empty): the band centre is the MEDIAN row over them.
    An s8 cell ``i`` covers pixel rows ``8i .. 8i+7`` (centre ``8i + 3.5``), so pixel row ``p`` is s8 coordinate ``(p + 0.5) / 8 - 0.5``; the band is
    ``n_rows`` cells centred there: ``row0 = clip(round(centre - (n_rows - 1) / 2), 0, s8_h - n_rows)``.  Returns the band + its evidence."""
    import statistics
    n_rows = _band_rows() if n_rows is None else int(n_rows)
    h8, w8 = int(s8_hw[0]), int(s8_hw[1])
    if cam is None:
        raise SystemExit("[box-c5] --r9-box-s8-horizon needs the rig camera (model._rig_camera, built by `_build_rig_camera` from --agent-rig-camera extrinsics) "
                         "or the stamped band (config.json r9_box_c5.s8_horizon): neither reached attach()")
    by_ep = getattr(cam, "by_episode", None)
    if by_ep is not None:
        cams = list(by_ep.values()) or ([cam.default] if getattr(cam, "default", None) is not None else [])
    else:
        cams = [cam]
    if not cams:
        raise SystemExit("[box-c5] --r9-box-s8-horizon: the rig camera bank holds no camera")
    rows = []
    for c in cams:
        H, f = int(c.frame.height), float(c.frame.f_ref)
        if H != 8 * h8:
            raise SystemExit(f"[box-c5] the rig camera's frame is {H} rows but the s8 map has {h8} rows (8 x {h8} = {8 * h8}): the horizon row would be read in the wrong grid")
        r21, r22 = float(c.R_cam_to_rig[2, 1]), float(c.R_cam_to_rig[2, 2])
        if abs(r21) < 1e-6:
            raise SystemExit(f"[box-c5] a camera's optical y axis is horizontal in the rig (R[2,1] = {r21:.3g}): its horizon row is undefined")
        rows.append((H - 1) / 2.0 + f * (-r22 / r21))
    med = float(statistics.median(rows))
    centre = (med + 0.5) / 8.0 - 0.5
    raw = round(centre - (n_rows - 1) / 2.0)
    row0 = min(max(int(raw), 0), h8 - n_rows)
    return {"row0": int(row0), "n_rows": int(n_rows), "s8_hw": [h8, w8], "n_tokens": int(n_rows * w8),
            "median_row_px": med, "min_row_px": float(min(rows)), "max_row_px": float(max(rows)), "n_cameras": len(rows),
            "centre_s8_row": float(centre), "clipped": bool(row0 != int(raw)), "source": "rig_camera"}


def _attach_s8(model, args, stamped) -> None:
    import torch
    mem, dec = _mem(model), _box_dec(model)
    if mem is None or dec is None:
        raise SystemExit("[box-c5] --r9-box-s8-horizon but the model has no box memory / decoder (--w-box3d > 0 builds them)")
    enc = model.core.encoder
    if not getattr(enc, "s8_tap", False) or getattr(enc, "s8_dim", None) is None or getattr(enc, "s8_shape", None) is None:
        raise SystemExit("[box-c5] --r9-box-s8-horizon: the trunk has no stride-8 tap (--trunk timm --map-hires on enables it BEFORE the perception branch)")
    s8_hw = tuple(int(v) for v in enc.s8_shape)
    if stamped is not None and stamped.get("s8_horizon") is not None:
        info = dict(stamped["s8_horizon"])
        if tuple(int(v) for v in info.get("s8_hw", ())) != s8_hw:
            raise SystemExit(f"[box-c5] the stamped s8 band was built for an s8 map {info.get('s8_hw')} but this trunk's is {list(s8_hw)}")
        if int(info.get("n_rows", -1)) != _band_rows():
            raise SystemExit(f"[box-c5] the stamped s8 band has {info.get('n_rows')} rows, this build's is {_band_rows()}")
        info["source"] = "stamp"
    else:
        info = horizon_row0(getattr(model, "_rig_camera", None), s8_hw)
    with torch.random.fork_rng(devices=[]):          # the new params' init draws move NO other parameter's draw (B4 vs B4-minus-s8 share every shared init)
        mem.enable_s8(int(enc.s8_dim), s8_hw, int(info["row0"]), int(info["n_rows"]))
        dec.enable_s8_memory(mem.s8_n_tokens)
    mem.s8_info = info


def _attach_pfq(model, args, seed_standin: bool) -> None:
    import torch
    from tanitad.models import slot_query_select as _sqs
    br = getattr(model, "_perception", None)
    dec = _box_dec(model)
    if br is None or dec is None:
        raise SystemExit("[box-c5] --r9-box-perframe-queries but the model has no box3d decoder (--w-box3d > 0 builds it)")
    if int(dec.n_queries) != _sqs.PFQ_N_SEEDS + _sqs.PFQ_N_GLOBAL:
        raise SystemExit(f"[box-c5] --r9-box-perframe-queries: the box decoder has {dec.n_queries} queries, the design's slots are {_sqs.PFQ_N_SEEDS} seeds + "
                         f"{_sqs.PFQ_N_GLOBAL} global anchors = {_sqs.PFQ_N_SEEDS + _sqs.PFQ_N_GLOBAL}")
    enc = model.core.encoder
    if not getattr(enc, "s8_tap", False) or getattr(enc, "s8_dim", None) is None:
        raise SystemExit("[box-c5] --r9-box-perframe-queries: the trunk has no stride-8 tap (--trunk timm --map-hires on)")
    if getattr(br, "box_heat", None) is not None or getattr(br, "box_refpts", None) is not None:
        raise SystemExit("[box-c5] --r9-box-perframe-queries: the branch already has another anchor supplier (heatmap / learned_ref)")
    dev = next(br.parameters()).device
    with torch.random.fork_rng(devices=[]):        # no other parameter's init draw moves (B4 vs B4-minus-pfq share every shared init)
        torch.manual_seed(0)
        pfq = _sqs.PerFrameQueries(br.cfg.planner_grid)
        qpos = _sqs.AnchorPosEmbed(int(br.cfg.d_model), br.cfg.planner_grid)
        if getattr(br, "box_persp_aux", None) is None:
            # train(): `_p3f.attach` ran first, so the head exists.  The eval loader attaches the head AFTER this call (dvb_check_pfq reports a head that never arrives).
            # `seed_standin=True` is the harness REPLAY only (it strips --r9-box-persp-aux): a random, FROZEN, deterministic stand-in -- stamped as such.
            if seed_standin:
                from tanitad.models import persp3d_aux as P
                pfq.standin = P.Persp3DAuxHead(int(enc.s8_dim), int(enc.s16_dim))
                for p_ in pfq.standin.parameters():
                    p_.requires_grad_(False)
    br.box_pfq = pfq.to(dev)
    br.box_qpos = qpos.to(dev)


def stage(model, batch: dict, device) -> None:
    """Called by ``compute_losses_perc_rig`` AND ``compute_losses_v3`` BEFORE ``model(...)`` (and by an eval / dump caller): stages the per-row rig camera on
    the seed module, one-shot.  OFF (no ``box_pfq``): nothing.  A batch without ``agent_ep``, a row with no camera, or cameras on different frames REFUSES."""
    import torch
    br = getattr(model, "_perception", None)
    pfq = None if br is None else getattr(br, "box_pfq", None)
    if pfq is None:
        return
    from tanitad.refs import refc_agents as RA
    ep = batch.get("agent_ep")
    if ep is None:
        raise SystemExit("[box-c5] --r9-box-perframe-queries: the batch carries no `agent_ep` (--agent-join): no camera to lift the seeds through")
    cam = getattr(model, "_rig_camera", None)
    cams = cam.for_episodes(ep) if isinstance(cam, RA.RigCameraBank) else RA.row_cameras(cam, int(ep.shape[0]))
    if any(c is None for c in cams):
        raise SystemExit("[box-c5] --r9-box-perframe-queries: a batch row has no rig camera (the extrinsics table does not cover its clip)")
    fr = cams[0].frame
    if any((c.frame.height, c.frame.width, float(c.frame.f_ref), c.frame.projection) != (fr.height, fr.width, float(fr.f_ref), fr.projection) for c in cams):
        raise SystemExit("[box-c5] --r9-box-perframe-queries: the batch's cameras are on different canonical frames")
    R = torch.stack([c.R_cam_to_rig for c in cams]).to(device).float()
    t = torch.stack([c.t_cam_in_rig for c in cams]).to(device).float()
    pfq.stage(R, t, fr)


def expected_groups(args) -> set:
    """The optimizer's param-group names the recipe demands: ``--opt dd``'s encoder / head, plus ``box_anchor`` when the per-frame queries are named."""
    return {"encoder", "head"} | ({"box_anchor"} if pfq_named(args) else set())


def split_anchor_group(model, groups: list, args, lr: float) -> list:
    """``build_optimizer``'s hook: move the global anchors out of the ``head`` group into their own ``box_anchor`` group at ``lr`` x ``PFQ_ANCHOR_LR_MULT``
    (same weight decay).  Identity-based (never a name guess).  OFF: ``groups`` unchanged."""
    if not pfq_named(args):
        return groups
    from tanitad.models.slot_query_select import PFQ_ANCHOR_LR_MULT
    br = getattr(model, "_perception", None)
    pfq = None if br is None else getattr(br, "box_pfq", None)
    if pfq is None:
        raise SystemExit("[box-c5] --r9-box-perframe-queries but the model has no box_pfq: the anchors' LR group would not exist")
    head = next(g for g in groups if g.get("name") == "head")
    keep = [p for p in head["params"] if p is not pfq.anchors]
    if len(keep) != len(head["params"]) - 1:
        raise SystemExit("[box-c5] the global anchors are not in the head group (frozen, or already split)")
    head["params"] = keep
    return groups + [{"params": [pfq.anchors], "lr": float(lr) * PFQ_ANCHOR_LR_MULT, "weight_decay": head["weight_decay"], "name": "box_anchor"}]


def attach(model, args, *, stamped=None, seed_standin: bool = False) -> None:
    """Build-time wiring, called by the trainer (and the eval loader) once the perception branch exists and BEFORE the optimizer is built and
    before any warm start.  OFF: touches nothing.  ``stamped`` = the run's ``config.json["r9_box_c5"]`` (eval loader): the s8 band's row is read from it
    instead of from a camera bank (which the eval does not have)."""
    if not is_any(args):
        return
    if layers_named(args) > 0:
        dec = _box_dec(model)
        if dec is None:
            raise SystemExit("[box-c5] --r9-box-layers but the model has no box3d decoder (--w-box3d > 0 builds it)")
        dec.extend_depth(layers_named(args))             # raises ValueError above the D2 cap (BOX3D_PARAM_CAP)
    if s8_named(args):
        _attach_s8(model, args, stamped)
    if pfq_named(args):
        _attach_pfq(model, args, bool(seed_standin or getattr(model, "_pfq_seed_standin", False)))


def stamp(model, args) -> dict | None:
    """``config.json["r9_box_c5"]`` -- read OFF the built decoder / memory; ``None`` (key absent) when every flag is OFF."""
    if not is_any(args):
        return None
    from tanitad.models.box3d_head import BOX3D_PARAM_CAP
    dec = _box_dec(model)
    out = {"flags": {}}
    if layers_named(args) > 0:
        out["flags"][FLAGS["layers"]] = layers_named(args)
        out["layers"] = {"depth": None if dec is None else len(dec.blocks.layers), "tip_depth": TIP_DEPTH,
                         "new_layer_init": "deep copy of the template layer (= nn.TransformerDecoder(layer, num_layers=N)); consumes no RNG",
                         "box_decoder_params": None if dec is None else int(dec.n_params), "d2_cap": int(BOX3D_PARAM_CAP),
                         "d2_scope": "Box3DSlotDecoder (the decoder module); box_memory and the training-only aux heads are reported separately "
                                     "in perception.branch_params"}
    if pfq_named(args):
        from tanitad.models.slot_query_select import PFQ_ANCHOR_LR_MULT
        br = getattr(model, "_perception", None)
        pfq = None if br is None else getattr(br, "box_pfq", None)
        out["flags"][FLAGS["perframe_queries"]] = True
        out["perframe_queries"] = None if pfq is None else {
            "n_seeds": int(pfq.n_seeds), "n_global": int(pfq.n_global), "n_queries": int(dec.n_queries),
            "seed_source": "replay_standin_random_frozen_head" if pfq.standin is not None else "perspective_head_s8 (box_persp_aux)",
            "seed_rule": "3x3 max-pool NMS top-200 of sigmoid(s8 heat); cell centre ((j+.5)*8-.5, (i+.5)*8-.5) -> frame_to_cam_ray (unit) x E[LID range] -> RigCamera.to_rig -> (x, y) clamped to the planner grid",
            "seed_grad": "DETACHED (no_grad): the box loss never reaches the perspective head or the trunk through the seeds",
            "global_anchors": "nn.Parameter [100, 2], normalised [0,1] (x / x_fwd, (y + y_half) / (2 y_half)), init torch.rand seed 0",
            "anchor_lr_mult": float(PFQ_ANCHOR_LR_MULT), "anchor_group": "box_anchor (third optimizer group, split from head; --opt dd)",
            "anchor_offset_m": 4.0, "new_params": {"box_pfq.anchors": int(pfq.anchors.numel()),
                                                   "box_qpos": int(sum(p.numel() for p in br.box_qpos.parameters()))}}
    if s8_named(args):
        out["flags"][FLAGS["s8_horizon"]] = True
        mem = _mem(model)
        if mem is None or getattr(mem, "s8_proj", None) is None:
            out["s8_horizon"] = None                                                # declared but NOT built: dvb_check_s8 reports it
        else:
            info = dict(getattr(mem, "s8_info", None) or {})
            pos = None if dec is None else getattr(dec, "mem_pos_s8", None)
            info.update({"row0": int(mem.s8_rows[0]), "n_rows": int(mem.s8_rows[1]), "s8_hw": [int(v) for v in mem.s8_hw],
                         "n_tokens": int(mem.s8_n_tokens), "memory_tokens_total": None if dec is None else int(dec.n_memory_in),
                         "new_params": {"box_mem.s8_proj": int(sum(p.numel() for p in mem.s8_proj.parameters())),
                                        "box_mem.src_embed_s8": int(mem.src_embed_s8.numel()),
                                        "box_dec.mem_pos_s8": None if pos is None else int(pos.numel())},
                         "box_decoder_params": None if dec is None else int(dec.n_params), "d2_cap": int(BOX3D_PARAM_CAP),
                         "placement": "appended LAST to the box memory (after the s16 image tokens and the BEV tokens)",
                         "band_scope": "pixel rows 8*row0 .. 8*(row0+n_rows)-1 of the canonical frame, all s8 columns"})
            out["s8_horizon"] = info
    return out


# ---- G-DVB ("built" levers): the argv's declaration equals what the BUILT model holds --------------------------------------------- #
def _mm(dest, declared, built, where, why):
    from tanitad.train.declared_vs_built import Mismatch
    return Mismatch(dest, declared, built, where, why)


def dvb_check_layers(model, args) -> list:
    """declared <=> the decoder has that many layers.  Unnamed (0) means the tip's depth: a model with a box decoder must hold exactly
    ``TIP_DEPTH`` layers (a build that deepened the decoder without the argv naming it is a mismatch), and a model with none is fine."""
    dest = DESTS["layers"]
    want = declared_depth(args)
    dec = _box_dec(model)
    out = []
    if dec is None:
        if layers_named(args) > 0:
            out.append(_mm(dest, want, "no box3d decoder", "model._perception.box_dec", "argv names a depth and the model has no box3d decoder"))
        return out
    built = len(dec.blocks.layers)
    if built != want:
        out.append(_mm(dest, want, built, "model._perception.box_dec.blocks.layers", "argv and the BUILT decoder depth disagree"))
    if int(getattr(dec.blocks, "num_layers", built)) != built:
        out.append(_mm(dest, built, int(dec.blocks.num_layers), "model._perception.box_dec.blocks.num_layers",
                       "the stack's own num_layers attribute disagrees with its layer list"))
    return out


def dvb_check_s8(model, args) -> list:
    """declared <=> the BUILT box memory carries the s8 band: ``s8_proj`` exists exactly when the argv names the flag, the decoder's memory length is the
    memory's token count + the band's, the decoder holds the band's positional table, the band lies inside the s8 map, and ``s8_proj`` reads the trunk's s8 width."""
    dest = DESTS["s8_horizon"]
    want = s8_named(args)
    mem, dec = _mem(model), _box_dec(model)
    built = mem is not None and getattr(mem, "s8_proj", None) is not None
    out = []
    if want and (mem is None or dec is None):
        return [_mm(dest, True, "no box memory / decoder", "model._perception.box_mem / box_dec", "argv names the s8 band and the model has no box3d memory")]
    if want != built:
        out.append(_mm(dest, want, built, "model._perception.box_mem.s8_proj", "argv and the BUILT memory disagree about the s8 horizon band"))
    has_pos = dec is not None and getattr(dec, "mem_pos_s8", None) is not None
    if has_pos != built:
        out.append(_mm(dest, built, has_pos, "model._perception.box_dec.mem_pos_s8", "the memory's s8 band and the decoder's s8 positional table disagree"))
    if not built:
        return out
    if has_pos:
        if int(dec.n_memory_in) != int(mem.n_tokens) + int(mem.s8_n_tokens):
            out.append(_mm(dest, int(mem.n_tokens) + int(mem.s8_n_tokens), int(dec.n_memory_in), "model._perception.box_dec.n_memory_in",
                           "the decoder does not take the memory length the box memory emits (s16 + BEV + s8 band)"))
        if int(dec.mem_pos_s8.shape[1]) != int(mem.s8_n_tokens):
            out.append(_mm(dest, int(mem.s8_n_tokens), int(dec.mem_pos_s8.shape[1]), "model._perception.box_dec.mem_pos_s8",
                           "the band's positional table is not one row per band token"))
    r0, nr = mem.s8_rows
    if not (0 <= r0 and r0 + nr <= int(mem.s8_hw[0])) or nr != _band_rows():
        out.append(_mm(dest, f"{_band_rows()} rows inside [0, {mem.s8_hw[0]})", f"row0 {r0}, {nr} rows", "model._perception.box_mem.s8_rows",
                       "the s8 band is outside the s8 map or is not the design's 16 rows"))
    enc = getattr(getattr(model, "core", None), "encoder", None)
    if enc is not None and getattr(enc, "s8_dim", None) is not None:
        if int(mem.s8_proj.in_features) != int(enc.s8_dim):
            out.append(_mm(dest, int(enc.s8_dim), int(mem.s8_proj.in_features), "model._perception.box_mem.s8_proj.in_features vs core.encoder.s8_dim",
                           "the band's projection does not read the trunk's stride-8 width"))
        if tuple(int(v) for v in mem.s8_hw) != tuple(int(v) for v in enc.s8_shape):
            out.append(_mm(dest, tuple(enc.s8_shape), tuple(mem.s8_hw), "model._perception.box_mem.s8_hw vs core.encoder.s8_shape",
                           "the band was built for another s8 map than the trunk emits"))
    br = getattr(model, "_perception", None)
    if br is not None and not getattr(br, "wants_s8", False):
        out.append(_mm(dest, True, False, "model._perception.wants_s8", "the memory has the band but the branch would not ask for fmap_s8"))
    return out


def dvb_check_pfq(model, args) -> list:
    """declared <=> the BUILT branch holds ``box_pfq`` (100 x 2 global anchors, 300 slots = 200 + 100), its position embed, a seed head and the s8 hand-in."""
    dest = DESTS["perframe_queries"]
    want = pfq_named(args)
    br = getattr(model, "_perception", None)
    pfq = None if br is None else getattr(br, "box_pfq", None)
    out = []
    if want and br is None:
        return [_mm(dest, True, "no perception branch", "model._perception", "argv names the per-frame queries and the model has no perception branch")]
    if want != (pfq is not None):
        out.append(_mm(dest, want, pfq is not None, "model._perception.box_pfq", "argv and the BUILT branch disagree about the per-frame queries"))
    if pfq is None:
        return out
    if getattr(br, "box_qpos", None) is None:
        out.append(_mm(dest, True, False, "model._perception.box_qpos", "the per-frame queries have no anchor position embed"))
    if tuple(pfq.anchors.shape) != (pfq.n_global, 2) or pfq.n_seeds + pfq.n_global != int(br.box_dec.n_queries):
        out.append(_mm(dest, f"{pfq.n_seeds}+{pfq.n_global}={br.box_dec.n_queries} slots, anchors ({pfq.n_global}, 2)", f"anchors {tuple(pfq.anchors.shape)}",
                       "model._perception.box_pfq.anchors", "the global anchors / seed count do not fill the decoder's queries"))
    if pfq.standin is None and getattr(br, "box_persp_aux", None) is None:
        out.append(_mm(dest, "a perspective seed head", None, "model._perception.box_persp_aux", "no seed head: --r9-box-persp-aux was not built"))
    if not getattr(br, "wants_s8", False):
        out.append(_mm(dest, True, False, "model._perception.wants_s8", "the branch would not be handed fmap_s8 (the seed head reads it)"))
    return out


def register_dvb_levers(register) -> None:
    """Called by the trainer at import (the registry is pinned two-way against its parser)."""
    register(DESTS["layers"], "built", dvb_check_layers)
    register(DESTS["s8_horizon"], "built", dvb_check_s8)
    register(DESTS["perframe_queries"], "built", dvb_check_pfq)
