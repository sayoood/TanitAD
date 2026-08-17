"""SP1 — cache the FROZEN v6 trunk's spatial memory + agent-slot targets.

The P8 idiom (``train_p8_occupancy.py``): run the frozen trunk ONCE, bank the
latents, and let every arm and every control be a re-read of that tensor. This
is what turns the estimate from a GPU-day into minutes and what makes C-SHUF /
C-CONST free.

⛔ FROZEN TRUNK. The trunk is loaded through
``tanitad.eval.v6_probe_trunk.load_trunk_auto`` with ``requires_grad_(False)``
on every parameter, and the cache this writes is a DETACHED tensor with no
autograd graph — so the probe trained on it cannot reach the trunk even in
principle. That is a strictly stronger isolation than the ``perception_to_trunk``
edge, which is additionally checked by ``sp0_isolation.py``.

⭐ WHY PER-FRAME AND NOT PER-WINDOW. ``V6Stack.encode_window`` is per-frame
(``flat = frames.reshape(b*w, …); tok = self.encoder(flat)``), so the present
frame's memory does not depend on the rest of the window. The cache therefore
encodes ``frames[:, -1:]`` only — the same one-frame path
``train_p8_occupancy.p8_latents_ex`` takes when it does not need the roll — and
costs 1/window of the naive pass.

TWO MEMORY SURFACES, the pre-registered arm and its control (AGENT_SLOT_DECODER
§1.4):
  * ``cells``  — ``V6Stack.cells(z_op)`` [B, n_cells, d_readout]; failure means
    the LATENT does not carry agents.
  * ``tokens`` — the encoder's raw patch tokens [B, n_tokens, d_model]; failure
    means the ENCODER does not carry them.

TARGETS come from the EXISTING join only (``build_obstacle_join.py`` ->
``train_p8_occupancy.JoinFileReader``), never a second derivation. A frame
ABSENT from the join is NO_LABEL and is skipped+counted; an EMPTY agents list IS
a label (road clear) and is kept.

⚠️ IN-GRID RESTRICTION — a stated decision, not a default. Targets are
restricted to ``0 < cx <= grid.x_fwd_m`` and ``|cy| <= grid.y_half_m``
(``bev_raster.GRID_DEFAULT``: 60 m fwd, +-16 m), the extents
``SlotDecodeRanges`` decodes into. An agent 200 m behind the ego is not
representable by this head's decode, so scoring it would measure the coordinate
transform rather than the latent. The count dropped by this rule is recorded per
frame (``n_out_of_grid``), never silently discarded.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def build_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-json", default=None,
                    help="run config.json when the ckpt travelled without one")
    ap.add_argument("--step", type=int, default=None,
                    help="explicit <run>@<step> stamp when the artifact carries "
                         "no `step` key (§1.4b). RECORDED with its source.")
    ap.add_argument("--run-name", default="v6F-SW-30k")
    ap.add_argument("--v2-cache", nargs="+", required=True)
    ap.add_argument("--join-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=2,
                    help="frame stride over labelled frames (10 Hz corpus; "
                         "consecutive frames are near-duplicates)")
    ap.add_argument("--want-tokens", action="store_true",
                    help="also bank the encoder patch tokens (the C-TOK arm). "
                         "~1 MB/frame in fp16 — off by default.")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--v2-lru", type=int, default=6)
    ap.add_argument("--max-episodes", type=int, default=0,
                    help="0 = every episode the cache dir holds")
    # geometry — must match the trunk's own; the loader REFUSES a contradiction
    ap.add_argument("--frame-h", type=int, default=256)
    ap.add_argument("--frame-w", type=int, default=640)
    ap.add_argument("--frame-hfov", type=float, default=120.0)
    ap.add_argument("--projection", default="cylindrical")
    ap.add_argument("--v2-subframe", default=None)
    ap.add_argument("--require-parity", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = build_args(argv)
    sys.path.insert(0, str(Path(
        r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\stack\scripts")))
    from eval_flagship_v4 import _eval_cfg, _plan, resolve_eval_frames
    from train_flagship4b import FlagshipWindowDataset
    from train_p8_occupancy import JoinFileReader, window_frame
    from train_v58f_unicycle_head import build_train_episodes

    from tanitad.data.bev_raster import GRID_DEFAULT as GRID
    from tanitad.eval.v6_probe_trunk import load_trunk_auto

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- geometry FIRST (the eval seam, not re-resolved here) ---------------
    cfg = _eval_cfg()
    cache_frame, model_frame = resolve_eval_frames(a, cfg, label="sp1_cache")
    plan = _plan(cfg)

    # ---- the frozen trunk ---------------------------------------------------
    # ⛔ §1.4b: the checkpoint IS part of the arm. The step is read from the
    # ARTIFACT (`_meta['step']` for the trainer's fp16 snapshot, `ck['step']`
    # for a full ckpt) — never from prose, never from a filename.
    print(f"[sp1] loading checkpoint {a.ckpt} ...", flush=True)
    from sp_common import merge_run_args, read_fp16_snapshot
    sd, run_args, art_step, prov_ck = read_fp16_snapshot(a.ckpt, a.config_json)
    merged = vars(merge_run_args(run_args))
    step = int(a.step) if a.step is not None else art_step
    step_src = (prov_ck["step_source"] if a.step is None
                else "--step (operator override)")
    ck = {"stack": sd, "config": {"args": merged}, "step": step}
    world, _g, base_step = load_trunk_auto(ck, device, ckpt_path=a.ckpt,
                                           frame=model_frame)
    del ck, sd
    # ⛔ the isolation this whole probe rests on, asserted before any compute
    assert not any(p.requires_grad for p in world.parameters()), \
        "trunk parameters are not frozen"
    window = int(getattr(world, "window", cfg.predictor.window))
    trunk_frame = getattr(world, "frame", None)
    if trunk_frame is not None and trunk_frame != model_frame:
        raise SystemExit(
            f"[sp1] GEOMETRY CONTRADICTION: trunk {trunk_frame.height}x"
            f"{trunk_frame.width} HFOV {trunk_frame.hfov_deg:.3f} "
            f"{trunk_frame.projection} vs CLI {model_frame.height}x"
            f"{model_frame.width} HFOV {model_frame.hfov_deg:.3f} "
            f"{model_frame.projection}. Refusing to feed the encoder a field "
            f"it never saw.")
    trunk_ch = getattr(world, "in_channels", None)
    if trunk_ch is not None and int(trunk_ch) != int(cfg.encoder.in_channels):
        raise SystemExit(
            f"[sp1] CHANNEL CONTRADICTION: the trunk's encoder takes "
            f"{trunk_ch} input channels, the window loader would deliver "
            f"{cfg.encoder.in_channels}. Refusing rather than handing the "
            f"encoder a different stack depth than it was trained on.")
    stamp = f"{a.run_name}@{base_step}"
    print(f"[sp1] trunk FROZEN · {stamp} (step from {step_src}) · "
          f"d_op {world.state_dim} · cells {world.n_cells}x{world.d_readout} "
          f"· token_grid {world.token_grid} · window {window}", flush=True)

    # ---- data ---------------------------------------------------------------
    a.v2_lru = int(a.v2_lru)
    train_eps, prov = build_train_episodes(a, cache_frame=cache_frame,
                                           train_frame=model_frame)
    if a.max_episodes:
        train_eps = train_eps[:int(a.max_episodes)]
    ds = FlagshipWindowDataset(train_eps, window=window,
                               max_horizon=plan.max_horizon,
                               maneuver_h=plan.maneuver_h,
                               channels=cfg.encoder.in_channels)
    src = JoinFileReader(a.join_file)
    print(f"[sp1] {len(train_eps)} episodes / {len(ds)} windows; join "
          f"{src.n_records} records / {src.n_clips} clips "
          f"(occ_flags={src.has_occlusion_flags} classes={src.has_classes})",
          flush=True)

    # ---- raw join records, for the per-track finite-difference rates --------
    # `JoinFileReader` deliberately exposes ARRAYS; `track_rates_from_join`
    # needs the RECORDS (prev/cur/next, matched by track_id). Same file, read
    # once, and the record count is cross-checked against the reader's so the
    # two views cannot silently disagree.
    recs: dict[tuple[str, int], dict] = {}
    with open(a.join_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            recs[(str(r["clip_id"]), int(r["frame_idx"]))] = r
    if len(recs) != src.n_records:
        raise SystemExit(f"[sp1] record-view mismatch: {len(recs)} raw vs "
                         f"{src.n_records} reader records")

    # ---- the window grid, restricted to LABELLED present frames -------------
    from tanitad.models.agent_slots import (targets_from_join,
                                            track_rates_from_join)

    idx_all = list(range(len(ds.index)))
    covered, n_nolabel = [], 0
    for i in idx_all:
        eid, pf = window_frame(ds, i)
        if src.lookup(eid, pf) is None:
            n_nolabel += 1
        else:
            covered.append(i)
    covered = covered[::max(1, int(a.stride))]
    print(f"[sp1] labelled windows {len(covered)} (stride {a.stride}); "
          f"NO_LABEL skipped {n_nolabel}", flush=True)

    clip_of = {int(e.episode_id): None for e in train_eps}
    for uid in list(clip_of):
        clip_of[uid] = src._clip_of(uid)

    # ---- the frozen forward -------------------------------------------------
    from torch.utils.data import default_collate

    n_q_needed = 0
    rows: list[dict] = []
    t0 = time.time()
    peak_note = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for s in range(0, len(covered), int(a.batch)):
        chunk = covered[s:s + int(a.batch)]
        batch = default_collate([ds[i] for i in chunk])
        frames = batch["frames"][:, -1:].to(device)          # [B,1,C,H,W]
        # ⚠️ v0 is a PRIVILEGED ego channel and is banked for METRIC use only
        # (the time-gap denominator, `taniteval.lead_metrics`). It NEVER enters
        # the head: `AgentSlotDecoder.forward` takes one positional tensor and
        # has no keyword and no **kwargs, so there is no door for it.
        v0_b = batch["pose_last"][:, 3].float().cpu()
        with torch.no_grad(), torch.autocast(
                device.type, dtype=torch.bfloat16,
                enabled=(device.type == "cuda")):
            if a.want_tokens:
                z, tok = world.stack.encode_window(frames, return_tokens=True)
                tok = tok[:, 0].float()
            else:
                z = world.stack.encode_window(frames)
                tok = None
            z_op = z[:, 0].float()
            cells = world.cells(z_op)                        # [B, C, d_r]
        for j, i in enumerate(chunk):
            eid, pf = window_frame(ds, i)
            cid = clip_of[eid]
            ag = src.lookup(eid, pf)
            cls = src.lookup_classes(eid, pf)
            rec = recs[(cid, pf)]
            rates, rmask = track_rates_from_join(
                recs.get((cid, pf - 1)), rec, recs.get((cid, pf + 1)))
            ag = np.asarray(ag, dtype=np.float64).reshape(-1, 6)
            keep = ((ag[:, 0] > 0.0) & (ag[:, 0] <= GRID.x_fwd_m)
                    & (np.abs(ag[:, 1]) <= GRID.y_half_m)) if len(ag) \
                else np.zeros(0, dtype=bool)
            n_out = int(len(ag) - keep.sum())
            ag_k = ag[keep]
            cls_k = (np.asarray(cls, dtype=object)[keep].tolist()
                     if cls is not None and len(ag) else None)
            rt_k = rates[keep] if len(ag) else rates
            rm_k = rmask[keep] if len(ag) else rmask
            n_q_needed = max(n_q_needed, int(ag_k.shape[0]))
            rows.append({"episode_uid": int(eid), "clip_id": cid,
                         "frame_idx": int(pf),
                         "cells": cells[j].to(torch.float16).cpu(),
                         "tokens": (tok[j].to(torch.float16).cpu()
                                    if tok is not None else None),
                         "agents": torch.from_numpy(ag_k).float(),
                         "classes": cls_k,
                         "rates": torch.from_numpy(rt_k).float(),
                         "rates_mask": torch.from_numpy(rm_k),
                         "v0": float(v0_b[j]),
                         "n_out_of_grid": n_out})
        if (s // int(a.batch)) % 25 == 0:
            el = time.time() - t0
            print(f"[sp1] {len(rows)}/{len(covered)} frames  {el/60:.1f} min  "
                  f"{len(rows)/max(el,1e-9):.1f} fr/s", flush=True)
    if device.type == "cuda":
        peak_note = float(torch.cuda.max_memory_allocated()) / 1e9

    meta = {
        "run_stamp": stamp, "run": a.run_name, "step": int(base_step),
        "step_source": step_src, "ckpt_provenance": prov_ck,
        "ckpt_path": str(a.ckpt),
        "_evidence_class": "MEASURED (ours; frozen-trunk forward)",
        "eval_tier": "T0-DIAGNOSTIC",
        "n_frames": len(rows), "n_episodes": len(train_eps),
        "n_nolabel_skipped": n_nolabel, "stride": int(a.stride),
        "d_readout": int(world.d_readout), "n_cells": int(world.n_cells),
        "d_op": int(world.state_dim), "window": window,
        "token_grid": list(world.token_grid),
        "tokens_banked": bool(a.want_tokens),
        "d_model_tokens": (int(rows[0]["tokens"].shape[-1])
                           if a.want_tokens and rows else None),
        "n_tokens": (int(rows[0]["tokens"].shape[0])
                     if a.want_tokens and rows else None),
        "max_in_grid_agents_seen": n_q_needed,
        "grid": {"x_fwd_m": GRID.x_fwd_m, "y_half_m": GRID.y_half_m},
        "frame": {"h": model_frame.height, "w": model_frame.width,
                  "hfov_deg": round(model_frame.hfov_deg, 6),
                  "projection": model_frame.projection},
        "cuda_max_mem_gb": peak_note,
        "wall_s": round(time.time() - t0, 1),
        "join_file": str(a.join_file),
        "parity": {k: str(v)[:400] for k, v in prov.items()},
        "trunk_frozen_assert": "no parameter has requires_grad",
    }
    torch.save({"rows": rows, "meta": meta}, out / "latents.pt")
    (out / "sp1_meta.json").write_text(json.dumps(meta, indent=1), "utf-8")
    print("[sp1] DONE " + json.dumps(
        {k: meta[k] for k in ("run_stamp", "n_frames", "n_episodes",
                              "max_in_grid_agents_seen", "cuda_max_mem_gb",
                              "wall_s")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
