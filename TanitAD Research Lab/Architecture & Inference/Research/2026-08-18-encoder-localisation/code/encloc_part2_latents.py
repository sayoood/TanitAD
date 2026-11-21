"""PART 2, rungs 2-3 - the TRAINED checkpoint's latents over the same windows.

WHAT THIS BUILDS. E-GEOM measured rung 1 (raw refa-geometry DINOv2 features:
lead_gap 0.5285) and E-ADAPT-0 bounded the adapter ARCHITECTURE (tied per-cell
random Linear: 0.4888). This script forwards the TRAINED refa-dinov2 checkpoint
(HF Sayood/tanitad-refa-dinov2-4b, PI-authorized download) over the SAME banked
rows and banks, per row:

  rung2_trained   the trained TemporalGridAdapter's output state s_f   [2048]
  rung2_rand<s>   the SAME adapter class at random init, seeds 0/1/2   [2048]
                  (the positive control: if random reads signal through this
                  exact class+dims and trained reads none, TRAINING subtracted)
  rung3_h1        the predictor's one-step imagined latent z_hat_{f+1} [2048]
  rung3_h4        the 4-step (0.4 s) imagined latent z_hat_{f+4}       [2048]

The predictor is called EXACTLY as the T0 eval surface called it
(taniteval/rollout.py:182 -> rollout_decode(model.predictor, ...) -- NO intent),
so rung 3 probes the representation that produced the published 2.1675 ADE.

MODEL RECONSTRUCTION follows taniteval/taniteval/loaders.py:117-134 verbatim:
flagship4b_config(), action_dim -> 3 (--speed-input arm), RefAModelPlus.
from_stack_config(cfg, n_tokens=256, adapter_kind="temporal", d_dino=768),
STRICT state_dict load.

WINDOW/ACTION ALIGNMENT follows stack/scripts/refa_train.py:76-112: the window
is frames [f-W+1 .. f] (current frame LAST), actions are the per-frame
(steer, accel) at those frames, and --speed-input appends v0 = poses[f, 3] /
SPEED_SCALE (10.0) broadcast over the window (refa_train_plus.py:63-105,
channel order [steer, accel, v0]).

DECLARED RESIDUAL (unchanged from E-GEOM SS2.2): features are the refa1f
EMULATION (cylindrical 51.4-deg crop, 7% anisotropic stretch) of REF-A's
f-theta-crop input; the checkpoint's standardizer was fitted on the real
corpus. The residual is SHARED by every rung including rung 1, so within-input
contrasts cancel it. Rows with f < W-1 are PAD-LEFT (earliest frame repeated;
delta 0 at the pad, matching the adapter's own first-frame convention) and
counted in the meta.

COLLAPSE CHECK: the checkpoint banks NO training history, so adapter_std (the
monitor refa_train4b.py:360 logged) is COMPUTED here - global per-dim std of
each banked arm, trained vs random-init. That is the free check the download
was gated on, answered by direct measurement.
"""
from __future__ import annotations

import pyarrow  # noqa: F401  # isort: skip
import truststore  # isort: skip

truststore.inject_into_ssl()

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[5]
for _p in (_REPO / "stack", _REPO / "stack/experiments/reset-speed4b",
           _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from encloc_geom_cache import (ARMS as GEOM_ARMS,  # noqa: E402
                               IMAGENET_MEAN, IMAGENET_STD, crop_cols,
                               hf_token)

SPEED_SCALE = 10.0        # refa_train_plus.py:50 - v0 normalizer, verbatim
WINDOW = 8                # predictor window (registry: window 8)
EXPECTED_CKPT_BYTES = 1_905_662_297


def sha256_file(p: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def build_model(ckpt_path: str, device):
    """loaders.py:117-134, verbatim recipe (arch 'refa-plus', speed-input)."""
    from refa_plus import RefAModelPlus, TemporalGridAdapter
    from tanitad.config import flagship4b_config
    cfg = flagship4b_config()
    adim = 3                                   # 2 + v0 (--speed-input), no yaw
    if adim != cfg.predictor.action_dim:
        object.__setattr__(cfg.predictor, "action_dim", adim)
        if cfg.tactical_pred is not None:
            object.__setattr__(cfg.tactical_pred, "action_dim", adim)
    model = RefAModelPlus.from_stack_config(
        cfg, n_tokens=256, adapter_kind="temporal", d_dino=768)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])         # STRICT - 438 keys must match
    assert bool(model.standardizer.fitted), "standardizer not fitted"
    step = int(ck.get("step", -1))
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, step, TemporalGridAdapter


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--row-index", required=True)
    ap.add_argument("--episodes-dir", required=True)
    ap.add_argument("--tok-cache", default=None,
                    help="tok_refa1f.pt for the frame-f sanity cross-check")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="facebook/dinov2-base")
    ap.add_argument("--rand-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch-rows", type=int, default=4)
    ap.add_argument("--lru", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-rows", type=int, default=None,
                    help="SMOKE ONLY - strided subsample, never a finding")
    a = ap.parse_args(argv)

    n_ck = Path(a.ckpt).stat().st_size
    if n_ck != EXPECTED_CKPT_BYTES:
        raise SystemExit(f"[p2] GATE: ckpt is {n_ck} bytes, expected "
                         f"{EXPECTED_CKPT_BYTES} - refusing a partial file")
    ck_sha = sha256_file(a.ckpt)
    print(f"[p2] ckpt sha256 {ck_sha}", flush=True)

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model, step, TGA = build_model(a.ckpt, dev)
    print(f"[p2] model loaded STRICT, step={step}, state_dim="
          f"{model.state_dim}, horizons={model.pred_cfg.horizons}", flush=True)
    horizons = tuple(model.pred_cfg.horizons)
    assert 1 in horizons and 4 in horizons, horizons

    # random-init adapters: SAME class, SAME shape, fresh seeds.
    rand_adapters = {}
    for s in a.rand_seeds:
        torch.manual_seed(int(s))
        ad = TGA(256, 768, grid=4, d_readout=128).to(dev).eval()
        for p in ad.parameters():
            p.requires_grad_(False)
        rand_adapters[int(s)] = ad

    # ---- rows + geometry (identical to the refa1f arm) --------------------
    idx = torch.load(a.row_index, map_location="cpu", weights_only=False)
    rows, src_meta = idx["rows"], idx["meta"]
    if a.limit_rows:
        rows = rows[::max(1, len(rows) // int(a.limit_rows))]
    src_h = int(src_meta["frame"]["h"])
    src_w = int(src_meta["frame"]["w"])
    src_hfov = float(src_meta["frame"]["hfov_deg"])
    want_hfov = GEOM_ARMS["refa1f"][1]
    c0, c1, got_hfov = crop_cols(src_w, src_hfov, want_hfov)
    crop_w = c1 - c0
    H = W = 224
    print(f"[p2] refa1f geometry: crop cols [{c0}:{c1}] hfov {got_hfov:.2f} "
          f"deg -> {H}x{W}", flush=True)

    from transformers import AutoConfig, AutoModel
    tok_hf = hf_token()
    dcfg = AutoConfig.from_pretrained(a.model, token=tok_hf)
    dino = AutoModel.from_pretrained(a.model, token=tok_hf).to(dev).eval()
    for p in dino.parameters():
        p.requires_grad_(False)
    n_prefix = 1 + int(getattr(dcfg, "num_register_tokens", 0) or 0)
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)

    from tanitad.data.v2_dataset import _decode_stacked, _jpeg_offsets
    eps = Path(a.episodes_dir)
    payload_cache: dict[str, tuple] = {}

    def payload(cid: str):
        if cid not in payload_cache:
            if len(payload_cache) >= int(a.lru):
                payload_cache.pop(next(iter(payload_cache)))
            d = torch.load(eps / f"{cid}.v2ep.pt", map_location="cpu",
                           weights_only=False)
            payload_cache[cid] = (d["jpeg_buf"], _jpeg_offsets(d["jpeg_len"]),
                                  int(d["n_stack"]), str(d.get("codec", "jpeg")),
                                  d["actions"].float(), d["poses"].float())
        return payload_cache[cid]

    @torch.no_grad()
    def dino_frames(cid: str, f_lo: int, f_hi: int) -> torch.Tensor:
        """DINOv2 refa1f tokens for frames [f_lo..f_hi] -> [n, 256, 768] fp16.
        Latest RGB of each stack (dino_precompute.py:43 convention)."""
        buf, offs, n_stack, codec, _, _ = payload(cid)
        fr = _decode_stacked(buf, offs, n_stack, f_lo, f_hi + 1, codec, None)
        x = fr.to(dev).float() / 255.0                     # [n, 9, h, w]
        x = x[:, -3:, :, c0:c1]                            # latest RGB + crop
        x = F.interpolate(x, size=(H, W), mode="bilinear",
                          align_corners=False, antialias=True)
        x = (x - mean) / std
        with torch.autocast(dev.type, dtype=torch.float16,
                            enabled=(dev.type == "cuda")):
            hs = dino(pixel_values=x,
                      interpolate_pos_encoding=True).last_hidden_state
        return hs[:, n_prefix:].half()                     # [n, 256, 768]

    # ---- forward all rows --------------------------------------------------
    arms = (["rung2_trained"]
            + [f"rung2_rand{s}" for s in a.rand_seeds]
            + ["rung3_h1", "rung3_h4"])
    out = {k: [None] * len(rows) for k in arms}
    order = sorted(range(len(rows)),
                   key=lambda i: (rows[i]["clip_id"], rows[i]["frame_idx"]))
    n_pad_rows, sanity = 0, []
    t0 = time.time()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tok_ref = None
    if a.tok_cache:
        tk = torch.load(a.tok_cache, map_location="cpu", weights_only=False)
        tok_ref = {(r["clip_id"], int(r["frame_idx"])): r["tokens"]
                   for r in tk["rows"][:400]}      # sample subset is enough
        del tk

    for s0 in range(0, len(order), int(a.batch_rows)):
        sel = order[s0:s0 + int(a.batch_rows)]
        feats_w, acts_w = [], []
        for i in sel:
            cid, f = rows[i]["clip_id"], int(rows[i]["frame_idx"])
            f_lo = max(0, f - WINDOW + 1)
            tokw = dino_frames(cid, f_lo, f)               # [n, 256, 768]
            n_have = tokw.shape[0]
            _, _, _, _, act, pose = payload(cid)
            aw = act[f_lo:f + 1].clone()                   # [n, 2]
            if n_have < WINDOW:                            # pad-left, repeat
                n_pad_rows += 1
                padt = tokw[:1].expand(WINDOW - n_have, -1, -1)
                tokw = torch.cat([padt, tokw], 0)
                pada = aw[:1].expand(WINDOW - n_have, -1)
                aw = torch.cat([pada, aw], 0)
            v0 = (pose[f, 3] / SPEED_SCALE).reshape(1, 1).expand(WINDOW, 1)
            aw3 = torch.cat([aw, v0], 1)                   # [W, 3] steer,accel,v0
            feats_w.append(tokw)
            acts_w.append(aw3)
            if tok_ref is not None and (cid, f) in tok_ref:
                ref = tok_ref[(cid, f)].float()
                got = tokw[-1].float().cpu()
                c = float(np.corrcoef(ref.flatten(), got.flatten())[0, 1])
                sanity.append(c)
        fw = torch.stack(feats_w).to(dev)                  # [b, W, 256, 768]
        actions = torch.stack(acts_w).float().to(dev)      # [b, W, 3]
        with torch.no_grad():
            std_w = model.standardizer(fw)                 # [b, W, 256, 768] f32
            s_win = model.adapter.forward_window(std_w)    # [b, W, 2048]
            preds = model.predictor(s_win, actions, intent=None)
            for j, i in enumerate(sel):
                out["rung2_trained"][i] = s_win[j, -1].half().cpu().clone()
                out["rung3_h1"][i] = preds[1][j].half().cpu().clone()
                out["rung3_h4"][i] = preds[4][j].half().cpu().clone()
            for s, ad in rand_adapters.items():
                s_r = ad.forward_window(std_w)
                for j, i in enumerate(sel):
                    out[f"rung2_rand{s}"][i] = s_r[j, -1].half().cpu().clone()
        if (s0 // int(a.batch_rows)) % 40 == 0:
            el = time.time() - t0
            print(f"[p2] {s0 + len(sel)}/{len(order)}  {el/60:.1f} min  "
                  f"{(s0 + len(sel))/max(el, 1e-9):.1f} row/s", flush=True)

    peak = (float(torch.cuda.max_memory_allocated()) / 1e9
            if dev.type == "cuda" else None)
    if sanity:
        s_arr = np.array(sanity)
        print(f"[p2] SANITY frame-f tokens vs banked tok_refa1f: corr "
              f"min {s_arr.min():.6f} mean {s_arr.mean():.6f} (n={len(s_arr)})",
              flush=True)
        if s_arr.min() < 0.99:
            raise SystemExit("[p2] frame-f tokens DIVERGE from the banked "
                             "refa1f cache - transform drift, refusing")

    # ---- collapse check (the free check, computed not archived) ------------
    collapse = {}
    for arm in arms:
        Z = torch.stack(out[arm]).float()                  # [n, 2048]
        collapse[arm] = {
            "per_dim_std_mean": round(float(Z.std(0).mean()), 6),
            "per_dim_std_min": round(float(Z.std(0).min()), 6),
            "frac_dims_std_lt_1e-3": round(
                float((Z.std(0) < 1e-3).float().mean()), 6),
            "global_mean_abs": round(float(Z.abs().mean()), 6)}
        print(f"[p2] adapter_dim_std {arm:>14s}: "
              f"{collapse[arm]['per_dim_std_mean']:.6f} "
              f"(min {collapse[arm]['per_dim_std_min']:.6f}, "
              f"dead dims {collapse[arm]['frac_dims_std_lt_1e-3']:.4f})",
              flush=True)

    # ---- bank one ladder-cells cache per arm -------------------------------
    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    key = [(r["clip_id"], int(r["frame_idx"])) for r in rows]
    for arm in arms:
        rws = []
        for i, r in enumerate(rows):
            rr = {k: v for k, v in r.items() if k not in ("tokens", "cells")}
            rr["cells"] = out[arm][i]
            rws.append(rr)
        meta = dict(src_meta)
        meta.update({
            "_evidence_class": "MEASURED (ours; TRAINED refa-dinov2-4b ckpt "
                               "forward on the SAME banked windows; features "
                               "are the refa1f EMULATION - residual declared)",
            "eval_tier": "T0-DIAGNOSTIC",
            "part2_arm": arm,
            "ckpt_path": str(a.ckpt),
            "ckpt_sha256": ck_sha,
            "ckpt_bytes": n_ck,
            "ckpt_step": step,
            "ckpt_hf_repo": "Sayood/tanitad-refa-dinov2-4b",
            "model_recipe": "taniteval loaders.py refa-plus (flagship4b_config,"
                            " adim 3, temporal adapter, STRICT load)",
            "predictor_call": "model.predictor(states, actions, intent=None) - "
                              "the T0 eval surface (rollout.py:182)",
            "window": WINDOW,
            "action_channels": ["steer_road_rad", "accel_mps2",
                                "v0/SPEED_SCALE(10.0)"],
            "n_pad_rows": n_pad_rows,
            "n_cells": 16,
            "d_model_tokens": 768,
            "token_grid": [16, 16],
            "tokens_banked": False,
            "cells_present": True,
            "state_dim": int(model.state_dim),
            "collapse_check": collapse[arm],
            "window_identity": {"copied_from": str(a.row_index),
                                "n_rows": len(rows),
                                "first": list(key[0]), "last": list(key[-1])},
            "run_stamp": f"refa-dinov2-4b@{step}/{arm} "
                         f"(windows: {src_meta.get('run_stamp')})",
            "cuda_max_mem_gb": peak,
            "wall_s": round(time.time() - t0, 1)})
        torch.save({"rows": rws, "meta": meta}, outdir / f"cells_{arm}.pt")
        print(f"[p2] banked {outdir / f'cells_{arm}.pt'}", flush=True)

    (outdir / "part2_build_meta.json").write_text(json.dumps({
        "ckpt_sha256": ck_sha, "ckpt_bytes": n_ck, "ckpt_step": step,
        "collapse_check": collapse, "n_pad_rows": n_pad_rows,
        "sanity_corr_min": (float(np.min(sanity)) if sanity else None),
        "sanity_n": len(sanity),
        "wall_s": round(time.time() - t0, 1), "cuda_max_mem_gb": peak},
        indent=1), "utf-8")
    print(f"[p2] DONE {len(rows)} rows, {len(arms)} arms, "
          f"{time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
