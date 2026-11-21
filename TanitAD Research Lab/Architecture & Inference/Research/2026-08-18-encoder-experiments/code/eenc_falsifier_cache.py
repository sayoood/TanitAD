"""THE C104 FALSIFIER BATTERY — token caches for the controls the DINOv2
headline does NOT yet carry.

⛔ WHY THIS EXISTS. C104's headline is a comparison of TWO THINGS THAT DIFFER IN
MORE THAN ONE WAY, and the conclusion drawn from it — *"the encoder/objective is
the constraint"* — is only one of the readings the comparison admits:

  | axis                | ours (v6F S-W @11250)      | DINOv2-B/14 arm            |
  |---------------------|----------------------------|----------------------------|
  | pretraining         | our S-W objective, 2376 eps| LVD-142M self-supervised   |
  | ARCHITECTURE        | ViT-5 (RMS/QK/LS/RoPE) /16 | vanilla ViT /14            |
  | ⛔ INPUT FORMAT      | ONE 9-channel tensor       | THREE 3-channel sub-frames |
  | ⛔ TOKEN WIDTH       | 768                        | 2304 (3 x 768 concat)      |

The last two are NOT the encoder. A 3-view concatenation makes inter-frame
differences available to a LINEAR readout by construction; a single fused
9-channel patch conv does not. ⇒ *"the encoder is the constraint"* is one night
old, rests on a single external encoder, and has three live alternatives. This
builds the caches that separate them — ZERO TRAINING, same windows, same rows,
same pool, same ladder.

THE FOUR MODES, and the exact claim each can kill
  randenc   OUR OWN ViT5Encoder ARCHITECTURE at the LIVE geometry, RANDOMLY
            INITIALISED, never trained. Width 768, input format ours.
            ⇒ if random-ours ~= trained-ours, our S-W objective added nothing
              linearly readable on these rungs (C104's reading SURVIVES and
              sharpens to "the OBJECTIVE").
            ⇒ if random-ours >> trained-ours, S-W REMOVED readable geometry —
              a much sharper and different claim.
            ⇒ if random-ours ~= DINOv2, the gap is NOT pretraining at all and
              C104's headline reading is REFUTED.
  dinorand  DINOv2's ARCHITECTURE, randomly initialised, 3 sub-frames, width
            2304 — the headline arm with ONLY the pretrained weights removed.
            ⇒ isolates PRETRAINING from ARCHITECTURE+INPUT-FORMAT.
  dino1f    Pretrained DINOv2 on ONE sub-frame -> width 768, EXACTLY ours.
            ⇒ isolates the 3-view CONCATENATION.
            ⚠️ Read `lead_gap` here, not `ego_v0`/`lead_closing`: gap is STATIC
              geometry and one frame suffices; the motion rungs cannot be read
              from one frame by anyone, so a drop there is uninformative.
  ours      re-emit the banked v6 tokens unchanged (an identity pass-through
            used only to prove this harness reproduces the banked arm).

⛔ PARITY. NO episode is selected here. The row set is COPIED from the banked v6
token cache — same clips, same frame indices, same order, same targets, same
split — and ONLY `tokens` is replaced. The (clip_id, frame_idx) sequence is
asserted equal to the source. `physicalai-train-e438721ae894` is untouched: this
reads a frozen checkpoint's window cache, it does not build a corpus.
"""
from __future__ import annotations

import pyarrow  # noqa: F401  # isort: skip

import truststore  # isort: skip

truststore.inject_into_ssl()      # certifi fails behind this box's TLS proxy

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[5]
for _p in (_REPO / "stack",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODES = ("randenc", "trained", "dinorand", "dino1f", "dino3f", "ours")


def hf_token() -> str | None:
    p = _REPO / "Keys.txt"
    if not p.exists():
        return None
    m = re.findall(r"hf_[A-Za-z0-9]+", p.read_text("utf-8", "ignore"))
    return m[0] if m else None


# --------------------------------------------------------------------------- #
def build_ours_encoder(ckpt: str, seed: int, device, trained: bool = False):
    """OUR ViT5Encoder at the LIVE encoder config.

    ``trained=False`` → RANDOMLY INITIALISED (the step-0 control).
    ``trained=True``  → the checkpoint's OWN encoder weights (a step point on
    the S-W trajectory). ⛔ The load is STRICT and the loaded key count is
    reported: an encoder silently left at random init would look exactly like
    the control and quietly turn the trend into a flat line.

    ⛔ The config is READ FROM THE CHECKPOINT, never retyped: a control that is
    'our architecture' but at a remembered geometry is not a control."""
    from tanitad.config import EncoderConfig
    from tanitad.models.encoder import ViT5Encoder
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    vc = ck["_meta"]["config"]["v6_config"]
    enc_cfg = EncoderConfig(**vc["encoder"])
    n_reg = int(vc.get("n_registers", 4))
    vit5 = bool(vc.get("vit5_encoder", True))
    if not vit5:
        raise SystemExit("[fals] live config is not vit5_encoder — refusing to "
                         "build a control that is not the live architecture")
    torch.manual_seed(int(seed))
    enc = ViT5Encoder(enc_cfg, n_registers=n_reg)
    loaded = None
    if trained:
        sd = {k[len("encoder."):]: v for k, v in ck["model"].items()
              if k.startswith("encoder.")}
        if not sd:
            raise SystemExit("[fals] ⛔ no `encoder.*` keys in the checkpoint")
        missing, unexpected = enc.load_state_dict(sd, strict=False)
        # rope_cos/rope_sin are persistent=False buffers ⇒ legitimately absent
        fatal = [k for k in missing if not k.startswith("rope_")]
        if fatal or unexpected:
            raise SystemExit(f"[fals] ⛔ encoder load mismatch: missing={fatal[:6]} "
                             f"unexpected={sorted(unexpected)[:6]}")
        loaded = {"n_keys_loaded": len(sd), "missing_nonrope": fatal,
                  "unexpected": sorted(unexpected),
                  "step": int(ck["_meta"].get("step", -1))}
    enc = enc.to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in enc.parameters())
    return enc, {"encoder_class": "ViT5Encoder",
                 "init": ("TRAINED (checkpoint weights)" if trained
                          else "RANDOM (untrained)"),
                 "loaded": loaded,
                 "init_seed": int(seed), "params": int(n),
                 "d_model": int(enc_cfg.d_model), "depth": int(enc_cfg.depth),
                 "n_heads": int(enc_cfg.n_heads), "n_registers": n_reg,
                 "patch": int(enc_cfg.patch_size),
                 "in_channels": int(enc_cfg.in_channels),
                 "image_hw": list(enc_cfg.image_hw()),
                 "config_source": "the LIVE checkpoint's v6_config.encoder"}


def build_dino(model_id: str, pretrained: bool, th: int, tw: int, device):
    from transformers import AutoConfig, AutoModel
    tok = hf_token()
    cfg = AutoConfig.from_pretrained(model_id, token=tok)
    patch = int(cfg.patch_size)
    H, W = th * patch, tw * patch
    if pretrained:
        m = AutoModel.from_pretrained(model_id, token=tok)
    else:
        torch.manual_seed(20260818)
        m = AutoModel.from_config(cfg)
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    n_prefix = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)
    return m, {"encoder_class": model_id,
               "init": "PRETRAINED" if pretrained else "RANDOM (untrained)",
               "params": int(sum(p.numel() for p in m.parameters())),
               "patch": patch, "hidden": int(cfg.hidden_size),
               "n_prefix_tokens": n_prefix, "external_input_hw": [H, W]}, H, W, n_prefix


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--row-index", required=True,
                    help="the token-free row index (er10_dino_cache.py --stage index)")
    ap.add_argument("--src-cache", default=None,
                    help="mode=ours only: the banked v6 token cache")
    ap.add_argument("--episodes-dir", required=True)
    ap.add_argument("--ckpt", default=None, help="mode=randenc: live ckpt for the config")
    ap.add_argument("--model", default="facebook/dinov2-base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sub-frame", type=int, default=0,
                    help="mode=dino1f: WHICH sub-frame of the D-015 stack")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lru", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-rows", type=int, default=None,
                    help="SMOKE ONLY — strided subsample, never a finding")
    a = ap.parse_args(argv)

    from tanitad.data.v2_dataset import _decode_stacked, _jpeg_offsets

    idx = torch.load(a.row_index, map_location="cpu", weights_only=False)
    rows, src_meta = idx["rows"], idx["meta"]
    if a.limit_rows:
        rows = rows[::max(1, len(rows) // int(a.limit_rows))]
    th, tw = int(src_meta["token_grid"][0]), int(src_meta["token_grid"][1])
    n_tok = th * tw
    src_h, src_w = int(src_meta["frame"]["h"]), int(src_meta["frame"]["w"])
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    enc_meta: dict = {}
    H = W = n_prefix = None
    model = None
    if a.mode in ("randenc", "trained"):
        if not a.ckpt:
            raise SystemExit(f"[fals] --ckpt is required for mode={a.mode}")
        model, enc_meta = build_ours_encoder(a.ckpt, a.seed, dev,
                                             trained=(a.mode == "trained"))
        if tuple(enc_meta["image_hw"]) != (src_h, src_w):
            raise SystemExit(f"[fals] ⛔ live encoder geometry "
                             f"{enc_meta['image_hw']} != cache frame "
                             f"{[src_h, src_w]}")
    elif a.mode in ("dinorand", "dino1f", "dino3f"):
        model, enc_meta, H, W, n_prefix = build_dino(
            a.model, a.mode != "dinorand", th, tw, dev)
        ar_src, ar_dst = src_h / src_w, H / W
        if abs(ar_src - ar_dst) > 1e-6:
            raise SystemExit(f"[fals] ⛔ ASPECT CHANGE {ar_src:.6f} -> "
                             f"{ar_dst:.6f}; refusing an anisotropic resize")
        enc_meta["aspect_ratio_src_dst"] = [round(ar_src, 6), round(ar_dst, 6)]
    elif a.mode == "ours":
        if not a.src_cache:
            raise SystemExit("[fals] --src-cache is required for mode=ours")
    print(f"[fals] mode={a.mode} {json.dumps(enc_meta, default=str)}", flush=True)

    eps = Path(a.episodes_dir)
    payload_cache: dict[str, tuple] = {}

    def payload(cid: str):
        if cid not in payload_cache:
            if len(payload_cache) >= int(a.lru):
                payload_cache.pop(next(iter(payload_cache)))
            d = torch.load(eps / f"{cid}.v2ep.pt", map_location="cpu",
                           weights_only=False)
            payload_cache[cid] = (d["jpeg_buf"], _jpeg_offsets(d["jpeg_len"]),
                                  int(d["n_stack"]), str(d.get("codec", "jpeg")))
        return payload_cache[cid]

    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    order = sorted(range(len(rows)),
                   key=lambda i: (rows[i]["clip_id"], rows[i]["frame_idx"]))
    out_tokens: list[torch.Tensor | None] = [None] * len(rows)
    t0, done = time.time(), 0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if a.mode == "ours":
        blob = torch.load(a.src_cache, map_location="cpu", weights_only=False)
        by = {(r["clip_id"], int(r["frame_idx"])): r["tokens"]
              for r in blob["rows"]}
        for i, r in enumerate(rows):
            out_tokens[i] = by[(r["clip_id"], int(r["frame_idx"]))]
        del blob, by
    else:
        for s in range(0, len(order), int(a.batch)):
            sel = order[s:s + int(a.batch)]
            stk = []
            for i in sel:
                buf, offs, n_stack, codec = payload(rows[i]["clip_id"])
                pf = int(rows[i]["frame_idx"])
                f = _decode_stacked(buf, offs, n_stack, pf, pf + 1, codec, None)
                stk.append(f[0])                          # [9, h, w] uint8
            x = torch.stack(stk).to(dev).float() / 255.0  # [B, 9, h, w]
            if x.shape[-2:] != (src_h, src_w):
                raise SystemExit(f"[fals] ⛔ decoded {tuple(x.shape[-2:])} != "
                                 f"cache meta {(src_h, src_w)}")
            b = x.shape[0]
            if a.mode in ("randenc", "trained"):
                with torch.no_grad(), torch.autocast(
                        dev.type, dtype=torch.float16,
                        enabled=(dev.type == "cuda")):
                    hs = model(x)                          # [B, 640, 768]
                hs = hs.float()
            else:
                n_sub = x.shape[1] // 3
                if a.mode == "dino1f":
                    k = int(a.sub_frame)
                    sub = x[:, 3 * k:3 * k + 3]
                    n_sub = 1
                else:
                    sub = x.reshape(b * n_sub, 3, src_h, src_w)
                sub = F.interpolate(sub, size=(H, W), mode="bilinear",
                                    align_corners=False, antialias=True)
                sub = (sub - mean) / std
                with torch.no_grad(), torch.autocast(
                        dev.type, dtype=torch.float16,
                        enabled=(dev.type == "cuda")):
                    hs = model(pixel_values=sub,
                               interpolate_pos_encoding=True).last_hidden_state
                hs = hs[:, n_prefix:].float()
                hs = hs.reshape(b, n_sub, n_tok, hs.shape[-1])
                hs = hs.permute(0, 2, 1, 3).reshape(b, n_tok, -1)
            if hs.shape[1] != n_tok:
                raise SystemExit(f"[fals] ⛔ {hs.shape[1]} tokens, expected "
                                 f"{n_tok} — the grid is NOT ours")
            hs = hs.to(torch.float16).cpu()
            for j, i in enumerate(sel):
                out_tokens[i] = hs[j].clone()
            done += b
            if (s // int(a.batch)) % 25 == 0:
                el = time.time() - t0
                print(f"[fals] {done}/{len(order)}  {el/60:.1f} min  "
                      f"{done/max(el,1e-9):.1f} row/s", flush=True)

    peak = (float(torch.cuda.max_memory_allocated()) / 1e9
            if dev.type == "cuda" else None)
    d_out = int(out_tokens[0].shape[-1])
    for i, r in enumerate(rows):
        r["tokens"] = out_tokens[i]
    key = [(r["clip_id"], int(r["frame_idx"])) for r in rows]
    meta = dict(src_meta)
    meta.update({
        "_evidence_class": "MEASURED (ours; FROZEN encoder forward on the SAME "
                           "banked windows)",
        "eval_tier": "T0-DIAGNOSTIC",
        "falsifier_mode": a.mode,
        "encoder_meta": enc_meta,
        "sub_frames_concatenated": (
            1 if a.mode in ("dino1f", "randenc", "trained", "ours") else 3),
        "sub_frame_index": (int(a.sub_frame) if a.mode == "dino1f" else None),
        "d_model_tokens": d_out,
        "n_tokens": n_tok,
        "token_grid": [th, tw],
        "tokens_banked": True,
        "cells_present": False,
        "window_identity": {"n_rows": len(rows), "first": list(key[0]),
                            "last": list(key[-1]),
                            "note": "rows copied from the banked v6 cache with "
                                    "ONLY `tokens` replaced; NO episode selected"},
        "run_stamp": f"FALSIFIER:{a.mode} (windows: {src_meta.get('run_stamp')})",
        "cuda_max_mem_gb": peak,
        "wall_s": round(time.time() - t0, 1)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rows": rows, "meta": meta}, a.out)
    (Path(a.out).parent / f"meta_{a.mode}_s{a.seed}.json").write_text(
        json.dumps(meta, indent=1, default=str), "utf-8")
    print(f"[fals] DONE {a.mode} {len(rows)} rows, d_model {d_out}, "
          f"{meta['wall_s']} s -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
