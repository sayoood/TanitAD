"""E-GEOM - build DINOv2 token caches at DIFFERENT INPUT GEOMETRIES over the
SAME banked windows.

WHY THIS EXISTS. C104 measured `facebook/dinov2-base` at 224x560 (120 deg
horizontal field, 3 sub-frames concatenated) and read `lead_gap` r2 0.44997,
against our encoder's 0.00496 - the "91x encoder gap". REF-A trained on the
SAME WEIGHTS but a DIFFERENT TENSOR: `dino_precompute.py` resizes the phase-0
256x256 f-theta-crop frame (f_eff = F_REF = 266 -> 2*atan(128/266) = 51.39 deg)
to 224x224 and keeps ONE frame, giving a 16x16 / 768-d grid.

=> 0.44997 is not a fact about the features REF-A consumed. This script varies
ONLY the input geometry and the sub-frame count, holding the encoder weights,
the windows, the pool, the projection, the ridge and the seeds fixed, so the
2x2 attributes the gap to FIELD OF VIEW vs TEMPORAL CONTENT vs the encoder.

THE ANISOTROPY GUARD IS RELAXED HERE DELIBERATELY AND LOUDLY.
`er10_dino_cache.py` REFUSES any aspect change, because for ITS question an
anisotropic resize would have been an unlogged confound. Here the aspect change
IS the treatment (REF-A really did consume a square grid), so the guard becomes
a DECLARATION: every arm records `aspect_ratio_src_dst` and `anisotropic` in its
meta, and the ladder reads them back into the result.

WINDOW IDENTITY IS PINNED, NOT ASSUMED: rows are COPIED from the banked v6
row index (same clips, same frame indices, same order) and only `tokens` is
replaced. No episode is selected; parity is untouched.
"""
from __future__ import annotations

import pyarrow  # noqa: F401  # isort: skip
import truststore  # isort: skip  # certifi fails behind this box's TLS proxy

truststore.inject_into_ssl()

import argparse
import math
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

# REF-A's phase-0 geometry, DERIVED from source, never hard-coded:
#   stack/tanitad/data/physicalai.py:140-143  CORPUS_META image_size = 256
#   stack/tanitad/data/calib.py:38            F_REF = 266.0
REFA_IMAGE_SIZE = 256.0
REFA_F_EFF_PX = 266.0
REFA_HFOV_DEG = 2.0 * math.degrees(math.atan(REFA_IMAGE_SIZE / 2.0 / REFA_F_EFF_PX))


def hf_token() -> str | None:
    p = _REPO / "Keys.txt"
    if not p.exists():
        return None
    m = re.findall(r"hf_[A-Za-z0-9]+", p.read_text("utf-8", "ignore"))
    return m[0] if m else None


def crop_cols(src_w: int, src_hfov_deg: float, want_hfov_deg: float):
    """Centre column range covering `want_hfov_deg` of a CYLINDRICAL frame.

    A cylindrical projection is LINEAR IN AZIMUTH (x = f*theta), so px/deg is
    constant and the crop is exact arithmetic, not an approximation. Returns
    (c0, c1, achieved_deg).
    """
    px_per_deg = src_w / float(src_hfov_deg)
    w = int(round(want_hfov_deg * px_per_deg))
    w = max(14, min(int(src_w), w))
    c0 = (int(src_w) - w) // 2
    return c0, c0 + w, w / px_per_deg


# arm -> (token grid, horizontal field wanted, sub-frame policy)
ARMS = {
    "wide3f":   ((16, 40), None,          "all"),
    "wide1f":   ((16, 40), None,          "latest"),
    "refa3f":   ((16, 16), REFA_HFOV_DEG, "all"),
    "refa1f":   ((16, 16), REFA_HFOV_DEG, "latest"),
    "squash1f": ((16, 16), None,          "latest"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row-index", required=True)
    ap.add_argument("--episodes-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, help=" | ".join(sorted(ARMS)))
    ap.add_argument("--model", default="facebook/dinov2-base")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lru", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-rows", type=int, default=None,
                    help="SMOKE ONLY - strided subsample, never a finding")
    a = ap.parse_args(argv)

    if a.arm not in ARMS:
        raise SystemExit("[encloc] unknown arm %s; have %s"
                         % (a.arm, sorted(ARMS)))
    (th, tw), want_hfov, subs = ARMS[a.arm]

    from tanitad.data.v2_dataset import _decode_stacked, _jpeg_offsets

    idx = torch.load(a.row_index, map_location="cpu", weights_only=False)
    rows, src_meta = idx["rows"], idx["meta"]
    if a.limit_rows:
        rows = rows[::max(1, len(rows) // int(a.limit_rows))]
    n_tok = th * tw

    tok_hf = hf_token()
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained(a.model, token=tok_hf)
    patch = int(cfg.patch_size)
    H, W = th * patch, tw * patch
    src_h = int(src_meta["frame"]["h"])
    src_w = int(src_meta["frame"]["w"])
    src_hfov = float(src_meta["frame"]["hfov_deg"])
    proj = str(src_meta["frame"].get("projection"))
    if proj != "cylindrical":
        raise SystemExit("[encloc] the angular crop assumes a CYLINDRICAL "
                         "source; meta says %s" % proj)

    if want_hfov is None:
        c0, c1, got_hfov = 0, src_w, src_hfov
    else:
        c0, c1, got_hfov = crop_cols(src_w, src_hfov, want_hfov)
    crop_h, crop_w = src_h, c1 - c0
    ar_src, ar_dst = crop_h / crop_w, H / W
    aniso = abs(ar_src - ar_dst) > 1e-6
    print("[encloc] arm=%s model=%s patch=%d -> input %dx%d (%dx%d=%d tokens)"
          % (a.arm, a.model, patch, H, W, th, tw, n_tok), flush=True)
    print("[encloc] crop cols [%d:%d] of %d => hfov %.2f deg (source %.1f); "
          "aspect %.4f -> %.4f %s; sub_frames=%s"
          % (c0, c1, src_w, got_hfov, src_hfov, ar_src, ar_dst,
             "ANISOTROPIC (declared)" if aniso else "isotropic", subs),
          flush=True)

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(a.model, token=tok_hf).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_prefix = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)

    eps = Path(a.episodes_dir)
    payload_cache: dict[str, tuple] = {}

    def payload(cid: str):
        if cid not in payload_cache:
            if len(payload_cache) >= int(a.lru):
                payload_cache.pop(next(iter(payload_cache)))
            d = torch.load(eps / ("%s.v2ep.pt" % cid), map_location="cpu",
                           weights_only=False)
            payload_cache[cid] = (d["jpeg_buf"], _jpeg_offsets(d["jpeg_len"]),
                                  int(d["n_stack"]), str(d.get("codec", "jpeg")))
        return payload_cache[cid]

    order = sorted(range(len(rows)),
                   key=lambda i: (rows[i]["clip_id"], rows[i]["frame_idx"]))
    out_tokens: list[torch.Tensor | None] = [None] * len(rows)
    t0, done = time.time(), 0
    n_sub_used, n_sub_all = None, None
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for s in range(0, len(order), int(a.batch)):
        sel = order[s:s + int(a.batch)]
        stk = []
        for i in sel:
            buf, offs, n_stack, codec = payload(rows[i]["clip_id"])
            pf = int(rows[i]["frame_idx"])
            f = _decode_stacked(buf, offs, n_stack, pf, pf + 1, codec, None)
            stk.append(f[0])                                   # [9,h,w] uint8
        x = torch.stack(stk).to(dev).float() / 255.0           # [B,9,h,w]
        if tuple(x.shape[-2:]) != (src_h, src_w):
            raise SystemExit("[encloc] decoded %s != meta %s"
                             % (tuple(x.shape[-2:]), (src_h, src_w)))
        x = x[..., :, c0:c1]                                   # ANGULAR CROP
        b = x.shape[0]
        n_sub_all = x.shape[1] // 3
        if subs == "latest":
            x = x[:, -3:]                                      # REF-A: latest RGB
        n_sub = x.shape[1] // 3
        n_sub_used = n_sub
        sub = x.reshape(b * n_sub, 3, crop_h, crop_w)
        sub = F.interpolate(sub, size=(H, W), mode="bilinear",
                            align_corners=False, antialias=True)
        sub = (sub - mean) / std
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16,
                                             enabled=(dev.type == "cuda")):
            hs = model(pixel_values=sub,
                       interpolate_pos_encoding=True).last_hidden_state
        hs = hs[:, n_prefix:].float()
        if hs.shape[1] != n_tok:
            raise SystemExit("[encloc] %d patch tokens, expected %d - the grid "
                             "is NOT this arm's" % (hs.shape[1], n_tok))
        hs = hs.reshape(b, n_sub, n_tok, hs.shape[-1])
        hs = hs.permute(0, 2, 1, 3).reshape(b, n_tok, -1)      # concat per token
        hs = hs.to(torch.float16).cpu()
        for j, i in enumerate(sel):
            out_tokens[i] = hs[j].clone()
        done += b
        if (s // int(a.batch)) % 25 == 0:
            el = time.time() - t0
            print("[encloc] %d/%d  %.1f min  %.1f row/s"
                  % (done, len(order), el / 60, done / max(el, 1e-9)), flush=True)

    peak = (float(torch.cuda.max_memory_allocated()) / 1e9
            if dev.type == "cuda" else None)
    d_out = int(out_tokens[0].shape[-1])
    for i, r in enumerate(rows):
        r["tokens"] = out_tokens[i]
    key = [(r["clip_id"], int(r["frame_idx"])) for r in rows]
    meta = dict(src_meta)
    meta.update({
        "_evidence_class": "MEASURED (ours; FROZEN EXTERNAL encoder forward, "
                           "GEOMETRY VARIED, same banked windows)",
        "eval_tier": "T0-DIAGNOSTIC",
        "encloc_arm": a.arm,
        "external_encoder": a.model,
        "external_encoder_patch": patch,
        "external_encoder_hidden": int(cfg.hidden_size),
        "external_encoder_n_prefix_tokens": n_prefix,
        "external_input_hw": [H, W],
        "source_frame_hw": [src_h, src_w],
        "source_projection": proj,
        "crop_cols": [c0, c1],
        "crop_hw": [crop_h, crop_w],
        "source_hfov_deg": round(src_hfov, 4),
        "achieved_hfov_deg": round(got_hfov, 4),
        "refa_hfov_deg_derived": round(REFA_HFOV_DEG, 4),
        "refa_hfov_source": "physicalai.py CORPUS_META image_size=256 / "
                            "calib.py F_REF=266.0 -> 2*atan(128/266)",
        "aspect_ratio_src_dst": [round(ar_src, 6), round(ar_dst, 6)],
        "anisotropic": bool(aniso),
        "resize": ("bilinear+antialias, ANISOTROPIC (DECLARED: the square grid "
                   "IS the treatment)" if aniso else
                   "bilinear+antialias, isotropic"),
        "normalisation": "ImageNet mean/std (the encoder's own)",
        "sub_frames_available": int(n_sub_all),
        "sub_frames_concatenated": int(n_sub_used),
        "d_model_tokens": d_out,
        "n_tokens": n_tok,
        "token_grid": [th, tw],
        "tokens_banked": True,
        "cells_present": False,
        "window_identity": {"copied_from": str(a.row_index), "n_rows": len(rows),
                            "first": list(key[0]), "last": list(key[-1])},
        "run_stamp": "%s@frozen/%s (windows: %s)"
                     % (a.model, a.arm, src_meta.get("run_stamp")),
        "cuda_max_mem_gb": peak,
        "wall_s": round(time.time() - t0, 1)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rows": rows, "meta": meta}, a.out)
    print("[encloc] DONE %s: %d rows, d_model %d, %s s -> %s"
          % (a.arm, len(rows), d_out, meta["wall_s"], a.out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
