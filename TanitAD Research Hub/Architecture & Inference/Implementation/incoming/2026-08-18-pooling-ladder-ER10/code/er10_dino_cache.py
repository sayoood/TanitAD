"""JOB 2 — THE CORPUS-NARROWNESS DISCRIMINATOR (POOLING_BOTTLENECK_R1R2.md §8.1).

⭐ WHAT IT SETTLES, AND WHY IT OUTRANKS ANY FURTHER VARIANT OF E-R1-0.
E-R1-0 alone cannot interpret its own FAILURE. If our pre-pool tokens do not
carry relative motion, two hypotheses remain indistinguishable:
  (a) our OBJECTIVE never asked the encoder to encode it  -> R1 is warranted;
  (b) the information is not linearly present in OUR IMAGES at all -> R1, R2 and
      R3 die together and the follow-on is the corpus, not the loss.
Running the IDENTICAL ladder on a FROZEN HIGH-DIVERSITY EXTERNAL encoder over
the SAME windows separates them, with NO training whatsoever.

⛔ THE SUBSTITUTION, AND EXACTLY WHAT IT COSTS — DECLARED, NOT SILENT.
The spec names DINOv3. MEASURED here (three independent probes, `raw/
dino_availability.json` + this file's own load):
  probe 1  `HfApi.model_info`      on facebook/dinov3-vitb16-... -> OK
  probe 2  `HfApi.list_repo_files` on the same repo             -> OK
  probe 3  an actual weight/config FETCH                        -> ⛔ 403,
           "You are trying to access a gated repo" (`gated: manual`).
⇒ the METADATA is public and the WEIGHTS ARE NOT, which is precisely the shape
that makes a one-probe absence claim wrong in EITHER direction. Accepting the
licence is a human action on huggingface.co and is NOT mine to take.

⇒ SUBSTITUTE: **DINOv2 ViT-B/14 (`facebook/dinov2-base`), ungated**, at
**224x560**, which tiles at patch 14 into EXACTLY **16 x 40 = 640 tokens** — the
IDENTICAL grid our encoder produces at 256x640 / patch 16. The aspect ratio is
identical to 4 decimal places (0.4000 both), so the resize is a pure isotropic
0.875x downscale: the median lead's 37.8 px becomes 33.1 px, i.e.
**2.36 patches instead of 2.40** — the object's size IN PATCHES, which is what
the pooling argument is about, is preserved to within 2 %.

WHAT THE SUBSTITUTION COSTS, stated per axis:
  * corpus: LVD-142M (142 M curated images) instead of LVD-1689M (1.689 B).
    ⭐ Still ~60 000x our 2 376 driving episodes and still multi-domain, so it
    serves the diversity argument — but it is a WEAKER instance of it, and a
    NEGATIVE result is therefore weaker evidence than it would be from DINOv3.
  * architecture: no RoPE, no register tokens, patch 14 not 16.
  * ⇒ the RIGHT-COLUMN verdict of §8.1's 2x2 ("the information is not present in
    our images at all") may be quoted ONLY as "not present to DINOv2-B/14".

⚠️ AND THE LIMIT THE SPEC ITSELF DECLARES, which survives the substitution:
DINOv2 is IMAGE-trained, so per-token relative motion is available to a LINEAR
readout only through the 3-sub-frame concatenation below. A NEGATIVE on the
relative-motion rungs is weaker evidence than a positive one.

CONSTRUCTION
  Our encoder input is ONE 9-channel tensor = the D-015 3-frame stack
  (`physicalai.py:19`). DINOv2 takes 3 channels, so each sub-frame is encoded
  separately and the three token sets are CONCATENATED PER TOKEN
  ([640, 3*768] = [640, 2304]) — which is what makes relative motion linearly
  available, exactly as §8.1 specifies.

⛔ PARITY / WINDOW IDENTITY IS PINNED, NOT ASSUMED. The row set is COPIED from
the banked v6 token cache (`cache_tok11250`) — same clips, same frame indices,
same order, same targets, same split — and only ``tokens`` is replaced. The
(clip_id, frame_idx) sequence is asserted equal. NO episode is selected.
"""
from __future__ import annotations

import pyarrow  # noqa: F401  # isort: skip

# ⚠️ certifi fails behind this box's TLS proxy — the HF fetch dies with an
# unhelpful "Can't load the model for ..." that reads like a gated-repo error.
# MEASURED here: without this line the DINOv2 pull fails; with it, it succeeds.
import truststore  # isort: skip

truststore.inject_into_ssl()

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[6]
for _p in (_REPO / "stack",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def hf_token() -> str | None:
    p = _REPO / "Keys.txt"
    if not p.exists():
        return None
    m = re.findall(r"hf_[A-Za-z0-9]+", p.read_text("utf-8", "ignore"))
    return m[0] if m else None


# ---------------------------------------------------------------------------
def stage_index(a) -> int:
    """Pass 1 — extract the ROW INDEX from the banked cache and drop the 2.8 GB
    of v6 tokens, so pass 2 never holds both caches in RAM at once."""
    blob = torch.load(a.cache, map_location="cpu", weights_only=False)
    rows, meta = blob["rows"], blob["meta"]
    for r in rows:
        r.pop("tokens", None)
        r.pop("cells", None)
    torch.save({"rows": rows, "meta": meta}, a.row_index)
    print(f"[dino] row index: {len(rows)} rows -> {a.row_index}", flush=True)
    return 0


# ---------------------------------------------------------------------------
def stage_build(a) -> int:
    from tanitad.data.v2_dataset import _decode_stacked, _jpeg_offsets

    idx = torch.load(a.row_index, map_location="cpu", weights_only=False)
    rows, src_meta = idx["rows"], idx["meta"]
    if a.limit_rows:                       # ⚠️ SMOKE ONLY — never a finding
        rows = rows[::max(1, len(rows) // int(a.limit_rows))]
    th, tw = int(src_meta["token_grid"][0]), int(src_meta["token_grid"][1])
    n_tok = th * tw
    # ⭐ the resize that makes the foreign grid EQUAL ours, derived from the
    # patch size, never hard-coded to a remembered number.
    tok = hf_token()
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained(a.model, token=tok)
    patch = int(cfg.patch_size)
    H, W = th * patch, tw * patch
    print(f"[dino] {a.model}: patch {patch}, hidden {cfg.hidden_size} -> "
          f"input {H}x{W} for a {th}x{tw} = {n_tok}-token grid", flush=True)
    src_h, src_w = int(src_meta["frame"]["h"]), int(src_meta["frame"]["w"])
    ar_src, ar_dst = src_h / src_w, H / W
    if abs(ar_src - ar_dst) > 1e-6:
        raise SystemExit(f"[dino] ⛔ ASPECT CHANGE {ar_src:.6f} -> {ar_dst:.6f}: "
                         f"an anisotropic resize would distort the geometry the "
                         f"pooling argument is about. Refusing.")

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(a.model, token=tok).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_reg = int(getattr(cfg, "num_register_tokens", 0) or 0)
    n_prefix = 1 + n_reg                       # CLS (+ registers)
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)

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

    order = sorted(range(len(rows)),
                   key=lambda i: (rows[i]["clip_id"], rows[i]["frame_idx"]))
    out_tokens: list[torch.Tensor | None] = [None] * len(rows)
    t0, done, peak_ck = time.time(), 0, None
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for s in range(0, len(order), int(a.batch)):
        sel = order[s:s + int(a.batch)]
        stk = []
        for i in sel:
            buf, offs, n_stack, codec = payload(rows[i]["clip_id"])
            pf = int(rows[i]["frame_idx"])
            f = _decode_stacked(buf, offs, n_stack, pf, pf + 1, codec, None)
            stk.append(f[0])                              # [9, h, w] uint8
        x = torch.stack(stk).to(dev).float() / 255.0      # [B, 9, h, w]
        if x.shape[-2:] != (src_h, src_w):
            raise SystemExit(f"[dino] ⛔ decoded frame {tuple(x.shape[-2:])} "
                             f"!= cache meta {(src_h, src_w)}")
        b = x.shape[0]
        n_sub = x.shape[1] // 3
        sub = x.reshape(b * n_sub, 3, src_h, src_w)
        sub = F.interpolate(sub, size=(H, W), mode="bilinear",
                            align_corners=False, antialias=True)
        sub = (sub - mean) / std
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16,
                                             enabled=(dev.type == "cuda")):
            hs = model(pixel_values=sub,
                       interpolate_pos_encoding=True).last_hidden_state
        hs = hs[:, n_prefix:].float()                     # [B*n_sub, 640, D]
        if hs.shape[1] != n_tok:
            raise SystemExit(f"[dino] ⛔ {hs.shape[1]} patch tokens, expected "
                             f"{n_tok} — the grid is NOT ours")
        hs = hs.reshape(b, n_sub, n_tok, hs.shape[-1])
        hs = hs.permute(0, 2, 1, 3).reshape(b, n_tok, -1)  # concat per token
        hs = hs.to(torch.float16).cpu()
        for j, i in enumerate(sel):
            out_tokens[i] = hs[j].clone()
        done += b
        if (s // int(a.batch)) % 20 == 0:
            el = time.time() - t0
            print(f"[dino] {done}/{len(order)}  {el/60:.1f} min  "
                  f"{done/max(el,1e-9):.1f} row/s", flush=True)
    if dev.type == "cuda":
        peak_ck = float(torch.cuda.max_memory_allocated()) / 1e9

    d_out = int(out_tokens[0].shape[-1])
    for i, r in enumerate(rows):
        r["tokens"] = out_tokens[i]
    # ⛔ WINDOW-IDENTITY PIN: the row sequence must be the banked one, verbatim.
    key = [(r["clip_id"], int(r["frame_idx"])) for r in rows]
    meta = dict(src_meta)
    meta.update({
        "_evidence_class": "MEASURED (ours; FROZEN EXTERNAL encoder forward on "
                           "the SAME banked windows)",
        "eval_tier": "T0-DIAGNOSTIC",
        "external_encoder": a.model,
        "external_encoder_patch": patch,
        "external_encoder_hidden": int(cfg.hidden_size),
        "external_encoder_n_prefix_tokens": n_prefix,
        "external_input_hw": [H, W],
        "source_frame_hw": [src_h, src_w],
        "aspect_ratio_src_dst": [round(ar_src, 6), round(ar_dst, 6)],
        "resize": "bilinear+antialias, ISOTROPIC (aspect asserted equal)",
        "normalisation": "ImageNet mean/std (the encoder's own)",
        "sub_frames_concatenated": n_sub,
        "d_model_tokens": d_out,
        "n_tokens": n_tok,
        "token_grid": [th, tw],
        "tokens_banked": True,
        "cells_present": False,
        "window_identity": {
            "copied_from": str(a.cache),
            "n_rows": len(rows),
            "first": list(key[0]), "last": list(key[-1]),
            "note": "rows are the banked v6 cache's rows with ONLY `tokens` "
                    "replaced; `cells` is deliberately ABSENT so an accidental "
                    "--arms cells raises instead of reading v6 numbers"},
        "run_stamp": f"{a.model}@frozen (windows: {src_meta.get('run_stamp')})",
        "cuda_max_mem_gb": peak_ck,
        "wall_s": round(time.time() - t0, 1)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rows": rows, "meta": meta}, a.out)
    (Path(a.out).parent / "dino_meta.json").write_text(
        json.dumps(meta, indent=1, default=str), "utf-8")
    print(f"[dino] DONE {len(rows)} rows, d_model {d_out}, "
          f"{meta['wall_s']} s -> {a.out}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["index", "build"], required=True)
    ap.add_argument("--cache", required=True, help="the banked v6 TOKEN cache")
    ap.add_argument("--row-index", required=True)
    ap.add_argument("--episodes-dir")
    ap.add_argument("--out")
    ap.add_argument("--model", default="facebook/dinov2-base")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lru", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-rows", type=int, default=None,
                    help="SMOKE ONLY — strided subsample, never a finding")
    a = ap.parse_args(argv)
    return stage_index(a) if a.stage == "index" else stage_build(a)


if __name__ == "__main__":
    sys.exit(main())
