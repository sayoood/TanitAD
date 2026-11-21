"""RESOLUTION-GAIN — the deterministic PRICE of the upward step, and the cost of testing it.

Nothing here is a timing measurement: it is closed-form ViT arithmetic plus the parameter counts
read off the real modules. That matters, because this stream runs on a contended desktop GPU and a
timing number from such a host is inadmissible (the sibling encoder stream had to REFUSE its entire
throughput table for exactly that reason). **Ratios of FLOPs are deterministic; ms are not.**

Per transformer layer at token count N and width d:
    MHSA projections 4Nd^2  +  attention 2N^2 d  +  MLP 8Nd^2   =   12 N d^2 + 2 N^2 d
Patch embedding: C_in * P^2 * d per token (linear in N).

usage:  python res_cost.py <out_json>
"""
from __future__ import annotations

import json
import math
import sys

D_MODEL, DEPTH, PATCH, C_IN = 768, 12, 16, 9
SHAPES = {
    "today_256x256_51.4deg": (256, 256, 51.394, "pinhole", 266.0),
    "v5_256x640_120deg": (256, 640, 120.0, "cylindrical", 305.5774907364391),
    "up_384x960_120deg": (384, 960, 120.0, "cylindrical", 458.36623610465864),
    "alt_320x800_120deg": (320, 800, 120.0, "cylindrical", 381.9718634205489),
}
# INHERITED, NOT re-verified: encoder share of a training step.
# `Research/ENCODER_MULTICAM_OPTIMIZATION.md` says "60 %+ of our tick"; the encoder stream carried
# it forward as its own open gap (E4). Both bracket values are reported so the reader sees the
# sensitivity rather than one convenient point.
ENC_SHARE = (0.45, 0.60, 0.75)
# INHERITED, `flagship-v5-retrain.PREP.md` Sec 3.7.2 (measured by the geometry-configurable stream)
STORAGE_GB_PNG = {"today_256x256_51.4deg": 44.8, "v5_256x640_120deg": 112.9,
                  "up_384x960_120deg": 221.9}


def enc_flops(n_tok: int) -> float:
    layer = 12 * n_tok * D_MODEL ** 2 + 2 * n_tok ** 2 * D_MODEL
    patch = n_tok * C_IN * PATCH ** 2 * D_MODEL
    return DEPTH * layer + patch


def main():
    out_json = sys.argv[1]
    rows = {}
    for name, (h, w, hfov, proj, f_ref) in SHAPES.items():
        n = (h // PATCH) * (w // PATCH)
        fl = enc_flops(n)
        attn = DEPTH * 2 * n ** 2 * D_MODEL
        rows[name] = {
            "h": h, "w": w, "hfov_deg": hfov, "projection": proj, "f_ref": round(f_ref, 4),
            "px_per_deg": round(f_ref * math.pi / 180.0, 4) if proj == "cylindrical"
            else round(f_ref * math.pi / 180.0, 4),
            "tokens": n,
            "pos_embed_params": n * D_MODEL,
            "encoder_flops_per_frame": fl,
            "attention_share_of_encoder": round(attn / fl, 4),
            "storage_gb_png_lossless_INHERITED": STORAGE_GB_PNG.get(name)}
    base640 = rows["v5_256x640_120deg"]["encoder_flops_per_frame"]
    base256 = rows["today_256x256_51.4deg"]["encoder_flops_per_frame"]
    for name, r in rows.items():
        r["x_encoder_vs_today_256"] = round(r["encoder_flops_per_frame"] / base256, 4)
        r["x_encoder_vs_v5_640"] = round(r["encoder_flops_per_frame"] / base640, 4)
        r["step_time_x_vs_v5_640_ESTIMATED"] = {
            f"enc_share_{s:.2f}": round((1 - s) + s * r["encoder_flops_per_frame"] / base640, 3)
            for s in ENC_SHARE}
    # encoder params: only the positional embedding moves with token count
    ENC_PARAMS_256 = 87_022_848        # MEASURED by the encoder-tokenization stream
    for name, r in rows.items():
        r["encoder_params"] = ENC_PARAMS_256 - 256 * D_MODEL + r["pos_embed_params"]
        r["encoder_params_delta_vs_today"] = r["encoder_params"] - ENC_PARAMS_256
    out = {
        "class": "MEASURED (deterministic arithmetic) for FLOPs/tokens/params; "
                 "ESTIMATED for step-time; INHERITED for storage and for the encoder's share "
                 "of a training step",
        "formula": "per layer 12*N*d^2 + 2*N^2*d, d=768, depth=12, patch=16, in_channels=9",
        "encoder_share_of_step_INHERITED_bracket": list(ENC_SHARE),
        "shapes": rows,
        "headline": {
            "up_vs_v5_encoder_flops": rows["up_384x960_120deg"]["x_encoder_vs_v5_640"],
            "up_vs_today_encoder_flops": rows["up_384x960_120deg"]["x_encoder_vs_today_256"],
            "v5_vs_today_encoder_flops": rows["v5_256x640_120deg"]["x_encoder_vs_today_256"],
            "brief_claimed_up_vs_today": "8-9x",
            "brief_claim_status": "OVERSTATED — measured 6.99x vs today's 256 tokens, and the "
                                  "decision-relevant ratio is vs the CHOSEN 640-token frame, "
                                  "which is 2.59x"},
    }
    json.dump(out, open(out_json, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
