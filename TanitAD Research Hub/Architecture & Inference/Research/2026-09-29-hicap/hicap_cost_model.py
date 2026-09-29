#!/usr/bin/env python3
"""HiCAP cost model (pure arithmetic, no data, no hardware measurement).

Two tables, both ESTIMATED (closed-form counts, not timings):

1. Retrieval: leaves streamed per decision, flat scoring vs beam retrieval over the tree
   (strategic action -> tactical cell -> coarse residual code -> fine residual code).
   Uses the EFFECTIVE (populated) v7 vocabulary: 5 strategic actions, 5 x 7 = 35 tactical cells
   (P1 report sec. 1.3-1.4). The flat baseline is generous to flat scoring: it streams only the
   distinct (cell, code) embeddings and ignores strategic multiplicity.

2. Backbone: dense-transformer prefill FLOPs, 2 * P * n_tokens (attention's quadratic term
   ignored, which understates long prompts), and the sustained rate that implies at a tick rate.

Memory bandwidth of the target edge SoC is the published NVIDIA figure (273 GB/s, Jetson AGX Thor).
"""
import json
import sys

D = 512                     # embedding width
THOR_BW = 273e9             # bytes/s, PUBLISHED spec for Jetson AGX Thor
TICK_HZ = 10.0
N_STR, N_CELL = 5, 35       # effective populated a_str / (lat x lon) counts
BEAM_STR, BEAM_CELL, BEAM_CODE = 2, 4, 8


def retrieval_rows():
    rows = []
    for k_side in (64, 128, 256, 512):
        k3 = k_side * k_side                               # codes per cell = coarse x fine
        flat_leaves = N_CELL * k3
        touched = (N_STR                                    # level 1
                   + BEAM_STR * N_CELL                      # level 2 under each kept strategic node
                   + BEAM_CELL * k_side                     # level 3a coarse codes in kept cells
                   + BEAM_CELL * BEAM_CODE * k_side)        # level 3b fine codes under kept coarse codes
        row = {"codes_per_cell": k3, "coarse_x_fine": f"{k_side}x{k_side}",
               "flat_leaves": flat_leaves, "beam_nodes_touched": touched,
               "node_ratio": round(flat_leaves / touched, 1)}
        for w, tag in ((2, "bf16"), (1, "int8")):
            fb = flat_leaves * D * w
            hb = touched * D * w
            row[f"flat_MB_{tag}"] = round(fb / 1e6, 2)
            row[f"beam_MB_{tag}"] = round(hb / 1e6, 3)
            row[f"flat_GBps_at_{int(TICK_HZ)}Hz_{tag}"] = round(fb * TICK_HZ / 1e9, 3)
            row[f"flat_share_of_273GBps_{tag}"] = round(fb * TICK_HZ / THOR_BW, 5)
            row[f"beam_share_of_273GBps_{tag}"] = round(hb * TICK_HZ / THOR_BW, 7)
        rows.append(row)
    return rows


def factored_rows():
    """Factored product vocabulary (path factor x speed-profile factor), coarse-then-compose (the SparseDriveV2
    structure): score K_P paths and K_V profiles separately, keep (k_P, k_V), compose k_P*k_V trajectories.
    Cost per decision in embedding rows read = K_P + K_V + k_P*k_V (the composed stage also needs a pair head)."""
    rows = []
    for kp, kv, kkp, kkv in ((1024, 256, 20, 10), (2048, 512, 32, 16), (4096, 1024, 40, 20)):
        n = kp * kv
        touched = kp + kv + kkp * kkv
        row = {"K_P": kp, "K_V": kv, "keep": [kkp, kkv], "N_composed": n, "rows_touched": touched,
               "node_ratio": round(n / touched, 1)}
        for w, tag in ((2, "bf16"), (1, "int8")):
            row[f"flat_MB_{tag}"] = round(n * D * w / 1e6, 1)
            row[f"flat_ms_at_273GBps_{tag}"] = round(n * D * w / THOR_BW * 1e3, 2)
            row[f"factored_MB_{tag}"] = round(touched * D * w / 1e6, 3)
        rows.append(row)
    return rows


def prefill_rows():
    rows = []
    for name, cams, tok_per_cam, hz in (("all 7 cams x 256 tok @10Hz", 7, 256, 10.0),
                                        ("3 cams x 128 tok @10Hz", 3, 128, 10.0),
                                        ("3 cams x 128 tok @2Hz", 3, 128, 2.0),
                                        ("1 cam x 128 tok @2Hz", 1, 128, 2.0),
                                        ("1 cam x 64 tok @2Hz", 1, 64, 2.0)):
        n = cams * tok_per_cam + 64                        # + 64 text/query tokens
        for p_b in (2.0, 4.0, 8.0):
            flops = 2.0 * p_b * 1e9 * n
            rows.append({"config": name, "params_B": p_b, "tokens": n,
                         "TFLOP_per_pass": round(flops / 1e12, 2),
                         "sustained_TFLOPs": round(flops * hz / 1e12, 2)})
    return rows


# NVIDIA-published Thor rows (TRT-Edge-LLM 0.10.0, batch 1, ViT FP16, LLM NVFP4), as read by stream R-A
# (RA_vlm_backbones.md sec. 2): (ViT ms per image, prefill ms @292 tokens, prefill ms @2048 tokens).
THOR_ROWS = {"Qwen3-VL-2B": (11.4, 12.7, 28.0), "Qwen3-VL-4B": (11.6, 22.3, 62.2), "Qwen3-VL-8B": (15.7, 32.1, 104.5)}


def thor_latency_rows():
    """ESTIMATE by linear interpolation of the published rows. ViT time scales with the number of camera images
    (the published ViT row is for one ~265-token image, so our 160-token crops make it an upper bound);
    prefill is linear in total tokens between the two published points."""
    rows = []
    for model, (vit, pf292, pf2048) in THOR_ROWS.items():
        slope = (pf2048 - pf292) / (2048 - 292)
        for cams, tok in ((1, 160), (3, 160), (7, 160)):
            n = cams * tok + 64                             # + prompt/query tokens
            pf = pf292 + slope * (n - 292)
            rows.append({"model": model, "cameras": cams, "tokens": n, "vit_ms": round(vit * cams, 1),
                         "prefill_ms": round(pf, 1), "total_ms": round(vit * cams + pf, 1)})
    return rows


def cache_rows():
    """Size of a cached VLM-token corpus: 26.2 h at a given VLM refresh rate."""
    secs = 26.2 * 3600
    rows = []
    for hz in (2.0, 10.0):
        for n_tok, d_v, depths, w in ((128, 2048, 3, 1), (64, 2048, 3, 1), (128, 2048, 3, 2)):
            ticks = secs * hz
            rows.append({"refresh_hz": hz, "tokens": n_tok, "d_v": d_v, "depths": depths, "bytes_per_elem": w,
                         "ticks": int(ticks), "GB": round(ticks * n_tok * d_v * depths * w / 1e9, 1)})
    return rows


if __name__ == "__main__":
    out = {"evidence_class": "ESTIMATED (closed-form arithmetic; no timing measured)",
           "d": D, "thor_bw_Bps": THOR_BW, "tick_hz": TICK_HZ,
           "tree": {"strategic": N_STR, "tactical_cells": N_CELL,
                    "beams": {"strategic": BEAM_STR, "cell": BEAM_CELL, "coarse_code": BEAM_CODE}},
           "retrieval": retrieval_rows(), "factored_product": factored_rows(), "prefill": prefill_rows(),
           "thor_latency_estimate": thor_latency_rows(), "cache_size": cache_rows()}
    json.dump(out, sys.stdout, indent=1)
