"""What does APPROACH A actually cost? — MEASURE it, do not estimate it.

Approach A = make the tokens the anchor decoder cross-attends TEMPORAL: keep all
``W`` frames' feature maps instead of ``fmap_all[..., -1]``
(``stack/tanitad/refs/refc.py:1412-1422``).

⭐ THE FACT THAT SETS THE PRICE, read from source: with ``hierarchy=True`` (the
default, and ``true`` in REF-C-XL's run config) the encoder ALREADY runs on all
``b*w`` frames — ``fmap_all, pooled_all = self.encoder(frames.reshape(b*w, …))``
— and then ``[:, -1]`` throws **7 of 8 feature maps away**. So approach A costs
**ZERO extra ENCODER FLOPs**. Everything it costs is in the decoder's
cross-attention, and this script measures exactly that:

  * parameter delta (a per-frame KV embedding + one gate scalar),
  * decoder wall-time at KV=64 (today) vs KV=64*W (approach A),
  * peak CUDA memory at both,
  * the analytic MAC split so the measurement can be read (attention grows
    linearly in KV length; the per-anchor MLP does not grow at all — which is
    why the honest answer is much smaller than "8x").

usage: python analyze_temporal_kv_cost.py --out raw/temporal_kv_cost.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "stack"))

W_DEFAULT = 8            # RefCConfig.window (refc.py:386) — MEASURED from source


def bench(dec, fmap, m, ctx, steps, device, iters=8):
    """-> (median seconds, peak MiB). Warms up first; peak memory is reset."""
    torch.cuda.reset_peak_memory_stats(device) if device.startswith("cuda") else None
    with torch.no_grad():
        for _ in range(2):
            dec(fmap, m, ctx=ctx, steps=steps)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            dec(fmap, m, ctx=ctx, steps=steps)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    peak = (torch.cuda.max_memory_allocated(device) / 2 ** 20
            if device.startswith("cuda") else float("nan"))
    ts.sort()
    return ts[len(ts) // 2], peak


def analytic_macs(n_anchors, d, layers, ff_mult, kv_len, passes):
    """MACs for the decoder's cross-attention stack, split so it can be read.

    Per CrossAttnLayer (``refc.py:770-787``): q/k/v/out projections
    ``4*N*d^2`` + ``N*d^2``-ish, the two attention matmuls ``2*N*L*d``, and the
    FiLM'd MLP ``2*N*ff_mult*d^2``. Only the attention term carries ``L``.
    """
    proj = (n_anchors * d * d) + 2 * (kv_len * d * d) + (n_anchors * d * d)
    attn = 2 * n_anchors * kv_len * d
    mlp = 2 * n_anchors * ff_mult * d * d
    per_layer = proj + attn + mlp
    return {"per_layer_proj": proj, "per_layer_attn": attn,
            "per_layer_mlp": mlp, "per_layer_total": per_layer,
            "decoder_total": per_layer * layers * passes,
            "attn_share_of_layer": round(attn / per_layer, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="raw/temporal_kv_cost.json")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--window", type=int, default=W_DEFAULT)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from tanitad.refs.refc import (RefCModel, refc_config, refc_small_config,
                                   refc_xl_config, param_breakdown)

    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "device": args.device, "torch": torch.__version__,
           "window_W": args.window,
           "source_facts": {
               "single_instant_site": "stack/tanitad/refs/refc.py:1412-1422 "
                                      "(fmap = fmap_all.reshape(b,w,...)[:, -1])",
               "window_default": "RefCConfig.window = 8 (refc.py:386)",
               "encoder_already_runs_all_W_frames":
                   "refc.py:1414-1415 encoder(frames.reshape(b*w, ...)) under "
                   "hierarchy=True (default; 'hierarchy true' in REF-C-XL's run "
                   "config per MODEL_REGISTRY.md 4.1) -> approach A adds ZERO "
                   "encoder FLOPs; 7 of 8 feature maps are computed then dropped",
               "input_is_a_3_frame_stack":
                   "refc.py:241 in_channels=9 'D-015 3-frame RGB stack "
                   "(latest = [-3:])' -> one model 'frame' already spans 300 ms"},
           "presets": {}}

    for name, mk, steps in (("refc_small", refc_small_config, 0),
                            ("refc_base", refc_config, 0),
                            ("refc_xl", refc_xl_config, 2)):
        cfg = mk()
        model = RefCModel(cfg).to(args.device).eval()
        pb = param_breakdown(model)
        dec = model.decoder
        feat = model.encoder.feat_dim
        g = cfg.encoder.grid
        n_anchors = int(dec.anchors.shape[0])
        d = cfg.decoder.d
        b = args.batch
        rec = {"params_total": int(sum(p.numel() for p in model.parameters())),
               "param_breakdown": {k: int(v) for k, v in pb.items()},
               "feat_dim": int(feat), "token_grid": [int(g), int(g)],
               "n_tokens_per_frame": int(g * g), "n_anchors": n_anchors,
               "d": int(d), "layers": int(cfg.decoder.layers),
               "ff_mult": int(cfg.decoder.ff_mult), "diffusion_steps": int(steps),
               "decoder_passes": int(steps + 1)}
        # the two KV lengths
        fmap1 = torch.randn(b, feat, g, g, device=args.device)
        fmapW = torch.randn(b, feat, g, g * args.window, device=args.device)
        m = torch.randn(b, cfg.measurement.d_out, device=args.device)
        ctx = (torch.randn(b, cfg.strategic.d_ctx, device=args.device)
               if cfg.hierarchy else None)
        for tag, fm in (("today_KV64", fmap1), ("approachA_KVW", fmapW)):
            kv = int(fm.shape[2] * fm.shape[3])
            sec, peak = bench(dec, fm, m, ctx, steps, args.device)
            rec[tag] = {
                "kv_len": kv, "median_s": round(sec, 6),
                "peak_mib": round(peak, 1),
                "analytic_macs": analytic_macs(n_anchors, d, cfg.decoder.layers,
                                               cfg.decoder.ff_mult, kv,
                                               steps + 1)}
        rec["ratio_time"] = round(rec["approachA_KVW"]["median_s"]
                                  / max(rec["today_KV64"]["median_s"], 1e-9), 3)
        rec["ratio_decoder_macs"] = round(
            rec["approachA_KVW"]["analytic_macs"]["decoder_total"]
            / rec["today_KV64"]["analytic_macs"]["decoder_total"], 3)
        rec["ratio_peak_mib"] = round(rec["approachA_KVW"]["peak_mib"]
                                      / max(rec["today_KV64"]["peak_mib"], 1e-9), 3)
        #: the +params approach A needs: a per-frame KV embedding (W x d) and one
        #: scalar attention-bias gate (the `lan_gate` / `cons_gate` discipline —
        #: a large negative bias on the non-latest frames, so the model is
        #: BYTE-IDENTICAL at step 0 and the gradient is still non-zero).
        rec["approachA_added_params"] = int(args.window * d + 1)
        rec["approachA_added_params_pct"] = round(
            100.0 * (args.window * d + 1) / rec["params_total"], 5)
        out["presets"][name] = rec
        print(f"{name}: params {rec['params_total']:,}  "
              f"KV {rec['today_KV64']['kv_len']} -> {rec['approachA_KVW']['kv_len']}  "
              f"time x{rec['ratio_time']}  decoderMACs x{rec['ratio_decoder_macs']}  "
              f"peak x{rec['ratio_peak_mib']}  "
              f"+params {rec['approachA_added_params']} "
              f"({rec['approachA_added_params_pct']}%)", flush=True)
        del model, dec, fmap1, fmapW
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
