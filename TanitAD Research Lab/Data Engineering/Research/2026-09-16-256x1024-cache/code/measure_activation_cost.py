"""Measure the resnet101 activation footprint of a K=3 window at each geometry.

⚠️ WHAT THIS MEASURES, EXACTLY: the summed bytes of every module's OUTPUT tensor
in one forward pass, batch 1, ``in_chans = 3*K`` (the K-frame stack), on CPU, via
forward hooks. That is the quantity that dominates training memory, because
activations are retained for backward.

⚠️ WHAT IT IS **NOT**: a CUDA ``max_memory_allocated`` figure. This box's GPU is
owned by another agent and was not used. A CUDA peak additionally carries
workspace, fragmentation, parameters, gradients and optimizer state, so it is a
LARGER number measured a DIFFERENT way. Do not compare an activation-sum here
against a CUDA peak from elsewhere as if they were the same quantity -- compare
the RATIOS between geometries, which is what the batch envelope actually turns
on, and which this script reports directly.

Reported per geometry: total activation bytes, the returned feature-map bytes,
and a per-level breakdown. Ratios are taken against 256x1024.
"""
from __future__ import annotations
import argparse, json, sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                                # noqa: BLE001
        pass
import torch                                                         # noqa: E402


def measure(model, h, w, k, dtype_bytes=4):
    """Sum the output bytes of every leaf module in one forward, batch 1."""
    seen, total = [], 0

    def hook(mod, inp, out):
        nonlocal total
        ts = out if isinstance(out, (list, tuple)) else [out]
        for t in ts:
            if torch.is_tensor(t):
                n = t.numel() * dtype_bytes
                total += n
                seen.append((type(mod).__name__, tuple(t.shape), n))

    hs = [m.register_forward_hook(hook) for m in model.modules()
          if not list(m.children())]
    x = torch.zeros(1, 3 * k, h, w)
    with torch.no_grad():
        fs = model(x)
    for hh in hs:
        hh.remove()
    feat = sum(f.numel() * dtype_bytes for f in fs)
    levels = [{"shape": list(f.shape), "tokens": int(f.shape[-2] * f.shape[-1]),
               "mb": round(f.numel() * dtype_bytes / 2**20, 2)} for f in fs]
    return {"activation_sum_mb": round(total / 2**20, 1),
            "returned_features_mb": round(feat / 2**20, 2),
            "n_module_outputs": len(seen), "levels": levels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--model", default="resnet101")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cuda-peak-256x1024-mb", type=float, default=0.0,
                    help="an EXTERNAL CUDA peak figure for 256x1024, if one "
                         "exists, used ONLY to scale by the measured ratio")
    a = ap.parse_args()
    import timm
    torch.set_num_threads(4)
    m = timm.create_model(a.model, pretrained=False, features_only=True,
                          in_chans=3 * a.k)
    m.eval()
    red = m.feature_info.reduction()
    geoms = [("256x640", 256, 640), ("256x1024", 256, 1024), ("408x1024", 408, 1024)]
    R = {"model": a.model, "K": a.k, "in_chans": 3 * a.k, "batch": 1,
         "dtype": "fp32", "device": "cpu",
         "what_is_measured": "sum of every leaf module's output tensor bytes in "
                             "one forward pass; NOT a CUDA max_memory_allocated",
         "strides": list(red), "geometries": {}}
    base = None
    for tag, h, w in geoms:
        r = measure(m, h, w, a.k)
        r["height"], r["width"] = h, w
        R["geometries"][tag] = r
        if tag == "256x1024":
            base = r["activation_sum_mb"]
        print(f"  {tag}: activations {r['activation_sum_mb']:>8.1f} MB   "
              f"features {r['returned_features_mb']:>6.2f} MB   "
              f"stride16 {r['levels'][3]['shape'][-2]}x{r['levels'][3]['shape'][-1]}"
              f" stride32 {r['levels'][4]['shape'][-2]}x{r['levels'][4]['shape'][-1]}",
              flush=True)
    for tag, r in R["geometries"].items():
        r["ratio_vs_256x1024"] = round(r["activation_sum_mb"] / base, 4)
    R["ratios"] = {t: r["ratio_vs_256x1024"] for t, r in R["geometries"].items()}
    if a.cuda_peak_256x1024_mb > 0:
        R["external_cuda_peak_scaling"] = {
            "caveat": "The 256x1024 CUDA peak below was NOT measured by this "
                      "package. It is scaled by the activation RATIO measured "
                      "here, which is only valid if activations dominate that "
                      "peak. Treat as ESTIMATED, evidence class: derived.",
            "given_256x1024_mb": a.cuda_peak_256x1024_mb,
            "implied": {t: round(a.cuda_peak_256x1024_mb * r, 1)
                        for t, r in R["ratios"].items()}}
    json.dump(R, open(a.out, "w"), indent=1)
    print(f"\nratios vs 256x1024: {R['ratios']} -> {a.out}")


if __name__ == "__main__":
    main()
