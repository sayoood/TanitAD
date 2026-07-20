"""TanitEval — profiling. Works directly on checkpoint state_dicts (mmap, no
arch construction) for compute/memory footprint, and optionally times a live
forward pass. Distinguishes TRAINED params (in the ckpt) from the FROZEN encoder
(external, not in the ckpt) — critical for a fair compute comparison, since the
frozen-encoder REF-A ckpts exclude the DINOv2/I-JEPA weights they run at inference."""
import collections
import os
import torch

# Frozen encoders live OUTSIDE the REF-A ckpt (loaded separately at inference).
FROZEN_ENCODER_PARAMS_M = {
    "frozen DINOv2-B/14": 86.6,
    "frozen I-JEPA ViT-H/14": 632.0,
}


def analyze_state_dict(ckpt_path, encoder=None, encoder_frozen=False):
    ck = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    sd = ck.get("model", ck)
    by_comp, dtypes, total = collections.Counter(), collections.Counter(), 0
    for k, v in sd.items():
        if not hasattr(v, "numel"):
            continue
        n = v.numel()
        total += n
        by_comp[k.split(".")[0]] += n
        dtypes[str(v.dtype)] += n
    trained_m = total / 1e6
    frozen_m = FROZEN_ENCODER_PARAMS_M.get(encoder, 0.0) if encoder_frozen else 0.0
    total_infer_m = trained_m + frozen_m
    return dict(
        trained_params_m=round(trained_m, 2),
        frozen_encoder_params_m=round(frozen_m, 2),
        total_inference_params_m=round(total_infer_m, 2),
        by_component_m={k: round(v / 1e6, 3) for k, v in by_comp.most_common()},
        ckpt_mb=round(os.path.getsize(ckpt_path) / 1e6, 1),
        trained_mem_fp32_mb=round(trained_m * 4, 1),
        trained_mem_fp16_mb=round(trained_m * 2, 1),
        infer_mem_fp16_mb=round(total_infer_m * 2, 1),
        dtypes_m={k: round(v / 1e6, 2) for k, v in dtypes.items()},
        step=ck.get("step"),
        aux_heads=[k for k in ck if k.startswith(("aux_", "step_readout", "metric_", "grounding"))],
    )


@torch.no_grad()
def time_forward(fn, *args, warmup=3, iters=20, device="cuda"):
    """Latency (ms/call), throughput, peak VRAM for a callable forward `fn`."""
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize(device)
    ms = start.elapsed_time(end) / iters
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
    return dict(latency_ms=round(ms, 3), peak_vram_mb=round(peak_mb, 1))
