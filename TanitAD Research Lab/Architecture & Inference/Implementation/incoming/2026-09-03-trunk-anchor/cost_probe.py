"""D-V7-TRUNK-ANCHOR cost probe — MEASURED, CPU, then scaled to v7f geometry."""
import json
import os
import sys
import time
from pathlib import Path

import torch

# The stack to import. TANITAD_STACK wins (the G: mount cannot RUN the
# stack, so this normally points at the off-Drive clone); otherwise walk
# up for a sibling `stack/` directory.
_env = os.environ.get("TANITAD_STACK")
_S = Path(_env) if _env else None
if _S is None:
    for _d in Path(__file__).resolve().parents:
        if (_d / "stack" / "scripts" / "train_v6_staged.py").exists():
            _S = _d / "stack"
            break
    else:
        raise SystemExit("set TANITAD_STACK to the stack/ directory")
sys.path.insert(0, str(_S))
sys.path.insert(0, str(_S / "scripts"))
torch.set_num_threads(4)

from tanitad.config import EncoderConfig            # noqa: E402
from tanitad.models.encoder import ViTEncoder       # noqa: E402

OUT = {}


def _t(fn, reps=3, warm=1):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


# --------------------------------------------------------------------------
# A. tiny CPU config — the full step, three arms
# --------------------------------------------------------------------------
from train_v6_staged import (  # noqa: E402
    EncoderTokenTap, V6LossWeights, anchor_and_monitor_step,
    apply_encoder_seed, build_observer_monitor, build_parser,
    build_stack_from_args, build_trunk_anchor, synthetic_train_batch,
    v6_loss_step, ENCODER_SEED_FORMAT, ENCODER_SEED_MARKER,
)

BASE = ["--out", "UNUSED", "--stage", "S-W", "--frame-h", "64",
        "--frame-w", "128", "--enc-dim", "64", "--enc-depth", "3",
        "--enc-heads", "4", "--patch", "16", "--readout-grid", "2",
        "--readout-grid-w", "4", "--readout-dim", "16", "--pred-dim", "32",
        "--pred-depth", "1", "--pred-heads", "2", "--window", "3",
        "--d-tac", "16", "--d-str", "8"]


def _seed_file(tmp, st):
    sd = {k: v.clone() for k, v in st.encoder.state_dict().items() if k != "pos"}
    ec = st.cfg.encoder
    ih, iw = ec.image_hw()
    prov = {"format": ENCODER_SEED_FORMAT,
            "dinov3_model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "dinov3_variant": "vitb16", "source_sha256": "0" * 64,
            "left_at_init_keys": ["pos"], "layer_scale_folded": True,
            "n_mapped": len(sd), "n_skipped_allowlist": 0, "n_left_at_init": 1,
            "declared_losses": [],
            "target_geometry": {"class": type(st.encoder).__name__,
                                "d_model": int(ec.d_model),
                                "depth": int(ec.depth),
                                "n_heads": int(ec.n_heads),
                                "patch_size": int(ec.patch_size),
                                "in_channels": int(ec.in_channels),
                                "image_size": int(ih), "image_width": int(iw),
                                "n_tokens": int(st.encoder.n_tokens)}}
    p = Path(tmp) / "seed.pt"
    torch.save({ENCODER_SEED_MARKER: 1, "encoder": sd, "_provenance": prov}, p)
    return p


def arm(extra, tmp, label, batch=4):
    a = build_parser().parse_args(BASE + extra)
    st = build_stack_from_args(a)
    rep = {"init_encoder_from": None}
    if getattr(a, "init_encoder_from", None):
        rep = apply_encoder_seed(a, st)
    anch = build_trunk_anchor(a, st, rep)
    mon = build_observer_monitor(a, st)
    tap = (EncoderTokenTap(st.encoder)
           if (anch is not None or mon is not None) else None)
    trainable = [p for p in st.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    b = synthetic_train_batch(st, batch=batch, k=3, seed=1)
    b["gt_wp"] = torch.randn(batch, 3, 2)

    def step():
        if tap is not None:
            tap.arm()
        L = v6_loss_step(st, b, stage="S-W", weights=V6LossWeights(),
                         o1_k=3, o5_k=3)
        if tap is not None:
            tap.disarm()
            anchor_and_monitor_step(anch, mon, tap, L, b, 1)
        opt.zero_grad(set_to_none=True)
        L["loss"].backward()
        opt.step()

    dt = _t(step, reps=5, warm=2)
    n_teacher = 0 if anch is None else anch.n_params
    return {"arm": label, "step_s": round(dt, 5),
            "teacher_params": n_teacher,
            "teacher_bytes_fp32": n_teacher * 4}


def main():
    import tempfile
    tmp = tempfile.mkdtemp()
    st0 = build_stack_from_args(build_parser().parse_args(BASE))
    seed = _seed_file(tmp, st0)
    arms = [
        arm([], tmp, "default (w=0, monitor off)"),
        arm(["--obs-monitor-every", "50"], tmp, "monitor only"),
        arm(["--init-encoder-from", str(seed), "--w-trunk-anchor", "1.0",
             "--obs-monitor-every", "50"], tmp, "anchor + monitor"),
    ]
    base = arms[0]["step_s"]
    for r in arms:
        r["vs_default_pct"] = round(100.0 * (r["step_s"] / base - 1.0), 2)
    OUT["A_tiny_cpu_full_step"] = arms

    # ----------------------------------------------------------------------
    # B. the ENCODER alone at the v7f geometry (768x12, 256x640, patch 16)
    #    measured per-image on CPU, then scaled by batch (linear in B).
    # ----------------------------------------------------------------------
    cfg = EncoderConfig(d_model=768, depth=12, n_heads=12, patch_size=16,
                        in_channels=3, image_size=256, image_width=640)
    enc = ViTEncoder(cfg).eval()
    n_par = sum(p.numel() for p in enc.parameters())
    x1 = torch.randn(1, 3, 256, 640)
    fwd_ng = _t(lambda: torch.no_grad()(lambda: enc(x1))(), reps=3, warm=1)

    for p in enc.parameters():
        p.requires_grad_(True)

    def fwd_bwd():
        y = enc(x1)
        y.sum().backward()
        enc.zero_grad(set_to_none=True)

    fb = _t(fwd_bwd, reps=3, warm=1)
    OUT["B_v7f_encoder_per_image_cpu"] = {
        "geometry": "768x12 heads12 patch16 256x640 in_ch3",
        "n_tokens": int(enc.n_tokens), "n_params": n_par,
        "params_bytes_fp32": n_par * 4,
        "forward_no_grad_s": round(fwd_ng, 4),
        "forward_plus_backward_s": round(fb, 4),
        "bwd_over_fwd": round(fb / fwd_ng, 3),
    }

    # v7f: batch 8, window 6 -> the live trunk sees 48 images per step
    B, W = 8, 6
    live = B * W * fb                       # forward+backward, all W frames
    teacher = B * fwd_ng                    # no_grad, newest frame only
    second_live = B * fb                    # the rejected alternative
    OUT["C_v7f_projection"] = {
        "batch": B, "window": W,
        "live_encoder_fwd_bwd_s": round(live, 3),
        "tap_route_extra_s": round(teacher, 3),
        "tap_route_extra_pct_of_encoder": round(100 * teacher / live, 2),
        "second_live_forward_route_extra_s": round(teacher + second_live, 3),
        "second_live_route_extra_pct_of_encoder":
            round(100 * (teacher + second_live) / live, 2),
        "all_W_frames_anchor_extra_pct_of_encoder":
            round(100 * (W * teacher) / live, 2),
        "teacher_memory_MB_fp32": round(n_par * 4 / 1e6, 1),
        "teacher_memory_MB_if_TRAINABLE_with_adamw":
            round(n_par * 4 * 4 / 1e6, 1),
        "_read": "CPU per-image timings scaled by batch (the encoder is "
                 "linear in batch at fixed token count). The RATIO is the "
                 "quotable quantity; the absolute seconds are CPU numbers and "
                 "are NOT a Thor/A40 estimate.",
        "_evidence_class": "MEASURED per-image (ours, CPU) + arithmetic "
                           "batch scaling",
    }
    print(json.dumps(OUT, indent=1))


if __name__ == "__main__":
    main()
