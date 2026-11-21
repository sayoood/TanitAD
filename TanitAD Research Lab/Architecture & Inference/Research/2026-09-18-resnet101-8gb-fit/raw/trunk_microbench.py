"""TRUNK-ONLY microbenchmark: rank the 8 GB levers on the refcv6 trunk.

WHY A MICROBENCH AT ALL. The binding question is whether resnet101 trains at
416 x 1024 on an 8 GB card. The full rig answers "fits / does not fit", but a
configuration that OOMs yields NO peak number, so a sweep run only in the rig
cannot RANK the levers -- every losing arm reads the same "OOM". This isolates
the trunk (forward + backward, batch 1, K=3 shared passes, the rig's exact
geometry) so every lever produces a number even when the full rig would not.

It is NOT the answer. It ranks. The winner is then re-measured in the FULL rig
(all heads live, real caches, real SAM3 maps) by `rig_arm.py`.

Only `torch.cuda.max_memory_allocated()` is admissible for device memory here
(the dev box's `nvidia-smi` reading includes the other tenant and the caching
allocator's reserve).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

from tanitad.models.timm_trunk import TimmResNetTrunk, TimmTrunkConfig

H, W = 416, 1024
K = 3
#: ⛔⛔ THE SHAPE THAT DECIDES EVERYTHING. `--arm hier` calls the trunk ONCE on
#: `frames.reshape(b*w, ...)` (`refc.py:3889`), and `refc_v3_sized_config("tiny",
#: hier=True).core.window == 8` -- so at batch 1 the trunk sees 8 stacks, which
#: `TimmResNetTrunk.forward_features` folds again into `8*K = 24` 3-channel
#: images. A microbench at WB=1 measures a shape the rig never runs.
WB_DEFAULT = 8


def build(model_name: str, mode: str) -> TimmResNetTrunk:
    cfg = TimmTrunkConfig(
        model_name=model_name,
        pretrained=False,            # ⚠️ DEVIATION, stated: weight VALUES do not
        verify_imagenet_stats=False,  # change activation bytes or op count. The
        frames=K,                    # rig arms below DO use --trunk-pretrained.
        mode=mode,                   # in_channels is a PROPERTY (= 3 * frames)
        fuse="concat1x1",
        image_hw=(H, W),
    )
    return TimmResNetTrunk(cfg)


def freeze_early(trunk: TimmResNetTrunk, upto: str) -> list[str]:
    """requires_grad_(False) on the stem (+ optionally layer1/layer2).

    ⛔ This CHANGES THE ARM: the frozen stages keep their ImageNet weights for
    the whole run and are no longer adapted to the cylindrical 416x1024 input.
    It is reported as a different claim, never as "resnet101 fits".
    """
    order = ["conv1", "bn1", "act1", "maxpool", "layer1", "layer2", "layer3"]
    stop = order.index(upto) + 1
    frozen = []
    for name in order[:stop]:
        mod = getattr(trunk.net, name, None)
        if mod is None:
            continue
        n = 0
        for p in mod.parameters(recurse=True):
            p.requires_grad_(False)
            n += 1
        if n:
            frozen.append(name)
        if isinstance(mod, nn.Module):
            mod.eval()          # BN in the frozen stem must not update stats
    return frozen


class _ChunkCkpt(nn.Module):
    """Run the backbone in leading-batch chunks, each under a checkpoint.

    ⭐ THE ONLY LEVER HERE THAT CHANGES NOTHING ABOUT THE ARM. The forward is
    mathematically identical (same weights, same dtype, same frames, same
    fusion); it trades ONE extra forward per chunk for not retaining 24 images'
    activations at once. `use_reentrant=False` plus out-of-place activations,
    because timm's own `set_grad_checkpointing(True)` raises on this backbone
    (MEASURED: *"modified by an inplace operation ... ReluBackward0"*).
    """

    def __init__(self, net: nn.Module, chunk: int):
        super().__init__()
        self.net = net
        self.chunk = int(chunk)

    def forward(self, x):
        outs = []
        for i in range(0, x.shape[0], self.chunk):
            outs.append(torch.utils.checkpoint.checkpoint(
                self.net, x[i:i + self.chunk], use_reentrant=False))
        n = len(outs[0])
        return [torch.cat([o[j] for o in outs], dim=0) for j in range(n)]


def freeze_bn(net: nn.Module) -> int:
    """Put every BatchNorm in the backbone into EVAL mode (ImageNet running
    statistics), leaving its affine weights trainable.

    ⛔⛔ WHY THIS IS HERE AND NOT AN AFTERTHOUGHT. Chunked checkpointing splits
    the 24-image batch, and BatchNorm's statistics are computed OVER THE BATCH
    -- so chunking silently changes the forward. MEASURED on the full rig:
    resnet34 `ga_trunk` 1,213,951 unchunked vs 724,635 chunked (-40 %). With BN
    on running statistics the normalisation no longer depends on batch
    composition, so chunk=1 and chunk=24 are the SAME function and the fit is a
    fit rather than a different arm wearing the arm's name.

    ⚠️ It is still a CHANGE vs today's reference, which trains BN on the batch.
    It is the standard choice for a batch-1 detection backbone, but it is a
    choice and it is stamped.
    """
    k = 0
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            k += 1
    return k


def kill_inplace(net: nn.Module) -> int:
    k = 0
    for m in net.modules():
        if getattr(m, "inplace", False):
            m.inplace = False
            k += 1
    return k


def run_arm(name: str, model_name: str, levers: dict, iters: int = 3,
            wb: int = WB_DEFAULT) -> dict:
    mode = str(levers.get("trunk_mode", "shared"))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rec: dict = {"arm": name, "model": model_name, "levers": dict(levers),
                 "image_hw": [H, W], "wb": wb, "frames": K, "mode": mode,
                 "images_through_trunk": wb * (1 if mode == "inflate" else K)}
    try:
        torch.manual_seed(0)      # identical weights + identical x per arm
        trunk = build(model_name, mode).cuda().train()
        rec["backbone_params"] = trunk.trunk_param_count()
        rec["total_params"] = trunk.param_count()
        if levers.get("grad_checkpoint"):
            # ⛔ inplace FIRST or timm's own `checkpoint_seq` raises on the
            # version counter (MEASURED, r34 and r101 both).
            rec["relu_outofplace"] = kill_inplace(trunk.net)
            trunk.net.set_grad_checkpointing(True)
            rec["gc_applied"] = bool(getattr(trunk.net, "grad_checkpointing",
                                             False))
        if levers.get("freeze"):
            rec["frozen"] = freeze_early(trunk, str(levers["freeze"]))
        if levers.get("frozen_bn"):
            rec["bn_eval"] = freeze_bn(trunk.net)
        if levers.get("chunk_ckpt"):        # AFTER freeze: it reads net.conv1
            rec["relu_outofplace"] = kill_inplace(trunk.net)
            trunk.net = _ChunkCkpt(trunk.net, int(levers["chunk_ckpt"]))
            rec["chunk"] = int(levers["chunk_ckpt"])
        if levers.get("channels_last"):
            trunk = trunk.to(memory_format=torch.channels_last)
        rec["trainable_params"] = sum(p.numel() for p in trunk.parameters()
                                      if p.requires_grad)
        opt = torch.optim.AdamW([p for p in trunk.parameters()
                                 if p.requires_grad], lr=1e-4)
        amp = bool(levers.get("bf16"))
        dt = []
        for i in range(iters):
            torch.manual_seed(100 + i)
            x = torch.rand(wb, 3 * K, H, W, device="cuda")
            if levers.get("channels_last"):
                x = x.contiguous(memory_format=torch.channels_last)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16, enabled=amp):
                s16, s32, _pooled = trunk.forward_features(x)
            # ⛔ .float() so the SURROGATE loss is fp32 in every arm: a bf16
            # loss would also change the number being backwarded, confounding
            # "bf16 saved activations" with "bf16 changed the objective".
            loss = s16.float().pow(2).mean() + s32.float().pow(2).mean()
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            dt.append(time.perf_counter() - t0)
        rec["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
        rec["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024 ** 3
        rec["iter_s_all"] = [round(v, 4) for v in dt]
        rec["iter_s"] = round(sorted(dt[1:])[len(dt[1:]) // 2], 4)
        rec["s16_shape"] = list(s16.shape)
        rec["s32_shape"] = list(s32.shape)
        # a trunk whose grads are all zero/absent is a false fit
        gn = sum(float(p.grad.detach().abs().sum())
                 for p in trunk.parameters() if p.grad is not None)
        rec["trunk_grad_abs_sum"] = gn
        rec["grad_fingerprint"] = {
            n: float(p.grad.detach().abs().sum())
            for n, p in list(trunk.named_parameters())[:6] if p.grad is not None}
        rec["n_grad_none"] = sum(1 for p in trunk.parameters()
                                 if p.requires_grad and p.grad is None)
        rec["status"] = "OK"
    except torch.cuda.OutOfMemoryError as e:
        rec["status"] = "OOM"
        rec["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
        rec["error"] = str(e).split("\n")[0][:400]
    except Exception as e:                                # pragma: no cover
        rec["status"] = "ERROR"
        rec["error"] = f"{type(e).__name__}: {e}"[:400]
    finally:
        torch.cuda.empty_cache()
    return rec


ARMS: list[tuple[str, str, dict]] = [
    ("r34_base",        "resnet34.a1_in1k",  {}),
    ("r101_base",       "resnet101.a1_in1k", {}),
    ("r101_ck1",        "resnet101.a1_in1k", {"chunk_ckpt": 1}),
    ("r101_ck2",        "resnet101.a1_in1k", {"chunk_ckpt": 2}),
    ("r101_ck4",        "resnet101.a1_in1k", {"chunk_ckpt": 4}),
    ("r101_timmgc",     "resnet101.a1_in1k", {"grad_checkpoint": True}),
    ("r34_timmgc",      "resnet34.a1_in1k",  {"grad_checkpoint": True}),
    ("r101_timmgc_bf16", "resnet101.a1_in1k", {"grad_checkpoint": True,
                                               "bf16": True}),
    ("r101_bf16",       "resnet101.a1_in1k", {"bf16": True}),
    ("r101_chlast",     "resnet101.a1_in1k", {"channels_last": True}),
    ("r101_freeze_l1",  "resnet101.a1_in1k", {"freeze": "layer1"}),
    ("r101_freeze_l2",  "resnet101.a1_in1k", {"freeze": "layer2"}),
    ("r101_inflate",    "resnet101.a1_in1k", {"trunk_mode": "inflate"}),
    ("r101_ck1_bf16",   "resnet101.a1_in1k", {"chunk_ckpt": 1, "bf16": True}),
    ("r101_ck2_bf16",   "resnet101.a1_in1k", {"chunk_ckpt": 2, "bf16": True}),
    ("r101_ck1_bf16_fz", "resnet101.a1_in1k", {"chunk_ckpt": 1, "bf16": True,
                                               "freeze": "layer1"}),
    ("r101_infl_bf16",  "resnet101.a1_in1k", {"trunk_mode": "inflate",
                                              "bf16": True}),
    ("r34_ck1",         "resnet34.a1_in1k",  {"chunk_ckpt": 1}),
    ("r34_bf16",        "resnet34.a1_in1k",  {"bf16": True}),
    # ---- THE EXACTNESS PAIR: frozen BN makes chunking the SAME function ---- #
    ("r34_fbn",         "resnet34.a1_in1k",  {"frozen_bn": True}),
    ("r34_fbn_ck1",     "resnet34.a1_in1k",  {"frozen_bn": True,
                                              "chunk_ckpt": 1}),
    ("r101_fbn_ck1",    "resnet101.a1_in1k", {"frozen_bn": True,
                                              "chunk_ckpt": 1}),
    ("r101_fbn_ck2",    "resnet101.a1_in1k", {"frozen_bn": True,
                                              "chunk_ckpt": 2}),
    ("r101_fbn_ck3",    "resnet101.a1_in1k", {"frozen_bn": True,
                                              "chunk_ckpt": 3}),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--wb", type=int, default=WB_DEFAULT,
                    help="leading batch = batch * window. 8 = the rig at "
                         "--batch 1 --arm hier (core.window == 8).")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("ZZABORT-NO-CUDAZZ")
        return 2
    sel = set(x for x in a.only.split(",") if x)
    rows = []
    for name, model, levers in ARMS:
        if sel and name not in sel:
            continue
        rec = run_arm(name, model, levers, wb=a.wb)
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"host": "dev-box RTX 4060 8188 MiB",
                   "torch": torch.__version__,
                   "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
                   "geometry": [H, W], "wb": a.wb, "frames": K,
                   "arms": rows}, f, indent=1)
    print("ZZMICRO-DONE %d armsZZ" % len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
