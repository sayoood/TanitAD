"""ONE FULL-RIG ARM of the resnet101-on-8GB fit question, with levers applied.

The rig is `cover139d.sh`'s command line (all heads live, real v2 caches, real
SAM3 map GT, real 3-D agent join, real extrinsics) at batch 1, 416 x 1024 --
the ONLY difference from that script is `--batch 1`, `--steps 3` and no eval
pass, because the question is the TRAINING step's peak, not the eval's.

⛔ THE TRAINER IS NOT EDITED. `refc_v3_train.py` has no grad-checkpoint, no
AMP, no channels_last and no accumulation flag (verified by reading it: the
only `--enc-grad-checkpoint` in this repo belongs to `train_v6_staged.py`'s v6
encoder, a different model). Every lever here is therefore a MONKEYPATCH
applied from outside, so the measurement cannot silently become a permanent
behaviour change to a trainer other arms share.

⛔ Only `torch.cuda.max_memory_allocated()` is read for device memory, and it
is reset immediately before `train()`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

ART = "D:/Projects/TanitAD-artifacts"
LAB = ("C:/Users/Admin/tanitad-wt/_s2build/release/v8/"
       "s2_labels_v8_eval.jsonl.gz")
A_CACHE = f"{ART}/v2ep-eval139-416x1024cyl-splitA"
MAPS = f"{ART}/sam3-maps-eval"
JOIN = f"{ART}/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
EXTR = f"{ART}/refcv5v2_final/extrinsics141.json"


def base_argv(out: str, trunk_name: str, batch: int, steps: int,
              image_hw: tuple[int, int], trunk_mode: str) -> list[str]:
    h, w = image_hw
    return [
        "--arm", "hier", "--size", "tiny", "--out", out,
        "--device", "cuda", "--seed", "0", "--steps", str(steps),
        "--batch", str(batch), "--workers", "0",
        "--log-every", "1", "--save-every", "1000000",
        "--v2-cache", A_CACHE, "--image-hw", str(h), str(w), "--v2-lru", "4",
        "--trunk", "timm", "--trunk-name", trunk_name,
        "--trunk-in-channels", "9", "--trunk-pretrained",
        "--trunk-mode", trunk_mode,
        "--v7-labels", LAB,
        "--agent-join", JOIN, "--agent-join-verify", "off",
        "--agents", "head", "--w-agent", "1.0",
        "--agent-queries", "16", "--agent-pad", "32",
        "--agent-rig-camera", "extrinsics", "--agent-rig-extrinsics", EXTR,
        "--map-gt-root", MAPS, "--map-lru", "2", "--map-min-coverage", "0.90",
        "--w-map", "1.0", "--join3d", JOIN, "--w-box3d", "1.0",
        "--tac-decoder-v6", "--w-tac-v6", "1.0", "--tac-decoder-d-bev", "96",
        "--conflict-detector", "off",
        # non-empty => `_grad_probe_row` also emits `gp_cuda_max_mem_gb`
        "--grad-probe-modules", "core.decoder",
    ]


def _kill_inplace_relu(net) -> int:
    """timm's `checkpoint_seq` recomputes a segment whose INPLACE ReLU then
    trips autograd's version counter (MEASURED: `set_grad_checkpointing(True)`
    on resnet34 AND resnet101 raises *"modified by an inplace operation ...
    ReluBackward0 ... version 1"*). Grad checkpointing is unusable on this
    trunk until the activations are made out-of-place, so the lever's real
    price includes this.
    """
    import torch.nn as nn
    n = 0
    for m in net.modules():
        if isinstance(m, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.Hardswish)) \
                and getattr(m, "inplace", False):
            m.inplace = False
            n += 1
    return n


class _ChunkCkpt:
    """Leading-batch chunked checkpointing of the backbone -- see
    `trunk_microbench._ChunkCkpt`. ⭐ The ONE lever that does not change the
    arm: same weights, same dtype, same frames, same fusion, one extra forward.
    ⛔ It is NOT timm's `set_grad_checkpointing`, which RAISES on this backbone
    (inplace ReLU vs the version counter, MEASURED both on resnet34 and 101).
    """

    def __new__(cls, net, chunk):
        import torch
        import torch.nn as nn

        class _Impl(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = net
                self.chunk = int(chunk)

            def forward(self, x):
                outs = []
                for i in range(0, x.shape[0], self.chunk):
                    outs.append(torch.utils.checkpoint.checkpoint(
                        self.net, x[i:i + self.chunk], use_reentrant=False))
                n = len(outs[0])
                return [torch.cat([o[j] for o in outs], dim=0)
                        for j in range(n)]

        return _Impl()


def _freeze_bn(net) -> int:
    """Pin every BatchNorm in the backbone to EVAL (ImageNet running stats).

    ⛔ It REFUSES to be un-frozen. `nn.Module.train()` recurses into children,
    and the trainer calls `model.train()` -- a plain `.eval()` here would be
    silently undone on the first step and the run would report a frozen-BN arm
    while training BN on the batch. Replacing the bound `train` is what makes
    the claim survive contact with the trainer; `bn_training_on_first_forward`
    below READS it back during the real run rather than trusting this.
    """
    import torch.nn as nn
    k = 0
    for m in net.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            m.train = (lambda mode=True, _m=m: _m)
            k += 1
    return k


def apply_levers(levers: dict, rec: dict) -> None:
    import torch
    import torch.nn as nn
    from tanitad.models import timm_trunk as TT

    orig_init = TT.TimmResNetTrunk.__init__
    orig_ff = TT.TimmResNetTrunk.forward_features

    def patched_init(self, cfg=None):
        orig_init(self, cfg)
        if levers.get("grad_checkpoint"):
            rec["relu_outofplace"] = _kill_inplace_relu(self.net)
            self.net.set_grad_checkpointing(True)
            rec["gc_applied"] = bool(getattr(self.net, "grad_checkpointing",
                                             False))
        fz = levers.get("freeze")
        if fz:
            order = ["conv1", "bn1", "act1", "maxpool",
                     "layer1", "layer2", "layer3"]
            frozen = []
            for name in order[:order.index(fz) + 1]:
                mod = getattr(self.net, name, None)
                if mod is None:
                    continue
                k = 0
                for p in mod.parameters(recurse=True):
                    p.requires_grad_(False)
                    k += 1
                if k:
                    frozen.append(name)
                mod.eval()
            rec["frozen"] = frozen
        if levers.get("channels_last"):
            self.net.to(memory_format=torch.channels_last)
        if levers.get("frozen_bn"):
            rec["bn_eval"] = _freeze_bn(self.net)
        if levers.get("chunk_ckpt"):        # AFTER freeze: freeze reads .conv1
            rec["relu_outofplace"] = _kill_inplace_relu(self.net)
            self.net = _ChunkCkpt(self.net, int(levers["chunk_ckpt"]))
            rec["chunk"] = int(levers["chunk_ckpt"])

    def patched_ff(self, x, already_normalised=False):
        if "bn_training_on_first_forward" not in rec:
            import torch.nn as nn
            rec["bn_training_on_first_forward"] = sum(
                1 for m in self.modules()
                if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training)
            rec["trunk_input_shape"] = list(x.shape)
        if levers.get("channels_last"):
            x = x.contiguous(memory_format=torch.channels_last)
        if levers.get("bf16_trunk"):
            with torch.autocast("cuda", torch.bfloat16):
                s16, s32, pooled = orig_ff(self, x, already_normalised)
            # ⛔ back to fp32 at the trunk boundary: everything downstream is
            # then BIT-FOR-BIT the unpatched arm's dtype, so a delta measured
            # here is the TRUNK's, not the whole model's.
            return s16.float(), s32.float(), pooled.float()
        return orig_ff(self, x, already_normalised)

    TT.TimmResNetTrunk.__init__ = patched_init
    TT.TimmResNetTrunk.forward_features = patched_ff
    # `forward` calls forward_features, so it inherits the patch.


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--trunk-name", default="resnet101.a1_in1k")
    ap.add_argument("--trunk-mode", default="shared")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--image-hw", type=int, nargs=2, default=[416, 1024])
    ap.add_argument("--levers", default="",
                    help="comma list of: chunk_ckpt=N, grad_checkpoint, "
                         "bf16_trunk, bf16_all, channels_last, freeze=layerN")
    ap.add_argument("--outroot", required=True)
    a = ap.parse_args()

    levers: dict = {}
    for tok in [t for t in a.levers.split(",") if t]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            levers[k] = v
        else:
            levers[tok] = True

    rec: dict = {"tag": a.tag, "trunk_name": a.trunk_name,
                 "trunk_mode": a.trunk_mode, "batch": a.batch,
                 "steps": a.steps, "image_hw": list(a.image_hw),
                 "levers": dict(levers),
                 "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")}

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    import torch
    stack = "C:/Users/Admin/tanitad-wt-bevtac/stack"
    sys.path.insert(0, os.path.join(stack, "scripts"))
    import refc_v3_train as T                       # noqa: E402
    rec["trainer_file"] = T.__file__

    apply_levers(levers, rec)

    out = os.path.join(a.outroot, a.tag)
    argv = base_argv(out, a.trunk_name, a.batch, a.steps,
                     tuple(a.image_hw), a.trunk_mode)
    rec["argv"] = argv
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        if levers.get("bf16_all"):
            with torch.autocast("cuda", torch.bfloat16):
                T.main(argv)
        else:
            T.main(argv)
        rec["status"] = "OK"
    except torch.cuda.OutOfMemoryError as e:
        rec["status"] = "OOM"
        rec["error"] = str(e).split("\n")[0][:500]
        rec["traceback_tail"] = traceback.format_exc()[-1500:]
    except SystemExit as e:
        rec["status"] = "OK" if not e.code else "EXIT"
        rec["exit_code"] = e.code
    except Exception as e:                            # pragma: no cover
        rec["status"] = "ERROR"
        rec["error"] = f"{type(e).__name__}: {e}"[:500]
        rec["traceback_tail"] = traceback.format_exc()[-1500:]
    rec["wallclock_s"] = round(time.time() - t0, 1)
    rec["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    rec["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024 ** 3

    # ---- the trainer's own rows: per-head reach and per-step wall time ---- #
    mpath = os.path.join(out, "metrics.jsonl")
    rows = []
    if os.path.exists(mpath):
        with open(mpath, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        rows.append(json.loads(ln))
                    except json.JSONDecodeError:
                        pass
    steps = [r for r in rows if "step" in r and "elapsed_s" in r]
    rec["n_metric_rows"] = len(rows)
    if len(steps) >= 2:
        # ⛔ step 1 carries cudnn autotune + the first cache decode; the
        # steady-state number is the LAST consecutive delta.
        d = [round(steps[i]["elapsed_s"] - steps[i - 1]["elapsed_s"], 2)
             for i in range(1, len(steps))]
        rec["step_deltas_s"] = d
        rec["s_per_step"] = d[-1]
    last = steps[-1] if steps else {}
    rec["head_grad_reach"] = {k: v for k, v in last.items()
                              if k.startswith("ga_") or k.startswith("gp_")}
    rec["losses_last"] = {k: v for k, v in last.items()
                          if not k.startswith(("ga_", "gp_"))}
    with open(os.path.join(a.outroot, f"{a.tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print("ZZARM %s status=%s peak=%.3fGB s_per_step=%s heads=%dZZ"
          % (a.tag, rec["status"], rec["peak_alloc_gb"],
             rec.get("s_per_step"), len(rec["head_grad_reach"])), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
