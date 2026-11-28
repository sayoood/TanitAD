"""L5 TF32 diagnosis (MM 2026-10-05, BOTH halves on the Thor GPU): the gradient-share linearity control on a micro-net
built like the trunk (timm resnet34, 9 input channels, random init), with three readings on the SAME batch:

  ON   fp32 weights, TF32 ON (cuDNN default)        -> must reproduce the >1e-4 order (the G-SMOKE's 4.2e-4)
  OFF  fp32 weights, TF32 OFF                        -> must read <= 1e-4
  REP  the L5 `grad_share.fp32_replay` with the GLOBAL switches left ON and a bf16 lever on the trunk
                                                      -> must read <= 1e-4 (the replay turns TF32 off itself)
plus a REPEAT of OFF (reported, not gated): MEASURED on Thor the two OFF readings are NOT bit-identical (cuDNN's
backward is non-deterministic) but agree to 2e-10 absolute in lin_rel_err -- ~600x below the TF32 effect.
Writes <out>.json. Small: batch 2, 9 x 208 x 512, a few seconds of GPU.
Usage (Thor, under the GPU lock): python tf32_microcheck.py --gs <path to the L5 grad_share.py> --out <json>
"""
from __future__ import annotations

import argparse
import importlib.util
import json

import torch


class Trunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.net = timm.create_model("resnet34", pretrained=False, in_chans=9, features_only=True)
        self.memory_levers = {"bf16": False}

    def forward(self, x):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(self.memory_levers["bf16"])):
            f = self.net(x)
        return [t.float() for t in f]


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Trunk()
        self.h_agent = torch.nn.Conv2d(512, 16, 1)
        self.h_box = torch.nn.Conv2d(256, 16, 1)
        self.h_traj = torch.nn.Linear(512, 16)

    def losses(self, x):
        f = self.encoder(x)
        a = (self.h_agent(f[-1]) ** 2).mean()
        b = (self.h_box(f[-2]) - 0.5).abs().mean()
        t = ((self.h_traj(f[-1].mean((2, 3))) - 1.0) ** 2).mean()
        return {"agent": a, "box3d": b, "traj": t, "loss": a + 0.7 * b + 0.01 * t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("gs_l5", a.gs)
    GS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(GS)
    torch.manual_seed(0)
    net = Net().cuda().train()
    x = torch.randn(2, 9, 208, 512, generator=torch.Generator().manual_seed(1)).cuda()
    groups = {"trunk": list(net.encoder.parameters())}
    w = {"traj": 0.01, "agent": 1.0, "box3d": 0.7}

    def reading(tf32: bool):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
        torch.manual_seed(5)
        lo = net.losses(x)
        return GS.measure(GS.term_tensors(lo, w), lo["loss"], groups)

    on = reading(True)
    off1 = reading(False)
    off2 = reading(False)
    # REP: global switches ON (the trainer's configuration) + bf16 lever ON; the replay must handle both itself
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    net.encoder.memory_levers["bf16"] = True
    torch.manual_seed(5)
    rep = GS.fp32_replay(net, lambda: net.losses(x), groups, w)
    restored = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
                net.encoder.memory_levers["bf16"])
    rec = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "lin_rel_err": {"TF32_ON": on["gs_trunk_lin_rel_err"], "TF32_OFF": off1["gs_trunk_lin_rel_err"],
                           "TF32_OFF_repeat": off2["gs_trunk_lin_rel_err"],
                           "L5_replay_global_ON_bf16_lever": rep["gs_trunk_lin_rel_err"]},
           "determinism_off_identical": off1 == off2,
           "switches_restored_after_replay": list(restored),
           "bar": 1e-4}
    rec["PASS_cause_reproduced"] = rec["lin_rel_err"]["TF32_ON"] > 1e-4
    rec["PASS_off_meets_bar"] = rec["lin_rel_err"]["TF32_OFF"] <= 1e-4
    rec["PASS_replay_meets_bar"] = rec["lin_rel_err"]["L5_replay_global_ON_bf16_lever"] <= 1e-4
    rec["PASS"] = bool(rec["PASS_cause_reproduced"] and rec["PASS_off_meets_bar"] and rec["PASS_replay_meets_bar"]
                       and restored == (True, True, True))
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
