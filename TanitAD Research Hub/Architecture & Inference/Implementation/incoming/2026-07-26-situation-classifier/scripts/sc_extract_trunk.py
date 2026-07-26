"""Situation classifier — extract the FROZEN v1 ENCODER+READOUT from the full v1 checkpoint.

Why: pod3 (this stream's assigned host) has the parity episode cache but **not** the v1 checkpoint,
and the dev-box -> pod relay runs at ~1 MB/s (MEASURED here: 32 MiB in 34.0 s = 0.99 MB/s). The full
v1 ckpt is 3.31 GB (55 min); the encoder + readout are 87.1 M params = 348 MB (~6 min).

⚠️ Weights are copied at FULL float32 precision. Casting the trunk to fp16 to save transfer time
would change the features and silently invalidate every comparison with H2, which used fp32 weights.

Verification on both ends: SHA256 of the payload + a per-tensor count/param-count assertion, and on
the pod side a STRICT `load_state_dict` into the config-built modules (so a missing or extra key is
a hard failure, not a silent partial load).

usage:  python sc_extract_trunk.py <full_ckpt.pt> <out.pt>
"""
from __future__ import annotations

import hashlib
import json
import sys

import torch


def main():
    src, dst = sys.argv[1:3]
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    if not isinstance(sd, dict):
        raise SystemExit(f"unexpected checkpoint layout: {type(sd)}")
    keys = [k for k in sd if k.startswith("encoder.") or k.startswith("readout.")]
    if not keys:
        raise SystemExit(f"no encoder./readout. keys; top-level keys = {list(sd)[:12]}")
    out = {k: sd[k].detach().clone() for k in keys}
    n_enc = sum(v.numel() for k, v in out.items() if k.startswith("encoder."))
    n_ro = sum(v.numel() for k, v in out.items() if k.startswith("readout."))
    payload = {"trunk": out, "src": src,
               "step": int(ck.get("step", ck.get("global_step", -1))),
               "encoder_params": int(n_enc), "readout_params": int(n_ro),
               "n_tensors": len(out), "dtype": str(next(iter(out.values())).dtype)}
    torch.save(payload, dst)
    h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    meta = {k: v for k, v in payload.items() if k != "trunk"}
    meta["sha256"] = h
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
