#!/usr/bin/env python3
"""Planning-tick latency on THOR — a NEW hardware row, NOT a replacement for the A40 row.

⛔ WHY THIS DOES NOT REPLACE §6's SIX FIGURES. Every committed efficiency artifact records
`env.gpu = "NVIDIA A40"`. Thor is aarch64 / NVIDIA Thor. Writing a Thor number into the A40 row
would repeat precisely the defect MODEL_REGISTRY §6 already documents at length (the 11.16 ms
"deploy tick" that was quoted against a 103.42 ms planning tick — "the two figures differ in five
dimensions at once and are not comparable"). This run is labelled as its own hardware.

⚠️ A FIRST-CALL NUMBER IS NOT A STEADY-STATE NUMBER. The harness warms up (default 30) and reports
a distribution over `iters` (default 200); p50 AND p95/p99 are reported, never a single call.

Registry ckpt paths are `/root/models/...` (the dead eval pod). Thor has no sudo and no `/root`
write, so the entries are overridden IN-PROCESS — no file in the checkout is edited.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/nvidia/TanitAD/taniteval")

THOR_CKPT = {
    "flagship-30k": "/home/nvidia/models/flagship-v1-speedjerk/ckpt.pt",
    "refc-xl-30k": "/home/nvidia/models/refc-xl/ckpt.pt",
}
THOR_VAL = "/home/nvidia/valdata/physicalai-val-0c5f7dac3b11"
THOR_RES = Path("/home/nvidia/leadwork/eff_thor")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="flagship-30k,refc-xl-30k")
    ap.add_argument("--precisions", default="fp32,tf32,amp16")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    a = ap.parse_args()

    import torch
    from taniteval import efficiency
    from taniteval.registry import MODELS

    # --- in-process overrides -------------------------------------------------
    efficiency.VAL = THOR_VAL
    THOR_RES.mkdir(parents=True, exist_ok=True)
    efficiency.RES = THOR_RES
    n_patched = 0
    for m in MODELS:
        if m["key"] in THOR_CKPT:
            m["ckpt"] = THOR_CKPT[m["key"]]
            n_patched += 1
    print(f"[thor-eff] patched {n_patched} registry ckpt paths; VAL={THOR_VAL}", flush=True)

    props = torch.cuda.get_device_properties(0)
    env = {"gpu": torch.cuda.get_device_name(0),
           "capability": f"{props.major}.{props.minor}",
           "total_mem_MiB": round(props.total_memory / 2**20),
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "platform": sys.platform, "machine": __import__("platform").machine()}
    print("[thor-eff] env:", json.dumps(env), flush=True)

    summary = {"host": "tanitad-thor", "env": env,
               "protocol_note": ("planning tick = encode(8-frame window) + 20 SEQUENTIAL "
                                 "predictor steps -> per-step metric dpose -> SE(2) accumulate; "
                                 "batch 1; warmup discarded; cuda.Event timing"),
               "arms": {}}
    precs = tuple(p.strip() for p in a.precisions.split(","))
    for key in [k.strip() for k in a.models.split(",")]:
        try:
            out = efficiency.run_and_save(key, precisions=precs, batch=1,
                                          iters=a.iters, warmup=a.warmup,
                                          throughput=False, res_dir=THOR_RES)
        except Exception as e:                      # loud, per-arm
            print(f"[thor-eff] {key} FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
            summary["arms"][key] = {"status": "FAILED",
                                    "error": f"{type(e).__name__}: {str(e)[:400]}"}
            continue
        arm = {"status": "OK", "ckpt": THOR_CKPT[key], "ckpt_step": out.get("ckpt_step")}
        for p in precs:
            if p in out and "plan_step" in out[p]:
                ps = out[p]["plan_step"]
                arm[p] = {k: ps.get(k) for k in
                          ("p50_ms", "p95_ms", "p99_ms", "mean_ms", "std_ms",
                           "min_ms", "max_ms", "iters", "warmup")}
                arm[p]["gpu_reported"] = out[p].get("env", {}).get("gpu")
        summary["arms"][key] = arm
        print(f"[thor-eff] {key}: " + json.dumps(
            {p: arm[p]["p50_ms"] for p in precs if p in arm}), flush=True)

    # peak memory — the ONLY admissible memory read on Thor
    summary["torch_max_memory_allocated_MiB"] = round(
        torch.cuda.max_memory_allocated() / 2**20, 1)
    (THOR_RES / "THOR_PLANNING_TICK.json").write_text(json.dumps(summary, indent=1))
    print("\n" + json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
