#!/usr/bin/env python3
"""Planning-fan latency: is imagine-and-select / CEM affordable on this tick?

The arithmetic that makes CEM look impossible ("8 candidates x 20 steps") assumes
the fan costs 8 SEQUENTIAL ticks. It does not — a K-candidate fan is ONE rollout
at batch K. This measures the real cost per candidate instead of multiplying.

Run on an EXCLUSIVE GPU (take `gpu_lock.sh` first):
    PYTHONPATH=/root/taniteval:/root/TanitAD/stack:/root/TanitAD/stack/scripts \
      python3 taniteval/levers_fan.py [--model flagship-30k]
-> results/eff_levers_fan_<key>.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from taniteval import efficiency as E  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser("levers_fan")
    ap.add_argument("--model", default="flagship-30k")
    ap.add_argument("--precision", default="fp32,tf32")
    ap.add_argument("--batches", default="1,2,4,8,16")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--variants", default="eager,graph_rollout,all_levers")
    ap.add_argument("--tag", default="",
                    help="suffix for the output filename (keeps two fan runs "
                         "from overwriting each other)")
    a = ap.parse_args(argv)
    variants = tuple(v.strip() for v in a.variants.split(","))
    e, L, eps = E._load_eps(a.model, "cuda", 1)
    batches = tuple(int(b) for b in a.batches.split(","))
    out = {"key": a.model, "ckpt_step": L["step"], "env": E._env(),
           "question": "what does a K-candidate plan fan actually cost?",
           "gpu_state_before": E._gpu_state(),
           "protocol": {"batches": list(batches), "iters": a.iters,
                        "warmup": a.warmup, "rollout_k": E.K_MAX,
                        "window": E.WINDOW, "variants": list(variants)}}
    for prec in (p.strip() for p in a.precision.split(",")):
        out[prec] = E.fan_latency(e, L, eps[0], precision=prec,
                                  batches=batches, iters=a.iters,
                                  warmup=a.warmup, variants=variants)
        for b, row in out[prec]["by_batch"].items():
            for name, cell in row.items():
                if isinstance(cell, dict) and cell.get("p50_ms"):
                    print(f"[fan] {prec:5s} K={b:>2s} {name:16s} "
                          f"p50={cell['p50_ms']:8.2f} ms "
                          f"({cell['ms_per_candidate']:7.2f} ms/candidate) "
                          f"vs naive {cell.get('naive_Kx_projection_ms')} ms "
                          f"= {cell.get('cheaper_than_naive_by')}x cheaper",
                          flush=True)
    out["gpu_state_after"] = E._gpu_state()
    ex = out["gpu_state_before"].get("exclusive")
    out["contamination_check"] = {
        "gpu_exclusive_before": ex,
        "gpu_exclusive_after": out["gpu_state_after"].get("exclusive"),
        "valid": bool(ex and out["gpu_state_after"].get("exclusive"))}
    dest = E.RES / (f"eff_levers_fan{a.tag}_{a.model}.json"
                    if out["contamination_check"]["valid"]
                    else f"eff_levers_fan{a.tag}_{a.model}.CONTAMINATED.json")
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"[fan] wrote {dest} clean={out['contamination_check']['valid']}",
          flush=True)


if __name__ == "__main__":
    main()
