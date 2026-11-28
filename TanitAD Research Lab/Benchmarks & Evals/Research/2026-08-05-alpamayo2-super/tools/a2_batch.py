"""Batch Alpamayo 2 Super (4-bit) over OUR OOD-val clips, one model load.

⛔ WHY OUR CLIPS AND NOT THEIRS. The only comparison worth GPU time is
Alpamayo vs our flagship on the SAME clips at the SAME t0. Their
`validation_samples.json` is a curated set we have no flagship numbers for.
Ours is the 290-clip PhysicalAI official OOD-val split our arm was scored on.

⛔ THE CONTAMINATION CAVEAT TRAVELS WITH EVERY ROW. Alpamayo lists
PhysicalAI-AV as a TRAINING dataset and this corpus is that dataset's own val
split. NVIDIA do not state whether they excluded it. If they did not, these
clips may be INSIDE Alpamayo's training set and every number here is
contaminated in Alpamayo's favour. Recorded per row, not in a footnote.

⛔ QUANTISED-4BIT-UNVALIDATED. NF4 backbone; not an NVIDIA-validated config.
May not be compared to their published minADE_6 0.911 m.

The model is loaded ONCE and cached: their CLI reloads per invocation, which at
~4.5 min/load off MooseFS would dominate the run.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch
from transformers import BitsAndBytesConfig

SKIP = ["visual", "lm_head", "expert", "action_in_proj", "action_out_proj"]
BNB = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=SKIP,
)

from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super  # noqa: E402

_ORIG = Alpamayo2Super.from_pretrained.__func__
_CACHE = {}


@classmethod
def _cached_4bit(cls, *args, **kwargs):
    key = str(args[0]) if args else kwargs.get("pretrained_model_name_or_path")
    if key in _CACHE:
        print("[quant] reusing cached 4-bit model", flush=True)
        return _CACHE[key]
    kwargs["quantization_config"] = BNB
    kwargs["device_map"] = {"": 0}
    kwargs.setdefault("dtype", torch.bfloat16)
    kwargs.setdefault("attn_implementation", "sdpa")
    print(f"[quant] NF4 backbone · BF16 skip {SKIP}", flush=True)
    t0 = time.time()
    m = _ORIG(cls, *args, **kwargs)
    print(f"[quant] loaded in {time.time()-t0:.0f}s · "
          f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
    _CACHE[key] = m
    return m


Alpamayo2Super.from_pretrained = _cached_4bit


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()

    from alpamayo2_super import inference_smoke

    manifest = json.load(open(a.manifest))
    n = min(a.n, len(manifest) - a.start)
    outdir = os.path.dirname(a.out) or "."
    os.makedirs(outdir, exist_ok=True)
    # resume: a long overnight batch must not restart from zero after a restart
    rows = []
    if os.path.exists(a.out):
        rows = [json.loads(l) for l in open(a.out) if l.strip()]
        done = {r["sample_index"] for r in rows}
        print(f"[resume] {len(done)} rows already present", flush=True)
    else:
        done = set()

    fh = open(a.out, "a")
    t_all = time.time()
    for i in range(a.start, a.start + n):
        if i in done:
            continue
        entry = manifest[i]
        t0 = time.time()
        try:
            # run_smoke takes clip_id/t0_us directly and returns None; the
            # numbers come from the sidecar it writes.
            sj = f"{outdir}/s{i:04d}.json"
            inference_smoke.run_smoke(
                model_id=a.model_id, clip_id=entry["clip_id"],
                t0_us=int(entry["t0_us"]), save_viz=None, save_json=sj,
                diffusion_steps=10, seed=42)
            res = json.load(open(sj))
        except Exception as e:
            # ⛔ one bad clip must not kill an overnight batch; the failure is a
            # ROW, so the denominator stays honest.
            print(f"[{i}] FAIL {type(e).__name__}: {str(e)[:120]}", flush=True)
            fh.write(json.dumps({"sample_index": i, "clip_id": entry["clip_id"],
                                 "t0_us": entry["t0_us"], "error":
                                 f"{type(e).__name__}: {str(e)[:200]}"}) + "\n")
            fh.flush()
            continue
        rec = {"sample_index": i, "clip_id": entry["clip_id"],
               "t0_us": entry["t0_us"], "wall_s": round(time.time() - t0, 1),
               "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "_quantisation": "NF4-backbone-4bit-UNVALIDATED",
               "_contamination": ("clip is from PhysicalAI-AV, which Alpamayo "
                                  "lists as a TRAINING dataset; overlap with "
                                  "its training split is UNRESOLVED")}
        for k in ("min_ade_m", "ade_m", "fde_at_min_ade_m",
                  "best_ade_sample_index", "cot", "num_trajectory_samples",
                  "pred_xyz_shape", "image_frames_shape"):
            if k in res:
                rec[k] = res[k]
        fh.write(json.dumps(rec, default=str) + "\n")
        fh.flush()
        print(f"[{i}] {entry['clip_id'][:8]} minADE={rec.get('min_ade_m','?')} "
              f"{rec['wall_s']}s peak={rec['peak_gib']}GiB "
              f"cot={str(rec.get('cot',''))[:60]!r}", flush=True)
    fh.close()
    print(f"[batch] {n} samples in {time.time()-t_all:.0f}s -> {a.out}",
          flush=True)


if __name__ == "__main__":
    main()
