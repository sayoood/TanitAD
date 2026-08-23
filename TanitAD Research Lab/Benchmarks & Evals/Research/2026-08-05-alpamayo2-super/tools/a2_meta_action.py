"""Alpamayo 2 Super's DECLARED manoeuvre — the missing half of the TACTICAL family.

⛔ WHY THIS EXISTS. `a2_four_families.json` reports Alpamayo's declared manoeuvre as
UNAVAILABLE, because the first pass ran the TRAJECTORY task only. Under the binding rule
of 2026-08-02 a missing family member is a WORK ITEM, not a pass — so this closes it by
running the SAME 39 clips at the SAME t0 under Alpamayo's `meta_action` text task, which
emits `["cot", "meta_action", "traj_future"]`.

⭐ WHAT IT MAKES POSSIBLE, and it is not a leaderboard row. Our own arm's declared
manoeuvre is only WEAKLY coupled to the path it drives (κ = 0.3432, MEASURED). The
question that matters is whether a 34 B model with an explicit reasoning chain is
*better coupled* — i.e. whether declaring the manoeuvre in language buys coherence
between the decision layer and the trajectory layer, or whether the drift is universal.
Either answer is informative and neither is decided by parameter count.

⛔ SAME CAVEATS AS THE TRAJECTORY PASS, carried per row: NF4-quantised backbone (not an
NVIDIA-validated configuration), and the clips are PhysicalAI-AV, which Alpamayo lists as
TRAINING data — contamination UNRESOLVED.

⚠️ Text generation is SAMPLED (`do_sample=True`, temperature 0.6 by default in
`generate_text`). A single sample is a draw, not the model's mode. The seed is set and
recorded so the run is reproducible, and `--num-samples` is exposed so the stability of
the declared manoeuvre can be measured rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from transformers import BitsAndBytesConfig

# ⛔ Identical skip list to the trajectory pass. Changing it would make the two passes
# different models and the TACTICAL comparison would no longer be within-arm.
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--indices", required=True,
                    help="JSON list of manifest indices — the SAME 39 the four-family "
                         "block was scored on, so the join is exact and not by clip name")
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from alpamayo2_super import helper, text_tasks
    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset

    manifest = json.load(open(a.manifest))
    want = json.loads(a.indices) if a.indices.strip().startswith("[") \
        else json.load(open(a.indices))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    # resume — an unattended pass must not restart from zero after a restart
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["sample_index"] for l in open(a.out) if l.strip()}
        print(f"[resume] {len(done)} rows present", flush=True)
    fh = open(a.out, "a")

    model = None
    t_all = time.time()
    for i in want:
        if i in done:
            continue
        entry = manifest[i]
        t0 = time.time()
        try:
            src = load_physical_aiavdataset(entry["clip_id"], t0_us=int(entry["t0_us"]))
            # ⛔ meta_action uses the DRIVING profile (camera ids 0,1,2,3,5,6) exactly as
            # the trajectory task does — same six cameras, same four frames. Asking for
            # the task by name rather than hand-picking cameras is what guarantees that.
            data = select_task_input(src, "meta_action")
            if model is None:
                model = Alpamayo2Super.from_pretrained(
                    a.model_id, dtype=torch.bfloat16, device_map="cuda:0")
            inputs = text_tasks.prepare_text_generation_inputs(
                data, model.config, model.tokenizer, task="meta_action")
            inputs = helper.to_device(inputs, "cuda")
            torch.cuda.manual_seed_all(a.seed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = text_tasks.generate_text(model, inputs,
                                               num_samples=a.num_samples)
        except Exception as e:
            # one bad clip is a ROW, not the end of the pass — the denominator stays honest
            print(f"[{i}] FAIL {type(e).__name__}: {str(e)[:140]}", flush=True)
            fh.write(json.dumps({"sample_index": i, "clip_id": entry["clip_id"],
                                 "error": f"{type(e).__name__}: {str(e)[:300]}"}) + "\n")
            fh.flush()
            continue
        rec = {"sample_index": i, "clip_id": entry["clip_id"],
               "t0_us": entry["t0_us"], "task": "meta_action",
               "wall_s": round(time.time() - t0, 1),
               "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "seed": a.seed, "num_samples": a.num_samples,
               "camera_ids": [int(c) for c in data["camera_indices"].tolist()]
               if hasattr(data.get("camera_indices"), "tolist") else None,
               "image_frames_shape": list(data["image_frames"].shape)
               if "image_frames" in data else None,
               "_quantisation": "NF4-backbone-4bit-UNVALIDATED",
               "_sampling": ("generate_text samples (do_sample=True, temperature 0.6). "
                             "One sample is a DRAW, not the model's mode."),
               "_contamination": ("clip is from PhysicalAI-AV, which Alpamayo lists as a "
                                  "TRAINING dataset; overlap UNRESOLVED"),
               # ⛔ the raw generation is banked verbatim. Parsing it into a class is a
               # SEPARATE, auditable step — a parser that silently mislabels would be
               # indistinguishable from a model that decided wrongly.
               "raw": {k: (v if isinstance(v, (str, int, float, list)) else str(v))
                       for k, v in out.items()},
               }
        fh.write(json.dumps(rec, default=str) + "\n")
        fh.flush()
        print(f"[{i}] {entry['clip_id'][:8]} {rec['wall_s']}s "
              f"{str(rec['raw'])[:160]}", flush=True)
    fh.close()
    print(f"[meta_action] {len(want)} requested in {time.time()-t_all:.0f}s -> {a.out}",
          flush=True)


if __name__ == "__main__":
    main()
