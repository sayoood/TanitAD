"""ALL-TASK Alpamayo 2 Super logger — the augmentation-dataset inference engine.

Per clip x task: run, time, and bank the COMPLETE raw output (PI: "no exceptions").
Tasks: trajectory (inference_smoke sidecar: pred trajectories + CoC + metrics),
meta_action, auto_labeling, vqa (random question from the 506 bank, seeded per clip).
Grounding: the released text_tasks API supports only {meta_action, auto_labeling, vqa}
(TextTask literal, text_tasks.py:39) — probed at import; if absent, grounding runs as a
grounding-formatted VQA question and the row says task="grounding_via_vqa". A missing
entry point is a MEASURED fact about the release, recorded per row, never silent.

Resumable: (clip_index, task) rows already in --out are skipped. One bad row never
kills the batch. Every row carries wall_s so the 4-A40-day budget is priced from
measurement, not estimates.
"""
from __future__ import annotations

import argparse
import json
import os
import random
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
        return _CACHE[key]
    kwargs["quantization_config"] = BNB
    kwargs["device_map"] = {"": 0}
    kwargs.setdefault("dtype", torch.bfloat16)
    kwargs.setdefault("attn_implementation", "sdpa")
    t0 = time.time()
    m = _ORIG(cls, *args, **kwargs)
    print(f"[quant] loaded in {time.time()-t0:.0f}s", flush=True)
    _CACHE[key] = m
    return m


Alpamayo2Super.from_pretrained = _cached_4bit

GROUNDING_QS = [
    "Provide the 2D bounding box coordinates of the lead vehicle in the front wide camera.",
    "Provide the 2D bounding box coordinates of the nearest pedestrian, if any.",
    "Provide the 2D bounding box coordinates of the nearest traffic light.",
    "Provide the 2D bounding box coordinates of the closest vehicle in the adjacent lane.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--vqa-bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from alpamayo2_super import helper, inference_smoke, text_tasks
    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset

    text_task_names = set(getattr(text_tasks, "_TASK_PROMPTS", {}))
    grounding_native = "grounding" in text_task_names
    print(f"[api] text tasks: {sorted(text_task_names)}  grounding_native={grounding_native}",
          flush=True)

    manifest = json.load(open(a.manifest))
    bank = json.load(open(a.vqa_bank))["questions"]
    outdir = os.path.dirname(a.out) or "."
    os.makedirs(outdir, exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            if l.strip():
                r = json.loads(l)
                done.add((r["sample_index"], r["task"]))
        print(f"[resume] {len(done)} rows present", flush=True)
    fh = open(a.out, "a")

    model = None

    def bank_row(i, entry, task, t0, extra):
        rec = {"sample_index": i, "clip_id": entry["clip_id"],
               "t0_us": entry["t0_us"], "task": task,
               "wall_s": round(time.time() - t0, 1),
               "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "model_id": a.model_id, "seed": a.seed,
               "_quantisation": "NF4-backbone-4bit-UNVALIDATED",
               "_contamination": ("PhysicalAI-AV is an Alpamayo TRAINING dataset; "
                                  "overlap UNRESOLVED")}
        rec.update(extra)
        fh.write(json.dumps(rec, default=str) + "\n")
        fh.flush()
        return rec

    def run_text(task, src, question=None):
        data = select_task_input(src, task if task in ("meta_action", "auto_labeling",
                                                       "vqa") else "grounding")
        nonlocal model
        if model is None:
            model = Alpamayo2Super.from_pretrained(a.model_id, dtype=torch.bfloat16,
                                                   device_map="cuda:0")
        api_task = task if task in text_task_names else "vqa"
        kw = {"question": question} if question is not None else {}
        inputs = text_tasks.prepare_text_generation_inputs(
            data, model.config, model.tokenizer, task=api_task, **kw)
        inputs = helper.to_device(inputs, "cuda")
        torch.cuda.manual_seed_all(a.seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = text_tasks.generate_text(model, inputs, num_samples=1)
        return {"raw": {k: (v if isinstance(v, (str, int, float, list)) else str(v))
                        for k, v in out.items()},
                "api_task": api_task,
                "camera_ids": [int(c) for c in data["camera_indices"].tolist()]
                if hasattr(data.get("camera_indices"), "tolist") else None}

    for i in range(a.start, a.start + a.n):
        if i >= len(manifest):
            break
        entry = manifest[i]
        rng = random.Random(a.seed * 100003 + i)

        # ---- trajectory (inference_smoke banks pred trajs + CoC + metrics) ----
        if (i, "trajectory") not in done:
            t0 = time.time()
            try:
                sj = f"{outdir}/traj_{i:05d}.json"
                inference_smoke.run_smoke(model_id=a.model_id, clip_id=entry["clip_id"],
                                          t0_us=int(entry["t0_us"]), save_viz=None,
                                          save_json=sj, diffusion_steps=10, seed=a.seed)
                res = json.load(open(sj))
                bank_row(i, entry, "trajectory", t0, {"raw": res})
            except Exception as e:
                bank_row(i, entry, "trajectory", t0,
                         {"error": f"{type(e).__name__}: {str(e)[:300]}"})

        # ---- the text tasks share one source load --------------------------------
        src = None
        for task in ("meta_action", "auto_labeling", "vqa", "grounding"):
            if (i, task) in done and not (task == "grounding" and (i, "grounding_via_vqa") not in done):
                continue
            t0 = time.time()
            try:
                if src is None:
                    src = load_physical_aiavdataset(entry["clip_id"],
                                                    t0_us=int(entry["t0_us"]))
                q = None
                logged = task
                if task == "vqa":
                    qrec = rng.choice(bank)
                    q = qrec["question"]
                elif task == "grounding":
                    q = rng.choice(GROUNDING_QS)
                    if not grounding_native:
                        logged = "grounding_via_vqa"
                if (i, logged) in done:
                    continue
                extra = run_text(task, src, question=q)
                if q is not None:
                    extra["question"] = q
                    if task == "vqa":
                        extra["vqa_qid"] = qrec["qid"]
                        extra["vqa_category"] = qrec["category"]
                bank_row(i, entry, logged, t0, extra)
            except Exception as e:
                bank_row(i, entry, task, t0,
                         {"error": f"{type(e).__name__}: {str(e)[:300]}"})
        print(f"[{i}] done in accumulated wall; see rows", flush=True)
    print("ALLTASKS_DONE", flush=True)


if __name__ == "__main__":
    main()
