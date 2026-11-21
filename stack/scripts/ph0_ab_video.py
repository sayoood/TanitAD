"""A/B/C — does the model actually SEE the video? (PI, 2026-08-12)

⛔ WHY THIS EXISTS. The PI read the PH0 v2 outputs as "plausible" — generic
`follow_main_road` + `hold_corridor` on nearly every clip, every sign classified
`other` — and asked whether the inference path is right, pointing at the Qwen
cookbooks. It is not. Checked against QwenLM/Qwen3-VL's own examples, our call
has THREE defects:

  1. ⛔ NO ``video_metadata``. transformers warns
     "Asked to sample `fps` frames per second but no video metadata was provided
      which is required when sampling with `fps`. Defaulting to `fps=24`."
     We hand it 40 frames spanning 20 REAL seconds; with the 24 fps default the
     model believes it is watching **1.67 seconds**. A ~12x temporal error, and
     B4's entire premise is that the frames after t0 are 12 s of hindsight.
  2. Wrong key: the cookbook uses ``sample_fps`` for the image-list-as-video
     form; we passed ``fps``.
  3. We never call ``qwen_vl_utils.process_vision_info``, so no
     ``video_kwargs``, and no ``do_resize=False`` to avoid double-resizing.

This harness runs the SAME prompt on the SAME clip three ways and dumps every
raw output, so the question is settled by comparison rather than by argument:

  A_broken   what we shipped: PIL list + ``fps``, no metadata (fps=24 default)
  B_video    the cookbook path: process_vision_info + video_metadata + video_kwargs
  C_images   frames as individual IMAGES, no video semantics at all — the control
             that says whether "video" is buying us anything over a frame grid

⚠️ If A and C agree and B differs, the temporal channel was dead and the 8/8
format gate said nothing about temporal understanding. If all three agree, the
generic answers are the model's, not the plumbing's — and that is a different
(harder) problem. Both outcomes are informative; neither is assumed here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def build_video_inputs_official(processor, pil_frames, prompt, sample_fps):
    """The QwenLM/Qwen3-VL cookbook path, including video_metadata."""
    from qwen_vl_utils import process_vision_info
    msgs = [{"role": "user", "content": [
        {"type": "video", "video": pil_frames, "sample_fps": sample_fps},
        {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True,
                                         tokenize=False)
    images, videos, video_kwargs = process_vision_info(
        msgs, image_patch_size=16, return_video_kwargs=True,
        return_video_metadata=True)
    video_metadatas = None
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    return processor(text=text, images=images, videos=videos,
                     video_metadata=video_metadatas, return_tensors="pt",
                     do_resize=False, **video_kwargs)


def build_video_inputs_broken(processor, pil_frames, prompt, sample_fps):
    """Exactly what ph0_v2 shipped — kept verbatim so the A/B is honest."""
    msgs = [{"role": "user", "content": [
        {"type": "video", "video": pil_frames, "fps": sample_fps},
        {"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True,
                                         tokenize=True, return_dict=True,
                                         return_tensors="pt")


def build_image_inputs(processor, pil_frames, prompt, _fps):
    """No video semantics — a numbered grid of stills."""
    content = []
    for i, im in enumerate(pil_frames):
        content.append({"type": "text", "text": f"frame {i}:"})
        content.append({"type": "image", "image": im})
    content.append({"type": "text", "text": prompt})
    msgs = [{"role": "user", "content": content}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True,
                                         tokenize=True, return_dict=True,
                                         return_tensors="pt")


ARMS = {"A_broken": build_video_inputs_broken,
        "B_video": build_video_inputs_official,
        "C_images": build_image_inputs}

# A prompt whose answer is only knowable if the TIME BASE is right. If the model
# thinks the clip is 1.67 s it cannot say 20 s, and the arms separate.
P_TIME = """You are shown a driving video.
How many SECONDS of real time does this video span in total, and how many
distinct frames were you given? Answer with JSON only:
{"seconds": <number>, "n_frames": <int>, "basis": "<one short phrase>"}"""

P_SCENE = """You are shown a driving video: the first half is BEFORE a decision
time, the second half is AFTER it.
Describe what CHANGES between the first half and the second half — motion,
speed, direction. Answer with JSON only:
{"changes": "<one sentence>", "ego_turns": "left|right|none",
 "ego_speed": "increases|decreases|steady"}"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("ph0_ab_video")
    ap.add_argument("--clips", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--arm", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--sample-fps", type=float, default=2.0)
    ap.add_argument("--max-new", type=int, default=160)
    a = ap.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ph0_pilot import sample_clip_frames

    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(a.arm, trust_remote_code=True)
    model, auto_class = None, None
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(a.arm, torch_dtype="auto",
                                        device_map="cuda:0",
                                        trust_remote_code=True)
            auto_class = name
            break
        except Exception as e:
            print(f"[ab] {name} failed: {type(e).__name__}: {e}", flush=True)
    if model is None:
        raise SystemExit("[ab] no usable auto-class")
    model.eval()
    tok = getattr(proc, "tokenizer", proc)
    print(f"[ab] {auto_class} · transformers {transformers.__version__}",
          flush=True)

    os.makedirs(a.out, exist_ok=True)
    clips = json.load(open(a.clips))[:a.n]
    rows = []
    for cid in clips:
        frames, _t, n_past = sample_clip_frames(
            os.path.join(a.video_root, f"{cid}.mp4"), t0_s=8.0)
        pil = [Image.fromarray(f) for f in frames]
        true_span_s = len(frames) / a.sample_fps
        for pname, prompt in (("time", P_TIME), ("scene", P_SCENE)):
            for arm, builder in ARMS.items():
                t0 = time.time()
                try:
                    inputs = builder(proc, pil, prompt, a.sample_fps)
                    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
                              for k, v in inputs.items()}
                    n_in = inputs["input_ids"].shape[1]
                    with torch.no_grad():
                        out = model.generate(**inputs,
                                             max_new_tokens=a.max_new,
                                             do_sample=False)
                    raw = tok.decode(out[0, n_in:], skip_special_tokens=True)
                    err = None
                except Exception as e:
                    raw, n_in, err = "", 0, f"{type(e).__name__}: {e}"[:180]
                rows.append({"clip": str(cid)[:8], "prompt": pname, "arm": arm,
                             "n_input_tokens": n_in,
                             "raw": raw.strip()[:400], "error": err,
                             "wall_s": round(time.time() - t0, 1)})
                print(f"[ab] {str(cid)[:8]} {pname} {arm} "
                      f"tok={n_in} {'ERR' if err else 'ok'}", flush=True)

    res = {"arm_model": a.arm, "auto_class": auto_class,
           "sample_fps": a.sample_fps, "n_frames": len(frames),
           "true_span_s": true_span_s,
           "_note": "A_broken omits video_metadata so transformers defaults to "
                    "fps=24; B_video is the QwenLM cookbook path; C_images has "
                    "no video semantics at all",
           "rows": rows}
    json.dump(res, open(os.path.join(a.out, "ab_video.json"), "w"), indent=1)
    print(f"[ab] {len(rows)} rows · true span {true_span_s:.1f}s", flush=True)
    print("AB_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
