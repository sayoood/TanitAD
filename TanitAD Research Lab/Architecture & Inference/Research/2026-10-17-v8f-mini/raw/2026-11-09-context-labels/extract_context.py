"""Context label extraction. Stage A: decode + Tier 1 image statistics (CPU) + Tier 2 SigLIP2 zero-shot and embeddings (GPU).
Stage B: Tier 3 Qwen2.5-VL-3B (4-bit) constrained-choice scores at 1 Hz (GPU). One small npz per clip, resumable per clip.

Run with C:/Users/Admin/venvs/tanitad/Scripts/python.exe and PYTHONPATH=F:/strat_models/pylibs (bitsandbytes, installed --no-deps).
Every GPU stage takes the shared dev-box GPU lock (gpu_lock.py) and gives it back after --budget-min minutes.
Labels are build-time only; none is an inference input.

usage: extract_context.py --stage A|B --clips clips.json --out F:/strat_out [--budget-min 30] [--no-lock] [--limit N]
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strat_common as C

H, W = 252, 448
DEV = os.environ.get("STRAT_DEVICE", "cuda")      # "cpu" only for plumbing tests that must not touch the shared GPU
SIG_DIR = "F:/strat_models/siglip2"
QWEN_DIR = "F:/strat_models/qwen25vl3b"
QWEN_NF4 = "F:/strat_models/qwen25vl3b_nf4"
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

ILLUM = ("dark", "lit_night", "twilight", "day", "glare")
WEATHER = ("clear", "rain", "snow", "fog")
SURFACE = ("dry", "wet", "snow_ice")
VISIB = ("good", "reduced", "poor")
VIEWQ = ("clean", "wiper", "glare", "occluded", "tunnel")

SIG_PROMPTS = {
    "illum": ["a dashcam photo at night on a dark road with no street lighting", "a dashcam photo at night on a street lit by street lamps",
              "a dashcam photo at dusk or dawn in twilight", "a dashcam photo in daylight",
              "a dashcam photo with strong sun glare and blown-out highlights"],
    "weather": ["a dashcam photo in clear weather", "a dashcam photo in the rain with raindrops", "a dashcam photo in snowy weather",
                "a dashcam photo in fog or mist"],
    "surface": ["a dashcam photo of a dry road", "a dashcam photo of a wet road with reflections", "a dashcam photo of a snow or ice covered road"],
    "viewq": ["a dashcam photo through a clean windshield", "a dashcam photo with a windshield wiper or raindrops on the glass",
              "a dashcam photo with strong glare", "a dashcam photo with the view blocked by a truck or the car hood",
              "a dashcam photo inside a tunnel"],
}
VLM_Q = {
    "weather": "What is the weather in this front-camera driving image? A. clear or overcast, no precipitation. B. rain. C. snow. D. fog or mist. Answer with one letter.",
    "surface": "What is the state of the road surface in this driving image? A. dry. B. wet. C. covered with snow or ice. Answer with one letter.",
    "visibility": "How is the visibility in this driving image? A. good, far objects are clearly visible. B. reduced, haze, glare, darkness or spray limit the view. C. poor, very short view distance. Answer with one letter.",
    "vis2": "Is the air clear in this driving image? A. yes, the air is clear. B. no, there is fog, haze, heavy rain, falling snow or spray in the air. C. no, and it is so dense that only a few metres can be seen. Answer with one letter.",
}
VLM_NCLS = {"weather": 4, "surface": 3, "visibility": 3, "vis2": 3}
PROMPT_SHA = hashlib.sha256(json.dumps([SIG_PROMPTS, VLM_Q], sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ decode and Tier 1
def decode(clip_id, every=6, threads=4):
    """frames at video frame 0, every, 2*every ... as uint8 [n,H,W,3]; no frame file is written."""
    n_expect = 101 if every == 6 else 21
    cmd = [FFMPEG, "-v", "error", "-threads", str(threads), "-i", f"{os.environ.get('STRAT_CAM_DIR', C.CAM_DIR)}/{clip_id}.mp4", "-vf",
           f"select=not(mod(n\\,{every})),scale={W}:{H}:flags=area", "-fps_mode", "passthrough", "-frames:v", str(n_expect),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg rc {r.returncode}: {r.stderr[-200:]!r}")
    fr = np.frombuffer(r.stdout, np.uint8)
    n = fr.size // (H * W * 3)
    return fr[: n * H * W * 3].reshape(n, H, W, 3)


def tier1(fr):
    """Image statistics per frame (definitions are ours; luma = mean of R, G, B as in D3's brightness). Region 'road' = rows 84..209
    (below the upper third, above the bottom strip); 'sky' = rows 0..83."""
    f = fr.astype(np.float32)
    luma = f.mean(-1)
    road, sky = luma[:, 84:210], luma[:, :84]
    mx, mn = f.max(-1), f.min(-1)
    sat = ((mx - mn) / np.maximum(mx, 1.0))[:, :210]
    q = np.percentile(road.reshape(len(fr), -1), [5, 95], axis=1)
    return {"luma_mean": road.mean((1, 2)), "luma_p05": q[0], "luma_p95": q[1], "sky_luma": sky.mean((1, 2)), "sat_mean": sat.mean((1, 2)),
            "headlight_frac": (road >= 235).mean((1, 2)), "lit_frac": (road >= 150).mean((1, 2))}


# ------------------------------------------------------------------------------------------------ Tier 2
class Sig:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.model = AutoModel.from_pretrained(SIG_DIR, dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()
        proc = AutoProcessor.from_pretrained(SIG_DIR)
        self.txt = {}
        with torch.no_grad():
            for k, ps in SIG_PROMPTS.items():
                t = proc.tokenizer(ps, padding="max_length", max_length=64, return_tensors="pt").to(DEV)
                e = self.model.get_text_features(**t)
                e = e.pooler_output if hasattr(e, "pooler_output") else e
                self.txt[k] = torch.nn.functional.normalize(e.float(), dim=-1)
        self.scale = float(self.model.logit_scale.exp()) if hasattr(self.model, "logit_scale") else 100.0
        self.bias = float(self.model.logit_bias) if hasattr(self.model, "logit_bias") else 0.0

    def run(self, fr, bs=32):
        torch = self.torch
        probs = {k: [] for k in self.txt}
        embs = []
        with torch.no_grad():
            for i in range(0, len(fr), bs):
                x = torch.from_numpy(fr[i:i + bs]).to(DEV).permute(0, 3, 1, 2).to(self.model.dtype)
                x = torch.nn.functional.interpolate(x, size=(384, 384), mode="bilinear", antialias=True, align_corners=False)
                x = (x / 255.0 - 0.5) / 0.5
                e = self.model.get_image_features(pixel_values=x)
                e = e.pooler_output if hasattr(e, "pooler_output") else e
                e = torch.nn.functional.normalize(e.float(), dim=-1)
                embs.append(e.half().cpu().numpy())
                for k, t in self.txt.items():
                    lg = self.scale * e @ t.T + self.bias
                    probs[k].append(torch.softmax(lg, -1).cpu().numpy())
        return {k: np.concatenate(v).astype(np.float32) for k, v in probs.items()}, np.concatenate(embs)


# ------------------------------------------------------------------------------------------------ Tier 3
class Qwen:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        dm = {"": 0 if DEV == "cuda" else "cpu"}
        # MEASURED 2026-11-08: loading + quantizing the 7 GB fp16 shards from F: took 22 min (cold, 1.7 GB free RAM), which would eat the whole
        # 30-minute lock budget. The first load therefore saves the quantized weights (about 2 GB) and every later job loads those.
        if os.path.exists(QWEN_NF4 + "/NF4_DONE"):
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_NF4, device_map=dm, dtype=torch.float16).eval()
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_DIR, quantization_config=q, device_map=dm,
                                                                           dtype=torch.float16, low_cpu_mem_usage=True).eval()
            if DEV == "cuda":
                try:
                    self.model.save_pretrained(QWEN_NF4)
                    open(QWEN_NF4 + "/NF4_DONE", "w").write("ok")
                except Exception as e:      # a failed save must not cost the job
                    print("nf4 save failed:", repr(e), flush=True)
        self.proc = AutoProcessor.from_pretrained(QWEN_DIR, min_pixels=W * H, max_pixels=W * H)
        self.proc.tokenizer.padding_side = "left"
        self.letter_ids = [self.proc.tokenizer.encode(c, add_special_tokens=False)[0] for c in "ABCDE"]

    def score(self, fr, bs=12):
        """fr uint8 [n,H,W,3]. Returns {field: [n, ncls]} probabilities over the answer letters (constrained choice)."""
        from PIL import Image
        torch = self.torch
        out = {k: np.zeros((len(fr), n), np.float32) for k, n in VLM_NCLS.items()}
        jobs = [(i, k) for i in range(len(fr)) for k in VLM_Q]
        for s in range(0, len(jobs), bs):
            chunk = jobs[s:s + bs]
            texts, imgs = [], []
            for i, k in chunk:
                msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": VLM_Q[k]}]}]
                texts.append(self.proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
                imgs.append(Image.fromarray(fr[i]))
            inp = self.proc(text=texts, images=imgs, return_tensors="pt", padding=True).to(DEV)
            with torch.no_grad():
                lg = self.model(**inp).logits[:, -1, :].float()
            for j, (i, k) in enumerate(chunk):
                n = VLM_NCLS[k]
                out[k][i] = torch.softmax(lg[j, self.letter_ids[:n]], -1).cpu().numpy()
        return out


# ------------------------------------------------------------------------------------------------ drivers
def part_path(out, stage, clip):
    return f"{out}/ctx{stage}/{clip}.npz"


def replace_retry(tmp, path, tries=40):
    """MEASURED 2026-11-08: os.replace on F: raised WinError 32 (a scanner or indexer briefly holds a new file); wait for it instead of losing the job."""
    for i in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.5)


def save_atomic(path, **arrs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    replace_retry(tmp, path)


QWEN_MIN_AVAIL_GB = float(os.environ.get("STRAT_MIN_AVAIL_GB", "9.0"))   # MM 2026-11-08: the Qwen load crashed natively (0xc0000005 in torch_cpu.dll) with 3.5 GB available; wait BEFORE taking the GPU lock.


def wait_ram(min_gb, out, max_wait_s=12 * 3600):
    import psutil
    t0 = time.time()
    while True:
        avail = psutil.virtual_memory().available / 2 ** 30
        try:                                     # the Master Mind may change the gate while this waits
            min_gb = float(open("F:/strat_out/MIN_AVAIL_GB").read())
        except (OSError, ValueError):
            pass
        if int(time.time() - t0) % 300 < 60:
            print(f"[ram] waiting: {avail:.1f} GB available, gate {min_gb} GB", flush=True)
        if avail >= min_gb:
            print(f"[ram] {avail:.1f} GB available after {time.time() - t0:.0f}s", flush=True)
            return
        if os.path.exists("F:/strat_out/STOP") or time.time() - t0 > max_wait_s:
            raise SystemExit(f"[ram] gave up at {avail:.1f} GB available")
        time.sleep(60)


def run(stage, clips, out, budget_min, use_lock, limit):
    todo = [c for c in clips if not os.path.exists(part_path(out, stage, c))]
    if stage == "B" and todo:
        wait_ram(QWEN_MIN_AVAIL_GB, out)
    if limit:
        todo = todo[:limit]
    print(json.dumps({"stage": stage, "todo": len(todo), "of": len(clips)}), flush=True)
    if not todo:
        return 0
    import gpu_lock
    body = gpu_lock.acquire() if use_lock else None
    t_start = time.time()
    done = 0
    try:
        model = Sig() if stage == "A" else Qwen()
        print(json.dumps({"stage": stage, "model_load_s": round(time.time() - t_start)}), flush=True)
        every = 6 if stage == "A" else 30
        pool = cf.ThreadPoolExecutor(2)
        futs = {}
        it = iter(todo)

        def sub():
            c = next(it, None)
            if c is not None:
                futs[c] = pool.submit(decode, c, every)
        sub(); sub()
        while futs and time.time() - t_start < budget_min * 60 and not os.path.exists("F:/strat_out/STOP"):
            c = next(iter(futs))
            t0 = time.time()
            fr = futs.pop(c).result()
            if os.path.exists(part_path(out, stage, c)):      # finished by another job while this one waited for the lock
                sub()
                continue
            t_dec = time.time() - t0
            sub()
            t1 = time.time()
            if stage == "A":
                st = tier1(fr)
                t_st = time.time() - t1
                pr, emb = model.run(fr)
                save_atomic(part_path(out, "A", c), n_frames=np.int32(len(fr)), emb=emb, **{f"t1_{k}": v.astype(np.float32) for k, v in st.items()},
                            **{f"z_{k}": v for k, v in pr.items()})
                tm = {"decode_wait_s": t_dec, "stats_s": t_st, "siglip_s": time.time() - t1 - t_st}
            else:
                sc = model.score(fr)
                save_atomic(part_path(out, "B", c), n_frames=np.int32(len(fr)), **{f"v_{k}": v for k, v in sc.items()})
                tm = {"decode_wait_s": t_dec, "qwen_s": time.time() - t1}
            done += 1
            print(json.dumps({"clip": c, **{k: round(v, 2) for k, v in tm.items()}}), flush=True)
    finally:
        if body:
            gpu_lock.release(body)
    print(json.dumps({"stage": stage, "done_this_job": done, "elapsed_s": round(time.time() - t_start)}), flush=True)
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "B"])
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", default="F:/strat_out")
    ap.add_argument("--budget-min", type=float, default=30)
    ap.add_argument("--no-lock", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run(a.stage, json.load(open(a.clips))["clips"], a.out, a.budget_min, not a.no_lock, a.limit)
