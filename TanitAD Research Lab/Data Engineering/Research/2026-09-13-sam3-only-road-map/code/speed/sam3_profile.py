"""Where does SAM3's time go per front-camera frame on Thor? (PI 2026-09-14: "We need to optimize the inference time of sam3 on
thor, I need concrete approved measures".) Synchronised CUDA timing of every stage the map pipeline runs per frame:
set_image (image backbone), and for each of the 19 prompts (17 of the v6 extractor + 2 of the stripe pass): text encoder
(backbone.forward_text), grounding (model.forward_grounding: prompt encoding, fusion encoder, decoder, mask head), post-processing
(score threshold, mask upsampling to 1920x1080, sigmoid, > 0.5) and the device-to-host copy the extractor does (.cpu().numpy()).
Plus the classify() CPU logic (everything in classify() that is not a SAM3 call). Environment facts printed first.
Usage: sam3_profile.py <c8>=<frame,frame,...> [...]"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3map/eval")
import numpy as np
import torch
from PIL import Image
import sam3map_extract_v6 as E
from sam3.model import box_ops
from sam3.model.data_misc import interpolate

ROOT = Path("/home/nvidia/sam3map/native7")
proc, fix = E.S.build(conf=0.25)
m = proc.model
dt = {p.dtype for p in m.parameters()}
print("torch", torch.__version__, "| device", torch.cuda.get_device_name(0), "| param dtypes", dt, "| n params %.0f M" % (sum(p.numel() for p in m.parameters()) / 1e6),
      "| bf16 native", torch.cuda.is_bf16_supported(including_emulation=False), "| cudnn", torch.backends.cudnn.version(),
      "| flash sdp", torch.backends.cuda.flash_sdp_enabled(), "| mem-eff sdp", torch.backends.cuda.mem_efficient_sdp_enabled(), flush=True)
PROMPTS = [(p, t) for fam, table in E.PROMPTS.items() for p, t in table.items()] + [("crosswalk stripe", 0.4), ("white stripe on road", 0.4)]


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def one_prompt(state, prompt):
    t0 = sync()
    text = m.backbone.forward_text([prompt], device="cuda")
    t1 = sync()
    state["backbone_out"].update(text)
    if "geometric_prompt" not in state:
        state["geometric_prompt"] = m._get_dummy_prompt()
    out = m.forward_grounding(backbone_out=state["backbone_out"], find_input=proc.find_stage, geometric_prompt=state["geometric_prompt"], find_target=None)
    t2 = sync()
    probs = out["pred_logits"].sigmoid() * out["presence_logit_dec"].sigmoid().unsqueeze(1)
    probs = probs.squeeze(-1)
    keep = probs > proc.confidence_threshold
    masks = out["pred_masks"][keep]
    n = int(keep.sum())
    if n:
        masks = interpolate(masks.unsqueeze(1), (state["original_height"], state["original_width"]), mode="bilinear", align_corners=False).sigmoid() > 0.5
    t3 = sync()
    if n:
        _ = masks.cpu().numpy(); _ = probs[keep].float().cpu().numpy()
    t4 = sync()
    return {"text": t1 - t0, "ground": t2 - t1, "post": t3 - t2, "d2h": t4 - t3, "n": n}


rows, cls_rows = [], []
for arg in sys.argv[1:]:
    c8, frames = arg.split("=")
    sd = ROOT / f"seq_{c8}"; toks = sorted(p.name for p in sd.iterdir() if p.is_dir())
    for j in [int(x) for x in frames.split(",")]:
        fd = sd / toks[j]
        img = Image.open(fd / "images" / "CAM_FW.jpg").convert("RGB")
        t0 = sync()
        state = proc.set_image(img)
        t1 = sync()
        per = {p: one_prompt(state, p) for p, _ in PROMPTS}
        rows.append({"clip": c8, "frame": j, "set_image": t1 - t0, "prompts": per})
        # classify() end to end (its own SAM3 calls) to get the CPU share
        c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
        grid, fb = E.P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64)); sg = E.GS.smooth_grid(grid, fb)
        C = E.CM.Camera.from_calib(c, fr["cam_order"].index("CAM_FW"), img.width, img.height)
        calls = {"t": 0.0}
        orig_inst = E.instances

        def timed_instances(proc_, state_, prompt_, thr_):
            a = sync(); r = orig_inst(proc_, state_, prompt_, thr_); calls["t"] += sync() - a; return r
        E.instances = timed_instances
        a = sync(); orig_set = proc.set_image
        b0 = {"t": 0.0}

        def timed_set(img_, state=None):
            x = sync(); r = orig_set(img_, state); b0["t"] += sync() - x; return r
        proc.set_image = timed_set
        E.classify(proc, img, (C, sg), True)
        tot = sync() - a
        E.instances = orig_inst; proc.set_image = orig_set
        cls_rows.append({"classify_total": tot, "sam3_calls": calls["t"] + b0["t"], "cpu_logic": tot - calls["t"] - b0["t"]})
        print(f"{c8} {j}: set_image {rows[-1]['set_image']:.3f}s | text {sum(v['text'] for v in per.values()):.3f} ground {sum(v['ground'] for v in per.values()):.3f} "
              f"post {sum(v['post'] for v in per.values()):.3f} d2h {sum(v['d2h'] for v in per.values()):.3f} | classify {tot:.3f} (cpu logic {cls_rows[-1]['cpu_logic']:.3f})", flush=True)
rows_w = rows[1:] if len(rows) > 2 else rows                                  # first frame warms CUDA kernels
agg = {"set_image": float(np.mean([r["set_image"] for r in rows_w]))}
for k in ("text", "ground", "post", "d2h"):
    agg[k] = float(np.mean([sum(v[k] for v in r["prompts"].values()) for r in rows_w]))
agg["prompts_total"] = agg["text"] + agg["ground"] + agg["post"] + agg["d2h"]
agg["per_prompt_ground_mean"] = float(np.mean([v["ground"] for r in rows_w for v in r["prompts"].values()]))
agg["instances_per_frame"] = float(np.mean([sum(v["n"] for v in r["prompts"].values()) for r in rows_w]))
cw = cls_rows[1:] if len(cls_rows) > 2 else cls_rows
agg["classify_total"] = float(np.mean([r["classify_total"] for r in cw])); agg["classify_cpu_logic"] = float(np.mean([r["cpu_logic"] for r in cw]))
print("MEAN per frame (warm):", {k: round(v, 3) for k, v in agg.items()})
Path("/home/nvidia/sam3map/sam3_profile.json").write_text(json.dumps({"env": {"torch": torch.__version__, "param_dtypes": str(dt)}, "mean": agg, "rows": rows, "classify": cls_rows}, indent=1, default=float), encoding="utf-8")
print("ZZPROFILE-DONEZZ")
