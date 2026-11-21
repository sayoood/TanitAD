"""Next SAM3 levers after fp16 grounding (feasibility; approval stays with the SPEC's full-pipeline bars). On real front frames:
  S1  segmentation-head breakdown (prompt cross-attention, pixel decoder, instance head, mask predictor, semantic head) at
      fp32 B1 / fp16 B19 -- what the 0.31-0.37 s/frame that no precision change moved is made of
  S2  image backbone under fp16 autocast: time, feature deviation, and grounding deviation (fp16 B19 grounding on fp16 vs fp32
      features): max |dp|, keep flips at the extractor thresholds, kept-mask pixel disagreement
  S3  torch.compile (inductor, static shapes) of the fusion encoder, decoder and segmentation head under fp16 B19: compile
      time, steady-state grounding time, deviation vs the uncompiled fp16 B19 path
Usage: sam3_stage_probe.py <c8>=<j,j> [...]   (first frame = warm-up)"""
import dataclasses, json, sys, time
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3map/eval")
import numpy as np
import torch
from PIL import Image
import sam3map_extract_v6 as E
from sam3.model.data_misc import interpolate

ROOT = Path("/home/nvidia/sam3map/native7")
proc, _ = E.S.build(conf=0.25)
m = proc.model
PT = [(p, t) for fam, table in E.PROMPTS.items() for p, t in table.items()] + [("crosswalk stripe", 0.4), ("white stripe on road", 0.4)]
P = [p for p, _ in PT]; THR = dict(PT)
FR = []
for arg in sys.argv[1:]:
    c8, fr = arg.split("=")
    sd = ROOT / f"seq_{c8}"; toks = sorted(p.name for p in sd.iterdir() if p.is_dir())
    FR += [(c8, int(j), Image.open(sd / toks[int(j)] / "images" / "CAM_FW.jpg").convert("RGB")) for j in fr.split(",")]
with torch.inference_mode():
    TXT = {p: m.backbone.forward_text([p], device="cuda") for p in P}


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def to32(x):
    if torch.is_tensor(x):
        return x.float() if x.is_floating_point() else x
    if isinstance(x, dict):
        return {k: to32(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to32(v) for v in x)
    return x


ORIG = {n: getattr(m, n) for n in ("_encode_prompt", "_run_encoder", "_run_decoder", "_run_segmentation_heads")}


def set_half(on):
    for n, fn in ORIG.items():
        if not on:
            if n in m.__dict__:
                delattr(m, n)
            continue

        def w(*a, _fn=fn, **k):
            with torch.autocast("cuda", dtype=torch.float16):
                r = _fn(*a, **k)
            return to32(r)
        setattr(m, n, w)


@torch.inference_mode()
def ground_all(bo, B):
    res = {}
    for s in range(0, len(P), B):
        ps = P[s:s + B]; n = len(ps); b = dict(bo)
        for k in ("language_features", "language_embeds"):
            b[k] = torch.cat([TXT[p][k] for p in ps], 1)
        b["language_mask"] = torch.cat([TXT[p]["language_mask"] for p in ps], 0)
        fs = dataclasses.replace(proc.find_stage, img_ids=torch.zeros(n, dtype=torch.long, device="cuda"), text_ids=torch.arange(n, device="cuda"))
        out = m.forward_grounding(backbone_out=b, find_input=fs, geometric_prompt=m._get_dummy_prompt(n), find_target=None)
        pr = (out["pred_logits"].float().sigmoid() * out["presence_logit_dec"].float().sigmoid().unsqueeze(1)).squeeze(-1)
        for i, p in enumerate(ps):
            res[p] = (pr[i], out["pred_masks"][i].float())
    return res


@torch.inference_mode()
def compare(ref, got, H, W):
    dmax, flips, kept, dis, uni, nonfinite = 0.0, 0, 0, 0, 0, 0
    for p in P:
        pr, mr = ref[p]; pg, mg = got[p]
        nonfinite += int((~torch.isfinite(pg)).sum()) + int((~torch.isfinite(mg)).sum())
        dmax = max(dmax, float((pr - pg).abs().max()))
        kr, kg = pr >= THR[p], pg >= THR[p]; flips += int((kr ^ kg).sum()); kept += int(kr.sum())
        both = torch.nonzero(kr & kg).flatten()
        if len(both):
            a = interpolate(mr[both].unsqueeze(1), (H, W), mode="bilinear", align_corners=False).sigmoid() > 0.5
            g = interpolate(mg[both].unsqueeze(1), (H, W), mode="bilinear", align_corners=False).sigmoid() > 0.5
            dis += int((a ^ g).sum()); uni += int((a | g).sum())
    return {"max_dp": round(dmax, 4), "flips": flips, "kept": kept, "mask_dis": dis / max(uni, 1), "nonfinite": nonfinite}


# ---- S1 segmentation-head breakdown via module forward wrappers
sh = m.segmentation_head
SUB = {}


def tw(name, fn):
    def w(*a, **k):
        t = sync(); r = fn(*a, **k); SUB[name] = SUB.get(name, 0.0) + sync() - t; return r
    return w


subs = {"cross_attend_prompt": sh.cross_attend_prompt, "pixel_decoder": sh.pixel_decoder, "instance_seg_head": sh.instance_seg_head,
        "mask_predictor": sh.mask_predictor, "semantic_seg_head": sh.semantic_seg_head}
REP = {"S1": {}, "S2": [], "S3": {}}
for fi, (c8, j, img) in enumerate(FR):
    H, W = img.height, img.width
    with torch.inference_mode():
        t = sync(); bo = proc.set_image(img)["backbone_out"]; t32 = sync() - t
        with torch.autocast("cuda", dtype=torch.float16):
            t = sync(); bo16 = proc.set_image(img)["backbone_out"]; t16 = sync() - t
        bo16 = to32(bo16)
    # S1
    for name, half, B in (("fp32 B1", False, 1), ("fp16 B19", True, 19)):
        set_half(half)
        for k, mod in subs.items():
            mod.forward = tw(k, type(mod).forward.__get__(mod))
        SUB.clear(); t = sync(); ground_all(bo, B); tt = sync() - t
        for k, mod in subs.items():
            del mod.forward
        if fi > 0:
            d = REP["S1"].setdefault(name, {"total": []})
            d["total"].append(tt)
            for k, v in SUB.items():
                d.setdefault(k, []).append(v)
    # S2: fp16 B19 grounding on fp32 vs fp16 backbone features
    set_half(True)
    ref = ground_all(bo, 19); got = ground_all(bo16, 19)
    fpn = max(float((a - b).abs().max() / (a.abs().max() + 1e-9)) for a, b in zip(bo["backbone_fpn"], bo16["backbone_fpn"]))
    if fi > 0:
        REP["S2"].append({"frame": f"{c8}:{j}", "backbone_fp32_s": t32, "backbone_fp16_s": t16, "fpn_rel_maxdiff": fpn, **compare(ref, got, H, W)})
    set_half(False)
    print(f"frame {c8}:{j} done", flush=True)

# bank S1/S2 before S3: a slow or hanging compile must not take the other two results with it
print("\n== S1 segmentation head (s per frame, 19 prompts)")
for name, d in REP["S1"].items():
    print(" ", name, {k: round(float(np.mean(v)), 3) for k, v in d.items()})
print("== S2 backbone fp16 (per frame)")
for r in REP["S2"]:
    print(" ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
Path("/home/nvidia/sam3map/sam3_stage_probe_s1s2.json").write_text(json.dumps({"S1": {n: {k: float(np.mean(v)) for k, v in d.items()} for n, d in REP["S1"].items()},
                                                                              "S2": REP["S2"]}, indent=1), encoding="utf-8")
print("ZZSTAGE-S1S2-BANKEDZZ", flush=True)
# ---- S3 torch.compile of encoder / decoder / segmentation head under fp16 B19 (static shapes)
c8, j, img = FR[1]; H, W = img.height, img.width
with torch.inference_mode():
    bo = proc.set_image(img)["backbone_out"]
set_half(True)
ref = ground_all(bo, 19)
times_eager = []
for _ in range(3):
    t = sync(); ground_all(bo, 19); times_eager.append(sync() - t)
enc0, dec0, seg0 = m.transformer.encoder.forward, m.transformer.decoder.forward, m.segmentation_head.forward
try:
    m.transformer.encoder.forward = torch.compile(enc0, dynamic=False)
    m.transformer.decoder.forward = torch.compile(dec0, dynamic=False)
    m.segmentation_head.forward = torch.compile(seg0, dynamic=False)
    t = sync(); got = ground_all(bo, 19); t_compile = sync() - t
    times_c = []
    for _ in range(3):
        t = sync(); got = ground_all(bo, 19); times_c.append(sync() - t)
    REP["S3"] = {"first_call_s": t_compile, "eager_fp16_B19_s": float(np.median(times_eager)), "compiled_fp16_B19_s": float(np.median(times_c)), **compare(ref, got, H, W)}
    # second frame, same shapes: no recompile expected
    c8b, jb, imgb = FR[-1]
    with torch.inference_mode():
        bob = proc.set_image(imgb)["backbone_out"]
    t = sync(); ground_all(bob, 19); REP["S3"]["compiled_other_frame_s"] = sync() - t
except Exception as e:
    REP["S3"] = {"error": f"{type(e).__name__}: {str(e)[:400]}", "eager_fp16_B19_s": float(np.median(times_eager))}
set_half(False)

print("\n== S1 segmentation head (s per frame, 19 prompts)")
for name, d in REP["S1"].items():
    print(" ", name, {k: round(float(np.mean(v)), 3) for k, v in d.items()})
print("== S2 backbone fp16 (per frame)")
for r in REP["S2"]:
    print(" ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
print("== S3 compile", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in REP["S3"].items()})
Path("/home/nvidia/sam3map/sam3_stage_probe.json").write_text(json.dumps({"S1": {n: {k: float(np.mean(v)) for k, v in d.items()} for n, d in REP["S1"].items()},
                                                                         "S2": REP["S2"], "S3": REP["S3"]}, indent=1), encoding="utf-8")
print("ZZSTAGE-DONEZZ")
