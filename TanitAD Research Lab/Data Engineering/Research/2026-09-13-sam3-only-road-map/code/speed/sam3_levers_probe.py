"""SAM3 speed levers on Thor -- feasibility pass (timings plus a first numerical look; the approval test is a separate script).
All 19 prompts of one front frame (17 of the v6 extractor + 2 of the stripe pass), grounding only unless stated:
  L0   baseline: forward_text + forward_grounding per prompt, fp32 (what the pipeline runs today)
  L1   text features cached once per prompt for the whole run
  L2   sub-stage split of the grounding (prompt encoding, fusion encoder, decoder, mask head)
  L3   prompts batched in one grounding forward (B = 19 / 10 / 5), fp32
  L4   bf16 autocast grounding, B = 1 and B = 19
  L5   image backbone: fp32 vs bf16 autocast; and fp32 grounding on the bf16 backbone features
  L6   cudnn.benchmark on backbone + grounding
Numerical look vs L0 (per prompt, per query): max |delta p|, keep-decision flips at the extractor threshold, pixel disagreement of
masks kept by both at 1920x1080 (logit > 0). Usage: sam3_levers_probe.py <c8>=<j,j> [...]  (first frame = warm-up, untimed)"""
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


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


FR = []
for arg in sys.argv[1:]:
    c8, fr = arg.split("=")
    sd = ROOT / f"seq_{c8}"; toks = sorted(p.name for p in sd.iterdir() if p.is_dir())
    FR += [(c8, int(j), Image.open(sd / toks[int(j)] / "images" / "CAM_FW.jpg").convert("RGB")) for j in fr.split(",")]

with torch.inference_mode():
    TXT = {p: m.backbone.forward_text([p], device="cuda") for p in P}
print("text keys/shapes:", {k: tuple(v.shape) for k, v in TXT[P[0]].items()}, flush=True)

# ---- sub-stage timers (instance attributes shadow the class methods forward_grounding calls)
STAGE = {}


def timed(name, fn):
    def w(*a, **k):
        t = sync(); r = fn(*a, **k); STAGE[name] = STAGE.get(name, 0.0) + sync() - t; return r
    return w


ORIG = {n: getattr(m, n) for n in ("_encode_prompt", "_run_encoder", "_run_decoder", "_run_segmentation_heads")}


def stages(on):
    for n, f in ORIG.items():
        if on:
            setattr(m, n, timed(n, f))
        elif n in m.__dict__:
            delattr(m, n)


@torch.inference_mode()
def ground(bo, ps, bf16=False):
    n = len(ps)
    b = dict(bo)
    for k in ("language_features", "language_embeds"):
        b[k] = torch.cat([TXT[p][k] for p in ps], 1)
    b["language_mask"] = torch.cat([TXT[p]["language_mask"] for p in ps], 0)
    fs = dataclasses.replace(proc.find_stage, img_ids=torch.zeros(n, dtype=torch.long, device="cuda"), text_ids=torch.arange(n, device="cuda"))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        out = m.forward_grounding(backbone_out=b, find_input=fs, geometric_prompt=m._get_dummy_prompt(n), find_target=None)
    return out


@torch.inference_mode()
def rows(out, ps):
    """per prompt: probs [200] (float32) and raw mask logits [200,h,w]"""
    pr = (out["pred_logits"].float().sigmoid() * out["presence_logit_dec"].float().sigmoid().unsqueeze(1)).squeeze(-1)
    return {p: (pr[i], out["pred_masks"][i]) for i, p in enumerate(ps)}


@torch.inference_mode()
def run_cfg(img_bo, cfg):
    """returns per-prompt rows and the grounding time for the 19 prompts"""
    res = {}
    t0 = sync()
    if cfg["mode"] == "L0":
        for p in P:
            txt = m.backbone.forward_text([p], device="cuda")
            b = dict(img_bo); b.update(txt)
            out = m.forward_grounding(backbone_out=b, find_input=proc.find_stage, geometric_prompt=m._get_dummy_prompt(), find_target=None)
            res.update(rows(out, [p]))
    else:
        B = cfg.get("B", 1)
        for s in range(0, len(P), B):
            ps = P[s:s + B]
            res.update(rows(ground(img_bo, ps, cfg.get("bf16", False)), ps))
    return res, sync() - t0


@torch.inference_mode()
def compare(ref, got, H, W):
    dmax, flips, kept, dis, uni = 0.0, 0, 0, 0, 0
    for p in P:
        pr, mr = ref[p]; pg, mg = got[p]
        dmax = max(dmax, float((pr - pg).abs().max()))
        kr, kg = pr >= THR[p], pg >= THR[p]
        flips += int((kr ^ kg).sum()); both = torch.nonzero(kr & kg).flatten()
        kept += int(kr.sum())
        if len(both):
            a = interpolate(mr[both].float().unsqueeze(1), (H, W), mode="bilinear", align_corners=False) > 0
            g = interpolate(mg[both].float().unsqueeze(1), (H, W), mode="bilinear", align_corners=False) > 0
            dis += int((a ^ g).sum()); uni += int((a | g).sum())
    return {"max_dp": dmax, "keep_flips": flips, "kept_ref": kept, "mask_px_disagree_over_union": dis / max(uni, 1)}


CFGS = [{"name": "L1 cached text B1 fp32", "mode": "L1"}, {"name": "L3 B19 fp32", "mode": "L3", "B": 19}, {"name": "L3 B10 fp32", "mode": "L3", "B": 10},
        {"name": "L3 B5 fp32", "mode": "L3", "B": 5}, {"name": "L4 B1 bf16", "mode": "L4", "bf16": True}, {"name": "L4 B19 bf16", "mode": "L4", "B": 19, "bf16": True}]
T = {c["name"]: [] for c in CFGS}; T["L0 baseline"] = []; T["set_image fp32"] = []; T["set_image bf16"] = []; T["L5 fp32 grounding on bf16 backbone"] = []
T["L6 cudnn.benchmark set_image"] = []; T["L6 cudnn.benchmark L1"] = []
NUM = {k: [] for k in T}; MEM = {}; SPLIT = {}
for fi, (c8, j, img) in enumerate(FR):
    warm = fi == 0
    H, W = img.height, img.width
    with torch.inference_mode():
        x = sync(); st = proc.set_image(img); t_img = sync() - x
        bo = st["backbone_out"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = sync(); st16 = proc.set_image(img, state={}); t_img16 = sync() - x
        bo16 = {k: v for k, v in st16["backbone_out"].items()}
    if not warm:
        T["set_image fp32"].append(t_img); T["set_image bf16"].append(t_img16)
        fp = [float((a.float() - b.float()).abs().max() / (a.float().abs().max() + 1e-9)) for a, b in zip(bo["backbone_fpn"], bo16["backbone_fpn"])]
        NUM["set_image bf16"].append({"fpn_rel_maxdiff": max(fp)})
    ref, t0 = run_cfg(bo, {"mode": "L0"})
    if not warm:
        T["L0 baseline"].append(t0)
    for c in CFGS:
        torch.cuda.reset_peak_memory_stats()
        got, t = run_cfg(bo, c)
        MEM[c["name"]] = max(MEM.get(c["name"], 0.0), torch.cuda.max_memory_allocated() / 1e9)
        if not warm:
            T[c["name"]].append(t); NUM[c["name"]].append(compare(ref, got, H, W))
    # L5: fp32 grounding on bf16 backbone features (cast back to fp32)
    bo16f = {k: ([t.float() for t in v] if isinstance(v, list) and v and torch.is_tensor(v[0]) else (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)) for k, v in bo16.items()}
    got, t = run_cfg(bo16f, {"mode": "L1"})
    if not warm:
        T["L5 fp32 grounding on bf16 backbone"].append(t); NUM["L5 fp32 grounding on bf16 backbone"].append(compare(ref, got, H, W))
    # L2 split for B1 fp32, B19 fp32, B1 bf16
    if not warm:
        for name, cfg in (("B1 fp32", {"mode": "L1"}), ("B19 fp32", {"mode": "L3", "B": 19}), ("B1 bf16", {"mode": "L4", "bf16": True}), ("B19 bf16", {"mode": "L4", "B": 19, "bf16": True})):
            STAGE.clear(); stages(True); run_cfg(bo, cfg); stages(False)
            for k, v in STAGE.items():
                SPLIT.setdefault(name, {}).setdefault(k, []).append(v)
    # L6 cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    with torch.inference_mode():
        x = sync(); stb = proc.set_image(img, state={}); tb = sync() - x
    got, t = run_cfg(stb["backbone_out"], {"mode": "L1"})
    if fi > 0:                                            # benchmark mode autotunes on first sight of a shape
        T["L6 cudnn.benchmark set_image"].append(tb); T["L6 cudnn.benchmark L1"].append(t)
        NUM["L6 cudnn.benchmark L1"].append(compare(ref, got, H, W))
    torch.backends.cudnn.benchmark = False
    print(f"{c8} {j} {'(warm-up)' if warm else ''} L0 {t0:.3f}s", flush=True)

print("\n=== grounding time per frame for 19 prompts (mean over timed frames), peak mem, numerical look vs L0")
rep = {}
for k, v in T.items():
    if not v:
        continue
    num = NUM.get(k) or []
    agg = {}
    if num:
        for key in num[0]:
            vals = [d[key] for d in num]
            agg[key] = max(vals) if key in ("max_dp", "fpn_rel_maxdiff", "mask_px_disagree_over_union") else int(sum(vals))
    rep[k] = {"s_mean": float(np.mean(v)), "n": len(v), "peak_mem_gb": MEM.get(k), **agg}
    print(f"{k:40s} {np.mean(v):7.3f}s  n={len(v)}  mem={MEM.get(k, float('nan')):.2f}GB  {agg}")
print("\n=== sub-stage split (s per frame, 19 prompts)")
for name, d in SPLIT.items():
    print(name, {k.replace('_run_', '').replace('_encode_prompt', 'prompt'): round(float(np.mean(v)), 3) for k, v in d.items()})
Path("/home/nvidia/sam3map/sam3_levers_probe.json").write_text(json.dumps({"timings": rep, "split": {n: {k: float(np.mean(v)) for k, v in d.items()} for n, d in SPLIT.items()}}, indent=1), encoding="utf-8")
print("ZZLEVERS-DONEZZ")
