"""PI 2026-09-14: "Could quantization of the model help?" -- measured on Thor, nothing installed (torch._scaled_mm, native FP8 on
compute capability 11.0). The image backbone is the only compute-bound matmul-heavy part left after fp16 grounding, so FP8 goes
there: every nn.Linear of the ViT trunk (attention qkv + proj, MLP fc1 + fc2) runs as float8_e4m3fn matmul -- weights per-tensor
scaled once, activations per-tensor scaled dynamically -- with the rest of the trunk under fp16 autocast.
Arms on the same frames, backbone only: fp32 (reference) · fp16 autocast · FP8 linears + fp16 rest (per-tensor) · FP8 with
per-ROW weight scales (finer grid, same kernel). For each: backbone time, FPN feature deviation vs fp32, and the downstream
effect -- fp16 B19 grounding on those features vs on fp32 features: max |dp|, keep flips at the extractor thresholds, kept-mask
pixel disagreement (the same instance-level look used for fp16/bf16 grounding). Feasibility only; approval = SPEC bars.
Usage: sam3_fp8_probe.py <c8>=<j,j> [...]  (first frame = warm-up)"""
import dataclasses, json, sys, time
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3map/eval")
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import sam3map_extract_v6 as E
import sam3.model.vitdet as vitdet
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
E4 = torch.finfo(torch.float8_e4m3fn).max


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


# ---- FP8 linear
class FP8:
    def __init__(self, lin, rowwise):
        w = lin.weight.detach().float()
        self.rowwise = rowwise
        if rowwise:
            sw = (w.abs().amax(dim=1, keepdim=True) / E4).clamp(min=1e-12)          # [out, 1]
            self.sw = sw.t().contiguous()                                            # [1, out] for scale_b
        else:
            sw = (w.abs().max() / E4).clamp(min=1e-12); self.sw = sw
        self.w8 = (w / sw).clamp(-E4, E4).to(torch.float8_e4m3fn).t()               # [in, out]
        self.b = lin.bias.detach().half() if lin.bias is not None else None
        self.out_features = w.shape[0]

    def __call__(self, x):
        shp = x.shape; x2 = x.reshape(-1, shp[-1]).float()
        if self.rowwise:
            sx = (x2.abs().amax(dim=1, keepdim=True) / E4).clamp(min=1e-12)          # [n, 1]
        else:
            sx = (x2.abs().max() / E4).clamp(min=1e-12)
        x8 = (x2 / sx).clamp(-E4, E4).to(torch.float8_e4m3fn)
        if self.rowwise:                                                            # row-wise kernels want bf16/fp32 scales & out
            y = torch._scaled_mm(x8, self.w8, scale_a=sx, scale_b=self.sw, bias=None, out_dtype=torch.bfloat16).float()
            y = y + self.b.float() if self.b is not None else y
            y = y.half()
        else:
            y = torch._scaled_mm(x8, self.w8, scale_a=sx, scale_b=self.sw, bias=self.b, out_dtype=torch.float16)
        return y.reshape(*shp[:-1], self.out_features)


trunk = m.backbone.vision_backbone.trunk if hasattr(m.backbone, "vision_backbone") and hasattr(m.backbone.vision_backbone, "trunk") else None
if trunk is None:
    for name, mod in m.backbone.named_modules():
        if isinstance(mod, vitdet.ViT):
            trunk = mod; break
assert trunk is not None, "ViT trunk not found"
LINS = [(n, mod) for n, mod in trunk.named_modules() if isinstance(mod, torch.nn.Linear)]
print("ViT trunk", type(trunk).__name__, "linear layers", len(LINS), "params in linears %.0f M" % (sum(l.weight.numel() for _, l in LINS) / 1e6), flush=True)
MLPS = [mod for _, mod in trunk.named_modules() if isinstance(mod, vitdet.Mlp)]


def install_fp8(rowwise):
    q = {id(l): FP8(l, rowwise) for _, l in LINS}
    for _, l in LINS:
        l.forward = q[id(l)]
    for mlp in MLPS:
        def fwd(x, _m=mlp):
            x = F.gelu(_m.fc1(x)) if isinstance(_m.act, torch.nn.GELU) else _m.act(_m.fc1(x))
            return _m.drop2(_m.fc2(_m.norm(_m.drop1(x))))
        mlp.forward = fwd


def remove_fp8():
    for _, l in LINS:
        if "forward" in l.__dict__:
            del l.forward
    for mlp in MLPS:
        if "forward" in mlp.__dict__:
            del mlp.forward


ORIG = {n: getattr(m, n) for n in ("_encode_prompt", "_run_encoder", "_run_decoder", "_run_segmentation_heads")}


def half_grounding(on):
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
def ground_all(bo):
    res = {}; n = len(P); b = dict(bo)
    for k in ("language_features", "language_embeds"):
        b[k] = torch.cat([TXT[p][k] for p in P], 1)
    b["language_mask"] = torch.cat([TXT[p]["language_mask"] for p in P], 0)
    fs = dataclasses.replace(proc.find_stage, img_ids=torch.zeros(n, dtype=torch.long, device="cuda"), text_ids=torch.arange(n, device="cuda"))
    out = m.forward_grounding(backbone_out=b, find_input=fs, geometric_prompt=m._get_dummy_prompt(n), find_target=None)
    pr = (out["pred_logits"].float().sigmoid() * out["presence_logit_dec"].float().sigmoid().unsqueeze(1)).squeeze(-1)
    return {p: (pr[i], out["pred_masks"][i].float()) for i, p in enumerate(P)}


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


ARMS = ["fp32", "fp16", "fp8_tensor", "fp8_row"]
R = {a: {"t": [], "fpn": [], "cmp": []} for a in ARMS}
for fi, (c8, j, img) in enumerate(FR):
    H, W = img.height, img.width
    feats = {}
    for a in ARMS:
        remove_fp8()
        if a.startswith("fp8"):
            install_fp8(a == "fp8_row")
        with torch.inference_mode():
            ctx = torch.autocast("cuda", dtype=torch.float16) if a != "fp32" else torch.autocast("cuda", enabled=False)
            try:
                with ctx:
                    t = sync(); st = proc.set_image(img); dt = sync() - t
                feats[a] = to32(st["backbone_out"])
            except Exception as e:
                print(a, "FAILED", type(e).__name__, str(e)[:300], flush=True); feats[a] = None; dt = float("nan")
        if fi > 0:
            R[a]["t"].append(dt)
    remove_fp8()
    half_grounding(True)
    ref = ground_all(feats["fp32"])
    for a in ARMS[1:]:
        if feats[a] is None or fi == 0:
            continue
        R[a]["fpn"].append(max(float((x - y).abs().max() / (x.abs().max() + 1e-9)) for x, y in zip(feats["fp32"]["backbone_fpn"], feats[a]["backbone_fpn"])))
        R[a]["cmp"].append(compare(ref, ground_all(feats[a]), H, W))
    half_grounding(False)
    print(f"frame {c8}:{j} done", flush=True)

out = {}
print(f"\n{'backbone':12s} {'s':>7s} {'fpn_rel':>8s} {'max_dp':>7s} {'flips':>9s} {'mask_dis':>9s} {'nonfinite':>9s}")
for a in ARMS:
    r = R[a]; c = r["cmp"]
    out[a] = {"s": float(np.nanmean(r["t"])) if r["t"] else None, "fpn_rel_maxdiff": max(r["fpn"]) if r["fpn"] else None,
              "max_dp": max(x["max_dp"] for x in c) if c else None, "flips": sum(x["flips"] for x in c) if c else None,
              "kept": sum(x["kept"] for x in c) if c else None, "mask_dis": max(x["mask_dis"] for x in c) if c else None,
              "nonfinite": sum(x["nonfinite"] for x in c) if c else None}
    o = out[a]
    print(f"{a:12s} {o['s'] if o['s'] is None else round(o['s'], 3):>7} {str(None if o['fpn_rel_maxdiff'] is None else round(o['fpn_rel_maxdiff'], 4)):>8s} "
          f"{str(o['max_dp']):>7s} {str(o['flips']) + '/' + str(o['kept']):>9s} {str(None if o['mask_dis'] is None else '%.2e' % o['mask_dis']):>9s} {str(o['nonfinite']):>9s}")
Path("/home/nvidia/sam3map/sam3_fp8_probe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("ZZFP8-DONEZZ")
