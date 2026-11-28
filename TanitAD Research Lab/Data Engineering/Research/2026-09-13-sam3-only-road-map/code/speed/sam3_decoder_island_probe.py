"""Which part of SAM3's decoder is precision-sensitive? (spdF2/spdF3 failed N2 on the edge class; the precision screen put the extra
mask-boundary deviation in the fp16 decoder.) The fp16 decoder is worth 0.22 s/frame at 19 prompts batched, so the question is
whether a small fp32 island keeps the masks while the rest of the decoder stays fp16. Feasibility / diagnosis only -- a candidate
still has to pass the SPEC's map-level bars.
All variants: fusion encoder fp16, mask head fp32, 19 prompts batched; decoder as listed. Per variant vs the fp32 per-prompt reference:
decoder time, max |dp|, keep flips at the extractor thresholds, kept-mask pixel disagreement, non-finite count.
  V_floor   everything fp32 (batched): numerical floor
  V_F4      decoder fp32 (the part-3 arm spdF4a)
  V_dec16   decoder fp16, no island (as in the failed spdF2)
  islands on V_dec16: image cross-attention · box-RPB bias (matrix + its MLPs) · box refinement (bbox MLP + ref-point head) ·
  query self-attention + text cross-attention · image cross-attention AND RPB bias
Usage: sam3_decoder_island_probe.py <c8>=<j,j> [...]  (first frame = warm-up)"""
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
dec = m.transformer.decoder
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


STAGE_T = {}
ORIG = {n: getattr(m, n) for n in ("_run_encoder", "_run_decoder")}


def set_stages(enc16, dec16):
    for n, half in (("_run_encoder", enc16), ("_run_decoder", dec16)):
        fn = ORIG[n]

        def w(*a, _fn=fn, _half=half, _n=n, **k):
            t = sync()
            if _half:
                with torch.autocast("cuda", dtype=torch.float16):
                    r = _fn(*a, **k)
                r = to32(r)
            else:
                r = _fn(*a, **k)
            STAGE_T[_n] = STAGE_T.get(_n, 0.0) + sync() - t
            return r
        setattr(m, n, w)


ISLANDS = {}


def island_module(mod):
    fwd = type(mod).forward.__get__(mod)

    def f(*a, **k):
        a = [x.float() if torch.is_tensor(x) and x.is_floating_point() else x for x in a]
        k = {kk: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for kk, v in k.items()}
        with torch.autocast("cuda", enabled=False):
            return fwd(*a, **k)
    mod.forward = f
    ISLANDS.setdefault("modules", []).append(mod)


def island_rpb():
    fn = type(dec)._get_rpb_matrix.__get__(dec)

    def f(reference_boxes, feat_size):
        with torch.autocast("cuda", enabled=False):
            return fn(reference_boxes.float(), feat_size)
    dec._get_rpb_matrix = f
    ISLANDS["rpb"] = True


def clear_islands():
    for mod in ISLANDS.get("modules", []):
        if "forward" in mod.__dict__:
            del mod.forward
    if "_get_rpb_matrix" in dec.__dict__:
        del dec._get_rpb_matrix
    ISLANDS.clear()


def apply(names):
    for nm in names:
        if nm == "cross_attn":
            for L in dec.layers:
                island_module(L.cross_attn)
        elif nm == "rpb":
            island_rpb(); island_module(dec.boxRPB_embed_x); island_module(dec.boxRPB_embed_y)
        elif nm == "box":
            island_module(dec.bbox_embed); island_module(dec.ref_point_head)
        elif nm == "sa_text":
            for L in dec.layers:
                island_module(L.self_attn)
                if getattr(L, "ca_text", None) is not None:
                    island_module(L.ca_text)


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
def ground_ref(bo):
    res = {}
    for p in P:
        b = dict(bo); b.update(TXT[p])
        out = m.forward_grounding(backbone_out=b, find_input=proc.find_stage, geometric_prompt=m._get_dummy_prompt(), find_target=None)
        pr = (out["pred_logits"].float().sigmoid() * out["presence_logit_dec"].float().sigmoid().unsqueeze(1)).squeeze(-1)
        res[p] = (pr[0], out["pred_masks"][0].float())
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
    return {"max_dp": dmax, "flips": flips, "kept": kept, "dis": dis, "uni": uni, "nonfinite": nonfinite}


VARIANTS = [("V_floor", False, False, []), ("V_F4", True, False, []), ("V_dec16", True, True, []),
            ("V_dec16+cross_attn32", True, True, ["cross_attn"]), ("V_dec16+rpb32", True, True, ["rpb"]),
            ("V_dec16+box32", True, True, ["box"]), ("V_dec16+sa_text32", True, True, ["sa_text"]),
            ("V_dec16+cross_attn32+rpb32", True, True, ["cross_attn", "rpb"])]
R = {v[0]: {"dec_s": [], "tot_s": [], "max_dp": 0.0, "flips": 0, "kept": 0, "dis": 0, "uni": 0, "nonfinite": 0} for v in VARIANTS}
for fi, (c8, j, img) in enumerate(FR):
    H, W = img.height, img.width
    with torch.inference_mode():
        bo = proc.set_image(img)["backbone_out"]
    for n in ORIG:
        if n in m.__dict__:
            delattr(m, n)
    ref = ground_ref(bo)
    for name, e16, d16, isl in VARIANTS:
        clear_islands(); apply(isl); set_stages(e16, d16)
        STAGE_T.clear(); t = sync(); got = ground_all(bo); tt = sync() - t
        clear_islands()
        for n in ORIG:
            if n in m.__dict__:
                delattr(m, n)
        if fi == 0:
            continue
        r = R[name]; r["dec_s"].append(STAGE_T.get("_run_decoder", 0.0)); r["tot_s"].append(tt)
        c = compare(ref, got, H, W)
        r["max_dp"] = max(r["max_dp"], c["max_dp"])
        for k in ("flips", "kept", "dis", "uni", "nonfinite"):
            r[k] += c[k]
    print(f"frame {c8}:{j} done", flush=True)

out = {}
print(f"\n{'variant':30s} {'grounding s':>11s} {'decoder s':>9s} {'max_dp':>7s} {'flips':>9s} {'mask_dis':>9s} {'nonfinite':>9s}")
for name, *_ in VARIANTS:
    r = R[name]
    out[name] = {"grounding_s": float(np.mean(r["tot_s"])), "decoder_s": float(np.mean(r["dec_s"])), "max_dp": r["max_dp"], "flips": r["flips"],
                 "kept": r["kept"], "mask_dis": r["dis"] / max(r["uni"], 1), "nonfinite": r["nonfinite"]}
    o = out[name]
    print(f"{name:30s} {o['grounding_s']:11.3f} {o['decoder_s']:9.3f} {o['max_dp']:7.3f} {str(o['flips']) + '/' + str(o['kept']):>9s} {o['mask_dis']:9.2e} {o['nonfinite']:9d}")
Path("/home/nvidia/sam3map/sam3_decoder_island_probe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("ZZISLAND-DONEZZ")
