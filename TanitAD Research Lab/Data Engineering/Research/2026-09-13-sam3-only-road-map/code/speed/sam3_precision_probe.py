"""SAM3 grounding precision variants on Thor (feasibility, not the approval test). The lever probe showed bf16 autocast halves
grounding (2.16 -> 1.03 s per frame) but flips 6 of 158 kept instances. Where does the speed come from and where do the flips
come from? Variants on the 19 prompts of real front frames, text features cached:
  fp32 B1 (reference) · fp32 B19 (mathematically equal, numerically different: the numerical floor)
  bf16 / fp16 autocast on: encoder only · encoder+decoder · all stages · all stages B19
Stage outputs are cast back to float32 at the boundary of a half-precision stage. Per variant: grounding time, sub-stage split,
max |delta p|, keep flips at the extractor thresholds, the worst queries (prompt, frame, p_ref, p_new), mask pixel disagreement.
Usage: sam3_precision_probe.py <c8>=<j,j> [...]   (first frame = warm-up)"""
import contextlib, dataclasses, json, sys, time
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

ORIG = {n: getattr(m, n) for n in ("_encode_prompt", "_run_encoder", "_run_decoder", "_run_segmentation_heads")}
SHORT = {"_encode_prompt": "prompt", "_run_encoder": "enc", "_run_decoder": "dec", "_run_segmentation_heads": "seg"}
STAGE = {}


def to32(x):
    if torch.is_tensor(x):
        return x.float() if x.is_floating_point() else x
    if isinstance(x, dict):
        for k in list(x.keys()):
            x[k] = to32(x[k])
        return x
    if isinstance(x, (list, tuple)):
        return type(x)(to32(v) for v in x)
    return x


def wrap(name, dtype):
    fn = ORIG[name]

    def w(*a, **k):
        t = sync()
        ctx = torch.autocast("cuda", dtype=dtype) if dtype is not None else contextlib.nullcontext()
        with ctx:
            r = fn(*a, **k)
        if dtype is not None:
            r = to32(r)
        STAGE[SHORT[name]] = STAGE.get(SHORT[name], 0.0) + sync() - t
        return r
    return w


def install(half_stages, dtype):
    for n in ORIG:
        setattr(m, n, wrap(n, dtype if SHORT[n] in half_stages else None))


@torch.inference_mode()
def ground(bo, ps):
    n = len(ps); b = dict(bo)
    for k in ("language_features", "language_embeds"):
        b[k] = torch.cat([TXT[p][k] for p in ps], 1)
    b["language_mask"] = torch.cat([TXT[p]["language_mask"] for p in ps], 0)
    fs = dataclasses.replace(proc.find_stage, img_ids=torch.zeros(n, dtype=torch.long, device="cuda"), text_ids=torch.arange(n, device="cuda"))
    out = m.forward_grounding(backbone_out=b, find_input=fs, geometric_prompt=m._get_dummy_prompt(n), find_target=None)
    pr = (out["pred_logits"].float().sigmoid() * out["presence_logit_dec"].float().sigmoid().unsqueeze(1)).squeeze(-1)
    return {p: (pr[i], out["pred_masks"][i].float()) for i, p in enumerate(ps)}


def run(bo, B):
    res = {}; t = sync()
    for s in range(0, len(P), B):
        res.update(ground(bo, P[s:s + B]))
    return res, sync() - t


VAR = [("fp32 B1", (), None, 1), ("fp32 B19 (floor)", (), None, 19)]
for nm, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
    VAR += [(f"{nm} enc", ("enc",), dt, 1), (f"{nm} enc+dec", ("enc", "dec"), dt, 1), (f"{nm} all", ("prompt", "enc", "dec", "seg"), dt, 1),
            (f"{nm} all B19", ("prompt", "enc", "dec", "seg"), dt, 19)]
R = {v[0]: {"t": [], "split": {}, "max_dp": 0.0, "flips": 0, "kept": 0, "dis": 0, "uni": 0, "worst": []} for v in VAR}
for fi, (c8, j, img) in enumerate(FR):
    H, W = img.height, img.width
    with torch.inference_mode():
        bo = proc.set_image(img)["backbone_out"]
    ref = None
    for name, stages, dt, B in VAR:
        STAGE.clear(); install(stages, dt)
        got, t = run(bo, B)
        if ref is None:
            ref = got
        if fi == 0:
            continue
        r = R[name]; r["t"].append(t)
        for k, v in STAGE.items():
            r["split"].setdefault(k, []).append(v)
        with torch.inference_mode():
            for p in P:
                pr, mr = ref[p]; pg, mg = got[p]
                d = (pr - pg).abs(); r["max_dp"] = max(r["max_dp"], float(d.max()))
                kr, kg = pr >= THR[p], pg >= THR[p]; r["kept"] += int(kr.sum())
                for q in torch.nonzero(kr ^ kg).flatten().tolist():
                    r["flips"] += 1; r["worst"].append((round(float(d[q]), 3), p, f"{c8}:{j}", q, round(float(pr[q]), 3), round(float(pg[q]), 3)))
                qd = int(d.argmax())
                if float(d[qd]) > 0.1:
                    r["worst"].append((round(float(d[qd]), 3), p, f"{c8}:{j}", qd, round(float(pr[qd]), 3), round(float(pg[qd]), 3)))
                both = torch.nonzero(kr & kg).flatten()
                if len(both):
                    a = interpolate(mr[both].unsqueeze(1), (H, W), mode="bilinear", align_corners=False) > 0
                    g = interpolate(mg[both].unsqueeze(1), (H, W), mode="bilinear", align_corners=False) > 0
                    r["dis"] += int((a ^ g).sum()); r["uni"] += int((a | g).sum())
    for n in ORIG:
        if n in m.__dict__:
            delattr(m, n)
    print(f"frame {c8}:{j} {'warm-up' if fi == 0 else 'timed'}", flush=True)

out = {}
print(f"\n{'variant':22s} {'ground s':>8s} {'prompt':>7s} {'enc':>6s} {'dec':>6s} {'seg':>6s} {'max_dp':>7s} {'flips':>9s} {'mask_dis':>9s}")
for name, *_ in VAR:
    r = R[name]; sp = {k: float(np.mean(v)) for k, v in r["split"].items()}
    out[name] = {"s": float(np.mean(r["t"])), "split": sp, "max_dp": r["max_dp"], "flips": r["flips"], "kept": r["kept"], "mask_dis_over_union": r["dis"] / max(r["uni"], 1),
                 "worst": sorted(set(r["worst"]), reverse=True)[:6]}
    print(f"{name:22s} {out[name]['s']:8.3f} {sp.get('prompt', 0):7.3f} {sp.get('enc', 0):6.3f} {sp.get('dec', 0):6.3f} {sp.get('seg', 0):6.3f} {r['max_dp']:7.3f} {r['flips']:4d}/{r['kept']:<4d} {out[name]['mask_dis_over_union']:9.2e}")
print("\nworst queries (|dp|, prompt, frame, query, p_ref, p_new):")
for name in out:
    if out[name]["worst"]:
        print(" ", name, out[name]["worst"][:4])
Path("/home/nvidia/sam3map/sam3_precision_probe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("ZZPREC-DONEZZ")
