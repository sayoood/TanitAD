"""PREREG_V8F_LORA route question, probe 3 (pixels only; no encoder, no GPU): where do the eval clip's FIVE pixel products differ?  Clip dca157a0 (the one eval clip whose Thor 416x1024 v2ep survives locally).

  S4C   production eval frames: C:/Users/Admin/tanitad-data/refav1-eval141-256rb (native 256x640 rebuild from the mp4, dev box 2026-10-17; box 2's eval cache was encoded from it: encode_B2s.log)
  L256  this session's rebuild from the mp4 (build_v2ep.py --frame 256)                                      -> must equal S4C bit for bit (content)
  A30   the Aug-30 eval eps C:/Users/Admin/tanitad-data/refav1-eval141/eps (older Thor product; what the identity control's E_FIELD / E_POOLED cases used)
  T416R Thor's 416x1024 v2ep (refav1_hj_cache/scratch/thor416, scp'd 2026-10-17) -> resize416_to_256         (= the 'thor416-resize' route)
  L416R this session's local 416x1024 rebuild from the mp4 -> resize416_to_256 (build_v2ep.py --frame 416)  -> equals T416R iff the local builder reproduces Thor's 416 product

Compared on raw frames 0, 20, ... (every 10th cache step), as decoded uint8: exact-equal flag, MAE in grey levels (of 255), max abs level, fraction of differing values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torchvision

C = "dca157a0-20a9-464b-a74c-084b678e7be1"
P = {"S4C": Path("C:/Users/Admin/tanitad-data/refav1-eval141-256rb") / f"{C}.v2ep.pt",
     "L256": Path("R:/work/v2ep_256") / f"{C}.v2ep.pt",
     "A30": Path("C:/Users/Admin/tanitad-data/refav1-eval141/eps") / f"{C}.v2ep.pt",
     "T416": Path("C:/Users/Admin/refav1_hj_cache/scratch/thor416") / f"{C}.v2ep.pt",
     "L416R": Path("R:/work/v2ep_416") / f"{C}.v2ep.pt"}
HERE = Path(__file__).resolve()
REPO = next(p for p in HERE.parents if (p / "EXPORT_BASE.txt").is_file())
PEP_P = REPO / "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini/code/pull_encode_pooled.py"


def frames(o, raw_idx):
    lens, buf = o["jpeg_len"], o["jpeg_buf"]
    offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
    return torch.stack([torchvision.io.decode_png(buf[int(offs[j]):int(offs[j + 1])].clone()) for j in raw_idx])


def cmp(x: torch.Tensor, y: torch.Tensor) -> dict:
    d = (x.int() - y.int()).abs()
    return {"exact_equal": bool(torch.equal(x, y)), "mae_grey_of_255": float(d.float().mean()), "max_abs_level": int(d.max()), "frac_values_differing": float((d > 0).float().mean())}


def main() -> int:
    import importlib.util
    spec = importlib.util.spec_from_file_location("pull_encode_pooled", str(PEP_P))
    pep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pep)
    o = {k: torch.load(p, map_location="cpu", weights_only=False) for k, p in P.items()}
    out = {"clip": C, "paths": {k: str(v) for k, v in P.items()}, "geometry": {k: [int(v["image_h"]), int(v["image_w"]), str(v.get("codec")), v.get("projection_mode"), v["frame"]] for k, v in o.items()},
           "n_raw_frames": {k: int(v["jpeg_len"].shape[0]) for k, v in o.items()}}
    o["T416R"] = pep.resize416_to_256(o.pop("T416"))
    T = (int(o["S4C"]["jpeg_len"].shape[0]) + 1) // 2
    raw = [2 * g for g in range(0, T, 10)]
    out["raw_frames"] = raw
    out["poses_actions_equal_S4C"] = {k: bool(torch.equal(v["poses"], o["S4C"]["poses"]) and torch.equal(v["actions"], o["S4C"]["actions"])) for k, v in o.items() if k != "S4C"}
    fr = {k: frames(v, raw) for k, v in o.items()}
    pairs = [("L256", "S4C"), ("A30", "S4C"), ("T416R", "S4C"), ("T416R", "A30"), ("L416R", "T416R"), ("L416R", "S4C"), ("L416R", "L256")]
    out["pairs"] = {f"{a}_vs_{b}": cmp(fr[a], fr[b]) for a, b in pairs}
    out["jpeg_buf_bytes_equal_L256_S4C"] = bool(torch.equal(o["L256"]["jpeg_buf"], o["S4C"]["jpeg_buf"]) and torch.equal(o["L256"]["jpeg_len"], o["S4C"]["jpeg_len"]))
    json.dump(out, open(Path(__file__).with_suffix(".json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
