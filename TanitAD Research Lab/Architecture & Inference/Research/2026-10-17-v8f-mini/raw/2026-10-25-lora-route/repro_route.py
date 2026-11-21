"""PREREG_V8F_LORA route question, probe 2 (CPU only, no GPU, no box 2, no download): reproduce a cache clip's pooled fp8 tokens from the LOCAL mp4s by the BUILDER'S OWN code.

One variant x one dtype per invocation:
  --variant native256   v2ep rebuilt from the mp4 at 256x640 (the PRODUCTION route: pod_build_b1_epcache.build_one -> v2_compressed.build_compressed, frame = CanonicalFrame.from_hfov(120, 256, 640, 'cylindrical'))
  --variant local416    v2ep rebuilt at 416x1024 then ``resize416_to_256`` (the dev-box TEST route 'thor416-resize', rebuilt from the mp4 here instead of scp'd from Thor)
  --dtype fp32|bf16     the encoder dtype (pull_encode_pooled.build_model); the production cache is bf16 on CUDA, the test_cpu_fp32 reference is fp32 on CPU.

The encode is ``dinov3_fp8_encode_ship.encode_episode`` IMPORTED (never re-implemented), then ``pull_encode_pooled.pool_field(f, 4, 10).to(float8_e4m3fn)`` -- exactly ``payload_from_field`` for the 4x10 grid.
A SUBSET of cache steps is encoded (``--frames N``, linspace over T like the identity control): the sub-episode keeps the ORIGINAL PNG bytes of the chosen frames (no re-encode), odd slots length 0,
so ``encode_episode`` decodes exactly those frames. Comparisons are against the two dev-box caches that exist locally for these clips (both are source=thor416-resize, NOT the production cache):
  cpu_fp32 = refav1_hj_cache/test_cpu_fp32/train (CPU fp32)   gpu_bf16 = refav1_hj_cache/test_runscript/train (dev-box GPU bf16)
Per-clip: pooled-fp8 rel-L2, max abs, mean abs, fraction bitwise equal. The pooled fp8 dump is written beside the json for cross-variant comparison (compare_variants.py).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve()
REPO = next(p for p in HERE.parents if (p / "EXPORT_BASE.txt").is_file())
sys.path[:0] = [str(REPO / "stack"), str(REPO / "taniteval")]
PEP_P = REPO / "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini/code/pull_encode_pooled.py"
SHIP_P = REPO / "stack/scripts/dinov3_fp8_encode_ship.py"
REF = {"cpu_fp32": Path("C:/Users/Admin/refav1_hj_cache/test_cpu_fp32/train/clips"), "gpu_bf16": Path("C:/Users/Admin/refav1_hj_cache/test_runscript/train/clips")}
S4B = Path("C:/Users/Admin/tanitad-data/refav1-train600-256")
CLIPS = ["8f387977-4086-466d-bba4-b04f8cdc0743", "bf5ec76e-e8b5-4f0d-a9cc-4fe1687d3d3a"]


def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def stats(mine: torch.Tensor, ref: torch.Tensor) -> dict:
    mine, ref = mine.float().reshape(-1), ref.float().reshape(-1)
    d = (mine - ref).abs()
    return {"n_values": int(d.numel()), "rel_l2": float((mine - ref).norm() / ref.norm().clamp_min(1e-12)), "max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "ref_mean_abs": float(ref.abs().mean()), "frac_bitwise_equal": float((mine == ref).float().mean())}


def pooled_stats(mine_list, ref_list) -> dict:
    """Pool several clips: rel_l2 from the summed squares, the rest over all values."""
    m, r = torch.cat([x.float().reshape(-1) for x in mine_list]), torch.cat([x.float().reshape(-1) for x in ref_list])
    return stats(m, r)


def content_vs(x: dict, y: dict) -> dict:
    """Everything the encoder, the labels and the poses read from a v2ep: tensors bit-equal, scalars/strings equal."""
    out = {}
    for k in sorted(set(x) | set(y)):
        a_, b_ = x.get(k), y.get(k)
        if isinstance(a_, torch.Tensor) and isinstance(b_, torch.Tensor):
            out[k] = bool(a_.dtype == b_.dtype and a_.shape == b_.shape and torch.equal(a_, b_))
        else:
            out[k] = bool(a_ == b_)
    out["ALL_EQUAL"] = all(out.values())
    return out


def sample_steps(T: int, n: int) -> list:
    return sorted({int(round(x)) for x in np.linspace(0, T - 1, n)})


def sub_episode(o: dict, steps) -> dict:
    """An episode dict holding exactly the PNG bytes of raw frames 2*g, g in steps, on the even slots (odd slots length 0)."""
    lens, buf = o["jpeg_len"], o["jpeg_buf"]
    offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
    parts, L = [], []
    for k, g in enumerate(steps):
        seg = buf[int(offs[2 * g]):int(offs[2 * g + 1])]
        parts.append(seg)
        L.append(int(seg.numel()))
        if k < len(steps) - 1:
            L.append(0)
    return {"codec": "png", "jpeg_buf": torch.cat(parts), "jpeg_len": torch.tensor(L, dtype=torch.int64)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["native256", "local416"], required=True)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], required=True)
    ap.add_argument("--frames", type=int, default=21, help="cache steps encoded per clip (linspace over T; 21 -> every 5th of 101)")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--work", default="R:/work")
    ap.add_argument("--out", required=True, help="output stem (.json, and .pt beside it)")
    ap.add_argument("--clips", nargs="+", default=CLIPS)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    assert not torch.cuda.is_available(), "CPU only"
    spec = importlib.util.spec_from_file_location("pull_encode_pooled", str(PEP_P))
    pep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pep)
    ship = pep.load_encoder_module(str(SHIP_P))
    model, mean, std, n_special = pep.build_model("cpu", a.dtype)
    v2dir = Path(a.work) / ("v2ep_256" if a.variant == "native256" else "v2ep_416")
    res = {"meta": {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "variant": a.variant, "dtype": a.dtype, "frames": a.frames, "threads": a.threads, "torch": torch.__version__,
                    "n_special": n_special, "builder_encode_episode_sha256": hashlib.sha256(__import__("inspect").getsource(ship.encode_episode).encode()).hexdigest(),
                    "pull_encode_pooled_sha256": sha256(PEP_P), "argv": sys.argv[1:]}, "clips": {}}
    dump = {}
    for c in a.clips:
        p = v2dir / f"{c}.v2ep.pt"
        o = torch.load(p, map_location="cpu", weights_only=False)
        T = (int(o["jpeg_len"].shape[0]) + 1) // 2
        steps = sample_steps(T, a.frames)
        sub = sub_episode(o, steps)
        t0 = time.time()
        f = ship.encode_episode(sub, model, mean, std, n_special, "cpu")
        pooled = pep.pool_field(f, 4, 10).to(torch.float8_e4m3fn).contiguous()
        secs = time.time() - t0
        rec = {"v2ep_path": str(p), "v2ep_sha256": sha256(p), "image_hw": [int(o["image_h"]), int(o["image_w"])], "T": T, "steps": steps, "seconds": secs, "vs": {}}
        if a.variant == "native256" and (S4B / f"{c}.v2ep.pt").is_file():
            rec["s4b_sha256"] = sha256(S4B / f"{c}.v2ep.pt")
            rec["v2ep_file_sha256_equal_S4b"] = rec["v2ep_sha256"] == rec["s4b_sha256"]
            rec["v2ep_content_vs_S4b"] = content_vs(o, torch.load(S4B / f"{c}.v2ep.pt", map_location="cpu", weights_only=False))
            # torch.save names the zip members after the FILE name, so build_v2ep.py's '<clip>.v2ep.pt.build' differs from the production '<clip>.v2ep.pt' in 60 bytes
            # and nothing else; re-saving under the production temp name must give the production bytes:
            rs = Path(a.work) / "resave"
            rs.mkdir(parents=True, exist_ok=True)
            torch.save(o, rs / f"{c}.v2ep.pt.tmp")
            rec["v2ep_resaved_as_production_name_sha256"] = sha256(rs / f"{c}.v2ep.pt.tmp")
            rec["v2ep_resaved_sha256_equal_S4b"] = rec["v2ep_resaved_as_production_name_sha256"] == rec["s4b_sha256"]
            (rs / f"{c}.v2ep.pt.tmp").unlink()
        dump[c] = {"steps": steps, "pooled4x10_fp8": pooled}
        for name, root in REF.items():
            ref = torch.load(root / f"{c}.pt", map_location="cpu", weights_only=True)["pool4x10"][steps]
            assert ref.shape == pooled.shape, (ref.shape, pooled.shape)
            rec["vs"][name] = stats(pooled, ref)
        res["clips"][c] = rec
        print(f"{a.variant}/{a.dtype} {c[:8]}: {len(steps)} frames in {secs:.0f} s; " + "; ".join(f"vs {k}: rel_l2 {v['rel_l2']:.4g}, max {v['max_abs']:.3g}, bitwise {v['frac_bitwise_equal']:.3f}" for k, v in rec["vs"].items()), flush=True)
    agg = {}
    for name, root in REF.items():
        refs = [torch.load(root / f"{c}.pt", map_location="cpu", weights_only=True)["pool4x10"][dump[c]["steps"]] for c in a.clips]
        agg[name] = pooled_stats([dump[c]["pooled4x10_fp8"] for c in a.clips], refs)
    res["both_clips_pooled"] = agg
    res["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(res, open(a.out + ".json", "w", encoding="utf-8"), indent=1)
    torch.save(dump, a.out + ".pt")
    print("DONE " + a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
