"""PREREG_V8F_LORA proof 1: the IDENTITY CONTROL -- with the LoRA B-matrices at zero, the on-the-fly tokens must equal the cached tokens.

THIS FILE REPORTS NUMBERS. It does NOT choose tau (the tolerance) or epsilon: the Master Mind commits them in an amendment from these measurements, before any training.

What is compared, and against what (every reference is a file that already existed on this box; nothing was downloaded, nothing was read from box 2):

  E_FIELD    eval clips, the frames BEHIND the cache (``--eval-eps``, the builder's own input), bf16 CPU, vs the dev-box GPU-bf16 RAW-FIELD fp8 cache of the same clips
             (``refav1-fp8-eval/<clip>.pt``, [T, 640, 1024], built by ``encode_episode`` itself). Compared (a) as fields, (b) after the builder's ``pool_field`` 4x10.
             That cache quantises BEFORE pooling, so its noise floor differs from the v8F cache's (pool, THEN quantise): reported, not hidden.
  E_POOLED   the one v8F-format eval cache clip on this box (``test_cpu_fp32/eval``: pool4x10 fp8, built CPU/fp32), fp32 CPU, every frame.
  T_POOLED   train clips REBUILT FROM THE LOCAL MP4s (``build_v2ep.py --frame 416``: the route the dev-box train test cache took, ``source: thor416-resize``), bf16 CPU,
             vs that cache (``test_runscript/train``: pool4x10 fp8, built on the dev-box GPU, bf16). Includes the pixel-route noise of a rebuild (MEASURED separately:
             a rebuilt eval v2ep differs from the Aug-30 eval v2ep by up to 3/255 grey levels, mean 0.75, 71 % of pixels).
  SAME_DEVICE  the builder's OWN ``encode_episode`` vs the LoRA encoder at B = 0, same process, same dtype, real frames: must be EXACTLY 0.0. The mutants (one preprocessing
             constant changed) are scored here -- where the baseline is exactly zero, any change is visible -- and against the cross-device references above.

The production train cache (``cache_full``, 4,512+60 clips) and the production eval-141 pooled cache live on box 2 (``D:\\tanitad\\cache_full``); box 2 is out of bounds for this
work. ``--train-cache`` / ``--eval-pooled-cache`` take any cache root, so the same script re-runs on the real caches (and on a GPU: ``--device cuda``) unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[6]
sys.path[:0] = [str(REPO / "stack"), str(REPO / "taniteval")]
from tanitad.models import v8f_lora_encoder as E  # noqa: E402

EVAL_EPS = "C:/Users/Admin/tanitad-data/refav1-eval141/eps"
EVAL_RAW_FP8 = "C:/Users/Admin/tanitad-data/refav1-eval141/refav1-fp8-eval"
EVAL_POOLED_CACHE = "C:/Users/Admin/refav1_hj_cache/test_cpu_fp32"            # eval/clips/dca157a0-....pt (CPU fp32, pool4x10 fp8)
TRAIN_POOLED_CACHE = "C:/Users/Admin/refav1_hj_cache/test_runscript"          # train/clips/*.pt (dev-box GPU bf16, thor416-resize route)
EVAL_CLIP_POOLED = "dca157a0-20a9-464b-a74c-084b678e7be1"


def stats(mine: torch.Tensor, ref: torch.Tensor) -> dict:
    """Elementwise comparison of two fp32 tensors of one shape."""
    mine, ref = mine.float().reshape(-1, mine.shape[-1]), ref.float().reshape(-1, ref.shape[-1])
    d = (mine - ref).abs()
    n = mine.shape[0]
    fr = (mine - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-12)
    return {"n_values": int(d.numel()), "max_abs": float(d.max()), "mean_abs": float(d.mean()), "p99_abs": float(torch.quantile(d.flatten()[::max(1, d.numel() // 2_000_000)], 0.99)),
            "ref_mean_abs": float(ref.abs().mean()), "rel_l2": float((mine - ref).norm() / ref.norm().clamp_min(1e-12)),
            "rel_l2_per_row_mean": float(fr.mean()), "rel_l2_per_row_max": float(fr.max()), "frac_bitwise_equal": float((mine == ref).float().mean())}


def merge_stats(parts: list) -> dict:
    """Combine per-batch stats (weighted by n_values) -- max of maxes, weighted means; rel_l2 is recomputed from sums of squares by the caller when exact."""
    w = np.array([p["n_values"] for p in parts], float)
    out = {"n_values": int(w.sum()), "max_abs": max(p["max_abs"] for p in parts)}
    for k in ("mean_abs", "ref_mean_abs", "frac_bitwise_equal", "p99_abs"):
        out[k] = float((np.array([p[k] for p in parts]) * w).sum() / w.sum())
    out["p99_abs"] = float(max(p["p99_abs"] for p in parts))
    out["rel_l2"] = float(np.sqrt((np.array([p["rel_l2"] ** 2 for p in parts]) * np.array([p["_ref_sq"] for p in parts])).sum() / np.array([p["_ref_sq"] for p in parts]).sum()))
    out["rel_l2_per_row_mean"] = float((np.array([p["rel_l2_per_row_mean"] for p in parts]) * w).sum() / w.sum())
    out["rel_l2_per_row_max"] = max(p["rel_l2_per_row_max"] for p in parts)
    return out


def stats_acc(mine, ref):
    s = stats(mine, ref)
    s["_ref_sq"] = float(ref.float().pow(2).sum())
    return s


def sample_steps(T: int, n: int) -> list:
    return sorted({int(round(x)) for x in np.linspace(0, T - 1, n)})


def run_batches(enc, o, steps, ref_fn, *, bs: int, tag: str, log):
    """Encode ``steps`` of v2ep ``o`` in batches of ``bs``; ``ref_fn(batch_steps) -> dict(name -> (mine_fn(enc, frames) -> tensor, ref tensor))``."""
    parts: dict = {}
    t0 = time.time()
    for s in range(0, len(steps), bs):
        st = steps[s:s + bs]
        fr = E.decode_cache_frames(o, st)
        with torch.no_grad():
            for name, (mine, ref) in ref_fn(enc, fr, st).items():
                parts.setdefault(name, []).append(stats_acc(mine, ref))
        log(f"  {tag}: {min(s + bs, len(steps))}/{len(steps)} frames, {time.time() - t0:.0f} s")
    return parts, time.time() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output stem; writes <out>.json (and the caller tees <out>.out)")
    ap.add_argument("--work", required=True, help="scratch dir for the rebuilt v2ep (<= ~1 GB)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--eval-eps", default=EVAL_EPS)
    ap.add_argument("--eval-raw-fp8", default=EVAL_RAW_FP8)
    ap.add_argument("--eval-pooled-cache", default=EVAL_POOLED_CACHE)
    ap.add_argument("--train-cache", default=TRAIN_POOLED_CACHE)
    ap.add_argument("--train-clips", nargs="+", default=["8f387977-4086-466d-bba4-b04f8cdc0743", "bf5ec76e-e8b5-4f0d-a9cc-4fe1687d3d3a"])
    ap.add_argument("--eval-clips", nargs="+", default=None, help="default: dca157a0 + 3 more of the 141, sha12 order")
    ap.add_argument("--n-eval-frames", type=int, default=204, help="E_FIELD frames in total (spread over --eval-clips)")
    ap.add_argument("--n-train-frames", type=int, default=202, help="T_POOLED frames in total")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--only", default="", help="comma list of cases to run: E_FIELD,E_POOLED,T_POOLED,ROUTE,SAME_DEVICE")
    ap.add_argument("--mutant-frames", type=int, default=8)
    ap.add_argument("--route-frames", type=int, default=12)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    only = set(x for x in a.only.split(",") if x)
    want = lambda c: not only or c in only
    logf = open(a.out + ".log", "a", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    pep, ship = E.load_builder()
    dev = a.device
    res = {"meta": {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": torch.__version__, "threads": a.threads, "device": dev,
                    "model_safetensors_sha256": E.file_sha256(E.model_safetensors_path()), "builder_model_sha256": E.BUILDER_MODEL_SHA256,
                    "builder_encode_episode_sha256": E.builder_encode_episode_sha256(), "builder_pin": E.BUILDER_ENCODE_EPISODE_SHA256,
                    "argv": sys.argv[1:], "tau": "NOT CHOSEN HERE (the Master Mind commits tau / epsilon in an amendment)"},
           "cases": {}}
    assert res["meta"]["model_safetensors_sha256"] == E.BUILDER_MODEL_SHA256, "the local DINOv3-L weights are not the cache builder's"
    outp = a.out + ".json"

    def save():
        json.dump(res, open(outp, "w", encoding="utf-8"), indent=1)

    def encoder(dtype):
        t = time.time()
        enc = E.DinoV3LoRAEncoder.from_pretrained(r=32, alpha=32.0, targets=("q", "v"), device=dev, dtype=dtype, verify_sha=False)
        assert enc.built_levers()["lora_B_zero_at_build"] and enc.trainable_count() == 3_145_728
        log(f"encoder {dtype} built in {time.time() - t:.0f} s, trainable {enc.trainable_count():,}")
        return enc

    ev_all = sorted(p.name[:-len(".v2ep.pt")] for p in Path(a.eval_eps).glob("*.v2ep.pt"))
    if not ev_all:
        raise SystemExit(f"no eval v2ep in {a.eval_eps} (looked there; also nothing else is tried: no download)")
    ev_clips = a.eval_clips or ([EVAL_CLIP_POOLED] + [c for c in sorted(ev_all, key=lambda c: hashlib.sha256(c.encode()).hexdigest()[:12]) if c != EVAL_CLIP_POOLED][:3])

    # ---------------------------------------------------------------------------------------- E_FIELD (bf16)
    if want("E_FIELD"):
        enc = encoder(torch.bfloat16)
        per = max(1, a.n_eval_frames // len(ev_clips))
        agg = {"field": [], "pooled_unq": [], "fp8_of_mine_vs_cache_fp8": []}
        clips_done = {}
        for c in ev_clips:
            o = torch.load(Path(a.eval_eps) / f"{c}.v2ep.pt", map_location="cpu", weights_only=False)
            ref_all = torch.load(Path(a.eval_raw_fp8) / f"{c}.pt", map_location="cpu", weights_only=True)
            T = E.n_cache_frames(o)
            assert ref_all.shape[0] == T, (ref_all.shape, T)
            steps = sample_steps(T, per)

            def ref_fn(enc_, fr, st, ref_all=ref_all):
                f = enc_.field(fr)
                r8 = ref_all[st]
                r = r8.float()
                return {"field": (f, r), "pooled_unq": (pep.pool_field(f, 4, 10), pep.pool_field(r, 4, 10)),
                        "fp8_of_mine_vs_cache_fp8": (f.to(torch.float8_e4m3fn).float(), r)}
            parts, secs = run_batches(enc, o, steps, ref_fn, bs=a.bs, tag=f"E_FIELD {c[:8]}", log=log)
            for k, v in parts.items():
                agg[k] += v
            clips_done[c] = {"steps": steps, "seconds": secs, "v2ep_bytes": (Path(a.eval_eps) / f"{c}.v2ep.pt").stat().st_size}
        res["cases"]["E_FIELD_bf16_cpu"] = {
            "what": "frames behind the cache (eval eps) -> bf16 LoRA(B=0) encoder vs the GPU-bf16 raw-field fp8 cache of the same clips",
            "reference": a.eval_raw_fp8, "n_frames": sum(len(v["steps"]) for v in clips_done.values()), "clips": clips_done,
            "field": merge_stats(agg["field"]), "pooled4x10_mine_unquantised_vs_pool_of_cache": merge_stats(agg["pooled_unq"]),
            "fp8_of_mine_vs_cache_fp8_field": merge_stats(agg["fp8_of_mine_vs_cache_fp8"])}
        save()
        del enc

    # ---------------------------------------------------------------------------------------- E_POOLED (fp32, v8F format)
    if want("E_POOLED"):
        c = EVAL_CLIP_POOLED
        p = Path(a.eval_pooled_cache) / "eval" / "clips" / f"{c}.pt"
        if not p.is_file():
            res["cases"]["E_POOLED_fp32_cpu"] = {"skipped": f"{p} does not exist (looked there and nowhere else)"}
        else:
            enc = encoder(torch.float32)
            o = torch.load(Path(a.eval_eps) / f"{c}.v2ep.pt", map_location="cpu", weights_only=False)
            cache = torch.load(p, map_location="cpu", weights_only=True)
            ref = cache["pool4x10"]
            T = E.n_cache_frames(o)
            steps = list(range(T))

            def ref_fn(enc_, fr, st):
                return {"pooled_fp8": (enc_.pooled(fr, "4x10", quantize=True), ref[st].float()),
                        "pooled_unquantised": (enc_.pooled(fr, "4x10", quantize=False), ref[st].float())}
            parts, secs = run_batches(enc, o, steps, ref_fn, bs=a.bs, tag="E_POOLED fp32", log=log)
            res["cases"]["E_POOLED_fp32_cpu"] = {
                "what": "the v8F-format eval cache clip (pool4x10 fp8, CPU fp32 build) vs the LoRA(B=0) encoder in fp32, every frame", "reference": str(p),
                "clip": c, "n_frames": len(steps), "seconds": secs, "pooled_fp8_vs_cache": merge_stats(parts["pooled_fp8"]),
                "pooled_unquantised_vs_cache": merge_stats(parts["pooled_unquantised"]),
                "cache_enc_dtype": json.loads([l for l in open(Path(a.eval_pooled_cache) / "eval" / "manifest.jsonl", encoding="utf-8") if c in l][-1]).get("enc_dtype")}
            save()
            del enc

    # ---------------------------------------------------------------------------------------- T_POOLED (bf16, rebuilt frames)
    if want("T_POOLED"):
        import subprocess
        v2 = work / "v2ep_416"
        todo = [c for c in a.train_clips if not (v2 / f"{c}.v2ep.pt").is_file()]
        if todo:
            r = subprocess.run([sys.executable, str(HERE.parent / "build_v2ep.py"), "--out", str(v2), "--root", str(work / "build_root"), "--frame", "416",
                                "--clips", *todo], capture_output=True, text=True, encoding="utf-8", errors="replace")
            log(r.stdout[-600:])
            if r.returncode:
                raise SystemExit("build_v2ep failed: " + r.stderr[-1200:])
        enc = encoder(torch.bfloat16)
        per = max(1, a.n_train_frames // len(a.train_clips))
        agg = {"pooled_fp8": [], "pooled_unquantised": []}
        clips_done = {}
        for c in a.train_clips:
            p = Path(a.train_cache) / "train" / "clips" / f"{c}.pt"
            if not p.is_file():
                clips_done[c] = {"skipped": f"{p} does not exist"}
                continue
            o = torch.load(v2 / f"{c}.v2ep.pt", map_location="cpu", weights_only=False)
            ref = torch.load(p, map_location="cpu", weights_only=True)["pool4x10"]
            assert ref.shape[0] == E.n_cache_frames(o), (ref.shape, E.n_cache_frames(o))
            steps = sample_steps(ref.shape[0], per)

            def ref_fn(enc_, fr, st, ref=ref):
                return {"pooled_fp8": (enc_.pooled(fr, "4x10", quantize=True), ref[st].float()),
                        "pooled_unquantised": (enc_.pooled(fr, "4x10", quantize=False), ref[st].float())}
            parts, secs = run_batches(enc, o, steps, ref_fn, bs=a.bs, tag=f"T_POOLED {c[:8]}", log=log)
            for k, v in parts.items():
                agg[k] += v
            clips_done[c] = {"steps": steps, "seconds": secs, "v2ep_md5": hashlib.md5(open(v2 / f"{c}.v2ep.pt", "rb").read(1 << 24)).hexdigest()}
        if agg["pooled_fp8"]:
            res["cases"]["T_POOLED_bf16_cpu"] = {
                "what": "train clips rebuilt from the local mp4 (416x1024 cylinder -> resize416_to_256, the route of the dev-box test cache) -> bf16 LoRA(B=0) vs that cache (dev-box GPU bf16)",
                "reference": a.train_cache, "n_frames": sum(len(v.get("steps", [])) for v in clips_done.values()), "clips": clips_done,
                "pooled_fp8_vs_cache": merge_stats(agg["pooled_fp8"]), "pooled_unquantised_vs_cache": merge_stats(agg["pooled_unquantised"])}
        else:
            res["cases"]["T_POOLED_bf16_cpu"] = {"skipped": clips_done}
        save()
        del enc

    # ---------------------------------------------------------------------------------------- ROUTE: the pixel-route noise of a rebuild, in isolation
    if want("ROUTE"):
        import subprocess
        out = {}
        ev_c = EVAL_CLIP_POOLED
        pe = Path(a.eval_pooled_cache) / "eval" / "clips" / f"{ev_c}.pt"
        nat = work / "v2ep_256"
        todo = [c for c in [ev_c, *a.train_clips] if not (nat / f"{c}.v2ep.pt").is_file()]
        if todo:
            r = subprocess.run([sys.executable, str(HERE.parent / "build_v2ep.py"), "--out", str(nat), "--root", str(work / "build_root"), "--frame", "256", "--clips", *todo],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            log(r.stdout[-400:])
            if r.returncode:
                raise SystemExit("build_v2ep failed: " + r.stderr[-1200:])
        cases = []
        if pe.is_file():
            cases.append(("eval_native_rebuild_vs_v8F_eval_cache(CPU fp32)", ev_c, nat, pe, torch.float32, "eval_exact_eps_same_clip_for_comparison", Path(a.eval_eps)))
        for c in a.train_clips:
            pt = Path(a.train_cache) / "train" / "clips" / f"{c}.pt"
            if pt.is_file():
                cases.append((f"train_{c[:8]}_native256_rebuild_vs_thor416_route_cache(GPU bf16)", c, nat, pt, torch.bfloat16, None, None))
        for name, c, root, refp, dt, _, _ in cases:
            enc = encoder(dt)
            o = torch.load(root / f"{c}.v2ep.pt", map_location="cpu", weights_only=False)
            ref = torch.load(refp, map_location="cpu", weights_only=True)["pool4x10"]
            steps = sample_steps(ref.shape[0], a.route_frames)

            def ref_fn(enc_, fr, st, ref=ref):
                return {"pooled_fp8": (enc_.pooled(fr, "4x10", quantize=True), ref[st].float())}
            parts, secs = run_batches(enc, o, steps, ref_fn, bs=a.bs, tag=f"ROUTE {name[:30]}", log=log)
            out[name] = {"n_frames": len(steps), "seconds": secs, "pooled_fp8_vs_cache": merge_stats(parts["pooled_fp8"])}
            del enc
        res["cases"]["ROUTE_rebuilt_native256"] = out
        save()

    # ---------------------------------------------------------------------------------------- SAME_DEVICE: builder's own function vs ours, + mutants
    if want("SAME_DEVICE"):
        enc = encoder(torch.bfloat16)
        c = ev_clips[0]
        o = torch.load(Path(a.eval_eps) / f"{c}.v2ep.pt", map_location="cpu", weights_only=False)
        steps = sample_steps(E.n_cache_frames(o), a.mutant_frames)
        fr = E.decode_cache_frames(o, steps)
        # a sub-episode holding exactly these frames (lossless PNG re-encode), so the builder's own encode_episode runs on them
        import torchvision
        pngs = [torchvision.io.encode_png(fr[i], compression_level=1) for i in range(fr.shape[0])]
        lens, k = [], 0
        for j in range(2 * len(pngs) - 1):
            if j % 2 == 0:
                lens.append(int(pngs[j // 2].numel()))
            else:
                lens.append(0)
        sub = {"codec": "png", "jpeg_buf": torch.cat(pngs), "jpeg_len": torch.tensor(lens, dtype=torch.int64)}
        assert torch.equal(E.decode_cache_frames(sub, list(range(len(pngs)))), fr)
        frozen = enc.merge()
        t = time.time()
        ref = ship.encode_episode(sub, frozen, enc.mean, enc.std, 5, dev)                   # THE BUILDER'S OWN FUNCTION
        log(f"SAME_DEVICE builder encode_episode on {len(steps)} frames: {time.time() - t:.0f} s")
        with torch.no_grad():
            mine = enc.field(fr)
        out = {"clip": c, "steps": steps, "baseline_B0_vs_builder_encode_episode": stats(mine, ref), "mutants": {}}
        out["baseline_pooled_fp8_vs_builder_payload"] = stats(enc.pooled(fr, "4x10", quantize=True), pep.pool_field(ref, 4, 10).to(torch.float8_e4m3fn).float())
        # the cross-device reference for the same frames
        raw8 = torch.load(Path(a.eval_raw_fp8) / f"{c}.pt", map_location="cpu", weights_only=True)[steps].float()
        out["baseline_vs_cache_field"] = stats(mine, raw8)
        muts = {"PIXEL_SCALE_255_to_256": ("PIXEL_SCALE", 256.0),
                "IMAGE_MEAN0_0.485_to_0.486": ("IMAGE_MEAN", (0.486, 0.456, 0.406)),
                "IMAGE_MEAN0_0.485_to_0.5": ("IMAGE_MEAN", (0.5, 0.456, 0.406)),
                "IMAGE_STD2_0.225_to_0.226": ("IMAGE_STD", (0.229, 0.224, 0.226)),
                "IMAGE_STD0_0.229_to_0.25": ("IMAGE_STD", (0.25, 0.224, 0.225))}
        for name, (attr, val) in muts.items():
            old = getattr(E, attr)
            try:
                setattr(E, attr, val)
                m_, s_ = E.mean_std(torch.bfloat16, dev)                                   # the mutated constants, through the same rounding to bf16
                saved = (enc.mean, enc.std)
                enc.mean, enc.std = m_, s_
                with torch.no_grad():
                    mm = enc.field(fr)
                enc.mean, enc.std = saved
            finally:
                setattr(E, attr, old)
            out["mutants"][name] = {"vs_builder_same_device": stats(mm, ref), "vs_cache_field_cross_device": stats(mm, raw8)}
            log(f"  mutant {name}: vs builder rel_l2 {out['mutants'][name]['vs_builder_same_device']['rel_l2']:.3e}, vs cache {out['mutants'][name]['vs_cache_field_cross_device']['rel_l2']:.3e}")
        res["cases"]["SAME_DEVICE_bf16_cpu"] = out
        save()

    res["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    log("DONE " + outp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
