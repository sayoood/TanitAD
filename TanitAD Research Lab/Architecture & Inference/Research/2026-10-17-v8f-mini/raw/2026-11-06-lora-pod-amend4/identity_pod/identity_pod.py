#!/usr/bin/env python3
"""PREREG_V8F_LORA, pod-side token-level identity control (device + Linux frame route) and the encoder throughput.

Three frame/token sets per clip, all encoded by the cache BUILDER's numerics on THIS pod's GPU:
  (a) the dev-box BANKED v2ep frames   -> encoded here
  (b) the POD-built v2ep frames        -> encoded here
  (c) box 2's STORED pooled fp8 tokens (cache_full, pulled by sftp, md5-checked)
  a vs c = the device effect;  b vs c = device + Linux route (the card's decision comparison);  a vs b = the route alone.

Two encoder paths, run on the same GPU in the same process:
  MODULE  = v8f_lora_encoder.DinoV3LoRAEncoder at B = 0, driven by the AR-2 identity driver's own functions
            (batches of 8, inference_mode, fp32 field to CPU, builder pool_field 4x10, then fp8 e4m3fn). PRIMARY.
  BUILDER = dinov3_fp8_encode_ship.encode_episode on the model built by pull_encode_pooled.build_model (the code that made cache_full). CROSS-CHECK.

Controls: (1) re-encoding the same frames twice must be bit-equal (run-to-run noise floor on this GPU);
          (2) the MODULE path must equal the BUILDER path bit for bit on this GPU;
          (3) the AR-2 pixel-scale mutant must be caught (the instrument is not blind).
Decision rule (fixed by the Master Mind before this measurement):  B KEPT iff  b vs c  rel-L2 <= 1e-3 on all 3 decision clips; else B' ROUTE.
No address, port or pod id lives in this file.
"""
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import torch

torch.set_num_threads(int(os.environ.get("IDENT_THREADS", "2")))

ROOT = Path("/root/identity_pod")
DRIVER = ROOT / "code" / "drv" / "d1" / "d2" / "v8f_lora_identity_dev.py"   # depth 5 below ROOT: the driver computes REPO = HERE.parents[5]
DRIVER_SHA256 = "fec465465bd826cea511d1bf2a79d8b5198f267f3ae471c92e13a3917096841d"   # identity_lora.json "driver_sha256" of the banked AR-2 run 3
PEP_SHA256_PREFIX = "a7dee3884e9667c6"
FRAMES_A = ROOT / "frames_a"
CACHE_C = ROOT / "cache_c"
V2EP_B = Path("/root/data/v2ep")
LEDGER = Path("/root/data/v2ep_ledger.jsonl")
BANKED = json.load(open("/root/ops/banked_hashes.json"))
OUT = ROOT / "out"
TAU = 1e-3
DECISION = [("train", "8f387977-4086-466d-bba4-b04f8cdc0743"), ("train", "bf5ec76e-e8b5-4f0d-a9cc-4fe1687d3d3a"), ("eval", "dca157a0-20a9-464b-a74c-084b678e7be1")]
EXTRA = [("train", "0e1526b3-d2b6-46b2-95fa-2903bb34e2ec"), ("train", "83e32d0f-ed19-4116-8c1c-9e4c843de993"), ("eval", "e6cb7cd4-8349-4b9d-9674-e3ebf789e611")]


def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_driver():
    got = sha256(DRIVER)
    assert got == DRIVER_SHA256, f"the AR-2 driver copy changed: {got}"
    spec = importlib.util.spec_from_file_location("v8f_lora_identity_dev", str(DRIVER))
    m = importlib.util.module_from_spec(spec)
    sys.modules["v8f_lora_identity_dev"] = m
    spec.loader.exec_module(m)
    return m


def ledger_last():
    last = {}
    for ln in open(LEDGER, encoding="utf-8"):
        if ln.strip():
            d = json.loads(ln)
            last[d["clip"]] = d
    return last


def compare(x, y):
    """x, y fp8 e4m3fn [T, 40, 1024]. The driver's compare_pooled plus the bitwise fraction over whole 1024-vectors (tokens)."""
    r = D.compare_pooled(x, y)
    eq = (x.view(torch.uint8) == y.view(torch.uint8)).all(dim=-1)
    r["frac_bitwise_equal_tokens"] = float(eq.double().mean())
    r["n_tokens"] = int(eq.numel())
    r["n_values_differ"] = int(r["n_values"] - round(r["frac_bitwise_equal"] * r["n_values"]))
    return r


def encode_module(enc, o, n, pool_field):
    """MODULE path, the AR-2 driver's own steps; also returns the fp32 pooled field before the fp8 rounding."""
    feats = []
    with torch.inference_mode():
        for j in range(0, n, D.BUILDER_BATCH):
            feats.append(enc.field(E.decode_cache_frames(o, list(range(j, min(j + D.BUILDER_BATCH, n))))).cpu())
    f = torch.cat(feats)
    assert bool(torch.isfinite(f).all())
    pooled = pool_field(f, 4, 10)
    return pooled, pooled.to(torch.float8_e4m3fn).contiguous()


def encode_builder(o, model, mean, std, n_special, pep, ship):
    f = ship.encode_episode(o, model, mean, std, n_special, "cuda")
    pooled = pep.pool_field(f, 4, 10)
    return pooled, pooled.to(torch.float8_e4m3fn).contiguous()


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def pixel_diff(oa, ob, n):
    """Cache frames only (raw frame 2g): fraction of differing u8 values, max abs, signed mean."""
    tot = dif = 0
    mx = 0
    sgn = 0.0
    nfr_diff = 0
    for g in range(n):
        xa = E.decode_cache_frames(oa, [g]).to(torch.int16)
        xb = E.decode_cache_frames(ob, [g]).to(torch.int16)
        d = xb - xa
        c = int((d != 0).sum())
        tot += d.numel()
        dif += c
        nfr_diff += c > 0
        mx = max(mx, int(d.abs().max()))
        sgn += float(d.sum())
    return {"n_frames": n, "frames_with_any_diff": nfr_diff, "frac_values_differ": dif / tot, "max_abs_level": mx, "signed_mean_level": sgn / tot}


def main():
    t_all = time.time()
    global D, E
    D = load_driver()
    E = D.E
    res = {"what": "PREREG_V8F_LORA pod token-level identity control", "started": time.strftime("%F %T"), "tau_device": TAU,
           "torch": torch.__version__, "cuda_device": torch.cuda.get_device_name(0), "cuda_capability": list(torch.cuda.get_device_capability(0)),
           "cuda_runtime": torch.version.cuda, "driver_sha256": sha256(DRIVER), "python": platform.python_version(),
           "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
           "cpu": platform.processor() or platform.machine(), "torch_threads": torch.get_num_threads()}
    import transformers
    res["transformers"] = transformers.__version__
    assert torch.__version__ == "2.11.0+cu128", torch.__version__
    pep, ship = E.load_builder()
    res["pep_sha256"] = sha256(pep.__file__)
    assert res["pep_sha256"].startswith(PEP_SHA256_PREFIX), res["pep_sha256"]
    res["ship_sha256"] = sha256(ship.__file__)
    res["builder_encode_episode_sha256"] = E.builder_encode_episode_sha256()
    res["builder_unchanged"] = res["builder_encode_episode_sha256"] == E.BUILDER_ENCODE_EPISODE_SHA256
    assert res["builder_unchanged"]
    pool_field = E.builder_pool_field()

    enc = D.build_encoder("cuda")
    lv = enc.built_levers()
    assert lv["lora_B_zero_at_build"] and enc.trainable_count() == D.EXPECTED_TRAINABLE and str(lv["backbone_dtype"]) == "torch.bfloat16"
    res.update(weights_sha256=lv["weights_sha256"], backbone_dtype=str(lv["backbone_dtype"]), trainable=enc.trainable_count(),
               attn_implementation=getattr(getattr(enc.backbone, "config", None), "_attn_implementation", None))
    assert lv["weights_sha256"] == E.BUILDER_MODEL_SHA256
    model, mean, std, n_special = pep.build_model("cuda", "bf16")      # the builder's own construction (no LoRA wrappers)

    ledger = ledger_last()
    clips = {}
    mutant_done = False
    for role, lst in (("decision", DECISION), ("supplementary", EXTRA)):
        for split, clip in lst:
            t0 = time.time()
            pa = FRAMES_A / split / f"{clip}.v2ep.pt"
            pb = V2EP_B / split / f"{clip}.v2ep.pt"
            pc = CACHE_C / split / f"{clip}.pt"
            sa, sb = sha256(pa), sha256(pb)
            want_a = BANKED["gate"].get(f"{split}/{clip}") or BANKED["extra"].get(f"{split}/{clip}")
            assert sa == want_a, f"{clip}: banked frames sha {sa} != {want_a}"
            led = ledger.get(clip)
            r = {"split": split, "role": role, "frames_a_sha256": sa, "frames_b_sha256": sb, "frames_a_equals_banked_hash": sa == want_a,
                 "frames_b_equals_banked_hash": sb == want_a, "frames_b_equals_ledger": bool(led and led["sha256"] == sb),
                 "stored_c_md5": hashlib.md5(open(pc, "rb").read()).hexdigest()}
            oa = torch.load(pa, map_location="cpu", weights_only=False)
            ob = torch.load(pb, map_location="cpu", weights_only=False)
            for tag, o in (("a", oa), ("b", ob)):
                assert o["clip_id"] == clip and str(o["codec"]) == "png"
            n = E.n_cache_frames(oa)
            assert n == E.n_cache_frames(ob)
            c = torch.load(pc, map_location="cpu", weights_only=True)
            ref = c["pool4x10"]
            assert ref.dtype == torch.float8_e4m3fn and tuple(ref.shape) == (n, 40, 1024), (ref.dtype, tuple(ref.shape), n)
            r["n_frames"] = n
            r["frame_pixels_a_vs_b"] = pixel_diff(oa, ob, n)
            # ---- MODULE path (primary)
            pa_f, fa = encode_module(enc, oa, n, pool_field)
            pb_f, fb = encode_module(enc, ob, n, pool_field)
            pa2_f, fa2 = encode_module(enc, oa, n, pool_field)
            # the driver's own encode_pooled must give the same tensor as my local copy of its steps
            fa_drv = D.encode_pooled(enc, oa, list(range(n)), pool_field)
            r["module_local_equals_driver_encode_pooled"] = bool(torch.equal(fa.view(torch.uint8), fa_drv.view(torch.uint8)))
            r["module"] = {"a_vs_c": compare(fa, ref), "b_vs_c": compare(fb, ref), "a_vs_b": compare(fa, fb),
                           "repeat_a_vs_a": compare(fa2, fa),
                           "fp32_pooled_rel_l2_a_vs_b": rel(pa_f, pb_f), "fp32_pooled_rel_l2_a_vs_c": rel(pa_f, ref.float()), "fp32_pooled_rel_l2_b_vs_c": rel(pb_f, ref.float())}
            # ---- BUILDER path (cross-check)
            ba_f, ga = encode_builder(oa, model, mean, std, n_special, pep, ship)
            bb_f, gb = encode_builder(ob, model, mean, std, n_special, pep, ship)
            r["builder"] = {"a_vs_c": compare(ga, ref), "b_vs_c": compare(gb, ref), "a_vs_b": compare(ga, gb),
                            "module_vs_builder_on_a": compare(fa, ga), "module_vs_builder_on_b": compare(fb, gb)}
            # ---- instrument sensitivity (once): the AR-2 pixel-scale mutant must read >> tau
            if not mutant_done:
                restore = D.apply_mutant(enc, D.DEFAULT_MUTANT, torch.device("cuda"))
                try:
                    _, fm = encode_module(enc, oa, n, pool_field)
                finally:
                    restore()
                r["mutant_PIXEL_SCALE_255_to_256_a_vs_c"] = compare(fm, ref)
                mutant_done = True
            r["seconds"] = time.time() - t0
            clips[clip] = r
            m = r["module"]
            print(f"{role:13s} {split:5s} {clip[:8]} MODULE a-c {m['a_vs_c']['rel_l2']:.3e} b-c {m['b_vs_c']['rel_l2']:.3e} a-b {m['a_vs_b']['rel_l2']:.3e} | "
                  f"BUILDER a-c {r['builder']['a_vs_c']['rel_l2']:.3e} b-c {r['builder']['b_vs_c']['rel_l2']:.3e} | rep {m['repeat_a_vs_a']['rel_l2']:.1e} "
                  f"mod-bld {r['builder']['module_vs_builder_on_b']['rel_l2']:.1e} ({r['seconds']:.1f}s)", flush=True)
            res["clips"] = clips
            res["partial"] = True
            json.dump(res, open(OUT / "identity_pod.partial.json", "w"), indent=1, default=str)
            del oa, ob, c

    # ---- the decision rule, applied as fixed
    dec = {}
    for path in ("module", "builder"):
        worst = max((clips[c][path]["b_vs_c"]["rel_l2"], c) for _, c in DECISION)
        dec[path] = {"b_vs_c_rel_l2": {c[:8]: clips[c][path]["b_vs_c"]["rel_l2"] for _, c in DECISION}, "worst": worst[0], "worst_clip": worst[1][:8],
                     "all_le_tau": all(clips[c][path]["b_vs_c"]["rel_l2"] <= TAU for _, c in DECISION)}
    res["decision_rule"] = "B KEPT iff b-vs-c rel-L2 <= 1e-3 on all 3 decision clips (8f387977, bf5ec76e, dca157a0); else B' ROUTE"
    res["decision_inputs"] = dec
    res["decision_primary_path"] = "module"
    res["decision"] = "B KEPT" if dec["module"]["all_le_tau"] else "B' ROUTE"
    res["decision_paths_agree"] = dec["module"]["all_le_tau"] == dec["builder"]["all_le_tau"]
    res["controls"] = {
        "repeat_bit_equal_all_clips": all(clips[c]["module"]["repeat_a_vs_a"]["rel_l2"] == 0.0 and clips[c]["module"]["repeat_a_vs_a"]["frac_bitwise_equal"] == 1.0 for c in clips),
        "module_equals_builder_bitwise_all_clips": all(clips[c]["builder"]["module_vs_builder_on_a"]["frac_bitwise_equal"] == 1.0 and clips[c]["builder"]["module_vs_builder_on_b"]["frac_bitwise_equal"] == 1.0 for c in clips),
        "module_local_equals_driver_all_clips": all(clips[c]["module_local_equals_driver_encode_pooled"] for c in clips),
        "mutant_caught": next(v["mutant_PIXEL_SCALE_255_to_256_a_vs_c"]["rel_l2"] for v in clips.values() if "mutant_PIXEL_SCALE_255_to_256_a_vs_c" in v) > TAU * 10}
    res["partial"] = False
    res["seconds"] = time.time() - t_all
    json.dump(res, open(OUT / "identity_pod.json", "w"), indent=1, default=str)
    print("DECISION", res["decision"], json.dumps(dec["module"]["b_vs_c_rel_l2"]), "controls", res["controls"], flush=True)

    # ---- throughput (item 5): the real encoder, no grad, batch 8, on this GPU
    bench = {}
    o = torch.load(FRAMES_A / "train" / "8f387977-4086-466d-bba4-b04f8cdc0743.v2ep.pt", map_location="cpu", weights_only=False)
    x8 = E.decode_cache_frames(o, list(range(8))).cuda()
    xn = E.normalise(x8, enc.mean, enc.std)

    def timed(fn, warm=15, iters=120):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        return {"ms_per_batch8": 1000 * dt / iters, "ms_per_frame": 1000 * dt / iters / 8, "frames_per_s": 8 * iters / dt, "iters": iters,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    base_alloc = torch.cuda.memory_allocated() / 2**30
    with torch.inference_mode():
        bench["builder_model_forward_gpu_only"] = timed(lambda: model(pixel_values=xn.to(torch.bfloat16)).last_hidden_state)
        bench["builder_encode_path_gpu_to_cpu_fp32"] = timed(lambda: model(pixel_values=xn.to(torch.bfloat16)).last_hidden_state[:, 5:, :].cpu().float())
        bench["lora_module_B0_forward_gpu_only"] = timed(lambda: enc.field(x8))
    bench["resident_weights_gib_both_models"] = base_alloc
    # one real clip end to end through the builder's encode_episode (PNG decode + 13 busy CPU workers elsewhere -> pessimistic on the CPU side)
    ob = torch.load(V2EP_B / "train" / "8f387977-4086-466d-bba4-b04f8cdc0743.v2ep.pt", map_location="cpu", weights_only=False)
    torch.cuda.synchronize()
    t = time.perf_counter()
    f = ship.encode_episode(ob, model, mean, std, n_special, "cuda")
    t_enc = time.perf_counter() - t
    t = time.perf_counter()
    pooled = pep.pool_field(f, 4, 10)
    p8 = pooled.to(torch.float8_e4m3fn)
    p8b = pep.pool_field(f, 8, 20).to(torch.float8_e4m3fn)
    t_pool = time.perf_counter() - t
    bench["clip_end_to_end_busy_cpu"] = {"frames": int(f.shape[0]), "encode_episode_s": t_enc, "encode_ms_per_frame": 1000 * t_enc / f.shape[0],
                                          "pool_both_grids_and_fp8_s": t_pool, "note": "measured while the 13-worker frame build keeps the CPU quota busy"}
    bench["projection_470k_frames"] = {k: bench[k]["ms_per_frame"] * 470_000 / 1000 / 3600 for k in
                                       ("builder_model_forward_gpu_only", "builder_encode_path_gpu_to_cpu_fp32", "lora_module_B0_forward_gpu_only")}
    bench["projection_470k_frames_unit"] = "hours of GPU time at the measured ms/frame (encoder only; PNG decode and CPU pooling overlap or add on the CPU side)"
    res["throughput"] = bench
    json.dump(res, open(OUT / "identity_pod.json", "w"), indent=1, default=str)
    print("THROUGHPUT", json.dumps({k: (round(v["ms_per_frame"], 3), round(v["peak_allocated_gib"], 3)) for k, v in bench.items() if isinstance(v, dict) and "ms_per_frame" in v}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
