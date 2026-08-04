"""LATENCY vs FAN WIDTH on the deployment target (Jetson AGX Thor).

Prereg §7: a fan-width result is a COMPUTE claim as much as an accuracy one.

⚠️ WARM UP FIRST AND REPORT p50/p95, NEVER A FIRST CALL. A 224.98 ms render
figure in this programme turned out to be a first call against a 0.10-0.17 s
steady state.

⚠️ On Thor ``mem_get_info`` / ``free`` / ``tegrastats`` / ``VmRSS`` all lie. The
only admissible memory number is in-process ``torch.cuda.max_memory_allocated``.

⚠️ Thor inverts A40 batching instincts (saturates at batch 8), so this times
BATCH 1 — the deployed single-frame case — and reports the encoder cost beside
the decoder cost, because only the DECODER scales with N. A speed-up quoted on
the decoder alone would overstate the end-to-end win.

Run ON Thor:
    OMP_NUM_THREADS=6 python thor_fan_latency.py --ckpt /home/nvidia/models/refc-xl/ckpt.pt \
        --preset xl --out /home/nvidia/fan_latency_xl.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("OMP_NUM_THREADS", "6")
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
sys.path.insert(0, "/home/nvidia/TanitAD/stack")
sys.path.insert(0, "/home/nvidia/TanitAD/taniteval")

WARMUP = 15
ITERS = 60
LADDER = [8, 16, 23, 32, 46, 48, 64, 92, 96, 128, 192, 256]


def build_model(ckpt: str, preset: str):
    from taniteval.loaders import _apply_overrides
    from tanitad.refs.refc import (RefCModel, refc_config, refc_small_config,
                                   refc_xl_config)
    presets = {"small": refc_small_config, "base": refc_config,
               "xl": refc_xl_config}
    cfg = presets[preset]()
    cj = Path(ckpt).parent / "config.json"
    if cj.exists():
        _apply_overrides(cfg, json.loads(cj.read_text()).get("cfg", {}))
    model = RefCModel(cfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    miss, unexp = model.load_state_dict(ck.get("model", ck), strict=False)
    return model, cfg, ck.get("step"), list(miss), list(unexp)


def timed(fn, dev, warmup=WARMUP, iters=ITERS) -> dict:
    """p50/p95 over ``iters`` timed calls AFTER ``warmup`` untimed ones."""
    for _ in range(warmup):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"p50_ms": round(ts[len(ts) // 2], 4),
            "p95_ms": round(ts[int(0.95 * (len(ts) - 1))], 4),
            "min_ms": round(ts[0], 4), "max_ms": round(ts[-1], 4),
            "n_warmup": warmup, "n_timed": iters,
            "⚠️": "p50/p95 after warm-up; a first call is NOT quotable"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--preset", required=True, choices=["small", "base", "xl"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2)
    a = ap.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, step, miss, unexp = build_model(a.ckpt, a.preset)
    model = model.to(dev).eval()
    dec = model.decoder
    n_full = int(dec.anchors.shape[0])

    res = {"what": "REF-C latency vs fan width on the deployment target",
           "host": platform.node(), "arch": platform.machine(),
           "torch": torch.__version__,
           "device": (torch.cuda.get_device_name(0) if dev.type == "cuda"
                      else "cpu"),
           "ckpt": a.ckpt, "ckpt_step": step, "preset": a.preset,
           "n_anchors_full": n_full, "diffusion_steps": a.steps,
           "batch": 1,
           "sd_missing": len(miss), "sd_unexpected": len(unexp),
           "protocol": {"warmup": WARMUP, "timed_iters": ITERS,
                        "sync": "torch.cuda.synchronize() around every region",
                        "⚠️": ("mem_get_info / free / tegrastats / VmRSS all "
                               "LIE on Thor — only in-process "
                               "torch.cuda.max_memory_allocated is admissible")},
           "rungs": []}

    # one real image batch shape, from the arm's own raster (256x256, 9ch)
    x = torch.randn(1, 9, 256, 256, device=dev)
    with torch.no_grad():
        # ---- the N-INDEPENDENT half: the encoder --------------------------- #
        def enc():
            with torch.no_grad():
                return model.encoder(x)
        try:
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            res["encoder_ms"] = timed(enc, dev)
            fmap = enc()
            if isinstance(fmap, (tuple, list)):
                fmap = fmap[0]
            if isinstance(fmap, dict):
                fmap = fmap.get("fmap", next(iter(fmap.values())))
        except Exception as exc:
            res["encoder_ms"] = {"status": f"UNAVAILABLE — {type(exc).__name__}: {exc}"}
            fmap = torch.randn(1, dec.feat_proj.in_features, 8, 8, device=dev)
        res["encoder_feat_dim"] = int(model.encoder.feat_dim)
        res["decoder_feat_proj_in"] = int(dec.feat_proj.in_features)
        res["fmap_shape"] = list(fmap.shape)
        res["encoder_note"] = (
            "the encoder does NOT scale with N. Only the decoder does, so an "
            "end-to-end speed-up is bounded by decoder_share = "
            "decoder / (encoder + decoder).")

        d_meas = dec.cond_proj.in_features
        m = torch.zeros(1, d_meas, device=dev)
        full_anchors = dec.anchors.detach().clone()

        for n in [k for k in LADDER if k <= n_full] + [n_full]:
            dec.anchors = full_anchors[:n].contiguous()
            if dec.maneuver_to_anchor is not None:
                w = dec.maneuver_to_anchor.weight.detach()
                dec.maneuver_to_anchor.weight = torch.nn.Parameter(
                    w[:n].contiguous(), requires_grad=False)

            def fwd(n=n):
                with torch.no_grad():
                    return dec(fmap, m, steps=a.steps)
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            try:
                t = timed(fwd, dev)
                mem = (round(torch.cuda.max_memory_allocated() / 2**20, 2)
                       if dev.type == "cuda" else None)
            except Exception as exc:
                t = {"status": f"FAILED — {type(exc).__name__}: {exc}"}
                mem = None
            res["rungs"].append({
                "n_anchors": n, "decoder": t,
                "decoder_peak_mem_MiB_in_process": mem})
            # restore the graft weight for the next rung
            if dec.maneuver_to_anchor is not None:
                dec.maneuver_to_anchor.weight = torch.nn.Parameter(
                    torch.zeros(n_full, dec.maneuver_to_anchor.in_features,
                                device=dev), requires_grad=False)
            print(f"  N={n:>4} decoder p50={t.get('p50_ms')} ms "
                  f"p95={t.get('p95_ms')} ms mem={mem} MiB", flush=True)
        dec.anchors = full_anchors

    # derived: end-to-end Hz at each rung, and the decoder share
    e = res.get("encoder_ms", {}).get("p50_ms")
    for r in res["rungs"]:
        p = r["decoder"].get("p50_ms")
        if p is None:
            continue
        r["decoder_only_hz"] = round(1000.0 / p, 2)
        if e:
            r["end_to_end_p50_ms"] = round(e + p, 4)
            r["end_to_end_hz"] = round(1000.0 / (e + p), 2)
            r["decoder_share_of_total"] = round(p / (e + p), 4)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False),
                           encoding="utf-8")
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
