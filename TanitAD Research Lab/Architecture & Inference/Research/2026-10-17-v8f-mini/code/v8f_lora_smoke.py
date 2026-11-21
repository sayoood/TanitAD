"""PREREG_V8F_LORA proof 4 / plan §13.5 AR-2: the LoRA-on-DINOv3 SMOKE -- N real-data optimiser steps at batch 1, gradient checkpointing + bf16, instrumented.

    python v8f_lora_smoke.py [--steps 200] --out <dir> [--census-batches 20] [--heads-b64-steps 5] -- <the arm's trainer argv: B's argv + --frames-root <dir>>

It drives the TRAINER ITSELF (``v8f_mini_train.main``) with ``--encoder lora --batch 1 --steps N --enc-ckpt --log-every 1`` appended (the later flag wins), so the
launch gate (G-DVB, G-LIVE through the real encoder), the data path and the optimiser are the arm's, not a re-implementation. It wraps the trainer's ``lora_step`` to record,
at EVERY step: every declared loss term, the RAW (pre-clip) gradient of each parameter group -- LoRA A, LoRA B, adapter / cond / S / T / O / WM heads -- the frozen-DINOv3
control (exactly 0.0) and the frozen-prior control (exactly 0.0), the step's three phase times, the iteration period, and the peak memory.
Then it PROJECTS the hours of one 19,164-step seed at effective batch 64 on the device it ran on (EVERY projected number is ESTIMATED from the measured per-frame costs):

    step_s(64) = 64 * U * (t_phase1_per_frame + t_phase2_per_frame) + t_heads(64)
      U              mean UNIQUE frames per window in real batches of 64 (``--census-batches``, index-only, nothing decoded)
      t_phase*       the median per-frame time of the no-grad encode and of the with-grad encode + backward, from the batch-1 steps (first 5 steps = warm-up, dropped)
      t_heads(64)    the heads' forward + backward on a real batch of 64 CACHED windows (``--heads-b64-steps``), model copy, no encoder
    The optimiser step (10 M parameters) and the data-loading wait (overlapped by the prefetch thread) are not in the model; the batch-1 ``iter_s`` is reported beside it.

AR-2 CHAIN (``box2_ops/chain_v8f_lora_ar2.sh``, queue stage ``smokelora``) ADDITIONS (PREREG_V8F_LORA Amendment 2):
  * ``--amp`` is OPT-IN and the chain never passes it: autocast is UNMEASURED against the cache and Amendment 2 pins it OFF (the frozen encoder runs in the builder's pure bf16, the
    LoRA matrices in fp32). It used to be appended unconditionally; the report now records ``amp`` from the trainer argv actually run.
  * the smoke's windows come from a declared small set of train clips (``--max-clips N`` in the trainer argv, frames shipped for exactly those). The PROJECTION must not: with 4 clips a
    batch of 64 windows shares most of its frames, so U (unique frames per window) would be badly under-estimated against the real 4,512-clip run (48-clip chunks, ~1.3 windows per clip).
    The census and the heads-at-64 measurement therefore run on the trainer argv WITHOUT ``--max-clips`` (index-only census; the heads read the cache's pooled tokens), and the report
    says so (``projection_ESTIMATED.census_scope``).
  * ``identity_probe_tau_device``: the trainer's launch-gate identity probe (B = 0 on-the-fly features of the probe windows vs the cached tokens, the TRAINING path's 16-frame
    micro-batches) read against tau_device = 1e-3 (Amendment 1 item 2). Reported beside the verdict, not part of it.
  * ``peak_reserved_gib`` beside ``peak_alloc_gib`` (an 8 GB card runs out of RESERVED memory).

It does not run itself on a GPU: the GPU run is its own go. ``--cpu-fp32`` + ``--proof-frames`` make the CPU mechanics proof (real DINOv3-L weights in fp32, real PNG frames, the
trainer's synthetic world for labels): 2-3 steps, the numbers are CPU numbers and are labelled so.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[5]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import v8f_mini_train as TR  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_lora_encoder as L  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

FULL_STEPS = 19_164          # PREREG_V8F_LORA: B's step budget (S = 4,791 steps/epoch at batch 64, x 4)
VENUE_HOURS = 36.0           # PREREG_V8F_LORA proof 5: box 2 iff one seed projects <= 36 h
TAU_DEVICE = 1e-3            # PREREG_V8F_LORA Amendment 1 item 2 (unchanged by Amendment 2)


def group_grads(model, enc) -> dict:
    """RAW gradient statistics per parameter group: sum |g|, L2 norm, number of tensors with an exactly-zero gradient, tensors."""
    def st(params):
        gs = [p.grad for p in params if p.grad is not None]
        return {"abs_sum": float(sum(float(g.abs().sum()) for g in gs)), "l2": float(math.sqrt(sum(float(g.float().pow(2).sum()) for g in gs))),
                "n_tensors": len(params), "n_zero_grad_tensors": int(sum(1 for p in params if p.grad is None or float(p.grad.abs().sum()) == 0.0))}
    out = {}
    for name, mods in model.trainable_groups().items():
        out[f"head_{name}"] = st([p for m in mods for p in (m.parameters() if isinstance(m, torch.nn.Module) else [m]) if p.requires_grad])
    named = list(enc.named_parameters())
    out["lora_A"] = st([p for n, p in named if "lora_A" in n])
    out["lora_B"] = st([p for n, p in named if "lora_B" in n])
    out["control_frozen_dinov3"] = st([p for n, p in named if "lora_" not in n])           # no grad at all: must read exactly 0.0
    out["control_frozen_prior"] = st(list(model.prior.parameters()))
    return out


def make_proof_frames(root: Path, eps_dir: Path, names, n_steps: int = 60, clips=None):
    """Real PNG frames for the trainer's SYNTHETIC world: ``<root>/<syn-train-i>.v2ep.pt`` = a trimmed copy (the first/offset ``n_steps`` cache steps) of a real eval v2ep.
    Mechanics proof only: the labels and kinematics are the synthetic world's, the PIXELS are real."""
    root.mkdir(parents=True, exist_ok=True)
    srcs = sorted(Path(eps_dir).glob("*.v2ep.pt"))[:max(1, len(names) // 3 + 1)] if clips is None else [Path(eps_dir) / f"{c}.v2ep.pt" for c in clips]
    if not srcs:
        raise SystemExit(f"no v2ep in {eps_dir}")
    for i, nm in enumerate(names):
        o = torch.load(srcs[i % len(srcs)], map_location="cpu", weights_only=False)
        off = 2 * (7 * (i // len(srcs)))                                                  # distinct windows of the same clip
        lens = o["jpeg_len"]
        offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
        a, b = off, off + 2 * n_steps - 1
        assert b <= len(lens), (b, len(lens))
        o2 = {k: o[k] for k in ("codec", "image_h", "image_w")}
        o2.update(clip_id=nm, jpeg_buf=o["jpeg_buf"][offs[a]:offs[b]].clone(), jpeg_len=lens[a:b].clone())
        torch.save(o2, root / f"{nm}.v2ep.pt")
    return root


def strip_flag(argv, flag: str, nvals: int = 1) -> list:
    """``argv`` without ``flag`` and its ``nvals`` values (also the ``flag=value`` spelling)."""
    out, i = [], 0
    while i < len(argv):
        if argv[i] == flag:
            i += 1 + nvals
        elif argv[i].startswith(flag + "="):
            i += 1
        else:
            out.append(argv[i])
            i += 1
    return out


def census(argv, n_batches: int) -> dict:
    """Mean unique frames per window in real batches of 64 (index-only: nothing is decoded)."""
    a = TR.build_parser().parse_args([*argv, "--batch", "64"])
    cfg = TR.make_config(a)
    store, J, wins, clips, info = TR.build_data(a, cfg)
    ds = D.V8FMiniFrameWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    sampler = D.ChunkedSampler(wins, 64, seed=a.seed, chunk=max(4, min(48, info["lru"] - 2)), max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    U = []
    for ep in range(10_000):
        for idx in sampler.epoch(ep):
            if len(idx) < 64:
                continue
            items = [ds.item(*w) for w in idx]
            keys, _ = D.unique_frames([it["fclip"] for it in items], np.stack([it["fidx"] for it in items]))
            U.append(len(keys) / len(items))
            if len(U) >= n_batches:
                break
        if len(U) >= n_batches:
            break
    steps_per_epoch = math.ceil(info["n_windows_all_clips"] / 64)
    return {"unique_frames_per_window_mean": float(np.mean(U)), "min": float(np.min(U)), "max": float(np.max(U)), "n_batches": len(U), "hmax": cfg.hmax,
            "frames_per_window_upper_bound": cfg.hmax, "S_steps_per_epoch_b64": steps_per_epoch, "steps_total_declared": a.steps or steps_per_epoch * a.steps_mult,
            "n_windows_all_clips": info["n_windows_all_clips"], "_args": a, "_cfg": cfg, "_store": store, "_J": J, "_wins": wins, "_info": info}


def heads_at_64(c: dict, n: int, dev) -> dict:
    """The heads' forward + backward on REAL batches of 64 CACHED windows (no encoder), on a fresh copy of the model, bf16 autocast iff the trainer's --amp."""
    a, cfg, store, J, wins, info = c["_args"], c["_cfg"], c["_store"], c["_J"], c["_wins"], c["_info"]
    ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    sampler = D.ChunkedSampler(wins, 64, seed=a.seed, chunk=max(4, min(48, info["lru"] - 2)), max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    prior = M.load_p10(a.p10) if not a.synthetic else M.P10Prior.synthetic(a.seed)
    mu, sd = M.cons_stats(J)
    model = M.V8FMini(cfg, prior, mu, sd).to(dev)
    model.train()
    amp = a.amp and dev.type == "cuda"
    times, mems = [], []

    def full_batches():
        for ep in range(10_000):                                                           # a tiny world has few full batches per epoch
            for idx in sampler.epoch(ep):
                if len(idx) == 64:
                    yield idx
    gen = full_batches()
    for i in range(n + 2):
        idx = next(gen)
        b = TR.to_dev(D.collate([ds.item(*w) for w in idx]), dev)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
            out = model.loss(b, p_gt=0.5, gen=None)
        model.zero_grad(set_to_none=True)
        out["terms"]["total"].backward()
        if dev.type == "cuda":
            torch.cuda.synchronize()
            mems.append(torch.cuda.max_memory_allocated() / 2 ** 30)
        if i >= 2:                                                                          # two warm-up steps
            times.append(time.perf_counter() - t0)
    return {"heads_fwd_bwd_b64_s_median": float(np.median(times)), "n": len(times), "peak_alloc_gib_max": float(max(mems)) if mems else None}


def main(argv=None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    smoke_args, trainer_argv = (raw[:raw.index("--")], raw[raw.index("--") + 1:]) if "--" in raw else (raw, [])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--census-batches", type=int, default=20)
    ap.add_argument("--heads-b64-steps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=5, help="steps dropped from the medians")
    ap.add_argument("--cpu-fp32", action="store_true", help="CPU MECHANICS PROOF: the encoder in fp32 (bf16 on CPU is ~5x slower); the numbers are CPU numbers")
    ap.add_argument("--proof-frames", default=None, help="eval v2ep dir: make real PNG frames for the trainer's --synthetic world (mechanics proof)")
    ap.add_argument("--live-windows", type=int, default=0, help="override the launch gate's G-LIVE probe size (windows); 0 = the trainer's")
    ap.add_argument("--no-projection", action="store_true")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast over the heads AND the loss (OFF by default: Amendment 2 pins the builder's pure-bf16 numerics; "
                    "turning it on needs the same-device identity control to pass with it first)")
    a = ap.parse_args(smoke_args)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if "--out" in trainer_argv:
        raise SystemExit("do not pass --out in the trainer argv: the smoke owns it")
    targv = [*trainer_argv, "--encoder", "lora", "--batch", "1", "--accum", "1", "--steps", str(a.steps), "--enc-ckpt", *(["--amp"] if a.amp else []), "--log-every", "1",
             "--ckpt-every", "100000000", "--out", str(out / "run")]
    if "--synthetic" in targv and a.proof_frames:
        names = [f"syn-train-{i:03d}" for i in range(6)]
        fr = make_proof_frames(out / "proof_frames", Path(a.proof_frames), names)
        targv += ["--frames-root", str(fr)]
    if a.cpu_fp32:
        def raw_fp32(args, dev):
            return L.DinoV3LoRAEncoder.from_pretrained(r=args.lora_r, alpha=args.lora_alpha, targets=TR.parse_targets(args.lora_targets), seed=args.seed,
                                                      grid=args.grid[len("pool"):], device=dev, dtype=torch.float32)
        TR.build_encoder_raw = raw_fp32
    if a.live_windows:
        TR.LIVE_PROBE_WINDOWS_LORA = a.live_windows
    recs, t_last = [], [None]
    orig = TR.lora_step
    dev_holder = {}

    def wrapped(model, enc, b, frames, fmap, args, dev, amp, p_gt, chunk, step, **kw):
        cuda = dev.type == "cuda"
        dev_holder["dev"] = dev
        if cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        grads = {}
        res = orig(model, enc, b, frames, fmap, args, dev, amp, p_gt, chunk, step, timing=True, on_grads=lambda m, e: grads.update(group_grads(m, e)))
        if cuda:
            torch.cuda.synchronize()
        now = time.perf_counter()
        out_, gn, gn_l = res
        rec = {"step": step, "terms": {k: float(v) for k, v in out_["terms"].items()}, "timing": out_.get("timing", {}), "step_wall_s": now - t0,
               "iter_s": None if t_last[0] is None else now - t_last[0], "grad_norm_heads_preclip": float(gn), "grad_norm_lora_preclip": float(gn_l),
               "groups": grads, "peak_alloc_gib": (torch.cuda.max_memory_allocated() / 2 ** 30) if cuda else None,
               "peak_reserved_gib": (torch.cuda.max_memory_reserved() / 2 ** 30) if cuda else None, "n_windows": int(b["t"].shape[0])}
        if not cuda:
            try:
                import psutil
                rec["rss_gib"] = psutil.Process().memory_info().rss / 2 ** 30
            except Exception:                                                              # noqa: BLE001
                rec["rss_gib"] = None
        t_last[0] = now
        recs.append(rec)
        with open(out / "smoke_steps.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[smoke {step + 1}/{a.steps}] total {rec['terms']['total']:.3f} ade {rec['terms']['ade']:.3f} step {rec['step_wall_s']:.2f}s "
              f"U {rec['timing'].get('n_unique_frames')} B-grad {grads['lora_B']['l2']:.3e} A-grad {grads['lora_A']['l2']:.3e} "
              f"frozen-ctl {grads['control_frozen_dinov3']['abs_sum']:.1f}", flush=True)
        return res
    TR.lora_step = wrapped
    t_start = time.time()
    try:
        rc = TR.main(targv)
    finally:
        TR.lora_step = orig
    run_s = time.time() - t_start
    sc = json.load(open(out / "run" / "selfcheck.json", encoding="utf-8"))
    dev = dev_holder.get("dev", torch.device("cpu"))
    log = [json.loads(l) for l in open(out / "run" / "train_log.jsonl", encoding="utf-8")]
    declared = sorted(set(sc["live"]["terms"]))
    verdict = {"every_declared_term_finite_every_step": all(all(np.isfinite(r["terms"].get(t, float("nan"))) for t in declared) for r in recs),
               "declared_terms": declared, "n_steps_run": len(recs),
               "lora_B_grad_nonzero_every_step": all(r["groups"]["lora_B"]["abs_sum"] > 0 for r in recs),
               "lora_A_grad_exactly_zero_at_step_1": bool(recs and recs[0]["groups"]["lora_A"]["abs_sum"] == 0.0),
               "lora_A_grad_nonzero_from_step_2": all(r["groups"]["lora_A"]["abs_sum"] > 0 for r in recs[1:]),
               "every_head_group_nonzero_every_step": all(all(g["abs_sum"] > 0 for k, g in r["groups"].items() if k.startswith("head_")) for r in recs),
               "frozen_dinov3_control_exactly_zero": all(r["groups"]["control_frozen_dinov3"]["abs_sum"] == 0.0 for r in recs),
               "frozen_prior_control_exactly_zero": all(r["groups"]["control_frozen_prior"]["abs_sum"] == 0.0 for r in recs)}
    verdict["AR2_gradient_and_finite_checks_pass"] = all(v for k, v in verdict.items() if isinstance(v, bool))
    keep = recs[a.warmup:] or recs
    med = lambda f: float(np.median([f(r) for r in keep]))
    ip = sc.get("identity_probe")
    rep = {"what": "PREREG_V8F_LORA AR-2 smoke", "amp": "--amp" in targv, "device": str(dev), "cuda": torch.cuda.get_device_name(0) if dev.type == "cuda" else None, "torch": torch.__version__,
           "cpu_fp32_mechanics_proof": bool(a.cpu_fp32), "trainer_argv": targv, "rc": rc, "wall_s": run_s, "selfcheck": {k: sc[k] for k in ("declared_vs_built", "levers", "identity_probe") if k in sc},
           "live_check": sc["live"], "verdict": verdict,
           "identity_probe_tau_device": None if not ip else {"tau_device": TAU_DEVICE, "rel_l2": ip["rel_l2"], "max_abs": ip["max_abs"], "frac_bitwise_equal": ip["frac_bitwise_equal"],
                                                            "pass": bool(ip["rel_l2"] <= TAU_DEVICE), "device": str(dev), "n_frames_unique": ip["n_frames_unique"],
                                                            "scope": "B = 0 features of the launch gate's probe windows through the TRAINING path (16-frame micro-batches) vs the cached tokens; "
                                                                     "Amendment 1 item 2 tau_device; recorded beside the verdict, not part of it"},
           "measured_batch1": {
               "n_steps": len(recs), "n_steps_in_medians": len(keep), "step_wall_s_median": med(lambda r: r["step_wall_s"]),
               "iter_s_median": float(np.median([r["iter_s"] for r in keep if r["iter_s"] is not None])) if any(r["iter_s"] for r in keep) else None,
               "unique_frames_per_step_median": med(lambda r: r["timing"].get("n_unique_frames", float("nan"))),
               "phase1_encode_nograd_s_median": med(lambda r: r["timing"].get("phase1_encode_nograd_s", float("nan"))),
               "heads_fwd_bwd_s_median": med(lambda r: r["timing"].get("heads_fwd_bwd_s", float("nan"))),
               "phase2_encode_grad_s_median": med(lambda r: r["timing"].get("phase2_encode_grad_s", float("nan"))),
               "peak_alloc_gib_max": max([r["peak_alloc_gib"] for r in recs if r["peak_alloc_gib"] is not None], default=None),
               "peak_reserved_gib_max": max([r["peak_reserved_gib"] for r in recs if r.get("peak_reserved_gib") is not None], default=None),
               "rss_gib_max": max([r.get("rss_gib") or 0 for r in recs], default=None),
               "loss_first_to_last": [recs[0]["terms"]["total"], recs[-1]["terms"]["total"]] if recs else None,
               "terms_last_step": recs[-1]["terms"] if recs else None}}
    if not a.no_projection and recs:
        cargv = strip_flag(targv, "--max-clips")
        c = census(cargv, a.census_batches)
        h64 = heads_at_64(c, a.heads_b64_steps, dev)
        mb = rep["measured_batch1"]
        U1 = mb["unique_frames_per_step_median"]
        f1, f2 = mb["phase1_encode_nograd_s_median"] / U1, mb["phase2_encode_grad_s_median"] / U1
        U64 = c["unique_frames_per_window_mean"]
        step64 = 64 * U64 * (f1 + f2) + h64["heads_fwd_bwd_b64_s_median"]
        total = c["steps_total_declared"] if c["steps_total_declared"] != a.steps else FULL_STEPS
        hours = FULL_STEPS * step64 / 3600.0
        rep["projection_ESTIMATED"] = {
            "formula": "step_s(64) = 64 * U * (t_phase1_per_frame + t_phase2_per_frame) + t_heads(64); hours = steps / 3600 * step_s(64)",
            "U_unique_frames_per_window_b64": U64, "census": {k: v for k, v in c.items() if not k.startswith("_")},
            "census_scope": {"smoke_trainer_argv_has_max_clips": cargv != targv, "census_and_heads_clips": c["_info"]["n_clips"], "census_clips_all": c["_info"]["n_clips_all"],
                             "why": "a batch of 64 windows from a handful of clips shares most of its frames; U must come from the real clip list"},
            "assumes_micro_batch_frames": int(rep["selfcheck"]["levers"]["accum"]["chunk_frames"]) if "accum" in rep["selfcheck"].get("levers", {}) else None,
            "t_phase1_per_frame_s": f1, "t_phase2_per_frame_s": f2, "t_heads_b64": h64, "step_s_b64_ESTIMATED": step64, "steps": FULL_STEPS,
            "hours_one_seed_ESTIMATED": hours, "device_of_measurement": str(dev),
            "venue_rule_proof5": {"box2_if_hours_le": VENUE_HOURS, "box2_eligible_by_this_estimate": bool(hours <= VENUE_HOURS) if dev.type == "cuda" else "n/a (not measured on a GPU)"},
            "not_in_the_model": ["optimiser step", "data-loading wait (overlapped by the prefetch thread)", "the first-batch / LRU-miss decode bursts", "temperature throttling"]}
        print(f"PROJECTION (ESTIMATED, {dev}): U {U64:.2f} frames/window, step(64) {step64:.2f} s -> {hours:.1f} h for {FULL_STEPS:,} steps")
    json.dump(rep, open(out / "smoke_report.json", "w", encoding="utf-8"), indent=1, default=str)
    print("VERDICT", json.dumps(verdict))
    return 0 if (rc == 0 and verdict["AR2_gradient_and_finite_checks_pass"]) else 1


if __name__ == "__main__":
    sys.exit(main())
