#!/usr/bin/env python3
"""LOCAL_GPU capability probe -- what actually fits on the DEV BOX's own GPU.

MEASURES rather than estimates: every capacity below is obtained by building the
REAL `tanitad` module (no checkpoint, no corpus, no pod) and pushing batch size
until CUDA OOMs. Nothing here reads the episode cache, so parity is untouched.

`torch.compile` is NOT used -- there is no Triton on Windows (inductor fails,
dynamo-cudagraphs is ~20x slower). Everything runs eager.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "stack"))
OUT = Path(__file__).resolve().parents[1] / "raw" / "fc_localgpu.json"

DEV = "cuda"
GIB = 1024 ** 3


def _reset():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _peak_gib() -> float:
    return round(torch.cuda.max_memory_allocated() / GIB, 3)


VRAM_GIB = None          # set in main() from the device properties


def sweep(build, step, sizes, label):
    """Push batch size until OOM.

    ⚠️ On Windows/WDDM a CUDA allocation that exceeds VRAM does NOT always
    raise -- the driver silently spills into shared host memory and the run
    completes ~100x slower. `max_memory_allocated` still counts those bytes, so
    any row whose peak exceeds the card's VRAM is flagged `spilled_to_host` and
    is NOT counted as capacity. `max_batch_in_vram` is the honest answer.
    """
    rows, best, best_mem, best_ms = [], None, None, None
    for b in sizes:
        _reset()
        m = None
        try:
            m = build()
            t0 = time.time()
            step(m, b)
            torch.cuda.synchronize()
            ms = (time.time() - t0) * 1e3
            pk = _peak_gib()
            spill = VRAM_GIB is not None and pk > 0.95 * VRAM_GIB
            rows.append({"batch": b, "peak_gib": pk, "ms": round(ms, 1),
                         "spilled_to_host": spill})
            if not spill:
                best, best_mem, best_ms = b, pk, round(ms, 1)
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch": b, "oom": True})
            break
        except Exception as e:                     # a real bug, not a capacity limit
            rows.append({"batch": b, "error": f"{type(e).__name__}: {e}"[:200]})
            break
        finally:
            m = None
            _reset()
    return {"label": label, "max_batch_in_vram": best,
            "peak_gib_at_max": best_mem, "ms_at_max": best_ms, "rows": rows}


def main() -> None:
    global VRAM_GIB
    assert torch.cuda.is_available(), "no CUDA on this box"
    p = torch.cuda.get_device_properties(0)
    VRAM_GIB = p.total_memory / GIB
    R = {
        "_experiment": "LOCAL_GPU capability probe (dev box)",
        "_evidence_class": "MEASURED (ours, dev box)",
        "_tier": "CONFIRMED",
        "_host": {"node": platform.node(), "python": platform.python_version(),
                  "torch": torch.__version__, "gpu": p.name,
                  "total_vram_gib": round(p.total_memory / GIB, 2),
                  "sm": f"{p.major}.{p.minor}", "sms": p.multi_processor_count},
        "_method": "batch size pushed until torch.cuda.OutOfMemoryError; the "
                   "largest batch that COMPLETED is reported with its peak "
                   "allocated bytes. Eager only -- no torch.compile (no Triton "
                   "on Windows).",
        "_parity": "no corpus, no checkpoint, no pod. Modules are built from "
                   "`stack/tanitad` source at their committed default configs.",
    }
    import tanitad.models.flagship_v4 as f4
    import tanitad.models.fourbrain as fb

    # ---- 0. reference: raw compute + allocation ceiling --------------------
    _reset()
    a = torch.randn(8192, 8192, device=DEV)
    b = torch.randn(8192, 8192, device=DEV)
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 10
    R["S1_raw_fp32_matmul_8192"] = {
        "tflops": round(2 * 8192 ** 3 / dt / 1e12, 2), "ms": round(dt * 1e3, 2),
        "peak_gib": _peak_gib()}
    del a, b, c
    _reset()

    biggest = None
    for gib in (1, 2, 3, 4, 5, 6, 6.5, 7, 7.5):
        try:
            t = torch.empty(int(gib * GIB / 4), dtype=torch.float32, device=DEV)
            biggest = gib
            del t
            _reset()
        except torch.cuda.OutOfMemoryError:
            _reset()
            break
    R["S2_largest_single_fp32_alloc_gib"] = biggest

    # ---- 1. the v4 planner HEAD -- head-only training ----------------------
    cfg = f4.v4_config()
    head = f4.FlagshipV4Head(cfg).to(DEV)
    n_head = sum(q.numel() for q in head.parameters())
    S = int(cfg.state_dim)
    W = int(getattr(cfg, "window", 8))

    P_IMAG = int(head.imag_pos.shape[0])

    def _head_inputs(bsz):
        return dict(
            states=torch.randn(bsz, W, S, device=DEV),
            v0=torch.rand(bsz, device=DEV) * 20,
            imagined=torch.randn(bsz, P_IMAG, S, device=DEV),
            vt_band=torch.zeros(bsz, dtype=torch.long, device=DEV),
            route=torch.zeros(bsz, dtype=torch.long, device=DEV),
            route_graded=torch.zeros(bsz, device=DEV),
            vt_speed=torch.rand(bsz, device=DEV) * 10)

    def _head_fwd_bwd(m, bsz):
        out = m(**_head_inputs(bsz))
        loss = sum(v.float().pow(2).mean() for v in out.values()
                   if torch.is_tensor(v) and v.is_floating_point())
        loss.backward()
        m.zero_grad(set_to_none=True)

    def _head_fwd(m, bsz):
        with torch.no_grad():
            m(**_head_inputs(bsz))

    R["S3_v4_head"] = {
        "_read": "the v4 planner head is the piece that EMITS THE FAN "
                 "(`anchor_traj` [B, 256, 20, 2]). It consumes `states` "
                 "[B, 8, 2048] -- so with the head checkpoint and the staged "
                 "per-window states, fan regeneration and any re-scoring become "
                 "a LOCAL experiment. Neither is currently in the repo (S2 of "
                 "FAN_CLIP_LOCAL.md).",
        "params": n_head, "state_dim": int(S), "window": int(W),
        "n_imagination_probe_tokens": P_IMAG,
        "n_anchors": int(getattr(cfg, "n_anchors", 0)) or None,
        "param_breakdown": {k: int(v) for k, v in f4.param_breakdown(head).items()},
        "train_fwd_bwd": sweep(lambda: head, _head_fwd_bwd,
                               [8, 32, 128, 256, 512, 1024, 2048],
                               "v4 head fwd+bwd (head-only training)"),
        "infer_no_grad": sweep(lambda: head, _head_fwd,
                               [8, 32, 128, 512, 1024, 2048, 4096],
                               "v4 head forward, no_grad (fan generation)"),
    }
    del head
    _reset()

    # ---- 2. the REAL 4-brain world models -- encode + train ---------------
    import tanitad.config as tcfg
    R["S4_world_model"] = {}
    for name, builder in (("flagship4b (the canonical 286 M stack)",
                           tcfg.flagship4b_config),
                          ("flagship4b_reduced (the ~65 M pre-check rig)",
                           tcfg.flagship4b_reduced_config)):
        try:
            scfg = builder()
            wm = fb.WorldModel(scfg).to(DEV)
        except torch.cuda.OutOfMemoryError:
            R["S4_world_model"][name] = {"build": "OOM -- weights alone do not fit"}
            _reset()
            continue
        n_wm = sum(q.numel() for q in wm.parameters())
        enc_n = sum(q.numel() for q in wm.encoder.parameters())
        pred_n = sum(q.numel() for q in wm.predictor.parameters())
        ecfg = scfg.encoder
        img = int(getattr(ecfg, "image_size", 256))
        ch = int(getattr(ecfg, "in_channels", 9))
        Wn = int(getattr(scfg.predictor, "window", 8))

        def _encode(m, bsz, img=img, ch=ch, Wn=Wn):
            x = torch.randn(bsz, Wn, ch, img, img, device=DEV)
            with torch.no_grad():
                m.encoder(x.reshape(bsz * Wn, ch, img, img))

        def _full_train(m, bsz, img=img, ch=ch, Wn=Wn):
            x = torch.randn(bsz, Wn, ch, img, img, device=DEV)
            z = m.encoder(x.reshape(bsz * Wn, ch, img, img))
            z.float().pow(2).mean().backward()

        # the IMAGINATION step -- the unit E-V5-3 costed on the pod (25.183 ms
        # at batch 256 on an A40). Measured here for the same shape.
        S_ = int(getattr(scfg.readout, "state_dim", 0) or 2048)
        A_ = int(getattr(scfg.predictor, "action_dim", 2))

        def _pred_step(m, bsz, S_=S_, A_=A_, Wn=Wn):
            st = torch.randn(bsz, Wn, S_, device=DEV)
            ac = torch.randn(bsz, Wn, A_, device=DEV)
            with torch.no_grad():
                m.predictor(st, ac)

        R["S4_world_model"][name] = {
            "predictor_step_no_grad": sweep(
                lambda: wm, _pred_step, [1, 64, 256, 1024, 2048],
                "one predictor imagination step, no_grad (the E-V5-3 cost "
                "unit; pod A40 = 25.183 ms at batch 256)"),
            "total_params": n_wm, "encoder_params": enc_n,
            "predictor_params": pred_n, "image_size": img,
            "in_channels": ch, "window": Wn,
            "weights_only_peak_gib": _peak_gib(),
            "encode_no_grad": sweep(lambda: wm, _encode, [1, 2, 4, 8, 16, 32],
                                    "ViT encode, no_grad (window of frames)"),
            "encoder_fwd_bwd": sweep(lambda: wm, _full_train, [1, 2, 4, 8, 16],
                                     "encoder fwd+bwd (full-model training)"),
        }
        del wm
        _reset()

    # ---- 3. can a 286 M-param model be TRAINED here at all? ---------------
    _reset()
    fit = {}
    for n_par in (50e6, 100e6, 200e6, 286.34e6):
        try:
            ps = [torch.nn.Parameter(torch.randn(int(n_par / 8), 8, device=DEV))]
            opt = torch.optim.AdamW(ps, lr=1e-4)
            ps[0].grad = torch.randn_like(ps[0])
            opt.step()
            fit[f"{n_par/1e6:.2f}M"] = {"fits": True, "peak_gib": _peak_gib()}
        except torch.cuda.OutOfMemoryError:
            fit[f"{n_par/1e6:.2f}M"] = {"fits": False}
        finally:
            ps = opt = None
            _reset()
    R["S5_adamw_param_state_only"] = {
        "_read": "weights + grad + 2 Adam moments ONLY -- no activations, so "
                 "this is a LOWER bound on what training a model of that size "
                 "costs. The canonical flagship is 286.34 M.",
        "rows": fit}

    # ---- 4. what the CPU already does, for the contrast --------------------
    R["S6_cpu_workloads_measured_this_session"] = {
        "taniteval_pytest": {"result": "449 passed", "wall_s": 59.06,
                             "device": "CPU"},
        "fc_gate.py (all committed v5 + registry bars)": {"wall_s": 0.44,
                                                          "device": "CPU"},
        "fc_clip.py (3 REF-C fans x 16 bands + v4 fan x 2 scorers x 16 bands, "
        "each with a B=2000 paired episode-cluster bootstrap)": {
            "wall_s": 26.88, "device": "CPU"},
    }

    R["_VERDICT"] = "probe complete"
    OUT.write_text(json.dumps(R, indent=2))
    print(json.dumps({k: v for k, v in R.items()
                      if k.startswith("S") or k == "_host"}, indent=2)[:6000])


if __name__ == "__main__":
    main()
