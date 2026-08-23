"""B1 — the COMPLETE-tick question, and the fastpath-graph question, in one run.

Plan ref: `Production & Optimization/THOR_OPTIMISATION_PLAN.md` §3 Tier B (B1), findings F1 + F4.

TWO pre-registered questions, both about the published 5.33x / 51.2 ms:

Q1 (F1) — THE FAN. The published tick rolls ONE candidate. The deployed selector loops over
   NINE (`config.py:95` n_maneuvers=9; `fourbrain.py:571` `for m in maneuvers:`). So the real
   imagination cost is M x K predictor steps, not 1 x K.
     * FALSIFIER: if a batch-9 roll costs > 1.1x a batch-1 roll's *per-candidate* time — i.e. the
       fan does NOT amortise — then the 9 candidates are ~9x and the tick blows the 100 ms budget
       (projected ~238 ms), and the maneuver vocabulary itself becomes a deployment parameter.
     * PRIOR (MEASURED, 4060, `latency_cnce_baseline.py:71-80`): a batch-9 `imagine` cost 0.93x of
       a batch-1 one -- nine candidates for less than the price of one. Expect the same mechanism
       here (small tensors, launch-bound), but Thor's encoder was compute-bound where the 4060's
       was not, so it MUST be measured, not transferred.

Q2 (F4) — THE GRAPH THE ENGINE WAS BUILT FROM. `thor_trt.py:80-82` exported at opset 17 with the
   MHA fastpath left ON, and that run produced the published 1.168 ms fp16 median -> the 5.33x.
   The LATER gate (`thor_trt_gate.json`) proved that same export path is SILENTLY WRONG
   (rel-err 0.726, and `thor_trt_accuracy.json` at 18:13 shows the identical-across-precisions
   signature the runbook itself names as a wiring bug). The gate re-exported with
   `set_fastpath_enabled(False)` and passed -- but NOBODY RE-TIMED THE CORRECTED ENGINE.
     * FALSIFIER: if the fastpath-OFF engine is > 1.1x slower than 1.168 ms, the published
       5.33x was measured on a graph that computes the wrong thing and the real speedup is lower.
     * The corrected graph decomposes MHA into more ops, so "TRT re-fuses it and the timing is
       unchanged" is a hypothesis, not a given.

Accuracy beside speed (G-P2): every graph is checked against the eager fp32 reference with
onnxruntime before any timing claim is read.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.expanduser("~/TanitAD/stack"))
sys.path.insert(0, os.path.expanduser("~/TanitAD/stack/scripts"))
sys.path.insert(0, "/usr/lib/python3.12/dist-packages")   # TRT bindings are system-level

import dataclasses
from types import SimpleNamespace

import torch

from tanitad.config import flagship4b_config
from tanitad.models.fourbrain import WorldModel
from train_flagship_v4 import resolve_v2_frames

DEV, H, W = "cuda", 176, 624
K_ROLL = 20
M_FAN = 9                      # config.py:95 n_maneuvers (3 steer x 3 accel)
OUT = os.path.expanduser("~/trt_b1")
os.makedirs(OUT, exist_ok=True)

out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "plan_ref": "THOR_OPTIMISATION_PLAN.md Tier B / B1 (F1 + F4)",
       "thor_repo_sha": subprocess.run(["git", "-C", os.path.expanduser("~/TanitAD"),
                                        "rev-parse", "--short", "HEAD"],
                                       capture_output=True, text=True).stdout.strip(),
       "provenance": "imported surface (tanitad/models, config.py, resolve_v2_frames) verified "
                     "IDENTICAL to repo HEAD -- the pod-drift check, done before the launch"}
try:
    import tensorrt as trt
    out["trt_version"] = trt.__version__
except Exception as e:
    out["trt_version"] = f"IMPORT FAILED {e}"

cfg = flagship4b_config()
ns = SimpleNamespace(frame_h=256, frame_w=640, frame_hfov=120.0,
                     projection="cylindrical", v2_subframe="176x624", f_ref=None)
resolve_v2_frames(ns, cfg, label="b1")
cfg.speed_input = True
cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
if getattr(cfg, "tactical_pred", None) is not None:
    cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)

model = WorldModel(cfg).to(DEV).eval()
Wn, C, A = cfg.predictor.window, cfg.encoder.in_channels, cfg.predictor.action_dim
S = model.state_dim
out["shapes"] = {"window": Wn, "state_dim": S, "action_dim": A, "H": H, "W": W,
                 "K_roll": K_ROLL, "M_fan": M_FAN}


def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"p50_ms": round(ts[len(ts) // 2], 3),
            "p99_ms": round(ts[min(len(ts) - 1, int(len(ts) * 0.99))], 3)}


# ==================================================================== Q1: THE FAN
with torch.no_grad():
    frames = torch.randn(1, Wn, C, H, W, device=DEV)
    states1 = model.encode_window(frames)

    def enc_bf16():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.encode_window(frames)
    out["encoder_bf16"] = bench(enc_bf16, warmup=5, iters=20)

    fan = {}
    for B in (1, 3, 5, 9):
        stB = states1.expand(B, -1, -1).contiguous()
        acB = torch.randn(B, Wn, A, device=DEV)
        r = bench(lambda: model.predictor(stB, acB), warmup=10, iters=50)
        fan[f"step_B{B}"] = r
        fan[f"step_B{B}_per_candidate_ms"] = round(r["p50_ms"] / B, 4)
    out["eager_fan_step"] = fan

    b1 = fan["step_B1"]["p50_ms"]
    b9 = fan["step_B9"]["p50_ms"]
    out["FAN_AMORTISATION"] = {
        "batch9_vs_batch1_x": round(b9 / b1, 3),
        "per_candidate_speedup_x": round((b1 * M_FAN) / b9, 2),
        "prior_4060": "batch-9 imagine = 0.93x of batch-1 (latency_cnce_baseline.py)",
        "FALSIFIER_fires": bool(b9 / b1 > 1.1 * M_FAN / M_FAN * 1.0 and b9 > 1.1 * b1),
        "verdict": ("FAN AMORTISES -- 9 candidates cost ~1 candidate; the deployed fan is affordable"
                    if b9 < 1.5 * b1 else
                    "FAN DOES NOT AMORTISE -- the 9-candidate deployed tick is ~%.0f ms of roll"
                    % (b9 * K_ROLL))}

    # the full deployed-shape roll: M candidates x K steps, batched vs serialised
    st9 = states1.expand(M_FAN, -1, -1).contiguous()
    ac9 = torch.randn(M_FAN, Wn, A, device=DEV)
    ac1 = torch.randn(1, Wn, A, device=DEV)

    def roll_batched():
        ws = st9
        for _ in range(K_ROLL):
            z = model.predictor(ws, ac9)[1]
            ws = torch.cat([ws[:, 1:], z.unsqueeze(1)], dim=1)
        return ws

    def roll_serialised():
        for _ in range(M_FAN):
            ws = states1
            for _ in range(K_ROLL):
                z = model.predictor(ws, ac1)[1]
                ws = torch.cat([ws[:, 1:], z.unsqueeze(1)], dim=1)
        return ws

    out["eager_roll_M9_batched"] = bench(roll_batched, warmup=2, iters=10)
    out["eager_roll_M9_serialised"] = bench(roll_serialised, warmup=1, iters=5)
    out["eager_roll_M1"] = bench(
        lambda: [model.predictor(states1, ac1) for _ in range(K_ROLL)], warmup=2, iters=10)

# ============================================== Q2: THE GRAPH THE ENGINE CAME FROM
class PredWrap(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, states, actions):
        return self.p(states, actions)[1]


wrap = PredWrap(model.predictor).eval()


def export(tag, fastpath: bool, B: int):
    torch.backends.mha.set_fastpath_enabled(fastpath)
    st = torch.randn(B, Wn, S, device=DEV)
    ac = torch.randn(B, Wn, A, device=DEV)
    with torch.no_grad():
        ref = wrap(st, ac).float().cpu().numpy()
    path = f"{OUT}/pred_{tag}.onnx"
    info = {"fastpath": fastpath, "batch": B}
    try:
        torch.onnx.export(wrap, (st, ac), path,
                          input_names=["states", "actions"], output_names=["z_next"],
                          opset_version=17, dynamo=False)
        info["ok"] = True
        info["MB"] = round(os.path.getsize(path) / 1e6, 1)
    except Exception as e:
        info["ok"] = False
        info["err"] = f"{type(e).__name__}: {str(e)[:200]}"
        return info, path, None, None
    # does the graph still carry the fused op that opset 17 exports WRONG?
    try:
        import onnx
        m = onnx.load(path, load_external_data=False)
        ops = {n.op_type for n in m.graph.node}
        info["has_native_mha"] = any("MultiHeadAttention" in o or "native_multi_head" in o
                                     for o in ops)
        info["n_nodes"] = len(m.graph.node)
    except Exception as e:
        info["onnx_scan"] = f"skipped: {type(e).__name__}"
    return info, path, (st.cpu().numpy(), ac.cpu().numpy()), ref


def ort_parity(path, inputs, ref):
    """ORT-CPU vs eager -- proves WHICH graph is the correct one before any timing is read."""
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        y = s.run(["z_next"], {"states": inputs[0], "actions": inputs[1]})[0]
        import numpy as np
        num = float(np.linalg.norm(y - ref))
        den = float(np.linalg.norm(ref)) or 1e-12
        return {"rel_err_vs_eager": round(num / den, 8)}
    except Exception as e:
        return {"rel_err_vs_eager": None, "err": f"{type(e).__name__}: {str(e)[:160]}"}


def build_and_time(tag, path):
    eng = f"{OUT}/pred_{tag}.plan"
    t0 = time.perf_counter()
    r = subprocess.run(["/usr/src/tensorrt/bin/trtexec", f"--onnx={path}",
                        f"--saveEngine={eng}", "--fp16", "--skipInference"],
                       capture_output=True, text=True, timeout=2400)
    if not os.path.exists(eng):
        return {"ok": False, "err": (r.stderr or r.stdout)[-300:]}
    info = {"ok": True, "build_s": round(time.perf_counter() - t0, 1),
            "MB": round(os.path.getsize(eng) / 1e6, 1)}
    r2 = subprocess.run(["/usr/src/tensorrt/bin/trtexec", f"--loadEngine={eng}",
                         "--iterations=200", "--warmUp=500", "--avgRuns=100"],
                        capture_output=True, text=True, timeout=1200)
    for line in (r2.stdout or "").splitlines():
        if "GPU Compute Time" in line and "median" in line:
            for part in line.split(","):
                if "median" in part:
                    info["median_ms"] = float(part.split("=")[1].strip().split()[0])
    return info


trt_res = {}
for tag, fastpath, B in (("fpON_b1", True, 1), ("fpOFF_b1", False, 1), ("fpOFF_b9", False, M_FAN)):
    einfo, path, inputs, ref = export(tag, fastpath, B)
    if einfo.get("ok"):
        einfo["ort_parity"] = ort_parity(path, inputs, ref)
        einfo["engine"] = build_and_time(tag, path)
    trt_res[tag] = einfo
torch.backends.mha.set_fastpath_enabled(False)     # leave the process in the SAFE state
out["trt"] = trt_res

# ==================================================================== the verdicts
pub = 1.168        # the PUBLISHED fp16 median that produced the 5.33x
on_ms = trt_res.get("fpON_b1", {}).get("engine", {}).get("median_ms")
off_ms = trt_res.get("fpOFF_b1", {}).get("engine", {}).get("median_ms")
b9_ms = trt_res.get("fpOFF_b9", {}).get("engine", {}).get("median_ms")

out["Q2_VERDICT"] = {
    "published_ms_from_fastpathON_graph": pub,
    "reproduced_fastpathON_ms": on_ms,
    "CORRECT_graph_fastpathOFF_ms": off_ms,
    "correct_vs_published_x": round(off_ms / pub, 3) if off_ms else None,
    "falsifier_fires": bool(off_ms and off_ms > 1.1 * pub),
    "reading": None if not off_ms else (
        "⛔ THE PUBLISHED 1.168 ms WAS THE WRONG GRAPH -- the correct engine is %.3f ms (%.2fx "
        "slower); the 5.33x and the 51.2 ms tick must be restated" % (off_ms, off_ms / pub)
        if off_ms > 1.1 * pub else
        "✅ the correction is latency-neutral (%.3f vs %.3f ms) -- the published speedup survives, "
        "but it was measured on a wrong graph and this run is what makes it admissible" % (off_ms, pub))}

if off_ms and b9_ms:
    enc = out["encoder_bf16"]["p50_ms"]
    out["TICK_RECONSTRUCTION_ms"] = {
        "encoder_bf16": round(enc, 2),
        "published_1candidate_tick": round(enc + off_ms * K_ROLL, 2),
        "deployed_M9_SERIALISED_tick": round(enc + off_ms * K_ROLL * M_FAN, 2),
        "deployed_M9_BATCHED_tick": round(enc + b9_ms * K_ROLL, 2),
        "budget_ms": 100.0,
        "serialised_pct_of_budget": round(100 * (enc + off_ms * K_ROLL * M_FAN) / 100.0, 1),
        "batched_pct_of_budget": round(100 * (enc + b9_ms * K_ROLL) / 100.0, 1),
        "note": "encoder + imagination fan only. It EXCLUDES step_readout decode, scoring and the "
                "tactical/strategic head decode -- still a partial tick, but now covering the "
                "candidate dimension the published number omitted."}

print(json.dumps(out, indent=1), flush=True)
with open(os.path.expanduser("~/thor_b1_fan_and_fastpath.json"), "w") as f:
    json.dump(out, f, indent=1)
