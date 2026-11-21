"""E-DEPLOY-3 - TF32: a free lever the compile probe surfaced.

WHY. Inductor warned on Thor: "TensorFloat32 tensor cores for float32 matrix
multiplication available but not enabled." TF32 keeps fp32 range and fp32
storage and reduces only the MANTISSA of the matmul accumulate path - a much
smaller numerical step than fp16 autocast, and it needs no autocast wrapper, no
dtype policy, and no changes to the kinematic integrator (which is not a matmul
and is therefore untouched by definition).

PRE-REGISTERED (both outcomes before the run):
  H-DEPLOY-6: enabling TF32 is a net win on the Thor v7-tiny tick.
    SUPPORTED : median latency improves by >= 5 % vs fp32 AND the plan deviation
                is strictly SMALLER than fp16 autocast's 8.663e-02 m. Then TF32
                enters the playbook as a rung between fp32 and fp16.
    REFUTED   : < 5 % gain (this model is not matmul-bound, consistent with
                H-DEPLOY-3), or deviation >= the fp16 bar. Then TF32 is dropped
                and the playbook is unchanged.

⚠️ TF32 affects MATMULS ONLY. The integrator finding (D-DEPLOY-INTEG) predicts
its waypoint deviation should be far below fp16's, because `unicycle_rollout`
never sees a reduced dtype here. That is a falsifiable prediction of today's
mechanism, tested for free by this arm.

Also measures TF32 x CUDA-graph, since graphs were today's winning lever.
Installs nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

STACK = os.path.expanduser("~/TanitAD/stack")
sys.path.insert(0, STACK)
sys.path.insert(0, os.path.join(STACK, "scripts"))

import torch  # noqa: E402

R = {"spec": "E-DEPLOY-3", "controls": {}, "arms": []}


def timeit(fn, *, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return {"n": len(ts), "median_ms": statistics.median(ts) * 1e3,
            "p90_ms": ts[int(0.9 * (len(ts) - 1))] * 1e3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/v7tiny/champ30k"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dv = torch.device("cuda")

    R["env"] = {"torch": torch.__version__,
                "device": torch.cuda.get_device_name(0),
                "default_matmul_precision":
                    torch.get_float32_matmul_precision(),
                "cudnn_allow_tf32_default": bool(torch.backends.cudnn.allow_tf32),
                "matmul_allow_tf32_default":
                    bool(torch.backends.cuda.matmul.allow_tf32)}
    print(json.dumps(R["env"], indent=1), flush=True)

    with open(os.path.join(a.ckpt, "config.json")) as fh:
        cfg = json.load(fh)
    from train_v6_staged import build_stack_from_args, synthetic_train_batch
    stack = build_stack_from_args(argparse.Namespace(**cfg["args"]))
    ck = torch.load(os.path.join(a.ckpt, "ckpt.pt"), map_location="cpu",
                    weights_only=False)
    res = stack.load_state_dict(ck["stack"], strict=False)
    ok = not res.missing_keys and not res.unexpected_keys
    R["controls"]["state_dict_loads_clean"] = {
        "expect": "0/0", "n_missing": len(res.missing_keys),
        "n_unexpected": len(res.unexpected_keys), "pass": bool(ok)}
    del ck
    if not ok:
        R["voided"] = "state_dict mismatch"
        json.dump(R, open(a.out, "w"), indent=1)
        return
    stack = stack.to(dv).eval()
    for p in stack.parameters():
        p.requires_grad_(False)

    batch = synthetic_train_batch(stack, batch=1, k=4, seed=0, device=dv)
    frames, acts, v0 = batch["frames"], batch["actions2"], batch["v0"]
    A = stack.cfg.predictor.action_dim
    if acts.shape[-1] != A:
        acts = torch.cat([acts, torch.zeros(acts.shape[0], acts.shape[1],
                                            A - acts.shape[-1], device=dv,
                                            dtype=acts.dtype)], dim=-1)

    def fwd():
        with torch.no_grad():
            return stack(frames, acts, v0)

    # ---- strict fp32 reference ------------------------------------------- #
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    ref = fwd()["plan"]["waypoints"].float().clone()
    same = bool(torch.equal(fwd()["plan"]["waypoints"].float(), ref))
    R["controls"]["fp32_deterministic"] = {"expect": "bit-identical repeat",
                                           "measured": same, "pass": same}
    print("CONTROL determinism:", same, flush=True)
    base = timeit(fwd, warmup=15, iters=40)
    R["arms"].append({"arm": "fp32_highest", **base, "deviation_m": 0.0})
    print("fp32 (highest) median %.3f ms" % base["median_ms"], flush=True)

    # ---- CONTROL: the TF32 switch must actually CHANGE something ---------- #
    # A flag that flips with no numerical effect anywhere would make a null
    # result unattributable - it could mean "no gain" or "never applied".
    x = torch.randn(2048, 2048, device=dv)
    y = torch.randn(2048, 2048, device=dv)
    m_hi = (x @ y).clone()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    m_tf = (x @ y).clone()
    delta = float((m_tf - m_hi).abs().max().item())
    R["controls"]["tf32_switch_is_live"] = {
        "expect": "a 2048^2 fp32 matmul changes when TF32 is enabled",
        "max_abs_change": delta, "pass": bool(delta > 0.0)}
    print("CONTROL tf32 switch live: max change %.4e -> %s"
          % (delta, delta > 0.0), flush=True)
    del x, y, m_hi, m_tf
    torch.cuda.empty_cache()

    # ---- TF32 arms -------------------------------------------------------- #
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    wp = fwd()["plan"]["waypoints"].float()
    d = float((wp - ref).abs().max().item())
    t = timeit(fwd, warmup=15, iters=40)
    R["arms"].append({"arm": "tf32_high", **t, "deviation_m": d,
                      "finite": bool(torch.isfinite(wp).all().item()),
                      "speedup_vs_fp32":
                          round(base["median_ms"] / t["median_ms"], 3)})
    print("tf32 (high) median %.3f ms  dev %.4e m  %.3fx"
          % (t["median_ms"], d, base["median_ms"] / t["median_ms"]), flush=True)

    # TF32 + CUDA graph
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                fwd()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(g):
                so = stack(frames, acts, v0)
        torch.cuda.synchronize()
        tg = timeit(g.replay, warmup=15, iters=40)
        dg = float((so["plan"]["waypoints"].float() - ref).abs().max().item())
        R["arms"].append({"arm": "tf32_high_cuda_graph", **tg,
                          "deviation_m": dg,
                          "speedup_vs_fp32":
                              round(base["median_ms"] / tg["median_ms"], 3)})
        print("tf32 + graph median %.3f ms  dev %.4e m  %.3fx"
              % (tg["median_ms"], dg, base["median_ms"] / tg["median_ms"]),
              flush=True)
    except Exception as e:
        R["arms"].append({"arm": "tf32_high_cuda_graph", "error": repr(e)[:400]})
        print("tf32+graph FAILED", repr(e)[:200], flush=True)

    # ---- verdict ---------------------------------------------------------- #
    tf = [x for x in R["arms"] if x["arm"] == "tf32_high"][0]
    gain = 1.0 - tf["median_ms"] / base["median_ms"]
    FP16_BAR = 8.663e-02
    R["fp16_deviation_bar_m"] = FP16_BAR
    R["verdict"] = ("H-DEPLOY-6 %s — TF32 gain %.1f %% (bar 5 %%), deviation "
                    "%.4e m vs fp16 bar %.4e m"
                    % ("SUPPORTED" if (gain >= 0.05 and
                                       tf["deviation_m"] < FP16_BAR)
                       else "REFUTED", gain * 100, tf["deviation_m"], FP16_BAR))
    R["integrator_prediction_check"] = {
        "prediction": "TF32 touches matmuls only, so waypoint deviation should "
                      "be far below fp16 autocast's (which downcast the "
                      "integrator's inputs)",
        "tf32_dev_m": tf["deviation_m"], "fp16_dev_m": FP16_BAR,
        "ratio_tf32_over_fp16": tf["deviation_m"] / FP16_BAR,
        "prediction_holds": bool(tf["deviation_m"] < FP16_BAR)}
    print(R["verdict"], flush=True)
    R["_evidence_class"] = "MEASURED (ours; Thor, in-process probes only)"
    json.dump(R, open(a.out, "w"), indent=1)
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
