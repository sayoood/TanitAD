"""E-DEPLOY-2 - does torch.compile beat manual CUDA graphs on Thor?

WHY THIS IS NEWLY POSSIBLE. The programme's standing rule is "no Triton => no
torch.compile, use manual CUDA graphs". Verified 2026-08-23: that is a DEV-BOX
fact. Thor carries **Triton 3.7.1**, and torch.compile has never been tried on
the primary deployment target.

PRE-REGISTERED (both outcomes committed before the run):
  H-DEPLOY-5: torch.compile beats manual CUDA graphs on the v7-tiny b1 tick.
    SUPPORTED  : best compiled median < 7.840 ms (today's graph result) AND the
                 output deviation vs eager fp32 is <= the fp16-autocast
                 deviation already measured (8.663e-02 m). Then compile enters
                 the playbook ABOVE graphs - but still behind the four-family
                 gate, because unlike graph replay it is NOT bit-identical.
    REFUTED    : compiled median >= 7.840 ms, or compilation fails, or the
                 deviation exceeds that bar. Then manual CUDA graphs remain the
                 Thor lever and the "no compile" rule stands for a NEW reason
                 (measured on Thor), not an inherited dev-box one.

⛔ COST NOTE: compile warmup is minutes. Warmup is timed and reported
separately - a lever whose warmup exceeds a deployment's startup budget is a
different trade than one that does not.

Installs nothing. Writes one JSON.
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

R = {"spec": "E-DEPLOY-2", "controls": {}, "arms": []}


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
    ap.add_argument("--batch", type=int, default=1)
    a = ap.parse_args()
    dv = torch.device("cuda")

    R["env"] = {"torch": torch.__version__,
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0))}
    try:
        import triton
        R["env"]["triton"] = triton.__version__
    except Exception as e:
        R["env"]["triton"] = "MISSING " + type(e).__name__

    with open(os.path.join(a.ckpt, "config.json")) as fh:
        cfg = json.load(fh)
    from train_v6_staged import build_stack_from_args, synthetic_train_batch
    stack = build_stack_from_args(argparse.Namespace(**cfg["args"]))
    ck = torch.load(os.path.join(a.ckpt, "ckpt.pt"), map_location="cpu",
                    weights_only=False)
    res = stack.load_state_dict(ck["stack"], strict=False)
    ok = not res.missing_keys and not res.unexpected_keys
    R["controls"]["state_dict_loads_clean"] = {
        "expect": "0 missing / 0 unexpected", "n_missing": len(res.missing_keys),
        "n_unexpected": len(res.unexpected_keys), "pass": bool(ok)}
    if not ok:
        R["voided"] = "state_dict mismatch"
        json.dump(R, open(a.out, "w"), indent=1)
        return
    del ck
    stack = stack.to(dv).eval()
    for p in stack.parameters():
        p.requires_grad_(False)

    B = a.batch
    batch = synthetic_train_batch(stack, batch=B, k=4, seed=0, device=dv)
    frames, acts, v0 = batch["frames"], batch["actions2"], batch["v0"]
    A = stack.cfg.predictor.action_dim
    if acts.shape[-1] != A:
        acts = torch.cat([acts, torch.zeros(acts.shape[0], acts.shape[1],
                                            A - acts.shape[-1], device=dv,
                                            dtype=acts.dtype)], dim=-1)

    def eager():
        with torch.no_grad():
            return stack(frames, acts, v0)

    ref = eager()["plan"]["waypoints"].float().clone()
    # CONTROL: eager fp32 must repeat bit-identically, else deviations are noise
    same = bool(torch.equal(eager()["plan"]["waypoints"].float(), ref))
    R["controls"]["eager_fp32_deterministic"] = {
        "expect": "bit-identical repeat", "measured": same, "pass": same}
    print("CONTROL determinism:", same, flush=True)

    base = timeit(eager, warmup=15, iters=40)
    R["arms"].append({"arm": "eager_fp32", **base, "deviation_m": 0.0,
                      "compile_warmup_s": None})
    print("eager fp32 median %.3f ms" % base["median_ms"], flush=True)

    MODES = [("compile_default", {}, None),
             ("compile_reduce_overhead", {"mode": "reduce-overhead"}, None),
             ("compile_default_fp16", {}, torch.float16)]

    for name, kw, dtp in MODES:
        try:
            torch.compiler.reset()
        except Exception:
            pass
        try:
            cm = torch.compile(stack, **kw)

            def run():
                with torch.no_grad():
                    if dtp is None:
                        return cm(frames, acts, v0)
                    with torch.autocast("cuda", dtype=dtp):
                        return cm(frames, acts, v0)

            t0 = time.perf_counter()
            out = run()                      # triggers compilation
            torch.cuda.synchronize()
            warm_s = time.perf_counter() - t0
            wp = out["plan"]["waypoints"].float()
            d = float((wp - ref).abs().max().item())
            t = timeit(run, warmup=15, iters=40)
            R["arms"].append({
                "arm": name, **t, "deviation_m": d,
                "finite": bool(torch.isfinite(wp).all().item()),
                "compile_warmup_s": round(warm_s, 2),
                "speedup_vs_eager": round(base["median_ms"] / t["median_ms"], 3),
                "vs_cuda_graph_7_840ms": round(7.840 / t["median_ms"], 3)})
            print("%-26s median %.3f ms  dev %.4e m  warmup %.1f s  "
                  "%.2fx eager" % (name, t["median_ms"], d, warm_s,
                                   base["median_ms"] / t["median_ms"]),
                  flush=True)
        except Exception as e:
            R["arms"].append({"arm": name, "error": repr(e)[:600]})
            print("%-26s FAILED %s" % (name, repr(e)[:250]), flush=True)

    # verdict
    okarms = [x for x in R["arms"] if "median_ms" in x and x["arm"] != "eager_fp32"]
    best = min(okarms, key=lambda x: x["median_ms"]) if okarms else None
    R["cuda_graph_reference_ms"] = 7.840
    R["fp16_deviation_bar_m"] = 8.663e-02
    if best is None:
        R["verdict"] = ("H-DEPLOY-5 REFUTED — every torch.compile arm failed on "
                        "Thor; manual CUDA graphs remain the lever")
    else:
        beats = best["median_ms"] < 7.840
        dev_ok = best["deviation_m"] <= 8.663e-02
        R["verdict"] = ("H-DEPLOY-5 %s — best %s at %.3f ms (graph 7.840), "
                        "deviation %.4e m (bar 8.663e-02)"
                        % ("SUPPORTED" if (beats and dev_ok) else "REFUTED",
                           best["arm"], best["median_ms"], best["deviation_m"]))
    print(R["verdict"], flush=True)
    R["_evidence_class"] = "MEASURED (ours; Thor, in-process probes only)"
    json.dump(R, open(a.out, "w"), indent=1)
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
