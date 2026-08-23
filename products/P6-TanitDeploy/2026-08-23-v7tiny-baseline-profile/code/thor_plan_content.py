"""Content check on the plan dict + CUDA-graph across the batch grid.

WHY: E-DEPLOY-1b read `plan.a` and `plan.kappa` as EXACTLY zero. Before any
"the integrator amplifies the error" claim is written down, the actual contents
of the plan dict must be looked at. A mechanism story told over the wrong tensor
is a confident wrong answer.

Also completes the deploy playbook: CUDA-graph replay at every profiled batch,
and graph+fp16 combined, since graph replay was bit-identical at b1.
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

R = {"spec": "E-DEPLOY-1c"}


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


def describe(t):
    if not torch.is_tensor(t):
        return {"type": type(t).__name__}
    tf = t.float()
    return {"shape": list(t.shape), "dtype": str(t.dtype),
            "absmax": float(tf.abs().max().item()),
            "mean": float(tf.mean().item()),
            "std": float(tf.std().item()) if tf.numel() > 1 else 0.0,
            "n_zero": int((tf == 0).sum().item()), "numel": int(tf.numel()),
            "all_zero": bool((tf == 0).all().item()),
            "finite": bool(torch.isfinite(tf).all().item())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/v7tiny/champ30k"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dv = torch.device("cuda")

    with open(os.path.join(a.ckpt, "config.json")) as fh:
        cfg = json.load(fh)
    from train_v6_staged import build_stack_from_args, synthetic_train_batch
    stack = build_stack_from_args(argparse.Namespace(**cfg["args"]))
    ck = torch.load(os.path.join(a.ckpt, "ckpt.pt"), map_location="cpu",
                    weights_only=False)
    res = stack.load_state_dict(ck["stack"], strict=False)
    assert not res.missing_keys and not res.unexpected_keys
    del ck
    stack = stack.to(dv).eval()
    for p in stack.parameters():
        p.requires_grad_(False)

    def mkbatch(B):
        b = synthetic_train_batch(stack, batch=B, k=4, seed=0, device=dv)
        f, ac, v = b["frames"], b["actions2"], b["v0"]
        A = stack.cfg.predictor.action_dim
        if ac.shape[-1] != A:
            ac = torch.cat([ac, torch.zeros(ac.shape[0], ac.shape[1],
                                            A - ac.shape[-1], device=dv,
                                            dtype=ac.dtype)], dim=-1)
        return f, ac, v

    frames, acts, v0 = mkbatch(1)
    with torch.no_grad():
        out = stack(frames, acts, v0)

    R["plan_keys"] = sorted(out["plan"].keys()) if isinstance(
        out["plan"], dict) else str(type(out["plan"]))
    R["plan_contents"] = {k: describe(v) for k, v in out["plan"].items()} \
        if isinstance(out["plan"], dict) else {}
    R["top_keys"] = sorted(out.keys())
    print("plan keys:", R["plan_keys"], flush=True)
    for k, d in R["plan_contents"].items():
        if "shape" in d:
            print("  %-14s %-22s absmax %.4g  std %.4g  zeros %d/%d %s"
                  % (k, str(d["shape"]), d["absmax"], d["std"], d["n_zero"],
                     d["numel"], "ALL-ZERO" if d["all_zero"] else ""),
                  flush=True)
        else:
            print("  %-14s %s" % (k, d), flush=True)

    # which emission path is live?
    R["cfg_planner"] = {
        "proposals": stack.cfg.proposals, "selector": stack.cfg.selector,
        "n_candidates": int(stack.cfg.n_candidates),
        "plan_steps": int(stack.cfg.plan_steps), "dt": float(stack.cfg.dt),
        "a_max": float(stack.cfg.a_max), "kappa_max": float(stack.cfg.kappa_max),
        "emission_squash": stack.cfg.emission_squash,
        "mpc_refine": bool(getattr(stack.cfg, "mpc_refine", False)),
        "diffusion_steps": int(getattr(stack.cfg, "diffusion_steps", 0))}
    print("planner cfg:", json.dumps(R["cfg_planner"]), flush=True)

    # ---- CUDA graph across the batch grid, eager vs replay, fp32 + fp16 ---- #
    R["graph_grid"] = []
    for B in (1, 4, 8):
        frames, acts, v0 = mkbatch(B)
        for nm, dt in (("fp32", None), ("fp16", torch.float16)):

            def fwd():
                with torch.no_grad():
                    if dt is None:
                        return stack(frames, acts, v0)
                    with torch.autocast("cuda", dtype=dt):
                        return stack(frames, acts, v0)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            eager = timeit(fwd, warmup=15, iters=40)
            ref_wp = fwd()["plan"]["waypoints"].float().clone()
            row = {"batch": B, "dtype": nm, "eager": eager,
                   "eager_peak_mb": round(
                       torch.cuda.max_memory_allocated() / 2 ** 20, 2)}
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(5):
                        fwd()
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                g = torch.cuda.CUDAGraph()
                with torch.no_grad():
                    if dt is None:
                        with torch.cuda.graph(g):
                            so = stack(frames, acts, v0)
                    else:
                        with torch.autocast("cuda", dtype=dt):
                            with torch.cuda.graph(g):
                                so = stack(frames, acts, v0)
                torch.cuda.synchronize()
                rep = timeit(g.replay, warmup=15, iters=40)
                d = (so["plan"]["waypoints"].float() - ref_wp).abs().max()
                row.update({
                    "captured": True, "replay": rep,
                    "replay_peak_mb": round(
                        torch.cuda.max_memory_allocated() / 2 ** 20, 2),
                    "speedup": round(eager["median_ms"] / rep["median_ms"], 3),
                    "saved_ms": round(eager["median_ms"] - rep["median_ms"], 3),
                    "replay_vs_eager_max_abs": float(d.item()),
                    "bit_identical": bool(d.item() == 0.0),
                    "windows_per_s": B / (rep["median_ms"] / 1e3)})
                print("b=%d %s eager %.3f -> graph %.3f ms  %.2fx  saved %.2f "
                      "ms  bitident=%s  win/s %.1f"
                      % (B, nm, eager["median_ms"], rep["median_ms"],
                         row["speedup"], row["saved_ms"], row["bit_identical"],
                         row["windows_per_s"]), flush=True)
                del g, so
            except Exception as e:
                row.update({"captured": False, "error": repr(e)[:400]})
                print("b=%d %s GRAPH FAILED %s" % (B, nm, repr(e)[:200]),
                      flush=True)
            R["graph_grid"].append(row)
            torch.cuda.empty_cache()

    R["_evidence_class"] = "MEASURED (ours; Thor, in-process probes only)"
    with open(a.out, "w") as fh:
        json.dump(R, fh, indent=1)
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
