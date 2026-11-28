"""E-DEPLOY-1b - mechanism probes behind the E-DEPLOY-1 baseline.

Two questions the baseline raised but could not answer:

  A. WHERE does the fp16/bf16 plan deviation come from? If it is born in the
     trunk, reduced precision is unsafe everywhere. If it is BORN SMALL and
     GROWN by the 60-step kinematic integrator, the fix is a per-module dtype
     policy (pin the emission/integration head) and the rest of the net can be
     halved.  Discriminator: deviation per integration step. Integration-grown
     error starts at ~0 at step 0 and grows monotonically; a broken layer does
     not.

  B. Is the ~7.5 ms batch-independent part of the tick KERNEL-LAUNCH bound?
     The affine fit t(B) = a + b*B over the fp32 arms gives a ~ 7.5 ms of
     fixed cost. If CUDA-graph replay removes most of it, the b1 deploy lever
     is graph capture, not precision.  A capture FAILURE is also an answer and
     is reported as one.

Writes one JSON. Installs nothing.
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

R = {"spec": "E-DEPLOY-1b", "controls": {}}


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


def dev(x, ref):
    d = (x - ref).abs()
    return {"max_abs": float(d.max().item()),
            "mean_abs": float(d.mean().item()),
            "max_rel": float((d / ref.abs().clamp_min(1e-6)).max().item()),
            "ref_absmax": float(ref.abs().max().item())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/v7tiny/champ30k"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev_t = torch.device("cuda")

    with open(os.path.join(a.ckpt, "config.json")) as fh:
        cfg = json.load(fh)
    from train_v6_staged import build_stack_from_args, synthetic_train_batch
    stack = build_stack_from_args(argparse.Namespace(**cfg["args"]))
    ck = torch.load(os.path.join(a.ckpt, "ckpt.pt"), map_location="cpu",
                    weights_only=False)
    res = stack.load_state_dict(ck["stack"], strict=False)
    assert not res.missing_keys and not res.unexpected_keys, "state_dict gap"
    del ck
    stack = stack.to(dev_t).eval()
    for p in stack.parameters():
        p.requires_grad_(False)

    B = 1
    batch = synthetic_train_batch(stack, batch=B, k=4, seed=0, device=dev_t)
    frames, acts, v0 = batch["frames"], batch["actions2"], batch["v0"]
    A_want = stack.cfg.predictor.action_dim
    if acts.shape[-1] != A_want:
        acts = torch.cat([acts, torch.zeros(acts.shape[0], acts.shape[1],
                                            A_want - acts.shape[-1],
                                            device=dev_t, dtype=acts.dtype)],
                         dim=-1)
    R["shapes"] = {"frames": list(frames.shape), "actions": list(acts.shape),
                   "action_dim_cfg": A_want,
                   "plan_steps": int(stack.cfg.plan_steps),
                   "dt": float(stack.cfg.dt),
                   "n_candidates": int(stack.cfg.n_candidates)}

    def fwd(dt=None):
        with torch.no_grad():
            if dt is None:
                return stack(frames, acts, v0)
            with torch.autocast("cuda", dtype=dt):
                return stack(frames, acts, v0)

    # ------------------------- A. WHERE IS THE ERROR BORN ------------------ #
    ref = fwd(None)
    stages_ref = {
        "z_op (trunk latent)": ref["z_op"].float(),
        "z_tac": ref["z_tac"].float(),
        "z_str": ref["z_str"].float(),
        "plan.feat (planner feature)": ref["plan"]["feat"].float(),
        "plan.a (accel command)": ref["plan"]["a"].float(),
        "plan.kappa (curvature command)": ref["plan"]["kappa"].float(),
        "plan.waypoints (integrated)": ref["plan"]["waypoints"].float(),
    }

    # CONTROL: fp32 vs fp32 must be exactly 0 at every stage. A stage that is
    # nonzero here is nondeterministic and its dtype deviation is meaningless.
    ref2 = fwd(None)
    ctrl = {}
    for k, v in stages_ref.items():
        key = k.split(" ")[0]
        cur = ref2
        for part in key.split("."):
            cur = cur[part]
        ctrl[k] = float((cur.float() - v).abs().max().item())
    R["controls"]["fp32_repeat_is_exactly_zero"] = {
        "expect": "0.0 at every stage", "measured": ctrl,
        "pass": all(x == 0.0 for x in ctrl.values())}
    print("CONTROL fp32 repeat:", R["controls"]
          ["fp32_repeat_is_exactly_zero"]["pass"], flush=True)

    R["stage_deviation"] = {}
    for name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        o = fwd(dtype)
        per = {}
        for k, refv in stages_ref.items():
            key = k.split(" ")[0]
            cur = o
            for part in key.split("."):
                cur = cur[part]
            per[k] = dev(cur.float(), refv)
        R["stage_deviation"][name] = per
        print("--", name, flush=True)
        for k, v in per.items():
            print("   %-32s max_abs %.4e  max_rel %.4e  (ref absmax %.3g)"
                  % (k, v["max_abs"], v["max_rel"], v["ref_absmax"]),
                  flush=True)

        # deviation AS A FUNCTION OF INTEGRATION STEP -> the discriminator
        wp_ref = stages_ref["plan.waypoints (integrated)"]     # [B,C,T,2]
        wp = o["plan"]["waypoints"].float()
        per_step = (wp - wp_ref).abs().amax(dim=(0, 1, 3))     # [T]
        R["stage_deviation"][name]["_per_step_max_abs"] = [
            round(float(x), 6) for x in per_step.tolist()]
        s = per_step.tolist()
        R["stage_deviation"][name]["_per_step_summary"] = {
            "step0": s[0], "step9": s[9], "step29": s[29], "step59": s[-1],
            "monotone_nondecreasing": all(
                s[i] <= s[i + 1] + 1e-9 for i in range(len(s) - 1)),
            "growth_last_over_first": (s[-1] / s[0]) if s[0] > 0 else None}
        print("   per-step dev: t0 %.3e  t9 %.3e  t29 %.3e  t59 %.3e  mono=%s"
              % (s[0], s[9], s[29], s[-1],
                 R["stage_deviation"][name]["_per_step_summary"]
                 ["monotone_nondecreasing"]), flush=True)

    # ------------------------- B. IS b1 LAUNCH-BOUND ----------------------- #
    eager = timeit(lambda: fwd(None), warmup=20, iters=50)
    R["cuda_graph"] = {"eager_b1_fp32": eager}
    print("eager b1 fp32 median %.3f ms (n=%d)"
          % (eager["median_ms"], eager["n"]), flush=True)

    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                fwd(None)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(g):
                static_out = stack(frames, acts, v0)
        torch.cuda.synchronize()
        graphed = timeit(g.replay, warmup=20, iters=50)
        wp_g = static_out["plan"]["waypoints"].float()
        agree = dev(wp_g, stages_ref["plan.waypoints (integrated)"])
        R["cuda_graph"].update({
            "captured": True, "replay_b1_fp32": graphed,
            "speedup": round(eager["median_ms"] / graphed["median_ms"], 3),
            "saved_ms": round(eager["median_ms"] - graphed["median_ms"], 3),
            "replay_matches_eager": agree,
            "replay_bit_identical": bool(agree["max_abs"] == 0.0)})
        print("GRAPH replay median %.3f ms -> speedup %.2fx, saved %.2f ms, "
              "bit-identical=%s"
              % (graphed["median_ms"], R["cuda_graph"]["speedup"],
                 R["cuda_graph"]["saved_ms"],
                 R["cuda_graph"]["replay_bit_identical"]), flush=True)
    except Exception as e:
        R["cuda_graph"].update({"captured": False, "error": repr(e)[:600]})
        print("GRAPH CAPTURE FAILED:", repr(e)[:400], flush=True)

    R["_evidence_class"] = "MEASURED (ours; Thor, in-process probes only)"
    with open(a.out, "w") as fh:
        json.dump(R, fh, indent=1)
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
