"""E-DEPLOY-4 - does TF32 help UNDER a CUDA graph? A within-process paired A/B.

WHY THIS RUN EXISTS. Two numbers measured in DIFFERENT processes today suggested
TF32 helps once graphs remove the launch overhead: fp32+graph 7.840 ms
(E-DEPLOY-1c) vs tf32+graph 6.112 ms (E-DEPLOY-3). That is a 22 % gap - but the
same nominal eager-fp32 arm read 13.020 / 12.529 / 12.369 ms across today's
three processes (+-4 %), so a cross-run comparison is not admissible evidence.
Quoting 6.112 against 7.840 without this run would be a textbook false positive:
two arms, two processes, one conclusion.

DESIGN. One process. Two graphs captured from the SAME loaded model, differing
only in the TF32 matmul flag AT CAPTURE TIME (a graph freezes the kernels chosen
when it was captured, so the flag must be set before capture, not before replay).
Replays are INTERLEAVED in ABAB order and repeated in two rounds, so any thermal
or clock drift hits both arms equally instead of accumulating on whichever ran
second.

PRE-REGISTERED:
  H-DEPLOY-7: TF32 gives a real speedup under CUDA-graph replay.
    SUPPORTED : tf32-graph median < fp32-graph median by >= 5 % in BOTH
                interleaved rounds, AND output stays bit-identical to strict
                fp32. Then "enable TF32 before capture" joins the playbook as a
                free lever.
    REFUTED   : gain < 5 %, or it does not reproduce in both rounds, or the
                output moves. Then the 6.112-vs-7.840 gap was cross-run noise
                and must be retracted before it reaches a report.

CONTROLS
  * TF32 switch live: a 2048^2 fp32 matmul must change when the flag flips.
  * Order control: ABAB interleaving, two rounds; a gain that appears in one
    round only is drift, not a lever.
  * Bit-identity: both graphs' outputs compared to the strict-fp32 eager
    reference.
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

R = {"spec": "E-DEPLOY-4", "controls": {}, "rounds": []}


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


def set_tf32(on: bool):
    torch.backends.cuda.matmul.allow_tf32 = bool(on)
    torch.backends.cudnn.allow_tf32 = bool(on)
    torch.set_float32_matmul_precision("high" if on else "highest")


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
    ok = not res.missing_keys and not res.unexpected_keys
    R["controls"]["state_dict_loads_clean"] = {
        "n_missing": len(res.missing_keys),
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

    # strict fp32 reference
    set_tf32(False)
    ref = fwd()["plan"]["waypoints"].float().clone()
    eager = timeit(fwd, warmup=15, iters=40)
    R["eager_fp32_highest"] = eager
    print("eager fp32(highest) median %.3f ms" % eager["median_ms"], flush=True)

    # CONTROL: the flag is live
    x = torch.randn(2048, 2048, device=dv)
    y = torch.randn(2048, 2048, device=dv)
    set_tf32(False)
    m_hi = (x @ y).clone()
    set_tf32(True)
    m_tf = (x @ y).clone()
    delta = float((m_tf - m_hi).abs().max().item())
    R["controls"]["tf32_switch_is_live"] = {
        "max_abs_change": delta, "pass": bool(delta > 0.0)}
    print("CONTROL tf32 live: %.4e -> %s" % (delta, delta > 0.0), flush=True)
    del x, y, m_hi, m_tf
    torch.cuda.empty_cache()

    def capture(tf32_on):
        set_tf32(tf32_on)
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
        return g, so

    g_off, out_off = capture(False)
    g_on, out_on = capture(True)

    # ⛔ A GRAPH'S STATIC OUTPUT TENSOR IS ONLY VALID UNTIL ITS MEMORY POOL IS
    # REUSED. Reading `out_off` after capturing the SECOND graph reads clobbered
    # memory: the first version of this probe did exactly that and reported a
    # 5.613e+01 m "deviation" — precisely `waypoints_absmax`, i.e. the tensor had
    # been zeroed, not perturbed. ⇒ REPLAY each graph and read its output
    # IMMEDIATELY, so the values are the ones that graph just produced.
    g_off.replay()
    torch.cuda.synchronize()
    d_off = float((out_off["plan"]["waypoints"].float() - ref).abs().max())
    g_on.replay()
    torch.cuda.synchronize()
    d_on = float((out_on["plan"]["waypoints"].float() - ref).abs().max())

    # CONTROL: a deviation equal to the reference's own magnitude means the
    # tensor was clobbered, not that the arm is wrong. Refuse to call that a
    # numerical result.
    ref_absmax = float(ref.abs().max())
    clobbered = [nm for nm, d in (("fp32_graph", d_off), ("tf32_graph", d_on))
                 if abs(d - ref_absmax) < 1e-6]
    R["controls"]["graph_outputs_not_clobbered"] = {
        "expect": "deviation != the reference's own absmax (%.4f)" % ref_absmax,
        "fp32_graph_dev": d_off, "tf32_graph_dev": d_on,
        "ref_absmax": ref_absmax, "clobbered_arms": clobbered,
        "pass": not clobbered}

    R["bit_identity"] = {"graph_fp32_vs_eager_fp32_max_abs": d_off,
                         "graph_tf32_vs_eager_fp32_max_abs": d_on}
    print("bit-identity (post-replay): fp32graph %.3e  tf32graph %.3e  "
          "clobber-control pass=%s" % (d_off, d_on, not clobbered), flush=True)

    # ---- ABAB interleaved, two rounds ------------------------------------- #
    for rnd in (1, 2):
        off = timeit(g_off.replay, warmup=15, iters=40)
        on = timeit(g_on.replay, warmup=15, iters=40)
        off2 = timeit(g_off.replay, warmup=5, iters=40)
        on2 = timeit(g_on.replay, warmup=5, iters=40)
        m_off = statistics.median([off["median_ms"], off2["median_ms"]])
        m_on = statistics.median([on["median_ms"], on2["median_ms"]])
        gain = 1.0 - m_on / m_off
        R["rounds"].append({
            "round": rnd, "fp32_graph_ms": [off["median_ms"], off2["median_ms"]],
            "tf32_graph_ms": [on["median_ms"], on2["median_ms"]],
            "fp32_graph_median_ms": m_off, "tf32_graph_median_ms": m_on,
            "gain_frac": gain, "gain_pct": round(gain * 100, 2)})
        print("round %d: fp32graph %.3f ms  tf32graph %.3f ms  gain %.1f %%"
              % (rnd, m_off, m_on, gain * 100), flush=True)

    gains = [r["gain_frac"] for r in R["rounds"]]
    bit_ok = (R["bit_identity"]["graph_tf32_vs_eager_fp32_max_abs"] == 0.0)
    if not R["controls"]["graph_outputs_not_clobbered"]["pass"]:
        R["voided"] = ("graph output tensors were clobbered; no numerical "
                       "verdict is admissible from this run")
        bit_ok = None
    supported = all(g >= 0.05 for g in gains) and bit_ok is True
    R["verdict"] = ("H-DEPLOY-7 %s — gains %s (bar 5 %% in BOTH rounds), "
                    "tf32-graph bit-identical to strict fp32 = %s"
                    % ("SUPPORTED" if supported else "REFUTED",
                       [round(g * 100, 2) for g in gains], bit_ok))
    print(R["verdict"], flush=True)
    R["_evidence_class"] = "MEASURED (ours; Thor, within-process paired A/B)"
    json.dump(R, open(a.out, "w"), indent=1)
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
