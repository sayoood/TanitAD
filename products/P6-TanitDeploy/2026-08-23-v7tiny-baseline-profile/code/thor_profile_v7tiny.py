"""E-DEPLOY-1 - v7-tiny (champ30k) inference profile on Thor.

Pre-registered in products/P6-TanitDeploy/2026-08-23-v7tiny-baseline-profile/SPEC.md.

RULES ENCODED HERE (each earned):
  * Only torch.cuda.max_memory_allocated() is admissible on Thor. free /
    tegrastats / mem_get_info / VmRSS all lie, in both directions.
  * Every panel carries a control that must read a KNOWN value; a failed
    control VOIDS the run rather than annotating it.
  * n is printed on every timing panel.
  * Reduced precision is never reported as a speedup without its numerical
    deviation alongside.

READ ONLY w.r.t. the box: allocates GPU memory, installs nothing, writes exactly
one JSON to the path given by --out.
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

R = {"spec": "E-DEPLOY-1", "controls": {}, "arms": [], "voided": False,
     "void_reason": None}


def fail(reason):
    R["voided"] = True
    R["void_reason"] = reason
    print("!! VOID:", reason, flush=True)


def timeit(fn, *, warmup, iters):
    """Median/p90 wall time of fn(), CUDA-synchronised on both sides."""
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
    return {"n": len(ts),
            "median_ms": statistics.median(ts) * 1e3,
            "p90_ms": ts[int(0.9 * (len(ts) - 1))] * 1e3,
            "min_ms": ts[0] * 1e3,
            "max_ms": ts[-1] * 1e3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/v7tiny/champ30k"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    a = ap.parse_args()

    R["env"] = {
        "python": sys.version.split()[0], "executable": sys.executable,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "warmup": a.warmup, "iters": a.iters,
    }
    print(json.dumps(R["env"], indent=1), flush=True)
    dev = torch.device("cuda")

    # ---------------- CONTROL 2: the memory probe must MOVE ---------------- #
    # Run first: if max_memory_allocated() cannot see a known 256 MiB
    # allocation, every memory number below is worthless.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.max_memory_allocated()
    blob = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
    blob.fill_(1.0)
    torch.cuda.synchronize()
    rise_mb = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    del blob
    torch.cuda.empty_cache()
    ok = rise_mb >= 256.0
    R["controls"]["memory_probe_moves"] = {
        "expect": ">=256 MiB rise for a known 256 MiB allocation",
        "measured_mb": round(rise_mb, 2), "pass": bool(ok)}
    print("CONTROL memory_probe_moves:", rise_mb, "MiB", ok, flush=True)
    if not ok:
        fail("max_memory_allocated did not move for a known 256MiB alloc")
        return finish(a)

    # ---------------- CONTROL 1: the timer can resolve a known op ---------- #
    x32 = torch.randn(4096, 4096, device=dev)
    y32 = torch.randn(4096, 4096, device=dev)
    t32 = timeit(lambda: x32 @ y32, warmup=5, iters=20)
    x16, y16 = x32.half(), y32.half()
    t16 = timeit(lambda: x16 @ y16, warmup=5, iters=20)
    band = 0.1 < t32["median_ms"] < 1000.0
    faster = t16["median_ms"] < t32["median_ms"]
    R["controls"]["timer_resolves_matmul"] = {
        "expect": "fp32 4096^2 matmul in (0.1ms,1000ms) AND fp16 strictly faster",
        "fp32": t32, "fp16": t16,
        "speedup_fp16": round(t32["median_ms"] / t16["median_ms"], 3),
        "pass": bool(band and faster)}
    print("CONTROL timer_resolves_matmul: fp32", round(t32["median_ms"], 3),
          "ms fp16", round(t16["median_ms"], 3), "ms ->",
          band and faster, flush=True)
    del x32, y32, x16, y16
    torch.cuda.empty_cache()
    if not (band and faster):
        fail("timer could not resolve a definitionally matmul-bound speedup")
        return finish(a)

    # ---------------- build the model from the RUN'S OWN config ------------ #
    cfgp = os.path.join(a.ckpt, "config.json")
    with open(cfgp) as fh:
        cfg = json.load(fh)
    from train_v6_staged import build_stack_from_args, synthetic_train_batch

    args_ns = argparse.Namespace(**cfg["args"])
    stack = build_stack_from_args(args_ns)
    n_par = sum(p.numel() for p in stack.parameters())

    # ---------------- CONTROL 4: parameter count matches the run ----------- #
    with open(os.path.join(a.ckpt, "summary.json")) as fh:
        summ = json.load(fh)
    want = int(summ["param_report"]["total"])
    ok = (n_par == want)
    R["controls"]["param_count_matches_run"] = {
        "expect": want, "measured": n_par, "pass": bool(ok)}
    print("CONTROL param_count:", n_par, "vs", want, ok, flush=True)
    if not ok:
        fail("rebuilt model has %d params, run recorded %d" % (n_par, want))
        return finish(a)

    # ---------------- CONTROL 5: state_dict loads with zero gaps ----------- #
    ck = torch.load(os.path.join(a.ckpt, "ckpt.pt"), map_location="cpu",
                    weights_only=False)
    sd = ck["stack"] if isinstance(ck, dict) and "stack" in ck else ck
    res = stack.load_state_dict(sd, strict=False)
    miss, unexp = list(res.missing_keys), list(res.unexpected_keys)
    ok = not miss and not unexp
    R["controls"]["state_dict_loads_clean"] = {
        "expect": "0 missing, 0 unexpected",
        "missing": miss[:20], "n_missing": len(miss),
        "unexpected": unexp[:20], "n_unexpected": len(unexp),
        "ckpt_step": ck.get("step") if isinstance(ck, dict) else None,
        "pass": bool(ok)}
    print("CONTROL state_dict: missing", len(miss), "unexpected", len(unexp),
          ok, flush=True)
    if not ok:
        fail("state_dict mismatch: %d missing, %d unexpected" % (len(miss),
                                                                 len(unexp)))
        return finish(a)
    del ck, sd

    stack = stack.to(dev).eval()
    for p in stack.parameters():
        p.requires_grad_(False)

    # ---------------- the arm grid ----------------------------------------- #
    DT = [("fp32", None), ("fp16", torch.float16), ("bf16", torch.bfloat16)]
    ref_wp = {}          # batch -> fp32 reference waypoints (cpu float32)

    for B in a.batches:
        batch = synthetic_train_batch(stack, batch=B, k=4, seed=0, device=dev)
        frames, acts, v0 = batch["frames"], batch["actions2"], batch["v0"]
        # the predictor trains on the 3-channel lifted action format; lift the
        # 2-channel synthetic actions the same way the trainer does at the seam.
        A_want = stack.cfg.predictor.action_dim
        if acts.shape[-1] != A_want:
            pad = torch.zeros(acts.shape[0], acts.shape[1],
                              A_want - acts.shape[-1], device=dev,
                              dtype=acts.dtype)
            acts = torch.cat([acts, pad], dim=-1)

        for name, dt in DT:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            def run():
                with torch.no_grad():
                    if dt is None:
                        return stack(frames, acts, v0)
                    with torch.autocast("cuda", dtype=dt):
                        return stack(frames, acts, v0)

            try:
                out = run()
            except Exception as e:
                R["arms"].append({"batch": B, "dtype": name, "error":
                                  repr(e)[:400]})
                print("ARM", B, name, "ERROR", repr(e)[:200], flush=True)
                continue

            wp = out["plan"]["waypoints"].float().cpu()
            finite = bool(torch.isfinite(wp).all().item())
            t = timeit(run, warmup=a.warmup, iters=a.iters)
            peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20

            arm = {"batch": B, "dtype": name, **t,
                   "windows_per_s": B / (t["median_ms"] / 1e3),
                   "peak_alloc_mb": round(peak_mb, 2),
                   "waypoints_shape": list(wp.shape),
                   "waypoints_finite": finite,
                   "waypoints_absmax": float(wp.abs().max().item()),
                   "probe": "torch.cuda.max_memory_allocated"}

            if dt is None:
                # CONTROL 3: determinism - same arm twice, bit-identical.
                wp2 = run()["plan"]["waypoints"].float().cpu()
                same = bool(torch.equal(wp, wp2))
                arm["determinism_bit_identical"] = same
                if B == a.batches[0]:
                    R["controls"]["fp32_deterministic"] = {
                        "expect": "same seed, same arm -> bit-identical plan",
                        "measured": same, "pass": same}
                    if not same:
                        fail("fp32 forward is not deterministic; dtype "
                             "deviations would measure noise")
                        return finish(a)
                ref_wp[B] = wp
            else:
                d = (wp - ref_wp[B]).abs()
                den = ref_wp[B].abs().clamp_min(1e-6)
                arm["dev_vs_fp32"] = {
                    "max_abs_m": float(d.max().item()),
                    "mean_abs_m": float(d.mean().item()),
                    "max_rel": float((d / den).max().item()),
                    "screening_threshold_max_abs_m": 1e-2,
                    "passes_screen": bool(finite and d.max().item() <= 1e-2)}

            R["arms"].append(arm)
            print("ARM b=%d %s median %.3f ms p90 %.3f ms peak %.1f MiB n=%d %s"
                  % (B, name, t["median_ms"], t["p90_ms"], peak_mb, t["n"],
                     ("dev_max %.3e m" % arm["dev_vs_fp32"]["max_abs_m"])
                     if dt is not None else ""), flush=True)

        del batch, frames, acts, v0
        torch.cuda.empty_cache()

    finish(a)


def finish(a):
    R["all_controls_pass"] = all(
        c.get("pass") for c in R["controls"].values()) and not R["voided"]
    R["_evidence_class"] = "MEASURED (ours; Thor, in-process probes only)"
    with open(a.out, "w") as fh:
        json.dump(R, fh, indent=1)
    print("WROTE", a.out, "controls_pass=", R["all_controls_pass"], flush=True)


if __name__ == "__main__":
    main()
