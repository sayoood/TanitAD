#!/usr/bin/env python3
"""l3_gpu_smoke.py ARM -- the GPU-memory smoke of the card's launch gate (control section 3.6): the trainer's own `--selfcheck-only` (the real-data G-LIVE / reach probe path of the pod's s3_dry_mint.sh, rh_selfcheck.py)
run on the LAUNCH argv with `--device cuda` at batch 64, on the RTX 4060, while holding the shared dev-box GPU lock (l3_gpulock.py). Records peak GPU memory (torch max allocated / reserved), wall seconds and the
selfcheck verdict to <out>/GPU_SMOKE.json. If batch 64 does not fit (CUDA OOM) it records that and stops: the batch is NOT changed.
Takes the lock with one try (or waits up to --wait seconds, polling every 30 s) and releases it in a finally block. Never forces another holder's lock.
usage: l3_gpu_smoke.py ARM [--root C:/Users/Admin/lever3_dry] [--wait 0] [--max-clips N]
Run it via a detached process (CPU env must NOT hide the GPU: CUDA_VISIBLE_DEVICES unset)."""
import argparse, json, os, runpy, sys, time
KIT = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, KIT)
import l3_gpulock as L

ap = argparse.ArgumentParser()
ap.add_argument("arm")
ap.add_argument("--root", default="C:/Users/Admin/lever3_dry")
ap.add_argument("--wait", type=int, default=0)
ap.add_argument("--max-clips", type=int, default=0)
a = ap.parse_args()
D = f"{a.root}/runs/{a.arm}"
out = f"{D}/gpu_smoke"
os.makedirs(out, exist_ok=True)
job = f"l3-gpu-smoke-{a.arm}"
res = {"arm": a.arm, "argv": f"{D}/argv.json"}
if not L.take(job, a.wait):
    res.update(status="BLOCKED", holder=L.holder())
    json.dump(res, open(f"{out}/GPU_SMOKE.json", "w"), indent=1)
    print("ZZ GPU SMOKE BLOCKED ZZ", L.holder())
    sys.exit(3)
try:
    import torch
    res["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    if not torch.cuda.is_available():
        res.update(status="NO_CUDA")
    else:
        torch.cuda.reset_peak_memory_stats()
        res["free_before_mib"] = torch.cuda.mem_get_info()[0] // 2**20
        t0 = time.time()
        os.environ["RH_TREE"] = f"{a.root}/tree_cdb"
        sys.argv = ["rh_selfcheck.py", f"{D}/argv.json", f"{out}/sc", f"{out}/result.json"] + (["--max-clips", str(a.max_clips)] if a.max_clips else [])
        try:
            runpy.run_path(KIT + "/rh_selfcheck.py", run_name="__main__")
        except SystemExit:
            pass
        res["wall_s"] = round(time.time() - t0, 1)
        res["peak_alloc_mib"] = torch.cuda.max_memory_allocated() // 2**20
        res["peak_reserved_mib"] = torch.cuda.max_memory_reserved() // 2**20
        r = json.load(open(f"{out}/result.json")) if os.path.isfile(f"{out}/result.json") else {}
        res["selfcheck_result"] = {k: r.get(k) for k in ("rc", "refused", "message", "declared_vs_built", "dagger_probe_pass", "acc_probe_pass")}
        msg = str(r.get("message", "")) + str(r.get("traceback_tail", ""))
        res["status"] = "OOM_AT_BATCH_64" if "out of memory" in msg.lower() else ("PASS" if r.get("rc") == 0 else "FAIL")
finally:
    res["lock_released"] = L.release(job)
    json.dump(res, open(f"{out}/GPU_SMOKE.json", "w"), indent=1)
    print("ZZ GPU SMOKE", res.get("status"), json.dumps({k: res.get(k) for k in ("peak_alloc_mib", "peak_reserved_mib", "wall_s", "lock_released")}))
