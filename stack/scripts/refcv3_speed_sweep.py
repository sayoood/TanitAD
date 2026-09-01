#!/usr/bin/env python3
"""Cheap A40 throughput sweep for the REF-C v3 B1 launch config.

WHY: the registered knobs (batch 20, `--workers 0`, `--v2-lru 6`) were set on a
different device and a different corpus. At an ESTIMATED 15-22 h per 30 k run,
an hour of measurement here is worth a day of run time — and the launch must be
picked for throughput UNDER a stability and OOM-headroom constraint, not for
peak throughput.

METHOD (assumption-free where it matters):
* **s/step is MARGINAL, never total/steps.** The trainer logs cumulative
  `elapsed_s`; this takes the MEDIAN of consecutive-row diffs from
  `SKIP_ROWS` onward, so process start, cuDNN autotune and cache warm-up are
  excluded by construction rather than by subtracting a guess.
* **Peak memory is `torch.cuda.max_memory_allocated`**, read in-process via an
  atexit hook in the wrapper. `nvidia-smi` sees the caching allocator's
  reservation, not the requirement.
* **Each config is a SUBPROCESS.** An OOM then kills one config, not the sweep,
  and is RECORDED as a result (the OOM edge is data, not a failure).
* **Stability is checked, not assumed:** a config is only admissible if every
  logged loss is finite and its gnorm median stays within `GNORM_FACTOR` of the
  baseline's. A faster config that destabilises training is not faster.

⛔ This sweep changes NO numerics beyond TF32/cuDNN-autotune (both standard on
Ampere and both reported separately). bf16 autocast and channels_last would
change the arithmetic and need a trainer flag plus an equivalence check — out
of scope here, flagged in the report.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

SKIP_ROWS = 4          # rows dropped before timing (autotune + cache warm-up)
N_ROWS = 14            # rows collected per config
GNORM_FACTOR = 3.0     # stability band vs the baseline config

WRAP = r'''
import atexit, json, os, runpy, sys, torch
cfg = json.loads(os.environ["SWEEP_CFG"])
torch.backends.cuda.matmul.allow_tf32 = bool(cfg["tf32"])
torch.backends.cudnn.allow_tf32 = bool(cfg["tf32"])
torch.backends.cudnn.benchmark = bool(cfg["cudnn_bench"])
@atexit.register
def _peak():
    if torch.cuda.is_available():
        print("SWEEPMEM " + json.dumps({
            "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}),
            flush=True)
sys.argv = [cfg["trainer"]] + cfg["argv"]
runpy.run_path(cfg["trainer"], run_name="__main__")
'''


def run_cfg(name: str, argv: list[str], *, trainer: str, out: Path,
            tf32: bool = True, cudnn_bench: bool = True,
            timeout: int = 900) -> dict:
    log = out / f"{name}.jsonl"
    if log.exists():
        log.unlink()
    env = dict(os.environ)
    env["SWEEP_CFG"] = json.dumps({"trainer": trainer, "argv": argv,
                                   "tf32": tf32, "cudnn_bench": cudnn_bench})
    # ⚠️ torch spawns ~113 threads per process; unbounded intra-op threads on a
    # 96-core box thrash rather than help, and the effect looks like a hang.
    env.setdefault("OMP_NUM_THREADS", "8")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", WRAP], env=env,
                       capture_output=True, text=True, timeout=timeout)
    wall = time.time() - t0
    blob = (p.stdout or "") + (p.stderr or "")
    res: dict = {"name": name, "argv": argv, "tf32": tf32,
                 "cudnn_bench": cudnn_bench, "wall_s": round(wall, 1),
                 "rc": p.returncode}
    if re.search(r"out of memory|OutOfMemoryError|CUDA error: out of memory",
                 blob, re.I):
        res["status"] = "OOM"
        return res
    if p.returncode != 0:
        res["status"] = "FAIL"
        res["tail"] = blob.strip().splitlines()[-6:]
        return res

    rows = []
    for line in log.read_text(encoding="utf-8").splitlines() if log.exists() else []:
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "elapsed_s" in r and "step" in r:
            rows.append(r)
    if len(rows) < SKIP_ROWS + 3:
        res["status"] = "TOO_FEW_ROWS"
        res["n_rows"] = len(rows)
        return res
    use = rows[SKIP_ROWS:]
    dt = [(b["elapsed_s"] - a["elapsed_s"]) / max(1, b["step"] - a["step"])
          for a, b in zip(use, use[1:])]
    losses = [r.get("loss") for r in rows if r.get("loss") is not None]
    gnorms = [r.get("grad_norm", r.get("gnorm")) for r in rows
              if r.get("grad_norm", r.get("gnorm")) is not None]
    batch = int(argv[argv.index("--batch") + 1])
    res.update(status="OK", n_rows=len(rows),
               step_s=round(statistics.median(dt), 4),
               step_s_spread=round(max(dt) - min(dt), 4),
               windows_s=round(batch / statistics.median(dt), 2),
               loss_finite=all(x == x and abs(x) != float("inf")
                               for x in losses),
               loss_first=losses[0] if losses else None,
               loss_last=losses[-1] if losses else None,
               gnorm_median=(round(statistics.median(gnorms), 3)
                             if gnorms else None))
    m = re.findall(r"SWEEPMEM (\{.*\})", blob)
    if m:
        res.update(json.loads(m[-1]))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", default="/workspace/TanitAD/stack/scripts/refc_v3_train.py")
    ap.add_argument("--v2-cache", default="/workspace/TanitAD/data/eps")
    ap.add_argument("--out", type=Path, default=Path("/workspace/sweep"))
    ap.add_argument("--arm", default="hier")
    ap.add_argument("--gpu-gb", type=float, default=44.3,
                    help="usable device memory; headroom is judged against it")
    ap.add_argument("--phase", default="all")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    # ⛔ CORPUS PREFLIGHT — 2 s, and it exists because of a real failure.
    # MEASURED 2026-09-02: the first sweep launched while the upload was still
    # in flight (an `until [ $(ls | wc -l) -ge 24 ]` loop saw the last file
    # APPEAR before its bytes landed). Every config then died on a miniz
    # "failed finding central directory" and the run read as 7 broken configs
    # instead of one unreadable episode. Verify by CONTENT, never by presence —
    # and fail here, in seconds, naming the file.
    import torch
    eps = sorted(Path(a.v2_cache).glob("*.v2ep.pt"))
    if not eps:
        print(f"REFUSING: no *.v2ep.pt under {a.v2_cache}", flush=True)
        return 2
    bad = []
    for f in eps:
        try:
            o = torch.load(f, map_location="cpu", weights_only=False)
            assert int(o["poses"].shape[0]) > 0
        except Exception as e:
            bad.append(f"{f.name}: {type(e).__name__}")
    if bad:
        print(f"REFUSING: {len(bad)} of {len(eps)} episodes unreadable — a "
              f"sweep over a broken corpus measures nothing:", flush=True)
        for b in bad[:8]:
            print("   " + b, flush=True)
        return 2
    print(f"corpus preflight OK: {len(eps)} episodes readable", flush=True)

    def base(name, batch, workers, lru, **kw):
        argv = ["--arm", a.arm, "--v2-cache", a.v2_cache, "--image-hw",
                "256", "640", "--steps", str(N_ROWS), "--log-every", "1",
                "--save-every", "100000", "--batch", str(batch),
                "--workers", str(workers), "--v2-lru", str(lru),
                "--out", str(a.out / name)]
        return run_cfg(name, argv, trainer=a.trainer, out=a.out / name, **kw)

    results = []

    def add(r):
        results.append(r)
        keep = {k: r.get(k) for k in ("status", "step_s", "windows_s",
                                      "peak_alloc_gb", "gnorm_median",
                                      "loss_finite")}
        print(f"  {r['name']:<28} {json.dumps(keep)}", flush=True)
        (a.out / "sweep_results.json").write_text(json.dumps(results, indent=1))
        return r

    print("PHASE 1 — the registered config, as the reference", flush=True)
    ref = add(base("ref_b20_w0_lru6", 20, 0, 6))

    print("PHASE 2 — dataloader workers (PNG decode is synchronous at 0)",
          flush=True)
    for w in (4, 8, 16, 32):
        add(base(f"w{w}_b20_lru6", 20, w, 6))
    ok = [r for r in results if r["status"] == "OK"
          and r["name"].startswith("w")]
    if ok:
        wrow = min(ok, key=lambda r: r["step_s"])
        best_w = wrow["argv"][wrow["argv"].index("--workers") + 1]
    else:
        best_w = "0"

    print(f"PHASE 3 — episode LRU at workers={best_w} (503 GB host RAM)",
          flush=True)
    for lru in (24, 64):
        add(base(f"lru{lru}_b20_w{best_w}", 20, int(best_w), lru))
    pool = [r for r in results if r["status"] == "OK"]
    if not pool:
        print("NO CONFIG SUCCEEDED — stopping rather than reporting a winner "
              "among failures; see sweep_results.json 'tail' fields", flush=True)
        return 1
    fastest = min(pool, key=lambda r: r["step_s"])
    best_lru = int(fastest["argv"][fastest["argv"].index("--v2-lru") + 1])

    print(f"PHASE 4 — batch scaling to the OOM edge "
          f"(workers={best_w}, lru={best_lru})", flush=True)
    for b in (32, 48, 64, 96):
        r = add(base(f"b{b}_w{best_w}_lru{best_lru}", b, int(best_w), best_lru))
        if r["status"] == "OOM":
            print(f"  OOM edge found at batch {b}", flush=True)
            break

    print("PHASE 5 — TF32 / cuDNN-autotune contribution at the best config",
          flush=True)
    okb = [r for r in results if r["status"] == "OK"]
    top = max(okb, key=lambda r: r["windows_s"])
    tb = int(top["argv"][top["argv"].index("--batch") + 1])
    add(base(f"notf32_b{tb}", tb, int(best_w), best_lru, tf32=False))
    add(base(f"nobench_b{tb}", tb, int(best_w), best_lru, cudnn_bench=False))

    print("\n=== SWEEP TABLE ===", flush=True)
    print(f"{'config':<28}{'status':>8}{'s/step':>9}{'win/s':>9}"
          f"{'peakGB':>9}{'gnorm':>10}", flush=True)
    for r in results:
        print(f"{r['name']:<28}{r['status']:>8}"
              f"{r.get('step_s', float('nan')):>9.3f}"
              f"{r.get('windows_s', float('nan')):>9.2f}"
              f"{r.get('peak_alloc_gb', float('nan')):>9.2f}"
              f"{str(r.get('gnorm_median')):>10}", flush=True)

    if ref.get("status") == "OK":
        good = [r for r in results if r["status"] == "OK"
                and r.get("loss_finite")
                and (r.get("gnorm_median") or 0) <=
                GNORM_FACTOR * (ref.get("gnorm_median") or 1e9)
                and r.get("peak_alloc_gb", 1e9) < 0.85 * a.gpu_gb]
        if good:
            win = max(good, key=lambda r: r["windows_s"])
            print(f"\nRECOMMENDED (stable, <85% of {a.gpu_gb} GB): {win['name']}"
                  f"  {win['windows_s']:.2f} win/s = "
                  f"{win['windows_s']/max(ref['windows_s'],1e-9):.2f}x the "
                  f"registered config", flush=True)
            print(f"  30k-step wall clock: "
                  f"{win['step_s']*30000/3600:.1f} h  (registered config: "
                  f"{ref['step_s']*30000/3600:.1f} h)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
