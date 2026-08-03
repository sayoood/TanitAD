"""Turn a Thor training-benchmark sweep into per-arm statistics.

Reads what the REAL trainers already emit (one JSON object per step, because the
sweep runs them with ``--log-every 1``) plus the probe's ``*.summary.json`` and a
``tegrastats`` capture, and reports **p50 and p95** rather than a mean — a mean
hides exactly the thermal tail this benchmark exists to find.

⚠️ ``step_s`` in a trainer log is ACCUMULATED over ``--log-every`` (CLAUDE.md
§Traps). This script REQUIRES the sweep's ``--log-every 1``, where the
accumulator is reset every step and therefore already IS the per-step time; it
asserts that assumption against the wall-clock span instead of trusting it.

Usage: thor_bench_report.py <sweep_dir> [--tegrastats FILE] [--warmup N]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return ys[i]


def parse_steps(log_path: str, warmup: int) -> dict:
    """Per-step times from a trainer log run at --log-every 1."""
    steps, data_s, step_s = [], [], []
    for line in open(log_path, errors="replace"):
        line = line.strip()
        if not (line.startswith("{") and '"step"' in line):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step_s" not in d:
            continue
        steps.append(d["step"])
        step_s.append(float(d["step_s"]))
        data_s.append(float(d.get("data_s", 0.0)))
    if len(step_s) > warmup:
        step_s, data_s, steps = step_s[warmup:], data_s[warmup:], steps[warmup:]
    return {"n": len(step_s), "step_s": step_s, "data_s": data_s, "steps": steps}


def tegrastats_summary(path: str) -> dict:
    """Thermals / power / clocks over a capture — the throttle evidence."""
    temps, gpu_mw, ram_mb, cpu_clk = [], [], [], []
    if not path or not os.path.exists(path):
        return {}
    for line in open(path, errors="replace"):
        m = re.search(r"tj@([\d.]+)C", line)
        if m:
            temps.append(float(m.group(1)))
        m = re.search(r"VDD_GPU (\d+)mW", line)
        if m:
            gpu_mw.append(int(m.group(1)))
        m = re.search(r"RAM (\d+)/(\d+)MB", line)
        if m:
            ram_mb.append(int(m.group(1)))
        clks = [int(c) for c in re.findall(r"\d+%@(\d+)", line)]
        if clks:
            cpu_clk.append(max(clks))
    out: dict = {"n_samples": len(temps)}
    if temps:
        out |= {"tj_start_C": temps[0], "tj_max_C": max(temps),
                "tj_p50_C": _pct(temps, .5), "tj_end_C": temps[-1],
                "tj_rise_C": round(max(temps) - temps[0], 2)}
    if gpu_mw:
        nz = [g for g in gpu_mw if g > 0]
        out |= {"gpu_mw_max": max(gpu_mw),
                "gpu_mw_p50_active": _pct(nz, .5) if nz else 0,
                "gpu_active_frac": round(len(nz) / len(gpu_mw), 3)}
    if ram_mb:
        out |= {"ram_mb_start": ram_mb[0], "ram_mb_max": max(ram_mb),
                "ram_mb_delta": max(ram_mb) - ram_mb[0]}
    if cpu_clk:
        out |= {"cpu_clk_max_mhz": max(cpu_clk), "cpu_clk_min_mhz": min(cpu_clk)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--tegrastats", default="")
    ap.add_argument("--warmup", type=int, default=5,
                    help="steps discarded (allocator warmup / cudnn autotune)")
    ap.add_argument("--window", type=int, default=8,
                    help="frames per training window (REF-C cfg.window=8)")
    a = ap.parse_args()

    rows = []
    for log in sorted(glob.glob(os.path.join(a.sweep_dir, "refc_b*.log")),
                      key=lambda p: int(re.search(r"_b(\d+)\.log", p).group(1))):
        b = int(re.search(r"_b(\d+)\.log", log).group(1))
        s = parse_steps(log, a.warmup)
        blob = os.path.join(a.sweep_dir, f"refc_b{b}.jsonl.summary.json")
        mem = json.load(open(blob)) if os.path.exists(blob) else {}
        oom = bool(re.search(r"out of memory|CUDA error",
                             open(log, errors="replace").read(), re.I))
        row = {"batch": b, "n_steps_measured": s["n"], "oom": oom}
        if s["n"]:
            p50, p95 = _pct(s["step_s"], .5), _pct(s["step_s"], .95)
            row |= {
                "step_s_p50": round(p50, 4), "step_s_p95": round(p95, 4),
                "step_s_mean": round(st.mean(s["step_s"]), 4),
                "step_s_max": round(max(s["step_s"]), 4),
                "data_s_p50": round(_pct(s["data_s"], .5), 4),
                "windows_per_s_p50": round(b / p50, 2) if p50 else None,
                "images_per_s_p50": round(b * a.window / p50, 1) if p50 else None,
                "p95_over_p50": round(p95 / p50, 3) if p50 else None,
            }
        row |= {k: mem.get(k) for k in ("peak_cuda_alloc_gb",
                                        "peak_cuda_reserved_gb",
                                        "peak_host_vmhwm_gb")}
        rows.append(row)

    out = {"sweep_dir": a.sweep_dir, "warmup_discarded": a.warmup,
           "arms": rows, "tegrastats": tegrastats_summary(a.tegrastats)}
    print(json.dumps(out, indent=2))

    hdr = (f"\n{'batch':>6} {'p50 s':>8} {'p95 s':>8} {'p95/p50':>8} "
           f"{'win/s':>8} {'img/s':>9} {'data s':>8} {'GPU GB':>8} {'n':>4}")
    print(hdr); print("-" * len(hdr.strip()))
    for r in rows:
        if not r.get("step_s_p50"):
            print(f"{r['batch']:>6} {'OOM/FAIL' if r['oom'] else 'no data':>8}")
            continue
        print(f"{r['batch']:>6} {r['step_s_p50']:>8.3f} {r['step_s_p95']:>8.3f} "
              f"{r['p95_over_p50']:>8.2f} {r['windows_per_s_p50']:>8.2f} "
              f"{r['images_per_s_p50']:>9.1f} {r['data_s_p50']:>8.3f} "
              f"{(r['peak_cuda_alloc_gb'] or 0):>8.2f} {r['n_steps_measured']:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
