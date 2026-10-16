#!/usr/bin/env python3
"""MEASURE the real thing on the dev box: `launch_gate.py suite-prerun` on a REAL profile (default refe_p2rep = P2-s1's), with the process tree's peak RSS / threads / CPU sampled, then
`launch_gate.py run --only G-SUITE` ONCE with the pre-run (--suite-evidence) and ONCE without it.  Writes <out>/measure.json.  Dev box, CPU only.  The numbers are this box's (shared, Windows);
the POD's G-SUITE was 22 min (P2-s1, 2026-10-15), so use the RATIOS (live vs accepted) and the memory figures, not the seconds.

    python measure_real_profile.py --pkg <package root> --out <dir> [--profile refe_p2rep]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

PY = sys.executable


def sample(proc: subprocess.Popen, out: dict, stop: threading.Event) -> None:
    peak_rss = peak_thr = peak_n = 0
    cpu = 0.0
    seen: dict = {}
    p = psutil.Process(proc.pid)
    while not stop.is_set():
        try:
            tree = [p] + p.children(recursive=True)
        except psutil.Error:
            break
        rss = thr = 0
        for q in tree:
            try:
                rss += q.memory_info().rss
                thr += q.num_threads()
                t = q.cpu_times()
                seen[q.pid] = t.user + t.system
            except psutil.Error:
                continue
        peak_rss, peak_thr, peak_n = max(peak_rss, rss), max(peak_thr, thr), max(peak_n, len(tree))
        time.sleep(0.25)
    out.update({"peak_rss_gib": round(peak_rss / 2 ** 30, 2), "peak_threads": peak_thr, "peak_processes": peak_n, "cpu_seconds": round(sum(seen.values()), 1)})


def timed(argv: list, env: dict, log: Path, monitor: bool = False) -> dict:
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, env=env)
        res: dict = {}
        stop = threading.Event()
        th = threading.Thread(target=sample, args=(proc, res, stop), daemon=True) if monitor else None
        if th:
            th.start()
        rc = proc.wait()
        stop.set()
        if th:
            th.join(timeout=5)
    return {"rc": rc, "seconds": round(time.perf_counter() - t0, 1), **res}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default="refe_p2rep")
    a = ap.parse_args()
    pkg, out = Path(a.pkg), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gate = str(pkg / "refe" / "launch_gate.py")
    sys.path.insert(0, str(pkg / "refe"))
    import launch_gate as L                                              # the profile's REAL argv for the --only G-SUITE runs
    prof = L.PROFILES[a.profile]
    args = " ".join(L.P2REP_ARGV) if a.profile == "refe_p2rep" else " ".join(L.P2_ARGV)
    key = out / "key" / "gate.key"
    env = {k: v for k, v in os.environ.items() if k != "TANITAD_GATE_ARM"}
    env.update({"PYTHONUTF8": "1", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "TANITAD_GATE_KEY_FILE": str(key)})
    ev = out / "suite_prerun.json"
    res: dict = {"profile": a.profile, "suite": [(e.get("script", "raw"), e.get("args", [])) for e in prof["suite"]], "host": {"cpus": os.cpu_count(), "ram_gib": round(psutil.virtual_memory().total / 2 ** 30, 1)}}
    res["prerun"] = timed([PY, gate, "suite-prerun", "--profile", a.profile, "--pkg", str(pkg), "--out", str(ev)], env, out / "prerun.log", monitor=True)
    res["prerun"]["evidence_bytes"] = ev.stat().st_size if ev.is_file() else None
    res["prerun"]["verdict"] = json.loads(ev.read_text(encoding="utf-8"))["verdict"] if ev.is_file() else None

    def run_only_suite(tag: str, with_evidence: bool) -> dict:
        argv = [PY, gate, "run", "--profile", a.profile, "--pkg", str(pkg), "--args", args, "--out-dir", str(out / f"run_{tag}"), "--only", "G-SUITE"] + (["--suite-evidence", str(ev)] if with_evidence else [])
        r = timed(argv, env, out / f"run_{tag}.log")
        ev_g = out / f"run_{tag}" / "evidence_G-SUITE.json"
        d = json.loads(ev_g.read_text(encoding="utf-8")) if ev_g.is_file() else {}
        r["g_suite_status"] = d.get("status")
        r["suite_source"] = (d.get("details") or {}).get("suite_source")
        return r
    res["run_with_evidence"] = run_only_suite("with_evidence", True)
    res["run_without_flag"] = run_only_suite("without_flag", False)
    (out / "measure.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
