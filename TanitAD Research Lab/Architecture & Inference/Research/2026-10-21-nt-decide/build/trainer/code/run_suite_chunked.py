#!/usr/bin/env python3
"""Run a stack's pytest suite in CHUNKS of test files, one fresh process per chunk, under a private-memory ceiling
(the MM's 4 GB per-process rule on the dev box while C2 runs). A chunk whose process crosses the ceiling is killed BY
ITS OWN PID and re-run file by file, so one heavy file is isolated rather than hiding a chunk.

    python run_suite_chunked.py --stack <tree>/stack --out <dir> [--chunk 25] [--mem-gb 3.8]

Writes ``<out>/SUITE.json``: per file pass/fail/skip counts from each chunk's junit XML, the killed files, and the
exact failing node ids. The admissible evidence is the JUNIT CONTENT, never the exit code."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import psutil


def run(files, stack: Path, xml: Path, mem_bytes: int, log) -> dict:
    env = dict(os.environ, PYTHONPATH=str(stack), PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--continue-on-collection-errors",
           f"--junitxml={xml}", *files]
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as lf:
        lf.write(f"\n### {time.strftime('%H:%M:%S')} {len(files)} files: {files[0]} .. {files[-1]}\n")
        lf.flush()
        p = subprocess.Popen(cmd, cwd=str(stack), env=env, stdout=lf, stderr=subprocess.STDOUT)
        peak, killed = 0, False
        pp = psutil.Process(p.pid)
        while p.poll() is None:
            try:
                m = pp.memory_info().private if hasattr(pp.memory_info(), "private") else pp.memory_info().rss
                for ch in pp.children(recursive=True):
                    try:
                        mi = ch.memory_info()
                        m += getattr(mi, "private", mi.rss)
                    except psutil.Error:
                        pass
            except psutil.Error:
                m = 0
            peak = max(peak, m)
            if m > mem_bytes:
                for ch in pp.children(recursive=True):
                    try:
                        ch.kill()
                    except psutil.Error:
                        pass
                p.kill()
                killed = True
                break
            time.sleep(0.5)
        p.wait()
    return {"rc": p.returncode, "killed_for_memory": killed, "peak_gb": round(peak / 2**30, 2),
            "secs": round(time.time() - t0, 1)}


def parse(xml: Path) -> dict:
    if not xml.is_file():
        return {"read": False}
    root = ET.parse(xml).getroot()
    out = {"read": True, "passed": 0, "failed": [], "errors": [], "skipped": 0}
    for tc in root.iter("testcase"):
        nid = f"{tc.get('classname')}::{tc.get('name')}"
        if tc.find("failure") is not None:
            out["failed"].append(nid)
        elif tc.find("error") is not None:
            out["errors"].append(nid)
        elif tc.find("skipped") is not None:
            out["skipped"] += 1
        else:
            out["passed"] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--mem-gb", type=float, default=3.8)
    ap.add_argument("--files", nargs="*", default=None, help="restrict to these test files (relative to stack)")
    a = ap.parse_args()
    stack = Path(a.stack).resolve()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = a.files or sorted(str(p.relative_to(stack)).replace("\\", "/") for p in (stack / "tests").glob("test_*.py"))
    mem = int(a.mem_gb * 2**30)
    log = out / "suite.log"
    res = {"stack": str(stack), "n_files": len(files), "mem_gb_ceiling": a.mem_gb, "chunks": [], "killed_files": [],
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    tot = {"passed": 0, "skipped": 0, "failed": [], "errors": [], "unread": []}

    def account(fs, tag):
        xml = out / f"junit_{tag}.xml"
        r = run(fs, stack, xml, mem, log)
        r.update(tag=tag, files=fs, **{k: v for k, v in parse(xml).items()})
        res["chunks"].append({k: v for k, v in r.items() if k not in ("failed", "errors")} |
                             {"n_failed": len(r.get("failed", [])), "n_errors": len(r.get("errors", []))})
        return r, xml

    for ci in range(0, len(files), a.chunk):
        fs = files[ci:ci + a.chunk]
        r, _ = account(fs, f"c{ci:04d}")
        if r["killed_for_memory"] or not r.get("read"):
            for f in fs:                                    # isolate: one file per process
                r1, _ = account([f], "f_" + Path(f).stem)
                if r1["killed_for_memory"]:
                    res["killed_files"].append({"file": f, "peak_gb": r1["peak_gb"]})
                elif not r1.get("read"):
                    tot["unread"].append(f)
                else:
                    tot["passed"] += r1["passed"]
                    tot["skipped"] += r1["skipped"]
                    tot["failed"] += r1["failed"]
                    tot["errors"] += r1["errors"]
        else:
            tot["passed"] += r["passed"]
            tot["skipped"] += r["skipped"]
            tot["failed"] += r["failed"]
            tot["errors"] += r["errors"]
        res["totals"] = tot
        res["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        (out / "SUITE.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out / "SUITE.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({"passed": tot["passed"], "skipped": tot["skipped"], "failed": len(tot["failed"]),
                      "errors": len(tot["errors"]), "killed": len(res["killed_files"]), "unread": len(tot["unread"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
