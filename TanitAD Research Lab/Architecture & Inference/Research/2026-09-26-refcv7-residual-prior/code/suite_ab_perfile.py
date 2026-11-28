#!/usr/bin/env python3
"""Clean-tree A/B, ONE TEST FILE PER PROCESS, with a RAM watchdog (the dev box is the PI's desktop).

For each file in the list, run it on the TIP tree and then on the CANDIDATE tree, each in a fresh
pytest process with its own JUnit XML (never a log tail). A watchdog polls the machine's available
RAM every 0.5 s and KILLS the pytest process tree if it falls below --min-free-gb; that file is
then recorded as ABORTED_RAM for that tree -- reported, never counted as a pass. Per-test outcomes
are diffed tip vs cand; a regression = failure/error in cand where the tip passed or skipped.

usage: python suite_ab_perfile.py <tip_tree> <cand_tree> <out_dir> <tests.txt> [--min-free-gb 6]
Resumable: a file whose two XMLs already exist is not re-run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import psutil


def _kill_tree(pid: int) -> None:
    try:
        p = psutil.Process(pid)
        for c in p.children(recursive=True):
            c.kill()
        p.kill()
    except psutil.NoSuchProcess:
        pass


def run_one(tree: str, test: str, xml_path: str, min_free_gb: float, timeout_s: int) -> dict:
    stack = os.path.join(tree, "stack")
    env = dict(os.environ, PYTHONPATH=stack.replace("\\", "/"),
               OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"), HF_HUB_OFFLINE="1")
    while psutil.virtual_memory().available / 2**30 < min_free_gb + 1.0:
        time.sleep(5)                      # wait for headroom BEFORE starting
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                          f"--junitxml={xml_path}", os.path.join("tests", test)], cwd=stack, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    aborted = None
    peak = 0.0
    while p.poll() is None:
        avail = psutil.virtual_memory().available / 2**30
        try:
            rss = sum(c.memory_info().rss for c in [psutil.Process(p.pid)] +
                      psutil.Process(p.pid).children(recursive=True)) / 2**30
            peak = max(peak, rss)
        except psutil.NoSuchProcess:
            pass
        if avail < min_free_gb:
            aborted = f"ABORTED_RAM (available {avail:.2f} GB < {min_free_gb})"
        elif time.time() - t0 > timeout_s:
            aborted = f"ABORTED_TIMEOUT (> {timeout_s} s)"
        if aborted:
            _kill_tree(p.pid)
            break
        time.sleep(0.5)
    out = p.communicate()[0] if aborted is None else ""
    rec = {"test": test, "wall_s": round(time.time() - t0, 1), "peak_rss_gb": round(peak, 2),
           "exit": p.returncode, "aborted": aborted}
    if aborted:
        return rec
    if not os.path.isfile(xml_path):
        rec["aborted"] = "NO_XML (collection crashed?)"
        rec["tail"] = out[-1500:]
        return rec
    oc = {}
    for tc in ET.parse(xml_path).getroot().iter("testcase"):
        kind = "passed"
        for ch in tc:
            if ch.tag in ("failure", "error"):
                kind = ch.tag
            elif ch.tag == "skipped":
                kind = "skipped"
        oc[f"{tc.get('classname')}::{tc.get('name')}"] = kind
    rec["outcomes"] = oc
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tip")
    ap.add_argument("cand")
    ap.add_argument("out")
    ap.add_argument("tests")
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    ap.add_argument("--timeout-s", type=int, default=900)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tests = [l.strip() for l in open(a.tests, encoding="utf-8") if l.strip()]
    state_path = os.path.join(a.out, "perfile_state.json")
    state = json.load(open(state_path, encoding="utf-8")) if os.path.isfile(state_path) else {}
    for i, t in enumerate(tests):
        if t in state and all(state[t].get(k, {}).get("aborted") is None and "outcomes" in state[t].get(k, {})
                              for k in ("tip", "cand")):
            continue
        state[t] = {}
        for name, tree in (("tip", a.tip), ("cand", a.cand)):
            xml = os.path.join(a.out, f"{name}__{t}.xml")
            if os.path.isfile(xml):
                os.remove(xml)
            state[t][name] = run_one(tree, t, xml, a.min_free_gb, a.timeout_s)
        json.dump(state, open(state_path, "w", encoding="utf-8"), indent=1)
        r = state[t]
        cnt = lambda rr: {k: sum(1 for v in rr.get("outcomes", {}).values() if v == k)  # noqa: E731
                          for k in ("passed", "failure", "error", "skipped")}
        print(f"[{i + 1}/{len(tests)}] {t}: tip {cnt(r['tip'])} {r['tip'].get('aborted') or ''} | "
              f"cand {cnt(r['cand'])} {r['cand'].get('aborted') or ''} | peak "
              f"{r['tip'].get('peak_rss_gb')}/{r['cand'].get('peak_rss_gb')} GB", flush=True)
    bad = lambda o: o in ("failure", "error")  # noqa: E731
    tot = {n: {"passed": 0, "failure": 0, "error": 0, "skipped": 0} for n in ("tip", "cand")}
    regress, fixed, aborted = [], [], []
    for t, r in state.items():
        for n in ("tip", "cand"):
            if r[n].get("aborted"):
                aborted.append((t, n, r[n]["aborted"]))
            for v in r[n].get("outcomes", {}).values():
                tot[n][v] += 1
        ot, oc = r["tip"].get("outcomes", {}), r["cand"].get("outcomes", {})
        regress += [x for x, o in oc.items() if bad(o) and not bad(ot.get(x, "absent"))]
        fixed += [x for x, o in ot.items() if bad(o) and not bad(oc.get(x, "absent"))]
    summary = {"n_files": len(tests), "totals": tot, "regressions_in_cand": sorted(regress),
               "fixed_in_cand": sorted(fixed), "aborted": aborted,
               "tip_failures": sorted(x for r in state.values() for x, o in r["tip"].get("outcomes", {}).items() if bad(o)),
               "cand_failures": sorted(x for r in state.values() for x, o in r["cand"].get("outcomes", {}).items() if bad(o))}
    json.dump(summary, open(os.path.join(a.out, "suite_ab_perfile_summary.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
