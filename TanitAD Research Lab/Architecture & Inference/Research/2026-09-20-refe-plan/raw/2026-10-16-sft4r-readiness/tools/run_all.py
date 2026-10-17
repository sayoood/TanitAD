#!/usr/bin/env python3
"""run_all.py -- SFT-4R readiness: run every test that names an edited file, in FULL form, in a SHORT path tree, each under the watchdog (bounded_run.py).  CPU only (CUDA_VISIBLE_DEVICES="").

usage: python run_all.py [--workers 3] [--only id,id]        (detached: see HANDOFF.md; it banks SUMMARY.txt line by line and writes TEST_TABLE.md + queue_done.flag at the end)

Forms.  "gate" = the environment check_suite gives a suite script (REFE_GATE_NESTED = the JSON stack holding the script itself on the PATCHED tree; "1" on the unpatched tree, which is what the old
gate exported).  "alone" = no sentinel (how a person runs it; the form the 2026-10-16 runaway started from).  Every test runs in "gate" form; the fast ones also "alone".
A failure is re-run ALONE (serial, after the pool is done) to separate contention from a defect, and then on the UNPATCHED tree of the same commit to separate PRE-EXISTING from REGRESSION.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path("C:/s4r")
PKR = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TREES = {"patched": ROOT / "p" / PKR / "refe", "baseline": ROOT / "b" / PKR / "refe"}
PYV = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
BOUND = ROOT / "tools" / "bounded_run.py"
LOGS = ROOT / "logs" / "q"
SUMMARY = ROOT / "SUMMARY.txt"
LOCK = threading.Lock()

# id, script, args, expected token, timeout s, run "alone" too, has an unpatched counterpart (a NEW test is judged on the patched tree alone)
TESTS = [
    ("test_launch_gate--full", "test_launch_gate.py", [], "ZZLG_FULL_PASS", 3000, False, True),
    ("test_p6_chain", "test_p6_chain.py", [], "ZZP6CHAIN_PASS", 2400, False, True),
    ("test_gate_suite_prerun", "test_gate_suite_prerun.py", [], "ZZTEST_PASS", 1800, False, True),
    ("test_p2rep_chain", "test_p2rep_chain.py", [], "ZZTEST_PASS", 1800, False, True),
    ("test_p6_levers", "test_p6_levers.py", [], "ZZP6_PASS", 1800, False, True),
    ("test_rls1_chain", "test_rls1_chain.py", [], "ZZTEST_PASS", 1500, False, True),
    ("test_s1c", "test_s1c.py", [], "ZZTEST_PASS", 1500, False, True),
    ("test_aux_s1_wiring", "test_aux_s1_wiring.py", [], "ZZTEST_PASS", 1500, False, True),
    ("test_aux_g12_bf16", "test_aux_g12_bf16.py", [], "ZZTEST_PASS", 1500, False, True),
    ("test_s1aux_chain", "test_s1aux_chain.py", [], "ZZTEST_PASS", 1500, False, True),
    ("test_rl_r0c", "test_rl_r0c.py", [], "ZZTEST_PASS", 1200, False, True),
    ("e2e_suite_prerun_refe_sft4r", "launch_gate.py", ["suite-prerun", "--profile", "refe_sft4r", "--pkg", str(ROOT / "p" / PKR), "--out", str(ROOT / "logs" / "q" / "suite_prerun_final.json"),
                                                      "--key-file", str(ROOT / "keys" / "gate.key")], "ZZSUITE-PRERUN-PASSZZ", 1500, False, False),
    ("test_launch_gate--static", "test_launch_gate.py", ["--static", "--no-banked"], "ZZLG_STATIC_PASS", 1200, True, True),
    ("test_sft4r_gate", "test_sft4r_gate.py", [], "ZZTEST_PASS", 900, True, False),
    ("test_suite_nesting", "test_suite_nesting.py", [], "ZZTEST_PASS", 900, True, False),
    ("test_p2r_chain", "test_p2r_chain.py", [], "ZZTEST_PASS", 900, True, True),
    ("test_rl_s1b", "test_rl_s1b.py", [], "ZZTEST_PASS", 900, False, True),
    ("test_p6_cpu_probe", "test_p6_cpu_probe.py", [], "ZZTEST_PASS", 900, False, True),
    ("test_p6_overlap", "test_p6_overlap.py", [], "ZZTEST_PASS", 900, False, True),
    ("test_rls1_queue", "test_rls1_queue.py", [], "ZZTEST_PASS", 600, True, True),
    ("test_sft4r_chain", "test_sft4r_chain.py", [], "ZZTEST_PASS", 600, True, True),
    ("test_hf_egress", "test_hf_egress.py", [], "ZZTEST_PASS", 600, True, True),
    ("test_pdm_label_complete", "test_pdm_label_complete.py", [], "ZZTEST_PASS", 600, True, True),
    ("test_gate_gpu_probes", "test_gate_gpu_probes.py", [], "ZZTEST_PASS", 600, True, True),
]


def stack_env(tree: str, script: str, args: list) -> str:
    """What check_suite exports for this entry: the JSON stack on the patched tree, '1' on the unpatched one."""
    if tree == "baseline":
        return "1"
    sys.path.insert(0, str(TREES["patched"]))
    import gate_checks as C
    return json.dumps([C.suite_entry_id({"script": script, "args": args})])


def bank(line: str) -> None:
    with LOCK:
        with open(SUMMARY, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def one(tid, script, args, token, timeout, tree, form, tag):
    d = TREES[tree]
    if not (d / script).is_file():
        return {"tid": tid, "tree": tree, "form": form, "tag": tag, "rc": None, "token": None, "secs": 0, "peak": None, "killed": "", "last": "ABSENT in this tree", "ok": None}
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{tid}.{tree}.{form}{tag}.txt"
    if log.exists():
        log.unlink()
    env_args = ["--env", "CUDA_VISIBLE_DEVICES=", "--env", "PYTHONIOENCODING=utf-8"]
    if form == "gate":
        env_args += ["--env", f"REFE_GATE_NESTED={stack_env(tree, script, args)}"]
    else:
        env_args += ["--env", "REFE_GATE_NESTED="]
    for cap in (20, 40):
        if log.exists():
            log.unlink()
        cmd = [PYV, "-X", "utf8", str(BOUND), "--timeout", str(timeout), "--max-children", str(cap), "--log", str(log), "--cwd", str(d), *env_args, "--", PYV, "-X", "utf8", script, *args]
        subprocess.run(cmd, capture_output=True, text=True)
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        m = re.search(r"ZZBOUNDED rc=(-?\d+) peak_children=(\d+) secs=([\d.]+) killed=(\w+)", text)
        rc, peak, secs, killed = (int(m.group(1)), int(m.group(2)), float(m.group(3)), m.group(4)) if m else (None, None, 0.0, "?")
        if killed != "children":
            break                                                          # a legitimate peak above 20 is re-run once at 40 and RECORDED, never silently allowed
    mt = re.findall(rf"^({re.escape(token)}\b.*)$", text, flags=re.M)
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("ZZBOUNDED") and not ln.startswith("---") and not re.match(r"^\d+ \d+ ", ln)]
    last = (mt[-1] if mt else (lines[-1] if lines else ""))[:170]
    skips = len(re.findall(r"^ZZTEST_SKIP", text, flags=re.M))
    ok = rc == 0 and bool(mt) and skips == 0 and killed == "none"
    return {"tid": tid, "tree": tree, "form": form, "tag": tag, "rc": rc, "token": (mt[-1] if mt else None), "secs": secs, "peak": peak, "killed": killed, "cap": cap, "skips": skips, "last": last, "ok": ok}


def line_of(r: dict, extra: str = "") -> str:
    return (f"{r['tid']:26s} {r['tree']:8s} {r['form']:6s}{r['tag']:7s} rc={r['rc']} {'PASS' if r['ok'] else ('ABSENT' if r['ok'] is None else 'FAIL'):6s} secs={r['secs']:<7} peak_children={r['peak']} "
            f"killed={r['killed']} cap={r.get('cap')} skips={r.get('skips')} token=[{(r['token'] or 'NONE')[:60]}] last=[{r['last'][:110]}] {extra}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only")
    a = ap.parse_args()
    # bash on PATH for the chain tests (a detached process may not inherit Git Bash's dirs)
    for extra in ("C:/Program Files/Git/bin", "C:/Program Files/Git/usr/bin"):
        if Path(extra).is_dir():
            os.environ["PATH"] = extra.replace("/", "\\") + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("HOME", os.environ.get("USERPROFILE", "C:\\Users\\Admin"))      # a WMI-detached process has no HOME (Git Bash sets one)
    tests = [t for t in TESTS if not a.only or t[0] in a.only.split(",")]
    SUMMARY.write_text(f"# SFT-4R readiness test queue, started {time.strftime('%Y-%m-%d %H:%M:%S')}; workers={a.workers}; CUDA_VISIBLE_DEVICES=''; each line = one run under bounded_run.py\n", encoding="utf-8")
    results = []
    jobs = []
    for tid, script, args, token, timeout, alone, _hb in tests:
        jobs.append((tid, script, args, token, timeout, "patched", "gate", ""))
        if alone:
            jobs.append((tid, script, args, token, timeout, "patched", "alone", ""))

    def run_job(j):
        r = one(*j)
        bank(line_of(r))
        return r
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(run_job, jobs))
    # ---- failures: re-run ALONE (serial), then on the unpatched tree
    final = {}
    for r in results:
        key = (r["tid"], r["form"])
        final[key] = {"first": r, "alone": None, "baseline": None, "class": "PASS" if r["ok"] else None}
    spec = {t[0]: t for t in TESTS}
    for (tid, form), d in final.items():
        if d["first"]["ok"]:
            continue
        _, script, args, token, timeout, _a, has_base = spec[tid]
        r2 = one(tid, script, args, token, timeout, "patched", form, ".rerun")
        bank(line_of(r2, "<- serial re-run of a failure"))
        d["alone"] = r2
        if r2["ok"]:
            d["class"] = "PASS-ON-SERIAL-RERUN (failed under contention)"
            continue
        if not has_base:
            d["class"] = "NEW-TEST-FAILS (no unpatched counterpart)"
            continue
        rb = one(tid, script, args, token, timeout, "baseline", form, "")
        bank(line_of(rb, "<- the same test on the UNPATCHED tree"))
        d["baseline"] = rb
        if rb["ok"] is None:
            d["class"] = "NEW-TEST-FAILS (no unpatched counterpart)"
        elif rb["ok"]:
            d["class"] = "REGRESSION (passes unpatched)"
        else:
            d["class"] = "PRE-EXISTING (fails unpatched too)"
    # ---- table
    rows = ["| test | form | patched (rc / token / s / peak procs) | verdict | unpatched, if it failed |", "|---|---|---|---|---|"]
    for (tid, form), d in final.items():
        f = d["alone"] if (d["alone"] and d["alone"]["ok"]) else d["first"]
        b = d["baseline"]
        rows.append(f"| {tid} | {form} | rc {f['rc']} / `{(f['token'] or 'NONE')[:40]}` / {f['secs']} s / {f['peak']} | **{d['class']}** | "
                    f"{('rc ' + str(b['rc']) + ' ' + b['last'][:70]) if b else ''} |")
    (ROOT / "TEST_TABLE.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    n_bad = sum(1 for d in final.values() if d["class"] not in ("PASS", "PASS-ON-SERIAL-RERUN (failed under contention)"))
    bank(f"# DONE {time.strftime('%Y-%m-%d %H:%M:%S')}: {len(final)} (test, form) pairs; {n_bad} not clean; table: C:/s4r/TEST_TABLE.md")
    (ROOT / "queue_done.flag").write_text(f"{len(final)} {n_bad}\n", encoding="utf-8")
    try:                                                                                  # copy the results into the package (E:), best effort; HANDOFF.md says how to repeat it
        subprocess.run([PYV, "-X", "utf8", str(ROOT / "tools" / "build_package.py"), "--results-only"], capture_output=True, text=True, timeout=300)
    except Exception as e:                                                                # noqa: BLE001
        bank(f"# finalize_package.py failed: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
