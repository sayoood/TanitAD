"""Mutation proof for orc_read.py: every mutant is a textual edit of a COPY of the reader; the test suite (pointed at the copy through ORC_READ_PATH)
must turn RED. A mutant that leaves the suite GREEN is reported SURVIVED and the script exits 1.

    python mutate_orc.py [--only M01,M05] [--full] [--out out/mutation.out]

Each mutant is first run against its TARGETED tests (`-k`, a few seconds); one that survives them is re-run against the WHOLE suite before it is called
survived. The pytest cap (never a third concurrent pytest process) is honoured by polling the process list before every run.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "orc_read.py"
WORK = pathlib.Path("E:/Projects/TanitAD-artifacts/orcread_build/mut")
PT = pathlib.Path("E:/Projects/TanitAD-artifacts/orcread_build/pt")
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"


def sub(old, new, count=1):
    def f(s):
        assert s.count(old) == count, (s.count(old), old)
        return s.replace(old, new)
    return f


def swap_table_rows(s):
    a, b = '("ROBUST", "R-TLAT and R-TLON', '("PARTIAL", "at least one F1 row'
    assert s.count(a) == 1 and s.count(b) == 1
    return s.replace(a, "@@A@@").replace(b, "@@B@@").replace("@@A@@", b).replace("@@B@@", a)


# id, what it breaks, edit, targeted -k
MUTANTS = [
    ("M01", "flip a bar sign (>= 2F becomes <= 2F)", sub('sep_better(s, hd) and r["delta"] >= TWO_F[hd]', 'sep_better(s, hd) and r["delta"] <= TWO_F[hd]'), "decide"),
    ("M02", "drop a seed (SEEDS = (0,))", sub("SEEDS = (0, 1)", "SEEDS = (0,)"), "decide or world_ROBUST"),
    ("M03", "swap the lat / lon bars", sub('TWO_F = {"lat": 0.0248, "lon": 0.0146}', 'TWO_F = {"lat": 0.0146, "lon": 0.0248}'), "decide or committed"),
    ("M04", "skip a control (the perturbed-fraction band)", sub("if not (RATIO_CENTRE - RATIO_TOL - 1e-12 <= ratio <= RATIO_CENTRE + RATIO_TOL + 1e-12):", "if False:"),
     "g_pert_ratio"),
    ("M05", "off-by-one tick >= 1 (ticks >= 2)", sub('n)[:, 1:]', 'n)[:, 2:]', count=2), "ticks_ge1 or matched or world_ROBUST"),
    ("M23", "off-by-one tick, non-crashing (ticks 0-1 instead of 1-2)", sub('n)[:, 1:]', 'n)[:, :2]', count=2), "ticks_ge1 or matched or world_ROBUST"),
    ("M06", "wrong reducer direction (ORC0 - ORC)", sub("a, b, eid2, n_boot, red, record=False,\n                  lower_is_better=False", "b, a, eid2, n_boot, red, record=False,\n                  lower_is_better=False"),
     "F1_sign or world_ROBUST"),
    ("M07", "outcome order swap (ROBUST and PARTIAL rows trade places)", swap_table_rows, "table"),
    ("M08", "pair runs by different windows (the SAME-WINDOWS control is blind)", sub('ctl.add(f"SAME-WINDOWS:{arm}-vs-ORC0:s{s}", same,', 'ctl.add(f"SAME-WINDOWS:{arm}-vs-ORC0:s{s}", True,'),
     "different_windows or t0_episodes"),
    ("M09", "R-ADE guard direction (separated-BETTER counts as worse)", sub('rows[s]["R-ADE"]["delta"] > 0) for s in SEEDS}', 'rows[s]["R-ADE"]["delta"] < 0) for s in SEEDS}'), "decide or HARMS"),
    ("M10", "R-T0 bar is exclusive (>= F/2 is above)", sub('rows[s]["R-T0"]["point"] > HALF_F', 'rows[s]["R-T0"]["point"] >= HALF_F'), "guard_boundaries"),
    ("M11", "matched cells off (each run's own cells)", sub("    if matched:\n        m = (ea >= 0)", "    if False:\n        m = (ea >= 0)"), "matched_cells"),
    ("M12", "the K1-K7 results of a T1 run are ignored", sub('ctl.add(f"{k}:{tag}", ok, det)', 'ctl.add(f"{k}:{tag}", True, det)'), "every_control_is_wired"),
    ("M13", "A2 reverted: NO EFFECT on the as-written predicate (a sub-bar separated head falls into the hole)", sub('"NO EFFECT": bool(not moved_both["lat"] and not moved_both["lon"] and guards_held)', '"NO EFFECT": bool(not sep_both["lat"] and not sep_both["lon"] and guards_held)'), "former_hole or a2_table"),
    ("M14", "the init-checkpoint sha control always passes", sub("    bad = [f\"{w} = {str(v)[:16]}\" for w, v in vals if str(v) != m2b_sha]\n    return not bad,", "    bad = []\n    return not bad,"),
     "g_init_sha or every_control_is_wired"),
    ("M15", "ORC-0 may perturb (n == 0 not required)", sub('if not (isinstance(n, (int, float)) and n == 0):', "if False:"), "g_pert_orc0 or every_control_is_wired"),
    ("M16", "the launch-gate verdict is not read", sub('if tok.get("verdict") != "PASS":', "if False:"), "g_token or every_control_is_wired"),
    ("M17", "the reader's own K1 identity control is blind", sub('if not (r["delta"] == 0 and r["lo"] == 0 and r["hi"] == 0):', "if False:"), "k1_own"),
    ("M18", "the guards ignore R-T0", sub("guards_held = not any(ade_worse.values()) and not any(t0_above.values())", "guards_held = not any(ade_worse.values())"), "decide or HARMS"),
    ("M19", "a head 'moved' at ONE seed counts as moved at both", sub("moved_both = {hd: all(moved(s, hd) for s in SEEDS) for hd in (\"lat\", \"lon\")}",
                                                                    "moved_both = {hd: any(moved(s, hd) for s in SEEDS) for hd in (\"lat\", \"lon\")}"), "decide or needs_both_seeds"),
    ("M20", "separation is not required for 'better'", sub('return bool(r["separated"] and r["delta"] > 0)', 'return bool(r["delta"] > 0)'), "decide"),
    ("M21", "the registered ORC perturbation p is wrong (0.0 instead of 0.3)", sub('"--ego-perturb-p": PERT_P[arm]}', '"--ego-perturb-p": 0.0}'), "g_argv or world_ROBUST"),
    ("M22", "a failed control still computes rows (REFUSED does not stop the read)", sub("    if ctl.failed():\n        raise Refusal(", "    if False:\n        raise Refusal("), "failed_control or every_control_is_wired"),
]


def n_pytest():
    """Concurrent pytest RUNS. A venv launcher and its child python are ONE run, so a plain process count reads 2 for a single run: count the pytest
    processes whose parent is not itself a pytest process."""
    q = ("$p = @(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { $_.CommandLine -match 'pytest' }); $ids = @($p | % { $_.ProcessId }); "
         "@($p | ? { $ids -notcontains $_.ParentProcessId }).Count")
    r = subprocess.run(["powershell", "-NoProfile", "-c", q], capture_output=True, text=True)
    try:
        return int(r.stdout.strip() or 0)
    except ValueError:
        return 0


def run_pytest(mut_path, k, tag, timeout=900):
    while n_pytest() >= 2:
        time.sleep(60)
    env = dict(os.environ, ORC_READ_PATH=str(mut_path), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    PT.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", f"--basetemp={PT}/mut_{tag}_{int(time.time())}", str(HERE / "test_orc_read.py")]
    if k:
        cmd += ["-k", k]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    tail = [ln for ln in r.stdout.splitlines() if ln.strip()][-3:]
    return r.returncode, time.time() - t, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--full", action="store_true", help="run every mutant against the whole suite, not the targeted tests")
    ap.add_argument("--out", default=str(HERE / "out" / "mutation.out"))
    a = ap.parse_args()
    only = {x for x in a.only.split(",") if x}
    WORK.mkdir(parents=True, exist_ok=True)
    src = SRC.read_text(encoding="utf-8")
    lines, survived = [], []
    for mid, what, edit, k in MUTANTS:
        if only and mid not in only:
            continue
        mut = edit(src)
        assert mut != src, mid
        p = WORK / f"orc_read_{mid}.py"
        p.write_text(mut, encoding="utf-8")
        rc, dt, tail = run_pytest(p, None if a.full else k, mid)
        scope = "whole suite" if a.full else f"-k '{k}'"
        if rc == 0 and not a.full:                       # survived the targeted tests: the whole suite decides
            rc, dt2, tail = run_pytest(p, None, mid + "_full")
            dt += dt2
            scope = "whole suite (after the targeted tests stayed green)"
        if rc == 5:                                      # pytest: no tests collected -- a typo in -k must not read as a kill
            verdict = "ERROR (no test selected)"
            survived.append(mid)
        else:
            verdict = "KILLED (RED)" if rc != 0 else "SURVIVED (GREEN)"
            if rc == 0:
                survived.append(mid)
        ln = f"{mid} {verdict:17s} {dt:5.1f}s  [{scope}]  {what}\n      pytest tail: {' | '.join(tail)}"
        print(ln, flush=True)
        lines.append(ln)
    outp = pathlib.Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if only else "w"
    with open(outp, mode, encoding="utf-8", newline="\n") as f:
        f.write(f"# mutation proof of orc_read.py (sha256 of the unmutated reader: {__import__('hashlib').sha256(SRC.read_bytes()).hexdigest()})\n")
        f.write("\n".join(lines) + "\n")
        f.write(f"# {len(lines) - len(survived)} of {len(lines)} mutants KILLED; survived: {survived or 'none'}\n")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
