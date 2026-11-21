"""Detached finisher for the PREREG_V8F_OACC SECOND mutation proof (run 2). It outlives the build agent. Launch: Invoke-CimMethod Win32_Process Create 'cmd /c ...run_final.cmd < NUL'.

 0. snapshots the work dir into three fresh short-path copies: two mutation workers (C:/Users/Admin/oacc_r1, oacc_r2) and the FINAL tree for the suites (oacc_final);
 1. the FULL mutation proof (every mutant of raw/2026-10-25-oacc/mutate_oacc.py, baseline before and byte-identical restore after) in TWO workers (at most 2 concurrent pytest), each ONE process over a greedy cost-balanced
    `--idx` list (so each pays its baseline / restore once); C: free >= 4 GB before each launch; short basetemps C:/Users/Admin/pt_oacc_m1 / m2 (the harness deletes them after every run);
 2. banks raw/2026-10-25-oacc/mutate_oacc.out (the old one is mutate_oacc_run1.out) and checks: 0 SURVIVED, 0 RED-BY-EXCEPTION-ONLY, every baseline rc 0, every restore byte-identical;
 3. the suites: every stack/tests/test_v8f_*.py of the FINAL tree, SHORT basetemps (C:/Users/Admin/pt_oacc_s1 / s2), at most 2 concurrent pytest, C: free >= 4 GB before each run, basetemp deleted after each run; a file
    with failures is re-run on the BASE tree (C:/Users/Admin/oacc_base): red on both = PRE-EXISTING, red only on the final tree = REGRESSION. Banked as raw/2026-10-25-oacc/suites.out (the old one is suites_run1_longbasetemp.out);
 4. regenerates FILES.txt (tmp/mk_files.py) and writes tmp/final.done: its LAST line is `OK` or `NEEDS-REVIEW: <why>`. Log: tmp/final.log.
CPU only (CUDA_VISIBLE_DEVICES=-1). No git. Touches nothing outside the scratch work dir and C:/Users/Admin/{oacc_r1,oacc_r2,oacc_final,oacc_base,oacc_launch2,pt_oacc_*}."""
import concurrent.futures as cf
import ctypes
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# MEASURED 2026-10-25 (tmp/h/ctrlc_probe3.py): a console process launched through WMI can receive a CTRL_C console event within seconds of its start (two launches of this very script died in the
# first robocopy, rc 0xC000013A, and the output file held a bare "^C"). SetConsoleCtrlHandler(NULL, TRUE) makes this process IGNORE CTRL+C and the flag is INHERITED by every child (robocopy, the harness,
# pytest, the queue tests' `cmd /c`): with it the whole chain survived a CTRL_C sent to its console; without it every process died (KeyboardInterrupt / 0xC000013A).
try:
    ctypes.windll.kernel32.SetConsoleCtrlHandler(None, True)
except Exception:                                                                   # noqa: BLE001 -- not Windows: nothing to ignore
    pass

WORK = Path(r"C:\Users\Admin\AppData\Local\Temp\claude\E--Projects-TanitAD\bfe78918-15e8-439b-8566-593ca3297bd8\scratchpad\oacc")
TMP = WORK / "tmp"
PKREL = Path("TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini")
RAWO = WORK / PKREL / "raw" / "2026-10-25-oacc"
HOME = Path(r"C:\Users\Admin")
PY = str(HOME / "venvs" / "tanitad" / "Scripts" / "python.exe")
WORKERS = [HOME / "oacc_r1", HOME / "oacc_r2"]
FINAL = HOME / "oacc_final"
BASE = HOME / "oacc_base"
LAUNCH = HOME / "oacc_launch2"
MBT = [HOME / "pt_oacc_m1", HOME / "pt_oacc_m2"]
SBT = [HOME / "pt_oacc_s1", HOME / "pt_oacc_s2"]
MIN_FREE_GB = 4.0
LOG = TMP / "final.log"
DONE = TMP / "final.done"
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2",
       "PATH": r"C:\Program Files\Git\usr\bin;C:\Program Files\Git\bin;" + os.environ.get("PATH", "")}
TREE = ("stack", "taniteval", "Project Steering", "TanitAD Research Lab")
# cost per mutant in minutes, from run 1 (the wall time of its tests); a mutant that run 1 did not have costs 1.0
OLD_COST_MIN = None


def log(*a):
    with open(LOG, "a", encoding="utf-8", newline="\n") as f:
        f.write(time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a) + "\n")


def free_gb():
    return shutil.disk_usage("C:/").free / 2 ** 30


def wait_free():
    while free_gb() < MIN_FREE_GB:
        log("C: free below the floor, waiting", round(free_gb(), 2))
        time.sleep(30)


def sync(dst):
    for sub in TREE:
        r = subprocess.run(["robocopy", str(WORK / sub), str(dst / sub), "/MIR", "/NFL", "/NDL", "/NJH", "/NJS", "/NP", "/XD", "__pycache__", ".pytest_cache"], capture_output=True, text=True)
        if r.returncode >= 8:
            raise RuntimeError(f"robocopy {sub} -> {dst}: rc {r.returncode}: {r.stdout[-300:]}")
    shutil.copy2(WORK / "EXPORT_COMMIT.txt", dst / "EXPORT_COMMIT.txt")


def same_tree(dst):
    """The copy's test file, harness and every product file equal the work dir's (bytes)."""
    bad = []
    for rel in ("stack/tests/test_v8f_oacc.py", str(PKREL / "raw/2026-10-25-oacc/mutate_oacc.py"), "stack/tanitad/models/v8f_mini.py", "stack/scripts/v8f_mini_train.py", "stack/scripts/v8f_mini_eval_oacc.py",
                str(PKREL / "code/v8f_oacc_read.py"), str(PKREL / "code/v8f_mini_box2_queue.cmd"), str(PKREL / "code/box2_ops/chain_v8f_oacc.sh")):
        if (WORK / rel).read_bytes() != (dst / rel).read_bytes():
            bad.append(rel)
    return bad


def plan(copy):
    """-> the number of mutants and the two greedy cost-balanced `--idx` lists."""
    r = subprocess.run([PY, str(copy / PKREL / "raw/2026-10-25-oacc/mutate_oacc.py"), str(copy), PY, "--list"], capture_output=True, text=True, env=ENV, cwd=copy)
    n = len([ln for ln in r.stdout.splitlines() if re.match(r"^\d+ ", ln)])
    old = (RAWO / "mutate_oacc_run1.out").read_text(encoding="utf-8", errors="replace")
    cost = {int(m.group(1)): float(m.group(2)) / 60.0 for m in re.finditer(r"^(?:RED|SURVIVED)\s+#(\d+) .*? in ([\d.]+)s", old, re.M)}
    loads, lists = [0.0, 0.0], [[], []]
    for i in sorted(range(n), key=lambda i: -cost.get(i, 1.0)):                    # longest first onto the lighter worker
        w = 0 if loads[0] <= loads[1] else 1
        lists[w].append(i)
        loads[w] += cost.get(i, 1.0) * 1.3 + 0.2
    return n, [sorted(x) for x in lists], loads


def run_worker(k, idx):
    wait_free()
    out = LAUNCH / f"m{k + 1}.out"
    cmd = [PY, str(WORKERS[k] / PKREL / "raw/2026-10-25-oacc/mutate_oacc.py"), str(WORKERS[k]), PY, "--idx", ",".join(map(str, idx)), "--tmp", str(MBT[k])]
    log(f"worker {k + 1}: {len(idx)} mutants -> {out}")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=ENV, cwd=WORKERS[k])
    log(f"worker {k + 1}: rc {r.returncode}")
    return r.returncode


def bank_mutation(n):
    parts = []
    for k in range(2):
        p = LAUNCH / f"m{k + 1}.out"
        parts.append(f"===== worker m{k + 1} (--idx list, one process, its own baseline and restore) =====\n" + (p.read_text(encoding="utf-8", errors="replace") if p.exists() else "(no output)\n"))
    txt = "".join(parts)
    (RAWO / "mutate_oacc.out").write_bytes(txt.replace("\r\n", "\n").encode("utf-8"))
    ran = sum(int(m.group(2)) for m in re.finditer(r"^(\d+)/(\d+) RED", txt, re.M))
    n_red = sum(int(m.group(1)) for m in re.finditer(r"^(\d+)/(\d+) RED", txt, re.M))
    surv = re.findall(r"^SURVIVED .*$", txt, re.M)
    exc = re.findall(r"RED BY EXCEPTION ONLY[^\n]*", txt)
    seen = {int(m.group(1)) for m in re.finditer(r"^(?:RED|SURVIVED)\s+#(\d+) ", txt, re.M)}
    bad_base = [ln for ln in txt.splitlines() if ln.startswith("baseline:") and not ln.startswith("baseline: 0")]
    bad_rest = [ln for ln in txt.splitlines() if ln.startswith("restored:") and not (ln.startswith("restored: 0") and "byte-identical: True" in ln)]
    n_rest = len([ln for ln in txt.splitlines() if ln.startswith("restored:")])
    return dict(n=n, ran=ran, red=n_red, surv=surv, exc=exc, seen=len(seen), bad_base=bad_base, bad_rest=bad_rest, restores=n_rest)


def pytest_ids(out):
    return {m.group(2) for m in (re.match(r"^(FAILED|ERROR) (\S+)", ln) for ln in out.splitlines()) if m}


def run_file(root, rel, bt):
    wait_free()
    shutil.rmtree(bt, ignore_errors=True)
    t0 = time.time()
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(bt), "-o", "tmp_path_retention_policy=none", rel], cwd=root, capture_output=True, text=True, env=ENV, timeout=5400)
    shutil.rmtree(bt, ignore_errors=True)
    lines = [ln for ln in r.stdout.strip().splitlines() if re.search(r"\b(passed|failed|error|errors|skipped|deselected)\b", ln) and not ln.startswith(("FAILED", "ERROR"))]
    return r.returncode, (lines[-1] if lines else (r.stdout.strip().splitlines() or ["(no output)"])[-1]), pytest_ids(r.stdout), time.time() - t0


def suites():
    files = sorted(str(p.relative_to(FINAL)).replace("\\", "/") for p in (FINAL / "stack" / "tests").glob("test_v8f_*.py"))
    log("suites over", len(files), "files")
    slot = [0]

    def one(rel):
        k = 0 if rel.endswith("test_v8f_oacc.py") else 1                                 # distinct basetemps per concurrent run (a file is run on one at a time per worker thread)
        bt = SBT[k] / Path(rel).stem
        rc, summ, ids, dt = run_file(FINAL, rel, bt)
        rec = {"rel": rel, "rc": rc, "summary": summ, "ids": ids, "dt": dt, "base": None}
        if ids and (BASE / rel).exists() and not rel.endswith("test_v8f_oacc.py"):
            rc2, summ2, ids2, dt2 = run_file(BASE, rel, SBT[k] / (Path(rel).stem + "_base"))
            rec["base"] = {"rc": rc2, "summary": summ2, "ids": ids2}
        return rec
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        recs = list(ex.map(one, files))
    out = ["# suites.out (run 2) -- every stack/tests/test_v8f_*.py of the FINAL tree (a clean copy, C:/Users/Admin/oacc_final), CUDA_VISIBLE_DEVICES=-1, at most 2 concurrent pytest, SHORT basetemps (C:/Users/Admin/pt_oacc_s1 / s2: the first run's",
           "# long scratch-path basetemps broke fixtures on Windows and made it report 58 false regressions), C: free >= 4 GB before each run, basetemp deleted after each run, tmp_path_retention_policy=none.",
           "# A file with failing tests is re-run on the BASE tree (C:/Users/Admin/oacc_base = the export before any OACC change): a failure present on BOTH is PRE-EXISTING (environment / export), a failure ONLY on the final tree is a REGRESSION.", ""]
    regress, pre = [], 0
    for r in recs:
        line = f"{r['rel']}: rc {r['rc']} | {r['summary']} | {r['dt']:.0f}s"
        if r["ids"]:
            if r["base"] is None:
                line += f" | {len(r['ids'])} failing, NO BASE COPY OF THIS FILE (a new file): treated as REGRESSION"
                regress += [f"{r['rel']}::{i}" for i in sorted(r["ids"])]
            else:
                new = sorted(r["ids"] - r["base"]["ids"])
                same = sorted(r["ids"] & r["base"]["ids"])
                pre += len(same)
                line += f" | failing {len(r['ids'])}: PRE-EXISTING {len(same)} (also red on base: {r['base']['summary']}), REGRESSION {len(new)}"
                regress += new
        out.append(line)
        for i in sorted(r["ids"]):
            out.append(f"    {'PRE-EXISTING' if (r['base'] and i in r['base']['ids']) else 'REGRESSION'}  {i}")
    out += ["", f"files: {len(recs)}", f"PRE-EXISTING failures: {pre}", "REGRESSIONS: " + ("none" if not regress else f"{len(regress)}: {regress}")]
    (RAWO / "suites.out").write_bytes(("\n".join(out) + "\n").encode("utf-8"))
    return regress, recs


def cleanup():
    for d in [*MBT, *SBT]:
        shutil.rmtree(d, ignore_errors=True)
        log("basetemp removed:", d, "exists now:", d.exists())


def main():
    LAUNCH.mkdir(exist_ok=True)
    if DONE.exists():
        DONE.unlink()
    log("final.py start; C: free GB", round(free_gb(), 1))
    # the previous outputs are kept under their run-1 names (once)
    for old, new in (("mutate_oacc.out", "mutate_oacc_run1.out"), ("suites.out", "suites_run1_longbasetemp.out")):
        if not (RAWO / new).exists() and (RAWO / old).exists():
            shutil.copy2(RAWO / old, RAWO / new)
    why = []
    try:
        for d in (*WORKERS, FINAL):
            sync(d)
            bad = same_tree(d)
            log("synced", d, "differs from the work dir in:", bad)
            if bad:
                why.append(f"{d} differs from the work dir: {bad}")
        if why:
            raise RuntimeError("; ".join(why))
        n, lists, loads = plan(WORKERS[0])
        log(f"{n} mutants; worker lists {len(lists[0])} / {len(lists[1])}; estimated minutes {loads[0]:.0f} / {loads[1]:.0f}")
        with cf.ThreadPoolExecutor(max_workers=2) as ex:
            rcs = list(ex.map(lambda k: run_worker(k, lists[k]), (0, 1)))
        res = bank_mutation(n)
        log("mutation banked:", {k: v for k, v in res.items() if k not in ("surv", "exc", "bad_base", "bad_rest")}, "survivors", len(res["surv"]), "exception-only", len(res["exc"]))
        if rcs != [0, 0]:
            why.append(f"a mutation worker exited {rcs}")
        if res["surv"]:
            why.append(f"{len(res['surv'])} mutant(s) SURVIVED")
        if res["exc"]:
            why.append("mutants RED by exception only: " + "; ".join(res["exc"])[:300])
        if res["bad_base"] or res["bad_rest"] or res["restores"] != 2:
            why.append(f"baseline / restore not clean: {res['bad_base']} {res['bad_rest']} restores {res['restores']}")
        if res["seen"] != n or res["red"] != n:
            why.append(f"{res['red']}/{res['seen']} RED of {n} mutants")
        regress, recs = suites()
        log("suites banked; regressions:", len(regress))
        if regress:
            why.append(f"{len(regress)} suite regression(s)")
    except BaseException as e:                                                       # noqa: BLE001
        why.append(f"the finisher itself failed: {type(e).__name__}: {e}")
        log("EXCEPTION", repr(e))
    finally:
        cleanup()
        r = subprocess.run([sys.executable, "mk_files.py"], cwd=TMP, capture_output=True, text=True)
        log("mk_files rc", r.returncode, r.stderr[-200:])
        DONE.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + " final.py finished\n" + ("OK" if not why else "NEEDS-REVIEW: " + "; ".join(why)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
