"""OACC proof, run 3 (MM, 2026-10-25 22:2x): after final.py's run 2 read NEEDS-REVIEW.

Run 2's two findings, diagnosed before this script:
 1. 4 mutants RED BY EXCEPTION ONLY (#37, #66, #164, #307). Three were test evaluation order (a `r["key"]` read before the assertion
    that the key exists); #66 was caught by `_no_crash`'s pytest.fail, but pytest printed the instrument's chained ValueError first and
    the harness reads the FIRST E line. test_v8f_oacc.py hardened (tmp/test_v8f_oacc.before_exc_hardening.py is the old copy).
 2. 131 "regressions" + 882 "pre-existing": every suite run pointed --basetemp at C:/Users/Admin/pt_oacc_s{1,2}/<stem> after the
    parents had been removed; pytest creates the basetemp WITHOUT parents, so every tmp_path test errored with FileNotFoundError on
    both trees (MEASURED: test_v8f_dk.py 25 errors with the parent absent, 25 passed with it present).
This run: the 4 mutants again (baseline + restore included) on oacc_r1, then the suites with the parents created, then FILES.txt.
"""
import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path(r"C:\Users\Admin\AppData\Local\Temp\claude\E--Projects-TanitAD\bfe78918-15e8-439b-8566-593ca3297bd8\scratchpad\oacc")
TMP = WORK / "tmp"
spec = importlib.util.spec_from_file_location("final_oacc", TMP / "final.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)                                     # main() is guarded by __name__; this also installs the CTRL_C ignore
RUN = sys.argv[1] if len(sys.argv) > 1 else "3"
IDX = sys.argv[2] if len(sys.argv) > 2 else "37,66,164,307"
DONE = TMP / f"final{RUN}.done"
T = "stack/tests/test_v8f_oacc.py"


def log(*a):
    F.log(f"[run{RUN}]", *a)


def main():
    if DONE.exists():
        DONE.unlink()
    why = []
    for d in (F.WORKERS[0], F.FINAL):
        shutil.copy2(WORK / T, d / T)
        assert (d / T).read_bytes() == (WORK / T).read_bytes(), d
        bad = F.same_tree(d)
        log("synced", d, "differs from the work dir in:", bad)
        if bad:
            why.append(f"{d} differs from the work dir: {bad}")
    if (F.RAWO / "suites.out").exists() and not (F.RAWO / "suites_run2_missing_basetemp_parent.out").exists():
        shutil.copy2(F.RAWO / "suites.out", F.RAWO / "suites_run2_missing_basetemp_parent.out")
    elif RUN != "3" and (F.RAWO / "suites.out").exists():
        shutil.copy2(F.RAWO / "suites.out", F.RAWO / f"suites_run{int(RUN) - 1}.out")
    mt = Path("C:/Users/Admin/pt_oacc_x")
    mt.mkdir(parents=True, exist_ok=True)
    F.wait_free()
    t0 = time.time()
    r = subprocess.run([F.PY, str(F.WORKERS[0] / F.PKREL / "raw" / "2026-10-25-oacc" / "mutate_oacc.py"), str(F.WORKERS[0]), F.PY, "--idx", IDX, "--tmp", str(mt)],
                       cwd=F.WORKERS[0], capture_output=True, text=True, env=F.ENV, timeout=5400)
    out = r.stdout + ("\n--- stderr ---\n" + r.stderr if r.stderr.strip() else "")
    (F.RAWO / f"mutate_oacc_run{RUN}_exc_hardening.out").write_bytes(out.encode("utf-8"))
    last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    log(f"mutants {IDX}: rc", r.returncode, f"{time.time() - t0:.0f}s |", last)
    n_idx = len(IDX.split(","))
    if r.returncode != 0 or f"{n_idx}/{n_idx} RED" not in last or "EXCEPTION ONLY" in last or "SURVIVED" in last:
        why.append(f"run-{RUN} mutants: rc {r.returncode}: {last}")
    shutil.rmtree(mt, ignore_errors=True)
    for d in F.SBT:
        d.mkdir(parents=True, exist_ok=True)                   # THE FIX: pytest makes --basetemp without parents
    regress, recs = F.suites()
    pre = sum(len(x["ids"] & x["base"]["ids"]) for x in recs if x["ids"] and x["base"])
    log("suites banked; regressions:", len(regress), "pre-existing:", pre)
    if regress:
        why.append(f"{len(regress)} suite regression(s): {regress[:5]}")
    for d in F.SBT:
        shutil.rmtree(d, ignore_errors=True)
    m = subprocess.run([sys.executable, str(TMP / "mk_files.py")], capture_output=True, text=True)
    log("mk_files rc", m.returncode, m.stdout.strip()[-200:])
    if m.returncode != 0:
        why.append("mk_files failed")
    DONE.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + " rerun_exc.py finished\n" + ("OK" if not why else "NEEDS-REVIEW: " + "; ".join(why)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:                                     # noqa: BLE001
        DONE.write_text(f"NEEDS-REVIEW: rerun_exc.py crashed: {type(e).__name__}: {e}\n", encoding="utf-8")
        raise
