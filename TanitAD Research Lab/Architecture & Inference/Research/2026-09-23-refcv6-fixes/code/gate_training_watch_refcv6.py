"""Gate for commit I: the watch test on a CLEAN tree (tip + exactly this commit's files), then a
3-arm mutation proof of the health verdicts. Writes mutation_proof_training_watch_refcv6.json."""
import hashlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile

TIP = "d8bd5d4b55b2f112e497f37783eeae621235ab0f"
GIT = dict(os.environ, GIT_DIR="C:/Users/Admin/tanitad-push/.git")
TC = pathlib.Path("C:/Users/Admin/tcI")
W = pathlib.Path("C:/Users/Admin/qland/work/watch6")
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
FILES = {"taniteval/tools/training_watch/build_watch_refcv6.py": W / "build_watch_refcv6.py",
         "taniteval/tests/test_training_watch_refcv6.py": W / "test_training_watch_refcv6.py"}
BUILDER = TC / "taniteval/tools/training_watch/build_watch_refcv6.py"
TEST = TC / "taniteval/tests/test_training_watch_refcv6.py"
ARMS = [
    ("W1_unplanned_pinned_to_zero", "    unplanned = max(0, len(segs) - 1 - planned)",
     "    unplanned = 0"),
    ("W2_readings_always_ok", "    readings_ok = cd_last is not None and cd_lag is not None and cd_lag <= 60",
     "    readings_ok = True"),
    ("W3_chip_ignores_the_trainer_pid",
     '    chips = (chip(tr_alive and sup_alive and not done, "training", "finished" if done else "NOT RUNNING", warn=done)',
     '    chips = (chip(sup_alive and not done, "training", "finished" if done else "NOT RUNNING", warn=done)'),
]


def pytest_rc():
    r = subprocess.run([PY, "-m", "pytest", str(TEST), "-q", "--no-header", "-p", "no:cacheprovider"],
                       cwd=str(TC), capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    out = r.stdout + r.stderr
    failed = sorted({ln.split("::")[-1].split()[0] for ln in out.splitlines()
                     if ln.startswith("FAILED") and "::" in ln})
    return r.returncode, failed, out[-300:]


def main():
    if TC.exists():
        shutil.rmtree(TC)
    TC.mkdir()
    raw = subprocess.run(["git", "-c", "core.autocrlf=false", "archive", TIP, "taniteval"],
                         env=GIT, capture_output=True).stdout
    assert raw, "empty archive"
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        tf.extractall(TC)
    for rel, src in FILES.items():
        (TC / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, TC / rel)
    rc, failed, tail = pytest_rc()
    res = {"_what": "clean-tree run + mutation proof of the refcv6 Training Watch health verdicts",
           "_evidence_class": "MEASURED (ours), CPU", "tip": TIP,
           "clean_tree": {"rc": rc, "failed": failed}, "arms": []}
    print("clean tree:", rc, failed)
    if rc != 0:
        print(tail)
        return 3
    orig = BUILDER.read_bytes()
    md5 = hashlib.md5(orig).hexdigest()
    try:
        for name, old, new in ARMS:
            src = orig.decode("utf-8")
            hits = [i for i, ln in enumerate(src.split("\n")) if ln == old]
            assert len(hits) == 1, (name, len(hits))
            lines = src.split("\n")
            lines[hits[0]] = new
            BUILDER.write_bytes("\n".join(lines).encode("utf-8"))
            rc_m, failed_m, _ = pytest_rc()
            BUILDER.write_bytes(orig)
            red = rc_m != 0 and bool(failed_m)
            print(f"  {name:<34} RED={red} caught_by={failed_m}")
            res["arms"].append({"arm": name, "went_RED": red, "caught_by": failed_m})
    finally:
        BUILDER.write_bytes(orig)
    res["restored_ok"] = hashlib.md5(BUILDER.read_bytes()).hexdigest() == md5
    rc_f, failed_f, _ = pytest_rc()
    res["final_clean_run"] = {"rc": rc_f, "failed": failed_f}
    n = sum(a["went_RED"] for a in res["arms"])
    res["_VERDICT"] = (f"MUTATION-PROVEN -- {n}/{len(ARMS)} arms RED; restored; final run green."
                       if n == len(ARMS) and res["restored_ok"] and rc_f == 0 else f"ONLY {n}/{len(ARMS)}")
    (W / "mutation_proof_training_watch_refcv6.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(res["_VERDICT"])
    return 0 if n == len(ARMS) else 4


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
