"""Gate for the Training Watch NavSim + battery section (PI 2026-09-26: "include the navsim pdms KPIs
for the last checkpoints"): the watch test on a CLEAN tree (tip + exactly the changed files), then a
mutation proof of the new section -- a typed published reference, an ignored live lane, an unscaled
interval, an unconditional "no checkpoint beats STOP" sentence, a verdict chip that looks like a health
chip, and a wrong pre-switch stamp must each turn the suite RED. Writes
mutation_proof_training_watch_refcv6_navsim.json."""
import hashlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile

TIP = subprocess.run(["git", "--git-dir=C:/Users/Admin/tanitad-push/.git", "rev-parse", "agent/arch-inf-20260803"],
                     capture_output=True, text=True).stdout.strip()
GIT = dict(os.environ, GIT_DIR="C:/Users/Admin/tanitad-push/.git")
TC = pathlib.Path("C:/Users/Admin/tcL")
W = pathlib.Path("C:/Users/Admin/qland/work/watch6")
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
FILES = {"taniteval/tools/training_watch/build_watch_refcv6.py": W / "build_watch_refcv6.py",
         "taniteval/tests/test_training_watch_refcv6.py": W / "test_training_watch_refcv6.py"}
BUILDER = TC / "taniteval/tools/training_watch/build_watch_refcv6.py"
TEST = TC / "taniteval/tests/test_training_watch_refcv6.py"

ARMS = [
    ("published reference typed, not read",
     "            published = pub                          # the banked reference text, read, never typed",
     '            published = {"DiffusionDrive": "88.1 PDMS"}'),
    ("the live lane is ignored (running reads not run)",
     '                row[split] = {"status": "running" if started else "not run"}',
     '                row[split] = {"status": "not run"}'),
    ("the navtest interval is not scaled x100",
     '"lo": _num(iv.get("lo")) and iv["lo"] * 100, "hi": _num(iv.get("hi")) and iv["hi"] * 100,',
     '"lo": _num(iv.get("lo")), "hi": _num(iv.get("hi")),'),
    ("the no-checkpoint-beats-STOP sentence is unconditional",
     'and ns[s][sp]["value"] < ns[s][sp]["STOP"]',
     'and True'),
    ("a FAIL verdict chip that looks like a health chip",
     """'<span class="chip crit verdict"><i></i>bar failed</span>'""",
     """'<span class="chip crit"><i></i>bar failed</span>'"""),
    ("every checkpoint stamped post-switch",
     "if step <= SWITCH_STEP else",
     "if step < 0 else"),
]


def pytest_rc():
    r = subprocess.run([PY, "-m", "pytest", str(TEST), "-q", "--no-header", "-p", "no:cacheprovider"],
                       cwd=str(TC), capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    out = r.stdout + r.stderr
    failed = sorted({ln.split("::")[-1].split()[0] for ln in out.splitlines()
                     if ln.startswith("FAILED") and "::" in ln})
    passed = [ln for ln in out.splitlines() if " passed" in ln]
    return r.returncode, failed, (passed[-1] if passed else out[-300:])


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
    res = {"_what": "clean-tree run + mutation proof of the refcv6 Training Watch NavSim + battery section "
                    "(added 2026-09-26)",
           "_evidence_class": "MEASURED (ours), CPU", "tip": TIP,
           "clean_tree": {"rc": rc, "failed": failed, "summary": tail}, "arms": []}
    print("clean tree:", rc, failed, tail)
    if rc != 0:
        return 3
    orig = BUILDER.read_bytes()
    md5 = hashlib.md5(orig).hexdigest()
    src0 = orig.decode("utf-8")
    try:
        for name, old, new in ARMS:
            n = src0.count(old)
            assert n == 1, (name, n)
            BUILDER.write_bytes(src0.replace(old, new).encode("utf-8"))
            rc_m, failed_m, _ = pytest_rc()
            BUILDER.write_bytes(orig)
            red = rc_m != 0 and bool(failed_m)
            print(f"  {name:<46} RED={red} caught_by={failed_m}")
            res["arms"].append({"arm": name, "went_RED": red, "caught_by": failed_m})
    finally:
        BUILDER.write_bytes(orig)
    res["restored_ok"] = hashlib.md5(BUILDER.read_bytes()).hexdigest() == md5
    rc_f, failed_f, tail_f = pytest_rc()
    res["final_clean_run"] = {"rc": rc_f, "failed": failed_f, "summary": tail_f}
    n = sum(a["went_RED"] for a in res["arms"])
    res["_VERDICT"] = (f"MUTATION-PROVEN -- {n}/{len(ARMS)} arms RED; restored; final run green."
                       if n == len(ARMS) and res["restored_ok"] and rc_f == 0 else f"ONLY {n}/{len(ARMS)}")
    (W / "mutation_proof_training_watch_refcv6_navsim.json").write_text(json.dumps(res, indent=1),
                                                                         encoding="utf-8")
    print(res["_VERDICT"])
    return 0 if n == len(ARMS) else 4


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
