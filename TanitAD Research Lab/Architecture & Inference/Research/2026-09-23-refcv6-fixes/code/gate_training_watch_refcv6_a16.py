"""Gate for the Training Watch stderr/token fix: the watch test on a CLEAN tree (tip + exactly the
changed files), then a mutation proof of every health verdict -- the three from commit I and six
new ones, including a VERBATIM restore of the historical whitespace-split token parser, which read
the live run's split token as 50,400 tracebacks. Writes
mutation_proof_training_watch_refcv6_a16.json."""
import hashlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile

TIP = "121892142a5c6b1ad1890543d11537b281b844c1"
GIT = dict(os.environ, GIT_DIR="C:/Users/Admin/tanitad-push/.git")
TC = pathlib.Path("C:/Users/Admin/tcK")
W = pathlib.Path("C:/Users/Admin/qland/work/watch6")
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
FILES = {"taniteval/tools/training_watch/build_watch_refcv6.py": W / "build_watch_refcv6.py",
         "taniteval/tests/test_training_watch_refcv6.py": W / "test_training_watch_refcv6.py"}
BUILDER = TC / "taniteval/tools/training_watch/build_watch_refcv6.py"
TEST = TC / "taniteval/tests/test_training_watch_refcv6.py"

NEW_PARSE = '''    toks = list(TOKEN_RE.finditer(sup_log))
    token = toks[-1].group(0) if toks else None
    n_err = launch_no = None
    token_split = False
    field = ""
    if toks:
        field = toks[-1].group(3)            # "U" = the supervisor could not READ its stderr
        token_split = any(c.isspace() for c in field)
        ints = re.findall(r"\\d+", field)
        n_err = int(ints[0]) if ints else None   # grep -c's own count; `|| echo 0` only APPENDS a 0
        launch_no = int(toks[-1].group(4))
'''
# the parser as landed in fe5872f, restored verbatim (only its variable renamed so the rest of
# the function still runs): the log split on whitespace, the last half-token's last two fields
HISTORICAL_PARSE = '''    _tk = [t for t in sup_log.split() if t.startswith("ZZrefcv6-r101-s0-")]
    token = _tk[-1] if _tk else None
    n_err = launch_no = None
    token_split = False
    field = ""
    if _tk:
        parts = _tk[-1].strip("Z").split("-")
        try:
            n_err, launch_no = int(parts[-2]), int(parts[-1])
        except (ValueError, IndexError):
            pass
'''
ARMS = [
    ("W1_unplanned_pinned_to_zero", "    unplanned = max(0, len(segs) - 1 - planned)\n",
     "    unplanned = 0\n"),
    ("W2_readings_always_ok", "    readings_ok = cd_last is not None and cd_lag is not None and cd_lag <= 60\n",
     "    readings_ok = True\n"),
    ("W3_chip_ignores_the_trainer_pid",
     '    chips = (chip(tr_alive and sup_alive and not done, "training", "finished" if done else "NOT RUNNING", warn=done)\n',
     '    chips = (chip(sup_alive and not done, "training", "finished" if done else "NOT RUNNING", warn=done)\n'),
    ("W4_HISTORICAL_whitespace_split_token_parser", NEW_PARSE, HISTORICAL_PARSE),
    ("W5_stderr_content_ignored", "    stderr_undiag = [ln for ln in stderr_lines if ln not in diag]\n",
     "    stderr_undiag = []\n"),
    ("W6_stderr_read_check_dropped",
     '    stderr_read_ok = stderr_b >= 0 and len(stderr_txt.encode("utf-8")) >= min(stderr_b, STDERR_CAP)\n',
     "    stderr_read_ok = True\n"),
    ("W7_diagnosis_by_message_not_by_exact_line",
     "    stderr_undiag = [ln for ln in stderr_lines if ln not in diag]\n",
     '    stderr_undiag = [ln for ln in stderr_lines if not any(k.rsplit("] ", 1)[-1] in ln for k in diag)]\n'),
    ("W8_client_traceback_count_dropped",
     "    n_err_client = sum(1 for ln in stderr_lines if ERR_PAT.search(ln))\n",
     "    n_err_client = 0\n"),
    ("W9_unread_supervisor_count_read_as_zero",
     "        n_err = int(ints[0]) if ints else None   # grep -c's own count; `|| echo 0` only APPENDS a 0\n",
     "        n_err = int(ints[0]) if ints else 0\n"),
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
    res = {"_what": "clean-tree run + mutation proof of the refcv6 Training Watch health verdicts, "
                    "incl. the stderr-content and split-token checks added 2026-09-26",
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
    (W / "mutation_proof_training_watch_refcv6_a16.json").write_text(json.dumps(res, indent=1),
                                                                         encoding="utf-8")
    print(res["_VERDICT"])
    return 0 if n == len(ARMS) else 4


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
