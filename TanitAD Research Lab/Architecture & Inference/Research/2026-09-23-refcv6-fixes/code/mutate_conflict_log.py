"""MUTATION PROOF for `test_conflict_readings_are_logged.py`.

  M1  the own-row branch is removed (readings off the log cadence are discarded again)
  M2  the own row is written under the WRONG step
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path("D:/Projects/TanitAD")
TRAIN = REPO / "stack" / "scripts" / "refc_v3_train.py"
TESTS = REPO / "stack" / "tests" / "test_conflict_readings_are_logged.py"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
OUT = pathlib.Path("C:/Users/Admin/qland/work/f_dedup/mutation_proof_conflict_log.json")
EOL_CHARS = chr(13) + chr(10)
BS = chr(92)

MUTATIONS = [
    ("M1_own_row_branch_removed", TRAIN,
     '        elif _cd_row:',
     '        elif False:'),
    ("M2_own_row_under_the_wrong_step", TRAIN,
     '            log.write(json.dumps({"step": step, **_cd_row}) + "' + BS + 'n")',
     '            log.write(json.dumps({"step": step - 1, **_cd_row}) + "' + BS + 'n")'),
]


def run_tests():
    r = subprocess.run(
        [PY, "-m", "pytest", str(TESTS), "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=str(REPO), capture_output=True, encoding="utf-8", errors="replace",
        env={**{k: v for k, v in os.environ.items()
                if k in ("USERNAME", "USERPROFILE", "HOME", "HOMEDRIVE",
                         "HOMEPATH", "TEMP", "TMP", "SYSTEMROOT", "COMSPEC")},
             "PYTHONPATH": str(REPO / "stack"), "PATH": os.environ["PATH"],
             "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": ""})
    out = (r.stdout or "") + (r.stderr or "")
    failed = sorted({ln.strip().split("::")[-1].split()[0] for ln in out.splitlines()
                     if ln.strip().startswith("FAILED") and "::" in ln})
    return r.returncode, failed, (not out.strip()), out[-400:]


def main() -> int:
    orig = TRAIN.read_bytes()
    md5 = hashlib.md5(orig).hexdigest()
    res = {"_what": "mutation proof that test_conflict_readings_are_logged.py can FAIL",
           "_evidence_class": "MEASURED (ours), CPU",
           "targets": {str(TRAIN.relative_to(REPO)): md5,
                       str(TESTS.relative_to(REPO)): hashlib.md5(TESTS.read_bytes()).hexdigest()},
           "arms": []}
    try:
        rc, failed, dead, tail = run_tests()
        res["baseline"] = {"rc": rc, "failed": failed}
        if dead or rc != 0:
            print("ZZABORT baseline not green\n" + tail)
            return 3
        for name, target, old, new in MUTATIONS:
            lines = orig.decode("utf-8").splitlines(keepends=True)
            hits = [i for i, ln in enumerate(lines) if ln.rstrip(EOL_CHARS) == old]
            if len(hits) != 1:
                print(f"ZZABORT {name}: anchor matches {len(hits)} lines")
                return 5
            try:
                eol = lines[hits[0]][len(lines[hits[0]].rstrip(EOL_CHARS)):]
                lines[hits[0]] = new + eol
                target.write_bytes("".join(lines).encode("utf-8"))
                rc_m, failed_m, dead_m, tail_m = run_tests()
            finally:
                target.write_bytes(orig)
            caught = bool(rc_m != 0 and failed_m and not dead_m)
            print(f"  {name:<36} rc={rc_m} RED={caught} caught_by={failed_m}")
            res["arms"].append({"arm": name, "rc": rc_m, "went_RED": caught,
                                "caught_by": failed_m, "tail": None if caught else tail_m})
    finally:
        TRAIN.write_bytes(orig)
    res["restored_ok"] = hashlib.md5(TRAIN.read_bytes()).hexdigest() == md5
    rc_f, failed_f, _, _ = run_tests()
    res["final_clean_run"] = {"rc": rc_f, "failed": failed_f}
    n = sum(a["went_RED"] for a in res["arms"])
    res["arms_caught"], res["arms_total"] = n, len(MUTATIONS)
    res["_VERDICT"] = (f"MUTATION-PROVEN -- {n}/{len(MUTATIONS)} arms RED; restored; final run green."
                       if n == len(MUTATIONS) and res["restored_ok"] and rc_f == 0
                       else f"ONLY {n}/{len(MUTATIONS)} CAUGHT.")
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(res["_VERDICT"])
    return 0 if n == len(MUTATIONS) else 4


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
