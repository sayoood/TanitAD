"""MUTATION PROOF for `test_conflict_controls_defer_on_zero_init.py`. Anchors are WHOLE LINES.

  F1  no deferral: an exactly-zero plan gradient fails the controls again (the refusal Thor hit)
  F2  no bound: a detached planner is deferred forever
  F3  NaN deferred too: a non-finite gradient stops failing
  F4  the trainer measures on a deferred step
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
TESTF = REPO / "stack" / "tests" / "test_conflict_controls_defer_on_zero_init.py"
TESTS = REPO / "stack" / "tests" / "test_conflict_controls_defer_on_zero_init.py"
GC = REPO / "stack" / "tanitad" / "train" / "grad_conflict.py"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"

# ⛔ Built with chr() ON PURPOSE. Writing this constant as a backslash escape inside a generated
# script is the trap that has now bitten three times in this session: a shell heredoc consumes one
# level of backslash, so the escape arrives as a REAL newline and the file dies at PARSE time with
# "unterminated string literal". The lesson in memory is "use the Edit tool, not a heredoc"; this
# is the belt to that suspenders — the value cannot be mangled because there is no escape to eat.
EOL_CHARS = chr(13) + chr(10)

# ⛔ ANCHORS ARE WHOLE LINES, MATCHED BY EQUALITY OVER LINES — not substrings.
# TWO traps are closed by that choice, and I hit BOTH on the way here:
#   (a) CRLF: an anchor carrying a line terminator matches nothing in these files, and 4 of 5
#       arms once skipped silently while the prover still printed a ratio;
#   (b) INDENTATION AS A SUBSTRING: the two loss call sites differ only by indent, so the
#       12-space anchor is a proper substring of the 20-space line and matched BOTH.
# Full-line equality is immune to both, and the uniqueness check still aborts the run.
MUTATIONS = [
    ("F1_no_deferral", GC,
     '        if st_self["norm_traj"] == 0.0:',
     '        if False:'),
    ("F2_no_bound_on_the_deferral", GC,
     '            if self._n_deferred > int(self.cfg.max_deferred_steps):',
     '            if False:'),
    ("F3_nan_deferred_too", GC,
     '        if st_self["norm_traj"] == 0.0:',
     '        if not (st_self["norm_traj"] > 0.0):'),
    ("F4_trainer_measures_on_a_deferred_step", TRAIN,
     '                if (_cd_checked and step % _cd_cfg.every == 0',
     '                if (True and step % _cd_cfg.every == 0'),
]


def run_tests():
    r = subprocess.run(
        [PY, "-m", "pytest", str(TESTS), "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=str(REPO), capture_output=True, encoding="utf-8", errors="replace",
        env={**{k: v for k, v in os.environ.items()
                if k in ("USERNAME", "USERPROFILE", "HOME", "HOMEDRIVE",
                         "HOMEPATH", "TEMP", "TMP", "SYSTEMROOT", "COMSPEC")},
             "PYTHONPATH": str(REPO / "stack"), "PATH": os.environ["PATH"],
             "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "2"})
    out = (r.stdout or "") + (r.stderr or "")
    failed, seen = [], set()
    for ln in out.splitlines():
        s = ln.strip()
        if s.startswith("FAILED") and "::" in s:
            nm = s.split("::")[-1].split()[0].strip()
            if nm not in seen:
                seen.add(nm)
                failed.append(nm)
    return r.returncode, failed, (not out.strip()), out[-500:]


def main() -> int:
    files = {TRAIN, TESTF, GC}
    orig = {p: p.read_bytes() for p in files}
    md5 = {p: hashlib.md5(b).hexdigest() for p, b in orig.items()}
    res = {"_what": "mutation proof that test_conflict_controls_defer_on_zero_init.py can FAIL",
           "_evidence_class": "MEASURED (ours), CPU",
           "targets": {str(p.relative_to(REPO)): m for p, m in md5.items()}, "arms": []}
    try:
        rc, failed, dead, tail = run_tests()
        res["baseline"] = {"rc": rc, "failed": failed}
        print(f"BASELINE rc={rc} failed={failed}")
        if dead:
            print("ZZABORT baseline produced NO OUTPUT — INCONCLUSIVE, not green")
            return 6
        if rc != 0:
            print("ZZABORT baseline not green\n" + tail)
            return 3
        for name, target, old, new in MUTATIONS:
            txt = orig[target].decode("utf-8")
            lines = txt.splitlines(keepends=True)
            hits = [i for i, ln in enumerate(lines) if ln.rstrip(EOL_CHARS) == old]
            if len(hits) != 1:
                print(f"ZZABORT {name}: anchor matches {len(hits)} LINES in "
                      f"{target.name} — the proof is INVALID, not weak")
                res["_VERDICT"] = "⛔ INVALID — an arm could not be applied."
                res["arms"].append({"arm": name, "error": "anchor not unique",
                                    "occurrences": len(hits)})
                pathlib.Path("C:/Users/Admin/qland/work/pbox/mutation_proof_conflict_defer.json"
                             ).write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                          encoding="utf-8")
                print(json.dumps(res, indent=1, ensure_ascii=False))
                return 5
            try:
                eol = lines[hits[0]][len(lines[hits[0]].rstrip(EOL_CHARS)):]
                lines[hits[0]] = new + eol
                target.write_bytes("".join(lines).encode("utf-8"))
                rc_m, failed_m, dead_m, tail_m = run_tests()
            finally:
                target.write_bytes(orig[target])
            caught = bool(rc_m != 0 and failed_m and not dead_m)
            print(f"  {name:<44} rc={rc_m} RED={caught} caught_by={failed_m[:3]}")
            res["arms"].append({"arm": name, "file": str(target.relative_to(REPO)),
                                "rc": rc_m, "went_RED": caught, "caught_by": failed_m,
                                "tail": None if caught else tail_m})
    finally:
        for p, b in orig.items():
            p.write_bytes(b)
    back = {p: hashlib.md5(p.read_bytes()).hexdigest() for p in files}
    res["restored_ok"] = all(back[p] == md5[p] for p in files)
    assert res["restored_ok"], "ZZABORT targets NOT restored"
    rc_f, failed_f, _, _ = run_tests()
    res["final_clean_run"] = {"rc": rc_f, "failed": failed_f}
    n = sum(1 for a in res["arms"] if a.get("went_RED"))
    res["arms_caught"], res["arms_total"] = n, len(MUTATIONS)
    res["_VERDICT"] = (
        f"⭐ MUTATION-PROVEN — all {n}/{len(MUTATIONS)} arms RED, including the "
        "the step-1 refusal Thor hit; all files restored "
        "byte-identical and the final clean run is green."
        if n == len(MUTATIONS) and res["restored_ok"] and rc_f == 0 else
        f"⛔ ONLY {n}/{len(MUTATIONS)} CAUGHT.")
    pathlib.Path("C:/Users/Admin/qland/work/pbox/mutation_proof_conflict_defer.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print(res["_VERDICT"])
    return 0 if n == len(MUTATIONS) else 4


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
