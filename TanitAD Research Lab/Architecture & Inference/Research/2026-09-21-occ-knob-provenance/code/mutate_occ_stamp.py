"""MUTATION PROOF for `test_occ_knob_is_stamped.py` — the provenance guard must be able to FAIL.

⛔ The defect being guarded is one I CREATED: `6a472d1` added a behaviour knob with no config field
and no stamp entry. The fix (config + stamp + plumbing) is only worth the bytes if a guard can tell
when it regresses, and inspection cannot establish that.

Arms, each a defect that really happens in this tree:

  S1  the stamp entry is dropped from the AGENT seam          (the original defect, restored)
  S2  the stamp entry is dropped from the PERCEPTION branch   (same, on the scored head)
  S3  the stamp HARDCODES the default instead of the value    (key present, value a lie)
  S4  the AGENT builder stops applying the field              (declared, never plumbed)
  S5  the PERCEPTION branch stops applying the field          (declared, never plumbed — and this
                                                               is the arm the FIRST version of the
                                                               test could not catch, because it
                                                               only read `as_dict`)

⛔ Two rules carried over from `mutate_occ_geometry.py`, both paid for:
  * the child's output is decoded as **utf-8**, never `text=True` — that decodes with the PARENT's
    cp1252 locale, and a traceback containing one non-cp1252 byte returns EMPTY streams with rc 1,
    which the prover then scores as "not caught". The arms most likely to trip it are the WORKING
    ones, because their failure output is richest.
  * an arm counts only if a **NAMED** test failed. Red-by-syntax-error exercises no assertion.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path("D:/Projects/TanitAD")
SEAM = REPO / "stack" / "tanitad" / "refs" / "refc_agents.py"
BRANCH = REPO / "stack" / "tanitad" / "models" / "refcv6_perception_branch.py"
TESTS = REPO / "stack" / "tests" / "test_occ_knob_is_stamped.py"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"

# ⛔ ANCHORS CARRY NO LINE TERMINATOR. These files are CRLF, so an anchor ending in a newline
# matches NOTHING — the first version of this prover silently skipped FOUR of five arms and then
# printed "ONLY 1/5 CAUGHT", a verdict about the TESTS manufactured by a defect in the PROVER.
# Replacements are syntactically neutral (`pass`, or an emptied dict slot) so a dropped line cannot
# masquerade as a syntax error, which would exercise no assertion either.
MUTATIONS = [
    ("S1_agent_stamp_entry_dropped", SEAM,
     '"occ_from_geometry": bool(self.occ_from_geometry),', ""),
    ("S2_perception_stamp_entry_dropped", BRANCH,
     '"occ_from_geometry": bool(self.occ_from_geometry),', ""),
    ("S3_agent_stamp_hardcodes_the_default", SEAM,
     '"occ_from_geometry": bool(self.occ_from_geometry),',
     '"occ_from_geometry": False,'),
    ("S4_agent_builder_stops_plumbing_it", SEAM,
     "head.occ_from_geometry = bool(cfg.occ_from_geometry)", "pass"),
    ("S5_perception_stops_plumbing_it", BRANCH,
     "self.box_dec.occ_from_geometry = bool(cfg.occ_from_geometry)", "pass"),
]


def run_tests():
    r = subprocess.run(
        [PY, "-m", "pytest", str(TESTS), "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=str(REPO), capture_output=True, encoding="utf-8", errors="replace",
        env={"PYTHONPATH": str(REPO / "stack"), "PATH": os.environ["PATH"],
             "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "6"})
    out = (r.stdout or "") + (r.stderr or "")
    failed, seen = [], set()
    for ln in out.splitlines():
        s = ln.strip()
        if s.startswith("FAILED") and "::" in s:
            nm = s.split("::")[-1].split()[0].strip()
            if nm not in seen:
                seen.add(nm)
                failed.append(nm)
    return r.returncode, failed, (not out.strip()), out[-400:]


def main() -> int:
    orig = {p: p.read_bytes() for p in (SEAM, BRANCH)}
    md5 = {p: hashlib.md5(b).hexdigest() for p, b in orig.items()}
    res = {"_what": "mutation proof that test_occ_knob_is_stamped.py can actually FAIL",
           "_evidence_class": "MEASURED (ours), CPU",
           "targets": {str(p.relative_to(REPO)): m for p, m in md5.items()},
           "arms": []}
    try:
        rc, failed, dead, _ = run_tests()
        res["baseline"] = {"rc": rc, "failed": failed}
        print(f"BASELINE rc={rc} failed={failed}")
        if rc != 0:
            print("ZZABORT baseline not green — a mutation proof on a red suite proves nothing")
            return 3
        for name, target, old, new in MUTATIONS:
            txt = orig[target].decode("utf-8")
            if txt.count(old) != 1:
                # ⛔ AN ARM THAT NEVER APPLIED IS NOT A FAILED ARM — IT IS NO ARM AT ALL,
                # and folding it into an n/total ratio turns a PROVER defect into a verdict about
                # the tests. Same family as scoring an empty output as "not caught". Abort.
                print(f"ZZABORT {name}: anchor appears {txt.count(old)} times in "
                      f"{target.name}, need exactly 1 — the proof is INVALID, not weak")
                res["arms"].append({"arm": name, "error": "anchor not unique",
                                    "occurrences": txt.count(old)})
                res["_VERDICT"] = ("⛔ INVALID — an arm could not be applied, so this run "
                                   "says nothing about the tests either way.")
                print(json.dumps(res, indent=1, ensure_ascii=False))
                return 5
            try:
                target.write_bytes(txt.replace(old, new).encode("utf-8"))
                rc_m, failed_m, dead_m, tail_m = run_tests()
            finally:
                target.write_bytes(orig[target])
            caught = bool(rc_m != 0 and failed_m and not dead_m)
            print(f"  {name:<40} rc={rc_m} RED={caught} caught_by={failed_m[:3]}")
            res["arms"].append({"arm": name, "file": str(target.relative_to(REPO)),
                                "rc": rc_m, "went_RED": caught, "caught_by": failed_m,
                                "empty_output": dead_m,
                                "tail": None if caught else tail_m})
    finally:
        for p, b in orig.items():
            p.write_bytes(b)

    back = {p: hashlib.md5(p.read_bytes()).hexdigest() for p in orig}
    res["restored_ok"] = all(back[p] == md5[p] for p in orig)
    assert res["restored_ok"], f"ZZABORT targets NOT restored: {back} != {md5}"
    rc_f, failed_f, _, _ = run_tests()
    res["final_clean_run"] = {"rc": rc_f, "failed": failed_f}
    n = sum(1 for a in res["arms"] if a.get("went_RED"))
    res["arms_caught"], res["arms_total"] = n, len(MUTATIONS)
    res["_VERDICT"] = (
        f"⭐ MUTATION-PROVEN — all {n}/{len(MUTATIONS)} provenance defects turn the suite RED, "
        "including both 'declared but never plumbed' arms, and both files are restored "
        "byte-identical."
        if n == len(MUTATIONS) and res["restored_ok"] and rc_f == 0 else
        f"⛔ ONLY {n}/{len(MUTATIONS)} CAUGHT — the uncaught arms are exactly the provenance "
        "mistakes this guard would license.")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print(res["_VERDICT"])
    pathlib.Path("C:/Users/Admin/qland/work/pbox/mutation_proof_occ_stamp.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0 if n == len(MUTATIONS) else 4


if __name__ == "__main__":
    sys.exit(main())
