"""MUTATION PROOF for `test_occ_from_geometry.py` — reintroduce each real defect, prove RED.

⛔ A CHECK THAT SHARES THE DEFECT IT CHECKS FOR IS GREEN FOREVER, and inspection cannot tell the
two apart. MEASURED precedent in this programme: an AST census read 0 suspects on BOTH the fixed
and the broken trainer. So this file does not read the tests — it BREAKS the code they guard, one
defect at a time, and requires the suite to fail each time.

Each arm is a defect that could really happen, not a synthetic typo:

  M1  the half-angle drifts out of sync with `bev_raster.fov_mask`   (the gate-value 5-site class)
  M2  the logit's SIGN is inverted                                   (occluded/visible swapped)
  M3  the `abs()` on azimuth is dropped                              (breaks only one side — the
                                                                      asymmetric half of the bug)
  M4  the decode reads the RAW slice instead of the DECODED centre   (correct formula, wrong UNITS
                                                                      — the 396 g anchor family)
  M5  the opt-in default is flipped ON                               (a silent contract change)

⛔ SAFETY: the target file is backed up by BYTES before anything runs, every arm restores it in a
`finally`, and the run ends by asserting the restored file's md5 equals the original's. A mutation
prover that leaves the tree mutated is worse than no prover.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path("D:/Projects/TanitAD")
TARGET = REPO / "stack" / "tanitad" / "models" / "agent_slots.py"
TESTS = REPO / "stack" / "tests" / "test_occ_from_geometry.py"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"

MUTATIONS = [
    ("M1_half_angle_drifts_from_fov_mask",
     "OCC_HALF_ANGLE_RAD: float = math.radians(60.0)",
     "OCC_HALF_ANGLE_RAD: float = math.radians(50.0)"),
    ("M2_logit_sign_inverted",
     "    return float(temperature) * (az - float(half_angle_rad))",
     "    return -float(temperature) * (az - float(half_angle_rad))"),
    ("M3_abs_dropped_from_azimuth",
     "    az = torch.atan2(cy, cx).abs()",
     "    az = torch.atan2(cy, cx)"),
    ("M4_reads_raw_slice_not_decoded_centre",
     '            "occ_logit": (occ_logit_from_centre(cx, cy)',
     '            "occ_logit": (occ_logit_from_centre(\n'
     '                raw[..., s["cx"]].squeeze(-1), raw[..., s["cy"]].squeeze(-1))'),
    ("M5_opt_in_silently_defaults_on",
     "        self.occ_from_geometry: bool = False",
     "        self.occ_from_geometry: bool = True"),
]


def run_tests() -> tuple[int, list[str]]:
    r = subprocess.run(
        [PY, "-m", "pytest", str(TESTS), "-q", "--no-header", "-p", "no:cacheprovider"],
        # ⛔ NEVER `text=True` HERE. It decodes the CHILD's output with the PARENT's locale,
        # which is cp1252 on this box. MEASURED 2026-09-21: arm M4's pytest traceback echoes the
        # failing test's docstring, which carries a non-cp1252 byte -- the reader thread raised
        # UnicodeDecodeError, BOTH stdout and stderr came back EMPTY, and the prover recorded the
        # arm as NOT CAUGHT. ⚠️ The arms most likely to trip it are the ones with the RICHEST
        # failure output, i.e. the ones that are WORKING. An empty read is a claim about the PIPE,
        # never about the tests -- same family as `grep` reporting 0 hits for a file it could not
        # open. Hand-run, M4 fails 3 named assertions.
        cwd=str(REPO), capture_output=True,
        encoding="utf-8", errors="replace",
        env={"PYTHONPATH": str(REPO / "stack"), "PATH": __import__("os").environ["PATH"],
             "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "6"})
    out = (r.stdout or "") + (r.stderr or "")
    # ⚠️ MEASURED: the first version of this parser reported ZERO catchers for M4 while the
    # SAME mutation, run by hand, failed 3 assertions by name. An empty catcher list is
    # indistinguishable from "the suite died before running" -- i.e. from a mutation proof that
    # proves nothing -- so the parse is now belt-and-braces and `collected` is asserted below.
    failed, seen = [], set()
    for ln in out.splitlines():
        s_ = ln.strip()
        if "::" in s_ and ("FAILED" in s_ or s_.startswith("E ") is False and s_.startswith("FAILED")):
            if "FAILED" not in s_:
                continue
            nm = s_.split("::")[-1].split()[0].strip()
            if nm and nm not in seen:
                seen.add(nm); failed.append(nm)
    collapsed = "error" in out.lower() and "collected 0 items" in out.lower()
    # ⛔ An arm that produced NO OUTPUT AT ALL has told us nothing; it must not be scored either
    # way. Distinguishing "died" from "passed silently" is exactly what the empty read destroys.
    if not out.strip():
        collapsed = True
    return r.returncode, failed, bool(collapsed), out[-400:]


def main() -> int:
    orig = TARGET.read_bytes()
    md5 = hashlib.md5(orig).hexdigest()
    backup = TARGET.with_suffix(".py.mutation-backup")
    backup.write_bytes(orig)
    print(f"target {TARGET.name}  md5 {md5}  backup {backup.name}")

    res = {"_what": "mutation proof that test_occ_from_geometry.py can actually FAIL",
           "_evidence_class": "MEASURED (ours), CPU",
           "target": str(TARGET.relative_to(REPO)), "target_md5": md5, "arms": []}
    try:
        rc, failed, dead, _ = run_tests()
        print(f"BASELINE rc={rc} failed={failed}")
        res["baseline"] = {"rc": rc, "failed": failed}
        if rc != 0:
            print("ZZABORT baseline is not green; a mutation proof on a red suite proves nothing")
            return 3

        for name, old, new in MUTATIONS:
            txt = orig.decode("utf-8")
            if txt.count(old) != 1:
                print(f"ZZABORT {name}: anchor appears {txt.count(old)} times, need exactly 1")
                res["arms"].append({"arm": name, "error": "anchor not unique"})
                continue
            try:
                TARGET.write_bytes(txt.replace(old, new).encode("utf-8"))
                rc_m, failed_m, dead_m, tail_m = run_tests()
            finally:
                TARGET.write_bytes(orig)
            # ⛔ RED BY SYNTAX ERROR IS NOT A PROOF. A mutation that stops the suite from
            # COLLECTING never exercises the assertions, so it cannot demonstrate they can fail.
            # An arm counts only if a NAMED test failed.
            caught = bool(rc_m != 0 and failed_m and not dead_m)
            print(f"  {name:<42} rc={rc_m}  RED={caught}  caught_by={failed_m[:4]}")
            res["arms"].append({"arm": name, "rc": rc_m, "went_RED": caught,
                                "caught_by": failed_m,
                                "collection_died": dead_m,
                                "_rule": "an arm counts only if a NAMED test failed -- red by "
                                         "syntax error exercises no assertion",
                                "tail": None if caught else tail_m})
    finally:
        TARGET.write_bytes(orig)

    back = hashlib.md5(TARGET.read_bytes()).hexdigest()
    res["restored_md5"] = back
    res["restored_ok"] = bool(back == md5)
    assert back == md5, f"ZZABORT the target was NOT restored: {back} != {md5}"
    backup.unlink(missing_ok=True)

    rc_f, failed_f, _, _ = run_tests()
    res["final_clean_run"] = {"rc": rc_f, "failed": failed_f}
    n = sum(1 for a in res["arms"] if a.get("went_RED"))
    res["arms_caught"] = n
    res["arms_total"] = len(MUTATIONS)
    res["_VERDICT"] = (
        f"⭐ MUTATION-PROVEN — all {n}/{len(MUTATIONS)} reintroduced defects turn the suite RED, "
        "and the file is restored byte-identical."
        if n == len(MUTATIONS) and res["restored_ok"] and rc_f == 0 else
        f"⛔ ONLY {n}/{len(MUTATIONS)} DEFECTS ARE CAUGHT — the uncaught ones are exactly the "
        "mistakes this guard would license. Fix the test before trusting it.")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print(res["_VERDICT"])
    pathlib.Path("C:/Users/Admin/qland/work/pbox/mutation_proof_occ_geometry.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0 if n == len(MUTATIONS) else 4


if __name__ == "__main__":
    sys.exit(main())
