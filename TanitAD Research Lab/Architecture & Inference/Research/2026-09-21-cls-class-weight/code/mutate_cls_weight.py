"""MUTATION PROOF for `test_cls_class_weight.py`. Anchors carry NO line terminator (CRLF files).

Arms:
  W1  the denominator stops following the weights   -> the SCALE control must fire
  W2  `weight=_cw` dropped from cross_entropy       -> the non-uniform literals must fire
  W3  box3d_set_loss stops forwarding               -> declared-not-plumbed, 3-D head
  W4  agent_losses stops forwarding                 -> declared-not-plumbed, 2-D seam

⛔ Carried over, each paid for: decode the child as utf-8 (never `text=True`, which uses the
PARENT's cp1252 and returns EMPTY streams on one stray byte); an arm counts only if a NAMED test
failed; and an arm whose anchor does not apply ABORTS the run as INVALID rather than entering a
ratio — an arm that never applied is no arm at all.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path("D:/Projects/TanitAD")
SLOTS = REPO / "stack" / "tanitad" / "models" / "agent_slots.py"
BOX3D = REPO / "stack" / "tanitad" / "models" / "box3d_head.py"
SEAM = REPO / "stack" / "tanitad" / "refs" / "refc_agents.py"
TESTS = REPO / "stack" / "tests" / "test_cls_class_weight.py"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"

MUTATIONS = [
    # ⛔ SINGLE-LINE ANCHORS ONLY. These files are CRLF, so any anchor spanning a line break
    # matches nothing — and the first run of this prover aborted on exactly that, which is the
    # abort doing its job rather than a nuisance.
    ("W1_denominator_stops_following_the_weights", SLOTS,
     "else float(_cw[ct[ok]].sum()))", "else int(ok.sum()))"),
    ("W2_weight_dropped_from_cross_entropy", SLOTS,
     "weight=_cw)", "weight=None)"),
    ("W3_box3d_stops_forwarding", BOX3D,
     "cls_class_weight=cls_class_weight)", "cls_class_weight=None)"),
    ("W4_agent_losses_stops_forwarding", SEAM,
     "cls_class_weight=cls_class_weight)", "cls_class_weight=None)"),
]


def run_tests():
    r = subprocess.run(
        [PY, "-m", "pytest", str(TESTS), "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=str(REPO), capture_output=True, encoding="utf-8", errors="replace",
        env={"PYTHONPATH": str(REPO / "stack"), "PATH": os.environ["PATH"],
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
    return r.returncode, failed, (not out.strip()), out[-400:]


def main() -> int:
    files = {SLOTS, BOX3D, SEAM}
    orig = {p: p.read_bytes() for p in files}
    md5 = {p: hashlib.md5(b).hexdigest() for p, b in orig.items()}
    res = {"_what": "mutation proof that test_cls_class_weight.py can actually FAIL",
           "_evidence_class": "MEASURED (ours), CPU",
           "targets": {str(p.relative_to(REPO)): m for p, m in md5.items()}, "arms": []}
    try:
        rc, failed, _, _ = run_tests()
        res["baseline"] = {"rc": rc, "failed": failed}
        print(f"BASELINE rc={rc} failed={failed}")
        if rc != 0:
            print("ZZABORT baseline not green")
            return 3
        for name, target, old, new in MUTATIONS:
            txt = orig[target].decode("utf-8")
            if txt.count(old) != 1:
                print(f"ZZABORT {name}: anchor appears {txt.count(old)} times in {target.name}"
                      f" — the proof is INVALID, not weak")
                res["_VERDICT"] = "⛔ INVALID — an arm could not be applied."
                res["arms"].append({"arm": name, "error": "anchor not unique",
                                    "occurrences": txt.count(old)})
                print(json.dumps(res, indent=1, ensure_ascii=False))
                return 5
            try:
                target.write_bytes(txt.replace(old, new).encode("utf-8"))
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
        f"⭐ MUTATION-PROVEN — all {n}/{len(MUTATIONS)} arms RED, including both "
        "declared-but-not-plumbed forwards, all three files restored byte-identical."
        if n == len(MUTATIONS) and res["restored_ok"] and rc_f == 0 else
        f"⛔ ONLY {n}/{len(MUTATIONS)} CAUGHT.")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print(res["_VERDICT"])
    pathlib.Path("C:/Users/Admin/qland/work/pbox/mutation_proof_cls_weight.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0 if n == len(MUTATIONS) else 4


if __name__ == "__main__":
    sys.exit(main())
