"""⛔ THE SECOND HALF OF "PROVEN BY MUTATION": defeat each guard, watch its test go red.

``test_cot_negative_policy.py`` mutates the ARTIFACT and asserts the guard
fires. That proves the failure branch is reachable. It does NOT prove the test
is load-bearing: a test can pass because something ELSE happened to raise. So
this script mutates the GUARD instead -- it deletes the check, reruns only that
guard's test, and REQUIRES it to fail.

A guard whose test still passes with the guard removed is not a guard; it is a
comment with a docstring. The table it prints (guard, the line defeated, the
test, RED/GREEN-with-guard-removed) is the evidence that belongs beside the
mutation table in the package.

⛔ It restores the file in a ``finally`` and re-verifies the restored bytes by
md5 before exiting, because a half-applied patch left on disk would be a far
worse outcome than an unproven guard.

ASCII only in the printed output: this box is cp1252.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "tanitad" / "data" / "v7_labels.py"
TESTS = Path(__file__).resolve().parents[1] / "tests" / "test_cot_negative_policy.py"

#: guard -> (a UNIQUE snippet that IS the check, its replacement that defeats
#: it, the pytest -k selector for the test that must then fail)
DEFEATS = [
    ("blob-md5 binding",
     'if meta.get("source_blob_md5") != manifest.md5:',
     'if False:',
     "sidecar_built_over_another_blob"),
    ("schema tag",
     'if doc.get("schema") != COT_SIDECAR_SCHEMA:',
     'if False:',
     "schema_tag_changed"),
    ("policy id",
     "if sc.policy != COT_ABSENCE_POLICY_ID:",
     "if False:",
     "policy_id_changed"),
    ("digest-algorithm declaration",
     'if meta.get("digest_algorithm") != COT_SIDECAR_DIGEST_ALGO:',
     'if False:',
     "digest_algorithm_undeclared"),
    ("row width",
     "bad = {d: b for d, b in sc.by_digest.items() if len(b) != len(sc.tokens)}",
     "bad = {}",
     "row_not_token_wide"),
    ("empty policy",
     "if not sc.tokens or not sc.by_digest:",
     "if False:",
     "empty_policy"),
    ("clip coverage (blob -> sidecar)",
     "if missing:",
     "if False:",
     "clip_dropped_from_sidecar"),
    ("clip coverage (sidecar -> blob)",
     "if extra:",
     "if False:",
     "clip_not_in_blob"),
    ("positive agreement",
     "if n_disagree:",
     "if False:",
     "positive_bit_flipped"),
    ("new CoT token left undecided",
     "if new_tokens:",
     "if False:",
     "new_cot_token_left_undecided"),
    ("stale sidecar token",
     "if gone_tokens:",
     "if False:",
     "sidecar_token_no_longer_cot_backed"),
    ("policy needs a sidecar",
     "if sidecar is None:\n            raise CotAbsenceNegativeRefused(",
     "if False:\n            raise CotAbsenceNegativeRefused(",
     "test_policy_without_sidecar_is_refused"),
    ("sidecar needs the policy",
     "elif sidecar is not None:\n        raise CotAbsenceNegativeRefused(",
     "elif False:\n        raise CotAbsenceNegativeRefused(",
     "test_sidecar_without_policy_is_refused"),
    ("per-clip coverage at target time",
     "if not sidecar.covers(label.clip_id):",
     "if False:",
     "test_uncovered_clip_is_refused_not_defaulted"),
]


def run(sel: str) -> tuple[bool, str]:
    p = subprocess.run([sys.executable, "-m", "pytest", str(TESTS), "-q",
                        "-k", sel, "-x", "--no-header"],
                       capture_output=True, text=True)
    return p.returncode == 0, (p.stdout or "")[-400:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    original = TARGET.read_bytes()
    md5 = hashlib.md5(original).hexdigest()
    src = original.decode("utf-8")
    rows = []
    try:
        ok, tail = run("test_control_unmutated_passes")
        if not ok:
            raise SystemExit(f"[audit] the CONTROL is already red; every "
                             f"'guard removed -> test fails' below would be "
                             f"meaningless.\n{tail}")
        print("[audit] control green: the clean artifact loads and validates")
        for name, old, new, sel in DEFEATS:
            n = src.count(old)
            if n != 1:
                rows.append({"guard": name, "test": sel,
                             "verdict": "NOT AUDITED",
                             "reason": f"the snippet matches {n} times, so the "
                                       f"defeat is not uniquely placeable"})
                print(f"[audit] {name:38s} NOT AUDITED ({n} matches)")
                continue
            TARGET.write_text(src.replace(old, new), encoding="utf-8")
            passed, tail = run(sel)
            TARGET.write_bytes(original)
            rows.append({"guard": name, "test": sel,
                         "defeat": old.splitlines()[0],
                         "test_result_with_guard_removed":
                             "PASSED (guard is NOT load-bearing)" if passed
                             else "FAILED (guard is load-bearing)",
                         "verdict": "NOT LOAD-BEARING" if passed else "PROVEN"})
            print(f"[audit] {name:38s} "
                  f"{'*** NOT LOAD-BEARING ***' if passed else 'PROVEN'}")
    finally:
        TARGET.write_bytes(original)
        back = hashlib.md5(TARGET.read_bytes()).hexdigest()
        if back != md5:
            raise SystemExit(f"[audit] ⛔ RESTORE FAILED: {TARGET} is now {back}, "
                             f"was {md5}. Fix this before anything else.")
        print(f"[audit] restored {TARGET.name} md5={back}")

    bad = [r for r in rows if r["verdict"] != "PROVEN"]
    doc = {"_evidence_class": "MEASURED (ours)", "target": str(TARGET),
           "target_md5": md5, "n_guards": len(rows),
           "n_proven": sum(1 for r in rows if r["verdict"] == "PROVEN"),
           "rows": rows}
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print(f"[audit] wrote {a.out}")
    print(f"[audit] {doc['n_proven']}/{len(rows)} guards proven load-bearing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
