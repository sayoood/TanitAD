"""GUARD-REMOVAL AUDIT for `prelaunch_v2` — delete each refusal, require a test to go RED.

⛔ The house rule: a guard is proven by MUTATION, never by inspection. Asserting
that a check exists says nothing about whether removing it would be NOTICED. This
neuters one branch at a time and demands the suite fail each time.

Run:  python guard_removal_audit.py <path to prelaunch_v2.py> <path to its tests>
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: (name, regex on the source, replacement) — each removes ONE refusal.
REMOVALS = [
    ("R3-unregistered",
     r"    missing = sorted\(set\(claimed\) - known\)",
     "    missing = []                      # GUARD REMOVED"),
    ("R3-empty-claim",
     r"    if not claimed:",
     "    if False:                         # GUARD REMOVED"),
    ("R3-missing-registry",
     r"    if not p\.is_file\(\):",
     "    if False:                         # GUARD REMOVED"),
    ("R4-no-tuned-block",
     r"    if tuned is None:",
     "    if False:                         # GUARD REMOVED"),
    ("R4-fitted-on-scored",
     r"    if leaked:",
     "    if False:                         # GUARD REMOVED"),
    ("R4-undeclared",
     r"    if undeclared:",
     "    if False:                         # GUARD REMOVED"),
    ("SPLIT-empty-scored",
     r"    if not test:",
     "    if False:                         # GUARD REMOVED"),
    ("SPLIT-overlap",
     r"        if shared:",
     "        if False:                     # GUARD REMOVED"),
]


#: Same idea for the REPORT-time refusals (2 and 9).
REMOVALS_PANEL = [
    ("R2-no-controls",
     r"    if not controls:",
     "    if False:                         # GUARD REMOVED"),
    ("R2-undeclared-expectation",
     r"    if undeclared:",
     "    if False:                         # GUARD REMOVED"),
    ("R2-not-exact",
     r"    if wrong:",
     "    if False:                         # GUARD REMOVED"),
    ("R9-pooled",
     r"    if pooled:",
     "    if False:                         # GUARD REMOVED"),
    ("R9-no-per-token",
     r"    if not per_token:",
     "    if False:                         # GUARD REMOVED"),
    ("R9-missing-n",
     r"    if missing_n:",
     "    if False:                         # GUARD REMOVED"),
    ("R9-headline-under-floor",
     r"    if under:",
     "    if False:                         # GUARD REMOVED"),
    ("R12-no-policy-named",
     r"    if pol is None:",
     "    if False:                         # GUARD REMOVED"),
    ("R12-unknown-policy-name",
     r"    if pol not in NEGATIVES_POLICIES:",
     "    if False:                         # GUARD REMOVED"),
    ("R12-counts-contradict-policy",
     r"    if mismatched:",
     "    if False:                         # GUARD REMOVED"),
]

#: ⛔ Keyed by module stem so the audit cannot be pointed at one module while
#: silently using another's removal list -- which would report PATTERN-MISSED for
#: every branch and still print a tidy summary.
#: §12 refusal 1, both halves. The last two are the ones E5 added.
REMOVALS_ONEVAR = [
    ("R1-lever-absent",
     r"    if lever not in nd:",
     "    if False:                         # GUARD REMOVED"),
    ("R1-lever-wrong-way",
     r"    if \(got_from, got_to\) != \(expected_from, expected_to\):",
     "    if False:                         # GUARD REMOVED"),
    ("R1-second-argv-lever",
     r"    if extra:",
     "    if False:                         # GUARD REMOVED"),
    ("R1-BUILT-configs-identical",
     r"    if not cd:",
     "    if False:                         # GUARD REMOVED"),
    ("R1-BUILT-undeclared-diff",
     r"    if unexplained:",
     "    if False:                         # GUARD REMOVED"),
]

BY_MODULE = {"prelaunch_v2": REMOVALS, "panel_refusals": REMOVALS_PANEL,
             "one_variable_v2": REMOVALS_ONEVAR}


def main(mod_path: str, test_path: str) -> int:
    mod, test = Path(mod_path), Path(test_path)
    removals = BY_MODULE.get(mod.stem)
    if removals is None:
        raise SystemExit(f"no removal list for {mod.stem!r}; known: "
                         f"{sorted(BY_MODULE)}")
    original = mod.read_text(encoding="utf-8")
    root = mod.parents[2]                      # .../stack (train -> tanitad -> stack)
    env_path = f"{root};{root.parent};{root.parent / 'taniteval'}"
    results, escaped = [], []
    try:
        for name, pat, repl in removals:
            new, n = re.subn(pat, repl, original, count=1)
            if n != 1:
                results.append({"guard": name, "verdict": "PATTERN-MISSED"})
                escaped.append(name)
                continue
            mod.write_text(new, encoding="utf-8")
            r = subprocess.run(
                [sys.executable, "-m", "pytest", str(test), "-q", "--tb=no"],
                cwd=str(root), capture_output=True, text=True,
                env={**__import__("os").environ, "PYTHONPATH": env_path,
                     "CUDA_VISIBLE_DEVICES": "", "PYTHONIOENCODING": "utf-8"})
            # ⛔ returncode 2 is a COLLECTION error, not a caught defect. Counting
            # it as a kill would let a guard-removal that merely breaks the import
            # masquerade as a guard that bites -- the audit would then prove nothing.
            killed = r.returncode == 1
            if r.returncode not in (0, 1):
                results.append({"guard": name, "verdict": "BROKEN-RUN",
                                "returncode": r.returncode,
                                "stderr_tail": r.stdout[-400:]})
                escaped.append(name)
                print(f'  {name:22s} -> BROKEN-RUN rc={r.returncode}')
                continue
            results.append({"guard": name,
                            "verdict": "KILLED" if killed else "ESCAPED",
                            "returncode": r.returncode})
            if not killed:
                escaped.append(name)
            print(f"  {name:22s} -> {'KILLED' if killed else 'ESCAPED ⛔'}")
    finally:
        mod.write_text(original, encoding="utf-8")

    # the baseline must be GREEN with everything restored, or the audit proves nothing
    r = subprocess.run([sys.executable, "-m", "pytest", str(test), "-q", "--tb=no"],
                       cwd=str(root), capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": env_path,
                            "CUDA_VISIBLE_DEVICES": "", "PYTHONIOENCODING": "utf-8"})
    out = {"n_guards": len(removals),
           "killed": sum(1 for x in results if x["verdict"] == "KILLED"),
           "escaped": escaped,
           "baseline_restored_green": r.returncode == 0,
           "results": results}
    print(json.dumps(out, indent=1))
    ok = not escaped and out["baseline_restored_green"]
    print("AUDIT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
