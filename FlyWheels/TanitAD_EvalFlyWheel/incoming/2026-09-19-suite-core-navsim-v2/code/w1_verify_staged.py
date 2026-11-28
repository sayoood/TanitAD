#!/usr/bin/env python3
"""W1 — verify that EXACTLY the W1 paths are staged, and that each staged blob IS the worktree blob.

⚠️ NAMESPACED ON PURPOSE (`w1_…`) and it lives in W1's package, not in the session scratchpad:
MEASURED 2026-09-20 by W2 — the scratchpad is SHARED between streams, W6 overwrote W2's
`verify_staged.py` with its own, and W2's next "verification" silently verified W6's 51 paths while
printing a sentence that was true of the wrong script. ⇒ every run of this file prints WHICH
package and WHICH path list it is checking, and refuses any path outside W1's owned trees, so a
swapped script cannot read as success.

    python w1_verify_staged.py [--json <out>]

Per path: the index blob (`git ls-files --stage`) against the worktree blob (`git hash-object`),
⛔ both asserted 40 chars first — an empty answer from either side is INCONCLUSIVE, never a match
(CLAUDE.md: the blob comparison's own hole).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PACKAGE = "EvalFlyWheel W1 — FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-suite-core-navsim-v2"
REPO = Path(__file__).resolve().parents[5]
PKG = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-suite-core-navsim-v2"
BENCH = "taniteval/taniteval/bench"
#: W1's OWNED trees — a path outside them is refused, whoever put it in the list
OWNED_PREFIXES = (f"{BENCH}/", "taniteval/tests/test_bench_suite_", f"{PKG}/",
                  "taniteval/results/bench/")
#: W3 owns navsim_v1.py and W6 owns nuscenes_ol.py inside bench/plugins/ (orchestrator, 2026-09-19)
NOT_MINE = (f"{BENCH}/plugins/navsim_v1.py", f"{BENCH}/plugins/nuscenes_ol.py")

PATHS = [
    f"{BENCH}/__init__.py", f"{BENCH}/__main__.py", f"{BENCH}/_legacy.py", f"{BENCH}/cli.py",
    f"{BENCH}/contract.py", f"{BENCH}/gpu_gap.py", f"{BENCH}/internal_t1.py", f"{BENCH}/report_hook.py",
    f"{BENCH}/schema_check.py", f"{BENCH}/submission.py",
    f"{BENCH}/schema/bench_run.schema.json", f"{BENCH}/schema/summary.schema.json",
    f"{BENCH}/plugins/__init__.py",
    f"{BENCH}/navsim/__init__.py", f"{BENCH}/navsim/artifacts.py", f"{BENCH}/navsim/benchmark.py",
    f"{BENCH}/navsim/bridge.py", f"{BENCH}/navsim/export.py", f"{BENCH}/navsim/model_arms.py",
    f"{BENCH}/navsim/plans.py", f"{BENCH}/navsim/profiles.py", f"{BENCH}/navsim/references.json",
    f"{BENCH}/navsim/scoring.py",
    f"{BENCH}/navsim/seams.py", f"{BENCH}/navsim/summarize.py",
    f"{BENCH}/navsim/devkit_side/navsim_win.py", f"{BENCH}/navsim/devkit_side/tanitad_seam_agent.py",
    f"{BENCH}/navsim/devkit_side/export_agent_inputs.py", f"{BENCH}/navsim/devkit_side/reaggregate.py",
    f"{BENCH}/navsim/devkit_side/PROVENANCE.json",
    "taniteval/tests/test_bench_suite_contract.py", "taniteval/tests/test_bench_suite_gpu_gap.py",
    "taniteval/tests/test_bench_suite_internal_t1.py", "taniteval/tests/test_bench_suite_legacy_compat.py",
    "taniteval/tests/test_bench_suite_navsim_offline.py", "taniteval/tests/test_bench_suite_promotion.py",
    "taniteval/tests/test_bench_suite_schema.py", "taniteval/tests/test_bench_suite_submission.py",
    f"{PKG}/SPEC.md", f"{PKG}/PLAN.md", f"{PKG}/RESULT.md", f"{PKG}/COMMS.md",
    f"{PKG}/code/run_acceptance.sh", f"{PKG}/code/clean_shell.py", f"{PKG}/code/verify_acceptance.py",
    f"{PKG}/code/w1_verify_staged.py", f"{PKG}/code/w1_repromote_navsim_win.py",
    f"{PKG}/code/w1_stage1_check.py",
    f"{PKG}/raw/SPEC_PREREG_HASH.txt", f"{PKG}/raw/legacy_importers_BEFORE.txt",
    f"{PKG}/raw/legacy_importers_AFTER.txt", f"{PKG}/raw/legacy_cli_help_BEFORE.txt",
    f"{PKG}/raw/w1_tests_final.txt", f"{PKG}/raw/acceptance_verdict.json", f"{PKG}/raw/acceptance_run.log",
    f"{PKG}/raw/internal_t1_smoke.log", f"{PKG}/raw/navhard_CV_stage1_check.json", f"{PKG}/raw/ram_guard_abort_CV_counts.json",
    # ⚠ SELF-REFERENCE: this file is THIS script's own output, so a run WITH --json checks it
    # BEFORE overwriting it and the row describes the PREVIOUS content. ⇒ the END-OF-TURN pass
    # must be run WITHOUT --json (nothing changes underneath it) AFTER the JSON has been staged.
    f"{PKG}/raw/w1_staged_verify.json",
]


def git(args) -> str:
    r = subprocess.run(["git", "-c", f"safe.directory={str(REPO).replace(chr(92), '/')}", "-C", str(REPO)] + args,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    return r.stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", dest="out", default=None)
    ap.add_argument("--extra", nargs="*", default=[], help="extra paths (e.g. an acceptance run dir's files)")
    a = ap.parse_args(argv)
    paths = list(dict.fromkeys(PATHS + list(a.extra)))
    print(f"[w1_verify_staged] PACKAGE: {PACKAGE}")
    print(f"[w1_verify_staged] PATH LIST: {len(paths)} paths, from THIS file's PATHS constant"
          + (f" + {len(a.extra)} --extra" if a.extra else ""))
    res = {"package": PACKAGE, "n_paths": len(paths), "paths": {}, "problems": []}
    for p in paths:
        if p in NOT_MINE or not p.startswith(OWNED_PREFIXES):
            res["problems"].append(f"NOT W1's PATH: {p}")
            res["paths"][p] = "REFUSED_NOT_MINE"
            continue
        line = git(["ls-files", "--stage", "--", p]).strip()
        idx = line.split()[1] if line else ""
        wt = git(["hash-object", "--", p]).strip()
        if len(idx) != 40 or len(wt) != 40:
            res["paths"][p] = f"INCONCLUSIVE index={idx[:8]!r} worktree={wt[:8]!r}"
            res["problems"].append(f"INCONCLUSIVE {p}")
        elif idx == wt:
            res["paths"][p] = f"VERIFIED {idx}"
        else:
            res["paths"][p] = f"MISMATCH index={idx} worktree={wt}"
            res["problems"].append(f"MISMATCH {p}")
    res["all_verified"] = not res["problems"]
    for p in res["problems"]:
        print("  ⛔ " + p)
    print(f"[w1_verify_staged] {sum(1 for v in res['paths'].values() if v.startswith('VERIFIED'))}/"
          f"{len(paths)} VERIFIED · all_verified={res['all_verified']}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"[w1_verify_staged] -> {a.out}")
    return 0 if res["all_verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
