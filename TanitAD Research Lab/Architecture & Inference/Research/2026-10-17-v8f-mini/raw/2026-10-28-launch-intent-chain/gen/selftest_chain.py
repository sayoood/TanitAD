#!/usr/bin/env python3
"""Self-test + mutation proof of the LAUNCH-INTENT chain (PREREG_V8F_LAUNCH_INTENT arm A).  DRY RUN ONLY: box 2 is never contacted (ssh / scp are stubs that log and fail), no chain is launched, nothing is installed.

    python selftest_chain.py [--layer py|sh|all] [--tests-only|--mutants-only] [--filter SUBSTR] [--quiet]

Layers:  py = the asserter / run-record / K-SHARE / REP-licence / predecessor-verdict modules, driven directly (original and mutated text);
         sh = the chain, lib, waker, guard and the staged queue under REAL bash / REAL cmd.exe, in scratch worlds under F:/TanitAD-scratch/launch_intent_chain/.
A mutant is one text edit that a registered probe must flip; worlds are built with the ORIGINAL code so a mutant cannot make its own world consistent.
Exit 0 only when every test passes AND every mutant is killed.  Writes logs/selftest_chain_result.json next to the deliverable.
"""
from __future__ import annotations

import json
import sys
import time

import st_common as S
import st_py



def tripwire(before: dict, after: dict) -> list:
    """A NEW (or changed) file in the REAL installed ship dir / its logs that mentions launchintent: the self-test reached the box-2 side of the world."""
    return sorted(k for k, v in after.items() if "launchintent" in k.lower() and before.get(k) != v)


def main(argv) -> int:
    import sealed
    snap0 = sealed.real_ship_snapshot()
    layer = argv[argv.index("--layer") + 1] if "--layer" in argv else "all"
    flt = argv[argv.index("--filter") + 1] if "--filter" in argv else None
    do_t, do_m = "--mutants-only" not in argv, "--tests-only" not in argv
    t0 = time.time()
    if do_t:
        if layer in ("py", "all"):
            st_py.run_tests()
        if layer in ("sh", "all"):
            import st_sh2
            st_sh2.run_tests()
    if do_m:
        if layer in ("py", "all"):
            st_py.run_mutants()
        if layer in ("sh", "all"):
            import st_sh2
            st_sh2.run_mutants()
    if layer in ("sh", "all"):
        stray = tripwire(snap0, sealed.real_ship_snapshot())
        S.T("tripwire: the self-test created or changed NO launchintent file in the REAL installed ship dir or its logs (the incident's stray files predate this run and are in the baseline)", not stray, stray)
        S.T("tripwire: the tripwire flags a new launchintent file (positive control)", tripwire({}, {"logs/x_launchintent.log": (1, 2)}) == ["logs/x_launchintent.log"] and tripwire({"a_launchintent": (1, 2)}, {"a_launchintent": (1, 2)}) == [], "")
    res = [r for r in S.RESULTS if flt is None or flt in r[0]]
    mut = [m for m in S.MUTANTS if flt is None or flt in m[0]]
    nt, nm = sum(1 for _, ok in res if ok), sum(1 for _, ok in mut if ok)
    failed = [n for n, ok in res if not ok]
    survivors = [n for n, ok in mut if not ok]
    line = f"SELFTEST: {nt}/{len(res)} tests pass; MUTATION: {nm}/{len(mut)} mutants killed, {len(survivors)} survivors  ({time.time() - t0:.0f} s, layer={layer})"
    print(line, flush=True)
    if failed:
        print("FAILED TESTS:", *failed, sep="\n  ")
    if survivors:
        print("SURVIVORS:", *survivors, sep="\n  ")
    out = S.C.OUT / "logs" / ("selftest_chain_result.json" if layer == "all" and flt is None and do_t and do_m else f"selftest_chain_result_{layer}{'_t' if not do_m else ''}{'_m' if not do_t else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": line, "tests": res, "mutants": mut, "failed": failed, "survivors": survivors}, indent=1), encoding="utf-8")
    return 0 if (not failed and not survivors and res and (mut or not do_m)) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
