#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refcv6 -- ⛔ PROVE THE ONE-VARIABLE CHECK CAN GO RED.

    python mutation_proof.py --trainer <path> --json raw/mutation_proof.json

*A check that shares the defect it checks for is green forever* -- four measured
instances in one night (`CLAUDE.md`). `check_one_variable.py` currently reads PASS
on all ten arms. That is worth **nothing** until each defect it claims to catch has
been re-introduced and observed to turn it RED.

Each mutant below is a REAL failure this programme actually suffered:

 M1  bundle two levers in one arm          the `--v2` conflation failure (ten levers
                                            on two axes, result non-attributable) and
                                            refcv5-v2 moving several at once
 M2  a lever that silently moves lambda    "a past sweep was invalidated because an
                                            arm silently changed the effective lambda
                                            alongside its named lever"
 M3  a replicate that forgets its seed     an arm that measures nothing while looking
                                            like a noise-floor read
 M4  a per-arm anchors file                a hidden second lever: a different anchor
                                            vocabulary per arm, which `PARITY_PINNED`
                                            exists to catch
 M5  the lever moves the WRONG WAY         ⭐ the discriminating control. A pair that
                                            differs in exactly ONE key passes a
                                            key-counting check while the lever is OFF.
                                            A positive-only assertion is blind here.
 M6  B built on V0 instead of A            `--wp-index on` refuses `--agents off`; the
                                            arm would read as "the index does not help"
                                            while never having had agent tokens

⛔ A mutant that SURVIVES is information about the mutant as often as about the
check -- it is reported, never quietly dropped.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import arms as A
import check_one_variable as C


def _ns(mod, argv):
    ns, err = C.parse_arm(mod.build_parser, argv)
    if ns is None:
        raise RuntimeError(f"could not parse mutant argv: {err}")
    return ns


def run_mutants(mod, *, steps: int, w_agent: float, w_tac_goal: float) -> list[dict]:
    kw = dict(steps=steps, w_agent=w_agent, w_tac_goal=w_tac_goal)
    base = {a: _ns(mod, A.arm_argv(a, **kw)) for a in
            ("V0", "V0b", "D", "Db", "A", "Ab", "B", "C")}
    results: list[dict] = []

    def record(mid, desc, arm, pair_base, ns_arm, ns_base, defect):
        v = C.check_pair(arm, pair_base, ns_arm, ns_base)
        results.append({
            "mutant": mid, "description": desc, "real_defect": defect,
            "arm": arm, "base": pair_base,
            "verdict": v["verdict"],
            "classified": v["classified"],
            "problems": v["problems"],
            "KILLED": v["verdict"] == "REFUSE",
        })

    # ---- M1: bundle two levers ------------------------------------------- #
    argv = A.arm_argv("D", **kw) + ["--wp-index", "on"]
    record("M1", "arm D additionally carries --wp-index on",
           "D", "V0", _ns(mod, argv), base["V0"],
           "the `--v2` conflation failure: ten levers on two axes, non-attributable")

    # ---- M2: a silent lambda change riding along -------------------------- #
    argv = A.arm_argv("D", **kw)
    argv = A._replace_flag(argv, "--lr", ["3e-4"])
    record("M2", "arm D silently changes --lr 1e-4 -> 3e-4 alongside its lever",
           "D", "V0", _ns(mod, argv), base["V0"],
           "'a past sweep was invalidated because an arm silently changed the "
           "effective lambda alongside its named lever'")

    # ---- M3: replicate forgets its seed ----------------------------------- #
    argv = A._replace_flag(A.arm_argv("V0b", **kw), "--seed", ["0"])
    record("M3", "replicate V0b carries seed 0, the same as V0",
           "V0b", "V0", _ns(mod, argv), base["V0"],
           "a noise-floor arm that measures nothing while looking like one that does")

    # ---- M4: a per-arm anchors file --------------------------------------- #
    argv = A.arm_argv("D", **kw)
    argv = A._replace_flag(argv, "--anchors", ["/root/data/refcv6/anchors_D.pt"])
    record("M4", "arm D points at its OWN anchors.pt",
           "D", "V0", _ns(mod, argv), base["V0"],
           "a hidden second lever: a different anchor vocabulary per arm")

    # ---- M5: the lever moves the WRONG WAY -------------------------------- #
    # ⭐ Exactly ONE key differs, so a key-counting check passes. The arm is
    #    labelled 'D' and its w_u0 is 0.5 -- identical to the control's -- so it
    #    is really the control wearing D's name. Only the value assertion sees it.
    argv = A.arm_argv("D", **kw)
    argv = A._replace_flag(argv, "--w-u0", ["0.9"])
    record("M5", "arm D sets --w-u0 0.9 instead of 0 (one key differs, wrong value)",
           "D", "V0", _ns(mod, argv), base["V0"],
           "a lever recorded as moved that never reached its pre-registered target")

    # ---- M6: B paired against V0 instead of A ----------------------------- #
    # The argv is arm B's own; only the PAIRING is wrong. Against V0 the diff
    # carries `agents` + `w_agent` as well as `wp_index` -- two levers.
    record("M6", "arm B compared against V0 instead of A (wp-index needs agents on)",
           "B", "V0", base["B"], base["V0"],
           "`--wp-index on` refuses `--agents off`; the arm would read as 'the "
           "index does not help' while never having had agent tokens")

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--steps", type=int, default=A.STEP_BUDGETS["cut"])
    ap.add_argument("--w-agent", type=float, default=1.0,
                    help="PROBE value only -- this tool checks STRUCTURE, and no "
                         "number here is a recommendation")
    ap.add_argument("--w-tac-goal", type=float, default=0.05)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    mod, why = C.load_trainer(a.trainer)
    if mod is None:
        out = {"verdict": "INCONCLUSIVE", "reason": why,
               "note": "⛔ NOT a pass -- no mutant was ever built."}
        _emit(out, a.json)
        return 3

    # ⛔ The baseline must be GREEN first. A mutation proof run against an
    # already-red check proves nothing about the mutants.
    baseline = []
    kw = dict(steps=a.steps, w_agent=a.w_agent, w_tac_goal=a.w_tac_goal)
    ns = {arm: _ns(mod, A.arm_argv(arm, **kw)) for arm in A.ARMS}
    for arm in sorted(A.ARMS):
        b = A.PAIRING.get(arm)
        if b:
            baseline.append(C.check_pair(arm, b, ns[arm], ns[b]))
    baseline_green = all(p["verdict"] == "PASS" for p in baseline)

    mutants = run_mutants(mod, **kw)
    killed = sum(1 for m in mutants if m["KILLED"])
    out = {
        "tool": "mutation_proof.py",
        "trainer_path": a.trainer,
        "baseline_pairs": len(baseline),
        "baseline_green": baseline_green,
        "n_mutants": len(mutants),
        "n_killed": killed,
        "n_survived": len(mutants) - killed,
        "survivors": [m["mutant"] for m in mutants if not m["KILLED"]],
        "mutants": mutants,
        "verdict": ("PASS" if (baseline_green and killed == len(mutants))
                    else "REFUSE"),
    }
    _emit(out, a.json)
    return 0 if out["verdict"] == "PASS" else 1


def _emit(out: dict, path: str | None) -> None:
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False, default=str)
        print(f"[mutation] wrote {path}")
    print(f"[mutation] baseline_green = {out.get('baseline_green')}")
    for m in out.get("mutants", []):
        print(f"  [{'KILLED  ' if m['KILLED'] else 'SURVIVED'}] {m['mutant']}: "
              f"{m['description']}")
        for p in m["problems"]:
            print(f"        -> {p[:150]}")
    print(f"[mutation] {out.get('n_killed')}/{out.get('n_mutants')} killed "
          f"| verdict = {out['verdict']}")


if __name__ == "__main__":
    raise SystemExit(main())
