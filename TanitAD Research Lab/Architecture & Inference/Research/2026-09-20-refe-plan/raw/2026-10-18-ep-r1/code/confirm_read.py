#!/usr/bin/env python3
"""THE CONFIRM READ ENTRY POINT: the landed `refe_head_confirm.py read` plus the two pieces MM_DECISIONS_EPR1B D8 made launch blockers -- the navtest C2 reading and the four metric families -- in ONE
ordered run.  CPU only.  NEW file; the landed applier is called as a library and is not edited.

WHY A WRAPPER.  The landed `read` stage cannot call modules written after it, and its result prints `families6: PENDING`.  Running `refe_head_confirm.py read` directly spends the one navtest look WITHOUT the C2
reading and WITHOUT the families.  This module is the entry point that cannot do that: it refuses to spend the look unless both can follow.  (If the landed CLI is run anyway, `confirm_read.py post` finishes
the job on the spent look and `confirm_read.py verify` states what is missing.)

ORDER (each step's reason is the outcome it may or may not open)
  0 preflight   the ledger is ABSENT (else the look is spent); the families6 tool, its navtest inputs export and its interpreter RESOLVE (an unrunnable tool must refuse BEFORE the look is spent);
                no final report exists in the out dir.
  1 C2 picks    `navtest_c2.run_c2_picks`.  No outcome.  Re-validates the look gating, binds the C2 rows to the dev gates', runs the bank pass.  A RED control refuses here: the look stays unspent.
  2 READ        `refe_head_confirm.stage_read`.  SPENDS THE LOOK (the ledger is created exclusively before pdms / sub are opened).
  3 C2 score    `navtest_c2.run_c2_score`.  Refuses unless the ledger exists and describes this look; then opens the outcomes once more and scores the shuffled-label readings.
  4 families    `confirm_families.run_families`.  families6 on every seam of the look; the four families per reading; the registered families6 clause per arm.
  5 final       `confirm_final_report.json`, created exclusively, LAST: per arm, the landed clauses, the families6 clause and the navtest C2 side by side, and a CANDIDATE verdict that is a mechanical
                application of the registered bar (PREREG_EPR1.md:110, PREREG_SELRANK_ARM.md:94) -- the Master Mind signs it.
Steps 3-5 are `post`; a failure after step 2 raises `PostLedgerError` ("THE LOOK IS SPENT") and `post` resumes.  Everything but the final report is overwritable, because it is derived from the spent look.

    confirm_read.py read|post|verify  <the refe_head_confirm.py flags>  --look-ledger F  [--c2-out DIR] [--families6 F] [--inputs J] [--python P] [--n-boot 2000] [--runner subprocess|inprocess]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import refe_head_confirm as C   # noqa: E402  (LANDED, frozen)
import epr1_dev as D            # noqa: E402  (LANDED, frozen)
import navtest_c2 as N          # noqa: E402
import confirm_families as F    # noqa: E402

SCHEMA = "confirm-final-v1"
FINAL_NAME = "confirm_final_report.json"
ARMS = (("EP-R1", "EPR1"), ("SR-TTC", "SRTTC"))
VERDICTS = ("CLEARS_ALL_REGISTERED_CLAUSES", "DOES_NOT_CLEAR", "CONTROL_RED_RULING_NEEDED", "INCOMPLETE")


class PostLedgerError(RuntimeError):
    """A step after the look was spent failed.  The look IS spent; `confirm_read.py post` resumes."""


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the candidate verdict (pure; tested on hand-built inputs)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def arm_verdict(arm: str, fam: str, res: dict, c2: dict | None, famrep: dict | None) -> dict:
    """The registered navtest bar for ONE arm, assembled from the three reports.  Precedence: INCOMPLETE (a piece is missing or not computed) > CONTROL_RED (navtest C2 failed: a ruling is needed, GN1)
    > CLEARS (every clause, both seeds) > DOES_NOT_CLEAR (with the failing clauses named)."""
    why, parts = [], {}
    prim = (res.get("primaries") or {}).get(arm)
    if prim is None:
        return {"candidate_verdict": "INCOMPLETE", "reasons": [f"{arm}: its primary reading is not in the look"], "parts": {}}
    parts["landed_clauses_both_seeds"] = bool(prim["both_seeds_clear_all_clauses_except_families6"])
    parts["landed_clauses_per_seed"] = {sd: prim["per_seed"][sd] for sd in C.SEED_KEYS}
    fa = (famrep or {}).get("arms", {}).get(arm)
    readings = (famrep or {}).get("readings", {})
    needed = ["P2"] + [f"{fam}_s{sd}" for sd in C.SEED_KEYS]
    inc = [n for n in needed if not (readings.get(n) or {}).get("complete")]
    if famrep is None or fa is None or inc:                                                  # MUTATION-TARGET-CRD-V-INCOMPLETE
        why.append(f"four-family report incomplete or not computed for {inc or 'the arm'}: an eval missing a family is not a result")
    parts["families6_clause_both_seeds"] = None if fa is None else bool(fa["both_seeds_clear_families6_clause"])
    parts["families6_clause_per_seed"] = None if fa is None else fa["per_seed_clears"]
    ca = ((c2 or {}).get("arms") or {}).get(arm)
    if ca is None:
        why.append("navtest C2 has no reading for this arm")
    parts["navtest_c2_pass"] = None if ca is None else bool(ca["pass"])
    parts["navtest_c2_vacuous_seeds"] = None if ca is None else ca.get("vacuous_seeds")
    if why:
        return {"candidate_verdict": "INCOMPLETE", "reasons": why, "parts": parts}
    if not ca["pass"]:                                                                       # MUTATION-TARGET-CRD-V-C2RED
        return {"candidate_verdict": "CONTROL_RED_RULING_NEEDED",
                "reasons": ["navtest C2 (shuffled-label rows) read a gain > 0 on a seed: the instrument manufactured a gain from nothing.  GAP GN1: no consequence is registered; the Master Mind rules."], "parts": parts}
    fails = []
    if not parts["landed_clauses_both_seeds"]:
        fails.append("landed clauses (total point >= +0.30, lower bound > 0, no adverse sub-score" + (", EP component" if fam == "EPR1" else "") + ") do not clear on both seeds")
    if not parts["families6_clause_both_seeds"]:
        fails.append("a longitudinal or lateral family component is separated adverse on a seed, or nothing was compared (families6 clause)")
    if not fails:                                                                            # MUTATION-TARGET-CRD-V-CLEARS
        return {"candidate_verdict": "CLEARS_ALL_REGISTERED_CLAUSES", "reasons": [], "parts": parts}
    return {"candidate_verdict": "DOES_NOT_CLEAR", "reasons": fails, "parts": parts}


def assemble_final(res: dict, c2: dict | None, famrep: dict | None, hashes: dict, notes: dict | None = None) -> dict:
    arms = {}
    for arm, fam in ARMS:
        if fam in res.get("readings_open", []):
            v = arm_verdict(arm, fam, res, c2, famrep)
            v["other_primary_beside_it"] = (res.get("primaries") or {}).get(arm, {}).get("other_primary_beside_it")
            arms[arm] = v
    readings = {}
    for nm, r in ((famrep or {}).get("readings") or {}).items():
        readings[nm] = {"status": r["status"], "complete": r["complete"], "longitudinal": r["longitudinal"].get("complete"), "lateral": r["lateral"].get("complete"),
                        "tactical": r["tactical"].get("complete"), "strategic": r["strategic"]["status"], "vs_p2_clears": (r.get("vs_p2") or {}).get("clears")}
    strategic_ok = bool(readings) and all(v["strategic"] == "PAUSED" for v in readings.values())
    comp = {"c2_reported_for_every_open_arm": bool(c2 is not None and all(a in (c2.get("arms") or {}) for a in arms)),
            "families_reported_for_every_reading": bool(famrep is not None and famrep.get("all_readings_reported") and famrep.get("all_complete")),
            "strategic_printed_paused_on_every_reading": strategic_ok, "families_not_computed": (famrep or {}).get("not_computed")}
    comp["complete"] = bool(comp["c2_reported_for_every_open_arm"] and comp["families_reported_for_every_reading"] and comp["strategic_printed_paused_on_every_reading"])
    joint = {nm: {"gain_x100": res["readings"][nm]["gain_x100"], "label": res["readings"][nm].get("label")} for nm in res.get("readings", {}) if nm.startswith("JOINT")}
    return {"schema": SCHEMA, "stage": "FINAL", "tier": "T1 NAVSIM v1.1 navtest, self-action open loop (EVAL_DOCTRINE)", "look": res.get("look"),
            "multiplicity_disclosure": res.get("multiplicity_disclosure"), "readings_open": res.get("readings_open"), "readings_not_open": res.get("readings_not_open"),
            "arms": arms, "joint_secondary": joint, "families": readings, "completeness": comp,
            "navtest_c2": {"pass_all_arms": (c2 or {}).get("c2_pass_all_arms"), "red_arms": (c2 or {}).get("red_arms"), "consequence": (c2 or {}).get("consequence")},
            "strategic": {"status": "PAUSED", "ruling": F.STRATEGIC_RULING}, "gaps": {**N.GAPS},
            "verdict_note": "CANDIDATE: the registered bar applied mechanically (PREREG_EPR1.md:110, PREREG_SELRANK_ARM.md:94); the Master Mind signs it.  Distance-keeping is UNAVAILABLE until lead-agent tracks reach families6.py.",
            "notes": notes or {}, "hashes": hashes, "evidence_class": "MEASURED (ours; the files named in `hashes`)"}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the run
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def preflight(opts: C.ConfirmOptions, cfg: F.FamiliesConfig) -> dict:
    guard = opts.guard()
    if not opts.look_ledger:
        raise C.ConfirmRefusal("`read` needs --look-ledger: ONE ledger file outside the out dir records the single navtest look of the whole head programme")
    if C._io_exists(guard, opts.look_ledger):                                                # MUTATION-TARGET-CRD-PF-SPENT
        raise C.ConfirmRefusal(f"{opts.look_ledger} exists: the navtest look is SPENT.  `confirm_read.py post` finishes the C2 score and the families on a spent look; `verify` lists what is missing.")
    if C._io_exists(guard, os.path.join(opts.out_dir, FINAL_NAME)):
        raise C.ConfirmRefusal(f"{opts.out_dir}/{FINAL_NAME} exists: use a fresh output directory")
    tool = F.resolve_tool(cfg)                                                               # MUTATION-TARGET-CRD-PF-TOOL
    return {"tool": tool}


def _hashes(opts: C.ConfirmOptions, c2_out: str | None) -> dict:
    guard = opts.guard()
    out = {"look_ledger_sha256": C._io_sha256(guard, opts.look_ledger),
           "confirm_read_result_sha256": C._io_sha256(guard, os.path.join(opts.out_dir, "confirm_read_result.json")),
           "navtest_c2_result_sha256": C._io_sha256(guard, os.path.join(N.c2_dir_of(opts, c2_out), N.SCORE_RESULT)),
           "navtest_c2_picks_report_sha256": C._io_sha256(guard, os.path.join(N.c2_dir_of(opts, c2_out), N.PICKS_REPORT)),
           "families_report_sha256": C._io_sha256(guard, os.path.join(opts.out_dir, F.FAMILIES_DIR, F.REPORT_NAME))}
    return out


def post(opts: C.ConfirmOptions, cfg: F.FamiliesConfig, c2_out: str | None = None) -> dict:
    """Steps 3-5 on a SPENT look (also the resume path)."""
    guard = opts.guard()
    if not opts.look_ledger or not C._io_exists(guard, opts.look_ledger):
        raise C.ConfirmRefusal(f"{opts.look_ledger} does not exist: the look is not spent, so there is nothing to finish")
    if C._io_exists(guard, os.path.join(opts.out_dir, FINAL_NAME)):                          # MUTATION-TARGET-CRD-POST-ONCE
        raise C.ConfirmRefusal(f"{opts.out_dir}/{FINAL_NAME} exists: the final report is created once")
    res, _ = D.load_json(guard, os.path.join(opts.out_dir, "confirm_read_result.json"))
    late = not C._io_exists(guard, os.path.join(N.c2_dir_of(opts, c2_out), N.PICKS_REPORT))          # the landed CLI was run directly: the C2 picks (no outcome) are made now and DISCLOSED
    if late:
        N.run_c2_picks(opts, c2_out)
    c2 = N.run_c2_score(opts, c2_out)
    fam = F.run_families(opts, cfg)
    final = assemble_final(res, c2, fam, _hashes(opts, c2_out), {"c2_picks_made_after_the_look": bool(late)})
    C._io_write_bytes(guard, os.path.join(opts.out_dir, FINAL_NAME), C.canon(final), exclusive=True)         # MUTATION-TARGET-CRD-FINAL-EXCL
    print(f"[{time.strftime('%H:%M:%S')}] FINAL: " + "; ".join(f"{a}: {v['candidate_verdict']}" for a, v in final["arms"].items()) + f"; complete={final['completeness']['complete']}", flush=True)
    return final


def run_confirm_read(opts: C.ConfirmOptions, cfg: F.FamiliesConfig, c2_out: str | None = None) -> dict:
    preflight(opts, cfg)
    N.run_c2_picks(opts, c2_out)                                                             # MUTATION-TARGET-CRD-C2PICKS
    C.stage_read(opts)                                                                       # THE LOOK IS SPENT HERE
    try:
        return post(opts, cfg, c2_out)
    except Exception as ex:                                                                  # noqa: BLE001
        raise PostLedgerError(f"THE LOOK IS SPENT ({opts.look_ledger}) but a later step failed: {type(ex).__name__}: {ex}.  Fix the cause and run `confirm_read.py post` with the same flags; do not read again.") from ex


def verify(opts: C.ConfirmOptions, c2_out: str | None = None) -> dict:
    """What exists, and whether the pieces describe the SAME look.  Nothing is computed and no outcome is opened."""
    guard = opts.guard()
    ex = lambda *p: C._io_exists(guard, os.path.join(*p))
    c2dir = N.c2_dir_of(opts, c2_out)
    chk = {"ledger": bool(opts.look_ledger) and C._io_exists(guard, opts.look_ledger), "landed_read_result": ex(opts.out_dir, "confirm_read_result.json"),
           "navtest_c2_picks_report": ex(c2dir, N.PICKS_REPORT), "navtest_c2_result": ex(c2dir, N.SCORE_RESULT), "families_report": ex(opts.out_dir, F.FAMILIES_DIR, F.REPORT_NAME),
           "final_report": ex(opts.out_dir, FINAL_NAME)}
    missing = [k for k, v in chk.items() if not v]
    bound = None
    if not missing:
        fin, _ = D.load_json(guard, os.path.join(opts.out_dir, FINAL_NAME))
        now = _hashes(opts, c2_out)
        bound = {k: bool(fin.get("hashes", {}).get(k) == v) for k, v in now.items()}
    out = {"checks": chk, "missing": missing, "final_bound_to_these_files": bound,
           "complete": bool(not missing and bound and all(bound.values())),                    # MUTATION-TARGET-CRD-VERIFY
           "next": ("run `confirm_read.py read`" if not chk["ledger"] else ("run `confirm_read.py post` with the same flags: the look is spent but the C2 score / families / final report are missing" if missing else None))}
    return out


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = N.build_parser()
    ap.description = ("The confirm READ entry point: preflight -> navtest C2 picks -> the landed read (SPENDS THE LOOK) -> C2 score -> four-family report -> final report.  "
                      "`post` resumes on a spent look; `verify` lists what is missing.")
    for act in ap._actions:
        if act.dest == "cmd":
            act.choices = ("read", "post", "verify")
    ap.add_argument("--families6", default=None)
    ap.add_argument("--inputs", default=None, help="the navtest inputs export for families6 (default: $REFE_NAVTEST_INPUTS, then the archived w3_navtest_v1 export)")
    ap.add_argument("--python", default=F.PY_T)
    ap.add_argument("--n-boot", type=int, default=F.N_BOOT_DEFAULT)
    ap.add_argument("--runner", choices=sorted(F.RUNNERS), default="subprocess")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    opts = N.options_from_args(a)
    cfg = F.FamiliesConfig(families6=a.families6, inputs=a.inputs, python=a.python, n_boot=a.n_boot, runner=a.runner)
    try:
        if a.cmd == "verify":
            v = verify(opts, a.c2_out)
            print(json.dumps(v, indent=1))
            return 0 if v["complete"] else 1
        {"read": run_confirm_read, "post": post}[a.cmd](opts, cfg, a.c2_out)
    except PostLedgerError as ex:
        print(f"FAILED AFTER THE LOOK WAS SPENT: {ex}", flush=True)
        return 3
    except (C.ConfirmRefusal, D.DevRefusal, D.DevGuardError) as ex:
        print(f"REFUSED: {ex}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
