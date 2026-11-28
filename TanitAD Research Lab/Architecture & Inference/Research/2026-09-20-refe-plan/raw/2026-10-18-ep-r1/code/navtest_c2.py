#!/usr/bin/env python3
"""NAVTEST C2: the shuffled-label negative control, read inside the confirm flow (MM_DECISIONS_EPR1B D8, first bullet).  CPU only.  NEW file; it calls the LANDED `refe_head_confirm` as a library.

WHAT IT IS.  Dev C2 (PREREG_SELRANK_ARM.md:71, PREREG_EPR1.md:85; no tolerance, MM_DECISIONS_EPR1B D7) fits the arm's head on SHUFFLED labels and requires the dev gain of those rows to be <= 0 on each
seed.  This module applies the SAME shuffled-label rows to the NAVTEST feature bank, picks by the frozen rule, and scores them against P2's pick with the SAME paired log-cluster draws as the real
readings.  A shuffled-label row has no signal to carry: a point gain > 0 on a seed means the navtest instrument can manufacture a gain from nothing, and the arm's real reading is not quotable until the
Master Mind rules (GAP GN1: the registration states no navtest C2 rule or consequence; this mirrors dev C2 per the brief and D7/D8, and the consequence is REPORTED, never applied here).

TWO STEPS, split exactly where the outcome is first needed
  picks   NO outcome array is opened (a test runs it on a table that has no `pdms` / `sub` at all).  Runs before the look.  Re-validates the look gating through `refe_head_confirm.prepare`, binds each
          C2 row to the one the DEV gate applied (the dev result records `inputs.c2_result_sha256` / `inputs.c2_rows_sha256`), runs the landed bank pass (integrity, C0, C1 reach, C4) with the C2 readings in
          the plan, and writes the C2 picks + their sha256.  A RED control refuses here: the look stays unspent.
  score   AFTER the ledger exists.  Its first statement refuses when the ledger is absent, so the outcome arrays cannot be opened before the look is spent.  It binds the ledger (table, bank manifest, the real
          rows, P2's pick) to the picks file, then opens the outcomes once and scores every C2 reading: gain point <= 0.0 on EVERY seed of an arm, else that arm's C2 is RED.

    navtest_c2.py picks|score  <the refe_head_confirm.py flags>  [--c2-out DIR]        (score also needs --look-ledger)
`confirm_read.py` runs picks -> the landed `stage_read` -> score as ONE entry point; use this CLI directly only for the documented pre-read step.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import refe_head_confirm as C   # noqa: E402  (LANDED, frozen)
import epr1 as R                # noqa: E402  (LANDED, frozen)
import epr1_dev as D            # noqa: E402  (LANDED, frozen)

SCHEMA = "navtest-c2-v1"
SEED_KEYS = C.SEED_KEYS
PICKS_REPORT, PICKS_NPZ, SCORE_RESULT = "navtest_c2_picks_report.json", "navtest_c2_picks.npz", "navtest_c2_result.json"
SUBDIR = "navtest_c2"
# per arm: the reading family it serves, the column it replaces, the C2 result file its fit wrote (srttc_c2.py / epr1.py c2) and the key under which the dev result recorded its sha256
ARM_SPEC = {"EP-R1": {"fam": "EPR1", "col": C.EP_COL, "c2_result": "epr1_c2_result.json", "run_dir": "ep_run_dir", "name": "C2EPR1"},
            "SR-TTC": {"fam": "SRTTC", "col": C.TTC_COL, "c2_result": "r1_c2_result.json", "run_dir": "ttc_run_dir", "name": "C2SRTTC"}}
GAPS = {"GN1": "no navtest C2 rule or consequence is registered (PREREG_EPR1.md:152 lists 'C2 and C4 reads' for the confirm without stating either); this mirrors dev C2 (PREREG_SELRANK_ARM.md:71; "
               "PREREG_EPR1.md:85; no tolerance, MM_DECISIONS_EPR1B D7).  A RED C2 is REPORTED with its arm; whether it voids that arm's reading is the Master Mind's ruling",
        "GN2": "the C2 picks cost a second pass over the navtest feature bank, and the C2 score a second open of the outcome arrays (after the ledger); neither changes any real reading"}


class C2Refusal(C.ConfirmRefusal):
    """A navtest-C2 precondition is not met.  In the `picks` step nothing outcome-bearing was opened; in `score` the refusal is raised before the outcomes are touched unless the message says otherwise."""


def c2_name(arm: str, sd: str) -> str:
    return f"{ARM_SPEC[arm]['name']}_s{sd}"


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the registered chain: dev result -> C2 result file -> C2 rows
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def dev_c2_binding(guard, arm: str, state: dict) -> dict:
    """The C2 files the DEV gate applied, as the dev result recorded them: {"c2_result_sha256", "c2_rows_sha256": {seed: sha}}.  A dev result that does not record them cannot bind navtest C2 to dev C2."""
    doc, sha = D.load_json(guard, state["path"])
    if sha != state["sha256"]:
        raise C2Refusal(f"{arm}: the dev result changed on disk between the look-gating read and this one")
    inp = doc.get("inputs") or {}
    rsha, rows = inp.get("c2_result_sha256"), inp.get("c2_rows_sha256") or {}
    if not rsha or set(rows) != set(SEED_KEYS) or not all(rows.get(sd) for sd in SEED_KEYS):
        raise C2Refusal(f"{arm}: the dev result {state['path']} does not record inputs.c2_result_sha256 and inputs.c2_rows_sha256 for seeds {SEED_KEYS}: navtest C2 cannot be bound to the C2 rows dev tested")
    return {"c2_result_sha256": rsha, "c2_rows_sha256": dict(rows)}


def load_c2_rows(guard, arm: str, run_dir: str, binding: dict, arm_doc: dict, real_rows_sha: dict) -> dict:
    """The arm's C2 result file and its two shuffled-label rows, bound to dev: the file is the one dev read, each row is the one dev applied, the C2 fit belongs to this c* and this bank, and no row
    is the arm's real row.  -> {"rows": {seed: (w, b, sha)}, "sha256", "doc"}."""
    spec = ARM_SPEC[arm]
    path = os.path.join(run_dir, spec["c2_result"])
    if not C._io_exists(guard, path):
        raise C2Refusal(f"{arm}: {path} does not exist: navtest C2 needs the shuffled-label rows the dev gate applied")
    doc, sha = D.load_json(guard, path)
    if sha != binding["c2_result_sha256"]:                                                    # MUTATION-TARGET-NTC2-BIND-RESULT
        raise C2Refusal(f"{arm}: {spec['c2_result']} sha256 {sha} != the one its dev result recorded ({binding['c2_result_sha256']}): these are not the C2 rows dev tested")
    if doc.get("stage") != "C2":
        raise C2Refusal(f"{arm}: {path} is not a C2 stage result (stage {doc.get('stage')!r})")
    if (doc.get("c_star") or {}).get("config_sha256") != arm_doc["c_star"]["config_sha256"]:  # MUTATION-TARGET-NTC2-BIND-CSTAR
        raise C2Refusal(f"{arm}: the C2 result was fitted at a different c* than the arm's result file")
    if doc.get("bank_manifest_sha256") != (arm_doc.get("inputs") or {}).get("bank_manifest_sha256"):      # MUTATION-TARGET-NTC2-BIND-BANK
        raise C2Refusal(f"{arm}: the C2 result was fitted on a different S0 bank than the arm's result file")
    rows = {}
    for sd in SEED_KEYS:
        ent = (doc.get("final_rows") or {}).get(sd)
        if not ent:
            raise C2Refusal(f"{arm}: the C2 result has no row for seed {sd}")
        w, b, rsha = D.load_row(guard, os.path.join(run_dir, ent["file"]), ent["sha256"])
        if rsha != binding["c2_rows_sha256"][sd]:                                              # MUTATION-TARGET-NTC2-BIND-ROWS
            raise C2Refusal(f"{arm}: C2 row seed {sd} sha256 {rsha} != the one the dev gate applied ({binding['c2_rows_sha256'][sd]})")
        if rsha == real_rows_sha[sd]:                                                          # MUTATION-TARGET-NTC2-DIFFERS
            raise C2Refusal(f"{arm}: the C2 row of seed {sd} IS the real row ({rsha[:12]}): a shuffled-label control that equals the real fit tests nothing")
        rows[sd] = (w, b, rsha)
    return {"rows": rows, "sha256": sha, "doc": doc}


def c2_plan(c2: dict, families: list) -> dict:
    """{reading name: {col: (w, b)}} for the arms whose primary reading is open; P2 first (the paired baseline).  The JOINT has no C2 (it is a secondary, never a primary)."""
    plan = {"P2": {}}
    for arm, spec in ARM_SPEC.items():
        if spec["fam"] in families:
            for sd in SEED_KEYS:
                plan[c2_name(arm, sd)] = {spec["col"]: c2[arm]["rows"][sd][:2]}
    return plan


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# step 1: picks (NO outcome)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def c2_dir_of(opts: C.ConfirmOptions, c2_out: str | None) -> str:
    return c2_out or os.path.join(opts.out_dir, SUBDIR)


def run_c2_picks(opts: C.ConfirmOptions, c2_out: str | None = None) -> dict:
    t0 = time.time()
    guard = opts.guard()
    out = c2_dir_of(opts, c2_out)
    if C._io_exists(guard, os.path.join(out, PICKS_REPORT)):
        raise C2Refusal(f"{out}/{PICKS_REPORT} exists: use a fresh C2 output directory")
    prep = C.prepare(opts)                                                                    # the look gating, the rows bound to the dev results: raises before anything else is opened
    arms_open = [arm for arm, spec in ARM_SPEC.items() if spec["fam"] in prep["families"]]
    c2, binding = {}, {}
    for arm in arms_open:
        state = prep["ep"] if arm == "EP-R1" else prep["ttc"]
        binding[arm] = dev_c2_binding(guard, arm, state)
        run_dir = getattr(opts, ARM_SPEC[arm]["run_dir"])
        c2[arm] = load_c2_rows(guard, arm, run_dir, binding[arm], prep["docs"][arm]["doc"], {sd: v[2] for sd, v in prep["arm_rows"][arm].items()})
    plan = c2_plan(c2, prep["families"])
    tab = C.load_table_inputs(guard, opts)
    man = C.read_bank_manifest(guard, opts.bank_dir)
    stats = C.process_bank(C.iter_bank(guard, opts, man), tab, plan, prep["W6"], prep["b6"], opts)
    ctl = C.controls_from_stats(stats, tab, plan, opts)
    if not C.controls_pass(ctl):                                                              # MUTATION-TARGET-NTC2-CTLGATE
        red = [k for k, v in ctl.items() if not v.get("pass", False)]
        raise C2Refusal(f"control(s) RED in the C2 pass: {red}. The look is UNSPENT. Detail: {json.dumps(D._jsonable({k: ctl[k] for k in red}))[:1500]}")
    P = stats["picks"]
    readings = {}
    for nm in plan:
        if nm == "P2":
            continue
        readings[nm] = {"changed_picks_vs_p2": int((P[nm] != P["P2"]).sum()), "picks_sha256": C.sha256_bytes(P[nm].tobytes()),
                        "entries_changed_in_named_column": ctl["C1_reach"]["per_reading"][nm]["entries_changed_in_named_columns"]}
    C._io_makedirs(guard, out)
    picks_blob = C.npz_bytes({"token": tab.token, **P})
    picks_sha = C._io_write_bytes(guard, os.path.join(out, PICKS_NPZ), picks_blob, exclusive=True)
    rep = {"schema": SCHEMA, "stage": "C2-PICKS", "look": {"spent": False, "note": "no outcome array was opened (pdms / sub / valid)"}, "gaps": GAPS,
           "arms": arms_open, "readings": readings, "picks_file": {"name": PICKS_NPZ, "sha256": picks_sha}, "p2_picks_sha256": C.sha256_bytes(P["P2"].tobytes()),
           "binding": {arm: {"dev_c2_result_sha256": binding[arm]["c2_result_sha256"], "dev_c2_rows_sha256": binding[arm]["c2_rows_sha256"], "c2_result_sha256": c2[arm]["sha256"],
                             "c2_rows_sha256": {sd: v[2] for sd, v in c2[arm]["rows"].items()}, "real_rows_sha256": {sd: v[2] for sd, v in prep["arm_rows"][arm].items()},
                             "real_result_sha256": prep["docs"][arm]["sha256"]} for arm in arms_open},
           "inputs": {"navtest_table_sha256": tab.sha256, "bank_manifest_sha256": man["sha256"], "n_tokens": tab.n, "n_logs": len(tab.log_names), "p2_ckpt": opts.p2_ckpt, "pins_overridden": opts.pins_overridden()},
           "controls": ctl, "run_meta": {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_s": round(time.time() - t0, 1)}}
    C._io_write_bytes(guard, os.path.join(out, PICKS_REPORT), C.canon(rep), exclusive=True)
    print(f"[{time.strftime('%H:%M:%S')}] navtest C2 picks done: arms {arms_open}, {out}", flush=True)
    return rep


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# step 2: score (AFTER the ledger)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def check_ledger(guard, opts: C.ConfirmOptions, rep: dict) -> dict:
    """The ledger is the look.  It must exist (checked by the caller BEFORE this runs) and describe the SAME look the C2 picks were made for: table, bank manifest, the real rows, and P2's pick."""
    led, led_sha = D.load_json(guard, opts.look_ledger)
    why = []
    if led.get("schema") != C.SCHEMA:
        why.append(f"ledger schema {led.get('schema')!r}")
    if led.get("table_sha256") != rep["inputs"]["navtest_table_sha256"] or led.get("bank_manifest_sha256") != rep["inputs"]["bank_manifest_sha256"]:    # MUTATION-TARGET-NTC2-LEDGER-BIND
        why.append("table / bank manifest differ from the ones the C2 picks were made on")
    if (led.get("picks_sha256") or {}).get("P2") != rep["p2_picks_sha256"]:
        why.append("P2's pick in the ledger differs from the C2 pass's P2 pick")
    for arm in rep["arms"]:
        if (led.get("rows_sha256") or {}).get(arm) != rep["binding"][arm]["real_rows_sha256"]:
            why.append(f"{arm}: the real rows in the ledger differ from the ones the C2 binding used")
        if ARM_SPEC[arm]["fam"] not in (led.get("readings") or []):
            why.append(f"{arm}: its reading is not in the ledger")
    if why:
        raise C2Refusal("the ledger does not describe the look these C2 picks belong to: " + "; ".join(why))
    return {"path": opts.look_ledger, "sha256": led_sha, "spent": led.get("spent")}


def c2_verdict(per_seed: dict) -> dict:
    """An arm's C2 passes iff the point gain is <= 0.0 on EVERY seed (no tolerance: MM_DECISIONS_EPR1B D7).  A seed whose C2 changed no pick reads exactly 0.0, which satisfies the rule and tests
    nothing: it is reported as vacuous (never as a pass of substance)."""
    seeds = {}
    for sd, rec in per_seed.items():
        pt = rec["gain_x100"]["point"]
        seeds[sd] = {"gain_x100": rec["gain_x100"], "changed_picks": rec["changed_picks"], "meets_rule": bool(pt <= 0.0), "vacuous": bool(rec["changed_picks"] == 0)}      # MUTATION-TARGET-NTC2-RULE
    ok = bool(all(v["meets_rule"] for v in seeds.values()))                                    # MUTATION-TARGET-NTC2-BOTHSEEDS
    return {"per_seed": seeds, "pass": ok, "vacuous_seeds": [sd for sd, v in seeds.items() if v["vacuous"]]}


def run_c2_score(opts: C.ConfirmOptions, c2_out: str | None = None) -> dict:
    t0 = time.time()
    guard = opts.guard()
    if not opts.look_ledger:
        raise C2Refusal("`score` needs --look-ledger")
    if not C._io_exists(guard, opts.look_ledger):                                              # MUTATION-TARGET-NTC2-LEDGER
        raise C2Refusal(f"{opts.look_ledger} does not exist: the navtest look is not spent, so no outcome array is opened (the C2 score waits for the ledger)")
    out = c2_dir_of(opts, c2_out)
    rep, rep_sha = D.load_json(guard, os.path.join(out, PICKS_REPORT))
    if rep.get("schema") != SCHEMA or rep.get("stage") != "C2-PICKS":
        raise C2Refusal(f"{out}/{PICKS_REPORT} is not a C2 picks report")
    ledger = check_ledger(guard, opts, rep)
    pk_path = os.path.join(out, PICKS_NPZ)
    if C._io_sha256(guard, pk_path) != rep["picks_file"]["sha256"]:                            # MUTATION-TARGET-NTC2-PICKSHA
        raise C2Refusal(f"{pk_path}: sha256 differs from the picks report (edited after the pass)")
    z = C._io_npz(guard, pk_path)
    picks = {k: z[k] for k in z.files if k != "token"}
    token_c2 = np.array([str(x) for x in z["token"]])
    z.close()
    tab = C.load_table_inputs(guard, opts)
    if not np.array_equal(token_c2, tab.token):
        raise C2Refusal("the C2 picks' token order differs from the navtest table's")
    for nm, ent in rep["readings"].items():
        if C.sha256_bytes(picks[nm].tobytes()) != ent["picks_sha256"]:
            raise C2Refusal(f"{nm}: picks differ from the report's sha256")
    # ---- the look is spent and bound: ONLY NOW are the outcomes opened
    outc = D.read_outcomes(guard, opts.table)
    nav = types.SimpleNamespace(n=tab.n, log_idx=tab.log_idx, n_logs=len(tab.log_names))
    draws = R.boot_draws(nav.n_logs)
    base = picks["P2"].astype(np.int64)
    arms = {}
    for arm in rep["arms"]:
        per_seed = {sd: D.score_arm(nav, outc, picks[c2_name(arm, sd)].astype(np.int64), base, draws) for sd in SEED_KEYS}
        arms[arm] = c2_verdict(per_seed)
        arms[arm]["rule"] = "point gain <= 0.0 on each seed, no tolerance (PREREG_SELRANK_ARM.md:71, PREREG_EPR1.md:85; MM_DECISIONS_EPR1B D7), mirrored from dev C2 (GAP GN1)"
        arms[arm]["components_x100"] = {sd: {k: per_seed[sd]["components"][k]["delta_x100"] for k in per_seed[sd]["components"]} for sd in SEED_KEYS}
    red = [arm for arm, v in arms.items() if not v["pass"]]
    res = {"schema": SCHEMA, "stage": "C2-SCORE", "tier": "T1 NAVSIM v1.1 navtest, self-action open loop (EVAL_DOCTRINE); a CONTROL, not a claim", "look": ledger,
           "arms": arms, "c2_pass_all_arms": bool(not red), "red_arms": red,
           "consequence": ({"registered": False, "note": "GAP GN1: no consequence is registered for a RED navtest C2.  Reported, not applied: the Master Mind rules whether the arm's reading is quotable."} if red
                           else {"registered": False, "note": "all C2 readings meet the rule; nothing to apply"}),
           "estimator": {"name": "paired log-cluster bootstrap (epr1.boot_draws, the SAME draws as the real readings)", "B": R.BOOT_B, "seed": R.BOOT_SEED, "n_logs": nav.n_logs, "n_tokens": nav.n},
           "gaps": GAPS, "picks_report_sha256": rep_sha, "run_meta": {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_s": round(time.time() - t0, 1)}}
    C._io_write_bytes(guard, os.path.join(out, SCORE_RESULT), C.canon(res), exclusive=False)
    print(f"[{time.strftime('%H:%M:%S')}] navtest C2 score: pass_all={res['c2_pass_all_arms']} red_arms={red}", flush=True)
    return res


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def options_from_args(a) -> C.ConfirmOptions:
    return C.ConfirmOptions(out_dir=a.out_dir or os.path.join(a.bank_dir, "confirm_out"), bank_dir=a.bank_dir, ep_run_dir=a.ep_run_dir, ep_registered_result_sha256=a.ep_registered_result_sha256,
                            ep_dev_result=a.ep_dev_result, ep_abandon=a.ep_abandon, ttc_run_dir=a.ttc_run_dir, ttc_registered_result_sha256=a.ttc_registered_result_sha256,
                            ttc_dev_result=a.ttc_dev_result, ttc_abandon=a.ttc_abandon, table=a.table, p2_ckpt=a.p2_ckpt, banked_p2_seam=a.banked_p2_seam, look_ledger=a.look_ledger)


def build_parser() -> argparse.ArgumentParser:
    ap = C.build_parser()
    ap.description = "Navtest C2 (shuffled-label negative control) for the REFE head confirm.  picks: no outcome, before the look.  score: after the ledger exists."
    for act in ap._actions:
        if act.dest == "cmd":
            act.choices = ("picks", "score")
    ap.add_argument("--c2-out", default=None, help="default <out-dir>/navtest_c2")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    opts = options_from_args(a)
    try:
        {"picks": run_c2_picks, "score": run_c2_score}[a.cmd](opts, a.c2_out)
    except (C.ConfirmRefusal, D.DevRefusal, D.DevGuardError) as ex:
        print(f"REFUSED: {ex}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
