#!/usr/bin/env python3
"""r0e_id_read: the READER of R0e-id's identity verdict (PREREG_R0E_PICK.md section 5.1; launch blocker B10).  Stdlib only; no GPU, no network, no pod access of its own.

    python eval/r0e_id_read.py --arm-dir <RLS1>/arms/R0e-id

Writes `<arm-dir>/id_verdict.json` and prints `ZZR0EID R0e-id <ID-PASS|ID-FAIL|ID-DIVERGED|NOT_READ> <why>`.  `rls1_checks.py id-gate` (and so `rls1_pod.sh`, before R0e-pick) reads that file and trusts it only if THIS tool wrote it for this
directory as it stands now (tool, record_sha256 by `record_digest`, the sha256 of every input, and a fresh `read_arm` reproducing the record) -- the same contract as eval/safeobj_read.py.  The operator runs it after R0e-id finishes and BEFORE
RLS1_ARMS="R0e-pick"; the chain never writes the verdict itself (a record made by anything else is refused).

THE DEFINITION, quoted (PREREG_R0E_PICK.md 5.1, "run in this order; the first that matches is the outcome"):
  1. ID-FAIL (blocks R0e-pick), ANY of: the data event's `epo_con` block != R0d's eight keys + `temp: 0.1`;  its `epo` block != R0d's;  `train_sets` != 16,000;  the arms-file self-check fails;  G13 not PASS;  the step0 eval's
     base.pick, A.pick or B.pick differs from R0d's step0 by > 0.10 points.
  2. ID-DIVERGED (blocks R0e-pick; escalate): at u250 the |difference| of the B - own A contrast between R0e-id and R0d is > 0.50 for TTC or for PDMS (R0d: TTC -1.167, PDMS +0.683).
  3. ID-PASS: otherwise.
"Otherwise" is a NEGATIVE definition, so this reader opens only when EVERY clause above was actually EVALUATED and none fired; a clause whose input is missing, unparsable, duplicated or from another run is NOT_READ (the reasons are listed),
never a pass.  An ID-FAIL clause that fired wins over a NOT_READ in the same directory (a definite failure is not made softer by a missing file), and ID-FAIL wins over ID-DIVERGED (the order above).

WHAT IT READS (all in <arm-dir>, written by the trainer / the chain):
  sft.jsonl             the `data` event (clauses 1-3), the step0 `eval` event (clause 6), every `score_dump` event (run identity: each dump path must lie under /arms/R0e-id/)
  PREFLIGHT_PASS.json   the preflight event, written by scorer_finetune --preflight only when EVERY gate is ok and no regression switch was set (clause 5: its G13 entry must be ok and un-mutated)
  rlval_id_u250.json    rl_val_eval.py --scores A=scores_update250_A.npz B=scores_update250_B.npz --base A  (the chain writes it in step 9d for R0e-id; the same call as R0d's run_checks_numeric.sh) (ID-DIVERGED)
 and, from the package (sha256 listed in the record): eval/rls1e_arms.json + eval/rls1d_arms.json (clause 4, RE-DERIVED here, not trusted from the file's stored flags) and eval/r0d_id_reference.json (R0d's numbers, frozen from its banked files).
NOT READ HERE, on purpose: whether R0e-id ran to u1000 or was killed after u250.  The prereg's identity readings are step0 and u250 only and it is silent on a later kill; the chain already refuses to continue past a killed arm.
Exit code: 0 = ID-PASS, 1 = ID-FAIL or ID-DIVERGED, 2 = NOT_READ, 3 = usage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

TOOL = "r0e_id_read/1"
ARM = "R0e-id"
OUT_NAME = "id_verdict.json"
U250_NAME = "rlval_id_u250.json"
REQUIRED_INPUTS = ("sft.jsonl", "PREFLIGHT_PASS.json", U250_NAME)           # a verdict always lists these three (the reader fails closed without any); the gate refuses a record that lists fewer
PACKAGE_FILES = ("eval/rls1e_arms.json", "eval/rls1d_arms.json", "eval/r0d_id_reference.json")
POD_RLS1 = "/workspace/data/rls1"
REGISTERED_TEMP = 0.1
TRAIN_SETS = 16000
EXPECTED_CHECKS = ("epo_con_block_is_R0d_plus_temp_0.1", "epo_block_is_R0d", "train_sets_is_16000", "step0_base_pick_within_0.10", "step0_A_pick_within_0.10", "step0_B_pick_within_0.10",
                   "arms_file_self_check", "g13_pass", "u250_contrast_within_0.50_of_R0d")        # every clause of 5.1: a PASS evaluated all nine
PICK_TOL = 0.10                                  # clause 6: points (x100), strictly greater fails
DIV_TOL = 0.50                                   # ID-DIVERGED: points (x100), strictly greater diverges
EPS = 1e-9                                       # float slack on the two thresholds only (a difference of exactly 0.10 / 0.50 is not "greater")
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
OUTCOMES = ("ID-PASS", "ID-FAIL", "ID-DIVERGED", "NOT_READ")
VOLATILE_FIELDS = ()                             # the record is a pure function of the files it hashes (no clock, no path, no host): a fresh read of an unchanged directory reproduces it EXACTLY


def sha256_file(path, lf=False):
    b = open(path, "rb").read()
    return hashlib.sha256(b.replace(b"\r\n", b"\n") if lf else b).hexdigest()


def record_digest(rec):
    """THE canonicalisation of `record_sha256`: sha256 of the compact, key-sorted JSON of the record WITHOUT that field (the gate calls THIS function and keeps no copy)."""
    body = json.dumps({k: v for k, v in rec.items() if k != "record_sha256"}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def comparable(rec):
    return {k: v for k, v in rec.items() if k != "record_sha256" and k not in VOLATILE_FIELDS}


def differing_fields(stored, fresh):
    """-> sorted names of the fields in which `stored` differs from `fresh` (empty = a read of the directory as it is now reproduces `stored`)."""
    a, b = comparable(stored), comparable(fresh)
    dump = lambda v: json.dumps(v, sort_keys=True)                   # noqa: E731
    return sorted(k for k in set(a) | set(b) if k not in a or k not in b or dump(a[k]) != dump(b[k]))


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _canon(v):
    return json.dumps(v, sort_keys=True)


def _jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _rows(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                out.append(json.loads(ln))                       # a line that does not parse makes the whole file unreadable (caught by the caller)
    return out


def arms_file_self_check(eval_dir):
    """PREREG 2.1 self-check for R0e-id, RE-DERIVED from the two arms files (never from the stored flags alone): the argv minus `--epo-con-temp 0.1`, its paths swapped back to /arms/R0d and its seed read as R0d's, equals
    arms.R0d.argv TOKEN FOR TOKEN; the lever is exactly [--epo-con-temp, 0.1] at the end; the stored self-check block says the same.  -> (ok, why)."""
    try:
        e, d = _jload(os.path.join(eval_dir, "rls1e_arms.json")), _jload(os.path.join(eval_dir, "rls1d_arms.json"))
        r0d, a = d["arms"]["R0d"]["argv"], e["arms"][ARM]["argv"]
        if a[-2:] != ["--epo-con-temp", str(REGISTERED_TEMP)]:
            return False, f"R0e-id's argv does not end in --epo-con-temp {REGISTERED_TEMP}: {a[-2:]}"
        rebuilt = [t.replace(f"{POD_RLS1}/arms/{ARM}", f"{POD_RLS1}/arms/R0d") for t in a[:-2]]
        if "--seed" not in rebuilt:
            return False, "R0e-id's argv has no --seed"
        rebuilt[rebuilt.index("--seed") + 1] = "0"
        if rebuilt != r0d:
            diff = [i for i, (x, y) in enumerate(zip(rebuilt, r0d)) if x != y][:3]
            return False, f"R0e-id's argv minus the lever, own paths swapped back, is not R0d's token for token (lengths {len(rebuilt)} vs {len(r0d)}; first differing indices {diff})"
        if e["arms"][ARM].get("registered_con_temp") != REGISTERED_TEMP:
            return False, f"arms file registered_con_temp is {e['arms'][ARM].get('registered_con_temp')!r}, not {REGISTERED_TEMP}"
        sc = (e.get("self_check") or {}).get("per_arm", {}).get(ARM) or {}
        bad = sorted(k for k, v in sc.items() if isinstance(v, bool) and not v)
        if not sc or bad:
            return False, f"the arms file's stored self-check for {ARM} is absent or false on {bad}"
        return True, "ok"
    except Exception as ex:                                                                                              # noqa: BLE001
        return False, f"the arms-file self-check cannot be evaluated ({type(ex).__name__}: {str(ex)[:100]})"


def read_arm(arm_dir, eval_dir=None):
    """-> the verdict record for `arm_dir` (a pure function of the files it hashes)."""
    eval_dir = eval_dir or EVAL_DIR
    fails, probs, vals, chk = [], [], {}, {}
    inputs = {}
    for name in REQUIRED_INPUTS:
        p = os.path.join(arm_dir, name)
        if os.path.isfile(p):
            inputs[name] = sha256_file(p)
        else:
            probs.append(f"{name} is missing")
    pkg = {}
    for rel in PACKAGE_FILES:
        p = os.path.join(eval_dir, os.path.basename(rel))
        if os.path.isfile(p):
            pkg[rel] = sha256_file(p, lf=True)
        else:
            probs.append(f"{rel} is missing from the package")
    ref = None
    try:
        ref = _jload(os.path.join(eval_dir, "r0d_id_reference.json"))
        assert ref.get("schema") == "r0d_id_reference/1" and ref.get("arm") == "R0d"
    except Exception as ex:                                                                                              # noqa: BLE001
        ref = None
        probs.append(f"eval/r0d_id_reference.json cannot be read ({type(ex).__name__})")
    # ---- sft.jsonl: the data event (clauses 1-3), the step0 eval (clause 6), the dump paths (run identity)
    rows = []
    if "sft.jsonl" in inputs:
        try:
            rows = _rows(os.path.join(arm_dir, "sft.jsonl"))
        except Exception as ex:                                                                                          # noqa: BLE001
            probs.append(f"sft.jsonl is not parsable ({type(ex).__name__})")
    data = [r for r in rows if r.get("event") == "data"]
    s0 = [r for r in rows if r.get("event") == "eval" and r.get("tag") == "step0"]
    dumps = [r for r in rows if r.get("event") == "score_dump"]
    if "sft.jsonl" in inputs and not probs_for(probs, "sft.jsonl"):
        if len(data) != 1:
            probs.append(f"sft.jsonl has {len(data)} data events, not exactly one")
        if len(s0) != 1:
            probs.append(f"sft.jsonl has {len(s0)} step0 eval events, not exactly one")
        tags = [r.get("tag") for r in dumps]
        if "step0" not in tags or "update250" not in tags:
            probs.append(f"sft.jsonl lacks the step0 / update250 score_dump events (has {tags})")
        foreign = sorted({str(p) for r in dumps for p in (r.get("files") or {}).values() if f"/arms/{ARM}/" not in str(p).replace("\\", "/")})
        if foreign or not dumps:
            probs.append(f"run identity: score_dump paths outside /arms/{ARM}/ (this directory holds another run's log): {foreign[:2]}")
    if ref is not None and len(data) == 1:
        d = data[0]
        chk["epo_con_block_is_R0d_plus_temp_0.1"] = _canon(d.get("epo_con")) == _canon({**ref["epo_con"], "temp": REGISTERED_TEMP})
        chk["epo_block_is_R0d"] = _canon(d.get("epo")) == _canon(ref["epo"])
        ts = d.get("train_sets")
        chk["train_sets_is_16000"] = _num(ts) and ts == TRAIN_SETS
        for k, v in chk.items():
            if not v:
                fails.append(f"clause: {k} is false")
    if ref is not None and len(s0) == 1:
        for side in ("base", "A", "B"):
            try:
                x, y = float(s0[0][side]["pick"]), float(ref["step0_pick"][side])
                assert math.isfinite(x)
            except Exception:                                                                                            # noqa: BLE001
                probs.append(f"step0 {side}.pick is missing or not a number")
                continue
            vals[f"step0_{side}_pick_minus_R0d"] = x - y
            chk[f"step0_{side}_pick_within_0.10"] = abs(x - y) <= PICK_TOL + EPS
            if not chk[f"step0_{side}_pick_within_0.10"]:
                fails.append(f"clause: step0 {side}.pick differs from R0d's by {abs(x - y):.4f} points (> {PICK_TOL})")
    ok, why = arms_file_self_check(eval_dir)
    chk["arms_file_self_check"] = ok
    if not ok:
        fails.append(f"clause: arms-file self-check: {why}")
    # ---- G13 (clause 5)
    g13_ok = None
    if "PREFLIGHT_PASS.json" in inputs:
        try:
            pf = _jload(os.path.join(arm_dir, "PREFLIGHT_PASS.json"))
            g13 = [v for k, v in (pf.get("gates") or {}).items() if str(k).startswith("G13")]
            if len(g13) != 1 or not isinstance(g13[0], dict):
                probs.append(f"PREFLIGHT_PASS.json carries {len(g13)} G13 entries, not exactly one")
            else:
                g13_ok = bool(pf.get("verdict") == "PASS" and g13[0].get("ok") is True and not g13[0].get("mutate"))
                chk["g13_pass"] = g13_ok
                if not g13_ok:
                    fails.append(f"clause: G13 is not PASS (verdict {pf.get('verdict')!r}, ok {g13[0].get('ok')!r}, mutate {g13[0].get('mutate')!r})")
        except Exception as ex:                                                                                          # noqa: BLE001
            probs.append(f"PREFLIGHT_PASS.json is not parsable ({type(ex).__name__})")
    # ---- ID-DIVERGED: the u250 contrast
    diverged = None
    if U250_NAME in inputs and ref is not None:
        try:
            q = _jload(os.path.join(arm_dir, U250_NAME))
            if q.get("base") != "A" or list((q.get("paired_vs_base") or {})) != ["B"]:
                probs.append(f"{U250_NAME} is not a B-vs-own-A read (base {q.get('base')!r}, paired arms {list(q.get('paired_vs_base') or {})})")
            elif (q.get("eligibility") or {}).get("sets_used") != 3000:
                probs.append(f"{U250_NAME} scored {(q.get('eligibility') or {}).get('sets_used')!r} sets, not the 3,000 of RL-val")
            else:
                for side in ("A", "B"):
                    f = os.path.join(arm_dir, "scores", f"scores_update250_{side}.npz")
                    want = ((q.get("inputs") or {}).get("scores_sha256") or {}).get(side)
                    if not os.path.isfile(f):
                        probs.append(f"scores/scores_update250_{side}.npz is not on disk: the u250 read cannot be tied to this run")
                    elif sha256_file(f) != want:
                        probs.append(f"{U250_NAME} was read from other dumps than this run's scores_update250_{side}.npz (sha256 differs)")
                qq = q["paired_vs_base"]["B"]["quantities"]
                ttc, pdms = float(qq["TTC"]["mean"]), float(qq["PDMS"]["mean"])
                assert math.isfinite(ttc) and math.isfinite(pdms)
                rc = ref["u250_contrast_B_minus_ownA_x100"]
                vals["u250_TTC_contrast"], vals["u250_PDMS_contrast"] = ttc, pdms
                vals["u250_TTC_delta_vs_R0d"], vals["u250_PDMS_delta_vs_R0d"] = ttc - rc["TTC"], pdms - rc["PDMS"]
                diverged = abs(ttc - rc["TTC"]) > DIV_TOL + EPS or abs(pdms - rc["PDMS"]) > DIV_TOL + EPS
                chk["u250_contrast_within_0.50_of_R0d"] = not diverged
        except Exception as ex:                                                                                          # noqa: BLE001
            probs.append(f"{U250_NAME} is not a readable B-vs-own-A read ({type(ex).__name__}: {str(ex)[:80]})")
    # ---- a PASS must have EVALUATED every clause: a clause that never ran (its input absent) is a problem, not a pass
    if not fails:
        probs.extend(f"clause {k} was not evaluated" for k in EXPECTED_CHECKS if k not in chk)
    # ---- the verdict (the order of the prereg; a definite failure wins over an unreadable input)
    if fails:
        outcome = "ID-FAIL"
    elif probs:
        outcome = "NOT_READ"
    elif diverged:
        outcome = "ID-DIVERGED"
    else:
        outcome = "ID-PASS"
    return {"tool": TOOL, "arm": ARM, "outcome": outcome, "fail_reasons": fails, "problems": probs, "checks": chk, "values": vals,
            "inputs_sha256": inputs, "package_sha256": pkg, "thresholds": {"step0_pick_points": PICK_TOL, "u250_contrast_points": DIV_TOL}}


def probs_for(probs, name):
    return [p for p in probs if p.startswith(name)]


def _why(rec):
    return "; ".join(rec["fail_reasons"] + rec["problems"])[:300] or "every clause evaluated, none fired"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm-dir", required=True)
    ap.add_argument("--no-write", action="store_true")
    try:
        a = ap.parse_args(argv)
    except SystemExit:
        return 3
    rec = read_arm(a.arm_dir)
    rec["record_sha256"] = record_digest(rec)
    if not a.no_write:
        with open(os.path.join(a.arm_dir, OUT_NAME), "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1, sort_keys=True)
            f.write("\n")
    print(f"ZZR0EID {ARM} {rec['outcome']} {_why(rec)}", flush=True)
    return {"ID-PASS": 0, "ID-FAIL": 1, "ID-DIVERGED": 1}.get(rec["outcome"], 2)


if __name__ == "__main__":
    sys.exit(main())
