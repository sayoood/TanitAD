#!/usr/bin/env python3
"""Derive box2_ship/v8f_dagoaccorc_r2_chain_checks.py from the r1 checks file (exact-count insertions only; r1's file is never touched and its K-REACH / one-lever / run-record code is carried BYTE FOR BYTE):

  r1 = E:/Projects/TanitAD-artifacts/dagorc_rebase/chain/box2_ship/v8f_dagoaccorc_chain_checks.py   (installed as C:/Users/Admin/box2_ship/v8f_dagoaccorc_chain_checks.py, sha256 cff827e3...)

  ADDED (r2):
    1. pred-verdict-r2 --out LOG --busy FILE     the r2 predecessor rule: GO only when the r1 DAG-ORC chain log ENDS in exactly the K-REACH stop for M2b_s0_dagoaccorc (and has no DONE, no other STOP, no guard stop, no disk-floor stop);
                                                 WAIT while it has no end marker; BUSY while BOX2_BUSY is held after that stop; HALT for everything else. (r1's installed pred-verdict, shared by every chain, would GO after ANY predecessor STOP.)
    2. run-record --stream-record SPEC           the separated perturbation stream recorded in the run's summary.json (A: present / equal to the literal; REP: absent). SPEC = NONE | <dotted.path> | <dotted.path>=<json literal>.
                                                 Unset ("<<UNSET") fails closed; NONE skips (the landed fix records no field).
  UNCHANGED: run-record's other clauses, reached (K-REACH bands), one-lever, pin-sha, argv-sha.
    python build_r2_checks.py          writes ../box2_ship/v8f_dagoaccorc_r2_chain_checks.py (LF)
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R1 = Path("E:/Projects/TanitAD-artifacts/dagorc_rebase/chain/box2_ship/v8f_dagoaccorc_chain_checks.py")
OUT = HERE.parent / "box2_ship" / "v8f_dagoaccorc_r2_chain_checks.py"


def rep(t: str, old: str, new: str, n: int = 1) -> str:
    assert t.count(old) == n, (old[:80], t.count(old))
    return t.replace(old, new)


NEW_DOC = '''
r2 (the re-run after the r1 chain stopped on K-REACH, the perturbation share 0.2492 vs 0.3 +- 0.02: the swap and the perturbation shared ONE RNG stream) ADDS two things to the r1 checks (every other clause is r1's, byte for byte):
    pred-verdict-r2 --out LOG --busy FILE
        GO only when LOG (the r1 DAG-ORC chain's log) ENDS in exactly  "ZZDAGOACCORC-CONTROL-FAIL M2b_s0_dagoaccorc: ..." then "ZZV8FMINI-STOP reached-control fail M2b_s0_dagoaccorc"  (one STOP, no DONE, no guard stop, no disk-floor stop, a K-REACH FAIL
        line for dagoaccorc) and BOX2_BUSY is released.  WAIT: no end marker yet.  BUSY: the right stop but BOX2_BUSY.txt still held (the caller retries, then HALTs).  HALT: anything else.  Prints  <VERDICT> <reason>.
    run-record ... --stream-record SPEC
        SPEC = NONE (the landed fix records no field: skipped) | <dotted.path> | <dotted.path>=<A json> | <dotted.path>=<A json>;<REP json>, read in summary.json AND in the run's own selfcheck.json.
        A: the field is present and (equal to the A literal | not a "no stream" value).  REP: with a REP literal, present and EXACTLY it (typed; absent fails); without, absent or a "no stream" value.
        "no stream" values: null, false, "", "none", "off", "shared".  A SPEC that still contains "<<" (the placeholder) is a failure.
        The registered spec (Master Mind, 2026-10-28, from the staged trainer fix: v8f_mini_train.py:1739 records levers["ego_perturb_stream"] = comp.ego.stream):   levers.ego_perturb_stream=1;0
'''

NEW_FUNCS = '''
# ---------------------------------------------------------------------------------------------------------------- r2 additions
EXPECTED_R1_STOP = "ZZV8FMINI-STOP reached-control fail M2b_s0_dagoaccorc"
EXPECTED_R1_CTRL_PREFIX = "ZZDAGOACCORC-CONTROL-FAIL M2b_s0_dagoaccorc:"
NO_STREAM_STR = ("", "none", "off", "shared")


def _no_stream(v) -> bool:
    return v is None or v is False or (isinstance(v, str) and v.strip().lower() in NO_STREAM_STR)


def stream_problems(sm, p, spec) -> list:
    """The separated perturbation stream in a run record (summary.json or selfcheck.json; see the module doc). [] = fine.
    SPEC = <path>=<A literal>;<REP literal>  -> A: present and EXACTLY the A literal; REP: present and EXACTLY the REP literal (typed: 1 is not true, 1 is not 1.0; ABSENT fails on both arms)."""
    spec = (spec or "").strip()
    if spec in ("", "NONE"):
        return []
    if "<<" in spec:
        return [f"the stream-record spec {spec!r} is still a placeholder (the Master Mind sets it from the landed trainer fix)"]
    path, _, lits = spec.partition("=")
    lit, rep_sep, lit_rep = lits.partition(";")
    cur, found = sm, True
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            found = False
            break
    if p > 0.0:
        if not found:
            return [f"the run record carries no {path}: the separated perturbation stream is not recorded on A"]
        if lit:
            try:
                want = json.loads(lit)
            except ValueError:
                return [f"the stream-record literal {lit!r} is not JSON"]
            if json.dumps(cur, sort_keys=True) != json.dumps(want, sort_keys=True):
                return [f"the run record's {path} = {cur!r} on A, the registered {want!r} (the separated stream)"]
            return []
        if _no_stream(cur):
            return [f"the run record's {path} = {cur!r} on A: that is not a separated stream"]
        return []
    if rep_sep:
        try:
            want = json.loads(lit_rep)
        except ValueError:
            return [f"the stream-record REP literal {lit_rep!r} is not JSON"]
        if not found:
            return [f"the run record carries no {path}: REP must record exactly {want!r}"]
        if json.dumps(cur, sort_keys=True) != json.dumps(want, sort_keys=True):
            return [f"the run record's {path} = {cur!r} on REP, the registered {want!r}"]
        return []
    if found and not _no_stream(cur):
        return [f"the run record's {path} = {cur!r} on REP (the replicate perturbs nothing: the separated stream must be absent)"]
    return []


def pred_verdict_r2(out_text: str, busy_exists: bool) -> tuple:
    """The r2 predecessor rule. The r1 chain log must END in exactly the K-REACH stop of M2b_s0_dagoaccorc and nothing else. Markers are ANCHORED at the start of a line."""
    lines = [x.rstrip() for x in out_text.replace("\\r", "").split("\\n")]
    body = [x for x in lines if x.strip()]
    done = [x for x in lines if x.startswith("ZZV8FMINI-DONE")]
    stop = [x for x in lines if x.startswith("ZZV8FMINI-STOP")]
    guard = [x for x in lines if "ZZGUARD-STOP" in x]
    floor = [x for x in lines if "ZZDISK-FLOOR-STOP" in x]
    ctrl = [x for x in lines if x.startswith("ZZDAGOACCORC-CONTROL-FAIL")]
    kfail = [x for x in lines if "K-REACH: FAIL | dagoaccorc |" in x]
    if guard:
        return "HALT", "the r1 chain was stopped by the GUARD (" + guard[-1][:160] + "): the box may be hot"
    if done:
        return "HALT", "the r1 chain ended DONE (" + done[-1][:120] + "): r2 re-runs only after the r1 K-REACH stop"
    if not stop:
        return "WAIT", "no end marker yet"
    if floor:
        return "HALT", "the r1 chain hit a disk-floor stop (" + floor[-1][:120] + "), not the K-REACH stop"
    if len(stop) != 1:
        return "HALT", f"{len(stop)} ZZV8FMINI-STOP lines in the r1 log (exactly one is expected)"
    if not body or body[-1] != EXPECTED_R1_STOP:
        if stop[0] != EXPECTED_R1_STOP:
            return "HALT", "the r1 chain stopped on something else: " + stop[0][:200] + "  (r2 accepts only: " + EXPECTED_R1_STOP + ")"
        return "HALT", "the r1 log does not END in the K-REACH stop (last line: " + (body[-1][:160] if body else "<empty>") + ")"
    if len(ctrl) != 1 or len(body) < 2 or body[-2] != ctrl[0] or not ctrl[0].startswith(EXPECTED_R1_CTRL_PREFIX):
        return "HALT", "the line before the stop is not the one ZZDAGOACCORC-CONTROL-FAIL M2b_s0_dagoaccorc: line"
    if not kfail:
        return "HALT", "the r1 log has no 'K-REACH: FAIL | dagoaccorc |' line for the stop"
    if busy_exists:
        return "BUSY", "the r1 chain stopped on the K-REACH stop but its BOX2_BUSY.txt is not released (yet)"
    return "GO", "the r1 chain ended on its K-REACH stop for M2b_s0_dagoaccorc and released the box: " + EXPECTED_R1_STOP
'''


def derive(src: str) -> str:
    t = src
    t = rep(t, '"""PREREG_V8F_DAG_ORC Amendment 1 (re-based on DAG-OACC) on box 2:', '"""[r2 of the DAG-ORC chain] PREREG_V8F_DAG_ORC Amendment 1 (re-based on DAG-OACC) on box 2:')
    t = rep(t, "    pin-sha FILE / argv-sha LINEFILE      sha256 of a file / launch_gate.argv_sha256 of the training argv of a printed line\n\"\"\"\n",
            "    pin-sha FILE / argv-sha LINEFILE      sha256 of a file / launch_gate.argv_sha256 of the training argv of a printed line\n" + NEW_DOC + "\"\"\"\n")
    # the stream clause: right before the dagger_p clause of run_record (after the composition-flag clauses)
    t = rep(t, '    if lv.get("dagger_p") != G.SWAP_P:\n', '    bad += stream_problems(sm, p, getattr(a, "stream_record", ""))      # r2: the separated perturbation stream (summary.json) ...\n'
            '    bad += ["selfcheck.json: " + x for x in stream_problems(sc, p, getattr(a, "stream_record", ""))]      # ... and the run\'s own selfcheck.json\n    if lv.get("dagger_p") != G.SWAP_P:\n')
    # r2 asserter: the checks load the r2 gate asserter (r1's is untouched and keeps its name)
    t = rep(t, '_spec = importlib.util.spec_from_file_location("gate_dagoaccorc_assert", os.path.join(HERE, "gate_dagoaccorc_assert.py"))',
            '_spec = importlib.util.spec_from_file_location("gate_dagoaccorc_r2_assert", os.path.join(HERE, "gate_dagoaccorc_r2_assert.py"))      # r2: the asserter with ego_stream_problems')
    t = rep(t, "def main(argv=None) -> int:\n", NEW_FUNCS.lstrip("\n") + "\n\ndef main(argv=None) -> int:\n")
    t = rep(t, '    r.add_argument("--expect", default="")\n', '    r.add_argument("--expect", default="")\n    r.add_argument("--stream-record", default="")      # r2\n')
    t = rep(t, '    sub.add_parser("pin-sha").add_argument("file")\n',
            '    pv = sub.add_parser("pred-verdict-r2")      # r2\n    pv.add_argument("--out", required=True)\n    pv.add_argument("--busy", required=True)\n    sub.add_parser("pin-sha").add_argument("file")\n')
    t = rep(t, '    if a.cmd == "pin-sha":\n',
            '    if a.cmd == "pred-verdict-r2":      # r2\n        v, why = pred_verdict_r2(open(a.out, encoding="utf-8", errors="replace").read() if os.path.exists(a.out) else "", os.path.exists(a.busy))\n        print(v, why)\n        return 0\n    if a.cmd == "pin-sha":\n')
    return t


def main() -> int:
    t = derive(R1.read_text(encoding="utf-8"))
    assert "\r" not in t
    OUT.write_bytes(t.encode("utf-8"))
    print("wrote", OUT, len(t), "bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
