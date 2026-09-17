"""MUTATION PROOF for every refcv6 §4/§5 guard — never inspection.

⛔⛔ WHY THIS FILE EXISTS RATHER THAN A CENSUS. The programme's own measured
lesson (2026-09-07, four instances in one night): *"a check that shares the
defect it checks for is green forever"*, and an AST census read **0 suspects
on BOTH the fixed and the broken trainer**. So nothing here inspects source.
Every row BUILDS a real argv through the REAL parser, runs the REAL pin
function (or the REAL reader), and records what actually happened.

⭐ EVERY REFUSAL SHIPS ITS **GREEN CONTROL** — the nearest argv that must
PASS. A guard that refuses everything is not a guard, and a RED-only table
cannot tell the two apart. The pairs are adjacent in the output on purpose.

Usage:  python refusals.py            # prints JSON to stdout
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.environ.get("TAC_WT", ""), "stack"))
sys.path.insert(0, os.path.join(os.environ.get("TAC_WT", ""), "stack",
                                "scripts"))

import refc_v3_train as T                                    # noqa: E402
from tanitad.refs import refcv6_max_speed as v6ms            # noqa: E402
from tanitad.refs import refc_v3 as v3                       # noqa: E402

LABELS = os.environ["TAC_LABELS"]
SIDECAR = os.environ["TAC_SIDECAR"]
JOIN = os.environ["TAC_JOIN"]

#: The argv every case starts from — a VALID refcv6 tactical arm. Each case
#: mutates exactly one thing, so the refusal is attributable to that one thing.
#:
#: ⚠️ `--agent-join` IS LOAD-BEARING HERE AND IT WAS MISSING IN THE FIRST
#: VERSION OF THIS FILE. Without it a PRE-EXISTING guard (`--agents head`
#: without labels) refused the base arm, so all four GREEN controls read
#: REFUSED — while all fifteen REDs read REFUSED too. The table would have
#: been 'every refusal fires' with a brick, not a guard, underneath. ⇒ This is
#: precisely what the green controls are for, and why a RED-only table is not
#: evidence.
BASE = [
    "--arm", "hier", "--out", "UNUSED", "--device", "cpu",
    "--v7-labels", LABELS,
    "--agents", "head", "--w-agent", "1.0",
    "--agent-join", JOIN, "--agent-join-verify", "off",
    "--tac-decoder-v6", "--w-tac-v6", "1.0",
]


def drop(argv, flag, n_values=1):
    """Remove ``flag`` (and its ``n_values`` values) from an argv list.

    ⛔ EXPLICIT, not index slicing. The first version of this file mutated the
    list by position and silently produced argv that argparse rejected as
    malformed — every case then 'REFUSED' for the wrong reason and the table
    would have read as a perfect score. A mutation harness that can corrupt
    its own input is the check-shares-the-defect failure in test costume.
    """
    out, i = [], 0
    while i < len(argv):
        if argv[i] == flag:
            i += 1 + n_values
            continue
        out.append(argv[i])
        i += 1
    if len(out) == len(argv):
        raise AssertionError(
            f"drop({flag!r}) removed NOTHING — the mutation did not apply, so "
            f"the case would test the unmutated argv and pass vacuously.")
    return out


def setval(argv, flag, value):
    """Replace ``flag``'s value, asserting the flag was present."""
    out = list(argv)
    if flag not in out:
        raise AssertionError(f"setval({flag!r}) — flag absent from base argv")
    out[out.index(flag) + 1] = value
    return out


def _pin(argv):
    """Run the REAL parser and the REAL pin. Returns (ok, message)."""
    ap = T.build_parser()
    args = ap.parse_args(argv)
    cfg = v3.RefCV3Config(hier=(args.arm == "hier"))
    try:
        T._pin_trainer_cfg(cfg, args)
        return True, None
    except SystemExit as exc:
        return False, str(exc)
    except Exception as exc:                      # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def case(name, argv, expect, why):
    ok, msg = _pin(argv)
    got = "PASS" if ok else "REFUSED"
    return {"case": name, "expect": expect, "got": got,
            "verdict": "OK" if got == expect else "⛔ WRONG",
            "why": why,
            "message": (msg or "")[:400]}


rows = []

# ---- §4: the core refusal and its green control ------------------------- #
rows.append(case(
    "GREEN  decoder + live weight", BASE, "PASS",
    "the valid arm — if this refuses, every RED below is meaningless"))
rows.append(case(
    "RED    decoder with w=0", setval(BASE, "--w-tac-v6", "0"), "REFUSED",
    "THE defect: 2,262,020 params built, stamped, and unable to learn"))
rows.append(case(
    "RED    weight without decoder",
    drop(BASE, "--tac-decoder-v6", 0), "REFUSED",
    "no decoder => no `tacv6_goal_logits` for the loss to read"))
rows.append(case(
    "RED    decoder on a FLAT arm", setval(BASE, "--arm", "flat"), "REFUSED",
    "the agent slots and the selection seam exist only under --arm hier"))
rows.append(case(
    "RED    decoder without --v7-labels (kin3 vocab)",
    drop(BASE, "--v7-labels"), "REFUSED",
    "kin3 has no 22-token goal set: the behaviour queries have no label"))
rows.append(case(
    "RED    decoder AND the old tac_goal_tok_head",
    BASE + ["--tac-goal-tok-head"], "REFUSED",
    "two heads on one target = two experiments, non-attributable"))
rows.append(case(
    "RED    --tac-decoder-d-bev > 0 (the BLOCKED map half)",
    BASE + ["--tac-decoder-d-bev", "256"], "REFUSED",
    "no BEV token reaches the hook; a declared-but-absent source would read "
    "as 'the map adds nothing'"))
rows.append(case(
    "GREEN  --tac-decoder-d-bev 0 is the agent-only arm",
    BASE + ["--tac-decoder-d-bev", "0"], "PASS",
    "the control for the row above — 0 is the buildable arm"))
rows.append(case(
    "RED    --graft-behaviour-sel without the decoder",
    drop(drop(BASE, "--tac-decoder-v6", 0), "--w-tac-v6")
    + ["--graft-behaviour-sel"], "REFUSED",
    "the selection term has no producer: a silently inert flag"))
rows.append(case(
    "GREEN  --graft-behaviour-sel WITH the decoder",
    BASE + ["--graft-behaviour-sel"], "PASS",
    "the control for the row above"))

# ---- §5: the 4-way set-speed -------------------------------------------- #
rows.append(case(
    "RED    --max-speed-input-v6 without a sidecar",
    BASE + ["--max-speed-input-v6"], "REFUSED",
    "the 4-value ladder is NOT a label field; with no sidecar the 4 slots "
    "are a constant all-zero pad"))
rows.append(case(
    "GREEN  --max-speed-input-v6 WITH a sidecar",
    BASE + ["--max-speed-input-v6", "--speed-max-sidecar-v6", SIDECAR],
    "PASS", "the control for the row above"))
rows.append(case(
    "RED    E16 and refcv6 ceilings together",
    BASE + ["--max-speed-input-v6", "--speed-max-sidecar-v6", SIDECAR,
            "--max-speed-input"], "REFUSED",
    "two ladders over two source fields: the channel becomes "
    "non-attributable"))
rows.append(case(
    "RED    --max-speed-input-v6 without the decoder",
    drop(drop(BASE, "--tac-decoder-v6", 0), "--w-tac-v6")
    + ["--max-speed-input-v6", "--speed-max-sidecar-v6", SIDECAR],
    "REFUSED", "the one-hot's only consumer is the decoder's condition"))

# ---- the SIDECAR reader's own guards ------------------------------------ #


def _scrub(msg: str, d: str) -> str:
    """⛔ THE READER'S MESSAGES NAME THE FILE, AND THE FILE LIVES IN A TEMP DIR
    UNDER THE USER'S HOME. Recording them verbatim puts an absolute home path
    into a repo artifact, which `tests/test_refcv6_no_session_paths.py`
    forbids. Scrubbed at the PRODUCER, so re-running this file cannot
    reintroduce it — patching only the banked JSON would leave the next run
    broken in the same way."""
    # ⚠️ THE REPR FORM IS THE ONE THAT ACTUALLY APPEARS. The reader formats the
    # path with `{path!r}`, which DOUBLES every backslash — so scrubbing only
    # the raw path silently misses it, and the artifact still carries the home
    # path while this function reports success. MEASURED: the first version of
    # this scrub did exactly that.
    b = d.replace("/", "\\")
    for v in (d, b, b.replace("\\", "\\\\"), d.replace("\\", "/")):
        msg = msg.replace(v, "<TMP>")
    return msg


def reader_case(name, mutate, expect, why):
    """Write a MUTATED copy of the real sidecar and read it back."""
    with open(SIDECAR, "r", encoding="utf-8") as fh:
        lines = [json.loads(x) for x in fh if x.strip()]
    lines = mutate(lines)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "mutant.jsonl")
    with open(p, "w", encoding="utf-8") as fh:
        for r in lines:
            fh.write(json.dumps(r) + "\n")
    try:
        by_sid, rep = v6ms.read_speed_max_sidecar_v6(p)
        return {"case": name, "expect": expect, "got": "PASS",
                "verdict": "OK" if expect == "PASS" else "⛔ WRONG",
                "why": why, "message": f"n_valid={rep['n_valid']}"}
    except SystemExit as exc:
        return {"case": name, "expect": expect, "got": "REFUSED",
                "verdict": "OK" if expect == "REFUSED" else "⛔ WRONG",
                "why": why, "message": _scrub(str(exc), d)[:400]}


rows.append(reader_case(
    "GREEN  the real sidecar reads", lambda ls: ls, "PASS",
    "if this refuses, every reader RED below is meaningless"))
rows.append(reader_case(
    "RED    a clip_id-keyed sidecar (no `sid`)",
    lambda ls: [{k: v for k, v in r.items() if k != "sid"} for r in ls],
    "REFUSED",
    "those rows can never join; and the message must name THE SID, not the "
    "downstream 'no valid ceiling' symptom"))
rows.append(reader_case(
    "RED    a sidecar whose `bin` disagrees with the ladder",
    lambda ls: [dict(r, bin=(0 if r.get("bin") else 3)) for r in ls],
    "REFUSED", "two ladders is two experiments"))
rows.append(reader_case(
    "RED    an all-invalid sidecar", lambda ls: [dict(r, valid=0) for r in ls],
    "REFUSED", "the channel would be a constant pad measured as noise"))
rows.append(reader_case(
    "RED    an EMPTY sidecar", lambda ls: [], "REFUSED",
    "an empty read and a genuine absence of ceilings are different facts"))

# ⭐ THE MESSAGE-CONTENT ASSERTION for the ordering defect MEASURED while
# wiring this: on a `sid`-less sidecar the reader used to report *"NOT ONE of
# the 147 rows carries a valid ceiling"*, which was FALSE — all 147 did. A
# guard may not name a symptom as its cause.
_sidless = [r for r in rows if r["case"].startswith("RED    a clip_id-keyed")]
rows.append({
    "case": "ASSERT the sid-less message names the CAUSE, not the symptom",
    "expect": "names `sid`, not `valid ceiling`",
    "got": ("names `sid`"
            if ("`sid`" in _sidless[0]["message"]
                and "valid ceiling" not in _sidless[0]["message"])
            else "names the SYMPTOM"),
    "verdict": ("OK" if ("`sid`" in _sidless[0]["message"]
                         and "valid ceiling" not in _sidless[0]["message"])
                else "⛔ WRONG"),
    "why": "a true-sounding message naming the wrong cause sends the reader "
           "to rebuild the labels instead of re-keying the sidecar",
    "message": _sidless[0]["message"][:200]})

# ---- the STAMP, both directions ----------------------------------------- #
for on, cfgd, exp, why in (
        (True, {}, "REFUSED", "channel ON, no stamp -> an arm whose record "
                              "cannot say what fed it"),
        (True, {"speed_max_derivation_v6": v6ms.SPEED_MAX_DERIVATION_V6},
         "PASS", "the control: ON with the stamp"),
        (False, {"speed_max_derivation_v6": v6ms.SPEED_MAX_DERIVATION_V6},
         "REFUSED", "the MIRROR: a control stamped as conditioned"),
        (False, {}, "PASS", "the control: OFF with no stamp"),
        (True, {"speed_max_derivation_v6": "4-way one-hot, some ladder"},
         "REFUSED", "a stamp missing the required tokens is a key, not a "
                    "declaration")):
    try:
        v6ms.assert_speed_max_stamp_v6(cfgd, on)
        got, msg = "PASS", ""
    except SystemExit as exc:
        got, msg = "REFUSED", str(exc)[:200]
    rows.append({"case": f"STAMP on={on} stamp={bool(cfgd)}",
                 "expect": exp, "got": got,
                 "verdict": "OK" if got == exp else "⛔ WRONG",
                 "why": why, "message": msg})

out = {
    "n_cases": len(rows),
    "n_ok": sum(1 for r in rows if r["verdict"] == "OK"),
    "n_wrong": sum(1 for r in rows if r["verdict"] != "OK"),
    "n_green_controls": sum(1 for r in rows
                            if str(r["case"]).startswith("GREEN")
                            or r.get("expect") == "PASS"),
    "⛔": "every RED is paired with the nearest GREEN that must pass; a "
         "table of refusals alone cannot distinguish a guard from a brick",
    "cases": rows,
}
print(json.dumps(out, indent=1, ensure_ascii=False))
sys.exit(1 if out["n_wrong"] else 0)
