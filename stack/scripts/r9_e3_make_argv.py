#!/usr/bin/env python3
"""r9_e3_make_argv.py -- the refcv9 E3 full-run argv and its D1 early-branch argv, composed from the E2 GO arm's argv doc.

  E3   profile ``refcv9-full``     = the E2 arm's argv with ONLY the full-run schedule rows changed (REFCV9_DESIGN 4.4: ``--steps 73280``, ``--warmup 1000``, ``--wsd-decay-steps 12458``),
                                     ``--save-at`` ADDED (the stable-phase full-state checkpoint ``ckpt_full_30000.pt`` the D1 branch resumes) and ``--out`` set to the run directory the
                                     operator passes. Every other token is the E2 source's, in the E2 source's order (``--init-from``, ``--w-r8-drivable`` and the lever block included).
  D1   profile ``refcv9-full-d1``  = the E3 argv with EXACTLY the gate's 4-flag delta (``launch_gate._R9F_BRANCH["delta"]``): ``--out`` (the D1 run directory), ``--steps 36000`` (= 30,000 + 6,000),
                                     ``--wsd-branch-from <E3 --out>/ckpt_full_30000.pt``, ``--wsd-branch-decay 6000``. ``--save-at`` is the one EXEMPT cadence flag
                                     (``_R9F_BRANCH["exempt"]``): the D1 argv carries none -- the trainer REFUSES a ``--save-at`` step <= the branch's start step as "never written"
                                     (``lr_schedule.check_branch_checkpoint``), and E3's list names 30000 -- and profile ``refcv9-full-d1`` does not require the flag.

The E2 source is an ``r9e2-argv-v1`` doc (``stack/ops/runs.d/r9e2_<V>.argv.json``, profile ``refcv9-e2``) as ``r9_e2_make_argv.py`` writes it. The docs this file writes keep that layout (``dump`` = json
indent 1 + a closing newline, LF writes) with schema ``r9e3-argv-v1``; the source's ``lever`` block and its W-COST stamp (``w_cost_ratio`` / ``w_cost_source``, or ``w_cost: "provisional"``) are CARRIED THROUGH
unchanged, and ``e2_source`` records the source doc's path (as passed) + the sha256 of its file bytes. ``--w-r8-drivable`` is never recomputed here: whatever weight the E2 doc holds is the weight E3 trains with.
The gate does not trust the stamp -- G-LIVE (``launch_gate.wcost_reasons``) holds the weight to 0.1 / R of ITS OWN smoke -- so a ``provisional`` source yields a NOTE here, never a token.

  python r9_e3_make_argv.py --e2 <r9e2_V.argv.json> --out /home/nvidia/r8ns/r9e3/runs/r9e3-<V> --write-e3 <r9e3_V.argv.json> --write-d1 <r9e3_V_d1.argv.json> [--d1-out <D1 run dir>] [--save-at 30000[,...]]
  python r9_e3_make_argv.py ... --check          # regenerate and require the two files BYTE-EQUAL; writes nothing

``--d1-out`` defaults to ``<E3 --out>-d1``. ``--save-at`` defaults to ``30000`` and must include it (parsed by ``lr_schedule.parse_save_at``, written sorted). Run from git-bash with
``MSYS_NO_PATHCONV=1``: a ``/home/...`` argument is otherwise rewritten to ``C:/Program Files/Git/...`` (``r9_e1_make_argv._thor_path`` refuses that form).
Nothing here edits an existing file; nothing in ``stack/ops/runs.d/`` is written unless the operator names it.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import r9_e2_make_argv as E2  # noqa: E402  (dump / write / pairs / flat and the E2 constants: reused, never copied)
from tanitad.train import lr_schedule as LRS  # noqa: E402  (parse_save_at / FULL_CKPT_FMT: the trainer's own, pure stdlib)

E1 = E2.E1                                  # r9_e1_make_argv: split_groups / _thor_path
SCHEMA = "r9e3-argv-v1"
E3_PROFILE, D1_PROFILE = "refcv9-full", "refcv9-full-d1"
#: the full-run schedule rows (profile refcv9-full required_values) -- pinned to the gate by tests/test_r9_e3_make_argv.py
E3_STEPS, E3_WARMUP, E3_DECAY = 73280, 1000, 12458
#: the D1 branch (launch_gate._R9F_BRANCH): the stable-phase step, the decay length, the branch's step count
SRC_STEP, D1_DECAY = 30000, 6000
D1_STEPS = SRC_STEP + D1_DECAY
SCHEDULE_ROWS = (("--steps", str(E3_STEPS)), ("--warmup", str(E3_WARMUP)), ("--wsd-decay-steps", str(E3_DECAY)))
#: E3's gradient-share readings (REFCV9_DESIGN 4.5 costs 12): E2's 1000,3000,5000 would cover only the first 7 % of the run. All <= D1_STEPS because the trainer refuses a reading past --steps
#: (grad_share.parse_at) and D1 must stay the gate's exact 4-flag delta (_R9F_BRANCH), so no reading exists past 36,000; the in-run evals (--eval-every 500) cover the rest of the run.
GRAD_SHARE_AT = "1000,3000,6000,9000,12000,15000,18000,21000,24000,27000,30000,36000"
BRANCH_FLAGS = ("--wsd-branch-from", "--wsd-branch-decay")
SAVE_AT = "--save-at"
W_COST_KEYS = ("w_cost", "w_cost_ratio", "w_cost_source")
CRITIC, W_CRITIC = "--r8-critic-drivable", "--w-r8-drivable"
_ARM = re.compile(r"^r9e2_([A-Za-z0-9]+)$")
_TAG = "[r9-e3-argv]"


def _die(msg: str):
    raise SystemExit("%s %s" % (_TAG, msg))


def _posix(p) -> str:
    return str(p).replace("\\", "/")


def dump(rec: dict) -> str:
    return E2.dump(rec)


def write(path, rec: dict) -> None:
    E2.write(path, rec)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pairs(argv):
    try:
        return E2.pairs(argv)
    except ValueError as e:
        _die(str(e))


def _one(ps, flag: str) -> list:
    got = [v for f, v in ps if f == flag]
    if len(got) != 1:
        _die("the argv must carry %s exactly once (has %d)" % (flag, len(got)))
    return got[0]


def flag_value(argv, flag: str) -> str:
    """The flag's values joined by a space (the flag must occur exactly once)."""
    return " ".join(_one(pairs(argv), flag))


def _run_dir(p: str, what: str) -> str:
    """A run directory ON THOR: absolute POSIX, no trailing slash, no whitespace (the branch path is ``<dir>/ckpt_full_<s>.pt``)."""
    q = E1._thor_path(p, what)
    if q != q.rstrip("/") or q == "" or re.search(r"\s", q) or "//" in q or "/../" in q + "/":
        _die("%s %r must be a clean absolute POSIX run directory (no trailing slash, whitespace, '//' or '..')" % (what, p))
    return q


def save_at_text(spec) -> str:
    """``--save-at`` for E3: ``lr_schedule.parse_save_at`` against E3's step count (integers in [1, 73280], no duplicate), MUST name 30000 (the D1 branch's source), written sorted."""
    steps = LRS.parse_save_at(spec, E3_STEPS)
    if SRC_STEP not in steps:
        _die("--save-at %r does not name %d: the D1 branch has no stable-phase checkpoint (%s) to decay" % (spec, SRC_STEP, LRS.FULL_CKPT_FMT % SRC_STEP))
    return ",".join(str(s) for s in sorted(steps))


def _check_e2(e2_doc: dict) -> tuple:
    """The E2 source: an ``r9e2-argv-v1`` doc, profile refcv9-e2, an E2-shaped schedule, none of the E3 / D1 flags yet. -> (argv, variant)."""
    if not isinstance(e2_doc, dict) or e2_doc.get("schema") != "r9e2-argv-v1" or e2_doc.get("profile") != "refcv9-e2":
        _die("--e2 must be an r9e2-argv-v1 doc of profile refcv9-e2 (got schema %r, profile %r)" % (
            e2_doc.get("schema") if isinstance(e2_doc, dict) else None, e2_doc.get("profile") if isinstance(e2_doc, dict) else None))
    m = _ARM.match(str(e2_doc.get("arm")))
    if not m:
        _die("--e2's arm %r is not r9e2_<variant>" % (e2_doc.get("arm"),))
    argv = e2_doc.get("argv")
    if not (isinstance(argv, list) and argv and all(isinstance(t, str) for t in argv)):
        _die("--e2 carries no argv list of strings")
    eq = [t for t in argv if t.startswith("--") and "=" in t]
    if eq:
        _die("the argv has --flag=value token(s) %s: the composer edits whole 'flag value' groups; write them as two tokens" % eq)
    ps = pairs(argv)
    for f in (f for f, _ in ps if f in (SAVE_AT, *BRANCH_FLAGS)):
        _die("the E2 source already carries %s -- E3 is composed from an E2 argv, not from an E3 / D1 one" % f)
    for f in ("--out", "--init-from", "--lr-schedule", "--steps", "--warmup", "--wsd-decay-steps"):
        _one(ps, f)
    if _one(ps, "--lr-schedule") != ["wsd"]:
        _die("the E2 source's --lr-schedule is %s, not ['wsd']" % _one(ps, "--lr-schedule"))
    if (_one(ps, "--steps"), _one(ps, "--wsd-decay-steps")) != ([str(E2.E2_STEPS)], [str(E2.WSD_DECAY)]):
        _die("the E2 source's --steps / --wsd-decay-steps are %s / %s, not E2's %d / %d" % (_one(ps, "--steps"), _one(ps, "--wsd-decay-steps"), E2.E2_STEPS, E2.WSD_DECAY))
    return list(argv), m.group(1)


def compose_e3(e2_doc: dict, out: str, save_at=str(SRC_STEP)) -> list:
    """The E2 argv with ``--steps`` / ``--warmup`` / ``--wsd-decay-steps`` / ``--grad-share-at`` / ``--out`` replaced IN PLACE by the full-run rows and ``--save-at`` appended. Nothing else moves."""
    argv, _ = _check_e2(e2_doc)
    out = _run_dir(out, "--out")
    ps = pairs(argv)
    if out == _one(ps, "--out")[0]:
        _die("--out %s is the E2 source's own run directory: E3 is a new run" % out)
    _one(ps, "--grad-share-at")                      # the E2 source must carry the flag: it is replaced in place, never added
    sets = dict(SCHEDULE_ROWS, **{"--out": out, "--grad-share-at": GRAD_SHARE_AT})
    res = [(f, [sets[f]] if f in sets else v) for f, v in ps]
    return E2.flat(res + [(SAVE_AT, [save_at_text(save_at)])])


def compose_d1(e3_argv: list, d1_out: str | None = None) -> list:
    """The E3 argv with the gate's 4-flag delta: ``--out`` / ``--steps`` replaced in place, ``--wsd-branch-from`` / ``--wsd-branch-decay`` appended, ``--save-at`` dropped (exempt; see the module doc)."""
    ps = pairs(e3_argv)
    e3_out = _run_dir(_one(ps, "--out")[0], "the E3 --out")
    if _one(ps, "--steps") != [str(E3_STEPS)]:
        _die("the E3 argv's --steps is %s, not %d" % (_one(ps, "--steps"), E3_STEPS))
    have = _one(ps, SAVE_AT)
    if len(have) != 1 or SRC_STEP not in LRS.parse_save_at(have[0], E3_STEPS):
        _die("the E3 argv's --save-at %s does not name %d: no stable-phase checkpoint to branch from" % (have, SRC_STEP))
    for f in BRANCH_FLAGS:
        if any(g == f for g, _ in ps):
            _die("the E3 argv already carries %s" % f)
    d1_out = _run_dir(d1_out if d1_out is not None else e3_out + "-d1", "--d1-out")
    if d1_out == e3_out:
        _die("--d1-out %s is the E3 run directory: the delta flag --out would be identical to the source's (not a new run)" % d1_out)
    sets = {"--out": d1_out, "--steps": str(D1_STEPS)}
    res = [(f, [sets[f]] if f in sets else v) for f, v in ps if f != SAVE_AT]
    res += [("--wsd-branch-from", [e3_out + "/" + LRS.FULL_CKPT_FMT % SRC_STEP]), ("--wsd-branch-decay", [str(D1_DECAY)])]
    return E2.flat(res)


def _carried(e2_doc: dict) -> dict:
    """The source's lever block + W-COST stamp, verbatim (a deep copy): only the keys the source has."""
    c = {}
    if "lever" in e2_doc:
        c["lever"] = copy.deepcopy(e2_doc["lever"])
    for k in W_COST_KEYS:
        if k in e2_doc:
            c[k] = copy.deepcopy(e2_doc[k])
    return c


def _set_lines(e2_argv: list, e3_argv: list) -> list:
    old, new = {f: v for f, v in pairs(e2_argv)}, {f: v for f, v in pairs(e3_argv)}
    why = {"--out": "the E3 run directory (operator-passed)", "--steps": "REFCV9_DESIGN 4.4 E3: 73,280 steps",
           "--warmup": "REFCV9_DESIGN 4.4 E3: warm-up 1,000", "--wsd-decay-steps": "REFCV9_DESIGN 4.4 E3: WSD decay 12,458",
           SAVE_AT: "full-state ckpt_full_<s>.pt at these steps; 30000 is the D1 branch's stable-phase source"}
    lines = []
    for f in ("--out", "--steps", "--warmup", "--wsd-decay-steps", SAVE_AT):
        was = old.get(f)
        lines.append("%s %s  <- %s%s" % (f, " ".join(new[f]), why[f], "" if was == new[f] else " (E2 source: %s)" % (" ".join(was) if was is not None else "absent")))
    return lines


def e3_doc(e2_doc: dict, e2_bytes: bytes, e2_path, out: str, save_at=str(SRC_STEP)) -> dict:
    argv = compose_e3(e2_doc, out, save_at)
    variant = _check_e2(e2_doc)[1]
    rec = {"schema": SCHEMA, "arm": "r9e3_%s" % variant, "seed": e2_doc.get("seed", 0), "steps": E3_STEPS, "argv": argv, "base_sha256": sha256_bytes(e2_bytes),
           "derived_from": e2_doc["arm"], "e2_source": {"path": _posix(e2_path), "sha256": sha256_bytes(e2_bytes)}}
    carried = _carried(e2_doc)
    if "lever" in carried:
        rec["lever"] = carried.pop("lever")
    rec["set_lines"] = _set_lines(e2_doc["argv"], argv)
    rec["what"] = ("E3 (the refcv9 full run, REFCV9_DESIGN 4.4) = %s with ONLY the full-run schedule rows (--steps %d, --warmup %d, --wsd-decay-steps %d), --grad-share-at (12 readings), --save-at %s and --out changed; every other "
                   "token is the E2 source's, in order. Composed by stack/scripts/r9_e3_make_argv.py from %s -- never hand-written; base_sha256 = sha256 of the E2 FILE bytes."
                   % (e2_doc["arm"], E3_STEPS, E3_WARMUP, E3_DECAY, save_at_text(save_at), _posix(e2_path)))
    rec["profile"] = E3_PROFILE
    rec.update(carried)
    return rec


def d1_doc(e3_rec: dict, d1_out: str | None = None) -> dict:
    argv = compose_d1(e3_rec["argv"], d1_out)
    e3_bytes = dump(e3_rec).encode("utf-8")
    rec = {"schema": SCHEMA, "arm": e3_rec["arm"] + "_d1", "seed": e3_rec.get("seed", 0), "steps": D1_STEPS, "argv": argv, "base_sha256": sha256_bytes(e3_bytes),
           "derived_from": e3_rec["arm"], "e2_source": copy.deepcopy(e3_rec["e2_source"])}
    if "lever" in e3_rec:
        rec["lever"] = copy.deepcopy(e3_rec["lever"])
    rec["set_lines"] = [
        "--out %s  <- the D1 run directory" % flag_value(argv, "--out"),
        "--steps %d  <- D1: the branch ends at source step %d + decay %d" % (D1_STEPS, SRC_STEP, D1_DECAY),
        "--wsd-branch-from %s  <- the E3 run's stable-phase full-state checkpoint (config.json beside it holds the source argv)" % flag_value(argv, "--wsd-branch-from"),
        "--wsd-branch-decay %d  <- D1: decay the %d checkpoint over %d" % (D1_DECAY, SRC_STEP, D1_DECAY),
        "--save-at dropped  <- the gate's EXEMPT cadence flag: the trainer refuses a step <= %d as never written; refcv9-full-d1 does not require it" % SRC_STEP]
    rec["what"] = ("D1 (the early decay branch of E3, REFCV9_DESIGN 4.4) = %s with EXACTLY the 4-flag delta (--out, --steps %d, --wsd-branch-from, --wsd-branch-decay %d); every other token is E3's, in order, "
                   "--save-at dropped. Composed by stack/scripts/r9_e3_make_argv.py; base_sha256 = sha256 of the E3 doc's bytes (r9_e3_make_argv.dump)." % (e3_rec["arm"], D1_STEPS, D1_DECAY))
    rec["profile"] = D1_PROFILE
    rec.update({k: copy.deepcopy(e3_rec[k]) for k in W_COST_KEYS if k in e3_rec})
    return rec


def build(e2_doc: dict, e2_bytes: bytes, e2_path, out: str, d1_out: str | None = None, save_at=str(SRC_STEP)) -> tuple:
    """(E3 doc, D1 doc). Pure: no file is read or written."""
    e3 = e3_doc(e2_doc, e2_bytes, e2_path, out, save_at)
    return e3, d1_doc(e3, d1_out)


def _same_file(a, b) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(a) == str(b)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compose the refcv9 E3 (profile refcv9-full) and D1 (profile refcv9-full-d1) argv docs from an E2 argv doc.")
    ap.add_argument("--e2", required=True, help="stack/ops/runs.d/r9e2_<V>.argv.json (the E2 GO arm)")
    ap.add_argument("--out", required=True, help="the E3 run directory on Thor (absolute POSIX), e.g. /home/nvidia/r8ns/r9e3/runs/r9e3-K2")
    ap.add_argument("--d1-out", default=None, help="the D1 run directory (default: <--out>-d1)")
    ap.add_argument("--save-at", default=str(SRC_STEP), help="E3's --save-at list; must include %d (default %d)" % (SRC_STEP, SRC_STEP))
    ap.add_argument("--write-e3", help="write the E3 doc here")
    ap.add_argument("--write-d1", help="write the D1 doc here")
    ap.add_argument("--check", action="store_true", help="regenerate and require the --write-e3 / --write-d1 files BYTE-EQUAL; write nothing")
    a = ap.parse_args(argv)
    if not (a.write_e3 or a.write_d1):
        ap.error("--write-e3 and/or --write-d1 is required")
    targets = [p for p in (a.write_e3, a.write_d1) if p]
    for p in targets:
        if _same_file(p, a.e2):
            _die("%s is the E2 source: refusing to write over it" % p)
    if a.write_e3 and a.write_d1 and _same_file(a.write_e3, a.write_d1):
        _die("--write-e3 and --write-d1 name the same file")
    e2_bytes = Path(a.e2).read_bytes()
    e2_doc = json.loads(e2_bytes.decode("utf-8"))
    e3, d1 = build(e2_doc, e2_bytes, a.e2, a.out, a.d1_out, a.save_at)
    if CRITIC in e3["argv"] and "w_cost_ratio" not in e3:
        print("NOTE: %s carries the drivable critic with %s %s and %s -- NOT a measured W-COST weight: the gate's G-LIVE holds it to 0.1 / R of its own smoke (a token cannot bind a provisional weight)"
              % (e2_doc["arm"], W_CRITIC, flag_value(e3["argv"], W_CRITIC), "w_cost: %s" % e3["w_cost"] if "w_cost" in e3 else "no W-COST stamp"))
    for tag, p, rec in (("E3", a.write_e3, e3), ("D1", a.write_d1, d1)):
        if not p:
            continue
        got = dump(rec).encode("utf-8")
        if a.check:
            have = Path(p).read_bytes() if Path(p).is_file() else None
            if have != got:
                _die("%s doc %s does NOT regenerate byte-equal (%s vs %d bytes)" % (tag, p, "absent" if have is None else "%d" % len(have), len(got)))
            print("%s regenerates BYTE-EQUAL (%d bytes, sha256 %s)" % (tag, len(got), sha256_bytes(got)))
        else:
            write(p, rec)
            print("wrote %s (%s, %d bytes, sha256 %s)" % (p, tag, len(got), sha256_bytes(got)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
