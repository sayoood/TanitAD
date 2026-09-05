#!/usr/bin/env python
"""Reintroduce each M18-class defect and MEASURE whether the suite catches it.

⛔⛔ WHY THIS EXISTS — A GUARD THAT CANNOT FAIL IS NOT EVIDENCE.
--------------------------------------------------------------
`test_refc_v3_agent_provenance.py` opens with the right doctrine — *"every gate
here is shown to FAIL the defect: a control that has never read the wrong value
certifies nothing"* — but a static assertion cannot show that. The showing was
done ONCE, by hand, on 2026-09-05, and lived only in a transcript. The next
refactor of `_build_rig_camera` can turn every one of those tests into a
tautology and the suite will still read 67 passed.

This script is that showing, made repeatable. Each entry in :data:`MUTATIONS`
reintroduces a defect that was ACTUALLY MEASURED in this repo, names the test
that must catch it, and the audit FAILS if the suite stays green.

⭐ THE INSTRUMENT THAT LOOKS RIGHT AND IS BLIND. Before writing this, the
obvious audit was tried and REFUTED: an AST census of `build_parser` asking
*"is every CLI dest read outside the parser and the stamps?"*. It reads
**0 suspects** — on both the fixed trainer AND on the trainer carrying the
original M18 defect (MEASURED here 2026-09-05, mutation `rig_camera_none`).
It cannot see the defect because `--agent-w-project` WAS read: it reached
`cfg.core.agents.w_project` and then died against a `cam is not None` guard
that was always False. Reachability of the FLAG is not reachability of the
TERM, and only running the thing separates them. Same family as the `df` /
Thor-`free` traps: a probe answering the adjacent question.

WHAT A RESULT MEANS
-------------------
* ``CAUGHT``   — the named test(s) failed. The guard is load-bearing.
* ``ESCAPED``  — the suite stayed green with the defect installed. **The guard
  is decorative.** Non-zero exit.
* ``MISCREDITED`` — the suite went red, but not via the named test. Something
  else is holding the line; the guard credited in the test name is not the one
  doing the work, and deleting it would be invisible. Non-zero exit.

SAFETY — THE FILES ARE MUTATED IN PLACE, AND THAT IS DELIBERATE
---------------------------------------------------------------
The two files are `stack/scripts/refc_v3_train.py` (imported by path, not as a
package) and `stack/tanitad/refs/refc_agents.py` (inside the `tanitad`
package). Shadowing either from a partial copy on `sys.path` cannot work, and
copying `stack/` is 2.1 GB. So the audit edits in place and restores, with:

1. **A byte snapshot before anything is touched**, written to
   ``.guard_mutation_backup/`` beside the stack root, plus an ``ACTIVE``
   marker naming every path and its sha256.
2. **Crash recovery on startup** — an ``ACTIVE`` marker found at launch means
   a previous run died mid-mutation: the audit restores from the snapshot,
   says so loudly, and REFUSES to continue.
3. **Restore verified by sha256 after every single mutation**, never assumed.
   A failed restore aborts immediately and prints the recovery command.
4. **Bytes, not text.** These files are CRLF in the working tree; a
   `read_text`/`write_text` round trip would rewrite every line ending in the
   file and make the diff unreadable. Anchors are converted to the file's own
   ending instead.
5. **Every anchor must occur EXACTLY ONCE.** A `replace(..., 1)` against an
   anchor that appears twice mutates whichever came first — a silently
   different experiment. A non-unique or absent anchor is a hard failure of
   the audit, not a skipped case: it means the registry has rotted against a
   refactor, which is precisely when this instrument is needed.

USAGE
-----
    python stack/scripts/guard_mutation_audit.py               # the full audit
    python stack/scripts/guard_mutation_audit.py --list        # registry only
    python stack/scripts/guard_mutation_audit.py --check-anchors  # no pytest
    python stack/scripts/guard_mutation_audit.py -k rig_camera # subset

⚠️ Do NOT run this against a tree with uncommitted edits to the mutated files:
the audit refuses, because "restore" would then mean "restore to something I
cannot name". Commit or stash first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

STACK = Path(__file__).resolve().parents[1]
BACKUP = STACK / ".guard_mutation_backup"
MARKER = BACKUP / "ACTIVE"

#: Every test file a mutation can be caught by. Run as a set so a mutation that
#: breaks a NEIGHBOURING guard is visible as MISCREDITED rather than silently
#: passing because we only ran the file we expected to fail.
TEST_FILES = ("tests/test_refc_v3_agent_provenance.py",
              "tests/test_refc_v3_refcv5_wiring.py",
              "tests/test_refc_v3_agent_join.py",
              "tests/test_guard_mutation_audit.py")

#: Set in the pytest subprocess to the KEY of the mutation currently applied
#: (``"-"`` for the baseline), so `test_guard_mutation_audit.py` can tell
#: "the tree is mutated because a run DIED" from "…because I am that run".
#:
#: ⛔ THE KEY, NOT A BOOLEAN, AND THAT DISTINCTION IS THE INSTRUMENT. The
#: applied mutation's own anchor is — correctly — absent from the source while
#: it is applied, so a blanket rot check goes red on EVERY mutation. MEASURED
#: here 2026-09-05: with it red every time, `named` was never the only failure
#: and the ESCAPED verdict became UNREACHABLE — a decorative guard would have
#: read MISCREDITED. An instrument that can only return one answer is the very
#: defect this file audits, so the check excludes exactly one key and still
#: covers every other anchor (which is how `registry_anchor_rotted` is caught).
RUNNING_ENV = "GUARD_MUTATION_AUDIT_RUNNING"


@dataclass(frozen=True)
class Mutation:
    """One reintroduced defect, and the guard that must catch it.

    ``caught_by`` is the MINIMUM set of node ids that must fail. Parametrised
    tests are matched by prefix, so ``…::test_x`` matches ``…::test_x[case]``.
    Extra failures are reported but are not themselves a verdict: a defect
    that trips two guards is fine, a defect that trips none of the NAMED ones
    is not.
    """

    key: str
    defect: str
    path: str
    old: str
    new: str
    caught_by: tuple[str, ...]
    source: str = ""
    notes: str = field(default="")


_TRAIN = "scripts/refc_v3_train.py"
_AGENTS = "tanitad/refs/refc_agents.py"
_PROV = "tests/test_refc_v3_agent_provenance.py"
_WIRE = "tests/test_refc_v3_refcv5_wiring.py"
_SELF = "tests/test_guard_mutation_audit.py"

MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        key="rig_camera_none",
        defect=("`model._rig_camera = None`, never assigned — the LITERAL "
                "2026-09-05 defect. `--agent-w-project 0.2` parses, is "
                "serialised into config.json['seams']['agents'], and computes "
                "nothing, so the arm reads as 'the image-plane term does not "
                "help' — a refutation manufactured by a missing term."),
        path=_TRAIN,
        old="    model._rig_camera, _cam_stamp = _build_rig_camera(cfg, args)",
        new=("    model._rig_camera = None\n"
             "    _cam_stamp = _build_rig_camera(cfg, args)[1]"),
        caught_by=(f"{_PROV}::test_P1_the_trainer_never_assigns_a_CONSTANT_"
                   "None_rig_camera",),
        source="mm-decisions M18; DataFlyWheel escalation #2",
    ),
    Mutation(
        key="off_camera_refusal_removed",
        defect=("`--agent-rig-camera off` with a non-zero monocular weight no "
                "longer refuses. This is the defect's REACHABILITY: with the "
                "refusal gone, an argv exists that stamps a weight it never "
                "trains, whatever the model-setup line says."),
        path=_TRAIN,
        old=('    if src == "off":\n'
             "        if w_pj > 0.0 or w_gr > 0.0:\n"
             "            raise SystemExit("),
        new=('    if src == "off":\n'
             "        if False:\n"
             "            raise SystemExit("),
        caught_by=(f"{_PROV}::test_P1_weight_without_a_camera_REFUSES",),
        source="mm-decisions M18",
    ),
    Mutation(
        key="monocular_terms_skipped",
        defect=("`agent_losses` skips BOTH monocular terms unconditionally — "
                "the downstream half of the same defect. The trainer can hold "
                "a perfectly good RigCamera and the loss still not contain "
                "the term, which is what made the original bug survivable: "
                "nothing downstream noticed a missing addend."),
        path=_AGENTS,
        old="    if cam is not None and cfg.w_project > 0.0:",
        new="    if False and cam is not None and cfg.w_project > 0.0:",
        caught_by=(f"{_PROV}::test_P1_the_camera_the_TRAINER_builds_makes_the_"
                   "terms_COMPUTE",),
        source="mm-decisions M18",
    ),
    Mutation(
        key="ground_term_skipped",
        defect=("as `monocular_terms_skipped`, but the ground-range prior "
                "alone. Registered SEPARATELY because a guard that only ever "
                "exercises `w_project` would certify `w_ground` for free — "
                "the two are independent addends behind independent guards."),
        path=_AGENTS,
        old="    if cam is not None and cfg.w_ground > 0.0:",
        new="    if False and cam is not None and cfg.w_ground > 0.0:",
        caught_by=(f"{_PROV}::test_P1_the_camera_the_TRAINER_builds_makes_the_"
                   "terms_COMPUTE",),
        source="mm-decisions M18",
    ),
    Mutation(
        key="unsupervised_detector_allowed",
        defect=("`--agents head --w-agent 0` no longer refuses: a detector "
                "that is built, wired into cross-attention, and never "
                "supervised. Its tokens are noise and the arm reads as 'agent "
                "tokens do not help'. Same family, one level up — this is the "
                "one M18's own audit missed."),
        path=_TRAIN,
        old=('        if args.agents == "head" and float(getattr(args, '
             '"w_agent", 0.0)) <= 0.0:'),
        new=('        if False and args.agents == "head" and float(getattr('
             'args, "w_agent", 0.0)) <= 0.0:'),
        caught_by=(f"{_WIRE}::test_agent_head_without_its_LOSS_REFUSES",),
        source="commit 606d938",
    ),
    Mutation(
        key="weights_absent_from_config",
        defect=("`w_agent` / `w_u0` dropped from the top level of "
                "`_seam_stamp`: a finished run's own record cannot say what "
                "weight its detector was trained at. The SEAM_STATE.md "
                "failure — a run record that cannot rebuild its own model "
                "config is not a run record."),
        path=_TRAIN,
        old=('        "w_agent": float(getattr(args, "w_agent", '
             "AGENT_WEIGHT_DEFAULT)),\n"
             '        "w_u0": float(getattr(args, "w_u0", '
             "U0_WEIGHT_DEFAULT)),\n"),
        new="",
        caught_by=(f"{_WIRE}::test_seam_stamp_carries_every_refcv5_lever",),
        source="mm-decisions M18 escalation #3; SEAM_STATE.md",
    ),
    # --- the recursive pair: the audit's own evidence, audited ----------
    # ⛔ Without these two, `test_guard_mutation_audit.py` is exactly the thing
    # it exists to prevent: a guard nobody has ever seen fail. They mutate THIS
    # FILE, which the pytest subprocess re-imports from disk while the parent
    # keeps its own in-memory registry — so the audit can rot itself, watch the
    # rot check catch it, and put itself back.
    Mutation(
        key="registry_anchor_rotted",
        defect=("an anchor in this registry no longer matches the shipped "
                "source — the silent-vacuity failure. A rotted audit applies "
                "nothing, catches nothing, and reports a clean run: a GREEN "
                "audit certifying an absent guard, which is worse than no "
                "audit because it converts an unknown into a reassurance."),
        path="scripts/guard_mutation_audit.py",
        old=('        old="    model._rig_camera, _cam_stamp = '
             '_build_rig_camera(cfg, args)",'),
        new='        old="    model._rig_camera = GONE_IN_A_REFACTOR",',
        caught_by=(f"{_SELF}::test_every_anchor_is_present_in_the_shipped_"
                   "source_exactly_once",),
        source="this file, 2026-09-05",
    ),
    Mutation(
        key="registry_names_a_dead_guard",
        defect=("a `caught_by` names a test that no longer exists — a "
                "renamed or deleted guard. The audit would then report "
                "MISCREDITED forever, burying the real message (*this defect "
                "is unguarded again*) inside a verdict about crediting."),
        path="scripts/guard_mutation_audit.py",
        old=(f'        caught_by=(f"{{_WIRE}}::test_agent_head_without_its_'
             'LOSS_REFUSES",),'),
        new=('        caught_by=(f"{_WIRE}::test_a_guard_that_was_renamed_'
             'away",),'),
        caught_by=(f"{_SELF}::test_every_named_guard_actually_exists",),
        source="this file, 2026-09-05",
    ),
    Mutation(
        key="knob_stamp_emptied",
        defect=("`agent_knobs` stamped as `{}` — the DERIVED provenance "
                "closure defeated. Every `--agent-*` / `--w-*` knob "
                "disappears from config.json at once, which is the failure "
                "the derived-from-argparse design exists to make impossible."),
        path=_TRAIN,
        old='        "agent_knobs": agent_knob_stamp(args),',
        new='        "agent_knobs": {},',
        caught_by=(f"{_PROV}::test_P2_every_knob_is_recoverable_from_the_"
                   "stamp_BY_VALUE",),
        source="mm-decisions M18 escalation #3",
    ),
)


# --------------------------------------------------------------------------
# byte-level edit + snapshot machinery
# --------------------------------------------------------------------------

def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _anchor_bytes(text: str, data: bytes) -> bytes:
    """Encode an anchor into the file's OWN line ending.

    The working tree is CRLF; the anchors are written LF because that is what
    a diff and a `git show` display. Converting here keeps both readable and
    means a future LF checkout needs no edit to this file.
    """
    nl = b"\r\n" if b"\r\n" in data else b"\n"
    return text.replace("\n", nl.decode()).encode("utf-8")


def check_anchors(muts=MUTATIONS) -> list[str]:
    """Every anchor present EXACTLY ONCE. Returns a list of problems.

    ⛔ This is the anti-vacuity check, and it is the reason the audit is worth
    committing. A registry whose anchors have rotted against a refactor would
    otherwise apply nothing, catch nothing, and report a clean run — a green
    audit certifying an absent guard, which is strictly worse than no audit.
    """
    bad = []
    for m in muts:
        f = STACK / m.path
        if not f.is_file():
            bad.append(f"{m.key}: {m.path} does not exist")
            continue
        data = f.read_bytes()
        n = data.count(_anchor_bytes(m.old, data))
        if n != 1:
            bad.append(
                f"{m.key}: anchor occurs {n}x in {m.path} (need exactly 1). "
                "The registry has rotted against a refactor — re-point it at "
                "the guard's current text, do not delete the entry.")
    return bad


def _snapshot(paths) -> dict:
    BACKUP.mkdir(exist_ok=True)
    rec = {}
    for rel in paths:
        src = STACK / rel
        dst = BACKUP / rel.replace("/", "__")
        shutil.copyfile(src, dst)
        rec[rel] = {"sha256": _sha(src), "backup": dst.name}
    MARKER.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def _restore(rec: dict) -> None:
    for rel, meta in rec.items():
        dst = STACK / rel
        shutil.copyfile(BACKUP / meta["backup"], dst)
        got = _sha(dst)
        if got != meta["sha256"]:
            raise SystemExit(
                f"!! RESTORE FAILED for {rel}: sha256 {got} != "
                f"{meta['sha256']}. The working tree is LEFT MUTATED. "
                f"Recover with:\n    cp '{BACKUP / meta['backup']}' "
                f"'{dst}'\nor `git checkout -- {rel}` if the file was clean.")


def _apply(m: Mutation) -> None:
    f = STACK / m.path
    data = f.read_bytes()
    old = _anchor_bytes(m.old, data)
    new = _anchor_bytes(m.new, data) if m.new else b""
    if data.count(old) != 1:
        raise SystemExit(f"!! {m.key}: anchor is not unique in {m.path}.")
    f.write_bytes(data.replace(old, new, 1))


def _pytest(extra=(), applied: str = "-") -> tuple[int, list[str], str]:
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "6")
    env[RUNNING_ENV] = applied
    env["PYTHONPATH"] = os.pathsep.join(
        [str(STACK), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])])
    # ⛔ `encoding=` IS NOT OPTIONAL, and this call is the reason why. What is
    # being decoded is pytest's failure summary, and the assertion messages of
    # the very guards under audit contain U+26D4. `text=True` alone decodes
    # with the LOCALE codec — cp1252 here — so the first genuinely-caught
    # defect would surface as `UnicodeDecodeError` inside the audit rather
    # than as a verdict. Repo-wide rule, pinned by
    # `tests/test_text_encoding_is_explicit.py`.
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p",
         "no:cacheprovider", "-rf", *TEST_FILES, *extra],
        cwd=STACK, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    failed = sorted({ln.split(" ", 1)[1].split(" ")[0].strip()
                     for ln in p.stdout.splitlines()
                     if ln.startswith("FAILED ")})
    tail = ([ln for ln in p.stdout.splitlines() if ln.strip()] or ["<none>"])[-1]
    return p.returncode, failed, tail


def _matches(expect: str, failed: list[str]) -> bool:
    return any(f == expect or f.startswith(expect + "[") for f in failed)


# --------------------------------------------------------------------------

def _harden_streams() -> None:
    """Never let an ENCODING error swallow a verdict.

    MEASURED 2026-09-05: this console is cp1252 and the ESCAPED branch printed
    a U+26D4, so the one verdict that says *your guard is decorative* died in
    `UnicodeEncodeError` instead of being read. The printed strings are ASCII
    now; this is the backstop for text that is not ours (a mutation's `defect`
    prose, a pytest tail), where a crash would still lose the result.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass


def main(argv=None) -> int:
    _harden_streams()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true",
                    help="print the registry and exit")
    ap.add_argument("--check-anchors", action="store_true",
                    help="verify every anchor is present exactly once; no "
                         "pytest, no mutation")
    ap.add_argument("-k", default=None,
                    help="substring filter on the mutation key")
    args = ap.parse_args(argv)

    muts = [m for m in MUTATIONS if not args.k or args.k in m.key]

    if args.list:
        for m in muts:
            print(f"{m.key}\n    defect : {m.defect}")
            print(f"    file   : {m.path}")
            print(f"    caught : {', '.join(m.caught_by)}")
            print(f"    source : {m.source}\n")
        return 0

    bad = check_anchors(muts)
    if bad:
        print("!! REGISTRY ROT -- the audit cannot certify anything:")
        for b in bad:
            print(f"   {b}")
        return 2
    print(f"anchors: {len(muts)}/{len(muts)} present exactly once")
    if args.check_anchors:
        return 0

    if MARKER.exists():
        rec = json.loads(MARKER.read_text(encoding="utf-8"))
        print("!! a previous run died mid-mutation. Restoring from the "
              "snapshot and refusing to continue:")
        _restore(rec)
        MARKER.unlink()
        for rel in rec:
            print(f"   restored {rel}")
        print("Re-run the audit now that the tree is clean.")
        return 2

    rec = _snapshot(sorted({m.path for m in muts}))
    try:
        rc, failed, tail = _pytest()
        print(f"baseline: rc={rc}  {tail}")
        if rc != 0:
            print("!! the suite is ALREADY red. Mutation results would be "
                  "unreadable -- fix the baseline first.")
            for f in failed:
                print(f"   {f}")
            return 2

        rows = []
        for m in muts:
            _apply(m)
            rc, failed, tail = _pytest(applied=m.key)
            _restore(rec)
            named = [e for e in m.caught_by if _matches(e, failed)]
            if named:
                verdict = "CAUGHT"
            elif rc != 0:
                verdict = "MISCREDITED"
            else:
                verdict = "ESCAPED"
            rows.append((m, verdict, failed, tail))
            print(f"\n{verdict:12s} {m.key}")
            print(f"             {tail}")
            for f in failed[:4]:
                mark = "<-- named" if any(
                    _matches(e, [f]) for e in m.caught_by) else ""
                print(f"             {f} {mark}")
            if verdict == "ESCAPED":
                print("             !! the suite stayed GREEN with this "
                      "defect installed. The guard is decorative.")
            if verdict == "MISCREDITED":
                print("             !! red, but not via "
                      f"{', '.join(m.caught_by)} -- the guard credited is not "
                      "the guard holding.")
    finally:
        _restore(rec)
        if MARKER.exists():
            MARKER.unlink()
        shutil.rmtree(BACKUP, ignore_errors=True)

    ok = sum(1 for _, v, _, _ in rows if v == "CAUGHT")
    print(f"\n=== {ok}/{len(rows)} defects CAUGHT by the named guard ===")
    for m, v, _, _ in rows:
        if v != "CAUGHT":
            print(f"  {v}: {m.key} - {m.defect.split(chr(46))[0]}.")
    print("tree restored and verified by sha256.")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
