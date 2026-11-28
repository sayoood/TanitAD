#!/usr/bin/env python3
"""Two-sided mutation proof for `tools/backlog_drift.py`.

WHY. A test that passes on BOTH the fixed and the broken code is worthless, and
this programme has shipped four of those in one night (a label builder verified
against its own rounded ladder; a census whose test pinned a set against the
status computed from that set; a channel guard asserting "expect whatever the
code does"; a conditioning test whose stub could not express the nested fields).
The only proof that a guard guards is: reintroduce the defect, watch the suite go
RED, restore, watch it go GREEN.

⛔ EVERY MUTATION BELOW IS A DEFECT THIS TOOL ACTUALLY HAD OR WAS ONE EDIT AWAY
FROM. None is a synthetic "change a constant" mutation, because the question is
not "do the tests touch this line" but "would the tests have caught the mistake".

⚠️ EXIT STATUS IS READ FROM `returncode`, NEVER THROUGH A PIPE. `cmd | tail`
reports tail's status (MEASURED twice on 2026-09-07, on two different tools, one
of them a pre-launch gate that printed GATE_EXIT=0 after being killed with no
output at all). And the restore is asserted on the FILE's md5, not on the exit
code of the write.

Usage:  python run_mutation_proof.py            # writes MUTATION_PROOF.json beside itself
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _find_repo(start: Path) -> Path:
    """Walk up to the root that actually HOLDS the tool.

    ⛔ A hardcoded `parents[N]` was wrong on the first run (`parents[5]` resolved
    to `D:\\Projects`, one level too high) and the failure was a FileNotFoundError
    -- loud, and therefore lucky. Anchoring on a file that must exist means the
    wrong answer cannot be silent.
    """
    for p in [start, *start.parents]:
        if (p / "tools" / "backlog_drift.py").is_file():
            return p
    raise SystemExit("REFUSING: no repo root above %s holds tools/backlog_drift.py"
                     % start)


REPO = _find_repo(HERE)
TOOL = REPO / "tools" / "backlog_drift.py"
TESTS = REPO / "tools" / "tests" / "test_backlog_drift.py"
PY = sys.executable

#: (name, what the defect IS, old, new). Each `old` must occur EXACTLY ONCE, or
#: the mutation is ambiguous and the proof is refused -- a mutation that lands in
#: the wrong place proves nothing about the right one.
MUTATIONS = [
    ("M1_no_zero_row_guard",
     "the central anti-silent-pass guard: an unparseable file reports CLEAN",
     '    if not rows:\n        print("REFUSING: parsed 0 rows',
     '    if False:\n        print("REFUSING: parsed 0 rows'),

    ("M2_substring_done_words",
     "the ORIGINAL defect: done-words matched as substrings, so `closed-loop`, "
     "`done-marker`, `roll_closed` and `-RESOLVED` all read as closures",
     '    head = re.split(r"[^0-9a-z]+", t, maxsplit=1)[0] if t else ""\n'
     '    return head in DONE_WORDS',
     '    return any(w in t for w in DONE_WORDS)'),

    # ⭐ THIS SLOT PREVIOUSLY HELD `M3_ignore_negations`, AND IT SURVIVED GREEN.
    # That is the proof that the tool's `NEGATIONS` list was dead code -- first
    # word anchoring already rejected `PARTLY STRUCK` and `REFRAMED (not
    # struck)`, and removing the list moved 0 of 117 live verdicts. The list was
    # DELETED rather than kept as "defence in depth". A surviving mutation is a
    # finding about the CODE, not a reason to weaken the proof.
    ("M3_bullet_bold_prefix_dropped",
     "a bullet's consumed opening `**` is not restored, so spans mis-pair and a "
     "phantom span beginning with a closure verb reads as DONE -- a FALSE "
     "closure, which silently removes work from the list",
     '            done = declares_done(body, bold_src="**" + body)',
     '            done = declares_done(body)'),

    ("M9_no_min_bytes_guard",
     "a truncated read on a flaky mount is accepted as a short file",
     '    if len(text.encode("utf-8", "replace")) < MIN_BYTES:',
     '    if False:'),

    ("M10_no_min_rows_floor",
     "a parser regression that drops a whole row family reads as a tidy backlog",
     "    if n < a.min_rows:",
     "    if False:"),

    ("M11_no_zero_closed_guard",
     "a close-detector finding nothing reports no problems instead of refusing",
     "    if closed == 0:",
     "    if False:"),

    # ⚠️ MUTATE THE PATTERN, NOT `.match` -> `.search`. The first attempt swapped
    # the method and survived GREEN, because the pattern still carries `^` and
    # `search` on an anchored pattern IS `match`. An inert mutation reports the
    # suite as weak when the suite is fine -- the mirror of a test that passes on
    # broken code, and just as misleading.
    ("M12_heading_items_unanchored",
     "an unanchored heading match also catches the SECTION CONTAINER whose "
     "blocker glyph is a note inside the heading, not a status on it",
     # ⚠️ Anchored on the CALL SITE, not on the pattern literal: the pattern is
     # written with real glyph characters inside a raw string, so a `\\uXXXX`
     # spelling of it matches 0 times and the proof refuses (it did, once).
     "            if level == 2 and HEADING_ITEM.match(raw):",
     '            if re.search("BLOCKING|\\u26d4", raw):'),

    ("M4_gate_never_fails",
     "the gate prints its findings and exits 0 -- a check nobody has to read",
     "    print(\"=> DONE-BUT-OPEN : strike the id (~~A11~~) and keep the evidence.\")",
     "    return 0\n    print(\"=> DONE-BUT-OPEN : strike the id (~~A11~~) and keep the evidence.\")"),

    ("M5_no_id_aggregation",
     "per-OCCURRENCE verdicts instead of per-ID: every bullet later closed by a "
     "separate `-- STRUCK` line is reported stale, and the gate cries wolf",
     "        closed_elsewhere = any(o.body_done for o in occ[1:])",
     "        closed_elsewhere = False"),

    ("M6_blocker_includes_item_cell",
     "`cells[2:]` instead of `cells[3:]`: a struck ITEM is reported as a cleared "
     "BLOCKER -- a wrong reason attached to a right row",
     '    return any("~~" in c for c in cells[3:])',
     '    return any("~~" in c for c in cells[2:])'),

    ("M7_no_all_closed_guard",
     "a strike-detector broken the SILENT way (everything reads closed) reports "
     "no problems instead of refusing",
     "    if closed == n:\n        print(\"REFUSING: all %d rows read as closed",
     "    if False:\n        print(\"REFUSING: all %d rows read as closed"),

    ("M8_non_ascii_to_stdout",
     "row excerpts reach stdout un-transliterated -- on the cp1252 console this "
     "is a crash in a guard, i.e. it fails closed for the wrong reason",
     '            "excerpt": ascii_only(first.text)[:200],',
     '            "excerpt": first.text[:200],'),
]


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def run_suite() -> tuple[int, str]:
    r = subprocess.run([PY, "-m", "pytest", str(TESTS), "-q", "--no-header",
                        "-p", "no:cacheprovider"],
                       cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tail = "\n".join((r.stdout or "").strip().splitlines()[-3:])
    return r.returncode, tail


def main() -> int:
    src = TOOL.read_text(encoding="utf-8")
    baseline_md5 = md5(TOOL)

    rc, tail = run_suite()
    results = {"tool": str(TOOL.relative_to(REPO)).replace("\\", "/"),
               "baseline_md5": baseline_md5,
               "baseline": {"returncode": rc, "tail": tail},
               "mutations": []}
    if rc != 0:
        print("REFUSING: the baseline suite is not green (rc=%d). A mutation "
              "proof against a red baseline proves nothing.\n%s" % (rc, tail))
        return 2
    print("baseline: GREEN (rc=0)  %s" % tail)

    ok = True
    for name, defect, old, new in MUTATIONS:
        if src.count(old) != 1:
            print("REFUSING: %s anchor occurs %d times, not once" % (name, src.count(old)))
            return 2
        TOOL.write_text(src.replace(old, new), encoding="utf-8")
        try:
            rc, tail = run_suite()
        finally:
            TOOL.write_text(src, encoding="utf-8")
            # ⛔ Assert the RESTORE on the file's md5, not on the write's status.
            assert md5(TOOL) == baseline_md5, "restore failed for " + name
        red = rc != 0
        ok &= red
        print("%-32s %s  (rc=%d)  %s" % (name, "RED  " if red else "GREEN",
                                         rc, tail.splitlines()[-1] if tail else ""))
        results["mutations"].append({"name": name, "defect": defect,
                                     "returncode": rc, "went_red": red,
                                     "tail": tail})

    rc2, tail2 = run_suite()
    results["after_restore"] = {"returncode": rc2, "tail": tail2,
                                "md5_matches_baseline": md5(TOOL) == baseline_md5}
    print("after restore: %s (rc=%d)  md5 match=%s"
          % ("GREEN" if rc2 == 0 else "RED", rc2, md5(TOOL) == baseline_md5))

    results["verdict"] = ("ALL MUTATIONS CAUGHT" if ok and rc2 == 0
                          else "A MUTATION SURVIVED - the suite is not a guard")
    (HERE / "MUTATION_PROOF.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(results["verdict"])
    return 0 if (ok and rc2 == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
