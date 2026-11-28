"""Mutation audit for the named prefix floor in ``tools/clipid_scan.py``.

Each mutant removes or weakens ONE guard in a COPY of the tool, then runs the suite. A
guard is proven only when the test that pins it goes red, so every mutant names the
test it expects to fail, and the audit reports any difference in either direction.
The repo is never touched.

    python mutation_audit.py <repo root> <out.json>
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: (id, what the mutant does, anchor, replacement, tests expected to go red)
MUTANTS = [
    ("a", "digest check reduced to a count check",
     'if (rec.get("sha256_sorted"), rec.get("n")) != (ident["sha256_sorted"], ident["n"]):',
     'if rec.get("n") != ident["n"]:',
     ["test_MUT_a_DIFFERENT_list_of_the_SAME_size_is_REFUSED_not_compared"]),
    ("b", "a baseline with no named list is silently accepted",
     "        if rec is None:\n            raise ClipListMismatch(",
     "        if rec is None:\n            rec = dict(ident)\n"
     "        if rec is None:\n            raise ClipListMismatch(",
     ["test_MUT_a_baseline_WITHOUT_a_named_list_refuses_the_prefix_half"]),
    ("c", "the prefix half is compared although it was not counted",
     "prefixes_counted=bool(ident))", "prefixes_counted=True)",
     ["test_the_default_run_does_not_report_the_prefix_floor_as_a_SHRINK"]),
    ("d", "a baseline is written over unreadable files",
     '        if cur.get("_unreadable"):\n            raise LeakGrew(f"refusing to write',
     '        if False:\n            raise LeakGrew(f"refusing to write',
     ["test_MUT_a_baseline_is_NOT_written_over_UNREADABLE_files"]),
    ("e", "the list identity depends on order and duplicates",
     "s = sorted(set(ids))", "s = list(ids)",
     ["test_the_SAME_list_written_differently_is_the_SAME_list"]),
    ("f", "a BOM is not stripped from the list file",
     'read_text(encoding="utf-8-sig")', 'read_text(encoding="utf-8")',
     ["test_the_SAME_list_written_differently_is_the_SAME_list"]),
    # EXPECTED TO SURVIVE: `compare_to_baseline` walks the SCAN's files and looks each
    # one up, so a `_clips_list` key in the baseline is never visited. The filter is
    # defence for a future compare that walks the baseline; it is not load-bearing today.
    ("g", "the '_'-key filter is removed before comparing",
     '{k: v for k, v in base.items() if not k.startswith("_")}', "base",
     []),
]

_SUMMARY = re.compile(r"(\d+ (?:failed|passed)[^\n]*?) in [0-9.]+s")
_FAILED = re.compile(r"FAILED \S+::(\w+)")


def _suite(tmp: Path) -> dict:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp / "tools" / "tests" / "test_clipid_scan.py"),
         "-q", "-rf", "-p", "no:cacheprovider"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(tmp))
    out = r.stdout + r.stderr
    s = _SUMMARY.findall(out)
    return {"exit": r.returncode, "summary": s[-1] if s else out[-400:],
            "red": sorted(set(_FAILED.findall(out)))}


def main(repo: str, out_path: str) -> int:
    repo = Path(repo)
    src = (repo / "tools" / "clipid_scan.py").read_text(encoding="utf-8")   # universal newlines
    raw = (repo / "tools" / "clipid_scan.py").read_bytes()
    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "tools" / "tests").mkdir(parents=True)
        shutil.copy2(repo / "tools" / "tests" / "test_clipid_scan.py",
                     tmp / "tools" / "tests" / "test_clipid_scan.py")
        tool = tmp / "tools" / "clipid_scan.py"
        tool.write_text(src, encoding="utf-8", newline="\n")
        unmutated = _suite(tmp)
        for mid, what, anchor, repl, expect in MUTANTS:
            n = src.count(anchor)
            # ⛔ the duplicate-anchor trap: a mutant applied to the wrong copy of a line
            # proves nothing, so an anchor must occur exactly once.
            if n != 1:
                rows.append({"id": mid, "mutant": what, "error": f"anchor occurs {n}x"})
                continue
            tool.write_text(src.replace(anchor, repl), encoding="utf-8", newline="\n")
            got = _suite(tmp)
            if expect:
                verdict = ("KILLED by exactly the expected test" if got["red"] == sorted(expect)
                           else f"UNEXPECTED: expected {expect}")
            else:
                verdict = ("SURVIVES, as expected (equivalent mutant)" if not got["red"]
                           else "UNEXPECTED: an equivalent mutant was killed")
            rows.append({"id": mid, "mutant": what, "expected_red": expect, **got,
                         "verdict": verdict})
        tool.write_text(src, encoding="utf-8", newline="\n")
        restored = _suite(tmp)
    rep = {"tool_line_endings": "CRLF" if b"\r\n" in raw else "LF",
           "unmutated": unmutated, "mutants": rows, "restored": restored,
           "all_as_expected": all(r.get("verdict", "").startswith(("KILLED", "SURVIVES"))
                                  for r in rows) and unmutated["exit"] == 0
                              and restored["exit"] == 0}
    Path(out_path).write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n",
                              encoding="utf-8", newline="\n")
    print(f"unmutated: {unmutated['summary']}")
    for r in rows:
        print(f"  {r['id']} {r['mutant']:<55} {r.get('summary', r.get('error'))!s:<22} "
              f"{r.get('verdict', '')}")
    print(f"restored: {restored['summary']} | all as expected: {rep['all_as_expected']}")
    return 0 if rep["all_as_expected"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
