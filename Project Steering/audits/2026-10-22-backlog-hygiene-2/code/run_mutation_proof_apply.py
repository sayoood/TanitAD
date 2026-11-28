#!/usr/bin/env python3
"""Mutation proof for `apply_hygiene.py` (same discipline as run_mutation_proof.py).

    python run_mutation_proof_apply.py > ../raw/mutation_check_apply.out

Runs `test_apply_hygiene.py` against a scratch copy with ONE guard removed at a time.
Control first (must be green); exit code = number of survivors.
"""
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["apply_hygiene.py", "test_apply_hygiene.py", "tools/backlog_drift.py",
         "tools/tests/fixtures/backlog_rows_2026-10-22.md"]
TIP = HERE.parent / "raw" / "BACKLOG.tip_blob.md"

MUTATIONS = [
    ("A1_anchor_check_removed", "a row whose closure evidence vanished is struck anyway",
     "        if anchor not in body:\n            problems.append(\"%s: closure evidence",
     "        if False:\n            problems.append(\"%s: closure evidence"),
    ("A2_duplicate_ids_allowed", "a duplicated id is no longer a refusal",
     "        if len(hits) == 1:\n            return hits[0]\n",
     "        if len(hits) >= 1:\n            return hits[0]\n"),
    ("A3_no_deletion_guard_off", "the never-delete-text contract is not enforced",
     "        if not _no_text_lost(orig, new, inserted):\n",
     "        if False:\n"),
    ("A4_not_idempotent", "an already-struck row is struck AGAIN",
     "        if \"~~\" in _FIRST_CELL.match(body).group(1):\n            report[\"already_struck\"]",
     "        if False:\n            report[\"already_struck\"]"),
    ("A5_relabel_marker_ignored", "a relabel is re-inserted on every run",
     "        if marker in body:\n            report[\"already_relabelled\"].append(rid)\n            continue\n",
     "        if False:\n            continue\n"),
    ("A6_eol_normalised", "CRLF input comes out LF",
     "        lines[i] = new + eol\n",
     "        lines[i] = new + chr(10)\n"),
    ("A7_c2_strike_kept", "C2's display-strike is left in place",
     "            new = C2_BLOCKER.sub(lambda m: m.group(1) + C2_INLINE_TAG, body, count=1)\n",
     "            new = body\n"),
]


def run(root: Path):
    r = subprocess.run(
        ["uv", "run", "--no-project", "--with", "pytest", "python", "-m", "pytest", "-q",
         "-p", "no:cacheprovider", "-rf", "test_apply_hygiene.py"],
        cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    red = sorted(set(re.findall(r"^FAILED\s+\S+::(\S+)", r.stdout, re.M)))
    tail = [ln for ln in r.stdout.splitlines()
            if re.search(r"\d+ (passed|failed)", ln) and not ln.startswith("FAILED")]
    return r.returncode, red, (tail[-1] if tail else r.stdout[-160:])


def main() -> int:
    before = hashlib.md5((HERE / "apply_hygiene.py").read_bytes()).hexdigest()
    src = (HERE / "apply_hygiene.py").read_text(encoding="utf-8")
    survivors = 0
    with tempfile.TemporaryDirectory(prefix="blhyg_mutA_") as td:
        root = Path(td) / "code"
        for rel in FILES:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HERE / rel, root / rel)
        (Path(td) / "raw").mkdir()
        shutil.copyfile(TIP, Path(td) / "raw" / "BACKLOG.tip_blob.md")
        rc, red, tail = run(root)
        print("CONTROL (unmutated)  rc=%d  %s" % (rc, tail))
        if rc != 0:
            print("  CONTROL IS RED: %s" % red)
            return 99
        for name, what, find, repl in MUTATIONS:
            if find not in src:
                print("\n%-28s PATCH TARGET NOT FOUND" % name)
                survivors += 1
                continue
            (root / "apply_hygiene.py").write_text(src.replace(find, repl, 1),
                                                   encoding="utf-8", newline="")
            rc, red, tail = run(root)
            survivors += rc == 0
            print("\n%-28s rc=%d  %s" % (name, rc, "RED (killed)" if rc else "GREEN (SURVIVED)"))
            print("  mutation : %s\n  result   : %s" % (what, tail))
            for t in red[:6]:
                print("    RED  %s" % t)
        (root / "apply_hygiene.py").write_text(src, encoding="utf-8", newline="")
    after = hashlib.md5((HERE / "apply_hygiene.py").read_bytes()).hexdigest()
    print("\napply_hygiene.py md5 %s -> %s (%s)" % (before, after,
          "UNCHANGED" if before == after else "CHANGED -- BUG"))
    print("survivors: %d of %d" % (survivors, len(MUTATIONS)))
    return survivors


if __name__ == "__main__":
    sys.exit(main())
