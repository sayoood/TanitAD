"""Classify the failures of the patched tree against the unmodified tip: python compare_suites.py [logs_dir]   (writes logs/COMPARE.txt as well)

Input: logs/files_NEW.log (every test file of the patched tree, one pytest process per file) and logs/files_TIP.log (the SAME files that failed on NEW, run on the unmodified tip export); both written by
run_files.py. A failure id is a FAILED / ERROR line up to the first ' - '.
  NEW-ONLY  a failing id on NEW that does not fail on the tip: a regression of the patch (must be empty)
  BOTH      fails on both: pre-existing
  TIP-ONLY  fails on the tip only (fixed by the patch, or flaky)
Exit code 0 when NEW-ONLY is empty and both logs reached ALL-DONE; 1 when NEW-ONLY is not empty; 2 when a log is unfinished."""
import re
import sys
from pathlib import Path

d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "logs"


def load(name):
    p = d / name
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8", errors="replace")
    files = {m.group(1): (int(m.group(2)), m.group(3)) for m in re.finditer(r"^FILE (\S+) rc=(-?\d+) (.*)$", txt, re.M)}
    fails = {re.split(r"\s+-\s+", ln.strip(), maxsplit=1)[0] for ln in txt.splitlines() if re.match(r"^(FAILED|ERROR)\s", ln)}
    return files, fails, "ALL-DONE" in txt


def count(files, key):
    return sum(int(m.group(1)) for _, s in files.values() for m in [re.search(r"(\d+) " + key, s)] if m)


new, tip = load("files_NEW.log"), load("files_TIP.log")
out = []
if new is None:
    print("files_NEW.log is missing")
    sys.exit(2)
nf, nfail, ndone = new
out.append(f"NEW: {len(nf)} files, {count(nf, 'passed')} passed, {count(nf, 'failed')} failed, {count(nf, 'error')} errors, {count(nf, 'skipped')} skipped | finished: {ndone}")
bad_files = sorted(f for f, (rc, _) in nf.items() if rc not in (0, 5))
out.append(f"NEW files with rc not in (0, 5): {len(bad_files)}")
if tip is None:
    out += ["files_TIP.log is missing: the failing files have not been run on the tip yet", *bad_files]
    code = 2
else:
    tf, tfail, tdone = tip
    same = {f: nf[f] for f in tf if f in nf}
    out.append(f"TIP (only the files failing on NEW): {len(tf)} files, {count(tf, 'passed')} passed, {count(tf, 'failed')} failed, {count(tf, 'error')} errors | finished: {tdone}")
    out.append(f"the same {len(same)} files on NEW: {count(same, 'passed')} passed, {count(same, 'failed')} failed, {count(same, 'error')} errors")
    missing = [f for f in bad_files if f not in tf]
    only_new, only_tip, both = sorted(nfail - tfail), sorted(tfail - nfail), sorted(nfail & tfail)
    out += [f"\nfailing files on NEW not yet run on the tip ({len(missing)}): {missing}", f"\nNEW-ONLY ({len(only_new)}):", *only_new, f"\nTIP-ONLY ({len(only_tip)}):", *only_tip,
            f"\nBOTH, pre-existing ({len(both)}):", *both[:80]] + ([f"  ... and {len(both) - 80} more"] if len(both) > 80 else [])
    code = 2 if not (ndone and tdone) or missing else (1 if only_new else 0)
txt = "\n".join(out)
print(txt)
(d / "COMPARE.txt").write_text(txt + f"\nexit {code}\n", encoding="utf-8")
sys.exit(code)
