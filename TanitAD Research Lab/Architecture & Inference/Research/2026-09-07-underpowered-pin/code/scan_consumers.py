"""Mechanism B: pure-Python walk + read. Independent of ripgrep/git/shell grep.

Every count is paired with a control that MUST read non-zero; a zero control
means the mount flapped and the result is INCONCLUSIVE, not an absence.
"""
import os
import sys

ROOT = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"

NEEDLES = [
    "TACTICAL_GOAL_UNDERPOWERED",
    "GOAL_MIN_N_FOR_METRIC",
]
# controls: symbols from the SAME module that are known to be consumed
CONTROLS = [
    "TACTICAL_GOAL_TOKENS_V7",
    "vocab_v7",
]

SCOPES = [
    "stack/scripts",
    "stack/tests",
    "stack/tanitad",
    "taniteval",
    "tools",
    "FlyWheels",
]

EXTS = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt", ".ipynb"}


def walk(scope):
    base = os.path.join(ROOT, scope.replace("/", os.sep))
    if not os.path.isdir(base):
        return None
    out = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames
                       if d not in (".git", "__pycache__", ".ipynb_checkpoints")]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in EXTS:
                out.append(os.path.join(dirpath, fn))
    return out


def main():
    grand = {n: [] for n in NEEDLES + CONTROLS}
    for scope in SCOPES:
        files = walk(scope)
        if files is None:
            print("SCOPE %-16s : DIR MISSING -> INCONCLUSIVE" % scope)
            continue
        hits = {n: [] for n in NEEDLES + CONTROLS}
        read_ok = 0
        read_fail = 0
        for path in files:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
                read_ok += 1
            except OSError as exc:
                read_fail += 1
                print("  READ-FAIL %s : %s" % (path, exc))
                continue
            for n in NEEDLES + CONTROLS:
                if n in text:
                    for i, line in enumerate(text.splitlines(), 1):
                        if n in line:
                            hits[n].append((path, i, line.strip()[:150]))
        print("=" * 78)
        print("SCOPE %s : files=%d read_ok=%d read_fail=%d" %
              (scope, len(files), read_ok, read_fail))
        for n in CONTROLS:
            mark = "OK" if hits[n] else "ZERO-CONTROL -> MOUNT FLAP, INCONCLUSIVE"
            print("  CONTROL %-26s %4d  %s" % (n, len(hits[n]), mark))
        for n in NEEDLES:
            print("  NEEDLE  %-26s %4d" % (n, len(hits[n])))
            for path, i, line in hits[n]:
                rel = path[len(ROOT) + 1:]
                print("      %s:%d: %s" % (rel, i, line))
        for n in NEEDLES + CONTROLS:
            grand[n] += hits[n]
    print("=" * 78)
    print("GRAND TOTALS")
    for n in CONTROLS:
        print("  CONTROL %-26s %4d" % (n, len(grand[n])))
    for n in NEEDLES:
        print("  NEEDLE  %-26s %4d" % (n, len(grand[n])))


if __name__ == "__main__":
    sys.exit(main())
