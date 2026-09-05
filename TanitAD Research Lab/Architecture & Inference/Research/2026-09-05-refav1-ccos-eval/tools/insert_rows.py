"""INSERT one row into GOALS_AND_CLAIMS.md (after the live-claims table header) and one row
into MODEL_REGISTRY.md (after the §2.4 `| **Eval** |` row), read-modify-write with retries on
the flapping mount, idempotent by marker, verified by re-reading. Never rewrites anything else.

Usage: python insert_rows.py <goals_row_file> <registry_row_file> [--update]
  --update replaces an existing row carrying the marker instead of refusing.
"""
import os, sys, time

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
GOALS = os.path.join(REPO, "Project Steering", "GOALS_AND_CLAIMS.md")
REG = os.path.join(REPO, "Project Steering", "MODEL_REGISTRY.md")
MARK_G = "| D-REFAV1-CCOS-EVAL |"
MARK_R = "| **Eval — `ccos` (2026-09-05, D-REFAV1-CCOS-EVAL)** |"
G_ANCHOR = "| id | claim / hypothesis | status | evidence |\n|---|---|---|---|\n"
R_ANCHOR_PREFIX = "| **Eval** | ✅ **DONE 2026-09-04, OPEN LOOP.**"


def rd(p):
    for i in range(30):
        try:
            with open(p, encoding="utf-8", newline="") as fh:
                return fh.read()
        except OSError as e:
            print("  retry read", i, os.path.basename(p), e, flush=True); time.sleep(10)
    raise SystemExit("cannot read " + p)


def wr(p, s):
    for i in range(30):
        try:
            with open(p, "w", encoding="utf-8", newline="") as fh:
                fh.write(s)
            if rd(p) == s:
                return True
            print("  re-read differs after write; retrying", flush=True)
        except OSError as e:
            print("  retry write", i, os.path.basename(p), e, flush=True); time.sleep(10)
    return False


def upsert(path, marker, row, locate, update):
    s = rd(path); nl = "\r\n" if "\r\n" in s else "\n"
    lines = s.split(nl)
    row = row.rstrip("\r\n")
    hit = [i for i, l in enumerate(lines) if l.startswith(marker)]
    if hit:
        if not update:
            print("PRESENT already:", os.path.basename(path), "line", hit[0] + 1); return True
        lines[hit[0]] = row
    else:
        idx = locate(lines)
        if idx is None:
            print("ANCHOR NOT FOUND in", path); return False
        lines.insert(idx, row)
    out = nl.join(lines)
    if len(out) < len(s) - 10:
        print("REFUSING: output shorter than input"); return False
    ok = wr(path, out)
    got = rd(path)
    present = any(l.startswith(marker) for l in got.split(nl))
    print(("OK   " if (ok and present) else "FAIL ") + os.path.basename(path), "bytes", len(s), "->", len(got))
    return ok and present


def locate_goals(lines):
    for i in range(len(lines) - 1):
        if lines[i].startswith("| id | claim / hypothesis |") and lines[i + 1].startswith("|---|"):
            return i + 2
    return None


def locate_reg(lines):
    for i, l in enumerate(lines):
        if l.startswith(R_ANCHOR_PREFIX):
            return i + 1
    return None


def main():
    update = "--update" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    g_row = open(args[0], encoding="utf-8").read()
    r_row = open(args[1], encoding="utf-8").read()
    ok1 = upsert(GOALS, MARK_G, g_row, locate_goals, update)
    ok2 = upsert(REG, MARK_R, r_row, locate_reg, update)
    print("ALL_OK" if (ok1 and ok2) else "SOME_FAILED")
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
