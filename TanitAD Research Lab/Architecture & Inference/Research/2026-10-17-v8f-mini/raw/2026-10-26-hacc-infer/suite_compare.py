#!/usr/bin/env python3
"""Compare two ``pytest -q -rfEs`` logs (base = c8ea1623, final = with the HACC post-map): a REGRESSION is a test that fails or errors on FINAL only.
  python suite_compare.py <base.log> <final.log> [--out suite_compare.txt]"""
import re
import sys
from pathlib import Path


def parse(path):
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    bad = {}
    for ln in txt.splitlines():
        m = re.match(r"^(FAILED|ERROR) (\S+)", ln)
        if m:
            bad[m.group(2)] = m.group(1)
    summ = [ln for ln in txt.splitlines() if re.search(r"\b(passed|failed|error)\b.* in [\d.]+s", ln)]
    code = re.findall(r"EXIT_CODE=(\d+)", txt)
    return bad, (" ;; ".join(x.strip() for x in summ) if summ else "NO SUMMARY LINE"), (",".join(code) if code else "NOT FINISHED")   # a log may hold SEVERAL pytest runs (the final suite ran in two halves)


def main():
    a, b = sys.argv[1], sys.argv[2]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    base, sb, cb = parse(a)
    fin, sf, cf = parse(b)
    lines = [f"base : exit {cb}  |  {sb}", f"final: exit {cf}  |  {sf}", f"failing on base : {len(base)}", f"failing on final: {len(fin)}"]
    reg = sorted(set(fin) - set(base))
    fixed = sorted(set(base) - set(fin))
    lines.append(f"REGRESSIONS (fail/error on FINAL only): {len(reg)}")
    lines += [f"  {fin[k]} {k}" for k in reg]
    lines.append(f"failing on base only: {len(fixed)}")
    lines += [f"  {base[k]} {k}" for k in fixed]
    lines.append(f"failing on both (pre-existing): {len(set(base) & set(fin))}")
    lines += [f"  {base[k]} {k}" for k in sorted(set(base) & set(fin))]
    if cb == "NOT FINISHED" or cf == "NOT FINISHED":
        lines.append("INCOMPLETE: at least one log has not finished")
    txt = "\n".join(lines)
    print(txt)
    if out:
        Path(out).write_text(txt + "\n", encoding="utf-8")
    return 0 if (not reg and cb != "NOT FINISHED" and cf != "NOT FINISHED") else 1


if __name__ == "__main__":
    sys.exit(main())
