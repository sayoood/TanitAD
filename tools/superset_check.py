"""Refuse a REWRITE-class MOD that silently DROPS evidence the landed version carried.

⛔ The lander's append-only MOD guard (`new_lines > tip_lines` and the tip's last
non-blank line survives) is a PROXY for "not a revert". It is the wrong proxy for a
document whose headline legitimately changes -- and it is also WEAKER than it looks:
a rewrite can keep the last line, add 40 lines, and still delete every number in the
middle. MEASURED 2026-09-17: my own 2-arm rewrite of RESULT.md dropped seven measured
values the 1-arm version carried (`human_pdms`, `fan_pdms_mean`, `fan_nc_fail_frac`,
`sel_ep`, `sel_comfort`, `traj_matches_fan_sel`, `is_release`) while passing every
other check I had.

⇒ This checks the thing the CLAUDE.md rule actually says: *your blob must contain
HEAD's content PLUS your change*. The admissible unit of "content" for a research
artifact is its NUMBERS and its CODE IDENTIFIERS, so every one of them must survive
or be explicitly waived with a reason.
"""
from __future__ import annotations
import re, sys, json, subprocess, pathlib

# A number: optional sign, digits with optional thousands commas, optional fraction,
# optional exponent. Anchored so it is not a fragment of a longer token.
NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])")
CODE = re.compile(r"`([^`\n]{2,})`")

def norm_num(s: str) -> str:
    s = s.replace(",", "").lstrip("+")
    if "." in s and "e" not in s.lower():
        s = s.rstrip("0").rstrip(".") or "0"
    return s

def tokens(text: str):
    nums = {norm_num(m.group(0)) for m in NUM.finditer(text)}
    nums = {n for n in nums if n not in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-1"}}
    code = {m.group(1).strip() for m in CODE.finditer(text)}
    return nums, code

def check(tip_text: str, new_text: str, allow: set[str]):
    tn, tc = tokens(tip_text)
    nn, nc = tokens(new_text)
    # a number survives if the SAME normalised value appears anywhere in the new text
    nn_all = {norm_num(m.group(0)) for m in NUM.finditer(new_text)}
    miss_n = sorted(n for n in tn if n not in nn_all and n not in allow)
    miss_c = sorted(c for c in tc if c not in nc and c not in new_text and c not in allow)
    return miss_n, miss_c, len(tn), len(tc)

def main(argv=None) -> int:
    a = argv or sys.argv[1:]
    tip_blob, new_file = a[0], a[1]   # "<ref>:<path>" and the candidate file
    allow_file = a[2] if len(a) > 2 else None
    import os
    env = dict(os.environ)
    if len(a) > 3:
        env["GIT_DIR"] = a[3]
    tip = subprocess.run(["git", "cat-file", "-p", tip_blob], env=env,
                         capture_output=True).stdout.decode("utf-8", "replace")
    if not tip.strip():
        print("ZZSUPERSET-INCONCLUSIVE-EMPTY-TIPZZ"); return 2
    new = pathlib.Path(new_file).read_text(encoding="utf-8")
    allow = set()
    if allow_file and pathlib.Path(allow_file).exists():
        allow = {l.split("#")[0].strip() for l in
                 pathlib.Path(allow_file).read_text(encoding="utf-8").splitlines()
                 if l.split("#")[0].strip()}
    mn, mc, tot_n, tot_c = check(tip, new, allow)
    print(f"tip tokens: {tot_n} numbers, {tot_c} code spans | waived: {len(allow)}")
    for n in mn:
        print(f"  DROPPED-NUMBER {n}")
    for c in mc:
        print(f"  DROPPED-CODE   `{c}`")
    if mn or mc:
        print(f"ZZSUPERSET-FAIL {len(mn)} numbers, {len(mc)} code spans dropped"
              f" -- each must survive or be waived WITH A REASON in the .allowdrop fileZZ")
        return 1
    print("ZZSUPERSET-OKZZ")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
