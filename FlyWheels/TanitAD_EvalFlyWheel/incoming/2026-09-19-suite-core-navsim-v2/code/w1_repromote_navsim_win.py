#!/usr/bin/env python3
"""W1 -- RE-PROMOTE devkit_side/navsim_win.py onto E1's CURRENT origin, re-applying the two marked
W1 ADDITION blocks by a 3-way ALIGNMENT (never by line numbers: E1's edit sits ABOVE both blocks
and shifts them).

The binding assertion is the one the promotion test makes: stripping the marked blocks from the
NEW product must reproduce E1's NEW origin BYTE-FOR-BYTE. Anything less is a claim, not a check.
Writes atomically (os.replace) because a scorer subprocess may import this file at any moment.
"""
import difflib, hashlib, os, sys
from pathlib import Path

REPO = Path("D:/Projects/TanitAD")
PROD = REPO / "taniteval/taniteval/bench/navsim/devkit_side/navsim_win.py"
ORIGIN = REPO / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-warmup-reference-epdms/code/navsim_win.py"
PINNED_OLD = "96d13540aa3bb678de1d34577391ced29c2eb07f"
START, END = "# --- W1 ADDITION", "# --- end W1 ADDITION"

def blob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()

def split_blocks(lines):
    """-> (kept_lines, [(insert_at_in_kept, block_lines)])"""
    kept, blocks, i = [], [], 0
    while i < len(lines):
        if START in lines[i] and "end" not in lines[i]:
            j = i
            while j < len(lines) and END not in lines[j]:
                j += 1
            assert j < len(lines), "unterminated W1 ADDITION block"
            blocks.append((len(kept), lines[i:j + 1]))
            i = j + 1
            continue
        kept.append(lines[i]); i += 1
    return kept, blocks

prod_b = PROD.read_bytes()
new_origin_b = ORIGIN.read_bytes()
assert b"\r\n" not in prod_b and b"\r\n" not in new_origin_b, "CRLF present -- refusing (merge would be bogus)"

prod_lines = prod_b.decode("utf-8").split("\n")
kept, blocks = split_blocks(prod_lines)
assert len(blocks) == 2, f"expected 2 W1 ADDITION blocks, found {len(blocks)}"

old_origin_b = ("\n".join(kept)).encode("utf-8")
got = blob(old_origin_b)
print(f"[repromote] reconstructed OLD origin blob {got}")
assert got == PINNED_OLD, f"reconstruction != pinned old origin ({got} != {PINNED_OLD})"
print("[repromote] OK: my product really is E1's OLD file + exactly these two blocks")

old_lines = old_origin_b.decode("utf-8").split("\n")
new_lines = new_origin_b.decode("utf-8").split("\n")

# map an insertion index in OLD -> the same logical point in NEW, via the alignment
sm = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
ops = sm.get_opcodes()
def map_index(io: int) -> int:
    for tag, i1, i2, j1, j2 in ops:
        if i1 <= io < i2 or (io == i2 and tag == "equal"):
            if tag == "equal":
                return j1 + (io - i1)
            return j1 if io == i1 else j2
    return len(new_lines)

out = list(new_lines)
for ins_old, block in sorted(blocks, key=lambda t: -t[0]):
    ins_new = map_index(ins_old)
    ctx_old = old_lines[ins_old - 1] if ins_old else "<bof>"
    ctx_new = out[ins_new - 1] if ins_new else "<bof>"
    assert ctx_old == ctx_new, f"alignment anchor mismatch:\n  old[{ins_old-1}] {ctx_old!r}\n  new[{ins_new-1}] {ctx_new!r}"
    print(f"[repromote] block of {len(block)} lines: old idx {ins_old} -> new idx {ins_new} (anchor OK)")
    out[ins_new:ins_new] = block

new_prod_b = ("\n".join(out)).encode("utf-8")

# THE BINDING ASSERTION: strip the marked blocks -> must be E1's new origin, byte for byte
check_kept, check_blocks = split_blocks(new_prod_b.decode("utf-8").split("\n"))
rt = ("\n".join(check_kept)).encode("utf-8")
assert len(check_blocks) == 2, "block count changed"
assert rt == new_origin_b, "STRIPPING THE MARKED BLOCKS DOES NOT REPRODUCE E1'S NEW ORIGIN -- refusing to write"
print(f"[repromote] BINDING CHECK PASSED: strip(product) == origin byte-for-byte "
      f"({blob(rt)} == {blob(new_origin_b)})")

if "--write" not in sys.argv:
    print("[repromote] DRY RUN (pass --write to apply)"); raise SystemExit(0)

tmp = PROD.with_suffix(".py.w1tmp")
tmp.write_bytes(new_prod_b)
os.replace(tmp, PROD)                      # atomic: a scorer sees old or new, never partial
print(f"[repromote] WROTE {PROD}")
print(f"[repromote] product blob {blob(prod_b)} -> {blob(new_prod_b)}")
print(f"[repromote] origin  blob {PINNED_OLD} -> {blob(new_origin_b)}")
