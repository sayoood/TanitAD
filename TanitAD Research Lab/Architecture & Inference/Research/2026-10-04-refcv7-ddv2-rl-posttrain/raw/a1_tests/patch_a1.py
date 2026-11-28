"""Apply SPEC_RL_A1 to the three WP-RL mirror files: CRLF-preserving, count==1 anchors, ASCII-only.

Usage: python patch_a1.py <package code/stack dir> [--dry-run]
"""
import hashlib
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
stack = Path(sys.argv[1])
dry = "--dry-run" in sys.argv[2:]

lib_p = stack / "tanitad" / "rl" / "ddv2_refcv7.py"
trn_p = stack / "scripts" / "ddv2_rl_refcv7.py"
tst_p = stack / "tests" / "test_ddv2_refcv7.py"

EXPECT = {lib_p: "701638df39c958d5d1b8580b4c0aef14", trn_p: "b46d4014128a608b3abaa56c720dc2c1",
          tst_p: "b049a992bf5c90aacc3b5e232eb013d7"}


def crlf(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\n", "\r\n")


def md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


src = {}
for p, want in EXPECT.items():
    b = p.read_bytes()
    got = md5(b)
    if got != want:
        sys.exit(f"REFUSE: {p.name} md5 {got} != expected pre-patch {want} (already patched or drifted)")
    src[p] = b.decode("ascii")

# ---- trainer -----------------------------------------------------------------------------
edits = runpy.run_path(str(HERE / "trainer_edits.py"))["EDITS"]
t = src[trn_p]
for k, (old, new) in enumerate(edits):
    o, n = crlf(old), crlf(new)
    c = t.count(o)
    if c != 1:
        sys.exit(f"REFUSE: trainer edit {k} anchor count {c} != 1:\n{old[:200]}")
    t = t.replace(o, n)

# ---- library -----------------------------------------------------------------------------
L = src[lib_p]
anchor = '    "TRAINABLE_PREFIXES",\r\n]'
if L.count(anchor) != 1:
    sys.exit(f"REFUSE: library __all__ anchor count {L.count(anchor)}")
L = L.replace(anchor, '    "TRAINABLE_PREFIXES", "step_kl_to_reference", "swapped_params", "halt_check",\r\n'
                      '    "HALT_KEYS",\r\n]')
add = (HERE / "lib_add.py").read_text(encoding="ascii").lstrip("\n")
if not L.endswith("\r\n"):
    L += "\r\n"
L = L + "\r\n\r\n" + crlf(add)
if not L.endswith("\r\n"):
    L += "\r\n"

# ---- tests -------------------------------------------------------------------------------
T = src[tst_p]
tadd = (HERE / "tests_add.py").read_text(encoding="ascii").lstrip("\n")
if not T.endswith("\r\n"):
    T += "\r\n"
T = T + "\r\n\r\n" + crlf(tadd)
if not T.endswith("\r\n"):
    T += "\r\n"

out = {lib_p: L, trn_p: t, tst_p: T}
for p, s in out.items():
    s.encode("ascii")                           # raises on any non-ASCII byte
    assert "\n" not in s.replace("\r\n", ""), f"bare LF in {p.name}"
    b = s.encode("ascii")
    print(f"{p.name}: {len(src[p].splitlines())} -> {len(s.splitlines())} lines, md5 {md5(b)}"
          + ("  (dry-run)" if dry else ""))
    if not dry:
        p.write_bytes(b)
print("OK")
