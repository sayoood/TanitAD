"""Mutation proof for code/ra1_full_amend2.py: every mutation must turn stack/tests/test_ra1_full_amend2.py RED; the unmutated
reader must be GREEN before and after.  python mutate_amend2.py <repo root> <python exe>"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
PY = sys.argv[2]
READER = ROOT / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-refav1-hj-mini" / "code" / "ra1_full_amend2.py"
MUT = [
    ("separation is not required", 'ok = all(r["below"] and r["separated"] and r["beats_2x_floor"] for r in rows)',
     'ok = all(r["below"] and r["beats_2x_floor"] for r in rows)'),
    ("the floor is A1g200's alone, not the larger", 'two_F = 2 * max(F["A1g200"], F["A0"])', 'two_F = 2 * F["A1g200"]'),
    ("one seed suffices", 'ok = all(r["below"] and r["separated"] and r["beats_2x_floor"] for r in rows)',
     'ok = any(r["below"] and r["separated"] and r["beats_2x_floor"] for r in rows)'),
    ("the built gamma is not checked", "        if got != BUILT[cell]:", "        if False:"),
    ("a missing run is not refused", "    if miss:", "    if False:"),
    ("A1g200 is paired with A1g50, not A0", 'g, z = runs[("A1g200", s)], runs[("A0", s)]', 'g, z = runs[("A1g200", s)], runs[("A1g50", s)]'),
    ("the families pair a run with itself", "return E.paired_family_block(pa, pb,", "return E.paired_family_block(pa, pa,"),
    ("the sign of the bar is dropped", 'below=d["delta"] < 0', 'below=True'),
]


def run():
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "stack/tests/test_ra1_full_amend2.py"], cwd=ROOT,
                       capture_output=True, text=True, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "-1"})
    return r.returncode, r.stdout.strip().splitlines()[-1]


orig = READER.read_bytes()
rc, tail = run()
print("baseline:", rc, tail)
assert rc == 0
red = 0
try:
    for name, a, b in MUT:
        src = orig.decode("utf-8")
        assert src.count(a) == 1, name
        READER.write_bytes(src.replace(a, b).encode("utf-8"))
        rc, tail = run()
        red += rc != 0
        print(f"{'RED ' if rc else 'SURVIVED'} {name}: {tail}")
finally:
    READER.write_bytes(orig)
rc, tail = run()
print("restored:", rc, tail, "| byte-identical:", READER.read_bytes() == orig)
print(f"{red}/{len(MUT)} RED")
sys.exit(0 if red == len(MUT) and rc == 0 else 1)
