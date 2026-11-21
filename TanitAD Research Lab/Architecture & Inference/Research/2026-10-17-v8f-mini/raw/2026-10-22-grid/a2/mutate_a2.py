"""Amendment 2's mutants: each edits the patched reader, runs the a2 tests, and must make them FAIL; the reader is restored byte for byte after each.
   python mutate_a2.py <export root> <python>"""
import shutil
import subprocess
import sys
from pathlib import Path

X, PY = Path(sys.argv[1]), sys.argv[2]
RD = X / "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/v8f_grid_read.py"
SAFE = RD.with_suffix(".a2safe")
shutil.copyfile(RD, SAFE)
MUTS = {
    "no fill (B_UNRECORDED_W empty)": ('B_UNRECORDED_W = {"w_tac": (1.0, "--w-tac"), "w_cons": (0.5, "--w-cons")}', "B_UNRECORDED_W = {}"),
    "fill at the wrong value (w_tac 2.0)": ('"w_tac": (1.0, "--w-tac")', '"w_tac": (2.0, "--w-tac")'),
    "argv guard removed": (" or B_UNRECORDED_W[k][1] in b_flags", ""),
    "fill a key B records too": ("        if k in wb:\n", "        if k in wb and k not in B_UNRECORDED_W:\n"),
    "w compared as one dict again": ('"residual_zero_init", "w")', '"residual_zero_init")'),
    "w not compared at all": ('            diff += [f"w.{k}" for k in w_diff(lg.get("w") or {}, lb.get("w") or {}, rfl)]\n', ""),
}
env = {"PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1"}
import os
env = dict(os.environ, **env, PYTHONPATH=f"{X / 'stack'};{X / 'taniteval'}")
def run():
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_v8f_grid.py", "-k", "a2_"], cwd=X / "stack", env=env,
                       capture_output=True, text=True, encoding="utf-8")
    return r.returncode, (r.stdout.strip().splitlines() or ["?"])[-1]
rc, tail = run()
print(f"clean: rc {rc}  {tail}")
assert rc == 0
n_red = 0
for name, (a, b) in MUTS.items():
    src = SAFE.read_text(encoding="utf-8")
    assert src.count(a) == 1, name
    RD.write_text(src.replace(a, b), encoding="utf-8", newline="\n")
    rc, tail = run()
    red = rc != 0
    n_red += red
    print(f"{'RED  ' if red else 'GREEN'} {name}: rc {rc}  {tail}")
    shutil.copyfile(SAFE, RD)
assert RD.read_bytes() == SAFE.read_bytes()
rc, tail = run()
print(f"restored (bytes equal); clean again: rc {rc}  {tail}")
print(f"{n_red} / {len(MUTS)} mutants RED")
