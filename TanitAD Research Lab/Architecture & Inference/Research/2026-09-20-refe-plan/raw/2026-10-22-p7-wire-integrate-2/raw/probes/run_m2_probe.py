"""Driver for probe_m2.py: build the M2-mutated copy of the tree under test, run the probe on both, write the two outputs.
usage: python run_m2_probe.py <refe dir> <out dir>"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

refe, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
here = Path(__file__).resolve().parent
out.mkdir(parents=True, exist_ok=True)
OLD = '    if missing:\n        return verdict("G-RESERVE", reasons, det, arm=arm, rehearsal=rehearsal)\n'
NEW = '    if False:\n        return verdict("G-RESERVE", reasons, det, arm=arm, rehearsal=rehearsal)\n'
with tempfile.TemporaryDirectory() as td:
    mut = Path(td) / "refe"
    shutil.copytree(refe, mut, ignore=shutil.ignore_patterns("__pycache__"))
    f = mut / "gate_checks.py"
    s = f.read_text(encoding="utf-8")
    assert s.count(OLD) == 1, s.count(OLD)
    f.write_text(s.replace(OLD, NEW), encoding="utf-8", newline="")
    for name, d in (("original", refe), ("mutated_M2", mut)):
        r = subprocess.run([sys.executable, str(here / "probe_m2.py"), str(d)], capture_output=True, text=True, encoding="utf-8")
        lines = [ln for ln in r.stdout.splitlines() if " | " in ln]
        (out / f"probe_m2_{name}.out").write_text("\n".join(lines) + "\n" + (r.stderr[-400:] if r.returncode else ""), encoding="utf-8", newline="\n")
        print(f"--- {name} (rc {r.returncode})")
        print("\n".join(lines))
