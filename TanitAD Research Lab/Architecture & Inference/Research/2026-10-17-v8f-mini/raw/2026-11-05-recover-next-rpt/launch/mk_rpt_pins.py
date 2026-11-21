"""Re-pin the three files the RPT launcher ships over box 2's repo (one command).

The SHIPPED block of box2_files/rnext_check.py binds three files by sha256; their sources are:
  stack/scripts/v8f_mini_train.py       pertick/stack_patch   (the new trainer lever --recover-dev-shape)
  stack/tanitad/data/v8f_recover.py     pertick/stack_patch   (the per-tick deviation shape)
  stack/tanitad/data/v8f_mini_data.py   rpt_stack/            (the git mirror tip 76a0d73c, ClipStore.get_kin, staged byte for byte)
usage:  python -I mk_rpt_pins.py            recompute, rewrite the SHIPPED block when it differs, regenerate launch_rpt.sh, print old -> new
        python -I mk_rpt_pins.py --check    exit 1 when SHIPPED differs from the source files (writes nothing)
"""
import hashlib
import re
import subprocess
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
PERTICK = Path("E:/Projects/TanitAD-artifacts/dagland/pertick/stack_patch")
SOURCES = {"stack/scripts/v8f_mini_train.py": PERTICK / "stack/scripts/v8f_mini_train.py",
           "stack/tanitad/data/v8f_recover.py": PERTICK / "stack/tanitad/data/v8f_recover.py",
           "stack/tanitad/data/v8f_mini_data.py": H / "rpt_stack/stack/tanitad/data/v8f_mini_data.py"}
CHECK = H / "box2_files/rnext_check.py"
BLOCK = re.compile(r'SHIPPED = \{[^}]*\}\n')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def computed():
    return {rel: sha(p) for rel, p in SOURCES.items()}


def block(d):
    items = [f'"{rel}": "{h}"' for rel, h in d.items()]
    return "SHIPPED = {" + (",\n           ").join(items) + "}\n"


def current():
    m = BLOCK.search(CHECK.read_text(encoding="utf-8"))
    assert m, "no SHIPPED block in rnext_check.py"
    return dict(re.findall(r'"([^"]+)": "([0-9a-f]{64})"', m.group(0)))


def main(argv):
    new, old = computed(), current()
    for rel in SOURCES:
        print(f"{rel}\n  pinned   {old.get(rel)}\n  computed {new[rel]}  {'same' if old.get(rel) == new[rel] else 'DIFFERS'}")
    if argv[:1] == ["--check"]:
        return 0 if old == new else 1
    if old != new:
        s = CHECK.read_text(encoding="utf-8")
        assert len(BLOCK.findall(s)) == 1
        CHECK.write_text(BLOCK.sub(lambda _m: block(new), s), encoding="utf-8", newline="\n")
        print("rewrote the SHIPPED block of box2_files/rnext_check.py")
    r = subprocess.run([sys.executable, "-I", str(H / "mk_launch_rpt.py")], capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip())
    return r.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
