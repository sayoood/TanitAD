"""Build code/fix_a19/ -- the A19 gate change STACKED ON the MAPLIFT blobs (not on a git ref: the
MAPLIFT candidate lands in the same gate): per file the FULL new file, one unified diff against the
MAPLIFT base (code/fix_maplift/...), VERIFIED by `git apply` on a scratch copy of the bases
(byte-for-byte), and LANDING_BLOCKS.txt in the Master Mind's Thor-gate format with
`base <MAPLIFT blob>`. Usage: make_a19_bundle.py <candidate tree> <package dir>"""
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CAND, PKG = Path(sys.argv[1]), Path(sys.argv[2])
PKG_REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-map-hires"
FILES = {"stack/scripts/launch_gate.py": "ea55acc6268a3062c9c04617c298823598f40ad3",
         "stack/tests/test_launch_gate.py": "675cea50b45831a5b6beb86ff9550928ddd74ca7"}


def blob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


fix = PKG / "code" / "fix_a19"
if fix.exists():
    shutil.rmtree(fix)
fix.mkdir(parents=True)
rows, diffs = [], []
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    for f, base_blob in FILES.items():
        base = (PKG / "code" / "fix_maplift" / f).read_bytes()
        assert blob(base) == base_blob, (f, blob(base))
        new = (CAND / f).read_bytes()
        assert b"\r\n" not in new and b"\r\n" not in base, f
        for root, data in ((td / "a", base), (td / "b", new), (fix, new)):
            (root / f).parent.mkdir(parents=True, exist_ok=True)
            (root / f).write_bytes(data)
        r = subprocess.run(["git", "-c", "core.autocrlf=false", "diff", "--no-index", "--no-color",
                            "--no-prefix", f"a/{f}", f"b/{f}"], cwd=td, capture_output=True)
        assert r.returncode in (0, 1), r.stderr
        diffs.append(r.stdout)
        rows.append((f, base_blob, blob(new)))
    dpath = fix / "NEW2_A19_edits.diff"
    dpath.write_bytes(b"".join(diffs))
    vt = td / "verify"
    for f in FILES:
        (vt / f).parent.mkdir(parents=True, exist_ok=True)
        (vt / f).write_bytes((td / "a" / f).read_bytes())
    subprocess.run(["git", "init", "-q"], cwd=vt, check=True)
    r = subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "--whitespace=nowarn",
                        str(dpath)], cwd=vt, capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
    bad = [f for f in FILES if (vt / f).read_bytes() != (fix / f).read_bytes()]
(fix / "BASE_BLOBS.txt").write_bytes(("# base = the MAPLIFT candidate's NEW blobs (LANDING_READY_MAPLIFT"
                                      ".txt), not a git ref\n" + "".join(
    f"{b}  {f}  (LF blob, code/fix_maplift/{f})\n" for f, b, _n in rows)).encode("utf-8"))
blocks = ["# A19 (every file EDITED against the MAPLIFT blobs)"]
for f, b, n in rows:
    blocks += [f"# {f}: base {b} | new {n} | LF", f"{PKG_REL}/code/fix_a19/{f}  ->  {f}"]
(fix / "LANDING_BLOCKS.txt").write_bytes(("\n".join(blocks) + "\n").encode("utf-8"))
print(f"{len(rows)} files; diff {dpath.read_bytes().count(b'\n')} lines; verify "
      f"{'MATCH' if not bad else 'MISMATCH ' + str(bad)}")
for f, b, n in rows:
    print(f"  {f}: {b[:8]} -> {n}")
