"""Local driver: build the import closure, ship it to a Colab session, bring
the kernel up. Everything after this is `colab exec -f <step>.py`.

Reuses the MEASURED headless path of `…/2026-08-16-sam3-backfill-run/`
SAM3_BACKFILL_RUN.md §2 — there is no Drive mount in a CLI session, so the
16-file import closure travels as one tar.

⛔ THE TOKEN. Read from `Keys.txt` IN PLACE, written to one scratchpad file,
uploaded, and deleted on BOTH sides in the same step by `bootstrap_v2.py`. It
never enters argv, never a repo file, never stdout. (`Keys.txt` is git-ignored
and must stay that way.)

⚠️ MSYS: `colab upload <local> /content/x` from Git Bash rewrites `/content/…`
into `C:/Program Files/Git/content/…` and the VM's contents API 500s. Windows
local path + POSIX remote path + `MSYS_NO_PATHCONV=1` is the working triple —
so this driver shells out itself rather than being a bash one-liner.

usage:  python ship.py <session-name>
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tarfile
import tempfile

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
COLAB = r"C:\Users\Admin\venvs\colab\Scripts\colab.exe"
SHIM = os.path.join(REPO, "colab", "win_shims")
HERE = os.path.dirname(os.path.abspath(__file__))

CLOSURE = [
    "colab/s2_lab_lib.py", "colab/s2_schema.py",
    "colab/fixtures/sam3_backfill_expected.json",
    "stack/scripts/ph0_v2.py", "stack/scripts/ph0_sam3.py",
    "stack/scripts/ph0_pilot.py", "stack/scripts/v2_to_pilot.py",
    "stack/scripts/ph1_fuse.py", "stack/scripts/ph0_rich_overlay.py",
    "stack/tanitad/__init__.py",
    "stack/tanitad/data/__init__.py", "stack/tanitad/data/_contract.py",
    "stack/tanitad/data/toy_driving.py", "stack/tanitad/data/metadrive_env.py",
    "stack/tanitad/data/comma2k19.py", "stack/tanitad/data/stats.py",
    "stack/tanitad/data/calib.py", "stack/tanitad/data/v2_dataset.py",
]


def run(*args, **kw):
    # ⚠️ PYTHONUTF8=1 is REQUIRED: colab-cli 0.6.0 opens the script with the
    # locale codec, and on this box that is cp1252 — any file carrying a ⛔/⚠️
    # dies with `UnicodeDecodeError: 'charmap' codec can't decode byte 0x8f`
    # before a single line reaches the VM. MEASURED 2026-08-16.
    env = dict(os.environ, PYTHONPATH=SHIM, MSYS_NO_PATHCONV="1",
               PYTHONUTF8="1")
    p = subprocess.run([COLAB, *args], env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", **kw)
    sys.stdout.write(p.stdout or "")
    sys.stderr.write(p.stderr or "")
    return p


def build_closure(dst: str) -> int:
    with tarfile.open(dst, "w:gz") as tf:
        for rel in CLOSURE:
            src = os.path.join(REPO, rel.replace("/", os.sep))
            if not os.path.exists(src):
                raise SystemExit(f"closure member MISSING: {rel}")
            tf.add(src, arcname=rel)
    return os.path.getsize(dst)


def token_file(dst: str) -> None:
    """Read the token IN PLACE and write exactly it. Never printed, never
    returned — the only copy that leaves this process goes to `dst`, which
    `bootstrap_v2.py` deletes on the VM and the caller deletes here."""
    with open(os.path.join(REPO, "Keys.txt"), encoding="utf-8",
              errors="replace") as fh:
        m = re.findall(r"hf_[A-Za-z0-9]+", fh.read())
    if not m:
        raise SystemExit("no HF token found in Keys.txt")
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(max(m, key=len))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    sess = argv[0] if argv else "tanitad-sam3v2"
    tmp = tempfile.mkdtemp(prefix="sam3v2-")
    tgz, tokf = os.path.join(tmp, "repo_closure.tgz"), os.path.join(tmp, "t")
    try:
        print(f"[ship] closure {build_closure(tgz)} B -> {sess}")
        token_file(tokf)
        for local, remote in ((tgz, "/content/repo_closure.tgz"),
                              (tokf, "/content/hf_tok.txt"),
                              (os.path.join(HERE, "bootstrap_v2.py"),
                               "/content/bootstrap_v2.py")):
            r = run("upload", "-s", sess, local, remote)
            if r.returncode:
                return r.returncode
    finally:
        for f in (tgz, tokf):
            if os.path.exists(f):
                os.remove(f)                     # local copy: seconds of life
        os.rmdir(tmp)
    return run("exec", "-s", sess, "-f", os.path.join(HERE, "bootstrap_v2.py"),
               "--timeout", "900").returncode


if __name__ == "__main__":
    raise SystemExit(main())
