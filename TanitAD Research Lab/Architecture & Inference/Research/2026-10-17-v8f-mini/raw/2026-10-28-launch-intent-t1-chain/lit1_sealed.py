#!/usr/bin/env python3
"""lit1_sealed.py -- THE SEALED SHELL RUNNER of lit1_selftest.py: every Git-bash step of the self-test goes through `run` here, and nowhere else.

WHY (the incident of 2026-10-28 05:56, recorded in dagland/launch_intent_chain/HANDOFF.md): a previous self-test put a stub remote-shell command first on the PATH it handed to `bash -c`; Git bash PREPENDS /usr/bin at
start-up, so the REAL command ran and reached box 2.  The fix is a GUARD that is part of the SAME shell as the step it protects.  Before the step's text is parsed, that shell

  1. exports the world's stub directory FIRST on PATH (inside the shell, after /usr/bin was prepended);
  2. asserts `command -v ssh` and `command -v scp` ARE the world's stubs (they only log the call and exit 255);
  3. asserts $HOME is the EMPTY scratch directory and that the box-2 alias is UNRESOLVED (`ssh -G <alias>` is a config PRINT: it opens no connection);
  4. asserts every variable the step WRITES or READS through (LIT1_OUT, LIT1_LOG, LIT1_ART, LIT1_WORLD, LIT1_TREE_LI, LIT1_B2_OUT, TMPDIR) resolves INSIDE the scratch world and never inside C:/Users/Admin/box2_ship.

Any failure is `exit 97` with ZZHARNESS-REFUSED, having run NOTHING.  `ARMS` are deliberate-regression arms (a defective harness must be REFUSED); `drop=` removes one check so the self-test can show each is load-bearing.
The chain under test (lit1_chain.sh) never calls a remote-shell command; the guard exists so that no future edit of it or of this harness can reach box 2 unnoticed.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCR = Path("F:/TanitAD-scratch/launch_intent_t1")
BASH = "C:/Program Files/Git/bin/bash.exe"
REAL_SHIP = "C:/Users/Admin/box2_ship"
ALIAS = "tanitad-" + "box2"          # the ssh alias of box 2, spelled in this ONE place; it is only ever PRINTED-CONFIG-CHECKED, never connected to
WRITE_VARS = ("LIT1_OUT", "LIT1_LOG", "LIT1_ART", "LIT1_WORLD", "LIT1_TREE_LI", "LIT1_B2_OUT", "LIT1_READER", "LIT1_PREREG", "TMPDIR")
NOHOME = SCR / "home_empty"
GSTUB = SCR / "global_stub"
STATS = {"sealed_runs": 0, "refused": 0}


def px(p) -> str:
    s = str(p).replace("\\", "/")
    return "/" + s[0].lower() + s[2:] if len(s) > 1 and s[1] == ":" else s


def stub_src(name: str) -> bytes:
    return ('#!/usr/bin/env bash\necho "' + name + ' $*" >> "$LIT1_STUB_LOG"\nexit 255\n').encode()


def make_stub_dir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for n in ("ssh", "scp"):
        (d / n).write_bytes(stub_src(n))
        os.chmod(d / n, 0o755)
    return d


def ensure_global() -> None:
    NOHOME.mkdir(parents=True, exist_ok=True)
    assert not any(NOHOME.iterdir()), f"{NOHOME} must stay EMPTY (it is the sealed shells' $HOME)"
    make_stub_dir(GSTUB)
    (SCR / "tmp").mkdir(parents=True, exist_ok=True)


def prefix(stub, home, need=(), drop=()) -> str:
    """The guard text.  ``drop`` names checks to REMOVE: path_export, ssh_stub, scp_stub, home, home_empty, alias, writepaths, normalise."""
    stub, home = px(stub), px(home)
    scr = px(SCR).lower()
    scr_m = str(SCR).replace("\\", "/").lower()
    real = REAL_SHIP.lower()
    L = ['refuse() { echo "ZZHARNESS-REFUSED: $1: NOTHING was run"; exit 97; }']
    if "path_export" not in drop:
        L.append(f'export PATH="{stub}:$PATH"')
    if "ssh_stub" not in drop:
        L.append(f'[ "$(command -v ssh)" = "{stub}/ssh" ] || refuse "ssh is not the world stub: $(command -v ssh)"')
    if "scp_stub" not in drop:
        L.append(f'[ "$(command -v scp)" = "{stub}/scp" ] || refuse "scp is not the world stub: $(command -v scp)"')
    if "home" not in drop:
        L.append(f'[ "$HOME" = "{home}" ] || refuse "HOME is not the scratch HOME {home}: $HOME"')
    if "home_empty" not in drop:
        L.append('[ -z "$(ls -A "$HOME" 2>/dev/null)" ] || refuse "HOME is not an EMPTY directory (no ssh config may exist)"')
    if "alias" not in drop:
        L.append(f'/usr/bin/ssh -G {ALIAS} 2>/dev/null | grep -qx "hostname {ALIAS}" || refuse "the box-2 alias RESOLVES (a config would send the remote shell to a host)"')
    if "writepaths" not in drop:
        resolve = 'r=$(realpath -m "$1" 2>/dev/null) && r=$(cygpath -m "$r" 2>/dev/null | tr "A-Z" "a-z") || return 1; '
        if "normalise" in drop:
            resolve = 'r=$(echo "$1" | tr "A-Z" "a-z"); '
        L.append('inw() { local r; ' + resolve + f'case "$r" in "{real}"|"{real}"/*) return 1 ;; "{scr_m}"/*|"{scr}"/*) return 0 ;; esac; return 1; }}')
        L.append(f'for v in {" ".join(WRITE_VARS)}; do eval "p=\\${{$v:-}}"; [ -z "$p" ] || inw "$p" || refuse "$v=$p is not inside the scratch world {scr_m} (or is inside {real})"; done')
    return "\n".join(L) + "\n"


def run(cmd: str, env=None, timeout=900, drop=(), cwd=None, stub_dir=None):
    """Run ``cmd`` in Git bash behind the guard.  Returns (rc, stdout+stderr).  rc 97 = the guard refused (nothing ran)."""
    ensure_global()
    stub = Path(stub_dir) if stub_dir else GSTUB
    calls = SCR / "global_stub" / "calls.log"
    e = dict(os.environ)
    e.update({"CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1", "HOME": px(NOHOME), "TMPDIR": px(SCR / "tmp"), "LIT1_STUB_LOG": px(calls), "OMP_NUM_THREADS": "1"})
    e.update(env or {})
    for k in WRITE_VARS:
        if k in e and len(e[k]) > 1 and e[k][1] == ":":
            e[k] = px(e[k])
    e["PATH"] = str(stub) + os.pathsep + e.get("PATH", os.environ["PATH"])      # NOT trusted by the guard: Git bash re-prepends /usr/bin
    full = prefix(stub, NOHOME, drop=drop) + cmd
    STATS["sealed_runs"] += 1
    p = subprocess.Popen([BASH, "-c", full], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=e, cwd=cwd, stdin=subprocess.DEVNULL)
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        out, _ = p.communicate()
        return 124, out.decode("utf-8", "replace") + "\nTIMEOUT"
    if p.returncode == 97:
        STATS["refused"] += 1
    return p.returncode, out.decode("utf-8", "replace")


def popen(cmd: str, env=None, drop=(), stub_dir=None):
    """Like run, detached (for the two-instance test); returns the Popen."""
    ensure_global()
    stub = Path(stub_dir) if stub_dir else GSTUB
    e = dict(os.environ)
    e.update({"CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1", "HOME": px(NOHOME), "TMPDIR": px(SCR / "tmp"), "LIT1_STUB_LOG": px(SCR / "global_stub" / "calls.log"), "OMP_NUM_THREADS": "1"})
    e.update(env or {})
    for k in WRITE_VARS:
        if k in e and len(e[k]) > 1 and e[k][1] == ":":
            e[k] = px(e[k])
    e["PATH"] = str(stub) + os.pathsep + e.get("PATH", os.environ["PATH"])
    STATS["sealed_runs"] += 1
    return subprocess.Popen([BASH, "-c", prefix(stub, NOHOME, drop=drop) + cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=e, stdin=subprocess.DEVNULL)


# the guard's own deliberate-regression arms: (name, env of the defect, checks removed to CREATE the defect, checks removed in the MUTANT).  Guard intact + defect => rc 97 and no witness; the mutant must let the defect through.
def arms():
    return [
        ("old harness: PATH only, no in-shell export", {}, ("path_export",), ("path_export", "ssh_stub", "scp_stub")),
        ("HOME is not the scratch HOME", {"HOME": "/c/Users/Admin"}, (), ("home", "home_empty", "alias")),
        ("a write variable names the real ship directory", {"LIT1_OUT": REAL_SHIP + "/logs/x"}, (), ("writepaths",)),
        ("a write variable escapes the scratch world with ..", {"LIT1_LOG": str(SCR).replace("\\", "/") + "/../../x"}, (), ("normalise",)),
    ]
