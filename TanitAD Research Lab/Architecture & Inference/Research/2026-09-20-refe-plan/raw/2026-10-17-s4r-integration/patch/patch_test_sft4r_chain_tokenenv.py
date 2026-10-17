#!/usr/bin/env python3
"""One-shot patcher (S4R-1, 2026-10-17): test_sft4r_chain.py pins that sft4r_chain.sh takes the HF token-file path from a REQUIRED env var (no default in source, fail closed), and runs bash by an
absolute POSIX path.  usage: ... <refe dir>
The chain is RUN (its precheck only), with a stub `flock` on PATH and SFT4R_WS in a temp dir, in four situations: variable unset, variable empty, variable set (the refusal must then be a LATER one), and two
MUTATIONS of a temp copy of the chain (a default path put back; the check removed) -- both must make the unset case stop refusing on the variable, i.e. the test can go RED.  The test names no token-file path:
the literal is assembled at run time."""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "test_sft4r_chain.py"
s = open(p, encoding="utf-8", newline="").read()
assert "\r\n" not in s


def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)


sub('''def _sh(helpers_path, script):
    py = sys.executable.replace("\\\\", "/")
    r = subprocess.run([shutil.which("bash"), "-c", f'PY_C="{py}"; . "$0"; {script}', helpers_path],
                       capture_output=True, text=True)
    return r.returncode, r.stdout.strip()
''', '''def posix_bash():
    """Absolute path of a REAL POSIX bash (same resolver as test_launch_gate.posix_bash, SO-A16): `shutil.which("bash")` answers whatever is first on PATH, which on a Windows box can be System32's WSL
    launcher (it cannot read a Windows path); a bare name given to CreateProcess goes there BEFORE PATH.  None = no usable bash."""
    if os.name != "nt":
        return shutil.which("bash")

    def wsl_launcher(q):
        q = os.path.normcase(os.path.abspath(q))
        sysroot = os.path.normcase(os.environ.get("SystemRoot", "C:" + os.sep + "Windows"))
        return q.startswith(sysroot + os.sep) or (os.sep + "windowsapps" + os.sep) in q

    cands = [os.path.join(d, "bash.exe") for d in os.environ.get("PATH", "").split(os.pathsep) if d]
    git = shutil.which("git")
    roots = [os.path.dirname(os.path.dirname(git))] if git else []
    roots += [os.path.join(os.environ[v], "Git") for v in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)") if os.environ.get(v)]
    for r in roots:
        cands += [os.path.join(r, "usr", "bin", "bash.exe"), os.path.join(r, "bin", "bash.exe")]
    for c in cands:
        if os.path.isfile(c) and not wsl_launcher(c):
            return os.path.abspath(c)
    return None


BASH = posix_bash()


def _sh(helpers_path, script):
    py = sys.executable.replace("\\\\", "/")
    r = subprocess.run([BASH, "-c", f'PY_C="{py}"; . "$0"; {script}', helpers_path],
                       capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def _chain_precheck(chain_path, tmp, tag, token_env):
    """Run the chain's precheck against an empty pod root with a stub flock.  token_env: None = SFT4R_HF_TOKEN_FILE unset, else its value.  -> (rc, chain.log text)."""
    q = lambda x: str(x).replace("\\\\", "/")  # noqa: E731
    ws, stub = os.path.join(tmp, "ws_" + tag), os.path.join(tmp, "stub_" + tag)
    os.makedirs(ws)
    os.makedirs(stub)
    open(os.path.join(stub, "flock"), "wb").write(b"#!/bin/bash" + bytes([10]) + b"exit 0" + bytes([10]))          # LF bytes, no escapes
    assert subprocess.run([BASH, "-c", f"chmod +x '{q(stub)}/flock'"]).returncode == 0
    env = {k: v for k, v in os.environ.items() if k not in ("SFT4R_HF_TOKEN_FILE", "HF_TOKEN", "SFT4R_STOP_AFTER")}
    env["PATH"] = stub + os.pathsep + env.get("PATH", "")
    env["SFT4R_WS"] = q(ws)
    if token_env is not None:
        env["SFT4R_HF_TOKEN_FILE"] = token_env
    r = subprocess.run([BASH, q(chain_path)], capture_output=True, text=True, env=env, timeout=120)
    log = os.path.join(ws, "data", "refe_sft4r", "chain.log")
    return r.returncode, (open(log, encoding="utf-8").read() if os.path.isfile(log) else "")
''')

sub('''    print(f"ZZTEST_PASS {n}")
    return 0
''', '''    # ---- S4R-1: the HF token FILE path is a REQUIRED env var.  The repo is public and its lander refuses any file that names the pod's token-file path, so the chain carries no default (3 files were withheld
    # from the banked readiness package for naming it).  Unset or empty => `REFUSED: stage=precheck` FIRST, before the manifest and before anything on the pod is read.
    assert BASH is not None, "no POSIX bash (only the WSL launcher, or none): the chain's precheck cannot be run"; n += 1
    assert (".hf_" + "token") not in text; n += 1                                                 # the chain's SOURCE names no token-file path (the literal is assembled here: this test names none either)
    assert "HFTOK=${SFT4R_HF_TOKEN_FILE:-}" in text and "SFT4R_HF_TOKEN_FILE:-/" not in text; n += 1
    NOTSET = "REFUSED: stage=precheck: SFT4R_HF_TOKEN_FILE is not set"
    with tempfile.TemporaryDirectory() as tmp:
        rc, log = _chain_precheck(CHAIN, tmp, "unset", None)
        assert rc == 1 and NOTSET in log and "chain armed" not in log, (rc, log[-300:]); n += 1   # fail closed, and FIRST (before 'chain armed', the manifest, the lock's work)
        assert not os.path.isdir(os.path.join(tmp, "ws_unset", "data", "refe_sft4r", "stage")) or not os.listdir(os.path.join(tmp, "ws_unset", "data", "refe_sft4r", "stage")); n += 1   # no stage marker
        rc, log = _chain_precheck(CHAIN, tmp, "empty", "")
        assert rc == 1 and NOTSET in log, (rc, log[-300:]); n += 1                                  # empty is unset
        rc, log = _chain_precheck(CHAIN, tmp, "set", "/nonexistent/token_file_for_the_test")
        assert rc == 1 and NOTSET not in log and "chain armed" in log and "REFUSED: stage=precheck: no " in log and "SHIP_MANIFEST.json" in log, (rc, log[-300:]); n += 1   # the variable was accepted; a LATER check refused
        # MUTATIONS of a temp copy: each must make the unset case stop refusing on the variable (the test can go RED)
        for tag, old, new in (("default_back", "HFTOK=${SFT4R_HF_TOKEN_FILE:-}", "HFTOK=${SFT4R_HF_TOKEN_FILE:-/some/default/path}"),
                              ("check_removed", '[ -n "$HFTOK" ] || refuse "SFT4R_HF_TOKEN_FILE is not set', '[ -n "$HFTOK" ] || true "SFT4R_HF_TOKEN_FILE is not set')):
            assert text.count(old) == 1, (tag, text.count(old)); n += 1
            mut = os.path.join(tmp, f"sft4r_chain.{tag}.sh")
            open(mut, "wb").write(text.replace(old, new).encode("utf-8"))
            rc, log = _chain_precheck(mut, tmp, tag, None)
            assert NOTSET not in log, (tag, "the test would not notice this regression", log[-300:]); n += 1
    print(f"ZZTEST_PASS {n}")
    return 0
''')
open(p, "w", encoding="utf-8", newline="").write(s)
print("patched", p.name)
