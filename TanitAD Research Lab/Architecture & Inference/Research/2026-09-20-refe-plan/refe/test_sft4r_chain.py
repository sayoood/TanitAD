#!/usr/bin/env python3
"""Test of sft4r_chain.sh's two gate helpers, extracted VERBATIM from the chain and run under bash on SFT-4's banked
smoke logs (raw/2026-10-04-sft4-smoke/: r2_smoke2.log = preflight PASS, r2_smoke2_g3mut.log = the G3 regression, RED).

Why it exists (PREREG_SFT4R.md Amendment 2): the chain used to GREP the preflight JSON, and json.dumps writes G3's "ok"
key LAST, so the G3 grep could never match and the chain would have refused every launch. Each helper is tested in
both directions -- the reading it must accept AND the one it must reject -- so a helper that always answers the same
way fails here.

Run: python refe/test_sft4r_chain.py   -> ZZTEST_PASS <n>     (needs bash on PATH; Git Bash on the dev box)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHAIN = os.path.join(HERE, "sft4r_chain.sh")
SMOKE = os.path.join(HERE, "..", "raw", "2026-10-04-sft4-smoke")
G3 = "G3_copies_equal_model_at_step0"
G9 = "G9_pdm_targets_in_bank"


def _helpers(text):
    m = re.search(r"^last_exit\(\).*?^PYEOF$", text, re.S | re.M)
    assert m, "the helpers are no longer in sft4r_chain.sh"
    return m.group(0) + "\n}\n"


def posix_bash():
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
    py = sys.executable.replace("\\", "/")
    r = subprocess.run([BASH, "-c", f'PY_C="{py}"; . "$0"; {script}', helpers_path],
                       capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def _chain_precheck(chain_path, tmp, tag, token_env):
    """Run the chain's precheck against an empty pod root with a stub flock.  token_env: None = SFT4R_HF_TOKEN_FILE unset, else its value.  -> (rc, chain.log text)."""
    q = lambda x: str(x).replace("\\", "/")  # noqa: E731
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


def main() -> int:
    n = 0
    text = open(CHAIN, encoding="utf-8").read()
    assert "\r\n" not in text, "sft4r_chain.sh must stay LF"; n += 1
    # the chain USES the parser for both regressions, and the order-dependent greps are gone
    assert f"gate_is $OUT/mutate_g3.log {G3} ok false" in text; n += 1
    assert f"gate_is $OUT/mutate.log {G9} ok false" in text; n += 1
    assert '{\\"ok\\": false' not in text and '{\\"train_pdm\\": 0' not in text; n += 1
    ok_log, mut_log = (os.path.join(SMOKE, f) for f in ("r2_smoke2.log", "r2_smoke2_g3mut.log"))
    # the defect, reproduced: the old grep MISSES the real G3 failure (its "ok" is the last key, not the first)
    old = open(mut_log, encoding="utf-8").read()
    assert '"%s": {"ok": false' % G3 not in old and '"ok": false' in old; n += 1
    with tempfile.TemporaryDirectory() as tmp:
        h = os.path.join(tmp, "helpers.sh")
        open(h, "w", encoding="utf-8", newline="\n").write(_helpers(text))
        q = lambda p: p.replace("\\", "/")  # noqa: E731
        gate = lambda log, g, k, v: _sh(h, f'gate_is "{q(log)}" {g} {k} {v}')[0]  # noqa: E731
        # gate_is: accept the true reading, reject the false one, on both logs
        assert gate(mut_log, G3, "ok", "false") == 0; n += 1
        assert gate(mut_log, G3, "ok", "true") != 0; n += 1
        assert gate(ok_log, G3, "ok", "false") != 0; n += 1
        assert gate(ok_log, G3, "ok", "true") == 0; n += 1
        assert gate(ok_log, G9, "ok", "false") != 0; n += 1
        assert gate(ok_log, G9, "train_pdm", "0") != 0; n += 1
        assert gate(ok_log, G9, "train_pdm", "1708") == 0; n += 1
        assert gate(ok_log, "G9_absent", "ok", "false") != 0; n += 1
        none, two = os.path.join(tmp, "none.log"), os.path.join(tmp, "two.log")
        open(none, "w", encoding="utf-8").write("ZZSFT_PREFLIGHT_FAIL\n")
        open(two, "w", encoding="utf-8").write(old + old)
        assert gate(none, G3, "ok", "false") != 0; n += 1   # no preflight row
        assert gate(two, G3, "ok", "false") != 0; n += 1    # two rows: a log written across attempts
        # last_exit: the LAST marker decides, in both directions
        def last(lines, marker="ZZPDM_TRAIN_EXIT"):
            p = os.path.join(tmp, "ex.log")
            open(p, "w", encoding="utf-8", newline="\n").write("".join(x + "\n" for x in lines))
            return _sh(h, f'last_exit {marker} "{q(p)}"')[1]
        assert last(["ZZPDM_TRAIN_EXIT 0", "ZZPDM_TRAIN_EXIT 1"]) == "1"; n += 1
        assert last(["ZZPDM_TRAIN_EXIT 1", "ZZPDM_TRAIN_EXIT 0"]) == "0"; n += 1
        assert last(["ZZPDM_TRAIN_EXIT_X 0", "ZZPDM_TRAIN_EXIT 2"]) == "2"; n += 1
        assert last(["ZZLANE_TRAIN_EXIT 0 9"], "ZZLANE_TRAIN_EXIT") == "0 9"; n += 1
        assert _sh(h, f'last_exit ZZPDM_TRAIN_EXIT "{q(tmp)}/missing.log"')[1] == ""; n += 1
    # ---- S4R-1: the HF token FILE path is a REQUIRED env var.  The repo is public and its lander refuses any file that names the pod's token-file path, so the chain carries no default (3 files were withheld
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


if __name__ == "__main__":
    sys.exit(main())
