#!/usr/bin/env python3
"""THE SEALED SHELL RUNNER: every Git-bash step of the LAUNCH-INTENT self-test (and of make_dry_run.py) goes through ``run`` / ``popen`` here, and nowhere else.

WHY (the incident of 2026-10-28 05:56, HANDOFF.md): the previous harness put a stub ``ssh`` / ``scp`` first on the PATH it handed to ``bash -c``, but Git bash PREPENDS /usr/bin to PATH at start-up, so the REAL ``ssh`` ran and
the chain's refresh step replaced the trainer, the model and the queue of the box-2 run that was in flight.  The fix is a GUARD that is part of the same shell as the step it protects.  Before the step's text is even parsed, that shell

  1. exports the world's stub directory FIRST on PATH (inside the shell, after /usr/bin was prepended);
  2. asserts ``command -v ssh`` and ``command -v scp`` ARE the world's stubs;
  3. asserts ``$HOME`` is the empty scratch directory and that ``/usr/bin/ssh -G tanitad-box2`` (a config PRINT: it opens no connection) leaves the alias UNRESOLVED (``hostname tanitad-box2``);
  4. asserts every variable the steps WRITE through (OAF_SHIP_DIR, LAUNCHINTENT_ART, TMPDIR, LI_LOG_DIR, LI_STUB_LOG, OAF_MIRROR, V8F_CHAIN_OUT, V8F_CURRENT) is set when the step needs it and resolves INSIDE the scratch world,
     and never inside C:/Users/Admin/box2_ship (the installed ship directory the chain's helpers hardcode);

and any failure is ``exit 97`` with ``ZZHARNESS-REFUSED`` having run NOTHING.  A step with no world of its own (a plain ``bash -n``) gets the GLOBAL stub world, so there is no unguarded path.
``regression_arms`` / ``MUTANT_PREFIXES`` below are the guard's own deliberate-regression tests: a defective harness (the old PATH-only behaviour, a missing scp stub, an alias that resolves, a write path in the real ship
directory, ...) must be REFUSED, and a guard with one check removed must LET THE DEFECT THROUGH (that is how the guard is shown to be load-bearing).

Nothing here names an address of box 2; the only mention of its ssh alias is the guard's own ``ssh -G`` print.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import common as C

BASHX = C.BASH
SCR_ROOT = C.SCRATCH                                                  # F:/TanitAD-scratch/launch_intent_chain: the ONLY place a sealed step may write
REAL_SHIP = "C:/Users/Admin/box2_ship"                                # the INSTALLED ship directory (mk_v8f_job.py, waitjob.sh and the installed guard hardcode it): NEVER written by a test
NOHOME = SCR_ROOT / "home_nobox2"                                     # an EMPTY $HOME: no ~/.ssh/config, so the alias cannot resolve even if a stub were bypassed
GSTUB = SCR_ROOT / "global_stub"                                      # the stub world of a step that has none of its own
WRITE_VARS = ("OAF_SHIP_DIR", "LAUNCHINTENT_ART", "TMPDIR", "LI_LOG_DIR", "LI_STUB_LOG", "OAF_MIRROR", "V8F_CHAIN_OUT", "V8F_CURRENT")
STATS = {"sealed_runs": 0, "refused": 0}


def px(p) -> str:
    s = str(p).replace("\\", "/")
    return "/" + s[0].lower() + s[2:] if len(s) > 1 and s[1] == ":" else s


def stub_src(name: str) -> bytes:
    return ('#!/usr/bin/env bash\necho "' + name + ' BUSY=$([ -f "$LI_BUSY_FILE" ] && echo yes || echo no) $*" >> "$LI_STUB_LOG"\nexit 255\n').encode()


def make_stub_dir(d: Path, names=("ssh", "scp")) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(stub_src(n))
        os.chmod(d / n, 0o755)
    return d


def ensure_global() -> None:
    NOHOME.mkdir(parents=True, exist_ok=True)
    assert not any(NOHOME.iterdir()), f"{NOHOME} must stay EMPTY (it is the sealed shells' $HOME)"
    make_stub_dir(GSTUB)
    (SCR_ROOT / "tmp").mkdir(parents=True, exist_ok=True)


def prefix(stub, home, need=(), drop=()) -> str:
    """The guard text.  ``drop`` names checks to REMOVE (the guard's own mutants and the regression arms that isolate one check): path_export, ssh_stub, scp_stub, home (HOME is the scratch HOME), home_empty (it is empty), alias, need, writepaths."""
    stub, home = px(stub), px(home)
    scr = px(SCR_ROOT).lower()
    scr_m = str(SCR_ROOT).replace("\\", "/").lower()
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
        L.append('[ -z "$(ls -A "$HOME" 2>/dev/null)" ] || refuse "HOME is not an EMPTY directory (no ~/.ssh/config may exist)"')
    if "alias" not in drop:
        L.append('/usr/bin/ssh -G tanitad-box2 2>/dev/null | grep -qx "hostname tanitad-box2" || refuse "the box-2 ssh alias RESOLVES (a config would send ssh to a host)"')
    if "need" not in drop:
        for v in need:
            L.append(f'[ -n "${{{v}:-}}" ] || refuse "{v} is not set (this step writes through it)"')
    if "writepaths" not in drop:
        resolve = 'r=$(realpath -m "$1" 2>/dev/null) && r=$(cygpath -m "$r" 2>/dev/null | tr "A-Z" "a-z") || return 1; '      # collapses `..` (realpath -m), then makes it a lower-case Windows path (cygpath -m also collapses `..`)
        if "normalise" in drop:                                                                     # GM11: the path TEXT as given is prefix-tested, so scratch/../../x passes
            resolve = 'r=$(echo "$1" | tr "A-Z" "a-z"); '
        L.append('inw() { local r; ' + resolve +
                 f'case "$r" in "{real}"|"{real}"/*) return 1 ;; "{scr_m}"/*|"{scr}"/*) return 0 ;; esac; return 1; }}')
        L.append(f'for v in {" ".join(WRITE_VARS)}; do eval "p=\\${{$v:-}}"; [ -z "$p" ] || inw "$p" || refuse "$v=$p is not inside the scratch world {scr_m} (or is inside {real})"; done')
    return "\n".join(L) + "\n"


def _env(env, stub, calls) -> dict:
    e = dict(os.environ)
    e.update({"CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1", "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "safe.directory", "GIT_CONFIG_VALUE_0": "*", "OMP_NUM_THREADS": "1",
              "HOME": px(NOHOME), "TMPDIR": px(SCR_ROOT / "tmp"), "LI_STUB_LOG": px(calls), "LI_BUSY_FILE": px(SCR_ROOT / "global_stub" / "no_busy")})
    e.update(env or {})
    for k in WRITE_VARS:                                                # realpath -m needs MSYS-style paths: a Windows-style value would be read as a relative path
        if k in e and len(e[k]) > 1 and e[k][1] == ":":
            e[k] = px(e[k])
    e["PATH"] = str(stub) + os.pathsep + e.get("PATH", os.environ["PATH"])      # NOT trusted by the guard: Git bash re-prepends /usr/bin. Kept so the old assumption stays visible to the regression arms
    return e


def _world(env):
    ensure_global()
    env = dict(env or {})
    stub = Path(env.pop("LI_STUB_DIR", "") or GSTUB)
    calls = Path(env.get("LI_STUB_LOG") or (GSTUB / "calls.log"))
    return env, stub, calls


def run(cmd: str, env=None, timeout=300, need=(), cwd=None, drop=(), home=None):
    """Run ``cmd`` in Git bash behind the guard.  Returns (rc, stdout+stderr).  rc 97 = the guard refused (nothing ran)."""
    env, stub, calls = _world(env)
    e = _env(env, stub, calls)
    full = prefix(stub, home or NOHOME, need, drop) + cmd
    STATS["sealed_runs"] += 1
    p = subprocess.Popen([BASHX, "-c", full], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=e, cwd=cwd, stdin=subprocess.DEVNULL)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:                                  # a hung step (e.g. a mutant that waits forever): kill the WHOLE tree, report rc 124 like coreutils `timeout`
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True, stdin=subprocess.DEVNULL)
        try:
            out, err = p.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
        return 124, (out + err).decode("utf-8", "replace").replace("\r", "") + "\nZZHARNESS-TIMEOUT: the step was killed after %s s" % timeout
    if p.returncode == 97:
        STATS["refused"] += 1
    return p.returncode, (out + err).decode("utf-8", "replace").replace("\r", "")


def popen(cmd: str, args=(), env=None, need=()):
    env, stub, calls = _world(env)
    e = _env(env, stub, calls)
    STATS["sealed_runs"] += 1
    return subprocess.Popen([BASHX, "-c", prefix(stub, NOHOME, need) + cmd, "x", *map(str, args)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=e, stdin=subprocess.DEVNULL)


def real_ship_snapshot() -> dict:
    """{relative name: (size, mtime_ns)} of the installed ship dir and its logs (read-only listing): the tripwire compares it before and after the self-test."""
    out = {}
    base = Path(REAL_SHIP)
    for d in (base, base / "logs"):
        if d.is_dir():
            for f in d.iterdir():
                if f.is_file():
                    st = f.stat()
                    out[str(f.relative_to(base)).replace("\\", "/")] = (st.st_size, st.st_mtime_ns)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------- the guard's deliberate-regression arms
def _arm_env(scr: Path, **over):
    """A GOOD world env (a stub world + write paths inside the scratch world) with ONE defect applied by the arm."""
    w = scr
    ship, art, tmp = w / "ship", w / "art", w / "tmp"
    for d in (ship, art, tmp):
        d.mkdir(parents=True, exist_ok=True)
    stub = make_stub_dir(w / "stub")
    e = {"LI_STUB_DIR": str(stub), "OAF_SHIP_DIR": px(ship), "LAUNCHINTENT_ART": px(art), "TMPDIR": px(tmp), "LI_STUB_LOG": px(w / "calls.log"), "LI_BUSY_FILE": px(w / "no_busy")}
    e.update(over)
    return e


def regression_arms(root: Path):
    """[(name, kwargs for run(), expect)] -- each a harness variant that LACKS one protection; expect 'refused' = rc 97 with ZZHARNESS-REFUSED, the witness file absent and no stub call logged.  The first is the POSITIVE control."""
    arms = []
    arms.append(("good world (positive control)", dict(env=_arm_env(root / "a_good"), need=("OAF_SHIP_DIR", "LAUNCHINTENT_ART")), "ran"))
    # 1. the OLD harness: the stub only in the env PATH (Git bash re-prepends /usr/bin) -> `command -v ssh` is /usr/bin/ssh. The variant is "the guard without its export line".
    arms.append(("old harness: stub only on the env PATH, not exported inside the shell", dict(env=_arm_env(root / "a_oldpath"), drop=("path_export",)), "refused"))
    # 2. a world with an ssh stub but no scp stub, and the reverse
    e2 = _arm_env(root / "a_noscp")
    (Path(e2["LI_STUB_DIR"]) / "scp").unlink()
    arms.append(("world without an scp stub", dict(env=e2), "refused"))
    e3 = _arm_env(root / "a_nossh")
    (Path(e3["LI_STUB_DIR"]) / "ssh").unlink()
    arms.append(("world without an ssh stub", dict(env=e3), "refused"))
    # 4. the alias RESOLVES: a $HOME that holds an ssh config naming the alias (a TEST-NET address; `ssh -G` only PRINTS it). The HOME-emptiness check is relaxed for THIS arm so the alias check is the one under test.
    hd = root / "a_alias_home"
    (hd / ".ssh").mkdir(parents=True, exist_ok=True)
    (hd / ".ssh" / "config").write_bytes(b"Host tanitad-box2\n  HostName 203.0.113.9\n")
    arms.append(("a $HOME whose ssh config resolves the box-2 alias (HOME-emptiness relaxed to isolate the alias check)", dict(env=_arm_env(root / "a_alias", HOME=px(hd)), home=hd, drop=("home_empty",)), "refused"))
    # 5. $HOME is not the scratch HOME (an empty dir elsewhere); and a scratch HOME that is not empty
    hd5 = root / "a_home_other"
    hd5.mkdir(parents=True, exist_ok=True)
    arms.append(("$HOME is not the scratch HOME", dict(env=_arm_env(root / "a_home", HOME=px(hd5))), "refused"))
    hd6 = root / "a_home_full"
    hd6.mkdir(parents=True, exist_ok=True)
    (hd6 / "stray.txt").write_bytes(b"x")
    arms.append(("the scratch $HOME is not empty", dict(env=_arm_env(root / "a_homefull", HOME=px(hd6)), home=hd6), "refused"))
    # 6. a write path in the INSTALLED ship dir (the hardcoded one)
    arms.append(("OAF_SHIP_DIR is the installed C:/Users/Admin/box2_ship", dict(env=_arm_env(root / "a_realship", OAF_SHIP_DIR=REAL_SHIP)), "refused"))
    arms.append(("LAUNCHINTENT_ART inside the installed ship dir's logs", dict(env=_arm_env(root / "a_reallogs", LAUNCHINTENT_ART=REAL_SHIP + "/logs/x")), "refused"))
    # 7. a write path outside the scratch world
    arms.append(("TMPDIR outside the scratch world", dict(env=_arm_env(root / "a_outside", TMPDIR="/c/Windows/Temp/li_outside")), "refused"))
    dd = root / "a_dotdot" / "ship"
    try:
        depth = len(dd.resolve().relative_to(SCR_ROOT.resolve()).parts) + 1             # enough `..` to leave the scratch world, wherever `root` sits under it
    except ValueError:
        depth = 12
    arms.append(("a write path that climbs out of the scratch world with ..", dict(env=_arm_env(root / "a_dotdot", OAF_SHIP_DIR=px(dd) + "/.." * depth + "/escaped_x")), "refused"))
    # 8. a step that needs OAF_SHIP_DIR but the world did not set it (the chain would fall back to its hardcoded default)
    e8 = _arm_env(root / "a_unset")
    e8.pop("OAF_SHIP_DIR")
    arms.append(("OAF_SHIP_DIR unset for a step that needs it (the chain would use C:/Users/Admin/box2_ship)", dict(env=e8, need=("OAF_SHIP_DIR",)), "refused"))
    return arms


# a guard with ONE check removed must change the behaviour of the arm that targets it (a defect gets through, or the positive control is refused): that is how each check is shown to be load-bearing.  (mutant name, drop, the arm it must flip)
MUTANT_PREFIXES = [
    ("GM1 the guard no longer exports the stub dir first on PATH (every step is refused)", ("path_export",), "good world (positive control)"),
    ("GM2 the guard no longer checks that ssh AND scp are the stubs (the old PATH-only harness runs)", ("ssh_stub", "scp_stub"), "old harness: stub only on the env PATH, not exported inside the shell"),
    ("GM3 the guard no longer checks that ssh is the stub", ("ssh_stub",), "world without an ssh stub"),
    ("GM4 the guard no longer checks that scp is the stub", ("scp_stub",), "world without an scp stub"),
    ("GM5 the guard no longer checks that $HOME is the scratch HOME", ("home",), "$HOME is not the scratch HOME"),
    ("GM6 the guard no longer checks that the scratch $HOME is empty", ("home_empty",), "the scratch $HOME is not empty"),
    ("GM7 the guard no longer prints the alias with ssh -G", ("alias",), "a $HOME whose ssh config resolves the box-2 alias (HOME-emptiness relaxed to isolate the alias check)"),
    ("GM8 the guard no longer resolves the write paths (the installed ship dir is accepted)", ("writepaths",), "OAF_SHIP_DIR is the installed C:/Users/Admin/box2_ship"),
    ("GM9 the guard no longer resolves the write paths (a path outside the scratch world is accepted)", ("writepaths",), "TMPDIR outside the scratch world"),
    ("GM11 the guard no longer collapses `..` before testing a write path (a path that climbs out of the scratch world passes)", ("normalise",), "a write path that climbs out of the scratch world with .."),
    ("GM10 the guard no longer requires the step's variables to be set", ("need",), "OAF_SHIP_DIR unset for a step that needs it (the chain would use C:/Users/Admin/box2_ship)"),
]


def exercise_arm(arm, drop=()):
    """Run one arm with the guard minus ``drop``; returns (behaviour, detail): 'refused' (rc 97, witness absent, no stub call) or 'ran' (rc 0, witness present) or 'other'."""
    name, kw, _expect = arm
    kw = dict(kw)
    env = kw["env"]
    root = Path(env["LI_STUB_DIR"]).parent
    witness = root / "WITNESS"
    if witness.exists():
        witness.unlink()
    calls = root / "calls.log"
    if calls.exists():
        calls.unlink()
    cmd = f'touch "{px(witness)}"; echo STEP-RAN'
    kw["drop"] = tuple(kw.get("drop", ())) + tuple(drop)
    rc, out = run(cmd, **kw)
    ncalls = len([x for x in calls.read_text(errors="replace").split("\n") if x.strip()]) if calls.exists() else 0
    if rc == 97 and "ZZHARNESS-REFUSED" in out and not witness.exists() and "STEP-RAN" not in out and ncalls == 0:
        return "refused", out.strip()[:200]
    if rc == 0 and witness.exists() and "STEP-RAN" in out:
        return "ran", out.strip()[:200]
    return "other", f"rc={rc} witness={witness.exists()} calls={ncalls} {out.strip()[:200]}"


def incident_mechanism() -> tuple:
    """Reproduce the 2026-10-28 mechanism WITHOUT contacting anything: with a stub dir first on the ENV path and no export inside the shell, Git bash resolves `ssh` to /usr/bin/ssh.  Only `command -v` runs.
    Returns (resolved path, the stub dir): the old harness is unsafe iff they differ."""
    ensure_global()
    stub = make_stub_dir(SCR_ROOT / "incident_stub")
    e = dict(os.environ)
    e["PATH"] = str(stub) + os.pathsep + e["PATH"]
    e["HOME"] = px(NOHOME)
    r = subprocess.run([BASHX, "-c", "command -v ssh"], capture_output=True, env=e, stdin=subprocess.DEVNULL)
    return r.stdout.decode().strip(), px(stub)
