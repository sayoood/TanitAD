#!/usr/bin/env python3
"""One-shot patcher (S4R-1, 2026-10-17): test_sft4r_gate.py (a) is HERMETIC about the ambient G-SUITE nesting depth, (b) runs the pod script under a REAL POSIX bash.  usage: ... <refe dir>

(a) ROOT CAUSE of "test_sft4r_gate.py exits rc 1 INSIDE the refe_sft4r G-SUITE but passes standalone" (readiness table, row e2e_suite_prerun_refe_sft4r; REPRODUCED 2026-10-17, MEASURED, see S4R_1_RESULT.md):
    section 6 runs `L.run_checks(<profile with a STUB suite>)` IN-PROCESS.  That reaches gate_checks.check_suite, which reads the ambient sentinel REFE_GATE_NESTED (the stack of suite entries above this
    process) and, by design (SUITE_NEST_MAX = 2), lets a suite script run exactly ONE nested suite.  When the ambient stack was already 2 deep -- the readiness harness's "gate" form put a depth-1 stack on
    the `suite-prerun` job, which is not a suite entry, so every suite child ran at depth 2 -- the stub suite was REFUSED ("G-SUITE refuses to nest: this process is already 2 suite level(s) deep"),
    G-SUITE read FAIL where the test requires PASS, and the file exited rc 1 after 132 checks in ~3 s.  Standalone (no sentinel) it never saw a depth.  Not the nesting guard misfiring: the guard did its job;
    the TEST was not independent of where it was started from.  Section 6's subject is run -> evidence -> token -> verify on a miniature of the pod, not the guard (section 8 and test_suite_nesting.py own that).
    Fix: section 6 pops the ambient sentinel for its in-process stub-suite runs and restores it before section 7 (whose subprocess calls reach no G-SUITE and keep the real ambient stack).
(b) `subprocess.run(["bash", ...])` is a BARE bash: on a Windows box CreateProcess resolves it to System32's WSL launcher BEFORE PATH (since the WSL distro was registered on this box, 2026-10-17 10:04),
    which cannot read a Windows path (rc 127).  Same class as SO-A16 (test_launch_gate.py:339).  Resolved to an absolute Git/POSIX bash, skipping the WSL launcher; a box with none FAILS LOUD here (the check
    is a registry cross-check the profile cannot do without the pod script's output, so it is not skippable)."""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "test_sft4r_gate.py"
s = open(p, encoding="utf-8", newline="").read()
assert "\r\n" not in s


def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)


# ---- (b) the helper + the call
sub('''def main() -> int:
    import scorer_finetune as S
''', '''def posix_bash():
    """Absolute path of a REAL POSIX bash (same resolver as test_launch_gate.posix_bash, SO-A16).  A bare `bash` given to CreateProcess goes to System32's WSL launcher BEFORE PATH, which cannot read a
    Windows path.  Skips the WSL launcher wherever it sits on PATH; falls back to Git for Windows' own bash; POSIX: the first `bash` on PATH.  None = no usable bash (the caller fails loud)."""
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


def main() -> int:
    import scorer_finetune as S
''')
sub('''    r = subprocess.run(["bash", str(HERE / "sft4r_pod.sh")], capture_output=True, text=True, env=dict(os.environ, MODE="argv", SFT4R_WS="/workspace"))
''', '''    bash = posix_bash()
    ok(bash is not None, "no POSIX bash on this box (only the WSL launcher, or none): the pod script's argv cannot be cross-checked against the profile")
    r = subprocess.run([bash, str(HERE / "sft4r_pod.sh")], capture_output=True, text=True, env=dict(os.environ, MODE="argv", SFT4R_WS="/workspace"))
''')

# ---- (a) section 6 hermetic about the ambient nesting depth
sub('''    # ======================================================================== 6. end to end: run -> evidence -> token -> verify, on a miniature of the pod (the real checks, a stub G-SUITE)
    work = Path(tempfile.mkdtemp())
''', '''    # ======================================================================== 6. end to end: run -> evidence -> token -> verify, on a miniature of the pod (the real checks, a stub G-SUITE)
    # HERMETIC about where this file was started from: `run_checks` below runs the STUB suite IN-PROCESS, and check_suite reads the ambient REFE_GATE_NESTED (the stack of suite entries above this
    # process).  A suite script may run exactly ONE nested suite (SUITE_NEST_MAX = 2), so an ambient stack that is already 2 deep (e.g. `suite-prerun` launched from a wrapper that exported a depth-1 stack)
    # made the stub suite REFUSED: G-SUITE FAIL, rc 1 after 132 checks (S4R-1, reproduced).  Pop it here, restore it before section 7; sections 7-8 deal with the REAL ambient stack.
    ambient_nest = os.environ.pop(C.SUITE_NEST_ENV, None)
    work = Path(tempfile.mkdtemp())
''')
sub('''    # ======================================================================== 7. the CLI the chain calls
''', '''    if ambient_nest is not None:
        os.environ[C.SUITE_NEST_ENV] = ambient_nest                                                                  # sections 7-8 see the real ambient stack again
    # ======================================================================== 7. the CLI the chain calls
''')
# imports: shutil is needed by posix_bash
if "\nimport shutil\n" not in s:
    sub("\nimport subprocess\n", "\nimport shutil\nimport subprocess\n")
open(p, "w", encoding="utf-8", newline="").write(s)
print("patched", p.name)
