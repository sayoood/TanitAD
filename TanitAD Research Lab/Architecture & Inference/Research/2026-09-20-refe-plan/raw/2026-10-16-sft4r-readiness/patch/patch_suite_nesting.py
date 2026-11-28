#!/usr/bin/env python3
"""One-shot patcher: G-SUITE REFUSES TO NEST (SFT-4R readiness, 2026-10-16).  usage: patch_suite_nesting.py <refe dir>

Root cause it closes (raw/2026-10-16-sft4r-readiness/READINESS.md, "The recursion"):
  gate_checks.check_suite SET `REFE_GATE_NESTED=1` on every suite script it ran and NOTHING READ IT.  refe_sft4r has no arm table, so `launch_gate.py run --profile refe_sft4r` (unlike
  refe_rls1 / refe_aux_s1*, which refuse an unnamed arm BEFORE any check) reaches G-SUITE; test_sft4r_gate.py is itself a G-SUITE entry AND called that command (line 286 of the P0
  version) => the suite ran the test that ran the suite ... ~80 python processes, RAM 8.6 GB.

Fix: the sentinel now carries the JSON list of suite entries running ABOVE the process (outermost first).  check_suite spawns NOTHING and returns FAIL when
  (a) an entry it is about to run is already on that stack (the suite contains the script that invoked it), or
  (b) the stack is already SUITE_NEST_MAX (2) deep (a suite script may run ONE nested, different suite -- a stub or synthetic one; a script inside that may run none).
The old value "1" (still exported by raw/*/run_suite_scripts.sh and the pod timing scripts to reproduce the gate's environment) reads as depth 1; an unreadable value fails closed.
gate_token.SUITE_ENV_IGNORE already ignores REFE_GATE_NESTED, so the environment binding of suite evidence is unchanged.
Also adds refe/test_suite_nesting.py to the refe_sft4r suite (launch_gate.py), after patch_launch_gate.py has created SFT4R_SUITE.
"""
import sys
from pathlib import Path

refe = Path(sys.argv[1])

# ------------------------------------------------------------------------------------------------------------------------------------- gate_checks.py
p = refe / "gate_checks.py"
s = open(p, encoding="utf-8", newline="").read()
assert "\r\n" not in s

old = '''# ============================================================================================ G-SUITE, G-CLOCK
def check_suite(profile: dict, arm=None, rehearsal=False) -> dict:
    reasons, det = [], {"scripts": []}
    entries = [dict(e) for e in profile.get("suite", ())]
    if arm_is("suite_failing_test", arm):
        entries.append({"raw": [sys.executable, "-c", "import sys; print('ZZTEST_FAIL'); sys.exit(1)"], "token": "ZZTEST_PASS"})
    for e in entries:
        argv = e["raw"] if "raw" in e else [sys.executable, str(HERE / e["script"]), *e.get("args", [])]
        token = e["token"]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", REFE_GATE_NESTED="1")
'''
assert s.count(old) == 1
new = '''# ============================================================================================ G-SUITE, G-CLOCK
# G-SUITE REFUSES TO NEST (2026-10-16).  `REFE_GATE_NESTED` used to be set to "1" and read by nobody; a suite script that ran a command reaching G-SUITE for a profile whose suite CONTAINS that script
# re-ran itself without end (SFT-4R: test_sft4r_gate.py -> `launch_gate.py run --profile refe_sft4r` -> G-SUITE -> test_sft4r_gate.py ..., ~80 python processes).  The sentinel is now the JSON list of the
# suite entries running ABOVE this process, outermost first; check_suite spawns nothing when an entry is already on it, or when it is already SUITE_NEST_MAX deep.  Tests: refe/test_suite_nesting.py.
SUITE_NEST_ENV = "REFE_GATE_NESTED"
SUITE_NEST_MAX = 2                       # a suite script may run ONE nested suite (a stub / synthetic one); a script inside THAT one may run none


def suite_entry_id(e: dict) -> str:
    """Identity of a G-SUITE entry for the nesting guard: sha256[:16] of the resolved script path + its args (or of a raw argv), then the script's basename."""
    if "raw" in e:
        argv, base = [str(x) for x in e["raw"]], "raw"
    else:
        argv, base = [os.path.normcase(os.path.realpath(str(HERE / e["script"]))), *[str(x) for x in e.get("args", ())]], os.path.basename(str(e["script"]))
    return hashlib.sha256("\\0".join(argv).encode("utf-8", "replace")).hexdigest()[:16] + ":" + base


def suite_nest_stack() -> list:
    """The suite entries running above this process (outermost first), from the sentinel.  "" = none.  The pre-2026-10-16 value "1" = one level of unknown identity; anything unreadable
    FAILS CLOSED (it reads as already at the limit), so a damaged sentinel can only refuse, never allow."""
    raw = os.environ.get(SUITE_NEST_ENV, "")
    if not raw:
        return []
    try:
        v = json.loads(raw)
    except ValueError:
        return ["<unreadable>"] * SUITE_NEST_MAX
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return v
    if isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= SUITE_NEST_MAX:
        return ["<legacy>"] * v
    return ["<unreadable>"] * SUITE_NEST_MAX


def suite_nest_refusals(stack: list, entries: list) -> list:
    """Why check_suite must spawn NOTHING (empty list = it may)."""
    out = []
    if len(stack) >= SUITE_NEST_MAX:
        out.append(f"G-SUITE refuses to nest: this process is already {len(stack)} suite level(s) deep (limit {SUITE_NEST_MAX}); running above it: {stack}")
    for e in entries:
        if suite_entry_id(e) in stack:
            out.append(f"G-SUITE refuses to nest: entry {e.get('script') or 'raw'} is ALREADY RUNNING above this call, so this suite contains the script that invoked it and would never end "
                       f"(run the command with --only <checks without G-SUITE>, or give the test a stub suite)")
    return out


def check_suite(profile: dict, arm=None, rehearsal=False) -> dict:
    reasons, det = [], {"scripts": []}
    entries = [dict(e) for e in profile.get("suite", ())]
    if arm_is("suite_failing_test", arm):
        entries.append({"raw": [sys.executable, "-c", "import sys; print('ZZTEST_FAIL'); sys.exit(1)"], "token": "ZZTEST_PASS"})
    stack = suite_nest_stack()
    refused = suite_nest_refusals(stack, entries)
    if refused:                                                                      # NOTHING is spawned: a suite that contains its own caller cannot end
        det["nested"] = {"stack": stack, "refused": refused}
        return verdict("G-SUITE", refused, det, arm=arm, rehearsal=rehearsal)
    for e in entries:
        argv = e["raw"] if "raw" in e else [sys.executable, str(HERE / e["script"]), *e.get("args", [])]
        token = e["token"]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", **{SUITE_NEST_ENV: json.dumps(stack + [suite_entry_id(e)])})
'''
s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="").write(s)
print("patched", p.name)

# ------------------------------------------------------------------------------------------------------------------------------------- launch_gate.py
p = refe / "launch_gate.py"
s = open(p, encoding="utf-8", newline="").read()
assert "\r\n" not in s
old = '''SFT4R_SUITE = [{"script": "test_sft4r_chain.py", "token": "ZZTEST_PASS"}, {"script": "test_sft4r_gate.py", "token": "ZZTEST_PASS"},
               {"script": "test_launch_gate.py", "args": ["--static", "--no-banked"], "token": "ZZLG_STATIC_PASS"}]
'''
assert s.count(old) == 1, "run patch_launch_gate.py first (it creates SFT4R_SUITE)"
new = '''SFT4R_SUITE = [{"script": "test_sft4r_chain.py", "token": "ZZTEST_PASS"}, {"script": "test_sft4r_gate.py", "token": "ZZTEST_PASS"},
               {"script": "test_suite_nesting.py", "token": "ZZTEST_PASS"},                                   # G-SUITE refuses to nest (gate_checks.check_suite); mutation-proven, process-bounded
               {"script": "test_launch_gate.py", "args": ["--static", "--no-banked"], "token": "ZZLG_STATIC_PASS"}]
'''
s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="").write(s)
print("patched", p.name)
