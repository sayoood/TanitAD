"""PHYSICAL mutation proof of the RL channel-admissibility contract.

Runs in the OFF-DRIVE MIRROR only (C:/Users/Admin/tanitad-wt). The repo on G: is
never touched: every mutation is written, run, and restored from an in-memory
backup, and the mirror is re-synced from the repo afterwards regardless.

A guard nobody has broken on purpose is decoration. Each mutation below
reintroduces one real defect and REQUIRES a red; the control requires a green,
because a check that always fails proves nothing either.
"""
import io
import os
import subprocess
import sys

WT = r"C:\Users\Admin\tanitad-wt"
PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
SUITES = [
    r"stack\tests\test_rl_forward_keys_cover_signature.py",
    r"stack\tests\test_rl_channel_guard.py",
    r"stack\tests\test_rl_refc_adapter_robust.py",
]

ENV = dict(os.environ)
ENV["PYTHONIOENCODING"] = "utf-8"
ENV["PYTHONPATH"] = rf"{WT}\stack;{WT}\taniteval"
ENV["CUDA_VISIBLE_DEVICES"] = ""


def run_suites():
    p = subprocess.run([PY, "-m", "pytest", *SUITES, "-q", "--no-header",
                        "-p", "no:cacheprovider"],
                       cwd=WT, env=ENV, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tail = [ln for ln in (p.stdout or "").splitlines() if ln.strip()][-1:]
    fails = [ln for ln in (p.stdout or "").splitlines() if ln.startswith("FAILED")]
    return p.returncode, (tail[0] if tail else "?"), fails


def run_preflight():
    code = ("from tanitad.rl.channel_guard import assert_forward_channels_complete\n"
            "try:\n"
            "    assert_forward_channels_complete()\n"
            "    print('PREFLIGHT=PASS')\n"
            "except Exception as e:\n"
            "    print('PREFLIGHT=REFUSED', type(e).__name__)\n"
            "    print(str(e).splitlines()[0][:200])\n")
    p = subprocess.run([PY, "-c", code], cwd=WT, env=ENV, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    return (p.stdout or "").strip() or (p.stderr or "").strip()[:300]


def patch(rel, old, new):
    path = os.path.join(WT, rel)
    s = io.open(path, encoding="utf-8").read()
    assert s.count(old) == 1, f"{rel}: anchor count {s.count(old)} != 1"
    io.open(path, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
    return path, s


def restore(path, s):
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)


def report(label, expect_red, rc, tail, fails, pre):
    got = "RED" if rc != 0 else "GREEN"
    want = "RED" if expect_red else "GREEN"
    ok = "OK  " if got == want else "!!! "
    print(f"{ok}{label}\n     expected {want}, got {got}  -- {tail}")
    for f in fails[:6]:
        print(f"       {f}")
    print(f"     preflight: {pre.splitlines()[0] if pre else '?'}")
    print()
    return got == want


results = []

# ---------------------------------------------------------------- CONTROL ---
rc, tail, fails = run_suites()
results.append(report("CONTROL (unmutated tree)", False, rc, tail, fails,
                      run_preflight()))

# -------------------------------------------- M1: an exclusion loses its REASON
p = os.path.join(WT, r"stack\tanitad\refs\max_speed_input.py")
s = io.open(p, encoding="utf-8").read()
start = s.index('    ChannelExclusion(\n        channel="v_max_valid"')
r0 = s.index('        reason=(', start)
r1 = s.index('        unblock=(', r0)
mutated = s[:r0] + '        reason="",\n' + s[r1:]
io.open(p, "w", encoding="utf-8", newline="\n").write(mutated)
rc, tail, fails = run_suites()
results.append(report("M1  an exclusion with NO REASON", True, rc, tail, fails,
                      run_preflight()))
restore(p, s)

# ------------------------- M2: a declared channel is MOVED INTO FORWARD_KEYS ---
p, bak = patch(
    r"stack\tanitad\rl\refc_adapter.py",
    'FORWARD_KEYS = ("nav_cmd", "v0", "lan", "nav_known", "ego_state", "withheld_speed",\n                "agent_gt")',
    'FORWARD_KEYS = ("nav_cmd", "v0", "lan", "nav_known", "ego_state", "withheld_speed",\n                "agent_gt", "gp_point")')
rc, tail, fails = run_suites()
results.append(report("M2  declared-excluded gp_point PLUMBED", True, rc, tail,
                      fails, run_preflight()))
restore(p, bak)

# ------------- M3: a NEW forward channel, plumbed nowhere and declared nowhere --
p, bak = patch(
    r"stack\tanitad\refs\refc_v3.py",
    "                v_max_valid: Tensor | None = None) -> dict:",
    "                v_max_valid: Tensor | None = None,\n"
    "                brand_new_channel: Tensor | None = None) -> dict:")
rc, tail, fails = run_suites()
results.append(report("M3  a channel in NEITHER place", True, rc, tail, fails,
                      run_preflight()))
restore(p, bak)

# ------------------------------------------------- CONTROL AGAIN (restored) ---
rc, tail, fails = run_suites()
results.append(report("CONTROL (after restore)", False, rc, tail, fails,
                      run_preflight()))

print("=" * 74)
print("MUTATION PROOF:", "ALL EXPECTATIONS MET" if all(results) else "FAILED",
      f"({sum(results)}/{len(results)})")
sys.exit(0 if all(results) else 1)
