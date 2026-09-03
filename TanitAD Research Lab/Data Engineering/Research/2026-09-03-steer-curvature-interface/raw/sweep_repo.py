"""WHOLE-REPO action-channel consumer sweep, enumerated via git (not directory
recursion), so it covers files created during this session and inherently
excludes .git and .claude/worktrees.

Probe 1 of 2 for the blast-radius table; the mirror-based grep was probe 0 and
is what missed cost_surface_probe.py.
"""
import re
import subprocess
from pathlib import Path

REPO = Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD")

PATTERNS = {
    "ch1_read": re.compile(r"actions\[\s*:\s*,\s*0\s*\]|actions\[\.\.\.\s*,\s*0\s*\]"
                           r"|\"actions\"\]\[\s*:\s*,\s*0\s*\]"),
    "integrator": re.compile(r"rollout_unicycle|unicycle_paths|paths_from_controls"),
    "bridge": re.compile(r"kappa_of_steer|steer_of_kappa|STEER_WHEELBASE_M"
                         r"|as_curvature|as_command|action_units"),
}


def tracked():
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", "--", "*.py"],
                         capture_output=True).stdout.decode("utf-8")
    files = [f for f in out.split("\0") if f]
    st = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "-z"],
                        capture_output=True).stdout.decode("utf-8")
    for e in st.split("\0"):
        if len(e) > 3 and e[3:].endswith(".py"):
            files.append(e[3:])
    return sorted(set(files))


files = tracked()
print("[sweep] " + str(len(files)) + " tracked/pending .py files (worktrees excluded by git)")
hits, unread = {k: [] for k in PATTERNS}, 0
for rel in files:
    if rel.startswith(".claude/worktrees"):
        continue
    try:
        txt = (REPO / rel).read_bytes().decode("utf-8", "replace")
    except OSError:
        unread += 1
        continue
    for name, rx in PATTERNS.items():
        n = len(rx.findall(txt))
        if n:
            hits[name].append((rel, n))

for name in ("ch1_read", "integrator", "bridge"):
    print()
    print("=== " + name + " : " + str(len(hits[name])) + " files ===")
    for rel, n in sorted(hits[name]):
        print("  " + str(n).rjust(3) + "  " + rel)
print()
print("[sweep] unreadable (G: flap): " + str(unread))
