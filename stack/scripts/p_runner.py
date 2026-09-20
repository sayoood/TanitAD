#!/usr/bin/env python3
"""p_runner.py — the gated, resumable runner for `PREREG_PERCEPTION_BOX_QUALITY.md`'s P arms.

Same shape as the A7 panel, with the decisions in PURE FUNCTIONS so the ordering and the gate can
be pinned by tests and broken on purpose (`stack/scripts/mutate_p_runner.py`), instead of living
in a shell script nobody can unit-test.

⛔ THE FOUR RULES THIS FILE ENFORCES, each pinned by a test:
 1. **`P0-REPLICATE` RUNS FIRST AND IS UNSKIPPABLE.** `box3d_centre` has no measured run-to-run
    floor, and `H-ESTIM-SEED-1` binds: a separated CI answers *"would another draw of EPISODES say
    this?"*, never *"would another training run?"*. Until P0 is VALID, :func:`next_arm` returns P0
    even if a later arm has already been run.
 2. **ONE ARM ON THE CARD AT A TIME, AND IT CHAINS BEHIND A7.** The boxstat gate (GPU ≤ 2,500 MiB,
    host ≥ 8 GB) is checked before EVERY arm, and A7's panel must be COMPLETE first — a gate that
    happens to read clear between two A7 arms is not permission to start one.
 3. **A per-arm check must read VALID or the panel STOPS.** Never "carry on and see".
 4. **RESUMABLE:** VALID arms are skipped; a partial arm directory is moved aside, never deleted.

⛔ An unreadable probe is INCONCLUSIVE, never clear (`gate_ok` returns False with the reason).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

#: The arms, in `PREREG_PERCEPTION_BOX_QUALITY` order. ⛔ `ARMS[0]` is the replicate and the
#: ordering rule in :func:`next_arm` depends on it being first.
ARMS: tuple[dict, ...] = (
    {"name": "P0-REPLICATE", "kind": "replicate", "steps": 5000,
     "why": "the run-to-run floor; no arm may be called SUPPORTED by a smaller margin",
     "flags": ["--seed", "1"]},
    {"name": "P1-STEPS", "kind": "lever", "steps": 15000,
     "why": "steps at this corpus size", "flags": ["--seed", "0", "--steps", "15000"]},
    {"name": "P3-SUPERVISION", "kind": "lever", "steps": 5000,
     "why": "the target population the loss sees", "flags": ["--seed", "0"]},
    {"name": "P4-DECODE", "kind": "lever", "steps": 5000,
     "why": "the decode normalisation", "flags": ["--seed", "0"]},
    {"name": "P5-TRUNK", "kind": "lever", "steps": 5000,
     "why": "capacity: resnet34 -> resnet101", "flags": ["--seed", "0"]},
)
GPU_MAX_MIB = 2500
HOST_MIN_GB = 8


# ============================================================================ pure decisions
def arm_status(out_dir: Path, arm: str) -> str:
    """``VALID`` (its check says so) · ``PARTIAL`` (a directory with no VALID check) · ``ABSENT``."""
    d = Path(out_dir) / arm
    if not d.exists():
        return "ABSENT"
    chk = d / "p_arm_check.json"
    if not chk.exists():
        return "PARTIAL"
    try:
        return "VALID" if json.loads(chk.read_text(encoding="utf-8")).get("status") == "VALID" \
            else "INVALID"
    except Exception:                                    # noqa: BLE001 — unreadable is not valid
        return "PARTIAL"


def next_arm(out_dir: Path, arms: tuple[dict, ...] = ARMS) -> dict | None:
    """The next arm to run, or ``None`` when every arm is VALID.

    ⛔ RULE 1: while ``arms[0]`` (the replicate) is not VALID it IS the answer, whatever any later
    arm's state says. A later arm that was somehow run first does not license skipping the floor —
    without it, no lever result can be read at all.
    """
    if arm_status(out_dir, arms[0]["name"]) != "VALID":
        return arms[0]
    for a in arms[1:]:
        if arm_status(out_dir, a["name"]) != "VALID":
            return a
    return None


def gate_ok(probe: str) -> tuple[bool, str]:
    """boxstat's ``"<gpu_mib> <host_free_gb>"`` -> (clear, reason). Unreadable ⇒ INCONCLUSIVE."""
    m = re.fullmatch(r"\s*(\d+)\s+(\d+)\s*", probe or "")
    if not m:
        return False, f"INCONCLUSIVE probe {probe!r} — never read as clear"
    gpu, host = int(m.group(1)), int(m.group(2))
    if gpu > GPU_MAX_MIB:
        return False, f"GPU busy: {gpu} MiB > {GPU_MAX_MIB} (someone else's job is NOT a reason to lower the gate)"
    if host < HOST_MIN_GB:
        return False, f"host tight: {host} GB < {HOST_MIN_GB}"
    return True, f"clear: gpu {gpu} MiB, host {host} GB"


def a7_clear(a7_dir: Path, arms=("A7-IN-s0", "A7-RND-s0", "A7-IN-s1", "A7-RND-s1"),
             panel_procs: int = 0) -> tuple[bool, str]:
    """⛔ RULE 2: A7's panel must be COMPLETE before any P arm starts.

    Complete = every A7 arm's check says VALID **and** no A7 panel process is alive. A boxstat
    gate that reads clear BETWEEN two A7 arms is not permission to start one: that is the race
    this function exists to prevent.
    """
    if panel_procs > 0:
        return False, f"A7 panel still running ({panel_procs} process(es))"
    d = Path(a7_dir)
    if not d.exists():
        return False, f"A7 output dir absent: {d}"
    missing = []
    for a in arms:
        chk = d / a / "a7_arm_check.json"
        try:
            if json.loads(chk.read_text(encoding="utf-8")).get("status") != "VALID":
                missing.append(a)
        except Exception:                                # noqa: BLE001
            missing.append(a)
    if missing:
        return False, f"A7 arms not VALID yet: {missing}"
    return True, "A7 complete: 4/4 arms VALID, no panel process"


def plan(out_dir: Path, a7_dir: Path, probe: str, panel_procs: int = 0) -> dict:
    """The whole decision in one place: ``RUN`` / ``WAIT`` / ``STOP`` / ``DONE`` + the reason."""
    for a in ARMS:
        if arm_status(out_dir, a["name"]) == "INVALID":   # ⛔ RULE 3
            return {"action": "STOP", "arm": a["name"],
                    "reason": f"{a['name']} check is INVALID — the panel stops rather than "
                              f"spending the card on arms that cannot be read"}
    nxt = next_arm(out_dir)
    if nxt is None:
        return {"action": "DONE", "arm": None, "reason": "every arm VALID"}
    ok7, why7 = a7_clear(a7_dir, panel_procs=panel_procs)
    if not ok7:
        return {"action": "WAIT", "arm": nxt["name"], "reason": why7}
    okg, whyg = gate_ok(probe)
    if not okg:
        return {"action": "WAIT", "arm": nxt["name"], "reason": whyg}
    return {"action": "RUN", "arm": nxt["name"], "reason": whyg}


def move_aside(out_dir: Path, arm: str) -> Path | None:
    """⛔ RULE 4: a partial arm is MOVED, never deleted — its log is the only record of why."""
    d = Path(out_dir) / arm
    if not d.exists():
        return None
    dst = d.with_name(f"{arm}.aborted-{int(time.time())}")
    shutil.move(str(d), str(dst))
    return dst


# ============================================================================ the executor
def boxstat(py: str, box: str) -> str:
    try:
        return subprocess.run([py, box], capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as exc:                             # noqa: BLE001 — unreadable ⇒ INCONCLUSIVE
        return f"probe failed: {type(exc).__name__}"


def count_a7_procs() -> int:
    """A7 panel processes, counted by their script path — never by a pattern that could match
    this runner's own command line (the `pgrep -f` self-match trap)."""
    ps = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                         "(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "
                         "'*a7-imagenet-knockout*a7_run.sh*' }).Count"],
                        capture_output=True, text=True, timeout=120).stdout.strip()
    try:
        return int(ps)
    except ValueError:
        return 1        # unreadable ⇒ assume BUSY, never assume free


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the P panel directory")
    ap.add_argument("--a7-dir", required=True)
    ap.add_argument("--py", default="/c/Users/Admin/venvs/tanitad/Scripts/python.exe")
    ap.add_argument("--boxstat", default="/c/Users/Admin/qland/boxstat.py")
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument("--max-wait-h", type=float, default=48.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the decision and exit — never launches an arm")
    a = ap.parse_args(argv)
    out, a7 = Path(a.out), Path(a.a7_dir)
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + a.max_wait_h * 3600
    while True:
        d = plan(out, a7, boxstat(a.py, a.boxstat), count_a7_procs())
        print("ZZP-%s %s | %s ZZ" % (d["action"], d["arm"], d["reason"]), flush=True)
        if a.dry_run or d["action"] in ("DONE", "STOP"):
            return 0 if d["action"] in ("DONE", "RUN") else (3 if d["action"] == "STOP" else 0)
        if d["action"] == "RUN":
            # ⛔ The launch itself is deliberately NOT implemented here yet: the P arms' trainer
            # flags are pre-registered but not yet authorised to run, and a runner that could
            # start one by accident is worse than one that cannot. The Master Mind's word turns
            # this into a subprocess call; until then RUN is reported and the loop exits.
            print("ZZP-READY-TO-LAUNCH %s — launch not armed in this build ZZ" % d["arm"],
                  flush=True)
            return 0
        if time.time() > deadline:
            print("ZZP-TIMEOUT after %.1f h ZZ" % a.max_wait_h, flush=True)
            return 4
        time.sleep(a.poll_s)


if __name__ == "__main__":
    sys.exit(main())
