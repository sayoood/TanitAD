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
import os
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
    # ⛔ P0b EXISTS BECAUSE P0-vs-A8 IS NOT A SEED FLOOR. A8's `config.json` carries no commit,
    # no code hash and no trainer md5, and the repo's trainer already differs from the pinned
    # tree by 313 diff lines — so |P0 − A8| is `seed + an unquantified CODE DELTA`, an UPPER
    # BOUND only. Every P arm is judged against this floor, so an arm "clearing" a contaminated
    # bound might only be clearing a systematic code shift: the exact defect `H-ESTIM-SEED-1`
    # exists to prevent. |P0 − P0b|, same pinned tree, differing ONLY in seed, IS the floor.
    # (Master Mind authorisation, 2026-09-20; A8 = seed 0, P0 = 1, P0b = a THIRD value.)
    {"name": "P0B-REPLICATE", "kind": "replicate", "steps": 5000,
     "why": "the SEED floor proper: same pinned tree as P0, third seed, nothing else moved",
     "flags": ["--seed", "2"]},
    {"name": "P1-STEPS", "kind": "lever", "steps": 15000,
     "why": "steps at this corpus size", "flags": ["--seed", "0", "--steps", "15000"]},
    {"name": "P3-SUPERVISION", "kind": "lever", "steps": 5000,
     "why": "the target population the loss sees", "flags": ["--seed", "0"]},
    {"name": "P4-DECODE", "kind": "lever", "steps": 5000,
     "why": "the decode normalisation", "flags": ["--seed", "0"]},
    {"name": "P5-TRUNK", "kind": "lever", "steps": 5000,
     "why": "capacity: resnet34 -> resnet101", "flags": ["--seed", "0"]},
)
#: ⛔ RAISED 2,500 -> 4,300 MiB ON THE PI'S DIRECT AUTHORISATION (Sayed, 2026-09-20: *"raise the
#: gate to 4300 and re-arm"*), on measured evidence, NOT to make a blocked chain move.
#: WHY THE OLD VALUE WAS WRONG: it was calibrated for a HEADLESS box. This box has a desktop on
#: it, and the desktop ALONE holds 3,111-3,958 MiB (19 samples) — so `used <= 2500` could never
#: clear, and over ~9.4 h it cleared 0/19 times while rejecting a configuration that in fact runs.
#: THE MEASUREMENT (`…/raw/vram_probe/`, instrument `code/vram_probe.py`): the real arm's
#: in-process `torch.cuda.max_memory_allocated()` peak is **2,573 MiB**, identical at step 10 and
#: at step 12 after a forced eval, i.e. PLATEAUED; total card footprint ~2,939 MiB including CUDA
#: context. At the WORST observed desktop level 3,958 + 2,939 = 6,897 of 8,188 MiB, leaving
#: 1,291 MiB (15.8 %). 4,300 = arm (2,939) + ~900 margin (the desktop's own observed swing is
#: 847 MiB), expressed as a used-ceiling on an 8,188 MiB card; it would have cleared all 19 samples.
#: ⚠️ RESIDUAL RISK, ACCEPTED BY THE PI, NOT HIDDEN: 1,291 MiB of headroom across an ~11 h arm, so
#: a large desktop spike mid-run can still OOM it. That is a trade, not an oversight.
#: ⛔ DO NOT "restore" 2,500: it is not a safer number here, it is an unsatisfiable one.
GPU_MAX_MIB = 4300
HOST_MIN_GB = 8

#: ⛔ Flags that must NEVER be carried from the base arm into a replicate: they name the base
#: run's own output, its seed, or a dump the replicate does not take. Everything else is copied
#: VERBATIM, because `PREREG_PERCEPTION_BOX_QUALITY` §3.1 defines P0 as *"A8's exact flags, a
#: different --seed, same steps"* — a replicate that quietly drops a flag is not a replicate.
_REPLICATE_DROP = {"--out": 1, "--seed": 1}


def build_argv(base_argv, *, seed: int, out: str) -> list:
    """P0's command line from the BASE run's own recorded ``argv``. Pure, so it can be diffed.

    ⛔ Copied verbatim except ``--out`` and ``--seed``. Nothing is added — in particular no
    ``--eval-window-dump``, because an added flag makes the arm a different experiment and the
    arm-to-arm difference stops being a floor.
    """
    out_argv, i = [], 0
    base = list(base_argv)
    while i < len(base):
        tok = base[i]
        n = _REPLICATE_DROP.get(tok)
        if n is not None:
            i += 1 + n
            continue
        out_argv.append(tok)
        i += 1
    return ["--out", str(out), "--seed", str(int(seed))] + out_argv


def bounded_next(out_dir: Path, stop_after: str | None,
                 arms: tuple[dict, ...] = ARMS) -> tuple[dict | None, str]:
    """:func:`next_arm`, with a HARD BOUND. -> ``(arm_or_None, reason)``.

    ⛔ THE AUTHORISATION BOUNDARY. The Master Mind authorised **P0-REPLICATE only** (2026-09-20):
    chained behind A7, on the unchanged gate, and it *"must not roll on to P1 unattended"*. This
    function is where that boundary lives, so it can be broken on purpose and seen to go RED
    (`mutate_p_runner.py`). Once ``stop_after`` is VALID there is no next arm at any gate state.
    """
    nxt = next_arm(out_dir, arms)
    if stop_after is None:
        return nxt, "unbounded"
    names = [a["name"] for a in arms]
    if stop_after not in names:
        return None, f"⛔ unknown bound {stop_after!r} — refusing to run anything"
    if arm_status(out_dir, stop_after) == "VALID":
        return None, f"BOUND REACHED: {stop_after} is VALID and the authorisation ends there"
    if nxt is None:
        return None, "every arm VALID"
    if names.index(nxt["name"]) > names.index(stop_after):
        return None, (f"⛔ next arm {nxt['name']} is BEYOND the authorised bound {stop_after} — "
                      f"refusing to launch it unattended")
    return nxt, f"within the bound ({stop_after})"


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


def plan(out_dir: Path, a7_dir: Path, probe: str, panel_procs: int = 0,
         stop_after: str | None = None) -> dict:
    """The whole decision in one place: ``RUN`` / ``WAIT`` / ``STOP`` / ``DONE`` + the reason."""
    for a in ARMS:
        if arm_status(out_dir, a["name"]) == "INVALID":   # ⛔ RULE 3
            return {"action": "STOP", "arm": a["name"],
                    "reason": f"{a['name']} check is INVALID — the panel stops rather than "
                              f"spending the card on arms that cannot be read"}
    nxt, why = bounded_next(out_dir, stop_after)
    if nxt is None:
        return {"action": "DONE", "arm": None, "reason": why}
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


#: ⛔ THE SEARCH STRING IS ASSEMBLED AT RUNTIME AND NEVER APPEARS WHOLE IN ANY COMMAND LINE.
#: This is the **`pgrep -f` self-match trap** (CLAUDE.md, "Traps preflight"), in a WMI costume.
#: MEASURED 2026-09-20, in this very function, whose previous docstring claimed to avoid exactly
#: this: a literal `'*a7-imagenet-knockout*a7_run.sh*'` in the PowerShell command **matched the
#: PowerShell process running it**, plus every shell that had ever typed the pattern — so the
#: probe read 1..4 while the A7 launcher was demonstrably still WAITING at 3,950 MiB. The runner
#: would then have waited FOREVER and P0 would never have launched. The documented fix is to make
#: the emitted token disjoint from the searched token; here the parts are concatenated inside
#: PowerShell, and the querying process excludes ITSELF by PID.
_A7_PARTS = ("a7-imagenet", "knockout", "a7_run", "refc_v3_train")


def _a7_query() -> str:
    a, b, c, t = _A7_PARTS
    return (f"$a='{a}';$b='{b}';$c='{c}';$t='{t}';"
            "$m=@(Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and "
            "$_.CommandLine -ne $null -and ("
            "($_.CommandLine -like \"*$a-$b*$c.sh*\") -or "
            "($_.CommandLine -like \"*$t.py*\" -and $_.CommandLine -like \"*$a-$b*\")"
            ") });"
            "Write-Output (\"ZQ\" + $m.Count + \"ZQ\")")


def count_a7_procs() -> int:
    """Live A7 panel processes (the panel script OR its trainer), counted pod-side-style.

    ⛔ Returns **1 on an unreadable probe** — assume BUSY, never assume free. The count is emitted
    as an opaque ``ZQ<n>ZQ`` marker and parsed from that, so the parser cannot match the words the
    query itself contains.
    """
    try:
        raw = subprocess.run(["powershell.exe", "-NoProfile", "-Command", _a7_query()],
                             capture_output=True, text=True, timeout=120).stdout
    except Exception:                                    # noqa: BLE001
        return 1
    m = re.search(r"ZQ(\d+)ZQ", raw or "")
    return int(m.group(1)) if m else 1


def flag_value(argv, flag):
    """The value after ``flag`` in an argv list, or None. Shared with `p_check_arm`."""
    for i, t in enumerate(argv):
        if t == flag:
            return argv[i + 1] if i + 1 < len(argv) else None
    return None


def md5(path) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight(arm: str, authorised: str | None, files: dict) -> tuple[bool, list]:
    """⛔ Everything that must be TRUE before the card is spent. -> ``(ok, reasons)``.

    ``files`` maps a label to ``(path, expected_md5_or_None)``. A file that cannot be read is a
    REFUSAL, never a pass: the admissible evidence that an input is right is its bytes, and an
    unreadable file is indistinguishable from a wrong one.
    """
    bad = []
    allowed = [authorised] if isinstance(authorised, str) else list(authorised or [])
    if not allowed:
        bad.append("no --authorise-arm given: this build launches nothing without an explicit "
                   "arm name")
    elif arm not in allowed:
        bad.append(f"⛔ arm {arm!r} is NOT among the authorised arms {allowed!r} — refusing")
    for label, (path, want) in files.items():
        p = Path(path)
        if not p.exists():
            bad.append(f"{label}: MISSING {path}")
            continue
        if want:
            try:
                got = md5(p)
            except OSError as exc:
                bad.append(f"{label}: UNREADABLE ({type(exc).__name__}) — INCONCLUSIVE, refusing")
                continue
            if got != want:
                bad.append(f"{label}: md5 {got} != expected {want}")
    return (not bad), bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the P panel directory")
    ap.add_argument("--a7-dir", required=True)
    ap.add_argument("--py", default="C:/Users/Admin/venvs/tanitad/Scripts/python.exe")
    ap.add_argument("--boxstat", default="C:/Users/Admin/qland/boxstat.py")
    ap.add_argument("--run-tree", default="C:/Users/Admin/tanitad-a7-run",
                    help="the PINNED tree the arm runs from — never the moving repo")
    ap.add_argument("--base-config", default=None,
                    help="the BASE run's config.json; P0 copies its argv verbatim (§3.1)")
    ap.add_argument("--trainer-md5", default=None, help="the pinned trainer's expected md5")
    ap.add_argument("--labels-md5", default=None, help="the v7/v8 label file's expected md5")
    ap.add_argument("--authorise-arm", action="append", default=None,
                    help="⛔ an arm this invocation may launch; repeat for each. Without it "
                         "nothing launches. An arm the bound allows but this list omits is "
                         "still REFUSED — the two are independent locks, on purpose.")
    ap.add_argument("--stop-after", default=None,
                    help="⛔ hard bound: once this arm is VALID the runner is DONE and will not "
                         "roll on to the next arm")
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument("--max-wait-h", type=float, default=72.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the decision and exit — never launches an arm")
    a = ap.parse_args(argv)
    out, a7, tree = Path(a.out), Path(a.a7_dir), Path(a.run_tree)
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + a.max_wait_h * 3600
    trainer = tree / "stack" / "scripts" / "refc_v3_train.py"
    attempted: set = set()
    while True:
        d = plan(out, a7, boxstat(a.py, a.boxstat), count_a7_procs(), stop_after=a.stop_after)
        print("ZZP-%s %s | %s ZZ" % (d["action"], d["arm"], d["reason"]), flush=True)
        if a.dry_run or d["action"] in ("DONE", "STOP"):
            return 3 if d["action"] == "STOP" else 0
        if d["action"] == "RUN":
            arm = d["arm"]
            # ⛔ LIVELOCK GUARD. An arm that finished but whose check did NOT read VALID comes
            # back as PARTIAL, and `plan()` would hand it to us again — and `move_aside` would
            # delete-by-rename the run we just paid 11 h for, forever. One attempt per arm per
            # process; anything else is a STOP with the reason named.
            if arm in attempted:
                print("ZZP-STOP %s was already attempted in this process and is still not "
                      "VALID — refusing to spend the card on it again ZZ" % arm, flush=True)
                return 9
            attempted.add(arm)
            spec = next(x for x in ARMS if x["name"] == arm)
            if a.base_config is None:
                print("ZZP-REFUSED no --base-config: a replicate with no base is not a "
                      "replicate ZZ", flush=True)
                return 6
            base = json.loads(Path(a.base_config).read_text(encoding="utf-8"))
            base_argv = list(base.get("argv") or [])
            labels = next((base_argv[i + 1] for i, t in enumerate(base_argv)
                           if t == "--v7-labels"), None)
            checks = {"trainer": (trainer, a.trainer_md5)}
            if labels and a.labels_md5:
                checks["labels"] = (labels, a.labels_md5)
            ok, why = preflight(arm, a.authorise_arm, checks)
            if not ok:
                for r in why:
                    print("ZZP-PREFLIGHT-FAIL %s ZZ" % r, flush=True)
                return 7
            d_arm = out / arm
            if arm_status(out, arm) == "PARTIAL":
                print("ZZP-MOVED-ASIDE %s ZZ" % move_aside(out, arm), flush=True)
            d_arm.mkdir(parents=True, exist_ok=True)
            seed = int(spec["flags"][spec["flags"].index("--seed") + 1])
            cmd_argv = build_argv(base_argv, seed=seed, out=str(d_arm / "run"))
            (d_arm / "launch.json").write_text(json.dumps(
                {"arm": arm, "authorised": a.authorise_arm, "stop_after": a.stop_after,
                 "base_config": str(a.base_config), "run_tree": str(tree),
                 "trainer_md5": md5(trainer), "seed": seed, "argv": cmd_argv,
                 "base_argv_n": len(base_argv),
                 "launched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                indent=1), encoding="utf-8")
            print("ZZP-LAUNCH %s seed=%d ZZ" % (arm, seed), flush=True)
            t0 = time.time()
            with open(d_arm / "train.log", "wb") as log:
                rc = subprocess.run([a.py, "-u", "scripts/refc_v3_train.py"] + cmd_argv,
                                    cwd=str(tree / "stack"), stdout=log,
                                    stderr=subprocess.STDOUT,
                                    env={**os.environ,
                                         "PYTHONPATH": os.pathsep.join(
                                             [str(tree / "stack"), str(tree),
                                              str(tree / "taniteval")]),
                                         "PYTHONIOENCODING": "utf-8",
                                         "OMP_NUM_THREADS": "6"}).returncode
            print("ZZP-TRAIN-RC %s %d wall=%ds ZZ" % (arm, rc, int(time.time() - t0)), flush=True)
            # ⛔ THE CHECK IS WHAT MAKES THIS RESUMABLE. Without `p_arm_check.json` a FINISHED
            # 11 h arm reads as PARTIAL and the next invocation moves it aside. Run it whatever
            # the trainer's status was — the admissible evidence is the artifact, not the rc.
            steps = int(flag_value(cmd_argv, "--steps") or spec["steps"])
            chk = subprocess.run(
                [a.py, str(Path(__file__).resolve().parent / "p_check_arm.py"), str(d_arm),
                 "--seed", str(seed), "--steps", str(steps)],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            (d_arm / "check.out").write_text(chk.stdout + chk.stderr, encoding="utf-8")
            st = arm_status(out, arm)
            print("ZZP-ARM-%s-%s ZZ" % (arm, st), flush=True)
            # ⛔ BACK THROUGH `plan()`, NEVER STRAIGHT INTO THE NEXT ARM. `plan` re-reads the
            # bound, the A7 chain and the gate from scratch, so the ONLY way a second arm starts
            # is by passing every lock again. An INVALID check makes `plan` return STOP here.
            continue
        if time.time() > deadline:
            print("ZZP-TIMEOUT after %.1f h ZZ" % a.max_wait_h, flush=True)
            return 4
        time.sleep(a.poll_s)


if __name__ == "__main__":
    sys.exit(main())
