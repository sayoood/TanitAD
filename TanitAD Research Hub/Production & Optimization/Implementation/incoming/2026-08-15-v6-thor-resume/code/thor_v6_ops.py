#!/usr/bin/env python3
"""Thor-native ops loop for the v6F S-W resume: durability + milestone marking.

WHY NOT `stack/ops/pbattery_watcher.py` AS-IS. That script is a faithful record of
what ran on pod4 and its assumptions do not hold here:

  * paths/venv are pod-shaped (`/workspace/TanitAD/stack`, `/workspace/a2venv`);
  * it polls the **HF-mirrored** step, which needs a push loop already running,
    while on Thor the checkpoint is on local disk;
  * it pauses/restarts `aug120` — a pipeline that does not exist on this box;
  * ⛔ and decisively, it assumes **the GPU is free when the battery starts**
    ("GPU is never shared: the battery only starts after the aug120 PID is gone").
    On Thor the trainer is *still running* at step 10 000, so auto-running the
    battery here would contend with training on the one GPU we have.

⇒ THIS LOOP DOES THE TWO THINGS THAT ARE SAFE WHILE TRAINING, AND REFUSES THE THIRD:

1. **Durability.** Push `ckpt.pt` + logs to HF on a LONG cycle and **verify from the
   far side by size**, never from the push log — the silent-push-failure class
   (a loop reporting success while 100 % of a 3.5 GB payload failed) is written up
   in `POD_HANDOVER_2026-08-13.md §4b`. Right now every step past 6 250 exists on
   exactly one disk; 6 250 itself is safe on HF.
2. **Milestone marking.** At `TARGET_STEP` it snapshots the checkpoint under a
   stable name and writes a marker. Copying a file costs no GPU.
3. ⛔ **It does NOT launch the P-battery.** That needs the GPU, and taking it from a
   4.8-day training run to score a probe is the wrong trade. The marker is the
   handoff: run the battery at a deliberate pause, or on another machine.

⚠️ Bandwidth: the corpus pull just took 223 min over this household line. The cycle
is deliberately long, and a push that overruns its cycle is skipped rather than
queued — a backlog of 3.5 GB uploads would starve the link the trainer does not
need but the household does.
"""
import json
import os
import shutil
import time

from huggingface_hub import HfApi

REPO = "Sayood/tanitad-v6"
PREFIX = "v6F-SW-30k/"
OUT = os.path.expanduser("~/experiments/v6F-SW-30k")
LOG = os.path.expanduser("~/logs/v6_ops.log")
TARGET = int(os.environ.get("TARGET_STEP", "10000"))
CYCLE = int(os.environ.get("CYCLE_S", "5400"))          # 90 min
#: Push the 3.5 GB checkpoint every N steps of PROGRESS — a monotone difference,
#: not a modular window (see the fix note in the loop). 1000 steps at the
#: MEASURED 27.18 s/step is ~7.6 h, which is the durability exposure we accept.
PUSH_EVERY = int(os.environ.get("PUSH_EVERY_STEPS", "1000"))
MARKER = os.path.expanduser("~/experiments/v6F-SW-30k/.last_ckpt_push")
SMALL = ("config.json", "metrics.json", "train_log.jsonl", "train.out")

api = HfApi()
#: set by push_and_verify(); the marker only advances when the FAR SIDE agreed.
_last_verify_ok = False


def _read_marker():
    """Survive a restart of this loop. Without it, every restart re-pushes 3.5 GB
    — which on this household line is ~50 min of the link the trainer does not
    need but the household does."""
    try:
        with open(MARKER) as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def say(msg: str) -> None:
    line = f"[{time.strftime('%FT%TZ', time.gmtime())}] {msg}"
    with open(LOG, "a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def local_step() -> int:
    """Read the step from the LOG, not the checkpoint — torch.load of a 3.5 GB
    file every cycle is wasteful, and the log is written by the same process."""
    p = os.path.join(OUT, "train_log.jsonl")
    best = 0
    try:
        with open(p, errors="ignore") as fh:
            for ln in fh:
                if ln.startswith("{"):
                    try:
                        s = json.loads(ln).get("step")
                    except ValueError:
                        continue
                    if isinstance(s, int) and s > best:
                        best = s
    except OSError:
        return 0
    return best


def push_and_verify(names) -> None:
    global _last_verify_ok
    _last_verify_ok = False
    ok_all = True
    for f in names:
        src = os.path.join(OUT, f)
        if not os.path.exists(src):
            continue
        want = os.path.getsize(src)
        try:
            api.upload_file(path_or_fileobj=src, path_in_repo=PREFIX + f,
                            repo_id=REPO, repo_type="model")
        except Exception as e:                              # noqa: BLE001
            ok_all = False
            say(f"PUSH_FAIL {f}: {type(e).__name__}")
            continue
        # ⛔ Far-side listing, never the push log.
        try:
            info = api.model_info(REPO, files_metadata=True)
            got = {s.rfilename: s.size for s in info.siblings}.get(PREFIX + f)
            ok = (got == want)
            ok_all = ok_all and ok
            say(f"{'VERIFIED' if ok else 'MISMATCH'} {f}: far {got} vs local {want}")
        except Exception as e:                              # noqa: BLE001
            ok_all = False
            say(f"VERIFY_UNKNOWN {f}: {type(e).__name__} — not claiming success")
    _last_verify_ok = ok_all


say(f"ops loop up · TARGET_STEP={TARGET} · cycle={CYCLE}s · "
    f"push_every={PUSH_EVERY} steps · out={OUT}")
last_pushed = _read_marker()
say(f"last confirmed ckpt push: step {last_pushed}")
marked = os.path.exists(os.path.join(OUT, f"MILESTONE_{TARGET}"))

while True:
    step = local_step()
    say(f"step {step}")

    if step and step >= TARGET and not marked:
        snap = os.path.join(OUT, f"ckpt_step{TARGET}.pt")
        src = os.path.join(OUT, "ckpt.pt")
        if os.path.exists(src):
            shutil.copy2(src, snap)
            with open(os.path.join(OUT, f"MILESTONE_{TARGET}"), "w") as fh:
                fh.write(f"reached {step} at {time.strftime('%FT%TZ', time.gmtime())}\n"
                         f"snapshot: {snap}\n"
                         "P-battery NOT auto-run: it needs the GPU, which the "
                         "trainer is using. Run it at a deliberate pause, ALWAYS "
                         "with --speed-echo-control (the P1 speed row is an ECHO: "
                         "R2 0.995 collapses to -0.72 under the v0 shuffle).\n")
            marked = True
            say(f"MILESTONE {TARGET} reached at step {step} — snapshot written, "
                f"battery deliberately NOT launched (GPU belongs to the trainer)")
            push_and_verify([f"ckpt_step{TARGET}.pt"])

    push_and_verify(SMALL)

    # ⛔ FIXED 2026-08-16 — THE OLD CONDITION WAS `step % 1000 < 60`, AND IT
    # STRUCTURALLY ALMOST NEVER FIRED. That window is 60 steps wide; at the
    # MEASURED 27.18 s/step it is open for ~27 minutes, and this loop samples
    # every 90. So it had roughly a 30 % chance of ever catching it, and the
    # 3.5 GB checkpoint — the ONLY artifact that matters here — sat on one disk.
    #
    # Same family as C13 (a guard that cannot fail): a condition whose firing
    # WINDOW IS NARROWER THAN ITS SAMPLING PERIOD is not a schedule, it is a
    # lottery. ⇒ Track the last pushed step and compare a MONOTONE difference,
    # which cannot be missed by sampling — a late cycle pushes late, never never.
    if step and (last_pushed is None or step - last_pushed >= PUSH_EVERY):
        say(f"ckpt push due: step {step} vs last_pushed {last_pushed} "
            f"(interval {PUSH_EVERY})")
        before = os.path.getmtime(os.path.join(OUT, "ckpt.pt")) \
            if os.path.exists(os.path.join(OUT, "ckpt.pt")) else None
        push_and_verify(["ckpt.pt"])
        # ⚠️ Only advance the marker if the far side actually agreed. A push
        # that silently failed must be RETRIED next cycle, not recorded as done
        # — the silent-push-failure class (a loop reporting success while 100 %
        # of a 3.5 GB payload failed) is why this loop verifies at all.
        if _last_verify_ok:
            last_pushed = step
            with open(MARKER, "w") as fh:
                fh.write(str(step))
        else:
            say(f"ckpt push NOT confirmed at step {step} — marker NOT advanced, "
                f"will retry next cycle (mtime was {before})")

    time.sleep(CYCLE)
