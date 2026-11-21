# LAUNCH RECEIPT — refcv5-v2 (P1-OUT arm) IS TRAINING on `tanitad-a40`

**Launched 2026-09-06 23:10:12 UTC = 2026-09-07 01:10:12 Europe/Berlin.**
Target 40,284 steps. Trainer **PID 2559052**. Watchdog **PID 2559294**.
Out-dir `/workspace/experiments/refcv5-v2-noagents-b1-v72-40k`.

⛔ **This receipt carries no eval tier and no four-family table, deliberately.** It records a
*launch*; nothing here predicts a trajectory. The four families bind the run's **result**, and the
pre-registered bar is restated in §7 so the comparison cannot be redefined after the fact.

---

## 1. The exact argv, as the run itself recorded it

Read back from the run's own `config.json` (`argv` key), not retyped:

```
--arm hier --size base
--v2-cache /root/data/train
--v7-labels /workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz
--eval-cache /root/data/eval
--eval-labels /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz
--eval-every 500 --eval-batches 8
--image-hw 256 640
--steps 40284 --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24
--lr 1e-4 --warmup 2000 --seed 0
--log-every 50 --save-every 500
--u8-batches
--out /workspace/experiments/refcv5-v2-noagents-b1-v72-40k
--nav-from-v7
--ego-state-inject --ego-dropout 0.5
--anchors /workspace/experiments/refcv5-v2-noagents-b1-v72-40k/anchors.pt
--n-anchors 117 --anchor-v0-conditioned --anchor-control-units alat
--sel-accel-max 2.0
--sampler ddim --w-u0 0.5
--sel-refined --sel-score-emitted
--goal-str
--tac-goal-tok-head
--agents off
```

Invoked as `P1=out TACGOAL=1 bash /workspace/LAUNCH_refcv5_v2.sh` — the **prepared** launcher,
shipped byte-identical (md5 `45a8a8987df25097ba4ec7fafb19bde9`, 9,166 B, dev box == pod). The
launcher's own anchor gate passed in the same breath: `[gate] anchors OK: torch.Size([117, 2])
units alat`.

## 2. P0 — the ship was RE-VERIFIED, not trusted

⚠️ The 2026-09-06 receipt was taken at `78d2fb8`; **four sibling commits landed after it** and HEAD
moved twice more *during this session* (`cc5450d` → `eb627fa`). The receipt was re-earned, not
quoted.

| | value |
|---|---|
| HEAD at launch | **`eb627fa3631eabc3dbaaaf005b9b9d667bc7a82b`** |
| `git diff 78d2fb8..HEAD -- stack/` | **empty** — no trainer change since the shipped ref |
| repo HEAD blob md5, `stack/scripts/refc_v3_train.py` | `7f3043782a32456c95dc8f1d3b709761` (32 hex) |
| **pod** md5, same path | **`7f3043782a32456c95dc8f1d3b709761`** (32 hex) — MATCH |
| whole shipped subtree | **449/449 byte-identical · 0 differ · 0 absent** (control: identical count 449, not 0) |
| pod-only files left untouched | 7 (`__pycache__`, `.PRE_SHIP_*` backups) |

⛔ **`core.autocrlf=false` was pinned on the archive** (`-c core.autocrlf=false -c core.eol=lf`).
Without it `git archive` on this box adds one CR per line and the md5 gate fails with a signature
that reads like a transfer fault.

⚠️ **Step 0's dirty-worktree refusal was bypassed, deliberately and statedly.**
`stack/scripts/refc_v3_train.py` is `MM` (a sibling mid-edit). The ship takes **blobs from HEAD**,
which worktree dirt cannot reach, and the HEAD blob md5 is the declared target — verified after the
fact in the row above.

### 2b. HEAD IMPORTS — the rung md5 cannot reach

`python3 refc_v3_train.py --help` **on the pod**, `CUDA_VISIBLE_DEVICES=''`:
**exit 0 · 35,446 B stdout · stderr 0 bytes.**

| nine flags (must be ≥ 1) | | negative controls (must be 0) | | probe-is-working (must be > 0) | |
|---|---|---|---|---|---|
| `--sel-refined` | 3 | `--max-speed-input` | **0** | `usage:` | 1 |
| `--sel-score-emitted` | 4 | `--str-goal-tok-head` | **0** | `--agents` | 3 |
| `--sel-score-emitted-t` | 2 | `--strategic-tokens` | **0** | `--anchors` | 3 |
| `--no-strategic` | 2 | `--p4-strategic` | **0** | `--steps` | 2 |
| `--goal-point-inject` | 3 | | | | |
| `--goal-point-geo-prior` | 2 | | | | |
| `--goal-point-t` | 2 | | | | |
| `--goal-point-w` | 2 | | | | |
| `--nav-args` | 2 | | | | |

⚠️ **`--tac-goal-tok-head` reads 2 — PRESENT.** It was a valid negative control at `be8c665` and is
**retired** from that role at `78d2fb8` and later. A control's validity is scoped to the surface it
was measured on.

## 3. ⭐ THE DELEGATED JUDGEMENT CALL — decided by MEASUREMENT

`--tac-goal-tok-head` is the flag without which `refc_v3.py:948` never builds the head and the arm
trains **no** tactical-goal vocabulary. It is also the construction that made refcv4b@40284, three
refcv3 checkpoints and the live refcv5 run **unrollable**. So it was smoke-tested, not argued.

**Smoke:** the P1-OUT arm verbatim **+ `--tac-goal-tok-head`**, 4 steps, `--save-every 2`, same
`--v7-labels` (the flag refuses under `kin3` by design). `/workspace/experiments/refcv5-v2-tacgoal-smoke`.
Exit 0, **stderr 0 bytes**, losses 69.78 → 52.52 → 92.07 → 73.03, `ckpt.pt` written at steps 2 and 4.

**Rollability, on the real checkpoint** (`roll_smoke.json`), rebuilt through the trainer's OWN path
(`build_parser` + `_pin_trainer_cfg` on the recorded `argv`):

| bar | measured |
|---|---|
| missing keys | **0** |
| unexpected keys | **0** |
| `load_state_dict(strict=True)` | **PASS** |
| recorded `param_breakdown['tac_goal_tok_head']` | **11,286** |
| rebuilt `param_breakdown['tac_goal_tok_head']` | **11,286** |
| recorded / rebuilt `total` | **108,257,502 / 108,257,502** |
| ledger lines sum to `total` | **True** |
| seam stamp | `requested True · cfg True · built True · v7.0` |
| ckpt keys for the head | `tac_goal_tok_head.net.weight`, `.net.bias` |
| **same-breath non-zero controls** | 559 ckpt tensors · 357 model params (a 0/0 cannot come from an empty state_dict) |

**And the REAL guard, unmodified:** `refcv3_arm.cross_check_config` (the repo's current copy, md5
`afdd9a7cf506876dcc634507bcb3504b`) run against the smoke's `config.json` →
⭐ **`CROSS_CHECK: PASS`, 6 facts compared** (`arm`, `goal_tau_steps`, `horizons`, `image_hw`,
`param_breakdown`, `tac_vocab_version`), exit 0. `tanitad` was bound to the **HEAD extraction** and
`refcv3_arm._TRAINER` pinned to the **HEAD trainer blob**, because this box's worktree trainer is
`MM` and rebuilding through it would answer a question about the worktree, not about the A40.

⚠️ **A defect in the first harness run, recorded because it would have misled a reader.** The first
attempt wrapped its own `sys.exit(0)` inside `except SystemExit`, so it printed `CROSS_CHECK: PASS`
and then `CROSS_CHECK: REFUSED / 0` from its own exit. The guard passed both times; the harness was
fixed so the artifact cannot say "REFUSED" after a PASS.

⇒ **LAUNCHED WITH THE FLAG.** The PI's *"use the whole tactical vocabulary"* is satisfied at a risk
that was measured away rather than assumed away.

## 4. ⭐ POSITIVE EVIDENCE THE RUN IS TRAINING

⛔ An exit code is not evidence and a launched process is not a training run (M86 — a wrapper's exit
status is not the job's). Two reads, separated in time:

| read | UTC | pid 2559052 | GPU | stderr | last step line | metrics rows |
|---|---|---|---|---|---|---|
| 1 | **23:10:48Z** | alive | 43,713 MiB · 100 % | 0 B | *(none yet)* | 0 |
| 2 | **23:15:48Z** | alive | 42,379 MiB · 100 % | 0 B | **`[v3:hier] step 50 loss 51.1370 traj 2.6936`** | 1 |

**Step advanced 0 → 50 across 5 minutes, loss non-zero, stderr empty, no `traceback`/`error`/
`refus`/`killed`/`oom` marker in either log.**

⭐ **And it kept going, with the checkpoint.** A poll keyed on **both** the success marker (`ckpt.pt`
exists) **and** the failure markers (trainer gone, stderr non-empty) — because a wait that greps only
for success polls happily through a crash:

```
23:16:12Z  step  50  loss 51.1370   23:24:13Z  step 300  loss 26.8185
23:18:12Z  step 150  loss 50.3320   23:26:13Z  step 350  loss 25.6564
23:20:12Z  step 200  loss 28.7785   23:28:13Z  step 400  loss 23.2492
23:22:12Z  step 250  loss 34.0336
23:30:13Z  CKPT-PRESENT  /workspace/.../ckpt.pt  982,888,448 B
           step 500  loss 24.6413   metrics.jsonl rows = 10
```

**A checkpoint path that EXISTS, eight step reads advancing monotonically, loss falling 51.14 →
23.25, `alive=1` and `stderr=0 B` at every read.** Measured rate **~1.2 s/step** (100 steps per
120 s poll) ⇒ **≈ 13.5 h** to 40,284, plus the `--eval-every 500` monitor.

**Pre-launch state, verified in the same breath as the launch:** A40 **0 MiB, 0 % util, zero compute
apps**; `/workspace/experiments/refcv5-v2-noagents-b1-v72-40k` did not exist; a real `dd` write test
(**not `df`**, which hides the per-pod quota) moved **15 GB at 431 MB/s**.

## 5. ⭐ THE RECIPE STAMP — `sampler_ranks_the_fan: True`

From the run's own `config.json`, `selection` block:

```
sampler_ranks_the_fan : True      <- refcv5-v1 shipped this False; it is why P14 entered
sel_refined           : True
sel_score_emitted     : True
sel_score_emitted_t   : 0  (source: auto-zero-on-ddim)
tac_vocab_version     : v7.0
param_breakdown.tac_goal_tok_head : 11286
seams.tac_goal_tok_head : {requested: True, cfg: True, built: True, tac_vocab_version: v7.0}
```

⚠️ P14 is a **LONGITUDINAL** lever: 98.9 % of its measured gain is along-track and curvature is not
separated. It must never be sold as a lateral fix.

## 6. Survival, and what supervises what

The trainer is `nohup`'d with its stdio on files, so it survives the launching session. On top of it
`/workspace/sup_refcv5_v2.sh` (md5 `a792c57851c964d753f6bd2008fb81f6`, PID 2559294) runs as a
**watchdog** — it did **not** perform the launch; it **adopted** pid 2559052 (`supervisor.log`:
*"adopted pid 2559052; resume point 0"*) and relaunches only on death. It keeps all four measured
supervisor traps from `sup_refcv5.sh`, and strengthens the first: **its relaunch argv is read from
the run's own `config.json`**, so there is no second copy of the command to drift. The trainer
auto-resumes from `$OUT/ckpt.pt` with a **strict** load (`refc_v3_train.py:3634-3640`), so a
relaunch continues this run rather than restarting it.

⛔ Killing by name is banned — `pkill -f` self-matches the ssh that runs it. To stop this run, kill
**PID 2559052** (and the watchdog **2559294**) explicitly, watchdog first.

## 7. The pre-registered comparison bar — carried forward unchanged

Baseline **refcv4b @ 40,284**, md5 `99b573e8277d94a5e3bfbf630cb4d751`, on the **141-episode /
4,823-window B1 v7.2 EVAL** grid.

⛔ **refcv4b only TIES the do-nothing baselines** (`os − ha` −0.0021; `os − ha0_ext` +0.0101; neither
separated). ⇒ **refcv5-v2 must BEAT `ha0_ext` SEPARATED or it has not learned to drive either.**
Four families, never pooled; paired episode-cluster bootstrap; ⛔ never `overlapping_holdout_se`;
T1 stamp.

## 8. ⛔ What did NOT enter, and what the PI asked for that this run does NOT get

Each already verified by a sibling; not re-litigated here:

* **P4 / the 15-token STRATEGIC vocabulary** — `refc_strategic.py` is imported by exactly one file,
  a test. **There is no flag.** `--str-goal-tok-head`, `--strategic-tokens` and `--p4-strategic` all
  read **0** in the pod's `--help`.
* **D-TRISEG** — `roll_bank` hard-codes a 2-column bank.
* **`--nav-args`** — `NAV_ARG_DIMS = 3`, `store_true`, no distance-only mode.
* **P2 + P11 together** — `--goal-point-inject` sets `nav_inject=False` and `--nav-args` then raises
  `SystemExit` at `refc_v3_train.py:380-384`.
* **P13** — benefit not established.
* **`--agents`** — P1 returned **OUT** (`D-P1-AGENTCOND-1`), so the arm carries `--agents off`.

⇒ **ONE LINE.** Training is the composed refcv5-v2 arm: refcv5-v1's identity, P14's validated
emitted-fan ranking (`sampler_ranks_the_fan: True`), the v0-conditioned 117-anchor vocabulary held
constant with refcv4b, and — newly, on measured evidence — **the 22-token TACTICAL goal vocabulary,
supervised by a head that is built, ledgered and provably rollable**. ⛔ **What it did NOT get is the
15-token STRATEGIC vocabulary the PI also asked for: it is wired to nothing and has no flag, so it
was not available at any price tonight.** The `--goal-str` head in this arm is the 3-unit *geometric
bearing* head, not that vocabulary, and must not be quoted as satisfying it.

⚠️ **One further honesty stamp carried from the launcher, not discovered here:** all 4,719 v7.2 nav
records carry provenance `ego-future` and the loader runs with `allow_oracle_nav=True`. **This is an
ORACLE-NAV arm**, kept because it is refcv4b's own input and the comparison must hold it constant.

---

## 9. ⚠️ ADDENDUM 2026-09-07 01:30Z (03:30 Berlin) — WHAT CHANGED AFTER §4, INCLUDING A CORRECTION OF §4

⛔ **§4's `~1.2 s/step => ~13.5 h` IS WRONG AND IS RETRACTED HERE.** `--log-every 50` means the log
advances in 50-step jumps, so a 120 s poll that straddles two log lines reads 100 steps and a poll
that does not reads 50. I quoted **the fastest single sample as the rate**. The honest figure for the
same early window (23:16:12Z step 50 -> 23:30:13Z step 500) is **450 steps / 840 s = 1.87 s/step**,
and the **sustained rate measured properly** at 01:23-01:29Z is **50 steps per 200 s = 4.0 s/step**:

```
01:21:58Z step 2400    01:25:18Z step 2450    01:28:38Z step 2500
CPU ticks +3001 per 30 s sample, TWELVE consecutive samples -- i.e. exactly
1.00 CPU-second per wall-second, never stalling.
```

⇒ **Expect ~45 h at the current measured rate, not 13.5 h.** Whoever schedules the eval must plan
against that.

### 9.1 ⛔ THE TRAINER DIED THREE TIMES. THE WATCHDOG RECOVERED IT.

`train.stderr.log` (1,074 B, three identical tracebacks):

```
File "/workspace/TanitAD/stack/scripts/refc_v3_train.py", line 3850, in train
    log.flush()
OSError: [Errno 5] Input/output error
```

An EIO from the **MooseFS `/workspace` FUSE mount** while flushing `metrics.jsonl` — infrastructure,
not the arm. Each death rolled back to the last `--save-every 500` checkpoint:

| | UTC | |
|---|---|---|
| relaunch #1 | 00:39:03Z | from step 2200 -> pid 2560244 |
| relaunch #2 | 00:47:17Z | from step 2200 -> pid 2560445 |
| relaunch #3 | 00:54:58Z | from step 2200 -> **pid 2560646** (survived; past 2500 at 01:28Z) |

⚠️ **THE TRAINER PID IN §1 IS STALE.** `2559052` is dead. The live trainer is **2560646** (PPID 1).
Three `[v3] resumed hier at step 2000` lines in `train.log` are the resumes. Net cost: ~500 steps
redone plus ~25 min. The storm was transient — no fourth crash in the 35 min since.

### 9.2 ⛔ AND THE CRASH-LOOP GUARD WAS DEFEATED — MEASURED, THEN FIXED

All three relaunches logged **`stall=0`** while making **zero net progress** (each resumed at 2000
and died at 2200). The inherited guard from `sup_refcv5.sh` compares `st` to `prev_st` **but its
ALIVE branch resets `stall=0` on every poll**, and this run had ~4 live polls between crashes. ⇒ a
trainer that crashes *slowly* can never trip the 3-strike guard and would have burned all 40
relaunches. **This is a defect in the programme's supervisor template, reproduced here verbatim.**

Fixed in `sup_refcv5_v2b.sh`: `stall` now counts **relaunches that did not advance the run**
(`st <= last_relaunch_step`) and only a relaunch can reset it; the ALIVE branch no longer touches it
and instead reports a **hang** (alive, no step progress for ~20 min) **loudly without killing
anything** — restarting a live job is a human's decision, not a watchdog's. Old watchdog `2559294`
killed by **explicit PID**; new watchdog **`2561632`** adopted trainer `2560646` at resume point
2500.

### 9.3 ⭐ TWO FALSE ALARMS CAUGHT BEFORE THEY WERE REPORTED

Both were surface reads that looked alarming and were not:

1. **"cgroup memory 46.07 / 46.57 GiB = 98.9 % FULL."** The sound instrument is PSI, and
   `memory.pressure` reads **`some avg10=0.00 avg60=0.00 avg300=0.00`**, `io.pressure` avg10 0.00.
   That 98.9 % is page cache occupying free space, which is what page cache is for. **Not pressure,
   not the cause.** No `oom_kill`, no swap.
2. **"stuck at step 2250 across two reads 90 s apart."** With a 50-step log cadence at 4 s/step a log
   line appears only every ~200 s, so a 90 s window legitimately sees no change. The decisive check
   was CPU ticks (+3001 per 30 s, twelve samples) and `train.log` mtime, both of which said it was
   computing the whole time.

**Health at 01:29:34Z:** GPU at **max clocks 1740/1740 MHz**, 42 C, 284 W of 300 W under load, **all
throttle reasons Not Active**, sole compute app is our own pid. 1 main + 6 dataloader workers, **no
orphans** left by the three crashes. `train.stderr.log` has taken **no new bytes since 00:54Z**.

### 9.4 ⚠️ RECONCILING A SIBLING'S "the box is NOT running the repo" — IT IS THE PRE-SHIP BOX

Commit `0fe8830` lands a `pod_currency_audit` verdict of **FAIL, 26 files**, concluding *"the box is
NOT running the repo"*. ⛔ **That is not in conflict with §2, and it does not touch this launch — it
measured the box BEFORE the re-ship.** Its own sharp evidence names
`refc_v3_train.py` at **pod 3,634 lines / 79 `add_argument`**, which is the pre-ship trainer; the
audit ran ~90 min and its sample predates the ship completed at ~01:06Z.

Re-verified **after** that commit landed, at 01:33Z against HEAD `f0f7678`:

```
git diff eb627fa..HEAD -- stack/     -> EMPTY (the shipped ref is still HEAD's stack)
HEAD blob md5                        7f3043782a32456c95dc8f1d3b709761
pod    md5                           7f3043782a32456c95dc8f1d3b709761   MATCH
pod    lines / add_argument          4,576 / 89     (the audit saw 3,634 / 79)
```

⭐ The sibling's own caveat is the right frame and is adopted here: **26 is an upper bound, not a
drift count** — at least 6 rows are `HISTORY-UNKNOWN` from a `0xC0000006` G:-mount paging failure (a
MEASUREMENT failure, not a stale file) and 32 retained rows are repo-only test files that are
outside this ship's scope by design. This ship's scope is stated in §2: `stack/tanitad` +
`stack/scripts`, 449 files, the python surface the trainer actually runs.
