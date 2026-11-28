# refcv5 STOP — PI-ordered clean stop of `refcv5-ddim-b1-v72-40k`

**Date:** 2026-09-06 · **Agent:** refcv5-stop · **Branch:** `agent/arch-inf-20260803`
**Pod:** `tanitad-refcv3` = `tanitad-a40` = 69.30.85.211:22001 (A40, 46,068 MiB)

## Headline

The run was stopped **cleanly** at **step 5,950 of 40,284 (14.77 %)** on PI directive
(it implements one lever, not the instructed design). Everything is preserved and
md5-verified; **the A40 is free — 0 MiB used, 0 % util, no compute apps.**

No `summary.json` was written, so nothing can read this as a completed 40,284-step run.

---

## 1. The kill, in order, with the `ps` verification after each

⛔ The supervisor was killed **FIRST**. Reversing this order is how `supervise_run.sh`-family
supervisors relaunch the trainer from the command captured at boot.

| # | PID | what | signal | verification (MEASURED) |
|---|---|---|---|---|
| 1 | **2530709** | supervisor `bash /workspace/sup_refcv5.sh` | `kill -9` | `/proc/2530709` absent → `SUPERVISOR 2530709 GONE`; same-breath control `CONTROL_OK trainer 2530863 ALIVE` |
| 2 | **2530708** | orphaned launch wrapper (`bash -c … setsid nohup …`) | — | had **already exited** when its child died: `kill` returned `No such process`, `/proc` absent → `GONE` |
| 3 | **2538212** | the supervisor's `sleep 120` child, orphaned by step 1 | `kill -9` | `/proc` absent → `GONE` |
| 4 | **2530863** | trainer `refc_v3_train.py` | `kill -TERM` (sufficient) | `/proc` absent → `TRAINER 2530863 GONE`; SIGKILL escalation never needed |
| 5 | **2530978–2530983** | 6 dataloader workers | died with parent | all six `/proc` absent, re-checked after 4 s → `GONE` ×6 |

**SIGTERM was safe and sufficient** because the trainer carries **no signal handlers** —
verified positively before killing: `grep -c "signal\." refc_v3_train.py` = **1**, and that one
hit is a *comment* about `DataLoader worker killed by signal: Bus error`; the same-breath control
`grep -c "def main"` read **1** (non-zero), so the grep channel was live. Consequence: the
trainer could not have written a `summary.json` on the way out. `summary.json` is written only at
`refc_v3_train.py`'s completion path, and it is **confirmed absent** in both directories.

⚠️ **Every kill was by explicit PID.** `pgrep -f` / `pkill -f` were never used — they self-match
the ssh command. Every `ps` probe used **bracketed patterns** (`refcv[5]`, `refc[_]v3[_]train`,
`sup[_]refcv[5]`) so the probe could not match its own argv.

### Final process sweep (MEASURED, opaque marker + live control)

```
FINAL-ZZ arm=0 trainer=0 sup=0 supshell=0 CONTROL_jupyter=3 ZZ
```

The control reads **3**, so the `ps | grep` channel is live and the four zeros are genuine
absences rather than a broken probe.

⚠️ **One reading corrected in-flight:** an earlier sweep printed `supervise2`, which looks like two
surviving supervisors. It was a **self-match on my own echo string** (`-supervise${D}-` contains the
literal `supervise`, and the `$(…)` subshell fork duplicates the argv line). Re-probed with a
pattern that could not self-match: `supshell=0`, and a full `bash`/`sh` process listing showed
**only my own probe**. This is the documented monitor-self-match trap, caught by the standing rule
that the emitted token must be disjoint from the searched token.

## 2. Lock-holder scan (`/proc/*/fd`)

Lock path `/workspace/.sup_refcv5.lock`.

| when | holders | detail |
|---|---|---|
| **before** the kill | **1** | `pid=2530709 fd=200` — the supervisor itself, and **only** the supervisor |
| **after** the kill | **0** | `FINAL-QQ lockholders=0 CONTROL_devnull_fds=14 QQ` |

The control (14 live `/dev/null` fds) proves the `/proc/*/fd` scan channel was working, so
`0` is a genuine absence.

⭐ **The `200>&-` fix in `sup_refcv5.sh` demonstrably worked.** Neither the trainer nor any of its
six dataloader workers nor the `sleep 120` ever appeared as a lock holder — the failure mode that
made a run permanently unsupervisable on 2026-09-02 did not occur here. **No leftover holder
blocks a future supervisor on that path.**

## 3. Final step, loss, wall-clock (MEASURED)

Artifact: `…-STOPPED-SAFE/metrics.jsonl` (130 rows = **119 train + 11 eval**, parsed, 0 parse failures).

| quantity | value |
|---|---|
| **final step** | **5,950** of 40,284 target (**14.77 %**) |
| **final train loss** | **9.19979** (`traj` 0.88517, `lat` 0.4236, `lon` 2.1335, `goal2s_err_m` 1.7883, `anchor_acc` 0.40, `u0` 0.12835) |
| **final held-out eval** | step **5,500** — `eval_loss` **8.40540**, `eval_traj` **0.77617**, `eval_lat_tac` 1.0158 |
| **wall-clock** | **24,082.7 s = 6.690 h** (`elapsed_s` of the final train row) |
| **observed rate** | **4.048 s/step** (24,082.7 / 5,950) — matches the briefed ~4.046 |
| launched → stopped | 2026-09-06 **10:13:52Z** → **~16:57Z** |
| checkpoint step in `ckpt.pt` | **5,500** (the last completed save; `save-every` 500) |

⚠️ The eval rows are a **T0 in-training monitor** on the same loss surface. The trainer's own
docstring says it "is NOT the four-metric-family result and must never be quoted as one."

## 4. Checkpoint preservation

**Preserved at:** `tanitad-refcv3:/workspace/experiments/refcv5-ddim-b1-v72-40k-STOPPED-SAFE/`

`cp -p` of 8 files; **md5 computed on both ends and compared — 7/7 identical** (the 8th,
`train.stderr.log`, is legitimately 0 bytes: the run never wrote to stderr).

| file | bytes | md5 (src == dst) |
|---|---|---|
| `ckpt.pt` | 1,299,509,609 | `1f24ad3a53e5a66b27656c4cceb599f0` |
| `ckpt_5000.pt` | 433,377,035 | `6bf675bf4068e5ee31d89560bcdb019e` |
| `config.json` | 8,569 | `faf054b4f618959d7d5376fa9b3d1de1` |
| `metrics.jsonl` | 93,992 | `cdf1f875d885f31d9b4af52a00b1e5aa` |
| `train.log` | 8,403 | `f5fb6a6c08ed83ed54ed1ca12397fd0d` |
| `supervisor.log` | 430 | `a9cbec671be2a677b932207a156a484f` |
| `anchors.pt` | 10,597 | `297f6f1db52f6a56094846b0d7f71ed9` |
| `train.stderr.log` | 0 | (empty by design) |

### Not-all-NUL assertion — the mount has been producing correct-size, all-NUL files

⛔ Size and md5-match alone would **not** catch a poisoned copy if both ends were poisoned
identically, so content was asserted directly.

```
ckpt.pt        nonzero_in_first4096 = 3580
ckpt_5000.pt   nonzero_in_first4096 = 3580
config.json    nonzero_in_first4096 = 4096
metrics.jsonl  nonzero_in_first4096 = 4096
anchors.pt     nonzero_in_first4096 = 3732
train.log      nonzero_in_first4096 = 4096
--- discriminating control ---
all-NUL file   nonzero_in_first4096 = 0
```

The control reads **exactly 0**, so the assertion can in fact detect an all-NUL file; the non-zero
readings above are therefore meaningful and **not** an artifact of a probe that always passes.

### Real `torch.load` on the PRESERVED copy (the strongest assertion)

```
CKPT_TOPLEVEL_KEYS ['model', 'opt', 'step']
CKPT_STEP 5500
CKPT_N_MODEL_TENSORS 557
CKPT_FLOAT_TENSORS_CHECKED 40  NONZERO_NORM 40  SUM_NORM 255.648
CKPT_ALLZERO_ASSERT PASS_NONZERO
CKPT_HAS_OPT True  OPT_STATE_ENTRIES 351
```

The checkpoint is **loadable, complete (model + optimizer + step), and non-zero** — it is a real
resumable artifact, not a correct-sized husk.

**Disk was checked with a real `dd` write test, never `df`:** 2.0 GiB written at **485 MB/s** and
removed cleanly. (`df` reported 204 T available, which is the cluster lie and hides the per-pod
MooseFS quota — it was not used to make the decision.)

## 5. The STOPPED marker

Written to **both** directories as `STOPPED.json` (3,289 bytes each), read back and asserted:

```
READBACK_OK …/refcv5-ddim-b1-v72-40k/STOPPED.json            done= False final_step= 5950
SUMMARY_JSON_PRESENT …/refcv5-ddim-b1-v72-40k                False
READBACK_OK …-STOPPED-SAFE/STOPPED.json                      done= False final_step= 5950
SUMMARY_JSON_PRESENT …-STOPPED-SAFE                          False
```

It carries `"done": false`, `"stopped": true`, `"stop_kind": "DELIBERATE_PI_ORDERED_STOP"`, the
step/loss/wall-clock, the full kill order with verifications, the lock-scan result, every md5, and
this explicit field:

> `NOT_A_COMPLETION_MARKER`: "This run did NOT reach its 40284-step target. It was stopped on
> purpose. summary.json was deliberately NOT written, because {'done': true} would falsely mark a
> 40284-step run complete. Do not resume, do not relaunch, and do not read this file as a
> done-marker."

⛔ **`{"done": true}` was deliberately NOT written**, and `summary.json` is confirmed absent in both
directories — so neither this supervisor's data-derived done-check nor any future reader can
conclude the 40,284-step run finished.

### Second guard — the supervisor script is tombstoned

`STOPPED.json` alone would **not** stop a resurrection, because `sup_refcv5.sh` derives its done
condition from `metrics.jsonl` (`last_step >= TARGET`) and never reads `summary.json` on the way
in. At 5,950 < 40,284 a re-run would have relaunched the trainer and begun overwriting
`ckpt.pt` / `config.json` / `metrics.jsonl` in the canonical run directory. So:

```
/workspace/sup_refcv5.sh  ->  /workspace/sup_refcv5.sh.STOPPED-BY-PI-2026-09-06   (10,464 B, body intact)
/workspace/sup_refcv5.sh.STOPPED-NOTICE                                            (526 B, explains why)
```

Proven loud, not silent: `bash /workspace/sup_refcv5.sh` now returns
`bash: /workspace/sup_refcv5.sh: No such file or directory`. Same tombstone pattern the programme
already uses for terminated pods in `~/.ssh/config`.

## 6. Is the A40 free?

**YES.** MEASURED at 2026-09-06 17:03:48 UTC:

```
NVIDIA A40, 0 MiB, 46068 MiB, 0 %, 30.08 W, 27 C
nvidia-smi --query-compute-apps  ->  (empty)
FINAL-ZZ arm=0 trainer=0 sup=0 supshell=0 CONTROL_jupyter=3 ZZ
FINAL-QQ lockholders=0 CONTROL_devnull_fds=14 QQ
```

**0 MiB of 46,068 MiB used · 0 % utilisation · 30.08 W of 300 W · 27 C · no compute apps · no
python3 · no supervisor · no lock holder.** The only surviving process on the pod is the RunPod
`jupyter-lab` (which is also the live control proving the probe works).

---

## 7. What the stopped arm still bought — preserve this, it is the point of the stop being clean

The arm was a **matched** comparison against `refcv4b-b1-v72-40k`, and the match is verified, not
asserted. Diffing the two `config.json` `argv` records (out-dir tokens normalised):

```
ARGV_ONLY_IN_REFCV5   ['--sampler', 'ddim', '--w-u0', '--agents', 'off']
ARGV_ONLY_IN_REFCV4B  []
```

Same corpus, same label file, same seed 0, same batch/lr/warmup — **nothing in the control is
absent from the treatment.** (That the moved levers are these three flags is precisely the PI's
objection: it moves one mechanism, not the instructed design.)

Matched-step **held-out T0 monitor** (`refcv4b` ran the full 40,284; rows read from each arm's own
`metrics.jsonl`):

| step | refcv5 `eval_loss` | refcv4b `eval_loss` | refcv5 `eval_traj` | refcv4b `eval_traj` |
|---|---|---|---|---|
| 5,000 | **8.46785** | 9.03487 | **0.79721** | 0.95433 |
| 5,500 | **8.40540** | 8.81104 | **0.77617** | 1.01171 |

At both matched eval points the DDIM arm read **lower (better) than its control on both metrics**.

⛔ **This is a diagnostic, not a result, and must not be quoted as a capability claim.** Four
reasons, all binding here:

1. **Tier.** These are **T0** in-training monitor rows on the same loss surface. Per
   `EVAL_DOCTRINE.md` only **T1** is admissible for a capability claim.
2. **No estimator, no interval.** No episode-cluster bootstrap was run; these are point values.
3. **Single seed, no replicate.** Per `H-ESTIM-SEED-1`, even a *separated* CI from a one-seed arm
   cannot distinguish a lever effect from the rig's run-to-run noise floor. There is no
   `refcv5b_replicate`, so the sign of this difference is not attributable to `--sampler ddim`.
4. **14.77 % of the curve.** Early-curve ordering does not predict the 40,284-step ordering, and
   refcv5's own train rows are noisy in the same window (`traj` 0.807 → 1.042 → 0.885 across
   steps 5,000/5,500/5,950).

⇒ The honest statement is: **the DDIM-in-control-space arm was tracking ahead of its matched
control at 5,500 steps on the in-training monitor, and the artifacts to test that properly are
banked.** Whoever designs the replacement arm should read this before discarding the sampler —
the mechanism was not failing when it was stopped, it was stopped for scope.

## 8. Escalations

1. ⭐ **The A40 is free and idle as of 17:03Z.** It is the replacement arm's to take; nothing on
   this pod needs to be cleared first.
2. ⚠️ **`refcv5-smoke`, `refcv4-smoke`, `refcv4-b1-v72-40k.ABORTED-step6400` and
   `refcv4b-b1-v72-40k.SUPERSEDED-flatkappa-step200` are still on the pod's `/workspace/experiments`.**
   I did not touch them — they are outside my remit — but they are the obvious space to reclaim if
   the replacement arm needs quota. Space is not currently tight (2.0 GiB `dd` at 485 MB/s passed).
3. ⚠️ **The preserved checkpoint exists in TWO places on ONE pod, and nowhere else.** Both copies
   are on `tanitad-refcv3:/workspace`. If that pod is terminated the arm is lost. It is 1.7 GB, so
   it cannot go in the repo — **if this arm's checkpoint is wanted durably, it needs an HF push,
   and that is a PI/Master-Mind call, not mine.** This is the only single-point-of-failure left.

## 9. Deliverable manifest

| artifact | where it lives | only copy? |
|---|---|---|
| this record | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-refcv5-stop/RESULT.md` | no |
| `STOPPED.json` (banked copy) | `repo:…/2026-09-06-refcv5-stop/raw/STOPPED.json` | no |
| matched-step comparison output | `repo:…/2026-09-06-refcv5-stop/raw/matched_step_vs_refcv4b.txt` | no |
| checkpoint verification output | `repo:…/2026-09-06-refcv5-stop/raw/ckpt_verification.txt` | no |
| `metrics.jsonl` (full 130-row curve) | `repo:…/2026-09-06-refcv5-stop/raw/metrics.jsonl` + pod ×2 | no |
| `config.json` | `repo:…/2026-09-06-refcv5-stop/raw/config.json` + pod ×2 | no |
| `train.log`, `supervisor.log` | `repo:…/2026-09-06-refcv5-stop/raw/` + pod ×2 | no |
| **`ckpt.pt` (1.3 GB, step 5,500)** | `tanitad-refcv3:/workspace/experiments/refcv5-ddim-b1-v72-40k-STOPPED-SAFE/ckpt.pt` **and** `…/refcv5-ddim-b1-v72-40k/ckpt.pt` | ⚠️ **one pod only** — see escalation 3 |
| **`ckpt_5000.pt` (433 MB)** | same two pod dirs | ⚠️ **one pod only** |
| tombstoned supervisor | `tanitad-refcv3:/workspace/sup_refcv5.sh.STOPPED-BY-PI-2026-09-06` | pod only (body preserved) |
