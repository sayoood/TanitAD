# COMMS — refcv7 NavSim suite (EvalFlyWheel NavSim stream, 2026-09-28)

## ⭐ Escalations (each also in the report headline)

1. ⛔ **refcv7's SPEED-CEILING FILTER NEVER REACHES THE EMITTED PLAN — the declared-but-not-reaching
   class of refcv6's escalation 9, surviving FIX-4.** MEASURED on the step-1,500 checkpoint
   (VALIDATION ONLY), warmup, CUDA: R7_A1 (filter ON) and R7_FILTOFF (filter OFF, a real second
   forward) emitted **bit-identical plans on 204/204 scenes**, although the ceiling masked ≥ 1
   candidate in 97 of the 126 scenes with a finite ceiling and the emitted plan lay OVER the ceiling
   in 2 (`raw/validation/step1500/_superseded_pre_A1/bridge_warmup/`; re-measured on CPU by
   `tests/test_model_seam7.py::test_KF_…`, 4/4 identical). Mechanism, read at source:
   `RefCV3Model.forward` RE-SELECTS over the fan after the decoder (E9 goal selection,
   `stack/tanitad/refs/refc_v3.py:2285-2298`): `blended = apply_seam_clamp(sel_score, goal_gate ·
   scorer)` → `rank` masked by `reach_keep` ONLY → `argmax` → `out["traj"]`; the decoder's
   ceiling-filtered pick (`refc.py:3605-3610`) is demoted to `sel_idx_base`. The additive terms (nav
   compliance, behaviour) survive inside `sel_score`; the ceiling — a MASK — is dropped.
   ⇒ SPEC_REFCV7 A2 ("the fed set-speed filters the ARGMAX"; "every refcv7 evaluation reports filter
   ON and OFF") is BUILT in the decoder and INERT on the plan the car would drive. G-DVB checks the
   filter is built; G-LIVE checks the mask is active in an eval step; **neither asserts that the
   EMITTED plan obeys it** — that is the missing guard. This applies to EVERY refcv7 evaluation, the
   PhysicalAI four-family battery included (its filter ON/OFF readings will be identical by the same
   code path). **Owner: Master Mind / Arch.** The zero-training fix is to apply the ceiling mask at the
   E9 argmax; this suite prices it with the DIAGNOSTIC arm `R7_CEILDECL_d` (SPEC amendment A1) and
   takes no decision — evaluating or shipping a model other than the one trained is the PI's call.
2. ⚠️ **The shared dev box is RAM-bound this morning.** At 05:50 Berlin two other jobs held ~15 GB
   RSS (the battery's `g0_refcv7.py` 9.9 GB; a Research Lab job 5.1 GB), available RAM fell to
   1.8 GB and E1's RAM guard (correctly) aborted BOTH of this stream's harness scorers (navhard CV at
   ~74/450 stage-1, navtest STOP). Both are RAM-gated and retried automatically; they cost only time.
   At the milestone the same squeeze would stretch the CPU scorers — the suite gates every scorer on
   ≥ 6 GB free and retries a guard abort up to 8 times.
3. ⚠️ **A MSYS-path defect in the refcv6 suite's bash queue (`score_queue6.sh`, ported here as
   `score_queue7.sh`)**: `P="$(cd … && pwd)"` is a `/d/…` path, and every JSON read the script does
   through `python -c "open(r'$P/…')"` fails and reads ABSENT — the refcv6 queue's
   `status=NO_COUNTS` log lines are this defect, and its PASS_ALREADY skip never fires (a re-invoked
   queue re-scores a passed arm). Harmless where a queue runs once; it would have re-scored a passed
   12,146-token control six times in this stream's first harness chain (caught, killed by PID, and
   replaced by `code/harness7.py`, which reads artifacts in Python). The milestone runner does not
   use the bash queue. **Owner: whoever maintains the refcv6 suite** (fix: `pwd -W`, or read in
   Python).

4. ⛔ **THIS STREAM HELD THE DEV-BOX GPU LOCK STALE FOR 42 MINUTES (06:59–07:41 Berlin) — my error,
   now fixed.** I killed a waiting validation runner by explicit PID (`taskkill /F`) at ~06:59 without
   re-reading the lock; it had acquired the lock seconds earlier (04:58:54Z) and a forced kill skips
   its `finally: release`. The lock (`refcv7-navsim-warmup`, pid 34848, dead) blocked the battery
   and the Research Lab until I found it at 07:41 and released it with its own token; the battery
   (`refcv6-38k-perc-dump`) took the card at 07:41:13. **Fix:** `gpu_lock.reap_own_stale()` — before
   every acquisition, a lock whose job is `refcv7-navsim*` AND whose pid is dead is released; a
   foreign lock is NEVER touched, dead pid or not (`tests/test_gpu_lock7.py`, 6 cases). Rule for this
   stream: after killing any process of this suite, read the lock.

## Decisions made in this stream (reversible)

| # | decision | why |
|---|---|---|
| D1 | the NavSim-side input functions are the refcv6 suite's, IMPORTED from the clean tree (`boot7.py`), under refcv7 arm specs mapped to the R6 arm with the identical declared set / nav / max-speed source (`R6_TEMPLATE`, literal test) | "import, never copy"; the functions and their literal tests are already validated on refcv6 (KL bit-exact) |
| D2 | the model is built only by `stack/tanitad/eval/refcv7_loader.build_model` (strict, G-DVB, stamps), and the forward is fed by `forward_kwargs7` = compute_losses_v3's keyword set | the brief; KT7 proves bit-identity with the trainer's own call on a real eval window |
| D3 | the 0.25 m lift for a NavSim rig is `build_lift_geometry` with every parameter READ OFF `model._lift_bank_hires` | never retype a model constant; KL-lift + mutation arms |
| D4 | the prior is scored as its own model-free arm (PRIOR_ha0p) | what the residual adds is THE NEW-1 question; it costs no forward |
| D5 | SPEC amendment A1 (before any score): R7_FILTOFF_d withdrawn, KF redefined, R7_CEILDECL_d added as a diagnostic | escalation 1 |
| D6 | the validation checkpoint is the battery stream's `ckpt_1500.pt` (md5 `c35966f7…`, verified); this stream's own duplicate pull of the same file (same md5 before/after on Thor) was deleted | the brief: reuse the battery's pull |
| D7 | the milestone waiter waits for the battery's lock to be SEEN held and then quiet for 30 min (fallback: 4 h without a battery lock) | the brief: the battery goes first; a gap between battery stages is not a release |
| D8 | the runner releases the GPU lock between splits and never holds it during CPU scoring | the battery may need the card between NavSim splits |
| D9 | on CUDA the exact-duplicate dedup is DROPPED when KD fails, CUDA itself is refused only when K0 fails (the refcv6 suite's D15 rule); the runner first refused CUDA on a KD miss and was corrected | MEASURED 2026-09-28 on `ckpt_1500.pt`, CUDA bf16: dedup vs native = selections 4/4 identical, max \|Δ\| **1.196 mm** > the registered 1 mm (`raw/controls/KD_exact_dedup_cuda.json`); K0 PASS. The native path costs 10 backbone passes per static-window scene: **0.35–0.42 s/scene** on the card (vs 0.24–0.27 with the dedup) ⇒ a milestone's bridges ≈ 4.4 GPU-hours (navtest 2 × 12,146 + navhard 2 × 5,912 + warmup 7 × 220 + 3 × 200) |
| D10 | the step-5,000 waiter was RE-ARMED at 06:35 Berlin (the 05:52 instance killed by PID while it was still waiting for the checkpoint): its battery-release condition now reads the LOCK only (a lock-less python on the card is logged, not waited on) and it gains a 10 h timeout | a lock-less `uv` python 3.11 GPU process from another session was on the card at 06:34; under the first rule it could have held the waiter forever |
| D11 | the waiter was RE-ARMED again at 06:46 Berlin (Windows PID 33704): the battery is identified by a lock job naming the milestone step (`refcv7-…-5000`), the Research Lab's lock on the SAME file is logged (`LOCK_SEEN_OTHER`) and waited out, foreign lock fields are tolerated, and a lock that is not ours is never removed (`gpu_lock.release` deletes only a lock carrying our token) | Master Mind 06:45 Berlin: the Research Lab now takes the same lock (GPU runs ~08:00-12:30, a REFe snapshot eval ≈ 13:25 for 50-70 min); a Research Lab release must never be read as the battery's |
| D12 | the waiter was re-armed a third time at 07:42 Berlin (Windows PID 25012): the battery tag match also accepts `5k` / `5,000` in the lock's job name | the battery's job names seen so far (`refcv7-g0-1500`, `refcv7-smoke-1500`, `refcv6-38k-perc-dump`) show the step is not always spelled the same way; a missed match only delays NavSim to the 4 h grace fallback, never lets it jump the battery while the lock is held |

## ⭐ Escalations added 2026-10-04 (EvalFlyWheel NavSim operator; each also in the report headline)

5. ⛔ **THE STEP-30,000 MILESTONE WAS KILLED TWICE, AND NEITHER DEATH WAS VISIBLE IN ITS OWN LOGS.**
   MEASURED from the Windows System log and the artifacts' mtimes (the runner logs are silent on
   both): **(a)** 2026-10-02 12:46–12:50 Berlin, a burst of USB-storage resets on the external drive
   then lettered D: (`UASPStor` 129 ×14, `disk` 153 ×5, disk 1). The navtest MAIN bridge (CPU fp32)
   died at 12:50 at R7_A1 row **11,074 / 12,146** with **no traceback** — its log lived on the drive
   that was resetting. `run_navsim_refcv7.py` does not read a bridge's exit status (it reads seams), so
   it logged `SEAMS navtest … NO_SEAM` and **never launched a navtest main scorer**. **(b)** 2026-10-02
   16:55:03 Berlin, a **user-initiated restart from the Start menu** (`User32` 1074,
   `StartMenuExperienceHost` on behalf of `FREEDOM2035\Admin`; boot 16:55:31). Every D: artifact
   stops at 16:55:06; the navhard PRIOR_ha0p scorer was at stage-2 scenario 4,281 / 5,462. The drive
   came back as **E:**; the Master Mind's `subst D: E:\` restores the paths **for this logon session
   only** (a reboot or another logon loses it again). ⇒ **Owner: Master Mind / PI** — a restart of the
   dev box kills every detached NavSim job; the external drive is a single point of failure for every
   banked artifact of this suite (it holds the only copy of the 30k rows and scores).
   **Fix shipped here:** `code/complete_milestone7.py` completes a dead milestone with zero
   re-computation (resumes rows, never rewrites a seam, mirrors its log to local disk). **Fix NOT
   shipped (owner: this stream, next):** the runner should log each bridge's exit code and mirror
   `runner.log` to local disk.
6. ⛔ **MY ERROR — THE STEP-50,400 WARMUP SPLIT RUNS ON CPU BECAUSE I LAUNCHED THE RUNNER FROM A C: WORKING
   DIRECTORY.** `run_navsim_refcv7.py` decides K0/KD from the literal pytest line
   `PASSED tests/test_model_seam7.py::test_K0…`. Launched via WMI with cwd `C:\Users\Admin`, pytest took
   the cwd as rootdir, could not express the D: test path relative to it, and printed
   `PASSED ::test_K0_same_seed_is_bit_identical` — **K0 PASSED and the runner read it as FAILED**
   (`CUDA controls warmup: K0=False KD=False` → `CUDA REFUSED for warmup` → CPU fp32;
   `raw/milestones/step50400/cuda_controls_warmup.log`). A kill-and-relaunch of the runner was
   **refused by the session's permission classifier**, so the runner continues: warmup on CPU fp32 (one
   device per split holds; it is the same device as step 30,000's warmup, not step 5,000's).
   **Fix shipped:** `pytest.ini` at the package root pins rootdir to the package — MEASURED from cwd
   `C:\Users\Admin`: `--collect-only` now prints `tests/test_model_seam7.py::test_K0_…`, so the live
   runner's navtest and navhard K0 checks read correctly. Same family as the CLAUDE.md "check that shares
   the defect it checks for": a pass decided by a string whose shape depends on the caller's cwd.
7. ⚠️ **RAM is the throttle tonight.** At 00:30 Berlin the box had **4.9 GB available** (31.8 GB total;
   commit free 3.3 GB) with other sessions' REFe scoring and a Research Lab census running. Both NavSim
   pipelines gate on it (CPU bridge ≥ 6 GB sustained; scorers ≥ 8 GB on two samples) and **wait, never
   override** — so the completion times below are lower bounds.

## Decisions made in this stream, 2026-10-04 (reversible)

| # | decision | why |
|---|---|---|
| D13 | step 30,000's navtest is completed on **CPU fp32**, resuming the 11,074 banked R7_A1 rows; R7_A1_s1 (0 rows) also runs on CPU | the banked rows are CPU fp32 and `run_bridge7` refuses a mixed-device arm; one device per split (A3) keeps the seed replicate on the same device as the arm it floors. Re-running the split on CUDA (≈ 2.6 GPU-h, faster) would re-compute 11,074 banked rows — the brief forbids it |
| D14 | the completion driver never rewrites an existing seam | `np.savez` stamps zip times: a rewrite changes the bytes and orphans the `seam_sha256` recorded by the PASS score beside it |
| D15 | the step-50,400 runner gets `--gpu-wait-s 43200` (12 h) instead of the per-split defaults (900 s warmup / 10,800 s others) | at step 30,000 the 3 h default sent navtest to CPU (≈ 12 h CPU bridge, then lost to the USB resets); the brief asks for CUDA bridges. A scheduling knob only: arms, splits, inputs, estimators and every other argument are identical to steps 5,000 / 30,000 |
| D16 | `pytest.ini` (no options) added at the package root | escalation 6 |

8. ⚠️ **Collision with the battery stream at 00:32–00:33 Berlin, probably mine.** I launched the step-50,400
   runner (00:32:43) and the step-30,000 completion (00:33:01) when commit headroom read **3.3 GB free**
   (`Win32_OperatingSystem.FreeVirtualMemory`, 00:32). The battery's first `g0_diag_r7.py` attempt took
   the GPU lock at 00:32:47 and died with **`MemoryError`** while hashing the checkpoint at 00:33
   (`D:/refcv7_eval_kit/battery/g0diag_step5000/diag_attempt1_memoryerror.log`). Attribution to my two
   launches is **UNVERIFIED** (other sessions were also running), but the timing fits. The battery
   stream recovered on its own (a commit gate, `wait_commit.json`; lock re-taken 01:10:32). Since then
   that job holds **~15.3 GB private**, which aborted this stream's navhard PRIOR_ha0p re-score twice
   through E1's RAM guard (01:09, 01:26) — retried automatically, no data lost, only time.

## 2026-10-04 evening (22:00-23:00 Berlin): RAM governor for the step-50,400 scorers + the LEGAL-input row (EvalFlyWheel NavSim operator, second shift)

| # | what | evidence |
|---|---|---|
| E1 | **22 of the 29 step-50,400 scorer launches ended on the scorer's OWN RAM guard** (`E1/W3_RAM_GUARD_ABORT`), ~7.96 scorer-hours lost, although the scorers held only 0.84 GB (navhard) / 2.22 GB (navtest). The guard's HARD floor is ONE 2 s sample under 2,000 MB; the box swings 1.0-10.4 GB inside 90 s (MEASURED 22:00) because of other streams (4 pytest suites of 1-3 GB each, the battery, Chrome 8.7 GB, D6) | `raw/milestones/step50400/runner.log`, `scores_*/**driver.txt`, `scores_*/**manifest.json` (`resources`) |
| E2 | Fix, inside THIS package's drivers (the running runner PID 31600 is untouched): `code/ram_governor7.py` - slot lock (2), start window, PAUSE (suspend + trim) below 3,500 MB, resume only with room for the pages given back, per-job lock, PASS-skip guard. **The guard's floors are NOT lowered** (soft 3,000 MB x 60 samples, hard 2,000 MB x 1 sample); the governor sits 1.5 GB above the hard floor. `score_arm7.py` / `score_navtest7.py` call it; the originals are in `raw/milestones/driver_backups_pre_governor/` | `tests/test_ram_governor7.py` (26 tests, 6 mutations RED); live: warmup R7_A1 PASS after 2 pauses / 643 s; governed navtest sub200 re-run reproduces the banked VMAXOFF PDMS 71.6977 to 4 dp |
| E3 | The runner was started with `--splits navtest,navhard`: **warmup was never scored at step 50,400** (first incarnation died on ENOSPC before its pool drained). 9 warmup seams were OK and unscored; `score_queue_gov7.py` now scores them | `scores_warmup/queue.log` |
| E4 | LEGAL-input row for BAR-R8-N4: `R7_VMAXOFF` verified to feed the bare command + the UNKNOWN speed row and nothing privileged (file:line in the report); full navtest + navhard bridge + score queued behind the GPU lock: `code/run_vmaxoff_legal7.py` | `raw/milestones/step50400/vmaxoff_legal/` |
| E5 | Detached finishers: `code/finish_step50400_7.py --phase runner` (after the runner exits: re-scores leftovers, warmup + every split post-processed with the runner's own steps, MILESTONE_SUMMARY, BARS, compares vs 30k and 5k) and `--phase vmaxoff` | `finish.log`, `vmaxoff_legal/finish_vmaxoff.log` |

Restart recipe if the runner (PID 31600) must be replaced: do NOT re-run `run_navsim_refcv7.py` (it would re-enter the CUDA bridges). `python code/complete_milestone7.py --milestone raw/milestones/step50400 --ckpt D:/refcv7_eval_kit/ckpt/ckpt_50400.pt --md5 b418d0fc4a92a6848c246a6a7c50207b --compare-to raw/milestones/step30000` re-uses every OK seam and every PASS score, scores the rest through the (governed) drivers, and rebuilds MILESTONE_SUMMARY / BARS (its summary text says "runner died on the host restart" - that wording is step-30,000's). Arm claims for the Thor backend: `raw/milestones/step50400/CLAIMED_ARMS.txt`.
