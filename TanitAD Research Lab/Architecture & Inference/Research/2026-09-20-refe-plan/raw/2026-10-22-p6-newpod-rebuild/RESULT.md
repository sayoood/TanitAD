# P6 on the new pod: the 3-C bank reproduces across machines; the rebuild is running (2026-10-22, MM)

**Status: REBUILD RUNNING.** P6 (`p6_paper_faithful`) is not training yet. Its progress is safe on HF: `run/ckpt_last.pt` from 2026-10-20 12:52:56Z, and 6 of 25 epochs.
**PI rulings in force:**
- 2026-10-21: *"Use the token from our keys.txt. accelerate things to avoid again the credit issue"*.
- 2026-10-22: *"Stay on this pod"*. The ~23 h CPU rebuild runs on the A40 pod itself, not on a separate CPU pod.

## 1. Why a rebuild

- The old pod stopped on 2026-10-20 at about 13:00Z when the RunPod credit ran out. Its volume could not be restarted.
- The new pod (2026-10-21) started empty.
- The trainer resumes only on the bank it started on. Its preflight compares content: `n_tuples` 167,300, per_rank {0: 103,037, 1: 64,263}, `targets_sha256 8775e73e…`.
- That bank, `train_grow_cmd_3c`, was never banked off the old pod. It has to be rebuilt byte-identically, or P6 restarts from step 0.
- Full input census: `stage_agent/FEASIBILITY.md`.

## 2. The cross-pod control: PASS (MEASURED)

**What it did:** `code/ctl74.sh` re-ran the banked 2026-10-17 74-twin control on the new pod, then compared bit for bit with `code/ctl74_compare.py`. The control covers parts for 26 logs, 16 re-roll shards and the frozen D7 tree plus the 3-C code.

**Verdict (`raw/ctl74_verdict.txt`, `raw/ctl74_result.json`):**
> `ZZCTL74_PASS audit 74/74 bit-equal (max|d f10_raw| 0, max|d v0| 0); kept rows 23/23 equal, new rows 23; max divergence vs banked 0.0 m`

**What it means:**
- The CPU teacher's float32 arithmetic gives the same bits on this machine as on the old one, for n = 74 twins, 23 kept rows and about 2,800 floats.
- This machine is an Intel Xeon Gold 6342 (Ice Lake). The old pod's CPU model was never recorded.
- So the full re-roll (44,409 twins) is expected to reproduce `train_grow_cmd_3c/targets_rank1.jsonl` sha256 `2a63acc2f634…`.

**This is necessary, not sufficient.**
- A bit flip in one twin near the commit boundary would change the file. The merge report from 2026-10-17 shows `f10_round_vs_raw_disagree` 4/1/0 for K1/K2/K3.
- So the acceptance test is the sha of the merged file, not this control.
- ⛔ A mismatch stops the driver. P6 is never resumed on a non-identical bank.

## 3. The volume quota (MEASURED, `raw/volume_probe_2026-10-22.txt`)

**The finding:** `/workspace` is a RunPod **network volume** (MooseFS, EU-SE-1), and `df` cannot see its quota.
- With 252 GiB used, only **16 GiB** more could be written.
- The chain's own stage-0 write gate refused at 05:21:21Z: *"dd write test of 16 GB failed"*.

**This also explains the 2026-10-21 pixel loss.** 14 of 32 shards failed (`ZZPIX_PAR_DONE 18 of 32`). That fetch was anonymous and had no `curl -f`. With `-f` missing, an HTTP error body is saved as the `.tgz`, and the DBs were filling the quota at the same time.

**Fixes:**
1. **The 1,192 nuPlan DBs (164 GiB) moved to the local container disk** with `code/move_dbs.sh`.
   - The DBs are public and re-fetchable.
   - The move runs a 20 GiB local write test first.
   - It compares the full name+size listing before it swaps in the symlink at `data/navtrain_dbs`.
   - SQLite random reads are also far faster on local NVMe than over FUSE.
   - ⚠️ The container disk does not survive a pod stop. After a stop, the chain's `have_dbs || fetch_dbs` re-fetches the DBs (about 40 min).
2. **Pixels:** shards are fetched with the PI's token, with `curl -f` and 5 retries; temp `.tgz` files go on local disk and extraction goes to the volume.

## 4. The rebuild (`code/rebuild_driver.sh`, detached on the pod, resumable, stops on any mismatch)

The banked copy has the token-file path redacted to `<TOKEN_FILE>`. The executed bytes' sha256 is in `raw/executed_script_sha256.txt`.

1. **Wait for the DB move** (`ZZMOVE_OK`).
2. **Pixels 19–32**, authenticated, running beside step 3.
3. **Prebuild.** The chain at **`bcc04b5`** runs with `P6_STOP_AFTER=cache` and without `P6_CBANK`: env → weights → recover (HF) → inputs (bank restore from HF) → commands → D9 metric cache.
   - **Why `bcc04b5` and not the tip:** the tip's resume-bank guard (`raw/2026-10-20-p6-resume-bank/`) refuses a chain without `P6_CBANK` once `recover` has restored `config.json` (3-C). A chain *with* `P6_CBANK` refuses to build `commands`. So the pre-3C bank cannot come from one honest tip run.
   - **What differs:** up to `cache`, `bcc04b5` differs from the tip only by that guard. `p6_pod.sh` diff: guard lines only. `annotate_commands.py`, `build_metric_cache.py` and `p2r_sync.py` are byte-identical. `p6_prep.py` differs (`d2de5ed6` vs `a590b2a7`; the tip adds `resume-bank`).
   - **Acceptance:** the outputs are sha-checked: `train_grow_cmd` rank0 `daa23b96…`, rank1 `2f3e4613…`.
   - No stage marker is written by hand.
4. **Twin parts** (`build_twin_parts_3c.sh`, 44,409, frozen D7 tree, 8 workers).
5. **Re-roll** (`run_full.sh launch`, 720 shards, chunk 32, `NPROC=8`).
   - ⚠️ **Disclosed:** its preflight reads `CONTROLS_JSON` = the **banked** I1–I4 result `raw/2026-10-17-twin-bank-3c/results/controls/controls.json` (old pod, `all_pass`). It does not read a fresh I1–I4 run.
   - Justification: §2 shows this pod reproduces that control bit for bit. The override is the script's own env knob.
6. **Merge**, then verify rank1 `2a63acc2…` and rank0 `daa23b96…`.
7. **Final chain** from the tip tree (`f2f604f`) with `p6_start.env`. That file sets `P6_CBANK=…/train_grow_cmd_3c` and `P6_LABEL_WORKERS=6`, and pins ARGS sha `570619873b1ebaa0`.
   - From there it runs the launch gate, the trainer preflight (`targets_sha256 8775e73e…`), and resume from `ckpt_last.pt`.

**ESTIMATED cost:** about 23 h of CPU at the 7.65-CPU cgroup quota, then the chain's lanes/lab/mem/egr/gate stages (1–2 h). The A40 is idle until then, which the PI accepted ("Stay on this pod").

## 5. Disclosure for the pre-registered run

P6 resumes across a pod change, as in 3-C addendum 5. Disclosed here before any new training data:
- different CPU (§2)
- network volume, with the DBs on local disk (§3)
- `CONTROLS_JSON` override (§4.5)
- the prebuild at `bcc04b5` (§4.3)

The run's identity checks are unchanged: ARGS sha, launch-gate token and trainer content identity.

## Deliverable manifest

| artifact | where |
|---|---|
| this note, `code/`, `raw/`, `stage_agent/` (the staging agent's brief, feasibility, handoff and log) | repo, this directory |
| driver log `driver.log`; pixel log; DB-move log | pod `/workspace/ops/rebuild/` |
| the 74-twin control's full outputs | pod `/workspace/ops/bank3c/controls74/` |
| P6 checkpoint (6 epochs) | HF `Sayood/tanitad-refe-p6-paper-faithful` |
| the 3-C bank, once merged and sha-verified | pod volume, then HF dataset `Sayood/tanitad-refe-navtrain-banks` as `refe_train_grow_cmd_3c.tar.zst`. The pod sidecar `code/bank3c_hf.sh` waits for `merge.done`, uploads, and checks the remote size and LFS sha256 (`ZZB3_OK`). The bank was never banked off the old pod, which is why this rebuild was needed. |

## Addendum 1 (2026-10-22 13:16Z): the twin-parts acceptance check could not import its module; fixed by a one-line PYTHONPATH copy

**What happened.**
- The rank-1 twin parts built cleanly: `ZZMETRIC_CACHE_DONE built=32929 failed=0`, 4,984 s.
- The D7 acceptance check right after them, `check_twin_parts_distinct.py`, crashed with `ModuleNotFoundError: No module named 'online_labeller'`.
- The driver stopped (`ZZRB_FAIL`), as designed.

**Why.**
- The checker's own header requires *"pod, teacher venv, PYTHONPATH incl. $PKG/refe"*. `build_twin_parts_3c.sh` exports only `$PKG/code`.
- `teacher_env.sh` resets PYTHONPATH, so it cannot be passed in from outside.
- `online_labeller.py` exists only in `$PKG/refe` (sha256 `917ca293…`).

**Fix.** `code/build_twin_parts_3c_pp.sh` is a copy of the pinned `build_twin_parts_3c.sh` with **exactly one line changed** (diff: 1 line): `export PYTHONPATH=$PKG/code:$PKG/refe:${PYTHONPATH:-}`.
- The pinned script was not edited.
- The parts are unchanged by it: `build_metric_cache.py` runs from `$PKG/refe`, so its import path already starts there.
- The driver now calls the copy (`code/rebuild_driver.sh`; executed sha in `raw/executed_script_sha256.txt`).

**Result.**
- On relaunch, rank 1 skipped all 32,929 existing parts (14 s).
- The check passed: `ZZD7_DISTINCT_OK k=1 sampled=40 differ=37 frac=0.925 min_frac=0.9`.
- Ranks 2 and 3 followed.
- This is a fifth disclosed deviation for the resumed run (§5).
