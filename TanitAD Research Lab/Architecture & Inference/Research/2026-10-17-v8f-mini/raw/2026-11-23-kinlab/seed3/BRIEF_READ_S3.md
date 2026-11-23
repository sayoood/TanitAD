# Brief: the SEED3 read extension on the dev box (card PREREG_V8F_SEED3, 75f3b191): the one-seed s2 read of B0_s2 / RH_s2 / KI_s2 and the card's naming. Code only; no s2 checkpoint exists yet

You are a TanitAD build agent working for the Master Mind (MM). First read:
* `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip (`git -C C:/Users/Admin/tanitad-push show origin/agent/arch-inf-20260803:"Project Steering/AGENT_OPERATING_STANDARD.md"`);
* the card `F:/TanitAD-scratch/rollhist/seed3/PREREG_V8F_SEED3.md` (FROZEN, sha256 75f3b191). Sections 3 and 4 list the reads and the names;
* the KI read as your model, in `F:/TanitAD-scratch/rollhist/read/`:
  * `rh_common.py` (the `RH_AKIND` switch, `PAIRS`, the seed list), `rh_read.py` (its one-seed mode and the floor file it reads), `mech_ki.py`;
  * `rh_read_ki_chain.sh`, `rh_read_ki_retry.sh` / `.cmd`;
  * the KI outputs in `out/read_ki/`.

## Rules
* Stage, commit and push nothing.
* No GPU, and nothing run on the pod. Pulling checkpoints is the chain's job at read time, not yours.
* Do not edit protected files: `launch_gate.py`, `stack/tanitad/models/__init__.py`, anything under `E:/Projects/TanitAD/stack`, or the sha-pinned `v8f_t1_eval.py`.
* **Change no frozen reader** (`headway_read_r2.py`, `q3_read_r3.py`, `rh_strat.py`, the family reader), and no cell, floor value, estimator, tier or executor.
  * The finished RA, RK and KI reads must stay reproducible. `RH_AKIND=KI python rh_read.py --selftest`, and the same for RK and RA, must still pass after your edit; show it.
  * Keep a `.bk_s3` copy of every file you edit.
* **Permission refusals:** if a tool call is refused, do not retry it in another form and do not ask anyone else to run it. Record it and hand back.

## What to build (in `F:/TanitAD-scratch/rollhist/read/`)
1. **The s2 arms in the read:**
   * pull B0_s2, RH_s2 and KI_s2 from `/root/rollhist/runs/<arm>/train/ckpt_last.pt` after each arm's `ARM_DONE`, checked by the pod's sha256, as `rh_read_ki_chain.sh` does;
   * 6 forms × 3 arms = 18 rollouts at T1-kin/hb0.2, and T0 evals of the three checkpoints with the read's T0 instrument;
   * outputs to `out/read_s3/`.
2. **The decisive reads**, card section 3, one-seed at s2:
   * **G2:** RH_s2 − B0_s2 at T0 ADE6, paired, with the upper bound against **0.1174**, and against 0.0705 printed beside;
   * **P1-s2:** RH_s2 − B0_s2 under `R0C10XHb20`, four families, through the frozen readers in their one-seed mode, with the frozen two-seed F_c floors of the RH read (name the floor file and its sha256 in your facts).
3. **The beside reads** of section 3:
   * KI_s2 − B0_s2: G-T0 under both bars, and P1 with KI's two-seed floors;
   * MECH4 at s2 (KI_s2 − RH_s2 at T0), with the three-seed MECH4 list;
   * KI_s2 − RH_s2 under C\* and R0Hb20;
   * RH_s2 − B0_s2 under R0Hb20; P2, P3 and P4 on RH_s2;
   * the s0 / s1 G-T0 points of RH, RL, RA, RK and KI under 0.1174, from their existing GT0 / MECH JSON files, printed only;
   * three-seed pooled reads only if the frozen readers take three seeds unchanged; otherwise print NOT-COMPUTED with the reason.
4. **The name, `s3_name.py`:** card section 4 exactly.
   * Precedence: BLOCKED, then T0-FAILS, then HARMS, then REPLICATES, then NO-EFFECT.
   * The `-STRADDLE` suffix when a decisive cell's sign-resolved reading changes the name.
   * Output `RESULT_SEED3.json` with every input value and the name.
5. **`rh_read_s3_chain.sh`** with `rh_read_s3_retry.sh` / `.cmd`, from the KI ones.
   * Single quotes inside every `python -c` string. The RK chain once broke on a double-quoted `"RK"`.
   * Resumable steps (`out/steps_s3/`).
6. **Tests:**
   * a planted check that drives `s3_name.py` through each of the five names and through the `-STRADDLE` case, with the expected outputs;
   * `bash -n` on the scripts;
   * a dry walk of the chain that stops before the pull step and prints the pairs it would read.

## Dev box
* CPU only, `CUDA_VISIBLE_DEVICES=-1`, one python process at a time behind `F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60` (4.0 for a run under 1.5 GB; say so).
* Nothing on C:. TMP and TEMP go under `F:/TanitAD-scratch/rollhist/read/tmp`.
* Stop if F: drops below 6 GB free.
* Never kill a process you did not start.
* Never touch `F:/TanitAD-scratch/rollhist/seed3/pod_s3/` (another agent works there).
* Never wait more than about 4 minutes in one call. Hand off at about 150k context by writing `HANDOFF_READ_S3.md`.
* Delete your own temporary files by literal path when done, and log each one in `F:/TanitAD-scratch/rollhist/seed3/DELETIONS.log`.

## Deliverables
In `F:/TanitAD-scratch/rollhist/read/`:
* the edited and new files, with `CHANGED_FILES_READ_S3.txt` (path, sha256, backup path);
* `FACTS_READ_S3.md`: the pairs, the floor file, the selftests for KI, RK and RA, and the planted name check;
* `HANDOFF_READ_S3.md`.

Hand back in at most 200 words, saying plainly what passed and what did not run.
