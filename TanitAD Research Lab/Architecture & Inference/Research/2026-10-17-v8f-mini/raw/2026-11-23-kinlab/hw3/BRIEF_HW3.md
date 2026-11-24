# Brief: run the three-seed headway read of the adopted RH recipe (card PREREG_V8F_RHHW3, FROZEN sha256 3c85a42b). Dev-box CPU only, zero training

You are a TanitAD build agent working for the Master Mind (MM). First read:
* `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip (`git -C C:/Users/Admin/tanitad-push show origin/agent/arch-inf-20260803:"Project Steering/AGENT_OPERATING_STANDARD.md"`);
* the card `F:/TanitAD-scratch/rollhist/hw3/PREREG_V8F_RHHW3.md`. It binds every choice below; if this brief and the card disagree, the card wins;
* the frozen headway reader `F:/TanitAD-scratch/rollhist/read/code/headway/headway_read_r2.py`, and how the chains call it: `F:/TanitAD-scratch/rollhist/read/rh_read_s3_chain.sh` line ~91 (`read --t1-root … --pairs … --seeds … --out-dir …`), and the older two-seed RH chain in the same folder.

## Rules
* Stage, commit and push nothing.
* No GPU, nothing on the pod, no rollouts. Every dump already exists:
  * s0 and s1 under `F:/TanitAD-scratch/rollhist/read/out/t1/`;
  * s2 in the SEED3 read's t1 root (find it from `rh_read_s3_chain.sh`).
* **Do not edit the frozen reader** or any other file in `F:/TanitAD-scratch/rollhist/read/code/`. New code goes in `F:/TanitAD-scratch/rollhist/hw3/`.
* If you need one t1 root with all three seeds, build it under `F:/TanitAD-scratch/rollhist/hw3/t1/` from file copies. Check each copy's sha256 against its source, and delete the copies at the end by literal path, logging each one in `F:/TanitAD-scratch/rollhist/hw3/DELETIONS.log`.
* CPU only, `CUDA_VISIBLE_DEVICES=-1`, the venv python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe` (the bare `python` lacks numpy). One python process at a time behind `F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60`.
* Nothing on C:. TMP and TEMP go under `F:/TanitAD-scratch/rollhist/hw3/tmp`. Stop if F: drops below 6 GB free.
* **Permission refusals:** do not retry in another form and do not ask anyone else to run it. Record it and hand back.
* Never wait more than about 4 minutes in one call. Hand off at about 150k context by writing `HANDOFF_HW3.md`.

## What to do, in order
1. **Control R0 first.** The pooling path must reproduce, exactly (D and interval), the frozen two-seed RH read's U value for `I10RH_R0KHb20_minus_I10B0_R0KHb20` and `I10RH_R0C10XHb20_minus_I10B0_R0C10XHb20` on seeds 0 and 1. The reference is in the two-seed RH read's headway output under `F:/TanitAD-scratch/rollhist/read/out/read/`.
   * Use `--seeds 0,1,2` of the frozen reader only if it takes three seeds unchanged. Otherwise run it per seed and pool in a new tool that implements the reader's own two-seed pooling.
   * Record which path you used, and why.
   * **Print no three-seed pooled number until R0 passes.** If R0 fails, stop: the name is HW3-BLOCKED. Hand back with the difference.
2. **The read** (card section 3):
   * U for RH − B0 at seeds 0, 1 and 2, pooled, under K (decides) and C\* (beside);
   * C and F beside;
   * the per-seed points;
   * the floor F3 per form, from the arms' U levels, exactly as the card defines it.
3. **The name, `hw3_name.py`:** card section 4.
   * Precedence: BLOCKED, then COST-REAL, then COST-BELOW-FLOOR, then NOT-SEPARATED.
   * A planted check drives it through all four names, including a known +0.03 cost (COST-REAL when its F3 is small) and a 0.0 control (NOT-SEPARATED).
   * Output `F:/TanitAD-scratch/rollhist/hw3/RESULT_HW3.json` with every input value, R0's values, F3, and the name.
4. **`FACTS_HW3.md`:** R0 (reference against yours), the path used, the per-seed and pooled values, F3, the planted check, and the files with their sha256.

Hand back in at most 200 words: R0, the name, the decisive numbers (pooled D, interval, F3, per-seed points under K), and what did not run.
