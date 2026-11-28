# Brief: build the ROLLHIST read package (dev-box CPU; run only control c-EXEC; do NOT start the read chain)

You are a TanitAD build agent working for the Master Mind (MM). Read `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip first:
`git -C C:/Users/Admin/tanitad-push show $(git -C C:/Users/Admin/tanitad-push ls-remote origin agent/arch-inf-20260803 | cut -f1):"Project Steering/AGENT_OPERATING_STANDARD.md"`.

Rules:
* Stage nothing, commit nothing.
* No secrets. Never write an IP, port or pod id.
* HF token, if you need it: read it in place, `HF_TOKEN="$(grep -oE 'hf_[A-Za-z0-9]+' E:/Projects/TanitAD/Keys.txt | head -1)"`. Never print it. Set `HF_HUB_DISABLE_XET=1`.

## What is decided (read; do not redesign)
* Card `F:/TanitAD-scratch/rollhist/PREREG_V8F_ROLLHIST.md` (sha256 781eea87, FROZEN). Read section 2 (arms, contrasts, floors) and section 3 (runs 3.1, cells 3.2, reads 3.3, outcomes 3.4 with precedence). The reader prints exactly one committed name, or the same name prefixed PROVISIONAL on one seed.
* Amendment 1, `PREREG_V8F_ROLLHIST_AMENDMENT_1.md` (87928487): the code of record is **43319227** (= cdbff971 + lever C + `stack/scripts/v8f_t1_eval_hb.py`, sha256 74fd8445). Export that tree for the rollouts (`git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive 43319227 stack taniteval`) and blob-verify it.
* Checkpoints, all arriving after the pod chain: RH_s0, RH_s1, B0_s0, B0_s1 at `/root/rollhist/runs/<arm>/train/ckpt_last.pt`. Pull them later with the pod's own `sha256sum`. Do NOT pull them now. They do not exist yet.
* Tier of record: `python stack/scripts/v8f_t1_eval_hb.py --kappa-max 0.2 --inner <readout wrapper> <argv>`. Six forms per checkpoint (3.1): R0Hb20, R0C10Hb20, R0G50Hb20, R0G0Hb20, R0KHb20, R0C10XHb20. That makes 24 rollouts.
* The STRATEGIC family is in full (PI 2026-11-16). Fog windows are unscored.

## Templates (copy their mechanics)
* The chain shape: N1DC eval-141 at `F:/TanitAD-scratch/n1dc/read/eval141/`. Read `HANDOFF_EVAL141.md`, `run_n1dc141.sh`, `.cmd` and `dry_check.py`. It provides:
  * tree export plus blobcheck;
  * resumable steps with `out/steps/*.ok`;
  * CPU, 2 threads, one process at a time, each step gated on commit >= 5.5 GB;
  * a WMI launch, and a last log line of `ZZ ... DONE|FAIL ZZ`, written with `>> "$LOG"`, never `| tee`.
* The forms, family reader and strategic reader come from the coordinator's integ10k: `F:/TanitAD-scratch/v8proof/integ10k/`. Read `ACCGAIN10K_DEFS.md`, `STRATROLL10K_DEFS.md`, `read_accgain10k.py`, `strat_roll_read.py`, the `v8f_t1_eval_reader_rule_rzg*.py` wrappers, and `chain_accgain10k.sh` for how a form renders to argv.
  * Use them read only, copied into your folder with their sha256.
  * Do not edit anything in integ10k. It belongs to another session.
* The G-T0 guard is a policy-rolled T0 ADE6 by `v8f_dscale_read.py`. Find where it lives and how DSCALE10K ran it (`F:/TanitAD-scratch/datascale/`).

## Do
1. **The stage-A references**, found, not re-run:
   * Card ADD67 banked R0Hb20 and R0KHb20 on UE.A. Card ADD68 (`raw/2026-11-12-v8f-proof/PREREG_V8F_PROOF_ADD68.md` at the mirror tip) runs R0C10Hb20 and R0C10XHb20 at both UE.A seeds.
   * Locate those dumps (dev box, F:, HF) and say for each one whether it exists, where, and under which executor.
   * If any is missing, list it as a reference the chain must add, with its cost. The one-seed floor F_U,c and P2's reference need them.
2. **The package**, in `F:/TanitAD-scratch/rollhist/read/`:
   * the tree plus blobcheck;
   * `rh_read_chain.sh` / `.cmd`, which pulls and checks the 4 checkpoints, runs the 24 rollouts, then STRATEGIC, then G-T0, then the reader;
   * `rh_read.py`, which implements 3.2 to 3.4 exactly: cells per family, never pooled; the cell and family verdict rule; floors F_c, two-seed and one-seed; P1 to P4 with their attribution controls; precedence; and `RESULT_ROLLHIST.json`. Use the paired episode-cluster bootstrap, B 2,000, seed 0, `taniteval/ci.py`.
3. **Controls, each named, with the value it must read:**
   * **c-EXEC (RUN IT NOW; it uses only UE_s0.A):**
     * UE_s0.A under the ADD68 study wrapper `rzgfh` (sha256 373ffe72f2655c35...);
     * versus the tier-of-record form `v8f_t1_eval_hb.py --kappa-max 0.2 --inner v8f_t1_eval_reader_rule_rzg`;
     * on one form (R0C10XHb20), all 141 eval files;
     * pass = 0 differing arrays.
     * UE_s0.A is on HF `Sayood/tanitad-v8f-dscale10k-runs`, `UE_s0/A/ckpt_last.pt`, sha256 2fd4b747.... Use a local copy if one exists. Check by sha.
   * **K-REPLAY:** one chunk re-run reads bit-identical (inference is deterministic).
   * **K-SELF:** a self-pair reads exactly 0 on every cell.
   * **K-PLANT:** a planted offset in one arm's dump moves exactly the cells it should. T1 readers prepend the origin, so build the planted track from its own start point.
   * **Mutation:** feed the reader the other seed's arm, and it must go RED.
4. **A dry check:** render all 24 argvs with UE_s0.A standing in for each arm, run imports, `bash -n`, and sha256. No rollout beyond c-EXEC.
5. **Cost:** minutes per rollout from c-EXEC's own wall time, times 24 plus the references and readers. This is ESTIMATED.

## Dev-box rules
* CPU only. The dev-box GPU is in use by another session; never force `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock`.
* One heavy process at a time. Gate each on commit >= 5.5 GB (`F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60`).
* Free space now: C: 9.5 GB, E: 5.8 GB, F: 14.3 GB. Stop if F: falls below 6 GB. Keep TMP/TEMP under your folder.
* Python: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe` (never python3). Windows PYTHONPATH uses `;` (cygpath -w). Assert `tanitad.__file__` is inside your tree.
* Never wait more than about 4 minutes in one call. For a longer job, launch it detached (WMI, CreateFlags 0x08000200), write HANDOFF, and end.
* Delete your own temp by literal path, logged in `F:/TanitAD-scratch/rollhist/DELETIONS.log`.

## Deliverables in `F:/TanitAD-scratch/rollhist/read/`
* the scripts;
* `C_EXEC.json`;
* `REFS_FOUND.json`;
* `DRY_CHECK.json`;
* `HANDOFF_READ.md` (at most 1k tokens, with the launch line);
* `MANIFEST_SHA256.txt`.

Hand back in at most 200 words.
