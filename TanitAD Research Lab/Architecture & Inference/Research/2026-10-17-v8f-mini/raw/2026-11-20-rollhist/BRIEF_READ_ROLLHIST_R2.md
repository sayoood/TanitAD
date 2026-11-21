# Brief: ROLLHIST read package R2. Implement Amendment 2 in the reader and launch the 7 stage-A reference rollouts now (dev-box CPU)

You are a TanitAD build agent working for the Master Mind (MM). Read `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip first:
`git -C C:/Users/Admin/tanitad-push show $(git -C C:/Users/Admin/tanitad-push ls-remote origin agent/arch-inf-20260803 | cut -f1):"Project Steering/AGENT_OPERATING_STANDARD.md"`.

Rules:
* Stage nothing, commit nothing.
* No secrets. Never write an IP, port or pod id.

## Read first
* `F:/TanitAD-scratch/rollhist/PREREG_V8F_ROLLHIST_AMENDMENT_2.md` (sha256 15004dc9, FROZEN 01:29Z). It is binding. Do not reinterpret it; if a sentence cannot be implemented as written, stop and say which.
* The package built in R1: `F:/TanitAD-scratch/rollhist/read/`. Read `HANDOFF_READ.md`, `rh_read.py`, `rh_read_chain.sh`, `rh_roll.py`, `rh_common.py`, `REFS_FOUND.json` and `C_EXEC.json`. c-EXEC's UE_s0.A × R0C10XHb20 dumps are the ones under the tier-of-record form; locate them.

## Do
1. **The reader, `rh_read.py`:** implement Amendment 2 items 2, 3, 4 and 6. Keep the R1 file as `rh_read_r1.py`, and record its sha256.
   * Add one selftest per new rule, each with a deliberate-regression arm that must go RED:
     * **G1** reads ROLLHIST-PARTIAL;
     * **G2** reads SEED-DISAGREE;
     * a **straddling cell that flips the name** reads SEED-DISAGREE and prints both names;
     * a **straddling cell that does not flip it** keeps the first name;
     * the **P2 attribution control failing** fails P2.
   * Every name from a one-seed read carries `PROVISIONAL`.
   * The existing selftests must still pass.
2. **References (item 5):** a resumable chain `rh_refs_chain.sh` / `.cmd` running the 7 rollouts:
   * UE_s0.A and UE_s1.A, each × R0G50Hb20, R0G0Hb20, R0C10Hb20;
   * UE_s1.A × R0C10XHb20.

   Use the form of record (`v8f_t1_eval_hb.py --kappa-max 0.2 --inner v8f_t1_eval_reader_rule_rzgi`, tree `read/tree` = 43319227), the same argv rendering as `rh_roll.py`, CPU, 2 threads, one rollout at a time, and each one gated on commit >= 5.5 GB.
   * UE_s1.A is on HF `Sayood/tanitad-v8f-dscale10k-runs`, `UE_s1/A/ckpt_last.pt`, sha256 `f13df8ce...`; check the full sha in the records.
   * Read the token in place: `HF_TOKEN="$(grep -oE 'hf_[A-Za-z0-9]+' E:/Projects/TanitAD/Keys.txt | head -1)"`, with `HF_HUB_DISABLE_XET=1`. Never print it.
   * The last log line is `ZZ ROLLHIST-REFS DONE ZZ` or `ZZ ROLLHIST-REFS FAIL ZZ <why>`, written with `>> "$LOG"`, never `| tee`.
   * **Launch it detached NOW**, by WMI with CreateFlags 0x08000200 (the launch form is in HANDOFF_READ.md). Check that it started from its log, write the handoff, and end. Do not wait for it.
3. **Wiring:** make `rh_read_chain.sh` / `rh_read.py` read F_U,c and the UE.A P2 reference from these dumps (ADD68's JSON printed beside). Also have the chain refuse to start unless `ZZ ROLLHIST-REFS DONE ZZ` is present, or `RH_REFS_SKIP=1` is set.
4. **Dry check:** re-run `dry_check.py` and the reader selftests. Update `DRY_CHECK.json` and `MANIFEST_SHA256.txt`.

## Dev-box rules
* CPU only. Never touch the dev-box GPU or `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock`.
* One heavy process at a time. Gate each run on commit >= 5.5 GB.
* Stop if F: falls below 6 GB. Keep TMP/TEMP under your folder.
* Python is `C:/Users/Admin/venvs/tanitad/Scripts/python.exe` (never python3). Windows PYTHONPATH uses `;`.
* Never wait more than about 4 minutes in one call.
* Delete your own temp by literal path, logged in `F:/TanitAD-scratch/rollhist/DELETIONS.log`.

## Deliverables in `F:/TanitAD-scratch/rollhist/read/`
* `rh_read.py` (R2) and `rh_read_r1.py`;
* the refs chain;
* `SELFTEST_R2.json`;
* `HANDOFF_READ_R2.md` (at most 800 tokens, with the refs chain's state and the main chain's launch line);
* the updated manifest.

Hand back in at most 200 words.
