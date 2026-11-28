# Brief: ROLLHIST read package R3. Add the one-seed STRATEGIC floor, and run the reference readers on the real UE.A dumps (dev-box CPU; no rollout)

You are a TanitAD build agent working for the Master Mind (MM). Read `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip first:
`git -C C:/Users/Admin/tanitad-push show $(git -C C:/Users/Admin/tanitad-push ls-remote origin agent/arch-inf-20260803 | cut -f1):"Project Steering/AGENT_OPERATING_STANDARD.md"`.

Rules:
* Stage nothing, commit nothing.
* No secrets. Never write an IP, port or pod id.

## State
* Package: `F:/TanitAD-scratch/rollhist/read/`. Read `HANDOFF_READ_R2.md` and `MM_CONFIRM_R2.md` (item 6 is this task).
* Amendment 2: `F:/TanitAD-scratch/rollhist/PREREG_V8F_ROLLHIST_AMENDMENT_2.md` (15004dc9), items 5 and 6. It is binding.
* The refs chain is DONE (`out/rh_refs_chain.log` ends `ZZ ROLLHIST-REFS DONE ZZ`).
  * The 7 UE.A reference dumps are in `out/t1/I10UEA_*`.
  * UE_s0.A × R0C10XHb20 is c-EXEC's tier-of-record dump. Locate it from `C_EXEC.json`; `rh_read_chain.sh` adopts it as `out/t1/I10UEA_R0C10XHb20_s0`.
* `rh_read.py` `decide(..., strat_floor=None)`: in a one-seed read, STRATEGIC prints NOT-DECIDABLE because no floor file is wired.

## Do
1. **`rh_strat_floor.py`:**
   * Inputs: the UE_s0.A and UE_s1.A R0C10XHb20 dumps.
   * Method: the frozen ADD66 functions (the same imports and cell computation as `rh_strat.py`, which must not change).
   * Output: per STRATEGIC cell of card 3.2 (ADD66 `pos` at 30, 50 and 80 m at the third tick, plus the per-family-reported cells), the floor F_U,c = abs(UE_s1.A - UE_s0.A), in exactly the key format `decide()` reads.
   * Output file: `out/read/STRAT_FLOOR_UEA.json`, with input sha256s and evidence class MEASURED.
   * Controls, each with its required value:
     * **K-SELF:** s0 against itself gives floor 0.0 exactly on every cell;
     * **K-PLANT:** a planted known offset in one copy moves the floor by that amount on the cells it touches, and only those;
     * **mutation:** feeding the B0/RH run names instead of UE.A must be refused.
2. **Wiring:**
   * Add a `strat_floor` step to `rh_read_chain.sh`, before `read`.
   * Add `--strat-floor` to `rh_read.py`, passed into `decide`.
   * Add a selftest: a one-seed read with the floor prints a STRATEGIC verdict, not NOT-DECIDABLE. Its regression arm, without the floor, must print NOT-DECIDABLE.
3. **Run the reference readers on the real UE.A dumps now** (headway and family readers into `out/read/refs_hw` and `refs_q3`, the chain's own commands):
   * Report whether they run.
   * Report the P2 reference on UE.A (D1, D2, ACC per family). The card expects WORSE. This is a reference that decides nothing; report it as MEASURED with the reader's paths.
   * Also produce `STRAT_FLOOR_UEA.json` for real.
4. **Update:** re-run `dry_check.py` and the selftests, then update `DRY_CHECK.json` and `MANIFEST_SHA256.txt`.

## Dev-box rules
* CPU only, one process at a time. Gate each process on commit >= 5.5 GB (`F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60`).
  * Another session runs jobs on this box. If one of yours dies on a MemoryError, wait for commit and re-run it; do not raise thread counts.
* Stop if F: falls below 6 GB. Keep TMP/TEMP under your folder.
* Python: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`. Never wait more than about 4 minutes in one call.

## Deliverables
* `rh_strat_floor.py`;
* `STRAT_FLOOR_UEA.json`;
* `REFS_READ_UEA.json` (the reader outputs' paths and the P2 reference);
* `SELFTEST_R3.json`;
* `HANDOFF_READ_R3.md` (at most 600 tokens);
* the updated manifest.

Hand back in at most 200 words.
