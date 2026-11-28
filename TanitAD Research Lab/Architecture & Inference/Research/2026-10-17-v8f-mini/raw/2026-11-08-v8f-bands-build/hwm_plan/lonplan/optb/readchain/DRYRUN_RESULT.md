# Read-chain dry run: result (MM, 2026-11-18 ~03:05Z)

Evidence class MEASURED on the dev box (CPU, 2 threads, under the load of other jobs). Nothing touched box 2.

## Dry run B (tampered bank): PASS
* One byte flipped after the bank manifest was written.
* The chain refused at the evidence stage, rc 3, and wrote the sentinel to its own file, in 4 s. (This is the agent's run, `tmp/dryB`.)

## Dry run A (an R35_s0 copy as `FAKEa_s0`, then the unbanked `FAKEc_s0` with a box-2 STOP present): PASS, `chain rc=0`

The MM fixed two defects before it passed:
1. **`optb_readchain.sh` waited for a `DONE` that `lonplan_run.py` never writes.**
   - The chain of record writes it (`lonplan/score_chain.py:64`), so the first run stopped `plan-nodone`.
   - Fix: after rc 0, the chain checks that every seed's `plans_hwm_oracle_s<seed>.npz` exists, then writes `DONE`.
2. **The fake bank had no `argv.txt`, which a real bank carries** (`stages/optb_run.sh:59`). The BAND-MET read (`lr_lib.load_arm`) needs it.
   - Fix: `mk_fakebank.py` copies the source run's `argv.txt`.
   - This was a dry-run defect only, not a chain defect.

After the fixes, every stage for `FAKEa_s0` wrote its output:
* **evidence:** sha256 against `RUN.done`.
* **planner:** `--n 8`, seed 0. `DONE` reads "ok seeds 0".
* **IA-3P-K and the beside read:** 12,827 anchors, held-out R² 0.2429, the C10' CI above 0. This reproduces the read side's step-0 numbers on R35.
* **BAND-MET:** `band_met_lost` False. This is correct, since R35 is read against itself.
* **`mech.json` and `manifest_decide.json`:** refreshed.
* **The box-2 STOP:** it ended the chain before the unbanked tag, with `ZZ_ENDED_BY_BOX2_STOP.txt`.

## Timing (MEASURED)
* **Planner, full windows, one inference seed, R35_s0 through the route:** 1,347 s (`tmp/timing/timing.txt`).
* **Per checkpoint:** about 45 min for the two seeds, plus IA-3P-K (16 s) and BAND-MET (297 s), so **about 50 min**.
* **Ten arms:** about 8.5 h of dev-box CPU. The card's ESTIMATE was about 2.2 h per checkpoint.
