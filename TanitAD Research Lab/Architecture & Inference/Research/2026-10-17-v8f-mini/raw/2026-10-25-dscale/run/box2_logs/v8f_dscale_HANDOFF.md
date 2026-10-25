# PREREG_V8F_DSCALE handoff (written 2026-10-25 ~01:35) -- ARMED AND RUNNING

* PI go (2026-10-25, verbatim): "Go: run PREREG_V8F_DSCALE seed 0 on box 2 once its instruments land." (register row PI-V8F-2026-10-25)
* PREREG: `Project Steering/PREREG_V8F_DSCALE.md` + Amendment 1. Instruments landed at 2d85add1 (agent/arch-inf-20260803).
* Chain: `box2_ship/chain_v8f_dscale.sh` (== the tip's), wrapper `run_chain_v8f_dscale.cmd`, log `logs/chain_v8f_dscale.out`, launched via WMI 01:25.
  Inputs OK, gate PASS 01:31 (2,256 of 4,512 clips, 153,271 windows, 19,164 steps, params 7,214,495 = B's). Training launched 01:31; ESTIMATED ~4.4 h.
* Guard: `logs/v8fdscale_guard.out`. Waker: `box2_ship/wake_v8fdscale.sh s0` (exits on logs/v8f_dscale/dscale_read.json or STOP/DONE).
* On DONE:
  1. Report the outcome as committed (D-T: DATA-LIMITED / NOT DATA-LIMITED / EPOCH-LIMITED) with the four families beside, tier T1-class open loop, paired episode-cluster bootstrap. ADE6 at one seed decides nothing.
  2. Bank `RESULT_V8F_DSCALE.md` + `raw/2026-10-25-dscale/run/` (template: grid_a2/mk_hold24.py). Update register H-V8F-DSCALE-1 and H-DATA-SCALE-1. Copy the checkpoint to E:/Projects/TanitAD-artifacts/v8f_dscale/ (E: has ~2 GB free: check first).
  3. Follow the decision table: DATA-LIMITED -> D3 PREREG (fix the three traps in `raw/2026-10-25-d3-scoping/D3_SCOPING.md` first); NOT DATA-LIMITED -> PREREG_V8F_LORA first (instruments being built in scratchpad/lora/); EPOCH-LIMITED -> B at 2x steps.
  4. Ask the PI for the next box-2 go together with the verdict.

## CLOSED 2026-10-25 04:30
* Trained 01:31-04:18, evaluated to 04:22, read 04:23 (`v8f_dscale/dscale_read.json`): NOT DATA-LIMITED as committed (r +6.25 %, MSE delta ns). ZZV8FMINI-DONE. Guard max 45 C.
* Banked as RESULT_V8F_DSCALE.md + raw/2026-10-25-dscale/run/; checkpoint on E:/Projects/TanitAD-artifacts/v8f_dscale/M2b_s0_df50/ (sha256 242c92cb...).
* Next (committed): PREREG_V8F_LORA first. Its AR-2 chain landed at 3b4ba0c1 and needs the PI's box-2 go.
