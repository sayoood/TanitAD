# Note on the per-tick window length (written after the read, 2026-11-05)

Addendum 4 and the code commit `eccdabd` describe the per-tick shape over "rows j − 9 ... j" (10 rows). The shape as built and run
spans the batch's whole history window, F = 16 frames:

* RPT's launch-gate selfcheck reads `levers.recover_table.per_tick.ticks` = 16 at both seeds (`runs/M2b_s{0,1}_recft_RPT/gate_selfcheck.json`).
* The library refuses any other count: `RecoverPerturb` raises when the table's tick count differs from `batch["p10f"].shape[1]`
  (`stack/tanitad/data/v8f_recover.py:326` at `eccdabd`), so a run cannot train on a shorter window.
* R's constant shape offsets the same 16 history frames (`_rebuilt`, the same `F`), so RPT and R differ in the shape of the deviation
  over one and the same window, as the addendum intends.

The prose count is wrong; the arm, its gate and its read are the ones the addendum commits. The read result and row K2 are unchanged.
