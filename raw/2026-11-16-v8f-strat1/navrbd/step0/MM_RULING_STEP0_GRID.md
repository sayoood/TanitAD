# MM ruling: the held-out grid for D1-HO is the T1-feasible subset (2026-11-18, before any D1-HO number exists)

State when written:
* No held-out dump has finished. The two step-0 dumps were stopped at 12:42Z for UM_s0 (PI go), and their partial output (`out/t1`) is deleted.
* No D1-HO or F_HO number exists.

1. **D1-HO and F_HO are read on the T1-feasible held-out grid: 5,736 windows / 497 episodes** (`eval_grid_x_feasible.json`, sha256 ca289d0f).
   * The pinned T1 tool refuses a window whose 6 s future is incomplete (`gt_ok` False). That drops 1,858 of the 7,594 grid windows.
   * This is the same completeness rule every T1-kin cell on eval-141 already applies. It does not select on the label or on any arm's output.
   * 497 episodes is still 31 times eval-141's 16 roundabout episodes.
2. **The card draft is amended before freeze to state this grid**, its sha256 and the 1,858 dropped windows with the reason. The full 7,594 / 545 counts stay in the card as the pool size.
3. **Order of operations is unchanged.** F_HO comes from the banked N1R pair and is written into the card before any N1D checkpoint exists.
4. **When step 0 resumes:** after UM_s0's chain ends, never beside a training job (CPU dumps add RAM and threads). Both N1R dumps run in one environment.
