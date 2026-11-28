# A3_RULING_1: three readings of PREREG_LEVELFLOOR_A3.md (977a785e…), fixed before any LF rollout (MM, 2026-11-05 21:4x)

1. **The bare arm's flags.** The card's section 2 says the bare arm runs `--smooth off --feasible off`, with F on O's raw plan. The
   banked bases it is read against do not use those flags. RIC0_s{0,1} and C0_s{0,1} ran `--smooth sg72` with no `--feasible`: their
   done markers (`F:/TanitAD-scratch/motion_c/ri/out/done_RIC0_s0.txt`: `"flags": "--smooth sg72 --scene reindexed"`) and
   `motion_c/ri/ri_c0_chain.sh` lines 83 to 86 show this. The card's design statement is that "LF − C0 isolates the floor", and with
   `--smooth off` the contrast would also carry S. **Ruling:** the bare arm runs the base's own flags, `--smooth sg72`, with no
   `--feasible`, and F is applied after S: O's plan → S → F. The composite is unchanged: S → F → kbox. In every LF contrast the only
   difference from its base is F.
2. **The M1 tick of the reach control and the instruments.** These use the reader's own M1 tick definition, the pair thresholds of
   the M1 row. The card's crawl band is reported beside them. The 3 m/s band of the floor operator (step 1 of F) is a constant of F
   and is unchanged.
3. **"M1 moved" in section 5:** RILF − RIC0 on M1 is separated-better at one or more seeds.

No LF rollout, smoke or known answer with the floor on had run when this was written. The sha256 is recorded in
`a3_prereg/FROZEN_SHA.txt`.
