# MM rulings on AMEND_OPTB (2026-11-17 ~21:10Z)

The rulings concern `AMEND_OPTB.md` (sha256 7cb0245c), section 6. PI go for ACTRSP Option B: "4: yes" (2026-11-17).

1. **Learning rate.** The recipe's peak 3e-4 stands. A lower peak would be a second lever and would need its own arm, not a silent change.
2. **Seeds and order.** As written in section 2.4. Chassis 0 completes first, so a chassis-0 failure stops the chain before chassis 1 spends box-2 time.
3. **FT-NULL also freezes `wm.proj`.** Accepted. The RSP and NULL arms then differ only in the response term, which is the attribution the read needs.
4. **Land base.** A new branch `agent/v8f-optb-20261117` from `a13f6f16` (ACTRSP branch), because the tip trainer has no `--wm-only`. The MM commits by plumbing with the revert guard.
5. **Inputs to box 2.** Ship the two R35 checkpoints and the two probe files, with sha256 checked on arrival. The read side (IA-3P-K, C10') is built before the first arm's read, not before launch.
6. **The `HJEPAStack.built_rollout` change** (reads on a float64 copy) is accepted only under two conditions:
   - a caller census shows it is reached only by gate / G-DVB readers, never by the training or eval forward;
   - `test_v8f_ladder` and the 15-file regression set pass with 0 regressions against `a13f6f16`.
   If the method sits on the forward path, the float64 read moves into the gate's reader instead, and the method stays unchanged.

**Before launch, still owed:** the regression set, the launch-gate-reading tests, the MR11-MR14 mutation run, the box-2 package (argvs, checker, per-arm tokens, ship script, waiter), and the bank of the build.
