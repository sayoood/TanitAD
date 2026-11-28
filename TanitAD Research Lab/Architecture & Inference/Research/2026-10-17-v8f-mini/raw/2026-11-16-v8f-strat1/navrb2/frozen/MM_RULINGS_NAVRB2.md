# MM rulings on PREREG_V8F_NAVRB2 (draft sha256 a12642af; 2026-11-17 ~22:20Z)

PI go: "add the channel for the roadabout" (2026-11-17). The base form is 4 arms.

1. **Code base `4dedac45` (agent/v8f-strat1-s1d-20261117), not the arch-inf tip.** Accepted.
   - The arms are paired with the banked N1 / S1 at s0 and s1, so they must train on the code those arms trained on.
   - The arch-inf tip has no `strat1_check.py` or `build_navtext_nav.py`, and its model, data and trainer files differ from every STRAT1 training commit.
   - The difference between `4dedac45` and `22999538` (the s0/s1 training commit) is two gate probes, not training math. The card states this, with the `git diff --stat`.
2. **`--nav-rb-lr-mult 10`.** Accepted. It is the programme's convention for a fresh parameter in a fine-tune (`--ctx-lr-mult 10`, `--s-far-lr-mult 10`). The card's RB-UNREAD branch keeps 30 as the named next dose, with a PI go on cost.
3. **Cost.** The 4 base arms (N1R, S1R at s0 / s1, about 0.63 L40-h, ESTIMATED) are inside the PI's go. Each conditional arm goes to the PI with its cost, and only when its pre-registered trigger fires: N1R / S1R at s10 / s11 under COST4, S1dR and N1Z.
4. **Freeze.** The continuation folds rulings 1-3 into the card and writes its frozen sha256. The MM lands the card before any arm is minted.
5. **Pod order.** CKPT120 fix3, then C0REP, then DSCALE block 3, then these arms, all under `flock /root/gpu_queue.lock`. The queue script waits for block 3's DONE / FAIL line.
