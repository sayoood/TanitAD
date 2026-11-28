# L2 SAFE-HEAD post-verdict diagnostic: why NULL-HELDOUT (MM, 2026-11-27, DESCRIPTIVE)

Not a decision and not a card. Written AFTER the card's outcome (NULL-HELDOUT, both seeds chose step 0), to pick the next lever.
Zero training: the three gated heads (`run1/sh_{s0,s1,red}/head_step02000.npz`, PASS tokens at 63a4af88) applied to the navtrain
cache's `s`. Script `veto_diag.py`, output `veto_diag.json`. MEASURED, T0 (navtrain cache, NAVSIM v1.1 PDM labels, LQR-tracked).

| split (n tokens) | rule | safety misses | PDMS | EP | switch vs C1 |
|---|---|---|---|---|---|
| train (1,847; SH was FITTED here) | C1 | 69 (3.74 %) | 93.513 | 89.24 | 0 |
| | SH_s0 / SH_s1 | **41 / 43** | 92.627 / 92.525 | 85.25 / 85.14 | 0.84 |
| | best VETO (s1, tau 0.5) | 57 | 93.831 | 89.20 | 0.04 |
| held-out (425, 6 logs) | C1 | 18 (4.24 %) | 93.360 | 90.37 | 0 |
| | SH_s0 / SH_s1 | 20 / 21 | 91.276 / 91.081 | 85.56 / 85.32 | 0.87 |
| | VETO s0 tau 0.5 | 19 | 93.321 | 90.42 | 0.04 |

Control: SH_s0 on held-out reproduces `heldout_select.json` step 2000 (20 misses, EP 85.56); the numpy agg32 pick equals C1's stored pick on 100 %.
Control: SH_red (permuted labels) is the worst rule on both splits.

**Reading.**
1. On its fit split the head DOES learn the tracked safety boundary: misses fall ~40 % (69 -> 41) in both seeds. The signal is in `s`.
2. It does not transfer to the 6 held-out logs (18 -> 20/21). With 1,847 tokens in 29 logs, the effective sample is the LOG, and 29 logs do not pin down a boundary that generalises.
3. Every rule that moves picks away from C1 pays in EP (-2 to -4 pts). The agg32 argmax sits among near-tied proposals, so re-scaled safety probabilities reshuffle it.
4. Hence capacity (SH-MLP, card 8.1) is not the measured bottleneck; DATA is. SH-MLP stays the committed next arm. Its expected failure mode is a wider train/held-out gap, and it should be read against this table.

Caveats: train numbers are optimistic (SH fitted there); the held-out rule resolves ~+/-2 misses (18 C1 misses in total); tau was not fitted, so these rows are a grid and not a selection.
