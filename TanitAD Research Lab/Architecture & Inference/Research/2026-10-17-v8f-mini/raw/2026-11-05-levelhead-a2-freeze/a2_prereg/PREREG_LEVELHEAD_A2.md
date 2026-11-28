# Addendum A2 to PREREG_LEVELHEAD_CL: arms CLR and LLR, the vision heads inside the recovery recipe (written 2026-11-05 after the
# A1 read-out, before any CLR or LLR run)

**Why.** CL (C0's recipe + the level head) did not move the crawl, and A1 measured why: O ignores the level it is handed (a +3 m/s
shift moves the plan's mean speed by about 0.0005 m/s; e_level is 0.4 to 0.5 % of the ego token), and the head's band error is 1.8 to
1.9 m/s (PARTIAL). A1's committed branch is the lead-aware head. A second head wired the same way in the same recipe would face the
same cause: in C0's recipe the logged history carries the speed exactly, so no vision input is ever needed. This card therefore puts
the heads inside RAPR-P01's recovery recipe (one deviation per window, a-priori law, p 0.1, rejoin target), where the history is
wrong and the target is the human's, so vision is the informative input. PI go: "Do c then a" (2026-11-05). Claim row H-LEVELHEAD-2.

## Arms (box 2, one at a time, CLR first)
* **CLR_s{0,1}** = RAPR-P01's argv (`v8f-recover/M2b_s{s}_recft_RAPRP01/argv.json`) + `--level-head on --w-level 0.2027` (CL's frozen
  W), `--out ...\M2b_s{s}_recft_CLR`. No new code (6ece77f). The level share of the loss at step 0 is recorded, gated at [0.08, 0.12].
* **LLR_s{0,1}** = CLR's argv + the lead head (new code, its own build and gate): from the current tick frame's adapted tokens, a
  presence logit (lead within 40 m) and a gap estimate (gap / 40, clipped to [0, 1]), BCE + Huber on the frames with a lead label
  (`--lead-train` labels the trainer already loads), weight W_lead fixed by the same formula as W (its term a tenth of the total at
  step 0, own probe). O reads [sigmoid(logit), gap_hat] detached. **How O reads both heads is fixed by CLR's sensitivity (below):**
  CLR O-SENSITIVE ⇒ LLR adds e_lead to the ego token like e_level (zero-init); CLR O-INSENSITIVE ⇒ LLR feeds BOTH heads' outputs as one
  separate conditioning token of O (`--aux-cond token`, non-zero init), the wiring O cannot average away.

## Reads (T1-kin on the RI tool, paired episode-cluster bootstrap B = 2000, seed 0; deciding scene `reindexed`)
Per arm X and seed: X, XC (kbox) on `reindexed` and `logged`, FZX on `hold_now`; bases C0 (P, G, D) and RAPR-P01 (attribution),
which needs RIRAPRP01 rollouts (RAPR-P01 under `reindexed`). Rules P, G, F, D and LEAK-BORNE exactly as the CL card's section 4 with
CL := X. Beside, deciding the wiring: **O-sensitivity of CLR** by A1's measurement (O-SENSITIVE >= 0.5 m/s mean |plan-speed change|
for a +3 m/s shift; O-INSENSITIVE < 0.1). Attribution, reported: CLR − RAPRP01 and LLR − CLR on M1 (reindexed).

## Outcomes, committed now
* D holds for an arm at both seeds: counted DOMINANT. Signal the PI for scale, with the video evidence.
* CLR P fails and CLR − RAPRP01 on M1 not separated-better: the level adds nothing even when needed; LLR is the next arm (wiring per
  the sensitivity rule).
* LLR P fails: the vision-head family on frozen single-frame features is exhausted at these forms; the remaining levers are the named
  blockers of option (c) (T2 re-perception, the A40), reported to the PI with the numbers.
* P holds, G or D fails: the arm's dose (W / 2) under the same card.
