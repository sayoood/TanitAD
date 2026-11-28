# PROPOSED outcome rules for the S1d read (builder draft 2026-11-17; the MM freezes them BEFORE any S1d number exists)

The card (PREREG_V8F_STRAT1 section 6, COSTS-HEADS row) names the arm (S1d = S1 with `--ctx-detach on`, same seeds and reads) and gives no outcome rule for it. These rules are the reader's (`s1d_read.py` docstring, printed into its result).

seeds {0, 1, 10, 11}; init 0 = {0, 10}, init 1 = {1, 11}; the count rule is the SEEDS read rule r2 (>= 3 of the 4 seeds, >= 1 in each init).
  COST4_S1(cell)  S1 - C0 costs (count rule).   COST4_S1d(cell)  S1d - C0 costs (count rule).   GAIN4_DET(cell)  S1d - S1 gains, i.e. S1d is separated better than S1 and |Delta| > F (count rule).
  For each COST4_S1 cell:  CURED = not COST4_S1d and GAIN4_DET;  PARTIAL = not COST4_S1d and not GAIN4_DET;  REMAINS = COST4_S1d (the cost is not carried by the ctx-loss gradient into the adapter).
  NEW-COST = a COST4_S1d cell that is not a COST4_S1 cell (listed, never hidden).   GAIN4 of S1d - C0 is listed.
  next step (PROPOSED): CURED -> S1d replaces S1 as the route-text arm and the card KEEPS / ADDS next step (N2 card) is read on S1d;  PARTIAL -> reported as written, S1d is the cost-free arm but the detach gain is not
  separated: S1d and the N1 attribution are read together;  REMAINS -> reported as written; the cost is not the ctx-loss gradient, the next lever is named from the N1 read attribution (NAV4 -> the N2 card).
  STRATEGIC (card sections 4.4 / 4.5, read on S1d at the pairs {0, 1} and {10, 11}): route following label, route setting RC-A, detected info per variable with K4; a variable DETECTS in S1 and not in S1d is listed as DETECT-LOST.
  K-RE: every recomputed pair that exists in a banked result (RESULT_STRAT1.json, RESULT_STRAT1_SEEDS.json, RESULT_STRAT1_N1S10.json when present) equals it exactly; each file must have >= 1 compared row.
