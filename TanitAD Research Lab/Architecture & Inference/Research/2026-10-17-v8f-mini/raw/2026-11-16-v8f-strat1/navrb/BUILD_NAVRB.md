# BUILD_NAVRB -- roundabout rows of the route text

Evidence class MEASURED, CPU, zero training. Results `out/RESULT_NAVRB.md` / `.json`; rule and bars `PREREG_NAVRB.md` (before any number). Tests `test_navrb.py` (7 pass): a mutation per rule (left/right swap, causal scan, flag-0 requirement, state rule) goes RED; a future-read mutant goes RED; the grammar check shows roundabout rows carry no side. Inputs sha256 in the json; `navtok_src.py` / `texts.json` are copies of the landed navtok files. Windows are the 43 ROUTE-LACKS windows of the 6 roundabout episodes of RESULT_NAVAUDIT (n 43, 6 episodes).

## Reading
**RB-STATE-ONLY.** The router text has no side or bearing for a roundabout. The exit-number convention (R1) names the v9 side on 20 of 43 (0.465), below the 0.90 bar and below a constant-R guess (35 of 43), so the exit number does not give the side (the roundabout arm count is not in the text). A roundabout state is the router row at NOW on 43 of 43.

## Cost of each option (from the STRAT1 records; the pod-GPU name L40 and the 0.19 s/step are INHERITED from the chain comment `launch/strat1_chain2.sh:2`, no local timing log)
All arms are 3,000 steps (`argv.json`), about 0.16 h each at 0.19 s/step, excluding mint / gate / read. Text-nav arms: N1, S1 (and S1d) at s0, s1, s10, s11; C0 uses v9 and is not repeated.
* Option A, change nothing: 0 h.
* Option B, a new nav state (ROUNDABOUT, an extra channel or token: an architecture change): N1 and S1 at s0 / s1 = 4 arms, about 0.63 L40-h; with s10 / s11 and S1d, 12 arms, about 1.9 L40-h; plus a new builder, gate profile and read.
* Option C, the same content as the existing 5-channel nav (a wider cut: +12 windows, 3 wrong-side, RESULT_NAVAUDIT): same arm counts, no architecture change.

## Decision for the PI
A route-only side for roundabout exits is not recoverable from the router text (45 % right against 81 % for always-right), so no change to the existing 5-channel nav can name those turns without reading the ego path, which the vision-only rule forbids at inference. A roundabout state is present at NOW on all 43 windows and 16 of 141 episodes; adding it needs a new nav channel (an architecture change, your decision) and a repeat of the text-nav arms: about 0.63 L40-h for N1 / S1 at s0 / s1, about 1.9 L40-h for all 12 text-nav arms (estimates). Whether the state helps the lateral head has not been measured; the cheapest test before any training is the nav-only classifier of navecho2 with a state channel added.
