Each DQ gives the options, the design-calc numbers (section 8; R = random log effects, E = conservative; ESTIMATED), and my recommendation. The numbers are for C1 (905 tokens / 24 logs) unless stated.

**DQ1. What does C1 count as CONFIRMED?**
- (A) **O1 verbatim only** (the ruling's wording). At a true +0.327 it passes 56 % (R) / 34 % (E); at +0.16, 10 % / 6 %; false pass at 0: 0.4 % / 0.5 %. A fail is mostly the point bar's noise (point sd about 0.13), so it cannot refute.
- (B) **the sign bar only** (lower CI > 0). True +0.327: 93 % / 49 %; +0.16: 27 % / 8 %; false pass at 0: 1.8 % / 0.8 %.
- (C) **two named outcomes from the one read**: O1 = CONFIRMED (size and direction), O2 = DIRECTIONAL (lower CI > 0, point below +0.30 or the seed-range term). No new look is spent; the O1 bar is unchanged.
- **Recommendation: (C),** written in section 6. It keeps L1's bar verbatim as the word "confirmed", and it stops a true effect near the observed one being read as "not confirmed" 44-66 % of the time (1 - P(O1) at +0.327, R / E).

**DQ2. What licenses the EP-ADMISSION bank build and, through it, the navtest look?**
- (A) O1 only (the strict reading of Step B decision 0). (B) O1 or O2.
- At the observed +0.327 an O1-only licence blocks navtest in 44 % (R) / 66 % (E) of reads, at +0.16 in 90 % / 94 %. Navtest is where the size is resolved (half-width 0.065 R / 0.178 E; sign bar at +0.16: 100 % / 47 %), and navtest is still gated by admission and by its own bar (DQ3).
- **Recommendation: (B).** Nothing is licensed by C1 alone beyond building the bank and asking admission.

**DQ3. The navtest (C2, T1) bar, to be sealed in its own amendment.**
- (A) L1's O1 verbatim on T1 PDMS: at +0.327, 79 % (R) / 58 % (E); at +0.24 only 5 % / 25 %. (B) the sign bar (lower CI > 0) with the point and the four families reported: at +0.16, 100 % / 47 %; at +0.08, 71 % / 12 %; false pass at 0, 2.1 % / 1.3 %.
- Navtest P2 is 88.82 [87.86, 89.77] (T1, MEASURED), against the paper's 94.55 (PUBLISHED). A gain smaller than +0.30 is still what the programme wants, so a +0.30 magnitude bar would call a real +0.24 a miss 95 % of the time (R).
- **Recommendation: (B)** for the decision, with a pre-stated minimum worthwhile effect to be chosen by you (I do not have a justified number), and the safety families and the per-seed range as sealed O1 terms.

**DQ4. Keep the seed-range term (point > max - min of the 3 per-seed points)?** It passes 90 % at +0.327 and 70 % at +0.16 at B-REST, so it is not what binds. **Recommendation: keep** (it is the only guard against the inference-seed variance this rig has been shown to have).

**DQ5. May a pooled carve B + B-REST line (1,305 tokens) appear?** Never as the verdict (your ruling: the confirm is C1 alone; carve B is post hoc and shares the same 24 logs). **Recommendation: allow it only as a descriptive line labelled POST-HOC-POOLED, with no outcome attached.** Its power was not simulated.

**DQ6. EP-ADMISSION's bank geometry (section 7).** `ep_admission.py` is hard-wired to S0 (117 logs, 15,808 sets, a C-KEYS bijection onto P2's S0 bank), and its error rates were simulated there.
- (a) Build a bank adapter and re-simulate false refusal and power at 24 logs / 905 tokens x 3 seeds under an amendment sealed before the bank is read (CPU, about 1 h, ESTIMATED; the R1 simulator exists).
- (b) Admit on an S0-geometry bank of the search (P2 + search on S0's 8,922 in-sample navtrain tokens: needs a new GPU forward, hours, and S0 is in-sample for P2).
- (c) Waive A1 / A2 for the search (contradicts your ruling's decision 9; I do not recommend it).
- **Recommendation: (a).** If the re-simulated power at 24 logs is too low to decide, fall back to (b) and say so.

**DQ7. The SEARCH-DOSE arm (the pinned O4 config, M = 256) after C1.** It reuses the B-REST bundle and costs CPU only (about 4 h ESTIMATED if linear in M, which is 4x). **Recommendation: not in this seal; its own amendment after the C1 read,** because it is a second lever on the same tokens. RULE ZERO is met by naming it now and by the chain leaving the bundle on disk.

**DQ8. C-ID strictness.** The brief says bit for bit; I implemented bit for bit on 16 carve-B tokens (8 moved, 8 never moved) for the bundle rows, the ctx rows and the search outputs. UNVERIFIED hazard: a CPU float reduction can differ in the last bits across thread counts; the search is run at `--threads 4`, the Step B setting. **Recommendation: keep bit for bit; on a miss, diagnose threads first; do not loosen the gate silently.**
