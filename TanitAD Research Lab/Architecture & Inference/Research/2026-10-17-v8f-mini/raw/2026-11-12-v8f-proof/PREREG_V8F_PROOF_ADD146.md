# PREREG_V8F_PROOF_ADD146 — FCREAD10K: can the reader carry the lead forecast of card ADD144? A design read before any reader-form forecast rollout

Frozen before any tool of this card, before the runs of card ADD145 have ended and before any of their files is opened. Open at this time: row 210 (landed at b29fe129). Tier **T0-style design read** on logged frames (the reader reads logged camera tokens; nothing is rolled out by this card). Never driving performance.

## 1. Why this card

Row 210 (label form): the lead forecast with the lead's measured change of speed lowers U. The reader form has no lead-speed input (its closing is zero). The other session's read of row 210 names the admissible input: the reader's gap of the logged ego with the logged speed of the same ego, v_L,r(f) = max(0, v_l(f) + (g_r(f) - g_r(f - 5)) / 1.0 s), and a_L,r(f) = (v_L,r(f) - v_L,r(f - 5)) / 1.0 s, past reads only. Card ADD50 measured that a reader gap rate is weak against the label closing (Pearson r 0.275 and 0.198, bar 0.30, control d2 failed). An acceleration is a further difference of the same reads. Rolling out a reader-form forecast is only worth box time if the input carries the signal. This card measures that first.

Stated guess before data: `FCREAD10K-NOT-CARRIED` (r of the acceleration well below 0.30).

## 2. Data

The reader reads and logged speeds that the runs `H_RQ50KE1_s0..s2` of card ADD145 log at every step (tick logs `rule_log_*.npz` with their `_rd.npz` side files and the later-look logs `rule_log_mid_*.npz` with `_rd.npz`: clip, frame j or j + q, logged speed v_l, reader probability p, used gap g_used, history flag ok). Reader reads are of logged frames, so they do not depend on the model; seeds 0 and 2 share the reader TK16_s0, seed 1 uses TK16_s1. The label side: `LabelSource` of the rule wrapper on `join_eval141.npz` (rows at or before the frame), the label form's v_L and a_L exactly as `rulefc/build/v8f_t1_eval_reader_rule_rzgfhfc.py` builds them (`lead_accel`).

Per reader seed, one row per (clip, frame f) with a reader lead (p >= 0.5, ok) at f, f - 5 and f - 10, and a label lead at f and f - 5 (deduplicated over windows; a read of a frame is the same in every window).

## 3. Quantities and bars

* S: Pearson r of the reader closing (g_r(f) - g_r(f - 5)) / 1.0 s against the label closing at f. Bar 0.30 (card ADD50 d2).
* A: Pearson r of a_L,r(f) against the label a_L(f). Bar 0.30.
* Beside: n, the share of label-lead frames with a reader row, the RMS of a_L,r - a_L, the same r per seed, and the Spearman r.

## 4. Controls (must pass before any value is opened)

* k1: the label a_L of this tool equals the logged a_L of `H_RLcE1FC_s0`'s forecast log at the same (clip, frame) on every shared row (exact).
* k2: a planted reader gap g = 30 - 2 t (lead closing at 2 m/s) with v_l = 10 gives closing -2.0 and a_L,r = 0 exactly; a planted ramp of the closing gives the planted a_L.
* k3: a planted future read (frames after f) leaves both quantities unchanged.
* k4: at least 200 rows per reader seed.

## 5. Names and consequence (committed)

`FCREAD10K-CARRIED` if A >= 0.30 for every reader seed; then the reader-form forecast is rolled out (a successor card with the runs of card ADD144 in form RR). `FCREAD10K-SPEED-ONLY` if S >= 0.30 and A < 0.30 for some seed; then the reader-form arm uses the reader closing without the acceleration (a successor card). `FCREAD10K-NOT-CARRIED` otherwise; then no reader-form forecast is rolled out, and the lever is the reader's own lead-speed read (a vision closing input), named as the blocker with what it needs.

## 6. Cost

CPU minutes on the dev box. No rollout, no training.
