# PREREG_V8F_PROOF_ADD141 — CONTACTRH10K: in the reader form, does RH make its extra contacts where the vision-only reader misreads the gap, or does it follow closer on the same read?

Frozen before any tool of this card is written and before any value of it exists. Open at this time: row 206 (card ADD140, landed at 89e34ebc). Tier **T1-kin/hb0.2** (kinematic rollout on the model's own actions; logged camera frames; the lead does not react). CPU only on the banked runs and rule logs of card ADD140; no new rollout, no training. Never driving performance.

## 1. Why this card

Row 206: with the reader fixed, RH under readout K makes more contact than UE under readout K in the reader form, +0.0073 [+0.0022, +0.0143] (+0.0062, +0.0062, +0.0093 per seed), and not in the label form (-0.0005). The difference is a recipe effect. Two mechanisms predict it: (M1) RH drives into states where the reader misreads the gap more (a larger overestimate of the gap, or a missed lead, at the tick before contact); (M2) on the same quality of read, RH's plans close the gap faster than UE's.

Stated guess before data: M2, name `CONTACTRH10K-SAME-READ`.

## 2. Windows and quantities

Runs: `H_RQ50K_s<m>` and `U_RQ50K_s<m>`, m = 0, 1, 2 (card ADD140 and card ADD139, controls passed), eval-141 grid, their rule logs (`rule_log_<arm>.npz`, fields of `dist_rule.py` LOG_FIELDS) and dumps. A contact window and its first contact step follow `headway/HEADWAY_DEFS.md` and the landed headway reader (`headway_read_r2.py`, window cells) unchanged. Per model seed: set X = windows with contact for RH and none for UE; set S = windows with contact for both. Per window in X (RH's run) and in S (both runs): the last rule tick at or before the first contact step; at that tick the reader's gap `g_src` minus the label gap of the same (clip, step) from `join_eval141.npz` (signed, metres; positive = the reader sees the lead farther than it is), and the presence miss (reader presence 0 while the label has a lead).

* Q1 (misread): mean signed gap error in X (RH) minus mean signed gap error in S (RH), pooled over seeds; episode-cluster bootstrap, 2,000 draws, seed 141.
* Q2 (presence): share of presence misses in X (RH) minus in S (RH).
* Q3 (closing on the same read): in S, RH's closing rate at that tick (rule log `c_src` or the logged own-speed minus lead-speed, whichever the rule used) minus UE's, paired by window.
* Beside: the counts |X| and |S| per seed, and the same Q1 for windows where UE makes contact and RH does not.

## 3. Controls (all must pass before any value is opened; a failed control stops the card, a successor is frozen)

* c1 the contact windows recomputed by this tool equal the landed headway reader's contact cells of each run (same windows, same first steps).
* c2 every window of X and S has a rule tick at or before its first contact step in its rule log; the logged `g_src` of that tick equals the reader's gap recomputed from the dump within 1e-3 m on a planted sample of 20 ticks.
* c3 planted check: a copy of one run's rule log with the gap error set to +2.0 m in X windows reads Q1 = +2.0 within 1e-6; with no change Q1 = the tool's value.
* c4 tests of the tool pass on synthetic data and on the dev box.

Names opened after the controls pass.

## 4. Names

* `CONTACTRH10K-MISREAD` if Q1 is above zero (interval excludes zero) or Q2 is above zero (interval excludes zero).
* `CONTACTRH10K-SAME-READ` if Q1 and Q2 are not separated and Q3 is above zero (RH closes faster on the same read, interval excludes zero).
* `CONTACTRH10K-UNRESOLVED` otherwise, with |X| stated (a set below 10 windows over three seeds is reported as too small to read).
* Next lever committed: MISREAD sends the cost to the vision-only reader (rows 117, 123, 124); SAME-READ sends it to the rule (the row-204 rule change, a PI decision) or to the recipe (the other session).

## 5. Cost

CPU on the dev box, minutes; the dumps and rule logs are already pulled to `rhacc10k/pull`.
