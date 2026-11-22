# Card ADD110 (FALLATTR10K): where in the gap range does the price in falling behind of the camera-fed lead rule come from

Pre-registration of the v8proof fork, written 2026-11-22 after row 174 (card ADD108) was read and before any tool of this card is written. Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions, executor of record `v8f_t1_eval_hb.py` with the heading bound 0.2 1/m), pre-roll of 30 steps (6 s), the two 10k models `UE_s0` and `UE_s1`. No training, no fit, no encode: rollouts of study arms on existing models with gap files composed from files that exist.

## 1. Question and why

Row 172: with the gap of the camera head the lead rule holds contact near zero, and it falls behind by more than 10 m on more windows than the form fed by the label gap (GEc against RLc: +0.0246 [+0.0053, +0.0486]). An 18 % cut of the head's T0 gap error (row 172) did not move this price, and windows at the head's direction did not cut the T0 error in a separated way (row 174). Before another input is built, this card asks which part of the gap error the price sits on: the gap on frames where the label has a lead at all, and there the near, middle or far range. The label gap enters the gap files here as a tool of attribution only; a file with a label value is not admissible at inference and no arm of this card is a candidate.

## 2. Arms

Base: GEc of card ADD106 (banked rollouts; gap file `VGAPLE.npz`, the mean gap of the three large fits). All arms have the form VD5c of card ADD102 (wrapper variant 28bc31f21ecb4ccb unchanged, the run rule of note N1 of card ADD102: presence from the reader, the gap of a gap file, the closing input as the slope of the gap the rule receives). Each arm has its own gap file, composed row by row on the frames of the evaluation join:

* Label rows: frames where the label has a lead (valid flag 1, finite gap from 0 to 60 m).
* LA: the label gap on every label row; the gap of the base file on every other row.
* LN: the label gap on the label rows with a label gap below 20 m; the base file elsewhere.
* LM: the label gap on the label rows with a label gap from 20 m to below 40 m; the base file elsewhere.
* LF: the label gap on the label rows with a label gap of 40 m and above; the base file elsewhere.

A row with the label gap has an output whether or not the base file has one; elsewhere the output flag is that of the base file. Nothing is smoothed, clipped or blended. Four arms, both model seeds: eight rollouts.

References, banked and recomputed by the reader: GEc (contact 0.0062, falling behind 0.5154, unsafe 0.0839), RLc (0.0000, 0.4908, 0.0714), RR (0.0954, 0.4954, 0.1339).

## 3. Measures and names

Measures as in cards ADD101 to ADD106 on the 1,271 windows per model seed (650 pooled windows with a lead): CONTACT, UNSAFE, FALLING-BEHIND (behind the logged ego by more than 10 m). Paired differences by the two-level bootstrap over model seeds and evaluation episodes, 2,000 draws, the rules of those cards.

* Per arm X of LA, LN, LM, LF and measure M of FALLING-BEHIND, CONTACT, UNSAFE, against GEc: `FALLATTR10K-<X>-LOWERS-<M>` (upper end of the interval of X minus GEc below 0), `FALLATTR10K-<X>-RAISES-<M>` (lower end above 0), else `FALLATTR10K-<X>-<M>-NOT-SEPARATED`.
* The whole, from falling behind of LA: `FALLATTR10K-GAP-CARRIES-THE-PRICE` (LA lowers falling behind against GEc, and the interval of LA minus RLc holds 0 or lies below it); `FALLATTR10K-GAP-CARRIES-PART` (LA lowers falling behind against GEc, and the lower end of LA minus RLc is above 0); `FALLATTR10K-PRICE-NOT-IN-THE-GAP` (LA does not lower falling behind against GEc, and the lower end of LA minus RLc is above 0); every other case `FALLATTR10K-NOT-DECIDED`.
* The range: `FALLATTR10K-PRICE-IN-NEAR`, `-IN-MIDDLE`, `-IN-FAR` for each of LN, LM, LF that lowers falling behind against GEc; when none does, `FALLATTR10K-NO-RANGE-SEPARATED`.

Printed with no name: levels of every arm per model seed and pooled; every arm against RLc and against RR on the three measures; per gap file the count of join rows that carry the label gap and the count of frame-to-frame changes of the source inside a clip; per rollout the share of rule steps with a lead, of them with a non-zero closing input, and the share of rule steps with a lead whose gap came from the label.

Stated before the data: the whole price is 0.0246 of 650 windows, about 16 windows. LA can show it; each range arm moves a part of it, and `FALLATTR10K-NO-RANGE-SEPARATED` is a likely reading with two model seeds. The T0 error of the head is largest and most one-sided at 40 to 60 m (it reads short by more than 5 m on about 0.4 of those frames), so the far range is the first suspect; this is a guess written down before the data, not a finding.

Readings committed in advance. GAP-CARRIES-THE-PRICE with a range: the next lever is the gap reading in that range (for the far range a head or a target built for range, or a second narrow window; each as its own card). GAP-CARRIES-THE-PRICE with no range separated: the gap error as a whole is the lever and the split needs more model seeds, which is a PI decision. PRICE-NOT-IN-THE-GAP: the price sits on frames where the reader reports a lead and the label has none, or on the presence output; the next card swaps the presence, not the gap. GAP-CARRIES-PART: both. NOT-DECIDED: reported as written; more model seeds are the PI's call.

## 4. Controls

All have to pass before any name or any value of this card is opened; logged in `TIMES_NOTE.md` before the names. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

* a0 (the gap files are what the card says): each composed file equals, on every join row, the label gap where its range holds and the gap and output flag of the base file elsewhere, by a second routine (largest absolute difference at most 1e-9 m); the rows of LN, LM and LF with a label gap are disjoint and together are those of LA; the base file is the file whose hash the gap-file record of card ADD106 names; the evaluation join is the pinned one.
* a1 (wrapper): the wrapper file is 28bc31f21ecb4ccb, the pod check record of card ADD102 names it with a pass, and every run text names it.
* a2 (each arm is what the card says): in the rule log of every rollout, on every logged step where the rule had a lead and the gap file has an output, the gap the rule received equals the gap of its file at that clip and frame within 1e-3 m; a step with a lead and no output equals a gap of the control run V0 of card ADD102 within 1e-3 m; the meta names the gap file and the closing switch; the closing input is non-zero on a positive share of the steps with a lead.
* a3: the reader recomputes contact, falling behind and unsafe of GEc, RLc and RR as written in section 2 (within 1e-4; the values there are rounded to four places) from the banked rollouts.
* a4: the tests of every kit of this card pass in the lead's run on the dev box before the first rollout starts.
* a5: the engine's own controls of the grid read and of the strategic grid pass.

Limits by construction, stated before the freeze: none of a0 to a5 depends on the subject. In a range arm the source of the gap changes when the label gap crosses 20 or 40 m and when the label's lead starts or ends; by the run rule of the slope a step of more than 3.0 m ends the run and the closing input is 0 until three frames are in the new run, and a smaller step enters the slope. A range arm therefore differs from GEc in the gap and, near the changes, in the closing input; the counts of such changes are printed, and LA has them only where the label's lead starts or ends.

## 5. Four metric families

LONGITUDINAL (distance keeping): contact, unsafe rate, falling behind, with the engine's pair read. LATERAL, TACTICAL: the engine's pair read against the logged drive. STRATEGIC: read at T1-kin inside the reader under its own engine controls for LA, LN, LM, LF against GEc and against RLc.

## 6. Cost, stop rule, limits

Eight rollouts on the fork pod, at most three at a time, ESTIMATED 1.2 to 1.6 pod-hours from the ten rollouts of card ADD106 (about 92 minutes); stop rule 3 pod-hours from the start of the queue. No PI decision is needed for rollouts of study arms on existing models; nothing of this card enters the candidate spec.

Limits: T1-kin only, no re-perception; the lead moves as logged whatever the ego does; two model seeds of one line; the label gap is the reference and not ground truth; an attribution with label values in the input says where the price sits, not how to remove it; differences rest on few windows.
