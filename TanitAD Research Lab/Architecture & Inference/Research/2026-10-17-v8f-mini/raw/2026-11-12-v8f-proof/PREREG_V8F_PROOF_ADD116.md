# PREREG_V8F_PROOF_ADD116 — PRESERR-T0: how the camera reader's presence errs against the label, and whether its threshold has room

Frozen before the tool of this card is written and before any value of it exists; no value of cards ADD114 and ADD115 is open at this time. Tier **T0** (the reader's reads on recorded evaluation frames; a diagnostic of the reader, not driving performance). CPU only, on banked files of the fork pod; no rollout, no fit, no encode.

## 1. Why this card

The lead rule of the candidate takes "a lead is present" from the token reader of each 10k model: probability at or above 0.5 with a full five-frame history. Cards ADD114 and ADD115 ask at T1-kin whether the unsafe rate that remains above the label form sits in that presence input. Whatever they read, the lever that follows needs to know how the presence errs: which way (a lead of the label that the reader does not mark, or a mark with no lead of the label), at which gaps, and whether another threshold on the same probability would remove misses at an acceptable number of extra marks. The threshold is one constant of the wrapper, so a threshold arm is the cheapest lever there is; a new presence head is the next one. This card decides between the two before either is paid for.

Stated guess before data: misses sit at the larger gaps, and a lower threshold halves them only with more than twice the marks with no lead (no room).

## 2. Data

For each model seed m in {0, 1}: the rule log and the later-look rule log of the banked reader-fed run at readout K with no pre-roll (`I10RQ50KF5Hb20_s<m>`) and their side files `_rd.npz` (the reader's probability `p` and history flag `ok`, row for row). A frame is a pair (clip index, frame index); the frames of a seed are the unique pairs of the two logs with a full history (`ok`). The reader's read of a frame does not depend on the rollout. The label of a frame is the row `off[ci] + j` of the evaluation join `join_eval141.npz`: a lead of the label L means marking 1 and a finite gap from 0 to 60 m (the rule's own test). Structure of these logs was checked on the banked files before this freeze (no rate was computed): `present` of the log equals `p >= 0.5 and ok and not src_bad` on every row, and `p` is the same on every repeat of a frame.

## 3. Measures and estimator

For a threshold t, the reader marks a frame when `p >= t`. Miss rate M(t) = share of frames with L that are not marked; extra-mark rate X(t) = share of frames without L that are marked. Per model seed and pooled as the mean of the two seeds:

* M(0.5), X(0.5), and the counts behind them; M(0.5) by label gap in the bins 0–10, 10–20, 20–40, 40–60 m with counts;
* M(t) and X(t) on the grid t = 0.1, 0.2, ..., 0.9;
* the area under the curve of `p` for L against not L.

Intervals: bootstrap over clips (the clips of the evaluation join that have frames), 2,000 draws, the same draw of clips for both seeds; reported for M(0.5), X(0.5), and for M(t)/M(0.5) and X(t)/X(0.5) at every t. The interval answers what another draw of clips would say for these two readers, not what another reader fit would say.

Four families: this is a T0 read of one input of the longitudinal rule. LONGITUDINAL: the measures above. LATERAL, TACTICAL and STRATEGIC: not read by this card; reason: the card scores a presence input on recorded frames and runs no plan; n = 0 for each.

## 4. Names

Let t* be the largest t below 0.5 on the grid with pooled M(t)/M(0.5) at most 0.5.

* `PRESERR-T0-THRESHOLD-HAS-ROOM` — t* exists, the upper end of the interval of M(t*)/M(0.5) is below 0.75, and pooled X(t*)/X(0.5) is at most 2.0;
* `PRESERR-T0-THRESHOLD-HAS-NO-ROOM` — otherwise (no such t on the grid, or the miss cut is not held by the interval, or the extra marks more than double; also when M(0.5) or X(0.5) is 0 pooled, where a ratio does not exist).

One descriptive name on where the misses are, from M(0.5) by bin: `PRESERR-T0-MISSES-AT-LARGE-GAPS` when the pooled miss rate of the bins from 20 m up is at least twice that of the bins below 20 m with the interval of the difference above 0; `PRESERR-T0-MISSES-AT-SMALL-GAPS` for the reverse; else `PRESERR-T0-MISSES-NOT-ORDERED-BY-GAP`.

## 5. Controls (all have to pass before any name or value is opened)

* **h0 mapping** — on the banked label-form run `I10RLcKF5Hb20` of each seed, `present` of the rule log equals L of the join row `off[ci] + j` on every logged row (the definition the lead's rehearsal for card ADD114 confirmed).
* **h1 side files** — for each seed and both logs: the side file has as many rows as the log, `p` equals `p_src` of the log within 1e-6, and `present` equals `p >= 0.5 and ok and not src_bad` on every row.
* **h2 frames** — `p` is the same within 1e-6 on every repeat of a frame within a seed; both seeds have the same set of frames; every frame maps to a row of the join.
* **h3 threshold identity** — M(0.5) and X(0.5) computed from `p` equal the rates computed from `present` of the logs on the same frames (rows with `src_bad` counted and reported; the control fails if there are any, since then the two are not the same thing).
* **h4 planted** — the tool's test passes in the lead's run on the pod before the read (it includes a synthetic world with known rates, a shifted-threshold world where HAS-ROOM has to come out, and one where it must not), output outside the kit folder.
* **h5 inputs** — the hashes of the join and of the log files read are recorded; the hash of the join equals the join hash that the label-form logs of control h0 name.

This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

## 6. Committed readings

The lever of this card is taken up only if card ADD114 or card ADD115 names the presence input (PRESENCE-CARRIES-UNSAFE or BOTH-CARRY-UNSAFE at a setting); the read itself is reported whatever they give.

* **THRESHOLD-HAS-ROOM** → threshold arms of the candidate at T1-kin on the fork pod: t* and its lower neighbour on the grid, both model seeds, at the convention and at readout K with no pre-roll, through a generated wrapper variant whose one changed constant is the threshold, with an identity control at 0.5 against the banked candidate run. Own card. A changed threshold is a rule parameter: as a study arm it is measured; its adoption goes to the PI and the other session.
* **THRESHOLD-HAS-NO-ROOM** → the presence head card: a head fitted on the existing tokens of the train clips, with its T0 bar taken from this card (miss rate at most half of M(0.5) with at most twice X(0.5) on these frames), read at T0 before any rollout.
* MISSES-AT-LARGE-GAPS / AT-SMALL-GAPS / NOT-ORDERED → stated in the lever card as where its bar is read (the bins named here).

## 7. Cost and limits

CPU, minutes; no pod GPU. Limits: T0 only; two readers (one per model seed) of one recipe; the label marking is a reference and not ground truth, so a "miss" is a disagreement with the label; frames are those the rollout grid logs (ticks and later looks of 141 clips), not every frame of the clips; frames within a clip are not independent, which the bootstrap over clips answers and a per-frame count does not.
