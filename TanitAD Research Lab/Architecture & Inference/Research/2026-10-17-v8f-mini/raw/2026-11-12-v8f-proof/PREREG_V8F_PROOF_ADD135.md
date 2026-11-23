# PREREG_V8F_PROOF_ADD135 — LABELUNSAFE10K: what are the unsafe windows of the lead rule when it is fed label inputs?

Frozen before any tool of this card is written and before any value of it exists. Open at this time: rows 200 to 202. Tier **T1-kin/hb0.2** (kinematic rollout on the model's own actions, executor of record `v8f_t1_eval_hb.py` with the heading bound 0.2 1/m). CPU only, on banked runs; no new rollout, no training, no fit.

## 1. Why this card

At readout K with no pre-roll the label form of the lead rule (label presence, label gap; a reference, not a candidate) has an unsafe rate of 0.0369 on four model seeds; the candidate adds 0.0111 to it through its camera gap, and rows 172 to 202 have read that gap line out. The larger term has not been read: about 21 of 562 scored windows per model seed are unsafe with exact inputs. An unsafe window is one where the rollout has a step with a gap below 2.0 m or a time gap below 1.0 s while the logged driver has none (definitions D3 to D6 of the headway reader). Two readings are open and call for different levers: the rule follows about as close as the logged driver did, in windows where the driver himself is near the line of 1.0 s, so that the count is set by the line; or the rule lets the model close in where the driver did not, so that the rule's response is the lever. This card tells them apart on the banked runs.

Stated guess before data: nearly all unsafe windows are unsafe by the time gap alone (M1 at least 0.9); most are shallow (lowest time gap at least 0.8 s); in more than half the logged driver's own lowest time gap is below 1.2 s; the name is `THRESHOLD-NEAR-THE-LOGGED-DRIVER`, and more than half of the windows repeat over the model seeds.

## 2. Runs and windows

Banked runs at K0 on the model seeds 0 to 3: the label form RLc `I10RLcKF5Hb20` (the subject), and beside it the candidate GEc `I10RQ50KF5Hb20gGEc` and the margin form A90 `I10RQ50KF5Hb20gA90`. Windows: the lead windows of the engine on which U is defined (the logged driver has no unsafe step). Per window and step the engine's own quantities: model gap g_m, model time gap tau_m, logged gap gap_l, logged time gap tau_l. Unsafe windows of an arm: U = 1, pooled over the four model seeds (a window of the grid counts once per model seed).

## 3. Measures and estimator

On the unsafe windows of RLc:

* **M1** — share that have no step with g_m below 2.0 m (unsafe by the time gap alone).
* **M2** — share whose lowest tau_m is at least 0.9 s; beside it the median of the lowest tau_m.
* **M3** — share in which the logged driver's lowest tau_l over the scored steps is below 1.2 s; beside it the same share over all scored windows.
* **M4** — share in which the lowest tau_m lies more than 0.2 s below the lowest tau_l.
* **M5** — of the distinct grid windows unsafe at one model seed or more, the share unsafe at three or four.

Each share of M1 to M4 with a cluster bootstrap over the evaluation episodes, 2,000 draws, seed 135, 95 % interval; M5 with the same bootstrap. Beside, no name: the counts; the label gap and the model speed at the first unsafe step (by gap 0 to 10, 10 to 20, 20 to 40, 40 to 60 m); the step index of the first unsafe step; M1 to M5 for GEc and A90; of the unsafe windows of GEc, the share that are also unsafe windows of RLc at the same model seed. The interval answers what another draw of episodes would say, not what another training run would say.

Families: LONGITUDINAL (distance keeping: time gap and gap against the logged driver) is the subject. LATERAL, TACTICAL and STRATEGIC are not read by this card (it re-reads banked runs whose family reads are in rows 193 and 202; n = 0 here).

## 4. Names

Mechanism, one name: `LABELUNSAFE10K-K0-THRESHOLD-NEAR-THE-LOGGED-DRIVER` (the lower end of the interval of M3 is at least 0.5 and the upper end of the interval of M4 is at most 0.5); `LABELUNSAFE10K-K0-RULE-FOLLOWS-CLOSER-THAN-LOGGED` (the lower end of the interval of M4 is at least 0.5); `LABELUNSAFE10K-K0-MIXED` (else).

Repeat, one name: `LABELUNSAFE10K-K0-SAME-WINDOWS-ACROSS-SEEDS` (M5 at least 0.5), else `LABELUNSAFE10K-K0-WINDOWS-DIFFER-BY-SEED`.

## 5. Controls (all have to pass before any name or value is opened)

* **u0 identity** — the tool's unsafe rate per model seed and pooled reproduces row 202 within 1e-4 for RLc (0.0391, 0.0338, 0.0391, 0.0356; pooled 0.0369), GEc (pooled 0.0480) and A90 (pooled 0.0365), from the engine's own per-window values; the engine files are imported unchanged (hashes recorded).
* **u1 second routine** — the lowest time gap and the first unsafe step of every window agree with a second routine, step by step, within 1e-9.
* **u2 planted** — on the real windows with the model track replaced by the logged track, no window is unsafe and M4 has no window; with the logged track advanced along the path by 0.35 s of travel at each step, every window that becomes unsafe counts for M4 and none of them is unsafe by the gap alone unless its gap is below 2.0 m.
* **u3 tests** — the tool's test file passes in the lead's run on the dev box and on the pod before the run of record; outputs outside the kit folder.
* **u4 counts** — 141 dump files and the done marker for each of the twelve runs; 642 lead windows per model seed.

This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

## 6. Committed readings

* **THRESHOLD-NEAR-THE-LOGGED-DRIVER** → the unsafe rate that remains with exact inputs is set by the line of 1.0 s against how the logged driver drives; it is stated as a property of the measure and of the rule's target time gap, and the lever is that target, a parameter of the rule: a card with a rollout cost goes to the PI and the other session.
* **RULE-FOLLOWS-CLOSER-THAN-LOGGED** → the rule's response is the lever; the next card reads the rule logs of these windows on banked runs (what the rule commanded and what the rollout did), CPU only.
* **MIXED** → both shares are stated and the next card splits the windows by M4.
* `SAME-WINDOWS-ACROSS-SEEDS` beside any of them says the windows are a property of the scenes and of the rule and not of the model seed.

## 7. Cost and limits

ESTIMATED under 15 minutes of CPU on the fork pod, no GPU time. Limits: T1-kin only, no re-perception, the lead moves as logged; about 83 unsafe windows over four model seeds, so the intervals of the shares are wide; the lines of 0.9 s, 1.2 s and 0.2 s are choices of this card; four model seeds of one line; the label form is a reference with label values and not a candidate.
