# Card PREREG_V8F_PROOF_ADD46 (INTEG10K-PRE): the first integrated candidate at 10k clips without new model training

Card of the v8F proof study, written by the coordinator fork. Frozen before any tool of this card exists and before any of its runs (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Tier of every rate and family below: T1-kin (kinematic rollout on the model's own actions; the lead does not react); the reads of the lead reader alone are T0; none is driving performance.

## 1. Why

* Coordination decisions D6 and D7 (register `COORD_V8F_2026-11-19.md`): the integrated candidate is built on the code line of the other v8F session; its first stage keeps the 10k model frozen and adds the lead levers at inference.
* INHERITED from the other session (its LLR read package): its built-in lead head, which feeds the decoder, read a separated LONGITUDINAL cost at both seeds; no setting keeps that head away from the decoder.
* MEASURED on this fork: a head that puts no gradient into the model carries the distance rule to the label-fed level on the unsafe rate (row H-V8F-PROOF-90) and confirms the planner result (row 99); a probe on the frozen encoder tokens reads the lead gap at R2 0.704 and presence at AUROC 0.897 over eight seeds (row 97, T0), independent of any driving model.

## 2. Parts (no weight of any driving model changes)

* **Models:** `UE_s0`, `UE_s1`, the two 10k arms of the other session (union of the corpus and the 10k clips), final checkpoints with sha256 2fd4b747532f55b2... and f13df8ce026eed31..., loaded by the tree of commit cdbff971.
* **Lead reader:** the token probe of card ADD43 (grid 4 by 10, per-token width 16, five-frame causal history, the loss and schedule of that card, 19,164 steps), two new arms `LR16_s0`, `LR16_s1`, trained on the corpus training clips of this fork's pod with one tenth held out: a clip is held out if the integer value of the first 8 hex digits of sha256(clip id) is divisible by 10. The reader reads cached encoder tokens only.
* **Quantile map:** on the held-out clips, the 0.20 quantile of the label gap as a monotone function of the reader's gap read (the map form of card ADD31). It is fitted on the held-out clips whose hash value is divisible by 20 and checked on the other held-out clips: the share of check rows with a label gap below the mapped value must lie in [0.15, 0.25].
* **Rule:** the banked distance rule in its form without the closing speed (cards ADD26 and ADD33), parameters unchanged, fed by `g_used = min(g_reader, q_map(g_reader))` where the reader's presence is at least 0.5.
* **Runs** with the evaluation call of the other session's reads (`run --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks --dump-heads`) through a study wrapper; the pinned evaluation tool itself is not edited. Per model seed k with reader seed k: `R0` (no rule), `RLb` (the rule fed by labels), `RR` (the rule fed by the reader and the map), `RRid` (the reader's gap replaced by 1e9 m), `RR0` (replaced by 0 m). Crossed, reported as the spread over reader seeds: `RR` of model seed 0 with reader seed 1 and of model seed 1 with reader seed 0.
* **Evaluation inputs:** the evaluation cache of the other session's reads (141 clips, manifest sha256 78631f832d73afdb...), copied to this fork's pod and checked clip by clip against its manifest; `eval_grid_windows.json` 659c3b65..., `join_eval141.npz` f0ea722f..., `native_tracks.pt` 1db431f1... (MEASURED equal on this fork's pod).
* Budget (ESTIMATED): about 1 pod-h for the two reader arms, under 1 pod-h for twelve rollouts, one GPU job at a time behind the three-job limit and the memory guard; under 3 pod-h with the copies. No planner run belongs to this card.

## 3. Reads (paired on windows; 2,399 windows of 141 episodes; U = model-induced unsafe rate of the headway read on the lead windows; two-level bootstrap over the two model seeds, then episodes, B 2,000; per-seed values beside every difference)

* E1 = U(`RR`) minus U(`R0`). Primary. Share reached = E1 divided by (U(`RLb`) minus U(`R0`)), per seed.
* E2 = U(`RR`) minus U(`RLb`); contact rate of the same pair.
* Families: `RR` against `R0` by the rule of card ADD12 Q3 with the same-sign rule of ADD13; the 6 s position error difference in metres with its interval. LONGITUDINAL, LATERAL and TACTICAL are read with this fork's readers. STRATEGIC: the model carries a strategic layer and this fork's reader has no strategic instrument, so the family is requested from the reader of the other session on the same dumps; until that read exists it is reported as NOT COMPUTED with this reason, never as a pass.

## 4. Names

* `INTEG10K-CUTS-UNSAFE`: E1 is below 0 at both model seeds, its two-level interval excludes 0, and both crossed runs read a U below that of `R0` of their model seed. Otherwise `INTEG10K-NO-CUT`.
* `INTEG10K-REACHES-LABEL-RULE`: the share reached is at least 0.5 at both seeds. Otherwise not named.
* Price: `INTEG10K-NO-COST` or `INTEG10K-COSTS-<family>` (`RR` against `R0`).

## 5. Controls (all must pass; else nothing is named and the outcome differences stay withheld)

* k1 (the reader is good enough; the rule frozen on 2026-11-19 before any head was read): on the 534 reference windows (`leversheet/LS_REF40_WINDOWS.json`), with the tokens of the evaluation cache above, the reader's gap error (mean absolute error, read clipped at 40 m, mean of the two reader seeds) is at most 5.80 m, and its presence AUROC over the labelled evaluation windows (positive: a lead with a gap of at most 40 m) is at least 0.85 at both seeds. The label gap at the mapped frame equals `gap_t` of the reference file on the scored windows (tolerance 1e-3); the share that matches is stated.
* k2 (identity): the dumps of `RRid` equal those of `R0` bit for bit, at both seeds.
* k3 (planted): `RR0` raises the 6 s position error against `R0`, interval excluding 0.
* k4 (known positive): `RLb` cuts U against `R0` at both seeds, pooled interval excluding 0.
* k5 (the model is the banked one): sha256 of both checkpoints as listed; `R0` of each seed reproduces the banked read of the other session for the same checkpoint (its position error at 6 s within 0.02 m; the banked value and its file are cited before `R0` is read).
* k6 (no leak): no evaluation clip is among the reader's training or held-out clips (count exactly 0; a planted shared id fails the check); fit and check clips of the map are disjoint.
* k7 (causal reader): with the tokens after the read tick replaced by noise the reader's outputs at that tick are unchanged.
* k8 (counts): 2,399 windows and the same lead windows in every dump; the reader arms pass the train-log check of card ADD43 at full length.
* k9 (map): the coverage on the check clips is inside the band, for both reader seeds.
* k10: the headway controls h1 to h6 and the family controls q1 to q5.
* Reported, no bar: the same `R0` on this fork's own encode of the evaluation clips (the two encodes of the same clips differ in content; MEASURED 0 of 141 content hashes equal), as the spread from the encode run.
* Tool checks before any run: tests of the wrapper against the pinned tool (identity, a hand-computed rule case), a counted mutation list (the min dropped, the reader fed a future frame, the reader of the other seed, the held-out split ignored) that goes red, and a dry run.

## 6. What follows (committed before the data)

* `INTEG10K-CUTS-UNSAFE`: candidate v1 exists without any new model training. Its price is set against rows 90 and 99. Next: the same reader fitted on the union of the corpus and the 10k clips (card ADD45, the data-scale step of the lead read, a go of the PI), the planner form beside the rule, and the STRATEGIC read of the other session on the same dumps.
* `INTEG10K-NO-CUT` with k1 passing: reported as written; the rule does not carry over to the 10k model. The share reached and E2 say whether the label-fed rule itself still cuts; the next arm is the rule fed by a five-frame head on the model's own adapted tokens.
* k1 fails: the token reader is not good enough on this evaluation encode; no name; the next arm is the same head fitted on the model's adapted tokens, and card ADD45 becomes the first step.
* Any `INTEG10K-COSTS-<family>`: reported beside the cut with its size.

## 7. Limits

Two model seeds and two reader seeds; one corpus for the reader (the 10k clips do not enter it in this card); the reader was trained on this fork's encode of the corpus clips and reads the other session's encode of the evaluation clips; T1-kin (the lead does not react, no re-perception); the rule and the reader are deterministic at inference, so there is no inference-seed spread; a rule at inference does not stand for a model trained to keep distance.
