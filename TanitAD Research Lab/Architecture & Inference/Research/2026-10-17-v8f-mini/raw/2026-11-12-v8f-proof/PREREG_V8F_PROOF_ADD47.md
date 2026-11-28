# PREREG_V8F_PROOF_ADD47 (PRICE10K): the price of the reader-fed lead rule on the 10k checkpoints, by the level of the map; no training

Card of the v8F proof study, written by the coordinator fork after row H-V8F-PROOF-101 and after a post hoc split of its price (section 1). Frozen before any map of this card is fitted and before any run of it (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions; the lead does not react and follows its logged track); never driving performance. Tier of the map facts: T0.

## 1. What is known (MEASURED, row 101 and one post hoc split)

* Row 101: on the frozen 10k union checkpoints `UE_s0`, `UE_s1` the reader-fed rule (`RR`, map level 0.20) cuts the unsafe rate U from 0.1459 / 0.1139 to 0.0836 / 0.0516 and reaches the label-fed rule (`RLb`, 0.0890 / 0.0516). Its price is above that of the label-fed rule: F, the share of lead windows in which the ego falls behind by more than 10 m at the last scored step, is 0.3162 / 0.3614 against 0.1978 / 0.2648 (no rule 0.0950 / 0.1854); `RR` against `RLb` +0.1075 [+0.0665, +0.1551].
* Post hoc split of that price on the existing dumps (`integ10k/price_split.py`, `integ10k/read/PRICE_SPLIT.json`; it names nothing, it ranks the levers): of the 6 s position error that `RR` adds above `RLb`, the 642 lead windows hold 0.835 (seed 0) and 0.664 (seed 1); there `RR` is 4.61 and 2.86 m further behind than `RLb`. The 1,757 other windows hold the rest: `RR` is more than 2 m further behind than the run without the rule in 0.078 and 0.060 of them, the label-fed rule in 0.024 and 0.020.
* So the larger term is the gap that is fed where a lead exists, and the map level is the dose that acts on it. The presence threshold is the smaller term and is not an arm of this card.

## 2. Question

Does the reader-fed rule keep its cut of the unsafe rate when the gap is fed without added caution, that is with the map at the median (level 0.50) instead of the 0.20 quantile, and is its price then that of the label-fed rule?

## 3. Arms (no training, no new reader)

Models, reader arms, rule form, presence threshold 0.5, evaluation cache, window grid and tool are those of card ADD46 (checkpoints `UE_s0`, `UE_s1`; reader arms `LR16_s0`, `LR16_s1`; model seed k with reader seed k). Only the gap that enters the rule changes, `g_used = min(g_reader, Q(g_reader))`:

| run | Q | role |
|---|---|---|
| `RQ50` | the map of card ADD46 fitted at level 0.50 | deciding |
| `RQ35` | the same at level 0.35 | beside, a point of the curve |
| `RRaw` | the identity (`g_used = g_reader`; a map file with the knots (1, 1) and (150, 150)) | beside, the end of the curve |

The maps at 0.35 and 0.50 are fitted on the pod from the banked reader rows of card ADD46, with the tool of that card and the same fit and check clips; only the level differs. Reused without a rerun: the dumps `R0`, `RLb` and `RR` of card ADD46. Six new rollouts.

## 4. Reads

Headway reader and family reader of card ADD46, pooled over the two model seeds, paired episode-cluster bootstrap (it answers another draw of episodes; the two model seeds are the only replicate for training variance; the rollout does not sample). U on the 562 scored windows per seed, F on the 642 lead windows.

* P1 = U(`RQ50`) - U(`R0`), P2 = U(`RQ50`) - U(`RLb`), P3 = F(`RQ50`) - F(`RLb`), P4 = F(`RQ50`) - F(`RR`).
* Beside: P1 to P3 for `RQ35` and `RRaw`; the values of `RR` from row 101; the four families for `RQ50` against `R0`, `RLb` and `RR`.

## 5. Names (the deciding level is 0.50 and is fixed here; no level is chosen on the evaluation windows)

* `PRICE10K-MEDIAN-KEEPS-CUT`: P1 below 0 at both model seeds and its pooled interval excludes 0. It carries one of two endings:
  * `-AT-LABEL-PRICE`: the interval of P3 includes 0 or lies below 0;
  * `-PRICE-REMAINS`: the interval of P3 lies above 0.
* `PRICE10K-MEDIAN-LOSES-CUT`: otherwise.
* Beside, no name: whether the interval of P2 includes 0 ("reaches the label-fed rule on the unsafe rate") and the curve over the levels 0.20, 0.35, 0.50 and identity.

What follows, fixed now:
* KEEPS-CUT-AT-LABEL-PRICE: the candidate takes level 0.50. Next: the reader fitted on the union (card ADD45, a go of the PI) and the planner form beside the rule.
* KEEPS-CUT-PRICE-REMAINS: the candidate takes level 0.50 if the interval of P4 lies below 0, else it stays at 0.20. The remaining price of `RQ50` above `RLb` is split again between lead windows and other windows; if the other windows hold more than half of it the next arm is the presence threshold, else the reader (card ADD45).
* LOSES-CUT: the caution of the map is what cuts. The candidate keeps level 0.20 with its price, the curve is reported, and the next lever is a reader with a smaller gap error (card ADD45, a go of the PI) and the planner form.

## 6. Controls (all must pass; else nothing is named)

* m1 (maps, T0): the knot positions of the 0.35 and 0.50 maps equal those of the banked 0.20 map of the same reader arm; at every knot the values are ordered 0.20 <= 0.35 <= 0.50 <= position; the coverage on the check rows rises with the level and is at most the level + 0.05 (a coverage under the level is expected where the map is capped at the read value; it is reported); the unit check of the tool passes; fit and check clips are disjoint; the identity file gives `g_used = g_reader` on a grid from 0 to 150 m.
* m2 (identity in the run): in the rule log of `RRaw` the fed gap equals the read gap at every tick with a finite read.
* m3 (same read, ordered feed): the reader reads cached tokens of logged frames, so at a model seed the read gap and the presence probability of every tick are equal in the logs of `RR`, `RQ35`, `RQ50` and `RRaw`, and the fed gap is ordered `RR` <= `RQ35` <= `RQ50` <= `RRaw` at every tick with a finite read.
* m4 (counts): 141 dump files, 2,399 windows and the same lead windows in every dump.
* m5 (reused dumps): the result files of `R0`, `RLb` and `RR` have the sha256 recorded before the run.
* m6: the controls of the headway reader and of the family reader on the new pairs.
* m7 (model): the checkpoint sha256 written by each new run equals that of card ADD46 for its model seed.

## 7. Limits

As row 101: 56 episodes with a lead; a lead that does not react; a rule without closing speed; a reader fitted on corpus tokens. Three more comparisons are made on the same lead episodes; only the level 0.50 can give a name. Control k8 of card ADD46 stands as read by its note N2 (the train-log check of the tool failed on both reader arms).

STRATEGIC: as note N1 of card ADD46 (at T0 on the first tick the banked read of the two models without the rule; along the rule-driven rollout NOT COMPUTED, n = 0, no instrument).

Cost: six rollouts of about four minutes each on this fork's pod beside one other job; no training.
