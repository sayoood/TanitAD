# PREREG_V8F_PROOF addendum 1: RESPROBE, the first test of theorem T5 on cached tokens (frozen 2026-11-13 ~01:40, before its data)

Card: PREREG_V8F_PROOF.md (sha256 38d33abd...c3a7), hypothesis H-V8F-PROOF-4. This addendum adds a test that needs no pod and no new encode: the dev-box eval-141 cache holds the frozen DINOv3 tokens at two grids, 4x10 (64 px cells) and 8x20 (32 px cells). Tier T0 (frozen-feature probe on logged frames): a perception diagnostic, never driving performance.

## Targets (labels may use ego and other agents; the probe inputs are vision tokens, with an ego-only control)
* GAP: range to the lead vehicle in metres, on windows with a lead present (the obstacle.offline join used by LATSEM2B; the same lead definition, quoted with file and line).
* NEAR: range to the nearest agent within 30 m. COUNT: number of agents within 30 m. LEADP: lead present (binary).

## Feature sets and probes
* E: the ego kinematic vector the readers get (control, no vision). V40: the 40 tokens of the current frame (4x10). V160: the 160 tokens (8x20). V40+E, V160+E.
* Probe class P1: ridge on the flattened tokens after a PCA fitted on the training folds only (rank 256). Probe class P2: one hidden layer (256 units), early stopping on an inner split. The ego-only control gets BOTH classes and its better score is used, so the vision gain is never credit for an under-fitted control.
* 5 folds by episode, fixed seed. Metrics: R2 for GAP, NEAR, COUNT; AUROC for LEADP. Paired episode-cluster bootstrap, B 2,000, for every difference.
* Controls that must read known values: (k1) targets shuffled within fold: every R2 <= 0.02 and AUROC within 0.03 of 0.5; (k2) a synthetic target that is a fixed linear function of V40 must read R2 >= 0.95 under P1.

## Predictions (theorem T5 with f = 305.58 px/rad, h = 1.31 m)
* P4a-1: for GAP, V160+E minus V40+E is positive with a CI excluding 0.
* P4a-2: the vision gain (V+E minus E) for GAP is smaller on windows with a true gap >= 12.5 m than on windows with a gap < 12.5 m, at both grids.
* P4a-3: LEADP and COUNT (which need no range) are decodable from V40 well above the ego-only control.
## Outcomes
* RES-CONFIRMED: P4a-1 and P4a-2 both hold. RES-PARTIAL: one holds. GAP-ABSENT: neither grid beats the ego-only control on GAP (CI includes 0): the range is not in the frozen tokens at this image scale at all, the strong form of T5; next lever is the native-resolution crop. RES-REFUTED: V40+E beats E on GAP by R2 >= 0.20 in the >= 12.5 m stratum: the pooled tokens do carry far range and T5 as stated is wrong for this encoder (it would then be a size or context cue; the theorem bounds position only).
* Next lever, committed for every outcome: un-pooled 16x40 tokens and a native-resolution centre crop on the pod (P4a full, P4b, P4c).
