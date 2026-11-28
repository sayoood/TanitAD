# SFT-2: teach the scorer NAVSIM's driving direction and use it to choose (pre-registration)

> Written 2026-10-04 before any SFT-2 training step; landed in git before its first held-out read is looked at.
> It follows LANE-1 (`raw/2026-10-04-lane-discipline/PREREG_LANE1.md`) and the PI's video review: *"even if the fan is
> within the target lane, sometimes the selected trajectory is leaving the lane"*.

## Why (MEASURED before this registration)

- **The selection picks oncoming plans while clean ones exist.** On navtest (LANE-1 census, partial read of 1,180 tokens):
  - the selected plan enters an oncoming or off-route lane on 3.9 % of tokens, against 1.4 % for the human;
  - in 93 % of those cases a clean hypothesis with at least the same true PDMS existed.
- **The scorer was trained on a different event.** Its direction label is DriveRL's wrong-way category. On the 3,137 held-out sets (200,768 proposals) that label catches only 31.2 % of NAVSIM's direction violations, and only 46.4 % of its flags are NAVSIM violations (`teacher_vs_navsim_ddc_heldout.json`). The direction head's within-set AUC against NAVSIM's verdict is about 0.60 on navtest.
- **The selection rule ignores the head.** `navsim_v1` never multiplies in the direction head.
- **A faithful label now exists.** `refe/navsim_lane.py` agrees with NAVSIM's scorer on 13,000 of 13,000 navtest trajectories (`validate_navsim_ddc.json`). It is written for every training and held-out set by `refe/onpolicy_relabel_lane.py` as side files keyed to the exact proposals (key + ckpt_step).

## Design (fixed now)

- **Base:** `model_final.pt` (md5 `b54773f8…`), loaded exactly as the planner does.
  - Trainable: ONLY `score_q_mlp`, `score_dec`, `score_head`.
  - The 64 proposals stay bit-identical.
- **Labels:** SFT-1's, except that the DDC component is NAVSIM's verdict (1.0 / 0.5 / 0.0), set by `OnPolicyBank(lane_labels=…)` on the sets whose key AND ckpt_step match. All others keep DriveRL's label.
- **Arms, one pass over the same batches:**
  - **A:** per-component BCE (the training loss) on the new labels.
  - **B:** weighted per-component BCE, weights NC 1, DAC 2, EP 1, TTC 1, C 1, DDC 3 (`--b-mode compw --b-compw 1,2,1,1,1,3`).
  - Drivable area is the largest failure and direction the new signal. No ranking term: SFT-1's ListNet arm read −0.73 [−1.58, +0.25] at its first held-out read.
- **Settings:** AdamW, lr 1e-4 (50-update warm-up, cosine to 0), weight decay 0.01, batch 8 × accumulation 8, 1 epoch, seed 0, bf16.
  - The higher lr than SFT-1 (3e-5) is because SFT-1's control arm A did not move at 3e-5 (+0.02 [−0.33, +0.49] at update 400).
- **Selection rule for SFT-2:** L1 = argmax navsim_v1(p) × p_DDC. The held-out read also reports navsim_v1 and both base rules.
- **Launch gate (`--preflight` must PASS):**
  - G1–G7 as SFT-1;
  - G4 in compw form: unit weights reproduce A's loss, and the declared weights differ;
  - **G8:** the lane labels are in both banks (matched sets > 0, ≤ 5 % stale, NAVSIM's 0.5 values present);
  - deliberate regression: launching with an empty lane-label dir must turn G8 RED.

## Stage 1: held-out (pod, label-only; the 3,137 sets of 24 logs, NAVSIM direction labels)

Each arm is compared with the DEPLOYED system: the base scorer under navsim_v1.

- **Primary:** pick(arm, L1) − pick(base, navsim_v1), on the true navsim_v1 score × 100.
- **Co-primary:** the share of picks with NAVSIM DDC < 1, arm under L1 minus base under navsim_v1, in percentage points.
- **Estimator:** both with the paired log-cluster bootstrap (24 logs, 10,000 resamples, seed 20260927).
- **An arm PASSES** iff, at the final update, the pick lower bound > −0.10 (non-inferior) AND the oncoming upper bound < 0.
- **If both pass,** the larger pick goes forward; if tied within 0.10, the lower oncoming rate.
- **Reported:** base under L1 (the zero-training lever on navtrain), AUC of NC, DAC and DDC, picks scoring 0, skill.

## Stage 2: navtest (dev box; never on the pod)

- **Seam:** the winner's checkpoint + L1 in the unchanged seam (route fix ON, A7 ON).
- **Tokens:** Amendment 5's 923 confirmation tokens.
- **ADOPT iff ALL hold:**
  - the paired PDMS lower bound > −0.10;
  - the harness's DDC < 1 pick share falls (upper bound < 0);
  - no longitudinal or lateral family component separates adversely.
- **REFUTED** iff the PDMS upper bound < −0.10, or the oncoming share does not fall (lower bound ≥ 0).
- Otherwise **NOT PROVEN**.
- The full navtest is then reported, with the four families.

## Next lever if SFT-2 fails or leaves lane keeping unsolved

- **Lane keeping (lk10)** enters the labels as a 7th component, if LANE-1's census shows a selection effect over the human floor.
- **The fan:** the trajectory teacher leaves NAVSIM's drivable area on 4.97 % [1.30, 10.31] of held-out samples where the human does 0.00 %, and the model's hypotheses fail it 60 % of the time on those samples, against 24 % overall (`teacher_check_heldout_summary.json`). The proposal-head lever is to drop or replace those targets. That is a trajectory-head fine-tune, outside this scorer-only registration.

The interval answers *another draw of logs* only: one fine-tune seed, deterministic inference.

## Status (added 2026-10-04 ~11:50 local, before any SFT-2 step)

**NOT LAUNCHED — the PI chose the paper-pure labels** (*"let's do the paper pure way"*).
- The pod chain was disarmed by explicit PID.
- The NAVSIM direction and lane labels already written stay as the EVALUATION yardstick: oncoming and lane-centre pick shares on the held-out sets. They are not training labels.

The paper-pure replacement has two parts:
1. the teacher simulator's own lane signals (`refe/onpolicy_relabel_teacher_lane.py`), validated on the held-out sets before any training;
2. SFT-3 (`eval/PREREG_SFT3.md`), the expected-score objective on the existing teacher labels.
