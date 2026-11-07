# PREREG_DRAFT_SCORER1: does a speed-balanced pairwise loss on the safety heads remove u400's speed pessimism? (DRAFT, not frozen)

**Status.** This is a draft for the Master Mind to freeze, by sha256, before any arm trains or any held-out number is read.
- Stream: SCORER-GYM, 2026-11-07.
- Every number quoted for u400 is MEASURED, from `refe_selection_probe_20261107/P1_DIAG.md` and `p1_diag_all.json`.
- ⛔ **navtest is never used for training.**

## 0 · Why this is the cheapest discriminating experiment

**What is known (MEASURED).**
- u400 = SFT-4 arm B at update 400. It was **already** trained with per-head BCE plus a listwise expected-true-PDMS term (`sft4_pod.sh:10`: `--b-mode expreward --lam 1.0`; `scorer_finetune.py:113-119`).
- EPO-style RL on top of that has run three times and was K3-killed on TTC each time (`D-REFE-RLS1-R0`, `-R0B`, `-SAFEOBJ-R0D`).
- Head calibration cannot fix within-scene ranking (`D-REFE-RLS1-O2-CALIB`).
- A global speed recalibration was refuted at screen (`H-SEL-RECAL-1`).

**What is left untried.** The defect's own signature, P1 Q4, is a **within-token** error on the safety heads. On CRUISE, Spearman against v4/v0, predicted vs true:

| head | predicted | true | gap Δρ |
|---|---|---|---|
| TTC | −0.479 | −0.049 | −0.430 |
| NC | −0.443 | −0.039 | −0.404 |
| DAC | −0.290 | −0.039 | — |

No objective in the trainer acts on that error:
- BCE is pointwise across tokens;
- the listwise term acts on the aggregate at τ_s 0.1, i.e. on the top of the set.

So the cheapest lever aimed at the measured defect is a **within-token, speed-balanced pairwise loss on the NC / TTC / DAC heads**:
- the generator is frozen, so the proposals are bit-identical;
- it reuses u400's data and recipe;
- it needs ~3 short pod arms.

## 1 · Stage 0 (zero GPU, decides whether Stage 1 runs at all)

**What it does.** Re-run `p1_diag.py`'s Q4 statistic UNCHANGED on **navtrain held-out sets**: `refe_heldout/sets` (24 logs), or RL-val. Use u400's own logits and the true PDM labels.

**What it needs.** u400's per-proposal logits on those sets.
- The location of a banked dump is **UNVERIFIED**: none was found on E: or F:.
- If none exists, producing one is a dev-box GPU inference job (the dev box has CUDA, per `refe_ego_probe_20261106/full/results_meta.json`).

**Committed readings.**
- **Z1, the gap is present on navtrain held-out** (mean Δρ over TTC and NC ≤ −0.20, with CI-hi < 0): the defect is in the objective or the representation, not a navtrain → navtest shift. ⇒ **run Stage 1.**
- **Z2, the gap is absent on navtrain held-out** (CI includes 0, or |mean Δρ| < 0.10): the defect is a distribution shift. ⇒ **Stage 1 is not the lever.**
  - Next lever: localise the shift by city, scenario type and speed band between navtrain held-out and navtest.
  - The candidate fix is then data (resampling or reweighting navtrain toward navtest's CRUISE profile), not the loss.
- Anything between Z1 and Z2 ⇒ run Stage 1 and record Stage 0 as inconclusive.

## 2 · Stage 1 arms (all start from u400, `model_sft4_B_u400.pt`, sha256 `8b4dba48…`)

Every arm shares u400's `sft4_pod.sh` argv unchanged:
- data `--onpolicy /workspace/data/refe_navtrain/onpolicy/sets --pdm-labels-train $D/train_pdm --require-pdm`;
- `--lr 1e-4 --b-mode expreward --lam 1.0 --lane-head --lane-weight …`.

Each arm runs **400 further updates**, and is read **only at update 400**. Update 200 is a monitor read that cannot stop or select an arm.

| arm | difference from u400's recipe | role |
|---|---|---|
| **C0** | none (same recipe, 400 more updates, seed 0) | same-code "more training" control |
| **C0r** | C0 with data-order seed 1 | **training-variance floor** (`H-ESTIM-SEED-1`) |
| **S1** | C0 + λ_p · L_pair (below), seed 0 | the lever |
| u400 | no training | reference (zero point) |
| gBG(0.20, 0.05) | u400 + the P2 re-pick rule | descriptive only |
| P6 online-labelled scorer | — | reference **once it exists**; it is not a gate on this run |

**L_pair (new, ~40 lines in `scorer_finetune.py`; default OFF, so the OFF path is bit-identical).** For each labelled set and each head h ∈ {NC, DAC, TTC}:

1. **Form candidate pairs.** Take all pairs (i, j) with:
   - true y_h,i > y_h,j;
   - |v4_i − v4_j| / max(v0, 1) ≥ 0.15, where v4 is the 0.2 s chord at 4 s on the raw proposal, as in `p1_diag.py`.
2. **Split by speed.** Call a pair F if the truly-better proposal is the faster one, and S if it is the slower one.
3. **⭐ Balance.** Sample min(|F|, |S|) pairs from each side, without replacement, at seed 20261107.
   - If either side is empty, the set contributes **no** pairs for h.
   - ⇒ The loss's speed direction is zero by construction: it **cannot** teach "fast is good". It can only teach the head to order by its own truth at a fixed speed contrast.
4. **Loss.** L_pair = mean over the kept pairs of −log σ(s_h,i − s_h,j), on the head logits.
5. **Weight.** λ_p is fixed **before** any read by the gradient-norm rule (`refe/weight_rule.py`): ‖∇L_pair‖ = 0.5 · ‖∇L_BCE‖ on the first 50 training micro-batches. It is never tuned on held-out data.

**Launch gate (all must PASS; `launch_gate.py` profile `refe_scorer1`, to be built):**
- G-DVB on `--pair-w`, `--pair-dv`, `--pair-heads`.
- G-LIVE: a non-zero gradient on the scorer from L_pair. The **control must read exactly 0.0 with `--pair-w 0`**.
- A mutation arm:
  - delete the balancing step ⇒ the F/S pair ratio ≠ 1 must turn the check RED;
  - drop L_pair from `loss` ⇒ G-LIVE RED.
- ⛔ A navtest log/token gate on the training data, RED-tested.

## 3 · Evaluation sets (navtest, EVAL ONLY)

- **E1 = C3 1,058 tokens** (`table.npz`; 154 CRUISE).
  - The defect was diagnosed here, so E1 is **reported but never decisive alone**.
- **E2 = the deciding set.** A fresh navtest sample of **2,000 tokens, token-disjoint from C3**, drawn uniformly over navtest (no zeroed-stratum oversampling; P1's caveat) with seed 20261108.
  - A frozen fan means the true per-proposal PDMS table of u400's 64 proposals is computed **once** on CPU and shared by all arms.
  - Each arm then needs only its logits on E2 (dev-box GPU).
  - The pick's PDMS is a lookup, exact at the selection tier.
  - **G0 gate on E2:** the table reproduces u400's banked picks and its per-token PDMS from the full-navtest run (`D-REFE-SFT-4`) on 100 % of the overlapping tokens.
- **E3 = the full navtest, 12,146 tokens.**
  - Only the guard G1 is read on it.
  - Only the tokens where the arm's pick differs from u400's are scored (CPU harness `eval/score_navtest_refe.py`).

## 4 · Endpoints, estimator and bars (committed)

**Estimator.** Paired log-cluster bootstrap over E2's logs, 10,000 draws, seed 20260927, 95 % percentile.
- It answers *"another draw of logs"* only.
- Training variance is read from |C0r − C0|.
- The planner is deterministic, so there is no inference variance.

**Tier.** The selection tier (T0/T1 lookup) on E2/E3. Closed loop is read separately (§6).

| id | endpoint (on E2 unless stated) | bar for S1 vs C0 |
|---|---|---|
| **M1** (mechanism, primary) | Δρ = ρ(pred_h, v4/v0) − ρ(true_h, v4/v0), within-token Spearman on CRUISE, mean over h ∈ {TTC, NC} | S1 − C0 ≥ +0.5·\|Δρ_C0\| **and** CI-lo > 0 **and** \|S1 − C0\| > 2·\|C0r − C0\| |
| **B1** (behaviour, primary) | CRUISE slow-down share (plan v4/v0 < 0.85) | drop ≥ 10 points **and** CI-hi < 0 |
| **G1** (guard) | PDMS ×100 on E3 | lower CI > −0.5 |
| **G2** (guard) | TTC and NC sub-score ×100 on E3 | each point ≥ −0.5 (the registered g5 form) |
| **G3** (guard) | ranking regret (oracle − pick) on E2 | CI-hi of Δregret < +0.5 PDMS points |
| reported, not deciding | DAC Δρ, EP Δρ, lane margin share, pairwise ranking accuracy (≥ 0.95 vs ≤ 0.50), top-K oracle curve, longitudinal family (CRUISE v4 bias, overspeed share > 1.15), lateral family (heading, curvature, yaw rate, cross-track), tactical where available; STRATEGIC **PAUSED** (PI 2026-09-19); distance-keeping UNAVAILABLE at this tier (no lead track), read in §6 | — |

## 5 · Committed outcomes (each names its next lever, per Rule Zero)

- **O1 PASS** (M1, B1, G1, G2 and G3 all hold): *"a speed-balanced pairwise loss removes u400's speed pessimism"* is SUPPORTED, for one seed plus a replicate floor.
  - ⇒ Run S1 seed 1 (S1r).
  - ⇒ Run the sim session's RefePolicy on S1 (closed-loop scorecard, §6).
  - ⇒ If phantom braking persists in closed loop, move to gym rung A2/B1.
- **O2 MECHANISM ONLY** (M1 holds, B1 fails): the safety heads were not the deciding term often enough. P1 Q2 says the lane head leads on 31/55 tokens.
  - ⇒ Next lever: put the lane truth (the teacher centre-line distance) into the E2 table and add the lane head to L_pair (one arm).
- **O3 HARM** (M1 or B1 holds, G1 or G2 fails): balancing traded safety for speed.
  - ⇒ Next lever: the same loss under the SAFEOBJ dual constraint (the `--epo-con g5` machinery, `scorer_finetune.py:1178`).
- **O4 NULL** (M1 fails): the detached scorer cannot separate the pairs from its inputs.
  - ⇒ Next lever: the `--scorer-sees-trunk` ablation (`train.py:2141`), or privileged-teacher distillation (`SCORER_GYM_DESIGN.md` §2.7).
  - O4 is the predicted outcome if Stage 0 gave Z2.
- **O5 SWAMPED** (S1 − C0 inside 2·|C0r − C0| on M1 or B1): NOT PROVEN.
  - ⇒ Next: S1r, plus a second C0 replicate.
- ⛔ **A miss is reported as FAILED/NOT PROVEN as written. The bars do not move after a read.**

## 6 · Closed-loop read (after O1 only; descriptive, not part of the bar)

On the sim session's rig, run arms u400, C0 and S1:
- token `41d7b533797c5209`: the lead 17–19 m ahead, the banked 5.3 → 2.1 m/s phantom brake, and the 6.9 m/s overshoot;
- the PI scorecard: free-road speed hold, time gap 1.5–2.5 s, junction stall/yield, jerk/oscillation, lane keeping, collisions and TTC;
- the deliberate-regression arms *always-slowest*, *always-fastest* and *do-nothing*, each of which must read its known value.

## 7 · Cost and authorisation

**Cost (ESTIMATED).**
- Stage 0: a dev-box GPU logit dump on ~3,000 sets (≤ 1 h) plus CPU statistics (< 1 min).
- Stage 1: 3 arms × ≈ 3.5 A40-h ≈ **10.5 A40-h** (the RL-S1b dose).
- E2 table: 2,000 × 64 proposal simulations on CPU (minutes at the inherited labeller rate) plus 4 dev-box GPU logit dumps.

**⛔ PI decisions needed.**
- Pod GPU time for Stage 1.
- A new trainer flag set (`--pair-*`), which needs its G-DVB entries.
