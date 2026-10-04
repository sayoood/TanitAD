# SFT-4: the paper's scorer labels (NAVSIM PDM targets) + the teacher's lane signal as a 7th output (pre-registration)

> Written 2026-10-04 before any SFT-4 step; landed in git before its first held-out read.
> PI 2026-10-04: *"do all three"*:
> 1. let SFT-3 run;
> 2. build the NAVSIM PDM relabel and train on it;
> 3. add the teacher's centre-line distance as a lane label.

## Why (MEASURED / PUBLISHED)

**The paper's labels.** DriveZero (2609.06055) Sec. 3, verified in the banked PDF:
- *"These components are supervised by their corresponding PDM targets [17]"*;
- only the candidate trajectories are detached.

Ours were the teacher's calculators: RETRACTION_LOG R30, `REVIEW_7_GAP_TO_PAPER.md`. The review measured that selection is the bottleneck:
- best of 64 is 97.07 against a pick of 83.24 on the full navtest;
- an oracle on NC + DAC + TTC alone reads 95.16.

**The relabeller is validated.** `refe/navsim_pdm_targets.py` rebuilds NAVSIM's metric cache from the nuPlan DB and runs NAVSIM's own simulator and scorer. On 300 navtest tokens / 19,500 trajectories, against NAVSIM's REAL cache (`validate_pdm_targets.json`):
- DAC / C / DDC agree 100 %;
- NC agrees 99.93 % and TTC 99.89 %;
- PDMS mean |Δ| is 0.0008;
- the argmax is identical on 300/300.

On the pod, DDC equals the independently computed NAVSIM direction label on 4/4 held-out sets.

**The lane signal is validated.** The teacher simulator's own centre-line distance (`center_line.CenterLine.info`) ranks NAVSIM-clean plans above violating ones at within-set AUC **0.904** (oncoming) and **0.856** (lane keeping) on the 3,137 held-out sets (`teacher_lane_validation.json`). The leaves our labels had used (wrong-way, centre-line reward) read 0.50–0.62.

## Design (fixed now)

- **Base:** `model_final.pt` (md5 `b54773f8…`).
  - Trainable: ONLY `score_q_mlp`, `score_dec`, `score_head`; the 64 proposals stay bit-identical.
  - `score_head` grows from 6 to 7 outputs. Rows 0–5 are copied exactly from the base (step-0 outputs identical, gate G3). Row 6 starts at weight 0, bias 1.5.
- **Labels, on the EXECUTED plans** (A7 last-heading repair → the seam's `to_navsim`):
  - **Components 0–5 = NAVSIM PDM targets:** NC, DAC, EP (pairwise vs PDM-Closed, as the harness), TTC, C, DDC. Source: `refe/onpolicy_relabel_pdm.py` on the 174,912 served training sets and the 3,137 held-out sets.
  - **Pure-PDM sets only (`--require-pdm`):** training and the held-out read use ONLY sets with PDM targets, so no set mixes label sources. Sets the 5 s guard excludes are counted and reported.
  - **Output 6 = the teacher lane label:** soft target clip(1 − (d − 0.5 m) / 1.5 m, 0, 1) of the teacher's centre-line distance d. Source: a seeded 15,000-set subset of the served sets (seed 20261004) and all held-out sets.
    - The run uses the subset rows WRITTEN BY LAUNCH TIME; the count is logged in the `data` event and gate G10. The relabeller runs at about 32 s per set per worker, so the subset may be partial at launch.
    - The loss is MASKED wherever the label is absent.
- **Arms, one pass over the same batches:**
  - **A:** BCE on the six PDM targets + 1.0 × masked BCE on the lane output.
  - **B:** A + 1.0 × L_ER. L_ER is SFT-3's expected-score loss, with the selection policy π = softmax(log(navsim_v1(p) × p_lane) / 0.1) and R = the PDM-target navsim_v1 score.
- **Selection rule for SFT-4:** `navsim_v1_lane` = navsim_v1 × p(lane) (planner rule, 7-output checkpoints only). The read also reports plain navsim_v1.
- **Settings:** AdamW, lr 1e-4 (50-update warm-up, cosine to 0), weight decay 0.01, batch 8 × accumulation 8, 1 epoch over the PDM-labelled sets, seed 0, bf16; held-out read every 400 updates.
- **Launch gate (`--preflight` must PASS):**
  - G1 trainable = scorer;
  - G2 live;
  - G3 step-0 identity on outputs 0–5;
  - G4 B at λ 0 = A;
  - G5 round trip through the planner's loader, sized by the checkpoint's `n_score_components`;
  - G7 held-out disjoint;
  - **G9:** PDM targets served on both banks, ≤ 5 % stale, pairwise-EP values present;
  - **G10:** the teacher lane label reaches output 6.
  - **Deliberate regression:** an empty PDM label dir must turn G9 RED.

### Amendment 1 (2026-10-04, before any SFT-4 training step): how G3 is checked with a 7-row head

The first memory-light smoke (`refe/sft4_smoke2.sh`, 1,654 PDM-labelled training sets) failed only G3.
- MEASURED: max |base − A| on outputs 0–5 = **0.0625**; B vs A = 0.0.
- 0.0625 is exactly one bf16 rounding step at |logit| 8–16. The 7-row head is a different GEMM shape from the base's
  6-row head, so under bf16 autocast the shared outputs may round differently. The check is wrong here; the copy is not.
- The ulp formula was checked against real bf16 spacing: adjacent bf16 values read exactly 1.0 ulp.

With `--lane-head`, G3 is now asserted EXACTLY wherever identity holds by construction:
- the score head's INPUT is bit-identical across base, A and B (the same unchanged modules);
- rows 0–5 applied as a 6-row GEMM reproduce the base output bit for bit;
- the copied parameters are bit-equal (head rows 0–5, `score_q_mlp`, `score_dec`);
- row 6 is initialised to weight 0, bias 1.5;
- B equals A exactly.

For the full 7-row head, the gating residual is computed in strict fp32 (TF32 off) on the same captured input, and it
must stay within the standard forward-error bound for any summation order: 2 · K · eps32 · Σ|h·w|, plus the bias add.
The bf16 residual is reported, not gated.

A first attempt at that bound used one bf16 ulp **of the output**. It read 89 ulps, because the bias is added after the
GEMM's bf16 rounding: an output near 0 carries rounding at the larger pre-bias magnitude. That was a wrong scale, not a
copy defect. It was replaced before any training step.

Without `--lane-head`, G3 is unchanged (exact).

**New deliberate regression:** `REFE_SFT4_MUTATE_G3=1` mis-copies one weight of row 2 by 1e-3. It must turn G3 RED. It
runs before launch, beside the chain's registered G9 mutation.

**MEASURED, smoke round 2 with the final code** (`raw/2026-10-04-sft4-smoke/r2_*.log`; 1,654 PDM-labelled training sets,
1,500 lane-labelled):
- **Normal smoke: all 9 gates PASS.**
  - Head inputs are identical.
  - Rows 0–5 computed as a 6-row GEMM reproduce the base with residual 0.0.
  - The fp32 residual of the full head is **0.0** (bound 0.0011).
  - The bf16 residual is 0.0625, i.e. 12 pre-bias ulps (reported, not gated).
  - Main-process peak RSS: 6.3 GB.
- **Mutated smoke: G3 RED on three independent checks.**
  - The copied parameters are not equal.
  - The 6-row GEMM differs from the base by 0.03125.
  - The fp32 residual is 0.0054, above the 0.0011 bound.

### Amendment 2 (2026-10-04 15:33 UTC, before any SFT-4 training step): the PDM labels get the CPU first

The pod's CPU quota is 7.65 CPUs (R32). Shared with the teacher-lane relabel, the PDM training relabel projected to about
20 h. The 12 teacher-lane workers are therefore SIGSTOPped (no work lost) until `ZZPDM_TRAIN_EXIT`, then resumed
automatically (`refe/lane_pause.sh`, log `/workspace/data/refe_sft2/lane_pause.log`).
- The run's design is unchanged.
- The lane subset at launch will be smaller; it is logged in the `data` event and G10, as already declared above.

### Amendment 3 (2026-10-04 16:07 UTC, before any SFT-4 training step): a bounded lane catch-up window

Measured timing:
- SFT-3 runs at 10.7 s per update, so it should finish around 01:30 UTC.
- The PDM relabel, with the CPU to itself (Amendment 2), runs at about 258 sets/min and finishes around 02:00 UTC.

Chain v1 would therefore launch the moment the lane relabel resumes, with only the ~2,100 lane sets written before the
pause (1.2 % of training). Chain v2 (`refe/sft4_chain2.sh`) inserts a window after `ZZPDM_TRAIN_EXIT`: it waits until
**7,000 lane sets OR 150 min, whichever comes first**. Then it runs the unchanged gate → G9 mutation → run.
- v1 was stopped by explicit PID while it slept; `chain.log` records the swap.
- The count at launch is logged; the design is otherwise unchanged.

## Stage 1: held-out (pod, label-only; the 3,137 sets of 24 logs that carry PDM targets)

Truth = the PDM-target navsim_v1 score of the chosen plan (the harness score of executing it). Comparisons are against the deployed system: the base scorer under navsim_v1.

- **Primary:** pick(arm, navsim_v1_lane) − pick(base, navsim_v1) × 100.
- **Estimator:** paired log-cluster bootstrap (10,000 resamples, seed 20260927).
- **An arm PASSES** iff its lower bound > 0 at the final update. If both pass, B goes forward only if pick(B) − pick(A) > 0; otherwise A.
- **Reported, not gating:**
  - the same under navsim_v1;
  - oncoming picks (NAVSIM DDC < 1) and lane-keeping picks (NAVSIM lk10, `--lane-labels-heldout`) with paired CIs;
  - AUC of NC / DAC / DDC and of the lane output vs the teacher target;
  - picks scoring 0.

## Stage 2: navtest (dev box; never on the pod)

- **Seam:** the winner's checkpoint with `--rule navsim_v1_lane` in the unchanged seam (route fix ON, A7 ON).
- **Tokens:** Amendment 5's 923 confirmation tokens.
- **ADOPT** iff the paired PDMS lower bound > 0 AND no longitudinal or lateral family component separates adversely.
- **REFUTED** iff the upper bound < 0.
- Otherwise **NOT PROVEN**.
- The full 12,146-token navtest and the four families follow for an adopted scorer.

The interval answers *another draw of logs* only: one fine-tune seed, deterministic inference.
