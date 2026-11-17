# PREREG_V8F_ACTRSP, Amendment 2 (Option B): a calibrated 2 s acceleration term in a frozen-latent fine-tune. Arm spec, FROZEN before any code of this build ran

**Status.** Written 2026-11-17 by the Option B build agent, before any test or smoke of the new flags. Frozen card `ce2d02df...` (sha256, `PREREG_V8F_ACTRSP.md`); Amendment 1 draft at commit `76a8012f`, sha256 `4e122aa5...` (the file `amend1/AMENDMENT_1_BCAL_DRAFT.md`, section 2.3 is this spec's source). PI approval: 2026-11-17 (ACTRSP Option B, box 2). Nothing here is launched. Tier of every planner row read later: T0 (a world-model diagnostic, never driving performance).
**Code base.** Branch `agent/v8f-actrsp-20261116`, commit `a13f6f16` (trainer with `--wm-only`, `--wm-rsp-w`). It is NOT the tip of `agent/arch-inf-20260803` (`5e7083d8`): that trainer has no `--wm-only` (the card's section "Code base" measured the same). The lever lands on a branch from `a13f6f16`.

## 1 Arms (exactly as Amendment 1 section 2.3)
Per chassis k (k = 0, 1): start = `R35_s{k}` `ckpt_last.pt` with ALL `wm.*` tensors loaded (`--wm-init-from-ckpt-wm on`), `wm.proj` frozen (`--wm-freeze-proj on`), 4,000 steps, lambda 1.0 (brackets 0.3 and 3.0 are the named next doses, not run). Readout = the FROZEN RA of chassis k (`readout_K10_R35_s{k}.pt`, sha256 `edab7dce...` for k = 0, `a6d252aa...` for k = 1; s_a 0.20076 / 0.20323).

| order | tag | chassis | seed | term | pairs with |
|---|---|---|---|---|---|
| 0 | PILOT | 0 | 0 | FTRSP_s0 argv + `--steps 200`, own out dir | none (s/step, GPU memory, `rsp/L1` path) |
| 1 | FTRSP_s0 | 0 | 0 | `--wm-rsp-w 1.0 --wm-rsp-readout fixed_k` + probe + sha | FTNULL_s0 |
| 2 | FTNULL_s0 | 0 | 0 | none | FTRSP_s0 |
| 3 | FTRSP_s2 | 0 | 2 | as 1 | FTNULL_s2 |
| 4 | FTNULL_s2 | 0 | 2 | none | FTRSP_s2 |
| 5 | FTNULL_s4 | 0 | 4 | none | third floor arm |
| 6-10 | FTRSP_s1, FTNULL_s1, FTRSP_s3, FTNULL_s3, FTNULL_s5 | 1 | 1, 1, 3, 3, 5 | same pattern | same pattern |

* 10 arms: per chassis two FT-RSP (seeds A, B) and three FT-NULL (seeds A, B, C). Each FT-RSP pairs with the FT-NULL of its seed: same start, same data order; the term draws no random number (C4).
* Every other flag is `actsens_NULL_s{2,3}`'s (R35's recipe) except `--seed`, `--out`, `--steps 4000` (R35 used 19164), and the three new flags below. `--init-ckpt` = the R35 checkpoint (its non-WM tensors are M2b_s{k}'s, bit for bit).
* **Argv naming (flags whose default disables them are named in EVERY arm):** all ten name `--wm-init-from-ckpt-wm on --wm-freeze-proj on`. The four RSP arms also name `--wm-rsp-w 1.0 --wm-rsp-readout fixed_k --wm-rsp-probe <file> --wm-rsp-probe-sha256 <hex>`. The six NULL arms name none of the four (the checker refuses it).
* **The term.** Acceleration channel only, two contrasts (+/- 0.42 WM units), 9 anchors every 10 states, K = 10, context 1, one batched `rollout`; read = `(z_K^+ - z_K^-) @ w`, `w = RA.W / RA.sd` (the ridge mean and the anchor state cancel); `G = read / (2 delta_a s_a)`; loss `mean (G - 1)^2`; no floor, no batch slope, no head. Yaw is dropped. Gradient reaches `predictors.0` and `act_embed` only.
* **Replicate design.** Box-2 WM-only training is bit-identical at a fixed seed (`same-seed-replicate-is-bit-identical`), so a same-seed rerun reads a floor of 0.0 and says nothing about training variance. The replicate is therefore a DIFFERENT seed: FTNULL_s4 / FTNULL_s5 (same flags, term off). Floor F_T,k = max - min of the three FT-NULL point means of chassis k on the row. No same-seed rerun is run; if the MM wants one as a determinism control it is a CPU-side bit-compare of two selfchecks, not an arm.

## 2 Deviations from section 2.3, each with its reason
1. **Base commit** `a13f6f16`, not the repo tip (above).
2. **FT-NULL also freezes `wm.proj` and loads the checkpoint's WM** (section 2.3 says "the same flags with the term off"; this is that reading). Reason: only the term then differs between a pair.
3. **Schedule.** Recipe defaults (`--lr 3e-4`, `--warmup 200`, cosine over the 4,000 steps) because section 2.3 names none. A fine-tune at the recipe peak LR is the conservative reading of "R35's recipe"; see rulings.
4. **Seeds** chosen: chassis 0 {0, 2, 4}, chassis 1 {1, 3, 5} (the card's seed lattice). Section 2.3 says "seeds A, B, C" without values.
5. Controls renamed C9' / C10' as 2.3 states; C9' is built (a test, below); C10' is a read-side instrument and is NOT built here.

## 3 Read and committed outcomes (both chassis, both lever seeds per chassis required)
**Mechanism read.** IA-3P-K: RA's form cross-fitted on the 141 eval clips with IA-3's folds, calibrated by its s_swap on real pairs, acceleration channel only. MECH-MOVED = at both chassis and both lever seeds the arm's calibrated L1 2.0 s longitudinal gain lies above its FT-NULL pair's, CIs disjoint, gap above F_T,k. TARGET-READ (beside, no bar): the arm's own cycle gain on the eval clips and the `rsp/L1` trajectory; step-0 prediction (ESTIMATED from Amendment 1 1.5): mean G_aa 1.2, step-0 loss 0.94 / 0.99.
**Deciding read.** HWM-PLAN phase-1 rows (card 4.2), paired against the FT-NULL of the same seed; band 0-2 s, `acc_mae` (LONGITUDINAL) and `yaw_mae` (LATERAL). BETTER / WORSE exactly as card 4.2 with Floor_k = max(F_T,k, F_I). Absolute rows against HOLD and against R35 (reference only).
**Families, per band, never pooled:** LONGITUDINAL (`acc_mae`, `speed_err`, `hw_err`/`tg_err`/`ttc_err` with n), LATERAL (`yaw_mae`, `head_err`, `curv_err`, `xtrack`, `ade`), TACTICAL (plan-derived, goal/anchor selection NOT COMPUTABLE with its reason), STRATEGIC (REPORTED IN FULL, full 6 s band; `S_CKPT`, 6-30 s intent, derived goals, detected info structural n/a with reasons, never a pass). Estimator: paired episode-cluster bootstrap B 2,000 seed 0, 994 SCORE windows, 71 episodes (answers another draw of episodes only); training floor from three runs per chassis (another training run); inference floor F_I (another inference run). Each interval names its question.

| outcome | condition | next step (run in the same package) |
|---|---|---|
| OPTB-CEILING | HWM-ORACLE(arm) - HOLD at 0-2 s: CI wholly below 0 on `acc_mae` or `yaw_mae` for all four lever arms, not WORSE on the other | this WM replaces R35 for planning; read 2-6 s and 6 s at once; T1 read of HWM-PLAN phase 2 goal-head arms on it; term at levels 2 and 3 by amendment |
| OPTB-NARROWS | BETTER on a deciding quantity, not WORSE on the other, not CEILING | lambda 3.0, four runs, same pairs and floors, carded before any window is scored |
| OPTB-MECH-ONLY | MECH-MOVED, no deciding quantity BETTER | the response is right and the plan did not improve: LON-PLAN, zero training, dev-box CPU, prior-regularised planner (`--lam > 0`, dose fixed by an HWM-PLAN amendment), then the level-2 image constraint |
| OPTB-NULL | not MECH-MOVED, not BETTER, C2/C3 holding | (i) term trained down (end `rsp/L1` <= 0.5 x first, eval cycle gain >= 0.8): the WM met its own readout without a response a second readout sees, so the lever is the readout/latent: LAT-TOK (Amendment 1 2.2) and the adapter, PI decision. (ii) did not train down: lambda 3.0 |
| OPTB-HARM | WORSE at either chassis, or BAND-MET lost at a level | term stays off; lambda 0.3, four runs |
| SEEDS-DISAGREE | a label at one chassis only, or the two lever seeds of a chassis differ in sign | reported as such; the less favourable label's next step applies |

An arm that misses its bar FAILED and is reported so; its next step runs (Rule Zero). Done = CEILING with its T1 read started, or every next step blocked with the blocker and its unblock named.

## 4 Controls
C1 flag-absent parity with `a13f6f16` (state dict, loss terms); C2 log identity `(level/L1 - pred - 0.08 sigreg - 100 idm - 1.0 rollout - 1.0 rsp)/...` = 1.0 within 1e-4 on RSP arms; C3 step-0 identity (`live.c3`); C4 pairing (FT-RSP and FT-NULL of one seed: equal start state sha256 and equal (ci, t) lists over the first 50 batches); C5 post-training: Adam state on every `wm.*` tensor except `wm.proj` (141 of 145) AND `wm.proj` bit-equal to the checkpoint's; C6 windows equal; C7 repro + self; C8 zero-act regression; **C9'** analytic identity (float64): world `z' = z + B* a`, W = I, constant `s = K w.B*[:,0]` gives G = 1, loss and gradient 0; B = 0 gives loss exactly 1.0; `s` doubled gives G = 0.5 exactly (a batch slope would not); **C10'** the s_swap CI wholly above 0 on acceleration at both chassis, measured once on R35 before any FT checkpoint is read (read side, not built here).

## 5 Cost (ESTIMATED from the card's MEASURED 0.48-0.51 s/step and 49 min overhead per arm)
10 arms x (4,000 x 0.55-0.70 s + 49 min) = 14-16 h box 2, plus the 0.8 h pilot. The two-contrast term is cheaper than the four-contrast one (about +21 % positions against +42 %), so the low end is likely. Read: 10 checkpoints x about 2.2 h dev-box CPU = about 22 h.

## 6 MM rulings needed (conservative option taken, listed)
1. LR/warmup for the fine-tune: recipe peak 3e-4 taken; a lower peak is the alternative.
2. Seeds (section 2.4) and the order (chassis 0 complete first).
3. FT-NULL freezes `wm.proj` too (2.2).
4. Land base: a branch from `a13f6f16` (not the tip).
5. R35 checkpoints (45,266,310 bytes each) and the two probe files must be shipped to box 2 (inputs); the read side (IA-3P-K, C10') is not built.
