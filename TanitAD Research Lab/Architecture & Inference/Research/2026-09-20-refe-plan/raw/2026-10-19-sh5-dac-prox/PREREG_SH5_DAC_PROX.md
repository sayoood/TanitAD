# PREREG SH5-DAC-PROX: does P2's own `visual_ctx` carry the boundary-PROXIMITY information, so that a small proposal-anchored branch trained on cached ctx recovers it and sharpens DAC? (Architecture FlyWheel agent for the Master Mind, 2026-10-19)

**Written BEFORE any SH5 branch is trained on real data, any ctx is built, and this arm's label array (`sh5_labels.npz`) is written (PRIVGEOM's own labels were read for the three design calculations of section 0.2).** Its sha256 is in `PREREG_SH5_DAC_PROX.sha256` / `SEALED_AT.txt`; the code is pinned in `CODE_PINS.json` (section 13); every stage that can produce a number from real data (`prep --labels`, `fit`, `analyze`, `sh5_ctx_build.py run`) refuses unless both still hash (`sh5_guard.py`). A change comes only by a written amendment (`PREREG_SH5_DAC_PROX_AMENDMENT<k>.md`) saved and hashed before the number it governs. Sealed at 2026-10-19T13:39:48+02:00.

- Tier: SELECTION, SEEN scenes (`D-L1-G0-1`), P2 IN-SAMPLE (P2's frozen DAC row is in-sample on these fit sets; every SH5 arm is out-of-fold, so the comparison is biased AGAINST SH5). Label-lookup pick level. Not a driving result; no capability claim.
- ⛔ **INFERENCE STAYS VISION-ONLY.** The privileged clearance is a TRAINING TARGET only (PI rule 2026-08-03: labels may use privileged signals; inference is vision only). At inference the branch reads (a) P2's cached `visual_ctx` (image-derived) and (b) the proposal's own footprint points. It reads no ego state, no map, no privileged channel. The goal / route input is untouched, and nothing derived from the situation classifier enters (PI rule 2026-08-03: the goal path and the situation path stay information-disjoint).
- Registers read (GAC copy `landA/tip_GOALS_AND_CLAIMS.md`): GAC `D-REFE-EP-RELIAB-1`, `D-REFE-FASTNEG-CENSUS-1`, `D-REFE-HEADROW-PROBE-1`; `RETRACTION_LOG` R51 / R52 / R53; Lab report `2026-10-19-safety-head-lit/REPORT.md` section 2, 4.3 (SH5 / SH4).
- **Design FIXED by the Master Mind (brief 2026-10-19) and amended by the Master Mind's ruling of the same day** (size the power against about +0.03; every privileged target sign-clean, tested by a planted check). Everything the MM did not fix is marked **[AGENT]** and is open to a veto; every ADDITION beyond the brief is listed in section 14.

## 0. Disclosures (what I had seen before sealing)

1. PRIVGEOM-SAFE's whole result: the registered calls (DAC primary +0.161, DAC-nosign +0.079, both INFO-UPSTREAM), its pick-level read, and its descriptive section 5 (the nosign companion re-derives the verdict's sign through the pair `(|c_min|, |c_t|)`: precision 1.000, recall 88.7 %; the sign-clean 9-column read is +0.028 [+0.013, +0.043], centred +0.035; the registered C-NULL is not centred).
2. I used PRIVGEOM-SAFE's banked arrays (its join arrays, the `L` and `drop0` OOF logits, the labels it opened) for THREE design calculations, all before this seal and none an SH5 outcome: the sign-admissibility test of the target block with its descriptive MLP recomputer (section 3.3, `raw/atest.json`), the C-POWER plant calibration (section 7, `raw/plant_calibration.json`), and the minimum detectable effect (section 8, `raw/mde.json`). They read PRIVGEOM's labels, so the labels of THIS arm's analysis set are not unseen in aggregate (the P2 DAC row's F-AUC 0.827 on the full set is known); no SH5 prediction exists.
3. I know the local camera coverage: of the 2,782 PRIVGEOM tokens only **816 (29.3 %), in 38 of 41 logs, 1,444 of 4,941 sets**, have four resolvable local camera frames (`raw/plan/coverage.json`, identical to S0's own audit, `raw/plan/` vs `srttc_s0/code/srttc_s0_audit.py`). I have NOT built any ctx.
4. I know L1 Step A's step 3 timing (MEASURED: 400 tokens in 664 s, `l1run/step3.log`, `ZZBUNDLE_OK 400 664 s`) and that its attempt 2 died on the fp32 `visual_ctx` (fixed in L1's copy; the same fix is built in here).
5. I trained SH5 only on SYNTHETIC data (`sh5_synth.py`) for the CPU tests and the false-fire simulation.

## 1. Question and the one-line answer each outcome gives

`D-REFE-FASTNEG-CENSUS-1`: the DAC row has a fast-band deficit (-0.07 within-token AUC). `D-REFE-HEADROW-PROBE-1`: DAC is FEATURE-LIMITED on the 256-d head token. PRIVGEOM-SAFE: privileged boundary geometry beside the token closes the gap (+0.161 with the verdict's own sign test, +0.079 with the unsigned companion, **about +0.03 sign-clean**). **Question: does P2's own `visual_ctx` (7680 x 256 per token) carry that sign-clean proximity information, so that a small proposal-anchored branch (SH5) trained on the cached ctx recovers it and lifts the fast-band within-token DAC AUC beyond P2's DAC row, and does the lift move P2's own picks?**

## 2. Data and analysis sets (every fact below was fixed before any label of this arm was opened)

- The PRIVGEOM-SAFE subset: 41 logs, 4,941 S0 fit sets (f = 10 %), 2,782 tokens, the census folds (log rank mod 5; logs per fold 10 / 8 / 8 / 3 / 12), the census strata (`code`, sha256 `67f43e55dd7c693e31cd991efc3d51fedc278331ba4f8f882e6adbdae8203dc6`; **F = Q4**, Q1-3 = Q1 u Q2 u Q3), `labels6`, P2's stored logits `z6`, EP-R1's seed-matched OOF EP columns `ep_s0 / ep_s1`. Inputs `sh5_inputs.npz` (label-free: keys, folds, strata, `z6`, the key-time points, the sign-clean targets; sha256 `75445020987177ca26968d4e0cc64b4f65eda8bd60a2598cfe89149dd71a99be`, built by `sh5_train_eval.py prep` from PRIVGEOM's join arrays and S0's set stores, **all 4,941 sets joined by (log, token, step, rank, ckpt_step)**).
- **ANALYSIS SET (AS) = the sets whose token's ctx was built.** The ctx can only be built for tokens with four resolvable camera frames. Two variants share this prereg and every rule below:
  - **LOCAL**: the 816 tokens / 1,444 sets / 38 logs that have frames today (sets per fold 322 / 237 / 421 / 66 / 398). The frame zips are a *stratified sample* of the pod's pixels (`srttc_s0_audit.py`); the selection into LOCAL is therefore a FILE-PRESENCE filter, not a random sample, and what it licenses is a statement about these logs.
  - **FULL**: all 2,782 tokens / 4,941 sets / 41 logs, after the missing 1,966 tokens' frames are staged from the pod (a Master Mind decision: `raw/plan/needed_frames.json` lists the exact jpg names; about 1.5 GB ESTIMATED from the mean local frame size 193 KB x 4 cameras). **The decision thresholds do not change between variants; the achievable calls do (section 8).**
- Bootstrap logs are re-indexed contiguously over the AS (38 for LOCAL, 41 for FULL). Folds are the census folds of the retained logs. Each fold is predicted by ONE learner trained on the tokens of the other four folds; a token is never in its own fold's training set.
- **Per-token fidelity is NOT used to drop tokens** (no cherry-picking): every built token stays in the AS; the global fidelity gate C-FID (section 7) decides whether the ctx is the ctx S0 saw.

## 3. The branch, its inputs and its targets

### 3.1 ctx
`visual_ctx` [7680, 256] per token from P2's checkpoint (`D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt`), captured with `ctx_cache.VisualCtxTap` exactly as L1's `make_stepA_bundle.py` does (one `planner.infer` per token; the tap raises unless the scoring decoder's ctx IS `scene_proj`'s second output). ⚠️ The planner runs fp32 (no autocast): the ctx is **rounded explicitly to bf16** before `ctx_cache.to_u16` (L1's 2026-10-19 fix). Stored as uint16 bf16 bits, `ctx.npy` [N_tokens, 7680, 256], 3.75 MiB per token, on D: (`D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/`): 3.0 GiB LOCAL, 10.2 GiB FULL.

### 3.2 SH5 (`sh5_branch.py`)
Per proposal, 9 key-time footprint points (x, y, yaw) at t = 0, 0.5, ..., 4.0 s (the PRIVGEOM key times; interpolated from the 20-pose plan at 0.2 s, heading by unit vector, the A7 last-heading repair applied, t = 0 the ego origin). Each point -> MLP (Fourier features of x, y at 6 octaves + scaled x, y + cos / sin yaw) + a learned time embedding -> 2 layers of pre-LN multi-head cross-attention (d 256, 4 heads, FFN 512) into that token's ctx (LayerNorm on the ctx; K, V projected once per token) -> **pooling over the 9 points = LayerNorm of the time-ordered concatenation (9 x 256) then a learned Linear to 256 [AGENT: a time-indexed pooling keeps each key time's identity, which a mean pool would blur and the 9-value regression needs]** -> two heads: **(i)** regression of the 9 key-time |clearance| (metres, Huber delta 1 m), **(ii)** a DAC logit (BCE on the DAC label). No set-level attention (P2's scorer has set self-attention; SH5 deliberately does not). About 1.7 M parameters. **Both heads are zero-initialised, so an untrained branch emits exactly 0.0 and L+S = L (C-ID).** P2's trunk and rows are never touched (ctx is cached; nothing back-propagates into P2).

### 3.3 ⛔ Targets are SIGN-CLEAN, tested by a planted check (MM ruling 2026-10-19)
Targets: the **9 key-time |clearance|** (`g[..., 2:11]` of PRIVGEOM's scalars, capped at 10 m). **Excluded: `|dac_clear_min|` and `dac_frac_out`** — `c_min` is the minimum over all steps, so `c_t >= c_min` at every key step, hence for a DAC-safe proposal `|c_t| >= |c_min|` everywhere and any key step with `|c_t| < |c_min|` PROVES DAC = 0 (PRIVGEOM-SAFE RESULT 5.2). The regression head therefore has 9 outputs, and `L.clearance_targets` provably ignores columns 0 and 1 (test: the targets do not move when they move; a mutated version that includes `c_min` is caught).

**Admissibility test (A-TEST, `sh5_admit.py`): "could the DAC label be recomputed from this target?"** Answered by a SEARCH, with PLANTED controls, never by inspection. The recomputer searches 792 rules over the block's columns (unary thresholds, every ordered pair `C_a + m < C_b` with margins 0 / 0.25 / 0.5 / 1 / 2 m, V-dips `C_m + m < min(C_a, C_b)`, max / min aggregates) and reports precision and recall per class over ALL 316,224 proposals. **A block is NOT ADMISSIBLE iff some rule PROVES DAC-UNSAFE (precision >= 0.999 at recall >= 0.5 % of the unsafe class).** One-sided on purpose: "every key clearance is large, so the proposal is DAC-safe" is what proximity MEANS (the ego starts inside; a path that never nears the boundary cannot have left it) and is reported, not penalised; a proof of UNSAFE from unsigned magnitudes is the sign re-derivation. Controls: **POS-1** the registered 10-column DAC-nosign block must be flagged; **POS-2** the candidate plus one planted column `label x 5 + noise` must be flagged; **NEG** the candidate with every column permuted within each set must NOT be flagged. MEASURED (`raw/atest.json`): all three controls read right (POS-1: 150 unsafe-proof rules, best precision 1.000 at recall 77.6 % — `C8 + 0 < C0`, i.e. `|c_t| < |c_min|`; POS-2: flagged; NEG: not flagged); **the 9-column candidate is ADMISSIBLE: no rule reaches precision 0.999 at any recall >= 0.5 % of the unsafe class (792 rules searched; best UNSAFE proof: the V-dip `C5 + 0.5 < min(C4, C6)`, precision 0.884 at recall 3.8 %)**; it does prove SAFE for 31.6 % of the safe class (`min_k |c_t| > 2 m`, precision 1.000) — that is proximity, and it is reported, not penalised. DESCRIPTIVE, no authority: a flexible recomputer (2 x 64 MLP, census-fold OOF, within-set) reaches F-AUC **0.900 [0.866, 0.926]** from the candidate block alone (10-column known-leaky block: 0.974); the candidate is not information-free about the label (it is the clearance geometry the label is computed from), it carries no proof of unsafe; its admissibility is a statement about the SIGN path, not about information content.

### 3.4 Inference
The deployed model would be P2 + SH5; nothing privileged is read. The signed clearances exist in this package only inside the planted controls (C-REACH, C-POWER), which are controls of the INSTRUMENT, not arms.

## 4. Arms, recipe, folds

Per training seed in {0, 1}, 5 folds = the census fold of the log (train on the other four, predict the held-out one: OUT-OF-FOLD logits for every AS set). Common recipe **[AGENT, except where the brief fixed it]**: Adam lr **3e-4**, weight decay **1e-4** (coupled L2, as PRIVGEOM), betas (0.9, 0.999), eps 1e-8, 50-step linear warm-up then constant; **12 token passes** (FIXED epoch count; = 3 cycles x 4 inner passes when the ctx is sharded, 1 x 12 when it is resident); an optimizer step per chunk of **8 shard tokens restricted to the learner's training tokens** (about 6.4 tokens, 11 sets); per set **32 of the 64 proposals** sampled each step (all 64 at evaluation); loss = BCE(DAC logit, DAC label) + **0.1** x Huber(delta 1 m)(9 |clearance|). Batch by token. Trunk and P2 rows frozen. The stored P2 DAC row `z6[..., 1]` is **L** (not refit).

- **(L)** P2's DAC row.
- **(L+S)** `a x z_P2 + b x z_S`, a 2-parameter logistic (no intercept, BCE on the soft DAC label, Newton from (1, 0), ridge 1e-6 toward (1, 0)), fitted per held-out fold on the OOF predictions of the OTHER four folds ("leave-one-fold-out stacker"). **If std(z_S) < 1e-9 the answer is exactly (1, 0)** (C-ID). *Disclosed second-order path:* the models that produced the other folds' predictions DID train on the held-out fold, so its labels reach the two stacker weights through them; bounded by C-NULL-LABEL.
- **(S-noprox)** SH5 with the clearance-regression weight 0 (the reg head then receives exactly zero gradient). Separates "the proximity target helps" from "more parameters on ctx".
- **(S-shuf)** [ADDITION, section 14] SH5 with the SAME recipe but the ctx of ANOTHER token of the same fold that is resident in the shard (cyclic next token). Separates "reads THIS scene's ctx" from "re-learns a better DAC prior from the proposal geometry alone" (the branch sees the footprint points, which P2's logit already used).
- Controls' arms (7): **plant** (C-REACH), **plantweak** (C-POWER), **nullprox** (C-NULL), **nulllabel** (C-NULL-LABEL).

⚠️ **Lockstep trainer.** The ctx is 3.75 MiB per token and the dev box has an 8 GB RTX 4060 (about 1.6 GB held by the desktop) and a D: that read ~24 MiB/s in a cold probe. The trainer streams a shard (2.6 GiB, about 700 tokens) to the GPU once and trains EVERY learner of the stage (all folds x arms) on it (`sh5_train_eval.py`). Shards are re-drawn each cycle.

## 5. Reads and estimator

- **Within-token DAC AUC** (the census definition, A1.3): unsafe = label 0, safe = label 1 (0.5 excluded), pairs inside ONE set, both members in the stratum, score = the arm's OOF logit, ties 0.5. Strata from the census `code`: **F = Q4**, **Q1-3**.
- **Estimator**: paired LOG-cluster bootstrap over the AS's logs, B = 10,000, `numpy.random.default_rng(20260927).integers(0, n_logs, (10000, n_logs))`, the SAME draws for every quantity, percentile 2.5 / 97.5, every ratio a ratio of per-log sums, every difference on the same resampled logs. It answers "would another draw of LOGS say this?" and **nothing about another training run** (`H-ESTIM-SEED-1`) or another inference run; the two seeds are a floor on training variance, not an estimate of it.
- **Decision quantity**: `Delta_F,s = AUC_F(L+S) - AUC_F(L)`, one interval per seed; the same in Q1-3 (reported).
- **Pick level (the endpoint that matters for driving; DESCRIPTIVE for the call except PICK-GAIN below)**: P2's OWN rule (`pickrule_lib.rule_pick`, planner.py:558-561, float64) with the DAC column replaced by the stacked L+S logit: label-lookup PDMS x 100 and the DAC-unsafe pick rate, paired log-cluster differences to P2 (`rule[S]` vs `P2`). Secondary: the EP-led composite (column 2 = EP-R1's seed-matched OOF EP) with and without the replacement.
- **Reliability at the picks (R53; DESCRIPTIVE)**: at the picks of P2's rule and of `rule[S]`, observed DAC-unsafe mass / predicted mass (`1 - sigmoid(z_dac)` of the row at the pick), whole set and the F band, log-cluster CIs. A threshold share is NOT a calibration claim.
- **Attribution (DESCRIPTIVE, no authority over any branch)**: `AUC_F(L+S) - AUC_F(L+S-noprox)` (L cancels), paired; read against the empirical null of the C-NULL replicates (section 7), with the rank p-value and its resolution `1/(1+R)` stated.

## 6. Decision (fixed by the Master Mind) and the conversions

- **CTX-REACHABLE** iff `Delta_F` lower CI > **+0.02** on BOTH seeds. **NOT-REACHABLE** iff its upper CI < **+0.01** on BOTH seeds. Otherwise **INCONCLUSIVE** (two seeds that disagree are INCONCLUSIVE). (A degenerate zero-width interval at 0 satisfies the NOT rule: that is why NOT-REACHABLE needs C-POWER.)
- **PICK-GAIN** iff the P2-rule `dPDMS x 100` lower CI > 0 AND the DAC-unsafe-rate change upper CI <= 0, on BOTH seeds.
- **Conversions (they can only REMOVE a call, never create one):**
  1. **NO-CALL(GATE)** if C-ID, G-LIVE, C-FID or the RED table fails.
  2. **NO-CALL(C-REACH)** if the planted-clearance arm is not recovered.
  3. **VOID(C-NULL-LABEL)** if ANY machinery-null replicate fires.
  4. **UNDERPOWERED** (read as INCONCLUSIVE) if the call is NOT-REACHABLE and C-POWER (the +0.03 plant) is not detected on both seeds.
  5. **GAIN-NOT-FROM-CTX** if the call is CTX-REACHABLE and the ctx-shuffled arm S-shuf ALSO passes the CTX-REACHABLE rule on both seeds.
- Report (L+S) - (S-noprox) as the attribution to the proximity target (descriptive).

## 7. Controls (each built so it can fail; every one reported; false-fire SIMULATED on noise-trained replicates first — R51 / R52 / R53)

- **C-ID (identity).** (a) the branch emits exactly 0.0 at init (bitwise; mutation: a non-zero head breaks it); (b) the stacker returns exactly (1, 0) on a zero branch (mutation: removing the degenerate rule changes it); (c) END-TO-END on real tokens: `fit --epochs 0` takes no optimizer step, its OOF logits are exactly 0.0 on every AS set, `L+S == L` bit for bit, `Delta_F` = 0 with CI [0, 0] and `rule[S]` picks == P2 picks (`sh5_gates.check_cid` re-derives the exact zeros and the stacker identity from the arrays; the picks follow, and the CPU end-to-end test checks `dPDMS` = 0 on synthetic data).
- **G-LIVE (launch-gate style).** On the first real chunk: every loss term finite; every trainable parameter group has a gradient after the zero-initialised heads have moved (3 Adam steps on one batch; at step 1 the layers below the zero heads receive an EXACT zero by construction, tested); the ZERO CONTROL — `ctx_ln` frozen in a deep copy — reads **exactly 0.0** while the live copy's reads > 0; for `w_reg = 0` the reg head reads exactly 0.0 and nothing else is dead. RED mutations (detached pooling; a probe that always reads 0) are caught. `fit` refuses to take a step unless it passes.
- **C-FID (the ctx is the ctx S0 saw).** For every token the census logits `z6` of its S0 sets are recomputed from the CAPTURED ctx with the very call S0 made (`model.score_trajectories(traj3, sctx)`); the gate passes iff **>= 99 %** of the AS sets reproduce `z6` to **<= 1e-3** (max over the 64 x 6 logits). An early stop (rc 5) aborts the build if the first 10 sets reproduce at < 50 %. The bf16 rounding effect is measured (`bf16_maxabs`, diagnostic). *Reach* of the gate: a planted-wrong ctx (the next token's) fails it (tested through `fid_gate`).
- **C-REACH (planted ctx channel).** The last `n` ctx slots of each token are OVERWRITTEN with entries `[Fourier code of (x, y) | signed key clearance / 10 | 1]` for every (set, proposal, key time), built from the PRIVILEGED SIGNED clearance. The arm `plant` must recover it: **DAC-logit-alone F-AUC lower CI >= 0.95 AND held-out regression R^2 >= 0.50 on BOTH seeds.** [AGENT: the R^2 bar of 0.50 is set from the CPU synthetic (0.589-0.675 across two synthetic worlds at 40 epochs of a 3 k-proposal task, DAC AUC 0.987-0.995); the real task has 316 k proposals and 12 token passes of a 1.7 M-parameter branch, and the bar is a gate on a CONTROL, not on the finding.] Pre-flight: `fit --stage PRE --folds 0` (one learner, minutes) prints `ZZPREFLIGHT` before the 20-learner stage is paid for. A failed C-REACH is NO-CALL.
- **C-POWER (planted weak signal at +0.03; MM ruling).** Same plant with the signed key clearances plus `N(0, 1.15 m)` per entry (noise seeded by the token name). **Calibration (`raw/plant_calibration.json`, MEASURED on PRIVGEOM's banked arrays; P2's DAC row F-AUC 0.8267): sigma = 1.154 m is the noise at which an IDEAL reader (the leave-fold-out stacker on `min_k` of the noisy signed clearances) gains +0.0300 F-AUC over P2's DAC row (that scalar alone: F-AUC 0.766; for sigma 1.0 m the ideal gain is +0.040, for 1.5 m +0.017).** The branch reads it through attention and will not be ideal. **C-POWER passes iff the weak plant's `Delta_F` lower CI > 0 and its point lies in [+0.015, +0.045] on BOTH seeds** (it cannot be asked to clear the +0.02 lower-CI bar: see section 8).
- **C-NULL (clearance targets permuted within token; MM design).** 5 replicates, the 9-vector of every proposal replaced by another proposal's of the SAME token (permutation seed `20261019 + 7919 x rep`), DAC labels intact, weight 0.1. Replicate `r` runs on training seed `r mod 2` (3 on seed 0, 2 on seed 1; a conservative upper bound on the two-seed rule's false fire). **Read against the EMPIRICAL distribution of the replicates, never against 0 with a log-only CI (R52):** report each replicate's `Delta_F` and `AUC(L+S_null) - AUC(L+S-noprox)`; the real attribution's rank among them, `p = (1 + #{null >= real}) / (1 + 5)`, resolution 1/6. It is a null for the ATTRIBUTION (the DAC head still learns the true label), not for `Delta_F`.
- **C-NULL-LABEL (machinery null; ADDITION).** 3 replicates in which the branch trains on a DAC label permuted within each set AND permuted clearance targets (a noise-trained branch); the stacker and all evaluation use the REAL label. A replicate FIRES iff `Delta_F` lower CI > +0.02 (the CTX-REACHABLE rule). **VOID iff ANY replicate fires** (a fire under a noise-trained branch is a leak through the CV / stacker, not a finding).
- **A-TEST (target admissibility)**: section 3.3.
- **C-RED.** `sh5_lib.branch_table_failures()` over `seed_call`, `reach_call`, `pick_gain_*`, `null_label_*`, `power_ok`, `next_arm`, `final_outcome`, with 13 RED mutations (comparator flipped, threshold changed, `all` -> `any`, bound swapped, power band relaxed, conversion disabled) that each MUST make it fail (`test_sh5_lib.py`).

**False-fire simulation (R52: simulate it first, with TRAINING noise), `raw/falsefire_sim.json` (`sh5_sim_falsefire.py`, ESTIMATED on a synthetic world with the SAME code path):** 16 noise-trained replicates (the machinery null of the prereg: the branch trains on DAC labels permuted within set and clearance targets permuted within token; 20 epochs; CPU learning rate 2e-3, i.e. 10x the registered 3e-4, because the CPU world has ~25x fewer steps; 5 folds; T = 60 tokens) through the real Trainer / stacker / bootstrap (B = 10,000) / decision functions, read with EVERY registered rule, next to a POSITIVE control (4 replicates of the real branch, 40 epochs, in the world where the ctx DOES carry the information) so that "never fired" cannot mean "cannot fire":

| rule read on the replicate | noise-trained (n = 16) | branch that reads the ctx (n = 4) |
|---|---|---|
| CTX-REACHABLE per-seed rule (lower CI > +0.02), F | **0 / 16** (one-sided 95 % upper bound on the rate 0.171) | 4 / 4 (`Delta_F` +0.058 to +0.064) |
| same rule, Q1-3 | 0 / 16 | 4 / 4 (+0.056 to +0.058) |
| NOT-REACHABLE per-seed rule (upper CI < +0.01), F | **16 / 16** (a branch that learns nothing satisfies it: this is WHY the raw call needs C-POWER) | - |
| C-REACH (S-alone F-AUC lower CI >= 0.95 and R^2 >= 0.50) | 0 / 16 (S-alone F-AUC 0.385-0.505, R^2 0.004-0.287) | - |
| C-POWER (lower CI > 0 and point in [+0.015, +0.045]) | 0 / 16 | - |
| PICK-GAIN per seed | 0 / 16 | 3 / 4 |

`Delta_F` of the noise-trained replicates: mean -0.0006, sd 0.0006, range [-0.0016, +0.0008] (Q1-3: mean -0.0004, sd 0.0006); the stacker weights wandered (a 0.53-0.63, b 0.38-1.35) without moving the within-set AUC (a noise-trained branch can carry a set-level offset, which a within-set AUC cannot see). **Consequences registered in advance:** (i) the VOID conversion's per-replicate false-alarm rate is estimated 0 with a one-sided 95 % bound of 0.17, so the chance that ANY of the 3 label-null replicates fires is estimated 0 and bounded by 0.43 - a wide bound from n = 16; if a replicate does fire, the report says which interval it was and does not hide behind the estimate; (ii) the bootstrap interval is about 2x the noise-trained training sd here (sd / half-width 0.54), which is NOT evidence that it covers training variance on the real rig (`H-ESTIM-SEED-1`); (iii) the false-pass rates of C-REACH and C-POWER on REAL ctx cannot be simulated before a ctx exists: they are first measured by the PRE pre-flight and by stage A2, and a pass of either is read against the planted-channel design above, not against these CPU numbers; (iv) C-NULL (clearance permutation) has no false-fire rate of its own: it is a DESCRIPTIVE empirical null for the attribution whose resolution is 1/6, and no call depends on it. ⚠️ **A design finding from building this world (kept because it transfers):** the within-set AUC is blind to anything that shifts all 64 proposals of a set alike - my first synthetic world used a symmetric road half-width, which changes only the label threshold, and the ctx was then useless by construction (Delta +0.001). The information SH5 can add to the fast-band within-token AUC is the PROPOSAL-SPECIFIC part of the proximity - which of the 64 proposals approach which boundary - and the synthetic world that tests it has an asymmetric boundary (`sh5_synth.py`).

## 8. Power, MDE and the sizing (MM ruling: size against +0.03, not +0.16) — read this before reading any result

Measured on PRIVGEOM's banked sign-clean arm (`drop0`, +0.0276 [+0.0130, +0.0430]), `raw/mde.json`:

| analysis set | sets / logs | CI half-width at a +0.03 effect (se) | MDE of "lower CI > +0.02" | P(CTX-REACHABLE per seed \| true +0.03) | both seeds if independent |
|---|---|---|---|---|---|
| FULL | 4,941 / 41 | 0.0150 (0.0077) | **+0.035** | 0.26 | 0.07 |
| LOCAL | 1,444 / 38 | 0.0302 (0.0154) | **+0.050** | 0.09 | 0.01 |

⇒ **The committed bar (lower CI > +0.02) sits ABOVE what a true +0.03 effect clears: it would read INCONCLUSIVE most of the time (normal approximation; the two seeds share the logs and the bootstrap draws, so the true two-seed probability lies between the independent value and the per-seed value).** A true effect of +0.03 is CTX-REACHABLE only by luck; the MDE of the rule is +0.035 (FULL) / +0.050 (LOCAL). The half-widths above are read off the banked sign-clean ARM (effect +0.028); an interval around a TRUE ZERO effect is far narrower (PRIVGEOM RESULT 5.3: a centred null's interval is ~0.001 wide, and a useless branch adds ~0 within-set AUC: the 16 noise-trained CPU replicates of section 7 had intervals ~0.002 wide), so a branch that learns nothing can still be RAW-called NOT-REACHABLE even on LOCAL. **But that raw call converts to UNDERPOWERED unless C-POWER passes, and C-POWER needs the +0.03 plant's own lower CI > 0: with a half-width of 0.030 (LOCAL) an ideal reader's +0.03 puts that bound near 0, so on LOCAL C-POWER passes roughly half the time at best and a branch reading the plant imperfectly passes less often. On LOCAL the calls I expect to be returned are therefore INCONCLUSIVE and UNDERPOWERED; NOT-REACHABLE as a FINAL call needs FULL (half-width 0.015, plant lower CI ~ +0.015).** I did NOT move the bar (a goalpost fixed by the MM); the finding is stated here so that an INCONCLUSIVE is not read as a refutation, and the FULL variant (frame staging) is the lever that buys resolution. The point estimate and its interval are reported whatever the call.

## 9. Committed next arms (by outcome; fixed by the MM, extended only where marked)

| outcome | next arm |
|---|---|
| CTX-REACHABLE + PICK-GAIN | integrate SH5 into REFe's scorer: a launch-gate profile (G-HYG / G-DVB / G-LIVE for the new flags), a dev gate, then the shared navtest look; the EP-ADMISSION gate applies |
| CTX-REACHABLE, no PICK-GAIN | STOP at the diagnostic; report why the AUC gain does not move picks (reliability at the picks, the F-band composition) |
| NOT-REACHABLE | the information is not in `visual_ctx`; the trunk must learn it (P6): a proximity auxiliary loss on the trunk becomes the A40 arm draft |
| INCONCLUSIVE / UNDERPOWERED / NO-CALL / VOID / GAIN-NOT-FROM-CTX | named by the report; the first lever is FULL (stage the 1,966 tokens' frames, section 10) |

## 10. Compute, IO and the queue (ESTIMATED; the first 20 chunk-steps of `fit` print the MEASURED rate as `ZZTHROUGHPUT`)

- **ctx build**: MEASURED 1.66 s / token (L1 step 3: 400 tokens in 664 s, incl. the capture) -> LOCAL 816 tokens ~ **23 min**, 3.0 GiB; FULL 2,782 tokens ~ **77 min**, 10.2 GiB; plus a fidelity re-score per set (a decoder pass, negligible).
- **training** (`sh5_cost.py`, ESTIMATED at 3 / 6 / 10 TFLOPs effective on the RTX 4060; 38.7 GF per learner-token-visit): LOCAL stage A1 (20 learners: S, S-noprox, S-shuf, plant x 5 folds) **0.17-0.56 h per seed**; A2 (5 learners) 0.04-0.14 h; B (25 / 15 learners) 0.13-0.70 h per seed; PRE (1 learner) < 0.03 h. FULL: A1 0.58-1.9 h per seed, A2 0.14-0.48 h, B 0.43-2.4 h per seed, PRE 0.03-0.1 h. ctx IO (24 MiB/s, a cold probe; ESTIMATED): with the default 2.6 GiB shard LOCAL (3.0 GiB) is 2 shards x 3 cycles = three passes over the file, ~6 min per run; FULL (10.2 GiB, 4 shards) is three passes, ~22 min per run. Both are small beside the training time.
- **Order (priority; a killed job still yields value):** C-ID pre-flight (`--epochs 0`, both seeds) -> PRE (plant on fold 0) -> A1 seed 0, seed 1 -> `analyze` -> [A2 iff the call is NOT-REACHABLE or INCONCLUSIVE with an upper CI < +0.03] -> B (nulls; for the attribution and the VOID check; run iff A1 CTX-REACHABLE or INCONCLUSIVE). GPU jobs run only through `sh5_gpulock` (RAM gate, compute-type check, never removes a foreign lock, exit 75 if the GPU never frees). The Master Mind launches them; this agent launched nothing.

## 11. Forecast (a prediction, not a bar; HYPOTHESIS, committed before any number)

`Delta_F` centred near **+0.02** (range -0.01 to +0.05; the sign-clean linear read of privileged geometry is +0.028 / +0.035 and a vision-reachable read recovers a fraction of it). LOCAL (final outcome): INCONCLUSIVE 0.55, UNDERPOWERED 0.15, CTX-REACHABLE 0.08, GAIN-NOT-FROM-CTX 0.08, NO-CALL / VOID 0.12 (C-REACH is the main risk), NOT-REACHABLE 0.02. PICK-GAIN: 0.10. The S-shuf arm gains a little (the branch re-learns the DAC prior from the footprint): +0.00 to +0.01.

## 12. Limits stated in advance

SELECTION tier, SEEN scenes, P2 in-sample; one rig, two training seeds; the interval answers "another draw of LOGS" only; LOCAL is 29 % of the tokens by a file-presence filter (a stratified frame sample) and the sets of fold 3 are 66; the label-lookup PDMS is not driving; SH5 has no set-level attention while P2's rows do; the clearance targets are NAVSIM's simulated (LQR + bicycle) states, not the raw plan; the stacker's leave-fold-out leakage path (section 4); the C-NULL replicates run one seed each; evidence class of every number this arm produces: MEASURED, SELECTION tier, T0-class label lookup.

## 13. Code pins and environment

Interpreter `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe` (torch 2.7.1+cu128, numpy 1.26.4); landed modules loaded by sha256 (`fastneg_lib.py` `0c0b399b37a47ee3e17f9d5484bc7f9521ff7dbe6bea37e8c7a38042668667ca`, `pickrule_lib.py` `555c0a3468acd94adeeeb10f47527644a2b9d49722f554fed2da1fb29668eaeb`, `epr1.py` `a751c63eb8e38343a7ec1286114a158046ce4ec13390b2eea0a4b1d2237c38d8`); tree `par18/l1run/tree_hybrid` (the p2heldout tree's exact code + the tip's `ctx_cache.py`).

| file | sha256 |
|---|---|
| `code/sh5_lib.py` | `08781b86b7a21d016f29dd47ee82d029a7f50ec3d4070d10f33c5610abdada9e` |
| `code/sh5_branch.py` | `0f93dd4021b41e71310a93731dbb52245e5194f09a79d38c514ffc82c1b1e825` |
| `code/sh5_train_eval.py` | `828ca9a89199068049cf31c357cadb2b2665eb2c9b3239a84484a435e43d057d` |
| `code/sh5_ctx_build.py` | `142adfe998cca0acc6301b0a81dca4677713d3829a0eb0c12162b56752c437ed` |
| `code/sh5_gpulock.py` | `f93c7e4e2149822a10680ed5329e35494de90b856d4caa1eba13a00da5c28ce0` |
| `code/sh5_admit.py` | `c65ea1e8717e27b4ccc003378ba10c1b5cff47e1f4f7a5e463c5e6a4dbc090fb` |
| `code/sh5_gates.py` | `c6cf32eaed5279250cc60ef0c431e55a34e1bbd218fc50148dc0c5c84ebb837a` |
| `code/sh5_synth.py` | `909174edc7896af08561ff137d498c583ef7bc2c227e13147750355538e19d23` |
| `code/sh5_design_calc.py` | `c63ebbd540bc06fdc3fa2f7d396f5e64b87d97337f3902a502f9e2016a87fca2` |
| `code/sh5_cost.py` | `eee3ca7fb894495ce212ec67a042713bdb54625dcea00305a0c2743c7649fa31` |
| `code/sh5_sim_falsefire.py` | `015102fa6725ac90e9cb2555df807ef7dd6fc380043cdfe95aa7663c00a6987b` |
| `code/test_sh5_lib.py` | `f0ebf31f19d056a38a4788b34319c2514715b79c9817dd5b82b5650cd3778f7f` |
| `code/test_sh5_branch.py` | `971fe270dcb49a948d6608036a6315fdcc7961f26ffc670f43094fd3879798a9` |
| `code/test_sh5_ctx.py` | `32cd8e0491101299d9a6b5a3edf0948ee6cb0499cab869852b2e913387445fa6` |
| `code/test_sh5_pipeline.py` | `7fd71187529d6d6a438ca32976cb8f50f45f6f9073ba763da0ef0ea2b96895b1` |
| `code/run_tests.py` | `df9f0a9bb0b87a79dce37d89ec2a651646052c67fdc77ec977eb8f5f3a72f02e` |

## 14. ADDITIONS beyond the brief (each can only REMOVE a call; the MM may strike any)

1. **S-shuf** (ctx-shuffled arm) and the GAIN-NOT-FROM-CTX conversion: without it, a gain from the footprint points alone would be read as "ctx carries it".
2. **C-POWER** as a conversion of NOT-REACHABLE (PRIVGEOM's own precedent), sized at +0.03 by the ruling.
3. **C-NULL-LABEL** (3 noise-trained replicates) — the machinery null of the leave-fold-out stacker, because C-NULL (clearance permuted) is a null for the attribution, not for `Delta_F`.
4. **A-TEST** with planted controls (the ruling), and the seal guard + evidence-built gates (`sh5_guard.py`, `sh5_gates.py`).
5. The PRE plant-only pre-flight and the C-ID `--epochs 0` pre-flight (cheap, before the 20-learner stage).

## 15. Escalations

- **Frames for 1,966 tokens (1.5 GB ESTIMATED) are on the pod, not here** — a Master Mind decision (a pod read; `needed_frames.json` is S0's stager format). Without them SH5 is LOCAL and cannot call NOT-REACHABLE.
- **The committed CTX-REACHABLE bar is above the +0.03 the ruling sizes against** (section 8): a PI / MM decision whether to keep +0.02 or to register a different bar BEFORE any number exists.
