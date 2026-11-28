# PREREG_ENSSCORE: an equal-weight two-seed ensemble of REFe's scoring heads (plan L1b, `H-REFE-ENSSCORE-1`)

REFe-line agent, 2026-11-26. **Frozen at 2026-11-26T21:19Z, before any ENS number exists.**
State at freezing:
- The ENS code is written. Its 8 synthetic unit tests (tiny random REFe models, CPU) PASS, and two deliberate mutations turned them RED (§6).
- **No real navtest token has been run through the ENS path.** No smoke, pick, switch count or score exists. The smoke is stage 2 of the chain (§9), after this freeze.
- C1r was pulled from HF and its sha256 verified against MODEL_REGISTRY 14.6. Nothing was computed from it.

## 1. Question
C1 is REFe's best (MODEL_REGISTRY 14.6: **C1 92.5548 [91.7982, 93.2579]**, T1, 12,146 tokens).
Does C1's navtest tracked PDMS rise if its 64 planned proposals are scored by the **equal-weight mean of C1's and C1r's per-term head probabilities**? The planner's own aggregate is applied to that mean, and the **planned** proposal at the argmax is submitted.

**Motivation, disclosed as SNOOPED.**
- `PK/raw/2026-11-26-refe-next/PLAN_REFE_9455.md` §1–2 (MEASURED, T1, descriptive, on SCORER-FT C's 64-proposal tables): the scorer pick reaches 92.67 and the best of 64 reaches 99.35. About 2.9 pts of the gap are **unsafe as planned**: the safety heads are wrong at the pass/fail boundary (`D-REFE-HEADATTR-1`).
- `H-ESTIM-SEED-1-REFE` (registry 14.6): C1r − C1 = +0.1158 [−0.1783, +0.4007] (R-NULL), yet seed alone separates 21 of 70 family components. Two seeds' heads disagree without either being better on PDMS. That is the condition under which an equal-weight ensemble can improve decisions at the boundary.
- Both are navtest populations, so navtest is snooped for this question.
- The rule has **zero parameters**: nothing is fitted, there is no tune split, and it is read once on navtest under this card.

## 2. Arms

| arm | primary: proposal set and submitted trajectory | heads (members) | comparator |
|---|---|---|---|
| **C1** | C1 `f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4` | C1 | the banked seam `stage2_p5c_c1` (CSV `refe_stage2_p5c_c1.csv` sha256 `7ae39f85d3c3dc875ccbbb43036f59b03fe953d09f47356ad134dd30e9a49d25`) |
| **ENS1** | C1 | C1 + C1r `6e39a57c9a2e3ead0fef2b0d00d1684a708c901b6602ffe0717444f6180654a8` | C1 |
| **C1r** | C1r | C1r | the banked seam `stage2_p5c_c1r` (CSV `refe_stage2_p5c_c1r.csv` sha256 `246c694142a8a41e3e74d38a0253cccde7ccf327a2bb96b2c1b53950d90a9acc`) |
| **ENS1r** (replicate) | C1r | C1r + C1 | C1r |

**What the replicate can and cannot say.**
- ENS1r uses the **same two head sets** as ENS1, on the other model's proposals, against the other comparator. So it replicates across proposal sets and comparators, not across ensembles.
- It controls the cheapest alternative explanation: if ENS1 beats C1 only because C1r's heads are better, ENS1r cannot beat C1r.
- An independent ensemble replicate needs a third seed (C1r2). That is training, so it is a PI decision (§5).

### 2.1 The rule, exactly (parameter count 0)
Per token, in the unchanged navtest seam (`eval/refe_navtest_seam.py`, tip export `F:/TanitAD_p5cread_tip` = `751f3ae7`). The seam runs under `ens_select.py`, which patches `REFePlanner` in-process (the `refe/ego_yaw.py` / `trk_select.py` pattern). The yaw shim stays on for both models.
1. **The primary's one forward** gives `traj_A [64, 20, 3]`, `score_A [64, 6]` and `k_ship`.
   - `sctx_A` is captured by a forward pre-hook on `model.score_dec[0]`.
   - The forward's exact inputs (camera images, the 7-D ego vector, the NAVSIM driving-command one-hot, the per-sample calibration) are captured by a forward pre-hook **with kwargs** on the primary model. That hook runs above the primary's yaw shim, so the captured ego is the raw one.
2. **The member B** is built once, by `REFePlanner`'s own `__init__`, with the same keyword arguments except `checkpoint`.
   - Its config and selection settings must equal the primary's (CTRL-CFG).
   - It gets its **own** `EgoYawShim`: from its own checkpoint meta, the same navtest table, and the token from the primary's scenario.
   - Its own forward on the captured inputs gives `traj_B`, `score_B`, `k_b_own` and `sctx_B`.
3. **Scoring.** `s_A = A.score_trajectories(traj_A, sctx_A)` and `s_B = B.score_trajectories(traj_A, sctx_B)` (`refe/model.py:699`). Both are `[64, 6]` on the **complete set A**, each with its own context.
4. **The combination.** `p_A = sigmoid(s_A)`, `p_B = sigmoid(s_B)`, and **`p_ens = (p_A + p_B) / 2`**, per term, float32 on the model's device (`ens_lib.ens_probs`).
5. **The pick.** `k_ens = argmax planner.aggregate(p_ens)`: the planner's own `navsim_v1` rule, NC·DAC·(5 EP + 5 TTC + 2 C)/12.
   - It is computed through `ens_lib.aggregate_p`, which hands the averaged probabilities to the planner's own `aggregate` function, so the rule is not re-implemented.
   - The unit tests pin that this equals `aggregate(logits)` bit for bit when the probabilities are `sigmoid(logits)`.
6. **The submission.** The seam row is `to_navsim(A.executed(traj_A, k_ens))`. **B's proposals are never submitted.** The seam's proposal dump keeps A's set and A's own logits, with `pick = k_ens`.

**Why per-term probabilities are averaged.**
- The heads are trained per term with binary cross-entropy on logits (`refe/train.py:1919`), so each output is a Bernoulli logit for its term.
- `REFePlanner.aggregate` (`refe/planner.py:537-575`) consumes `sigmoid(logit)` per term. It multiplies NC and DAC and takes a weighted mean of EP, TTC and C.
- The mean of the members' probabilities is the mixture predictive probability of each Bernoulli term, which is the standard deep-ensemble combination. The planner's own aggregate is then applied to it unchanged.
- A member ensembled with itself is that member bit for bit, because `(x + x) / 2 == x` in IEEE float. CTRL-ID (a) checks this.
- **Alternatives not read under this card:** a logit mean, a mean of the two aggregates, and max or min pooling. No number for them is printed. The only exception is ENS-MIN, the committed next arm in §7.

**Admissibility.**
- At inference the rule reads only what C1 already reads through the NAVSIM agent interface: camera images, ego kinematics at t0 (with the navtest yaw table, the inference input both models were trained on) and the driving command.
- The member receives exactly the primary's inputs.
- No map, agents, future, PDM-Closed trajectory or metric cache is read.
- No goal input is added, so the goal/situation rule is untouched.
- **Cost:** inference runs two trunks, about twice the compute. It is recorded, not judged here (PI ruling: latency later, results first).

## 3. Tier, estimator, population
- **Tier T1:** NAVSIM v1.1 PDMS, LQR-tracked, open loop against logged agents. Scored by the pinned harness (`eval/harness`, `score_e.py`) through `eval/stage2_sft.py`.
- **Population:** all 12,146 navtest tokens, 136 logs.
- **Estimator:** paired log-cluster bootstrap (`p5c_read.Boot`, as in the C1r read), **10,000 draws, seed 20260927**, 95 % percentile, on `full_set` means.
- It answers *"would another draw of LOGS say this?"*.
  - Training variance enters through the members. ENS1r is the replicate across proposal sets; an independent ensemble needs C1r2.
  - Inference variance is checked by CTRL-DET-RUN.

## 4. Measures
- **M1** `Δ = PDMS(ENS1) − PDMS(C1)`, with its CI.
- **M2** The sub-score deltas NC, DAC, TTC, EP and C (judged), and DDC (reported only).
- **M3** The replicate: `Δ_r = PDMS(ENS1r) − PDMS(C1r)`, with sub-scores.
- **M4** The four families, paired ENS vs its comparator, per arm. Each gets its CI and n, or a per-family reason and n where it cannot be computed.
  - LONGITUDINAL and LATERAL: `p5c_families.py` to `famci`.
  - TACTICAL: `p5c_tactical.py` command adherence, plus selection quality from the banked C1 64-proposal tables (regret to the oracle, top-1 rate; C1's set only, because no C1r table exists).
  - **STRATEGIC in full** (PI 2026-11-16): `instruments/strategic_family.py read`.
  - Every separated-adverse family component is listed by name in the verdict.
- **M5** Descriptive only:
  - the switch rate (`k_ens ≠ k_ship`) and the member-alone switch rate (`k_b_only ≠ k_ship`, with `k_b_only = argmax aggregate(p_B)` on A's set);
  - on switched tokens, the sub-score transitions read from the banked C1 64-proposal table (`full_C1_s{0,1}of2.jsonl`, sha256 `9d1239a2…`, `77588849…`);
  - the unsafe-as-planned misses fixed and created, using the `D-REFE-HEADATTR-1` definition;
  - the table PDMS of `k_b_only` on C1's set. This splits the effect into "other heads" and "averaging". ⛔ It can never be adopted: keeping one seed's heads is selection on noise (C1r card §4(c)).
- **M6** Descriptive only: the boundary calibration of `p_A`, `p_B` and `p_ens` for NC, DAC and TTC, against the banked C1 table's tracked sub-scores on all 64 proposals. Reported as the Brier score per term.
- **F_REFe** (C1r card §4(b)): the only seed floor measured on REFe is F_REFe = **0.1158** (one draw, 1 df), so **2·max(F, F_REFe) = 0.2316**.

## 5. Committed reading (both outcomes written before the numbers)
**Bars.**
- **P1** CI-lo(Δ) > 0.
- **P2** Δ point ≥ 2·max(F, F_REFe) = 0.2316.
- **G** (guard) No separated-adverse sub-score on ENS1: for none of NC, DAC, TTC, EP and C does CI-hi(ENS1 − C1) fall below 0.
- **R** (replicate) Δ_r point > 0, and no separated-adverse PDMS or sub-score (NC, DAC, TTC, EP, C) on ENS1r vs C1r.
- **All controls in §6 PASS.** Otherwise the outcome is **INVALID**, and every number is descriptive only.

**Outcomes and the next arm each one commits (Rule Zero).**
- **ADOPT** = P1 ∧ P2 ∧ G ∧ R.
  - ENS becomes a named REFe selection option: a SPEC amendment that the MM lands.
  - ENS1 gets a registry row. Goal status is printed as written: BELOW GOAL if the point is below 94.55.
  - **Next arms:**
    - (i) ENS-TRK (§8) under its own card, if TRK reads at least SEPARATED-SMALL;
    - (ii) the third seed C1r2, for an independent ensemble replicate and a 3-member ensemble (training: a PI decision);
    - (iii) L2 SAFE-HEAD (GPU: a PI decision).
- **SEPARATED-SMALL** = P1 ∧ G, but P2 or R fails.
  - Labelled *"separated, single ensemble"* or *"inside the REFe seed floor"*. **Not adopted.**
  - **Next arm:** ENS-MIN (§7), then L2.
- **NULL** = the CI of Δ contains 0. **Not adopted.** **Next arm:** ENS-MIN (§7), then L2.
- **ADVERSE** = CI-hi(Δ) < 0, or G fails.
  - **Not adopted.** The verdict names each separated-adverse sub-score.
  - **Next arm:** ENS-MIN (§7). By construction it is at least as conservative on every safety term as either member. Then L2.
- **In every outcome**, M5 and M6 are printed to explain the mechanism. Nothing is fitted on them.

## 6. Controls (each can fail; any FAIL = INVALID)
- **CTRL-SHA** (launcher, before the GPU lock): both model files equal the sha256 values in §2. The launcher refuses a run whose member is the primary.
- **CTRL-CFG** (launcher):
  - the member's `REFeConfig` equals the primary's (dataclass equality);
  - `rule`, `select`, `repair_last_heading`, `per_sample_calib` and `trained` are equal;
  - both models use the ego-yaw shim, or neither does, on the same table (sha256 checked).
- **CTRL-ID** (every token, both arms):
  - (a) the primary ensembled with **itself**, `argmax aggregate((p_A + p_A)/2)`, equals `k_ship` on **every** token. The max |logit difference| between `s_A` and the forward's own `score_A` is reported. There is no float64 recompute anywhere (the `D-REFE-SRULE-1` dtype trap).
  - (b) the member's re-score of its **own** set with its own context, `argmax aggregate(B.score_trajectories(traj_B, sctx_B))`, equals `k_b_own` on every token. This proves `sctx_B` is the member's own context.
  - ⚠️ **What CTRL-ID cannot see:** the member term. Mutation 1 (below) removed it, and the self-ensemble test still PASSED. CTRL-RED and CTRL-REACH are the controls that see it.
- **CTRL-DET** (every token): the primary must reproduce its banked seam **bit for bit**. This licenses the banked seam, CSV and 64-proposal tables as comparators.
  - `k_ship` equals the banked `pick`.
  - The ship row equals the banked seam row.
  - The run's 64 planned proposals equal the banked `proposals`.
  - The run's forward logits equal the banked `logits`.
- **CTRL-DET-B** (every token): the member's own forward must reproduce **its** banked seam (`stage2_p5c_c1r` in ENS1, `stage2_p5c_c1` in ENS1r).
  - Bar: max |Δ logits| ≤ 1e-4 and max |Δ proposals| ≤ 1e-4 m.
  - The bitwise token counts, the pick agreement and the ship-row agreement are reported.
  - This proves the member received the primary's exact inputs: image, ego plus its own yaw, command and calibration. A wrong input moves logits by far more than 1e-4.
- **CTRL-DET-RUN**: the smoke runs **twice**, and every shared token must be bit-identical in `k_ship`, `k_ens`, `k_b_own`, `s_A`, `s_B` and `a_ens`. The full ENS1 run is compared with the smoke on its 50 tokens by the same test.
- **CTRL-RED** (every token): the member's probabilities misaligned by **one proposal index** (`torch.roll(p_B, 1, 0)`) before the mean must change the pick (`k_red ≠ k_ens`) on ≥ 1 % of tokens. A path that ignores the member FAILS.
- **CTRL-REACH** (every token):
  - (i) the dumped device aggregate `a_ens` equals an **independent** numpy float32 recompute of the navsim_v1 rule on `(sig(s_A) + sig(s_B))/2`, to ≤ 1e-6;
  - (ii) `argmax a_ens = k_ens` on every token;
  - (iii) on ≥ 99 % of tokens, `a_ens` differs from the primary-alone aggregate by more than 1e-5 somewhere in the set.
- **CTRL-SUB** (every token): the seam row equals `props[k_ens]` and `to_navsim(pick20)` bit for bit, and the dump's `pick` equals `k_ens`.
- **CTRL-SET / CTRL-SEAM**:
  - 12,146 rows and 0 misses;
  - tokens identical to both banked seams;
  - rule `navsim_v1` with A7 on;
  - the frame-control max equals the declared **6.18063877792441 m** (the known Deviation-1 set);
  - the seam log shows `device=cuda`.
- **CTRL-T**:
  - (b) the harness CSV equals the `transp_score` pass-1 tracked rows to 1e-6 (`p5c_read_prep.py identity`), for all four arms;
  - (c) the ENS1 harness CSV equals the banked C1 64-proposal table at `k_ens` to 1e-6. This requires CTRL-DET. There is no C1r table, so (c) is **n/a for ENS1r**, by name.
- **CTRL-CODE**: the sha256 of `ens_lib.py` and `ens_select.py` recorded in each dump equals the staged files (`ens_ctrl.py`). At the read, every ENS file equals `CODE.sha256`. Tip export `751f3ae7`.
- **Pre-run unit tests** (`test_ens_score.py`, `run_tests.sh`): **8/8 PASS at freezing**.
  - aggregate-on-probabilities bitwise;
  - self-ensemble bitwise;
  - the member term reaches the pick;
  - RED changes the pick;
  - completeness, shape and member-config refusal;
  - the patched infer end to end: captured inputs including the calib kwarg, the member's own forward equal to its direct forward bit for bit, and the submitted trajectory equal to the primary's planned proposal `k_ens`;
  - the member's own yaw shim;
  - the controls' numpy recompute.
  - **Mutation 1** (`ens_probs` returns `p_A`, the member term removed): 4 tests FAIL (member-reaches-pick, RED, end-to-end, recompute). The self-ensemble identity still PASSES, which is the blind spot noted under CTRL-ID.
  - **Mutation 2** (the member's yaw shim not installed): the yaw test FAILS.

## 7. ENS-MIN: the committed next arm (specified now, ⛔ NOT computed in this card's read)
- The rule:
  - for the safety terms NC, DAC and TTC: `p_t = min(p_A,t, p_B,t)`;
  - for EP and C: `p_t = (p_A,t + p_B,t)/2`;
  - `k_min = argmax aggregate(p)`.
- Zero parameters.
- It is computed from the dumps' `s_a` and `s_b`, so it needs no new seam.
- Its PDMS comes from two routes:
  - on C1's set, the banked 64-proposal tables (under CTRL-DET), confirmed by the harness on rows taken from the ENS1 run's proposal dump at `k_min`;
  - on C1r's set, harness scoring of rows taken from the ENS1r proposal dump.
- It is read under an amendment frozen before its numbers, with the bars of §5.

## 8. ENS-TRK: the composition with TRK-SCORE (a NAMED FUTURE ARM, ⛔ not computed and not registered here)
- The rule: score each proposal's LQR-tracked trajectory (`PREREG_TRKSCORE.md` §2.1, steps 2–4) with **both** members, each with its own context. Average the per-term probabilities as in §2.1, apply the planner's aggregate, and submit the **planned** proposal at the argmax.
- It needs a new seam pass, because the TRK dumps hold no member context.
- Its card is frozen only after both the TRK and the ENS reads exist.

## 9. Run plan and tools
1. **Gate:** the TRK chain's `E:/Projects/TanitAD-artifacts/trkscore/work/markers/chain.DONE`. TRK holds the GPU and RAM first. This chain reads no TRK result and touches no TRK file.
2. **Pull:** C1 from HF `Sayood/tanitad-refe-p5c` `p5c/c1/model_final.pt`, sha-verified, with `ens_pull.py`. The TRK chain deletes its own copy.
   - C1r was pulled from `p5c/c1r/model_final.pt` on 2026-11-26 and verified (`6e39a57c…`, 1,288,515,000 B).
3. **Smoke:** 50 tokens (`p5c_next/propscore/smoke_tokens50.json`, sha256 `1b6288b8…`), ENS1, GPU under `devbox_gpu.lock`, **run twice**.
   - `ens_ctrl.py` reads controls only: SET, ID, DET, DET-B, RED, REACH, SUB, SEAM, CFG, CODE and DET-RUN. ⛔ No PDMS is read.
   - **Any smoke control FAIL stops the chain before the full run.** The only admissible change after a failed smoke is an amendment frozen before any PDMS.
4. **Full run:** ENS1, then ENS1r, one after the other, each on the dev-box GPU under the lock.
   - Gates: available RAM ≥ 4,500 MB on 5 consecutive 1-min samples, and commit free ≥ 6 GB, before every seam and every score attempt.
   - The harness then scores each seam through `stage2_sft.py` (the seam exists, so it only scores).
5. **Read:** `ens_read.py` is written after this freeze, following `c1r_read.py`. Its only freedom is mechanics: it prints every measure in §4 and every control in §6, and nothing chosen after the numbers. `GOALS_AND_CLAIMS.md` `H-REFE-ENSSCORE-1` is updated in the same turn as the read.
