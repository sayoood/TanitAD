# PREREG_TRKSCORE: tracker-consistent selection on REFe C1 (plan L1, `H-REFE-TRKSCORE-1`)

REFe-line agent, 2026-11-26. **Frozen at 2026-11-26T20:33Z, before any TRK number exists.** At freezing, no TRK seam, smoke, pick or score had been computed, and the TRK code was not yet written. The only run so far was a synthetic import probe: the NAVSIM tracker on made-up poses, in the seam's venv, to check that the tracker can be imported there.

## 1. Question
C1 is REFe's best (MODEL_REGISTRY 14.6: **C1 92.5548 [91.7982, 93.2579]**, T1, 12,146 tokens).
Does C1's navtest tracked PDMS rise if each of its 64 proposals is scored on its **LQR-TRACKED** trajectory instead of its planned one? The heads stay C1's own, the aggregate stays the same, and the submitted trajectory stays the **planned** proposal at the chosen index.

**Motivation, disclosed as SNOOPED.**
- `D-REFE-SAFETYMISS-1` (MEASURED, T1, descriptive): on SCORER-FT C's 64-proposal tables, 1.31 PDMS points of the selection regret are tracker-induced safety misses. The pick passes the term as planned and fails only after NAVSIM's LQR re-simulation.
- `D-REFE-HEADATTR-1`: C1's tables, the same kind of observation.
- Both are navtest populations, so navtest is snooped for this question.
- The rule below has **zero parameters**, so nothing is fitted, there is no tune split, and the rule is read once on navtest under this card.

## 2. Arms

| arm | checkpoint (sha256, registry 14.6) | selection | submitted trajectory |
|---|---|---|---|
| **C1** (comparator) | `f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4` | the shipped planner pick | the banked seam `stage2_p5c_c1` (CSV `refe_stage2_p5c_c1.csv`, sha256 `7ae39f85…`) |
| **C1-TRK** | the same file | the rule in §2.1 | the planned proposal `k_trk` |
| **C0** (replicate comparator) | `58fe4aa309ceeb415de9716416a375cdd218eb452f51a5854703bb309ac88ea2` | the shipped planner pick | the banked seam `stage2_p5c_c0` (CSV sha256 `0071900c…`) |
| **C0-TRK** (training-free replicate) | the same file | the rule in §2.1 | the planned proposal `k_trk` |

The rule trains nothing, so the replicate runs across **checkpoints**, not seeds.
The seam is deterministic: registry 14.6 records a C1 re-run with max |dpose| 0.0 and pick agreement 1.0. CTRL-DET re-checks this on this run.

### 2.1 The rule, exactly (parameter count 0)
Per token, in the unchanged navtest seam (`eval/refe_navtest_seam.py`, tip export `F:/TanitAD_p5cread_tip` = `751f3ae7`). The seam runs under a launcher that patches `REFePlanner.infer` in-process (the `refe/ego_yaw.py` pattern; the yaw shim stays on).
1. The planner's **one** forward gives `traj [64, 20, 3]`, `score [64, 6]` and `k_ship`.
   - `traj` is in the REFe frame: rear axle at t0, x forward, y left, heading in rad, t = 0.2 … 4.0 s.
   - `sctx` is the scoring decoder's context, captured by a forward pre-hook on `model.score_dec[0]`. It is the same tensor the forward used.
2. `props8[j] = float32(to_navsim(planner.executed(traj, j)))` for j = 0 … 63. This is the exact form the harness tracks for a pick.
3. **The tracker is the harness's route** (`transp_score.score_arms` / `rl_terms_label.executed_states`), with the PDM-Closed row left out because it is privileged.
   - `navsim@3e8291b` `transform_trajectory(Trajectory(props8[j], 8 × 0.5 s), ego)`, then `get_trajectory_as_array(40 × 0.1 s)`, then `PDMSimulator.simulate_proposals` over the 64 (LQR + kinematic bicycle).
   - The start state is **the planner's own `EgoState` at t0**: the object whose kinematics already form the model's ego input.
   - Output: `[64, 41, 11]` states in the world frame.
4. `trk20[j, i]` is the simulated state at t = 0.2(i+1), state index 2(i+1), expressed in the t0 rear-axle frame.
   - Heading is placed on the planned heading's branch: `h = h_plan + wrap(h_sim − h_plan)`.
5. `s_trk = model.score_trajectories(trk20 → float32 [1, 64, 20, 3] on the model's device, sctx)` gives `[64, 6]`.
   - The set is complete (64), as `score_trajectories` requires.
   - `k_trk = argmax planner.aggregate(s_trk)`: the planner's own `navsim_v1` function, NC·DAC·(5 EP + 5 TTC + 2 C)/12, in float32 on the device.
6. The seam row is `to_navsim(planner.executed(traj, k_trk))`. **The tracked trajectory is never submitted.**

**Admissibility.**
- At inference the rule reads only the agent's own plan and its ego status at t0: pose (the origin), velocity, acceleration and steering angle. C1 already receives these through the NAVSIM agent interface.
- No map, agents, future, PDM-Closed trajectory or metric cache is read at inference.
- The tracker constants are NAVSIM's published ones.
- No goal input is added, so the goal/situation rule is untouched.

## 3. Tier, estimator, population
- **Tier T1:** NAVSIM v1.1 PDMS, LQR-tracked, open loop against logged agents. Scored by the pinned harness (`eval/harness`, `score_e.py`) through `eval/stage2_sft.py`.
- **Population:** all 12,146 navtest tokens, 136 logs.
- **Estimator:** paired log-cluster bootstrap (`p5c_read.Boot`, as in the C1r read; `taniteval/ci.py` family), **10,000 draws, seed 20260927**, 95 % percentile, on `full_set` means.
- It answers *"would another draw of LOGS say this?"*.
  - Training variance does not apply here.
  - Checkpoint variance is the C0 replicate's job.
  - Inference variance is checked by CTRL-DET.

## 4. Measures
- **M1** `Δ_C1 = PDMS(C1-TRK) − PDMS(C1)`, with its CI.
- **M2** The sub-score deltas NC, DAC, TTC, EP and C (judged), and DDC (reported only).
- **M3** The replicate: `Δ_C0 = PDMS(C0-TRK) − PDMS(C0)`, with sub-scores.
- **M4** The four families, paired TRK vs its comparator, per checkpoint. Each gets its CI and n, or a per-family reason and n where it cannot be computed.
  - LONGITUDINAL and LATERAL: `p5c_families.py` to `famci`.
  - TACTICAL: `p5c_tactical.py` command adherence, plus selection quality from the banked 64-proposal tables (regret to the oracle, top-1 rate).
  - **STRATEGIC in full** (PI 2026-11-16): `instruments/strategic_family.py read`.
  - Every separated-adverse family component is listed by name in the verdict.
- **M5** Descriptive only:
  - the switch rate (`k_trk ≠ k_ship`);
  - on switched tokens, the sub-score transitions read from the banked 64-proposal table;
  - the number of tracker-induced misses fixed and created, using the `D-REFE-SAFETYMISS-1` definition.
- **M6** Descriptive only: the transparent (no-tracker) Δ.
- **F_REFe** (C1r card §4(b)): bar P2 has a 2F term, so it uses `max(F, F_REFe)`.
  - F is this rig's seed floor; the only one measured on REFe is F_REFe = **0.1158** (one draw, 1 df).
  - So **2·max(F, F_REFe) = 0.2316**.

## 5. Committed reading (both outcomes written before the numbers)
**Bars.**
- **P1** CI-lo(Δ_C1) > 0.
- **P2** Δ_C1 point ≥ 2·max(F, F_REFe) = 0.2316.
- **G** (guard) No separated-adverse sub-score on C1: for none of NC, DAC, TTC, EP and C does CI-hi(TRK − C1) fall below 0.
- **R** (replicate) Δ_C0 point > 0, and no separated-adverse PDMS or sub-score (NC, DAC, TTC, EP, C) on C0.
- **All controls in §6 PASS.** Otherwise the outcome is **INVALID**, and every number is descriptive only.

**Outcomes and the next arm each one commits (Rule Zero).**
- **ADOPT** = P1 ∧ P2 ∧ G ∧ R.
  - TRK becomes REFe's selection rule: the planner option is wired in by a SPEC amendment that the MM lands.
  - C1-TRK is REFe's best, with a registry row. Goal status is printed as written: BELOW GOAL if the point is below 94.55.
  - **Next arm:** L2 SAFE-HEAD on top of TRK. Its training uses the TRACKED proposals as score queries (consistent with TRK) and tracked harness labels. It needs a GPU, which is a PI decision.
  - In parallel, at no GPU cost: the L3 EP-GEOM design inside the TRK-scored safe set, on navtrain tables.
- **SEPARATED-SMALL** = P1 ∧ G, but P2 or R fails.
  - Labelled *"separated, single checkpoint"* or *"inside the REFe seed floor"*. **Not adopted.**
  - **Next arm:** TRK-MIN (§7), then L2.
- **NULL** = the CI of Δ_C1 contains 0. **Not adopted.** **Next arm:** TRK-MIN (§7), then L2.
- **ADVERSE** = CI-hi(Δ_C1) < 0, or G fails.
  - **Not adopted.** The verdict names each separated-adverse sub-score.
  - **Next arm:** TRK-MIN (§7). By construction it is at least as conservative on every proposal as either set alone. Then L2.
- **In every outcome**, M5 is printed to explain the mechanism. Nothing is fitted on it.

## 6. Controls (each can fail; any FAIL = INVALID)
- **CTRL-ID** (every token, both arms): the new scoring path fed the **planned** set gives `k_id = argmax aggregate(score_trajectories(traj, sctx))`, in float32 on the model's device. It must equal `k_ship` on **12,146 / 12,146** tokens. The max |logit difference| is reported. (The `D-REFE-SRULE-1` dtype trap: no float64 recompute anywhere.)
- **CTRL-DET** (every token): the run must reproduce the banked dump of the same checkpoint bit for bit. This licenses the banked seam, CSV and 64-proposal tables as comparators.
  - `k_ship` equals the banked dump's `pick`.
  - The run's ship row, `float32(to_navsim(executed(traj, k_ship)))`, equals the banked seam row.
  - The run's 64 `props8` equal the banked dump's `proposals`.
- **CTRL-RED**: the tracker is re-run from a deliberately **wrong start state**: the rear axle shifted 1.0 m to the left (+y in the ego frame), everything else equal.
  - It runs on every smoke token, and on every full-run token with `sha1(token) mod 10 == 0`.
  - **PASS iff** (a) the RED tracked set differs from the TRK set at t = 0.2 s by ≥ 0.5 m on every RED token (the wrong state reached the tracker), **and** (b) `k_red ≠ k_trk` on ≥ 1 % of RED tokens (the picks read the tracked set).
  - A path that ignores the tracked set FAILS (b).
- **CTRL-T** (every token):
  - (a) The seam rows equal float32 `to_navsim(pick20)` bit for bit.
  - (b) The harness CSV equals the `transp_score` pass-1 tracked rows to 1e-6 (`p5c_read_prep.py identity`), for both arms.
  - (c) The TRK harness CSV equals the banked 64-proposal table `tr` at `k_trk` (`full_C1_s*of2.jsonl` / `full_C0_s*of2.jsonl`) to 1e-6. This requires CTRL-DET.
- **CTRL-SET**: 12,146 rows, 0 misses, and tokens identical to the comparator seam. The frame-control max must equal the declared **6.18063877792441 m** (the known Deviation-1 set, model-independent; `p7_read_chain.py`).
- **CTRL-SHA**: both model files equal the sha256 values in §2.
- **CTRL-PARITY** (the smoke's 50 tokens): this licenses both "the tracker exactly as the harness runs it" and the start state's admissibility.
  - The in-seam tracked states (seam venv, the planner's `EgoState`) are compared, for all 64 proposals, with the harness route: navsim-crun venv, the harness `MetricCache.ego_state`, PDM-Closed in row 0, as `score_arms` runs it.
  - Bars: max |Δx|, |Δy| ≤ 1e-4 m; |Δheading| ≤ 1e-4 rad.
  - The start state arrays must be equal to ≤ 1e-9.
- **CTRL-CODE**: the sha256 of every TRK file is recorded in the run output and must equal the staged files. Tip export `751f3ae7`.
- **Pre-run unit tests**, which must pass before the smoke:
  - identity: `score_trajectories` on the planned set equals the forward's scores bit for bit, with the same pick, on a small REFe on CPU;
  - frame round-trip ≤ 1e-9;
  - the heading branch;
  - RED: a wrong start state changes the tracked set, and a scorer that reads the set changes the pick;
  - completeness: refuse any set other than 64;
  - the submitted trajectory is the planned one.

## 7. TRK-MIN: the committed next arm (specified now, ⛔ NOT computed in this card's read)
- The rule: `agg_min(j) = min(agg_planned(j), agg_tracked(j))`, `k_min = argmax agg_min`. Zero parameters.
- It is computed from the two logit sets that the TRK dumps already hold, so it needs no new seam.
- Its PDMS comes from the banked 64-proposal tables (under CTRL-DET), with a harness confirmation on its own seam rows.
- It is read under an amendment frozen before its numbers, with the bars of §5 and the C0 replicate.

## 8. Run plan and tools
1. **Smoke:** 50 tokens (`p5c_next/propscore/smoke_tokens50.json`) on C1.
   - It reads controls and timing only: CTRL-ID, CTRL-DET, CTRL-RED, CTRL-PARITY.
   - ⛔ No PDMS is read.
2. **Full run:** C1-TRK, then C0-TRK, one after the other.
   - The seam runs on the dev-box GPU under `devbox_gpu.lock`.
   - It is gated on `scorerft_read/markers_readp/read.DONE`, on available RAM ≥ 4,500 MB (5 samples) and on commit free ≥ 6 GB.
   - The harness then scores the seam through `stage2_sft.py`: the seam exists, so it only scores, behind its own RAM guard.
3. **Read:** `trk_read.py` is written after this freeze, following the `c1r_read.py` pattern.
   - Its only freedom is mechanics. It prints every measure in §4 and every control in §6, and nothing chosen after the numbers.
   - `GOALS_AND_CLAIMS.md` `H-REFE-TRKSCORE-1` is updated in the same turn as the read.
