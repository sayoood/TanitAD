# SCORER_GYM_DESIGN: a scorer-only training environment for REFe (stream SCORER-GYM, 2026-11-07, DRAFT)

**Status.** This is a design. Nothing was built, run or staged. It needs PI decisions on compute: pod GPU, and multi-day dev-box CPU.

**Evidence classes.**
- **MEASURED** = read in source at the mirror tip `6d0e629f` (2026-11-07 19:34) or on disk today.
- **INHERITED** = a register row or a banked review.
- **ESTIMATED** = arithmetic here.

The literature basis is `LIT_REVIEW.md`.

---

## 0 · The design in one paragraph

The gym has two stages, and the cheap one is mostly built already.

**Stage A (one-step, full information)** is what REFe already trains on. Every one of the 64 proposals gets its true NAVSIM sub-scores from a 4 s non-reactive simulation, so the "RL" problem is a contextual bandit with every arm observed. EPO and listwise ranking are then the same gradient family (`LIT_REVIEW.md` §0.1).
- ⇒ No RL algorithm is needed there. The gaps are elsewhere:
  - **what the objective penalises**: the speed-as-danger proxy, which BCE plus aggregate listwise does not;
  - **which behaviour terms the label carries**: time gap, phantom braking, stall, oscillation.

**Stage B (multi-step)** is the genuinely new part. It runs a privileged closed loop in the nuPlan simulator:
- state = ego + agents + map;
- action = one of K proposals per 0.5 s cycle;
- reactive or log-replay agents;
- CaRL/DriveZero-style multiplicative reward.

In Stage B a **privileged teacher scorer** learns multi-step Q values by rollout policy iteration. It is then distilled into REFe's **image scorer** on REFe's own proposals at log states.

⛔ **Order of work:** Stage A's discriminating experiment (`PREREG_DRAFT_SCORER1.md`) runs first. It is the cheapest and directly targets `D-REFE-SPEEDPESS-1`.

---

## 1 · What already exists (MEASURED unless marked)

### 1.1 Label banks and scorer-training paths (navtrain only)

| what | where (file:line at tip `6d0e629f`, path prefix `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/`) |
|---|---|
| fixed six-component PDM targets | `train.py:2098` `--scorer-targets`, default `D:/Projects/TanitAD/data/refe_scorer_targets_full` (from `build_scorer_targets.py`). A local copy dir exists at `E:/Projects/TanitAD/data/refe_scorer_targets_full`; its contents were not inspected. |
| fixed vs on-policy supervision | `train.py:2190` `--scorer-mode fixed\|onpolicy`; `:2194` `--onpolicy-targets` (from `onpolicy_label.py`) |
| **true PDM labels on executed plans** | `train.py:2202` `--pdm-labels` (pdm_*.jsonl from `onpolicy_relabel_pdm.py`); `:2205` `--pdm-only` |
| **online labeller** (P6 D1) | `train.py:2213` `--labeller bank\|online`; `:2216` `--label-backend pdm\|synthetic`; `:2219` `--label-workers` (help: *"12 workers sustain >= 2x the 4.49 sets/s needed"*, INHERITED); `:2225` `--metric-cache` (D9 parts). BCE on the labels at `:1887-1890`. |
| **scorer-only fine-tune** (frozen generator, so the 64 proposals stay bit-identical) | `scorer_finetune.py`: `--onpolicy :1128`, `--heldout :1129`, `--pdm-labels-train :1151`, `--require-pdm :1153`, `--b-mode listnet\|compw\|expreward :1147`, `expected_reward_loss :113-119`, `--tau-s :1158`, `--epo :1161-1172`, `--epo-con g5 :1178` (SAFEOBJ) |
| **u400's own recipe** | `sft4_pod.sh:10`: `--onpolicy /workspace/data/refe_navtrain/onpolicy/sets --heldout /workspace/data/refe_heldout/sets --lr 1e-4 --b-mode expreward --lam 1.0 --lane-head --lane-weight …`, plus `--pdm-labels-train $D/train_pdm --require-pdm` (`:16`) |
| reward terms already coded | `rl_reward.py`: `pdms_v1 :73`, `safe_mask :87`, `progress_shortfall :92`, `guard_penalty :103`, `drivable_excess :114`, `lane_consistency :123`, `lane_change_term :138` |
| RL-val partition | `rl_partition.py` (3,000 sets / 75 logs, INHERITED from `D-REFE-RLS1-R0`) |

⚠️ **The on-policy sets and `train_pdm` live on the pod (`/workspace/data/…`).**
- Locally, `E:/Projects/TanitAD/data/refe_navtrain/` holds only `targets_rank0.jsonl` and the `aug_s*`, `r0_*` and `sc0_*` shards.
- No local copy of the labelled on-policy sets was found. Two locations were probed: `E:/…/data` and `F:/` (depth 6). So **Stage A training is a pod job**.
- **No banked base/A/B logit dumps** (`scores_*_base.npz`) were found locally either. They probably sit on HF `Sayood/tanitad-refe-rls1-*` or the pod. **UNVERIFIED.**

### 1.2 Dev box (CPU) simulation assets

| asset | where | note |
|---|---|---|
| NAVSIM v1.1 devkit + nuplan-devkit + venv | `F:/Archive/devbox-C/navsim/` (`navsim-3e8291bf…`, `nuplan-devkit`, `venv`) | the PDM scorer the labeller uses |
| nuPlan maps, 4 cities | `F:/Archive/devbox-C/navsim/data/maps/` | |
| navtest scene logs (147) and navhard two-stage | `…/data/openscene/navsim_logs/test` | ⛔ **EVAL ONLY** |
| nuPlan v1.1 DBs (325) | `E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/` | **0 of 325 overlap the 147 navtest log names** (MEASURED today). **Split identity is UNVERIFIED**: `val_extract.json` and `test_extract.json` point to D:, which is not mounted. A gate must prove navtest-disjointness before any training use. |
| navtrain DBs, metric cache, image pixels | pod | named in the brief, not re-probed |

---

## 2 · Gym specification

### 2.1 State
Privileged state from nuPlan DBs at 10 Hz:
- ego pose, velocity, acceleration and yaw rate;
- tracked agents (boxes, velocities);
- map (lanes, roadblocks, drivable polygons, intersections, route roadblocks).

These come through the same `metric_cache_parts(scenario)` the labeller uses (`navsim_pdm_targets.py:87`, INHERITED from RL_POSTTRAIN_REVIEW §4.1).

The teacher sees this state. ⛔ The student never does (§5).

### 2.2 Action
Choose 1 of K proposals every 0.5 s replanning cycle. Each proposal is a 4 s, 2 Hz plan, tracked by the nuPlan LQR + kinematic bicycle, as in PDM-Closed.

### 2.3 Proposal sources (K ≈ 24 per cycle in Stage B)
1. **PDM-Closed set (15):** IDM at {20, 40, 60, 80, 100 %} of the limit × lateral offsets {−1, 0, +1 m} (2306.07962). IDM carries distance keeping by construction.
2. **Speed sweep of the centreline path (9):** v_target ∈ {0.5…1.3} × the current speed, capped at the limit. ⭐ This is the set that **exposes phantom braking**: a selector that prefers 0.5× on a free road is visible immediately.
3. **REFe's own 64 banked proposals, at LOG states only.** Images exist only at log states, so REFe proposals can be the **first action** of an episode (step 0) and no later.
   - ⇒ Multi-step credit for REFe's own proposals takes the form *"execute REFe proposal a for 0.5 s, then continue under the teacher policy"*. That is DriveZero's first-action value reranking.

### 2.4 Transition
- nuPlan simulator at 10 Hz.
- Agents run in **two modes, both mandatory**:
  - (a) log-replay, which is non-reactive and matches NAVSIM;
  - (b) IDM-reactive, nuPlan's reactive agents.
- ⚠️ Reactive agents brake for an aggressive ego and make it look safe. Log-replay agents rear-end a slow ego. Each mode alone is biased.
- Episode = 15 s (nuPlan standard), or until a terminal event.

### 2.5 Reward
The reward encodes the PI's behaviour rules. It is multiplicative and terminates on hard events, following CaRL and DriveZero, because a summed dense reward is the measured route to "never move" (CaRL §4).

```
r_t = (1 - d_t) * Δprogress_t / v_ref_dt * Π_k q_{t,k}   +   d_t * h      (h = 0; a hard event ends the episode, so its future progress is lost)
hard events d_t (terminate):  at-fault collision · off-drivable (DAC) · wrong-way (DDC) · TTC < 1.0 s with closing speed (NAVSIM TTC rule)
soft factors q_{t,k} in [0,1]:
  q_free   free road (no lead within max(2.5 s · v, 20 m) on the route lane, no conflict agent):  1 if v >= 0.9·v*, linear to 0.3 at v = 0.5·v*   (v* = min(limit, route-curvature speed))   ← phantom braking
  q_gap    lead present:  1 if time gap in [1.5, 2.5] s;  linear to 0.5 at 1.0 s and at 3.5 s;  0.2 below 0.8 s                                  ← distance keeping
  q_stall  junction / stop-and-go:  0.5 if v < 0.5 m/s for > 2 s while the conflict zone is CLEAR (no agent with TTC < 4 s to it)               ← no stall when clear
  q_yield  entering a conflict zone with an agent at TTC < 3 s to it:  0.2                                                                         ← yield when not clear
  q_comf   |a_lon|, |jerk|, |a_lat|, |yaw rate|, |yaw accel| inside nuPlan's ego_is_comfortable bounds: 1, else 0.7 per violated channel
  q_osc    sign changes of a_lon over the last 4 s > 2 with |a_lon| > 0.5 m/s²: 0.8                                                              ← no speed oscillation
  q_lane   NAVSIM v2 lane keeping (0.5 m over a 2 s window, skipped in intersections; local devkit pdm_scorer.py:59-60, 615-650, INHERITED): 1 / 0.7
  q_cons   previous-pick consistency (ZTRS's b term): 0.9 if the new pick's first 1 s differs from the old plan's by > 1 m
```

- Every constant above is a **pre-registration candidate, not a tuned value**.
- Each term is logged per step. Per-term activation rates are measured **before** any weighting: `D-RL-COLL-SPARSE-1` found a term whose advantage was 0 on 92 % of windows.
- ⛔ None of the constants may be tuned on navtest.

### 2.6 Algorithms: a ladder, each rung justified by the previous one's failure

| rung | credit | algorithm | status / cost |
|---|---|---|---|
| **A0** | one-step, every arm observed | per-head BCE + listwise expected reward on true PDMS | **EXISTS = u400** (SFT-4 B); EPO variants R0 / R0b / R0d **ran and were K3-killed on TTC** (`D-REFE-RLS1-R0`, `-R0B`, `-SAFEOBJ-R0D`) |
| **A1** | one-step | A0 + **speed-balanced within-token pairwise loss on the NC / TTC / DAC heads** | **`PREREG_DRAFT_SCORER1.md`**, pod, ~3 arms × ~3.5 A40-h (ESTIMATED from RL-S1b's 10.5 A40-h for 3 arms) |
| **A2** | one-step | A1 + behaviour-term heads (q_free, q_gap, q_osc computed on the same 4 s non-reactive simulation the labeller already runs) and listwise on the multiplicative reward | labeller extension: CPU, then a pod fine-tune |
| **B1** | **multi-step**, rollout Q | **approximate policy iteration**: at sampled states, roll out each of K first actions for 0.5 s, then continue under the current teacher (round 0 = PDM-Closed) for 8 s. That gives Monte-Carlo Q(s, a) for all K arms, so the information is full again, and a listwise fit follows. 2–3 rounds. | dev-box CPU (§3) |
| **B2** | multi-step | PPO over the K-way choice with a decomposed per-term critic (DriveZero), only if B1 leaves a closed-loop gap | ⛔ **blocked on compute**: CaRL used 500 M samples on one 8-GPU node, and the nuPlan simulator is CPU-bound. Unblock = a GPU-vectorised simulator or a PI-approved large CPU budget. |

**Why B1 before PPO.**
- Once the multi-step consequence of each of K discrete actions is simulated, the problem is full-information again. Listwise on Q is a stable, low-variance estimator.
- ZTRS found sampled-subset GRPO unstable against exhaustive enumeration. Our DDv2 port diverged.
- PPO is the right tool only when K or the horizon makes enumeration unaffordable.

### 2.7 Transfer into REFe's image scorer (teacher → student)
1. The teacher T is a vector-input transformer over the state plus the proposal. It is trained in B1 and outputs Q_k per behaviour term.
2. **Relabel REFe's own 64 proposals at navtrain log states** (where images exist) with T's first-action Q. Use soft targets `softmax(Q/τ)` and keep PDM truth BCE on the six heads.
3. The student = REFe's scorer modules only (`score_q_mlp`, `score_dec`, `score_head`; detached, per `scorer_finetune.py` docstring). It is trained with KL to T plus BCE. The generator stays frozen, so the proposals are bit-identical.
4. **What "DAgger" can mean here, stated honestly.**
   - The student can only be queried at **log states**. Re-rendering images at off-log states is T2, which is **not provisioned**.
   - So the iteration is over the student's **own proposals and picks** (proposal-level on-policy), not over the state distribution its picks induce.
   - State-level DAgger is a named blocker. It would be unblocked by T2 re-perception, or by an AlpaSim/3DGS route.

---

## 3 · Compute (ESTIMATED; a pilot must MEASURE it first, gate M-GYM-1)

- **Labeller throughput (INHERITED, `train.py:2219` help):** 12 workers sustain ≥ 2 × 4.49 sets/s. That is about **0.75 sets/s per worker, ≈ 48 proposal-simulations/s per worker**.
- **A1, A2:** these reuse banked or online labels. The cost is the pod fine-tune, ≈ 3.5 A40-h per arm (INHERITED dose from RL-S1 R0c's estimate).
  - Re-scoring navtest needs the scorer's logits on navtest. Whether the detached scorer can run from a cached context (`ctx_cache.py`, `query_cache.py`) without the trunk is **UNVERIFIED**. If it can, the scorer is CPU-cheap; if not, it needs ≈ 1 GPU-h of navtest inference per arm.
- **B1, per labelled state:**
  - K = 24 arms × a continuation of 16 replans × 15 PDM proposals ≈ **5,760 proposal-simulations**.
  - At the inherited rate of ≈ 576 per second on 12 workers, that is ≈ 10 s.
  - 3 states per episode gives ≈ 30 s per episode. **10,000 labelled states ≈ 1.2 dev-box days.**
  - ⚠️ That rate was measured for the **cached** 4 s non-reactive NAVSIM scoring (the D9 metric cache).
    - B1 states are **off-log**, so each one needs its scene object and drivable map rebuilt, and its reactive agents stepped.
    - That can be 10–100× slower (a HYPOTHESIS).
  - ⇒ **Gate M-GYM-1:** measure the per-state cost on 20 states before committing to any B1 scale.
  - If the rate is 100× slower, B1 at 10k states becomes a named compute blocker. The pod is no CPU win either: its quota is measured at 7.65 CPUs (`D-REFCV3-THROUGHPUT`).
- **GPU inference on the dev box exists.** `refe_ego_probe_20261106/results_meta.json` records `device: cuda` for u400 inference over 1,058 navtest tokens. So fresh-token navtest dumps are a **dev-box GPU job, not a pod job** (MEASURED, meta file). P2's lever 1 recorded this as "blocked on compute" only because that brief was CPU-only.
- **Teacher and student training:** small next to the labelling cost. Teacher on CPU or a single GPU-h; student ≈ an A1-sized pod run.

---

## 4 · Validation battery (every arm, every read)

Each read reports all four families, each with its paired log-cluster bootstrap, and names the question each CI answers.
1. **Selection tier (T0/T1, navtest C3 1,058 tokens with true per-proposal PDMS, `table.npz`).**
   - Regret = oracle − pick, per stratum (random / zeroed).
   - Pairwise ranking accuracy on the ≥ 0.95 vs ≤ 0.50 groups.
   - Top-K oracle curve.
   - ⭐ **The P1 Q4 instrument:** within-token Spearman of each predicted head against v4/v0, minus the true head's Spearman. This is the defect's own metric.
2. **T1, full navtest 12,146 tokens.**
   - PDMS and sub-scores.
   - LONGITUDINAL: speed bias, CRUISE slow-down share (v4/v0 < 0.85), along-track.
   - LATERAL: heading, curvature, yaw-rate, cross-track.
   - TACTICAL: κ lat/lon where available.
   - STRATEGIC: **PAUSED** (PI 2026-09-19).
   - Distance-keeping: currently UNAVAILABLE on navtest (no lead track). ⇒ the gym's q_gap is computed in Stage-B episodes **on navtest logs, eval only**.
3. **Closed-loop behaviour scorecard (Stage-B simulator, navtest logs, EVAL ONLY, both agent modes).**
   - Free-road speed-hold rate.
   - Phantom-brake events per km.
   - Time-gap distribution.
   - Junction stall rate when clear, and yield violations.
   - Jerk / oscillation.
   - Lane keeping, collisions and TTC.
   - Plus the sim session's token `41d7b533797c5209`.
4. **Deliberate-regression arms (must read as known):**
   - *always-slowest proposal* ⇒ must FAIL q_free;
   - *always-fastest* ⇒ must FAIL q_gap / TTC;
   - *do-nothing (v = 0)* ⇒ reward ≈ 0 and FAIL;
   - the oracle ⇒ PASS.
5. **Noise floors:**
   - a **replicate arm** (same flags, new seed) for training variance (`H-ESTIM-SEED-1`);
   - for Stage B, a second simulator seed for agent-mode randomness.

---

## 5 · Risks and guards

| risk | guard |
|---|---|
| **"never moving is safe"** (CaRL's local minimum) | progress-multiplicative reward (zero progress ⇒ zero reward), q_stall, the do-nothing regression arm |
| **over-optimisation** of a learned teacher (Gen-Drive: 0.8753 → 0.8123 by doubling steps) | the true simulator metric at every read; K3-style kill when it falls past the bar |
| **scorer exploitation off-distribution** (TOAD) | the student is used only as a selector over REFe's own fan, never as a search objective; the fan stays frozen |
| **EP bought with TTC** (R0: +1.16 PDMS, TTC −1.00) | the g5 guard (TTC, NC ≥ −0.5 versus the control) stays registered |
| **sim-to-real for the student**: T sees GT agent velocities, the student sees images | report the teacher–student gap at log states; a student that cannot match T on a term is a perception limit, not a selection one |
| **reactive vs non-reactive bias** | both agent modes; a claim must hold in both |
| ⛔ **the leak test** (inference vision + measured ego only) | The student's inputs are images + `ego_vec` (measured motion incl. yaw rate, PI ruling) + nav goal. ⛔ No GT track, map feature or teacher output may enter the student at inference. The goal must not carry the situation classifier. Speed balancing uses truth **only to form the loss**; the proposal's own v4/v0 is admissible. **Mutation:** shuffle T's labels across tokens, and the student must fall back to base. |
| ⛔ **navtest in training** | a log-name gate against the 147 navtest logs plus a token gate, both RED-tested; the 325 local DBs stay quarantined until their split is proven |
| **the lane head has no truth in the C3 table**, yet it is the largest margin term (31/55) | add the teacher centre-line distance (the SFT-4 7th target) to the C3 validation table before reading lane effects |
