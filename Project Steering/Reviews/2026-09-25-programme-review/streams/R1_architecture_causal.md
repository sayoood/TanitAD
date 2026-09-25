# R1 — Architecture & causal diagnosis of the 4-brain latent world model

**Stream:** R1 of the 2026-09-25 whole-programme review (commissioned by the PI, Sayed).
**Mode:** static review only — repo at `467ce8a` (branch `claude/optimistic-shannon-vixpo6`, last commit 2026-08-04) + web. No pod, no torch, nothing re-run.
**Evidence classes:** MEASURED (ours + artifact path) · PUBLISHED (cited) · INHERITED (another doc, not re-verified) · ESTIMATED · HYPOTHESIS. "CODE" = established by reading source at the quoted file:line (a static fact about what the code does, not a measurement).
**ADE estimator:** every ADE below is the `full_set` mean over the 881 val windows with episode-cluster bootstrap CI95, unless explicitly marked `heldout` (legacy split-mean).

> Status: IN PROGRESS — sections are banked as they are finished.

---

## Q1 — What does each headline number actually measure? (banked first)

### 1.1 The surfaces, side by side

Every row: 881 windows / 40 val episodes (canonical deployment), `full_set` mean, episode-cluster bootstrap CI95 (B = 2000) unless marked.

| # | Number | What the scored path RECEIVES at inference | Which brains are ON the path | What it therefore measures | Evidence |
|---|---|---|---|---|---|
| a | **flagship v1 `flagship-30k` 0.4271 [0.3675, 0.4871]** | 8 encoded frames (each a 9-ch stack of 3 RGB frames @100 ms, 256²) **+ the past (steer, accel) window + v0 (held constant) + the EXPERT'S LOGGED FUTURE (steer, accel) for all 20 steps** | encoder → readout → operative predictor (1-step head, rolled 20×, **`intent=None`**) → `grounding.step['op']` → SE(2) accumulate | **World-model fidelity: how well the latent integrates known controls.** Nothing is chosen. | CODE `taniteval/taniteval/rollout.py:146-151` (fa = `ep.actions[t+window : t+window+fwd_k]`), `:181` `actions_source="expert_future"`, `:185` `honest_metric_name="wm_fidelity_ade_2s"`; `stack/tanitad/models/metric_dynamics.py:236` (`predictor(win_s, win_a)` — no intent). Value MEASURED `taniteval/results/driving_flagship-30k.json` |
| b | **Zero-parameter kinematic bicycle fed the SAME expert controls: 0.4518 [0.3097, 0.6174]** | v0 + the same expert future (steer, accel) | none (a 5-line Euler integrator) | the information content of the controls themselves | MEASURED `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-26-closedloop-artifact-rerun/closedloop_flagship-30k.CORRECTED.json` → `summary.open_bike_ade@2s_kinematic_floor`; code `taniteval/taniteval/closedloop.py:438-441` |
| c | **REF-C-base 0.4728 [0.3835, 0.5699]** (XL 0.4714) | the same 8-frame window + v0 (measurement encoder; nav = constant `follow`) — **no future information of any kind** | its whole model (ResNet trunk → 8×8×F map → anchored decoder → argmax-confidence anchor) | **a forecaster/planner: it must predict the future** | CODE `taniteval/taniteval/refc_eval.py:153-163`; values MEASURED `MODEL_REGISTRY.md` §6 (src `driving_refc-base-30k.json`) |
| d | **v1 tactical head, direct: 3.3839 [2.8336, 3.9722]** | the 8-state window, nav = `follow`; **no v0** (v1's planners have no ego port) | encoder → readout → strategic → tactical `wp_heads` | the hierarchy's OWN decision | MEASURED `…/closedloop_flagship-30k.CORRECTED.json` → `imagination_comparison.plan_direct_ade@2s_no_executor` (the registry §6 row still says "no windows dump — legacy only"; this cluster-bootstrap value exists and should replace it) |
| e | **v1 hierarchy plan → pure-pursuit → bicycle, open loop: 1.9028 [1.732, 2.0745]**; closed loop re-planning on imagined latents: **1.7318 [1.5707, 1.9070]** | as (d), plus a harness controller | strategic → tactical → (closed loop: operative predictor as simulator) | the hierarchy driving its own plan | MEASURED same JSON, `A_open_plan_bike_ade@2s`, `B_closed_bike_ade@2s` |
| f | **P2 CEM over the frozen v1 WM: 0.893 ± 0.114** (legacy `overlapping_holdout_se`; un-recomputable) | v1 WM + a **future-derived VTARGET** (85th-pct free-flow speed over the next 10–20 s) | encoder, operative predictor, op step-readout, CEM | the WM used as a *planner* — with a privileged speed target | INHERITED `MODEL_REGISTRY.md` §5; VTARGET provenance CODE `taniteval/taniteval/planner_p2.py:106-129` |
| g | CV 0.8377 · CTRV 0.523 · best-of-3 kinematic 0.5005 · no-vision ego-MLP 0.5735 | v0 (+ yaw rate) only | — | trivial forecasters | MEASURED registry §6 (full-set by construction) |

### 1.2 Verdict on the "three-way tie"

**The 1= tie (v1 0.4271 ≈ REF-C-XL 0.4714 ≈ REF-C-base 0.4728) is not a like-for-like comparison and must not be read as one.** Row (a) is a world model *handed the expert's future controls*; rows (c) *predict* the future from the same pixels plus v0. The registry has said so since 2026-07-27 (§1.2 "⛔ WHAT THIS ROW MEASURES"), and `rollout.py` now stamps a PC2 record on it — but §6 still ranks the two in one column under one "Rank 1=" label, and the prior review (R2 F1/F2, 2026-07-25) built its "the control ties the treatment" finding on that column. The correct statement is stronger than the prior review's:

- **On the like-for-like surface (the model must choose), REF-C-base beats every path the 4-brain hierarchy owns by a wide margin:** 0.4728 vs P2-over-v1 0.893 (legacy, *and* P2 has a privileged future speed target) vs hierarchy-plan-tracked 1.9028 vs tactical head 3.3839 — a **1.9× to 7.2×** gap in REF-C's favour. CODE + MEASURED as cited; the 0.893 is INHERITED/legacy.
- **The one flagship-line planner that ties REF-C is REF-C's own decoder transplanted onto the WM.** v1.6 (`flagship-v16-ab-ft`: the REF-C anchored decoder on v1's state, then 4 ViT blocks + predictor unfrozen) scores **0.4375 full-set**, paired vs REF-C-XL **Δ −0.0340 [−0.1060, +0.0511]** (not separated) and vs v1's WM-fidelity 0.4271 **Δ +0.0104 [−0.0888, +0.1147]** — and it cost the world model **144 %** (canary 0.452 → 1.1022). INHERITED registry §1.4b (raw `…/incoming/2026-07-25-v16-paired-interval/v16_vs_v1_paired_bootstrap.json`). So a like-for-like tie *does* exist — but between REF-C and **REF-C-on-a-WM-state**, not between REF-C and the hierarchy.
- **Given the expert's controls, the 263 M world model is statistically indistinguishable from a zero-parameter bicycle given the same controls**: 0.4271 [0.3675, 0.4871] vs 0.4518 [0.3097, 0.6174] (MEASURED, same JSON, unpaired intervals; a paired delta has not been computed — see §8 open questions). The bicycle figure is *pessimistic*: it integrates steer minted with L = 2.9 m using L = 2.7 m (`closedloop.py:118`, a +7.4 % curvature overshoot documented at `:102-117`), so a matched bicycle would sit lower still (HYPOTHESIS: at or below 0.4271).
- **The split of that 0.4271 shows where the information comes from.** Mean |along-track| / |cross-track| error at 2 s (MEASURED, `…/2026-07-26-closedloop-artifact-rerun/latlon_decomposition.json`): WM-with-expert-controls **0.826 / 0.274 m**; bicycle-with-expert-controls **0.829 / 0.309 m**; CV **0.915 / 1.096 m**. ⇒ essentially the entire margin over CV is **lateral, and it is the future curvature being handed in**; longitudinally, even the *true* controls barely beat constant velocity (0.83 vs 0.92 m). The reason is a data fact, not a model fact: PhysicalAI's `accel` channel (`ax`, `stack/tanitad/data/physicalai.py:623-624`) correlates only **r = 0.434** with the pose-derived dv/dt (INHERITED, `…/Implementation/incoming/2026-07-26-idm-v2/IDM_DIAGNOSIS.md` table row `long_accel`; that the IDM's "CAN `long_accel`" is the same `ax` column is UNVERIFIED but consistent with `signals_at`).

### 1.3 What the v1 tactical head's 3.38 m is, causally

The head that must *predict* is **speed-blind by construction** in v1: `TacticalPolicy.forward` (`stack/tanitad/models/fourbrain.py:328-361`) reads only the 2048-d state window and the strategic ctx; the `ego_emb` port exists only under `v2_ego_to_planners` (`fourbrain.py:326`), which v1 did not use. Three numbers line up on the "no metric speed on the path" level: v1 tactical head **3.38 m**, the **no-speed** operative control **3.0175 m** (registry §1.1), and both far above CV **0.8377** (which knows v0 exactly). The latent does carry *some* speed — a linear probe on frozen v1 z reaches R² 0.772 with a systematic 17 % shrinkage toward the training mean (INHERITED, `IDM_DIAGNOSIS.md` row `speed`) — but at R² 0.77 the residual speed error is metres-per-second, which is metres of ADE at 2 s (ESTIMATED). **So the ranking of the entire leaderboard is explained, to first order, by how much speed/control information each scored path is handed**: expert future controls + v0 (0.43) ≈ bicycle on the same (0.45) < v0 + vision (REF-C 0.47) < v0 + yaw-rate (CTRV 0.52) < v0 alone (CV 0.84) ≪ no metric speed at all (no-speed 3.02, v1 head 3.38). Evidence class: each number MEASURED as cited; the "first-order explanation" is an ANALYSIS of those numbers (HYPOTHESIS-grade as a causal claim, but no number in the table contradicts it).

### 1.4 Collateral: the programme's "strongest scientific claim" is measured on the same surface

The vision-anticipation panel (prior review F0: "flagship beats the CTRV oracle by +0.796 m on high-divergence windows; mean-replacing the scene inverts it; vision effect +1.325 m [+1.04, +1.64]") is computed by `taniteval/taniteval/generalization.py:371-399` (`rollout_modes`), which **keeps the true future actions `fa` and replaces only the latent states**. On a high-CTRV-divergence window the upcoming turn/brake is *already encoded in the expert's future steer/accel* that the rollout is given. The ablation therefore shows that the metric decode *needs the scene latent to turn given controls into metres* (e.g. the latent's speed/scale content), not that the model *anticipates the manoeuvre from the scene*. **The claim is not refuted — it is not identified.** Only test D (upcoming-curvature linear probe on the latent, R² 0.254 vs 0.031 ego-only) is free of the confound. CODE (`generalization.py:4-8` states the design; `:381-398` implements it). Cheapest fix: re-run tests A/B with `fa` replaced by zero-order-hold (the `future_actions=None` branch of `rollout_decode`) — ~1 A40-hour.

### 1.5 The same confusion inside the programme's own "hierarchy-supporting" evidence

`…/incoming/2026-07-25-hpp0-confound-audit/HPP0_CONFOUND_AUDIT.md:457` reads "grounded operative rollout 0.4271 vs ungrounded tactical head 3.3839 … Strong and directly hierarchy-supporting: the grounded consequence readout beats the direct supervised head by 7.9×", and registry §6 reading 2 turns it into "the head is a lossy readout of a good world model". **Both compare a path handed the expert's controls against a path that must predict them without v0.** The 7.9× is dominated by the controls and by v0, not by grounding. This reading underpinned the v3 pivot (P2) and should be withdrawn as evidence for "grounding > supervision". CODE + MEASURED (rows a, d above).

---

## Q2 — Hierarchy information flow: intended vs real

### 2.1 Per-brain ledger (flagship v1 `flagship4b-speedjerk-30k`)

Params MEASURED from `taniteval/results/eff_flagship-30k.json` → `fp32.params.by_module_m` (model 263.44 M + 13.43 M grounding heads kept outside the model). Loss weights CODE `stack/tanitad/train/flagship_losses.py:151-164` (`LossWeights`) + `stack/scripts/train_flagship4b.py:224-231,655-656` (H15). Horizons CODE `train_flagship4b.py:794-796`.

| brain | params | trained on (inputs) | receives at inference, **as scored** | outputs | loss that trains it | conditions whom |
|---|---:|---|---|---|---|---|
| ViT encoder + 4×4 readout | 87.02 + 0.10 M | 8 × (9-ch 256² stack) | same | z ∈ ℝ^{8×2048} | *everything* flows here: JEPA (1.0), K=4 rollout (0.5), tactical-pred JEPA (0.5), goal-latent (0.5), **metric inv-dyn on real pairs 2.0 × 3 levels**, **forward-consistency 1.0 × 3 levels**, SIGReg (0.1, 64 dims exempt), action inv-dyn (0.5), H15 NLL (0.5 × p=0.5), and the planner CEs/L2 through the state window | — |
| operative predictor | 91.36 M | z window + (steer, `ax`, v0-constant) window; **intent only on the JEPA path** (`flagship_losses.py:247`) | z window + **expert future actions**, **`intent=None`** (`metric_dynamics.py:236,259`) | ẑ_{t+1,2,4} | JEPA 1.0 (intent-conditioned); rollout + grounding fwd (intent-free) | — |
| tactical predictor | 26.54 M | z window + actions | **nothing reads it** | ẑ_{t+8,16} | JEPA 0.5 | none |
| strategic policy | 8.39 M | z window + `nav_cmd` (an **echo** of the route target, `config.py:282-292`: echo rate 1.0000, `_ROUTE_TO_NAV` is a bijection) | z window + **constant `follow`** | ctx ∈ ℝ²⁵⁶, route logits | route CE 0.5 (on the echo) | FiLM → tactical |
| tactical policy | 22.74 M | z window + ctx | z window + ctx(follow) — only in the closed-loop / plan_direct harness, never in the headline | 5-way manoeuvre, 4 waypoints, goal latent, intent | wp L2 1.0 (**÷ pose_scale² = ÷100**), manoeuvre CE 0.5, goal-latent JEPA 0.5 | intent FiLM → operative (**JEPA path only**) |
| H15 imagination field | 22.06 M | token grid with a masked sector | **nothing reads it** | belief tokens + log-var | NLL 0.5 on 50 % of batches | none |
| action inverse dynamics | 5.25 M | (z_{t-1}, z_t) | nothing | action | 0.5 | none |
| grounding op / tac / str | 13.43 M (3 × invdyn + 3 × step readout) | real latent pairs; the **same** intent-free rollout decoded to k = 4 / 16 / 20 | **`step['op']` only** (`taniteval/taniteval/loaders.py:65,85`) | Δpose | invdyn 2.0, fwd 1.0 (per level) | — |

**Three code facts the ledger exposes (CODE):**
1. **The "cadence" does not exist.** `TacticalPolicyConfig.cadence = 5` and `StrategicPolicyConfig.cadence = 20` (`config.py:118,140`) are read by **no code in the repository** (`grep '\.cadence'` → only the `run_hierarchy` docstring, `fourbrain.py:376-377`). Training runs every brain every step; the closed-loop harness re-plans strategic and tactical **every tick** (`taniteval/taniteval/closedloop.py:265-271`). "Thinking fast/slow" has never been executed on any measured path.
2. **All three levels read the identical 8-frame, 0.8 s state window.** The strategic brain has no memory, no longer receptive field, no route/map input, and (at inference) a constant command — so it is a function of *the same 0.8 s the operative sees*; with a 20-tick cadence it could only ever be a **staler** copy of that information. The three "levels" of grounding are three copies of the same head decoding the **same** intent-free rollout truncated at 0.4 / 1.6 / 2.0 s (`train_flagship4b.py:794-796`; `metric_dynamics.py:348,375`) — levels of *horizon*, not of *abstraction*.
3. **The eval decodes a 20-step rollout with the readout trained on 4 steps.** TanitEval uses `grounding.step['op']`, whose forward-consistency loss covers only the first `op_fwd_k = 4` transitions, while `step['str']` was trained on all 20. Which of the three identical heads decodes k = 20 best has never been measured (0 GPU-training, ~1 A40-hour eval; §8).

**Parameter consequence.** On the scored 0.4271 path the model uses encoder + readout + predictor + one step readout ≈ **180.6 M of 276.9 M (65 %)**. **~85.0 M (32 % of the 263.4 M model) — tactical_pred 26.5, tactical 22.7, H15 22.1, strategic 8.4, inv-dyn 5.2 — plus 11.3 M of the 13.4 M grounding heads are training-time auxiliaries or sit on paths no headline scores** (180.6 + 85.0 + 11.3 = 276.9 M). (MEASURED param counts; the "not on the path" statement is CODE.)

### 2.2 Are the seams load-bearing? — corrected verdict **0 of 3**

| seam | published | corrected / current | verdict | source |
|---|---|---|---|---|
| nav → strategic | "load-bearing by construction" (2026-07-25 dry-run) | route acc 1.0 *with* the command, `route_skill_vs_chance` **0.0** without; the label is a deterministic function of the fed command | **ECHO** — not a seam | MEASURED registry §8 D-A6 / D-033; CODE `config.py:282-292` |
| strategic ctx → tactical | man-acc Δ(real − mean ctx) **+0.0439** (`heldout`) | **+0.0148** full-set (fails the 0.02 practical floor); goal-latent cos Δ **0.0050** (fails 0.01); wp-ADE Δ 0.0437 m on a 3.38 m head | **NOT load-bearing** (point estimates; the panel persists no per-window arrays, so no interval exists) | MEASURED `…/Benchmarks & Eval/Implementation/incoming/2026-07-25-jack-blast-radius/JACK_BLAST_RADIUS.md` §5.1, `jack_hierarchy_recompute.json` |
| tactical intent → operative | harmful when ungated (cos vs none −0.238; intent-proj norm 31.4 swamps act-emb 28.3) | excluded from every deployed / scored path by design | **HARMFUL → removed** | INHERITED registry §8 D-033; CODE `rollout.py:151` |

The 2026-08-02 v5 deep review independently counts **0/3 beneficial seams for v1 and 0/3 for v2corpus**, manoeuvre κ (declared vs driven) 0.253 (v1) / 0.0072 (v2corpus), and vision-route accuracy equal to the majority-straight rate (0.9474 vs 0.9474) — INHERITED, `Project Steering/V5_FLAGSHIP_DEEP_REVIEW.md` §1 P2. **What changed since the 2026-07-25 review: its "1 of 3 seams load-bearing" is now 0 of 3** (the one positive seam was a `heldout`-estimator artefact).

### 2.3 Intended vs real dataflow

```
INTENDED (design docs, fourbrain.py:1-16, config.py:372-388)
  frames ─► ViT ─► readout z[8] ─┬─► STRATEGIC(z, nav)  every 20 ticks ─ctx──┐
                                 │                                           ▼
                                 ├─► TACTICAL(z, ctx)   every 5 ticks  ─intent┐ ─► manoeuvre, 2 s goal
                                 │                                            ▼
                                 └─► OPERATIVE(z, a, intent) every tick ─► ẑ ─► step readout ─► trajectory
       (brain 4: fallback monitor on imagination error)

REAL — v1, as trained and as scored
  TRAIN:  z[8] ─► STRATEGIC(z, nav=ECHO of route label) ─ctx─► TACTICAL ─intent─► OPERATIVE(JEPA k=1,2,4 only)
          z[8] + EXPERT (steer, ax, v0) ─► OPERATIVE(intent=None) ×20 ─► step['op'|'tac'|'str'] ─► metres  (grounding)
          token grid ─► H15 (masked sector)            z ─► tactical_pred (k=8,16)          z ─► inv-dyn
          └─ none of these three outputs is read at inference

  SCORED HEADLINE (0.4271, "wm_fidelity_ade_2s"):
          z[8] + EXPERT FUTURE (steer, ax) + v0 ─► OPERATIVE(intent=None) ×20 ─► step['op'] ─► SE(2) ─► metres
          [strategic, tactical, tactical_pred, H15, intent seam, cadence: all ABSENT]

  THE HIERARCHY'S OWN DECISION (3.3839 direct / 1.9028 tracked / 1.7318 closed loop):
          z[8] ─► STRATEGIC(z, follow) ─ctx─► TACTICAL(z, ctx) ─► 4 waypoints   (no v0; re-run every tick)
```

### 2.4 The nav-echo defect, precisely

The strategic brain is trained on `nav_cmd` minted by the *same* `route_from_future*` call as its route target (`config.py:282-287`), so the CE is solvable by the identity map — and it is (`route_skill_vs_chance` 0.0). At inference it receives `follow` (index 0) on every window (`fourbrain.py:384-385`, `closedloop.py:260`), so its ctx is a function of 0.8 s of pixels plus a constant. The dataset has **no** map, route, lane graph or traffic-light signal (CLAUDE.md rule 2, five probes), and ego coordinates are clip-local (no GNSS). ⇒ **In this corpus the strategic level has no information source that the operative level lacks.** The only non-circular strategic gradient in the repo is LEVER A (`v2_route_from_vision`, nav forced to `follow`), used only in the killed v2/v3enc arms. This is a **data-imposed** ceiling on the strategic brain, not a tuning problem: no architecture change can make a strategic brain load-bearing on inputs that carry no strategic information (HYPOTHESIS-grade as stated, but it follows from the CODE + dataset facts cited).

---

## Q3 — The decision / selection defect

### 3.1 The defect, arm by arm

| arm | decision mechanism | measured symptom | architectural cause | evidence |
|---|---|---|---|---|
| **v1 tactical head** | `TacticalPolicy`: one causal transformer over z[8] → linear 5-way manoeuvre softmax + 4 **unimodal** L2 waypoint regressors | 3.3839 m direct (worse than CV); tracked 1.9028; closed loop 1.7318 | (i) **no v0 input** (§1.3); (ii) unimodal L2 on a multimodal future ⇒ regression to the conditional mean; (iii) wp loss is `((wp−gt)/10)²` — a 3 m error costs 0.09, tiny next to JEPA/grounding mass; (iv) 5-way softmax is a **priority collapse** (`turn > brake > accel > lane_keep`) of two orthogonal axes | MEASURED §1.1 row d/e; CODE `fourbrain.py:328-361`, `flagship_losses.py:287`; `refb_labels.py:100-109` (via PREREG_D-TAC1 §1 fact 2) |
| **REF-C (base/XL)** | 128/256 FPS anchors, per-anchor conf + offset, argmax of the **t=0 (unrefined) confidence** | oracle-in-fan **0.1640 [0.1414, 0.1902]** vs pick 0.4714 (XL); pick > 2× oracle on **45.4 %** of windows; manoeuvre head never emits `accelerate` (0/93) | selection logit trained by CE against the *nearest anchor to one realised future*; v0 enters only the measurement condition with **50 % ego-dropout** in training (`refc.py:1888-1891`); no acceleration input at all; manoeuvre head reads the image only | MEASURED `taniteval/results/scaleab_refc-base-30k_vs_refc-xl-30k.json` (`gap`, `oracle_vs_K`); PREREG_D-TAC1 §1 facts 5-7 (INHERITED) |
| **v4 / v5 `FlagshipV15Head`** | REF-C decoder KV-swapped onto the WM's 16 cells × 8 frames; select on the **refined** logits + zero-init factorised LAT/LON/DIST grafts | v4.1 gate FAIL 0.8522 [0.7468, 0.98]; v4 fan oracle 0.2505 vs picks ≥ 0.47 | selector trained by a **hard-argmin CE against the hindsight-best candidate** (`flagship_v15.py:861-866`, `cls_refined`) — see 3.2(b) | MEASURED registry §1.5 / §6; CODE |
| **v5f "conditional imagination"** | the same head + 32 extra KV tokens = 8 FPS **probe** action sequences rolled 20 steps through the predictor (`imagine_probes`) | trainer log (NOT eval, NOT quotable): `plan_ade` 1.0251 vs `oracle_ade` 0.5254 at step 3.5–4 k, `rank_acc` 0–0.375 | **the probe tokens are identical for all 256 candidates** — the code's own measured constant `IMAGINATION_HAS_CANDIDATE_AXIS = False` (`flagship_v15.py:638-656`, MEASURED 2026-07-27). They add context; **they cannot rank** | CODE; INHERITED registry §1.8 |

⚠️ **On the v5f "the SELECTOR is the defect, the arm would be ~2× better if it chose correctly" reading (registry §1.8).** Three reasons it overstates: (a) the numbers are trainer-log values over a handful of *training* windows per logged line (`rank_acc` moves in steps of 0.125), which the registry itself says are not quotable; (b) `oracle_ade` is a **best-of-256 against one realised future**, and it falls monotonically with fan size — REF-C-XL's oracle is 12.0 m at K=1, 0.81 m at K=32, 0.26 m at K=128, 0.16 m at K=256 (MEASURED, `scaleab_…json → oracle_vs_K`) — so `sel_gap` grows with K **by construction**; the same file shows REF-C-XL's gap 0.3075 at 256 anchors vs 0.2091 when restricted to 128; (c) registry §4.1 already records that on REF-C the oracle gap is "~92 % irreducible" (INHERITED; prose, 47 arms). **"Oracle − pick" is not a selector defect metric; it is a dispersion metric.** The admissible selector metric is the pick against a **learned expected-cost minimiser that uses only inference-time inputs** — which the programme has measured (3.3) and which shows a large but *not* 2× recoverable share.

⚠️ **v5f's defining lever was adopted against a measurement already in the code.** The 2026-08-02 deep review justified the restart ("the decoder sees the CONSEQUENCES of candidate controls before it denoises", `V5_FLAGSHIP_DEEP_REVIEW.md` §1 P1) six days after `flagship_v15.py` recorded that the probe roll has no candidate axis and that E-V5-1's imagination-scoring negative was "over-determined". The restart discarded 5,300 steps. Step time is now 19.58 s (MEASURED registry §1.8) vs ~12 s for v5 (INHERITED, deep review §3; different pod and batch config, so not attributable to the lever alone). CODE + INHERITED.

### 3.2 Root cause, architecturally

**(a) There is no cost model anywhere in the stack.** The only thing the world model knows is how the ego's latent evolves under given controls; nothing in the loss represents other agents, drivable area, rules, progress or comfort (CODE — the full loss list is §2.1). A WM without a cost cannot rank: E-V5-1 (MEASURED, `V5_PLAN.md` §8) found that the WM **"obediently simulates"** a 181 km/h candidate and rates it maximally self-consistent (+19.66 m along-track bias). **Consistency is not plausibility.** This refutes WM-rollout scoring *as built*, not in principle.

**(b) The selector is trained on the wrong decision-theoretic target.** v4/v5 train the ranking score with a hard CE against the index of the candidate that *happened* to be closest to one realised future (`flagship_v15.py:861-866`); REF-C trains anchor CE against the nearest anchor. The argmax of P(c is the hindsight-best) is **not** argmin_c E[ADE(c) | observation] when the future is multimodal: it favours candidates that are *sometimes* spectacular over candidates that are *reliably* good. The in-code record already found the symptom — as the fan sharpens, the hard target "becomes a coin flip between equally-good plans" and `frac_sel_2x_worse` degrades 0.099 → 0.40 (`refc_rescorer.py:50-58`, INHERITED). The programme's one strong selector result uses the correct target: a per-candidate **regression of each candidate's own realised ADE**, pick = argmin (E-GOAL-4, below).

**(c) The selector is starved of the longitudinal state it most needs.** The residual is longitudinal: with the true goal, oracle **along-track** recovers **+83.7 %** of the headroom and oracle **cross-track** only **+2.9 %** (not separated) (INHERITED, `V5_PLAN.md` §8). The single most informative longitudinal observable is the current acceleration measured as a **0.1 s speed difference `ax_fd`**; the dataset's native `ax` correlates only **0.759** with it (INHERITED, E-GOAL-3) and **r = 0.434** with pose dv/dt (INHERITED, IDM_DIAGNOSIS). **No arm in the programme is fed `ax_fd`**: v1's action channel is native `ax` and v0 is constant-expanded across the window (`flagship_losses.py:227-238`) — i.e. the WM never sees a speed *history*; REF-C sees v0 only, dropped 50 % of the time in training.

**(d) Candidate consequences never reach the ranking** (probe imagination has no candidate axis — CODE); and the per-candidate roll that would supply them (`imagine_candidates`, `flagship_v15.py:699-745`) is unaffordable at 256 candidates on Thor (§Q8).

**(e) The 5-way manoeuvre label destroys information before any head sees it**: **9.68 %** of windows (132/1364) carry a live longitudinal manoeuvre that the priority collapse relabels as a turn; the factorised head is **readout-limited** (`auc_lon_active` **0.7294**), `accelerate` recall never exceeds **0.153** at any decode temperature (MEASURED by the D-TAC1 stream, `…/incoming/2026-08-03-dtac1-tactical-head/`, INHERITED here).

### 3.3 What the programme has already settled (so v6 does not re-buy it)

| experiment | result | class |
|---|---|---|
| Bar A — learned discriminative re-scoring of the frozen v4 fan over existing latent features | in-sample ceiling **0.4907** (881 win) — cannot reach v1's 0.4271 even when memorising | INHERITED (`…/2026-07-26-bar-a-selector/raw/bar_a_produced.json` per registry §1.2) |
| E-V5-1 — per-candidate imagination-consistency scoring | **REFUTE**: best 0.5645 vs bar 0.4907 | INHERITED V5_PLAN §8 |
| Fan conditioning on v0 | refuted: 100 % of windows already hold a candidate within 0.5 m/s of the realised speed; best-in-fan is speed-matched | INHERITED V5_PLAN §8 |
| REF-C v1.2 learned re-scorer, 47 arms (top-8, soft/pair/regress/hard) | ≤ **8.4 %** of the oracle gap; headline not separated (+0.00893 [−0.0062, +0.0250]) | INHERITED (prose; registry §4.1 says "INHERITED until re-derived") |
| ⭐ **E-GOAL-4 — gradient-boosted per-candidate expected-ADE regressor over the frozen REF-C-XL fan**, 600-episode deployment (13,198 windows), 5 episode-disjoint OOF folds, 18 past-only columns (REF-C logit/softmax/rank, candidate end/mid along/cross, path length, mean speed, heading, v0, **ax_fd**, CV endpoint) | as-trained **0.5015** → **no goal 0.3917** (paired **−0.1098 [−0.1239, −0.0960]**); + a `(v, ax_fd)` goal head **0.3040 [0.2930, 0.3157]** (future-blind background); a naive **2·v0** goal through the trained selector 0.3102 | INHERITED from `…/incoming/2026-07-28-egoal-4-joint/EGOAL_4.md` table §2; recovery fractions re-read from `raw/e4_summary.json` (they match) |

**Reading.** On the 600-episode deployment the flagship's *own* WM-fidelity number (handed the expert's future controls) is **0.4108 [0.3956, 0.4273]** (registry §1.2a). A learned selector over a flat REF-C fan, using only past observables, reaches **0.304** there. The two have not been paired (window identity is plausible — both 13,198 / 600 — but UNVERIFIED), so this is a juxtaposition, not a result; but it is the programme's clearest pointer to where decision quality comes from. **The E-GOAL-4 (+32 % of the oracle gap with no goal) and v1.2 (≤ 8.4 %) findings disagree by ~4× and must be reconciled before v6 is funded** — the leading HYPOTHESES are `ax_fd` (absent from v1.2), v1.2's top-8 restriction, and deployment (40 vs 600 episodes).

### 3.4 Fix options, ranked (full recommendation cards in §R)

| rank | fix | defect addressed | evidence for it | verdict |
|---|---|---|---|---|
| 1 | **Per-candidate expected-cost selector** (regress each candidate's cost; pick argmin), fed candidate kinematics + ego `v0`, **`ax_fd`** + scene tokens; trained on the parity train corpus, not on val folds | 3.2(b), (c) | E-GOAL-4 −0.11 to −0.20 m (INHERITED, MEASURED by that stream) | **do first** — cheapest, most evidenced |
| 2 | **Hydra-MDP-style multi-target distillation** into that selector: add per-candidate collision / TTC / progress / comfort sub-scores computed **offline from `obstacle.offline` agent tracks** (97.44 % coverage) as extra regression targets | 3.2(a) | PUBLISHED: Hydra-MDP (arXiv 2406.06978) distils rule-based metric teachers into per-anchor scorers; NAVSIM PDMS family. Here: HYPOTHESIS (drivable-area terms impossible — no map) | do second; the only route to a *cost* the data can supply |
| 3 | **Factorised lat × lon tactical heads, lon trained on minted-from-kinematics labels, retrained** | 3.2(e) | D-TAC1: readout-limited; 9.68 % label-destroyed windows; accel recall ≤ 0.153 | worth it as a feature for rank 1, not as a selector |
| 4 | **Predicted goal-point conditioning** (from vision + ego kinematics, never from the situation classifier) | inductive bias | E-GOAL-4: +26 recovery points *capacity-matched*, but the goal carries no information beyond (v, ax_fd) (R² 0.9999) | take the free inductive bias; do **not** fund a goal *supplier* |
| 5 | WM-rollout scoring / MPC with a cost model | 3.2(a),(d) | E-V5-1 refuted consistency-scoring; C2 (one WM roll as a reference) helps with v1's WM (−0.2918) but hurts with v4's WM on its own fan (+0.2090) | only after a cost exists (rank 2); budget-infeasible at 256 candidates on Thor |
| 6 | Learned critic / value (offline RL) | 3.2(a) | none in-programme; needs outcome labels or closed-loop data | defer |
| 7 | Energy-based selection | — | none in-programme; mathematically rank 1 with a contrastive loss | no case to fund separately |

---

## Q4 — The longitudinal deficit: why lateral is good and longitudinal is weak

### 4.1 The measured shape (all on the canonical 881 windows unless stated)

| fact | number | class · source |
|---|---|---|
| v1's separated win over CV is **entirely lateral** | cross-track +0.7720 [+0.4166, +1.1914]; along-track +0.2543 [−0.0278, +0.5304] **not separated**; speed MAE Δ −0.0032 [−0.1285, +0.1182] | MEASURED `taniteval/results/driving_flagship-30k.json` (quoted verbatim in `taniteval/taniteval/driving.py:22-31`) |
| even **handed the true future accel**, along-track error at 2 s barely beats CV | WM 0.826 m · bicycle 0.829 m · CV 0.915 m (mean \|along\|) | MEASURED `latlon_decomposition.json` |
| v1 is **2.0× worse than holding v0** on the 639 steady windows (72 % of val) | speed MAE 0.4231 vs 0.2109 m/s, paired Δ −0.2122 [−0.2778, −0.1443] | MEASURED (same source); every one of the 14 arms with a dump is separated-worse than hold-v0 there (registry §6 reading 4) |
| the fed longitudinal action is a poor measurement of what it claims to be | native `ax` ↔ pose dv/dt **r = 0.434**; ↔ `ax_fd` **0.759** | INHERITED `IDM_DIAGNOSIS.md`; `V5_PLAN.md` §8 E-GOAL-3 |
| v1's WM-fidelity squared error is **87 % longitudinal** | 0.8733 share | MEASURED `latlon_decomposition.json` (open_grnd) |
| the selection headroom is longitudinal | oracle along-track recovers **+83.7 %**, oracle cross-track **+2.9 %** (n.s.) | INHERITED `V5_PLAN.md` §8 |
| perfect lead-vehicle state buys little at 2 s on this corpus | 41.65 % of windows have no vehicle ahead within 50 m; gap/closing/TTC worth **+2.3** recovery points (n.s. on the primary axis) | INHERITED `V5_PLAN.md` §8 E-GOAL-1 (privileged `obstacle.offline` tracks) |
| the single largest longitudinal lever measured anywhere | a **0.1 s speed difference `ax_fd`**; `v + ax_fd` ≈ the full 10-column history block (+0.0002 [−0.0023, +0.0027]) | INHERITED `V5_PLAN.md` §8 E-GOAL-3 |

### 4.2 Causal account — four stacked causes

1. **Lateral is handed in; longitudinal is not.** Road curvature ahead is visible in a single frame (lane lines and road edges occupy large image areas), and on the headline surface the future **curvature is supplied** as `steer = atan(2.9 κ)` (`stack/tanitad/data/physicalai.py:12,621`). Longitudinal change depends on things that are either invisible at this resolution (closing rates, §Q5) or not in the data (signals, stop lines, route), and on the current acceleration — which is fed through the wrong column.
2. **The architecture withholds the best longitudinal observable.** The WM is fed native `ax` (r = 0.434 with dv/dt) and a **constant** v0 broadcast over the whole 8-step window and all 20 future steps (`flagship_losses.py:227-238`), so it never observes a speed *history*; `ax_fd` — which by itself carries most of the predictable longitudinal signal (E-GOAL-3/4) — is fed to **no** arm. REF-C sees v0 only, zeroed on 50 % of training samples, and no acceleration. This is a **one-column input fix**, parity-neutral (it changes an input, not the episode set).
3. **No architectural "keep speed" default.** Both families emit absolute trajectories: REF-C's anchors are FPS draws from a v0-agnostic synthetic unicycle pool (candidate speed tracks ego speed at slope **−0.129** vs GT **+1.0003**, INHERITED V5_PLAN §8) and the WM integrates a noisy control. Holding the current speed — correct on 72 % of windows — must be *reproduced* by an offset head or by integration rather than being the zero-residual default. That is why every arm loses to hold-v0 on steady windows while winning brake/accel transients. HYPOTHESIS (mechanism), consistent with every MEASURED number above; a residual-over-kinematic-prior parameterisation is the cheapest test (R-4).
4. **Train/eval horizon mismatch in the operative.** v1 trains the recursive rollout at K = 4 and the `op` readout at 4 steps but is scored at 20 (`train_flagship4b.py:794`; §2.1 fact 3). A fine-tune at K = 20 (RR-20) took WM-fidelity ADE 0.424 → **0.348** (paired Δ CI [0.0613, 0.0906]) and erased the speed bias (+0.9397 → −0.0092 m/s), at a **2.2×** curvature-MAE cost (INHERITED, `V5_FLAGSHIP_DEEP_REVIEW.md` §1 P5). The +0.94 m/s bias is therefore a **compounding** artefact, not a vision deficit.

### 4.3 Can the encoder see speed or closing rate?

- **Ego speed: partly.** Near-field ground flow is resolvable: with f = 266 px and camera height ≈ 1.22 m (INHERITED `stack/tanitad/replay/rr_log.py:47`), a ground point 10 m ahead sits ~32 px below the principal point and moves ~3–4 px per 100 ms at 10 m/s (ESTIMATED, pinhole geometry). A linear probe on frozen v1 z reaches R² **0.772** for speed, with a **17 % shrinkage** toward the training mean (gain 0.830) — a classic regression-dilution signature of monocular scale ambiguity across two camera rigs (INHERITED `IDM_DIAGNOSIS.md`). R² 0.77 leaves speed errors of order metres-per-second — metres at 2 s. **That is why every head without v0 sits at ~3 m** (no-speed 3.0175, v1 tactical head 3.3839) and why the fed v0 is load-bearing: on the v3enc checkpoint, replacing v0 by another window's (in-distribution) v0 raises the 0.4 s grounded error **0.26 → 2.09 m**, zero-filling it **0.94 m** (MEASURED `taniteval/results/postmortem_b_egodropout_v3enc10k.json`, 6,400 training windows).
- **Lead-vehicle closing rate: essentially no.** A 1.8 m-wide car at 40 m is ~12 px wide (¾ of one 16-px patch); closing at 5 m/s its width changes by ~1.5 px/s — **~0.3 px across the 200 ms three-frame stack** and ~1.2 px across the 0.8 s predictor window, after the patch embedding and a 4×4 average pool have diluted it into a cell 64 × 64 px wide (ESTIMATED). Range-rate is therefore not observable at the distances that matter for braking at speed. On this corpus and at 2 s this costs little (E-GOAL-1, 4.1), but it is decisive for closed-loop safety (TTC), which no open-loop metric here measures.

### 4.4 Why the no-speed control collapses to 3.0 m

It is the same mechanism as the tactical head: without v0 the step readout must infer per-step displacement from a latent whose speed content is R² ≈ 0.77–0.86 with shrinkage, so the per-step error is a *systematic* fraction of speed, and SE(2) accumulation over 20 steps turns it into metres (a 2 m/s speed error ≈ 2.5 m mean error over 0.5–2 s; ESTIMATED). The speed input is not a "shortcut" the model should be weaned off — it is the only metric-speed channel the architecture has. The v2/v3enc "anti-shortcut" levers that zero-filled it and penalised the encoder for encoding it (§Q6) are the programme's clearest self-inflicted architectural wound.

---

## Q5 — Latent state and resolution

### 5.1 What the state is, geometrically (CODE + ESTIMATED)

| | v1 (256×256 pinhole) | v5f (256×640 canvas, 176×624 valid, 120° cylindrical) | REF-C decoder input |
|---|---|---|---|
| angular sampling | f = 266 px ⇒ ~4.6 px/deg at centre, 51.4° HFOV (`geometry.py:277`; HFOV per `flagship_v4.py:109` comment) | **5.333 px/deg** (INHERITED `…/2026-07-28-resolution-gain/RESOLUTION_GAIN.md`) | same frames as v1 |
| patch grid | 16×16 = 256 tokens × d768 | 11×39 = 429 tokens (INHERITED `readout.py:91-95` comment) | 8×8 conv map × F (704 base / 992 XL) |
| **what the planner / predictor sees per frame** | **4×4 cells × 128 = 2048 floats**; each cell = mean of 16 tokens = **64×64 px ≈ 13.8°×13.8°**, then a 768→128 linear | **the same 2048 floats**; each cell ≈ 3×10 tokens ≈ **48×160 px ≈ 9°×30°** — horizontally **2.2× coarser** than v1 because `grid_w` stays `None` (`config.py:81-90`, `readout.py:47-58`) | **45,056 floats (base) / 63,488 (XL)** — **22–31× more** than one WM state; 4× the cells, 5.5–7.75× the width per cell |
| effective dimensionality | active_k ≈ 19, covariance effective rank ≈ **30 ≪ 2048** (INHERITED `TanitAD Research Hub/Architecture & Inference/Research/STATE.md` 2026-07-18 E2) | not measured | — |

Pixels per object at f = 266 px (ESTIMATED, pinhole): a 1.8 m-wide car is **24 px at 20 m, 12 px at 40 m, 8 px at 60 m** (1.5 / 0.75 / 0.5 patches); a 0.5 m pedestrian is 6.7 px at 20 m; a 0.3 × 1.0 m signal head is 2 × 7 px at 40 m. At 25 m/s a 2–4 s headway puts the lead at 50–100 m — **5–10 px, i.e. under half a patch, averaged into a 64×64-px cell with 15 other tokens.**

### 5.2 Is the readout a bottleneck?

- **For agents, lights and closing rates: yes, severely** (ESTIMATED from the geometry above plus the ~30 effective dimensions). A state whose covariance has ~30 effective directions, most of them carrying ego-motion (in-latent yaw R² 0.89, speed R² 0.77–0.86 — INHERITED), has no room to represent a set of agents. The in-programme comparison that isolates the substrate points the same way: on the *same* decoder algorithm, **REF-C on its 8×8×F map proposes ~2× better than v1.5 on the WM state (oracle-in-fan 0.164 vs 0.338) while v1.5 mis-ranks less (0.235 vs 0.454)** (INHERITED registry §4.1, line ~1393). Proposal quality tracks the richness of the planner-visible map.
- **For 2 s open-loop ego ADE on this corpus: not the binding constraint.** Perfect lead-vehicle state from privileged tracks is worth ~2 recovery points (E-GOAL-1), and a 1.5× angular-resolution change is a separated null on the two probes tested (intersection-class AP, image-only speed R²; INHERITED `RESOLUTION_GAIN.md`). ⚠️ Both of those probes are *global* (scene class, ego speed); neither can see a 10-px lead car, so the resolution null says nothing about object-level observability — it should not be quoted as "resolution does not matter".
- **Temporal context is short for the tasks the hierarchy claims.** The encoder sees 200 ms (three frames at 100 ms), the predictor and every planner 0.8 s (8 states). Nothing in any arm has memory beyond 0.8 s — too short for closing-rate estimation at range and, by construction, for anything "strategic" (§2.1 fact 2).

**Net:** the 4×4×128 readout was chosen for a WM whose job is ego-motion (A7: "never global-pool", `readout.py`), and for that job it is adequate. It is the wrong substrate for a *decision* layer, and it makes a cost model over agents (Q3(a)) impossible in principle. v5f's wide FOV buys periphery but, because the state stays 2048-d, spends it at 2.2× coarser horizontal resolution per cell. The v6 fix is to let the planner read the encoder's patch tokens (or a finer planner-only readout) while the WM keeps its compact state (R-6).

---

## Q6 — Representation: frozen vs from-scratch, and why the later lines failed

### 6.1 REF-A (frozen DINOv2 / I-JEPA) — what actually failed

- **REF-A fails on the WM-fidelity surface, i.e. while being handed the expert's future controls:** 2.1675 [1.9081, 2.4212] (DINOv2-4B) and 3.0471 (dyn-in) vs **0.4518 for a zero-parameter bicycle given the same controls** (MEASURED registry §6; §1.1 row b). A model that cannot convert *given* controls into metres is failing at **metric ego-motion decoding**, not at "understanding the scene". Its signature confirms it: **94.2 % longitudinal**, speed bias +0.77 m/s, train fwd-ADE 0.65 → held-out 2.92 (**4.5× generalisation gap**), `vision_use` 3.4 % on the pre-fix arm (INHERITED registry §2.3).
- **H4 is confounded by the input, not only by freezing.** REF-A encodes **only the latest RGB frame** of each 3-frame stack (`stack/scripts/dino_precompute.py:44`: `latest = frames[:, -3:]`), resized 256 → 224 and stored fp16; motion must be recovered from differences between per-frame semantic features by a "temporal adapter". The flagship encodes the **9-channel three-frame stack**, so optical flow is linearly available to its first (patch) layer, and its encoder is shaped by ~9 units of ego-motion loss mass (invdyn 2.0 × 3 + fwd 1.0 × 3). DINOv2's objective is built on multi-crop augmentation invariance (PUBLISHED, Oquab et al. 2023, arXiv 2304.07193), i.e. to *discard* the small geometric shifts that encode 100 ms of ego-motion. CODE + PUBLISHED; the causal attribution is HYPOTHESIS.
- **Reading, architecturally:** H4 is established for *"frozen single-frame semantic features + a temporal adapter as the substrate for metric ego-motion integration"* — not for *"pretrained encoders are worse for driving world models"*. PUBLISHED counter-evidence exists for planning on frozen DINOv2 patch features in simpler domains (DINO-WM, Zhou et al. 2024, arXiv 2411.04983). What the result really tells us is that **the task signal the programme optimises and measures is visual odometry + control integration**, for which a from-scratch motion-input encoder is the right tool. It says little about scene understanding, which nothing in the objective asks for.

### 6.2 v2 / v3enc — architecture-caused, by lever design

The `--v2` pack (registry §1.3) switched on three "anti-shortcut" levers that between them attack **every metric-speed channel the architecture has**: `v2_ego_dropout 0.25` zero-fills v0 **in the operative action channel** (`flagship_losses.py:229-230` — and 0.0 m/s is an in-distribution "stationary", the X15 zero-fill lie); `v2_encoder_ego_decorr` penalises the encoder for linearly encoding [v0, yr0] (`flagship_losses.py:391-393`); `v2_invdyn_gradscale 0.25` scales down the metric-inverse-dynamics gradient, the loss that teaches the encoder metric ego-motion (`metric_dynamics.py:354-365`). Result: encoder speed-probe R² **0.30** (v1: 0.861), step-1 speed error +2.39 m/s, error 79 % longitudinal, +9.74 m overshoot (INHERITED registry §1.3). v3enc kept decorr at 0 until 10 k and still reached only R² **0.393** at 10 k with gradscale 0.5 and the zero-fill intact (INHERITED §1.4); its operative is a near-pure v0 integrator (permuting v0 across windows: 0.4 s error 0.26 → **2.09 m**; MEASURED `taniteval/results/postmortem_b_egodropout_v3enc10k.json`). **HYPOTHESIS (three points, confounded by training step and by the other levers): encoder speed representability tracks the invdyn gradient scale — 1.0 / 0.5 / 0.25 ↔ R² 0.861 / 0.393 / 0.30.** Cheapest check: two 5 k-step `flagship4b_reduced` arms differing only in `v2_invdyn_gradscale` (≈ 1 A40-day).

### 6.3 v4.x / v5 — recipe-caused in the registry's reading; the architectural read

Every arm that put a planner on the WM state and let its gradient into the trunk degraded the world model: v4 hot trunk canary 0.452 → ~1.3; v4.2 0.7222; v1.6 (4 ViT blocks + predictor unfrozen) **0.452 → 1.1022 (+144 %)**; v4 from scratch reached only **1.1409** at 30 k (INHERITED registry §1.4b, §1.5). The architectural common factor: **the planner and the WM compete for one ~30-effective-dimension, 16-cell state**. Everything the planner needs (agents, finer geometry, longitudinal cues) must be written into the same 2048 floats the JEPA/grounding losses shape for ego-motion; gradient surgery found the two gradients orthogonal on average (cos +0.0043, INHERITED prior review F7), which is exactly the signature of two objectives wanting *different* representations rather than fighting over one direction. HYPOTHESIS; v6 separates them (R-6).

**⚠️ Code-vs-registry discrepancy found in passing (CODE).** The registry §1.5 parameter table lists v1's `tactical_policy`, `tactical_pred` and `strategic_policy` as "REMOVED" in the v4 line and describes "three planners". The production trainer builds `WorldModel(flagship4b_config())` (`stack/scripts/train_flagship_v4.py:1282,1294`) — which *contains* all three v1 brains, trained as auxiliaries through `flagship_loss` (`train_flagship_v4.py:98-102`) — plus **one** `FlagshipV4Head(v4_config())` (the operative planner, `:1320,1345`) and a 4-scalar MLP `GoalScalarHead` that reads the operative state and feeds nothing (`:1354`; `train_flagship_v4.py:133-141`). The tactical planner ② and strategic planner ① are **not instantiated** (the module docstring, `:19-21`, says so). And the H15 field (22 M) is instantiated but **no loss in this trainer reaches it** (no `h15_loss` call anywhere in `train_flagship_v4.py`). ⇒ **v5f is architecturally "v1's WM + v1's auxiliary brains + one REF-C-class operative planner + an unused goal MLP + 22 M of untrained parameters"** — not a three-planner hierarchy. Every published v4 MODE-B number was also produced with future-derived route/VTARGET goal tokens (`--goal-mode oracle` default, `stack/scripts/eval_flagship_v4.py:91-107`).

---

## Q7 — Imagination

**Three different mechanisms share the word** (CODE): (i) **H15**, a 22 M sector-masked token-level belief field trained with a *one-step* heteroscedastic NLL (`train_flagship4b.py:224-231`; `imagination.py:121-162`), on no inference path; (ii) **blind predictor rollout** — every "grounded rollout" in the programme, the headline included, is already a 20-step blind rollout (`taniteval/taniteval/blindimag.py:1-20`); (iii) **v5f probe tokens** (§Q3).

**The anti-calibration finding is expected by construction.** Under blind multi-step rollout, fidelity falls to chance by k ≈ 3–4 while the predicted log-variance *shrinks* (−9.461 → −9.564 on the operative model; INHERITED `…/Research/STATE.md` 2026-07-18 E1; prior review F3). The σ-head was only ever trained at one step on encodings of real frames; a single-horizon Gaussian NLL cannot learn that error grows with horizon, and applying it to its own rollouts is out of distribution. Two further design issues (CODE, consequences HYPOTHESIS): the H15 target branch is **not stop-gradient** (`train_flagship4b.py:229`, `tok_true = model.encode_tokens(fut[:, 0])`), so its NLL rewards the encoder for making next-frame *tokens* predictable, while SIGReg polices only the pooled 2048-d state; and the advection prior is a 2-D flow on a 16×16 token grid — for the agent-permanence use it was designed for, the agents are sub-patch (§Q5).

**What imagination has actually bought (MEASURED unless marked):** re-planning every tick on the imagined latent beats committing to one open-loop plan, **−0.1711 [−0.2615, −0.0838]** closed-loop (`closedloop_flagship-30k.CORRECTED.json → imagination_comparison`); one v1-WM reference roll under the held last action as a selection reference, **−0.2918 [−0.4233, −0.1598]** on v4's fan (INHERITED `wm_reference_select.py:30-40`; but **+0.2090** harmful when v4's own WM scores its own fan); the imagined latent beats a frozen percept only up to ≤ 6 s, peaking at 1 s (INHERITED V5_PLAN §6). **Per-candidate imagination-consistency scoring is refuted** (E-V5-1).

**Can this design work?** As a *consequence model for choosing*, only with three things it does not have: a **cost** evaluated on imagined states (Q3(a)), **horizon-aware uncertainty** (train σ on the model's own multi-step rollouts, or use disagreement between two predictor heads — the canary-proxy study found 2-WM ensemble features predict WM error at R² 0.5526 where v0 gives 0.070, INHERITED V5_PLAN §8), and **candidate-conditioned** rolls restricted to a small top-K (Q8). As built — a one-step σ, a probe vocabulary shared by all candidates, and no cost — **no**. The one role with measured positive value is the cheapest: **one reference roll per window as a prior / selector feature.**

---

## Q8 — Deployability

| quantity | value | class · source |
|---|---|---|
| v1 "plan tick" A40 batch 1 p50 (fp32 / tf32 / amp16) | **97.32 / 97.70 / 123.83 ms**; encode 8 frames 28.1 ms + 20-step rollout 85.2 ms; one predictor call 3.86 ms | MEASURED `taniteval/results/eff_flagship-30k.json` |
| same, all levers (CUDA graph + encoder cache + fp16) | **18.75 ms** | MEASURED `taniteval/results/eff_levers_flagship-30k.json → fp32.levers.all_levers` |
| rollout length sensitivity (eager fp32) | k=20 99.7 · k=10 56.9 · k=5 36.8 ms | MEASURED same file, `rollout_k_latency` |
| REF-C-base / XL plan step p50 (fp32 / tf32 / amp16) | **21.8 / 15.8 / 15.9 ms** · 44.1 / 27.8 / 21.0 ms | MEASURED `eff_refc-base-30k.json`, `eff_refc-xl-30k.json` |
| Thor, v5f geometry (176×624): encoder 8 frames fp32 / bf16 | 187.8 / **27.8 ms** | INHERITED `…/incoming/2026-08-02-thor-deployment-profile/THOR_PROFILE.md` |
| Thor: 20-step predictor roll eager / CUDA graph | 80.9 / **69.6 ms**; combined bf16 encoder + graph roll **98.6 ms p50 = 98.6 % of the 100 ms budget** | INHERITED same |
| Thor batching | SMs saturate at batch 8; throughput flat 12.3–14.1 windows/s across a 6× batch range | INHERITED CLAUDE.md (MEASURED 2026-08-03) |
| Thor, 9-candidate fan through a batch-9 TRT engine | 1.09× one candidate (serialised: 9×) | INHERITED `stack/tanitad/models/fourbrain.py:608-616` docstring |

**Reading.** (1) **The timed "plan tick" is not a decision.** It is one rollout of *given* actions (the WM-fidelity path). A WM-scored decision multiplies that roll by the number of candidates: a 256-candidate fan on Thor is ~32 saturated batches × ~70 ms ≈ **2.2 s, ~22× over budget** (ESTIMATED); even the 72 %-reachability-pruned fan is ~6× over. (2) **The sequential 20-step roll is launch-bound, not FLOP-bound**: 91 M predictor parameters over 8 tokens per step, 3.7–4.3 TFLOPs achieved vs REF-C's 15.9–25.2 (INHERITED registry §6 reading 3). (3) **What can be parallelised or distilled:** encoder caching (one new frame per tick; measured 57 → 33 ms on A40 with graphs); direct multi-horizon heads instead of 20 sequential steps (the predictor already emits k = 1, 2, 4 in one pass; the v4 line built horizon-scaling direct-head baselines); a smaller, wider predictor (8-token sequences do not need 10 × d768 layers); rolling at 5 Hz (k = 10); batching top-K ≤ 8 candidates to match Thor's SM saturation; and distilling any WM-informed selector back into the single-pass decoder. (4) The strategic + tactical brains cost 8–10 ms per tick on A40 (MEASURED eff JSON, `hierarchy_strategic+tactical`), negligible — and, since the cadence is unimplemented (§2.1), never amortised.

**Budget implication for v6 (ESTIMATED):** a REF-C-class decoder (~16–28 ms A40) + a per-candidate scorer over 256 candidates (an MLP/cross-attention over candidate features: ~1–3 ms) + **one** reference WM roll at k = 10 with a cached encoder (~35–45 ms Thor) fits 10 Hz on Thor with margin; a top-8 imagined-consequence pass (~1.1× one roll) is the most that can be added.

