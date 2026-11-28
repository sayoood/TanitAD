# P9 — TEST/VALIDATE THE IDM: the first four-family read, and a leak finding that changes what the IDM's numbers mean

**Date** 2026-08-03 · **Substrate** dev box (RTX 4060), **0 pod GPU-h** · **Suite** `pytest -q` 1680 passed / 12 skipped / 2 xfailed

---

## 0. HEADLINE

1. ⛔ **`ADE` cannot see a total lateral-decision failure.** A trajectory that is the *ground truth with the lateral sign flipped* scores **ADE 0.5538 m [0.4377, 0.6755]** against the shipped IDM's **0.4482 m [0.4121, 0.4866]** — overlapping intervals — while its lateral manoeuvre balanced accuracy is **0.3333 (chance)** versus the IDM's **0.7722**. This is the binding four-family rule demonstrated on real data, not asserted.
2. ⛔ **The IDM's manoeuvre quality is far below what its R² implies.** Speed R² **0.9986**, steer R² **0.9929** — but lateral manoeuvre BA **0.7722 [0.7216, 0.8553]** and longitudinal **0.6865 [0.6123, 0.7377]** on a 3-way decision whose chance level is 0.3333. Accelerate recall is **0.4539**: the IDM misses over half of all accelerations.
3. ⛔ **NEW, and it re-labels every comma number the IDM has ever published: the local comma val cache IS the head's `cmx_` training pool.** Two independent probes. Consequence: **there is no held-out comma substrate on the dev box**, and the program's "content-clean" audit covers only **42 of 121** comma training episodes (**35 %**).
4. ✅ The instrument itself is validated: it separates the shipped head from three negative controls with **separated** paired intervals, in both directions.

---

## 1. WHAT EXISTED, WHAT IT SCORED, ON WHAT DATA

`idm_head_v4_steer_ens3.pt` (34,841,012 B), recipe R0, k=4 / 9 frames, d_model 256, 3-seed **mean of per-seed predictions**, rung 757 (cm 121 / pai 636), label protocol *REPAIRED (heading_repair, v_min=0.5)*.

**MEASURED (mine, re-derived today from `…/2026-07-27-fleet-sync-idm-steer/raw/idm5_ensemble.json`, not inherited from any prose):**

| channel | pooled R² | pai | comma | pooled MAE | comma MAE |
|---|---:|---:|---:|---:|---:|
| speed | +0.8650 | +0.9312 | +0.7453 | 3.2231 | **3.6429** |
| yaw_rate | +0.9188 | +0.9624 | +0.6948 | 0.01697 | 0.016410 |
| steer | +0.7993 | +0.7858 | +0.8071 | 0.008647 | 0.003633 |
| long_accel | −0.0591 | −0.0369 | −0.2258 | 0.4378 | — |

n = 4,195 windows / 36 episodes (pai 1,203/14, cm 2,992/22). Estimator `paired_episode_cluster_bootstrap`, B=2000.
Also re-verified: `speed` paired vs A0 = **+0.2287 [−0.377, +0.835], not separated** — i.e. the ensemble is **nominally worse than A0 on the one channel the YouTube pipeline ships as `primary`**.

**⛔ The four-family rule was violated outright.** Two independent probes over `stack/scripts/idm_*.py`, `run_idm_*.py` and every `idm*` hub script: **no IDM script imports `four_families`**. Every consumer is world-model side. LONGITUDINAL, LATERAL, TACTICAL and STRATEGIC were all absent; four scalar R² values were the entire published validation.

---

## 2. THE BIGGEST DEFECT, ARGUED FROM PRIMARY SOURCE

**A scalar R² cannot see a manoeuvre error, and the IDM's product IS a manoeuvre label.** The IDM exists to mint pseudo-labels for action-free web video. `IDM_VIDEO_PRETRAIN_DESIGN` ships `speed` + `long_traj` as *primary*. Nothing in the pipeline ever asks *"did it pick the right manoeuvre?"* — so a head that sets speed correctly while inverting turns is undetectable, and would silently poison every downstream pseudo-label.

Two source facts make this concrete rather than theoretical:

* `taniteval/four_families.py:52` hard-codes `DT_S = 0.1`, but the IDM emits **4** waypoints at horizons {5,10,15,20} steps = **0.5 s apart**. Feeding IDM trajectories to that module unchanged reads every speed and yaw-rate **5× too large** — so the family functions were not merely unused, they were *unusable as-is*.
* `four_families.tactical()`/`.strategic()` (`:206`, `:254`) consume a **window dict** with `pred_key`/`gt_key` via `_decision_family` (`:166`), not trajectories — and the IDM emits no manoeuvre class at all (`idm3_arms.py:46` mentions "manoeuvre" once, in prose).

---

## 3. WHAT I IMPLEMENTED

`stack/tanitad/eval/idm_families.py` (+ `stack/tests/test_idm_families.py`, 16 tests).

* **Cadence made explicit.** `geometry(wp, dt)` is `four_families._seq_geometry` verbatim with `DT_S` as a parameter. `test_geometry_matches_four_families_at_10hz` asserts bit-equality with taniteval at `dt=0.1` — same definitions, right cadence. `test_cadence_is_load_bearing` pins the 5× error.
* **The manoeuvre class is FACTORED.** The program's diagnosed root cause of longitudinal blindness is that one 5-way softmax mixes the lateral and longitudinal axes. Reporting a pooled class here would rebuild that defect inside the instrument meant to detect it, so `manoeuvre_classes()` emits a **lateral** class and a **longitudinal** class separately. The legacy `mixed` 5-way is emitted too — **only so its collapse stays measurable**.
* **STRATEGIC is declared UNAVAILABLE with its reason and n**, never dropped.
* `stack/tanitad/eval/ap_ci.py` adds `stat_episode_cluster_bootstrap` / `paired_stat_episode_cluster_bootstrap` so set-level statistics (balanced accuracy, recall, R²) get the program's binding interval instead of being quoted bare.

### Two bugs my own first run produced, and the tests that now pin them

* **A leaked comprehension variable** made all three "paired vs control" deltas compare against the *same* arm — the tell was three identical deltas. Fixed to index `arms[n][1]` explicitly.
* **Balanced accuracy degenerates inside a bootstrap.** Turns are <1 % of a highway corpus, so many episode resamples contain no turn; with one class present, "mean recall over present classes" scores a **blind constant predictor at 1.0**. The first run duly reported the blind control at **[0.3333, 1.0000]**. `balanced_accuracy(..., require_all=True)` now returns `nan` for such draws, they are dropped and **counted** (1,727/2,000 survived on the lateral axis), and `test_balanced_accuracy_require_all_blocks_the_bootstrap_degeneracy` is a regression test.

---

## 4. MEASURED — four families, 50 episodes / 6,900 windows, B=2000

⚠️ Read §5 first: these rows are **IN-CORPUS**, not held out.

### TACTICAL (the family that did not exist before today)

| arm | lateral BA | longitudinal BA |
|---|---|---|
| **shipped `idm_head_v4_steer_ens3`** | **0.7722 [0.7216, 0.8553]** | **0.6865 [0.6123, 0.7377]** |
| NEG1 latents shuffled across windows | 0.3297 [0.3231, 0.3319] | 0.3335 [0.3245, 0.3433] |
| NEG2 blind mean predictor | 0.3333 [0.3333, 0.3333] | 0.3333 [0.3333, 0.3333] |
| NEG3 GT with lateral sign flipped | 0.3333 [0.3333, 0.3333] | **1.0000 [1.0000, 1.0000]** |

Paired (shipped − control), episode-cluster bootstrap, **all separated**:
lateral vs blind **+0.4389 [+0.3883, +0.5220]**; longitudinal vs blind **+0.3532 [+0.2790, +0.4043]**; longitudinal vs NEG3 **−0.3135 [−0.3877, −0.2624]** — the instrument correctly reports the IDM as *worse* than a longitudinally-perfect arm.

Per-class recall, shipped arm:
* lateral — right **0.5357** (support 56) · straight 0.9844 (6,785) · left **0.7966** (59)
* longitudinal — decelerate **0.6191** (575) · cruise 0.9865 (5,869) · **accelerate 0.4539** (456)
* mixed 5-way BA **0.6653** — **below both factored axes**, with accelerate recall dropping 0.4539 → 0.4226 and decelerate 0.6191 → 0.5961. The mixed class measurably loses longitudinal information on turning windows, exactly as the program's diagnosis predicts.

### LONGITUDINAL / LATERAL / STRATEGIC / ADE

| | shipped | NEG1 shuffled | NEG2 blind |
|---|---|---|---|
| ADE@2s (m) | **0.4482 [0.4121, 0.4866]** | 16.3159 | 12.6727 |
| scalar speed MAE (m/s) | **0.2938 [0.2724, 0.3149]** | 13.014 | 10.1235 |
| scalar yaw-rate MAE (rad/s) | **0.0831 [0.0098, 0.2279]** | 0.1022 | **0.0916** |

* LONGITUDINAL: traj speed MAE 0.3135, along-track MAE 0.3499 m, accel MAE 0.2488 m/s². **`distance_keeping` UNAVAILABLE, n=0** — comma2k19 ships no object annotation; PhysicalAI ships `obstacle.offline` but the ingest does not read it. A work item, not a pass.
* LATERAL: heading MAE 0.01756 rad (n=25,204), **curvature MAE 0.0065 1/m (n=18,842)**, traj yaw-rate MAE 0.0197 rad/s, cross-track MAE 0.1952 m, final cross-track MAE 0.4152 m.
* ⛔ **The scalar `yaw_rate` channel is not evaluable on this substrate.** Its R² is **0.0027**, and its MAE (0.0831) has an interval overlapping the **blind mean predictor's** (0.0916). Meanwhile the *trajectory-derived* yaw-rate MAE is 0.0197. The scalar channel's GT is label-noise dominated here — the documented comma heading fragility (9.1 % of windows carrying 61 % of the sum of squares). **Use the trajectory-derived lateral geometry; do not quote the scalar yaw R² on comma.**
* STRATEGIC: **UNAVAILABLE, n=6,900**, reason recorded — no route/goal label on any IDM substrate; comma has none, and PhysicalAI-AV is settled at five probes as carrying no map, lane graph or route signal, with clip-local metres and no GNSS.

---

## 5. ⛔ THE LEAK FINDING — two probes

**Claim: local `comma2k19-val-61c46fca8f7f` is the `cmx_` extras pool the v4 ensemble trained on. There is no held-out comma substrate on the dev box.**

**Probe 1 — performance signature.** Banked held-out comma speed MAE = **3.6429 m/s** (R² +0.7453). Mine on the supposedly content-clean 50 = **0.2938 m/s** — **12.4× better**, on the *same corpus and the same head*, through a **re-encoded** latent substrate that the program has already flagged as *worse* than the pod's. A worse substrate cannot produce a 12× better score. Memorisation can. The 9 known-leaked episodes score **0.2997 m/s** — statistically the same as the "clean" 50.

**Probe 2 — provenance, from the checkpoint's own leak block.** `idm5_ensemble.json.leak_check` records `excluded_val_collisions`: `cmx_00008 ↔ cm_00018` and `cmx_00020 ↔ cm_00039`. From `anchor_overlap.json`, local `ep_00008` and `ep_00020` are byte-identical to pod `ep_00018` / `ep_00039`. **The extras are indexed exactly as the local cache**, which holds 90 episodes — the right size to supply the 79 comma episodes beyond V3_TRAIN's 42 that make up `pool_cm: 121`. The leak check excluded only episodes colliding with its own 36 VAL episodes; **every other local episode went into training.**

**Consequences.**
* The §4 numbers are an **IN-CORPUS UPPER BOUND**. That does not weaken the tactical finding — it strengthens it: under the easiest possible condition, with the head having seen these episodes, lateral manoeuvre BA is still only **0.7722** and accelerate recall **0.4539**. Held-out will be worse.
* The program's content-clean claim on comma is verified against **42 of 121** episodes. `anchor_overlap.json` cleared `A0_HELDOUT_cm_40_70 × V3_TRAIN = 0`, which is true and was **not** the sufficient check.
* ⛔ **`E-IDM-CLEAN20` as planned cannot be run on the dev box**, and CLEAN20 itself is defined on cache `76b6e94a97a1`, of which only **2 of 36** episodes exist locally by content.

---

## 6. WHAT IS STILL OPEN

* A genuinely held-out comma read needs the pod's `/root/idm2/lat` or a fresh comma2k19 pull whose fingerprints are checked against **all 121** training episodes.
* `distance_keeping` needs `obstacle.offline` in the ingest.
* STRATEGIC needs a route-labelled substrate (AlpaSim / NuRec `map.xodr`).
* **Escalation, unchanged and still live:** `MODEL_REGISTRY.md` has **no `idm_head` row**, and §8.1 #6 (line 1852) still publishes the withdrawn 77.5 %-leaked *"held-out speed R² 0.930"*. Under the source-of-truth rule every number here is admissible as raw-eval-JSON only.

---

## 7. EVIDENCE CLASS

| claim | class |
|---|---|
| §1 banked table, paired-vs-A0, leak_check block | **MEASURED (mine, re-derived from `idm5_ensemble.json`)** |
| §4 all four families, ADE, CIs, controls | **MEASURED (mine, `results_idm_four_families.json`)** |
| §5 probe 1 (12.4× MAE gap) | **MEASURED (mine)** |
| §5 probe 2 (`cmx_` indexing) | **MEASURED (mine, from `idm5_ensemble.json` + `anchor_overlap.json`)** |
| "no IDM script imports four_families" | **MEASURED (mine, two probes)** |
| comma heading fragility, 9.1 %/61 % | **INHERITED** (`ANCHOR_SETTLEMENT.md`), not re-derived |
| no map/route in PhysicalAI-AV | **INHERITED** (CLAUDE.md five-probe settlement) |
