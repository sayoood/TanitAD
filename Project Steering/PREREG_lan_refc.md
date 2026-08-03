# PRE-REGISTRATION — LAN (Lane-Anchored Navigation) for REF-C

**Date:** 2026-08-03 (Europe/Berlin) · **Author:** LAN/REF-C implementation agent ·
**Status:** code + instruments DELIVERED and STAGED, **no training launched**, no GPU spent.
**Estimator for every interval below:** paired episode-cluster bootstrap over the 40 val episodes,
`taniteval/taniteval/ci.py::paired_episode_cluster_bootstrap`, `n_boot = 2000`, unit = **episode**.
`overlapping_holdout_se` is never called (CLAUDE.md: it biases the point estimate, −6.67 %…+11.69 %,
bidirectional).

**This document is written BEFORE any LAN number exists. Both outcomes are committed in §6.**

---

## 0. First: what "LAN" resolves to, and what it does not

The brief asked me to establish the referent before building. Two candidate threads exist and I
probed both.

| candidate | status | evidence |
|---|---|---|
| **LAL-v2 anticipation** (the "implemented but unmerged 12 days" item) | ⛔ **NOT the referent — it is MERGED and has been for weeks.** | `compute_lal_v2` + `decel_onset_index` + `LAL2_*` constants live at `stack/tanitad/eval/metrics.py:224`; `run_scenario_suite` emits `LAL_v2_s` at line 485; `stack/tests/test_lal_v2.py` is tracked (`git ls-files`) and passes. The intake's own verdict block records *"Integrated as: stack/tanitad/eval/metrics.py … Suite 188 green."* |
| **LANE work** — lane graph / lane-anchored route conditioning | ✅ **the live thread, and the one that touches REF-C** | `…/Research/2026-08-02-nurec-xodr-map/XODR_MAP.md` (yesterday) §8.3 states the connection verbatim: *"The programme noted REF-C evaluates with `nav_cmd=None`, confounding the … refusal. A real route signal now exists to condition on."* |

**⇒ LAN is read as LANE-ANCHORED NAVIGATION: the lane-graph-derived route/goal signal, and the
REF-C route input it is supposed to repair.** The lane-*keeping* half of the lane thread
(`corridor_departure_rate`, peak XTE) is already merged as `taniteval/taniteval/corridor.py` and is
re-used here as a readout rather than rebuilt.

⚠️ If the PI meant something else by "LAN", the disambiguation above is the thing to correct first —
everything below hangs off it.

---

## 1. The defect, stated as a chain of MEASURED facts

Every link is MEASURED with a primary-source path. None is INHERITED.

| # | fact | class · source |
|---|---|---|
| 1 | REF-C's entire route input is a **4-way one-hot** `nav_cmd` ∈ {follow, left, right, straight}, concatenated with `v0` into a 2-layer MLP (`d_meas_in = 1 + 4`). | **MEASURED** — `stack/tanitad/refs/refc.py`, `RefCModel.__init__` / `forward` |
| 2 | The label thresholds **net heading change over 15–25 s** at 45°, and returns `(follow, valid=False)` when <15 s of future exists. `dyaw = κ·v·t`, so it conflates a gentle motorway curve with a junction turn — the module's own v2 docstring says so. | **MEASURED** — `stack/scripts/refb_labels.py:138-169`, and the v2 rationale block at `:230-259` |
| 3 | `nav_valid_frac` is **0.21–0.25 in all four arms, including the deployed v1**. | **MEASURED** — RETRACTION_LOG 2026-07-21 |
| 4 | The route head has **`route_skill_vs_chance = 0.0`** (pure command echo); the gate metric `nonav_route_beats_majority` FAILS, straight 240/240. | **MEASURED** — `stack/tests/test_refb_labels_v2.py:224`; `stack/tanitad/config.py:222` |
| 5 | **Every published REF-C number is decoded with `nav_cmd=None`**, which `refc.py:795` substitutes with index 0 for the whole batch. The route pathway is a **constant** at eval. | **MEASURED** — `refc.py:795-796`; flagged in `stack/tanitad/train/heldout_gate.py:262` |
| 6 | This is the **C6 confound logged twice**: 2026-07-21 (*"nearly designed the hierarchy away"*) and 2026-07-25 (hierarchy-proof pre-condition #1, *"a working route input"*). | **MEASURED** — `Project Steering/RETRACTION_LOG.md:41`, `:68` |

**Consequence:** the route input is degenerate at train time *and* constant at eval time. A decoder
in that position learns the marginal, and any "the hierarchy does not help" reading taken from it is
a null from an instrument structurally unable to see the effect — the exact error class the
2026-07-25 retraction names.

---

## 2. What LAN is

Replace the 4-way scalar with a **geometric route corridor**, ADDITIVELY (`nav_cmd` is kept):

* **K route anchors at fixed ARC-LENGTH** — default (20, 40, 80, 160) m — not at fixed time. This
  removes the κ·v·t conflation by construction, and is pinned by a test that the encoding is
  **bit-identical for the same road driven at two different speeds**.
* **Per anchor: `[cos bearing, sin bearing, lat_norm, valid]`** in the ego frame (x forward,
  y left — the repo's `_ego` / CCW convention, cross-checked against `refb_labels.ego_frame`).
* **No along-track distance is encoded, anywhere.** This is deliberate and is the single most
  important design constraint — see §3.
* **A leak guard**: an anchor is masked unless its arc-length exceeds
  `max(2 s GT path length, v0 · 2 s) + min_lead_m`. Taking the **max** makes the guard conservative
  (it can only mask more). Pinned by a randomised test over v0 ∈ [0, 30] m/s.
* **Route dropout p = 0.5** at train time, per sample, so the planner can never become
  route-DEPENDENT — it must still drive when the route is missing, which on a corpus with 0.21–0.25
  nav validity it usually is.

**Two suppliers, one contract:**

* **S1 `ego_future`** — the ego's own future path, arc-length resampled. The **only** supplier
  available on the parity corpus `physicalai-train-e438721ae894`, which has no map (settled at five
  probes, CLAUDE.md).
* **S2 `lane_graph`** — snap to lane centrelines, read the route off the graph. Available for NuRec
  (`map.xodr`) and Argoverse 2 (`tanitad.data.argoverse2.LaneGraph`).

**Model seam (`RefCConfig.graft_lan`, default False, byte-identical when off):**

1. `lan_enc` (MLP) → `decoder.lan_to_cond` (**zero-init**) added to the decoder condition;
2. `decoder.lan_gate` — ONE scalar, **init 0**, on a **param-free geometric compatibility**
   `cos(φ_anchor − θ_route)` between each trajectory anchor's terminal bearing and the route
   bearing. Selection among the fan is where a route can act at all, and a geometric score cannot
   become a route-shaped shortcut the way a learned `route → n_anchors` matrix could.

⚠️ **A known, stated property, not hidden:** with `lan_to_cond.weight == 0` the chain rule sends
exactly **0** into `lan_enc` at step 0 — byte-identity at init and a live input gradient at init are
mutually exclusive for a zero-init gate. `lan_to_cond` itself has a non-zero gradient, so one
optimiser step opens the path. Both facts are pinned by tests
(`test_the_two_seams_that_carry_the_route_have_live_gradients_at_init`,
`test_the_route_encoder_is_gradient_blocked_at_init_and_unblocks_after_a_step`). The programme's
existing `ctx_to_cond → StrategicCtx` graft has the identical property.

---

## 3. ⛔ The prior that constrains this whole experiment — and it is adverse

`GOAL_INPUT.md` (2026-07-27; paired episode-cluster bootstrap, B = 2000, 40 ep / 881 win) MEASURED
how the oracle-goal advantage splits on the **2 s selection surface**:

| what is oracle | what is learned | recovery of the −0.2705 | separated |
|---|---|---:|---|
| **along-track** (how far) | cross-track | **+83.7 %** | ✅ |
| **cross-track** (which way) | along-track | **+2.9 %** | ❌ |

**A route / map / lane-graph signal supplies lateral topology.** On ADE@2s that axis is worth
**≈ 3 %, not separated.** Two things follow, and both are pre-registered rather than discovered
later:

1. **LAN must not encode along-track distance** — that axis is worth 83.7 % *and* is the answer to
   the prediction task, so encoding it would manufacture a win. Hence bearing + normalised lateral
   offset only.
2. **ADE@2s is NOT the primary readout, and a null there is the PREDICTION, not a disappointment.**
   Reading LAN through ADE@2s would repeat the 2026-07-25 error of judging a strategic quantity with
   a 2 s instrument that a paired horizon sweep proved blind to an 18 s failure.

The horizon-capable instrument already exists and is merged: E1a (2026-07-25) MEASURED
corridor-departure **0.0035 → 0.5877** going from K=20 (2 s) to K=185 (18.5 s), peak XTE
0.35 → 38.94 m, paired Δ +0.5842 [0.5071, 0.6565] **separated**, with the OOD envelope ≤ 1.30
(genuine in-distribution failure). E2a localised the lever: the lateral offset **is** representable
(oracle R² 0.72, ceiling ρ 0.91) and the loss is **91 % downstream** — the planner ignores available
information. **A route input is precisely a mechanism for telling the planner which lateral option
to take.** That is the hypothesis; §6 says how it dies.

---

## 4. Measurement already made (MEASURED, 2026-08-03, dev box, 0 GPU)

**Question:** S1 (map-free) is the only supplier the parity corpus can have. Is it a faithful
stand-in for a real lane-graph route? That is a measurement, not an assumption.

**Method:** on the banked NuRec artifacts (scene `00040136-e651-4abd-991d-0655ccda9430`, 356
driving-lane centrelines, 340 directed edges, 299 georeferenced ego poses), build the LAN corridor
both ways at every pose with the identical leak guard and compare.
Reproduce: `python stack/scripts/lan_probe.py --agreement --centerlines … --edges … --track …`

| snapping | route lanes | hops on graph | **pos L2 median** | mean | p90 | lat median | bearing median | side agree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nearest-neighbour (ungated) | 30 | 16/29 (0.55) | **4.685 m** | 19.801 | 57.479 | 2.529 | 1.90° | 0.791 |
| **+ heading gate 60° + 1 m hysteresis** | 19 | 12/18 (0.67) | **1.186 m** | **1.459** | **2.732** | **0.711** | **0.41°** | **0.844** |
| gate 35° + 3 m hysteresis | 18 | 13/17 (0.76) | 1.186 | 1.230 | 2.535 | 0.658 | 0.40° | 0.849 |

n = 275 comparable samples (of 298; the rest have no commonly-valid anchor).

**Three readings, kept separate:**

1. ✅ **S1 ≈ S2 to a median 1.19 m / p90 2.73 m.** That is inside the 1.75 m lane half-width the
   programme's own lane-keeping instrument uses as its corridor threshold. **The map-free supplier
   is licensed.**
2. ✅ **The heading gate is load-bearing, and the improvement is diagnostic, not cosmetic** — mean
   19.80 → 1.46 m (**13.6×**), p90 57.5 → 2.73 m (**21×**). The ungated disagreement was
   route-reconstruction detours among near-coincident junction roads, exactly the failure the XODR
   probe reported (3 of 14 hops) and named heading-gated matching as the fix for. A pinned test
   asserts the gate must beat the ungated version by ≥5×, so this cannot silently regress.
3. ⚠️ **`side_agree = 0.844` — 15.6 % of anchors disagree on WHICH SIDE.** Not swept under the rug:
   that residual is in the same channel LAN exists to carry, and it bounds how clean the lateral
   topology signal can be on a lane-graph-supplied route.

**Honest limits.** *n = 1 scene* (RETRACTION_LOG C5 — n=1 headlines). One corpus (NuRec, not the
parity corpus). The hysteresis/gate values are chosen from a sweep on **this** scene; the verdict
does not swing across the sweep (every gated+hysteresis setting lands at median ≈ 1.19 m) but the
exact constants are a fit and must not be quoted as tuned optima. `hops_on_graph_frac` reproduces
the XODR probe's own 16/28 over the full record, which is the cross-check that this implementation
matches the published one.

---

## 5. Arms, and what differs (C6 discipline — every difference named)

| | **BASE** | **LAN** |
|---|---|---|
| checkpoint | REF-C-base, canonical recipe | same recipe |
| flag delta | — | **`--graft-lan` only** |
| `nav_cmd` input | v1 derivation, unchanged | v1 derivation, **unchanged** |
| route CE target | unchanged | **unchanged** (LAN touches an INPUT, never the aux target) |
| extra params | 0 | `lan_enc` + `lan_to_cond` + `lan_gate` (≈ 0.02 % of 104 M at k=4, hidden 64) |
| eval route input | `nav_cmd` supplied (**not** `None` — see §7) | `nav_cmd` supplied **+ LAN corridor** |
| everything else | identical: frames, warp, corpus, parity key `e438721ae894`, skip-hash `f09e44db`, windows, seeds, optimiser | identical |

**Named non-identities (there are two, and they are the point):** (i) the LAN arm receives an extra
input; (ii) the LAN arm carries ~20 k extra parameters. (ii) is bounded to <0.03 % of the arm and is
addressed by the params-matched control **C-null** below.

**Controls:**
* **C-null (params-matched):** LAN arm fed the **inert** (all-masked) corridor at eval. Same weights,
  same params, no route. Separates "the route helped" from "the extra capacity helped".
* **C-straight (majority class):** LAN arm fed a synthetic dead-ahead corridor. On a 74 %-straight
  corpus, a model reading the base rate separates from neither the true nor the straight route.
* **C-identical (harness determinism):** the same input twice must give a **bit-identical** decode.
  Enforced in the instrument at `CONTROL_TOL_M = 1e-6`; a violation reports **INSTRUMENT-FAIL** and
  voids every sensitivity number in that run.

---

## 6. ⛔ THE PRE-REGISTRATION — both outcomes, committed now

### 6.0 E-pre — the launch gate (0 GPU, runs at dataset construction)

`lan_stats()` prints `any_valid_frac` before step 0. **If LAN's `any_valid_frac` is not ≥ 0.50 on
the parity train cache — i.e. at least ~2× the 0.21–0.25 that `nav_valid_frac` achieves — DO NOT
LAUNCH.** A route input that is absent as often as the one it replaces has not fixed the defect and
does not deserve a GPU. *(For calibration: on the CPU smoke corpus the same code returns
`any_valid_frac = 0.5663`, per-anchor [0.0095, 0.5568, 0.3655, 0.0] — MEASURED, synthetic episodes,
not a corpus claim.)*

### 6.1 H-LAN-1 — **is the route input consumed at all?** (primary, mechanism)

**Readout:** `route_sensitivity_m` (mean-over-horizons L2 shift, true vs `mirror` corridor) and
`lat_compliance` (fraction moving the way the route asked; ties excluded), per-window components →
paired episode-cluster bootstrap over the 40 val episodes.

| outcome | condition | what it means |
|---|---|---|
| **CONFIRM** | `route_sensitivity_m` separated from C-null AND `lat_compliance` CI excludes 0.5 | the route input is live and direction-correct. Pre-condition #1 of the Hierarchy Proof Program is **met for the first time** |
| **REFUTE** | `route_sensitivity_m` not separated from C-null | **the architecture, not the label, is the defect.** LAN is refuted as a fix, and the next question is why a live seam is ignored — not more route engineering |
| **PARTIAL** | sensitive but `lat_compliance` CI ∋ 0.5 | the model is *destabilised* by the route rather than following it. Report as a negative; do not spin as "responsive" |
| **VOID** | C-identical fires (control > 1e-6 m) | instrument fault; no number quoted |

### 6.2 H-LAN-2 — **does it improve lateral behaviour at the horizon where the failure lives?** (primary, outcome)

**Readout:** `corridor_departure_rate @ 1.75 m` at **K = 185 (18.5 s)** — `taniteval/corridor.py`,
the instrument E1a proved is the one that can see the failure — plus peak XTE. Paired, same windows.

| outcome | condition |
|---|---|
| **CONFIRM** | paired Δ(BASE − LAN) > 0 with CI excluding 0, junction stratum ≥ overall |
| **REFUTE** | CI ∋ 0, or Δ < 0 (LAN worse) |
| **PARTIAL** | separated in the junction stratum only |

### 6.3 ⛔ ADE@2s — the pre-committed null, and a no-harm bound

**Predicted:** |ΔADE@2s| ≤ ~0.01 m and **NOT separated** (from §3: the lateral axis recovers +2.9 %,
not separated, and the realised ADE base is 0.4714). **This prediction is registered so that a null
cannot later be spun as either success or failure.**

**No-harm bound (a stopping rule, not a readout):** if paired ΔADE@2s is separated in the WORSE
direction by **> +0.02 m**, the LAN arm is rejected regardless of §6.2 — a route conditioning that
costs 2 s accuracy is not deployable. *(0.02 m is the same threshold the INT8 study used as its
falsifier, so it is the programme's existing tolerance, not a new one invented for this arm.)*

**⛔ A separated ADE@2s IMPROVEMENT > 0.05 m is a RED FLAG, not a win.** §3 measured that ceiling at
+2.9 % of 0.2705 ≈ 0.008 m. An effect 6× the measured ceiling means the leak guard failed and the
route is carrying along-track information. Response: **stop and audit the guard**, do not publish.

### 6.4 The four metric families — BINDING, reported per family, never pooled

| family | instrument | expectation, pre-committed |
|---|---|---|
| **LONGITUDINAL** — target-speed accuracy, headway / time-gap / TTC to the lead agent | speed head + lead-agent state from `obstacle.offline` (97.44 % corpus coverage) | **no change expected.** LAN encodes no along-track quantity by design. This family is a **no-harm check**: any separated longitudinal *degradation* is a finding against LAN |
| **LATERAL** — cross-track, heading, **curvature, yaw-rate** error | `taniteval/lateral.py` + `corridor.py` | **where LAN must pay.** §6.2 is the primary |
| **TACTICAL** — manoeuvre-decision quality, selected-vs-executed, confusion over the 5 classes | `maneuver_logits` confusion | LAN should reduce lat/lon mixing in the 5-way softmax (the programme's largest known defect). Directional, not a gate |
| **STRATEGIC** — route/goal-setting quality | §6.1 `route_sensitivity` + `lat_compliance`, **plus** `nonav_route_beats_majority` | the family that has never been measurable on this arm. §6.1 IS this family |

Where a family cannot be computed (no lead agent in frame, no route label), it is reported **per
family with the reason and the n** — never silently dropped.

---

## 7. The cheapest discriminating experiment — E0, and it comes FIRST

**`stack/scripts/lan_probe.py --navcf --ckpt <deployed REF-C> --cache <val cache>`**
Forward passes only. No training, no new labels, no new data. ~4 passes over 881 windows.

Sweep the EXISTING 4-way `nav_cmd` over the same windows on the **already-trained** REF-C checkpoint
and measure `max_pairwise_mean_m` — how far the decoded trajectory moves between commands.

| reading | verdict | consequence |
|---|---|---|
| **`max_pairwise_mean_m` ≈ 0** (≤ 1e-6) | **INERT** | The C6 confound becomes MEASURED rather than argued: the deployed model's route pathway does nothing. LAN is justified, and every "`nav_cmd=None` was fine" defence is closed |
| **materially > 0** | **RESPONSIVE** | The input IS consumed, and evaluating at `nav_cmd=None` was **discarding a live signal**. The cheap fix is then to **supply the label at eval and re-score every REF-C row** — which would be a bigger, cheaper correction than LAN, and LAN's marginal value drops accordingly |
| control > 1e-6 | **INSTRUMENT-FAIL** | decode is nondeterministic; fix before any number |

**Either way the programme learns and the next step is decided by the number.** This is the
experiment to run first because it can *cancel* the expensive one.

⚠️ **It is not run in this turn**: `tanitad-new` is mid data-pull, `tanitad-pod2` is faulty and
running v5f, `tanitad-pod4` is ⛔ observe-only. No GPU is free that I am permitted to use. The script
is delivered, tested against inert/responsive/nondeterministic stubs, and ready.

---

## 8. What would make me wrong (stated in advance)

* **The whole design rests on `nav_cmd` being degenerate.** If E0 returns RESPONSIVE, §1's chain is
  intact but the remedy changes — supply the label at eval, do not retrain.
* **S1 is validated on ONE scene of a DIFFERENT corpus.** Sub-metre agreement on Stockholm NuRec is
  not sub-metre agreement on PhysicalAI-AV. The claim is "the map-free construction is faithful
  where a map exists to check it", not "the parity corpus routes are correct".
* **15.6 % side disagreement** bounds the achievable topology signal.
* **The 2 s prior (§3) may itself be horizon-limited.** GOAL_INPUT measured the lateral axis on the
  2 s surface — which is the same instrument class the 2026-07-25 retraction proved blind at 18.5 s.
  I am *using* that limitation as the reason to read LAN at K=185, and it cuts both ways: a 2 s null
  is not evidence about 18.5 s, and neither is a 2 s ceiling.
* **`route_dropout = 0.5` may starve the seam.** If §6.1 returns REFUTE, re-run at
  `--route-dropout 0.0` before concluding the architecture ignores routes — that is a named,
  pre-registered follow-up, not a post-hoc rescue.

---

## 9. Deliverable manifest

| artifact | where it lives | what it is |
|---|---|---|
| `stack/tanitad/data/lan.py` | repo, staged | LAN encoding, leak guard, counterfactuals, `LaneCorridor` (S2) with the heading gate, `route_agreement`, `lan_window_features` trainer bridge |
| `stack/tanitad/eval/route_cf.py` | repo, staged | route-counterfactual instrument + `nav_cmd_sensitivity` (E0), with the three negative controls |
| `stack/tanitad/refs/refc.py` | repo, staged | `LanConfig`, `RefCConfig.graft_lan` / `route_dropout`, decoder `lan_to_cond` + `lan_gate` + `_lan_anchor_prior`, `RefCModel.lan_direction`, `param_breakdown["lan"]` |
| `stack/scripts/refc_train.py` | repo, staged | `lan_dataset_class`, `--graft-lan` / `--lan-arclengths` / `--lan-min-lead-m` / `--route-dropout`, `lan_valid_frac` in the step log, LAN provenance + coverage in `config.json`, **val set carries the same route input as train** |
| `stack/scripts/lan_probe.py` | repo, staged | `--agreement` (§4, runs on the dev box) and `--navcf` (§7, E0) |
| `stack/tests/test_lan.py` | repo, staged | 34 tests incl. the leak guard, speed-invariance, the three instrument negative controls, the gated-graft key set, the gradient block/unblock pair, the trainer smoke, and the §4 regression pin |
| `Project Steering/PREREG_lan_refc.md` | repo, staged | this document |
| `lan_agreement*.json` | scratchpad only (`…/scratchpad/lan_agreement*.json`) | raw §4 output; **regenerable in seconds** by the pinned test and by `lan_probe.py --agreement`, so it is not banked |

**Test status:** `pytest -q` in `stack/` — green, LAN adds 34 tests, no existing test changed.

**Not done, and named:** no training launched (forbidden by the brief and by the fleet state); E0
not run (no free GPU); the four-family readout in §6.4 is specified against existing instruments but
its LAN-arm numbers do not exist yet — that is the work E0 and §6 authorise, not work this document
claims.
