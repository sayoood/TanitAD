# REF-C v3 — the goal-mediated hierarchy on the supervised arm (4B-dominance proof track)

**Date:** 2026-08-18 · **Stream:** Architecture & Inference (REF-C v3 design agent)
**Status:** DESIGN + SKELETON (inert at default) — ⛔ nothing here launches while Thor trains.
**PI ask, verbatim intent:** *"a REF-C version with proven improvements, correctly designed around
our Multi-Hierarchy using our new tactical and strategic knowledge, implementing WITHOUT COMPROMISES
the conditioning of lower layers by strategic and tactical goals, usage of tactical capabilities to
improve trajectory selection, and make it very efficient. In parallel to our unsupervised-WM
efficiency claim, prove the dominance of the 4B architecture in an optimized REF-C model, ready to
train."*

**Companion documents:** `PREREG_REFC_V3.md` (the dominance experiment, both outcomes committed) ·
`code/refc_v3_launch_line.txt` (launch + preflight, NOT to be run now).
**Skeleton:** `stack/tanitad/refs/refc_v3.py` + `stack/tests/test_refc_v3.py` (+ one 12-line gated
hook in `refc.py`, default `None`, byte-identical off — §6.1).

⚠️ **Sibling packages: absent at design time, LANDED before finalisation, reconciled in §9bis.**
Both paths were probed absent at the start of this stream; both landed mid-stream
(`…/2026-08-18-refc-architecture-review/`, `…/2026-08-18-refc-improvement-review/`) and were read
before this document closed. The architecture review CORROBORATES this design (same measured
params, same seam inventory, and it independently recommends the C120 intervention probe "before
any external-tactical-brain wiring lands" — which is §6 here). The improvement review sharpens two
points and resolves one dependency — §9bis carries the reconciliation; §9's D-1/D-3 are updated in
place.

---

## 0. Honesty header — what "proven improvements" can and cannot mean today

The registry (`Project Steering/MODEL_REGISTRY.md` §4) shows **only three REF-C arms were ever
trained**: small (54,690,001), base (104,191,577), XL (251,932,584), all 30 k on the parity corpus.
The D-TAC1 / D-SEL / S6 presets **exist in code and were never trained**. So the honest partition is:

| class | levers | v3 treatment |
|---|---|---|
| **MEASURED-proven** (registry §4.2/§4.3, raw eval JSON) | encoder shrink to ~48 M costs no fan quality; anchor width ≥128 is the fan lever; base≈XL tie (Δ +0.0013, NOT separated); reach clamp exactly inert on ADE + 3.58× per-candidate compute; v2.1 labels improve the proposal set (oracle −0.058, flagship calibration) | **default-ON, in BOTH arms** of the dominance pair (they are sizing/label facts, not hierarchy levers) |
| **MEASURED-mechanism, retrain pending** | factored lat×lon head (the 5-way collapse provably destroys the longitudinal decision — algebra + confusion matrix in `refc_tactical.py`; Alpamayo measured **40.62 % dual-axis manoeuvres**, so the action space must represent simultaneous lat+lon — binding) | **default-ON in BOTH arms** (an action-space fact, not a hierarchy lever — keeping it out of the H−F delta is what makes the dominance read attributable) |
| **MEASURED-conditional** | goal-distance selection: on the banked REF-C-XL fan a goal point at σ=0.5 m beats the trained selector **−0.1591 [−0.2300, −0.0894] SEPARATED**, at σ=1.0 m it is **+0.0943 [+0.0241, +0.1650] WORSE** (`v6.py::GoalDistanceScorer` docstring, `sel_winners_curse_law.py`) | **ON in v3-H behind its measured admission gate (σ ≤ 0.8 m @ 2 s)** — a gate, not a hope (§5) |
| **BINDING, not yet measured here** | 6 s horizon (`PLAN_STEPS=60, DT=0.1`, operative (0,2], tactical (2,6]); goal/situation information-disjointness; four metric families; window-integrating tactical layer with PROVEN sensitivity (C115/C116) | designed in, each with its enforcement mechanism named (§4, §6, §7) |

Evidence classes below: **MEASURED** carries an artifact path; **BINDING** carries the ruling;
**ESTIMATED** is marked. No INHERITED number decides anything.

---

## 1. What v3 is, in one paragraph

REF-C v3 is the **supervised (imitation) arm carrying the full strategic/tactical/operative
hierarchy as an explicit goal cascade**: a window-integrating strategic state emits a **predicted
geometric route goal**; a window-integrating tactical state, conditioned on that strategic goal,
emits a **factored manoeuvre decision** and **predicted geometric tactical goals** at {2, 4, 6} s;
the operative anchored-diffusion decoder is conditioned on both (through REF-C's two *existing,
tested* external-tactical-brain ports), and the emitted trajectory is **selected** by distance to
the predicted tactical goal — the one selection mechanism our own measurements support
(candidate-independent reference; no winner's curse) — behind its measured admission gate. The
dominance experiment trains this arm (v3-H) against a config-identical flat arm (v3-F) whose only
delta IS the goal cascade, with "everything else identical" **derived from the two configs and
pinned by a test**, not asserted (C122's lesson).

Sizing is the registry's own recommendation (§4.3 "what this settles"): small-class encoder
(~48 M) + 128-anchor base-class decoder — **MEASURED at build (2026-08-18, `param_breakdown_v3`,
pinned by `tests/test_refc_v3.py`): v3-H 62,930,419 · v3-F 60,389,402 · delta +2,541,017
(+4.04 %)** — against a 300 M budget, ≥10 Hz by construction (small ticked 11.50 ms p50 fp32; base
decoder adds ~1.7 ms — MEASURED, `eff_refc-base-30k.json`).

---

## 2. Sizing — every choice cites the measurement that licenses it

| choice | value | measured basis (registry §4.2/§4.3 unless noted) |
|---|---|---|
| encoder | `base_width 64, blocks (3,6,16,6)` (small trunk, ~48 M) | small's fan is **at least as tight as base's at every matched K** (oracle-over-first-K, small ≤ base ≤ XL); the 2.4× encoder cut "did NOT degrade proposal quality"; registry: *"v4 can shrink the encoder toward small's ~48 M with no measured loss of fan quality, provided the anchor vocabulary stays wide (≥128)"* |
| anchors | **128** FPS (nested prefix of the existing 256 pool, seed 0) — rebuilt for 6 s (§3) | the selected-ADE knee is **anchor count, not encoder scale**: small@64 loses ~0.053 SEPARATED, base@128 ≈ XL@256 on ADE; anchors are buffers (~0.05 MB), not params |
| decoder | `d 384, 4 layers, 8 heads` (base geometry) | base ≈ XL on everything that ships (+0.0013 NOT separated); decoder is **~1.7 ms of base's 21.8 ms tick** (encoder 90.7 %) — decoder capacity is nearly free, so base geometry is kept rather than small's d256 |
| horizon | **(5,10,15,20,30,40,50,60) steps = 6.0 s @ 10 Hz**, 0.5 s stride to 2 s then 1 s stride | BINDING (`PLAN_STEPS=60, DT=0.1`; operative (0,2] / tactical (2,6]). Slot 20 (=2 s) is the operative/tactical seam and the selection slot (§5) |
| tactical head | **factored lat(3)×lon(3)**, shared trunk (+897 params MEASURED, `tests/test_refc_tactical.py`) | the 5-way priority collapse destroys the longitudinal decision (9.68 % of windows carry a live lon manoeuvre AND are labelled a turn — unrecoverable by any decode rule); Alpamayo 40.62 % dual-axis (BINDING) |
| labels | route **v2.1** (`--labels v21`) | the only end-to-end label measurement: proposal set **improved** (oracle −0.058) at ADE +0.025 not separated (flagship v1.5 calibration, registry §4.3 confound note) |
| hierarchy modules | `PhiTac` (**the existing, tested, trained implementation** — `tanitad/models/tactical.py:99`) + `StrategicCtx` (existing GRU) + 3 small heads + `GoalDistanceScorer` port | C116: *"a correct component exists, is tested, and is not called"* — v3 **wires the orphaned component instead of rebuilding it** (one-implementation rule; `refc_select.py` precedent) |
| param budget | **≤ 80 M hard; MEASURED: v3-H 62,930,419 / v3-F 60,389,402 (delta +4.04 %; hierarchy = PhiTac 1,708,288 + tac_latent_proj 262,656 + gstr_cond 66,816 + heads 9,429 + scorer 1,156; core delta 492,672 is the decoder's target-latent FiLM)** | "very efficient": 0.60× base with measured-equal fan quality; 4.8× under the Sub-300M headline; Thor batch-8 shape. Band pinned by `tests/test_refc_v3.py` |

**Efficiency claim shape** (parallel to the unsupervised-WM efficiency claim): *a ~65 M supervised
hierarchy at ≥10 Hz that matches or beats the 104–252 M flat arms* — resolvable by the pre-registered
experiment, not asserted in advance.

---

## 3. The horizon extension without breaking parity — the one place a naive design corrupts the corpus

**MEASURED from source** (`stack/tanitad/data/_contract.py:120`): the window enumerator computes
`t_max = ep.frames.shape[0] - window - max_horizon`. Raising `max_horizon` 20→60 therefore
**re-selects the window set** (drops every window with <6 s of remaining episode), which breaks the
sacred parity contract (canonical 406,099 windows on `physicalai-train-e438721ae894`) and cross-arm
comparability with every trained arm.

**Design rule (parity-preserving):** v3 keeps `max_horizon=20` for **window enumeration** — the
window set stays bit-identical to canonical — and fetches future steps 21..60 by **CLAMP + validity
mask** through a dataset wrapper (`V3GoalDataset`), exactly the E4.1 convention the tactical label
contract already uses (*"taus beyond the episode end are valid=False and CLAMP, never NaN"* —
`tactical.py` docstring). Consequences, stated rather than discovered:

* trajectory slots 30..60 carry a per-slot `traj_valid` mask; masked slots contribute **exactly
  zero** loss (zero-grad pinned by test);
* **band weighting is declared, not implicit** (the sibling review's H2 warning: an unweighted mean
  over a 6 s plan silently re-weights toward the far band by arc length): the slot layout is
  **band-balanced by construction** — 4 slots in (0, 2] and 4 slots in (2, 6] — and the loss is the
  per-valid-slot mean, so operative and tactical carry equal slot weight; any re-weighting is a
  pre-registered amendment;
* the nearest-anchor assignment for the CE runs on **valid slots only**;
* tactical goal targets at {2,4,6} s carry the `[K]` validity mask from `refb_labels.goal_tac_targets`;
* eval states per-family n where 6 s GT is short (four-families rule 5: *"say so per family with the
  reason and the n"*).

The anchor vocabulary is rebuilt for the 8-slot space by the existing `build_refc_anchors.py`
machinery (`synth_anchor_pool` already generalises over `horizons`); seed 0, pool 4096, and the
**2 s prefix of the new anchors is checked against the trained-arm anchors on the shared slots** so
the near-fan geometry stays comparable (report, not gate).

---

## 4. The hierarchy — every conditioning edge, named

### 4.1 The levels

```
frames [B, W=8, 9, 256, 256]
   │ ResNet encoder (small trunk)                      — encode ONCE per frame
   ▼
fmap (last frame) [B, F=512, 8, 8]  ·  pooled_seq [B, W, F]
   │                                        │
   │                          ┌─────────────┴──────────────┐
   │                          ▼                            ▼
   │                 StrategicCtx GRU (existing)      PhiTac causal TCN (existing,
   │                 z_str = ctx [B, 64]              tactical.py:99, wired at last)
   │                          │                       z_tac [B, 512]
   │                          ▼                            ▲ FiLM(e_g_str) ── detach ──┐
   │                 g_str = strat_goal_head(z_str)        │                           │
   │                 [B, 3] (cos, sin, dist_pref)  ────────┘                           │
   │                          │                            │                           │
   │                          │              ┌─────────────┼─────────────────┐         │
   │                          │              ▼             ▼                 ▼         │
   │                          │        lat/lon heads   tac_goal_head    tac_latent_proj│
   │                          │        [B,3],[B,3]     ĝ_tac [B,K=3,4]  [B, S=512]     │
   │                          │              │         @ {2,4,6} s           │         │
   │                          ▼              ▼ derive_man5   │ detach        ▼ detach  │
   │                    ctx token      maneuver_logits port  │         target_latent   │
   │                    (existing)     (existing H19 seam)   │         port (existing  │
   ▼                          ▼              ▼               │         FiLM seam)      │
AnchoredDiffusionDecoder(fmap, measurement, ctx, maneuver_logits, target_latent)       │
   │  → anchor_traj [B, N=128, 8, 2] · sel_score [B, N] · reach_keep                   │
   ▼                                                         │                         │
GOAL SELECTION (v3-owned re-rank):                           ▼                         │
   score' = sel_score + goal_gate · (−‖traj@2s − ĝ_tac(2s)_xy‖/τ) + cand_bias          │
   masked to reach_keep survivors · seam-clamped · zero-init gates                     │
   → sel_idx' → traj                                                                   │
                                                                                       │
(measurement = [v0, nav one-hot] — v0 NEVER reaches any goal head ──────────────────────┘
 or the tactical pool; pinned by the intervention audit, §6)
```

### 4.2 The edge list (the deliverable the ask names: "every conditioning edge named")

| # | edge | mechanism | grad policy | status |
|---|---|---|---|---|
| E1 | frames → encoder → {fmap, pooled_seq} | shared trunk (declared common ancestor) | trains by everything | existing |
| E2 | pooled_seq → z_str (StrategicCtx GRU) | window integration (strategic state) | trains by route aux + g_str loss | existing module |
| E3 | z_str → **g_str** (strategic goal head, Linear 64→3) | **predicted geometric** route goal: bearing (cos,sin) + signed along-track pref | trains by leak-guarded LAN label (TRAIN-ONLY; §4.4) | **new (+195 params)** |
| E4 | g_str → tactical pool conditioning (FiLM after `PhiTac`) | **strategic goal conditions tactical** | **detached downward** (v6 `_cut()` discipline) | **new** |
| E5 | pooled_seq → z_tac (**PhiTac**, causal TCN, W=8) | window integration (tactical state) — C115/C116: sensitivity **proven by freeze-history gate**, never asserted | trains by tactical losses only (detached into decoder) | **existing module, first wiring** |
| E6 | z_tac → lat(3) + lon(3) heads → `derive_man5_logprobs` → **maneuver_logits port** | **tactical decision conditions anchor priors** (existing H19 graft, prior-centered) | live grads within tactical branch; port value enters decoder graph | existing port, new supplier |
| E7 | z_tac → `tac_latent_proj` → **target_latent port** | **tactical state conditions the operative decode** (existing zero-init FiLM seam — *"only activates when a real tactical brain feeds a target_latent"*; v3 IS that brain) | **detached** (decoder cannot train the pool — attribution) | existing port, first supplier |
| E8 | z_tac → **ĝ_tac** (tactical goal head → [K=3, 4] = (x, y, heading, speed) @ {2,4,6} s) | **predicted geometric tactical goals** — the E4.1 layout, verbatim | trains by hindsight-goal regression, masked | **new (~+26 k params)** |
| E9 | ĝ_tac(2 s).xy → **goal selection re-rank** over the emitted fan | **tactical capability improves trajectory selection** — `GoalDistanceScorer` rule: `−‖endpoint − ĝ‖/τ + b`, zero-init gate, seam-clamped, reach-masked | goal **detached** into selection (selection trains gates/τ/bias only — the winner's-curse firewall, §5) | **new (+~270 params)** |
| E10 | v0 → measurement encoder → decoder condition | ego speed, existing convention (C122: `ego_v0` is a legitimate driving input) | — | existing |
| E11 | **v0 ↛ {z_tac, g_str, ĝ_tac}** — a REFUSED edge | goal path stays vision-pure (matches v6's `GoalDistanceScorer` admissibility declaration and the C120 audit finding *"every goal head is a function of frames alone"*) | n/a | **pinned by intervention audit** (§6) |
| E12 | LAN corridor ↛ inference (label-only) | supplied route is optimistic by construction on PhysicalAI (BINDING) — corridor is the **training label** for E3, never a model input (`refc_goal_config` precedent: the trainer mints `lan` without building the input pathway) | n/a | pinned by test (goal terms bit-unchanged when `lan` withheld at eval — `test_refc_select.py` pattern) |

**Why detach on E4/E7/E9 (grad policy):** the v6 stack routes every downward goal port through
`_cut()` (= detach), and the live v6F run trains with `--uplink stopgrad`. Detaching keeps each
level trained by **its own supervision** — attribution survives, and the selection objective cannot
corrupt the goal head (the frozen-predictor discipline `cons_detach` already applies on S3). The
cost — C120 — is that **gradient probes are structurally blind to these forward paths**, which is
exactly why the intervention audit is wired in as a gate (§6), not a doc note. A non-detached
variant is a registered ablation lever, not the default.

### 4.3 Tactical supervision targets the band, not the seam (C89b)

**MEASURED** (`RETRACTION_LOG.md:4746`, 201 clips, episode-cluster bootstrap, production
thresholds): the tactical band's true κ is **LON 0.1428 [0.0540, 0.2250] / LAT 0.1777 [0.0658,
0.2953]** on `TAC_BAND_S` (2.0, 6.0] — against seam values 0.3270/0.3132 on (0, 2]. The tactical
layer's labels therefore live IN the band:

* ĝ_tac taus {2, 4, 6} s — 2 s is the seam/handoff slot (it is also the selection slot, §5); 4 s
  and 6 s are strictly in-band;
* the factored manoeuvre label stays at the 2 s horizon **for the H19 prior** (that prior gates the
  near fan, which is the operative surface) — and the **band-manoeuvre label over (2, 6]** is
  carried as a *reported* second target on the tactical trunk (aux CE, weight 0 in v3.0 default —
  registered lever, so turning it on is one flag, not a redesign).

### 4.4 The strategic goal — provenance, verbatim rules

* **Predicted, never supplied** (BINDING; goal point is the literature lever: categorical command
  +0.2 PDMS vs goal point +4.7 — `goal_provenance()` docstring carries the citations).
* Label = `tanitad.data.lan.lan_window_features` (ego future path, arc-length resampled,
  **leak-guarded**) — the sanctioned direction of *"LABELS MAY USE EGO; INFERENCE IS VISION-ONLY"*.
* The **dist_pref confound is inherited and declared** (`refc.py::goal_targets`): the first
  admissible arc-length depends on ego speed through the leak guard, so the along-track half is
  partly a K7-unrecoverable quantity. The prereg predicts the bearing half works and the
  along-track half may not — **the gates are separate so that prediction is measured** (same
  sequencing rule as S6).

---

## 5. Selection — "usage of tactical capabilities to improve trajectory selection", without re-buying a refuted lever

### 5.1 The conflict the brief requires surfacing, surfaced

The registry says, in bold, *"Selection is no longer the productive lever on REF-C"* (§4.1: the
oracle gap is ~92 % irreducible; v1.2's learned re-scorer NOT separated across 47 arms). The PI ask
says selection improvement is mandatory. **These do not actually collide, and the design says
precisely why:** every refuted selection mechanism was **candidate-conditioned** (a scorer reading
the fan: v1.0 hand-cost, v1.2 re-scorer, SEL-1's roll-consistency argmin — the last REFUSED for a
measured **winner's curse**: error-rank RISES with N, 0.241→0.286; lower-tail hit collapses
0.57→0.28). The goal-distance rule is **candidate-INDEPENDENT** — its reference cannot be gamed by
the fan, its error-rank **falls** with N (0.006→0.001 at σ=0), lower-tail hit **1.00** — and on the
banked REF-C-XL fan it **beats the trained selector when the goal is good enough** (σ=0.5:
−0.1591 SEPARATED) and **loses when it is not** (σ=1.0: +0.0943 SEPARATED). All MEASURED
(`v6.py::GoalDistanceScorer`, 881 windows / 40 eps, paired episode-cluster bootstrap).

⇒ The design is not "selection is a lever again"; it is **"selection is exactly as good as the
predicted goal, and the requirement curve is measured"**. That gives a clean, non-compromising
resolution: **the selection gate opens only if the trained goal head clears the measured admission —
σ ≤ 0.8 m 1-sigma endpoint accuracy at 2 s on val** — *"a gate, not a hope"*. If the head misses
admission, v3-H ships with selection gate at 0 (bit-identical ranking to v3-F) and the dominance
read still stands on the conditioning edges; the goal-σ result is then the finding. **Both branches
are pre-registered** (PREREG §5).

⚠️ **The measured adverse prior, stated with its numbers (improvement review §5.2, E-WC2, fired
2026-08-16):** a ridge on **frozen REF-C pooled latents** predicts the 2 s goal at
**σ = 4.7104 m [3.8087, 5.6860] — σ/ADE 9.99, REFUSED at 2.48× the refusal threshold**; a
**0-parameter constant-yaw-rate kinematic extrapolation reaches σ = 1.1888 m** (3.96× better than
that ridge, still above the 0.8 m bar). v3's goal head differs from the refused surface in exactly
the two ways the refusal names — it is **trained end-to-end** (not a ridge on frozen features) and
it reads **z_tac** (window-integrating) rather than the last-frame pool — but the honest
pre-registered expectation is that **Branch B (not admitted) is the likelier outcome**, and the
prereg now carries (a) the **kinematic-extrapolation goal as a mandatory comparator** — a trained
head that cannot beat a 0-param extrapolation has not earned the selection story — and (b) the
**E-WC shape read** (error-rank vs N) on the v3 goal score as an admission control (the sibling's
S-B rule: a new rank graft is shape-tested before it costs an arm).

### 5.2 The selection stack (all v3-owned, post-decoder; decoder untouched)

1. **Reach clamp ON** (`sel_reach_clamp`, accel_max 2.5) — MEASURED exactly inert on ADE (paired
   delta 0.0) and deletes 72.08 % of the fan, 3.58× cheaper per-candidate work; the decoder exports
   `reach_keep` and the re-rank masks to survivors. A precondition, not a win — pitched as such.
2. **Goal-distance term** at the **2 s slot** (`anchor_traj[:, :, IDX_2S]` vs ĝ_tac(2 s).xy):
   `score' = sel_score + goal_gate · (−d/τ) + cand_bias`, `goal_gate` zero-init (bit-identical at
   step 0), τ learned, goal **detached**. The 2 s slot is chosen because the admission curve was
   measured at 2 s endpoints; **the 6 s-endpoint variant is REPORTED but may not select until its
   own requirement curve is measured** (the E-WC machinery generalises; registered follow-up).
3. **Seam clamp** (`apply_seam_clamp`, clamp 1.0) on the goal graft vs the base score — the
   actuator for telemetry REF-C already emits, with the population-over-time fail-loud (C51: a
   batch-max fail-loud killed two healthy arms).
4. **Training**: the blended score enters a CE **normalised over exactly the survivor set the
   argmax ranks over** (the S1c lesson — a full-fan softmax on a ~27 % problem is dominated by
   candidates no selector picks).
5. **Controls carried with every selection number** (from the scorer's own doctrine): the
   **goal-echo control** (corpus-marginal ĝ — measured far-null at 7.82 m vs 0.79 m live) and the
   **capacity control** (`MLPCandidateScorer` shape — pre-registered conditional arm, §PREREG 6.3,
   so a win cannot be read as (mechanism) when it is (capacity) — the C6 confound).

### 5.3 What is refused, with the measurement that refuses it

* **L1 re-ranker / learned fan re-scorer** — SEL-1 REFUSED (winner's curse, E-WC2 σ/ADE 9.9915);
  v1.2 NOT separated. Not re-proposed (brief: do NOT re-propose it).
* **Roll-consistency argmin (cons_score alone)** — +5.9787 [+5.3217, +6.7625] WORSE, MEASURED.
  `graft_cons` stays available as a zero-init *additive prior* only (S3 semantics), OFF in v3.0.
* **MPC / CEM over a rolled model** — C101 (T1, CI-separated): the CEM planner is **35.8 % worse
  than CV** and loses on the family its own cost targets; REF-C additionally has **no rollable
  predictor** (`law_head` consumes a trajectory, emits a pooled vector — cannot iterate;
  `refc_select.py` argues this from source). Deferred to the improvement-review package; not in v3.0.
* **Target-speed term in the selection score** — REFUTED in registry §4.1 (0.0 % recovered;
  GT-perfect speed-matcher WORSE than baseline). The speed capability lives in the tactical goal's
  speed component and the longitudinal eval instead (§7).

---

## 6. The disjointness audit — wired in, not a doc note

### 6.1 The mechanism

`stack/tanitad/eval/goal_provenance.py` (C120 instrument) is the **intervention** probe —
detach-transparent, distinguishes DIRECT_PATH from COMMON_ANCESTOR — precisely what the detach
policy (§4.2) makes mandatory, because `assert_isolation`-style gradient probes are structurally
blind to detached forward paths (C120, demonstrated on a deliberately wired arm).

Wiring (in the skeleton, not future work):

* `RefCV3Model.provenance_roles()` returns the role map — GOAL_ROLE on {g_str, ĝ_tac, goal
  embedding}; SITUATION_OUTPUT_ROLE: **none in graph** (REF-C neither imports, loads, nor receives
  the situation classifier — `RefCModel.goal_provenance()` already declares this; v3 re-declares and
  the audit **measures** it rather than trusting the declaration).
* `stack/tests/test_refc_v3.py` runs the audit at smoke size with **both controls**: a live
  positive control (intervening on `frames` MUST move every goal node — else UNPOWERED, never
  DISJOINT) and the **pinned negative edge E11** (intervening on `v0` must move NO goal node).
  An all-clean report whose positive control is also clean is refused (C109 lesson, built into
  `audit_arm`).
* The launch preflight (§8) runs the same audit at full config on the dev box **before** any GPU
  day, and writes the verdict into the run's `config.json` next to `goal_provenance()`.

### 6.2 The generalised leak test (the ruling's own check), applied to every new head

*"For ANY head, ask whether its inputs at inference include something the label was derived from."*

| head | label derived from | inference inputs | verdict |
|---|---|---|---|
| g_str | LAN corridor (ego future path, leak-guarded) | z_str ← frames only | clean (label-may-use-ego direction); dist_pref speed-confound **declared** (§4.4) |
| ĝ_tac | hindsight ego pose at t+τ | z_tac ← frames only | clean; this is the E4.1 contract verbatim |
| lat/lon | ego kinematics over the label horizon | z_tac ← frames only | clean — **and note the contrast:** the sitclf anomaly came from feeding ego AT INFERENCE to a head whose label is an ego function; v3's tactical head is exactly the `head_img`-shaped deployable arm |
| selection gates | fan error (CE) | detached ĝ_tac + fan | goal cannot be trained by the fan (detach) — the honesty hazard bound (goal stays 2 numbers vs a 60×2 plan: an information bottleneck by construction) |

### 6.3 The C115 gate — hierarchy must be proven, not asserted

Before ANY dominance read counts, the H arm must pass the **freeze-history sensitivity gate**
(skeleton: `refc_v3.freeze_history_report`):

* replacing every non-last window frame with frozen copies of the last MUST move z_tac and ĝ_tac
  (a bit-identical result = the C115 defect = the arm is flat-in-disguise and the experiment is
  VOID, not "H lost");
* the gradient probe ∂/∂frame_t is the corroborating second mechanism (C116 pattern), run through a
  **random projection**, never `.sum()` — the C116 instrument hazard (LayerNorm makes `.sum()`
  identically zero-grad) is pinned as a negative control in the tests.

---

## 7. Longitudinal — designed around the lead data, eval-first

**The facts:** 88.7 % of the oracle gap is longitudinal; C122 measured the flagship
indistinguishable-from-GT on distance-keeping while both frozen-DINO arms are CI-separated in the
**unsafe** direction (REF-A min-TTC −5.82 s). The lead data is first-class: `obstacle.offline` join,
`taniteval/taniteval/lead_source.py`, banked `val40_lead_block.npz` attaching row-for-row (881=881,
speed corr 1.0 — C122).

**v3.0 design:**

1. **Supervision** (train-side, no new data dependency): (a) the tactical goal's **speed
   component** at {2,4,6} s — longitudinal goal-setting supervised directly, in-band (C89b); (b) the
   **factored lon head** — the decision no longer destroyed by the collapse; (c) the refc1
   target-speed class head stays available (gated, measured pattern) but OFF in v3.0 — one
   longitudinal decision surface (the goal) rather than two half-trained ones.
2. **Eval** (the binding four families, per family, never pooled — PREREG §4): LONGITUDINAL =
   target-speed accuracy + **distance-keeping (headway / time-gap / TTC) via the val40 lead block**;
   strata **edge-free LEAD vs NO_LEAD** as primary (C121: the 3-band splits can be made to pass or
   fail by band choice — reported always, gating never).
3. **Deferred, dependency named:** lead-conditioned *training* losses (e.g. gap-aware weighting)
   require the obstacle join **on the train corpus**; only the val40 block is verified banked.
   Absence-discipline: this is a one-location observation — the prereg lists the two probes to run
   (pod-side join outputs; `build_obstacle_join.py` provenance) before the v3.1 rider is specced.
   Not in v3.0; **not silently dropped either** — it is PREREG §8's registered rider.

---

## 8. Trainer, launch line, and preflight

* **Trainer:** `stack/scripts/refc_v3_train.py` — **BUILT AND SMOKE-PROVEN** (2026-08-18, dev box
  CPU): a thin composition over `refc_train.py`'s machinery (its dataset classes and loss-weight
  constants IMPORTED, never copied), adding the parity-preserving `V3Dataset` (§3),
  `compute_losses_v3` (masked 8-slot trajectory/assignment, factored CE on both decision surfaces,
  masked goal regression, survivor-set selection CE), `--arm hier|flat` with the config delta
  REFUSED at build if off-register, `--preflight`, and the done-marker discipline
  (`summary.json {"done": true}` written by the trainer itself). MEASURED green: smoke preflight
  PASS (hier + flat), full-size preflight PASS (62.9 M build, C115 gate, E11 audit, finite loss
  step), 3-step smoke train both arms, strict resume 3→5, `--goal-str` label path.
* **Launch line:** `code/refc_v3_launch_line.txt` — pod A40 shape, both arms + seeds, à la
  `st_launch_line_fixed.txt`. ⛔ NOT to be run: Thor is training; pods carry the usual traps
  (PYTHONPATH, quota-by-dd, stale-checkout grep-verify — the preflight encodes them).
* **Preflight (in the launch file, executable checks):** (1) suite green (`pytest -q`, `$?` before
  any pipe — exit codes are not evidence); (2) the intervention audit at full config (§6.1);
  (3) freeze-history gate (§6.3); (4) config-delta pin (PREREG §3); (5) parity: window count equals
  canonical, skip-hash `f09e44db`, §10c ingest gate untouched; (6) anchors rebuilt + 2 s-prefix
  report; (7) pod: dd write test, grep-verify shipped fix markers, OMP_NUM_THREADS=6.

---

## 9. Open dependencies and PI decisions (each with a default)

| # | item | default if nothing said |
|---|---|---|
| D-1 | **RESOLVED against the landed improvement review:** comfort-as-COST is refuted program-wide (C46 composite rewards not-driving; C101 comfort-costed CEM lost to CV), comfort-as-**PARAMETERISATION** is proven (v1.6: free-Δ jerk RMS 52.13 → unicycle-integrated 1.13, separated). ⇒ the comfort axis is the (a,κ) question, which is D-6 — no jerk *loss* enters v3.0 | no comfort cost term; the axis rides D-6 |
| D-2 | 6 s selection slot (endpoint) requirement curve | 2 s slot selects (measured admission); 6 s reported-only until its σ* is measured on the v3 fan |
| D-3 | **RESOLVED: the train-side obstacle join EXISTS** — 2,308 episodes / 12,122,129 agent boxes (`…/2026-08-17-train-obstacle-join/TRAIN_OBSTACLE_JOIN.md`, cited by the improvement review §4.2) | v3.0 still trains without lead-conditioned losses (one decision surface at a time); the v3.1 lead rider (sibling's L2b) is **UNBLOCKED, spec next** |
| D-4 | Non-detached uplink variant (E7 grad policy) | detached (v6 discipline); the variant is a registered ablation lever |
| D-5 | Where the runs go (pods vs post-S-T Thor) | pod A40 ×1, sequential arms (PREREG §7 cost table); Thor untouched |
| D-6 | ⚠️ **THE ONE REAL CONFLICT WITH THE LANDED REVIEW — the plan parameterisation.** The improvement review's PRIMARY for the horizon is the **(a,κ) bounded-control re-parameterisation** (J1+H1: unicycle-integrated 60-step controls — fixes comfort + the 72.08 % unflyable fan + the horizon in one; convergent with Alpamayo-2's `UnicycleAccelCurvatureActionSpace` and v6 §4b; v1.6-measured mechanism). This design chose the review's OWN NAMED FALLBACK — the sparse-slot free-XY extension — for two stated reasons: (i) the dominance experiment's lever is the HIERARCHY, and the fallback keeps the fan comparable to every trained arm at the shared 2 s slots; (ii) stacking an untrained action-space rewrite under the hierarchy pair risks the whole 6-run ladder failing for action-space reasons. Both options costed: **fold (a,κ) into v3.0** = decoder offset-head rewrite + anchor rebuild in control space + new prereg riders (~2–4 days work), one retrain answers three binding axes, but the pair loses fan-level comparability to the 30 k arms and carries the untested-mechanism risk in both arms; **sequence as v3.1** = v3.0 launches as skeleton'd (tested today), (a,κ) lands as the registered shared-surface upgrade after the dominance verdict, per the review's own priority table (J1+H1 is its #5, behind the 0-GPU items). ⛔ *"Without compromises" cuts toward folding it in; risk isolation cuts toward sequencing — this is exactly the class the brief says to surface, not decide silently* | **sequence as v3.1** (the registered pair stays as built); PI may override to fold-in before launch |
| D-7 | `sel_ce_objective = softade` (E-OBJ-1: longitudinal recovery −0.0974/−0.1670 m separated on frozen weights) for v3's selection CE | **CE in v3.0** (both arms identical; the one-hot-CE pathology E-S1-0 measured is noted). The in-training softade question is OWNED by the sibling's S-A ladder on incumbent REF-C — running it twice in parallel duplicates an experiment; a v3 amendment follows S-A's verdict |

---

## 9bis. Reconciliation with the two sibling packages (landed mid-stream, read before closing)

**Convergent — no change needed:** factored lat×lon default (their A1 = §2 here) · 6 s horizon
with band labels (their H3 = §4.3) · lead-block eval-first (their L2a = §7.2) · edge-free strata
(C121, both) · winner's-curse / SEL-1 / v1.2 / hand-cost / top-m refusals (their §5.2 = §5.3 here)
· MPC sequenced behind the selection verdict (their M1 dependency = §5.3's refusal) · the C120
intervention gate before external-brain wiring (their architecture review §5 = §6 here) · the
architecture review's param table reproduces the numbers this design sized against.

**Sharpened by the review, folded in:** the E-WC2 adverse prior + kinematic comparator + shape-read
admission control (§5.1 note, PREREG §5/§6) · band-balanced loss weighting declared (§3) · the
train-side obstacle join exists → D-3 resolved, v3.1 lead rider unblocked · the
`driving.py:606-608` distance-keeping refusal must be retired before the four-family eval (0-GPU
instrument prerequisite, now PREREG §4).

**Genuinely conflicting, surfaced as D-6:** (a,κ) control-space parameterisation (their primary)
vs sparse-slot free-XY (their named fallback; this design's choice) — costed both ways, default
sequence-as-v3.1, PI may override. **And D-7:** softade selection objective — owned by their S-A
ladder; v3 follows its verdict rather than duplicating the experiment.

**One caution imported verbatim:** REF-C currently has **no T1 adapter** (their §6.1) — the
dominance experiment's tier framing in PREREG §4 (open-loop metric tier, no capability claims)
is therefore not just discipline but the only available read; the T1 adapter is the sibling's M1
prerequisite and stays out of v3.0 scope.

---

## 10. Deliverable manifest (this package)

| artifact | where | state |
|---|---|---|
| this design | `…/2026-08-18-refc-v3-design/REFC_V3_DESIGN.md` | staged |
| pre-registration | `…/PREREG_REFC_V3.md` | staged |
| model skeleton | `stack/tanitad/refs/refc_v3.py` | staged; inert at default (nothing imports it in default builds) |
| forward hook (shared code) | `stack/tanitad/refs/refc.py` (one gated arg, default `None`) | staged; default path byte-identity pinned by existing `test_refc.py` + new test |
| tests | `stack/tests/test_refc_v3.py` | staged; run green on dev box (see manifest in final report for the measured counts) |
| launch line + preflight | `…/code/refc_v3_launch_line.txt` | staged; ⛔ not run |
