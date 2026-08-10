# DIFFUSION PLANNER COMPARISON — REF-C's AnchoredDiffusionDecoder vs the v5f operative head

**Date:** 2026-08-10 · **Author:** analysis agent (read-only pass over `stack/` + `taniteval/`)
**Trigger:** PI observation — v5f's fan/anchors/trajectory hypotheses look JUMPY and far less
rich than REF-C's diffusion planner; the highlighted trajectories do not read as road-driving
hypotheses, whereas REF-C's fan looks smoother and better. This document explains WHY, from
code, and ranks the causes with their cheapest discriminating experiments.

**Tier discipline:** every number below is a **T0 diagnostic** on the fixed 881-window / 40-episode
val grid unless stamped otherwise. Point estimates on the corpus grid; the decision-grade interval
for any registry claim is the episode-cluster bootstrap (`taniteval/ci.py`). Per the 2026-08-02
binding rule, the four-families numbers for v5f live in
`v5f_four_families_30k.json` (this directory); this doc quotes the accel/feasibility slice because
smoothness is the question under analysis, not as a substitute for the four families.

---

## §1 Measured anchor — the roughness gap, separated from drawing resolution

Same 881 windows, same feasibility rule (**infeasible step: |a| > 4 m/s² ∨ |yaw_rate| >
0.33·v + 0.05**), matched 4-waypoint 0.5 s resolution where stated.
Evidence class: **MEASURED** — artifacts: `taniteval/results/fan_refc-xl-30k.pt` census (local) +
pod5 `x0_fan_dump.npz` census (`tools_x0_lite.py`, this directory); dense-fan row banked in
`MODEL_REGISTRY.md` §1.13 (`x0_lite_f32.json`, this directory). The matched-resolution census
numbers were supplied by the orchestrating brief citing those artifacts and are **INHERITED into
this document** (not re-run by this agent); the §1.13 and w4 rows were verified against the
registry and the JSONs in this directory.

| quantity | REF-C-XL (native 4 wp @ 0.5 s) | v5f subsampled to the SAME 4 wp | v5f native dense (20 wp @ 0.1 s, registry §1.13) |
|---|---|---|---|
| fan accel MAE (all candidates) | **6.39 m/s²** | **17.00 m/s² (2.7×)** | **252 m/s²** (mean |a|, `x0_lite_f32.json`) |
| selected-candidate accel MAE | **0.366 m/s²** | **0.627 m/s² (1.7×)** | **8.10 m/s²** (cross-validates §1.8's 8.11) |
| candidate-infeasible fraction | 71.7 % | 89.0 % | 97.6 % of all 256×20×881 steps; 100 % of candidates |
| selected-infeasible fraction | 4.5 % | 7.3 % | (selected accel MAE 8.10 tells the story) |
| selected ADE @2s | 0.4714 m (registry §1.5/§4, full-set) | — | 0.4011 m (registry §1.8) |
| oracle-in-fan | 0.1640 m | — | 0.1975 m (registry §1.8) |

**Three readings, before any hypothesis:**

1. **Most of v5f's roughness lives BETWEEN the 0.5 s waypoints** — at frequencies REF-C's 4-point
   head cannot even express: dense 252 → matched-resolution 17.0 is a ~15× drop from subsampling
   alone. A 4-segment polyline is band-limited by construction; the drawing and the census agree.
2. **At matched resolution v5f is still genuinely rougher — 2.7× on the fan, 1.7× on the pick.**
   The visual gap is therefore *mostly* resolution-of-expression, but not entirely.
3. **In BOTH planners the fan tails are far worse than the pick** (REF-C 6.39 vs 0.366; v5f 17.0
   vs 0.627). The selector avoids the worst candidates in both; what the PI sees as "not
   road-driving hypotheses" is dominated by the unsupervised bulk of the fan (§3, H2).
   Note v5f's *selected ADE is actually better* than REF-C-XL's (0.4011 vs 0.4714) — the gap under
   discussion is smoothness/feasibility, not endpoint accuracy.

---

## §2 Side-by-side architecture (file:line per row)

Both heads run the SAME decoder algorithm — v5f's `V15Decoder` literally subclasses REF-C's
`AnchoredDiffusionDecoder` (`stack/tanitad/models/flagship_v15.py:210-217`) and overrides only the
KV source and the scoring fix. Every difference below is therefore a *configuration or wiring*
difference, not a different algorithm.

| axis | REF-C-XL (`refc-xl-30k`) | v5f (`flagship-v5f-w120-30k`) |
|---|---|---|
| **decoder class** | `AnchoredDiffusionDecoder` (`stack/tanitad/refs/refc.py:1039`) | `V15Decoder(AnchoredDiffusionDecoder)` (`flagship_v15.py:210`), instantiated by `FlagshipV4Head(FlagshipV15Head)` (`stack/tanitad/models/flagship_v4.py:180`) |
| **output resolution** | **4 time waypoints** at steps (5,10,15,20) = 0.5/1/1.5/2 s (`refc.py:292`, `TrajectoryConfig.horizons`) | **20 dense waypoints** at steps 1..20 = 0.1 s ticks (`flagship_v4.py:72` `DENSE_HORIZONS`, `:91`) |
| **anchor vocabulary** | 256 FPS anchors over REAL ego-frame waypoint targets, [256, 4, 2] (`refc.py:667-688` XL preset; builder `stack/scripts/build_refc_anchors.py:37-79` — FPS over `refb_labels.waypoint_targets`, data pool ~200 k, NOT the synthetic default; RETRACTION_LOG.md:3106) | 256 FPS anchors over the SAME real targets at DENSE horizons 1..20, [256, 20, 2] (`build_refc_anchors.py` at `DENSE_HORIZONS`; provenance `Project Steering/LOOP_STATE.md:1015,1018`; loaded `train_flagship_v4.py:1368-1371` from `--anchors-dense flagship_v4_anchors_dense.pt`, run manifest `stack/ops/runs.d/flagship-v5f-w120-30k.env`). **Both vocabularies are FPS-from-data — real human trajectories, smooth by construction.** The dense vocabulary is NOT an upsampled 4-pt set. |
| **offset head** | `nn.Linear(d, n_steps*2)` → **8 free numbers**/anchor (`refc.py:1079`) | the SAME `nn.Linear(d, n_steps*2)` → **40 free numbers**/anchor — one independent (x, y) per 0.1 s step, **no kinematic coupling between steps** (inherited; `refc.py:1079`) |
| **decoder size** | d=512, 6 cross-attn layers (`refc.py:682`) | d=384, 4 layers (`flagship_v4.py:93-95`) — deliberately v4-sized, not v1.5's d512×8L |
| **denoise** | 2-step truncated denoise, noise_std 0.1 m per waypoint train-time, deterministic at eval (`refc.py:309-310, 1386-1400`) | identical: 2 steps, noise_std 0.1 (`flagship_v4.py:95`; loop `flagship_v15.py:265-271`). ⚠️ Same **0.1 m absolute** noise per waypoint — but at 0.1 s spacing that is ~the entire inter-waypoint displacement at city speed, vs ~10 % of it at 0.5 s spacing (§3 H1b). |
| **KV source (what the anchors cross-attend)** | one 8×8 conv feature map from the ResNet trunk → 64 spatial tokens (`refc.py:1307`) | heterogeneous token set: W·16 = 128 frozen/co-trained readout cells + 32 imagination tokens (8 probes × 4 read steps) (`flagship_v15.py:332-351, 393-411`; probe path `flagship_v15.py:747-804`) |
| **conditioning** | measurement (speed, ego-dropout 0.5) + strategic-ctx GRU token + H19 manoeuvre prior on conf (`refc.py:1275-1384`) | speed (ego-dropout 0.5, **learned null row** P5b `flagship_v15.py:361-363, 437-444`) + VTARGET & ROUTE goal tokens through ReZero gates under goal-dropout 0.5 (`flagship_v15.py:365-378, 453-481`) |
| **scoring / selection** | conf of the **t=0 classifier pass over the RAW anchors**; the denoise passes' confidences are **discarded**; `sel_idx = argmax(conf)`; the emitted geometry is the 2-step-refined one (`refc.py:1364, 1393, 1437` with `sel.refined=False` default; code-fact summary `taniteval/taniteval/plan_fan.py:29-44`) | **refined_logits kept from the last denoise pass** (`flagship_v15.py:270` — the deliberate "scoring fix") + factorised LAT×LON×DIST zero-init grafts, norm-clamped at 1.0× (`flagship_v4.py:214-218, 221-290`) + gated longitudinal VTARGET term (`flagship_v15.py:575-588`); reach clamp **OFF** on v4/v5f (`flagship_v4.py:154`, deliberate — unmeasured on v4's own fan) |
| **trunk state during head training** | **co-trained, but BY the planner's own losses**: one Adam over `model.parameters()` (`stack/scripts/refc_train.py:891`); the trajectory/cls CE are the trunk's primary objective (+ LAW aux). ⚠️ NOT frozen — the frozen-trunk arm of this lineage is flagship **v1.5** (`flagship_v15.py:1-9`). | **co-trained under a FOREIGN objective**: from-scratch joint run — random-init trunk, WM loss stack live from step 0, planner gradient into the trunk scaled by λ_plan (Phase A λ=0 to 2 k, ramp to 8 k, joint after; `train_flagship_v4.py:94-114, 1319-1329, 1878`; `v4_curriculum.py:49-85`). The head trains for 30 k steps on features being continuously reshaped by the WM losses. |
| **geometry losses** | anchor-cls CE + **L1 on the ONE GT-nearest anchor's refined trajectory** (winner-takes-all at 4-pt/0.5 s resolution); **no smoothness, jerk, curvature or kinematic term anywhere** (`refc_train.py:419-423, 602-606`) | same WTA CE + L1 at 20-pt/0.1 s + refined-rank CE (`flagship_v15.py:844-888`), **kinematic weights all default 0.0** (`flagship_v15.py:838-841`, parity-frozen) + `plan_smoothness_loss` (jerk 0.02 + curvature-rate 0.01) — but **on `wp_seq`, the SELECTED plan only** (`train_flagship_v4.py:138-141`; `v4_curriculum.py:107-134`). **255/256 candidates per window receive no geometry gradient in either model.** |
| **selection supervision** | `loss_cls` CE supervises the t=0 conf — the same object the argmax ranks (supervised-at-ranked-object ✓) | `loss_rcls` CE supervises `sel_score` — also the object ranked (✓, `flagship_v15.py:882-885`); plus factorised CE on the lat/lon/dist heads (`train_flagship_v4.py:127-136`) |

---

## §3 Ranked root-cause hypotheses for the roughness gap

Ranked by evidence strength. Each carries its cheapest discriminating experiment. H1 is
effectively confirmed; H2–H4 are supported but not isolated; H5 is plausible and cheap to kill.

### H1 — Per-waypoint-independent position offsets at 10 Hz: the 1/dt² amplifier (STATUS: essentially CONFIRMED by W4)

The offset head emits one independent (x, y) per step (`refc.py:1079`). For a per-waypoint
position error ε, discrete acceleration is a second difference divided by dt²
(std ≈ ε·√6/dt²): the SAME ε costs **1.0 m/s² at dt = 0.5 s and 24.5 m/s² at dt = 0.1 s — a 25×
amplification for identical waypoint-level fit quality**. Nothing in either loss penalises this
(position L1 is ~flat in it), so v5f pays it and REF-C structurally cannot.

* Supporting measurements: dense 252 vs matched 17.0 (≈15× from resolution alone, §1); registry
  §1.13's W2b — a free 3-tap smoother improves **both** selected ADE (0.4011→0.3975) and oracle
  (0.1975→0.1879), which "noise around signal" predicts and signal content does not; the jitter
  survives f16→f32 (storage-precision ruled out, §1.13).
* **The confirming experiment already ran:** W4 re-parameterised emission as bounded unicycle
  controls (a, κ) integrated to waypoints — same frozen trunk+head, 109 k new params, 4 k steps —
  and the fan went to **violations 0.0, selected accel MAE 0.774, oracle 0.1077 (nearly halved)**
  (`w4_gate.json`, this directory; `train_v58f_unicycle_head.py:10-15, 141-182`). A
  parameterisation change alone removed the roughness AND improved coverage — the jitter was
  hiding coverage, not providing it.
* H1b (sub-cause, same family): train-time denoise noise is 0.1 m **absolute** per waypoint
  (`flagship_v4.py:95`). At 0.5 s spacing that is ~10 % of a city-speed step displacement; at
  0.1 s spacing it is ~100 % of it. The offset head is therefore trained to make per-step
  corrections of the same order as the step itself — a much rougher learned offset field at 10 Hz
  for the same config constant. *Cheapest discriminator:* retrain the head (frozen trunk) with
  noise_std scaled ∝ dt (0.02) and census the fan; if fan accel MAE drops materially, the constant
  was mis-scaled for the dense grid.

### H2 — No geometry loss on 255/256 of the fan; and the 4-pt WTA is an implicit smoothness prior REF-C gets for free

In both models, only the GT-nearest anchor's trajectory receives L1 (`refc_train.py:420-423`;
`flagship_v15.py:876-879`); v5f's smoothness term touches only the SELECTED plan
(`train_flagship_v4.py:138-141`). The fan tails are shaped solely by gradients leaking through the
shared offset head — which is why tails ≫ pick in both censuses (6.39/0.366 and 17.0/0.627). But
REF-C's tail roughness is *capped* by its parameterisation: a 4-point polyline at 0.5 s spacing
cannot represent high-frequency jitter at all, so its unsupervised tails stay visually plausible.
v5f's 20-point tails can and do express it. This explains most of the remaining 2.7× at matched
resolution being fan-dominated (selected only 1.7×).

* *Cheapest discriminator:* one head-only retrain arm (frozen trunk) adding
  `plan_smoothness_loss` over the FULL fan (or over the S2-reachable subset for cost) — if fan
  accel MAE at matched resolution converges toward REF-C's, unsupervised-tails is the driver; the
  W2b post-hoc smoother (sel accel 8.10→3.09, free) is the zero-training lower bound already in
  hand.

### H3 — Moving features under a foreign objective (the co-training hypothesis)

REF-C's trunk is co-trained but its gradients COME FROM the planner losses — features and head
co-adapt toward the same objective (`refc_train.py:891`, loss assembly `:602-606`). v5f's trunk is
random-init and dominated by the WM stack from step 0; the planner head chases features that move
under someone else's loss for 30 k steps, with λ_plan only ramping in at 2 k–8 k
(`train_flagship_v4.py:1319-1329`; `v4_curriculum.py:53-54`). A head that never sees stable
features can hedge with high-frequency offsets; a conf head trained on moving features ranks
worse.

* Consistent with: W4/W4b — on the FROZEN 30 k trunk, a 109 k-param emission head reached
  accel 0.774 in 4 k steps (`w4_gate.json`), i.e. the frozen features were sufficient for a smooth
  fan; the deficiency was in what was trained on top of them while they moved.
* ⚠️ Confounded as evidence: W4 changed parameterisation AND froze the trunk simultaneously.
  *Cheapest discriminator:* the missing arm is a **freeze-trunk, offset-head-only retrain at the
  ORIGINAL waypoint parameterisation** (same 4 k-step recipe as W4). If the original-parameterised
  head also lands near-feasible, moving features dominate; if it stays ~89 % infeasible at matched
  resolution, parameterisation (H1) dominates. This cleanly separates H1 from H3 for one
  frozen-trunk GPU-evening.

### H4 — Vocabulary jitter vs offset jitter (STATUS: vocabulary is very likely innocent — verify with a 5-minute census)

Both vocabularies are FPS over real human trajectories (`build_refc_anchors.py:37-79`;
LOOP_STATE.md:1015). `furthest_point_sample` returns `pool[chosen]` (`refc.py:193-214`) — every
anchor is bitwise a real human window, so the RAW dense vocabulary should be near-feasible by
construction. The registry's §1.13 reading ("truncated-denoise residue AROUND the true path")
assumes this but the raw-vocabulary census was never run.

* *Cheapest discriminator (CPU, minutes):* run the |a|/|yr| census on
  `flagship_v4_anchors_dense.pt` **alone** (no model). Near-zero violations ⇒ all fan roughness is
  offset+denoise-added and H4 dies; material violations ⇒ FPS's preference for extreme pool
  members selected sensor-noise-corrupted targets and the vocabulary needs re-building with a
  smoothness filter.

### H5 — Conditioning/token-set differences (weakest; rank last)

v5f's KV is 160 abstract tokens (128 readout cells + 32 imagination tokens) vs REF-C's 64
conv-map tokens; v5f's decoder is smaller (d384×4L vs XL's d512×6L); goal tokens and factorised
grafts differ. Any of these could contribute to the residual matched-resolution gap, but no
measurement separates them, and the imagination tokens are known to be candidate-invariant
(`flagship_v15.py:654-694`) so they cannot be *ranking* noise. Do not spend a GPU-day here before
H1–H3 are separated.

---

## §4 What v5.8f already fixes — and what remains

**Fixed (W4, MEASURED, both pre-registered gates PASS — `w4_gate.json`, registry §1.13):**
the fan itself. `UnicycleEmission` (`train_v58f_unicycle_head.py:141-182`) emits per-candidate
bounded controls a = 4·tanh, κ = 0.2·tanh integrated by a unicycle rollout — **feasible by
construction** (yaw_rate ≤ 0.2·v sits inside the census band for every v ≥ 0):

| | W4 unicycle fan | original v5f fan (same grid) |
|---|---|---|
| oracle ADE | **0.1077 m** | 0.1991 m |
| selected-candidate accel MAE | **0.774 m/s²** (winner 0.261; violation frac **0.0**) | 9.297 m/s² |
| selected ADE (frozen selector) | 0.7933 m | 0.4056 m |

Trunk provably untouched (md5 identical before/after, `w4_gate.json.trunk_frozen_proof`). This is
the PI's "road-driving hypotheses" property, delivered: every candidate is now a drivable plan.

**Remaining: SELECTION.** The frozen selector's scores were learned against the old fan's
geometry; on the clean fan its pick is near-uninformed (0.7933). The W4b recalibration
(feat variant) **FAILED its G1 gate held-out: selected ADE 0.5600 vs ≤ 0.45**
(`w4b_gate_feat.json`; train monitor 0.21–0.33 vs held-out 0.56 — it memorises train-window
selection), and the top-8 pruner role is not viable either (top-8 oracle 0.3185 vs ≤ 0.15). Per
the prereg's bound consequence, **W7 (WM-roll re-rank on the clean fan) is now the primary
selection mechanism**; the kin variant (adds (a, κ) inputs) is in flight. Note W1's refutation of
waypoint-space kinematic costs (−16.7 %) does **not** transfer to the unicycle fan — there the
controls ARE the kinematics, which is what makes `v58f.kinematic_cost`
(`stack/tanitad/models/v58f.py:100-121`) admissible in the G2 shortlist rule.

---

## §5 What REF-C still does better, and what is worth porting

1. **Supervised-at-the-ranked-object scoring on stable features.** REF-C's conf is trained
   (`loss_cls`) on exactly the distribution it ranks, and its features are shaped by the same
   objective. The E-S1-0 dose-response (`refc_train.py:759-765`) is the sharpest statement in the
   program of how much this matters: the supervised t=0 conf selects at **0.4728** while the SAME
   weights' unsupervised refined readout selects at **1.3100** — a 2.8× penalty purely for scoring
   off-distribution. v5.8f's W4b failure is the same lesson (a rescorer trained on features never
   supervised for the new fan's geometry does not generalise). **Port: whatever scores the W4 fan
   must be trained against that fan's own geometry with the CE normalised over the candidates it
   actually ranks** (REF-C's S1c `sel_ce_reach` idea, `refc.py:487-499`) — or be a param-free
   physical cost (W7).
2. **Coarse WTA resolution as a free smoothness prior.** REF-C's 4-pt/0.5 s WTA cannot reward
   sub-0.5 s structure, so none is learned. v5.8f keeps dense emission (needed for jerk/curvature
   terms and for control) but gets the same prior a better way — a low-dimensional control
   parameterisation. If any future head returns to waypoint-space emission, a **coarse-to-fine
   scheme (supervise 0.5 s knots, interpolate or lightly refine between)** is the REF-C property
   worth keeping. The 25× 1/dt² arithmetic in §3 H1 says dense position-space WTA at 10 Hz should
   never again be trained without either a control parameterisation or a fan-wide kinematic term.
3. **The reachability/prefilter discipline on the candidate set.** REF-C's S2/S2b band deletes
   72.08 % of its fan for a paired ΔADE of exactly 0.0000 and a 3.5× per-candidate compute saving
   (`refc.py:546-555, 1320-1362`; `flagship_v15.py:544-566`). v4/v5f deliberately shipped with the
   clamp OFF because the zero-change property was unmeasured on its own fan
   (`flagship_v4.py:138-154`) — correct discipline then; on the W4 unicycle fan the band is
   trivially satisfiable from the controls, and the S2b pre-decode form is the cheap admission
   ticket for W7's per-candidate WM rolls (`imagine_candidates` already takes a `keep` mask,
   `flagship_v15.py:698-744`).
4. **Fan-level honesty in the visualisation.** `plan_fan.py` renders exactly what the decoder
   computes, draws its 4 waypoints as explicitly-labelled straight segments ("a drawing device
   only, no curvature is invented", `plan_fan.py:67-71`), asserts `sel_idx == argmax` per batch,
   and prints the oracle-in-fan gap in the HUD. Two consequences for the v5f/v5.8f videos:
   (a) the comparison the PI made was **structurally unequal** — REF-C's drawn fan is band-limited
   to 4 segments while v5f's 20-point polylines faithfully display 10 Hz jitter REF-C could not
   have shown; (b) the fix is NOT to smooth the drawing. Render the v5.8f fan at native density
   (it is now feasible by construction, so it can afford honesty) and add a **matched-resolution
   0.5 s-subsampled pane** whenever a REF-C fan is shown alongside, so smoothness comparisons are
   at equal bandwidth. A re-render of the OLD v5f fan at 4-waypoint subsampling is also the
   cheapest confirmation of §1's split: if it looks near-REF-C-smooth on screen, drawing
   resolution dominated the visual impression (the census says it should — 17.0 vs 6.39 is
   visible but not the "jumpy" 252).

**Honest uncertainty.** The 2.7×-at-matched-resolution residual is NOT yet attributed: H2
(unsupervised tails at an expressive resolution), H3 (moving features) and H1b (noise_std
mis-scaling) all predict it, and only the §3 experiments separate them. The claim this document
does make firmly, because W4 measured it: the dominant, PI-visible roughness — 97.6 % infeasible
steps, 252 m/s² fan accel — is **parameterisation, not vocabulary, not the trunk, and not the
diffusion algorithm**, which is shared line-for-line between the two planners.

---

### Provenance / evidence classes

- MEASURED (artifacts in this directory): `w4_gate.json`, `w4b_gate_feat.json`,
  `x0_lite_f32.json`, `v5f_four_families_30k.json`, `p7_calibration.json`.
- MEASURED (registry-banked): `MODEL_REGISTRY.md` §1.8 (v5f sel 0.4011 / oracle 0.1975),
  §1.13 (W1/W2/W2b/W4/W4b), §1.5/§4 (REF-C-XL 0.4714 full-set / 0.1640 oracle).
- INHERITED (from the orchestrating brief, citing `fan_refc-xl-30k.pt` + pod5
  `x0_fan_dump.npz` censuses; not re-run by this agent): the matched-resolution accel MAE /
  infeasibility table in §1 (6.39 / 17.00 / 0.366 / 0.627 / 71.7 % / 89.0 % / 4.5 % / 7.3 %).
- Architecture claims: file:line cited inline throughout; all paths relative to the repo root
  (`stack/…`, `taniteval/…`).
- ⚠️ One correction to the framing this analysis was briefed with: REF-C's trunk was **not
  frozen** during head training (`refc_train.py:891` — one Adam over `model.parameters()`); it is
  co-trained under the planner's own losses. The frozen-trunk member of this lineage is flagship
  v1.5. §3 H3 is stated accordingly.

---

## H4 CLOSED — MEASURED 2026-08-10 ~19:05Z (the doc's named 5-minute check, run same-day)

Census on the RAW `anchors_dense` vocabulary alone (256×20×2 @10 Hz, from the fanfull dump's
`anchors.npz`), identical rule: **accel MAE 1.97 m/s²**, step-infeasible **10.6 %**,
candidate-infeasible 35.2 %. Against the refined fan's 252 m/s² / 97.6 %: the vocabulary is
data-plausible and largely innocent — **the roughness is manufactured by the per-step offset +
truncated-denoise path (H1), not the anchor build.** H1's standing is upgraded from "ranked
first" to "confirmed by elimination on this axis too" (W4's reparameterisation already
confirmed it constructively). Evidence class MEASURED; tool = the W2 census inline, pod5.
