# LAN / REF-C — E0 EXECUTED. The pre-registered falsifier FIRES, and so does the fallback.

**Date:** 2026-08-03 (Europe/Berlin) · **Pre-registration:** `Project Steering/PREREG_lan_refc.md`
**Run directory (quote this, not a number):**
`tanitad-thor:/home/nvidia/lan_e0/` → banked in this directory.
**Estimator, every interval below:** paired episode-cluster bootstrap,
`taniteval/taniteval/ci.py::paired_episode_cluster_bootstrap`, `n_boot = 2000`, unit = **episode**.
⛔ `overlapping_holdout_se` is never called.

---

## 0. The one paragraph

**E0 ran, and it refutes the premise LAN was built on.** REF-C's 4-way `nav_cmd` is **not** inert:
sweeping it moves the decoded trajectory by **0.2416 m** (label-reachable commands) with a
bit-identical-input control at **exactly 0.0**. §7 pre-committed that a RESPONSIVE reading makes the
cheap fix *"supply the label at eval and re-score every REF-C row"* — **so I ran that too, and it is
refuted as well**: feeding REF-C the **oracle** route changes ADE@2s by **+0.0024 m
[−0.0107, +0.0147], not separated**, while making cross-track **+0.0031 [+0.0001, +0.0063]** and
curvature **+0.0013 [+0.0003, +0.0024]** *separably WORSE*. Feeding the model's own produced route is
separably worse on ADE (**+0.0118 [+0.0011, +0.0227]**), along-track, speed and cross-track.
⇒ **The route pathway is live but anti-compliant.** The remaining live hypothesis is the one §6.1
assigned to its REFUTE branch, promoted from LAN to `nav_cmd`: **the architecture, not the label, is
the defect.**

**No training was launched. No GPU-days were spent. Total compute: forward passes on Thor.**

---

## 1. What ran, and on what

| item | value | class |
|---|---|---|
| checkpoint | `tanitad-thor:/home/nvidia/models/refc-base/ckpt.pt`, REF-C-base, **104.192 M params**, `sd_missing 0 / sd_unexpected 0` | **MEASURED** |
| val cache | `tanitad-thor:/home/nvidia/valdata/physicalai-val-0c5f7dac3b11/` | **MEASURED** |
| windows | **859** windows / **39** episodes, window 8, **stride 8**, WP_STEPS (5,10,15,20) | **MEASURED** |
| decode | anchored diffusion, `steps = 2` (`cfg.decoder.diffusion_steps`), `model.eval()` | **MEASURED** |
| geometry | encoder `image_size 256`; cache frames **256×256**; token grid **8×8**; `matches_training_raster: true` | **MEASURED** |

### ⚠️ Two coverage facts, stated up front rather than in a footnote

1. **39 of 40 val episodes, not 40.** `ep_00028.pt` on Thor is a **truncated transfer**
   (92,299,264 B vs 117,383,256 B for its siblings) and does not load
   (`PytorchStreamReader … failed finding central directory`). The probe now **records** the
   omission in every result JSON (`coverage` block) instead of dying or silently skipping.
   ⇒ **This run is 859 windows, NOT the 881 the published REF-C rows sit on, and is therefore not
   window-for-window comparable to them.** The episode-cluster bootstrap is unaffected in kind — its
   unit is the episode and it resamples the 39 present. **Re-pulling that one file is a work item.**
2. **`refc-xl` was not run.** MEASURED 2026-08-02 (`tanitad-thor:~/refc_thor_eval.json`) it dies on
   the 256×640 cylindrical cache with `shape '[8, 512, 8, 8]' is invalid for input of size 491520`
   (= 8·512·8·**15**, an 8×15 grid). That is the wrong-raster defect retracted 2026-08-02. The
   geometry assertion added here **refuses before any forward pass** rather than reporting.

---

## 2. ⛔ NEGATIVE CONTROLS FIRST — all four PASS

Per the brief, no metric below is quoted until its controls have fired.

| # | control | result | verdict |
|---|---|---|---|
| **C-identical** | decode the *same* input twice; must be bit-identical | `control_max_disp_m = **0.0**` (tol 1e-6) | ✅ **PASS** — decode is deterministic, so any non-zero sweep number is signal |
| **Geometry** | checkpoint raster vs cache raster | 256×256 vs 256×256, grid 8×8 | ✅ **PASS** |
| **CV sanity** | the trivial predictor must be present and beatable | CV `ade_0_2s` **0.8089** [0.5923, 1.0445] vs REF-C **0.4644** | ✅ **PASS** — surface behaves |
| **Registry agreement** | does this harness land where the published row does? | this run **0.4644** (859 win / 39 ep, `follow_constant`) vs `MODEL_REGISTRY.md` `refc-base-30k` **0.4728** [0.3835, 0.5699] (881 win / 40 ep) | ✅ **PASS** — Δ 0.0084, deep inside the published interval, on one fewer episode |
| **Component-vs-family** | do the per-window components fed to the bootstrap reduce to the family means beside them? | 5 EXACT metrics at **0.0** relative error; 2 GATED metrics within a stated estimator gap; **0 failed of 28** | ✅ **PASS** |

### ⭐ The fourth control caught a real bug in my own instrument, before any number was quoted

The first version of the per-window reducer back-filled windows where **every** step was gated out by
the `MIN_DS` speed gate with their **unmasked** row mean — importing exactly the crawling/stopped
windows the gate exists to remove. MEASURED: `heading_mae` read **4.0181°** against
`four_families`' **1.1486°**, and `curvature_mae` **42.1365 1/m** against **0.00788** — a **5,347×**
inflation. Those windows are now **dropped**, with their `eid` dropped alongside so the bootstrap
clusters stay aligned.

⚠️ **This is not cosmetic: the buggy version reported `LATERAL_heading_mae_deg` for the oracle arm as
+0.2241 [+0.0178, +0.5856] SEPARATED. The corrected value is +0.0185 [−0.0080, +0.0458], NOT
separated.** A separated lateral degradation would have been reported that does not exist.

The residual GATED gap (heading 0.83 %, curvature 7.9–19.7 %) is **not** a bug and is not tuned away:
`four_families` pools over (window, step) while a cluster bootstrap must reduce to one value per
window, and the two agree only when every window contributes equally many valid steps — curvature has
just **2** pair-steps on a 4-waypoint view. Both are correct; the **paired** deltas use the
per-window components for *both* arms, so the contrast is unaffected.

---

## 3. E0 (§7) — **RESPONSIVE**. The pre-registered "INERT" reading is REFUTED

`stack/scripts/lan_probe.py --navcf` · artifact `E0_refc-base_navcf_full.json`

| pair | mean L2 shift (m) |
|---|---:|
| follow vs left (`0v1`) | **0.24159** |
| follow vs right (`0v2`) | **0.226289** |
| left vs right (`1v2`) | **0.092954** |
| **`max_pairwise_mean_m`, label-reachable commands** | **0.24159** |
| *(all four commands, incl. the unreachable index 3)* | *1.768415* |

**Verdict: `RESPONSIVE — nav_cmd moves the trajectory`.**

### ⚠️ A correction the prereg did not anticipate: one of the four commands is unreachable

`refc_eval.ROUTE_TO_NAV = {0: 1, 1: 0, 2: 2}` maps the 3-class route label onto nav indices
**{0, 1, 2} only**. Nav index **3** — whose name happens to be `"straight"` — is an embedding row no
label and no eval mode can ever emit. MEASURED, it dominates the raw sweep (`0v3` **1.5126 m** vs
`0v1` **0.2416 m**) and its constant-command arm scores ADE **1.5712** vs follow's **0.4644**. That is
an **out-of-distribution probe, not a route counterfactual**. Reporting the bare 4-way max would have
overstated route sensitivity **7.3×** with an artefact. The instrument now reports both, and the
verdict is taken from the reachable set.

**Either way the verdict is the same and the reachable number is the honest one: 0.2416 m ≫ 1e-6.**

⇒ §7's RESPONSIVE consequence, quoted verbatim from the pre-registration:
> *"The input IS consumed, and evaluating at `nav_cmd=None` was **discarding a live signal**. The
> cheap fix is then to **supply the label at eval and re-score every REF-C row** … and LAN's marginal
> value drops accordingly."*

**The C6 confound is now MEASURED in the opposite direction from the one that was argued.** It was not
that the route pathway does nothing; it is that every published REF-C number held a live input at a
constant.

---

## 4. E0-B — I ran the fix §7 committed to, and **it is refuted too**

Costs **zero** extra forward passes: `nav_cmd` enters only through the `measurement` MLP, so the four
constant-command decodes already contain every trajectory a per-window route assignment could pick.
The arms are *assembled* from them, which also makes them bit-comparable.

* **`follow_constant`** — the historical C6 path (nav 0 for every window). The reference.
* **`produced_route`** — the model's own image-only `route_head` argmax. **Deployable**: no future,
  no label. Feeds not-follow on **32.71 %** of windows.
* **`oracle_route`** — the GT v2.1 route label. **Future-derived; an UPPER BOUND only, never a
  leaderboard number.** Feeds not-follow on **38.77 %** of windows.

### Paired Δ vs `follow_constant` — positive = WORSE (these are error metrics)

| metric | **`oracle_route`** Δ [CI95] | sep | **`produced_route`** Δ [CI95] | sep |
|---|---|:--:|---|:--:|
| **ADE@2s** | **+0.0024** [−0.0107, +0.0147] | ❌ | **+0.0118** [+0.0011, +0.0227] | ✅ **worse** |
| ADE@2s, route-valid windows only (n=727) | +0.0028 [−0.0128, +0.0173] | ❌ | +0.0109 [−0.0008, +0.0233] | ❌ |
| LONGITUDINAL `along_mae_m` | +0.0028 [−0.0098, +0.0150] | ❌ | **+0.0123** [+0.0012, +0.0238] | ✅ **worse** |
| LONGITUDINAL `speed_mae_mps` | +0.0031 [−0.0119, +0.0171] | ❌ | **+0.0136** [+0.0013, +0.0270] | ✅ **worse** |
| LATERAL `cross_mae_m` | **+0.0031** [+0.0001, +0.0063] | ✅ **worse** | **+0.0028** [+0.0003, +0.0053] | ✅ **worse** |
| LATERAL `heading_mae_deg` (n=820) | +0.0185 [−0.0080, +0.0458] | ❌ | +0.0243 [−0.0021, +0.0510] | ❌ |
| LATERAL `curvature_mae_1pm` (n=817/818) | **+0.0013** [+0.0003, +0.0024] | ✅ **worse** | +0.0008 [−0.0000, +0.0017] | ❌ |

**Absolute ADE@2s:** `follow_constant` **0.4644** [0.3767, 0.5560] · `oracle_route` **0.4668**
[0.3823, 0.5534] · `produced_route` **0.4761** [0.3862, 0.5689] · CV **0.8089**.

### Reading, with the pre-registration's own guardrails applied

1. **The ADE@2s null for the oracle arm is EXACTLY what §6.3 pre-committed.** §6.3 predicted
   `|ΔADE@2s| ≤ ~0.01 m` and **not separated**; measured **+0.0024, not separated**. Per the
   pre-registration this *"cannot later be spun as either success or failure"*. It is not evidence
   against a route signal — it is the registered null landing where it was told to.
2. **§6.3's no-harm bound is NOT breached.** The stopping rule is a separated worsening
   **> +0.02 m**; the worst measured is `produced_route` **+0.0118**. Below the bar, but a
   *separated degradation* nonetheless.
3. **The RED FLAG did not fire** — no arm improves ADE@2s by anything, let alone the >0.05 m that
   would have indicated a leak. (Not applicable here: no LAN corridor was fed.)
4. ⛔ **The result that is NOT a registered null is the lateral one.** §3 and §6.2 make LATERAL the
   family a route signal *must* pay in. Under an **ORACLE** route — the most favourable condition
   obtainable — cross-track and curvature are **separably worse**. A model that degrades laterally
   when handed the correct route is **destabilised by it, not following it**. That is §6.1's
   **PARTIAL** category, and the pre-registration is explicit: *"Report as a negative; do not spin as
   'responsive'."*

---

## 5. ⛔ THE FOUR METRIC FAMILIES — deployed arm (`follow_constant`), 859 windows / 39 episodes

Per family, never pooled. Every CI is the episode-cluster bootstrap, n_boot 2000, unit = episode.

### LONGITUDINAL

| metric | value | CI95 (episode-cluster, 39 ep) |
|---|---:|---|
| `speed_mae_mps` | **0.4388** | **[0.3519, 0.5330]** |
| `speed_bias_mps` | +0.0163 | — |
| `along_mae_m` | **0.4099** | **[0.3270, 0.4985]** |
| `along_bias_m` | +0.0224 (+ = ahead of the human) | — |
| `accel_mae_mps2` | 0.4915 | — |
| **distance-keeping (headway / time-gap / TTC)** | ⛔ **UNAVAILABLE**, n = 0 | — |

⛔ **The distance-keeping half is UNAVAILABLE and that is a WORK ITEM, not a pass.** Reason: no
lead-agent track was supplied to `four_families.longitudinal(lead=…)`. The reader exists
(`…/incoming/2026-08-03-longitudinal-distance-keeping/build_lead_tracks.py`, `obstacle.offline`,
97.44 % corpus coverage) and was **not wired into this run** — naming it is the honest state.

⚠️ Rates are on the **0.5 s** grid derived from `wp_steps` (5,10,15,20), carried in `_grid.dt_s`.
`along_*` are dt-invariant; `speed_*` and `accel_*` are not.

### LATERAL

| metric | family mean (pooled over steps) | per-window mean | CI95 (episode-cluster) | n |
|---|---:|---:|---|---:|
| `heading_mae_deg` | **1.1486** | 1.1581 | **[0.8421, 1.4976]** | 820 win |
| `yaw_rate_mae_degps` | **1.8589** | — | — | 3259 steps |
| `curvature_mae_1pm` | **0.00788** | 0.0085 | **[0.0053, 0.0123]** | 818 win |
| `curvature_bias_1pm` | −0.001024 | — | — | 2438 steps |
| `cross_mae_m` | **0.1276** | 0.1276 | **[0.0973, 0.1624]** | 859 win |
| `cross_bias_m` | +0.0014 | — | — | — |
| `cross_final_mae_m` | 0.2964 | — | — | — |

⚠️ Two columns on purpose for heading and curvature: the CI is on the **per-window** reduction (the
unit a cluster bootstrap requires); the family table pools over (window, step). §2 explains the gap
and bounds it. `cross_mae_m` shows the two coinciding exactly, which is the expected behaviour when
every window contributes equally many steps.

### TACTICAL — ⭐ the defect the brief reported in closed loop, REPRODUCED open-loop at n = 859

| class | n true | recall | **n predicted** |
|---|---:|---:|---:|
| `lane_keep` | 510 | 0.9745 | **675** |
| `turn_left` | 110 | 0.8182 | 106 |
| `turn_right` | 68 | 0.8529 | 71 |
| **`accelerate`** | **93** | **0.0000** | **0** |
| **`brake_stop`** | **78** | **0.0385** | **7** |

Accuracy **0.7544** [0.6849, 0.8235]. `never_predicted: ["accelerate"]`.

**171 of 859 windows (19.9 %) carry a longitudinal manoeuvre label; the head emits 7 of them (0.8 %)
and zero `accelerate`.** The brief's closed-loop Thor observation (`accelerate 0 / brake_stop 0` while
42 % of logged windows were `brake_stop`) is confirmed on an **independent surface** — real held-out
open-loop windows, a different corpus split, a different harness.

⛔ **And it is INVARIANT to the route input.** All four nav arms produce a **bit-identical**
manoeuvre histogram, because `maneuver_head` and `route_head` both read `pooled` (image only) while
`nav_cmd` enters solely through `measurement` → decoder (`stack/tanitad/refs/refc.py`, `forward`).
**MEASURED, and it has a hard consequence for LAN (§7 below).**

### STRATEGIC

| class | n true | recall | n predicted |
|---|---:|---:|---:|
| `route_left` | 212 | 0.4151 | 161 |
| `route_straight` | 394 | 0.9264 | **578** |
| `route_right` | 121 | 0.4298 | 120 |
| `UNKNOWN` (no future to derive a route) | 132 | 0.0000 | 0 |

Route accuracy **0.5879** [0.4837, 0.6946] over all 859; **0.6946** [0.5846, 0.7973] over the 727
route-valid windows. Majority-straight rate on the valid subset = 394/727 = **0.542**.

⚠️ `nonav_route_beats_majority` remains **VOID BY CONSTRUCTION** (GATE_PROTOCOL §0.7) and is not
quoted as a model verdict. What *is* quotable: the head predicts `straight` on **578** windows where
the label says **394** — a **+46.7 %** over-emission of the majority class, with left/right recall
at ~0.42–0.43.

---

## 6. §6.0 E-pre — the LAUNCH GATE: **PASS**

`stack/scripts/lan_probe.py --epre` · artifact `Epre_lan_coverage_val39.json` · **0 GPU**

| quantity | value |
|---|---:|
| LAN `any_valid_frac` | **0.8801** |
| per-anchor valid frac (20/40/80/160 m) | [0.3015, 0.5867, 0.5867, 0.3015] |
| **`nav_cmd` valid frac, RE-MEASURED ON THE SAME 859 WINDOWS** | **0.2724** |
| route-v2.1 valid frac, same windows | 0.8463 |
| **ratio LAN / nav_cmd** | **3.231×** |
| gate threshold | 0.50 |
| ego speed m/s (median / mean / p90) | 9.958 / 12.457 / 28.011 |

⭐ The bar is no longer INHERITED. The historical *"`nav_valid_frac` is 0.21–0.25 in all four arms"*
(RETRACTION_LOG 2026-07-21) is re-measured here **on the same windows as the LAN number**, at
**0.2724** — close to, and slightly above, the quoted band. A coverage comparison across two
different window sets is not a comparison; this one is.

⚠️ Measured on the **val** split (the parity **train** cache is not on any host I may use). Same
corpus and same geometry, different split. Labelled, not hidden.

---

## 7. ⛔ Two structural findings that change what LAN can and cannot fix

### 7.1 §6.2's PRIMARY OUTCOME READOUT IS NOT COMPUTABLE ON REF-C

§6.2 nominates `corridor_departure_rate @ 1.75 m` at **K = 185 (18.5 s)** as the primary outcome.

**MEASURED on Thor:** REF-C's decoder emits `traj` of shape **[B, 4, 2]** — four waypoints at
`cfg.trajectory.horizons = (5, 10, 15, 20)`, i.e. a **2.0 s** trajectory. `anchor_traj` is
[B, 128, **4**, 2]. There is no 18.5 s open-loop path to measure corridor departure on.

⇒ **§6.2 cannot be executed against REF-C open-loop as written.** This is a defect in the
pre-registration, not a failure to run it. E1a's K=185 result was obtained on an arm with a
**rollout** decoder; REF-C has an anchored one-shot decoder. Reaching 18.5 s on REF-C requires
closed-loop re-feeding — a different experiment, with the reconstruction-OOD confound (RETRACTION_LOG
C6) attached.

### 7.2 LAN AS SPECIFIED CANNOT REACH THE TACTICAL OR STRATEGIC HEAD

Both `maneuver_head(pooled)` and `route_head(pooled)` are computed from the image embedding **before**
and **independently of** the route path; `lan_emb` / `lan_dir` are passed **only** into
`self.decoder(...)` (`stack/tanitad/refs/refc.py`, `RefCModel.forward`). The nav arms prove the
consequence empirically: **identical head histograms across all four commands.**

⇒ **LAN, as §2 specifies it, is a decoder-side conditioning signal.** It can move the trajectory. It
**cannot** change the 5-way manoeuvre softmax that the programme calls its single largest known
defect, and it cannot change the route head. Any claim that LAN addresses the lat/lon mixing defect
would be wrong by construction.

---

## 8. §4 S1-vs-S2 agreement — REPRODUCED and now BANKED (was scratchpad-only)

`--agreement` on the banked NuRec artifacts, dev box, 0 GPU.

| | **gated 60° + 1 m hysteresis** | **ungated control** | ratio |
|---|---:|---:|---:|
| `pos_l2_m` median | **1.1862** | 4.6848 | 3.9× |
| mean | **1.4589** | 19.8009 | **13.6×** |
| p90 | **2.7322** | 57.4787 | **21.0×** |
| `lat_delta_m` median | 0.7112 | — | |
| `bearing_deg` median | 0.4143 | — | |
| `side_agree` mean | **0.8436** | 0.7915 | |
| route lanes / hops / on-graph | 19 / 18 / 12 (0.667) | 30 / 29 / 16 (0.552) | |

n = **275** comparable samples of 298. **Every figure reproduces the prereg §4 table**, which had
existed only in a scratchpad. The pinned ≥5× gate requirement holds at 13.6× (mean).

⚠️ Limits unchanged and restated: **n = 1 scene**, a **different corpus** from the parity set, and
**15.6 % of anchors disagree on which side** — that residual bounds how clean a lane-graph-supplied
lateral topology signal can be.

---

## 9. Verdict against the pre-registration, clause by clause

| clause | pre-registered outcome | **measured** |
|---|---|---|
| **§6.0 E-pre** | launch gate, `any_valid_frac ≥ 0.50` | ✅ **PASS** 0.8801 (3.231× the nav bar re-measured at 0.2724) |
| **§7 E0** | INERT ⇒ LAN justified · RESPONSIVE ⇒ supply the label instead | ⭐ **RESPONSIVE** (0.2416 m, control 0.0). **The INERT prediction is REFUTED.** |
| **§7 follow-up** | *"supply the label at eval and re-score"* | ⛔ **ALSO REFUTED.** Oracle route: ADE not separated; cross-track & curvature separably WORSE. Produced route: separably worse on ADE, along, speed, cross. |
| **§6.3 ADE@2s null** | \|Δ\| ≤ ~0.01 m, not separated | ✅ **as predicted** (oracle +0.0024, not separated) |
| **§6.3 no-harm bound** | reject if separated worse by > +0.02 m | ✅ not breached (worst +0.0118) |
| **§6.3 red flag** | separated improvement > 0.05 m ⇒ audit the guard | ✅ did not fire |
| **§6.1 H-LAN-1** | needs a LAN-grafted checkpoint | **NOT RUN** — no LAN arm exists; no training launched |
| **§6.2 H-LAN-2** | corridor departure @ K=185 | ⛔ **NOT COMPUTABLE on REF-C** — the decoder emits 2.0 s (MEASURED) |
| **§6.4 four families** | per family, with CIs | ✅ reported; **distance-keeping UNAVAILABLE** (named work item) |
| **§4 agreement** | S1 ≈ S2 | ✅ **reproduced and banked** |

### What this means for LAN, stated plainly

**LAN's premise — a dead route input — is refuted. Its remedy remains untested.** The two are
separable and I am not collapsing them:

* The chain of §1 facts still holds *as facts* (degenerate label, 0.27 validity, constant at eval).
* What is now **false** is the inference that the pathway is therefore unused. It is used.
* What is now **also false** is §7's proposed cheap alternative. Supplying the route — even the
  oracle route — does not help and degrades the lateral family.
* ⇒ The evidence points at **§6.1's REFUTE branch**: *"the architecture, not the label, is the
  defect, and the next question is why a live seam is ignored — not more route engineering."*
  Reached via `nav_cmd` rather than via a trained LAN arm, and **more cheaply**.

**I am not declaring LAN dead.** LAN encodes lateral topology at 20–160 m, which `nav_cmd` does not,
and its coverage is 3.2× better. But the cheapest discriminating experiment now says a *better route
label* is not obviously the lever, because a *perfect* one is not.

---

## 10. Named limits of this result

1. **n = 39 episodes / 859 windows**, not the canonical 40 / 881. One truncated val file.
2. **REF-C-base only.** XL untested here (it needs its own square-raster run); small untested.
3. **`oracle_route` is assembled, not re-decoded.** Valid because `nav_cmd` enters only through
   `measurement`, so the constant-command decodes span every reachable assignment — but it is an
   assembly, and that is why it is stated.
4. **The 2 s surface may be horizon-limited**, exactly as §8 of the prereg warns, and §7.1 above
   shows the 18.5 s instrument cannot be applied to this architecture. A 2 s null is not evidence
   about 18.5 s.
5. **Distance-keeping is missing** from LONGITUDINAL.
6. **No LAN-grafted checkpoint exists**, so §6.1/§6.2 remain genuinely unanswered about LAN itself.

---

## 11. Deliverable manifest

| artifact | where it lives | what it is |
|---|---|---|
| `LAN_E0_RESULTS.md` | repo, this directory, **staged** | this report |
| `E0_refc-base_navcf_full.json` | repo, this directory, **staged** | E0 + E0-B + four families + all CIs + all controls (the quotable artifact) |
| `Epre_lan_coverage_val39.json` | repo, this directory, **staged** | §6.0 launch gate |
| `lan_agreement_gated60_hyst1.json` | repo, this directory, **staged** | §4 agreement (was scratchpad-only) |
| `lan_agreement_UNGATED_control.json` | repo, this directory, **staged** | §4 heading-gate negative control |
| `stack/scripts/lan_probe.py` | repo, **staged** (modified) | `--navcf` now runs the real val cache + four families + paired bootstrap + `--epre` + geometry refusal + both negative controls |
| run directory | `tanitad-thor:/home/nvidia/lan_e0/` | source of every JSON above; logs at `tanitad-thor:/tmp/lan_e0_*.log` |
| Thor code sync | `tanitad-thor:~/TanitAD/{stack,taniteval}` | synced from repo HEAD and **verified by a real import**, not a file listing |

**Tests:** `pytest -q` in `stack/` — **1722 passed, 12 skipped, 2 xfailed**, 0 failures.
⚠️ The baseline given in the brief is **1719**. The **+3 is not mine**: a sibling agent's *unstaged*
`stack/tests/test_flagship4b.py` (+82 lines, 14 tests) is live in the working tree. Verified by
running that file alone. **This work adds no tests and changes no existing test**;
`tests/test_lan.py` is unchanged at 34 passed.

## 12. DONE vs NOT DONE

**DONE**
- E0 (§7) executed on the deployed REF-C-base at the correct 256px square raster, with all controls.
- E0-B executed: the fix §7 committed to on a RESPONSIVE reading — refuted.
- §6.0 E-pre launch gate executed: PASS, with the comparison bar re-measured rather than inherited.
- §4 agreement reproduced and banked out of the scratchpad, with its ungated control.
- Four metric families reported for every arm with episode-cluster CIs.
- Two instrument defects found and fixed (the reducer back-fill; the unreachable nav command).
- Two structural findings recorded (§6.2 not computable on REF-C; LAN cannot reach the aux heads).

**NOT DONE**
- **No LAN arm trained** — forbidden by the brief and unnecessary until the §6.1 question is re-framed.
- **§6.1 / §6.2 unanswered about LAN itself.** §6.2 needs a different instrument for this architecture.
- **distance-keeping (LONGITUDINAL half)** not wired — the lead-track reader exists and was not used.
- **`ep_00028.pt` not re-pulled** — 39/40 episodes.
- **REF-C-XL / small not swept.**
