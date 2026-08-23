# Q2 — The plan to dramatically improve AND validate our rendering

**Date:** 2026-08-03 · **Stream:** Benchmarks & Eval (research) · **Author:** benchmarks/render research agent
**Evidence classes:** `MEASURED (ours + artifact path)` · `PUBLISHED (cited)` · `INHERITED (our doc, NOT
re-verified)` · `ESTIMATED` · `HYPOTHESIS`.
⛔ **Ownership.** `stack/experiments/alpasim-gsplat/` and `stack/experiments/nurec-gsplat/` belong to the
renderer stream. **I changed none of their files.** Every item below that touches them is written as an
escalation with the exact file and the exact change, per the "escalate integration, don't write please-merge
in a doc" rule.

---

## 0. The reframing this plan turns on

**We are already rendering TRAINING views, and have been all along.** PUBLISHED (HF dataset card,
`nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`): the NuRec scenes were reconstructed **from 6 camera views —
front-wide 120°, front-tele 30°, cross left/right 120°, rear left/right 70°**. `camera_front_wide_120fov` —
the exact camera whose mp4 we score against — is one of them. Rendering at a rig pose from
`rig_trajectories.json` is therefore an **in-training-distribution view**, not a novel view.

⇒ **The brief's proposed ceiling probe ("render a training view the reconstruction was fitted on") is
already our production condition.** That is good news and it changes the experiment: the ceiling cannot be
found by choosing a different pose, because there is no *more* favourable pose. It has to be found by
changing **who renders it** and by establishing **what the metric pays for a perfect match**. §2 does both.

⚠️ Second probe on the 6-view claim is **owed and cheap**: `NuRecRig(rig_trajectories.json).camera_names()`
already exists in `nurec_loader.py:402` and has never been run for the record. One command on Thor. Until
it is run, "6 views including front-wide-120" is PUBLISHED-single-source.

---

## 1. State of play — with the absolutes marked as provisional

| fact | class | source |
|---|---|---|
| gsplat renders NuRec natively on aarch64; 26–36 ms/frame; closed loop 0.15–0.24 s/step | MEASURED | commit `c9926e6`; `results/2026-08-03-rolling-shutter/` |
| grad-NCC 0.2774 → **0.3424** (+23.4 %) via 4 layers + scale-cull 0.95 + gated sky 0.3 | MEASURED, **but scored against a 6-frame-wrong reference** | `panel6_chosen.report.json` |
| ⭐ the reference video is **+6 frames** offset on `00040136`: argmax histogram `{6: 12}`, unanimous 12/12, **0.3114 → 0.4911 (+0.1797, +57.7 %), free** | MEASURED | `rs_frame_offset_k10.json`, `ROLLING_SHUTTER.md` §11 |
| ⭐ the rule is **per scene, not a constant**: `video_index = rig_index + (n_mp4_decodable − n_rig_frames)`; **+6** on `00040136`, **+5** on `7c72937c`; the mp4 opens with a frozen leader block of exactly that length; `data_info.json` independently says 599 | MEASURED, 3 independent probes incl. a renderer-free one | `results/2026-08-03-rolling-shutter-adversarial/ALIGNMENT_DIRECTION_GPUFREE.json` |
| rolling shutter is offline-only: 0.3747 at **161×** cost, and a **free** render placed 2 readouts earlier beats it (0.3499 vs 0.3451, 12/12 neg-control, ~68× cheaper) | MEASURED | `ROLLING_SHUTTER.md` §8a |
| the black band is a **reconstruction hole**, not an FOV clip (`frac_pixels_beyond_max_angle = 0.0`, IoU 0.0) | MEASURED | `diagnose_f0.json` / `diagnose_f150.json` |
| the "magenta smear" does not exist (0.006 % of frame; at f150 the **reference** has 6× more than we render); the real artifact was over-sized splat streaks, now culled | MEASURED | same |
| ⛔ **grad-NCC is the only admissible metric on these night clips**; PSNR, NCC **and MAE** are retracted (5/5 vs 1–4/5, arm-dependent) | MEASURED | `ROLLING_SHUTTER.md`, open-loop README |
| production renders with **appearance basis `f0` evaluated once at `tau = 0.0`** — the scene ships a `time_embed` block with `timestamps_us_min/max` and per-layer `fourier_features_dim`, and we use the **DC term only, frozen for the whole clip** | MEASURED | `gsplat_renderer.py:22, 373, 398-409`; `nurec_loader.py:131` |
| `sh_degree=3` in every rasterisation call | MEASURED | `gsplat_renderer.py:699, 875` |

> ⛔ **Every render-fidelity ABSOLUTE we have published is provisional until §R0 lands.** Paired deltas
> between arms survive (all arms shared the same wrong reference — stated by the renderer stream itself);
> levels do not. The renderer is materially better than any number we have quoted.

---

## 2. THE PLAN — ordered by expected gain per unit effort

Gain is in grad-NCC on `00040136` unless stated. Effort is wall-clock for one engineer on Thor, GPU-free
items marked ⓪.

### R0 — Land the per-scene frame-alignment rule, then RE-BASELINE everything · gain **+0.1797 MEASURED** · effort ~2 h

The single largest, already-proven, free lever. Three parts:

1. **Apply the rule** `video_index = rig_index + (n_mp4_decodable − n_rig_frames)`, computed **per scene**,
   in the scorer's reference loader (`render_quality.py:load_refs`).
2. **Add a neighbour-offset scan (±3) to the negative control.** ⚠️ The existing control **cannot** catch
   this class of error *by construction*: `render_quality.py:48` sets `MIN_WRONG_GAP = 40`, so a 6-frame
   error is invisible to it. A control that cannot fail on the defect that actually occurred is not a
   control. Regression condition: `argmax` of the ±3 scan must be **0**.
3. **Re-run panels 1–6** and restate every absolute. Paired deltas need no rerun.

**Falsifier.** On a **third** scene, the fitted argmax must equal `n_mp4 − n_rig`. If it does not, the rule
is wrong and we revert to per-scene fitting (which is still correct, just not predictable).
**Escalation:** owner = renderer stream; files = `render_quality.py`, `rs_frame_offset.py`. Do **not** hard-code
`+6` — the second scene already refutes a constant (MEASURED, `+5`).

### R1 — Establish the CEILING, in three legs · gain: interpretation of every number we own · effort ⓪ + 1 pod-day

Without a ceiling we cannot tell a renderer bug from a reconstruction limit. Because §0 shows we already
render training views, the ceiling is a **three-leg decomposition**, and the cheapest leg comes first.

**Leg C ⓪ — the metric's own floor (do this first; no GPU, no renderer).**
Score the reference against *itself* at neighbouring indices: `grad-NCC(ref_t, ref_{t±1})`, `(ref_t, ref_{t±2})`
over the same 12 frames. This is what the metric pays for **two genuinely consecutive real frames of the
same scene** — the empirical "perfect but not identical" value.
⇒ Without it, "0.49" is uninterpretable: we do not know whether 1.0 is reachable or whether real
consecutive frames already score 0.6. Everything downstream is expressed as a fraction of this floor.
**Falsifier:** if `grad-NCC(ref_t, ref_{t+1})` ≈ 1.0, the metric has no motion floor and the raw scale
stands as-is.

**Leg A — the asset ceiling: a SECOND renderer on the SAME pose.**
Render the identical pose with NVIDIA's own NuRec/NRE renderer and score it with the identical harness.
`grad-NCC(NRE, ref)` upper-bounds what any renderer can extract from this reconstruction; the **gap between
it and our gsplat is the only part fixable in our code**.
⚠️ Availability is a real cost, and there are two paths (absence at one location is not absence):
(i) the NGC-gated `nvcr.io/nvidia/nre/nre-ga:26.04` image — 38 GB, pull recipe is **prose-only** and the
eval pod was re-provisioned bare (MEASURED, `ALPASIM_STATE.md` row 14 + `LOOP_STATE.md`); (ii) **CARLA now
ships a NuRec integration** (PUBLISHED, `carla.readthedocs.io/en/latest/nvidia_nurec/`) — a non-NGC third
party that consumes the same USDZ. Path (ii) is the fallback and should be probed before paying for (i).
**Falsifier:** if NRE scores within ~0.02 of our gsplat at the corrected index, **our renderer is done** and
every remaining lever in §R3 is worth ≤0.02 — a result that would save weeks.

**Leg B — a cross-camera control (cheap, and it discriminates FOV handling from asset quality).**
Render `camera_front_tele_30fov` — another *training* view, 30° instead of 120° — and score it. If tele
scores far higher than wide, the defect is our wide-angle / f-θ handling, not the reconstruction.
Cost: one extra mp4 download + one render pass.
**Falsifier:** tele ≈ wide ⇒ FOV handling is not the defect; the residual is the asset.

### R2 — Where the residual LIVES, masked by coverage · gain: correct attribution · effort ~3 h

After R0, decompose the residual into three populations and report grad-NCC **on the covered mask** plus
coverage as its own number:
(i) fully claimed (`alpha ≥ 0.995`), (ii) partially claimed, (iii) unclaimed.
The corrections already in hand and to be honoured: `mean_alpha = 0.5975` (not 0.5145 — that came from a
retracted run), **"no gaussian at all" is 0.00 % at frame 0**, and "79–81 % of error in uncovered pixels"
is RETRACTED. The black band is a hole in population (iii).
**Falsifier:** if masked and full-frame grad-NCC rank the same across ≥6 arms (Spearman ≥ 0.95), masking
adds nothing and we drop it rather than carry two numbers.

### R3 — The remaining levers, ranked with an honest prior

| # | lever | why it is plausible | expected | effort |
|---|---|---|---|---|
| **L1** | ⭐ **Appearance basis: use the scene's TIME-VARYING Fourier features instead of `f0` at `tau=0`** | MEASURED: the asset ships `time_embed` (`timestamps_us_min/max`) and per-layer `fourier_features_dim`; production evaluates the **DC term once for the whole clip** (`gsplat_renderer.py:373`). On a **night** clip — headlights, streetlights, exposure drift — a time-frozen appearance is exactly the wrong approximation. The production renderer's `_basis` does not even implement the Fourier bases; `render_probe.py` does (`fourier_cs`/`fourier_sc`/`tent`). | **HYPOTHESIS — the largest untested renderer-side lever we have** | 0.5–1 d + a per-frame re-activation cost to measure (~760 k gaussians) |
| **L2** | **Principled hole-fill for the reconstruction hole** (the black band). The scene already ships a sky env-map (deployed at gain 0.3); the published instrument for holes is **NuRec Fixer**, a transformer post-process (PUBLISHED, NVIDIA NuRec docs) | the reference shows `[28.7, 26.0, 22.8]` where we render black — a bounded, localised error | ESTIMATED small-to-moderate; bounded by the band's pixel fraction | 1 d |
| **L3** | **Scale-cull percentile sweep** (0.90 / 0.95 / 0.98 / off) | MEASURED: 0.95 removed the streaks **and was cheaper** (20.8 ms). The optimum was never bracketed | ESTIMATED ±0.01 | 2 h |
| **L4** | **Sky env-map gain sweep** re-run at the corrected index | the 0.3 gain was chosen against a 6-frame-wrong reference — its optimum may have moved | ESTIMATED small | 1 h |
| **L5** | **Per-layer compositing order** (4 layers) | order errors show as haloing at layer seams; the seam control already exists (`rs_seam_control.py`) | ESTIMATED small | 2 h |
| **L6** | **Exposure / tonemap fit** — `isp_experiment.py`, `ppisp.py`, `isp_report.json` exist | MEASURED: per-frame ISP was already **refuted** as a residual source (total effect **0.18 %**) — so a *global* per-clip gain/gamma is what remains | ESTIMATED ≤0.005 | 2 h |
| **L7** | ⛔ **Densification / pruning thresholds** | **Not available to us**: the asset is a *pre-fit* reconstruction, we do not train it. We can cull (L3), not densify. Listed so it is not re-proposed | n/a | n/a |
| **L8** | ⛔ **Rolling shutter** | MEASURED and settled: +0.021 at ~90× cost, and the gain is **not the shutter** (a backwards sweep reproduces it). Do not reopen | n/a | n/a |

⚠️ **L6 carries a leak risk that must be designed out:** fitting exposure against the reference *is fitting
to the metric*. Fit on a held-out subset of frames, validate on the rest, and report both — otherwise it is
the same family as the REF-A I-JEPA leak.

### R4 — The STANDING VALIDATION TEST (this is what the PI asked for) · effort ~1 d ⓪-ish

A render change is accepted or rejected by this, and only this.

| element | specification | why |
|---|---|---|
| **frames** | **ALL frames of the clip**, not 5, not 12 | 26–36 ms/frame × 599 ≈ **16–22 s of GPU** per arm. The sampling excuse does not survive its own cost |
| **scenes** | **≥3**, spanning night / day / junction | the offset finding was **+6** on one scene and **+5** on the next — one scene has already refuted one generalisation |
| **metric** | **grad-NCC only**. MAE/PSNR/NCC may be *printed* and decide **nothing** | 5/5 vs 1–4/5, arm-dependent |
| **estimator** | **paired bootstrap over FRAMES**, before-vs-after on identical frames; report mean, p10, p90 and `frac(after < before)` | the mean is not what a viewer sees (§R5) |
| **negative control 1** | wrong-frame identification at gap ≥ 40 — must be **12/12 on every scene** | already exists |
| **negative control 2** | **neighbour-offset scan ±3; argmax must be 0** | ⛔ the defect that actually happened was invisible to control 1 *by construction* |
| **cost gate** | render ≤ **36 ms** | MEASURED: the 10 Hz tick has ~36 ms left after the model's 60.3 ms p50 |
| **REGRESSION** | paired Δ CI entirely **below** 0 on any scene, **or** control 1 < 12/12, **or** control 2 argmax ≠ 0, **or** cost > 36 ms | any one is disqualifying |
| **IMPROVEMENT** | paired Δ CI entirely **above** 0 on **≥2 of 3 scenes**, controls intact, cost gate met | a single-scene win is not a win |
| **artifact** | **a side-by-side VIDEO with the per-frame grad-NCC trace burned in**, plus the per-frame curve as data | §R5 |

### R5 — ⭐ "The still looked better but the VIDEO did not" — the PI's observation, made testable

This is the most interesting open item in the render work, because a 5-frame panel measures a **mean** while
a viewer perceives a **distribution and its time-derivative**. Three mechanisms, each with a falsifier;
they are not exclusive.

- **H-A — sampling.** 5 frames is not the clip; the after-arm may have improved the mean while adding a fat
  lower tail. **Test:** the full-clip per-frame curve from R4; report `frac(after < before)`.
  **FALSIFIER:** if `frac(after < before) < 5 %`, sampling is not the mechanism.
- **H-B — temporal flicker (my primary hypothesis).** Per-frame quality can be fine while *changes between
  frames* are wrong. Scale-culling is a **hard threshold**, so a splat can pop in and out between frames;
  a still cannot show this and grad-NCC-per-frame cannot either. **Instrument to build (new, ~half a day):
  a differential flicker score** — `grad-NCC(render_t, render_{t+1}) − grad-NCC(ref_t, ref_{t+1})`. It is
  zero when our temporal behaviour matches the reference's own, and negative when we flicker more.
  **FALSIFIER:** if the after-arm's differential flicker is ≥ the before-arm's, flicker is not the
  mechanism and H-A/H-C carry it.
- **H-C — a temporal LAG, not a quality drop.** A 6-frame reference offset is **~0.2 s**. A still cannot
  reveal a 0.2 s lag; a moving video makes it obvious as "everything is slightly late". This predicts that
  **after R0 the video improves more than the stills do** — a genuinely falsifiable, cheap prediction, and
  the cleanest link between the two halves of this document. **FALSIFIER:** if the re-rendered video after
  R0 still reads as worse than the stills, H-C is dead. *(HYPOTHESIS — mine, not measured.)*

**Recommendation on the standard artifact: YES — the side-by-side VIDEO becomes the standard, and the still
is demoted to an illustration.** Concretely: `overlay_video.py` already produces side-by-side clips; the
addition is (i) the per-frame grad-NCC value burned into each frame and (ii) the curve emitted as JSON next
to the mp4, so the number and the pixels can never be quoted apart. ⚠️ `.gitignore:24` excludes `*.mp4`
(MEASURED, 4 probes in `ALPASIM_STATE.md` row 12) — the video artifact must be staged with `git add -f` or
it will be silently absent while a manifest claims otherwise. That has already happened once.

---

## 3. Execution order (what I would do, in order)

| order | item | why here |
|---|---|---|
| 1 | **R0** frame alignment + control 2 + re-baseline | +0.1797 measured, free, and it invalidates the numbers every other item would be compared against |
| 2 | **R1 Leg C** ⓪ adjacent-frame metric floor | zero GPU, and it makes every number in this document interpretable |
| 3 | **R4** full-clip per-frame harness + video artifact | it *is* the validation the PI asked for, and it is the measurement instrument for everything below |
| 4 | **R5 H-B** differential flicker instrument | explains the PI's observation or kills my hypothesis; half a day |
| 5 | **R3 L1** time-varying appearance basis | the largest untested renderer lever, grounded at file:line |
| 6 | **R2** masked residual | attribution, cheap |
| 7 | **R1 Leg B** tele cross-camera control | discriminates FOV handling from asset quality, one download |
| 8 | **R1 Leg A** second renderer (NRE, else CARLA-NuRec) | highest value for the ceiling question, highest acquisition cost |
| 9 | **R3 L2–L6** hole-fill, cull sweep, sky gain, layer order, global exposure | each gated by R4 |

**The one thing that would change this order:** if R1 Leg A can be obtained cheaply (CARLA-NuRec path
works), promote it to position 3 — knowing the ceiling before spending on levers is worth more than any
individual lever.

---

## 4. What this plan does NOT claim

- **No render-fidelity number here is a driving-quality claim.** Whether any of it moves the four families
  closed-loop is **NOT ESTABLISHED** — no closed-loop arm was run in this stream. The link that *is*
  testable is the reconstruction-OOD factor (REF-C **3.21×**, `REFC_openloop_diagnostic.json`): if render
  fidelity rises and 3.21× falls, the closed-loop levels become more trustworthy. **Pre-register both
  outcomes** — a fidelity gain that leaves 3.21× unmoved would mean grad-NCC is not measuring what the
  policy is sensitive to, which is a more important finding than the fidelity gain itself.
- **The +0.1797 and the ±offset rule are MEASURED by the renderer stream, not re-verified by me.** I read
  the artifacts (`rs_frame_offset_k10.json`, `ALIGNMENT_DIRECTION_GPUFREE.json`) and quote the run
  directories; I did not re-run them.
- **The 6-camera reconstruction claim is PUBLISHED single-source** until `camera_names()` is run (§0).
- **L1's cost is unmeasured.** Per-frame SH re-activation over ~760 k gaussians may not fit the 36 ms gate;
  the quality question is still worth answering **offline**, where the gate does not apply.
- **Nothing here was executed on Thor in this stream.** This is a plan plus one executable instrument
  (`bench_geometry_check.py`, Q1) — the render items are escalations to the stream that owns those files.

## 5. Sources

MEASURED (ours): `stack/experiments/alpasim-gsplat/results/2026-08-03-rolling-shutter/ROLLING_SHUTTER.md`
(§7, §8a, §11, §12) · `…/rs_frame_offset_k10.json` · `…/rs_sweep_chosen.report.json` ·
`stack/experiments/alpasim-gsplat/results/2026-08-03-rolling-shutter-adversarial/ALIGNMENT_DIRECTION_GPUFREE.json` ·
`…/results/render-quality/{panel6_chosen.report.json, diagnose_f0.json, diagnose_f150.json}` ·
`stack/experiments/alpasim-gsplat/{gsplat_renderer.py, render_quality.py, overlay_video.py}` ·
`stack/experiments/nurec-gsplat/{nurec_loader.py, render_probe.py}` · commit `c9926e6` ·
`TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-26-alpasim-consolidation/ALPASIM_STATE.md` ·
`…/2026-07-22-alpasim-closedloop-evalpod/REFC_openloop_diagnostic.json`.

PUBLISHED: [PhysicalAI-AV-NuRec dataset card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec)
(6 reconstruction views) · [NVIDIA Omniverse NuRec](https://developer.nvidia.com/omniverse/nurec) (Fixer,
novel-view rendering, quality evaluation) · [NuRec data guide](https://docs.nvidia.com/nurec/nurec/physical-ai-data.html)
(scene ships `data_info.json`, `rig_trajectories.json`, `sequence_tracks.json`, `map.xodr`) ·
[CARLA × NuRec integration](https://carla.readthedocs.io/en/latest/nvidia_nurec/) (the non-NGC second-renderer path).
