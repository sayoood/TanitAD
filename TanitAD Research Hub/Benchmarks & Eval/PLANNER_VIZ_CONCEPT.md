# PLANNER VIZ — the plan fan

**What it is.** A metric-BEV rendering of a planner's *entire* proposal set, coloured by
the planner's own score, with the selected plan highlighted — plus the diagnostic that
falls out of having the whole set on screen: **oracle-in-fan**.

**Status.** Built 2026-07-20 for REF-C-XL (anchored diffusion, step 29999). The renderer
is deliberately written so the *same* panel serves the v3 / P2 CEM planner with
`cost` substituted for `confidence` — see [§7 Reuse for v3](#7-reuse-for-v3--p2).

**Code.** `taniteval/taniteval/plan_fan.py` (repo) → `/root/taniteval/taniteval/plan_fan.py`
(eval pod). It imports its drawing vocabulary from `direct_overlay` / `corpus_overlay` /
`flagship_overlay` (`_fit`, `FlatProjector`, `pretty_man`, `pretty_route`, `WP_IDX`,
`ego_future_path`, the `COL_*` / `HUD_*` palette) rather than forking it, so the plan fan
and the existing overlay videos share one visual language.

---

## 1. What the decoder actually computes

Everything below is read off `stack/tanitad/refs/refc.py`
(`AnchoredDiffusionDecoder.forward`) and the run's own `config.json`, **not** off the
DiffusionDrive paper. Where our implementation differs from the paper, the code wins and
the difference is called out — the whole point of the panel is that it renders what the
model *does*.

| # | Fact | Consequence for the viz |
|---|---|---|
| 1 | One classifier pass over the **full** anchor vocabulary: N=256 anchor trajectories become queries, cross-attend the 8×8×F conv map under FiLM(condition), emit `conf [B,N]` (logits) and `offset [B,N,S,2]`. `x = anchors + offset`. | Every anchor is refined, so there is a proposal to draw for all 256. |
| 2 | **All N anchors are denoised — there is no top-K gate.** The truncated-diffusion loop calls `_decode` on the whole `[B,N,S,2]` tensor each step; no `topk`/gather appears anywhere in the module. `model.eval()` zeroes the noise, so the 2 steps are deterministic. | The scored fan is *all 256 refined* proposals; the shadow layer is the *raw* pre-refinement vocabulary. (Had only top-K been denoised the fan would have been the top-K and the shadow would have been anchor+offset. It is not the case here.) |
| 3 | **The score is computed on the anchor; the geometry shown is the denoised one.** The denoise passes return `_, off` — their confidence output is *discarded*. Selection uses the t=0 classifier-pass confidence over the *original* anchors. | Scoring and refinement are decoupled *inside the model*. This is the structural reason the oracle-in-fan gap can be large: the thing being scored is not the thing being driven. |
| 4 | H19 maneuver prior is in the logits: `conf ← conf + maneuver_to_anchor(log_softmax(maneuver_logits))` (`graft_maneuver=True`). | The returned `anchor_logits` are *post*-reweight, so their softmax **is** the selection distribution. The HUD's tactical maneuver is the head that produced this prior — when the fan's colour is wrong, that head is a suspect. |
| 5 | `grounded_selector=False` in this run → `score == conf` and `sel_idx == argmax(anchor_logits)`. | The renderer **asserts** this every batch and refuses to draw if a future ckpt breaks it (with `grounded_selector=True` the colours would no longer be the selection score). |

**Trajectory surface.** Every proposal is 4 *time* waypoints at `WP_STEPS` 5/10/15/20
(= 0.5/1/1.5/2 s), ego frame of the last window pose — the same surface, same GT and same
ADE definition as every other arm's leaderboard row (`refc_eval.collect`). Polylines are
drawn ego-origin → wp1 → wp2 → wp3 → wp4 as straight segments: **a drawing device only**.
No curvature is invented between waypoints and every number in the HUD is computed from
the 4 waypoints alone.

---

## 2. Layer spec

Back to front. Ego at bottom-centre, heading up, **isotropic** metres (a circle is a
circle — unlike the small BEV inset in `corpus_overlay`, which scales the axes
independently).

| # | Layer | Encoding |
|---|---|---|
| 1 | **Vocabulary shadow** | all N *raw* anchors (pre-refinement), flat grey, α≈46, width 1. The reachable set the model was given. |
| 2 | **Scored fan** | all N *refined* proposals. Colour = softmax confidence through the viridis LUT; α = 18 + 216·t; width = 1 + 2.6·t. Drawn in **ascending score order** so winners land on top of losers. |
| 3 | **Top-8 emphasis** | +2 px width, full α, waypoint dots. |
| 4 | **Selected plan** | score-coloured halo (width 11) under a white core (width 4); 4 waypoint markers labelled `0.5s … 2.0s`. |
| 5a | **Oracle proposal** | the best-available proposal in the fan, thin cyan, endpoint ringed and labelled. Only drawn when it is not the selected one. |
| 5b | **Ground truth** | dashed bright green, width 4, filled waypoint dots, drawn **last** so the reference is never occluded. |
| 5c | **Per-horizon error bars** | one dashed red *dimension line* per horizon, from the selected waypoint to the GT waypoint, offset into its own lateral lane (14 + 10·j px) and labelled in metres. |
| 6 | **HUD / colorbar / legend / 5 m grid** | see below. |

### Why the error bars are offset

REF-C's dominant error is **longitudinal** — right shape, wrong distance along it. Drawn
*on* the path, an error tie-line is exactly collinear with the plan and disappears under
the selected plan's white core; the panel then shows a plan and a GT that look identical
while the HUD says ADE 2.9 m. Offsetting each horizon into its own lane turns the four
ADE terms into a legible little bar chart and makes "the model is cruising while the ego
brakes" a one-glance read. This was found the hard way during bring-up (v3 of the panel
looked *wrong* until the bars went in), and it is the single most important legibility
decision in the design.

### BEV range

Derived from the clip's speed profile — `max(v0)·2 s + 8 m`, floored at 20 m, never
clipping GT or the selected plan, rounded up to 5 m, capped at 90 m — and held **fixed
for the whole clip**. A per-frame rescale would jitter the video and, worse, would make
the fan's apparent spread incomparable between frames. The active range is printed in
the panel title.

---

## 3. Score colormap semantics

- **Colormap**: viridis, hardcoded 10-stop LUT with linear interpolation.
  `matplotlib is NOT installed on the eval pod` — do not add it as a dependency for this.
  Perceptually uniform and colour-blind-safe, so "brighter = the model likes it more"
  survives greyscale printing and reviewers.
- **Scale**: `t = (log10 p − log10 1e-4) / (0 − log10 1e-4)`, i.e. a **fixed** 4-decade
  log scale from p = 1e-4 to p = 1. *Fixed*, not per-frame renormalised: a colour means
  the same probability in every frame of every clip, so the fan's brightness is
  comparable across a video and across clips. Log, not linear, because a peaked softmax
  over 256 anchors would render as one bright line on a black field and destroy exactly
  the structure the panel exists to show.
- **The uniform tick.** The colorbar carries a red tick at p = 1/N (= 1/256 ≈ 3.9e-3).
  Everything brighter than that tick is a proposal the model *prefers* to ignorance;
  everything darker is one it has actively down-weighted. This makes "how peaked is the
  planner" readable without a number.
- **HUD companions**: `top-1 p`, `entropy` in nats against its ceiling `ln N` (5.55 for
  256), and `modes>1%` — the count of proposals above 1 % probability. Together they say
  whether a frame is genuinely multimodal or a one-mode collapse.

---

## 4. The oracle-in-fan diagnostic

Three numbers per frame, all with the *same* ADE definition:

| Quantity | Definition | Reads as |
|---|---|---|
| `ADE(selected)` | ADE of the proposal the model **picked** | what the model delivers |
| `oracle-in-fan` | **min over all N refined proposals** of that proposal's ADE | the best the model *could have* delivered without changing a single weight of the generator |
| `vocab-oracle` | min over all N **raw** anchors | what the anchor vocabulary alone can reach — the coverage floor, independent of the offset head |

**The gap `ADE(selected) − oracle-in-fan` separates two completely different failures:**

- **Large gap** → the fan **contained** a good plan and the model failed to **score** it.
  The generator is fine; the *selector* is broken. The fix is a better **planning cost**,
  not a bigger vocabulary, not more capacity, not longer training.
- **Small gap, both large** → nothing in the fan was any good. That is a **coverage**
  failure: vocabulary, offset head, or conditioning. Compare against `vocab-oracle` to
  decide whether the anchors or the refinement is at fault.

This is the panel's reason to exist. A single-trajectory overlay cannot distinguish these
two cases at all — both look like "the line is in the wrong place".

**Honest limits of the oracle.** It is an *oracle* — it uses the GT to pick. It is a
diagnostic upper bound on what better scoring could buy, **not** an achievable score and
**not** a metric to report on a leaderboard. It also flatters the model slightly: with
256 proposals, min-over-256 has some free-lunch variance. Read it as "there exists a
proposal this close", not "a realizable planner would get this".

---

## 5. First read — REF-C-XL step 29999, canonical physicalai val

Four clips, stride 1, whole episode (170–171 windows each). Every number is the
leaderboard ADE definition; per-frame tables in
`Research/videos-2026-07-20/planfan_*.json`.

| clip | v0 regime | ADE(sel) | oracle-in-fan | vocab-oracle | gap | top-1 p | entropy | modes>1% | frames where sel > 2× oracle *and* gap > 0.3 m |
|---|---|---|---|---|---|---|---|---|---|
| ep31 highspeed-straight | 36.5 m/s | **0.146** | 0.117 | 0.294 | +0.029 | 0.999 | 0.01 | 1.0 | **0 / 171 (0 %)** |
| ep28 highspeed-curve | ~20 m/s | 0.700 | 0.296 | 0.626 | +0.404 | 0.590 | 1.12 | 5.1 | 82 / 171 (48 %) |
| ep03 sharpturn | ~4–7 m/s | 0.896 | 0.229 | 0.596 | +0.667 | 0.521 | 1.38 | 6.6 | 94 / 171 (55 %) |
| ep11 failure-worstwindow | ~6 m/s | 1.110 | 0.295 | 0.592 | +0.815 | 0.508 | 1.38 | 6.4 | **111 / 170 (65 %)** |

Worst windows (the frames the stills capture):

| clip | worst frame | ADE(sel) | oracle-in-fan | vocab-oracle | ratio | top-1 p | modes>1% |
|---|---|---|---|---|---|---|---|
| ep11 | f080 | **2.572** | **0.305** | 0.290 | **8.4×** | 0.314 | 9 |
| ep03 | f149 | 5.059 | 0.435 | 0.700 | 11.6× | 0.316 | 9 |
| ep28 | f129 | 1.792 | 0.357 | 0.950 | 5.0× | 0.390 | 8 |
| ep31 | f020 | 0.287 | 0.163 | 0.523 | 1.8× | 0.998 | 1 |

**Verdict: REF-C's failure is SCORING, not coverage.** On the ep11 worst window the fan
contained a plan **8.4× better** than the one selected (0.305 m vs 2.572 m), and the raw
anchor vocabulary alone already contained a 0.290 m plan — so neither the vocabulary nor
the refinement is the bottleneck; the confidence head simply ranks the right proposal
below a wrong one. Clip-wide, the selected plan is more than twice as bad as the
best-available one in **65 %** of ep11's frames.

Two further reads fall straight out of the table:

- **The gap tracks uncertainty.** Where the model is confident it is also right
  (ep31: top-1 p 0.999, entropy 0.01, one mode, gap 0.03 m — a *complete* mode collapse,
  and harmless). Where it is uncertain the gap explodes (ep11/ep03: top-1 p ≈ 0.51,
  entropy 1.38, ~6.5 modes, gap 0.7–0.8 m). The whole 256-proposal apparatus buys nothing
  on easy straights and is mis-ranked exactly where it would matter.
- **Refinement roughly halves the reachable error** (vocab-oracle 0.29–0.63 →
  oracle-in-fan 0.12–0.30), so the offset head is doing real work. It is the *score* that
  fails to follow it — consistent with code fact #3 in §1: the score is computed on the
  raw anchor at t=0 and never updated after the denoise steps refine the geometry.

The dominant error direction is **longitudinal**. In the ep03 worst window the ego is
braking (GT forward 3.36 / 5.97 / 7.96 / 9.50 m over 2 s) while the selected plan is
essentially constant-velocity at v0 (3.77 / 7.60 / 11.48 / 15.32 m ≈ v0·t = 3.7 / 7.4 /
11.1 / 14.8) — per-horizon errors 0.41 / 1.64 / 3.54 / 5.85 m. Proposal #203 in the same
fan predicts the deceleration (3.56 / 6.33 / 8.48 / 10.28 m, ADE 0.478 m) and is not
picked. This is the same longitudinal lever already logged for the flagship, now shown to
be a *selection* problem in REF-C rather than a generation one.

**Direct consequence for v3:** a planning **cost** with an explicit longitudinal /
target-speed term is the lever, and it must be evaluated on the *refined* trajectory
rather than inherited from the anchor. An oracle-in-fan gap this large is the strongest
available evidence that ranking — not capacity, not vocabulary, not training length — is
what is left on the table.

---

## 6. Reading guide

| What you see | What it means |
|---|---|
| Wide grey shadow, narrow bright fan | refinement is collapsing the vocabulary toward one mode — check whether the collapse is *toward* the GT or *away* from it |
| Bright fan spread wide, GT inside it, big error bars | **scoring failure** — the classic large-gap case |
| GT outside the whole shadow | **coverage failure** — the vocabulary cannot reach the manoeuvre |
| Error bars all pointing the same way, growing with horizon | **longitudinal** (speed) error, not shape error |
| `modes>1%` = 1, `top-1 p` ≈ 1 | the planner has collapsed to a single mode; a fan viz is not buying you anything on that frame |
| Cyan oracle line far from the white plan but both plausible | the scoring is picking the wrong *mode*, not mis-refining the right one |

---

## 7. Reuse for v3 / P2

The v3 direction (frozen-encoder DINO-WM predictor + CEM/diffusion/MPC planner) replaces
"anchor confidence" with "planning **cost**", but the panel is structurally identical:
a set of candidate trajectories, a scalar per candidate, one of them selected.

To retarget the renderer:

1. **Feed candidates instead of anchors.** `draw_bev(...)` takes `fan [M,S,2]` and
   `probs [M]` as plain lists — a CEM iterate's rollouts drop straight in. The shadow
   layer becomes the *previous* CEM iterate (or the sampling prior), which makes CEM's
   contraction over iterations visible in one image.
2. **Invert the scale.** Cost is better when low, so map `t = 1 − normalise(cost)` and
   **relabel the colorbar** — the same viridis, but bright = *cheap*. Do not silently
   reuse the confidence labelling; the colorbar text is part of the honesty contract.
3. **Replace the fixed log-probability scale.** A cost is not a probability and has no
   natural 1/N reference. Use a fixed scale in cost units per cost term, and drop the
   uniform tick (there is no "uniform" for a cost). If the planner is a softmax-over-cost
   sampler, the probability scale can be kept as-is and the cost shown as a second bar.
4. **Keep the oracle-in-fan.** It transfers unchanged and is arguably *more* valuable for
   CEM: min-over-candidates ADE vs selected ADE separates "the sampler never proposed a
   good trajectory" (widen/re-seed the sampler) from "the cost ranked a good trajectory
   below a bad one" (fix the cost terms). That is the core design loop for the v3
   planning cost.
5. **Per-cost-term panels.** When the cost has terms (progress, collision, comfort,
   target-speed), render one small fan per term coloured by that term alone. The frame
   where the terms disagree is the frame that tells you the weights are wrong.

The longitudinal lesson from §2 carries over verbatim: if the v3 cost has a target-speed
term, its failures will be longitudinal and **invisible** without the offset error bars.

---

## 8. Implementation hazards — read before reusing

**PIL wide lines divide by the segment length.** `ImageDraw.line(..., width>1)`
builds each segment's quad from a normal scaled by `width / hypot(dx, dy)`. A zero- or
near-zero-length segment therefore divides by ~0, and the resulting non-finite polygon
sends the C rasteriser off to fill a ~1e16-wide box. It does **not** crash — it *grinds*:
one core pinned, minutes per frame, no output, no error. During bring-up this stalled the
renderer on exactly one frame of ep03 for 8+ minutes before it was caught.

Two ways it bites, both of which **will** recur with CEM candidates:

1. **Repeated waypoints.** Any plan for a stopped or nearly-stopped vehicle has
   coincident waypoints. The anchor vocabulary contains such trajectories by
   construction, and a CEM sampler will produce them too. `_clean()` drops
   non-finite points and consecutive duplicates before anything reaches PIL; every
   data-driven polyline goes through `_line()` / `_dashed()`, never raw `d.line`.
2. **Dash-walk underflow.** The natural dashed-line implementation carries the leftover
   dash length across segments. That carry accumulates float error until the residual
   span underflows to ~1e-15, emitting precisely the degenerate wide line above.
   `_dashed()` therefore walks a **single cumulative arc-length parameter** with a
   closed-form loop bound and refuses to emit any dash shorter than 0.5 px. Do not
   "simplify" it back into an incremental carry.

Because the failure mode is a silent hang rather than an exception, anything that renders
long clips should be sanity-checked by watching the frame files appear (`ls | wc -l` over
a fixed interval), not by waiting for a traceback. `py-spy` is not usable on the eval pod
(the container lacks `SYS_PTRACE`); `faulthandler.dump_traceback_later(N, exit=True)`
around the frame loop is the working substitute and is how this was localised.

---

## 9. Provenance

```bash
# eval pod
PYTHONPATH=/root/taniteval:/root/TanitAD/stack \
  python3 -m taniteval.plan_fan --model refc-xl-30k \
    --clips 3:sharpturn,31:highspeed-straight,28:highspeed-curve,11:failure-worstwindow \
    --stills 2 --batch 4 --max-frames 400
```

- Checkpoint: `refc-xl-30k` → `/root/models/refc-xl-30k/ckpt.pt`, **step 29999** (the last
  of the 30 000-step schedule; `metrics.json` `final.step = 29999`, `steps = 30000`).
  Pulled pod3 → eval over the direct agent-forwarded path,
  md5 `966d4eff1ea5ddf86efba01b8344e198` verified identical on both sides, with the pod3
  trainer already exited (GPU idle) so the source file was quiescent.
- Corpus: `physicalai` canonical val, `/root/valdata/physicalai-val-0c5f7dac3b11`.
- Decoder: 256 FPS anchors (externally built `refc_anchors_full.pt`, carried in the
  `decoder.anchors` buffer), 2 truncated-denoise steps, `grounded_selector=False`,
  `graft_maneuver=True`, `graft_imagination=True`, `refc1=False`.
- Per-frame evidence tables: `/root/taniteval/results/planfan_<clip>.json` (selected /
  oracle / vocab ADE, top-1 p, entropy, mode count, v0, selected + oracle indices for
  every frame). These are the numbers behind every claim in §5.

### Artifacts

Videos (10 Hz, 1280×800, H.264) live on the eval pod at
`/root/taniteval/results/videos/` — reachable through the jupyter proxy — and are
mirrored into `Research/videos-2026-07-20/` in this repo. **The `.mp4` files are
`.gitignore`d (`*.mp4`), so the repo copy is a local mirror only; the pod is the
canonical location for the videos.** The PNG stills and the per-frame JSON tables are
tracked.

| artifact | eval pod | repo |
|---|---|---|
| `refc-planfan_step29999_ep03_sharpturn.mp4` (171 f) | `results/videos/` | mirror, gitignored |
| `refc-planfan_step29999_ep31_highspeed-straight.mp4` (171 f) | `results/videos/` | mirror, gitignored |
| `refc-planfan_step29999_ep28_highspeed-curve.mp4` (171 f) | `results/videos/` | mirror, gitignored |
| `refc-planfan_step29999_ep11_failure-worstwindow.mp4` (170 f) | `results/videos/` | mirror, gitignored |
| 11 × `..._f<NNN>_{worstwindow,multimodal}.png` | `results/videos/` | tracked |
| 4 × `planfan_refc-planfan_step29999_*.json` | `results/` | tracked |

Paper-ready stills, in order of usefulness:

1. `refc-planfan_step29999_ep11_failure-worstwindow_f080_worstwindow.png` — the headline
   scoring failure: selected 2.57 m, oracle-in-fan 0.31 m, 9 live modes, GT curving away
   from a straight-ahead pick.
2. `refc-planfan_step29999_ep03_sharpturn_f149_worstwindow.png` — the largest gap
   observed (5.06 m vs 0.44 m) and the clearest longitudinal error-bar ladder.
3. `refc-planfan_step29999_ep31_highspeed-straight_f020_worstwindow.png` — the control:
   total mode collapse at 36.5 m/s, one mode, gap 0.12 m. Shows the panel reads honestly
   when there is nothing multimodal to show.
