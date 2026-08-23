# v1arch open-loop overlays on PhysicalAI's own OOD-val — 2026-08-05

**Arm:** `flagship-v1arch-v2bal-30k`, step 29999 — the **v1 ARCHITECTURE** trained on the v2bal
9000-clip pool (every `v2_*` lever in its config is `false`). Architecture held constant, data
varied: that is what makes the PI's question about *more and better-distributed data* attributable.

**Corpus:** `physicalai-oodval-6f4b94e4c7ce-q90` — PhysicalAI-AV's **own official eval split**
(290 clips), **zero overlap** with the arm's training pool, JPEG-q90 round-tripped to match the
format the arm trained on.

---

## ⛔ What these videos ARE — read this before quoting a frame

| | |
|---|---|
| **OPEN LOOP** | the ego follows the **LOGGED** trajectory. Every frame is the rig's real image at the real pose. The model **never drives**, so there is no control drift and no divergence. |
| **WORLD-MODEL FIDELITY, not driving** | the rollout decodes the **expert's TRUE FUTURE ACTIONS**. `rollout.collect`'s own PC2 record on this surface says `actions_source="expert_future"` and `pc2_pass = False` **by construction**. |
| **NOT a hierarchy result** | `rollout_decode` takes no `intent`/`ctx`/`nav`. The HUD *displays* the decoded tactical manoeuvre and strategic route, but the trajectory does not flow through them. |

Both facts are **burned into the banner of every frame**, so a clip cut out of a reel keeps its
caveat. A README beside an mp4 does not travel with the mp4.

⚠️ **An orange line tracking a green line is the most over-claimable artefact this programme
produces.** It looks like the car driving. It is not.

---

## The three reels

| file | selection | episodes | frames | duration | mean ADE over rendered eps |
|---|---|---|---|---|---|
| [`v1arch_oodval_openloop_representative.mp4`](v1arch_oodval_openloop_representative.mp4) | **`spread`** — evenly spaced across the 290-clip corpus | 30 | 5,128 | **8 min 33 s** | **0.6284 m** |
| [`v1arch_oodval_openloop_worst12.mp4`](v1arch_oodval_openloop_worst12.mp4) | `explicit:worst-12-by-ADE-from-banked-eval` | 12 | 2,050 | 3 min 25 s | 1.4626 m |
| [`v1arch_oodval_openloop_best12.mp4`](v1arch_oodval_openloop_best12.mp4) | `explicit:best-12-by-ADE-CHERRY-PICKED` | 12 | 2,059 | 3 min 26 s | 0.1701 m |

All **962 × 684 @ 10 fps** (68.4 MB / 19.7 MB / 23.7 MB). Per-episode ADE ranges: representative **0.215 – 1.443 m**,
worst12 **1.336 – 1.630 m**, best12 **0.138 – 0.197 m**.

⛔ ⚠️ **`.gitignore` bans `*.mp4`** to stop raw dataset video being committed; these three are curated evaluation artifacts and were added with `git add -f`, the same convention as the 16 mp4s already tracked under `Videos/`.

⛔ **The selection mode is in the banner of every frame**, including the literal word
`CHERRY-PICKED` on the best-12 reel, so a hand-picked clip can never be quoted as a representative
one. The corpus-wide mean is **0.5752 m** and the median episode is **0.4702 m**
(`raw/episode_ade.json` has all 290).

⚠️ The `spread` reel's 0.6284 m is **above** the corpus mean 0.5752 m — 30 evenly-spaced episodes
are a sample, not the corpus. Quote the corpus number for the corpus.

Every file was verified by **decoding it back** (`raw/video_verification.json`: `decode_rc 0`, zero
decode errors on all three) and md5-matched between pod4 and this repo.


### ⛔ THE WORST/BEST SPLIT IS MOSTLY A SPEED SPLIT — do not read it as good-vs-bad model

MEASURED over the same 6,382 windows the reels are drawn from:

| reel | n windows | mean `v0` | median `v0` | windows stopped (< 0.5 m/s) | mean ADE |
|---|---|---|---|---|---|
| **best12** | 266 | **2.34 m/s** | 2.30 | **32.7 %** | 0.173 m |
| whole corpus | 6,382 | 7.54 m/s | 6.44 | 11.8 % | 0.575 m |
| **worst12** | 264 | **19.19 m/s** | 19.61 | **0.0 %** | 1.454 m |

**Pearson r(`v0`, per-window ADE) = 0.6408.** ADE is a displacement over a fixed 2 s horizon, so it
scales with how far the car travels: at 19 m/s the ego covers ~38 m in the window and at 2 m/s it
covers ~4 m. An identical *relative* error is then ~9× larger in metres.

⇒ **The two ranked reels are largely a 19 m/s clip next to a 2 m/s clip.** A third of the best-12
windows are the car standing still, where the prediction is close to trivial. Watching them
side by side and concluding *"the model falls apart here and nails it there"* is reading the speed
distribution, not the model. The `spread` reel is the one to judge behaviour from; the ranked reels
are useful for **seeing what a large error looks like**, not for attributing it.

⚠️ This is the same defect class as quoting a pooled distance-keeping number: a statistic averaged
over regimes that do not resemble each other. A speed-matched comparison is the fix, and it is a
WORK ITEM — not done here.

---

## The layout

![a frame from the worst-12 reel](stills/v1arch_oodval_openloop_worst12_frame136.jpg)

*Frame 136 of ep 00163 (worst-12 reel). Note the BEV: the orange 2.0 s marker sits
**beyond** the green one — the over-speed defect, visible directly. The banner carries the
full caveat on two lines; the ADE trace shows the per-frame history rather than a mean.*

1. **CAMERA (512 × 512)** — ground truth **green (wide)** underneath the prediction **orange
   (narrow)**, projected through the flat-ground pinhole (`cx = cy = 128`, `f_eff 266`,
   `cam_h 1.5`) — correct by build for a principal-point-centred 256 crop. Markers at
   **0.5 / 1.0 / 1.5 / 2.0 s**.
2. **METRIC BEV (420 px, full height)** — the same two paths in metres, top-down, ego at
   bottom-centre, plus the ego's **past 3 s** in grey. ⭐ It is **calibration-independent**, which
   is why it is a full panel here and not the usual 152 px inset: if the two panels disagree, the
   *projection* is wrong and the BEV is the one to believe.
3. **HUD** — decoded **tactical manoeuvre** and **strategic route**, per-frame ADE, clip-mean ADE,
   `v0`, and a **scrolling 24 s ADE trace** so a spike is visible as it happens instead of being
   averaged into a clip mean.

---

## What to look for — the four-family result these reels illustrate

The complete four-family block for this arm and corpus is
[`Benchmarks & Eval/…/2026-08-05-v1arch-oodval-four-families/RESULT.md`](../../../Benchmarks%20&%20Eval/Implementation/incoming/2026-08-05-v1arch-oodval-four-families/RESULT.md).
Three of its findings are visible in these reels:

* **The error is LONGITUDINAL and it has a sign.** `speed_bias +0.484 m/s`,
  `along_final_bias +0.943 m`, `ego_progress 1.0795×`. In the BEV panel the orange path's
  **2.0 s marker sits beyond the green one** — MEASURED over all 6,382 windows, it does so on
  **71.95 %** of them (median overshoot **+0.72 m**), and the predicted mean speed exceeds the
  human's on **75.51 %**. So this is not a few large outliers dragging a mean: it is **three
  windows in four**, and you should expect to see it in most frames of the reel. Lateral
  cross-track is **5.5 cm**.
* **The route head never turns.** The HUD's `strategic: route straight` is not a coincidence of the
  clip: MEASURED, the vision-only head predicts *straight* on **1,737 of 1,737** valid windows and
  never once predicts a turn — `route_acc_follow 0.8031` equals the majority-straight rate
  **0.8031** to four decimals.
* **The manoeuvre label is honest** (κ 0.6033, SUBSTANTIAL) even though **0 of 3 hierarchy seams**
  are beneficial at this checkpoint.

---

## Reproduce

```
python3 taniteval/tools/render_openloop_video.py \
  --corpus  <epcache>/physicalai-oodval-6f4b94e4c7ce-q90 \
  --ckpt    <run>/ckpt.pt --run-config <run>/config.json \
  --arm     flagship-v1arch-v2bal-30k --speed-input \
  --corpus-label PhysicalAI-OODval-official-q90 \
  --episodes 30 --select spread --out <out>.mp4
```

The rollout is `corpus_overlay.episode_rollouts` — **imported, not re-implemented** — so the video
and the scored metric cannot drift apart. Ranked reels use `--episode-list` with a
`--select-label`, and the label is burned into the banner.

## Provenance

* rendered on **pod4** (`/workspace/videos`), A40, 2026-08-05
* `raw/*.mp4.json` — per-reel sidecars: every episode index, its frame count, mean and max ADE
* `raw/video_verification.json` — md5 + decode-back result per file
* `raw/episode_ade.json` — per-episode ADE for all 290, read off the **banked windows dump** the
  four families were computed on, so the reels' ranking and the published metric cannot disagree
* `raw/render_all.log` — the full render transcript

## ⚠️ Two defects this render caught, recorded because both were silent

1. **The banner caveat was clipped off every frame of the first render.** At 962 px it stopped at
   *"NOT autonomous driving, N"* — the words *"NOT a hierarchy result"* appeared in **zero frames of
   zero reels**, and the episode title was cut mid-token (`(2/12, expli`), which reads as data
   rather than truncation. PIL draws past the canvas edge without raising, so neither the encode nor
   the decode-back verification could see it, and a downscaled still hid it. The caveat now wraps
   against measured text width and three tests pin that it fits.
2. **`--select worst|best` silently rendered `spread`.** A reel labelled *worst* that was nothing of
   the kind. Those options are gone; ranked reels go through a range-checked `--episode-list`.
