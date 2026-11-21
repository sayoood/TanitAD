# RESULT — refcv6's semantic-map head at step 35,000, rendered for examination

EvalFlyWheel · 2026-09-26 · answers the PI's request: *"Can we visualize the sematic map output of the
last checkpoint and render a video to examine the results"*.

## Headline

**The video** is `D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-refcv6-map-video/raw/refcv6_map_step35000.mp4`.

* It is 20,551,884 B (19.60 MiB, under the 30 MiB delivery limit), 588 frames, 58.8 s at 10 fps,
  1920 × 1200, H.264 High yuv420p.
  * This is the **caption-corrected re-encode** of 2026-09-27 01:11 Berlin (see below).
  * The first delivery was 20,613,316 B and carried the false trunk caption. Only the caption box
    differs between the two.
* It was verified by decoding every frame back: 588 frames, clean decode, container duration equal to
  n/fps (`verify_mp4.py`).
* `*.mp4` is gitignored, so **this file exists on the D: working tree only**.

**Drivable IoU per clip.** Trainer rule `refc_v3_train.py:4506-4521`, computed on every eval window of
each clip:

| clip (sha12) | v7 nav | n windows | **mean IoU** | pooled ΣI/ΣU | min / median / max | IoU by time-thirds |
|---|---|---|---|---|---|---|
| `693335b810c3` | left | 171 | **0.703** | 0.703 | 0.407 / 0.777 / 0.832 | 0.811 → 0.773 → **0.526** |
| `0191487845ef` | right | 171 | **0.622** | 0.609 | 0.436 / 0.644 / 0.762 | 0.676 → **0.515** → 0.674 |
| `00213e99adca` | follow | 171 | **0.727** | 0.724 | 0.625 / 0.729 / 0.820 | 0.735 → 0.702 → 0.746 |
| all 513 windows | | 513 | 0.684 | 0.677 | | |

**The in-run reference.** The run's own `eval_map_iou_drivable` at step 35,000 reads **0.67745**
(`metrics_20260926T1500.jsonl`, md5 `ff30a57f…`). That is 8 batches of 16 windows drawn from a
seeded random subset of the 23,772 eval windows. It is a **different window set**, so it is a sanity
range and not an identity. Our three clips fall on either side of it, and the pooled number over all
513 windows (0.677) sits on it.

⚠️ **One clip is one episode.** Its 171 windows are strongly correlated and are not independent
samples, so **no interval is quoted**. Three clips cannot rank the head against anything.

**Evidence class:** MEASURED (ours). Artifacts: `raw/per_frame.jsonl`, `raw/per_clip.json`,
`raw/render_record.json`, and the logs in `raw/logs/`.

**Tier:** an open-loop perception readout. It carries **no driving claim**. The trajectory drawn is
one DDIM draw of a one-shot planner conditioned on an oracle nav token.

⛔ **Correction, same evening: the video's camera caption is false.** It reads *"bottom 43 rows are
ZEROED inside the trunk (--equalize-bottom-rows 43)"*. The trainer's pin sets
`trunk_equalize_bottom_rows` as an undeclared attribute (`refc_v3_train.py:381`), and the
`--image-hw` rebuild `dataclasses.replace` at `:459` drops it. So the trunk zeroes **nothing**
(D-REFCV6-EQUALIZE-DROPPED); only the BEV lift treats those rows as unobserved.

The loader replays the same pin, so **the rendered model is the trained one, and every number here
stands.** Only the caption's claim about the trunk is wrong.

**The caption is corrected (2026-09-27 01:11 Berlin).** `code/fix_equalize_caption.py` rewrote the
caption box in each saved frame. It checks every frame for pixels changed OUTSIDE the box before saving
that frame.
- **513 / 513 frames patched, with 0 pixels changed outside the box** (box xyxy [220, 459, 1038, 475]).
- It then re-encoded with the renderer's own ffmpeg arguments, and `verify_mp4.py` decoded the result
  back: **588 frames, clean, 58.80 s**.
- The temp file was swapped in only after that verification.
- Record: `raw/caption_fix_record.json`. Log: `raw/logs/caption_fix.log`.
- Run gated under the Master Mind's light-job rule: 5.5 GB start floor, 4.0 GB abort watchdog.
The step-38,000 boxes video reads the flag off the built trunk (`render_record.json` →
`equalize_bottom_rows`).

## 1. What the video shows

Each clip opens with a title card of 2.5 s. The card gives the clip, its nav token, the clip's IoU
statistics, the controls' results, and a "how to read a frame" guide. Every eval window of the clip
then follows, in time order. The layout is specified in
`taniteval/tools/RENDER_REFCV6_MAP_VIDEO.md` §3.

* **Front camera.** The last input frame, with the GT future path (green) and refcv6's **selected**
  `out["traj"]` (orange). They are projected with the clip's measured extrinsics, and a horizon line
  is drawn as a self-check.
* **Three BEV panels.**
  * **Predicted map:** the argmax of the softmax.
  * **SAM3 GT map:** the argmax of `map_frac`.
  * **Drivable agreement:** TP in blue, FP in pink, FN in yellow, TN in grey.

  In all three, unscored cells are hatched. Diagonal lines mean SAM3 never saw the cell. Dots mean
  SAM3 saw it, but the lift's camera cannot reach it at this instant.
* **HUD.** It shows:
  * the run, the step, and the hybrid note;
  * the caveat that this is a perception diagnostic, open loop, and not driving performance;
  * the clip sha12, t on the label clock, the measured v0, and the nav token (a GIVEN INPUT);
  * the frame IoU, the clip mean and the pooled IoU;
  * a per-frame IoU sparkline against the 0.677 in-run line;
  * the 9-class palette and a per-class table.
* **Contact sheet.** `raw/contact_sheet.png` holds 12 keyframes: the 0.1, 0.37, 0.63 and 0.9
  quantiles of each clip's windows.

**What the frames show.** These are descriptive observations. Each is MEASURED on these 3 clips
only.

1. **The head effectively predicts three of nine classes: seen-no-class, drivable, and
   sidewalk/verge.** The argmax IoU, averaged over windows where each class has a union, reads:
   * lane / road line: 0.000 (left), 0.000 (right), 0.022 (follow);
   * crosswalk: 0.000 and 0.001;
   * arrow/text and non-drivable edge: 0.000 wherever they occur.

   SAM3 labels all of these classes on these clips. The GT panel shows crosswalk stripes and lane
   lines, and the predicted panel does not.
2. **Drivable IoU falls at intersections.**
   * The **left** clip drops from 0.81 to **0.53** in its last third, at t ≈ 12.6–18.2 s. Its worst
     window is 0.407, at t = 16.81 s. At t = 16.51 s (window 154, IoU 0.417) the street ends in a
     **T-junction about 30 m ahead**: a wall with a crosswalk in front of it and a left-turn sign.
     * SAM3's drivable area stops at about 33 m and spreads sideways into the junction.
     * The head instead **extends drivable road straight through the wall to about 55 m**, and misses
       the junction's lateral arms. This gives FP = 734 and FN = 183 cells, against TP = 656.

     Read as a straight-road prior, this is a **hypothesis from one frame**, not a measured property
     of the head.
   * The **right** clip's middle third reads **0.515**, at t ≈ 6.7–12.4 s and v0 ≥ 4.4 m/s. This is
     the approach to and entry into the right turn. The GT path's largest rightward excursion,
     6.05 m, is at t = 12.16 s, and the worst window is 0.436, at t = 9.04 s. The clip starts almost
     stationary: v0 is 0.16 m/s at window 1.
3. **The error mode depends on the scene.**
   * On the right-turn clip, **77 % of the errors are FN**: SAM3-drivable cells the head misses. The
     model calls 31.0 % of scored cells drivable against SAM3's 40.0 %.
   * On the left clip and on the night-highway follow clip, the errors are **65 % and 62 % FP**. The
     model calls 20.3 % drivable against 18.1 % on the left clip, and 45.5 % against 42.1 % on the
     follow clip.
4. **The selected path is context, not a measured claim.** On the right clip at t = 12.16 s (window
   112), the GT path ends 6.1 m to the right while refcv6's selected path ends 0.1 m left of centre.
   The planner goes straight while the GT turns right under nav = RIGHT. This is one DDIM draw
   (seed 0), and the draw moves the path by up to 7.6 m on the follow clip (§3c). The battery, not
   this reel, owns any trajectory claim.

## 2. What ran (provenance)

| item | value |
|---|---|
| checkpoint | `D:/refcv6_eval_kit/ckpt/ckpt_35000.pt`, 1,177,897,033 B. md5 `68a4ef3bbb0a3ec707b04f0fb2b6b86f` was re-hashed in-process (and before with `md5sum`) and equals `ckpt/MD5SUMS`. `ck["step"]` = 35000. The strict load had 0 missing / 0 unexpected keys over 1,101. `param_breakdown` equals the config. The anchor file equals the checkpoint's anchor buffers with max \|Δ\| 0.0. |
| config | `C:/Users/Admin/ev6_battery/raw/thor_reads/config_resume34500_20260926.json`, md5 `a3193a4685ce0d07a6ae6b89fdc994a8`. This equals `RESUME_CFG_MD5` in `chain_final_v2.sh`, the post-switch config of `gate_dryrun_35k.sh`. |
| code tree | `C:/Users/Admin/ev6_82c2331` (`REFCV6_REPO`). `tanitad` resolved to `C:\Users\Admin\ev6_82c2331\stack\tanitad\__init__.py`, and this was asserted. `refc_v3_train.py` md5 is `ba58e746…`. |
| model / dataset | `C:/Users/Admin/ev6_battery/code/refcv6_loader.py` (md5 `6b7d07ac…`): `build_model` and `build_eval_dataset(..., with_perception_targets=True)` |
| label clock | sidecar `D:/refcv6_eval_kit/data/refcv6_clip_clock_sidecar.jsonl`, used for 3/3 clips, with dt 0.1005–0.1007 s |
| map GT | `D:/refcv6_eval_kit/data/sam3_gt_eval_thor137` (flat sha12 layout). All 513/513 windows are `ok`. |
| extrinsics | `D:/refcv6_eval_kit/data/refcv6_train_eval139_extrinsics.json` (md5 `3bd70d98…`), found for 3/3 clips. `CylProjector.R` equals `pai_extrinsics_table.quat_to_R` with max \|Δ\| 0.0. |
| device | the RTX 4060, behind the dev-box gate. At start the gate read 1,395 MiB used, no other python on the GPU, and 13.6 GB free RAM. Peak CUDA allocation was 1.105 GiB. Wall time was 222.9 s. Host RAM stayed ≥ 10.71 GB. |

**Departures.** All are recorded in `render_record.json`:

* Providers were filtered to the 3 clips (the `Q3_ONLY_EPISODE_SHA12` pattern).
* The agent and 3-D joins were not attached. They are LOSS-only targets, and `agents.oracle` was
  asserted False.
* Future frames were not decoded.
* The forward is the trainer's own `compute_losses_v3`, stopped by a forward hook once `model(...)`
  returns.
* The DDIM draw is seeded per window with `manual_seed(0)`.
* The checkpoint is read through `mmap`.
* The loader's own departures apply: `--trunk-compile` is dropped, and the startup gradient probe is
  not run.

## 3. The clip rule and what it chose

The rule was written into `taniteval/tools/RENDER_REFCV6_MAP_VIDEO.md` §2 **before** rendering. The
selection is in `raw/clip_selection.json`, and the smoke run and the render run chose identically.

* **Coverage.** 137 of 139 eval clips have full SAM3 coverage. The 2 clips without full coverage
  (`081b986f8888` right, `2aa810802777` left) have **no GT file**, so 171/171 windows each are
  `no_file`. No clip has partial coverage.
* **Candidates by nav.** Left has 12 candidates (of 13), right 37 (of 38), follow 88 (of 88).
* **Chosen, as the smallest sha12 per token:** `693335b810c3` (left), `0191487845ef` (right),
  `00213e99adca` (follow), with 171 windows each.
* ⚠️ **The "left" clip does not turn left within the 6 s horizon of any window.** Its GT slots never
  leave ±0.83 m laterally, and the largest leftward excursion is 0.56 m. The v7 nav token is a
  **per-clip constant**, and the left turn lies at or beyond the clip's end.

  By contrast, the **follow** clip's GT path reaches 7.38 m to the LEFT. It runs at 26–27 m/s, where
  6 s covers about 160 m, so this is a gentle highway curve and not a turn.

## 4. Controls

**(a) Known-value control: the GT fed as logits through the same prediction path.**

| check | left | right | follow |
|---|---|---|---|
| threshold-consistent GT-as-logits → drivable IoU **== 1.0** exactly | **171/171** | **171/171** | **171/171** |
| its predicted panel == GT panel, pixel for pixel, overlays included | **171/171** | not run (clip 1 only, per brief) | not run |
| literal one-hot(argmax) panel == GT panel | 171/171 | not run | not run |
| literal one-hot(argmax) IoU: mean (min) | 0.9955 (0.9866) | 0.9906 (0.9792) | 0.9923 (0.9785) |
| "mixed" cells over the clip (argmax drivable, fraction < 0.5; all cells) | 1,033 | 4,613 | 4,131 |
| cells no logits can serve both rules on | 0 | 0 | 0 |
| **mutation: lateral mirror** must go RED (IoU ≠ 1) | 171/171 red (max 0.746) | 171/171 (max 0.618) | 171/171 (max 0.834) |
| **mutation: prediction channels off by one** must go RED | 171/171 red | 171/171 | 171/171 |
| scored-cell count == `map_loss_row` `n_map_cells` (the trainer's loss cell set) | 171/171 | 171/171 | 171/171 |

⚠️ **The brief's literal control does not read 1.0, and this is a property of the GT, not a
defect.** A plain one-hot of `argmax(map_frac)` gives p(drivable) ≈ 1 on "mixed" cells, where
drivable is the argmax but its fraction is below 0.5. The trainer's GT threshold calls those cells
not drivable. That is the "two rules" gap the trainer's own comment names
(`refc_v3_train.py:4501-4503`).

The threshold-consistent variant keeps the one-hot everywhere except those cells. On them the
drivable logit leads the other eight by exactly 1, so p = e/(e+8) = 0.2536. That variant reads
exactly 1.0 and redraws the GT panel pixel for pixel.

The same gap applies to the video. The class panels (argmax) show slightly more drivable area than
the IoU's GT (threshold) counts: 0.09–0.38 % of scored cells per clip.

**(b) Independent check against the run's in-run eval.** The per-clip means are 0.703, 0.622 and
0.727. The in-run `eval_map_iou_drivable` at step 35,000 is **0.67745**, over 128 other windows. All
three sit within ±0.06 of it, and the all-window pooled value is 0.677. This is a sanity range, not
an identity.

**(c) Draw independence, on the first window of each clip.** The forward was repeated with seed 1.
* `map_logits` max |Δ| = **0.0** on all three clips, on the GPU as well as on the CPU smoke.
* The drivable mask and the argmax are identical, so the map panels do not depend on the DDIM draw.
* The **selected path moved 0.35 m, 1.43 m and 7.60 m**. The orange path is one draw.

**(d) Orientation, derived from data independently of the renderer**
(`code/orientation_control.py` → `raw/orientation_control.json`). The ego's own GT future slots
(x ≥ 4 m, on cells SAM3 saw) are placed on the SAM3 drivable fraction. The file contract puts
col 0 on the RIGHT.

| clip | on drivable, file contract | on drivable, lateral mirror | lateral-path windows (\|y\| ≥ 1 m): contract vs mirror |
|---|---|---|---|
| right | **0.969** (n = 1,053 points) | 0.889 | **0.994 vs 0.834** (75 windows) |
| follow | **1.000** (n = 684) | 0.964 | **1.000 vs 0.954** (135 windows) |
| left | **0.995** (n = 1,101) | 0.964 | no lateral windows |

The contract beats the mirror on every clip. The drawing code is tied to that same contract by the
unit test: cell (0, 0) lands at the bottom-right block, and the ego is at the bottom centre.

⚠️ A ±2 s shift of the label frame is **not** discriminated by this probe, because the path stays on
the road either way. The frame axis rests on the reader's own identity and time checks.

**(e) CPU versus GPU on the same two windows.** The CPU smoke read 0.776848 and 0.767698. The GPU
render read 0.777003 and 0.767857. Exactly **one cell** differs in each of the intersection and the
union, from device arithmetic at the 0.5 threshold. The scored-cell counts are identical. The
selected path differs by 1.1 m, because the CPU and CUDA generators produce different DDIM noise for
the same seed.

**(f) Unit test.** `taniteval/tests/test_render_refcv6_map_video.py` has 8 tests, all passing. Its
expected values are literals: IoU 0.8 on the synthetic 4 × 4 grid, and 4/6 without the valid mask.
It tests that the known-value control reads 1.0 and that the literal one-hot reads 0.5 there.

The **test-power check** is `code/unit_test_mutation_check.py`, logged in
`raw/logs/unit_test_mutation_check.log`. Each defect it reintroduces turns tests RED:
* a strict `>` threshold: 2 tests;
* an argmax prediction: 4 tests;
* a missing orientation flip: 1 test.

## 5. What this does NOT show

* **Driving performance, and any closed-loop behaviour.** This is open loop, one forward per window.
* **A four-family result.** The families are LONGITUDINAL, LATERAL, TACTICAL and STRATEGIC, and **none
  is computed here, by design**: this artifact is a perception diagnostic of the map head. The
  four-family eval of this checkpoint belongs to the EvalFlyWheel battery. Its post-switch gate
  (`gate_dryrun_35k.sh`) is queued after A6.
* **A representative estimate of the head's quality.** There are 3 clips chosen by a fixed rule, not
  a sample. The windows within a clip are correlated, and no interval is quoted.
* **A trajectory claim.** The orange path is one DDIM draw under an **oracle** nav token (provenance
  ego-future). No ADE or other trajectory number is reported.
* **The unobserved cells.** Hatched cells are not scored. Near the ego, under the hood and outside the
  ±60° field of view, the model's argmax is visible only through the hatch.
* **Map classes other than drivable, scored under the trainer's rule.** Per-class numbers use the
  programme's argmax `map_metrics` and are reported as companions only.

## 6. Deliverables (landing list in `LANDING_READY.txt`)

* **Renderer:** `taniteval/tools/render_refcv6_map_video.py`, with its README
  `taniteval/tools/RENDER_REFCV6_MAP_VIDEO.md`.
* **Unit test:** `taniteval/tests/test_render_refcv6_map_video.py`.
* **This package:** `RESULT.md`, `LANDING_READY.txt`, `code/orientation_control.py`,
  `code/unit_test_mutation_check.py`.
* **Numbers:** `raw/per_frame.jsonl` (513 rows), `raw/per_clip.json`, `raw/render_record.json`,
  `raw/clip_selection.json`, `raw/orientation_control.json`.
* **Picture and logs:** `raw/contact_sheet.png` and `raw/logs/*`.
* **Smoke:** `raw/smoke/*`, except its mp4.
* **Not landable (gitignored `*.mp4`):**
  * `raw/refcv6_map_step35000.mp4`, 20,551,884 B, the caption-corrected re-encode;
  * `raw/smoke/refcv6_map_step35000_smoke.mp4`, 424,669 B.

  Both live on D: only. The rendered PNG frames live on C: only, in
  `C:/Users/Admin/qland/work/mapvid/frames_refcv6_map_step35000/` (513 windows and 3 title cards).
