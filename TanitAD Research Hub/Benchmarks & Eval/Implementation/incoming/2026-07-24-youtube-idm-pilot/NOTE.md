# YouTube-IDM pilot — running NOTE (bank-as-you-go)

**Agent:** youtube-idm-pilot subagent · **Pod:** tanitad-pod3 (A40, idle) · **Started:** 2026-07-24
**Goal:** stand up a bounded, privacy-safe harvest -> pseudo-label -> pretrain -> measure
pipeline on REAL Creative-Commons YouTube dashcam video, and return the one read we cannot
get from our own labeled data: **does WM pretraining on YouTube-pseudo-labeled video lift
downstream driving vs no-YouTube pretraining?**

Evidence classes: MEASURED (ours+path) · PUBLISHED · INHERITED · ESTIMATED · HYPOTHESIS.

---

## P1 — PIPELINE SCAFFOLDING  [status: DONE, env green]

### Environment (MEASURED, pod3, 2026-07-24)
- Disk: `dd` 3 GB write OK @ 521 MB/s; `/workspace/tmp` = 83 GB used; MooseFS quota headroom
  sufficient. Pilot footprint kept < 40 GB by deleting source mp4 after decode and clip frames
  after encode (only latents+labels+pointers persist).  *(artifact: dd test output)*
- Egress WORKS: pypi/youtube/google/hf reachable; **yt-dlp extracts from pod3's datacenter IP
  with NO bot-block** (verified single-video + CC search).  *(artifact: `scripts/ytdlp_test.py` output)*
- yt-dlp 2026.7.4 + opencv-python-headless **4.11.0.86** (5.0 dropped `CascadeClassifier`; pinned 4.x)
  installed into `/workspace/venv`. PyAV + ffmpeg present. torch 2.8.0+cu128, CUDA A40.
- **CC license filter works**: YouTube search param `sp=EgIwAQ%3D%3D` surfaces CC uploads; the
  per-video `license` field == "Creative Commons Attribution license (reuse allowed)" is the gate.
- **Privacy detector ready**: opencv 4.11 Haar cascades (frontalface_default/alt2, profileface,
  russian_plate_number, license_plate_rus_16stages, full/upper body) all load.  *(`scripts/get_cascades.py`)*

### Frame/encoder contract (MEASURED from source, reused verbatim)
- Encoder input = **`[T, 9, 256, 256]` uint8** = 3 consecutive RGB frames `[t-200ms,t-100ms,t]`
  channel-stacked at 10 Hz (config `flagship4b`: in_channels=9, image_size=256, patch=16 -> 16x16
  tokens; readout 4x4x128 -> **state_dim 2048**).  *(tanitad/config.py, models/encoder.py, data/comma2k19.py)*
- Canonical geometry: `focal_crop_resize` to effective focal **F_REF=266** (comma2k19 reference).
  **YouTube intrinsics are unknown** -> pipeline assumes a nominal HFOV (default 100 deg) and crops
  to the same canonical half-angle. **This is the pilot's key geometry approximation and a NAMED
  domain-shift source** (biases apparent-motion scale -> pseudo speed). Tunable via `--hfov-deg`.
- YouTube has **no CAN/GPS** -> no real poses/actions; the IDM head pseudo-labels them. Dummy zero
  poses/actions are fine for `build_windows` because `pseudo_targets` overwrites the targets.

### Artifacts present on pod3 (MEASURED, existence-checked)
- v1 frozen encoder ckpt `/workspace/tmp/idm/ckpt.pt` (3.3 GB) — encoder+readout loaded strict.
- IDM machinery `stack/scripts/{idm_head,run_idm_proof,run_idm_ft}.py`; harnesses
  `/workspace/tmp/run_idm_{parity_validation,downstream_ablation,pipeline_derisk}.py`.
- Cached v1 latents `/workspace/tmp/branchb_eval/lat_flagshipv1/` (tr_a_/tr_b_/cm_/va_a_/va_b_).
- Parity caches `physicalai-train-e438721ae894`, `physicalai-val-f1b378f295ae`, comma2k19-val.

### Scripts written (durable deliverable — the point of P1)
| script | role |
|---|---|
| `yt_pilot_common.py` | CC verify · face/plate/body Haar blur (full-res, pre-downscale) · fps-agnostic 10 Hz resample · canonical focal crop (reuses `tanitad.data.calib`+`comma2k19`) · shot-cut score · pointer records |
| `harvest.py` | CC search + seed list -> per-video CC gate + reject time-manipulated/duration -> download <=480p -> decode+anonymize+crop -> DELETE mp4 -> segment -> shot-cut filter -> 3-frame 9ch stack -> save clip + pointer |
| `pseudo_label.py` | frozen v1 encode -> latents (durable) · build multi-domain labeler {parity rigA+rigB+comma} · pseudo ego-motion (speed+traj primary, yaw caveat, accel dropped) · distributional speed sanity · delete frames |
| `run_youtube_pilot_downstream.py` | **P4 decision read** — mirrors `run_idm_parity_validation.py`, swaps pretrain corpus D=parity->YouTube; FLOOR vs PSEUDO_YT on identical parity-val ft15/test65 split; clip-cluster bootstrap CI on speed_r2 gap |
| `env_probe.py`, `ytdlp_test.py`, `get_cascades.py` | env/egress/privacy-tool verification |
| `queries.txt`, `seed_urls.txt` | CC search queries + hand-verified seed ids |

---

## P4 PRE-REGISTRATION (committed BEFORE running — both outcomes)
- **Metric:** downstream parity-val **test speed_r2** (primary) + traj ade + yaw_r2 (caveat),
  after {FLOOR = no-YouTube pretrain, finetune-from-random} vs {PSEUDO_YT = YouTube-pilot pseudo
  pretrain -> finetune}, identical finetune protocol/split to `run_idm_parity_validation.py`.
- **WIN:** PSEUDO_YT beats FLOOR on speed_r2 for ALL seeds AND the clip-cluster bootstrap 95% CI
  of the (PSEUDO_YT - FLOOR) speed_r2 gap excludes 0 -> YouTube domain transfers -> **full harvest
  justified** (directional).
- **BOUND:** no CI-separated lift -> name the gap (domain shift / yield too low / label noise
  dominates) -> **full harvest NOT yet justified**.
- **Standing caveat:** a 2-3 seed pilot is a DIRECTIONAL read, explicitly not decision-grade for
  the full multi-thousand-hour commitment.

## Known yield risk (MEASURED, surfaced in the CC search test)
CC-licensed forward-dashcam is scarce and skewed to **time-lapsed ("5x Fast Forwarding")** and
**fails/idiots compilations** (scene cuts). harvest.py rejects time-manipulated titles and drops
cut-spanning clips (shot-cut filter). If clean continuous yield is too low to pretrain, that is
itself a reportable BOUND cause — escalated, not silently run on 5 clips.

## Smoke validation (MEASURED, 2 clips, pod3 2026-07-24) — all 3 scripts proven end-to-end
- **harvest.py**: seed video EOuxSmdPHPU -> download 2 MB @480p -> anonymize (6 faces/290
  plates blurred; plate Haar over-fires on rectangular road features = privacy-safe over-blur)
  -> 1 clip. Correctly REJECTED 3x "5x Fast Forwarding" (time-manipulated), 1 duration, handled
  1x HTTP-403 gracefully; shot-cut filter dropped a spliced compilation clip (0 clips from it).
  Yield confirmed skewed to SHORT videos (~1 clip/video).
- **pseudo_label.py**: encoder loaded (87M enc + 0.1M readout, state_dim 2048, ckpt step 29999);
  labeler built on 16,063 parity windows; 2 YT clips -> 124 windows. **Speed sanity: mean 10.58
  m/s (~38 km/h), std 5.18, range 2.0-27.5 m/s, 100% in plausible 0-45 band, not collapsed** ->
  the labeler transfers DISTRIBUTIONALLY to real YouTube. *(bug fixed: @no_grad on main() had
  killed the labeler's training grad — removed.)*
- **run_youtube_pilot_downstream.py**: FLOOR seed-0 speed_r2 = **-0.754**, exactly reproducing
  the parity-validation reference floor (-0.7540) -> identical split/recipe confirmed. Even the
  2-clip YT corpus lifted PSEUDO_YT to -0.137 (gap +0.634, clip-bootstrap 95% CI [+0.48,+0.83]
  excludes 0). Harness + bootstrap CI validated.

## Same-domain reference (MEASURED, PARITY, 4 seeds — `/workspace/tmp/idm_parity/results_idm_parity_validation.json`)
Identical parity-val ft15/test65 split, ckpt md5 b5f07d9e...585 (= IDM encoder):
| arm | speed_r2 (mean±std) | yaw_r2 | ade_2s |
|---|---|---|---|
| FLOOR (no pretrain) | **-0.4387 ± 0.224** | 0.5505 | 12.61 |
| PSEUDO (parity pseudo-pretrain) | **0.7508 ± 0.044** | 0.6911 | 4.81 |
| CEILING (parity real-pretrain) | 0.6507 ± 0.060 | 0.7485 | 5.34 |

Same-domain pseudo captures **109% of the real-label ceiling** on speed. The pilot tests whether
the YouTube domain (unknown intrinsics, no CAN GT, compilation noise) captures a comparable
fraction. `fraction_of_ceiling_youtube = (PSEUDO_YT - FLOOR)/(CEILING - FLOOR)`, CEILING cited = 0.6507.

## P2 — BOUNDED HARVEST  [status: DONE, MEASURED]
Completed cleanly (78-clip target hit, process exited — did NOT die).
`/workspace/tmp/yt_pilot/{manifest.json, pointers.jsonl, harvest_state.json}`.
- **Yield: 80 clips (~27 min @10 Hz) from 31 producing videos / 63 tried.** ~10 CC search
  queries -> ~339 unique candidates discovered.
- Rejects: **15 duration**, **3 time-manipulated** (5x/timelapse), **2 dl-fail** (HTTP 403),
  **0 non-CC** (the CC gate saw only CC uploads — search filter + per-video license check clean),
  0 decode-fail. ~12 videos passed all gates but yielded **0 clips** (every candidate clip dropped
  by the shot-cut filter — the "fails/compilation" content).
- **CC-dashcam scarcity is REAL (datapoint):** from ~339 CC candidates, only ~31 videos produced
  usable continuous forward-dashcam clips; the CC pool is dominated by time-lapsed and
  compilation/fails uploads. A full harvest needs either a much larger CC candidate pool or the
  non-CC tiers (a separate Sayed+legal-gated decision — see the 2026-07-22 LICENSING_TIER_ANALYSIS).
- Footprint 11 GB during harvest (clip frames), -> ~64 MB latents after P3 deletes frames. Under 40 GB.
- **80 clips is SMALL/DIRECTIONAL** (parity-validation used ~300); P4 is a directional signal, not
  decision-grade for the full commitment.

## Related groundwork (reference, NOT modified)
`repo:TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-22-youtube-idm-pipeline/`
established the geometry canonicalization (f_eff≈266) this pilot reuses and the **"ship pointers,
never bytes" (OpenDV)** licensing model this pilot's privacy design implements. It pre-identified a
**per-video intrinsics estimator (GeoCalib)** as the follow-up module — which is exactly the fix if
the pilot is BOUND on the fixed-HFOV geometry approximation.

## P3 — PSEUDO-LABEL  [status: DONE, MEASURED]
`/workspace/tmp/yt_pilot/results/pseudo_labels.json`; 80 latents `latents/yt_*.pt`.
- Encoded all 80 clips with frozen v1 (state_dim 2048) -> **8,960 windows**; frames DELETED after
  encode (footprint 11 GB -> **90 MB**; only latents+labels+pointers persist — privacy honored).
- **Speed distributional sanity (GT-free, MEASURED):** mean **8.51 m/s (~31 km/h)**, std 5.55,
  range -2.63 .. 29.39 m/s, **100% in the plausible 0-45 band**, not collapsed to a constant.
  -> the multi-domain IDM labeler transfers DISTRIBUTIONALLY to the full YouTube set.
- Speedometer spot-check: the auto-harvested CC set carried no readable speed overlay we could
  parse GT-free (mechanism provided in `spot_check_speed.py` for any clip with a stated speed).
  The MEASURED comma2k19 cross-domain speed R2 0.62-0.66 remains the closest direct-accuracy proxy.

## P4 — DOWNSTREAM DECISION READ  [status: DONE, MEASURED]  →  **VERDICT: WIN (directional)**
`/workspace/tmp/yt_pilot/results/results_youtube_pilot_downstream.json` (copied to `pod_artifacts/`).
3 seeds · pt_epochs 25 · ft_epochs 60 · pretrain corpus = **80 YouTube clips / 8,960 windows** ·
downstream = parity-val both rigs, finetune 15 / test 65 (5,710 windows) — the SAME split as
`run_idm_parity_validation.py` (parity firewall: own IDM split, no WM arm, no canonical re-selection).

**Arms on parity-val test (mean ± std, 3 seeds):**
| arm | speed_r2 (primary) | yaw_r2 (caveat) | ade_2s (m) |
|---|---|---|---|
| FLOOR (no-YouTube pretrain) | **−0.520 ± 0.201** | 0.546 | 12.82 |
| **PSEUDO_YT (YouTube-pilot pseudo-pretrain)** | **+0.563 ± 0.047** | 0.748 | 6.31 |

- **pseudo_yt beats floor on speed_r2 for ALL 3 seeds**, and the **clip-cluster bootstrap 95% CI of
  the (pseudo_yt − floor) speed_r2 gap EXCLUDES 0 for every seed** (gaps +1.37/+0.88/+1.05;
  CIs [+1.07,+1.73]/[+0.65,+1.17]/[+0.82,+1.31]; frac_boot>0 = 1.00 all). → the pre-registered
  **WIN** condition is met.
- Lift is across ALL channels: speed −0.52→+0.56, yaw 0.55→0.75, trajectory ADE 12.8→6.3 m (halved).
- Per-seed FLOOR reproduces the parity reference bit-for-bit (−0.754/−0.264/−0.542) → identical
  split/recipe confirmed; my harness's FLOOR is the same object as parity-validation's FLOOR.

**Fraction of the real-label ceiling** (MEASURED here + cited parity reference, identical split):
using parity CEILING (real-label pretrain) = 0.6507 and the common parity FLOOR = −0.4387:
`(0.563 − (−0.439)) / (0.651 − (−0.439)) = 1.002 / 1.089 ≈` **0.92**. YouTube-pilot pseudo-pretraining
captures **~92% of the real-parity-label ceiling** on parity-val speed_r2, and **~84% of the
same-domain (parity) pseudo benefit** (parity PSEUDO = 0.7508). *(EVIDENCE: mine MEASURED
`results_youtube_pilot_downstream.json`; parity ref MEASURED `results_idm_parity_validation.json`.)*

### What this means
The YouTube domain — **unknown intrinsics (fixed-HFOV approx), no CAN/GPS ground truth,
compilation noise, and only 80 clips** — **TRANSFERS**: pretraining the small WM on YouTube
pseudo-labels lifts parity-val driving from a broken negative floor to near the real-label ceiling,
CI-separated on every seed. This is the one read we could not get from our own labeled data, and it
points GO for the larger harvest.

### Honest caveats (do not overclaim)
1. **DIRECTIONAL, not decision-grade.** 80 clips (~27 min) vs parity-validation's ~300; 3 seeds.
   A directional signal for the full multi-thousand-hour commitment, not a final green light.
2. **The FLOOR is negative/unstable** (finetune 15 clips from random) so part of the raw gap is
   "any competent pretraining rescues a broken low-data floor." The substantive, non-trivial claim
   is the **fraction-of-ceiling ≈ 0.92** — YouTube pretraining is nearly as good as REAL-parity-label
   pretraining for this readout; that is not explained by "any pretraining."
3. **speed + trajectory are the trustworthy channels** (MEASURED zero-shot cross-domain R2 0.60–0.66);
   yaw's downstream lift here rides on the 15-clip real finetune, not zero-shot yaw quality.
4. **Geometry is approximated** (fixed HFOV 100°). Transfer held despite it; a per-video intrinsics
   estimator (GeoCalib — the 2026-07-22 groundwork's named follow-up) would only tighten it.

---

## DELIVERABLE MANIFEST
| artifact | repo path (staged) | pod3 path |
|---|---|---|
| harvest/pseudo/downstream scripts + common + spot-check + probes | `repo:.../incoming/2026-07-24-youtube-idm-pilot/*.py` | `/workspace/tmp/yt_pilot/scripts/*.py` |
| CC search queries + seed ids | `repo:.../*.txt` | `/workspace/tmp/yt_pilot/scripts/*.txt` |
| README + this NOTE | `repo:.../README.md`, `repo:.../NOTE.md` | — |
| harvest manifest + 80 URL/timestamp pointers + state | `repo:.../pod_artifacts/{manifest,pointers.jsonl,harvest_state}` | `/workspace/tmp/yt_pilot/{manifest.json,pointers.jsonl,harvest_state.json}` |
| pseudo-label results (speed sanity + per-clip) | `repo:.../pod_artifacts/pseudo_labels.json` | `/workspace/tmp/yt_pilot/results/pseudo_labels.json` |
| **P4 downstream verdict JSON** | `repo:.../pod_artifacts/results_youtube_pilot_downstream.json` | `/workspace/tmp/yt_pilot/results/results_youtube_pilot_downstream.json` |
| 80 pseudo-label latents (non-imagery) | — (not staged; 90 MB, reproducible) | `/workspace/tmp/yt_pilot/latents/yt_*.pt` |
| opencv Haar cascades (privacy) | — | `/workspace/tmp/yt_pilot/cascades/` |

Nothing of value lives ONLY on the pod: all scripts, provenance, and result JSONs are staged in the
repo working tree. The latents (90 MB) and cascades are reproducible from the staged scripts + pointers.

## ESCALATIONS
1. **Pilot WINS (directional).** The larger CC harvest is justified as the next step — but CC-dashcam
   yield is scarce (80 clips from ~339 CC candidates / 32 usable videos). Scaling needs either a much
   larger CC candidate pool or the **non-CC tiers**, which is a **Sayed + legal-gated decision** (see
   `Data Engineering/.../2026-07-22-youtube-idm-pipeline/LICENSING_TIER_ANALYSIS.md`), not this agent's.
2. **Decision-grade confirmation** would want ~300+ clips and 4+ seeds (mirror parity-validation), plus
   the **per-video intrinsics estimator** to remove the fixed-HFOV approximation.
3. **Intake:** this `incoming/` folder should be intaken into the hub proper alongside the 2026-07-22
   pipeline groundwork it executes.

