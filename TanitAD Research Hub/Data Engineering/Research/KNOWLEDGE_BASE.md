# KNOWLEDGE_BASE — Data Engineering

> **Curated, deduplicated, newest first.** Format:
> `- [YYYY-MM-DD] [source] finding (1-3 lines) — impact: H_x / WP_y — link`
>
> This is the **Data Engineering** agent's findings log. The router across all areas is
> [`../../KNOWLEDGE_BASE.md`](../../KNOWLEDGE_BASE.md).
>
> ⛔ **Three layers, and they are not interchangeable.**
> **PAPER** (`Paper/TANITAD_PAPER.md`) = the scientific account of the frontier work — derivations,
> argument, results in narrative form.
> **KNOWLEDGE_BASE** (this file) = what we learned, written for an agent about to make a decision.
> **LIBRARY** (`../../Library/`) = the evidence. Every `[PUBLISHED]` entry cites a **library key**,
> not only a URL — bank it with `python tools/kb_add.py <arxiv-id> --tag <topic> --cited-by <report>`.

- [2026-08-02] [measured/corpus] **A "clean v2 val" is NOT clean for v1 — 62 of a 600-clip draw sit in v1's
  TRAIN split** (24 in v1's val). Disjointness proofs are written against ONE corpus and are silent about every
  other arm that will be scored on the split; C64 in mirror image. ⇒ any new split excludes **every** corpus an
  arm was trained on (`load(..., exclude_paths=[...])`). Parity-free remainder = **8,298** of 9,987, so 600 clips
  is **not available** once v1 is excluded (headroom 0.95) — **n=400 is the largest fully-clean balanced split**
  (max |d| 0.0409, cell-census 69.4 %, sha256 `abe041db72a045b3…`). Bonus, no pod needed: at CLIP granularity
  **256 of v1's 600 parity-val clips (42.7 %) are inside v2corpus's training selection** (≠ C64's 21/40 EVAL
  EPISODES — different unit, never merge) — impact: C64/A3/A5/MODEL_REGISTRY comparability —
  `2026-08-02-v2-clean-val-semantics-and-selector.md` §5
- [2026-08-02] [measured/method] **Stratum headroom belongs to its AXIS SET, and cell-matching ≠ balance.**
  The same remainder gives headroom **6.77× (junction only) → 4.05× (+has_turn) → 1.07× (+speed) → 0.77 =
  INFEASIBLE (+has_brake)** at n=600 — so "6.77× ⇒ feasible" was one axis wide, exactly like quoting an exponent
  without its window. And the cell-quota design it justified leaves **max |d| 0.3997 (10/13 axes over the 0.10
  bar)**; greedy covariate balancing on the four-family axes reaches **0.0094** in 0.2 s, the shipped hybrid
  (quota+balance) **0.0532** with cell-L1 0.0047. ⚠️ Neither is exchangeable: within each matched cell the
  remainder is still skewed (median |d| **0.359**, p90 0.915) and max KS stays 0.12–0.19 vs a 0.057 critical
  value — the residue of a quota selector cannot be matched back into its parent, only mean-balanced — impact:
  every future split/selector, D1 strata, four-family reporting — same note §2–4
- [2026-08-02] [measured/schema] **`v2_pool_scored.parquet` semantics are now a machine-checked contract**
  (`pool_columns.py`, 34/34 checks PASS on 18,988 rows, corruption-tested): `lk,tl,tr,ac,bs(+v2)` are frame
  **COUNTS** out of `nlab` (⇒ `lk_rate` train **0.4500** vs remainder 0.6718, reproducing the design's
  "lane_keep 45.0 %"); `stopped/city/hw/stop_frac` are **FRACTIONS** with `stopped+city+hw ≡ 1`; `nlab ≡ 179`
  for every clip so the `lk` misread was a pure scale error. Also: the pool has **18,988 rows but 18,987 unique
  clips** — `32ad1a3a-…` is registered under chunks 1573 AND 3117 and is in the v2 selection (both figures were
  in circulation, neither labelled) — impact: A3 unblocked, any consumer of the pool — same note §1
- [2026-08-02] [measured/absence] **PhysicalAI-AV has NO session/drive id and NO absolute clock** — probed at
  four locations: `clip_index` (3 cols), `data_collection` (5 cols), `feature_presence` (36 presence flags), and
  `egomotion.timestamp` itself, which starts at −0.2 ms and spans ~137 s ⇒ **clip-local microseconds**. ⇒ the
  L2D-style time-overlap dedup that retired that trap on 2026-07-22 **cannot be run here**, and with no lat/lon
  either there is no cross-clip identity linkage on any axis we hold. **Clip granularity is the provable ceiling
  for every PhysicalAI split** — state it, don't imply drive-level cleanliness — impact: I3/split doctrine/C64 —
  same note §8
- [2026-07-18] [measured/mix] **Curve-rebalance measured on real bytes (FLEET P0#3): the "74% straight" is a
  comma/HIGHWAY property, not a whole-corpus one.** 630 eps / 125,247 windows, D1 eval strata (`|net yaw@2s|`
  <5°/5-20°/>20°): **comma2k19 83.1% straight** (10.5/6.4 gentle/sharp), **PhysicalAI 56.0%** (23.4/20.6) —
  urban is already at the 55-60% target. Natural window pool = **63.9% straight**; the fleet's ~74% ≈ a comma
  0.65-0.70 mix (comma 0.70/pai 0.30 → 75.0%). Two quantified levers to 57.5%: **source-mix** (+10pp comma ≈
  +2.7pp straight → shift to urban/ZOD/PandaSet = the ZOD rationale, in numbers) and **window turn-weighted
  sampling** β=s(1-t)/(t(1-s)) = **1.31** (natural pool) → 2.22 (comma-0.70). Recipe verified by construction;
  a drift-guard test pins the strata to `driving_diagnostic.py`. Training-recipe change → D-018 ESCALATE before
  flipping the trainer — impact: top-risk/D1-straight/H25/G3 — `2026-07-18-curve-rebalance-measured-and-idm-lit.md` §2
- [2026-07-18] [arXiv/lit] **IDM/latent-action + intrinsic-canon delta (seeds G2/H7, no status change P8):**
  Sensorimotor World Models (2606.20104, IDM-regularizer stops collapse + preserves the action — the pose-less
  pseudo-label recipe); ACID (2607.02403, action-cycle-consistency = a per-clip pseudo-label QUALITY gate for the
  H7 bridge); Latent-WAM (2603.24581) / DriveWAM (2605.28544) driving latent world-action models; "What Do Latent
  Action Models Actually Learn?" (2506.15691, LAM failure-mode caution — read before trusting any latent
  pseudo-label); **X-Lens (2607.12993)** intrinsic-guided canonicalization to a reference camera + calibration
  tokens = the closest recent work to our D-016 `f_eff=266` / H17 unified-FOV. New datasets → Benchmarks seam:
  global urban dashcam corpus (2604.01044, pose-less, curve/IDM candidate, license TBD), DrivingGen (2601.01528),
  ScenePilot-4K (2601.19582), LiAuto-DriveAction (HF) — impact: H7/G2/D-016/H17/D-028 — same note §4
- [2026-07-18] [measured/loader] **ZOD loader SHIPPED + geometry falsifier PASS — the #1 owned real-urban ingest.**
  ZOD front is a **Kannala-Brandt fisheye** (8 MP, 3848×2168, **HFOV 120°**, 10 Hz) and KB's radius
  `r(θ)=f(θ+k1θ³+k2θ⁵+k3θ⁷+k4θ⁹)` IS exactly `calib.FThetaIntrinsics.poly` (odd-power) → `kb_to_ftheta`
  reuses the proven f-theta crop path with **zero new geometry math** (confirms OWN_DATASET_PLAN's
  "fisheye→ftheta_*" with numbers). **Pre-registered falsifier ANSWERED (grounded on the published 120°
  spec, robust to real KB coeffs): f_eff=266.0 px, observed_frac=1.00, drop_in=True** — a 120° fisheye crops
  INWARD to canonical ~51.4° so nothing is padded. ZOD is **geometrically UNBLOCKED** (contrast PandaSet,
  height-bound f_eff 467; **ZOD needs NO calib.py R1 change**). Narrow-40° witness falsifies (observed_frac
  0.34) so the ≥0.5 gate is not vacuous. Real CAN steer + OxTS RT3000 (0.01 m/0.1°, @100 Hz) → OxTS heading
  drives yaw offset-free (better than PandaSet's motion-heading fallback); `zod_signals` reuses tested
  `cosmos_drive.poses_to_signals`. CC-BY-SA → SEPARATE shard (ShareAlike firewall). Intake `2026-07-18-zod-loader/`
  (19✓ standalone) + runnable real-bytes job card (blocked only on ZOD ACCESS, escalated) — impact:
  OWN_DATASET_PLAN §7#1 / FLEET_REVIEW P0#1 corpus-diversity / H4 arm-B / H7 — `2026-07-18-zod-loader-and-geometry-falsifier.md`
- [2026-07-18] [lit/seam] **New driving-WM benchmarks + a new urban dashcam corpus.** WorldLens (CVPR-2026
  Oral, WorldLens-26K human-rated realism/plausibility/safety) + DrivingGen (arXiv 2601.01528, generative-WM
  bench: trajectory plausibility + controllability ≈ our D2/D4) → **Benchmarks&Eval seam** (D-028). NEW
  **"A global dataset of continuous urban dashcam driving"** (arXiv 2604.01044) — a candidate curve-rebalance
  urban source (BACKLOG P0#3); **license + actions-availability probe queued** (owned-tier add vs YouTube-class
  barrier unknown until probed) — impact: D-012/curve-rebalance/H4 — same note §1
- [2026-07-17] [measured/tool] **D-016 R1 pinhole rectify UNBLOCKS the owned real-urban tier.** New primitive
  `pinhole_rectify` (grid_sample rectify-to-canvas, Brown-Conrady undistort + pad; mirrors the existing fisheye
  `ftheta_undistort`) lands `f_eff=266` **exactly by construction** where the square-crop is height-bound.
  Measured (grounded real intrinsics): **PandaSet front 467→266.0** (drop-in), at a cost of **37.7% masked
  periphery** (native VFOV 30.7° < canonical 51.4°; sky/hood band unobserved, road band retained) + **109px k1
  barrel distortion corrected**; comma2k19 reference untouched (266.0, 99.6% observed). **New ingest rule:** gate
  every source on `observed_frac ≥ ~0.5` — Udacity-like falsifies at 0.13 (narrow FOV = 87% mask). Undistort
  correctness: fwd↔iterative-inverse <1e-4, checkerboard recovery corr>0.9. Contract-drop-in (G-D2). Intake pkg
  `2026-07-17-d016-r1-pinhole-rectify/` (9✓). Coverage map: pinhole (PandaSet/Udacity/comma) → this; fisheye
  (ZOD KB/PhysicalAI/Cosmos f-theta) → existing `ftheta_*` — impact: D-016/G1/OWN_DATASET_PLAN/H17 —
  `2026-07-17-d016-r1-pinhole-rectify-unblocks-owned-real-urban.md`
- [2026-07-17] [measured/pitfall] **A8 on 12 real comma-val eps (3,600 frames): 0.0596@0.05 / 0.0240@0.10**
  (curr-frame slice) — reproduces the 2026-07-07 baseline (~0.053/0.012), low-consequence highway regime holds
  on held-out val. **Harness pitfall:** `stats.frame_change_fraction` assumes float [0,1] but the epcache stores
  uint8 [0,255] → a direct caller gets a meaningless ~0.74 (uint8 subtract wraps); convert via `to_float_frames`
  first. BACKLOG: make `stats` uint8-safe — impact: H3/A8, stats harness — same note §4
- [2026-07-15] [measured/gate] **WorldModel-Synthetic-Scenarios is POSE-LESS** (BACKLOG P0.1 gate CLOSED on
  real bytes): each clip = `<family>/<clip>/{description,video}`; `video/` = **7 camera mp4s** (front_wide,
  front_tele, 3 fisheyes, rear_left/right) @24 fps ~462 frames; `description/<cam>.json` = a **Qwen2.5-7B
  caption + `{weather,time_of_day,surface_type,region}`**, NOT a pose. No vehicle_pose/CAN/trajectory anywhere.
  → the "near-zero cosmos-mirror" assumption is DEAD; loader path is (a) video-only, (b) **IDM/H7 pseudo-label**
  (Phase-1, needs a trained inv-dyn head), or (c) **usable-today semantic-label index** (captions+metadata →
  BACKLOG P1 2d + SC-02/05/06). Families emergency/lanechange/nudging/pedestrian/weather_degradation. Do NOT
  fetch the 8.3 TB pixels until (b)/(c) scheduled — impact: H7/D-014/D-022/BACKLOG — `2026-07-15-worldmodel-pose-gate-and-pandaset-geometry.md` §1
- [2026-07-15] [measured/loader] **PandaSet loader shipped (intake, 16✓) + a grounded D-016 GEOMETRY BLOCKER.**
  CC-BY-4.0 real-urban adapter (plan §7 #2), reuses cosmos geometry (motion-heading 4×4 → poses_to_signals),
  I7≡comma2k19, I3 seq-split. Grounded on REAL front calib (arXiv 2112.12610: fx=1970.01, 1920×1080, k1=−0.589):
  centered square-crop canonicalization is **height-bound** (ideal crop 1896 px > 1080 frame height) → lands
  **f_eff=467 px vs canonical 266** (~1.75× scale mismatch) → NOT drop-in; **rule: any fx>1122 px on a 1080-tall
  frame is height-bound.** Distortion k1=−0.589 also ignored by the pinhole path. Loader **fails loud**
  (GeometryError) so it can't pollute the mix. Fix = D-016 R1 pad-crop+undistort in calib.py — a **prerequisite
  for the whole owned real-urban tier** (ZOD/Udacity hit it too), promoted from "R1 nicety" to blocking — impact:
  H7/H4/D-016/OWN_DATASET_PLAN — same note §2
- [2026-07-15] [HF sweep/D-012] New `nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Synthetic` (card-only, no data
  payload yet — watch; sibling of Cosmos-DD). `Newsflare/…-autonomous-driving-videos` = commercial stock video
  (copyright barrier, excluded, like OpenDV). No new ungated real-AV video corpus → owned real-urban gap stays
  **ZOD-shaped** (plan §7 #1). Literature: IDM/latent-action-in-the-wild now dense (2601.05230, 2602.16229,
  LatentVLA, FLAM) → frozen-encoder IDM+WM on unlabelled video is the standard recipe → makes pose-less corpora
  (WorldModel-Synth, YouTube) trainable via the comma/ZOD real-CAN bridge — impact: H7/D-012 — same note §3–4
- [2026-07-14] [loader/license] **Cosmos-Drive-Dreams** (`nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams`)
  = **CC-BY-4.0** → the one *publicly-claimable* rich AV corpus (closes the gap left by the real
  PhysicalAI-AV exclusion). RDS-HQ: 5 843 clips + 81 802 synth videos, 7 weathers, 30 fps, per-frame
  4×4 `vehicle_pose`; front_wide_120fov = same 120° HFOV as PhysicalAI (D-016 focal reuse). Loader
  ships (intake pkg, 9 tests): derives steer/accel from geometry (`κ=yaw_rate/v`, low-speed clip),
  D-015 9-ch, `CORPUS_META` byte-identical to comma2k19 (D-017 I7 → admissible in the D-010 mix) —
  impact: D-014/D-002/H7/H4 — `2026-07-14-cosmos-drive-dreams-loader-and-landscape.md`
- [2026-07-14] [doc] `DATASET_LANDSCAPE.md` created (D-012 standing duty, was missing): 3 tiers, per-corpus
  license class / size / actions / urban-richness / cost-to-first-batch. Firewall: public numbers =
  comma2k19 + Cosmos-DD only. Next: verify WorldModel-Synthetic-Scenarios card; add Zenseact ZOD (real-CAN
  #2, H4 arm-B) — impact: D-012/G-D1 — `DATASET_LANDSCAPE.md`
- [2026-07-14] [arXiv] H7 latent-action/IDM surge: **LAWM** (2509.18428, latent actions from unlabeled
  video via world modeling → the labeled-bridge our comma2k19 IDM serves), **Drive-JEPA** (2601.22032,
  V-JEPA latent WM for E2E driving → "world model" no longer differentiates; moat = hierarchy+efficiency+
  imagination+self-monitoring), **HiLAM** (2603.05815, hierarchy×latent-action), **CLAW**/**DeFI**
  (label-free forward/inverse dynamics → flow/forward-consistency term for the IDM). External support
  only, no status upgrade (P8) — impact: H7/H3/H1 — same note §5
- [2026-07-09] [measured] **PhysicalAI-AV R1 selection from cached egomotion**: 30 cached chunks → 2,850
  clips scored (0 errors), **1,926 pass the driving gate (67.6 %)** → R1=2,000 is 74 short of reachable
  from cache (needs ~1–2 more egomotion chunks). Gate failures (924) are **all speed-band**. Camera fetch
  = same 30 chunks as R0 (~60 GB) but **3.85× the clips for identical bandwidth** (per-chunk cost) →
  fetch-plan rule: extract ALL gate-passing clips per downloaded chunk. Episode-contract PASS on a real
  clip (`[199,9,256,256]` u8, 6.5 s/clip). Tool = intake pkg `2026-07-09-physicalai-r1-selection/` (3 tests)
  — impact: H7/H4/DATASET_LANDSCAPE rank #1 — `2026-07-09-physicalai-r1-selection-and-worldmodel-scenarios-license.md`
- [2026-07-09] [license/loader] **PhysicalAI-WorldModel-Synthetic-Scenarios** (`nvidia/…-Synthetic-Autonomous-Driving-Scenarios`)
  = **OpenMDW-1.1, UNGATED** (Linux Foundation permissive; NVIDIA's Cosmos/Nemotron license) → *preliminarily
  public-claimable* (proposed D-022; firewall held to comma+Cosmos until Sayed/legal confirms). 264 k clips /
  8.3 TB / 4K@24 fps; families cut-in 32.9 % · veh–ped 21.1 % · lanechange 12.9 % · ped 12.4 % · **weather-deg
  9.2 %** · nudging 8.8 % · **emergency-veh 2.7 %**. **⚠ card lists RGB+captions+metadata but NO ego pose/actions**
  → the "near-zero cosmos-mirror" assumption is at risk; confirm a pose field before loader work (else IDM/H7 or
  video-only). Advances SC-02/05/06 data rows — impact: D-014/H6/H15/D9/H4 — same note §2
- [2026-07-07] [measured] comma2k19 loader (D-009) decode path validated on REAL bytes: `av` decodes
  real HEVC → [200,3,256,256] uint8 @ ~105 fps (py3.13/Win), stack→[199,6,256,256] — impact: D-009/H7 —
  see `2026-07-07-comma2k19-data-card.md` §5
- [2026-07-07] [license/D-002] PhysicalAI-AV **real** sets (-Vehicles/-NCore/-NuRec) = NVIDIA AV Dataset
  License: **internal-dev-only, confidential, 12-month expiry, no public claims** → EXCLUDED from public
  benchmarks (comma2k19/MIT stays the public corpus). Cosmos-Drive-Dreams = **CC-BY-4.0**, ungated → the
  one publicly-claimable AV asset. Internal use needs Sayed+NVIDIA-legal sign-off ("using NVIDIA tech")
  — impact: D-002, all public claims — `2026-07-07-physicalai-av-license-review.md`
- [2026-07-07] [tool+finding] A8 statistics harness shipped (`stack/tanitad/data/stats.py`, 6 tests):
  per-corpus/per-domain `frame_change_fraction` distribution for change-weighting + D-010 mix. Measured
  toy=0.046 (threshold-INsensitive, hard-edged) vs comma-real=0.053→0.012 @0.05→0.10 (threshold-sensitive:
  real change is mostly small gradient, ~1.2 % large) — change-weight the small-but-real residuals —
  impact: H3/A8, D-009, D-010 — `...-validation-and-h7.md` §4a
- [2026-07-07] [measured] A8 on REAL highway camera `frame_change_fraction`≈0.053@0.05 / 0.012@0.10 — only
  ~1.7× the toy floor; raw-RGB under-reads consequence on low-texture highway → change-weighted loss
  justified — impact: H3/A8, W2 bake-off — `2026-07-07-comma2k19-validation-and-h7.md` §4
- [2026-07-07] [measured] comma2k19 = **MIT**, ungated HF `commaai/comma2k19`, ~100 GB (10×8.73 GB chunks);
  real CAN steering (steering-WHEEL deg, ratio ~15.3) + GNSS poses, zero labels; first real batch ≈1–2
  engineer-h, 0 new code — impact: D-009/H7/G-D1 — data card
- [2026-07-07] [finding] Windows `|` path bug: comma2k19 route dirs (`dongle|date`) illegal on Win32 →
  extract/train on Linux A40 pod only; do NOT extract Chunk zips on the Windows dev box — impact: ops —
  data card §6
- [2026-07-07] [arXiv] H7 deltas: LAOF (optical-flow-consistent latent actions, label-sparse gains) +
  Sensorimotor World Models (IDM-as-perception) → add flow-consistency term to seed IDM; log
  steering-ratio calibration residual — impact: H7 — `...-validation-and-h7.md` §3
- [2026-07-05] [kickoff] Initial research baseline for all hypotheses established; discipline agenda
  seeds defined — impact: all — see `../../INITIAL_RESEARCH_SYNTHESIS.md`
