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

- [2026-08-31] [MEASURED/descriptor + PUBLISHED/card] ⭐⭐ **THE GATE'S *"ONLY CORPUS THAT COULD TEST P2(b)"* IS WRONG,
  AND THE CHEAP TRIAGE IS ALREADY UNBLOCKED.** `yaak-ai/L2D` separates realised state from actuation IN ITS OWN
  DESCRIPTOR (`l2d_info.json` = the corpus's `meta/info.json`, re-read today): `observation.state.vehicle[8]`
  (speed/heading/GPS/IMU) + `waypoints[10,2]` vs **`action.continuous[3]` = gas_pedal_normalized ·
  brake_pedal_normalized · steering_angle_normalized** and **`action.discrete[2]` = gear · turn_signal**.
  ⭐ `turn_signal` is a DECLARED INTENT — not recoverable from the path even in principle, and discrete, which is
  the shape Codevilla branching needs. ⭐ **Neither of the gate's two blockers binds**: the triage is state-only
  (~155 MB parquet, no video) so the 65.2°-vs-120° geometry call and "not on Thor" both drop out. ⛔ But: L2D ships
  **NO intrinsics at all** (worse than comma2k19 for a vision arm), signal provenance is **UNVERIFIED at the card**
  (no CAN/bus statement), and `LOOP_STATE.md:1025` holds a 🔴 **GDPR gate on frames** — which does NOT bind a
  state-only correlation. ⛔ Parity untouched: data-only, or a separate parity key — impact: **V7 gate P2(b)
  premise (escalated)** — `2026-08-31-command-bearing-corpora/RESULT.md` F1/F2
- [2026-08-31] [PUBLISHED lib `2606.12987` §3 FULL TEXT] ⭐⭐⭐ **nuScenes SHIPS OUR EXACT ACTION PARAMETERISATION,
  FROM THE BUS — probably the cheapest of the three routes.** *"Ego-vehicle actions are extracted from CAN-bus data
  as 2D vectors aₜ=(steerₜ, accelₜ), z-score normalized using training-set statistics only"*, on 150 held-out
  nuScenes scenes. ⇒ the *form* is held constant and only the PROVENANCE varies — the one-variable version of the
  gate's triage. ⚠️ THREE THINGS UNVERIFIED and each is load-bearing: which nuScenes CAN field, whether we hold the
  expansion, and the licence. ⭐ Same paper is *"genuinely action-controllable (steering drives scene displacement,
  Spearman ρ = 0.81, vs −0.18 for regression)"* — impact: P2(b) corpus choice — same RESULT.md 4b
- [2026-08-31] [MEASURED/repo] ⭐ **TWO B1 OPEN ITEMS EXIST AS NATIVE L2D FIELDS.** D-B1-100H's own unfilled WORK
  ITEM — the **road-class stratum**, *"the axis governing our 88.7 %-longitudinal gap"* — is `observation.state.road`
  (OSM highway class). And the regime D-SPEED-GATE **hard-excludes** (`physicalai_r0.py:100`, `score = 0.0` unless
  `2.0 <= mean_v <= 14.0`, i.e. every motorway clip deleted, not down-weighted) is **directly selectable** via
  `max_speed` + `road = motorway`. Also native: lane count, surface, precipitation/lighting, 4,219 nav instructions
  with metric distance-to-manoeuvre, EXPERT 86.2 %/STUDENT 13.8 %. ⛔ A cross-corpus arm confounds corpus with
  regime (Germany, one vehicle, one rig) ⇒ it cannot answer *"does high-speed data help"* — impact: D-B1-100H,
  D-SPEED-GATE sourcing options — same RESULT.md F3
- [2026-08-31] [PUBLISHED lib `2607.27017` abstract-only] ⚠️ **D-DATA-EFFICIENCY HAS AN OBJECTIVE-SIDE
  PRECONDITION, MEASURED OVER A 5× DATA RANGE.** *"Every arm missing information or prediction pressure stays flat
  over a fivefold data range"* and *"additional data improves only the parameters it already acquires"* (RH20T, two
  robots, 4,258 episodes). ⇒ **a data lever cannot be scored honestly on an arm whose objective does not already
  acquire the quantity** — a direct warning about the planned diversity ablation, which would read FLAT for a
  reason that has nothing to do with the data. ⚠️ Robotics not driving; supports SEQUENCING, not a magnitude —
  impact: D-DATA-EFFICIENCY experiment order — same RESULT.md F4
- [2026-08-31] [method/retraction-class] ⛔ **A NARROW `Select-String` PROBE MANUFACTURED A FALSE ABSENCE AND WOULD
  HAVE HIDDEN A LIVE BLOCKER.** Probing `'L2D|yaak'` over 3 steering files returned 1 hit, which was
  **`adaptive_avg_pool2d`** (`pool2d` contains `l2d`) — it read as "L2D is unknown to steering". A case-sensitive
  probe over the whole `Project Steering` dir returned **9 hits in 3 files**, including the 🔴 GDPR gate. ⇒ the
  claim had to be rewritten DOWN from *"a corpus nobody knew"* to *"a corpus that is known, adapted and sliced,
  whose JOIN to P2(b) was never made"*. ⭐ The generalisable rule: an absence probe whose pattern can match INSIDE
  an unrelated identifier must be re-run case-sensitively and directory-wide before the absence is written —
  impact: absence-claim discipline — same RESULT.md F1
- [2026-08-30] [PUBLISHED lib `2511.00088`] **The two label sources are probably NOT conditionally independent:
  Alpamayo-R1's auto-labeler is given "the ego vehicle's trajectory, dynamic states, and meta actions" — so the
  VLM saw the geometric source's own input.** ⇒ Dawid-Skene, data programming (`1605.07723`) and spectral
  meta-learners (`1407.7644`) are ALL inadmissible by their own identifiability assumption; `2601.22336` shows
  ignoring the dependence can REVERSE the aggregate. ⚠️ UNVERIFIED for our clips — a 0-GPU provenance check that
  must precede any aggregation choice — impact: D-LAT-AGREE (DataFlyWheel owns) —
  `2026-08-30-label-source-adjudication/RESULT.md` B1
- [2026-08-30] [PUBLISHED lib `2507.20174`+`2310.19785`] **The VLM literature PREDICTS our measured
  `D-DATA-ALPA-LAT` a priori:** LRR-Bench finds VLMs reach human level "only on the two simplest tasks" of
  left/right discrimination, several near zero; `2310.19785` finds all 18 VLMs poor on spatial relations (56 % vs
  human 99 %). Our lateral 31.2 % vs 23.9 % shuffled (p=0.335) is the field's expected result, not an anomaly —
  impact: D-LAT-AGREE background, D-DATA-COT-HALLUC — same RESULT.md B3
- [2026-08-30] [PUBLISHED lib `1911.00068`,`1804.06872`] ⛔ **cleanlab and co-teaching are FALSE-POSITIVE
  GENERATORS on this label problem** — both assume class-conditional noise; a deterministic geometric rule's
  errors are a function of x (instance-dependent by construction, `2110.12088`), so small-loss/confidence
  selection picks the MAJORITY CONVENTION and reports it clean. Admissible instead: model both sources with
  per-annotator heads (`2110.05719`), soft targets (`2511.14117`), ordinal adaptive boundary (`2509.02351`),
  noise-ignorant ERM on frozen features (`2411.00079`) — impact: D-LAT-AGREE method choice — same RESULT.md B4/B5
- [2026-08-30] [measured/repo] **The lane-detector package does NOT close the `D-NUDGE-ABSORB` gap (zero detector
  runs, zero measured transfer) — but its geometry ALREADY covers that scope.** A lane change crosses a boundary
  at ~1.6–1.85 m (half of a 3.2–3.7 m lane), inside the package's ≤3.1 % near-pinhole ego corridor (±25°) ⇒ as-is
  CLRerNet is geometrically sufficient, no cylindrical→pinhole rectify needed. The 39 LANE_CHANGE records are
  absent from its 68+41 waiting-consumer list and should be a third set (minutes) — impact: D-NUDGE-ABSORB —
  same RESULT.md A1–A3
- [2026-08-30] [measured/source] ⚠️ **`D-NUDGE-ABSORB`'s auditability caveat went STALE the day it was written:
  `lat_peak_m` IS now persisted** — `ego_manoeuvre.py:112` (dataclass field), emitted `:379`, with a `:316`
  comment dating the unconditional lateral track to 2026-08-30. ⛔ Unresolved: whether the 4,719-record RELEASE
  blob was re-emitted with it. Also: the row's NUDGE predicate is incomplete — the live condition conjoins a 5°
  yaw gate (`:335`) — impact: D-NUDGE-ABSORB row correction — same RESULT.md A4/A5
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

- [2026-09-01] [MEASURED ours, analytic trajectories + set arithmetic over source] **THE TWO `g_tac` PRODUCERS ARE NOT A FORK - THEY ARE NEAR-DISJOINT** - `g_tac_geom` REFUSES **3 of 4** of `tactical_goals`' LAT tokens and **5 of 8** of its LON tokens; A's only surviving LAT token is `LAT_UNCONSTRAINED`. LON agreement **1 of 7 scenarios (14.3 %)**, the one agreement being the clean-stop control (and there they agree to **0.06 m**: 16.4 vs 16.34 m). Both controls (K1 no-stop / K2 clear-stop) PASS, so the rate is interpretable.
  impact: **row 7's "CORRIDOR_OFFSET 2-vs-1 threshold contest" is the WRONG QUESTION. Two admissibility defects measured instead: (a) A emits `SPEED_BAND` with provenance `geometry(held-speed)` - exactly the hindsight-ego substitution B's F-14 blocker names as inadmissible; (b) A's `CORRIDOR_OFFSET` reads **21.65 m** on a plain constant-curvature arc, 21x its own 1.0 m threshold, independently reproducing B's refutation** - `Data Engineering/Research/2026-09-01-gtac-producer-cross-agreement/RESULT.md`
