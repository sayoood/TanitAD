# Stream R3 — Data, Encoding, Resolution, Labels & In-Repo Data-Efficiency Levers

**Scope:** whole-programme static review, stream R3 of 9. Repo `/home/user/TanitAD`, branch
`claude/optimistic-shannon-vixpo6`, HEAD `467ce8a` (2026-08-04). No pod/torch access — everything
below is read from source, `Project Steering/MODEL_REGISTRY.md`, and dated research notes under
`TanitAD Research Hub/`. "As of HEAD" means the repo state reviewed, not today's calendar date
(2026-09-25); several items below may have moved on a pod since 2026-08-04 and that is flagged
where relevant. All evidence classes are stated per CLAUDE.md: **MEASURED (ours + path)** ·
**PUBLISHED (cited)** · **INHERITED (another doc, not re-verified by me)** · **ESTIMATED** ·
**HYPOTHESIS**.

---

## 1. Headline — ranked findings

1. **The whole programme trains on well under 1% of PhysicalAI-AV, on 1 of 6 camera views, and 0% of
   LiDAR/radar.** PUBLISHED (HF card, `nvidia/PhysicalAI-Autonomous-Vehicles`, via web search
   2026-09-25): the source corpus is **306,152 clips × 20 s = 1,700 h**, 6 camera views (front-wide
   120°, front-tele 30°, cross L/R 120°, rear L/R 70°), LiDAR on 298,326 clips, radar on 160,761.
   The parity train corpus is **2,376 clips = 13.2 h = 0.776%** of total hours
   (`stack/tanitad/data/physicalai.py`, corpus key `physicalai-train-e438721ae894`); v2bal is 9,000
   clips (2.94%); v5f is 2,400 (0.78%). Every trained arm reads **only `camera_front_wide`**
   (`physicalai.py:1-15`) — never front-tele, cross, rear, LiDAR or radar. This is the single largest
   utilization gap in the programme and it is a resourcing/ambition question (CLAUDE.md rule 5), not
   a bug.
2. **`obstacle.offline` (3D agent tracks) exists on 97.44% of the corpus and is still not fed into
   any trained arm's inputs or losses; the eval instrument that would score against it is
   code-complete but not wired to any val-set eval as of HEAD.** MEASURED (ours, reading
   `taniteval/taniteval/four_families.py:225-235`, `lead_source.py:1-20`, and
   `TanitAD Research Hub/Architecture & Inference/Research/2026-08-03-longitudinal-distance-keeping.md`
   §6): the LONGITUDINAL family's `distance_keeping` sub-metric (headway/TTC/time-gap — binding since
   2026-08-02) is implemented, unit-tested (14 tests, `taniteval` suite green), and its
   ground-truth-vs-null discrimination control **D-LEAD-1 PASSED** on 14,027 windows / 1,431 clip
   clusters (Δ min-TTC +1.7474 s [1.5813, 1.9218], Δ headway +0.9769 m [0.8830, 1.0758], Δ time-gap
   +0.1641 s [0.1499, 0.1786], all separated, correct sign). But the val-40 episodes' own
   `obstacle.offline` chunks were never fetched and `win["lead"]` was never built for the eval
   runner — so **every arm's eval as of HEAD still reports `distance_keeping: UNAVAILABLE`**
   (`four_families.py:230-233`). This is the closest-to-finished, highest-leverage item against the
   binding "four metric families" rule and against the programme's own finding that 88.7% of the
   oracle gap is longitudinal.
3. **The situation-classifier labels (lane-change / intersection) are a pure deterministic function
   of ego pose alone, and the in-repo intersection detector only computes half of what it claims.**
   MEASURED (ours, `stack/tanitad/data/situations.py:35-38,62-67`): the emitter reads only
   `d["poses"]`; every detector takes `K = kinematics(P)` and the emitter passes `cross=None`, so
   "intersection" in `stack/` is **the turn-half only** — the cross-traffic half
   (`sc_cross.py`, needs `obstacle.offline` + calibration) exists but was never promoted out of
   `TanitAD Research Hub/.../incoming/2026-07-26-situation-classifier/scripts/sc_cross.py` (confirmed
   absent from `stack/` and `stack/scripts/` by repo-wide grep — second probe per CLAUDE.md rule 2).
   Separately, INHERITED from `situations.py:19-45` (already corrected in-repo 2026-08-03, not my
   finding, cited because it governs admissibility): an ego-input situation head partly reads the
   label's own generating process (a leak, not a capability); the deployable arm is vision-only
   (`head_img`), and its AP (0.03741) beats its own permuted-feature null (0.01715) by
   **+1.1749 [+0.7930, +1.6890]** — i.e. 2.18× its own null, MEASURED per that file.
4. **No pixel-space augmentation of any kind exists in the real-camera training path — in
   particular, no horizontal mirror + steer-sign-flip.** MEASURED (ours, repo-wide grep across
   `stack/tanitad/data/*.py`, `stack/tanitad/train/*.py`, `stack/scripts/train_flagship4b.py`): the
   only "flip"/"mirror" hits are `comma2k19.py`'s heading-**mode** default flip (a labeling-regime
   switch, not image augmentation), `lan.py:390`'s symbolic `mirror_route()` (topology, not pixels),
   and a MetaDrive rendering flag. For a laterally near-symmetric driving task this is the cheapest
   available ~2× multiplier on the parity corpus and it is entirely unused.
5. **The v2corpus vs v1 comparison is confounded on two independent axes, both already caught
   in-repo before any headline number was quoted — but the lever-matched control that would fix it
   has not been run.** MEASURED (ours, `Project Steering/PREREG_v2corpus_vs_v1.md`, amendment
   2026-08-01 + `V2BAL_LEAK_FINDING.md` 2026-07-29): (a) **21 of the 40 canonical val episodes
   (52.5%)** sit inside v2corpus's 9,000-clip training selection — reconciled by base rate (v2bal
   selects 47.4% of the shared 18,987-clip scored pool); the adopted headline is the **19 leak-free
   episodes** (`leak_free_n=19`). This is a *different* val surface and a *different* number from
   the fact sheet's "256 of 600 (42.7%)", which is the same leakage mechanism measured against the
   larger 600-episode deployment val — both are real, neither supersedes the other, and they must
   not be averaged or conflated. (b) v2corpus's launch command differs from v1's in the **entire
   `--v2` lever pack** (ego-dropout 0.25, fa-dropout 0.3, goal-decode, nav-dropout 0.5, gated-intent,
   anchor-tactical, invdyn-gradscale 0.25, v2-label gate) **and** `rollout_k` (12 vs 4) — so "the v2
   corpus produced X" is not an admissible claim today; only "the v2-line arm, which differs in
   corpus AND the whole `--v2` pack, produced X" is. The lever-matched control (v2bal data + v1
   flags, or the reverse) is designed in the same document but not launched as of HEAD.
6. **The own-dynamics-encoder / "IDM-for-YouTube" pseudo-labelling thesis is refuted at its current
   design point, not merely unproven.** INHERITED from `Project Steering/MODEL_REGISTRY.md` §10.1
   (re-read, not re-run by me): `dynenc-branchB` (40k steps, 2,466 clips, rig-A+rig-B+comma2k19,
   GAIA-2-style camera-conditioning) failed its pre-registered held-out-rig-transfer gate decisively
   (cross rig-B speed R² needed **>0.9**, best achieved **−0.667**; yaw R² negative on every
   cross-rig set) and underperformed the plain (non-rig-conditioned) flagship-v1 ViT encoder even
   in-domain (+0.039/−0.603 vs v1's +0.862/+0.910 with matched converged heads). Two independent
   heads reading the latent weakly means the finding is about the **representation**, not head
   capacity, and it is not an artifact of the val leakage also found in that arm (the one
   Branch-B-favoring row has a CI spanning zero and is the leaked row). The idea of pseudo-labelling
   large uncurated video via this encoder is therefore unsupported until a materially different
   design is tried.
7. **The route/nav signal every strategic and REF-C head has received to date is a documented
   architecture-level echo, and its own designed replacement (LAN) carries an adverse prior from a
   prior test, not a confirmed win.** INHERITED (`stack/tanitad/data/lan.py:1-24`, re-read not
   re-run): `nav_command`'s 4-way one-hot is valid on only 21–25% of windows
   (`nav_valid_frac` 0.21–0.25 in all four arms including deployed v1); the route head's
   `route_skill_vs_chance = 0.0`; and every published REF-C eval number decodes with `nav_cmd=None`
   (a constant). LAN (dense arc-length route corridor in ego frame) is built and pre-registered
   (`Project Steering/PREREG_lan_refc.md`) as the fix, but that same pre-registration records that
   a prior "supply the oracle route at eval" follow-up on REF-C was **already refuted**: ADE Δ
   +0.0024 (not separated), cross-track **+0.0031 [+0.0001, +0.0063] separated WORSE**, with a
   decomposition showing **91% of the oracle-route benefit is lost downstream** of the planner — a
   fusion/architecture bottleneck, not a label-quality one. A better goal label alone is not
   guaranteed to help; how it is consumed matters as much.
8. **`obstacle.offline`-as-model-input and `obstacle.offline`-as-eval-metric are two different,
   already-separately-tested questions, and the programme's own docs show the risk of conflating
   them.** MEASURED (ours, `TanitAD Research Hub/Data Engineering/DATA_STRATEGY_FOR_HIERARCHY.md`
   §0 correction, 2026-07-21): concatenating lead-state features onto an ego-only regressor
   predicting the ego's own future 2 s longitudinal displacement moved held-out error by only
   **+1.16% [−0.92, +3.19]** — inside the pre-registered FAIL band (≤5%) — so lead state is *not*
   shown to be a useful **model input** for that specific proxy task. This is compatible with, and
   distinct from, finding 2 above (the metric itself discriminates GT from a null). A post-hoc,
   **not pre-registered**, "lead-conditioned specialist" showed ~8–10% gain on the 38.5% of windows
   with a lead — flagged HYPOTHESIS-class in the source doc itself, not a pass.
9. **Two curation/balancing implementations exist; the one actually used to build v2bal is not the
   one that lives in `stack/tanitad/lake/curation.py`.** MEASURED (ours, repo-wide grep):
   `curation.py`'s `curate_corpus`/`weakness_boost`/`inverse_frequency_weights` are imported only by
   `lake/curation.py` and `lake/enrich.py` themselves — no corpus-build script
   (`build_pai_cache.py`, `physicalai_r0.py`, or the v2bal selector) imports them. v2bal's actual
   rebalancing (turns 14.25%→28.0%, lane-keep 59.6%→45.0%, junction presence 37.7%→61.3%) was done
   by a separate, bespoke greedy water-filling script
   (`.../incoming/2026-07-24-v2-corpus-50h-balanced/select_v2_corpus.py` +
   `score_v2_pool.py`). The lake curation module is designed and unit-tested infrastructure that,
   as far as this review can tell, has never built a corpus that trained an arm.
10. **Wheelbase and rig-geometry corrections exist as opt-in, non-default modes precisely to protect
    parity — but that means every currently-deployed arm still trains on the biased default.**
    MEASURED (`physicalai.py:46-63,85-119`): 98.2% of clips carry a wheelbase >5% off the frozen
    constant 2.9 m (true values 2.730/2.850/3.135/3.165/3.216 m across 5 clusters, clip-mean
    2.9568); the corrected `per_clip_v1` regime exists and is measured to move flagship-v1 ADE
    **+0.0056 m [+0.0007, +0.0113]** and cross-track **+8.6% relative** — small but separated, and
    currently switched off for every deployed/gated arm because turning it on would silently change
    the meaning of the parity cache key (by design, `label_params()` only emits a cache-key
    fragment for the non-legacy regime).

---

## 2. Analysis per question

### Q1 — Utilization

| corpus / sensor | used by any trained arm? | size actually used | source |
|---|---|---|---|
| PhysicalAI-AV front-wide, parity | ✅ every flagship arm (v1, v3enc) | 2,376 clips / 13.2 h (0.776% of 306,152/1,700 h) | `physicalai.py`; MODEL_REGISTRY §0.1; HF card (PUBLISHED) |
| PhysicalAI-AV front-wide, v2bal | ✅ v2corpus only | 9,000 clips / 49.742 h (2.94%) | MODEL_REGISTRY §1.7 |
| PhysicalAI-AV front-wide, w120 | ✅ v5f only | 2,400 clips / 0.78% | MODEL_REGISTRY §1.8; `parity_manifest.json:2514` |
| PhysicalAI front-tele/cross/rear cameras | ❌ none | 0 | grep-clean of any adapter reading them |
| PhysicalAI LiDAR / radar | ❌ none | 0 | grep-clean; not in the episode contract at all |
| `obstacle.offline` (agent tracks, 97.44% coverage) | ⚠️ eval-side probes/gates only, **not** any trained arm's loss/input | n/a | `lead_state_gate.py`, `four_families.py`, `taniteval/lead_source.py` |
| comma2k19 (MIT) | ⚠️ one **dead** pre-reset run only | `p0-sB01-realmix`, stale since 2026-07-12 step 28,600 | MODEL_REGISTRY §948; R6 |
| MetaDrive front-cam (sim) | ❌ tests only | 0 | `test_metadrive_frontcam.py`; no launch command references it |
| Cosmos-Drive | ❌ tests only | 0 | `test_cosmos_drive.py`; `d8_preview.py`/`cosmos_verify.py` are previews, not training |
| nuScenes | ❌ ingest/registration only | 0 | `lake/ingest.py`, `scripts/ingest_nuscenes.py` |
| Argoverse2 | ⚠️ feeds `lan.py`'s lane-graph representation for the REF-C/LAN side-experiment only | n/a (not pixels, not the flagship) | `lan.py:57,478` |
| L2D (`yaak-ai/L2D`, Apache-2.0) | ❌ designed pilot, not executed | 0 (registered `ship` tier in the lake) | `DATA_STRATEGY_FOR_HIERARCHY.md` §3, §9; `stack/tanitad/data/l2d.py` exists and is contract-shaped but has no trained arm |

`MixedWindowDataset` (`stack/tanitad/data/mixing.py:78`) is the real+sim mixing mechanism (D-010):
fully implemented, contract-checked, unit-tested against comma2k19, Cosmos-Drive and MetaDrive, and
wired into `train_flagship4b.py:214` and `train_worldmodel.py:120,199` — but `build_datasets()`
(`train_flagship4b.py:175-218`) truncates to a **single** cache root whenever `--data cached` is
passed (`roots = roots[:1]`, line 194), which is what every flagship launch command in the registry
actually uses. The `realmix` path exists and is coded correctly; it is simply not invoked by any
arm on the current leaderboard.

**Bottom line:** the programme is data-rich (306k clips, 6 cameras, LiDAR+radar, agent tracks on
97%+ of clips, five additional real/sim corpora fully coded) and data-poor in what it actually
trains on (one camera view, <1% of clips, zero non-camera modalities, zero cross-corpus mixing in
any live arm).

### Q2 — Encoding & resolution

**Temporal stacking.** D-015: 3 RGB frames at 100 ms spacing (`t-200ms, t-100ms, t`), channel-stacked
into 9 channels by `comma2k19.stack_frames` (shared helper, `comma2k19.py:633-641`), called from
`physicalai.build_episode` (`physicalai.py:679-746`). This is a fixed 0.2 s receptive field baked
into the input tensor, not a learned temporal token; the encoder's own causal window
(`predictor.window=8` in the `flagship4b`-family configs, `stack/tanitad/config.py:344-347`) adds a
further 0.8 s of frame-to-frame history at the token/latent level. There is no tubelet or video-token
encoding anywhere in the repo — every arm treats video as channel-stacked stills.

**Resolution / crop.** The deployed geometry (`calib.py`) canonicalizes every camera to a shared
effective focal `F_REF = 266` px at a 256 px square output (`calib.py:38`), chosen so comma2k19's
own ~65.2° sensor is nearly uncropped. Applied to PhysicalAI's 120°-HFOV front-wide fisheye
(`PHYSICALAI_FRONT_WIDE_FTHETA`, native 1920×1080, `calib.py:389-391`), this keeps **only ~51.4°**
of horizontal field (`half-angle = atan(128/266) = 25.72°`; confirmed independently in
`situations.py:67`: *"the model receives the 256 px / 51.4 deg front crop"*). Final density is
**256 px / 51.4° ≈ 4.98 px/deg**. ESTIMATED pixel sizes at that density (simple trig from the
measured intrinsics, not empirically measured on real frames):

| object | range | angular size | pixel size (ESTIMATED) |
|---|---|---|---|
| car, 1.8 m wide | 20 m | 5.15° | ~25.7 px |
| car, 1.8 m wide | 50 m | 2.06° | ~10.3 px |
| car, 1.8 m wide | 100 m | 1.03° | ~5.1 px |
| traffic-light face, ~0.3 m | 50 m | 0.34° | ~1.7 px |
| lane-marking stripe, ~0.12 m | 30 m | 0.23° | ~1.2 px |

A distant lead vehicle (50–100 m) is a handful of pixels — enough for coarse presence, marginal for
closing-speed from a 100 ms-spaced 3-frame stack. Traffic lights and lane markings fall at or below
one pixel by 30–50 m, which is consistent with (and helps explain, not merely restate) the
programme's separately-established fact that PhysicalAI-AV carries no traffic-light or lane-graph
feature at all — even a labeled traffic-light state would be barely legible pixel-wise at this input
resolution and would need a dedicated tele crop (the front-tele 30° camera exists in the source
dataset and is entirely unused, finding §1.1).

**Two-rig handling.** PhysicalAI's front-wide has two physical rigs with different vertical mount
(cy ≈ 543 "rig A" vs cy ≈ 755 "rig B", a ~215 px raw offset). The deployed crop
(`ftheta_crop_resize(center="principal")`) centers on each clip's **own** per-clip `(cx, cy)`
(sourced from the dataset's own `calibration/` tables, `physicalai.py:150-245`) so the horizon lands
at the same output row for both rigs. ⚠️ When per-clip calibration is unavailable, the code falls
back to a corpus-median intrinsic whose `cy` is a rig-B value — the crop then reverts to
geometric-center and **silently loses the rig fix** for that clip (`physicalai.py:198-244`,
warns once per clip). How often this fallback fires in the actual built caches is UNVERIFIED from
static review.

**Wide-FOV / v5f cylindrical geometry.** v5f trains at the full 120° HFOV via equidistant-azimuth
(cylindrical) resampling to 256×640 (`f_ref` 305.5775, `calib.py:1247`). ⭐ **This build has a real,
measured defect that was caught and fixed before training, not after:** the raw 256×640 build is
**8.897% masked (fabricated/black) on rig B and ~0.06%→0.0017% on rig A**
(`calib.py:1245-1248`, C26 defect — replicate-padding rig-correlated pixels is "a free rig label").
The fix (`slice_v2_cache.py`, `calib.py:1250-1264`) is a bit-exact centred-pixel-slice (no
re-decode) to **176×624 (117.0°×32.1°)**, measured 0.000000 masked for every clip of both rigs over
all 3,000 canonical clips — and the v5f registry row confirms training actually runs at this
176×624 **sub**-frame of the 256×640 parent (`MODEL_REGISTRY.md:998`, "Geometry | 256×640 ... subframe
176×624"). A stricter 128×576 (108°×23.7°) option exists for exact 64-px readout tiling at 33% fewer
tokens but is not what v5f trains at.

**Augmentation.** None on real-camera pixels (see headline §1.4). No random crop, no color jitter,
no mirror. `parity.py` and `v2_dataset.py` do carry a `geom_augment` reference for the (separate,
side) `dynenc-branchB` run (±12 px vertical shift + matched camera params), but that is not part of
any flagship training path.

### Q3 — Labels

**Actions.** `steer = atan(WHEELBASE · curvature)` (`physicalai.py:596-642`, D-016 R1); `accel` is
the dataset's **own measured longitudinal `ax`** (`signals_at`, `physicalai.py:623-624`), explicitly
*not* a finite difference of speed ("differentiates interpolation noise and lags the true signal",
`physicalai.py:14-16`) — `finite_diff_accel` (`_contract.py:28-40`) is used only as a defensive
fallback if the `ax` column is ever absent, and by the generic (non-PhysicalAI) toy/adapter contract.
Wheelbase is frozen at the legacy constant 2.9 m for every parity-locked arm (see headline §1.10);
the corrected per-clip regime exists but is opt-in and off by default. `yaw` comes from the
orientation quaternion (unwrapped before interpolation, re-wrapped after) specifically because it is
standstill-robust where `atan2(vy,vx)` is not (`physicalai.py:606-609`).

**Maneuver / nav labels** (`stack/scripts/refb_labels.py`): a 5-class kinematic labeler
(lane_keep/turn_left/turn_right/accelerate/brake_stop) over a fixed 2 s horizon with documented,
frozen thresholds (`YAW_TURN_RAD=0.15 rad`, `DV_ACCEL_MS=+1.0`, `DV_BRAKE_MS=-1.0`,
`refb_labels.py:57-59`); curvature (turn) is checked with priority over accel so a braking turn is
classified as a turn. `nav_command` (route, "rev2 strategic layer", `refb_labels.py:35,138`) derives
left/right from the **net heading change over 15–25 s of future poses** — a privileged, future-only
signal, which is fine as *ground truth for scoring* but is the exact signal already shown (INHERITED,
`lan.py:1-24`) to have been fed as a *model input* and echoed rather than learned by the deployed v1
route head (`route_skill_vs_chance = 0.0`, `nav_valid_frac` only 0.21–0.25). This is not a new
finding of mine; I flag it here because it is the direct answer to "what ground truth exists for
STRATEGIC" and because Q3 explicitly asks about leakage wherever ego is a model input.

**Situation labels** (`stack/tanitad/data/situations.py`): lane-change and intersection(-turn-half)
detectors, thresholds frozen by `PRE_REGISTRATION.md` §2 and explicitly must not be swept. Input is
`P = [x, y, yaw, v]` **only** — "nothing in here is a model input" (`situations.py:67`). The emitter
(`scripts/emit_situation_labels.py:54-62`) reads only `poses`; the intersection detector's
cross-traffic half is stranded in `incoming/` (headline §1.3). A 2026-08-03 causality fix replaced a
centred-difference (`np.gradient`, leaking 0.1 s into the future despite a "STRICTLY CAUSAL" comment)
with a true backward difference for `omega_pre`/`alon_pre` (`situations.py:39-46`) — every
pre-fix `ego`-arm number built before that date is stale for causal claims (INHERITED, blast-radius
list at `situations.py:47-49`).

**LONGITUDINAL ground truth** (lead agent, headway/TTC): see headline §2 and §8. In one sentence —
the **instrument** is validated (D-LEAD-1 passed) but the **eval path** is prototype (val-40 not
wired), and a *different*, earlier test showed lead-state features do not obviously help a *model*
predict its own future motion. Both facts are true and answer different halves of Q3.

**STRATEGIC ground truth** (route/goal beyond nav_command): `lake/goal_labels.py` mints a richer
TanitDataSet-v3 vocabulary (VTARGET, VSOURCE, LONMODE, LATMANEUVER, DYN, ROUTE, conditionally SIGNAL
from CAN) purely kinematically, but explicitly leaves `HEADWAY, INTERACT, TACPOINT, LIGHTSTATE,
RULECTX, and the rest of the STRATEGIC layer` as `unknown`, deferred to "the deferred Cosmos-Reason2
pass" (`goal_labels.py:6-7,200-211`) — i.e., as of HEAD, most of the STRATEGIC vocabulary is
unfilled, and the module that fills what exists is consumed only by `lake/curation.py` and
`lake/enrich.py`, not by any flagship trainer (repo-wide grep, headline §1.9's sibling finding).
LAN (`lan.py`) is the actual candidate fix for a *dense, non-degenerate* route/goal signal; see
headline §1.7 for its status and adverse prior.

### Q4 — Sampling & curation

- **Window/stride:** episode-level windowing (`EpisodeWindowDataset`, `_contract.py:107-133`,
  default `window=6, max_horizon=4`); the flagship configs use `predictor.window=8` (0.8 s causal
  history) with multi-horizon rollout targets, e.g. `horizons=(1,2,4)` operative /
  `horizons=(8,16)` tactical in one reduced-scale preset (`config.py:344-347`) — I did not confirm
  the exact horizon tuple used by the full-scale `flagship4b` config used for v1/v2corpus/v5f;
  flag as UNVERIFIED which exact tuple trains the deployed arm, though the 2 s (20-step) planning
  horizon is consistent across the CEM/P2 and REF-C sections of the registry.
- **Splits are always CLIP-level (I3), never window-level**, both in the generic contract
  (`_contract.py:100-133`, docstring) and in `physicalai.split_clips`
  (`physicalai.py:749-755`, seeded `torch.randperm`) and in the lake's
  `is_eval_holdout`/`stable_unit_frac` salted-hash split (`lake/curation.py:184-207`).
- **Dedup** (`lake/dedup.py`): perceptual-hash (aHash luma + pHash) near-duplicate clustering
  within-source and geo/time-bucketed cross-source dedup, plus a two-pass union-find verdict
  (`dedup.py:35-216`). This is lake-ingest-side machinery; I could not confirm it ran against the
  parity, v2bal or v5f clip selections specifically (their build scripts are separate, bespoke
  selectors — see next bullet and headline §1.9) — UNVERIFIED whether the frozen parity corpus itself
  is dedup-clean by this mechanism or by an earlier, unaudited step.
- **Quality filtering** (`lake/filtering.py`): blur (variance-of-Laplacian), exposure bands,
  truncation fraction, corrupt-frame/pose sanity (`egomotion_sane`), and rig assignment by `cy`
  (`filtering.py:29-315`) — same caveat: designed and tested, application to the actual trained
  corpora not confirmed from static review (UNVERIFIED).
- **Class balance:** the parity corpus's natural balance is unknown to me from static review (no
  in-repo table found); the **v2bal** corpus was deliberately rebalanced by a bespoke greedy
  water-filling selector (`.../2026-07-24-v2-corpus-50h-balanced/select_v2_corpus.py`), MEASURED:
  turns 14.25%→28.0% (by the v1 labeler; 18.83% by the v2 labeler actually in force at training,
  a labeler-dependent discrepancy the registry itself flags, `MODEL_REGISTRY.md:975-977`),
  lane-keep 59.6%→45.0%, highway held at 38%, junction presence 37.7%→61.3%, stop presence to
  27.3%. This selection **breaks parity by design** (Sayed's ruling, recorded in
  `V2_CORPUS_DESIGN.md`).
- **Val design / leakage:** the canonical val is 40 episodes (881 windows,
  `physicalai-val-0c5f7dac3b11`), content-verified clean against parity train
  (0/40 and 0/600 by sha256 of raw `poses` and `frames_u8`, `MODEL_REGISTRY.md:1901`). A separate,
  larger 600-episode/13,198-window "deployment" val also exists (CV floor 0.6917 vs the 40-ep val's
  0.8377 — an easier split, per the fact sheet). v2bal's training selection leaks into **both**
  val surfaces, independently measured: **21/40 (52.5%)** of the canonical val
  (`V2BAL_LEAK_FINDING.md`, 2026-07-29) and **256/600 (42.7%)** of the deployment val (fact sheet,
  re-cite from `MODEL_REGISTRY.md`, not independently re-verified by me this session —
  INHERITED). A disjoint-by-construction remedy (draw val from the 9,987-clip unselected remainder
  of the scored pool) was designed 2026-07-29 (`V2_CLEAN_VAL_PLAN.md`) as "option B"; the later
  (2026-08-01) `PREREG_v2corpus_vs_v1.md` amendment instead headlines the **19 leak-free** canonical
  episodes ("option A") — I read this as option A having been adopted, but the registry text I
  found does not say so in as many words; flag as UNVERIFIED which option is the one actually used
  for any already-published v2corpus-vs-v1 number.

### Q5 — In-repo data-efficiency levers

| lever | status | evidence |
|---|---|---|
| Mirror augmentation + steer-sign flip | ❌ **not implemented anywhere** | repo-wide grep, headline §1.4 |
| Trajectory-perturbation / recovery synthesis | 🟡 **designed only**, blocked on rendering | `TANITDATASET_V1_STRATEGY.md:148` names CARLA perturbation+recovery as the "antidote to the open-loop→closed-loop collapse (1.69 m drift)"; blocked because RunPod GPUs expose `compute,utility` only → no Vulkan/EGL ICD → CARLA is `-nullrhi` (no pixels), MEASURED 2026-07-09, `DATA_STRATEGY_FOR_HIERARCHY.md` §4.2. One `vulkaninfo` probe (~1 pod-hour) would resolve whether this is still true. |
| IDM pseudo-labelling of unlabelled video | 🟥 **refuted at current design point** | `dynenc-branchB` decisive held-out-rig-transfer FAIL, MODEL_REGISTRY §10.1; see headline §1.6. `IDM_VIDEO_PRETRAIN_DESIGN.md` (2026-07-22) is the design doc this refutes; its own §5 pre-registered a cheapest discriminating experiment that appears to be exactly what branchB ran. |
| Multi-corpus mixing (`MixedWindowDataset`) | 🟡 **implemented + tested, unused by any live arm** | headline §1.1/Q1 table; `mixing.py:78`; sole real+real run stale since 2026-07-12 |
| Self-supervised pretraining (separate stage) | ❌ **does not exist**; the flagship's JEPA+SIGReg objective *is* its only "pretraining" | no masked-image/DINO-style large-pool pretraining stage found anywhere in `stack/tanitad/train/`; REF-A uses externally-pretrained DINOv2/I-JEPA instead, so the flagship's own from-scratch ViT never gets large-pool self-supervision |
| `obstacle.offline` as an auxiliary training loss | ❌ **unexplored** (distinct from the falsified use as a concatenated model *input*, headline §1.8) | no auxiliary lead/agent-occupancy head found in `stack/tanitad/models/`; `fourbrain.py:568` has a one-line stub comment ("costs (TTC proxy) attach here once the obstacle probe lands") and nothing implementing it |
| L2D as a strategic/route corpus | 🟡 **designed pilot, not executed** | headline §1's Q1 table; `DATA_STRATEGY_FOR_HIERARCHY.md` §3, §7 (ranked option #2, 3–5 eng-days, ~0.5 GPU-day, Apache-2.0); `l2d.py` exists to the episode contract but is not in any `--cache-dirs` |
| Lake curation/balancing infra (`curation.py`) | 🟡 **implemented + tested, not wired to any corpus build** | headline §1.9 |

### Q6 — see ranked recommendations below.

---

## 3. Ranked recommendations

Each row: **fix** → **defect addressed** → **expected effect** → **cost** → **risk** →
**cheapest discriminating experiment (both outcomes pre-stated)**.

### R1 — Finish wiring `win["lead"]` into the val-40 eval path and re-report LONGITUDINAL on every banked arm
- **Defect:** binding four-families rule violated (headline §1.2); the instrument is validated but
  idle.
- **Expected effect:** unblocks the single metric family the programme's own evidence says explains
  88.7% of the oracle gap (fact sheet) — turns a currently-invisible failure mode into a scored,
  comparable number across all banked arms.
- **Cost:** ~1 eng-day (per the source doc's own estimate, `2026-08-03-longitudinal-distance-keeping.md`
  §6, "backlog P0 L1") — fetch `obstacle.offline` chunks for the 40 val episodes, build `win["lead"]`
  in the runner. No re-selection, no parity risk.
- **Risk:** low. The falsifier is already named: if <20% of val windows carry a causal lead, report
  the family NOT-APPLICABLE with its n rather than silently dropping it (already specified in the
  source doc).
- **Cheapest discriminating experiment:** run the existing `four_families.longitudinal()` over the
  val-40 windows once `win["lead"]` exists; PASS = family populates with a non-trivial n and
  separated deltas between at least two banked arms; FAIL/NOT-APPLICABLE = <20% lead coverage on
  val-40, report and stop (both outcomes are informative; this is not a "maybe re-run" case).

### R2 — Run the lever-matched v1-vs-v2bal control (corpus-only contrast)
- **Defect:** headline §1.5 — no clean corpus-only comparison exists between v1 and any v2-line arm.
- **Expected effect:** either validates or refutes "bigger/balanced corpus is a lever" as a
  standalone claim, separated from the `--v2` architecture pack and `rollout_k`.
- **Cost:** one more 30k-step A40 run (v2bal data + v1's exact flags: `--speed-input
  --jerk-weight 0.02 --aux-accel --rollout-k 4`, no `--v2`) — same order of GPU-cost as v2corpus
  itself; already scoped in `PREREG_v2corpus_vs_v1.md` "What would make it a real corpus contrast".
- **Risk:** medium GPU cost, low design risk (the control is already specified, just not launched).
- **Cheapest discriminating experiment:** matched-step (5,000) ADE@2s, episode-cluster bootstrap,
  v1 vs this control, on the 19-leak-free-episode surface already established. Pre-registered
  outcomes: CI excludes 0 favoring the control ⇒ corpus is a standalone lever; CI spans 0 ⇒ v2corpus's
  earlier apparent gains were the `--v2` pack, not the data.

### R3 — Implement mirror augmentation with steer-sign flip
- **Defect:** headline §1.4 — a free ~2× data multiplier is unused.
- **Expected effect:** HYPOTHESIS — most valuable for the TACTICAL/lateral families (turn-class
  balance is the scarce class per Q4: turns are 14–28% even after deliberate rebalancing) and for
  the LATERAL family's curvature/yaw-rate error, which the binding four-families rule requires and
  which a symmetric task should improve near-for-free.
- **Cost:** low, ~1–2 eng-days: flip frames left-right, negate `steer`, negate any lateral pose/goal
  component (`y`, `LATMANEUVER` left/right, `ROUTE` turn_left/turn_right), leave `accel`/speed
  untouched. Must NOT flip anything asymmetric if it exists in a given corpus (none identified for
  PhysicalAI front-wide; would need re-checking before extending to L2D's `turn_signal`).
  **Never re-selects episodes — parity-safe by construction** (same episodes, doubled windows).
- **Risk:** low; the main risk is silently missing an asymmetric field when flipping (e.g., a future
  route/goal vector) — enumerate every field touched before shipping.
- **Cheapest discriminating experiment:** train the existing flagship config with vs without mirror
  augmentation to a matched step count on the parity corpus; primary = paired episode-cluster
  bootstrap on turn-stratified ADE and on curvature/yaw-rate error (LATERAL family). PASS = separated
  improvement on turn-stratified windows; FAIL = CI spans 0, in which case turn-count (not
  augmentation) is the bottleneck and the lever is not free after all.

### R4 — Pilot L2D as a new (non-parity) arm for STRATEGIC ground truth
- **Defect:** Q3's STRATEGIC gap (most of the v3 vocabulary is `unknown`; `nav_command` is a
  documented echo) and headline §1.1's utilization gap.
- **Expected effect:** per the existing design doc's own measurement (not mine, INHERITED but from a
  primary MEASURED table, `DATA_STRATEGY_FOR_HIERARCHY.md` §3.2): a ~10× density improvement on
  route/topology labels (e.g., 3,532 roundabout episodes vs today's 8-of-2,201-windows-from-one-clip),
  map-derived rather than kinematically guessed, plus OSM road class/lane count/speed limit and a
  real CAN turn-signal per frame.
- **Cost:** already costed in-repo: 3–5 eng-days, ~0.5 GPU-day, ~16 GB, Apache-2.0
  (`DATA_STRATEGY_FOR_HIERARCHY.md` §7, ranked option #2). **Kept as a separate arm/tier — zero
  parity impact by construction** (no re-selection of PhysicalAI episodes).
- **Risk:** low-medium — flagged in the same doc: verify the geometry gate (can L2D's front-camera
  intrinsics canonicalize to the shared `F_REF=266`?) before any training use; if not, that is itself
  a HIGH-severity, reportable finding, not a blocker to silently work around.
- **Cheapest discriminating experiment:** the doc's own §10 "cheapest falsifying experiment" —
  CPU-only, tests whether L2D's route/topology labels carry decision-relevant information before any
  head is written. I did not re-verify whether this specific experiment has been run; if not, it is
  the correct next step before committing the full 3–5 eng-days.

### R5 — Do not repeat the branchB design point; either re-scope the IDM/YouTube pseudo-labelling thesis or retire it
- **Defect:** headline §1.6 — the thesis this line depends on failed decisively, not ambiguously.
- **Expected effect of re-scoping:** if the goal is still "pseudo-label uncurated video", the
  evidence points at using the plain flagship-v1 ViT encoder (which DOES show positive cross-rig
  transfer with a converged head, +0.382/+0.657) as the substrate instead of a from-scratch
  camera-conditioned encoder — a materially different design, not a re-run of branchB at more steps.
- **Cost of the recommendation itself:** ~0 (a decision, not a build) — the cost belongs to whichever
  re-scoped design is chosen next.
- **Risk:** the risk of *not* doing this is sunk-cost continuation of a refuted design point.
- **Cheapest discriminating experiment:** before any new from-scratch encoder run, re-run branchB's
  own held-out-rig-transfer gate using flagship-v1's frozen encoder + a fresh multi-rig head as the
  substrate (cheap: no new training, only a head-fit + eval, reusing `run_branchb_transfer.py`).
  PASS (cross rig-B speed R² > 0.9) ⇒ the plain encoder is a viable pseudo-labelling substrate and
  the camera-conditioning architecture was the defect, not the thesis; FAIL ⇒ the thesis itself
  (not just branchB's architecture) is the problem and should be retired, per CLAUDE.md's
  "aim above SOTA, settle with experiments" rule (rule 5) rather than scoped down quietly.

### R6 — Execute (or explicitly re-schedule) LAN's own pre-registered E0 before assuming a goal signal helps
- **Defect:** headline §1.7 — risk of repeating the "goal input helps" assumption CLAUDE.md's
  binding goal-input rule was written to guard against, when the programme's own prior data says the
  bottleneck may be downstream consumption, not the label.
- **Expected effect:** resolves whether LAN's dense corridor is consumed at all (H-LAN-1, mechanism)
  before spending GPU on whether it improves lateral behavior (H-LAN-2, outcome) — exactly the
  discipline CLAUDE.md's Operating Standard rule 5 asks for (cheapest discriminating experiment
  first).
- **Cost:** per `PREREG_lan_refc.md` §7, described in-repo as "comes first" — I did not extract its
  exact eng/GPU cost in this pass; UNVERIFIED, re-read that section before costing externally.
- **Risk:** low — it is explicitly the cheap gate before the expensive one.
- **Cheapest discriminating experiment:** `PREREG_lan_refc.md` §6.0–6.1 (E-pre, H-LAN-1) as written;
  I recommend running it (or confirming it has already run) before any claim that "a predicted goal
  point" will move the needle on this programme specifically — the published literature figure
  (+4.7 for a goal point, cited in CLAUDE.md's binding rule) is PUBLISHED-elsewhere evidence, not yet
  MEASURED-ours evidence on this architecture.

### R7 — Promote `sc_cross.py` (intersection cross-traffic half) into `stack/`
- **Defect:** headline §1.3 — the shipped intersection detector is half of what its own docstring
  implies.
- **Expected effect:** completes the STRATEGIC/situation-label ground truth for intersections
  (currently turn-half only), which is a precondition for any honest claim about intersection
  handling in the TACTICAL/STRATEGIC families.
- **Cost:** low — the code exists and is described as already measured
  (`situations.py:65`); this is an integration task (CLAUDE.md rule 3: "escalate integration"), not
  a research task.
- **Risk:** low, given roundabout was deliberately deferred by the PI for a *different* reason
  (unpowered, 26 clusters) — cross-traffic for intersection is not the same item and should not
  inherit that deferral silently.
- **Cheapest discriminating experiment:** none needed — this is promoting already-measured code, not
  a new claim. The action item is integration + a regression test that the promoted detector's
  output matches the `incoming/` version bit-for-bit on a fixed clip sample.

### R8 — Fix the stale `96.90%` obstacle.offline coverage figure in `lead_state_gate.py:6`
- **Defect:** minor but concrete propagation of a superseded number (96.90% → 97.4438%, per
  `RETRACTION_LOG.md:36` and every later citing site) into a still-live script's docstring — the
  same failure class CLAUDE.md's rule 2 warns is easy to miss because the file "looks read" (it is
  read; the number in it is just old).
- **Expected effect:** none functional (the script doesn't compute from that literal), purely
  hygiene — but this is exactly the kind of doc-vs-registry drift CLAUDE.md's source-of-truth rule
  exists to catch before it re-propagates.
- **Cost:** trivial (one-line docstring edit — not made by me, per this stream's "report bugs, don't
  fix them" constraint).
- **Risk:** none.
- **Cheapest discriminating experiment:** n/a (a correction, not a claim).

---

## 4. Open questions / UNVERIFIED

- Which exact `predictor.horizons` tuple and window size trains the full-scale `flagship4b` config
  used by v1/v2corpus/v5f (I read `config.py:344-347`'s reduced-scale preset; did not locate and
  confirm the production preset's exact horizon tuple in the time available).
- Whether `lake/dedup.py` and `lake/filtering.py` were actually run against the parity, v2bal or
  v5f clip selections, or only exist as tested-but-unapplied lake-ingest infrastructure (same
  open question as the curation module, headline §1.9, but for dedup/quality rather than balance).
- Whether v2corpus-vs-v1's published/banked numbers use the 19-leak-free-episode val ("option A")
  or the disjoint-remainder val ("option B") — both were designed, the later document reads to me
  as adopting option A, but I did not find an explicit "option A chosen" statement.
- How often the corpus-median-intrinsics fallback (losing the per-clip rig fix) actually fires in
  the built parity/v2bal/v5f caches — the code warns once per clip but I did not find an aggregate
  count in any build log or manifest reachable from static review.
- Whether `PREREG_lan_refc.md`'s E0/H-LAN-1 experiment has already been run (its cost/timeline was
  not fully extracted in this pass) — if it has, R6 above is a "read the result" action, not a "run
  it" action; I recommend the orchestrator or a follow-up check `PREREG_lan_refc.md` §4/§6 directly
  for a completion date before re-costing it.
- The 96.90% vs 97.4438% `obstacle.offline` coverage figures may also differ because they were
  measured over different chunk subsets (26 chunks vs a fuller sample) rather than one being simply
  "stale" — I read the RETRACTION_LOG entry as a supersession, but did not independently recompute
  either number, so treat R8 as a documentation-hygiene finding, not a re-verified factual claim.

---

## 5. Deliverable manifest

| Artifact | Location | Notes |
|---|---|---|
| This report | `repo:Project Steering/Reviews/2026-09-25-programme-review/streams/R3_data_encoding_labels.md` | staged (`git add`), not committed, per this stream's operating rules |

No code was changed, no experiments were run, and no other files were edited — this stream is a
static read-only review. All claims above cite file:line or a dated in-repo document; anything
re-cited from another agent's prior work is marked INHERITED and was not independently re-run by me
this session. The two web searches used for the PUBLISHED PhysicalAI-AV corpus-size figures in
headline §1.1 are cited inline; no other external sources were used.
