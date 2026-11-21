# Data Engineering — 2026-07-18

**Run:** W4 (Tuesday). **Author:** data-engineering-agent.
**Resource declaration (G-I):** local **RTX-4060 dev box** only (CPU-side geometry +
loader unit tests), **~1.6 h wall**, **$0**, 0 cloud. *Why not the eval pod / Colab:*
this run's experiment is a **loader + a closed-form geometry falsifier** — pure CPU
math validated by 19 unit tests; no model, no GPU, no real bytes needed to answer it.
The compute-heavy half (5-drive real-bytes fetch + feature precompute) is **access-
blocked** (ZOD needs a signed CC-BY-SA agreement) and shipped as a **runnable job card**
per M-3, not skipped.
**Readiness:** loader = **validated** (19 falsifier/contract tests green, measured
geometry); gap to *production* = the real-bytes `verify_real_clip` numbers (job card,
pending ZOD access) + orchestrator intake of `zod.py` into `stack/`.

---

## 0. Headline

1. **Shipped the ZOD loader — the FLEET_REVIEW #1 unlock and OWN_DATASET_PLAN's headline
   owned real-urban ingest.** Intake pkg `2026-07-18-zod-loader/` (`zod.py` + 19 tests +
   job card + INTAKE). ZOD = **CC-BY-SA-4.0**, 14 EU countries, day/night/seasons/weather,
   **real CAN steering + OxTS RT3000 ego-motion** — the exact diversity the current
   74%-straight, day-only mix lacks (the enabling condition of the ego-status shortcut).
2. **The pre-registered geometry falsifier is ANSWERED — PASS.** "Can ZOD front reach
   f_eff=266 at ≥50% observed_frac?" → **YES: f_eff=266.0, observed_frac=1.00, drop_in=
   True**, and *robust to the real KB coefficients* (the FOV alone decides it). ZOD is
   **geometrically unblocked** — unlike PandaSet (height-bound at f_eff 467). No
   escalation on geometry; the falsifier did not trip.
3. **Key reuse result: Kannala-Brandt ≡ f-theta poly.** ZOD's KB fisheye radius
   `r(θ)=f(θ+k1θ³+k2θ⁵+k3θ⁷+k4θ⁹)` IS `calib.FThetaIntrinsics.poly` (odd-power) →
   `kb_to_ftheta` reuses the proven crop path with **zero new geometry math**. This
   confirms OWN_DATASET_PLAN's "fisheye → existing `ftheta_*`" assumption with numbers.
4. **Red-suite blocker (Monday's #1) already resolved** — `test_physicalai_rig.py` is now
   tracked and `calib.py` ships all two-rig symbols; suite collects 391 tests, no error.

## 1. SEARCH — recency sweep (D-028) + anchor walk

- **[owned-tier, mine] ZOD grounded (arXiv 2305.02008 Table 2 + zod1-sdk):** front cam
  **8 MP, 3848×2168, HFOV 120°, 10 Hz, Kannala-Brandt**; OxTS RT3000 @100 Hz (pos 0.01 m,
  heading 0.1°: poses, vel/accel XYZ, angular rates, WGS84); vehicle_data @100 Hz (steering
  angle+rate+torque, pedals, indicator). HF `Zenseact/ZOD` is a code-loader (no plain
  download) → **access-gated** via the SDK + agreement. Corrects the internal landscape
  "research/NC" mis-tag → **CC-BY-SA-4.0** (3rd independent confirmation, matches
  OWN_DATASET_PLAN §3).
- **[SEAM → Benchmarks&Eval] driving world-model benchmarks surging:** **WorldLens** (CVPR
  2026 Oral, `worldbench/WorldLens`, "WorldLens-26K" human-rated realism/plausibility/
  safety) and **DrivingGen** (arXiv 2601.01528, generative-WM benchmark: visual realism,
  trajectory plausibility, temporal coherence, controllability). Both grade *generative*
  WMs; TanitAD is a *predictive* WM, but their trajectory-plausibility + controllability
  axes are exactly D2/D4 material. Handed to Benchmarks&Eval (benchmark releases = their
  seam, D-028).
- **[SEAM → Benchmarks / candidate curve-rebalance source] "A global dataset of continuous
  urban dashcam driving"** (arXiv 2604.01044). A NEW *urban continuous dashcam* corpus —
  directly on my curve-rebalance duty (BACKLOG P0 #3: move the mix off 74% straight). Flagged
  for a license/actions probe next run (is it CC? does it carry ego-motion/CAN or is it
  video-only → IDM?). Could be an owned-tier urban add or a YouTube-class copyright barrier;
  unknown until probed.
- **[watch] "Creating Impactful AD Datasets: A Strategic Guide"** (arXiv 2607.00710, 1 Jul
  2026) — meta-guidance on dataset design; useful framing for the own-dataset data card, no
  action. **ORAD-3D** (ICRA 2026, off-road) — out of our on-road scope, noted only.
- Anchor walk (LAWM/Drive-JEPA/HiLAM/IDM lineage): no new latent-action release since
  2026-07-15; the frozen-encoder IDM+WM recipe for pose-less corpora (WorldModel-Synth,
  the new dashcam set) stands unchanged.

## 2. EXPERIMENT — the ZOD geometry falsifier (measured, RTX-4060 CPU, $0)

Pre-registered (BACKLOG P0 #1): *"ZOD's front-cam geometry can't reach f_eff=266 at ≥50%
observed_frac → escalate with the measured number before building further."*

Method: build `FThetaIntrinsics` from the published spec via `kb_to_ftheta` (equidistant
KB, k=0, f_px=(W/2)/θ_max=1837.28 px), then `front_camera_canonicalization` (crop the
canonical half-angle, measure achieved f_eff through the real poly, compute observed_frac =
in-frame fraction of the ideal geometric-centered crop box). Unit-tested (19 green).

| Camera (KB) | f_px | ideal crop | achieved f_eff | observed_frac | drop_in |
|---|---|---|---|---|---|
| **ZOD front (120° repr.)** | 1837.3 | 1648 px (fits 3848×2168) | **266.0** | **1.00** | **True** |
| ZOD front (real-ish k1=−0.05) | 1780.0 | 1581 px | 266.0 | 1.00 | True |
| Narrow 40° witness | 5511.9 | 4944 px (spills frame) | 642.4 | 0.34 | False |

**Verdict: PASS — falsifier did NOT trip. ZOD is geometrically unblocked.** A 120° fisheye
crops *inward* to the canonical ~51.4° half-angle, so the crop box sits fully inside the
native frame → nothing padded → observed_frac=1.0. The result is **robust to the exact KB
coefficients** (FOV, not the k terms, sets observed_frac; k only nudges f_eff by <5%), which
is precisely why grounding on the published FOV is sufficient to answer the go/no-go while
the exact per-drive KB stays access-gated. The narrow-40° witness (observed_frac 0.34, the
Udacity-class 0.13 failure mode) proves the ≥0.5 gate is not vacuous.

Contrast (2026-07-15): PandaSet's pinhole front is **height-bound** (square crop clamps to
1080 → f_eff 467, ~1.75× off) → its ingest is blocked pending the R1 pinhole_rectify
integration. **ZOD has no such blocker** — the fisheye path already covers it.

## 3. ANALYZE — what this changes for TanitAD

- **The corpus-diversity gap now has a concrete, geometrically-cleared instrument.** ZOD is
  the one owned corpus that adds *real* night/weather/EU-urban with *real CAN + OxTS* — the
  half of the REF-B curve failure (74% straight) that PandaSet (SF day) and Cosmos (synthetic)
  can't fill. The loader is ready; only ZOD access + a big-disk fetch stand between us and the
  first owned real-night episodes.
- **OxTS heading > motion heading.** ZOD's RT3000 gives true vehicle heading (0.1°), so yaw is
  offset-free and defined at standstill — a cleaner action source than PandaSet's camera-heading
  fallback and than Cosmos's pose-derived synthetic. ZOD's real CAN steer becomes a *second*
  action channel once `can_steer_ratio` confirms a stable wheel:road ratio on real bytes.
- **The KB≡f-theta identity de-risks the whole fisheye owned tier.** Any future KB-modelled
  source (many EU datasets) drops in via `kb_to_ftheta` — no per-source geometry code.

## 4. Actionable recommendations

1. **Sayed / orchestrator: request ZOD access** (`opendataset@zenseact.com`, CC-BY-SA-4.0 +
   privacy/no-military notice) — the ONE blocker on the #1 owned real-urban ingest. Accepting
   CC-BY-SA for a *separate public ZOD shard* is OWN_DATASET_PLAN §9 open-question #1
   (recommended: accept for a public shard; the permissive core stays proprietary-capable).
2. **Orchestrator: intake `zod.py`** (additive, 19 tests green, no `stack/` module touched) as
   a ready-but-access-blocked loader — same status PandaSet got, minus the geometry blocker.
3. **On access: run the job card** (pod3-idle or Colab T4) → 5-drive `verify_real_clip` numbers
   (geometry drop-in confirmed on real KB, A8 vs comma's 0.06, steer-ratio) → data card + the
   curve-rebalance evidence (ZOD night/intersection fraction per drive).
4. **Benchmarks&Eval:** WorldLens + DrivingGen on the Phase-1 WM-eval radar (trajectory
   plausibility / controllability axes ≈ D2/D4).

## 5. BACKLOG re-prioritization

- P0 #1 (ZOD ingest) → **loader DONE + falsifier PASS**; remainder = access + real-bytes job
  card (now P0, access-blocked). P0 #3 (curve-rebalance) gains a concrete source (ZOD) + a
  probe target (the global urban dashcam set 2604.01044). P0 #4 (calib_r1 consolidation) still
  pending — note ZOD needs NO calib.py change (fisheye path suffices), only PandaSet does.
- New P1: probe **arXiv 2604.01044** (global urban dashcam) — license + actions availability.

**Falsifier ledger (P8):** geometry falsifier — PASS (not tripped), grounded + robust.
Real-bytes falsifiers (drop-in on real KB, timestamp alignment, steer-ratio stability) —
PRE-REGISTERED in the job card, pending access.
