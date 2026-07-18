# Curve-rebalance measured on real bytes + IDM/latent-action lit delta — 2026-07-18

**Agent:** Data Engineering (Tuesday), second increment of the day.
**Context:** the Tuesday P0#1 (ZOD ingest) was already shipped earlier today
(`agent/data-engineering-20260718` @ `5812ec2`/`2a4739c`: ZOD loader + geometry
falsifier PASS + access escalation). This note is the **chained next step**
(that run's STATE "Next #3"): the FLEET_REVIEW 2026-07-17 **P0#3 curve-rebalance**
— the data-side attack on the #1 program risk. Resource: local RTX-4060 dev box,
CPU only, ~1.5 min wall, **$0** (pure closed-form geometry + epcache pose reads;
no model/GPU/network needed to answer it).

---

## 1. The question

The fleet review named the top risk (single-camera driving-capability gap) and its
**enabling condition**: a training corpus that is *mostly straight* (cited ~74%;
nuScenes 73.9%). A straight-dominated mix is satisfiable by the **ego-status
shortcut** — "keep going straight at v0" — so the world model never has to learn
action→consequence on the turns that exercise steering. The loss-side half of the
REF-B curve failure shipped in `refbpatch`; this is the **data-side half**:

> Measure the straight/gentle/sharp distribution **per source, on real bytes,
> using the exact D1 eval convention**, then derive a turn-weighted sampling recipe
> that moves the mix toward ~55–60% straight — and quantify what the incoming
> urban corpora (ZOD/PandaSet) actually buy.

**Stratum convention (identical to `stack/scripts/driving_diagnostic.py`, so the
data-side numbers are directly comparable to the D1 per-stratum ADE):**
`|net yaw change over the 2 s / 20-step horizon|` → straight `<5°` · gentle
`5–20°` · sharp `>20°`. Copied verbatim into `curve_rebalance.py`; a unit test
asserts the constants still equal the eval script (drift guard).

## 2. Measured result (630 real episodes / 125,247 windows, local epcache)

| Source | episodes | windows | **straight** | gentle | sharp | regime |
|---|---|---|---|---|---|---|
| comma2k19 | 130 | 36,270 | **83.1%** | 10.5% | 6.4% | CA-280 highway, day |
| PhysicalAI | 500 | 88,977 | **56.0%** | 23.4% | 20.6% | urban/interactive |
| **combined (natural window pool)** | 630 | 125,247 | **63.9%** | 19.6% | 16.5% | — |

### Findings
1. **The straightness is a comma2k19 / HIGHWAY property, not a whole-corpus one.**
   comma alone is **83.1% straight**; PhysicalAI urban is **56.0%** — already inside
   the 55–60% target. Turn-richness is exactly what the urban corpus supplies.
2. **The "~74% straight" fleet figure reconciles to a comma-dominant mix.** At
   source weights (comma / PhysicalAI): 0.30/0.70 → 64.2%; 0.50/0.50 → 69.6%;
   0.65/0.35 → **73.6%**; 0.70/0.30 → **75.0%**. So "74%" corresponds to a comma
   ≈0.65–0.70 weighting — the highway anchor dominating. The **natural window pool
   is only 63.9%** (PhysicalAI has 2.45× comma's windows locally).
3. **Two independent, quantified levers to reach 57.5% straight:**
   - **Source-mix (primary):** every +10 pp of comma weight adds ≈ +2.7 pp
     straight. Shifting weight off comma toward urban (PhysicalAI now; **ZOD +
     PandaSet** next — the OWN_DATASET_PLAN §6.2 ~35% owned-urban share) is the
     biggest knob. This is the *quantified rationale for the ZOD ingest*: a
     ~56%-straight-class real EU-urban corpus directly dilutes comma's 83%.
   - **Window turn-weighted sampling:** a `WeightedRandomSampler` keyed by each
     window's stratum with weights `{straight:1, gentle:β, sharp:β}`.
     Single knob `β = s(1−t)/(t(1−s))` for measured straight `s`, target `t`:
     **β = 1.31** at the natural pool → **2.22** at a comma-0.70 mix. Verified by
     construction (round-trip unit test).

### Interpretation for the program
- The data-side fix is **cheaper than feared**: the pool is 63.9%, not 74%, so a
  modest β≈1.3 (or a small comma downweight) reaches 57.5% without starving the
  straight regime (straights still dominate absolute count).
- **Do NOT over-upweight**: pushing much below ~55% straight would under-sample
  the highway regime where CV is near-unbeatable and the D1 straight-stratum bar
  lives — the target band is 55–60%, not "as few straights as possible".
- **This is a training-recipe change (D-018 tactics) → ESCALATE before flipping
  the live trainer.** This package ships the measurement + recipe + a tested
  sampler-weight function; the actual sampler/mix wiring is a separate proposal
  gated on Sayed (flagship-v2 retrain). Filed as intake `2026-07-18-curve-rebalance/`.

## 3. Falsifier / honesty (P8)
- **Pre-registered check:** if PhysicalAI turned out *more* straight than comma,
  the "urban buys turns" thesis (and the ZOD rationale) would be falsified. It did
  not — 56.0% vs 83.1%, a 27 pp gap in the expected direction. ✔
- **Sampling caveat:** these are the *locally cached* comma (Chunk-1-class) + the
  R0/R1 500–2000-clip PhysicalAI selection, not the full corpora — the per-source
  *shape* is robust (tens of thousands of windows each) but the absolute mix ratio
  is whatever the trainer sets, so the recipe is parameterized by the live weights,
  not hard-coded. The β formula holds for any measured `s`.
- **Definition-locked:** numbers use the eval owner's exact strata; a drift-guard
  test fails if `driving_diagnostic.py` changes them.

## 4. Literature delta (SEARCH step — new since the 2026-07-14 KB entries)

Scout sweep (arXiv cs.CV/cs.RO/cs.LG + HF datasets, ~last 2 weeks). External
support only — **no hypothesis status change (P8)**; these seed the H7 IDM loop
(G2) and the intrinsic-canonicalization line (D-016/H17).

- **Sensorimotor World Models: Perception for Action via Inverse Dynamics**
  (arXiv 2606.20104) — a latent WM trained with an **IDM regularizer** that stops
  collapse and forces latents to preserve the action behind each transition.
  Most on-point for pseudo-labeling speed/yaw out of pose-less driving video (G2,
  the WorldModel-Synth/YouTube path).
- **ACID: Action Consistency via Inverse Dynamics** (arXiv 2607.02403) —
  cycle-consistency: an action re-inferred by an IDM must match the conditioned
  action. A cheap per-clip **pseudo-label quality gate** for the H7 bridge (the
  "action-agreement r" our G2 target already anticipates).
- **Latent-WAM** (2603.24581) / **DriveWAM** (2605.28544) — driving-specific
  latent world-action modeling from video priors (the AV analog of LAPA/AdaWorld);
  check the action-tokenizer for the REF arm.
- **What Do Latent Action Models Actually Learn?** (2506.15691) — diagnostic of
  LAM failure modes (latents leaking non-action info) — a caution to read *before*
  trusting any latent-action pseudo-label (protects G2 from a silent-bias trap).
- **X-Lens: Real-Time Metric Depth with Heterogeneous Cameras** (2607.12993) —
  **intrinsic-guided normalization to a canonical reference camera** + calibration
  tokens to absorb heterogeneous resolution/lens. The closest recent work to our
  `f_eff=266` canonicalization (D-016) and the H17 unified-FOV direction — worth a
  design read even though it targets depth, not driving.
- **Datasets/benchmarks (→ Benchmarks & Eval seam, D-028):** *A global dataset of
  continuous urban dashcam driving* (arXiv 2604.01044, monocular pose-less — a
  curve-rebalance / IDM candidate, license+actions TBD, queued P1); *DrivingGen*
  (2601.01528) and *ScenePilot-4K* (2601.19582) generative-WM / VLM-driving
  benchmarks; *LiAuto-DriveAction* (HF, driving-action labels, small eval ref).

## 5. Provenance
- Analyzer + tests + INTAKE + report JSON: intake `2026-07-18-curve-rebalance/`
  (`curve_rebalance.py`, `tests/test_curve_rebalance.py` 12✓, `run_analysis.py`,
  `curve_rebalance_report.json`).
- Stratum definition source: `stack/scripts/driving_diagnostic.py`
  (`CURV_STRAIGHT_DEG=5`, `CURV_GENTLE_DEG=20`, `K_MAX=20`).
- Data: local epcache `C:/Users/Admin/tanitad-data/{comma2k19,physicalai,eval}`.
- Prior increment today: `2026-07-18-zod-loader-and-geometry-falsifier.md`.
- Fleet directive: `Project Steering/FLEET_REVIEW_2026-07-17.md` §P0#3;
  `Data Engineering/OWN_DATASET_PLAN.md` §6.2. Lit sources linked inline in §4.
