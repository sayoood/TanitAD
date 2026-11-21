<title>Frontier Scan 2026-09-13 — a hierarchy control, a version artefact, and a trunk that ties pixels</title>

# Frontier Scan — 2026-09-13 (eighth pass, LAB-RUN-012)

`TanitAD Research Lab · DAILY_RESEARCH_CHARTER.md + v2 amendment · branch agent/arch-inf-20260803 · staged, never committed`

**Load-bearing full texts (V-2), named:** `2604.03208` HWM (29 pp) · `2506.04218` v3 Pseudo-Simulation (the one that reversed a finding) · `2609.10464` SG-JEPA (v1, 2026-09-09) · `2508.02047` MVV. Plus `2608.30144` (A4), `2404.14027` OccFeat (B13), `2606.07170` TOAD tables.
**V-1 (library before web):** 4 of the 8 deep-read primaries were already banked (`2604.03208`, `2608.30144`, `2506.04218`, `2606.07170`). ⚠️ The check ran **after** the scan queries but **before** every deep read — and it is what caught F2's version artefact.

---

## F1 — Band D: Waymo *"A look under our trunk: what's in our compute"* (2026-08-20), adjudicated in 7 steps

`PUBLISHED-BLOG · Jeyachandran (VP Eng) & Rosenband (Compute Lead) · read via WebFetch summariser (short quotes only) · register rows A17-1…A17-3`

**Step 1 — document and context.** A hardware doctrine post published **two weeks before Tesla's Cybercab launch (2026-09-03)** and paired in the press with Waymo's sensor-sufficiency argument (*"Cameras are incredible, but they aren't enough"*, Thirumalai, relayed by TechCrunch 2026-09-01). Competitive purpose: defend the cost of a multi-sensor, custom-silicon stack. ⚠️ **The 10-lessons post this press cycle cites is ALREADY adjudicated** (W-1…W-13, 2026-08-31); only the compute post is new.

| claim | step 2 — experiment? | step 3 — confirming (independent) | step 4 — contradicting (sought) | step 5 — verdict | step 6 — binds on us? | step 7 — guideline |
|---|---|---|---|---|---|---|
| **A17-1** a custom 5 nm ASIC with **>1,000 TOPS dedicated to front-end processing** is what L4 needs | ⛔ **No.** A spec sheet; no ablation against a non-custom stack, no latency or accuracy number | trade press restates it (DesignNews, TipRanks) — **not independent** (relays of the post) | ⭐ **TOPS is a synthetic-workload proxy:** an accelerator with **7.5× the TOPS of a GPU delivered ~4× the network throughput** (RELAYED, aiMotive / Hailo engineering blogs); `1911.02987` (not banked, scan) on misleading accelerator benchmarks | **UNSUPPORTED-AS-STATED** — TOPS cannot carry a capability claim | ⛔ **No** as hardware (we deploy on Thor); ⭐ **yes as a measurement rule** | **T-8 (tactical):** never quote TOPS; every deployment number is **measured ms at a named percentile on the target**, which `quant_gate.py` (row 1) already requires |
| **A17-2** *"pixels-to-actuation latency"* is optimised **across every percentile** | ⛔ No numbers, no percentiles published | ✅ `2305.07147` **COLA**: tail latency in a real L4 stack is safety-critical through latency aggregation across pipelined modules (lib, abstract-only); `2209.05487` (scan): 99th-percentile DNN inference outliers near 4,000 ms under concurrency | none found that argues tail latency is irrelevant (2 queries, EMPTY) | **CONFIRMS-US** | ✅ **Yes** — identical to **D10-2** (gate on `Tpeak`, not `Tavg`; Drive-HWM 84.8 ms avg / 107.2 ms peak) | **S-6 (strategic):** our latency claims are **worst-case claims**; an average in a deployment table is incomplete |
| **A17-3** L4 compute must be **two independent engines** that normally share load and fail over | ⛔ No — engineering doctrine | fail-operational requirement for L4 in the ISO 26262 literature (`2011.00892` formally verified fail-operational concept, scan; PatSnap engineering blog RELAYED) | ⭐ **the common alternative is an ASYMMETRIC failover** — a primary plus a less powerful unit running only safety-critical functions (patent literature as summarised by search, e.g. US 11281547 *"Redundant processor architecture"* — **RELAYED, not read**) ⇒ full duplication is one design, not a necessity | **SUPPORTED** (as industry practice), **not** as the only design | ⚠️ **Not now** (a research reference implementation on one Thor) — **but** it is a P6 requirement the day we claim deployability | **S-7 (strategic):** *"small at deployment"* gains a second argument — **a sub-300 M model can fit on the degraded half of a fail-over pair**; state it in the paper only after we measure it on half a Thor |

⭐ **THE CONCESSION.** *"…especially in the low-batch regimes we often operate"* — Waymo concedes its real operating point is **batch ≈ 1**, where TOPS is least meaningful. That is **our Thor measurement** (throughput flat 12.3–14.1 windows/s across a 6× batch range, saturation at batch 8). The largest opponent operates where headline accelerator numbers say least.
⭐ **OPPONENT STRENGTHS (recorded, not comfort):** a production custom ASIC; two-engine redundancy already shipping; compute integrated into vehicle liquid cooling (Phoenix heat to Midwest winters); *"20× compute in eight years"*; **200 M fully autonomous miles**; 14 cities with riders as of 2026-09-01.

### F1b — Band D: Tesla Cybercab (2026-09-03), claim A17-4 — *camera-based end-to-end networks are the core of L4*

`RELAYED (trade press; no Tesla primary read) — adjudicated because the claim is doctrine, marked as the weakest-sourced row in the register`

* **Step 2 — experiment?** ⛔ **Attribution-from-exposure** twice over: *"billions of miles of human driving data"* and *"more than 380,000 unsupervised miles … no notable incidents"* (RELAYED). No counterfactual.
* **Confirming:** NHTSA SGO filings mid-June → mid-July 2026 — two incidents, both with the robotaxi stationary and another driver at fault (RELAYED via EV press). **Contradicting (sought):** 17 Austin incidents Jul 2025 → Mar 2026 at a **5.9 mph** average speed; a remote-operator crash in Houston (Electrek, 2026-07-20) — ⚠️ advocacy press on both sides.
* ⭐ **CONCESSION (RELAYED, needs a primary):** press describes FSD 14.3.3 on Cybercab as a **"vision-plus-radar stack"** — if true, the vision-only flagship ships with radar.
* **Verdict: CONTESTED.** **Binds on us: NO** — our vision-only rule is an *admissibility* rule against leakage, not a sufficiency claim; a fleet mileage number neither supports nor threatens it (same separation as W-1).
* **Guideline:** S-1 again — exposure is not evidence; **T-9 (tactical):** read the NHTSA SGO primary before this row is quoted anywhere.

## F2 — A5: the 51.3-vs-56.6 PDM-Closed "contradiction" was two versions of one paper ⭐⭐⭐

`PUBLISHED lib 2506.04218 v3 (2026-03-20) FULL TEXT, v2 (2025-08-27) via HTML · lib 2606.07170` → package `Benchmarks & Evals/Research/2026-09-13-pdmc-stage2-provenance/`

Stage 1 identical in 9/9 subscores; Stage 2 moved in 7/9 between the **08/2025** and **03/2026** leaderboard snapshots; v3 Table 2 equals TOAD. **BE10-1's committed "different versions" branch fires ⇒ D-10 CLOSES.** ⛔ **Self-correction:** 09-10 quoted v2's 51.3 while the Library had held v3 since 08-23 — **LI10-1's failure class, reached a ⛔-blocking row.**

| dim | |
|---|---|
| RELEVANCE ⭐⭐ | G3 comparability; row 32; row 3 |
| CONSEQUENCE | every external EPDMS carries a **snapshot stamp**; row 32's same-table sentence (56.3 vs 56.6) is admissible |
| COMBINATION | the Stage-1 fingerprint is a reusable admission test; `H-ESTIM-SEED-1`'s "name the substrate" logic |
| CHANCES / RISKS | 0-GPU closure / the board can re-score again |
| EXPERIMENT | `E-BE-S1S2-1` — fingerprint every external EPDMS we hold (committed branches in the package) |

## F3 — A1: HWM `2604.03208` — the capacity-controlled hierarchy result row 18 needed, on a FROZEN encoder ⭐⭐⭐

→ package `Architecture & Inference/Research/2026-09-13-hwm-capacity-matched-hierarchy/`. Table 16: flat 98 M **35 % / 15 %** vs hierarchy 94 M **78 % / 61 %** (Push-T); flat 178 k 82/59 vs hierarchy 182 k 95/83 (Maze). ⛔ No interval, ±4 % matching, waypoint-data confound. **A10-2 fires branch 1**; A10-1 becomes `E-ARCH-HWM-1` with a *flat + waypoint-data* control. ⭐ **The first lever this month compatible with the trunk-freeze.**
Scan neighbours: **SV-WAM `2609.03602`** (surround-view world-action model, future video as dense supervision — scan only); **`2609.03225`** (long-horizon interaction-aware WM + GRPO, multi-style driving — scan only).

## F4 — A2 × B11: SG-JEPA `2609.10464` — a physical parameter as an ACTION COORDINATE ⭐⭐

`PUBLISHED lib 2609.10464 · FULL TEXT (v1, 2026-09-09, Liu, Sun, Baker, Balestriero, Sous) · verified against the banked PDF`

* Gravity `g` is **concatenated to the control** (`ã_t = [u_t; g]`, constant per episode). Rollout loss K = 5, γ = 0.95, SIGReg, **no stop-gradient**, Muon/AdamW. Relative to DINO-WM: **31–48 %** lower 3-D prediction error; Arm Catcher Ball capture **9.5 % → 23.3 %**; five seeds for control.
* ⭐ **Their own attribution: the gain is the ENCODER.** Swapping fresh predictors onto each *frozen* encoder, *"the GRU-trained encoder lowers mean rollout error by about 12 %"* (1.376 vs 1.555; 1.269 vs 1.453) regardless of predictor.

| dim | |
|---|---|
| RELEVANCE ⭐⭐ | B11 transfer (dynamics priors); the max-speed input channel design; the trunk-freeze |
| CONSEQUENCE | gives the **input form** for a posted limit if a supplier ever exists: an **episode-constant action coordinate**, exactly as `g` — no new pathway |
| COMBINATION | ⭐ **This is our own historical lever, independently rediscovered:** *speed as a third action channel* took REF-A from 3.73 → 0.83 m. SG-JEPA generalises it to *physics as action*. ⛔ **And it is the sixth this month to locate the gain in the encoder** (LDAD, AITS, IQL-encoder-loss, Qwen-Drive, OccFeat, SG-JEPA) — the frozen trunk forfeits it |
| CHANCES / RISKS | chance: a friction/weather/limit coordinate at near-zero cost; risk: one scalar (gravity) in toy physics — their own limitation |
| EXPERIMENT | **`E-B11-SGJ-1` (tiny ladder):** add a **per-episode scalar coordinate** that is admissible at inference (candidate: the day/night flag, which is vision-derivable) to the action vector of the v7-tiny predictor, frozen trunk. **Committed:** Δ ≥ 3 % on the LONGITUDINAL family with a shuffled-coordinate control reading within noise ⇒ the channel form is validated for the max-speed input; otherwise the form is not the bottleneck and the supplier question stands alone |

## F5 — B13: OccFeat `2404.14027` — FS9-9's supervision question answered, and the answer has changed since we asked it ⭐⭐

`PUBLISHED lib 2404.14027 · v3 · read (HTML + PDF table verified)`

Supervision = **LiDAR occupancy** (*"a voxel occupied if it contains at least one Lidar point"*) + DINOv2 features on occupied voxels. nuScenes, SimpleBEV EN-B0 vehicles IoU: **+10.6 at 1 % labels, +4.9 at 10 %, +0.3 at 100 %**.

| dim | |
|---|---|
| RELEVANCE ⭐⭐ | FS9-9, GS-4, S-3; today's A&I BEV-LiDAR head |
| CONSEQUENCE | ⭐ **FS9-9 is answerable: YES, we can now produce OccFeat's supervision** — the A&I FlyWheel built a LiDAR BEV ground truth over 139 PhysicalAI clips *today* (`…/2026-09-13-bev-lidar-corpus-and-head/`). The line S-3 redirected to is **unblocked on data**. |
| COMBINATION | ⛔ **But OccFeat is a PRETRAINING loss on the ENCODER** — the seventh encoder-side lever. Its gain is largest exactly in the low-label regime the A&I head is in (82 train clips, frozen-trunk AP 0.414, `main − pixel` not separated) |
| CHANCES / RISKS | chance: the published fix for a head that ties pixels on 82 clips; risk: its advantage vanishes at 100 % labels (+0.3) and it needs the trunk unfrozen |
| EXPERIMENT | **`E-B13-OCCF-1`:** OccFeat-style auxiliary head during a **short partial unfreeze** (last stage only) on the 139-clip LiDAR GT, then the A&I head re-run frozen. **Committed:** `main − pixel` lower CI > 0 ⇒ the tie was the trunk, fixable by a stage unfreeze; still tied ⇒ the 82-clip data volume, not the trunk, binds (A&I lever L4 decides) — ⛔ needs an MM decision on the freeze |

## F6 — A4: *Rethinking Language's Role in Efficient VLA for AVs* `2608.30144` ⭐

`PUBLISHED lib 2608.30144 · read (survey)` — **Language Residue** taxonomy: **L1 train-time-only** (zero inference cost) · L2 latent reasoning · L3 conditional invocation · L4 per-frame generation. Relayed cost frame: one onboard autoregressive VLM call **500–2,000 ms** vs a **20–100 ms** control loop (Huang et al. 2026 — RELAYED, not banked). ⭐ **TanitAD is an L1 system by construction** (Alpamayo CoT → v7 labels, never at inference). **Positioning, not a lever.** EXPERIMENT: none new — BLUE `2606.08684` (banked, L3 gate 0.11 M params, 2.54× speed-up) stays the L3 reference if a language path is ever added.

## F7 — B1: fine-grained sign recognition is ~50 % accurate at best ⭐ → DE package

`PUBLISHED lib 2508.02047 FULL TEXT` — MVV T3 signs: **DINOv2 0.484 vs InternVL-3-9B 0.467**, no interval; the abstract's *"consistently outperforms"* rests on 1.7 pt here. Fine classes are sign **types**, not **values**. Serves the standing max-speed question (DE package F8).

## F8 — Band C sweep

| track | item | class |
|---|---|---|
| C1 | Waymo riders in **Denver, San Diego, Tampa** (2026-09-01) → **14 cities**; Tesla **Cybercab launch** Austin 2026-09-03 (closed event); Waymo compute post (F1) | PUBLISHED-BLOG / RELAYED |
| C1 | NVIDIA **NuRec 26.04 = 1,607 PhysicalAI clips** (26.01 = 918) — the `map.xodr` carrier set; **PhysicalAI-AV-NCore** (~1.1 k clips, calibration/egomotion/cuboids, no map) | PUBLISHED-RELEASE-NOTE (HF cards) |
| C2 | ⭐ **TensorRT 11.2.1 does NOT support JetPack / Thor** — JetPack 7.2.1 still ships **TensorRT 10.16.2**; no JetPack 7.3. ⇒ **FS-4 stays gated a third pass**, and a TRT-11 feature must never be assumed on Thor | PUBLISHED-RELEASE-NOTE (NVIDIA support matrix, relayed via search) |
| C3 | **UN ADS Regulation + GTR adopted by WP.29, 2026-06-24** — safety-case approval, certified SMS, in-service monitoring & reporting, **DSSAD**, performance *"at least at the level of a competent and careful human driver"*; software updates via UN R156 (RELAYED, 2 secondaries). **NHTSA AV Framework interim exemption guidance — comment period extended to 2026-09-30** (Federal Register 2026-08-31, PUBLISHED). ⛔ **D-4 still unread at THREE MORE routes** (UNECE doc page 403; UNECE press release 403; direct `ECE-TRANS-WP.29-GRVA-2026-02e.pdf` via curl 403) — eight routes total | mixed, stated per item |
| C4 | Bench2Drive leaderboard last updated **2026-08-28**; navhard 03/2026 snapshot table (11 entries, one substrate) now held — F2 | RELAYED / PUBLISHED |

## Coverage honesty

**22 / 22 tracks carry a named query** (search log). **DEEP:** Band D ✅ (2 items, 7 steps, counter-searches recorded, strengths named) · A1 ✅ · A2 ✅ · A4 ✅ (survey) · A5 ✅ · **A3 ⚠️ scan only** (empty for a September-2026 encoder release, probe 1 of a varied term) · **Band B: B1 ✅, B11 ✅, B13 ✅** (three) · Band C ✅ C1–C4 · ≥2 full texts ✅ (four) · ledger appends ✅.
⚠️ **Not done:** the Tesla row rests on RELAYED sources only; the regulatory primary (D-4) is still unread; B2/B3/B4/B6/B7/B8/B9/B10/B12 are scan-only.

## Recommendations (≤3 for the pass)

1. ⭐⭐⭐ **Re-specify A10-1 as `E-ARCH-HWM-1`** (flat / flat + waypoint data / hierarchy at ±1 % params) — the external positive now exists and the missing control is named.
2. ⭐⭐ **Put the trunk-freeze on the MM agenda with SEVEN measured exclusions** (F4, F5 and today's Deployment tie are three more) — and note that HWM's hierarchy is the one lever that survives it.
3. ⭐⭐ **Adopt LI10-1 today** — two version artefacts in four days, the second reached a ⛔-blocking row.
