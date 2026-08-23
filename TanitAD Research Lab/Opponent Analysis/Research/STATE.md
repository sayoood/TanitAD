# STATE — Opponent Analyzer

LAST_RUN: **run #5 — 2026-08-02 (real date; the narrative clock is RETIRED, see below).**
  Branch: `agent/opponent-20260802` (worktree `C:/Users/Admin/wt-opp-20260802`).
  **SC-13 resolved and the open-loop probe retired** + **new W-11** + **a retraction of my own run-#3
  inference** + **Orbis 2 displaces HWM as the top differentiation risk** + **the intake stall measured**.
QUALITY: complete (G-A…G-F, G-H, G-I, G-O1/G-O2 met; 15/25 searches, 1 iteration).

> **⛔ DATING CHANGE — read once.** This discipline had been dating notes on a **narrative clock** ~2.5
> weeks ahead of wall-clock (a loop artefact that runs #2–#4 perpetuated for internal consistency).
> **Retired from run #5.** Notes now carry the **real** date. Consequence: run #4's note is filenamed
> `2026-08-07` and is **OLDER** than run #5's `2026-08-02`. **Order by run number, never by filename.**
>
> | run | note filename | real wall-clock |
> |---|---|---|
> | #2 | 2026-07-24 | 2026-07-09 |
> | #3 | 2026-07-31 | 2026-07-17 |
> | #4 | 2026-08-07 | 2026-07-20 |
> | **#5** | **2026-08-02** | **2026-08-02** |
>
> **D-026 — strand debt CLEARED for this discipline.** Run #4's work was sitting unmerged on
> `agent/opponent-20260721` (2 commits ahead of the shared tip). This run branched from the tip and
> **merged it in**, so run #4's deliverables are no longer stranded. The older
> `agent/opponent-20260715` SC-13/SC-14 dedup flag is **still open** and still needs one SC-13 chosen at
> merge — but it is now moot for SC-13's *status*, which run #5 measures directly.

## This run (run #5)

- **MEASURED (G-H/G-I) — SC-13 RESOLVED; THE OPEN-LOOP ANTICIPATION PROBE IS RETIRED.**
  `sc13_probe_v5.py` on the eval-pod A40, **flagship v1** (step 30000), canonical 40-ep PhysicalAI val,
  **stride 1 → 6,444 anchors**, **n=44 BRAKE_FAR events over 15 episodes** (run #4: 23), 1,097 s.
  Two things run #4 lacked: **real-statistics vision controls** (`shuffled` = a real window from a
  *different episode*; `frozen` = this anchor's own last frame ×8) and **episode-cluster** bootstraps.
  - **Replication passed exactly.** Run #4's stride-2 anchor set is recoverable as a subset and
    reproduces **to three decimals** (held 0.723 / blind 0.653 / reactive 0.434 / informed 0.680 /
    gt_oracle 0.633) — which also proves the re-provisioned pod's `v1_modelonly.pt` is the same
    checkpoint run #4 measured.
  - **Speed-matched AUROC (stride 1):** held **0.736** · frozen **0.723** · blind **0.672** · shuffled
    **0.634** · gt_oracle 0.620 · reactive **0.455**.
  - **Pre-registered F-A (volume) did NOT fire:** `held − reactive` **+0.281**, episode-cluster CI
    **[+0.009, +0.562]**, essentially unchanged from stride 2's +0.289 at twice the events ⇒
    **run #4's in-domain positive was NOT small-n noise.** That question is closed.
  - **But the SURVIVAL condition is NOT met:** `held − blind` +0.064 CI **[−0.019, +0.162]** and
    `held − shuffled` +0.102 CI **[−0.011, +0.245]** both include 0 ⇒ **vision attribution NOT
    established.** Pre-registration means taking this reading, not the flattering one.
  - ⚠️ **The estimator decides the verdict.** On run #4's **anchor-level** bootstrap both differences
    exclude 0 and this would read "confirmed". 44 events live in **15 episodes**; they are not 44
    independent facts. Same class as the `overlapping_holdout_se` retraction in `CLAUDE.md` — **caught
    before publication this time.**
  - **The decomposition is the real finding:** of the +0.281 over the reactive floor, **≈64 % survives
    with the scene DESTROYED**, **≈32 %** is the correct **static** scene, **≈5 %** is scene motion
    (`held − frozen` +0.013, CI [−0.024, +0.050]) ⇒ **a static-frame + ego-kinematic property, not a
    rolled-forward consequence** — the opposite of the mechanism SC-13 was built to argue. Vision still
    matters for *accuracy* (2 s ADE held 1.186 m vs shuffled 1.321 vs CV 1.743), just not for this
    detector.
  - ⇒ **The only remaining test of the H15 claim is the closed loop.** No more open-loop SC-13 probing.
- **MEASURED — the scenario pipeline is stalled at intake, with evidence.**
  `stack/tanitad/eval/scenarios/` holds only `work_zone_phantom.py` + `traffic_light.py`. **SC-04
  (run #2), SC-13 (run #3) and SC-06 (run #4) are all still in `incoming/` with UNFILLED verdict
  blocks**; I **re-ran all three today: 41/41 green** (py3.13.5 / numpy 2.5.1, CPU, <0.15 s each) so the
  ask carries no inherited numbers. **H6 row corrected in the ledger (DoA 45 % → 35 %): the binding
  blocker is intake triage, not the renderer.**
- **⛔ RETRACTION (mine, run #3).** "EU political resistance to Chinese key-tech" as an EU-market-access
  weakness for Momenta/Pony is **FALSIFIED**: Momenta won a **Germany-wide KBA Level-4 permit
  (2026-07-29)**, first Chinese firm to hold one, and **Uber increased its stake** the same week.
  Root-cause class: **single-source geopolitical INFER promoted to a market-structure conclusion.**
  Removed from Momenta's profile and from WeRide's (which had inherited the same premise unsourced).
- **New W-11 — no exposure denominator.** IIHS (2026-07-31): most operators **don't report miles
  driven, so no crash rate is computable**, and there is **no standard for which incidents must be
  reported**. Pairs with W-07. **The weakness where our own measurement discipline IS the counter.**
- **W-09 strategic read INVERTS.** The NHTSA deadline **lapsed 2026-07-30 with no public fix** — and on
  that same day NHTSA **granted Zoox a commercial exemption** (paid rides, no steering wheel, **2,500
  vehicles / 2 years**, FR 07-31). Pressure moved to a **draft statute** (Mullin, "AV Emergency Response
  Coordination Act": responder protocols, 24 h hotline, NHTSA minimum standards, **city geofencing
  authority**). ⇒ **Stop narrating "the regulator will stop them."** The correct line: *the failure is
  documented, unfixed, and not a barrier to scale — so the capability is worth more than the exemption.*
- **★ Orbis 2 (arXiv 2607.15898, 2026-07-17, Brox / LMB Freiburg) displaces HWM as the top
  differentiation risk** — *"A Hierarchical World Model for Driving"*: high-level coarse-scene predictor
  + low-level conditioned generator. HWM was hierarchy off-driving; **this is hierarchy ON driving.**
  Still ours (INFER, abstract-level): representation/temporal hierarchy **not planning-time**, no planner
  over imagined futures, **no params, no compute figure, no self-monitoring**. ⇒ **H1 may never again be
  pitched as "hierarchy"; the claim is "hierarchy a planner USES, with a number attached."** Second
  pillar also moving: **CheckVLA (2607.26789)** = run-time verification with an action-conditioned world
  model ≈ H11/A9, published.
- **W-05 wedge re-verified OPEN at NVIDIA's own text** (3rd consecutive run): Alpamayo 2 Super = 32 B,
  "3× prior params", SOTA claims, **no benchmark table / latency / compute figure / Nano tier**.
  ⚠️ **AlpaGym is NOT new to us** — Tools & DevEnv logged it 2026-07-06 as "Phase-1 cloud (40–60 GB
  VRAM)"; what's new is the open release + a **single-GPU** claim that **contradicts that figure**.
- KB: **8 new dated findings.** Research note: `2026-08-02-opponent-sweep-run5.md`.

## Resource declaration (G-I)

| item | value |
|---|---|
| Resources | **Eval pod A40 48 GB** (`tanitad-eval`) — 5-arm probe over 6,444 anchors + a 2,000-draw episode-cluster bootstrap; local dev box (`venvs/tanitad`, CPU) for the 41-test re-verification and authoring |
| Wall-clock | probe **1,097 s** (~33 s/episode × 40) + analysis; ~3 h total incl. the sweep, the pod-reprovisioning detour and authoring |
| Cost | **$0** (standing pod, no new spend) |
| Why not bigger | The eval pod **was** the resource. Nothing here needs training compute — and §1 shows anchor count was never the limiting factor, so a bigger run would not have bought a different answer |
| Coordination | The pod had been **re-provisioned** for the v1-vs-v2corpus work (`/root/models/` and the comma2k19 val are **gone**) and a **v1 checkpoint relay was in flight** on arrival. The driver **polled the transfer PID**, verified the checkpoint loads, touched `LOCK.opponent-analyzer`, ran with `OMP_NUM_THREADS=6`, released the lock. A concurrent CPU-only job (`run_ctrv_readjudication.py`) from another workstream was left untouched. |

**Reproducibility (a near-miss worth keeping).** Run #4's probe scripts and its `*_windows.pt` substrate
were on the *old* eval pod and are **gone**. This run was only possible because run #4 had **banked the
scripts into the repo**. Accordingly run #5 banks **the raw substrate too** (1.3 MB) — every future
re-analysis is now free and survives the next re-provision. The comma2k19 val cache was **not** banked
and is genuinely lost from this pod, which is why §1 is in-domain only.

## `session_guard` debt list (G-F — surfaced, not owned by me)

Run at session end from `C:/Users/Admin/wt-opp-20260802`; `RESULT: PASS` after this run's commit.
Its two WARN classes are the orchestrator's sweep list, and both got **worse**, not better:

- **9 unmerged `agent/*` branches vs tip** (D-026 strand debt): `benchmarks-eval-20260802` (+3),
  `phase0-highway-dataset` (+3), `benchmarks-eval-20260721` (+2), `data-engineering-20260711` (+2),
  `pod-code-intake-20260720` (+2), `prod-opt-20260711` (+2), `data-engineering-20260710` (+1),
  **`opponent-20260715` (+1 — the SC-13/SC-14 dedup branch, open since run #3)**,
  `tools-devenv-20260721` (+1). *(My own run-#4 strand, `agent/opponent-20260721`, is no longer on this
  list — this run merged it.)*
- **26 INTAKE packages with an unfilled verdict past the age budget**, the oldest **24 days**
  (`lal-v2-anticipation`, `physicalai-r1-selection`, `models-predictor-failfast`,
  `testsuite-io-profiling`). Three of them are mine (§ above). The W33 report called this out as the
  "3rd report unfixed"; it is now the **4th**, and it has grown from 19 to **26**.

**INFER:** at 26 unverdicted packages the intake step is not a queue with a backlog, it is an unstaffed
stage. Every discipline's `Implementation/incoming/` is now write-only. This is worth a program-level
decision (assign a triager, or declare intake advisory and let agents merge behind tests), not another
per-agent escalation — which is why mine (§) asks for a `defer` as an acceptable answer.

## Recommendations logged for other disciplines (no cross-boundary writes)

- **Architecture & Inference (Wed) — top priority:** **deep-read Orbis 2 (2607.15898)** ahead of SGDrive
  and HWM; the one question is **planning-time vs representation-only** (our whole H1 positioning turns
  on it) plus any param count. Second: **Temporally Centered SIGReg (2607.26924)** — our own
  anti-collapse method, and our SIGReg readout is currently `NOT-YET-ADMISSIBLE` (rms_offdiag 0.32 >
  0.1), so it is a candidate **fix**, not just news. Third: latent-WM **identifiability** (2607.27017).
- **Benchmarks & Eval (Thu):** (a) run #4's `D = CV_fwd − pred_fwd` monitor recommendation **stands with
  a rewritten rationale** — real vs a naive decel floor in-domain (CI-separated), but **not**
  vision/imagination-driven and **unproven vs a plain ego-kinematic feature**; keep the competence
  guard; (b) **adopt the episode-cluster bootstrap over anchor-level for any AUROC on windowed
  anchors** — §1 is a worked case where the estimator flips the verdict; (c) standing: SC-06 `_extra`
  reducers; **one OOD head for SC-05 + SC-06**; **SC-05's D8 bar remains GATING for SC-06 scoring**.
- **Data Eng (Tue):** the stopped-lead tagging ask is **withdrawn** — §1 shows event count was never the
  limit. Higher value: **screen for smoke / flare / flashing-light events** (W-09/SC-06), whose
  regulatory value went **up** this window.
- **Tools & DevEnv (Mon):** **re-check AlpaGym's real VRAM footprint** against the on-record 40–60 GB
  Phase-1-cloud figure — NVIDIA now claims single-GPU scaling, and if the floor is under 48 GB our
  existing A40 becomes a closed-loop host. This is now the **highest-value unblock in the program for
  H15**, because §1 makes the closed loop the only remaining test. AlpaSim eval still open from run #2.
- **Production & Optimization (Sat):** **DriftWorld** (2607.15065, 30+ fps, 17× faster than diffusion)
  and **GigaWorld-Policy-0.5** (2607.13960, **85 ms**) — the two efficiency-axis reads of this window.
- **Orchestrator:** (a) **triage the three stalled packages or return an explicit `defer`** — 41/41
  green today; silence is the one answer that costs us the H6 row; (b) **log the Momenta retraction** in
  `RETRACTION_LOG.md` under *single-source geopolitical INFER promoted to a market-structure
  conclusion*; (c) **narrative correction — stop using "the regulator is closing in on them"**; (d)
  **H1 positioning is time-critical** (Orbis 2); (e) **W-10 scope-or-defer is unanswered for a second
  run.**

## HANDOFF / next run (run #6)

See `BACKLOG.md` P0 — reweighted after this run: **(1) unblock the intake queue** (do not author a
fifth package into a stalled buffer); **(2) the closed-loop harness question** — measure AlpaGym's real
VRAM vs the 40–60 GB figure on record; **(3) re-purpose the probe as a cross-encoder *scene-dependence*
read** (v2corpus is already staged on the pod; ~6 min/arm; feeds H4/H26 — falsifier: a flat
scene-dependent fraction across encoders retires the whole probe family); **(4) complete the Waabi
profile** (stub only, deliberately thin); **(5) W-10 re-raise, do not author**; **(6) deltas** — the one
dated item is **Pony.ai Q2 on 2026-08-18**; also whether Alpamayo 2 Super weights ship with a
params-vs-benchmark table, and the Mullin bill's progress.

Anchors (citation-graph walk): **Orbis 2 2607.15898 (new, top)**; CheckVLA 2607.26789 (new);
HWM 2604.03208; WorldRFT 2512.19133 (id now pinned); SGDrive 2601.05640; Wayve GAIA line;
NVIDIA Alpamayo/AlpaGym/AlpaSim; Momenta R7; Metis 2606.15869; DriveFuture 2605.09701;
latent-WM taxonomy 2603.09086; adjacent-domain SkyJEPA 2606.23444.
