# STATE — Opponent Analyzer

LAST_RUN: run #4 — narrative 2026-08-07, **real wall-clock 2026-07-20** (Fri scheduled run).
  Branch: `agent/opponent-20260720` (worktree `C:/Users/Admin/wt-opponent-20260720`).
  **First MEASURED SC-13 test on our own checkpoint — falsifier FIRED on cross-corpus replication
  (negative result, P8)** + SC-06 emergency-scene intake (16✓) + new W-10.
QUALITY: complete (G-A…G-F, G-H, G-I, G-O1/G-O2 met; loop used 7/25 searches, 1 iteration —
  the real news window was only **3 days**, so budget went to compute, not search).

> **Clock note (honesty, memory `narrative-clock-ahead-of-wallclock`):** this discipline's narrative
> clock runs ~2.5 weeks ahead of real time (a loop artefact). Runs are keyed by **run number**
> (this = run #4), not by note date. Notes/packages are dated **2026-08-07** to keep folder names,
> module docstrings and this STATE internally consistent; the real calendar date is **2026-07-20**.
>
> **Orchestrator dedup flag (D-026), STILL OPEN from run #3:** the unmerged off-schedule branch
> `agent/opponent-20260715` authored SC-13 (commit `787671a`) and SC-14 speculatively. **At merge, pick
> ONE SC-13; do not integrate both.** This now *blocks* wiring the new `live-measured` SC-13 row cleanly.

## This run (run #4)

- **MEASURED EXPERIMENT (G-H/G-I) — the first non-oracle scenario number this discipline has produced.**
  `sc13_real_probe.py` on the **eval pod** (A40 48 GB) against **flagship-30k** (step 29999), canonical
  40-ep held-out PhysicalAI val, window 8 / stride 2 → **3,241 anchors**. Signal =
  `D = CV_forward(2 s) − pred_forward(2 s)`; arms = informed (true future actions — **leaks**) / **held**
  (last observed action repeated — the real test) / **blind** (held + vision replaced by a mean frame) /
  **reactive** (−Δv/0.5 s kinematic floor).
  **BRAKE_FAR** (braking starts 2–3 s out = **outside** the 2 s rollout; n=23 events vs 1,283 cruise),
  after controlling a genuine speed confound (events 8.94 m/s vs cruise 17.34 m/s) by per-event ±1 m/s
  matching **and** v0-stratification: **held 0.723 / 0.740** (raw 0.821, boot-CI [0.702, 0.917]) ·
  **blind 0.654 / 0.685** · gt-oracle 0.633 / 0.668 · **reactive 0.434 / 0.450**. `held` **exceeds the
  true 2 s trajectory's own score**, so it is not merely tracking the near future.
- **…AND THE FALSIFIER FIRED ON REPLICATION.** The same probe on **comma2k19** (64 eps, **8,384
  anchors**, n=45 BRAKE_FAR, 22.6 min) **contradicts** it: speed-matched **held 0.538 / 0.605 ≈ blind
  0.608 / 0.549 ≈ reactive 0.588 / 0.549 — mutually indistinguishable.** **We may not claim a measured
  consequence-forward-model advantage.** Two confounds keep it from being a clean refutation (INFER):
  comma2k19 is **out-of-domain** — there **CV beats the model outright** (CV 1.302 m vs held 1.874 m
  ADE), and a "deficit vs CV" signal is unreliable by construction where the model loses to CV — and it
  is **highway-dominated** (cruise 29.1 vs 17.3 m/s), the regime where CV is near-unbeatable. Both n's
  are small: the negative is as noisy as the positive. **SC-13 → `live-measured (falsifier fired)`;**
  its oracle collision-rate contrast is now **unsupported** and must stay out of external narrative.
  **Net: evidence moves AGAINST the open-loop form of the H15 claim** → strengthens the case for
  prioritizing the closed loop over more open-loop probing.
- **Caught error, recorded (P8):** the first probe version fed **true future actions** and scored AUROC
  1.00 — command leakage, not anticipation. Any future "our model anticipates" claim must state its
  action conditioning explicitly.
- **SC-06 / W-09 emergency-scene scenario shipped** — intake pkg
  `Implementation/incoming/2026-08-07-emergency-scene-scenario/` (**16/16 tests**, 4060 CPU, <1 s, $0).
  Incursion rate **0.0 vs 0.2**, blockage **0.0 s vs 2.54 s** (12.7 s at thick smoke), detection lead
  **+5.70 s vs +2.84 s**. Mechanism: obscurant collapses **object** range 90→13.5 m while **scene**-level
  OOD range falls only 80→68 m. **The failure is a CLIFF, not a slope** ⇒ graded obscurant sweeps are
  mandatory. **SC-06 → spec-drafted, with a BLOCKING CONDITION**: it depends on the same OOD detector
  SC-05 measures, and SC-05's D8 probe is currently **failing** — SC-06 must not be scored until it clears.
- **New W-10 (fleet-scale mission/energy/network-disruption blindness)** from the **Waymo 2026-07-04 SF
  breakdown** (64 vehicles retrieved, depleted batteries, unplanned closures, one occupied car drove over
  a lit firework). **Marked `no-counter-yet` — including for us**: our strategic brain is the only layer
  that could own this and nothing is specified or measured. **SC-08 evidence FACT-upgraded.**
- **W-09 is now CROSS-OPERATOR:** **Zoox recalled 105 vehicles** after a robotaxi drove into thick smoke
  from an active fire, failed to recognize it, panic-braked and halted inside the scene. Also fuses
  W-09 with W-04 (smoke = obscurant *and* emergency cue → one shared OOD head).
- **Correction (P8):** the NHTSA end-of-July first-responder deadline is for **presenting fixes in
  meetings**, not deployed fixes. Do not overstate it in the vision deck.
- **Field scan:** **HWM (arXiv 2604.03208)** deep-read — **planning-time hierarchy over multi-timescale
  latent world models, 3× less planning compute** — i.e. our H1 claim is now published (on manipulation/
  maze, no params, no self-monitoring). Also the closest published relative of the **v3** direction.
  New watch item: **WorldRFT**.
- Deltas: Wayve **$85 M employee tender** (liquidity, not capital); Pony reaffirms **>3,500 robotaxis /
  20+ cities** (W-06 unchanged); NVIDIA — Alpamayo 1 = **10 B** open weights, 2 Super = **32 B**
  "this summer", **still no Nano-tier CNCE number**.
- Ledger: H15 (real-data evidence, partial) / H11 (SC-06 blocked on SC-05) / H1 (differentiation
  pressure) / H0/H6 (cross-operator W-09; W-10 gap). **No status upgrades** — nothing closed-loop, and
  the one real-data result is explicitly under-powered (P8).
- KB: 6 new dated findings. Research note: `2026-08-07-opponent-sweep-w5.md`.

## Resource declaration (G-I)

| item | value |
|---|---|
| Resources | **Eval pod A40 48 GB** (`tanitad-eval`) — probe, 2 full passes + speed-matched re-analysis + a cross-corpus comma2k19 pass; local RTX-4060 box (CPU) for the SC-06 oracle + tests |
| Wall-clock | 309 s + 349 s (PhysicalAI) + **1,359 s** (comma2k19, 8,384 anchors) + re-analyses + <1 s (oracle); ~1 h incl. authoring/iteration |
| Cost | **$0** (standing pod, no new spend) |
| Why not bigger | The eval pod **was** the resource used. Nothing here needs training compute — the binding constraint is **event count in the val corpus**, not FLOPs. |
| Coordination | `results/LOCK.opponent-analyzer` touched; GPU idle (0 MiB) at start; training pods untouched. |

Pod artefacts: `/root/taniteval/sc13_real_probe.py`, `sc13_speedmatch.py`;
`results/sc13_flagship30k{,_comma}{,_speedmatched}.json` + `*_windows.pt` (raw substrate — re-analysis
needs no model re-run). In-repo archive with a README on how to read the arms:
`Opponent Analyzer/Implementation/sc13-real-probe/`.

**Process note (keep):** this run's headline **reversed** between the first corpus and the replication.
Had the session ended after the PhysicalAI pass, STATE would have claimed a positive measured H15
result. The replication cost 23 min of idle pod time. **Single-corpus results must not leave this
discipline.**

## Recommendations logged for other disciplines (no cross-boundary writes)

- **Benchmarks & Eval (Thu):** (a) `D = CV_forward − pred_forward` is a cheap, label-free monitor
  feature that beat a kinematic floor at 2–3 s lead **in-domain only** — if adopted, adopt it **with a
  competence guard**; it is unreliable on any corpus where the model loses to CV (it did on comma2k19).
  **Do not wire it unconditionally**; (b) **blockage-duration +
  incursion-rate** reducers over SC-06 `_extra`, and **unify SC-06's `non_nominal_detected` with the
  SC-05 OOD head — one detector, not two**; (c) treat the **SC-05 D8 bar as GATING for SC-06 scoring**,
  not informational; (d) [standing] min-TTC + collision-rate reducers over SC-13 `_extra`.
- **Data Eng (Tue):** the stopped-lead ask **changed shape** after the replication failed — raw event
  count is no longer the top need, **in-domain** event count is. Priority: (1) more **PhysicalAI** val
  episodes / denser sampling; (2) **true stopped-lead tags** so the label stops being "any deceleration"
  (curves + traffic lights currently pollute it); (3) only then cross-corpus volume.
  Also screen for **smoke** / flare / flashing-light events
  (W-09) — the Zoox recall makes smoke the highest-value visual cue in the corpus.
- **Architecture & Inference (Wed):** **deep-read HWM (2604.03208) — top priority.** Planning-time
  hierarchy over multi-timescale latent world models with 3× less planning compute, from the DINO-WM
  lineage: simultaneously the closest published competitor to **H1** and the closest published prior art
  for **v3**. Second: SGDrive (2601.05640), still open from run #3.
- **Tools & DevEnv (Mon):** CARLA emergency-vehicle / flashing-light / flare / cone assets **+ a smoke
  volumetric or photometric overlay** for SC-06's `carla_recipe()`; **AlpaSim** evaluation still open
  from run #2.
- **Orchestrator:** (a) triage the SC-06 intake; **the run-#3 SC-13 dedup verdict is still outstanding**
  and now blocks wiring the `live-measured` SC-13 row; (b) **decide on W-10** — mission-feasibility
  (energy / network disruption / fleet self-interference) is `no-counter-yet` **for us as well**: scope
  it into Phase 0 or record an explicit deferral, but do not let it be narrated as a differentiator;
  (c) narrative: "emergency scenes are not edge cases" is now backed by **two operators** (Waymo + Zoox)
  — but the end-July deadline is for **meetings, not fixes**.

## HANDOFF / next run (run #5)

- Priorities (see `BACKLOG.md` P0): (1) **resolve the SC-13 contradiction** — the question is no longer
  statistical power but whether the in-domain positive was domain-specific or an artefact: more
  **in-domain** PhysicalAI events (stride 1, more episodes) **and** the probe on an arm whose ADE
  **beats CV** on the target corpus (if anticipation appears exactly where the model beats CV, it is a
  competence artefact, not a capability); (2) add
  **shuffled-real-frame** and **temporally-frozen** vision controls (the mean-frame `blind` arm may
  understate vision); (3) port the probe to REF-A dyn-in / REF-B v2 / REF-C-XL (~6 min/arm — a cheap
  cross-encoder anticipation read that also feeds H4/H26); (4) SC-06 → CARLA-executable with a **graded**
  obscurant sweep; (5) author the tractable single-vehicle W-10/SC-08 slice; (6) deltas: end-July NHTSA
  meeting outcomes, new recalls, Alpamayo 2 Super params table.
- Anchors (citation-graph walk): **HWM 2604.03208 (new, top)**; WorldRFT (new); SGDrive 2601.05640;
  Wayve GAIA line; NVIDIA Alpamayo/Cosmos/AlpaSim; Momenta R7; Metis 2606.15869; DriveFuture 2605.09701;
  latent-WM taxonomy 2603.09086; adjacent-domain SkyJEPA 2606.23444.
