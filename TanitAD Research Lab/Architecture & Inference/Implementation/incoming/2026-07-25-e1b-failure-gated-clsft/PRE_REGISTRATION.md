# E1b — failure-gated closed-loop SFT for the REF-C anchored-diffusion planner

**Pre-registration — written and staged BEFORE the fine-tune was launched.**
`2026-07-25 (Europe/Berlin)` · `tanitad-pod3` (A40, idle) · renderer-free
imagination/kinematic closed loop (NOT AlpaSim). PI: Sayed.

Evidence-class legend (CLAUDE.md operating standard): `MEASURED` (ours + artifact
path) · `PUBLISHED` (cited) · `INHERITED` · `ESTIMATED` · `HYPOTHESIS`.

---

## 0. Why E1b (the two experiments that license it)

`MEASURED` (`…/2026-07-25-closedloop-horizon-and-shift/E1a_E2a_RESULTS.md`, commit `2d6589b`):

- **E1a fired.** REF-C base closed-loop **corridor-departure** goes `0.0035` (K=20 / 2.0 s)
  → **`0.5877`** overall / **`0.8414` at junctions** (K=185 / 18.5 s); paired Δ(K185−K20)
  **+0.5842 [0.5071, 0.6565]** SEP, OOD-peak ratio ≤ 1.30 (genuine in-distribution failure,
  not extrapolation). The 2 s instrument on which every standing closed-loop number was measured
  **hid the failure by ~170×**. (`e1a_horizon_heldout44_K185.json`.)
- **E2a = PERCEIVABLE.** The lateral offset **is** in the representation (oracle linear R²
  **0.7176**, ceiling ρ **0.9112**); **91.1 %** of the recovery loss is **downstream** (the planner
  ignores information it has), and it is **neither truncation** (0.01 %, more denoise steps don't
  help) **nor conditioning** (0.11 %). ⇒ the lever is the **TRAINING OBJECTIVE**, not the encoder
  and not inference-time denoise steps. (`e2a_localize_heldout44.json`.)

**E1b hypothesis** `HYPOTHESIS`: a failure-gated closed-loop SFT that supervises the anchor SCORE
head toward the *return* anchor at recoverable pre-failure states will force the planner to USE the
offset it already perceives, lowering long-horizon corridor departure **without** regressing
open-loop ADE. Renderer-free, R2LPL-shaped. `PUBLISHED` precedent: R2LPL (nuPlan Test14-hard
60.67 → 83.51, renderer-free score-learning on failure-adjacent states).

---

## 1. Substrate (all `MEASURED`, `probe_substrate.json` on pod3, this dir's SUBSTRATE.md)

| item | value |
|---|---|
| REF-C base ckpt | `tanitad-pod3:/workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt`, step **29999**, strict-load 0 missing/0 unexpected, **128** anchors `[128,4,2]`, 104,191,577 params |
| mining source (train) | `…/pai_epcache/physicalai-train-e438721ae894` — the **sacred parity corpus**, **2376** episodes |
| held-out EVAL set | `…/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6` — **44** episodes (E1a's exact eval set) |
| **leak guard** | parity-train (2342 distinct ids) ∩ held-out eval (44 ids) = **0 — DISJOINT (byte level)** |
| harness | E1a `e1a_horizon.py` rollout body reused VERBATIM (`import e1a_horizon`) |
| estimator | episode-cluster bootstrap `taniteval/ci.py`, B=2000, **paired** for two arms on identical windows |

**The split that matters:** we fine-tune (mine failure states) on **parity-TRAIN**
`e438721ae894` and evaluate on **held-out** `heldout-79d4e3d2d4c6`. These are physically different
caches with **0** episode-id overlap (re-verified at train startup by
`--assert-disjoint-heldout`, which REFUSES to train on any overlap). The leaky split
`physicalai-val-f1b378f295ae` (78.5 % into parity train) is **NOT** used anywhere.

---

## 2. The intervention (fixed before launch)

Fine-tune REF-C base (DiffusionDrive-style, 128 anchors, 2 denoise steps) with two interleaved
objectives per step, **encoder FROZEN** (E2a: perception is not the bottleneck; freezing directly
targets the identified downstream lever and protects open-loop ADE + the perceived offset):

1. **CL-SFT (mined failure states).** Roll REF-C base closed-loop to **K=185** on parity-train
   (`e1b_mine.py`). Mine the **recoverable pre-failure** states — |XTE| in-corridor (< 1.75 m) but
   heading out, in the `h_pre=10`-step lead-up before the first corridor crossing; **exclude
   already-departed states**. The R2LPL target is the logged corridor path ahead of the nearest
   reference expressed in the **current offset ego frame** (the return-and-follow demonstration);
   supervise **anchor-cls CE toward the nearest ("return") anchor + traj-recon L1** on it — the
   open-loop anchor block of `refc_train.compute_losses` with the recovery target substituted for GT.
2. **REPLAY (open-loop forgetting guard).** The full `refc_train.compute_losses` (traj+cls+law+
   route+man, **v21 labels** — the set base trained with) on parity-train windows, interleaved.

`loss = 1.0 · cl_loss + 1.0 · replay_loss`; Adam lr 2e-5, warmup 100, cosine, 4000 steps,
cl-batch 32 / replay-batch 32. (Weights/steps are the pre-registered defaults; if changed at
launch the LAUNCH_CONFIRMED.md records the exact command.)

---

## 3. PRE-REGISTERED OUTCOMES (both committed here, in advance)

**Primary metric.** Junction corridor-departure-rate **@ K=185** on the held-out 44 episodes
(E1a's exact eval set → paired & apples-to-apples), **paired episode-cluster bootstrap**
(`taniteval/ci.py`, B=2000, resampling episodes). Reported overall + junction + longitudinal.

- **SUCCESS (significant).** FT junction departure@K185 is **CI-separated LOWER** than base
  (paired Δ(FT−base) `hi < 0`), **AND** the guardrails hold:
  - (a) **open-loop ADE@2s** Δ(FT−base) CI **includes 0 or better** (no CI-separated open-loop regression);
  - (b) the plan-free / open-loop canary block (open-loop anchor-cls + traj on parity val) unchanged.
- **BOUND / FAILURE (equally publishable).** Junction departure@K185 **not CI-separated** from base,
  **OR** open-loop ADE@2s **regresses CI-separated-worse** (the CL-SFT bought closed-loop at the cost
  of open-loop — a real finding; the replay branch is designed to prevent exactly this, so its
  failure is informative about the objective's tension).

**Secondary (reported, non-deciding):** overall & longitudinal departure@K185; closed ADE@2s@K185;
the **K=20 (2 s) departure** for both arms — if the FT *raises* the 2 s departure while lowering the
18.5 s one, that trade is stated explicitly, not hidden.

**Estimator discipline.** Every interval is the **episode-cluster bootstrap**, paired for the
two-arm deltas. The deprecated `overlapping_holdout_se` (1.28–2.06× too narrow) is **never** used.
No interval is quoted without its estimator.

**On the "WM-canary".** REF-C is a *direct* anchored-diffusion planner, not a world-model rollout
policy; it has no deployed operative imagination rollout to canary. For REF-C the operative/plan-free
guardrail is the **open-loop anchor-cls + traj-recon on held-out** (guardrail b above). This
interpretation is flagged for Sayed rather than silently substituted.

---

## 4. Falsifiability & honest bounds (stated in advance)

- The closed loop is **map/agent-free** (drift/stability, not collision/off-road safety) — unchanged
  from E1a. A win here is a corridor-keeping win, not a certified-safety claim.
- The recovery target is a **kinematic demonstration** (logged corridor in the offset ego frame), not
  a renderer rollout; it is the cheapest renderer-free proving ground, per the brief.
- K=185 is the structural ceiling on this 190–199-frame corpus; 20 s is impossible here (E1a §1.4).
- Mining draws from **600 parity-train episodes** (~1 window/episode at K=185; the buffer size is
  reported in `mined_buffer.meta.json`). This is a mining-set size, **not** a re-selection of the
  parity TRAIN corpus for any cross-arm comparison — parity is preserved; the eval is on the disjoint
  held-out set.

---

## 5. Deliverables (paths in DELIVERABLE_MANIFEST / report)

`e1b_mine.py` (mining) · `e1b_clsft.py` (CL-SFT) · `e1b_eval.py` (paired verdict, later) ·
`run_e1b.sh` (launcher) · `e1b_probe_substrate.py` (P0 gate). Runnable copies on
`tanitad-pod3:/workspace/e1b/`; staged copies in this repo directory.
