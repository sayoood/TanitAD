# INTAKE — D2 REF-C closed-loop planner (recovery augmentation)

**What this is.** Research + design + a ready-to-run, CPU-smoked proof that a **data-efficient** covariate-
shift fix — in-envelope geometric recovery augmentation — reduces REF-C's closed-loop lane departure. It is
the open quadrant left by the program's prior closed-loop sweeps (renderer-free ∧ non-self-referential ∧
data-efficient). REF-C's deployed weights are read-only; nothing running was touched.

**Priority order banked** (a killed agent still yields value): (1) `P0_RESEARCH.md` + `DESIGN.md` — the
durable analysis and the lever; (2) `perturb.py` — the validated geometry (the load-bearing correctness
piece, `identity_target_maxerr 0.0`); (3) `PRE_REGISTRATION.md` + the FT/probe/eval scripts (CPU-smoked,
ready-to-run); (4) the measured run, when eval frees.

## The one thing that needs an owner/decision (do NOT bury it)

**Run P2 on the eval pod when `abe82f1f` (LaneKeep) frees it.** The scripts are ready and coordinate through
`gpu_lock refc-cl-improve` + `abe82f1f`'s `LOWOOD_LANEKEEP_DONE` marker. Order + cost + commands are in
`PRE_REGISTRATION.md §4`. Two gates make this cheap and safe:
- **P2a (`recovery_probe.py`) runs first, ~0 GPU, and can KILL the FT** if REF-C already recovers from a
  warped view (redirecting the planner-closed-loop effort to the controller — a program-level finding for
  free).
- **P2b** is decoder-only on a **frozen** encoder → ~1–2 GPU-h and the world model cannot be degraded.

The scripts locate the abe82f1f instrument (`lowood_lanekeep.py`) by rglob and import it verbatim; if it has
moved, put its dir on `PYTHONPATH`. No `stack/` code is modified → `pytest` unaffected.

## If eval stays busy
This intake + the pre-registration + the CPU-smoked scripts ARE the deliverable (the mission's stated
fallback). The run is a single, coordinated, pre-registered step whenever eval is free.

## Merge note
Nothing here belongs in `stack/` yet. The merge candidate is `perturb.py` + `recovery_aug_ft.py` promoted to
`stack/tanitad/train/` as a REF-C training option, gated behind a `--recovery-aug` flag (default off =
byte-identical to base `refc_train.py`), plus an AlpaSim confirmation before any deploy claim. Do not merge on
the low-OOD read alone.

## Direction-2 OUTCOME (phases 1+2 complete, MEASURED) — and what a PROMOTER runs next

**Result.** In-envelope geometric recovery augmentation is a **validated, data-efficient, generalizing**
closed-loop lane-departure lever (held-out corridor_departure halved, strongest at junctions; beats Gate-1's
memorization wall). **But decoder-only over a frozen encoder it is Pareto-bound and NOT promotable** — the
departure↓/ADE↑ trade survived the full gentle sweep (naive/g1/g2/g3) AND the return-to-GT-speed term
(g2s1/g2s2). Twice-confirmed bottleneck (P2a probe + the sweeps): the **frozen encoder does not encode the
lateral offset** (recovery_ratio ~0), so the decoder cannot cut departures without global over-reactivity.
Full numbers + Pareto frontier in `RESULTS.md` (P2b/P2c/P2d).

**The sole remaining lever = ENCODER-IN-THE-LOOP light-FT, plan-free-CANARY-GATED.** What a promoter runs:
1. **Reuse this stream's machinery verbatim** — `recovery_aug_ft.py` already builds the frozen-encoder path;
   the change is to make the encoder (last k blocks, or full at low lr) **trainable** instead of frozen. Add
   a `--unfreeze-encoder-blocks k` (or `--lr-encoder`) flag; the recovery objective + envelope + λ_dev/λ_prog
   are unchanged. Start from `g2` settings (the least-ADE-cost Pareto point): steps 700, lat_max 1.0, yaw 3°,
   clean 0.5, λ_dev 1.0, lr_head 5e-5, **lr_encoder ~5e-6** (10× lower — the warm-trunk lesson from v4.1).
2. **GATE on the operative-rollout canary** (roll REF-C's WM/predictor under TRUE actions → ADE@2s) EVERY
   eval: if the canary degrades past ~+0.05 over base, the encoder is being pulled off-manifold → stop (the
   v4/v4.1/v4.2 silent-WM-degradation hazard). REF-C's LAW aux already couples the encoder to a prediction
   target — use it, or add a light frozen-teacher feature-distillation term to pin the encoder.
3. **Pre-register the SAME predicate** (`SPEED_TERM_PREREG.md` NET WIN: held-out dCDR ≥ +0.005 S AND dADE
   CI∋0 AND peak guard AND canary intact). Then an **AlpaSim** confirmation before any deploy claim — this is
   low-OOD lane-keeping, not a safety rate.
This is a *training escalation* beyond the additive decoder-only research this stream was scoped to; it needs
a sanctioned arm + Sayed's go, not an autonomous launch.
