# INTAKE — planner↔WM coupling by gradient surgery (one-sided PCGrad at the seam)

- **Discipline:** Architecture & Inference
- **Date:** 2026-07-23
- **Slug:** `2026-07-23-planner-wm-gradient-coupling`
- **Stream:** GradCouple (`a4fde3c6`, pod-free design)

## What

1. **`DESIGN.md`** — the mechanism: replace v4's blunt scalar `lam_mult` floor with a **one-sided PCGrad
   projection at the `states` seam** (the single activation the planner uses to touch the trunk). Cites
   the adapted method (PCGrad, Yu et al. NeurIPS 2020; GradVaccine fallback), the exact hook points, the
   cost, and an honest §7 on where it fails.
2. **`PRE_REGISTRATION.md`** — a ~1.3 A40-day discriminating experiment (arm S surgery + arm C₀ λ=0
   control; reuse v4.2 as the scalar baseline), **all three verdicts committed in advance**, plus a
   near-free Phase-0 cosine pre-probe that can kill the experiment for ~0 GPU.
3. **`grad_surgery.py`** — reference implementation: `seam_project`, `deconflict` (one-sided PCGrad /
   GradVaccine), `_SeamProject` autograd Function, diagnostics. Pure-torch, no tanitad imports, CPU smoke.
4. **`tests/test_grad_surgery.py`** — 9 cases.

## Why

The scalar planner→trunk gradient knob has been mis-tuned **three times** (`MEASURED`): v4 hot (canary
1.30+), v4.1 starved (WM 0.4599 healthy but planner ade 0.8522 FAIL), v4.2 degrade (0.72–0.86). A scalar
trades planner-signal against WM-integrity **globally**; the conflict is **directional** and lives in one
shared `[B,8,2048]` activation (`train_flagship_v4.v4_loss_step`: `states` feeds both `flagship_loss` and
`head`). One-sided PCGrad removes only the planner-gradient component that **opposes** the WM gradient,
leaving the WM gradient untouched — the principled alternative to floor-tuning AND to a from-scratch
restart.

## Evidence / tests run

- `python grad_surgery.py` → smoke: `fwd_identical:true`, `wm_grad_untouched_maxabs:0.0` (one-sided
  proven), `noop_when_aligned_maxabs:0.0` (strict no-op when tasks agree), acts under conflict, finite.
- `pytest tests/test_grad_surgery.py -q` → **9 passed** (venv `C:/Users/Admin/venvs/tanitad`, torch
  2.11.0). CPU, < 2 s, $0.
- No training pod touched; no run launched (pod-free design stream).

## Proposed target location in `stack/`

- **`grad_surgery.py` → `stack/tanitad/train/grad_surgery.py`** (pure module; add the two tests to
  `stack/tests/test_grad_surgery.py`).
- **Splice into `stack/scripts/train_flagship_v4.py`**: in `v4_loss_step`, fork `states` via
  `seam_project` before `flagship_loss`/`head`/`goal_head` (see `DESIGN.md` §3/§5, ~8 lines); in
  `_training_loop`, log `_SeamProject.last_diag`; in `build_parser`, add `--coupling {scalar,seam,seam+floor}`
  (default `scalar` = byte-identical to today), `--seam-per-sample/--seam-global`, `--seam-target-cos`.
  One line in `preflight_asserts`: `seam` requires `cond_imagination:false` (or the §7.3 second seam).

## Risk / rollback

- **Risk:** low and opt-in. `--coupling scalar` stays the default and is byte-identical to the current
  seam, so merging cannot perturb v4.2b or any existing arm. The projection is a strict no-op whenever the
  planner and WM gradients agree (proven byte-exact in tests). It is a **refinement of the existing O-20
  lever (the λ_plan seam), not a third encoder-touching lever** — the door stays closed; state this on the
  launch card.
- **⚠️ Do NOT apply the trainer splice while v4.2b is training off `train_flagship_v4.py` on pod2.** Queue
  it for the next v4.x launch Sayed approves.
- **Rollback:** delete the module + tests + the CLI branch; the scalar path is untouched.

## Verdict (orchestrator writes here)

- **Verdict:** _pending_
- **Date / by:** —
- **Reason:** —
