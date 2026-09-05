# REF-A v1 — MAKE IT DRIVE

**STATUS: IN PROGRESS — Step 1 (zero-GPU decision-rule sweep) starting**
**Agent:** TanitAD Architecture & Inference FlyWheel
**Date opened:** 2026-09-05
**Branch:** `agent/arch-inf-20260803`
**Register rows (to be filled):** `D-REFAV1-DRIVE-*`

---

## The objective (PI, verbatim, 2026-09-05)

> *"our goals in TanitAD programme is not to refute hypotheses, it's about achieving excellent
> results and really driving autonomously with a reference implementation. Why I'm saying this,
> we have too many refutes, and I have the feeling, we are satisfied sometimes if the hypothesis
> is refuted rather than solving the problem"*

⇒ **A refutation is not a deliverable. A number that moves is.**

**Win condition:** a decision-rule setting or a goal-head fine-tune under which refav1's four
metric families beat `ha` and `ha0_ext` at T1 with the echo gate holding.

---

## What is already MEASURED (inherited, not re-derived)

Sources: `…/Research/2026-09-05-refav1-ccos-eval/RESULT.md`,
`…/Research/2026-09-05-refav1-surface-sweep/RESULT.md`; register rows `D-REFAV1-CCOS-ARMS`,
`D-REFAV1-SURFACE-PAIRED`, `D-REFAV1-SURFACE-SCREEN`, `H-REFAV1-SURFACE-1`.

| fact | value | class |
|---|---|---|
| World model turn separation | TURN_L +0.502 [+0.383, +0.579]; TURN_R −0.387 [−0.498, −0.267] | MEASURED (inherited) |
| `W_JERK` inert | 24/26 candidates jerk ≡ 0; 9 settings over 1e13 identical | MEASURED (inherited) |
| `cv` wins argmin | 50.7 % of windows at **all 59** weight settings | MEASURED (inherited) |
| GT turn rate | 46.8 % of windows | MEASURED (inherited) |
| Decoded-goal turn proposal | 20.5 % of GT-turn windows | MEASURED (inherited) |
| Direction correct when it commits | 77.8 % [0.625, 0.923] | MEASURED (inherited) |
| Token histogram | LANE_KEEP 244, TURN_L 7, TURN_R 31, five tokens never emitted | MEASURED (inherited) |

⇒ **RECALL problem in the goal head.** Recall problems have cheap fixes before expensive ones.

---

## Plan (cheapest first, STOP when it drives)

- **Step 1 (ZERO GPU, today):** sweep the goal-head DECISION RULE on banked dumps —
  argmax vs temperature vs commit-threshold vs class-prior correction. Measure
  turn-proposal recall, direction precision, then the four families + ADE through the
  actual planner. Controls: current argmax must reproduce 20.5 % / 77.8 % exactly;
  random-token control must read chance; print `n` and `d`.
- **Step 2 (cheap, only if 1 insufficient):** fine-tune ONLY the goal head vs v7.2
  tactical labels with a recall-oriented loss, trunk + WM frozen (asserted by
  parameter count, never assumed).
- **Step 3 (only if 1 and 2 fail):** state plainly what a full retrain must change and
  what it costs; escalate to Master Mind rather than launching.

**Stage B bound:** `/home/nvidia/refav1_ccos/stageB.log` on Thor — the `cl_oraclegoal` arm
bounds the prize. If a perfect goal clears the controls, Steps 1–2 are worth every hour;
if not, the ceiling is elsewhere and that is reported immediately.

⛔ Canonical `ha0_ext` is the INTEGRATOR `refav1_arm.hold_ext_controls`, never
`echo_gate.ha0_ext` (they differ by 0.540642 m at 2 s — Decisions §M11).

---

## Log

| time (Europe/Berlin) | event |
|---|---|
| 2026-09-05 ~14:15 | Package opened, skeleton banked. Local 4060 busy (RL arm, 76 % / 3329 MiB) — Step 1 is CPU-only by design. |

---

## Deliverable manifest

| artifact | location | status |
|---|---|---|
| This RESULT.md | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-make-it-drive/RESULT.md` | banked |
