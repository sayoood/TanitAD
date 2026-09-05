# H-REFAV1-SURFACE-1 — the paired delta and the cost-surface weight sweep

**status: IN PROGRESS — done: nothing yet (file created as the bank-early anchor) / next: PHASE 1 (zero GPU) — the decision-grade PAIRED episode-cluster bootstrap over the three banked `rec_*.json` records on ADE + the four families separately; then PHASE 2 SPEC.md (pre-registered) and the weight/goal-source arms.**

Arch+Inference FlyWheel · 2026-09-05 · checkpoint `refav1-b1-v72-ep3-speed/ckpt.pt` step **21,109** ·
**T1** (self-action open loop) for every planner arm; **T0** for `ol` (world-model diagnostic only).
⛔ Nothing in this document is driving performance (PI ruling 2026-09-02: a planner feeding its own
predictor is still OPEN LOOP).

## The question the PI asked

> *"our goal is to let refav1 drive and prove the performance of WM based architectures and also our
> hierarchy architecture."*

Two open items from `D-REFAV1-CCOS-EVAL`:

1. **Phase 1 — the paired delta.** The `ccos` refutation currently rests on PER-ARM intervals. The
   decision-grade form is the PAIRED episode-cluster bootstrap on the same 282 windows / 141 clusters.
   Does the refutation in `D-REFAV1-CCOS-ARMS` survive it?
2. **Phase 2 — the weight sweep** (`H-REFAV1-SURFACE-1`). With the goal term audible (`ccos`
   compensated), is there a weight setting at which refav1's planner beats the trivial controls on the
   four families — or is the iCEM line dead?

⚠️ **Scope statement that travels with every number below:** the world model is SOUND and separately
banked (TURN_L +0.502 [+0.383, +0.579], TURN_R −0.387 [−0.498, −0.267], separated). Everything here
is a statement about the **PLANNER over that model**, never about the world model itself.

---

*(sections land as they are measured; this file is committed after every phase and every arm)*
