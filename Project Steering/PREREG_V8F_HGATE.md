# PREREG_V8F_HGATE: the constraint-aware form of HBLEND, shrinking only where T itself predicts no constraint (written 2026-10-25, before any HGATE data)

**Status: pre-registered.** Zero training. Eval-only and offline, from banked outputs, on the dev box's CPU, under `PI-V8F-2026-10-25b`. It is the committed next step of `PREREG_V8F_HPROJ` (outcome CONSISTENCY NOT RESTORED).

## Why (MEASURED; `raw/2026-10-25-hblend/`, `raw/2026-10-25-hproj/`)

* **HBLEND (H) PASSED:** ADE6 H − B = −0.207 / −0.070 m and ADE2 −0.103 / −0.078 m. It COSTS CONSISTENCY(a): 'any' +0.101 / +0.075, separated.
* **HPROJ** (H + the posted-limit projection) **FAILED:** 'any' +0.054 / +0.031 against a bar of ≤ +0.02, and the ADE gain was lost.
* **Where the cost lives** (per rule, HPROJ §7–9):
  * SPEED against a cap of min(posted limit, GT v_hi). v_hi binds in 2,084 of 2,399 windows, and a posted-limit projection cannot see it;
  * then LEAD (active in 1,027 windows), STOP and TURN.
  * Diagnostic (post hoc, no bar): T's own predicted v_hi takes 'any' to +0.023 / +0.019 when used as a cap.
* ⇒ H's plan keeps speed where the human slowed or followed. The information about WHERE that happens is in T's own predictions: its follow decision, stop and turn goals, and predicted v_hi.

## The committed wording, and how this PREREG makes it concrete

HPROJ committed: *"the same closed form, fitted only on training windows with no active GT constraint, plus O's predicted stop / lead constraints in the projection."*
* **The fit** follows that verbatim, with one stated detail. "Active GT constraint" means the eval's STOP, TURN or LEAD rule being active on the window. SPEED is active on every window (2,399 / 2,399), so including it would leave nothing to fit.
* **"In the projection" becomes a GATE.** A LEAD constraint cannot be projected at inference: the model predicts no lead trajectory, only gap and time-gap cells. The deployable equivalent is to use B's plan wherever T predicts that constraint. Using B there is never a regression on that window.

## The arm (ONE lever on top of HBLEND: the constraint-aware fit and gate)

**Fit.** HBLEND's closed form, per step and axis: clip [0, 1], isotonic, on the same training-window set (k = 32, 141 clips). It is restricted to windows where the GT STOP, TURN and LEAD rules are all inactive. Labels may use GT; this is the fit, not inference. This gives $w^{G}$ per seed.

**Gate** (inference: model outputs only, vision-only). Use $\tau_G = \mathrm{up}(P10) + w^{G} \odot (\tau - \mathrm{up}(P10))$ unless ANY of the following holds, in which case use $\tau_B$:
1. T's longitudinal argmax is FOLLOW (`LON_FOLLOW = 3`);
2. T's goal probability for STOP (`STOP_I = 6`) is ≥ 0.5;
3. T's goal probability for TURN_L or TURN_R (indices 1, 2) is ≥ 0.5;
4. $\tau_G$'s speed exceeds T's own predicted `v_hi_ms` at any step with t ≥ 2.0 s, the SPEED rule's band.

**Identity controls:**
* with the gate forced closed everywhere, the plan is $\tau_B$ exactly (Δ = 0.0);
* with the gate forced open and $w^{G}$ replaced by HBLEND's banked $w$, the plan is H exactly;
* the fit restricted to all windows reproduces HBLEND's banked weights to 1e-12.

## The reading (T1-class open loop, eval-141, paired episode-cluster bootstrap, B = 2,000; F = 0.1207 m; the eval's OWN `evaluate()`)

| read | definition | bar |
|---|---|---|
| **PRIMARY-C** | CONSISTENCY(a) 'any': HG − B | **upper 95 % bound ≤ +0.02, at BOTH seeds** |
| **PRIMARY-K** | ADE6 HG − B and ADE2 HG − B | **both < 0 and separated, at BOTH seeds** |
| beside | the gate's open share (overall and per trigger); CONSISTENCY(a) per rule and (b); accel MAE; distance keeping; the four families; the dominance row against P10 (ADE2, ADE6, accel) | descriptive |

| outcome | condition | next (committed) |
|---|---|---|
| **PASS** | PRIMARY-C and PRIMARY-K at both seeds | HG is the deployed v8F output. The weights and the gate become part of `forward_infer` in the next trained arm, and MIXO and FLAT are re-scored with their own fitted HG (descriptive) |
| **CONSISTENCY NOT RESTORED** | PRIMARY-C fails | T's own constraint predictions are not good enough to tell where the human constrains: the T-side ceiling again (`D-V8F-T-ATTRIB-1`). Next: no further post-processing arm. The lever moves to T, after MIXO and FLAT read |
| **GAIN LOST** | PRIMARY-C passes, PRIMARY-K fails | H's gain lives in the constrained windows. Report it. Next: the same T-side ruling |

⛔ No threshold (0.5, 2.0 s), trigger, weight or bar changes after any HGATE number is seen.
