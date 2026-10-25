# PREREG_V8F_KINDX: once O consumes partial codes, which kind of T's code is worth fixing? A seed-0 screen of KIND on MIXO's checkpoint (written 2026-10-25, before any KINDX data)

**Status: pre-registered.** Zero training. KIND's own oracle evals, run on X = `M2b-mixo_s0` (MIXO's seed-0 checkpoint, sha256 `c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d`), on the dev-box CPU under `PI-V8F-2026-10-25d`. It is the re-read that `PREREG_V8F_KIND` commits to after its O-side arm ("this instrument re-reads its result"), and the T-side step `PREREG_V8F_MIXO` names on FAIL, O LEARNED.

## Why (MEASURED)

* **KIND on B (`RESULT_V8F_KIND.md`, T0): INTERACTIVE.** GT in one kind of cells alone was worth little or made O worse (tte −0.24 / −0.32 m, geom −0.43 / −0.42 m); additivity A = −0.30 / −0.55. O could not use a partly right code, so the single-kind gains were lower bounds that could not rank T-side levers.
* **MIXO s0 (`mixo_read_s0.json`, re-derived by hand): FAIL, O LEARNED.** Own-code ADE6 X − B +0.022 [−0.059, +0.101] (not separated); but O on X now uses partial codes: r(0.5) 0.644 against B's 0.365.
* ⇒ On X the single-kind gains are no longer forced to be lower bounds. They can now rank which kind of T's code is worth a T-side arm.

## The read (X = M2b-mixo_s0 only; T0; paired episode-cluster bootstrap, B = 2,000, 141 episodes, 2,399 windows)

KIND's six kinds and cells, unchanged (`PREREG_V8F_KIND.md`, `CONS_NAMES`): speed, angle, tte, timegap, geom, leadgap; and the union (all 26 cells).

| read | definition | role |
|---|---|---|
| **gain_k(X)** | ADE6_X(T's own code) − ADE6_X(GT in kind k only, T's code elsewhere) | **decides** |
| beside | gain_union(X); additivity A_X = Σ_k gain_k / gain_union; necessity_k(X); gain_k(X) − gain_k(B s0) paired (does O on X use kind k more than B does?); the four families | descriptive |

| outcome | condition (s0) | next (committed) |
|---|---|---|
| **CANDIDATE** | some kind k has gain_k(X) ≥ 2F = 0.2415 m and separated | train X at seed 1 on box 2 (MIXO's argv, `--seed 1`, behind its launch gate) and re-read gain_k at s1. k is a CARRIER only if it clears 2F at s1 too (KIND's both-seed rule). A carrier gets a T-side arm aimed at it, under its own PREREG, along KIND's routing: SPEED → T's speed variance; ANGLE, GEOM or TTE → the encoder (LoRA); TIMEGAP or LEADGAP → lead perception. Several candidates are ranked by gain_k(X) and all go to the s1 read |
| **NO CANDIDATE** | no kind clears 2F at s0 | single-kind T-side arms stay unrankable even with an O that consumes partial codes; X is not trained at s1 for this purpose. The longitudinal line (KSCORE, then OACC) remains the top lever |

One seed decides only whether a second seed is worth training. ⛔ No claim about a lever is made from s0 alone (`H-ESTIM-SEED-1`).

## Controls (must read their known values before any gain is read)

1. X's own code (`--tac pred`) reproduces the banked `eval_M2b-mixo_s0.npz` per-window ADE6 to ≤ 1e-4 m (CPU float32 against box 2's GPU).
2. The union oracle reproduces the banked `eval_M2b-mixo_s0-tacorc-constraints.npz` to ≤ 1e-4 m, and the full GT code the banked `eval_M2b-mixo_s0-tacorc.npz`.
3. An identity oracle (GT in no kind) gives gain exactly 0.0 per window.

⛔ No kind, cell, bar, estimator or routing changes after any KINDX number is seen. Tier T0: a diagnostic of how O consumes T's code, never a driving claim. The four families are reported beside; tactical and strategic are identical across the oracle arms by construction (the oracle replaces only the code O reads).

## Compute

Dev-box CPU (`CUDA_VISIBLE_DEVICES=-1`), 10 evals of X (own code; six kinds; union; full GT code; the identity oracle) over the eval-141 windows; ESTIMATED 30–60 min. Box 2 is untouched (KSCORE runs there).
