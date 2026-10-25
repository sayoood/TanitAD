# PREREG_V8F_DSEFF: is the hierarchy more data-efficient than FLAT? (written 2026-10-25, before any DSEFF data)

**Status: pre-registered.** Three training arms on box 2, each behind a launch-gate PASS. They run under `PI-V8F-2026-10-25b` and the PI's 2026-10-25 go on the proposed ideas, *"our priority is to proof very quickly dominance"*. They queue after `PREREG_V8F_FLAT`'s two arms, or run on other compute if the PI allocates it.

## Why

The PI asked for dominance "regarding result, data and compute efficiency". The programme has one data point on the hierarchy alone:
* `DSCALE`, half the training clips: ADE6 +0.039 [−0.080, +0.158] m, not separated: **NOT DATA-LIMITED** (`…/2026-10-17-v8f-mini/raw/2026-10-25-dscale/run/dscale_read.json`).
* That says the hierarchy loses little when the data halves. It says nothing about whether a model **without** the hierarchy loses more. FLAT is that model (`PREREG_V8F_FLAT`: O reads constants in place of S's checkpoints and T's code; same trainer, same budget).

## The arms (ONE lever each: `--data-frac 0.5`, the trainer's existing flag, `H-DATA-SCALE-1`)

| arm | flags | seeds | status |
|---|---|---|---|
| M2b-100 | B's argv | s0, s1 | banked |
| FLAT-100 | B's argv + `--o-interface cut` | s0, s1 | queued (`PREREG_V8F_FLAT`) |
| **M2b-50** | B's argv + `--data-frac 0.5` | s0: the banked DSCALE arm **iff** its argv equals B's plus `--data-frac 0.5` and nothing else (verified by an argv diff); s1: NEW | s1 new |
| **FLAT-50** | FLAT's argv + `--data-frac 0.5` | s0, s1 | NEW |

* `--data-frac` selects clips by the trainer's own `select_train_clips` (a fixed, seed-independent subset). The optimiser-step budget is the SAME as at 100 % (`S × steps-mult`).
* The EVAL set is the unchanged eval-141, so every comparison is paired on the same windows.
* ⚠️ The 50 % subset re-selects TRAINING clips **by design**; that is the lever, and parity of the eval set is untouched.

## The reading (T1-class open loop, eval-141, the eval's own evaluate(), paired episode-cluster bootstrap, B = 2,000; F = 0.1207 m)

Data cost of an architecture A: $\Delta_A = \mathrm{ADE6}(A_{50}) - \mathrm{ADE6}(A_{100})$, per seed.

| read | definition | bar |
|---|---|---|
| **PRIMARY-INT** (the interaction) | $\Delta_{FLAT} - \Delta_{M2b}$, paired on the same windows | **> 0 and separated, at BOTH seeds**: FLAT loses more when the data halves |
| **PRIMARY-MATCH** | ADE6 M2b-50 − ADE6 FLAT-100 | **≤ 0 (point) at both seeds**: the hierarchy at half the data is at least as good as FLAT at full data |
| beside | the four families for every arm; the same reads on ADE2 and accel; H-blended versions (descriptive) | descriptive |

| outcome | condition | next (committed) |
|---|---|---|
| **DATA-EFFICIENT HIERARCHY** | PRIMARY-INT and PRIMARY-MATCH at both seeds | the data-efficiency claim is made, with its tier and estimator. Next: 25 % for both architectures, to draw the curve |
| **EQUAL DATA COST** | PRIMARY-INT not separated at either seed | the hierarchy is not more data-efficient on this corpus at this size. Reported plainly; the hierarchy's case rests on FLAT's PRIMARY (accuracy) alone |
| **FLAT MORE DATA-EFFICIENT** | $\Delta_{FLAT} - \Delta_{M2b} < 0$, separated at both seeds | reported as a negative for the hierarchy, and the T-side lever is re-ranked against it |
| **SEEDS DISAGREE** | anything else | reported; seeds before windows (`H-ESTIM-SEED-1`) |

⚠️ **Dependency:** if `PREREG_V8F_FLAT` reads FLAT BETTER, DSEFF still runs, because data efficiency is a separate question. If FLAT's instruments are not landed, DSEFF cannot launch.

⛔ No fraction, step budget, seed or bar changes after any DSEFF number is seen.
