# PREREG_V8F_KSCORE: a trained scorer that reads the scene and picks one of O's longitudinal hypotheses (written 2026-10-25, before any KSCORE data)

**Status: pre-registered.** One scorer-training job on box 2. B stays frozen; only a small head trains. It runs behind a launch gate under `PI-V8F-2026-10-25b` / `-25c`. It is `PREREG_V8F_KHYP`'s committed next step (outcome HEADROOM, NO SELECTOR).

## Why (MEASURED; `raw/2026-10-25-khyp/READ.json`, eval-141, paired episode-cluster bootstrap)

* **HEADROOM** (T0, oracle best of 7 longitudinal one-hot codes through O): ADE6 −0.423 [−0.466, −0.383] / −0.359 [−0.407, −0.315] m against B. **The alternatives O can produce are good enough.**
* **SEL-LEARN** (logistic regression on plan statistics and T's longitudinal softmax, fitted on 141 training clips) − B: −0.098 (s0, separated, short of −F) and +0.020 (s1, not separated). It picks one class on 89 % of s0 windows. Its accuracy against k\* is 0.513 / 0.358, against 0.504 / 0.356 for the majority class.
* **Post hoc, no bar:** at s0, "always class 5" gives −0.0996. Control 3 (permuted labels) also separates at s0. ⇒ The s0 gain is a **constant bias correction of T's longitudinal decision, not selection**, and the best constant differs by seed (class 6 at s1, −0.019).

⇒ Selection must be learned from what distinguishes the windows, the scene among it, on far more data than 141 clips. It must also be measured against the **best constant class**, so that a bias shift cannot pass as selection.

## The arm

**Scorer $g$** (about 0.1–0.3 M parameters). Inputs, all from B's frozen forward at inference, so vision-only, with no situation-classifier output:
* O's 7 hypothesis plans $\tau_k$ (the 0.2 s plan, ego-relative);
* T's longitudinal softmax;
* B's adapted scene tokens at "now" (mean over the 40 tokens of each of the 4 frames);
* the ego token.

Output: a softmax over k = 0..6.

**Training:**
* Cross-entropy to $k^\ast$ = argmin over k of ADE6($\tau_k$, GT), on the **full training set** (4,512 clips; the trainer's own windows).
* AdamW, lr 1e-3, 3 epochs, batch 256, fixed in advance; no tuning on eval.
* Seeds 0 and 1 follow B's seeds (s0 scorer on B s0, s1 on B s1).

**Deployed plan:** $\tau_{\hat k}$, $\hat k = \arg\max g$.

**Baselines (all on the same eval windows):**
* B (own code);
* SEL-ARGMAX (T's own longitudinal argmax, one-hot);
* **SEL-CONST:** the single class with the lowest mean ADE6 over the TRAINING windows, chosen per seed before eval.

**Launch gate** (scorer form):
* G-DVB on every scorer flag;
* G-LIVE: cross-entropy finite; the scorer's gradient non-zero; B's parameters receive **exactly 0.0** gradient (frozen);
* a deliberate regression: with the scene input zeroed, a test asserts the scorer's scene path is dead.

## Controls

1. A scorer forced to T's argmax reproduces SEL-ARGMAX exactly (Δ = 0.0).
2. A scorer trained on permuted $k^\ast$ must not separate from SEL-CONST.
3. B's own-code ADE6 reproduces the banked 2.1753 / 2.0546 m.

## The reading (T1-class open loop, eval-141, paired episode-cluster bootstrap, B = 2,000; F = 0.1207 m)

| read | definition | bar |
|---|---|---|
| **PRIMARY-B** | ADE6 KSCORE − B | **≤ −F and separated, at BOTH seeds** |
| **PRIMARY-SEL** | ADE6 KSCORE − SEL-CONST | **< 0 and separated, at BOTH seeds** (selection, not a bias shift) |
| **SECONDARY** | speed MAE, accel MAE, CONSISTENCY(a) 'any', LEAD-VIOL: KSCORE − B | not separated-worse at either seed |
| beside | four families; the scorer's accuracy against $k^\ast$ and the majority class; KSCORE stacked with H / HS (descriptive) | descriptive |

| outcome | condition | next (committed) |
|---|---|---|
| **SELECTION PAYS** | PRIMARY-B, PRIMARY-SEL and SECONDARY at both seeds | the scorer joins `forward_infer` as a K-hypothesis planner, and the dominance row is re-read on it against P10 and FLAT |
| **BIAS, NOT SELECTION** | PRIMARY-B passes, PRIMARY-SEL fails | the gain is a constant correction of T's longitudinal decision. Next: re-train T's longitudinal head with class-balanced loss (a T-side arm) |
| **NOT PREDICTABLE** | PRIMARY-B fails | the best hypothesis is not predictable from these inputs at this scale. The longitudinal lever moves to O's speed profile (OACC's line) |
| **COSTS SAFETY** | PRIMARY passes, SECONDARY fails on CONSISTENCY or LEAD-VIOL | reported; not deployed |

⛔ No input, architecture, optimiser setting, constant-class rule or bar changes after any KSCORE number is seen.

## Compute and order

On box 2, a forward pass of O with 7 hypotheses over the training windows plus scorer training: about 1–3 h per seed (ESTIMATED). The build touches the eval harness, not the trainer, so it can be built in parallel with OACC's build. Queue: after OACC on box 2, unless the PI allocates other compute.

## Amendment 1 (2026-10-25, written BEFORE any KSCORE data: nothing has run on real data)

The instrument build (`v8f_kscore.py read`) found that the outcome table does not cover one case: PRIMARY-B and PRIMARY-SEL pass, and
SECONDARY fails **only** on speed MAE or accel MAE (CONSISTENCY(a) 'any' and LEAD-VIOL not separated-worse). It now reads, with this
precedence: **COSTS SAFETY** if CONSISTENCY(a) 'any' or LEAD-VIOL is separated-worse at either seed (unchanged); otherwise:

| outcome | condition | next (committed) |
|---|---|---|
| **COSTS DYNAMICS** | PRIMARY-B and PRIMARY-SEL at both seeds; SECONDARY separated-worse on speed or accel MAE only | reported; not deployed as is. Next: the same scorer re-read on top of OACC's base once OACC exists (descriptive until its own read) |

Any state the table still does not map is a hard refusal (rc 2), never a default. No bar, input, setting or baseline changes.
