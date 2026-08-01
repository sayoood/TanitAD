# PRE-REGISTRATION — v2corpus (50 h balanced) vs v1 (13 h parity)

**Written 2026-07-29 04:0x UTC, with the v2corpus arm at step 25,750 / 30,000 — i.e. BEFORE the
checkpoint exists and before any comparison number has been computed.** Both outcomes are committed
below. Estimator named. Void conditions stated in advance.

| arm | checkpoint | train corpus | scale |
|---|---|---|---|
| **v1** | `flagship4b-speedjerk-30k` (step 29,999) | `physicalai-train-e438721ae894` (**the parity set**) | 2,376 clips · 13.13 h · 472,627 frames |
| **v2corpus** | `flagship-v2corpus-30k` (in training, ETA ~12 h) | **`physicalai-v2bal-4b7eeeac222d`** | 9,000 clips · ~50 h |

## ⛔ THE FRAMING RULE — this is a CORPUS CONTRAST, NOT A PARITY COMPARISON

`physicalai-v2bal-4b7eeeac222d` **is not the parity corpus and breaks parity BY DESIGN** (the PI's
2026-07-24 decision: *"a NEW 'v2' canonical corpus for the NEXT flagship gen — BREAKS PARITY with
`e438721ae894` BY DESIGN"*). The two arms differ in **corpus size AND manoeuvre distribution
simultaneously** (turns 14.25 % → 28.0 %, lane_keep 59.6 % → 45.0 %, junction-clip presence
37.7 % → 61.3 %, 1,311 original clips re-selected + 7,689 new).

⇒ **A difference cannot be attributed to "more data" OR to "better distribution" — the design
confounds them.** Any report saying "more data helped" is unsupported by this experiment.
The admissible claim is: *"the v2 corpus, which differs in both size and balance, produced X."*

⛔ **This result may NOT be entered in `MODEL_REGISTRY.md` as a cross-arm comparison** alongside the
parity-corpus arms. It belongs in its own section, labelled as a corpus contrast.

## ⚠️ VOID CONDITION — RUN THIS BEFORE ANY SCORING

`physicalai-v2bal` re-selected 9,000 clips from an 18,731-clip pool. **If any episode of the
evaluation val set appears in v2corpus's TRAIN set, the v2 arm is scoring on its own training data
and the contrast is contaminated.**

**MANDATORY FIRST STEP — before computing any metric:**
1. Resolve v2corpus's train clip list (the `r0_selection_v2.parquet` 9,000-clip selection).
2. Intersect it with the evaluation val episodes.
3. **If the intersection is non-empty**: the harness's existing leakage guard
   (`taniteval.runner`, the `train_ids` path) drops the affected episodes and prints
   `[guard] DROPPED n/N val eps`. **Report that count in the headline, not a footnote.**
4. **If fewer than 8 leak-free val episodes remain, REFUSE the comparison** — that is the harness's
   own bar (`assert len(files) >= 8`) and it is the correct one.

⚠️ **A silent zero-drop must be VERIFIED, not assumed.** Print the intersection size even when it is
zero, so the artifact records that the check ran.

## Evaluation surface — identical for both arms

- **Val:** the canonical clean split, **40-episode canonical surface (881 windows)** and, if
  available on the scoring host, the **600-episode** surface. ⛔ `physicalai-val-f1b378f295ae` is a
  **code-level refusal** (`parity.py:24`, 78.5 % train-contaminated) and must not be used.
- **Metric:** `ade_0_2s` = **`wm_fidelity_ade_2s`** — the world model handed the expert's true future
  actions. ⛔ **It is NOT a planning bar** (registry §1.2). Reported as fidelity, never as driving.
- **Reference:** v1's **full-set** value **0.4271** (40 eps) / **0.4108** (600 eps). ⛔ Never the
  trainer's optimistic dense-20 reading, and never a `heldout` split-mean.
- **Estimator:** **PAIRED episode-cluster bootstrap** (`taniteval/ci.py`, B=2000) on the shared
  windows. ⛔ **`overlapping_holdout_se` is forbidden** — it is not a jackknife and it biases the
  point estimate. ⛔ No combination in quadrature.
- **Decomposition:** lateral / longitudinal via `driving.py:201 frenet()`, reported AT 2 s. The
  programme's largest known gap is longitudinal, so a pooled-only number would hide the mechanism.
- **Per corpus, never pooled.**

## Outcomes, committed in advance

| outcome | reading | consequence |
|---|---|---|
| **v2corpus better, paired CI excludes 0** | the 50 h balanced corpus helps world-model fidelity | Corpus work is validated as a lever. ⚠️ Still **cannot** separate size from balance — the next experiment is a **size-matched** arm (2,376 clips drawn from the v2 balance) to isolate distribution. |
| **Tie, CI covers 0** | 3.8× the data with a better manoeuvre mix did **not** move fidelity | **A genuine and publishable negative.** It would say fidelity is not data-limited at this scale — consistent with E-CR's finding that the failure is *compounding*, not capacity. ⇒ redirect effort from corpus expansion to **rollout-recovery training**. |
| **v2corpus WORSE, CI excludes 0** | ⚠️ Do **not** conclude "more data hurts". First check, in order: (a) leakage guard dropped episodes asymmetrically; (b) the v2 arm is under-trained at 30 k on 3.8× the data (**4.73 epochs on 13 h vs ~1.25 on 50 h** — *this is the leading candidate and must be stated up front*); (c) the JPEG-q90 v2 cache vs the raw parity cache. | Only after (a)–(c) are excluded does a corpus-quality reading become admissible. |

⭐ **The epoch confound is registered NOW because it is the most likely explanation of a negative
result and the easiest to rationalise away afterwards.** 30 k steps is **4.73 epochs** over the 13 h
parity corpus but only **~1.25 epochs** over 50 h. **The arms are matched on STEPS, not on epochs.**
Whatever the outcome, that sentence travels with the number.

## Cost and sequencing

Both checkpoints exist (or will) — **eval only, ~0.3 GPU-day**, no retrain. Runs on whichever host
carries the canonical val. ⛔ Do not run it on a pod that is training.

## Evidence class

| item | class |
|---|---|
| the corpus profiles (2,376 / 13.13 h / 472,627 frames; 9,000 clips; turn fractions) | **MEASURED** (`corpus_profile.json`, `V2_CORPUS_DESIGN.md`) |
| v1's 0.4271 / 0.4108 | **MEASURED**, registry §1.2a |
| the epoch counts (4.73 vs ~1.25) | **MEASURED / ESTIMATED** — 4.73 is from the corpus profile; the v2 figure is derived from clip count and must be recomputed from the actual v2 window count before quoting |
| everything about the outcome | **UNKNOWN — the run is at step 25,750 and no comparison has been computed** |


---

## AMENDMENT 2026-08-01 — ⛔ THIS IS NOT A CORPUS CONTRAST. IT IS CORPUS **+ THE ENTIRE `--v2` LEVER PACK**.

**Found while preparing the eval, BEFORE any number existed** — the same mandatory pre-flight check
that caught C64. The pre-registration above registers the **epoch** confound and the
**size-vs-balance** confound, and licenses the claim *"the v2 corpus, which differs in both size and
balance, produced X."* **That claim is still too strong**, because the two arms differ by far more
than the corpus.

### MEASURED — the two exact commands

| | v1 `flagship4b-speedjerk-30k` | v2corpus `flagship-v2corpus-30k` |
|---|---|---|
| source | `MODEL_REGISTRY.md` §1.2 "Exact command" | `newpod:/workspace/run_v2corpus.sh` line 24-30 |
| corpus | parity `physicalai-train-e438721ae894` (2,376 eps, ~13 h) | `physicalai-v2bal-4b7eeeac222d` (9,000 clips, 49.742 h) |
| **`--v2`** | ⛔ **ABSENT** | ✅ **PRESENT** |
| `--rollout-k` | **4** (explicit) | **not passed** |
| `--speed-input` | explicit | implied by `--v2` |
| `--jerk-weight 0.02` | explicit | included in the `--v2` pack |
| `--aux-accel` | explicit | ⚠️ **not listed in the `--v2` pack — unresolved** |

### What `--v2` actually turns on

`stack/scripts/train_flagship4b.py:651-656` documents the pack verbatim:
*ego->planners + ego-dropout 0.25 + fa-dropout 0.3 + goal-decode + nav-dropout 0.5 + jerk 0.02 +
gated-intent + anchor-tactical + speed-input; defaults rollout-k to 12 if unset.*

Plus, in the same configuration block:

- `:302-303` — `if args.v2 and args.rollout_k is None: cfg.train.rollout_k = 12`
  => **v2corpus trained at rollout_k 12, v1 at rollout_k 4.**
- `:285` — `cfg.v2_invdyn_gradscale = 0.25` (default 1.0)
- `cfg.v2_labels` — the v2 LABEL gate, not overridden (`--no-labels-v2` absent)

### Why this matters more than the confounds already registered

⭐ **`rollout_k` is not a neutral knob here — it is the exact quantity E-CR just proved matters.**
E-CR returned **H-COMPOUND** (CR 3.50 -> 80.77, teacher-forced arm flat), and the whole RR-20 /
RR-CTL experiment now running exists to measure what changing rollout_k does. v2corpus has
**k=12 vs v1's k=4** baked in. So a v2corpus-vs-v1 delta partly measures the *same lever*
RR-20/RR-CTL is measuring in isolation — on a different corpus, with eight other levers moving.

### ⛔ REVISED admissible claim

**Not** *"the v2 corpus produced X"*. The only admissible statement is:

> *"The v2-line arm — which differs from v1 in corpus (3.8x data, manoeuvre-balanced), in
> `rollout_k` (12 vs 4), and in the whole `--v2` lever pack (ego-dropout, fa-dropout, goal-decode,
> nav-dropout, gated-intent, anchor-tactical, invdyn-gradscale 0.25, v2 labels) — produced X."*

⛔ **No outcome of this experiment can validate or refute "corpus work is a lever."** The outcome
row in the table above that reads *"Corpus work is validated as a lever"* is **hereby struck** —
it was never separable, and a positive result would be attributed to the corpus purely by
association.

### What would make it a real corpus contrast

A **lever-matched** control: the v2bal corpus trained with **v1's flags** (`--speed-input
--jerk-weight 0.02 --aux-accel --rollout-k 4`, no `--v2`), or equivalently v1's corpus trained with
`--v2`. Either isolates one axis. Neither exists today. ⇒ **that arm is the experiment that
answers the corpus question**, and it should be costed before more v2-line training.

### Status of the other constraints — UNCHANGED and still binding

- **C64 still binds**: 21 of the 40 canonical val episodes are inside v2corpus's training corpus.
  Headline on the **19 leak-free** episodes, `leak_free_n = 19`, **v1 re-scored there**.
- Estimator: **paired episode-cluster bootstrap** (`taniteval/ci.py`). Never `overlapping_holdout_se`.
- `wm_fidelity_ade_2s` is **fidelity, not a planning bar**.

### Evidence class

| claim | class |
|---|---|
| both exact commands, flag by flag | **MEASURED (ours)** — registry §1.2 + the launcher read off the pod 2026-08-01 |
| `--v2` sets `rollout_k = 12` | **MEASURED** — `train_flagship4b.py:302-303`, read directly |
| the lever list | **MEASURED** — the `--v2` help string, `:651-656` |
| `--aux-accel` parity between the arms | ⛔ **UNRESOLVED** — not in the `--v2` help text; must be settled before the write-up |
