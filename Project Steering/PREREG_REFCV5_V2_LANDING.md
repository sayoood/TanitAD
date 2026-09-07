# PREREG — refcv5-v2 landing: what it must clear, committed BEFORE the result

**Status:** written 2026-09-07 while `refcv5-v2` is at **step ~17,650 of 40,284** on the A40.
The result does not exist yet. **Evidence class: MEASURED** for every live-config fact below
(read from the arm's own `config.json` on the pod); **PUBLISHED** for the bar text, which is
copied VERBATIM from code registered earlier the same day.

⭐ **Why this file exists.** The bar was already pre-registered properly — in
`refcv5_compare.py::BAR_PRIMARY`, stamped `registered_before: "refcv5-v2 landing (it was at
step ~2,600 of 40,284)"`. But it lived only in harness source and one Research Lab
`RESULT.md`. A reader asking *"what is the arm currently training committed to?"* would search
`Project Steering/`, find **38 `PREREG_*.md` files and none for refcv5-v2**, and conclude
nothing was committed. ⛔ **This file changes NO criterion.** Altering a bar after seeing an
arm's data is the goalpost move the operating standard forbids; the text below is copied, not
rewritten.


⛔⛔ **SEE `PREREG_REFCV5_V2_LANDING.ERRATUM-1.md` BEFORE READING §1.** Two of this file's statements about **the arm** are corrected there — `--tac-goal-tok-head` is passed and built but receives **NO GRADIENT** (`grad_abs_sum` 0, 2/2 grads `None`), and `--max-speed-input` is **UNRUNNABLE on v7.2** (`speed_max_input` on 0/4,572 records), not merely unpassed. ⛔ **No CRITERION changes**: `BAR-REFCV5V2-1` and `-2` stand exactly as written.

## 1. The arm — MEASURED from its live `config.json`, not from the launch intent

`refcv5-v2-noagents-b1-v72-40k`, argv carries:
`--nav-from-v7` · `--tac-goal-tok-head` · `--sel-refined` · `--sampler ddim` ·
`--anchor-v0-conditioned` · `--agents off` · anchors `[117, 8, 2]`, `control_units alat`.

⛔ **It does NOT carry `--max-speed-input`.** That is a TIMING fact and not a launch defect:
the flag landed `a1d52e6` at **2026-09-07 05:11 Berlin**, and the arm launched **2026-09-06
23:10 UTC** — roughly four hours earlier. ⛔ The run is **not** to be restarted for it, and the
reason is not merely cost: refcv5-v2 already moves several levers against refcv4b, and adding a
further one would make the comparison **less attributable** — the `--v2` conflation failure
(ten levers on two axes, result non-attributable). Max speed belongs in its own arm as a single
lever.

## 2. The bar — copied verbatim from `refcv5_compare.py`, registered at step ~2,600

**BAR-REFCV5V2-1 (primary).** *"refcv5-v2 must BEAT the echo control `ha0_ext` on ADE,
SEPARATED: paired `os - ha0_ext` delta < 0 AND the whole episode-cluster interval below zero."*
Predicate `delta < 0 and hi < 0`, metric `ade_m`, **plus `relative_margin_required: 0.10`.**

**BAR-REFCV5V2-2 (secondary).** *"refcv5-v2 must BEAT the hold-action control `ha` on ADE,
SEPARATED. `ha` and `ha0_ext` are the bar TOGETHER."*

⭐ **The margin is the half that is easy to lose.** `echo_gate.py` requires BOTH the CI to
exclude zero AND a relative point margin fixed in advance, because *"a separated CI on a 0.3 %
margin is a real but useless difference"*. The 0.10 is `echo_gate`'s own documented example —
**taken rather than invented** — and it is ~28x the ~0.001 m cross-hardware roll noise measured
between the A40 landing (`os` 0.2975) and the Thor re-roll (0.2965).

**Why these controls and not another model:** refcv4b @40,284 only TIED the do-nothing
baselines (`os-ha` -0.0021 [-0.0178, +0.0154]; `os-ha0_ext` +0.0101 [-0.0050, +0.0273],
neither separated). ⛔ An arm that ties a plan holding its own t0 action and curvature has not
learned to drive, whatever it does against another model.

## 3. Both outcomes, committed in advance

| outcome | what it means | what happens next |
|---|---|---|
| **CLEARS** both bars with the 0.10 margin | the compose arm beat the echo controls | score the four families vs refcv4b; registry row; the levers become candidates for refcv5c |
| **CLEARS the CI but MISSES the margin** | ⛔ **FAIL as written.** A real but useless difference | reported as FAIL; the next lever is chosen by attribution, not by relaxing the margin |
| **MISSES** | ⛔ **FAIL.** Reported plainly | RULE ZERO: diagnose, pick the next lever, and execute it in the SAME run — a refutation is a waypoint |

⛔ **A FAIL is reported as a FAIL.** "I need excellent results" makes the stopping condition
stricter, never the evidence bar lower.

## 4. Estimator, tier, and the three variances

* **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap` ONLY. ⛔ `overlapping_holdout_se`
  is FORBIDDEN — it biases the POINT ESTIMATE bidirectionally, up to a sign flip.
* **Tier: T1**, self-action open loop. A planner feeding its own predictor is STILL open loop
  (PI ruling 2026-09-02). ⛔ None of this is "driving performance" — no simulator, no vehicle.
* ⛔⛔ **A separated CI from a one-seed arm is NECESSARY, NOT SUFFICIENT.** Three different
  questions ride on one interval, and the landing must name which it answered:
  1. *another draw of EPISODES?* — **ANSWERED** by the bootstrap.
  2. *another TRAINING RUN?* — **NOT MEASURED.** One seed. A pure replicate produced "separated"
     differences on 6 of 42 cells (14.3 % false-positive rate).
  3. *another INFERENCE RUN?* — ⭐ **BEING MEASURED NOW, and it applies to this arm and not to
     its baseline.** `--sampler ddim` draws `torch.randn_like` AT EVAL (`refc.py:1926`), so the
     same checkpoint rolled twice need not agree; refcv4b is bit-identical across seeds (4/4
     dumps by md5), so the noise enters the paired delta from ONE side only. Three rolls of this
     arm's own `ckpt_15000.pt` at `--infer-seed 0/1/2` are in flight on the dev-box 4060.
     ⛔ Step 15,000 is MID-TRAINING: its LEVEL is not a capability number and must never be
     quoted as one. Only the SPREAD is the deliverable, and it is an ESTIMATE for the 40k arm —
     the landing still runs its own replicate.

⚠️ **ORACLE-NAV stamp.** Both arms consume the v7.2 nav command, and all 4,719 v7.2 nav records
are `ego-future`. Per the binding PI ruling of 2026-09-04 that is a first-class ROUTE INPUT, not
a leak — but it is noiseless and perfectly timed where a real router is coarse, so no arm's nav
may be read as a production command.

## 5. The discriminating control that makes this bar credible

⭐ The harness was run against **refcv4b itself** and **writes FAIL on both bars**. A bar that
only ever passes is not a bar. It also reproduces refcv4b's published landing on a third box —
66/66 model-free rows bit-exact, `REPRO_CHECK.json` md5-identical
`031784a48ea6c1c2761d9ac4be0aa974`, `refcv4b - refcv3` -0.1455 [-0.1655, -0.1240] separated.

⛔ **Four metric families, never pooled.** ADE alone is one row of four (LONGITUDINAL /
LATERAL / TACTICAL / STRATEGIC). LATERAL is read on **masked** curvature MAE with the
straight-line floor beside it — unmasked, 43 stopped windows (4.9 %) carried an entire 84x
figure and reversed its sign.

*Instrument: `TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-07-refcv5-v2-comparison/`.
Bar source: `scripts/refcv5_compare.py::BAR_PRIMARY` / `BAR_SECONDARY`.*
