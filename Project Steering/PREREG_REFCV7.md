# PRE-REGISTRATION — refcv7: do DrivoR's two planning heads move OUR four families?

`Issued 2026-09-19 (PI request the same day). Reads with SPEC_REFCV7.md and, for everything it
inherits, SPEC_REFCV6_V2.md + PREREG_REFCV6_V2.md. ⛔ Nothing here may be edited after the first
refcv7 GPU hour except by a dated amendment.`

## 0. What is under test

**H-REFCV7-1 (proposals).** A WTA set of 64 free-form proposals, added to the 117-anchor fan,
gives the planner candidates the vocabulary cannot express.
**H-REFCV7-2 (selection).** A DISENTANGLED scorer trained against the PhysicalAI oracle picks a
better candidate than refcv6's goal-distance scorer (`scorer` = 1,145 params over the fan).
**H-REFCV7-3 (search).** With that scorer as the reward, TOAD's test-time CEM improves the
deployed plan at ≈ 2 ms.

⭐ **The bar these exist to clear, and it is OURS, not DrivoR's:** refcv5-v2's deployed plan
**LOST to the kinematic echo control** — T1 `os` ADE 0.3079 m vs `ha0_ext` 0.2874 m, +0.0205
[+0.0043, +0.0390], a FAILED pre-registered bar (`MODEL_REGISTRY.md` §4.8).

## 1. The arms — `one_variable`, everything else held constant

| arm | the one variable | flags |
|---|---|---|
| **A0** (baseline) | — | the refcv6 arm exactly (`SPEC_REFCV6_V2` §10–§12) |
| **A0b** (replicate) | nothing at all — same flags, same seed | ⭐ the run-to-run floor (`H-ESTIM-SEED-1`: a same-seed replicate produced "separated" differences on 14.3 % of family cells) |
| **B** (heads only) | `--refcv7 --w-r7-wta --w-r7-scorer --r7-no-select` | the heads train; the DEPLOYED pick stays refcv6's ⇒ isolates "do the extra objectives hurt the planner?" |
| **C** (selection) | B + selection by the scorer | `--refcv7 … ` (default `refcv7_select`) |
| **D** (search) | C + TOAD **at eval only** | `--r7-toad` — ⛔ a re-EVAL of C's checkpoint, NOT a new training run, so C→D is a pure inference delta |

⛔ **B→C→D is a LADDER of single steps.** A vs C differs in three things at once and may never be
reported as "refcv7 works"; each rung carries its own number.

## 2. Splits, corpus, estimator

* Corpus: the v7 training corpus at 416 × 1024 (`SPEC_REFCV7.md` §4). Train/eval split by clip,
  leak-guarded by `refcv3_make_split.py` (it refuses intersecting label sets).
* ⛔ **`--r7-nav-tau-rad` is derived on the TRAIN split** (`scripts/refcv7_derive_nav_tau.py`) and
  stamped. A tau derived on the scored split is tuning on the scored split.
* Estimator: **paired episode-cluster bootstrap** (`taniteval/ci.py`), n_boot 2000, cluster =
  episode. ⛔ `overlapping_holdout_se` appears nowhere.
* Tier: **T1, self-action open loop**, on the same windows for every arm. ⛔ No closed-loop claim.

## 3. Controls that must read a KNOWN value

| control | must read | why it exists |
|---|---|---|
| **refcv7 OFF is bit-identical** | every pre-existing parameter identical at a fixed seed; no `refcv7_*` state-dict key; no ledger line | `test_refcv7_model.py` — a live refcv6 recipe must resume through this code |
| **scorer ⇏ proposals** | WTA parameter gradient EXACTLY 0.0 from a scorer-only backward | the stop-gradient IS the disentanglement (DrivoR 86.8 → 90.0) |
| **candidate independence** | a candidate's sub-scores unchanged when its companions change; and CHANGED with `--r7-scorer-self-attn` | a search reward must be a function of the trajectory alone |
| **oracle abstention** | NC/TTC mask False on any window with an unlabelled future agent frame; DAC mask False on unseen ground | an abstention must never be scored as a pass |
| **nav timing** | nav sub-score = 1 for a straight candidate when the human did NOT turn within 6 s | MEASURED 2026-09-19: the per-clip nav token is usually NOT due inside the horizon (median \|GT terminal heading\| 0.048 rad, Youden J 0.22) — a side-only target would reward turning early |
| **TOAD never worse** | `r7_toad_reward >= r7_toad_base_reward` on every row, against the ACTUAL pick | the fit residual is ~2 m on free-form proposals, so the rolled base is not the pick |
| **selection vs random** | `r7_sel_oracle_pick` reported beside `r7_sel_oracle_random` and `r7_sel_oracle_best` on the same windows | *"best − actual is large for a GOOD selector too"* — an oracle gap without a random control is not a claim |

## 4. Gates — per family, never pooled (the four-families rule)

**G1 (the bar refcv5-v2 failed).** C's deployed plan beats `ha0_ext` on T1 ADE with a **separated
paired interval**, AND the same holds at the A0b replicate floor.
**G2 (selection).** C's `r7_sel_oracle_pick` − A0's (same windows, same oracle) > 0, separated,
and above the random control's spread.
**G3 (no family regresses).** Longitudinal (speed MAE, target-speed acc, headway/TTC with its
censoring count), lateral (heading, yaw-rate, cross-track, curvature vs the straight-line floor),
tactical (lat/lon acc + κ, anchor/candidate selection acc, goal FDE), strategic (⛔ n = 0 by
construction: the strategic layer is deactivated — stated per family, never silently dropped).
**G4 (search).** D − C on T1 ADE, over **5 inference seeds**, beyond the inference-seed spread.
⛔ An effect smaller than that spread is not an effect.
**G5 (cost).** Per-step training cost vs A0, and eval-time ms/window for D − C, both MEASURED.

⛔ **A missing instrument is a work item, not an excuse** — and a family that cannot be computed
says so per family with its reason and its n.

## 5. Refusals — the launch checker exits non-zero on any of these

1. `--refcv7` without `--tac-decoder-v6`, `--arm hier` or `--trunk timm`;
2. either refcv7 weight > 0 without `--refcv7`, or `--refcv7` with both at 0;
3. `--w-r7-scorer > 0` without `--agent-join`, without a live `--w-map`, or without
   `--r7-nav-tau-rad > 0`;
4. scorer selection with an untrained scorer; `--r7-toad` with `--w-r7-scorer 0`;
5. the built model and the weights disagreeing (read off the OBJECT, not argv);
6. a tau whose provenance is not the TRAIN split;
7. any hyper-parameter selected on the scored split.

## 6. What this pre-registration does NOT claim

1. ⛔ **No closed-loop claim.** T1 only; AlpaSim is unprovisioned (`D-12`).
2. ⛔ **No transfer of DrivoR's numbers.** DrivoR's ablations are single-run navval PDMS on
   NAVSIM; they set the DIRECTION of these arms, never an expected effect size, and no NAVSIM
   number may be compared with a PhysicalAI number (V-5).
3. ⛔ **No claim about registers.** refcv7 keeps the PI-ruled ResNet trunk; DrivoR's register
   compression is the separate arm **DR-4**.
4. ⚠️ **OWED — the oracle is partial.** No map-based lane/direction terms exist on PhysicalAI, so
   DAC is drivable-area only and the oracle's coverage is reported per sub-score every step
   (`r7_orc_*_cov`).
5. ⚠️ **OWED — the ego-frame assumption** of the future-agent block (`ep.poses` are the rig poses
   the join's boxes are expressed in) is stated in the code and has an analytic test, but has not
   been re-derived from the join's own builder.
