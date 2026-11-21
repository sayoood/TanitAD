# SPEC — E-ARCH-EGOZERO-1: the ego-zero collision in refcv3's speed channel

*Pre-registration per `.claude/skills/TanitAD_ValidateAIDesign/SKILL.md` §1. Written
2026-09-03, Architecture & Inference FlyWheel. **NO ARM LAUNCHED — the fleet is fully
committed** (refcv3 at 37,450/40,284 on the A40, refav1 at ~40 % on Thor). This document
is the contract that must exist before the compute is spent.*

The measurements this SPEC is built on are in `RESULT.md`; the raw counts are under `raw/`.

```yaml
hypothesis: H-ARCH-EGOZERO-1     # ADDED to GOALS_AND_CLAIMS.md in the same turn
one_variable: ego_valid_channel  # RefCConfig.ego_valid_channel, False -> True
held_constant: [seed, corpus, v7_labels, nav_source, steps, batch, window,
                lr, warmup, ego_dropout, size, image_hw, every other cfg flag]
success: >
  On the LOW-SPEED stratum (v0 <= 0.5 m/s at the window anchor, 6.55 % of train /
  5.23 % of eval windows), arm B beats arm A on the longitudinal family
  (target-speed error at 2 s), paired episode-cluster bootstrap over the eval
  episodes, 95 % CI ENTIRELY BELOW ZERO and point improvement >= 5 % relative;
  AND arm B is not worse than arm A on the full eval corpus (its 95 % CI must not
  lie entirely above zero).
failure: >
  The low-speed-stratum CI includes zero, OR arm B is worse than arm A on the full
  corpus with a 95 % CI entirely above zero. Either outcome refutes
  H-ARCH-EGOZERO-1 as a driving-relevant defect and the gate is left as it is.
controls: [constant_only, raw_input_floor, deliberate_regression]
splits:
  fit:  the 4,572-clip v7.2 B1 train cache (/root/data/train), clip-level
  val:  carved from FIT only — 200 held-back TRAIN clips; every hyper-parameter
        (including the stratum threshold if it is ever tuned) is chosen here
  test: the 141-clip v7.2 eval cache (/root/data/eval) — SCORED, NEVER TUNED ON
```

---

## 1. What is being tested, in one paragraph

`refc.py:2033` applies `ego_dropout = 0.5` as `v = v * keep`, and `refc.py:2060` builds
`meas_in = [v, nav] + ([keep] if cfg.ego_valid_channel else [])`. With
`ego_valid_channel = False` — the live refcv3 build — a **withheld** speed and a
**genuine 0.0 m/s standstill** are BYTE-IDENTICAL at the measurement encoder's input.
`keep` is computed on every forward and is already handed to the decoder as `ego_keep`
for the S2 reachability band (`refc.py:2157`, honoured at `:1354` and `:1494`); it is
withheld from the measurement encoder **only** by this gate. The hypothesis is that
closing that gate improves driving on the population where the ambiguity is real.

## 2. The arms — exactly five, differing in exactly one thing where it matters

| arm | `ego_valid_channel` | the bit fed | role |
|---|---|---|---|
| **A** incumbent | `False` | *(none)* | today's build. **This IS the collision.** |
| **B** treatment | `True` | the true `keep` | the fix. **A→B is the one variable.** |
| **R** deliberate regression | `True` | `keep` RANDOMLY PERMUTED across the batch each step | same width, same params, **zero information** |
| **F** raw-input floor | `False`, `ego_dropout = 1.0` | *(none, and no speed either)* | the speed channel removed entirely |
| **C** constant-only | — | — | predicts the FIT-split marginal; no model |

⛔ **R is what makes a PASS on B mean anything.** B adds one input channel and a row of
weights; without R, a win could be width or capacity rather than the *information* in the
bit. **The gate MUST fail R.** If R passes, the panel is VOID and nothing is concluded —
that outcome is committed here in advance.

⛔ **F is what makes the whole question meaningful.** If B does not beat F, the speed
channel is contributing nothing at all and the ambiguity inside it is moot.

**C must read its no-information value EXACTLY** (0.000 skill against its own marginal),
per the four-control rule; a control that reads *approximately* its known value is an
instrument failure, not a rounding matter.

### 2.1 Implementation the panel needs before it can run (0 GPU)

1. `RefCV3Config`/`RefCConfig` already carry `ego_valid_channel`; `refc_v3_train.py`
   exposes **no CLI flag for it** (grep is empty on the repo AND on the pod — two probes,
   different path bindings). Add `--ego-valid-channel` / `--ego-dropout`, **and stamp both
   into `config.json`.** They are absent from the live run's `config.json` today, so that
   file cannot answer "was the speed masked?" — refav1's config stamps `speed_channel`
   and refcv3's stamps neither. *(This is a provenance gap independent of the outcome.)*
2. Add `--ego-valid-shuffle` (default off, R only): permutes `keep` within the batch
   **after** it has been applied to `v`, so `v` is masked identically to B and only the
   companion bit is destroyed.
3. ⛔ **The in-training eval must dump per-window values with episode ids.** The register
   already records that **no confidence interval is computable from `metrics.jsonl` by any
   estimator** (`GOALS_AND_CLAIMS.md`, D-REFCV3-EPOCH-READ point 5): every row is already a
   pooled mean over 160 windows with no `eid`. This SPEC commits to a CI, so **without this
   dump the panel cannot be scored** and must not be launched.

## 3. The rig

`refc_v3_train.py --size small` (width 64 vs the live `base`'s 88) on the parity corpus,
identical steps/batch/window across all four trained arms. ⚠️ The skill's `v7-tiny` ladder
is **v6's** trainer and does **not** carry `ego_valid_channel`; this lever lives in
`refc.py`, so the rung is a reduced **refcv3**, not v7-tiny. Param count is MEASURED at
launch and recorded — not assumed from the width ratio.

⛔ `ego_valid_channel` changes the measurement encoder's input width, so **no arm can
resume from a checkpoint trained under a different setting.** All four trained arms are
fresh and equal-length. Cost is 4 arms × one tiny run; it must not be charged against a
live run's GPU.

## 4. Metrics — four families, per family, never pooled

Primary is the LONGITUDINAL family; the other three are reported because an eval that
reports one family is incomplete (binding, 2026-08-02).

| family | metric | stratified? |
|---|---|---|
| **LONGITUDINAL** (primary) | target-speed error at 2 s; headway / time-gap where a lead agent is in frame | **yes** — full corpus AND `v0 <= 0.5 m/s` |
| **LATERAL** | heading, curvature, yaw-rate, cross-track error | full corpus (the pre-registered **null**: see §5) |
| **TACTICAL** | `lon` / `lat` decision quality vs the kin3 marginal, as `1 - CE/H(marginal)`; confusion over the classes | full corpus + stratum |
| **STRATEGIC** | route/goal setting quality, 2 s goal error | full corpus |

Every number carries its **tier stamp** (T0 for train-side reads, T1 for any capability
claim) and its **estimator** (paired episode-cluster bootstrap over the eval episodes,
`taniteval/ci.py` — never `overlapping_holdout_se`, never a quadrature combination).
Where a family cannot be computed, it is reported **per family with the reason and the n**.

## 5. The pre-registered prediction, so it is falsifiable

**The lateral family should NOT move.** `refc_tactical.py:202-242` makes `lat = f(dyaw)`
alone while `lon = f(dv, v0, v1)` reads the dropped channel; and `tactical_speed_input =
False` (`refc_v3.py:213`) means the core lat/lon heads never see `v` at all. So a fix to
the speed channel that improves the LATERAL family as much as the longitudinal one would
indicate the effect is not the mechanism claimed here — it would point at capacity or
noise. **Predicted:** longitudinal improves on the stratum, lateral does not move
(its 95 % CI includes zero). If that prediction is wrong it goes in `RETRACTION_LOG.md`.

## 6. Preflight refusals (the run does not start if any fires)

- arms A and B differ in **anything** other than `ego_valid_channel` — diff the actual
  launch commands, not the intent;
- any of C / F / R missing;
- `H-ARCH-EGOZERO-1` not present in `GOALS_AND_CLAIMS.md`;
- any hyper-parameter selected on the **test** split;
- the per-window eval dump (§2.1.3) absent, so no CI is computable;
- `config.json` not stamping `ego_valid_channel` and `ego_dropout` for the arm.

⛔ **Do not cite the retired `participation_ratio >= 8.56` floor.** It is withdrawn
(`C-PARTICIPATION-FLOOR-RETIRED`); a participation number is admissible only against a
reference at matched corpus, matched episode count and matched ambient `d`, and this panel
does not need one — it is a driving-metric question, not a rank question.

## 7. Verify by CONTENT

Each arm "succeeded" only if its checkpoint exists, is non-trivial, and its per-window
dump parses with the expected window count and episode ids present. Exit codes are not
evidence.
