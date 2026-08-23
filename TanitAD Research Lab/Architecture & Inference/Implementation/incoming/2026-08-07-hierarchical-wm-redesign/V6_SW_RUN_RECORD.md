# v6 S-W — launch record, and the measured cost of the 30k run

**Launched** 2026-08-12 on **pod5** (A40), PID 380439, out
`/workspace/experiments/v6-SW-30k`. Evidence class: **MEASURED (ours)**, from the run's own
log.

## Gate results at startup

```
[v6] params 87.89 M / budget 300 M · arm shared-encoder+adapters
     per-group {encoder 15.33, readout 0.05, predictor_op 60.29,
                layer_tac 5.77, layer_str 4.15, planner 0.66, aux 1.65}
[v6] X3 isolation pass=True
     violations={planner_to_encoder: 0, tactical_to_below: 0, strategic_to_below: 0}
[v2] parity: 2400 train providers · val 600 eps / 103 732 windows
[v6] O4: 415 002 train windows scored by ego-kinematic saliency (label-free, ACTIONS ONLY)
     weight_max_over_min 30.80 · span_steps 26
```

## ⚠️ THE COST, MEASURED — this is the D1 number

```
[50] step_s 7.2466   (elapsed/step over the 50 steps THIS process ran)
```

**7.2466 s/step × 30 000 = 217 400 s = 60.4 A40-hours.**

- step 500 (the D1 re-cost gate) ≈ **1.0 h** — cheap, and already funded by any reading of D1.
- The full 30k is **60 h**, which is a materially larger spend than D1 discussed. The run
  checkpoints every 500 steps and `--resume auto` is the default, so **it can be stopped at
  any step and continued later at no waste**. Flagging the number rather than assuming the
  budget: *the decision to run past the P-battery gate is the PI's.*

⭐ The log line carries its own definition — *"elapsed/step over the 50 steps THIS process
ran (NOT accumulated over --log-every, and NOT divided by the resumed step number)"* — which
is the `step_s` trap from CLAUDE.md fixed at the source rather than remembered.

## ⛔ A DEFECT IN THE DOCUMENTED LAUNCH COMMAND, caught by a guard

`V6_TRAINER_DESIGN.md` §3.1's canonical S-W block passes **`--v2-subframe 176x624`** — v5's
crop — while also passing `--frame-h 256 --frame-w 640`, which is what the encoder config is
built from. The first launch died in 40 lines:

```
ValueError: encoder input is (176, 624) but the config declares (256, 640)
(image_size=256, image_width=640). The positional embedding is sized for the
declared geometry — a mismatched input is the stale-default failure this check
exists to catch.
```

**The subframe is wrong for v6 and the guard is right.** v6's `SpatialGridReadout` is the
*geometry firewall* — a wide 256×640 input still yields `state_dim` 2048 — so v6 consumes
the cache at its native geometry and does not want v5's crop. The relaunched command drops
`--v2-subframe` entirely. ⇒ **§3.1's block should be corrected**; it would otherwise fail
identically for anyone who copies it.

*(Parity is unaffected: the subframe is a geometry choice, not an episode-selection one, and
`--require-parity` passed on 2400 train providers either way.)*

## Two silent pod drifts fixed to get here

pod5 had **neither `tanitad/models/v6.py` nor `scripts/train_v6_staged.py`** — the v6 code
had never been shipped to it — and its `tanitad/eval/spectral.py` predated
`participation_ratio`. All three shipped md5-verified by file-ship (pods have no git
credentials). **The dry-run caught the stale import in 2 seconds instead of after GPU
time** — the preflight-probe rule working exactly as intended.

## Config in force

`--steps 30000 --batch 16 --lr 1e-4 --o1-k 10 --o5-k 20 --o5-mode uniform --o3-mode action
--o3-blocks 2 --o3-block-h 2 --o3-block-w 2 --o2-tau-s 2.0 --o4-alpha 1.0 --w-o6 0.1
--spectrum-every 200 --save-every 500`

⚠️ **`--o5-k 20`, not 60** — the catalog's ≤2 s row, NOT the §4b 6 s representation lever
(LF4). Stated here because the design doc requires whichever was run to be named in the run
row. A 6 s O5 arm needs `--o5-k 60 --max-horizon 60` and costs extra future-frame encodes.

## Gate this run must pass (S-W, from V6_TRAINER_DESIGN §5)

P1 retention ≥ 0.85× R²(z) at k=10 per target · P3 sign ≥ 0.95 **both** channels · P3 gain
median ∈ [0.5, 2.0] **without post-training** · P6 action-subspace dims ≤ 32.
`pass: false` is REFUSED with no override — a failed stage never propagates upward.
