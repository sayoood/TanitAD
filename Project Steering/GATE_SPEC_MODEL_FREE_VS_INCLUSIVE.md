# How to specify an anchor-vocabulary gate, in units that mean something

`Drafted 2026-09-04 by the Master Mind, after setting the refcv4 abort gate wrong.
Queued for commit to Project Steering/ — git is ~80 % unavailable on this mount as of
08:50Z, so this is written and verified on local disk first.`

## What I got wrong

I told the refcv4 launch stream to abort unless the new anchor vocabulary's
**oracle-in-vocabulary ADE beat 0.2996 m** (`ha`, the hold-action control). Two
independent streams refused that, correctly, and one of them named it exactly: *"the
abort gate was set on a model-inclusive number and applied to a model-free one."*

The two quantities are not the same kind of thing:

| quantity | includes the trained offset head? | how obtained |
|---|---|---|
| `ha` **0.2996**, `oracle_sel` **0.3668**, `os` **0.4419** | **YES** | eval of a trained model |
| oracle-in-vocabulary (synthetic **1.0882** → k-means **0.3796**) | **NO** | min over raw anchors of ADE to GT — no model, no forward pass |

A raw anchor set has had no chance to be corrected. Comparing it to a decoded arm is a
scope error of the same family as every other one logged tonight, and it would have
killed a run that was in fact fine.

## The two gates, and which can run when

**PRE-LAUNCH (zero GPU, minutes on CPU) — a NECESSARY condition on the vocabulary.**
Compute, over held-out clips, `min over anchors of ADE(anchor, GT)`. Require:

1. **Beat the straight-line control.** Not a floor of convenience — MEASURED, every FPS
   variant at 6 s scores 0.7666–0.86 against a straight line's 0.6843, i.e. *worse than
   drawing one line*. refcv4's k-means/slot-norm set scores **0.3796**.
2. **Report it split ALONG vs LATERAL, and require the gain on the deficient axis.**
   MEASURED three independent ways, 92.2 % of refcv3's deficit is along-track while its
   fan's lateral half already beat the floor. refcv4's split: ALONG 0.3066 / LAT 0.1496,
   with the along-track reduction 69.5 % of the total. A pooled ADE hides this entirely.
3. **Check flyability, not just distance.** refcv3's synthetic pool reached
   **14.15 m/s²** and 1.07 g lateral, and could not turn past 74.5° — because
   `synth_anchor_pool` samples speed and yaw-rate *independently*. A fan of trajectories
   no car can drive is not a vocabulary.

**POST-TRAINING — the SUFFICIENT condition, and it cannot be moved earlier.**
Whether the arm beats `ha` is a property of vocabulary **plus** offset head **plus**
selector. No zero-GPU check can produce it, because the offset must be trained first.

## The empirical bridge, so the pre-launch number can be read

```
synthetic raw 1.0882  --[trained offset]-->  decoded oracle_sel 0.3668
                       the offset head recovered 0.7214 m
new raw       0.3796  --[offset not yet trained]-->  ?
```

**The new RAW vocabulary already sits roughly level with the old fan's DECODED ceiling.**
That is the honest reading of 0.3796, and it is a large gain rather than the failure my
gate implied.

⚠️ Do not turn that bridge into a prediction. The offset recovered 0.72 m from a fan that
was *catastrophically* bad; it will not recover the same absolute amount from a good one.
The bridge licenses "the vocabulary is no longer the binding constraint", nothing more.

## The rule

**State whether a number is MODEL-FREE or MODEL-INCLUSIVE before comparing it to
anything.** A gate that mixes the two will either kill a healthy run or pass a doomed
one, and it will look rigorous while doing it. The same discipline the programme already
applies to estimators (never quote an interval without its estimator) and to tiers (never
compare across T0/T1) applies here: **never compare a ceiling to an achievement.**

## Provenance

All arm levels read at one path, `/arms/<arm>/intervals/metrics/ade_dense_m/mean`, from
`taniteval/results/refcv3-40284-openloop.ARM.json` (n = 4,823 windows / 141 episodes,
paired episode-cluster bootstrap): `ha` 0.2996 · `oracle_sel` 0.3668 · `os` 0.4419 ·
`os_navshuf` 0.4563 · `os_navzero` 0.4659 · `ha0` 0.6723. Vocabulary figures from the
refcv4 launch package's held-out gate, 19,602 windows / 141 EVAL clips.
