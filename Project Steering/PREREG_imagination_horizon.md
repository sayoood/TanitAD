# PRE-REGISTRATION — how fast does PURE IMAGINATION decay without the camera?

**Written BEFORE the runs.** 2026-07-29. PI directive: *"work on the driving performance of pure
imagination based prediction of the WM by conducting experiments where the camera is not fed anymore."*

## The reframing that defines the experiment

**We already measure pure imagination at 2 s.** The WM canary encodes an 8-frame context window and
then rolls the operative predictor forward **in latent space with no further frames**, grounding to
SE(2). v1's **0.4271** IS that number. So "the camera is not fed" is not a new mode — it is the mode
the headline metric already runs in.

**What is missing is the DECAY CURVE.** One horizon tells us nothing about how imagination fails:
gracefully (error ~ √t, integration noise) or catastrophically (error ~ t², the latent drifting off
manifold). Those imply completely different engineering.

## Design

Sweep the rollout horizon on the **same** checkpoint, episodes and windows, changing **only** how
many steps are imagined after the last real frame:

- horizons **K = 10, 20, 40, 80** steps @10 Hz = **1 s, 2 s, 4 s, 8 s**
- arms: **v1** (`flagship4b-speedjerk-30k`) and **v4 from-scratch** (step 29,999)
- actions: the expert's true future actions (`actions_source="expert_future"`), unchanged — this
  isolates WORLD-MODEL fidelity from action selection, which is the whole point of the canary
- estimator: **episode-cluster bootstrap** per horizon; `overlapping_holdout_se` nowhere
- host: pod3 (idle). ⛔ pod2 (v5) and newpod (v2corpus) untouched.

## Both outcomes, committed in advance

- **OUTCOME A — GRACEFUL.** ADE grows sub-linearly or ~linearly in K (e.g. doubling K less than
  doubles error), and the 8 s number stays finite and usable. ⇒ imagination is a **usable planning
  substrate** at horizons well beyond 2 s, and the 2 s convention is a conservative choice rather
  than a limit. This would materially raise what the hierarchy can plan over.
- **OUTCOME B — CATASTROPHIC.** ADE grows super-linearly (accelerating), or the latent leaves the
  manifold and the grounded pose diverges. ⇒ **2 s is near the usable ceiling of open-loop
  imagination on this world model**, every longer-horizon plan needs re-perception, and "imagine
  further" is not available as a lever until the predictor is changed.

⚠️ **Both are informative and both will be published.** Outcome B is the more consequential: it would
put a hard number on how often the stack must look at the world.

## Pre-committed reporting

Per horizon: ADE, its CI, and the **ratio to the 2 s value** (the decay shape is the result, not the
absolute). Reported whatever the shape. ⚠️ Longer horizons have fewer usable windows per episode —
**n per horizon is reported next to every number**, since a shrinking n is itself a confound.
