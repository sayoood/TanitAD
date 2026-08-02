# v5 — the cheapest guard, and a finding that outranks it

**Written 2026-08-02.** Two things the PI asked: what is the cheapest guard on v5, and *"how is the
planner working if it is not conditioned on its imagination — prediction is a core functionality of
our world model."*

⭐ **The second question turns out to be the bigger finding, so it is stated first.**

---

## A. ⛔ v5's PLANNER NEVER SEES THE WORLD MODEL'S IMAGINATION — and it is HARD-WIRED, not configured

**MEASURED 2026-08-02, read directly from source, not inferred from a flag name:**

`stack/tanitad/models/flagship_v15.py:20-40` describes the three conditioning sources of the head
v5 uses. On `cond_imagination` it says, verbatim:

> *"**THE NOVEL PART**. The frozen predictor is rolled forward under a fixed vocabulary of probe
> ACTION sequences (`imagine_probes`), and the imagined future latents are fed as conditioning
> tokens. **The decoder therefore sees the CONSEQUENCES of candidate controls before it denoises,
> instead of inferring them from the present frame.** Latents are read at the same horizons the
> anchors live at."*

**In v5 it is OFF**, and the token count proves the consequence is total, not partial —
`flagship_v15.py:342-343`:

```python
self.n_imag_tokens = (cfg.n_probes * len(cfg.imag_read)
                      if cfg.cond_imagination else 0)
```

⇒ With it false the planner receives **0 imagination tokens**. Not fewer — none.

⚠️ **It is not a launch-time choice we can flip in a config.** `stack/scripts/train_flagship_v4.py:1242`
hard-codes it inside the trainer:

```python
hcfg.cond_imagination = False   # "imagination off per real_smoke"
```

Two further sites (`:341`, `:409`) do the same for smoke/proof configs, with the comment
*"imagination is v1.5-inherited, tested elsewhere; off here so the proof isolates the labels."*
**There is no phase, curriculum or step condition that turns it on** — checked against `phases`
(`phase_a 2000`, `phase_b 8000`, `gate_step 10000`), which govern the λ_plan ramp only.

### Why this matters more than a hyper-parameter

The PI's framing is exactly right: **prediction is the core functionality of a world model.** v5
still *has* a world model — the predictor is trained, and the operative rollout is grounded through
it (that is what `g_op_fwd_ade_m` measures). What is missing is the **planner's use of it**:

| what v5 DOES condition the planner on | |
|---|---|
| `cond_states: true` | the encoder's spatial-temporal tokens — "what the scene looks like now" |
| `cond_vtarget: true` | tactical set-speed goal token |
| `cond_route: true` | strategic route goal token |
| ⛔ `cond_imagination: false` | **"what happens if I do X" — absent** |

⇒ **v5's planner selects among 256 anchors from the PRESENT FRAME plus GOAL TOKENS.** It does not
evaluate candidate controls by rolling them forward. That is a *reactive* planner with a
world-model-grounded operative layer — not imagination-in-the-loop planning.

⚠️ This bears directly on a result we already have: **0/3 hierarchy seams beneficial on both v1 and
v2corpus**, and manoeuvre κ of 0.253 (v1) / 0.0072 (v2corpus). A planner that cannot see the
consequences of its options has no mechanism by which top-down conditioning *could* become
load-bearing. ⭐ **The seam result and the imagination gap are plausibly the same defect** — that is
a **HYPOTHESIS**, not established, and it is testable.

### The honest counter-argument, recorded so it is not lost

Imagination conditioning is **expensive**: `n_probes 8 × |imag_read| 4 = 32` extra tokens, each
requiring a 20-step predictor roll per training step. Turning it on is not free, and the original
comment says it was disabled deliberately *"so the proof isolates the labels"* — i.e. it was an
attributability choice for a specific experiment, not an oversight. **What is an oversight is that
it stayed off in a 30k flagship run** without that being restated as a decision.

⇒ **DECISION OWED BY THE PI**, stated plainly: is v5 meant to be the imagination-conditioned
flagship, or a goal-conditioned reactive baseline? Both are defensible; only one matches
*"each planner predicts via imagination"*.

---

## B. The cheapest guard on v5 — the goal-dropout route collapse

### The risk

v5 runs `goal_dropout: 0.5` and `ego_dropout: 0.5`. That is the same magnitude as v2corpus's
`nav-dropout 0.5`, and we **MEASURED** what that did to v2corpus:

| | v1 | v2corpus |
|---|---|---|
| `route_acc_nav` (command given) | **1.0000** | **0.5351** |
| `route_acc_follow` (vision only) | 0.9474 | 0.5088 |
| `majority_straight_rate` | 0.9474 | 0.9474 |

The model was trained to ignore the route command half the time, **and it learned to** — route
accuracy collapsed to near chance-of-copying. v5 is exposed to the same mechanism.

⚠️ Note the dropout is defended in the source as *"the H25/H26 anti-shortcut rule: without it the
head reads the goal and ignores vision"* — so 0.5 is **intentional**. The guard is not "dropout is
a bug"; it is **"did the cure overshoot into collapse, as it demonstrably did on v2corpus?"**

### The guard — cost, and why it is the cheapest available

**Run the hierarchy pass on v5's EXISTING step-5000 checkpoint.** No new training, no new data.

```
taniteval hierarchy --model v5-5k --episodes 40
```

then read `seam_nav_to_strategic` through `four_families.strategic()`.

| | |
|---|---|
| GPU cost | **~2–3 minutes** on one A40 (the v1/v2corpus hierarchy passes took ~100 s each at n=418) |
| new training | **none** |
| input | `v5_modelonly.pt` — already extracted (1,144 MB, step 5000, `model`+`grounding`+`controller`+`goal_head`) |
| when | ⛔ **after v5 stops.** CLAUDE.md invariant: never add GPU/RAM load to a training pod — an eval OOM-killed the flagship on 2026-07-16 |

### ⭐ Pre-registered outcomes — both committed in advance

| outcome | reading | consequence |
|---|---|---|
| `route_acc_nav` **≥ 0.9** | route conditioning survives goal-dropout at v5's geometry | dropout 0.5 is safe here; the v2corpus collapse was the `--v2` pack, not dropout alone. Continue v5 unchanged. |
| `route_acc_nav` **≈ 0.5** | ⛔ **same collapse as v2corpus** | goal-dropout 0.5 is eating the route signal. Fix BEFORE spending 3.5 more days: lower to 0.25, or ramp it. Caught at step 5k instead of 30k. |
| between, or CI covers both | underpowered at 5k | re-run at the 10k milestone. ⛔ Do not pick the convenient reading. |

⚠️ **Adjudication rule:** GATE_PROTOCOL §0.7 — `nonav_route_beats_majority` is **VOID BY
CONSTRUCTION**. If a strategic number looks impossible, adjudicate **INSTRUMENT-FAIL, never
MODEL-FAIL**. And the skill test is `route_acc_follow` vs `majority_straight_rate`, **never**
`route_acc_nav`, which is privileged — v1 scores 1.0 there purely by copying the command it is fed.

### Why this and not something else

It is the only check that (a) needs **zero new training**, (b) tests a failure we have **already
measured happening once**, and (c) can still change the plan — v5 is at step 5,000 of 30,000, so
~3.5 days of compute are still ahead of it.

---

## Evidence class

| claim | class |
|---|---|
| `cond_imagination=False` in v5's head_cfg | **MEASURED (ours)** — the run's own `config.json`, rescued 2026-08-02 |
| it yields exactly 0 imagination tokens | **MEASURED** — `flagship_v15.py:342-343`, read directly |
| it is hard-wired in the trainer, not launch-configurable | **MEASURED** — `train_flagship_v4.py:1242` (+ `:341`, `:409`) |
| no phase turns it on | **MEASURED** — grep over the v4 trainer; `phases` govern the λ_plan ramp only |
| v2corpus route collapse 1.0 → 0.5351 | **MEASURED (ours)** — `hier_v2corpus-lf19.json` |
| goal-dropout 0.5 is intentional (anti-shortcut) | **MEASURED** — `flagship_v15.py` docstring |
| "the seam failure and the imagination gap are the same defect" | ⚠️ **HYPOTHESIS** — testable, not established |
| hierarchy-pass cost ~2–3 min | **ESTIMATED** — from the v1/v2corpus passes at ~100 s, n=418 |
