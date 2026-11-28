# Architecture review: code vs the paper's diagram, and the nav gap

**Date** 2026-08-30 · **Owner** DataFlyWheel · **Trigger** PI: *"review both our
code and our architecture diagramme in the paper folder, update any missing
details like the nav commands as input and check the correctness of hierarchy
wiring."*

## Verdict in one line

**The hierarchy wiring in the diagram is CORRECT and matches the code. The nav
command was ENTIRELY ABSENT from the diagram and is now added. And the live v6
trainer does not yet supply the channel the PI made mandatory today.**

## 1. ⛔ The gap: nav command was missing from the diagram — now fixed

`grep -ci "nav.?command|nav_cmd|route input" v6_architecture.svg` returned **0**.
The figure showed the input column as *"Front camera video + proprioception
(v₀, controls)"* and nothing else.

**What the code actually says** (`stack/tanitad/models/nav_conditioning.py:1-30`,
PI directive 2026-08-30, spec `Project Steering/SPEC_NAV_CONDITIONING_ALL_LAYERS.md`):

* the nav token is a **MANDATORY input to ALL THREE layers**, *"conditioning the
  WM like the actions"* — an **input**, not a head and not a loss;
* **ONE embedding shared across the layers so they cannot drift apart**, 3-token
  vocab plus two continuous args;
* ⛔ it is an **ORACLE on our corpus** — every `nav_command` in B1 carries
  `provenance: "ego-future"`, 4,719/4,719, computed from the ego's own future
  path. Flagship v1's route head was an exact bijection of the nav it was fed
  (369/369, 81/81) and **scored 1.0000** — an echo read as skill;
* its **controls ship with it**: `real · hold · shuffled · none`, and **no
  capability claim from a nav arm is admissible without `shuffled` reported
  beside it** — `shuffled` is the decisive one (no degradation ⇒ the channel is
  INERT);
* **a missing token RAISES. It never defaults** — a silent default would let an
  arm train without the channel while its config claimed otherwise.

**Diagram updated** (`Paper/figures/v6_architecture.svg`): a NAV COMMAND block in
the input column with its own hue and arrowhead, **three arrows into all three
layers**, the oracle warning, the raise-never-default rule, and the four controls
named with `shuffled` marked decisive. A legend row was added. Verified:
the SVG parses, **0 layout collisions** with existing elements, and each arrow
terminates INSIDE its target box (strategic 132–264, tactical 338–486, operative
544–688).

⚠️ **`Paper/figures/v6_architecture.png` is now STALE against the SVG.** cairosvg
is not installed here so I could not re-render it; the PNG must be regenerated
wherever the paper's figure pipeline runs, or the paper will ship the old figure.

## 2. ✅ Hierarchy wiring — the diagram is CORRECT, verified in code

| diagram claim | code | status |
|---|---|---|
| `g_str` conditions the tactical predictor | `v6.py:3828` *"THE g_str → P_T CONDITIONING PORT (F-1, 2026-08-16)"*, port `cond_tac_dyn` | ✅ |
| `g_tac` conditions the operative predictor | `v6.py:13` — the FiLM `intent` port **IS** the `g_tac` seam, `P_O(z_op, (a,κ) \| g_tac)` | ✅ |
| goal head reads `goal_head_tac(z_tac_p, cond=e_g_str)` | `v6.py:2560, 2653, 2750` | ✅ |
| gradient-isolation barrier "enforced in code, tested" | `V6Stack.assert_isolation` — a **real autograd probe**, not a convention | ✅ |
| latents upward are stop-grad / EMA-slow | detach sites present at the declared seams | ✅ |

**Tests run: `test_nav_conditioning.py` + `test_nav_v6stack.py` — 39 passed.**

## 3. ⚠️ THE FINDING THAT IS NOT IN EITHER ARTIFACT: the live trainer has no nav

`stack/scripts/train_v6_staged.py` contains **ZERO** references to
`nav_command` / `nav_conditioning` / `NavConditioner`. The channel is built into
the MODEL (`v6.py`, `predictor.py`) and correctly guarded, but **the live v6
training run does not supply it**, so `self.nav is None` and the mandatory input
is simply absent from the run.

⭐ **This is safe rather than silent** — precisely because the code raises instead
of defaulting, a run without nav cannot masquerade as a run with it. But it means
**the PI's 2026-08-30 mandatory-nav directive is not yet in effect in training**,
and any arm currently training is a `none`-control arm whether or not it was
labelled as one.

⇒ Work item for the TrainingFlyWheel, not for me: thread `nav_command` through
the collate whitelist and set the flag. The model side is ready and tested.

## Deliverable manifest

| artifact | where |
|---|---|
| updated `v6_architecture.svg` (nav block, 3 arrows, legend row) | repo `Paper/figures/` |
| this review | repo, this package |
