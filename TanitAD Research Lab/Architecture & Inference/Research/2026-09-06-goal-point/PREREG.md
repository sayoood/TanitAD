# PRE-REGISTRATION — `H-GOALPOINT-1`: a PREDICTED METRIC GOAL POINT replaces the categorical route command

**Written BEFORE the arm exists.** Skill: `TanitAD_ValidateAIDesign`.
**Author:** goal-point agent, 2026-09-06. **Branch:** `agent/arch-inf-20260803`.
**Module:** `stack/tanitad/refs/goal_point.py` (E15). **Wiring:** `stack/tanitad/refs/refc_v3.py`,
additive, `goal_point_inject` **defaults False**.

---

## 0. ⭐⭐ WHY — the exact measurement this attacks, and the ceiling it is aimed at

`…/2026-09-06-refcv4b-navpred/RESULT.md` established on refcv4b, T1, 4,823 windows /
141 episodes: **E13's categorical nav edge is a PRESENCE-GATED BIAS.** Inverting the token
left↔right on all 1,743 commanded windows costs **`+0.0022 m [−0.0006, +0.0052]`, NOT
separated**, while removing it costs **`+0.0961 m` separated** ⇒ **content 2.3 %, presence
97.7 %.** From the model's internals, a value change moves `g_str` by 0.0103 and removal by
0.1902 — **18.6×**.

This turn adds the mechanism behind that number, MEASURED on refcv4b's own live,
v0-conditioned 117-anchor fan (`raw/GP_LATERAL_ADAPTIVE.json`, `raw/GP_LONG_AXIS_t4.json`):

| what | measured |
|---|---|
| ⛔ an **ORACLE 3-way categorical command**, decoded optimally onto the LATERAL anchor index | its best deterministic map is **"go straight" for ALL THREE route classes** (modal oracle lateral level 4/9 at 51.6 % / 88.6 % / 55.1 %) — **bit-identical to the no-information goal**, and separated **WORSE** than the model's own pick: `+0.1356 [+0.0880, +0.1946]` m, `−0.0754` lat acc, **`−0.7842` turn acc** |
| ⭐ an **ORACLE metric goal point**, LATERAL axis | `−0.0414 [−0.0508, −0.0328]` m = **78.4 % of the lateral selection ceiling**; `+0.1115 [+0.0713, +0.1548]` turn accuracy — separated |
| ⭐⭐ an **ORACLE metric goal point**, LONGITUDINAL axis (t = 4 s) | `−0.0945 [−0.1164, −0.0757]` m = **57.1 % of the longitudinal ceiling**; speed MAE `−0.1184` m/s — separated |
| ⛔ the SAME goal with its **RANGE stripped** (a bearing — the shipped S6 form, and what a route command is) | **separated WORSE by `+2.3632 [+2.2004, +2.5166]` m** |

⇒ **the goal point's entire marginal value over a bearing is RANGE, and range is LONGITUDINAL —
the axis that owns 88.7 % of the oracle gap and which `nav_cmd`
(`follow`/`left`/`right`/`straight`) has no vocabulary for at all.**

---

## 1. ⛔ THE ADMISSIBILITY CHECK, RUN EXPLICITLY

| | |
|---|---|
| **inference inputs** | the strategic context token — **vision only** |
| **label source** | the ego's own FUTURE path, sampled at `t_goal_s = 4.0 s`, **TRAIN ONLY** |
| **could it have been computed from the situation classifier's output?** | ⛔ **NO.** That model (`sitclf`'s `head_img`, labelled off `tanitad/data/situations.py`) is not imported, not in this graph, not a batch field and not a label source. There is no output to launder. |
| **does it read the tactical state or logits?** | **NO** — the cascade runs strategic → tactical, never the reverse. Stronger than the ruling requires. |
| **shared trunk** | ⚠️ **YES, declared**: with `route_head` and the tactical head (all read the encoder). Not a back door for the same reason `RefCModel.goal_provenance` gives — a shared ENCODER can only launder a signal that EXISTS in the graph. Attributability is bought by the ZERO-INIT projections, so the exact ablation is "set them to 0". |
| **supplied or predicted** | **PREDICTED** — sidestepping *"a supplied route is optimistic by construction on PhysicalAI"* |

Machine-readable: `goal_point.goal_point_provenance()`, written into the run's `config.json`
and asserted by `tests/test_goal_point.py`, so the declaration lives in the RUN RECORD.

### ⛔ The leak guard is a CONSTRUCTOR REFUSAL, not a convention
A goal INSIDE the scored 2 s horizon **is the answer**. `GoalPointConfig.__post_init__` raises
when `t_goal_s <= t_pred_s`, so **no arm that leaks the scored horizon can be built at all**.
*(This is not hypothetical: the first version of the lateral probe put the goal at 15–20 m of
arc and read the best recovery there. With `lan.horizon_lead_m` applied — guard mean **28.23 m**
— that advantage largely disappeared. The leak looked exactly like a finding.)*

---

## 2. THE ARMS — ONE VARIABLE EACH

**Training arms** (identical to refcv5's landed recipe in every other respect):

| arm | the ONE variable vs its comparator |
|---|---|
| `base` | — the incumbent: E13 categorical nav (`nav_inject=True`, `goal_point_inject=False`) |
| `gp_cond` | vs `base`: the **TYPE of the route conditioning signal** — E13's `Embedding(4)→Linear→add` replaced by `GoalPointConditioning` (metric 3-vector → MLP → the SAME two zero-init projections into `z_tac` and `ctx`). `nav_inject=False`. |
| `gp_geo` | vs `gp_cond`: **the geometric selection seam** — `anchor_goal_prior_at_time` added to the ranked score behind one zero-init gate (§8). |

**Eval-time interventions on `gp_geo`** — zero extra training, one process, one surface:

| arm | intervention |
|---|---|
| `gp_geo_mirror` | goal point `y → −y` — ⛔ **the LATERAL deliberate-regression arm** |
| `gp_geo_rng05` | goal point range `× 0.5` — ⛔ **the LONGITUDINAL deliberate-regression arm** (a mirror is inert on that axis) |
| `gp_geo_straight` | goal replaced by `(v0·t_goal, 0)` — **the constant / no-information goal** |
| `gp_geo_shuf` | goal points permuted across windows within an episode — **distribution-matched random** (the control that FAILED for `os_navpred`) |
| `gp_geo_zero` | goal withheld (`valid = 0`) — **the PRESENCE control** |

**Trivial controls, mandatory:** `ha0_ext`, `ha`, `ha0` — must reproduce **0.2874 / 0.2996 /
0.6723** to `< 5e-4`, or the surface is not the banked surface and the panel is VOID.

⛔ **Every arm rolls in ONE process on ONE surface.** A recent cross-hardware reproduction moved
two argmax tie-breaks and missed two pre-registered controls by ~0.00003; that is a solved
problem and it is solved by not splitting the panel.

---

## 3. ⛔ THE PRIMARY IS NOT ADE — AND THAT IS MEASURED, NOT STYLISTIC

The flip experiment degraded **curvature +13.7 %**, **heading +6.2 %** and **`turn_right` recall
−6.1 pts while ADE did not move at all.** Those margins were unpaired; **this turn paired them**
(`raw/NP2_PAIRED_LATERAL_TACTICAL.json`, work item NP-2, now CLOSED):

| `os_navflip − os` | delta | CI95 | separated |
|---|---|---|---|
| curvature MAE (1/m) | **+0.001120** | [+0.000090, +0.002539] | **YES** |
| heading MAE (deg) | **+0.0810** | [+0.0303, +0.1660] | **YES** |
| `turn_right` recall | **−0.0606** | [−0.0969, −0.0297] | **YES** |
| *(ADE, for contrast)* | *+0.0022* | *[−0.0006, +0.0052]* | *no* |

⇒ **the nav CONTENT reaches path shape and the lateral decision with separated margins while
its ADE margin is not separated.** An ADE-only read of this programme's route work has been
reading the wrong instrument.

### Committed endpoints (all paired episode-cluster bootstrap, `taniteval/ci.py`, B = 2000, seed 0)

| # | endpoint | SUCCESS criterion, committed in advance |
|---|---|---|
| **P1** | **lateral decision accuracy on TURN windows** (trajectory-derived kin3 via `refc_tactical.factor_from_kinematics` — the same instrument as the table above; n ≈ 548 turn windows) | `gp_geo − base` **separated POSITIVE** with delta **≥ +0.020** |
| **P2** | **heading MAE (deg)** | `gp_geo − base` **separated NEGATIVE** |
| **P3** | **curvature MAE (1/m)** | `gp_geo − base` **separated NEGATIVE** |
| — | **ADE** and the LONGITUDINAL / STRATEGIC families | **reported beside them, and explicitly NOT the headline** |

⛔ **FAILURE** is any of: P1 not separated, P1 separated but < +0.020, or P1 separated NEGATIVE.
A failure is reported as a failure — and §7 already names what happens next.

---

## 4. ⛔⛔ THE VALUE-SENSITIVITY GATE — a GATE, not an observation

E13 passed a value-flip trivially. A PASS on P1–P3 means nothing unless a **corrupted goal
provably FAILS**. All three must hold, or the arm FAILS **regardless of P1–P3**:

| # | gate | committed bar | basis |
|---|---|---|---|
| **V1** | `gp_geo_mirror − gp_geo`, turn-window lateral accuracy | **≤ −0.1212, separated** | E13's OWN measured value-sensitivity is **−0.0606** separated on `turn_right` recall; the bar is **2×** it |
| **V2** | `gp_geo_rng05 − gp_geo`, **speed MAE** | **≥ +0.05 m/s, separated worse** | the range corruption is separated worse by **+2.11 m/s** on the oracle selection surface; the bar is ~2.4 % of that |
| **V3** | **content share** `(ADE(mirror) − ADE(gp_geo)) / (ADE(gp_geo_zero) − ADE(gp_geo))` | **≥ 0.25** | E13's measured content share is **0.023**; the bar is ~11× |

⭐ **Proof the gate can fail — required, and supplied in advance.** On the live fan, the
mirrored goal is separated worse by **+0.3843 [+0.2727, +0.5168] m** bank ADE and **−0.7842
[−0.8385, −0.7190]** turn accuracy; the range corruption by **+1.8809 [+1.7111, +2.0430] m**.
In code, `tests/test_goal_point.py::test_R_a_MIRRORED_goal_point_selects_the_MIRRORED_anchor`
and `::test_R_the_RANGE_half_is_what_a_BEARING_CANNOT_CARRY` are **mutation** tests: they
reintroduce the collapse that killed E13 and assert the categorical path is blind to it while
the geometric prior is not. **A gate a corrupted goal cannot fail is not a gate.**

---

## 5. ⛔ THE HEAD GATE — checkable WITHOUT the planner, and it comes first

| # | gate | bar | basis (noise injected into the ORACLE goal, recovery re-read) |
|---|---|---|---|
| **H1** | RMS **lateral** error of the predicted goal at 4 s | **≤ 1.0 m** | recovery of the lateral bank-ADE ceiling: **+78.4 % (σ 0) → +44.8 % (0.5 m) → −80.7 % (1.0 m)** |
| **H2** | RMS **range** error | **≤ 2.0 m** | recovery of the longitudinal ceiling: **57.1 % → 47.2 % (1 m) → 17.2 % (2 m, still separated) → −57.9 % (4 m, separated WORSE)** |

⛔ **If H1/H2 FAIL, the planner comparison is NOT evidence about the goal-point FORM** — it is
evidence about an under-trained head, and the pre-registered next step is head capacity /
schedule / loss weight, **not** abandoning the lever. Stated now so it cannot be decided later.
Readouts: `goal_point.lateral_rmse_m` / `goal_point.range_rmse_m`, in METRES, every eval.

---

## 6. ATTRIBUTION — which seam did the work

**A1** `gp_geo − gp_cond` separated ⇒ the **geometric selection seam** carries it.
**A2** `gp_cond − base` separated ⇒ the **conditioning type** carries it.
Both reported; neither is required for the primary. This split exists because the navpred
turn's lesson is that *"the effect is real"* and *"the effect came from where I thought"* are
two different claims, and only the second one was refused.

---

## 7. ⚠️ ADVERSE PRIORS, COMMITTED IN ADVANCE — so a failure is not a surprise and the next lever is already named

1. ⛔ **PREDICTION: P3 (curvature) will FAIL.** MEASURED this turn on the banked dump: the
   PICKED ANCHOR's curvature MAE is **0.004019** while the EMITTED plan's is **0.008150** — the
   decoder's free-waypoint refinement **roughly DOUBLES curvature error (2.03×)** while halving
   ADE (0.4281 → 0.2965) and improving heading (1.7665 → 1.2964) and cross-track (0.1754 →
   0.0978). The anchors are constant-curvature unicycle rolls, smooth **by construction**; the
   decoder emits free waypoints trained on an ADE-shaped loss and **buys position with
   smoothness**. ⇒ **the `os − ha0_ext` curvature gap (`+0.004438 [+0.002692, +0.006464]`,
   separated; `+0.003563` on the intersection mask) is generated in the DECODER OUTPUT
   PARAMETERISATION, not in routing.**
   ⇒ **If P3 fails, the pre-registered next lever is `R4b`: a curvature-parameterised or
   residual-over-anchor decoder output** — not a goal change, and not a re-run.
2. **PREDICTION: the LONGITUDINAL recovery will exceed the lateral one.** Measured ceilings:
   longitudinal `−0.1655 m` vs lateral `−0.0528 m`.
3. **PREDICTION: `gp_cond` alone (A2) will be small.** A signal that enters only through a
   learned additive projection has the same degree of freedom E13 collapsed into. The geometric
   prior is the part with no such freedom.

---

## 8. ⛔ THE INTEGRATION ITEM — named, not buried in a README

`gp_geo` needs **one gated `r_terms` entry** in `refc.py`'s ranked-score block, beside the two
that already exist (`goal_gate * _lan_anchor_prior`, `goal_dist_gate * _goal_along_prior`):

```python
# refc.py, the ranked-score block, next to the existing S6 terms
if self.gp_point_gate is not None and gp_point is not None:
    r_terms.append(self.gp_point_gate * gpm.anchor_goal_prior_at_time(
        gp_point, gp_valid, prior_bank, self.gp_slot, self.gp_scale_m))
```
plus `self.gp_point_gate = nn.Parameter(torch.zeros(1))` (zero-init, so the ranked score is
bit-identical to the goal-free baseline at step 0 and the exact ablation is "set the gate to 0")
and reading `hook_out["goal_point"]` — **which `refc_v3.py` already emits** (this turn), so the
patch needs no second forward.

⛔ **`refc.py` is not this agent's file and the panel cannot run without that patch.**
**BLOCKER, named per RULE ZERO (3): it is a ~6-line change owned by the REF-C model owner.**
Everything upstream of it — the module, the conditioning wiring, the tests, the labels, the
bars — is landed and green.

Trainer flags to add in `scripts/refc_v3_train.py` (also not this agent's file):
`--goal-point-inject`, `--goal-point-geo-prior`, `--goal-point-t`, `--goal-point-w`.

---

## 9. THE LAUNCH COMMAND

Once §8 lands, on the A40 **after refcv5 finishes (ETA ≈ 2026-09-08 06:20 UTC)** — ⛔ this
agent did not touch the A40:

```bash
cd /workspace/TanitAD && PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6 \
python3 stack/scripts/refc_v3_train.py \
  --arm hier --size base --steps 40000 --batch 24 --lr 3e-4 --warmup 1500 \
  --data-root /workspace/data/physicalai --v2-cache /workspace/data/_v2ep \
  --v2-lru 64 --require-parity --v7-labels \
  --anchors /workspace/anchors/anchors_refcv4b.pt --n-anchors 117 \
  --anchor-v0-conditioned --anchor-control-units alat --anchor-ref-speed 10.0 \
  --ego-state-inject --ego-valid-channel --ego-dropout 0.15 \
  --goal-point-inject --goal-point-geo-prior --goal-point-t 4.0 --goal-point-w 1.0 \
  --eval-every 2000 --save-every 2000 --log-every 50 --seed 0 \
  --out /workspace/experiments/refcv6-gp-40k
```

`base` (`--nav-from-v7`, no `--goal-point-*`) and `gp_cond` (`--goal-point-inject` without
`--goal-point-geo-prior`) are the same command with those flags moved — **one variable each**.

⚠️ **`--anchor-control-units alat` is not optional.** `anchors.pt`'s `controls[:, 1]` is
LATERAL ACCELERATION; read as curvature it yields 3,888 m/s² at 36 m/s and 104/117 anchors over
a μ = 0.7 friction circle. This turn RE-VALIDATED the units independently: rebuilding the
v0-conditioned bank under `alat` reproduces the banked `sel_bank_nav_true` to **1.53e-05 m**,
while the fixed-anchor reading is **157.62 m** off.

---

## 10. WHICH VARIANCE THE INTERVALS ANSWER

* Every **eval-time** intervention (`gp_geo_mirror`, `_rng05`, `_straight`, `_shuf`, `_zero`) is
  ONE checkpoint on ONE dump — the episode-cluster bootstrap answers exactly *"would another
  draw of EPISODES say this?"*, and refcv4b-class decoders are deterministic at inference
  (`D-REFCV4B-SEED-SCOPE`), so no inference-sampling variance enters.
* ⛔ **`gp_geo − base` and `gp_cond − base` compare TWO TRAINED MODELS and therefore inherit
  `H-ESTIM-SEED-1`**: a separated interval there is **necessary, not sufficient**. A
  **replicate arm** (same flags, re-run) is required before P1 is quoted as a lever effect —
  the measured false-positive rate for `separated` on a tiny rig with zero levers moved was
  **~17 %**. Budgeted as part of the arm, not as an afterthought.
