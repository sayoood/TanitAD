# PRE-REGISTRATION — WP-B: waypoint-indexed cross-attention into REF-C's SPARSE agent tokens

**Date:** 2026-09-08 (Europe/Berlin) · **Author:** Architecture & Inference FlyWheel ·
**Branch:** `agent/arch-inf-20260803` · **Parent HEAD:** `7609490`
**Status:** design + code + tests **DELIVERED and STAGED**, **no training launched**, **0 GPU
spent** on any arm. Every number below is either MEASURED on the dev-box CPU today or cited to a
banked artifact.
**Hypothesis id:** `E-WP-INDEX-1` (registered in `Project Steering/GOALS_AND_CLAIMS.md` in the same
turn as this file).

**Owner files:** `stack/tanitad/refs/refc_wp_index.py` (new), `stack/tanitad/refs/refc.py` (gated
wiring), `stack/scripts/refc_v3_train.py` (flags, refusals, stamp, seam assertion),
`stack/tests/test_wp_index.py` (new, 35 tests).

**Estimator for every interval below:** episode-cluster bootstrap over the val episodes,
`taniteval/taniteval/ci.py`; **paired** for two arms on the same windows. ⛔ `overlapping_holdout_se`
is never called — it biases the point estimate as well as the interval (−6.67 %…+11.69 %,
bidirectional, over 27 dumps).

⛔⛔ **AND A SEPARATED INTERVAL IS NECESSARY, NOT SUFFICIENT — TWICE OVER FOR THIS ARM.**
1. The episode-cluster bootstrap resamples EPISODES with the models held fixed, so it answers
   *"would another draw of episodes say this?"* and never *"would another TRAINING RUN say this?"*
   (`H-ESTIM-SEED-1`: two arms differing in **nothing** read `separated` on 6 of 42 family cells).
2. ⛔ **refcv5-v2 runs `--sampler ddim`, so its planner SAMPLES at eval and the same checkpoint
   evaluated twice does not give the same answer** (`refc.py::_sample`'s own ⛔⛔ note;
   `D-REFAV1-SEED-GOAL-MISMATCH` measured a ~0.30 m ADE inference-seed floor on refav1). A third
   question — *"would another INFERENCE RUN say this?"* — rides on the same interval.
⇒ **§5 states every bar as a RATIO to a floor measured in the SAME panel, and §7 funds BOTH
replicate kinds** (a training replicate and an inference-seed replicate). An effect smaller than
either floor is not an effect.

**Eval tier:** every planner bar in §5 is **T1** (`taniteval/tools/t1_eval.py` — the tier binding
ruling renamed this *self-action open loop*: a planner feeding its own predictor is still open loop,
and closed loop needs AlpaSim or a vehicle). The §5A mechanism bars are **CPU instrument checks on
the module** and carry **no tier** — no model there emits a trajectory, so a tier stamp would be a
category error.

**This document is written BEFORE any WP-B training number exists. Both outcomes are committed in
§5, with the thresholds fixed here, in advance.**

---

## 0. The claim under test

> **`E-WP-INDEX-1`** — Letting each anchor query's **own waypoint estimate, in metres**, address the
> **sparse agent tokens** — DiffusionDrive coupling (1), the trajectory's geometry as the attention
> ADDRESS rather than a second latent to fuse — improves REF-C's planner on the LONGITUDINAL family
> (headway / time-gap / TTC) beyond what the un-indexed agent cross-attention already gives, and it
> does so for **content** reasons rather than **capacity** reasons.

⭐ **It is an INDEX, not a fusion block, and that distinction is the work package.** A fusion block
concatenates or gates two learned latents and lets a network decide how to mix them; the mixing is
then a function of *content*. Here the query's latent `q` **never enters the address computation** —
`refc_wp_index.waypoint_agent_geometry` takes two tensors of metres and nothing else, which is the
audit in signature form. Swap the trajectory's geometry and the addressed agent changes
deterministically; that is what §6's analytic guards assert.

⭐ **PUBLISHED PRECEDENT FOR OUR EXACT DEGRADED SHAPE.** DiffusionDrive ran this coupling on
Transfuser with BEV-only spatial cross-attention and **agent cross-attention only, no map**, which
is the configuration refcv5 is in. `PUBLISHED` — not an extrapolation to our setting.
⚠️ **Evidence class of the precedent is `PUBLISHED-SECONDARY` until the primary is banked** — see
§9.2. It is cited for *design justification*, and **no numeric claim in this document rests on it**.

---

## 1. The chain of MEASURED facts this rests on

| # | fact | class · source |
|---|---|---|
| 1 | ⛔ **The address must go into the SPARSE tokens, never a dense BEV raster.** A median of **4 BEV cells (max 313)** share one token cell and only **242 of 640** token cells receive any ground-plane cell, so **even a perfect front-end caps at AP 0.4713** — a floor **no training removes**. | **MEASURED** — WP-A, `E-READOUT-CEILING-1` |
| 2 | `refc_agents.slot_features` already carries **continuous metric range and bearing** per slot (15 features, `TOKEN_FEAT_DIM`), decoded in **metres** by `AgentSlotDecoder` (`SlotDecodeRanges`: `x_fwd_m` 60.0, `y_half_m` 16.0). No grid, so no quantisation floor. | **MEASURED** — source, `refc_agents.py:96-120, 214-244`; `agent_slots.py:236-249` |
| 3 | refcv5-v2 runs at `--image-hw 256 640`; REF-C's ResNet is **stride 32**, so its map is **8×20 = 20 azimuth columns = 6.0°/column**, and the anchor decoder already cross-attends its 160 flattened tokens. | **MEASURED** — `refc.py:294-296, 333-337`; the anchor decoder's `feat_proj`/`CrossAttnLayer` path |
| 4 | ⛔ **refcv5 has NO `SpatialGridReadout` at all.** The "4 readout columns / 30° per bin" figure is a **v6/v7** fact and does not apply to REF-C. There is no readout here to change. | **MEASURED** — WP-A §1.2; `grep SpatialGridReadout stack/tanitad/refs/refc.py` returns nothing |
| 5 | The decoder's trajectory estimate is **[B, N, S, 2] in METRES, ego frame, x forward / y left** at every point the index reads it — `x0`/`x` in `_decode`, `x_path` in `_decode_ctrl` (which is `_state_to_path`'s integrated output). Same frame, same units as the agent boxes ⇒ **the address is a subtraction, not a projection.** | **MEASURED** — `refc.py::_decode`, `::_decode_ctrl`, `::_state_to_path`; `kinematic.rollout_unicycle` (`x += v cos(yaw) dt`) |
| 6 | The agent seam exists and is wired end-to-end but is **gated behind `--agents off`**; `CrossAttnLayer.cross_agent` + `agent_gate` (zero-init) are the seam. | **MEASURED** — `refc.py::CrossAttnLayer`, `refc.py::RefCModel.__init__`, `refc_v3_train.py::_pin_refcv5_seams` |
| 7 | ⛔ **Turning agents on at 108 M / 40 k is WP-C — a compute + PI decision** (`PI_DECISION_QUEUE.md` item 9) on an arm that **FAILED its two-seed tiny-rig gate**. WP-B does not touch that decision and leaves it exactly as open as it found it. | **INHERITED** — `PI_DECISION_QUEUE.md`, not re-verified here |

⚠️ **Fact 7 is the blocker on §7's arm, and it is named rather than worked around.** WP-B is
implemented, tested and inert; it cannot launch until WP-C is authorised, because an index with no
agent tokens is refused at pin time (§6.5).

---

## 2. What the waypoint index addresses, and how the address is computed

### 2.1 The address, exactly

For anchor query `n` with waypoints `W_n = {w_s}_{s=1..S}` (metres, ego) and agent slot `m` at
`p_m = (cx, cy)` (metres, ego — `agent_slots["box"][..., :2]`),
`refc_wp_index.waypoint_agent_geometry` computes, **with zero parameters**:

| quantity | definition | what it says |
|---|---|---|
| `d_min` | `min_s ‖p_m − w_s‖` | *does this plan pass near this thing* |
| `s_star` / `tau` | `argmin_s`, and `s_star/(S−1)` | **WHEN** on the plan the closest approach happens |
| `lon` | `(p_m − w_{s*}) · h`, `h` the plan's local heading at `s*` | along-track offset |
| `lat` | `(p_m − w_{s*}) · h⊥`, `h⊥ = (−h_y, h_x)` | cross-track offset, **left positive** |
| `d_range` | `‖p_m‖ − ‖w_{s*}‖` | signed metric range difference |
| `cos_db`, `sin_db` | bearing of `p_m` relative to bearing of `w_{s*}` | angular offset, as a pair so there is **no `atan2` wrap** to learn around |

⭐ **The local heading is measured with the EGO ORIGIN PREPENDED**: waypoint 0's heading is
`w_1 − (0,0)`, the direction the plan sets off in. Dropping that prepend is an off-by-one that
shifts every heading by one slot — **mutation `M6`, and it initially escaped the entire guard set**
(§6.4).

These eight quantities, scaled by `scale_m` (10.0 m, a NORMALISER not a cut-off), are the
`[B, N, M, 8]` relation. A per-layer 2-layer MLP maps it to a **per-head additive attention logit
bias** `[B·H, N, M]`, which is exactly `nn.MultiheadAttention`'s float `attn_mask`.

### 2.2 Why a bias and not a gather

An additive per-head logit bias **is** the index: it is computed only from geometry, and it decides
which key receives the softmax mass. §6.1's analytic guard measures that steering in closed form —
with all agent tokens made identical (so every content logit is equal), the mass ratio between two
slots is exactly `exp(bias gap)`, and at a gap of 6.0 it reads **403.4288** (`math.exp(6.0)`).

The **hard**, strictly-local variant — the closer analogue of deformable attention's bounded
sampling offsets — is implemented as `--wp-index-radius-m` and is **OFF by default**. It is the
pre-registered **next lever** if the soft arm misses its bar (§5C).

### 2.3 ⛔ What WP-B is NOT

* Not a dense-raster index (fact 1: a floor no training removes).
* Not a readout change (fact 4: refcv5 has no readout).
* Not a new information channel: `agent_slots["box"]` is what the agent cross-attention **already
  consumes** through `AgentTokenEmbed`. WP-B changes *how the attention is addressed*, not *what the
  planner can see*. ⇒ the vision-only rule is untouched: the head still takes `fmap` and nothing
  else, and on the `oracle` path the arm is stamped inadmissible as a capability claim exactly as
  WP-6 already stamps it.

---

## 3. `one_variable` — the launch commands, DIFFED

⛔ **Diffed as parsed namespaces, not as intent.** `code/check_trainer.py` parses both arms and
prints the symmetric difference. **MEASURED today:**

```
launch_diff_B0_vs_B1 = { "wp_index": ["off", "on"] }        # exactly one key
```

B0 (control) and B1 (treatment) are otherwise character-identical:

```
# B0 — the WP-C arm as it would run without WP-B
refc_v3_train.py --arm hier --out <run> --steps 40284 \
    --agents oracle --agent-join <join> ... [refcv5-v2's full flag set]

# B1 — the same command, plus ONE flag
    ... --wp-index on
```

⚠️ Every other WP-B knob keeps its default in both arms, and `--wp-index off` **REFUSES** any
non-default knob (§6.5), so a stray `--wp-index-mode shuffle` cannot silently ride along in a
record while doing nothing.

---

## 4. The arm ladder — and why the ORACLE rung comes first

| rung | `--agents` | `--wp-index` | what it answers | admissible as a capability claim? |
|---|---|---|---|---|
| **B0** | `oracle` | `off` | the agent seam alone | ⛔ **NO** — reads a privileged label at inference |
| **B1** | `oracle` | `on` (geom) | **does the ADDRESS help, given good agents?** | ⛔ **NO**, same reason |
| **B1-shuf** | `oracle` | `on` (shuffle) | content or capacity? | control |
| **B1-const** | `oracle` | `on` (const) | the no-information address | control |
| **B1-det** | `oracle` | `on` (detach) | is the second-order gradient path load-bearing? | control |
| **B0r / B1r** | as B0 / B1 | as B0 / B1 | **the replicate floor** — training seed AND inference seed | floor |
| **C1** | `head` | `on` | the deployable arm | ✅ yes — pending WP-C |

⭐ **The oracle rung is deliberately first, and it is the cheapest discriminating experiment.** It
separates *"does the index mechanism work"* from *"can we detect agents well enough for it to have
anything to address"*. Running C1 first would confound the two, and a null would be uninterpretable —
the C6 confound in a new costume. ⛔ **Nothing in the oracle ladder may be quoted as a capability
claim**, and `AgentSeamConfig.oracle` is stamped into `config.json` so no reader can mistake one for
the other.

---

### 4.1 ⭐⭐ THE ORACLE RUNG IS NOT A CONVENIENCE — IT REMOVES THE EXACT CAUSE WP-C's GATE MEASURED

⛔ **This is the single most important design fact in the ladder, and it comes from the PI decision
queue itself.** `PI_DECISION_QUEUE.md` item 9 records that turning the agent seam on **DEGRADED the
distance-keeping family at BOTH seeds** — MEASURED, `head − off`, T1, paired episode-cluster
bootstrap, n = 776–779 windows / 28 episodes:

| distance-keeping | seed 0 | seed 1 |
|---|---|---|
| min headway (m) | **−0.5015** [−0.7712, −0.2366] | **−0.4535** [−0.7915, −0.1639] |
| min time-gap (s) | **−0.0702** [−0.1074, −0.0385] | **−0.0351** [−0.0694, −0.0037] |
| min TTC (s) | **−2.1959** [−4.1408, −0.6110] | **−1.2529** [−2.2522, −0.3694] |

⚠️ **That is the very family WP-B's primary bar sits on.** A WP-B arm layered on `--agents head`
would be measuring the index **through** a known, separated, two-seed regression — the C6 confound
in a new costume, and a null would be uninterpretable.

⭐ **But item 9 also RELOCATED the cause, and the relocation is what makes the oracle rung correct
rather than merely cautious.** A third arm with the join's `clip_id` **deranged** reproduces the
**entire** degradation, while `head − shuf` is **not separated on any distance-keeping metric at
either seed** ⇒ the cost is **the auxiliary DETECTION TASK competing for a 17 M trunk at 500 steps,
not the agent information**.

⇒ **`--agents oracle` has no detection loss at all.** `OracleAgentEmbed` has no parameters to
supervise and the trainer requires `--w-agent > 0` only on the `head` path, so B0/B1 run with
**`--w-agent 0`** and the measured cause is **structurally absent**, not merely hoped away.

⇒ **the ladder therefore asks two separable questions in the right order:**
1. **B1 − B0 (oracle, no detection loss):** *does the ADDRESS help, given agents?* — free of item 9's
   confound;
2. **C1 (head):** *can a detector supply agents good enough for it?* — which inherits item 9's
   open question (**"whether that cost vanishes at 108 M over 40 k steps is UNTESTED"**) and must not
   be run before (1) returns.

⛔ And the honest framing item 9 demands travels with this document: any retry of the head path is
*"an arm whose exclusion may not generalise"*, **never** *"recovering a lever we know is there"*.

⚠️ **INHERITED, not re-verified here.** The item-9 table is quoted from
`Project Steering/PI_DECISION_QUEUE.md` §9 and was not re-measured by WP-B. It is admissible for
*design* (it selects the ladder's order); it is **not** admissible as a WP-B result.

---

## 5. SUCCESS and FAILURE, committed in advance

⛔ Every bar is **stated as a ratio to a floor measured in the same panel**, because a separated CI
from a one-seed arm is necessary and not sufficient (header). `Δ` is always B1 − B0, paired,
same windows, episode-cluster bootstrap. Every table prints **n windows and n episodes**.

### 5A. Mechanism bars — CPU instrument checks, **no tier** (ALREADY MEASURED, §6)

| # | bar | verdict today |
|---|---|---|
| A1 | the address reproduces analytic metric literals (`6.0`, `sqrt(436)−20`, `e**6`) | ✅ **MET** |
| A2 | flag OFF ⇒ planner output **bitwise** identical to a build that never had the seam | ✅ **MET** |
| A3 | every mutation turns ≥1 guard RED; baseline all-GREEN | ✅ **MET** (12 guards, 9 mutations) |
| A4 | each of the three controls reads its known value, VERIFIED | ✅ **MET** |

### 5B. Planner bars — **T1**, four families, paired episode-cluster bootstrap

⛔ **ADE alone is not the result.** All four families are reported per family, never pooled.

| family | primary statistic | **SUCCESS** | **FAILURE** |
|---|---|---|---|
| **LONGITUDINAL** (the pre-declared primary — 88.7 % of the oracle gap lives here, and headway/TTC are *by definition* functions of range and bearing, which is what the address is) | time-gap error to the lead agent at 2 s | `Δ` separated in B1's favour **and** `|Δ| ≥ 3 ×` the replicate floor on the same statistic | `Δ` not separated, **or** `|Δ| <` the replicate floor, **or** separated in B0's favour |
| **LATERAL** | curvature error, yaw-rate error, cross-track | must not regress: `Δ` not separated in B0's favour | any lateral statistic separated in B0's favour ⇒ the arm FAILS regardless of the longitudinal result |
| **TACTICAL** | selected-vs-executed manoeuvre, 5-way confusion, anchor-selection agreement | reported with `n`; no bar (WP-B is not a tactical lever) | — |
| **STRATEGIC** | route/goal setting quality | reported with `n`, or **stated absent with its reason and `n`** | — |
| **ADE** (accompanying, never the headline) | `ade_m` at 1/2/3 s | reported with the same estimator and windows | — |

### 5C. THE FAILURE TWINS — each names its next lever, and each lever is BUILT

⛔ Rule Zero: *"X is refuted"* is never a finished turn.

| if B1 fails because… | the reading | **the next lever, already implemented** |
|---|---|---|
| `Δ` is inside the replicate floor | the effect, if any, is under this rig's noise | **more SEEDS, not more windows** (`H-ESTIM-SEED-1`) — and on this rig the inference seed too, because the sampler samples |
| **B1-shuf ≈ B1** | the gain is CAPACITY, not content | drop the bias head to `--wp-index-hidden 8` (**+152 params/layer** at H=8) and re-run; if the gain survives at 1/3 the capacity it was never capacity |
| **B1-const ≈ B1** | the address carries no anchor-specific information — the attention is not being steered | ⇒ **`--wp-index-radius-m 12`**: a soft bias may be too weak to overcome content logits; the hard local gate makes the address decisive rather than advisory |
| B1 helps LONGITUDINAL but regresses LATERAL | the index is pulling the plan toward agents | **`--wp-index-detach`** (the address stops shaping the trunk) and, if that is not enough, restrict the address to the **first `feasible_prefix_slots` waypoints** so the index cannot reach past the horizon the scorer differentiates |
| **B1-det ≈ B1** | the second-order path through the address is inert | keep `--wp-index-detach` as the **default** — it is strictly cheaper and removes a gradient route into the trunk that nothing needs |
| B1 ≈ B0 in **every** family and every control | the coupling does nothing **on the oracle rung**, which is the strongest possible negative | ⛔ **do NOT run C1.** Report the elimination; the remaining DiffusionDrive lever is coupling (2), the cascade's second stage, which is a different work package |

### 5D. The deliberate regression the panel MUST catch

**B1-const** is a genuine deliberate-regression arm: it feeds a no-information address through the
same parameter count and the same gradient path. If the panel cannot separate B1 from B1-const, the
panel has no power to detect this lever at all and **no verdict may be issued** — the arm is
`UNDERPOWERED`, not `REFUTED`. *(This is the `n ≪ d` failure of the 2026-08-22 ridge probe, in
planner costume: maximal regularisation read `+0.0000` for the latent AND the constant control, and
the panel concluded "the latent carries no dynamics".)*

### 5E. `n` and `d`, in every table

`d` = 8 address features (named, `WP_INDEX_FEATURES`, and carried in `config.json`).
`n` = windows and episodes, printed per family per row. A row without both is not admissible.

---

## 6. The controls, their KNOWN VALUES, and whether each was VERIFIED to read it

⛔⛔ **A cross-check must be derived independently of the value it checks.** Every expectation below
is a **LITERAL** — computed by hand or by `math` — never an expression over the code under test.
Artifacts: `code/mutation_proof.py`, `raw/mutation_log.json`, `raw/MUTATION_LOG.md`,
`stack/tests/test_wp_index.py`.

### 6.1 Analytic — the address resolves a computable slot

Scene (all literals): a straight plan at `(5,0) (10,0) (15,0) (20,0)`; agents at `(20,0)`,
`(20,6)`, `(5,0)`.

| quantity | **literal** | MEASURED | verified |
|---|---|---|---|
| `d_min` | `[0.0, 6.0, 0.0]` | `[0.0, 6.0, 0.0]` | ✅ |
| `s_star` | `[3, 3, 0]` | `[3, 3, 0]` | ✅ |
| `tau` | `[1.0, 1.0, 0.0]` | `[1.0, 1.0, 0.0]` | ✅ |
| `lat` | `[0.0, 6.0, 0.0]` | `[0.0, 6.0, 0.0]` | ✅ |
| `d_range` (agent 1) | `sqrt(436) − 20 = 0.880613` | `0.880613` | ✅ |
| `cos_db` / `sin_db` (agent 1) | `20/sqrt(436)` / `6/sqrt(436)` | `0.957826` / `0.287348` | ✅ |
| **attention mass ratio** at a bias gap of 6.0, all agent tokens identical | `e**6 = 403.4287934927` | matches to `rel=1e-3` | ✅ |
| first-waypoint heading on a TURNING plan, agent at `(5, 1.5)` | `lon = 0.5/sqrt(25.25)`, `lat = 5.0/sqrt(25.25)` | matches to `1e-5` | ✅ |

⭐ **The discriminating control:** the same agent set with a LEFT-going plan addresses agent **1**,
not agent 0. A positive assertion on the straight plan alone passes on an implementation that always
returns slot 0.

### 6.2 The three pre-registered controls

| control | **known value** | MEASURED | verified |
|---|---|---|---|
| **detached** | the graph is **severed** — `rel.requires_grad is False`, `grad_fn is None`, both operands' `.grad` stay exactly `None` | as stated; the treatment's gradient is non-zero on **both** operands | ✅ |
| **shuffled** | row `b`'s relation equals **bitwise** the geometric relation of row `perm[b]` | `torch.equal` | ✅ |
| **constant** | the bias is **bitwise identical across the anchor axis** for every anchor | `torch.equal` for all `n`, with a **non-zero-init** head so the check cannot pass trivially; and the treatment is verified **not** to have that property | ✅ |

⛔ `shuffle` **refuses `B < 2`** — a batch permutation of one row is the identity, so the control
would silently BE the treatment. Verified to raise (`ValueError: batch >= 2`).

### 6.3 ⭐ A DEFECT THIS PRE-REGISTRATION'S OWN CONTROL FOUND — in my own guard

⛔⛔ **The removability proof was GREEN FOR A REASON THAT WAS NOT WP-B, and a discriminating control
is what caught it.** `agent_gate` is zero-init (WP-6's own discipline), so the *entire* agent branch
contributes exactly `0` at step 0 — and the "index on vs off is bit-identical" assertion therefore
**passes even with a deliberately corrupted bias head** (MEASURED: `mlp[-1]` filled with
`N(0, 10)`, planner output still bitwise equal). A guard whose PASS does not depend on the thing it
guards is the `CLAUDE.md` *"a check that shares the defect it checks for"* family.

⇒ two tests now exist rather than one:
`test_the_shipped_init_proof_is_not_carried_by_the_index_alone` states the confound as an executable
fact, and `test_at_a_LIVE_agent_gate_the_zero_init_index_agrees_to_float32_ROUNDING` re-runs the
comparison with `agent_gate` opened to 1.0 — where the honest number is **not** bit-identity:

MEASURED, `raw/removability.json`, smoke rig, seed 1234, dev-box CPU float32:

| regime | max-abs `traj` difference, index-ON vs index-OFF | reading |
|---|---|---|
| `agent_gate = 0` (shipped init), **zero-init** bias | **0.0** — bitwise | the required proof |
| `agent_gate = 0`, **CORRUPTED** bias (`N(0, 10)` on `mlp[-1]`) | **0.0** — bitwise | ⛔ **THE CONFOUND, MEASURED.** The agent branch is multiplied away, so the proof above passes on a head that is garbage |
| `agent_gate = 1`, zero-init bias | **1.9073486328e-06** (`= 2**-19`) | float32 kernel rounding — supplying a float `attn_mask` **at all** changes which SDPA kernel torch dispatches to |
| `agent_gate = 1`, bias `N(0, 1)` on `mlp[-1]` | **2.169e-03** (`stack/tests/test_wp_index.py`) | the lever, three orders above the rounding floor |
| `agent_gate = 1`, bias `N(0, 10)` on `mlp[-1]` | **5.391e-03** (`raw/removability.json`) | ditto, at 10× the scale |

⇒ **the report says "agrees to float32 rounding", never "bit-identical", in the live regime** — and
the second row is why the first row alone is never quoted as the proof.

**The required proof, separately and unambiguously** (`raw/removability.json`
`required_flag_off_vs_no_field`): flag OFF vs a build that never had the field —
**182 of 182 state-dict entries bitwise equal**, key lists identical, **no `wp_index` key present at
all**, and **15 of 15 emitted planner tensors bitwise equal**. Index-OFF vs index-ON shares
**182 of 182** parameters bitwise, with **+8** new keys and nothing else touched.

### 6.4 ⛔ THE MUTATION PROOF — the guard set CAN go RED, and it was INCOMPLETE

`code/mutation_proof.py --out raw/` · `torch 2.11.0+cu128` · `python 3.13.5` · dev-box CPU, float32 ·
**12 guards · 9 mutations** · **exit 0 = PASS**.

| mutation (the REAL defect) | guards that went **RED** |
|---|---|
| `M1` index by ARRAY POSITION instead of geometry | G1, G2, G3, G5, G7, G12 |
| `M2` swap x and y (the `anchors.pt` units family, one axis over) | G1, G2, G3, G4, G12 |
| `M3` ignore the `detach` flag | G7 |
| `M4` transpose the bias to `[B·H, M, N]` | G4, G5, G10 |
| `M5` non-zero output init | G11 |
| `M6` drop the origin prepend (heading off-by-one) | **G12** |
| `M7` radius gate ignores padding in its emptiness test | G9 |
| `M8` shuffle accepts `B = 1` | G6 |
| `M9` no denominator clamp (NaN on a padded slot at the origin) | G8 |

⭐⭐ **`M6` turned NO guard RED on the first run, and that is the most valuable line in this
document.** Every analytic guard used the STRAIGHT plan, whose per-step heading is constant, so
borrowing step 1's heading for step 0 changed nothing. **G12** was written to close it: a TURNING
plan whose closest approach is at `s* = 0`, with literals `0.5/sqrt(25.25)` and `5.0/sqrt(25.25)`.
⇒ the mutation proof is not ceremony; it found a hole in the guard set that no amount of re-reading
the code would have.

⚠️ `M7` was a bug **actually present in the first draft** of `apply_radius_gate` — the emptiness
test looked only at the radius, so a query whose only in-range slot was PADDING kept its gate and put
every unit of attention mass on a slot that does not exist. Caught by G9 before it shipped.

### 6.5 The refusals, VERIFIED to FIRE

| situation | must refuse | MEASURED |
|---|---|---|
| `--wp-index on --agents off` | *"there are no tokens to address … a REFUTATION MANUFACTURED BY A MISSING SEAM"* | ✅ refuses at pin time, before the GPU |
| `--wp-index off --wp-index-mode shuffle` (and `--wp-index-detach`, `--wp-index-radius-m`) | no-op knob stamped into `config.json` | ✅ refuses — *"`--wp-index-mode` is the dangerous one: it names a CONTROL ARM, so a reader would believe a control had been run"* |
| `wp_index.enable` with `cross_agent` False, at model construction | ⛔ | ✅ `ValueError: no agent cross-attention` |

### 6.5b ⛔⛔ A DEFECT THIS PRE-REGISTRATION SHIPPED, AND THE REPO'S OWN GUARD CAUGHT IT

The refusals in §6.5 were **incomplete when first written**. They covered `--wp-index-mode`,
`--wp-index-detach` and `--wp-index-radius-m`, and left `--wp-index-hidden`, `--wp-index-scale-m`
and `--wp-index-const-xy` free — three flags that parse, are stamped into `config.json` under
`agent_knobs`, and do nothing. That is the **M18 no-op-flag defect verbatim**, inside the package
whose §6.5 table asserts the class is closed.

**MEASURED:** the full 7,520-test suite returned `9 failed, 7449 passed`, and exactly one failure was
WP-B's — `test_refc_v3_agent_provenance.py::test_P2_every_knob_is_recoverable_from_the_stamp_BY_VALUE`,
which reported *"no admissible probe value for `--wp-index-mode`"*.

⭐ **It was caught because that test derives its knob list FROM ARGPARSE rather than from a
hand-written list** — a new flag enrols itself, and there is no list to forget. A hand-maintained
list would have rotted silently, which is the whole reason it is written that way.

**Closed:** all six knobs refuse under `--wp-index off`; six parametrised refusal cases plus a
**same-breath control** (the knobs at their defaults must NOT trip the refusal, or `--wp-index off`
would be unusable and the six cases would pass for the wrong reason) now live in
`stack/tests/test_refc_v3_refcv5_wiring.py`. The provenance test gains WP-B's enabling context, as
it already carries WP-D's.

⚠️ **Logged rather than quietly repaired**, because the lesson is the transferable part: a guard
list written by hand beside the flags it guards is the thing that rots; a guard derived from the
parser cannot.

### 6.6 `assert_seams_are_built` — BIDIRECTIONAL, plus a third direction

| direction | MEASURED |
|---|---|
| stamped **and** built | PASSED |
| stamped, **not** built | refuses — *"the record would claim a waypoint index the weights do not contain"* |
| built, **not** stamped | refuses — *"a live seam absent from the run record"* |
| **stamped mode ≠ attached mode** | refuses — *"a CONTROL arm would be reported as the treatment, or the reverse"* |

⭐ The third pair is the one WP-4's `sampler` defect did not have: with WP-B, a run whose record says
`geom` while the weights carry `const` would report a control as the treatment, and that is worse
than either half of the classic false-provenance failure.

---

## 7. The GPU arm — precise, costed, and NOT started

⛔ **BLOCKED on the PI, and the blocker is named rather than worked around: WP-C
(`PI_DECISION_QUEUE.md` item 9) has not authorised turning the agent seam on at 108 M / 40 k.**
The live refcv5-v2 arm carries **`--agents off`** (its `LAUNCH_RECEIPT.md` §1, line 39), so B0 is
not "refcv5-v2 plus a flag" — it is refcv5-v2 **plus the whole WP-C decision**. WP-B changes nothing
about that decision.

### 7.1 ⛔ THE COST IS ~14 GPU-DAYS, NOT "when a GPU frees" — and the rate I nearly quoted is one
the receipt itself RETRACTS

**MEASURED**, `…/2026-09-07-refcv5-v2-compose/LAUNCH_RECEIPT.md` §4 + its own correction block:
the sustained rate of the live refcv5-v2 arm on the A40 is **4.0 s/step** (50 steps per 200 s,
01:23–01:29Z). ⚠️ The same receipt **explicitly retracts** its earlier `~1.2 s/step ⇒ ~13.5 h`
because `--log-every 50` makes `step_s` an accumulated figure — the `step_s` trap, in this exact
run's own log. ⇒ **40,284 steps ≈ 44.8 h per arm**, before the agent head.

| item | value | class |
|---|---|---|
| refcv5-v2 sustained rate, A40, `--batch 20 --workers 6` | **4.0 s/step** | **MEASURED** (its receipt) |
| one 40,284-step arm | **≈ 44.8 h** | derived from the above |
| the agent head's step cost | **+8.2 %** (3.346 → 3.622 s, non-overlapping medians) | **MEASURED** but ⚠️ **on dev-box CPU at 256×640, `n_pad 94`** — a scope caveat, not an A40 number |
| ⇒ one B-ladder arm | **≈ 48 h** (ESTIMATED — the +8.2 % is ported across hardware) | **ESTIMATED** |
| the full 7-arm ladder in §4 | **≈ 14 GPU-days** | **ESTIMATED** |

⛔ **14 GPU-days is not a footnote, it is the finding.** The ladder as designed is a
PI-scale spend and must not be presented as an opportunistic gap-filler.

### 7.2 ⭐ THE CHEAPER FIRST CUT, and why it is scientifically sufficient

Every §5B bar is a **paired** contrast whose threshold is a **ratio to a replicate floor measured in
the same panel**. That construction is scale-free: it is valid at 6,000 steps exactly as at 40,284,
because both the effect and the floor are measured at the same budget. ⇒ the mechanism question
(*"does the address steer the plan at all?"*) does not need a full-length arm.

| # | arm | steps | ≈ wall-clock | what it decides |
|---|---|---|---|---|
| 1 | **B0-short**, **B1-short** | 6,000 | 2 × ≈ 6.7 h | does the index move ANY family beyond the floor? |
| 2 | **B0r-short**, **B1r-short** (seed) | 6,000 | 2 × ≈ 6.7 h | **the replicate floor** — without it neither of the above is readable |
| 3 | **B1-const-short**, **B1-shuf-short** | 6,000 | 2 × ≈ 6.7 h | content vs capacity, and the deliberate regression the panel must catch (§5D) |
| — | **gate** | — | — | ⛔ if B1 − B0 is inside the floor, or B1 ≈ B1-const, **STOP**: the full ladder is not funded |
| 4 | full-length B0/B1 (+ controls) | 40,284 | ≈ 14 GPU-days | only if the gate is cleared |

**≈ 40 GPU-hours** for the whole gate, against ≈ 14 GPU-days for the ladder it protects.

**Eval, every rung:** `taniteval/tools/t1_eval.py`, **T1**, four families, paired episode-cluster
bootstrap, **≥ 3 INFERENCE seeds per checkpoint** (the arm carries `--sampler ddim --w-u0 0.5`, so
the planner samples and one decode is one sample). ≈ 11 min/arm/seed on the dev-box 4060 — the eval
is free relative to the training and is **not** a reason to skip seeds.

### 7.3 The exact flags

```
# B0  (control) — refcv5-v2's flag set, PLUS the WP-C agent seam
refc_v3_train.py --arm hier --out <run>/B0 --steps 6000 \
    [refcv5-v2's full flag set: --sampler ddim --w-u0 0.5 --sel-refined \
     --sel-score-emitted --batch 20 --workers 6 --v2-lru 24 --u8-batches ...] \
    --agents oracle --agent-join <join> --w-agent 0

# B1  (treatment) — the SAME command plus ONE flag
    ... --wp-index on
```
⛔ **`--w-agent 0` is load-bearing, not a default.** The oracle path has no detector to supervise,
and a detection loss is precisely what item 9 MEASURED as the cause of the distance-keeping
regression (§4.1). An oracle arm carrying `--w-agent > 0` would re-import the confound the rung
exists to remove.
⛔ Diffed as parsed namespaces: **exactly one key differs** (§3).

### 7.4 ⚠️ The cost that is NOT measured, and is the real risk

**MEASURED ON THE REAL MODEL, not extrapolated** (`raw/e2e_real_model.json`; `v3.RefCV3Model` built
from the actual parsed args at `--image-hw 256 640 --agents oracle`, dev-box CPU):

| | index OFF | index ON | delta |
|---|---|---|---|
| decoder layers with `cross_agent` / with `wp_index` | 4 / **0** | 4 / **4** | — |
| `param_breakdown["wp_index"]` | 0 | **2,208** | **+2,208** |
| `param_breakdown["decoder"]` | 11,003,021 | 11,003,021 | **0** — carved out, not added on |
| `param_breakdown["total"]` | 106,666,205 | 106,668,413 | **+2,208 = +0.00207 %** |
| `state_dict` keys containing `wp_index` | **0** | 16 | — |
| `assert_seams_are_built` | PASSED | PASSED | — |

⭐ **+2,208 is exactly the analytic form** `(8h + h + hH + H) × layers` = `(256+32+256+8) × 4` — an
**independently derived** target the built model hit on the nose, not a figure read back from the
code. *(On the smoke rig the same form predicts and measures **+840** at `n_heads 4, layers 2`.)*

**ACTIVATION memory is the risk, and it is priced ON PAPER ONLY.** At the deployed shape
(`B=20, N=117 anchors, M=100 slots, S=8, H=8, 4 layers`), float32:

| tensor | shape | size |
|---|---|---|
| the displacement intermediate | `[B, N, S, M, 2]` | **15.0 MB** |
| the relation | `[B, N, M, 8]` | 7.5 MB |
| the MLP hidden | `[B, N, M, 32]` | 30.0 MB / layer |
| **the bias** | `[B·H, N, M]` | **74.9 MB / layer** |

⇒ order **0.4 GB per decoder pass**, and the decoder runs **three** passes per forward (classifier +
2 DDIM steps) with autograd retaining them ⇒ order **1.3 GB**. On a 48 GB A40 that is affordable;
it is **not** affordable silently.

⛔ **This is an ESTIMATE and must be re-measured on the real path before launch.** Porting a paper
figure onto a live configuration is exactly the `e_trunk_pooling` failure (a dense windowed tensor
that looked cheap and was 25.6 GB) and the `--v2-cache` artifact-cost failure (a 1.38 TB "wall" that
did not exist). The cheap mitigation if it binds is `--wp-index-hidden 8` (30.0 → 7.5 MB/layer);
the structural one is to compute the bias in `bf16`, which is a one-line change and is **not**
made here because it would be an unmeasured second variable.

## 8. What this pre-registration does NOT claim

1. ⛔ **No claim that WP-B helps.** No arm has run. Every §5B cell is empty by design.
2. ⛔ **No capability claim from any oracle rung.** B0/B1/B1-* read privileged labels at inference.
3. ⛔ **No claim about the DENSE raster route.** WP-A's floor stands; WP-B does not test it.
4. ⛔ **No claim that WP-C should be authorised.** §4's ladder is what WP-B would run *if* it is; the
   decision is the PI's and is untouched.
5. ⚠️ **"bit-identical" is claimed only for the flag-OFF build.** In the live-`agent_gate` regime the
   honest number is `1.907e-06` (§6.3), and it is stated that way everywhere.
6. ⚠️ **The DiffusionDrive precedent is `PUBLISHED-SECONDARY`** until the primary is banked (§9.2).
   It justifies a design; it supports no number here.
7. ⚠️ Everything MEASURED here is **dev-box CPU, float32, `torch 2.11.0+cu128`**. A pod-side or
   CUDA re-measure of the removability proof is owed before the arm launches.

---

## 9. Deliverables that exist as of this pre-registration

### 9.1 Code and tests — staged, `pytest -q` green

| path | what |
|---|---|
| `stack/tanitad/refs/refc_wp_index.py` | the address, the bias head, the controls, the radius gate |
| `stack/tanitad/refs/refc.py` | gated wiring: `DecoderConfig.wp_index`, `attach_wp_index`, `_agent_index`, `agent_pos` threaded, `param_breakdown` line |
| `stack/scripts/refc_v3_train.py` | 7 flags, 2 refusals, the stamp, the bidirectional seam assertion |
| `stack/tests/test_wp_index.py` | **35 tests** — analytic literals, the three controls, both NaN traps, removability, 4 mutations |
| `…/Research/2026-09-08-wpb-waypoint-index/code/mutation_proof.py` | 12 guards × 9 mutations, exit-coded |
| `…/Research/2026-09-08-wpb-waypoint-index/code/check_trainer.py` | the launch-command diff and the refusal/stamp/seam probe |
| `…/Research/2026-09-08-wpb-waypoint-index/raw/` | `mutation_log.json`, `MUTATION_LOG.md`, `trainer_wiring.json`, `removability.json` |

### 9.2 ⛔ OWED — banking the primary

The DiffusionDrive precedent is cited for design justification and is **`PUBLISHED-SECONDARY`**: the
primary is not banked in `TanitAD Research Lab/Library/`. `python tools/kb_add.py <arxiv-id> --tag
planning --cited-by "<this file>"` converts it. ⚠️ **No numeric claim in this document depends on
it**, so nothing here is retracted by the gap — but §0's precedent sentence is not quotable for the
paper or the registry until the PDF is banked and `--verify` re-hashes it.
