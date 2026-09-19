# PRE-REGISTRATION — the agent seam and the collision gate (`H-SEL-GATE-1`)

**Written 2026-09-17, BEFORE any arm has been run.** Both outcomes committed below.
PI decision the same day on `PI_DECISION_QUEUE` **item 19**: *"for 3 chose defult"* —
option (a), **carry it as a named arm with its own pre-registration**. This is that file.

**Status: PRE-REGISTERED, NOT RUN.** Nothing here is a result.

---

## 1. The measured headroom this exists to reach

MEASURED at zero GPU on 493 held-out windows (item 19):

| | |
|---|---|
| windows where the **selected** plan collides (untouched `refcv5-v2`) | **28 / 493** |
| …of which the fan **still held a collision-free candidate** | **28 / 28 = 100 %** |
| mean share of the fan that was collision-free there | **55.3 %** |
| oracle-repair ceiling from fixing collision-selection **alone** | **+0.0485 `sel_pdms` [+0.0228, +0.0801] = 62.4 % of the entire oracle gap** |
| windows it covers | **25 / 493 = 5.1 %** |

⭐ Reproduces on **four checkpoints in three training states** (`base` 28/28,
`base_repeat` 28/28, `L1-NORL-s0` 25/25, `L1-RL-s0` 37/37) ⇒ a **`refcv5-v2` property**,
not an artifact of the RL experiment that surfaced it.

---

## 2. ⛔ The three things this pre-registration must not let anyone forget

1. ⛔ **The 62.4 % is a T0 ORACLE CEILING, not an available gain.** `sel_nc` is scored
   against the **recorded future**. At inference the vision-only rule forbids that, so a
   deployable gate needs a **PREDICTED** occupancy. **The deliverable is what the
   predicted gate buys; the oracle arm only says how much room there is.**
2. ⛔ **An oracle gap needs a RANDOM control.** *(Standing rule, cost a landed claim
   2026-09-17: best − actual is large for a GOOD selector too.)* Random, actual and
   oracle are quoted **on the same windows**, or none of them is quoted.
3. ⛔ **Two levers, not one.** The agent channel is a **representation** change; the
   collision gate is a **hard constraint at selection time**. Repairing both in one arm
   makes a PASS non-attributable — the `--v2` conflation precedent (ten levers on two
   axes) and the C6 confound.

---

## 3. Hypothesis

**`H-SEL-GATE-1`** — *A hard selection constraint — never select a colliding candidate
while a collision-free one is in the fan — driven by a **predicted** occupancy from the
refcv6 perception heads, recovers a measurable share of the collision-selection gap at
T1, without harming the other three metric families.*

**`H-SEL-AGENT-1`** — *Re-opening the structured agent channel (`--agents on`) improves
collision-relevant selection, independently of the gate.*

⚠️ `H-SEL-AGENT-1` is **an arm whose tiny-rig exclusion may or may not generalise**. The
2-seed tiny-rig result that gated `--agents off` was about an **auxiliary training task**;
this is about a **selection constraint**. The two do not meet, and neither overturns the
other.

---

## 4. Arms — ⛔ ONE VARIABLE EACH

| arm | the ONE change | what it tests |
|---|---|---|
| **`S1-BASE`** | none (refcv6 baseline, `--agents off`, no gate) | the base every delta is read against |
| **`S1-AGENTS`** | `--agents on` | `H-SEL-AGENT-1`: the representation lever alone |
| **`S1-GATE-PRED`** | collision gate on **predicted** occupancy (BEV map + 3-D box heads) | `H-SEL-GATE-1`: **THE DELIVERABLE** |
| **`S1-GATE-ORACLE`** | collision gate on the **recorded future** | ⛔ **T0 ONLY — the CEILING.** Never a capability claim, never quoted as driving performance |
| **`S1-RANDOM`** | selection replaced by a uniform draw from the fan | ⛔ **the no-information control.** Without it the oracle gap is unreadable |
| **`S1-GATE-CONST`** | the gate wired to a **constant** occupancy (everything free) | ⛔ **the deliberate-regression arm** — see §6 |
| **`S1-BOTH`** | `--agents on` **and** the predicted gate | only if a single-lever arm clears its bar |
| **`S1-REPL`** | `S1-BASE`'s flags, **different seed, zero levers moved** | ⛔ the run-to-run floor |

**Held constant:** corpus, split, steps, batch, trunk (`resnet34` for the matrix;
`resnet101` when a card allows), **416 × 1024**, anchors, `n_anchors` 117, the fan, the T1
harness, the held-out episodes. Every arm runs **two seeds**.

⛔ **Preflight refuses on an ARGV DIFF, not on intent.**

---

## 5. Committed criteria

### 5.1 PRIMARY — `H-SEL-GATE-1` (T1)

* **SUPPORTED** — `S1-GATE-PRED` reduces the **collided-selection rate** vs `S1-BASE`
  with a paired episode-cluster bootstrap CI excluding zero, **on both seeds**, by more
  than the `S1-REPL` floor — **and** `ade_m` and the other three families are not worse
  than `S1-BASE` by more than that floor.
* ⛔ **FAIL-HARM** — any family is worse than `S1-BASE` by more than the floor on either
  seed. A gate that fixes collisions by driving worse everywhere else has not helped.
* **REFUTED** — the collided-selection rate does not move beyond the floor.
* **INCONCLUSIVE** — `S1-REPL` shows the rig noisier than the effect.

### 5.2 The ceiling, reported but never claimed

`S1-GATE-ORACLE` is reported **beside** `S1-GATE-PRED` as *"what a perfect collision
checker would have bought on these windows"*, stamped **T0**, with `S1-RANDOM` and
`S1-BASE` on the **same windows**. The quotable number is
`(pred − base) / (oracle − base)` — **the share of the ceiling actually reached.**

### 5.3 `H-SEL-AGENT-1`

**SUPPORTED** only if `S1-AGENTS` improves the collision-relevant selection statistic
over `S1-BASE`, both seeds, beyond the floor. ⛔ It is **not** supported by an ADE move:
item 19's whole point is that this seam is pre-registered **against the collision
statistic, not against ADE**.

---

## 6. ⛔ The deliberate-regression arm

`S1-GATE-CONST` runs the identical gate with an occupancy that says **everything is
free**. The gate then reorders nothing and **must recover nothing**.

If `S1-GATE-CONST` recovers as much as `S1-GATE-PRED`, then the gain is coming from the
**re-ranking machinery**, not from the occupancy signal, and a PASS on `S1-GATE-PRED`
means nothing. *(A gate that does not FAIL the reintroduced defect cannot certify the
fix.)*

---

## 7. Controls that must read known values

| control | what it must read |
|---|---|
| **`S1-RANDOM`** | the no-information value of the selection statistic. ⛔ Quoted on the **same windows** as actual and oracle, or none is quoted |
| **`S1-GATE-CONST`** | **exactly zero** recovery |
| **`S1-REPL`** | the run-to-run floor; an effect smaller than it is not an effect |
| **the fan's own collision-free share** | ≈ **55.3 %** on the affected windows, as MEASURED — if a run reports a wildly different share, the fan or the collision checker changed and no arm is readable |

---

## 8. Dependencies and the gate on starting

* ⛔ **`S1-GATE-PRED` cannot start before a predicted occupancy exists.** That is what the
  BEV map head and the 3-D box head are for; both are now wired (R2/R3) and the chain
  trains end to end at 416 × 1024, but **no arm has yet shown the predicted occupancy is
  good enough to gate on**. The first work item is therefore a **read of the predicted
  occupancy's quality against the SAM3 map GT** — not a gate arm.
* `S1-GATE-ORACLE` and `S1-RANDOM` need **no** trained perception and can run first: they
  bound the problem and make everything after them readable.

---

## 9. Tier and reporting

**T1 primary.** `S1-GATE-ORACLE` is **T0 and stamped as such.** Every number carries its
evidence class, tier, `n` and estimator (`paired_episode_cluster_bootstrap`). All four
metric families — longitudinal, lateral, tactical, strategic — **in addition** to ADE,
per family, never pooled.

---

## 10. What would make me abandon this direction

Committed in advance: if `S1-GATE-ORACLE` reproduces the **+0.0485** ceiling but
`S1-GATE-PRED` recovers **less than 10 %** of it on both seeds, then the headroom is real
and the **predicted occupancy is not good enough to reach it**. That is reported as a
refutation of `H-SEL-GATE-1` *as configured*, and the next work is **perception quality**
— the occupancy head's own accuracy against SAM3 map GT — not a better gate.


---

<!-- S1-AMENDMENT-DEVBOX-2026-09-19 -->
## ⛔ S1 AMENDMENT, 2026-09-19 — the dev-box inference-only arms on refcv6's OWN fan, pre-registered BEFORE ANY S1 DATA

**Author:** TanitAD_TrainingFlyWheel. **Decided with the Master Mind 2026-09-19** (fan source,
checkpoint, box-read dependency and ORACLE-CV all accepted; tip `69bbd52`). **Status: 0 S1 data,
0 S1 GPU seconds.** This governs `PREREG_REFCV6_DEVBOX_PREPARATION.md` rows **A4–A6** and adds two
pre-registered reads (S1A.4, S1A.5). Everything in §1–§10 above still binds unless a line below
says otherwise. ⚠️ **Name collision, stated once:** this file's **S1** is the collision-gate
programme. `verdict_refcv6.py`'s **S1** is the STRATEGIC clause. They are unrelated.

### S1A.1 ⛔ DEVIATION — the fan (every "known value" in §1 and §7 was measured on ANOTHER fan)

* **Fan:** refcv6's own emitted fan, the **117** candidates of `anchor_traj` over the 8-slot horizon,
  from checkpoint **A8** (`a8-occupancy-5k-20260919/run/ckpt.pt`: 5,000 steps × batch 2 = **0.95
  of one halfA epoch**) at **416 × 1024**, eval mode, with the **fed** nav command (PI item 23: fed
  nav leads; nav-zero is the ablation and is NOT run here).
* **Windows:** the **1,000 fixed halfB windows** (`torch.Generator().manual_seed(12345)` in
  `refc_v3_train.py`). These are the same windows as A3's held-out occupancy read and A7's eval,
  and they are held out from A8's training. Windows the proxy cannot score are dropped **with
  reasons and `n` stated**, by the item-19 `_select` rules (`ddv2_rl_refcv5.py:190-215`): no full
  route future, or a missing agent-label frame on `t0 … t0+49`.
* ⛔ **§1's 28/493, 55.3 % and +0.0485 were MEASURED on `refcv5-v2`'s DDv2 fan over 493 OTHER
  windows.** On this fan, the collision-free share, the collided-selection rate and the oracle
  ceiling are **NEW MEASUREMENTS, never checks**. §7's *"≈ 55.3 %"* row does **not** apply to this
  fan. It applies only to the optional **external-validity replicate** on refcv5-v2's fan, which
  is deferred.

### S1A.2 ONE forward pass, every arm post-hoc, ONE checker

* Per window, **one** eval-mode forward of A8 yields three things: the fan, the model's own
  per-candidate ranking score (`sel_score_v3` on the goal-point path, else the core's `sel_score`)
  with the model's own `reach_keep` mask when present, and the box-head slots. **Every arm is a
  different selection rule over that SAME fan**, so each moves one variable and the pairing is
  exact.
* **The checker is `tanitad.rl.pdm_proxy.score_candidates` / `no_at_fault_collision`, IMPORTED,
  never re-implemented.** The item-19 per-window inputs are built exactly as
  `ddv2_rl_refcv5.py::fetch` builds them: human states and route from the recorded poses,
  `AgentTracks.from_frames` over the agent join, and the SAM3 drivable map keyed on the raw frame.
  ⛔ `taniteval/tools/fan_safety.py` uses a DIFFERENT collision model (lead only, 2 m) and is not
  used.

| arm | selection rule | tracks the GATE sees | tier |
|---|---|---|---|
| `S1-BASE` | the model's own pick: argmax of its ranking score, its own mask applied | — | T1* |
| `S1-RANDOM` | the **exact uniform expectation** over the fan (the candidate mean, no RNG) | — | T1* |
| `S1-GATE-ORACLE` | BASE's rule, with candidates colliding under the gate's tracks masked to −inf; BASE's pick if every candidate collides | **recorded** future tracks | **T0** |
| `S1-GATE-CONST` | the same gate | **empty** (everything free) | T1* |
| `S1-GATE-PRED` | the same gate | **predicted** tracks (S1A.4) | T1* |
| `S1-ORACLE-CV` | the same gate — ⚠️ diagnostic only (S1A.5) | recorded **t0** boxes + t0 velocities, constant-velocity extrapolated | **T0** |

* ⛔ **Every selected candidate is SCORED against the RECORDED future** (recorded tracks, human,
  route, map), whichever tracks gated it. A gate scored against its own tracks would certify
  anything.
* **Waypoints → proxy states.** A candidate's 8 ego-frame slots are expanded to the proxy's 10 Hz
  grid (40 ticks) by a **C² cubic spline in time** through the origin at t = 0 and the slots, with
  the initial velocity fixed to `(v0, 0)`. Yaw comes from the tangent and speed from the
  derivative. The result goes through `pdm_proxy.ego_states_from_poses`, **the constructor the
  human is scored with**, so candidate and human share one construction.

### S1A.3 The inference replicate is a CONTROL that must read EXACTLY 0 (MM point 1(b))

A8's argv carries **no sampler flag**, so the fan is deterministic. A second full forward on the
same windows must reproduce **every arm's selected index and every recorded-nc flag EXACTLY**
(difference 0). Raw fan coordinates are reported as `max |Δ|`: on CUDA, cuDNN may move them at
float precision, but it must not flip one selection. ⛔ The effect is therefore read against the
**episode bootstrap only**. The verdict says so, because a training replicate does not apply (no
retraining) and the inference floor is structurally 0.

### S1A.4 ⛔ PRED's REAL first dependency: a HELD-OUT BOX READ (corrects the "DISCHARGED" claim)

The collided-selection statistic is **agent NC**, so PRED gates on the **box head**. A3's map IoU
(0.576 vs a 0.339 floor) discharged **only the map/DAC half** of §8. Pre-registered on the same
windows, from A8's box head (same pass):

* **Detection:** AP at the box head's own distance threshold against `random_ap_base_rate`
  (closed form, `box3d_head.py:445`), with an episode-cluster bootstrap CI through a callable
  reducer. **PASS iff the CI lower bound > the base rate.**
* **Velocity:** on the AP's own matched pairs, the MAE of the predicted ego-frame relative
  velocity (`v_rel_x`, `v_rel_y`, `agent_slots.py:172`) against the recorded one, compared with
  the **zero-velocity floor** on the same pairs. **PASS iff the paired CI of (floor MAE − pred
  MAE) excludes 0 in PRED's favour.**
* **Predicted tracks:** slots with `sigmoid(presence) > 0.5` (fixed now), decoded to metres by the
  head's own decode, and extrapolated at **constant velocity in the t0 ego frame** as
  `v = v_rel + (v0, 0)`. v0 is admissible (PI ruling 2026-09-02). Yaw is held constant, because
  the ego yaw rate is not an admissible input. Length and width come from the head. The tracks
  are built in the t0 ego frame with `AgentTracks.from_frames`' **own conventions**: `static` from
  the predicted **class** (argmax over `AGENT_CLASSES` ∈ `cfg.static_classes`, as
  `pdm_proxy.py:246` sets it from the recorded class), and `speed` by the **same finite
  difference** of positions (`:247-252`). *(Corrected while writing: an earlier draft of this line
  said "static = speed < threshold". That is not how the checker defines it, and it was caught by
  reading `from_frames` before this was appended.)*
* ⛔ **PRED is computed in the same pass regardless (zero marginal cost) but is QUOTABLE ONLY IF
  BOTH reads PASS.** Otherwise it is reported as *"dependency failed — the next lever is box-head
  quality"* (§10's own commitment), with its number stamped **NOT A RESULT**.

### S1A.5 `S1-ORACLE-CV` — the pre-registered decomposition (MM point 3)

Recorded **t0** boxes with recorded t0 velocities (finite difference of the recorded tracks at t0)
are extrapolated **by the same code path as PRED**. `ORACLE − ORACLE-CV` = the cost of having no
motion forecast. `ORACLE-CV − PRED` = the cost of detection and velocity error. It is a diagnostic
only, stamped T0 and never a claim.

### S1A.6 Controls that must read known values, and the mutations that prove them

| control | known value |
|---|---|
| BASE vs the model | the harness's BASE index == `out["sel_idx"]` on **every** window |
| `S1-GATE-CONST` | recovery **exactly 0**: empty tracks ⇒ nc ≡ 1 ⇒ BASE's pick on every window |
| `S1-RANDOM` | exactly the candidate mean (an identity, tested) |
| inference replicate | selections and nc flags identical (S1A.3) |
| **human round-trip** | the human's recorded future, cut to the 8 slots and re-expanded by the S1A.2 spline, must reproduce the directly-scored human's NC and DAC on **≥ 99 %** of windows. The residual is the waypoint representation's own error floor and is printed |

⛔ **Mutations — each must turn its test RED:**
* **(M-a)** delete the gate's mask ⇒ ORACLE recovers nothing. This also proves CONST's 0 is **not
  a dead gate**: under recorded tracks the gate must change the pick on ≥ 1 window where BASE
  collides and a free candidate exists.
* **(M-b)** score the selected candidate against the **gate's** tracks instead of the recorded
  ones.
* **(M-c)** replace the imported checker with a stub that returns nc ≡ 1.
* **(M-d)** break the spline's `(v0, 0)` boundary; the round-trip control must catch it.

### S1A.7 Verdict — PREREG_S1 §5.1 at dev-box scale, committed now

* **Statistic:** the **collided-selection rate**, the share of windows whose SELECTED candidate has
  recorded nc = 0. **Estimator:** `paired_episode_cluster_bootstrap` (`taniteval.ci`), PRED − BASE,
  with `n` windows and `n` episodes printed.
* ⭐ **SUPPORTED (dev-box scale)** iff PRED's rate is below BASE's with a paired CI excluding zero,
  **and** both S1A.4 reads PASSED, **and** no family (S1A.8) is separated-worse. **FAIL-HARM** if
  any family is separated-worse. **REFUTED as configured** if the rate does not separate.
  **Recovery < 10 % of the ceiling** triggers §10's abandonment reading: the next work is
  perception quality.
* **Quotable:** `(pred − base) / (oracle − base)`, the share of the ceiling reached, with RANDOM,
  BASE and ORACLE quoted **on the same windows** or none of them.
* ⛔ **The verdict text MUST say (MM point 1(c)):** *internal validity holds (gate vs base on the
  same fan); EXTERNAL validity to a trained planner does NOT: A8's selector is weak (A3 measured
  `anchor_acc` 0.092 vs chance 1/117 = 0.0085).*

### S1A.8 Families — per family, never pooled, ADE added (not replaced)

The families are computed on the SELECTED candidate against the human's recorded future, over the
proxy horizon.
* **LONGITUDINAL:** along-track error at 2 s and 4 s, speed error at 4 s, and the TTC sub-score.
* **LATERAL:** cross-track error at 2 s and 4 s, heading error at 4 s, and curvature error masked
  to windows with `|κ_human| > 1e-3` (the straight-line floor is printed beside it).
* **TACTICAL:** agreement of the selected candidate's kinematic manoeuvre class with the human's,
  by the **same** `tac.window_factored_labels` rule on 2 s, with lat and lon reported separately.
* **STRATEGIC:** nav compliance of the selection (its lateral class at 4 s against the fed turn
  command), **EVALUATED and reported as PAUSED per PI item 25**. It is never counted as a pass and
  never blocks.
* PDMS sub-scores (NC, DAC, EP, TTC, comfort) and ADE/FDE are reported beside the families.

### S1A.9 Compute

* **Passes:** one A8 forward over ≤ 1,000 windows, which also yields the box read, plus the S1A.3
  replicate pass.
* **Device:** CPU while A7 holds the GPU (Master Mind), with host RAM watched and `s/window`
  recorded; or GPU once A7's panel releases it. **The device changes no criterion** and is stated
  in the result.
* **Build proof:** analytic unit tests (two boxes that overlap at tick k ⇒ nc = 0 at exactly k; a
  straight constant-velocity plan ⇒ constant speed and zero yaw), the S1A.6 mutations, and a CPU
  smoke on ≤ 10 halfB windows before the full pass.

<!-- /S1-AMENDMENT-DEVBOX-2026-09-19 -->

<!-- S1-AMENDMENT-ERRATUM-1-2026-09-19 -->
### ⛔ S1A ERRATUM-1 (same day, still 0 S1 data) — the fan has 128 candidates, not 117

**Wrong in S1A.1, S1A.7 and the Master Mind's decision text:** *"the 117 candidates"* and *"chance
1/117 = 0.0085"*. Both were carried over from §4's *"`n_anchors` 117"* without being re-read from the
checkpoint's own record.

**True (MEASURED from `a8-occupancy-5k-20260919/run/config.json`, and identically from A3's):**
`anchors = {shape [128, 8, 2], source "refc.default_anchors (SYNTHETIC)", path null,
v0_conditioned false, sha256 42ec4ce281ee…}`. Neither run passes `--n-anchors`.
* The S1 fan is **N = 128**, a **synthetic, speed-independent** bank plus learned offsets.
* Chance for `anchor_acc` is **1/128 = 0.0078**, so A3's **0.092** is **11.8× chance**, not 10.8×.
* ⚠️ §4's *"held constant: … anchors, `n_anchors` 117"* does **not** describe the dev-box refcv6 runs.
  Which bank the pod arms will use is `PREREG_REFCV6_V2`'s question and is not settled here.
* ⭐ **Consequence for the harness, pre-registered now:** N is **read from the checkpoint**
  (`model.n_anchors` / `config.json:anchors.shape[0]`), never written as a literal. RANDOM's
  expectation is over **all N** candidates. The verdict's validity sentence reads
  *"`anchor_acc` 0.092 vs chance 1/128 = 0.0078"*. *(The same defect class as
  `D-EVALTOOL-ANCHOR-CHANCE`, 2026-09-06, which hard-coded 1/128 where the bank held 117. Here it is
  the other way round.)*
* A speed-independent bank also narrows **external** validity further: at high v0 the synthetic
  fan may cover the human's path worse than a v0-conditioned bank would. The fan's `min-ADE` to
  the human is reported per speed tercile so this is visible, not assumed.

<!-- /S1-AMENDMENT-ERRATUM-1-2026-09-19 -->

<!-- S1-AMENDMENT-ERRATUM-2-2026-09-19 -->
### ⛔ S1A ERRATUM-2 (same day, still 0 S1 data) — "collided" is `nc != 1`, not `nc = 0`

**Wrong in S1A.7:** *"the share of windows whose SELECTED candidate has recorded nc = 0"*.
**True (MEASURED from source):** `pdm_proxy.no_at_fault_collision` returns NC ∈ **{0, 0.5, 1}**
(`pdm_proxy.py:279-307`). An at-fault collision with a **static-class** track scores **0.5**
(`static_classes = ("protruding_object",)`); any other at-fault collision scores 0. Item 19 counted
**`nc != 1`** (`fan_nc_fail_frac`, `ddv2_rl_refcv5.py:679`), and the gate treats a candidate as free
only if **`nc == 1`**. `nc = 0` would silently drop every static-object collision from the
statistic while the gate still avoided them. ⇒ **The collided-selection rate is the share of
windows whose selected candidate has recorded `nc != 1`**, which is the same definition as item 19,
the gate and `s1_gate.arm_metrics`. The `nc = 0.5` share is reported beside it, so the split stays
visible.

<!-- /S1-AMENDMENT-ERRATUM-2-2026-09-19 -->
