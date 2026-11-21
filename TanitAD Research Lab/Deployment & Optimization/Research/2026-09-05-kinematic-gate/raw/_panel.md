## 1. ⛔ THE GUARDS — every one printed, whatever it said

`raw/kingate_base.json`. Two **episode-disjoint** draws of 400 EVAL windows / 70 episodes each,
one forward per window, paired episode-cluster bootstrap (cluster = clip), n_boot 4,000, seed 11.

| guard | result |
|---|---|
| **G-CKPT** | **PASS** — `ckpt_step40284_frozen.pt`, md5 `b1ed7075ff73…` (32 hex), `step == 40284` |
| **G-DET** (V3) | **PASS** — the same batch through the same model twice in one process is **bitwise identical** on `anchor_traj` (max&#124;d&#124; **0.000e+00**), `sel_score` and `sel_idx`. ⇒ the decode is **deterministic at eval on this config**, so an inference-seed replicate is a **structural identity, not an estimate of zero** |
| ⭐ **G-OFF** (the deliberate-regression control) | **PASS on both draws** — with the gate disabled (`k = 1`) the selected **INDEX** is identical to the model's on **400/400** (A) and **400/400** (B), and **every** metric's max&#124;diff&#124; is **exactly 0.0** |
| ⛔ **G-OFF, as-published ranking** | **FAIL — 35/400 (8.75 %) on draw A and 29/400 (7.25 %) on draw B.** This is the confound of §0.1, replicated on a disjoint draw |
| **G-PAIR** | **PASS** — every rule indexes one banked tensor stack |
| **G-REP** | **PASS** — A 400 w / 70 ep, B 400 w / 70 ep, **episode overlap 0** |
| **G-CTRL** | **PASS** — `oracle` moves `ade_m` −0.2423 (A) / −0.2851 (B), separated. The panel can see a difference |

⭐ **The SPEC predicted this confound in writing, before the arm ran** (SPEC §4): *"If those two
differ on any window, the gate's candidate ordering is not the model's own selection, and part of
any measured gain would come from re-deriving the ranking rather than from the kinematic
tiebreak. That is a confound, and `G-OFF` is the instrument that finds it."* It found it.

## 2. P2 — "NO SCENE INPUT", VERIFIED AT SOURCE AND SHARPENED

`raw/no_scene_input.json` — an **AST call-graph walk** and a **runtime permutation test**, with
three positive controls.

| probe | reading |
|---|---|
| AST | `feasibility` + `comfort` reach only `_motion_gate` and `kinematics`, and read **six** ctx keys: `dt`, `a_max`, `kappa_max`, `jerk_max`, `lat_acc_max`, `min_motion_m` — **every one a programme CONSTANT**. No `v0`, no `lead_path`, no `obstacles`, no `gt_traj`. **No dynamic ctx access** |
| permutation | under a full scene garble the gate score **and its argmax** are **bitwise unchanged**, max&#124;diff&#124; **0.000e+00** |
| ⭐ positive controls | the same garble moves `headway` **7.5e-01**, `collision` **1.0**, `progress` **3.06** — so the garble took, and the invariance is a property of the components, not of the test |

⇒ ✅ **`D-REFC-KINGATE-1`'s claim — *"no lead, no obstacle track, no ego state"* — is CONFIRMED
EXACTLY for the ranking function.**

⚠️ **And it must be stated one notch more precisely.** The gate's **candidate set** is
`out["sel_score_v3"]` masked by `out["reach_keep"]` — both **scene-conditioned model outputs**.
⇒ The admissible claim is **"adds no NEW scene input and no new perception"**: every input the
gate uses, the deployed model already computes. It is admissible under the
vision-only-at-inference rule for that reason, **not because it is blind**.

## 3. ⭐ THE PRIMARY ENDPOINT — and the SPEC's committed SUCCESS text is MET

Deltas are **gate2 − model**; negative = the gate is safer, positive `ade_m` = the gate is worse.

| rule | `ade_m` | Δ | sep | `sel_envelope` | Δ | sep | `sel_peak_g` | Δ | agree |
|---|---|---|---|---|---|---|---|---|---|
| **DRAW A** (400 w / 70 ep) | | | | | | | | | |
| model | 0.4486 | — | — | 0.1075 | — | — | 0.1718 | — | 1.000 |
| ⛔ gate1 *(control)* | 0.4486 | **+0.0000** | no | 0.1075 | **+0.0000** | no | 0.1718 | **+0.0000** | **1.000** |
| ⭐ **gate2** | 0.4550 | **+0.0064** | **no** | **0.0775** | **−0.0300** | **yes** | **0.1340** | **−0.0378** | 0.510 |
| gate4 | 0.4485 | −0.0001 | no | 0.0650 | −0.0425 | yes | 0.1150 | −0.0568 | 0.300 |
| gate8 | 0.4875 | +0.0388 | no | 0.0675 | −0.0400 | yes | 0.1084 | −0.0634 | 0.210 |
| gate128 / `kin_only` | 0.5020 | +0.0534 | **yes** | 0.0675 | −0.0400 | yes | 0.1061 | −0.0657 | 0.185 |
| `oracle` | 0.2063 | −0.2423 | yes | 0.1300 | **+0.0225** | no | 0.1851 | **+0.0133** | 0.215 |
| **DRAW B** (400 w / 70 ep, episode-disjoint) | | | | | | | | | |
| model | 0.5211 | — | — | 0.1100 | — | — | 0.1809 | — | 1.000 |
| ⛔ gate1 *(control)* | 0.5211 | **+0.0000** | no | 0.1100 | **+0.0000** | no | 0.1809 | **+0.0000** | **1.000** |
| ⭐ **gate2** | 0.5112 | **−0.0099** | **no** | **0.0900** | **−0.0200** | **yes** | **0.1480** | **−0.0330** | 0.520 |
| gate4 | 0.5543 | +0.0332 | no | 0.0675 | −0.0425 | yes | 0.1234 | −0.0576 | 0.273 |
| gate128 / `kin_only` | 0.5969 | +0.0759 | yes | 0.0625 | −0.0475 | yes | 0.1125 | −0.0684 | 0.145 |
| `oracle` | 0.2360 | −0.2851 | yes | 0.1625 | **+0.0525** | yes | 0.2125 | **+0.0316** | 0.210 |

### The three-part quotability test (SPEC §5) — `gate2`

| metric | Δ draw A | Δ draw B | replicate floor | sign | verdict |
|---|---|---|---|---|---|
| ⭐ `sel_peak_g` | **−0.0378** | **−0.0330** | 0.0048 | same | **QUOTABLE** |
| ⭐ `sel_envelope` | **−0.0300** | **−0.0200** | 0.0100 | same | **QUOTABLE** |
| ⭐ `sel_infeasible` | **−0.0300** | **−0.0200** | 0.0100 | same | **QUOTABLE** |
| ⭐ `sel_flagged` | **−0.0300** | **−0.0200** | 0.0100 | same | **QUOTABLE** |
| `ade_m` | +0.0064 | −0.0099 | 0.0163 | **FLIP** | WITHIN-NOISE, **neither separated** |
| `kamm_over` | −0.0100 | −0.0050 | 0.0050 | same | WITHIN-NOISE |
| `off_reach` / `contact` | +0.0000 / +0.0025 | +0.0000 / +0.0000 | — | — | **UNDETECTABLE-DOWNWARD** |
| `ttc_below` | +0.0000 | +0.0000 | 0.0000 | same | WITHIN-NOISE |

> ### ✅ THE COMMITTED SUCCESS TEXT IS MET, ON ALL FOUR CLAUSES
> 1. `sel_envelope` **negative, separated on both draws, same sign, min &#124;Δ&#124; 0.0200 > floor 0.0100** ✓
> 2. `sel_peak_g` **negative and separated on both, min &#124;Δ&#124; 0.0330 > floor 0.0048** ✓
> 3. `ade_m` **NOT separated on either draw**, and &#124;Δ&#124; ≤ 0.0099 < the committed 0.010 m — it satisfies **both** halves of clause 3, and its sign **flips** between draws ✓
> 4. every guard passes, **`G-OFF` included** ✓

⚠️ **Read the frontier before reading `gate2` as "the setting".** Wider k buys more feasibility
and starts paying separated ADE: `gate8` is separated-worse on ADE in draw B, and
`gate128`/`kin_only` are separated-worse on both. `gate4` is ADE-neutral on draw A (−0.0001) and
+0.0332 on draw B; it is a defensible second choice, and it is **not** validated here because the
SPEC named `gate2` as the arm.

⚠️ **The `oracle` row is the mechanism.** The fan's best-ADE candidate is its **least** drivable
one (`sel_envelope` +0.0225 / **+0.0525 separated**). The gate buys feasibility by discarding
candidates that were never worth their ADE — `D-RL-FANSAFE-1` reproduced from the other side.

## 4. ⛔ THE FOUR FAMILIES, PER FAMILY, NEVER POOLED (T0)

`raw/families.txt`, `raw/kingate_base.json`, `raw/longitudinal_lead.json`. ADE sits **beside**
them in §3 and is never "the result".

### [1] LONGITUDINAL — ⭐ the one separated four-family movement, and it is an improvement

| | model | gate2 | Δ (paired) draw A | Δ draw B |
|---|---|---|---|---|
| `accel_mae` (m/s²) | 0.7007 / 0.7597 | **0.5640 / 0.6511** | ⭐ **−0.1846 [−0.2840, −0.1014] SEP** | ⭐ **−0.1402 [−0.2327, −0.0629] SEP** |
| `speed_mae` (m/s) | 0.3878 / 0.4479 | 0.3729 / 0.4306 | −0.0186 ns | −0.0216 ns |
| `along_mae` (m) | 0.3254 / 0.3832 | 0.3317 / 0.3760 | +0.0063 ns | −0.0072 ns |
| `speed_bias` (m/s) | 0.0418 / 0.0411 | 0.0640 / 0.0472 | — | — |

⭐ **`accel_mae` is QUOTABLE by the SPEC's own three-part test**: separated on both draws, same
sign, replicate floor **0.0444**, min &#124;Δ&#124; **0.1402 > 0.0444**. A **−19.5 % / −14.3 %**
reduction in acceleration error — the `comfort` term's jerk penalty showing up exactly where the
mechanism predicts it should.

**Distance keeping** (`raw/longitudinal_lead.json`, **136 lead windows / 33 episodes**;
no-lead windows masked to NaN because `lead_track` returns a **far sentinel**, not an absence):
`gate1` reads **+0.0000 [0, 0]** on all three metrics (the identity control again), and `gate2`
moves nothing — headway **−0.0008 ns**, time gap **+0.0012 ns**, min-TTC **−0.274 ns**
(human 27.87 m / 4.56 s / 22.46 s; model 27.75 / 4.57 / 21.92).

### [2] LATERAL — ⛔ my committed hypothesis is NOT supported

| | Δ draw A | Δ draw B | verdict |
|---|---|---|---|
| `kappa_mae` (1/m) | −0.0937 ns | **+0.1519** ns | **SIGN FLIP** |
| `yawrate_mae` (rad/s) | −0.0457 ns | **+0.0761** ns | **SIGN FLIP** |
| `latacc_max` (m/s²) | −0.0647 ns | −0.0500 ns | same sign, neither separated |
| `cross_mae` (m) | −0.0000 ns | −0.0038 ns | ns |
| `heading_mae` (deg) | +0.0228 ns | +0.0092 ns | ns |

⛔ **The SPEC committed, before the arm ran, that LATERAL is *"the family the gate should
move"*, and that a `sel_envelope` gain with no lateral movement *"would be evidence the gain is
bookkeeping, not geometry."* Draw A looks like the predicted signature; draw B flips two of the
three. ⇒ Reported as **NOT SUPPORTED**, not quietly dropped.** The honest reading is that the
gain is carried by the **longitudinal/jerk** half of `comfort` rather than the curvature half —
which the LONGITUDINAL table independently supports.

### [3] TACTICAL — the gate's real cost, and it is consistent across both draws

| | model | gate2 | direction |
|---|---|---|---|
| lateral-decision accuracy | 0.9375 / 0.9525 | 0.9150 / 0.9225 | **−2.3 / −3.0 pp** |
| lateral-decision κ | 0.6726 / 0.8112 | 0.5957 / 0.7263 | −0.077 / −0.085 |
| longitudinal-decision accuracy | 0.9750 / 0.9725 | 0.9700 / 0.9500 | −0.5 / −2.3 pp |
| 5-way collapsed accuracy | 0.9225 / 0.9350 | 0.8975 / 0.9075 | −2.5 / −2.8 pp |
| goal point error (m) | 0.9419 / 1.0976 | 0.9473 / 1.0893 | ±0.005 / −0.008 |
| **anchor/goal SELECTION** (`agrees_model`) | 1.000 | **0.510 / 0.520** | the gate changes the pick on **~49 %** of windows |

⚠️ **This family's core row for a selection rule is the selection itself**, and the gate changes
it on about half of all windows — it is not a rounding artifact of rarely intervening.
⚠️ **The manoeuvre-class rows carry NO paired CI**: `four_families.tactical_from_trajectory`
returns an aggregate accuracy, not a per-window series, so these are absolute values with their
`n` and a direction that is **consistent across two episode-disjoint draws**. Treat the ~2.5 pp
manoeuvre-agreement cost as a real but un-intervalled cost, and closing it is a work item.

### [4] STRATEGIC — readable, and its answer is a corpus property, not a defect

`status: UNAVAILABLE`, **`defect: false`**, `n = 400` per draw. The reason is the canonical one:
PhysicalAI-AV carries no map, lane graph, junction label, traffic-light feature or route signal.
⛔ **This is not the `D-NAVCOMP-SHAPE-1` defect** — the block is well-formed, carries its reason
and its `n`, and no `TypeError` from our own module is involved.

⭐ **And for THIS lever the family is a structural identity, asserted rather than assumed:** the
gate re-ranks candidates from **one** forward, so `core.route` is not re-run and `route_logits`
are the **same tensor** for `model` and every gate arm. That is an identity (`H-ECHO-4`), not an
estimate of zero.
⚠️ Reported beside it because it is the trap this programme has already fallen into: route argmax
versus **the `nav_cmd` the model was fed** reads **0.2100 / 0.1850** — so refcv3's route head is
*not* the exact bijection flagship v1's was (1.0000), but the number is an **echo diagnostic**,
never a strategic-skill claim.
