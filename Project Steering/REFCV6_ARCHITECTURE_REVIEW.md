# refcv6 — architecture review, and what is still missing from the DiffusionDrive papers

**Master Mind, 2026-09-10.** Evidence class on every claim. ⛔ Nothing here is a capability
result; refcv6 does not exist yet. This is a design grounded in what refcv5-v2 MEASURED.

---

## 1. What refcv5-v2 actually told us — and it is not what the headline said

**MEASURED**, T1, paired episode-cluster bootstrap, 4,823 windows / 141 episodes, confirmed at two
inference seeds (the bar statistic moved **0.0001** against a measured **0.00073** floor):

| | refcv5-v2 | refcv4b | |
|---|---|---|---|
| `os − ha0_ext` (**the bar**) | **+0.0205** [+0.0043, +0.0390] | +0.0101 | ⛔ **FAIL**, separated the wrong way |
| speed MAE | 0.2919 | **0.2540** | ⛔ lost |
| along-track | 0.2655 | **0.2348** | ⛔ lost |
| heading / yaw-rate / cross-track | **1.2121 / 1.0534 / 0.0994** | 1.5489 / 1.4542 / 0.1226 | ✅ won, all three |
| curvature MAE (masked) | **0.003485** | 0.004030 | ✅ won; **0.512× the straight-line floor** |
| tactical lateral κ | **0.8193** | 0.7374 | ✅ won |
| **STRATEGIC route acc** | **0.7708** [0.7146, 0.8254], κ 0.4614, n 3,622 | ⛔ **n = 0** | — |

⇒ **refcv5-v2 bought lateral precision and a working strategic level, and paid for it
longitudinally.** ADE is a weighted sum of both, so the FAIL is real — and it is still the only arm
in the programme with a strategic output at all.

⛔ **Two things in that arm were dead weight**, both MEASURED:

* `tac_goal_tok_head` — **11,286 params, `grad_abs_sum` exactly 0 for all 40,284 steps.** Built,
  rollable, never trained. Wired 2026-09-09. ⛔ No refcv5-v2 result may be credited to the 22-token
  tactical vocabulary.
* `--agents off` for the entire run ⇒ the DiffusionDrive coupling the paper calls *vital* was
  **absent, not tested**.

⭐ **The 771-parameter observation.** `str_goal_head` is **771 params** and produces route accuracy
**0.7708** against a 0.3333 chance, while `core` is **106,067,312** — 98 % of the model. The
strategic level is nearly free. refcv6 should not spend capacity re-earning it.

---

## 2. What is still missing from the DiffusionDrive papers

Read from the **banked source** (`PUBLISHED-CODE`), not from summaries.

### 2.1 ⛔ Missing — and it is the paper's own central mechanism

**The learned agent detector head.** `D-DDV1-AGENT-TOKENS-ARE-A-LEARNED-HEAD`: DD's agent tokens
come from the **model's own auxiliary detector** — 30 queries through a 3-layer transformer decoder
over BEV + status, supervised by a **Hungarian-matched 3D box + class loss** — never an external
detector, never a tracker, **never GT at inference**.

⇒ **`--agents head` is the faithful rung; `--agents oracle` is a diagnostic**, and a result from it
prices the *address*, not the system.

⚠️ Our `head` rung **failed its two-seed tiny-rig gate**, and the deranged-join arm reproduced the
entire degradation ⇒ the cost was **the auxiliary detection task competing for a 17 M trunk**, not
the agent information. ⛔ Whether that survives at 108 M is **untested — a hypothesis**.

### 2.2 ⛔ Missing — and unblocked for the first time on 2026-09-10

**Waypoint-indexed cross-attention (coupling 1).** Implemented (`refc_wp_index.py`, +2,208 params),
tested, mutation-proofed, **never run**: it refuses `--agents off` by design, and until yesterday
the oracle path had **no supplier at all** — `agent_gt` occurred **zero** times in the trainer.
That is now fixed and pinned by an 11-test regression arm. ⇒ **runnable the moment agents are on.**

### 2.3 ⛔ Missing — blocked on exactly one PI decision

**The whole V2 RL stage.** The pilot **cannot load its own cold start**: 487 keys against 488 built
(`decoder.anchor_controls`). Queue item 8. ⛔ Not to be worked around with `strict=False`, which the
code itself calls *"a plausible-looking WRONG experiment"*.

⭐ **This is the piece aimed at our actual weakness.** `H-DDA-5`: V2's gain profile is **EP +5.3,
DAC +1.7, NC / TTC / comfort flat** — a **longitudinal-scale** effect. And **92.2 % of our own
`os − ha` gap is along-track** (`D-REFCV3-AXIS1`). The mechanism and the deficit are the same axis.

### 2.4 ✅ Present, contrary to the analysis

§3.2's "usable now" table was **stale in both directions**: the ≥GT positive mask, intra-anchor
grouping and two-scalar exploration were **already in `rl/`**; the selector library was correct but
had **no caller**; foreign-vocabulary augmentation **did not exist** and now does.

⛔ The foreign bank is **train-only** in the source — a foreign candidate is by construction not
emittable, so an inference pick on one would report **the bank's** quality as REF-C's.

### 2.5 ⛔ Cannot be implemented — recorded so it stops being re-asked

| piece | why not |
|---|---|
| Per-step chain REINFORCE (γ 0.8, η 1) | no `log π(τ_prev given τ_t)` in our decoder — an architecture change, not a port |
| Map-based selector head, DAC / lane-keeping | ⛔ **PhysicalAI contains no map**, pinned by `test_physicalai_feature_readset.py`; the card says *"we do not include open maps data"* |
| PDM simulator reward | no simulator; the analogue is a rule-based T0 reward over a kinematic rollout |
| 800-candidate inference fan | meaningless without a stochastic generator |

### 2.6 ⚠️ And one place we are BEYOND the paper — which is not automatically good

`D-DDV1-NO-DENOISING-LOSS`: **DD-v1 has no ε-prediction and no denoising MSE at all.**
`diff_loss_weight = 20.0` is **dead code**; its only trajectory supervision is **matched-anchor L1
plus focal scoring**.

⇒ our `--w-u0 0.5` is an **invention, not a port** — and refcv5-v2, the arm carrying it, is the one
that lost longitudinally. ⛔ That is a correlation, not a cause. **refcv6 must test it rather than
inherit it.**

---

## 3. refcv6 — the design

**Thesis:** *keep refcv5's lateral and strategic gains, stop paying for them longitudinally, and
finally run the coupling the paper calls vital.*

### 3.1 Inherited unchanged (proven; do not re-litigate)

P14 emitted-fan ranking · the 22-token tactical vocabulary **now actually trained** ·
v0-conditioned anchors (n = 117, `alat`) · P12 `a_star` fix · P5 refinement ·
`--sel-refined --sel-score-emitted` · ego dropout 0.5 · `--goal-str`
(**771 params, 0.7708 route accuracy — the cheapest thing in the model**).

### 3.2 The four changes, each a single lever with its own arm

| # | change | why, from measurement | cost |
|---|---|---|---|
| **A** | **`--agents head` ON** — DD's faithful learned detector | the paper's central mechanism; absent in v5 | re-asks the gate that failed at 17 M, now at 108 M |
| **B** | **`--wp-index on`** — coupling (1) into the sparse agent tokens | +2,208 params; unblocked 2026-09-10 | ⛔ requires A |
| **C** | **`--w-tac-goal > 0`** — the head that received zero gradient | 11,286 params trained instead of dead | ⚠️ needs the `MANEUVER_WEIGHT` budget — **PI** |
| **D** | ⭐ **`--w-u0 = 0` arm** — remove our non-DD denoising term | DD has no such term; v5 lost longitudinally | free, one flag |

⛔ **Not all four at once.** refcv5-v2 moved several levers and its result is barely attributable;
the `--v2` conflation failure — ten levers on two axes, non-attributable — is the precedent.
**D is one flag and free: run it first.**

### 3.3 The order, and why

1. **D** (`--w-u0 0`) — cheapest possible test of whether our own invention is the longitudinal cost.
2. **A** — the paper's mechanism, and a gate that failed at 17 M deserves re-asking at 6× the trunk.
3. **B** — only meaningful once A is on.
4. **C** — as soon as the weight budget is settled.
5. **The V2 RL stage** — the moment item 8 is unblocked. It targets our worst axis.

### 3.4 The bar, committed in advance

Same as refcv5-v2, for the same reason: **beat `ha0_ext` AND `ha` on ADE, separated, plus a 0.10
relative margin.** ⛔ Clearing the CI while missing the margin is a **FAIL as written**.

⚠️ **Plus a lateral non-regression clause:** refcv6 may not give back v5's heading, yaw-rate,
cross-track or curvature wins in order to buy ADE. That trade is exactly what we are trying to
stop paying.

⛔ **Every arm carries a replicate from the start.** MEASURED on WP-D: an arm with **zero levers
moved** read "separably worse" on **5 of 9** family metrics and reproduced a headline ADE effect at
**+0.02460** against the lever's **+0.02610**. A one-seed panel cannot adjudicate this rig.

---

## 4. What refcv6 needs from the PI

1. **Item 8** — the RL pilot's 487-vs-488 cold start. Blocks the entire V2 RL stage, the piece
   aimed at our measured weakness.
2. **Item 10** — the `MANEUVER_WEIGHT` budget for change **C**. ⭐ Or simply: may I **measure** the
   weight on Thor rather than guess it?
3. **WP-C's rung** — approved as the oracle gate, but its arm died with the pod. ⚠️ Note §2.1: the
   oracle rung prices the **address**; `--agents head` is what the paper actually does.
4. **Compute** — a fresh pod for the 40 k arms. Nothing else here is blocked on hardware.
