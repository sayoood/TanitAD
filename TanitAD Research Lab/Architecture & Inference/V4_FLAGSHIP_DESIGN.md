# Flagship v4 — the joint world-model + diffusion-planner stack

**Date:** 2026-07-21 · **Status:** **BUILD-READY** — awaiting Sayed's line-by-line gate approval (§9)

> ⭐ **Build-readiness pass, 2026-07-21 (§14–§17).** Every open marker in §0–§13 was swept: **20 items
> → 15 CLOSED, 3 CLOSED-AS-KNOB, 2 DEFERRED** (both deferrals are named, and **neither blocks the
> build**). Added: **§14** the closure register · **§15** the ordered build plan (**P1–P9, ~118 h**) ·
> **§16** the exact CLI surface with the four one-lever reproduction diffs · **§17** the pre-flight
> checklist. **Six defects were found that would have cost real time:** the gate card would not have
> registered *and* would have converted four non-fatal falsifiers into restarts (**O-03**); the
> hierarchy panel decides on the estimator `CLAUDE.md` forbids (**O-07**); the scheduled-sampling
> ramps straddled the gate step (**O-17**); the tactical imagination is 2.5× longer than any measured
> figure (**O-10**); the strategic action set was never enumerated (**O-18**); and *"LAL-v2 is
> unmerged, 12 days idle"* is **false** — it merged on the day of the intake (**O-05**, retracted).
**Charter (Sayed, verbatim, 2026-07-21):** *"I'm interested in combining our world model from v1 and
the diffusion planner from REF-C, **end2end trained and at the same time**, strengthen the prediction
quality and imagination of the WM, improve the planned speed and smoothness of the trajectory
(minimize jerk, etc.), improve the selection of the trajectory candidates based on tactical and
strategic constraints… Let's develop the plan, stop v3enc and retrain v4 flagship."*

**This is a design document, not a training order. Nothing here launches until §9 is approved.
No training was started; no pod was written to while this was written.**

**Number discipline (binding, per `CLAUDE.md`).** Every quantity below is marked **MEASURED** with a
primary source (`Project Steering/MODEL_REGISTRY.md`, a raw eval JSON, an in-repo research note, or a
source-code read) or **PROPOSED** (chosen by me, before launch, and falsifiable). Intervals carry
their estimator. No exponent is quoted bare.

**v4 supersedes and absorbs `V35_DESIGN.md`.** §1 states exactly what survives and what today's
measurements overturned. V35 stays in the repo as provenance; it must not be quoted as current.

**⭐ This revision incorporates the v3enc post-mortem**
(`Research/2026-07-21-flagship-v3enc-postmortem.md`, staged). It changed five things in this design —
they are marked **[PM]** throughout — and it resolved the one dependency that could have re-opened §2.
Its DO-NOT-CARRY list (§8 there) is honoured item by item in §10 and §12.1 here.

---

## 0. Executive summary

**v4 is ONE model carrying THREE end-to-end planners over one shared world model — strategic,
tactical, operative — all predicting, all using imagination, trained jointly in a single run. ①
The strategic planner is a learned planner in a compressed 128-d "strategy-relevant" subspace: it
rolls its own predictor forward under a discrete strategic-action set, evaluates the options, and
picks the most probable one that fits context and goals — including the no-navigation case, where it
estimates the most probable path from vision alone. ② and ③ are two anchored-diffusion instances at
different horizons (5 s coarse and 2 s dense), the second of which ships the trajectory. The v1
world-model trunk keeps its full objective throughout while planner gradient reaches it through a
scheduled, canary-clamped scale — the one structural difference from v1.6, which deleted the
world-model objective, tied v1 on ADE and destroyed the world model (canary 0.452 → 1.1022, +144 %,
MEASURED). Selection gains a longitudinal input for the first time via a factorised LAT × LON × DIST
graft into the anchor logits. The shipped path becomes the operative planner's dense 20-step anchored
plan, which is what finally makes a jerk/smoothness penalty computable on the path we actually ship —
v1's `--jerk-weight` acts on a 4-point head that is not the scored path and contributes ≤1e-4 of a
~4.0 loss, so it and `--aux-accel` are dropped rather than tuned. [PM]**

**Two binding rules come from the v3enc post-mortem and shape the whole design: (i) never zero-fill a
channel whose zero is in-distribution — that single line of code is the measured root cause of
v3enc's failure, and v4 obeys it in three places including one it inherited (§5.3); (ii) never change
more than two encoder-touching levers in one arm — v4 changes exactly one, and the lever count is a
pre-registered gate item (§9). Ask #2 is retargeted accordingly: the objective is multi-step ROLLOUT
fidelity, not the encoder, and it gets its own gate secondary (§7.5).**

**v4 lands at ≈ 247.9 M trainable — 62 % of the 400 M cap, and ~30 M SMALLER than v1. The whole
three-planner hierarchy costs ~25 M, about 9 % of v1**, because the trunk dominates, a diffusion
instance is 9.8 M, the strategic planner thinks in 128 dimensions, and **anchor vocabularies are
buffers with 0 parameters**. Capacity was never the obstacle to hierarchy; nobody had priced it.

**⭐ Two things changed this design after it was first written, and both were the PI's objection
rather than my analysis.**

*First — §7A, from Sayed's "how are we assessing tactical, operative and strategic plan if we are
eliminating tactical prediction?"* A source read corrected me: `tactical_pred`'s horizons are
**frames**, so it ran at **0.8 / 1.6 s**, not "2–8 s". But the deeper point was worse than the
premise: `str_h = (goal_h,) = 20 frames = **2.0 s**` — **the strategic level has never had a timescale
of its own in any arm this program has trained**, which makes `per_window_content_helps ≈ 0` the
*predicted* outcome and the hierarchy decorative **by construction**. Measured this session on
**500 episodes / 95,477 windows**: 5 s is supervisable on **74.3 %**, 10 s on **48.2 %**, 20 s on
**0.0 %**. So v4 reasons at ≤5 s and defers 10–20 s to v5 on a data constraint, not a taste.

*Second — §2/§6.3, from Sayed's rejection of the single-planner shape.* 🔴 **I have RETRACTED the
"the fan is a SPEED fan, so strategic choice is a ~2 % lever" scoping.** The measurement stands
(32.6× long/lat, 0.0 % laterally dominated, 1.4 cm lateral spread) but the inference is
**confounded**: REF-C evaluates with `nav_cmd=None → follow` on all 881 windows and trains on the
circular v1 net-heading command, so **its decoder never had a working route input and could only
learn the marginal — which on this corpus is "straight, at some speed."** v4 is the first arm that
can settle it, and §6.3.1 is the pre-registered experiment that does, with both outcomes committed to
in advance.

### The four asks, answered honestly up front

| # | Sayed's ask | Verdict | Where |
|---|---|---|---|
| 1 | WM + diffusion planner, **end2end at the same time** | ✅ **Achievable**, with one caveat stated plainly: the plan→predictor path is **stop-gradient** by design (§5.3), so "end-to-end" means one model, one run, one optimiser, shared trunk, mutual conditioning — *not* a differentiable loop from the WM loss back into the planner. That loop is degenerate (the planner would learn to propose easy-to-predict plans) and it is refused. | §2, §5 |
| 2 | strengthen WM prediction quality + imagination | ⚠️ **Partly — and the post-mortem told us exactly where to aim. [PM]** The target is **multi-step ROLLOUT fidelity**, not the encoder and not one-step prediction: at matched steps v3enc's encoder metric-grounding was **1.13–1.38×** v1 (nearly intact) while its action-conditioned rollout was **3.89–4.48×** worse, with step-1 operative speed R² still **0.9529** and the imagined latent barely moving (`znorm` ×1.86 vs v1's ×3.68, `zcos` 0.618 vs 0.397, non-monotone rollout speed). MEASURED, post-mortem §2. v4 makes rollout fidelity a **pre-registered gate secondary** (§9) using the post-mortem's own summary statistic. But **"improve beyond v1" is still not something v4 is designed to buy** — v4's WM recipe is v1's, deliberately (N6). | §5, §7.5, §9 |
| 3 | improve planned speed + smoothness, minimise jerk | ✅ **Achievable, and this is where v4 has the most new headroom** — because two source reads (§7.1, §7.2) show the current jerk penalty **does not touch the scored path at all**, and the scored path's recursive horizon is 5× shorter than the horizon it is scored on. **[PM] confirms both are dead ends worth abandoning rather than tuning**: v1's `jerk` term contributed **≤1e-4 of a ~4.0 total loss**, and v1's `aux_accel_r2` never stabilised (0.003 → −0.119 → −0.423 → 0.447, negative twice). v4 drops both and moves smoothness onto the dense emitted plan. | §7 |
| 4 | improve candidate selection on **tactical AND strategic** constraints | ⚠️ **Tactical: yes, highest-confidence change in the design** — 88.9 % of the fan's spread is longitudinal and there is currently **zero** longitudinal signal anywhere in the selection path (`accelerate` predicted **0/881**). **Strategic: 🔴 my earlier "~2 % lever" refusal is RETRACTED as confounded** (§6.3) — REF-C's decoder never had a working route input, so the lateral collapse is at least as consistent with *"nothing conditioned it"*. **UNRESOLVED, and v4 is the first arm that can resolve it.** §6.3.1 is the discriminating test, with both outcomes pre-committed. What stays binding: post-hoc **re-ranking** is closed (0.0 % / +2.9 % n.s., ~92 % aleatoric) — a strategic constraint that *removes modes* is a different operation and that door is open (§6.3.2). | §6 |

### The three highest-risk decisions (full treatment in §11)

1. **Joint training will not repeat v1.6's world-model collapse** — because v4 keeps the WM loss live
   and bounds the planner's pull. *Falsified by:* the canary rising > +0.05 m at any milestone while
   λ_plan is at or below its scheduled value — **or**, in the quieter failure mode [PM] found,
   `speed_benefit_recovered_frac` sitting flat below 0.70 for three consecutive 2 k buckets while the
   canary itself looks fine (v3enc plateaued; it did not blow up).
1b. **The strategic planner earns its place** — ⭐ *the highest-uncertainty decision in the revised
   design, and the one Sayed cares most about.* Three independent falsifiers, any of which demotes it
   to the produced-goal fallback (§2.6) rather than killing the run: the **subspace two-probe test**
   (§7A.3(iii)), the **imagination-horizon scaling test** (§7A.4 — his central hypothesis), and
   **`nonav_route_beats_majority`** (§7A.5). *Falsified by:* `imag_win_at_5s ≤ 0`, or sufficiency
   < 0.90, or compression > 0.50, or no-nav route accuracy not separated from the straight base rate.
2. **The anchored planner path beats the recursive rollout on the tail** — REF-C's decoder brings a
   heavier tail (miss@2m 0.1419 full-set vs v1's 0.0602 heldout, MEASURED). *Falsified by:* miss@2m
   > 0.10 at the 10 k gate on the planner path while the rollout path passes.
3. **Decoder capacity is irrelevant, so d384×4L is enough** — v4 spends 8.56 M on the decoder where
   v1.5 spent 30.98 M. *Falsified by:* oracle-in-fan > 0.30 at 10 k (i.e. worse than v1.5-`ab`'s
   frozen-trunk 0.3073) with healthy seam norms and a flat canary.

---

## 1. What v4 inherits from V35_DESIGN, and what today overturned

`V35_DESIGN.md` (470 lines, 2026-07-20) is the parent. Its budget table, hierarchical wiring spec
(§2A), circularity decisions and gate-ladder shape are carried. Six of its positions are now dead.

| V35 position | v4 | The measurement that decided it |
|---|---|---|
| §7.1 *"decoder width is NOT the lever; spend the extra ~140 M on the **encoder** (d1024×16, ~170–190 M)"* | 🟥 **OVERTURNED.** No encoder widening. Encoder stays v1's 87.1 M. | REF-C-base (encoder **90,458,632**) vs REF-C-XL (encoder **199,496,532**): paired Δ ADE@2s **+0.0013 [−0.0281, +0.0316] NOT separated**, FDE **−0.0030 [−0.0619,+0.0584]**, miss **+0.0000 [−0.0261,+0.0272]**, per-window corr **0.789**, episode-cluster bootstrap B=2000. A **2.2× encoder cut cost 0.001 m.** MEASURED, REGISTRY §4.3 |
| §2.2 Branch 1 — *"if v1.6 closes the oracle gap the v1.5 lineage IS the skeleton"* | 🟥 **DEAD.** v1.6 ran and failed 3 of 5 sub-gates. | v1.6 paired vs v1: **Δ +0.0104 [−0.0888, +0.1147] NOT separated**; canary **0.452 → 1.1022 (+144 %)**; oracle 0.3073 → 0.2815 (only **8.4 %** of the gap). MEASURED, REGISTRY §1.4b |
| §2.3 alt ② — *"REF-C-base's 90.5 M encoder as a second KV source (≈348 M) is the option the raised cap unlocks"* | 🟥 **DROPPED.** A second encoder costs a second forward pass per tick and buys nothing measurable. | Same base-vs-XL null. If 2.2× more encoder buys 0.001 m, a second *different* encoder has no measured case at all. REGISTRY §4.3 verdict (iii) |
| §2A S2 — *"strategic G_s → operative via **FiLM on the predictor's conditioning vector**"* | ⚠️ **DEMOTED to a pre-registered A/B, default OFF.** The strategic signal reaches the operative through the **plan** (S3, the action channel). | Seam track record, corrected (§2.4): *additive-into-a-conditioning-vector* is **0 for 4** (F3 ① HARMFUL cos −0.238 · A3 ② INERT at a 124× norm ratio · V2 ② INERT · V3 ② INERT + monitor fired at 2.80×). The action channel is **1 for 1** and the KV-token seam is **1 for 1**. Source: `ARCHITECTURE_WIRING_COMPARISON.md` §2.4 |
| §7.2 budget *"≈345–365 M recommended"* | 🟥 **OVERTURNED → ≈ 247.9 M.** *(O-02: this row read "≈239 M", the pre-strategic-planner figure. The single authoritative total is §3.1's **247,878,786**.)* | §3, all modules re-measured this session by instantiation |
| §7.5 *"imagination conditioning is the design's dominant latency term; a CEM fan may be unaffordable"* + the 723 ms projection | 🟥 **REFUTED.** | Composed levers take v1's tick **100.29 → 18.75 ms p50 / 18.76 p99 (5.35×, 53.3 Hz)**, CUDA-graph capture **bit-exact**; an **8-candidate** shared-encoder fan is **20.82 ms p50 / 23.72 p99**, marginal candidate ≈ **0.3 ms**. MEASURED, `Production & Optimization/Research/2026-07-20-flagship-v1-inference-levers-measured.md` §5–§6 (raw `taniteval/results/eff_levers_{,fan_sharedenc_}flagship-30k.json`) |

**[PM] adds a seventh supersession** that neither V35 nor the task brief could have known:

| Position | v4 | The measurement |
|---|---|---|
| v1's `--aux-accel` head + `--jerk-weight` as part of "the v1 recipe verbatim" | 🟥 **BOTH DROPPED.** v4's WM recipe is v1's *minus* these two. | `aux_accel_r2` over v1's own run: **0.003 / 0.031 / −0.119 / 0.209 / −0.423 / 0.049 / 0.447 / 0.328** — it never learned a stable signal and went negative twice. v1's `jerk` term ran 3e-4…3.7e-3 at weight 0.02 = **≤1e-4 of a ~4.0 loss.** MEASURED, post-mortem §4.5 / DO-NOT-CARRY #7 |

**Carried forward unchanged from V35:** the hierarchical-wiring requirement itself; S1 as KV tokens;
S3 as inverse-dynamics-through-the-existing-action-channel with a stop-gradient; goal-dropout ≥ 0.5
on every seam; the hard in-graph norm clamp at 1.0× with fail-loud at 1.5×; the plan-free WM canary;
the causality gate (G-H) with `per_window_content_helps` rather than `helps_vs_none`; parity on
`physicalai-train-e438721ae894` / skip-hash `f09e44db`.

> ✅ **RESOLVED (O-06) — the registry was refreshed.** `MODEL_REGISTRY.md` §1.4 now carries v3enc as
> **⏹️ STOPPED at step 10,800**, so the warning below is spent and is kept only as provenance.
> ~~`MODEL_REGISTRY.md` §1.4 still carries v3enc as
> **🟢 STILL RUNNING**.~~ Sayed stopped it today at ~10,800 and pod1 is free. The §1.4 row also already
> carries the **🟥 10 k gate verdict `RESTART`** (primary `ade_0_2s` 1.9654 ≤ 2.5 PASS;
> **`encoder_speed_probe_r2` 0.393 < 0.55 FAIL**; `highspeed_long_overshoot_m` +2.195 ≤ 8.0 PASS;
> restart budget **1/2** on lever family `encoder-grounding`). The status field must be updated to
> STOPPED before anyone quotes this row.

---

## 2. Architecture — three end-to-end planners over one world model

> **This section was rewritten 2026-07-21 after Sayed rejected the single-planner shape.** He is the
> PI and hierarchical reasoning is a declared program pillar (H26). His architecture is three
> end-to-end planners — strategic, tactical, operative — **all predicting, all using the world
> model's imagination.** §2 states that architecture as the target (§2.1), then scopes v4 honestly
> against the three data constraints (§2.2), then specifies each planner (§2.3–2.5). The earlier
> single-planner framing is gone; what survives it (the anchored decoder, the zero-fill rule, the
> canary, the factorised selection, the seam discipline) is carried into the new shape.

### 2.1 The target architecture — fully realised (this is the program's destination, not just v4)

```
        nav command (optional)          9-ch 3-frame stack, 256 px
              │                                   │
              │                        [ ViT d768 × 12 ]  87.1 M   (TRAINED)
              │                                   │
              │                     ( SpatialGridReadout → state 2048 )
              │                                   │
    ┌─────────▼───────────────────────────────────┴──────────────────────────────┐
    │  ① STRATEGIC PLANNER  — a LEARNED planner in a COMPRESSED strategy subspace  │
    │     z_strat = E_strat(state)   (2048 → d_strat, the "strategy-relevant" dims)│
    │     strategic predictor rolls z_strat forward UNDER strategic actions        │
    │        (its OWN imagination, in the small subspace = cheap = many options)   │
    │     evaluate a discrete strategic-action set → pick the most probable that   │
    │        fits semantic context + our goals ("be efficient")                    │
    │     NO-NAV case: estimate the most probable path from vision alone           │
    │     emits G_s = ⟨route/exit choice, target-speed band, ODD envelope⟩         │
    └─────────┬───────────────────────────────────────────────────────────────────┘
              │ G_s  ══> KV tokens (S1)
    ┌─────────▼───────────────────────────────────────────────────────────────────┐
    │  ② TACTICAL PLANNER  — anchored-diffusion instance #1, LONGER horizon        │
    │     "reach the correct lane": evaluate options — wait for a gap, change      │
    │     early enough, indicate — over ~5 s, conditioned on G_s                   │
    │     imagination: operative-predictor consequence rollout at the tac horizon  │
    │     emits G_t = the selected coarse maneuver plan                            │
    └─────────┬───────────────────────────────────────────────────────────────────┘
              │ G_t  ══> KV tokens (S1) + ── S3 action channel (first 2 s, stop-grad) ──┐
    ┌─────────▼───────────────────────────────────────────────────────────────────┐   │
    │  ③ OPERATIVE PLANNER — anchored-diffusion instance #2, 2 s DENSE             │   │
    │     selects the trajectory that executes the maneuver; factorised           │   │
    │     LAT×LON×DIST selection; the SHIPPED path                                 │   │
    │     imagination: operative-predictor consequence rollout at 2 s (the         │   │
    │     PROVEN −0.1355 m mechanism)                                              │   │
    └─────────┬───────────────────────────────────────────────────────────────────┘   │
              │                                                                          │
    actions [steer, accel, v0/10] ─────────────────► [ OPERATIVE PREDICTOR d768×10 ] ◄──┘
                                                              │
                          ( H15 belief field 22.1 M · grounding 13.4 M )
                                                              │
       ── WM canary path: rollout under TRUE actions, ALL planner seams ZEROED
          (exactly v1's intent-free deployed path) ──
```

**The three imaginations, one per planner — this is the spine of Sayed's design:**

| Planner | Its imagination | Horizon |
|---|---|---|
| ① strategic | the **strategic predictor** rolling `z_strat` under strategic actions in the compressed subspace | mid/long (≤5 s buildable) |
| ② tactical | the **operative predictor's** consequence rollout (the v1.5 mechanism) at the tactical horizon | ~5 s |
| ③ operative | the **operative predictor's** consequence rollout at 2 s — the PROVEN −0.1355 m seam | 2 s |

> **Sayed's central hypothesis, made testable (§7A.4):** *imagination is decisive specifically for
> mid/long-term prediction.* The falsifier: rolling the predictor forward must beat a direct head **by
> a margin that GROWS with horizon**. If it helps equally at 0.2 s and 5 s, it is not "decisive for
> mid/long-term" — it is just generically useful, and his specific claim is false. Measured at three
> horizons in one panel.

### 2.2 Scoping v4 against the three DATA constraints — buildable-now vs v5

The target needs three things this corpus does not fully have. **Each is measured, not assumed**, and
each decides a v4/v5 line:

| Constraint | MEASURED | v4 (buildable now) | v5 (deferred, pending data) |
|---|---|---|---|
| **Supervision ceiling** | 5 s = **74.3 %** · 7 s = 63.9 % · 10 s = **48.2 %** · 20 s = **0.0 %** of window starts (§7A.2, 500 eps / 95,477 windows) | strategic + tactical reasoning at **≤5 s**, masked to the 74.3 % | a **10–20 s** strategic horizon — needs longer clips |
| **"Take the next exit of the roundabout"** | route v3 mints `roundabout` on **8/2201 windows, from ONE episode**, 24 u-turn confusions | strategic option set = the **kinematically mintable** ROUTE tokens (turn L/R, u-turn, follow) + target-speed band | map-dependent routing (exits, merges, lane graph) — needs a map |
| **"Wait for the vehicles to pass / indicate"** | `lead_state` is a **`None` stub** — no boxes, no tracks, no depth | tactical option evaluation over the **ego-realisable** maneuver set (a gap-in-traffic is not one of them yet) | agent-interaction tactics — needs detections / tracks / monodepth |

**So v4 is the three-planner *shape* at the supervisable horizon and over the ego-realisable action
set — not the fully realised architecture, and I am saying so plainly.** The parts that need a map or
agent state are v5, gated on data, not architecture. **A staged path to Sayed's architecture is the
recommendation** (§2.6): v4 proves the shape works where it can be supervised; v5 extends the horizon
and the action set when the data exists.

### 2.3 ① The strategic planner and its subspace (§7A is the full spec)

The novel piece, and the one Sayed's efficiency argument rests on. Three sub-parts:

- **E_strat: the compression.** `z_strat = E_strat(state)`, 2048 → d_strat (PROPOSED **128**), a small
  MLP off the readout state. **"Strategy-relevant" is defined operationally** as *sufficient to
  predict the 5 s coarse direction, route and target-speed band, while invariant to the exact 2 s
  waypoint.* It is **verified by a two-probe information-plane test** (§7A.3), not asserted — that is
  the direct answer to Sayed's "verified to carry strategy rather than being a bottleneck that
  discards it."
- **The strategic predictor.** A small causal transformer **in the d128 subspace** (MEASURED 3.92 M),
  rolling `z_strat` forward under a **discrete strategic-action set** — its own imagination. Small
  dim = cheap = **many options evaluable per tick**, which is the efficiency argument.
- **The option evaluator (planning, not a head).** Enumerate the strategic actions, imagine each
  forward, score by `P(action | context)` (a learned route prior, supervised) **×** fit-to-goals
  (a cost), pick the best. This mirrors P2 exactly — a prior seed plus imagined-rollout evaluation —
  which is why it is **not a "head in disguise"** (Sayed's explicit warning; heads score 3.15 vs the
  rollout's 0.45). The no-nav case is this evaluator with the command absent (§7A.5).

### 2.4 ② + ③ Two diffusion instances — the tactical and operative planners

Both are the **in-repo `FlagshipV15Head` / `V15Decoder`** (REF-C's `AnchoredDiffusionDecoder`, KV
swapped for a token set, REF-C's selection flaw already repaired), instantiated twice at different
horizons. Sayed explicitly accepts the tactical/operative overlap *"as done by the diffusion planner."*

| | ② TACTICAL instance | ③ OPERATIVE instance |
|---|---|---|
| horizon | **5 s coarse** (10 knots @ 0.5 s) | **2 s dense** (20 steps @ 0.1 s) |
| anchors | 256 FPS over 5 s GT trajectories | 256 FPS over 2 s GT trajectories |
| geometry | d384 × 4L | d384 × 4L |
| params (MEASURED) | **9,767,320** | **9,778,604** + factorised heads ≤811,543 |
| conditioned by | G_s via KV tokens (S1) | G_s + G_t via KV (S1) and G_t's first 2 s via the action channel (S3) |
| supervised on | the **74.3 %** with 5 s future (masked) | 100 % (2 s = 90.1 %; the 2 s waypoint target is v1's) |
| ships? | no — it is a constraint on ③ | **yes — the scored trajectory** |

**Two instances cost ~19.5 M combined vs one at 9.8 M** — because the decoder is 9.8 M and the anchor
vocabulary is a **buffer (0 params)** (§3.3). Two planners is cheap; that is why the hierarchy is
affordable.

### 2.5 The seams — only the 1-for-1 mechanism families

The program's seam record, re-partitioned by **mechanism** (from `ARCHITECTURE_WIRING_COMPARISON.md`
§2.4, sourced to `HYPOTHESIS_LEDGER.md` + in-repo `flagship-v15-*-ckpt.json`):

| Mechanism | Record | Members |
|---|---|---|
| **Action channel** (a new input dim on the action vector) | **1 for 1** | A4: REF-A ego-through-actions **3.73 → 0.83** isolated; flagship no-speed **2.918** vs speed **0.4522**, paired **+2.21 m [2.04, 2.39]** |
| **KV cross-attention tokens** | **1 for 1** | V1: v1.5 imagination tokens `a`→`ab` **−0.1355 m [0.038, 0.233] SEPARATED** |
| **FiLM on a block condition** | **1 for 2** | F2 ctx→tactical ④ (at 30 k only); F1 nav→strategic ② INERT |
| **Additive into a conditioning vector** | **0 for 4** | F3 ① HARMFUL (cos −0.238) · A3 ② (norm 1792 vs 14.5) · V2 ② · V3 ② (monitor fired 2.80×) |

**Every v4 seam is in a 1-for-1 family:**

| Seam | From → To | Mechanism | Discipline |
|---|---|---|---|
| **S1** | `G_s`/`G_t` → each diffusion instance | **KV tokens** appended to the decoder's heterogeneous KV set | ReZero per token-group init 0.1 · goal-dropout 0.5, DROPPED row ≠ UNKNOWN · **hard in-graph rescale at ratio > 1.0, FAIL LOUD > 1.5** |
| **S3** | tactical plan `G_t` (first 2 s) → operative predictor | **inverse-dynamics → action sequence into the EXISTING 3-dim channel** — a proven port, no new swamping surface, identical to what P2's CEM does at inference | **stop-gradient on G_t** · scheduled sampling ⅓→⅔ · goal-dropout 0.5 · canary with all seams zeroed |
| **S_strat** | `z_strat` → strategic KV | the strategic predictor's readout, projected to a KV token | same norm cap; `--strategic off` reproduces the two-diffusion arm byte-identically |
| ~~S2 (additive-into-operative-conditioning)~~ | — | **NOT BUILT.** 0-for-4 family, and its position holds the only HARMFUL seam. A `--s2-film` flag exists only for a pre-registered A/B. | — |

> **What "end2end at the same time" means here, stated exactly (the caveat Sayed must own).** One
> model, one run, one optimiser; the encoder receives gradient from all three planner stacks and the
> WM stack; the operative predictor is shared and trains under all of them. But **`G_t → operative`
> and `z_strat`'s consumption by the operative are stop-gradient**: a planner's own quality gradient
> does not flow back through the plan it hands down. Without that cut, a planner minimises the
> downstream loss by proposing *easy-to-predict* plans instead of *good* ones — a degenerate optimum,
> measured-adjacent (it is the same failure the WM-canary guards against). This is the one place the
> "end-to-end" is deliberately not a differentiable loop, and it is a feature.

### 2.6 The staged path — v4 → v5, and a conservative fallback within v4

| Stage | What it adds | Cost | Gate that must pass first |
|---|---|---|---|
| **v4** (this doc) | the three-planner shape at ≤5 s over the ego-realisable action set: strategic subspace planner + tactical (5 s) + operative (2 s) diffusion, joint WM training, factorised selection | ~5 A40-days | G1 (§9) |
| ↳ **v4 fallback** (in-run, not a separate arm) | if the strategic subspace planner fails its 5 k verification (§7A.3) or the imagination-horizon test (§7A.4), the strategic layer **degrades to a produced-goal head** (E_strat + a route/speed classifier, no rollout). The arm still ships a strategic layer; only the *imagined-planning* part is cut. | 0 | subspace + imagination gates at 5 k |
| **v5** (deferred, data-gated) | 10–20 s strategic horizon · map-dependent routing (exits/merges) · agent-interaction tactics ("wait for the gap") | — | a map + agent-state labels + longer clips |

**Recommendation: v4 as one arm with switches, not two.** `--strategic {full,head,off}` and
`--lambda-plan 0` (Phase A) give clean in-run controls, so the arm can attribute its own result even
at 2 encoder-touching levers (§9). A two-arm split (prove two-diffusion first, add strategic second)
is the lower-risk alternative and is Sayed's call — it costs a second 30 k run (~5 A40-days) to buy
cleaner attribution the switches already provide. I recommend the single arm; I flag the choice.

### 2.7 What is kept, what is removed, and why

| Choice | The measurement behind it |
|---|---|
| **v1's ViT trunk**, not REF-C's ResNet | v1 owns the tail (miss@2m **0.0602 ± 0.0121** heldout vs REF-C-XL **0.1419** full-set), the causal-vision proof (vision effect **+1.325 m [+1.04, +1.64]**, CI-separated), and the only rollable dynamics (REF-C has `step_readout = None`). REGISTRY §1.2 / §4.1 |
| **REF-C's anchored decoder** as both planner instances | Works inside a from-scratch stack: REF-B v2 (time-anchored) **0.5921 ± 0.0685**, first REF-B to beat CV in every speed stratum. And [PM] §4.6 **exonerates it**: `n_modes` 1→13, `conf_norm` 40→189, `wta` 0.253→0.034, `man_acc` at v1's level. REGISTRY §3.5 |
| the **in-repo** `FlagshipV15Head`/`V15Decoder`, not a rewrite | Already subclasses `AnchoredDiffusionDecoder`, already repairs REF-C's selection flaw (ranks on `refined_logits`, supervises `sel_score`), already carries ReZero-gated goal seams with per-seam norm telemetry. `stack/tanitad/models/flagship_v15.py:170-234, 399-441, 537-592` |
| **d384 × 4L** decoders (8.5 M each), not v1.5's d512×8L (30.98 M) | base's 8.6 M decoder ties XL's 22.7 M on everything that ships and is **≥ XL at every matched K** (K=8: 1.686 vs 2.274; K=64: 0.283 vs 0.437; K=128: 0.191 vs 0.262). v1.5's 30.98 M decoder proposes **1.87× worse**. REGISTRY §4.3 |
| **256 anchors**, held at REF-C's vocabulary | ⚠️ **Justification CHANGED (§6.3).** The earlier reason — *"the fan is 32.6× longitudinal so lateral width is wasted"* — is **confounded and withdrawn**. The surviving reason is narrower and still sufficient: anchors cost **0 params**, so width is not a budget question, and the vocabulary question is answerable by an **eval-side K-sweep at zero GPU cost**. Hold at 256 for bit-comparability with REF-C and v1.5. **If v4's fan-diversification test (§6.3) shows the lateral collapse was conditioning-driven, a wider or lateral-stratified vocabulary becomes a live v4.1 lever.** |
| **dense (20-step) operative anchors** | +**24,608 params** and a 40 KB buffer (MEASURED §3.3) — the precondition for every smoothness term in §7. A 4-point path admits exactly **one** third difference. |

| Removed from v1 | Params | Why |
|---|---|---|
| `tactical_policy` (unimodal `wp_heads` + maneuver + intent) | **22,736,141** | Replaced by planner ③. Its own output scores **3.1501** against the same model's rollout **0.4522** — heads are lossy readouts (REGISTRY §5). Its `intent` seam is the program's only measured **HARMFUL** seam (F3, cos −0.238, norm 31.4 vs 28.3). |
| `tactical_pred` (a second predictor at horizons 8/16) | **26,534,912** | 🔬 Its horizons are **frames**: it runs at **0.8 s / 1.6 s**, strictly *inside* planner ③'s 2 s — **not** a "2–8 s coarse rollout" (§7A.1 corrects an earlier draft of this table). A second trajectory-latent head with no consumer. Its *actual* gap — a horizon above 2 s — is filled by planner ① and ②, whose consumer is the **goal level**, never another trajectory output. |
| `strategic_policy` d384×4 | **8,385,027** | Replaced by planner ①. Its measured behaviour is the reason: `route_skill_vs_chance` **0.0** — a pure command echo (F1 ② INERT). |
| `aux_accel` head | **528,897** | **[PM] #7** — `aux_accel_r2` never stabilised (0.003 → −0.119 → −0.423 → 0.447). |

---

## 3. Parameter budget — every module MEASURED

**Method.** Each module was instantiated this session from the committed configs and counted
(`scratchpad/v4_param_budget.py`, `v4_dense.py`, `v4_pred.py`, run under
`C:/Users/Admin/venvs/tanitad`). Validation: the same script reproduces
`WorldModel(flagship4b_config())` **total = 263,440,533**, byte-identical to REGISTRY §1.1. ✅

### 3.1 The v4 build — three planners over one world model

| Module | Params | Status | Source |
|---|---:|---|---|
| **— shared trunk —** | | | |
| ViT encoder d768×12, 9-ch/256 px/p16 | **87,022,848** | v1 verbatim | instantiated ✅ |
| SpatialGridReadout 4×4×128 → 2048 | **98,432** | v1 verbatim | instantiated ✅ (sum = **87,121,280** = REGISTRY §1.2 encoder row) |
| Operative predictor d768×10 + inverse-dynamics, `action_dim 3` | **96,609,283** | v1 verbatim | REGISTRY §1.2 run config ✅ *(at `action_dim 2` instantiation gives 96,607,490; the 1,793 delta is the speed channel)* |
| H15 ImaginationField | **22,055,683** | v1 verbatim | instantiated ✅ = REGISTRY |
| **— ① STRATEGIC PLANNER (new) —** | | | |
| `E_strat` compression 2048 → **d_strat 128** | **1,114,752** | NEW | instantiated ✅ |
| strategic predictor **in the 128-d subspace** (d256 × 4L, `action_dim 4`) | **3,919,744** | NEW — its own imagination | instantiated ✅ |
| strategic option-prior head (8 strategic actions) | **35,080** | NEW | instantiated ✅ |
| goal-scalar head (ttm · curvature@3/5 s · target-speed@5 s) | **17,286** | NEW | instantiated ✅ |
| `z_strat` → KV token projection (+ ReZero) | **66,049** | NEW | instantiated ✅ |
| **strategic planner subtotal** | **5,152,911** | | |
| **direct-head baselines** for `imagination_horizon_scaling`: predictor horizons {1,2,4} → **{1,2,4,20,50}** (2 × `Linear(768,2048)`) | **3,149,824** | NEW — instrumentation, and it *is* the falsifier's control arm (§7A.4) | instantiated ✅ (91,360,512 → 94,510,336) |
| **— ② TACTICAL PLANNER (new) — diffusion instance #1, 5 s coarse** | | | |
| `FlagshipV15Head` d384×4L, 256 anchors, horizons (5,10,…,50) | **9,767,320** *(decoder 8,544,405; anchor buffer 5,120 floats)* | NEW | instantiated ✅ |
| **— ③ OPERATIVE PLANNER — diffusion instance #2, 2 s dense** | | | |
| `FlagshipV15Head` d384×4L, 256 anchors, **dense 20-step** | **9,778,604** *(decoder 8,559,785)* | NEW | instantiated ✅ |
| Factorised LAT(8)/LON(7)/DIST(8) heads + 3 zero-init anchor grafts | **≤ 811,543** | NEW | instantiated ✅ *(upper bound at `aux_hidden` 512)* |
| S3 plan → action channel | **0** | reuses `InverseDynamicsHead` + the existing 3-dim channel | — |
| **— removed from v1 —** | | | |
| ~~`tactical_policy`~~ | −22,736,141 | REMOVED (§2.7) | REGISTRY §1.2 |
| ~~`tactical_pred`~~ | −26,534,912 | REMOVED (§2.7) | REGISTRY §1.1 |
| ~~`strategic_policy` d384×4~~ | −8,385,027 | REMOVED — replaced by planner ① | REGISTRY §1.2 |
| ~~aux-accel head~~ | −528,897 | **REMOVED [PM] #7** | `aux_accel_r2` never stabilised |
| **model subtotal** | **234,446,448** | | |
| grounding heads (op/tac/str, outside the model) | 13,432,338 | v1 verbatim | REGISTRY §1.2 |
| **v4 TRAINABLE TOTAL** | **≈ 247,878,786** | | **62 % of the 400 M cap** |

**For comparison (all MEASURED):** flagship v1 trainable **277,404,073** · flagship v2/v3enc
**286,339,251** · REF-C-XL **251,932,584** · REF-B v2 **271,619,880** · REF-C-base **104,191,577**.
**Three planners still cost less than any flagship the program has trained** — because the trunk
dominates and the planners are 9.8 M each.

> ⭐ **Why the hierarchy is affordable, in one line:** the two diffusion instances together are
> **19,545,924** — a **9.8 M marginal cost** for a whole extra planner — and the strategic planner is
> **5.2 M** because it thinks in a 128-d subspace. **The anchor vocabularies are buffers: 0
> parameters** (§3.3). Sayed's three-planner architecture costs ~**25 M**, about **9 %** of v1.

### 3.2 Encoder share — the allocation argument, re-read

V35 §7.1 argued from *"REF-C spends 79 % of its budget on the encoder and owns the better fan"*.
That argument is now **dead**, because the within-REF-C control exists: base's **90,458,632**-param
encoder ties XL's **199,496,532** on ADE/FDE/miss with all three paired intervals straddling zero,
and base is **≥ XL at every matched anchor count**. On the only near-matched cross-arm test we have —
base's 90.5 M vs flagship v1's 87.1 M encoder, within **3.8 %** — 2.2× the encoder bought **0.001 m**.
v4 therefore keeps v1's 87.1 M encoder and treats encoder capacity as a closed question at this
corpus size. (MEASURED, REGISTRY §4.3 verdict + fan table.)

### 3.3 Two cheap structural facts, both measured this session

| Fact | Measurement |
|---|---|
| **Anchor count costs zero parameters.** | `FlagshipV15Head` decoder params at 128 / 256 / 512 anchors are **identical: 8,535,177** (sparse) and **8,559,785** (dense). Anchors are a **buffer**: 2,048 floats at 256×4, **10,240** at 256×20, 20,480 at 512×20. |
| **Making the plan dense (4 waypoints → 20 steps) costs +24,608 params.** | d384×4L: 8,535,177 → **8,559,785**. d512×8L: 29,761,033 → 29,793,833 (+32,800). |

The second is the cheapest high-leverage change in this document; §7 is why.

### 3.4 The unspent ~159 M — and what I refuse to spend it on

Every capacity axis with a measurement returned nothing:

- **encoder**: 2.2× → +0.0013 m, not separated (REGISTRY §4.3).
- **decoder**: 22.7 M → 8.6 M, ties; and the program's *largest* decoder (v1.5, 30.98 M measured) is
  its *worst* proposer (oracle 0.3073). (REGISTRY §4.1/§4.3 + §3.1 here.)
- **anchors**: free in params; the constraint is not vocabulary width (§2.2).

The one un-measured axis that plausibly serves ask #2 is the **operative predictor**. Priced this
session: d768×10 → d768×14 = **124,436,736** (+33.1 M); → d768×16 = **140,974,848** (+49.6 M), taking
v4 to ≈ 288.6 M — still under cap. **I recommend against folding it into v4.** Predictor capacity has
never been measured in this program, and adding an unmeasured capacity change to three structural
changes is exactly the v2 mistake (N6: v2 switched ten levers at once and died of simultaneity).
It belongs in a **separate arm `v4.1-pred`** with its own gate, after v4 has a number.

**Recommendation to Sayed: the 400 M cap is not binding on v4, and it should not be made binding.
The binding constraint is structure, and v4 spends everything it has on structure.**

⭐ **The three-planner hierarchy (§2, §7A) is the one exception, and it proves the rule.** It spends **~25 M** — not
because headroom exists, but because a source read showed the strategic level has no timescale of its
own, and a coverage measurement showed exactly how far a timescale can honestly reach (5 s at 74.3 %).
**That is a capacity spend with a reason and a falsifier**, which is precisely what encoder widening
lacked.

---

## 4. Losses

### 4.1 Carried verbatim from flagship v1 (the WM stack — this is the half v1.6 deleted)

| Term | Weight | Status |
|---|---|---|
| JEPA latent prediction at k ∈ {1,2,4}, residual, change-weighted | 1.0 | v1 verbatim (REGISTRY §1 preset) |
| SIGReg `full_relaxed`, n_slices 512, β 1.0, **free_dims 64** | 0.1 | v1 verbatim |
| Inverse dynamics | 0.5 | v1 verbatim — and S3 reuses this head |
| K-step recursive rollout, `rollout_k = **4**, constant` | per preset | v1 verbatim. **[PM] DO-NOT-CARRY #2**: never raise K before the rollout is healthy — v1 reached `g_op_fwd_ade_m` **0.1052** at 8–10 k with a constant K=4, while v3enc's staged 4→8→12 plateaued at **0.4717**; K also drives `needed_fut` 10→16 = **+33 % encoder forwards per step**. (§7.2) |
| Hierarchical metric grounding (`op`/`tac`/`str`: metric inv-dyn on real pairs + forward SE(2) consistency) | per preset | v1 verbatim — the closed-loop substrate, **and the instrument [PM] used to localise v3enc's failure** (term (a) vs term (b)) |
| H15 imagination loss (mask 0.5, w 0.5, observed_weight 0.1) | 0.5 | v1 verbatim |
| speed channel `v0 = poses[t,3] / SPEED_SCALE`, **SPEED_SCALE = 10.0**, **NEVER dropped, never zero-filled** | contract | v1 verbatim + **[PM] DO-NOT-CARRY #1** (§5.3) |
| ~~aux-accel head~~ | **OFF** | **[PM] DO-NOT-CARRY #7.** *"neither the cause of the regression nor worth the params"* |
| ~~v1-style 4-point `--jerk-weight`~~ | **OFF** | **[PM] #7** + §7.1. Replaced by the dense-plan penalty in §4.3 |
| `nav`/goal dropout as an echo-killer | 0.5 | **[PM] DO carry** — the only lever in the v2 pack with a clean isolated positive read: `route_acc` **0.98–1.00 → 0.70–0.83 at identical `nav_valid_frac`**, i.e. removing the command removed v1's measured command-echo. v4's stronger form: the goal is **produced, never fed** (§6.4) |

### 4.2 Carried verbatim from REF-C / v1.5 (the planner stack)

| Term | Weight | Status |
|---|---|---|
| Anchor-classification CE against the GT-nearest **original** anchor | 1.0 | `v15_losses` verbatim |
| L1 trajectory reconstruction from the **assigned** anchor | 1.0 | `v15_losses` verbatim |
| **Ranking CE on `sel_score`** — the exact quantity `argmax` selects on, i.e. the **refined** confidence after the denoise passes | 1.0 | `v15_losses` verbatim. **This IS registry rule N2 satisfied.** REF-C ranks the *un*-refined score; selecting on its discarded refined confidence scores **1.36593 = 2.9× WORSE than baseline** precisely because `refc_train` never supervises the conf head at denoise timesteps. v1.5 already supervises it; v4 inherits that, and it is the reason v4 may rank on refined confidence at all. MEASURED, REGISTRY §4.1 |
| 2 truncated denoise passes, `noise_std 0.1` | — | REF-C geometry verbatim |

### 4.3 New in v4

| Term | Weight | What it supervises | Evidence |
|---|---|---|---|
| **LAT CE** (8 logits: 7 kinematic LATMANEUVER + `unknown` sentinel, masked) | **PROPOSED 0.05** | the lateral tactical mode | `V3_FACTORIZED_TACTICAL_HEAD_SPEC.md` §2.1; labels already implemented (`--labels v3`, staged) |
| **LON CE** (7 logits: 6 kinematic LONMODE + sentinel, masked) | **PROPOSED 0.05** | the longitudinal tactical mode — **the term that does not exist today** | **24.9 %** of windows have a live LONMODE the 5-way cannot express (`coast` 308, `hold_stop` 273, `stop_at_point` 181, `creep` 152, `launch` 57); **44.1 %** have a LONMODE ≠ `free_cruise`. MEASURED, `label_v3_audit_val100.json` (⚠️ 100-ep local val build — see §6.5) |
| **DIST CE** (8 `DIST_BAND_TOKENS`, `d_unknown` masked) | **PROPOSED 0.05** | *distance to the next maneuver, in metres* — the actual instruction | metres not seconds: the same junction is "5 s" at 10 m/s and "2.5 s" at 20 m/s, and a deceleration profile is set by distance (`v² = 2ad`). `V3_FACTORIZED_TACTICAL_HEAD_SPEC.md` §2.3 |
| **Strategic goal CE** — ROUTE (9 tokens, `unknown` masked) + DIST band | **PROPOSED 0.1** | the strategic module must *produce* G_s at inference, so it must be trained to | §6.4 |
| ⭐ **Strategic 5 s prediction** — the strategic predictor's forward consistency in `z_strat` space — change-weighted latent MSE, **masked to the 74.3 % of windows that have 5 s of future** | **PROPOSED 0.5** (= v1's `pred` weight halved; it is the same objective at a longer, noisier horizon) | gives the strategic level a timescale of its own | §7A.4. Coverage **MEASURED**, 500 eps / 95,477 windows |
| ⭐ **Strategic goal-scalar regression off the imagined `z_strat`** — time-to-maneuver, curvature@3/5 s, target speed@5 s, each with its own valid mask | **PROPOSED 0.05 each** | makes the strategic goal *per-window and time-varying* — the quantity `per_window_content_helps ≈ 0` says is missing | §7A.4 |
| **Plan-smoothness penalty on the DENSE emitted plan**: `w_j·‖Δ³p‖² + w_κ·‖Δκ‖²` over 20 steps at 0.1 s — **the ONLY smoothness mechanism in v4** | **PROPOSED w_j 0.02, w_κ 0.01** | jerk and curvature-rate of the trajectory we actually ship | §7. This is the *"find another mechanism"* [PM] asks for: v1's `--jerk-weight 0.02` acts on a **4-point, non-scored** head and contributes **≤1e-4 of a ~4.0 loss** |
| **λ_plan** — a scalar on the planner gradient reaching the trunk | scheduled, §5.2 | the WM/planner trade, made explicit and controllable | v1.6's collapse is what this exists to prevent |

Total aux pressure from the three factorised CEs is 0.15, against REF-C's single 0.1 maneuver CE —
**PROPOSED, a knob, not a finding.** The `MANEUVER_WEIGHT` split is the same shape as the spec's.

**Kept but demoted to an instrument, not a lever:** v1.5's `sel_gate` longitudinal selection term
(`−|v_term(i) − v_target_reachable|`, learned scale, **init 0.0**). Its learned value *is* the
measurement of whether a VTARGET-derived term in the 2 s score helps. It stays at init 0 so it starts
byte-identical, is reported every milestone, and **v4's actual longitudinal selection signal is the
LON graft, not this.** (VTARGET at 2 s is refuted: GT-perfect speed-matcher **1.1236** vs baseline
0.4714; MAE 1.65 vs hold-v0's 0.475; braking windows **+0.51 m worse**. MEASURED, REGISTRY §4.1.)

---

## 5. The joint-training curriculum — the hard part

### 5.1 What v1.6 actually proves, and what it does not

MEASURED (REGISTRY §1.4b, `eval_v16.json`, canonical harness, 881 windows):

```
v1.6 = warm `ab` head + 4 unfrozen ViT blocks + unfrozen predictor, head-LR 1e-4 / trunk-LR 1e-5
   paired Δ(v1.6 − v1)  = +0.0104 m   CI95 [−0.0888, +0.1147]   NOT separated   (episode-cluster, B=2000)
   oracle-in-fan        0.3073 → 0.2815   (−8.4 % of the gap to REF-C's 0.1640)
   WM canary            0.452  → 1.1022   (+144 %, 2.44×)
```

**The mechanism is not mysterious and it is decisive for v4.** `train_flagship_v16.py`'s loss is
`v15_losses` **only** — anchor CE + L1 + refined-rank CE. Its own docstring says it:
*"There is no separate trunk loss — the trunk adapts to minimise the SAME planning objective."*
**Nothing in v1.6 was holding the world model.** The trunk received planning gradient and no
world-model gradient for 6,000 steps, and the world model went away. That is the expected outcome of
that experiment, not a surprising one — and it is **not** evidence that a jointly-trained WM+planner
collapses, because that configuration has never been run.

**What v1.6 does prove and v4 must respect:** planning gradient on the trunk is *strong enough to
reshape it in 6 k steps*, so it must be **bounded**, and the canary must be a **gate, not a log
line**.

### 5.2 The curriculum

**Warm-start the trunk from `flagship4b-speedjerk-30k`; one run; three phases; λ_plan is the control
variable.** Warm-start (not from-scratch) because it gives the canary a *real* baseline from step 0,
because the trunk is already at 0.4271 full-set, and because it converts the LP phase of LP-FT into
the first phase of the same run (P16: LP-FT is proven; v2's everything-at-once diverged).

| Phase | Steps (PROPOSED) | λ_plan (planner grad → trunk) | What trains | Rationale |
|---|---|---|---|---|
| **A — LP** | 0 → 2,000 | **0.0** | planner head + factorised heads + S1 + strategic heads. Trunk trains under the **WM losses only** (which are live from step 0). | Reproduces v1.5's regime with a *moving* trunk underneath, so the head never emits garbage gradient into the trunk. LP-FT ordering, P16. |
| **B — ramp** | 2,000 → 8,000 | **0 → 1.0 linear**, with the canary clamp active | everything | The only phase where the WM/planner trade is actually negotiated. The clamp is a controller, not a kill (§5.5). |
| **C — joint** | 8,000 → 30,000 | **1.0** | everything | Sayed's "at the same time", in full. |

> 🔧 **REVISED 2026-07-21 (§14.3 O-17) — both scheduled-sampling ramps now share the phase
> boundaries above.** They previously ran teacher-forced to **10,000** and ramped to **20,000**, on a
> clock unrelated to λ_plan's — which put **the G1 gate at step 10,000 inside the ramp**, i.e. the arm
> would have been judged (on a produced-goal primary, with `nonav_route_beats_majority` as a KILL
> secondary) at the one step where it had **zero** training in its evaluated configuration. That is a
> restart for a schedule reason, from a family capped at 2. **One curriculum, one set of boundaries.**

**Scheduled sampling on S3** (V35 §2A.4, carried): teacher-forced GT-nearest anchor for steps
**0–2,000 (Phase A)**, linear ramp to own-selected across **2,000–8,000 (Phase B)**, own-selected
from **8,000**. **PROPOSED — chosen by convention (DAgger-style annealing), not measured in this
program.** Read at every milestone, under both conditions.

**Scheduled sampling on the GOAL too** (new, and it is the nav_cmd fix): S1's tokens come from the GT
label for steps **0–2,000**, ramp to the strategic module's own **produced** G_s across
**2,000–8,000**, produced from **8,000** — so **G1 sees 2,000 steps of fully produced-goal training**.
**Every leaderboard number is computed with the produced goal.** Both are reported.

**Discriminative LR (PROPOSED, standard practice, not program-measured):** head 1e-4, trunk 3e-4 with
v1's warmup 2000 + cosine — note this is *not* v1.6's 0.1× trunk LR, because in v4 the trunk is
being trained by its own objective, not merely fine-tuned. Optimizer AdamW 3e-4 / wd 0.05 /
betas (0.9, 0.95) — v1 verbatim.

### 5.3 Stop-gradients and dropouts (the circularity contract)

1. **`stop_gradient(G_t)`** where the plan enters the operative. Without it the planner can minimise
   the operative loss by proposing easy-to-predict plans. Non-negotiable.
2. **Goal-dropout 0.5 on S1 and S3** — half of every batch trains the plan-free, goal-free path,
   which is the path the canary measures and the path v1 deploys.
3. **No leaderboard number may come from a GT-derived plan or a GT-derived goal.** During teacher
   forcing, G_t is derived from the future GT while the operative is scored on predicting that same
   future; the headline eval therefore always runs produced-goal / own-selected-plan.
4. **⭐ [PM] THE ZERO-FILL RULE — now the single most load-bearing implementation constraint in this
   design, because it is the measured root cause of v3enc.**

   > **Never zero-fill a channel whose zero is a valid in-distribution value, and never apply a
   > planner dropout mask to the operative action channel.**

   Mechanism, read from source (`flagship_losses.py:227-231`): v3enc's `v2_ego_dropout=0.25` multiplies
   the planner's keep-mask into `v0a` and concatenates it onto **both `actions` and `fut_actions`**.
   The base channels are `(steer_rad, accel_mps2)` — control space, carrying **no absolute speed** — so
   `v0 = 0.0` after `SPEED_SCALE` is a perfectly in-distribution **"stationary"**. It is not a mask; it
   is a confident lie, through the entire K-step rollout, with no validity flag and no null embedding.
   MEASURED consequences: `inv` **0.3784** vs v1's **0.2194** and the **no-speed control's 0.3644** —
   v3enc's action-inverse-dynamics carries no more speed information than an arm with no speed channel
   at all; and only **18.6 %** of the speed-channel benefit recovered at 8–10 k (v1: **81.8 %**).

   **How v4 obeys it, in three places:**
   - **The operative action channel is never masked.** `v0` is fed at full strength on 100 % of samples.
   - **The planner's own `ego_dropout 0.5`** (REF-C convention, `flagship_v15.py:348-351`) currently
     zero-fills the same way (`v = v * keep`). **v4 changes it to a learned null-embedding row** — the
     exact discipline `FlagshipV15Head` already uses for `VT_DROPPED` and `ROUTE_DROPPED`, where
     *"the DROPPED index is a real embedding row, so the head learns an explicit 'no goal given' state
     rather than seeing a zero it could confuse with a class."* The same argument applies verbatim to
     speed and was simply not applied there. **This is a required code change, not an option.**
   - **S1/S3 goal-dropout** already uses distinct DROPPED rows and stays as-is.

### 5.4 The imagination-gradient decision — and its cost

`imagine_probes` is `@torch.no_grad()` (`flagship_v15.py:478`). In v1.5 the predictor was frozen, so
that was free. In v4 the predictor trains, and a fully differentiable probe rollout would be
**8 probes × 20 sequential predictor steps with autograd, every training step** — against v1's
existing 4-step rollout. That is not affordable and is **refused**.

**PROPOSED: `--probe-grad one`.** 7 of 8 probes stay under `no_grad` (mechanism byte-identical to the
proven v1.5 seam); **one probe, sampled stochastically per batch, is rolled with gradient**, giving
the planner→predictor coupling that ask #1 asks for at a bounded cost of roughly one extra 20-step
recursion. Flag values `{none, one, all}`; `all` exists only so the ablation is possible.

⚠️ **This cost is an ESTIMATE and must be measured before launch.** G0 (§9) is a 200-step pace probe
on pod1: if s/step exceeds the pre-registered ceiling, `--probe-grad` drops to `none` automatically
and that is recorded, not argued.

> 🔴 **CORRECTION (§14.3 O-10) — the recursion is 50 steps, not 20, and that changes the number.**
> `V15Config.probe_steps = 20` (`flagship_v15.py:120`) and **every MEASURED latency figure in this
> document is at 20 steps.** But §7A.4 has the tactical planner ② read the *shared* probe rollout at
> **(10,20,30,40,50)** frames, which requires **`probe_steps = 50` — a 2.5× longer recursion**, in
> training and at inference alike. So the grad-probe surcharge is *"one extra **50**-step recursion"*,
> and §9.2's **+25 %** becomes a PROPOSED **+45 % ⇒ ~15.6 s/step** — **inside the 16 s/step G0
> ceiling, but only just.** G0 is therefore load-bearing rather than a formality, and the fallback
> ladder is pre-committed in priority order: `--probe-grad none` → `--tac-probe-steps 20` (② keeps 5 s
> *anchors* but drops to a 2 s *imagination* — a documented weakening, not a silent one) → probes
> 8→4. **Never the encoder** (R4).

### 5.5 The canary — a gate and a controller, and the distinction matters

**Definition (MEASURED protocol, already implemented at `train_flagship_v16.py:271-303`):** roll the
operative predictor under **TRUE** actions through `grounding.step['op']` and accumulate SE(2) —
exactly v1's intent-free deployed path — with **S1 and S3 zeroed**. Baseline established at step 0 on
the warm trunk, harness-consistent. v1's reference value is **0.452**.

| Layer | Trigger | Action | Protocol status |
|---|---|---|---|
| **Controller** (in-loop, every 500 steps) | `canary_vs_base > +0.05 m` | λ_plan × 0.5, logged, and re-checked at the next eval | ✅ compatible with `GATE_PROTOCOL.md` — it is a *controller*, it never kills a run |
| **Alarm** (in-loop) | `canary_vs_base > +0.30 m` on 3 consecutive evals | λ_plan → 0, alarm raised, run continues to its gate | ✅ still not a kill |
| **Gate** (at the pre-registered step ONLY) | `wm_canary_ade_2s > 0.55` at step 10,000 | verdict `RESTART` via `run_gate.py check` | ✅ the only place a run may be killed |

> This is deliberate. `GATE_PROTOCOL.md` §1: *"No run is killed or continued except at a
> pre-registered gate step, on a held-out metric, against a threshold written down before launch."*
> A canary-triggered mid-run kill would violate it. A canary-triggered λ_plan reduction does not.

---

## 6. Selection conditioned on tactical and strategic constraints

### 6.1 The defect being fixed — measured, and it is structural

```
refc.py:88-91   N_MANEUVERS = 5
                0 lane_keep | 1 turn_left | 2 turn_right ‖ 3 accelerate | 4 brake_stop
                \______ 3 LATERAL ______/                \_ 2 LONGITUDINAL _/
                                    ONE softmax
refb_labels.py  classify_maneuver resolves the collision by PRIORITY: turn > brake > accel > lane_keep
refc.py:569-571 conf += maneuver_to_anchor(log_softmax(maneuver_logits))   ← straight into the RANKING
```

| Measured | Value | Source |
|---|---|---|
| `accelerate` predicted | **0 / 881** (both REF-C arms) | `taniteval/results/planfan_clips_tactical_head_val.json` |
| `brake_stop` predicted | 7/881 base · 4/881 XL | same |
| GT longitudinal (survivors of the priority) | **195/881 = 22.1 %** | same |
| GT longitudinal **present** before the priority collapse | **242/881 = 27.5 %** | `canonical_fan_probe.json` (881 windows, 40 canonical episodes) |
| destroyed by `turn > brake > accel` | **63/881 = 7.2 %** | same |

So a **~99.5 % lateral-or-neutral prior** enters the anchor logits, and **there is no longitudinal
signal anywhere in the selection path.** Meanwhile the fan itself is a *speed* fan:

| | base (128) | XL (256) | source |
|---|---|---|---|
| spread along track (mean/median) | 0.719 / 0.703 m | 0.779 / 0.748 m | `PLANNER_VIZ_CONCEPT.md` §10(a), 881 windows |
| spread across track (mean/median) | 0.104 / **0.014 m** | 0.122 / **0.017 m** | same |
| median long/lat ratio | **32.6×** | **32.8×** | same |
| windows longitudinally dominated (>2×) | **88.9 %** | 88.5 % | same |
| windows laterally dominated (<0.5×) | **0.0 %** | **0.0 %** | same |
| selected-vs-oracle separation, longitudinal share | **80.7 %** | **81.1 %** | same |
| oracle is a genuinely different lateral mode (\|Δlat\| > 2 m) | 12/881 (**1.4 %**) | 17/881 (**1.9 %**) | same |

Even on sharp-curvature windows (|net heading@2s| > 20°, n = 122) the ratio is 1.89 / 1.63 — the
lateral spread never exceeds the longitudinal one. **1.4 cm** lateral spread on straight windows.

### 6.2 The change — factorised LAT × LON × DIST into the anchor logits

```python
# today (refc.py:569-571)
conf = conf + self.maneuver_to_anchor(log_softmax(maneuver_logits))          # rank-5, ~99.5 % lateral

# v4
conf = conf + self.lat_to_anchor (log_softmax(lat_logits))    # Linear(8, N), default init
            + self.lon_to_anchor (log_softmax(lon_logits))    # Linear(7, N), ZERO-INIT
            + self.dist_to_anchor(log_softmax(dist_logits))   # Linear(8, N), ZERO-INIT
```

Four disciplines, each with a reason:

1. **Additive and separate, never concatenated into one Linear.** Two rank-limited terms are
   ablatable; `lon_to_anchor` can be zeroed and measured. A single graft never allowed that.
2. **Zero-init the two new grafts** (ReZero discipline, as `ctx_to_cond` already does) so the model
   starts **byte-identical in the selection path** to the 5-way baseline and the graft's effect is
   attributable rather than confounded with everything else v4 changes.
3. **Keep `maneuver_logits` [B,5] emitted and supervised for one milestone**, so the A/B is
   `5-way only` vs `5-way + LAT×LON×DIST` and every published REF-C number stays reproducible.
4. **Norm parity monitored and clamped.** Log `‖lat_to_anchor(·)‖` / `‖lon_to_anchor(·)‖` /
   `‖dist_to_anchor(·)‖` against `‖conf‖` every log step. *A graft that swamps `conf` is not a prior,
   it is a second selector.* Same hard clamp as S1: rescale in-graph at 1.0×, fail loud at 1.5×.

**Cost: ~5 k parameters for the grafts, ≤ 811,543 including the three heads (MEASURED, §3.1). This is
a structure change, not a capacity change.**

**The read (pre-registered):** `frac_sel_2x_worse` — REF-C **0.4540** XL / **0.4109** base
(MEASURED, REGISTRY §4.3) — and the *longitudinal share* of the selection gap, currently **80.7–81.1 %**.
The claim under test is that the longitudinal share shrinks once a longitudinal prior can reach the
ranking at all.

### 6.3 🔴 RETRACTED — "the fan is a SPEED fan, so strategic choice is a ~2 % lever"

**An earlier revision of this section scoped the strategic layer down to ~1.4–1.9 % of windows on the
strength of the fan-geometry measurement. That inference is CONFOUNDED and I withdraw it.**

**The measurement stands; the inference does not.** MEASURED (`PLANNER_VIZ_CONCEPT.md` §10(a),
881 windows): median long/lat spread ratio **32.6×**, **88.9 %** of windows longitudinally dominated,
**0.0 %** laterally dominated, **1.4 cm** lateral spread on straights, selected-vs-oracle separation
**80.7–81.1 %** longitudinal, oracle is a genuinely different lateral mode on **12–17/881
(1.4–1.9 %)**.

**Why it cannot be used to scope the strategic layer — a source read, three call sites:**

```
refc_eval.py:78     out = model(fw, nav_cmd=None, ...)      # ALL 881 windows -> follow
refc.py:786-788     if nav_cmd is None: nav_cmd = zeros(...)  # index 0 = "follow"
refc_train.py:250   nav_cmd = batch["nav_cmd"]               # the pre-v2 v1 net-heading command
plan_fan.py:549     o = model(fw, nav_cmd=None, ...)         # the viz too
```

**REF-C's decoder never had a working route input — at train time it got the circular,
straight-by-default v1 derivation, and at eval time a constant.** A decoder that was never
conditioned on a route cannot have learned to diversify laterally; it learns the marginal, and **the
marginal of this corpus is "straight, at some speed."** The collapsed lateral spread is therefore at
least as consistent with *"nothing ever conditioned it"* as with *"lateral choice does not matter."*

**Ruling: the 32.6× is UNRESOLVED. It is a lower bound measured under a broken route input, not a
ceiling on what strategic conditioning can do.** It may not be used to scope the strategic layer down,
and this document no longer does so.

#### 6.3.1 The discriminating experiment — and v4 is the first arm that can run it

v4 is the first arm in the program with a **working route input**: the goal is **produced, not fed**
(§6.4), trained on **v2.1/v3** route labels at ~75–80 % judgeable coverage (vs the 21–25 %
`nav_valid_frac` every previous arm trained on, [PM] §6), and evaluated with the produced goal rather
than a constant.

**Pre-registered test — `fan_lateral_diversification`, run at 5 k and every milestone:**

| | Procedure | Read |
|---|---|---|
| **Sensitivity** | Swap the produced `G_s` route token left↔right on the same window; re-decode the fan | Lateral spread of the live modes (p > 1 %) and the 2 s endpoint |
| **Direction-correctness** | — | route-right must move the lateral offset **right** — sensitivity alone is satisfiable by noise |
| **Content** | Real per-window goal vs a **constant/mean** goal | **`per_window_content_helps`**, never `helps_vs_none` (which flipped sign by encoder in v1's H26 panel while content stayed inert) |
| **Floor** | — | CI-separated **and** ≥ 0.02 acc / 0.05 m / 0.01 cos (`hierarchy.py` MIN_*) |

**Both outcomes are informative, and I am pre-committing to both readings:**

- **Fan diversifies laterally, direction-correct, content-positive** ⇒ the collapse was a
  *conditioning* artefact. The 32.6× is explained, strategic conditioning is load-bearing, and a
  **lateral-stratified anchor vocabulary becomes a live v4.1 lever** (§2.7).
- **Fan still collapses at ~1.4 cm under a real, produced, per-window route** ⇒ lateral choice
  genuinely does not matter at 2 s on this corpus. Then the strategic layer's value is **not** in the
  operative fan at all — it is at the tactical/5 s horizon and in the no-nav path prediction, and v4
  says so rather than quietly keeping a decorative seam.

#### 6.3.2 A distinction that must stay explicit — pruning is not re-ranking

⛔ **Post-hoc re-ranking of a fixed fan is CLOSED** (v1.0 recovered **0.0 %**; v1.2 **+2.9 %, not
significant**, paired Δ +0.00893 [−0.0062, +0.0250]; ~**92 %** of the oracle gap aleatoric, across 47
trained arms). v4 adds no re-scorer.

**A strategic constraint that REMOVES modes is a different operation**, and the distinction is not
rhetorical:

| | Re-scorer (CLOSED) | Strategic constraint (OPEN) |
|---|---|---|
| operates on | a **fixed** fan, after the fact | the fan's **generation and conditioning**, in-graph, during training |
| changes | the ranking over an unchanged candidate set | **which candidates exist** and what the decoder was trained to propose |
| evidence | 47 arms, ≤8.4 % of the gap learnable | untested — because no arm ever had a working route input (§6.3) |

The same distinction covers the **LON graft** (§6.2): it is a *missing input to the anchor logits*,
trained jointly, not a post-hoc score adjustment. That door is open; the re-scorer door is shut.

#### 6.3.3 What survives from the previous null

v1.5's `ab`→`abc` **+0.0106 m, CI [−0.094, +0.072]** remains a **broken test, not a result**: frozen
trunk, a seam the monitor measured **2.80× swamped**, 8 k head-only steps, mid-label-repair
(`fc2c484`, `3d41bd0` flag 2). What survives is narrow and still binding: **goal tokens bolted onto a
frozen fan at a mis-scaled seam do not improve ADE.** v4 tests something else.

> **In one sentence for Sayed:** *the tactical/longitudinal graft is still the highest-confidence
> change in this design; the strategic layer is no longer scoped down to 2 % — its value is genuinely
> UNRESOLVED, v4 is the first arm that can resolve it, and §6.3.1 is the experiment that does.*

### 6.4 The `nav_cmd` train/eval mismatch — fixed by construction

**MEASURED defect (source read, three call sites):** REF-C training passes a per-window v1-derived
`nav_cmd` (`refc_train.py:250,256`), while **eval passes `None` → index 0 = `follow` for all 881
windows** (`refc_eval.py:78` → `refc.py:786-788`; same at `plan_fan.py:549`). REF-C's goal
conditioning is therefore **untestable** as trained. (The flagship path is clean — verified: v1's
scored rollout takes no intent at all.)

**v4's fix is structural, not a patch:** the strategic module **produces** G_s from vision + ego and
is supervised by a masked CE against the label. **The label is a target, never an input.** The
planner conditions on the *produced* G_s at both train and eval (with the goal scheduled-sampling
ramp of §5.2 handling the early-training quality problem). This simultaneously removes:

- the train/eval mismatch (there is no fed command to mismatch);
- v1's **route echo** — F1's `route_logits` CE target is derived from the same future heading that
  produces `nav_cmd`, and the measured consequence is `route_skill_vs_chance` **0.0**;
- the eval-time ambiguity in the planner HUD (today's "strategic: route left" is the model's own
  `route_logits.argmax`, not an input — `plan_fan.py:561`).

**Precondition (code):** the v4 evaluator must feed the produced goal. `refc_eval.py` /
`plan_fan.py`'s `nav_cmd=None` constant must not be inherited.

### 6.5 Label reality — what is mintable and what is not

| Slot | Mintable today | Not mintable, and why |
|---|---|---|
| **ROUTE** (9 frozen tokens) | **5**: `follow`, `turn_left`, `turn_right`, `u_turn`, `roundabout` | **4 never minted**: `straight`, `exit_left`, `exit_right`, `merge` — all assert a **map** fact. `straight` is never emitted *by design* (`refb_labels.py:997-999`). ⚠️ **Correction to the task brief, which said "6 of 9": the in-repo figure is 4 of 9.** |
| **LONMODE** (9) | **6**: `free_cruise`, `coast`, `stop_at_point`, `hold_stop`, `creep`, `launch` | **3 structurally unmintable**: `follow_lead`, `close_gap`, `open_gap` — **`lead_state` is a `None` stub.** They are excluded from the head: *a logit no label can ever train is a dead parameter that only invites a shortcut.* |
| **TACPOINT** | — | `stop_line` stays `unknown`. Kinematics mints *where* the vehicle stops, never *why*: pedestrian crossing, stop line and a queue behind a lead vehicle are the same ego track. |
| **DIST band** (8) | yes, in metres | **39.7 % `d_unknown`** (873/2201) — the window did not reach far enough. `d_unknown` and `d_none` are separate tokens on purpose and `d_unknown` is **masked out of the CE**, exactly like `ROUTE_UNKNOWN`. |

**⭐ One genuine v4 advantage on labels, quantified by [PM].** Every flagship arm ever trained —
including the deployed v1 — trained with `nav_valid_frac` of only **0.21–0.25**, i.e. the strategic
route CE was masked out on ~75–79 % of windows in **all four arms** ([PM] §6, which is also why the
label confound is refuted as a *differentiator*). The v2.1/v3 labelers raise judgeable coverage to
**80.05 %** (REF-C-base config, 4,000-window sample) and **75.2 %** (v3 route audit, 546/2201
`unknown`). **v4 is the first flagship to train its strategic level on ~3× the supervision.** That
does not change the §6.3 refusal — coverage is not the binding constraint on strategic *selection*,
lateral-mode rarity is — but it is the difference between a level that is trained and one that is not.

⚠️ **Substrate caveat, binding on every label rate above.** The `label_v3_audit_val100.json` figures
are on the **100-episode local build `physicalai-val-bb543bdf7836`**, not the canonical 40-episode
`physicalai-val-0c5f7dac3b11`. Measured divergence: median `v0` **5.62 vs 10.29 m/s**; `v0 < 1 m/s`
**19.0 % vs 5.7 %**; 5-way `lane_keep` share **33.3 % vs 61.7 %**. The local build is markedly more
urban, so **24.9 % / 44.1 % / 39.7 % are UPPER estimates for the canonical corpus.**
**Highest-value follow-up before launch:** run `scripts/label_v3_audit.py --val <canonical epcache>`
on the eval pod — minutes of work, and it converts three PROPOSED loss weights from
upper-bound-motivated to corpus-motivated.

**No VLM labels in v4 training.** 595 records vs 406,099 train windows = **0.15 %**; direction is
chance (**57.1 %, CI [0.400, 0.745]**, and the enum-order probe attributes the bias to the model, not
our prompt: recall on true right turns was **bit-identical 0.2069** in both enum orders); turn recall
**76–79 %**; LONMODE render-stability 62 %. Provenance split stays binding: **kinematics own
VTARGET/LONMODE/LATMANEUVER/DYN/HEADWAY; the VLM owns the WHY.**

---

## 7. Smoothness and planned speed — the axis we are worst on

### 7.1 🔬 The finding: v1's jerk penalty does not touch the scored path

**Source read, `stack/tanitad/train/flagship_losses.py:303-312`:**

```python
wp_h = cfg.tactical_policy.waypoint_horizons          # = (5, 10, 15, 20)  -> FOUR points
...
if w_jerk > 0.0 and len(wp_h) >= 4:
    paths = [wp_pred]                                  # the TACTICAL HEAD's waypoints
    loss_jerk = sum(torch.diff(p / pose_scale, n=3, dim=1).pow(2).mean() for p in paths) / len(paths)
```

Two facts follow, and neither has been stated in the program before:

1. **`torch.diff(n=3)` on a 4-point path returns exactly ONE number.** The deployed v1's
   `--jerk-weight 0.02` is a single third-difference over 0.5 s-spaced waypoints.
2. **It is applied to `wp_pred` — the tactical head's output — which is not the scored path.** The
   scored path is `metric_dynamics.rollout_decode`: 20 sequential predictor steps → per-step metric
   Δpose → SE(2) accumulate. **It receives no smoothness penalty of any kind.** (Confirmed
   independently: `ARCHITECTURE_WIRING_COMPARISON.md` §2.6 — v1's headline number comes from a path
   that touches none of its seams either.)

**So "v1 already has `--jerk-weight`" is true and nearly vacuous.** The measured symptom is exactly
what an unsmoothed emitted path looks like.

**[PM] independently quantified how vacuous, and extended it to `--aux-accel`:** v1's `jerk` term ran
**3e-4…3.7e-3 at weight 0.02 — ≤1e-4 of a ~4.0 total loss**; and v1's auxiliary acceleration head
(528,897 params) **never learned a stable signal** — `aux_accel_r2` = 0.003 / 0.031 / **−0.119** /
0.209 / **−0.423** / 0.049 / 0.447 / 0.328 across the run, negative twice.
**DO-NOT-CARRY #7: neither is worth restoring in v4.** v4 therefore drops both and buys smoothness
with a mechanism that acts on the path we ship.

### 7.2 The second finding: the recursive horizon is 5× shorter than the scored horizon

`rollout_k = 4` (REGISTRY §1.2 flags; `flagship_losses.py:258`) — the recursive rollout loss composes
**4** predictor steps (0.4 s). The evaluator composes **20** (2 s) (REGISTRY §1.2, planning-tick
definition). **Steps 5–20 of the shipped trajectory are never trained as a composed path.**

**[PM] settles what to do about it, and the answer is: do not raise K.** v3enc staged
`rollout_k` 4 → 8 → 12 and its `g_op_fwd_ade_m` **plateaued from ~4,500 at ~0.47**, while v1 reached
**0.1052** at 8–10 k on a **constant K=4**. Raising K also drives `needed_fut` 10 → 16 = **+33 %
encoder forwards per step**, so you pay a third more FLOPs for the regression. **DO-NOT-CARRY #2:
raise K only after a matched-step check against K=4 shows the rollout is healthy.**

**v4 holds `rollout_k = 4` and solves the horizon mismatch on the planner side instead: the planner
emits the whole 2 s path in one shot, with no recursion at all.** Raising K becomes a v4.1 lever,
unlocked only by the rollout-health secondary of §7.5 passing.

### 7.3 What v4 changes, and what each change is measured against

| Change | Mechanism | Pre-registered read |
|---|---|---|
| **The shipped open-loop trajectory becomes the planner's dense 20-step anchored plan** | anchors are FPS samples of **real GT trajectories** — every member of the vocabulary is a smooth, physically realised path by construction | Straight-road heading. MEASURED pins: flagship **7.980°** mean / **1.105°** median vs CV **1.399° / 0.451°**; REF-C arms **3.863°** — *"7.980° is a flagship property, not a corpus artifact"* (`TANITEVAL_V2_METRIC_SUITE.md` §8). Metric key `heading_med_2s_deg`, median reducer |
| **Dense anchors make jerk computable** | 20 steps at 0.1 s → 17 third-differences instead of 1. Cost **+24,608 params** (MEASURED, §3.3) | jerk term reported per milestone; TMS (open-loop) as the corpus-level companion |
| **Explicit smoothness penalty on the emitted plan** | `w_j·‖Δ³p‖² + w_κ·‖Δκ‖²`, PROPOSED 0.02 / 0.01 | ditto |
| **The LON head + `lon_to_anchor` graft** | gives the *speed profile* of the plan a supervised prior for the first time (§6.2) | **L1 CRUISE-QUALITY.** MEASURED pin: on the **639 steady windows** flagship speed MAE **0.4231 m/s** vs hold-v0 **0.2109**, paired Δ **−0.2122 [−0.2778, −0.1443] separated AGAINST the model** — *2.0× worse than doing nothing*. Metric keys `speed_mae_mps`, verdict block `cruise_speed_vs_holdv0`, estimator episode-cluster bootstrap paired |
| ~~`aux_accel` + v1's jerk term~~ **DROPPED [PM] #7** | — | L2 TRANSIENT-RESPONSE must nonetheless not regress: MEASURED pins brake n=95 Δ **+0.6433 [+0.4466, +0.8276]**, accel n=147 Δ **+0.5716 [+0.4336, +0.7076]**, both separated **for** the model. **If dropping aux-accel costs L2, that is a falsification of [PM] #7 and it will be visible here.** |

⚠️ **The heading gate must be PER CURVATURE BUCKET, never aggregate — [PM] proved why.** v3enc improved
straight-road heading to **3.642°** (v1: 7.980°) on the identical 634 windows, and *simultaneously*
degraded gentle curves **2.060° → 10.085° (4.9×)** and sharp **3.811° → 17.777° (4.7×)**. The aggregate
was a **tie** (6.706 vs 6.606). An aggregate heading number would have hidden both the win and the
loss. v4's gate reads straight / gentle / sharp separately, against CV's **1.399 / 7.852 / 28.743°**.
*(One honest credit to v3enc that v4 should not discard: it **beats the CV floor on sharp-curve
heading**, paired Δ **+10.97 [+7.18, +15.26]**, CI-separated for the model.)*

> ⭐ **The gate that matters most for ask #3, and it would be a program first.**
> **All 14 leaderboard arms are CI-separated *against* hold-v0 on the 639 steady windows**, from
> −0.054 (refc-base) to −7.28 (flagship v2). MEASURED, `TANITEVAL_V2_METRIC_SUITE.md` §8 /
> `MODEL_REGISTRY.md` §1120-1124. **v4's 30 k gate requires `cruise_speed_vs_holdv0` to be
> NOT-separated-against, i.e. its paired CI must include zero or favour the model.** No arm in the
> program has ever done that. If v4 passes it, that is the answer to *"improve the planned speed"* —
> and it is a claim with a pre-registered, corpus-wide, interval-based test behind it.
>
> ⚠️ One arm is close and should be read before launch: **`flagship-v16-ab-ft` is the only arm in the
> program whose speed MAE beats CV with a separated interval** (REGISTRY §1120-1124) — the same arm
> whose world model collapsed. That is a hint that *planner-emitted paths track speed better*, which
> is exactly v4's thesis, and simultaneously a warning about the price v1.6 paid for it.

### 7.4 The two "imaginations" must not be conflated

| | What it is | Measured |
|---|---|---|
| **(a) H15 `ImaginationField`** (22.06 M) | a belief field over *unobserved space*, refining encoder tokens | flat at `vision_use ≈ 12 %` since 19 k |
| **(b) consequence rollout** (v1.5's) | the predictor rolled under probe action sequences; the decoder sees the *consequences* of candidate controls | `a`→`ab` **−0.1355 m (−19.9 %), CI [0.038, 0.233] SEPARATED** |

**Only (b) has ever been measured to help.** v4 improves (b) — it is the KV source, it now carries a
gradient path to the predictor (§5.4), and it is what S3 converts into actions. (a) is kept
**verbatim, unchanged**, because removing it would be a fourth simultaneous change; it is named as a
v4.1 ablation candidate (worth 22.06 M and a latency slice if it is genuinely inert).

### 7.5 ⭐ [PM] Ask #2, retargeted: the objective is multi-step ROLLOUT fidelity

The post-mortem's term-level localisation is the most useful thing anyone has produced for ask #2,
because it says where *not* to aim. `grounding_losses` separates into two logged terms per level:

| term | what it is | v3enc / v1 at the matched 8–10 k bucket |
|---|---|---|
| **(a) `g_*_mid_de_m`** | metric inverse dynamics on **real latent pairs**, **no action input** — the encoder's own metric grounding | `op` **1.15×** · `tac` 1.13× · `str` 1.38× — **nearly intact** |
| **(b) `g_*_fwd_ade_m`** | forward metric consistency on the **predictor rollout with true actions** | `op` **4.48×** · `tac` 4.18× · `str` 3.89× — **collapsed** |

Plus: step-1 operative speed R² **0.9529** (v1 0.9987) — the readout is fine; the failure **compounds
over the rollout**. And the imagination is **damped and oscillatory**: `znorm` 40.7 → 75.9 (**×1.86**)
against v1's 40.1 → 147.9 (**×3.68**); `zcos` ends **0.618** vs v1's 0.397; the mean rollout speed
profile is **non-monotone** (12.91 → 12.14 → 13.05 → 12.63) where v1's drifts monotonically.

**So "strengthen the prediction quality and imagination of the WM" means: make the 20-step rollout
commit.** Not the encoder (intact), not the one-step readout (0.95 R²).

**The gate secondary this earns — the post-mortem's own recommendation, adopted verbatim.**
`encoder_speed_probe_r2` stays a secondary but is *not where the damage was*, so v4 adds a
**rollout-side** statistic: the **fraction of the speed-channel benefit recovered** against the
no-speed ablation control, `(nospeed − arm) / nospeed` on matched-step `g_op_fwd_ade_m`. It removes
the level, localises the failure in one number, and both reference logs already exist.

| bucket | 2–4k | 4–6k | 6–8k | **8–10k** |
|---|---:|---:|---:|---:|
| **v1 recovers** | 64.0 % | 67.7 % | 75.7 % | **81.8 %** |
| v3enc recovers | 32.4 % | 37.0 % | 36.2 % | **18.6 %** *(and falling)* |

**v4's pre-registered secondary: `speed_benefit_recovered_frac ≥ 0.70` at the 10 k gate** — PROPOSED,
anchored on v1's MEASURED 81.8 % and v3enc's 18.6 %. It also doubles as the unlock condition for
raising `rollout_k` in v4.1 (§7.2).

---

## 7A. The horizon question, and the strategic planner spec

> **Sayed, 2026-07-21:** *"how are we assessing tactical, operative and strategic plan if we are
> eliminating tactical prediction?"*

### 7A.1 🔬 First, a correction to §2.3 of this document — I had the horizons wrong

**Source read, `flagship_losses.py:59-98` + `config.py`.** All horizons are in **operative frames at
10 Hz**, and `horizon_plan()` composes them:

| symbol | value | seconds | what it is |
|---|---|---|---|
| `op_h` = `predictor.horizons` | **(1, 2, 4)** | 0.1 / 0.2 / **0.4 s** | the operative JEPA heads |
| `tac_h` = `tactical_pred.horizons` | **(8, 16)** | 0.8 / **1.6 s** | the head v4 removes |
| `wp_h` = `tactical_policy.waypoint_horizons` | (5,10,15,20) | 0.5…**2.0 s** | the waypoint surface |
| `goal_h` = `max(wp_h)` | **20** | **2.0 s** | the tactical goal latent |
| `str_h` = `(goal_h,)` | **(20,)** | **2.0 s** | *the strategic level's own grounding horizon* |
| `max_h` | **20** | **2.0 s** | the ceiling the dataset must supply |

Two consequences, and the second is the important one:

1. 🟥 **§2.3's claim that `tactical_pred` had a "2–8 s coarse-rollout role" is WRONG and is corrected
   here.** It runs at **0.8 s and 1.6 s** — strictly *inside* the 2 s the planner covers densely.
   Removing it removes nothing above 2 s. v4's predictive horizon is **not shorter than v1's**.
2. ⭐ **But Sayed's underlying point survives the correction and is stronger than its premise.**
   `str_h = (goal_h,) = 2.0 s`. **The "strategic" level has never had a timescale of its own — in v1,
   v2, v3enc, REF-A, REF-B or REF-C.** Every arm this program has trained tops out at `max_h = 20`
   frames. So the R3 hypothesis the coordinator raised is not speculation: if strategic information
   is only ever consumed to nudge a 2 s plan, **`per_window_content_helps ≈ 0` is what you would
   predict**, and no seam fix repairs it. The hierarchy would be decorative *by construction*.

### 7A.2 The supervision ceiling — MEASURED, and it settles the range

Measured this session by counting the local epcaches directly
(`scratchpad/horizon_coverage.py`, read-only; **500 episodes / 95,477 window starts** across
`physicalai-val-bb543bdf7836` (100 ep) and `physicalai-train-14231cd29c74` (400 ep); the two builds
agree to <0.1 pp on every row, so this is a property of the PhysicalAI source, not of a build):

**Clip length: 19.7–20.5 s, p50 19.9 s (≈199 frames @ 10 Hz).**

| future horizon needed | 2 s | 3 s | 5 s | **7 s** | **10 s** | 15 s | **20 s** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **fraction of window starts that have it** (train, stride 1) | 90.1 % | 84.8 % | **74.3 %** | **63.9 %** | **48.2 %** | 22.0 % | **0.0 %** |
| (eval, stride 8) | 91.7 % | 87.5 % | 75.0 % | 66.6 % | 50.0 % | 25.0 % | **0.0 %** |

> **A "2–20 s" module is not designable on this corpus. 20 s is supervisable on 0.0 % of windows —
> not rare, structurally impossible: no window start can have 20 s of future inside a 19.9 s clip.**
> This independently corroborates the VTARGET-mint defect analysis, which measured *"realized
> lookahead 3–19 s; only **53.5 %** had ≥10 s"* and responded by adopting a **5 s floor with an
> explicit valid mask** (V35 P12, commit `c4f75d6`). My 48.2 % lands in the same place by a different
> route.

**And the route labels say the same thing about what a longer horizon would even buy.** From
`label_v3_audit_val100.json`, distance-to-next-route-maneuver quantiles: **p50 6.61 m · p75 24.65 m ·
p89 58.96 m** (n_with_distance 1251). At the canonical val median `v0` of **10.29 m/s** that is
**≈0.6 s / 2.4 s / 5.7 s**. **A 5 s horizon already reaches ~p89 of route maneuvers; 10 s reaches
almost no additional ones while halving supervision coverage.**

### 7A.3 ⭐ The strategic latent subspace — spec, and how it is VERIFIED not asserted

Sayed's efficiency argument rests on this: *"its own strategic predictor operating on a latent
subspace containing only strategy-relevant information — that compression is what makes strategic
planning efficient."* Three questions have to be answered concretely, and the third is the one that
decides whether it works.

**(i) What defines "strategy-relevant"?** An operational definition, so it can be tested:

> `z_strat` is **strategy-relevant** iff it is **SUFFICIENT** to predict the 5 s coarse direction,
> the route/exit choice and the target-speed band, while being **INVARIANT** to the exact 2 s
> waypoint and to instantaneous pose jitter.

Sufficiency without invariance is a copy of the state (no compression, no efficiency). Invariance
without sufficiency is Sayed's own named failure — *a bottleneck that discards strategy*. **Both
halves are gated.**

**(ii) How is the subspace carved?** Three mechanisms considered:

| | Mechanism | Verdict |
|---|---|---|
| 1 | **Supervised bottleneck.** `z_strat = E_strat(state)` at a chosen `d_strat`, trained **only** by the strategic objectives (route/exit CE, 5 s coarse direction, target-speed band) plus the strategic predictor's own forward consistency **in z_strat space**. Nothing operative supervises it, so it keeps only what predicts strategy; the dimension is the compression knob. | ✅ **ADOPT for v4.** No extra loss term, therefore **no extra encoder-touching lever** — which matters, because v4 is at the [PM] #6 limit (§9). |
| 2 | **Information bottleneck (VIB-style)** — an explicit KL/rate penalty on `z_strat` | ⚠️ **v4.1.** It is the principled version of "compression", but it is a *new objective that shapes the encoder* = a third encoder-touching lever. **Refused for v4 on the lever budget, not on the merits.** Fires if the §7A.3(iii) compression probe shows leakage. |
| 3 | **Time-scale separation by construction** — define `z_strat` as the linearly-5 s-predictive part of the state | ⚠️ Elegant, but it presumes linearity and forecloses the learned subspace Sayed asked for. Named, not adopted. |

**Geometry (PROPOSED, all MEASURED costs):** `E_strat` = `Linear(2048→512) → GELU → Linear(512→128)`
= **1,114,752**; `d_strat = 128`. Strategic predictor = a causal transformer **in the 128-d space**,
`d_model 256 × depth 4`, `action_dim 4` (the strategic action) = **3,919,744**. `d_strat = 128` is a
16× compression of the 2048-d state — that ratio *is* the efficiency claim, and §7A.3(iii) is what
checks it did not become a lie.

**(iii) 🔬 The verification — a two-probe information-plane test, pre-registered at 5 k**

Fit both probes on held-out episodes; compare decoding from `z_strat` (128-d) against decoding from
the **full 2048-d state** on the identical windows.

| Probe | Target | Requirement | What failure means |
|---|---|---|---|
| **SUFFICIENCY** | route/exit class · 5 s coarse direction · target-speed band | `z_strat` retains **≥ 90 %** of full-state decodability (PROPOSED) | 🟥 **The bottleneck discarded strategy** — Sayed's named risk. Widen `d_strat` or cut the layer to the fallback head (§2.6). |
| **COMPRESSION** | exact 2 s waypoint · instantaneous curvature jitter | `z_strat` decodability **≤ 0.5×** full-state (PROPOSED) | 🟥 **It is not a strategic subspace, it is a copy** — no efficiency gain, and the "compression makes planning efficient" claim is unearned. Fire mechanism 2 (VIB) in v4.1. |

**Both must hold.** This is the falsifiable form of "verified to carry strategy rather than being a
bottleneck that discards it", and it costs one CPU probe-fit per milestone.

### 7A.4 ⭐ Imagination per planner — and Sayed's central hypothesis, made falsifiable

**His claim:** *imagination is decisive specifically for mid/long-term prediction.* It is the
load-bearing belief behind the whole three-planner design, and **it has never been tested at any
horizon above 2 s in this program.** What we do know:

| MEASURED | |
|---|---|
| **consequence rollout** (v1.5: roll the predictor under probe actions, feed the imagined latents as decoder KV) | `a`→`ab` **−0.1355 m (−19.9 %), CI [0.038, 0.233] SEPARATED** — the only conditioning mechanism in the program with a positive CI-separated effect. **At 2 s.** |
| **H15 belief field** (refine encoder tokens over unobserved space) | flat at `vision_use ≈ 12 %` since 19 k |
| above 2 s | **nothing. No measurement exists.** |

**How each planner uses imagination in v4:**

| Planner | Mechanism | Cost |
|---|---|---|
| ③ operative | operative-predictor consequence rollout, 8 FPS probes × 20 steps, read at (5,10,15,20) — **the proven seam, byte-identical to v1.5** | 7/8 probes `no_grad`; 1 stochastic grad-probe (§5.4) |
| ② tactical | the **same** rollout, read at the tactical horizon (10,20,30,40,50) | shares the rollout — the probes are computed once and both decoders attend |
| ① strategic | its **own** predictor rolling `z_strat` under each strategic action — a *different, cheaper* imagination in the 128-d subspace | ~free: 128-d × ~8 actions × ~10 coarse steps |

**🔬 The pre-registered falsifier — `imagination_horizon_scaling`, run at 5 k and every milestone.**
At three horizons k ∈ {2 (0.2 s), 20 (2 s), 50 (5 s)}, compare on held-out windows:

- **(A) direct head** — the predictor's own `heads[k]` prediction of `z(t+k)` in one shot;
- **(B) imagined rollout** — recursion to the same horizon under true actions.

Define `imag_win(k) = err(A) − err(B)`, normalised by `err(A)`.

| Outcome | Reading |
|---|---|
| `imag_win` **increases monotonically** with k, and `imag_win(50) > 0` CI-separated | ✅ **Sayed's hypothesis CONFIRMED** — imagination is decisive for mid/long-term specifically, and the strategic planner's rollout is earning its place |
| `imag_win(50) ≤ 0` | 🟥 **FALSIFIED at the target horizon.** The strategic planner degrades to the produced-goal head (§2.6 fallback); rollout-based strategic planning is not supported by this corpus |
| `imag_win` **flat** across k (helps equally at 0.2 s and 5 s) | 🟥 **The specific claim is falsified** — imagination is generically useful, not *mid/long-term-decisive*. The architecture survives; the justification for the *separate* strategic predictor does not, and it collapses into (b)-style scalars |

⚠️ **This test must be run with S1/S3 zeroed and on held-out episodes**, or the planners' own
conditioning contaminates it.

### 7A.5 ⭐ The no-navigation case — how "estimate the most probable path" is trained and evaluated

Sayed called this out specifically, and it is load-bearing: *when no navigation is set it must
estimate the most probable path by evaluating options and choosing the most probable one that fits
the semantic context and our own goals.*

**Two components, deliberately separated** — this is the same split P2 proved (a proposal prior plus
imagined-rollout evaluation), and it is what keeps the strategic planner from being a head:

1. **The prior — `P(strategic action | context)`.** A small head off `z_strat` (**35,080** params
   MEASURED) over the discrete strategic-action set, supervised by the **realised** future:
   `route_from_future_v3`, masked on `unknown`. **Every training window has a known future**, so this
   is fully supervised at the label's coverage (~75 % judgeable under v3, versus the **21–25 %**
   `nav_valid_frac` every previous arm trained on — [PM] §6).
2. **The evaluation — imagined rollout in `z_strat`.** For each candidate strategic action, roll the
   strategic predictor forward and score `log P(a|context) + fit-to-goals` (efficiency/progress,
   ODD). Pick the argmax. **The choice comes from imagined consequences, not from the head's
   argmax** — the head only seeds it.

**Training discipline.** Goal-dropout ≥ 0.5 on the nav command means **half of every batch trains the
no-command path explicitly.** Goal scheduled-sampling (§5.2) ramps from GT goal to produced goal, and
**every leaderboard number is computed with the produced goal**.

**Evaluation — three reads, all pre-registered:**

| Read | Metric | Floor |
|---|---|---|
| Does it pick the right route with no command? | route accuracy **on the valid subset**, reported next to coverage, never averaged over `unknown` | **majority-class (straight) base rate** via `hierarchy.py::majority_straight_rate` — v1's `route_skill_vs_chance` was **0.0**, a pure command echo, and that is the bar to clear |
| Is the predicted path better than "keep going"? | 5 s coarse-direction error of the produced most-probable path | **persistence / CV floor** — `longh_5s_beats_persistence` (§9) |
| Does the *imagined evaluation* beat the bare prior? | argmax-of-prior vs imagined-rollout selection, paired | the prior head itself — **if imagination does not beat its own seed, the planner is a head and should be cut to one** |

That third read is the same discipline as the whole design: *a mechanism that cannot beat the thing
it replaces does not ship.*

### 7A.5b How each level is assessed — the map Sayed asked for

| Level | What it **predicts** | **Prediction-quality** axis | **Decision-quality** axis |
|---|---|---|---|
| **① Strategic** | ⭐ `z_strat(t+k)` under strategic actions, and the goal scalars read off it | 5 s coarse-direction / latent error on the **masked 74.3 %**, held-out, **vs a persistence floor** · **`imagination_horizon_scaling`** (§7A.4) · the **two-probe subspace verification** (§7A.3) | route accuracy on the valid subset **vs majority-class** · G-H `per_window_content_helps` (never `helps_vs_none`) · **imagined-selection vs prior-argmax** |
| **② Tactical** | ⭐ the 5 s coarse plan, and the imagined consequences it is conditioned on | oracle-in-fan **at the 5 s horizon** · the tactical fan's own anchor/rank accuracy on the masked subset | `fan_lateral_diversification` (§6.3.1) · whether G_t improves ③'s selection (ablate S3) |
| **③ Operative** | the 2 s dense plan; and the WM predicts `z(t+1,2,4)` + the 20-step rollout | `g_op_fwd_ade_m` (term b) · `g_op_mid_de_m` (term a) · **`speed_benefit_recovered_frac`** · the plan-free **WM canary** · step-1 speed R² · rollout commitment (`znorm`, `zcos`) | ADE@2s · miss@2m · oracle-in-fan · `sel_gap` · `frac_sel_2x_worse` · LAT/LON/DIST accuracy · L1 cruise / plan-smoothness |

**What changed versus the previous revision:** the tactical level **now predicts** (a 5 s coarse plan
and its imagined consequences), so the "🟥 nothing predicts at the tactical level" line is
**withdrawn**. Sayed's objection was correct and this is the fix.

**Two gaps that remain, named rather than papered over:**

1. **Nothing predicts above 5 s, and nothing can be supervised there** (48.2 % at 10 s, 0.0 % at 20 s).
2. **Intersection/roundabout capability stays unmeasurable, and 5 s does not fix it.** The binding
   constraint is **no map and ~20 s clips**, not the horizon: route v3 mints `roundabout` on
   **8/2201 windows (0.36 %) from ONE episode**, with 40 candidates unpromoted and 24 u-turn
   confusions. Deferred with its reason (§7A.7).

### 7A.6 LAL-v2 — yes, merge it; no, it does not fill this gap

I read the intake (`Benchmarks & Eval/Implementation/incoming/2026-07-09-lal-v2-anticipation/lal_v2.py`).

| Question | Answer |
|---|---|
| Is it the right *family*? | ✅ **Yes.** It exists precisely because LAL-v1 measured reaction *hardness*: on the first live SC-01 run the reactive baseline and the anticipating world-model policy **both scored −0.7 s — zero discrimination**, because an anticipatory policy slows *smoothly* and never trips a −1.5 m/s³ jerk threshold. LAL-v2 detects onset by **sustained speed drop vs a free-cruise reference**, so a gentle anticipatory slowdown registers. That is exactly the behaviour v4's smoothness work is meant to produce. |
| Is `LAL_v2` computable on our corpus? | 🟥 **No, not as defined.** `LAL_v2 = t_LoS − t_decel_onset` needs a **hazard line-of-sight flag**. PhysicalAI has no hazard annotation. It is a **scenario-suite (CARLA/SC-01)** metric. |
| Is its *onset half* computable? | ✅ **Yes** — `decel_onset_index` consumes only `ego_v`, and the metric suite's **L5 DECEL-ONSET LEAD TIME** (`t(GT onset) − t(planned onset)`) is built on it. It is currently **tier-1 blocked for one reason: *"the dense 20-step path (10 Hz) — the 0.5 s waypoint surface cannot resolve an onset."*** ⭐ **v4's dense 20-step anchored plan removes exactly that blocker. v4 is the first arm in the program that can be scored on L5.** |
| Does it fill the >2 s assessment gap? | ⚠️ **No, and I will not claim it does.** It measures anticipation lead time *within the plan horizon* (≤2 s) — "does the plan commit to decelerating before the log does". Real, currently unmeasured, and directly on Sayed's ask #3. But it cannot validate *"the model knows what is 5 s ahead"*. That is what the strategic level's own held-out 5 s prediction against the persistence floor is for (§7A.5b). |

> 🟥 **ESCALATION WITHDRAWN AND REPLACED (O-05) — LAL-v2 was never idle.** It merged on **2026-07-09,
> the day of the intake** (`3784e34` → `stack/tanitad/eval/metrics.py:202-251`; `metrics.py:339` emits
> `LAL_v2_s`; `stack/tests/test_lal_v2.py` green in the 686-test suite). *"Implemented and tested but
> unmerged, 12 days idle"* was **inherited from `TANITEVAL_V2_METRIC_SUITE.md` §7 E5 without checking
> `git ls-files` or grepping the named module — class C4+C2**, logged in `RETRACTION_LOG.md`, and the
> metric suite has been corrected at source. *An escalation asking for work git already contains is
> the mirror image of the stranding failure the Agent Operating Standard was written for, and it is
> the more embarrassing of the two.*
>
> 🔺 **The replacement escalation is real, one line, and still on v4's path:**
> `taniteval/taniteval/rollout.py:94` computes the dense `wp_full [b,20,2]` and persists **4 of 20
> steps**. That single discard is what actually keeps L5 — and jerk, the comfort bounds and a real
> curvature profile — at tier 1. **v4 emits a dense plan and is the first arm that can use them.**
> Requested: persist all 20 steps, promote L5. (P7(d), ~2 h.)

### 7A.7 What is deferred to v5, with the reason

| Deferred | Reason |
|---|---|
| Any **10–20 s** horizon | **0.0 % / 48.2 % supervisable** on 19.9 s clips (MEASURED). Needs longer clips or a different corpus — a **data** decision, not an architecture one. |
| **Intersection / roundabout capability** as a scored axis | No map; 0.36 % roundabout mint rate; 24 u-turn/roundabout confusions. The metric suite already refused it and the refusal stands. |
| A **horizon ladder** (k = 30 / 50 / 70) | v4 adds **one** head. The ladder is a v4.1 sweep with its own gate — N6. |
| **VTARGET as a live 10–20 s conditioning signal** | Still refused at 2 s (X3). The strategic planner predicts target speed at **5 s**, which is inside its supervisable range and is a *different quantity* from the 10–20 s aspiration. |

---

## 8. Latency

**All MEASURED**, A40, batch 1, exclusive under `gpu_lock.sh`, contamination-checked
(`Production & Optimization/Research/2026-07-20-flagship-v1-inference-levers-measured.md`; raw
`taniteval/results/eff_levers_flagship-30k.json`, `eff_levers_fan_sharedenc_flagship-30k.json`):

| Anchor | p50 | p99 |
|---|---:|---:|
| v1 eager planning tick (20-step rollout) | 100.29 | 113.98 |
| v1 **L4 = L1b graph + L2 enc-cache + L3 fp16 + L7 pruned heads** | **18.75** | **18.76** (**5.35×, 53.3 Hz**) |
| v1 **8-candidate shared-encoder fan** | **20.82** | **23.72** (marginal candidate ≈ **0.3 ms**) |
| v1 32-candidate fan | 28.41 | 30.55 |
| REF-C-base full tick fp32 (encoder 90.7 % of it) | 21.78 | 22.33 |
| REF-C-XL full tick fp32 | 44.06 | 44.44 |
| REF-C-XL decoder alone (1 classifier + 2 denoise, 256 anchors) | **8.80 ms** (classifier 3.03, 2 denoise 5.78) | — |

⚠️ **Levers are SEQUENCED, not additive — capture FIRST.** L2/L3/L7 are worth ~1.0× *before* L1 and
24/32/0.6 ms *after* it. CUDA-graph capture is **bit-exact** (max abs dev 0.0 m);
`torch.compile(reduce-overhead)` is faster (52.89) but **not** bit-identical, so **manual capture
stays the deploy default**.

**v4's open-loop tick is structurally CHEAPER than v1's**, because the shipped trajectory no longer
requires the 20-step rollout — planner ③ emits it in one shot. The rollout survives only for
(i) the 8-probe imagination conditioning, shared by ② and ③, and (ii) closed-loop CEM.

**Composing three planners from the measured parts (PROPOSED estimate, G0 measures it):**

| stage | ms | basis |
|---|---:|---|
| encoder + 8-probe shared-encoder imagination fan | **20.82** p50 / **23.72** p99 | **MEASURED** (`eff_levers_fan_sharedenc_flagship-30k.json`); marginal candidate ≈ **0.3 ms** |
| ① strategic planner — 8 strategic actions × ~10 coarse steps **in 128-d** | **< 1** | PROPOSED. A 128-d 4-layer transformer is ~3 orders below the 768-d trunk; the whole point of the subspace |
| ② tactical decoder (d384×4L, 256 anchors, 2 denoise) | **~1.7** | **MEASURED** by proxy: REF-C-base's decoder is ~1.7 ms of its 21.8 ms tick (encoder 90.7 %) |
| ③ operative decoder (same geometry, dense) | **~1.7** | same |
| **v4 composed tick** | **≈ 25–28 p50 / ~30 p99** | **PROPOSED** — inside the 50 ms budget with ~1.7× headroom |

> ⚠️ **The row above is built on a 20-step probe rollout and v4 runs 50 (§14.3 O-10).** The encoder
> term is unaffected (it is one forward, shared), but the imagination term scales with probe length,
> so **≈25–28 ms is a floor, not an estimate.** It is not re-derived here — stage arithmetic
> over-estimates when the encoder hides behind rollout launches (the retracted *"encoder caching →
> 84.74 ms"* read **95.11** measured, class C5). **G0 measures the real tick; `deploy_tick_p99_ms ≤ 50`
> stays a KILL secondary and the probe count is the first thing cut if it misses.**

**The second planner costs ~1.7 ms and the strategic planner costs under 1 ms.** The tick is
dominated by the encoder and the imagination fan, both of which are already measured and both of
which are **shared** across the planners — which is the latency counterpart of §3.1's parameter
finding. **Latency is reported in every gate table from G1 onward and is a pre-registered secondary.**

---

## 9. The gate ladder

**Pre-registered card, written BEFORE launch (`GATE_PROTOCOL.md` §1, `stack/scripts/run_gate.py`):**

> 🔴 **O-03 — THE CARD BELOW REPLACES AN EARLIER ONE THAT WOULD NOT HAVE RUN, AND WOULD HAVE KILLED
> THE ARM ON A TECHNICALITY. Read this before registering anything.** Two source reads of
> `stack/scripts/run_gate.py`, both MEASURED:
>
> 1. **`_parse_secondary` (`run_gate.py:410-415`) accepts only `>=`, `<=`, `>`, `<`.** The previous
>    card carried `longh_5s_beats_persistence=1` and `nonav_route_beats_majority=1`; `register`
>    would have exited with `[gate] bad --secondary`. Booleans are now written `>=1`.
> 2. ⚠️ **`cmd_check` ANDs every secondary (`run_gate.py:549 sec_ok &= p`) and a single FAIL yields
>    verdict `RESTART` — or `REFUTE_LEVER_FAMILY` once the budget is out.** The earlier card listed
>    **14** secondaries, four of which this document *elsewhere explicitly says must not kill the run*
>    (§7A.3(iii), §7A.4, §9.1: *"Failing it does not kill the run: the strategic layer degrades to the
>    produced-goal head"*). **Registering a falsifier you have already decided not to act on is not
>    pre-registration — it is a trapdoor**, and with `restart-cap 2` a third trip would have
>    `REFUTE_LEVER_FAMILY`-ed *"jointly training a world model and an anchored planner"*, which is
>    v4's entire thesis, on the strength of an unbuilt CPU probe.
>
> **The card is therefore split in two, and the split is itself pre-registered.** KILL secondaries go
> on the card: each one means *the arm is damaged*. REPORT-ONLY falsifiers stay in this document with
> their pre-committed in-run fallbacks (they are emitted into the eval JSON and read at the gate, they
> just do not adjudicate it). Nothing is deleted; the consequence of each read is stated in advance.

```bash
# --- KILL secondaries: each one means the ARM IS DAMAGED. 8 of them. -------------
python stack/scripts/run_gate.py register --run flagship-v4 --gate-step 10000 \
  --primary-metric ade_0_2s --primary-threshold 0.60 --primary-direction "<=" \
  --primary-source "held-out taniteval full-set, planner path, produced goal" \
  --secondary "wm_canary_ade_2s<=0.55" \
  --secondary "speed_benefit_recovered_frac>=0.70" \
  --secondary "oracle_in_fan<=0.30" \
  --secondary "miss_at_2m<=0.10" \
  --secondary "seam_norm_ratio_max<=1.0" \
  --secondary "encoder_touching_levers<=2" \
  --secondary "deploy_tick_p99_ms<=50" \
  --secondary "nonav_route_beats_majority>=1" \
  --reference-run flagship-v1 \
  --reference-log taniteval/results/trainlogs/v1-speedjerk_train_log.jsonl \
  --compare-metric g_op_fwd_ade_m --tau 1.5 \
  --lever-family joint-planner-wm --restarts-used 0 --restart-cap 2 \
  --card "Project Steering/Gates/flagship-v4.card.json"
```

| Secondary | KILL or REPORT | Why, and the pre-committed consequence if it fails |
|---|---|---|
| `wm_canary_ade_2s<=0.55` | **KILL** | the WM is gone — v4's whole premise. R1. |
| `speed_benefit_recovered_frac>=0.70` | **KILL** | the quiet plateau [PM] found in v3enc; the canary alone does not catch it. |
| `oracle_in_fan<=0.30` | **KILL** | joint training proposes worse than a *frozen* trunk (v1.5-`ab` 0.3073). |
| `miss_at_2m<=0.10` | **KILL** | R2, the tail imported with REF-C's decoder. |
| `seam_norm_ratio_max<=1.0` | **KILL** | it is an in-graph clamp; a breach means the clamp is broken, i.e. a code fault. |
| `encoder_touching_levers<=2` | **KILL** | constant, checked at registration. Free to carry; catches a mid-run scope creep. |
| `deploy_tick_p99_ms<=50` | **KILL** | the arm is undeployable. |
| `nonav_route_beats_majority>=1` | **KILL** | v1's `route_skill_vs_chance` was **0.0** — a pure command echo. An arm whose *produced* goal cannot beat "always straight" has not built a strategic level at all, it has relabelled one. This is the one hierarchy read that is genuinely pass/fail. |
| `strat_subspace_sufficiency>=0.90` | **REPORT** | §7A.3(iii). Fail ⇒ **widen `d_strat`, or `--strategic head`** for the rest of the run, recorded. Not a restart. |
| `strat_subspace_compression<=0.50` | **REPORT** | §7A.3(iii). Fail ⇒ the compression claim is unearned; fire the VIB penalty in v4.1 (X19). Not a restart. |
| `imag_win_at_5s>0` | **REPORT** | §7A.4 — **Sayed's central hypothesis**. Fail ⇒ reported *falsified on this corpus*, strategic layer degrades to `--strategic head`. **A first-class result either way, and never a reason to restart.** |
| `longh_5s_beats_persistence>=1` | **REPORT** | §9.1. Fail ⇒ the 5 s predictor is not predicting; report it and cut to the head. Not a restart. |
| `cruise_delta_vs_holdv0>=-0.1443` | **REPORT at G1, KILL at G3** | it is a *capability* read one third of the way through training; no arm in 14 has ever passed its G3 form. Killing v4 at 10 k for not yet clearing a program-first bar would be an own goal. G3 keeps it hard (§9 rung G3). |

**Net: 8 kill secondaries instead of 14, and 5 falsifiers whose consequence is a documented in-run
degrade rather than a burnt restart.** The information content is identical; the failure surface is
40 % smaller. All 13 are supplied to `run_gate.py check --secondary-value` and land in the gate JSON;
only the 8 adjudicate.

**⭐ [PM] DO-NOT-CARRY #6 is baked in as a gate item — the lever-count limit.** *"More than ~2
encoder-touching levers per arm"* is named as **the actual repeat root cause**: v2 fired 12 at once
and died; v3enc softened 4 and plateaued; **neither run can attribute its own failure.** Counted by
the post-mortem's own taxonomy (§3 lever table, class `encoder-grounding`):

| v4 change | class | encoder-touching? |
|---|---|---|
| λ_plan — planner gradient (② + ③) reaches the encoder | encoder-grounding | ✅ **1** |
| the strategic layer ① — `E_strat` + its predictor, whose objectives shape the trunk | encoder-grounding | ✅ **2** |
| two diffusion instances instead of one | decode-side — both sit under the **same** λ_plan lever, and [PM] §4.6 exonerates the anchored decoder | no |
| factorised LAT/LON/DIST · dense anchors · plan-smoothness · S1/S3 | decode-side | no |
| aux-accel + v1 jerk removed | removal, per [PM] #7 | no |
| **TOTAL** | | **2 of a limit of 2 — the door is CLOSED** |

⚠️ **v4 is AT the limit, not under it.** Nothing further may be added to this arm. That is why every
other change in this design is deliberately decode-side. **Attributability is provided by switches,
which is what the rule actually requires:** `--lambda-plan 0` (= Phase A) isolates lever 1,
`--strategic {full,head,off}` isolates lever 2, and each reproduces its sub-configuration
byte-identically. **Unlike v2 (12 levers, no controls) and v3enc (4 levers, no controls), this arm can
attribute its own result.** If Sayed prefers strict serialisation, §2.6 gives the two-arm split.

> 🔧 **CLOSED 2026-07-21 (O-01).** An orphaned duplicate of the lever table used to sit here and
> totalled **"1 of a limit of 2"**, contradicting the table above it. **The count is 2 of 2** —
> λ_plan and the strategic layer — and the duplicate has been deleted. Every other statement of the
> count in this document (§0, §9.1, §12.1 #6) has been reconciled to **2**. *The lower number was the
> more dangerous of the two: it implied headroom that does not exist.*

**And the v1-identical control [PM] demands costs zero GPU-days:** the 10 k gate reports matched-step
`g_op_fwd_ade_m` against **both** v1's and the no-speed control's archived logs (the
`speed_benefit_recovered_frac` secondary, §7.5). Phase A (λ_plan = 0, steps 0–2 k) is the in-run
control for the planner-gradient lever itself.

⚠️ **Tool defect to route around.** [PM] §7.1 shows `run_gate.py`'s `reference_reached_at` is
noise-driven — its "v1 reached v3enc's value at step 450 ⇒ 22.7×" is an artifact of a 3-point rolling
median over a series that swings 2× between adjacent rows (v1 raw at 300–550: 0.758, 0.616, 0.404,
0.687, 0.384, 0.816); the defensible figure is **~3.5–5×** and it has already propagated into
REGISTRY §1.4. **v4's gate must decide on the matched-step ratio and the bucket means, never on
`reference_reached_at`.**

| Rung | What runs | Threshold | Basis | A40-day cost |
|---|---|---|---|---|
| **G0 — preflight** (pod1, ~30 min) | 200 steps: param count vs §3.1, step-0 canary baseline, seam-norm telemetry alive, **s/step pace**, `pytest -q` green | pace ≤ **16 s/step** or `--probe-grad` drops to `none` and it is recorded | **PROPOSED** (pace ceiling derived from the cost model below) | **0.02** |
| **G1 — THE gate** (step 10,000) | full canonical panel on the archived milestone | primary + all six secondaries above | see the threshold table | **1.66** cumulative |
| **G2 — diagnostics** (5 k / 15 k / 20 k) | same panel, **no verdict** — `run_gate.py check` returns `NOT_YET` by design | — | one gate step only; no garden of forking paths | included |
| **G-H — causality** (5 k and every milestone, eval pod) | sensitivity + direction-correctness + `per_window_content_helps` + effect floor + norm parity, on the **scoped lateral-confusion panel** | CI-separated **and** ≥0.02 acc / 0.05 m / 0.01 cos | `hierarchy.py` MIN_* constants | ~0.1 eval-day each |
| **G3 — completion** (step 30,000) | full panel + strata + OOD triple + driving panel | **ADE ≤ 0.4271 full-set** (beat v1) · **miss@2m ≤ 0.10** · **high-speed ≤ 0.45** · **`cruise_speed_vs_holdv0` CI includes zero or favours the model** · canary ≤ +0.05 m · tick p99 ≤ 50 ms | pins below | **4.96** cumulative |
| **G4 — closed loop** (eval pod) | P2 CEM over v4's own WM, **warm-started from v4's decoder top-K** instead of the 16-seed constant-action grid | closed ADE < **1.038** and divergence < **8.7 %** (P2-over-v1's MEASURED values) | REGISTRY §5 | ~0.2 eval-day |
| **G5 — registry + HF** | registry §1.7 row with `lineage:`, leaderboard row, paper §7 | — | — | ~0.3 eval-day |

### 9.1 Threshold provenance — every one of them

| Threshold | Value | MEASURED / PROPOSED |
|---|---|---|
| G1 primary `ade_0_2s` ≤ 0.60 | **PROPOSED.** Derivation stated, not hidden: strictly below hold-v0 **0.7876** and CV **0.8377** (MEASURED floors, `LEADERBOARD.md` §0) by a clear margin, and above v1's 30 k full-set **0.4271** to allow for being ⅓ through training. We do not have v1's own 10 k held-out eval, so this is a catastrophe-catcher; the *comparative* judgement is the matched-step ratio the tool computes (τ = 1.5). |
| G1 `wm_canary_ade_2s` ≤ 0.55 | **PROPOSED**, anchored on MEASURED values: v1's canary **0.452**; v1.6's collapse **1.1022**. 0.55 is +0.10 on the baseline — ~22 % — well inside the corridor between "healthy" and "collapsed". |
| G1 `speed_benefit_recovered_frac` ≥ 0.70 | **PROPOSED**, anchored on MEASURED: v1 **81.8 %** at the 8–10 k bucket, v3enc **18.6 %**. Adopted verbatim from [PM] §8 *"KEEP AS A GATE, FIX THE ESTIMATOR"*. Both reference logs exist; costs zero GPU. |
| G1 `longh_5s_beats_persistence` = 1 | **PROPOSED, and deliberately a floor test rather than a level.** The strategic level's held-out 5 s prediction, on the masked 74.3 %, must beat the **persistence floor** `ẑ = z_t`, CI-separated (episode-cluster bootstrap). *A long-horizon predictor that cannot beat "nothing changes" is not predicting, and should be reported as such and cut* — that is the failure mode `tactical_pred` was never tested for. Reported alongside coverage, always. |
| G1 `strat_subspace_sufficiency` ≥ 0.90 · `strat_subspace_compression` ≤ 0.50 | **PROPOSED, the two-probe information-plane test of §7A.3(iii).** Sufficiency: `z_strat` retains ≥ 90 % of full-state decodability for route/5 s-direction/speed-band. Compression: ≤ 50 % of full-state decodability for the exact 2 s waypoint. **Both must hold** — sufficiency alone means no compression (no efficiency claim), compression alone means the bottleneck discarded strategy (Sayed's own named risk). CPU probe-fit, no GPU. |
| G1 `imag_win_at_5s` > 0 | **PROPOSED — Sayed's central hypothesis as a gate.** Imagined rollout must beat the direct head at 5 s, CI-separated, with `imag_win` increasing across k ∈ {2, 20, 50} (§7A.4). Failing it does **not** kill the run: the strategic layer degrades to the produced-goal head (§2.6 fallback) and *"imagination is decisive for mid/long-term"* is reported **falsified on this corpus** — which is itself a first-class result. |
| G1 `nonav_route_beats_majority` = 1 | **MEASURED-anchored pin.** With the command withheld, route accuracy on the **valid subset** must beat the majority-class (straight) base rate, CI-separated (`hierarchy.py::majority_straight_rate`). v1's `route_skill_vs_chance` is **0.0** — a pure command echo — and that is exactly the bar the no-nav planner must clear to be more than a relabelled echo. |
| G1 `encoder_touching_levers` ≤ 2 | **[PM] DO-NOT-CARRY #6**, a design-audit item checked at registration and re-checked at the gate. **v4 stands at 2 of 2 — the door is CLOSED** (O-01: the earlier "stands at 1" reading came from a duplicated table and is withdrawn; the strict count is the binding one). Attributability is provided by `--lambda-plan 0` and `--strategic {full,head,off}`, each reproducing its sub-configuration byte-identically. |
| G1 `oracle_in_fan` ≤ 0.30 | **MEASURED pin.** v1.5-`ab`'s frozen-trunk fan is **0.3073**. Joint training must not propose *worse* than a frozen trunk. (REF-C's 0.1640 is the aspiration, not the gate — the fan is not the ship metric, §10.) |
| G1/G3 `miss_at_2m` ≤ 0.10 | **MEASURED pin.** v1 **0.0602 ± 0.0121** (heldout, deprecated estimator — label carried); REF-C-XL **0.1419** full-set. This is the tail-import risk (§10 R2) made into a gate. |
| G1 `cruise_delta_vs_holdv0` ≥ −0.1443 | **MEASURED pin** — the *favourable* end of v1's own CI (**−0.2122 [−0.2778, −0.1443]**). v4 must be at least as good at 10 k as the best v1 could plausibly be. |
| G3 `cruise_speed_vs_holdv0` CI includes zero or favours | **MEASURED-anchored, program-first.** All 14 arms currently fail it. |
| G3 high-speed ≤ 0.45 | **MEASURED pins:** v1 **0.5513**, REF-C-XL **0.3243**, REF-C-base 0.3510, CV 0.6468 (n=294). 0.45 sits between the two arms — **PROPOSED** as the midpoint. |
| seam norm ratio ≤ 1.0, fail-loud at 1.5 | **MEASURED-motivated:** the ROUTE seam fired at **2.80×** with `rt_gate` at 0.10; v1's `intent_proj` at **31.4** vs act-emb **28.3**; REF-A's at **1792** vs **14.5**. |
| tick p99 ≤ 50 ms | **PROPOSED** (half the 10 Hz control budget). MEASURED context in §8. |

### 9.2 Cost model — ✅ the pace conflict is now RESOLVED (O-11)

> ✅ **SETTLED 2026-07-21 from the primary log, which is now in the repo.** `sum(step_s)` over
> `taniteval/results/trainlogs/v1-speedjerk_train_log.jsonl` = **326,638.2 s = 90.73 h / 29,999
> steps = 10.888 s/step**, reproducing `GATE_PROTOCOL.md`'s **10.89**. `summary.json`'s
> `wallclock_s 191206.2` (⇒ 6.37 s/step) is **refuted**: in-loop step time is a subset of wallclock,
> so 90.73 h of logged step time cannot sit inside a 53.1 h wallclock. **Use 10.89. The v4 estimate
> below already did, so nothing in the budget changes** — it is simply no longer resting on a
> coin-flip between two primary sources. The historical conflict is preserved below as provenance.

⚠️ *(Historical — the conflict as it stood before O-11.)* **The program had two incompatible figures
for v1's training pace:**

- `summary.json` for `flagship4b-speedjerk-30k`: `wallclock_s 191206.2` for 30,000 steps ⇒
  **6.37 s/step = 53.1 A40-hours = 2.21 A40-days** (REGISTRY §1.2).
- `GATE_PROTOCOL.md` §4: *"v3enc 10.22 s/step vs v1 **10.89 s/step** → 352 vs 331 steps/GPU-hour"*
  ⇒ 30 k = 90.6 h = **3.78 A40-days**.

**[PM] §3 independently confirms the conflict** (*"contradicts its own log by 1.7×. Flagged
UNRESOLVED"*) and adds the third data point: v3enc **10.37 s/step** at `needed_fut` 16 vs v1's
10.89 s/step at `needed_fut` 10 — *"equal-step ≈ equal-wallclock here, but equal-step is not
equal-FLOP."* v4 runs at `rollout_k = 4` ⇒ `needed_fut` **10** ⇒ **v1's FLOP profile, not v3enc's.**

**I cost with the conservative figure (10.89 s/step)** and flag the conflict for the registry owner.
v4 adds the planner forward+backward (PROPOSED **+5 %**) and one grad-probe 20-step recursion
(PROPOSED **+25 %**) ⇒ **~14.3 s/step ⇒ 10 k = 39.7 h (1.66 A40-days) · 30 k = 119 h (4.96 A40-days)**.
**G0 measures this and the pre-registered pace ceiling of 16 s/step is what makes the estimate
falsifiable rather than decorative.**

### 9.3 Kill criterion

Per `GATE_PROTOCOL.md`: at step 10,000 exactly, `run_gate.py check` returns
`CONTINUE` / `RESTART` / `REFUTE_LEVER_FAMILY` / `NOT_YET` / `BLOCKED`. Lever family
**`joint-planner-wm`**, restarts used **0**, cap **2**. **A third failure refutes the family — i.e.
it refutes "jointly training a world model and an anchored planner in one stack" — and it does not
license more schedule tuning.** That is the honest price of asking the question.

Before then, nothing kills the run: the canary reduces λ_plan (§5.5); it does not stop training.

---

## 10. What v4 explicitly does NOT do

| # | Excluded | The measurement that rules it out |
|---|---|---|
| X1 | **Any post-hoc re-scorer / re-ranker of a fixed fan** | v1.0 hand-written cost recovered **0.0 %** (best blend point λ=0; pure cost −171 %); v1.2 learned re-scorer **+2.9 %, NOT significant** (paired Δ +0.00893, CI [−0.0062, +0.0250]) across **47 trained arms**; **~92 % of the oracle gap is aleatoric.** REGISTRY §4.1. *(The LON graft is **not** this: it is a **missing input** to the anchor logits, trained in-graph and jointly — a different object, and that door is open.)* |
| X2 | **Ranking on unsupervised denoise-time confidence** | Selecting on REF-C's discarded refined confidence scores **1.36593 = 2.9× worse than baseline**. v4 ranks on refined confidence **only because** `v15_losses` supervises `sel_score` at the denoise timesteps. Registry rule N2. |
| X3 | **VTARGET as a 2 s ranking/selection signal** | GT-perfect speed-matcher **1.1236** vs baseline 0.4714; VTARGET sits **+1.42 m/s** above v0, MAE 1.65 vs hold-v0's 0.475; braking windows **+0.51 m worse**. Right quantity, wrong timescale. |
| X4 | **Encoder widening / a second encoder** | REF-C base vs XL: 2.2× encoder, paired Δ **+0.0013 [−0.0281, +0.0316]**, not separated. |
| X5 | **Decoder widening** | base's 8.6 M ties XL's 22.7 M and is ≥ XL at every matched K; the program's largest decoder (v1.5, **30,979,852** measured) is its worst proposer (oracle 0.3073). |
| X6 | **The whole v2/v3enc encoder-grounding pack** — `v2_ego_dropout`, `v2_fa_dropout ≥ 0.15`, `v2_invdyn_gradscale < 1.0`, `v2_encoder_ego_decorr`, `rollout_k > 4`, `v2_ego_to_planners` | v3enc 10 k gate **RESTART** on `encoder_speed_probe_r2` **0.393 < 0.55** (v1 0.861); family at **1/2 restarts**. **[PM] attributes it precisely** (DO-NOT-CARRY #1–#5): 🥇 `v2_ego_dropout` zero-fills the v0 **action** channel (§5.3) — `inv` 0.3784 vs the *no-speed control's* 0.3644 vs v1's 0.2194; 🥈 the rollout pack (`rollout_k` 4→8→12 + `fa_dropout` 0.15) makes the conditioning unreliable on ~40 % of samples ⇒ damped imagination; 🥉 `invdyn_gradscale` 0.5 is *designed* to decouple the ego probe from the trunk (monotone dose-response 1.0/0.5/0.25 → +0 %/+15 %/+38 % on `g_str_mid_de_m`) — real but contributory, not causal; and **`decorr` was inert for its entire measured life** (`decorr_w = 0.0` for step < 10 k), so it is unattributable in *either* direction and must not be inherited as "known good". ⚠️ Note also: **v3enc PLATEAUED (flat 0.43–0.48 from ~4.5 k), it did not diverge like v2** — a categorically different failure. **v4 uses none of these.** |
| X6b | **The label confound as an explanation for anything** | 🟥 **REFUTED [PM] §6**, by a route the gate did not have: `nav_valid_frac` is **0.21–0.25 in all four arms, including the deployed v1** that scores probe 0.861. The coverage defect is program-wide and pre-dates v3enc, so it cannot differentiate arms on any metric. *(What it still invalidates — every route/strategic reading from v3enc — is unchanged.)* |
| X15 | **Zero-filling ANY channel as a dropout mechanism** | The measured root cause of v3enc (§5.3). Binding across the whole design: masks are learned null rows or explicit validity flags, never zeros on channels whose zero is in-distribution. |
| X7 | **Unfreezing-under-a-planning-loss-only** (the v1.6 recipe) | ADE tie (**+0.0104 [−0.0888, +0.1147]**) for a **+144 %** canary. |
| X8 | **Additive-into-a-conditioning-vector seams** (incl. V35's S2 as specified) | 0 for 4 measured (§2.4). S2 survives only as a default-OFF pre-registered A/B. |
| X9 | **Lead-referenced LONMODE** (`follow_lead`/`close_gap`/`open_gap`) and headway/TTC costs | **`lead_state` is a `None` stub.** No detector, no monodepth, no VLM lead pass. Their logits are excluded from the head, not merely unweighted. |
| X10 | **`TACPOINT` / `stop_line` naming, and map-dependent ROUTE tokens** | Kinematics mints *where*, never *why*; **4 of 9 ROUTE tokens** (`straight`, `exit_left`, `exit_right`, `merge`) are unmintable without a map. |
| X11 | **VLM labels in training** | 595 records = **0.15 %** of 406,099 train windows; direction at chance (**57.1 %**, CI [0.400,0.745]); the left bias is the model's, not the prompt's (right-turn recall **bit-identical 0.2069** across enum orders). |
| X12 | **Widening the anchor vocabulary as a v4 lever** | ⚠️ **Reason CHANGED (§2.7).** *Not* because the fan is longitudinal — that scoping is retracted as confounded (§6.3). Because anchors cost **0 params**, so width is not a budget question, and an **eval-side K-sweep answers it for zero GPU-days**. Hold at 256 for bit-comparability with REF-C/v1.5. |
| X13 | **Stratified / lateral-rebalanced anchors in v4** | Would break bit-comparability with REF-C and v1.5 (base's 128 are a **bit-exact prefix** of XL's 256, `max|A − B[:128]| = 0`). ⚠️ **Now a LIVE v4.1 lever, conditional on §6.3.1**: if the fan-diversification test shows the lateral collapse was conditioning-driven, this is the first thing to try. |
| X14 | **Predictor / encoder capacity increase inside v4** | Unmeasured axis, and v4 is already at the [PM] #6 lever limit. Priced in §3.4 as a *separate* arm. |
| X16 | **A map-dependent strategic action set** (roundabout exits, merges, lane graph) | `roundabout` mints on **8/2201 windows (0.36 %) from ONE episode**, with 40 candidates unpromoted and 24 u-turn confusions. **v5, gated on a map** (§2.2). |
| X17 | **Agent-interaction tactics** ("wait for the vehicles to pass", "indicate intention", gap acceptance) | **`lead_state` is a `None` stub** — no boxes, no tracks, no depth. Sayed named these as the weaknesses of monolithic E2E systems and he is right, but they are **unrepresentable on this corpus**. **v5, gated on detections/tracks/monodepth** (§2.2). |
| X18 | **A 10–20 s strategic horizon in v4** | **48.2 % / 0.0 %** supervisable (MEASURED, §7A.2). A horizon we cannot supervise is a horizon we cannot verify. **v5, gated on longer clips.** |
| X19 | **An explicit information-bottleneck (VIB) penalty on `z_strat`** | It is the principled form of the compression Sayed wants, but it is a *third* encoder-touching lever and v4 is at the limit of 2. **v4.1, fires if the §7A.3(iii) compression probe shows leakage.** Refused on the lever budget, not on the merits. |

---

## 11. Risk register — the three highest, and what falsifies each

| # | Risk | Early-warning metric (when) | What would falsify my position | Fallback |
|---|---|---|---|---|
| **R1** | **Joint training destroys the world model anyway.** My claim is that v1.6 collapsed because it deleted the WM loss, not because joint training is impossible. That claim has **never been tested** — this is the single largest unknown in the design. **[PM] sharpens the shape of the danger**: v3enc's failure was *not* an encoder failure but a **rollout** failure that compounded from a healthy 0.95-R² first step, and it **plateaued rather than diverged** — i.e. the damage class v4 must watch for is a quiet, flat, under-committed dynamics model, not a blow-up. | `canary_vs_base` every 500 steps from step 0 (the controller at +0.05 m, the gate at 10 k) **AND `speed_benefit_recovered_frac` at every 2 k bucket** — the latter is the one that catches a *plateau* | The canary rising past +0.05 m **while λ_plan is at or below its scheduled value**; or `speed_benefit_recovered_frac` flat below 0.70 across three consecutive buckets while the canary looks fine. | λ_plan capped at its last healthy value for the rest of the run; if that does not hold it, freeze the encoder and let only the predictor move (LP-FT fallback); ultimate fallback: v4 ships as *planner-over-frozen-v1*, i.e. v1.5 with the v4 head, at ~0.5 A40-day. |
| **R2** | **The tail arrives with REF-C's decoder.** v1 is the program's tail king (miss@2m **0.0602 ± 0.0121** heldout); REF-C-XL is **0.1419** full-set — 2.4×. v4 adopts REF-C's decoder as its emitter. | miss@2m at every milestone, **hard gate ≤ 0.10** at G1 and G3; straight-stratum **median AND mean** tracked separately (REF-C's straight *median* is better, 0.219 vs 0.347, while its tail is worse) | miss@2m > 0.10 on the planner path at 10 k while the WM-rollout path on the same checkpoint passes ⇒ the decoder, not the trunk, owns the tail. | **Deploy contract, pre-declared:** the shipped trajectory stays the grounded rollout unless the planner path beats it on **both** mean and miss. The planner then serves proposals + the closed-loop warm start (G4), which is where its value is anyway. |
| **R3** | **The hierarchy is decorative** — seams pass norm parity and carry no per-window content. This is v1's *measured* failure mode (`per_window_content_helps ≈ 0` on both arms while `helps_vs_none` flipped sign by encoder). ⭐ **§7A identifies a candidate cause nobody had considered: the strategic level has no timescale of its own** (`str_h = 2.0 s` in every arm ever trained), which would make the hierarchy decorative **by construction** — no seam fix would repair it. **Planner ① is the mitigation, and it converts R3 from an unexplained risk into a testable one.** | **G-H at 5 k**, not 30 k · **plus `longh_5s_beats_persistence` at the 10 k gate** — the two together separate "the seam is broken" from "there was nothing on the other side of it" | `per_window_content_helps` inside the effect floor at 15 k **with healthy norms AND a 5 s predictor that beats persistence** — that combination would refute the timescale hypothesis and point back at mechanism | Change **mechanism, not scale**: S1 KV → the S2 A/B, or make the operative attend to the plan explicitly. If the 5 s prediction *fails* persistence, degrade to `--strategic head` (§2.6) and report the timescale hypothesis as refuted. If content is still inert at 15 k with both healthy, **report H26 as falsified in this architecture** rather than shipping wiring that does nothing. |

*(R4 latency: §8 — the levers are measured and the budget has 5.3× headroom on the v1 path; the first
thing cut if the tick misses is probe count 8→4→2, never the encoder. R5 ops: standing — milestone +
atomic checkpoints, RAM guard, `supervise_run.sh` auto-resume, and pod disk judged by a real `dd`
write test, never `df`.)*

---

## 12. Dependencies, preconditions and escalations

### 12.1 ✅ RESOLVED — the v3enc post-mortem landed, item by item

`Research/2026-07-21-flagship-v3enc-postmortem.md` (staged). Every DO-NOT-CARRY item, and what v4 does:

| # | DO-NOT-CARRY | v4 |
|---|---|---|
| 1 | `v2_ego_dropout` as implemented (zero-fill of the v0 action channel) | ✅ **not used** — and generalised into the binding zero-fill rule (§5.3, X15), which also **changes the planner's own ego-dropout to a learned null row**. This is the item that most improved the design. |
| 2 | `rollout_k > 4` before the rollout is healthy | ✅ **held at 4**; raising it is a v4.1 lever unlocked only by `speed_benefit_recovered_frac` (§7.2, §7.5) |
| 3 | `v2_fa_dropout ≥ 0.15` | ✅ **not used** (v1 recipe: absent) |
| 4 | `v2_invdyn_gradscale < 1.0` | ✅ **held at 1.0** (v1 default) |
| 5 | `v2_encoder_ego_decorr` as "already validated" | ✅ **not used**; and v4 does not treat it as refuted either — it was inert, i.e. unattributable both ways |
| 6 | **> ~2 encoder-touching levers per arm** | ✅ **v4 stands at 2 of 2 — AT the limit, door CLOSED** (O-01 reconciliation; the earlier "1" was a duplicated-table artefact). The count is a pre-registered gate secondary with the v1-identical control attached (§9). **Nothing further may be added to this arm** — including anything an open-item closure might have wanted (§14 O-13, O-14). |
| 7 | restoring `--aux-accel` / `--jerk-weight` | ✅ **both dropped**; smoothness moves to the dense emitted plan (§4.3, §7.1) |
| 8 | the "v1 was ~23× more step-efficient" framing | ✅ not used anywhere; and v4's gate is instructed not to decide on `reference_reached_at` (§9) |

| ✅ DO CARRY | v4 |
|---|---|
| Staging itself (v3enc beat v2 at every matched bucket, 21–30 % on `g_op_fwd_ade_m`) | ✅ the three-phase λ_plan curriculum (§5.2) is exactly this |
| `nav_dropout` / an echo-killer — the only lever in the pack with a clean isolated positive read | ✅ stronger form: the goal is **produced, never fed** (§6.4), plus goal-dropout 0.5 |
| **The anchored tactical decoder — mechanically healthy** (`n_modes` 1→13, `conf_norm` 40→189, `wta` 0.253→0.034, `man_acc` at v1's level; level ordering same as v1's) | ✅ ⭐ **this resolves the one dependency that could have re-opened §2.** v4's central component is exonerated by the arm that failed. *(Its ADE contribution remains UNMEASURED — v4 is the first arm that will measure it.)* |
| `speed_input` at full strength, undropped | ✅ §5.3 |

### 12.2 Preconditions before G0 (none are optional)

| # | Precondition | Owner | State |
|---|---|---|---|
| P1 | **Build the v4 trainer + model.** `stack/tanitad/models/flagship_v4.py` (extend `flagship_v15.py`: dense horizons, LAT/LON/DIST heads + grafts, S1 goal-KV, S3 plan→action, λ_plan) and `stack/scripts/train_flagship_v4.py` (merge `train_flagship_v16.py`'s canary + LP-FT plumbing with `train_flagship4b.py`'s WM loss stack). **PROPOSED estimate: 2–3 agent-days.** v4 is not a "press go tomorrow" run. | eng | ❌ not started |
| P2 | **Dense anchor vocabulary**: `build_refc_anchors.py` at horizons 1…20, 256 FPS anchors, same source/pool-cap/seed. **Parity-safe — it re-samples trajectories, never episodes.** | eng | ❌ |
| P3 | **Add step 10,000 to the milestone archive list.** D-032 archives 5k/15k/20k/30k; the pre-registered gate is at 10 k and `run_gate.py check` needs an archived checkpoint there. (This exact gap bit v3enc: *"10 k was NOT on D-032's archive list — this is the only 10 k state that will ever exist."*) | eng | ❌ |
| P4 | ~~Copy v1's `train_log.jsonl` to pod1~~ → ✅ **CLOSED (O-04).** All four reference logs are **in the repo and git-tracked**: `taniteval/results/trainlogs/{v1-speedjerk,nospeed-phase0,v2,v3enc}_{train_log.jsonl,config.json}` (MEASURED: `git ls-files taniteval/results/trainlogs/` returns 8 files; v1 584,276 B, no-speed 403,763 B). The gate's `--reference-log` now points at a repo path, not a pod path. | ops | ✅ |
| P5 | **Fix the evaluator's `nav_cmd`**: v4's eval path must feed the **produced** goal, not `None → follow` (`refc_eval.py:78`, `plan_fan.py:549`). | eng | ❌ |
| P5b | **[PM] Replace the planner's zero-fill ego-dropout with a learned null-embedding row** (`flagship_v15.py:348-351`). Small, but it is the exact bug class that cost v3enc its arm, and leaving it in would mean v4 shipped the root cause it just documented. | eng | ❌ |
| P5c | **Wire the `speed_benefit_recovered_frac` secondary.** ✅ **Data dependency CLOSED (O-04)** — both logs are in-repo. What remains is code: a `run_gate.py`-callable reducer that computes `(nospeed − arm) / nospeed` on matched-step **bucket means** of `g_op_fwd_ade_m`. ⭐ **It already exists in prototype**: `taniteval/postmortem_a_analyze.py` implements the exact bucket convention (`lo < step <= hi`) and *pins* it — it reproduces the post-mortem's published 0–2 k row (v1 0.6458 / v3enc 1.0364 / no-speed 1.3152) and 8–10 k row (v1 0.1062 / v3enc 0.4699 / no-speed 0.5740). **Promote that reducer; do not re-derive the convention** (four candidates exist and only one reproduces the tables). | ops + eng | 🟡 code only |
| P5d | **Build the strategic planner ①** (§7A.3–7A.5): `E_strat` 2048→128, the 128-d strategic predictor, the discrete strategic-action set, the option prior + imagined-rollout evaluator, goal scalars, the norm-capped `z_strat`→KV projection, **per-sample valid masking at 5 s** (never window dropping — parity), and the `--strategic {full,head,off}` switch that reproduces each sub-configuration byte-identically. | eng | ❌ |
| P5f | **Build the second (tactical) diffusion instance** ② at 5 s coarse + its own 256-anchor FPS vocabulary over 5 s GT trajectories, and wire S1 (G_s→both decoders) and S3 (G_t's first 2 s → the action channel, stop-grad). | eng | ❌ |
| P5g | **Build the three verification panels** — `strat_subspace_{sufficiency,compression}` (CPU probe-fit), `imagination_horizon_scaling` at k ∈ {2,20,50}, and `fan_lateral_diversification` (§6.3.1). All three are gate secondaries; none exists today. | eval | ❌ |
| P5e | 🟥 **WITHDRAWN — LAL-v2 WAS ALREADY MERGED, ON THE DAY OF THE INTAKE (O-05).** MEASURED, two probes: `stack/tanitad/eval/metrics.py:202-251` defines `decel_onset_index` **and** `compute_lal_v2` (and `metrics.py:339` emits `LAL_v2_s`); `git log -- stack/tests/test_lal_v2.py` → **`3784e34`, 2026-07-09**, *"intake(bench): LAL-v2 anticipation lead integrated"*. **The "12 days idle" claim is retracted (class C4/C2)** — it was inherited from `TANITEVAL_V2_METRIC_SUITE.md` §7 E5, which asserts "unmerged" **on the same line that names the merged file path**. What actually remains is a **different and much smaller** job: L5 is tier-1-blocked by **one line**, `taniteval/taniteval/rollout.py:94`, which computes the dense `wp_full [b,20,2]` and then keeps only 4 of 20 steps. v4 emits a dense path, so **persist all 20**. | eval | 🟡 merge done; unblock = 1 line |
| P6 | **Re-run `label_v3_audit.py` on the CANONICAL 40-episode val** (`physicalai-val-0c5f7dac3b11`, eval pod). Minutes of work; converts three PROPOSED loss weights from upper-bound-motivated to corpus-motivated. | eval | ❌ |
| P7 | **Register the gate card BEFORE launch** (§9). `run_gate.py register` refuses to overwrite. | PI/orchestrator | ❌ |
| P8 | `pytest -q` green in `stack/` after P1/P2. **Baseline MEASURED 2026-07-21: `686 passed, 2 skipped, 2 warnings in 82.99 s`** under `C:/Users/Admin/venvs/tanitad` (⚠️ *not* the bare `python` on PATH — `C:\Python314` has no pytest, which reads as a failure and is not one). Any earlier "637 passed" figure is stale. | eng | ✅ baseline pinned |

### 12.3 Escalations (these need a decision, not a paragraph in a doc)

0. ✅ **CLOSED (O-04) — the four reference logs are in the repo.** `taniteval/results/trainlogs/`
   carries all four `train_log.jsonl` + `config.json` pairs (v1-speedjerk, nospeed-phase0, v2,
   v3enc), git-tracked. Both gate criteria that depended on them (`--reference-log` for the
   matched-step ratio, and `speed_benefit_recovered_frac` against the no-speed control) are now
   computable **off a pod entirely**. *This was v4's most dangerous single-disk dependency and it is
   gone.*
1. 🟥 **`planner_p2.py` is still uncommitted** and exists only on `tanitad-eval` (REGISTRY R3). It is
   the single strongest piece of evidence for the planning direction and it is **one pod-loss from
   gone**. It is also a **hard precondition for G4**. Vendor it now, independently of v4.
1b. 🟥 **RETRACTED (O-05) — "LAL-v2 is unmerged, 12 days idle" is FALSE.** It was merged in `3784e34`
   on **2026-07-09**, the same day as the intake, and lives at `stack/tanitad/eval/metrics.py:202-251`
   with `stack/tests/test_lal_v2.py` green in the 686-test suite. **Root-cause class C4** (inherited
   from `TANITEVAL_V2_METRIC_SUITE.md` §7 E5 without running `git ls-files` / grepping the target
   module) **and C2** (absence asserted from a single probe — the intake directory still existing).
   ⚠️ **The doc-side defect is real and should be fixed at the source:** metric-suite line 80 states
   *"implemented but unmerged"* **while citing the merged path**, and §7 E5 asks the orchestrator to
   perform a merge that already happened. *Ironically this is the mirror image of the failure the
   Agent Operating Standard was written about: not work stranded outside git, but a stale escalation
   demanding work that git already contains.* **The residual real task is one line**
   (`rollout.py:94`, persist 20 of 20 steps) and it moves to P5e.
2. 🟥 **595 VLM records live pod-only** (`tanitad-eval:/root/vlm_pilot/bulk/out/`). Same
   reconstruction class. Pull to the lake or HF before the eval pod is recycled.
3. ✅ **CLOSED (O-06).** `MODEL_REGISTRY.md` §1.4 now reads *"⏹️ **STOPPED at step 10,800** · 🟥 **10 k
   GATE: `RESTART`**"* (registry line 303). The §1 banner in this document warning that the status
   field is stale is therefore also spent, and is marked resolved there.
4. ✅ **CLOSED (O-11) — v1's pace is `10.888 s/step`, and `summary.json`'s 6.37 is refuted by
   arithmetic, not by opinion.** MEASURED this session off the now-in-repo log
   (`taniteval/results/trainlogs/v1-speedjerk_train_log.jsonl`, 620 rows, last step 29,999):
   `sum(step_s)` over all `step > 0` rows = **326,638.2 s = 90.73 h** across a 29,999-step span ⇒
   **10.888 s/step**, which reproduces `GATE_PROTOCOL.md` §4's **10.89** to three decimals.
   **The refutation is a proof:** accumulated *in-loop* step time is a strict subset of wallclock, so
   a run whose logged step time totals 90.73 h **cannot** have a wallclock of 53.1 h —
   `summary.json:wallclock_s 191206.2` is therefore not the wallclock of the full 30 k run (a resumed
   segment is the likely reading) and must not be used for pacing. ⚠️ **Do not re-derive this from
   `median(step_s)`**: `step_s` is accumulated over `--log-every 50`, and median/50 gives **9.240**,
   a 15 % under-read caused by the slow tail. *Companion figures, same method:* v3enc **10.365**
   (217 rows / 10,800), no-speed control **11.246** (474 rows / 22,950); dataloader share
   `data_s/step_s` = **5.9 %** (v1) vs **13.0 %** (v3enc). **§9.2's cost model already used the
   conservative 10.89, so the v4 budget is unchanged — it is now MEASURED rather than "the safer of
   two contradictory numbers".**
5. ⚠️ **The task brief's "6 of 9 ROUTE tokens never minted" is 4 of 9** in the repo (§6.5). Corrected
   here so it does not propagate.
6. ⚠️ **Do not let `tanitad-pod` be recycled.** It holds v3enc's `ckpt_step5000.pt` and
   `ckpt_step10000.pt` — and **10 k was never on D-032's archive list, so it is the only 10 k state
   that will ever exist.** [PM] §9 names two <1 GPU-h experiments that would confirm its root cause
   (B: evaluate 10 k with the ego-dropout mask forced off vs on; C: fit the held-out speed probe on 5 k
   and 10 k to see whether the probe falls over training or starts low). **Experiment B is the
   cheapest possible confirmation of the rule v4 is now built around (§5.3) and I recommend running it
   before G0** — a 1-GPU-hour check on a design rule that touches three places in this document.

---

## 13. Naming and registry contract

| Name | Meaning |
|---|---|
| **v4** (`flagship-v4-joint-30k`) | this document. New registry section **§1.7 flagship-v4** with `lineage: v1-trunk × REF-C-decoder(d384×4L,dense) × v1.5-ab-conditioning × factorised-LATxLONxDIST`, and an explicit record of which gate rungs fired and why. |
| **v3.5** | 🟥 **SUPERSEDED by v4.** `V35_DESIGN.md` stays as provenance; it must carry a banner and must not be quoted as current. |
| **v3enc** | its own axis (encoder-grounding, OOD). 10 k verdict `RESTART`, family at 1/2. Its outcome is an *input* to v4's lever set and nothing more. |
| **v3** (`V3_HIERARCHICAL_PLANNING_DESIGN`, frozen vocab) | unchanged. v4 feeds it a better substrate (better fan → better M3 warm starts) and steals nothing from its scope. |
| **v4.1 candidates** (named, not scheduled) | predictor capacity (§3.4) · dropping H15 if it stays inert (§7.4) · **lateral-stratified anchors, LIVE conditional on §6.3.1 (X13)** · **the VIB penalty on `z_strat` if the compression probe leaks (X19)** · `--probe-grad all` · the S2 A/B · the horizon ladder k = 30/50/70 · raising `rollout_k` above 4, unlocked only by `speed_benefit_recovered_frac` (§7.2). |
| **v5 scope — Sayed's architecture fully realised, deferred on DATA not taste (§2.2)** | **10–20 s** strategic horizon (0.0 %/48.2 % supervisable on 19.9 s clips → needs longer clips) · **map-dependent routing** — "take the next exit of the roundabout", merges, lane graph (roundabout mints on 0.36 % of windows from one episode → needs a map) · **agent-interaction tactics** — "wait for the vehicles to pass", indicate, gap acceptance (`lead_state` is a `None` stub → needs detections/tracks/monodepth). |

---

## 14. OPEN-ITEMS CLOSURE REGISTER — the build-readiness sweep (2026-07-21)

*Added in the "designed → buildable" pass. Every `PROPOSED`, `UNRESOLVED`, `❌` and un-sourced number
in §0–§13 was enumerated and given one of three dispositions: **CLOSED** (decided here, with the
evidence), **CLOSED-AS-KNOB** (a PROPOSED value that is a legitimate tunable — it gets a default, a
range and the telemetry that reads it, and it never blocks a build), or **DEFERRED** (cannot close
here — with the exact thing that would close it). **No item is left implicit.***

**Score: 20 items · 15 CLOSED · 3 CLOSED-AS-KNOB · 2 DEFERRED.** IDs run O-01…O-20, contiguous.

### 14.1 Doc-integrity and tooling defects — all CLOSED

| # | Item | Disposition |
|---|---|---|
| **O-01** | §9's lever table was **duplicated**, and the orphan totalled **"1 of a limit of 2"** against the primary table's **"2 of 2"**; §9.1 and §12.1 #6 propagated the "1". | ✅ **CLOSED.** Duplicate deleted; all four sites reconciled to **2 of 2, door CLOSED**. The lower count was the more dangerous — it implied headroom that does not exist. |
| **O-02** | §1 said the budget was overturned "→ ≈239 M"; §0/§3.1 say **247.9 M**. | ✅ **CLOSED.** 239 M was the pre-strategic-planner figure. Single authoritative total = §3.1 **247,878,786**. |
| **O-03** | The §9 `run_gate.py register` command **would have exited with an error** (`=1` secondaries; `_parse_secondary` takes only `>= <= > <`), and `cmd_check` **ANDs all 14 secondaries** (`run_gate.py:549`) so any one FAIL ⇒ `RESTART`. Four of the 14 are falsifiers this document *explicitly says must not kill the run*. | ✅ **CLOSED.** Card rewritten and **split into 8 KILL + 5 REPORT-ONLY** with the consequence of each failure pre-committed (§9). This is the highest-value single fix in the sweep. |
| **O-04** | P4 / P5c / escalation 0: the four reference `train_log.jsonl` were **pod-only**, and two gate criteria were uncomputable without them. | ✅ **CLOSED.** All four pairs are git-tracked at `taniteval/results/trainlogs/`. Gate now runs **off-pod**. |
| **O-05** | P5e / escalation 1b / §7A.6: *"LAL-v2 unmerged, 12 days idle."* | 🟥 **RETRACTED, class C4+C2.** Merged **2026-07-09** in `3784e34`; lives at `stack/tanitad/eval/metrics.py:202-251`; `stack/tests/test_lal_v2.py` green. Residual real work = **one line** at `rollout.py:94`. |
| **O-06** | Escalation 3 / §1 banner: registry says v3enc is 🟢 STILL RUNNING. | ✅ **CLOSED.** Registry §1.4 already reads ⏹️ STOPPED at 10,800. |
| **O-11** | §9.2: v1's pace **6.37 vs 10.89 s/step**, "UNRESOLVED", flagged as blocking every future GPU-day estimate. | ✅ **CLOSED by arithmetic on the in-repo log — 10.888 s/step.** See escalation 4. Budget unchanged. |

### 14.2 Instrument defects that would have made gate reads inadmissible — CLOSED

| # | Item | Disposition |
|---|---|---|
| **O-07** | 🔴 **The G-H causality panel runs on the estimator `CLAUDE.md` forbids.** `taniteval/taniteval/hierarchy.py:158 _jack()` is the **overlapping-random-holdout** aggregate — the module's own header says so: *"Overlapping-random-holdout statistics (bench.py DEPRECATED protocol). NOT a jackknife. New claims: taniteval/ci.py episode_cluster_bootstrap."* Its `separated = (mean − ci95 > 0)` flag is what `per_window_content_helps`, `vision_route_beats_majority` and every `_meaningful()` verdict rest on — **and CLAUDE.md measures this estimator 1.28–2.06× too narrow.** So *three* of v4's hierarchy reads, including a **KILL** secondary, would have been decided on intervals known to over-declare significance. | ✅ **CLOSED as a required work package (P7-a).** Port `_jack` → `ci.episode_cluster_bootstrap` (paired form for two-condition contrasts on the same windows), keep `MIN_ACC 0.02 / MIN_ADE_M 0.05 / MIN_COS 0.01` unchanged, and run `assert_no_deprecated_estimator` on the hierarchy block exactly as `driving.py` already does. **~4 h, CPU-only, no GPU, no lever.** ⚠️ Expect some currently-"separated" historical hierarchy findings to stop being separated — that is the point, and it must be reported rather than quietly absorbed. ⭐ **Independently corroborated** — `TANITEVAL_V2_METRIC_SUITE.md` §7 **E3** already names all three sites (`hierarchy.py::_jack`, `closedloop.py::_jack`, `bench.py::_agg`) and says *"every hierarchy and closed-loop verdict currently rests on an estimator measured 1.28–2.06× too narrow."* **v4 is the arm that forces it, because a KILL secondary now depends on it.** Scope note: **v4 only requires the `hierarchy.py` site**; `closedloop.py` matters at G4 and `bench.py` is out of v4's path. |
| **O-08** | **Metric names in the gate card do not match what the code emits** — a silent `INCOMPLETE` verdict waiting to happen, since `cmd_check` marks any unsupplied secondary `NOT SUPPLIED` and refuses to complete. MEASURED mismatches: `train_flagship_v16.py:canary_rollout` returns **`canary_ade@2s`**, the card says `wm_canary_ade_2s`; `hierarchy.py` emits **`vision_route_beats_majority`**, the card says `nonav_route_beats_majority`; `majority_straight_rate` is a **JSON key**, not the callable `hierarchy.py::majority_straight_rate` this doc cites; §7.3/§9 use both `cruise_delta_vs_holdv0` and `cruise_speed_vs_holdv0` for the G1 and G3 forms of one quantity. | ✅ **CLOSED** by the name-mapping table in §17.3, which is the *only* admissible source for `--secondary-value` keys. Rule: **the card's name is canonical; the emitter is renamed to match, never the reverse** (renaming the card after seeing a number is a forking path). |
| **O-09** | Which `driving.py` keys the gates consume was never stated — v4 was still specified largely on ADE. | ✅ **CLOSED.** §17.3 binds each gate row to a `driving.py` key and reducer. Tier-0 runs inline in every eval, so this costs nothing: `ade_0_2s · miss_2m · long_abs_2s_m · lat_abs_2s_m · speed_mae_mps · heading_med_2s_deg` (**per curvature bucket**, §7.3) · `curv_sign_agree` · the paired verdict blocks vs **both** floors. **Three arms tie on ADE and separate on the along/cross split — a v4 gate that reads ADE alone would throw that away.** |

### 14.3 Physics, schedule and cost items — CLOSED

| # | Item | Disposition |
|---|---|---|
| **O-10** | 🔴 **The tactical planner's imagination is unpriced, and §8's latency table is built on the wrong rollout length.** §7A.4 says ② reads the *shared* probe rollout at **(10,20,30,40,50)** frames, but `V15Config.probe_steps = 20` (`flagship_v15.py:120`) and every MEASURED latency number — the 20.82 ms 8-probe fan, the ~0.3 ms marginal candidate — is at **20 steps**. Serving a 5 s tactical read requires **`probe_steps = 50`, a 2.5× longer recursion**, in *training* (§5.4's `--probe-grad one` becomes one extra **50**-step recursion, not 20) and at *inference*. | ✅ **CLOSED as a priced, pre-registered G0 measurement with a committed fallback.** (i) `--probe-steps` becomes an explicit flag, **default 50**, and §8's composed-tick estimate is restated as **PROPOSED and untested at 50 steps**. (ii) The §9.2 grad-probe surcharge rises from +25 % to **PROPOSED +45 %** ⇒ **~15.6 s/step**, which is **inside the pre-registered 16 s/step G0 ceiling but only just** — so G0 is now genuinely load-bearing rather than a formality. (iii) **Pre-committed fallback, in priority order, if G0 misses the ceiling:** `--probe-grad none` first (recorded, per §5.4) → then `--tac-probe-steps 20` with the tactical decoder reading its *anchors* at 5 s but its *imagination* at 2 s (a documented weakening of ②, not a silent one) → probes 8→4 last. **Never the encoder** (R4). |
| **O-17** | 🔴 **The scheduled-sampling ramps end AFTER the gate that judges them.** §5.2 teacher-forces the goal for steps **0–10,000** and ramps to the produced goal by **20,000**; S3 likewise teacher-forces to 10,000. But §5.3 rule 3 binds *every* leaderboard number to the **produced** goal and own-selected plan, and the G1 gate — with `nonav_route_beats_majority` as a **KILL** secondary — sits at exactly **step 10,000**. **As written, v4 is gated at the one step where it has had zero training under the conditions it is scored on.** That is a near-certain restart for a schedule reason, on an arm whose lever family has a cap of 2. | ✅ **CLOSED — the ramps are re-aligned to the λ_plan curriculum, which is free and strictly more coherent.** Goal and S3 scheduled sampling now share the phase boundaries: **teacher-forced 0 → 2,000 (Phase A) · linear ramp 2,000 → 8,000 (Phase B) · fully produced / own-selected from 8,000 (Phase C)**. At G1 the arm has **2,000 steps** of training in its evaluated configuration, and the ramp no longer straddles the gate. *One curriculum, one set of boundaries — the previous design had two schedules on unrelated clocks and the gate fell in the gap between them.* Both quantities stay reported at every milestone under teacher-forced **and** produced conditions, so the ramp's effect remains attributable. |
| **O-18** | The **strategic action set is never enumerated** — §3.1 prices an "option-prior head (8 strategic actions)" and a strategic predictor at `action_dim 4`, and §7A.5 evaluates "each candidate strategic action", but no list of 8 and no definition of the 4 dims appears anywhere. A builder cannot instantiate either module. | ✅ **CLOSED — specified in §15.4** as a **kinematically-mintable-only** set, consistent with §2.2 and X16: **8 actions = {follow, turn_left, turn_right, u_turn} × {maintain-band, change-band}**, i.e. the 4 mintable ROUTE tokens (§6.5) crossed with a binary target-speed-band intent. `action_dim 4` = the one-hot ROUTE quarter (4 dims); the speed-band intent enters as the sign of the band delta folded into the option prior, **not** as a predictor input dim. `roundabout` is deliberately **excluded** (mints on 8/2201 windows from ONE episode) — including it would put a logit behind 0.36 % of labels, the dead-parameter failure §6.5 already refuses for the lead-referenced LONMODEs. |
| **O-19** | The tactical (5 s) anchor vocabulary is specified as "256 FPS over 5 s GT trajectories" but **only 74.3 % of windows have 5 s of future** — how the vocabulary is built and how ② is supervised on the other 25.7 % was never stated. | ✅ **CLOSED.** (i) **Vocabulary**: FPS is run over the **74.3 % subset only** — it is a sampling of realisable 5 s paths, and windows without 5 s of future simply contribute none. Parity is untouched: this **re-samples trajectories, never episodes** (identical to P2's guarantee). (ii) **Supervision**: per-sample **valid masking**, never window dropping — a window with < 5 s of future keeps its full operative supervision and contributes **zero** to every ②/strategic term. **Dropping the windows instead would change the effective corpus and silently break cross-arm comparability**, which is the one thing parity forbids. (iii) The mask fraction is logged every step as `tac5s_valid_frac` and must sit at **0.74 ± 0.02**; a drift is a data bug, caught in minutes rather than at the gate. |
| **O-20** | **λ_plan's mechanism is never specified** — it is called "a scalar on the planner gradient reaching the trunk", but a loss-weight and a gradient-scale are different objects with different effects on the planner's own convergence. | ✅ **CLOSED: λ_plan is a GRADIENT SCALE at the seam, not a loss weight.** A `torch.autograd.Function` that is identity on the forward pass and multiplies `grad_output` by λ on the backward, inserted on the **trunk→planner activation boundary** (the readout state as it enters ② and ③, and `z_strat` as it enters `E_strat`). **Why it must be the gradient scale:** a loss weight of 0 would stop the planner learning at all, so Phase A (λ_plan = 0) would train nothing; a *gradient* scale of 0 lets the planner heads train at full strength on a trunk they cannot move — which is exactly the LP phase of LP-FT that Phase A is supposed to be. **`--lambda-plan 0` therefore reproduces the frozen-trunk v1.5 regime byte-identically**, which is the attributability claim §9 rests on and which a loss weight would not have delivered. |

### 14.4 PROPOSED values — CLOSED-AS-KNOB (defaults, ranges, and the telemetry that reads them)

*A PROPOSED number is not an open question if it has a default, a stated range, and a logged quantity
that tells you whether it was wrong. These do. **None of them blocks the build**, and none may be
changed after launch without re-registering the card.*

| # | Knobs | Default | Range | The telemetry that reads it |
|---|---|---|---|---|
| **O-12** | **Loss weights** (§4.3): LAT / LON / DIST CE **0.05** each · strategic goal CE **0.1** · strategic 5 s prediction **0.5** · goal scalars **0.05** each · plan-smoothness `w_j` **0.02**, `w_κ` **0.01** | as listed | ×⅓…×3 | per-term loss magnitudes logged every 50 steps **as a fraction of total** — the check [PM] #7 wishes it had run on v1's jerk term (**≤1e-4 of ~4.0**, i.e. the term was never on). ⚠️ **Any new term reading below 1e-3 of total at 2 k is inert and must be reported as such, not defended.** P6 (canonical-val label audit) converts LAT/LON/DIST from upper-bound-motivated to corpus-motivated **before** launch. |
| **O-13** | **Geometry**: `d_strat` **128** · strategic predictor **d256 × 4L** · `E_strat` 2048→512→128 | as listed | `d_strat` ∈ {64, 128, 256} | the §7A.3(iii) two-probe test **is** the read: sufficiency < 0.90 ⇒ widen, compression > 0.50 ⇒ narrow or fire VIB (v4.1, X19). |
| **O-14** | **Curriculum**: Phase A 0→2 k · B 2→8 k · C 8→30 k · discriminative LR head 1e-4 / trunk 3e-4 · `--probe-grad one` | as listed | — | canary every 500 steps (controller at +0.05, §5.5) · `speed_benefit_recovered_frac` per 2 k bucket · seam-norm ratios every log step. **The λ_plan schedule is the one thing the canary controller is allowed to move mid-run, and it may only move it *down*.** |

### 14.5 DEFERRED — with exactly what would close each

| # | Item | Why it cannot close here | What closes it | Does v4 depend on it? |
|---|---|---|---|---|
| **O-15** | **Post-mortem experiment A's verdict** (`v2_ego_dropout = 0.0`, 2 k steps, pod1). | 🔴 **The result is not available off-pod and I did not touch a pod.** MEASURED absence, two probes: `taniteval/results/postmortem_a_egodropout_off_expA2k.json` does not exist, and the analyzer's expected input `taniteval/results/trainlogs/expA-nodrop_train_log.jsonl` is absent while its three sibling logs are present. **Stage 2 of 2 is written and waiting** (`taniteval/postmortem_a_analyze.py`, with the estimator, the bucket convention and the `artifact_only_null_band` all pre-registered in its docstring). | `scp` the exp-A `train_log.jsonl` off pod1 → `taniteval/results/trainlogs/expA-nodrop_train_log.jsonl` → run `postmortem_a_analyze.py`. **Minutes, zero GPU.** | 🟢 **NO — and this is deliberate, per the brief's warning.** §5.3's rule is *"never zero-fill a channel whose zero is in-distribution"*, and its two v4 consequences stand on evidence that is **independent of exp-A's verdict**: (a) **the operative action channel is simply never masked in v4** — a design choice needing no attribution at all; (b) **P5b's learned-null row** is justified by exp-B's *direct* measurement that the model built an implicit null embedding at 0.0 aliased with the **6.45 %** genuinely-stopped windows, so **78.4 %** of the zeros it saw under the mask were lies — which is a statement about the mechanism, not about v3enc's eval failure. ⚠️ **Binding: v3enc's held-out failure remains UNATTRIBUTED** (exp-B: the mask explains **~51 %** of the *training-log* gap and **NONE** of the eval gap, because it is gated on `model.training` and TanitEval loads `.eval()`). **No v4 claim, gate or threshold may rest on a zero-fill story.** If exp-A tracks v3enc (mask = side-show), **nothing in v4 changes** — that is the test of whether this decoupling is real, and it passes. |
| **O-16** | **`obstacle.offline` in v4's model** — 3D agent tracks on **96.90 %** of the corpus, never ingested (our loader reads **2 of 36** features). | Three independent reasons, any one sufficient. **(1) The lever door is closed** (O-01): agent state is a new input surface, and v4 is at **2 of 2**. **(2) It does not exist yet** — the ingest is **2–3 days** of data-engineering (`DATA_STRATEGY_FOR_HIERARCHY.md` §5, 197 chunks / 12.4 GB) and is not started. **(3) It has no measured effect on anything**, so folding it in would be adding an unmeasured lever to three structural changes — the v2 mistake (N6). | The ingest lands and a **separate arm** prices it. It is parity-safe (re-derives labels on the *same* episodes, never re-selects), so it costs no comparability. | 🟢 **NO — v4's model is unchanged, and X17 already scopes agent-interaction tactics to v5.** ⭐ **But there is a zero-lever use of it that v4 should take if the ingest lands before G3: eval-side stratification only.** `obstacle.offline` supplies an exact **lead-present** flag (MEASURED: **45.2 %** of frames, **87 %** of clips) at **zero** model change and zero levers. v4's hardest gate — `cruise_speed_vs_holdv0`, which all 14 arms fail — is a *free-flow* speed-tracking claim being scored on a corpus where nearly half the frames are lead-constrained. **Reading it split free-flow vs lead-constrained would not move the gate, but it would say for the first time whether the cruise failure is a speed-control failure or a following failure.** Recommended as a **REPORT-ONLY** companion, never a threshold. Model-side: **v5**. |

---

## 15. THE BUILD PLAN — ordered work packages

**Ordering rule: every package leaves the tree green and each earlier package is independently useful
if the later ones are cancelled.** P1→P4 alone give a *runnable two-diffusion arm* (`--strategic off`),
which is §2.6's lower-risk alternative; the strategic layer is added on top rather than woven through.

⚠️ **Honest headline: §12.2 P1 estimated "2–3 agent-days" for the model + trainer alone. The whole
build is ~118 h ≈ 3 agent-weeks of one engineer.** The **build is the critical path, not the GPU** —
G1 costs 1.66 A40-days and G3 4.96, against ~15 working days of engineering. Anyone planning around
"press go tomorrow" is planning around the wrong number.

| P | Package | Files touched / new | Tests required | h |
|---|---|---|---|---|
| **P1** | **Model core.** `FlagshipV4Head(FlagshipV15Head)`: dense horizons `(1..20)`, LAT(8)/LON(7)/DIST(8) heads, the three **zero-init** `*_to_anchor` grafts (§6.2), per-graft norm telemetry + the in-graph clamp at 1.0× / fail-loud 1.5×. **P5b lands here**: `ego_dropout` zero-fill → **learned null-embedding row** (`flagship_v15.py:348-351`). | **new** `stack/tanitad/models/flagship_v4.py`; edit `flagship_v15.py` (null row) | param-count pin vs §3.1 · **zero-init graft ⇒ selection path bit-identical to the 5-way baseline** (the attributability claim) · null-row ≠ zero row · clamp fires at 1.5× | **14** |
| **P2** | **λ_plan as a gradient scale** (O-20) + the `--strategic {full,head,off}` / `--lambda-plan` switch plumbing. | `flagship_v4.py`, new `stack/tanitad/train/grad_scale.py` | **λ=0 reproduces the frozen-trunk forward AND leaves trunk grads exactly zero** · λ=1 is identity · the three `--strategic` modes each reproduce their sub-config byte-identically | **8** |
| **P3** | **Anchor vocabularies.** `build_refc_anchors.py` at horizons 1…20 (dense, 256 FPS) **and** the 5 s coarse vocabulary over the **74.3 %** subset (O-19). Same source / pool-cap / seed. | `stack/scripts/build_refc_anchors.py` | **parity: no episode re-selection** · dense buffer = 10,240 floats · base-128 remains a bit-exact prefix of 256 (`max|A − B[:128]| = 0`) | **6** |
| **P4** | **Trainer.** `train_flagship_v4.py` = `train_flagship4b.py`'s WM stack + `train_flagship_v16.py`'s canary/LP-FT plumbing + `v15_losses` + the new terms (§4.3) + the **three-phase curriculum with the re-aligned ramps** (O-17). | **new** `stack/scripts/train_flagship_v4.py` | 20-step CPU smoke on toy data · phase boundaries fire at 2 k / 8 k · canary computes at step 0 · **every new loss term appears in the log with a non-zero value** | **20** |
| **P5** | **Tactical instance ② + S3.** Second `FlagshipV4Head` at 5 s coarse; S1 goal-KV into both decoders; S3 = G_t's first 2 s → inverse-dynamics → the **existing** 3-dim action channel, **stop-grad**; shared probe rollout at `--probe-steps 50` (O-10). | `flagship_v4.py`, `train_flagship_v4.py` | **stop-grad verified by a grad-norm assert, not by inspection** · S3 adds no new action dims · `tac5s_valid_frac` ≈ 0.74 | **16** |
| **P6** | **Strategic planner ①.** `E_strat` 2048→512→128, the d256×4L predictor in the subspace, the **8-action set of O-18**, the option prior, the imagined-rollout evaluator, goal scalars, the norm-capped `z_strat`→KV projection, per-sample 5 s masking. | **new** `stack/tanitad/models/strategic_v4.py` | param pin **5,152,911** · `--strategic head` skips the rollout · `--strategic off` byte-identical to P5's arm · **evaluator ≠ argmax-of-prior** (else it is a head, §7A.5) | **20** |
| **P7** | **Eval + instruments.** **(a) the estimator port** — `hierarchy.py::_jack` → `ci.episode_cluster_bootstrap` (**O-07, do this first — it changes what the other panels mean**); (b) the three verification panels `strat_subspace_{sufficiency,compression}`, `imagination_horizon_scaling` at k ∈ {2,20,50}, `fan_lateral_diversification` (§6.3.1); (c) **P5 the evaluator `nav_cmd` fix** — produced goal, never `None → follow`; (d) `rollout.py:94` persist 20 of 20 (unblocks L5, O-05). | `taniteval/taniteval/{hierarchy,rollout}.py`, new `taniteval/taniteval/strategic_probes.py`, `refc_eval.py`, `plan_fan.py` | `assert_no_deprecated_estimator` passes on the hierarchy block · panels run on **held-out** episodes with **S1/S3 zeroed** (§7A.4) · produced-goal path asserted in the eval config | **22** |
| **P8** | **Gate wiring.** Promote `postmortem_a_analyze.py`'s bucket reducer into a `speed_benefit_recovered_frac` function (**reuse its pinned convention — do not re-derive**); the §17.3 name map; register the split card. | `stack/scripts/run_gate.py`, new `stack/tanitad/eval/speed_benefit.py` | **reducer reproduces the published rows** (v1 0.1062 / v3enc 0.4699 / no-speed 0.5740 at 8–10 k) · card registers without error · every secondary name resolves | **8** |
| **P9** | **Pre-flight + G0** (§17), incl. the **§17.1b gate dry-run against flagship v1**. Milestone archive list gains **step 10,000** (P3 of §12.2 — the exact gap that bit v3enc). | `stack/scripts/*` archive list | the §17 checklist, executed | **4** |
| | **TOTAL** | | | **118** |

**Priority order if the work is cut short** (per the orchestration rule that a killed stream still
yields value): **P1 → P2 → P3 → P4** ships a two-diffusion arm that can be gated on its own and is
already a program first (joint WM + planner). **P7(a) alone** is worth doing regardless of v4 — it
retroactively fixes the admissibility of every hierarchy claim the program has made. **P6 last**, so
that a cancelled strategic layer costs nothing already built.

### 15.4 The strategic action set, enumerated (closes O-18)

| idx | ROUTE quarter (`action_dim 4`, one-hot) | speed-band intent | mintable from |
|---|---|---|---|
| 0–1 | `follow` | maintain / change | `route_from_future_v3` + the 5 s target-speed band |
| 2–3 | `turn_left` | maintain / change | ″ |
| 4–5 | `turn_right` | maintain / change | ″ |
| 6–7 | `u_turn` | maintain / change | ″ |

**Excluded and why:** `roundabout` (**8/2201 windows from ONE episode**, 24 u-turn confusions — a logit
behind 0.36 % of labels is a dead parameter that invites a shortcut, §6.5) · `straight`, `exit_*`,
`merge` (**map facts**, unmintable, X10). **The "fit-to-goals" cost is `progress_m − w · |v̂₅ₛ − v_band|`
with `w` PROPOSED 0.1** — a knob under §14.4's rule, logged per option so its influence on the argmax
is visible. **The prior head only seeds; the imagined rollout selects** (§7A.5 read 3 is the check).

---

## 16. THE EXACT CONFIG / CLI SURFACE

**Contract: v4's trainer is `train_flagship4b.py`'s surface plus the flags below and nothing else.**
Every v1 flag keeps its v1 default, so a v4 invocation with all new flags at their *reproduction*
values is the v1 recipe minus `--aux-accel`/`--jerk-weight` ([PM] #7).

| Flag | Default | Reproduces | Notes |
|---|---|---|---|
| `--lambda-plan {0\|1\|sched}` | `sched` | **`0` ⇒ frozen-trunk v1.5 regime, byte-identical** | gradient scale (O-20), not a loss weight. `sched` = 0 / 0→1 linear / 1 over phases A/B/C |
| `--phase-a-steps` · `--phase-b-steps` | `2000` · `8000` | — | also the goal + S3 ramp boundaries (O-17) |
| `--strategic {full,head,off}` | `full` | **`off` ⇒ the two-diffusion arm, byte-identical** | `head` = the §2.6 fallback (E_strat + classifier, no rollout) |
| `--d-strat` | `128` | — | ∈ {64,128,256} |
| `--long-horizon-k` | `50` | **`0` ⇒ no tactical instance, no 5 s terms** | 50 frames = 5 s; the O-19 mask rides on it |
| `--probe-steps` | `50` | `20` ⇒ v1.5's MEASURED seam | ⚠️ **O-10: 50 is 2.5× the only length ever measured** |
| `--probe-grad {none,one,all}` | `one` | `none` ⇒ v1.5 byte-identical | auto-drops to `none` if G0 misses the pace ceiling — **recorded, not argued** |
| `--dense-plan / --no-dense-plan` | on | `--no-dense-plan` ⇒ the 4-point surface | +24,608 params |
| `--lat-weight` · `--lon-weight` · `--dist-weight` | `0.05` each | **`0` each ⇒ the 5-way baseline selection path** | grafts are zero-init, so 0 is also the step-0 state |
| `--strat-goal-weight` · `--strat-pred-weight` · `--strat-scalar-weight` | `0.1` · `0.5` · `0.05` | `0` ⇒ no strategic supervision | |
| `--jerk-w` · `--curv-w` | `0.02` · `0.01` | `0` ⇒ no smoothness term | acts on the **dense emitted plan** only |
| `--ego-null-row / --ego-zero-fill` | `--ego-null-row` | `--ego-zero-fill` ⇒ the v3enc bug, **for ablation only** | ⛔ **X15. Never in a shipping run.** |
| `--s2-film` | **off** | — | the 0-for-4 family; exists only for the pre-registered A/B |
| `--rollout-k` | **4** | v1 verbatim | ⛔ **[PM] #2 — do not raise before `speed_benefit_recovered_frac` unlocks it** |
| ~~`--aux-accel`~~ · ~~`--jerk-weight`~~ | **removed** | — | [PM] #7 |

**The four one-lever diffs that make v4 attributable** — each changes exactly one thing against the
full arm, and this is the answer to *"v2 and v3enc could not attribute their own results"*:

```
full          : --lambda-plan sched --strategic full --long-horizon-k 50 --lat/lon/dist-weight 0.05
isolate λ_plan: --lambda-plan 0        # LP regime; trunk unmoved by the planner
isolate ①     : --strategic off        # the two-diffusion arm
isolate ②     : --long-horizon-k 0     # operative-only, no 5 s layer
isolate select: --lat-weight 0 --lon-weight 0 --dist-weight 0    # the 5-way baseline path
```

⚠️ **Byte-identity is a TEST, not a promise** (P1/P2/P6 test rows). Each switch is asserted to produce
an identical forward on a fixed seed against the arm it claims to reproduce. **v3enc believed it had
controls too.**

---

## 17. PRE-FLIGHT CHECKLIST — run before the first real step

*Every row is a command with a pass condition. A row that cannot be evaluated is a FAIL, not a skip.*

### 17.1 Build gates (no GPU)

| # | Check | Pass condition |
|---|---|---|
| 1 | `pytest -q` in `stack/` **under `C:/Users/Admin/venvs/tanitad`** | **≥ 686 passed, 2 skipped**, 0 failed. ⚠️ the bare `python` on PATH has no pytest and reads as a failure |
| 2 | Param count vs §3.1 | trainable = **247,878,786**; and the harness reproduces `WorldModel(flagship4b_config())` = **263,440,533** as the faithfulness check |
| 3 | **One-lever config diff** | all four diffs of §16 asserted byte-identical on a fixed seed |
| 4 | Gate card registers | `run_gate.py register …` exits 0 and writes `Project Steering/Gates/flagship-v4.card.json`; **every secondary name resolves against §17.3** |
| 5 | Estimator admissibility | `assert_no_deprecated_estimator` passes on the hierarchy block (O-07). **No `_jack` interval reaches a verdict** |
| 6 | Parity | corpus key `physicalai-train-e438721ae894`, skip-hash `f09e44db`, **2376 episodes**, printed by the trainer at step 0 and matched against the registry |
| 7 | Milestone archive list | contains **5k / 10k / 15k / 20k / 30k** — 10 k is the gate step and its omission is what bit v3enc |
| 8 | **P6 label audit on the CANONICAL val** | `label_v3_audit.py --val physicalai-val-0c5f7dac3b11` has run; the LAT/LON/DIST weights are corpus-motivated, not upper-bound-motivated (the 100-ep local build is markedly more urban: median `v0` **5.62 vs 10.29 m/s**) |

### 17.1b ⭐ THE GATE DRY-RUN — the single highest-value pre-flight item

> **Before v4 trains one step, run the ENTIRE G1 panel end-to-end against flagship **v1**'s archived
> checkpoint** — the arm whose every number we already know (`ade_0_2s` **0.4271** full-set, canary
> **0.452**, miss@2m **0.0602**, oracle 0.3073, `speed_benefit_recovered_frac` **81.8 %** at 8–10 k).
> Register a throwaway card (`--card .../flagship-v4-dryrun.card.json`), supply every
> `--secondary-value`, and require the verdict machinery to print a full row for **all 13**.
>
> **Why this is the cheapest insurance in the document.** v4's gate is a **13-way conjunction on
> instruments that mostly do not exist yet** — five panels are being written in P7, one reducer in P8,
> and four metric names need renaming (§17.3). `cmd_check` marks any unsupplied secondary
> `NOT SUPPLIED` and returns **`INCOMPLETE`**; a mistyped or unrenamed key therefore burns the
> milestone rather than failing loudly at registration. **The dry-run converts "13 unbuilt
> instruments" into "13 instruments that have each produced a number at least once, on a checkpoint
> whose answer is known."** A panel that disagrees with v1's published value is a panel bug, caught for
> **~0.2 eval-days and zero training GPU** — against a real G1 miss, which costs **1.66 A40-days and a
> restart from a family capped at 2.**
>
> **Pass condition:** all 13 secondaries resolve and print; v1's values reproduce the registry to the
> precision each metric is published at; verdict is whatever it is — *we are testing the instrument,
> not v1.* ⚠️ **Use `--force` on the throwaway card only. Never on `flagship-v4.card.json`.**

### 17.2 Step-0 / G0 gates (pod1, ~30 min)

| # | Check | Pass condition |
|---|---|---|
| 9 | **Canary baseline captured at step 0** on the warm trunk | a number is written to the run dir. **v1's reference is 0.452** — a step-0 canary far from it means the warm-start or the harness is wrong, and that is a stop-before-you-start, not a gate |
| 10 | **Seam norms instrumented and non-zero** | `S1`, `S_strat`, `lat/lon/dist_to_anchor` ratios all present in the log. ⚠️ a seam whose telemetry is missing is a seam nobody can prove did anything — **F3 was HARMFUL for a whole arm before anyone measured it** |
| 11 | Every new loss term is live | each appears at step 50 with a non-zero value ≥ 1e-3 of total (the [PM] #7 test) |
| 12 | **Pace** | ≤ **16 s/step**. ⚠️ **O-10 raised the estimate to ~15.6, so this is now a real test.** Miss ⇒ the §14.3 fallback ladder, in order, recorded |
| 13 | `tac5s_valid_frac` | **0.74 ± 0.02** (O-19) |
| 14 | Disk | a real **`dd`** write test on pod1. ⛔ **never `df`** |

### 17.3 The name map — the only admissible source for `--secondary-value` keys

| Gate name (canonical) | Emitter today | Action |
|---|---|---|
| `wm_canary_ade_2s` | `canary_rollout` → `canary_ade@2s` | **rename the emitter** |
| `nonav_route_beats_majority` | `hierarchy.py` → `vision_route_beats_majority` (vs key `majority_straight_rate`) | **rename the emitter**; ⚠️ it is a **JSON key**, not the callable this doc cited |
| `cruise_delta_vs_holdv0` (G1) / `cruise_speed_vs_holdv0` (G3) | `driving.py` → `speed_mae_mps` paired vs `holdv0`, reducer `mean`, on the **639 steady** windows | two names, one quantity: **G1 reads the paired Δ, G3 reads the separation verdict** |
| `ade_0_2s` · `miss_at_2m` | `driving.py` → `ade_0_2s` · `miss_2m` | align `miss_at_2m` → `miss_2m` |
| `speed_benefit_recovered_frac` | — | **new**, P8, off the two in-repo logs |
| `strat_subspace_*` · `imag_win_at_5s` · `longh_5s_beats_persistence` | — | **new**, P7(b). REPORT-ONLY (§9) |
| `oracle_in_fan` · `seam_norm_ratio_max` · `deploy_tick_p99_ms` · `encoder_touching_levers` | fan probe · trainer telemetry · efficiency panel · design audit | as-is |

**Driving-panel rows every gate reports** (§14.2 O-09), all with the episode-cluster bootstrap and
paired against **both** floors: `long_abs_2s_m` · `lat_abs_2s_m` (the split three arms tie on ADE and
separate on) · `speed_mae_mps` · `heading_med_2s_deg` **per curvature bucket, never aggregate** (an
aggregate hid both v3enc's straight-road win **and** its 4.9× gentle-curve loss) · `curv_sign_agree`.

---

*Verification note. Pods were not touched while writing this. Parameter counts were measured locally
by instantiating the committed configs (`C:/Users/Admin/venvs/tanitad`, torch 2.11.0+cu128); the same
script reproduces `WorldModel(flagship4b_config())` = 263,440,533 exactly, which is the validation
that the instantiation is faithful. The §7A.2 horizon-coverage table was measured by reading the two
local epcaches read-only (`physicalai-val-bb543bdf7836` 100 ep + `physicalai-train-14231cd29c74`
400 ep = 500 episodes / 95,477 window starts) and counting clip lengths — **no episode was selected or
re-selected; parity is untouched.* ⚠️ Neither local build is the canonical `physicalai-train-e438721ae894`;
clip length is a property of the PhysicalAI source and the two builds agree to <0.1 pp on every row,
but the canonical figure should be re-read on the pod when convenient. Every other number carries its
source inline. No training was launched, no checkpoint touched, nothing committed or pushed — this
file is staged only.*
