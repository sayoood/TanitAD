# COMMS — `2026-08-30-t1-floor-action-sensitivity`

## Integration status: ⛔ ESCALATED — register rows NOT applied by me

`Project Steering/GOALS_AND_CLAIMS.md` was **modified at 09:15:06 while this run was in
progress** (it moved under me between my 08:53 read and my 09:20 check). With four agents live
I did **not** bulk-edit a contended 285 KB file — the paste-ready deltas are below instead.
**Owner to apply: TanitAD Master Mind.**

---

## Delta 1 — AMEND `D-T1-V7-READ`'s scope clause

The row currently states the arms are *"a world-model trunk plus a grounding readout, **never
trained to drive**"*. That is true of the **objective** and false of the **architecture**, and
the difference decides which literature applies (RESULT.md F1). Proposed replacement for that
clause:

> ⚠️ **SCOPE, BINDING (amended 2026-08-30):** these are ~19 M-param v7-TINY arms trained on
> stage S-W ONLY, with every planner objective at zero
> (`--w-o1-ctrl 0 --w-o1-fact 0 --w-o1-scene 0 --w-o2 0 --w-o3 0`). ⚠️ **They are
> action-conditioned BY CONSTRUCTION and action-free BY OBJECTIVE** — `models/v6.py:5455`
> `forward(self, frames, actions, v0, …)` takes actions positionally and the predictor is built
> `action_dim=3` (`train_v6_staged.py:4122`), while **O1 — the action-conditioned term whose
> documented job is to be "the ANTI-ACTION-ECHO measure", `default=1.0` at
> `train_v6_staged.py:7042` — was switched off.** ⛔ Therefore *"of course it cannot drive, it
> was never trained to"* is **NOT established by the configuration**: `2411.04983` (DINO-WM) and
> `2412.03572` (Navigation World Models) both plan closed-loop with **no planner objective at
> all**, via a test-time optimiser. The supported claim is narrower: **the action-response
> objective was at zero, so ACTION-INSENSITIVITY is unexcluded** — and the untrained-planner
> floor is MEASURED at T1.

## Delta 2 — REGISTER new hypothesis `H-ARCH-ACTINS`

| id | claim | status | evidence |
|---|---|---|---|
| `H-ARCH-ACTINS` | ⭐ **HYPOTHESIS: the v7-tiny T1 floor is ACTION-INSENSITIVITY (the model barely uses its action input), not bad driving.** The closed-loop vs hold-action gap is ~1 % on every distance metric — `emao14_30k` ade +1.37 %, fde +0.64 %; `o14fut30k` ade +1.25 %, fde +0.94 % — i.e. handing the model the wheel barely changes the rollout. A model *driving badly* would diverge from a frozen-action rollout substantially. The literature names the mode: **"Context Collapse"** [`2607.26712`] = *"nearly indistinguishable futures under different action sequences"*, and [`2606.31232`] = *"reconstruction-free joint-embedding objectives can collapse to action-insensitive representations"*. Mechanism is available: O1 (the action-response term) was at weight 0. | **HYPOTHESIS — registered, discriminating test NOT YET RUN** | ⚠️ **A competing account fits the same evidence and banked numbers cannot separate them:** at heading MAE ≈ 95° (chance) and ADE ≈ 14 m the rollout may be degenerate enough that *no* action choice moves the metrics. Both predict a small gap ⇒ ⛔ **no verdict drawn.** Committed discriminating test: **action-divergence probe** — roll one initial state under DISTINCT action sequences, measure the spread of predicted futures. **Spread ≈ 0 ⇒ COLLAPSE** (and the frozen-teacher lever is aimed at the wrong defect); **spread large ⇒ RESPONSIVE** (the floor is a planner/decoder problem). T0, label-free, no training. Second instrument: action recoverability [`2606.07687`] — ⚠️ `InverseDynamicsHead` EXISTS (`stack/tanitad/models/inverse_dynamics.py`, used `fourbrain.py:461`) but is **NOT wired in `models/v6.py`**, so it is a port not a build. Derived from MEASURED T1 raws `/home/nvidia/t1dumps/{emao14_30k,o14fut30k}/t1.json`. T1-derived, arithmetic only. · `…/2026-08-30-t1-floor-action-sensitivity/RESULT.md` F2–F4 |

## Delta 3 — CLOSE the τ-ramp arm

| id | claim | status | evidence |
|---|---|---|---|
| `D-TAU-RAMP` | **The cosine τ-ramp is NEUTRAL vs fixed τ at 30k and is DROPPED; the EMA teacher stays at FIXED τ = 0.996.** drift **0.6936 vs 0.6952**, nrmse **0.7408 vs 0.7466**, cos **0.7513 vs 0.7524** — all ~**20× smaller than the only seed spread we have measured**, so the comparison cannot resolve the arms. | **MEASURED — CLOSED (neutral)** | T0-DIAGNOSTIC. ⚠️ Neutral ≠ refuted: the arm is under-powered against seed noise, so this closes the *recipe* question (fixed τ stays) without licensing a claim that ramping cannot help. Relayed from the Master Mind 2026-08-30. |

---

## Note for `MM-E5` (no row change requested)

[`2510.26782`] — *"the primary bottleneck for long-horizon fidelity is the geometric structure
of the latent representation, not the dynamics model itself"* — is a **third, mechanism-
independent** reason the drift↔prediction coupling is worth its discriminating arms. ⛔ Per
`MM-E5`'s own binding clause this does **not** license re-reading any result through the
frontier framing before L2/L3/L4 report, and it is recorded as motivation only.

## Decision asked of the PI / Master Mind

**Should the action-divergence probe run BEFORE the L2 frozen-teacher arm?** RESULT.md F4 argues
yes: it is T0, label-free, needs no training, and it determines whether the drift lever is aimed
at the right defect at all. If `H-ARCH-ACTINS` holds, the indicated lever is `--w-o1-ctrl > 0`
(the term we switched off), not the teacher.
