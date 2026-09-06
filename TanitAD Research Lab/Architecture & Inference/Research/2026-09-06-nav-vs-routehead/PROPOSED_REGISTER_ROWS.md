# PROPOSED REGISTER ROWS — D-NAVROUTE-1

⛔ **For the Master Mind to merge into `Project Steering/GOALS_AND_CLAIMS.md`.** Written as a
proposal rather than applied in place: the register is a 1.6 MB shared file, several agents are
live this session, and the shared index has been regenerating phantom deletions. **This is an
integration escalation, not a "please merge" note** — it is named explicitly in the agent's
final report.

Artifacts: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-nav-vs-routehead/`
(`PREREG.md`, `RESULT.md`, `raw/CONTINGENCY.json`, `raw/NAV_ARGS_CENSUS.json`,
`raw/GSTR_INTERVENTION.json`).

---

### D-NAVARGS-1 — the nav command's range and time exist in the label and never reach refcv4b
**Status: SUPPORTED (MEASURED, ours).** **Tier:** n/a (source + corpus census, no model).

`vocab_v7.NAV_ARG_SLOTS = ("distance_m","time_s")`, written by
`s2_geom_emit_v7.nav_command()` on TURN tokens only; `NAV_FOLLOW_ROAD` carries `args: {}` on
**2,897/2,897 train** and **96/96 eval** records. Three consumers disagree: **v6/v7f FEEDS** the
two args through `NavConditioner.arg_proj = nn.Linear(2, d_embed)`; **refav1 DISCARDS** them
(`ids, _args = em(eis)`); **refcv3/refcv4b NEVER READS** them (`refc_v3_train.py` takes
`nav.get("token")` only and stores `item["nav_cmd"]` as a scalar index into the 4-row legacy
`NAV_COMMANDS`). ⇒ The deployed arm's nav input is a **bare 3-way categorical, zero range, zero
time**. Corpus: turns are commanded at median **27.3 m / 7.2 s** (TURN_L) and **36.6 m / 7.4 s**
(TURN_R) on train; under the consumer's `t0_constant` default **69.97 %** of train records present
`distance_m = 0.0`.
⭐ **Consequence:** this is exactly the stripped **bearing** the programme already measured as
separated WORSE by **+2.3632 m**, against a range-carrying goal point worth **57.1 %** of the
longitudinal selection ceiling. The lever is implemented already, in `NavConditioner`.

### D-NAVROUTE-2 — the plan tracks the model's OWN route readout ~3.2x more than the command
**Status: SUPPORTED (MEASURED, ours).** **Tier: T1.** **Estimator:** paired episode-cluster
bootstrap, 2,000 resamples, seed 0.

On the **1,908** windows (96 episodes, of 4,823/141) where the core route head and the commanded
nav token DISAGREE: plan matches the **route head 1,248 (0.6541)**, the **command 391 (0.2049)**,
neither 269. `follow_gap = +0.4492 [+0.3369, +0.5551]`, **separated**; robust across all
pre-registered deadbands (`tau_y` 0.5/1.0/2.0 -> +0.2993/+0.4492/+0.5372, separated at each).
Orientation control PASSES (mean GT terminal y **+2.8608 m** under nav=LEFT, **-4.1898 m** under
nav=RIGHT).
⛔ **This is ASSOCIATION, not causation** — see D-NAVROUTE-3.
⚠️ Variance answered: **episode draw only.**

### D-NAVROUTE-3 — the head the plan tracks CANNOT be causing it (`graft_route=false`)
**Status: SUPPORTED (MEASURED, ours — config + source).**

`route_prior = log_softmax(route_logits) if cfg.graft_route else None`, and `route_to_anchor` is
constructed only under the same flag. refcv4b's **own banked config records `"graft_route": false`**
(verified independently in `refcv4b_navflip.json` and `refcv4b_t1.json`). ⇒ `route_logits` reach
**nothing** downstream. The correlation in D-NAVROUTE-2 is **shared vision**: the route head is a
faithful *telltale* of what the vision-driven planner will do, not its cause.
⭐ **This refines the PI's hypothesis rather than refuting his observation:** the behaviour he saw
is real and typical; the mechanism is that the planner is **vision-dominated**, not that a route
head overrides the command.

### D-NAVROUTE-4 — `g_str` is SIGN-DEGENERATE: it never points left on any eval window
**Status: SUPPORTED (MEASURED, ours).** ⛔ **This is a defect, not a metric.**

`gstr_nav_true[:,1]` (the lateral bearing component) is **negative on 4,823/4,823 windows**
(max **-0.1022**, min -0.9711) while ground truth turns LEFT on **1,597** windows (33.1 %). The
strategic goal head's direction *class* is constant RIGHT.
⇒ (a) any "does the plan follow `g_str`" readout is **DEGENERATE and must not be quoted as a
measurement**; (b) the banked `gstr: NAV_BLIND` verdict, whose deltas are **exactly 0.0 with
CI [0,0]**, is an **identity, not an estimate** (same family as `H-ECHO-4`); (c) a goal signal that
cannot express "left" cannot be steering the planner into left turns.

### D-NAVROUTE-5 — presence-gating reproduces AT THE STRATEGIC HEAD, not only at the planner
**Status: SUPPORTED (MEASURED, ours).** **Tier: T1.**

Mean relative movement of `g_str`: nav **value** changed **0.0098**, shuffled 0.0207, nav
**removed 0.4018** — **40.9x**. At the plan, terminal displacement under navflip is
**mean 1.1226 m but MEDIAN 0.0000 m** (inverting the command changes most plans by *exactly
nothing*); under navzero mean 1.8644 m, median 0.4901 m.
⚠️ **Supersedes the bare "18.6x"** figure with a stated normalisation (mean relative movement over
4,823 windows) — the ratio's direction is confirmed, its magnitude is normalisation-dependent, and
it should be quoted with the normalisation or not at all.

### D-ESTIM-SHARED-SCRATCHPAD-1 — a bare `RESULT.md` in the shared scratchpad is not yours
**Status: SUPPORTED (MEASURED, ours — this session).** **Class: mis-scoped artifact.**

A push helper that copied `<scratchpad>/RESULT.md` into this work package banked **another agent's**
deliverable (the register-repair agent's, 23,452 B, timestamped before this session began) under
this package's name. Caught by reading the first line of what had been written; removed. ⇒ **Name
deliverables distinctly in the shared scratchpad (`MY_RESULT.md`) and verify a banked file by a
CONTENT marker you authored, never by the copier's exit code.** Same family as the existing
"verify by content, never by presence" rule, with the object being *authorship* rather than bytes.
