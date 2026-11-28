# PRE-REGISTRATION — does the route threshold fix move ADE, or only the route metric?

**Written and committed BEFORE the arm was run.** 2026-07-29.

## The question

Task #43 showed that lowering the produced-goal route threshold from `tanh(1.0) = 0.7616` to **0.35**
lifts **balanced route accuracy 0.4242 → 0.5493** (right-turn recall 0.041 → 0.289) with **no
training**. That is a *route-metric* result. It says nothing yet about whether the planner's
**trajectory** improves — the route is only one of the goal channels the head is conditioned on, and
a head can be largely insensitive to it.

**This is the test that decides whether the fix is worth deploying.** It was named as follow-on
item 1 in `ROUTE_THRESHOLD_SWEEP.md` §4 before any of it was run.

## Design

- **Arm A (control):** `--goal-mode produced`, default threshold (`tanh(1.0)`). Already measured:
  `ade_0_2s` **0.8563 [0.7282, 1.0035]**.
- **Arm B (treatment):** identical in every respect except `--route-thr 0.35`.
- Same checkpoint (step 29,999), same parity val cache, same `--episodes 40 --stride 8`, same
  anchors, same seed. **Same 881 windows / 40 episodes**, verified by identical `eid` sequence and
  bit-identical ground truth before any comparison is computed.
- **Estimator: paired episode-cluster bootstrap** (`taniteval.ci`, B=2000, seed 0).
  `overlapping_holdout_se` is not used. The paired form is required because the arms share windows.
- **Primary:** paired Δ `ade_0_2s` (B − A). **Secondary, reported either way:** the paired Frenet
  split (`long_abs_2s_m`, `lat_abs_2s_m`) and `miss_at_2m`.

## Both outcomes, committed in advance

- **OUTCOME A — the fix transfers.** Paired Δ `ade_0_2s` is **negative and CI-separated**. ⇒ route
  quality is on the causal path to trajectory quality; adopt the threshold subject to the held-out
  confirmation still owed (§3.1 of the sweep doc), and re-open whether the produced-goal arm can now
  beat the constant-velocity floor it currently ties.
- **OUTCOME B — it does not transfer.** Paired Δ is **not separated**, or is separated **worse**.
  ⇒ **the head is largely insensitive to the route channel**, and the +0.1251 balanced-accuracy gain
  is real but inert. That would be a genuinely useful negative: it would mean the oracle-vs-produced
  ADE gap is carried by the OTHER goal channels (`vt_band` exact 0.1725, `tspeed_5s` RMSE 4.45 m/s),
  and the next lever is the speed channel, not route. **It would also retire "fix the route head" as
  the program's headline goal-side work item.**

⚠️ **Outcome B is not a failure and will not be reported as one.** Both directions are informative
and both will be published with the same prominence.

⚠️ **Pre-committed guard against a tempting mistake:** a *route*-metric improvement will NOT be
quoted as evidence of a *trajectory* improvement, in either direction. They are separate claims on
separate metrics, and only the paired ADE delta speaks to the second.

⚠️ **Threshold 0.35 was selected on this same evaluation split.** Even under Outcome A this run
cannot validate the *value* — only whether the *direction* transfers to ADE. Held-out confirmation
remains owed before deployment.
