# AUDIT R4 — does v7F's tactical layer condition the operative plan?

**2026-09-27 · TrainingFlyWheel (P4) · read at origin tip `c36b6ddd` (read-only snapshot
`C:/Users/Admin/tipsnap/c36b6ddd/`; every `file:line` below is a TIP line number, re-verified by
exact-string search, `raw/r4_audit_probe.py` + `raw/r4_audit_probe.out.json`).**

PI, 2026-09-27 (verbatim): *"All our tactical labels must be used to train the tactical layer to
estimate and choose the right tactical behaviors and goals which MUST condition the operative
planning."*

## Verdict: **PARTIAL** — the tactical GOAL reaches the plan, the tactical BEHAVIOURS do not

| tactical output | reaches the emitted 6 s plan? | evidence |
|---|---|---|
| goal posterior `g_tac` (22 v7.0 tokens, incl. `SPEED_BAND`) + its 8 arg slots | ✅ **PRESENT** (soft, end-to-end) | path §1; MEASURED §4: forcing a different goal token moves the plan 0.0592 m |
| behaviour posteriors `a_lat` × `a_lon` (the LAT/LON decisions) | ⛔ **ABSENT** | path §2; MEASURED §4: forcing a different LAT or LON behaviour moves the plan **exactly 0.0** while the decision itself moved by 1.0 |
| goal point (`anchor_head`, the selector's `ĝ`) | selection only, both default OFF | `v6.py:5801`, `:5820` read `e_g_tac` |
| "speed band" | only as the `SPEED_BAND` token's probability + the unsupervised arg slots inside `e_g_tac` | `V72_ARGS_SUPERVISED = False` (`train_v6_staged.py:2409`) |
| `--tac-goal-cond` | **not tactical→operative**: it is the STRATEGIC→TACTICAL-dynamics port | §3 |

⇒ **Built:** `--tac-op-cond {off,detached,e2e}` (default `off`, byte-identical) — the behaviour
posteriors now condition the operative plan through a zero-init port. See `RESULT.md`.

## 1. The goal path (PRESENT) — every hop

1. `z_tac, z_tac_tgt = self.uplink_tac(z_op, own_tac)` — `v6.py:5916`; inside, `src = self._cut(z_op, cfg.isolate_uplink)` (`:5562`) and `online = self.adapter_tac(x)` (`:5573`): the tactical latent is a vision-derived function of the operative latent, stop-gradded below.
2. Planner cut: `cut = cfg.isolate_planner_from_encoder` (`:5927`), `z_tac_p = self._cut(z_tac, cut)` (`:5929`).
3. `g_tac = self.goal_head_tac(z_tac_p, cond=e_g_str)` — `:5938` (`GoalHead.forward`: `logits = self.type_head(h)` `:2668`, then `probs` = softmax, `args` = `arg_head`).
4. `e_g_tac = self._encode_goal(self.cond_op, g_tac)` — `:5957` (factored variant `:5954`); `_encode_goal` (`:5633`) → `GoalVocabulary.encode` = LayerNorm(`probs @ table.weight` (`:2546`) + `arg_proj(args)`). `cond_op` is built at `:5142` over the SAME `vocab_tac` object the head emits into.
5. `plan = self.emit(z_plan, e_g_tac, v0, roll_ctx=roll_ctx)` — `:6035`.
6. Inside `emit` (`:5736`): `base = self.plan_proj(z_op)[:, None, :]` (`:5773`), `feat = base + self.cand_queries.weight[None]` (`:5774`), `g = g_tac_embed[:, None, :].expand(...)` (`:5775`), `feat = torch.cat([feat, g.to(feat.dtype)], dim=-1)` (`:5776`), `a_ctl, kappa, wp = self.emission(feat, v0)` (`:5778`) — the emission's input width is `d_plan_feat + d_goal_embed` (`:5182`).

So the emitted plan = f(`z_op` [vision], `e_g_tac` [tactical goal], `v0`). The same `e_g_tac` also
feeds the diffusion proposals (`:5783`), the anchor head (`:5801`) and the selector (`:5820`),
and — detached — the operative WORLD-MODEL predictor as `intent` (`:5997`, seam `:6019`), which is
imagination (`P_O | g_tac`), not the emitted plan.

**Training path (no teacher forcing).** `v6_loss_step` runs the shared forward
(`train_v6_staged.py:4253`) and the planner loss reads `out["plan"]["waypoints"]` (`:4575`,
`terms["plan"] = w.lambda_plan * lp` `:4594`) — i.e. the plan emitted from the model's OWN soft goal
posterior. S-T trains `("layer_tac", "planner")` (`v6.py:4569`) with λ_plan 1.0
(`train_v6_staged.py:596`). **MEASURED** (§4): the plan loss's gradient reaches 13 `layer_tac`
tensors (`goal_head_tac.*`, `vocab_tac.*`) and the planner — and **13 `layer_str` tensors**
(`goal_head_str.*`, `vocab_str.*`) through `goal_head_tac(..., cond=e_g_str)` (`:5938`, not
detached). `layer_str` is frozen in S-T; it would train in S-J.

## 2. The behaviour path (ABSENT)

* `a_lat = self.act_head_lat(z_tac_p)` / `a_lon = self.act_head_lon(z_tac_p)` — `v6.py:5939-5940`.
* `e_a_tac = torch.cat([...vocab_a_lat.encode..., ...vocab_a_lon.encode...])` — `:5958`.
* Its ONLY consumer: `g_cond_tac = e_a_tac` (`:5984`; `+ cond_tac_dyn(...)` `:5986`; `+ nav` `:5988`)
  → `zh_tac = self.predictor_tac(z_tac, g_cond_tac)` (`:5989`) — the TACTICAL world model.
* The trainer reads `out["a_lat"]`/`out["a_lon"]` only for the T5 plan-switch-rate LOG
  (`train_v6_staged.py:4694`); the v7.2 tactical label keys land in the batch unread
  (`:2413-2415`: "a tactical CE on `out["a_lat"]` / `out["a_lon"]` is a pre-registered follow-up").
* No path from `a_lat`/`a_lon`/`e_a_tac` into `emit`. The plan-loss gradient never reaches
  `act_head_lat`/`act_head_lon` (MEASURED, §4).

## 3. What `--tac-goal-cond` actually wires

`train_v6_staged.py:9434` (help: "build the g_str->P_T conditioning port"), mapped at `:5309`;
`V6Config.tac_goal_cond` `v6.py:4043`; built at `:5288` (`cond_tac_dyn = nn.Linear(d_goal_embed,
2*d_goal_embed)`, zero-init), used at `:5986`: `g_cond_tac = e_a_tac + self.cond_tac_dyn(self._cut(e_g_str, cut))`.
⇒ the STRATEGIC goal conditions the TACTICAL dynamics predictor. It does not touch the operative
plan. The brief's reading is **confirmed**.

## 4. MEASURED — counterfactual on the tip source (`raw/r4_audit_probe.py`, CPU, seed 0, toy geometry)

The emission's final Linear is zero-init (`train_v58f_unicycle_head.py:205-206`, the CV warm
start), so at init NOTHING upstream moves the plan; the probe wakes it (random N(0, 0.5)) first.

| lever (same batch, same weights) | max \|Δ waypoints\| |
|---|---|
| goal token 0 → 7 (`goal_head_tac` forced one-hot) | **0.0592 m** |
| LAT behaviour 0 → 1 (`act_head_lat` forced one-hot) — decision moved by **1.0** | **0.0** |
| LON behaviour 0 → 7 (`act_head_lon`) | **0.0** |
| goal token 0 → 7 **at init** (emission zero) | 0.0 (dead until the emission trains — by design) |

Plan-loss gradient reach (`torch.autograd.grad`, allow_unused): `goal_head_tac.*` + `vocab_tac.*`
(13), `layer_str` (13), planner (7); `act_head_lat`/`act_head_lon`: **none**.

## 5. Observations that bear on R4 (not fixed here)

* **No tactical label loss at the tip (R3).** `g_tac` is therefore a latent code shaped only by
  the plan loss; "choose the right tactical behaviours and goals" is not met until R3 lands (another
  agent). The behaviour heads are trained only by the tactical-dynamics t1 loss.
* **R6 is "untrained", not "off".** `goal_head_tac` reads `cond=e_g_str` (`:5938`) from the
  strategic head; with S-S skipped that head keeps its initial weights, so a fixed random function of
  `z_str` still conditions the tactical goal, and the plan loss's gradient reaches `layer_str`
  (frozen in S-T). A literally-off strategic layer would need `e_g_str` zeroed or bypassed — a
  decision for the R6 owner.
* **The goal path is soft only.** No hard (argmax) tactical decision exists at inference; the plan
  is conditioned on the posterior embedding. A hard-choice inference mode is not built.

## 6. The build, and what it is computed from (the PI 2026-08-03 disjointness rule)

`--tac-op-cond` → `V6Stack.tac_op_port = Linear(2*d_goal_embed → d_goal_embed)`, zero-init, group
`planner`, added as `e_plan = e_g_tac + tac_op_port(e_a_tac)` before `emit` (the fan, the anchor
head and the selector all read `e_plan`, information-matched). **Computed from** `e_a_tac` = the
behaviour posteriors of `act_head_lat`/`act_head_lon` applied to `z_tac_p` — the vision-derived
tactical latent (plus, when `--max-speed-input-v6` is on, the PI-authorised max-speed input).
**No situation-classifier output enters it in any form.** `detached` = the plan loss trains the
port only (the F-1 downward-port rule); `e2e` = it also trains the behaviour heads. Both are proved
live by counterfactual and by gradient reach, and OFF is proved bit-identical to the tip
(`tests/test_v7f_r1r4.py`, 14/14 mutants killed — `RESULT.md`).
