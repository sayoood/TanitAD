# PROPOSED REGISTER ROWS — L0 / L1 / E0

⛔ **PROPOSED, not applied.** This FlyWheel does not own `Project Steering/GOALS_AND_CLAIMS.md`,
`MODEL_REGISTRY.md` or `RETRACTION_LOG.md`. Owner: the **Master Mind**.
All numbers below carry their evidence class, their tier and their artifact path, and every one is
reproducible from `./raw/` at 0 GPU except E0's rollouts (40.8 s on the dev-box 4060).

---

## 1. `D-TAC-LOSS-LOGGED` — the tactical/strategic label terms are now in the trainer's row

| field | value |
|---|---|
| **id** | `D-TAC-LOSS-LOGGED` |
| **status** | **LANDED** (staged, not committed; `agent/arch-inf-20260803`) |
| **class** | instrument |
| **claim** | `stack/scripts/refa_v1_train.py`'s log row now carries `loss_lat_label`, `loss_lon_label`, `loss_route_label` and `loss_label_weighted_share`. Each is **`None`** — never `0.0` — when the model did not emit the term. |
| **why it mattered** | The model computes those terms (`refa_v1.py:1645`) and folds them in with `w_tac_label` (`:1648`) / `w_str_label` (`:1663`), and they are **13.5924 %** (incumbent) / **20.5106 %** (ep2) of the label+feature sum — while **0 rows across all 9 banked `train_log.jsonl`** carried them. A term with no instrument cannot be tuned. |
| **evidence** | MEASURED. Absence: two differently-bound probes (POSIX `grep -c` → 0 exit 1; CPython absolute-path `read().count()` → 0,0,0). Share: `…/2026-09-03-tactical-decoder/raw/intent_probe_{incumbent,ep2}.json` → `loss_share.nav_true.tactical_label_share_of_those_terms_pct`. Row artifact: `raw/l1_train_log_rows.json` (labelled share **0.18101985041464227**, hand computation matches to `1e-12`; unlabelled row: all four `null`). |
| **tier** | n/a (a trainer-contract change) |
| **pinned by** | `stack/tests/test_tac_loss_logging.py` — 8 tests: the ON path, the OFF path (`is None`, and `0.0` rejected by identity), the hand computation, a partly-labelled step, an all-ignored family, the **load-bearing regression** (21 pre-change keys present, same types, bit-identical across two identically-seeded runs, and `set(new) − PRE_CHANGE_KEYS` exactly the four added keys), and the accounting identity `Σ w·L_feat + share·loss == loss` to `1e-6` rel. |
| **discharges** | `PREREG_TACTICAL_DECODER.md` §3's preflight refusal *"`loss_lat_label` / `loss_lon_label` are still absent from the trainer's log row"*. |
| **⛔ not shipped** | Thor's live `refav1-b1-v72-ep3-speed` keeps its launch line. This is for the NEXT launch. |

---

## 2. `D-SEED-GOAL-FIXED` — the planner's seed can now reproduce its own goal, flag-gated

| field | value |
|---|---|
| **id** | `D-SEED-GOAL-FIXED` |
| **status** | **LANDED, DEFAULT OFF** (staged, not committed) |
| **class** | planner / cost |
| **claim** | `RefAV1.plan(goal_time_grid="plan")` re-rolls the imagined goal from the SEED's own action feed, so the canonical seed's cost rollout and the goal rollout are the same forward pass. Agreement goes from **2/64 → 64/64** token pairs and the `TURN` heading excess from **82.506° → exactly 0.000°**, in **both** `COST_TIME_GRIDS`. `"full"` remains the default and is byte-identical to the shipped code. |
| **⚠️ the limit, which must travel with the claim** | **IT BUYS IDENTITY, NOT HORIZON.** Under `"plan"` the goal is the manoeuvre's first `plan_horizon_s` (2.0 s), not the 6.0 s manoeuvre the token names. Making the seed span the token's manoeuvre requires `plan_horizon_s` 2.0 → 6.0 — a **DESIGN decision** (it changes the optimised window, the search dimensionality, the proposal head's output shape `refa_v1.py:1093`, the returned plan length and every banked comparison) and it is **ESCALATED, not taken**. |
| **the diagnosis, exact** | goal ← `refa_v1.py:1814`, operative indices `[0,3,…,27]`; seed ← `:2014`, operative indices `[0..9]`. The goal's actions at indices 12,15,18,21,24,27 are **not in the seed at all**, so the cause is the **TRUNCATION, not the regrid** and `4139203`'s regrid cannot close it. |
| **rejected repair, recorded in-source so it is not re-proposed** | packing `ctrl[::stride][:plan_steps]` into the seed gives 64/64 under `"dense"` only, and makes the seed a **3× time-compressed control** — the plan is executed on the operative grid at `op_dt` and the coarse→fine re-score rolls `self.operative` with no regrid, so it would drive a 6 s manoeuvre in 2 s. Under `"tactical"` it cannot work at all. |
| **evidence** | MEASURED. BEFORE: `…/2026-09-03-tactical-decoder/raw/seed_goal_mismatch.json`. DEFAULT-UNCHANGED (the deliberate regression): the **unmodified** read-only tool re-run on the patched source → `raw/seed_goal_mismatch_AFTER_L0_default.json`, still 62/64 and still 82.506°. AFTER, on RUNNING `plan()`: `raw/seed_goal_agreement.json` — 64/64 both grids, worst seed-goal term `1.788e-07` / `−2.384e-07` against a float32 cosine resolution of `4.768e-07`. |
| **tier** | n/a (a wiring identity on a tiny random-init rig; the identity is a property of which ACTIONS are fed, not of what the weights know) |
| **pinned by** | `stack/tests/test_seed_goal_agreement.py` — 18 tests, including the deliberate regression (the default must **still** read 2/64 and 82.506°), 64/64 on running `plan()` for both grids, the numerical `1−cos ≤ 4.768e-07`, `plan_level="operative"`, the flag's refusal-by-name, and `test_h_the_repair_buys_identity_not_horizon`. |
| **files** | `stack/tanitad/refs/refa_v1.py` — `GOAL_TIME_GRIDS` (`:202`), `_check_goal_time_grid` (`:205`), `plan(..., goal_time_grid="full")` (`:1852`), the repair at the seed site (`:2014-2041`), `res.goal_time_grid` provenance. |
| **cost** | one extra tactical rollout per plan tick when the flag is ON (`_imagine_tactical_goal` must still run — its `ctrl` **is** the seed). |

---

## 3. `D-TACDEC-E0-GOALSPACE` — E0's outcome row

| field | value |
|---|---|
| **id** | `D-TACDEC-E0-GOALSPACE` |
| **status** | **MEASURED, decided** |
| **hypotheses touched** | `H-REFAV1-COST-SEED-1` **survives**; `H-REFAV1-TAC-DECODER-1` **survives**; `PREREG_TACTICAL_DECODER.md` §10's goal-SPACE branch **does NOT fire**. |
| **the committed branch, verbatim** | §8.2 Q1: *"If the goal-term advantage stays ≤ 0 on a majority of the 25 even at seed/goal identity and in f64, then the goal SPACE — not the decoder, not the metric, not the weights — is what cannot represent a manoeuvre, and the whole decoder-then-cost line is REFUTED."* |
| **the measurement against it** | at seed/goal identity, f64: advantage **> 0 on 25/25** windows, **both** conventions (majority threshold 13). ⇒ **NOT REFUTED.** |
| **§8.4's committed branch, verbatim** | *"If the oracle-goal cost also fails to prefer the turn, the defect is the cost; if it succeeds, the defect is the imagined goal."* |
| **the measurement against it** | with the **ground-truth 6 s field** as the goal, the turn is FURTHER than cv on **22/25** (A) / **24/25** (B), and on **6/9** / **8/9** of the windows where `gt_turn_deg ≥ 5°`. ⇒ **THE DEFECT IS THE COST.** |
| **the size of it (Q2)** | advantage median **1.491e-09** against a `0.05·κ²` charge of **2.240e-04** — a factor **1.5 × 10⁵**. Turn total-cost wins under the shipped `1 − cos`: **0/25 on every arm, both conventions.** The multiplier on both explicit weights that would tip the median window is **3.328e-07**, i.e. `w_kappa` ≈ **1.66e-08** — a deletion of the penalty, not a weight. |
| **⭐ the one configuration that ranks the turn** | **L0 identity + chord `√(2(1−cos))` + `model_action_units="steer"` (conv B): 25/25 total-cost wins at the SHIPPED `w_kappa = 0.05`.** Under conv A the same arm wins 5/25. |
| **⚠️ the n, which must be quoted with it** | **25 windows = 4 episodes (7/7/7/4), and ALL 25 carry the identical token `TURN_R × ADAPT_SPEED_FOR_CURVE`.** Margin chord ÷ charge: **min 1.187, median 1.232**, max 3.813. This is a 4-cluster, one-token result. |
| **S3's verdict** | §7 S3 requires ≥ 13/25 *"both conventions"*. A = 5/25 ⇒ **S3 FAILS AS WRITTEN**, upholding the pre-registered prediction — but the failure is **convention-dependent**, which the pre-registration did not anticipate. The named mechanism (float32 saturation) is also confirmed: the same cosine read in fp32 gives **13/25** positive instead of 25/25. |
| **controls** | K1 identity `3.331e-16` / `2.220e-16`; K2 zero-goal `4.441e-16`; K3 `C_REG` does **not** read identity (`1.428e-06`); K6 banked agreement **25/25 matched on (ep_name, t)**, **25/25 token agrees**, max diff `1.788e-07`/`2.384e-07`. All pass. |
| **evidence** | MEASURED, **T0**. `raw/e0_goalspace_ep2.json`, `raw/e0_goalspace_ep2_rows.json`, `raw/e0_ep2.log`, `raw/run_e0.sh`; tool `tools/e0_goalspace_probe.py`. Checkpoint `C:\Users\Admin\refav1_eval_slice\ckpt_ep2\ckpt.pt` (step 1,000), config `…\config.json`. 40.8 s on the dev-box 4060. |
| **NOT claimed** | anything outside `TURN_R × ADAPT_SPEED_FOR_CURVE` or outside those 4 episodes (**UNVERIFIED**); anything at T1; whether the WM's imagined turn is *correct* (the oracle floor compares an imagination with an adapter-encoded real field — it bounds "can the cost rank a turn against the truth", not "is the turn right"). No driving claim; the four metric families are not reported because no arm reaches G-DRIVE. |

---

## 4. `RETRACTION_LOG.md` — one entry proposed, and it is mine

| field | value |
|---|---|
| **what** | E0's K6 agreement gate reported a **6.2e-06 disagreement** with the banked cost surface on its first pass — the same order as the values being compared, i.e. an apparent refutation of my own instrument. |
| **root cause** | **A JOIN KEY THAT IS NOT A KEY.** I keyed the join on the window index `t` alone; the 25 curved windows take only **six** distinct `t` values across four episodes, so the dict silently compared each window with a **different episode's** row. Matching on `(ep_name, t)` gives agreement to **1.788e-07**, inside the float32 cosine resolution. |
| **class** | same family as the `df` / cgroup / `step_s` scope errors: **a probe that answers a different question than the one asked, and looks like an answer.** Its specific shape — a non-unique join key on a small stratum — is not yet in `CLAUDE.md`. |
| **cost** | one re-analysis, **0 GPU** (the tool has `--from-rows`; the compute was already paid for). |
| **durable fix** | the matcher and the warning are in `tools/e0_goalspace_probe.py`'s `analyse()`, and the tool now also asserts the **decoded (lat, lon) token agrees** on every matched window — a second, independent key. |

---

## 5. Escalations for the Master Mind (integration, not a note in a README)

1. **`plan_horizon_s` 2.0 → 6.0 is now the load-bearing DESIGN decision.** L0 buys identity, not
   horizon. This FlyWheel will not take it unilaterally.
2. **The chord (R29 / L3) is no longer optional and is not weight-neutral.** It is the only metric
   under which any arm ranks a turn at the shipped weights, and §5 requires it to land **jointly
   with L4**.
3. **`D-REFAV1-BOUNDARY-NULL` needs a qualifier.** `model_action_units="steer"` was recorded as
   *"removes a trap, wins nothing"* (`share = 0.0000 [0, 0]` on 140/140). E0 measures the 25/25
   chord win **only** under it (conv A: 5/25). The row is not wrong — it was measured on a different
   quantity — but it must not be quoted as *"steer buys nothing"* after this.
4. **`PREREG_TACTICAL_DECODER.md` §3's preflight is discharged** (L1), so the arms in that document
   are no longer blocked on the log row. **D3 remains a PI / Master-Mind scheduling decision and
   Thor stays untouched until ≥ 2026-09-04 02:00Z.**
5. **S3's wording needs an amendment before it is re-used**, because it is convention-dependent and
   silently assumed convention parity. Amending it *after* seeing the data is not something this
   FlyWheel will do to its own success criterion — it is recorded here for the owner.
