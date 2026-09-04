# Bisecting refcv3 against refc-base — the fan's deficit has an AXIS, and three unrecorded defects

**WP-2, PI priority.** *"You are saying refc-base was better than refcv3 despite the several
improvements we introduced in refcv3, so we need to analyse the differences."*
**Written:** 2026-09-04 · **owner:** Architecture & Inference FlyWheel · **compute:** 0 GPU.

⭐ **READ THE SIBLING FIRST — it owns the anchor-vocabulary root cause and I do not restate it.**
`taniteval/results/2026-09-04-refcv3-smoothness/RESULT.md` (rows `D-REFCV3-SMOOTH1/2/3` in
`GOALS_AND_CLAIMS.md`) established, MEASURED and launch-blocking, that **refcv3 trained on the
SYNTHETIC bootstrap anchor vocabulary** (no `--anchors` passed → `refc.default_anchors`, FPS over
4,096 `synth_anchor_pool` rollouts) while refc-base/XL/small trained on **data-driven**
`refc_anchors_*.pt`; oracle-in-vocabulary 0–2 s **0.9433 m synthetic vs 0.4369 m data-FPS (×2.16)**,
worse than a single straight line; and that **the Alpamayo action space was never applied to the
output**. **This document does not compete with that. It adds the axis, and three defects that
package does not touch.**

**Primary sources:** `Project Steering/MODEL_REGISTRY.md` §4.3 / §4.5 ·
`PREREG_REFC_V3.md` · `REFC_V3_DESIGN.md` · the run's own `config.json` / `metrics.jsonl` / dump
`manifest.json` · `stack/tanitad/refs/{refc_v3,refc,refc_tactical}.py`,
`stack/scripts/{refc_v3_train,refc_train}.py`, `stack/tanitad/models/vocab_v7.py` ·
`taniteval/results/2026-09-04-refcv3-closedloop/raw/V3_refcv3_vs_refc-base_empty.json` ·
`taniteval/results/refcv3-40284-openloop-dump.tar.gz`. **No number here is quoted from prose.**

---

## ⭐ WHAT THIS PASS ADDS, IN SIX LINES

1. ⭐⭐ **THE FAN'S DEFICIT HAS AN AXIS, AND IT IS LONGITUDINAL.** The residual gap between the
   fan's assignment oracle and the trivial floor is **entirely along-track**:
   `oracle_sel − ha` **ALONG +0.0916 [+0.0763, +0.1073] SEPARATED** while
   **LATERAL −0.0193 [−0.0340, −0.0052] SEPARATED in refcv3's favour** (n = 4,823 / 141 eps).
   ⇒ **refcv4's anchor rebuild must spend its budget on the speed/acceleration dimension of the
   vocabulary, not the curvature dimension — the lateral half already beats the floor.**
2. ⛔ **DEFECT A (new, source-confirmed): the tactical→operative port is fed a MIS-INDEXED vector.**
   `refc_v3.py:890` calls `derive_man5_logprobs` — contract `[B,3]×[B,3]`, hard-indexed 0/1/2 —
   on the **8-wide v7 heads**. Four of the five port slots are fed classes with **0.000 % predicted
   mass**; `BRAKE_TO`, `ACCELERATE`, `TURN_L`, `TURN_R` are **structurally unreachable** by the
   decoder's anchor prior, which then **overrides** the core's own healthy kin3 prior.
3. ⛔ **DEFECT B (new): the strategic level was NEVER SUPERVISED.** `--goal-str` is absent from the
   run's argv and `goal_str` appears in **0 of 559** logged training rows while every other
   conditional hierarchy key appears in **559/559**. ⇒ **refcv3 cannot be read as evidence for or
   against the goal cascade it exists to test.**
4. ⛔ **DEFECT C (new): the E9 goal re-rank shipped 2.47× outside its own pre-registered
   admission** (`goal_gate` 0.17444 while `eval_goal2s_err_m` 1.97867 vs `admission_sigma_m` 0.8) —
   **but it is near-inert** (changes the emitted anchor on **5.52 %** of windows; effect on
   assignment-oracle agreement −0.0002 [−0.0071, +0.0066], not separated). Governance defect,
   not a cause. ⭐ **This kills the selection suspect with a number.**
5. ⚠️ **LABEL COVERAGE, measured three ways — and its LOCAL form is refuted.** v7.2 tactical labels
   reach **23.53 %** of training rows (run log) and **23.99 %** of eval windows (dump), against
   refc-base's **100 %** runtime kinematic manoeuvre label. But windows near a label are **not**
   better: DiD **+0.0274 [−0.0135, +0.0726], not separated**. The global form is untested.
6. ⛔ **THE COMPARISON CANNOT ATTRIBUTE, AND THE EXPERIMENT THAT COULD WAS NEVER RUN.** The arms
   share no training contract (§1); `PREREG_REFC_V3.md` §2 registered **v3-H vs v3-F** on the same
   data and says cross-links to the 30 k arms *"never decide the dominance question"*. Only
   `--arm hier` launched, and on a corpus the prereg did not register.

---

## 1 · DOES CORPUS NON-PARITY CONFOUND THE COMPARISON? — establish first, as instructed

**Yes for attribution, no for the observation.** Every cell is from the arm's own `config.json` /
registry row.

| | **refc-base** (`refc-diffusion-base-v21-30k`) | **refcv3** (`refcv3-b1-v72-30k`) |
|---|---|---|
| corpus | `physicalai-train-e438721ae894`, **2,376 eps / 406,099 windows**, skip-hash `f09e44db` (registry §4.3) | ⛔ **B1 `physicalai-b1-w120-256x640cyl`, 4,572 train clips** (§4.5; `v2_parity.parity false`, `checked false`, `corpus_key null`, `require_parity false`) |
| image | **256×256**, in_channels 9 | **256×640 cylindrical, 120° FOV** (`config.json` `image_hw`); encoder grid 8×20 = **160 KV tokens** vs 8×8 = 64 (`refc.py:291-293`, non-square-safe) |
| plan horizon | **(5,10,15,20) = 2.0 s, 4 slots** | **(5,10,15,20,30,40,50,60) = 6.0 s, 8 slots** (`refc_v3.py:120`) |
| anchors | 128 × 4 × 2, **data-driven** `refc_anchors_base128.pt` | 128 × 8 × 2, ⛔ **synthetic default** (sibling `D-REFCV3-SMOOTH1`) |
| route label | v2.1 per window, **80.05 % judgeable** (registry §4.3) | v7.2 per-**clip**; **75.10 %** of eval windows carry one (probe P2) |
| tactical label | kinematic 5-way, per window at runtime, **100 %** (`refc_train.py:497`, CE `:536`) | v7.2 **8×8** joined per clip; **23.99 %** of eval windows, **23.53 %** of training rows |
| nav | v1 `refb_labels.nav_command` | ⚠️ **ORACLE** v7.2 token, `allow_oracle_nav True`, added **mid-run at step 17,000** |
| eval | 881 windows / 40 val eps | 4,823 windows / 141 eps of the v7.2 split |

⇒ **A cross-arm delta between these two arms cannot name a design change.** The registry says so
(*"NON-PARITY … not cross-arm comparable; only the margin over the shared `ha0` floor is
admissible"*) and so does the closed-loop panel's limitation **L1**.

**What survives.** The closed-loop pairing is an **identity** — one render pass drove all three
arms, the 199-frame `frame_md5` list is byte-identical (digest
`9a058d5d8cdf892d61cfde616f1ecf85`), and patch neutrality is **bit-exact** (450/450 steps, max
|Δ| 0.0; 15/15 paired metrics `delta 0.0`, zero-width CIs). *"refc-base scored better on these two
metrics"* is a real measurement of two arms **as deployed**. Only *"because of change X"* is blocked.

⇒ **So this bisection runs on WITHIN-ARM evidence**: refcv3 against its own trivial floor, on its
own corpus, at **n = 4,823 windows / 141 episodes** — 11× the closed-loop panel's n and free of the
corpus confound.

⚠️ **Two governance facts, flagged not adjudicated.** (i) `PREREG_REFC_V3.md` §8 registered the
train corpus as `physicalai-train-e438721ae894`, *"window-set bit-identical to the canonical
406,099"*; the run used B1, with no amendment, in a document requiring *"a written amendment BEFORE
unblinding"*. (ii) The registered **v3-F flat control was never trained**, so no hierarchy claim
rests on refcv3 either way.

---

## 2 · THE CHANGE LIST — every substantive difference, with its source

| # | change | refc-base | refcv3 | source |
|---|---|---|---|---|
| C1 | plan horizon / slots | 2.0 s, 4 | **6.0 s, 8** | `refc_v3.py:120`; registry §4.5 caveat 3 |
| C2 | **anchor vocabulary** | **data-driven** FPS over real GT | ⛔ **synthetic** `synth_anchor_pool` default | sibling `D-REFCV3-SMOOTH1` (3 probes) |
| C3 | anchor budget | 128 over 4 slots | 128 over **8** slots (unchanged budget) | `refc_v3.py:421`; dump `manifest.json` |
| C4 | trajectory loss | per-valid-slot L1 over 4 slots | per-valid-slot L1 over **8**, **unweighted in metres** | `refc_v3_train.py:509-511` |
| C5 | anchor CE target | GT-nearest anchor over 4 slots | GT-nearest anchor over **all 8 slots** | `refc_v3_train.py:502-508`; `refcv3_arm.py:1052-1056` |
| C6 | manoeuvre head | native 5-way `maneuver_head` | **`factored_maneuver=True`** kin3 core head + z_tac **8×8** v7 heads | `refc.py:451` (default `False`) vs `refc_v3.py:437` |
| C7 | **tactical→anchor edge** | own 5-way posterior → `maneuver_to_anchor` | z_tac → `derive_man5_logprobs` → port → `invert_man5` → two rank-3 grafts, **overriding the core's priors** | `refc.py:2113-2136`; `refc_v3.py:890` — **DEFECT A, §4.1** |
| C8 | new hierarchy modules | — | PhiTac 1,757,440 · `tac_latent_proj` 262,656 · `gstr_cond` 66,816 · `nav_inject` 50,176 · `tac_heads` 14,364 · `scorer` 1,156 · `str_goal_head` 771 | `config.json` `param_breakdown` |
| C9 | new loss terms | — | `goal_tac` **0.5** · `sel_v3` **1.0** · (`goal_str` 0.1 — **never fired**, §4.2) | `refc_v3_train.py:93-99` |
| C10 | selection | trained scorer | + **E9 goal-distance re-rank**, `goal_gate` **0.17444** live | `refc_v3.py:995-1027` — **DEFECT C, §4.3** |
| C11 | strategic width | `d_ctx` 64 | **`d_ctx` 256** (PI rebalance 2026-09-02) | `refc_v3.py:436`; `str_goal_head 771 = 256·3+3` |
| C12 | image geometry / corpus / labels / nav | §1 | §1 | §1 |
| C13 | steps | 29,999 | **40,284**, resumed composite, recipe changed at 17,000 and 18,500 | registry §4.5 caveat 1 |
| C14 | ⛔ **NOT changed** | free-XY waypoints | **free-XY waypoints** | sibling `D-REFCV3-SMOOTH2` |
| C15 | **reach clamp** | ⛔ **OFF** (`refc.py:563` default) | **ON** (`refc_v3.py:448`) but at a band derived from the 6 s horizon: **±15.0 m/s** instead of ±5.0; kills **37.1 %** where the same rule kills **77.28 %** on a 2 s fan | `refc.py:652`, `:563`; §4.5 |
| C16 | **offset head** | unbounded raw `nn.Linear` | **unchanged** — decoder config bit-identical to refc-base's defaults | `refc.py:1114`/`:1233`/`:1400` vs `refc_v3.py:418-420` |

---

## 3 · ⭐⭐ THE AXIS — the contribution of this pass

### 3.1 Positive control first

My probe (`raw/bisect_probe.py`, numpy only, on the in-repo banked dump) reproduces the registry
§4.5 row **exactly**: `os` **0.4419 [0.4098, 0.4743]**, `ha` **0.2996**, `os − ha`
**+0.1423 [+0.1187, +0.1658]**. *If that had not matched, the probe would be wrong, not the model.*
Estimator throughout: paired episode-cluster bootstrap over the 141 eval episodes, B = 2000, seed 0.
⛔ Never `overlapping_holdout_se`.

### 3.2 The deficit is longitudinal magnitude — 92.2 % of it

| decomposition of `os − ha` | Δ [95 %] | |
|---|---|---|
| **ALONG-TRACK `\|dx\|`** | **+0.1682 [+0.1441, +0.1923]** | ⛔ **SEPARATED — refcv3 worse** |
| **LATERAL `\|dy\|`** | −0.0142 [−0.0288, +0.0004] | not separated, refcv3 directionally better |
| ALONG vs the CV floor `ha0` | −0.0674 [−0.0923, −0.0415] | refcv3 wins… |
| LATERAL vs `ha0` | −0.2048 [−0.2686, −0.1500] | …but wins **3× more** laterally |

Per horizon the along-track deficit is present **from the first slot** and grows —
0.5 s **+0.0193**, 1 s **+0.0991**, 1.5 s **+0.2092**, 2 s **+0.3453**, all separated — while
laterally refcv3 is *better* at 2 s (**−0.0589 [−0.0960, −0.0219]**, separated). ⇒ This is a
**speed-estimation** failure, not a late-divergence failure.

**And the class is right while the magnitude is wrong.** Grouped by the model's own longitudinal
decision, the plan's net Δv tracks GT closely — `BRAKE_TO` −0.3853 (GT −0.4088), `ACCELERATE`
+0.4746 (GT +0.4423), spread −0.8600 (GT −0.8511); laterally `TURN_R` y −1.0373 (GT −1.0635),
`NUDGE_L` +1.3005 (GT +1.2277). Yet
**`corr(Δv_plan, Δv_GT) = 0.5552` against `corr(Δv_hold-action, Δv_GT) = 0.8506`**, while
`corr(y_plan, y_GT) = 0.9566` **beats** hold-action's 0.9073. **The tactical decision reaches the
plan; the longitudinal amount does not.**

### 3.3 ⭐⭐ …and the FAN carries it — with the lateral half already winning

| | Δ vs hold-action [95 %] | |
|---|---|---|
| `oracle_sel − ha`, **ADE** | **+0.0672 [+0.0496, +0.0849]** | ⛔ SEPARATED |
| `oracle_sel − ha`, **ALONG** | **+0.0916 [+0.0763, +0.1073]** | ⛔ SEPARATED |
| `oracle_sel − ha`, **LATERAL** | **−0.0193 [−0.0340, −0.0052]** | ✅ SEPARATED — **the fan wins** |

Already at **0.5 s** the oracle's along-track error (0.0614 m) exceeds hold-action's (0.0484 m).
⚠️ Scope: `oracle_sel` is the **refined** trajectory of the raw anchor nearest GT over all 8 slots
(`refcv3_arm.py:1052-1058`) — the *assignment* oracle, i.e. the quantity the trainer optimises, not
the fan's true minimum.

⇒ **This is the actionable half of the sibling's `D-REFCV3-SMOOTH1`.** They measured *that* the
synthetic vocabulary is ×2.16 worse than a data-driven one; this measures *which axis survives into
the deployed arm*: after the offset head has done its work, the vocabulary's residual failure is
**purely longitudinal**, and its lateral half is already better than the floor. Two probes from
different directions, one conclusion, and a concrete instruction for the rebuild.

### 3.4 The two separated CLOSED-LOOP metrics, re-read

`manoeuvre_plan_eq_logged` −0.0736 [−0.1309, −0.0244] separates against refcv3 while
`manoeuvre_head_eq_logged` (+0.0138) does **not** — so **the head is fine and only the PLAN is
worse**, and the class it cannot emit is longitudinal: on a clip whose logged share is
`lane_keep 0.3655 / accelerate 0.2138 / brake_stop 0.4207`, refcv3's plan emits `brake_stop`
**1.38 %** and `accelerate` 45.5 %. Consistent with §3.2–3.3.

⛔ **The yaw-rate separation (+0.0074 [+0.0027, +0.0119]) I could NOT explain, and I tested the
obvious story and it failed.** `ω = κ·v` invites *"κ is better, v is worse, so ω is a speed
error"*. Rider R0 (`raw/yawrate_decomp.out`, 450 steps/arm from the banked rollouts) writes
`Δω = κ_gt·Δv + v_gt·Δκ`: it reproduces each arm's yaw-rate **level** (0.02011 / 0.01301 vs the
panel's 0.0206 / 0.0132 — a control that it is on the right quantity), and then the
**curvature-driven term dominates for both arms** (refcv3 0.02011 vs speed-driven 0.00968) and is
where refcv3 loses most (**+0.004369** vs **+0.002526**).
⚠️ **And it hits an unresolved scope conflict:** my `|κ_plan − κ_gt|` says refcv3 is **worse**
(0.001435 vs 0.001168) while the panel's `curvature_err_1pm` says **better** (0.00110 vs 0.00160).
They are **different objects** — mine is the single commanded curvature at the control tick against
a finite-differenced GT curvature, `cl_metrics.py`'s is computed along the plan polyline. **Neither
refutes the other; mine must not be quoted as a correction.** ⇒ **Evidence class UNVERIFIED. The
one separated LATERAL regression has no mechanism, and no v4 decision may rest on it.**

---

## 4 · THREE DEFECTS THE SMOOTHNESS PACKAGE DOES NOT COVER

*(Verified absent there by content search: `derive_man5`, `vocab_version`, `goal_gate`,
`admission`, `sel_idx_base`, `lat_label`, `strategic` — 0 hits each in its RESULT.md.)*

### 4.1 DEFECT A — the tactical→operative port is fed a MIS-INDEXED vector

**Evidence class: MEASURED (source + the trained checkpoint's own param breakdown + the dump).**

`refc_v3.py:890` calls `tac.derive_man5_logprobs(lat, lon)` on the **z_tac heads**. That function's
contract is `(lat_logits [B, 3], lon_logits [B, 3])` and it hard-indexes positions 0, 1, 2
(`refc_tactical.py:293-320`; `LAT_LANE_KEEP/TURN_LEFT/TURN_RIGHT = range(3)`,
`LON_BRAKE_STOP/STEADY/ACCELERATE = range(3)`).

**The heads are 8-wide.** `tac_vocab_version = "v7.0"` (`config.json` **and** the dump
`manifest.json`, written from the final checkpoint) sizes them from `TACTICAL_LAT_ACTIONS_V7` /
`TACTICAL_LON_ACTIONS_V7`, both **8 classes** (`vocab_v7.py:290,298`). Confirmed arithmetically
against the trained model's own record: `tac_heads = 14,364 = 2·(512·8 + 8) + (512·12 + 12)` —
**exact**; a 3-wide build would read 9,234.

**What the port therefore reads, MEASURED over the 4,823 eval windows (probe P3):**

| port slot | actually fed | predicted mass |
|---|---|---|
| `lane_keep` | `LANE_KEEP` + `CRUISE` | 81.96 % · 43.77 % |
| `turn_left` | **`LANE_CHANGE_L`** | **0.000 %** |
| `turn_right` | **`LANE_CHANGE_R`** | **0.000 %** |
| `accelerate` | `LANE_KEEP` + **`YIELD_MERGE`** | **0.000 %** |
| `brake_stop` | `LANE_KEEP` + **`FOLLOW`** | **0.000 %** |
| ⛔ *never read* | `TURN_L` 1.68 %, `TURN_R` 4.75 %, `NUDGE_L/R` 11.61 %, **`BRAKE_TO` 15.97 %**, **`ACCELERATE` 16.11 %**, `ADAPT_SPEED_FOR_CURVE` 18.85 %, `HOLD` 4.29 %, `CREEP` 1.02 % | — |

And `refc.py:2127-2136` compounds it: with an external port present the core **computes its own
healthy kin3 priors and then discards them** (`reweight = maneuver_logits`;
`lat_prior, lon_prior = invert_man5(maneuver_logits) − log_prior`). The `log_softmax` is also taken
over 8 classes, so the returned 5-vector is not the distribution the docstring promises
(`invert_man5` re-normalises, so only the *semantics* survive as the defect).

⇒ **The hierarchy's only live conditioning edge into the operative anchor prior (E6) transmits a
5-number function of six logits, four of which are dead classes — and the two most
decision-relevant longitudinal classes are structurally unreachable.**

⛔ **Against, stated plainly.** The decision still reaches the plan through the **shared trunk**
(E1, the declared common ancestor) — §3.2's by-class table proves it. So the starved port does
**not** sever tactical→plan; it removes a **direct** path while the **common-ancestor** path
carries the signal. Textbook C120 (`DIRECT_PATH` vs `COMMON_ANCESTOR`). **Fix it in v4; do not bill
the regression to it until E3 bounds it.**

### 4.2 DEFECT B — the strategic level was never supervised

**Evidence class: MEASURED, three probes of different path-binding.**

1. `--goal-str` / `--graft-lan` are **absent** from the run's `argv` (`config.json`).
2. `loss_gstr` is added **only** when `lan is not None` (`refc_v3_train.py:661-666`), and `lan`
   enters the batch only through the LAN dataset wrapper gated on those flags (`:981`, `:1028`).
3. **`goal_str` appears in 0 of 559 logged training rows**, while every other conditional hierarchy
   key (`goal_tac`, `sel_v3`, `goal_gate`, `goal_score_absmean`, `nav_injected`) appears in
   **559/559**.

With `uplink_grad = True` (dump `manifest.json`) `str_goal_head` (771 params) is not detached, so
it trains — but only as a free 3-dimensional bottleneck driven by the *tactical* losses. **It is
not a route goal in any supervised sense.** ⇒ refcv3 is the goal-mediated hierarchy **with its top
level unlabelled**. This does not explain the ADE regression (771 params behind a zero-init FiLM),
but it does mean **refcv3 is not evidence about the 4B goal cascade**, in either direction.

### 4.3 DEFECT C — E9 shipped outside its own admission, and it is near-inert

**Evidence class: MEASURED.** `goal_gate` = **0.17444** (live at inference) while
`eval_goal2s_err_m` = **1.97867 m** against `admission_sigma_m` **0.8** — **2.47× outside**, in the
σ ≥ 1.0 regime the requirement curve MEASURED to be **+0.0943 m worse** than the trained selector.
`PREREG` §5 registered *Branch B* (gate stays 0) as the likelier outcome; the gate opened anyway
because `refc_v3.py:343-348` makes admission *"an eval-read rule, not a runtime switch"*.
⚠️ `eval_goal2s_err_m` **regressed** 1.91019 → 1.97867 between step 30,000 and 40,284 while the
loss fell.

⛔ **But it barely acts:** the graft changes the emitted anchor on **266 / 4,823 = 5.52 %** of
windows, and its effect on assignment-oracle agreement is **−0.0002 [−0.0071, +0.0066]**, not
separated (**−0.0038 [−0.1250, +0.1183]** on the changed windows alone). ⇒ **The selection suspect
is killed with a number.** What remains is a governance defect: a mechanism shipped outside its
pre-registered admission. ⚠️ My read is an anchor-identity proxy, not ADE — **E3** converts it.

### 4.4 Label coverage — measured three ways, LOCAL form refuted

Three unrelated path-bindings agree: the run's own training log (`tac_label_rows` mean
**4.7066 of batch 20 = 23.53 %**, n = 559 steps), the banked eval dump (`lat_label != −100` on
**23.99 %** of 4,823 windows), and the brief's join analysis (**26.92 %**). refc-base's manoeuvre
label is a *runtime kinematic derivation* at **100 %** coverage (`refc_train.py:497`, CE `:536`),
its route label **80.05 %** judgeable. So refcv3's tactical head trains on ~¼ the data, on **8
classes instead of 5**, at the same 0.05 + 0.05 weight — and the log shows it: `lat_tac` mean
1.0333 vs the core's `lat` 0.2420, last row *worse* than the first (0.7257 vs 0.6374), because
~4.7 labelled rows per batch of 20 is mostly sampling noise.

⛔ **The LOCAL form is refuted.** In-band `os − ha` +0.1631 [+0.1235, +0.2068] vs no-label
+0.1357 [+0.1100, +0.1611]; DiD **+0.0274 [−0.0135, +0.0726], not separated.** ⚠️ That tests only
*"windows near a label score better"*. A head made **globally** weaker by 24 % coverage would not
show as a stratum contrast — so this stays live as a hypothesis, not a cause.

### 4.5 ⭐ WHY THE AXIS IS LONGITUDINAL — four mechanisms, all consequences of "6 s + synthetic default"

*(Source facts in this subsection are from the anchor-construction sweep run for this pass; they
corroborate `D-REFCV3-SMOOTH1` from a different direction and were re-derived from source.)*

1. **The vocabulary's longitudinal family is ONE CONSTANT ACCELERATION PER ANCHOR.**
   `refc.py:190-207`: `v0 ~ U[0, 30) m/s`, `yaw_rate ~ U(±0.35) rad/s` **constant per rollout**,
   `accel ~ U(±3) m/s²` **constant per rollout**, integrated at dt 0.1. So every one of the 128
   candidates carries a *single* scalar acceleration for the whole 6 s. A real 2 s speed profile is
   not in that family, and no amount of FPS can put it there. ⇒ **the longitudinal half of the fan
   is structurally one-dimensional while the lateral half is a genuine curvature family** — which
   is exactly the asymmetry §3.3 measures.
2. **FPS is nested in `n`, NEVER in `S`, and its metric is dominated by the 6 s tail.**
   `refc.py:222` flattens the pool to `[M, S·2]` before the greedy selection, so once the slot list
   ran to 60 steps the distance that picks the 128 candidates is dominated by positions that grow
   ~linearly in t (the 6 s coordinate carries ~9× the squared weight of the 2 s one). Near-field
   resolution is spent buying far-field spread. **This is why the 128 anchors are not a 2 s
   vocabulary any more, and it is a source fact, not an inference.**
3. ⭐⭐ **THE ONE FILTER THAT CONSTRAINS THE SPEED PROFILE WAS ADOPTED ON A MEASUREMENT MADE
   OUTSIDE ITS SCOPE, AND ARRIVED 3× TOO WIDE.** refcv3 turned the reach clamp **on**
   (`refc_v3.py:448`; `refc.py:563` defaults it **off**, so refc-base ran without it), citing the
   MEASURED result that it is *"exactly inert on ADE and deletes 72.08 % of the fan"* — a figure
   taken at **`horizon_s = 2.0`**. But `horizon_s` is **derived** from `max(horizons)`
   (`refc.py:652`), so at 6 s the band `a_max · T` opens from **±5.0 m/s to ±15.0 m/s** at the same
   `sel_accel_max 2.5`. MEASURED consequence: it removes **37.1 %** of refcv3's anchors where the
   same rule removes **77.28 %** on a 2 s fan. ⇒ **The only longitudinal feasibility filter in the
   stack was introduced at one third of its intended strength**, and `refc_v3.py:79-82` had warned
   in writing that these band statistics *"MUST NOT be quoted for the 6 s band … re-measure, never
   inherit"*. This is the `df` / Thor-`free` / `step_s` family again: a true measurement applied
   outside its scope, in an anchor-gate costume.
4. **The unbounded offset head then has to override the vocabulary.** `refc.py:1114`/`:1233`/`:1400`
   — a plain `nn.Linear`, returned **raw**: no tanh, no clamp, no scale, and nothing in
   `refc_v3_train.py:634-637` penalises jerk, curvature or feasibility. The sibling measures the
   size of that override at **+0.4948 m** over the best raw anchor (0.9368 → 0.3668). ⚠️ The
   decoder config itself is **bit-identical** to refc-base's defaults (`refc_v3.py:418-420` vs
   `refc.py:320-327`, confirmed in the eval manifest) — so this is not a v3 change, it is a v3
   *load* on an unchanged mechanism.

⚠️ **Braking anchors DO exist** in the pool (`accel ∈ [−3, 0)`), so refcv3's `brake_stop` plan share
of 1.38 % is **not** vocabulary emptiness — it is the model failing to reach them. The sharper read
is the sibling's `Q1b`: on a decelerating window (GT 15.78 → 13.78 m/s) refcv3's own 6 s speed
profile **accelerates** (16.10 → 16.84 at `v0` 15.9).

⚠️ **What is still ESTIMATED, not measured:** the loss-side half of the same story — the trajectory
L1 is a per-valid-slot mean in metres (`refc_v3_train.py:509-511`) and the CE assignment runs over
all eight slots (`:502-508`), so under a speed misestimate ε the along-track error at t is ε·t and
the (0, 2] band carries Σt = 5 of Σt = 23 = **21.7 %** of the along-track gradient against **100 %**
for refc-base. `REFC_V3_DESIGN.md` §3 answered the improvement review's H2 warning with *"the slot
layout is band-balanced by construction — 4 slots in (0, 2] and 4 in (2, 6]"*: **balancing the slot
COUNT does not balance the gradient, because the metre-scale error grows with t.** Mechanism 3
above is the measured version of this concern; the loss-side version is separated only by E2.

⚠️ **And the design's own guard rails did not fire.** The prescribed 6 s data-driven rebuild
(`code/refc_v3_launch_line.txt:34-40`, `READINESS.md:196-200`) never ran — `refcv3_b1_launch.sh:122-132`,
`sup_refcv3.sh:64-75` and the run's own `argv` all carry **no `--anchors`** — and the design's own
P5 command would have **exited with `unrecognized arguments`** anyway (`--pool-cap`, while
`build_refc_anchors.py:112` defines `--max-pool`). The **2 s-prefix comparison report** the design
prescribed (`REFC_V3_DESIGN.md:113-116`, preflight item 6 at `:379`) was **never produced** — two
absence probes of different path-binding. It was specified as *"report, not gate"*, so nothing
refused the launch.

---

## 5 · VERDICT AND THE CHEAPEST DISCRIMINATING EXPERIMENTS

> **VERDICT.** The regression is **in the anchor fan** — the sibling's `D-REFCV3-SMOOTH1`
> (**synthetic instead of data-driven vocabulary**, ×2.16 on oracle-in-vocabulary) is the most
> probable named cause, **and this pass adds its axis and its mechanism: what survives into the
> deployed arm is purely LONGITUDINAL**, because the synthetic pool's longitudinal family is one
> constant acceleration per anchor, FPS spends its resolution on the 6 s tail, and the horizon
> change **tripled the reach band** (±5 → ±15 m/s) so the only longitudinal feasibility filter now
> kills 37.1 % of candidates instead of 77.28 % (§4.5) (`oracle_sel − ha` ALONG **+0.0916 [+0.0763, +0.1073]** SEPARATED; LATERAL
> **−0.0193 [−0.0340, −0.0052]** SEPARATED in refcv3's favour, n = 4,823 / 141 eps). Selection is
> **killed** as a cause (§4.3, 5.52 % of windows, no separated effect); nav is worth 0.0239 m; the
> tactical head is fine and only the plan is wrong (§3.4). **Evidence class:** the axis is MEASURED
> and separated; the vocabulary cause is MEASURED by the sibling; the causal attribution *to refcv3
> vs refc-base* remains **HYPOTHESIS**, because the arms are corpus-confounded and the
> pre-registered flat control was never trained.

**E1 — the axis-aware vocabulary rebuild. 0 GPU for the diagnosis; it is the sibling's R1 with one
addition.** When the 6 s **data-driven** vocabulary is built (`build_refc_anchors.py --data-root`,
`--anchors` actually passed), score oracle-in-vocabulary **decomposed into along vs lateral**, not
just as ADE. **Pre-registered prediction from §3.3:** the lateral half is already at/below the
hold-action floor and the entire gain must appear along-track. If a data-driven rebuild does *not*
move the along-track half, the budget (128 anchors over 8 slots) is binding and E2 becomes
mandatory. **Control that must read a known value:** refc-base's 128 anchors are a bit-exact prefix
of XL's 256 (registry §4.3, `max|A − B[:128]| = 0`) — recompute it; if it fails, the loader is
wrong, not the fan.

**E2 — the horizon lever, isolated. ≈8 h on the IDLE A40. Separates §4.5 from the vocabulary.**
Train v3-H at `horizons = (5,10,15,20)` on B1 — same corpus, labels, seed, steps, anchors, and
everything else — and read it with the *same* instrument (`refcv3_arm.py` → `openloop_suite.py`)
against the *same* `ha` floor on the *same* 4,823 windows. **Within-corpus**, so unlike
refcv3-vs-refc-base it is not corpus-confounded. If `os − ha` closes materially, the horizon is a
first-order lever and refcv4's anchor budget is a decision, not a default. ⚠️ Register both
outcomes first; a null here promotes §4.4's global form.

**E3 — two port ablations on one dump. ≈1 GPU-h on the dev-box RTX 4060 (Thor untouched).**
Re-dump refcv3 with two extra arms: (i) the E6 hook returning `maneuver_logits = None` (one line;
`refc.py:2127` then uses the core's own kin3 `man_logits`) — bounds **DEFECT A** exactly; (ii)
`goal_gate` forced to 0 — converts my anchor-identity proxy for **DEFECT C** into an ADE read.
Paired on the same 4,823 windows, same estimator. Both are free riders on one forward pass.

**R0 (rider) — RUN, INCONCLUSIVE, re-scoped.** Re-do the `Δω = κ_gt·Δv + v_gt·Δκ` decomposition
*inside* `cl_metrics.py` so it uses the instrument's own `curvature_err_1pm` / `yawrate_err_rads`
definitions and its paired estimator, and emit both terms as telemetry (§3.4).

**What NOT to do next.** Do not spend the next arm on the selector, on nav conditioning, or on more
goal heads. **The trajectories on offer are the constraint, and their longitudinal half is where
the budget goes.**

---

## 6 · LIMITATIONS

* **L1 — the cross-arm contrast cannot attribute** (§1). Everything ranking a *design* change is a
  hypothesis about refcv3's own deficit, tested within refcv3.
* **L2 — the closed-loop panel is n = 1 scene / 9 clusters.** ⛔ *"Not separated" is not "no
  difference"*: `ade_0_2s` **+0.2185 [−0.6198, +1.1586]** is **underpowered**, not null
  (RETRACTION_LOG #17 is exactly this error) — the interval is 8.9× the point estimate. On that
  scene **neither arm beats a closed-loop trivial floor** (`refcv3 − cl_ha0`
  +0.1307 [−0.6213, +0.8809]; `refc-base − cl_ha0` −0.0897 [−0.4029, +0.1994]).
* **L3 — `oracle_sel` is the assignment oracle, not the fan's minimum** (§3.3).
* **L4 — different steps** (40,284 vs 29,999); refcv3 is a **resumed composite** whose recipe
  changed at 17,000 and 18,500.
* **L5 — §4.5's 21.7 % is ESTIMATED** under a linear error-growth model, not measured.
* **L6 — one seed, no replicate**, on either arm.
* **L7 — the `config.json` I read lacks `--u8-batches`**, so it is the step-17,000 relaunch's stamp,
  not the final one. Every fact taken from it (`tac_vocab_version`, `image_hw`, `horizons`,
  `param_breakdown`) is independently confirmed by the dump's `manifest.json`, written from the
  **final** checkpoint.
* **L8 — DEFECT A's effect size is UNMEASURED.** The defect is source-certain; its cost is not.

---

## 7 · DELIVERABLE MANIFEST

| artifact | where | state |
|---|---|---|
| this analysis | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-04-refcv3-vs-refcbase-bisect/RESULT.md` | staged |
| the 0-GPU probe (reusable, numpy-only, runs off the in-repo banked dump) | `repo:…/raw/bisect_probe.py` | staged |
| its verbatim output | `repo:…/raw/bisect_probe.out` | staged |
| R0 yaw-rate decomposition + its INCONCLUSIVE verdict | `repo:…/raw/yawrate_decomp.out` | staged |

Nothing lives in only one place: every input is in-repo
(`taniteval/results/refcv3-40284-openloop-dump.tar.gz`,
`taniteval/results/2026-09-04-refcv3-closedloop/raw/*`,
`stack/experiments/alpasim-gsplat/results/closedloop-hq-render/rollouts/HQ_refc-base_empty.json`).

**⛔ ESCALATION — needs a decision before refcv4 freezes, not a footnote.**
1. **DEFECT A** (`refc_v3.py:890`, 8-wide heads through a 3-wide contract) — one-line fix or a
   deliberate re-map; either way it must be a registered v4 lever, not a silent change.
2. **DEFECT B** — refcv4 either passes `--goal-str` or drops the strategic head; shipping an
   unsupervised goal head again makes the next run un-interpretable too.
3. **DEFECT C** — make the admission gate a **runtime** switch, or record in the run config that a
   mechanism is live outside its pre-registered admission.
4. **The v3-H/v3-F pair was never run**, and the launch deviated from `PREREG_REFC_V3.md` §8's
   registered corpus without an amendment (§1). No hierarchy claim may cite refcv3.
5. **E1's axis split** should be added to the sibling's R1 before the vocabulary is rebuilt — it
   says where the anchor budget goes.
