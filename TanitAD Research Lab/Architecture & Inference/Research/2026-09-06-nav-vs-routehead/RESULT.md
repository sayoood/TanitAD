# RESULT — D-NAVROUTE-1: the nav command's real content, and what the planner is actually following

`Work package: TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-nav-vs-routehead/`
`Owner: nav-vs-routehead agent. Date: 2026-09-06. Pre-registration: PREREG.md (written before any arm was scored).`

**Evidence class: MEASURED (ours)** unless a row says otherwise. **Tier: T1** (self-action OPEN
loop, per the 2026-09-02 ruling) for every model number. **Estimator:** paired episode-cluster
bootstrap. ⛔ No `overlapping_holdout_se` anywhere in this file.

---

## 0. The two answers, in one place

**P0 — does the nav command carry time and distance?**
**YES in the label record, NO to the model.** `nav_command` carries `args = {distance_m, time_s}`,
written by `s2_geom_emit_v7.nav_command()` and declared by `vocab_v7.NAV_ARG_SLOTS`. But
`NAV_FOLLOW_ROAD` carries `args: {}` — **empty on 2,897/2,897 train and 96/96 eval records** — and
the refcv3/refcv4b consumer **reads only the token and never the args**. The deployed model sees a
**bare 3-way categorical with zero range and zero time**.

**P1 — is the planner following the commanded route, its own predicted route, or neither?**
⭐ **Its OWN predicted route. OUTCOME (a) — the PI is right.** Perturbing the cascade's strategic
goal `g_str` moves the plan **3.3130 m [2.8950, 3.7743]** against a **0.0001 m** replicate floor —
**1.74x** the effect of removing the nav command entirely (1.9006 m [1.3428, 2.5020], non-overlapping
intervals) — while **inverting the commanded route changes the plan by EXACTLY ZERO on 69 % of
windows**. ⛔ But the head in the PI's video is **not** the head that steers: that one
(`route_logits`) is a categorical aux the arm's own config **disconnects** (`graft_route: false`).
The head that steers is `g_str` — and **`g_str` never points left on any of 4,823 windows.**

---

## 1. P0 — the nav command's exact content, from source

### 1.1 Type and cardinality

`nav_command` is a per-clip record `{token, args, provenance, oracle?}`.

| field | value | source |
|---|---|---|
| token | **3-way categorical** — `NAV_FOLLOW_ROAD`, `NAV_TURN_L`, `NAV_TURN_R` | `vocab_v7.NAV_COMMAND_TOKENS` |
| args | `distance_m`, `time_s` | `vocab_v7.NAV_ARG_SLOTS` |
| provenance | `nav-system` \| `ego-future` | `vocab_v7.NAV_PROVENANCE` |

MEASURED over the shipped v7.2 blobs (`s2_labels_v7.2_{train,eval}.jsonl.gz`,
`schema_version s2-geom-v7`, `vocab v7`):

| split | n | FOLLOW_ROAD | TURN_R | TURN_L | provenance |
|---|---|---|---|---|---|
| train | 4,572 | 2,897 | 864 | 811 | `ego-future` 4,572/4,572 |
| eval | 147 | 96 | 38 | 13 | `ego-future` 147/147 |

⚠️ The `oracle` **flag** is absent on 508 train / 21 eval records while `provenance` is present on
all — so a guard keyed on the flag lets those through as non-oracle. Key on `provenance`.
*(This reproduces the correction already recorded in `SPEC_V7_LABELS_CONSUMER.md` §1.)*

### 1.2 ⭐ Time and distance DO travel with it — but only on TURNS

`s2_geom_emit_v7.nav_command()` emits, for a turn:

```
{"token": f"NAV_TURN_{side}",
 "args": {"distance_m": _arc_to(poses, key, nxt[0], hz), "time_s": nxt[0]},
 "provenance": "ego-future", "oracle": True}
```

and for everything else `{"token": "NAV_FOLLOW_ROAD", "args": {}, ...}` — an **empty args dict**.

**How far is the next nav command?** MEASURED, over records that carry args:

| split | token | n | distance_m (p25 / median / p75 / max) | time_s (p25 / median / p75 / max) |
|---|---|---|---|---|
| train | `NAV_TURN_L` | 811 | 2.8 / **27.3** / 94.4 / 623.8 | 1.0 / **7.2** / 17.5 / 33.5 |
| train | `NAV_TURN_R` | 864 | 3.0 / **36.6** / 127.0 / 690.3 | 1.0 / **7.4** / 17.3 / 33.6 |
| eval | `NAV_TURN_L` | 13 | 13.6 / **99.1** / 228.2 / 588.7 | 2.9 / **14.7** / 26.1 / 30.7 |
| eval | `NAV_TURN_R` | 38 | 18.2 / **48.3** / 123.4 / 376.7 | 4.1 / **12.4** / 17.2 / 33.0 |

⚠️ **As a consumer actually sees it**, `_args_for_window` defaults a missing slot to `0.0`, so
**69.97 % of train records (3,199/4,572) and 68.03 % of eval (100/147) present `distance_m = 0.0`
and `time_s = 0.0`** — the FOLLOW_ROAD majority plus a handful of turns already underway.

### 1.3 ⛔⛔ THE ACTIONABLE FINDING — the field exists in the record and is NEVER FED to refcv4b

There are **three** consumers of `nav_command` in the tree and they do **not** agree:

| consumer | token | `distance_m` / `time_s` | evidence |
|---|---|---|---|
| **v6 / v7f** (`train_v6_staged.py` -> `v6.py` -> `NavConditioner`) | fed | ⭐ **FED** — `nav_args [B,2]` through a shared `arg_proj = nn.Linear(2, d_embed)`, injected at all three layers | `v6.py` calls `self.nav(nav_token, nav_args, <layer>)` for operative/tactical/strategic |
| **refav1** (`refav1_loader.py`) | fed | ⛔ **DISCARDED** — `ids, _args = em(eis)`; the args are bound to `_args` and never used again (`grep nav_args` = 0 hits in that file) | `refav1_loader.py` |
| **refcv3 / refcv4b** (`refc_v3_train.py`) | fed | ⛔ **NEVER READ** — the join takes `nav.get("token")` only, maps it through `NAV_TOKEN_TO_LEGACY`, and stores an integer: `item["nav_cmd"] = torch.tensor(nav_idx, dtype=torch.long)` | `refc_v3_train.py`, `enable_nav_from_v7` / `__getitem__` |

**refcv4b — the arm every reference number in this programme's nav work comes from — receives a
scalar index into `NAV_COMMANDS = ("follow","left","right","straight")` (4 rows, 3 live). No range.
No time.**

⭐ **Why this is the load-bearing design fact.** The programme has already MEASURED that a
**bearing** — direction with range stripped — is separated **WORSE by +2.3632 m**, while a goal
point **carrying range** recovers **57.1 %** of the longitudinal selection ceiling; and an ORACLE
3-way command's best possible decoding of the lateral anchor is *"go straight"* for all three
classes (**-0.7842 turn accuracy**, zero turns). **refcv4b's nav input is exactly that stripped
bearing.** Its measured near-zero nav effect is what the vocabulary predicts, not a surprise —
and the fix is already implemented one directory away, in `NavConditioner`.

---

## 2. P1 — the discriminator

### 2.1 ⛔ The two candidate "route heads" are different objects

| # | object | shape | can it reach the plan? |
|---|---|---|---|
| **R1** `route_head` -> `route_logits` | `Linear(feat, N_ROUTE=3)`, a **categorical** (left/straight/right); `N_ROUTE = 3  # route-heading aux` | ⛔ **NO.** `route_prior = log_softmax(route_logits) if cfg.graft_route else None`, and the arm's **own banked config records `"graft_route": false`** (verified in both `refcv4b_navflip.json` and `refcv4b_t1.json`). `route_to_anchor` is only constructed under the same flag. |
| **R2** `str_goal_head` -> `g_str` | `cat([unit_bearing(2), tanh(dist_pref)(1)])` — **geometric** | via a **zero-init FiLM** into `z_tac`, and `target_latent` into the decoder |

⭐ **A probability like the video's "RIGHT 0.68 -> 0.73" can only come from R1** — R2 is a unit
bearing plus a tanh scalar and has no softmax. **So the head the PI's video displays is the one
the arm's own config disconnects from the plan.**

### 2.2 The disagreement contingency table (zero GPU, banked dump)

n = **4,823 windows / 141 episodes**, `nav_valid` on all 4,823. Deadbands pre-registered at
`tau_y = 1.0 m`, `tau_g = 0.10`.

⭐ **Control that must read a known value — PASSES.** Orientation: mean GT terminal `y` is
**+2.8608 m** under `nav=LEFT` (n=342) and **-4.1898 m** under `nav=RIGHT` (n=1,002). `+y` is LEFT,
so every direction in this table is meaningful.

**R1 (core route head) vs the command, on DISAGREEMENT windows — n = 1,908 over 96 episodes:**

| | plan == R1 | plan == command | plan == neither |
|---|---|---|---|
| **count** | **1,248** | **391** | 269 |
| **rate** | **0.6541** | **0.2049** | — |

`follow_gap = P(plan==R1) - P(plan==command)` = **+0.4492 [+0.3369, +0.5551], SEPARATED**
(paired episode-cluster bootstrap, 2,000 resamples, seed 0, 96 episode clusters).

**Robust across every pre-registered deadband** — `tau_y` 0.5 / 1.0 / 2.0 gives gap
**+0.2993 / +0.4492 / +0.5372**, separated at all three.

⇒ **The plan tracks the model's own vision-derived route readout ~3.2x more often than the
command it was given. The PI's video is not a cherry-pick; it is the typical case.**

### 2.3 ⛔⛔ R2 is DEGENERATE — and that voids its half of the table

*Artifact: `raw/GSTR_CENSUS.json`. Evidence class MEASURED (ours), tier T1.*

`g_str`'s lateral component is **negative on 4,823/4,823 windows** (**0** LEFT-pointing; max **-0.1022**, min -0.9711, range 0.8689):
**it never points left, on any window, ever** — while the ground truth turns left on **1,597**
windows (33.1 %). Its direction *class* is constant RIGHT.

⇒ The R2 row of the contingency table (`plan==R2` 0.2723 vs `plan==command` 0.4670, gap
**-0.1946 [-0.3071, -0.0746]**) is **NOT evidence that the plan ignores `g_str`** — a constant
readout cannot be "followed" or "not followed", and that rate is simply the plan's own RIGHT base
rate (1,435/4,823 = 0.2975). **Reported as DEGENERATE, not as a measurement.** This is the same
structural-zero family as `H-ECHO-4`.

⚠️ It also explains the banked `gstr: NAV_BLIND` verdict whose deltas are **exactly 0.0 with
CI [0,0]** — an identity, not an estimate.

### 2.4 The presence-vs-content mechanism, reproduced at the strategic head

*Artifact: `raw/GSTR_CENSUS.json`, n = 4,823 windows / 141 episodes.* **The statistic is the MEAN over windows of the RELATIVE L2 displacement of the `g_str` 3-vector** -- normalisation-dependent, so it is quoted with that definition or not at all:

| intervention | move | |
|---|---|---|
| nav **value** changed (flip) | **0.0098** | |
| nav shuffled | 0.0207 | |
| nav **removed** (zero) | **0.4018** | **41.0x the value change** |

And at the plan itself, terminal displacement: navflip **mean 1.1226 m but MEDIAN 0.0000 m** —
on most windows, inverting the command changes the emitted plan by **exactly nothing**; navzero
mean 1.8644 m, median 0.4901 m.

⇒ **The nav edge is gated on PRESENCE, not CONTENT**, and this reproduces at the strategic head,
not merely at the planner. *(Confirms the programme's earlier 97.7 % / 2.3 % reading with an
independent statistic; the brief's 18.6x becomes 41.0x on this normalisation.)*

---

## 3. P1 — the INTERVENTION: does perturbing the strategic goal move the plan?

Pre-registered in `PREREG.md` §4 and run on **ONE surface, one checkpoint** (Thor,
`refcv4b ckpt_40284_FINAL.pt`), through the ablation hooks **already implemented and test-pinned**
in `refcv3_arm.py` (`--ablate gstr_zero`, `--ablate-frames`) — no model file was edited.
⭐ That the intervention was APPLIED is a **positive assertion**, not an assumption:
`dump_GZERO/ABLATION.txt` reads `gstr_zero`, `dump_REPL/ABLATION.txt` reads `FULL`,
`dump_BLIND/ABLATION.txt` reads `frames_blind`.

**Metric:** plan **terminal displacement** (m) at the 6 s point. **n = 993 matched windows over
29 episodes** — the panel was scored on a matched subset because the full 141-episode roll is
~3.9 h/arm on this box; the reduced `n` is stated here rather than hidden. Paired episode-cluster
bootstrap, 2,000 resamples, seed 0. Join key is the **clip index**, never the filename.

| row | mean (m) | CI95 | median (m) | plan UNCHANGED |
|---|---|---|---|---|
| **NOISE FLOOR** — replicate, same flags, same seed | **0.0001** | [0.0001, 0.0001] | 0.0001 | 0.000 |
| `nav_FLIP` — command inverted | 1.2334 | [0.4812, 2.1418] | **0.0000** | **0.690** |
| `nav_SHUFFLE` — another clip's command | 1.6417 | [1.1908, 2.1377] | 0.0000 | 0.526 |
| `nav_ZERO` — command withheld | 1.9006 | [1.3428, 2.5020] | 0.4561 | 0.000 |
| ⭐ **`gstr_ZERO`** — strategic goal replaced by straight-ahead | **3.3130** | **[2.8950, 3.7743]** | **1.3046** | 0.000 |
| ⭐ **`gstr_SHUFFLE`** — another window's goal | **3.3196** | [2.6339, 4.0829] | 0.6437 | 0.000 |

⭐⭐ **THE STRATEGIC GOAL MOVES THE PLAN MORE THAN THE COMMAND DOES.** `gstr_ZERO`'s interval
**[2.8950, 3.7743] does not overlap** `nav_ZERO`'s **[1.3428, 2.5020]**: removing `g_str` moves the
plan **1.74x** more than removing the nav command entirely, and **2.69x** more than inverting it.

⛔ **The `plan UNCHANGED` column is the sharpest fact in this file: inverting the commanded route
changes the emitted plan by EXACTLY ZERO on 69.0 % of windows** (shuffling it, 52.6 %), while
perturbing `g_str` changes it on **100 %**.

⭐ **`gstr_SHUFFLE` completes the pre-registered panel and sharpens the reading.** Feeding the
model **another window's** `g_str` moves the plan **3.3196 m [2.6339, 4.0829]** — statistically
indistinguishable from replacing it with a straight-ahead constant (3.3130 m; the intervals
overlap almost entirely). ⇒ **`g_str`'s authority is not merely its PRESENCE — its per-window
CONTENT drives the plan**, because serving a *different but equally valid* goal is as disruptive
as removing the goal altogether. ⛔ **This is the exact opposite of what the nav command shows**,
where shuffling leaves 52.6 % of plans bit-identical and flipping leaves 69.0 %.

⚠️ **Read the medians, not only the means.** `gstr_ZERO` median **1.3046 m** vs `gstr_SHUFFLE`
**0.6437 m**: the *typical* window moves about 2x more under zeroing, and `gstr_SHUFFLE`'s equal
mean is carried by a heavier tail. Both are ~33,000x the replicate floor, so the conclusion is
unaffected — but the two interventions are not interchangeable, and the means alone would hide it.
*(Mechanically consistent with §2.3: every window's `g_str` points right, so a shuffle swaps one
right-pointing bearing for another — a smaller directional change than zeroing, which is why its
median is lower while its tail is longer.)*

⚠️ **Which variance this answers.** The replicate row answers the **run-to-run** question directly
and reads **0.0001 m** — this rig is effectively deterministic (refcv4b selects by argmax over a
fixed anchor bank; there is no sampling planner, hence no inference-seed floor to clear). Every
effect above is ~**33,000x** that floor. The bootstrap CIs answer the **episode-draw** question.
⇒ the separated CIs here are backed by a MEASURED replicate, not asserted against the
14.3 % one-seed false-positive rate.

| ⛔ **`frames_blind`** — DELIBERATE REGRESSION, image-blind | **14.9837** | [13.0869, 16.8071] | 15.0759 | 0.000 |

⭐ **THE DELIBERATE REGRESSION FAILS, DECISIVELY — so the panel certifies something.** An
image-blind arm moves the plan **14.9837 m [13.0869, 16.8071]**, ~150,000x the replicate floor. A
gate that has never been shown to FAIL an image-blind arm certifies nothing (H-ECHO-4); this one
fails it by a factor of 4.5 over the next-largest effect, so its *small* readings are trustworthy
rather than merely quiet.

⭐ **And it supplies the SCALE the whole hierarchy should be read against:**

| what the plan depends on | plan displacement when removed | share of vision |
|---|---|---|
| **vision** (`frames_blind`) | **14.9837 m** | 100 % |
| **strategic goal** (`gstr_ZERO`) | **3.3130 m** | **22.1 %** |
| **nav command** (`nav_ZERO`) | **1.9006 m** | **12.7 %** |
| run-to-run noise (replicate) | 0.0001 m | 0.0007 % |

⇒ **The planner is vision-dominated; the strategic goal is a distant second with 1.74x the
authority of the command; the commanded route is last.**

### 3.1 The four metric families

⛔ Per the binding rule, this panel is **not** ADE, and the families are not pooled.

| family | what this panel reports | status |
|---|---|---|
| **LATERAL** | plan terminal displacement and its direction class; the R1/R2 bearing readouts; the orientation control (**PASSES**) | ⭐ **primary here** |
| **STRATEGIC** | route-vs-command contingency (§2.2); `g_str` degeneracy (§2.3); presence-vs-content at the strategic head (§2.4); the `gstr_ZERO` intervention | ⭐ **primary here** |
| **TACTICAL** | the dump carries `lat_pred_*` / `lon_pred_*` under all five conditionings against `lat_label` / `lon_label`, `n = 993` | ⚠️ **NOT SCORED HERE** — a decision-quality question, not a plan-displacement one; folding it in would pool two families. Named as a **work item**, not dropped. |
| **LONGITUDINAL** | speed / headway; `--lead-block` was supplied so it is computable, `n = 993` | ⚠️ **NOT SCORED HERE**, same reason — outside this panel's pre-registered scope. |

⛔ `oracle_sel` / `anchor_acc` / `sel_agrees_oracle` are **INVALID on refcv4b** and are not quoted
even though the dump carries them. The `|dyaw| > 0.15` turn gate is used nowhere in this file.

---

## 4. VERDICT — which of the committed outcomes

### ⭐ **OUTCOME (a): THE PLANNER FOLLOWS ITS OWN STRATEGIC GOAL, NOT THE COMMANDED ROUTE — with the head IDENTIFIED CORRECTLY.**

**The PI's hypothesis is SUPPORTED.** Three independent measurements agree:

1. **Causally** — perturbing `g_str` moves the plan **3.3130 m [2.8950, 3.7743]** against a
   **0.0001 m** replicate floor: **1.74x** the effect of removing the nav command, with
   non-overlapping intervals.
2. **Observationally** — on the 1,908 windows where the route readout and the command disagree, the
   plan matches the **readout 65.4 %** and the **command 20.5 %** (gap **+0.4492 [+0.3369,
   +0.5551]**, separated, robust across every pre-registered deadband).
3. **Structurally** — inverting the command changes the plan by **exactly nothing on 69 % of
   windows**.

⛔ **But the head in the PI's video is NOT the head that steers, and the difference decides the
fix.** The video's *"route head says RIGHT (0.68 -> 0.73)"* is **R1**, `route_logits` — a **3-way
categorical aux** that the arm's own config disconnects (`"graft_route": false`), and which
therefore **cannot** be causing anything. It is a faithful *telltale* of the vision the planner also
reads, not its cause. The head that actually steers is **R2**, the cascade's geometric `g_str`.

⛔⛔ **And R2 is broken in a way that is worse than a wiring preference: `g_str` NEVER POINTS LEFT
on any of 4,823 windows** (max lateral component **-0.1022**) while ground truth turns left on
**1,597** of them. ⇒ **The signal that dominates the planner is structurally incapable of commanding
a left turn.** That is exactly the PI's *"the strategic layer injects its own error AND overrides a
correct input"* — now measured, with the error localised.

**One line:** **the planner is following its own predicted route — specifically the cascade's
`g_str` — while the commanded route is inert on two-thirds of windows; the head the video shows is
a disconnected aux, and the head that actually steers can only ever turn right.**

### The next levers, ranked by measured effect (⛔ a refutation is a waypoint, not a deliverable)

1. ⭐ **Feed the nav args.** `distance_m` / `time_s` exist on every turn record and are **already
   consumed** by `NavConditioner` (`arg_proj = nn.Linear(2, d_embed)`); refcv3/refcv4b simply never
   read them. Largest measured lever available — a stripped bearing is separated WORSE by
   **+2.3632 m**, a range-carrying goal recovers **57.1 %** of the longitudinal ceiling — and it is
   a **loader change, not an architecture change**.
2. ⭐ **Fix or retire `g_str`.** A goal head whose bearing never points left is not a goal head. Its
   supervision (`strategic_goal_loss`) and its zero-init FiLM should be re-read together; until it
   can express "left", the hierarchy's strategic level is a right-turn bias generator holding
   **1.74x the planner authority of the nav command**.
3. **Decide R1's fate deliberately.** `route_logits` is trained, is an excellent predictor of the
   plan, and reaches nothing. Either graft it (with the C6-confound guard the code already
   documents) or stop training it — it should not sit in between as a head that *looks* like it
   steers, because that is what made this hypothesis look confirmed for the wrong reason.

### What is NOT claimed

* ⛔ This does **not** show the command is worthless: `nav_ZERO` moves the plan **1.9006 m**, and the
  banked panel's `plan` verdict is **FOLLOWS_NAV** (separated under both shuffle and zero). The
  command is **real but weak, and inert on most windows**.
* ⛔ The R2 half of the contingency table is **degenerate** and is not quoted as a measurement.
* ⚠️ Tactical and longitudinal families were **not scored** in this panel (§3.1) — work items, not
  silent omissions.
* ⚠️ `n = 993 windows / 29 episodes` for the intervention panel (not 4,823 / 141). The effect is
  ~33,000x the replicate floor and was stable as `n` grew 684 -> 720 -> 822 -> 993 windows, but a
  full-corpus re-roll would tighten the intervals.

