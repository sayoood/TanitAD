# 4-Brain Dominance Program — PC1/PC2 pre-conditions, implemented

**Date:** 2026-07-26 (Europe/Berlin) · **Agent:** 4brain-preconditions
**Spec:** `Project Steering/Reviews/2026-07-25-independent-chief-scientist-review/01_EXECUTION_PLAN.md`
Part A (PC1–PC4, HP-1…HP-6) · `…/incoming/2026-07-25-hpp0-confound-audit/HPP0_CONFOUND_AUDIT.md`
**Compute:** dev box + one short GPU forward pass on the eval pod. **No training launched.
Nothing staged, nothing committed, nothing pushed.**

**Evidence classes** (CLAUDE.md operating standard 1): `MEASURED` (ours + file:line / artifact) ·
`PUBLISHED` · `INHERITED` (another doc, NOT re-verified) · `ESTIMATED` · `HYPOTHESIS`.

**`pytest -q`** — `stack` **1020 passed / 3 skipped** (baseline 1004/3, **+16**) ·
`taniteval` **360 passed** (baseline 334, **+26**). `MEASURED`.

---

## 0. Two findings that change the brief before any code does

### ① The brief's PC1 fix #1 does not exist. `--labels-v2` selects the **v2** labeler, not v2.1.

The brief (and `HPP0_CONFOUND_AUDIT.md` §6 HPP-1 P1, and the audit's PC1 table row 1) says:

> `--labels-v2` ⇒ `route_target_v21` / `route_from_future_v3`, coverage **27 % → 80.4 %**.

`MEASURED` — `stack/scripts/refb_train.py` (pre-change, the `labels_v2` branch) wires `labels_v2=True` to
`nav_command_v2` / `route_target_v2`, i.e. `route_from_future` — **the v2 labeler, which keeps v1's
fixed 15 s/25 s lookahead.** The repo already knew: `stack/scripts/v15_prep.py:304` carries the
comment *"v2.1 EXPLICITLY (not config.v2_labels, which still selects v2)"*.

Measured on **17 100 real PhysicalAI trainer windows** (100-episode val cache
`physicalai-val-bb543bdf7836`, window 8 / max_horizon 20 / stride 1 — the trainer's own indexing;
`verify_pc1_labels.py`, `pc1_label_verification.json`):

| labeler as the trainer wires it | `nav_valid` coverage | fed `follow` while the road turns | share of ALL true turns lost |
|---|---:|---:|---:|
| **v1** (default) | 0.2456 | **0.4722** | **73.8 %** |
| **v2** = today's `--labels-v2` | **0.2307** | 0.4286 | 66.9 % |
| **v2.1** (adaptive-arc) | **0.7546** | **0.0000** | **0 %** |

`--labels-v2` does not buy coverage — it is **marginally worse than v1**. The 80.4 % figure the audit
quotes is from the **v4 label pass** (`labels_train_v4_provenance.json`), a different code path that
the flagship trainer never touches. PC1 fix #1 was therefore **not** config-only; it is implemented
here (T1).

### ② No labeler swap breaks the circularity. **LEVER A is the whole fix.**

`MEASURED`, same run: `route_target == _NAV_TO_ROUTE[nav_cmd]` on **100.0000 %** of CE-eligible
windows under **v1, v2 AND v2.1 alike** (`echo_rate_valid_windows = 1.0`, 4 200 / 3 945 / 12 904
valid windows respectively). Independently reproduced by the T3 firewall on real labels
(`blind_firewall_route_target.json`): blind accuracy **1.0000** for all three.

The reason is structural, not a bug in any one labeler: **the fed command is minted from the same
`route_from_future*` call as the target**, and `_ROUTE_TO_NAV` is a bijection. `nav_command_v21`
returns `_ROUTE_TO_NAV[r["route"]]` while `route_target_v21` returns `r["route"]`
(`refb_labels.py:705-733`). `route_target_v2`'s own docstring — *"NOT a function of any fed nav
command (breaks the v1 circularity)"* — is true of the **function signature** and false of the
**wiring**.

**Consequence for PC1.** The only mechanism in the repo that produces a non-circular route gradient
is **LEVER A** (`cfg.v2_route_from_vision`, `flagship_losses.py:339-353`): a second strategic pass
with nav forced to `follow(0)`, class-weighted CE against the true route. On that pass the input
carries no route information, so the loss can only fall by inferring the route from vision — which is
exactly why its value stays at **0.61–0.76** while the main route CE reaches 0.0
(`INHERITED` from HPP-0 §1.3; consistent with our CPU smoke, `route_vis` 0.54–1.15 on step 0–2).

v2.1 labels are still necessary — they multiply LEVER A's training set **3.07×** (4 200 → 12 904
valid windows) and stop it being taught `straight` on windows it cannot judge — but they are **not
sufficient**, and the two must ship together. The trainer now refuses the half-fix.

---

## 1. T1 — PC1: the two label/loss levers, verified and wired

### 1.1 What changed, file:line

| File | Line(s) | Change |
|---|---|---|
| `stack/tanitad/config.py` | `238-264` | new `v21_route_labels: bool = False`, with the measured coverage/echo table and the explicit statement of what the flag does **not** fix |
| `stack/scripts/refb_train.py` | `139-165`, `168`, `186`, `196-206` | `FailLoudWindowDataset(..., labels_v21=False)`; v2.1 branch calls `nav_command_v21` / `route_target_v21` and **asserts the two agree on validity** |
| `stack/scripts/train_flagship4b.py` | `115-118`, `148-152` | `FlagshipWindowDataset` + `_wrap` thread `cfg.v21_route_labels` |
| `stack/scripts/train_flagship4b.py` | `304-329` | `--labels-v21` / `--no-labels-v21` resolution, the **LEVER-A pairing assert**, and the ON banner |
| `stack/scripts/train_flagship4b.py` | `600-634` | `--labels-v21`, `--no-labels-v21`, `--v2-route-from-vision`, `--no-v2-route-from-vision` argument definitions |
| `stack/scripts/train_flagship4b.py` | `381-383` | the `[data]` line now prints `v21_route_labels` and `route_from_vision` |
| `stack/tanitad/train/flagship_losses.py` | `318-333` | fail-loud guard: a `ROUTE_UNKNOWN` (=3) that reaches the 3-way CE **asserts** instead of raising an opaque bincount/CE shape error |
| `stack/tests/test_labels_v21_wiring.py` | new, 16 tests | pins coverage↑, `UNKNOWN`≠`straight`, never-`follow`-through-a-turn, **the echo surviving every labeler swap**, the loss+LEVER-A path, and every CLI combination |

**Design decisions, stated because they are load-bearing:**

* `--labels-v21` is **NOT implied by `--v2`.** It changes what `nav_valid` means on ~75 % of windows,
  so every shipped `--v2` arm would silently stop being comparable. Pinned by
  `test_v21_defaults_off_even_under_v2`.
* `--labels-v21` **without** LEVER A is **refused** with the message *"half-fix"* — the target stays a
  lookup of the input, so `route_skill` would stay 0 by construction and the run would burn a GPU-week
  proving nothing. The labels-only control arm is still runnable, but only by saying so explicitly
  (`--route-vis-weight 0`), which makes it a named arm rather than an accident.
* v3 (`route_from_future_v3`) is deliberately **not** wired: `route_target_v3`'s own docstring says
  *"The 3-class CE target is unchanged: use `route_target_v21` for it (v3 does not touch it by
  construction)"* — v3's 9-token vocabulary does not fit `n_route = 3`.

### 1.2 Verification — `MEASURED`, not asserted

1. **Label properties on real poses** — the table in §0①, produced by
   `verify_pc1_labels.py` → `pc1_label_verification.json`, 17 100 windows, 100 episodes.
   *(Corpus note: this is a local 100-episode PhysicalAI val cache, key `bb543bdf7836`, **not** the
   canonical `0c5f7dac3b11`. It is read-only and re-selects nothing — parity is untouched — and the
   quantity measured is a property of the **labeler**, not of a split.)*
2. **Wiring** — 16 CPU tests, including a direct per-window recompute against
   `refb_labels.nav_command_v21` / `route_target_v21`.
3. **LEVER A works** — pre-existing `stack/tests/test_vision_levers.py::
   test_route_vis_present_finite_and_grad_reaches_route_head` plus the new
   `test_flagship_loss_finite_on_v21_batch_with_lever_a`, which asserts `log["route_vis"] > 0` **and**
   that the gradient reaches `strategic_policy.route_head`.
4. **End-to-end 3-step CPU run of the exact launch configuration** (`MEASURED`, scratchpad
   `v21smoke`): banner fires, `nav_valid_frac` **0.50–0.75** (vs 0.0625–0.3125 in every shipped arm's
   trainer log), `route_vis` finite and non-zero every step, checkpoint written, `FLAGSHIP4B_DONE`.

### 1.3 THE PC1 LAUNCH COMMAND (verified; **not** launched — PI decision)

Runs on the sacred parity corpus `physicalai-train-e438721ae894`. Shape copied from the live pod1
command, with `--v2-cache` swapped for the parity `--cache-dirs` and the two PC1 flags added.

```bash
cd /workspace/TanitAD/stack && PYTHONPATH=/workspace/TanitAD/stack \
ssh_detached_or_nohup python3 -u scripts/train_flagship4b.py \
  --data cached --cache-dirs /workspace/data/physicalai_phase0/_epcache \
  --config flagship4b --v2 --rollout-k 12 \
  --labels-v21 --v2-route-from-vision --route-vis-weight 0.3 \
  --sigreg-free-dims 64 --steps 30000 --batch-size 16 --accum 4 \
  --grad-checkpoint --lr 3e-4 --warmup 2000 --workers 8 \
  --guard-limit-gb 45 --ckpt-every 1000 --log-every 50 \
  --out /workspace/experiments/flagship-pc1-30k
```

**Preflight (CLAUDE.md traps):** `PYTHONPATH` is required or the trainer dies with
`ModuleNotFound: tanitad`; launch with `ssh -f`, never `cmd &`; judge pod disk with a real `dd` write,
never `df`; kill any predecessor **by explicit PID**, never `pkill -f train_flagship4b` (it matches
your own ssh command). `--v2` already implies `v2_route_from_vision`; `--v2-route-from-vision` is
passed anyway so the arm's `config.json` records the intent explicitly.

**How to read it at 1 k / 5 k:**

* `nav_valid_frac` should sit around **0.6–0.8**, not 0.06–0.31. If it does not, the labels did not
  switch.
* `route` (main CE) **may still go to 0** — that is the echo and it is expected; it is no longer the
  metric.
* `route_vis` (LEVER A) is the number that matters. It should fall **but not to 0**; ~0.6–0.76 is the
  measured non-degenerate band on the two abandoned arms.
* The PC1 exit criterion is not in the trainer log at all: it is
  `taniteval.hierarchy`'s `route_skill_vs_majority`, CI-separated, on the migrated estimator.

**Control arm** (isolates the label change from LEVER A), if the PI wants the contrast:
same command with `--labels-v21 --route-vis-weight 0` and `--no-v2-route-from-vision`.

---

## 2. T2 — PC2: the hierarchy assertion

**New module: `taniteval/taniteval/hierarchy_guard.py`** (`HierarchyTrace`,
`assert_hierarchy_traversed`, `assert_actions_are_chosen`, `guarded`, `HierarchyBypass`).

### 2.1 The mechanism

Forward hooks on three modules, counted over **the scored pass only**:

| seam | hooked module | proves |
|---|---|---|
| `strategic` | `model.strategic_policy` | brain ① ran |
| `tactical` | `model.tactical_policy` | brain ② ran |
| `operative_intent` | `model.predictor.intent_proj` | brain ③ **received** the tactical intent |

`intent_proj` is the correct third hook, not the predictor: `OperativePredictor.forward` calls it
**only** when `intent is not None` (`stack/tanitad/models/predictor.py:99-106`). The predictor runs
on every path, so hooking it would prove nothing.

`assert_hierarchy_traversed` raises `HierarchyBypass` naming each missing seam **and whether it was
`absent` (no such module on this arch) or `bypassed` (module exists, never called)** — different
defects, different fixes, and **neither is a pass**.

`assert_actions_are_chosen` is the separate second half — HPP-0 §2.2's *"decision bypass, and this
one is not in the review"*. An arm can traverse all three seams and still be handed the expert's
future actions. `actions_source="expert_future"` fails; `"model_chosen"` / `"planner"` pass.

### 2.2 Where it is wired

| Callsite | Mode | Effect |
|---|---|---|
| `taniteval/rollout.py:107-121, 126, 163-175` | **non-strict** | the headline surface now stamps its own `win["pc2"]`: `pc2_pass=false`, `missing_seams=[strategic, tactical, operative_intent]`, `actions_source="expert_future"`, `honest_metric_name="wm_fidelity_ade_2s"`. Non-strict on purpose — WM fidelity is a legitimate diagnostic; what it may not do is be quoted as driving or hierarchy |
| `taniteval/hierarchy.py:434-441, 603-612` | **STRICT** | the seam panel cannot assemble a JSON if its scored pass skipped a brain |
| `taniteval/strategic_probes.py:141-181, 228-235, 287-296` | **STRICT** | HP-3 asserts the seams it reports on (`operative_intent` required only in `--grounded`) |
| `taniteval/runner.py:134-155` | record + warn | `res["pc2"]` in every result JSON; `refb_eval`/`refc_eval` window dicts have no trace and are recorded as **`unavailable`, explicitly "NOT a pass"** |

### 2.3 Verified on a real checkpoint

`MEASURED` — `hp3_prefix_flagship-30k.json` → `pc2`:
`{"pc2_pass": true, "counts": {"strategic": 240, "tactical": 240, "operative_intent": 0},
"missing_seams": [], "absent_modules": []}` — the probe's two required seams fired 240 times each on
the real `flagship-30k` forward pass, and the third is correctly not required in non-grounded mode.

12 tests in `taniteval/tests/test_hierarchy_guard.py`, including
`test_rollout_collect_declares_itself_a_wm_fidelity_surface`, which pins that the leaderboard surface
reports `pc2_pass=false` **by construction**.

---

## 3. T4 — HP-3 pre-fix baseline. **The pre-registered expectation is FALSIFIED.**

Run on the eval pod (short forward pass; wheelbase co-tenant used 2.4 GB of 46 GB and the GPU was at
0–87 %; RAM headroom 24 GB of the 50 GB cgroup; disk verified by `dd` at 505 MB/s, not `df`).
pod1 and pod3 untouched; pod2 was at step 27 500/30 000 ≈ **4.9 h** from finishing, not 2 h.

### 3.1 `flagship-30k` — HP-3 **PASSES**, pre-fix

`MEASURED` — `hp3_prefix_flagship-30k.json`, 881 windows / 40 episodes,
**episode_cluster_bootstrap, B=2000**, surface = the tactical head's 2 s waypoints (IN-REGIME).

| quantity | left vs right | left vs follow | right vs follow |
|---|---|---|---|
| **cross-track @2 s (the HP-3 channel)** | **0.5617 [0.4741, 0.6583] m** | 0.5309 [0.4401, 0.6369] | 0.5269 [0.4284, 0.6424] |
| cross-track p90 | 1.2218 [1.0457, 1.3867] | 1.1783 [0.9603, 1.4115] | 1.2227 [1.0200, 1.4723] |
| waypoint L2 (mean) | 2.7590 [2.2756, 3.3216] m | 3.5251 [2.7658, 4.3559] | 3.3869 [2.6411, 4.2815] |
| waypoint L2 (2 s endpoint) | 4.4938 [3.7078, 5.3903] m | 5.8607 [4.6402, 7.2039] | 5.6322 [4.4176, 7.0929] |
| ctx cosine | 0.3540 [0.3374, 0.3704] | 0.5489 | 0.5430 |
| intent cosine | 0.8314 [0.8015, 0.8585] | 0.8292 | 0.8487 |
| maneuver changed | 0.4631 [0.3903, 0.5346] | 0.4030 | 0.4495 |

* **direction score 0.6215 [0.5659, 0.6772]** vs chance 0.5 → `separated_above_chance: true`
* `HP3_divergence_separated: true` · `HP3_direction_correct: true` · **`HP3_route_conditional: true`**
* route-logit echo **1.0000** (follow/left/right each 1.0), `n_nav_valid` 240 — the echo reproduced on
  the real checkpoint, exactly as HPP-0 §1.5 reported.

**⚠️ Read this with the asymmetry, which is the honest part.** The signed lateral response is
carried almost entirely by ONE branch:

| branch | signed lateral Δ vs `follow` (+y = left) | separated? |
|---|---|---|
| **left** | **+0.3392 [0.2048, 0.4808] m** | ✅ correct sign, separated |
| **right** | **−0.0324 [−0.1914, 0.1467] m** | ❌ **not separated from 0** |

and `both_correct_rate` is **0.3337 [0.2537, 0.4211]** — only a third of windows get *both* commands
pointing the right way. So the correct verdict is **"HP-3 passes on the left branch; the right
command barely moves the trajectory"**, not "route conditioning works".

**What this overturns.** The pre-registration in `strategic_probes.py` (and the brief) commits to
*"expect ~0 by construction"*. That is **wrong**, and the reason it is wrong matters: `route_skill = 0`
measures whether the head can **infer** the route without the command; HP-3 measures whether the
command **steers** the trajectory. They are different quantities, and the audit's one-line summary
("the hierarchy has never been given a route to follow") conflated them. `MEASURED`: the
strategic→tactical seam **does** carry the command into the tactical trajectory. What is broken is
(a) route inference from vision (echo 1.0, `route_skill` 0.0), and (b) PC2 — the command never
reaches the **scored** trajectory (`route_can_reach_scored_trajectory: false`).

**Caveat on magnitude, stated up front:** the tactical head's own absolute error is ~3.38 m
(`INHERITED`, HPP-0 §5 E1), so a 2.76 m L2 divergence is of the same order as the head's accuracy.
The cross-track channel (0.56 m) is the module's designated headline for precisely this reason, and
even it should be read as *"the command perturbs this head substantially"*, not as a competence claim.

### 3.2 `refc-base-30k` — **NOT MEASURED**, and it is a naming miss, not an absence

`MEASURED` — `hp3_prefix_refc-base-30k.json`. HP-3 returns a SKIP: `RefCModel` exposes no
`strategic_policy` / `tactical_policy`.

**But REF-C has a strategic level.** Probing the loaded model directly (CLAUDE.md rule 2 — absence at
one location is not absence) found: `strategic`, `strategic.gru`, `strategic.proj`,
`decoder.ctx_to_cond`, `decoder.maneuver_to_anchor`, `maneuver_head`, `route_head`. Reporting
"REF-C has no strategic level" would have manufactured a false structural claim about the exact
architecture this program benchmarks itself against.

The probe's skip path was therefore rewritten (`strategic_probes.py:141-181`, `_skip_report`) to
enumerate what it found and to expose `strategic_level_absent` (**false** here) as the only field
that may be read as absence. **HP-3 on REF-C needs a per-arch adapter — a small, zero-training work
item — and must not be recorded as a 0.**

---

## 4. T3 — the `blind_conditioning_baseline` firewall

**New module: `taniteval/taniteval/blind_baseline.py`** · 14 tests in
`taniteval/tests/test_blind_baseline.py`.

**The rule:** train a head on the symbolic context alone — no image — and if it matches the real
model, the target is circular and inadmissible.

| verdict | condition | consequence |
|---|---|---|
| `CIRCULAR` | blind ≥ 1 − 0.02, **or** blind ≥ real − 0.02 | **inadmissible** — registration refused |
| `LEAKY` | blind − majority ≥ 0.03 | admissible **only** as a skill-over-blind delta |
| `CLEAN` | otherwise | this leak ruled out, and nothing else |

Methodology, each choice load-bearing:

* **Episode-clustered CV split** and `ci.episode_cluster_bootstrap` intervals. A within-episode split
  makes almost any target look context-learnable, because context is near-constant inside an episode.
* **Categorical context is one-hot encoded, never fed as an integer.** A linear probe on an integer
  code can only express a monotone ramp and would miss a lookup-table leak — the encoding *is* the
  check.
* Both a 1-hidden-layer MLP and a linear probe are fitted and both are reported, so an
  under-expressive probe cannot manufacture a pass.

**It catches the real defect.** `MEASURED` on real PhysicalAI labels
(`python -m taniteval.blind_baseline --cache <epcache> --episodes 40`, →
`blind_firewall_route_target.json`), CPU, ~30 s:

```
v1_route_target : blind 1.0000 (linear 1.0000) vs majority 0.4240 -> CIRCULAR  [n=441/40 eps]
v2_route_target : blind 1.0000 (linear 1.0000) vs majority 0.4832 -> CIRCULAR  [n=416/38 eps]
v21_route_target: blind 1.0000 (linear 1.0000) vs majority 0.4565 -> CIRCULAR  [n=1264/37 eps]
```

All three labelers, blind accuracy exactly **1.0000** from `nav_cmd` alone — an independent
confirmation of §0②, and the check that would have caught `route_target` before it reached a shipped
checkpoint and was reported `load_bearing: true` for months.

**The gate.** `register_decision_problem(name, target=…, conditioning=…, firewall=…)` **refuses**
without a passing firewall record produced by `blind_conditioning_baseline` on that problem's own
data. There is deliberately **no `force` argument**: the check cannot be waived by the person whose
result depends on waiving it. `assert_registered(name)` fails loud for any scored problem that was
never firewalled. A `LEAKY` problem registers but is flagged `must_report_skill_over_blind: true`.

*(A note on the tests: the first `CLEAN` fixture came back `LEAKY` because `nav` and `y` were drawn
from generators seeded alike — deterministic functions of the same uniform stream. The firewall was
right and the fixture was wrong. That is recorded in the test file rather than smoothed over.)*

---

## 5. T5 — the eval-time route leaks

### 5.1 `nav_cmd=None` in the REF-C trio — **closed**

| File | Line | Change |
|---|---|---|
| `taniteval/taniteval/refc_eval.py` | `25-51` (docstring), `67-102` (`NAV_MODES`, `ROUTE_TO_NAV`, `resolve_nav`), `106-138`, `150-157`, `166-190` | `collect(..., nav_mode="produced")`. **Default is now the produced goal**: a first pass reads the model's OWN `route_head` logits (image-only — `nav` enters solely through `measurement`, so the logits are identical whatever nav that pass was given), a second pass feeds their `argmax` as `nav_cmd`. Every window dict carries a **`nav_provenance`** block (mode, fed-command histogram, `route_input_exercised`, `is_oracle`) and `method` names the mode |
| `taniteval/taniteval/refc_rerank.py` | `89-96` (`NAV_MODE`), `269-279`, `311-317` | the literal is gone; the mode is a named constant, kept at `follow_constant` **because this sweep's lam=0 row is asserted to reproduce the published `refc-xl-30k` number** — feeding a different command would move the anchor the whole sweep is measured against. Stamped into the dump |
| `taniteval/taniteval/plan_fan.py` | `516-534`, `558-562`, `596-602` | `episode_planfan(..., nav_mode="follow_constant")` routed through the same `resolve_nav`; every frame record carries `nav_mode` + `nav_fed`. Default kept at `follow_constant` deliberately: this module renders the fan a human inspects, and every fan clip in the record was drawn under the constant command — silently changing what the pictures mean is the same class of error as silently changing a number |

`nav_mode="oracle"` exists in `resolve_nav` and is labelled as an upper bound only.

**⚠️ Registry action.** REF-C's published numbers — **base 0.4728, XL 0.4714** (`PUBLISHED`,
`LEADERBOARD.md` §1) — were collected under `follow_constant`, i.e. with the route input never
exercised. They are **not** comparable to a `produced` run and not comparable to a route-conditional
arm. Re-running them is one short eval; the values must not be silently reinterpreted.

### 5.2 The eval-time route ORACLE — **disclosed in place, numbers untouched**

**New module: `stack/scripts/goal_provenance.py`** — one disclosure, three callers. It prints a
banner and returns a block for the result JSON. It changes **no number, no input, and no code path**,
per the brief.

| File | Line | Change |
|---|---|---|
| `stack/scripts/eval_flagship_v15.py` | `45-46`, `218-224` | `res["goal_provenance"] = goal_provenance.disclose("eval_flagship_v15")` |
| `stack/scripts/eval_flagship_v16.py` | `39-40`, `239-244` | same, with the v1.6 headline called out by name |
| `stack/scripts/eval_flagship_v4.py` | `82`, `589-597` | same for MODE B (MODE A returns earlier and feeds no goal inputs) |

### 5.3 **Published numbers produced with a goal oracle** — for the registry owner

`MEASURED` (the code path) + `PUBLISHED` (the values). Recorded in machine-readable form in
`goal_provenance.AFFECTED_PUBLISHED_NUMBERS` so the list cannot drift away from the mechanism.

| Arm | Published value | Evaluator | Oracle fields (all minted from the ego's own FUTURE poses) |
|---|---|---|---|
| **flagship v1.6** | **ADE@2 s = 0.4375 m** — the v1.6 headline | `eval_flagship_v16.py:135-143` | `route`, `route_graded`, `vt_band`, `vt_speed` |
| **flagship v1.5** | every v1.5 anchored-fan number (incl. `ab` 0.5437 / 0.5366) | `eval_flagship_v15.py:92-103` | same four |
| **flagship v4 MODE B** | every v4 MODE-B number, **including the 30 k gate's primary** | `eval_flagship_v4.py:322` (`_goal_inputs`) + `train_flagship_v4.py:167-173` | `route`, `route_graded`, `vt_band`, `vt_speed`, `strat_scalars` |

This violates `V4_FLAGSHIP_DESIGN.md:558-560` (*"No leaderboard number may come from a GT-derived
plan or a GT-derived goal"*) as a leaderboard value. Each is an **upper bound**, not a deployable
number, and none is comparable either to REF-C's constant-command numbers or to a produced-goal run.
Removing the oracle needs a produced-goal strategic head — which is what PC1 is for — plus a re-run;
**the correction to the published values belongs to the registry owner, not to this script.**

---

## 6. Where PC1–PC4 stand after this bundle

| PC | Before | After | Remaining |
|---|---|---|---|
| **PC1** | 🟥 | 🟧 **fix implemented and verified, NOT trained** | one 30 k run (§1.3, PI decision) + `hierarchy.route_skill_vs_majority` CI-separated. Note the audit's fix #1 was mis-specified and is corrected here |
| **PC2** | 🟥 | 🟩 **asserted in code** on 4 callsites; verified on a real checkpoint | rename the true-action rollout to `wm_fidelity_ade_2s` in `LEADERBOARD.md` / `MODEL_REGISTRY.md` (doc-side, HPP-3 item 2) |
| **PC3** | 🟥 | unchanged (out of scope) | `corridor_departure_rate @ K=max`, junction stratum, long-horizon rollout |
| **PC4** | 🟥 | unchanged (out of scope) | `route_eval_v1` on the v2 balanced corpus; HP-3/HP-4 with real branches still needs `obstacle.offline` or an external map-bearing benchmark |

---

## 7. Escalations — do not let these live only in this file

1. **The HPP-0 audit's PC1 item #1 is wrong and should be corrected at source.** `--labels-v2`
   selects v2, not v2.1; it does not raise coverage; and **no** labeler swap breaks the echo. The
   audit's own §6 table calls fix #1 "S (config)" — it was not. `RETRACTION_LOG.md` class: *quoting a
   flag's documented intent instead of tracing its wiring* — the same root cause as the
   `route_target_v2` docstring, which is literally true and operationally false.
2. **HP-3's pre-registered expectation is falsified** (§3.1). `strategic_probes.py`'s docstring and
   `INVOCATION` still say "expect ~0 by construction"; they now contradict a MEASURED result on the
   deployed checkpoint. Both outcomes were committed in advance, so this is a clean pre-registered
   surprise — but the framing "the hierarchy has never been given a route to follow" must be narrowed
   to "the route is an echo, and it never reaches the **scored** trajectory".
3. **Three shipped headline numbers are goal-oracle numbers** (§5.3), including **v1.6's 0.4375** and
   **every v4 MODE-B number, i.e. the 30 k gate's primary**. Registry owner action, today.
4. **REF-C's published 0.4728 / 0.4714 were collected with the route input never exercised** (§5.1)
   and are not comparable to a produced-goal run. One short re-run fixes it.
5. **HP-3 on REF-C needs a per-arch adapter** (§3.2) — zero training, small. Until then HP-3 is
   UNMEASURED on REF-C and must not be recorded as 0.
6. **Wire `blind_conditioning_baseline` into the gate.** The firewall exists and refuses registration,
   but nothing yet *calls* `assert_registered` from `run_gate.py`. That is the step that makes it
   unskippable rather than merely available.

---

## 8. Deliverable manifest

| Artifact | Where it lives |
|---|---|
| **This report** | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-4brain-preconditions/PRECONDITIONS_IMPLEMENTED.md` (repo working tree, **not staged**) |
| PC1 label verification script | `…/2026-07-26-4brain-preconditions/verify_pc1_labels.py` |
| PC1 label verification result | `…/2026-07-26-4brain-preconditions/pc1_label_verification.json` |
| HP-3 pre-fix, flagship-30k | `…/2026-07-26-4brain-preconditions/hp3_prefix_flagship-30k.json` (also `tanitad-eval:/root/taniteval_hp3/results/`) |
| HP-3 pre-fix, refc-base-30k (diagnostic SKIP) | `…/2026-07-26-4brain-preconditions/hp3_prefix_refc-base-30k.json` (also on the eval pod) |
| Blind-firewall result on real labels | `…/2026-07-26-4brain-preconditions/blind_firewall_route_target.json` |
| **T1 code** | `stack/tanitad/config.py`, `stack/scripts/refb_train.py`, `stack/scripts/train_flagship4b.py`, `stack/tanitad/train/flagship_losses.py`, `stack/tests/test_labels_v21_wiring.py` (new) |
| **T2 code** | `taniteval/taniteval/hierarchy_guard.py` (new), `taniteval/taniteval/{rollout,hierarchy,strategic_probes,runner}.py`, `taniteval/tests/test_hierarchy_guard.py` (new) |
| **T3 code** | `taniteval/taniteval/blind_baseline.py` (new), `taniteval/tests/test_blind_baseline.py` (new) |
| **T5 code** | `taniteval/taniteval/{refc_eval,refc_rerank,plan_fan}.py`, `stack/scripts/goal_provenance.py` (new), `stack/scripts/eval_flagship_{v15,v16,v4}.py` |
| Isolated eval-pod package used for T4 | `tanitad-eval:/root/taniteval_hp3/` — a **copy**; the pod's standing `/root/taniteval` was not modified |
| Nothing staged, nothing committed, nothing pushed | ✅ per brief |
| No training launched; pod1 / pod2 / pod3 untouched | ✅ per brief |
