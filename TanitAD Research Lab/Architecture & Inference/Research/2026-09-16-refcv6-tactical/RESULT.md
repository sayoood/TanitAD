# refcv6 §4/§5 — the tactical behaviour decoder, nav into selection, max speed as an input

**2026-09-16 (Europe/Berlin).** Architecture & Inference. Code only — **no GPU hour was spent and no arm was trained.** Everything below is either a MEASURED property of code that ran on this dev box, or a MEASURED property of the label corpus. Nothing here is a claim about a model's driving.

**Branch base:** `agent/arch-inf-20260803` @ **`e95af00`** (this work first landed as `cbadba5` @ `9a782fa`; §10 records the rebase). Worktree `C:/Users/Admin/tanitad-wt-tactical-v6`. Python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`, `CUDA_VISIBLE_DEVICES=""` throughout.

---

## ⛔ ESCALATE — INTEGRATION

Five things the Master Mind must decide or apply; the first two are blocking.

1. **`refc.py` is a PATCH, not a staged edit** (another agent owns it). ⭐ **REBASED 2026-09-17 onto `e95af00`** — see §10; the version staged beside this file is the rebased one and applies cleanly to `e95af00`'s `refc.py` (`git apply --check` clean; applied, exercised end-to-end, then reverted). **Nothing in §4/§5 reaches a forward without it** — `RefCV3Model.forward` passes `scene_hook=` unconditionally, so an unpatched core raises `TypeError` rather than silently running refcv5 under a refcv6 config. Apply-here notes in §6.
2. **The tactical decoder forces the agent seam on, and that is a SECOND variable.** `refc.py:2955` refuses `agents.enable` without `decoder.cross_agent` (refcv5 WP-6: a detector trained into a dead end — a correct guard, another owner's). So giving the TACTICAL layer agent slots also turns on agent cross-attention in the OPERATIVE decoder. ⇒ **a refcv6 tactical arm is one-variable only against a baseline that already has the agent seam on.** Not worked around, not silently relaxed. Decide the baseline before the arm is launched.
3. **The BEV port is built and unfed, and `d_bev` MUST be set from the trunk.** The decoder has a real `bev_tokens` port, but the lift belongs to the perception agent (`tanitad-wt-percep-v6`, locked). Until it lands, refcv6 tactical runs **agents-only** — the decoder refuses a `bev_tokens` argument on a `d_bev = 0` build and refuses a build with neither source, so there is no silent-drop path either way.
   ⭐ **PI 2026-09-16 (absorbed):** the input becomes **256×1024** and the trunk becomes **resnet101**, with **resnet34** as the comparison run. ⇒ the stride-16 map is **16 × 64** (was 16 × 40) and is **1024** channels wide (resnet34: **256**). The **token COUNT is read off the tensor at runtime and nothing is typed** — `bev_feats_to_tokens` derives `P = X·Y` from the grid, and the queries and the FiLM condition are shape-independent of it (tested at 16×64, 16×40 and 30×16 on one build). The **WIDTH is a parameter shape** (`bev_in = Linear(d_bev, 256)`) and therefore must be DECLARED: `tac_decoder_cfg.d_bev` = **1024 for resnet101, 256 for resnet34**. A mismatch is refused by name on the forward, not surfaced as a bare `mat1 and mat2 shapes cannot be multiplied` a hundred frames into a pod run.
4. **The v8.1 label blob is NOT on this dev box.** Only the 2026-09-10 **v8.0** release is (`s2_labels_v8.0_train.jsonl.gz`, md5 `fa89ea55dfce68403eb30300e57852ab`). Every census below is measured on that blob and stamped with its md5. See §2 for why the counts are nonetheless trustworthy.
5. **The trainer call site is a one-line patch too** (`refc_v3_train.py` is another owner's). §6.4.

---

## 1. What is delivered

| # | thing | file | state |
|---|---|---|---|
| 1 | the DETR behaviour decoder, its losses, its per-class report | `stack/tanitad/refs/refcv6_tactical.py` | NEW, staged |
| 2 | the 4-way containing-window set speed + the stamp guard | `stack/tanitad/refs/refcv6_max_speed.py` | NEW, staged |
| 3 | nav compliance / speed ceiling / behaviour gate for SELECTION | `stack/tanitad/refs/refcv6_selection.py` | NEW, staged |
| 4 | the channel REGENERATION script + census | `stack/scripts/build_refcv6_speed_max_window.py` | NEW, staged |
| 5 | the acceptance instruments (T-FLIP, T-ZERO, obedience) | `taniteval/taniteval/refcv6_acceptance.py` | NEW, staged |
| 6 | config, build, hook, provenance, ledger | `stack/tanitad/refs/refc_v3.py` | MODIFIED, staged |
| 7 | 93 tests, **41 mutations** | `stack/tests/test_refcv6_tactical.py` | NEW, staged |
| 8 | the `refc.py` wiring | `…/2026-09-16-refcv6-tactical/refcv6_refc_integration.patch` | PATCH, staged — **not applied** |
| 9 | this file | `…/2026-09-16-refcv6-tactical/RESULT.md` | NEW, staged |

**Test run, MEASURED:** `83 passed, 1 warning in 2.43s` **with the patch applied**.
On the tree as STAGED (patch NOT applied): `79 passed, 4 skipped` — the four end-to-end tests skip with the reason *"refc.py integration patch not applied"*. ⛔ A skip is not a pass, which is why the number is printed here rather than left to the reader to discover; the 33-row mutation table below was generated with the patch applied and four of its rows (1, 2, 3 and the `scene_hook` refusal) are unreachable without it.
Neighbouring modules (`test_refc.py`, `test_refc_v3.py`, `test_refc_select.py`, `test_refc_v3_rollability.py`, `test_speed_max_derivation_stamp.py`, `test_refc_tactical.py`, `test_refc_v3_refcv5_wiring.py`, `test_max_speed_input.py`): **197 passed, 1 failed**. ⚠️ That one failure — `test_refc_select.py::test_trainer_runs_every_lever_and_reports_the_selection_family`, `ValueError: min() iterable argument is empty` inside `torch/backends/cudnn/__init__.py:89` — was **verified PRE-EXISTING**: it fails identically on the reverted, unpatched tree (md5 `ee3468b6ea232a16ec77cdd8393060bc`). It is a CPU-only-box cudnn probe, not a regression.

---

## 2. The per-class label census — MEASURED, and it reproduces the spec

Measured through `v7_labels.goal_supervision_census`, i.e. through the **same negative policy the trainer uses** (`negatives="measured"`), not through naive presence/absence.

- **Source:** `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-09-10-v8-speed-max-label-release/raw/s2_labels_v8.0_train.jsonl.gz`
- **md5** `fa89ea55dfce68403eb30300e57852ab` · **n = 4,572 clips** · `release: v8.0`, `schema: s2-geom-v7`
- ⚠️ **This is NOT the v8.1 blob** the spec's census used (`b45377a1…`); that file is not on this dev box (searched `C:` and `D:`). Evidence class: **MEASURED on a named, hashed, locally-present artifact that is one release behind the spec's.**

| token | pos | neg | ignored | trainable | scoreable (n ≥ 200) |
|---|---:|---:|---:|---|---|
| `FOLLOW_LANE` | 3629 | 943 | 0 | YES | YES |
| `TURN_L` | 275 | 4297 | 0 | YES | YES |
| `TURN_R` | 259 | 4313 | 0 | YES | YES |
| `YIELD_FOR_TURN_L` | 21 | 4551 | 0 | YES | **NO** |
| `YIELD_FOR_TURN_R` | 20 | 4552 | 0 | YES | **NO** |
| `YIELD` | 609 | **0** | 3963 | **NO** | YES |
| `STOP_POINT` | 327 | 4245 | 0 | YES | YES |
| `SPEED_BAND` | 4572 | **0** | 0 | **NO** | YES |
| `CORRIDOR_OFFSET` | 860 | **0** | 3712 | **NO** | YES |
| `EVADE_IN_CORRIDOR` | 240 | 20 | 4312 | YES | YES |
| `OVERTAKE_VEHICLE` | 20 | 3635 | 917 | YES | **NO** |
| `MERGE` | 79 | 3629 | 864 | YES | **NO** |
| `GAP_TARGET` | 368 | **0** | 4204 | **NO** | YES |
| `REACT_ON_ONCOMING` | 333 | **0** | 4239 | **NO** | YES |
| `TAKE_EXIT_L` | 21 | 369 | 4182 | YES | **NO** |
| `TAKE_EXIT_R` | 128 | 290 | 4154 | YES | **NO** |
| `TRAFFIC_LIGHT_REACT` | 18 | 761 | 3793 | YES | **NO** |
| `TRAFFIC_LIGHT_REACT_RED` | 376 | 403 | 3793 | YES | YES |
| `TRAFFIC_LIGHT_REACT_YELLOW` | 22 | 757 | 3793 | YES | **NO** |
| `TRAFFIC_LIGHT_REACT_GREEN` | 363 | 416 | 3793 | YES | YES |
| `LANE_CHANGE_L` | 23 | 290 | 4259 | YES | **NO** |
| `LANE_CHANGE_R` | 15 | 282 | 4275 | YES | **NO** |

**⇒ 17 of 22 trainable; 5 masked for lack of a supervised negative** — `YIELD`, `SPEED_BAND`, `CORRIDOR_OFFSET`, `GAP_TARGET`, `REACT_ON_ONCOMING`. **The spec's "17 of 22, 5 masked" reproduces exactly**, on a blob one release older, and the five are the same five `tac_goal_head.mask_report`'s docstring names.

**⇒ 10 of 22 under the n = 200 scoreability floor** — and they are **token-for-token** `vocab_v7.TACTICAL_GOAL_UNDERPOWERED`. The spec's "10 under the floor" reproduces exactly.

**⇒ only 7 tokens are BOTH trainable and scoreable:** `FOLLOW_LANE`, `TURN_L`, `TURN_R`, `STOP_POINT`, `EVADE_IN_CORRIDOR`, `TRAFFIC_LIGHT_REACT_RED`, `TRAFFIC_LIGHT_REACT_GREEN`. **That is the number to report the head under, and it is smaller than either of the two numbers the spec quotes.**

### 2.1 ⚠️ A correction I owe, and the reason it matters

My first census used naive presence/absence ("absent ⇒ negative") and got **13 of 22 trainable, only 1 token with no negatives**. That contradicted the spec. It was **my measurement that was wrong, not the spec**: absence is a negative only for the 7 tokens this blob emits from *geometry* (`FOLLOW_LANE`, `SPEED_BAND`, `STOP_POINT`, `TURN_L`, `TURN_R`, `YIELD_FOR_TURN_L/R`); for the 15 `vlm-cot` tokens absence means *"the caption did not say so"*. Going through `goal_supervision_census` — the function the trainer uses — reproduced the spec to the token. **Stated because the wrong number was nearly reported as a contradiction of the spec.**

### 2.2 lat/lon actions (`a_tac.lat` / `a_tac.lon`, per clip, same blob)

| lateral | n | | longitudinal | n |
|---|---:|---|---|---:|
| `LANE_KEEP` | 2958 | | `CRUISE` | 1243 |
| `NUDGE_R` | 592 | | `ACCELERATE` | 998 |
| `NUDGE_L` | 488 | | `ADAPT_SPEED_FOR_CURVE` | 995 |
| `TURN_L` | 275 | | `BRAKE_TO` | 905 |
| `TURN_R` | 259 | | `FOLLOW` | 165 |
| **`LANE_CHANGE_L`** | **0** | | **`CREEP`** | 135 |
| **`LANE_CHANGE_R`** | **0** | | **`HOLD`** | 131 |
| **`ABORT_LC`** | **0** | | **`YIELD_MERGE`** | **0** |

⛔ Four classes have **ZERO** positives. A recall of 0.0 on `LANE_CHANGE_L/R`, `ABORT_LC` or `YIELD_MERGE` is a **LABEL fact, not a model fact** — `per_class_report` stamps exactly that sentence on those rows. And the labels are supervised only within the record's **±2 s band** (`v7_labels.window_in_band`, derived from the record's own bands, not hardcoded), which on the eval grid is **1,157 of 4,823 windows**.

---

## 3. The architecture, and what changed against the spec

### 3.1 The decoder (spec §4, as built)

2-layer pre-LN DETR decoder, d = 256, 8 heads, ff ×4.

- **38 queries** = 22 behaviour + 8 lateral + 8 longitudinal. One query per output: a rare class (`LANE_CHANGE_R`, 15 positives) gets a vector it owns, not a row of a shared 22-wide projection. The readouts are `Linear(d, 1)` applied per query, for the same reason.
- **Keys/values = agent slots ∪ BEV tokens**, each with its own input projection and a learned per-source code. ⛔ **There is no image-token port** — the exclusion is in the signature, which `test_decoder_has_NO_image_token_port` pins by introspecting `forward`'s parameter set.
  ⭐ **The number of keys is free.** The decoder concatenates whatever arrives and derives its pad mask from the tensor, so 16×40, 16×64 and 30×16 BEV grids all run on one build and any slot count works — MEASURED across all three grids × both backbone widths. Only the channel WIDTH is declared, because `bev_in` is a `Linear` and a checkpoint's shapes cannot be decided at runtime.
- **Self-attention among the queries is load-bearing**, and is why this is a DETR decoder and not 38 MLPs: the outputs are a SET with hard structure (`TURN_L` excludes `TURN_R`; `STOP_POINT` co-occurs with `BRAKE_TO`; lat and lon must stay separable — `refc_tactical.py`'s F2). The frozen exclusion table supervises that only indirectly, through entailed negatives, so the capacity has to exist.
- **FiLM on the queries at every layer** from `[nav(4), max-speed(4), v0/10, a0/3]`, zero-init on the delta (`gamma = 1 + dgamma`). A fresh decoder is **bit-identical under any condition** — pinned, and the pin is proven able to fail (mutation 13).
- **Validity is 22 independent sigmoids under BCE**, never a softmax: 2–7 tokens per record, mean 2.751. Collapsing a SET into a CHOICE is the 5-way manoeuvre defect rebuilt one layer up.

### 3.2 ⭐ One deviation from the spec's literal wording, and why

The spec lists **confidence** as an output but names only three loss terms (BCE 0.05, CE 0.025 ×2). Built that way, **`conf_head.weight.grad` was `None`** — 1,430 parameters that exist, are emitted, and are consumed by selection, with no gradient able to reach them. That is exactly the failure measured programme-wide on 2026-09-06 (42 of 138 optimizer tensors, 52.2 % of a declared budget). **A mutation test found it; it was not reasoned about in advance.**

⇒ The confidence is supervised as **self-assessment** — "will the validity decision on this token be correct?" — with the validity logits entering **detached**, so the confidence head cannot lower its own loss by making the validity head predictable instead of right. It is a **SUB-SPLIT of the goal BCE budget, not a fourth term**:

| term | weight |
|---|---:|
| goal validity BCE | 0.040 |
| goal confidence BCE | 0.010 |
| **goal total** | **0.050** |
| lateral CE | 0.025 |
| longitudinal CE | 0.025 |
| **TOTAL** | **0.100 = `refc_train.MANEUVER_WEIGHT`** |

Nothing is inflated. `assert_within_budget` refuses a re-weighting that would, and the test reads `MANEUVER_WEIGHT = 0.1` out of `refc_train.py` rather than typing it.

### 3.3 Parameter ledger — MEASURED

At the spec's geometry (d = 256, 2 layers, agent tokens at the REF-C base decoder width 384, 117 anchors). ⭐ `bev_in` is the ONLY line that moves with the PI's backbone choice; everything else is identical across the two arms.

| module | params |
|---|---:|
| `queries` (38 × 256) | 9,728 |
| `agent_in` (384 → 256) | 98,560 |
| `bev_in` — **resnet101** (1024 → 256) | 262,400 |
| *`bev_in` — resnet34 comparison run (256 → 256)* | *65,792* |
| `source_code` | 512 |
| `kv_norm` | 512 |
| `layers` (2 × [self-attn, cross-attn, FFN, FiLM]) | 2,118,144 |
| `out_norm` | 512 |
| heads (validity, confidence, lat, lon) | 1,028 |
| *(of `layers`: FiLM)* | *11,264* |
| **`tac_decoder_v6` total — resnet101** | **2,491,396** |
| *`tac_decoder_v6` total — resnet34* | *2,294,788* |
| `tac_behaviour_gate_v6` (22 → 117, bias-free, zero-init) | 2,574 |
| `tac8_lat_to_anchor` + `tac8_lon_to_anchor` (8 → 117 ×2, zero-init) | 1,872 |
| `navc_gate` | 1 |
| `MaxSpeedOneHotEncoder` | **0** |
| **refcv6 tactical TOTAL — resnet101** | **2,495,843** |
| *refcv6 tactical TOTAL — resnet34* | *2,299,235* |

Agents-only (no BEV port yet, `d_bev = 0`): `tac_decoder_v6` = **2,228,996** — the state every refcv6 arm runs in until the perception agent's lift lands.

⛔ `tac_decoder_v6` and `tac_behaviour_gate_v6` get **separate `param_breakdown_v3` ledger lines**, read off the built objects. That is the D-ROLL-1 rollability contract: parameters that exist unaccounted make every banked checkpoint unrollable through `refcv3_arm.cross_check_config`. `MaxSpeedOneHotEncoder` is deliberately absent from the ledger because it has zero parameters, and the code says so.

---

## 4. Max speed — REGENERATED under the PI's containing-window rule

**PI 2026-09-16:** *"I think we need to revise and generate the max speed values in the data set according to my new logic and its ok to derive this from future ego data but put in the right quantization window."*

Rule: `(0,30] → 30`, `(30,50] → 50`, `(50,100] → 100`, `(100,120] → 120`, `> 120` **clamps to 120 and is counted**. It models a max **SET** speed — a limiter — so the realised speed sits INSIDE the window and never above it. ⛔ Not a nearest-value rule: 96 km/h → 100.

**Regeneration script:** `stack/scripts/build_refcv6_speed_max_window.py`. ⛔ It writes a **SIDECAR** and refuses `--out` == the label blob; the v8 manifest's `hash_authority` block exists to forbid rewriting a hashed release in place.

**MEASURED, train** (source md5 `fa89ea55dfce68403eb30300e57852ab`, 4,572 clips):

| window | n | share |
|---|---:|---:|
| (0, 30] → **30 km/h** | 1,741 | 38.08 % |
| (30, 50] → **50 km/h** | 1,593 | 34.84 % |
| (50, 100] → **100 km/h** | 1,014 | 22.18 % |
| (100, 120] → **120 km/h** | 224 | 4.90 % |
| … of which **CLAMPED** (v_hi > 120) | **86** | **1.88 %** — inside the 120 bucket, not a fifth group |
| **clips with NO speed band** | **0** | — |

entropy **1.7555 bits** · snap-up slack p50 **12.32** km/h, p95 **44.60** km/h · corpus max v_hi **136.08** km/h.
**MEASURED, eval** (md5 `38a6239903478c078702d12b518b4bac`, 147 clips): 49 / 52 / 39 / 7, **1 clamped**, 0 without a band, H = 1.7757 bits.

**Sidecars written** (scratchpad; the script is the deliverable, the sidecars are reproducible from it):
`speed_max_window_v6_train.jsonl` md5 `5e16d0593f24eb353c6c3057c3caba64` · `speed_max_window_v6_eval.jsonl` md5 `af4f18309bc290c362870b250d488bee`.
⚠️ **Re-run against the v8.1 blob when it reaches the box.** No code changes; only the stamped source md5 does.

**Cross-check that makes this MEASURED and not merely computed.** Three numbers in `max_speed_input.py`'s docstring — banked independently, from a module that never read this blob — reproduce here to the digit: *"224 clips (4.90 %) have v_hi > 100 km/h"*, *"38.1 % of clips have v_hi ≤ 30 km/h"* (38.08 %), *"the corpus's maximum v_hi is 37.803 m/s = 136.1 km/h"* (136.08).

**The 4-value ladder is coarser than the 8-step one, and that is the point:** H drops 2.480 → **1.7555 bits**, p95 slack 19.4 → **44.60** km/h. It still does not launder the derivation, and the stamp says so.

**The obedience-test population is real:** **1,962 of 4,572 clips (42.91 %)** have a realised maximum above 40 km/h.

### 4.1 The stamp, and both refusals

`speed_max_derivation_v6` must contain all seven of: `oracle`, `ego-future`, `SPEED_BAND.v_hi_ms`, `[t0+2 s, +6 s]`, `{30, 50, 100, 120} km/h`, `4-way one-hot`, `CONTAINING-WINDOW`. Written as **literals**, never derived from the stamp they check.

- channel **ON**, stamp **absent** → `SpeedMaxStampError` (mutation 23)
- channel **OFF**, stamp **present** → `SpeedMaxStampError`, **the mirror case** (mutation 22)
- each required token deleted one at a time → refuses on **all 7** (mutation 24)
- E16's 8-step stamp under the 4-way key → refuses (mutation 21). ⛔ Two ladders may not share one key.

---

## 5. Nav, selection, and the acceptance instruments

### 5.1 Nav compliance is finally on the forward path

`refc_selector_targets.compliance_target:406-450` existed and was **MEASURED unreachable** (`REFCV6_CLARIFICATION.md` §3.1: *"the rule-based nav-compliance scorer exists but is not on the forward path"*). `refcv6_selection.nav_compliance_prior` is the adapter; the `refc.py` patch appends `navc_gate * compliance` to `r_terms`. Parameter-free predicate × **one zero-init scalar** — the `lan_gate` argument: a geometric predicate cannot become a route-shaped shortcut the way a learned `nav → n_anchors` matrix could.

⛔ `tau_rad` has **no usable default**: 0.0 is the "not set" sentinel and the decoder refuses to build (mutation 1). It is derived per corpus by `nav_compliance.derive_tolerance(pos, neg)` and the arm must state it.
⛔ On `follow`/`straight` the predicate is **undefined, not failed** — the term is identically zero, so it cannot push the ranked score around on the ~75–79 % of windows with no commanded side.

### 5.2 The speed ceiling is an ARGMAX FILTER, not a score term

The bar is *"≥ 99 %"*. A soft penalty behind a **learned** gate cannot promise that — a zero-init gate is free to stay at zero. So the ceiling takes the shape `refc.py` already uses for a hard constraint (S2's reachability band): **it filters the argmax only**; `score` is returned unmasked so no `-inf` reaches a cross-entropy, and **a row with no survivor keeps its whole fan and is counted** (`speed_rows_empty`). ⚠️ Those rows are the **only** structural reason obedience can fall below 1.0 — the acceptance report prints the count beside the rate and the FAIL message tells the reader to separate a vocabulary limit from disobedience.

⛔ **The dt is DERIVED from `anchor_horizons`, never typed.** The waypoint grid is `(5, 10, 15, 20)` **ticks** at 0.1 s = **0.5 s apart**. A typed 0.1 reports every planned speed **5× too high** and would fail a perfectly obedient model — measured in mutation 26 (50.0 vs 10.0 m/s). Same derivation `refc.py` makes for its own prefix dt, and it says why.

### 5.3 ⛔ Admissibility: 5 of the 22 behaviour columns are STRUCTURALLY DEAD

PI 2026-08-03, still binding: `tac_SIT`, the traffic-light tokens and `YIELD` are auxiliary **TARGETS** only, never inputs to a goal or to selection. refcv6 is the first arm in which the behaviour SET reaches the ranked score at all, so the ruling now binds selection.

`BehaviourSelectionGate` is `Linear(22, n_anchors)`, bias-free, zero-init, with a **registered-buffer** admissibility mask applied **to the input, on every forward** (masking the weight would leave the refused columns' gradient live and one `optimizer.step()` away from mattering). **17 admissible / 5 refused.**

`assert_situation_columns_dead` is an **ACTIVE PROBE, not an inspection**: it perturbs the gate's own weights in place, drives the refused columns to 1e3 and refuses if the output moves, then drives the admissible ones and refuses if it does **not**. The two measured deltas travel with the model as `tac_gate_mutation_proof_v6`.

⚠️ **The first version of this guard was broken and the mutation test caught it.** It probed a *reconstruction* (`load_state_dict` into a fresh module), so a gate whose `forward` had been replaced by an unmasked one still passed. A guard that probes a copy cannot see a defect in the object. Fixed to probe `gate` itself, with a bitwise `finally` restore so the build-time call cannot un-do the zero-init.

`provenance_roles` gains a **`selection_inputs`** key, and `assert_situation_tokens_are_targets_only` **refuses a declaration that omits it** — an absent key is not a pass (mutation 20). It is called from `RefCV3Model.__init__`, so it runs on every build, not only where someone runs the interventional audit.

⚠️ **A second guard defect, also caught by its own test, also stated.** The matcher originally carried the bare markers `"sit"` and `"situation"` as substrings — and it **refused its own correct declaration**, because the prose *"the 5 situation columns are structurally dead"* contains the word. A guard that fires on a description of itself is not stricter, it is broken: the only way to ship would have been to weaken it. The identifiers are now matched **as identifiers** (word-bounded), plus the explicit phrase forms. The real enforcement is the structurally dead columns; this check is the declaration half.

### 5.4 The acceptance instruments (coded; the GPU runs later)

| test | bar (FROZEN 2026-09-16, before any arm trains) | refcv5-v2 today |
|---|---|---|
| **T-FLIP** | plan follows the FED (flipped) command ≥ **0.50** AND true − shuffled ≥ **0.38**, both **separated** | 0.205 / +0.099 |
| **T-ZERO** | ⛔ **no bar — a reported DIAGNOSTIC on the NON-turn classes** | TURN_L 0.475 → 0.000 zeroed |
| **speed obedience** | planned max ≤ the forced 30 km/h on ≥ **0.99** of windows whose GT max > 40 km/h, ADE cost reported | never run |

**T-FLIP.** `refcv3_arm.py --with-navflip` (`taniteval/tools/refcv3_arm.py:1888-1898`) already rolls the arm, and `nav_compliance.compliance_arm` already computes `flipped.follows_FED_command`. **It has never been run on any REF-C arm.** What was missing is the verdict function, and it is now `tflip_verdict`.
⛔ **The verdict reads the CI LOWER BOUND, not the mean.** refcv5-v2's flip block had **n = 39 windows**; a PASS off the point estimate at that n is the winner's-curse read. Mutation 30 pins that a block with mean 0.62 / lo 0.31 reads **FAIL**. An absent `flipped` block is **NOT_RUN**, never FAIL — that would bank a verdict about a model from an absence of data about it. Below 30 windows / 5 episodes it is **UNPOWERED** and says so with the n.

**T-ZERO, and the PI's 2026-09-16 ruling — stated here so no later reader mistakes the turn numbers for a defect.**
> *"its totally fine to process the nav command and generate from it the turing command, we are not to aim in this stage to generate a route."*

**Deriving `TURN_L`/`TURN_R` from the nav command is BY DESIGN.** refcv5-v2's tactical turn head collapses to recall **0.000** with nav zeroed; under this ruling that is **admissible, not a failure**. Nothing in this delivery penalises or gates the turn classes on nav-independence.
⇒ Which is exactly why the turn classes cannot carry the diagnostic: **42.7 % of the TURN label's entropy is already in the nav token** and *no clip has `NAV_FOLLOW_ROAD` with a tactical turn (0 of 2,897)*. T-ZERO therefore **reports the turn classes but excludes them from its summary**, and the classes that carry it are the ones nav cannot explain — the longitudinal actions (nav explains **5.2 %**), the speed bucket (**5.5 %**), lane keeping, nudges, yielding, gap targets. That is where "learned from the scene" has to show.

**Speed obedience.** ⛔ **The ADE cost is part of the result, not a footnote.** A model can obey any ceiling by planning to stop. With no `ade_forced`/`ade_baseline` supplied the block reads `UNAVAILABLE` with the reason *"report both or report neither"*.

**The panel.** A missing instrument makes the panel **INCOMPLETE**, never a 2-of-3 pass (mutation 14). T-FLIP and OBEDIENCE gate; T-ZERO is a diagnostic and carries no verdict.

**The strategic layer stays DEACTIVATED** — route head, `g_str`, strategic GRU — per the PI (*"we are not to aim in this stage to generate a route"*). Nothing in this delivery builds, reads or scores a route.

---

## 6. ⭐ APPLY HERE — the `refc.py` patch

`refcv6_refc_integration.patch`, 12 hunks. `git apply --check` is clean against `9a782fa`. Apply from the repo root.

| # | where | what |
|---|---|---|
| 1 | `RefCConfig`, after `graft_lan` | 5 flags: `graft_tac8_prior`, `graft_behaviour_sel`, `graft_nav_compliance`, `nav_compliance_tau_rad`, `speed_ceiling_filter`. All default OFF. |
| 2 | `AnchoredDiffusionDecoder.__init__` signature + body | the 5 flags; **new zero-init `tac8_lat_to_anchor` / `tac8_lon_to_anchor` (8 → n_anchors)**; `navc_gate` (zero-init scalar) with the tau refusal. ⛔ NEW layers, not a widening of `lat_to_anchor` — the 3-wide graft is LIVE in every banked checkpoint and resizing it makes refcv4b/refcv5 unrollable (D-ROLL-1, paid for once). |
| 3 | `AnchoredDiffusionDecoder.forward` signature | `tac_lat_prior`, `tac_lon_prior`, `behaviour_term`, `nav_cmd_sel`, `v_limit_ms`. |
| 4 | the confidence-surface prior block | the 8-wide pair **REPLACES** lat3/lon3, and passing both **raises** (mutation 2). |
| 5 | the ranked-score `r_terms` block | `behaviour_term` (arrives already projected and already masked — this file owns no copy of the admissibility mask) + `navc_gate * compliance`, with telemetry. **Deferred import of `refcv6_selection` here, and the cycle is the reason:** `refc → refcv6_selection → refc_selector_targets → refc` (`refc_selector_targets.py:94` imports `NAV_COMMANDS`). MEASURED, not guessed — the module-level form raised `ImportError: cannot import name 'NAV_COMMANDS' from partially initialized module`. |
| 6 | just before `idx = rank.argmax` | the speed-ceiling argmax filter, dt derived from `anchor_horizons`/`anchor_dt`. |
| 7 | `RefCModel.forward` signature | `scene_hook=None`, `bev_tokens=None`, `bev_pad=None`. |
| 8 | after `agent_pos = …` | **the scene hook call.** ⭐ This is the whole content of the patch. `hierarchy_hook` fires at `refc.py:3170`, the pooled/ctx stage; `agent_slots` are not decoded until `refc.py:3365`. The behaviour decoder's keys and values ARE those slots, so the existing hook would hand it a scene that does not exist yet. This is the only point where the scene is built and the operative decoder has not yet run. |
| 9 | the `self.decoder(...)` call | the five new arguments. |
| 10 | the decoder construction | the five new flags. |

**6.4 The trainer (`refc_v3_train.py`, another owner) — one call and one stamp:**

```python
from tanitad.refs.refcv6_max_speed import (SPEED_MAX_DERIVATION_V6,
                                           assert_speed_max_stamp_v6)
# beside the existing "speed_max_derivation" line in _run_config:
"speed_max_derivation_v6": (SPEED_MAX_DERIVATION_V6
                           if args.max_speed_onehot_v6 else None),
# beside the existing _assert_speed_max_stamp(_run_config, args):
assert_speed_max_stamp_v6(_run_config, on=bool(args.max_speed_onehot_v6))
```

and the loss, inside the existing `MANEUVER_WEIGHT` budget (it replaces the lat/lon halves, it does not add to them):

```python
from tanitad.refs.refcv6_tactical import tactical_behaviour_losses
loss_tac6, tac6_tele = tactical_behaviour_losses(
    out, goal_y=..., goal_w=...,          # v7_labels.tactical_goal_targets
    lat_target=..., lon_target=...,       # v7_labels.tactical_class_ids
    goal_pos_weight=v7l.goal_pos_weight(labels),
    goal_class_mask=mask_report(census)["mask"])
# `loss_tac6` is ALREADY weighted (0.05 + 0.025 + 0.025 = MANEUVER_WEIGHT)
```

⛔ **`--tac-decoder-v6` must also pass `--v7-labels`** (`_pin_trainer_cfg` pins `kin3` otherwise, and the build refuses `kin3`).

---

## 7. The mutation table — 41 mutations, generated from the test file

Rendered from `test_refcv6_tactical.MUTATIONS`, so the table and the tests cannot drift.

| # | guard | mutation applied | result |
|---|---|---|---|
| 1 | `AnchoredDiffusionDecoder.__init__` (patched) | graft_nav_compliance on with nav_compliance_tau_rad left at 0.0 | ValueError RAISED (the tolerance is DERIVED per corpus and must be stated) |
| 2 | `AnchoredDiffusionDecoder.forward` | BOTH the image-only lat3 prior and the tactical 8-wide posterior passed to one decode | ValueError RAISED ('the 8-wide pair REPLACES the 3-wide one') |
| 3 | `RefCModel.forward` (patched) | a scene_hook supplied while the forward built neither agent nor BEV tokens | ValueError RAISED (the decoder would attend to nothing) |
| 4 | `RefCModel.forward` scene_hook (patched) | run on the FLAT (hierarchy=False) branch, where hierarchy_hook cannot go | the hook IS called and the agent tokens reach it — the call site is after both branches merge |
| 5 | `RefCV3Model.__init__` | E16's continuous 8-step channel AND refcv6's 4-way one-hot both on | ValueError RAISED (two ceilings, two stamps, one condition) |
| 6 | `RefCV3Model.__init__` | tac_decoder_v6 on a FLAT (hier=False) build | ValueError RAISED (it would build and never be called) |
| 7 | `RefCV3Model.__init__` | tac_decoder_v6 under the kin3 vocabulary | ValueError RAISED (22 logits that could never be supervised) |
| 8 | `RefCV3Model.__init__` | tac_decoder_v6 with agents OFF and d_bev = 0 | ValueError RAISED (it would attend to nothing) |
| 9 | `RefCV3Model.__init__` | tac_goal_tok_head AND tac_decoder_v6 both on | ValueError RAISED (a confound, not an ablation) |
| 10 | `RefCV3Model.forward (ego_poses)` | ego_poses passed to a build with no ego-history encoder | ValueError RAISED (it would be silently dropped and read as +ego-history) |
| 11 | `RefCV3Model.forward (ego_poses)` | ego_poses[:, n_past:] corrupted by +1e4 (a blatant FUTURE read) | the emitted plan is BIT-IDENTICAL — the future index cannot enter |
| 12 | `RefCV3Model.forward (ego_poses)` | two ego histories 15 m/s apart passed through the WRAPPER on identical frames | the emitted plan differs, and `core._ego_window` stays None — the value went through the signature, not the one-shot workaround |
| 13 | `SpeedCeilingFilter` | a row whose every candidate exceeds the ceiling | the whole fan is kept and `speed_rows_empty` counts it — the only structural reason obedience can fall below 1.0 |
| 14 | `TacticalBehaviourDecoder.forward` | BEV tokens of the OTHER backbone's width (resnet34 256 vs resnet101 1024) fed to a declared build | SceneInputRefused RAISED naming both widths |
| 15 | `TacticalBehaviourDecoder.forward` | a [B, C, X, Y] BEV feature map passed where flat tokens are expected | SceneInputRefused RAISED (pointing at `bev_feats_to_tokens`) |
| 16 | `TacticalBehaviourDecoder.forward` | a scene tensor passed to a build with that port switched off | SceneInputRefused RAISED (both directions) |
| 17 | `TacticalBehaviourDecoder.forward` | called with agent_tokens=None AND bev_tokens=None | SceneInputRefused RAISED (would emit the unconditional prior) |
| 18 | `TacticalLossWeights.assert_within_budget` | weights raised to 0.15 (over the 0.10 budget) | ValueError RAISED |
| 19 | `_QueryFiLM zero-init` | FiLM projections re-initialised to N(0, 0.5) after the equality check | the two conditions now DIFFER -> the equality test can fail |
| 20 | `acceptance_panel` | one of the three instruments omitted | overall = INCOMPLETE (a 2-of-3 panel must not read as a pass) |
| 21 | `assert_scene_only` | 'image' added to the key/value source list | SceneInputRefused RAISED |
| 22 | `assert_situation_columns_dead` | BehaviourSelectionGate.forward replaced by one that SKIPS the admissibility mask | SituationInputRefused RAISED ('PI 2026-08-03 VIOLATED') |
| 23 | `assert_situation_columns_dead` | the gate's forward replaced by a constant-zero function | SituationInputRefused RAISED (the positive control catches a dead layer passing the refused-column check vacuously) |
| 24 | `assert_situation_tokens_are_targets_only` | 'TRAFFIC_LIGHT_REACT_RED probability' added to selection_inputs | SituationInputRefused RAISED |
| 25 | `assert_situation_tokens_are_targets_only` | 'tac_SIT one-hot' added to inference_inputs_of_goals | SituationInputRefused RAISED |
| 26 | `assert_situation_tokens_are_targets_only` | the `selection_inputs` key deleted from the declaration | SituationInputRefused RAISED (an absent key is not a pass) |
| 27 | `assert_speed_max_stamp_v6` | E16's 8-step stamp supplied under the 4-way key | SpeedMaxStampError RAISED (no ladder declared) |
| 28 | `assert_speed_max_stamp_v6` | channel OFF but config carries the stamp (the MIRROR case) | SpeedMaxStampError RAISED |
| 29 | `assert_speed_max_stamp_v6` | channel ON, stamp absent from config | SpeedMaxStampError RAISED |
| 30 | `assert_speed_max_stamp_v6` | each of the 7 required tokens deleted from the stamp, one at a time | SpeedMaxStampError RAISED on every one |
| 31 | `bev_feats_to_tokens` | a [B, P, C] token sequence passed as a grid | SceneInputRefused RAISED |
| 32 | `build_condition` | a SOFT (0.25 each) distribution smuggled into the nav / max-speed slots | ValueError RAISED (a richer input than the PI authorised) |
| 33 | `planned_max_speed` | the waypoint grid mis-declared as 1 tick apart instead of 5 | the reported speed is 5x (50.0 vs 10.0) — why the dt is DERIVED |
| 34 | `refcv6 selection seams under F3/F4` | the split-loop arms turned on (f3_per_layer=False, f4_adaln=True) on a v0-conditioned ddim build | all three seams still report — they are whole-fan terms outside BOTH layer loops, not per-layer modules |
| 35 | `refcv6 selection seams under F3/F4` | the split-loop arms turned on (f3_per_layer=True, f4_adaln=False) on a v0-conditioned ddim build | all three seams still report — they are whole-fan terms outside BOTH layer loops, not per-layer modules |
| 36 | `refcv6 selection seams under F3/F4` | the split-loop arms turned on (f3_per_layer=True, f4_adaln=True) on a v0-conditioned ddim build | all three seams still report — they are whole-fan terms outside BOTH layer loops, not per-layer modules |
| 37 | `speed_max_bin` | fed NaN (would silently bin to 0 = 30 km/h) | ValueError RAISED |
| 38 | `tactical_behaviour_losses` | all three weights set to 0.0 (the guarded-term failure mode) | p.grad is a ZEROS tensor on every parameter, never None |
| 39 | `tactical_behaviour_losses` (CE) | a batch entirely OUTSIDE the ±2 s band; torch's mean-reduction on the same input is NaN | finite 0.0 with n_supervised = 0 reported beside it |
| 40 | `tflip_verdict` | a block whose MEAN (0.62) clears the 0.50 bar but whose CI lower bound (0.31) does not | FAIL — the verdict reads the lower bound |
| 41 | `the refcv6 selection seams` | each zero-init gate forced open in turn (navc_gate 5.0, tac8 weights N(0,1), behaviour gate N(0,5)), scored on the CONTINUOUS ranked score and on the argmax | all three move `sel_score`; navc_gate moves traj=True, tac8_lat_to_anchor moves traj=False, behaviour_gate moves traj=True — gated, not dead, and every gate restores bit-identically |

### 7.1 Three defects the mutations found in my own code, before any GPU hour

1. **`conf_head` had no gradient** — 1,430 parameters emitted and consumed with nothing able to train them. Found by mutation 28. Fixed by supervising the confidence as self-assessment inside the existing budget (§3.2).
2. **`assert_situation_columns_dead` probed a reconstruction**, so an unmasked forward passed it. Found by mutation 16. Fixed to probe the object (§5.3).
3. **`build_condition`'s one-hot check was a row-sum check**, which a uniform `(0.25, 0.25, 0.25, 0.25)` block passes exactly. Found by mutation 25. Fixed to require every element ∈ {0, 1}.

None of these would have raised at build time. Two of them would have read as a *result* — "the confidence head does not help", "the behaviour set does not reach selection" — rather than as a bug.

---

## 8. What is NOT proven

- **No model was trained and no trajectory was scored.** Every acceptance bar above is a frozen threshold with no measurement against it. T-FLIP, T-ZERO and the obedience test have **never been run on any arm**.
- **The `refc.py` patch is not applied.** It was applied, exercised end-to-end, and reverted; the staged tree has the original `refc.py` (md5 `ee3468b6ea232a16ec77cdd8393060bc`).
- **The census is on v8.0, not v8.1.** It reproduces the spec's two headline counts exactly, but the exact blob the spec measured is not on this box.
- **No BEV tokens exist yet.** The port is built, and it is exercised at both backbone widths (1024 / 256) and three grids (16×64, 16×40, 30×16) — but with **synthetic tensors**. The real lift is the perception agent's, and nothing here has seen a real map.
- **The behaviour decoder has never seen a real agent slot.** The end-to-end test runs the smoke model (1-channel, 64 px, 20 anchors) with the learned detector's slots on random frames — it proves the WIRING, not the behaviour.
- **`tac_decoder_v6` costs 2.23 M parameters agents-only, 2.50 M with a resnet101 BEV port.** That ratio against the trunk is untested at either backbone.

---

## 9. DELIVERABLE MANIFEST

Staged on `agent/arch-inf-20260803` in `C:/Users/Admin/tanitad-wt-tactical-v6`. **Staged by path; never committed, never pushed.**

| path | state | in only one place? |
|---|---|---|
| `stack/tanitad/refs/refcv6_tactical.py` | NEW | yes |
| `stack/tanitad/refs/refcv6_max_speed.py` | NEW | yes |
| `stack/tanitad/refs/refcv6_selection.py` | NEW | yes |
| `stack/tanitad/refs/refc_v3.py` | MODIFIED | yes |
| `stack/scripts/build_refcv6_speed_max_window.py` | NEW | yes |
| `taniteval/taniteval/refcv6_acceptance.py` | NEW | yes |
| `stack/tests/test_refcv6_tactical.py` | NEW | yes |
| `TanitAD Research Lab/…/2026-09-16-refcv6-tactical/refcv6_refc_integration.patch` | NEW — **PATCH, unapplied** | yes |
| `TanitAD Research Lab/…/2026-09-16-refcv6-tactical/RESULT.md` | NEW | yes |
| `TanitAD Research Lab/…/2026-09-16-refcv6-tactical/speed_max_window_v6_train.meta.json` | NEW — the regeneration's own record (source md5 + census) | yes |
| `TanitAD Research Lab/…/2026-09-16-refcv6-tactical/speed_max_window_v6_eval.meta.json` | NEW — same, eval split | yes |

⛔ **NOT staged and NOT modified:** `stack/tanitad/refs/refc.py`, `refc_sampler.py`, `scripts/refc_v3_train.py` — other owners'. The `refc.py` wiring is the patch; the trainer wiring is §6.4.
⚠️ `stack/tanitad/refs/refc_v3.py` was already modified on this branch before I started; **I staged only my own hunks by path.**

---

## 10. ⭐ REBASE ONTO e95af00 — 2026-09-17

My tactical work landed as **cbadba5**. `refc.py` then moved twice: **8c7d215**
(core: timm trunk, K-frame history, ego-condition seam, F1–F9) and **e95af00**
(perception: DiffusionDrive coupling (1), `out["fmap_s16"]`). 3,576 → 4,035 →
**4,183** lines. The 12-hunk patch was cut against `9a782fa` and no longer
applied.

### 10.1 Apply status

**REBASED. `git apply --check` CLEAN against `e95af00`; applied, exercised, and
reverted.** 7 of 12 anchors were still exact; **5 moved** and were re-cut:

| hunk | what moved | how it was re-cut |
|---|---|---|
| 4 | decoder `forward` signature gained `ego_hist` (8c7d215) and `bev` (e95af00) | appended after both |
| 5 | **F7 now wraps every `[B, N_ANCHOR]` prior in `_tile`** | ⚠️ the tac8 grafts are `Linear(8, n_anchors)` — exactly that shape — so **both go through `_tile` too**. Untiled they would be a shape error on the F7 arm, or a silent broadcast. Pinned by `test_TRAP1b_the_tac8_prior_is_TILED_on_the_F7_widened_fan`, which reads the source line. |
| 8 | `RefCModel.forward` gained `ego_poses` / `ego_n_past` / `bev` | appended after all three |
| 10 | the `self.decoder(...)` call gained `ego_hist=ego_vec, bev=bev` | appended |
| 12 | F5 changed the `base` line to `refined if (sel.refined or self.rv6.f5_emitting_conf) else conf` | matched the new line |

### 10.2 The two traps — both CHECKED, not assumed

**TRAP 1 — F3/F4 split `_decode_ctrl` into two layer loops.** The refcv6-tactical
seams are **not per-layer**: three whole-fan terms (`terms` on the confidence
surface, `r_terms` on the ranked score) and one argmax filter, each appended
ONCE in `forward`, outside both loops. ⛔ That is an argument, so it is also a
test: `test_TRAP1_seams_still_fire_under_the_F3_F4_SPLIT_LOOPS` builds a
v0-conditioned `ddim` decoder at **(f3, f4) = (T,F), (F,T), (T,T)**, asserts the
cascade / AdaLN stack actually built, and asserts all three seams still report.
**3/3 pass.**

**TRAP 2 — `hierarchy=True` is the default path.** The `scene_hook` call site is
after both branches merge (beside `agent_pos`, ~200 lines below the
`if self.cfg.hierarchy` / `else`), so it is branch-independent by construction.
Tested on **both**: `test_TRAP2_the_scene_hook_fires_on_the_FLAT_hierarchy_branch_too`
drives the flat core directly (the v3 wrapper refuses a flat refcv6 build by
design, so it cannot reach that branch), and `test_TRAP2b` covers the default
hierarchy path.

⛔ The `scene_hook` contract is unchanged: `RefCV3Model.forward` passes the
keyword **unconditionally**, so an unpatched core raises `TypeError` rather than
silently running refcv5 under a refcv6 config.

### 10.3 Bit-identity on 64 fixed windows — MEASURED

Harness `bitid.py`: seed `20260917`, 8 batches × 8 rows, refcv6-tactical seams
**OFF**, run once against the patched tree and once against the `e95af00`
baseline (md5 `32da1253eaeb6ae11ba089ba93f7dfc3`), compared with `torch.equal`
(⛔ not `allclose` — a tolerance would let a real perturbation hide).

| tensor | shape | equal | max abs delta |
|---|---|---|---|
| `traj` | (64, 8, 2) | **True** | 0 |
| `sel_idx` | (64,) | **True** | 0 |
| `sel_score` | (64, 20) | **True** | 0 |
| `anchor_logits` | (64, 20) | **True** | 0 |
| `refined_logits` | (64, 20) | **True** | 0 |
| `offset` | (64, 20, 8, 2) | **True** | 0 |
| `maneuver_logits` | (64, 5) | **True** | 0 |

Parameter count **131,209** and state-dict key count **193** identical on both.
⇒ **BIT-IDENTICAL ON ALL 7 TENSORS ACROSS 64 WINDOWS.**

**And gated, not dead.** `test_the_seams_are_GATED_not_DEAD` forces each
zero-init gate open in turn and scores the **continuous** ranked score AND the
argmax:

| gate | moves `sel_score` | moves `traj` |
|---|---|---|
| `navc_gate` (5.0) | **yes** | yes |
| `tac8_lat_to_anchor` (N(0,1)) | **yes** | *no* |
| `behaviour_gate` (N(0,5)) | **yes** | yes |

Restoring every gate returns `traj` and `sel_score` bit-identical to baseline.

⚠️ **A defect in my own test, stated.** The first version asserted only on
`traj` — which is `fan[argmax]`, a DISCRETE pick over 20 anchors — and reported
`tac8_lat_to_anchor` as **dead**. It is not: it moves the log-posterior without
flipping the winner on this fixture. The liveness bar is now the continuous
surface and the argmax is reported beside it. A discrete readout cannot falsify
a continuous graft.

### 10.4 Test counts, with the control

| sweep | patched | control (UNPATCHED, same tree) |
|---|---|---|
| `test_refcv6_tactical.py` | 93 passed | — |
| `test_wp_index.py` | 35 passed | — |
| `test_refcv6_trunk.py` | 40 passed | — |
| `test_refcv6_diffusion.py` | 34 passed | — |
| `test_refcv6_perception.py` | 33 passed | — |
| **all five together** | **235 passed, 0 failed** | **224 passed, 11 skipped, 0 failed** |

⛔ **The patch breaks nothing: N = 0.** The control is reported anyway — the
11 skips are exactly the end-to-end and trap tests, which skip with the reason
*"refc.py integration patch not applied"* rather than failing. A skip is not a
pass, which is why both columns are printed.

### 10.5 ⭐ The one-line fix only I own — ego history through the v3 wrapper

`RefCModel.forward` grew `ego_poses` / `ego_n_past` in 8c7d215, but
`RefCV3Model.forward` does not forward unknown kwargs — so ego history could not
reach the core through the wrapper at all. The core shipped
`RefCModel.set_ego_window` as a workaround and **named the permanent fix in its
own comment**: *"one line in `refc_v3.py` — add `ego_poses` to its signature and
pass it to both `self.core(...)` calls"*. That workaround is a ONE-SHOT channel,
popped by the next forward, which makes the ego window a property of CALL ORDER
rather than of the call.

**Done**, at both call sites (flat early-return and the hierarchy path), plus two
silent-drop refusals mirroring the existing `ego_state` guard. Three tests:

1. **It delivers.** Two ego histories 15 m/s apart on identical frames produce
   different plans, **and `core._ego_window` stays `None`** — proving the value
   went through the SIGNATURE, not the workaround.
   ⚠️ Both zero-init projections have to be opened for this to mean anything —
   the encoder's `zero_init_out` **and** the decoder's `ego_to_cond`
   (`refc.py:1952-1957`). Ego history is a removable graft at BOTH ends; with
   either at zero the plan cannot move and the test would pass whether or not
   the value arrived. Stated, not hidden.
2. **A FUTURE index can never enter.** Corrupting `ego_poses[:, n_past:]` by
   **+1e4** leaves the plan **bit-identical**. If it did not, the channel would
   be an oracle.
3. `ego_poses` to a build with no encoder **raises**.

### 10.6 What §10 does NOT prove

- Still **no GPU hour, no arm trained, no trajectory scored**. Every acceptance
  bar in §5 remains unmeasured.
- Bit-identity was measured on the **smoke** model (1-channel, 64 px, 20
  anchors, 131,209 params) — the structural claim, not a 256×1024 resnet101 one.
- The F3/F4 trap test runs at the **decoder** level on a 5-anchor synthetic
  fixture, not a full model: it proves the seams are outside both loops, not
  that they help.
- `tac8_lat_to_anchor` moves the ranked score but **did not flip the argmax** on
  this fixture. That is a fixture fact, not a finding about the graft's value.
