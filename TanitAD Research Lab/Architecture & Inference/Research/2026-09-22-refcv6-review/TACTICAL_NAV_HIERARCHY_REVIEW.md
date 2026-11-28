# refcv6 adversarial review — TACTICAL LAYER · NAV · VOCABULARY · MULTI-HIERARCHY

**Reviewer:** independent adversarial reviewer (Architecture & Inference).
**Date:** 2026-09-22 · **Repo:** `D:/Projects/TanitAD` · **Tip at review:** `37645fc`
(branch `agent/arch-inf-20260803`).
**Dimension:** the tactical layer, the nav command, the vocabulary, and multi-hierarchy
consistency with the strategic layer OFF. Trunk/input, diffusion, perception/data and
training/guards belong to four siblings and are out of scope here.

**Environment (all measurements):** `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`,
torch **2.11.0+cu128**, Python 3.13.5, **CPU only**.
`tanitad.__file__ = D:\Projects\TanitAD\stack\tanitad\__init__.py` (asserted, not assumed —
the editable-install / G:-mount trap).
`taniteval` is a namespace package; the real one is `D:\Projects\TanitAD\taniteval\taniteval\`.

---

## 0. Verdict in one table

| # | question | verdict | evidence |
|---|---|---|---|
| 1 | nav reaches all three consumers | **YES, all three — MEASURED** (with one caveat: two of the three are behind zero-init gates, both proven *gated, not dead*) | `raw/q1_nav_reach.json`, `raw/q1b_denoising_passes.json`, `raw/q1c_zero_init_gates.json` |
| 2 | tactical k/v = agent slots + BEV, image ABSENT | **YES — MEASURED, with a mutation** | `raw/q2_tactical_kv.json` |
| 3 | strategic heads "not built" | ⛔ **FALSE. Every strategic head is BUILT in both arms; parameter delta = 0. And the route head is still SUPERVISED, putting gradient on 28/60 trunk tensors.** | `raw/q3_strategic_heads.json`, `raw/q3b_route_grad.json` |
| 4 | information-disjointness + the detach | **YES — MEASURED** (detach real; 5 situation columns structurally dead; the 8→N graft exists, is zero-init, and is genuinely FED) | `raw/q4_detach_disjointness.json`, `raw/q4b_tac8_fed.json` |
| 5 | vocabulary vs the newest dataset | **counts REPRODUCE exactly (17/5/10/7) — but on release `v8.0`, not the `v8.1` the SPEC cites. No v8.1 pack exists in this tree.** | `raw/q5_vocab_census.json` |
| 6 | T-FLIP / T-ZERO exist and can fail | **YES — 9/9 constructed cases match their literal expected verdicts, 5 of them deliberate regressions that go RED** | `raw/q6_tflip_tzero.json` |

⭐ **The one finding that should change something before an arm runs is #3.** Everything else
in this dimension is either correct as specified or a documentation defect.

---

## 1. Q1 — does nav actually REACH all three consumers?

**SPEC §5:** nav *"reaches: the condition (FiLM in every decoder layer, every denoising pass),
the tactical decoder, and **selection** via a parameter-free nav-compliance term behind a
zero-init gate."*

### Method

Interventional, not a code reading (`D-TLIGHT-1` is the registered defect where a label existed,
was documented, and never reached training). The same batch is run twice changing **only**
`nav_cmd` (1 = `NAV_TURN_L` → 2 = `NAV_TURN_R`), with forward hooks on each consumer. Every
positive carries a **same-nav control** in the same breath: two runs with identical nav must be
bit-identical at the same hooks, or a "difference" could be forward nondeterminism.

Model: a real `RefCV3Model` built by `code/build_refcv6_probe_model.py`, recipe inherited from
`stack/tests/test_refcv6_bev_tactical_wiring.py::_model` (timm `resnet18.a1_in1k`,
`pretrained=False`, 64×64, agent seam on, `tac_decoder_v6` on, BEV branch attached,
`graft_nav_compliance` on with `tau_rad = 0.35`). The tactical decoder is built `d_model=64,
n_layers=2` for CPU speed; **the shipped default is `d_model=256, n_layers=2`** — MEASURED from
`TacticalDecoderConfig()` — so the *structure* under test is the spec's, the *width* is not.

### (a) the FiLM condition — every decoder layer AND every denoising pass

**MEASURED** (`raw/q1_nav_reach.json`, `raw/q1b_denoising_passes.json`):

| arm | `opdec.layer0.film` calls | `opdec.layer1.film` calls | `cond` Δ (nav 1 vs 2) | same-nav control |
|---|---|---|---|---|
| `sampler="none"` | **1** | **1** | 0.177101 | **0.0** |
| `sampler="ddim"`, `diffusion_steps=2` | **3** | **3** | 0.177101 (both layers) | **0.0** |

⭐ **The phrase "every denoising pass" has no referent unless the sampler is built.** With
`decoder.sampler == "none"` the decoder stack runs exactly once, so a probe on that build would
have passed the whole clause while measuring only half of it. With `sampler="ddim"` each layer's
FiLM fires **3 times** = 1 classifier pass + 2 denoising passes, matching `diffusion_steps = 2`,
and `every_invocation_moves_with_nav` is **True for every invocation of both layers**. The three
invocations see `cond` sums `[-0.704352, -0.704352, -0.704352]` — the **same condition object**
(ids `[…885120, …882160, …882160]`), so it is one nav-carrying condition reused, not three
re-derivations that could drift.

Source of the loop, for the reader: `stack/tanitad/refs/refc.py:2381` and `:2389`
(`_decode_ctrl`, `for layer in self.layers`), condition built at `refc.py:2801`
(`cond = self.cond_proj(m)`), nav entering `m` at `refc.py:4017` / `:4073`.

**INFLUENCE** at step 0 is non-zero here — the operative FiLM is deliberately **not** zero-init
(`refc.py:1506`, `FiLM(cond_dim, d, zero_init=False)`, *"the core condition FiLM in the decoder
layers is LIVE … so the measurement/nav/v0 condition steers the decoder from step 0"*).
MEASURED output Δ: layer0 **0.182435**, layer1 **0.196717**; same-nav control **0.0**.

### (b) the tactical decoder

**MEASURED.** `cond` Δ at both `tacv6.layer*.film` hooks is **exactly 1.0** — the one-hot flip
arriving intact in the first four channels of `[nav(4), max_speed(4), v0, a0]`
(`refcv6_tactical.py:608`, `build_condition`) — with a same-nav control of **0.0**.

⚠️ At step 0 the tactical FiLM moves its output by **0.0**, because `_QueryFiLM` *is* zero-init by
design (`refcv6_tactical.py:340-347`). That is **arrival without influence**, and the two are
reported separately rather than collapsed. With the FiLM weights given non-zero values (the
post-training state), the same intervention moves `tacv6_lat_logits` by **0.5284**,
`tacv6_goal_logits` by **0.2722**, `tacv6_lon_logits` by **0.0762**, the emitted `traj` by
**53.08** and `sel_idx` by **13 anchors** — against a same-nav control of **0.0 on every one**.

### (c) selection

**MEASURED, and this is the strongest single row.** `refcv6_selection.nav_compliance_prior` was
wrapped and every call recorded:

| run | calls | nav values seen | `navc_frac_complying` | `navc_gate` |
|---|---|---|---|---|
| nav = 1 | 1 | `[1]` | **0.05** | 0.0 |
| nav = 2 | 1 | `[2]` | **0.95** | 0.0 |
| nav = 1 (repeat) | 1 | `[1]` | 0.05 | 0.0 |
| **nav = None** | **0** | — | — | — |

The term fires with the real nav tensor and its value **inverts** with the command
(`sel_tele.navc_frac_complying` 0.05 → 0.95). With `nav_cmd=None` the seam correctly does not
fire at all (`nav_cmd_sel=(nav_cmd if nav_cmd_given else None)`, `refc.py:4285`). Call site:
`refc.py:3163-3167`.

### ⭐ the follow-up the spec's own wording invites: gated, or dead?

A zero-init gate that receives no gradient never leaves zero, and this programme has MEASURED
that failure at *"42 of 138 tensors with a declared budget and no gradient"*. So all four
zero-init selection/prior grafts were backpropped through `sel_score_v3`
(`raw/q1c_zero_init_gates.json`):

| parameter | Σ\|grad\|, nav supplied | Σ\|grad\|, nav = None | reading |
|---|---|---|---|
| `core.decoder.navc_gate` | **17.164156** | **no gradient at all** | gated, not dead — **and the gradient is nav's**, which is the discriminating half |
| `tac_behaviour_gate_v6.proj.weight` | 152.953232 | 152.953232 | gated, not dead |
| `core.decoder.tac8_lat_to_anchor.weight` | 224.779846 | 224.779846 | gated, not dead |
| `core.decoder.tac8_lon_to_anchor.weight` | 224.831314 | 224.831314 | gated, not dead |
| control: no backward | **0 / None on all four** | — | so "non-zero" means something |
| control: `core.decoder.conf_head.weight` (live) | 6238.54 | 6238.54 | so a zero would be about the graft, not the backward |

⚠️ **A correction inside this instrument, recorded because it is the interesting part.** Its
first run read *"`tac_behaviour_gate_v6` receives NO gradient"* and that would have been filed as
a dead §4(b) seam. It was a **probe** defect: `graft_behaviour_sel` defaults to `False`
(`refc.py:826`) and I had not set it, so the term never entered selection. With the flag on it
reads 152.95. ⇒ **the seam is live when the arm asks for it; a refcv6 launch that omits
`--graft-behaviour-sel` gets §4(b) silently switched off** — though the trainer does refuse the
mirror error loudly (`refc_v3_train.py:845-853`, `--graft-behaviour-sel` without
`--tac-decoder-v6`).

### Q1 verdict

**Nav reaches all three consumers — MEASURED, with same-breath controls on every row.** The
caveat that must travel with it: at step 0 nav's *influence* on the tactical decoder and on
selection is exactly 0.0 by design, and only the operative FiLM steers from step 0.

---

## 2. Q2 — is the tactical decoder's key/value set what §4 says?

**SPEC §4:** *"Keys/values: the **agent slots** and the **BEV tokens** (30×16) … Image tokens are
deliberately **not** its input."*

**MEASURED** (`raw/q2_tactical_kv.json`), by hooking `_DecoderLayer.cross_attn` in
`refcv6_tactical.py:396` and reconciling the key count arithmetically against all three candidate
sources in the same forward:

| BEV cells fed | K (both layers) | derived agent slots | image tokens P (operative) | K − agent − bev |
|---|---|---|---|---|
| 12 | **112** | **100** | 4 | **0** |
| 25 | **125** | **100** | 4 | **0** |

`K` tracks the BEV count **exactly** (+13 for +13 cells) with the agent count constant at 100, so
the key set is `agent ⊕ bev` and nothing else. `n_queries = 38` at both layers = **22 goals + 8
lat + 8 lon**, matching §4. `tacv6_n_scene` equals `K` (no padded rows in this synthetic batch).

**Image tokens are ABSENT, and the exclusion is structural, not conventional — three independent
probes:**

1. `TacticalBehaviourDecoder.forward`'s parameters are exactly
   `(cond, agent_tokens, agent_pad, bev_tokens, bev_pad)` — **there is no image port**.
   Passing `image_tokens=` raises `TypeError` (`refcv6_tactical.py:493`).
2. Declaring `sources=("agent","bev","image")` is **REFUSED** with `SceneInputRefused`
   (`assert_scene_only`, `refcv6_tactical.py:187`). **GREEN control in the same breath:**
   `sources=("agent","bev")` is accepted.
3. The arithmetic above.

⭐ **A number in §4 that looks wrong and is not — stated so nobody re-derives it.** §1's diagram
says *"BEV feats 120×64"* and §4 says *"BEV tokens (30×16)"*. Both are right at their own layer:
`refcv6_perception_branch.CART_SHAPE = (120, 64)` is the BEV **feature** grid, and
`PerceptionBranchConfig.bev_tokens_hw = (30, 16)` (`refcv6_perception_branch.py:158`) is a
**parameter-free `adaptive_avg_pool2d`** down to **480 tokens** before the decoder sees them
(`:357-391`). ⇒ on the real arm K = `n_agent_slots + 480`.

**Q2 verdict: as specified.** The spec's §4 k/v claim holds at runtime and its violation is
refused at build time with a proven mutation.

---

## 3. Q3 — multi-hierarchy with the strategic layer OFF ⛔ **THE LOAD-BEARING FINDING**

**SPEC §1, verbatim:** *"⛔ Deactivated for this experiment: the whole strategic layer — route
head, `g_str`, strategic GRU. No head estimates the route (PI). **The flags remain but default
OFF and the heads are not built.**"* `PREREG_REFCV6_V2.md:185-186` repeats it verbatim.

### 3.1 The heads ARE built — in both arms

**MEASURED** from `state_dict()` / `named_parameters()`, never from the config
(`raw/q3_strategic_heads.json`):

| thing §1 names | source | params, default flags | params, `--no-strategic` |
|---|---|---|---|
| route head | `refc.py:3575` `self.route_head = nn.Linear(feat, N_ROUTE)` | **1,539** | **1,539** |
| strategic GRU | `refc.py:3419` `StrategicCtx(...)` | **25,576** | **25,576** |
| `g_str` head | `refc_v3.py:943` `self.str_goal_head = nn.Linear(d_ctx, 3)` | **27** | **27** |
| `g_str` → tactical FiLM | `refc_v3.py:946-950` | **608** | **608** |
| `ctx` → operative cond | `refc.py:1737` `ctx_to_cond` | **288** | **288** |
| **total model params** | | **14,777,262** | **14,777,262** |

**Δ parameters = 0.** Controls in the same scan read non-zero in both arms
(`core.decoder.layers.` 42 keys, `tac_decoder_v6.` 58 keys), so the absences would have been
visible had there been any.

And they are not merely present, they are **live**: `route_logits` and `g_str` are still emitted
under `--no-strategic` (`[4, 3]` each) and still **move with the input** — `route_logits`
Δ 0.0342, `g_str` Δ 0.0675 between two different frame batches.

⭐ **The implementation is internally consistent and says so**; it is the SPEC and the PREREG
that are wrong about it. `refc_v3_train.py:8842` documents the design as *"**BYPASS, never
delete**: every strategic …"* and `refc_v3.py:1497` as *"`g_str` is still COMPUTED and still
cached"*. ⇒ **the correct sentence is "the heads are built and bypassed", and it is a different
claim with different consequences** — checkpoint size, optimiser state, and the gradient question
below.

⚠️ **A second, separable defect in the same sentence.** `--no-strategic` is
`action="store_true"` (`refc_v3_train.py:8834`), so its default is **False = strategic ACTIVE**.
"The flags remain but default OFF" is therefore false in the other direction too: the bypass is
**opt-in**, and a refcv6 launch that does not pass `--no-strategic` runs with the whole strategic
layer switched on. **MEASURED absence with a control:** `--no-strategic` appears **0 times** in
`PREREG_REFCV6_V2.md` while the control terms `refcv6` (16 hits) and `strategic` (2 hits) read
non-zero in the same file, and the file carries **no launch command at all**. ⇒ the registered
pre-registration does not pin the flag that implements its own §7 sentence.

### 3.2 ⛔ The route head is still SUPERVISED, and its gradient reaches the shared trunk

**Source, positive assertion with its control:**

* `stack/scripts/refc_v3_train.py:3633` — `… + LAW_WEIGHT * loss_law + ROUTE_WEIGHT * loss_route
  + …`, with **no `no_strategic` guard**. `ROUTE_WEIGHT = 0.1` (`stack/scripts/refc_train.py:79`).
* `route_tgt = batch["route_target"]` is read **unconditionally** at `refc_v3_train.py:3368`, so
  the term is not label-gated either.
* **CONTROL, ten lines below, proving the search can find guards:**
  `refc_v3_train.py:3674` — the strategic-goal term **is** guarded:
  `if lan is not None and not bool(getattr(core, "no_strategic", False))`.

**And the trainer's own comment at `:3664-3667` states exactly why that guard exists:**
*"Supervising it anyway would push gradient through `str_goal_head` → `StrategicCtx` → the
**SHARED ENCODER**, i.e. the strategic layer would still shape the trunk that produces the plan.
That is a **SECOND VARIABLE inside a one-variable arm**."*

**That sentence is true of `route_head` too — MEASURED** (`raw/q3b_route_grad.json`, model built
`no_strategic=True`, backward of `0.1 * F.cross_entropy(route_logits, tgt)`):

| census | n tensors | non-zero grad | Σ\|grad\| |
|---|---|---|---|
| **trunk** `core.encoder.` | 60 | **28** | **91.66** |
| `core.route_head.` | 2 | 2 | 27.02 |
| `core.strategic.` | 6 | 0 | 0.0 |
| `tac_decoder_v6.` | 58 | 0 | 0.0 |
| **control** — no backward, trunk | 60 | **0** | 0.0 |
| **control** — trajectory loss, trunk | 60 | **28** | 14331.99 |

The no-backward control reads 0/60 so "non-zero" is meaningful, and the trajectory-loss reference
reaches the same 28/60 so a zero would have been about the route path, not about the trunk.

⭐ **What IS correctly gated, so the finding is scoped honestly rather than inflated:**
`route_prior` into selection is forced to `None` under the bypass
(`refc.py:4139-4140`), and `loss_gstr` is gated (`:3674`). So the defect is precisely one term:
**the route CE**. It does not reach selection and it does not reach the tactical decoder; it
reaches the **trunk**.

### Q3 verdict

⛔ **The SPEC's and the PREREG's "the heads are not built" is FALSE (MEASURED, Δparams = 0), and
"default OFF" is false in the opposite direction (the bypass is opt-in and unnamed in the
prereg).** The substantive consequence is one line: **`ROUTE_WEIGHT * loss_route` is ungated, so
a "strategic-layer-deactivated" arm still trains a route head whose gradient shapes the shared
trunk on 28/60 tensors** — the PI's *"We dont need any head to estimate the route"* is satisfied
at inference (no route prior reaches selection) and **not** satisfied during training.

**What would settle it, cheaply:** gate the route term the way `loss_gstr` already is, and stamp
`route_loss_applied` in `config.json` beside the existing `goal_str_loss_applied`. Or, if the PI
wants the route aux kept as a trunk regulariser, say so **in the SPEC** and stop calling the layer
deactivated — either is defensible, the current pair is not.

---

## 4. Q4 — information-disjointness and the detach

**SPEC §4:** *"Feeds the planner, detached: (a) the lat/lon posterior replaces the image-only
lat3/lon3 as the **anchor prior** (new zero-init 8→117); (b) the valid-behaviour set gates
selection."* **CLAUDE.md BINDING (PI 2026-08-03):** no situation-classifier output may feed a goal
input.

### D1 — the detach is real

**MEASURED** (`raw/q4_detach_disjointness.json`), planner-side loss `traj².mean()` backward:

| census | n | non-zero | Σ\|grad\| |
|---|---|---|---|
| `tac_decoder_v6.` | 58 | **0** | **0.0** |
| `tac_behaviour_gate_v6` | 1 | **0** | **0.0** |
| **control** `core.encoder.` (trunk) | 60 | 28 | 6582.52 |
| **control** `core.decoder.layers.` | 42 | 30 | 3786.03 |

and the **same 58 parameters** take gradient 56/58, Σ 5173.58 from the *tactical* loss. ⇒ the zero
is a detach, not a backward that reached nothing. `planner_feeds` returns five tensors and
`all(not t.requires_grad)` is **True** (`refcv6_tactical.py:660-676`).

**SPEC §11 R3 (tactical decoder may backprop into the trunk) is confirmed live:** the tactical
loss puts gradient on **28/60** trunk tensors (Σ 524.30) and on the perception branch (1/1).
⚠️ Which is also the attribution cost §11 names: the trunk is now optimised by planner + map +
box + tactical, and the per-head gradient-reach reporting must be ON for every arm.

### D2/D4 — the 8 → n_anchors graft exists **and is fed**

⚠️ **A graft that is built-but-never-called is bit-identical, in every artifact, to one that is
called — because zero-init makes its contribution 0.0 either way.** So "exists and is zero-init"
was not accepted as the answer. **MEASURED** (`raw/q4b_tac8_fed.json`):

* `tac8_lat_to_anchor` / `tac8_lon_to_anchor` built as `Linear(8, n_anchors)`, **weights all
  zero** (`refc.py:1776-1786`; `n_anchors = 117` on the real arm, 20 in this smoke bank).
* Both are **CALLED once** in a live forward, and the tensor each receives is **bitwise equal**
  (`max_abs_delta 0.0`) to `planner_feeds["lat_logprob"]` / `["lon_logprob"]` — the tactical
  decoder's own detached posterior, not something else of the same width.
* The image-only `lat_to_anchor` / `lon_to_anchor` are present but **NOT CALLED**: *"replaces,
  never adds"* holds at runtime, not just in prose.
* **MUTATION, with its GREEN control:** feeding both the 3-wide and the 8-wide prior is
  **REFUSED** (`refc.py:2911-2918`, *"BOTH the image-only lat3/lon3 prior and the tactical 8x8
  posterior reached the decoder"*), while the tac8-only call runs clean.

### D3 — disjointness, interventionally

`SITUATION_OUTPUT_TOKENS` = `{tac_SIT, YIELD, TRAFFIC_LIGHT_REACT, _RED, _GREEN, _YELLOW}`
(`refcv6_tactical.py:149-151`). Five of them are in the 22-token vocabulary (indices
`[5, 16, 17, 18, 19]`). Mutating **all five** columns of the behaviour vector fed to
`BehaviourSelectionGate` moves the selection term by **exactly 0.0**; mutating **one
non-situation** column in the same breath moves it by **0.0853**. ⇒ structurally dead columns,
and the test is not passing on a brick. The gate's weights were given live values first, for the
same reason.

`provenance_roles()` declares `selection_inputs` explicitly and `assert_situation_tokens_are_
targets_only` (`refcv6_tactical.py:206`) refuses a declaration that is missing it.

### ⚠️ One disjointness gap I am reporting rather than dismissing

`provenance_roles()["goal"]` still lists **`g_str`** — the strategic goal — as a goal node, and
Q3 shows `g_str` is still computed and still moves with the input under the bypass. It has no
in-graph consumer (`refc_v3.py:1497`) so it cannot *carry* the situation classifier anywhere, and
the PI's 2026-08-03 ruling is therefore not violated. But a declaration that names a node the
SPEC says is deactivated is the kind of drift the declaration exists to prevent.

**Q4 verdict: as specified, on every measured row.**

---

## 5. Q5 — vocabulary alignment with the newest dataset

⛔ **LAYER, stated before any count.** For labels the three layers are
**L1 published corpus** (PhysicalAI-AV — carries no tactical behaviour vocabulary at all),
**L2 episode build**, **L3 the augmented label release** (`s2_labels_v*.jsonl.gz`, **one record
per CLIP**). ⭐ **Every number below is L3, and within L3 the RELEASE STRING is its own scope.**

**MEASURED** via `v7_labels.load_v7_labels` + `v7_labels.goal_supervision_census`
(`negatives="measured"` — the trainer's own negative policy, not naive presence/absence);
`raw/q5_vocab_census.json`:

| pack (L3) | n records | md5 | trainable | masked | under n=200 | trainable **AND** scoreable |
|---|---|---|---|---|---|---|
| `s2_labels_v7.2_train.jsonl.gz` | 4,572 | `0ff902130ce7…` | **17/22** | 5 | **10** | **7** |
| **`s2_labels_v8.0_train.jsonl.gz`** | **4,572** | **`fa89ea55dfce…`** | **17/22** | 5 | **10** | **7** |
| `s2_labels_v7.2_eval.jsonl.gz` | 147 | `aa12c948f062…` | 13/22 | 9 | **22** | **0** |
| `s2_labels_v8.0_eval.jsonl.gz` | 147 | `38a623990347…` | 13/22 | 9 | **22** | **0** |
| `s2_labels_v8.1_train…` **as SPEC §4 cites** | — | — | — | — | — | ⛔ **DOES NOT EXIST** |

* **SPEC §4's "17 of 22 trainable (5 masked); 10 under the n = 200 floor" REPRODUCES EXACTLY** on
  the train pack. The 5 masked are `CORRIDOR_OFFSET`, `GAP_TARGET`, `REACT_ON_ONCOMING`,
  `SPEED_BAND`, `YIELD`. The 10 under-floor set **reproduces `vocab_v7.TACTICAL_GOAL_UNDERPOWERED`
  token for token** (`UNDERPOWERED_constant_matches_census: true`).
* Vocabulary widths match §4: **22 goal / 8 lat / 8 lon = 38 queries**, `COND_DIMS = 10`,
  `GOAL_MIN_N_FOR_METRIC = 200`.

### ⛔ The version mismatch

SPEC §4 says *"Losses: **v8.1 GT labels**"*. **MEASURED, by reading the `release` field of all
4,572 records:** the on-disk train pack declares **`release: "v8.0"` on 4,572/4,572** records
(schema `s2-geom-v7`, vocab `v7`). An **exhaustive key walk** over 200 records finds
`speed_max_input` on **4,572/4,572** but **zero occurrences of `tac_SIT`, `nav_30s` or
`lane_change_text`** anywhere in the record tree — three of the four blocks the v8.1 datacard
(`…/2026-09-01-v8-tacsit-release/DATACARD_v8.1_shipped.md`) attributes to v8.1.

**Absence probed at four locations with controls:** `find` for `*v8.1*`/`*v8_1*` returns only the
**datacard**, never a label file, while the control `find` for `*v8.0*` returns the packs; the
directory `…/2026-09-01-v8-tacsit-release/raw/` **does not exist** (`parent_exists: false`) while
its sibling `…/2026-09-10-v8-speed-max-label-release/raw/` does and holds the v8.0 pair.

⚠️ **Two readings, and I cannot separate them from this tree.** Either (i) the datacard's v8.1
release was superseded by a 2026-09-10 "v8.0" pack that dropped `tac_SIT` and added
`speed_max_input`, or (ii) v8.1 exists only on the private HF repo and was never pulled here.
**What would settle it:** the `release` field of the HF `labels/s2_labels_v8_train.jsonl.gz`, and
`V8_MANIFEST.json`'s md5 against `fa89ea55dfce68403eb30300e57852ab`. *(`V8_MANIFEST.json` is on
disk but is not cp1252-decodable — read it with `encoding="utf-8"`.)*

⭐ **The module is honest about this and the SPEC is not.** `refcv6_tactical.py:40-45` states its
census was taken *"on the locally available v8 train blob (`s2_labels_v8.0_train.jsonl.gz`, md5
`fa89ea55dfce68403eb30300e57852ab`, n = 4,572)"* — which my independent run **reproduces exactly**.
It is SPEC §4's `v8.1` citation that has no loadable referent here.

⚠️ **Consequence worth naming:** `tac_SIT` is one of the six `SITUATION_OUTPUT_TOKENS` the
admissibility guard protects. On the pack that is actually loadable **the field does not exist**,
so that third of the guard currently protects nothing. Harmless, but it means the guard's coverage
is untested against real data for that token.

⛔ **And a scoping fact the tactical metric family must carry:** on the **eval** pack (n = 147
clips) **all 22 tokens sit under the n = 200 floor and 0 are both trainable and scoreable**. ⇒
per-token AP on the eval split cannot clear the programme's own scoreability floor for **any**
token. The four-metric-families rule requires the tactical family to be reported per class with
its n; on this split every class must be printed as under-floor, or the family must be scored on
a larger split. This is a *measurement-plan* gap, not a model defect, and it is in my lane because
`PREREG_REFCV6_V2.md:235` lists the tactical family as gating on per-token AP.

---

## 6. Q6 — do T-FLIP and T-ZERO exist, and can they FAIL?

Instrument: `taniteval/taniteval/refcv6_acceptance.py` (408 lines).
⚠️ **Found only on the second probe** — the module is at `taniteval/taniteval/`, not
`taniteval/`; my first listing read "absent" and was a claim about my search.

**Bars, MEASURED from the module:** `TFLIP_BAR_FOLLOWS_FED = 0.50`,
`TFLIP_BAR_TRUE_MINUS_SHUFFLED = 0.38`, `TFLIP_TODAY_FOLLOWS_FED = 0.205` — matching SPEC §5.

**Nine constructed cases, each with a LITERAL expected verdict (never an expression over the code
under test). 9/9 agree** (`raw/q6_tflip_tzero.json`):

| case | expected | got |
|---|---|---|
| GREEN — clears both bars | PASS | **PASS** |
| **REGRESSION** — today's refcv5-v2 numbers (0.205 / 0.099) | FAIL | **FAIL** |
| **REGRESSION** — fed lower bound 0.49 (one step under the bar) | FAIL | **FAIL** |
| **REGRESSION** — delta lower bound 0.37 (one step under the bar) | FAIL | **FAIL** |
| **REGRESSION** — winner's curse: mean 0.72, CI lo 0.20 | FAIL | **FAIL** |
| **REGRESSION** — delta clears its bar but is NOT `separated` | FAIL | **FAIL** |
| absence — no `flipped` block | NOT_RUN | **NOT_RUN** |
| underpowered — 20 windows / 3 episodes | UNPOWERED | **UNPOWERED** |
| underpowered — `powered: False` | UNPOWERED | **UNPOWERED** |

⭐ **T-FLIP can fail, and it fails for the right reasons** — including the two that matter most:
it gates on the **CI lower bound** rather than the point estimate (the winner's-curse row), and it
requires `separated` on the paired delta. An absent `flipped` block is `NOT_RUN`, never `FAIL`,
which is correct: that is an absence of data about a model, not a verdict on it.

**T-ZERO is a DIAGNOSTIC and carries no bar by design** (SPEC §10.5), so "can it fail" is the
wrong question; "can it discriminate" is the right one. **MEASURED:** a constructed nav-echo
(`recall_zero = 0`) gives median retained **0.4875**, min **0.0**; a constructed scene-learned arm
gives median **1.0**, min **1.0**. Turn classes are excluded from the summary
(`n_classes_summarised = 2` of 4 rows) and classes under `min_n` are excluded with their n
reported. `has_pass_fail_bar: False`, as §10.5 requires.

**Reachability — checked, because a verdict function nothing calls is a brick.** The chain is
complete end to end: `refcv3_arm.py --with-navflip` (flag at `taniteval/tools/refcv3_arm.py:3563`,
flip built at `:1983-1995`) → condition `nav_flipped`
(`taniteval/taniteval/nav_compliance.py:126`) → the `flipped` block (`:455-462`) →
`tflip_verdict` → `acceptance_panel`, which gates on `T-FLIP` and `OBEDIENCE` and lists `T-ZERO`
under `diagnostic_tests`.

### Two documentation defects in this instrument

1. ⚠️ **A stale file:line.** `refcv6_acceptance.py:21` and `:128` both cite
   `taniteval/tools/refcv3_arm.py:1888-1898` for the `--with-navflip` block. The real site is
   **`:1983-1995`** (the file is 3,710 lines). `:128` is inside the operator-facing `NOT_RUN`
   message, so it is the line an operator will follow.
2. ⚠️ **`42.7 %` is quoted bare at four sites** (`:37`, `:206`, `:230`, `:406`), and `:230` ships
   inside the per-class `_note` a reader sees in the artifact. **SPEC §11 rules exactly this
   inadmissible as cited** — nothing in the repo derives it — and requires either the reproduced
   form (*"39.2 %, v7.2 eval, n = 147"*) or the asymmetry (`P(turn | NAV_FOLLOW_ROAD) = 0.0000`,
   `P(turn | NAV_TURN_*) = 0.2549`). The instrument predates §11 and was not swept.

---

## 7. What I could NOT answer

1. ⛔ **Whether the `v8.1` label release exists anywhere the arm can reach.** Probed at four
   locations with controls (§5); only a datacard is in this tree. **Settles it:** the `release`
   field of the HF `labels/s2_labels_v8_train.jsonl.gz`, and `V8_MANIFEST.json`'s md5 against
   `fa89ea55dfce68403eb30300e57852ab`.
2. ⛔ **Whether T-FLIP has ever been RUN on a real arm.** I proved the instrument is correct and
   reachable; I did not find a banked `flipped` block. The instrument itself says *"T-FLIP has
   NEVER been run on any REF-C arm"* (`refcv6_acceptance.py:129`) — **INHERITED, not re-verified
   by me**. **Settles it:** grep the banked `compliance_arm` JSONs for a `flipped` key.
3. ⛔ **Whether the nav-compliance `tau_rad` the arm will use is DERIVED.** `refc.py:1797-1801`
   refuses `tau <= 0` and demands `nav_compliance.derive_tolerance(pos, neg)`; my probe passed an
   arbitrary **0.35**, stated as such. I did not check which value a refcv6 launch would pass, and
   an invented tau changes what "complies" means. **Settles it:**
   `scripts/refcv7_derive_nav_tau.py`'s output for this corpus, stamped in `config.json`.
4. ⚠️ **Everything here is at `d_model=64` / `image_size=64` / 20 anchors / a stubbed BEV branch.**
   Wiring, gradient topology and refusals are geometry-independent and that is what I measured.
   **Not measured:** anything scale-dependent — attention cost at K = agent+480, the real
   117-anchor graft, or the resnet101 trunk (`SPEC §12.5`: it OOMs on this 8 GB box at batch 1).
5. ⚠️ **I did not run the repo's own refcv6 test suite** and deliberately did not take any green
   test as evidence; every claim above is an independent measurement. A sibling reviewer owns the
   guards/training dimension.

---

## 8. DELIVERABLE MANIFEST

All paths are in the repo working tree at `D:/Projects/TanitAD`, **staged, never committed,
never pushed**.

| artifact | path | what it is |
|---|---|---|
| this report | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-22-refcv6-review/TACTICAL_NAV_HIERARCHY_REVIEW.md` | the review |
| shared model builder | `…/2026-09-22-refcv6-review/code/build_refcv6_probe_model.py` | one real `RefCV3Model` recipe + `fingerprint()` for every probe |
| Q1 instrument | `…/code/q1_nav_reach_three_consumers.py` | interventional nav trace to the three consumers, with same-nav controls |
| Q1b instrument | `…/code/q1b_nav_every_denoising_pass.py` | FiLM invocation census under `sampler=none` vs `ddim` |
| Q1c instrument | `…/code/q1c_zero_init_gates_are_gated_not_dead.py` | gradient census on the four zero-init grafts + the nav-discriminating control |
| Q2 instrument | `…/code/q2_tactical_kv_set.py` | k/v arithmetic + image-port mutation with GREEN control |
| Q3 instrument | `…/code/q3_strategic_heads_built.py` | `state_dict` census of the strategic heads, both arms |
| Q3b instrument | `…/code/q3b_route_loss_reaches_trunk.py` | route-CE gradient into the shared trunk, with two controls |
| Q4 instrument | `…/code/q4_detach_and_disjointness.py` | detach census, graft init, situation-column intervention |
| Q4b instrument | `…/code/q4b_tac8_graft_is_fed.py` | is the 8→N graft actually FED + the both-priors refusal |
| Q5 instrument | `…/code/q5_vocab_vs_label_pack.py` | L3 label census by pack, by layer, with md5 |
| Q6 instrument | `…/code/q6_tflip_tzero_can_fail.py` | 9 constructed T-FLIP cases + T-ZERO discrimination |
| raw — Q1 | `…/raw/q1_nav_reach.json` | |
| raw — Q1b | `…/raw/q1b_denoising_passes.json` | |
| raw — Q1c | `…/raw/q1c_zero_init_gates.json` | |
| raw — Q2 | `…/raw/q2_tactical_kv.json` | |
| raw — Q3 | `…/raw/q3_strategic_heads.json` | |
| raw — Q3b | `…/raw/q3b_route_grad.json` | |
| raw — Q4 | `…/raw/q4_detach_disjointness.json` | |
| raw — Q4b | `…/raw/q4b_tac8_fed.json` | |
| raw — Q5 | `…/raw/q5_vocab_census.json` | |
| raw — Q6 | `…/raw/q6_tflip_tzero.json` | |

**Reproduce any probe:**
```
cd "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-22-refcv6-review/code"
PYTHONPATH="D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval" \
  C:/Users/Admin/venvs/tanitad/Scripts/python.exe <probe>.py
```

⛔ **Nothing in `stack/`, `taniteval/` or `Project Steering/` was modified by this review.**
Every finding above is a measurement; the fixes it implies are for the owning agent to make.
