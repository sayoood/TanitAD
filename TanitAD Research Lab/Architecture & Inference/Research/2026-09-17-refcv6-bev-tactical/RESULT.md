# refcv6 §4 — THE BEV→TACTICAL SEAM IS WIRED. PI ruling R2/R3 applied.

**2026-09-17 · Architecture & Inference · branch `agent/arch-inf-20260803`, from tip `ee1635a`**

⛔ **THIS IS A WIRING RESULT. NO CAPABILITY CLAIM, NO METRIC FAMILY.** The GPU is held by an RL
arm; every measurement here is CPU-only (`CUDA_VISIBLE_DEVICES=""`) and answers *"does the map
reach the behaviour decoder, and does its gradient reach the trunk?"* — not *"does it drive
better?"* The four binding metric families (longitudinal / lateral / tactical / strategic) are
**not computed and must not be inferred from anything below.**

---

## 0. Headline

| | |
|---|---|
| ✅ **The blocker is lifted.** | The BEV encoder now runs **inside** the model forward, so a BEV token exists where the scene hook fires. `--tac-decoder-d-bev 96` builds; a run stamps `sources: ["agent", "bev"]` and `bev_tokens_reach_decoder: true`. |
| ✅ **R3 is honoured, not quietly declined.** | A **tactical-only** loss moves the BEV lift (`859.0177`) and the BEV encoder (`30282.3473`) — MEASURED — so the tactical layer shapes the shared trunk. The detach ablation exists, is refused when inert, and is stamped. |
| ✅ **Bit-identity re-established against the NEW baseline** | `ckpt.pt` **BITWISE-IDENTICAL** over 193 tensors / 131,977 elements; `metrics.jsonl` **BYTE-IDENTICAL**; mutation control **DETECTS-EVERY-ONE-BIT-FLIP (193/193)**. |
| ✅ **10/10 guards proven by MUTATION** | none BLIND, none NOT-APPLIED, worktree verified restored by reading the bytes back. |
| ⛔ **AND THE HONEST HALF OF THE LIVE RUN** | on **4 of 10 steps the tactical decoder got EXACTLY 0 gradient** — 2,253,828 params, the `tac_goal_tok_head` signature. It is **label sparsity**, not wiring (`n_scene` stayed 496 throughout), and it is answerable only because the supervised-cell counts sit beside the reach row. §3b. |
| ✅ **It runs END TO END on the REAL corpus** | 10 steps, CPU, **real SAM3 map GT** (135 npz), real 3-D agent join, real per-clip extrinsics (141 clips). `config.json` stamps `sources: ["agent","bev"]` / `bev_tokens_reach_decoder: true` / `built: true`; **`n_scene = 496` on every step** (16 agent + 480 BEV). §3b. |
| ⛔ **BLOCKER FOUND ON RULING R1 — 408 × 1024 DOES NOT BUILD.** | `TimmTrunkConfig` refuses it (`408 % 32 == 8`), **correctly**. **416 × 1024 is legal and strictly better on the PI's own criterion.** Needs a PI decision. §5. |
| ⚠️ **19 tests regressed and were fixed** | adding two forward kwargs moved a signature-derived RL channel contract, disarmed a deliberate-regression anchor, and my own new guard pre-empted an older refusal. **Confirmed back to tip parity: failure and error SETS identical (70 / 4), passes 8,173 → 8,189.** §6b. |
| ⚠️ **A latent hole was found and closed** | a decoder built `d_bev > 0` that receives no tokens runs **agent-only without raising**. My change is what made that state reachable. §4. |
| ⛔ **R3's OWN MITIGATION WAS BLIND TO R3's GRADIENT** | the gradient-conflict detector's aux table listed `bev` / `map` / `box3d` and **not the tactical term** — so it could not see the fourth gradient the ruling is about. Fixed; changes `cd_*` semantics. §4b. |

---

## 1. The blocker, verified before it was lifted

⛔ I re-verified §E8's claim from source rather than inheriting it. All three parts hold at `ee1635a`:

* `refc_v3.RefCV3Model.forward` builds `_core_kw` carrying **`scene_hook` alone**;
* the BEV encoder lives on `model._perception`, attached by the trainer, and ran **after** the core
  forward on `out["fmap_s16"]` (`refc_v3_train.py`, the `_br(out["fmap_s16"], …)` call);
* `refc.py`'s scene hook fires at the agent-slot stage, ~250 lines **after** `fmap_s16` is built.

⇒ no BEV token existed at the hook site, `--tac-decoder-d-bev > 0` refused, and every arm stamped
`sources: ["agent"]`. **EVIDENCE CLASS: MEASURED (read from the tip's own source).**

## 2. What was built

| seam | file | what it does |
|---|---|---|
| `bev_hook=None` keyword | `stack/tanitad/refs/refc.py` | called where `fmap_s16` is born, **before** the scene hook. With `None` the block is untouched dead code — the same construction `hierarchy_hook` and `scene_hook` use, and the whole bit-identity argument. |
| `out["perception"]` | `stack/tanitad/refs/refc.py` | the branch's own outputs, emitted with the trunk's graph attached, **namespaced** so no future head can shadow a planner key. |
| `RefCV3Model._bev_hook` | `stack/tanitad/refs/refc_v3.py` | runs the trainer-attached branch inside the forward; carries the **feed gate** and the **detach ablation**. |
| `perception_grid` / `perception_valid` | `stack/tanitad/refs/refc_v3.py` | this batch's per-clip lift geometry, as **explicit arguments**. ⚠️ Deliberately *not* a one-shot setter: `set_ego_window`'s POP idiom exists only because that signature could not be edited, and this one can. |
| `PerceptionBranch.bev_tokens` | `stack/tanitad/models/refcv6_perception_branch.py` | `[B, d_out, X, Y] → [B, P, d_out]`. **Zero parameters.** Reuses `refcv6_tactical.bev_feats_to_tokens` — *"the ONLY place a BEV grid is flattened"* — rather than spelling a second transpose. |
| `--tac-decoder-bev-detach` | `stack/scripts/refc_v3_train.py` | R3's ablation. Default OFF = the ruling. Refuses when `d_bev 0` (nothing to detach). |

⛔ **The BEV token width is DERIVED, never typed.** It is `BEVEncoderConfig.d_out` = **96** — *not*
the trunk's stride-16 width. The old help text said "resnet34 → 256, resnet101 → 1024", which is
true of `fmap_s16` and **false of these tokens**: they are the *supervised map branch's* features.
A hand-set width now refuses with the derived number named. *(Same family as C-ANCHOR-UNITS: a
correct number in the wrong scope.)*

⚠️ **The token COUNT is 480** (`bev_tokens_hw = (30, 16)` pooled from the 120 × 64 SAM3 grid) — the
**same** pool `Box3DMemory` already applies, deliberately the same config field. Two pools that can
drift apart is how the box head and the tactical decoder would silently read different maps.

### The refusal did not disappear — it was replaced

The old blanket refusal was structural and is gone. What stands in its place is a **precondition**:

| argv | verdict |
|---|---|
| `--tac-decoder-d-bev 96` with `--w-map 0` | ⛔ REFUSED — with no live map weight the lift and BEV encoder are **not built**, so the decoder would declare a source it never receives. |
| `--tac-decoder-d-bev 256` with `--w-map 1.0` | ⛔ REFUSED — naming the derived width 96. |
| `--tac-decoder-bev-detach` with `d_bev 0` | ⛔ REFUSED — nothing to detach; a silently inert flag is the dead-flag class. |
| `--tac-decoder-d-bev 96` with `--w-map 1.0` | ✅ builds, `sources: ["agent","bev"]`. |
| `--tac-decoder-d-bev 0` | ✅ builds, `sources: ["agent"]` — **still agent-only and still reported as such.** |

## 3. Per-head gradient reach — MEASURED

`code/grad_reach_bev.py` → `raw/grad_reach_bev.json`. Real `RefCV3Model` through the **real trainer
pin**, real perception branch, real forward + backward. The loss is the decoder's own `tacv6_*`
logits and **nothing else** — no planner term, no map term, no box term — so a non-zero gradient on
the lift or the BEV encoder cannot have come from anywhere but the tactical layer.

| arm | trunk `grad_abs_sum` | trunk params w/ grad | `lift` | `bev_encoder` | `tac_decoder` | `n_scene` |
|---|---|---|---|---|---|---|
| `agent_only` (`d_bev 0`) | 78146.4538 | 55,085,120 / 58,231,872 | **0.0000** | **0.0000** | 41441.7107 | 100 |
| **`bev`** (the ruling) | 34211.1473 | **58,231,872 / 58,231,872** | **859.0177** | **30282.3473** | 30340.9518 | **580** |
| `bev_detached` (R3 ablation) | 33542.2325 | 55,085,120 / 58,231,872 | **0.0000** | **0.0000** | 30340.9518 | 580 |
| `bev` + **planner** loss (control) | 449597.6435 | 55,085,120 / 58,231,872 | **0.0000** | **0.0000** | **0.0000** | 580 |

**Three controls, and each must read a known value:**
1. `agent_only` — the pre-ruling arm. The BEV modules **exist** and the tactical loss cannot see
   them: **exactly 0.0**. ✅
2. `bev_detached` — R3's ablation cuts the path: **exactly 0.0**. ✅
3. **planner-loss control** — a real, non-tactical loss on the same model. The trunk moves
   (449597.6435) and the tactical decoder reads **exactly 0.0**. ✅ *Without this third row,
   "0.0 on the BEV modules" is indistinguishable from "this script never ran a backward" — the
   constant-control rule, and it is the row I would have omitted.*

### ⭐ The sharpest number, and it is not the one I set out to measure

**`trunk_params_opened_by_bev_path = 3,146,752`.** The planner reads **stride 32**, so its gradient
never enters the stride-16-only parameters; the BEV lift reads **stride 16**, so it does. Under
agent-only the tactical loss touches 55,085,120 of 58,231,872 trunk parameters (94.6 %); with the
BEV path live it touches **100 %**. R3 does not merely add a term — it opens **3.15 M trunk
parameters** the tactical layer previously could not reach at all.

⚠️ **Trunk `grad_abs_sum` FALLS** (78146 → 34211) while coverage rises. That is not a weaker signal:
the two arms have different decoders (2,228,996 vs 2,253,828 params — the `+24,832` is `bev_in`,
96→256 plus bias) and different key counts, so the magnitudes are not comparable across arms. **The
admissible reading of this table is the ZERO/NON-ZERO pattern and the parameter counts, not the
magnitudes.**

### Map-derived behaviours — the brief's own question

`d(goal_logits[:, j]) / d(bev_tokens)`, per v7 tactical token, on the `bev` arm:

| token | value |
|---|---|
| `FOLLOW_LANE` (lane keeping) | **10.7880** |
| `CORRIDOR_OFFSET` | **10.6437** |
| `EVADE_IN_CORRIDOR` | **10.6798** |
| all 22 tokens non-zero | **22 / 22** |

⇒ **The two behaviours §E8 named as *uninterpretable* under agent-only now have a gradient path to
the map.** Under `agent_only` the same probe reports `NO-BEV-TOKENS-FED`, and under `bev_detached`
`BEV-TOKENS-DETACHED` — three different absences, named separately, because collapsing them into
one `None` would make the finding unreadable.

⛔ **THIS IS REACHABILITY, NOT SELECTIVITY.** At initialisation the 22 values are near-uniform (an
untrained cross-attention attends roughly evenly). It says the map **can** shape each behaviour's
logit — evidence that structurally did not exist before. It says **nothing** about whether any head
learns to use it. That needs a trained arm and a per-class result; **none was run.**

⚠️ **Reproducibility, checked rather than assumed.** Re-running `grad_reach_bev.py` against the
final tree reproduces **every boolean verdict and every integer exactly** — including
`trunk_params_opened_by_bev_path = 3,146,752` — while ONE per-token magnitude moved in the 7th
significant figure (`EVADE_IN_CORRIDOR` 10.679841995 → 10.679841042). That is CPU float reduction
order, not a code difference. ⇒ **the per-token values are quotable as NON-ZERO, not to 7 digits.**

### Heads reading exactly 0 — named, per arm

* `map_head` reads **0.0 on every arm**, and that is **correct**: `MapHead` is a *sibling* consumer
  of `bev_feats`, not an ancestor of the tokens. ⚠️ I had it in the "must be non-zero" set on the
  first run and the predicate read `false` on a correctly wired rig — my error, kept in the code as
  a comment so the next reader does not repeat it.
* `planner` reads **0.0 under the tactical loss on every arm** — the scene hook detaches its planner
  feeds (the winner's-curse firewall). Expected; stated so it is not later read as a defect.

## 3b. THE LIVE ARM — real corpus, real SAM3 maps, CPU

`code/live.sh` → `raw/live_run/{config.json, metrics.jsonl, launch_excerpt.txt}`. `--device cpu`,
resnet34 @ 256 × 1024, on the **real** artifacts: the 139-clip v2 cache, **135 SAM3 map GT npz**,
the 3-D agent join, the 141-clip extrinsics table, v8.0 tactical labels.

⚠️ **TWO runs, and the excerpt names which line came from which** — a merged excerpt with one header
would credit the conflict-detector line to a run that had the detector off:

* **A — `bevtac`**, the shipped defaults: 2 steps, batch 2, **`--conflict-detector on`**. This is the
  run that shows the detector naming the fourth gradient.
* **B — `bevtac10`**: `--steps 10 --batch 4 --conflict-detector off`, to reach enough batches for a
  supervised tactical step to occur. **`raw/live_run/{config.json, metrics.jsonl}` are this run.**

⭐ **Why this is not redundant with §3.** §3 proves the gradient reaches the trunk on a hand-built
geometry. It does **not** prove the branch survives a real SAM3 label grid, a real per-clip lift
geometry, or the trainer's own loss assembly. This does, on the path an operator would launch.

**The run record says the right things**, without being asked twice:

```
sources                  ['agent', 'bev']       bev_tokens_reach_decoder  True
bev_grad_reaches_trunk   True                   bev_detached              False
decoder_cfg.d_bev        96                     built                     True
fmap_s16                 256 ch, [16, 64]       branch params             4,651,776
conflict detector ON (probe, plan side=traj): aux terms = box3d, map, tac_v6
```

⭐ **`tacv6_n_scene_mean = 496.0` on EVERY step** — 16 agent queries + 480 BEV tokens. The behaviour
decoder attended to the map on the real corpus, at every step, not just in a unit test.

⭐ **R3's ablation is runnable on the live rig too, not only in a unit test.** A third arm
(`bash code/live.sh bevtac_detach --steps 1 --batch 2 --conflict-detector off
--tac-decoder-bev-detach`) launches clean and stamps
`sources=['agent','bev'] · bev_tokens_reach_decoder=True · bev_grad_reaches_trunk=False ·
bev_detached=True` — the seam wired, the gradient cut, and the record saying exactly that. ⛔ A
ruling whose alternative cannot be launched is not testable, which is why this was checked rather
than asserted.

### ⛔ AND THE HONEST HALF: on 4 of 10 steps the tactical decoder got EXACTLY ZERO gradient

| step | `n_supervised_goal_cells` | `tac_v6` | `ga_tac_decoder` | `n_scene` |
|---|---|---|---|---|
| 1 | **0** | 0.00000 | **0.0000** | 496 |
| 2 | 8 | 0.13508 | 495.8695 | 496 |
| 3 | 14 | 0.14004 | 533.8138 | 496 |
| 4 | **0** | 0.00000 | **0.0000** | 496 |
| 5 | **0** | 0.00000 | **0.0000** | 496 |
| 6 | **0** | 0.00000 | **0.0000** | 496 |
| 7 | 17 | 0.15062 | 362.2613 | 496 |
| 8 | 10 | 0.14126 | 506.2069 | 496 |
| 9 | 7 | 0.14347 | 564.6152 | 496 |
| 10 | 6 | 0.22541 | 1231.4153 | 496 |

**2,253,828 parameters at `grad_abs_sum` exactly 0.0 on four steps** — the `tac_goal_tok_head`
signature exactly. ⭐ **It is LABEL SPARSITY, not a wiring defect, and the only reason that is
answerable is that `tacv6_n_supervised_*` is logged beside the reach row.** A batch whose windows
fall outside the record's ±2 s band supervises nothing; `n_scene` stays 496 throughout, so the map
was present the whole time. ⛔ Without the counts, those four zeros and a dead head are
indistinguishable — which is the entire argument for emitting `n` per term.

⚠️ **Consequence for a real arm — and the rate is NOT quotable from this.** 4 of 10 is **n = 10 on
one seed at batch 4**; it is a demonstration that unsupervised steps *occur and are frequent*, not
an estimate of how often. ⛔ Do not carry "40 %" anywhere. What IS quotable: **a tactical arm must
report its supervised-step count, not only its step count**, and whoever sizes the GPU run should
measure the rate on the real batch size first. It was invisible before the reach row and the
supervised counts landed on the same log line.

## 4. ⚠️ A latent hole I found — and that my own change made reachable

⛔ **MEASURED:** `refcv6_tactical.TacticalBehaviourDecoder.forward` does **not** raise when
`bev_in` is built and `bev_tokens is None` **while agent tokens are present** — `kv_parts` is
non-empty, so it runs **agent-only** and returns a perfectly healthy dict, while `config.json` would
stamp `sources: ["agent","bev"]` and `bev_tokens_reach_decoder: true`. That is the defect class this
programme keeps paying for, and it was unreachable only because `d_bev > 0` was unreachable.

⚠️ **It cannot be closed in the decoder.** The symmetric guard would break the legitimate unit tests
that exercise ONE source on a two-source build. The decoder cannot know whether a missing source is
an experiment or an accident; **`RefCV3Model.forward` can**, and that is where the refusal went.
`test_MUTATION_the_decoder_alone_is_BLIND_to_that_case` pins the decoder's blindness deliberately,
so the day it changes, the argument for the guard's location goes RED rather than going stale.

**A second defect, caught by a real forward and not by reading:** the perception branch produces BEV
tokens on *any* map arm, so on `--tac-decoder-d-bev 0` the agent-only decoder was handed tokens it
has no `bev_in` for and `SceneInputRefused` fired on **every** forward. The **feed gate** in
`_bev_hook` is the fix; `bev_feats` still carries the tensor, and the dict records
`bev_tokens_fed: false` so a later reader never has to infer which happened.

## 4b. ⛔ The ruling's own mitigation could not see the ruling's own gradient

R3 states the trunk is now optimised **four** ways and names two mitigations: per-head gradient
reach, and the **gradient-conflict detector**. ⛔ **MEASURED:** the detector's aux table listed
**three** terms —

```
CONFLICT_PERCEPTION_TERMS = (("bev", …), ("map", …), ("box3d", …))
```

— and **not** the tactical decoder. `_conflict_terms` looks each term up **by key** in the loss dict
and **skips an absent key silently**, so on a BEV-tactical arm every `cd_*` row would have compared
the planner gradient against an aux sum with the tactical term **missing**, and reported a healthy
row while measuring one gradient fewer than it names.

⚠️ **THE BLINDNESS PRE-DATES THE RULING.** §3's table shows the tactical loss reaching the trunk
with `grad_abs_sum` **78,146** on the **agent-only** arm — agent slots are decoded from the trunk's
own feature map — so this gradient has been arriving **unmeasured** for as long as the term has
existed. R3 is what makes it load-bearing, not what creates it.

**Fixed:** `("tac_v6", …)` added to the table and `extra["tac_v6"] = _t6_loss` exposed (the
**unweighted** tensor — `_conflict_terms` applies the arm's own weight, so a pre-weighted tensor
would square it). Two mutations pin it; a producer/consumer cross-check pins that **every** named
term has something that writes it, derived from the producer's source rather than from the
consumer's own table.

⚠️ **CONSEQUENCE, stated rather than left to be discovered:** on an arm carrying **both** a
perception weight and `--w-tac-v6`, the aux side of every `cd_*` row now includes the tactical term,
so those numbers are **not comparable** with a pre-2026-09-17 run's. No banked arm is known to be
affected — `--tac-decoder-d-bev > 0` could not run before tonight — but a reader comparing across
that date must know.

## 5. ⛔ BLOCKER ON RULING R1: **408 × 1024 does not build** — PI decision needed

`code/geometry_408.py` → `raw/geometry_408.json`. **EVIDENCE CLASS: MEASURED.**

```
TimmTrunkConfig.__post_init__: "refcv6 trunk: image 408x1024 — each axis must
divide by 32 or the stride-32 map is silently mis-sized"      408 % 32 == 8
```

⭐ **That refusal is correct and must not be relaxed.** timm rounds the spatial size **up** at each
stride-2 stage, so at 408 the trunk really emits a **26**-row stride-16 map, while
`timm_trunk.py:406` (`s16_shape = (h // 16, w // 16)`) would **declare 25**. That declared shape is
what `build_perception_branch` sizes `BEVLift` and `Box3DMemory` from — so relaxing the guard does
not buy 408, it buys a positional table one row short of the map it indexes.

**The eval-139 cache really is 408 × 1024** (MEASURED on its own shards: `image_h 408`,
`image_w 1024`, `projection_mode cylindrical`, 139 shards), so the cache and the trunk disagree.

| height | 32-divisible | VFOV | nearest visible road | builds | declared == actual |
|---|---|---|---|---|---|
| 256 | ✅ | 29.341° | 5.02 m | ✅ | ✅ |
| 384 | ✅ | 42.880° | 3.35 m | ✅ | ✅ |
| **408 (the ruling)** | ⛔ **no** | **45.296°** | **3.15 m** | ⛔ **REFUSED** | — |
| **416 (proposed)** | ✅ | **46.092°** | **3.09 m** | ✅ | ✅ |

⭐ **My formula reproduces the ruling's own published numbers exactly** — VFOV **45.296°** and
nearest road **3.15 m** at 408. Without that cross-check a disagreement here would be unreadable.
*(The mount height 1.3143 m is back-solved from the ruling's own 408 → 3.15 m pair, and it falls
inside the corpus's MEASURED 1.2131–1.6672 m.)*

⇒ **416 × 1024 is not a concession, it is strictly better on the PI's own stated criterion:** the
nearest visible road moves **closer** (3.09 m vs 3.15 m) and the vertical field is **larger**
(46.09° vs 45.30°). Its stride-16 map is **26 × 64** — precisely the map timm would have produced at
408 anyway. **384 loses ground** (3.35 m) and is the wrong direction.

⚠️ **The cost, stated because the PI should not have to rediscover it:** +2.0 % pixels and +2.0 %
cache over 408 ⇒ ≈ **394 GB** against the ruling's 386.5 GB, on a **hard HF ceiling** the ruling
already flags. **This needs a PI decision** — it is a named blocker under Rule Zero (3), not a
choice an implementation agent may make: it changes a corpus rebuild and a quota.

## 6. Bit-identity against the NEW baseline

`code/bitid.sh` + the **existing** `bitid_check.py` (invoked by path, **not copied**, so this
package cannot drift from it) → `raw/bit_identity.json`, `raw/bitid_config_diff.json`.

Tip `ee1635a` vs patched, `--smoke --steps 3 --seed 0`, perception weights at their 0.0 default,
plus `tip2` (the same trainer run twice) as the wall-clock control.

| artifact | verdict |
|---|---|
| `metrics.jsonl` | **BYTE-IDENTICAL** (`sha256 1e4b1275…`, 1559 B, 3 rows, both sides) |
| `ckpt.pt` | **BITWISE-IDENTICAL** — 193 tensors, 131,977 elements, 0 differing, no key added or removed |
| **mutation control** | **DETECTS-EVERY-ONE-BIT-FLIP — 193 / 193** |
| metrics field control | tip↔tip2 **0** fields differ; tip↔patched **0** fields differ |
| no `map*` / `box3d*` / `ga_*` key in the patched metrics | ✅ (with `has_loss_key: true` as the same-breath control) |

### ⚠️ `config.json` DIFFERS — accounted for to the leaf, not smoothed

`raw/bitid_config_diff.json`: exactly **two** top-level keys differ, and **no seam leaf changed its
value**.

* **`argv`** — one token: the `--out` directory. The driver, not the patch.
* **`seams.tac_decoder_v6`** — `bev_blocked_by` **removed**; `bev_grad_reaches_trunk`,
  `bev_detached`, `bev_unblocked_by` **added**. `bev_tokens_reach_decoder` still reads **`False`**
  on a default arm.

⛔ **This block cannot be both truthful and byte-identical.** The tip stamped
`bev_blocked_by: "<prose about a structural block>"` unconditionally, and that sentence is now
**false**. Keeping it to preserve a digest would be a run record that lies. **Reported, resolved,
argued — never smoothed.** No tensor and no metric changed.

## 6b. ⚠️ 19 tests regressed — what broke, why, and what it taught

A full-suite run on **both** trees (tip `ee1635a` and patched, same ignores) showed **70 failures at
tip and 88 patched — 18 mine, 0 fixed by me.** A **19th** appeared afterwards, from a guard I added
later in the session (below). All 19 are now fixed, and the root of the first 18 is one thing:
**adding two keyword arguments to `RefCV3Model.forward` moved contracts that DERIVE themselves from
the signature.** That is the machinery working, not failing.

| what broke | why | fix |
|---|---|---|
| `test_rl_forward_keys_cover_signature` (2) | the RL adapter's channel requirement is derived from the live signature; `perception_grid` / `perception_valid` appeared and nothing declared them. ⛔ A channel absent from `FORWARD_KEYS` is **never passed at all**, with no error. | declared a **`ChannelExclusion`** for both in the seam that owns them (`refcv6_perception_branch`), with reason, evidence and unblock condition, and listed that seam in `SEAM_MODULES`. |
| `test_rl_refcv3_used_path_guard` (8) + `test_rl_channel_guard`, `test_rl_refc_adapter_robust`, `test_rl_refcv3_integration` (3) | `bev_hook` entered the core signature and had no `ChannelRequirement`; three hard-coded channel sets and two pinned counts (17) moved. | declared `bev_hook` as a `ChannelRequirement` (**NOT asserted** — it is an optional seam, the `scene_hook` ruling unchanged), updated the sets and counts **with their reasons**, and rewrote the now-stale `bev_tokens` unblock text. |
| `test_refc_v3_agent_gt_reaches_forward` (4) | its deliberate-regression arm anchors on the **verbatim text** of the trainer's `model(...)` call, which I changed. | re-anchored. ⭐ **The harness REFUSED rather than passing** — *"the mutation anchor matched 0 times … the deliberate-regression arm is DISARMED, so this test would pass without proving anything."* Exactly the right behaviour, and the reason this was caught in minutes. |
| `test_refcv6_tactical::test_e2e_a_scene_hook_with_no_scene_REFUSES` (1) | ⛔ **my own new guard PRE-EMPTED an older one.** The model-level refusal (§4) fires earlier and more specifically than `refc.py`'s *"NEITHER agent tokens NOR BEV tokens"*, so the test's target never ran. | re-targeted to the new refusal — **and a second half added that reaches the OLD one through a direct `RefCModel` call**. ⛔ Without that, this patch would have turned a live guard into dead code and nothing would have said so. |

### ⭐ The confirming run — parity, asserted on the SETS and not on the counts

A **third** full-suite run, on the final tree, same ignores:

| | tip `ee1635a` | patched (final) |
|---|---|---|
| failed | **70** | **70** |
| errors (collection) | **4** | **4** |
| passed | 8,173 | **8,189** (+16 — this package's new tests) |
| skipped / xfailed | 48 / 2 | 48 / 2 |

⛔ **Compared as SETS, not as counts** — two different 70s would look identical in a count:

```
ONLY in patched (regressions):  NONE
ONLY in tip (fixed by me):      NONE
FAILURE SETS IDENTICAL: True    ERROR SETS IDENTICAL: True
```

⇒ **zero regressions and zero accidental fixes**, and the +16 is exactly the tests this package adds.

⭐ **What it teaches, and it is the reusable part:** a forward signature is a **contract surface**,
not a parameter list. Two kwargs moved an RL admissibility contract, a mutation anchor and five
hard-coded sets. ⛔ The `ChannelExclusion` was the load-bearing judgement: these channels are rig
**calibration**, not labels — nothing in them comes from the future — so the exclusion is
**TEMPORARY** with a written route back (plumb a `LiftGeometryBank` into the rollout), not the
permanent kind `gp_point` carries. Recording that distinction is the whole point of the surface.

## 7. Guards proven by MUTATION — 10 / 10

`code/mutation_proof.py` → `raw/mutation_proof.json`. Each defect is reintroduced one at a time and
the named test must go **RED**; a mutation that leaves it green is reported `BLIND`.

| mutation | detected |
|---|---|
| feed gate removed | ✅ |
| declared-BEV-without-branch guard removed | ✅ |
| **`bev_hook` never passed to the core (the tip's own state)** | ✅ |
| detach flag ignored | ✅ |
| flat-arm refusal removed | ✅ |
| two-suppliers guard removed | ✅ |
| trainer `d_bev` precondition removed | ✅ |
| **stamp hardcoded `False` again (the tip's literal)** | ✅ |
| **conflict detector loses the tactical term (the tip's state)** | ✅ |
| tactical loss not exposed to the detector | ✅ |

`all_named_tests_pass_unmutated: true` (the control — without it a RED table cannot tell a guard
from a brick) · `blind: []` · `not_applied: []` · `worktree_restored: true` (verified by **reading
the bytes back**, not by an exit code).

## 8. What I could NOT do — with the blocking seam named

| not done | blocked on |
|---|---|
| **Any capability claim, any metric family** | ⛔ **No GPU** — held by an RL arm. Everything here is CPU-only. A trained arm is required and none was run. |
| **Whether the map actually IMPROVES behaviour prediction** | Same. §3 measures **reachability**; selectivity needs a trained arm and a per-class result. |
| **Run at the ruling's 408 × 1024** | ⛔ **The trunk refuses it** (§5). Needs a **PI decision** between 416 (better optics, +7.5 GB HF) and a `ceil`-based shape derivation. |
| **The gradient-conflict detector on a live four-way arm** | The tactical term is now IN its aux table (§4b) and unit-pinned, but exercising it needs a training run, i.e. **GPU**. ⚠️ Its `cos` channel is magnitude-blind by construction — `ratio` and `proj` must be read too (PREREG §8). |
| ~~End-to-end with REAL SAM3 map GT~~ | ✅ **DONE** — §3b. 10 CPU steps on the real corpus, real maps, real join, real extrinsics. |
| **A tactical result on the map-derived tokens** | Needs a **trained** arm. §3b shows unsupervised steps are frequent (4 of 10 at batch 4, n = 10 — a demonstration, not a rate), so a per-class number needs its supervised-step count and far more than 10 steps. GPU-gated. |

⚠️ **One integration coupling, named rather than worked around:** `refc.py` refuses `agents.enable`
without `decoder.cross_agent`, so giving the tactical layer agent slots forces agent
cross-attention on the **operative** decoder too. A refcv6 tactical arm is therefore only
one-variable against a baseline that **already has the agent seam on**. Identical on every arm here,
so it cannot separate them — but it constrains how a future A/B is read.

⚠️ **Two test modules fail to COLLECT on this branch** (`test_metric_decode_refusal.py`,
`test_refa_v1_dk_hook.py` — missing `UntrainedMetricReadout` / `DistanceKeepingSpec`). **They fail
identically at tip `ee1635a`** — verified on a clean tip worktree — so they are **pre-existing and
not mine**. Named here because a later reader must not attribute them to this patch.

## 9. Escalation — needs a decision, not a merge

1. ⛔ **RULING R1 (408 × 1024) IS UNBUILDABLE.** Choose **416 × 1024** (+7.5 GB on a hard HF
   ceiling), or authorise a `ceil`-based shape derivation in `timm_trunk.py` + a relaxed divisibility
   guard. **Nothing should be rebuilt at 408 until this is decided.** §5.
2. ⚠️ **`timm_trunk.py:406` uses floor division for `s16_shape`** and would silently understate the
   row count by 1 on any height that 16 does not divide. The 32-divisibility guard currently hides
   it. Worth fixing on its own merits even if 416 is chosen.
3. ⚠️ **`refcv6_tactical.forward` is blind to "declared BEV source, no tokens"** (§4). Closed at the
   model level here. If that decoder is ever driven from another call site, the hole travels.
4. ⚠️ **`cd_*` rows change meaning on any arm with both a perception weight and `--w-tac-v6`**
   (§4b). Intended and required by R3, but it is a semantics change to another package's instrument
   and its owner should know.

---

## Deliverable manifest

| artifact | where it lives | only one place? |
|---|---|---|
| `RESULT.md` (this file) | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-17-refcv6-bev-tactical/RESULT.md` | no — staged |
| `code/grad_reach_bev.py` | same package | no — staged |
| `code/geometry_408.py` | same package | no — staged |
| `code/mutation_proof.py` | same package | no — staged |
| `code/config_diff.py` | same package | no — staged |
| `code/bitid.sh` | same package | no — staged |
| `code/live.sh` | same package | no — staged |
| `raw/live_run/{config.json, metrics.jsonl, launch_excerpt.txt}` | same package | no — staged |
| `raw/grad_reach_bev.json` | same package | no — staged |
| `raw/geometry_408.json` | same package | no — staged |
| `raw/mutation_proof.json` | same package | no — staged |
| `raw/bit_identity.json` | same package | no — staged |
| `raw/bitid_config_diff.json` | same package | no — staged |
| `stack/tanitad/refs/refc.py` (`bev_hook` seam, `out["perception"]`) | `repo:` | no — staged |
| `stack/tanitad/refs/refc_v3.py` (`_bev_hook`, geometry args, guards) | `repo:` | no — staged |
| `stack/tanitad/models/refcv6_perception_branch.py` (`bev_tokens`, widened reach report) | `repo:` | no — staged |
| `stack/scripts/refc_v3_train.py` (precondition, detach flag, stamp, reach row) | `repo:` | no — staged |
| `stack/tests/test_refcv6_bev_tactical_wiring.py` (new) | `repo:` | no — staged |
| `stack/tests/test_refcv6_tactical_training.py` (two tests rewritten) | `repo:` | no — staged |
| `stack/tests/test_refcv6_no_session_paths.py` (OWNED extended) | `repo:` | no — staged |
| `stack/tanitad/channel_admissibility.py` (SEAM_MODULES + this seam) | `repo:` | no — staged |
| `stack/tanitad/rl/refcv3_adapter.py` (`bev_hook` requirement; `bev_tokens` unblock rewritten) | `repo:` | no — staged |
| `stack/tests/test_rl_refcv3_used_path_guard.py` (counts + three channel sets) | `repo:` | no — staged |
| `stack/tests/test_refc_v3_agent_gt_reaches_forward.py` (regression arm re-anchored) | `repo:` | no — staged |
| `stack/tests/test_refcv6_tactical.py` (refusal re-targeted + old one proven still live) | `repo:` | no — staged |

⛔ **Nothing is committed and nothing is pushed**, per the operating standard. Worktree:
`tanitad-wt-bevtac` (detached at `ee1635a` + these changes), with `tanitad-wt-bevtac-tip` as the
clean tip used for the bit-identity and pre-existing-failure controls.
