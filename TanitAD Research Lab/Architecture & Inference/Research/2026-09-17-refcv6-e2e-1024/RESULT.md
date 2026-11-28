# refcv6 SPEC-v2 END TO END on the 139 B1 eval clips at 256 x 1024 — what runs, and what is not wired

**PI instruction (`SPEC_REFCV6_V2.md` §10.6):** *"You validate the pipeline wit the 139 eval clips."*

`stack/tests/test_refcv6_perception_realdata.py` validates the **data** and the **geometry** (16/16
at 256 x 1024, `ec56eb9`). It does **not** run a model: its trunk stand-in is
`lift_orientation.lift_image_features`, which is `F.avg_pool2d(img, stride)` on raw pixels
(`stack/tanitad/data/lift_orientation.py:104`). **This package runs the model.**

**Worktree** `C:/Users/Admin/tanitad-wt-e2e`, detached at **`faecb29`** · **CPU only**
(`CUDA_VISIBLE_DEVICES=""`; the dev-box GPU was held at ~100 % by another package and CLAUDE.md
forbids adding load). The branch tip moved to `e50116a` while this ran; §8 re-verifies both
structural findings against that newer tree.

Evidence classes: **MEASURED** = produced here, raw JSON named · **PUBLISHED-CODE** = read out of
the repo, `file:line` given · **ESTIMATED** = arithmetic with its assumption stated. Nothing below
is INHERITED. Clip ids appear only as sha12.

---

## 0. THE HEADLINE

| SPEC-v2 stage | state | where |
|---|---|---|
| timm ResNet-101 trunk, ImageNet | ✅ runs; weights **verified by mutation** | §2 |
| stride-16 `16x64`, stride-32 `8x32`, read from `feature_info` | ✅ correct at 256 x 1024 | §1 |
| diffusion planner F1–F6 + F9, selection | ✅ runs on **all 139 clips**, no non-finite value | §3 |
| gradient into the trunk from the planner | ✅ **non-zero on 139/139** | §3 |
| BEV lift → map head + 3-D box head on the real GT | ✅ **runs, finite, gradient to the trunk** | §5 |
| **…but the perception branch is not in the model** | ⛔ **NOT WIRED** — no trainer builds it | §4 |
| **SPEC §4 tactical behaviour decoder** | ⛔ **no CLI flag; 2,262,020 params at grad EXACTLY 0** | §6 |
| **SPEC §5 four-value max-speed input** | ⛔ **no CLI flag AND no data channel — unreachable** | §6 |
| **SPEC §10.3 ego-history input** | ⛔ **DEAD ZERO PRODUCT — cannot ever learn** | §7 |
| bit-identity with the refcv6 seams off | ✅ numerically **exact**; one new output key | §9 |

⛔ **The single most consequential finding: the refcv6 PERCEPTION BRANCH IS NOT WIRED INTO ANY
TRAINING PATH.** `RefCV3Model` emits `out["fmap_s16"]` — the seam — and **nothing consumes it**.
The stride-16 map is computed on every step of a real run and thrown away.

⭐ **The second: three separate refcv6 seams are built, stamped into the run record, and receive
EXACTLY ZERO GRADIENT** — the tactical decoder (2.26 M params), the ego-history channel
(15,328 + 12,288), and, by a different and *benign* mechanism, two zero-init grafts that can still
open. §7 separates them **by mutation**, because a zero at step 0 is not automatically a defect.

---

## 1. The assembly — the trainer's own path, at the corpus's own geometry

The model is built by `refc_v3_train.build_parser` → `_pin_trainer_cfg` → `refc_v3.RefCV3Model`,
and the batch by `build_v2_providers` → `V3Dataset` → `compute_losses_v3`. Nothing is
re-implemented. `--image-hw` is fed from `v2_dataset.stored_frame_of(payload)`, so no geometry
literal appears in this package (`code/e2e_1024.py::frame_of_cache`).

MEASURED (`raw/assembly.json`):

| | |
|---|---|
| payload frame | **256 x 1024**, `f_ref` **488.92398517830253**, projection **cylindrical** |
| encoder built at | `image_hw` **(256, 1024)**, `in_channels` **9**, `window` **8** |
| stride-16 (perception) | **1024 ch, 16 x 64** — from timm `feature_info`, never a literal |
| stride-32 (planner) | **2048 ch, 8 x 32** = 256 image tokens |
| params | **86,181,732** total; trunk **58,231,872** (backbone 42,500,160) |
| v7.2 eval label join | **139/139 episodes = 100.0 %** |
| seam stamp | `trunk timm/resnet101.a1_in1k/shared/concat1x1` · `ego_history {enable:true, steps:8, out_dim:32, kind:gru, zero_init_out:true}` · `refcv6 {f1..f6, f9 = true, any_on: true}` · `no_strategic true` |

Launch line actually used (`raw/assembly.json:argv`):

```
--arm hier --size small --trunk timm --trunk-name resnet101.a1_in1k --trunk-mode shared
--trunk-fuse concat1x1 --ego-history --no-strategic --image-hw 256 1024 --sampler ddim
--f1-random-t --f2-dd-step --f3-per-layer --f4-adaln --f5-focal --f6-w-u0-zero --f9-assert-vocab
--v2-cache <eval139-256x1024cyl> --v7-labels <s2_labels_v7.2_eval> --anchors <hf-refcv5v2>
--anchor-v0-conditioned --n-anchors 117
```

## 2. The ImageNet weights are real — proven by mutation, not by a flag

`raw/pretrained_guard.json`. `timm_trunk._assert_pretrained_loaded` is exercised three ways:

| arm | resnet101 stem `\|w\|.sum()` | guard |
|---|---|---|
| CONTROL, as shipped | **1402.778076** (pinned 1402.778076, band [1374.72, 1430.83]) | **PASS** |
| MUTATION A — He-reinit the stem in place | 187.387 | **RED (correct)** |
| MUTATION B — a genuine `pretrained=False` build | 190.225 | **RED (correct)** |

resnet34 behaves identically (1279.854 / 187.480 / 191.127). `provenance.verified_pretrained` is
`true` on both.

⚠️ **A small real defect found on the way:** the guard's own message says *"a He init reads ~875"*.
MEASURED here, a He re-init of this stem reads **187.4** and a `pretrained=False` build **190.2** —
the quoted reference is **~4.7x high**. The guard still fires (187 is far outside the band), but the
number in the message is not what a He init actually reads at this shape. Worth correcting in
`stack/tanitad/models/timm_trunk.py`.

## 3. The forward pass — all 139 clips, the trainer's own loss

MEASURED (`raw/forward.json`, `logs/forward_full139.jsonl`; one middle window per clip,
batch 1, CPU).

* **139/139 clips completed. `n_nonfinite_values` = 0.** No NaN, no inf, in any term, on any clip.
* mean **39.36 s/clip** (min 23.34, max 72.40), total **1.52 h**, peak RSS **19.44 GB**.
* batch shapes: `frames [1, 8, 9, 256, 1024]`, `future_frames [1, 20, 9, 256, 1024]`,
  `pose_hist [1, 8, 4]`, `future_poses_ext [1, 60, 4]`.

| loss term | min | median | max | exactly 0 |
|---|---|---|---|---|
| `loss` (the total) | 11.9373 | 46.8351 | 125.4998 | 0 |
| `traj` | 0.0000 | 1.2517 | 11.0371 | **1** |
| `cls` | 0.6277 | 6.4976 | 25.8183 | 0 |
| `sel_v3` | 2.0997 | 15.1006 | 27.7229 | 0 |
| `goal_tac` | 0.1180 | 47.7082 | 177.6850 | 0 |
| `lat` / `lon` | 1.0487 / 1.0314 | 1.1069 / 1.1467 | 1.1334 / 1.1719 | 0 |
| `lat_tac` / `lon_tac` | 1.9023 / 1.8883 | 2.1782 / 2.0942 | 2.3156 / 2.2617 | 0 |
| `law` | 0.0442 | 0.6560 | 4.8742 | 0 |
| `route` | 0.0000 | 1.1128 | 1.1878 | **27** |

**The two exact zeros are explained, not waved through:**

* `route` = 0 on **27/139** clips — the term is masked by `nav_valid`, and this corpus's
  `nav_valid_frac` is 0.21–0.25 (`refc.py:602`). Expected.
* `traj` = 0 on **1** clip (episode index 83). MEASURED (`raw/traj_zero_clip.json`): that clip's
  speed column has **median 0.0 m/s** over its 199 poses (35.4 m of path in total). At `v0 = 0`
  every v0-conditioned anchor rolls to the origin, the fan collapses to a point, the waypoint
  target is also the origin, and the term is exactly 0. This is the already-known
  zero-speed degeneracy of the v0-conditioned vocabulary, reproduced here on real data.

**Absent terms** (`agent`, `bev_aux`, `goal_point`, `goal_str`, `gstr`, `tac_goal`, `u0`): each has
weight 0 or a gate off on this arm; `u0` is absent *because* F6 (`--f6-w-u0-zero`) is on, which is
the SPEC's own instruction.

**Gradient reaching each head, per clip (`grad_abs_sum`, read BEFORE clipping):**

| module | params | rows at exactly 0 | min | median | max |
|---|---|---|---|---|---|
| `core.encoder` (**the trunk**) | 58,231,872 | **0/139** | 7,484 | 12,420 | 17,620 |
| `core.decoder` (the planner) | 12,132,515 | 0/139 | 2.35e4 | 9.45e4 | 4.81e5 |
| `phi_tac` | 2,101,504 | 0/139 | 1,669 | 2,975 | 3,688 |
| `core.tactical_trunk` | 786,816 | 0/139 | 17.6 | 245.1 | 648.1 |
| `core.law_head` | 8,425,472 | 0/139 | 74.7 | 1,183 | 8,560 |
| `tac_goal_head` | 6,156 | 0/139 | 55.3 | 149.2 | 199.9 |
| `core.route_head` | 6,147 | 27/139 | 0 | 17.11 | 22.70 |
| **`core.strategic`** | 4,066,560 | **139/139** | 0 | 0 | 0 |
| **`core.ego_hist`** | 15,328 | **139/139** | 0 | 0 | 0 |
| **`tac_latent_proj`** | 262,656 | **139/139** | 0 | 0 | 0 |
| **`scorer`** | 1,144 | **139/139** | 0 | 0 | 0 |

`core.strategic` reading exactly 0 on every clip is the **control that must read a known value**:
`--no-strategic` is on, so a non-zero there would have meant the bypass does not bypass. The other
three zeros are diagnosed in §7.

⚠️ **The route head is BUILT and TRAINED under `--no-strategic`.** SPEC §1 says *"the heads are not
built"*; `refc.py:773-777` says the opposite on purpose — the head *"keeps running … it is a pure
READOUT with no path into the emitted plan"*. Both cannot be right in a reader's head: the code's
behaviour is the one that happens, and **the SPEC text is the stale half.** Worth one line of
correction in `SPEC_REFCV6_V2.md`.

## 4. ⛔ THE PERCEPTION BRANCH IS NOT IN THE MODEL — three independent probes

1. **Runtime**, on the built model (`raw/assembly.json`): every `named_modules()` type was
   collected; `BEVLift` / `BEVMapBranch` / `MapHead` / `Box3DMemory` / `Box3DSlotDecoder` appear
   **zero** times. `in_built_model: false` for both `S6_map_head` and `S6_box3d_head`.
2. **The loss**, at source: `compute_losses_v3` returns exactly
   `{loss, traj, cls, law, route, lat, lon, lat_tac, lon_tac, anchor_acc, slot_valid_frac, **extra}`
   (`refc_v3_train.py:3065-3069`). **No map term. No box term.**
3. **The repo**, whole-tree: `BEVMapBranch|Box3DSlotDecoder|box3d_set_loss|map_soft_ce` is
   referenced only by its own four modules, three test files, one research script and this
   harness — **by no trainer, no model and no eval harness** (`raw/tip_recheck.json`; the one hit
   inside `refc.py` is a *docstring* at `:2232`, not a construction).

⇒ A refcv6 training run today computes the stride-16 map, emits it as `out["fmap_s16"]`, and
discards it. **The SPEC §6 map head and 3-D box head exist, are tested, and are supervised by
nothing.**

## 5. ✅ …but the seam DOES carry them — measured on real SAM3 maps and real cuboids

The heads were assembled **by this harness** (5,082,816 params: lift 393,408 · map branch 758,505 ·
box memory 287,744 · box decoder 3,643,159) on the model's own `fmap_s16`, and scored against the
real GT. MEASURED (`raw/perception.json`, 13 clips evenly spaced over the 139).

⭐ **The stride-16 map is the model's own, proven per run**: the harness recomputes it as
`forward_features(frames.reshape(b*w, …))[0][:, -1]` — exactly `refc.py:3867-3869` — and the first
scored clip of every run asserts it against a full `model(...)` forward. **`bit_identical: true`,
`max_abs_delta 0.0`.**
⛔ **A cheaper version was tried and the guard caught it.** `forward_features(frames[:, -1])` — one
window position, 1/8 of the activations — is *not* equivalent, because the trunk runs in TRAIN mode
and its BatchNorm normalises over the batch: **max \|delta\| 72.7779**. An 8x saving that looked
obviously correct was refused by its own control.

| | measured |
|---|---|
| `fmap_s16` | **[1, 1024, 16, 64]**, `requires_grad` true |
| BEV after the lift | **[1, 96, 120, 64]** |
| map logits | **[1, 9, 120, 64]** — ⭐ the SAM3 label grid stays **120 x 64 @ 0.5 m** while the image is 1024 wide |
| map soft-CE | min **2.0047** · median **2.2546** · max **2.3424** — **0 exact zeros, 0 non-finite** |
| seen fraction | min 0.607 · median 1.000 |
| box3d set loss | min **47.996** · median **88.916** · max **130.438**, all finite |
| z/h targets reaching the loss | `n["z"] = n["h"]` > 0 on **13/13**; min 2, median 22, max 153, **569 total** |
| ⛔ CONTROL, z/h withheld | `n_z = 0` and `loss_z` **EXACTLY 0.0** on **13/13**; the z/h terms move the total by at least **3.169** ⇒ they are CONSUMED, not carried |
| **map loss → trunk** | grad_abs_sum **10,902 – 15,223**, **0/13** rows at zero |
| **box loss → trunk** | grad_abs_sum **198,686 – 2,350,482**, **0/13** rows at zero |
| cost | median **107.2 s/clip** (min 53.9, max 270.3 — two backward passes through the trunk), peak RSS **19.66 GB** |

⚠️ **13 of 45 requested, not 139 — stated plainly.** The run was stopped by explicit PID when the
box became CPU- and RAM-contended by other packages (a second `refc_v3_train` plus the GPU
package): s/clip went **53.9 → 270.3** and RAM fell to 0.65 GB free. Rows are banked incrementally,
the subset is evenly spaced over the corpus, and **every clip attempted was scored (0 skipped)**.

### 5.1 ⚠️ TWO INDEX SPACES MEET IN THE BOX HEAD, AND NOTHING ENFORCES WHICH IS WHICH

`AgentJoin3D` indexes `int(rec["frame"])` (`agent_cuboid_gt.py:353`), which
`build_b1_agent_join.py:1` declares to be the **RAW v2ep frame**. The trainer's own 2-D agent seam
passes the **EPISODE index** `t + w - 1` (`refc_v3_train.py:2172`) to `JoinFileReader`, which keys
on `frame_idx` (`train_p8_occupancy.py:246`). The same builder emits **both**, under different
names; they differ by `n_stack - 1 = 2` frames ≈ **0.2 s**.

MEASURED, independently of either reader (`raw/join_frame_key.json`, 12 clips / 67 window probes):
the join's `cz` against the **parquet's own `center_z`**, which is reached by TIME, not by any frame
index —

| offset from RAW | −3 | −2 | −1 | **0** | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| mean \|Δcz\| (m) | 0.2534 | 0.2043 | 0.1303 | **0.0209** | 0.0918 | 0.1751 | 0.2347 |

Clean minimum at RAW+0, **4.40x separation**. ⇒ the 3-D join's key is the RAW index. This is not a
defect today — the 3-D path has no consumer in the trainer — but it is a trap for whoever wires
§6, and it is the reason this harness passes `tgt.raw_frame` there and `t + w - 1` to the map store.

## 6. ⛔ SPEC §4 AND §5 CANNOT BE REACHED FROM THE TRAINER AT ALL

`raw/assembly.json:spec_seam_cli_reachability` enumerates the parser's own option strings.

| SPEC seam | config field exists | argparse dest exists | CLI flag |
|---|---|---|---|
| §4 tactical behaviour decoder | `RefCV3Config.tac_decoder_v6` ✅ | ❌ | **none** |
| §5 four-value max-speed one-hot | `RefCV3Config.max_speed_onehot_v6` ✅ | ❌ | **none** |
| §6 map head / 3-D box head | — | ❌ | **none** |

⇒ **No argv can turn any of them on.** To exercise §4 this harness set the field on the config
directly, and says so in its own output (`[e2e] CONFIG OVERRIDES (the CLI has no flag for these)`).

**§5 is worse than "no flag": it has no data channel either.** Forcing
`max_speed_onehot_v6 = True` builds the seam, and then *every* batch is refused —
`"this build is --max-speed-input but no v_max_ms reached the forward while a nav token did"`
(`refc_v3.py:1833`; full traceback in `raw/tactical_maxspeed_refusal.log`). The only supplier of
`v_max_ms` is `V3Dataset.enable_max_speed`, which the trainer calls **only** under
`--max-speed-input` — and that flag builds E16's *continuous 8-step* seam, which
`RefCV3Model.__init__` refuses to run alongside the refcv6 one-hot (*"BOTH max-speed channels are
on … Pick one."*). **SPEC §5 as written is unreachable end to end.**

### 6.1 ⛔ THE §4 DECODER RUNS — AND RECEIVES EXACTLY ZERO GRADIENT

With `tac_decoder_v6` forced on, `--agents head`, `--w-agent 0.05` and the banked B1 EVAL 2-D join
(139/139 episodes, **95.3 %** of windows labelled), 5 clips run clean — every agent term fires
(`agent_presence` 0.0437, `agent_cls` 3.60, `agent_centre` 28.96, `agent_size` 4.09, `agent_yaw`
0.151) and the trunk gradient rises to **13,540 – 153,700**. MEASURED (`raw/tactical.json`):

| module | params | rows at exactly 0 |
|---|---|---|
| **`tac_decoder_v6`** | **2,262,020** | **5/5** |
| **`tac_behaviour_gate_v6`** | **2,574** | **5/5** |
| `core.agent_head` | 3,781,909 | 0/5 |

**Two structural reasons, both read from source:**

1. `refcv6_tactical.planner_feeds` returns every feed **`.detach()`ed**, by design
   (`refcv6_tactical.py:670-675`: *"every bit this layer learns comes from its OWN losses"*).
2. **Its own losses do not exist.** The decoder's outputs are written into the cache as
   `tacv6_goal_logits` / `tacv6_goal_conf` / `tacv6_lat_logits` / `tacv6_lon_logits`
   (`refc_v3.py:1627-1632`) and reach `out` via `out.update(cache)` (`:1923`). A whole-tree probe
   finds **no reader outside `refc_v3.py` and the tests**. SPEC §4's *"BCE 0.05 on goal tokens, CE
   0.025 per action head"* is **not implemented in `compute_losses_v3`**.

And the third path is shut too: the behaviour term reaches the ranked score only
`if self.graft_behaviour_sel` (`refc.py:3135`), which defaults **False** and is set by **no flag and
no line of `_pin_trainer_cfg`**.

⇒ **2.26 M parameters, stamped live in `config.json`, with no gradient path whatsoever.** This is
`tac_goal_tok_head` (11,286 params, grad exactly 0 for 40,284 steps) again at **200x the size**.

## 7. ⛔ THE EGO-HISTORY INPUT IS A DEAD ZERO PRODUCT — proven by mutation

A zero gradient at step 0 is **not** automatically a defect: a zero-init graft reads zero and then
opens. So the three zeros of §3 were separated by breaking **one** factor at a time and reading the
other back (`code/zero_grad_forensics.py`, `raw/zero_grad_forensics.json`; every arm is a real
forward+backward on a real clip at 256 x 1024).

| pair | grad A as shipped | grad B as shipped | A when B broken | B when A broken | verdict |
|---|---|---|---|---|---|
| `core.ego_hist.out.weight` × `core.decoder.ego_to_cond.weight` | **0.0** | **0.0** | **5.2654** | **11.1364** | ⛔ **DEAD_ZERO_PRODUCT** |
| `tac_latent_proj.weight` × `tgt_film.to_scale_shift.weight` | 0.0 | **47.699** | 116.93 | 102.69 | ✅ GATED_RECOVERABLE |
| `scorer` × `goal_gate` | 0.0 | **3.808** | 0.149 | 3.743 | ✅ GATED_RECOVERABLE |

**The mechanism.** `EgoHistoryConfig.zero_init_out` defaults **True**, so `ego_hist.out.weight` and
`.bias` are exactly zero (`ego_history.py:158-160`); and `ego_to_cond.weight`/`.bias` are **also**
zero-init (`refc.py:1955-1957`). Then

* `d(loss)/d(ego_to_cond.weight) ∝ ego_vec`, and `ego_vec ≡ 0` ⇒ **0**;
* `d(loss)/d(ego_hist.out.*) ∝ ego_to_cond.weight`, and that is **0** ⇒ **0**.

Each factor's gradient is proportional to the other and **both are exactly zero**, so neither can
ever leave zero. The only thing that learns on this path is `ego_to_cond.bias` (grad **1.829**) — an
unconditional constant offset that **carries no ego information at all**.

⚠️ This is parameter-level, and that is load-bearing: read at MODULE level, `core.decoder.ego_to_cond`
reports grad **1.829** and looks alive. The bias hides the dead weight.

⇒ **27,616 parameters of a channel the PI asked for on 2026-09-16 are structurally incapable of
learning, while `config.json` stamps `ego_history: {enable: true, …}`.** The fix is one line:
either `zero_init_out=False` on the encoder, or drop the zero-init on `ego_to_cond.weight`. **One of
the two must be non-zero** — the removable-graft discipline needs exactly one gate, not two in
series. *(The contrast rows are what make this credible: two other zero-init grafts in the same
model, measured in the same breath, are healthy.)*

## 8. The findings re-checked at the CURRENT branch tip

An absence claim goes stale the moment someone lands code. `raw/tip_recheck.json` re-reads the blobs
from git at `faecb29` **and** at `e50116a` (the tip at the time of writing; it had been `bf8dad9`
mid-run), with a same-breath positive control so an unreadable blob cannot read as an absence:

| | `faecb29` | `e50116a` |
|---|---|---|
| perception-head constructions in trainer + both model files | **0** (1 docstring hit) | **0** (1 docstring hit) |
| trainer mentions of `tac_decoder_v6` | 0 | 0 |
| `--tac-decoder-*` flag | absent | absent |
| `--max-speed-onehot-*` flag | absent | absent |
| any map/box/sam3/cuboid flag | absent | absent |
| CLI flags | 138 | 141 (`--conflict-detector`, `--conflict-every`, `--conflict-mode`) |
| unreadable files | none | none |

⇒ **Both findings hold at the current tip, not only at the one this package ran on.**

## 9. Bit-identity with the refcv6 seams off

`raw/identity.json`. The pre-refcv6 `tanitad` package (**183 .py files**) is materialised **from
git** at **`4d74774`** — the last commit before `refcv6` entered `refc.py` — imported under its own
`sys.path`, and compared against the live one at 256 x 1024, same seed, same input.

| | result |
|---|---|
| state_dict keys | **identical** (0 added, 0 removed) |
| parameters | **63,158,525 both sides**, **0 weight mismatches** |
| every shared output key | **max \|delta\| = 0.0, exactly** |
| output keys | the new model emits **one extra: `fmap_s16`** |

⇒ **Numerically the forward is exact.** What changed is the output **contract**: one new key, which
is precisely the perception seam §4 shows has no consumer. Reporting this as a bare "bit-identical:
false" would have been misleading in the other direction, so both halves are stated.

## 10. Cost, and what it projects to

MEASURED by counting multiply-accumulates from the **shapes a real forward produces** (forward hooks
on every `Conv2d`/`Linear`; no published FLOP table, no pixel-ratio scaling) — `raw/cost_model.json`.

| | resnet101 | resnet34 |
|---|---|---|
| MACs, one window position | 128,685,441,024 | 57,818,480,640 |
| trunk GFLOP per sample, fwd | 2,058.97 | 925.10 |
| trunk GFLOP per sample, **fwd + bwd** | **6,176.90** | 2,775.29 |
| trunk passes per sample (`W x K`) | 24 | 24 |

MEASURED wall clock: **39.36 s/clip** mean over 139 (**1.52 h** total), giving an effective
**157 GFLOP/s** on the CPU for the trunk share alone. Peak RSS **19.44 GB at batch 1**.

ESTIMATED projection (assumption stated, not hidden): an A40 sustaining 40–60 TFLOP/s bf16 on convs
of this shape → **0.103 – 0.154 s/clip**, **14.3 – 21.5 s for all 139**; an RTX 4060 at 5–8 TFLOP/s
→ **0.77 – 1.24 s/clip**, **107 – 172 s for 139**. ⚠️ **Compute only.** A real pod step is bounded by
the PNG decode and the v2 LRU as often as by the GPU — price what the *consumer* reads.

⚠️ **`--ego-history` does NOT reduce the trunk cost today.** SPEC §10.3 argues K shared passes are
*"cheaper than today's model, which encodes all 8 window positions with gradient"* — but
`refc.py:3866` still encodes all `b*w` positions on the hierarchy path, and `in_channels` is still 9
(a 3-frame stack), so the trunk runs **W x K = 24** passes per sample. The saving §10.3 describes is
not implemented.

## 11. What did NOT work, and what was not tested

| | |
|---|---|
| ⛔ perception heads in a trainer | **do not exist** — §4 |
| ⛔ §4 tactical decoder gradient | **exactly 0 on 5/5** — §6.1 |
| ⛔ §5 max-speed one-hot | **unreachable**: no flag, no data channel, and mutually exclusive with the only channel that has one — §6 |
| ⛔ ego-history learning | **structurally impossible** — §7 |
| ⚠️ perception scored on 13 clips, not 139 | box contention; per-clip cost and the subset rule are stated — §5 |
| ⚠️ map/box losses never entered `compute_losses_v3` | they cannot: there is no term. The gradient measurements are per-head backward passes from the harness. |
| ⚠️ no GPU measurement | the dev-box GPU was held at ~100 % by another package; §10's GPU numbers are ESTIMATED and say so |
| ⚠️ the 1 ms frame-alignment assertion | `assert_frame_alignment` could not run: the `*.v2ep.pt` payloads carry **no per-frame timestamps**, so map targets are aligned by INDEX (`raw_frame_index`) and cross-checked only through the join-vs-parquet z agreement of §5.1 |
| ⚠️ `--agents` was **off** on the 139-clip forward | the SPEC's own launch line does not include it; the tactical arm turned it on and reports separately |
| ✅ the CPU-only cuDNN/GRU trap | did **not** bite: this harness never calls `.to(device)`, so no cudnn path is touched. Reported because it was expected. |

## 12. What should happen next (for the Master Mind, not decided here)

1. **Wire the perception branch or stop stamping it.** `fmap_s16` → `BEVLift` → `BEVMapBranch` +
   `Box3DMemory`/`Box3DSlotDecoder`, two terms in `compute_losses_v3`, two weights on the CLI. §5
   shows every piece works on real data at 256 x 1024; what is missing is ~40 lines of wiring.
2. **Fix the ego-history zero product** (one line) — otherwise the next run trains 27,616 dead
   parameters and reports them as a live channel.
3. **Implement SPEC §4's own losses**, or the decoder is 2.26 M dead parameters; and decide whether
   `graft_behaviour_sel` gets a flag.
4. **Give §5 a data channel**, or amend the SPEC to say the max-speed input is E16's continuous one.
5. **Add `--tac-decoder-v6`** (and whatever §6 needs) to `build_parser`, so a launch can express the
   SPEC.
6. Correct the two documentation defects found in passing: the SPEC's *"the heads are not built"*
   for the route head (§3), and the *"a He init reads ~875"* reference in `timm_trunk.py` (§2).

---

## Deliverable manifest

| artifact | where |
|---|---|
| `RESULT.md` (this file) | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-17-refcv6-e2e-1024/RESULT.md` |
| `code/e2e_1024.py` — assembly / forward / perception / tactical / identity arms | same, `code/` |
| `code/zero_grad_forensics.py` — the DEAD-vs-GATED mutation panel | same |
| `code/pretrained_guard.py` — ImageNet weights, 3 arms | same |
| `code/join_frame_key.py` — the RAW-vs-EPISODE index discriminator | same |
| `code/cost_model.py` — MAC counting + projection | same |
| `code/tip_recheck.py` — the findings re-asserted at the newer tip | same |
| `code/summarize.py`, `code/run_all.sh` | same |
| `raw/*.json` — every number in this file | same, `raw/` |
| `logs/*.jsonl`, `logs/*.log` — per-clip rows, banked incrementally | same, `logs/` |

Every path above exists in **two** places and is staged in both: the worktree
`C:/Users/Admin/tanitad-wt-e2e` (detached at `faecb29`, index clean: 32 entries, no foreign paths,
every staged blob equal to its worktree file) and the branch working tree
`C:/Users/Admin/tanitad-push`. Every blob resolves from the shared object store. Nothing exists in
only one place.

⛔ **A hazard found while staging, unrelated to refcv6 but urgent.**
`C:/Users/Admin/tanitad-push` **tracks 11,862 files and has 41 on disk** — its working tree is
effectively empty against its own `HEAD` (`e50116a`), and its shared index therefore carries
**1,297 staged deletions**. MEASURED: of the first 200, **0 are present on disk**, so these are
*not* the phantom deletions CLAUDE.md describes — they read as real to git. **A pathspec-free
`git commit` from that tree would land them.** `stack/scripts/mm_commit.py` is immune (private
scratch index seeded from HEAD) and will pick this package up from the worktree files, which are
there. This is reported, not repaired: 1,297 paths in a shared index under live contention is not
a subagent's call.
