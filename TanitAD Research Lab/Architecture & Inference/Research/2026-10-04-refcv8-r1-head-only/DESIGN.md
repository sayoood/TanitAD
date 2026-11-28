# DESIGN — refcv8 R1: retrain the planner heads of refcv7-r101-s0 @ 50,400 on a FROZEN trunk

*R1 agent, 2026-10-04. Plan: `Project Steering/PLAN_REFCV8.md` §0 (R8-3, R8-4), §6 R1. Code: `code/` in this package.*

**Stamp.** Model refcv7-r101-s0, step 50,400 (Thor `runs/refcv7-r101-s0/ckpt.pt`, the route package's checkpoint),
built by the launch tree's own eval loader `tanitad.eval.refcv7_loader.build_model` (STRICT, step asserted). Tree
`fec3a0dccf` on Thor is byte-identical to `git show fec3a0d:<path>` (checked for `refc_v3.py`, `refcv7_loader.py`);
the dev-box extract `C:/Users/Admin/ev7` differs only by CRLF (blob-verified). Line numbers below are `fec3a0d`.
Tier of every later R1 number: OPEN-LOOP single-shot planning on logged frames (EVAL_DOCTRINE), one TRAINING seed of the
trunk, nav = ORACLE (ego-future provenance).

**The question R1 answers.** If retraining ONLY the planner-side heads (tactical, trajectory decoder incl. its DDIM
sampler, selection) on frozen trunk features with corrected labels lifts route following to the gate (turn
direction-correct pick ≥ 0.95, heading-within-15° ≥ 0.70, straight not worse), the information is in the trunk.
Since PLAN §0.1 the default refcv8 run is a WARM-STARTED full run; R1 now measures how much of R8-4 the existing trunk
supports and is the place to debug the tactical conditioning (H5) before the trainer changes.

---

## 1. The cut point

The refcv7 forward (`RefCV3Model.forward`, `refc_v3.py:1963-2307` → `RefCModel.forward`, `refc.py:4312-4981`):

| stage | code | inputs | reads nav? | reads max-speed? | side of the cut |
|---|---|---|---|---|---|
| trunk (resnet101, K = 3 stacks × W = 8) | `encoder.forward_features_s8`, `refc.py:4394-4398` | frames | no | no | **FROZEN, cached** |
| stride-32 map → planner keys/values | `decoder.feat_proj`, `refc.py:3165` (Linear 2048→384) | fmap (last frame) | no | no | **FROZEN; its output `kv` is the cache** |
| pooled sequence | `pooled_seq`, `refc.py:4406` | trunk | no | no | **cached** (`[8, 2048]`) |
| strategic ctx | `self.strategic`, `refc.py:4409` | pooled_seq | via hook only | no | re-run, frozen, off the planner path (`--no-strategic`) |
| 10 cm map branch + planner BEV pool | `RefCV3Model._bev_hook`, `refc_v3.py:1836-1961`; `PlannerBEVPool`, `refcv6_perception_branch.py:293-350` | fmap_s8, fmap_s16, per-clip lift geometry | no | no | **FROZEN; `bev_feats` cached**, `bev_tokens` re-derived by the branch's zero-parameter `bev_tokens()` (`:542-577`) |
| agent slot decoder (300 slots) | `core.agent_head`, `refc.py:4717-4718` | fmap tokens | no | no | **FROZEN; slots cached** |
| measurement encoder | `refc.py:4545-4604` | v0/10, **nav one-hot** | **YES** | no | re-run, trainable |
| hierarchy hook (phi_tac → z_tac, lat/lon tac heads, g_tac, target_latent) | `refc_v3.py:1467-1713` | pooled_seq, **nav** (`nav_inj`→`nav_to_tac`, `:1484-1512`) | **YES** | E16 `max_speed_cond` not built in refcv7 | re-run, trainable |
| agent token embed | `refc.py:4719`, `refc_agents.py:341-361` | cached slots | no | no | re-run, trainable |
| scene hook = tactical behaviour decoder (8 lat / 8 lon / 22 goal) → tac8 anchor priors, behaviour term, `v_limit_ms` | `refc_v3.py:1752-1829` | agent tokens, BEV tokens, **nav one-hot, max-speed one-hot**, v0/a0 (zeros: `ego_state` is None in refcv7) | **YES** | **YES** | re-run, trainable |
| ego history + residual prior `ha0_ext_pose` | `refc.py:4776-4811` | observed pose window + actions (cached small) | no | no | re-run, trainable |
| anchored-diffusion decoder: classifier pass, **DDIM sampler** (`sampler_infer_t` 8, 2 steps, `--f5-emitting-conf`), 117-candidate fan | `refc.py:3165-3403` | kv, **cond = cond_proj(m) + ego_to_cond + tgt_film(target_latent)**, bank(v0, prior), agent tokens, BEV map (coupling 1, gates 0.08 / −0.05 / −0.03 / 0.04) | **YES (through `m` and `target_latent`)** | through the tac8 priors | **re-run, trainable** — the fan is nav-conditioned, so it is regenerated, never cached |
| selection grafts: tac8 lat/lon priors, behaviour term, nav-compliance (`navc_gate` 0.1625), reach band, speed-ceiling filter | `refc.py:3270-3303, 3474-3611` | fan, priors, **nav_cmd_sel**, **v_limit_ms** | **YES** | **YES** (ceiling filters the decoder argmax only — §26.1) | re-run, trainable |
| E9 goal selection | `refc_v3.py:2246-2302` | fan slot 2 s, z_tac (detached), g_tac goal point (detached), `goal_gate` 0.202 | via z_tac | no | re-run, trainable |

**⇒ The cut is the three trunk-side producers** (trunk, perception branch, agent slot decoder) plus `feat_proj`.
Everything a nav, max-speed or tactical input can reach is downstream of it and is re-run from the cache, including
the DDIM fan generation — which is what R8-4 / H5 needs (§7): an extra conditioning input enters `cond` and the
sampler re-generates the fan.

**Where nav enters (five places, all downstream):** (1) `m` → `cond` → every decoder layer and every denoising pass;
(2) `nav_inj` → `z_tac` → tac heads, `g_tac`, `target_latent` (→ decoder FiLM) and E9; (3) the tactical behaviour
decoder's condition → tac8 anchor priors + behaviour selection term; (4) `nav_cmd_sel` → nav-compliance rank term;
(5) `nav_to_str` → strategic ctx (bypassed). **Max-speed enters** in (3) only (one-hot condition and `v_limit_ms` →
the ceiling filter on the decoder argmax). A nav or max-speed substitution in R1 is therefore a pure batch-level
substitution of `nav_cmd` / `v_max_ms`: every consumer re-reads it.

**How the head-only path runs** (`code/r1_heads.py`): the tree's own `RefCV3Model.forward` with four instance-attribute
stubs (no tree file edited): `encoder.forward_features_s8` returns the cached `pooled_seq` and a stride-32 "map" that IS
`kv`; `decoder.feat_proj` becomes a contiguity-only identity; `agent_head.forward` returns the cached slots;
`_bev_hook` returns the cached `bev_feats` + `bev_tokens(bev_feats)`. The pre-forward block is
`forward_like_trainer`, a line-for-line copy of `compute_losses_v3`'s (`refc_v3_train.py:4357-4536`) — validated by
identity control I-0 against the route package's bank, which was made through `compute_losses_v3` itself.

## 2. Frozen vs trainable (MEASURED from `ckpt_50400.pt`, 1,131 tensors)

* **Frozen: 80.3 M** — encoder 58.2 M, law_head 8.4 M (its loss is weight 0 in R1: its target needs a trunk pass on a
  future frame), agent_head 3.9 M, perception box/BEV-pool 4.3 M, strategic 4.1 M, 10 cm map branch 0.64 M,
  `feat_proj` 0.79 M, route_head (no loss under `--no-strategic`).
* **Trainable: 20.15 M** — decoder 14.47 M (incl. sampler/control head, BEV coupling, tac8 projections, navc gate),
  tactical behaviour decoder 2.25 M, phi_tac 2.10 M, core tactical trunk 0.79 M, tac_latent_proj 0.26 M, agent_embed
  0.11 M, measurement / ego_hist / tac heads / behaviour gate / E9 scorer + gate (the remainder).
* Loss = the trainer's own `compute_losses_v3` on the stubbed model with the perception weights set to 0
  (`_w_agent`, `_w_box3d`, `_w_map`, `_w_map_hires` with `_map_hires = None`) and `LAW_WEIGHT = 0`: traj L1 + focal
  anchor CE + F3 cascade + E9 selection CE + goal_tac + the core and z_tac lat/lon CE + the tactical behaviour losses
  (`w_tac_v6`, budget 0.1) — the refcv7 recipe minus the frozen modules' terms.

## 3. The cache (format — shareable with WP-B / WP-D)

Thor `/home/nvidia/refcv8_r1/cache/{eval,train}/shard_XXXX.pt` (256 windows per shard, `torch.save` dict, written
atomically), `selection.json` (window set), `record.json` (provenance + identity summary), `identity_rows.json` (eval).

**Window key:** `win_sha12` (sha256(clip_id)[:12]) + `b_r1_t_now` (the trainer's `V3Dataset._now_s`, raw recording
seconds, float64) + the dataset window index `b_r1_i`; the window start row is `selection.json["windows"][k]["t"]`.

| key | shape (per window) | dtype | bytes | content |
|---|---|---|---|---|
| `kv` | [416, 384] | fp16 | 319,488 | `decoder.feat_proj(fmap_s32[-1])`, 13 × 32 tokens |
| `pooled` | [8, 2048] | fp16 | 32,768 | `pooled_seq` (the W = 8 window) |
| `bev_q` | [96, 120, 64] | uint8 | 737,280 | `bev_feats` (planner grid 60 m × ±16 m at 0.5 m), int8-affine per window × channel |
| `bev_lo`, `bev_scale` | [96] each | fp32 | 768 | dequant: `bev = lo + q * scale` (`r1_lib.dequant_u8`) |
| `slot_presence_logit` | [300] | fp32 | 1,200 | agent slots (the planner reads presence, cls, box, yaw_vec, rates) |
| `slot_cls_logits`, `slot_box`, `slot_yaw_vec`, `slot_yaw`, `slot_rates` | [300,10] [300,4] [300,2] [300] [300,3] | fp16 | 12,000 | |
| `gt`, `gt_valid` | [8, 2], [8] | fp32 / bool | 72 | the tree's `refb_labels.waypoint_targets` |
| `b_*` | the batch's small tensors | as loaded | ≈ 1.6 k | `pose_last`, `future_poses_ext` [60,4], `future_valid_ext`, `pose_hist` [8,4], `actions`, `nav_cmd`, `nav_valid`, `v_max_ms`, `v_max_valid`, `lat_v7`, `lon_v7`, `tac_goal_y/w`, `goal_tac(_valid)`, `route_target`, `map_ep`, `r1_i`, `r1_t_now`, … |
| `full_*` (eval only) | fan [117,8,2], traj, sel_idx, sel_idx_base, s_core, s_e9, reach, p_lat, p_lon, p_goal, g_tac, lat_tac, lon_tac, prior_path | fp32 | ≈ 8.9 k | the FULL model's own outputs at sampler seed 0 (= H0 / the paired baseline) |

**Price** (computed from the tensor the head path actually reads, before writing): ≈ **1.106 MB per train window**,
≈ 1.115 MB per eval window (with `full_*`). **Eval 5,528 windows ≈ 6.2 GB; train 7,000 windows ≈ 7.7 GB; total ≈ 13.9 GB**
(< 15 GB; Thor had 57 GB free at start). A pass refuses to start with < 20 GB free. Time: ESTIMATED ≈ 0.5 s per eval
window (full forward 0.42 s MEASURED by A6 on the same box + two head-only identity forwards) ≈ 46 min; train at
batch 8 ≈ 30–40 min. Actuals in `record.json`.

**Why int8 for the BEV map and fp16 for kv** (and not fp32): fp32 kv + bev would be 4.4 MB per window — 3 400 windows in
the budget. The storage error is MEASURED by identity control I-S (§5) on every eval window, and the arms are compared
against H0 run through the SAME stored tensors, so the codec is common to every arm and cannot create a lever effect.

**For WP-D (perception architecture) — what R1 does NOT cache, and what it would cost** (shapes MEASURED by the probe
hooks, `out_smoke_cpu/eval/probe.json` on Thor):

| tensor | shape | fp16 per window | R1 sample (12,528 windows) |
|---|---|---|---|
| 10 cm map branch input (lift output = encoder output = `map_hires_bev`) | [64, 400, 240] | 12.3 MB | 154 GB (int8 77 GB) |
| near-field branch | [32, 200, 600] | 7.7 MB | 96 GB |
| box decoder memory `box_mem` (fmap_s16 tokens + BEV tokens) | [2144, 256] | 1.10 MB | 13.7 GB |
| fmap_s16 | [1024, 26, 64] | 3.4 MB | 42.7 GB |

None fits beside R1's 13.9 GB. Cheapest useful option for box-head probes: `box_mem` for the 1,112 eval-grid windows +
≈ 2,000 train windows ≈ 3.4 GB — needs a separate budget decision (Thor disk 94 % full). The R1 cache already holds the
planner-grid BEV (`bev_feats`, the BEV half of `box_mem`'s input) for every R1 window.

## 4. Window sets (rules fixed before any number; `selection.json` carries every window)

* **EVAL (held-out eval139, 139 episodes, 23,772 windows):** TURN = every window whose GT class is turnL / turnR by
  the route package's SPEC §3 rule (terminal slot-50→60 heading ≥ 30°, slot 60 valid, ≥ 5 m) — **2,317** (849 L /
  1,468 R); SAMPLE = a seeded (numpy `default_rng(0)`) draw of 2,317 of the remaining windows; GRID = run_route's
  8-per-episode `eval_diag` grid (1,112, for the identity control against the bank). Union **5,528** windows. The
  R1 scored set is TURN ∪ SAMPLE (4,634 — the same counts as A6's dense capture; window-level equality with
  `a6_window_list.json` is checked and reported). ⚠️ Still 38 turn episodes: a denser capture narrows within-episode
  noise only.
* **TRAIN (no eval clip):** `code/prep_train_r1.py` — universe = the refcv7 train cache's clips with a measured clock
  (sidecar row) and a v8 record: 4,347 (22 excluded for no clock row); 1,483 contain ≥ 1 GT-turn window. 700 clips,
  seeded: 350 turn-containing + 350 others (turn-enriched on purpose). Per clip (in `r1_capture.select_windows`): up
  to 5 GT-turn windows + random fill to 10. **7,000 windows: 1,727 GT-turn (846 L / 881 R), 2,851 straight, 708 gentle,
  1,714 unclassified** (MEASURED, CPU smoke on Thor).
* CPU-side GT (window selection) vs the tree's `waypoint_targets` on the captured batch: max |Δ| 7.7e-6 m (eval) /
  9.9e-7 m (train) — the selection classifies the same geometry the arms will score.

## 5. Identity controls (tolerances stated BEFORE the GPU run)

All at sampler seed 0, `torch.manual_seed(0)` before every phase, batch 1, the route package's forward.

| control | compares | pass criterion (fixed now) |
|---|---|---|
| **I-0** | the capture's FULL forward (through `forward_like_trainer`) vs the route package's banked `eval_s0.npz` on the 1,112 GRID windows | fan / s_core / s_e9 / p_lat / p_lon max \|Δ\| ≤ 1e-5 and **0** differing `sel_idx` (the bank re-ran bit-identically, RESULT.md §0) |
| **I-X** | head-only path fed the FULL-precision captured tensors vs the FULL forward, every eval window | max \|Δ\| ≤ 1e-5 on fan (m), scores, posteriors; **0** differing `sel_idx` and `sel_idx_base` |
| **I-S** | head-only path fed exactly the STORED tensors (fp16 / int8) vs the FULL forward, every eval window | differing E9 picks ≤ **1 %** of windows (the inference-seed floor is 8.9 % of picks, RESULT.md §0); p99 fan \|Δ\| ≤ 0.05 m; flips reported with their E9 top-2 margins |
| **I-D** | H0 replayed from DISK (the arms harness) vs I-S's in-process STORE outputs | bit-identical (same tensors, same seed) |

If I-0 or I-X fails, R1 stops and reports. If only I-S fails, the codec is changed (fp16 BEV on fewer windows)
before any arm. **CPU smoke (Thor CPU, 1 eval window, MEASURED):** I-X **0.0** on every quantity; I-S fan 7.2 mm,
traj 1.0 mm, s_e9 5.1e-3, p_lat 1.8e-6, same picks.

## 6. Arm harness (built after registration; nothing scored before)

`code/r1_arms.py` (to be written against SPEC_R1 once registered): loads the train shards into RAM, builds batches
(`frames` = a [B, 8, 1, 1, 1] placeholder never read), per-arm substitutions — `lat_v7` / `lon_v7` ← dense labels
(H2), `nav_cmd` ← announced time-localised nav (H3, `raw/r1_nav_ann_table.json`, A7's G1′ PASSED: seq[0] rule
4,719/4,719, entries[0]==seq[0] 1,799/1,799, suppression-named 68/68), the listwise selection loss (H4), the
tactical-conditioning module (H5); AdamW on the trainable set from the 50,400 weights; eval = the same head path,
batch 1, sampler seeds 0 and 1, on the 4,634 scored windows.

## 7. R8-4 (H5) and R8-3 (H6) in this harness

* **H5 — the decoder conditioned on the tactical decision.** A new zero-init `Linear(16 → 384)` (`tac_cond`) on the
  8 + 8 lat/lon action vector, added to the decoder's `cond` by a forward hook on `decoder.cond_proj` — so it reaches
  every decoder layer, the classifier pass AND both DDIM denoising passes (the same route nav and ego history take).
  Training: teacher-forced, the vector = one-hot of the window's DENSE GT lat/lon label (IGNORE → zeros). Eval: the
  vector = the tactical behaviour decoder's own posteriors of the SAME forward (`tacv6_lat/lon_logits`, available
  because the scene hook runs before the decoder, `refc.py:4739-4752` vs `:4838`), detached. Selection is already
  tactically conditioned through the tac8 anchor priors; under forcing those are forced too (a hook on the tactical
  decoder's output logits, eval only, no loss).
* **Controllability read (H5, PLAN R8-4 ii):** on every classified scored window (GT path ≥ 5 m), force TURN_L /
  TURN_R / LANE_KEEP (lat one-hot, lon = the model's own) on the same scene and read (a) the pick's direction class
  (terminal heading vs the run's τ 10.35°) and (b) the share of fan candidates with that direction. **Bar: the pick
  follows the forced direction on ≥ 0.95 of windows for each forced class; a shuffled-condition arm H5c (trained with
  the conditioning vector of a random other window) must not.**
* **H6 slot (R8-3).** The harness reserves `b_route_ckpt` [3] = (x, y in the ego frame at NOW, valid) and a second
  zero-init `Linear(3 → 384)` into the same `cond`. Data: the Data FlyWheel's
  `FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-04-v9-labels/` (joined on sha12 + `_now_s`); until it exists H6 is
  BLOCKED, not approximated. ECHO control: a planner that drives straight to the checkpoint at constant v0.

## 8. What I could not do / UNVERIFIED

* The `strategic` ctx is computed but off the planner path (`--no-strategic`); it is frozen and not cached as a
  separate tensor (it is re-run from `pooled`).
* `law_head` is frozen and untrained in R1 (its target needs a trunk pass on a future frame). It is not on the planner
  path in refcv7 (no `cons_gate`, MEASURED from the state dict).
* The agent slot decoder, the 10 cm map branch and the box branch are frozen: R1 says nothing about perception.
* One training seed of the trunk; the arms' training seed is single per arm unless SPEC_R1 registers a replicate.
