# PREREG — PERC2PLAN-R8: does refcv8's planner USE its perception inputs?

* **Registered:** 2026-10-21 19:33:44 +0200 (Berlin; dev box `date +%FT%T%z`), BEFORE any intervention arm has
  been rolled on eval data. At registration no hook code exists yet; the only refcv8-48,852 eval numbers that exist
  are the banked BASE battery (`C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/`).
* **Owner:** EvalFlyWheel agent (perc2plan). Not staged, not committed (brief: write only under
  `C:/Users/Admin/refcv8_pf/perc2plan/`).
* **Decides:** whether refcv9 should spend on perception heads at all. If removing the heads' outputs barely moves the
  plan, the heads are decorative for driving and the perception budget belongs elsewhere.

## 1. Object under test

| item | value |
|---|---|
| checkpoint | refcv8 **EMA-48,852** `ckpt_ema_48852.pt`, md5 `b4364029b68f29a33eeb4aebbbbd002b` (dev box `E:/refcv8_eval_kit/ckpt/`, Thor `/home/nvidia/r8ns/ckpt/`) |
| config | `config.json` md5 `4ce2c77230dc4e802e78b39ce05af916` |
| code | launch tree **fb47a58** (`C:/lgs/Lfb47a58376`, Thor copy `/home/nvidia/r8ns/fanceil/tree`), battery package `2026-10-15-refcv8-milestone-battery/code`, per-frame wrapper `2026-10-20-refcv8-perframe-final/code/roll_seed_r8_pf.py`, nav-feed overlay sha256 `98293dd7…` (checked by the wrapper itself) |
| nav feed | `perframe_full` (SHIP-R8 AMENDMENT 3 (i)), `R8_NO_POSE_SYNC=1` (standalone clock) — identical to the banked SHIP-R8 battery |
| windows | eval139, `windows_s1.json` md5 `a3a78dba889e70aa69a05271559794ad` (= `windows_s0.json`), **4,754 windows / 139 episodes** |
| arm set per roll | `--os-only` (SPEC A3: `os` + model-free controls) — the arm set the banked **seed-1** roll used |
| inference seeds | **0 and 1** |
| tier | **T1 self-action OPEN LOOP on logged eval139 frames** (the SHIP-R8 tier); **one training seed** |

### 1.1 The perception → planner channels (read from source, fb47a58)

| channel | tensor | consumer(s) | source line |
|---|---|---|---|
| **AGENT** | `agent_tokens` `[B, 300, 384]` + `agent_pad` from `AgentTokenEmbed` (`refc_agents.py:341-361`) | (i) the operative decoder's cross-agent attention (`refc.py:5174`), (ii) the scene hook → `tac_decoder_v6` (sources `agent`,`bev`; `refc.py:5073`) whose lat/lon posterior, valid-behaviour set and speed ceiling feed anchor prior + selection | `refc.py:5037-5043` |
| **BEV (pooled)** | `bev_tokens` (`bev_source = map_hires_pool`, 30×16 = 480 tokens, d 96) | scene hook → `tac_decoder_v6` | `refc.py:4798`, `refc_v3.py` `_bev_hook` |
| **BEV (dense)** | `bev_feats` `[B, 96, X, Y]` | decoder BEV waypoint coupling (`bev_coupling.built.enabled = True`, 8 points) | `refc.py:4826`, `:5175` |
| agent_pos | `agent_slots["box"][..., :2]` | `_agent_index` only, which returns early when `wp_index` is None — **config `seams.wp_index = None` ⇒ INERT**; not intervened | `refc.py:2713`, `:5051` |
| drivable critic | `map_hires_logits` → `r8_drv` | **not built** (`critic_drivable: false`, `w_r8_drivable 0.0`) — asserted at runtime (`model.r8_drv is None`) | `refc_v3.py:2620-2629` |

The image feature map `fmap` (trunk) reaches the decoder directly and is **not** a perception-head output; it is
untouched. ⇒ The question answered is *"does the plan depend on the perception HEADS' outputs, over and above the
trunk features the decoder already reads?"*

## 2. Arms (all eval-time; no file of the shared code is edited)

Interventions are installed by a wrapper (`code/perc2plan_roll_r8_pf.py`) that imports `roll_seed_r8_pf` as a module
exactly as the banked battery child runs it (the `fan_roll_r8_pf.py` pattern), then:
* **AGENT hook:** `model.core.agent_embed.register_forward_hook` replaces the module's output `(tokens, pad)`.
  Because the replacement happens at the source, BOTH agent consumers (decoder + scene hook) receive it.
* **BEV hook:** the instance attribute `model._bev_hook` is wrapped; the returned closure's dict has `bev_tokens`
  and `bev_feats` (and `bev_pad` if present) replaced. Both BEV consumers receive it.

| arm | AGENT channel | BEV channel (`bev_tokens` AND `bev_feats`) |
|---|---|---|
| **BASE** | unchanged (hooks installed, identity) | unchanged |
| **AGT_ZERO** | `agent_tokens := 0` (pad unchanged) | unchanged |
| **AGT_SHUF** | `agent_tokens`, `agent_pad` := the DONOR window's | unchanged |
| **BEV_ZERO** | unchanged | `bev_tokens := 0`, `bev_feats := 0` (bev_pad unchanged) |
| **BEV_SHUF** | unchanged | `bev_tokens`, `bev_feats`, `bev_pad` := the DONOR window's |
| **BOTH_ZERO** | `agent_tokens := 0` | `bev_tokens := 0`, `bev_feats := 0` |

**Donor (SHUF arms).** The roll calls the model ONE window per forward (batch 1, `nav_true` row only), so a
within-batch permutation does not exist. The donor is therefore **cross-clip and fixed by processing order**:
window *j* of episode *e* (e ≥ 1) receives the **un-intervened** tensors of window `min(j, L_{e-1} − 1)` of episode
*e − 1*. Episode 0 has no predecessor: its window *j* receives the un-intervened tensors of its own window
`(j + ⌊L/2⌋) mod L` (≈ 8 s away in the same clip). **Disclosed in advance:** 35 of 4,754 windows (0.74 %) carry an
intra-clip donor. The verdict uses all 4,754 windows; a beside readout excludes episode 0.

## 3. Metrics (four families, battery's own family code)

Per arm and seed, against GT, then **arm − BASE at the SAME seed**:
* **S2** (2 s; waypoints 0.5/1/1.5/2 s; dt 0.5): `os` vs `g` from the roll's `ep*.npz`.
* **S6** (1..6 s; dt 1.0): `plan_full_nav_true[keep][:, [1,3,4,5,6,7]]`, keep = `gt_future_valid_ext[:, 59] > 0.5`;
  GT `g` from the banked `s6_s{seed}` (alignment asserted: same ws, same clip_index per episode).
* Components: `taniteval/tools/refav1_arm.py::_components` (fb47a58, sha256 `b0ec1de4…`) + pair-masked
  `LAT_curvature_mae_1pm` and `LAT_yaw_rate_mae_masked_radps` exactly as
  `2026-10-21-refcv8-wiseft/gate/code/pai_gate_families.py::components` adds them.
  Families: **ADE** (ade_m, fde_m), **longitudinal** (speed, along, accel MAE), **lateral** (cross-track, heading,
  yaw-rate, curvature, masked yaw-rate), **tactical** (`TAC_traj_lat_correct`, `TAC_traj_lon_correct`).
* **Strategic family: NOT COMPUTABLE by this instrument** — `_paired_families` has no strategic component, and the
  nav-obedience block exists only in the full-arm-set acceptance stage (not run on an os-only roll). Reported as
  absent with this reason, n = 0.
* Interval: `refav1_arm._paired_families` = **paired episode-cluster bootstrap, B 2000, 139 episodes**, bootstrap
  seed 0. ⚠️ It answers **"would another draw of EPISODES say this?"** only — not another training run (one training
  seed) and not another inference run (that is what the second inference seed and the floor below are for).
* Beside (descriptive, no verdict): **plan displacement** = mean / p95 over windows of the mean-over-waypoints L2
  distance between arm and BASE plans (S2 and S6), against the same quantity for BASE_s0 vs BASE_s1 (the
  inference-seed displacement); fraction of windows whose `sel_idx` changes; `tac_decoder_v6` lat/lon argmax
  change rate from `refcv6_extras.npz`.

## 4. Verdict — both outcomes committed now, read mechanically

**Inference-seed floor** (per surface): `floor_S = |Δ ade_m(BASE_s1 − BASE_s0)|` on surface S, paired, same windows,
from THIS run's BASE rolls.

Per arm X, per surface S ∈ {S2, S6}:
* **USED** iff at **both** seeds, `ade_m(X) − ade_m(BASE)` is **separated** (95 % CI excludes 0), **positive**
  (X worse), and **> floor_S**.
* **NOT USED** iff **no** family component is separated-WORSE at **both** seeds (worse = error metric up, `TAC_*`
  agreement down).
* **PARTIAL** otherwise.

Per arm X, headline: **USED** if USED on at least one surface (name it); **NOT USED** if NOT USED on both surfaces;
**PARTIAL** otherwise. A NOT USED arm that has components separated-**BETTER** at both seeds is annotated
`(+ separated-BETTER: …)` — the input moves the plan, but removing it does not hurt.

Programme-level reading committed now:
* AGT_ZERO **and** BEV_ZERO **and** BOTH_ZERO all NOT USED ⇒ "the perception heads are decorative for the plan at
  T1 open loop"; refcv9's perception budget should go elsewhere (or into making the planner read them).
* Any ZERO arm USED while its SHUF twin is NOT USED ⇒ the planner reads the *presence/scale* of the channel, not
  its *content* (a zero input is out of distribution; a donor scene is not). Reported as such, not as "used".
* SHUF arm USED ⇒ the planner reads scene-specific content of that channel — the strongest form of USED.

## 5. Controls that must read a known value (VOID gates unless stated)

* **C1 identity (CPU smoke, VOID):** wrapper with arm BASE == the unmodified battery child (`roll_seed_r8_pf.py`
  run directly) on the same device, same seed, same windows — `os` and `plan_full_nav_true` **bit-exact**.
* **C2 reproduction of the banked battery (Thor, s1):** BASE_s1 `os` vs banked
  `step48852ema_pf/dump_s1` `os` (dev box RTX 4060, torch 2.11): **bit-exact or max |Δ| ≤ 1e-5**. ⚠️ Stated in
  advance: Thor runs torch 2.13 on a different GPU, and DDIM noise is drawn on device, so a cross-device bitwise
  match is NOT expected to hold; if C2 fails it is reported as FAIL with the max |Δ|, the fraction of windows
  bit-identical, and the paired BASE_thor − banked family block, and the verdict (which is Thor-internal: every arm
  vs the Thor BASE on the same device) is stamped `C2-FAIL (cross-device reproduction not established)` — it is
  not silently passed. **Seed 0 is NOT C2-checkable:** the banked s0 roll used the FULL arm set (four nav
  conditions batched per forward + obey30/vmax-zero/filter-off calls), so its DDIM draw stream differs from an
  os-only roll by construction; BASE_s0 vs banked s0 is reported beside, not gated.
* **C3 same-device reproduction (Thor, s1, VOID if available):** BASE_s1 `os` == the fan-roll agent's Thor s1
  os-only dump (`/home/nvidia/r8ns/fanceil/out/s1/dump`, same code path, same seed, same device) **bit-exact**.
  If that dump does not exist at analysis time, C3 is reported ABSENT (not passed).
* **C4 hook reach (every Thor roll, VOID):** per arm, counted per forward: the hooked tensor differs from the
  original on ≥ 99 % of forwards for SHUF arms (a donor identical to the target is possible only by accident) and
  on 100 % of forwards with a non-zero original for ZERO arms; ZERO arms read **max |hooked| = 0 exactly**; BASE
  reads **max |hooked − original| = 0 exactly**; the decoder's `agent_tokens` / `bev` kwargs are the hooked tensors
  (forward pre-hook, identity check); `model.r8_drv is None`.
* **C5 alignment (VOID):** every arm's dump has the same (clip_index, ws) per episode and bit-identical `g` as BASE
  at the same seed; S6 keep masks identical.
* **C6 self-pair:** BASE − BASE paired block reads exactly 0 on every component.

## 6. Execution order (bank each step)

Thor, under `/home/nvidia/refcv7_post/thor_gpu.lock` (flock), only when neither the fan-roll
(`/home/nvidia/r8ns/fanceil/`) nor the perception pass (`/home/nvidia/r8ns/perc8/`) has a python process:
`s1: BASE, AGT_ZERO, BEV_ZERO, BOTH_ZERO, AGT_SHUF, BEV_SHUF` then `s0:` the same order. ≈ 24 min per roll
(fan-roll measured rate on Thor, ~10 s / episode) ⇒ ≈ 4.8 h. A killed queue still yields the s1 block.
Inputs on Thor are the fan-roll agent's staged files, reused read-only, md5-verified before launch.

## AMENDMENT 1 — 2026-10-21T19:37:09+0200 (Berlin), BEFORE any intervention arm has been rolled (smoke or eval)

**What changes:** the SHUF donor for the FIRST processed episode only. **Why:** the registered rule (§2: episode 0's
window *j* takes its own window `(j + ⌊L/2⌋) mod L`) is **causally unavailable** — for *j* < L/2 that donor window has
not been computed yet when window *j* is rolled (one window per forward, in order). Found while writing the hook, not
from any data.
**New rule:** the first processed episode's window *j* takes the un-intervened tensors of window `min(j, L − 1)` of the
**LAST** processed episode of the **same-seed BASE roll**, which that roll banks to disk (`--donor-bank-out`, a
torch file of the last episode's agent tokens/pad and bev_tokens/bev_feats/bev_pad, sha256 recorded) and every SHUF
roll loads read-only (`--donor-bank-in`). Every episode now has a **cross-clip** donor (cyclic predecessor in
processing order); the "intra-clip donor on 0.74 % of windows" disclosure and its beside readout are withdrawn.
A SHUF roll therefore runs after BASE at the same seed (already the registered order). Counted at run time:
`donor_same_clip` must read 0. Everything else in §1–§6 is unchanged.
