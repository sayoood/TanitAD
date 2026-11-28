# AUDIT — the refcv5 surface a reasoner would attach to

**Scope.** Everything below is read from SOURCE in this repo on 2026-09-06 and carries `file:line`.
No claim rests on `REFCV5_DESIGN_PLAN.md`, on `MODEL_REGISTRY.md`, or on any prior conversation.
Where a design document says something this audit could not confirm in code, the audit says so.

**Evidence classes.** `MEASURED (ours + artifact path)` · `PUBLISHED` · `INHERITED` · `ESTIMATED`.
Every parameter number in §5 is `MEASURED` by instantiating the model. Nothing was trained.

## ⚠️ CITATIONS ARE PINNED TO BLOBS — the files moved WHILE THIS AUDIT RAN

`refc.py` shifted by +13 lines below line ~2400 and `refc_v3_train.py` by ~+110 lines during this
session (siblings are editing them live). Every line number below was **re-derived by literal
string match against these exact blobs** in the final sweep:

| file | blob (`git hash-object`) | bytes |
|---|---|---|
| `stack/tanitad/refs/refc.py` | `407bf7e1966f9aca654fec7a1dd41045411a2756` | 182,980 |
| `stack/tanitad/refs/refc_v3.py` | `fea53c7ff2d617bdb6124c4008ae382902a6ce1b` | 74,603 |
| `stack/scripts/refc_v3_train.py` | `89f709507051464db656d5fce30e414ebe8fe407` | 213,308 |
| `stack/tanitad/refs/refc_agents.py` | `5cb639c491fe3f65562a1a04d04e866b20972c72` | — |
| `stack/tanitad/refs/refc_selector.py` | `bf5832fc83a0a47480a9abe3d435e4c377cbb844` | — |
| `stack/tanitad/refs/refc_sampler.py` | `1a908d87e6bb2cb59822cff790af231806a2ba0d` | — |
| `stack/tanitad/data/rig_projection.py` | `edc507bfa1fe814261cbc70e96b3f8a28cb39dbc` | — |
| `stack/tanitad/data/lan.py` | `0c9522c34382d8eb0065bfcd048f65a8f8c83c03` | — |
| `stack/tanitad/refs/anchor_meta.py` | `2b251bdec466223166f5c21439be925ddaf3b072` | — |
| `stack/tanitad/models/tactical.py` | `e0da1f626ca76243f5a0a8626a9407e80cae5581` | — |

⇒ **If a blob differs from the table, re-derive the line by symbol name.** `REFCV5_DESIGN_PLAN.md`
§9's own citations (*"the `hierarchy_hook` block at `:2268-2300`"*, *"`refc.py:1645-1647`"*,
*"`refc.py:1206`"*) are **already stale**; the live values are `refc.py:2762-2782`,
`refc.py:2061-2074` and `refc.py:1349`.

---

## ⛔⛔ FINDING 0 — READ THIS BEFORE ANY OF THE SEVEN ANSWERS

**refcv5's model-side seams exist in the WORKING TREE and are NOT IN `HEAD`.** The refcv5
*modules* are committed; the *wiring that calls them* is not.

MEASURED by extracting the HEAD blob (`HEAD:stack/tanitad/refs/refc.py` =
`0e6103e53bbf809c1b02dbfe75c557c1647d5f05`, 155,407 bytes read) and counting symbols, with
same-breath controls that must read non-zero — CLAUDE.md's rule that a zero count from a file
that could not be READ is indistinguishable from a genuine absence:

| symbol | HEAD `refc.py` | worktree `refc.py` |
|---|---:|---:|
| `agent_tok` (WP-6 wiring) | **0** | 19 |
| `control_head` (WP-4 denoiser) | **0** | 12 |
| `cross_agent` (WP-6 attention) | **0** | 11 |
| `time_mlp` (WP-4) | **0** | 6 |
| `sampler` (WP-4) | **0** | 52 |
| `lan_to_cond` — **CONTROL** | 7 | 7 |
| `maneuver_to_anchor` — **CONTROL** | 4 | 5 |

Both controls read non-zero on both sides, so the HEAD file was really read and the zeros are a
real absence, not a failed query.

Blob comparison per path (`git rev-parse HEAD:<p>` vs `git hash-object <p>`, both asserted to be
40 chars, INCONCLUSIVE otherwise):

* **IN-HEAD-IDENTICAL:** `refc_agents.py`, `refc_selector.py`, `refc_selector_targets.py`,
  `rig_projection.py`, `refcv5_validate.py`, `feasible_decode.py`, `contact_projection.py`
* **IN-HEAD-BUT-MODIFIED:** `refc.py`, `refc_sampler.py`, `refc_v3_train.py`, `refcv5_preflight.py`

⚠️ **The consequence is not cosmetic.** In `HEAD`, `refc_v3_train.py` already contains
`_pin_refcv5_seams` (3 occurrences) and `AgentSeamConfig` (2), while HEAD's `refc.py` has **no
`sampler` field and no `control_head`**. That is exactly the FALSE-PROVENANCE defect the
worktree's own guard documents at `stack/scripts/refc_v3_train.py:2064-2077`
(`assert_seams_are_built`): *"the model contained no denoiser at all … a run would have trained,
converged, written a checkpoint and STATED IN ITS OWN RECORD that it used a sampler it did not
have."* `assert_seams_are_built` occurs **0 times in HEAD's trainer** and 3 times in the
worktree's — **the fix is uncommitted.**

⇒ **ESCALATION (integration, not a note):** `refc.py`, `refc_v3_train.py`, `refc_sampler.py` and
`refcv5_preflight.py` must be committed before any refcv5 arm is launched or any TanitLang
attachment is built against these ports. Until then a fresh clone of `HEAD` **cannot build a
refcv5 model at all**, and every line number in §1–§7 refers to the working tree.

**Second finding, same family:** `stack/tanitad/refs/refc_selector.py` (WP-7, **4,530,444**
params, MEASURED §5c) is committed and **wired to nothing**. Two independent probes: a worktree
grep for `refc_selector|SubMetricSelector|SelectorConfig` across `stack/scripts/` +
`stack/tanitad/` returns exactly one hit outside the module — `stack/scripts/refcv5_preflight.py:100`,
an *import* probe; `git grep` on `HEAD` over the same paths returns the same single hit. It is
not constructed by `RefCModel.__init__` (`refc.py:2350`) and has **no trainer flag**.

---

## 1. The conditioning bus

**There is not one bus. There are THREE, and they enter the decoder by three different
mechanisms.** All three land on the same decoder width.

**`d_model` = 384** — `DecoderConfig.d` (`stack/tanitad/refs/refc.py:397`), pinned for every
REF-C v3/v4/v5 arm by `_v3_core_base` (`stack/tanitad/refs/refc_v3.py:418`:
`cfg.decoder = refc.DecoderConfig(d=384, n_heads=8, layers=4, ff_mult=4, …)`).

⚠️ **384 is the DECODER width and is not the model's only width.** `feat_dim = base_width * 8`
(`refc.py:306-308`) = **512** at `small`, **704** at `base` (`V3_SIZES`, `refc_v3.py:486-490`);
`d_ctx` = **256** (`refc_v3.py:436`); `d_tac` = **512** (`refc_v3.py:334`);
`tactical_latent_dim` = **512** (`refc.py:768`). A reasoner must read these off the built model,
never hardcode one.

### 1a. `cond` — the scalar condition bus, `[B, 384]`

Constructed in `AnchoredDiffusionDecoder.forward` (class at `refc.py:1260`), four consecutive
statements:

| line | term | shape in | init |
|---|---|---|---|
| `refc.py:1988` | `cond = self.cond_proj(m)` | `m [B, 128]` (`MeasurementConfig.d_out=128`, `refc.py:335`) | default (**LIVE**) |
| `refc.py:1990` | `cond = cond + self.ctx_to_cond(ctx)` | `ctx [B, 256]` | **ZERO** (`refc.py:1349-1351`) |
| `refc.py:1992` | `cond = cond + self.lan_to_cond(lan_emb)` | `lan_emb [B, 64]` | **ZERO** (`refc.py:1399-1401`) |
| `refc.py:1994` | `cond = self.tgt_film(cond, self.tgt_proj(target_latent))` | `target_latent [B, 512]` | `tgt_proj` `refc.py:1384`; `tgt_film = FiLM(d, d)` `refc.py:1385`, `zero_init=True` default (`refc.py:1181`) ⇒ **ZERO** |

**Consumed in exactly one place:** `CrossAttnLayer.forward`, `refc.py:1231` —
`q = q + self.mlp(self.film(self.norm_f(q), cond.unsqueeze(1)))`. That layer's own `film` is
**LIVE** (`zero_init=False`, `refc.py:1203`), so the measurement condition steers from step 0.
The bus reaches the layers via `_decode` (`refc.py:1680`) and, when the sampler is on, via
`_decode_ctrl` (`refc.py:1700`) — **the same weights**, deliberately (`refc.py:1705-1709`).

`m` is built at `refc.py:2845` by a 2-layer MLP (`refc.py:2390-2393`) over
`d_meas_in = 1 (v0) + 4 (nav one-hot) + ego_valid bit + nav_known bit` (`refc.py:2388-2389`;
`NAV_COMMANDS` at `refc.py:170`).

### 1b. `kv` — the perspective-view token bus, `[B, P, 384]`

`refc.py:1987`: `kv = self.feat_proj(fmap.flatten(2).transpose(1, 2))`, with
`feat_proj = nn.Linear(feat_dim, d)` (`refc.py:1336`). `P = gh * gw` from
`CNNEncoderConfig.grid_shape` = `(H // 32, W // 32)` (`refc.py:326-330`).
**MEASURED by building:** at the default square 256 that is `(8, 8) = 64` tokens; at the live
corpus geometry `--image-hw 256 640` (`refc_v3_train.py:200-206`) it is **`(8, 20)` = 160
tokens**. Cross-attended at `refc.py:1228` (`self.cross(h, kv, kv, …)`).

### 1c. `agent_tok` — the agent token bus, `[B, 100, 384]` (WORKTREE ONLY, default OFF)

A **separate** `nn.MultiheadAttention` per decoder layer, **not** a concatenation onto `kv`:
declared `refc.py:1214-1216`, built `refc.py:1217-1221`, applied `refc.py:1229-1230`
(`q = q + self.agent_gate * self._attend_agents(q, agent_tok, agent_pad)`), with
`agent_gate = nn.Parameter(torch.zeros(1))` (`refc.py:1221`) — **zero-init, so the branch is a
no-op at step 0**; rationale `refc.py:1206-1213`. Details in §2.

---

## 2. Grounding inputs

The PI is right that refcv5 addresses environment grounding. **The mechanism is AGENT TOKENS
(WP-6), not BEV tokens and not map tokens.** Both halves of that sentence are established below.

### 2a. What exists — the agent seam (`E-AGT-HEAD`)

* **Config:** `AgentSeamConfig`, `stack/tanitad/refs/refc_agents.py:106-190`.
  `refc_agents.py:108-110`: *"Every default is OFF — a build that does not ask for agent tokens
  constructs nothing and is bit-identical to refcv4b, including RNG draw order."*
* **How many tokens:** `queries: int = 100` (`refc_agents.py:157`). The comment at
  `refc_agents.py:124-155` records the ruling and its measurement — train join 2,308 episodes /
  433,040 frames / 12,122,129 boxes, mean 4.39 · p99 30 · max 94 per frame; at N=32,
  *"41,362 boxes (2.18 %) dropped across 3,250 frames, nearest sacrificed target at 13.1 m"*.
  `INHERITED` (source comment), not re-measured here.
* **Token width:** the head runs at `d_model: int = 256` (`refc_agents.py:158`), `depth: int = 3`
  (`:159`), `n_heads: int = 8` (`:160`); `AgentTokenEmbed` (`refc_agents.py:246-289`) projects to
  the **decoder's** width — `Sequential(Linear(TOKEN_FEAT_DIM + N_AGENT_CLASSES, 256), GELU,
  Linear(256, d_out))` + `LayerNorm(d_out)` (`refc_agents.py:262-267`), `d_out = cfg.decoder.d
  = 384`. `TOKEN_FEAT_DIM = 15` (`refc_agents.py:100`); the 15 geometric features are enumerated
  at `refc_agents.py:230-245` (normalised `cx, cy`; `l, w`; `sin/cos` yaw; range;
  `log1p(range)`; bearing `cos/sin`; `1/(1+range)`; three relative rates; presence).
* **Where they enter:** slots produced at `refc.py:2934-2959` (`RefCModel.forward`), tokens at
  `refc.py:2959`, passed to the decoder at `refc.py:2969`
  (`agent_tok=agent_tok, agent_pad=agent_pad`), forwarded to every layer by `_decode`
  (`refc.py:1680`) / `_decode_ctrl` (`refc.py:1700`), consumed at `refc.py:1229-1230`.
* **How they are fused: CROSS-ATTENTION, gated, additive on the residual — NOT concatenation, NOT
  additive on `cond`.** `refc.py:1217-1221` + `refc.py:1229-1230`. `refc.py:1209-1213`:
  *"a fresh +agents run starts bit-identical to the agent-free one … while the gradient
  (agent_out * dL/dq) is non-zero, so it is GATED, not dead."*
* ⚠️ **A padding trap already handled that a reasoner reusing this path must not undo:** a
  fully-padded row makes `MultiheadAttention` return **NaN, not zero**; the row is un-masked and
  its contribution zeroed afterwards (`_attend_agents`, `refc.py:1234-1266`).
* **Optional or always on: OPTIONAL — two flags that must agree.**
  * `AgentSeamConfig.enable: bool = False` (`refc_agents.py:113`)
  * `DecoderConfig.cross_agent: bool = False` (`refc.py:456`)
  * `RefCConfig.agents: "object | None" = None` (`refc.py:579`) — `None`, not a disabled config,
    is the OFF state so the run stamp can tell "off" from "absent" (`refc.py:570-578`).
  * Construction is refused unless both are set: `refc.py:2431-2440` raises
    *"`agents.enable` is set but `decoder.cross_agent` is False, so the head would be BUILT,
    SUPERVISED and STAMPED while its tokens reached no decoder layer."*
* **Vision-only enforcement, in signature form:** the learned head is called with `fmap` and
  nothing else — `refc.py:2957-2958`
  (`agent_slots = self.agent_head(fmap.flatten(2).transpose(1, 2))`), with the rule stated at
  `refc.py:2935-2941`: *"THE VISION-ONLY RULE IS ENFORCED BY WHERE THE TENSOR IS READ … the
  detection LABELS never enter here at all; they enter the LOSS, in the trainer."*
  The **oracle** path (`refc.py:2944-2956`) reads `agent_gt` at inference and is declared
  inadmissible as a capability claim (`refc.py:2420-2427`, `refc_agents.py:114-117`).
* **The rig↔image bridge** for the monocular auxiliaries is `stack/tanitad/data/rig_projection.py`
  (in HEAD, identical). Its header carries the projection fact any grounding of a token to image
  space must respect: 256×640 **cylindrical**, `f_ref = 305.577`,
  `HFOV = 2·(W/2)/f_ref = 120.0°`, and *"the pinhole formula `2*atan((W/2)/f)` yields 92.6 deg and
  is entirely plausible-looking"* (`rig_projection.py:10-21`). Frames: rig `+x fwd, +y LEFT,
  +z UP`; camera `+x right, +y down, +z boresight`; image centre `((H-1)/2, (W-1)/2)`
  (`rig_projection.py:24-40`).

### 2b. What does NOT exist — BEV tokens, LiDAR tokens, map tokens

**Two independent probes, each with a same-breath control that read non-zero:**

* Probe 1 (worktree grep, case-insensitive `bev`): `refc.py` **0**, `refc_v3.py` **0**,
  `refc_v3_train.py` **2**. Same-command controls: `agent` in `refc_v3_train.py` = **299**;
  `agent_tok` in `refc.py` = **19**.
* Probe 2 (`git grep` on `HEAD` for `bev_raster|v2bev|bev_tok` over `stack/tanitad/refs` +
  `stack/scripts/refc_v3_train.py`): the only hits are **four inside `refc_agents.py`** — a
  docstring table (`:18`, `:26`), the import of the **class enum and grid constants**
  `from tanitad.data.bev_raster import ALL_CLASSES, GRID_DEFAULT` (`:71`), and the FOV predicate
  (`:323`).

⇒ **`bev_raster` is used for the LABEL side (the 10-class enum, the FOV mask, the box frame),
never as a token stream into the decoder.** There is no BEV token, no LiDAR token, no map token
and no lane-graph token in the model. The nearest thing to a route input is `lan_emb`, and it is
refused for every registered arm (§4).

### 2c. The grounding flags and their defaults, verbatim from source

```
refc_agents.py:113   enable: bool = False
refc_agents.py:117   oracle: bool = False
refc_agents.py:120   oracle_sigma_range_m: float = 0.0
refc_agents.py:121   oracle_miss_rate: float = 0.0
refc_agents.py:157   queries: int = 100
refc_agents.py:158   d_model: int = 256
refc_agents.py:159   depth: int = 3
refc_agents.py:160   n_heads: int = 8
refc_agents.py:162   enforce_band: bool = True
refc_agents.py:166   w_project: float = 0.0
refc_agents.py:168   w_ground: float = 0.0
refc_agents.py:171   presence_gate: float = 0.5
refc_agents.py:175   presence_hard: bool = False
refc.py:456          cross_agent: bool = False
refc.py:579          agents: "object | None" = None
```

---

## 3. The anchor mechanism

### 3a. How many anchors, and where the anchor file is loaded

* **Config default 128:** `AnchorConfig.n_anchors: int = 128` (`refc.py:349`), set by
  `_v3_core_base` (`refc_v3.py:421`: `refc.AnchorConfig(n_anchors=128, pool_size=4096)`).
* **Overridden per run:** `refc_v3_train.py:241-245` —
  *"the vocabulary SIZE is a property of the built vocabulary, not of the model family: a
  v0-conditioned (accel, curvature) product grid is odd × odd, so it can never be exactly 128."*
  ⚠️ The **117 = 13 × 9** grid shape is **UNVERIFIED from source in this pass** — the artifact is
  pod-side and absent from the repo (`find . -maxdepth 4 -name "anchors*.pt"` → nothing;
  `ls stack/tanitad/refs/*.pt` → nothing). What IS in source is that `n_anchors` is a free
  run-time value and every anchor-shaped module sizes itself from `anchors.shape[0]`
  (`refc.py:1372`, `:1374`, `:1378`, `:1415`).
* **Loaded in two steps:**
  1. `refc_v3_train.py:1741-1767` — `_read_anchor_artifact(args)` reads `--anchors`
     (`refc_v3_train.py:3263`) through `tanitad.refs.anchor_meta.read_anchor_artifact`,
     **refusing before any GPU work** a `controls`-carrying file that declares no
     `control_units` unless `--anchor-control-units` (`refc_v3_train.py:3281`) is passed, and
     stashing the file's declared constants in `args._anchor_artifact_meta`.
  2. `AnchoredDiffusionDecoder.load_anchors` (`refc.py:1510`) copies into the buffers and
     **raises** if `controls` is absent on a `v0_conditioned` build.

### 3b. Shapes and `control_units` metadata

**MEASURED by building** (`refc_v4_config("base")`, `n_anchors=117`, `v0_conditioned=True`,
`control_units="alat"`, `--image-hw 256 640`):

```
anchors               buffer  [117, 8, 2]   persistent      refc.py:1296
anchor_controls       buffer  [117, 2]      persistent      refc.py:1312-1313
anchor_slots          buffer  [8]           NON-persistent  refc.py:1319-1321
anchor_control_units  "alat"                                refc.py:1309
```

`S = 8` because `V3_HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)` (`refc_v3.py:120`) — 6.0 s at
10 Hz. Units policy at `refc.py:1306-1308`: `control_units` must be `"kappa"` or `"alat"` or the
decoder raises. The rationale (`refc.py:379-392`) records the incident — *"a constant-CURVATURE
family is unflyable at speed (a_lat = v²·kappa, so kappa 0.06 at 27 m/s is 4.5 g and 104 of 117
anchors break a mu = 0.7 circle)"* vs `alat`'s *"0.1987 m oracle-in-vocabulary (−0.1009
[−0.1213, −0.0813] vs ha) with 0/117 over mu = 0.7 and a 0.68 g peak"* — `INHERITED` (source
comment), not re-measured here. The builder contract is
`anchor_meta.build_anchor_artifact` (`anchor_meta.py:151-217`) whose `REQUIRED_META` is
`("control_units", "horizon_s", "dt", "ref_speed_ms", …)` (`anchor_meta.py:61`).

### 3c. Selection logits — the complete family of zero-initialised prior ports

There are **two distinct surfaces plus one above the decoder**, and `REFCV5_DESIGN_PLAN.md` §9.3
conflates the first two. Getting this wrong means a reasoner writes into a surface that does not
decide.

**Surface A — the anchor CONFIDENCE (`conf`), assembled at `refc.py:2059-2075`.**
Every term is `[B, N]` and is summed through `_apply_grafts`.

| port | built at | shape | init | applied at |
|---|---|---|---|---|
| `maneuver_to_anchor` | `refc.py:1378-1379` | `Linear(n_maneuvers=5, N, bias=False)` | **default (LIVE)** — *"the coupling is the point of the seam"* (`refc.py:1352-1354`) | `refc.py:2061-2063` |
| `lat_to_anchor` | `refc.py:1372-1373` | `Linear(N_LAT_MAN=3, N, bias=False)` | **default (LIVE)** — *"inherits the LIVE H19 role"* (`refc.py:1360-1369`) | `refc.py:2067-2068` |
| `lon_to_anchor` | `refc.py:1374-1376` | `Linear(N_LON_MAN=3, N, bias=False)` | **ZERO** (`nn.init.zeros_`, `refc.py:1376`) | `refc.py:2069-2070` |
| `lan_gate` | `refc.py:1402` | `nn.Parameter(torch.zeros(1))` — **1 scalar** on a param-free geometric compatibility | **ZERO** | `refc.py:2072-2074` |

⚠️⚠️ **`maneuver_to_anchor` and the `lat/lon` pair are MUTUALLY EXCLUSIVE.** `refc.py:1371-1379`:
`if graft_maneuver and factored_maneuver:` builds the pair; `elif graft_maneuver:` builds the
5-way. `_v3_core_base` sets `cfg.factored_maneuver = True` (`refc_v3.py:437`), so **on every
registered refcv3 / v4 / v5 arm `maneuver_to_anchor` is `None`.** The design plan's claim that it
is *"the only port through which an external module already changes behaviour with zero new
wiring"* is therefore **FALSE for the built arm**. The equivalent live ports are `lat_to_anchor` /
`lon_to_anchor`, fed by `lat_prior` / `lon_prior` (`refc.py:2866-2872`).

**Surface B — the RANKED score (`sel_score`), assembled at `refc.py:2160-2183`.**
This is the surface the decoder's `argmax` reads (`refc.py:2209`).

| port | built at | shape | init | applied at |
|---|---|---|---|---|
| `route_to_anchor` | `refc.py:1415-1417` | `Linear(N_ROUTE=3, N, bias=False)` (`N_ROUTE` at `refc.py:182`) | **ZERO** (`refc.py:1417`) | `refc.py:2162-2163` |
| `goal_gate` (decoder) | `refc.py:1443` | `nn.Parameter(torch.zeros(1))` | **ZERO** | `refc.py:2169-2171` |
| `goal_dist_gate` | `refc.py:1444` | `nn.Parameter(torch.zeros(1))` | **ZERO** | `refc.py:2172-2175` |
| `cons_gate` | `refc.py:1424` | `nn.Parameter(torch.zeros(1))` | **ZERO** | `refc.py:2176-2181` |

**Surface B′ — the v3 hierarchy's own graft, ONE LEVEL ABOVE THE DECODER.**
`RefCV3Model.forward` re-ranks the emitted fan after the core returns:
`graft = self.goal_gate * sc["score"]` (`refc_v3.py:1075`), with
`self.goal_gate = nn.Parameter(torch.zeros(()))` — **ZERO-INIT** (`refc_v3.py:782`) — then
`sl.apply_seam_clamp` (`refc_v3.py:1096`) and `idx = rank.argmax(dim=1)` (`refc_v3.py:1104`).
**This, not the decoder's `sel_idx`, is what a hier arm actually emits:** `refc_v3.py:1108`
preserves the core's pick as `traj_base` / `sel_idx_base` and overwrites `out["traj"]` /
`out["sel_idx"]`.

**Zero-init ports in the hierarchy cascade (not selection, same discipline):**

| port | built at | shape | init |
|---|---|---|---|
| `gstr_film` (E4) | `refc_v3.py:660-662` | `Linear(d_gcond=64, 2·d_tac=1024)` | **ZERO** |
| `nav_to_tac` (E13) | `refc_v3.py:675`, zeroed `:677-679` | `Linear(d_nav=64, d_tac=512)` | **ZERO** |
| `nav_to_str` (E13) | `refc_v3.py:676`, zeroed `:677-679` | `Linear(d_nav=64, d_ctx=256)` | **ZERO** |
| `ego_to_tac` (E11′) | `refc_v3.py:732`, zeroed `:734-736` | `Linear(d_ego=32, d_tac=512)` | **ZERO** |
| `ego_to_str` (E11′) | `refc_v3.py:733`, zeroed `:734-736` | `Linear(d_ego=32, d_ctx=256)` | **ZERO** |
| `tac_goal_head` under `echo_base` (E14) | `refc_v3.py:761`, zeroed `:769-772` | `Linear(d_tac=512, k·GOAL_DIMS=12)` | **ZERO** |
| `ctx_to_cond` | `refc.py:1349-1351` | `Linear(256, 384)` | **ZERO** |
| `lan_to_cond` | `refc.py:1399-1401` | `Linear(64, 384)` | **ZERO** |
| `tgt_film` | `refc.py:1385` (`FiLM(384, 384)`, `zero_init=True` default `refc.py:1181`) | `Linear(384, 768)` | **ZERO** |
| `control_head` (WP-4) | `refc.py:1484`, zeroed `:1485-1486` | `Linear(384, S·2=16)` | **ZERO** |
| `agent_gate` (WP-6) | `refc.py:1221` × 4 layers | `Parameter(zeros(1))` | **ZERO** |

---

## 4. Any existing language port

**There is none. `lan_*` is Lane-Anchored Navigation, not language.**

* `stack/tanitad/data/lan.py:1` — *"LAN — Lane-Anchored Navigation: a dense, leak-guarded
  route/goal signal."* Input: `K=4` route anchors × `4` features
  `[cos bearing, sin bearing, lat_norm, valid]` (`LanConfig`, `refc.py:482-500`).
* **Shape and init:** `lan_enc = Sequential(Linear(16, 64), ReLU, Linear(64, 64), ReLU)`
  (`refc.py:2448-2451`), built only under `cfg.graft_lan`; `lan_to_cond = Linear(64, 384)`,
  **zero-init** (`refc.py:1399-1401`); `lan_gate = Parameter(zeros(1))` (`refc.py:1402`).
* **Is it trained?** Only if the flag is on — and **it is off for every registered arm.**
  `RefCConfig.graft_lan: bool = False` (`refc.py:617`); the trainer's own help at
  `refc_v3_train.py:3553-3555` reads *"supplied-corridor MODEL INPUT — ⛔ NOT part of any
  registered v3 arm (E12); exists for diagnostics only"*. `refc_v3.py:808` lists
  *"lan -> inference (E12; label-only)"* among the model's **refused edges**.
* **What currently drives it:** nothing, on a registered arm. On a diagnostic arm, a supplied
  route corridor built from the ego's own future path — which is why E12 refuses it.

**Absence established by two probes with same-breath controls:**

* Probe 1 — worktree grep for `tanitlang|\bvla\b|language|\bllm\b|text_emb|lang_emb|token_emb`
  across `stack/tanitad/refs/*.py`: **0 hits**. Control in the same command: `lan_` in `refc.py`
  = **43**.
* Probe 2 — `git grep` on `HEAD` for `tanitlang|\bvla\b|nn\.Embedding\(.*vocab|language_model|
  text_tokens` over `stack/tanitad`: **one hit**, and it is not a model port —
  `stack/tanitad/lake/schema.py:247-249`, a data-lake episode-record field
  (`language: dict | None = None`; `language_source: str = "none"`). Control:
  `git grep -c lan_to_cond HEAD -- stack/tanitad/refs/refc.py` = **7**.

⇒ **A reasoner attaching to refcv5 would be the FIRST language-shaped consumer in the model.**
There is no `lan_emb`-style port to reuse, and no tokenizer, embedding table or text head
anywhere under `stack/tanitad/refs/`.

---

## 5. The parameter budget

**Evidence class: MEASURED (ours).** Interpreter `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`
with `PYTHONPATH=<repo>/stack`, verified importing from this repo
(`tanitad.__file__ = G:\…\TanitAD\stack\tanitad\__init__.py`; torch 2.11.0+cu128).
Build recipe (reproducible in ~20 lines): `refc_v3.refc_v4_config(size)`
(`refc_v3.py:533-547`) → replace `cfg.core.encoder` with
`CNNEncoderConfig(image_size=256, image_width=640, base_width=<rung>, blocks=<rung>)` →
`cfg.core.anchors.n_anchors = 117`, `.v0_conditioned = True`, `.control_units = "alat"` →
optionally `cfg.core.decoder.sampler = "ddim"` and
`cfg.core.agents = refc_agents.AgentSeamConfig(enable=True)` + `cfg.core.decoder.cross_agent =
True` → `refc_v3.RefCV3Model(cfg)`; then `refc_v3.param_breakdown_v3(model)`
(`refc_v3.py`) and `refc.param_breakdown(model.core)` (`refc.py`).
Vocabulary `v7.0` (the `RefCV3Config` default, `refc_v3.py:322`).

⚠️ **No checkpoint and no anchor file were needed.** The vocabulary is a **buffer**, not a
parameter (`refc.py:1296`), and every anchor-shaped module is `Linear(k, anchors.shape[0])`, so
resizing the synthetic vocabulary (`default_anchors`, `refc.py:2395-2396`) to 117 gives the
**exact** parameter count of the live arm.

### 5a. `base` rung (`base_width = 88`, `feat_dim = 704`), 256×640 → 160 PV tokens

| module | refcv4b | +WP-4 sampler | +WP-4 +WP-6 agents |
|---|---:|---:|---:|
| `core.encoder` | 90,458,632 | 90,458,632 | 90,458,632 |
| `core.decoder` | 9,207,119 | 10,394,847 | 12,763,363 |
| `core.strategic` (ctx GRU) | 2,002,176 | 2,002,176 | 2,002,176 |
| `core.law` (LAW aux MLP) | 2,919,104 | 2,919,104 | 2,919,104 |
| `core.aux` (tactical trunk + lat/lon + route heads) | 275,145 | 275,145 | 275,145 |
| `core.measurement` | 17,408 | 17,408 | 17,408 |
| `core.agent_head` | — | — | 3,413,269 |
| `core.agent_embed` | — | — | 106,112 |
| `core.imagination` / `lan` / `speed` / `selection` / `goal` | 0 | 0 | 0 |
| **core subtotal** | **104,879,584** | **106,067,312** | **111,955,209** |
| `hier.phi_tac` | 1,757,440 | 1,757,440 | 1,757,440 |
| `hier.tac_latent_proj` | 262,656 | 262,656 | 262,656 |
| `hier.gstr_cond` | 66,816 | 66,816 | 66,816 |
| `hier.nav_inject` | 50,176 | 50,176 | 50,176 |
| `hier.ego_inject` | 25,536 | 25,536 | 25,536 |
| `hier.tac_heads` (lat 4,104 + lon 4,104 + goal 6,156) | 14,364 | 14,364 | 14,364 |
| `hier.scorer` (+ `goal_gate`) | 1,145 | 1,145 | 1,145 |
| `hier.str_goal_head` | 771 | 771 | 771 |
| **TOTAL** | **107,058,488** | **108,246,216** | **114,134,113** |

**Decoder interior (`base`, +sampler +agents), MEASURED:**

| decoder submodule | params |
|---|---:|
| `layers` (4 × `CrossAttnLayer`) | 10,649,092 |
| `feat_proj` `Linear(704, 384)` | 270,720 |
| `tgt_film` `FiLM(384, 384)` | 295,680 |
| `tgt_proj` `Linear(512, 384)` | 196,992 |
| `time_mlp` (WP-4) | 1,181,568 |
| `ctx_to_cond` `Linear(256, 384)` | 98,688 |
| `cond_proj` `Linear(128, 384)` | 49,536 |
| `traj_proj` `Linear(16, 384)` | 6,528 |
| `offset_head` `Linear(384, 16)` | 6,160 |
| `control_head` (WP-4) `Linear(384, 16)` | 6,160 |
| `time_embed` `Embedding(3, 384)` | 1,152 |
| `conf_head` `Linear(384, 1)` | 385 |
| `lat_to_anchor` `Linear(3, 117, bias=False)` | 351 |
| `lon_to_anchor` `Linear(3, 117, bias=False)` | 351 |
| **decoder total** | **12,763,363** |

**One `CrossAttnLayer` (`d = 384`), MEASURED:** `cross` 591,360 · `mlp` 1,181,568 · `film`
295,680 · `norm_q` / `norm_f` / `norm_a` 768 each · `cross_agent` 591,360 · `agent_gate` **1**
⇒ 2,662,273. ⇒ **the WP-6 cost inside the decoder is 4 × (591,360 + 768 + 1) = 2,368,516.**

**WP-6 agent-head interior (`base`, 160 memory tokens), MEASURED:** `mem_proj` 131,328 ·
`blocks` 3,160,320 · `norm` 512 · `head` 5,397 (+ positional table). At 64 memory tokens (square
256) the head is 3,388,693; at 160 it is 3,413,269 — the +24,576 is exactly `96 × 256`, the
per-token positional rows (`build_agent_head`, `refc_agents.py:197-212`, `n_memory = gh·gw`).

### 5b. `small` rung (`base_width = 64`, `feat_dim = 512`), 256×640

| | refcv4b | +WP-4 | +WP-4 +WP-6 |
|---|---:|---:|---:|
| `core.encoder` | 47,862,976 | 47,862,976 | 47,862,976 |
| `core.decoder` | 9,133,391 | 10,321,119 | 12,689,635 |
| `core.strategic` | 1,707,264 | 1,707,264 | 1,707,264 |
| `core.law` | 2,132,480 | 2,132,480 | 2,132,480 |
| `core.aux` | 200,841 | 200,841 | 200,841 |
| `core.measurement` | 17,408 | 17,408 | 17,408 |
| `core.agent_head` / `core.agent_embed` | — | — | 3,364,117 / 106,112 |
| hierarchy cascade (the 8 `hier.*` lines) | 2,129,752 | 2,129,752 | 2,129,752 |
| **TOTAL** | **63,184,112** | **64,371,840** | **70,210,585** |

### 5c. WP-7 selector (`refc_selector.SubMetricSelector`) — built, wired to nothing

MEASURED at `scene_dim=704`, `n_steps=8`:

| arm | total | `embed` | `scene_proj` | `coarse` (1 layer) | `fine` (3 layers) | heads |
|---|---:|---:|---:|---:|---:|---:|
| `SelectorConfig()` (default) | **4,530,444** | 133,120 | 180,480 | 1,053,440 | 3,160,320 | 1,542 + 1,542 |
| `cross_agent=True` | **5,585,164** | 133,120 | 180,480 | 1,317,120 | 3,951,360 | 1,542 + 1,542 |
| `self_attn=False` | **3,475,724** | 133,120 | 180,480 | 789,760 | 2,369,280 | 1,542 + 1,542 |

These three reproduce the module's own docstring table (`refc_selector.py:98-100`) to the digit —
an independent cross-check that this build recipe matches the author's.

### 5d. Vocabulary width — a 5,130-parameter lever not to be confused with a capacity change

MEASURED at `base`: `tac_vocab_version="kin3"` → `lat_head_tac` = `lon_head_tac` = 1,539, total
**107,053,358**; `"v7.0"` → 4,104 each, total **107,058,488**. Selected at `refc_v3.py:749-759`,
pinned by the trainer from `--v7-labels` (`refc_v3_train.py:3228`, applied `:198-199`).

### 5e. The headline for a reasoner's budget

**The whole hierarchy cascade above the trunk is 2,129,752 parameters** (`base`, the eight
`hier.*` lines). The decoder is 9.2–12.8 M. The trunk is 90.5 M. A ≤1 B reasoning module is
**~470×** the cascade it would advise and **~8×** the entire refcv5 model. That is the binding
constraint on the attachment design, and it is `MEASURED`, not argued.

---

## 6. The training entry point

**`stack/scripts/refc_v3_train.py`.** Its docstring (`:1-48`) calls it *"a THIN COMPOSITION over
`refc_train.py`'s measured machinery"*. There is **no refcv5-specific trainer** and **no separate
refcv5 model class** — `stack/tanitad/rl/refc_adapter.py:4-6` states it in source: *"refcv4b and
refcv5 … are not new model classes"*, naming `refc_v3.py:628` (`RefCV3Model`) and
`RefCV3Config:319`.

### 6a. Stages

⛔ **There is no multi-stage curriculum.** Grepping `stage|phase|curriculum` returns only the
label **"STAGE 0"** applied to the feasibility-aware decode (`refc_v3_train.py:310`, `:1844`,
`:3345`) and one optimiser schedule. What exists:

* **one optimisation schedule** — linear warmup then cosine, `refc_v3_train.py:2879-2881`;
  `--warmup` default 2000 (`:3547`), `--steps` default 30000 (`:3260`).
* **one decoder-mode switch** — `--mode {classifier, diffusion}` (`:3261`).
* **one intra-run switch** — the withheld-bank warmup, `--withheld-bank-warmup` (`:3529`),
  applied at `:2294` and `:3029`.
* **one arm switch** — `--arm {hier, flat}`, **required** (`:3183`).
* **`--preflight`** (`:3566`): build + pin the config delta + the C115 freeze-history gate + the
  E11 intervention audit + one synthetic loss step, then exit.

### 6b. The flags that gate the grounding features — verbatim from argparse

All refcv5 flags are in one group, `refc_v3_train.py:3312-3316`:

```
g5 = ap.add_argument_group(
    "refcv5", "WP-4 (control-space sampler) and WP-6 (agent tokens). "
    "EVERY flag here defaults to OFF, and OFF means NOT CONSTRUCTED: a "
    "run that passes none of them is bit-identical to refcv4b.")
```

**Grounding (WP-6):**

```
refc_v3_train.py:3397  g5.add_argument("--agents", default="off",
                                       choices=["off", "head", "oracle"], …)
refc_v3_train.py:3407  g5.add_argument("--w-agent", type=float, default=AGENT_WEIGHT_DEFAULT, …)
refc_v3_train.py:3409  g5.add_argument("--agent-queries", type=int, default=AGENT_QUERIES_DEFAULT, …)
refc_v3_train.py:3422  g5.add_argument("--agent-w-project", type=float, default=0.0, …)
refc_v3_train.py:3429  g5.add_argument("--agent-w-ground", type=float, default=0.0, …)
refc_v3_train.py:3437  g5.add_argument("--agent-rig-camera", default="off", …)
refc_v3_train.py:3446  g5.add_argument("--agent-rig-extrinsics", default=None, …)
refc_v3_train.py:3465  g5.add_argument("--agent-sigma-range", type=float, default=0.0, …)
refc_v3_train.py:3469  g5.add_argument("--agent-miss-rate", type=float, default=0.0, …)
refc_v3_train.py:3471  g5.add_argument("--agent-join", default=None, …)
refc_v3_train.py:3482  g5.add_argument("--agent-pad", type=int, default=0, …)
refc_v3_train.py:3511  g5.add_argument("--agent-presence-hard", action="store_true", …)
```

`--agents`'s help text (`:3398-3406`) fixes the arm semantics: *"'head' = the LEARNED monocular
3D head (the deliverable arm; vision-only at inference, obstacle.offline as TRAIN-TIME labels).
'oracle' = GROUND-TRUTH boxes fed at inference: the CEILING, INADMISSIBLE as a capability claim …
If the oracle does not separate on LONGITUDINAL AND TACTICAL the whole mechanism is refused."*

**Sampler (WP-4):**

```
refc_v3_train.py:3316  --sampler {none,ddim}              default "none"
refc_v3_train.py:3322  --sampler-space {control,metre}    default "control"
                       [metre = the pre-registered DELIBERATE REGRESSION]
refc_v3_train.py:3328  --sampler-train-t-max              default 50
refc_v3_train.py:3330  --sampler-infer-t                  default 8
refc_v3_train.py:3332  --sampler-steps                    default 2
refc_v3_train.py:3337  --sampler-groups                   default 1   [>1 refuses]
refc_v3_train.py:3341  --w-u0                             default U0_WEIGHT_DEFAULT
```

**Feasibility decode ("STAGE 0"):** `--feasible-decode` (`:3364`), `--feasible-mu` (`:3373`),
`--feasible-entry` (`:3376`), `--feasible-a-max` (`:3384`), `--feasible-kappa-max` (`:3389`),
`--feasible-prefix-slots` (`:3391`). Backed by `DecoderConfig.feasible_decode: bool = False`
(`refc.py:414`).

**Anchors:** `--anchors` (`:3263`), `--anchor-v0-conditioned` (`:3270`),
`--anchor-control-units` (`:3281`), `--anchor-ref-speed` (`:3305`).

**Geometry:** `--image-hw` (applied `refc_v3_train.py:200-206`) — this is what turns the 64-token
PV map into the live 160-token one.

**How they reach the model:** `_pin_refcv5_seams(cfg, args)` (`refc_v3_train.py:277`), called from
the config builder at `refc_v3_train.py:273`. It sets `core.decoder.sampler*` (`:286-294`),
refuses `ddim` without a v0-conditioned vocabulary (`:295-301`) and without `--w-u0 > 0`
(`:302-309`), sets the feasibility fields (`:316-324`), and for `--agents != off` builds
`AgentSeamConfig(...)` (`:354`) and sets `core.decoder.cross_agent = True` (`:365`), refusing
`head` without `--w-agent > 0` (`:371-378`) and without `--agent-join` (`:379-392`).

**The guard that makes the record honest:** `assert_seams_are_built(model, stamp)`
(`refc_v3_train.py:2064`) checks the **built modules** against the stamp, bidirectionally —
stamped-but-not-built AND built-but-not-stamped. ⛔ **Not in `HEAD`** (§0).

### 6c. Zero-GPU instruments that already exist

* `stack/scripts/refcv5_preflight.py` — *"can refcv5 start a REAL arm on the parity corpus?"*
  (`:1-40`). Every check is a positive assertion; an unreadable subject is **INCONCLUSIVE and
  counted as a FAILURE** (`:63-68`). Its import probe (`:97-101`) is the only consumer of
  `refc_selector`.
* `stack/scripts/refcv5_validate.py` — runs the `V-RC5-READY` tiny-rig arms A0 / A0r / A1, the
  four deliberate regressions R1–R4, and the converse control C1 (`:1-45`).

---

## 7. Attachment points for a reasoner, ranked by how little existing code they disturb

Ranked by **disturbance at initialisation** — the property that decides whether a TanitLang arm is
comparable to the base arm at all. A module that changes behaviour at step 0 turns the
base/attached contrast into a capacity confound instead of a lever.

### Rank 1 — `hierarchy_hook`: a second supplier on an existing gated seam. **Zero new wiring.**

* **Reads:** `hook(pooled_seq [B, W=8, feat_dim], ctx [B, 256])` — called at `refc.py:2769`,
  inside the block at `refc.py:2762-2782`. `W = cfg.window = 8` (`refc.py:565`).
* **Writes:** the returned dict's `"maneuver_logits"`, `"target_latent"`, `"bank_speed_pred"` —
  **only where the caller passed `None`** (`refc.py:2770-2781`).
* **Why Rank 1:** the gate exists and is already tested — `refc.py:2759-2761`: *"with hook=None
  this block is untouched dead code and the forward is byte-identical to the pre-hook file —
  pinned by tests/test_refc_v3.py"*. It also names the latency argument in advance
  (encoder ≈ 90 % of the tick), i.e. **the reasoner must consume REF-C's trunk tokens and must
  not run a second vision encoder.**
* ⛔ **Two corrections to `REFCV5_DESIGN_PLAN.md` §9.1/§9.3 before anyone codes to it:**
  1. `maneuver_logits` is **NOT** a live external port on a registered arm. `refc_v3.py:929-930`
     supplies it only when `tac_vocab_version == "kin3"`; under the config default `"v7.0"`
     (`refc_v3.py:322`) the hook returns **no** `maneuver_logits` and `refc.py:2875` falls back to
     the core's own head. And even then it would land on `maneuver_to_anchor`, which
     `factored_maneuver=True` never builds (§3c).
  2. `target_latent` is **already filled** by `refc_v3.py:969`. A reasoner writing here must
     **replace or blend**, and must say which — it is not an empty port.
* **Zero-init port to reuse:** `tgt_film` (`refc.py:1385`, `FiLM(384, 384)` zero-init) is the
  landing surface, so a blend of the form
  `target_latent = tac_latent_proj(z_up) + gate * lang_proj(h_lang)` with `gate` a new zero-init
  `nn.Parameter` is a **strict no-op at init**.

### Rank 2 — a new zero-init gate on the RANKED score, beside `route_to_anchor` / `goal_gate`

* **Reads:** whatever the reasoner has; the fan is `anchor_traj [B, N, 8, 2]` + `anchor_bank` in
  the decoder's return (`refc.py:2243-2246`).
* **Writes:** one `[B, N]` additive log-prior appended to `r_terms` (`refc.py:2160-2181`) before
  `_apply_grafts` (`refc.py:2182-2183`).
* **Why Rank 2:** that list is already an open sum of independently gated terms, three of which
  are single zero-init scalars (`goal_gate`, `goal_dist_gate`, `cons_gate`). A fourth is ~5 lines
  in `__init__` + 2 in `forward`, individually ablatable, and it inherits the saturation clamp
  for free.
* **Zero-init port to reuse:** structurally identical to `route_to_anchor`
  (`refc.py:1415-1417`) — `nn.Linear(K, N, bias=False)` + `nn.init.zeros_`.
* ⚠️ **It does not decide the emitted trajectory on a hier arm.** `RefCV3Model.forward` re-ranks
  after the core returns (`refc_v3.py:1096-1108`). A prior added at `refc.py:2160` enters
  `out["sel_score"]`, which v3 then **blends** with its own graft — the effect survives, but the
  arm must be read on `sel_idx`, never `sel_idx_base`.

### Rank 3 — `RefCV3Model`'s own `goal_gate`, the v3 selection graft

* **Reads:** `cache["z_tac"] [B, 512]`, `cache["g_tac"] [B, 3, 4]`, the emitted fan
  (`refc_v3.py:1063-1075`).
* **Writes:** `graft = self.goal_gate * sc["score"]` (`refc_v3.py:1075`) → blend
  (`refc_v3.py:1096`) → `argmax` (`refc_v3.py:1104`).
* **Why Rank 3:** this IS the surface that picks the emitted trajectory, and its gate is already a
  zero-init scalar (`refc_v3.py:782`), so a second parallel scorer changes nothing at init.
  Cost: it lives in `refc_v3.py`, the file the live 40,284-step run resumes through — a change
  here needs the `test_v3_parity` bit-identity check (`refc_v3.py:371-374`).

### Rank 4 — a zero-init residual on `tac_goal_head`'s output (the geometric goal port)

* **Reads:** `z_tac [B, 512]` plus whatever the reasoner produces.
* **Writes:** `g_tac [B, 3, 4]` at `refc_v3.py:931-953` — layout `(x, y, heading, speed)` per τ,
  τ ∈ `(20, 40, 60)` decisteps = 2 / 4 / 6 s (`GOAL_TAU_STEPS`, `refc_v3.py:128`;
  `GOAL_DIMS = 4`, `refc_v3.py:130` and `stack/tanitad/models/tactical.py:78`).
* **Zero-init port to reuse:** under `echo_base` (`refc_v3.py:388`; `refc_v4_config`'s default
  `echo_base=True`, `refc_v3.py:533-534`) `tac_goal_head` is **already zeroed**
  (`refc_v3.py:769-772`) and `g_tac = echo_base + g_delta` (`refc_v3.py:953`). A language
  residual as a third summand with its own zero-init projection is a no-op at init.
* ⭐ **This is the only write port where the ANTI-ECHO INSTRUMENT ALREADY EXISTS.**
  `echo_base_absmean` / `g_tac_delta_absmean` are emitted every step (`refc_v3.py:964-965`) and
  surfaced as `out["echo_ratio"]` (`refc_v3.py:1086`), so *"is the reasoner echoing the kinematic
  extrapolation?"* is readable off the training log from day one with no new instrument. That is a
  strong argument for making it the FIRST measured attachment.

### Rank 5 — the agent-token bus, as a second token supplier

* **Reads:** `agent_slots` (`refc.py:2953-2958`) → `AgentTokenEmbed` (`refc.py:2959`) →
  `[B, 100, 384]`.
* **Writes:** concatenating reasoner tokens onto `agent_tok` before `refc.py:2969` puts them into
  `cross_agent` in every layer through the **already zero-init** `agent_gate` (`refc.py:1221`).
* **Why Rank 5, not higher:** it needs **no new parameter at all** if the tokens are projected to
  384 and appended — but it then shares one gate with the agent seam, so the two become
  non-attributable. The attributable form is a separate `lang_gate` + `cross_lang` per layer:
  ~15 lines in `CrossAttnLayer` plus a fourth argument threaded through `_decode` /
  `_decode_ctrl` / `forward`. ⛔ It also **changes the RNG draw order** for every build, which
  breaks strict checkpoint loading — the property `refc.py:1206-1213` and `refc.py:1457-1459` are
  written to preserve.

### Rank 6 — `lan_to_cond`, the unused zero-init condition port

* **Reads:** a `[B, 64]` vector. **Writes:** `cond` (`refc.py:1992`), zero-init.
* **Why last:** it is the cleanest empty port in the model — but it is empty *because E12 refuses
  it*, and reusing a refused edge's plumbing for a new signal makes every future reader ask
  whether the label pathway leaked. `graft_lan` also builds `lan_enc` and `lan_gate` together
  (`refc.py:1398-1402`, `refc.py:2448-2451`), so switching it on for a language port switches on a
  route-corridor input as well. **Only usable behind a new flag that builds `lan_to_cond` without
  `lan_enc` / `lan_gate`.**

### Ports a reasoner must NOT use — from source, not from doctrine memory

* ⛔ **`ctx` is not a vision-only read.** `ctx = ctx + nav_s` inside the hook
  (`refc_v3.py:861`), and under E11′ also `ctx = ctx + self.ego_to_str(ego_e)`
  (`refc_v3.py:882`). A module reading `ctx` and claiming vision-only is wrong about its own
  inputs. `pooled_seq` is the vision-only read.
* ⛔ **No ego state beyond the measured `v0` at t0.** `EGO_DIMS = 5` = `(v0, a_long, yaw_rate,
  curvature, keep)` at the **last observed** frame (`refc_v3.py:165`), and the withholding draw
  has **one owner** (`refc_v3.py:1037-1041`). A reasoner reading `m` or `ego_state` must honour
  the same `keep` vector, or it is a second unsynchronised dropout.
* ⛔ **The refused edges, enumerated in source** at `refc_v3.py:808-815`:
  `lan -> inference (E12; label-only)`; `situation classifier output -> any goal node
  (PI 2026-08-03, UNCHANGED)`; and on a v4/v5 build,
  `future_poses/future_actions -> any goal node`.
* ⛔ **The oracle agent path** (`refc.py:2944-2956`) — a privileged label at inference,
  inadmissible as a capability claim.

---

## Manifest

| artifact | where it lives | only place? |
|---|---|---|
| `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-vla-design-dialogue/AUDIT_REFCV5_SURFACE.md` | `repo:` — staged, **not committed** | no |
| the parameter-count script | scratchpad (`…/ff19d6ac-…/scratchpad/count_refcv5.py`) | ⚠️ **yes** — deliberately not banked: it is a 60-line throwaway and §5's preamble reproduces it exactly. Every number it produced is in §5, and the WP-7 rows independently reproduce `refc_selector.py:98-100` to the digit. |

**Escalations — integration the Master Mind must schedule, not read about:**

1. ⛔ **The refcv5 model seams are uncommitted.** `stack/tanitad/refs/refc.py`,
   `stack/scripts/refc_v3_train.py`, `stack/tanitad/refs/refc_sampler.py` and
   `stack/scripts/refcv5_preflight.py` are `MM` / `M`, and `HEAD`'s `refc.py` contains **zero**
   occurrences of `sampler`, `control_head`, `cross_agent`, `time_mlp` and `agent_tok` (controls
   read 7 and 4 in the same file). A fresh clone of `HEAD` cannot build a refcv5 model, and
   `HEAD`'s trainer would stamp a sampler onto a model that has none — the exact FALSE-PROVENANCE
   defect its own (uncommitted) guard names.
2. ⛔ **`refc_selector.py` (4,530,444 params, WP-7) is committed and wired to nothing** — two
   probes, one hit outside the module, and that hit is an import check.
3. ⚠️ **`REFCV5_DESIGN_PLAN.md` §9 needs three corrections** before anyone codes against it:
   (a) all its `refc.py` line numbers are stale; (b) `maneuver_to_anchor` is **not built** on any
   `factored_maneuver=True` arm, so it is not *"the only port through which an external module
   already changes behaviour with zero new wiring"*; (c) `target_latent` is already filled by
   `refc_v3.py:969` — a VLA arm must replace or blend, and say which.
4. ⚠️ **The 117-anchor grid shape (`13 × 9`) is UNVERIFIED from source** in this pass — the
   artifact is pod-side and absent from the repo. `n_anchors` is a free run-time value
   (`refc_v3_train.py:241-245`); the parameter counts in §5 are exact regardless (the vocabulary
   is a buffer, not a parameter).
5. ⚠️ **`refc.py` and `refc_v3_train.py` were edited by siblings DURING this audit** (+13 and
   ~+110 lines respectively). Citations are pinned to the blobs in the header table; re-derive by
   symbol name if a blob differs.
