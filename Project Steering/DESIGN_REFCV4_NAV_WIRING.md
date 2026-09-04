# DESIGN — REF-C v4: nav wiring, the strategic horizon, and selector conditioning

*Arch+Inference FlyWheel, 2026-09-04. **DESIGN + AUDIT.** ⛔ Nothing here was trained. This gates
refcv4's launch on the free A40; it does not license one.*

**PI mandate (verbatim):**
> *"our v7.2 label and dataset include nav commands which must be fed to all three layers (review
> this). The task of strategic layer is to anticipate important routing manoeuvre within its time
> frame (up to 30 sec). In many times this corresponds to the over-next manoeuvre; these strategic
> goals are conditioning the tactical layer."*
> *"the selector is not benefiting from the nav command information and tactical/strategic goals.
> The model must develop the intelligence to always follow nav command and depending on the time
> constraints of the strategic goals to follow those; we need to check that the model is correctly
> wired."*

---

## 0. Six findings, in the order they change the plan

1. **The core's two decision heads are strictly `f(pooled)` — vision only.** Nav reaches neither.
   `route_head`'s output additionally reaches **nothing downstream** (`graft_route=False`): it is a
   dangling aux.
2. **The cascade's nav edges ALREADY EXIST and are correct** (E13, 2026-09-02). The defect is that
   **they are never exercised** — the live run feeds the *v1* nav derivation, which is `follow` on
   **94.6 %** of B1 windows. `--nav-from-v7` is **not** on the live launch line.
3. **⭐ The mechanism behind the 94.6 %:** `nav_command` needs `NAV_MIN_STEPS = 150` future poses
   (**15 s**) and reads to `NAV_HORIZON_STEPS = 250` (**25 s**) — `refb_labels.py:72-73`. Our
   `*.v2ep` episodes are **199 frames = 19.9 s**, so only windows with `t0 ≤ 4.9 s` (**28.5 %**) can
   *ever* emit a left/right; on a 74 %-straight corpus that lands at ≈ 5 % non-follow. **The nav
   channel is dead for a corpus reason, not a wiring reason.**
4. **⛔ Feeding nav into today's strategic head is a near-echo, and E20 is its precondition.**
   `nav_cmd` (v1) and `route_target` (v2.1) are **the same function of the same future poses** —
   `nav_input_v22` → `nav_command_v21_ex` → `route_from_future_v21` (`refb_labels.py:1437`), and
   `RouteV21Dataset.__getitem__` → `route_from_future_v21` (`refc_train.py:150-152`) — differing only
   in horizon. Under the **v7.2** nav source the producer is different
   (`s2_geom_emit_v7.nav_command()` over the manoeuvre sequence), so **E20 unlocks E15's
   admissibility**. This is a dependency, not a preference.
5. **⭐⭐ A 30 s strategic label IS constructible — I was wrong to doubt it, and the reason is
   worth stating.** `s2_geom_emit_v7.py:52,59` already emit `STRATEGIC_S = (8.0, 30.0)` and
   `LOOKAHEAD_S = 30.0`, and **1,882 train manoeuvres genuinely start in [8, 30] s**
   (`band_census.json`). The 30 s reach exists because the emitter reads the **RAW** clip
   (`recording_span_s` mean **139.7 s**, max 141.3), while the `*.v2ep` cache the trainer sees is a
   **19.9 s excerpt**. ⇒ the *label* reaches 30 s; the *window's NOW* is confined to
   `[0.7, 17.8] s`. **The horizon is not the constraint. Event DENSITY is.**
6. **The selection surface is NOT flat, and the reachability clamp is not the culprit.** MEASURED on
   the arm: **50 of 128** anchors are ever selected, entropy **2.8425 / 4.8520 nats (ratio 0.586)**,
   modal anchor **#57 at 14.8 %**, and `goal_gate` has **OPENED to 0.1744** (it is not stuck at
   zero). What is missing is not structure — it is any term that expresses *"this candidate executes
   the commanded manoeuvre."* `r_terms` is **empty** in the live config.

⭐ **The number that bounds this whole exercise.** MEASURED, `refcv3-40284-openloop.ARM.json`,
paired episode-cluster bootstrap, 4,823 win / 141 eps: oracle **selection** is worth **−0.0751 m
[−0.0884, −0.0618]** (17.0 % of deployed ADE 0.4419) and oracle **nav** only **−0.0239 m**. Together
**0.099 m** — still short of the **0.1423 m** by which the trivial **hold-action** control beats the
model. ⇒ **Perfect nav plus perfect selection does not reach a trivial baseline.** This wiring work
is necessary and provably not sufficient; the report must not imply otherwise.

---

## 1. The input × consumer matrix

**Evidence class: MEASURED (source read, 2026-09-04).** ⚠️ **Both files moved DURING this audit**
(`refc.py` 2,299 → **2,477** lines; `refc_v3.py` 1,228 → **1,270** lines, 70,468 → 73,424 B) — another
stream is editing them. **Every line number below was therefore RE-PINNED BY CONTENT against the
current files**, not offset-shifted, and every anchor is quoted verbatim beside it so it can be
re-found if the files move again. Snapshot: `refc.py` 2,477 lines · `refc_v3.py` 1,270 lines,
73,424 B. See §7.1.
Configuration = the **live 40,284-step arm**: `refc_v3_sized_config("base", hier=True)` +
`_pin_trainer_cfg`, launch line verified in **two** files (`refcv3_b1_launch.sh:126-142`,
`sup_refcv3.sh:62-76`).

### 1.1 The six consumers

| # | consumer | module (built at) | called at |
|---|---|---|---|
| C1 | **CORE strategic aux** `route_logits` | `route_head = nn.Linear(feat, 3)` `refc.py:1955` | `refc.py:2278` |
| C2 | **CORE tactical aux** `lat/lon_logits` | `tactical_trunk`/`lat_head`/`lon_head` `refc.py:1934-1937` | `refc.py:2287-2293` |
| C3 | **CASCADE strategic** `g_str` | `str_goal_head = nn.Linear(d_ctx=256, 3)` `refc_v3.py:655` | `refc_v3.py:883` |
| C4 | **CASCADE tactical** `z_tac`, `lat/lon_logits_tac`, `g_tac` | `phi_tac`/`lat_head_tac`/`lon_head_tac`/`tac_goal_head` `refc_v3.py:652,758-761` | `refc_v3.py:842,902-907` |
| C5 | **OPERATIVE decoder** `traj` | `AnchoredDiffusionDecoder` `refc.py:1470` | `refc.py:2364` |
| C6 | **SELECTOR** `sel_score → sel_score_v3 → argmax` | `refc.py:1639-1690`; `refc_v3.py:1058-1073` | `refc.py:1690`, `refc_v3.py:1066` |

### 1.2 The matrix

`✅` live · `⚠️` edge exists but **zero-init** / conditional · `❌` **no path, structural**

| input | C1 core route | C2 core tac | C3 casc strategic | C4 casc tactical | C5 decoder | C6 selector |
|---|---|---|---|---|---|---|
| `pooled` (vision @ t0) | ✅ `refc.py:2278` | ✅ `refc.py:2287-2289` | ❌ | ❌ | ✅ `fmap`→`kv` `refc.py:1502` | ✅ (via `conf0`) |
| `pooled_seq` (vision window) | ❌ | ❌ | ✅ `refc.py:2190`→`refc_v3.py:883` | ✅ `refc_v3.py:842` | ✅ `ctx_to_cond` `refc.py:1504-1505` | ✅ |
| **`nav` one-hot** | ❌ **NEVER** | ❌ **NEVER** | ⚠️ `nav_to_str` **zero-init** `refc_v3.py:861` (built `:674-679`) | ⚠️ `nav_to_tac` **zero-init** `refc_v3.py:860` (built `:674-679`) | ✅ via `m` `refc.py:2273-2275`→`cond_proj` `:1503` | ⚠️ only diluted through `m`→`conf0`; and through `man5` |
| `nav_known` bit (E1) | ❌ | ❌ | ❌ | ❌ | ⚠️ `refc.py:2273-2274` — **`nav_known_channel=False`** ⇒ dead | ❌ |
| `v0` / `v` @ t0 | ❌ | ❌ **`tactical_speed_input=False`** `refc_v3.py:447` | ❌ (E11) | ❌ v3 / ⚠️ v4 `ego_to_tac` zero-init `refc_v3.py:881` | ✅ via `m` | ✅ via `m`; **and** `v_ms` decides the reach mask `refc.py:2345`, `:1676-1685` |
| `m` (**the only carrier of nav**) | ❌ | ❌ | ❌ | ❌ | ✅ `refc.py:2364`, `:1503` | ✅ (via `conf0` only) |
| `ctx` | ❌ | ❌ | — | ❌ | ✅ `refc.py:1504-1505` — **the un-nav'd `ctx`** (§1.3b) | ✅ |
| `g_str` | ❌ | ❌ | — | ✅ FiLM **zero-init** `refc_v3.py:900-902` | ⚠️ only via `target_latent` `refc_v3.py:969`→`tgt_film` `refc.py:1508` | ⚠️ same chain |
| `g_tac` | ❌ | ❌ | ❌ | — | ❌ | ⚠️ `goal_gate·scorer` `refc_v3.py:1026-1037` — gate **has opened, 0.1744** |
| `z_tac` | ❌ | ❌ | ❌ | — | ✅ via `target_latent` | ✅ scorer input `refc_v3.py:1027` + H19 prior via `man5` |
| `route_logits` (C1 out) | — | ❌ | ❌ | ❌ | ❌ **`graft_route=False`** `refc.py:2341-2342` | ❌ **`graft_route=False`** `refc.py:1643-1644` |
| `lan` corridor | ❌ | ❌ | label-only `refc_v3_train.py:658-663` | ❌ | ❌ `graft_lan=False` | ❌ |

### 1.3 The two cells that ARE the finding

**(a) `m` carries nav to the DECODER and to nothing else.**
`meas_in = ([v, nav] + …)` → `m = self.measurement(...)` (`refc.py:2273-2275`);
`dec = self.decoder(fmap, m, …)` (`refc.py:2364`) → `cond = self.cond_proj(m)` (`refc.py:1503`).
The operative layer sees the route command — as ≤ 4 one-hot dims inside a 5-wide vector, MLP'd and
summed into a `d = 384` condition otherwise dominated by cross-attention over vision tokens. It
reaches the selector only through that condition. **Nav is present in the operative path and absent
from both decision heads** — the exact inversion of the hierarchy's thesis.

**(b) The nav-augmented `ctx` never leaves the hook.**
`refc_v3.py:861` is `ctx = ctx + nav_s` **inside `hook()`** — a rebind of the closure's local
parameter. `refc.py:2205` calls `hk = hierarchy_hook(pooled_seq, ctx)`, and the hook returns only
`{"maneuver_logits", "target_latent"}` (`refc_v3.py:968-969`). The `ctx` the decoder receives at
`refc.py:2364` is still the one built at `refc.py:2190` — **vision only**. ⇒ E13's strategic nav
injection conditions `str_goal_head` and *nothing else*.

### 1.4 The corroborating measurement — and why the standing claim is vacuous

**MEASURED**, `taniteval/results/refcv3-40284-openloop.ARM.json` →
`.refcv3.strategic.paired_true_minus_shuffled_accuracy`:

```
delta 0.0 · lo 0.0 · hi 0.0 · separated false · reducer mean
n_windows 3622 · n_episodes 128 · n_boot 2000 · estimator paired_episode_cluster_bootstrap
```

The shuffle **changed the token on 2,406 / 4,823 windows (49.89 %)** and accuracy read **0.7667
identically** under true / shuffled / zero; `nav_echo_index` = **0.1651** under all three. On the
1,736-window *changed* subset the head follows the **LABEL** 0.7362 [0.6729, 0.8001] and the
**SHUFFLED NAV** 0.2344 [0.2001, 0.2720].

⇒ The head is **nav-INSENSITIVE by construction** — it is `nn.Linear(feat, 3)(pooled)` and cannot
echo an input it never receives. **"The route head is not a nav echo" is therefore vacuous and must
be re-labelled wherever it is quoted**: it is a *pin on the wiring*, not a finding about the model.
(C109: a probe that cannot fire proves nothing.)

### 1.5 What the live run actually trains — three facts, each verified in two files

Flag list is exactly `--arm hier --size base --v2-cache --v7-labels --eval-cache --eval-labels
--eval-every --eval-batches --image-hw --steps --batch --workers --v2-lru --lr --warmup --seed
--log-every --save-every --out`.

1. **`--nav-from-v7` OFF.** Nav is the v1 derivation: `follow` (+`nav_valid` False) on **94.6 %** of
   B1 windows, agreeing with the v7.2 token on **65.5 %** — MEASURED, `refc_v3_train.py:1127-1128`.
   A zero-init edge fed a 94.6 %-constant channel has nothing to learn from. **This alone explains a
   dead E13.**
2. **`--goal-str` / `--graft-lan` OFF ⇒ `str_goal_head` has NO DIRECT LOSS.** `loss_gstr` fires only
   `if lan is not None` (`refc_v3_train.py:658`); `lan` exists only under those flags
   (`:1084-1085`). The strategic goal head trains **solely** by backprop through the tactical FiLM
   (`uplink_grad=True`, `refc_v3.py:899`). ⇒ *the strategic layer is an unsupervised bottleneck with
   a dangling aux beside it.*
3. **`--ego-state-inject` / `--echo-base` OFF** ⇒ the live arm is **v3**, not v4. E11'/E14 are built
   and inert.

⚠️ **`nav_known` plumbing is complete end-to-end and no trainer uses it.** Exactly **one** call site
in the repo passes `nav_known=` (`stack/experiments/alpasim-gsplat/closedloop_drive.py:367`), and
`nav_known_channel` defaults False on every v3 build. That matters: `_ROUTE_TO_NAV` has no
`ROUTE_UNKNOWN` key, so road-following **and** UNKNOWN both arrive as `nav = 0 = follow` — MEASURED
on 400 PhysicalAI episodes, **1,985 of 3,179 `follow` windows (62.4 %) are a collapsed UNKNOWN
sentinel** (`refb_labels.py:1367-1377`).

---

## 2. The edge-list delta for refcv4

Numbering continues the file's own (E1…E14 taken). Every edge is **additive and zero-init** unless
a shape change is stated, and every edge gets **its own gate** so it is individually ablatable.

### 2.1 ⛔ What is NOT in the delta

| refused | reason |
|---|---|
| `nav → route_head` **while the nav source is v1** | `nav_cmd` and `route_target` are the same function of the same future poses at different horizons (§0.4). Grafting nav is then an echo — the C6 confound `refc.py:2334-2337` already refuses. **Unblocked by E20**, not by argument. |
| `situation classifier output → any goal node` | PI 2026-08-03, **UNCHANGED**. `provenance_roles()` (`refc_v3.py:786-833`) declares `situation_output: []` and the audit *measures* it. §3's heads add one **pinned negative edge**: `situation_logits → {g_str, p_man, t_bin}` REFUSED, checked interventionally (bit-identity), never asserted in prose. |
| `future_poses / future_actions → any goal node` **at inference** | E11', **UNCHANGED**. §3's 30 s label is a **training target**, never an input. |
| `lan → inference` | E12, **UNCHANGED**. ⚠️ Note LAN S1 (`lan_from_future_path`, `lan.py:306-323`, provenance `"ego_future"`) is an ego-future oracle in the same sense as `nav_command`, and it does **not** route through `oracle_nav`'s manifest stamp — the v7.2 guard does not cover LAN. |
| moving `V3_HORIZONS` to reach 30 s | `MAX_H_EXT = max(V3_HORIZONS)` (`refc_v3_train.py:110`) and `n_steps = len(horizons)` sizes the decoder. Extending it changes the trajectory head and voids every registered comparison — the derived-constant-out-of-scope error (DE-C152 class). The strategic lookahead gets its **own** field (§3.4). |

### 2.2 The delta — seven edges

| id | edge | mechanism | params | init |
|---|---|---|---|---|
| **E20** | **nav SOURCE := the v7.2 token** | turn `--nav-from-v7` ON — already implemented, `refc_v3_train.py:266-326,364-372`, position-pinned by `assert_nav_token_alignment()` `:190-217` | **0** | — |
| **E21** | **`nav_known` ON** | `cfg.core.nav_known_channel = True`; trainer passes `nav_known` from `refb_labels.nav_input_v22` (`refb_labels.py:1443-1471`) or the v7.2 join | **+1·hidden** in `measurement` | — |
| **E15** | `nav ⊕ nav_known → CORE tactical head` | **additive, ckpt-safe form:** `nav_to_tacaux = nn.Linear(len(NAV)+1, aux_hidden)`, added to `tactical_trunk`'s pre-activation. *(Widening `d_tac` instead would force a fresh `tactical_trunk` — stated, not hidden.)* | **+1,920** | **zero** |
| **E16** | `strategic urgency → tactical FiLM` | `gstr_embed` input 3 → **13** = `[g_geo(3), p_man(4), p_man2(4), u, t_expect/τ]` | **+640** | FiLM stays **zero** |
| **E17 / S7** | `nav → SELECTOR` | `nav_gate · _lan_anchor_prior(nav_dir)` — the **existing param-free** geometric compatibility (`refc.py:1383`), `valid = nav_known` | **+1** | **zero** |
| **E18 / S8** | `g_str → SELECTOR, scaled by urgency` | `gstr_gate · u[:,None] · _lan_anchor_prior(g_str_dir)` | **+1** | **zero** |
| **E19** | **the 30 s strategic heads** (§3) | `str_goal_head: Linear(d_ctx, 3)` → `Linear(d_ctx, 3+4+K+4+K)`; at `K=6` that is 3→21 | **+5,120** | new rows **zero**; the first 3 rows keep their v3 weights ⇒ **partial ckpt load is exact** |

**Total ≈ 7,682 new parameters** on 106,847,621 = **+0.0072 %**. Deliberate: the capacity control
that caught a +272,001-param tactical head (`refc.py:1928-1934`) applies here, and a *wiring* claim
must not be confoundable with a *capacity* claim.

⚠️ **E20 is the highest-value item in the table and costs nothing.** Every zero-init nav edge in the
programme currently trains against a channel that is `follow` 94.6 % of the time. **Wiring more nav
edges without E20 adds five more dead gates.** If only one item ships, ship E20.

⚠️ **E20 does not fully repair the channel.** The v7.2 `nav_command` is a **per-CLIP constant**
(`refc_v3_train.py:289-298, 364-372`; `t0_constant` semantics), so it carries *what*, never *when* —
every window of a clip receives the same token. That is precisely why §3's **timing** head is the
pre-registered primary: nav cannot echo it.

### 2.3 Oracle degradation — binding on every nav edge

⛔ **Our nav is ORACLE-DERIVED and will not exist at deployment.** `v7_labels.oracle_nav` is gated by
`allow_oracle_nav=True` on the manifest (`refc_v3_train.py:283-291`);
`_provenance.nav_command` reads *"ORACLE (ego-future) — training input only"*
(`s2_geom_emit_v7.py:982`). It is a **label read as an input**, and it is present on only
**4,190 / 4,719 = 88.8 %** of records.

Three mechanisms, all mandatory:

1. **Per-sample nav dropout, `p = 0.3`, ONE draw shared by every nav consumer** — the
   one-draw-one-owner rule `refc.py:2233-2242` established for ego. Under dropout the one-hot goes
   to zeros **and `nav_known` goes to 0**: the X15 rule is that withheld and "genuinely follow"
   differ in the **flag**, not the value. ⇒ **E21 is a precondition of E15/E17, not an option
   beside them.**
2. **`os_navzero` is a first-class arm.** The deployable claim is the number with nav withheld on
   every window. `nav_on` is the ceiling. A registry row quoting only `nav_on` is incomplete. *(The
   harness already emits `os_navzero` — `refcv3_arm.py:1030`.)*
3. **A NAV-SHUFFLE control on every nav-conditioned reading.** Nav is constant on ~75-79 % of
   windows, so a nav-conditioned win can be a marginal-frequency win. Already obligatory per
   `refc_v3.py:852-855`; it becomes binding for every number in §3-§4.

---

## 3. The strategic horizon — 30 s, and its honest provenance

### 3.1 ⭐⭐ The 30 s label already exists in the v7.2 records

**MEASURED, `stack/scripts/s2_geom_emit_v7.py`:**

| constant | value | line |
|---|---|---|
| `OPERATIVE_S` | `(0.0, 2.0)` | `:50` |
| `TACTICAL_S` | `(2.0, 6.0)` — *"2-6 s, NOT 0-6 s"* | `:51` |
| **`STRATEGIC_S`** | **`(8.0, 30.0)`** — *"nothing before 8 s may be strategic"* | **`:52`** |
| `GAP_S` | `(6.0, 8.0)` — belongs to **no layer**; → `bands.unassigned_manoeuvres` | `:57` |
| **`LOOKAHEAD_S`** | **30.0**; loader `max_s = t0 + LOOKAHEAD_S + 5.0 = 43 s` | **`:59`, `:864`** |

Band occupancy, MEASURED (`…/2026-08-30-v72-copy-adjudication/raw/band_census.json`, train scope):
**1,882** manoeuvres start in `[8, 30)`, 463 in `[2, 6)`, 176 in the `[6, 8)` gap, 687 in `[0, 2)`,
and **256 start beyond 30 s**.

**Why the 30 s reach exists at all:** the emitter reads the **RAW** clip
(`horizon.recording_span_s` min/mean/max = 20.2 / **139.7** / 141.3 s), while the `*.v2ep` cache the
trainer consumes is a **19.9 s excerpt** (`frames_u8 [199, 9, 256, 256]`, CLAUDE.md DE-C152;
independently `170.95 windows/ep` at `window=8, max_horizon=20` ⇒ `T = 197.95 ≈ 198`,
`refcv3_b1_launch.sh:113-116` — **two probes, different path-binding, agree to one frame**).

⇒ **The label reaches 30 s. The window's NOW does not** — it is confined to
`t_now = (t+7)·0.1 ∈ [0.7, 17.8] s` for `t ∈ [0, 171]`. **The horizon is not the constraint.**

### 3.2 What IS the constraint: event density, and the censoring boundary

**MEASURED on the label blob** (`s2_labels_v7.2_{train,eval}.jsonl.gz`, 4,572 + 147 records, one per
clip, `t0_s ≡ 8.0` on **all 4,719**):

| quantity | value |
|---|---|
| events/clip (`manoeuvre_sequence`) | 0→**2,389** · 1→1,327 · 2→547 · 3→216 · 4→73 · 5→17 · 6→3 (3,464 total, mean 0.758) |
| clips with **≥ 1** event | 2,183 = **47.75 %** |
| clips with **≥ 2** events (over-next possible) | 856 = **18.72 %** |
| consecutive-event gap | min 1.7 · p25 5.1 · **median 8.9 s** · p75 14.7 · p90 21.2 · max 32.9 |
| event `t_start_s` | min 0.0 · median 12.5 · p90 29.0 · **max 33.7** |
| windows with ≥ 1 future event (NEXT) | **≈ 37.0 %** |
| windows with ≥ 2 future events (**OVER-NEXT**) | **≈ 12.4 %** |

⚠️ **`manoeuvre_sequence` is a geometry-derived turn list, not the tactical label.** All 4,572 clips
carry a tactical label while 2,389 have an empty sequence. **"0 events" ≠ "unlabelled".**

**The censoring boundary, DERIVED:** the emitter read to `t0 + 30 + 5 = 38 s`, so a window asking
about `[NOW, NOW+30]` is fully observed only when `NOW ≤ 8 s` ⇒ `t ≤ 73` ⇒ **74 / 172 = 43.0 %** of
windows. The remainder are **right-censored** at `38 − NOW` seconds. Censoring must be *modelled*:
a censored window is **not** a negative. ⚠️ Treating it as one biases the **point estimate**, not
merely the interval — the same class as the `heldout`-vs-`full_set` bias.

**⇒ The over-next manoeuvre the PI names is constructible, and it is THIN: 18.72 % of clips,
≈ 12.4 % of windows.** The median 8.9 s gap sits inside `[8, 30)`, so consecutive events are cleanly
separable in time. **The binding constraint is event density, not temporal resolution.**

### 3.3 The three label sources, honestly

| source | reach | provenance | verdict |
|---|---|---|---|
| **(a) v7.2 `manoeuvre_sequence` / ego future** | **30 s** (censored beyond 38 s absolute) | ⛔ **ORACLE** — `_provenance`: *"ORACLE (ego-future) — training input only"* | **Available today. Use it.** Stamped, never counted as a deployment capability. |
| **(b) LAN corridor** | arc-length `(20, 40, 80, 160) m`; at the corpus mean 5.98 m/s, 160 m ≈ 27 s | ⛔ ORACLE (S1 = ego future, `lan.py:51-54`) and **label-only at inference** (E12) | Supplies the *geometric* part of `g_str`. **No timing, no over-next.** |
| **(c) External / simulated map** | unbounded | ✅ non-oracle | **AlpaSim / NuRec `map.xodr` (356 driving lanes / 340 edges) is the only non-oracle route supplier we hold.** PhysicalAI-AV has *no* map, lane graph, junction annotation, traffic-light feature or route signal, and `egomotion` carries no lat/lon/GNSS — settled at five probes (CLAUDE.md). |

⇒ **On PhysicalAI-AV the 30 s strategic head is trainable today, against an ORACLE label, with
43.0 % of windows uncensored and ≈ 12.4 % carrying an over-next event.** The **non-oracle** version
of the PI's claim needs the AlpaSim/NuRec arm (W6). Both should ship; only the second can be quoted
as a deployment capability.

### 3.4 The head — E19

`str_goal_head: Linear(d_ctx, 3)` → `Linear(d_ctx, 3 + 4 + K + 4 + K)`, `K = 6`:

| field | shape | meaning | loss |
|---|---|---|---|
| `g_str_geo` | 3 | (cos, sin) route bearing + `tanh` along-track pref — **unchanged**, every downstream reader keeps its field | today's `strategic_goal_loss` |
| `p_man` | 4 | NEXT routing manoeuvre ∈ {left, right, straight-through, **none**} | CE; `none` is a **class**, never `IGNORE` |
| `t_bin` | 6 | time-to-event, bins over `[0, 30] s`: 0-2, 2-5, 5-9, 9-14, 14-21, 21-30 (log-spaced: resolution where decisions change) | CE, **masked** where `p_man = none` **or censored |
| `p_man2` | 4 | **OVER-NEXT** manoeuvre — same alphabet | CE |
| `t_bin2` | 6 | its time bin | CE, same masking |

**Label derivation** — offline, ego-only, permitted (*labels may use ego*, PI 2026-08-03). Read
directly off the record's `manoeuvre_sequence` (`t_start_s, t_end_s, dyaw_deg, radius_m, is_turn`),
re-anchored to the window's NOW: the first two elements with `t_start_s > t_now` give
`(p_man, t_bin)` and `(p_man2, t_bin2)`; class from `sign(dyaw_deg)` with `is_turn` gating.
⚠️ `straight-through` (a junction traversal) is **not derivable without a map** and is folded into
`none` on this corpus — **and that fold is recorded in the label manifest**. An under-specified class
silently absorbed is how a vocabulary drifts.

### 3.5 The dataset delta — one new field, never a moved constant

The strategic label comes from the **record**, not from new poses, so **no `future_poses` extension
is needed and the epcache is untouched**:

```
item["strat_next"]  = (p_man, t_bin, censored)      # 3 int64
item["strat_over"]  = (p_man2, t_bin2, censored2)   # 3 int64
```

derived per window in `V3Dataset.__getitem__` beside the existing `lat_v7`/`lon_v7` join
(`refc_v3_train.py:341-355`) — the same `stable_episode_id` key, the same `v7_dt = 0.1`.
`STRAT_LOOKAHEAD_S = 30.0` is its **own** constant, deliberately **not** `max(V3_HORIZONS)`.

⚠️ `MAX_H_EXT = max(v3.V3_HORIZONS)` (`refc_v3_train.py:110`) must stay exactly where it is. A
constant that becomes derived must be re-derived for every cache it meets (DE-C152); the safe move
here is not to make it derived at all, but to add a **second, independently named** one.

### 3.6 How `g_str` conditions the tactical layer **under a time constraint** — E16

Today: `gcond = gstr_embed(g_str[3])` → `gamma, beta = gstr_film(gcond)` →
`z_tac = z_tac_raw·(1+gamma) + beta`, FiLM **zero-init** (`refc_v3.py:900-902`). The *mechanism* is
right; the *content* is a bearing with **no time in it at all**, so the tactical layer cannot
distinguish a turn 2 s away from one 25 s away.

Add an explicit **urgency scalar**, computed from the head's own time distribution:

```
t_expect = Σ_k p_t[k] · t_k                   # expected time-to-event, seconds
u        = Σ_k p_t[k] · exp(−t_k / τ_urg)     # τ_urg = 6.0 s  ⇒ u→1 imminent, u→0 far
```

and feed `gstr_embed` the 13-vector `[g_str_geo(3), p_man(4), p_man2(4), u, t_expect/τ_urg]`.
FiLM stays zero-init ⇒ bit-inert at step 0, so any later effect is attributable.

⭐ **Why `u` and not raw `t`.** A raw time is unbounded and its gradient is dominated by far-future
windows, which are exactly the least reliable and most often censored. `u` saturates: ≈ 0 beyond
~3τ, and steep precisely where behaviour must change. It is also literally what the PI's sentence
names — *"depending on the time constraints of the strategic goals to follow those"* is a statement
about **how much weight the strategic goal gets now**, i.e. a gate in `[0, 1]`.

⚠️ `uplink_grad=True` in the live config, so the tactical loss trains the strategic head *through*
this FiLM (`refc_v3.py:899`) — today that is the **only** gradient `str_goal_head` receives (§1.5.2).
Keep it True and say so.

---

## 4. The selector

### 4.1 What `sel_score_v3` sees today — MEASURED, source

```
conf0        = decoder anchor confidence                       refc.py:1566   (cond carries nav via m)
terms        = [H19 maneuver_to_anchor(man5),                  refc.py:1572-1581
                lat_to_anchor(lat_prior), lon_to_anchor(lon_prior)]
conf         = _apply_grafts(conf0, terms)                     refc.py:1586
base         = conf                     (sel_refined = False)  refc.py:1640
r_terms      = []      ⛔ graft_route / graft_goal / graft_cons ALL False
score        = _apply_grafts(base, [])  ⇒ score == conf        refc.py:1663
rank         = score.masked_fill(~reach_keep, -inf)            refc.py:1685
blended      = apply_seam_clamp(score, goal_gate · scorer(...)) refc_v3.py:1058-1062
sel_score_v3 = blended ; idx = rank(blended).argmax            refc_v3.py:1066,1073
```

| does the selector see… | answer |
|---|---|
| **nav?** | Only *diluted*, through `m` → `cond` → `conf0`. **No explicit nav term in the ranking.** |
| **`g_str`?** | **No.** Only via the zero-init FiLM → `z_tac` → `man5` → the H19 anchor prior. |
| **`g_tac`?** | Only through `goal_gate · scorer` — and ⭐ **that gate HAS opened: `gate_mean = 0.17444`, `score_absmean = 4.0686`** (MEASURED, `refcv3-40284-openloop.ARM.json`). E9 is contributing ≈ 0.71 log-units against a `seam_clamp` of 1.0. |
| **`route_logits`?** | **No — `graft_route = False`.** The core strategic head's output reaches nothing. |

### 4.2 The surface is structured but under-informed — MEASURED

`refcv3-40284-openloop.ARM.json`, `.refcv3.selection_profile`, n = 4,823 win / 141 eps:

| quantity | value |
|---|---|
| distinct anchors ever selected | **50 of 128** |
| modal anchor / its share | **#57** / **0.1482** |
| entropy | **2.8425** nats of max 4.8520 ⇒ **ratio 0.586** |
| agrees with oracle | **0.5652** |
| anchor accuracy | 0.5654 [0.5320, 0.5992] (chance 1/128) |
| **oracle-selection headroom** | `oracle_sel` ADE **0.3668** vs `os` **0.4419** ⇒ **−0.0751 [−0.0884, −0.0618], separated = 17.0 % of deployed ADE** |

⇒ The selector is not random — it concentrates on 39 % of the fan at 59 % of maximum entropy and
agrees with the oracle on 57 % of windows. **It is leaving 17 % of the arm's ADE on the table**, and
it has **no term that expresses the commanded manoeuvre**. That is the gap, and §4.3 fills it for
two parameters.

### 4.3 The conditioning design — S7 + S8, on the mechanism that already exists

`AnchoredDiffusionDecoder._lan_anchor_prior(dir)` (`refc.py:1383`) is a **param-free geometric
compatibility** between a `[B, 3]` `(cos, sin, valid)` direction and the anchor set. It already
serves the supplied LAN corridor and the S6 predicted goal — *identical mechanism, different
provenance*, exactly the argument `refc.py:1645-1649` makes. Two new provenances:

```python
# S7 — the COMMANDED route reaches selection.
if self.nav_gate is not None and nav_dir is not None:
    r_terms.append(self.nav_gate * self._lan_anchor_prior(nav_dir))

# S8 — the STRATEGIC goal reaches selection, scaled by its own urgency.
if self.gstr_gate is not None and gstr_dir is not None:
    r_terms.append(self.gstr_gate * urgency[:, None]
                   * self._lan_anchor_prior(gstr_dir))
```

with, in `RefCModel.forward`:

```python
nav_bear = NAV_BEARING[nav_cmd]     # follow/straight -> 0 rad, left -> +b, right -> -b
nav_dir  = cat[cos(nav_bear), sin(nav_bear), known]   # `valid` = nav_known, NOT a constant 1
```

⭐ **`valid = nav_known` is load-bearing.** A window with no route command must contribute
**nothing** to the ranking, not "prefer straight". Mapping the unlabeled default (`follow`, index 0)
to a 0-rad bearing with `valid = 1` would assert a judgement the labeller never made — on 94.6 % of
windows today, of which **62.4 % are a collapsed UNKNOWN sentinel** (`refb_labels.py:1367-1377`).
This is the E1 defect reproduced inside the selector, and it is why **E21 is a precondition of S7**.

**Supervision.** Both gates train through the existing `selection_ce` over the survivor set
(`refc_v3.py:1189-1200`) at `SEL_V3_WEIGHT = 1.0`. **No new loss.**

**Anti-echo.** ⚠️ S7 lets the selector be *told* the answer where nav is informative. The
pre-registered read is the **paired `nav_on` − `nav_shuffled`** delta on the same windows, plus the
`os_navzero` arm (§2.3). A selection gain that vanishes under shuffle is the marginal, not the
command.

### 4.4 The reachability gate — what is measured, what is NOT, and the scope error to avoid

⛔ **The refcv3-40284 reach telemetry is NOT BANKED.** `reach_frac_candidates_clipped` and
`reach_frac_windows_empty` are computed at runtime (`refc.py:1686-1689`) and **never persisted** —
the shipped `taniteval/tools/refcv3_arm.py` never reads `sel_tele`. Two probes, different
path-binding: (A) content scan of all 63 files in the two 2026-09-04 result directories plus the
141-episode dump (the per-episode npz carries 11 keys, all trajectories); (B) repo-wide `git grep`
— the keys appear **only** in source and tests. **Zero artifacts.**

⛔ **The frequently-quoted 72.08 % is a DIFFERENT ARM and must not be carried over.**
MEASURED scope: **REF-C-XL step-30k, 256-anchor fan, 881 windows / 40 episodes**, band `v0 ± 5.0 m/s`
(`…/2026-07-27-percandidate-labels/PERCANDIDATE_LABELS.md:196-232`, artifact
`raw/t1_clip_fansize.json`): candidates removed **72.08 %**, windows empty **0.00 %**, ADE-oracle
survives **100 %**, paired Δ **+0.0000 [0.0000, 0.0000]**. refcv3-40284 is **128** anchors, 4,823
windows, a 6 s path. Same `accel_max = 2.5`; **different fan, different horizon.** Evidence class
here: **INHERITED, out of scope.** The `refc.py` comment calling the fan *"72-74 % unpickable"*
traces to exactly this measurement and inherits its scope.

⚠️ **And the counter under-reports by construction.** `reach_frac_candidates_clipped` is computed
**after** both restores (`ego_keep` rows and empty rows get their whole fan back,
`refc.py:1680-1687`), while `reach_frac_windows_empty` is computed from `dead` **before** the
restore. **They are not complements**, and neither is the raw kill rate.

⇒ **W-SEL1 (0 GPU beyond one forward pass): persist `sel_tele` — plus `sel_score` max/min/argmax,
which the sidecar schema already *declares* (`sel_score_max`, `sel_idx`, `sel_idx_base`,
`nav_injected_*`) and never populates.** A declared-but-empty field is worse than an absent one: it
reads as a measured invariant.

---

## 5. Label coverage arithmetic

### 5.1 ⛔ Scope correction on the headline number

**`45,466 / 168,910 = 26.92 % is a REF-A v1 number, not a REF-C v3 one.** It is derived from
`refav1_loader.py`'s grid (`dt = 0.2`, `op_window = 4` `refa_v1.py:346`, `reach = 60` from
`str_dt 3.0` / `str_horizon_s 12.0` `:361,375`) and `168,910` is banked at
`MODEL_REGISTRY.md:2092` / `GOALS_AND_CLAIMS.md:2267`. The literals `45466` and `26.92` are **banked
nowhere** — verified by two independent mechanisms (git-index `git grep`; filesystem walk including
untracked worktrees). They are reconstructible in closed form:
`168,910 − 27 × 4,572 = 45,466` exactly. The nearest *artifact* is a **20-episode slice**
(`…/2026-09-03-tactical-decoder/raw/window_band_census.json`: train 199/739 = 0.269283, eval
420/1,339 = 0.313667), stamped `MEASURED (slice) / INFERRED (corpus)`.

**REF-C v3's own coverage, DERIVED on its own grid** (`V3Dataset`, `window = 8`,
`max_horizon = 20`, `v7_dt = 0.1`, `T = 199` ⇒ `t ∈ [0, 171]`, `t_now = (t+7)·0.1 ∈ [0.7, 17.8] s`;
band `|t_now − 8.0| ≤ β` from `bands["tactical_s"] = (2.0, 6.0)` ⇒ `β = (6−2)/2 = 2.0`,
`v7_labels.py:310-311`):

| β | `t_now` window | admitted `t` | windows | share |
|---|---|---|---|---|
| **±2.0 s (today)** | [6.0, 10.0] | [53, 93] | **41 / 172** | **23.84 %** |
| **±4.0 s** | [4.0, 12.0] | [33, 113] | **81 / 172** | **47.09 %** |
| ±6.0 s | [2.0, 14.0] | [13, 133] | 121 / 172 | 70.35 % |
| ±8.0 s | [0.0, 16.0] | [0, 153] | 154 / 172 | 89.53 % |
| ±9.8 s | covers the whole excerpt | [0, 171] | 172 / 172 | 100 % |

⭐ **Cross-check:** `RESULT-refcv3-40284-stratified.md:361` banks **"4,823 windows (24.0 %)"** for
the clips carrying one v7.2 record. **My derivation gives 23.84 %.** Two independent routes agree to
0.16 pp — the refcv3 in-band figure is ≈ **24 %**, and **≈ 76 % of windows carry `IGNORE_ID = −100`**
(matching the trainer's own *"the −100 path the real corpus produces on ~76 % of windows"*,
`refc_v3_train.py:949`).

⚠️ **Why coverage is ~24 % and not the ~57 % the band width implies:** episodes are ~20 s, so
`t_now` only reaches 17.8 s while the band's upper half runs to 10.0 s from a `t0 ≡ 8.0` anchor —
and the **record anchors at a single point, `t0_s = 8.0` on 4,719/4,719 clips**. This is an
**ANCHOR-DENSITY limit, not a band-width limit**: there is exactly **one** labelled instant per clip.

### 5.2 The three policies

| policy | mechanism | refcv3 coverage |
|---|---|---|
| symmetric ±2.0 s (today) | one anchor at `t0 = 8.0` | **23.84 %** (banked ≈ 24.0 %) |
| symmetric ±4.0 s | widen the same anchor | **47.09 %** (exact — no clamping, since `t ≤ 171` admits the whole band) |
| **event-anchored, causal, `none` as a class** | label off `manoeuvre_sequence`, event in the window's **future** | **100 % of windows carry a strategic label**; ≈ **37.0 %** carry a NEXT event, ≈ **12.4 %** an OVER-NEXT, the rest `none` / `censored` (hard ceiling on non-`none`: **47.75 %** of clips have any event) |

⭐ **The decisive change is not the band width — it is that `none` becomes a CLASS instead of an
IGNORE.** Three reasons; the third is the one that matters:

1. **Coverage → 100 %.** Every window carries an event (class + time bin) or `none` / `censored`.
   Gradient on 100 % of rows instead of ~24 %. *(For batch 20 that also removes the
   zero-labelled-batch problem: at batch 8 today, **8.13 %** of steps carry **no** labelled row —
   MEASURED, `window_band_census.json`.)*
2. **The window becomes CAUSAL.** A symmetric ±β band also labels windows whose NOW is *after*
   `t0`, training the head to "predict" something already executed. The event-anchored band is
   `t_now ∈ [t* − H, t*]` — the event is always in the window's **future**, the only semantics a
   decision head can act on. ⚠️ Roughly **half** of today's 24 % is retrospective, so the *usable*
   coverage today is nearer **12 %**.
3. **`IGNORE` and `none` are different propositions, and conflating them is the X15 defect again.**
   `−100` says *"we do not know"*; `none` says *"we checked, and nothing happens in the horizon"*.
   A head that never sees the second **cannot learn to output "no manoeuvre"** — the majority class
   in driving, and exactly what the rendered frame shows: `LANE_KEEP p = 0.79` asserted on a window
   whose ground truth is a hard left. *(The v7.2 tactical census is itself 64.70 % `LANE_KEEP`, with
   `LANE_CHANGE_L/R`, `ABORT_LC`, `YIELD_MERGE` at exactly **0** — `GOALS_AND_CLAIMS.md:1907-1921`.)*

### 5.3 The recommendation

```
STRATEGIC : event-anchored, causal, none-as-a-class, censoring modelled
            H = 30.0 s (STRAT_LOOKAHEAD_S)   — the record already reaches it
            K = 6 log-spaced time bins over [0, 30] s
            43.0 % of windows fully observed; the rest right-censored at 38 s − NOW
TACTICAL  : band UNCHANGED at +-2.0 s. The tactical layer is a 2 s decision and a wider
            band would mislabel it. The widening is STRATEGIC only.
FALLBACK  : if event-anchoring is deferred, +-4.0 s doubles tactical coverage to 47.09 %
            for a one-line change -- but it is retrospective on half the gain and does
            NOT fix the IGNORE/none conflation.
```

---

## 6. Work items, in priority order

| # | item | cost | unblocks |
|---|---|---|---|
| **W0** | ⭐ **`--nav-from-v7` ON.** Zero code, already implemented and position-pinned. Without it every nav edge trains against a 94.6 %-constant channel. | 0 | everything nav; E15's admissibility |
| **W1** | `nav_known_channel = True` + pass `nav_known` from the trainer (E21). Plumbing is complete; **no trainer uses it.** | 0 | E15, S7 |
| **W2** | Event-anchored strategic label off `manoeuvre_sequence` + `strat_next`/`strat_over` dataset fields (§3.5) | 0 GPU | E19 |
| **W3** | S7/S8 selector terms — re-provenance `_lan_anchor_prior`, **+2 params** | 0 GPU | §4 |
| **W4** | E15/E16/E19 head changes + `REGISTERED_DELTA_KEYS_V5` + the `config_delta` pin | 0 GPU | launch |
| **W-SEL1** | **Persist `sel_tele` and the declared-but-empty `sel_score_max`/`sel_idx` sidecar fields.** A declared field that is never populated reads as a measured invariant. | 1 fwd pass | any selection claim |
| **W5** | Validate on the **`tiny` rig** (`V3_RIG_SIZES`, 16,989,725 params, ~29 min/arm on the dev box) per `TanitAD_ValidateAIDesign` **before** the A40 | dev box | A40 launch |
| **W6** | The **AlpaSim / NuRec** arm — the only **non-oracle** 30 s route label we hold | A40 | the PI's 30 s claim as a *capability* |

### The pre-registered primary

**`t_bin` accuracy, not `p_man`.** The v7.2 nav command is a **per-clip constant**: it can echo
*whether* a routing manoeuvre happens and **cannot** echo *when*. Timing is therefore the one
strategic quantity a nav-conditioned head cannot fake — and it is exactly what the PI's *"within its
time frame"* and *"depending on the time constraints"* name. Controls that must read known values:
a **constant-only** control (must read the class marginal exactly), a **nav-shuffled** control, and
`os_navzero`.

---

## 7. Escalations

1. ⚠️⚠️ **BOTH `refc.py` AND `refc_v3.py` were written concurrently while this audit ran.**
   `refc_v3.py` 70,468 B / 1,228 lines at 07:36 → 73,424 B / **1,270** lines; `refc.py` 2,299 →
   **2,477** lines. Mid-session, full reads of `refc_v3.py` failed under **two** mechanisms (MSYS
   `cp`, .NET `ReadAllBytes`) while a 60-byte `head` succeeded — an active writer, not a mount flap.
   A parallel audit additionally observed the **staged blob 532 lines SHORTER than HEAD** and read
   the worktree file as *missing*; the file is present and `git status` now reports it clean against
   the index. ⇒ **All line numbers in §1-§4 were re-pinned by CONTENT against the current files.**
   ⇒ **Before any refcv4 edit here, re-read and blob-compare index vs worktree**
   (`git ls-files --stage` vs `git hash-object`). A truncation committed to this file would delete
   the E13 nav plumbing at `:860-861` and `:674-679`. ⛔ And per CLAUDE.md, `git log` is not evidence
   your change is in HEAD — assert content per path at end of turn.
2. ⚠️ **C-NAV-SOURCE-DIVERGENCE is live** (`GOALS_AND_CLAIMS.md:2357`): refav1 and refcv3 share the
   3-way nav vocabulary but **not** the nav source. W0 aligns refcv3 to the v7.2 token and therefore
   *closes* this contradiction — it should be retired in the same turn.
3. ⚠️ **The `[6, 8) s` gap band belongs to no layer** (`s2_geom_emit_v7.py:57`), and **176 train
   manoeuvres start inside it** (`band_census.json`). They are emitted to
   `bands.unassigned_manoeuvres` and supervise nothing. Either the tactical band extends to 8 s or
   the strategic band starts at 6 s; leaving 176 events unowned is a silent hole.
4. ⚠️ **`sel_score_max` / `sel_idx` / `nav_injected_*` are declared in the arm's sidecar schema and
   populated in no artifact.** Fix or remove — see W-SEL1.
