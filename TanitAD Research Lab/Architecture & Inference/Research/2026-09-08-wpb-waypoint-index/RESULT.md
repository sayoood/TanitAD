# WP-B — `E-WP-INDEX-1`: the waypoint index into REF-C's sparse agent tokens

**Date:** 2026-09-08 (Europe/Berlin) · **Author:** Architecture & Inference FlyWheel ·
**Branch:** `agent/arch-inf-20260803` · **Parent HEAD:** `7609490`
**Pre-registration:** `Project Steering/PREREG_WPB_WAYPOINT_INDEX.md` (written before any WP-B
number existed; both outcomes committed in its §5).

**Status: code + tests + pre-registration DELIVERED and STAGED. NO training arm launched. 0 GPU
spent.** Every number below is **MEASURED** on the dev-box (CPU path, `torch 2.11.0+cu128`,
`python 3.13.5`, float32) unless its row says otherwise. Evidence class is on **every** claim.

---

## 1. What the waypoint index addresses, and how the address is computed

**MEASURED (source + `raw/`)** — the index addresses **agent SLOTS, by their metric position**.

Each anchor query carries a full trajectory — its current denoising estimate, `[B, N, S, 2]` **in
metres, ego frame, x forward / y left**. Each agent slot carries a decoded centre
`agent_slots["box"][..., :2]`, **in the same metres and the same frame**. The address is therefore a
**subtraction, not a projection**, and `refc_wp_index.waypoint_agent_geometry` computes, with **zero
parameters**:

| quantity | definition | what it says |
|---|---|---|
| `d_min` | `min_s ‖p_m − w_s‖` | does this plan pass near this thing |
| `s_star` / `tau` | `argmin_s` and `s_star/(S−1)` | **WHEN** on the plan the closest approach is |
| `lon` / `lat` | `(p_m − w_{s*})` projected on the plan's local heading at `s*` and on its left-normal | along-track / cross-track offset |
| `d_range` | `‖p_m‖ − ‖w_{s*}‖` | signed metric range difference |
| `cos_db` / `sin_db` | agent bearing relative to `w_{s*}`'s bearing | angular offset, wrap-free |

Those eight, scaled by `scale_m` (10 m — a NORMALISER, not a cut-off), form the `[B, N, M, 8]`
relation. A per-layer 2-layer MLP turns it into a **per-head additive attention logit bias**
`[B·H, N, M]` — exactly `nn.MultiheadAttention`'s float `attn_mask`.

⭐ **It is an INDEX, not a fusion block, and that is checkable rather than asserted:** the query
latent `q` **never enters** the address path. `waypoint_agent_geometry(wp, pos)` takes two tensors of
metres and nothing else — the audit in signature form. Change the plan's geometry and the addressed
slot changes deterministically (§3, `G3`).

⛔ **Into the SPARSE tokens, never a dense raster.** WP-A (`E-READOUT-CEILING-1`) **MEASURED** that a
median of 4 BEV cells (max 313) share one token cell and only 242 of 640 token cells receive any
ground-plane cell, so a raster route caps at **AP 0.4713** even with a perfect front-end — a floor no
training removes. `refc_agents.slot_features` carries continuous metric range and bearing, so the
sparse route has no such floor.

⚠️ **The geometry, restated so nobody re-derives it wrong:** refcv5 is `--image-hw 256 640` at stride
32 ⇒ its map is **8×20 = 20 azimuth columns = 6.0°/column**, and the anchor decoder already
cross-attends its 160 flattened tokens. The "4 readout columns / 30° per bin" figure is a **v6/v7**
fact; **refcv5 has no `SpatialGridReadout` at all**, so WP-B is not a readout change — there is
nothing there to change.

---

## 2. Removability — what was bit-identical, and how it was verified

**MEASURED**, `raw/removability.json`, smoke rig + `AgentSeamConfig(queries=6, d_model=32)`,
seed 1234.

### 2.1 The REQUIRED proof: flag OFF ≡ a build that never had the field

| assertion | result |
|---|---|
| state-dict key lists identical | ✅ |
| **182 of 182** parameters bitwise equal | ✅ |
| any `wp_index` key present in `state_dict` | ❌ **none** — `enable=False` means NOT CONSTRUCTED |
| **15 of 15** emitted planner tensors bitwise equal (`torch.equal`) | ✅ |
| max-abs diff over all tensors | **0.0** |

**How.** `CrossAttnLayer` constructs nothing at layer-build time; the bias heads are attached by
`AnchoredDiffusionDecoder.attach_wp_index`, called as the **very last statement of
`RefCModel.__init__`** (immediately after WP-D's `bev_aux_head`). There is exactly **one**
construction path, deliberately — a second, eager one is how two paths drift apart. And with the
flag off, `_agent_bias` returns `None` rather than a zero tensor, so the attention **never enters the
masked branch at all**.

### 2.2 Index OFF vs index ON — the one-variable proof at step 0

**182 of 182 shared parameters bitwise identical**, `+8` new keys, nothing else touched. This is the
WP-D discipline: a head attached anywhere but last shifts the RNG for every module after it, and the
A/B would then differ in the SEED as well as in the lever.

### 2.3 ⛔⛔ AND THE PROOF WAS GREEN FOR A REASON THAT WAS NOT WP-B — a defect in my own guard

`agent_gate` is zero-init (WP-6's own discipline), so the **entire** agent branch is multiplied away
at step 0. **MEASURED:** the on/off comparison is bitwise equal **even with the bias head
deliberately corrupted** (`N(0, 10)` on `mlp[-1]`). A guard whose PASS does not depend on the thing
it guards is the `CLAUDE.md` *"a check that shares the defect it checks for"* family — the same
class as the label builder that verified its buckets against its own rounded ladder.

| regime | max-abs `traj` diff | reading |
|---|---|---|
| `agent_gate = 0`, zero-init bias | **0.0** | the required proof |
| `agent_gate = 0`, **CORRUPTED** bias | **0.0** | ⛔ **the confound, measured** |
| `agent_gate = 1`, zero-init bias | **1.9073486328e-06** (`= 2**-19`) | float32 **kernel rounding**: supplying a float `attn_mask` at all changes which SDPA kernel torch dispatches to |
| `agent_gate = 1`, bias `N(0, 1)` | **2.169e-03** | the lever, 3 orders above the rounding floor |
| `agent_gate = 1`, bias `N(0, 10)` | **5.391e-03** | ditto at 10× scale |

⇒ **two tests now exist where one did.** `test_the_shipped_init_proof_is_not_carried_by_the_index_alone`
states the confound as an executable fact; `test_at_a_LIVE_agent_gate_the_zero_init_index_agrees_to_float32_ROUNDING`
re-runs the comparison where the bias head is load-bearing. ⛔ **The live-regime claim is "agrees to
float32 rounding", never "bit-identical".**

---

## 3. The controls — which read which known value, each VERIFIED

⛔ Every expectation is a **LITERAL** computed by hand or by `math`, never an expression over the
code under test.

| control | **known value (literal)** | MEASURED | verified |
|---|---|---|---|
| **detached** | the graph is **severed**: `rel.requires_grad is False`, `grad_fn is None`, both operands' `.grad` stay exactly `None` | as stated; the treatment's gradient is non-zero on **both** operands | ✅ |
| **shuffled** | row `b`'s relation equals **bitwise** the geometric relation of row `perm[b]` | `torch.equal` | ✅ |
| **constant** | the bias is **bitwise identical across the anchor axis**, for every anchor | `torch.equal` for all `n` — with a **non-zero-init** head so it cannot pass trivially, and the treatment verified **not** to have the property | ✅ |
| **shuffle at `B = 1`** | must **REFUSE** — a one-row permutation is the identity, so the control would silently BE the treatment | `ValueError: batch >= 2` | ✅ |
| **degenerate scene** (padded slot at `(0,0)`, plan that goes nowhere) | every feature **FINITE** — a NaN here poisons the whole attention row through `NaN + (-inf)` | all finite | ✅ |
| **radius gate** | never leaves a query row fully `-inf`, and counts **padding** in the emptiness test | 4/4 rows correct | ✅ |

### 3.1 The analytic address literals

Scene: a straight plan `(5,0) (10,0) (15,0) (20,0)`; agents at `(20,0)`, `(20,6)`, `(5,0)`.

| quantity | **literal** | MEASURED |
|---|---|---|
| `d_min` | `[0.0, 6.0, 0.0]` | `[0.0, 6.0, 0.0]` |
| `s_star` | `[3, 3, 0]` | `[3, 3, 0]` |
| `tau` | `[1.0, 1.0, 0.0]` | `[1.0, 1.0, 0.0]` |
| `lat` / `lon` | `[0.0, 6.0, 0.0]` / `[0.0, 0.0, 0.0]` | exact |
| `d_range` (agent 1) | `sqrt(436) − 20 = 0.880613` | `0.880613` |
| `cos_db` / `sin_db` (agent 1) | `20/sqrt(436)` / `6/sqrt(436)` | `0.957826` / `0.287348` |
| **attention mass ratio** at a bias gap of 6.0, agent tokens made identical | `e**6 = 403.4287934927` | matches, `rel=1e-3` |
| first-waypoint heading on a TURNING plan, agent at `(5,1.5)` | `lon = 0.5/sqrt(25.25)`, `lat = 5.0/sqrt(25.25)` | matches, `1e-5` |

⭐ **The discriminating control:** the same agent set with a LEFT-going plan addresses agent **1**,
not agent 0. A positive assertion on the straight plan alone passes on an implementation that always
returns slot 0 — which is exactly the `M1` defect.

⚠️ **No AP is computed anywhere in WP-B**, so the array-order tie-breaking trap (a constant arm
scoring `+0.881 %` / `+22.8 %` above its own base rate on two rigs) does not apply here. Said
explicitly rather than left to inference.

---

## 4. The mutation proof — what went RED, and the hole it found

`code/mutation_proof.py --out raw/` · **12 guards × 9 mutations** · baseline all-GREEN · **exit 0**.
Full record: `raw/MUTATION_LOG.md`, `raw/mutation_log.json`.

| mutation (the REAL defect) | guards that went **RED** |
|---|---|
| `M1` index by **ARRAY POSITION** instead of geometry | G1, G2, G3, G5, G7, G12 |
| `M2` swap x and y (the `anchors.pt` units family) | G1, G2, G3, G4, G12 |
| `M3` ignore the `detach` flag | G7 |
| `M4` transpose the bias to `[B·H, M, N]` | G4, G5, G10 |
| `M5` non-zero output init | G11 |
| `M6` drop the origin prepend (heading off-by-one) | **G12** |
| `M7` radius gate ignores padding in its emptiness test | G9 |
| `M8` shuffle accepts `B = 1` | G6 |
| `M9` no denominator clamp (NaN on a padded slot at the origin) | G8 |

### ⭐⭐ `M6` turned NO guard RED on the first run — and that is the most useful line here

Every analytic guard used the **straight** plan, whose per-step heading is constant, so borrowing
step 1's heading for step 0 changed nothing at all. The guard set was **incomplete**, and no amount
of re-reading the code would have shown it. **G12** closes it with a TURNING plan whose closest
approach is at `s* = 0` and literals `0.5/sqrt(25.25)` / `5.0/sqrt(25.25)`.

⚠️ **`M7` was a bug actually present in the first draft** of `apply_radius_gate`: the emptiness test
looked only at the radius, so a query whose only in-range slot was **padding** kept its gate and put
every unit of attention mass on a slot that does not exist. Caught by `G9` before it shipped.

---

## 5. The trainer wiring, and what it REFUSES

**MEASURED**, `raw/trainer_wiring.json`.

| check | result |
|---|---|
| **launch diff, B0 vs B1** (parsed namespaces, not intent) | **exactly one key**: `wp_index: off → on` |
| `--wp-index on --agents off` | ⛔ **REFUSES** at pin time — *"there are no tokens to address … a REFUTATION MANUFACTURED BY A MISSING SEAM"* |
| `--wp-index off --wp-index-mode shuffle` | ⛔ **REFUSES** — a no-op knob that would be stamped into `config.json`; *"`--wp-index-mode` is the dangerous one: it names a CONTROL ARM, so a reader would believe a control had been run"* |
| `wp_index.enable` with `cross_agent` False, at model construction | ⛔ `ValueError: no agent cross-attention` |
| `assert_seams_are_built`: stamped **and** built | PASSED |
| stamped, **not** built | ⛔ refuses — *"the record would claim a waypoint index the weights do not contain"* |
| built, **not** stamped | ⛔ refuses — *"a live seam absent from the run record"* |
| **stamped mode ≠ attached mode** | ⛔ refuses — *"a CONTROL arm would be reported as the treatment, or the reverse"* |
| `config.json` stamp | full `wp_index` block incl. `mode`, `detach`, `radius_m`, `feat_dim`, the 8 feature names |
| `agent_knob_dests()` | picks up all **7** `wp_index*` dests, so each lands in `agent_knobs` |

⭐ The **mode-mismatch** direction is the one WP-4's `sampler` defect did not have: a run whose record
says `geom` while the weights carry `const` would report a **control as the treatment**, which is
worse than either half of the classic false-provenance failure.

### 5.1 ⛔⛔ A SECOND DEFECT THIS PACKAGE SHIPPED AND THE REPO'S OWN GUARD CAUGHT

**MEASURED — the full 7,520-test suite, `9 failed, 7449 passed, 60 skipped, 2 xfailed`.**
Exactly **one** of the nine was WP-B's:
`test_refc_v3_agent_provenance.py::test_P2_every_knob_is_recoverable_from_the_stamp_BY_VALUE`.
*(The other eight touch files WP-B never edited — the v6 trainer's `--nav-cond` / `--horizons`
refusals, two corpus writers, four scripts missing `encoding=`, and one `OSError` on the G: mount.
Verified positively: `git diff HEAD --numstat` on each named file shows WP-B changed none of them.)*

**The defect.** My `--wp-index off` refusal covered `--wp-index-mode`, `--wp-index-detach` and
`--wp-index-radius-m` — and **left `--wp-index-hidden`, `--wp-index-scale-m` and
`--wp-index-const-xy` free**. Those three parse, land in `agent_knobs` inside `config.json`, and do
**nothing**: the M18 no-op-flag defect verbatim, three flags wide, in the very package whose §5 table
claims the class is closed.

⭐ **Why it was caught at all: that test derives its knob list FROM ARGPARSE**, not from a
hand-written list — so adding a flag automatically enrols it, and there was no list to forget to
update. It is the same discipline as the `test_physicalai_feature_readset.py` pin, and it is the
reason a stale hand-maintained list is never the right shape for this check.

**Fixed** (`stack/scripts/refc_v3_train.py`): all **six** knobs now refuse under `--wp-index off`.
**Pinned** (`stack/tests/test_refc_v3_refcv5_wiring.py`, new): six parametrised refusal cases, the
`--agents off` refusal, **and a same-breath control** asserting the knobs *at their defaults* do
**not** trip the refusal — without which `--wp-index off` would be unusable and the six cases would
pass for the wrong reason. The provenance test gains WP-B's enabling context (`--wp-index on` in its
`base`), exactly as it already does for WP-D's `--bev-aux col`, with the reason written beside it.

**After the fix:** `904 passed, 3 failed` over the whole refc/agent/trainer/runbook surface — and
all three remaining failures are the pre-existing ones above.

---

## 6. Cost

| item | value | class |
|---|---|---|
| **parameters on the REAL 106.7 M refcv5 model** (`v3.RefCV3Model`, `--image-hw 256 640 --agents oracle`) | **+2,208** exactly: total **106,666,205 → 106,668,413** = **+0.00207 %**; `decoder` line **11,003,021 unchanged**; 4/4 layers carry a bias head; 16 new `state_dict` keys; `assert_seams_are_built` PASSED both ways | **MEASURED** — `raw/e2e_real_model.json` |
| analytic form, derived independently | `(8h + h + hH + H) × layers` = `(256+32+256+8) × 4` = **2,208** | ⭐ hit **on the nose** by the built model |
| parameters, smoke rig (`hidden 32, n_heads 4, layers 2`) | **+840**, same form | **MEASURED** |
| activation memory at the deployed shape (`B=20, N=117, M=100, S=8, H=8`) | bias `[B·H,N,M]` **74.9 MB/layer**; MLP hidden **30.0 MB/layer**; displacement `[B,N,S,M,2]` **15.0 MB** ⇒ order **0.4 GB/pass**, ×3 passes/forward with autograd ⇒ order **1.3 GB** | ⚠️ **ESTIMATED — priced on paper, NOT run** |

⛔ **The memory figure must be re-measured on the real path before any launch.** Porting a paper
figure onto a live configuration is exactly the `e_trunk_pooling` failure (a windowed tensor that
looked cheap and was 25.6 GB) and the `--v2-cache` failure (a 1.38 TB "wall" that did not exist).

---

## 7. Findings at HEAD that WP-B did NOT change (reported, not silently fixed)

1. ⚠️ **The `diffusion_steps` refinement loop is AGENT-FREE at HEAD.**
   `refc.py`'s `for i in range(_loop_steps): ... self._decode(kv, cond, x_in, t_idx)` passes **no**
   `agent_tokens`, so WP-6's agent cross-attention — and therefore WP-B's index — does not reach that
   loop, while the classifier pass, the sampler and the emitted-fan pass all do. **MEASURED at
   source.** Adding the tokens there would change the agents-on arm at HEAD and break WP-B's
   removability proof, so it is **reported, not fixed**. ⚠️ It is inert for refcv5-v2 (`--sampler
   ddim` sets `_loop_steps = 0`), and live for any classifier/refine arm. A comment now marks it.
2. ⚠️ **`param_breakdown` does not report the WP-6 agent seam, so its sum invariant breaks with
   `--agents` on.** MEASURED on the smoke rig: `total` 144,892 against a line sum of 122,791 — a
   **22,101-parameter** gap that is `agent_head + agent_embed`. Pre-existing; WP-B's own line is
   carved out of `decoder` correctly and the invariant still holds for every agents-off build.

---

## 8. What this package does NOT claim

1. ⛔ **No claim that WP-B helps.** No arm has run; every planner cell of the pre-registration's §5B
   is empty by design.
2. ⛔ **No capability claim from any oracle rung.** The B0/B1 ladder uses `--agents oracle`, which
   reads a privileged label at inference and is stamped inadmissible.
3. ⛔ **WP-C is exactly as open as it was.** WP-B does not enable the agent seam, does not change any
   default, and does not argue for or against `PI_DECISION_QUEUE.md` item 9.
4. ⚠️ Everything MEASURED here is **dev-box CPU, float32, `torch 2.11.0+cu128`**. A CUDA re-measure
   of the removability proof is owed before launch.
5. ⚠️ The DiffusionDrive precedent is **`PUBLISHED-SECONDARY`** — the primary is not banked in
   `TanitAD Research Lab/Library/`. It justifies a design; **no number here rests on it.**

---

## 9. Deliverable manifest

| artifact | where it lives | only one place? |
|---|---|---|
| `refc_wp_index.py` — the address, bias head, controls, radius gate | `repo:stack/tanitad/refs/refc_wp_index.py` | no (staged) |
| gated wiring | `repo:stack/tanitad/refs/refc.py` | no (staged) |
| flags, refusals, stamp, seam assertion | `repo:stack/scripts/refc_v3_train.py` | no (staged) |
| **35** tests | `repo:stack/tests/test_wp_index.py` | no (staged) |
| pre-registration | `repo:Project Steering/PREREG_WPB_WAYPOINT_INDEX.md` | no (staged) |
| claims-register row `E-WP-INDEX-1` | `repo:Project Steering/GOALS_AND_CLAIMS.md` | no (staged) |
| mutation proof (12 guards × 9 mutations) | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-08-wpb-waypoint-index/code/mutation_proof.py` | no (staged) |
| removability measurement | `…/code/removability.py` | no (staged) |
| trainer-wiring probe | `…/code/check_trainer.py` | no (staged) |
| `MUTATION_LOG.md`, `mutation_log.json`, `removability.json`, `trainer_wiring.json` | `…/raw/` | no (staged) |
| this file | `…/RESULT.md` | no (staged) |

⛔ **Nothing produced here lives in only one place.** No pod was touched; nothing was shipped to Thor
or the A40.

---

## 10. ESCALATE — integration and the one decision

1. ⛔ **WP-B cannot launch until the agent seam is authorised** (`PI_DECISION_QUEUE.md` item 9). The
   index refuses `--agents off` by design. This is a **PI decision**, not an implementation gap.

   ⭐⭐ **BUT THE LADDER'S FIRST RUNG SIDESTEPS ITEM 9's MEASURED CAUSE, AND THAT IS THE KEY DESIGN
   POINT.** Item 9 MEASURED that `--agents head` degraded **distance-keeping at BOTH seeds** —
   min headway **−0.5015** / **−0.4535** m, min time-gap **−0.0702** / **−0.0351** s, min TTC
   **−2.1959** / **−1.2529** s, all separated **worse** — which is *exactly* the LONGITUDINAL family
   WP-B's primary bar sits on. A WP-B arm on the `head` path would be measured **through** a known
   two-seed regression, and a null would be uninterpretable.

   ⭐ Item 9 also **relocated the cause**: a `clip_id`-deranged join reproduces the *entire*
   degradation while `head − shuf` is not separated on any distance-keeping metric at either seed
   ⇒ the cost is **the auxiliary DETECTION TASK competing for a 17 M trunk at 500 steps, not the
   agent information**. ⇒ **`--agents oracle --w-agent 0` has no detection loss at all**, so the
   measured cause is **structurally absent** from B0/B1 rather than hoped away. The ladder therefore
   asks *"does the ADDRESS help, given agents?"* before *"can a detector supply them?"*, and the
   second question is not run until the first returns. ⚠️ **INHERITED** from
   `PI_DECISION_QUEUE.md` §9, not re-verified here — admissible for choosing the ladder's order,
   **not** as a WP-B result.
2. ⛔ **The ladder is ~14 GPU-days, not an opportunistic gap-filler.** The pre-registration's §7.2
   therefore proposes a **≈ 40 GPU-hour, 6,000-step gate** (B0/B1 + a seed replicate + const/shuffle
   controls) whose thresholds are scale-free because every bar is a ratio to a floor measured in the
   same panel. **Do not fund the full ladder before that gate clears.**
3. ⚠️ **Owed:** bank the DiffusionDrive primary (`tools/kb_add.py`) to convert the precedent from
   `PUBLISHED-SECONDARY`; re-measure the activation memory on the real path; re-run the removability
   proof on CUDA.
