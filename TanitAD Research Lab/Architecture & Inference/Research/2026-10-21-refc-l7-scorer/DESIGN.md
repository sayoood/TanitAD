# L7 — a learned trajectory SCORER / SELECTOR for refc: design

* **Author.** Architecture & Inference agent (scorer9), dev box, 2026-10-21, clock read 23:13 +0200 (Berlin).
* **Status.** DESIGN. Zero GPU used. Nothing here is a result. The registered experiment is
  `PREREG_L7_SCORER_DRAFT.md` beside this file (a DRAFT until the Master Mind seals it).
* **Evidence classes.** MEASURED (artifact named) · READ (source, `file:line`, at the frozen tree
  `C:/lgs/Lfb47a58376/stack` = launch tree `fb47a58`, and REFe at the repo path
  `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/`, blob-verified identical
  to the dev-box copy `C:/Users/Admin/par18/hnfp/base/refe/` for `model.py` `189e1ac0`, `navsim_dac.py` `1884fee7`,
  `train.py` `f19b63d2`, `planner.py` `6fb6598a` against `agent/arch-inf-20260803`) · ESTIMATED (basis stated) ·
  HYPOTHESIS.

---

## 0. Why this lever (MEASURED; GAC rows read from `agent/arch-inf-20260803` at `cd8c465`)

| fact | value | source |
|---|---|---|
| refcv8 EMA-48,852 fan ceiling, PhysicalAI eval139, 2 s grid (T0) | oracle-in-fan **0.0841 / 0.0847 m** (s1 / s0) | `R8-FANCEIL-1`; `C:/Users/Admin/refcv8_pf/fanceil/RESULT.md` |
| the model's own pick, same windows | **0.2990 / 0.2973 m**; ADE-rank median 13 of 149; pick = oracle 8.9 % / 9.2 % | same |
| where the loss is | selected − oracle splits **+0.194 m along-track vs +0.055 m cross-track** (78 % longitudinal) | same |
| 1–6 s grid | selected 2.80 m, oracle **0.68 m** | same |
| best GT-free pickers (post-hoc) | `sel_idx_base` −0.0156 / −0.0161; MBR32 frozen cell **−0.0308 [−0.0411, −0.0216]** (s0 replicate) | `R8-FANCEIL-1`, `FROZEN_PICKER_CELL.md` |
| what refcv8's selector is | refcv6's 1,145-param scorer + E9 goal graft + **4 X1h critics trained to imitate the log** (`dir_correct`, `progress_10pct`, `heading_15deg`, `hypothesis_correct`); drivable critic OFF; no sub-score selector, no RL | `R8-DD-CONFORM-1`; config §1.3 below |
| REFe's learned PDM scorer, NAVSIM navtest | scorer-only SFT-4: **84.46 → 88.25, +3.79 [+3.05, +4.54]** ×100 over 136 logs; every sub-score rises (NC 98.13→98.94, DAC 94.71→96.58, EP 76.70→81.20, TTC 93.69→95.63) | `D-REFE-SFT-4` |
| REFe full | 88.82 / 89.15 navtest | `R8-WHYPOOR` |
| the decision experiment this arm follows | `H-NT-DECIDE-1`: D_M ≥ 0.8 ⇒ "keep the refc trunk and port the PDM scorer first" | `PREREG_NT_DECIDE.md` §4.2 l.78 |

⚠️ **Conflict reported (rule 4 of the brief).** The brief quotes REFe's scorer as worth "about 4.5 PDMS (navtest 84.46 →
88.99)". **No `88.99` exists in GOALS_AND_CLAIMS.md at `cd8c465`** (searched `88\.99`). The register's numbers are
84.46 → 88.25 (+3.79, `D-REFE-SFT-4`, scorer-only) and REFe full at 88.82 / 89.15 (`R8-WHYPOOR`, which also differs in
training). This design cites the register; the brief's 88.99 is unsourced.

**Why a learned scorer and not another rule.** Four GT-free rules have been tried on this fan (own pick, BASE, MBR32,
argmax of the core score). The best recovers **0.031 m of a 0.215 m gap** (14 %). REFe's scorer closed 25 % of its
pick-to-oracle gap on NAVSIM (`D-REFE-SFT-4`: pick ADE 1.186 vs oracle-of-64 0.334 vs random 1.471, "closes 25 % of the
gap") and that was worth +3.79 PDMS. The selection loss on refcv8 is **78 % longitudinal**, and none of refcv8's four
critics has a speed-profile target finer than ±10 % progress (`refcv8_conditioning.py:537-556`). A scorer that is
taught the along-track error directly is the largest untried lever (Rule Zero item 5: go where the arithmetic points).

---

## 1. (a) What REFe's scorer exactly is (READ, REFe repo path above)

### 1.1 Architecture

| piece | definition | file:line |
|---|---|---|
| proposals | M = **64** WTA proposals, **20 poses @ 5 Hz = 4 s**, (x, y, yaw) | `model.py:93-95` |
| scorer query | the **detached candidate trajectory**, flattened, through `score_q_mlp` = Linear(20·3 → 256) · GELU · Linear(256 → 256). It used to be the proposal's latent query; the sizing study replaced it with the geometry ("a scorer that never sees the geometry it is scoring is judging the intent") | `model.py:554-561`, `:779` |
| scoring decoder | `score_dec` = `score_dec_depth` (default = `dec_depth` = **4**) `CrossBlock`s, d = 256, 8 heads, attending to the **VISUAL** tokens (`scene_proj(visual)`), not the 16 register scene tokens the trajectory decoder reads | `model.py:551-553`, `:757`, `:783-786` |
| self-attention | **ON**: "the decoder's self-attention spans the set, so a trajectory is scored IN ITS SET" | `model.py:699-709` |
| stop-gradients | trajectory detached (paper); visual context detached by default too (`detach_scorer_context=True`, "ours, not the paper's") | `model.py:152`, `:779-784` |
| heads | `score_head` = Linear(256 → **6**): NC, DAC, EP, TTC, C, DDC (SFT-4 adds a 7th, teacher-lane compliance) | `model.py:96`, `:562`; `train.py:265` |
| ego / goal input | **none in the scorer** — the ego + goal/command token is added to the PROPOSAL queries only | `model.py:762-772` vs `:779` |

### 1.2 Targets — NAVSIM's own PDM, per proposal

`navsim_pdm_targets.py` (whole file, 139 lines):

* `metric_cache_parts(scenario)` (`:78-87`) runs NAVSIM's **own** `MetricCacheProcessor` steps (navsim @ `3e8291b`) without
  writing a file: PDM-Closed reference, centreline, route lanes, drivable-area map, GT detections interpolated to 0.1 s.
* `score_plans(parts, plans8)` (`:91-132`) runs NAVSIM's **own** `PDMSimulator` (the LQR tracker; `navsim_dac.py` vendors
  `simulated_states` verbatim for the DAC-only path) and `PDMScorer` on `[PDM-Closed, proposal_1..M]` (41 steps @ 0.1 s)
  and returns per proposal `nc, dac, ep, ttc, c, ddc, pdms`, with **EP reproduced pairwise** — each proposal normalised
  against PDM-Closed alone, exactly as `pdm_score.py` scores `[pdm, pred]` (`:115-121`).
* `executed_plans8(props20)` (`:135-139`): the plans scored are the **executed** ones — last-heading repair (A7) then
  `navsim_dac.to_navsim` to NAVSIM's 8 poses @ 2 Hz.
* **On-policy** (PI 2026-09-26 option B, P6 D1/D2): the labeller scores the student's **own** current proposals with lag 0;
  the trainer re-hashes the array it submitted (`train.py:1724-1729`). The older path (`ScorerBank`, `train.py:239-262`)
  precomputed targets for a fixed candidate set and assigned each to the nearest proposal; it states that approximation.
* Validation: `eval/validate_pdm_targets.py` against NAVSIM's real metric cache; P6 D9: `score_plans(cached) ==
  score_plans(fresh)` on 9/9 seeded sets, regression arm 0/9 (`eval/P6_IMPLEMENTATION.md` §2 D9).

### 1.3 Loss

BCE-with-logits per component against the PDM value (soft labels: EP / TTC / C are continuous in [0, 1], NC ∈ {0, 0.5,
1}), **unweighted mean over the 6 components**, summed over masked samples and divided by a constant `cov_norm` (64 per
sample on the online path), times `score_w` (`train.py:1732-1735`; logged per component as `bce_nc … bce_ddc`,
`:1801-1802`). SFT-4 trained the scorer only, generator frozen (`D-REFE-SFT-4`: "scorer-only, NAVSIM PDM targets + lane
output").

### 1.4 Aggregation at selection

`REFePlanner.aggregate` (`planner.py:537-576`), rule `navsim_v1` (default since Amendment 5, `:525`):
**score = σ(NC) · σ(DAC) · (5 σ(EP) + 5 σ(TTC) + 2 σ(C)) / 12**; argmax over the 64 (`:578-583`). The old `logitsum`
(unweighted sum of six raw logits) is kept only as a named control (`:540-547`, `:581-582`). SFT-4 uses `navsim_v1_lane`
= the same × σ(lane) (`:555-558`).

### 1.5 Known blind spot (`D-REFE-ZEROTAIL-3`)

On u400's 343 DAC-only zeros: 60.1 % [51.4, 68.0] are near-twin grazes (the pick matches the best non-zero alternative to
a median 0.29 m); 42.0 % [33.9, 49.4] exist only in NAVSIM's simulated tracking; **the DAC head rates the pick at least
as drivable as the alternative on 72 %, within-token AUC 0.46 vs 0.92 on random tokens**. ⇒ L7 inherits this: a scorer
trained on PDM targets of the **raw** plan cannot see the tracker-made part. Design consequence (§3.4): L7's NAVSIM
targets are computed on the **executed** plan through NAVSIM's own LQR (as REFe's are), so the tracker-made DAC
failures are in the TARGET; whether a head can predict them from vision is measured (§6, the DAC-zero cell), not assumed.

---

## 2. (b) How refc produces its fan, and where selection happens today (READ, `C:/lgs/Lfb47a58376/stack`)

### 2.1 The fan (refcv8-r101-s0, EMA-48,852; config `D:/refcv8_eval_kit/ckpt/config.json`, md5-paired with the Thor copy)

| piece | value / where |
|---|---|
| base fan | **117** v0-conditioned anchors (`--anchors refc_anchors_6s_v0cond_alat_117.pt`, `--anchor-v0-conditioned`), refined by a truncated DDIM (**2 steps**, `seams.sampler_steps = 2`) in control space; base noise `eps_b = randn((b, 117, …))` at `refc.py:3036` |
| extra candidates | **32 allocated** (`--r8-n-alloc 32`): `refcv8_conditioning.allocate` spreads M over the top-k joint (lat3 × lon6) tactical hypotheses ∝ p (`refcv8_conditioning.py:395-433`); appended by `_r8_extend` (`refc.py:2486`, called `:3454`); order `[base 117 | allocated 32]`; prior-free group OFF |
| N | **149**, 8 slots at (0.5, 1, 1.5, 2, 3, 4, 5, 6) s (`config.horizons = [5,10,15,20,30,40,50,60]` ticks) |
| emitted tensors | `anchor_traj [B,149,8,2]`, `refined_logits [B,149]`, `anchor_logits [B,117]` (base only), `sel_score [B,149]`, `reach_keep`, `ceil_keep`, `sel_idx`, `traj`, plus the r8 keys (`r8_cand_src`, `r8_kind`, `r8_lat3`, `r8_lon6`, `r8_emit_keep`, `r8_phi`, `r8_sub_logits [B,149,4]`) — `refc.py:3940-3984` |

### 2.2 Today's selection (three stages)

1. **Decoder score** (`refc.py:3800-3855`): `base` + graft terms (behaviour gate, nav-compliance prior, the refcv8
   factorised term β·log p(lat) + β·log p(lon) + γ·sat_prog + γ·sat_head, the X1h critics Σ w_m logσ(s_m)) through
   `_apply_grafts`. ⚠️ `sel_refined = False`, `sel_score_emitted = False` in this checkpoint's config: **the ranked score
   is the t = 0 confidence, not the refined one** — the `D-REFC-DDAUDIT-3` defect (`refc_selector.py:7-11`) is live.
2. **Masks then argmax** (`refc.py:3860-3934`): reachability band (S2, `accel_max 2.5`), the set-speed ceiling filter
   (refcv6 §5), the emit mask; `score` is returned UNMASKED.
3. **E9 re-pick** in `RefCV3Model` (`refc_v3.py:2598-2638`): `blended = sel_score + clamp(goal-distance graft)` (+ the
   drivable critic, off here), masked by `e9_rank` (`refc_v3.py:1025-1053`: reach ∧ ceiling ∧ emit, empty-row
   fallback), argmax ⇒ `sel_idx`; the decoder's own pick survives as `sel_idx_base` (`:2637`).

### 2.3 What exists at selection time that a scorer may read

| tensor | computed from | admissible as a SCORER input? (§4) |
|---|---|---|
| `anchor_traj` (the candidates) | the generator | yes — the object scored, passed **detached** |
| `v0` | IMU-measurable speed at t0 (`config.ego.channels = ['v0']`) | **yes** |
| BEV tokens (`bev_source = map_hires_pool`, `--tac-decoder-d-bev 96`) | the 10 cm perception branch on the camera | **yes** (vision) |
| agent / box3d tokens (`--agents head`) | the agent head on the camera | **yes** (vision) |
| `cond = [nav(4), max_speed(4), v0, a0]` (`refc_v3.py:1899-1958`) | nav (route-derived), the fed ceiling (future-speed-derived on PhysicalAI, `max_speed_input.py:12-21`) | **no** (§4) |
| `z_tac` (`refc_v3.py:1610-1742`) | pooled image **+ nav + ego + max-speed + goal point** | **no** (carries nav and the ceiling) |
| `sel_score`, `refined_logits`, `anchor_logits`, `r8_sub_logits` | the model's own heads (vision + v0 + nav + ceiling) | **L7-F only**, declared as an inherited channel (§4) |
| `reach_keep`, `ceil_keep`, `r8_emit_keep` | v0 + geometry; the fed ceiling; schedule | used as **hard masks only**, exactly as the shipped plan uses them |

The refcv7 scene memory is already captured by the scene hook as `r7_agent_tokens / r7_bev_tokens / r7_cond`
(`refc_v3.py:1958-1963`) — **but only when `self.refcv7_wta is not None`**, which is false for refcv8-r101-s0
(`w_r7_wta = w_r7_scorer = 0`, none of `--refcv7 / --w-r7-*` in argv). The integration diff (§3.6) widens that
condition; it is the only model-side change L7-S needs.

### 2.4 Where the banked fans live (MEASURED)

`C:/Users/Admin/refcv8_pf/fanceil/runs/{s0,s1,dec_s1}/dump/`: `fan_extras.npz` (keys `clip_index, ws, t, anchor_traj
[4754,149,8,2], anchor_logits [4754,117], refined_logits [4754,149], sel_score [4754,149], sel_idx, traj, reach_keep`),
`refcv6_extras.npz` (tactical logits, `v0`, `v_max_ms`, `gt_max_speed_6s`, …), and `ep*.npz` (`g [n,4,2]` = GT at
0.5–2 s, `v0`, `os`, `ws`, `eid`, `clip_index`). Writer: `fanceil/code/fan_roll_r8_pf.py` (capture on
`r7_roll.Refcv6Proxy.__call__`, `FAN_KEYS` l.32). The S6 GT at (1, 2, 3, 4, 5, 6) s lives in the battery trees'
`s6_s<seed>/ep*.npz` (e.g. `picker/archive/pkident_pf/s6_s1`), keyed `(clip_index, ws)`. ⛔ **These are eval139 fans.
No train-split fan is banked anywhere** (searched `refcv8_pf/`, `D:/refcv8_eval_kit/`). A scorer may not be fitted on
eval139 (§5.1).

---

## 3. (c) The proposed refc scorer

### 3.1 One sentence

**L7 is REFe's / DrivoR's disentangled scoring decoder, re-pointed at refc's own 149-candidate fan and its own detached
perception tokens, trained scorer-only (generator frozen) on that fan, with a target that is the benchmark's own score on
NAVSIM and a GT-derived imitation + safety target on PhysicalAI, and selecting through the model's existing hard masks.**

### 3.2 Three arms, nested by what they may read

| arm | reads | role | params |
|---|---|---|---|
| **L7-E** (floor) | candidate geometry + `v0` | the **ego-scalar-only floor**: if this beats the own pick, the fan's own pick is beaten WITHOUT vision, and L7-F / L7-S must beat **L7-E**, not the own pick, to claim anything visual | **154,504** (MEASURED, `python code/refc_scorer.py --params`, d 128, 2 residual MLP blocks) |
| **L7-F** (fan-feature) | L7-E + the model's own per-candidate outputs (`sel_score`, `refined_logits`, the 4 `r8_sub_logits`, rank-in-row of `sel_score`, `is_alloc`) + a GT-free set-context (softmax-weighted MBR risk, the frozen picker's statistic, so the MBR32 rule is **inside** the hypothesis class) | the cheap arm: trainable from a fan bank with no scene tokens, on CPU in minutes | **155,400** (MEASURED, same) |
| **L7-S** (scene) | L7-E + cross-attention to the **detached** BEV and agent tokens | the real arm (REFe / DrivoR form) | **464,520** (MEASURED by `--params` at d_bev 96 / d_agent 256, d 128, 2 decoder layers, 4 heads; the capture's real source widths are re-read before training); the NAVSIM-head variant L7-NS is **464,262** |

All three share one module (`code/refc_scorer.py`): a per-candidate trajectory embedding (8 slots × (x, y, speed,
heading) normalised, + v0 — the DrivoR / refcv7 `trajectory_features` recipe), an optional feature MLP (L7-F), an
optional scene decoder (L7-S: **no self-attention across candidates** by default — a candidate's score is then a
function of that candidate and the scene alone, so it can serve as a TOAD search reward later; REFe's set-attention is a
registered secondary arm), and heads.

### 3.3 Heads and targets per benchmark

**NAVSIM (L7-NS).** Six BCE heads NC, DAC, EP, TTC, C, DDC — REFe's exact targets, built by REFe's **unchanged**
`navsim_pdm_targets.score_plans` (no PDM re-implementation; `code/l7_targets_navsim.py` only adapts refc's candidates to
it). The plan scored is the **executed** one: refc's 8 knots → `knots_to_navsim` (C2 not-a-knot cubic through (0,0) +
the 8 knots at 0.5…4.0 s, heading = atan2 of the tangent; `refcv8_bridge.py:18-20`, imported from the bridge package,
the same conversion the bridge applies to the emitted plan). Loss = REFe's (`train.py:1732-1735`). Selection =
`navsim_v1` aggregate (`planner.py:556-560`) over the masked fan.

**PhysicalAI (L7-PAI).** No PDM exists on PhysicalAI (no map, no simulator). Targets (all LABEL-side; built by
`code/l7_targets_pai.py`, which imports `refcv7_oracle` **unchanged** for the safety terms):

| head | target | loss | why |
|---|---|---|---|
| `imit` (1 logit / candidate) | listwise soft target p\* ∝ exp(−cost_k / T), cost_k = ADE of candidate k to the human over the valid slots of the 8-slot grid (0.5–6 s), **T = 0.5 m** | listwise CE over the masked fan (Hydra-MDP Eq 8 / TNT ψ form, as `refcv8_conditioning.listwise_selection_loss:559-582`, with ADE instead of its speed-profile + heading cost) | the bar is ADE, the loss is 78 % along-track, and a soft target over the whole fan is not SEL-1's one-hot "GT-nearest" rescorer (§3.5) |
| `along`, `cross` (regression) | mean \|along-track\| and \|cross-track\| error of candidate k to GT over the valid slots, `log1p(x / 0.5 m)` | Huber (β 0.1) | the LONGITUDINAL / LATERAL families get an interpretable per-candidate predictor; auxiliary, never in the selection rule |
| `nc`, `ttc` | `refcv7_oracle.oracle_subscores` (`refcv7_oracle.py:251-324`) on the GT agent futures `r7_agent_fut` (`refc_v3_train.py:3743-3797`, dumped by the capture roll) | BCE on the oracle mask (abstain where the join is unlabelled) | safety the imitation term cannot see |
| `dac` | same, on the SAM3 drivable raster (label-only) | BCE on mask | REFe's NAVSIM gap is 4.15 of 6.30 points in NC/DAC-zero tokens (`R8-WHYPOOR`) |
| `comf` | same (analytic kinematics, NAVSIM comfort limits) | BCE | analytic: a head that cannot reproduce it is broken (a built-in check) |
| `ep` | same (progress vs the human's, clamped [0, 1]) | BCE | |
| — | `spd`, `nav` are **NOT** heads | — | both are functions of an input the scorer may not read (§4): the ceiling stays a HARD mask, nav stays with the generator |

Selection rules (registered, §PREREG): **R-IMIT** (primary) = argmax of `imit` over the masked fan; **R-SAFE**
(secondary, reported) = argmax of log softmax(`imit`) + log σ(`nc`) + log σ(`dac`) (PDMS-shaped gates on top of the
imitation term). Masks = `e9_rank` semantics: reach ∧ ceiling ∧ emit with the shipped empty-row fallback.

### 3.4 How it trains: a FROZEN BANKED fan, on-policy for the frozen generator

The generator is frozen (scorer-only, as SFT-4 was), so the fan distribution is fixed and **a banked fan IS on-policy**.
That buys the property that matters most for decision-grade claims: **training replicates cost no GPU** (`H-ESTIM-SEED-1`
needs one), and every arm (E / F / S, both rules) is trained on byte-identical inputs.

1. **Capture roll** (Thor or the A40, GPU): the existing `fan_roll_r8_pf.py` wrapper pattern, extended to also dump the
   detached scene sources (`r7_bev_tokens`, `r7_agent_tokens`, pads) and, on PhysicalAI, `r7_agent_fut*` and the SAM3
   drivable crop at t0 — over a **train-split** window subset (PhysicalAI train4572, never eval139; NAVSIM navtrain logs
   minus the 38 selection-scan logs, minus navtest). **Two inference seeds** (augmentation; the DDIM fan is stochastic,
   `D-REFAV1-SEED-GOAL-MISMATCH` family).
2. **Targets** (CPU): `l7_targets_pai.py` / `l7_targets_navsim.py`, per window, written beside the bank.
3. **Train** (CPU for L7-E / L7-F; dev-box GPU or CPU for L7-S): AdamW, lr 3e-4, wd 0.01, batch 32 windows, early-stop on
   an **episode-disjoint (PhysicalAI) / log-disjoint (NAVSIM) validation slice of the TRAIN bank** (never eval139, never the
   navtrain selection sample, never navtest).
4. **Deploy**: `refc_scorer.select(...)` replaces the argmax in E9 behind a flag (integration diff, §3.6). With the flag
   off the emitted plan is bit-identical.

⚠️ **Known risk, stated.** The capture is on TRAIN windows the generator was trained on, so the training fans are
in-sample (closer to GT than eval fans). REFe's SFT-4 had the same property and transferred (+3.79 on navtest), so this
is a measured-elsewhere risk, not a blocker. It is monitored by the validation slice's own oracle-vs-pick gap compared with
eval139's (reported, never tuned on).

**On-policy in-trainer** (P6-style, the scorer reads the live fan each step) is the alternative. It is not recommended
first: it needs the trainer (owned by another agent today), a GPU for every scorer replicate, and it adds nothing while
the generator is frozen.

### 3.5 Why the PhysicalAI imitation head is not the refused SEL-1 rescorer

`refc_selector_targets.py:13-23` refused `refc_rescorer.py` (REF-C v1.2) because it was "trained toward the fan's own
GT-distance winner" — a one-hot on the argmin, which is an ADE head "that cannot be better than the fan's oracle by
construction". Three differences, each a guard:

1. **The bar here IS the fan's ADE.** Being bounded by the oracle (0.084 m) is no constraint at a pick of 0.299 m.
2. **Soft listwise target over all 149** (T = 0.5 m), not the one-hot winner: candidates 0.1 m apart get nearly equal
   mass, so the winner's-curse noise in the argmin is not what is learned.
3. **Safety heads beside it, never pooled into it**, and an R-SAFE arm that reads them — the four-family rule is about
   not hiding a regression behind ADE; §6 reads every family.

### 3.6 Integration (delivered as `integration.diff`, not applied — the trainer is owned by another agent)

* `refc_v3.py` scene hook: capture `r7_*` sources when `self.l7_scorer is not None` as well as when `refcv7_wta` is.
* `refc_v3.py` E9 (`:2631-2638`): `if self.l7_scorer is not None and self.l7_select: idx = refc_scorer.select(...)` on
  the same `e9_rank` masks; `sel_idx_e9` kept beside `sel_idx` / `sel_idx_base` for the paired read.
* `refcv8_bridge.run_model8`: expose `anchor_traj` and the scene sources so the NAVSIM capture needs no second forward.
* A loader for a trained scorer state dict (`--l7-scorer <pt>`), stamped into `config.json` with its sha256.

---

## 4. (e) Binding admissibility — every input, what it is computed from

| input | computed from | verdict |
|---|---|---|
| candidate trajectory (detached) | the generator: camera + v0 + nav + fed ceiling | **the object scored**; detached so the scorer cannot train the generator |
| `v0` | IMU-measurable speed at t0 (the PI's 09-02 / 10-18 rulings: v0 / a0 / yaw-rate at t0 admissible) | **admissible**. a0 and yaw-rate are admissible **only from the IMU**; refcv8's `config.ego` derives curvature and yaw rate from **steer** (`tan(steer)/2.9`), which is NOT admissible, so L7 reads **v0 only** |
| BEV tokens, agent tokens (detached) | the camera through the trunk and the perception heads | **admissible (vision)** |
| model per-candidate scores (L7-F) | the model's heads, i.e. camera + v0 + nav + ceiling | **declared inherited channel**: adds no input the generator did not already read; the nav / ceiling content is why L7-E (no model scores) is the floor every visual claim is read against |
| nav command / nav args | **route-derived; on PhysicalAI the route is the ego's own logged future** (CLAUDE.md: "A supplied route is optimistic by construction on PhysicalAI") | ⛔ **NOT an L7 input.** The imitation target is derived from the same future path, so a nav input would carry part of the target's source. Nav stays MANDATORY for the generator (PI 2026-09-16) and reaches L7 only through the fan |
| fed set-speed ceiling (`v_max_ms`) | on PhysicalAI, the quantised **max of the ego's actual speed over [t0+2 s, t0+6 s]** (`max_speed_input.py:12-21`) | ⛔ **NOT an L7 input** (future-derived; contains the imitation target's source). It stays exactly what the PI ruled it is in the shipped plan: a HARD cap (`ceil_keep`), applied identically to every arm |
| `gt_max_speed_6s`, `g`, agent futures, SAM3 raster, PDM sub-scores | labels | **targets only**; `code/l7_targets_pai.py` asserts that no target-side key is ever passed to `refc_scorer` (`FORBIDDEN_INPUT_KEYS`, unit-tested with a mutation) |
| situation classifier output (any form) | — | **not read anywhere**; no tensor in §2.3 that L7 reads descends from `situations.py` |
| `z_tac`, `cond`, tactical goal logits | nav + ceiling + ego mixed into the pooled image | ⛔ **NOT read** (contain nav and the ceiling) |

**Leak test, generalised** (CLAUDE.md "for ANY head, ask whether its inputs include something the label was derived
from"): `imit` / `along` / `cross` / `ep` are derived from the GT future path → L7 may not read nav or the ceiling (both
derived from that future on PhysicalAI) ✓. `nc` / `ttc` from GT agent futures → L7 reads detected agents at t0 only
(the agent head's output on the current camera), never futures ✓. `comf` from the candidate + v0 → computable at
inference by construction; that is a known-value control, not a leak.

**NAVSIM note.** NAVSIM's `driving_command` is a standard benchmark input and the PDM targets are computed from the
route centreline, not from the command. L7-NS nevertheless follows the same rule as L7-PAI (no command input), matching
REFe, whose scorer has no ego / command input (§1.1). A command-conditioned scorer is a separate, later arm.

---

## 5. (d) Cost — ESTIMATED unless marked MEASURED

### 5.1 PhysicalAI

| step | basis | cost |
|---|---|---|
| capture roll, train split | MEASURED fanceil s1: **4,754 windows in 1,346.5 s** on Thor (`runs/s1/fan_roll_s1.json`) = 0.283 s / window. Train4572 at eval139's density (34.2 windows / clip) ≈ 156 k windows; **subsample to 1 window in 4** (≈ 39 k windows, 2 s stride) × **2 inference seeds** | **≈ 6.2 Thor GPU-h** (2 × 39 k × 0.283 s). Scene-token dump adds I/O only. ⚠️ ESTIMATED: the train-split frame cache read rate on Thor is not measured for this wrapper |
| bank size | fan 149×8×2 fp16 = 4.8 kB; BEV + agent tokens ≈ 64–128 tokens × ≤ 256 × fp16 ≤ 64 kB; agent futures + SAM3 crop ≈ 10 kB | ≈ 79 kB / window → **≈ 6 GB** for 2 × 39 k (ESTIMATED; the capture prints the real shapes) |
| targets | **MEASURED (P4a)**: geometry + `comf` / `ep` on all 4,754 eval139 s1 windows × 149 in **0.71 CPU-s = 1.0 µs / candidate, 0.15 ms / window** (`smoke/TARGETS_PAI_S1_FULL.json`); NC / TTC add a circle-cover collision test per (candidate, agent, slot), NOT measured here (no agent futures in the bank) | ≈ **12 CPU-s** for 78 k windows without NC / TTC; NC / TTC ESTIMATED ≤ 100× that (≈ 0.3 CPU-h) |
| train L7-E / L7-F | 154–155 k params, 78 k windows × 149 | **MEASURED 1.0–1.5 ms per window-step on CPU** (batch 32, 4 threads, synthetic bank, `smoke/train_synth_{E,F}/train_log.json`) ⇒ ≈ 80–120 s / epoch, ≈ 1 CPU-h for 30 epochs |
| train L7-S | ≈ 1 MFLOP / candidate fwd (0.46 M params + cross-attention) ⇒ ≈ 0.45 GFLOP / window fwd+bwd ⇒ ≈ 35 TFLOP / epoch | ≈ 10–15 min / epoch on CPU, < 2 min on one GPU; 20 epochs + 1 replicate ≈ **0.5–1 dev-box GPU-h** or ≈ 10 CPU-h |
| eval | splice + battery lane on CPU (the picker lane, `picker/code/build_picker_trees.py` + `pk_lane.py`) | ≈ the picker lane's cost; **0 GPU** (the eval fans already exist: `fanceil/runs/s0`, `s1`) — ⚠️ **L7-S needs eval139's scene tokens**, which the fanceil roll did NOT dump ⇒ one more eval139 capture roll per seed, **≈ 2 × 1,346 s ≈ 0.75 Thor GPU-h** |

### 5.2 NAVSIM (after `H-NT-DECIDE-1` yields A1)

| step | basis | cost |
|---|---|---|
| capture roll of A1 on navtrain (log-disjoint from the 38 selection logs and navtest), 30 k tokens × 2 seeds | PhysicalAI rate 0.283 s / window (same model; the NAVSIM stitch input may be slower) | **≈ 4.7–8 Thor GPU-h** (ESTIMATED range: 0.283–0.48 s / token) |
| PDM targets, 149 candidates | MEASURED REFe: **1.78 CPU-s per 64-proposal set** (PDM) and **0.425 CPU-s per token for the parts** on the dev box (`HANDOFF_RL_S1.md:123`, `P6_ON_11VCPU.md:88`); PDMSimulator + PDMScorer scale linearly in proposals ⇒ 1.78 × 149 / 64 = 4.1 CPU-s + 0.85 (all-in parts) ≈ **5 CPU-s / token** | **≈ 83 CPU-h** for 60 k token-draws ≈ 4 h on 20 workers (ESTIMATED; P4b could not MEASURE it on this box, see REPORT) |
| train L7-S | as above | ≈ 1 GPU-h |
| navtrain-sample read | 2,089 tokens × 2 seeds through the bridge with `--l7-scorer` | ≈ 0.4 Thor GPU-h + harness CPU |
| navtest | **at most one read, after the PI's go** | ≈ 1.2 GPU-h + harness |

**Total to a decision-grade PhysicalAI read: ≈ 7 Thor GPU-h + ≈ 1 dev-box GPU-h + ≈ 15 CPU-h. NAVSIM after A1:
≈ 6–9 Thor GPU-h + ≈ 85 CPU-h.** Both need PI compute (named blockers in REPORT).

---

## 6. (f) Evaluation plan — four families, ADE, PDMS sub-scores, controls

### 6.1 Arms on the SAME fan (same capture, same windows, same masks; only the index differs)

| arm | index |
|---|---|
| OWN | `sel_idx` (the shipped E9 pick) |
| BASE | `sel_idx_base` |
| MBR32 | the frozen picker cell (`FROZEN_PICKER_CELL.md`) with the model's own `e9_keep` |
| CONST (control) | a constant scorer ⇒ deterministic tie-break = **lowest emittable index** after masks; must reproduce `refc_scorer.select` on a constant input exactly |
| RANDOM (control, T0) | analytic mean over the masked fan, + 12 seeded draws |
| ORACLE (control, T0) | argmin GT ADE over the masked fan |
| **L7-E** | floor |
| **L7-F** (R-IMIT, R-SAFE) | |
| **L7-S** (R-IMIT, R-SAFE) | |
| L7-*r | training-seed replicate of each L7 arm (mandatory before any lever claim) |

### 6.2 Metrics, each per family, never pooled, each with the paired episode-cluster bootstrap (PhysicalAI: `taniteval/ci.py`, 139 episodes, B 2000, seed 0; NAVSIM: `navsim_bars8.pair_record` → `paired_log_cluster_bootstrap`, log_name + nuplan_drive)

| family | PhysicalAI (eval139, via the battery lane on the spliced tree) | NAVSIM |
|---|---|---|
| ADE | S2 (0.5–2 s, 4,754 windows), S6 (1–6 s) | ADE over the 8 NAVSIM poses |
| LONGITUDINAL | speed MAE, along-track error, **distance keeping / lead TTC** (battery's lead-agent instrument; n stated) | speed MAE, along-track, accel; **distance keeping UNAVAILABLE (no lead track), n = 0** |
| LATERAL | heading, curvature, yaw rate, cross-track | heading, cross-track, pair-masked yaw rate, curvature |
| TACTICAL | selected vs executed manoeuvre (the battery's TAC controls; lat / lon confusion, 5-way); **the lateral head's constant control** (`tactical-head-needs-constant-control`) | trajectory-derived lateral / longitudinal decision agreement |
| STRATEGIC | **UNAVAILABLE, n = 0**: PhysicalAI carries no route / map signal (CLAUDE.md, five probes) | **UNAVAILABLE, n = 0**: NAVSIM scores no strategic decision |
| PDMS | — | PDMS and NC, DAC, EP, TTC, C, DDC; the DAC-zero cell (`D-REFE-ZEROTAIL-3`) |
| scorer-internal | per-head AUC on held-out windows (`nc`, `ttc`, `dac`), Spearman(imit, −ADE) within window, pick rank median, pick = oracle % | per-head AUC, within-token |

### 6.3 Required controls (each must read its known value or the read stops)

1. **CONST** reads `select()` on a constant score = lowest emittable index (literal, unit-tested).
2. **ORACLE**: a scorer whose `imit` logit = −ADE selects the T0 oracle on every window (literal: 0 mismatches); its
   ADE equals `FANCEIL_R8.json`'s oracle-in-fan at 1e-6 m on the same seed.
3. **Identity splice**: splicing `OWN` through the L7 path reproduces the banked SHIP-R8 tree exactly (K8 of R8-PICKER).
4. **Deliberate regression**: a scorer with its `imit` sign flipped must read **worse than RANDOM** (separated).
5. **Ego-scalar floor**: L7-E is reported beside every L7-F / L7-S number; "visual" claims are read as L7-F/S − L7-E.
6. **Leak guard**: `FORBIDDEN_INPUT_KEYS` refusal with a mutation that passes `v_max_ms` ⇒ RED.
7. **Replicate**: L7-*r (training seed 1) beside L7 (seed 0); the lever claim needs both separated (H-ESTIM-SEED-1), and
   the effect must exceed the inference floor (s0 vs s1 of the same arm).
