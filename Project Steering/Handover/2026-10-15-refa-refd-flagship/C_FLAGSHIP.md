# C — THE FLAGSHIP LINE (v1 → v7F) AND THE PROPOSED v8F

> **Source of truth:** landed tip `460508ed2f5e120667859e637ac233c28a48d90b` (2026-10-15 21:54 +0200), read
> through `git --git-dir=/c/Users/Admin/tanitad-push/.git show <tip>:<path>`; `MODEL_REGISTRY.md` at tip =
> `/c/lgs/reg_tip.md`. Nothing on the G: working tree was read. Abbreviations used below:
> **`P4V`** = `products/P4-training-pipelines/2026-09-27-v7f-refav1-state` · **`ARI`** = `TanitAD Research Lab/Architecture & Inference/Research`
> · **REG** = `Project Steering/MODEL_REGISTRY.md` · **GAC** = `Project Steering/GOALS_AND_CLAIMS.md`.
> **Evidence classes:** MEASURED (+ artifact) · PUBLISHED (+ key) · INHERITED (another doc, not re-verified) · ESTIMATED · HYPOTHESIS.
> **Tiers:** T0 = teacher-forced WM diagnostic, NEVER driving · T1 = self-action / action-closed, OPEN LOOP on logged frames
> (primary for capability) · NavSim = open-loop plan scoring (T1-family). Nothing in the flagship line has ever been closed loop.
> ⚠️ "v8F" does not occur anywhere in the repo text. Everything in §C.6 is a PROPOSAL (HYPOTHESIS), not a registered spec.

---

## C.1 Status — one paragraph

**The flagship has not trained a model since 2026-08-11, and v7F has never been trained at all.** The last
trained and evaluated flagship artifacts are v5f-30k plus the stage-A predictor repair, in the v5.8f assembly
(REG §1.8, §1.13c, §1.14; T0 / T1-diagnostic). The deployed v1 (`flagship4b-speedjerk-30k`) has a full-set
ADE@2 s of 0.4271 [0.3675, 0.4871]. That number is a **T0** world-model fidelity number: the model is given the expert's
true future actions. At T1 the v1.x lateral skill turned out to be an **action echo**: S-curve reproduction is
0.9785 open-loop and 0.0430–0.0538 self-action, and the hold-action control reproduces 0.0 % (REG §1.12).
v6F was retired by the PI on 2026-09-20: *"we will go directly to v7f"* (`PI_DECISION_QUEUE.md:1466`). v7F has
**no checkpoint, no run directory and no registry row** (`D-V7F-NEVER-TRAINED`, GAC:7243). The PI's binding
training hold of 2026-08-31 (`V7_LAUNCH_GATE.md:3`) has **no recorded release**. That file was last changed
2026-09-02. The PI's 2026-09-27 words *"close P3, solve the rest"* and the R1–R6 directive exist only in
`P4V/` and in auto-memory, not in GAC, the PI queue or `V7_LAUNCH_GATE.md` (MEASURED: git grep of the tip).

The whole R1–R6 v7F build is **unlanded**. That covers the merge with 4+2 nav fixes, the `v7f` launch-gate profile, the
`v6_chain --profile v7f` ladder and the exact-resume fix. It sits as source copies under `P4V/`. The tip blobs of
`train_v6_staged.py`, `v6.py` and `v6_chain.py` are **byte-identical to `c36b6ddd`**
(`91b675a7…`, `3d6e2235…`, `fd6f0c7c…`, MEASURED by `git rev-parse`). Their last commits are on 2026-09-06, and
`launch_gate.py` has profiles `refc / refcv6 / refcv7 / refcv8` only.

**The most important fact for v8F:** the programme's driving results now live on the **REF-C chassis**:

* refcv7-50,400 beats its echo control at 6 s by −0.98 m ADE, separated (T1).
* At 2 s it does NOT beat the echo control.
* It beats STOP on NavSim navtest.
* It **loses to STOP on navhard**.
* refcv8 is training on Thor, ETA ≈ 2026-10-20 (ESTIMATED).

The "beats doing nothing" bar has also moved. A **learned kinematic MLP** (P10) scores 6 s ADE 2.93 m against
5.36 m for the echo on the refav1 grid. **No REF-C or flagship arm has ever been read against it** (MEASURED absence: git grep
for `kin_mlp|damp50` hits only refav1/`P4V` paths).

---

## C.2 Lineage v1 → v7F

All rows before §1.12 predate `EVAL_DOCTRINE.md` (2026-08-06). Where a row carries no tier stamp it is marked
UNSTAMPED. v1.x rows use the parity corpus `physicalai-train-e438721ae894`. They are **not comparable** with the
non-parity refcv7/8 corpora or with the eval139 grid.

| # | run (REG §) | status | headline (class · tier · estimator) | verdict | what it taught |
|---|---|---|---|---|---|
| v1-ns | `flagship4b-phase0-30k` (§1.1) | killed at 22,950 by the 2026-07-14 speed reset | the 2.918 m no-speed control (CLAUDE.md, INHERITED) | ablation control, **not** the deployed v1 | speed as an input is causal |
| **v1** | `flagship4b-speedjerk-30k` (§1.2) | ✅ DEPLOYED, step 29,999, ~53 h A40 | `ade_0_2s` full-set **0.4271 [0.3675, 0.4871]** (40 ep, episode-cluster bootstrap); heldout split-mean 0.4522 (MEASURED `taniteval/results/driving_flagship-30k.json`) · **T0** | the deployed model; 263,442,838 params | a T0 fidelity number, never a planning number |
| v2 | `flagship4b-v2-30k` (§1.3) | ❌ abandoned at 7,800 | — | superseded by v3enc | — |
| v3enc | `flagship4b-v3enc-30k` (§1.4) | ⏹ stopped at 10,800 | 10 k gate `RESTART` (one secondary failed) | restart budget 1/2 on `encoder-grounding` | gates via `run_gate.py` |
| v1.6-ab | `flagship-v16-ab-ft` (§1.4b) | ✅ 5,999 | **TIED** with v1; the "best in program" claim is retracted (C1) | tie | quote the eval harness, not trainer val |
| v4 line | v4 → v4.1 → v4.2 → v4.2b (§1.5) | killed / superseded | v4.1@10k `ade_0_2s` **0.8522**, CI [0.7468, 0.98] vs bar 0.60 (MEASURED `Gates/flagship-v4-gate-10k-2026-07-23.json`) · UNSTAMPED | formally INCOMPLETE, substantively **FAIL** | joint planner+WM training starved the planner; the WM stayed healthy |
| v2corpus | `flagship-v2corpus-30k` (§1.7) | 🟢 "RUNNING" since 07-25, **UNVERIFIED / stale** (the 2026-08-16 sweep found no final row) | none | unknown | an unfinished row; do not wait on it |
| v1arch | `flagship-v1arch-v2bal-30k` (§1.9) | ✅ 29,999; HF `Sayood/tanitad-flagship-v1arch-v2bal` | first complete four-family block; `ade_mean_4wp` **0.5752 [0.5370, 0.6142]** (290-clip episode-cluster bootstrap, MEASURED) · OOD-val | complete; no head key (canary-only in `eval_flagship_v4.py`) | the v1 architecture on more data |
| v5f | `flagship-v5f-w120-30k` (§1.8) | ✅ 30,000 (2026-08-09); HF `Sayood/tanitad-flagship-v5f-w120` | selected ADE@2 s **0.4011**, oracle-in-fan **0.1975**, sel_gap **0.2036** (881 windows) · **T0** | w120 geometry, not comparable to 256² | the **selector leaves half the fan's quality** |
| v1.6-uni | `flagship-v16-unicycle` (§1.10) | ✅ 2026-08-06 | unicycle readout on the frozen v1arch trunk | — | ⚠️ "v1.6" names two different runs |
| v1.7 | `flagship-v17-speedloss` (§1.11) | ✅ pre-registered run 6, outcome **B** | P1 decel ratio 0.1547 vs ≥ 0.40 ❌; ADE 0.2849 vs 0.3398, CI-better · **T0** | hypothesis **REFUTED** | a speed loss on a frozen trunk does not fix the decel response |
| §1.12 | v1.6 / v1.7 self-action | MEASURED 2026-08-06; relabelled OPEN LOOP 2026-09-02 | ADE 0.3398→**0.4714** (v1.6), 0.2849→**0.4616** (v1.7); CV floor 0.5352; S-curve **0.9785→0.0538 / 0.0430**; hold-action **0.0 %** · **T1** | **lateral skill = action echo** | the founding failure. ⚠️ The `+7.1642` flagship-minus-refc-base contrast is 98.75 % "the flagship loses to a straight line" (`cl_ha0` decomposition) |
| v5.8f | wedge ladder, E4.4, stage-A, assembly (§1.13–1.14) | MEASURED 2026-08-09…11 | stage-A: lateral gain 0.27→0.971, lon sign →1.0 (**T1-diag**). v5.8f selected ADE **0.4815** vs v5f 0.4011 (+0.08 worse), oracle **0.1077**, accel MAE 0.515 vs 8.10 · **T0** | last trained flagship assembly; W7-on-repaired failed by INSTRUMENT COMPOSITION | the action interface can be repaired at head scale; selection remains the gap |
| v6F | `v6F-SW-30k` (§12; `train_v6_staged.py`) | **RETIRED** 2026-09-20 by the PI | the 40:1 pool is **not** the bottleneck. Through the same pool DINOv2-B reads `lead_gap` **0.44997** vs ours **0.00496** (91×) · **T0-diag** | encoder is the constraint | ⇒ a trained-from-scratch ViT trunk is the weak part |
| v7-tiny | `rdw8p30k`, `splitp30k`, `champ30k`, census (§13) | ~19 M proxies, S-W only, every planner objective 0 | `rdw8p30k`: participation 25.58, predictor cos **0.6224**, scene content `n_agents` +0.0777 (GAC:359). `splitp30k`: `n_agents` **+0.3864** from `z_t` (GAC:359; REG §13.0d prints +0.3881 on its ENV set), predictor beats the constant floor (nrmse 0.8416) but **adds nothing over `z_t`** (−0.0008/−0.0320/−0.0158 at k=1/3/6). Census: **0 of 32 arms** move nrmse > 0.1 % when actions are shuffled (max 0.0919 %) · **T0-diag** | **dissociation**: no arm both carries and predicts the environment | the WM extrapolates the scene and ignores actions |
| v7-tiny T1 | `emao14_30k`, `o14fut30k`, `…tauramp` (REG ~4724–4742) | MEASURED 2026-08-30 | ade 13.8–14.3 m, cl worse than hold-action on every arm · **T1 but VOID**: decoded through a random-init `step_readout_op` (`D-METRIC-DECODE-FALSE-RESULT`, GAC:8945) | **VOID**; the registry table still lacks a VOID banner (owed) | no admissible v7 T1 read exists |
| **v7F** | `PREREG_V7F.md` (2026-09-03) + R1–R6 build `P4V/` (2026-09-27) | **NEVER TRAINED**; R1–R6 code unlanded | none | — | see §C.3 |

---

## C.3 v7F in depth

### C.3.1 Architecture (MEASURED from source unless marked)

* **Trainer:** `stack/scripts/train_v6_staged.py`, 11,057 lines and 232 flags (`P4V/raw/flag_inventory_train_v6_staged.md`).
  * Model: `stack/tanitad/models/v6.py`. Chain: `stack/scripts/v6_chain.py`.
  * **245.6 M params**, MEASURED in `P4V/STATE_AND_GAPS.md` §3. The prereg's "336.5 M" is stale.
  * A trap inside that count: the O5 EMA teacher, **86,236,544 params, was being unfrozen into the optimizer** (`ARI/2026-09-06-v7f-budget/RESULT.md`). Commits `51dae02`/`982476c` (2026-09-06) address it. Read them before relying on the count.
* **Stages:** from `v6.py:4567 STAGE_GROUPS` (INHERITED from `P4V/V7F_PLAN_R1_R6.md` §1; the line is valid at tip because the blob is identical).
  * **S-W** = `encoder, readout, predictor_op, aux` (the world model).
  * **S-T** = `layer_tac, planner`.
  * **S-S** = `layer_str`.
  * **S-J** = everything trainable.
  * Under R6 the plan is S-W → S-T. S-S is skipped, and any joint phase must exclude `layer_str`.
* **Encoder:** ViT 768×12×12, 86.1 M params.
  * It is initialised from DINOv3 ViT-B/16 by transplant and is **TRAINABLE**, at LR ×0.1 with an MSE anchor to a frozen copy of the seed.
  * The trainable encoder is a register decision: E-DEC-64 measured freezing as DEGENERATE (nrmse 0.9301 vs 0.8115), GAC:365.
  * ⚠️ **The transplant loses DINOv3 content.**
    * E-SEED-2 at L/16 (GAC:227): `dino_hf − seed_asis` `n_agents` **+0.2050 [+0.0896, +0.3233]**, separated.
    * S0 at B/16 (`ARI/2026-09-27-v7f-s0-b16/RESULT.md:36-40`):
      * the ImageNet-norm + zero-`pos` repair is **null**: −0.0053 [−0.1663, +0.1393];
      * the transplant loses a **separated** share of ego-speed content: +0.2331 [+0.0884, +0.3661];
      * the seed as wired does beat random on scene content: +0.1941 [+0.0429, +0.3516].
    * ⇒ `PREREG_V7F` §10 **D1 option A** (wrap the real HF `DINOv3ViTModel`) is the evidence-backed choice. It is still **PI-open**.
* **World model:** the predictor is conditioned on measured ego state, `--cond-param omega_accel_v` = `[yaw_rate, a_long, v/SPEED_SCALE]`.
  * `train_v6_staged.py:9798`; semantics at `:171`, INHERITED via GAC:7261.
  * This is realised motion, not a command: E-DEC-57, r = 0.9988.
  * Recipe core (`PREREG_V7F.md` §3): O5 L1 rollout, O6 SIGReg (32 subspaces, 512 slices), O14 future-observation (k 4), EMA target τ 0.996.
* **Tactical / operative:**
  * S-T emits a goal-conditioned fan: `plan_proj(z_op) + cand_queries ⊕ g_tac_embed → UnicycleEmission(·, v0)`, giving N × 60 × (a, κ) at 0.1 s = 6 s, with the final layer zero-initialised to constant velocity (`v6.py:5766-5795`, INHERITED from `V7F_PLAN_R1_R6.md` §3′).
  * Selection uses `GoalDistanceScorer` (`v6.py:2710`) or `MLPCandidateScorer` (`v6.py:2809`).
  * **The WM is not the selector.** Roll-consistency selection was +5.98 m worse and is refuted (INHERITED, citing `v6.py:3946-3950`).
* **Selection gate SEL-1:** E-WC2 measured σ/ADE **9.9915 [7.4492, 13.5119]** against a 3.0 refusal line (MEASURED constant in `v6_chain.py:178`).
  * ⇒ every selector launch is refused, and the S-T default is `--selector none` (`v6_chain.py:43`).
  * The committed unblock is the E-WC2-SW latent dump at the S-W→S-T boundary (`v6_chain.py:224`), which needs a trained S-W.
* **Option B = a configuration, not a build:** `--n-candidates 1 --proposals query --selector none` (`train_v6_staged.py:9271/9274/9298`).
  * This gives one decoded 6 s trajectory with no selector, so SEL-1 does not apply.

### C.3.2 Data (as designed)

* B1 corpus `physicalai-b1-w120-256x640cyl`: 4,572 train + 141 eval (`D-CORPUS-B1`, `ARI/2026-09-05-v7f-state/RESULT.md:60`).
* v7.2 labels: `s2_labels_v7.2_train.jsonl.gz`, 4,572 records, md5 `0ff90213…`, as used by the chain dry-run.
* v8 max-speed sidecar: `refcv6_speed_max_v8_{train,eval}.jsonl`, `limit_kmh ∈ {30,50,100,120}`.
* nav: `nav.jsonl.gz`.
* ⛔ `BACKLOG.md` R8: the eval-clip exclusion in the v7 B1 path is a launch blocker. The 141 eval clips sit inside the 4,713-clip cache. Its status in the merge is UNVERIFIED.
* ⚠️ The v9 per-frame labels (refcv8's) were never wired into the v6 trainer.

### C.3.3 What was trained / evaluated

* **Trained:** nothing at v7F scale.
* **Evaluated:** the S0 trunk-seed panels (B/16, L/16), the T0 diagnostics on v7-tiny, and the dry ladders below.
* The CPU dry ladder S-W→S-T (tiny geometry, real data inputs) ran **rc 0 / rc 0** in 11.4 s (`P4V/v7f_chain/RESULT.md`).
* The gate rehearsal on dev-box CPU passed G-LIVE / G-DVB / G-EVAL / G-CKPT.
  * G-HYG FAILs on 4 open config classes.
  * G-CLOCK = ERROR: there is no clock reference for the v7F cache.
  * The token is INCOMPLETE by design (`P4V/v7f_gate/RESULT.md`, `P4V/v7f_ckpt/RESULT.md`).

### C.3.4 Defects found and fixed — all only in `P4V/` (unlanded)

| defect | evidence | fix location |
|---|---|---|
| `--nav-cond` could not START: (1) `synthetic_batch` lacks nav keys → `NavTokenMissing`; (2) `"nav_token": b.get(...)` written after the NavEmitter splat → None at step 1 (`train_v6_staged.py:7314` vs `:7336`); (3) the NavConditioner's 12 params are in no `_GROUP_PREFIXES` group | MEASURED, 4/4 mutants | `P4V/v7f_merge/` §1 |
| ⛔ (4) `build_stack_from_args` **never mapped `--nav-cond`** ⇒ every v6/v7 launch "with nav" at the tip built **no** nav conditioner | MEASURED by the gate's G-DVB | `P4V/v7f_merge/` §1b |
| F7: operative nav got **no gradient in S-W** (the WM-loss helpers never forward nav); F8: `--dry-run` died on nav | MEASURED; G-LIVE flipped FAIL→PASS | `P4V/v7f_merge/` §1c |
| F3b: a v6 resume replays the cosine schedule on top of the loaded LR, so the remainder trains at **(1+cos(πn/T))/2** of schedule; 0.34549 measured at n=3/5 | MEASURED (`P4V/v7f_ckpt/raw/probe_before_after.json`) | `P4V/v7f_ckpt/` |
| F3: resume not exact (RNG + data position) | MEASURED; now bit-identical, model 259/259 and 261/261 | `P4V/v7f_ckpt/` |
| F1 (open): the tactical nav path reaches the shared nav embedding (X3 isolation) | MEASURED | owner `tanitad/models/nav_conditioning.py`, **unfixed** |
| F2 (open): `nav.jsonl.gz` and the speed sidecar must be md5-consistent, or S-T dies at `build_vmax_join` after the corpus build | UNVERIFIED on Thor | `P4V/v7f_chain/RESULT.md` §6 |
| §9 launch line unrunnable: the LDAD flags do not exist anywhere; `--horizons` default refused; `--w-s2-goal` refused at S-W | MEASURED (`D-V7F-LAUNCH-LINE-UNRUNNABLE`, GAC:7255) | build or strike LDAD |
| S-W cannot answer its own G-DRIVE: the readout is left at random init | from source (`STATE_AND_GAPS.md` §3 #6) | needs a stage that trains the readout |

### C.3.5 R1–R6 compliance (PI 2026-09-27, binding for v7F)

| req | at TIP (what a launch today would get) | in `P4V/v7f_merge` + `v7f_chain` (unlanded) |
|---|---|---|
| R1 max speed input + hard cap | **ABSENT**: 0 hits for `max.speed\|vmax` in `train_v6_staged.py` (MEASURED) | MET: `--max-speed-input-v6 --speed-max-sidecar-v6 --plan-vmax-cap` |
| R2 nav → tac + op, train + inference | **ABSENT in effect**: `--nav-cond` exists (`:9207`) but is not mapped into the model and cannot start | MET after defects 1–4 + F7/F8; F1 isolation open |
| R3 ALL tactical labels → tactical loss | **ABSENT**: no family reaches any loss; S-T does not load the label file (`D-TLIGHT-1`, pinned by `stack/tests/test_tactical_label_reach.py`) | MET behind `--w-tac-label-all 1.0 --goal-multilabel --s2-labels` |
| R4 tactical conditions operative | **PARTIAL**: the goal path feeds the emission (`g_tac_embed`); `--tac-goal-cond` (`:9434`) is strategic→tactical, not tac→op | MET: `--tac-op-cond detached\|e2e`; **the mode is a PI decision (open)** |
| R5 one combined 6 s trajectory | **MET as config**: 60 × 0.1 s emission; option B | MET; `--n-candidates 1` enforced on both stages |
| R6 strategic OFF | **PARTIAL**: S-S can be skipped, but the chain's S-T turns `--tac-goal-cond` ON | MET: `--strategic-off`; the chain refuses S-S/S-J |

---

## C.4 Settled findings — do not re-litigate

| # | finding | evidence (class · tier) |
|---|---|---|
| S1 | **v1.x lateral skill is an action echo.** | S-curve 0.9785 open-loop vs 0.0430–0.0538 self-action, hold-action 0.0 % (REG §1.12; MEASURED · T1) |
| S2 | **Scene → action, not action → scene.** The action's marginal on future scene content is null (+0.0755 t 0.64 / −0.0179 / +0.1083), against a positive control at t 8.5–14.3 (E-DEC-48b). ⇒ reactions belong in the PLANNER, not the WM. | `V7_RECIPE_AND_SCALEUP.md` §2 (MEASURED · T0) |
| S3 | **The predictor is action-deaf on every arm.** | Census 0/32 arms > 0.1 % nrmse change under action shuffle; the action pathway is 2–9 % of the latent pathway (REG §13.0c). `ẑ_GT − ẑ_HOLD` is exactly 0 in 6/9 cells (`D-V7F-L3-ACTION-DEAF`, GAC:7751). 97 % of the conditioning response is the speed channel (`D-V7F-CONDITIONING-IS-SPEED`, GAC:7619). All MEASURED · T0 |
| S4 | **The encoder stays trainable.** Freezing is DEGENERATE (nrmse 0.9301 vs 0.8115). | E-DEC-64 (GAC:365, MEASURED · T0) |
| S5 | **The trunk, not the pool, is the constraint.** | 91× `lead_gap` gap vs DINOv2-B through the same pool (REG §12.1, MEASURED · T0) |
| S6 | **The DINOv3 transplant loses content, and the cheap repair is null.** | E-SEED-2; S0 B/16 (§C.3.1) |
| S7 | **v7F S-T does not select by imagination; SEL-1 gates it.** Option B is a config. | `v6_chain.py:43,178`; `v6.py:2710,2809`; RETRACTION #2 in `P4V/HANDOFF_TO_MASTER_MIND.md` §4 |
| S8 | **An imagination planner over an action-deaf WM cannot steer.** refav1's WM contributes 1.63e-10 of the lateral cost. The pre-registered full-grid `loncomb3 − ha0_ext` is +0.0896 / +0.1515 m ADE at both inference seeds: **FAIL-WORSE**. | `ARI/2026-09-27-refav1-fullgrid-loncomb3/RESULT.md` (MEASURED · T1) |
| S9 | **A frozen trunk adds nothing beyond t0 kinematics** (refav1, pre-registered A2+A3). `P9 − P10` = +0.0054 [+0.0016, +0.0093] @ 2 s and **+0.0987 [+0.0332, +0.1658] @ 6 s** (separated worse). | `ARI/2026-09-27-refav1-trunk-probe/RESULT.md:77` (MEASURED · T1-equivalent · paired episode-cluster bootstrap, 141 eval episodes) |
| S10 | **A learned kinematic MLP is the strongest "do nothing" bar.** P10 6 s ADE **2.9257** vs linear 3.4281 (−0.5024 [−0.6331, −0.3767]), `kd_x` 3.5584, `damp50` 3.45, `ha0_ext` 5.36. At 2 s vs `ha0_ext`: −0.3495 ADE, −0.9272 FDE. | same file :64-90 (MEASURED · T1-equiv · refav1 grid, 2,399 windows / 141 episodes). PUBLISHED analog: AD-MLP `2305.10430` (banked) |
| S11 | **Damping alone beats the echo by ~1.9 m at 6 s.** `damp50 − ha0_ext` ADE@6 s **−1.9025 [−2.3605, −1.4971]**. About 80 % of the p4 planner blend's gain was damping. | `P4V/HANDOFF_TO_MASTER_MIND.md` §00 (EXPLORATORY, refav1 grid; INHERITED from the full-grid package) |
| S12 | **The listwise selector loss is lateral-blind.** noLW vs V-R8 GT-turn cross 6 s: −4.57 [−8.04, −1.48] / −4.98 [−8.54, −1.64], both seeds. | GAC `PI-R8-NOLW` CONFIRMED (MEASURED · T1 tiny rung) |
| S13 | **Tactical conditioning has not shown a lateral benefit at the tiny rung.** Turn direction noLW −0.009 / 0.000 vs V0. The measured gain is longitudinal (along 6 s −0.31 / −0.34 m). | GAC `H-R8-RECIPE-1` 2026-10-14 (MEASURED · T1 tiny rung) |
| S14 | **No launch without a launch-gate PASS token** (PI 2026-09-26). refcv6 trained three declared-but-unbuilt levers. | memory `launch-gate-binding`; `stack/scripts/launch_gate.py` |

**Refuted levers, each with the next lever it named:**

* LDAD / encoder-shaping on v7F (`H-V7F-1`):
  * not refuted, but **unrunnable**: the loss does not exist;
  * next lever: build it, or strike it before any v7F compute (`D-V7F-LAUNCH-LINE-UNRUNNABLE`).
* Ten WM objectives (O1–O3, O7–O11, O13, PSG):
  * failed as a class (E-DEC-52);
  * next lever: the hierarchy owns reactions (S2), plus a genuine command channel (`D-V7F-COMMAND-CHANNEL`, GAC:7261: refcv4b's (a_lon, a_lat) control vocabulary, HYPOTHESIS, never run).
* The τ-ramp: NEUTRAL at 30 k. The drift arm (P3): CLOSED by the PI 2026-09-27 (`D-V7F-DRIFT-NULL`).
* Heads-only refav1 retrain:
  * withdrawn (S9);
  * next lever: option T, the trunk trained with the trajectory objective (`P4V/REFAV1_RETRAIN_DECISION.md`);
  * blocker: the ~303 GB fp8 train cache is gone from Thor (C82 correction in the handoff).
* The listwise selector:
  * refuted as the recipe (S12);
  * next lever: noLW, which is in the refcv8 launch.
* WM roll-consistency as selector:
  * +5.98 m worse;
  * next lever: goal / learned scorer, or no selector (option B).

---

## C.5 Planned, not finished

| item | spec / package | how far | blocker |
|---|---|---|---|
| v7F S-W→S-T under R1–R6 (option C = B first) | `P4V/V7F_PLAN_R1_R6.md` §3′/§4/§6′ | code merged in a package, 2,655 suite tests pass with the failure set identical to the tip (`P4V/v7f_merge/RESULT.md` §0); CPU dry ladder rc 0 | **unlanded**; PI: release the 08-31 hold, D1 (DINOv3 wrap), the R4 mode; G-HYG ×4, G-CLOCK, F1, F2; a Thor slot |
| `v7f` launch-gate profile | `P4V/v7f_gate/` (52 tests) | rehearsal only | unlanded; G-CLOCK reference; G-SUITE never run |
| `PREREG_V7F` tiny ladder R0→R3 (+LDAD) | `Project Steering/PREREG_V7F.md` §8; `BACKLOG.md` R17 (R0 is 0 GPU) | not started | LDAD code; `--w-trunk-anchor` (`BACKLOG.md` R23) |
| E-WC2-SW latent dump (SEL-1 unblock) | `v6_chain.py:224` | not started | needs a trained S-W |
| v7 training-seed floor | `STATE_AND_GAPS.md` §4 v7F #6 | none; the 14.3 % replicate panel is the REF-C rig (misattributed in CLAUDE.md) | one replicate arm |
| refav1 option T (trunk + trajectory objective) | `P4V/REFAV1_RETRAIN_DECISION.md` | decision FINAL, not built | ~303 GB fp8 cache rebuild + storage (PI) |
| A16: the six frozen-trunk defect classes for v7F | `BACKLOG.md:35`, `ADVISORY_FROZEN_TRUNK_DEFECT_CLASSES.md` | unanswered for v7F | ~1 h, 0 GPU |
| Register the 09-27 v7F findings + PI directives | `P4V/HANDOFF_TO_MASTER_MIND.md` §0, §4 | **not in GAC / PI queue / V7_LAUNCH_GATE** (MEASURED: git grep of "solve the rest" returns only `P4V/`) | doc work |

---

## C.6 PROPOSED v8F (HYPOTHESIS — needs PI authorisation; it changes the flagship architecture)

### C.6.1 The decision the PI must make first

**Option F1 (recommended): build v8F on the proven REF-C chassis (`refc_v3_train.py`) and graft the flagship's latent world model and hierarchy onto it.**

* That chassis has trained twice through the binding gate: refcv7 to 50,400 steps, and refcv8 now.
* It has a working `refcv8` gate profile, the four-family battery, the NavSim bridge and per-frame v9 labels.
* It is the only arm with a separated T1 win over its echo (S6).

**Option F2: train v7F as built (`train_v6_staged.py`, option B).**

* It first needs everything below:
  * landing `P4V/` and closing G-HYG / G-CLOCK / F1 / F2;
  * the PI hold release and D1;
  * an S-W run of ~9 days, ESTIMATED from v6F's MEASURED 26.47 s/step at 30 k steps (CLAUDE.md).
* Its trunk is the transplant that loses content (S6).
* Its v7-tiny ancestors have no admissible T1 read.

**Why F1:** RULE ZERO wants results, and F1's evidence is on its side. Every
measured driving gain in the programme is on the REF-C chassis, and every flagship WM lever has been action-deaf.

**What F1 keeps of the flagship thesis:** the hierarchy (tactical → operative, nav into both, strategic off) and the
WM as an encoder-side scene-evolution learner (S2). **What it drops:** the WM as action-conditioned selector.

F2's unique question, *does a trained-trunk WM become action-sensitive?*, moves to the tiny rig as research and is not on
v8F's critical path.

### C.6.2 v8F definition under F1

| component | choice | why (evidence) |
|---|---|---|
| **chassis** | `refc_v3_train.py` + `refc_v3.py`. Trunk ResNet-101 `a1_in1k` at 416×1024 cylindrical, lift → BEV, 10 cm map head, A9 box head, 117 anchors, per-frame v9 tactical conditioning of generation + selection, `--w-r8-listwise 0.0` | REG §15–16; S12 |
| **warm start** | refcv8 final if it clears BAR-R8-E1/E3/NR, else refcv7-50,400 (`ckpt_50400_full.pt` md5 `d5f104ee…`). New paths zero-gated so step 0 reproduces the parent (refcv8 §3.1 pattern) | `SPEC_REFCV8_DRAFT.md` §1/§3.1 |
| **residual prior** (largest measured lever) | replace NEW-1's `ha0_ext_pose`, which holds a0, κ0 for 6 s and so equals the echo, with a **learned kinematic prior** (P10-style MLP on t0 v0, a0, κ0, fitted on TRAIN) or the **damped prior** (`kd_x` / `damp50`). Keep prior dropout (X2). a0, κ0 at t0 are already consumed by refcv7's prior (PI 2026-09-02: measured state at cycle time is admissible) | S10, S11: echo-type priors are ~1.9 m worse at 6 s than damped ones on the refav1 grid. ⚠️ Transfer to eval139 is UNVERIFIED (experiment E1) |
| **flagship WM (new)** | an operative latent predictor over the pooled BEV/scene tokens. Two-term recipe (O5 L1 rollout to an EMA target + O6 SIGReg, 32 subspaces / 512 slices: the `rdw8p30k` recipe, the only parity-scale arm with participation 25.58 and predictor cos 0.6224), k ≤ 4 at 0.5 s. **Auxiliary only, never a selector**; conditioned on measured state at t0 only | S2–S4, S7, S8; REG §13.0 |
| **hierarchy** | strategic OFF (R6 / PI R5) · nav per frame + turn distance into tactical and operative (R2; **no route checkpoint**, PI 2026-10-11) · ALL v9 tactical labels supervise the tactical layer (R3) · tactical conditions fan + pick (R4) · one 6 s plan (R5) | PI R1–R6 (2026-09-27), R8-1..4 |
| **max speed** (R1) | 4-way road-type label as input (`--r8-speed-input label`, refused outside {30,50,100,120}); **hard cap enforced on the EMITTED plan**, not only the selection filter | PI 2026-10-11 Q-B; `SPEC_REFCV7.md` 26.1: refcv7's ceiling did not reach the emitted plan |
| **selector** | CE-trained, plus a learned **p(lane) / drivable sub-score** trained on map-derived labels (labels/rewards only, never an inference input) | REFe SFT-4: +3.79 [+3.05, +4.54] PDMS from a scorer-only fine-tune (REG §14.2, MEASURED · NAVSIM T1); refcv7 navtest has 2,624 NC/DAC-zeroed tokens vs STOP 732 (REG §15.4.1) |
| **params** | ~100.6 M (refcv8, MEASURED) + WM predictor ~10–20 M (ESTIMATED) ⇒ well under the sub-300 M budget | REG §16 |

### C.6.3 Training stages

* **Stage 0** (0 GPU, dev box): E1 below (floors on eval139).
  * Fit the P10 prior and refuse any prior that loses to `damp50`.
* **Stage 1** (tiny rung):
  * Rig: Thor batch 8, ~4.4–4.6 h/arm MEASURED for the refcv8 ladder (GAC `PI-R8-NOLW`).
  * E2 = prior swap; E3 = WM auxiliary.
  * Each arm runs at seeds 0 and 1 and is read against the rung's own training-seed floor `F = max(|V0r−V0|, |V-R8r−V-R8|)`, per `SPEC_WPB_LADDER.md` §4.
* **Stage 2** (full):
  * Warm start, one full pass over the 4,572-clip train view: refcv8 used 48,852 steps at batch 16, 781,635 windows (REG §16).
  * Fresh optimizer, re-warm 1,000.
  * Eval every 500 steps; gradient-share readings as in refcv8.
* **Stage 3**: a **replicate** at training seed 1 for any claim of a lever effect (`H-ESTIM-SEED-1`).

### C.6.4 Data

* Train view `/home/nvidia/data/refcv8-b1-416x1024-train-4572` on Thor, **NON-PARITY**.
  * ⚠️ `SPEC_REFCV8_DRAFT.md:65` still says refcv7's 4,369 clips. REG §16 (4,572-clip view, PI 2026-10-12 "one full pass and add the missing clips") wins.
* eval139 is disjoint by clip id.
* v9 labels: train md5 `f63ece410b725febb8a5242cf2b01d3c`, eval139 `6b5c7f207cffc3b7eb3cd527fd433599`.
* Speed sidecars: `/home/nvidia/data/refcv6_speed_max_v8_{train,eval}.jsonl` (`launch_gate.py:725-726`).
* SAM3 map GT and obstacle boxes as in refcv7/8.

### C.6.5 Committed bars (proposed; register them before any compute)

| bar | definition | floor / baseline (MEASURED) |
|---|---|---|
| **BAR-V8F-K (new, "vision helps")** | `os − P10_kin_mlp` ADE 0–6 s < 0, separated at both inference seeds. Also vs `damp50` and `kd_x`, on S6 and S2, the same windows | P10 refit on eval139: **not yet computed** (E1). On the refav1 grid P10 = 2.9257 m @ 6 s |
| BAR-V8F-E1 / E3 | inherit BAR-R8-E1/E3: `os − ha0_ext` < 0 separated at 0–2 s and 0–6 s, both seeds | refcv7-50,400: S2 −0.0102 [−0.0260, +0.0067] **not separated**; S6 **−0.9804 [−1.294, −0.6815]** (`SPEC_REFCV8_DRAFT.md` §5.1; `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-11-r7base-fourfamily/`, T1 open loop) |
| BAR-V8F-NR | per family not separated worse than the parent (refcv8 final), **and** \|Δ\| > the training-seed floor before calling a lever effect | refcv8 rows pending (training) |
| BAR-V8F-N2 (decisive NavSim) | navhard official two-stage EPDMS > max(STOP, CV, ECHO), LEGAL inputs | refcv7-50,400 LEGAL **0.1987** vs STOP **0.2985** (Δ −0.0998 [−0.1383, −0.0627], FAIL) (REG §15.4.1) |
| BAR-V8F-N1 / N4 | navtest PDMS > STOP and > refcv7 LEGAL 70.2048 [68.55, 71.82], separated | STOP 61.8202 |
| BAR-V8F-LEAD | LEAD-VIOL `share_of_at_risk` ≤ the echo, with `lead_progress_ratio_clipped` ≥ 0.90 | refcv7 0.0590 (s0) vs echo 0.0434, not separated; clipped progress 0.9539 (REG §15.4.4) |
| BAR-V8F-WM | WM predictor beats a constant **and** `ẑ` beats `z_t` (L3) on scene targets, carried with `nrmse_SHUFFLED` | REG §13.0c rule |
| Frontier reference (not a bar) | REFe SFT-4 navtest PDMS **88.25**; human 94.55 | REG §14.2. Trained on navtrain, so not comparable to zero-shot refcv |

Strategic family: **NOT APPLICABLE, n = 0, layer OFF (R6)** in every table.

### C.6.6 Compute and gate

* **Machine:** Thor is the training box.
  * It is occupied by refcv8 until ≈ 2026-10-20 (ESTIMATED).
  * ⛔ Do not co-locate on it: Thor GPU sharing is net-negative, 3 jobs ran 3.9× slower (memory, INHERITED).
* The dev box (RTX 4060, 8.6 GB) takes E1 and the WM-module unit tests.
* Colab takes small-data GPU jobs only. The 340 GB train cache cannot leave Thor at 1.2 MB/s (INHERITED, memory).
* **Full-run cost:** refcv8 MEASURED 8.94 s/step (G-LIVE), 31.26 GB peak.
  * The WM auxiliary adds future-frame target encodes, ESTIMATED +20–45 %, so ~10.7–13 s/step.
  * That is ~6–7.5 days per pass and **above the PI-approved 10.5 s/step cap**, so it needs PI approval or a cheaper target (EMA BEV tokens, k ≤ 2).
* **Gate:** a new profile `PROFILES["v8f"] = refcv8 profile + the WM flags`:
  * G-DVB: every new flag is declared-vs-built;
  * G-LIVE: gradient reaches the WM group and the prior head;
  * G-CLOCK: against the newly approved cap;
  * G-MAP / G-BOX-OVERFIT inherited;
  * one regression arm per new check.
  * The `refcv8` profile is the template (`launch_gate.py:666`).

---

## C.7 Open RESEARCH topics, ranked by measured effect size

1. **Kinematic prior / damping, 1.9 m @ 6 s.**
   * `damp50 − ha0_ext` −1.9025 m; P10 vs linear −0.50 m (refav1 grid, T1).
   * Never applied to refcv7/8, whose prior is `ha0_ext_pose`.
2. **Selection gap, 2.07 m on turns.**
   * refcv7 S-ROUTE REGRET pick − ORACLE-117 = **2.0718 [1.7426, 2.4477]**: a T0 denominator, 2,317 turn windows (REG §15.4.2).
   * v5f sel_gap 0.2036 (T0).
   * Re-ranking by the model's own logits recovers 0.0040 of 0.1630 m (REG §15.3).
3. **Turn failures under tactical conditioning: 4.6–5.0 m cross @ 6 s** (listwise). Resolved by noLW (S12). The lateral benefit of conditioning is still unshown (S13).
4. **navhard vs STOP: −0.0998 EPDMS** (REG §15.4.1).
   * On REFe, zeroed tokens are 4.15 of a 6.30 PDMS human gap.
   * A scorer-only p(lane) fine-tune gave +3.79.
5. **Trunk quality: 91× (v6F) and +0.205 (transplant).** REFe uses DINOv3 ViT-L + LoRA (88.25). A DINOv3-B/16 trunk inside the 300 M budget is untested on the REF-C chassis.
6. **Perception heads.**
   * Edge IoU 0.042, lane 0.164; box AP@2m 0.248 (REG §15.2, trainer-val, not eval-grade).
   * R8-6 levers are in refcv8.
7. **WM action-sensitivity: action pathway 2–9 % of latent.** Structural per S2. Its value to driving is **unmeasured**, which is exactly E3.
8. **Strategic horizon (P4):** label events have a median of 12.5 s, and the strategic layer is off. Deferred, not dropped.

## C.8 Open ENGINEERING topics

1. **Decide F1 vs F2. Then land `P4V/` (merge, gate, chain, ckpt) or formally archive it.**
   * If anything v6-family ever trains: land F3b first. Every v6 resume on the default schedule is LR-corrupted.
   * Land the nav-mapping fix too. At the tip `--nav-cond` is silently dropped.
2. Record in GAC, `PI_DECISION_QUEUE.md` and `V7_LAUNCH_GATE.md`:
   * the PI's 09-27 R1–R6;
   * "close P3, solve the rest";
   * the SEL-1 correction;
   * nav defects 1–4, F7, F8, F3b;
   * P10 as the programme bar.
3. Put a **VOID banner on REG ~4724–4742**, the v7-tiny T1 table decoded through a random-init readout.
4. Implement `kin_mlp` + `damp50` + `kd_x` as battery arms in `taniteval`. The code exists at `ARI/2026-09-27-refav1-trunk-probe/code/probe_te.py::fit_mlp_kin`.
5. Build a `v8f` gate profile (C.6.6).
6. Write a WM-auxiliary module for `refc_v3.py`, with unit tests:
   * a G-LIVE gradient check;
   * a time-shuffled-target deliberate-regression arm.
7. refcv8 owes model-side `BYPASS_DECLARATIONS` for `r8_rc_to_cond` and the strategic modules (REG §16 quoting stamp). v8F inherits this debt.
8. The speed ceiling must act on the emitted plan (`SPEC_REFCV7.md` 26.1).
9. A16: answer the six frozen-trunk defect classes for any trunk change (`BACKLOG.md:35`).

## C.9 Recommended next 3 experiments

**E1 — floors on the deployed surface.**
* **Hypothesis:** refcv7-50,400 `os` beats P10 at 6 s (`H-V8F-FLOOR`, pre-register both outcomes).
* **Cost:** 0 GPU, dev box, hours.
* **Method:**
  * Fit P10 (8 t0 kinematic features, past-only) on TRAIN windows, score on eval139 S2/S6.
  * Use the paired episode-cluster bootstrap against the banked dumps `D:/refcv7_eval_kit/battery/step50400_a6text/dump_s{0,1}`, plus `damp50` and `kd_x`.
* **Bar:** `os − P10` ADE 0–6 s < 0, separated, both inference seeds; families reported per row.
* **Unblocker:** TRAIN-window kinematics and 6 s futures. Pull poses only (MB) from the Thor train view over LAN (~63 MB/s, INHERITED).
* **Fallback:** grouped 5-fold CV inside eval139, as refav1 §1 did.
* **If NEGATIVE:** the v8F critical lever is the prior (E2), and every "beats doing nothing" row in LEADERBOARD gets re-read.
* **If POSITIVE:** the margin is the programme's first measured "vision helps" quantity.
* **E1b:** the same probe on refcv7's pooled BEV features, `P9 − P10`, the trunk-probe design. Inference over eval139 plus a train subset on the 4060.

**E2 — prior swap at the tiny rung.**
* **Hypothesis:** replacing `ha0_ext_pose` with the learned/damped prior improves S6 ADE and LAT cross beyond the training-seed floor F.
* **Cost:** 2 arms × ~4.5 h on Thor (seeds 0/1), read against the existing V0/V0r.
* **Bar:** paired ΔADE 0–6 s < 0, separated and |Δ| > F, both seeds; no family separated worse.
* **Unblocker:** a Thor slot after refcv8, or a PI preemption; G-DVB for the new prior flag.
* **Fallback lever:** a learned prior head trained jointly with prior dropout (X2).

**E3 — the flagship WM auxiliary at the tiny rung (the thesis test).**
* **Hypothesis:** an encoder-side two-term WM (O5 + O6) on pooled BEV tokens improves the planner's S6 four families beyond F. Its predictor must also beat `z_t` with a shuffled-action control.
* **Cost:** ~1–2 days implementation (ESTIMATED) + 3 arms × ~5 h: seeds 0/1, plus a time-shuffled-target deliberate-regression arm that must NOT help.
* **Bar:** BAR-V8F-NR-style paired deltas > F; WM L3 > 0 separated.
* **Unblocker:** PI authorisation of the architecture change and the cost cap.
* **Fallback:** keep the WM as a zero-weight diagnostic head. v8F then rests on hierarchy + prior + selector, and the WM question returns to F2's research rig.

## C.10 Traps specific to this line

* `flagship4b-phase0-30k` is the **no-speed control**. The deployed v1 is `flagship4b-speedjerk-30k`.
* v1's 0.4522 is a heldout split-mean; the full-set value is 0.4271. Both are **T0**.
* "v1.6" names two runs (§1.4b vs §1.10).
* §1.12 was "closed loop" until 2026-09-02. It is OPEN LOOP. Never quote +7.1642 without its `cl_ha0` decomposition.
* ⛔ **At the tip `--nav-cond` is accepted and then not built.** Any "nav-conditioned" v6/v7 claim from the tip code is false.
* ⛔ **F3b:** a resumed `train_v6_staged.py` run trains at (1+cos(πn/T))/2 of its LR. The LR-replay fix is unlanded (`P4V/v7f_ckpt/`).
* `train_v6_staged.py` `step_s` is **already divided**. Do not divide by `--log-every` (CLAUDE.md).
* ⛔ A v7-tiny T1 number is VOID if decoded via `step_readout_op`: random init in 9/9 arms.
* A low `nrmse` does not mean action-conditioned. Carry `nrmse_SHUFFLED` beside it.
* `lead_gap_m` is a superseded target (C150). Use `lead_present` + `lead_range_m`.
* The 14.3 % replicate panel is the **REF-C** tiny rig, not v7. The v7 line has **no** training-seed floor.
* The refav1 grid (141 eval episodes, 2,399 windows) ≠ eval139 (S2 4,754 / S6 3,668 windows). Never difference across grids, corpora (parity vs non-parity) or NavSim backends (CPU vs CUDA).
* refcv7 "beats the echo" is **6 s only**. At 2 s it is not separated.
* LEAD-VIOL: quote the controls from the same denominator as the headline (all-lead vs at-risk). `SPEC_REFCV8_DRAFT.md` §8.7(b) mixes them.
* EOL: `train_v6_staged.py` is LF at tip. R3's stream copy is CRLF and would flip 11k lines. Land only from `v7f_merge/`.
* `pgrep`/`ps | grep` over ssh self-matches. Thor trainer pid is in `train.pid`; kill by that pid only.

## C.11 File index

1. `Project Steering/MODEL_REGISTRY.md` — §1 (flagship), §12 (v6F readout), §13 (v7-tiny), §14.2 (REFe), §15–16 (refcv7/8).
2. `Project Steering/PREREG_V7F.md` — the v7F pre-registration (H-V7F-1, gates, D1–D9).
3. `ARI/2026-09-03-v7f-design/{DESIGN,SPEC}.md` — design + machine-readable spec.
4. `P4V/HANDOFF_TO_MASTER_MIND.md` — 09-27 state, retractions #1–4, asks.
5. `P4V/STATE_AND_GAPS.md` — v7F defects table, hold items P1–P5.
6. `P4V/V7F_PLAN_R1_R6.md` — stage plan, the SEL-1 correction (§3′), option B.
7. `P4V/v7f_merge/RESULT.md` — the complete R1–R6 change set + nav defects 1–4, F7, F8 (**land from here only**).
8. `P4V/v7f_gate/RESULT.md` — the `v7f` launch-gate profile (unlanded).
9. `P4V/v7f_chain/RESULT.md` — `v6_chain --profile v7f`, F1/F2 open.
10. `P4V/v7f_ckpt/RESULT.md` — exact resume + F3b LR bug.
11. `P4V/v7f_r3/RESULT.md`, `P4V/v7f_r1r4/RESULT.md` — superseded stream copies (read-only history).
12. `P4V/LANDING_READY.txt` — per-path landing map, superseded lists.
13. `P4V/raw/flag_inventory_train_v6_staged.md` — 232 flags, 36 levers default OFF.
14. `P4V/REFAV1_RETRAIN_DECISION.md` — option T.
15. `ARI/2026-09-27-refav1-trunk-probe/RESULT.md` (+ `code/probe_te.py`) — P10 kinematic MLP.
16. `ARI/2026-09-27-refav1-fullgrid-loncomb3/RESULT.md` — planner FAIL-WORSE, damped floors.
17. `ARI/2026-09-27-v7f-s0-b16/RESULT.md` — transplant panel at B/16.
18. `ARI/2026-09-06-v7f-budget/RESULT.md` — EMA teacher in the optimizer.
19. `ARI/2026-09-05-v7f-six-problems/STATUS_BOARD.md` — PI's six v7F problems.
20. `Project Steering/V7_LAUNCH_GATE.md` — the binding 08-31 hold, P1–P5 (stale since 09-02).
21. `Project Steering/V7_RECIPE_AND_SCALEUP.md` — v1.x echo, E-DEC-48b, dissociation, recipe.
22. `Project Steering/GOALS_AND_CLAIMS.md` — D-V7F-* rows (≈7225–7760), E-SEED-2 (227), E-DEC-64 (365), PI-R8-NOLW (16673–16740).
23. `Project Steering/PI_DECISION_QUEUE.md` — v6F retirement (1466), DINOv3 role (1218–1311).
24. `Project Steering/SPEC_REFCV8_DRAFT.md` — bars N1–N4, E1/E3/NR, refcv7 four-family baseline §5.1.
25. `Project Steering/SPEC_REFCV7.md` — NEW-1 residual prior, 26.1 ceiling defect.
26. `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-11-r7base-fourfamily/RESULT.md` — refcv7-50,400 T1 battery rows.
27. `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/RESULT.md` — NavSim bars.
28. `ARI/2026-10-04-refcv8-tactical-conditioning/` — W-PB ladder, noLW (`emit_off_diag_2026-10-12/`).
29. `stack/scripts/train_v6_staged.py`, `stack/tanitad/models/v6.py`, `stack/scripts/v6_chain.py` — v7F code at tip (= `c36b6ddd`).
30. `stack/scripts/refc_v3_train.py`, `stack/tanitad/refs/refc_v3.py` — the proposed v8F chassis.
31. `stack/scripts/launch_gate.py` — profiles refc/refcv6/refcv7/refcv8 (`:432`, `:666`); `stack/ops/runs.d/refcv8-r101-s0.argv.json`.
32. `stack/tests/test_tactical_label_reach.py`, `stack/tests/test_nav_v6stack.py` — pins for D-TLIGHT-1 and nav (the latter pinned the defect's line, not its order).
33. `taniteval/taniteval/lead_metrics.py`, `stack/tanitad/eval/refcv7_loader.py` — LEAD-VIOL instrument, refcv7/8 eval loader.
34. `Project Steering/BACKLOG.md` — A16 (:35), R8/R15/R17/R23/R24 (v7F blockers).
35. `Project Steering/EVAL_DOCTRINE.md` — tiers.

**Conflicts found (the registry or a later source wins):**
* (a) `SPEC_REFCV8_DRAFT.md:65` says refcv8 keeps refcv7's 4,369 train clips; REG §16 says a 4,572-clip view.
* (b) The brief's "splitp30k content +0.3864 with null predictor": +0.3864 is GAC:359's `z_t` panel and REG §13.0d prints +0.3881. "Null" means *adds nothing over `z_t`*; it still beats the constant floor (nrmse 0.8416).
* (c) `V7F_PLAN_R1_R6.md` §6′ "R2 ✅" is superseded by defect 4 + F7/F8.
* (d) REG §1.7 v2corpus "RUNNING" is stale (REG's own 2026-08-16 annotation).
* (e) REG ~4724–4742 v7-tiny T1 table lacks the VOID stamp GAC:8945 requires.
