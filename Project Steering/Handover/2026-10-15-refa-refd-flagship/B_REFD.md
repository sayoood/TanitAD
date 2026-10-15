# B_REFD — REF-D (SimWAM-style frozen-prior reference arm): state of the evidence, open work

Source of truth: landed tip `460508ed2f5e120667859e637ac233c28a48d90b` (read via `git --git-dir=/c/Users/Admin/tanitad-push/.git`). No G: working tree was read.
Evidence classes: **MEASURED** (ours, path/command) · **PUBLISHED** (library key) · **INHERITED** (another doc, not re-verified) · **ESTIMATED** · **HYPOTHESIS**. Tiers: **T0** = diagnostic only; nothing about REF-D has ever been measured at **T1** or in open-loop driving.
"MEASURED-tip-2026-10-15" = I exported `stack/tanitad` + the 3 named test files from the tip with `git archive` into a scratch dir (nothing written to the repo), ran on the dev box CPU (`C:/Users/Admin/venvs/tanitad`, `PYTHONPATH` = scratch, `tanitad.__file__` verified = the scratch copy).

---

## 1. Status in one paragraph

**REF-D has never been trained.** There is no checkpoint, no `MODEL_REGISTRY.md` row (`grep -i refd` on the tip registry = 0 hits), no trainer, no data loader, no feature cache, no launch-gate profile (`stack/scripts/launch_gate.py` PROFILES = `refcv6`, `refcv7`, `refcv8` only, `:432-438`, `:666`), no `taniteval` adapter, no pre-registration file, and no `GOALS_AND_CLAIMS.md` hypothesis row. What exists is a 411-line module (`stack/tanitad/refs/refd.py`), 19 test functions, a design doc and paper §12. **The module has no `forward()`**: `RefD` defines only `__init__`, `_pool`, `act`, `assert_no_future_at_inference` (MEASURED-tip-2026-10-15: `'forward' in RefD.__dict__` = False), so the training objective the design describes (multi-horizon future-field loss, `w_future_op/tac/str` at `refd.py:193-195`) is declared and implemented nowhere. Steering docs contradict each other ("REF-D does not exist" `V7_RECIPE_AND_SCALEUP.md:199`, `2026-09-03-v7f-design/DESIGN.md:291` vs "code EXISTS … no trained arm yet" `NEXT_ARMS.md:49`); the source resolves it: **code exists, arm does not**. It has been dormant in steering since 2026-09-03; the 2026-09-02 readiness report *proposed* "declare REF-D not a consumer until a trainer exists" (`Reports/2026-09-02-2300-v7-readiness.md:135`, a default option; the PI's answer is not in the tip — UNVERIFIED). The programme's live compute (INHERITED, auto-memory index, not repo-verified) is refcv8 / REFe, not REF-D.

Three corrections to what the design docs say about the code (all MEASURED-tip-2026-10-15, details in §4): (a) the "172.54 M that ships" is the *full-config* count only — `RefDConfig()` default builds **142,336,274** and `act()` reaches only **7,969,042 (4.6 %)** of the 172.5 M; (b) the strategic/tactical policies and both future predictors are built but **never called** by any method; (c) a tactical-only (strategic-OFF, R6-shaped) config **crashes with `AttributeError`** instead of building.

---

## 2. What REF-D is, and what it is a control FOR

**Construction.** SimWAM's economics (train with a big video prior, delete it at inference) spent on our hierarchy: frozen pre-extracted prior field → `WideAdapter` → three-rate hierarchy → a single flow-matching *generator* policy that emits ONE 6 s control sequence `(a, κ)×60` — "the policy is a generator, not a fan" (no selector, so it sidesteps SEL-1). `REFD_DESIGN.md` §1-2, `refd.py:1-110`.
The sentence that carries the idea: "The prior's size is a TRAINING cost, not a DEPLOYMENT cost." (`REFD_DESIGN.md` §1).

**Role in the PI's thesis (from source).** REF-D is a *reference opponent* in the hierarchy-dominance comparison, not a control for one lever:
- PI staged plan 2026-08-26 as recorded: "then show hierarchy dominance over REF-C/REF-D" (`V7_RECIPE_AND_SCALEUP.md:218`; same sequence in `NEXT_ARMS.md` Stage D, row D-b).
- Thesis restated by the Lab: "multi-hierarchy reasoning/planning over a JEPA-style WM", shown "against REF-C/REF-D" (`JEPA_DRIVING_SURVEY.md:90-91`).
- v7r phase P4: "L5 vs REF-C, REF-D, frozen-DINOv3/V-JEPA-2.1 fallback WM" (`V7R_DESIGN_PROPOSAL.md:161`).
- **But the operative gate row L5 lists only REF-C and the frozen-DINOv3 WM** (`V7_RECIPE_AND_SCALEUP.md:270`), and the v7f design says the L5 comparison is "against REF-C and the frozen-DINOv3 fallback WM only" (`2026-09-03-v7f-design/DESIGN.md:291`). So REF-D is *named* in the thesis text but has been *de facto dropped from the gate* since 09-03. Whether the PI wants it back is an open decision (§9, step 0).

**What axis REF-D separates (HYPOTHESIS — no source states it):** the REF-C line is a supervised one-shot anchor-trajectory model with no action input (stated for refcv3 in `GOALS_AND_CLAIMS.md` D-HF-COMPARABILITY, `:5017-5030`); REF-A v1 = frozen DINOv3 + latent WM + iCEM planner (`MODEL_REGISTRY.md` §2.4, `:2081`); REF-D = *strong frozen external prior + one-shot generative control policy + future-field auxiliary supervision*. It is the comparator that answers "does the dominance come from hierarchy over a self-supervised latent WM, or would a strong off-the-shelf prior plus a good generator get there anyway?". Caveat that matters for the thesis: **REF-D carries its own *declared* 3-rate hierarchy** (operative 0.2 s×30, tactical 0.6 s×10, strategic 1.5 s×4, `refd.py:155-166`), so it is NOT a flat control. As coded, that hierarchy is not wired into the policy (§4), i.e. today REF-D is effectively "frozen field → one pooled vector → flow policy + two linear tactical heads". A dominance claim *over REF-D* is undefined until its information path is.

**Not to be confused with:** `refdac_validation_step30000.json` (and `d6_ref_dac.py`) under `2026-10-04-refcv8-data-audit/D6_navsim_failure_anatomy/` is NAVSIM's **reference-proposal drivable-area-compliance** (`ref_dac`: token→0/1 from the PDM-Closed reference proposal, `d6_ref_dac.py:1-6`). It has nothing to do with REF-D. Likewise `REFDIR` in `rls1_pod.sh` is a shell variable. (Both read at the tip.)

---

## 3. Design summary (`REFD_DESIGN.md`, 227 lines; mechanism only — "No number here is a capability claim")

| item | design |
|---|---|
| frozen prior | **Cosmos3-Edge, 4 B**, pre-extracted, training-time only, never shipped (`refd.py:128-140` `PRIOR_GEOMETRY`). Field `[B, W, N=640, d_enc=1024]`. ⚠ That geometry is **identical to DINOv3 ViT-L/16 on a 256×640 frame** (`refa_v1.py:75-84` `DINOV3_GEOMETRY`: d 1024, 16×40 grid, 640 tokens). Whether Cosmos3-Edge yields such a field, and from which layer/timestep, is **not specified anywhere** (UNVERIFIED; §5, §8-E1). |
| architecture | `WideAdapter` (temporal mix over W=4) → `TokenFieldPredictor` operative (6 layers, 8 heads) and tactical (4 layers on 64 pooled queries) → `StrategicPolicy`/`TacticalPolicy` (optional) → `FlowControlPolicy` (Conv1d, hidden 256) |
| inputs | past frame window `[B,W,640,1024]` + `v0` (current speed). Nothing else. |
| outputs | `controls [B,2,60]` = (a ∈ ±4 m/s², κ ∈ ±0.2 1/m) at 10 Hz → 6.0 s; `lat_logits`, `lon_logits` (factored heads; v7.0 vocab = 8 lat × 8 lon, MEASURED); `cond` (pooled 1024-d) |
| policy | rectified flow on a *control* sequence (not waypoints), zero-init velocity head, 10 Euler steps, `noise ∈ {ou, iso}` declared axis (`refd.py:263-307`) |
| losses (design) | flow loss `w_policy=1.0`; multi-horizon *isolated* future-field loss `w_future_op/tac/str = 1.0/0.5/0.25`; lat/lon label CE implied by the heads. **Only the flow loss exists in code** (`FlowControlPolicy.loss`, `refd.py:284-290`). |
| action conditioning | default **broadcast ("concat")**, token arm declared (E-ACTSTREAM-2 reversal, §5) |
| isolation | `assert_no_future_at_inference()` signature-walk guard (`refd.py:394-411`) |
| vocabulary | `tac_vocab_version="v7.0"` default (`refd.py:145`); heads sized from `tactical_lat_actions(v)`/`tactical_lon_actions_v(v)` (`:355-357`); wired 2026-08-27 (commit `596d6a6`), blast radius pinned 08-28 (`937afe3`) |
| data (design) | the 2,376-episode parity corpus, ~95 k frames (`REFD_DESIGN.md` §5). **Superseded** by D-CORPUS-B1: B1 train 4,713 eps, v7.2 labels 4,572 train / 147 eval (141 with pixels), a new comparability domain (`GOALS_AND_CLAIMS.md:3325`) |
| storage (design) | "~125 GB" at fp16 — **stale**; MEASURED 2026-09-01 for the same 640×1024 DINOv3 geometry on the B1 0.2 s grid: **581.1 GiB fp16 / 290.5 GiB fp8 train (+74.0 / +37.0 GiB eval)** (`V7_LAUNCH_GATE.md`, section "CORPUS DECIDED + THE fp8 PATH MEASURED"). Pooling 640→160 tokens would be ~4× smaller (ESTIMATED). |
| compute (design) | extraction ~1 h on an A40 (ESTIMATED; no Cosmos run has ever happened); training "comfortable on one A40" |
| RL | designed-in hooks only (`rl_ready`); reward = NAVSIM PDM score; the "zero `import navsim`" statement (2026-08-19, Paper §12.4) is **stale** as of 2026-10 (NAVSIM/REFe work exists under `2026-09-20-refe-plan/` — path presence + memory index, INHERITED) |
| pre-registered experiments (as *described*; no PREREG file) | **E-REFD-1** action-only vs +future-supervision on frozen features; **E-REFD-2** OU vs iso noise (must precede any RL); **E-REFD-3** single 6 s horizon vs per-layer horizons |
| comparisons / bars | none quantitative. "No number … until `taniteval` at T1." "Not comparable to SimWAM's 91.5 until NAVSIM exists here." No bar, no hold-action control, no replicate arm specified. |

SimWAM facts the design leans on, re-checked against the banked primary (PDF sha256 prefix `03ce230505c4`, `TanitAD Research Lab/Library/papers/2608.07468_…pdf`, text extracted with `pdftotext`) — **PUBLISHED, lib:2608.07468**: Tab. 2 action-only 86.6 → +video 90.3 → +RL 91.5; Tab. 3 mask bidirectional 90.2 / action→video 90.1 / isolated 90.3; Tab. 4 LTX 88.7, Wan2.1-1.3B 90.2, Cosmos-Predict**2.5** 90.4, Wan2.2-5B 90.3; Tab. 8 (2 s, 2 Hz) 89.9 / (4 s, 1 Hz) 90.2 / (4 s, 2 Hz) 90.3; Tab. 10 flow steps 1→68.9, 5→90.1, 10→90.3, 20→90.2; 6 B params (single-view table row); action DiT 0.21 B→1.02 B = 89.9→90.3 (Tab. 5); input 384×672. **Not found in the PDF text: "32 GPUs"** (a search for GPU gives only an A100 latency note) — the "32 GPUs" in `REFD_DESIGN.md` §1 / `refd.py:4` is UNVERIFIED. Two scope notes: SimWAM **co-trains** a Wan2.2-5B video expert and its Cosmos row is Cosmos-Predict**2.5**, not "Cosmos3-Edge"; and its action expert is a 0.2–1.0 B DiT whereas REF-D's `FlowControlPolicy` is **921,858 params** (MEASURED-tip-2026-10-15).

---

## 4. Implementation status (tip `460508ed`)

### 4.1 What `refd.py` implements (line refs = `stack/tanitad/refs/refd.py`)
- `RefDConfig` `:143-229`; `sanity()` `:205-229` refuses a non-6.0 s rate, `d_state < d_enc`, bad `noise`/`action_mode`.
- `FlowControlPolicy` `:232-307`: `_noise` (OU ρ=0.9 or iso) `:263`, `velocity` `:276`, `loss` `:284` (rectified flow), `sample` `:293` (10-step Euler, `tanh` squash to `a_max`/`kappa_max` `:305-306`).
- `RefD` `:310-411`: builds `adapter`, `operative`, `tactical` predictors (`TokenFieldPredictor` or `ActionStreamPredictor` by `action_mode`, `:328-336`), `tac_queries` + `tac_pool` `:337-339`, **optional** `strategic_policy`/`tactical_policy` only when `cfg.strategic_cfg`/`cfg.tactical_cfg` are set (defaults `None`, `:199-200`; built `:341-347`), factored `lat_head`/`lon_head` `:355-361`, `policy` `:362`.
- **`act(window, v0, *, gen=None)` `:369-392` is the only inference method:** `adapter(window)[:, -1]` → `_pool` (64 learned queries attend over 640 tokens) → `.mean(dim=1)` (`:387-388`) → one 1024-d vector `pooled` → `policy.sample(pooled, v0)` + `lat_head(pooled)` + `lon_head(pooled)`.

### 4.2 What I measured at the tip (MEASURED-tip-2026-10-15, dev-box CPU, scratch export; scripts `count.py`, `count2.py` in the session scratchpad)
| fact | value |
|---|---|
| `tests/test_refd.py` | **21 passed** (19 test functions, one parametrised ×3); with `tests/test_refs_vocab_v7.py` **28 passed** |
| `RefDConfig()` default (v7.0, no brains) total params | **142,336,274** = adapter 2,762,752 + operative 79,779,840 + tactical 54,587,392 + tac_pool 4,198,400 + lat_head 10,248 + lon_head 10,248 + policy 921,858 + `tac_queries` 65,536 |
| full config (`tactical_cfg=TacticalPolicyConfig()`, `strategic_cfg=StrategicPolicyConfig()`, v7.0, concat) | **172,537,378** (adds strategic_policy 7,990,275 + tactical_policy 21,684,493); token mode 172,543,522; v6.0 heads 172,533,278 (6×6 heads). **`REFD_DESIGN.md`'s 172,539,422 is not reproduced** (Δ −2,044 vs v7.0; cause not isolated — UNVERIFIED). |
| params reachable from `act()` | **7,969,042 = 4.6 % of 172.5 M** (adapter + tac_pool + tac_queries + 2 heads + policy). A forward-hook run at real geometry `[1,4,640,1024]` touched exactly `adapter, tac_pool, lat_head, lon_head` (+ `policy.sample`, `tac_queries`). Output shapes `controls (1,2,60)`, `lat (1,8)`, `lon (1,8)`, `cond (1,1024)`. |
| params built but **called by nothing** | `operative` 80.0 M, `tactical` 54.9 M, `strategic_policy` 8.0 M, `tactical_policy` 21.7 M = **164.6 M**. By the isolated-mask logic the two *future predictors* (134.9 M) are legitimately train-only, which would make the *shipped* model ≈ **37.6 M** (172,537,378 − 80,043,008 − 54,850,560 = 37,643,810) — versus "~173 M … what ships … 35× smaller than SimWAM" (`Paper` §12.3, `REFD_DESIGN.md` §1). **Three different "sizes" are in circulation (172.5 M declared / 37.6 M if the predictors are train-only / 7.97 M what `act()` reaches); none is pinned by a test.** The two hierarchy policies (29.7 M) are not train-only by design but are equally unwired. |
| strategic-OFF config | `RefDConfig(); cfg.tactical_cfg = TacticalPolicyConfig()` → **`AttributeError: 'NoneType' object has no attribute 'd_ctx'`** at `refd.py:344-347` (the deref precedes the "matched set" `ValueError` at `:348-350`, which therefore guards only the strategic-only case). An R6-shaped (strategic off, tactical on) REF-D **cannot be built** without an edit — the same trap refav1 documented (`STATE_AND_GAPS.md` R6 row). |

Historical cross-check: `UNIFIED_HIERARCHY.md:21-39` (2026-08-21) built the default config and counted **142,332,174** with "no `strategic`" among `named_children()` — consistent with today's default (142,336,274; Δ +4,100 = the v7 head widths) and with `strategic_cfg=None`. It reads as a defect there because the default never builds the layer; the full-config build does.

### 4.3 What the tests pin (`stack/tests/test_refd.py`, line = def)
rates reach exactly 6.0 s `:43-51`; `d_state ≥ d_enc` `:53`; **one sequence, not a fan** `:61`; controls inside `a_max`/`kappa_max` `:71`; both noise arms build `:81`; OU lag-1 > 0.7 / iso < 0.1 (measured, not asserted) `:93`; default action mode = broadcast `:109`; token arm reachable `:124`; untested mode `add` refused `:132`; factored heads share v6 tuples **by identity** (under a `tac_vocab_version="v6.0"` frame pin, `:24`) `:139`; `assert_no_future_at_inference` passes and **fires on a mis-wired signature** `:153`, `:157`; `act` returns exactly `{controls, lat_logits, lon_logits, cond}` `:166`; 3-D input refused `:175`; prior recorded as FROZEN / never shipped, 4.0 B, "ties" Wan `:187-202`; flow loss finite and back-props `:205`. `test_refs_vocab_v7.py:29-31` pins `RefDConfig().tac_vocab_version == "v7.0"`.

**What no test pins:** any training path; the parameter counts above; the v7.0 head widths; `act()` reachability of the hierarchy; any numeric behaviour of a trained policy; `unicycle_rollout` — the docstrings at `refd.py:24` and `:238` say samples are "integrated through `unicycle_rollout`" but the module **never imports or calls it** (`git grep unicycle` in `refd.py` hits only comments). "Feasible by construction" therefore holds only for the box bounds (`a_max=4`, `kappa_max=0.2`, `:305-306`): there is **no speed-dependent κ cap / friction-circle clip** (pattern to copy: `tests/test_refa_v1_kamm_clip.py`); at 20 m/s κ = 0.2 would be 80 m/s² lateral (ESTIMATED arithmetic, v²κ).

### 4.4 What is missing to train / evaluate it (all verified absent by `git grep` at the tip)
| missing | evidence |
|---|---|
| `forward()` / loss assembly (flow + multi-horizon future + label CE) | §4.2; `w_future_*` referenced only in the config |
| trainer | no `refd*` under `stack/scripts/` (`refa_v1_train.py`, 805 lines, is the template) |
| data loader / frozen-field cache | `refav1_loader.py` reads a DINOv3 cache `[T_c, 640, 1024]` on the 0.2 s grid; **no Cosmos extractor exists** (`git grep -il cosmos3` finds only `refd.py`, `test_refd.py`, a `dynamics_encoder.py` comment; the Cosmos-named scripts are Cosmos-*Drive-Dreams* data loaders). The only Cosmos3 deployment evidence is the 2026-07-20 pilot: Nano (16 B) was **not deployed** (docker-only serving, 81-dependency pin, 34.9 GB; `2026-07-20-cosmos3-vlm-pilot.md`). Edge (4 B) has never been downloaded or run here; no model card is banked in the Library. |
| label join / nav / max-speed plumbing | `refd.py` has zero `nav`/`max_speed` tokens (the only `nav` hits are the word NAVSIM in docstrings); `v7_labels.py:3` names refd as a consumer but nothing calls it |
| launch-gate profile | PROFILES = refcv6/7/8; each profile pins a trainer (`_REFC["trainer"] = "stack/scripts/refc_v3_train.py"`, `launch_gate.py` ~`:158`) and an eval loader. A `refd` profile + `declared_vs_built` entries + the post-run `check_untrained_params.py` are needed. **REF-D today is a textbook instance of the D-REFCV6-CONFIG-BUILD class the gate was built for** (declared-but-never-built: policies, future predictors, `w_future_*`). Per the PI rule of 2026-09-26 (INHERITED, memory index) no training starts without a gate PASS. |
| eval adapter | no `refd` anywhere in `taniteval/`; a one-shot controls generator needs the same special handling as refcv3 (`D-HF-COMPARABILITY`: `t1_eval.roll_closed` assumes an action-conditioned WM) |
| pre-registration + claim rows | `E-REFD-1/2/3` exist only as prose in `REFD_DESIGN.md` §6 (and one comment at `refd.py:82`); no `PREREG_REFD.md`, no GOALS row |

---

## 5. Everything measured that bears on REF-D (none of it is a REF-D result)

| # | finding | class · tier · path | what it does / does not license |
|---|---|---|---|
| 1 | **E-ACTSTREAM-2**: frozen DINOv3 640×1024 fields, 3.4 M-param toy predictors, 3 seeds, 3,277 windows (1,514 train / 1,763 eval), 8 epochs. MSE: `add` 0.037732 · `concat` 0.037956 · `token` 0.038141 · persistence 0.039709. `token−concat` +0.000186 [+0.000152, +0.000222]; `token−add` +0.000409 [+0.000357, +0.000464]; `token−persist` −0.001572 [−0.001912, −0.001245]; all "separated". Token is worst in 3/3 seeds (per-seed MSE in the JSON). | MEASURED · **T0** · `Research/2026-08-19-simwam-analysis/raw/e_actstream_dino.json`; `E_ACTSTREAM_1.md:244-310` | Basis for the `action_mode="concat"` default. **The effect is 0.5–1.1 % of MSE on a 3.4 M toy predictor, not REF-D's 80 M operative predictor, and there is no replicate-arm noise floor, so per `H-ESTIM-SEED-1` a separated CI is necessary, not sufficient** (3/3 seeds in one direction is the only replicate-like evidence). `add` vs `concat` was never tested directly. The "beats persistence" half is the more interesting finding (on v6 cell fields nothing did). |
| 2 | **E-TRUNK-2**: linear (dual-ridge) decodability, 130 episodes / 5,617 frames, episode-disjoint 5-fold, cluster bootstrap. `lead_gap_m` R²: frozen DINOv3 unpooled **+0.4531**, through v6's 4×4 pool **+0.3792**, ego-only control +0.3340, v6F-SW@20k tokens +0.0130 / cells −0.0176; adjacent-lane occupancy AUC DINOv3 .83–.85 vs v6 .50–.59. | MEASURED · **T0** · `E_TRUNK_2_ENV_DECODABILITY.md` §3; `raw/e_trunk2_probe.json` | Supports "a strong **frozen** encoder carries what our trained trunk does not" — **for DINOv3 (303 M), not Cosmos3-Edge**, and for *linear decodability*; the doc itself says "nothing yet about closed-loop control" (`:190-192`). A linear negative is not a negative about learnability. |
| 3 | **E-TRUNK-3** ladder (v6F S-W steps 2k/16k/18k/20k): `lead_gap_m` R² −0.0093 … −0.0176, never separated; participation ratio 3.29→6.94→6.78→4.90 vs DINOv3 40.77 | MEASURED · **T0** · `E_TRUNK_3_LADDER.md` §1-3; `raw/e_trunk3_ladder.json`; pre-reg `PREREG_E_TRUNK_3.md` (committed `548548f` before any number) | Verdict "WORLD B — never acquired"; by the pre-registered rule it "directly supports" REF-D's frozen-prior bet (`PREREG_E_TRUNK_3.md:70`). Scope: one v6F run, trunk only. |
| 4 | **E-TRUNK-1** pooling probe; remark that REF-A v1/v1′/REF-D "all predict in token space (640 × 1024)" so the finding "supports those designs" | MEASURED (probe) · **T0** · `E_TRUNK_1_POOLING.md:115-119` | A design remark, not a REF-D test. |
| 5 | Static residual-seam audit: `REF-D: []` seams in `refd.py` itself; it instantiates the shared `StrategicPolicy`/`TacticalPolicy` | MEASURED (AST scan) · `raw/residual_scale_audit.json:397-415`; `stack/scripts/residual_scale_audit.py:57` | Says nothing about a trained REF-D. |
| 6 | Neighbour sharing REF-D's components (`WideAdapter`/`TokenFieldPredictor`, DINOv3 features): REF-A v1 `refav1-b1-v72-ep3-speed`, open-loop ADE@2 s on the 282-window / 141-cluster v7.2 EVAL split: `ol` 0.4237 < `ha0` (hold-action) 0.5316 [0.4710, 0.5939] < `cl` 0.5474; the deployed planner arm is **worse than constant velocity** (`cl−ha0` +0.0158 [+0.0007, +0.0315]); it emits the CV baseline on 40/40 turn-dense windows (`D-REFAV1-CG-*`). CV floor 0.5352 (`V7_RECIPE_AND_SCALEUP.md:269`). | MEASURED · open-loop · `MODEL_REGISTRY.md:2081-2096` → `taniteval/results/RESULT-refav1-21109-openloop.md` | This is the *bar REF-D has to clear* and the evidence that the iCEM-over-WM planner does not work on these features. It is **not** evidence about a flow-policy generator. |
| 7 | SimWAM numbers in §3 | PUBLISHED · lib:2608.07468 (PDF banked, sha prefix `03ce230505c4`) | The "+3.7 PDMS from co-trained video" does **not** transfer to a frozen prior by assumption (`REFD_DESIGN.md` §3.2). Different benchmark (NAVSIM PDMS, non-reactive). |
| 8 | This handover's tip measurements (§4.2): 21/21 + 28/28 tests green, parameter counts, `act()` reachability, no `forward`, strategic-OFF crash | MEASURED · mechanism only | Pure code facts. |

**Never measured:** any REF-D training run, loss, ADE, four-family metric or T1 number; Cosmos3-Edge extraction time/size/fidelity; flow-policy vs fan vs anchor; OU vs iso; frozen vs co-trained; multi-horizon vs single. Every "~1 h", "~125 GB", "one A40 suffices" in the design is ESTIMATED.

---

## 6. Compliance with the binding requirements (from `refd.py`/tests at the tip; "not yet determinable" where only a trained arm could show it)

Requirements: R1–R6 = PI 2026-09-27 as stated in the brief (refav1/v7F audit method in `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/STATE_AND_GAPS.md:129-139`); nav spec `SPEC_NAV_CONDITIONING_ALL_LAYERS.md` (BINDING, **names refd** as an affected arm, `:11`).

| requirement | REF-D | evidence |
|---|---|---|
| **R1** max speed = input + hard cap | **ABSENT** | `act(window, v0)` takes only the current speed (`:369`); `FlowControlPolicy.velocity` conditions on `[cond, τ, v0/10]` (`:276-282`); the only caps are control-box bounds `a_max`/`kappa_max` (`:305-306`), not a speed ceiling; zero `max_speed`/`vmax` tokens in the file |
| **R2** nav → tactical **and** operative, train **and** inference (also binding through the 2026-08-30 nav spec, all three layers) | **ABSENT** | no nav argument in `act`, no nav token in any module, no `forward` to carry one |
| **R3** ALL tactical labels train the tactical layer | **ABSENT** (architecture PARTIAL) | heads cover only the 8 lat + 8 lon *action* tokens (`:355-361`, measured widths); the v7 *goal* tokens incl. the traffic-light goals are in no REF-D head; `tactical_policy` (which would own the goal vocabulary) is unwired; no loss exists. The `D-TLIGHT-1` trap (labels exist, never reach training) is exactly the risk. |
| **R4** tactical conditions operative | **ABSENT** | `policy` conditions on `pooled` only; `lat`/`lon` logits are computed *beside* it and consumed by nothing; no FiLM/goal path; `operative`/`tactical` predictors uncalled |
| **R5** one combined 6 s trajectory | **MET by architecture, untrained** | `plan_steps=60` @ 10 Hz, one `[B,2,60]` sample, no fan (`:178`; `test_refd.py:61`) |
| **R6** strategic OFF (set for refav1/v7F; refcv6/7 carry it — whether the PI extends R6 to REF-D is UNVERIFIED) | **Default config has no strategic layer, but a tactical-on/strategic-off variant crashes** | `strategic_cfg=None` default `:200`; tactical-only → `AttributeError` at `:344-347` (§4.2) |
| Vision-only inference | **MET (by signature)** | inputs = frozen-field window + `v0` only (velocity at cycle time is the admissible non-visual input per the PI ruling of 2026-09-02, INHERITED from the memory index); `assert_no_future_at_inference` pinned |
| Goal input must not carry the situation classifier's output | **Trivially MET today (no goal input exists); becomes the live design question once R2/R4 are added** | `lat`/`lon` heads and `policy` both read `pooled`; the policy never reads the logits. R4 needs a tactical→operative path that does not leak classifier output into the "goal" (refav1's resolution: goal token → cost, FiLM tactical → operative) |
| Eval = four metric families, T1, hold-action + echo control, paired episode-cluster bootstrap | **not yet determinable** | no eval adapter; no arm |
| Launch-gate PASS before any training (PI 2026-09-26) | **cannot pass — no profile** | §4.4 |
| Corpus comparability | arms on B1 are a new domain, never same-data-compared with parity arms (`GOALS_AND_CLAIMS.md:3325`); eval = v7.2 EVAL split, 141 clips with pixels | — |

---

## 7. Open RESEARCH topics (ranked by expected value; all HYPOTHESIS — none has a result)

1. **Does a one-shot flow-control generator over a frozen field clear the hold-action / CV floor at T1 at all?** Highest EV because it is the whole arm and nothing measured speaks to it. Isolate it with the **DINOv3 field that is already designed and partly built** (REF-D's geometry equals DINOv3's), so "generator + future-field aux vs refav1's iCEM" is separated from "Cosmos vs DINOv3". Bar to clear on the 282-window / 141-cluster EVAL split: hold-action 0.5316 m, CV floor 0.5352 m @ 2 s (§5 row 6), plus a 6 s number the registry does not yet have.
2. **Conditioning bottleneck.** The policy sees one 1024-d vector = the mean of 64 attention-pooled queries over 640 tokens (`:387-388`) and is a 0.92 M Conv1d net, versus SimWAM's 0.21–1.0 B DiT (Tab. 5: 89.9 → 90.3). E-TRUNK-2 shows a *4×4* pool costs DINOv3 16 % of headway R² (+0.453 → +0.379); REF-D's pool is harsher. Test spatial cross-attention conditioning vs the mean vector. (HYPOTHESIS: the policy, not the prior, is the first bottleneck.)
3. **E-REFD-1 as stated** (does the frozen prior recover the co-training gain?) — `action-only` vs `+future-field supervision`, identical otherwise. Needs the missing `forward()`. E-TRUNK-3 says the *prior* matters (WORLD B) but not whether the *future-field loss* adds anything.
4. **Is Cosmos3-Edge a better prior than DINOv3 *for our metrics*?** Before paying ≥0.6 TB of features, repeat the E-TRUNK-2 decodability panel (lead gap, side occupancy, free space) on a small Cosmos extraction. SimWAM's own Tab. 4 Cosmos-vs-Wan gap is a tie (90.4 vs 90.3, no CI) and theirs is Predict2.5, not Edge. Kill if not separated from DINOv3.
5. **R4 inside a generator:** how does the tactical decision condition the operative flow policy (FiLM on `cond`? goal token as a flow condition?) while the goal path stays information-disjoint from the situation classifier? Needs a pre-registered admissibility check, not a convention.
6. **E-REFD-2 (OU vs iso)** — only matters if RL is reachable. The "no NAVSIM" blocker (08-19) is stale (REFe/NAVSIM work exists); re-check before spending anything. Low until topic 1 is positive.
7. **E-REFD-3 (per-layer horizons)** — **not expressible as the config stands**: `sanity()` refuses any rate that misses exactly 6.0 s (`:213-215`, pinned `test_refd.py:48`), yet the arm is "single-horizon (6 s) vs per-layer horizons (2/6/longer)". The guard has to be relaxed with a declared reason. Low EV before 1–3.
8. **Per-layer action-conditioning mode** (concat vs token on the 64-query tactical stack): the §5 row 1 effect is ≤ 1 % of toy MSE; low EV.
9. **Inference-seed floor.** A stochastic generator needs the third replicate (inference seed) per the CLAUDE.md variance rules; refav1's iCEM floor was ≈ 0.30 m ADE (registry/GOALS `D-REFAV1-SEED-GOAL-MISMATCH`, INHERITED) — REF-D's own floor is unmeasured and may be the dominant noise term.

---

## 8. Open ENGINEERING topics (to make REF-D trainable on Thor; sizes are ESTIMATED working days)

| id | item | size |
|---|---|---|
| E1 | **Define the prior-extraction contract** for Cosmos3-Edge (weights, encoder vs DiT hidden state, layer, diffusion timestep, frame conditioning, resolution, output grid, dtype) and smoke it on one A40/Thor box; today `d_enc=1024`/`n_tokens=640` is borrowed from DINOv3. If it cannot be defined cheaply, ship the DINOv3 proxy arm first. | M (2–4 d to a smoke), L for the corpus |
| E2 | **Feature cache at B1 scale:** reuse/rebuild the DINOv3 fp8 cache (MEASURED sizes 290.5 GiB train + 37.0 GiB eval; fp8 round-trip 2.65 % rel. err., cosine 0.99965, `V7_LAUNCH_GATE.md`). As of 2026-09-27 the train cache was **deleted from Thor** (0/4,572 links) and Thor was 89 % full; only the 141-episode eval copy survived on the dev box (`STATE_AND_GAPS.md:139`) — current state UNVERIFIED. A Cosmos field would be at least this size. | M–L (disk-bound) |
| E3 | **Write `RefD.forward()` and the losses:** flow loss (exists), multi-horizon future-field loss through `operative`/`tactical` with the *isolated* mask, lat/lon label CE over v7.2 labels, weights from `w_*`; decide what the predictors read as action (GT controls at train, never at inference). | M |
| E4 | **Trainer `refd_train.py`** on `refa_v1_train.py` + `refav1_loader.py` (0.2 s grid; refuses mis-gridded caches) + `v7_labels.py` + `NavEmitter` + the max-speed sidecar (`refcv6_max_speed`, 4-bin one-hot as in v7F/refcv6). ⚠ Unit trap: the loader's `actions[:,0]` is a **road-wheel steer angle ≈ 2.9·κ** (`refav1_loader.py` docstring; producer `physicalai.py`), while REF-D's channel 1 is **κ in 1/m** — convert at the boundary with the encoder's L = 2.9. | L |
| E5 | **R1–R6 architecture edits:** max-speed input + plan cap; nav tokens at operative + tactical (+ strategic if present); tactical→operative conditioning; tactical goal-token loss over the full v7 goal vocabulary; **fix the strategic-OFF crash** (`:344-347`: raise a clear `ValueError`, or allow tactical without strategic with `d_cond=None`). | M–L |
| E6 | **Wire or delete:** make `strategic_policy`/`tactical_policy`/predictors reachable from `forward`/`act`, or drop them and the 164.6 M of dead parameters; pin one declared **shipped parameter count** by test (172.5 M / 37.6 M / 7.97 M are all in circulation). | S–M |
| E7 | **Launch-gate profile `refd`** (trainer, required flags, `declared_vs_built` entries for every declared lever, G-EVAL loader, G-SUITE) + the `check_untrained_params.py` post-run gate. | M |
| E8 | **`taniteval` adapter** (one-shot controls → integrate with the repo's `unicycle_rollout`/kin conversion → T1 closed loop; hold-action and echo controls; four families; paired episode-cluster bootstrap). | M |
| E9 | **Feasibility guard:** speed-dependent κ cap / friction-circle clip (pattern `test_refa_v1_kamm_clip.py`); make the docstring's `unicycle_rollout` claim true or delete it. | S |
| E10 | **Pre-registration + register rows:** `PREREG_REFD.md` (both outcomes committed first), `GOALS_AND_CLAIMS.md` hypothesis rows, a `MODEL_REGISTRY.md` stub ("never trained"), and one edit to `NEXT_ARMS.md` / `V7_RECIPE…:199` / `v7f DESIGN:291` to kill the "does not exist" contradiction. | S |
| E11 | **Replicate arms** (training seed + inference seed) on the same windows before any lever claim (`H-ESTIM-SEED-1`). | compute |

---

## 9. Recommended next three steps (smallest first), with pass/fail bars

**Step 0 (decision, minutes; PI / Master Mind):** is REF-D still in the dominance comparison? Sources disagree (named in the thesis text, absent from the L5 gate row). If dropped, close it explicitly in `V7R_DESIGN_PROPOSAL.md` / `NEXT_ARMS.md` and stop; if kept, do 1–3.

1. **Make the arm honest on paper and in code (0 GPU, ~1 d).** Fix the strategic-OFF crash; add tests pinning (a) the declared / `act()`-reachable / shipped parameter counts, (b) that tactical-only builds, (c) the v7.0 head widths 8×8; correct `REFD_DESIGN.md` §1 and Paper §12.3 ("ships ~173 M", "32 GPUs", "~125 GB") to the measured values; create the PREREG and register rows (E10, E9, E6 decision).
   *Pass:* the new tests are green on the tip tree; `git grep` shows no remaining "ships 172" claim; the PREREG commits both outcomes of Step 3 before any run. *Fail:* a count test needs a number the code cannot produce.
2. **Build the trainable DINOv3-proxy REF-D and pass its launch gate (E3, E4, E5, E7; ~2–3 weeks).** Same B1 / v7.2 corpus, refav1 loader and cache, R1–R5 wired, strategic off.
   *Pass:* (i) the 30-step smoke runs; (ii) `declared_vs_built` PASS and `check_untrained_params.py` lists **zero** non-declared never-trained parameters; (iii) a one-batch overfit reaches ADE@6 s < 0.1 m with the constant control reading its known value; (iv) copy/echo detector `echo_index` ≈ 0. *Fail:* any declared module unreachable (today's state).
3. **Run E-REFD-1′ at T1 with controls and replicates (compute; pre-registered).** Arms: REF-D(DINOv3) `action-only`, `+future-field`, and a replicate of the better arm with a new training seed *and* a new inference seed; controls: hold-action, CV, and `refav1-b1-v72-ep3-speed` on the identical 282 windows / 141 clusters; four families; paired episode-cluster bootstrap. *Proposed bar (to be fixed in the PREREG, HYPOTHESIS until then):* ADE@2 s below hold-action 0.5316 m with a **separated** paired CI, the separation exceeding the replicate arm's seed floor, and no family worse than hold-action beyond its CI. *Fail →* per Rule Zero that is a waypoint, not a deliverable: next lever is topic 2 (spatial conditioning), then topic 4 (Cosmos decodability panel) before any extraction spend.

---

## 10. File index (tip paths; all read at `460508ed`)

1. `stack/tanitad/refs/refd.py` — the module (411 lines)
2. `stack/tests/test_refd.py` — 19 test functions (213 lines)
3. `stack/tests/test_refs_vocab_v7.py` — v7.0 default pin `:29-31`
4. `TanitAD Research Lab/Architecture & Inference/Research/2026-08-19-simwam-analysis/REFD_DESIGN.md` — design + E-REFD-1/2/3
5. `…/2026-08-19-simwam-analysis/E_ACTSTREAM_1.md` (+ `raw/e_actstream_dino.json`) — E-ACTSTREAM-2
6. `…/2026-08-19-simwam-analysis/E_TRUNK_2_ENV_DECODABILITY.md`, `E_TRUNK_3_LADDER.md`, `PREREG_E_TRUNK_3.md`, `E_TRUNK_1_POOLING.md` (+ `raw/e_trunk2_probe.json`, `raw/e_trunk3_ladder.json`)
7. `…/2026-08-19-simwam-analysis/raw/residual_scale_audit.json`; `stack/scripts/residual_scale_audit.py`
8. `TanitAD Research Lab/Architecture & Inference/Research/2026-08-21-refcv3-review/UNIFIED_HIERARCHY.md` — REF-D measured 142,332,174; "strategic never built"
9. `stack/tanitad/models/hierarchy.py` (`:21`, `:89`, `:243`), `stack/tests/test_hierarchy_rung.py:124`
10. `Project Steering/NEXT_ARMS.md:49`; `V7_RECIPE_AND_SCALEUP.md:199,218,265-270`; `V7R_DESIGN_PROPOSAL.md:161`; `B1_TRAINING_PREP.md:52`; `Reports/2026-09-02-2300-v7-readiness.md:119,135`; `Reports/2026-08-27-0757-program-report.md:35`; `Reports/2026-08-29-1830-program-report.md:69`
11. `Project Steering/SPEC_NAV_CONDITIONING_ALL_LAYERS.md:11`; `SPEC_V7_LABELS_CONSUMER.md:14`; `V7_LAUNCH_GATE.md` (fp8 cache sizes, B1 corpus decision); `GOALS_AND_CLAIMS.md:3325` (D-CORPUS-B1); `RETRACTION_LOG.md:9563`
12. `TanitAD Research Lab/Architecture & Inference/Research/2026-08-27-jepa-driving-survey/JEPA_DRIVING_SURVEY.md:90-91` (thesis statement); `…/2026-09-03-v7f-design/DESIGN.md:291`
13. `Paper/TANITAD_PAPER.md:3769-3925` — §12 (REF-D narrative; §12.4's NAVSIM statement now stale)
14. `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/STATE_AND_GAPS.md`, `V7F_PLAN_R1_R6.md` — R1–R6 compliance-audit method
15. `stack/scripts/launch_gate.py` (PROFILES `:432-438`, `:666`); `stack/scripts/check_untrained_params.py`; `stack/scripts/refa_v1_train.py`; `stack/tanitad/data/refav1_loader.py`; `stack/tanitad/data/v7_labels.py`; `stack/tanitad/refs/refa_v1.py:75-84,1094-1173`
16. `Project Steering/MODEL_REGISTRY.md:2081-2096` — refav1 baseline numbers (hold-action 0.5316)
17. `TanitAD Research Lab/Library/papers/2608.07468_SimWAM-…pdf` (lib:2608.07468) — primary for every SimWAM number
18. `TanitAD Research Lab/Data Engineering/Research/2026-07-20-cosmos3-vlm-pilot.md` — the only Cosmos3 deployment evidence (Nano, not deployed)
19. NOT REF-D (do not misattribute): `…/2026-10-04-refcv8-data-audit/D6_navsim_failure_anatomy/{code/d6_ref_dac.py, raw/refdac_validation_step30000.json}` = NAVSIM reference-proposal DAC
20. Session scratch (not in repo): `C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/f0c4bd12-5464-4952-995c-062072b3b8d0/scratchpad/` — `refd_tip/` (tip export), `count.py`, `count2.py`; reproduce §4.2 with `PYTHONPATH=<export>/stack` and the `tanitad` venv.
