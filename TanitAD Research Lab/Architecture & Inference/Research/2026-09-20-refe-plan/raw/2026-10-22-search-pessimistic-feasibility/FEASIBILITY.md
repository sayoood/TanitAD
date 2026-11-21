# SEARCH-PESSIMISTIC: feasibility and cost of the out-of-fold scorer route (design memo, 2026-10-22)

**Scope.** This memo is design only. It ran no GPU job, used no pod and no network, and sealed nothing.
- **Repo reads:** pinned to the mirror tip `0738fa3`. The remote has since moved to `abccc45`, which was not re-read.
- **`PK`** = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.
- **Probes** (CPU, read-only, in this package): `raw/session_probe.py` → `raw/session_probe.json`, and `raw/cover.py` → `raw/cover_probe.txt`.
- **Evidence classes:** MEASURED = ours, with the path given; INHERITED = another agent's, not re-run; ESTIMATED = arithmetic.

## Verdict first

**The route as named is NOT feasible, and it would not be valid for P2.** The route is log-held-out scorer heads over navtrain, on cached trunk features.

A **narrowed version on the 24-log held-out table IS feasible now, at 0 GPU-h and about 1 CPU-h.** It can only tell whether the C1→C2 loss lives in the HEAD. It reaches the pessimism question only if that head-level stage finds a loss to recover, and two banked reads predict it will not.

**New finding that drives the verdict (MEASURED, `raw/session_probe.json`):**
- The 24 "held-out" segments (1,705 tokens) come from **21 drive sessions**.
- **All 21 sessions also supplied M-PDM segments** to P2's PDM fine-tune: **16,633 tokens**. A session here is the log name minus its two segment indices.
- Navtest's 136 segments come from **44 sessions, and none of them is in navtrain**.
- ⇒ C1 versus C2 is a **session-seen versus session-unseen** contrast, for the trunk and for the scorer alike.
- `D-L1-G0-1`'s "0 logs overlap" holds at the **segment** level only.

## 1. What P2's scorer is

**What the search maximises.**
- The search climbs `σ(NC)·σ(DAC)·(5σ(EP)+5σ(TTC)+2σ(C))/12` (`l1_stepA/stepA.py:35,126-131`). The DDC logit is unused.
- It is applied to `score_trajectories` (`PK/refe/model.py:699-709`), which runs three modules in sequence:
  - `score_q_mlp`, which encodes the 20×3 plan;
  - `score_dec`, 4 × `CrossBlock(256, 8 heads)`: self-attention over the 64-set, then cross-attention to `visual_ctx`, which is [7,680 × 256];
  - `score_head`, a Linear(256 → 6) layer (`model.py:552-563`).
- `real_scorer.py:36-56` builds only these three modules, and `tts_cem.py:29-74` searches over θ = (speed scale, lateral offset) applied to the pick.
- **Parameter count (MEASURED, CPU mmap of `D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt`):**
  - 81,408 + 4,213,760 + 1,542 = **4,296,710 parameters in 78 tensors**;
  - the model total is 322,070,854, which matches the registry's 322.07 M; the checkpoint is at step 1,209.

**The input is the trunk.**
- `visual_ctx = scene_proj(ViT-L/16 with LoRA r32 + pos3d)` over 4 cameras at 512×960 (`model.py:745-759`).
- In P2 this context is **un-detached** (`--scorer-sees-trunk`; `model.py:784`; `PK/eval/PREREG_P2.md:34-36`, INHERITED). The PDM loss therefore also trained the LoRA.

**Which labels trained it.**
- **Stage 1** (base `b54773f8`, context detached):
  - teacher scorer targets on a fixed candidate set, up to step 3,707;
  - from then on, the student's own 64 proposals, labelled by the teacher, with NAVSIM's DAC and comfort verdicts (`MODEL_REGISTRY.md` l.5512, departures 5-7).
- **P2:** BCE on the six NAVSIM PDM sub-scores of the executed plans, relabelled by `refe/onpolicy_relabel_pdm.py`.
  - It covers **174,912 sets = 99,495 tokens / 1,167 segments / 162 sessions** (MEASURED, `PK/raw/rl_pool/navtrain_token_table.tsv.gz`).

**Where the labels and weights are banked.**

| what | location | evidence |
|---|---|---|
| PDM train labels | HF `Sayood/tanitad-refe-p2-joint-pdm` `pdm_labels/pdm_train_labels.tar.gz`, 51.5 MB | INHERITED, `PK/raw/2026-10-13-aux-s1-data-audit/S1_DATA_AUDIT.md` I8; **no local copy found** |
| set trajectories | `D:/Projects/TanitAD-artifacts/refe-pod-final-2026-10-04/onpolicy_sets_logs_full.tar.zst` (2.07 GB), also on HF `Sayood/tanitad-refe-navtrain-banks` | local copy MEASURED |
| held-out PDM labels | `D:/Projects/TanitAD-artifacts/s1_read_cache/labels/x/heldout_pdm/` (14 MB) and `refe_heldout_v5.tar.zst` (65.5 MB) | local, MEASURED |

## 2. Cached trunk features

All rows MEASURED (from manifests, `ctx.json` files and file sizes).

| cache | trunk | content | usable for P2? |
|---|---|---|---|
| `D:/Projects/TanitAD/data/refe_trunk_cache/d7c59f4f/` | `snap_epoch015` (stage 1, step 4,933) | visual_ctx for 1,600 navtrain-train tokens, a val split and 200 navtest tokens; 9.3 GB | **no**: wrong trunk |
| `D:/Projects/TanitAD/data/refe_proxy/cache_train_ep015/` | `snap_epoch015` | 10,000 frames, of which 3,530 have visual_ctx; 14.2 GB | **no**: wrong trunk |
| `D:/Projects/TanitAD-artifacts/l1_stepA_run/bundle/ctx.npy` | P2-s0 | Step A, 400 tokens, 1.57 GB | yes |
| `D:/Projects/TanitAD-artifacts/l2_fresh_stepA/B_token_disjoint/bundle/ctx.npy` | P2-s0 | carve B, 400 tokens, 1.57 GB | yes |
| `C:/Users/Admin/spc_c1/B_REST/bundle/ctx.npy` | P2-s0 | B-REST (C1), 905 tokens, 3.56 GB | yes |
| `D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/` (`prehead_features_p2_heldout.npz`, `table_p2_heldout.npz`) | P2-s0 | score_head input [1705, 64, 256] fp16, plus the true sub-scores [1705, 64, 6] | head-only |
| `D:/Projects/TanitAD-artifacts/l2_offprop/feats.npz` | P2-s0 | Step A pre-head features: the 64 in-fan proposals plus 96 perturbations | head-only |
| `C:/Users/Admin/srttc_s0/` (S0, 1.93 GB) | P2-s0 | pre-head features for 15,808 M-PDM sets over 117 segments | head training only: no ctx, so no search |

⇒ **P2's `visual_ctx` exists for exactly the 1,705 held-out tokens (6.70 GB), and for no M-PDM token.**
- The three bundles are token-disjoint, their union is 1,705, and all three name P2's checkpoint.

**Building a full M-PDM cache:**
- **Size:** 99,495 × 3.93 MB = **391 GB**.
- **Compute:** about **37 dev-box GPU-h**, at the MEASURED 1.35 s per token (`c1/logs/gpu_stage.log`: 905 tokens in 1,224 s).

It is blocked on three counts:
- **Disk.** C: has 13 GB free and D: 22 GB.
- **Frames.** The navtrain pixels (80 GB) were never retrieved (`refe-pod-final-2026-10-04/MANIFEST.txt`).
  - Locally there are only the proxy's 10,000 selected frames (12 of its zips sampled: 225 jpg for 761 tokens), m5's 1,900 frames, and the 1,705 held-out tokens.
- **DBs.** Labelling search outputs needs each segment's DB, and only **209 of the 1,167** M-PDM segments have one on the dev box (`raw/cover_probe.txt`).

## 3. Out-of-fold infrastructure that already exists

**L2-OFFPROP** (`PK/raw/2026-10-19-l2-offprop/DESIGN_CHOICES.md`) is the starting point:
- It refits only the **EP row** (257 parameters), starting from P2's row, on Step A's 400-token features.
- The loss is pairwise + 10×BCE + prox 0.3, with 5 folds by **segment**.
- G-LIVE holds the frozen rows' gradient at exactly 0.0.
- A cell takes 7.5 s on CPU.
- `l2fresh/code/logoof_companion.py` refits the fold heads bit for bit.
- Head substitution in the search already exists: `l2stepb/code/stepB.py search --arm L2 --head`, plus `l2_scorer.py` and G-REACH.

**Changes needed to cover the whole composite:**
1. Train rows {NC, DAC, EP, TTC, C}: a BCE term per row on the fan labels, plus pairwise terms. DDC's gradient must stay exactly 0.0.
2. Fold by **session**: 3 of the 21 sessions straddle L2's segment folds.
3. Train on the 1,705-token fans (already exported) plus Step A's perturbations.
4. Give the search a per-token map from fold to head.
5. `cem_search` returns its final elites, and `label_stepB.py` labels the pick plus 8 elites per seed.
6. Add a numpy selection-rule module with identity controls.

**Compute.**
- The fits take minutes of CPU (ESTIMATED from L2's 7.5 s per cell).
- A **decoder-level** variant (`score_q_mlp` + `score_dec` + head, 4.30 M parameters) is fine-tuned on the 6.7 GB of ctx:
  - about **1.5 GPU-h**, bound by I/O (ESTIMATED: about 10.6 GFLOP per forward of one 64-set);
  - or 30-60 CPU-h.

**L2-BRANCH lesson** (`l2-stepB/MM_RULING_STEPB_RESULT.md`, INHERITED):
- A refit head changes the search's direction mix **on SEEN scenes**: 23-28 % speed-ups, and slow-downs at −1.5.
- ⇒ Head quality must be equal across the two arms being contrasted.

## 4. The read design (24-log table)

**Folds.** Five session folds: greedy balancing gives about 341 tokens each, where rank-mod-5 gives 578/335/445/211/136. Inside each segment, a seeded hash splits the tokens into a fit half and a search half. Fold model *g* trains on the fit halves of the other 4 folds.

**The contrast.** It is paired, on the about 850 search-half tokens *t* of fold *g*:
- **OOF arm:** search with model *g*, which never saw *t*'s session.
- **IN arm:** search with model (*g*+1) mod 5, which saw *t*'s segment. This is token-fresh and log-shared: the C1 analogue.
- Both models were trained on equal amounts of data.

**Noise floor.**
- A model-swap null: two IN models on the same tokens.
- A training-seed replicate (H-ESTIM-SEED-1).

**What it needs.**
- Bundles A, B and B-REST, plus `table_p2_heldout.npz`.
- The CPU PDM labeller (`label_stepB.py`).
- DBs for all 24 segments: MEASURED, all resolve.

**Label cost.**
- Rate (MEASURED, C1 `ZZTIMING`): the metric cache costs about 1.04 s per token, then 4.4 ms per plan.
- Tier: SELECTION, label-only. It is not T1.
- The carve-B fixture read identically on C2's navpipe code (`c2/FIXTURE_RECORD.json`, INHERITED).

**Power** (ESTIMATED from banked intervals):

| read | tokens / logs | half-width |
|---|---|---|
| C1, single arm | 905 / 24 | 0.41 |
| carve B, single arm | 400 / 24 | 0.33 |
| dose O4−C8, paired | | 0.10 |
| Step B L2−P2HEAD, paired | | 0.37 |
| **C2 (comparison)** | **12,146 / 136** | **0.19** |

- The two single-arm half-widths do not shrink from 400 to 905 tokens: the 24 logs bound them.
- ⇒ **Minimum detectable effect ≈ 0.15-0.5 points** (80 % power).
- That can see a head-level gap of about 0.5 or more (C1−C2 = 0.71, unpaired).
- It **cannot** see a recovery the size of C2's −0.155.

## 5. Pessimism knobs worth pre-registering

All four are final-selection rules applied to the labelled elites. A penalised CEM is a follow-up only.

| knob | rule | grid | notes |
|---|---|---|---|
| **K1** trust region | agg − λ‖(u/0.10, d/0.20)‖ | λ ∈ {0, .0025, .005, .01, .02, .04, ∞} | λ = 0 must equal the plain search bit for bit, and λ = ∞ the pick |
| **K2** margin | move only if agg(out) − agg(pick) > τ | τ ∈ {0, .005, .01, .02, .05} | |
| **K3** ensemble LCB | mean − κ·sd over heads that all **exclude t's session** | κ ∈ {0, .5, 1, 2} | 5 session-bootstrap heads per fold; the plain fold heads are not a free ensemble here, because 4 of 5 saw t's session |
| **K4** slow-down guard | a separate τ_slow for u < 0 | {0, .01, .02, .05, ∞} | post hoc from C2, and declared as such |

**Fit/judge split, never navtest.** Cross-fitting over the 5 session folds:
- For each judge fold *g*, choose the knob value on the OOF outputs of the other 4 folds. The objective is mean dPDMS against the pick; a tie goes to the more pessimistic value.
- Apply that value to *g*.
- Report the pooled paired dPDMS twice: pessimistic minus plain, and pessimistic minus pick.
- The estimator is the session-cluster bootstrap over 21 sessions.

## 6. Confounds

1. **OOF heads are weaker.** They are trained on about 1,364 tokens, against 174,912 sets for P2.
   - Contrast IN with OOF at equal data, never against P2.
   - Report each head's in-fan concordance against P2's.
2. **The trunk and the decoder have seen the sessions.** P2's PDM label for the 16,633 sibling tokens reached the LoRA, because the context was un-detached.
   - On the 24-log table this exposure is equal across folds, so it cancels in IN−OOF. It also caps the test at **head-level** exposure.
   - Folds over M-PDM would turn it into **direct label leakage through the trunk**. That is why the navtrain-wide route is refused, not merely priced.
3. **Leakage across folds.**
   - Segment folds leak through sessions: M-PDM is 1,167 segments in 162 sessions, 147 of them with several segments.
   - Adjacent tokens share most of their 4 s future, so the token halves must stay strict.
   - Step A's perturbation pairs must be split by the same halves.
4. **Forking paths.** Step A, Step B, L2, L2-FRESH, C1 and the dose read all used these 24 logs. Cross-fitting protects λ, not the choice of knob family. Declare it.
5. **Population.** 21 sessions against navtest's 44. Slow-downs were 10.7 % of C1's outputs and 15.8 % of C2's.

## 7. Verdict and cost

**Stage 0, the head-level gap gate.** Feasible now: **0 GPU-h, ≤ 1 CPU-h**, all inputs local.
- Session-fold refits of the 5 rows on the cached pre-head features.
- It reads IN−OOF on three measures, against the model-swap null: in-fan concordance, A3 aggregate concordance on Step A pairs, and the regret of the argmax over the 64-proposal fan.
- **Prior: NULL** (MEASURED elsewhere):
  - The log-OOF EP heads score A1 0.841 and 0.838 (seeds 0 and 1), against 0.840 and 0.846 for the in-fold heads (`l2-fresh-stepA/FRESH_RESULT.md`).
  - HEADROW-PROBE's OOF linear NC, DAC and TTC rows are within 0.01 of P2's in-sample rows (`headrow-probe/MM_RULING_HEADROW.md`).

**Stage 1, the search and pessimism read.** It runs **only if Stage 0 finds a gap**.
- Four arms: P2HEAD on Step A's 400, OOF, IN, and a seed replicate.
- Cost at C1's rate of 1.48 s per token-seed on 4 threads: **about 30 CPU-h and 0 GPU-h**, about 8 h wall if run in parallel.
- Labels add about 2 CPU-h; K1-K4 are then free, applied offline.
- The decoder-level variant adds **about 1.5 GPU-h** once the GPU lock is released. (The lock file named in the brief was not present on 2026-10-22, and no GPU job was run.)

**If Stage 0 reads NULL, "does pessimism recover the unseen-scene loss?" cannot be answered on P2.**
- There is no session-unseen stratum outside navtest.
- **P6 inherits the same problem.** It trains on the same 103,037-token rank-0 bank (`2026-10-22-p6-newpod-rebuild/RESULT.md:12`) and resumes only on that identical bank.
- The next-cheapest routes, both blocked:
  - **(a)** Reserve whole navtrain sessions (about 16, i.e. about 10 %) when a future trunk is trained. **Blocked on a PI decision.**
  - **(b)** Split navtest by session, 22 sessions to fit and 22 to judge. **Blocked by the rule that navtest is used up.** It would need an explicit MM/PI ruling, and this memo does not recommend it.

**Register items for the MM:**
- Narrow `D-L1-G0-1`'s scope to segments, and add the session overlap.
- In C2 `RESULT.md`, change "logs P2 was trained on" to "sessions P2 was trained on".

## Deliverable manifest

| artifact | where | copies |
|---|---|---|
| `FEASIBILITY.md` (this memo) | `C:/Users/Admin/par18/spess/FEASIBILITY.md` | ONE (dev box; MM lands) |
| `raw/session_probe.py` + `raw/session_probe.json` (session overlap, folds, navtest disjointness) | `C:/Users/Admin/par18/spess/raw/` | ONE |
| `raw/cover.py` + `raw/cover_probe.txt` (local pixels and DBs for M-PDM) | `C:/Users/Admin/par18/spess/raw/` | ONE |

Nothing was staged, committed or pushed. No git index was touched.
