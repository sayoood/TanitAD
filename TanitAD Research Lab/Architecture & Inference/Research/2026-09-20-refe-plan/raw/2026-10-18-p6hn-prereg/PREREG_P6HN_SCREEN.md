# P6-HN-S: a scorer-only screen of the offset hard-negative bank on a frozen trunk (pre-registration)

> **Status.** Written 2026-10-18 on tip `e9b9f3db` by the pre-registration design agent. Registrable as written: the Master Mind
> hashes it (the `.sha256` beside it records this text), lands it, and amends it only by a dated amendment written before any
> screen number exists. **No screen number exists**: no bank, no cache, no arm, no read. Navtest is never opened by this screen.
>
> **Mandate.** PI, 2026-10-18, on `NOTES.md` Q1: *"screen first (Recommended)"*, i.e. option (b): a scorer-only fine-tune with
> and without the offset bank, trunk frozen, seeds 0-2, map-derived negatives excluded (C9 open). **This screen decides one
> thing: whether the 6.8-7.65 A40-day P6-HN arm** (`PREREG_P6HN.draft.md`, sha256 `4a9ea208…`) **runs.**
>
> Evidence classes: MEASURED (ours, artifact path) · PUBLISHED (library key) · INHERITED (another doc, not re-verified here) ·
> ESTIMATED · HYPOTHESIS. Paths are relative to `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/`
> (PK) unless written in full. "The draft" = `raw/2026-10-18-p6hn-prereg/PREREG_P6HN.draft.md`.

## 0. Question, decision, and the one lever

**Question.** With the trunk frozen, does adding the draft's offset bank to the scorer's loss cut the scorer's **zero-PDMS picks
on real fans**, robustly across training seeds, without paying in NC or TTC?

**Why this question and not ΔPDMS.** The draft's own power section (§10, ESTIMATED) puts the full arm's likeliest outcome at
HN-NOT-PROVEN: the one published effect of this form is **+0.22 PDMS** (`2608.19085` Tab. 3, single run, PUBLISHED), below any
interval our harness resolves. Zero picks are the lever's **direct target** (ZT3: the DAC zero tail is near-twin grazes,
`D-REFE-ZEROTAIL-3`, INHERITED) and the read with the most power per scene (§12). PDMS is a no-harm guard here, not the endpoint.

**The one lever.** `--hn-bank <path>` on **arm B** of `refe/scorer_finetune.py`'s paired pass. Arm A of the same pass is the
control. Everything else is RL-S1's registered BCE-only argv (§4.2).

## 1. The frozen trunk: u400, and how the answer transfers to P6

| item | value | source |
|---|---|---|
| checkpoint | `model_sft4_B_u400.pt`, sha256 `8e382eebb35b3e15…` = `model_final.pt` (`b54773f8…`) + `scorer_states_u400.pt` (`4dded9b1…`) | `MODEL_REGISTRY.md` row "SFT-4 B u400" |
| status | the SFT-4 deployment candidate; navtest 88.25 (T1 NAVSIM v1.1) | registry; GAC `D-REFE-SFT-4` (+3.79 [+3.05, +4.54] over `model_final`) |
| frozen | trunk (DINOv3 ViT-L + LoRA), registers, scene projection, trajectory decoder: everything except `score_q_mlp`, `score_dec`, `score_head` | `scorer_finetune.py` gate G1 `trainable_is_exactly_the_scorer` |
| scorer | 7 outputs (6 PDM components + teacher lane); selection rule `navsim_v1_lane`, A7 repair ON | `PREREG_RL_S1.md` base table A (INHERITED) |

**Why u400.**
1. It is the deployment candidate, and SFT-4 is itself a scorer-only fine-tune of `model_final`, so a frozen-trunk scorer
   fine-tune on it is the regime it was built in (the scorer reads the visual context **detached**, `model.py:784`).
2. The whole rig already exists for it and nowhere else: the log-disjoint RL-train / RL-val partition, the frozen 64-proposal fan,
   its PDM labels, the paired A/B pass, the `--dump-scores` → `eval/rl_val_eval.py` lookup, and the `refe_rls1*` launch-gate profiles.
3. Its arm A has a banked twin: RL-S1's **C0** (BCE only, seed 0, same recipe), banked on HF (`eval/rls1b_arms.json`).

**Why not P2 or P6.** P6 does not exist. P2 is NOT REPLICATED (`H-REFE-P2-REP-1`: TTC −0.49 [−0.97, −0.07] adverse on its
seed-1 replicate) and **has no clean pool**: it trained on 99,495 of the 101,583 eligible navtrain tokens, leaving 2,088
(`PREREG_RL_S1.md` OPEN-BASE, MEASURED there; INHERITED here). No RL-val fan or label exists for P2 (`D-REFE-SELRULE-FIT`,
Amendment A1). P2 would cost a new fan dump and ~54 CPU-h of labels, and it would still have no held-out navtrain set.

**Transfer to P6.** The screen's rig transfers. Its answer transfers only in part:

| transfers | does NOT transfer |
|---|---|
| **The labels.** The bank is labelled by `onpolicy_relabel_pdm.label_log`, the body P6's online labeller must equal bit for bit (G-LIVE-LAB D1(a), `refe/diag_p6_labeller_identity.py`). | **The trunk channel.** P6's scorer context is un-detached (D3), so in P6 the hard-negative term also trains the trunk LoRA. The screen measures the **head channel only**. |
| **The bank.** It is the same builder, δ grid, anchor rule and hard-negative definition as the draft §2. | **The base regime.** u400 is a 25,600-set BCE fine-tune of a frozen fan. P6's scorer trains from scratch for 25 epochs on 189,888 tuples (INHERITED). The marginal value of 8 extra rows per sample can differ either way. |
| **The loss term.** The term is ONE function in `refe/hn_bank.py`, imported by both trainers (§2.4), so the screen exercises the exact term P6-HN would. | **The magnitude.** RL-val is navtrain with u400's fan. A screen Δ is not an estimate of the full arm's navtest ΔPDMS (any such reading is ESTIMATED). |
| | **The fan.** P6 has 3-C twins and its own generator, so its near-twin supply differs from u400's (the census measured u400's). |

**What each outcome therefore licenses.**
- **GO** says the head channel alone can use the bank to cut real zero picks safely. The lever has a working mechanism, so the
  full arm is worth its compute.
- **NO-GO / HEAD-BLIND** (§11 row 9) does not prove P6-HN would fail. It does mean any benefit would have to come through the
  trunk, and P7.1 targets the trunk directly with a larger ESTIMATED reach (LAB-RUN-019 §4.3). The trunk channel is re-tested by
  re-running this screen on **frozen P6** (an amendment naming the checkpoint) **iff** P6's own ZT4 read shows HEAD-MOVES against
  u400.
  - ⚠️ The eval set of that re-screen must be disjoint from P6's training data. RL-val is navtrain and is probably inside P6's
    manifest. The candidate is the 24-log SFT held-out table, IF P6's manifest excludes those logs (UNVERIFIED).

## 2. The bank: the draft's §2, with one correction

### 2.1 Anchor
- **Anchor:** each RL-train scene's teacher plan (rank-0 target `traj`, 20 poses, 4 s) from the TargetBank. RL-S1 measured that
  all 51,000 partition scenes are bank rows (INHERITED).
- **STATIONARY rule:** an anchor whose path is shorter than 2.0 m gets no offsets (`PREREG_HN_CENSUS.md` §3.6). These are counted.

### 2.2 Offsets
- 8 per scene: pose k is shifted by δ along the anchor's unit left-normal at pose k, with
  δ ∈ {±0.15, ±0.30, ±0.45, ±0.60} m.
- The normal comes from the stored yaw if present, else from a central-difference tangent.
- Heading and time are kept.
- **No map geometry is read to build or choose a negative.**

### 2.3 Labels and the hard-negative definition
- **Labels:** the 8 offsets and the anchor, so 9 trajectories per scene.
  - Path: A7 last-heading repair → `to_navsim` → `refe/navsim_pdm_targets.py`, through `onpolicy_relabel_pdm.label_log`.
  - This is exactly the path and `label_version 5` of the fan labels the screen trains on (gate I1 verifies it bit for bit).
  - The drivable-area map enters only as a training-time label, as it does for every fan proposal (admissible, PI 2026-10-11).
- **Hard negative** (DA-WAM Eq. 11 in our terms, the draft verbatim): PDMS(offset) = 0 (NC = 0 or DAC = 0) AND PDMS(anchor) > 0.
  - NC = 0.5 is not a hard negative.
  - Sub-types (DAC-only / NC-only / both) are recorded.

### 2.4 How they enter the loss (correction to the draft)
**The draft says each hard negative is "appended to that sample's scorer candidates" and that "the fan term is unchanged bit for
bit". Both cannot hold in this architecture.**
- The scoring decoder **self-attends across the set**: `CrossBlock.self_attn`, `model.py:429/435`.
- `score_trajectories`' own docstring says: *"a trajectory is scored IN ITS SET"* (`model.py:699-705`).
- So a plain append changes every fan row's score, and it trains the head on 72-candidate sets that are never deployed.

**Committed form (the screen's, and recommended to the draft by amendment).** Arm B makes two calls:
1. **The fan call is unchanged.** Arm B's fan logits come from the existing call, so the fan BCE is bit-identical to arm A's code
   path.
2. **A second, masked call** runs over `[fan ; hn]` and keeps only the hard-negative rows. The attention mask lets each
   hard-negative query attend to the 64 fan queries and to itself, and nothing else. Each hard negative is therefore scored as a
   65th candidate in its real set, which is the situation selection faces.

**Loss.**

`L_B = bce_fan_B + Σ_{valid hn rows} Σ_{c ∈ 6 PDM comps} BCE(hn, c) / (B · 64 · 6)`

- `bce_fan_B` is the existing `F.binary_cross_entropy_with_logits(sxB[..., :6], tg[..., :6])`, a mean over B · 64 · 6 elements
  (`scorer_finetune.py` `losses6`).
- So **one hard negative weighs exactly one fan proposal**, and no free weight is added.
- The lane output (component 7) gets **no** loss from hard-negative rows.
- Padding to 8 per sample uses a mask, and masked rows contribute exactly 0.
- Hard negatives never enter evaluation or selection sets.

### 2.5 Excluded, named
1. **Map-derived negatives** (lane boundaries, drivable polygons, lane-graph alternatives). These wait on the open PI ruling C9.
2. **Negatives retrieved from the archived fan.** Excluded for the draft's §2 reasons.
3. **Any re-selection.** The bank only adds rows to RL-S1's frozen n16000 RL-train sets.

### 2.6 Where the bank is built
- **RL-train, 16,000 scenes:** the training bank.
- **RL-val, 3,000 scenes:** a held-out bank used **only** by read M3 (§9). It is never trained on.

## 3. Data, features, and what needs a fresh trunk forward

**Data (all RL-S1's, INHERITED from `eval/PREREG_RL_S1.md` and its n16000 amendment).**
- **RL-train:** 16,000 scenes / 166 logs. Manifest: `raw/rl_pool/partition_runs/A_no_trained_list_seed20261012_n16000.manifest.json`.
- **RL-val:** 3,000 scenes / 75 logs. TSV sha256 `39f425dc3f0866ec…`.
- The two sides are log-disjoint, and both exclude navtest and the 24 SFT held-out logs.
- **The fan:** `model_final`'s 64 proposals. It is u400's, because SFT-4 is scorer-only.
- **The labels:** PDM labels, `label_version 5`. RL-train carries 1,024,000 labelled proposals.
- **Residual, INHERITED:** u400 touched at most 25,600 sets, and which ones is not recoverable. So up to ~26 % (25,600 / 99,495, ESTIMATED) of RL-val tokens may
  have been seen by the base scorer. This shifts the levels of both arms equally, and it cannot bias B − A.

**The banked pre-head features cannot carry this screen.**
`raw/2026-10-18-p2-heldout-table/` and `raw/2026-10-16-p2-full-table/` bank `prehead_features`. These are the INPUT of
`score_head` (`[64, 256]` per set, `p2ho_featcheck.py`). They fail on four counts:
1. They are **P2's** scorer, not u400's.
2. They lie **downstream of `score_q_mlp` and `score_dec`**, which the screen trains. They go stale at the first update.
3. **The offsets are new trajectories** and have no feature at all.
4. The full table is **navtest**, which is reserved for P6 and the full arm, and the held-out table's 24 logs are outside RL-S1's
   partition.

A `score_head`-only fit on them would be a linear probe. Its negative would say nothing about learnability (TRAPS, "a negative
from a linear probe"). **None of them is used.**

**What IS cached: the frozen trunk's output.** The scoring decoder reads `visual_ctx = scene_proj(visual)`, of shape
`[B, 7,680, 256]` (4 cameras × 32 × 60 patches at 512 × 960 / 16; `model.py:759`, from source).
- With the trunk frozen and no image augmentation in `scorer_finetune.py`, it is a fixed function of (image, calib).
- It is computed **once per scene** under bf16 autocast, which is the trainer's own dtype (`scorer_finetune.py:994`).
- It is stored losslessly as the bf16 tensor the live path produces: **3.75 MiB per scene**.
- It is reused by all arms, all seeds and every eval.

**A fresh trunk forward is needed for exactly 19,000 scenes:** 16,000 RL-train + 3,000 RL-val. Nothing else in the screen
touches the trunk. The same forward also re-emits the 64-proposal fan, which is the reproduction check I6 (§5).

**Staging** (all sha256-verified on both sides, as the P2 held-out table did):
- the 4 camera jpgs of the 19,000 scenes;
- the nuPlan DBs of the 241 logs, needed to label the offsets;
- RL-S1's fan dumps and PDM labels, if they are not already on the dev box. Verify against RL-S1's manifests by sha256, never by
  presence.

## 4. Arms, recipe, and the launch-gate entry

### 4.1 Arms

| arm | definition | seeds |
|---|---|---|
| **base** | u400's scorer, frozen (the trainer's built-in reference copy) | none |
| **A_s** | arm A of pass s: BCE only. The control. | 0, 1, 2 |
| **B_s** | arm B of the same pass: BCE + the hard-negative term (§2.4) | 0, 1, 2 |
| extension | passes s = 3, 4. Run only under §7 rule E. | 3, 4 |

- **Pairing.** A_s and B_s read the same captured context and the same batches in the same order, so they are paired by
  construction.
- **What the seed varies.** The DataLoader order (and dropout masks, if the scorer has any; the implementer states which).
- **Determinism.** Strict deterministic algorithms are on (`scorer_finetune.py:1369`). A same-seed rerun is therefore bit-identical
  and is not a replicate. Seeds are the replicate.

### 4.2 Recipe and argv

The base argv is **RL-S1's C1 argv token for token** (`eval/rls1b_arms.json` arm C1):
- `--require-pdm --b-mode listnet --lam 0 --lr 1e-4 --max-updates 1000 --epochs 4 --workers 3 --eval-every 250 --lane-head`;
- the teacher-lane and PDM-label files;
- batch 8 × accum 8 (the defaults);
- bf16; SFT-4's AdamW, 50 warm-up, cosine, wd 0.01.

**Allowed differences, and nothing else:**
- `--seed s`;
- its own `--out` and `--dump-scores`;
- data paths re-rooted to the run box. **Content fingerprints must equal C1's / C0's token fingerprints**: the partition, the
  fan sets and the labels. The jpgs are a staged subset, verified file by file;
- plus `--hn-bank <bank>` and `--ctx-cache <dir>`, each exactly once.

**Inherited regime property, stated.** On navtest, C0's arm A sat −0.88 [−1.47, −0.14] PDMS below u400
(`D-REFE-FREEPOINT-1`, INHERITED): continued BCE on 16,000 scenes degrades the base. The screen keeps the recipe anyway. Its
contrast is B − A within that regime, and keeping the recipe keeps the C0 cross-check and the registered profile. Both arms'
levels against the base are reported.

### 4.3 Reachability, proved before the gate (CLAUDE.md "built, tested, and unreachable")

**Hop trace.** The implementer names file:line at each hop:
- argparse `--hn-bank`
- → `a.hn_bank`
- → the bank loader, joined on OnPolicyBank's key `(log, token, step, rank)` + `ckpt_step`. It refuses a bank whose partition id,
  label_version or fan fingerprint differs from the run's.
- → the batch fields `hn_xy [B, 8, 20, 3]`, `hn_tg [B, 8, 6]`, `hn_m [B, 8]`
- → `fwd()`'s masked second call (§2.4)
- → the hard-negative term in `losses6` → `L_B`
- → `.backward()`

**Gradient checks on a real cached batch with `n_hn > 0`, from the hard-negative term alone:**

| parameter group | required gradient |
|---|---|
| arm B `score_q_mlp`, `score_dec`, `score_head` rows 0-5 | finite, **non-zero** |
| arm B `score_head` row 6 (lane) | **exactly 0.0**. The built-in control that proves the probe works. |
| every arm A and base parameter | **exactly 0.0** |
| trunk / generator | `requires_grad False` (G1) |
| any group, with `--hn-bank` absent or the bank empty | **exactly 0.0**, and B ≡ A bit for bit after 2 optimiser steps |

**Analytic identity.** A bank row that duplicates fan row j (trajectory and labels) adds exactly row j's share of `bce_fan_B` to
`L_B`. Exact on fp32 CPU.

**Mutations. Each must turn a named test RED:**
1. delete the line that adds the term;
2. route the hard negatives into arm A;
3. unmask one padded row;
4. drop the attention mask, so fan rows see the hard negatives;
5. give hard-negative rows a lane target;
6. use a bank from another partition or label_version;
7. use a cache from another trunk fingerprint.

### 4.4 Launch gate: profile `refe_hnscreen` (`Project Steering/SPEC_REFCV7.md` §2; `eval/REFE_LAUNCH_GATE.md`)

| item | specification |
|---|---|
| profile | `{**PROFILES["refe_rls1"], "name": "refe_hnscreen"}`. Runner `scorer`. Arm seeds {0, 1, 2} (+{3, 4} once §7 rule E fires). Base argv = §4.2. Any moved, added, dropped or doubled token FAILS. |
| `DVB_KINDS_SCORER` | `hn_bank: "data"` and `ctx_cache: "data"`. Both are fingerprinted into the token. An unfingerprinted input path FAILS. |
| built facts | The trainer's `data` event carries two blocks, and `scorer_dvb_rows` compares them with the argv. Equal or FAIL. |
| — `hn` block | `{sha256, rows, scenes_with_hn, max_per_sample: 8, arm: "B", comps: 6, weight: "1/(B*64*6) per element", mask: "hn->fan+self"}` |
| — `ctx_cache` block | `{manifest_sha256, trunk_fingerprint, n_scenes, dtype: "bf16"}` |
| new preflight gate **G14** `hn_live_and_identity` | Re-judged from raw fields by a new `judge_g14`, never from its `ok`. Fields: the §4.3 gradient table; B ≡ A with the bank off and with it empty; the analytic identity; the pad-perturbation identity; and that the fan logits of B's two-call path equal the one-call path bit for bit. |
| new preflight gate **G15** `ctx_cache_identity` | On ≥ 64 RL-train and ≥ 64 RL-val scenes, base, A and B logits from the cache equal those from the live trunk forward, bit for bit, on the run box. A row-shuffled cache must read a max abs difference > 10× the identity's. The manifest's trunk fingerprint must equal the checkpoint's. |
| refusals | `--hn-bank` with `--epo`, with `--lam ≠ 0`, with `--aux-drv`, or with a bank of 0 hard-negative rows (`empty_bank_in_launch`). |
| deliberate-regression arms, each must FAIL | `lever_ignored` (parsed, never added); `hn_to_armA`; `hn_in_selection` (hard-negative rows in a dumped eval set); `hn_lane_target`; `unmasked_pad`; `fan_sees_hn` (no mask); `stale_bank` (bytes changed after minting); `empty_bank_in_launch`; `moved_lever`; `stale_cache`; `shuffled_cache`; `epo_with_hn` |
| G-CLOCK | `rls1_checks.py updates-check`: 1,000 planned updates = `--max-updates` |
| G-SUITE | 0 regressions, including the existing `refe_rls1*` arms, which `--hn-bank` absent must leave bit-identical. ⚠️ The suite's end-to-end G-CKPT arm OOM'd once on the dev box (2026-10-12, environmental, INHERITED), so run it with the CPU-fixture path. |
| token | Required for every pass, on whatever box runs it. The chain refuses to start the trainer without a matching token. |

## 5. Gates before any read (CPU unless stated; any failure ⇒ SCREEN-INVALID, except SY)

| id | gate | rule |
|---|---|---|
| **SY** | bank yield on RL-train | Share of RL-train scenes with ≥ 1 hard negative, by log-cluster bootstrap (166 logs, 10,000, seed 20260927). **< 0.10 ⇒ SCREEN-SPARSE**: no cache is built and no pass runs (§11 row 1). Also reported: rate per \|δ\| and side, sub-type split, the share of anchors with PDMS = 0, and STATIONARY skips. It reads supply, never an effect. |
| **I1** | label identity | For ≥ 64 RL-train sets, the bank builder's labelling call, fed the set's own 64 fan proposals, reproduces RL-S1's banked PDM labels bit for bit. |
| **I2** | geometry and no map | On a straight and a curved toy path, each offset's mean absolute cross-track to the anchor is \|δ\| (median error ≤ 0.01 m), with time and heading unchanged. The builder's call graph reaches no map API, paired with a control that must find one in the labeller. |
| **I3** | default-off identity | The real `main()` of the pre-edit trainer and of the edited one, without `--hn-bank` and `--ctx-cache`, give every loss, eval number and saved tensor bit-identical (the `test_aux_s1_wiring.py` pattern). |
| **I4** | eval reads no bank | B_s's RL-val dumps are bit-identical with the bank file present and absent. |
| **I5** | dump determinism | Re-dumping each final checkpoint gives bit-identical scores. Inference variance is then zero by test. |
| **I6** | cache and fan reproduction (GPU) | G15 holds. The cache pass's 64 proposals equal RL-S1's banked frozen fan within the registered fan tolerance (`raw/rl_fan/atol_calibration/`) on every cached scene. |
| **D1** | delivery accounting (in the run) | Σ hard-negative rows that entered `L_B` = Σ over served samples of that key's bank rows, exactly. The term is finite and > 0 at every logged update with `n_hn > 0`. |
| **IC** | instrument identity | Every two-arm joint-draw difference (M1-M3, the co-gate and PDMS) computed as an arm minus itself reads exactly 0 with CI [0, 0]. |

## 6. Eval set, tier, estimator

**Eval set: RL-val**, the 3,000 sets / 75 logs of the frozen u400 fan, read at the **update-1,000** checkpoints only. There is no
early stop and no checkpoint choice; the 250/500/750 dumps are logged and never read for the decision. Picks are made under
`navsim_v1_lane` with A7 ON, and ties go to the lowest index (`rl_val_eval.py`).

| considered | verdict |
|---|---|
| full navtest | ⛔ Reserved. It is P6's and the full arm's decisive read. Spending it here would turn the full arm's endpoint into a second look. |
| P2's 24-log held-out table | Rejected. It holds P2's fan, not u400's. Its 24 logs are about one third of RL-val's 75. The census and the draft's Gate Y already read supply there. It is kept for the frozen-P6 re-screen (§1). |
| "dev split" | No separate dev split exists beyond the two above. |
| **RL-val** | **Chosen.** It is log-disjoint from RL-train, navtest and the 24 held-out logs. Its labels are complete and frozen, so any scorer is a table lookup. It has never been read for a hard-negative effect. Disclosed: its base-scorer zero tail has been read (`D-REFE-SELRULE-FIT`: u400 picks 68 DAC = 0 sets of 3,000, median p_DAC 0.93); nothing in the bank was chosen from it. |

**Tier.** SELECTION: a lookup of the NAVSIM v1.1 PDMS of an open-loop plan on navtrain RL-val, frozen fan. It is not T1, and it is
"hyper-parameter / early-stop use only, never an endpoint" (`rl_val_eval.py` TIER). Every screen number is MEASURED **on this
tier**. Any reading of it as navtest, T1 or P6 performance is ESTIMATED. No screen number enters `MODEL_REGISTRY.md` as a
performance row.

**Estimator.** Paired log-cluster bootstrap over the 75 RL-val logs, 10,000 resamples, seed 20260927, percentile 2.5/97.5. It uses
the ratio-of-sums form of `rl_val_eval.py` / `proxy_eval.boot`.
- **One log draw per resample is applied jointly to every seed and arm.** A seed-mean's interval therefore answers *"another draw
  of LOGS?"* for the mean over seeds.
- The **training-seed** question is answered separately, by §7's sign and floor rules.
- The **inference** question is closed by I5.

## 7. Primary endpoint and the decision rule (how logs and seeds combine)

**Primary endpoint.**
- z(X) = the share of RL-val sets whose pick by scorer X has true PDMS = 0, in percentage points.
- Δz_s = z(B_s) − z(A_s), paired on the same sets.
- **Δz̄ = mean over the seeds of Δz_s.**
- **F_z = max_s z(A_s) − min_s z(A_s)**: the range of the no-lever controls. This is the rig's own run-to-run floor for this
  metric (`H-ESTIM-SEED-1`).

**ZEROS-FALL iff all three hold:**

| | condition | the question it answers |
|---|---|---|
| **Z1** | UB(Δz̄) < 0 | logs: would another draw of logs say this? |
| **Z2** | Δz_s < 0 for every seed | training: is the sign stable across training runs? |
| **Z3** | −Δz̄ > F_z | training: is the effect larger than two runs that differ in nothing? |

**Rule E: the extension, pre-committed, used at most once.**
- **Trigger:** Z1 holds at 3 seeds, but Z2 or Z3 fails.
- **Action:** run seeds 3 and 4 with the identical argv apart from `--seed`, then re-apply the rule over 5 seeds:
  - Z1 on the 5-seed mean;
  - Z2′: at least 4 of 5 Δz_s < 0;
  - Z3 with F_z as the range over the 5 controls.
- **Results:**
  - The conditions now hold ⇒ ZEROS-FALL.
  - They still fail ⇒ **SEED-FRAGILE**.
- If Z1 fails at 3 seeds, there is no extension.

**PDMS (a guard, not the endpoint).**
- ΔPDMS̄ = the seed mean of [PDMS(B_s) − PDMS(A_s)] ×100, read on the same draws.
- **GO requires its point estimate to be > 0.**
- Reported beside it: each arm against the base, and A_0 against the banked C0 (a rig check).
  - F_pdms = max_s PDMS(A_s) − min_s PDMS(A_s), the PDMS range of the controls. If A_0 − C0 separates **and** exceeds F_pdms, the verdict carries a **RIG-DRIFT** flag for the Master Mind. The outcome is
    unchanged.

## 8. Safety co-gate (`D-REFE-FREEPOINT-1`)

**Why it exists:** R0B gained PDMS +0.55 [+0.24, +0.84] on navtest but paid NC −0.25 [−0.45, −0.06] and TTC −0.40
[−0.71, −0.09] (INHERITED).

**Reads.** ΔNC̄ and ΔTTC̄ are the seed-mean paired differences of the pick's NC and TTC sub-scores (×100), using the §6 draws.

**The co-gate FAILS iff:**
- UB(ΔNC̄) < 0, **or**
- UB(ΔTTC̄) < 0, **or**
- for NC, or for TTC, at least 2 of the seeds individually read UB < 0 on their own paired interval.

**Reported beside it, per seed:** DDC, EP, and the NC-zero pick count.

## 9. Mechanism reads (they choose the next lever, never the GO)

- **M1, the zero split.** Δz̄ split into DAC-only, NC-only and both, with the same draws.
- **M2, the head on REAL grazes.**
  - Pairs: `zt4_baseline.twin_pairs`, unchanged. A pair has max-over-8-poses ≤ 0.5 m and DAC_i ≠ DAC_j, on the executed navsim
    plans.
  - Population: every RL-val set with ≥ 1 such pair. ZT4's own population (each arm's DAC-only selection-zero sets) is also
    reported.
  - Metric: pairwise DAC accuracy from each arm's DAC head, as a ratio of per-set sums.
  - **HEAD-MOVES iff LB(ΔM2̄) > 0**, where ΔM2̄ is the seed-mean of B − A.
  - Shown beside it: ZT4's 0.70 bar, and u400's 0.392 [0.268, 0.571] on navtest (`ZT4_BASELINE.json`, INHERITED).
- **M3, the head on the BANK'S geometry, held out.**
  - Sets: the RL-val bank (§2.6). Each offset is scored by each arm's final checkpoint as a 65th candidate in its real RL-val fan
    (§2.4's masked call), read from the cache with no training.
  - Pairs: within a scene, DAC-head pairwise accuracy between DAC = 1 members (anchor or offsets) and DAC = 0 offsets. Scenes
    lacking either side are counted.
  - **LEARNS-BANK iff LB(ΔM3̄) > 0.**
  - ⚠️ If fewer than 100 RL-val scenes qualify, M3 is reported NOT COMPUTABLE, with its n. Rows 8 and 9 of §11 then merge as
    "HEAD-BLIND OR SYNTHETIC-ONLY, unresolved".
- **Train-side.** Arm B's hard-negative loss over updates is logged and reported. It is not a held-out read.

## 10. Four families (binding: per family, never pooled; seed-mean paired differences on the §6 draws)

| family | components on RL-val |
|---|---|
| **LONGITUDINAL** | EP and TTC sub-scores. Speed MAE and bias, and along-track MAE and bias, against the logged human future. Headway or time-gap to the lead where a lead exists, with n and the reason where none does. EP is watched specifically (DA-WAM paid EP −0.50, PUBLISHED). |
| **LATERAL** | DAC sub-score. Heading, cross-track, curvature and yaw-rate MAE (yaw rate masked at standstill, `RETR-2026-09-26-YAWMASK`). Last-pose heading error. |
| **TACTICAL** | Lateral, longitudinal and 5-way decision accuracy and κ. Command adherence of the pick. Selection quality as random / actual / best-of-64 on the same sets. The random control is mandatory. |
| **STRATEGIC** | **PAUSED** (PI 2026-09-19, `PI_DECISION_QUEUE.md` item 25). Not a pass and not omitted. |

- **The PDM sub-scores gate** through §7 and §8. The trajectory-level components are **reported, not gating**: the screen is a
  selection tier, and with 21 components a gate would make false NO-GOs likely.
- **Any LONGITUDINAL or LATERAL component that separates adversely** on the seed mean is carried as a named **watch-item** into
  the full arm's amendment, where HN-ADOPT gates on it.
- **The result is not published without all four families present**, or each one stated NOT COMPUTABLE with its reason and n.

## 11. Outcomes and next levers (Rule Zero; committed now; first matching row wins)

| # | outcome | condition | full P6-HN arm | next lever, executed in the same run as the verdict |
|---|---|---|---|---|
| 0 | **SCREEN-INVALID** | a §5 gate (except SY), G14, G15 or the launch gate fails | not decided | fix the failed control, re-run that stage; nothing is read |
| 1 | **SCREEN-SPARSE** | SY < 0.10 | **NO-GO** (no supply) | the draft's SPARSE lever: pair-contrast on the fan's own discordant twins, under its own prereg, on this rig |
| 2 | **SCREEN-UNSAFE** | the §8 co-gate fails | **NO-GO** as specified | (i) 0-GPU attribution of changed picks by the sub-score that moved, from the dumps. (ii) **Arm S-DAC, pre-registered here:** this screen unchanged except that the bank keeps only DAC-only hard negatives. It reuses the cache and labels and costs 3 passes; read under §7-§11. An S-DAC GO ⇒ the full arm with a DAC-only bank, by amendment to the draft before its launch. |
| 3 | **SCREEN-GO** | ZEROS-FALL ∧ co-gate passes ∧ point ΔPDMS̄ > 0 | **GO**: meets the PI's 2026-10-18 condition. Launch still needs the draft's other preconditions and its registration. | Amend the draft (dated, before any P6-HN step) with §2.4's masked form, the screen's F_z, M2 and M3, and the watch-items. Then the 0-GPU selection-rule read on B's dumps, reported. |
| 4 | **NO-GO / NET-ZERO** | ZEROS-FALL ∧ co-gate passes ∧ point ΔPDMS̄ ≤ 0 | **NO-GO** as specified | The term hits its target but selection pays elsewhere. Next: 0-GPU EP-versus-safety calibration of the selection rule on B's dumps, cross-fitted over 5 log folds of RL-val (the SELRULE / R0D form). A cross-fitted net > 0 goes to the PI as a re-proposal of the full arm with that rule. |
| 5 | **NO-GO / SEED-FRAGILE** | rule E fired and still fails | **NO-GO** | The head-level effect does not survive training variance. Next: row 8's lever if LEARNS-BANK, else row 9's. |
| 6 | **NO-GO / REVERSED** | LB(Δz̄) > 0 | **NO-GO** | 0-GPU attribution by \|δ\|, side and sub-type of the offsets whose labels push picks into zeros. Then a label audit for tracker-made zeros on offsets (ZT3: 42 % tracker-made, INHERITED). |
| 7 | **NO-GO / HEAD-MOVES-NO-ZEROS** | not ZEROS-FALL ∧ HEAD-MOVES | **NO-GO** | The head sees grazes, but the pick does not change. Next: 0-GPU DAC-veto read with B's head (the SELRULE V(τ) instrument) on B's dumps, cross-fitted. If it reaches zeros, re-propose. |
| 8 | **NO-GO / SYNTHETIC-ONLY** | not ZEROS-FALL ∧ not HEAD-MOVES ∧ LEARNS-BANK | **NO-GO** | The offsets are not the fan's real failure geometry. Next: pair-contrast on the fan's own discordant near-twins as the next screen on this rig and cache. It needs no new labels. |
| 9 | **NO-GO / HEAD-BLIND** | not ZEROS-FALL ∧ not HEAD-MOVES ∧ not LEARNS-BANK | **NO-GO** | Representational at the frozen trunk (consistent with `D-REFE-SCENETAP-1`: frozen visual features carry the road to IoU 0.637, INHERITED). Next: P7.1, the map-supervised drivable-area aux head on the trunk. The frozen-P6 re-screen runs iff P6's ZT4 reads HEAD-MOVES against u400 (§1). |

## 12. Power (ESTIMATED, disclosed before any number)

**Supply of zeros.**
- u400 picks **68** DAC = 0 sets of 3,000 on RL-val (`D-REFE-SELRULE-FIT`, INHERITED).
- The NC-zero count on RL-val is not in any doc read here (UNVERIFIED), so total zeros are ≥ 68.

**Expected cut.**
- DA-WAM's matched pair cuts the NC and DAC failure masses by about 21 % (NC 98.88 → 99.11, DAC 98.58 → 98.88; `2608.19085`
  Tab. 3, PUBLISHED).
- At that rate the cut is **≈ 14-25 picks (0.47-0.83 pp)**, ESTIMATED.

**Detectable cut, for one paired seed.**
- Assume about 2c sets are discordant for a net cut c, and log clustering inflates the variance ×1.5 (both ESTIMATED).
- Then SE ≈ 1.22·√(2c), and separation needs c > 1.96 · 1.22 · √(2c) ⇒ **c ≳ 12 picks (≈ 0.4 pp)**.
- The seed mean does not shrink the log term, so this holds for Δz̄.

**What follows.**
- **A DA-WAM-sized effect is near or above the screen's resolution.** On ΔPDMS it would not be: RL-val's half-width is about 0.58
  (R0 − C0 +1.16 [+0.59, +1.74], INHERITED), against an expected +0.22.
- That is why the endpoint is zeros.
- F_z is UNMEASURED until the controls run.

## 13. Compute and where it runs (ESTIMATED; arithmetic shown; 0 A40-h)

| stage | where | arithmetic | result |
|---|---|---|---|
| S0 staging | pod (read-only) → dev box | jpgs: 19,000 × 4 × 0.195 MiB (P2HO measured 5,488 files = 1,069.9 MiB). DBs: 241 × 86.6 MiB (11 DBs = 952.3 MiB). Both INHERITED from `raw/2026-10-18-p2-heldout-table/HANDOFF_2.md`. | **14.5 GiB + ≤ 20.4 GiB** |
| S1 labels (bank + SY + I1) | dev-box CPU | 19,000 scenes × 9 trajectories. Cost ∝ trajectories: 10.2 CPU-s × 9/64 = 1.43 CPU-s per scene. Cost ∝ sets: 10.2 CPU-s per scene (RL-S1's measured per-set cost, INHERITED). | **7.6-53.8 CPU-h** = 1.9-13.5 h wall at 4 workers (3.8-27 h at 2) |
| S2 trunk cache (+ I6, G15) | **dev-box GPU, behind `devbox_gpu.lock`** | 19,000 × 1.35-1.65 s per token (P2 tables' dump rate on the dev box, INHERITED, a full-model forward). Units of ≤ 30 min with the lock released between them, so SR-TTC S0 interleaves. | **7.1-8.7 GPU-h**; disk 19,000 × 3.75 MiB = **69.6 GiB** |
| S3 paired passes | dev-box GPU, behind the lock | Per pass: 64,000 sample-visits. I/O 64,000 × 3.75 MiB = 234 GiB at 200-400 MB/s (UNMEASURED). Compute ≈ 70 GFLOP per visit for base + A + B, about 5 min. | **0.3-1.0 GPU-h per pass**; ×3 = 1-3 h (+0.7-2 h if rule E fires) |
| S4 reads | dev-box CPU (+ minutes of GPU for M3 scoring from the cache) | rl_val_eval lookups, M2, M3, families | **≤ 2 h** |
| **total** | | | **≈ 8-12 dev-box GPU-h; 0 A40-h; ≈ 105 GiB of disk; ≈ 1.5-3 days wall** |

**Against the alternatives:**
- the same screen without the cache: 3 × 64,000 × 1.35 s ≈ **72 GPU-h**;
- on the A40: 3 × 3.4-3.5 A40-h (C0 / R0 measured, INHERITED). Not available: the A40 is P6's (PI);
- the full arm: **6.8-7.65 A40-days**.

**Run-box rules.**
- The dev box (RTX 4060, 8 GiB) cannot train ViT-L × 4 (`raw/refe_vitl_devbox_memory.txt`). The screen never backpropagates into
  the trunk, and the frozen forward already runs there (the P2 tables).
- **Box 2** (a second RTX 4060) can take S2 units in parallel if the Master Mind assigns it.
- The pod is touched only as a read-only data source:
  - before P6 launches, or under P6's overlap rule (nice 19, at or below its cap);
  - never adding GPU or RAM load beside a training run.
- The chain writes a HANDOFF and ends; no agent waits on S1-S3 alive (token budget rule).

## 14. Not changed, out of scope

**Not changed:**
- the trunk, the fan, the partition, the labels and the labeller;
- the recipe, the selection rule and the inference inputs (the lever is training-only);
- the draft and its bars.

**Out of scope:**
- map-derived negatives (C9);
- the trunk channel;
- navtest;
- any re-selection of scenes.

## 15. Sources

**Package documents**
- the draft and `NOTES.md` (this package);
- `eval/PREREG_RL_S1.md` (partition, recipe, OPEN-BASE, §10.3 budget) and `eval/rls1b_arms.json`;
- `eval/rl_val_eval.py`, `eval/REFE_LAUNCH_GATE.md`.

**Code**
- `refe/scorer_finetune.py` (`fwd`, `losses6`, gates G1-G13);
- `refe/model.py` (`CrossBlock` :425-438, `score_trajectories` :699-709, the forward :759-793);
- `refe/gate_checks.py` (`DVB_KINDS_SCORER`, `SCORER_GATES`), `refe/launch_gate.py` (`PROFILES`);
- `refe/onpolicy_relabel_pdm.py`, `refe/diag_p6_labeller_identity.py`;
- `raw/2026-10-17-zero-tail-mechanism/code/zt4_baseline.py`.

**Raw artifacts**
- `raw/2026-10-18-p2-heldout-table/` (PLAN.md, HANDOFF_2.md, `code/p2ho_featcheck.py`), `raw/2026-10-16-p2-full-table/`;
- `raw/refe_vitl_devbox_memory.txt`;
- `raw/2026-10-18-hn-census/PREREG_HN_CENSUS.md`.

**Registers**
- `Project Steering/MODEL_REGISTRY.md` (row "SFT-4 B u400");
- `Project Steering/GOALS_AND_CLAIMS.md`: `D-REFE-SFT-4`, `H-REFE-P2-REP-1`, `D-REFE-FREEPOINT-1`, `D-REFE-SELRULE-FIT`,
  `D-REFE-ZEROTAIL-3`, `D-REFE-SCENETAP-1`, `D-REFE-HN-CENSUS-1`.

**Library**
- `2608.19085` (DA-WAM), `2605.19771` (BeyondDrive). Both were banked and checked against `library.json` per `NOTES.md`.
