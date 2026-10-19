# BACKLOG — the pull-list when the headline work is gated

**Purpose:** `CLAUDE.md` forbids ending a turn having only reported. When the top item is blocked,
**pull from here and execute in the same turn.** Gated ≠ idle.

**Rules for this file:** every item states **what unblocks it** and **whether it needs a GPU**.
0-GPU items are always executable — "blocked on the PI" blocks one item, never the programme.
Strike items through when done, with the commit.

---

## A. 0-GPU — ALWAYS EXECUTABLE, no excuse to idle

| # | item | why it matters |
|---|---|---|
| ~~A1~~ | ~~Build the emitters~~ ✅ **AUDITED `1ad18ea`/`9e09b54` — NEITHER needs writing.** #2 needs an **EVAL** (re-emit `g_op_fwd_ade_m` from the checkpoint, needs a free GPU); #7 needs **ONE ARCH BRANCH** in `efficiency.build_case` (a SPEC question — its own scoped task) |
| ~~A2~~ | ~~Wire the NC gate with filter_m~~ ✅ **CONTRACT RECORDED `8d4a138`** — `filter_m` appeared NOWHERE in pseudosim (grep=0), so the gate would have shipped PDMS-v1 and over-penalised every arm. Contract now surfaced in `composite()` output; denominator 16 is a live tripwire (12 = not EPDMS). ⛔ Still NOT applied — needs a human-reference rollout + the chunk download |
| A3 | **C64 option B** — build the clean v2-line val | ⭐ **BUILT 2026-08-02** (`…/incoming/2026-08-02-v2-clean-val-selector/`, 27 ✓). ✅ Column semantics are now a machine-checked contract (34/34 on 18,988 rows) — the blocker below is cleared. ⛔ **The 6.77× was ONE AXIS**: on junction+turn+speed it is **1.07×**, +brake **0.77 = infeasible**. ⛔ **Cell-quota matching does NOT balance** (max \|d\| 0.3997, 10/13 axes over bar) — greedy covariate balancing gets **0.0094**. ⛔⛔ **A clean v2 val is NOT clean for v1: 62 of a 600-draw are in v1's TRAIN** ⇒ manifests exclude the parity corpus, and then **600 is unavailable** (headroom 0.95). **SHIPPED: n=400**, max \|d\| 0.0409, sha256 `abe041db72a045b3…` (+n=300 variant). ⇒ **now a PI decision: freeze n=400, or reject B** — no draw from this remainder is exchangeable with train (within-cell median \|d\| 0.359) |
| A4 | **Re-verify the factorised path × velocity vocabulary** (refuted 0-3 in the survey) | maps 1:1 onto our LAT+LON softmax mechanism and the 88.7 % longitudinal gap — the single most valuable re-verification target |
| A5 | **Confirm C64 at clip_id granularity** — replay `discover_r0_clips` on pod2's raw root | 🟡 **PARTLY ANSWERED 2026-08-02 with NO pod**, from the two selection parquets: **256 of v1's 600 parity-val clips (42.7 %) are inside v2corpus's 9,000-clip training selection** (1,311 of the 9,000 come from the parity 3,000). ⚠️ **not** the same statistic as C64's 21/40 EVAL EPISODES — different unit, never merged. The pod2 replay remains the exact check but is no longer the only source |
| A6 | **Reconcile pod2's tree** — 317 modified tracked files, untracked `taniteval`, **untracked running trainer on one disk** | ⛔ **after v5 finishes, never during an incident** |
| A7 | **Delta-JEPA one-liner**: IDM decoder takes latent **displacement** `z_{t+1}−z_t` instead of concatenated endpoints | HYPOTHESIS-class lead; bears on our smaller-IDM-is-better ladder (input form, not capacity) |
| A8 | **Sub-JEPA regulariser swap**: isotropic prior in random **low-dimensional subspaces** vs our ambient `SigRegConfig` (`n_slices=512`, w 0.1) | HYPOTHESIS-class lead, public code exists |
| A9 | **Task-list hygiene** — #37/#39 are marked completed but were falsified (pod1 still down; pod3 had drifted back) | a task list that lies is worse than none |
| A10 | **Bank gen-1 `sc_train.py`'s remaining pod-only siblings** if any | stranding rule |
| A11 | ~~⛔ **Rescue the 45 stranded files on Thor**~~ ⏹ **DONE 2026-08-18, 0 GPU** (`…/incoming/2026-08-18-thor-stranded-rescue/`) | ⭐ **It was 98, not 45.** `pod_git_drift.py` scans **`.py`/`.sh` only** (`:91`), so every stranded result JSON, run log, `.md` and `.bak` was invisible to it; a content-hash sweep found **102 files on Thor absent from the repo by content**, of which **98 are now banked, 98/98 sha256-verified both sides** and 4 left with reasons (pypa `get-pip.py`; two regenerable NuRec extracts; one byte-identical dup). A11's own count also undercounted the `/home/nvidia` root by 2 (**21**, not 19). ⛔ **New A13-class item found:** the banked `…/2026-08-02-thor-deployment-profile/thor_profile.py` **cannot have produced its own `thor_profile.json`** (the JSON carries `"frame"`, the script never sets it) — the producer was Thor-only and is now in `rescued/home_nvidia_root/`; **the drift tool classified it `NAME_ONLY` = "weak evidence"**. `*.mp4`: **none** in the eight dirs; of 26 elsewhere on Thor, 18 are already banked byte-identical (md5-checked) and the 8 others are superseded lower-quality render passes ⇒ **no `git add -f` needed** |
| ~~A12~~ | ~~Ship the 4 stale/absent files to Thor~~ ✅ **DONE 2026-09-02 — 4/4 now on Thor.** `lf0_bev_lead.py`, `p8_bev_reel.py`, `refc_train.py` shipped **2026-08-15**; `sel_winners_curse_law.py` shipped tonight to `/home/nvidia/TanitAD/stack/scripts/`, md5 `80819c1f42c469b42a12b76db280c5fe`, 18,266 B, verified on the far side by md5 **and** by reading back its header. ⚠️ **The row's own "(absent)" annotation was STALE** — the file exists and is git-tracked at `stack/scripts/sel_winners_curse_law.py`; it had simply never been shipped. Shipped now rather than "with the next legitimate sync" because the caveat was *"not mid-run"* and Thor is idle (GPU 0 %, zero training processes) — the refcv3 run is on the A40, untouched. | found only after the closure audit's entry points were **derived** (161 files from 52 entries, vs 134 from the old hand-list of 14) |
| A14 | **Decide on 24 un-pulled per-window rollout dumps on Thor** (`cl_out*`/`ol_out*`/`cutin_out` `rollouts_*.json`, `video_*_openloop.json`) — **9,550,490 B**, mostly `long_*` re-runs | found by the A11 rescue's second sweep, **outside A11's eight dirs**. ⭐ **Nothing analytical is at risk** — all 6 `OL_PANEL_*.md` and every script there are already in the repo by content, and the 19 gate/summary JSONs are now banked. Path+size+sha256 of all 24: `…/incoming/2026-08-18-thor-stranded-rescue/raw/beyond_a11_NOT_pulled.json`. **Owner = the AlpaSim closed-loop stream** — a 9.6 MB bank of another stream's raw evidence is its call |
| ~~A15~~ | ~~Adjudicate `thor_profile.py`~~ ✅ **DONE — BOTH HALVES, verified 2026-09-02 (the row was stale, not open).** **(a) The adjudication landed 2026-08-18**: the package now carries `PROVENANCE_CORRECTION.md` (4,553 B) and the banked `thor_profile.py` self-documents the contradiction in its own header — *"it never assigns `out[\"frame\"]`, yet the JSON carries `\"frame\": \"176x624 hfov 117.0\"`"* — plus a second correction that the script's 120° comment is the PARENT cache's field, not the 176×624 sub-frame's. The pair no longer reads as provenance. **(b) The tool fix landed too**: `pod_git_drift.py` now splits `SUFFIXES` into `SOURCE_SUFFIXES` (`.py .sh .md .yaml .yml .toml .cfg …`) and `ARTIFACT_SUFFIXES` (`.json .jsonl .log .txt .csv .tsv …`) at `:137–150`, and splits the verdict so a program-authored basename becomes `NAME_DRIFT` (a real finding needing adjudication) while only a genuinely ambiguous name (`__init__.py`, `utils.py`) stays the weak `NAME_ONLY`. | ⚠️ **The standing finding survives the row's closure** and is worth keeping: *a banked script that cannot produce its banked result is worse than a missing one — the pair looks like provenance.* |

⚠️⚠️ **THIS PULL-LIST IS CARRYING COMPLETED WORK, AND THAT DEFEATS ITS PURPOSE (MEASURED 2026-09-02).** `BACKLOG.md` exists so a GPU-gated turn still ships something — it is the answer to "gated ≠ idle". But four consecutive items pulled from it tonight were **already done**: refav1's loader v7.2/nav join (implemented, 169 tests green, while the loader's own docstring still warned it was not), refav1's compute-placement blocker (Thor idle since the k=8 control finished), A15's adjudication (2026-08-18), and A15's tool fix. A12 was three-quarters done with a stale *"(absent)"* on the remaining quarter. ⇒ **a pull-list whose rows are stale costs exactly the turns it exists to make productive** — the puller spends the turn re-deriving state instead of doing work, which is the idling the rule forbids, wearing a checklist. **Sweep before pulling, and close a row in the same turn its work lands** — the same discipline `lab_backlog_drift.py` enforces for the Lab's list, which this list has no equivalent of. *(Root-cause class: a blocker note is not revisited when the thing it blocks on lands, so it keeps reading as a live gap — the stale-blocker class the operating standard already names.)*
| A13 | ~~**Adjudicate the 24 DRIFT rows in Thor's live `/home/nvidia/TanitAD` checkout**~~ ⏹ **DONE 2026-09-07, 0 GPU** (`TanitAD Research Lab/Tools & DevEnv/Research/2026-09-07-thor-checkout-drift-a13/`) | ⭐ **"Direction unknown" was answerable all along** — the box is a git checkout whose HEAD `30d6d60` (2026-08-15) is a **VERIFIED ANCESTOR** of ours (`merge-base --is-ancestor`, rc 0), 1,273 commits back ⇒ for 133 rows the direction is a **positive identification of a commit**, not an inference; 7 more by a range-limited history walk, 16 by a control-bracketed object-store probe, **0 inconclusive**. **THE REPO IS AUTHORITATIVE ON EVERY ROW — no row where Thor is newer, nothing on Thor is unique.** ⚠️ **The count is 156 real drift** at HEAD `7084b2cf` (165 raw-sha minus **9 that are CRLF-only**, from files shipped off the Windows box), not 24 — the row's 24 was not wrong, it was **DATED** (2026-08-18). All 24 originals reproduce, are still drifted, and are settled. ⛔ **THE ROW'S PREMISE WAS STALE AND SO WAS THE BRIEF I WROTE FROM IT: refav1 is NOT training on Thor** — it finished 2026-09-04 (`summary.json` done:true, `final_step 21109`); 0 python processes on 3 independent probes, GPU 0 %, re-verified by the Master Mind before acting. ⇒ **Thor is FREE COMPUTE.** ⛔ Retrospectively the question still bites: **24 of the 130 import-closure files are real drift and 13 more do not exist on the box**, and the banked arm ran a materially older stack (`refa_v1.py` **1,885 lines vs the repo's 3,030**) ⇒ **a re-run from today's repo is NOT a replication of it.** ⭐ **Nothing is stranded** — all 3,837 repo-absent files are `30d6d60` blobs, and every CRLF-shipped file's LF form is a committed blob. ⛔ **Do NOT sync Thor wholesale**: a blanket ship would destroy that arm's byte-level provenance for no gain. | ⚠️ Two self-caught instrument defects worth keeping: **blob shas are over RAW BYTES**, so a first pass scored **line endings** as drift and reported `parity.py` — which a safety oracle loads — as existing nowhere in git (caught by a **differently-derived** cross-check disagreeing, not by re-running the same probe; re-probed on LF forms, 14/14 present); and driving off Thor's `git ls-tree` covers **committed** paths only, while the entire refav1 stack **including its entry point** is UNTRACKED there — an instrument blind to exactly the files the question was about (class C110) |
| A13b | ⭐ **Ship the import closure to Thor before the NEXT launch, and gate on it** — **29 of 130 closure files differ and 13 are absent**; the checkout is 1,273 commits behind. ⛔ Ship the **CLOSURE, not the tree**, and only when a launch is imminent. | ✅ **The gate's MSYS defect is FIXED (2026-09-07)** — `pod_currency_audit.py` was unrunnable from the dev box because MSYS rewrites `SCAN_ROOT=/home/...` into a Windows path; the scan then found **nothing** and the audit reported **no drift**, ⛔ **a FALSE ALL-CLEAR on a pre-launch gate**, which is worse than a crash because it looks like a pass. Now `check_pod_root()` **refuses** a mangled root and names `MSYS_NO_PATHCONV=1`, and a scan returning **0 files** is UNTRUSTWORTHY rather than clean. Pinned by `stack/tests/test_pod_currency_audit_guards.py` (11 tests) whose RED arm asserts **behaviour** — that `pod_scan` is never called for a mangled root, and IS called once the validator is removed — rather than the tautology of asserting a neutered function returns "" |
| A13c | **Record the code provenance of `refav1-b1-v72-ep3-speed` in `MODEL_REGISTRY.md`** — it was trained from Thor's OLDER stack, not from repo HEAD | flows from A13; a registry row that does not say which code produced a checkpoint invites the re-run-is-a-replication error |

| **A16** | ⛔ **TO THE MASTER MIND — answer the six frozen-trunk questions for refcv6 · v7f · refav1** (`Project Steering/ADVISORY_FROZEN_TRUNK_DEFECT_CLASSES.md`, register row `D-TRUNK-ADVISORY-1`). Six classes MEASURED on REFe 2026-09-20/21, every one of which had survived a green suite: (A) trunk fed **un-normalised** images, (B) a positional tensor with **no checkpoint counterpart** and a "3D" encoding that consumed no geometry, (C) registers **concatenated instead of compressing** (1,936 tokens where 16 were intended, identical storage pointer), (D) units/rates (radians averaged with metres; a **2x** velocity from the wrong time base), (E) a **nominal** category aggregated with `max`, (F) **guards that cannot go red**. ⚠️ PI-directed 2026-09-21: Sayed asked for this to be sent to the Master Mind directly. ⛔ **The live session-to-session channel REFUSED** (`send_message`: *"unavailable in unattended sessions"*), so it is filed here and in the register instead — if it has not been picked up, the PI can forward it in one click. | **0 GPU, ~1 h per model.** Two MORE instances landed the same night, in REFe's own 4-camera extension: `build_targets.py` raised `AttributeError: 'dict' object has no attribute 'size'` on the **first step of the first log** (a single-camera leftover nobody executed), and its image index matched `_CAM_F0.zip` only, so three of four camera containers were invisible and every tuple would have been dropped as a *data* problem. Both are class F. ⭐ The general instrument is now written and self-tested: `refe/diag_consumer_conformance.py` — 8 arms, each pure over every component's belief, each proven to go RED alone under its own historical divergence. It is architecture-independent in shape and worth porting. |

## B. ONE GPU — executable now (pod3; ⛔ eval stays free for C64-A)

| # | item | notes |
|---|---|---|
| B1 | ⭐ **RR-FT** — fine-tune v1 with `--rollout-k 20`, exactly one flag different, ~3–5 k steps | **IN PROGRESS.** Measure `step_s` over 200 steps and report before committing to full length |
| B2 | **E-ROLL** — recursive k=1 rollout past 2 s on the deployed arm, measure ADE **and** CR to 4 s/6 s | ~2–4 GPU-h; divergence is the *expected* result and is informative |
| B3 | **Linear free/occupied probe** on frozen v1 latents from `obstacle.offline` | <0.5 GPU-day; depends on A2's chunk download |
| B4 | ~~**Matched-capacity camera head** for sitclf~~ **DONE 2026-08-03, 0 pod GPU-h** (`…/incoming/2026-08-03-sitclf-matched-capacity/`) | ⭐ **The curve PEAKS AT THE FLOOR.** On `intersection` (the only decision-grade situation, 230 clusters with a positive) a **129-param** ridge beats the deployed **417,028-param** head by **+0.3512 [+0.1269, +0.6049] SEPARATED**, on precision (0.2490 vs 0.1608 at the same 3,550 fires) as well as recall. The transformer is **FLAT 1.33–1.46 across 2,068→2,207,572 params**. ⛔ **NOT an optimisation shortcut** — 3× the epoch budget moves nothing. ⛔ **P8's "the head is the bottleneck" was read off `roundabout`, which at 39 clusters CANNOT SEPARATE** (+1.31 [−0.21, +4.39]); it REPLICATES on `intersection` and **REVERSES on `lane_change`** (−0.2795 [−0.5624, −0.0093]). ⚠️ This row previously said "~2k to ~2.17M" — both counts were transplanted from other streams, see `RETRACTION_LOG.md` **R-2026-08-03-h**. Next lever is the **INPUT** (B5), not the head |
| B5 | **Frozen VIDEO-pretrained encoder** (V-JEPA 2 ViT-L) vs from-scratch, same corpus/steps | ⭐ the one experiment nobody in the field has run; our frozen-encoder ceiling was measured on IMAGE-pretrained encoders only |
| B6 | **REF-C-base canonical eval + v1.6 paired bootstrap** (task #31) | queued since pod3 freed |
| B7 | **Opponent SC-13 on our checkpoint** (task #10) | |
| B8 | **Cosmos Reason1 vs Reason2 head-to-head** (task #29) | VLM labeler choice |

## C. GATED — and on what, precisely

| # | item | blocked on |
|---|---|---|
| C1 | **Rollout-recovery at scale / 8-GPU work** | ⛔ **pod1 console stop/start (PI)** — `/dev/nvidia*` empty, not fixable over SSH |
| C2 | **C64 option A** — score both arms on the 19 leak-free episodes, **v1 re-scored there** | ~~v2corpus reaching 30 k (~17:00 UTC 2026-07-29)~~ ⚠️ **RE-PROBE 2026-08-16 — the gating DATE passed 18 days ago and the gating FACT is unverified.** `MODEL_REGISTRY.md:953` §1.7 still reads 🟢 **RUNNING** with ETA **2026-07-29T01:10Z**, and no completion row was found for `flagship-v2corpus-30k`; meanwhile C1 below records pod1's `/dev/nvidia*` as empty. A 30 k arm on that corpus **has** been trained and evaluated (`flagship-v1arch-v2bal-30k`, OOD-val four-family run 2026-08-05), but that is the **v1-architecture / levers-false** arm, **not** §1.7's run. ⇒ **UNVERIFIABLE, not cleared — probe the run before waiting on it.** Swept by the 2026-08-16 stale-blocker sweep |
| C3 | **v5 gate verdict** | ~~v5 reaching step 2000; ⚠️ first probe is the INCUMBENT and cannot stop the run~~ ⏹ **CLOSED 2026-08-16 — blocker CLEARED 2026-08-09.** `flagship-v5f-w120-30k` is ✅ **COMPLETE at 30,000** (2026-08-09T19:23Z). Evidence (MEASURED): `Project Steering/MODEL_REGISTRY.md:993` §1.8. Swept by the 2026-08-16 stale-blocker sweep |
| C4 | **Old CPU pod release** (task #40) | ⛔ deletion needs the PI |
| C5 | **X2 verdict run (30 pod-days)** | ⛔ NOT AUTHORISED without the PI |
| C6 | **Wheelbase fix** | ⛔ PI chose C = measure first; decision pending |

## D. Standing / long-horizon

| # | item |
|---|---|
| D1 | **v3: DINO-WM proper** — feature-prediction + CEM/diffusion/MPC, no head (task #26) |
| D2 | **TanitEval clickable interactive tool** (task #23) |
| D3 | **H26 hierarchical cross-alignment proof** (task #15, core goal) |
| D4 | **Own dataset / lake v0** — ingest at scale + HF push (task #5) |
| D5 | **flagship-v2 10k gate** mechanism diagnostic (task #28) |

---

## E. 2026-08-11 NIGHT — the live pull-list (supersedes A/B where they conflict)

**Running (do not duplicate):** pod4 p8c4 BEV → W7-FULL → belief reel → PH0 smoke → PH0
mini-pilot+videos · pod5 H-COTRAIN milestones → T1 rows → four-families.

| # | item | GPU? | unblocks on |
|---|---|---|---|
| E1 | **Four-families rescore on the banked v5.8f windows** → completes registry §1.14 | 0-GPU (banked windows) | nothing — runnable the moment T1 lands |
| E2 | **HF release of the v5.8f artifact set** (ckpts, gates, windows, figures) | 0-GPU | the release row existing |
| E3 | **W5 / E-H1 6 s baseline for v5.8f** — REQUIRED precursor now that 6 s is the v6 spec | 1 GPU | pod5 after T1 |
| E4 | **I4b** — imagination ablation stratified by the P4/P8 occluded split | 1 GPU | p8c gate + the banked I4a arms |
| E5 | **LF0** — probe PRE-POOL spatial tokens + BEV lead read-off (routing vs learning) | ~0.5 GPU-h | p8c head existing |
| E6 | **PERCEPTION-AGENTS head** (slot decoder on frozen latents: bbox+state+class) | ~1 GPU-day | prereg + v6 GO |
| E7 | **Alpamayo meta-action → vocabulary mapping table + coverage measurement** | 0-GPU | records.parquet (present on pod4) |
| E8 | **E-ENC arm prereg** (shared encoder + adapters vs per-layer) incl. B5's frozen V-JEPA-2 control | 0-GPU to write | v6 GO |
| E9 | **Spectrum finding follow-up**: participation ratio ≈ 4.5 of 2048 dims at 5k (top-8 = 99 % of variance) — is this SIGReg working as designed or an anisotropy finding? | 0-GPU analysis | the full H-COTRAIN curve |
| ~~E10~~ | ~~**Registry hygiene**: give the Alpamayo augmentation counts their own registry row (the paper carries them INHERITED)~~ ✅ **DONE 2026-10-18, 0 GPU** — registry **§11.3**: 114 counts re-derived from the bytes by `count_alpamayo.py`, each with artifact sha256 + counting expression + LAYER (`…/incoming/2026-10-18-e10-alpamayo-counts/`). ⚠️ The premise was stale (paper MEASURED + registry-anchored since 2026-08-16); **25 of 26 paper figures reproduce**; the one conflict is the paper's "4.3 % contaminated" (L199/L3073/L3924/L6600) — bytes and §12.4 say **78.21 %** (201/257); paper NOT edited → PI/Master Mind call. Also: CLAUDE.md "779 … with the colour" = 761 coloured + 18 colourless. The paper's 4.3 % contamination rate is corrected to the buildable-set 78.21 % in paper v1.5.1. | 0-GPU | nothing |

---

## F — 2026-08-12 v6 launch pull-list

**Why this section exists:** S-W is **GO on code** and blocked only on two PI decisions (D1 cost,
D2 pod) — `…/incoming/2026-08-07-hierarchical-wm-redesign/PI_DECISIONS_2026-08-12.md`. Per
`CLAUDE.md`, *a blocked PI decision blocks ONE item, not the programme*: **every item below is
0-GPU and executable while the PI sleeps and while S-W trains.**

**Collision rule:** each item names its **owner-file(s)**. Two agents must not hold the same
owner-file. Items with different owner-files are independent streams and may run concurrently.

| # | item | size | GPU? | owner-file(s) — do not overlap | unblocks on | why it matters |
|---|---|---|---|---|---|---|
| **F1** | ⭐ **S-S gate amendment (D6)** — promote `S1_ade_8_30s` from `reported` to `required` for the PH2-pending case, keeping STRATEGIC `n/a` **with its reason and its n**. Pre-register **before** S-S runs, never after seeing S1's number | ~1 h | **0-GPU** | `stack/scripts/train_v6_staged.py` (`STAGE_GATE_SPEC["S-S"]`) + `stack/tests/test_v6_staged.py` | nothing | **MEASURED:** S-S's only REQUIRED probe is `STRATEGIC_family` and its supervision (S2) is unwired until PH2 ⇒ the stage terminates at `pass: null` and **S-J refuses to launch**. Fixing it after S-S runs means an override instead of a gate |
| **F2** | **S-W supervisor manifest** — write `runs.d/v6-SW-30k.env` with the §2.1 `TRAIN_CMD` verbatim, plus the R7/R8 discipline in the header (done-marker = off-switch; manifest sourced ONCE at supervisor startup) | ~1 h | **0-GPU** | `stack/ops/runs.d/v6-SW-30k.env` (new) | D2 (pod name) | a 7–12-day run left unattended without a supervisor is one crash from a lost GPU-week; and a manifest written *after* the supervisor boots changes nothing |
| **F3** | ⭐ **Noise-robust selection rule prereg (R2)** — sweep top-m rules and the **`--w-prog`** anti-degeneracy term off the **already-banked** `w7_eval_windows.pt`. Pre-register the rule **before** S-T launches | ~3 h | **0-GPU** (banked windows) | `stack/scripts/w7_selection_rules.py` + a new `PREREG_SELECTION_RULE.md` in the incoming dir | nothing | **MEASURED:** the roll-cost argmin sits at error-rank **132.32 of 256** — the median (`w7_selection_rules.json`) — while top-m **ceilings do fall** (0.356 at m=32). `--w-prog` has been **weight 0.0 in every W7 run to date**. This is the one open risk with no built-in defence, and it is the S-T gate |
| **F4** | **Correct the cost basis name in `V6_TRAINER_DESIGN §5`** — "v4.2" → **`flagship-v4-fromscratch`** (registry §1.5.5). `flagship-v4.2-30k` (§1.5.3) is a different arm, killed at ~step 5 k | ~10 min | **0-GPU** | `…/incoming/2026-08-07-hierarchical-wm-redesign/V6_TRAINER_DESIGN.md` | nothing | registry-vs-doc conflict; the registry wins and the doc gets fixed. The *estimate* is unaffected (same two MEASURED runs) — the *name* is what would propagate |
| **F5** | **The 23 standing `pytest` failures** — `onnx` absent (1), a Windows-basename assert in `test_resim.py` (2), a suite-order polluter in `test_rig_clean_fix.py` (20, all pass in isolation) | ~2 h | **0-GPU** | `stack/tests/test_resim.py` + bisect for the `rig_clean` polluter | nothing | **none of them are v6**, but they stand against the *"`pytest -q` must stay green before any commit"* invariant, so every future commit inherits a red suite |
| **F6** | **E-ENC short-arm prereg (D5)** — the matched pair at a **reduced, equal step count**: `--per-layer-encoders` (120 743 881) vs shared `--pred-dim 960` (118 105 417), **gap 2.19 %** quoted. Decision metric: per-layer P-battery pass rate; **tie → shared** | ~2 h | **0-GPU to write** | a new `PREREG_E_ENC.md` in the incoming dir | nothing | stops E-ENC turning into a second 175–290 A40-hour S-W. Param counts **MEASURED (mine 2026-08-12)** by re-instantiation |
| **F7** | **Pod-side ship + STEP-ZERO runbook as a script** — md5 + `grep -c assert_isolation` + cache `ls` + `dd` write test + `--dry-run`, one command, refusing loudly on any mismatch | ~1 h | **0-GPU** | `stack/scripts/v6_preflight.sh` (new) | D2 (pod name) | ⛔ pods have **no git credentials** (`git fetch` HANGS; a failed-fetch `checkout -B` **RESETS the tree**) and **both pod4/pod5 direct SSH mappings were measured dead 2026-08-11**. The preflight is what stopped three chains running stale code |
| **F8** | **Verify `F` (frames per episode) on the pod** and record the realised window count per stage from the trainer's own `[v6] windowing:` line | ~15 min | **0-GPU** (rides F7) | `…/incoming/2026-08-07-hierarchical-wm-redesign/V6_GO_PACKAGE.md` §1 | D2 | the −42.6 % window figure assumes **F = 120**, which is **INHERITED** (cache name `…-w120-…`), not measured. General form `(F−66)/(F−26)`. D4's magnitude depends on it |
| **F9** | **Registry row skeleton for S-W** — pre-write the §1.15 row with every field the run must fill (parity key, skip-hash, derived `max_horizon`, realised window count, `step_s` MEASURED, gate verdict, tier stamp, estimator) | ~1 h | **0-GPU** | `Project Steering/MODEL_REGISTRY.md` §1.15 (new row) — ⛔ **registry-owner only** | nothing | *"an artifact on one disk is NOT done"*; a row written **before** the run cannot be back-filled from prose later |

⚠️ **Not in this list on purpose:** anything needing a GPU, and anything needing the PI's spend
decision. Those are D1/D2 in the decision sheet. **Gated ≠ idle** — F1–F9 are ~12 h of 0-GPU work
that all lands before S-W's first gate.

## refav1 pull-list (2026-09-02, Master Mind) — 0-GPU items that gate the first refav1 read

| id | item | est | GPU | pointer | depends on | why |
|---|---|---|---|---|---|---|
| **R1** | ✅ **DONE 2026-09-02 (D-REFAV1-LEAD-BLOCK)** — **Lead block for the 141-clip eval grid** — nearest lead agent per window from `obstacle.offline` (headway / time-gap / TTC), so LONGITUDINAL's distance-keeping half is PRESENT instead of UNAVAILABLE | ~3 h | **0-GPU** | `taniteval/tools/refav1_arm.py` (the `_unavailable` site), `stack/scripts/build_obstacle_join.py` | nothing | binding four-families rule: *a missing metric is a work item, not an excuse*; the adapter refuses with reason today |
| **R2** | ✅ **DONE 2026-09-02 (`3a6a48c`)** — **Review the UNSTAGED `taniteval/tools/t1_eval.py` worktree diff** (2026-08-30, *"window follows the checkpoint"*, +18/−2) — stage with provenance or revert | ~30 min | **0-GPU** | `git diff -- taniteval/tools/t1_eval.py` | nothing | the harness behind published T1 numbers carries an unreviewed local edit; the T1 adapter was tested AGAINST it |
| **R3** | **Run `cl_oraclegoal` (T0) beside `cl` (T1) at the step-1000 read** — the cheapest search-vs-goal discriminator: if the oracle goal also returns baselines, the search budget is the ceiling, not the goal | ~1 h dev-box GPU | 4060 | `taniteval/tools/REFAV1_ARM.md` | C-REFAV1-PLAN-NOGOAL fix landed | separates *planner cannot search* from *planner has no goal* before any T1 number is quoted |
| **R4** | ✅ **DONE 2026-09-02 19:26Z (switch step 16,500; D-REFCV3-NAV-SWITCHED)** — **`--nav-from-v7` for refcv3** — ship + relaunch at the next 500-step checkpoint ONLY on the PI's go (C-NAV-SOURCE-DIVERGENCE option d) | ~10 min ops | pod, no extra load | `stack/scripts/refc_v3_train.py`, `stack/scripts/sup_refcv3.sh` | ⛔ PI decision + the flag's tests green | refcv3's nav is `follow` on 94.6 % of B1 windows (MEASURED) |
| **R5** | ✅ **DIAGNOSED 2026-09-02 (cgroup OOM kill; D-REFCV3-EVAL-DEATH-DIAGNOSED) — fix (save-before-eval) in progress** — **Diagnose refcv3's silent deaths at eval boundaries** (2,000 / 4,500 / 6,000 / 10,500 / 17,000 — all `--eval-every 500` steps; empty stderr) — log-wrapped eval, cgroup `failcnt`/`events` read after a death, then `--eval-workers 0` or fewer `--eval-batches` as the discriminating change | ~2 h | pod, on the PI's go (touches the launch line) | `stack/scripts/refc_v3_train.py` eval block, `/workspace/experiments/refcv3-b1-v72-30k/supervisor.log` | C-REFCV3-EVAL-DEATH | each death costs up to 500 steps + a relaunch; the 17,000 one cost the checkpoint |

## Added 2026-09-03 04:10 Berlin (from D-V7-WIRING)

- **R6 — `--bptt-truncate` for the O1 stage-A rolls** (`stack/scripts/train_stage_a.py::stage_a_losses`, 6 of 8 `rollout_transitions` calls per step): one line + the same property test; owner needed before any k = 60 launch relies on the flag. 0 GPU.
- **R7 — tactical CE term on the v7.2 factored labels** (`tac_lat_id`/`tac_lon_id`/`tac_valid` already in the batch; `out["a_lat"]`/`out["a_lon"]`): a NEW loss term → `TanitAD_ValidateAIDesign` pre-registration (one variable, deliberate-regression arm, the v7-tiny rig), never a silent addition.
- **R8 — v7f LAUNCH BLOCKER: eval-clip exclusion in the v7 trainer's B1 path.** The 141 eval clips with pixels sit inside the 4,713-clip cache; `build_train_episodes`/`build_v2_providers` need an exclusion list keyed on the eval index (the refav1/refcv3 caches exclude them by construction). Test: the window census of a `--require-parity` launch must show 0 eval clips.
- **R9 — strategic ARG encoder for v7.2 labels** (spec §3.2, Data FlyWheel): a named dict today, no slot encoder; args unsupervised until it exists.

## Added 2026-09-03 07:20 Berlin (from D-ACTDIV-ANCHORED-REFAV1)

- **R10 — the PLANNER COST-SURFACE probe** (the next refav1 read): evaluate `_cost_chunk`'s four terms SEPARATELY at h = 10 on the banked step-1,000 checkpoints, in BOTH conventions (raw κ, and `κ → arctan(L·κ)` at the model boundary), so the three-factor explanation (intrinsic lateral potency × the ×2.9 under-actuation × the `0.05·κ²` penalty charged at full κ) is quantified and its repair costed in ONE panel. GPU: minutes on the 4060.
- **R11 — wire the new diagnostics in**: `taniteval/tools/actdiv_anchored.py` and `transition_probe.py` are committed but called by nothing — `mm_e19_read.py`'s actdiv stage still calls the banked `actdiv_local.py` (which carries the H-LEAK-1 speed-scale defect). Route the v7 reads through the anchored tool.
- **R12 — read the banked GS-9 transition-probe JSON**: the rescued package's RESULT §4 still says pending while its raw JSON exists and is unread.
- **R13 — `t1_eval.DEFAULT_TIERS` needs `"ha0": "T1"`** (`taniteval/tools/t1_eval.py:145`); until then the standalone CLI needs `--tiers ha0=T1` (D-REFAV1-HA0-ARM's one-line integration).
- **R14 — the anti-stranding rule, operational**: a completed run's artifacts lived only in `C:\Users\Admin\tanitad-wt\_gs89_pkg\` until a later agent rescued 17 files. Every agent brief already says "never leave work that took real effort living ONLY in a worktree"; add the CHECK — the Master Mind verifies the manifest's 'only one place' column before closing a stream.

## Added 2026-09-03 07:45 Berlin (from D-PREREG-V7F)

- **R15 — v7f LAUNCH BLOCKER: the DINOv3 → seed-checkpoint converter.** No loader puts DINOv3 weights into `ViTEncoder`/`ViT5Encoder`; `--init-from` refuses a partial checkpoint (`train_v6_staged.py:7236-7247`). Needs the converter plus three flags: trunk LR multiplier, trunk warmup, distillation-anchor weight (the optimizer is one flat AdamW at `:6120`, `--freeze-encoder` is all-or-nothing at `:5964`). Tests: the seeded encoder reproduces DINOv3's features at step 0 to a stated tolerance, and the OFF path is byte-identical.
- **R16 — sweep every retired threshold through the instruction documents** (`.claude/skills/`, `CLAUDE.md`, the PREREG templates), not only through the register: the 8.56 floor survived eleven days in the validation skill after the code and the registry retired it (RETRACTION_LOG 2026-09-03 #5).
- **R17 — rung R0 of the v7f ladder is 0 GPU and can refute the LDAD line before an arm is spent** (`PREREG_V7F.md`); run it before any v7f compute is scheduled.

## Added 2026-09-03 08:15 Berlin

- **R8 — STRUCK 2026-09-03** (D-V7-EVAL-EXCLUSION): exclusion is ON by default and a contaminated run refuses at preflight without a stamped override.
- **R18 — gate the fp8 shipper** (`stack/scripts/dinov3_fp8_encode_ship.py`, C-FP8-SHIPPER-UNGATED): call `parity.guard_corpus_build` on the SOURCE split before encoding (preferred), or classify it in `test_build_parity_guard.NOT_AN_INGEST_DOOR` with a stated reason AND an assertion that the shipped set equals the source set. The guard stays RED until then. It built a clip of the live refav1 training cache on 2026-09-02.
- **R19 — bound `_artifact_path_names`' two-hop closure** in `test_build_parity_guard.py`: it reached ~200 unrelated locals and produced a spurious ungated-writer entry; the owner of that suite should bound it so the instrument's population is stable under unrelated edits.

## Added 2026-09-03 09:25 Berlin

- **R20 — refcv3 T1 adapter: triage the rescued draft** (`Benchmarks & Evals/Implementation/incoming/2026-09-03-refcv3-arm-UNVERIFIED/`). Re-derive the T1 definition for a supervised trajectory model FIRST (the killed agent never wrote it down), then decide: finish the draft or restart from the current `refav1_arm.py`, which has since gained `ha0`, the trivial-profile instrument and `--action-units`. It also carries C-REFCV3-ARM-SAME-DEFECT. refcv3 ends ≈ 2026-09-03 23:00Z and has no admissible T1 instrument until this lands.
- **R21 — decide when `--action-units steer` becomes the default** (D-STEER-INTERFACE-RESOLVED). Every banked refav1 lateral number was produced OFF; ON and OFF numbers are not comparable, so the flip needs a stated cut-over and a re-read of anything quoted across it.
- **R22 — run `actdiv_anchored`'s `d(a)` in BOTH unit conventions** (~15 min CPU): the only admissible test of what the unit costs the imagination, and item (iii) of D-ACTDIV-ANCHORED-REFAV1's next-probe list.

## Added 2026-09-03 10:05 Berlin (from D-V7-DINO-SEED)

- **R15 — STRUCK 2026-09-03**: the DINOv3 seed converter, `--init-encoder-from`, and the trunk-LR / warmup flags are wired and tested.
- **R23 — wire `--w-trunk-anchor`** (v7f rung R3's `anchored` arm is blocked without it): a SECOND frozen forward of the seed's own ViT-B/16 plus the live encoder's patch tokens at the loss site. O7's teacher cannot be reused (wrong network, wrong granularity, absent at `--w-o7-distill 0`). Loss-composition path; the flag currently refuses at any non-zero value rather than training inert.
- **R24 — PI decision: authorise (or refuse) the DINOv3 ViT-B/16 pull.** Only ViT-L/16 and dinov2-base are on the box; the converter never downloads. Blocks the v7f seed at the prereg's chosen geometry.
- **R25 — the `-k` selection in every v7 brief misses new test files by name.** `test_dinov3_seed.py` matches none of `v6|staged|parity|v7_labels|intrain|v7_wiring|eval_exclusion`; the agent correctly refused to rename tests to game the filter. Fix the brief template, not the tests.

## Added 2026-09-03 11:00 Berlin (from D-REFAV1-COST-SURFACE)

- **R10 — STRUCK 2026-09-03**: the cost-surface panel is measured and banked.
- **R26 — URGENT: complete the κ→steer conversion or remove the flag.** `as_command` is called at ONE site (`refa_v1.py:1832`); `_imagine_tactical_goal` (`:1680`) still rolls the goal in raw κ, so `model_action_units="steer"` converts one side and makes the turn ONE ULP worse. Convert both sides or neither. Blocks R21's cut-over decision.
- **R27 — the cost's TIME units**: `_cost_chunk` consumes the 2.0 s plan on the 0.6 s tactical clock and imagines 6.0 s, while the goal is subsampled correctly. Also: T4 (`target_speed`) is dead code — three live terms, not four.
- **R28 — THE LEVER (training, not planning): the tactical decoder.** It asks for a turn on 0 of 27 human-turn windows (incumbent; ep2 0.333) and the goal it emits IS the cv rollout on 79–92 % of windows. A planner cannot search for a turn it is never asked to make. This is repair R4 of the package and the highest -value refav1 item; it needs a pre-registration (`TanitAD_ValidateAIDesign`).
- **R29 — the cosine cost is float32-saturated along κ** (2.00 ULPs median vs 167.5/19,804 along `a`; the shipped winning cost takes six distinct values over 140 windows, some negative). Package repair R2 proposes the chord distance `‖ẑ−ĝ‖`, monotone-equivalent, taking κ from ~2 to ~10⁷ representable steps.

## Added 2026-09-03 09:05 Berlin (from D-REFCV3-EPOCH-READ)

- **R30 — PI/DOCTRINE RULING, blocks register decision 9**: does `EVAL_DOCTRINE.md` admit as T1 a model that consumes NO actions? refcv3 is one-shot with no rollout, so `roll_closed` cannot be ported. Default if unanswered: score refcv3 at its own tier, report the margin over the shared `ha0` floor per family, and refuse any head-to-head level comparison. Rename its one-shot arm `os`, never `cl`.
- **R31 — the in-training eval must dump PER-WINDOW values with episode ids.** Every number in `metrics.jsonl` is already a pooled mean, so no estimator can put an interval on it and the information is NOT recoverable later. Forward-looking fix; it makes every future in-training eval quotable with a CI.
- **R32 — reconcile the leaked-eval count**: `C-REFCV3-EVAL-PRIOR-LEAK` says 34, the metrics file measures 40, and the file starts at step 550 so neither covers the whole run.
- **R33 — pick up `taniteval/tools/refcv3_metrics_read.py`**: a naive read of `metrics.jsonl` is wrong in three separate ways (era segmentation, the `elapsed_s` resets across 10 launches, and the oracle-selected `eval_traj`). Route any future read through it.

## Added 2026-09-03 09:30 Berlin (from D-STEER-CONVERSION-COMPLETE)

- **R26 — STRUCK 2026-09-03**: the κ→steer crossing is single and complete (`_model_actions`).
- **R27 — STRUCK (repair)**: `cost_time_grid` implemented; the DEFAULT FLIP remains a decision (see R36).
- **R34 — DESIGN: the plan spans 2.0 s of a 6.0 s goal.** The planner's own canonical seed cannot reproduce its own goal, which sits directly beside the measurement that a correct turn buys one float32 ULP. Decide `plan_horizon_s` (or subsample the goal to the plan's reach) before any further planner tuning.
- **R35 — decide T4 (`target_speed`) deliberately**: unused by production, exercised by two tests, so neither deletion nor wiring is free. Recommendation: KEEP until the cost is redesigned.
- **R36 — decide when `cost_time_grid="tactical"` becomes the default.** The banked panel is provably unaffected (the winner is constant on 140/140), so the flip is safe for what is banked — but it changes what every FUTURE number means, so it needs a stated cut-over like R21's.

## Added 2026-09-03 10:15 Berlin (from D-REFAV1-TAC-DECODER-PANEL)

- **R37 — L1, PREFLIGHT for every arm in `PREREG_TACTICAL_DECODER.md`: log the decoder's loss.** Four keys (`loss_lat_label`, `loss_lon_label`, `loss_route_label`, and their weighted share) in `refa_v1_train.py`'s row. The term is 13.6–20.5 % of the loss and has NEVER appeared in a log. ⛔ Do NOT ship to Thor mid-run — stage for the next launch.
- **R38 — L0: make the planner's canonical seed reproduce its own goal.** 62 of 64 token pairs differ, `TURN` over-rotates its goal by 82.5°, and `4139203`'s time-grid repair does NOT close it (the cause is the truncation to `plan_steps`). Prior to every other planner or decoder fix.
- **R39 — THE DEPLOYMENT QUESTION: the tactical decision is oracle nav, not scene.** Under `nav_zero` the ranking collapses to 0.520 and a nav-only predictor beats the model (0.684 > 0.650). nav will not exist at deployment. This binds the v7 line as much as refav1 — every nav-conditioned tactical claim needs the nav-zero arm beside the nav-shuffle one.
- **R29 — REFRAMED (not struck)**: the chord cost alone flips 1 of 25 windows and silently reweights by 5,793×; it is necessary with L3 + L4 jointly, not sufficient alone.
- **R40 — 5 of 16 tactical classes have ZERO support** in the training labels, and 8.13 % of batch-8 steps see no labelled row at all (~2.15 rows per step). A class-balanced loss cannot conjure support that is absent; the label pipeline is the Data FlyWheel's item.

## Added 2026-09-03 10:10 Berlin (from D-TACDEC-L0-L1-E0)

- **R37 — STRUCK**: the decoder's loss is logged (four keys, `None` never `0.0`); the pre-registration's preflight is discharged and its arms are unblocked.
- **R38 — PARTLY STRUCK**: the seed reproduces its goal 64/64 at 0.000° under `goal_time_grid="plan"` (default OFF). What remains is the DESIGN half — see R34.
- **R41 — the cost repair is now the critical path, and the chord is NOT optional.** Only one configuration ranks a turn at shipped weights (L0 identity + chord + steer units) and it is 4 episodes of one token; the κ² weight that would tip the shipped cost is 1.66e-08, a deletion. Land the chord WITH L4, on the tiny ladder, per `PREREG_TACTICAL_DECODER.md` §5.
- **R42 — scope note on `D-REFAV1-BOUNDARY-NULL`**: the steer conversion wins nothing under the SHIPPED cost but is NECESSARY under the only cost that ranks a turn. Any future statement of its value must name the cost it is measured under.

## Added 2026-09-03 10:20 Berlin

- **R13 — STRUCK**: `"ha0": "T1"` is in `t1_eval.DEFAULT_TIERS`; a standalone read no longer aborts after the rollout, and the constant-velocity floor is present by default.

## Added 2026-09-03 10:40 Berlin (from D-SOTA-DECODABILITY)

- **R43 — MERGE `LAB-REP-2` into theme 1's `LAB-DRIFT-2` before either is scheduled.** The same lever (multi-horizon heads + a training-only cross-modal target) arrived from two independent primaries; running both spends the tiny-ladder budget twice on one variable.
- **R44 — adopt the three-gate ORDER for P5/L3** (capture → real-effect resolvability → propagation; `2608.29998`). ⛔ `rdw8p30k` currently FAILS gate 1 (pooled cross-clip R² −0.05…−0.09), so no propagation number on that arm may be quoted until it passes.
- **R45 — add the four controls we lack** to the criteria registry: a COUNTERFACTUAL input, a VALIDATED POSITIVE control (an engineered-leaky model that must fire), the PASSTHROUGH bound (an untrained copy), and the normalisation-state check (effective rank moved 16.5 → 67.8 with no weight change).
- **R46 — the oracle-input audit of refav1 as a PUBLISHABLE result** (LAB-REP-4): the field states it could not find a public-weights world model conditioning its predictor on language and calls demonstrating the leak in a released model the key remaining step. Verify refav1's conditioning form first (UNVERIFIED), then decide with the PI whether we publish.

## Added 2026-09-03 11:00 Berlin

- **R33 — STRUCK**: the two refcv3 readings are reconciled; the cause was the estimator and the package's definition (half-mean vs `sd_step`) is now primary on the page. The conclusion is unchanged.
- **R47 — the rule this earns**: when a number about to be published disagrees with a banked one, DIFF THE DEFINITIONS FIRST. Printing both and deferring is honest but is not the work.

## Added 2026-09-03 11:30 Berlin (from D-V7-TRUNK-ANCHOR)

- **R23 — STRUCK**: `--w-trunk-anchor` is wired, monitored and test-pinned; rung R3's four `trunk_policy` arms are unblocked by this change.
- **R48 — rung R3's `lastk` arm has no flag**: `--trunk-lr-scale` is whole-trunk. A per-block LR policy is a separate, unowned change; either implement it or drop `lastk` from R3 and say so.
- **R49 — `dry_run()` does not seed the global RNG** (two identical default dry-runs read 4.0784 vs 4.1158). Pre-existing; a seeding fix would move every banked dry-run number, so it needs a stated cut-over like R21's.

## Added 2026-09-03 11:50 Berlin (from D-REFCV3-ARM)

- **R20 — STRUCK**: refcv3 has a real T1 instrument, restarted from `refav1_arm.py` with the evidence for that choice; the rescued draft stays filed as UNVERIFIED history.
- **R50 — the SELECTION PROFILE gate, and it generalises.** A random-init anchor model picked ONE anchor on 42/42 windows while the trivial-profile gate read 0.0000: our degeneracy instrument asks whether a TRAJECTORY is trivial, not whether a SELECTION is. Add it as `T1_CHECKLIST.md` GATE 6's second bullet and carry it on every anchor-model read.
- **R51 — `REFCV3_ARM.md` §2 needs a human read before any refcv3 number is quoted** (GATE 2 is blocked on that, not on GPU).
- **R52 — the lead-block join reaches only EVEN raw frames** (`_odd_raw_frames` prints the count); fix before the real read or state the coverage in the result.

## Added 2026-09-03 12:40 Berlin (from D-REFCV3-NAVZERO)

- **R39 — DISCHARGED for refcv3**: the nav-zero arm ships on by default, with its per-layer removal stated and its lower-bound caveat carried in the record. refav1 still needs the same arm on its trajectory side.
- **R53 — `trivial_profile.identity_to`'s 1e-9 m threshold is unresolvable for CROSS-CALL arms.** The batching floor alone is 5.96e-07 m (different batch sizes take different GEMM paths), so two arms that are provably identical read as different, and a reader can take that as evidence. Fix the threshold in `refav1_arm.py` (batch-aware, or compare at matched batch), or make the instrument refuse to compare across calls.
- **R54 — PI decision D3b: which margin leads the H-vs-F table**, the fed-nav one or the deployment (nav-zero) one. Quoting only the fed-nav margin overstates the system, since nav is an oracle input that will not exist at deployment. Default: lead with the deployment margin and report the fed-nav one beside it.

## Added 2026-09-03 13:10 Berlin (from D-COST-CHORD)

- **R41 — STRUCK, with a FAIL**: the cost repair does not make the planner turn where the human turns (6/27 and 0/27 against a bar of 14). The programme moves to the goal-SPACE branch and the tactical decoder remains the lever.
- **R55 — `chord` must NOT become the default.** Default-safe as a flag, but a 5,793× reweighting wearing a metric's clothes; any adoption needs the weight re-derived WITH it.
- **R56 — unreconciled: the κ² charge reads 3.20e-04 (cost-repair) vs 2.24e-04 (cost-surface), 1.43×.** Both MEASURED in different packages, neither withdrawn. Reconcile before either number is quoted again.
- **R57 — an unowned file appeared inside another stream's package dir**: `…/2026-09-03-cost-repair/PROPOSED_REGISTER_ROWS.md` (12,695 B) at 11:28, after the Master Mind's 11:15 listing; a file of that name belongs to `2026-09-03-anchored-actdiv-refav1`. Unread, unstaged, unmodified by the stream that found it. Identify its owner before staging anything from that directory.

## Added 2026-09-03 13:45 Berlin (from H-GS9-1 / D-ACTDIV-UNITS-INVARIANCE)

- **R12, R22 — STRUCK**: the banked GS-9 raw is read (L3-TRANSITION-FAIL, on an ADMISSIBLE estimator) and the unit-invariance argument is tested (INVARIANCE-PARTIAL).
- **R58 — `v` leaks in at the MODEL BOUNDARY, not in a decode target.** `clip_features:521` lifts `v0` into the third channel while the zero-action arm zeroes only steer and accel; `v_t` alone scores `dx_fwd` at 0.9986. Every probe that feeds a model through `_lift3` needs a `v`-zeroed arm, and `D-V-EXCLUDED` must be restated to cover INPUTS, not only targets.
- **R59 — `run_panel()` silently skips unknown cells**, a false-negative generator: it would have dropped the `v_t` control that found R58 without a word. Two-line refusal needed (not the finding stream's file).
- **R60 — `transition_probe.py` is wired into nothing**; `mm_e19_read.py` still calls `actdiv_local.py`, which carries the H-LEAK-1 speed-scale defect. It is the ONLY instrument that answers L3's question on an admissible estimator.
- **R61 — raw pixels carry 5–200× more transition-specific structure than the learned Δz** (RFF floor, point estimates, no CI). If that survives a CI it is a representation finding, not an instrument one.
- **R62 — `MODEL_REGISTRY.md` has no row for `k8clip05p30k`** (0 hits, two probes).

---

## ⛔⛔ BLOCKING INTEGRATION ITEM — refcv5's model seams are NOT IN `HEAD` (raised 2026-09-06, TanitAD_TrainingFlyWheel)

**0 GPU. Blocks every refcv5 arm and every TanitLang attachment. Needs the Architecture &
Inference stream (whose uncommitted work it is) or a Master Mind ruling.**

MEASURED on `HEAD`'s blob of `stack/tanitad/refs/refc.py` (`0e6103e5`, 155,407 bytes read),
with same-breath controls that must read non-zero
(`…/2026-09-05-vla-design-dialogue/AUDIT_REFCV5_SURFACE.md`):

| symbol | HEAD | worktree |
|---|---:|---:|
| `agent_tok` · `control_head` · `cross_agent` · `time_mlp` · `sampler` | **0 · 0 · 0 · 0 · 0** | 19 · 12 · 11 · 6 · 52 |
| `lan_to_cond` / `maneuver_to_anchor` — **CONTROLS** | 7 / 4 | 7 / 5 |

⛔ `HEAD`'s trainer already calls `_pin_refcv5_seams` (3×) and `AgentSeamConfig` (2×), while
`assert_seams_are_built` appears **0×** in `HEAD`. That is precisely the false-provenance
state the guard exists to prevent: **a run that stamps `sampler="ddim"` on a model that has
no denoiser.** ⇒ **a fresh clone of `HEAD` cannot build a refcv5 model at all**, and any
refcv5 result produced from a clone would be silently about a different model.

⚠️ **Why I did not simply commit it.** It is another stream's in-progress model code.
Sweeping it into my commit is exactly the failure `CLAUDE.md`'s git-hygiene rule exists to
stop — and the same rule says a recorded sweep is recoverable while a silent one is not.
**Raised here rather than left in a research doc** (an orthogonality instrument once sat
unmerged for 10 days because the request lived in a README nobody re-read).

**What unblocks it:** the Architecture & Inference stream commits the refcv5 seams with an
explicit pathspec, and `assert_seams_are_built` lands in the same commit as the seams it
guards — a half-fix that ships the guard without the seams (or the seams without the guard)
is worse than either.

**Related, non-blocking:** `stack/tanitad/refs/refc_selector.py` (WP-7, **4,530,444** params)
*is* committed and is wired to nothing but an import probe.


---

## ✅ D-ROLL-1h — CLOSED 2026-09-06: the `--tac-goal-tok-head` TRAINER FLAG IS WIRED

**Blocks:** launching the **D-TACGOAL-1 arm at all.** The 22-token tactical-goal SET head exists,
is tested (17 tests) and is correct — but as of 2026-09-06 it is **OPT-IN and has no CLI flag**, so
**no training command line can switch it on.**

**Why it is opt-in** (D-ROLL-1, `…/Research/2026-09-06-rollability/RESULT.md`): built on the
vocabulary alone it added **11,286 params** to every rebuild of a checkpoint trained before it
existed — params absent from every recorded `param_breakdown` — so
`refcv3_arm.cross_check_config` REFUSED, and refcv4b@40284, three local refcv3 checkpoints **and
the LIVE refcv5 A40 run** became unrollable (refcv5 also **unresumable**, since a supervisor
relaunch rebuilds through the same file). ⛔ The guard was right; the construction was wrong.
`RefCV3Config.tac_goal_tok_head: bool = False` now gates it, and all six local checkpoints load
**0 unexpected keys** again.

**The exact diff** (two hunks, both in `stack/scripts/refc_v3_train.py`):

```diff
@@ in build_parser() @@
+    ap.add_argument("--tac-goal-tok-head", action="store_true",
+                    help="D-TACGOAL-1: build the 22-token tactical-goal SET head "
+                         "(+11,286 params at d_tac 512). Requires --v7-labels; "
+                         "kin3 has no tactical goal vocabulary and refuses it. "
+                         "OFF by default so a banked checkpoint rebuilds with the "
+                         "parameter set it was trained with (D-ROLL-1).")

@@ in _pin_trainer_cfg(cfg, args) @@
+    # D-ROLL-1: OPT-IN, recorded in argv. Building this head on the vocabulary
+    # alone made refcv4b, three refcv3 checkpoints and the LIVE refcv5 run
+    # unrollable, because its 11,286 params are absent from every recorded
+    # param_breakdown and cross_check_config correctly refuses.
+    if getattr(args, "tac_goal_tok_head", False):
+        cfg.tac_goal_tok_head = True
```

⚠️ **Do NOT instead revert the gate.** `stack/tests/test_refc_v3_rollability.py` (11 tests) pins
the contract, including the real recorded `argv` + `param_breakdown` of refcv4b, refcv3-b1-v72 and
the live refcv5 — the rollability bar, in CI, without the 1.3 GB weights. A two-sided mutation
proof (`…/raw/run_mutation_proof.py`) shows the defect REFUSES the real refcv4b checkpoint and the
suite goes RED, and the fix makes both PASS.

⭐ **The current state is the SAFE failure direction:** with no CLI flag, no new arm can
accidentally train the head and no banked arm can lose rollability. This is a capability gap, not a
hazard — but D-TACGOAL-1's arm cannot run until it is closed.

### ✅ CLOSED — 2026-09-06, `<!-- D-ROLL-1h-CLOSED-2026-09-06 -->`

The filed diff was applied **as written** (flag in `build_parser`, two-line pin in `_pin_trainer_cfg`), plus the run-record stamp and a `kin3` refusal so the flag cannot be silently inert. **`+97 / −0`, purely additive.**

* ⛔⛔ **The default path is BIT-IDENTICAL** to the pre-patch trainer (git blob `c4e52c3b…`, the file the live refcv5 supervisor relaunches through): `torch.equal` over **429,563,860 floats** across four recorded `argv` — refcv4b, refcv3-b1-v72, the **live refcv5**, and the kin3 branch — with `dataclasses.asdict(cfg)` identical on all four. **0 differences.** The MUST-DIFFER control (same script, flag ON) fails on the key set with `delta_total = +11,286`.
* **All six banked arms still load**: the four local checkpoints through `refcv3_arm.load_model` re-run unmodified, `n_refused = 0`, and the produced record is **byte-for-byte identical** to the pre-flag run; the other two (refcv3-b1-v72, live refcv5) reproduce their recorded totals by rebuild.
* **Two-sided**: the ledger line APPEARS and DISAPPEARS (`tac_goal_tok_head: 11286` at `d_tac 512`, built exactly once); deleting the pin at the SOURCE makes the flag **DEAD**, and the restore is md5-verified.
* **Reachability on both launch paths** (AST over `main`'s dispatch and both entry points' config production, plus an empirical counter: 0 by default, exactly 1 with the flag).
* Suite, **controlled**: the refc-adjacent modules read **195 passed** without the change and **205 passed** with it — the delta is exactly the 10 new tests, **no pre-existing test changed state**.

⚠ **`D-TACGOAL-TRAINER-SEAM-OPEN` stays OPEN.** The head now BUILDS, is STAMPED and is ROLLABLE from a command line; **supervising** it still needs the two additive edits in `V3Dataset` / `compute_losses_v3` and the `MANEUVER_WEIGHT` budget decision, which is an owner/PI call.

⭐ `…/Research/2026-09-06-tacgoal-flag/RESULT.md` · `stack/tests/test_tac_goal_trainer_flag.py`

<!-- EVAL-REFCV6-STANDARD-TESTS-2026-09-23 -->

### OPEN — 2026-09-23 · refcv6 standard test battery + NavSim suite · owner: EvalFlyWheel

The PI asked (2026-09-23): *"contact the eval fly wheel agent to prepare the required standard tests
inclduing the navsimn suite for refcv6"*. Sent to the EvalFlyWheel session by cross-session message
(**queued — that session was offline**); the durable brief is
`FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/BRIEF.md`: the four-family
held-out battery with controls, the NavSim bridge extended to refcv6 (incl. navtest PDMS if
provisionable), the other charter benchmarks, and the checkpoint steps to snapshot. **Deadline:**
ready before refcv6-r101-s0 finishes (≈ 2026-09-27). **Master Mind side:** snapshot the requested
checkpoints (md5) to the dev box; the run's training reports are the "refcv6 Training Watch"
artifact, refreshed at 00/04/08/12/16/20:13 Berlin.

<!-- COORD-LANDING-FOLLOWUPS-2026-09-26 -->

### OPEN — 2026-09-26 · two follow-ups from the coordinated landing · owner: EvalFlyWheel (leaderboard + internal_t1 streams)

The PI asked (2026-09-26) for all work to be committed and pushed, coordinated. The Master Mind
landed D:'s staged backlog in ten guarded commits, `da27b38` .. `4ac00b4`. Two items could not be
closed by landing alone:

1. **`test_leaderboard.py::test_fixture_is_valid_under_the_w1_contract_and_current` passes only on a
   CRLF checkout.** `taniteval/taniteval/leaderboard/sources.py:50` hashes a source's raw working-tree
   bytes (`short_sha`). The committed fixture holds the sha12 of the CRLF bytes of
   `…/2026-09-19-navsim-warmup-reference-epdms/raw/controls.json` (`5d297aeb2beb`), while its repo blob
   is LF (`dc6d84bb4cae`). MEASURED on a clean git-archive tree: the test fails on any LF checkout,
   pods included. Four sibling tests likewise assume `LEADERBOARD.md` is CRLF: they pass once it is,
   75/76 green. **Fix:** hash EOL-normalised bytes, or the blob, and regenerate the fixture. Do not
   regenerate it without that fix, or it will fail again on the next differently-configured checkout.
2. **One bench result was withheld:**
   `taniteval/results/bench/internal_t1/v2ep-eval124clean-416x1024cyl-halfB/20260920T073353Z-internal_t1-none-04264a/raw/refcv3_arm.json`
   carries raw PhysicalAI clip ids (`"clip_id": "<uuid>"`). The clip-id rule keeps new ones out of
   the record. It is still on D:, staged. **Fix:** re-emit it with `sha12(clip_id)`, and check
   whether its run summary cites the file's hash.

<!-- YAWMASK-FOLLOWUPS-2026-09-26 -->

### OPEN — 2026-09-26 · the yaw-rate mask landed; five follow-ups · owner: see each item

The shared paired yaw-rate cell (`taniteval/tools/refav1_arm.py::_components`) now masks tangent-less
step pairs exactly as heading does (`RETR-2026-09-26-YAWMASK`). Four landed claims moved, and the
register rows carry them. The fix itself cannot close the following:

1. **Propagation sweep (0 GPU; any doc agent).** About 80 lines still quote the defective numbers:
   6/42 = 14.3 % (now 4/42 = 9.5 %), M26's −80.4 % yaw (now −4.3 %), and refcv3's LATERAL "LOST"
   (now WON 3/3). The list is `…/2026-09-26-yaw-rate-mask/raw/propagation_sites.txt`. Annotate each
   site with the corrected value and the RETR id; never silently overwrite a number. `CLAUDE.md:183`
   waits on the PI (PI_DECISION_QUEUE).
2. **`H-KINGATE-1`'s `kin_gate_eval.py` has its OWN unmasked `_yaw` / `_head` / `_kappa`** (same
   defect class, a separate instrument). Mask them the way `four_families.lateral` does, add a
   known-value stopped-window control, and re-read H-KINGATE-1. Owner: EvalFlyWheel.
3. **The battery's clone `C:/Users/Admin/ev6` must be synced** to a tip that has the fix before its
   next tag. Until then the battery runs the historical cell, although its own A3 cell already
   masks. `recompute_a3.py` refuses any pre-A3 panel once the shared cell moves (by design). Owner:
   EvalFlyWheel battery.
4. **`PREREG_REFCV6.md` L2 amendment A-YAW** is drafted in the package's RESULT §4.6 and NOT applied.
   No code builds that panel today. If a builder is ever written, it must use the masked cell and
   °/s-vs-rad/s units. Owner: Arch.
5. **The Thor half of `D-REFAV1-LON-T1`** was not re-derived, because its dumps live on Thor only.
   Its dev-box twin kept its sign and separation (−0.1009 → −0.1161). Re-read it when Thor is free
   (after refcv6-r101-s0 finishes).

- 2026-09-26 update to `YAWMASK-FOLLOWUPS-2026-09-26`, item 1: `CLAUDE.md` is DONE on the PI's instruction (`226d39e`); it now
  reads 4/42 = 9.5 %. The rest of the propagation sweep (`raw/propagation_sites.txt`) stays open.

<!-- NAVSIM-NAVTEST1S-FOLLOWUPS-2026-09-26 -->

### OPEN — 2026-09-26 · navtest single-stage landed (W8); four follow-ups · owner: EvalFlyWheel

1. **Leaderboard comparability (W4).**
   - The navtest single-stage column mixes 8 pre-#151, 6 unverified and 4 post-#151 published rows.
   - Ours is post-#151 AND non-reactive, so it is comparable only to post-#151 rows that were also
     non-reactive. None of the 4 banked papers states which.
   - The renderer does not yet read our `fix151` stamp. Make it read the stamp and mark every
     published row's comparability; never compare across the #151 line silently.
2. **Wrong refusal text (W2).**
   - The NavSim STRATEGIC refusal says "PhysicalAI-AV carries NO map…", but NavSim HAS maps and
     routes.
   - Rewrite the refusal per benchmark.
   - It predates W8.
3. **Model arms on navtest single-stage.** They are refused until a
   `DEFAULT_BANKS['navtest_single_stage']` entry and a stage-1 join of W3's frame bank exist. Build
   both, then run refcv6's checkpoints on the split.
4. **Live outputs still to land.** Land the full-split cache build and the full floors
   (`raw/cache_full/**`, the driver logs, `verdict_full.json`, `cross_protocol_full.json`) once
   `navtest_full_after_final.py` writes its END marker.

- 2026-09-26 A16 audit Q3 follow-ups (owner: Arch). Evidence: `…/2026-09-26-refcv6-frozen-trunk-audit/raw/q3_hooks_forward.json`, which is MEASURED.
  - (1) At the launched config, the stem input carries NO bottom-43-row equalisation: the diff to the un-equalised reference is 0.0.
    Decide whether that was intended, from the SPEC and the train-time transform, before anyone "fixes" it.
  - (2) 6 attention key/memory sources read UNRESOLVED in the low-memory mode. Name them with one `Q3_KEEP_ALIVE=1` run when the dev box
    has ≥ 16 GB free (post-FINAL).

- 2026-09-26 **D-REFCV6-EQUALIZE-DROPPED** (owner: Arch). Evidence: `…/2026-09-26-refcv6-frozen-trunk-audit/code/eq_probe.py`.
  - The problem: `--equalize-bottom-rows` is dropped from the trunk by the `--image-hw` `dataclasses.replace` rebuild
    (`refc_v3_train.py:381` vs `:459`; the field is undeclared on `CNNEncoderConfig`).
  - Fix it for the next run: declare the field; add a post-build assert that the trunk equalises exactly what argv says;
    add a test via `_pin_trainer_cfg` + `--image-hw`, with a mutation arm.
  - Audit every OTHER ad-hoc `cfg.core.encoder.<x> = ...` in `_pin_trainer_cfg` for the same loss. Any undeclared attribute
    set before line 459 dies the same way.

<!-- DRIVORT-RENAME-2026-09-26 -->
### OPEN — 2026-09-26 — rename DrivoR-T's code from refcv7_* to drivort_* · owner: Master Mind, after the FIX landings
- **Files:**
  - `stack/tanitad/refs/refcv7_heads.py`, `refcv7_oracle.py`, `refcv7_toad.py`;
  - `stack/scripts/refcv7_derive_nav_tau.py`;
  - `refc_v3.py`'s refcv7 fields;
  - `refc_v3_train.py`'s `--refcv7`, `--w-r7-wta`, `--w-r7-scorer` and `_pin_refcv7`;
  - the tests `test_refcv7_*.py` and `test_trainer_imports_without_refcv7.py`.
- **Why:** the PI ruled on 2026-09-26 that refcv7 = path (b) and DrivoR-T takes another name. Until the rename, a refcv7 launch passes none of those flags, and G-DVB lists them as DrivoR-T levers that must be OFF.
- **Gate:**
  - a clean-tree suite shows 0 regressions against the tip;
  - afterwards, `git grep -i refcv7` hits only path-(b) code and history documents.

### OPEN — 2026-09-26 — Training Watch builder: refuse a NavSim summary whose count guard is not PASS
- The builder rendered the partial navtest@30k (1,464/12,146) as "64.14 FAIL (SUBSET)".
- The NavSim parsers now refuse to write such a summary, but the builder should refuse too, independently, and render "UNAVAILABLE (count guard FAIL n/N)".
- Mutation arm: a summary whose counts FAIL must render UNAVAILABLE.

### OPEN — 2026-09-26 — other sessions' register lines that live only on D:
- D:'s `GOALS_AND_CLAIMS.md` carries 25 non-blank lines absent from the tip, and `RETRACTION_LOG.md` carries 17. Example: the EvalFlyWheel's "R25 (2026-09-26) … SPEC_NAVTEST Amendment 3" retraction.
- The D: copies are otherwise ~3,700 lines behind the tip.
- Owners hand these lines over as append blocks via LANDING_READY; the Master Mind lands them.

### DONE — 2026-09-26 ~22:00 — the Training Watch builder refuses a NavSim summary whose count guard is not PASS
- **Counts:** `EXPECTED_N` holds the published split sizes as literals: navtest 12,146 tokens; navhard 450 + 5,462; warmup 16 + 204. They are authored from the NAVSIM protocol, independently of the summaries read. `_count_guard` renders a mismatch as "UNAVAILABLE — count guard FAIL (n of N)".
- **Designed subsets:** a designed navtest subset (`tokens_subset` names its token file) still shows, with its own n.
- **Tests:**
  - 3 new tests;
  - the old test `test_a_SUBSET_navtest_shows_its_own_n_and_keeps_the_qualifier`, which pinned the partial run as correct, is REPLACED;
  - clean tree: 17 passed;
  - mutation proof: 11 of 11 arms RED, including one per new guard;
  - the proof is in `…/2026-09-23-refcv6-fixes/raw/mutation_proof_training_watch_refcv6_navsim.json`.

### BUILT, LANDED UNWIRED — 2026-10-16 — the launch gate idles the A40 for ~40 min per arm; G-SUITE alone is 22 min of it
- **2026-10-16 status:**
  - The pre-run evidence lever is built: `launch_gate.py suite-prerun` / `run --suite-evidence`, 63/63 regression arms, 26/26 mutations caught.
  - It is landed UNWIRED: no chain passes `--suite-evidence`.
  - The real-profile check agrees: accepted 2.9 s PASS vs live 201.3 s PASS (`raw/2026-10-16-gate-suite-prerun/MEASURE_RERUN.md`).
- **The larger lever is ONE script** (MEASURED from pod suite logs): `test_p2r_chain.py` takes **1,290 s on the pod** vs 5–14 s on the dev box.
  - It is in the P2/P2-s1/P6/field suites; R0c's and S1c's pod G-SUITE take only 238 s and 479 s.
  - **Cause (MEASURED 2026-10-16 on the pod):** its `token_ok` scenarios minted and verified tokens on the chain's REAL argv.
    - On the pod, `/workspace/data/...` exists, so each mint and each `launch_gate.py verify` fingerprinted the real 413,184-file image listing and the train_grow bank, and sha256'd the 1.29 GB init checkpoint over NFS.
    - On the dev box those paths are absent, which is why the same test takes seconds there.
  - **Fixed in the test only** (the gate is untouched): the scenarios run on an absent data root on every host. Token logic unchanged, 139/139.
  - Pod A/B in P6's tree copy: as shipped **1,226 s** vs fixed **6 s**, both `ZZTEST_PASS 139` (`raw/2026-10-16-gate-suite-prerun/logs/pod_p2r_ab/`).
  - It reaches the next ship of any P2/P6/field profile. P6 itself keeps c1dc5d3, because a re-ship would rebuild its stages.
- **MEASURED** on P2-s1's successful launch (pod `/workspace/data/refe_p2rep/chain.log`, `lg/` evidence mtimes, 2026-10-15):

  | step | time (Z) |
  |---|---|
  | `WAIT_OK` (GPU free) | 15:33:19 |
  | G-HYG / G-DVB evidence | 15:38:41 |
  | G-LIVE | 15:38:48 |
  | G-EVAL | 15:39:01 |
  | G-CKPT | 15:41:13 |
  | **G-SUITE** | **16:03:26 (22 min)** |
  | PASS token | 16:04:38 |
  | `LAUNCH_GATE_OK` | 16:06:12 |
  | `TRAIN_START` | 16:13:26 |

  ⇒ **40 min of idle A40 per arm.** Two earlier P2-s1 attempts spent 64 min and 17 min before refusing.
- **Lever:** G-SUITE is CPU-only and depends only on the commit and tree. Run it during the chain's WAIT stage (nice 19, beside the previous arm), then let the post-wait gate ACCEPT that evidence only if its commit, tree sha, profile and suite selection are identical, it is under 12 h old, and its tree check is re-verified. The gate still mints the token itself; nothing is hand-written.
  - Deliberate-regression arms: a different tree, a different profile, stale evidence, and evidence with one failing test must each FAIL the gate.
  - Expected saving: ~22 of the 40 min per arm (ESTIMATED from the table).
  - Also measure the ~5 min before G-HYG and the 7 min `train.py` preflight.
- **Sequencing:** start after the S1c build lands. The S1c agent edits the gate profiles, so running both at once would race on `launch_gate.py`/`gate_checks.py`. Chains already shipped (R0c, P6 @ c1dc5d3) keep their gate unchanged.

## Added 2026-10-17 (Master Mind)

| # | item | unblocked by / GPU |
|---|---|---|
| S4R-1 | **DONE 2026-10-17** (`raw/2026-10-17-s4r-integration/`, TEST_TABLE.md: 22 full-form runs, 0 regressions): rebased onto the tip; root cause of the G-SUITE-only rc 1 = the test ran a stub suite in-process under an ambient nesting depth, fixed in the test (hermetic), not the guard; `sft4r_chain.sh` takes the token-file path from a REQUIRED env var (fails closed); Amendment 3 appended to `eval/PREREG_SFT4R.md`. Original row: **Integrate SFT-4R readiness** (`raw/2026-10-16-sft4r-readiness/`, banked `76eedfb` as a RECORD, code NOT applied): rebase its patch set onto the tip (`launch_gate.py`/`gate_checks.py` changed in `daa4568`); fix the open defect (`test_sft4r_gate.py` exits rc 1 INSIDE the refe_sft4r G-SUITE, passes standalone); make `sft4r_chain.sh` take the token-file path from a REQUIRED env var (3 files were withheld from the public repo for naming it, `NOT_BANKED.sha256`); land `AMENDMENT_3_DRAFT.md` into `eval/PREREG_SFT4R.md`; run every test naming an edited file in FULL form. | nothing / 0 GPU (the launch needs the A40 after the safety-objective arm) |
| S4R-2 | **The nesting guard protects every profile, but its test runs only in `SFT4R_SUITE`.** Decide whether `test_suite_nesting.py` joins the common SUITE (+10 s per G-SUITE and a new suite binding for every profile). | S4R-1 / 0 GPU |
| P2F-1 | **DONE 2026-10-18** (`raw/2026-10-16-p2-full-table/RESULT.md`, `D-REFE-P2-FULLTABLE`; the SELRULE-DAC confirm read ADOPT, +0.02 PDMS, `D-REFE-SELRULE-DAC-CONFIRM`). **P2 full 64-proposal table is GPU-starved on the dev box**: 4 of 14 chunks done (fidelity gates PASS), waiting > 19 h for the dev-box GPU lock (held in turn by other sessions' jobs). The SELRULE-DAC confirm waits on it (watcher re-armed 48 h). | a free dev-box GPU, or port the dump to the pod / 1 small GPU |
| SEL-RANK | **STEP 1 DONE 2026-10-18, 0 GPU** (`raw/2026-10-18-sel-rank/RESULT.md`, `D-REFE-SELRANK-1`): the 5.88 pool is EP 77 % in the swap; a perfect TTC head recovers +1.83 [1.57, 2.11] and a perfect EP head +1.57 [1.35, 1.80] (not separated on the pool; EP's bound is largely a violation oracle, +0.80 without it); TTC is the weaker head inside the top-8 (concordance 0.722 vs EP 0.887). CPU logit recalibration REFUTED on the dev table. **NEXT: SR-TTC, prereg REGISTERED 2026-10-18** (`PREREG_SELRANK_ARM.md`, sha256 `cd73c4d4…`; registration change R1 moves hyper-parameter choice to grouped-by-log CV on the fit set, so the dev gate is read once: head-only TTC retrain on the scorer's top-8 contenders, trunk and fan frozen; reach ESTIMATED 0.18 / 0.46 / 0.92 PDMS for capture 10 / 25 / 50 %; MDE about 0.17). First gate PARTIAL 2026-10-18 (`raw/2026-10-18-p2-heldout-table/`): P2's own held-out table covers 603 of 3,137 sets (11 / 24 logs), all gates pass; the rest needs pod-only camera frames (13 logs) and missing nuPlan DBs (11 logs). **Amendment 1: the fit set is 174,912 sets / 99,495 tokens, so S0 alone is ESTIMATED ~40 GPU-h, not 1.1; the PI chooses full set vs a pre-registered log subsample (Amendment 2) before any fit data is read.** The training run needs the prereg hashed and a launch-gate PASS. **Amendment 2 DRAFT 2026-10-18** (`PREREG_SELRANK_ARM.amendment2.draft.md`, sha256 `85a5ec11…`, `raw/a2_subsample/`): the log-subsample option for the PI. Recommended (b): a nested sha256-ranked log subsample starting at 10 % (117 of 1,167 fit logs, 8,922 tokens, MEASURED from metadata; S0 ESTIMATED 3.57 GPU-h, up to 6.3 if each set needs its own forward), growing 25 / 50 / 100 % only while the fit-set CV learning curve still rises, so the dev gate is read once and the worst case is the full set's 39.8 GPU-h. The dev box holds both the nuPlan DB and the camera zips for only 22 of those 117 logs (file existence). Not registrable until Amendment 1 item 3 (dev-gate coverage) is decided. Correction recorded in the draft: the ~0.17 MDE is the navtest MDE, not the dev gate's. PI: (a) full set vs (b) 10 % start + growth cap, and where S0 runs. **PI 2026-10-18: option (b), Amendment 2 REGISTERED** (`PREREG_SELRANK_ARM.amendment2.md`, sha256 `b9e81a38…`): f₀ = 10 %, f_max = 100 %, and A2.10 fixes the dev gate as P2's full 24-log held-out table (admissible only if the completion run ends complete with all gates PASS). NEXT: build and run S0 on the 10 % subsample (dev-box GPU, logs staged from the pod in batches). **First gate COMPLETE 2026-10-18 10:25** (`raw/2026-10-18-p2-heldout-table/RESULT_COMPLETION.md`): P2's own held-out table now covers all 3,137 sets / 1,705 tokens / 24 logs, every gate PASS, the 333 run-1 tokens recomputed bit-identical; A2.10's four admissibility conditions are met and the table (`bulk/table_p2_heldout.npz`, sha256 `8eacc19b…`) is in the repo. The dev gate is admissible and NOT read. **S0 RUNNING 2026-10-18 11:49 (PI approved the launch; it reads pod inputs beside P6 under the overlap rule: nice 19 / ionice idle, one ~120 MB batch, skip on a `P6_GUARD` line)**: 117 logs / 8,922 tokens / 15,808 sets, preflight PASS; smoke DG-DUMP 4/4 at max abs 0.0 (`raw/s0/smoke/`); code + `S0_HANDOFF.md` in `raw/2026-10-18-sel-rank/`. ETA ESTIMATED 4-5 h + guard waits. ⚠ Disclosed design property (S0 agent, MEASURED on 37/37 smoke sets): the fit sets are older checkpoints' fans (1,281 step/rank/ckpt combos), not P2's own, and P2 trains with `--scorer-sees-trunk`, so fit-vs-gate is a proposal-distribution shift inherent to the registered design; read R1's out-of-fold concordance against the dev gate's with that in mind (a gap is a transfer signal, not a head verdict). | dev-box GPU (dump: MM; arm: PI) |
| EP-1 | **DONE 2026-10-18, 0 GPU** (`raw/2026-10-18-ep-selection/RESULT.md`, `D-REFE-EP-1`/`-2`): P2's EP loss is a safe-but-slow pick caused by a DOUBLE CHARGE (the EP head is an unconditional expectation and `planner.py:558-561` multiplies pNC*pDAC on top); 4 pre-registered zero-training rules ELIMINATED AT FIT (best C4 +0.61 [+0.19, +1.20] x100 fit, TTC -0.70 [-1.34, -0.08] adverse); navtest look UNSPENT, held for the SR-TTC-retrained scorer (MM). **NEXT: EP-R1** = a training-side EP row (EP conditional on NC=DAC=1 + pairwise EP ranking) on S0's bank, CPU, beside SR-TTC R1; prereg in build; runs after S0_DONE. | 0 GPU (CPU head fit on S0's bank) |
| EP-R1 | **BUILT + PRE-REGISTERED 2026-10-18, 0 GPU** (`raw/2026-10-18-ep-r1/`, `H-EPR1`): conditional-EP row + safe-pair ranking term, head-only on S0's bank. **Next:** after `S0_DONE`, `epr1_gate.py mint` (real-manifest check), fit at f = 10 % beside SR-TTC R1, then the dev gate. **To build first (CPU):** `epr1_dev.py`, `epr1_confirm.py`, real-bank C2/C4 reads, `families6` on the new picks. ⛔ The ONE shared navtest look opens only after BOTH SR-TTC's and EP-R1's dev gates are read (`MM_DECISIONS_EPR1.md` D1) |
| EP-R1UB | **NEXT, CPU, ~10-20 min fit** (MM R7, `D-REFE-HEADFIT-F10`): E1 + D-LEAK fired at f = 10 %. Write the NEW registration PREREG_EPR1UB (hashed before any fit): lr {1e-3, 3e-3, 1e-2} x unsafe-anchor u {0.03, 0.1, 0.3}, anchor 0.1, same folds/seeds/logs. Selection: max OOF pick-level dPDMS s.t. the OOF unsafe-rate upper CI <= 0 on both seeds. The new flags need G-DVB entries and a gate mint. If a config is feasible, read EP-R1's dev gate ONCE; if none is, write the formal abandonment and pre-register the NC/DAC-row resolution read **DONE 2026-10-19: outcome N, NONE-FEASIBLE** (`D-REFE-EPR1UB-1`); EP-R1 closed at the fit; abandonment adopted. |
| NAVTEST-HEADBANK | **SCHEDULED (MM D4, 2026-10-19), dev-box GPU, 5.43 GPU-h**: the navtest P2 `score_head` input-feature bank for the ONE shared REFE head look. Plans and commands come from `raw/2026-10-18-ep-r1/code/make_navtest_bank_plan.py` (14 chunks, 12,146 tokens). Order under the lock: after S0, the items 3+4 GPU tests and L1 Step A. **FIRST** the 4-token DG-DUMP pre-check on a frame-control log: if C4 is RED, the bank does not run **CANCELLED 2026-10-19:** no head-row look opens (SR-TTC VOID, EP-R1 abandoned; R9). Rebuild only if HEADROW-PROBE yields a capacity-limited row with a fresh registration. |
| SRTTC-DEV-APPLIER | **DONE 2026-10-19** (`srttc_dev.py`, `navtest_c2.py`, `confirm_families.py`, `confirm_read.py`; `D-REFE-HEADFIT-F10`). **Next: SRTTC-C2-STAGE** — a NEW gate-stage file for SR-TTC's C2 (MM R5; G-DVB entry and a RED arm), then SR-TTC's C2 fit, then its dev read ONCE (R8). CPU **UPDATE 2026-10-19: C2 gate stage BUILT; SR-TTC dev read DONE = VOID (C2 RED), consumed** (`D-SRTTC-DEV-F10`) |
| HEADROW-PROBE | **NEXT, CPU, minutes** (the D-FLAT next arm of SR-TTC and, if EP-R1ub has no feasible config, the NC/DAC-resolution read of EP-R1): ONE pre-registered feature-vs-capacity probe over the P2 `score_head` rows TTC, EP, NC, DAC on S0's cached features, with the same folds and seeds. Each row: the linear row vs a 2-layer probe (width and epochs FIXED in the prereg), OOF concordance and pick-level OOF dPDMS / unsafe rate. It decides whether the head-row levers are capacity-limited (then a nonlinear head) or feature-limited (then the trunk: P6, or a conditional gradient into the trunk) **RUNNING 2026-10-19** (agent; PREREG first). It subsumes EP-R1ub's `DRAFT_PREREG_NCDAC_RESOLUTION.md`; NC/DAC rows are read first. **ADDENDUM 2026-10-19 (sent before its outcome was read):** also save the per-proposal OOF predictions (row x arm x seed, with fold id, labels6 and P2's frozen row scores) and the proposal / base-pick mean speeds, for `FASTNEG-CENSUS`. Its decision rule is unchanged. **DONE 2026-10-19: MIXED** — NC / DAC / TTC FEATURE-LIMITED (the head-row line closes for them); EP INCONCLUSIVE, with an f = 25 % re-run BLOCKED and DEFERRED jointly with FASTNEG-F25-RERUN (`D-REFE-HEADROW-PROBE-1`). |
| FASTNEG-CENSUS | **REGISTERED 2026-10-19 (`TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-fastneg-census/PREREG_FASTNEG_CENSUS.md`, sha256 `b6cece236d5aa77d…`); NEXT, CPU, selection tier, no dev table, no navtest.** Is P-B (the fast band an EP-led pick selects) a COVERAGE gap, a FEATURE gap, or no row-resolution gap? Per speed-ratio stratum (cut points from speeds only, written before any label is read): label support; within-token NC/DAC AUC of P2's rows and of HEADROW-PROBE's OOF predictions; a fast-stratum learning curve 25/50/100 % with a powered null. Outcomes → FASTNEG-BANK / PRIVGEOM-PROBE / a 0-GPU pick-rule threshold audit / one re-run at f = 25 % (`H-REFE-FASTNEG-1`) **DONE 2026-10-19: INCONCLUSIVE** (DAC FEATURE, NC NO DEFICIT; null 0/5; `D-REFE-FASTNEG-CENSUS-1`). The registered f = 25 % re-run is BLOCKED (S0 exists only at f = 10 %; about 6.7 dev-box GPU-h) and DEFERRED by MM ruling. |
| FASTNEG-F25-RERUN | **DEFERRED** (`MM_RULING_CENSUS.md`): the census's committed re-run at f = 25 % needs S0 at f = 25 % (14,984 new tokens / 292 logs, about 6.7 dev-box GPU-h EST, plus pod staging under the P6 overlap rule). Un-defer when the dev-box GPU has no higher-value queued item, or when PICKRULE-AUDIT-NC / PRIVGEOM-DAC return a result a row-class flip would change **UPDATE 2026-10-19:** HEADROW-PROBE's EP f = 25 % re-run shares this one S0 f = 25 % build; both run together if it is made. |
| CENSUS-SLOT-REREAD | 0 GPU, CPU minutes, DESCRIPTIVE (no decision authority): re-read FASTNEG-CENSUS's DAC fast-band deficit stratified by proposal SLOT as well as speed. The slot prior (`D-REFE-HEADROW-PROBE-1`: DAC 0.682, EP 0.979) may confound speed; does the deficit sit in particular fast slots or across them? |
| PICKRULE-AUDIT-NC | **NEXT, 0 GPU, CPU minutes; own PREREG before any number.** NC resolves the fast band, yet RL-1's head-wise composite lost NC on navtest. Test whether the NC loss arises in how `planner.py:558-561` COMPOSES the rows: the unsafe-NC pick rate of the EP-led composite vs P2's rule on S0's bank, as a function of the NC veto threshold and calibration, with the rows unchanged **DONE 2026-10-19: ROW-MISS** (`D-REFE-PICKRULE-NC-1`; Q1 = 0.000, no veto recovers). |
| NC-FASTCAL | **NEXT, CPU minutes; own PREREG** (the committed ROW-MISS arm): SPEED-CONDITIONAL NC calibration fitted OOF on S0, because a monotone transform of the NC logit cannot reorder confident errors. Read: the EP-led composite's NC-unsafe rate and dPDMS vs P2. Low prior (MM ruling 3), run because it is cheap and committed |
| PRIVGEOM-SAFE | **NEXT, CPU (budget about 16 CPU-h, sized by measured cost); own PREREG; supersedes PRIVGEOM-DAC.** Privileged per-proposal geometry as a DIAGNOSTIC only (never an inference input): (DAC) ego-footprint clearance to the drivable boundary over the horizon; (NC) time gap, closing speed, minimum TTC margin and minimum distance to agents. Arms L / L+G / G-only per row, OOF. Read: fast-band within-token AUC and the EP-led composite's unsafe-pick rate with the L+G rows. INFO-UPSTREAM → a scorer-branch geometry read (SH5 / SH3, dev-box GPU). NO-JUMP → label learnability and the trunk |
| PRIVGEOM-DAC | **NEXT, CPU; own PREREG before any number** (D2 of the Lab report, restricted to DAC): privileged drivable-area geometry scalars (ego-box clearance to the drivable boundary at key points along the proposal) added to the head-input token as a DIAGNOSTIC only, never a deployed input. Does it close DAC's fast-band deficit (-0.07 AUC)? Yes → the information exists upstream, and the lever is a scorer-branch map/geometry read (SH5 / SH4). No → the DAC label is not learnable at this resolution, or the trunk is the limit **UPDATE 2026-10-19: superseded by PRIVGEOM-SAFE** (the pick-rule audit puts NC on the same footing: confident errors at the selected picks). |
| FASTNEG-BANK | **BLOCKED on FASTNEG-CENSUS reading COVERAGE** (lever SH1, `2026-10-19-safety-head-lit/REPORT.md` §4.3, §4.5): fast and lead-ward candidates (24 per S0 token, EST 26.4 CPU-h) as a `fast` family in `build_scorer_targets.py` `candidate_set`, then HN-screen arm C (+5 dev-box GPU-h EST, 3 seeds + an A replicate; endpoints E1-E3). Needs an amendment of `PREREG_P6HN_SCREEN` and a G-DVB entry for any new flag **UPDATE 2026-10-19: stays BLOCKED** — the census does not support COVERAGE on either row (`D-REFE-FASTNEG-CENSUS-1`). |
| PRIVGEOM-PROBE | **CONDITIONAL on FASTNEG-CENSUS reading FEATURE** (D2 of the Lab report; CPU EST 13.5 h): privileged PDM-calculator geometry scalars added to the head-input token as a DIAGNOSTIC (never a deployed input). The scalars that carry the jump name the lever: map → SH5; agents or lead → SH3 / SH7. No jump → the trunk (P6 read; SH7 / SH8 are A40 and PI-gated) **UPDATE 2026-10-19: superseded by PRIVGEOM-DAC** (the census localises the deficit to DAC). |
| SCORER-ENS-S0S1 | CANDIDATE (lever SH6, EST 3 dev-box GPU-h, selection tier): P2-s0 and P2-s1 each score a common fan on the dev table. Needs `visual_ctx` for both on the 1,705 tokens (neither exists); reuses the `make_stepA_bundle.py` forward after `L1-STEPA-RUN`. First under CAPACITY-LIMITED |
| SCORER-BRANCH-PERSTEP | **CONDITIONAL** on PRIVGEOM-PROBE naming agent or lead scalars (SH3), or on SH5 failing (SH4): per-step / proposal-anchored scorer-branch reads of cached `visual_ctx` (EST 10 and 16 dev-box GPU-h). ⚠️ The scorer's gradient is detached from the trunk (`model.py:784`), so no scorer-only lever repairs a trunk-limited reading |
| EPR1UB-GATE-CACHE | **FIX BEFORE ANY REUSE** (`D-REFE-EPR1UB-1`): `epr1ub_gate.load_verified_ub` caches the module by `epr1ub.py`'s sha alone, so a tampered parent in another code_dir is not re-verified after a clean load in the same process. Key the cache on the ub sha AND the verified parent shas (or do not cache), and make `test_a_tampered_epr1_is_refused_when_the_ub_module_is_loaded` order-independent (run it under pytest after the other tests). Audit `srttc_c2_gate` / `epr1_gate` for the same pattern |
| NT-FIX1 | **PARTLY DONE 2026-10-19:** `run_v1_fix1.py` (W3 navtest fixes A cross-drive relpath / B stale-CSV quarantine / C no summary on a count mismatch; 20 tests, RED arm per fix) is LANDED and ADOPTED in REFe's `eval/score_navtest_refe.py` (the R0e-pick / head-confirm path; reachability test 3/3, RED arm 3/3 failed). **Remaining (0 GPU):** adopt it in `score_navtest6.py:42`, `score_navtest7.py:43`, `colab/navtest/linux_run_v1.py` and add it to `pack_code.py`'s file list; leave `score_navtest_thor.py` on the admitted copy until Thor is re-validated with `run_v1_fix1.py` beside it. ⚠️ Before the next REFe navtest, SYNC the D: working tree (`W3_CODE` = `D:/Projects/TanitAD/FlyWheels/.../code`): without `run_v1_fix1.py` there the caller refuses, by design. UNVERIFIED end to end against the real devkit until that look. `incoming/2026-10-19-navtest-harness-fix/` **UPDATE 2026-10-19 (NT-FIX1-B): `score_navtest6.py`, `score_navtest7.py`, `colab/navtest/linux_run_v1.py` adopted and `pack_code.py` ships `run_v1_fix1.py`.** No file was pinned (full-sha / md5 / blob-sha1 search: historical provenance only). 17 reachability tests pass on the staged bytes; each RED arm fails against its pre-edit caller. 6 neighbour test files are per-test identical before and after (115 pass, 1 skip, 2 pre-existing failures). `run_v1_fix1.py` is now in the D: W3 code dir (placed 2026-10-19, loads against the pinned `run_v1.py`). ⚠️ Re-running `port_from_refcv6.py` would overwrite score_navtest7's edit. **Remaining:** the taniteval `navsim_v1` plugin, `after_a8_chain.py`, `after_bridge_chain.py` load `run_v1` directly. `incoming/2026-10-19-navtest-harness-fix/adopt2/` |
| SRTTC-SUCCESSOR | **TO DRAFT (after HEADROW-PROBE):** a successor SR-TTC registration with a POWERED C2 (replicate shuffles, a one-sided rule on the interval, not a per-seed point <= 0; `RETRACTION_LOG` R51). Only if the probe says the TTC row is capacity-limited |
| LIT-EP-RL | **DONE 2026-10-18, 0 GPU** (`TanitAD Research Lab/Architecture & Inference/Research/2026-10-18-ep-rl-lit/REPORT.md`; GAC `D-RL-NAVSIM-PRIOR-2`, `D-LIT-EP-MECH-1`, `H-L1-TOAD-ADMIT`; 11 papers banked): EP above the human is SELECTED by the scorer; the EP-for-safety trade follows the side updated (ranking-side trades, proposal/encoder-side does not). Ranked levers L1-L10 in the report. **NEXT: L1 Step A** (off-proposal admission test of P2-s0 for a CEM search; DRAFT prereg sha256 `a76f62d8...`, adopt or amend before any run). | <= 4 dev-box GPU-h, after S0/R1 |
| L1-STEPA-RUN | **LAUNCH-READY 2026-10-18** (`2026-10-18-ep-rl-lit/l1_stepA/`, Amendment 1 ADOPTED): the off-proposal admission test of CEM search on P2-s0's scorer. Needs the dev-box GPU lock (after S0; R1/EP-R1 fits are CPU), ONE forward `make_stepA_bundle.py` of 0.25-0.4 GPU-h (ESTIMATED, 1.46 GiB), then ~1 CPU-h. Run `python launch_stepA.py plan --work-dir <D>`. Its first run is the first data test of `make_stepA_bundle.py` / `label_perturbations.py`; G2b (bit-for-bit fan) gates it. O1 leads to Step B (SEEN-SCENE); O2 to L2 (SR-TTC extension); O3 to a re-run on all 1,705 tokens |
| L1-UNSEEN-STRATUM | **OPTION, 0 GPU to start** (`D-L1-G0-1`, Amendment 1 A1.5): build a scene-disjoint stratum so that Step A's O1 can be unconditional. Candidates: the 251 navtrain yaml tokens absent from `train_grow` (logs unknown); the 1,167 local trainval DBs not in the bank (membership in the yaml UNVERIFIED; no local frames found). Needs a token-to-DB join and a frame copy. Only if a Step-B claim would need it |

## Added 2026-10-17 later (Master Mind)

| # | item | unblocked by / GPU |
|---|---|---|
| SO-A16 | **DONE 2026-10-17**: `test_launch_gate.py` resolves a real POSIX bash by absolute path (skips the WSL launcher, loud skip if none); full form WITHOUT the shim `ZZLG_FULL_PASS 719`, with it 719. Two other bare-bash sites listed in `S4R_1_RESULT.md` section 1 (unfixed: `code/finalize_mini.py:156`, a raw probe). Original row: `test_launch_gate.py:339` calls a bare `bash`, which Windows resolves to the WSL launcher, so the full-form test fails on the dev box without the safety build's `sitecustomize_bash.py` shim. Call Git/POSIX bash by an absolute path, or skip when `shutil.which('bash')` is the WSL launcher. | the safety-objective build landing / 0 GPU |
| TEST-RED-1 | **DONE 2026-10-18 -- none is a code defect** (`raw/2026-10-17-s4r-integration/TEST_RED_1_RESULT.md`): `test_aux_g12_bf16` = torch 2.7.1 needs `CUBLAS_WORKSPACE_CONFIG` at PROCESS START under strict determinism (the pod sets it in `s1aux_pod.sh`; the test now sets it before `import torch`; PASS 7, no check weakened); `test_s1aux_chain` = a tree without the repo's FlyWheels/ has no navtest list and `Sandbox` copied it unconditionally (now a LOUD skip of the sandbox cases only; full layout PASS 404, 0 skips); `test_s1c`'s 2 skips = the same absent file on a PK-only export, legitimate by design (full layout PASS 242, 0 skips). Both edited tests are in the S1c SHIP_MANIFEST: the next S1c ship re-mints it, as every ship does. Original row: **Three refe tests are NOT clean on the tip, independent of S4R-1** (each re-run on the pristine-tip baseline by `raw/2026-10-17-s4r-integration/` post_queue, verdict PRE-EXISTING): `test_s1aux_chain` rc 1 (AssertionError), `test_aux_g12_bf16` rc 1 ("the G12 bf16 checks FAIL on the real source"), `test_s1c` rc 0 with 2 skips. CLAUDE.md: the suite must stay green. Diagnose each from its log in that record (`logs/qfinal/`); fix the code or the test, never mute the check. | nothing / 0 GPU |
| SO-G | **R0d DONE 2026-10-18** (`raw/2026-10-18-r0d-outcome/`, O4). After EACH SAFEOBJ arm: the B − own-A `rl_val_eval.py --base A` read bar G needs (PREREG_SAFEOBJ addendum 1 R1; the chain does not write it). Keep the arm's u1000 dumps until it is banked. | the arm / 0 GPU (CPU read) |
| SO-CARVE | **DONE 2026-10-18** (`raw/2026-10-18-so-carve/`, `D-REFE-SAFEOBJ-CARVE`: CARVE-TRACKS-TRAIN [F1 in-sample carve, F2 unresolved]; A1 shows the constraint's soft quantity understates the pick's TTC loss on full RL-val; next lever SO-PICKCON). **The O4 next lever, its 0-A40 first step** (`PREREG_SAFEOBJ.md` section 6 for O4, `D-REFE-SAFEOBJ-R0D`): R0d's dual held in-sample (signed window g -0.00034 TTC) while RL-val TTC breached (-0.77 [-1.49, -0.06] at u1000), so the dual must be fed from a LOG-DISJOINT carve of RL-train, never RL-val. Measure the train-vs-carve TTC/NC gap on the banked R0 states with the A2 harness before any arm; the arm after it needs the PI's A40 decision. Note for its design: lambda peaked at 0.59 of a cap of 10. | nothing / 0 GPU (CPU read) |
| SO-PICKCON | **0-GPU part DONE 2026-10-18** (`raw/2026-10-18-so-pickcon/`, `D-REFE-SO-PICKCON-1`): `--epo-con-temp` built, gated (G-DVB) and tested; A2R UNDERPOWERED, A2R2 (post-hoc) ALIGNED at T* = 0.02, thin margin. Arm drafted: `raw/2026-10-18-r0e-pick/PREREG_R0E_PICK.draft.md` (T = 0.01, PI questions in NOTES.md). **The SAFEOBJ constraint measures the wrong quantity** (`D-REFE-SAFEOBJ-CARVE` A1/A2): E_pi[TTC] at T = 0.1 understates the PICK's TTC loss on full RL-val (no ratio is quotable: the dual's interval spans 0, R49) (dual quantity +0.22 [-0.15, +0.62] vs pick +1.00 [+0.30, +1.73] at R0 u1000); T about 0.02 tracks the pick within 0.25 at u250/500/1000 (A2 INVALID as pre-registered: control 3 failed, near-tie mass; a hypothesis until re-tested). 0-GPU now: (1) add a constraint-temperature trainer flag (separate from the selection temperature) with its G-DVB launch-gate entry, a reachability proof (argv -> constraint term -> non-zero grad, plus an exactly-0.0 control) and a mutation RED; (2) re-design A2's control 3 for near-ties and re-run it on the banked states. The ARM (R0e-pick: constrained EPO with the pick-aligned constraint, fed from a log-disjoint carve) needs the PI's A40 decision; P6 has the A40 first. | nothing / 0 GPU (arm: PI) |
| R0E-PICK | **The pick-aligned SAFEOBJ arm, pre-registration, FROZEN 2026-10-18** (`raw/2026-10-18-r0e-pick/PREREG_R0E_PICK.md`, Opus draft 2026-10-18): one arm = R0d's argv + `--epo-con-temp 0.01` (departs from A2R2's T* = 0.02, disclosed: 0.02's margin 0.208 vs bar 0.25 is thin; 0.02 vs 0.01 differ by less than the same-run noise, so no committed pair). Dual fed from training batches as in R0d (a carve is a second lever: new loader, about 1 day 0 GPU, committed next arm for outcome P-O4-PICK). Bars S/G/L/E and K0-K6 unchanged. Controls: R0e-id (T = 0.1 to u1000, a full same-seed replicate of R0d), R1e-pick on O1, launch-gate token. Cost ESTIMATED from R0d's wall clock: Stage 1 <= 7.36 A40-h, max 14.72. **PI decisions TAKEN 2026-10-18 ~17:30 Berlin (`PI_DECISION_QUEUE.md`; registered in `raw/2026-10-18-r0e-pick/PREREG_R0E_PICK.md`, sha256 `1be7a030…`):** 1a Stage 1 A40 spend authorised, queued after P6, Stage 2 only on O1; 1b T = 0.01 (the disclosed departure from A2R2's 0.02 kept); 1c R0e-id runs to u1000 (3.68 A40-h, +2.41), so **B4 (stop-after-u250) is DROPPED**; 1d an O2 outcome does not earn Stage 2; 1e the DEPLOYABLE-SAFE vs-BASE line stays report-only; 1f if DDC separates adverse again, g5 + DDC as a third constrained term is the pre-authorised next lever (in the next-arm text of seven outcome rows). Cost now Stage 1 <= 7.36 A40-h, max 14.72. **Remaining launch blockers (0 GPU): B2 profile/arms and B5 gate dirs** (both wait for the P6 launch-gate landing, which owns `refe/`); B8 landed (below). **B3, B6, B7 DONE 2026-10-18** (`raw/2026-10-18-r0e-blockers/`, `D-REFE-R0E-D4`): per-arm registered constraint T in `safeobj_outcome.py` (ZZTEST_PASS 153), the D1-D3 reader `r0e_diagnostics.py` (ZZTEST_PASS 44), and D4 = `D4-NC-ALIGNED(0.01)`, thin (V 0.281 vs bar 0.25; CIs at 0.01 span 0; 0.02 also aligned, so D4 does not decide T). **PI 2026-10-18: carve size m = 13** (SO-CARVE's draw rule over the RL-train index). **B8 (carve feed) LANDED 2026-10-18** with its G-DVB patch (`raw/2026-10-18-r0e-carvefeed/RESULT.md` §5.3, §7): base-vs-mine-vs-gate lanes on complete trees, 31 rows: 8 gate-interplay (RED without the gate patch, GREEN with it), 2 environmental (`rls1_queue` dies at the pod-only `ps -o` as on the tip; `aux_navtest_forward` ZZTEST_PASS 21 from a D: path), 0 real open; 3 pin-layer regressions found and fixed (`rl_safeobj`, `safeobj_outcome`, `rl_r0c`); the 7 gate-hashing consumers re-run on the landed LF bytes, all rc 0 (`logs/lf_rerun/`). Next 0-GPU: B2/B4/B5. | 0 GPU now / A40 (PI) **UPDATE 2026-10-19: committed next arm of RL-1 (N-BOTH); BLOCKED on A40 (P6) + PI grant.** |
| RL-1 | **DONE 2026-10-19, 0 GPU** (`raw/2026-10-18-rl-headwise/`, `D-REFE-RL1-HEADWISE`, `D-REFE-RL1-MECH`): the head-wise composite V1 reads **N-BOTH** on navtest (PDMS +0.095 [-0.074, +0.257], NC -0.124 ADVERSE, EP +0.530). **Next:** R0e-pick (committed, BLOCKED on A40/P6 + PI grant). **Also:** harness fix upstream (`run_v1.py:459` relpath fallback, full-row-count OK, no summary row on a failed/partial run; pattern `raw/code/nt_run2.py`, `score_e2.py`); `H-REFE-RL1-NCRES` read (needs PREREG) |
| RL-2 | **DONE 2026-10-18, 0 GPU** (`raw/2026-10-19-rl-lanemix/`, `D-REFE-RL2-LANEDOSE`): the lane-dose composite L0.25 passes RL-val vs base0 (+0.308 [+0.003, +0.609]) but is NOT separable from V1 (+0.023 [-0.150, +0.200]). **Next:** navtest look #2 for L0.25 ONLY IF RL-1 V1 reads N-GAIN-LOST (`MM_DECISION_NAVTEST2.md`): seam via `rl/nt_build.py`, ~40 min CPU. **Anchored RL arm: BLOCKED on the A40** (P6 holds it; the PI declined extra compute) |
| LOCK-STEAL-2 | ~~**Four more dev-box lock helpers can remove a live job's lock** (sweep 2026-10-18, `TRAPS.md` "two lock schemas share devbox_gpu.lock"): `c3_run.py:114`, `c3_run_p2.py:126`, `p2s1_autoread.py:449` use the same dead-pid-and-600-s rule the `aux_navtest_forward` fix removed; refav1's `gpu_lock.py:58-83` (called from `run_encode.sh:50`) renames aside any lock without a `t` field as pre-boot, even with a live holder. None was running when found. Apply the same rule (clear only a lock carrying your own owner marker) with a foreign-schema regression test BEFORE any of them is reused; refav1's first.~~ ⏹ **DONE 2026-10-18, 0 GPU.** All four helpers now clear ONLY a lock carrying their own marker (refav1 `gpu_lock.py`: `owner == --owner`, release needs owner AND pid; `c3_run.py` / `c3_run_p2.py`: `job` starts with `c3-selection-headroom-` / `p2-own-table-`; `p2s1_autoread.py`: `job == refe-stage2-<name>`, the lock `eval/stage2_sft.py:115` writes). Each has a new `test_*_foreign_lock.py` (foreign-schema cases, an own-marker zero control, RED mutations). Pins of the old steal inverted: refav1 `test_gpu_lock.py` tests 5-6 (+7-8 pin the inverse), `test_p2s1_autoread.py` `dead_holder_and_old_lock_is_removed`. ⚠️ Second defect fixed in both c3 helpers: `pid_alive` crashed (TypeError) on a dead pid because the German-locale `tasklist` line carries 0x81 (TRAPS). ⚠️ Behaviour change: a FOREIGN lock orphaned by a crash or reboot is now waited on, never cleared; a human renames it aside (memory: compare `acquired` with `LastBootUpTime`). Pre-existing, unchanged: `test_p2s1_autoread.py` `dry_run_never_writes_real_markers` fails on the tip because the banked `out/P2S1READ_DONE` exists (passes with it moved aside). | 0 GPU, ~1 h each |
| FP-POD-CLEAN | **Startup-check half CLOSED 2026-10-18 (INSTFIX)**: `taniteval/tools/chain_controls.py` (+34 tests) refuses a chain plan at startup when a gated control has no producer; on the REAL banked chain the original plan is refused with exactly `NO-PRODUCER 'D3.json'`, the fixed plan is clean, removing the producer goes RED; improved copy `…/2026-10-18-instfix/code/fp_chain_v2.py` (adds `ic1`, `d3_smoke`, `d3_link`; ledger checks). The banked free-point package and its navtestE copies are untouched and still carry the defect -- any successor chain must adopt the helper. **LEFT (needs pod access, not done by this agent):** delete the pod's `/workspace/ops/freept` once the pod is not training AND the D: copy is re-verified against `MANIFEST.sha256` (HANDOFF step 6). **Free-point navtest read DONE 2026-10-18** (`raw/2026-10-17-free-point-navtest/`, VALID, row FP-NONE, `D-REFE-FREEPOINT-1`). Left: delete the pod's `/workspace/ops/freept` (bulk outputs verified on two dev-box disks, `MANIFEST.sha256`) when the pod is not training; and give successor chains a startup check that every gated control has a producer stage (D3 was absent, not failed). | nothing / 0 GPU |
| P6-3C | **P6 on the 3-C twin bank.** The `P6_CBANK` knob + commands-stage guard are landed in `refe/p6_pod.sh` (this commit; full-form tests: test_p6_chain 341, test_p6_cpu_probe 111, test_rls1_chain 522, test_rls1_queue 254, test_p2r_chain 139). The orchestrator `raw/2026-10-17-twin-bank-3c/post_merge/post_merge_to_p6gate.sh` is landed with its rehearsal (68 scenarios, 15/15 mutants). **2026-10-17 ~20:50 CEST: the PI MIGRATED the pod to a new machine; the volume copy was still running and the new container is SMALLER: 8.5 CPUs / 55.0 GB (was 11.05 / 71.0, MEASURED). P6 derives its label workers from the quota (`p6_pod.sh:201`, cores - 4): 7 -> 4, so P6 is ESTIMATED at ~12 days instead of 6.8 if the labeller binds (`eval/P6_ON_11VCPU.md` section 6) -- **PI decision 2026-10-17: MEASURE FIRST.** Keep this pod; in P6's first half hour read the chain's early labeller-throughput reading (`lab_early.json`, `eval/P6_ON_11VCPU.md` section 9), project the run's days with section 6's formula, and if the projection is well over 7 days report it to the PI, who decides then. No spend now. The 3-C re-roll resumes at NPROC 9 (`PREREG_3C.addendum5.md`); the pod-side HF token file did not survive the migration and must be re-placed by the PI before any gate.** Remaining, in order, after the copy completes: (1) the 3-C re-roll resumes and merges; (2) re-ship P6's tree at this commit per `P6_POD_RUNBOOK` section 2 (the pod holds `c1dc5d33`: no knob, no R43); (3) run the orchestrator with `SSH`, `P6N_SHIP_COMMIT`, `P6N_HF_TOKEN_FILE` from the environment only; PI 2026-10-17: P6 takes the A40 before SAFEOBJ Stage 2, auto-launch on a judged gate PASS, any refusal stops. Then ZT4 (`D-REFE-ZEROTAIL-4`'s next lever) on P6's fan. **2026-10-18: the launch-gate FAIL (G-CLOCK, G-SUITE, G-LIVE-LAB) is FIXED and landed** (`raw/2026-10-18-p6-gate-diagnosis/`): G-CLOCK registers the tuple count per bank (167,300 for 3-C; `PREREG_3C.addendum6.md`, sha256 `d6462e23…`), G-SUITE's sandbox scrubs the chain namespace by prefix (the leaked `P6_CBANK` would have `rm -rf`'d the real bank but for the chain's guard), G-LIVE-LAB gets `--label-chunks 2`. Battery `raw/battery/TEST_RESULTS.md`: every test passes or fails on base too, except `test_rls1_queue.py`, shown environmental (a race in the test's setup on a starved box; its fix waits for a quiet full-form pass). NEXT: re-ship at this commit and run the orchestrator from stage e (gate dry run, auto-launch on a judged PASS). **P6 LAUNCHED 2026-10-18 ~11:45 Berlin** at `bcc04b5`: launch-gate dry run PASS (token `PASS_bcc04b52233e.json`, argv sha `20dcebecc3014930`, tree `aac65ffb51df`), real run started, first step logged (step 0); orchestrator `ZZPM DONE result=OK`. The `test_rls1_queue.py` race fix (staged at `C:/Users/Admin/p6gate/out/`) stays UNLANDED: its 'quiet' rerun was not quiet (commit headroom 6-10 GB, Git-bash fork failures in the fixed runs while base passed concurrently), so it is rerun only with no agents or test lanes on the box. | pod return + 3-C merge / A40 |
| P6-HN | **Step 0 DONE 2026-10-18** (`raw/2026-10-18-hn-census/`, `D-REFE-HN-CENSUS-1`: PARTIAL; expert-proximate hard negatives in 33.7 % [23.3, 46.7] of held-out sets at 0.5 m; reach ESTIMATED 0.29-0.32 PDMS gross on ADE, 0.46-0.57 on a post-hoc cross-track axis; the arm's prereg must fix the proximity axis first). **Expert-proximate hard negatives, the published form of ZT3's near-twin grazes** (LAB-RUN-019 finding 2: three single-run deltas cut the NC x DAC zero proxy by 21-23 %, `2608.19085` Tab.3, `2605.19771` Tab.4; PUBLISHED). A single-lever A/B AFTER P6 reads. Reach ESTIMATED 0.55-0.94 PDMS gross (13-23 % of the 4.15 zeroed-token term; arithmetic in the Lab report). Step 0, 0 GPU: a near-twin census on the 3-C bank (how many expert-proximate negatives exist per set, and at what distance). Map-admissibility of map-derived hard negatives is part of the open PI ruling C9 (labels vs inference; LAB-RUN-019 map flags). **Arm prereg DRAFT 2026-10-18** (`raw/2026-10-18-p6hn-prereg/PREREG_P6HN.draft.md`, sha256 `4a9ea208…`, NOTES.md): axis = mean absolute cross-track to the expert path, threshold 0.60 m, met by construction by shifting each tuple's own target ±{0.15, 0.30, 0.45, 0.60} m; one flag `--hn-bank` (G-DVB row, `refe_p6hn` profile = P6's argv plus two changes); primary full-navtest paired ΔPDMS vs P6 with safety co-gate HN-UNSAFE (NC or TTC adversely separated). Design finding: P6 already trains its scorer on online labels of its own 64 proposals, so the census's hard negatives are already in P6's loss; the lever's new content is the offset bank. Power (ESTIMATED): 6.8–7.65 A40-days and a priori most likely HN-NOT-PROVEN; the one published scorer effect is +0.22 PDMS (`2608.19085` Tab. 3). PI: Q1 the full arm vs a cheap scorer-only screen (seeds 0–2) first (draft recommends the screen); Q2 C9. **PI 2026-10-18: screen first** (NOTES Q1 option b): a scorer-only screen under its own prereg, trunk frozen, with and without the offset bank, seeds 0-2; the full arm only if the screen's pre-registered outcome says so. NEXT: the screen prereg. **Screen prereg REGISTERED 2026-10-18** (`PREREG_P6HN_SCREEN.md`, sha256 `9eb7cb2f…`; `H-REFE-HNSCREEN-1`): u400 frozen; RL-val zero-pick endpoint (B - A, seeds 0-2, joint log draws) with a PDMS and NC/TTC co-gate; 0 A40-h. MM decisions in `MM_DECISIONS_SCREEN.md`; the full-arm draft takes the masked two-call scoring (amendment A). BLOCKED for S0-S3 on ~105 GiB of disk (PI). NEXT: checklist items 1, 2, 5, 6, 7 (new files) now; 3, 4, 8 after B8 lands. **PI 2026-10-18 later: disk freed on D: (197 GB free, MEASURED) -> S0-S3 UNBLOCKED, cache on D:; a GO comes back to the PI (decide then).** **Screen implementation part 1 LANDED 2026-10-18** (`raw/2026-10-18-p6hn-prereg/impl1/`): `refe/hn_bank.py` (builder + the masked two-call term, ZZTEST_PASS 25), `refe/ctx_cache.py` (ZZTEST_PASS 15), the three `eval/hn_screen_*` readers (ZZTEST_PASS 18, every §11 row reached), synthetic fixtures only. **Amendment B** (`PREREG_P6HN_SCREEN.amendmentB.md`, before any data): depth>1 wording, the duplicate-row identity asserted with self-attention neutralised, I6 bounds defined as RL-S1's pinned 0.5 / 0.025 (the cited calibration chose none), §10 pairing kept (families instrument must export per-set values), SY on the point estimate. NEXT after B8 lands: items 3, 4, 8 + the families per-set export. **Part 2 LANDED 2026-10-18** (`impl2/`): taniteval per-set export (`per_window.py`, flag off byte-identical) and the screen's paired families (`test_hn_screen_read.py` ZZTEST_PASS 31); three reporting readings fixed in `MM_DECISIONS_SCREEN.md`. | P6 read / 0 GPU for step 0 **UPDATE 2026-10-19: screen checklist items 3 + 4 LANDED** (trainer wiring `--hn-bank`/`--ctx-cache`, G14/G15, the `refe_hnscreen` profile; 0 new failures incl. CUDA tests; merged-tree check green). **Next:** build the bank and cache, fill `fingerprints_expected`, then the screen run (~8-12 dev-box GPU-h; GPU queue: L1 Step A → navtest head bank → HN screen). AMBIGUITIES 1-12 in `impl3/RESULT.md` |
| ZG-WIRE | **Wire the post-training zero-gradient gate into the chains** (landed 2026-10-18: `stack/scripts/zerograd_gate.py` + `stack/tests/test_zerograd_gate.py` 14 passed; acceptance `…/Implementation/incoming/2026-10-18-zerograd-gate/`: on refcv5-v2's final ckpt it FAILs naming `offset_head` 6,160 + `tac_goal_tok_head` 11,286 + `scorer.goal_point` 1,026 = 18,472 params, MEASURED, reproducing the banked finding). Nothing calls it yet: (1) `stack/ops/sup_refcv7.sh` before `write_done_marker`, refusing marker/publish on rc != 0; (2) trainer payloads add `param_names` so naming is exact (`refc_v3_train.py`; pinned by `tests/test_refc_v3_save_before_eval.py`); (3) check `ddv2_rl_refcv5.py` carries `opt` (`ddv2_rl_refcv7.py`'s final file is weights-only, the gate exits 2). Side observation, uninvestigated: `lat_head_tac`/`lon_head_tac` Adam step 40,134 vs 40,284 for every other trained param. **DONE 2026-10-18** (`…/Implementation/incoming/2026-10-18-zerograd-wire/`): `sup_refcv7.sh` rule G5 runs the gate on `ckpt.pt` before the done-marker on every completion path, reads the verdict from the gate's JSON, fails closed (`ZEROGRAD_FAIL.json`, exit 6, never relaunched, also after a restart or a deleted marker) and records PASS / not-applicable in the marker; `refc_v3_train.py` writes `param_names` in optimizer-id order. Tests: 22 new pass, 0 regressions over the 118 stack test files that reach an edited file (2,473 vs 2,451 passed at base, the same 12 environment failures in both); MM re-run 40 passed. Open: (a) the trainer's own `summary.json {"done": true}` is untouched on FAIL, so consumers must also require `ZEROGRAD_FAIL.json` absent; (b) read the first real refcv7/8 ckpt by hand (a synthetic smoke without `--no-strategic` flags `core.route_head`); (c) `ddv2_rl_refcv7.py`: gate `ckpt_latest.pt`, not the weights-only final file; `ddv2_rl_refcv5.py`'s final ckpt carries the SFT optimizer, so gating it would judge the wrong run. | 0 GPU |
| PAPER-720 | Follow-ups from the §7.20 draft (`raw/2026-10-17-paper-s720-recompute/CONFLICTS.md`). **C3, C4, C11 DONE 2026-10-17**: registry §14.2 carries the P2-s0/P2-s1 rows, the marginal and gap-decomposition intervals and the zero tail (ZT3 reach marked GROSS per R47); the later duplicate register id is now `D-REFE-GAP-1b` with its citations fixed (register, registry, paper §7.20; banked raws left as written); 27 vs 25 epochs is CLARIFIED, not an error (27 = the trainer's grow-mode epoch counter, 25 = scheduled). **Open:** (C9) A9 reads NAVSIM's roadblock graph at inference while A8 was refused as a map use -- surface to the PI that the 2026-10-04 ruling is meant to stand; (C2) the registry's "best REFe" mark on u400 (88.25) now sits beside higher P2 points (88.82, 89.15), left as banked -- the PI's call; §14.1 has duplicate field rows (pre-existing). | PI / 0 GPU |
| LAB-D | **The daily Research Lab run lapsed after LAB-RUN-017 (2026-09-20)**: about four weeks with no Lab run. Restarted as LAB-RUN-018 on 2026-10-17 (constrained-RL levers for SAFEOBJ O4/O5, NAVSIM scorer-based state of the art). The Master Mind triggers it daily; a missed day is a stream lost. | nothing / 0 GPU **LAB-RUN 2026-10-19 DONE (SAFE-HEAD-LIT):** safety-row resolution in generator-scorer planners; 4 papers banked, 17 entries extended; 102/102 rows verified against the PDFs; `D-LIT-SAFEHEAD-1`, `H-REFE-FASTNEG-1`, FASTNEG-CENSUS registered. |
| HJ-UNITS | **CLOSED 2026-10-18 (INSTFIX / HJ-UNITS-3, `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-18-instfix/`)**: the last silent unit defaults are gone -- `actdiv_anchored.py --action-units` (was `kappa`; REQUIRED for `--family refav1`, not read by v7/selftest), and, per the s5 rule (live integrations of the recorded channel), `refcv3_arm.py` and `openloop_suite.py` (were `steer`; REQUIRED for a rollout, not for `--analyze-only` / `--arm-json`); the two live callers that relied on it now state `steer` (`bench/internal_t1.py`, `run_hierarchy_panel.py`). `test_refav1_units_required.py` 121->147 (declaration census over every live `--action-units` + invocation census over the four CLIs, positive/negative controls, mutation RED for each restored default and each deleted gate). 36 test files base 916/0/18 -> after 942/0/18 (+26, all in the units file); 19 bench-importer files identical. 16 frozen rows (8 launchers) now exit 2 if re-run (FROZEN_CALLERS.md). LEFT, each with its reason in RESULT s5: `render_refcv3_video` dead flag, `p10_prior` / `v8f_mini` / `ra1_pooled` steer defaults (training-data construction; closing = 2+2 call-site edits in `stack/tanitad/`, a pod-parity call), anchor-bank `control_units` (`kappa|alat`, no `steer`), the manifest-lacks-field `kappa` fallbacks. (`FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-18-hj-units-2/`) ✅ DONE 2026-10-18 (9939892 + 5fb509a + HJ-UNITS-2): no silent unit default in the refav1 path -- REQUIRED, keyword-only: paths_from_controls, floors_kin.const_control_paths/ha0_ext_paths, refa_v1_plan.unicycle_paths, RefAV1.plan/imagined_goal (model_action_units), _imagine_tactical_goal (units), cost_surface_probe.WindowContext.score (+ now refuses an unknown unit); refav1_arm.py --action-units has NO default (required for every rollout, not for --analyze-only). test_refav1_units_required.py 49->121 (census over the new callees with positive/negative controls; mutation RED for each restored default); 67 test files base==after (1212/0/17). 42 frozen files / 44 sites now fail loudly (FROZEN_CALLERS.md). OPEN (named, not in scope): actdiv_anchored.py:1384 --action-units default "kappa" (legacy direction), refcv3_arm/openloop_suite/render_refcv3_video CLI default "steer", p10_prior/v8f_mini steer defaults, refc*/anchor control_units="kappa", the manifest-lacks-field "kappa" fallbacks. | nothing / 0 GPU |
