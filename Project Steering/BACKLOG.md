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
| A13 | **Adjudicate the 24 DRIFT rows in Thor's live `/home/nvidia/TanitAD` checkout** — outside the launch closure (`stack/tests/*` ×8, `stack/scripts/*` ×9, `taniteval/*` ×5, `Paper/figures/*` ×1) | ⚠️ **direction unknown** — a content comparison cannot say which side is newer. Decide which is authoritative **before** either is overwritten |

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
| E10 | **Registry hygiene**: give the Alpamayo augmentation counts their own registry row (the paper carries them INHERITED) | 0-GPU | nothing |

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
