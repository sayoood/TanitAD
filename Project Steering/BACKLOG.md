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
| C2 | **C64 option A** — score both arms on the 19 leak-free episodes, **v1 re-scored there** | v2corpus reaching 30 k (~17:00 UTC 2026-07-29) |
| C3 | **v5 gate verdict** | v5 reaching step 2000; ⚠️ first probe is the INCUMBENT and cannot stop the run |
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
