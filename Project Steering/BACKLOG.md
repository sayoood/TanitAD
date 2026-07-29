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
| A1 | **Build the emitters** for the 30k gate (PI: *"build the emitters"*, task #42) | the gate returned NO-VERDICT for lack of them |
| A2 | **Wire the NC safety gate** with `filter_m(agent,human)=1.0 if m(human)==0 else m(agent)` — ⛔ **NOT PDMS-v1**, which over-penalises | our only multiplicative safety gate is empty; the join blocker is retracted, only a chunk download remains |
| A3 | **C64 option B** — build the clean v2-line val from the 9,987 unselected clips, stratified to match the v2 TRAIN distribution, emit the exclusion proof **even when zero** | PI said do both A and B; B is 0-GPU and needed before any further v2-line training |
| A4 | **Re-verify the factorised path × velocity vocabulary** (refuted 0-3 in the survey) | maps 1:1 onto our LAT+LON softmax mechanism and the 88.7 % longitudinal gap — the single most valuable re-verification target |
| A5 | **Confirm C64 at clip_id granularity** — replay `discover_r0_clips` on pod2's raw root | deferred only while pod2 trains; the number that voids an experiment should be exact |
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
| B4 | **Matched-capacity camera head** for sitclf — discriminates "optimisation shortcut" vs "representation limit" | the sitclf hypothesis is currently untestable without it |
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
