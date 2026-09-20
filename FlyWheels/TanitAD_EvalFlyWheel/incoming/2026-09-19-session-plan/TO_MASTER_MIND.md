# EvalFlyWheel → Master Mind — 2026-09-19 (written channel)

> ⭐ **RE-SCOPE RECEIVED (MM, 2026-09-19, later the same morning):** the PI's commission
> (NavSim/nuScenes + leaderboard) takes precedence. **Task 2 (S1) and task 3 (E9) move to the
> TrainingFlyWheel.** ⚠️ **E9 had ALREADY been completed** before the re-scope arrived (§2 below,
> staged). The TrainingFlyWheel should **reuse it, not redo it**. The **S1 fan-source proposal
> (§3)** is handed over as a design input; no S1 code was written. Task 1 (§1) was cheap and is done.

**Why this file exists:** three `SendMessage`s to `TanitAD project kickoff (fork 8)` were **held for
the recipient user's approval and expired undelivered** (different permission mode). Per the
FlyWheel README, the durable channel is `incoming/`. Everything below was sent live first and is
repeated here verbatim in substance.

## 0. ACK (your STEP 0)

**EvalFlyWheel (fresh, D:) awake.** Working in D:'s main checkout, branch `agent/arch-inf-20260803`.
Staging by exact file path only; never `add -A`, a directory, `reset`, `checkout`, `stash` or
`commit`; never the G: path. CPU only.
⚠️ **Fan-out status, honestly:** this session was PI-started *before* your brief arrived, with the
focus *"NavSim/nuScenes + leaderboard"*, and it had launched five sub-agents. Weekly usage read
**87 %** (MEASURED via the usage API), so **three were paused** (E3 estimator/route-leak, E4
leaderboard, E5 nuScenes). Their transcripts are intact and resumable after the reset. **Two NavSim
streams continue** (E1: first real EPDMS on warmup for the reference agents plus the metric cache;
E2: refcv4b through the official scorer), because the PI authorised the navhard download for them
in chat. There will be no new fan-out.

## 1. TASK 1 — old-session triage → `../2026-09-19-old-eval-session-triage/RESULT.md` (STAGED)

- **a1 drift framing:** STILL OWED but moot. `taniteval/taniteval/val40_overlap.py` does **not exist**
  in HEAD or in the `tanitad-det` mirror; the base never landed.
- **a2 blobs-found:** OBSOLETE. It concerned an 08-30 label pin, superseded by the v7.2/v8 releases.
- **a3 roots→MEASURED:** the `eval_corpus_root_union` entry is absent from HEAD, and so is
  `val40_shared_support`. ⚠️ **VERSION COLLISION:** HEAD's registry **2.8.0** is a *different*
  change (`hyg.inference_seed`, 09-06); the old 2.8.0 from 08-30 never landed.
- **b five drift tests:** STILL OWED. `test_val40_overlap.py` is absent; it ships only with a1's base.
- **c Hub→Lab sweep:** PARTIALLY LANDED. 34 files / 45 lines → **30 / 36** in HEAD (control: 64
  test files cite Research Lab). **7 stale path constants in 6 LIVE taniteval tests are still
  owed** (`test_clhorizon:39`, `test_corridor:47`, `test_k1_degeneracy_guard:300`,
  `test_lead_source:21`, `test_ood_guard:37,40`, `test_ridge_intercept_penalty:29`). They probably
  skip rather than fail. The other lines: 3 are directive-quoting comments (fine), 6 are the
  deliberate `test_library` dual-name guard (keep), and 20 sit in banked `incoming/**/tests`
  (historical).
- **The 46-file scratchpad: CONFIRMED ABSENT.** The val40 guard survives **only** in the transcript's
  `Write` inputs. `deployed_val40_clip_digests.json` in HEAD still cites the dead Hub path.
- val40 relevance: clean-124 / eval139 / eval6-REF all read **0** overlap with the 40 digests.
  ⚠️ **INCONCLUSIVE:** no val40-containing cache was available as a positive control.

## 2. TASK 3 — E9 → `../2026-09-19-e9-t1eval-refcv6/RESULT.md` (STAGED, + 4 raw logs)

**NO — `t1_eval.py` is UNWIRED for refcv6, by construction.** It refused three times, the last
verbatim: *"has no 'model'+'grounding' keys — the adapter path needs a flagship trunk checkpoint
(v1/v4/v5f shape). Keys: ['model', 'opt', 'step']"*. Its geometry also defaulted to DEPLOYED
256×256 pinhole. REF-C has **no action input**, so there is nothing for it to roll. **The wired
REF-C T1-family harness is `refcv3_arm.py`** (W1, `4000946`): RC 0 on A3's checkpoint at
416×1024 on halfB, 9 windows / 2 eps in **173 s ⇒ ≤ 19.2 s/window on CPU all-in**; GPU rate NOT
MEASURED. ⇒ A2's *"t1_eval at 416"* criterion cannot be met for any REF-C arm. §7's *"C8 is live"*
branch does **not** fire if A3–A6 read through `refcv3_arm.py`. Any refcv6 *"T1 via t1_eval"*
number must be relabelled. The tool stamps `os` **`T1*`** and itself asks whether EVAL_DOCTRINE
admits a no-action model as T1; that is **escalated, not decided**.

## 3. TASK 2 — S1 harness: the fan-source proposal (sent BEFORE building, as asked) — ⛔ AWAITING YOUR CALL

**(1) Fan source, proposed:** refcv5-v2 @40,284 (`C:/Users/Admin/refcv5v2_final/ckpt.pt`, item 19's
untouched base), regenerated through `stack/scripts/ddv2_rl_refcv5.py`'s own
`Ctx`/`fetch`/`capture`/`score_batch` path, with its pinned per-window seeds `500000+k`, on the
**same 493 windows / 40 clusters**, on CPU. **Acceptance gate before any S1 arm:** reproduce
`C:/Users/Admin/tanitad-caches/ddv2rl-20260915/heldout_base.json` — 493/493 rows, **28** collided
selections, **28/28** with a collision-free candidate, ≈**55.3 %** collision-free share. That makes
A4's control an exact reproduction. The gate re-ranks by the decoder's own `sel_score` (emitted with
`sel_idx`, `refc.py:3251`). S1-RANDOM uses the exact analytic no-information value.
S1-GATE-CONST passes an all-free occupancy and must return `sel_idx` **bit-for-bit** (recovery
exactly 0); a mutation test must go RED if the re-rank reads anything but the occupancy.
Alternatives rejected: the refcv6 planner (0.38 of an epoch), refcv4b (no measured collision
headroom), a model-free fan (tests only the machinery).

**(2) Held-out (MEASURED):** of the 493 windows / 40 clips, **halfB holds 18 clips / 218 windows /
14 of the 28 collided**. halfA (A3-trained, inadmissible for PRED) holds 17 / 208 / 12; 5 clips
(2 collided) sit in neither. ⇒ PRED is scored on the 218-window halfB subset only. BASE / RANDOM /
ORACLE / CONST are re-read on that same subset, and on all 493 for the item-19 reproduction.
**n = 14 events: underpowered, stated before any number exists.**

**(3) The open design point:** A3's 0.576 is `map_iou_drivable` (floor 0.3388), while item 19's
statistic is `sel_nc`, a collision with **agent** tracks. ⭐ **Correction to my first message:** A3's
checkpoint is **not map-only**. Its argv carries `--agents head --w-agent 1.0 --join3d …
--w-box3d 1.0`, so **trained agent and box3d heads exist (halfA only)**. ⇒ **S1-GATE-PRED-AGENTS
(= H-SEL-GATE-1 as registered) needs NO new training.** It needs a **held-out agent read** first
(box/agent quality on halfB vs the b1eval 3-D join, with a no-information floor, mirroring A3's map
read), then a declared motion model (constant velocity over the horizon). A PRED-MAP drivable gate
is optional and must carry a separate label, because it is not H-SEL-GATE-1.
**Recommendation:** a pluggable occupancy provider. Land the item-19 reproduction plus
BASE/RANDOM/ORACLE/CONST first (no perception needed), then the held-out agent read, then
PRED-AGENTS on the halfB subset. Side note: the 09-15 held-out logs never mention a map, so item
19's +0.0485 `sel_pdms` ceiling was probably computed with DAC = 1 (UNVERIFIED; checked during the
reproduction).

**Reply channel:** a `SendMessage` to *"Start TanitAD_EvalFlyWheel: NavSim/nuScenes + leaderboard"*
reaches this session directly. Mine to you expire unless the PI approves them in your session.

## 4. Other things you should know (for your register / integration)

- **navhard_two_stage** download PI-authorised in chat (*"Pickles + sensors"*). **38,537,174,373 B**
  per HEAD `X-Linked-Size`, not the 31 GB in `docs/splits.md`. Content-verified by **sha256 ==
  HF `X-Linked-ETag`** (pickles confirmed). 16 parallel ranges run at **12.4 MB/s** vs 1.3–1.4 MB/s
  single-connection. Fetcher and receipts are in `../2026-09-19-navhard-download/`.
- PI decision: **the PI will register for nuScenes personally** (agents still never register).
  Proposed rows `D-NAVHARD-DL-1` and `D-NUSC-REG-1` are in `COMMS.md` beside this file.
- `../2026-09-19-leaderboard-currency/raw/audit_eval_packages_since_0903.md`: a 61-package eval
  currency audit, including **7 MODEL_REGISTRY contradictions** (e.g. the v7 T1 block still headed
  *"CLOSED-LOOP CAPABILITY READ"*; §1.12 calling the in-proc gsplat rollouts *"the AlpaSim/NuRec
  panel"* although they record `"transport": "inproc"`).
- ⭐ For the NavSim harness check: published navhard reference values sit in the Lab's own packages
  (pseudo-sim v3 Table 2, post-fix #151): **CV 11.4 · Ego-MLP 14.1 · LTF 25.1 · PDM-Closed 56.6**
  (pre-fix: 10.9 / 12.7 / 23.1 / 51.3). Our CV number on devkit `0a380a9` should land near **11.4**
  if the harness is right.

## 5. DELIVERABLE MANIFEST — files DONE and ready to land (also sent by SendMessage; held for approval)

All NEW (absent from HEAD `37645fc`), all STAGED, index blob == worktree blob (40-char verified
2026-09-19). Paths are under `FlyWheels/TanitAD_EvalFlyWheel/incoming/`.

| path | blob | state |
|---|---|---|
| `2026-09-19-session-plan/PLAN.md` | 6371efd5 | DONE |
| `2026-09-19-session-plan/COMMS.md` | a3afb043 | DONE |
| `2026-09-19-session-plan/TO_MASTER_MIND.md` | *this file — re-staged after this section was added; verify its blob at landing* | DONE |
| `2026-09-19-old-eval-session-triage/RESULT.md` | 5a5989e1 | DONE |
| `2026-09-19-e9-t1eval-refcv6/RESULT.md` | 6d1d07b2 | DONE |
| `2026-09-19-e9-t1eval-refcv6/raw/e9_t1eval_attempt.log` | 8ae66f60 | DONE |
| `2026-09-19-e9-t1eval-refcv6/raw/e9_t1eval_attempt2.log` | 217956a4 | DONE |
| `2026-09-19-e9-t1eval-refcv6/raw/e9_t1eval_attempt3.log` | cbd03867 | DONE |
| `2026-09-19-e9-t1eval-refcv6/raw/e9_refcv3arm_rate.log` | 5adc1665 | DONE |
| `2026-09-19-navhard-download/code/navhard_fetch.py` | 755d5c45 | DONE |
| `2026-09-19-navhard-download/raw/receipt_scene_pickles.json` | 1f3fdd1a | DONE |
| `2026-09-19-leaderboard-currency/raw/audit_eval_packages_since_0903.md` | 252bb2f0 | DONE (INHERITED audit; the only durable copy) |
| `2026-09-19-navsim-warmup-reference-epdms/BRIEF.md` | 3cd430cb | DONE (record) |
| `2026-09-19-navsim-refcv4b-bridge/BRIEF.md` | 70a636dd | DONE (record) |
| `2026-09-19-navsim-estimator-and-route-leak/BRIEF.md` | 20cec5de | DONE (record) |
| `2026-09-19-leaderboard-currency/BRIEF.md` | 4812a8b8 | DONE (record) |
| `2026-09-19-nuscenes-planning-harness/BRIEF.md` | 28ecd8bd | DONE (record) |
| `2026-09-19-navsim-refcv4b-bridge/SPEC.md` | 3ca367ca | ⛔ NOT DONE — E2's pre-registration, still live; landing it before E2 scores would timestamp the bar (your call) |
| anything else E1/E2 stage later; `2026-09-19-navhard-download/raw/receipt_sensors.json` | — | ⛔ NOT YET — a later manifest will cover them |

## 6. ⭐ E1 LANDED — the programme's first locally computed EPDMS, and the harness is VALIDATED

Package `2026-09-19-navsim-warmup-reference-epdms/` — **89 files, all STAGED and verified by E1;
DONE, ready to land** (RESULT.md is the entry point). The orchestrator re-read the headline number
from the raw CSV.
- **Constant velocity, official two-stage runner, `warmup_two_stage`:** combined EPDMS
  **0.1853562745** (S1 0.4602892, S2 0.3341286) = **18.535627 ×100**, against the HF warmup
  leaderboard's `baseline_constant_velocity` **18.5356**: **Δ +0.000027**. MEASURED in
  `raw/A1/devkit_2026.09.19.12.24.20.csv`, row `extended_pdm_score_combined` (the leaderboard value
  is INHERITED from `NAVSIM_PROTOCOL.md` §6.3). ⇒ **Our local harness reproduces the public
  leaderboard at its printed precision.** That is the independent cross-check any TanitAD NavSim
  number rests on. Devkit `0a380a9` is on the **post-#151** side (two probes).
- ⛔ **The human agent is UNDEFINED on every two-stage split.** All 204 synthetic scenes have
  `num_future_frames = 0`, so the official runner crashes. The human is a stage-1-only reference:
  **0.951255** (n = 16, filter ON); **0.872014** with the filter OFF (mutation arm).
- Corrections to my brief (E1, from source): background traffic is **IDM-reactive in both stages**
  (`run_pdm_score.py:77-79, 132-134`); CV's EP is **not** far below the human's under v2 EP
  (0.8445 vs 0.8564).
- ⛔⛔ **Two truth-layer DEFECTS for the register — the "built, registered, unreachable" class:**
  (1) **`tools/criteria_check.py` never evaluates `benchmarks.navsim`.** The four BLOCKING NavSim
  gates (estimator_unit, ego_enforcement, modality_label, cross_protocol) exist only on paper; any
  NavSim artifact passes the checker regardless. (2) **The registry's `EPDMS_v2` variant lists the
  multipliers `NC, DAC, DDC` and omits `TLC`**, while the same registry's sub-metric formula
  multiplies by TLC. Both belong to E3 (paused for budget). Proposed rows:
  `D-NAVSIM-HARNESS-1` (the reproduction above), `D-NAVSIM-HUMAN-UNDEFINED-1`,
  `C-CRIT-NAVSIM-UNENFORCED-1`, `C-REG-EPDMS-TLC-1`.
- Windows devkit defect: `MetricCacheLoader` splits on `/` (`dataloader.py:316`), so it cannot read
  a cache written with backslash paths. E1 fixed it with a separator-only patch, unit-tested 6/6.
- E1 was **PAUSED** after landing, to protect the budget. Its navhard priority-2 steps
  (C:-copy verify → metric cache → CV two-stage → human stage-1) now run as an **agent-free
  background chain** (`../2026-09-19-navhard-download/code/navhard_reference_chain.sh`, log in that
  package's `raw/`). navhard has **76 logs** (MEASURED from the yaml), so a log-cluster CI is
  possible there, unlike warmup.

## 7. ⭐ E2 LANDED — refcv4b through the OFFICIAL NavSim scorer: the bar PASSES, and STANDING STILL beats it

Package `2026-09-19-navsim-refcv4b-bridge/` — **147 paths STAGED and verified by E2 (index blob ==
worktree blob, all under 1.6 MB); DONE, ready to land.** Entry point: RESULT.md. The orchestrator
re-read CV and A1 from `raw/scores_summary.json`. MEASURED; T1-family; S1 loop OPEN / S2 UNRULED;
IDM-reactive traffic; no CI (7 logs < RG-14 floor).
- **Pre-registered bar PASSES:** A1 (frames + t0 ego + command), stage-2 EPDMS-u **0.46702** vs CV
  **0.39713** on the same 204 tokens (+0.0699).
- ⛔ **But an all-zero STOP plan scores 0.5212 on stage 2**, i.e. **0.3009** on the official
  two-stage protocol vs CV's **0.1854**. Nothing that moves beats standing still. The vision-pure
  arm (A2) collapses to a stop on 202/204 scenes.
  ⇒ **This is NOT a driving claim.** Mechanism, from source: EP is set to 1 when the best
  rule-compliant progress is ≤ 5 m (`pdm_scorer.py:231-236`), and warmup's stage-2 starts are slow
  (median v0 4.14 m/s). ⇒ **Every NavSim row needs a STOP floor beside it** (proposed
  `D-NAVSIM-STOP-FLOOR-1`).
- A1 beats CV through **caution**: NC +0.118, TTC +0.113, DAC +0.054; it loses EP −0.152 and EC
  −0.186. **Vision does real work through the bridge:** A1 − frames-blind = +0.372, and A1 − its own
  ego echo = +0.0383 (on DAC/DDC).
- ⛔ **No refcv4b two-stage EPDMS can exist on warmup:** its 16 stage-1 scenes ship **0/192** camera
  jpgs. ⭐ **navhard does NOT have that gap.** MEASURED by the orchestrator: 60 random navhard
  `openscene_meta_datas` → **900/900** CAM_F0/L0/R0 paths present in the verified C: copy.
  ⇒ **the next lever is refcv4b on navhard two-stage, the official column.** The navhard metric
  cache is building now (agent-free chain: step `mirror` rc 0, `cache` running). E2's frame builder
  is navhard-ready and rebuilds the warmup bank bit-exactly (6/6).
- Controls all read their known values: frames 204/204 hash-equal; the GT round-trip is exact to
  7e-15 m; the seam reproduces the official CV run with max |Δ| 0.0; randomising undeclared ego
  fields leaves outputs byte-identical (20/20), while mutating a declared field changes them
  (20/20). All four `benchmarks.navsim` gates were self-checked PASS on every arm, because
  `criteria_check.py` does not evaluate them (§6 defect, confirmed independently).
- DataFlyWheel correction: its "204/204 scenes" is stage 2 only; stage 1 was never built.

## 8. Your "pause or throttle NAVSIM until A8 finishes" — OPTION (a) DONE

- **Stopped by explicit PID** (`Stop-Process`, each command line read first): retry-chain bash 2864 →
  run_navhard.sh bash 18964, 27732 → navsim-crun python 34108 → 45456 → pool workers 43672, 52952.
  A re-scan shows **0** NavSim processes; **A8's 16996/21724 were untouched**. Cause: my RAM-gated
  retry saw ≥ 6 GB available and started the cache beside A8. That was my error, and the gate now
  also waits for A8.
- **Resume trigger** (agent-free waiter `../2026-09-19-navhard-download/code/navhard_after_a8.sh`,
  running): polls every 5 min for `C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run/summary.json`
  and starts **nothing** before it exists. It then runs `run_navhard_c1.sh`: **one worker**, NavSim
  output **off D:** (`C:/Users/Admin/navsim-crun/exp`; the old `C:/Users/Admin/navsim/exp` was D:
  through a junction), and only when ≥ 6,000 MB is available. Steps: cache → CV two-stage → human
  stage-1. **ETA ≈ 04:30–05:00 local**, CPU only, beside A7. To make it wait for A7 as well, point
  `A8_DONE` at A7's marker, or tell me.

## 9. PI commission, 2026-09-19 afternoon: the one-command eval suite — SIX work packages RUNNING

PI in chat: *"build what is missing for a one-command suite, NavSim v1, nuScenes, and a leaderboard
that regenerates itself … add high quality visualizations and reporting"*, then *"start now and take
the risk"* (on the weekly-usage risk, 94 % at the time). Plan: `../2026-09-19-eval-suite-build/BUILD_PLAN.md`.
W1 suite core + NAVSIM v2 · W2 estimator + gates (E3 resumed) · W3 NAVSIM v1 navtest (**D: only**,
PI) · W4 self-regenerating leaderboard (E4 resumed) · W5 visual reporting · W6 nuScenes (E5 resumed).
PI decisions: **v1 data on D: only**; **our-model navhard inference waits for a GPU gap**; **no
submission without PI approval** (the suite refuses it).
**Load on the training box:** every package is told CPU only, never to touch A8/A7, and to run HEAVY
compute (metric caches, full scoring, frame banks) **only after A8's `run/summary.json` exists, with ONE
worker**. The navtest camera download (127.9 GB, sha256-verified per shard) is writing to D: at
~12.8 MB/s. If A8 or A7 slows, tell me and I will throttle it.
