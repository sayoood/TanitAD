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

## 10. THE EVAL SUITE IS BUILT — W1, W2, W4, W5, W6 COMPLETE (2026-09-20); W3 + E1 still running

**One command:** `python -m taniteval.bench <benchmark> --ckpt X --split Y` · `python -m taniteval.leaderboard build`
· `python -m taniteval.benchreport <run_dir>`. All staged, nothing committed.

| pkg | state | the number that proves it |
|---|---|---|
| **W1** suite core + NavSim v2 | ✅ COMPLETE, code-frozen | acceptance on frozen code, 289 s CPU: CV **0.1853562745165113** (= HF warmup LB 18.5356) and STOP **0.3009023137456225**, both CSVs **byte-identical** to E1/E2's banked runs (md5 equal; 4,237 numeric cells, max |Δ| 0.0); 129 tests; `internal_t1` end to end in 8.4 s, 0 GPU |
| **W2** estimator + gates | ✅ COMPLETE | log-cluster bootstrap reproduces the devkit's own `extended_pdm_score_combined` on E1's real run **exactly** (|Δ| 0.0) and matches the devkit's mapping on duplicated-log fixtures 12/12; `criteria_check` evaluated **NONE** of `benchmarks.navsim` before, now all 4 blocking gates + 5 criteria with **26 RED mutation arms**; registry **2.10.2**; 178 tests |
| **W4** self-regenerating leaderboard | ✅ COMPLETE | page `e1417dd8…`, 2,284 lines, rebuilt byte-identical under both deletion modes; **113 external rows**, 19/19 banked PDFs sha-clean; new `leaderboard readback` re-parses the rendered page with its own parser (109 cells, 0 errors, `--mutate` RED); 48 tests |
| **W5** visual reporting | ✅ COMPLETE | report re-parses its own HTML with a verifier sharing no renderer code: **916 numbers + 55 refusals, 0 errors**; failure gallery 8/8 (nuPlan map + agents via the devkit's own `navsim.visualization`, plans projected, round trip 2.8e-14 px); 69 tests |
| **W6** nuScenes | ✅ COMPLETE | harness in both published conventions, `claim_bearing:false` by API; **39 published rows** with harness commits pinned; 57 tests; ⛔ no number until the PI's registration |
| **W3** NAVSIM v1 navtest | ⏳ running | v1.1 runtime on the existing venv (asserts it loaded v1.1); frame bank bit-identical to E2's stitch; plugin validates; full-split cache in flight |
| **E1** navhard reference | ⏳ running | cache built (5,187 s, 1 worker); CV re-run through the suite after the aggregation crash |

### ⭐ Findings that outrank the deliverables

1. ⛔ **`tools/criteria_check.py` evaluated NONE of the NavSim gates** — the four "blocking" gates were paper-only and every NavSim artifact passed. Found independently by E1 and E2, fixed by W2.
2. ⛔ **The registry's `EPDMS_v2` omitted TLC while its own formula multiplied by it — and the OLD TEST PINNED THE DEFECT.** The `a check that shares the defect it checks` class, again.
3. ⭐ **The NavSim driving command is a ROUTE-LEVEL ORACLE** (W2, 4 probes): the command is a function of ego-pose-now + route + map (1,902/1,902 reproduced with OpenScene's own function), but the route IS the expert's driven path at roadblock granularity, and stage 2 COPIES command+route from the expert's frame (5,462/5,462 navhard). Not a trajectory leak. ⇒ no route-following claim from a command-conditioned row.
4. ⛔ **STOP beats CV on BOTH NavSim generations.** v2 warmup: STOP 0.3009 vs CV 0.1854 official two-stage. v1 smoke (20 tokens, NOT navtest): STOP 61.48 vs CV 37.62, human 92.12 — mechanism from source: navtest holds only scenes the human passes, so a stopped ego keeps NC=DAC=TTC=1 ⇒ **5/12 = 41.7 points free**. ⇒ **every NavSim row carries a STOP floor.**
5. ⛔ **Our warmup model arms have NO official two-stage EPDMS** (stage 1 was the devkit's CV stand-in ⇒ HYBRID; refused by name in both the report and the page). navhard is the fix; its stage-1 frames exist (900/900 sampled).
6. Corrected external facts: the 1024×256 3-camera stack is **Transfuser's**, not Drive-JEPA's; the "perception-free ladder" is **NOT camera-only** (LAW and World4Drive print `C & L`); PDM-Closed 51.3 is **pre-#151** (56.6 post); DrivoR 56.3 is +134k+TOAD; Drive-JEPA's 93.3 is its own checklist misquoting 93.7.
7. ⚠️ **Split nesting:** warmup ⊂ navhard (16/16), and **367 of navhard's 450** stage-1 tokens are navtest tokens ⇒ our navhard and navtest columns are not independent samples.

### PI decisions carried (chat, 2026-09-19/20)

NAVSIM v1 download **YES, D: only** (32/32 shards verified, 127,882,665,618 B) · our-model inference **waits for a GPU gap** (arbitrated: `--device cpu` explicit is allowed when no trainer is alive; with a trainer alive it needs a named override recorded in `bench_run.json`) · **no submission** without a named PI approval (the suite refuses) · PENDING with the PI: the 3 Ego-Status-MLP checkpoints (19.6 MB) and the memory-hungry sibling CPU smoke.

### Proposed RETRACTION_LOG entries (I did not edit that file)

* **W5 2026-09-20** — asserted `leaderboard/build.py:66` imports `taniteval.report`; **NOT REPRODUCED** (0 occurrences). ⭐ Class: *another stream's LIVE file observed CORRECTLY, then quoted later as a standing fact* — the file was rewritten at 09:42, after the 16:05 read (discriminator: the quoted docstring now has 0 hits repo-wide). **Remedy: a `file:line` into a sibling's in-flight code is PERISHABLE — re-read at write time, or cite it with its timestamp.**
* **W6 2026-09-20** — offered "15 of my rows share a (paper, table) with W4's and agree 15/15" as an INDEPENDENT control; W4's re-derivation: 15 share a PAPER, **5** share (library_key, table), **0** share (paper, page, system). ⇒ a CONSISTENCY check, not an independent re-read. (Relayed to the PI by me before the correction; corrected in the same session.)

### Operational findings for the programme

* ⛔ **The session scratchpad is NOT private between streams** — W2's `verify_staged.py` was overwritten by W6's file of the same name and its next "verification" silently verified the OTHER package's paths, printing success. ⇒ namespace scratch filenames; a verifier must print WHICH package and WHICH path list it checked.
* ⛔ **No module in a package on `sys.path` may share a stdlib name** — W5's `benchreport/html.py` shadowed stdlib `html` and silently killed the gallery.
* ⛔ **Published run dirs must be APPEND-ONLY** — `results/bench/navsim_v2/warmup_two_stage/` went 4 dirs → 2 under W4's build and a cited run was deleted; W4 now refuses to write when the tree moves.
* ⚠️ **Box contention:** a sibling CPU training smoke committed **33.9 GB** and drove available memory to **419 MB**, aborting W3's first cache pass.

## 11. ⛔⛔ FOR THE REGISTRY OWNER — our LATERAL family has been reporting an OFFSET, not an error (W2, 2026-09-20)

**Not a NavSim artefact.** The same reducer produces the LATERAL rows in the **refcv3 / refcv4b /
refcv5-v2 T1 panels and in `MODEL_REGISTRY.md`.** Three defects, all VERIFIED AT SOURCE, all fixed at
source without re-scoring anything banked.

1. ⛔ **`cross_mae_m` IS A LATERAL OFFSET, NOT A DISTANCE TO THE PATH.** `_seq_geometry` returns the
   ego-frame **y column** (`four_families.py:187`) differenced at matched time index (`:776`) — no
   projection, no arc-length matching, no rotation into the GT tangent frame. ⇒ **two arms whose plans
   have y ≈ 0 score identically however differently they drive.** W1 measured the consequence on the
   real warmup run: **CV and STOP both read 1.0658 m**, so a LATERAL row quoted alone reads as a tie
   between a moving arm and a parked one. W2 reproduced the mechanism from scratch: on a curving GT the
   stationary arm's `cross_mae_m` equals **the GT's own mean |y| to 5e-5** — a property of the human's
   path, not of the arm.
   ⭐ **The qualifier every existing LATERAL number needs, verbatim:** *"`cross_mae_m` is a lateral
   offset at matched time index, informative only while the along-track error is small; read it with the
   LONGITUDINAL family, and use `headline.pathgeom_crosstrack_m` (`lateral.py::frenet_dense`) when a
   distance-to-path is meant."* Every lateral block now emits `_cross_is`,
   `_along_mae_m_for_context` and `_projection_based_alternative` beside the number.
   ⛔ **Which banked rows get restated is the registry owner's call** — W2 re-scored nothing.
2. ⛔ **`None` was being read as ABSENT.** The lateral terms returned `null` when no step cleared
   `min_ds_m`; `criteria_check` read that as a missing key, i.e. a silent omission, on every stationary
   arm. Now an explicit `{status: UNAVAILABLE, reason, n, n_steps_total, min_ds_m}` — never null, and
   never `0.0`, which would read as *perfect lateral agreement from a car that never moved*. The defined
   branch is **byte-identical**, so no banked number moves.
3. ⛔⛔ **And the half nobody had probed: `yaw_rate_mae_degps` emitted a REACHABLE NaN.** A 2026-08-23
   comment noted the masking and left it. W2 probed reachability instead of arguing it: **MEASURED — a
   plan advancing on alternate steps gives `n_head` 24, `n_curv` 0, `yaw_rate_mae_degps` = `nan`.**
   ⚠️ **A NaN is worse than the null this task was about**: it is a `float`, so it passes every
   `isinstance(v, (int, float))` guard downstream — `summarize.py` would have written it into a
   published `metrics` block **as a number**. Guard is now `n_curv`; since `n_curv > 0 ⇒ n_head > 0`,
   every case that produced a real value produces the identical value.

**Proposed register rows (verbatim in W2's RESULT §D):** `D-FF-LATERAL-NULL` (SUPPORTED — fixed at
source, 13 tests incl. a null-mutation arm and the reachable-NaN arm) · `D-LAT-CROSS-IS-AN-OFFSET`
(SUPPORTED — source-verified + reproduced).
**Suites:** 571 passed / 1 skipped / 17 failed across every consumer of `four_families`; all 17 are
`tools/tests/test_release_gate.py`'s pre-existing RG-02 fixture failures, **reproduced identically from
HEAD blobs in an isolated `git archive` tree** (so: pre-existing, not caused here). The 1 skip is
"OpenCV not installed in this venv".

## 12. ⛔⛔ ACTION REQUIRED FROM YOU — A STAGED GATE IS IN FORCE FOR NOBODY

**THE ASK (two paths, through your guarded lander):** commit
`products/P7-TanitEval/CRITERIA_REGISTRY.json` (blob **c5f6bedad363**, v**2.10.3**) and
`tools/criteria_check.py` (blob **1975a0c1f7da**). Everything else of ours can wait for a normal
landing pass.

**Why it cannot wait.** The SETTLED NavSim estimator gate has lived in the working tree since
v2.10.0 — **staged and never committed**, because the operating standard tells agents to stage and
never commit. `git show HEAD:…CRITERIA_REGISTRY.json` still serves **2.9.0**, whose
`admissible_until_settled` reads *"the ONLY admissible interval is UNAVAILABLE"*.
⇒ **MEASURED consequence, today:** E1 built its navhard artifact against the registry **a fresh
process can READ**, and therefore declared `estimator.interval = UNAVAILABLE` **while holding the
real interval** (CV EPDMS ×100 **11.4816**, log-cluster CI **[8.25, 14.50]**, 76 clusters).
⭐ **A false refusal reads exactly like an honest one:** the checker would have filed the programme's
FIRST interval-bearing NavSim measurement as a WORK ITEM, and the number would have stayed
invisible. W2 caught it only by probing HEAD against the working tree instead of trusting the report
— and it did NOT loosen the gate: the checker now FAILS a declined interval that holds an admissible
one (naming the key to promote), with a control proving it fires on admissibility rather than on the
word "interval".
⚠️ **The general form deserves attention beyond this file: any guard that lives only in the index is
a guard nobody is running — and this entire eval suite was built today under exactly that rule.**
Proposed register row: `D-REGISTRY-STAGED-NOT-IN-FORCE` (W2, RESULT §A.9 carries the diagnosis).

⚠️ **Channel note:** four `SendMessage`s to you have now expired unapproved, including this ask. If
the approval friction persists, this file is the channel that works — but a commit still needs you.

## 13. ⭐⭐ THE NUMBERS — both NavSim reference columns COMPLETE, both leaderboards reproduced (E1, 2026-09-20)

| split | arm | EPDMS ×100 | against the official value |
|---|---|---|---|
| warmup_two_stage | CV, official two-stage runner | **18.535627** | HF warmup LB **18.5356** → Δ **+0.000027** |
| warmup_two_stage | human, **stage 1 only** | **95.1255** (filter off 87.2014) | ⛔ the two-stage human is **UNDEFINED** |
| **navhard_two_stage** | CV, official two-stage runner | **11.4816**, 95 % CI **[8.25, 14.50]** | HF navhard LB **11.4816** → **Δ 0.0000**; **19/19** published terms of [N2]v3 Tab. 2 reproduced exactly **under TRUNCATION** (9/19 under rounding) |
| **navhard_two_stage** | **STOP (our floor)** | **29.8532** | unpublished — ours |
| navhard_two_stage | human, **stage 1 only** | **93.4796** | — |

MEASURED · tier **T1-family** · stage-1 loop **OPEN**, stage-2 **UNRULED** · navhard **5,912/5,912 valid**,
76 log clusters (W2's settled estimator) — **the programme's first NavSim interval**; warmup can never
carry one (7 clusters < the RG-14 floor). Cross-run agreement: W1's checker reads
`identical_to_e1 = True`, 8/8 sub-metrics at 2 dp on n = 450.

⛔ **THE FINDING THAT OUTRANKS THE NUMBERS: on navhard, STOP (29.85) beats CV (11.48) by 2.60×.**
EPDMS multiplies its compliance terms, so a stopped ego holds DAC 0.9311 · DDC 1.0000 · NC 0.9967 ·
TLC 0.9978 · TTC 0.9978 while only EP (0.3409 vs CV's 0.7753) punishes it. ⇒ **"beats CV" is not
evidence of driving on this benchmark; the bar for any TanitAD arm is STOP.** It now leads the
navhard block of `LEADERBOARD.md`, computed from the run's own sub-metrics so the sentence cannot
drift from the numbers beneath it.

**Preconditions of the number existing** (both belong in the column, not a footnote): the IDM
degenerate-path devkit patch (166/5,462 stage-2 tokens otherwise abort the whole run; verified
effective 3/3 AND inert 4/4) and the Windows `MetricCacheLoader` separator patch.

**Reported as written rather than relabelled:** two pre-registered controls FAILED — warmup C7 (1 ULP
in 4 summary cells; all 220 per-token rows bit-identical) and navhard C5 (weight tolerance 2.46e-09
against a 1e-9 bar, while the summary identity holds at 6.6e-12). Tolerances were mis-specified at
larger n; neither was widened after the fact.

### Still open — two need the PI, one needs you

1. ⛔ **Commit the staged registry + checker** (§12). **Now TWICE-measured:** E1's false-refusal
   artifact, and E1's restructure DELETING three functions W1 had promoted (`short_row`,
   `navsim_gates`, `gate_mutations`) whose pre-edit version is **unrecoverable — not in HEAD, and the
   index already holds the new blob**. ⭐ *With nothing committed there is no history: an overwrite
   cannot be undone, and a fresh process reads a stale gate.* W1 kept the promotion pin honest
   (`ORPHANED_PROMOTIONS`, RED on restoration) instead of dropping the names, which would have gone
   green while the guarantee shrank.
2. **PI ruling needed on the stage-2 loop status** — the 3DGS re-rendered counterfactual start is
   stamped UNRULED by every artifact rather than guessed.
3. The warmup **C8 reference stays INHERITED**: two probes to read the official leaderboard failed
   (the `/leaderboard` endpoint returns HTTP 405; the Space renders its table dynamically). One
   successful read would make it PUB-LB.

## 14. ⭐⭐ NAVSIM v1 REPRODUCED TOO — and a stopped car scores 3× CV on navtest (W3, 2026-09-20)

Full split, all three arms **12,146 / 12,146 successful, 0 failed**; PDMS identity **max |Δ| = 0.0** on
every one of 3 × 12,146 rows; 0 `criteria_check` violations (registry 2.10.3); LONGITUDINAL /
LATERAL / TACTICAL OK with log-cluster CIs over **136 logs**, STRATEGIC UNAVAILABLE with its reason.

| arm | NC | DAC | TTC | C | EP | **PDMS** | CI95 (136 log clusters) |
|---|---|---|---|---|---|---|---|
| CV | 68.0183 | 57.8380 | 50.0329 | 100.0000 | 19.4370 | **20.6517** | [19.20, 22.22] |
| **STOP** | 97.3983 | 96.5256 | 96.4021 | 69.4385 | 30.9994 | **61.8202** | [60.70, 63.08] |
| HUMAN | 100.0000 | 100.0000 | 100.0000 | 99.9012 | 86.9629 | **94.5514** | [93.79, 95.23] |

⛔ **THE FLOOR FINDING NOW HOLDS ON BOTH GENERATIONS: on navtest a stopped car scores 61.82 — 3.0× CV's
20.65, and within 3.8 points of the published ego-status-MLP baseline (65.6).** Paired on identical
tokens **+41.17 [+39.50, +42.84]**, W/T/L **8,607 / 727 / 2,812**, leading in EVERY t0 speed band.
Mechanism MEASURED from the scorer: NC 97.40 · DAC 96.53 · TTC 96.40 make **41.67 points free (5/12)**;
the LQR's braking coast earns a median **5.41 m**; and on **1,006/12,146 (8.28 %)** tokens the best
compliant progress is ≤ 5 m so **EP ≡ 1 by rule**. ⇒ **CV is not a floor on navtest either.**

**Verdicts as pre-registered.** CV: five of five sub-scores REPRODUCED; the PDMS cell
**REPRODUCED_UNDER_TRUNCATION** vs the paper's 20.6 and **REPRODUCED_UNDER_ROUNDING** vs the
leaderboard's 20.6517. HUMAN: **CLOSE, not reproduced** (94.5514 vs 94.8) — NC/DAC/TTC/Comfort
reproduce exactly and the ENTIRE gap is **EP 86.9629 vs 87.5**, the only term normalised against a
COMPUTED quantity. ⚠️ **W3's own pre-registered STOP range [38, 60] was WRONG** (measured 61.82) and is
recorded as a failed prediction, not re-fitted; C6's NC/DAC thresholds also failed as written.

### ⛔ A retraction chain worth reading as one item — it was caught by checking the ARTIFACT, not the relay

1. **W1** banked navhard CV's external as *"HF leaderboard, 11.4, truncate, Δ 0.0000"* — an entry that
   **contradicted its own fields** (our 11.4816 vs 11.4 is Δ 0.0816). It had merged TWO artifacts. They
   never disagreed: the **paper** prints 11.4 at 1 dp (truncating our 11.4816) and the **leaderboard**
   prints 11.4816 at 4 dp (rounding it). ⭐ **RULE: state the artifact before the convention** —
   quoting one artifact's convention against the other's row manufactures a **FALSE AGREEMENT, worse
   than a mismatch because nothing looks wrong.** Pinned by `M-SELFREF`, which re-derives every external
   reference from the entry's own fields and needs no external data.
2. **The orchestrator (me)** then relayed W3's cell result as *"the convention is SETTLED — the paper
   truncates"* — to W4 and to the PI. **W4 checked that phrasing against W3's probe file** and found the
   opposite verdict standing: *"UNSETTLED … 3 ROUNDING / 1 TRUNCATION … no convention may be assumed"*.
3. **W3 adjudicated against its own RESULT.md** rather than defending it: the measurement settles the
   **CELL** (one 20.651651538606658 explains the paper's 20.6 — rounding would print 20.7 — and the
   leaderboard's 20.6517), **not the paper**; three cells still read as rounding, and those may record a
   different RUN rather than a different rule. What actually changed is the cell's **evidence class**,
   print-vs-INHERITED → print-vs-MEASURED, resting on a stated assumption (that the paper's CV row is
   our quantity) with its counterfactual named: had the paper's CV been 20.6499 both conventions print
   20.6 and the cell would say nothing.
⭐ **Class, in W3's words: *true but wrong for the reader* — a claim true of one cell, phrased as a claim
about the paper, propagated through a relay.** W4 has pinned it structurally: a test now FAILS if *"the
paper truncates"* appears as an assertion anywhere in the navtest block, admissible only inside the
quoted account of the relay itself.

## 15. FLEET at 2026-09-20 15:52 local — the GPU gap OPENED and the navhard model arm is running

**A8 wrote its done-marker 2026-09-19 21:10 local; GPU 1,018 / 8,188 MiB with no trainer.** That is
the condition `D-NAVHARD-GPU-GAP-1` named (*"wait for a gap in the gpu"*), so **W7 launched**:
refcv4b on `navhard_two_stage`, arms A1 + STOP + CV + ECHO, through W1's `device_gate` (backs off the
moment a trainer appears or GPU mem ≥ 1 GB). It is the programme's first model row on official EPDMS.

| stream | state |
|---|---|
| **W7 refcv4b @ navhard** | RUNNING (launched this turn) |
| **W3 navtest chain** (agent-free) | RUNNING — camera shards **32/32 verified**, metric cache 119 logs started / 74 marked done, frame bank shard 03 of 32, 6.0 GB |
| navhard floors | ✅ DONE — CV **0.11481608441648 == official exactly**, CI95 [0.0825, 0.1450] / 76 log clusters; STOP banked; 5,912/5,912 |
| navhard HUMAN stage 1 | ✅ DONE — EPDMS **0.934796**, 450/450 (NC/DAC/TLC/TTC/LK all 1.0; EP 0.84236 is the whole gap) |
| PI-gated | the commit · the stage-2 loop ruling · the 3 ego-MLP checkpoints · nuScenes registration |

⚠️ **RAM, not GPU, is this box's binding constraint** — banking ran the pool down to **5,020 MB** and the
navhard N1 chain step was killed at **2,634 MB against a 3,000 MB floor**. Two heavy streams is the
honest ceiling here; a third is how the guard-abort loop starts. Holding at two deliberately.

### Two corrections from this turn, both caught by an artifact rather than a status line

1. ⛔ **I misread `N1/N1_manifest.json` as the successful navhard CV. It says `ABORTED_RAM_GUARD`.**
   The chain's N1 has **never** succeeded; the CV number came from the **W1 suite run**
   (`20260920T082848Z-…-06e257`, 5,912/5,912 valid). The reference chain's own log still ends
   *"STOP at N1"*, which reads as *"navhard CV failed"* while the verified number sits beside it —
   ⭐ the same **status-disconnected-from-artifact** family the repo already documents. The number is
   sound; only my route to it was wrong.
2. ⛔ **I was one command from killing W3's live chain.** Two `python … code/after_a8_chain.py`
   processes looked like a redundant navhard retry; they are the **navtest** chain — *same script
   basename, different package*. ⭐ **The `pgrep -f` trap in a new costume: I matched a substring of a
   path instead of identifying the job.** Nothing was killed; the discriminator was the package
   directory in the cwd, not the process name.

### A third defect, found and already self-healing — worth the pattern, not an action

`S5_bank_s00` died **rc 1** at `build_navtest_frames.py:256` with `NameError: name 'cand' is not
defined` — a variable that moved into `complete_partial()` during a refactor while the manifest line
kept referencing it. ⛔ **The frames were already built** (`frames_s00.npy` on disk); it died writing
the manifest, so the shard produced no DONE marker — *the compute paid for, the output destroyed at
write time*, exactly the documented analysis-time-failure class. **W3 had already fixed it** (the line
now reads `len(complete) + len(partial)`, and `cand` survives only in a comment recording the defect),
which is why s01/s02 passed. ⭐ **And the chain repairs itself without help: line 172 recomputes its
work list from MISSING DONE MARKERS, not from a status** — so shard 00 is retried on the next pass,
against fixed code. Resume-by-artifact is what makes that true; a chain resuming from an exit code
would have skipped it forever.

## 16. ⭐ refcv4b on NAVSIM v1 navtest — FULL SPLIT: 58.9738, FAILS the STOP bar (W3, verified from the artifact)

`raw/BAR_W3_M1.json` (n = 12,146, `full_split: true`, `VERDICT: FAIL`), MEASURED, T1-family, loop OPEN:

| arm | PDMS | paired Δ vs A1's floor | separated |
|---|---|---|---|
| CV | 20.6517 | A1 − CV **+0.3832** ±0.0251 | ✅ both levels |
| STOP | **61.8202** | A1 − STOP **−0.0285** | log ✅ [−0.0497, −0.0077] · drive ✗ [−0.0567, +0.0029] ⇒ **not separated** under the SPEC §5 conjunction; `p_delta_gt0 = 0.0035` |
| **refcv4b A1** | **58.9738** | — | — |
| HUMAN | 94.5514 | — | — |

BAR-W3-M2 fails too (58.97 < the published Ego-Status MLP's **65.6**, PUBLISHED). Inference variance: 200 tokens
re-run a day later came back **bit-identical**. Training variance untested (one checkpoint) — no training claim.

**Mechanism, MEASURED:** the loss is **entirely at commanded turns** — under STRAIGHT (8,070 tokens, 2/3 of
navtest) A1 **beats** STOP 63.96 vs 62.97; LEFT/RIGHT score 47.24 / 52.03 and carry −3.51 of the −2.85 total.
3,828 A1 plans score **zero** (off-road 3,224, collision 903) vs STOP's 732; on the rest A1 scores **86.11 vs
64.58**. Failures are **sideways**: cross-track 5.4×, heading 3.8×, endpoint only 1.47×. Under STRAIGHT on curving
road it turns **0.149×** the human (60.8 % barely turn); at LEFT it turns the right AMOUNT (1.07×) in the wrong
PLACE, failing DAC 39.8 %. **Eliminated, each pre-registered:** jitter (smoother → 58.1502, worse), "vision isn't
helping" (frames-blind → 23.29, vision worth **+34.7**), the scoring path (human through the seam matches the
devkit to 6.7e-8), heading spin (≤ 6 % of off-road failures).

### ⛔ A relay correction — mine, to the PI, in chat

From W3's 200-token provisional I told the PI refcv4b's zeros came from *"a planner committing to long fast paths
through non-drivable area"* (median progress 32.07 m vs 18.81 m) and added my own gloss that it was *"a fixable
defect, not a capability ceiling."* **W3 then found it had read the REFERENCE planner's progress array, not
refcv4b's; the over-long-plans story is refuted.** The true mechanism is lateral — under-commitment to curves and
mis-placed turns. It reached the PI in chat only (grep: no file I wrote carries 32.07 or the phrase), and is
corrected there. ⭐ **Class: relaying a PROVISIONAL mechanism plus my own gloss.** W3 had labelled it a hypothesis;
the gloss was mine, and it is the part that made the claim read settled.

### ⛔ The bar can be gamed — needs a decision before the next model is scored

*"STOP at every commanded turn, refcv4b elsewhere"* scores **62.4813** and clears BAR-W3-M1. W3 refused it (chosen on
the test split; wins by exploiting PDMS's reward for not moving) and proposes a **no-stop-at-turns clause**. Adding
it for the NEXT model is forward design, not a moved goalpost — this model's FAIL stands as written.

### ⚠️ A reader trap in the verdict file, not in the result

`BAR_W3_M1.json`'s `paired_vs_floor.STOP` is `{delta −0.0285, ci95 0.021, separated false}` — which implies
[−0.0495, −0.0075], excluding zero, beside `separated: false`. The complete source
(`analysis_navtest.json → arms.A1.paired_interval.STOP`) carries `separated_log_name: true`, the drive-level
interval and `separated_scope: "log_name AND nuplan_drive (SPEC §5 conjunction)"`. **The projection dropped the three
fields that make `false` correct.** Anyone reading the verdict file alone could "fix" it to `true`. Fix in W3's
generator: carry those three fields through. Verdict unaffected either way (FAIL is on the point estimate).

## 17. navhard — the retry reported SUCCESS on a total failure; relaunched (W7 resumed 2026-09-21 ~09:10Z)

Attempt `…/20260920T201038Z-…-d3c2b2`: every arm **FAILED** — `E1_RAM_GUARD_ABORT` at **1,175 / 1,750 / 1,077 /
948 MB** (hard floor 2,000), CV killed at stage-two scenario **4,294 / 5,462 (79 %) at 23:21:57**, CV's own RSS
774 MB. The same killer as attempt 0 (STOP had reached 83 %). ⛔ **`retry_scoring.sh:73` tested
`[ -f summary.json ]`** — and the suite writes one for failures — so it logged SUCCESS 14 s after the kill, wrote
`SUCCESSFUL_RUN_DIR.txt` naming the all-FAILED run, and **exited with 5 of 6 attempts unused.** The same defect
W7 fixed in its finisher the day before, surviving in the sibling script. Its post-step then died on
`No module named taniteval.bench` (the namespace-shadow trap).

⭐ **My extraction waiter was right to refuse** — it was gated on FINISH_REPORT.md, which W7's corrected finisher
writes only on content, so the false SUCCESS never opened it. Luck of which file it watched, and noted as such:
⛔ **gate on the primary artifact's CONTENT, never on its existence nor on a derived report's.**

**Next lever handed to W7, largest measured effect first:** stop re-scoring the floors. CV + STOP on navhard are
already COMPLETE in `…-06e257` (5,912/5,912, CV == official exactly); re-scoring them is ~3 h of pure exposure to a
killer that has struck twice at ~80 % of the FIRST arm. Score only A1 + ECHO against the banked floors on
identical tokens, as an explicit, refuse-on-mismatch reuse (devkit sha, patch set, metric cache, token set by
value). Fallback: chunked scoring with a bit-identical merge control. Box quiet at 09:06Z (14.7 GB free).
