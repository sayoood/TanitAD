# HANDOVER — REF-A (refav1), REF-D and the flagship (v7F → v8F)

**From:** TanitAD Master Mind session · **written:** 2026-10-15 ~22:25 Berlin · **for:** a separate agent session that
continues these three research lines toward frontier results.
**Source of truth used:** the landed tip `460508ed2f5e120667859e637ac233c28a48d90b` of `agent/arch-inf-20260803`,
read through git, never through the G: working tree. `Project Steering/MODEL_REGISTRY.md` wins every conflict, and
where it is stale this package says so. Every number below carries an evidence class (MEASURED with an artifact
path · PUBLISHED with a library key · INHERITED · ESTIMATED · HYPOTHESIS) and a tier (T0 = teacher-forced WM
diagnostic, never driving · T1 = self-action, OPEN LOOP on logged frames · NAVSIM = open-loop plan scoring).

## How to use this package

| file | what it holds | read it when |
|---|---|---|
| `HANDOVER.md` (this file) | the synthesis: state, what changed the plan, the ranked plan, PI decisions, live-work constraints | first, fully |
| `A_REFA.md` | REF-A lineage and refav1 in depth: architecture, every eval since 09-02, the R1–R6 build that is not landed, option T, 3 experiments, 15 traps, file index | before touching refav1 |
| `B_REFD.md` | REF-D: what it is, what the code really does (no `forward()`), what is missing to train it, 3 steps | before touching REF-D |
| `C_FLAGSHIP.md` | the flagship line v1 → v7F, settled findings, the unlanded v7F build, and the **proposed v8F** with bars, stages, compute and gate | before touching the flagship |
| `D_PROGRAMME.md` | programme goals, every arm's status, open PI items, eval doctrine and bars, frontier numbers with library keys, data, compute, landing, process rules, 12 traps | before your first eval or commit |

Read `CLAUDE.md` at the repo root before acting. RULE ZERO there outranks everything here: a refutation is a waypoint.
When an arm fails its bar, name the next lever and run it.

---

## 1. The state in ten lines

1. **No TanitAD arm beats every trivial floor it is read against, at any tier** (M, `Benchmarks & Eval/LEADERBOARD.md`
   "5 October position"). G1 "a model that drives" is OPEN and needs T2 (closed loop), which is not provisioned.
2. **refav1 is finished as a zero-training lever.** Its one checkpoint (step 21,109, 182.5 M params, frozen DINOv3-L
   features + 3-rate WM + iCEM) emits an injected do-nothing plan on 282/282 windows. The best tuned planner
   (`loncomb3`) is separated WORSE than holding a0 at both inference seeds (+0.0896 / +0.1515 m ADE, T1,
   `…/Research/2026-09-27-refav1-fullgrid-loncomb3/RESULT.md`).
3. **The frozen refav1 trunk carries no trajectory information beyond 8 t0 kinematic features** under any readout
   (pre-registered, `…/Research/2026-09-27-refav1-trunk-probe/RESULT.md`). An MLP on those 8 features (**P10**) scores
   **2 s ADE 0.2229 m / 6 s 2.9257 m** on 2,399 windows / 141 episodes: the strongest "do nothing with vision" bar the
   programme has measured.
4. **REF-D has never been trained and cannot be today.** `refd.py` has no `forward()`; `act()` reaches 4.6 % of the
   172.5 M parameters it is said to ship; a strategic-OFF config crashes (MEASURED on tip, `B_REFD.md` §4).
5. **v7F has never been trained.** The PI's 2026-08-31 v7 training hold has no recorded release. All "v7 results" are
   ~19 M v7-tiny proxies; their only T1 table is VOID (decoded through a random-init readout).
6. **"v8F" does not exist in the repo.** `C_FLAGSHIP.md` §C.6 proposes a definition (a HYPOTHESIS for the PI).
7. **The built code for the PI's R1–R6 requirements on refav1 AND v7F is NOT landed.** Both builds are tested and banked
   only as source copies under `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/` (refav1: tip
   `refa_v1.py` is still blob `d504eb9b`; v7F: `train_v6_staged.py`, `v6.py`, `v6_chain.py` byte-identical to
   `c36b6ddd`). At the tip `--nav-cond` is accepted and then **not built**, and every v6 resume trains at a corrupted LR (F3b).
8. **No launch-gate profile exists for refav1, REF-D, v7F or v8F** (`stack/scripts/launch_gate.py` has
   refc/refcv6/refcv7/refcv8). The PI ruled 2026-09-26 that no training starts without a gate PASS.
9. **The driving results live on the REF-C chassis.** refcv7-50,400 beats its echo control at 6 s by −0.98 m ADE
   (separated, T1) but not at 2 s; on NAVSIM it beats STOP on navtest (PDMS 70.20 vs 61.82) and **loses to STOP on
   navhard** (EPDMS 0.1987 vs 0.2985). refcv8 (refcv7 + tactical conditioning) is training on Thor now.
10. **The registers are stale for these lines.** Neither MODEL_REGISTRY §2.4 nor GOALS_AND_CLAIMS records the 09-27
    refav1 full-grid FAIL-WORSE or the trunk-probe NEGATIVE; the PI's 09-27 R1–R6 directive and "close P3, solve the
    rest" exist only in the P4 package and auto-memory. Fixing that is your first cheap task (§5, step 0).

---

## 2. Findings that change the plan (cross-cutting)

| # | finding | evidence | consequence for you |
|---|---|---|---|
| X1 | **The bar moved: P10 (kinematic MLP) and damped floors beat the echo-type floors by a lot.** `damp50 − ha0_ext` ADE@6 s **−1.9025 [−2.3605, −1.4971]**; P10 6 s 2.93 vs `damp50` 3.45 vs `ha0_ext` 5.36 (refav1 grid) | M · T1 · A §4, C §C.4 S10–S11 | Every "beats doing nothing" claim (refav1, refcv7/8, LEADERBOARD) was read against an undamped floor. **No REF-C or flagship arm has ever been read against P10.** Any "vision helps" claim must beat P10 on the same windows |
| X2 | **World models in this programme are action-deaf.** 0 of 32 v7-tiny arms move > 0.1 % under action shuffle; refav1's WM varies along κ by a median 1.99e-06 | M · T0 · C §C.4 S3; A §4 (09-04) | Do not rebuild an imagination planner over an action-conditioned WM. Scene → action runs one way (E-DEC-48b): reactions belong in the planner. Use the WM as a scene-evolution learner, never as the selector |
| X3 | **Selection, not proposal, is the gap.** refcv7 turn windows: picked − oracle-117 = **2.0718 m [1.7426, 2.4477]** (T0 denominator); REFe: zeroed tokens are 4.15 of a 6.30 PDMS human gap, and a scorer-only fine-tune gave **+3.79 [+3.05, +4.54] PDMS** | M · C §C.7; D §2 | Scorer / p(lane) / drivable sub-score levers are the measured big ones on NAVSIM |
| X4 | **A separated CI from one seed is not a lever effect.** The bootstrap resamples episodes with models fixed; the REF-C tiny-rig same-seed replicate reads "separated" on 14.3 % of cells; stochastic planners add an inference-seed floor | CLAUDE.md `H-ESTIM-SEED-1`; A §10 trap 3 | Every arm claim needs a training-seed replicate, plus an inference-seed replicate if anything samples. The refav1 inference-seed floor is config-dependent (+0.0015 to +0.1035 m); the "≈0.30 m" in CLAUDE.md comes from `D-REFAV1-DRIVE-AB`, not `D-REFAV1-SEED-GOAL-MISMATCH` |
| X5 | **Echo detection.** v1.x lateral skill was an action echo (S-curve 0.9785 open loop → 0.0430 self-action, hold-action 0.0 %); refav1's route head is a pure nav echo | M · T1 · C §C.4 S1; A §4 | Every head gets an echo/shuffle control. A goal input must never carry the situation classifier's output (CLAUDE.md binding rule) |
| X6 | **Frozen DINOv3 at our resolution/scale is not enough on its own** (refav1 trunk probe), and the v7F DINOv3→ViT transplant loses separated scene and ego-speed content | M · probe/T0 · A §4, C §C.3.1 | Whether a **trained** trunk (or a trainable readout at full data) beats P10 is the open question that decides option T, REF-D's premise and v8F's trunk choice |

---

## 3. The three lines at a glance

| line | what exists | best measured result | single next lever (largest measured effect first) | blocked by |
|---|---|---|---|---|
| **refav1** | ckpt 21,109 (Thor + dev box, no HF copy); eval harness `taniteval/tools/refav1_arm.py`; R1/R3/R5/R6 + R1-VMAX builds banked, unlanded | `cl` ADE@2 s 0.5474 vs `ha0` 0.5316, floor wins (T1, 282 windows); trunk adds nothing over P10 | **E1 (A §9):** a trainable spatio-temporal readout over the frozen DINOv3 tokens at all 4,572 train clips, read against P10 refit on the same clips. It decides what option T must train | nothing for E1 (dev box, ~6 h encode). Option T: PI storage/compute (~303 GB fp8 features deleted from Thor, or on-the-fly encode) + PI D1–D5 |
| **REF-D** | `refd.py` (411 lines, 19 tests), design doc; no trainer/loader/forward | none | **Step 0:** PI/Master Mind decide whether REF-D stays in the dominance comparison. If yes: make the code honest (fix crash, add `forward()`, pin counts), then a DINOv3-proxy REF-D on the refav1 loader | the decision; 2–3 weeks of build before any GPU |
| **flagship v7F / v8F** | v7F code at tip (= `c36b6ddd`) + the unlanded R1–R6 merge, gate, chain, resume fix in `P4V/` | flagship v1 0.4271 m ADE@2 s (**T0 only**); every T1 read of the line is either an echo or VOID | **E1 (C §C.9):** read refcv7-50,400 (and refcv8 when done) against P10 / `damp50` / `kd_x` on eval139, 0 GPU. Then the v8F decision F1 vs F2 | PI: v8F definition, release of the 08-31 hold if F2, D1 (DINOv3 wrap), R4 mode, cost cap |

---

## 4. Recommended v8F (Master Mind recommendation; the PI decides)

`C_FLAGSHIP.md` §C.6 lays out two options. I recommend **F1**:

* **Build v8F on the REF-C chassis** (`refc_v3_train.py` / `refc_v3.py`, ResNet-101 at 416×1024, BEV lift, 10 cm map
  head, box head, 117 anchors, per-frame v9 tactical conditioning; ~100.6 M params). Warm-start it from refcv8 if refcv8
  clears its bars, otherwise from refcv7-50,400.
* **Graft the flagship thesis onto it:** the hierarchy (nav per frame + turn distance into tactical and operative,
  all tactical labels supervising the tactical layer, tactical conditioning the plan, one 6 s plan, strategic OFF),
  plus the flagship latent WM as an **auxiliary** scene-evolution learner on pooled BEV tokens (the `rdw8p30k`
  two-term recipe), never as a selector.
* **Change the two levers with the largest measured effect:** replace the echo-type residual prior `ha0_ext_pose` with
  a learned (P10-style) or damped prior; add a learned p(lane)/drivable scorer to the selector. Enforce the R1 speed cap
  on the **emitted** plan, not only on selection.
* **Bars to register before compute:** BAR-V8F-K (beat P10, `damp50`, `kd_x` at 0–6 s, separated, both inference
  seeds), inherited BAR-R8-E1/E3/NR, navhard EPDMS > max(STOP, CV, ECHO), navtest PDMS > refcv7 LEGAL 70.20, LEAD-VIOL
  ≤ echo, WM predictor beats `z_t` with a shuffled-action control. Full table: C §C.6.5.

**Why F1:** every driving gain the programme has measured is on that chassis (refcv7's −0.98 m at 6 s; REFe's NAVSIM
results use the same proposal + scorer pattern), it already has a working gate profile, battery and NAVSIM bridge,
and every flagship WM lever so far has been action-deaf. F2 (train v7F as built) needs the hold released, the P4
package landed, four gate classes closed and ~9 days of S-W on Thor (ESTIMATED) before the first admissible T1 read.
F2's research question (can a trained-trunk WM become action-sensitive?) stays on the tiny rig.

⚠️ Cost check before authorising: the WM auxiliary is ESTIMATED at +20–45 % step time (~10.7–13 s/step), which
exceeds the PI-approved 10.5 s/step cap. Either approve a new cap or use a cheaper target (EMA BEV tokens, k ≤ 2).

---

## 5. The plan, in order (Rule Zero: largest measured lever first; ≥5 streams in parallel)

**Step 0 — 0 GPU, day 1 (do these in parallel; none waits for the PI):**

1. **Sync the registers.** Add to MODEL_REGISTRY §2.4 and GOALS_AND_CLAIMS: the 09-27 full-grid FAIL-WORSE, the
   trunk-probe NEGATIVE, P10 as the programme bar; mark `ccos` delivered and refuted (REG:2095 still says IN PROGRESS);
   record the PI's 09-27 R1–R6 directive and "close P3, solve the rest" in GOALS_AND_CLAIMS, `PI_DECISION_QUEUE.md` and
   `V7_LAUNCH_GATE.md`; put a VOID banner on the v7-tiny T1 table (REG ~4724–4742). A §8 item 5, C §C.8 items 2–3.
2. **Land the banked builds, or archive them on purpose.** refav1 R1/R3/R5/R6 + R1-VMAX (14 files, A §5 B/C) and the
   v7F merge + gate + chain + resume fix (C §C.5). Land from `P4V/v7f_merge/` only (the R3 stream copy is CRLF). Use
   the per-path descendant check from CLAUDE.md's git section; the tip has moved since 09-27, so re-base each file
   on its current HEAD blob. Even if v7F never trains, land F3b (resume LR) and the `--nav-cond` mapping fix, because the
   tip silently drops nav.
3. **Battery arms for the new floors:** implement `kin_mlp` (P10), `damp50`, `kd_x` as `taniteval` arms
   (`…/2026-09-27-refav1-trunk-probe/code/probe_te.py::fit_mlp_kin`). Pin with a known-value test.
4. **REF-D step 0:** get a ruling (keep/drop). If kept, start B §9 step 1 (fix the strategic-OFF crash, pin parameter
   counts, correct the "ships 172 M / 32 GPUs / ~125 GB" claims).
5. **Pre-register** E1 (refcv7/8 vs P10) and refav1 E1 (frozen-encoder ceiling at full data), both outcomes committed.

**Step 1 — dev box / LAN, week 1:**

* **Flagship E1 (C §C.9):** fit P10 on TRAIN-window kinematics (poses only, MB, over LAN from Thor) and score
  refcv7-50,400 `os` against P10 / `damp50` / `kd_x` on eval139 S2/S6 with the banked dumps
  (`D:/refcv7_eval_kit/battery/step50400_a6text/dump_s{0,1}`). If refcv7 loses to P10 at 6 s, the v8F critical lever
  is the prior; if it wins, that margin is the programme's first measured "vision helps" quantity.
* **refav1 E1 (A §9):** stream-encode the 4,572 train clips with frozen DINOv3 (~6 h on the 4060, read-only pulls from
  Thor's v2ep sources), fit a trainable readout, read against P10 refit on the same clips, 2 training seeds.
* **refav1 E3 (A §9):** add a `kinmlp` R5 prior to the landed R5 build; identity check (zero-init head == P10, max |Δ| = 0).

**Step 2 — Thor after refcv8 finishes (≈ 2026-10-20 17:20 Berlin, ESTIMATED), tiny rung, each arm at seeds 0 and 1:**

* v8F E2: prior swap (`ha0_ext_pose` → learned/damped) on the refcv8 tiny ladder, read against the rung's own
  training-seed floor.
* v8F E3: WM auxiliary, plus a time-shuffled-target deliberate-regression arm that must NOT help.
* Each needs a gate profile (`PROFILES["v8f"]`, template `launch_gate.py:666`) before it runs.

**Step 3 — full runs (PI authorisation):** v8F full pass on Thor; refav1 option T if E1 says the encoder can stay
frozen (else the "REF-A v2" trainable-encoder direction); REF-D only after its build passes a gate.

You are done with a line only when its result clears its committed bar, or every remaining lever is blocked and named
with what would unblock it.

---

## 6. Decisions only the PI can make (bring each with a recommendation)

| # | decision | recommendation | source |
|---|---|---|---|
| P1 | v8F definition: F1 (REF-C chassis + WM auxiliary) vs F2 (v7F as built) | F1 (§4) | C §C.6.1 |
| P2 | v8F step-time cap if the WM auxiliary exceeds 10.5 s/step | approve only after the tiny rung shows the auxiliary helps beyond the seed floor | C §C.6.6 |
| P3 | refav1 option T storage/compute: ~303 GB fp8 features vs on-the-fly encode on Thor | run refav1 E1 first; it may change what option T trains | A §5 A |
| P4 | refav1 R1–R6 build questions D1–D5 (κ0 source at inference, cot-absence negatives on v7.2, SPEED_BAND as target while max speed is an input, default prior, max-speed inference source) | default prior = P10 (`kinmlp`) once it exists | A §5 B |
| P5 | REF-D: keep in the dominance comparison or retire | keep only if the PI wants the SimWAM-style comparator; it needs 2–3 weeks of build first | B §9 step 0 |
| P6 | v7F: release of the 08-31 hold, D1 (wrap HF DINOv3, evidence favours option A), D5 (parameter budget), R4 mode | only relevant under F2 | C §C.3, §C.5 |

---

## 7. Live work you must not disturb (as of 2026-10-15 22:25 Berlin)

* **refcv8-r101-s0 is training on Thor** (launched 2026-10-15 04:54 by the launch gate's PASS token on `fb47a58`;
  step 6,250 at 22:03; 9.74 s/step; ETA ≈ 2026-10-20 17:20). ⛔ Never add GPU/RAM load to Thor while it trains
  (GPU sharing there was 3.9× slower, MEASURED). Pulling small files read-only over LAN is fine.
* **refcv8 Training Watch** (private artifact, refreshed every 3 h by the Master Mind session):
  https://claude.ai/artifact/JzKCa8g4pxo2XnzC2vAGt4 — T0 in-run eval vs refcv7's final level; builder
  `taniteval/tools/training_watch/build_watch_refcv8.py` (landed `0da425a`).
* **refcv8 milestone battery:** the step-5,000 G0 check is queued on the dev-box GPU behind another session's NAVSIM
  job; package `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/` (not yet landed; it
  runs from `C:/Users/Admin/qland/hold_r8batt/`). Do not take the dev-box GPU lock from it.
* **Disk:** on 2026-10-15 the C: cleanup (PI-approved) deleted `C:/Users/Admin/refav1_trainprobe/` (the 600-clip
  TRAIN features of the trunk probe; regenerable with the banked `code/pull_encode.py`, 2,834 s for 600 clips) and
  ~45 stale repo-tree copies. D:/E: (one exFAT volume, 1 MiB clusters) was **full** at 22:20 Berlin; check free space
  before writing caches there and never write many small files to it.

---

## 8. Corrections this analysis found (fix them as you go)

* CLAUDE.md attributes the ≈0.30 m refav1 inference-seed floor to `D-REFAV1-SEED-GOAL-MISMATCH`; it comes from
  `D-REFAV1-DRIVE-AB` (+0.2956 on the unchanged windows of one panel) and the floor is config-dependent (A §10 trap 3).
* `…/2026-09-27-refav1-fullgrid-loncomb3/RESULT.md` §4.5 re-quotes the retracted 1.63e-10 (step-1,000 checkpoint;
  RETRACTION_LOG #21). At 21,109 the median is 1.99e-06 and the binding penalty is jerk.
* REG §2.4 says `ccos` "IN PROGRESS" (it was delivered and refuted on 09-05).
* `SPEC_REFCV8_DRAFT.md:65` says 4,369 train clips; REG §16 (4,572-clip view) wins.
* REG §1.7 v2corpus "RUNNING" is stale; REG ~4724–4742 lacks the VOID stamp GAC:8945 requires.
* `REFD_DESIGN.md` / Paper §12: "ships ~173 M", "32 GPUs", "~125 GB" are unsupported or stale (B §4, §9).
* `BACKLOG.md` head, `NEXT_ARMS.md` and `LOOP_STATE.md` are stale (last updated 09-26, 08-26 and 07-29).
* The REF-D bar in B §9 step 3 (hold-action 0.5316 m) predates X1; read REF-D against P10 and `damp50` as well.

---

## 9. Manifest

| artifact | where it lives |
|---|---|
| this package (5 files) | `repo: Project Steering/Handover/2026-10-15-refa-refd-flagship/` |
| refav1 checkpoint | Thor `/home/nvidia/experiments/refav1-b1-v72-ep3-speed/ckpt.pt`; dev box `C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt` (md5 `1189bc02…`) — **no HF copy** |
| refav1 + v7F R1–R6 builds (unlanded) | `repo: products/P4-training-pipelines/2026-09-27-v7f-refav1-state/` |
| refcv7-50,400 battery dumps (for flagship E1) | dev box `D:/refcv7_eval_kit/battery/step50400_a6text/` |
| refcv8 run | Thor `/home/nvidia/refcv8_run/runs/refcv8-r101-s0/` |
