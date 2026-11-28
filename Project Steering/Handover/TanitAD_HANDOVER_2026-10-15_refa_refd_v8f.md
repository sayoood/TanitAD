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


---

<!-- appendix: A_REFA.md -->

# Section A — the REF-A line (refav1) — handover

> **Source of truth for this section:** the landed tip `460508ed2f5e120667859e637ac233c28a48d90b` (2026-10-15 21:54 +0200), read with `git --git-dir=/c/Users/Admin/tanitad-push/.git show <tip>:<path>`. `Project Steering/MODEL_REGISTRY.md` wins over every other doc; where the registry is stale or silent this section says so. Abbreviations: **AI/** = `TanitAD Research Lab/Architecture & Inference/Research/`; **P4S/** = `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/`; **REG** = `Project Steering/MODEL_REGISTRY.md` (line numbers at tip); **GC** = `Project Steering/GOALS_AND_CLAIMS.md`.
> Evidence classes: **M** = MEASURED (artifact path given) · **P** = PUBLISHED · **I** = INHERITED (not re-verified by me) · **E** = ESTIMATED · **H** = HYPOTHESIS. Tiers: **T0** teacher-forced / WM diagnostic · **T1** self-action OPEN loop (PI ruling 2026-09-02: a planner feeding its own predictor is still open loop) · **probe** = representation readout, no tier. Every comparison is a **paired episode-cluster bootstrap, n_boot 2,000**, unless stated. ⛔ **Closed loop has never been run for refav1** (REG:1274). ⛔ **Training-seed variance has never been measured** for refav1 (one run).

---

## 1. Status in one paragraph

**refav1** (`refav1-b1-v72-ep3-speed`) is a 182,459,701-parameter latent world model on **frozen DINOv3 ViT-L/16 fp8 features** with a three-rate FiLM hierarchy and a **test-time iCEM planner**; its one complete run finished at **step 21,109 (one epoch) on 2026-09-04** and the line has had **no training since** (REG:2081-2095). Every zero-training lever has now been tried and **eliminated on this checkpoint**: the shipped planner emits an injected do-nothing plan on **282/282** full-grid windows (`cl − ha0` ADE **+0.0158 [+0.0007, +0.0315]**, separated WORSE; M, T1; REG:2094, 2129-2137); the best tuned configuration `loncomb3` **FAILS-WORSE on the full 141-episode grid at both inference seeds** (`A1 − ha0_ext` **+0.0896 [+0.0117, +0.1756]**; M, T1; `AI/2026-09-27-refav1-fullgrid-loncomb3/RESULT.md` §0) and a planner-free blend `damp50` beats it. ⭐ **The single most important fact:** the pre-registered trunk probe found that the **frozen refav1 trunk carries NO trajectory information beyond 8 t0 kinematic features** — under every readout tried (linear/MLP, global/4×10 spatial, 113 or 600 fitting episodes) — while an **MLP on those 8 kinematic features alone (`P10`) is the strongest predictor the programme has measured on this grid: 2 s ADE 0.2229 m, 6 s ADE 2.9257 m on 2,399 eval windows / 141 episodes** (M, T1-equivalent; `AI/2026-09-27-refav1-trunk-probe/RESULT.md` §3). ⇒ The only lever left on this line is **option T: retrain the trunk (adapter + world models) jointly with a trajectory loss** — blocked on a **PI storage/compute decision** (~303 GB fp8 train features, deleted from Thor, or on-the-fly encoding). The PI's R1–R6 code for it is **built and tested but NOT landed** at tip (§5, §6), and **neither the registry nor the claims register records the 2026-09-27 verdicts yet** (§8).

---

## 2. Architecture and recipe

| item | value | evidence |
|---|---|---|
| Encoder | **DINOv3 ViT-L/16**, frozen, d_enc 1024, **640 patch tokens (16×40) at 256×640 cylindrical, 120° HFOV**; CLS + 4 registers dropped; every 2nd frame encoded bf16 → **fp8 (e4m3) cache**; ImageNet mean/std at encode | `stack/tanitad/refs/refa_v1.py:75` (`DINOV3_GEOMETRY`); `stack/scripts/dinov3_fp8_encode_ship.py`; GC `D-REFAV1-CACHE-COMPLETE` (I) |
| Adapter | **WideAdapter, no bottleneck** (`d_state` 1024 ≥ `d_enc`), trained at `adapter_lr_mult 0.1` | `refa_v1.py:714-716`; ckpt `config.json` (M) |
| World models (3 rates, 6 s) | **operative** 0.2 s × 30 (6 layers, 8 heads, window 4) · **tactical** 0.6 s × 10 (64 queries, 4 layers) · **strategic** 3.0 s × 2 in-window + extension to 12 s on its own subspace predictor (str_dim 256); chained **strategic → FiLM → tactical → FiLM → operative**; objective = **predict future PATCH features (L2)**, `w_feat_op 1.0 / tac 0.5 / str 0.25` | `refa_v1.py:706-760`; ckpt `config.json` `cfg` (M) |
| Heads | `lat_head`, `lon_head` (tactical decision, v7.0 vocab), strategic `route_logits`; `w_tac_label 0.1`, `w_str_label 0.1`; **proposal/imitation head `w_aux_head 0.0` — never trained** | `refa_v1.py:1537-1539, 2060-2075`; `config.json` (M) |
| Actions / ego | `a_dim 2` = (a, steer); **`--speed-channel` ON** ⇒ predictor input (a, steer, v_k/30), v_k integrated from the measured v0 (never future GT); a missing v0 **raises** | REG:2093; GC `D-REFAV1-EP3-SPEED` |
| ⚠️ Action units | `actions[:,0]` is a **road-wheel STEER ANGLE** `arctan(L·κ)`, L = 2.9 — **not κ** (≈2.9× its magnitude); banked evals ran `--action-units kappa` (legacy) | REG:2170-2173; RETRACTION_LOG #15 |
| Nav | `nav_inject: True` — nav embedding added to the **tactical ctx** and to the **field predictors' intent**; strategic subspace predictor stays nav-free | `refa_v1.py:905, 1561-1564, 1658-1664` |
| Planner | **iCEM** 300 samples × 30 iters × 30 elites, coloured noise β 2.5, **horizon 10 × 0.2 s = 2.0 s** (`plan_horizon_s 2.0`) while the goal/cost spans 6 s; `plan_level 'tactical'`, `verify_on_operative`; injects `cv / hold_v0 / decel_1.5 …` baselines into its own population; `kappa_max 0.2`, `a_max 4.0` | `stack/tanitad/refs/refa_v1_plan.py:65-93`; `refa_v1.py:782` |
| Cost | goal term (`cos` default; `chord / ccos / ccosh` selectable) + `W_JERK 0.02` + `W_KAPPA 0.05` + `W_VEND 0.10`; turn token commands `GOAL_KAPPA_TURN 0.08` 1/m (R 12.5 m). ⚠️ `W_VEND` is **inert in every arm ever run**: `target_speed` is never passed to `plan()` | `refa_v1.py:119, 433-435`; `RESULT_HARVEST.md` §3 (M) |
| Params | **182,459,701** (checkpoint) | REG:2089 (M) |
| Training recipe (ep3) | `stack/scripts/refa_v1_train.py` · bs 8 · lr 3e-4 · adapter_lr_mult 0.1 · **21,109 steps = one epoch** · `--target-space frozen --detach-aux-targets --bptt-truncate 15 --ema-targets` (decay 0.996 → 0.999) `--precision bf16 --tf32 --skip-nonfinite --speed-channel` · lru 64 · seed 0 · clip 1.0 · torch 2.13.0+cu130 on Thor; **3.47 s/step at step 100 (≈ 20 h/epoch)** | dev-box `C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json` (md5 `bd3ebed7…`, read this session, M); GC `D-REFAV1-EP3-SPEED` (M there) |
| Corpus | **B1 / v7.2 labels: 168,910 windows over 4,572 episodes**, labels md5 `0ff90213…` (train) / `aa12c948…` (eval), join 4,572/4,572. ⛔ **NOT the parity corpus `physicalai-train-e438721ae894`** — allowed by PI ruling (GC:3325/5735, I) ⇒ refav1 numbers are not comparable with REF-A dinov2/dyn-in rows | REG:2092 (M) |
| ⚠️ Code provenance | trained from **Thor's older stack** (`refa_v1.py` 1,885 lines vs 3,030 at tip); a strict load into tip code works (`missing_keys: []`), but **a re-run from tip is NOT a replication**; the registry still does not record this (BACKLOG A13c, open) | `Project Steering/Reports/2026-09-15-1445-refcv6-refav1-v7-status.md:128` (I) |
| Checkpoint | `ckpt.pt` 2,122,997,633 B, md5 `1189bc020018c2c67ce03d566c390285`, keys `cfg/model/opt/step`, 407 tensors. Copies: **Thor** `/home/nvidia/experiments/refav1-b1-v72-ep3-speed/ckpt.pt` (+ `/home/nvidia/refav1_lon/ckpt/ckpt.pt`) and **dev box** `C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt` (size verified this session). **No HF copy** | REG:2084-2090; ccos-eval RESULT header (M) |
| Eval harness | `taniteval/tools/refav1_arm.py` (Thor invocation: `taniteval/tools/REFAV1_ARM.md` §6; dev-box invocation: `AI/2026-09-27-refav1-fullgrid-loncomb3/raw/run/run_fullgrid.sh`); paired deltas `taniteval/tools/refav1_paired_delta.py`. Arms: `cl` (planner, T1), `ha` (hold recorded action, T1), `ha0` (constant velocity, T1), `ha0_ext` (hold measured a0 + yaw-rate, T1), `ol` (recorded future actions, **T0**). Planner-free floors: `damp50 = 0.5·ha0 + 0.5·ha0_ext`; `kd_x` (fixed damped kinematic prior — ⚠️ defined only in the trunk-probe code, `AI/2026-09-27-refav1-trunk-probe/code/probe.py`, and the full-grid `raw/analysis/explore/` builders; **not in `stack/` or `taniteval/` at tip** — the unlanded R5 build carries it as `--r5-prior kdx`, and the 30-step and 10-step `kd_x` differ, so name which); `kd_ha0_ext`; `stack/tanitad/models/kinematic_prior.py` holds the refcv-line priors (`ha0_ext_pose`) | REFAV1_ARM.md; full-grid RESULT §0/§4 |
| Eval grids — ⛔ never mix them | **(a) full grid**: v7.2 EVAL split, 141 episodes, stride 40 ⇒ **282 windows**, K = 10 @ 0.2 s — the decision grid; ~2.1 h/arm on the 4060 (M: two arms 09:33–13:48 Berlin). **(b) p4 panel**: 8 turn-dense episodes, stride 16, **40 windows** — ⛔ *not admissible for lateral levers* (full-grid RESULT §4.2). **(c) trunk-probe grid**: the same 141 episodes, **2,399 windows**, 2 s and 6 s horizons | — |
| Data on the dev box (existence verified this session) | eval fp8 cache `C:/Users/Admin/tanitad-data/refav1-eval141/refav1-fp8-eval` (142 entries), episodes `…/refav1-eval141/eps`, labels `C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz`, probe workspace `C:/Users/Admin/refav1_probe/`, full-grid outputs `C:/Users/Admin/refav1_fullgrid/`, R1-VMAX dev copy `C:/Users/Admin/refav1_dev/`, p4 dumps `C:/Users/Admin/refav1_margin/p4out/`. ⚠️ `C:/Users/Admin/refav1_trainprobe/` (600-clip TRAIN features) is **GONE** — regenerable with the banked `code/pull_encode.py` (M: 2,834 s for 600 clips) | — |
| Thor-side | train fp8 cache `/home/nvidia/data/refav1-fp8-train` ⛔ **deleted** (~303 GB to rebuild); eval cache `/home/nvidia/data/refav1-fp8-eval`; v2ep source frames (4,713 episodes) still on Thor | full-grid RESULT §5; `P4S/REFAV1_RETRAIN_DECISION.md` §4 (I) |

---

## 3. Lineage

| run | encoder / key change | status | headline (class, tier, estimator) | verdict |
|---|---|---|---|---|
| `refa-phase0-30k` | frozen DINOv2-B/14, grid adapter, no speed | superseded | ADE@2s 3.726 (heldout split-mean — ⛔ deprecated estimator) | pre-fix baseline; voided by D-A3 |
| `refa-plus-speed-30k` | + `v0` as 3rd action channel | archived | trainer fwd-ADE 3.73 → 0.83, speed-R² 0.61 → 0.965 (trainer metric, not eval) | the isolated proof of the speed fix (D-A3) |
| `refa-4brain-speed-30k` = **REF-A dinov2-4b** | + `--four-brain` | ✅ accepted as reference | ADE@2s **2.1675 [1.9081, 2.4212]** full-set (M, T0, episode-cluster bootstrap; REG:3760); does not beat CV | the frozen-encoder reference (REG §2.1) |
| `refa-ijepa-4brain-speed-15k` | I-JEPA ViT-H/14, 320 eps | diagnostic only | ⛔ canonical val **80 % leaked** — unusable | REG §2.2, R8 |
| `refa-dynin-4brain-30k` = **REF-A dyn-in** | + `[v0, yr0]`, ego-dropout 0.25 | ✅ complete 29,999 | ADE@2s **3.0471 [2.4984, 3.6878]** full-set (M, T0); flagship paired Δ **+2.62 [+2.0945, +3.2570]** | **H4 closed NEGATIVE** (D-A5): capability ceiling, 94.2 % longitudinal error |
| `refav1-b1-v72-1ep-21109` | **new design**: DINOv3-L, WideAdapter, 3 WMs + iCEM | retired at step 1,000 | collapsed ~step 450 (I); relaunched with frozen targets; step-1,000 T1 read = CV straight line on 140/140 (M, 20-clip slice) | ⚠️ the directory name is not its step |
| `refav1-b1-v72-ep2-ema-bf16` | + EMA teacher, bf16/TF32 (6.1× faster) | retired at 6,850 | speed never reached the model (`act.0.weight` [1024, 2]) (M) | GC `D-REFAV1-EP3-SPEED` |
| **`refav1-b1-v72-ep3-speed`** | + `--speed-channel` | ✅ **complete 21,109** (2026-09-04) | `cl` ADE **0.5474 [0.4839, 0.6108]** vs `ha0` 0.5316, `ha` 0.5391, `ha0_ext` 0.5207, `ol` (T0) 0.4237 (M, T1, full grid; REG:2094; ccos-eval RESULT §3a) | **planner = injected baseline 282/282; does not beat doing nothing** |
| "REF-A v2" | trainable DINOv3-distilled encoder | ⛔ **never commissioned, no SPEC** | — | a recommendation only (`products/P4-training-pipelines/V7_FULLSCALE_RECOMMENDATION.md:344-350`, 2026-08-24). ⚠️ In that doc "REF-A v1" means the OLD DINOv2 arm, not refav1 |
| **option T** (refav1 trunk + R5 head retrained jointly) | trajectory loss as a first-class objective | ⏸ **recommended, not launched** | — | §5 item A |

---

## 4. What was measured on refav1 (chronology, 2026-09-02 → 09-27)

All on checkpoint 21,109 unless stated. "Eliminated" = a candidate ruled out; "named" = the next lever the package recorded.

| date · package | question | result (class · tier) | eliminated | named next |
|---|---|---|---|---|
| 09-02 `AI/2026-09-02-refav1-target-space-collapse/` | does the shipped anti-collapse config stop the collapse? | CONFIRMED on every pre-committed read (M, T0, 250 steps/arm, Thor); ⚠️ licenses the *configuration* (target_space + detach), not "frozen targets" alone | — | relaunch (done) |
| 09-02 `…-step-profile/` | where does the step go? | the 30-step operative rollout is ~83 % (bs 1, M) / ~92 % (bs 8, E) of the step; compute-bound; **bf16 3.0× faster** (M); the full step does **not fit an 8 GB card at any batch or precision** | launch/loader overhead | bf16 (adopted) |
| 09-02 `…-ema-inflation/` | does an EMA teacher pin the tactical scale? | **VOID** — dev-box rig could not reproduce the inflation (M, T0) | — | ran on Thor in ep2 |
| 09-03 `…-step1000-read/` | first T1 read (step 1,000, 20 clips) | plan = constant-velocity straight line on **140/140** (M, T1) | — | full-epoch read |
| 09-03 `…-anchored-actdiv-refav1/` | is the WM lateral-insensitive? | `H-REFAV1-LAT-INSENSITIVE` **REFUTED** (M, T0, CPU) | "the WM ignores steering" | cost surface |
| 09-03 `…-refav1-cost-surface/` | decompose the planner cost | goal term negligible against the penalties (M, T0) — ⚠️ its 1.63e-10 figure is the **step-1,000** checkpoint's (RETRACTION_LOG #21) | — | cost scale |
| 09-04 `taniteval/results/RESULT-refav1-21109-openloop.md` | the epoch-end read, full grid | injected baseline **282/282**, κ ≡ 0 on 282/282, **2 distinct plans** in total. Four families, `cl − ha0`: ADE +0.0158 sep. worse; LON speed **+0.0308 [+0.0049, +0.0585]** worse, along **+0.0258** worse; LAT exactly 0.0000 [0,0] ×3; TAC lat 0.0000, TAC lon −0.0035 n.s. `cl − ha`: LON speed **+0.2706** worse; LAT cross **−0.1569** better (a straight line beats a drifting held κ); TAC lon **−0.1277** worse. **STRATEGIC: route head = pure nav echo** (1.0000 true nav / κ −0.1427 shuffled / majority class with nav withheld). **What training DID buy:** WM beats persist-last-field from 1 s (+0.1129) to 6 s (+0.2454), loses at 0.2 s (−0.0442) (M, T0); LON tactical head **0.4326 [0.3546, 0.5177]** vs 0.2979 majority floor under `nav_zero` (M); tactical decoder asks for turns on 13.5 % of windows (M). Distance-keeping: `cl` ≈ `ha0` (n = 87, I via survey) | "early in training" | cost scale |
| 09-04 `AI/2026-09-04-refav1-cost-scale/` | the cost, in units | WM varies along κ by a **median 1.99e-06** (n = 282); float32 ties the κ argmin on 45.4 % of windows; **the binding penalty is JERK** (99.5 % of the median 0.362 penalty) (M; REG:2141-2150) | re-weighting alone | ccos |
| 09-05 `…-refav1-ccos-eval/` | centred cosine on the real code | `ccos` naive ADE **0.9268 [0.7808, 1.0819]**, weight-compensated **0.7116 [0.6291, 0.7973]** vs `ha0_ext` 0.5207 — **worse** (M, T1, full grid; per-arm CIs). ⚠️ REG:2095 still says "IN PROGRESS" — stale | `ccos` alone | make-it-drive |
| 09-05 `…-make-it-drive/` | why does it not turn? | **two gates in series**: the goal head decodes LANE_KEEP on 86.5 % ⇒ κ = 0 on 244/244; under `cos` a correctly decoded turn is refused on **38/38** (M, T1). Lever `--lat-logit-bias` A/B: ADE **+0.8927 worse**, 11.2× its seed replicate (M, T1, `AB_RESULT.md`) | decision rule; "must retrain the WM" | vocabulary |
| 09-05 `…-goal-margin/` | is it the margin? | **it is the VOCABULARY**: only κ ∈ {0, ±0.08} can be sustained while the corpus curves at R 100–1000 m; LANE_KEEP is vocabulary-optimal on 90.6 % of GT-turn windows; the head detects turns (AUC 0.88) and L/R (AUC 0.90) (M) | head re-fit (nav echo / shrink-to-zero) | `--goal-kappa-turn` |
| 09-05 `…-vocab-l3/` | L=3 curvature vocabulary | panel **un-runnable as pre-registered** (deficit inside 6.1× seed noise, 10 windows); vocab implemented and pinned; **the trained head cannot choose among 3 magnitudes** (M) | — | a chooser (never built) |
| 09-05 `…-surface-sweep/` | 59-setting cost screen | `W_JERK` inert on 24/26 candidates; decoded goal directionally right 0.778 [0.625, 0.923] but commits to a turn on only 0.205 of GT-turn windows (M, T1) | most weights | cost geometry |
| 09-05 `…-cost-geometry/` (p4 panel) | the cost's geometry | shipped config = `ha0` **bit for bit on 40/40**; `W_KAPPA` interior optimum **15.11245**: ADE 0.8934 vs `ha0_ext` 0.8772 — parity (`cl − ha0_ext` +0.0162 [−0.1648, +0.1980]); beats `ha0_ext` on turns, loses on straights; **leaves the μ = 0.7 friction circle on 29.6 % of windows at v0 ≥ 2 m/s, peak 3.262 g** — `--kamm-mu 0.7` fixes that at no measured cost; `ccosh` and the seed ladder are **nulls** (M, T1; REG:2096) | candidate set; hold branch | longitudinal |
| 09-05 `…-longitudinal/`, `…-lonshift-t1/` (p4) | LON levers | `a0_shift` cuts LON speed MAE **−0.2283 [−0.3227, −0.1420]** (Thor) / −0.2263 (dev box), 12× its seed floor, replicated on 2 GPUs (M, T1, 8 clusters); `lonshift` vs `ha0_ext` ADE −0.0889 [−0.2049, +0.0137] (parity), FDE separated better, LON still worse | — | full-grid confirmation |
| 09-05/06 `…-close-the-gaps/` | why still under-turning? | the operative cost was **goal-blind** ⇒ a **3.9× under-turn** (0.02066 vs commanded 0.08) (M, T1, p4); goal-conditioned lateral cost (`--w-kappa-by-goal`) implemented, 15 tests; pre-registered `kammshift` **FAILS** (ADE 1.0652 > bar 0.8892; M, `…/raw/refav1_margin_banked_2026-09-27/RESULT_HARVEST.md` §1). ⚠️ the A3/A4 goal-conditioned arms ran on Thor and were **never read back** (UNVERIFIED; GC rows still say RUNNING) | — | — |
| 09-06 `…-distance-keeping/` | can it keep distance? | DK cost term fires on 21/90 lead windows, re-ranks 21/21; moved headway/TTC the right way on an **n = 4 window** smoke (M, T1, plumbing only); the gap is an ORACLE label | — | decoded gap; DK sweep (never run) |
| 09-06 `…-perception-probe/` | is the lead in the latent? | lead gap decodable **R² 0.3632** vs pixels −0.0513 (paired +0.4145 [+0.2018, +0.6120]); **closing rate absent** (every arm's CI spans 0) (M, probe, 42 clusters) | "a perception defect" for the gap | closing-rate representation |
| 09-06 `…-decoded-gap/` | vision-only gap head for DK | **BAR-2 FAILED** 12/21 = 0.5714 vs a committed 0.80 (16/21 after a post-hoc bias correction — still a miss); BAR-3 passes (M) | — | — |
| 09-27 `RESULT_HARVEST.md` (p4) | read the 17 stranded arms | `loncomb3` (= `ccos`, weights 0.02/15.11245/64.297, `--a-sustain-mode a0_shift --jerk-seam a0`) is the best p4 point estimate: ADE 0.7739, LON speed 0.4665; seed floors config-dependent (+0.0015 to +0.1035) (M, T1, 8 clusters) | — | full grid |
| ⭐ 09-27 `AI/2026-09-27-refav1-fullgrid-loncomb3/` | does `loncomb3` beat doing nothing on 141 episodes? | **FAIL-WORSE** at both seeds (`A1/A2 − ha0_ext` **+0.0896 / +0.1515**, separated); `damp50 − ha0_ext` **−0.1576 [−0.2297, −0.0990]**; `b50(A1) − damp50` **+0.0505 [+0.0120, +0.0918]** worse; the inference-seed replicate is itself separated, **+0.0619 [+0.0094, +0.1169]**. Four families vs `damp50`: tied on all LON; loses all LAT (cross **+0.3109**, heading **+1.87°**, yaw-rate **+0.0460**); TAC lat **−0.0780**; **STRATEGIC UNAVAILABLE** (no route label on the planner surface, n = 0). Exploratory: `kd_ha0_ext − ha0_ext` **−0.2282 [−0.2946, −0.1702]**, a stronger planner-free floor (M, T1) | **every zero-training planner lever on 21,109** | a trained trajectory readout |
| ⭐ 09-27 `AI/2026-09-27-refav1-trunk-probe/` | does the frozen trunk carry trajectory info beyond t0 kinematics? | **NEGATIVE** under every readout: `P4 trunk − P2 kin` +0.0042 [+0.0016, +0.0071] (2 s); A3 `P9 trunk_sp_mlp − P10 kin_mlp` **+0.0054 [+0.0016, +0.0093] @ 2 s, +0.0987 [+0.0332, +0.1658] @ 6 s** (worse); validity controls OK. ⭐ `P10` (MLP on 8 kinematic features, fitted on 600 TRAIN episodes): **2 s 0.2229 / 6 s 2.9257 m**; vs the linear readout at 6 s **−0.5024 [−0.6331, −0.3767]**; vs `ha0_ext` at 2 s ADE **−0.3495**, FDE −0.9272, heading −2.92° (worse only on accel MAE +0.104); at 6 s, point estimates `damp50` 3.45, `kd_x` 3.56, `ha0_ext` 5.36 (M, T1-equiv., 2,399 windows / 141 clusters). The only stratum where the trunk helps: 2 s `stop` −0.0690 on 14 windows (reported, not read) | **option H (heads-only)** | **option T** |

PUBLISHED context for the trunk verdict (banked in the Library and cited by the RESULT): an ego-state-only MLP matches perception-based planners on open-loop nuScenes L2 — Zhai et al. 2023 (`2305.10430`); E2E planners given ego status come to rely on it — Li et al., CVPR 2024 (`2312.03031`). ⇒ vision has to earn its place where kinematics fail: the 6 s horizon and event windows (turn onsets, stops, lead braking), not the pooled 2 s mean.

---

## 5. Planned and NOT finished

| # | item | spec / where | how far it got (verified at tip) | blocked by |
|---|---|---|---|---|
| **A** | ⭐ **Option T — retrain the refav1 trunk (adapter + WMs) with the trajectory loss as a first-class objective; R5 head trained jointly; prior = the learned kinematic MLP** | `P4S/REFAV1_RETRAIN_DECISION.md` (FINAL, 2026-09-27); bars committed in its §3 | **Recommendation only.** No run, no PREREG file of its own, no launch-gate profile. ⚠️ The R5 prior the build offers is `{kdx, damp50, cv}`, **not** the P10 MLP the decision names (`P4S/refav1_r1r6/RESULT.md` §0) | **PI**: ~303 GB fp8 train features placed (D: 92 GB free, C: ~70 GB, Thor 96 GB — none fits; I) **or** train on Thor with DINOv3 encoded on the fly from the v2ep cache (per-step cost **UNMEASURED**); a compute slot (Thor runs refcv8 at time of writing — I, from the session's memory index, not re-verified); items B, C, F |
| **B** | **R1/R3/R5/R6 build** — `--strategic-off`, `--vmax-input`, `--w-goal --goal-negatives --cot-negative-sidecar`, `--w-speed-band`, `--w-traj --r5-prior --r5-kappa-source`, refav1 G-DVB registry (48 dests) | `P4S/refav1_r1r6/RESULT.md` (+ landing table §9.1, 11 files) | **Built, tested (511 passed / 10 skipped on its set; 0 new failures), bit-identical when off (6,088 tensors + 310 values) — and NOT LANDED.** Verified this session: `HEAD:stack/tanitad/refs/refa_v1.py` is still the base blob `d504eb9b…` (the build's is `6e0988aa…`); `refav1_traj.py`, `train/declared_vs_built_refav1.py` and all 6 new tests are ABSENT at tip. The sources are banked under `P4S/refav1_r1r6/code/fix/…` | Master Mind landing; PI decisions **D1** κ0 source at inference (`steer_t0` is "unruled at inference" per SPEC_REFCV7 §10) · **D2** cot-absence negatives on v7.2 (17/22 vs 21/22 goal classes) · **D3** SPEED_BAND as target while max speed is an input (partial echo) · **D4** prior default · **D5** max-speed inference source (oracle label today) |
| **C** | **R1-VMAX** — inference hard cap `PlanConfig.v_max` + `refav1_arm.py --vmax-from-labels` | `P4S/LANDING_READY.txt` batch R1-VMAX; sources `P4S/code/fix/{stack/tanitad/refs/refa_v1_plan.py, taniteval/tools/refav1_arm.py, stack/tests/test_refav1_vmax_cap.py}` | **NOT LANDED** (tip `refa_v1_plan.py` blob `ba3a4a67…` vs banked `c46ac16b…`; `test_refav1_vmax_cap.py` absent; 0 `v_max`/`vmax` hits in the model/planner/loader/trainer at tip) | landing; then run C′ |
| **C′** | **`A1cap` — R1 full-grid engineering check** (282 windows; `n_plan_over_limit` must be 0) | `P4S/R1_FULLGRID_CHECK.md` (pre-registered 2026-09-27) | **No result at tip** (no hits for `A1cap` beyond the SPEC) — UNVERIFIED whether it ran on the dev box | item C landed; ~2.1 h on the 4060 |
| **D** | Option H — heads-only retrain on the frozen trunk | `REFAV1_RETRAIN_DECISION.md` §2 | **WITHDRAWN** by the trunk probe's committed NEGATIVE branch | — |
| **E** | **First R5 arm** (trajectory head) — needs its own pre-registration: bars vs the chosen prior AND the other two floors at 2 s and 6 s, four families, a **training-seed** replicate (the head is deterministic) | `P4S/refav1_r1r6/RESULT.md` §8.1 item 1 | **No PREREG exists** | B landed |
| **F** | **refav1 launch-gate profile** (binding since 2026-09-26) | `refav1_r1r6/RESULT.md` §7 (proposal); 40-flag inventory in `P4S/raw/` | `stack/scripts/launch_gate.py:432-438` has profiles `refc`, `refcv6`, `refcv7` only — **no refav1** | Master Mind |
| **G** | **Eval harness wiring** for the new levers: an R5 arm calling `RefAV1.trajectory()`, `speed_max_ms` into `plan()`, pose-past κ0 | `refav1_r1r6/RESULT.md` §8.1 item 2 | not started | B, C |
| **H** | **Distance-keeping weight sweep** with inference-seed replicates on the full 141 panel | `AI/2026-09-06-refav1-distance-keeping/RESULT.md` §0 | queued, never run; the DK planner hook is **missing at tip** (`dk_spec` 0 hits in `refa_v1.py`/`refa_v1_plan.py` vs 19 in `refav1_arm.py`) | the hook; the planner is eliminated as a lever anyway (low priority) |
| **I** | Goal-conditioned lateral cost arms A3/A4 (Thor) | `AI/2026-09-05-refav1-close-the-gaps/` | ran, **never read back** (UNVERIFIED) | Thor access; superseded by the full-grid elimination |
| **J** | "REF-A v2" (trainable DINOv3-distilled encoder) | `V7_FULLSCALE_RECOMMENDATION.md:344-350` | never commissioned | would become the fallback if option T's frozen-encoder premise fails (§9 E1) |
| **K** | Closed-loop (AlpaSim/NuRec) read of refav1 | REG:1274 | never run | a policy worth driving |
| **L** | Register integration of the 09-27 verdicts + A13c code provenance | BACKLOG A13c; `P4S/HANDOFF_TO_MASTER_MIND.md` §2/§4 | **not done** — see §8 | Master Mind |

---

## 6. PI requirements R1–R6 (binding, 2026-09-27) — compliance AT TIP

⚠️ The build in §5 B/C would change R1/R3/R5/R6 to "built, default OFF" — **but none of it is in the tip**, so the tip state is:

| req | status at tip | file:line (tip `460508ed`) |
|---|---|---|
| **R1** max speed = input AND hard cap | ⛔ **ABSENT** | 0 hits for `v_max|speed_max|vmax|max_speed` in `stack/tanitad/refs/refa_v1.py`, `refa_v1_plan.py`, `stack/tanitad/data/refav1_loader.py`, `stack/scripts/refa_v1_train.py`; the 3 hits in `taniteval/tools/refav1_arm.py:1691,1735,1783` are a `speed_check` diagnostic, not a cap |
| **R2** nav → tactical + operative, train + inference | ✅ **MET** (as oracle nav) | `refa_v1.py:905` (`nav_inject: True`), `:1561-1564` (`nav_inj_emb`, `nav_to_ctx`, `nav_to_intent`), `:1658-1664` (nav → tactical ctx and → field-predictor intent). Caveats: nav is the label at inference; the route head is a pure nav echo (§4, 09-04) |
| **R3** ALL tactical labels train the tactical layer | 🟡 **PARTIAL** | only the lat/lon decision labels (+ route) reach a loss: `refa_v1.py:1537-1539` (heads), `:2060-2075` (v7.2 label supervision); the 22 goal families and SPEED_BAND do not |
| **R4** tactical choices condition the operative plan | 🟡 **MET IN FORM, DEFECTIVE IN FUNCTION** | goal token → planning cost + FiLM tactical → operative (`refa_v1.py:38-39` design table, `:1650-1666`); MEASURED: the operative cost was goal-blind (3.9× under-turn, §4) |
| **R5** one combined tactical+operative trajectory to 6 s | ⛔ **ABSENT** | plan horizon **2.0 s**: `refa_v1.py:782` (`plan_horizon_s: float = 2.0`), `refa_v1_plan.py:85` (`horizon: int = 10`); the cost spans 6 s but no 6 s trajectory is emitted |
| **R6** strategic layer OFF | ⛔ **ABSENT — needs an architecture change** | the strategic policy is always in the chain (`refa_v1.py:1656`, `strat = self.strategic_policy(...)`), `nav_to_ctx` is sized from `cfg.strategic_cfg.d_ctx` (`:1562`), and `no_hierarchy` drops the tactical layer too (`:2069`) |

---

## 7. Open RESEARCH topics — ranked by measured effect (Rule Zero: largest lever first)

| rank | topic | largest measured effect (class, tier) | what it implies |
|---|---|---|---|
| 1 | **A learned nonlinear kinematic trajectory readout (P10) as the prior and the bar** | 6 s ADE **−0.5024 [−0.6331, −0.3767]** vs the linear readout; 2 s ADE −0.3495 vs `ha0_ext`; 6 s 2.93 vs `damp50` 3.45 (point estimates) (M, T1-equiv.) | the programme's "vision helps" bar for refav1 / v7F / refc heads, and the best residual prior for R5 — ⚠️ but it is **not in the R5 build's prior menu** (§5 A) |
| 2 | **Planner-free damped floors** (`kd_ha0_ext`, `damp50`) | `kd_ha0_ext − ha0_ext` **−0.2282** (282 windows) / **−0.2638 [−0.3374, −0.2005]** (2,399 windows); `damp50 − ha0_ext` −0.1576 (M, T1) | every older refav1 claim "vs `ha0_ext`" must be restated vs `damp50`/`kd_ha0_ext`; a floor only makes tests stricter |
| 3 | **Option T — does a trunk trained WITH the trajectory objective carry information beyond kinematics?** | **UNMEASURED** (H). The negative it must overturn: frozen trunk `P9 − P10` +0.0987 m at 6 s (M) | the only lever that can make refav1's vision matter; pre-register on event strata + 6 s, not on the pooled 2 s mean |
| 4 | **Data scale vs readout capacity on the FROZEN encoder** | untested at 4,572 clips: the trunk probe fit at most 600 episodes, and inner CV picked the **smallest** image capacity (PCA m = 8) every fold (M) | the cheapest discriminator before buying option T (§9 E1) |
| 5 | Longitudinal planner levers (`a0_shift`, jerk seam) | LON speed **−0.2283** (p4, 12× seed floor) and seam **−0.2360** (p4, 8.5×) (M, T1) — but **tied with `damp50` on LON on the full grid** | eliminated as planner levers on this checkpoint; the LON *signal* (accel persistence) is what P10 learns |
| 6 | Lateral cost geometry (`W_KAPPA`, Kamm cap, goal-conditioned cost) | `W_KAPPA` 15.11 vs 0: ADE −0.4338 on p4 (M) — but on the full grid the tuned lateral is **worse than planning straight** (`A1 − shipped` cross +0.1977) | eliminated; keep `kamm_mu` as a safety constraint for any planner |
| 7 | Distance keeping / closing rate | lead gap decodable R² 0.3632 (+0.4145 over pixels); closing rate absent; decoded-gap BAR-2 0.5714 vs 0.80 (M, probe) | a temporal (multi-frame) representation item for option T |
| 8 | Tactical decoder & vocabulary | LON head 0.4326 vs 0.2979 under `nav_zero`; turn AUC 0.88, L/R AUC 0.90; vocabulary cannot express R 100–1000 m curves (M) | R3/R5 replace the vocabulary bottleneck with a regressed trajectory |
| 9 | Strategic layer | route head = nav echo (M); strategic family unmeasurable on the planner surface (n = 0) | R6 says OFF for now — measure strategic only after driving works |

---

## 8. Open ENGINEERING topics

1. **Land B + C** (14 files, banked under `P4S/refav1_r1r6/code/fix/` and `P4S/code/fix/`) — use `stack/scripts/mm_commit.py` / `scoped_commit.py` with the per-path **descendant** check (CLAUDE.md git traps); the build's LF blobs are listed in `refav1_r1r6/RESULT.md` §9.1. Then run `A1cap` (§5 C′).
2. **Add a `kinmlp` R5 prior** (the P10 MLP, frozen) to `refav1_traj.py`, with a known-value test: zero-init head == P10 output on the 2,399 windows, max |Δ| = 0. Confirm which κ0 P10 consumes (UNVERIFIED: if it is the t0 steer channel, D1 applies).
3. **refav1 launch-gate profile** in `stack/scripts/launch_gate.py` (none today; launch gate is binding).
4. **Trainer defects** (`stack/scripts/refa_v1_train.py`): no `summary.json` done-marker (prints `done:` only at `:800`) — the cause of the **80-relaunch** episode (REG:2114-2123); feature standardiser fit on **one batch** (`:581-582`).
5. **Register/registry are stale for refav1** (Rule: update in the same turn): REG §2.4 has no row for `a0_shift`, turn asymmetry, perception probe, `loncomb3` full grid or the trunk probe; REG:2095 says `ccos` "IN PROGRESS" (delivered and refuted); **GC has no row for the 09-27 full-grid FAIL-WORSE or the trunk-probe NEGATIVE** (grep at tip: 0 hits for `trunk-probe`, `damp50`, `FAIL-WORSE`); several GC rows still read QUEUED/RUNNING for landed or abandoned arms; A13c code provenance unrecorded.
6. **Planner instrument defects** (moot while the planner is eliminated, but they will bite the moment a planner sits on an R5 proposal): `plan_source` labels seed winners `cem`; `planner.baseline_won_frac` under-reports (0.6631 vs true 1.0000; tie-break repairs gated on the LABEL, `refa_v1_plan.py:277-281`, `refa_v1.py:2203` per REG — line numbers from the registry, not re-checked); `target_speed` never passed ⇒ `W_VEND` inert while `stack/tests/test_refa_v1_a_sustain.py:148-150` claims otherwise; `cost_fidelity` (design gate G1) has no caller; DK hook absent at tip; `D-REFAV1-SEED-GOAL-MISMATCH` (goal from the 30-step control, seed truncated to 10 steps — 62/64 token pairs disagree).
7. **`v7_labels._MEASURED_GEOMETRY_TOKENS` is module state** set by the last `load_v7_labels` call — loading eval after train in one process flips the negative policy (`refav1_r1r6/RESULT.md` §8.1 item 8; unfixed).
8. **Storage**: train fp8 features deleted (~303 GB); checkpoint has two copies (Thor, dev box), **no HF copy**.
9. **Instruments missing**: training-seed replicate (never run), closed-loop harness for refav1 (never run), strategic family on the planner surface (no route label), refav1 answers to the 7 frozen-trunk questions (`Project Steering/ADVISORY_FROZEN_TRUNK_DEFECT_CLASSES.md`, BACKLOG A16 — partially answered by the trunk probe's bit-exact encode check, not formally).

---

## 9. Recommended next 3 experiments

Common bar language (from `REFAV1_RETRAIN_DECISION.md` §3, committed before any retrain data): T1, the 141 EVAL episodes, paired episode-cluster bootstrap, **four families**, **2 s AND 6 s**, GT-defined event strata reported beside the pooled mean; floors **`damp50`, `kd_x`/`kd_ha0_ext`, and the nonlinear kinematic readout `P10` (2 s 0.2229 / 6 s 2.9257)**; a lever effect must exceed **2× the training-seed replicate** (and the inference-seed replicate if anything samples).

**E1 — The frozen-encoder ceiling at full data (dev box, unblocked today).**
* *Hypothesis (H):* the trunk-probe NEGATIVE is a data-scale / readout-capacity artefact — a **trainable** spatial-temporal attention readout over the frozen DINOv3 token field (t0 and ≥1 past frame, region-pooled e.g. 8×20), fitted on **all 4,572 TRAIN clips**, beats P10 refit on the same 4,572 clips.
* *Bar (pre-register first):* `readout − P10` ADE @ 6 s < 0, CI excluding 0, at **2 training seeds**, larger than 2× the seed replicate; plus the shuffled-feature and constant controls the probe used; event strata (turn onset, stop, lead braking) reported.
* *Cost:* streamed encode 4,572 clips at ~4.6–4.7 s/clip ≈ **5.8–6 h on the RTX 4060** (E, from M 2,834 s / 600 clips), pooled cache ~16–64 GB on D: (E); readout training a few GPU-h. Pull read-only from Thor's v2ep sources (never add load to a training box — check Thor first).
* *Why first:* it is the cheapest experiment that decides **what option T must train**: POSITIVE ⇒ the adapter/readout was the bottleneck and option T can keep DINOv3 frozen; NEGATIVE ⇒ frozen DINOv3 at 256×640 carries no 6 s information beyond kinematics at our scale, and option T must train the encoder (the "REF-A v2" direction, §5 J) or the vision lever moves to REF-C's trained encoder.
* *Fallback lever if it fails:* E2 with an unfrozen (or LoRA) encoder, priced first.

**E2 — Option T, first arm (Thor or the A40; needs the PI).**
* *Hypothesis (H):* refav1's adapter + WMs trained jointly with the R5 trajectory loss (`--strategic-off --vmax-input --w-goal --w-traj --r5-prior <kinmlp|kdx>`) produce a 6 s trajectory that beats P10.
* *Bar:* `retrained − P10-style kinematic head (same inputs minus vision, same data)` ADE @ 6 s < 0, CI excluding 0, at two training seeds, > 2× the training-seed replicate; R1: **0** plan exceedances of the clip limit on every window; R6: no strategic parameter updated (bit-identical before/after).
* *Cost:* one epoch ≈ **20 h on Thor** (M: 3.47 s/step × 21,109 for the ep3 recipe; the R5 head's extra cost UNMEASURED) **× 2 seeds**; plus either ~303 GB of fp8 features placed, or on-the-fly DINOv3 encode whose per-step cost must be **measured before launch**.
* *Unblocks:* §5 B + C landed, a refav1 launch-gate PASS, PI decisions D1–D5 and storage/compute, and Thor free (it was training refcv8 at last record — I).
* *Fallback:* if E1 was positive but E2 fails, ship a frozen-encoder R5 head (E1's readout) as refav1's trajectory; if both fail, retire refav1 as a WM-planner reference and record the negative with its scope (one checkpoint, frozen DINOv3, open loop).

**E3 — The kinematic prior inside refav1 (dev box, ~0 GPU, unblocked after landing).**
* *Hypothesis (M-anchored):* swapping the R5 prior from `kdx` to the frozen P10 MLP lowers every R5 arm's floor by ~0.5 m at 6 s (the measured P10 − linear gap), so E2's head only has to learn the vision residual.
* *Bar:* known value first — zero-init R5 head == P10 output on the 2,399 windows, **max |Δ| = 0.0**; then `kinmlp-prior − kdx-prior` (zero-init) ADE @ 6 s reproduces the trunk probe's −0.50 within its CI (an identity check, not a lever claim).
* *Cost:* code + CPU (the P10 weights are re-fit in minutes from the banked `code/probe_te.py`; the TRAIN features must be regenerated — `C:/Users/Admin/refav1_trainprobe/` is gone).
* *Unblocks:* §5 B landed; PI D1 if P10 uses the steer channel at t0.
* *Fallback:* `--r5-prior damp50` (best 6 s FDE of the three built priors: 10.150 vs `kdx` 11.216 — exploratory, M, `refav1_r1r6/RESULT.md` §6).

⚠️ None of E1–E3 is "tune the planner again": the full grid eliminated every zero-training planner lever on 21,109. A planner returns only after a trained trajectory exists, as a re-ranker over its proposals.

---

## 10. Traps specific to this arm (each has already cost time)

1. **Three grids, never mixed** — p4 (40 windows / 8 eps, turn-dense), full (282 / 141), trunk-probe (2,399 / 141). p4 tuning bought a lateral behaviour the lane-keep-dominated full grid punishes (247/282 windows are `lane_keep`).
2. **`ha0_ext` is no longer the strongest floor** — `damp50` and `kd_ha0_ext` beat it, separated. Restate any "beats `ha0_ext`" claim.
3. **The inference-seed floor is config-dependent**: 0.0607 m ADE (p4, `ccos`), +0.0619 (full grid, `loncomb3`), +0.0015 (`best`) to +0.1035 (`combined`) on p4. The "≈0.30 m" figure in CLAUDE.md/REG:3035/3321 comes from `D-REFAV1-DRIVE-AB` (+0.2956 on the *unchanged* windows of one panel, `AI/2026-09-05-refav1-make-it-drive/AB_RESULT.md:92-94`), **not** from `D-REFAV1-SEED-GOAL-MISMATCH` as CLAUDE.md attributes it. Measure the replicate on the arm you claim.
4. **Read the plan's CONTENT, not its label** — `planner.baseline_won_frac` read 0.6631 while the true trivial-plan rate was 1.0000; `plan_source` says `cem` for an injected baseline.
5. **Units**: `actions[:,0]` is a steer angle (×2.9 vs κ); the banked `ol`/`ha` arms ran `--action-units kappa` (`REFAV1_ARM.md` §3b).
6. **Retracted figure still in a 09-27 RESULT**: `loncomb3` RESULT §4.5 quotes "the world model's lateral contribution is 1.63e-10" — that is the **step-1,000** checkpoint's number (RETRACTION_LOG #21); at 21,109 the median is **1.99e-06** and the binding penalty under the shipped cost is **jerk** (REG:2144-2150). The registry wins.
7. **Price the file the consumer opens** — the trainer reads the **~303 GB fp8 feature cache** (4,572 × ~66.2 MB), not the ~155 GB v2ep source cache (the trunk-probe SPEC quoted the latter; corrected in its RESULT §5).
8. **Thor's `/home/nvidia/TanitAD/stack` is stale** (HEAD `30d6d60`, no `refav1_arm.py`) — use a self-contained shipped tree and assert `tanitad.__file__` (`run_fullgrid.sh` does this with a `ZZ…ZZ` marker).
9. **The checkpoint was trained from Thor's 1,885-line `refa_v1.py`** — re-running from tip is not a replication.
10. **Done-marker trap**: `refa_v1_train.py` writes no `summary.json`, so the supervisor relaunched the finished run 80 times and left a `DRIFT_ALARM` red herring; the log's last row (21,100) is not the final step (21,109) — the checkpoint's `step` field is authoritative.
11. **Dev-box commit limit, not RAM**: the trunk probe died on a 63 MiB allocation with 11 GB physical free (Windows commit 86.6 GB exhausted by a concurrent 23 GB test suite) — and its wrapper reported exit 0. Make long jobs resumable per arm and assert on the artifact, not the exit code.
12. **The full training step does not fit the 8 GB 4060** at any batch or precision — training is Thor/A40 only; the 4060 runs evals (~2.1 h/arm, full grid) and probes.
13. **`C:/Users/Admin/refav1_probe/` holds `*_LEAKY_*DO_NOT_READ*` smoke outputs** — never read them as results.
14. **Name collision**: "REF-A v1" in `V7_FULLSCALE_RECOMMENDATION.md` (2026-08-24) is the old DINOv2 arm; refav1 is the DINOv3 WM-planner. The directory `refav1-b1-v72-1ep-21109` holds a **step-1,000** checkpoint.
15. **`/c/lgs/tip_files.txt` is incomplete** (it omits e.g. `taniteval/tools/refav1_arm.py`, which `git cat-file -e` proves present) — the `ls-tree` truncation trap; assert paths positively.

---

## 11. File index

| path (at tip unless marked) | why |
|---|---|
| `Project Steering/MODEL_REGISTRY.md` §2 (lines 1967-2173) | REF-A lineage and refav1 §2.4 (stale after 09-06) |
| `AI/2026-09-27-refav1-trunk-probe/RESULT.md` (+ `SPEC.md`, `code/`, `raw/out_te/`) | ⭐ the frozen-trunk NEGATIVE and the P10 bar |
| `AI/2026-09-27-refav1-fullgrid-loncomb3/RESULT.md` (+ `SPEC.md`, `raw/run/run_fullgrid.sh`, `raw/analysis/pd_fullgrid*.md`) | ⭐ planner eliminated; exact dev-box invocation |
| `P4S/REFAV1_RETRAIN_DECISION.md` | option T vs H, bars, the PI ask |
| `P4S/refav1_r1r6/RESULT.md` (+ `code/fix/…`) | R1/R3/R5/R6 build, landing table, D1–D5 |
| `P4S/R1_FULLGRID_CHECK.md`, `P4S/LANDING_READY.txt`, `P4S/code/fix/…` | R1-VMAX batch + the `A1cap` pre-registration |
| `P4S/STATE_AND_GAPS.md`, `P4S/HANDOFF_TO_MASTER_MIND.md`, `P4S/raw/SURVEY_REFAV1.md` | 2026-09-27 state, gaps, contradictions (survey = I) |
| `taniteval/results/RESULT-refav1-21109-openloop.md` (+ `refav1-21109-openloop*.json`) | the epoch-end four-family read |
| `taniteval/tools/refav1_arm.py`, `taniteval/tools/REFAV1_ARM.md` | the eval adapter and its Thor invocation |
| `taniteval/tools/refav1_paired_delta.py` | the paired four-family delta tool |
| `stack/tanitad/refs/refa_v1.py` | model, config, cost constants, `plan()` (`:2393`) |
| `stack/tanitad/refs/refa_v1_plan.py` | iCEM, `PlanConfig`, `kamm_mu` |
| `stack/tanitad/refs/refav1_lon_cost.py` | longitudinal cost levers |
| `stack/tanitad/data/refav1_loader.py` | windows, labels, nav join |
| `stack/scripts/refa_v1_train.py` | trainer (no done-marker; one-batch standardiser) |
| `stack/scripts/dinov3_fp8_encode_ship.py` | the fp8 encoder/shipper (needed for option T) |
| `stack/tanitad/models/kinematic_prior.py` | the refcv-line kinematic priors (`ha0_ext_pose`); refav1's `kd_x` lives in the trunk-probe `code/` until the R5 build lands |
| `stack/scripts/launch_gate.py` | binding gate — no refav1 profile yet |
| `stack/tests/test_refa_v1*.py`, `stack/tests/test_refav1_*.py` | ~45 pinning tests |
| `AI/2026-09-05-refav1-cost-geometry/RESULT.md` (+ `HANDOFF.md`, `PREREG_COST_GEOMETRY.md`) | W_KAPPA optimum, Kamm cap, nulls |
| `AI/2026-09-05-refav1-make-it-drive/RESULT.md`, `AB_RESULT.md` | two gates; the 11.2× seed-ratio A/B |
| `AI/2026-09-05-refav1-goal-margin/RESULT.md` | the vocabulary diagnosis |
| `AI/2026-09-05-refav1-lonshift-t1/RESULT.md` | `a0_shift`, two-GPU replication |
| `AI/2026-09-05-refav1-close-the-gaps/RESULT.md`, `…/raw/refav1_margin_banked_2026-09-27/RESULT_HARVEST.md` | goal-blind cost; `kammshift` FAIL; `loncomb3` on p4 |
| `AI/2026-09-06-refav1-perception-probe/RESULT.md`, `…-decoded-gap/RESULT.md`, `…-distance-keeping/RESULT.md` | lead gap / closing rate / DK |
| `AI/2026-09-04-refav1-cost-scale/RESULT.md` | the cost in units |
| `AI/2026-09-05-refav1-ccos-eval/RESULT.md` | `ccos` on the full grid |
| `AI/2026-08-18-refa-v1/REFA_V1_DESIGN.md` | design rationale and gates G1–G5 |
| `Project Steering/ADVISORY_FROZEN_TRUNK_DEFECT_CLASSES.md` | the 7 questions owed for refav1 |
| `Project Steering/BACKLOG.md` (A13c, A16, refav1 pull-list from line 119) | open small items |
| `Project Steering/RETRACTION_LOG.md` (#15, #21, 2026-09-03 entries) | refav1 retractions |
| dev box: `C:/Users/Admin/refav1_eval_slice/ckpt_ep3/{ckpt.pt,config.json}` | the checkpoint + its training record |
| dev box: `C:/Users/Admin/tanitad-data/refav1-eval141/` | eval fp8 cache + episodes |
| Thor: `/home/nvidia/experiments/refav1-b1-v72-ep3-speed/`, `/home/nvidia/data/refav1-fp8-eval` | canonical run dir + eval cache |


---

<!-- appendix: B_REFD.md -->

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


---

<!-- appendix: C_FLAGSHIP.md -->

# C — THE FLAGSHIP LINE (v1 → v7F) AND THE PROPOSED v8F

> **Source of truth:** landed tip `460508ed2f5e120667859e637ac233c28a48d90b` (2026-10-15 21:54 +0200), read
> through `git --git-dir=/c/Users/Admin/tanitad-push/.git show <tip>:<path>`; `MODEL_REGISTRY.md` at tip =
> `/c/lgs/reg_tip.md`. Nothing on the G: working tree was read. Abbreviations used below:
> **`P4V`** = `products/P4-training-pipelines/2026-09-27-v7f-refav1-state` · **`ARI`** = `TanitAD Research Lab/Architecture & Inference/Research`
> · **REG** = `Project Steering/MODEL_REGISTRY.md` · **GAC** = `Project Steering/GOALS_AND_CLAIMS.md`.
> **Evidence classes:** MEASURED (+ artifact) · PUBLISHED (+ key) · INHERITED (another doc, not re-verified) · ESTIMATED · HYPOTHESIS.
> **Tiers:** T0 = teacher-forced WM diagnostic, NEVER driving · T1 = self-action / action-closed, OPEN LOOP on logged frames
> (primary for capability) · NavSim = open-loop plan scoring (T1-family). Nothing in the flagship line has ever been closed loop.
> ⚠️ "v8F" does not occur anywhere in the repo text. Everything in §C.6 is a PROPOSAL (HYPOTHESIS), not a registered spec.

---

## C.1 Status — one paragraph

**The flagship has not trained a model since 2026-08-11, and v7F has never been trained at all.** The last
trained and evaluated flagship artifacts are v5f-30k plus the stage-A predictor repair, in the v5.8f assembly
(REG §1.8, §1.13c, §1.14; T0 / T1-diagnostic). The deployed v1 (`flagship4b-speedjerk-30k`) has a full-set
ADE@2 s of 0.4271 [0.3675, 0.4871]. That number is a **T0** world-model fidelity number: the model is given the expert's
true future actions. At T1 the v1.x lateral skill turned out to be an **action echo**: S-curve reproduction is
0.9785 open-loop and 0.0430–0.0538 self-action, and the hold-action control reproduces 0.0 % (REG §1.12).
v6F was retired by the PI on 2026-09-20: *"we will go directly to v7f"* (`PI_DECISION_QUEUE.md:1466`). v7F has
**no checkpoint, no run directory and no registry row** (`D-V7F-NEVER-TRAINED`, GAC:7243). The PI's binding
training hold of 2026-08-31 (`V7_LAUNCH_GATE.md:3`) has **no recorded release**. That file was last changed
2026-09-02. The PI's 2026-09-27 words *"close P3, solve the rest"* and the R1–R6 directive exist only in
`P4V/` and in auto-memory, not in GAC, the PI queue or `V7_LAUNCH_GATE.md` (MEASURED: git grep of the tip).

The whole R1–R6 v7F build is **unlanded**. That covers the merge with 4+2 nav fixes, the `v7f` launch-gate profile, the
`v6_chain --profile v7f` ladder and the exact-resume fix. It sits as source copies under `P4V/`. The tip blobs of
`train_v6_staged.py`, `v6.py` and `v6_chain.py` are **byte-identical to `c36b6ddd`**
(`91b675a7…`, `3d6e2235…`, `fd6f0c7c…`, MEASURED by `git rev-parse`). Their last commits are on 2026-09-06, and
`launch_gate.py` has profiles `refc / refcv6 / refcv7 / refcv8` only.

**The most important fact for v8F:** the programme's driving results now live on the **REF-C chassis**:

* refcv7-50,400 beats its echo control at 6 s by −0.98 m ADE, separated (T1).
* At 2 s it does NOT beat the echo control.
* It beats STOP on NavSim navtest.
* It **loses to STOP on navhard**.
* refcv8 is training on Thor, ETA ≈ 2026-10-20 (ESTIMATED).

The "beats doing nothing" bar has also moved. A **learned kinematic MLP** (P10) scores 6 s ADE 2.93 m against
5.36 m for the echo on the refav1 grid. **No REF-C or flagship arm has ever been read against it** (MEASURED absence: git grep
for `kin_mlp|damp50` hits only refav1/`P4V` paths).

---

## C.2 Lineage v1 → v7F

All rows before §1.12 predate `EVAL_DOCTRINE.md` (2026-08-06). Where a row carries no tier stamp it is marked
UNSTAMPED. v1.x rows use the parity corpus `physicalai-train-e438721ae894`. They are **not comparable** with the
non-parity refcv7/8 corpora or with the eval139 grid.

| # | run (REG §) | status | headline (class · tier · estimator) | verdict | what it taught |
|---|---|---|---|---|---|
| v1-ns | `flagship4b-phase0-30k` (§1.1) | killed at 22,950 by the 2026-07-14 speed reset | the 2.918 m no-speed control (CLAUDE.md, INHERITED) | ablation control, **not** the deployed v1 | speed as an input is causal |
| **v1** | `flagship4b-speedjerk-30k` (§1.2) | ✅ DEPLOYED, step 29,999, ~53 h A40 | `ade_0_2s` full-set **0.4271 [0.3675, 0.4871]** (40 ep, episode-cluster bootstrap); heldout split-mean 0.4522 (MEASURED `taniteval/results/driving_flagship-30k.json`) · **T0** | the deployed model; 263,442,838 params | a T0 fidelity number, never a planning number |
| v2 | `flagship4b-v2-30k` (§1.3) | ❌ abandoned at 7,800 | — | superseded by v3enc | — |
| v3enc | `flagship4b-v3enc-30k` (§1.4) | ⏹ stopped at 10,800 | 10 k gate `RESTART` (one secondary failed) | restart budget 1/2 on `encoder-grounding` | gates via `run_gate.py` |
| v1.6-ab | `flagship-v16-ab-ft` (§1.4b) | ✅ 5,999 | **TIED** with v1; the "best in program" claim is retracted (C1) | tie | quote the eval harness, not trainer val |
| v4 line | v4 → v4.1 → v4.2 → v4.2b (§1.5) | killed / superseded | v4.1@10k `ade_0_2s` **0.8522**, CI [0.7468, 0.98] vs bar 0.60 (MEASURED `Gates/flagship-v4-gate-10k-2026-07-23.json`) · UNSTAMPED | formally INCOMPLETE, substantively **FAIL** | joint planner+WM training starved the planner; the WM stayed healthy |
| v2corpus | `flagship-v2corpus-30k` (§1.7) | 🟢 "RUNNING" since 07-25, **UNVERIFIED / stale** (the 2026-08-16 sweep found no final row) | none | unknown | an unfinished row; do not wait on it |
| v1arch | `flagship-v1arch-v2bal-30k` (§1.9) | ✅ 29,999; HF `Sayood/tanitad-flagship-v1arch-v2bal` | first complete four-family block; `ade_mean_4wp` **0.5752 [0.5370, 0.6142]** (290-clip episode-cluster bootstrap, MEASURED) · OOD-val | complete; no head key (canary-only in `eval_flagship_v4.py`) | the v1 architecture on more data |
| v5f | `flagship-v5f-w120-30k` (§1.8) | ✅ 30,000 (2026-08-09); HF `Sayood/tanitad-flagship-v5f-w120` | selected ADE@2 s **0.4011**, oracle-in-fan **0.1975**, sel_gap **0.2036** (881 windows) · **T0** | w120 geometry, not comparable to 256² | the **selector leaves half the fan's quality** |
| v1.6-uni | `flagship-v16-unicycle` (§1.10) | ✅ 2026-08-06 | unicycle readout on the frozen v1arch trunk | — | ⚠️ "v1.6" names two different runs |
| v1.7 | `flagship-v17-speedloss` (§1.11) | ✅ pre-registered run 6, outcome **B** | P1 decel ratio 0.1547 vs ≥ 0.40 ❌; ADE 0.2849 vs 0.3398, CI-better · **T0** | hypothesis **REFUTED** | a speed loss on a frozen trunk does not fix the decel response |
| §1.12 | v1.6 / v1.7 self-action | MEASURED 2026-08-06; relabelled OPEN LOOP 2026-09-02 | ADE 0.3398→**0.4714** (v1.6), 0.2849→**0.4616** (v1.7); CV floor 0.5352; S-curve **0.9785→0.0538 / 0.0430**; hold-action **0.0 %** · **T1** | **lateral skill = action echo** | the founding failure. ⚠️ The `+7.1642` flagship-minus-refc-base contrast is 98.75 % "the flagship loses to a straight line" (`cl_ha0` decomposition) |
| v5.8f | wedge ladder, E4.4, stage-A, assembly (§1.13–1.14) | MEASURED 2026-08-09…11 | stage-A: lateral gain 0.27→0.971, lon sign →1.0 (**T1-diag**). v5.8f selected ADE **0.4815** vs v5f 0.4011 (+0.08 worse), oracle **0.1077**, accel MAE 0.515 vs 8.10 · **T0** | last trained flagship assembly; W7-on-repaired failed by INSTRUMENT COMPOSITION | the action interface can be repaired at head scale; selection remains the gap |
| v6F | `v6F-SW-30k` (§12; `train_v6_staged.py`) | **RETIRED** 2026-09-20 by the PI | the 40:1 pool is **not** the bottleneck. Through the same pool DINOv2-B reads `lead_gap` **0.44997** vs ours **0.00496** (91×) · **T0-diag** | encoder is the constraint | ⇒ a trained-from-scratch ViT trunk is the weak part |
| v7-tiny | `rdw8p30k`, `splitp30k`, `champ30k`, census (§13) | ~19 M proxies, S-W only, every planner objective 0 | `rdw8p30k`: participation 25.58, predictor cos **0.6224**, scene content `n_agents` +0.0777 (GAC:359). `splitp30k`: `n_agents` **+0.3864** from `z_t` (GAC:359; REG §13.0d prints +0.3881 on its ENV set), predictor beats the constant floor (nrmse 0.8416) but **adds nothing over `z_t`** (−0.0008/−0.0320/−0.0158 at k=1/3/6). Census: **0 of 32 arms** move nrmse > 0.1 % when actions are shuffled (max 0.0919 %) · **T0-diag** | **dissociation**: no arm both carries and predicts the environment | the WM extrapolates the scene and ignores actions |
| v7-tiny T1 | `emao14_30k`, `o14fut30k`, `…tauramp` (REG ~4724–4742) | MEASURED 2026-08-30 | ade 13.8–14.3 m, cl worse than hold-action on every arm · **T1 but VOID**: decoded through a random-init `step_readout_op` (`D-METRIC-DECODE-FALSE-RESULT`, GAC:8945) | **VOID**; the registry table still lacks a VOID banner (owed) | no admissible v7 T1 read exists |
| **v7F** | `PREREG_V7F.md` (2026-09-03) + R1–R6 build `P4V/` (2026-09-27) | **NEVER TRAINED**; R1–R6 code unlanded | none | — | see §C.3 |

---

## C.3 v7F in depth

### C.3.1 Architecture (MEASURED from source unless marked)

* **Trainer:** `stack/scripts/train_v6_staged.py`, 11,057 lines and 232 flags (`P4V/raw/flag_inventory_train_v6_staged.md`).
  * Model: `stack/tanitad/models/v6.py`. Chain: `stack/scripts/v6_chain.py`.
  * **245.6 M params**, MEASURED in `P4V/STATE_AND_GAPS.md` §3. The prereg's "336.5 M" is stale.
  * A trap inside that count: the O5 EMA teacher, **86,236,544 params, was being unfrozen into the optimizer** (`ARI/2026-09-06-v7f-budget/RESULT.md`). Commits `51dae02`/`982476c` (2026-09-06) address it. Read them before relying on the count.
* **Stages:** from `v6.py:4567 STAGE_GROUPS` (INHERITED from `P4V/V7F_PLAN_R1_R6.md` §1; the line is valid at tip because the blob is identical).
  * **S-W** = `encoder, readout, predictor_op, aux` (the world model).
  * **S-T** = `layer_tac, planner`.
  * **S-S** = `layer_str`.
  * **S-J** = everything trainable.
  * Under R6 the plan is S-W → S-T. S-S is skipped, and any joint phase must exclude `layer_str`.
* **Encoder:** ViT 768×12×12, 86.1 M params.
  * It is initialised from DINOv3 ViT-B/16 by transplant and is **TRAINABLE**, at LR ×0.1 with an MSE anchor to a frozen copy of the seed.
  * The trainable encoder is a register decision: E-DEC-64 measured freezing as DEGENERATE (nrmse 0.9301 vs 0.8115), GAC:365.
  * ⚠️ **The transplant loses DINOv3 content.**
    * E-SEED-2 at L/16 (GAC:227): `dino_hf − seed_asis` `n_agents` **+0.2050 [+0.0896, +0.3233]**, separated.
    * S0 at B/16 (`ARI/2026-09-27-v7f-s0-b16/RESULT.md:36-40`):
      * the ImageNet-norm + zero-`pos` repair is **null**: −0.0053 [−0.1663, +0.1393];
      * the transplant loses a **separated** share of ego-speed content: +0.2331 [+0.0884, +0.3661];
      * the seed as wired does beat random on scene content: +0.1941 [+0.0429, +0.3516].
    * ⇒ `PREREG_V7F` §10 **D1 option A** (wrap the real HF `DINOv3ViTModel`) is the evidence-backed choice. It is still **PI-open**.
* **World model:** the predictor is conditioned on measured ego state, `--cond-param omega_accel_v` = `[yaw_rate, a_long, v/SPEED_SCALE]`.
  * `train_v6_staged.py:9798`; semantics at `:171`, INHERITED via GAC:7261.
  * This is realised motion, not a command: E-DEC-57, r = 0.9988.
  * Recipe core (`PREREG_V7F.md` §3): O5 L1 rollout, O6 SIGReg (32 subspaces, 512 slices), O14 future-observation (k 4), EMA target τ 0.996.
* **Tactical / operative:**
  * S-T emits a goal-conditioned fan: `plan_proj(z_op) + cand_queries ⊕ g_tac_embed → UnicycleEmission(·, v0)`, giving N × 60 × (a, κ) at 0.1 s = 6 s, with the final layer zero-initialised to constant velocity (`v6.py:5766-5795`, INHERITED from `V7F_PLAN_R1_R6.md` §3′).
  * Selection uses `GoalDistanceScorer` (`v6.py:2710`) or `MLPCandidateScorer` (`v6.py:2809`).
  * **The WM is not the selector.** Roll-consistency selection was +5.98 m worse and is refuted (INHERITED, citing `v6.py:3946-3950`).
* **Selection gate SEL-1:** E-WC2 measured σ/ADE **9.9915 [7.4492, 13.5119]** against a 3.0 refusal line (MEASURED constant in `v6_chain.py:178`).
  * ⇒ every selector launch is refused, and the S-T default is `--selector none` (`v6_chain.py:43`).
  * The committed unblock is the E-WC2-SW latent dump at the S-W→S-T boundary (`v6_chain.py:224`), which needs a trained S-W.
* **Option B = a configuration, not a build:** `--n-candidates 1 --proposals query --selector none` (`train_v6_staged.py:9271/9274/9298`).
  * This gives one decoded 6 s trajectory with no selector, so SEL-1 does not apply.

### C.3.2 Data (as designed)

* B1 corpus `physicalai-b1-w120-256x640cyl`: 4,572 train + 141 eval (`D-CORPUS-B1`, `ARI/2026-09-05-v7f-state/RESULT.md:60`).
* v7.2 labels: `s2_labels_v7.2_train.jsonl.gz`, 4,572 records, md5 `0ff90213…`, as used by the chain dry-run.
* v8 max-speed sidecar: `refcv6_speed_max_v8_{train,eval}.jsonl`, `limit_kmh ∈ {30,50,100,120}`.
* nav: `nav.jsonl.gz`.
* ⛔ `BACKLOG.md` R8: the eval-clip exclusion in the v7 B1 path is a launch blocker. The 141 eval clips sit inside the 4,713-clip cache. Its status in the merge is UNVERIFIED.
* ⚠️ The v9 per-frame labels (refcv8's) were never wired into the v6 trainer.

### C.3.3 What was trained / evaluated

* **Trained:** nothing at v7F scale.
* **Evaluated:** the S0 trunk-seed panels (B/16, L/16), the T0 diagnostics on v7-tiny, and the dry ladders below.
* The CPU dry ladder S-W→S-T (tiny geometry, real data inputs) ran **rc 0 / rc 0** in 11.4 s (`P4V/v7f_chain/RESULT.md`).
* The gate rehearsal on dev-box CPU passed G-LIVE / G-DVB / G-EVAL / G-CKPT.
  * G-HYG FAILs on 4 open config classes.
  * G-CLOCK = ERROR: there is no clock reference for the v7F cache.
  * The token is INCOMPLETE by design (`P4V/v7f_gate/RESULT.md`, `P4V/v7f_ckpt/RESULT.md`).

### C.3.4 Defects found and fixed — all only in `P4V/` (unlanded)

| defect | evidence | fix location |
|---|---|---|
| `--nav-cond` could not START: (1) `synthetic_batch` lacks nav keys → `NavTokenMissing`; (2) `"nav_token": b.get(...)` written after the NavEmitter splat → None at step 1 (`train_v6_staged.py:7314` vs `:7336`); (3) the NavConditioner's 12 params are in no `_GROUP_PREFIXES` group | MEASURED, 4/4 mutants | `P4V/v7f_merge/` §1 |
| ⛔ (4) `build_stack_from_args` **never mapped `--nav-cond`** ⇒ every v6/v7 launch "with nav" at the tip built **no** nav conditioner | MEASURED by the gate's G-DVB | `P4V/v7f_merge/` §1b |
| F7: operative nav got **no gradient in S-W** (the WM-loss helpers never forward nav); F8: `--dry-run` died on nav | MEASURED; G-LIVE flipped FAIL→PASS | `P4V/v7f_merge/` §1c |
| F3b: a v6 resume replays the cosine schedule on top of the loaded LR, so the remainder trains at **(1+cos(πn/T))/2** of schedule; 0.34549 measured at n=3/5 | MEASURED (`P4V/v7f_ckpt/raw/probe_before_after.json`) | `P4V/v7f_ckpt/` |
| F3: resume not exact (RNG + data position) | MEASURED; now bit-identical, model 259/259 and 261/261 | `P4V/v7f_ckpt/` |
| F1 (open): the tactical nav path reaches the shared nav embedding (X3 isolation) | MEASURED | owner `tanitad/models/nav_conditioning.py`, **unfixed** |
| F2 (open): `nav.jsonl.gz` and the speed sidecar must be md5-consistent, or S-T dies at `build_vmax_join` after the corpus build | UNVERIFIED on Thor | `P4V/v7f_chain/RESULT.md` §6 |
| §9 launch line unrunnable: the LDAD flags do not exist anywhere; `--horizons` default refused; `--w-s2-goal` refused at S-W | MEASURED (`D-V7F-LAUNCH-LINE-UNRUNNABLE`, GAC:7255) | build or strike LDAD |
| S-W cannot answer its own G-DRIVE: the readout is left at random init | from source (`STATE_AND_GAPS.md` §3 #6) | needs a stage that trains the readout |

### C.3.5 R1–R6 compliance (PI 2026-09-27, binding for v7F)

| req | at TIP (what a launch today would get) | in `P4V/v7f_merge` + `v7f_chain` (unlanded) |
|---|---|---|
| R1 max speed input + hard cap | **ABSENT**: 0 hits for `max.speed\|vmax` in `train_v6_staged.py` (MEASURED) | MET: `--max-speed-input-v6 --speed-max-sidecar-v6 --plan-vmax-cap` |
| R2 nav → tac + op, train + inference | **ABSENT in effect**: `--nav-cond` exists (`:9207`) but is not mapped into the model and cannot start | MET after defects 1–4 + F7/F8; F1 isolation open |
| R3 ALL tactical labels → tactical loss | **ABSENT**: no family reaches any loss; S-T does not load the label file (`D-TLIGHT-1`, pinned by `stack/tests/test_tactical_label_reach.py`) | MET behind `--w-tac-label-all 1.0 --goal-multilabel --s2-labels` |
| R4 tactical conditions operative | **PARTIAL**: the goal path feeds the emission (`g_tac_embed`); `--tac-goal-cond` (`:9434`) is strategic→tactical, not tac→op | MET: `--tac-op-cond detached\|e2e`; **the mode is a PI decision (open)** |
| R5 one combined 6 s trajectory | **MET as config**: 60 × 0.1 s emission; option B | MET; `--n-candidates 1` enforced on both stages |
| R6 strategic OFF | **PARTIAL**: S-S can be skipped, but the chain's S-T turns `--tac-goal-cond` ON | MET: `--strategic-off`; the chain refuses S-S/S-J |

---

## C.4 Settled findings — do not re-litigate

| # | finding | evidence (class · tier) |
|---|---|---|
| S1 | **v1.x lateral skill is an action echo.** | S-curve 0.9785 open-loop vs 0.0430–0.0538 self-action, hold-action 0.0 % (REG §1.12; MEASURED · T1) |
| S2 | **Scene → action, not action → scene.** The action's marginal on future scene content is null (+0.0755 t 0.64 / −0.0179 / +0.1083), against a positive control at t 8.5–14.3 (E-DEC-48b). ⇒ reactions belong in the PLANNER, not the WM. | `V7_RECIPE_AND_SCALEUP.md` §2 (MEASURED · T0) |
| S3 | **The predictor is action-deaf on every arm.** | Census 0/32 arms > 0.1 % nrmse change under action shuffle; the action pathway is 2–9 % of the latent pathway (REG §13.0c). `ẑ_GT − ẑ_HOLD` is exactly 0 in 6/9 cells (`D-V7F-L3-ACTION-DEAF`, GAC:7751). 97 % of the conditioning response is the speed channel (`D-V7F-CONDITIONING-IS-SPEED`, GAC:7619). All MEASURED · T0 |
| S4 | **The encoder stays trainable.** Freezing is DEGENERATE (nrmse 0.9301 vs 0.8115). | E-DEC-64 (GAC:365, MEASURED · T0) |
| S5 | **The trunk, not the pool, is the constraint.** | 91× `lead_gap` gap vs DINOv2-B through the same pool (REG §12.1, MEASURED · T0) |
| S6 | **The DINOv3 transplant loses content, and the cheap repair is null.** | E-SEED-2; S0 B/16 (§C.3.1) |
| S7 | **v7F S-T does not select by imagination; SEL-1 gates it.** Option B is a config. | `v6_chain.py:43,178`; `v6.py:2710,2809`; RETRACTION #2 in `P4V/HANDOFF_TO_MASTER_MIND.md` §4 |
| S8 | **An imagination planner over an action-deaf WM cannot steer.** refav1's WM contributes 1.63e-10 of the lateral cost. The pre-registered full-grid `loncomb3 − ha0_ext` is +0.0896 / +0.1515 m ADE at both inference seeds: **FAIL-WORSE**. | `ARI/2026-09-27-refav1-fullgrid-loncomb3/RESULT.md` (MEASURED · T1) |
| S9 | **A frozen trunk adds nothing beyond t0 kinematics** (refav1, pre-registered A2+A3). `P9 − P10` = +0.0054 [+0.0016, +0.0093] @ 2 s and **+0.0987 [+0.0332, +0.1658] @ 6 s** (separated worse). | `ARI/2026-09-27-refav1-trunk-probe/RESULT.md:77` (MEASURED · T1-equivalent · paired episode-cluster bootstrap, 141 eval episodes) |
| S10 | **A learned kinematic MLP is the strongest "do nothing" bar.** P10 6 s ADE **2.9257** vs linear 3.4281 (−0.5024 [−0.6331, −0.3767]), `kd_x` 3.5584, `damp50` 3.45, `ha0_ext` 5.36. At 2 s vs `ha0_ext`: −0.3495 ADE, −0.9272 FDE. | same file :64-90 (MEASURED · T1-equiv · refav1 grid, 2,399 windows / 141 episodes). PUBLISHED analog: AD-MLP `2305.10430` (banked) |
| S11 | **Damping alone beats the echo by ~1.9 m at 6 s.** `damp50 − ha0_ext` ADE@6 s **−1.9025 [−2.3605, −1.4971]**. About 80 % of the p4 planner blend's gain was damping. | `P4V/HANDOFF_TO_MASTER_MIND.md` §00 (EXPLORATORY, refav1 grid; INHERITED from the full-grid package) |
| S12 | **The listwise selector loss is lateral-blind.** noLW vs V-R8 GT-turn cross 6 s: −4.57 [−8.04, −1.48] / −4.98 [−8.54, −1.64], both seeds. | GAC `PI-R8-NOLW` CONFIRMED (MEASURED · T1 tiny rung) |
| S13 | **Tactical conditioning has not shown a lateral benefit at the tiny rung.** Turn direction noLW −0.009 / 0.000 vs V0. The measured gain is longitudinal (along 6 s −0.31 / −0.34 m). | GAC `H-R8-RECIPE-1` 2026-10-14 (MEASURED · T1 tiny rung) |
| S14 | **No launch without a launch-gate PASS token** (PI 2026-09-26). refcv6 trained three declared-but-unbuilt levers. | memory `launch-gate-binding`; `stack/scripts/launch_gate.py` |

**Refuted levers, each with the next lever it named:**

* LDAD / encoder-shaping on v7F (`H-V7F-1`):
  * not refuted, but **unrunnable**: the loss does not exist;
  * next lever: build it, or strike it before any v7F compute (`D-V7F-LAUNCH-LINE-UNRUNNABLE`).
* Ten WM objectives (O1–O3, O7–O11, O13, PSG):
  * failed as a class (E-DEC-52);
  * next lever: the hierarchy owns reactions (S2), plus a genuine command channel (`D-V7F-COMMAND-CHANNEL`, GAC:7261: refcv4b's (a_lon, a_lat) control vocabulary, HYPOTHESIS, never run).
* The τ-ramp: NEUTRAL at 30 k. The drift arm (P3): CLOSED by the PI 2026-09-27 (`D-V7F-DRIFT-NULL`).
* Heads-only refav1 retrain:
  * withdrawn (S9);
  * next lever: option T, the trunk trained with the trajectory objective (`P4V/REFAV1_RETRAIN_DECISION.md`);
  * blocker: the ~303 GB fp8 train cache is gone from Thor (C82 correction in the handoff).
* The listwise selector:
  * refuted as the recipe (S12);
  * next lever: noLW, which is in the refcv8 launch.
* WM roll-consistency as selector:
  * +5.98 m worse;
  * next lever: goal / learned scorer, or no selector (option B).

---

## C.5 Planned, not finished

| item | spec / package | how far | blocker |
|---|---|---|---|
| v7F S-W→S-T under R1–R6 (option C = B first) | `P4V/V7F_PLAN_R1_R6.md` §3′/§4/§6′ | code merged in a package, 2,655 suite tests pass with the failure set identical to the tip (`P4V/v7f_merge/RESULT.md` §0); CPU dry ladder rc 0 | **unlanded**; PI: release the 08-31 hold, D1 (DINOv3 wrap), the R4 mode; G-HYG ×4, G-CLOCK, F1, F2; a Thor slot |
| `v7f` launch-gate profile | `P4V/v7f_gate/` (52 tests) | rehearsal only | unlanded; G-CLOCK reference; G-SUITE never run |
| `PREREG_V7F` tiny ladder R0→R3 (+LDAD) | `Project Steering/PREREG_V7F.md` §8; `BACKLOG.md` R17 (R0 is 0 GPU) | not started | LDAD code; `--w-trunk-anchor` (`BACKLOG.md` R23) |
| E-WC2-SW latent dump (SEL-1 unblock) | `v6_chain.py:224` | not started | needs a trained S-W |
| v7 training-seed floor | `STATE_AND_GAPS.md` §4 v7F #6 | none; the 14.3 % replicate panel is the REF-C rig (misattributed in CLAUDE.md) | one replicate arm |
| refav1 option T (trunk + trajectory objective) | `P4V/REFAV1_RETRAIN_DECISION.md` | decision FINAL, not built | ~303 GB fp8 cache rebuild + storage (PI) |
| A16: the six frozen-trunk defect classes for v7F | `BACKLOG.md:35`, `ADVISORY_FROZEN_TRUNK_DEFECT_CLASSES.md` | unanswered for v7F | ~1 h, 0 GPU |
| Register the 09-27 v7F findings + PI directives | `P4V/HANDOFF_TO_MASTER_MIND.md` §0, §4 | **not in GAC / PI queue / V7_LAUNCH_GATE** (MEASURED: git grep of "solve the rest" returns only `P4V/`) | doc work |

---

## C.6 PROPOSED v8F (HYPOTHESIS — needs PI authorisation; it changes the flagship architecture)

### C.6.1 The decision the PI must make first

**Option F1 (recommended): build v8F on the proven REF-C chassis (`refc_v3_train.py`) and graft the flagship's latent world model and hierarchy onto it.**

* That chassis has trained twice through the binding gate: refcv7 to 50,400 steps, and refcv8 now.
* It has a working `refcv8` gate profile, the four-family battery, the NavSim bridge and per-frame v9 labels.
* It is the only arm with a separated T1 win over its echo (S6).

**Option F2: train v7F as built (`train_v6_staged.py`, option B).**

* It first needs everything below:
  * landing `P4V/` and closing G-HYG / G-CLOCK / F1 / F2;
  * the PI hold release and D1;
  * an S-W run of ~9 days, ESTIMATED from v6F's MEASURED 26.47 s/step at 30 k steps (CLAUDE.md).
* Its trunk is the transplant that loses content (S6).
* Its v7-tiny ancestors have no admissible T1 read.

**Why F1:** RULE ZERO wants results, and F1's evidence is on its side. Every
measured driving gain in the programme is on the REF-C chassis, and every flagship WM lever has been action-deaf.

**What F1 keeps of the flagship thesis:** the hierarchy (tactical → operative, nav into both, strategic off) and the
WM as an encoder-side scene-evolution learner (S2). **What it drops:** the WM as action-conditioned selector.

F2's unique question, *does a trained-trunk WM become action-sensitive?*, moves to the tiny rig as research and is not on
v8F's critical path.

### C.6.2 v8F definition under F1

| component | choice | why (evidence) |
|---|---|---|
| **chassis** | `refc_v3_train.py` + `refc_v3.py`. Trunk ResNet-101 `a1_in1k` at 416×1024 cylindrical, lift → BEV, 10 cm map head, A9 box head, 117 anchors, per-frame v9 tactical conditioning of generation + selection, `--w-r8-listwise 0.0` | REG §15–16; S12 |
| **warm start** | refcv8 final if it clears BAR-R8-E1/E3/NR, else refcv7-50,400 (`ckpt_50400_full.pt` md5 `d5f104ee…`). New paths zero-gated so step 0 reproduces the parent (refcv8 §3.1 pattern) | `SPEC_REFCV8_DRAFT.md` §1/§3.1 |
| **residual prior** (largest measured lever) | replace NEW-1's `ha0_ext_pose`, which holds a0, κ0 for 6 s and so equals the echo, with a **learned kinematic prior** (P10-style MLP on t0 v0, a0, κ0, fitted on TRAIN) or the **damped prior** (`kd_x` / `damp50`). Keep prior dropout (X2). a0, κ0 at t0 are already consumed by refcv7's prior (PI 2026-09-02: measured state at cycle time is admissible) | S10, S11: echo-type priors are ~1.9 m worse at 6 s than damped ones on the refav1 grid. ⚠️ Transfer to eval139 is UNVERIFIED (experiment E1) |
| **flagship WM (new)** | an operative latent predictor over the pooled BEV/scene tokens. Two-term recipe (O5 L1 rollout to an EMA target + O6 SIGReg, 32 subspaces / 512 slices: the `rdw8p30k` recipe, the only parity-scale arm with participation 25.58 and predictor cos 0.6224), k ≤ 4 at 0.5 s. **Auxiliary only, never a selector**; conditioned on measured state at t0 only | S2–S4, S7, S8; REG §13.0 |
| **hierarchy** | strategic OFF (R6 / PI R5) · nav per frame + turn distance into tactical and operative (R2; **no route checkpoint**, PI 2026-10-11) · ALL v9 tactical labels supervise the tactical layer (R3) · tactical conditions fan + pick (R4) · one 6 s plan (R5) | PI R1–R6 (2026-09-27), R8-1..4 |
| **max speed** (R1) | 4-way road-type label as input (`--r8-speed-input label`, refused outside {30,50,100,120}); **hard cap enforced on the EMITTED plan**, not only the selection filter | PI 2026-10-11 Q-B; `SPEC_REFCV7.md` 26.1: refcv7's ceiling did not reach the emitted plan |
| **selector** | CE-trained, plus a learned **p(lane) / drivable sub-score** trained on map-derived labels (labels/rewards only, never an inference input) | REFe SFT-4: +3.79 [+3.05, +4.54] PDMS from a scorer-only fine-tune (REG §14.2, MEASURED · NAVSIM T1); refcv7 navtest has 2,624 NC/DAC-zeroed tokens vs STOP 732 (REG §15.4.1) |
| **params** | ~100.6 M (refcv8, MEASURED) + WM predictor ~10–20 M (ESTIMATED) ⇒ well under the sub-300 M budget | REG §16 |

### C.6.3 Training stages

* **Stage 0** (0 GPU, dev box): E1 below (floors on eval139).
  * Fit the P10 prior and refuse any prior that loses to `damp50`.
* **Stage 1** (tiny rung):
  * Rig: Thor batch 8, ~4.4–4.6 h/arm MEASURED for the refcv8 ladder (GAC `PI-R8-NOLW`).
  * E2 = prior swap; E3 = WM auxiliary.
  * Each arm runs at seeds 0 and 1 and is read against the rung's own training-seed floor `F = max(|V0r−V0|, |V-R8r−V-R8|)`, per `SPEC_WPB_LADDER.md` §4.
* **Stage 2** (full):
  * Warm start, one full pass over the 4,572-clip train view: refcv8 used 48,852 steps at batch 16, 781,635 windows (REG §16).
  * Fresh optimizer, re-warm 1,000.
  * Eval every 500 steps; gradient-share readings as in refcv8.
* **Stage 3**: a **replicate** at training seed 1 for any claim of a lever effect (`H-ESTIM-SEED-1`).

### C.6.4 Data

* Train view `/home/nvidia/data/refcv8-b1-416x1024-train-4572` on Thor, **NON-PARITY**.
  * ⚠️ `SPEC_REFCV8_DRAFT.md:65` still says refcv7's 4,369 clips. REG §16 (4,572-clip view, PI 2026-10-12 "one full pass and add the missing clips") wins.
* eval139 is disjoint by clip id.
* v9 labels: train md5 `f63ece410b725febb8a5242cf2b01d3c`, eval139 `6b5c7f207cffc3b7eb3cd527fd433599`.
* Speed sidecars: `/home/nvidia/data/refcv6_speed_max_v8_{train,eval}.jsonl` (`launch_gate.py:725-726`).
* SAM3 map GT and obstacle boxes as in refcv7/8.

### C.6.5 Committed bars (proposed; register them before any compute)

| bar | definition | floor / baseline (MEASURED) |
|---|---|---|
| **BAR-V8F-K (new, "vision helps")** | `os − P10_kin_mlp` ADE 0–6 s < 0, separated at both inference seeds. Also vs `damp50` and `kd_x`, on S6 and S2, the same windows | P10 refit on eval139: **not yet computed** (E1). On the refav1 grid P10 = 2.9257 m @ 6 s |
| BAR-V8F-E1 / E3 | inherit BAR-R8-E1/E3: `os − ha0_ext` < 0 separated at 0–2 s and 0–6 s, both seeds | refcv7-50,400: S2 −0.0102 [−0.0260, +0.0067] **not separated**; S6 **−0.9804 [−1.294, −0.6815]** (`SPEC_REFCV8_DRAFT.md` §5.1; `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-11-r7base-fourfamily/`, T1 open loop) |
| BAR-V8F-NR | per family not separated worse than the parent (refcv8 final), **and** \|Δ\| > the training-seed floor before calling a lever effect | refcv8 rows pending (training) |
| BAR-V8F-N2 (decisive NavSim) | navhard official two-stage EPDMS > max(STOP, CV, ECHO), LEGAL inputs | refcv7-50,400 LEGAL **0.1987** vs STOP **0.2985** (Δ −0.0998 [−0.1383, −0.0627], FAIL) (REG §15.4.1) |
| BAR-V8F-N1 / N4 | navtest PDMS > STOP and > refcv7 LEGAL 70.2048 [68.55, 71.82], separated | STOP 61.8202 |
| BAR-V8F-LEAD | LEAD-VIOL `share_of_at_risk` ≤ the echo, with `lead_progress_ratio_clipped` ≥ 0.90 | refcv7 0.0590 (s0) vs echo 0.0434, not separated; clipped progress 0.9539 (REG §15.4.4) |
| BAR-V8F-WM | WM predictor beats a constant **and** `ẑ` beats `z_t` (L3) on scene targets, carried with `nrmse_SHUFFLED` | REG §13.0c rule |
| Frontier reference (not a bar) | REFe SFT-4 navtest PDMS **88.25**; human 94.55 | REG §14.2. Trained on navtrain, so not comparable to zero-shot refcv |

Strategic family: **NOT APPLICABLE, n = 0, layer OFF (R6)** in every table.

### C.6.6 Compute and gate

* **Machine:** Thor is the training box.
  * It is occupied by refcv8 until ≈ 2026-10-20 (ESTIMATED).
  * ⛔ Do not co-locate on it: Thor GPU sharing is net-negative, 3 jobs ran 3.9× slower (memory, INHERITED).
* The dev box (RTX 4060, 8.6 GB) takes E1 and the WM-module unit tests.
* Colab takes small-data GPU jobs only. The 340 GB train cache cannot leave Thor at 1.2 MB/s (INHERITED, memory).
* **Full-run cost:** refcv8 MEASURED 8.94 s/step (G-LIVE), 31.26 GB peak.
  * The WM auxiliary adds future-frame target encodes, ESTIMATED +20–45 %, so ~10.7–13 s/step.
  * That is ~6–7.5 days per pass and **above the PI-approved 10.5 s/step cap**, so it needs PI approval or a cheaper target (EMA BEV tokens, k ≤ 2).
* **Gate:** a new profile `PROFILES["v8f"] = refcv8 profile + the WM flags`:
  * G-DVB: every new flag is declared-vs-built;
  * G-LIVE: gradient reaches the WM group and the prior head;
  * G-CLOCK: against the newly approved cap;
  * G-MAP / G-BOX-OVERFIT inherited;
  * one regression arm per new check.
  * The `refcv8` profile is the template (`launch_gate.py:666`).

---

## C.7 Open RESEARCH topics, ranked by measured effect size

1. **Kinematic prior / damping, 1.9 m @ 6 s.**
   * `damp50 − ha0_ext` −1.9025 m; P10 vs linear −0.50 m (refav1 grid, T1).
   * Never applied to refcv7/8, whose prior is `ha0_ext_pose`.
2. **Selection gap, 2.07 m on turns.**
   * refcv7 S-ROUTE REGRET pick − ORACLE-117 = **2.0718 [1.7426, 2.4477]**: a T0 denominator, 2,317 turn windows (REG §15.4.2).
   * v5f sel_gap 0.2036 (T0).
   * Re-ranking by the model's own logits recovers 0.0040 of 0.1630 m (REG §15.3).
3. **Turn failures under tactical conditioning: 4.6–5.0 m cross @ 6 s** (listwise). Resolved by noLW (S12). The lateral benefit of conditioning is still unshown (S13).
4. **navhard vs STOP: −0.0998 EPDMS** (REG §15.4.1).
   * On REFe, zeroed tokens are 4.15 of a 6.30 PDMS human gap.
   * A scorer-only p(lane) fine-tune gave +3.79.
5. **Trunk quality: 91× (v6F) and +0.205 (transplant).** REFe uses DINOv3 ViT-L + LoRA (88.25). A DINOv3-B/16 trunk inside the 300 M budget is untested on the REF-C chassis.
6. **Perception heads.**
   * Edge IoU 0.042, lane 0.164; box AP@2m 0.248 (REG §15.2, trainer-val, not eval-grade).
   * R8-6 levers are in refcv8.
7. **WM action-sensitivity: action pathway 2–9 % of latent.** Structural per S2. Its value to driving is **unmeasured**, which is exactly E3.
8. **Strategic horizon (P4):** label events have a median of 12.5 s, and the strategic layer is off. Deferred, not dropped.

## C.8 Open ENGINEERING topics

1. **Decide F1 vs F2. Then land `P4V/` (merge, gate, chain, ckpt) or formally archive it.**
   * If anything v6-family ever trains: land F3b first. Every v6 resume on the default schedule is LR-corrupted.
   * Land the nav-mapping fix too. At the tip `--nav-cond` is silently dropped.
2. Record in GAC, `PI_DECISION_QUEUE.md` and `V7_LAUNCH_GATE.md`:
   * the PI's 09-27 R1–R6;
   * "close P3, solve the rest";
   * the SEL-1 correction;
   * nav defects 1–4, F7, F8, F3b;
   * P10 as the programme bar.
3. Put a **VOID banner on REG ~4724–4742**, the v7-tiny T1 table decoded through a random-init readout.
4. Implement `kin_mlp` + `damp50` + `kd_x` as battery arms in `taniteval`. The code exists at `ARI/2026-09-27-refav1-trunk-probe/code/probe_te.py::fit_mlp_kin`.
5. Build a `v8f` gate profile (C.6.6).
6. Write a WM-auxiliary module for `refc_v3.py`, with unit tests:
   * a G-LIVE gradient check;
   * a time-shuffled-target deliberate-regression arm.
7. refcv8 owes model-side `BYPASS_DECLARATIONS` for `r8_rc_to_cond` and the strategic modules (REG §16 quoting stamp). v8F inherits this debt.
8. The speed ceiling must act on the emitted plan (`SPEC_REFCV7.md` 26.1).
9. A16: answer the six frozen-trunk defect classes for any trunk change (`BACKLOG.md:35`).

## C.9 Recommended next 3 experiments

**E1 — floors on the deployed surface.**
* **Hypothesis:** refcv7-50,400 `os` beats P10 at 6 s (`H-V8F-FLOOR`, pre-register both outcomes).
* **Cost:** 0 GPU, dev box, hours.
* **Method:**
  * Fit P10 (8 t0 kinematic features, past-only) on TRAIN windows, score on eval139 S2/S6.
  * Use the paired episode-cluster bootstrap against the banked dumps `D:/refcv7_eval_kit/battery/step50400_a6text/dump_s{0,1}`, plus `damp50` and `kd_x`.
* **Bar:** `os − P10` ADE 0–6 s < 0, separated, both inference seeds; families reported per row.
* **Unblocker:** TRAIN-window kinematics and 6 s futures. Pull poses only (MB) from the Thor train view over LAN (~63 MB/s, INHERITED).
* **Fallback:** grouped 5-fold CV inside eval139, as refav1 §1 did.
* **If NEGATIVE:** the v8F critical lever is the prior (E2), and every "beats doing nothing" row in LEADERBOARD gets re-read.
* **If POSITIVE:** the margin is the programme's first measured "vision helps" quantity.
* **E1b:** the same probe on refcv7's pooled BEV features, `P9 − P10`, the trunk-probe design. Inference over eval139 plus a train subset on the 4060.

**E2 — prior swap at the tiny rung.**
* **Hypothesis:** replacing `ha0_ext_pose` with the learned/damped prior improves S6 ADE and LAT cross beyond the training-seed floor F.
* **Cost:** 2 arms × ~4.5 h on Thor (seeds 0/1), read against the existing V0/V0r.
* **Bar:** paired ΔADE 0–6 s < 0, separated and |Δ| > F, both seeds; no family separated worse.
* **Unblocker:** a Thor slot after refcv8, or a PI preemption; G-DVB for the new prior flag.
* **Fallback lever:** a learned prior head trained jointly with prior dropout (X2).

**E3 — the flagship WM auxiliary at the tiny rung (the thesis test).**
* **Hypothesis:** an encoder-side two-term WM (O5 + O6) on pooled BEV tokens improves the planner's S6 four families beyond F. Its predictor must also beat `z_t` with a shuffled-action control.
* **Cost:** ~1–2 days implementation (ESTIMATED) + 3 arms × ~5 h: seeds 0/1, plus a time-shuffled-target deliberate-regression arm that must NOT help.
* **Bar:** BAR-V8F-NR-style paired deltas > F; WM L3 > 0 separated.
* **Unblocker:** PI authorisation of the architecture change and the cost cap.
* **Fallback:** keep the WM as a zero-weight diagnostic head. v8F then rests on hierarchy + prior + selector, and the WM question returns to F2's research rig.

## C.10 Traps specific to this line

* `flagship4b-phase0-30k` is the **no-speed control**. The deployed v1 is `flagship4b-speedjerk-30k`.
* v1's 0.4522 is a heldout split-mean; the full-set value is 0.4271. Both are **T0**.
* "v1.6" names two runs (§1.4b vs §1.10).
* §1.12 was "closed loop" until 2026-09-02. It is OPEN LOOP. Never quote +7.1642 without its `cl_ha0` decomposition.
* ⛔ **At the tip `--nav-cond` is accepted and then not built.** Any "nav-conditioned" v6/v7 claim from the tip code is false.
* ⛔ **F3b:** a resumed `train_v6_staged.py` run trains at (1+cos(πn/T))/2 of its LR. The LR-replay fix is unlanded (`P4V/v7f_ckpt/`).
* `train_v6_staged.py` `step_s` is **already divided**. Do not divide by `--log-every` (CLAUDE.md).
* ⛔ A v7-tiny T1 number is VOID if decoded via `step_readout_op`: random init in 9/9 arms.
* A low `nrmse` does not mean action-conditioned. Carry `nrmse_SHUFFLED` beside it.
* `lead_gap_m` is a superseded target (C150). Use `lead_present` + `lead_range_m`.
* The 14.3 % replicate panel is the **REF-C** tiny rig, not v7. The v7 line has **no** training-seed floor.
* The refav1 grid (141 eval episodes, 2,399 windows) ≠ eval139 (S2 4,754 / S6 3,668 windows). Never difference across grids, corpora (parity vs non-parity) or NavSim backends (CPU vs CUDA).
* refcv7 "beats the echo" is **6 s only**. At 2 s it is not separated.
* LEAD-VIOL: quote the controls from the same denominator as the headline (all-lead vs at-risk). `SPEC_REFCV8_DRAFT.md` §8.7(b) mixes them.
* EOL: `train_v6_staged.py` is LF at tip. R3's stream copy is CRLF and would flip 11k lines. Land only from `v7f_merge/`.
* `pgrep`/`ps | grep` over ssh self-matches. Thor trainer pid is in `train.pid`; kill by that pid only.

## C.11 File index

1. `Project Steering/MODEL_REGISTRY.md` — §1 (flagship), §12 (v6F readout), §13 (v7-tiny), §14.2 (REFe), §15–16 (refcv7/8).
2. `Project Steering/PREREG_V7F.md` — the v7F pre-registration (H-V7F-1, gates, D1–D9).
3. `ARI/2026-09-03-v7f-design/{DESIGN,SPEC}.md` — design + machine-readable spec.
4. `P4V/HANDOFF_TO_MASTER_MIND.md` — 09-27 state, retractions #1–4, asks.
5. `P4V/STATE_AND_GAPS.md` — v7F defects table, hold items P1–P5.
6. `P4V/V7F_PLAN_R1_R6.md` — stage plan, the SEL-1 correction (§3′), option B.
7. `P4V/v7f_merge/RESULT.md` — the complete R1–R6 change set + nav defects 1–4, F7, F8 (**land from here only**).
8. `P4V/v7f_gate/RESULT.md` — the `v7f` launch-gate profile (unlanded).
9. `P4V/v7f_chain/RESULT.md` — `v6_chain --profile v7f`, F1/F2 open.
10. `P4V/v7f_ckpt/RESULT.md` — exact resume + F3b LR bug.
11. `P4V/v7f_r3/RESULT.md`, `P4V/v7f_r1r4/RESULT.md` — superseded stream copies (read-only history).
12. `P4V/LANDING_READY.txt` — per-path landing map, superseded lists.
13. `P4V/raw/flag_inventory_train_v6_staged.md` — 232 flags, 36 levers default OFF.
14. `P4V/REFAV1_RETRAIN_DECISION.md` — option T.
15. `ARI/2026-09-27-refav1-trunk-probe/RESULT.md` (+ `code/probe_te.py`) — P10 kinematic MLP.
16. `ARI/2026-09-27-refav1-fullgrid-loncomb3/RESULT.md` — planner FAIL-WORSE, damped floors.
17. `ARI/2026-09-27-v7f-s0-b16/RESULT.md` — transplant panel at B/16.
18. `ARI/2026-09-06-v7f-budget/RESULT.md` — EMA teacher in the optimizer.
19. `ARI/2026-09-05-v7f-six-problems/STATUS_BOARD.md` — PI's six v7F problems.
20. `Project Steering/V7_LAUNCH_GATE.md` — the binding 08-31 hold, P1–P5 (stale since 09-02).
21. `Project Steering/V7_RECIPE_AND_SCALEUP.md` — v1.x echo, E-DEC-48b, dissociation, recipe.
22. `Project Steering/GOALS_AND_CLAIMS.md` — D-V7F-* rows (≈7225–7760), E-SEED-2 (227), E-DEC-64 (365), PI-R8-NOLW (16673–16740).
23. `Project Steering/PI_DECISION_QUEUE.md` — v6F retirement (1466), DINOv3 role (1218–1311).
24. `Project Steering/SPEC_REFCV8_DRAFT.md` — bars N1–N4, E1/E3/NR, refcv7 four-family baseline §5.1.
25. `Project Steering/SPEC_REFCV7.md` — NEW-1 residual prior, 26.1 ceiling defect.
26. `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-11-r7base-fourfamily/RESULT.md` — refcv7-50,400 T1 battery rows.
27. `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/RESULT.md` — NavSim bars.
28. `ARI/2026-10-04-refcv8-tactical-conditioning/` — W-PB ladder, noLW (`emit_off_diag_2026-10-12/`).
29. `stack/scripts/train_v6_staged.py`, `stack/tanitad/models/v6.py`, `stack/scripts/v6_chain.py` — v7F code at tip (= `c36b6ddd`).
30. `stack/scripts/refc_v3_train.py`, `stack/tanitad/refs/refc_v3.py` — the proposed v8F chassis.
31. `stack/scripts/launch_gate.py` — profiles refc/refcv6/refcv7/refcv8 (`:432`, `:666`); `stack/ops/runs.d/refcv8-r101-s0.argv.json`.
32. `stack/tests/test_tactical_label_reach.py`, `stack/tests/test_nav_v6stack.py` — pins for D-TLIGHT-1 and nav (the latter pinned the defect's line, not its order).
33. `taniteval/taniteval/lead_metrics.py`, `stack/tanitad/eval/refcv7_loader.py` — LEAD-VIOL instrument, refcv7/8 eval loader.
34. `Project Steering/BACKLOG.md` — A16 (:35), R8/R15/R17/R23/R24 (v7F blockers).
35. `Project Steering/EVAL_DOCTRINE.md` — tiers.

**Conflicts found (the registry or a later source wins):**
* (a) `SPEC_REFCV8_DRAFT.md:65` says refcv8 keeps refcv7's 4,369 train clips; REG §16 says a 4,572-clip view.
* (b) The brief's "splitp30k content +0.3864 with null predictor": +0.3864 is GAC:359's `z_t` panel and REG §13.0d prints +0.3881. "Null" means *adds nothing over `z_t`*; it still beats the constant floor (nrmse 0.8416).
* (c) `V7F_PLAN_R1_R6.md` §6′ "R2 ✅" is superseded by defect 4 + F7/F8.
* (d) REG §1.7 v2corpus "RUNNING" is stale (REG's own 2026-08-16 annotation).
* (e) REG ~4724–4742 v7-tiny T1 table lacks the VOID stamp GAC:8945 requires.


---

<!-- appendix: D_PROGRAMME.md -->

# D. PROGRAMME CONTEXT — goals, status of every line, bars, data, compute, rules, traps

> Source of truth: landed tip `460508ed2f5e120667859e637ac233c28a48d90b` (read via `git --git-dir=C:/Users/Admin/tanitad-push/.git show <tip>:<path>`). Written 2026-10-15, times Europe/Berlin.
> Evidence classes: **M** MEASURED (ours, artifact named) · **P** PUBLISHED (library key) · **P-sec** PUBLISHED-SECONDARY (index summary; primary banked but table not re-read here) · **I** INHERITED (doc/agent/memory, not re-verified here) · **E** ESTIMATED · **UNVERIFIED**.
> Tiers: **T0** teacher-forced (WM diagnostic, never "driving") · **T1** self-action OPEN loop (primary offline tier; never say "closed loop") · **T2** the only closed loop (not provisioned; one NuRec scene panel only) · **NAVSIM** = NAVSIM v1.1 PDMS / v2 EPDMS pseudo-simulation, T1-family open-loop plan scoring.
> Registry beats every other doc (`Project Steering/MODEL_REGISTRY.md`). Quote only registry, raw eval JSON, LEADERBOARD.
> Path shorthand: `…/Research/<dir>` = `TanitAD Research Lab/Architecture & Inference/Research/<dir>` unless the text says `Benchmarks & Evals/Research/` (the refcv5-v2 comparison lives there); `FlyWheels/…` and `products/…` are repo-root directories.

---

## 1. Programme goal in 5 lines
1. **Product**: a sub-300 M hierarchical 4-brain latent world model (strategic / tactical / operative / world-model) that **drives**, plus 8 products (models, data pipelines, DataReconstruction, training, TanitScena, TanitDeploy, TanitEval, DataSetCreator) — `Project Steering/TANITAD_PROGRAMME.md` §1. Thesis (paper v1.3 of 2026-08-29, stale for October): data-efficient, hierarchically-imagining, self-supervised stack with built-in self-knowledge — `Paper/TANITAD_PAPER.md`.
2. **Goals** (`Project Steering/GOALS_AND_CLAIMS.md` lines 25–29): **G1** a model that drives — criterion AMENDED 2026-09-02: "drives" needs **T2**, T1 is open loop; OPEN and *blocked on an instrument we do not have*. **G2** products shipped production-ready — OPEN. **G3** beat published SOTA on community benchmarks (NAVSIM …) via TanitEval — OPEN, in progress.
3. **Where the claim stands** (M, LEADERBOARD "5 October position"): "No TanitAD arm beats every trivial floor it is read against — at any tier, in either loop." Only straight-line (`ha0`) wins are admissible (refcv4b −0.3758, refcv5-v2 −0.3644, refcv3 −0.2304 m, T1, 4,823 windows / 141 episodes); beating hold-action/echo is NOT shown. On NAVSIM the bar is **STOP** (61.82 navtest / 29.85 navhard), not CV.
4. **Ambition**: aim above published SOTA; settle ambition-vs-evidence conflicts with the cheapest pre-registered discriminating experiment, not by shrinking the goal (CLAUDE.md Operating standard 5). Frontier *research* programme: all research-licensed open models/datasets usable; augmented data PRIVATE on the PI's HF Pro account (hard quota).
5. **RULE ZERO (CLAUDE.md, PI 2026-09-05)**: a refutation is a waypoint, not a deliverable — when an arm fails its bar, diagnose, pick the next lever and RUN it in the same turn; you are done only when the result clears its committed bar or every remaining lever is blocked and named. It never lowers the evidence bar (a missed pre-registered bar is reported FAILED, then you continue).

---

## 2. Status table — one row per line (read the registry row before quoting any cell)

| line | what it is | status at tip / 2026-10-15 | best result (class · tier · artifact) | verdict |
|---|---|---|---|---|
| **REF-A refav1** `refav1-b1-v72-ep3-speed` | latent WM + iCEM planner over frozen-DINOv3-feature trunk; 182,459,701 params; Thor; v7.2 labels, 168,910 windows / 4,572 eps (registry §2.4) | ckpt **step 21,109 = 1 epoch**, COMPLETE. Every zero-training planner lever eliminated on the full 141-ep grid. No `refav1` launch-gate profile (§7.3). | `cl` ADE 0.5474 [0.4839,0.6108] vs `ha0` 0.5316: paired `cl−ha0` **+0.0158 [+0.0007,+0.0315] floor WINS** (M · T1 · 282 windows/141 eps · `taniteval/results/refav1-21109-openloop.json`). `loncomb3−ha0_ext` +0.0896 / +0.1515 (both inference seeds) separated WORSE (M · T1 · `Research/2026-09-27-refav1-fullgrid-loncomb3/RESULT.md`). Frozen trunk adds nothing over t0 kinematics: `trunk−kin` +0.0042 m [+0.0016,+0.0071] @2 s (M · T1-equivalent · `Research/2026-09-27-refav1-trunk-probe/RESULT.md`). The WM did learn dynamics: teacher-forced feature MSE 0.4387 vs 0.6173 raw floor, gap growing to 6 s (M · T0 · D-REFAV1-EPOCH-HEADS). | **Planner ELIMINATED; trunk uninformative beyond kinematics.** Next lever = option **T**: retrain the trunk WITH the 6 s trajectory objective, prior = learned kinematic MLP, behind a new gate profile; needs ~303 GB fp8 features or on-the-fly Thor encoding (PI) — `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/REFAV1_RETRAIN_DECISION.md` (I). |
| **REF-D** `stack/tanitad/refs/refd.py` | "SimWAM's training-time prior, our hierarchy, 1/20 the size": frozen Cosmos3-Edge 4 B prior (TRAINING ONLY) → adapter → 3-rate goal-conditioned hierarchy with action tokens, ~180 M shipped | **NEVER TRAINED.** `RefD` has `act()` + `FlowControlPolicy.loss` only — no `forward()`, trainer, loader or gate profile; no registry row (see section B). `REFD_DESIGN.md` calls it "PARKED, TRAINING-READY, 19 tests". | none exists | Design only. Template it echoes: SimWAM 91.5 PDMS with 6 B params on 32 GPUs (P-sec, `2608.07468` via `REFD_DESIGN.md`). |
| **Flagship v7F** (and the proposed "v8F" — the string `v8F` is absent from registry/steering at tip; define it from section C) | 4-brain WM, DINOv3-seeded trainable trunk, EMA teacher, LDAD anti-collapse, 60-step (6 s) control-space plan; `train_v6_staged.py` + `v6.py`; `Project Steering/PREREG_V7F.md` | **NEVER TRAINED** (D-V7F-NEVER-TRAINED 2026-09-05; at tip registry has 0 hits for `v7F`). PI hold 2026-08-31 (`V7_LAUNCH_GATE.md` P1–P5); P3 (drift) ruled NULL; PI 2026-09-27 "close P3, solve the rest"; v6F RETIRED, "the line goes to v7F" (PI 2026-09-20). All "v7 results" are **v7-tiny (~19 M, planner objectives at zero)**. | v7-tiny: no arm has ever beaten its own hold-action control at T1 (P1; M · first T1 read 2026-08-30, 3 arms, 6,924 windows; echo index 0.0000 = honest absence of skill, not fake skill). S0 at the launch geometry (ViT-B/16; step-0 decodability diagnostic, no tier): the cheap repair `seed_imnet_pos0 − seed_asis` −0.0053 [−0.1663,+0.1393] null, replicating the L/16 null; the DINOv3→`ViTEncoder` transplant loses a separated share of ego-speed content (+0.2331) = evidence for PREREG_V7F §10 D1 option A (M · `…/Research/2026-09-27-v7f-s0-b16/RESULT.md` lines 36–57). | **OPEN, unlaunched.** S-T selection is gated by SEL-1, not P2; `--nav-cond` did not build at tip `c36b6ddd` (3 defects fixed in `v7f_merge`, I). |
| Flagship v1…v5f (deployed = **`flagship4b-speedjerk-30k`**, 263.4 M) | trained-ViT 4-brain WM line | v1 FINAL 29,999; v2 abandoned (7,800); v3enc RESTART (10,800); v1.6 tied with v1; v4 jointly-trained planners formal INCOMPLETE / substantively FAIL; v5f COMPLETE 30,000 (2026-08-09) | v1 ADE@2 s **0.4271 [0.3675,0.4871]** (M · **T0** · C1 881 windows / 40 eps · `driving_flagship-30k.json`), paired-tied with REF-C-XL 0.4714 and REF-C-base 0.4728. T2 one-scene panel: flagship v1 vs `cl_ha0` **+7.0745 [+5.12,+9.04] floor WINS**. | `flagship4b-phase0-30k` is the NO-SPEED ablation control (2.918 m), not the deployed v1. |
| REF-B (hierarchical, no WM) | H1/D4 control | v2 FINAL | 0.5913 [0.4766,0.7131] (M · T0 · C1) | control only |
| REF-C family (anchored diffusion → refcv3 / refcv4b / refcv5-v2) | DiffusionDrive-style on the B1 corpus | all FINAL (refcv* at 40,284) | refcv4b ADE 0.2965 [0.2697,0.3280]: vs `ha0_ext` **+0.0091 [−0.0055,+0.0254] TIE**, vs `ha` −0.0032 TIE (M · T1 · `TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-07-refcv5-v2-comparison/raw/refcv5-v2_vs_refcv4b.json`); refcv5-v2 0.3079 and its bar BAR-REFCV5V2-1 FAILS (+0.0205 [+0.0043,+0.0390]) | beats straight line, NOT echo/hold |
| **refcv6** | refcv4b + PI R1–R6 | STOPPED by PI at step 38,250 (2026-09-26 20:40 Berlin) | NavSim 5k/30k subset only | superseded by refcv7 |
| **refcv7-r101-s0** | refcv6 + FIX-1..5 + residual on `ha0_ext_pose` + 10 cm map + A9 box head; 100,468,987 params; NON-PARITY 4,369 clips; commit `fec3a0d` | **FINISHED 50,400** (2026-10-03 20:38 Berlin), one seed | NAVSIM LEGAL inputs @50,400: navtest PDMS **70.2048 [68.55,71.82]**, paired vs STOP **+8.3846 [+6.36,+10.24]** (wins); navhard two-stage EPDMS **0.1987** vs STOP 0.2985, **−0.0998 [−0.1383,−0.0627] LOSES** (M · NAVSIM · registry §15.4.1) | beats STOP on navtest, loses to STOP on navhard; thin map classes and box conf_ratio (0.0059) still red |
| **refcv8-r101-s0** | refcv7-50,400 warm start + tactical conditioning of fan generation + selection, 32 allocated candidates, v9 labels (per-frame tactical 2–8 s + nav; NO route checkpoint), road-type speed label input; 100,598,182 params; 4,572-clip view / 781,635 windows; commit `fb47a58` | **TRAINING on Thor** since 2026-10-15 04:54 Berlin (launch gate 10/10 PASS); 48,852 steps ≈ one pass at 8.84–9.16 s/step ⇒ ETA ≈ 2026-10-20 (E); at 22:01 Berlin step 6,200, healthy (I, from caller) | none yet. Tiny-rung ladder: *no refcv8 arm moves TURN DIRECTION beyond the replicate floor vs V0; the measured gain is longitudinal*; the X1 listwise loss caused the turn regression (confirmed by intervention `V-R8-noLW`) (M · `…/2026-10-04-refcv8-tactical-conditioning/emit_off_diag_2026-10-12/`) | in flight; ONE seed, training variance UNTESTED |
| **REFe** (DriveZero reproduction on NAVSIM navtrain; A40 pod) | frozen DINOv3 ViT-L/16 + LoRA, 4 cams 512×960, 64 proposals + 6-component PDM scorer; 322 M total / 19 M trainable | `model_final` COMPLETE 2026-10-01; SFT-4 scorer-only fine-tune; P2 joint PDM fine-tune GO; S1 aux drivable head NOT PROMOTED; RL-S1 R0 and R0b KILLED by pre-registered K3; P2-s1 seed replicate STARTED | full navtest (12,146 tokens / 136 logs): model_final 83.24 [81.77,84.63]; +route fix 84.46; **SFT-4 B u400 88.25** (+3.79 [+3.05,+4.54]); **P2 88.82** (+0.57 [+0.02,+1.18] vs u400, ONE seed); teacher 89.79; human 94.55 (M · NAVSIM v1.1 PDMS · registry §14.2, D-REFE-P2-R2). Gap is selection: 455/529 zeroed tokens had a safe proposal (D-REFE-ZEROTAIL-1) | open: P2 seed floor (H-REFE-P2-REP-1), RL-S1 (PI item) |
| P2 CEM planner / dynenc line / v5f wedge ladder / v1arch | historical | registry §5, §10, §1.13, §1.9 | see registry | history |

Every number is a single-seed point; "training variance UNTESTED" travels with all of them (`H-ESTIM-SEED-1`).

---

## 3. Open PI decisions + backlog items relevant to refav1 / REF-D / flagship

**Open or standing PI items** (`Project Steering/PI_DECISION_QUEUE.md`, last entry 2026-10-15 ~07:30Z, unless noted)
1. **RL-S1 after outcome O2 (REFe)**: authorise R0c (ONE arm, `--epo-ref armA`, ~3.5 A40-h beyond RL-S1b's budget; MM recommends after P2-s1, before P6) or park. Default if silent = 3 (park) + zero-A40 build. The flag is built and re-verified at zero A40 (landed `ed39928`, `699d231`).
2. **refav1 retrain, option T**: ~303 GB fp8 features vs on-the-fly Thor encoding (PI) + four R1–R6 build questions (κ0 source; cot-absence ruling for v7.2; SPEED_BAND as target while max speed is an input; default prior) — `REFAV1_RETRAIN_DECISION.md` (I).
3. **v7F**: still under the PI hold of 2026-08-31 (`V7_LAUNCH_GATE.md`); open PREREG_V7F decisions D1 (DINOv3 seed option A vs transplant; S0 evidence favours A) and D5 (parameter budget: Config E 336.5 M vs the sub-300 M north star, I); a launch-gate profile for v7F is owed. "Close P3, solve the rest" (PI 2026-09-27) ⇒ P1, P2/P5, P4 must be SOLVED, not held.
4. **Binding PI requirements R1–R6 for refav1 AND v7F (2026-09-27; verbatim in auto-memory `pi-requirements-refav1-v7f-2026-09-27`, I)**: R1 max speed = input AND hard cap (v8 labels, per-clip `limit_kmh ∈ {30,50,100,120}`); R2 nav → tactical + operative at train AND inference; R3 ALL tactical labels train the tactical layer; R4 tactical choices condition the operative plan; R5 one combined tactical+operative trajectory to 6 s; R6 strategic layer OFF in first experiments. Audit at tip (I, `V7F_PLAN_R1_R6.md`): refav1 R2 met, R1 absent, R3 partial, R4 via goal token, R5 plans 2 s, R6 needs architecture change; the pre-registered v7F run (S-W only) trains none of R3/R4/R5 ⇒ not a compliant first run. A design that violates one is not a candidate.
5. **refcv8 requirements R8-1..R8-7 (2026-10-04; `Project Steering/PLAN_REFCV8.md` §0)** and rulings 2026-10-11/14 (PI queue lines 1824–1900): route input = nav command + turn distance only (`--r8-no-rc`); speed input = quantised road-type label; ladder stops at 11 arms; LEAD-VIOL counts only lead windows where the human keeps ≥ 1.0 s; I-0 amendment A1 ACCEPTED; launch with `--w-r8-listwise 0`.

**Backlog** (`Project Steering/BACKLOG.md`) — ⚠ stale head: items dated Jul–Sep, last entry 2026-09-26; `NEXT_ARMS.md` last rewritten 2026-08-26; `LOOP_STATE.md` last updated 2026-07-29 (fleet table describes dead pods). Do not pull blindly; still live for these lines:
* **R3** run `cl_oraclegoal` (T0) beside `cl` (T1): separates "planner cannot search" from "no goal" (refav1; ~1 h dev-box GPU).
* **R6 / R7 / R9** (v7 trainer): `--bptt-truncate` for the O1 stage-A rolls; tactical CE on v7.2 factored labels (needs a `TanitAD_ValidateAIDesign` prereg); strategic ARG encoder.
* **R15 / R17** v7F: DINOv3 seed converter + 3 flags (trunk LR multiplier, trunk warmup, distillation-anchor weight) — partly wired (`D-V7-DINO-SEED`, `D-V7-TRUNK-ANCHOR`); rung R0 (0-GPU) can refute the LDAD line before an arm is spent.
* **A16** answer the six frozen-trunk defect classes (`Project Steering/ADVISORY_FROZEN_TRUNK_DEFECT_CLASSES.md`, D-TRUNK-ADVISORY-1) for refcv6 · v7f · refav1.
* **Open registered hypotheses touching these lines** (`GOALS_AND_CLAIMS.md`): `H-REFAV1-TAC-DECODER-1`, `H-REFAV1-COST-SEED-1`, `H-ARCH-EGOZERO-1` (OPEN, pre-registered 2026-09-03, lines 4869–4873), `H-REFAV1-SURFACE-1` (OPEN, 6385), `H-V7F-1` (proposed in PREREG_V7F §1; no register row), `H-NAVC-1/2/3` (OPEN; H-NAVC-3 FALSE at step 9,500), `H-ENC-2`, `H-RANK-19`, `H-R8-RECIPE-1` (tiny-rung result negative on turn direction), `H-HIST-1` (OPEN O3; next lever = inference-seed averaging), `H-REFE-P2-REP-1` (OPEN), `H-ESTIM-SEED-1` (SUPPORTED; governs everything).

---

## 4. Evaluation & bars — exactly what a new arm must report and beat

**4.1 Tiers and loop** (`Project Steering/EVAL_DOCTRINE.md`, amended 2026-09-02): every number carries T0/T1/T2 *and* loop. T1 supports "open-loop trajectory prediction" claims only; a claim that the model *drives* needs T2 (not provisioned). Never write "closed loop" for T1.

**4.2 Four metric families, per family, never pooled, in addition to ADE** (CLAUDE.md BINDING 2026-08-02): LONGITUDINAL (target-speed accuracy + distance-keeping: headway / time-gap / TTC; LEAD-VIOL per PI), LATERAL (heading, **curvature, yaw-rate**, cross-track), TACTICAL (selected vs executed manoeuvre, confusion, goal/anchor selection), STRATEGIC (decision + goal/route). A family that cannot be computed is `NOT MEASURED` with reason and n, a work item, never dropped. Instruments: `taniteval/` (`four_families`, `taniteval/tools/refav1_paired_delta.py`, `t1_eval.py`, `taniteval/taniteval/lead_metrics.py`); criteria checker = skill `TanitAD_BenchmarkCriteria` over `products/P7-TanitEval/CRITERIA_REGISTRY.json`. Known pitfalls: refav1's route head is an exact bijection of the fed nav token (141/141 = 1.0000), so STRATEGIC route accuracy is unfalsifiable as skill (D-REFAV1-ROUTE-LABEL-IS-THE-NAV); distance-keeping is recoverable at analysis time from the banked lead block `b1_eval_lead_block.npz` (md5 `33a48e15a52eb9dbd69ecd6026fd5023`).

**4.3 Estimators** (CLAUDE.md): decision-grade interval = **episode-cluster bootstrap** (`taniteval/taniteval/ci.py`, B=2000; NAVSIM: log-cluster bootstrap, 10,000 resamples, seed 20260927); two arms on the same windows → the **paired** form, never quadrature. `overlapping_holdout_se` ("8-split jackknife") is forbidden (biases the point −6.67 %…+11.69 %, narrows CI 1.107–3.100×, median 1.499×). Name WHICH of three variances your CI answered: **episodes** (the bootstrap), **training seed** (`H-ESTIM-SEED-1`: a zero-lever replicate `A0b_replicate` read "separated" on 6/42 family cells, 3/14 at the 2 s rows ⇒ a replicate arm is required before any LEVER claim), **inference seed** (stochastic planners: refav1 iCEM seed floor ≈ 0.30 m ADE, D-REFAV1-SEED-GOAL-MISMATCH; refcv7 NAVSIM floors |A1−A1_s1| = 0.1398 PDMS navtest, 0.0061 EPDMS warmup; REFe S1 seed floor 0.352 PDMS). A separated CI from a one-seed arm is necessary, not sufficient. Never quote a learning-curve exponent without fit window, R² (≥ 0.80 else matched-step ratio) and n.

**4.4 Floors an arm must beat (and ACT)**
* B1 / PhysicalAI T1 grid (141 eval eps; 282 windows at stride 40, 4,823 full): `ha0` (straight at measured v0) → `ha` (hold action) → `ha0_ext` (echo) → **`damp50 = 0.5·ha0 + 0.5·ha0_ext`** (beats `ha0_ext` by −0.1576 [−0.2297,−0.0990], M) → **`kd_x`** (damped path re-timed to ha0_ext's distance) → **P10 kinematics-only MLP on 8 t0 features: ADE 0.2229 m @2 s, 2.9257 m @6 s on 2,399 eval windows**, vs damp50 3.45, kd_x 3.56, linear readout 3.43, ha0_ext 5.36 @6 s (M · `Research/2026-09-27-refav1-trunk-probe/RESULT.md` lines 68–91; = AD-MLP `2305.10430` on our data). **A "vision helps" claim must beat P10 with a matched-capacity readout** (A2's "MLP-positive" was kinematics nonlinearity, caught only by A3's control). "Beats the floor" is conjoined with "ACTS": emit mean |a| and the zero-acceleration fraction (D-REFAV1-LON-ACTS: `kamm07`/`l3ladder` improved LON by planning "do nothing" — |a| 0.3131→0.0532/0.0747, zero-accel 47.5 %→77.5 %).
* C1 (881 windows, T0): CV 0.8377, hold-v0 0.7876, **CTRV 0.5265**, best-of-3 oracle 0.4820, ego-status ceiling 0.5735 (LEADERBOARD §0.6; M).
* NAVSIM: **STOP** beats CV on both — navtest v1.1 STOP **61.82 [60.70,63.08]** vs CV 20.65 (12,146 tokens / 136 logs); navhard two-stage STOP **29.85 [27.37,32.53]** vs CV 11.48 (76 logs); logged human 94.55 [93.79,95.23] (single-stage only; undefined on two-stage). STOP gets 41.67 free PDMS points (NC·DAC·TTC ≈ 1; 5/12). Every NAVSIM row needs STOP beside CV. v1 PDMS must come from devkit v1.1 (pin 3e8291b); v2 EPDMS from `autonomousvision/navsim@0a380a9` (+5 patches); never mix PDMS and EPDMS tables; NAVSIM's driving command is a route-level oracle (D-NAVSIM-ROUTE-LEAK) — no route-following skill from a command-conditioned row.
* Oracle gaps are read against a **random-pick control** (refcv7: oracle-in-fan 0.1143 m, selected 0.2772 m, uniform-random 1.5501 m, skill fraction 0.8865; registry §15.3), not "best − actual" alone.
* Probe panels carry (a) a CONSTANT-ONLY control that must read the no-information value exactly, (b) a RAW-INPUT floor, (c) n and d printed; hyper-parameters fit on the FIT split only. A linear negative is not a learnability negative.

**4.5 Gates.** (a) Restart/continue: `Project Steering/GATE_PROTOCOL.md`, `stack/scripts/run_gate.py`; a verdict must name its metric's horizon and n (§0). (b) Launch gate: §7.3. (c) v7 line: `V7_LAUNCH_GATE.md` P1 (beat own hold-action at T1 across the four families with a clean echo control), P2 action-deafness, P3 drift (null), P5 predictor adds nothing over the current latent, P4 horizon ladder short (0.8 s trained vs 12.5 s label median). (d) Pre-registration (`TanitAD_ValidateAIDesign`): SPEC with H-ids, both outcomes committed BEFORE data exists, a deliberate-regression arm that must go RED, known-value control (`X−X` = +0.0000 [0,0]); tiny rig first.

**4.6 A new arm's report must contain**: tier+loop stamp · grid / n / episodes · ADE + all four families paired vs `ha0`, `ha`, `ha0_ext`, `damp50`, `kd_x`, P10 (and STOP on NAVSIM) · estimator and which variance · replicate arm or inference-seed pair for any lever claim · known-value control · leak/echo control (does an inference input contain what the label was derived from?) · registry row + GOALS_AND_CLAIMS update in the same turn · commit / params / argv-sha provenance · checkpoint **step read from the ckpt's own `step` field**, never the directory name.

---

## 5. Frontier reference points (banked: `TanitAD Research Lab/Library/LIBRARY.md`, `products/P7-TanitEval/benchmarks/published_results.json`, LEADERBOARD §0E)

| protocol | reference | number | class · library key |
|---|---|---|---|
| NAVSIM v1.1 navtest PDMS (×100) | human driver | 94.8 (ours on harness 94.55) | P · `2606.07170` Tab.1 p.5 (= `2406.15349` Tab.1 p.7); M for ours |
| | DrivoR / DrivoR+TOAD (4 cams) | 94.6 / 94.7 | P · `2606.07170` Tab.1 p.5 |
| | Drive-JEPA full (ViT/L) / perception-free | 93.7 / 89.0 | P · `2601.22032` Tab.2 p.8 |
| | SimWAM (6 B, 32 GPUs; REF-D's template) | 91.5 | P-sec · `2608.07468` via `REFD_DESIGN.md` |
| | Hydra-MDP | 91.3 | P · `2406.15349` Tab.3 p.9 |
| | Epona 86.2 · World4Drive 85.1 · PARA-Drive / TransFuser 84.0 · LTF / LAW 83.8 · UniAD 83.4 | | P · `2601.22032` Tab.2, `2406.15349` Tab.1 |
| | published floors: Ego-Status MLP / CV | 65.6 / 20.6 | P · `2406.15349` Tab.1 p.7 |
| | **TanitAD**: REFe SFT-4 B u400 88.25 · P2 88.82 · refcv7 LEGAL 70.2048 · STOP 61.82 | | M · registry §14.2, §15.4.1 |
| REFe bar | DriveZero ViT-L, rank-32 Q/V LoRA, navtrain only | 94.55 (Table A13) | P-sec · `2609.06055` (registry §14.2 "Bar"); Tab. A12: 18.58 M trainable / 338.46 M full |
| NAVSIM v2 navhard two-stage EPDMS (×100; OFFICIAL column) | PDM-Closed (privileged) 56.6 · DrivoR+TOAD 56.3 · DriveFuture 55.5 · DrivoR(+134k synth) 54.6 · SimScale 53.2 · DrivoR navtrain-only 48.3 · ZTRS 48.1 · LTF 25.1 · ego-history MLP 14.1 · CV 11.4 | | P · `2506.04218` Tab.2 p.8; `2606.07170` Tab.2; `2605.09701` Tab.1 p.7; `2601.05083` Tab.3 p.6 |
| | DriveZero navhard (Scale) | 57.1 | P-sec · `2609.06055` library index row |
| | **TanitAD floors**: STOP 29.85 · CV 11.48; refcv7 LEGAL 19.87 (loses to STOP) | | M · LEADERBOARD §0E.1, registry §15.4.1 |
| navtest single-stage EPDMS | human 90.3 · HAD 88.6 · DiffusionDrive-V2 85.5 · Transfuser 76.7 · Ego-MLP 64.0 (legacy EPDMS*; inferred) | | P · `2604.03581`, `2512.07745`, `2605.09701` |
| nuScenes open-loop L2 | — | **INADMISSIBLE as a TanitAD criterion** (H-EVAL-6 SUPPORTED; command is GT-derived on every published row; harness version moves the same checkpoint 0.72→1.22 m); ego-status MLP matches (`2305.10430`) | LEADERBOARD §0E.5–6 |
| PhysicalAI open-loop | none published | internal standard `TanitAD_T1_refc_physicalai` (LEADERBOARD §0D) | M |
| closed loop (Bench2Drive, AlpaSim) | — | not provisioned for TanitAD (G1 blocked) | LEADERBOARD §0E.7 |
| WM design anchor | V-JEPA 2-AC back-props ONE recurrent step (T=2); refav1's live `--bptt-truncate 15` is 7.5× the max published depth; v7F default 4 decided by rung R1 | | P · `2506.09985` |

Our vision-only rule forbids ego status at inference while every NAVSIM leader consumes it ⇒ our entry is a paired ego / vision-pure design.

---

## 6. Data & label assets (where they live)

* **Canonical parity corpus**: `physicalai-train-e438721ae894`, **2,376 episodes**, skip-hash `f09e44db`. Parity is sacred: anything that re-selects episodes must be refused. refcv7 / refcv8 / the B1 line are **NON-PARITY** views (4,369 / 4,572 clips) — never compared to parity-corpus arms.
* **B1 corpus** (Alpamayo-labelled, 4,719 clips; D-CORPUS-B1) and the **v7.2 label release**: `labels/s2_labels_v7.2_train.jsonl.gz` 4,572 clips md5 `0a586fe814e08bac81ae18e312d0dc24`; `…_eval.jsonl.gz` **147** clips md5 `bffc9df52fc4e6afd58ec5e509fe2a41` (6 eval clips overlap the deployed val40: not a leak, not fully independent; D-V72-SPLIT). refav1/refcv3/4b/5-v2 grid = the 141 eval clips with pixels; refcv7/8 batteries use **eval139**. State the layer: *published PhysicalAI-AV* has no map/lane/traffic-light/route feature; our *augmented v7.x release* carries GT traffic-light goals with colour on 779/4,572 records — but that label never reaches training (D-TLIGHT-1, pinned by `stack/tests/test_tactical_label_reach.py`).
* **v8 labels (max speed)**: `refcv6_speed_max_v8_{train,eval}.jsonl` on Thor, 4,572 + 147 clips, from `g_tac.goals.SPEED_BAND.v_hi_ms`, buckets {30,50,100,120} km/h (4,572/4,572 vs an independent exact-rational ladder, 0 mismatches). ⚠ `SPEED_BAND.v_hi` is the max of the ego's realised future speed (R² 0.97 recoverable) ⇒ an oracle the PI chose to feed WITH declaration (D-SPEEDBAND-EGO-ECHO-1; D-E16-STAMP-1: `config.json` must carry `speed_max_derivation` or the trainer refuses to start). nuPlan LV+PIT posted limits may supply max-speed *labels* from a non-parity corpus (PI 2026-09-19; `D:/Projects/TanitAD/data/nuplan-maps/`; its `max_speed` column is a constant-25 placeholder, never use).
* **v9 labels (refcv8)**: per-frame tactical goals / actions over [NOW+2 s, NOW+8 s] with constraints + per-frame nav; tactical labels on 95.09 % of 746,946 train windows (refcv7: 23.22 %); `v9_labels_train.npz` md5 `f63ece41…`, `v9_labels_eval139.npz` md5 `6b5c7f20…`; PRIVATE HF `Sayood/tanitad-refcv8-v9-labels` rev `130c0b94`; Thor train cache `/home/nvidia/data/refcv8-b1-416x1024-train-4572` (M · GOALS WP-A rows `H-V9-COV/GEOM/NAV`).
* **SAM3 map GT** (BEV-head ground truth, PI 2026-09-13): 4,719-clip production on Thor `/home/nvidia/sam3map/corpus` since 2026-09-15, auto-publishing to a private HF corpus repo (I, memory `sam3-corpus-production-2026-09`); non-causal map exported on the BEV head's grids incl. fine 600×320 @ 0.1 m (D-SAM3-GT-EXPORT-1).
* **NAVSIM**: navtrain (103,288 tokens / 1,192 logs; 413,152 OpenScene frames; 449.0 GB measured) lives on the **A40 pod** (PI 2026-09-20), not the dev box; navtest (12,146 tokens / 136 logs) is scored on the dev box (`E:/Projects/TanitAD/data/refe_navtest/score/<run>/`, harness CSV sha256s in `counts.json`); navhard bytes on D:.
* **HF (PRIVATE, account `Sayood`, Pro to 2026-12-31, HARD quota)**: `tanitad-refcv7-r101-s0` (5k / 30k / 50.4k), `tanitad-refcv8-v9-labels`, `tanitad-refe-vitl16-navtrain10-grow-final`, `tanitad-refe-p2-joint-pdm`, `tanitad-alpamayo2-augmentation`, `tanitad-ph0-aug120`; Space `Sayood/TanitAD`. `Sayood/tanitad-refc-v3` was PUBLIC (PI-queue item 4; resolution UNVERIFIED). Clip-id guard: `tools/clipid_baseline.json` (530 prefixes / 53 files against the named 4,719-clip list); banked ids stay, prefixes escape a UUID rewrite.
* **refav1 artifacts**: Thor `/home/nvidia/experiments/refav1-b1-v72-ep3-speed/ckpt.pt` (2,122,997,633 B, md5 `1189bc020018c2c67ce03d566c390285`). ⚠ `…/refav1-b1-v72-1ep-21109/ckpt.pt` is a DIFFERENT run at **step 1,000** (175,166,517 params): a directory name is not a step (D-REFAV1-CKPT-NAME). The 141-episode eval inputs live only on Thor (I).
* **Dev-box kits**: `D:/refcv7_eval_kit/ckpt/` (`ckpt_50400.pt` md5 `b418d0fc…`, full `d5f104ee…`, MD5SUMS), `D:/refcv6_eval_kit`, `D:/Projects/TanitAD-artifacts/refe-final-2026-10-01/`.

---

## 7. Compute & landing infrastructure

**7.1 Machines**
* **Jetson Thor** (`tanitad-thor`, 192.168.178.93, user `nvidia`): the training box (refcv7/8, refav1, v7-tiny). **Two venvs, never mixed**: `~/venvs/tanitad-train` (training) and `~/venvs/tanitad-edge` (inference / AlpaSim / TensorRT) — name the venv python explicitly. Unified memory: **only in-process `torch.cuda.max_memory_allocated()` is admissible**; `free` / `tegrastats` / `mem_get_info` / `VmRSS` lie. Throughput is flat across a 6× batch range (20 SMs saturate at batch 8) ⇒ small batches (refcv8: batch 16, peak 31.26 GB). GPU sharing is net-negative (3 GPU-bound jobs ran 3.9× slower, I). Its `/home/nvidia/TanitAD` checkout is STALE (HEAD `30d6d60`, 1,273 commits behind): ship the import CLOSURE of a launch tree; refav1 reads need `PYTHONPATH=<shipped tree>/stack:<…>/taniteval`.
* **Dev box** (Windows 11, RTX 4060 8.6 GB, usable CUDA): eval, probes, CPU analysis, landing. **Project lives at `D:/Projects/TanitAD`** (removable exFAT; the letter can flip to `E:` → `subst D: E:\`; exFAT: no links, 1 MiB clusters inflate many-small-files corpora ~5×). **The G: Drive mount cannot run the stack** (Errno 22, intermittent; reads fail while metadata resolves) — run from an off-Drive clone with `PYTHONPATH=<clone>/stack`, `PYTHONIOENCODING=utf-8`, `venvs/tanitad`; verify `python -c "import tanitad; print(tanitad.__file__)"`. The commit limit binds (63 MiB alloc failed with 11 GB RAM free). Dev box ↔ Thor LAN ≈ 63 MB/s (M); dev box → internet/HF ≈ 1.2 MB/s (M).
* **A40 pod** (REFe; `growing_orange_anaconda`, 7.65-CPU cgroup quota, **no persistent volume — never stop it**; `ssh root@<redacted-ip> -p <port>`, key `~/.ssh/tanitad_pod`; endpoint as of 2026-09-26, UNVERIFIED now). Pods have no git credentials (ship files; verify by real import, `git log` is not proof); `uv pip install` can swap in a torch wheel the driver cannot run (use `--no-deps`, pin torch last, test a real conv2d); a RunPod volume resize reassigns the SSH port.
* **Colab** via CLI (culls detached jobs — keep a heartbeat; uploads ≤ 40 MB chunks at ~1 MB/s; I). **HF Pro GPU / ZeroGPU** only for bursts, quota verified BEFORE launch (unknown ⇒ do not start).

**7.2 Train / eval workflow**: spec + pre-registration → tiny-rig ladder (`TanitAD_ValidateAIDesign`) → trainer flags default-OFF, bit-identical when off, test-pinned → argv assembled to `stack/ops/runs.d/<run>.argv.json` → **launch gate** → supervisor (`stack/ops/sup_refcv7.sh`, `flock`; manifest `runs.d/<run>.env`) → Training Watch → eval through `taniteval` (dumps banked; `--analyze-only` before re-running anything). Trainers: `stack/scripts/refc_v3_train.py` (REF-C family, `--opt dd`), `stack/scripts/train_v6_staged.py` + `stack/tanitad/models/v6.py` (flagship line, stage groups S-W / S-T / S-S / S-J), `stack/tanitad/refs/refa_v1*.py` with adapter `taniteval/tools/refav1_arm.py` (refav1).

**7.3 Launch gate — `stack/scripts/launch_gate.py` (PI 2026-09-26, BINDING: no training step without a PASS)**. Token `PASS_<commit12>.json`, HMAC-signed, bound to (profile, commit, tree_sha256, argv_sha256, data fingerprints); written only if every check PASSes on on-disk evidence (`FAIL_` / `INCOMPLETE_` otherwise); the supervisor runs `verify` / `exec`. Checks: **G-HYG** (configs refuse undeclared attrs), **G-DVB** (declared-vs-built: every explicit flag covered, the built model has the lever), **G-LIVE** (30-step smoke on the REAL argv + trainer: every loss term finite, every declared leaf gets gradient), **G-CKPT** (save→load→resume bit-identical), **G-CLOCK** (tactical label clock ≤ 0.05 s on every window), **G-EVAL** (eval loader builds the identical model, strict 0/0, bit-identical forward), **G-SUITE** + **G-SUITE-PINNED** (pytest on a clean tree of the commit; no skip counted as pass; guard-mutation audit all CAUGHT), **G-MAP-OVERFIT**, **G-BOX-OVERFIT** (refcv7/8). Each has a regression arm that must FAIL. **Profiles at tip: `refc`, `refcv6`, `refcv7`, `refcv8` only — NO `refav1`, `refd`, `v7f` or `v8f` profile** (0 hits of those names in `launch_gate.py`). ⇒ any new refav1 / REF-D / flagship training first needs a profile with required / forbidden values and G-DVB / G-LIVE entries for each lever (`V7F_PLAN_R1_R6.md`: `train_v6_staged.py` has 232 flags, 36 loss/model levers default OFF). REFe has its own `refe/launch_gate.py` (profiles e.g. `refe_rls1c`).

**7.4 Landing** (agents NEVER commit to `main`, never edit `Project Steering/Mission Plan.md`, never commit `<token-file>`): staged work → hold dir `C:/Users/Admin/qland/hold_<name>/{stage,mods,mods-rw,allowdrop,allowdemote,msg.txt}` → `bash /c/Users/Admin/qland/land_hold.sh hold_<name>` (wraps `land.sh`: lander repo `C:/Users/Admin/tanitad-push/.git`, branch `agent/arch-inf-20260803`; aborts unless local tip == remote; **scratch index on local disk seeded from the tip**; `stage/` is ADD-only — a path already on the tip is refused (ADD-OVERWRITE guard) — while existing paths go through `mods/` (append-only) or `mods-rw/` (superset + demotion checked); guards every non-named path; compare-and-swap `update-ref`; push; verify by content; outcomes `ZZLANDED` / `ZZHOLD-NOT-LANDED`). Other routes: `stack/scripts/mm_commit.py` (private scratch index from HEAD, immune to phantom index entries), `scoped_commit.py` (plumbing, never the segfaulting partial-commit path), `commit_cacheinfo.py` (blob-store staging when `git add` page-faults). Verify at the END of the turn by per-path blob assertions (both operands 40 chars); `git log` is not evidence.

---

## 8. Process rules (one line each, with source)
* **RULE ZERO** — a refutation is a waypoint; continue automatically; done = clears bar or every lever named-blocked; stop-and-ask only for unauthorised compute / spend / architecture (CLAUDE.md, PI 2026-09-05).
* **Never idle; ≥ 5 parallel streams** (5–8 band); a report is not work; the re-arm (`ScheduleWakeup` / cron) is not an action; ask "what did I CHANGE?"; priority list ≠ dependency chain (CLAUDE.md, PI 2026-07-29 / 08-02 / 08-03).
* **Registry is the only quotable source; every number carries evidence class + artifact path + tier + arm** (CLAUDE.md "Source of truth", Operating standard 1; programme §6 two-key quotes).
* **Retractions** logged with ROOT-CAUSE CLASS in `Project Steering/RETRACTION_LOG.md` (read before asserting in a known class); `GOALS_AND_CLAIMS.md` updated in the SAME turn as any claim (programme §4).
* **Spec before code, tests before results, both outcomes pre-registered**, deliberate-regression arm + known-value control; instrument changes ship with their regression test in the same commit (programme §3 / §6; `TanitAD_ValidateAIDesign`).
* **Absence needs ≥ 2 probes by different mechanisms**; "0 hits" is a claim about the search unless the file was READ, with a same-breath control (CLAUDE.md Operating standard 2).
* **Vision-only at inference; labels may use ego** (PI 2026-08-03); goal path and situation classifier information-disjoint; nav is GT and first-class, ego-future leakage is not (D-NAV-IS-GT, PI 2026-09-04); v0 at cycle time admissible (PI 2026-09-02).
* **Parity**: never re-select episodes; non-parity arms labelled and never compared with parity arms (CLAUDE.md Invariants).
* **Agent contract** (`Project Steering/AGENT_OPERATING_STANDARD.md`): stage never push; end with a deliverable manifest (every artifact + where it lives); escalate integration instead of "please merge" notes; every brief has a priority order; bank incrementally; fan-out capped.
* **Research banking**: every cited paper is banked — `python tools/kb_add.py <arxiv-id> --tag … --note "…" --cited-by "<path>"`; `--verify` re-hashes; never hand-edit `LIBRARY.md` (CLAUDE.md, PI 2026-08-18).
* **Licensing / compute**: research-licensed open models/data available; augmented data PRIVATE on HF; HF Pro quota is a hard ceiling; local compute is the default (programme §0).
* **Agent model by complexity** (PI 2026-10-04, I memory): set `model` explicitly on every Agent call — haiku = mechanics, sonnet = specified engineering, opus = diagnosis / prereg / novel analysis / adversarial review; re-run any gate a cheap agent was told to pass.
* **Token budget** (I, memory `token-budget-burn-2026-10-11`): the weekly limit burned in ~1.5 days twice; 61 % of cost was re-reading > 300k contexts, 29 % was subagent cache re-writes after ≥ 5 min waits ⇒ agents hand off at ≈ 200k context (write `HANDOFF.md` instead of waiting on a GPU job); Sonnet default.
* **Report times in Europe/Berlin** (I, memory `report-times-in-local-not-utc`); loop prompts must be complete handoffs.
* **Launch-gate PASS before any training; PI go for spend / launch; never add GPU/RAM load to a training pod; never eval on a training pod; `pytest -q` in `stack/` stays green** (CLAUDE.md Invariants; PI 2026-09-26).

---

## 9. Top 12 traps for refav1 / REF-D / flagship work (CLAUDE.md "Traps preflight" unless noted)
1. **`$?` after a pipeline is the LAST element's status**: `mm_commit.py` exit 1 showed as 0 through `| tail`; a timed-out gate printed `GATE_EXIT=0`. Assert on the artifact (file present, content non-empty), not the status.
2. **The G: mount**: `git ls-tree -r` truncates consistently with exit 0; content reads fail while listings resolve; "0 matches" can mean "could not open". Work off-Drive; admit only positive assertions (`git cat-file -e <ref>:<path>`, blob compare with both sides 40 chars); search comma-formatted numbers too.
3. **Stale / phantom shared index**: `D ` can be a phantom (269/269 were), `M ` can be a stale blob that would revert hours of work; a landed commit is not a latch; the CAS guards the ref, not the tree. Classify HEAD / index / worktree blobs before any pathspec-free commit — or use the qland lander / `mm_commit.py`.
4. **`step_s` differs by trainer**: `train_v6_staged.py` (flagship) logs it ALREADY divided (true 26.47 s/step; dividing by 50 again gives a 50× underestimate); other trainers accumulate over `--log-every`. Check which trainer wrote the log.
5. **Thor memory probes lie both ways** (`mem_get_info`, `free` / `tegrastats`, `VmRSS`); only `torch.cuda.max_memory_allocated()`; batch 8–16 is enough; no second GPU job beside a training run.
6. **Directory and log names are not run facts**: `refav1-b1-v72-1ep-21109` is step 1,000; "REF-B v2 died at 22,600" was a stale log; read `ckpt['step']` / `metrics.json`. Thor's own checkout is ancient — ship the closure and verify by import.
7. **A separated CI is not a lever effect**: the paired bootstrap resamples episodes with models fixed; a zero-lever replicate cleared it on 6/42 cells. Read every lever against a replicate / inference-seed pair (refav1 iCEM floor ≈ 0.30 m ADE; a 0.2 m "win" is noise).
8. **Floors**: `ha0_ext` is no longer the strongest do-nothing floor (use `damp50` / `kd_x` / P10); STOP, not CV, on NAVSIM; oracle gaps need a random-pick control; "beats floor" must be conjoined with "acts".
9. **Probe hygiene**: tune λ / PCA on the FIT split only; constant-only control (must read +0.0000) and raw-input floor; n ≪ d reads "no information" by construction; a linear negative is not a learnability negative; add the matched-capacity control (the trunk-probe A2 "positive" was kinematics).
10. **A check that shares the defect it checks is green forever**: re-running the producer's derivation measures determinism, not correctness — use analytic targets, independently authored references and a mutation that reintroduces the real defect. "Declared ≠ built": refcv6 trained 34,500 steps with F3 not reaching the loss; G-DVB / G-LIVE check the THING.
11. **Wrong-scope / wrong-unit numbers read like answers**: price what the CONSUMER's loader opens (`*.v2ep.pt` 34 MB/ep, not the 117 MB raw epcache — a "1.38 TB wall" that never existed); artifacts must carry their units (anchor `controls` = lateral acceleration, not curvature); cylindrical (not pinhole) FOV; a dense windowed tensor duplicates each frame W× (25.6 GB for 5.5 GB — index windows instead).
12. **Process mechanics**: `pgrep -f` / `pkill -f` self-match (kill by PID); a monitor whose filter contains the searched pattern matches its own echo (emit an opaque marker); the trainer and EVERY child inherit the supervisor's `flock` fd 200 (append `200>&-` to all, including `sleep`); `supervise_run.sh` sources its manifest once; write `summary.json {"done": true}` when a run ends or the supervisor resurrects it; an analysis-time import that fails after a long rollout destroys the output — preflight-import and keep the dumps (`--analyze-only`).
*(Also standing: `OMP_NUM_THREADS=6` for multi-arm panels — torch spawns ~113 threads / process; refav1's `nav_cmd` ↔ route head is a bijection, so strategic "skill" under true nav is an echo; SPEED_BAND as input is an ego-future oracle the PI accepted with declaration; `uv pip install` can pull a CUDA-13 torch.)*

---

## 10. File index (repo-relative at tip `460508ed…`)
**Governance**: `CLAUDE.md` · `Project Steering/TANITAD_PROGRAMME.md` · `Project Steering/GOALS_AND_CLAIMS.md` (16,754 lines) · `Project Steering/MODEL_REGISTRY.md` (§2.4 refav1, §6 leaderboard, §14 REFe, §15 refcv7, §16 refcv8) · `Project Steering/RETRACTION_LOG.md` · `Project Steering/PI_DECISION_QUEUE.md` · `Project Steering/BACKLOG.md` (stale head) · `Project Steering/AGENT_OPERATING_STANDARD.md` · `Project Steering/EVAL_DOCTRINE.md` · `Project Steering/GATE_PROTOCOL.md` · `Project Steering/V7_LAUNCH_GATE.md` · `Project Steering/PREREG_V7F.md` · `Project Steering/PLAN_REFCV8.md` · `Project Steering/SPEC_REFCV8_DRAFT.md` · `Project Steering/VOCABULARY.md` · `Project Steering/NEXT_ARMS.md`, `LOOP_STATE.md` (both stale) · `Paper/TANITAD_PAPER.md` (v1.3).
**Eval**: `Benchmarks & Eval/LEADERBOARD.md` (§0E external field, §0.6 floors) · `taniteval/taniteval/ci.py` · `taniteval/taniteval/leaderboard/build.py` · `taniteval/taniteval/lead_metrics.py` · `taniteval/tools/{refav1_arm,refav1_paired_delta,t1_eval}.py` · `taniteval/results/refav1-21109-openloop.json` · `products/P7-TanitEval/CRITERIA_REGISTRY.json` · `products/P7-TanitEval/benchmarks/published_results.json` · `TanitAD Research Lab/Library/LIBRARY.md` + `tools/kb_add.py`.
**refav1**: `stack/tanitad/refs/refa_v1.py`, `refa_v1_plan.py` · `TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refav1-fullgrid-loncomb3/RESULT.md` · `…/2026-09-27-refav1-trunk-probe/RESULT.md` · `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/{STATE_AND_GAPS,REFAV1_RETRAIN_DECISION,V7F_PLAN_R1_R6}.md`.
**REF-D / flagship**: `stack/tanitad/refs/refd.py` (411 lines) · `stack/tests/test_refd.py` · `…/Research/2026-08-19-simwam-analysis/REFD_DESIGN.md` · `…/Research/2026-09-03-v7f-design/{DESIGN,SPEC}.md` · `…/Research/2026-09-05-v7f-state/RESULT.md` · `…/Research/2026-09-05-v7f-six-problems/STATUS_BOARD.md` · `stack/scripts/train_v6_staged.py` · `stack/tanitad/models/v6.py`.
**REF-C / REFe**: `stack/scripts/refc_v3_train.py` · `stack/ops/runs.d/refcv8-r101-s0.argv.json` (+ `.l1_recipe.json`) · `…/Research/2026-10-04-refcv8-tactical-conditioning/` · `…/Research/2026-09-20-refe-plan/` (`refe/`, `eval/PREREG_*`, `READINESS.md`) · `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/`.
**Infra**: `stack/scripts/launch_gate.py` (+ `launch_gate_arms.py`, `launch_gate_refcv7.py`) · `stack/scripts/{run_gate,mm_commit,scoped_commit,commit_cacheinfo}.py` · `stack/ops/sup_refcv7.sh` · landing (off-repo, local disk): `C:/Users/Admin/qland/{land.sh,land_hold.sh}`.
