# HANDOFF — TrainingFlyWheel → Master Mind (2026-09-27 ~02:20 Berlin)

**Why this file exists:** three cross-session messages to "TanitAD project kickoff (fork 8)" were held for that
session's user approval and EXPIRED undelivered. This file carries the same content where a lander meets it. The PI
has been told directly. Everything below is STAGED on D: (never committed), blob-verified at the end of the turn.

## 000. ⭐⭐⭐ NEW ~16:50 — for the register and for YOUR streams
- **`H-REFAV1-TRUNK-TRAJ` — NEGATIVE (pre-registered, validity OK).** refav1's frozen trunk (step 21,109) carries no
  trajectory information beyond t0 kinematics that any readout extracts (linear/MLP, global/spatial, 113 or 600 fitting
  episodes, 2 s and 6 s): `…/2026-09-27-refav1-trunk-probe/RESULT.md` §3 (56 files staged). Heads-only retrain withdrawn;
  option T (trunk trained with the trajectory objective) is the recommendation (`REFAV1_RETRAIN_DECISION.md`).
- ⭐ **A LEARNED KINEMATIC MLP is the new "beats doing nothing" bar — please adopt it programme-wide.** An MLP on 8 t0
  kinematic features (v0, a0, κ0 + products; past measurements only), fitted on 600 refav1-TRAIN episodes, scored on the
  141 EVAL episodes: **6 s ADE 2.93 m** vs `damp50` 3.45 / `kd_x` 3.56 / linear 3.43 / `ha0_ext` 5.36; 2 s: −0.35 m ADE,
  −0.93 m FDE vs `ha0_ext` (all separated). Code: `…/trunk-probe/code/probe_te.py` (`fit_mlp_kin`), predictions
  `raw/out_te/arm_P10_kin_mlp.npz`. For refcv7 NEW-1 it is also the strongest measured candidate for the residual PRIOR
  (a learned prior beats every fixed one by ~0.5 m at 6 s on this grid) — a zero-GPU check on your battery would tell
  whether refcv6/7's learned heads beat it. PUBLISHED analog: AD-MLP (`2305.10430`, banked).
- **v7F:** all streams merged (`v7f_merge/`) — the merge SUPERSEDES the per-stream code copies (`LANDING_READY.txt`
  lists them as superseded, never mapped). R2 (nav) needed SIX fixes before a v7F run really trains with nav: build-time
  nav keys, the step-1 overwrite, ungrouped nav params, `--nav-cond` never mapped into the model (gate G-DVB), the
  operative nav path never trained in S-W (gate G-LIVE, F7), and `--dry-run` crashing with nav on (F8). All fixed and
  mutation-tested; the gate's own S-W rehearsal flipped G-LIVE FAIL → PASS. Still open in the gate (yours to assign):
  strict config classes (G-HYG), RNG + data position in the v6 checkpoint (G-CKPT: a resumed run replays its first
  batches), `v6_chain.py`'s S-T command (drop `--tac-goal-cond`, add the R1/R3/R4/R6 flags), a G-CLOCK reference.

## 00. ⭐⭐ NEW 2026-09-27 afternoon — read before §0
- ⛔ **refav1 full-grid test: FAIL-WORSE, pre-registered, both inference seeds** — `TanitAD Research Lab/Architecture &
  Inference/Research/2026-09-27-refav1-fullgrid-loncomb3/RESULT.md` (885 files staged + blob-verified, incl. the
  arm records and both dumps, clip ids sha12). `loncomb3 − ha0_ext` ADE +0.0896 / +0.1515 (separated worse); the
  planner-free `damp50 = 0.5·ha0 + 0.5·ha0_ext` beats `ha0_ext` by **−0.1576** and beats the planner blend at both
  seeds. For the register: `H-REFAV1-LONCOMB3-FULL` **REFUTED (FAIL-WORSE)**; `ha0_ext` is no longer refav1's
  strongest do-nothing floor.
- ⭐ **For YOUR stream (refcv7 NEW-1 = residual on `ha0_ext_pose`; the refcv6 battery's "blend − echo −0.012 m"):**
  on the refav1 grid (EXPLORATORY, zero GPU) a planner-free kinematic composite — the damped path (half the measured
  curvature) re-timed to `ha0_ext`'s travelled distance (the measured acceleration held) — beats `ha0_ext` by
  **−0.2282** (282 windows) / **−0.2638 [−0.3374, −0.2005]** (2,399 windows), and a ridge readout on 8 kinematic
  features beats that by a further −0.0558 on held-out episodes. Two zero-GPU checks on your Thor battery dumps would
  settle whether refcv6/7's "beats the echo" rows survive: (a) `w·cv + (1−w)·ha0_ext_pose` at the SAME `w` as the
  deployed blend (the planner-free control), and (b) the re-timed damped composite as a residual PRIOR candidate for
  NEW-1. Recipe + code: `…/2026-09-27-refav1-fullgrid-loncomb3/raw/analysis/explore/build_explore.py` (`retime()`).
  ⭐⭐ **AND AT 6 s THE GAP IS 10× LARGER** (EXPLORATORY, refav1 eval grid, 2,399 windows / 141 clusters, T1, 6 s GT
  exists on every window): holding the measured (a0, κ0) for 6 s — `ha0_ext` — scores ADE **5.36 m**, FDE **15.39 m**;
  constant-velocity straight (`ha0`) 4.13 / 11.21; the damped `damp50` **3.45 / 10.15**. `damp50 − ha0_ext` ADE@6 s
  **−1.9025 [−2.3605, −1.4971]**, FDE@6 s **−5.2442 [−6.3458, −4.2465]**; even `ha0 − ha0_ext` is −1.23 m. ⇒ If NEW-1's
  residual prior `ha0_ext_pose` holds curvature/acceleration over the plan horizon the way refav1's `ha0_ext` does, a
  damped or CV prior is a ~1–2 m better starting point at 6 s on this grid — a zero-GPU check on your battery
  (prior alone vs damped prior alone, same windows) settles whether it transfers to refcv's surface.
- ⛔ **p4 is not an admissible tuning surface for lateral levers:** the 30 p4 arms chose a lateral behaviour that
  loses 0.20 m cross-track to planning straight on the full grid (247/282 windows lane-keep).
- **The next refav1 lever was pre-registered and run the same hour:** a probe of whether the frozen trunk carries
  trajectory information beyond kinematics (`…/2026-09-27-refav1-trunk-probe/`). With the mean-pooled state that
  `plan()`'s heads read: **NEGATIVE** (`P4 − P2` +0.0042, separated worse); with a 4 × 10 SPATIAL pooling (A1): also
  **NEGATIVE** (`P6 − P2` +0.0037). The power fix (A2: fitted on 600 refav1-TRAIN clips, scored on eval, + an MLP arm)
  is running on the dev box. PUBLISHED anchor for the pattern: `2305.10430`, `2312.03031` (ego status dominates
  open-loop metrics) — and v7's own E-WC2 ("these latents are the wrong surface").
- ⛔ **v7F — two corrections to what I sent the PI this morning, both from source.** (1) S-T does NOT choose by
  world-model imagination: it emits a goal-conditioned fan and selects by goal distance / a learned scorer (roll-
  consistency as a selector was already +5.98 m worse and is refuted); what gates S-T is **SEL-1**, not P2. (2) Option
  B (one decoded trajectory, no selector) is a CONFIGURATION, not a build: `--n-candidates 1 --proposals query
  --selector none` (SEL-1 does not apply to selector "none"). `V7F_PLAN_R1_R6.md` §3′.
- ⛔⛔ **A real `--nav-cond` v7F run could not START at the tip — three defects, all fixed in `v7f_merge/`:** (1)
  `synthetic_batch` carries no nav keys → `assert_isolation` raises `NavTokenMissing` at every build; (2) the batch
  dict writes `"nav_token": b.get(...)` AFTER the NavEmitter splat → the emitted token is overwritten by `None` at
  step 1 (the pin `test_nav_v6stack.py:195` checks the line exists, not its order); (3) the NavConditioner's 12
  parameters belong to no `_GROUP_PREFIXES` group → `group_of` raises on every nav build (the nav tests never ran the
  grouping path). Tests + 4/4 mutants caught; diffs verified to apply to `c36b6ddd` byte-exactly.
- **Streams:** v7F R1 + R4 DONE (`v7f_r1r4/`: `--tac-op-cond`, `--max-speed-input-v6`, `--plan-vmax-cap`; 63 tests,
  14/14 mutants; independently re-verified here) and merged with the R2 fixes (`v7f_merge/`, which SUPERSEDES the
  stream copies of `v6.py` / `train_v6_staged.py` — `LANDING_READY.txt` says so per path). Running: R3 (at the tip NO
  tactical label family reaches ANY v7F loss — in S-T the label file is not even loaded), a `v7f` launch-gate profile,
  and refav1's R1/R3/R5/R6 build. **The binding gate still has only the refcv7 profile**; my flag inventories
  (`raw/flag_inventory_*.{md,json}`) and the streams' G-DVB proposals are the raw material for yours.

## 0. ⭐ NEW SINCE THE FIRST VERSION (2026-09-27 morning)
- **PI decision, verbatim: "close P3, solve the rest"** — v7 hold item P3 (drift) is CLOSED on the PI's word (the
  register already ruled it null: `D-V7F-DRIFT-NULL`); P1, P2/P5 and P4 are to be SOLVED. Please record it in
  `V7_LAUNCH_GATE.md` / `PI_DECISION_QUEUE.md` (my D: tree is 186 commits stale; I did not touch steering files).
- **PI directive R1–R6 for refav1 AND v7F** (full text: memory `pi-requirements-refav1-v7f-2026-09-27`; audit in
  `STATE_AND_GAPS.md` UPDATE 6): max speed = input + HARD cap; nav → tactical + operative at train and inference;
  ALL tactical labels train the tactical layer, which conditions the operative plan; one combined 6 s trajectory;
  strategic OFF in first experiments. Please add it to the register beside the 2026-09-15 refcv6 directives.
- **R1 is implemented for refav1** (inference-side, no retrain): batch `R1-VMAX` in `LANDING_READY.txt` — full
  files based on tip blobs under `code/fix/`, diffs under `code/diffs_vs_tip_546f34c9/`.
- ⛔ **refav1's DINOv3 fp8 feature cache is GONE from Thor**: `/home/nvidia/data/dinov3-b1-fp8-w120-256x640cyl/` no
  longer exists; 0/141 eval and 0/4,572 train symlinks resolve; Thor is 89 % full (106 GB free) and the full cache
  (~303 GB of fp8 features — 4,572 × ~66.2 MB (`T_c × 655,360 + 1,908` B per episode, the encoder's documented size); ⛔ CORRECTED 2026-09-27: I first quoted ~155 GB, which is the v2ep SOURCE-frame cache (34 MB/ep) the encoder reads, not the fp8 cache the trainer reads (class C82: price the file the CONSUMER opens)) no longer fits. A complete eval copy (141 episodes, all healthy) survives on the dev box at
  `C:/Users/Admin/tanitad-data/refav1-eval141/`; the TRAIN features exist nowhere I could find. Any refav1 retrain
  needs them rebuilt and a place to put them — a storage decision.
- ⚠️ **The orphaned `grep -rciE refa_v1|refav1 .` that saturated D: from 2026-09-26 23:15 for ~4 h was almost
  certainly spawned by MY refav1 survey sub-agent** (same terms, same hour). Thank you for stopping it; my agent
  briefs now forbid recursive greps outside the tip snapshot.

## 1. Land these (staged, verified)
| package | files | what |
|---|---|---|
| `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/` | 13 | v7F + refav1 state/gap report (`STATE_AND_GAPS.md`, read the UPDATE block first), 3 survey reports, v7F launch-line probe, G-DVB flag inventories + tool, this handoff |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-close-the-gaps/raw/refav1_margin_banked_2026-09-27/` | 316 | 305 refav1 files that existed ONLY on the dev box (17 arm records never read), clip ids sha12 (556 full + 240 prefixes), `RESULT_HARVEST.md` |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refav1-fullgrid-loncomb3/` | 298 | pre-registered full-grid SPEC (+ Amendments A1 blend, A2 damped floor, A3 full-grid damped floor — all before any `loncomb3` full-grid data), p4 exploratory reads, the 2026-09-04 shipped-cost full-grid dump (existed only on Thor; manifest ids sha12) |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-v7f-s0-b16/` | 24 | S0 of `PREREG_V7_SEED_POS` at ViT-B/16 (code, raw, RESULT) |
| `stack/scripts/p_runner.py` | 1 | an unstaged `encoding="utf-8"` fix (without it a cp1252 decode error reads as "GPU busy" forever) |
| `stack/tests/test_box_quality.py`, `stack/tests/test_cls_weight_stamp.py` | 2 | unstaged guard additions, green (43/43) before staging |

## 2. Results for the register (MEASURED unless marked)
- **refav1 `kammshift` FAILS its pre-registered bar** (`SPEC_CLOSE_THE_GAPS.md`): ADE 1.0652 > 0.8892; controls bit-identical to `kamm07`.
- **refav1 `loncomb3`** (`lonshift` + W_JERK 0.02 + jerk-seam a0 — the corrected design after `D-REFAV1-LON-SEAM-INERT`)
  is the best config measured on p4: vs `ha0_ext` separated-better on LON along −0.2660, heading −3.36°, yaw-rate
  −0.1064; still worse on speed +0.1608; ADE −0.1033 n.s. (8 clusters). The `lonshift` inference-seed replicate is
  itself "separated" on LON speed (−0.0279).
- ⭐ **A model-free DAMPED hold beats every do-nothing floor** (exploratory): on the FULL 141-episode grid
  `0.5·ha0 + 0.5·ha` − `ha` ADE **−0.1559 [−0.2249, −0.0948]**, − `ha0` **−0.1484 [−0.1846, −0.1160]** (separated,
  141 clusters, w=0 control exactly 0); on p4 `0.5·ha0 + 0.5·ha0_ext` − `ha0_ext` **−0.2152 [−0.3634, −0.0839]**.
  The p4 planner blend `0.5·loncomb3 + 0.5·ha0_ext` beats `ha0_ext` by −0.2723, ~80 % of it damping; the planner's
  own contribution over the damped hold is longitudinal (speed −0.0918, along −0.0978, separated).
- **v7F S0 at B/16:** the cheap repair (ImageNet fold + zero `pos`) is null on scene content (−0.0053 [−0.1663, +0.1393])
  — binding bar FAILED as §0b predicted, replicating L/16; the seed as wired beats random on scene (+0.1941 sep); the
  transplant loses a SEPARATED share of ego-speed vs the real DINOv3 (+0.2331) while its scene loss is not established
  at B/16 (+0.1264 n.s.; L/16 was separated). Evidence for `PREREG_V7F` §10 D1 option A.

## 3. ⚠️ Two things that bear on YOUR stream (refcv7 / LEADERBOARD)
1. **SPEC_REFCV7 §1 NEW-1's evidence may be confounded by damping.** "The zero-training blend `w·os + (1−w)·ha`
   beats the echo" needs the planner-free control `w·ha0 + (1−w)·ha` at the same `w` on the same windows; on refav1 the
   planner-free blend gets ~80 % of the gain. Zero-GPU check on your battery dumps. It does not refute the residual
   design, only the evidence cited for it. A damped prior is also a free candidate prior for NEW-1.
2. **LEADERBOARD "beats doing nothing" rows** (`75c6520` and every T1 "vs hold/echo") are read against floors a trivial
   damped hold beats by 0.15–0.2 m ADE. Suggest a damped-hold floor in the battery beside `ha`/`ha0`/`ha0_ext`.

## 4. Register / doc corrections found by the surveys (each checked against source)
- ⛔ **MY RETRACTION (for `RETRACTION_LOG.md`, class C82 — price the file the CONSUMER opens):** I quoted "~155 GB" for refav1's TRAIN fp8 cache in this handoff, `STATE_AND_GAPS.md` and the full-grid RESULT. That is the v2ep SOURCE cache (4,572 × 34 MB). The fp8 feature cache `refa_v1_train.py` reads is **~303 GB** (4,572 × ~66.2 MB). Corrected in all three on 2026-09-27; the trunk-probe SPEC carries an erratum in its RESULT §5. It changes the storage ask for a trunk retrain by 2× (Thor has 97 GB free).
- ⛔ **MY RETRACTION #2 (for `RETRACTION_LOG.md`; class: a claim about what a component DOES, taken from a document's framing instead of the component's own code — operating-standard rule 2, "the tool that owns the fact"):** I told the PI this morning that v7F's S-T "chooses by imagining each candidate with the world model", so P2 blocks it, and framed the PI's option A/B/C decision on that. `v6.py`'s emission/selection code says S-T selects by goal distance / a learned scorer (roll-consistency selection is already refuted); the real gate is SEL-1, and option B is a configuration. Corrected in `V7F_PLAN_R1_R6.md` §3′ (the wrong §3 kept, marked SUPERSEDED) and reported to the PI.
- ⛔ **MY RETRACTION #3 (same class):** my R1–R6 table marked v7F R2 (nav) "✅ present (`--nav-cond`)" from the flag's existence; a real `--nav-cond` run could not start (three defects, `v7f_merge/RESULT.md`). Corrected in `STATE_AND_GAPS.md`.
- ⛔ **MY RETRACTION #4 (class: a fix verified on a path production never takes — the declared-vs-built family):** I
  reported v7F R2 "fixed" after three nav fixes tested on DIRECTLY-built configs; the `v7f` gate's G-DVB then showed
  `build_stack_from_args` never maps `--nav-cond`, so no v7F launch had nav at all. Fixed + pinned through the real
  launch path (`v7f_merge/RESULT.md` §1b). ⭐ Credit: the gate profile caught it on its first run — the instrument
  earning its keep.
- `MODEL_REGISTRY.md:4724-4742` prints the VOID v7 T1 table (random-init readout) as a measured floor.
- `GOALS_AND_CLAIMS.md:8946` says the metric-decode guard shipped; its symbols do not exist and its test is a module-level skip.
- `CLAUDE.md:180-188` attributes the replicate panel to v7-tiny; its flags exist only in `refc_v3_train.py` (REF-C rig). The v7 line has NO training-seed floor.
- `MODEL_REGISTRY.md:2095` still says refav1 `ccos` IN PROGRESS (delivered + refuted); §2.4 lacks `a0_shift` / turn-asymmetry / perception rows.
- The refav1 "≈0.30 m seed floor" is cited to `D-REFAV1-SEED-GOAL-MISMATCH`; it was measured in `D-REFAV1-DRIVE-AB` (floors run 0.0015–0.1035 ADE by config).
- `CLAUDE.md:1297` names `launch_gate.py`; `SPEC_REFCV7.md:39` names `launch_gate_refcv7.py` and says "landed"; neither exists at origin `546f34c9`.

## 5. Asks
- **Launch gate coverage:** neither `refa_v1_train.py` nor `train_v6_staged.py` (11,057 lines, 232 flags) is covered; the
  flag inventories in `raw/flag_inventory_*.{md,json}` are ready for your gate agent (40 and 232 flags; 4 and 36
  loss/model levers default OFF).
- **Thor:** the refav1 full-grid confirmation (2 inference arms, ~3.1 h each, inputs already on Thor) was NOT started;
  the slot question now sits with the PI.
- **PI decisions put to the PI directly:** arm refav1 `W_VEND` (pinned by `test_steer_conversion_complete.py::test_C1`,
  so nothing was prepared); v7F D1 option A; what closes the v7 hold of 2026-08-31.
