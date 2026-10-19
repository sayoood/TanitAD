# PREREG_NAV_FEED -- does feeding refcv8 the nav it was TRAINED on change its plan? (sealed 2026-10-20, BEFORE any GPU number of this experiment)

Author: EvalFlyWheel engineering agent, for the Master Mind. Seal: `PREREG_NAV_FEED.md.sha256` (sha256 of this file's bytes). `run_navfeed.sh` and
`nav_feed_analyze.py` REFUSE to run if the seal does not verify, and the wrapper additionally refuses unless every sealed code file hashes to a value
listed in section 10. The label is a function of this text and of the dumps; nothing else.

Tier of every number here: **T1** (self-action, OPEN LOOP on logged eval139 frames). The nav token and the nav args fed are **ORACLE-derived** (the v9
labels are built offline from the ego's future path). This experiment is therefore a train/eval-consistency instrument, **never "driving performance"**.

## 1. Question and hypothesis

**Q.** refcv8 was TRAINED on the per-frame v9 nav token plus the 6-vector nav args; the battery roll FEEDS it the clip-level v7 token and no args. Does
repairing the eval feed change the plan of the EMA-40k checkpoint, and by how much, on the windows where the two feeds differ?

**H-NAVFEED-1 (committed prediction, written before any run).** On the 1,157 mismatch windows, `perframe_full - clip` ADE 6 s is negative (the plan gets
closer to the logged future) at BOTH inference seeds, separated, and larger than the inference-seed floor. The prior is the zero-compute natural
experiment below (token half only); I expect MATTERS. If it does not occur the label says so (sec 6) and sec 7 gives the next lever.

Evidence classes of what this rests on:

| fact | value | class and artifact |
|---|---|---|
| training feed | per-frame v9 token (`refc_v3_train.py:4665-4668`) + `r8_nav = scale_nav(nav_args, known)` (`refcv8_train.py:367-378`), `known` dropped p=0.5 in training only (`:560-567`) | MEASURED by reading, launch tree `C:/lgs/Lfb47a58376` (fb47a58) |
| eval feed | clip-level v7 token `refcv3_arm.py:2064-2071`, fed at `:2358`, banked `:2502`; `r8_nav` never passed | MEASURED by reading, same tree; INHERITED trace `2026-10-19-refcv8-nav-trace/NAV_TRACE.md` (sha256 6793f1f1187829887941aef5ce312f5436ad6abd66c909581ae224c561cc75a4) |
| mismatch | fed token != v9 per-frame token on 1,157/4,754 windows (24.3 %), 59 episodes; confusion [v9 row][fed col] = [[2909,280,739],[26,158,29],[76,7,530]] | MEASURED by the trace agent AND re-derived independently here through the real `R8LabelJoin.item` path (`raw/windows_navfeed.json` `counts`, `logs/make_windows.out`; test `test_b_whole_archive_reproduces_the_trace_agent_confusion_matrix`) |
| v9 args valid | 2,802/4,754 windows (58.9 %) | MEASURED, trace; re-derived here (same test) |
| natural experiment | banked rows whose shuffled token equals the v9 token: ADE 6 s -0.524 [-0.796, -0.237] on 662 windows / 58 episodes (token half only, args absent in both rows, non-random subset) | INHERITED, NAV_TRACE sec 4 (`raw/nav_natexp_step40000ema.json`); NOT re-verified here; it is the PRIOR, not evidence for this label |

## 2. The fix under test (opt-in; the default is byte-identical)

`--nav-feed {clip, perframe_token, perframe_full}`, default `clip`. Implemented as an OVERLAY: nothing under the launch tree or the battery package is edited.
`code/overlay/navfeed_overlay.py` exact-anchor-patches the in-memory source of `taniteval/tools/refcv3_arm.py` (each anchor must occur exactly once or the
overlay refuses), executes it with the original `__file__`, and registers it as `r7_roll._RA3`. `code/overlay/navfeed_roll.py` is the child-process roll.
Changes, as launch-tree line numbers of the ORIGINAL file (diff: `raw/refcv3_arm_navfeed.diff`):

* `refcv3_arm.py` after 2071: `nav_true/nav_valid` replaced by the v9 PER-FRAME token (and `nav_valid = all True`, as in training `:4668`) when `--nav-feed != clip`;
  the clip token is kept as `nav_clip`; shuffle/flip are then derived from the per-frame token. Mirrors `refc_v3_train.py:4657-4668` through
  `ds.r8_join.item(sid, t+W-1+raw_offset, now_s)`; it REFUSES (no silent fallback to the clip token) if the join is absent.
* after 2212 (`item = ds[wi]`): `assert_item_matches` (the item's own `nav_cmd` must equal the token we feed).
* 2358-2360 (the main fed forward only): `**model_kw(...)` adds `r8_nav [B,6]` for `perframe_full` ONLY (fed row on the `nav_true` slot, zeros elsewhere, as `r8_nav=None`
  is the unknown row `refc_v3.py:1968-1973`). The `nav_null` forward is untouched.
* after 2502: sidecar rows `nav_cmd_clip`, `r8_nav_known`, `r8_nav_fed`; after 2696: manifest block `nav_feed` (only when not `clip`); after 3765: the argparse flag.
* The ONLY removed line in the whole diff is `**_geom(...))` -> `**_geom(...),` (a trailing comma). With `clip` the new branches are not taken: `decisions.nav_cmd` equals the banked
  clip token (tests (a)). Note the launch-tree file has CRLF line endings; the overlay patches the LF-normalised text, sha256 of the file bytes `ef9b5b2562fdbef134ddd862541bd3152ff67961511f082cc175f0778f5dc20c`,
  of the normalised text `a45677cc8f8accda25ec45882eb2f7daa951873af3c502c7c9d905cc6b4eab07`, of the patched text `64e6ffbb1d0434d13badf82f39daf46608ad8404e2ec9e8797c375be5e5df543`.

CPU tests (literal expectations, real launch-tree source and real v9 join): `code/tests/test_navfeed.py` 24 passed (a clip default reproduces the banked `decisions.nav_cmd`;
b `perframe_token` reproduces the v9 token on 9 mismatch windows incl. 2 opposite-turn; c `perframe_full`'s `r8_nav` equals `scale_nav` on a hand-built example, log1p(120/50) = 1.2237754316221157;
d the mutation arms `flag_dropped`, `clip_under_flag` and a join-level clip join turn (b) RED, `logs/pytest_navfeed_mutation_*.txt`); `test_nav_feed_analyze.py` 21 passed (floor-gate
removal turns the truth table RED, `logs/pytest_nav_feed_analyze_mutation_floor_gate_removed.txt`); `test_wrapper_flow.py` 14 passed (stub-tool sandbox of the REAL shell code; three guard-mutation arms).

## 3. Design

* **Checkpoint:** `E:/refcv8_eval_kit/ckpt/ckpt_ema_40000.pt`, **md5 57807456bd7169800be159ff95871d22** (asserted by the wrapper under the lock; `ckpt_40000.md5` sidecar). Config
  `E:/refcv8_eval_kit/ckpt/config.json` (sha256 796fc41520e4f5d9e7a45e4599dd514dbfb143841ca3dd3aa007d03804d350b7). Arm rolled: `os` only (the true-nav forward; `full=False`).
* **Arms (3) x inference seeds (2) = 6 rolls:** `clip` (the battery feed; also the replicate/control arm), `perframe_token` (token fixed, args absent), `perframe_full` (token + args = what training saw).
  Seeds 0 and 1. Roll order: (clip,0), (full,0), (clip,1), (full,1), (token,0), (token,1) so the primary pair completes first.
* **Windows (sealed, `raw/windows_navfeed.json`, sha256 in sec 10):** the MISMATCH set = every window of the archived `step40000ema_unc/dump_s0` whose banked `decisions.nav_cmd` differs from the v9
  per-frame token at row `k = ws + 2` (positive `row__k == k` assertion per window): **1,157 windows / 59 episodes**. The CONTROL set = **300 windows whose fed token EQUALS the v9 token**, stratified
  100 FOLLOW / 100 LEFT / 100 RIGHT, drawn with `numpy.random.default_rng(20261020)` over candidates sorted by (episode_index, ws); zero overlap with the mismatch set; 95 episodes.
  Total 1,457 windows, rolled in ONE call per arm and seed in the same episode and window order, so the dumps are index-aligned. v9 source: `v9_labels_eval139.npz`, md5 6b5c7f207cffc3b7eb3cd527fd433599.
  Episodes are identified by `episode_index` and `sha12`; no raw clip id appears in any deliverable.
* **Common random numbers:** DDIM sampling is stochastic; arms rolled with the same seed and the same window order share their noise stream, so within-seed contrasts are low-variance. This is also why the
  token arm on the CONTROL windows should be bit-identical to `clip` (reported as `plans_bit_identical_to_clip`). CRN is a variance device, NOT evidence: the floor F below is.
* **Pose clock:** `R8_NO_POSE_SYNC=1` (the standalone clock the archived tree was rolled on). `OMP_NUM_THREADS=6`.
* **Inference variance:** the first-class unknown. Seeds 0 and 1 are read SEPARATELY and never pooled.

## 4. Metrics (all four families; ADE is one row of them)

Computed per window from the dumps (`nav_feed_analyze.window_metrics`; geometry from `taniteval.four_families._seq_geometry` and `refav1_arm._components`, the programme's canonical code), ego frame of `pose_last`.
Plan slots at 0.5, 1, 1.5, 2, 3, 4, 5, 6 s. Validated during development on the archived dump (MEASURED; the scratch log was not banked, the unit tests re-pin the geometry): FIX-subset mean ADE 6 s 2.103 reproduces the trace; GT-cruise n=1,739 reproduces the banked literal; frame check 7.7e-6 m.

| family | metric | definition |
|---|---|---|
| ADE | `ade_2s`, `ade_6s` (PRIMARY) | mean L2 over the first 4 slots / over the valid slots up to 6 s |
| LONGITUDINAL | `LON_speed_mae_mps`, `LON_along_abs_final_m`, `LON_along_signed_final_m`, `LON_false_decel_on_gt_cruise` | S6 grid (instants 1..6 s), windows with all six GT instants valid; false decel = plan speed over the last second minus measured v0 <= -1.5 m/s on windows whose GT class is cruising (|v6 - v0| < 1.5, S6 rules) |
| LATERAL | `LAT_cross_abs_final_m`, `LAT_heading_err_final_deg`, `LAT_cross_mae_m`, `LAT_heading_mae_deg` | same S6 windows |
| TACTICAL | `TAC_head_lat_acc`, `TAC_head_lon_acc`, `TAC_traj_lat_correct`, `TAC_traj_lon_correct` | decision-head argmax vs the labels; trajectory-derived manoeuvre correctness |
| STRATEGIC | `STR_route_acc` | route-head argmax vs `route_target`. The route head can ECHO the fed nav: read it beside the clip/token contrasts, never alone |

**NOT COMPUTABLE here (stated per family, not dropped):** LONGITUDINAL **distance-keeping** (headway / time-gap / TTC): the dumps carry no lead-block join (needs the B1 lead block on its 0.2 s grid against this 0.5 s plan grid).
STRATEGIC **goal/route-point setting** beyond the route-head accuracy: the g_str goal point is not dumped as a row. Both are work items, recorded in `NAV_FEED.json` `families_not_computable`.
Per family, never pooled into one score. Error metrics: lower is better; accuracies: higher is better. Contrast = A - B.

## 5. Estimator

`taniteval.ci.paired_episode_cluster_bootstrap`, B = 2000, seed 0, resampling EPISODES of the set, pairs on identical windows. It answers *"would another draw of EPISODES say this?"* at a FIXED checkpoint
and a FIXED inference seed. It is **blind to another training run** (H-ESTIM-SEED-1: this is a single EMA-40k checkpoint of a single training seed) and **blind to another inference run**, which is why the
inference-seed floor F below gates the controls and the effect, and why the two seeds are never pooled. A separated interval is necessary here, not sufficient.

## 6. Controls and the committed rule (pure function `nav_feed_analyze.decide`, pinned by the truth-table tests)

* **F, the inference-seed floor:** `F = max(|delta|, CI95 half-width)` of the paired contrast `clip(seed 1) - clip(seed 0)`, ADE 6 s, on ALL 1,457 sealed windows.
* **C1 (gates the label):** `perframe_token - clip` on the 300 CONTROL windows, ADE 6 s, must satisfy `|mean delta| <= 2F` at BOTH seeds (the instrument reads ~0 where nothing differs). Failure => `control_ok = false`
  and the label is reported as the literal rule's output marked **NOT ADMISSIBLE**.
* **C2 (descriptive, does not gate):** `clip` re-roll vs the banked `os` of `step40000ema_unc` on the same 1,457 windows (mean ADE 6 s abs difference), beside the banked seed-replicate difference as the yardstick.
  The banked roll batched all 4,754 windows, so bit-identity is not expected; it shows the re-roll is the same instrument, not that it is equal.
* **C3 (descriptive):** GT ego-frame vs the banked `g` (< 1e-4 m), `os` equals the first four plan slots; token and sidecar feed as designed (`nav_cmd` equals the sealed token per arm, `nav_cmd_clip` equals the banked clip token,
  `r8_nav_fed` non-zero exactly on the known rows for `perframe_full` and all-zero for `perframe_token`).
* **Primary cell:** `perframe_full - clip`, ADE 6 s, MISMATCH set (1,157 windows / 59 episodes), per seed.

**The label (exactly the code):**

* `NOT-EVALUABLE` if a primary cell is absent for either seed (a dump missing or malformed).
* `FEED-FIX-MATTERS` iff **all** of: the primary cell is separated NEGATIVE (`hi < 0`) at seed 0 AND seed 1; `|delta| > F` at BOTH seeds (strict); C1 passes (`control_ok = true`).
* `FEED-FIX-NULL` iff NO seed is separated in either direction (`lo > 0` or `hi < 0`).
* `MIXED` otherwise: separated at one seed only; separated WORSE; opposite signs; separated but not above F; or the MATTERS pattern with C1 failed.
* `label_admissible = control_ok`. A label with `control_ok != true` is printed with the explicit string "NOT an admissible result".

Everything else (the other 14 metrics, the `perframe_token - clip`, `perframe_full - perframe_token` contrasts, the by-(fed -> v9) split) is **descriptive and cannot change the label**; it is the attribution table
(token half vs args half). The split by (fed -> v9) is exploratory. Any claim from the descriptive tables needs its own pre-registration.

## 7. What each outcome triggers (Rule Zero: no outcome ends the turn)

* **MATTERS (admissible):** every `os`-derived refcv8 battery number at 12 trees x 2 seeds carries the feed defect (NAV_TRACE sec 5). Next lever, executed without asking: roll the FULL 4,754-window battery with
  `--nav-feed perframe_full` (or `perframe_token` if the descriptive `full - token` contrast is inside F), re-score the bars through the standard analysis path on that dump, and file the registry quarantine note on the old rows.
* **NULL:** the defect is real but inert at EMA-40k on the mismatch windows; no battery re-roll is triggered. The next lever is the next-largest measured term in the existing battery attribution; this result removes the nav feed from
  the suspect list for refcv8 (log it as an ELIMINATION, with F and the CI).
* **MIXED:** do not re-roll the battery. Read the descriptive tables to find which of {token, args, a fed -> v9 group} carries the movement; run a THIRD inference seed on the primary pair only (cost ~ 0.5 GPU-h) and a pre-registered
  follow-up on the dominant group. If the movement is WORSE (separated, positive), that is a finding about the trained model's reliance on the clip-level token and goes to the PI as such.
* **NOT-EVALUABLE / control not passed:** the instrument, not the model, failed. Read `controls` first (C1 bit-identity, C3 sidecars), fix the instrument (the roll is resumable per arm and seed) and re-run only the missing or failing roll.

## 8. Limits (said in advance)

* ORACLE nav: the v9 token/args are built from the ego's future path; a supplied route is optimistic by construction. This measures train/eval consistency, not what a deployed (predicted-goal) system would do.
* One checkpoint (EMA-40k), one training seed. Another training run is outside the estimator (H-ESTIM-SEED-1); the verdict is about THIS checkpoint.
* The mismatch set is the 1,157 windows where the feeds differ; it is not a sample of all windows. Any whole-battery effect needs the full re-roll of sec 7 (ESTIMATED ~0.13 m full-set pessimism, NAV_TRACE sec 5; not a correction).
* `os` only: the nav-null, shuffled and flipped arms, `ha`, and the other conditioned arms are not rolled here.
* The reference for C2 is a differently batched roll; the replicate yardstick makes it interpretable but it is not a bit-identity test.
* T1 only; no T2.

## 9. Execution protocol and cost

* `code/run_navfeed.sh` (wrapper) is started only by `code/wait_navfeed.sh` (detached waiter, single instance, pid in `<out>/state/waiter.pid`). Outputs under `C:/Users/Admin/refcv8_navfeed_out/` (about 50 MB, ESTIMATED from 27 MB for 4,754 windows).
* **Start conditions (all must hold):** the GPU lock file is absent and is then taken through `gpu_lock.py acquire --job refcv8-navfeed --pid $$` (a foreign lock is never broken; released by an EXIT trap); >= 3 GB free on the output
  volume (Get-PSDrive) AND a real 16 MB write test; >= 6 GB free commit (`wait_commit.py`); the final 48,852 chain is not active (the `yield48852_r8.sh` `bfinal_pending` rule verbatim: weights banked AND chain alive AND
  not ended AND battery/K-draw still open; plus no python process naming the tag; plus no chain-active marker). While the final chain is only idling in its pull wait (weights not banked) it is not "running".
* **Yield:** before every roll after the first, the wrapper re-asks `bfinal_pending`; if true it exits 10 and releases the lock. Completed rolls are never redone (resume by `roll_*.json done: true`).
* **Mini-smoke under the lock first** (clip and perframe_full, 3 mismatch + 3 matched windows of one episode, `smoke_check.py`: nav_cmd as designed, sidecars, plans finite, plans differ on the mismatch windows). A failed smoke
  stops the slot with exit 20 and the waiter HALTS (a failed GPU run is never retried unattended). The mini-smoke is the first execution of the model path: no CPU model-build smoke was possible (host commit 0.5-14 GB, another job holding ~13 GB).
* Markers `ZZNF...ZZ` in `navfeed.log` (opaque; counts computed inside the wrapper). The wrapper exits: 0 done, 10 yield, 11 no lock/commit/disk, 12 a condition flipped, 20 roll or smoke failure, 21 bad seal.
* **Analysis:** `PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -X utf8 code/nav_feed_analyze.py --out-dir C:/Users/Admin/refcv8_navfeed_out --windows raw/windows_navfeed.json --prereg PREREG_NAV_FEED.md`
  (CPU only; the wrapper runs it itself after the last roll and releases the GPU lock before it). Output `NAV_FEED.json` and `NAV_FEED.md`, copied to `raw/run/`.
* **Cost: ESTIMATED.** 6 rolls x 1,457 windows. At 0.515 s/window os-only (my working estimate from the roll's per-forward timing, not re-verified) that is ~1.25 h + model loads = ~1.3-1.4 GPU-h. Upper bound: the archived FULL-battery roll took
  9,463 s for 4,754 windows (1.99 s/window, MEASURED, `step40000ema_unc/roll_s0.log`), i.e. 4.8 h if the os-only arm cost as much as the whole battery. The roll logs give the measured rate after the first roll.

## 10. Sealed hash register (sha256 of the file bytes at seal time)

The wrapper requires the first seven hashes to appear in this text and to match the files on disk.

| file (relative to the hold) | sha256 |
|---|---|
| code/overlay/navfeed_overlay.py | 98293dd7411270160cb47d070bb1538babe1ff3a0f8ab85da61dcc84129ba72f |
| code/overlay/navfeed_roll.py | 0156b3d4d391b224f04a7869d8ce1bd718eb89d59dcfa3d8b8af205a7325c08c |
| code/make_windows_navfeed.py | d6177a5830abcd3eefe21548189261eda7935c1236a4af4049cc7063784ed0f5 |
| code/build_roll_windows.py | b77ebcfa9665858911951c7ee6f72c50c0f6f3af5b7b1e964f3115de5a10af57 |
| code/smoke_check.py | eebd1dfca747e18cae7526ebe48909ea9033df0d851c54f5ef7d48d2ad508faf |
| code/nav_feed_analyze.py | 6070c5778eab380d8d3bf2022065baae42d77f8221f679fff28a4b3c98aa0c23 |
| raw/windows_navfeed.json | 6bb6dd74ea9a67580b48dce6e119855d61aaf117b20a13acc48cf1f80e1b4a7b |

Recorded, not enforced by the wrapper (the shell scripts cannot hash themselves meaningfully; the tests and diff are evidence):

| file | sha256 |
|---|---|
| raw/refcv3_arm_navfeed.diff | 678794530c431ddf5b0e87304ba33cd9755e813d255432f9327bbc5b1fb9f5c3 |
| code/run_navfeed.sh | 1b432ca4c49abf5cb07b1d2542541709b4c653f1b4c1d89e61eae8f605f62af6 |
| code/navfeed_common.sh | a7c89fb6fc57c188dbd50fea779a9008d3de961232ee7f13a2027cf3aef946df |
| code/wait_navfeed.sh | a7a0b8483187ce7080518fdcc1f1f1a050443a3c1e422a3182e37632caf48a1c |
| code/tests/test_navfeed.py | 2457e8692c0258b06459dbd36d900766da9ca44e281e7a107c8050fb18b5274a |
| code/tests/test_nav_feed_analyze.py | 407da48a6a3235cbd818c96dbd77cdfec0622f2d5abe65f0e1e7b3fa4f32a736 |
| code/tests/test_wrapper_flow.py | 4f4c20eebb2f4ac94af19148a391a52011f2ab77cae61a199f8b5b1108783ec2 |

Launch tree `C:/lgs/Lfb47a58376` is read-only to this experiment; battery package `qland/hold_r8batt` is imported read-only (`r8_battery_patch.apply_child`).
