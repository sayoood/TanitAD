# P2 navtrain HELD-OUT 64-proposal table: plan, gates, coverage, budgets (written before the main job's first dump; launched 2026-10-18 08:40 Berlin)

PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. This package mirrors `<PK>/raw/2026-10-18-p2-heldout-table/`. Brief: Master Mind, BACKLOG `SEL-RANK` -> SR-TTC first gate, plus the correction on SR-TTC's roles.
Role of the table (PREREG_SELRANK_ARM.draft.md sec 3): SR-TTC's **DEV GATE**, not its fit set (the fit set is P2's own training labels). Navtest stays the confirmation set and is never fitted. **Nothing is fitted on this table.**
Tier T1 (open-loop plan, NAVSIM-style PDM scoring), P2 `model_final` sha256 `ea31db22...`, rule `navsim_v1`, goal fix pdm_route, A7 repair. One training seed; inference deterministic.

## 1. What can and cannot be built on this box (MEASURED, `plan/audit.json`)

| item | number | how it was measured |
|---|---|---|
| banked held-out sets / distinct (log, token) / logs | **3,137 / 1,705 / 24** | the 7 label files; keys (log, token, step, rank). Rank-0 = 1,705 (one per token), rank-1 = 1,432 goal-augmented twins. **P2's fan depends on the token only, so the table has one row per distinct token; `heldout_sets_index.npz` maps all 3,137 sets to their row.** |
| tokens whose log has a nuPlan DB | 1,343 tokens / 13 logs / 2,466 sets | `splits/*`, `splits/trainval/*` |
| tokens with a DB AND all four camera frames resolvable by the planner's own `FrameResolver` | **333 tokens / 11 logs / 603 sets (19.5 % of tokens)** | roots: `data/refe_proxy/pixels` (M6 proxy extract of the pod's navtrain_pixels, original jpgs), `data/refe_m5/pixels` |
| BLOCKED: DB present, no camera frames | **1,010 tokens / 13 logs** | no image of these tokens is on the box; `plan/needed_frames.json` lists the exact `filename_jpg` of 4 cameras per token (4,040 jpgs) |
| BLOCKED: no DB | **362 tokens / 11 logs** | the 11 trainval logs listed in `plan/audit.json` |

**P2 cannot be forwarded on a token without its four camera frames; the PDM scorer cannot score it without its DB.** So the table this job can build today is the **333-token slice (PARTIAL)**; the DONE marker says so. The remaining 1,372 tokens need (a) the 4,040 jpgs, (b) the 11 DBs: see HANDOFF.md.
Disjointness (G1, `disjointness_proof.json`): the 24 logs / 1,705 tokens share **0 logs and 0 tokens with navtest** (12,146 / 136) and **0 with the 174,912-set bank P2 trained on** (99,495 tokens / 1,167 logs, reproduced). That P2's RUN used the carved bank is INHERITED (PREREG_P2, P2R_POD_RUNBOOK:99-103, RESULT_P2 sec 4, PREREG_RL_S1:31): the pod's `carve.log` was not read (pod off limits).

## 2. Instrument: why not W3's harness (deviation from the brief, stated)
The landed full-navtest table was scored by W3's `run_v1.py cmd_score` (navtest metric cache, `navsim_logs/test`). A navtrain token has neither (`data/openscene/navsim_logs` holds `test` only). The only held-out-capable scorer is `refe/navsim_pdm_targets.py` (NAVSIM v1.1 @ 3e8291b's own MetricCacheProcessor + PDMSimulator + PDMScorer on a scenario rebuilt from the nuPlan DB). **It is the scorer that produced the banked held-out labels and P2's 174,912 training labels**, so the table is on the labels' instrument. Its difference from the W3 navtest instrument is MEASURED: `validate_pdm_targets.json` (300 tokens, 19,500 plans): PDMS exact 94.73 %, mean |d| 0.0008, argmax identical 100 %; re-measured here on P2's own landed proposals (control B): **PDMS exact (1e-4) on 95.1 % of 512 plans, mean |d| 1.7e-5, argmax 8/8**. A held-out number is therefore on the labeller's instrument, not W3's; a navtest number from the landed table is on W3's. Say which.

## 3. Pipeline and gates (reused from the landed pipeline: `landed/c3_run_p2full.py` blob f249a02a IMPORTED UNCHANGED for the lock / guard / state helpers; diffs: `p2ho_driver_vs_p2full_driver.diff`, `p2ho_dump_vs_refe_navtest_seam.diff`, `launch_p2ho_vs_launch_p2full.diff`)
audit -> per-LOG units (a log is never split) -> P2 forward (`code/p2ho_dump.py`, GPU behind `devbox_gpu.lock` via the landed `take_lock`: `wait_idle.py`, O_EXCL lock, `chain_active` respected, another job's lock never removed) -> dump gate -> PDM scoring of all 64 proposals (`code/p2ho_score.py`, 2 workers, RAM guard >= 8 GB, stagger 15 s) -> table.
Outputs go to `C:/Users/Admin/p2heldout_work/` (D: is full and is never written; the job STOPS CLEANLY below 5 GiB free on C:, rc 6, resumable).

| gate | rule | evidence in this package |
|---|---|---|
| G1 preflight | ckpt sha256; disjointness vs navtest AND P2's training bank; plan audit; wait_idle present | `disjointness_proof.json` PASS |
| G2 fidelity A | scorer reproduces the banked held-out labels, `round(x,5)` within 1e-5 on nc, dac, ep, ttc, c, ddc, pdms for EVERY proposal of 12 sets (3 logs x 4); negative control (reversed proposals) must FAIL | `fidelity.json`; smoke: 0.0 on all 7 heads, negative control 1.0 |
| G2b DG-DUMP | this dump script on the GPU reproduces the landed `table_p2_full.npz` rows (sha 33441082...) of 4 navtest tokens: pose <= 1e-3 m, logits <= 1e-4, pick equal; negative control (next row) must FAIL | `dgdump.json`: **proposals 0.0, logits 0.0, 4/4 bit-identical on the GPU** |
| G-DUMP per chunk | every asked token a finite row, 64 proposals [8,3], rule navsim_v1, A7 on, ckpt path, features [64,256] finite, misses 0 | `state/gate_dump_*.json` |
| G3 | every (token, proposal) cell valid and finite | `gate_table.json` |
| G4 same-pick | the planner rule recomputed from the table's own logits reproduces the dumped pick on every row; PDMS == NC*DAC*(5EP+5TTC+2C)/12 on every cell (<= 1e-6); DDC-in-product control must NOT reproduce | `gate_table.json` |
There is no banked P2 pick on held-out tokens, so "same pick" is proved two ways: the dump path equals the landed one on navtest (G2b), and the pick is a pure function of the table's logits (G4).

## 4. Pre-head scorer features (Master Mind correction 2)
CAPTURED, no model code touched, no second forward: a forward hook on `model.score_head` records its INPUT (`s`, [1, 64, 256]) in the same call as the logits; the hook's output is asserted bit-equal to the logits `infer()` returned and the hook is asserted to fire exactly once per token. fp16, `prehead_features_p2_heldout.npz` on C:, sha256 in the manifest.

## 5. Time (ESTIMATED; basis shown)
GPU: landed dump rate 1.35 s/token (16,402 s / 12,146) to 1.65 s/token (509-token run) x 333 = 450-550 s; 11 model loads (landed PLAN: ~38 s each) = 420 s; DG-DUMP measured 23 s -> **~16 min GPU**. Scoring: ZT5 dev rate 4.36-5.41 s/set -> 333 x 4.4-5.4 = 24-30 min of CPU, 2 workers overlapped with the dumps. **ETA was ~25-35 min after the lock.** MEASURED outcome: launched 08:40:29, lock taken 08:42:46, `P2HO_DONE` 09:07:00 (26 min wall; 11 units, 2 scoring workers, no RAM-guard pause, `score_fail` 0). The full 1,705-token table would be ~5x this: ~1.5-2.5 h wall once the inputs exist.

## 6. Not verified / limits
* Post-run: the GPU path, 11 units and 2 scoring workers all ran clean (`logs/`, `state/`). `featcheck.json` re-derives the logits from the saved features with P2's own `score_head` (1.1e-3, fp16 input). Still INHERITED: that P2's run used the carved bank.
* 333 tokens over 11 logs is a small, log-stratified-by-the-proxy-sample slice; a log-cluster interval over 11 logs is wide. The SR-TTC dev-gate bar (>= +0.30 on both seeds) is below what 11 logs can resolve: this slice supports plumbing and a first read, not the gate.
* The slice is the M6 proxy's stratified sample of navtrain frames: selected by that sample, not by this job.
