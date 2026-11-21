# N0, the nav probe: DAG at T1-kin, `--nav logged` vs `--nav deadreckon` (PREREG_V8F_DAG_ORC section 8 @ 85d0b125), 2026-10-27

**Verdict: NOT NAVMOVE. All controls PASS (not REFUSED).** Committed next (section 8 table, row "not NAVMOVE | not, or not yet read"): **LAUNCH-INTENT**. DAG-OACC's read does not exist, so the other not-NAVMOVE row (COMPOSE) is not reachable yet. DAG-ORC is **not held** by N0 (precondition 3).

Nothing was trained, no GPU, no box 2, no git, no pytest. Staged nothing (artifacts only, no repo files). Evidence class MEASURED (ours, artifact paths below). Tier **T1-kin: self-action OPEN loop**, plain (no post-map), m = 10, periodic, causal speed bucket, logged scene, executor `plan`. **`--nav logged` is an OPEN channel** (the tool's own `open_channels`: "route countdown from the logged human position"), so a gain would not have separated route extent from timing; the actual result is a null (below). Never "closed loop".

## What ran

| item | value |
|---|---|
| checkpoints | `v8f-dagger/M2b_s0_dag/ckpt_last.pt` sha256 `d186fd0b...`, `M2b_s1_dag/ckpt_last.pt` `1d12ec2e...`; re-hashed before AND after the rollouts, equal to each dir's `SHA256SUMS.txt` |
| tool tree | `dagt1_tree_b43c7a9a` (DAG's own); the 9 key files re-hashed before and after, equal to `v8f-dagger-t1/PINS.txt`; the tool has `--nav {deadreckon,logged}` (`NAV_MODES`, line 63) |
| rollouts | 2 (s0, s1), concurrent (the cap), `CUDA_VISIBLE_DEVICES=-1`, `OMP/MKL_NUM_THREADS=8` (DAG's chain env), TMP/TEMP on F:. Wall **347 s (s0), 349 s (s1)**, 16:31:16 to 16:37:05, rc 0. C: 6150 MB, E: 6684 MB free at start (guards 2048 / 1500) |
| reader | `n0_read.py` (new, thin) importing `dag2t1/dag2_read.py`'s layer A unchanged (`ReadA`: `rows_block`, `T.rows`, `table`, `rows_equal`, pins). 84 s. Tick-level beside rows: `n0_ticks.py` via `dag_read.RealBackend.analyze_t1` (213 s) |
| estimator | `paired_episode_cluster_bootstrap`, B = 2,000, seed 0, 141 episodes / 2,399 windows, per seed. Same checkpoint, deterministic planner: it answers "another draw of EPISODES" only; no training-run or inference-run term enters |

**argv diff (control 1).** N0's argv equals DAG's banked deadreckon argv (the `start DAG_s{s}_m10_causal_heads` line of `dagt1/logs/chain.out`, 32 tokens) in 28 of 32 tokens. The 4 differing tokens, per seed: `--tag` (`N0DAG_s{s}_m10_causal_heads_navlogged`), `--out-dir` (`dag-orc-n0`), `--dump-ticks` (`dag-orc-n0/ticks`), and **`--nav deadreckon -> logged`**. Full diff `argv/ARGV_DIFF.json`; both argv files `argv/`. The run JSON headers agree with DAG's on ckpt, kin_mode, tier, replan, schedule, scene, limit, n_windows, model_keys_read, envelope, strict.{kinematics,camera,speed_bucket}; they differ in `nav`, `strict.route` and `open_channels`, by design (K-HDR).

## Controls (any failure = REFUSED; all PASS)

| control | result |
|---|---|
| K-ARGV (1) | argv identity except `--nav` / tag / output paths, both seeds: PASS |
| K-IDENT (2) | the reader on (banked deadreckon vs itself): all 16 paired Read-A rows exactly 0.0 [0.0, 0.0], plus the 13 comps and 5 TAC / DK identity rows, both seeds: PASS. (The 3 S-POS rows are unpaired vs CTRV within a run, so outside the pair identity.) |
| K-RC (3) | layer A on the banked dumps reproduces `dagreread/raw/READ_A.json` DAG - P10 (19 rows x 2 seeds) with **0 differences, max abs diff 0.0**. Cross-check: `dag2_read.py --repro-only` re-run today, same result (`controls/RC_repro_dag2read.json`) |
| K-WIN (4) | 2,399 windows / 141 episodes in both the new and the banked dumps, `clip_index`, `ws`, `eid`, `g`, `v0` bit-equal: PASS |
| informational | K-REACH: nav-logged changes the executed track in **1,379 of 2,399 windows at both seeds** (max 23.2 / 16.7 m), so the lever reaches the output; the N0 - DAG rows are non-zero where the identity rows are exactly zero (the positive control for the zero control). K-FLOOR (P10 tracks bit-equal) PASS. C-SET: the crawl set re-derived from the banked READ_A.json equals the PREREG's C. Tick-level K1, K2, K3, K5, K6, K7(x1), GRID all PASS for both arms and seeds; DAG's S-POS re-derived from its banked ticks equals `DAG_READ.json` (max abs diff 0.0) |

## The decisive read: nav-logged - nav-deadreckon, rows of C (delta = X - Y, negative = lower error)

| row | s0 | s1 |
|---|---|---|
| curvature MAE (1/m) | -0.00010 [-0.00110, +0.00080] ns | -0.00040 [-0.00210, +0.00080] ns |
| L2 curvature MAE (GT>=5.2) | +0.00040 [-0.00010, +0.00120] ns | -0.00068 [-0.00183, -0.00001] **separated-better** |
| M1 crawl share | +0.00010 [-0.00040, +0.00070] ns | -0.00040 [-0.00100, +0.00010] ns |
| yaw-rate MAE (deg/s) | (not in C_0) +0.0229 [-0.0307, +0.0931] ns | -0.1073 [-0.2284, -0.0186] **separated-better** |

NAVMOVE needs every row of C_s separated-better at both seeds. At s0 **none** is; at s1 two of four are. **NOT NAVMOVE.** The size matters as much as the flag: as a share of DAG's measured excess over P10 (recomputed here, K-RC green) nav-logged moves curvature -0.2 % / -0.4 %, L2 +1.4 % / -1.1 %, M1 +0.2 % / -0.7 %, yaw-rate -1.2 % (s1). The two s1 separations are about 1 % of the gap. For scale, HACC's zero-training post-map cut the same rows by 36 to 72 % (PREREG section 8, INHERITED).

**Reading it (interpretation, not a registered claim).** `--nav logged` is a superset of any dead-reckoner fix: it carries the logged route extent and the logged timing. The channel is live (1,379 windows changed; the T head's lateral macro-F1 moves +0.0231 / +0.0292, separated at both seeds), yet the crawl rows do not follow. So neither a route-extent defect nor a timing leak is the source of the crawl rows in this supply, and N0 gives no support for the ROUTE-EXTENT PREREG. HYPOTHESIS to carry, not a finding: the crawl is born in the model's own executed-horizon along-track profile (section 8 rank 1), which is what LAUNCH-INTENT targets.

## Beside (all four families; never decisive; no multiplicity correction)

Full 19-row table, both seeds: `N0_compact.txt` (and `N0_READ.json` `beside`). Separated-better at s0: DK-HW -0.0143, DK-TG -0.0017, heading MAE -0.0620; at s1: ADE6 -0.0071 m, speed MAE -0.0050, along-track -0.0080, accel MAE -0.0135, heading -0.0570, L1 lat-accel -0.0155 (plus the two C rows above). Nothing separated-worse at either seed. All are small against the +-0.1207 m ADE6 seed floor. ADE6 N0 - DAG: -0.0037 [-0.0126, +0.0041] (s0), -0.0071 [-0.0151, -0.0001] (s1).
- **LONGITUDINAL:** rows above; distance keeping DK-VIOL / DK-HW / DK-TG are in the table (LEAD windows only). **LATERAL:** above. **TACTICAL:** TAC-LAT / TAC-LON ns at both seeds; T-head macro-F1 (ticks >= 1, matched): lat +0.0231 / +0.0292 (separated), lon -0.0012 / -0.0028 (ns). **STRATEGIC** (not paused for v8F): S-POS vs CTRV within run, N0 vs DAG-dr, unpaired: s0 @30 / 50 / 80 = -1.81 / -4.81 / -10.55 vs -1.72 / -4.48 / -9.69; s1 -1.94 / -5.15 / -10.88 vs -1.87 / -4.90 / -10.33 (all separated vs CTRV in both runs; a paired N0 - DAG S-POS contrast is not computed, PREREG section 5).
- N0 - P10 and DAG - P10 tables are in `N0_compact.txt` (N0 stays separated-worse on every C row vs P10 at both seeds, as DAG does).

## What this cannot claim

* It is a null on an **open** channel at **T1-kin**, one checkpoint pair (two seeds), one eval set. It says the nav input, as logged, does not move the crawl rows; it does not say the crawl has no input-side cause.
* A separated interval answers "another draw of episodes"; there is no training or inference variance here (same weights, deterministic planner), so no replicate arm is needed for this contrast.
* The 16 beside rows are not multiplicity-corrected; the s1 L2 interval's upper end is -0.00001 (borderline) and its size is 1 % of the excess.

## Next (committed, section 8)

LAUNCH-INTENT now (its own PREREG: O's along-track loss over the executed horizon, knots 1-12 at m = 10, up-weighted on GT-launch TRAIN samples; one flag with its G-DVB entry). It is replaced by **COMPOSE** only if DAG-OACC's read later shows DAG-OACC - DAG separated-better at both seeds on the rows of C; no DAG-OACC checkpoint or read exists (`v8f-dagoacc/` holds a gate dir and offline w_acc probes only). DAG-ORC's own launch remains gated on P-0 (RF0 frozen), its reader with K-RC and a mutation proof, and the PI's go for box-2 time (PREREG section 10).

## Deliverable manifest (all under `E:/Projects/TanitAD-artifacts/dag-orc-n0/`; none staged in git; single copy on E:, which is an external drive, so the integration request is in the hand-back)

`N0_READ.json` (decisive rows, controls, routing, beside), `N0_READ_core.json` (the reader's raw output, sha256 `103490aa...`), `N0_compact.txt`, `N0_NOTES.md`, `N0_ticks_beside.json`, `N0_SHA256SUMS.txt`, `argv/`, `controls/RC_repro_dag2read.json`, `t1v8f_N0DAG_s{0,1}_...json`, `ticks/`, `dump_N0DAG_s{0,1}_m10_causal_heads_navlogged/` (141 files each), `logs/`, code `n0_mkargv.py`, `n0_run.sh`, `n0_read.py`, `n0_ticks.py`, `n0_finalize.py`, and the PREREG blob copy `PREREG_V8F_DAG_ORC_85d0b125.md` (sha256 `38bc688b...`, equal to the mirror blob). Size 345 MB. Scratch: `F:/TanitAD-scratch/n0_tmp/` is empty.
