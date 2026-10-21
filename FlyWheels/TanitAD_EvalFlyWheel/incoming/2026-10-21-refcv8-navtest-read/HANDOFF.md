# HANDOFF — refcv8 EMA-48,852 navtest read (A-R8-NS1 + Amendments 1-3) on Thor

Agent: EvalFlyWheel (navtest-read builder), 2026-10-21 (Europe/Berlin). Nothing is committed and nothing was written to G:.
Tier: NavSim v1.1 open-loop PDM scorer (T1-family), zero-shot, non-parity, NOT closed loop, NOT leaderboard-submitted.
**No refcv8 navtest number exists yet.** The smoke rows are VALIDATION ONLY and are never quoted.

## 1. State at hand-off

* **FULL roll LAUNCHED 2026-10-21 09:57:23 (Berlin)** under `flock -n -o /home/nvidia/refcv7_post/thor_gpu.lock` (held by the flock
  process for the whole script; `/proc` scan: no child holds the fd). Verified progressing: R8_LEGAL rows 552 → 643 → 769 (09:59:30 →
  10:00:12), 0.19 s/scene, GPU 91–97 %. **ESTIMATED finish:** `ZZNAVTESTDONE` ≈ 12:00 (36,438 forwards × 0.19–0.20 s), then warmup
  (660 fwd) + navhard (11,824 fwd) → `ZZALLDONE` ≈ 12:40.
* Gates, all asserted on artifacts (copies in `raw/`): **G0** ckpt md5 `b4364029…`, deploy table `3319b003…` + sidecar, frozen `2aacc9ff…`,
  stage `b4bc7b79…`, derive_rt_seed `bebdc5c1…` (literals in the launch script). **G1** `prep/G1_CHECK.json` PASS: 12,146 tokens, all in
  the bank with 4 history rows, 32/32 shard files valid, 0 partial / 0 refused in the build reports, 136/136 logs present; mutation
  `refusal_check/REFUSAL_CHECK.json` PASS (a token removed from the index → the bridge exits "not in the 416 bank -- refusing", 0 rows).
  **G2** GPU 0 %, 0 compute apps, 0 python, lock free. **G3** smoke20 on the launch path: 6 seams × 20 rows, 0 stand-ins, KPR PASS,
  K0/KD/premises PASS on CUDA, RT_ASSERT PASS, r8_nav 0 / v_max_valid 0 on LEGAL rows; warmup + navhard tail smoked too (`--limit 10`).
  The launch re-ran K0/KD/premises: PASS (`raw/full_launch_snapshot/cuda_controls.json`, max mem 1.178 GB).
* **Known-value controls, scored on Thor through the SAME scorer path** (`raw/controls/CONTROLS.json`, VALIDATION): STOP **61.8202**
  (per token max |Δ| 6.2e-12 vs W3's dev-box CSV, 0 flips); refcv7-50,400 LEGAL re-scored **70.2048**, per token |Δ| 0.0, merged CSV
  sha256 `166ca2a0…` = the banked Thor CSV.
* Data shipped + verified (`raw/ship_verify/`): navtest bank 82/82 files md5 = dev box; metric cache 12,152/12,152; logs 147/147;
  warmup/navhard banks 12,238/12,238; maps 7/7 sha256; inputs md5 `335c82bc…`; speed / road-plane files sha256 = dev box.

## 2. Where everything is on Thor (`ssh tanitad-thor`)

| what | path |
|---|---|
| run dir (the bar read) | `/home/nvidia/r8ns/navtest_read/full/` (`bridge_navtest/`, then `bridge_warmup/`, `bridge_navhard/`) |
| stdout / markers | `/home/nvidia/r8ns/navtest_read/full.out` |
| code | `/home/nvidia/r8ns/code/navtest_read/` (launch, prep, score, controls, progress) |
| G1 + shard/smoke token files | `/home/nvidia/r8ns/navtest_read/prep/` |
| controls (STOP, refcv7-50,400 re-score) | `/home/nvidia/r8ns/navtest_read/controls/` (`CONTROLS.json`) |
| data | bank `/home/nvidia/r8ns/navtest/frame_bank`, cache `…/navtest/nsexp/w3_navtest_v1/metric_cache_navtest`, logs `…/navtest/nsdata/openscene/navsim_logs/test`, maps `…/navtest/nsdata/maps`, warmup/navhard banks `/home/nvidia/r8ns/banks/` |

## 3. Markers (grep the opaque tokens only; never an exit code)

`full.out`: `ZZNAVTESTDONE` (navtest rolls + post-gate + RT assertion PASS) → `ZZWARMUPDONE` → `ZZNAVHARDDONE` → `ZZALLDONE`;
`ZZREFUSE_<why>` = a gate refused (nothing after it ran). `QQFLOCKRC=<n>` is printed when the flock-held script exits (1 = lock busy).
Progress, one line: `ssh -n tanitad-thor 'bash /home/nvidia/r8ns/code/navtest_read/r8nt_progress.sh'` → `QQP navtest:R8_LEGAL=<rows> … QQ`.
Arms run in order R8_LEGAL → R8_LEGAL_s1 → R8_PRIV (12,146 rows each). A dead run is RESUMABLE: relaunch the same command (rows are skipped).

## 4. Next agent — exact commands

**(a) Score navtest on the Thor CPU backend** after `ZZNAVTESTDONE` (24 shard jobs, ≤ 4 at a time, ~45 min ESTIMATED from the
controls' shard walls). CPU-only, nice 19; it may overlap the warmup/navhard GPU roll (MemAvailable ≥ 35 GB gate) — or wait for `ZZALLDONE`.
```bash
ssh tanitad-thor   # then, on Thor:
X=/home/nvidia/r8ns/navtest_read; NX=/home/nvidia/r8ns/code/navtest_read; O=$X/full; B=$O/bridge_navtest; S=$O/scores_navtest
A="--arm R8_LEGAL=$B/seam_R8_LEGAL.npz --arm R8_LEGAL_s1=$B/seam_R8_LEGAL_s1.npz --arm R8_PRIV=$B/seam_R8_PRIV.npz \
   --arm R8_RT=$B/seam_R8_RT.npz --arm R8_RT_s1=$B/seam_R8_RT_s1.npz --arm PRIOR_ha0p=$B/seam_PRIOR_ha0p.npz"
python3 $NX/r8nt_score.py jobs --prep $X/prep --scores $S --tag s48852e $A --out $O/score_jobs.jsonl          # QQJOBS n=24
cd $S && OMP_NUM_THREADS=6 nohup setsid python3 /home/nvidia/navsim-thor/code/thor/thor_pool.py --jobs $O/score_jobs.jsonl \
   --state $O/pool_state.json > $O/pool.out 2>&1 < /dev/null &
# done when `tail -c 40 $O/pool.out` reads ZZ24-0-0-0-<GB>ZZ (done-fail-alive-queued-availGB); any fail -> read _shards/<label>.driver.txt
python3 $NX/r8nt_score.py finalize --prep $X/prep --scores $S --tag s48852e $A --order-ref $X/controls/STOP_navtest.csv
# QQFIN: every arm status PASS, rows_jsonl_tokens 12146.  Assert on scores_navtest/<label>/<label>.counts.json, never on rc.
```
**(b) Pull to the dev box** (Git Bash; relative paths for tar): seams + scores + INDEX.json (≈ 0.5 GB ESTIMATED):
`ssh -n tanitad-thor 'cd /home/nvidia/r8ns/navtest_read && tar -cf - full' | tar -xf - -C /c/Users/Admin/refcv8_pf/navtest_read/raw/`
**(c) Bars (dev box, tanitad venv).** Run first WITHOUT the two summaries (N1/N4 + strata), then again once (d) exists:
```bash
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
P=C:/Users/Admin/refcv8_pf/navsim_ckpt/pkg/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim   # = Thor copy (sha dd8243bf… bars8)
PYTHONPATH="$P/code;C:/lgs/Lfb47a58376/taniteval" $PY -X utf8 $P/code/navsim_bars8.py --run-dir C:/Users/Admin/refcv8_pf/navtest_read/raw/full \
  --tag step48852_ema --out-dir C:/Users/Admin/refcv8_pf/navtest_read/raw/bars --city-census C:/Users/Admin/navsim8/navtest_city_census.json \
  --ship-r8-passed 1 [--navhard-summary <summary_navhard.json> --warmup-summary <summary_warmup.json>]
```
Check before reading any verdict: `controls.city_census.status == OK` (1,914 / 12,146), `ep_clause_source_consistent` and
`v0_inputs_vs_scorer_rows` OK, `banked_STOP` OK, label `RESULT (EMA 48,852, the bar read)`. Report NC/DAC/TTC/EP/C + PDMS per arm.
N4 is MIXED-backend only in name: the refcv7 row was scored on Thor too, and this run's STOP control equals the dev-box STOP CSV.
**(d) warmup (N3) / navhard (N2) scoring** after `ZZALLDONE`: refcv7's `score_arm7.py` via the Thor backend's `score_arm_thor.py`
(navsim v2, `--split warmup_two_stage|navhard_two_stage`, navhard as 6 whole-group shards + `merge_shards.py navhard`, recipe in
`…/2026-10-04-navsim-thor-backend/RESULT.md` §"data"/§"Job argv"). Assets NOT yet on Thor (dev-box sizes): `C:/Users/Admin/navsim-crun/exp/metric_cache_navhard_two_stage` (1.3 GB),
`…/navsim-crun/data/openscene/navhard_two_stage/synthetic_scene_pickles` (857 MB), `D:/Archive/devbox-C/navsim/exp/metric_cache_warmup_two_stage`,
`…/navsim-crun/data/openscene/warmup_two_stage/synthetic_scene_pickles` (27 MB). Then `parse8.py` (HANDOFF_NAVSIM_R8.md §9) → bars8 with both summaries.
Warmup stage-1 rows are DECLARED CV stand-ins by design (no stage-1 warmup bank; S2-EPDMS-u reads stage 2) — same as refcv7.
**(e)** Update GOALS_AND_CLAIMS A-R8-NS1 + MODEL_REGISTRY in the same turn; bank this package (qland).

## 5. Not done here (named)

* **RAW weights beside** (lowest priority): not run — `ckpt_48852.pt` is not on Thor (`E:/refcv8_eval_kit/ckpt/`, 403 MB ESTIMATED);
  ship + md5, then `r8nt_navtest_read_LAUNCH.sh` needs a RAW variant (arm R8_LEGAL only, label BESIDE). Never a bar.
* **Ego-Status-MLP rows:** UNAVAILABLE (D-NS8-2 not approved; nothing downloaded). ⛔ No leaderboard submission (D-NS8-4 open).
* EMA-40k was never read on navtest (only 48,852 is in the launch script; G0 pins its md5).
* Integration: this package lives only in `C:/Users/Admin/refcv8_pf/navtest_read/` (local) + Thor; the Master Mind lands it via qland.
