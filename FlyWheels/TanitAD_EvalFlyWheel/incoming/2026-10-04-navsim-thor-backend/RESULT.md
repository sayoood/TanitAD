# NavSim CPU scoring backend on Thor — RESULT (EvalFlyWheel, 2026-10-04)

## Headline

**The backend is built and runs, but it FAILS the pre-registered bar (max abs diff 0.0 on every
column). So no production arm was scored, and `THOR_BACKEND_VALIDATED.json` was NOT written.**
The bar result is reported as written; the verdict is in `raw/THOR_BACKEND_VALIDATION.json`.

* **navhard R7_A1 @ 50,400 (the brief's control).** The first run FAILED. A refused AgentInput
  fingerprint made 6 of 204 rows invalid, and the two-stage aggregation crashed. With the candidate
  fingerprint fallback, the run is **EXACT**: 204 tokens × 20 columns, 0 text-differing cells, in the
  final frame, the pre-aggregation frame and the devkit CSV. I ran it twice; both were exact.
* **The control generalises imperfectly** (my extra arms, same 204 tokens). On 2 of 6 navhard arms
  the aggregator `weight` column differs: 13 cells max 6.3e-18 and 1 cell max 3.7e-96. All sub-scores,
  `score`, `two_frame_extended_comfort` and endpoints are exact on all 1,224 token-rows. The devkit CSV
  is exact on 6 of 6.
* **navtest FAILS the bar.** All 6 arms on W3's paired 200-token subset differ in `ego_progress` and
  `score` on 1–2 tokens (max 7.8e-13). On the **full 12,146-token split** (R7_A1 @ 30,000):
  * 34 tokens (0.28 %) differ, max \|Δ\| 1.2e-11 on EP and 4.9e-12 on PDMS.
  * **0 discrete flips.** The split means move by 9.0e-16 (EP) and 1.5e-16 (PDMS).
* **Root cause, measured: libm, not the port.** On identical inputs, float64 `exp` / `cos` / `sin` /
  `arctan2` differ by ≥ 1 ulp on 0.51 % / 3.29 % / 3.18 % / 0.125 % of values. That is the Windows CRT
  (dev box: x86-64, AVX2/FMA3, no AVX-512) against glibc 2.39 on aarch64 (`raw/libm_probe_summary.json`).
  Every Python package that does arithmetic has the same version on both machines (numpy 1.23.4,
  scipy 1.13.1, shapely 2.0.7/GEOS 3.11.4, pandas 2.3.3, pyproj 3.6.1/PROJ 9.3.0, pyogrio 0.11.1/GDAL
  3.10.3). The devkit trees and all data are tree-hash identical.
* **There is no technical lever left that reaches 0.0.** A bit-exact result would need the MSVC CRT's
  transcendentals on Thor, which means emulating x86 Windows Python. That is far over the 1.5 GB disk
  budget. ⇒ **This is a Master Mind decision** (see *Decision needed*).
* **Full navhard split** (R7_A1 @ 50,400, 5,912 tokens, 6 whole-group shards):
  * 54 tokens (0.91 %) differ in the final frame. Max \|Δ\| is 1.0e-11 on EP, 3.3e-12 on `score` and
    2.2e-19 on `weight`.
  * **0 discrete flips.** The split means move by ≤ 1.1e-15.
  * Merged back to one devkit CSV, the three `extended_pdm_score_*` rows move by ≤ 5.7e-15. **EPDMS
    reads 20.2174 ×100 on both machines.**
* **Throughput (MEASURED).**
  * navhard, full split: **0.81–0.98 s/scene per process with 6 concurrent, 7.27 scenes/s
    aggregate**, against **0.78 scenes/s** on the dev box (7,579 s per arm). **One full arm took
    21.3 min wall**, about 6× faster.
  * The 204-token subset is heavier per scene (2.04 s/scene alone and with 6 concurrent), because
    18 of its scenes are stage-1 scenes, which run a human-penalty double rollout.
  * navtest: 0.105 s/scene alone and 0.136 with 6 concurrent, **44 scenes/s aggregate**. The full split
    took **8.4 min wall**, against 1,601 s single-process on the dev box when it does not abort.
* **What the scorer opens (MEASURED with an audit hook, since Thor has no strace;
  `raw/opened_files_audit_navhard.json`).**
  * Opened: the cache metadata and one pickle per token; all 5,462 synthetic pickles and all 76
    navhard logs (the loader reads them at init); the 4 map `.gpkg` files and the maps json.
  * Never opened: **0 sensor_blobs** (the 12 GB set is not on Thor) and 0 `.npz` files.
* **D6 fan scorer on Thor (unchanged code): 8 tokens, 6,824 leaf values, 0 differ.** These include
  Tier-C exact `pdm_score` sub-scores and the K5 singles.

## Results, per control

### V-NH: navhard two-stage, step 50,400, R7_A1, 204 tokens = 9 whole groups (`raw/navhard_subset_200.json`)

| run | agent | result | evidence |
|---|---|---|---|
| run 1 (as-is) | E2 `TanitADSeamAgent`, byte fingerprint | **FAIL.** 6/204 `FINGERPRINT_MISMATCH` made the rows invalid. The aggregator then raised `AssertionError: Invalid interval nan`, so there is no final frame. On the 198 scored tokens, every pre-aggregation cell is bit-identical. | `raw/val_navhard_R7_A1_s50400_run1/`, `raw/val_navhard_run1_PRE_AGGREGATION_diff.json` |
| diagnosis | — | 1-ulp `np.cos(θ)` / `arctan2(sin,cos)` in the AgentInput relative transform, carried to ≤ 27 ulp in x/y. For a seam arm the AgentInput is **not a scoring input**: it only feeds the fingerprint. | `raw/diag_agent_input_{devbox,thor}.json` |
| run 2 | fallback v1: ≤ 64 ulp against a 6-token dev-box reference | **EXACT** on 204 × 20: final frame, pre-aggregation frame and devkit CSV | `raw/val_navhard_R7_A1_s50400_ulp/`, `raw/V-NH_*_diff.json` |
| run 3 (final agent) | fallback: atol 1e-9 against the full 481-token reference | **EXACT** on 204 × 20 (frame + CSV). 6 waivers, max \|Δ\| 8.9e-16 | `raw/pool_nh/vnh_s50400_R7_A1/`, `raw/V-NH6_s50400_R7_A1_*` |

**Six arms, same 204 tokens, run 6-way concurrently** (`raw/pool_nh/`, `raw/V-NH6_*`):

| arm | step | seam sha256 (= dev-box counts) | final frame | devkit CSV |
|---|---|---|---|---|
| R7_A1 | 50,400 | 996411cd7e | EXACT | EXACT |
| R7_A1_s1 | 50,400 | ad6a098194 | **`weight` 1 cell, 3.7e-96** (value ~3e-80) | EXACT |
| R7_A1 | 30,000 | aa26508a71 | EXACT | EXACT |
| PRIOR_ha0p | 30,000 | 8707b881bd | EXACT | EXACT |
| R7_A1_s1 | 30,000 | 0e87be674e | **`weight` 13 cells, max 6.3e-18** | EXACT |
| R7_CEILDECL_d | 30,000 | b587f4b597 | EXACT | EXACT |

`weight` is the Gaussian kernel `np.exp(-d²/2σ²)` of `SceneAggregator`, which is the `exp` row of the
libm probe. It does not appear in the devkit CSV's per-token columns. It does enter the
`extended_pdm_score_stage_two` summary.

**Full split, navhard** (R7_A1 @ 50,400, 5,912 tokens, 6 group shards): _see the **Full splits** section below._

### V-NT: navtest v1.1, step 30,000, W3's paired 200-token subset `A1_sub200_tokens.json`

Seams were proven to be the scored ones: for 6 of 6 arms, the 200 rows are bit-identical to the
dev-box hook's recorded agent poses (`raw/navtest_val_seam_vs_hooks_*.json`). AgentInput
fingerprints: **0 / 12,146 navtest tokens differ** (`raw/fp_audit_navtest_all.json`), so W3's
`SeamAgentV1` runs as-is.

| arm | result | differing columns (max \|Δ\|, cells) |
|---|---|---|
| R7_A1 | NOT_EXACT | ego_progress (6.6e-13, 1), score (2.8e-13, 1) |
| PRIOR_ha0p | NOT_EXACT | ego_progress (7.8e-13, 2), score (3.2e-13, 2) |
| R7_A1_s1 | NOT_EXACT | ego_progress (6.6e-13, 1), score (2.8e-13, 1) |
| R7_CEILDECL_d | NOT_EXACT | ego_progress (6.6e-13, 1), score (2.8e-13, 1) |
| R7_FILTOFF_sub | NOT_EXACT | ego_progress (6.6e-13, 1), score (2.8e-13, 1) |
| R7_VMAXOFF_sub | NOT_EXACT | ego_progress (6.6e-13, 1), score (2.8e-13, 1) |

The difference sits in the raw progress, `progress_raw_m`, for example 36.57543166795785 against
36.57543166792947 on the PDM-Closed reference. That value is the simulated path length after the
LQR/bicycle rollout, and the rollout calls `cos`/`sin` at every step.

### Full splits: the decision-grade measurement

| split / arm | tokens | tokens differing | max \|Δ\| | discrete flips | split-mean Δ | evidence |
|---|---|---|---|---|---|---|
| navtest R7_A1 @ 30,000 | 12,146 | **34 (0.28 %)** | EP 1.2e-11, PDMS 4.9e-12 | **0** | EP 9.0e-16, PDMS 1.5e-16 | `raw/FULL-NT_s30000_R7_A1_compare.json`, `raw/full_nt/` |
| navhard R7_A1 @ 50,400 (final frames) | 5,912 | **54 (0.91 %)** | EP 1.0e-11, pdm_score 3.7e-12, score 3.3e-12, weight 2.2e-19 | **0** | ≤ 1.1e-15 | `raw/FULL-NH_s50400_R7_A1_frame_compare.json`, `raw/full_nh/` |
| navhard, merged devkit CSV | 5,912 | 28 (0.47 %) | EP 1.0e-11 | **0** | summary rows ≤ 5.7e-15; EPDMS ×100 20.2174 on both | `raw/FULL-NH_s50400_R7_A1_merged_csv_compare.json` |

On the full navhard run the fingerprint fallback waived 481 tokens (all synthetic). Every one of
them re-hashed to the seam fingerprint, at max \|Δ\| ≤ 3.6e-15.

⚠ **A merge-tool artefact was caught and fixed.** The first merge parsed CSVs with pandas' default
float parser, which is not round-trip exact. It moved about 14 % of navhard tokens by 1 ulp. With
`float_precision="round_trip"`, the merged artifacts carry exactly the shard values.

Merging shards back into single-run artifacts (`code/merge_shards.py`): the merged navtest CSV has
the same header and the same token order as the dev-box CSV. Its `average` row is bit-identical on
NC/DAC/TTC/C/DDC and differs by 8.9e-16 (EP) and 4.4e-16 (PDMS). At the programme's ×100, 4-dp
reporting, both read **71.8849**.

## Decision needed (Master Mind / PI): this blocks production on Thor

The strict bar cannot be met across platforms; the measurements above are the evidence. Three options:

1. **Tolerance policy.** Admit Thor rows if, on a full-split control, they show 0 discrete flips,
   per-token \|Δ\| ≤ 1e-9 on continuous columns, and \|split-mean Δ\| ≤ 1e-12. Tag every Thor-scored
   artifact `backend: thor`. Never pair a Thor arm with a dev-box arm inside one paired bootstrap
   unless that tolerance is accepted for paired deltas too. Measured headroom: about 100× on per-token
   values (1.2e-11 vs 1e-9) and about 1,000× on split means (9e-16 vs 1e-12).
2. **Backend-consistent only.** Thor scores whole milestones: every arm that enters one comparison
   is scored on the same backend. That needs re-scoring the reference arms on Thor (cheap: 8.4 min
   per navtest arm and ~40 min per navhard arm).
3. **Keep the strict bar.** Thor does no NavSim production; the dev box stays the only scorer.

The fingerprint fallback (`code/tanitad_seam_agent_ulp.py`) is a **harness-guard change**, and it is
needed on navhard under options 1 and 2. 8.1 % of navhard tokens are refused as-is
(`raw/fp_audit_navhard_all.json`). It needs explicit acceptance. Its atol of 1e-9 was chosen **after**
seeing max \|Δ\| 3.6e-15 (`raw/ulp_reference_check_navhard.json`).

## Production status: none run, two independent blockers

1. **The bar above.** Per the brief: no production before both validations pass.
2. **Claims.** Every step-50,400 arm is claimed by `devbox` in `…/step50400/CLAIMED_ARMS.txt`. That
   covers navtest R7_A1, R7_A1_s1, R7_CEILDECL_d, PRIOR_ha0p and R7_VMAXOFF; navhard R7_CEILDECL_d,
   PRIOR_ha0p and R7_VMAXOFF; and all warmup arms. I therefore wrote no Thor claim. At 23:52 local the
   dev box had banked navtest R7_CEILDECL_d PASS; navtest R7_A1, R7_A1_s1 and PRIOR_ha0p were still
   FAIL (partial), and navhard PRIOR_ha0p and R7_CEILDECL_d were FAIL.

## Memory per scorer (peak RSS, wrapper's own monitor; MEASURED)

| job | peak RSS |
|---|---|
| navhard, 204-token subset | 657–664 MB |
| navtest, 200-token subset | 734–743 MB |
| navtest, one shard of 2,025 tokens over ~135 logs | 1.47–1.49 GB |
| navhard, one full-split shard (870–1,104 tokens) | 738–780 MB |

Every scorer's peak RSS is in `raw/peak_rss_per_scorer.json` (27 scorer processes; max 1,488 MB)
and is carried into `raw/THOR_BACKEND_VALIDATION.json`.

**Coordinator safety update (22:57 global OOM caused by another agent), applied.** The
`code/thor_pool.py` defaults are now **4 scorers**, a launch floor of **MemAvailable ≥ 35 GB** and a
SIGTERM floor of **20 GB**. The full-navhard validation pool had already been launched with 6
scorers (~4.5 GB total RSS). I covered it with an extra watchdog at the 20 GB floor, killing by PID
only (`raw/full_nh/extra_watchdog.out`). It never fired; MemAvailable stayed ≥ 32.8 GB throughout.

**State left on Thor.**
* Disk 1.10 GB, df 23 G free: `/home/nvidia/venvs/navsim-cpu`, `/home/nvidia/venvs/navsim-cpu-python`
  and `/home/nvidia/navsim-thor`. The thor code is sha256-identical to this package's `code/`.
* tmpfs: **data freed**, so RAM is not held on an OOM-prone box. 118 MB of seams, subsets, outputs and
  pool state remain in `/dev/shm/navsim`. Data must be re-copied before any run (recipe step 0, ~3–6 min).
* No process of mine is alive.
* The `~/.local/bin/python3.9` symlink that uv created is removed.

MemAvailable never dropped below 34.7 GB during any run. The training on Thor (ddv2_rl_refcv7 plus
a `pbox_arms.py` job) ran throughout; I never touched it. The GPU was never used: CPU-only torch
2.0.1 and `CUDA_VISIBLE_DEVICES=""`.

## Recipe: run any arm (or a D6 fan shard) on Thor

The venv and `/home/nvidia/navsim-thor/{src,code}` are in place on Thor. **The tmpfs data was
freed**, so step 0 is mandatory. The pool now defaults to **4 scorers / 35 GB launch / 20 GB kill**,
so use 4 shards, or 6 shard files with the pool capping at 4 alive. ⚠ **Do not run until the Master
Mind has (a) chosen one of the options under *Decision needed* and (b) assigned the arm in
`CLAIMED_ARMS.txt`.**

```bash
# 0) data (dev box -> Thor tmpfs, ~63 MB/s; keep /dev/shm/navsim <= 7 GB: navhard set OR navtest set)
#    navhard: exp/metric_cache_navhard_two_stage (1.2G) + data/openscene/navhard_two_stage/synthetic_scene_pickles (0.84G)
#    navtest: exp/w3_navtest_v1/metric_cache_navtest (3.0G, from D:/Archive/devbox-C/navsim/exp/w3_navtest_v1)
#    always:  data/openscene/navsim_logs/test (147 logs, 0.98G) + data/maps (1.28G, Intensity excluded)
cd C:/Users/Admin/navsim-crun/exp && tar -cf - metric_cache_navhard_two_stage | ssh tanitad-thor-wifi 'tar -xf - -C /dev/shm/navsim/exp'
# 1) seam -> Thor (verify sha256 against the arm's counts.json / bridge manifest)
scp <pkg>/raw/milestones/step<N>/bridge_<split>/seam_<ARM>.npz tanitad-thor-wifi:/dev/shm/navsim/seams/step<N>/seam_<ARM>__<split>.npz
# 2) PRODUCTION ONLY: append "<split>:<ARM> thor <UTC>" to …/step50400/CLAIMED_ARMS.txt; skip arms claimed by devbox
# 3) jobs file (one JSON per line: {"name","argv","cwd","log"}; examples raw/full_nh_jobs.jsonl, raw/full_nt_jobs.jsonl)
#    -> pool (<=4 scorers, launch only if MemAvailable>=35 GB, SIGTERM own PIDs <20 GB / NAVSIM_YIELD)
ssh tanitad-thor-wifi 'cd /dev/shm/navsim/pool && nohup setsid python3 /home/nvidia/navsim-thor/code/thor/thor_pool.py \
    --jobs jobs.jsonl --state state.json > pool.out 2>&1 < /dev/null &'      # monitor: tail -c 30 pool.out -> ZZdone-fail-alive-queued-availGBZZ
```
Job argv per split (`python` = `/home/nvidia/venvs/navsim-cpu/bin/python`, code in `/home/nvidia/navsim-thor/code/thor/`):
* **navhard shard k of 6.** The shard files are `raw/shards/navhard_shard_{k}of6.json` → `/dev/shm/navsim/subsets/`.
  ```
  python score_arm_thor.py --arm <ARM> --seam /dev/shm/navsim/seams/step<N>/seam_<ARM>__navhard.npz \
    --split navhard_two_stage --out /dev/shm/navsim/out/<run>/<ARM>_k{k}of6 --exp-tag e7s<N>k{k} \
    --tokens /dev/shm/navsim/subsets/navhard_shard_{k}of6.json \
    --agent-target tanitad_seam_agent_ulp.TanitADSeamAgentULP \
    --extra-override +agent.ulp_reference=/dev/shm/navsim/subsets/ulp_reference_navhard_devbox.json --extra-override +agent.atol=1e-9
  ```
  Then merge on the dev box: `merge_shards.py navhard --frames <6 final frames> --order-ref <any dev-box navhard CSV> --yaml <navhard_two_stage.yaml> --out-csv … --out-frame …`.
* **navtest, whole split, one process (~22 min).** This gives native artifacts with no merge.
  ```
  python score_navtest_thor.py --label r7s<N>_<ARM> --seam /dev/shm/navsim/seams/step<N>/seam_<ARM>__navtest.npz --out /dev/shm/navsim/out/<run>
  ```
  To use 6 shards (8.4 min) instead, add `--tokens /dev/shm/navsim/subsets/navtest_shard_{k}of6.json`
  and run `merge_shards.py navtest`.
* **D6 fan shard.** Unchanged D6 code is at `/home/nvidia/navsim-thor/code/d6`. Put the fan, hooks and
  token files in `/dev/shm/navsim/d6/`. On Linux, file names are case-sensitive: the file is
  `spec_tokens_P1X_all.txt`.
  ```
  CUDA_VISIBLE_DEVICES= PYTHONHASHSEED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=/home/nvidia/navsim-thor/code/stubs \
  python d6_thor.py --d6-code /home/nvidia/navsim-thor/code/d6 -- --fan <fan.jsonl> --hooks <hooks.json> --out <scored.jsonl> \
     --shard k/n --only-tokens <…> --exact-ids <…> --k5 40 [--shift-states 30 --no-tierc]
  ```
* Pull results back with `scp -r tanitad-thor-wifi:/dev/shm/navsim/out/<run> <dev-box dir>`, and add a
  `.thor` provenance note next to them. Every counts/`.thor.json` already records `backend: thor`,
  the driver, the agent sha256, the seam sha256 and the overrides.
