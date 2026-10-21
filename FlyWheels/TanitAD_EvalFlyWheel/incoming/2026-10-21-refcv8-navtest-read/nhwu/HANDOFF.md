# HANDOFF — refcv8 EMA-48,852 warmup (N3) + navhard (N2) scoring on the Thor CPU backend (nhwu)

EvalFlyWheel, 2026-10-21 (Europe/Berlin). Nothing written to G:, nothing committed. **No refcv8 warmup/navhard number exists yet.**
Tier of everything the waiter will produce: NavSim v2 open-loop two-stage EPDMS (T1-family), zero-shot, ONE training seed; model arms
Thor CPU backend, floors = refcv7 dev-box banks → every R8-vs-floor pair is **MIXED-backend** (RULING_BACKEND_POLICY.md).

## 1. Known-value control — PASSED (`raw/control/CONTROL.json`, VALIDATION only)
* **V-NH (the brief's control):** refcv7 R7_A1 @ 50,400 (seam sha256 `996411cd…`), navhard 204-token subset, through MY job path
  (`nhwu_jobs.py` → `thor_pool.py` → `score_arm_thor.py`, ULP agent, atol 1e-9): **EXACT on 204 × 20** vs the dev-box banked frame
  AND devkit CSV; vs the banked Thor run 3 the frame, pre-aggregation frame and CSV are EXACT, CSV byte-identical (sha256 `6a1379c8…`).
* **Merge path:** `merge_shards.py navhard` over that output reproduces the native CSV + frame **byte-identically**. MEASURED trap:
  it needs `PYTHONPATH=/home/nvidia/navsim-thor/code/stubs` (else `No module named rasterio`); the waiter sets it.
* **V-WU (extra — warmup had never been scored on Thor):** run 1 with E2's byte agent **FAILED** — 19/204 synthetic tokens
  `FINGERPRINT_MISMATCH`, no CSV. Fixed with the ACCEPTED fallback (ruling item 3) + a new dev-box reference
  `raw/ulp_reference_warmup_devbox.json` (make_ulp_reference.py lite mode, navsim-crun venv; 204/204 re-hash to the seam fingerprint;
  covers all 19). Run 2: PASS 220/220; vs the dev-box bank **NOT bit-exact**: 1/220 tokens differ (EP 6.5e-13, score 2.0e-13),
  0 flips, split-mean Δ 8.9e-16, summary rows ≤ 4.4e-15 ⇒ **ADMISSIBLE under the policy**, not exact (libm class).
* Smoke dry-run of the job builder (priority 3): smoke20 → 4 warmup + 18 navhard jobs PASS; the same seams without `--smoke` are
  REFUSED (11 ≠ 5,912 rows) — the gate works.

## 2. Waiter — ARMED and verified alive
`code/nhwu_waiter.sh`, dev-box PID 2076649 (Git Bash, launched 10:30:09), log `nhwu/nhwu_waiter.log`, markers `ZZNHW<STAGE>ZZ`:
`START → ROLLDONE` (ZZALLDONE in full.out; `REFUSED` on ZZREFUSE_) `→ POOLSIDLE` (navtest + lever pool tails alive=0/queued=0, 0 live
`thor_pool` procs, AND the navtest waiter's log shows POOLDONE/POOLFAIL/JOBSBAD/REFUSED/END; fallback 3 h) `→ STAGED` (ZZSTG9-0ZZ)
`→ ARMS → JOBSOK` (warmup = every seam in bridge_warmup, navhard = every seam × 6 shards; PRIOR_ha0p included when emitted)
`→ POOLSTARTED → POOLDONE → COUNTSOK → FINALIZEOK → PULLED → SUMMARIESOK → BARSOK|BARSSKIPPED → END`.
Any `…BAD`/`POOLDEAD` = it stopped; the artifact named in the marker says why. **ESTIMATED:** ZZALLDONE ≈ 12:40, navtest pool done
≈ 12:50, my pool (≈ 22 jobs, ≤ 4 alive) ≈ 80–90 min ⇒ summaries ≈ 14:30.

Thor outputs: `/home/nvidia/r8ns/navtest_read/full/scores_warmup/` (canonical names) and `…/full/scores_navhard_thor/`
(`_shards/<ARM>_THOR_k{k}of6`, merged `<ARM>_THOR/`); jobs/reports/pool state `/home/nvidia/r8ns/nhwu/s48852e/`.
Dev box: `nhwu/raw/full/` (bridges, scores, flat `scores_navhard/` assembled by byte copy, `<ARM>.thor.json`),
**`nhwu/summaries/summary_warmup.json`, `summary_navhard.json`, `NHWU_ASSERT.json`** (README asserts: arms_refused {}, no UNAVAILABLE
arm, seed floor non-null, navhard intervals.R8_LEGAL OK, label RESULT), `nhwu/bars/` (bars8 with both summaries, iff the navtest
waiter pulled `raw/full/scores_navtest`). Check: `tail -3 nhwu/nhwu_waiter.log`.

**Resume after a failure:** `score_arm_thor.py` skips any job whose counts.json is PASS for the same seam, so re-running
`bash nhwu/code/nhwu_waiter.sh` is safe (step 1/2 pass immediately once the markers exist).

## 3. Disclosures (Master Mind to acknowledge)
1. **Warmup uses the AgentInput fallback** with a reference built today (above). The ruling's condition is generic ("a dev-box
   reference re-hashes to the seam fingerprint and Thor's values are within 1e-9"), but warmup was not in its measured 481/5,912.
2. **navhard merge runs Thor-side** via `finalize_navhard_arm.py` (the refcv7 production route; it calls the same
   `merge_shards.py navhard` and adds the count/union/seam guards), not on the dev box as the brief worded it. Validated byte-identical.
3. order-ref = refcv7 step-50,400 dev-box navhard CSV (`/home/nvidia/r8ns/nhwu/ship/ref/`, md5 `89e3cd68…`): row order only.
4. Warmup stage-1 rows are DECLARED CV stand-ins (16/220) by design, as refcv7.
5. `/dev/shm/navsim` holds 4.55 GB staged data for the waiter (MemAvailable 77–85 GB). Free it (`rm -rf /dev/shm/navsim`) after END.

## 4. Not done (named)
* GOALS_AND_CLAIMS A-R8-NS1 + MODEL_REGISTRY: not touched (G: is off-limits for this brief) — the Master Mind, once SUMMARIESOK.
* ⛔ No leaderboard submission; EMA-40k never scored; Ego-Status-MLP rows UNAVAILABLE (D-NS8-2). The GPU and its flock untouched.
* Integration: this package exists only in `C:/Users/Admin/refcv8_pf/navtest_read/nhwu/` + Thor (`/home/nvidia/r8ns/nhwu/`) — qland.
