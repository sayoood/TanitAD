# HANDOFF — refcv8 EMA-48,852 warmup (N3) + navhard (N2) scoring on the Thor CPU backend (nhwu)

EvalFlyWheel, 2026-10-21 (Europe/Berlin). Nothing written to G:, nothing committed.

## 0. OUTCOME (waiter END 14:26:08; every stage OK — `nhwu_waiter.log`)
22/22 jobs PASS, counts guards PASS, 3/3 navhard arms finalized (5,912 rows each), assemble PASS, `summaries/NHWU_ASSERT.json`
**PASS** (0 refused, 0 UNAVAILABLE arms, seed floors non-null, navhard interval OK), bars8 controls all OK (city census, EP clause,
v0 inputs, banked floors). MEASURED from `summaries/summary_{navhard,warmup}.json` + `bars/NAVSIM_BARS_step48852_ema.md`:
* **N2 navhard, official two-stage EPDMS ×100:** R8_LEGAL **17.62**, R8_LEGAL_s1 18.43, PRIOR_ha0p 13.00 vs STOP 29.85, CV 11.48, ECHO 14.29.
  Paired R8_LEGAL−STOP **−12.23 [−15.49, −8.91]** (log), drive lo −16.02 ⇒ **N2 FAILED**. Seed floor 0.11 (×100).
* **N3 warmup, S2-EPDMS-u ×100:** R8_LEGAL **48.73**, R8_LEGAL_s1 47.00 vs STOP 52.12, CV 39.71, ECHO 42.87; R8_BLIND 1.55 (fails as it
  must → panel valid). LEGAL−STOP −3.40, seed floor 1.73 ⇒ **N3 FAILED**. (Official two-stage UNAVAILABLE on LEGAL arms: stage 1 = declared CV stand-ins.)
* bars8 with both summaries: **N1, N2, N3, N4 all FAILED** (N1 PDMS 57.84 vs STOP 61.82, −3.98 [−6.37, −1.65]; N4 −12.36 vs refcv7-50,400 LEGAL).
Tier: NavSim open-loop (T1-family), zero-shot, ONE training seed; Thor-scored model arms vs dev-box floors (MIXED-backend, stated).
`/dev/shm/navsim` freed after END (no fd/cmdline referenced it). Sections 1–4 below are the pre-run hand-off, unchanged.
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
