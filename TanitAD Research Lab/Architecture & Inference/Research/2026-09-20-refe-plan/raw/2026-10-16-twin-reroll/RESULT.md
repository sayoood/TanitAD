# RESULT — P6 lane-rank twin re-roll (PI option 3): the teacher commits to the rank-K lane on ~1/3 of twins, and only a COMMIT FILTER gives a clean bank

**Master Mind, 2026-10-16 ~21:20Z.** Pre-registration `PREREG.md` (base + addenda 1-5, each written before its numbers; addendum 4 sha256
`2be86759…`, addendum 5 `c0f251dc…`). Pod CPU only (0 GPU). Evidence class: MEASURED (pod teacher rollouts, D7's UNCHANGED verify),
on the same 74 D7 mini-bank twins (K=1 n 46-50, K=2 14, K=3 9; 16 logs). Small n: every share below is a one-bank point estimate with no
interval; the CPU teacher rollout is DETERMINISTIC (148/148 duplicated rows bit-identical), so there is no inference-seed variance to add.

## What was asked
PI, 2026-10-16: *"choose option 3 faithful for the paper"* — re-roll the twins so they follow their rank-K route, apply the PDM-route-rank fix,
re-gate; P6 does NOT run on the fallback.

## What was measured (route held unless stated; f(t) = lateral fraction toward the twin lane, 1 = on it)
| arm | K=1 median f(10 s) | commit share f(10 s) >= 0.75 K1 / K2 / K3 | D7 (c) K=1 (bar 0.75) | 4 s path ratio vs held-12 s (median; share < 0.9) | end-speed change |
|---|---|---|---|---|---|
| legacy (per-step re-rank) | 0.244 | 0.152 / 0.071 / 0.000 | 0.031 | 1.000; 0.00 | +0.00 |
| held, 12 s (the bank's horizon) | 0.238 | 0.174 / 0.000 / 0.000 | 0.000 | (base) | (base) |
| held, h8 | — | — | 0.000 | 0.996; 0.06 | +0.00 |
| held, **h6** | 0.500 | **0.370 / 0.286 / 0.222** | 0.044 | **0.988; 0.17** | -0.02 |
| held, h4 | 0.500 | 0.391 / 0.357 / 0.111 | — | 0.947; 0.36 | -0.26 |
| held, h3 | 0.610 | 0.435 / 0.286 / 0.111 | 0.320 | 0.893; 0.51 | -0.66 |
| held, h2 | **0.877** | 0.543 / 0.357 / 0.111 | 0.340 | **0.810; 0.68** | **-1.32** |
| held, h3 + goal floor v0max | 0.599 | 0.435 / 0.286 / 0.111 | 0.280 | 0.903; 0.50 | -0.60 |
| held, h2 + goal floor v0max | 0.749 | 0.500 / 0.286 / 0.111 | 0.340 | 0.829; 0.67 | -1.27 |
Sources: `results/long_lever/summary.json`, `results/ms/summary.json`, `results/ms/progress_ms.json`, `results/pod_probe/verify_*`, pod
`/workspace/ops/reroll_probe/results/verify_h*_k*.json`.

## Committed outcomes, as written
1. Probe 1: **HOLD NOT SUFFICIENT** (addendum 1). 2. h6/h8: **do not clear (c)** (addendum 2). 3. 10 s diagnostic: **B — NO COMMITMENT**
at the bank's 12 s horizon (addendum 3). 4. h6/h4: **BETWEEN**; h3 BETWEEN; **h2 COMMITS** (candidate generator, addenda 3-4) — but it
**SLOWS the teacher** (path -19 %, -1.32 m/s): a confound ("twin = slower"), refused. 5. Goal-distance floor: **NO-COMMIT** (addendum 5;
its mechanism hypothesis refuted, see the addendum's outcome).
**D7 check (c) FAILS at every horizon on the unchanged instrument**, as addendum 2's analytic bound predicts: (c) asks the 4 s trajectory to be
closer to the twin lane than to the logged lane on average, which needs the WHOLE lane change to finish inside 4 s. This teacher's lane changes
take longer (median first t with f >= 0.5 among committed twins: 3-4 s). **(c) as specified cannot be passed by this teacher.**

## The per-twin finding that makes option 3 buildable
Commitment is a property of the TWIN, not noise: across horizons the commit sets are nested (h12 -> h6 -> h4: 44 never, 8 always, 14 at
h6 and h4, 2 only h4, 1 only h6). And the slowdown sits in the twins that do NOT commit:
| h6, 4 s path ratio vs held-12 s | committed | not committed |
|---|---|---|
| K=1 | **1.021, 0 % < 0.9 (n 17)** | 0.987, 22 % (n 27) |
| K=2 | **1.005, 0 % (n 4)** | 0.953, 20 % (n 10) |
| K=3 | 0.943, 50 % (n 2) | 0.968, 29 % (n 7) |
Non-committing twins include identifiable classes: low-speed frames (K=2 median v0 3.3 m/s vs 8.2 committed), twins whose two centrelines
coincide by 10 s (f exactly 0.50: merging lanes, 8 of 28 K=1 non-commits at h4), and twins that reach the lane and return (max f 0.85-0.99).

## Recommendation (needs a PI decision: it changes D7's check and the twin yield)
**Option 3-C ("commit-filtered"):** re-roll every lane-rank twin with the route held and goal horizon **6 s**, roll it **10 s**, and KEEP it
only if it commits (f(10 s) >= 0.75); the bank row is the first 4 s. Replace D7 (c) by **(c')**: on a held-out sample, every kept twin's
10 s rollout ends on the rank-K lane (f >= 0.75) and the kept set shows no slowdown (path-ratio median >= 0.95). Expected yield (one bank,
no CI): **K=1 ~37 %, K=2 ~29 %, K=3 ~22 %** of twins. Cost (ESTIMATED from these runs: ~20 s per 10 s rollout per process): ~37 min per
10,000 twins on 90 pod processes, plus the D7 rebuild.
Alternatives: **3-S** keep h2 (more yield, 54 % at K=1) and accept a slower twin — I advise against it (confound). **3-T** a rule-based
teacher (PDM-Closed) for the twins only — commits by construction, but a teacher-identity confound between twins and rank-0 rows.
**Blocked:** the bank build waits for this decision. Every other lever inside the released goal pipeline has been run.

## Artifacts (all in this package)
`PREREG.md` (+ `PREREG.addendum4.sha256`, `PREREG.addendum5.sha256`); `code/` (the patched `route_lane_rank_patch.py`, `reroll_twins.py`,
the hold guard patch); `code_pod/` (the exact pod copies that produced every number) and `code_pod_v2/` + `probe/patch_minspeed.py` (the
opt-in `--goal-min-speed`, default unchanged); `probe/` (`long_rollout.py`, `analyze_long.py`, `progress_check.py`, runner scripts);
`results/` (`long_lever/` incl. `appended_raw/` and `dedup/`, `ms/`, `rr/`, `pod_probe/`). Pod scratch `/workspace/ops/reroll_probe/`
(rollout dirs `rr_*`, `vbank_*`, `long/`, verify JSONs, logs) — copies of the decisive files are pulled here.
