# PREREG_CRAWLLEAD: do crawl states sit in scenes with a lead agent?

Written 2026-11-05 before any set statistic was computed. The only quantities looked at before this file was frozen are schema facts (array names, shapes, dtypes, key coverage of the clip join, source code of the join). No L40, no set size, no v_GT distribution of any set had been computed.

## Question

A vision lever that reads other agents' motion can only help in the crawl if crawl states sit in scenes with a lead agent. Do crawl states have a lead agent more often than comparable non-crawl moving states?

## Data (read only, TRAIN clips only; eval-141 is not touched)

* Own-state bank tables (rollouts of the v8F model on 4,512 TRAIN clips): `E:/Projects/TanitAD-artifacts/dagland/bank_presteps/raw/bank_table_s0.npz` and `bank_table_s1.npz` (613,132 records each; ticks 1 and 2 pooled, as in the bank statistics of report section 7.30).
* Usable set: `reason == 0` (OK). Source: `bank_presteps/scripts/common.py` line 9: `OK, OFF_LAT, OFF_HEAD, END_EDGE, NONFINITE = range(5)`; `regret_prestep.py` line 24: `usable = tab["reason"] == C.OK`. Same set as the bank statistics.
* Logged speed v_GT(r): the bank statistics read it from `bank_presteps/raw/logged_cache.npz` key `V`, built by `build_logged_cache.py` from the native 0.1 s tracks with `bankio.grid_of` (`tr[0::2]`, i.e. grid step r is raw frame 2r; `bankio.py` lines 64-67). `v_gt_r` is attached by `common.attach_logged` (`common.py` lines 22-35) as `V[offs[clip] + r]`. The TRAIN cache `F:/TanitAD-scratch/speedread/cache/train/clips/<clip>.pt` key `v` is the same quantity (its `meta.units.pose_grid` states "frame 2j"). Primary source: `logged_cache.npz`. Cross-check (control C5): the speedread cache `v` on a seeded sample of 150 clips must agree.
* Lead labels: `F:/TanitAD-scratch/motion_c/crawllead/in/lead_train4572.npz` (copy of `D:/tanitad/data/v9/lead_train4572.npz`; 849,263 rows, 4,427 clips; keys `sha12, k, has, lead, gap, tg, s, v, meta`).

## What the lead-file fields are (from the builder source `stack/scripts/build_v9_labels.py` at commit b07982c0)

`cmd_lead_pass` (lines 1111-1139) writes one row per (clip, raw frame k): `sha12`, `k`, `has` (constant 1: a row exists), `lead` (0/1), `gap`, `tg`, `s`, `v`. Per row these come from `lead_for_frame` (line 507-538), which returns `(lead?, gap_m, time_gap_s, s_along_m, v_ego)`:
* `lead` = an agent of class automobile, heavy_truck, bus, trailer, other_vehicle, train_or_tram_car or rider lies within 1.75 m of the ego's smoothed path and between 2 and 100 m along it; the nearest such agent along the path is the lead.
* `s` = along-path distance (m) from the ego to the lead agent's centre. Not a speed.
* `gap` = `s - 0.5 * lead_length - 3.6` (line 537): bumper-to-bumper gap in m; can be negative.
* `tg` = `max(gap, 0) / max(v, 1)` (line 538): time gap in s, computed with the logged EGO speed.
* `v` = the logged ego (human) speed at that frame, from `sample(log, t)` (line 508-509; the fifth return value, line 538). It is NOT the lead's speed. The lead file carries no lead speed.
* `gap`, `tg`, `s` are NaN when `lead == 0`.

## The join (identical to the trainer's)

Trainer `stack/tanitad/data/v8f_mini_data.py` at b07982c0: `sha12(clip_id) = sha256(clip_id.utf8).hexdigest()[:12]` (lines 76-77); `lead_by[sha12][k] = gap if lead == 1 else NaN` (line 179); cache step j reads raw frame `k = 2 * j` (line 192); lookup at lines 221-223. The bank's matched route point r is a cache-grid index (raw frame 2r), so the lead context of a record is the lead-file row `(sha12(clip), k = 2 * r)`.

## Sets (per bank seed s0, s1; v_own = bank `v_own`)

* CRAWL: usable, finite v_own and v_GT(r), `v_GT(r) >= 3.0` and `v_own < 0.5 * v_GT(r)` (strict and non-strict as written; same mask as `regret_prestep.py` lines 24-26).
* MOVING-OK: usable, finite, `3.0 <= v_GT(r) <= 8.0` and `v_own >= 0.8 * v_GT(r)`.
* LABELLED record: its lead-file row exists (clip present in the lead file and a row at k = 2r). UNLABELLED records (clip absent from the lead file, or no row at that k) are unknown, not "no lead". Rule frozen here: they are EXCLUDED from the denominators of the decision statistic, and counted per set.

## Statistics

* Lead within d metres: `lead == 1 and gap <= d` (gap is NaN when lead == 0; a negative gap counts as within d).
* L40(S) = (labelled records of S with a lead within 40 m) / (labelled records of S). L20 and L60 likewise.
* DIFF = L40(CRAWL) - L40(MOVING-OK).
* Interval: clip-cluster bootstrap, B = 2000, seed 0, `numpy.random.default_rng(0).choice(4512, size=(m, 4512), replace=True)` in chunks of 200, 95 % percentile, resampling the 4,512 clips as clusters (clips with no record in a set contribute zero). Each share is a ratio of per-clip sums; DIFF resamples the four sums with the same clip draw. It answers another draw of CLIPS only: the bank rollouts, the lead labels, and (across the two seeds) the 4,512 clips are held fixed; the two bank seeds share the same clips and the same windows, so they are not independent replications of the clip draw.

## Committed outcome (decision statistic: L40 and DIFF at the point estimate, labelled records, required at BOTH seeds)

* **AGENT-TARGET** if L40(CRAWL) >= 0.50 AND DIFF >= 0.10.
* **NO-AGENT-TARGET** if L40(CRAWL) < 0.25.
* **PARTIAL** otherwise.
* If the two seeds give different verdicts, the overall verdict is PARTIAL (seeds disagree). If both give the same verdict, that is the overall verdict.
* If a decision threshold lies inside the 95 % interval of its statistic, that is stated next to the verdict; it does not change the verdict.

## Implication written in advance

* AGENT-TARGET: a pixel-motion perception arm (a vision lever that reads other agents' motion) has a target in the crawl.
* NO-AGENT-TARGET: it has none, and option (c) closes at the forms measured.
* PARTIAL: the lead-agent target covers part of the crawl; the share of crawl without a lead within 40 m is stated, and the lever's ceiling is bounded by L40(CRAWL).

## Reported beside (not deciding)

1. Per-seed n: usable records, CRAWL, MOVING-OK, labelled and unlabelled counts per set, number of clips with at least one CRAWL record.
2. v_GT distribution of CRAWL: percentiles 5, 25, 50, 75, 95 and the shares in [3,5), [5,8], (8,15], >15 m/s.
3. L20, L40, L60 for CRAWL and MOVING-OK, and DIFF at each.
4. Speed-matched comparison: L40 of CRAWL restricted to `3 <= v_GT <= 8` against L40(MOVING-OK) (same v_GT window), so the DIFF is not carried by a different v_GT mix.
5. Among CRAWL records with `lead == 1` (any gap): median and quartiles of gap, of the file's time gap `tg` (uses logged ego speed), and of a time gap computed with the model's own speed, `max(gap,0)/max(v_own,1)`.
6. Lead speed: not in the file. ESTIMATED from the gap sequence of the same clip: `v_lead_est = v_ego_logged(k) + (gap(k+2) - gap(k-2)) / 0.4`, valid only if `lead == 1` at k-2, k, k+2 and `|gap(k+2) - gap(k)| <= 4 m` and `|gap(k) - gap(k-2)| <= 4 m` (agent-identity continuity guard: relative speed below 20 m/s over 0.2 s; fixed here, not tuned). Report the share of CRAWL-with-lead records for which the estimate is valid, the median and quartiles of v_lead_est, and the share with `v_lead_est < v_GT(r)` (lead slower than the logged ego speed at the matched point) and with `v_lead_est < v_own`. Evidence class: ESTIMATED (single-lead identity assumed), not MEASURED.
7. Sensitivity of the verdict to unlabelled records: (a) trainer semantics, unlabelled treated as no lead (all records in the denominator); (b) bounds, all unlabelled records of CRAWL counted as lead within 40 m and as no lead.
8. Tick split (1 vs 2) of L40(CRAWL).

## Controls (must read the stated value, or the output is reported UNVERIFIED)

* C1: CRAWL mask replaced by the MOVING-OK mask gives DIFF exactly 0.
* C2: lead flag forced to all-ones gives L40 = 1.0 exactly for both sets; forced to all-zeros gives 0.0 exactly.
* C3: clock alignment of the join: over labelled usable records, the lead-file ego speed `v` at k = 2r against `v_GT(r)`: median absolute difference <= 0.05 m/s and 95th percentile <= 0.25 m/s.
* C4: deliberate-regression arm of C3: the same comparison at k = 2r + 2 (one grid step off) must give a larger median absolute difference than the correct join.
* C5: speedread cache `v` against `logged_cache.npz` `V` on a seeded sample of 150 clips (all grid steps with finite values): maximum absolute difference <= 0.01 m/s.
* C6: CRAWL and MOVING-OK are disjoint (no record in both).

## Hard limits / provenance

Dev-box CPU only, threads <= 8, TRAIN data only. Staged, not committed. Code in `code/`, raw JSON in `raw/`, verdict in `RESULT.md`.
