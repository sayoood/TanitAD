# RESULT: do crawl states sit in scenes with a lead agent?

Pre-registration `PREREG_CRAWLLEAD.md` (sha256 `72f86313...8606`, `FROZEN_SHA.txt`, re-verified after the run). All numbers: MEASURED, `raw/crawllead_result.json` (code `code/crawllead.py`), unless marked ESTIMATED. Tier: T1-kin bank states of the v8F model on 4,512 TRAIN clips (own-state rollouts); a statistic about where crawl states sit, no capability claim. Eval-141 is not involved.

## Verdict (rule as written)

| bank seed | L40(CRAWL) [95 % CI] | L40(MOVING-OK) [95 % CI] | DIFF [95 % CI] | verdict |
|---|---|---|---|---|
| s0 | 0.4639 [0.4341, 0.4919] | 0.2409 [0.2220, 0.2596] | +0.2230 [0.1935, 0.2524] | **PARTIAL** |
| s1 | 0.4956 [0.4628, 0.5294] | 0.2687 [0.2490, 0.2877] | +0.2269 [0.1945, 0.2622] | **PARTIAL** |

**Overall: PARTIAL** (both seeds PARTIAL, so the seeds agree). Neither seed is NO-AGENT-TARGET (L40(CRAWL) is about 0.46 to 0.50, far above 0.25). Neither is AGENT-TARGET, because L40(CRAWL) misses 0.50: s0 by 0.036 (CI upper bound 0.492 stays below 0.50); s1 by 0.0044, a near-tie with 0.50 inside its CI (flagged as the rule requires; the verdict stays PARTIAL). The DIFF condition (>= 0.10) is met at both seeds with margin (CI lower bounds 0.19).

Reading: crawl states have a lead agent within 40 m about 1.8 to 1.9 times as often as comparable moving states (0.46/0.50 against 0.24/0.27), and about half of all crawl states have no lead within 40 m. The contrast is real, the coverage is partial.

Interval estimator: episode (clip) cluster bootstrap, B = 2000, seed 0, over the 4,512 clips, 95 % percentile; DIFF resamples the same clip draw for both sets. It answers "would another draw of clips say this?" only. The bank rollouts and lead labels are held fixed, and the two bank seeds use the same 4,512 clips and windows, so they are not independent replications of the clip draw.

## n (per seed)

| | s0 | s1 |
|---|---|---|
| usable records (reason 0) | 407,450 | 414,765 |
| CRAWL (labelled / unlabelled), clips | 23,435 (22,906 / 529), 2,075 clips | 10,818 (10,559 / 259), 1,447 clips |
| MOVING-OK (labelled / unlabelled), clips | 69,065 (66,695 / 2,370), 2,447 clips | 86,044 (83,356 / 2,688), 2,524 clips |

CRAWL counts equal the bank-statistics counts in `bank_presteps/regret_prestep.json` (23,435 and 10,818, ticks 1+2): same set. Unlabelled = lead-file row absent; excluded from denominators as pre-registered. Cause, s0 CRAWL: 385 of 529 are in the 143 bank clips that have no rows in the lead file, 144 are missing frames inside clips that do (s1 CRAWL: 182 and 77).

## The join and the lead-file fields (from source)

* Clip key `sha12(clip) = sha256(clip.utf8).hexdigest()[:12]` (`stack/tanitad/data/v8f_mini_data.py:76-77`, commit b07982c0); cache step j reads raw frame `k = 2*j` (line 192); `lead_by[sha12][k] = gap if lead == 1 else NaN` (line 179); lookup at lines 221-223. The bank's matched row r is a cache-grid index, so a record reads row `(sha12(clip), k = 2r)`. Logged speed `v_GT(r)` = `logged_cache.npz V[offs + r]` (`bank_presteps/scripts/common.py:22-35`, grid = even raw frames, `bankio.py:64-67`).
* `s` = along-path distance from ego to the lead agent's centre (m); `gap = s - 0.5*length - 3.6` (`stack/scripts/build_v9_labels.py:537`); `tg = max(gap,0)/max(v,1)` (line 538); **`v` = the logged EGO speed at that frame, not the lead's speed** (`lead_for_frame` returns `v_ego`, lines 507-538). The lead file holds no lead speed. `has` is constant 1 (row exists); `lead` (0/1) carries presence. A lead is the nearest automobile/truck/bus/trailer/other-vehicle/tram/rider within 1.75 m of the ego's logged path, 2 to 100 m ahead.
* The lead context is the logged scene at the matched point, so it depends on (clip, r), not on the model's rollout.

## Beside (not deciding)

* **L20 / L60** (CRAWL | MOVING-OK | DIFF), s0: 0.321 | 0.153 | +0.168 and 0.517 | 0.293 | +0.224; s1: 0.361 | 0.173 | +0.187 and 0.551 | 0.320 | +0.231. L60 of CRAWL is 0.52 (s0) and 0.55 (s1).
* **v_GT of CRAWL** (m/s), s0: p5 3.34, p25 4.83, median 6.89, p75 9.34, p95 19.31; shares [3,5) 27 %, [5,8] 36 %, (8,15] 30 %, >15 7 %. s1: p5 3.27, p25 4.47, median 6.16, p75 8.37, p95 12.63; shares 33 %, 39 %, 25 %, 3 %. MOVING-OK median 6.0 (window 3 to 8 by construction). Crawl states are mostly low-speed scenes (63 % at s0, 72 % at s1 below 8 m/s). Model speed over logged speed in CRAWL: median 0.40 to 0.41 (p5 0.19 to 0.20, p95 0.49).
* **Speed-matched** (CRAWL restricted to v_GT in [3,8], same window as MOVING-OK): L40 = 0.5368 [0.5004, 0.5689] (s0, n = 14,520) and 0.5559 [0.5187, 0.5940] (s1, n = 7,653); DIFF against MOVING-OK +0.296 [0.262, 0.328] and +0.287 [0.251, 0.325]. The v_GT mix does not carry the contrast; it makes it larger. This subset was pre-registered as a beside row; the decision stays on the full CRAWL set (PARTIAL), the subset is not promoted to a verdict.
* **Tick split** of L40(CRAWL), s0: tick 1 0.522 (n 7,465), tick 2 0.436 (n 15,441); s1: tick 1 0.511 (n 2,266), tick 2 0.491 (n 8,293).
* **Among CRAWL with a lead (any gap)**, s0 / s1: 13,190 (57.6 % of labelled crawl) / 6,417 (60.8 %). Gap median 17.9 m / 16.2 m (p25 11.2 / 10.6, p75 33.7 / 32.0), against MOVING-OK-with-lead median 23.3 m / 21.8 m. File time gap `tg` (logged ego speed) median 2.71 s / 2.75 s (p25 1.90 / 1.86, p75 4.49 / 4.79), against 3.91 s / 3.61 s for MOVING-OK. With the model's own speed in the denominator, `max(gap,0)/max(v_own,1)`: median 7.5 s / 7.4 s (p25 5.0 / 4.8). So crawl-with-lead scenes have a closer lead than moving-OK scenes, and the human keeps a normal time gap while the model sits far behind.
* **Lead speed: ESTIMATED, not MEASURED.** `v_lead_est = v_ego_logged(k) + (gap(k+2) - gap(k-2))/0.4` under the pre-registered identity guard (lead at k-2, k, k+2; |gap step| <= 4 m). Valid for 96.4 % (s0, n = 12,717) and 96.2 % (s1, n = 6,173) of CRAWL-with-lead. Median 6.22 m/s (p25 3.74, p75 8.72) at s0, 6.22 (3.75, 8.70) at s1. Lead estimated slower than `v_GT(r)`: 51.3 % (s0), 47.0 % (s1); for MOVING-OK-with-lead 45.5 % and 48.2 %. Lead estimated slower than the model's own speed: 12.6 % (s0), 11.7 % (s1). Lead estimated below half of `v_GT`: 15.0 % and 14.1 %. Reading: where a lead exists, it moves at about the logged ego's pace (the human follows it), and in roughly 88 % of those states it is faster than the crawling model, so the model is not holding back for a slow lead. (The central quantiles of `v_lead_est` for CRAWL and MOVING-OK agree to the second decimal while their components differ, `raw/lead_speed_sanity_s0.txt`: a coincidence of ego speed plus relative speed, not a degenerate estimator; the all-moving population gives median 9.9 m/s.)
* **Sensitivity to unlabelled records.** Trainer semantics (unlabelled counted as no lead, all records in the denominator): L40(CRAWL) 0.4534 (s0), 0.4837 (s1); MOVING-OK 0.2326, 0.2603; DIFF +0.221, +0.223. Upper bound (all unlabelled CRAWL records counted as lead within 40 m): 0.4760 (s0), 0.5077 (s1). The verdict does not move under trainer semantics; the s1 upper bound crosses 0.50 only if every unlabelled crawl record had a lead (259 records).

## Controls (all PASS)

* C1: CRAWL mask := MOVING-OK mask gives DIFF exactly 0.0 (both seeds).
* C2: lead flag all-ones gives L40 = 1.0 exactly, all-zeros gives 0.0 exactly (both sets, both seeds).
* C3: lead-file ego speed `v` at k = 2r against `v_GT(r)` over labelled usable records: median |diff| 0.0013 m/s (s0), 0.0012 (s1); p95 0.0225 and 0.0222 (n = 381,180 and 388,702). The clock of the join is aligned.
* C4 (deliberate regression): the same comparison at k = 2r+2 gives median 0.0707 / 0.0720 and p95 0.355 / 0.364, 55 to 59 times the median of the correct join: the control can tell a misaligned join.
* C5: speedread TRAIN cache `v` against `logged_cache.npz V` on 150 seeded clips (14,993 values, no length mismatch): max |diff| = 0.0 m/s. The two logged-speed sources are the same numbers.
* C6: CRAWL and MOVING-OK are disjoint.

## What this means (implication committed in the pre-registration, PARTIAL branch)

The lead-agent target covers part of the crawl. Ceiling for a lever that acts only through a lead agent: about 46 to 50 % of crawl records at 40 m (52 to 55 % at 60 m); the other half of the crawl has no lead within 40 m and needs a different cue. Option (c) (a pixel-motion perception arm) is therefore not closed at the forms measured, and it is not established as the main lever either. Within the crawl states that do have a lead, the lead is mostly moving at the logged ego's pace and faster than the model, so the usable cue there is "keep pace with the lead you can see move", not "yield". Being in a scene with a lead is a necessary condition for the lever, not evidence that the lever works.

Unverified and out of scope: whether a motion-reading arm closes the gap in those states (needs the arm); what sits in the no-lead half of the crawl (not in this file; the v9 release join carries nav-token, route-condition, constraint and speed-limit fields (docstring of `build_label_join`, `v8f_mini_data.py:149-163`) that could be joined at the same (clip, r) keys); lead identity in the speed estimate (single-lead identity assumed).
