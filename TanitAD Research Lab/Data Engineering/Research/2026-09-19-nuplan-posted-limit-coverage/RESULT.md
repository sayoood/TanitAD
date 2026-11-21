<title>nuPlan posted-limit coverage — FS19-2</title>

# FS19-2 · Posted speed limit in the nuPlan maps — coverage MEASURED

`Data Engineering · 2026-09-19 · PI-approved (download of the map DBs; a non-parity corpus may supply labels for the max-speed head). 0 GPU. D: only.`
`Builds on 2026-09-13-posted-speed-limit-supplier F6, which FOUND the field (PUBLISHED-CODE) but did not measure it.`

## Findings first

| # | finding | class |
|---|---|---|
| **1** | **Coverage is bimodal by city, not partial everywhere.** Per-lane `speed_limit_mps`, lanes + lane connectors, weighted by lane-centre length: **Las Vegas 100 %** (148.9 km), **Pittsburgh 100 %** (32.6 km), **Boston 7.1 %** (81.7 km), **Singapore 0 %** (63.1 km). All four maps: **57.4 % of 326.3 km**. By count: LV 2,039/2,039 + 2,544/2,544 · PIT 368/368 + 594/594 · BOS 88/1,230 + 139/1,789 · SG 0/1,189 + 0/1,547 | MEASURED (`raw/coverage.json`) |
| **2** | **The values are posted limits, not a fitted speed.** Every finite value lands on a **whole-mph ladder in 5-mph steps**: LV {5, 10, 15, 20, 25, 30, 35, 45}, PIT {15, 25, 35}, BOS {25}. That is an independent check: US posted limits are set in 5-mph increments, and a quantity derived from driving would not snap to them | MEASURED |
| **3** | ⛔ **`lanes_polygons.max_speed` is a PLACEHOLDER** — **25.0 on every lane in all four cities**, including km/h Singapore; `min_speed` is empty. It is unusable and must never be mistaken for the limit column | MEASURED |
| **4** | **Expected ego-frame coverage ≈ 93 %.** Train DB archive bytes per city (S3 `Content-Length`, HEAD only): Las Vegas **913.5 GB (89.8 %)**, Boston 38.2 (3.8 %), Singapore 35.0 (3.4 %), Pittsburgh 30.6 (3.0 %). Weighting lane coverage by that mix gives **0.931** | **ESTIMATED** — two proxies (bytes ≈ hours; lane length ≈ ego occupancy) |
| **5** | **The clean route needs no estimate: restrict to Las Vegas + Pittsburgh logs.** Coverage is **100 %** there by measurement, and those two cities hold **92.8 %** of the train bytes | MEASURED (coverage) + ESTIMATED (share) |

## Verdict against the pre-registration (`LAB_BACKLOG` FS19-2)
The committed branch read **"≥ 80 % of ego lane-frames ⇒ nuPlan is a full-coverage label source"**. The ego-frame quantity itself is **not measured** (it needs the navtrain logs). The map-level measurement plus the city mix **estimates 93 %**, and the LV+PIT restriction reaches **100 % by measurement**. ⇒ **The branch fires in its restricted form: nuPlan LV+PIT is a full-coverage posted-limit label source.** It does **not** fire as "nuPlan as a whole": Boston and Singapore must be excluded or masked, not imputed.
⇒ WOMD (the < 50 % branch) is **not needed**, and it would also require accepting Waymo's licence terms in a browser, which is the PI's action.

## What this does and does not unblock
* ✅ **Labels.** Under the PI ruling of 2026-09-19, a non-parity corpus may supply labels for the max-speed head. The parity train corpus is untouched.
* ⚠️ **Not yet a head.** The head is vision-only at inference and must learn *road context → posted limit* on nuPlan's cameras, then transfer to PhysicalAI (25 countries, km/h signage outside the US). That domain shift is the real risk (09-13 §2 risk a), and it is untested.
* ⛔ Still refused: any ceiling derived from ego dynamics.

## Next lever (Rule Zero) — the cheapest experiment that could still break this
**FS19-2b (0 GPU, then 4060):** pull only the **LV + PIT navtrain log metadata** (no sensor blobs) and replace estimate 4 with the measured ego lane-frame coverage. Then, on a held-out nuPlan city split (PIT held out), train a frozen-trunk linear probe *camera → posted-limit bucket*. **Committed:** held-out top-1 bucket accuracy ≥ 2× the majority-class rate ⇒ the limit is vision-readable across cities and the transfer arm to PhysicalAI earns a slot; ≤ 1.2× ⇒ the head would learn the city, not the road, and the route stops at "label source for AlpaSim/nuPlan eval only". ⚠️ The sensor subset is a separate, larger download: **named blocker, needs the PI's size OK**.

## Provenance
* Archive: `https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip`, **970,997,691 B**, sha256 `444860429f9a3bcf89a6459d683fa82eb9219aa259d8d9b8cdefdb37f0b56b05`, at `D:/Projects/TanitAD/data/nuplan-maps/` (git-ignored). Map versions: BOS 9.12.1817, SG 9.17.1964, LV 9.15.1915, PIT 9.17.1937.
* Instrument: `stack/scripts/nuplan_speed_limit_coverage.py`. ⚠️ Its **first run reported 0.0 km per city**: geometries are **EPSG:4326 degrees**, read as metres. Fixed (CRS read from `gpkg_geometry_columns`). Pinned by `stack/tests/test_nuplan_speed_limit_coverage.py` with **literal analytic targets** (3-4-5 = 5 m; 0.001° lat ≈ 110.6 m; the cos-latitude shrink) **plus the historical defect as a regression arm** — 4/4 pass. Join keys read from the schema (`baseline_paths.lane_fid → lanes_polygons.lane_fid`, `baseline_paths.lane_connector_fid → lane_connectors.fid`).
* Raw: `raw/coverage.json`, `raw/coverage_stdout.txt`.
* Archive sizes: HEAD requests to the same bucket (`nuplan-v1.1_train_{boston,pittsburgh,singapore,vegas_1..6}.zip`), retrieved 2026-09-19.
* Empty searched: the nuPlan paper `2403.04133` (read locally) gives **1,282 h over 4 cities with no per-city split**; the devkit dataset-setup page lists no archive sizes.
