---
license: other
license_name: nvidia-physicalai-av-derived
pretty_name: TanitAD v10 labels (country component)
tags: [autonomous-driving, labels, tanitad]
---
# TanitAD v10 labels — country component (v10.0)

**v10 = v9 tactical labels + v8.1 strategic labels + new components keyed by clip, by reference** (nothing is copied from v9/v8.1;
those stay in `Sayood/tanitad-refcv8-v9-labels` and `Sayood/tanitad-v7-training-corpus`). This first v10 component is the
**recording country of every v9 clip**, the label of HiCAP's strategic-level country readout (rule retrieval keys on it).

## Files
| file | content |
|---|---|
| `country/v10_country.parquet` | one row per v9 clip: `split` (train / eval139), `sha12`, `clip_id` (upstream UUID), `country`, `country_iso2`, `country_id`, `month`, `hour_of_day` |
| `country/country_table_v1.json` | frozen id table (25 countries, sorted; **append-only** — ids never move) |
| `country/MANIFEST.json` | provenance, input/output sha256, counts per split and country |
| `code/build_v10_country.py` | the builder (repo `sayoood/TanitAD`, `stack/scripts/build_v10_country.py`) |

## Provenance
- Source: `nvidia/PhysicalAI-Autonomous-Vehicles` → `metadata/data_collection.parquet` (sha256 `9624750cadabf8028264cc1404e75e81777654be9def06b00e35c9f834b173d5`, 306,152 clips).
- Join: v9 `clip__sha12` = sha256(clip_id)[:12] → UUID via the corpus `index/clip_index_v7.2_{train,eval}.json`.
- Coverage (MEASURED 2026-10-25): **train 4369/4369, eval139 139/139**, 0 unmapped; 25 countries in train, 23 in eval139.
- Granularity: clip-level (upstream gives one country per clip); broadcast to every v9 row of the clip with `sha12`.

## Use rules
- It is a **label**. At inference the country is either **predicted from vision** by the strategic level (primary arm) or **supplied**
  (GNSS geofence; a separate, reported arm). Never feed this table to a model at inference in the predicted arm.
- It is not situation-classifier output and must not enter the goal/route input.
- eval139 has 5 countries with a single clip: report per-country metrics with their `n`.

## Counts
| country | train | eval139 |
|---|---|---|
| Denmark | 278 | 7 |
| France | 274 | 8 |
| Italy | 266 | 7 |
| Sweden | 266 | 11 |
| Germany | 265 | 9 |
| United States | 265 | 13 |
| Spain | 243 | 10 |
| Austria | 240 | 9 |
| Portugal | 230 | 6 |
| Belgium | 228 | 7 |
| Netherlands | 213 | 11 |
| Finland | 208 | 7 |
| Poland | 200 | 7 |
| Czechia | 149 | 1 |
| Luxembourg | 135 | 1 |
| Latvia | 117 | 6 |
| Romania | 114 | 1 |
| Greece | 110 | 5 |
| Hungary | 99 | 2 |
| Estonia | 88 | 4 |
| Croatia | 87 | 3 |
| Lithuania | 83 | 3 |
| Slovakia | 73 | 0 |
| Bulgaria | 70 | 1 |
| Slovenia | 68 | 0 |

## Licence
Derived from NVIDIA PhysicalAI-AV metadata; the upstream licence terms apply. Private dataset of the TanitAD programme.
