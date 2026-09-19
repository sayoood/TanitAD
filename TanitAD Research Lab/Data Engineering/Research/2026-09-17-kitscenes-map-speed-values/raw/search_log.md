# search_log — E-DE-KITS-1 + E-DE-SIGN-2 (2026-09-17)

⭐ **V-1 applied:** `2606.02956` (KITScenes) was **already banked** by the 09-15 pass — a re-find, so
only its `cited-by` was updated. No re-download.

## Probes that produced findings

| # | query / URL | tool | hits | what it settled |
|---|---|---|---|---|
| Q1 | `KITScenes multimodal dataset Lanelet2 HD map traffic sign regulatory element speed limit value devkit` | search | 10 | F4 — 62 km², 29 road-feature classes, signs/lights assigned to lanes topologically, Autoware closed-loop validated |
| Q2 | `arxiv.org/html/2606.02956v1` | fetch | 1 | **F1 probe 1** — signs are *"classified based on 220 German road traffic code classes (with 120 observed)"*, **class labels only**; no speed-limit value statement anywhere. Also F3 scale: 5.7 h, 162 km, 1,007 scenarios, 3 cities |
| Q3 | `github.com/KIT-MRT/kitscenes` (devkit README) | fetch | 1 | **F1 probe 2** — *"120 traffic-sign classes (GTSIGN-220 taxonomy)"*, no speed/maxspeed/SpeedLimit API; **code Apache-2.0** |
| Q4 | `huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal` | fetch | 1 | **F1 probe 3** + **F2** — same taxonomy phrasing, no value/sub-code; **licence CC BY-NC 4.0 + additional terms, gated**; **4.58 TB** |
| Q5 | `open dataset posted speed limit sign values map OpenStreetMap maxspeed vision derivable driving corpus 2026` | search | 9 | **F5** — `maxspeed`, `maxspeed:type`, `source:maxspeed`, `osm-legal-default-speeds`; ⛔ unusable for us (no coordinates in our traces) |
| L1 | `cot_tokens.speed_limit` over `s2_labels_v7.2_{train,eval}.jsonl.gz` | local, 0 GPU | — | **F6 / F7 / F8** — 69 / 4,572 train, **0 / 147 eval**; 66 Vienna / 3 US; 33 night / 36 day; 69/0/0 country cross-check vs parquet |

## Controls carried on the local read (⛔ a zero is a claim about the probe until one reads non-zero)

| control | must read | read | |
|---|---|---|---|
| train record count | 4,572 | **4,572** | ✅ |
| eval record count | 147 | **147** | ✅ |
| `cot_tokens.traffic_light` on **eval** (same field, same pass, same file as the 0) | non-zero | **26** | ✅ — this is what makes F6's eval zero a corpus fact rather than a failed read |
| `strata.country` vs `data_collection.parquet` | agreement | **69 agree / 0 disagree / 0 missing** | ✅ independent source |

## Empty searches (named, per charter §5.4)

| # | query | result | scope of the resulting statement |
|---|---|---|---|
| **E-DE1** | a documented **speed-limit VALUE / `maxspeed` / Lanelet2 `SpeedLimit`** field in KITScenes | ⛔ **EMPTY at three independent probes** (paper Q2, devkit README Q3, HF card Q4) | ⚠️ All three are **documentation**, not the map data. The supported statement is *"no value field is documented"*, **not** *"no value field exists"* — Lanelet2 natively supports a `SpeedLimit` element, so it could be present and undocumented. Settling it needs a gated 4.58 TB download, which **F2's CC BY-NC licence makes not worth requesting** |
| **E-DE2** | the **GTSIGN-220 class list itself** (would show whether StVO 274 is sub-coded per value, e.g. 274-50) | not obtained | *"no sub-codes"* rests on three descriptions agreeing, not on an enumeration. Recorded as the one way F1 could still be wrong |
| **E-DE3** | the **31 valued** CoT readings (as opposed to the 69 mentions) | the boolean field cannot answer it | ⇒ `E-DE-SIGN-3` registered to re-derive them from the CoT text |
