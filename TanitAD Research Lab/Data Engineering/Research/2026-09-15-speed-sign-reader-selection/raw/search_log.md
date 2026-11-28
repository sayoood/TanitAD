# Search log — 2026-09-15 speed-sign reader selection

V-1 first: the 09-13 package (`2026-09-13-posted-speed-limit-supplier`) and `library.json` were read before any web query. MVV `2508.02047` and TS-1M `2603.23034` were already banked, so they were not re-searched.

| # | query / probe | hits | note |
|---|---|---|---|
| Q1 | `Mapillary Traffic Sign Dataset speed limit value classes regulatory--maximum-speed-limit number of classes license` | 10 | MTSD ECCV/arXiv → banked `1909.04422`. ⚠️ The exact count of maximum-speed-limit value classes was **not found** (search summary and full-text keyword pass); licence wording taken from the full text |
| Q2 | `speed limit estimation from dashcam video without map deep learning dataset arXiv implicit speed limit road context` | 7 | ⛔ **EMPTY for our target**: only vehicle-speed estimation from fixed cameras, plus patents. No published dashcam posted-limit estimator found at probe 1 |
| Q3 | `OpenStreetMap maxspeed tag coverage percentage roads Germany United States completeness study` | 3 sub-searches | Sweden 84 %, a US study 43.18 % (RELAYED). ⛔ No Germany figure. Moot for PhysicalAI (no GNSS, standing) |
| Q4 | `European traffic sign dataset multiple countries merged speed limit classes ETSD 164 classes` | 9 | ETSD: 6 countries, > 80 k images, 164 classes (RELAYED, IEEE abstract). Not banked (not on arXiv) |
| Q5 | `Mapillary Traffic Sign Dataset license CC BY-NC-SA …` | 4 sub-searches | ⛔ **licence string not found by search**; full text says derived data are for *"academic research and approved applications"*. The mapillary.com dataset page fetch returned no content (JS page) |
| Q6 | `arxiv.org/abs/2511.17183` (INTSD) | 1 | banked. Table 4 extracted by pypdf; column alignment inferred from the header order (A.P., A.R., Acc) |
| Q7 | `Hugging Face new autonomous driving dataset release September 2026 open dataset` (shared with Band C4) | 7 | KITScenes Multimodal `2606.02956` → banked. Speed-value encoding **UNVERIFIED** |
| P1 | `data_collection.parquet` country field × v7.2 label ids, **probe 1** (UUID regex on first 400 chars of each JSONL record) | **0 / 4,572** | ⛔ **probe artefact**: the clip id sits beyond char 400. Same-breath control: corpus total 306,152 read fine |
| P2 | same, **probe 2** (UUID regex on the whole record) | **4,572 / 4,572**, 147 / 147 | the result used |
