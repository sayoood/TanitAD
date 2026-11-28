# search_log — A&I 2026-09-17 (frozen trunk + predictor objective)

⭐ **V-1 applied:** `2601.03460` (FROST-Drive) and `2606.31232` (Delta-JEPA) came back **"already
banked"** from `kb_add.py` — 2 of 5 today, the same re-find fraction the charter's V-1 rule was
written for. Only `cited-by` was updated on those.

| # | query / URL | tool | hits | what it settled |
|---|---|---|---|---|
| Q1 | `arxiv 2026 frozen vision foundation model encoder autonomous driving planning transfer comparison ablation` | search | 10 | ⭐⭐⭐ **discharges the A3 DEEP debt** — FROST-Drive `2601.03460`; also SPS `2601.10707`, Qwen-Drive `2609.00111` (frozen during both pretraining stages) |
| Q2 | `arxiv.org/html/2601.03460` | fetch | 1 | **F1/F2/F3/F6/F7** — Tab. 3 frozen vs finetuned at 14B; encoder sizes 300 M / 6 B; Tab. 4 width ablation 256→5120; open-loop statement; the RFS-alignment concession |
| Q3 | `arxiv September 2026 JEPA joint embedding predictive architecture latent world model driving` | search | 10 | ⭐⭐⭐ **WA-JEPA `2608.20974`**; UniJEPA `2608.07409` (*"plans up to tens of times faster than generative world models"*); Delta-JEPA `2606.31232`; Var-JEPA `2603.20111`; Causal-JEPA `2602.11389`; LeWorldModel `2603.19312` |
| Q4 | `arxiv.org/abs/2608.20974` | fetch | 1 | **F4** — the verbatim *"fundamentally ill-suited"* sentence; method; nuPlan→NAVSIM |
| Q5 | `arxiv.org/html/2608.20974v1` | fetch | 1 | **F5** + the ablation split — navtest caption; Tab. 4(b-c) masking +0.4 vs flow-matching **+1.0**; HUGSIM table |
| Q6 | `arxiv September 2026 vision encoder driving BEV dense prediction DINOv3 benchmark` | search | 10 | ⚠️ see **E-AI1** |

## Empty searches (named, per charter §5.4)

| # | query | result | scope |
|---|---|---|---|
| **E-AI1** | a **September-2026 DINOv3-class encoder benchmark on driving BEV / dense prediction** (Q6) | ⛔ **EMPTY — third probe, term varied each time** (09-13 and 08-31 are the prior two). Returns are medical/dental DINOv3 benchmarks (`2603.28297`, `2509.06467`) plus generic encoders (`2603.22387`, `2605.12491`) | ⭐ **But A3 is NOT empty today** — Q1 found the driving-specific encoder evidence with a *transfer/frozen* framing instead of a *benchmark* framing. ⇒ the standing empty is about **"DINOv3 + BEV benchmark"** specifically, and it should be re-worded on row 39 rather than re-run as-is |
| **E-AI2** | WA-JEPA **parameter counts** | ⛔ not published — *"ViT-L"* backbone and *"MMDiT-style design"* only | ⇒ its +1.0 EPDMS cannot be quoted as a matched-params result |
| **E-AI3** | a WA-JEPA **limitations section** | ⛔ none found | Recorded per charter §7.2 — a paper that concedes nothing is a fact about the paper |
| **E-AI4** | Delta-JEPA `2606.31232` full text | ⚠️ **PDF text extraction FAILED** via the fetch path (binary/corrupt render) | ⛔ **Not read.** It stays `banked, unread` on the A2 debt list — it is NOT cited in any finding above. Its title (*"Learning Action-Sensitive World Models via Latent Difference Decoding"*) suggests it is the nearest external relative of **row 13 / H-RANK-17**, which is exactly why it must be read properly rather than summarised from a title |
