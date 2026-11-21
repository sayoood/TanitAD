# search log — Frontier Scan 2026-09-13 (every query, hit count, every EMPTY named)

Engine: WebSearch (US) unless stated. "hits" = result links returned. Package-specific logs carry their own probes:
`Data Engineering/Research/2026-09-13-posted-speed-limit-supplier/raw/search_log.md` (S1–S16),
`Benchmarks & Evals/Research/2026-09-13-pdmc-stage2-provenance/raw/search_log.md` (1–12),
`Architecture & Inference/Research/2026-09-13-hwm-capacity-matched-hierarchy/raw/search_log.md` (1–4),
`Deployment & Optimization/Research/2026-09-13-i1-value-geometry-screen/raw/search_log.md` (1–5).

## Dedup gate
| probe | result |
|---|---|
| `ls "TanitAD Research Lab"/*/Research/2026-09-13-*` | 3 dirs (A&I `bev-lidar-corpus-and-head`, DE `qwen-drive-usage-review`, DE `sam3-only-road-map`) — all authored **Master Mind / A&I FlyWheel**, not a Lab pass; 2 domains only |
| `ls "Frontier Scan/Daily/"` (control: lists 2026-08-31 … 2026-09-10) | **no 2026-09-13** ⇒ gate (b) failed ⇒ full pass run |
| `LAB_ASKS.md` OPEN rows | 0 (ASK-1 ANSWERED); file 2,004 B, read cleanly |

## Band D
| # | query | hits | note |
|---|---|---|---|
| D1 | `Waymo blog September 2026` | 9 | compute post 08-20, riders Denver/SD/Tampa 09-01, TechCrunch 09-01 |
| D2 | `Wayve OR Tesla OR Mobileye autonomous driving lessons technical blog September 2026` | 8 | **EMPTY for a new Wayve/Mobileye doctrine post** (Mobileye robotaxi plan 06-16 is a release, not doctrine) |
| D3 | register grep `under our trunk / pure end-to-end / cybercab / mix of sensors` | 1 (W-4) | 10-lessons post already adjudicated; compute post **not** |
| D4 | WebFetch TechCrunch 2026-09-01 | 1 | Thirumalai quotes; Ferragu "incumbent rhetoric" |
| D5 | WebFetch waymo.com/blog/2026/08/look-under-our-trunk/ | 1 | claims + concession ("low-batch regimes") |
| D6 | `Waymo "under our trunk" compute 1000 TOPS analysis` | 9 | relays only (not independent) |
| D7 | `Tesla Cybercab event September 2026 FSD end-to-end neural network claims` | 9 | 380k unsupervised miles, "vision-plus-radar" (RELAYED) |
| D8 | `Tesla robotaxi NHTSA crash reports Austin 2026 standing general order incidents` | 10 | contradicting + confirming (RELAYED) |
| D9 | `tail latency p99 autonomous driving perception pipeline safety analysis paper` | 9 | COLA `2305.07147` (banked), `2209.05487` |
| D10 | `"TOPS" misleading metric AI accelerator automotive benchmark critique` | 7 | contradicting evidence for A17-1 |
| D11 | `autonomous vehicle redundant compute fail-operational ISO 26262 dual SoC architecture study` | 9 | A17-3 confirm + asymmetric-failover alternative |
| D12 | counter-search for A17-2 ("tail latency irrelevant / average latency sufficient") | via D9 set | **EMPTY** — no source arguing against percentile latency |

## Band A
| # | track | query | hits | note |
|---|---|---|---|---|
| A1 | A1 | `arXiv 2609 world model autonomous driving` | 9 | SV-WAM `2609.03602`, Drive-HWM `2609.03572` (read 09-10) |
| A2 | A2/B11 | `arXiv September 2026 JEPA action-conditioned predictive latent` | 9 | **SG-JEPA `2609.10464`** (banked, read), JEPA-x (banked 09-03), safety shields `2608.17496` (scan) |
| A3 | A3 | `DINOv3 OR vision encoder September 2026 arXiv new visual representation release` | 7 | **EMPTY for a Sept-2026 encoder release** (probe 1 of this term; E1 DINOv3×BEV not re-probed) |
| A4 | A4 | `arXiv 2609 vision-language-action autonomous driving model efficient` | 9 | BLUE `2606.08684` (banked), `2608.30144` (banked, read), DeeAD |
| A5 | A5/C4 | `NAVSIM leaderboard OR Bench2Drive leaderboard new state of the art September 2026` | 10 | Bench2Drive LB updated 2026-08-28; RAP-DINO 36.9; HiDrive `2605.09972` (scan) |
| A6 | A1 | V-1 on `2604.03208` | local | already banked 08-27 |

## Band B (scan all 13; DEEP = B1, B11, B13)
| # | track | query | hits | note |
|---|---|---|---|---|
| B1 | B1 | (DE package S2, S11) | 9 / 7 | MVV `2508.02047` full text; TS-1M `2603.23034` |
| B2 | B2 | `arXiv 2609 state space model OR linear attention OR hybrid architecture world model video prediction` | 8 | **EMPTY for a Sept-2026 item**; `2505.20171` long-context SSM video WM (old), `2605.16579` linear attention as cross-frame memory (scan) |
| B3 | B3 | `arXiv 2609 speculative decoding OR KV cache compression robotics policy real-time inference` | 9 | BeaconKV `2609.04971` (LLM-only); `2606.13355` real-time autoregressive policies (scan) |
| B4 | B4 | `arXiv 2609 efficient training vision transformer video small compute data-efficient pretraining recipe` | 8 | **EMPTY E-B4, SECOND probe with a varied term** (09-10 probe 1 was LLM-training terms). Nearest: `2605.19137` data-efficient video pretraining on **frozen image foundation models** (scan — relevant to the freeze) |
| B5 | B5 | `arXiv 2609 reinforcement learning post-training world model driving policy GRPO` | 10 | `2609.03225` WM-RL + GRPO multi-style driving (scan) |
| B6 | B6 | `arXiv 2609 self-improving self-play curriculum generation embodied agent world model` | 7 | **EMPTY for Sept 2026**; `2606.30639` self-evolving WMs for LLM agents, `2607.13104` survey (scan) |
| B7 | B7 | `arXiv 2609 flow matching trajectory planning autonomous driving diffusion planner` | 8 | `2609.04921` one diffusion model as planner + scenario generator (scan) |
| B8 | B8 | `arXiv 2609 video tokenizer OR FSQ discrete latent world model 2026` | 9 | **EMPTY for Sept 2026**; `2603.05438` 8-token tokenizer, `2604.04913` delta tokens (scan) |
| B9 | B9 | `arXiv 2609 driving data curation scenario mining long-tail selection dataset` | 10 | ScenarioCharacterization `2608.16041`, OpenLongTail `2607.09655` (scan) |
| B10 | B10 | `arXiv 2609 retrieval-augmented driving memory embedding scenario retrieval planning` | 7 | **EMPTY for Sept 2026**; VLADriver-RAG `2605.08133` (scan) |
| B11 | B11 | (A2 query) | — | SG-JEPA — DEEP |
| B12 | B12 | `arXiv 2609 long-horizon memory world model persistent state recurrent latent` | 9 | 2AM `2609.11308` (manipulation memory, scan); MemoryWAM `2606.20562` |
| B13 | B13 | `OccFeat self-supervised occupancy feature prediction pretraining BEV arXiv` + `arXiv 2609 camera-only 3D occupancy self-supervised BEV perception driving` | 9 / 9 | OccFeat `2404.14027` DEEP; **EMPTY for a Sept-2026 occupancy item** |

## Band C
| # | track | query / route | hits | note |
|---|---|---|---|---|
| C1 | C1 | D1, D7 + HF cards NuRec / NCore (DE S9b, S12) | — | releases table in RESULT F8 |
| C2 | C2 | `JetPack 7.3 OR TensorRT 11 release notes Jetson Thor September 2026` | 9 | **EMPTY E-C2c, FOURTH probe** (still JetPack 7.2.1 / TRT 10.16.2); TRT 11.2.1 excludes JetPack |
| C3 | C3 | `NHTSA AV framework rulemaking September 2026 automated driving systems exemption` | 10 | comment period → 2026-09-30 |
| C3b | C3 | `UNECE WP.29 GRVA ADS global regulation adopted 2026 session automated driving` | 10 | WP.29 adoption June 2026 |
| C3c | C3 / D-4 | WebFetch unece.org document page (UN R ADS, reissued 8 June 2026) | — | **HTTP 403** (D-4 route 6) |
| C3d | C3 / D-4 | WebFetch globalpolicywatch.com state-of-play | — | **HTTP 403** |
| C3e | C3 / D-4 | WebFetch unece.org press release | — | **HTTP 403** (route 7) |
| C3f | C3 / D-4 | `curl --ssl-no-revoke` `unece.org/sites/default/files/2026-01/ECE-TRANS-WP.29-GRVA-2026-02e.pdf` | — | **HTTP 403, 5,582 B HTML** (route 8) |
| C3g | C3 | curl `indico.un.org/.../English.pdf` | 200, 144,486 B | a **2024 GRVA-19 provisional agenda** — not the primary; not cited |
| C3h | C3 | `UN Regulation automated driving systems June 2026 adopted "machine learning" OR "software update" OR "in-service" provisions analysis` | 8 | SMS / ISMR / DSSAD / R156 (RELAYED ×2) |
| C4 | C4 | A5 query + v3 Table 2 (B&E package) | — | navhard 03/2026 snapshot |

## Library / banking (kb_add, all rc 0)
New: `2609.10464`, `2508.02047`, `2404.14027`, `2603.23034`, `2305.07147`, `2403.04133`. Citations/notes updated: `2604.03208`, `2608.30144`, `2506.04218`, `2606.07170`.
⚠️ **The six new entries were banked with EMPTY `title`/`published` metadata** (kb_add printed `— ` with no title); content verified independently by `%PDF-` magic bytes and by local full-text extraction for four of them (version strings `2609.10464v1 9 Sep 2026`, `2508.02047v1 4 Aug 2025`, `2404.14027v3 12 Jun 2024`, `2506.04218v3 20 Mar 2026`). Library: **493 entries, 3,262.1 MB**.
