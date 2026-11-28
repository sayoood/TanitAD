<title>Frontier scan search log — 2026-09-10</title>

# SEARCH LOG — 2026-09-10 (LAB-RUN-011)

`Every query, its hit count, and every EMPTY named. Charter §3 / §5.4.`
`⚠️ Retrieval environment: the G: mount was in an Errno-22 / OSError outage wave for the first ~40 min of the pass (content reads failed while metadata resolved — the FS9-12 signature). All web work below was done during the outage; repo reads were served from a local retry-cache.`

## Band A — core (scanned + deep)

| # | track | query | hits | outcome |
|---|---|---|---|---|
| Q1 | A1 | `world model autonomous driving latent September 2026 arxiv` | 8 | ⭐⭐⭐ **Drive-HWM `2609.03572`** (2026-09-03) surfaced — hierarchical slow–fast WM. **DEEP (full text).** |
| Q2 | A1 | `arxiv 2609.03572 Drive-HWM hierarchical world model dynamic latent` | 7 | confirmed authorship/date; also surfaced `2604.03208` *Hierarchical Planning with Latent World Models* (unread, proposed) |
| Q3 | A2 | `arxiv September 2026 JEPA joint embedding predictive architecture action conditioned` | 8 | ⭐⭐⭐ **`2601.00844` Value-guided action planning with JEPA world models** (LeCun et al.). **DEEP (full text).** Also re-surfaced ACPC `2608.12939` (already deep-read 09-09) |
| Q4 | A2 | `arxiv abs 2603.19312` (direct, LeWM) | 1 | ⭐⭐ **LeWM DEEP (abstract + HTML full text) — discharges register debt D-9** |
| Q5 | A3 | `arxiv 2026 vision encoder DINOv3 frozen BEV occupancy driving benchmark September` | 8 | ⛔ **EMPTY E1-5 (fifth probe, term varied to `BEV occupancy`)** — all hits are rainfall nowcasting, diffusion policy, Occ3D, or surveys. No DINOv3×BEV-driving benchmark. Consistent with four prior probes |
| Q6 | A4 | `arxiv August September 2026 vision language action model driving latency efficient decoding` | 9 | `2608.30144` *Rethinking Language's Role in Efficient VLA for AVs* (2026-08-31), DeeAD, FASTer, LinkVLA. Scanned, not deep-read |
| Q7 | A5 | `NAVSIM v2 navhard EPDMS leaderboard scenario count split September 2026` | 9 | ⭐⭐⭐ **navhard = 450 Stage-1 / 5462 Stage-2** — the primary count FS9-2 needed |
| Q8 | A5 | `NAVSIM navtest 12000 scenarios count PDMS benchmark split definition` | 10 | ⭐⭐ **navtest ≈ 12,000 samples** — the second population's count. Also the exact PDMS/EPDMS sub-metric formulas |

## Band B — neighbouring disciplines (all scanned; 3 deep-read)

| # | track | query | hits | outcome |
|---|---|---|---|---|
| Q9 | B3 | (via Q1/Q2 full text) Drive-HWM Table III latency decomposition | — | ⭐⭐⭐ **DEEP.** B3's missing transfer path is **temporal abstraction / amortised model calls**, not token serving. First B3 transfer path in four passes |
| Q10 | B5 | `closed-loop simulation post-training sim-to-real gap does not transfer autonomous driving policy degradation evidence` | 8 | ⭐⭐ **DEEP (counter-search for Band D).** Sim2Real-AD `2604.03497`, AD-R1 (CVPR 2026), RAD-2 `2604.15308`, `2607.08072` |
| Q11 | B9 | `arxiv 2026 Kairos 2606.16533 data curation two-level video` | 8 | ⭐⭐⭐ **DEEP (full text, v1 AND v3) — discharges register debt D-8 at the FIFTH route** |
| Q12 | B4 | `arxiv September 2026 efficient training optimizer muP low precision fixed GPU budget world model` | 8 | ⚠️ **EMPTY E-B4 (first probe).** LLMQ, MuonQ, µLO, MONA, BAOC, Megatron-MoE — all language-model training. **No world-model or fixed-small-fleet transfer found.** B4 has now been scanned four passes with no dedicated deep-read |
| Q13 | B7 | `arxiv 2026 flow matching diffusion planner trajectory multimodal futures autonomous driving September` | 6 | scanned. GuideFlow (deep-read 09-05), GoalFlow PDMS 90.3, FlowR2A `2606.24231`, WAM-Flow `2512.06112`, FlowDrive `2509.21961`. No new deep-read; FlowR2A remains the standing candidate for FS9-5 |
| Q14 | B1/B2/B6/B8/B10/B11/B12/B13 | swept via Q1/Q3/Q11 result sets and the Kairos v3 table of contents (hybrid multi-scale temporal memory §2.3; cross-embodiment data curriculum §3.1; tokeniser and VideoDiT sections) | — | scanned, no dedicated deep-read. **B11 (physics-informed) remains without a live primary since `2604.01349` was WITHDRAWN** |

## Band C — non-paper sources

| # | track | query | hits | outcome |
|---|---|---|---|---|
| Q15 | C1 | `NVIDIA Alpamayo blog post lessons autonomous driving reasoning 2026` | 6 | ⭐⭐ **Band-D item found** (see below). Also Alpamayo 2 Super commercial-use release |
| Q16 | C1 | `Wayve technical blog GAIA world model doctrine 2026 end-to-end argument` | 9 | GAIA-3 (2025-12) + GAIA-4 (2026-08-03, already registered 09-05); Wayve+Uber London. No new doctrine post |
| Q17 | C2 | `NVIDIA JetPack TensorRT release notes September 2026 Jetson Thor` | 8 | ⛔ **EMPTY E-C2c (THIRD consecutive probe).** Newest is still **JetPack 7.2.1 / Jetson Linux 39.2.1 / CUDA 13.2.1 / TensorRT 10.16.2** (2026-08-12). ⭐ Absence is now three-probe-confirmed: **there has been no Thor-relevant JetPack release in four weeks.** New signal: TensorRT Edge-LLM (C++ SDK) supports Thor; DOPE ONNX→TRT conversion **fails on AGX Thor on unsupported layers** — a live datapoint for the D-B1-GATE quantisation-gate row |
| Q18 | C4 | `NAVSIM v2 navhard leaderboard` (via Q7) | — | navhard leaderboard anchors extracted from the maintainers' own paper (see A5 finding F2) |
| Q19 | C3 | not probed this pass | 0 | ⛔ **NOT SCANNED — declared.** D-4 (UNECE GRVA) stands at five failed routes and is with the PI; no new route was attempted and none is claimed |

## Band D — opponent doctrine

| # | query | hits | outcome |
|---|---|---|---|
| Q20 | `NVIDIA Alpamayo blog post lessons autonomous driving reasoning 2026` → *How to Post-Train Autonomous Vehicle Models in Closed-Loop with NVIDIA Alpamayo* (Ivanovic & Pavone, 2026-05-31) | 1 primary | ⭐⭐ **ADJUDICATED in full (7 steps) → register rows A16-1 … A16-3** |
| Q21 | **counter-search (step 4, mandatory)** — Q10 above | 8 | ⭐ **CONTRADICTING evidence found at ≥2 independent sources** |

## ⛔ EMPTY SEARCHES NAMED THIS PASS

| id | query | probes | status |
|---|---|---|---|
| **E1-5** | DINOv3 × BEV/occupancy driving benchmark (Q5) | **5th** | ⛔ still empty under a varied term (`BEV occupancy` replacing `dense prediction`). Per FS5-6 the term has now been varied twice; this is approaching a genuine absence rather than a term artefact |
| **E-B4** | efficient-training transfer to a fixed small fleet / world models (Q12) | 1st | ⚠️ first probe only — **not** promotable to absence. A second probe must vary the term (try `data-efficient world model training compute budget`, dropping `optimizer`/`muP`) |
| **E-C2c** | Thor-relevant JetPack/TensorRT release since 2026-08-12 (Q17) | **3rd** | ⛔ three probes, consistent. Absence recorded: no new release in four weeks |
| **E-C3** | UNECE GRVA primary (not probed) | — | ⛔ **declared un-probed, not empty.** Five routes already failed; with the PI as D-4 |

## ⭐⭐ RETRIEVAL-METHOD RECORD — the fifth route, and the near-miss it caused

Kairos `2606.16533` failed at **three routes** on 2026-09-09 (PDF too large for the fetch tool; landing page only; HTML truncated mid-§4.2). Today:

- **Route 4** (`arxiv.org/html/2606.16533v1`) — **FAILED the same way**, truncated at the identical sentence.
- **Route 5** (`urllib` byte download → local PyMuPDF extraction) — **SUCCEEDED.** 42,560,348 B, `%PDF-` header, valid `%%EOF`. The three prior failures were all a **size** limit, not an access block.
- ⛔ **But route 5 pulled `v1`, and `v1` does not contain the claim.** A full-text grep of v1 (90 pp.) returned `control-relevant` ×1 (an unrelated VideoDiT ablation), `two-level` ×3 (all *two-level batching*, an I/O technique), `control-informative` ×0, `regret` ×0 — **while the paper's own title contains "Regret-Aware".**
- The version check (`arxiv.org/abs/…`) showed **v1, v2, v3**. `v3` is **119 pp.** and contains `control-relevant` ×42 and the claim verbatim.

⇒ **Reading v1 would have produced a confident, false `REFUTED` on a true claim.** See RESULT.md F5 and the proposed hygiene rule.
