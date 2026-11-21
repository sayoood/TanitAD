# Frontier Scan search log — 2026-09-19 (LAB-RUN-016)

`Every query, its hit count, and every EMPTY named. Tool: WebSearch (US index) + WebFetch on primaries + local pypdf on banked PDFs; Library probed FIRST (V-1). Retrieval date for all web items: 2026-09-19.`

## Library-first (V-1)

| # | probe | result |
|---|---|---|
| L1 | `2605.10564` DeepSight (pre-committed rotation #1) | **held (banked 09-18, unread)** → **full text extracted locally with pypdf, 14 pp, 56,075 chars**. Debt discharged |
| L2 | `2609.06055` DriveZero · `2602.11229` LGS (pre-committed rotation #3) | PDFs **held on disk** → full text extracted locally (32 pp / 28 pp) and every number quoted on 09-18 re-checked. ⛔ **One 09-18 figure does NOT reproduce (LGS "0/16 → 16/16")** — see RESULT §0 row 6 |
| L3 | `2606.14010` RT-VLA | **re-find**: already banked (tags/citations updated) |
| L4 | `library.json` itself | ⛔ **indexed 489 entries (= HEAD) while 26 PDFs from the 09-17/09-18 passes sat on disk UNINDEXED** (incl. DriveZero, LGS, `2607.07196`, `1902.03393` — every 09-18 bank but two). `--reindex-orphans` recovered all 26 → **521 entries, 0 orphans**. `--verify` then reports **49 MISSING**: entries indexed but the PDF is neither on D: nor tracked in git (spot check: present on the pre-move G: tree). Escalated, not touched (G: path rule) |

## SCAN — all 22 tracks + Band D

| track | query | hits | triage / EMPTY |
|---|---|---|---|
| **D** Mobileye | "Mobileye blog September 2026 autonomous driving compound AI system lessons" | 9 | no Sept-2026 post (newest 2026-08-11). ⭐ **"Autonomous decisions: the bias-variance tradeoff" (Shashua & Shalev-Shwartz, 2024-05-15) NEVER adjudicated** (register grep `bias-variance` = 0, control `Tesla` = 16) → **M-19-1/2** |
| **D** NVIDIA | "NVIDIA Alpamayo September 2026 blog reasoning VLA …" | 9 | Alpamayo 2 Super (34 B, 2026-05) = release, not doctrine; no Sept item |
| **D** Waabi/Zoox | "Waabi OR Zoox technical blog 2026 world model simulation safety lessons" | 9 | ⭐ **Waabi "Simulator realism: the new safety standard" (Urtasun, 2025-03-11), never adjudicated** (register `Waabi` = 0) → **WB-19-1/2**. Zoox edge-case journal (2026-06) — practice post, no doctrine claim |
| **D** Waymo | "Waymo blog September 2026 research driving model" | 10 | Singapore (09-17), Allianz (09-15) = releases. ⚠️ **Reference Driver (ReD), 2026-06 blog + Nature Comms — never adjudicated** (register `Reference Driver` = 0) → new debt **D-14** |
| **D** CN majors | "Momenta OR DeepRoute OR Horizon Robotics 2026 end-to-end world model VLA doctrine keynote" | 9 | ⛔ **EMPTY for CN-major doctrine** (market reports only). ⭐ Surfaced **Jim Fan (NVIDIA) "VLAs are dead, long live World Action Models"** → **N-19-1/2/3** |
| **D** Waymo | (TechCrunch 2026-09-01 "Waymo goes on offense") | 1 | primary = the *10 AI Lessons* post + an Axios interview ⇒ **V-1 re-find in the REGISTER** (W-lessons, adjudicated 08-31); no new claim |
| **D** counter (N-19) | "video generation model physical realism does not correlate with physical understanding Physics-IQ" | 9 | ⭐ `2501.09038` Physics-IQ (**contradicting**); `2606.18943` Physics-IQ Verified (audit; Kendall τ 0.46 re-ranking; silent on scale) |
| **D** counter/confirm (N-19) | "video world model pretraining improves robot policy success ablation vs VLM backbone 2026 …" | 9 | ⭐ **NVIDIA Tech Blog "Pretrained to Imagine, Fine-Tuned to Act" (2026-06-15)** — the doctrine's own channel concedes *no matched comparison*; ⭐ **World Tokens `2608.09730`** (matched action-only baseline) |
| **D** counter (WB-19) | "sim-to-real gap closed-loop autonomous driving simulator realism metric trajectory similarity validation …" | 9 | `2509.22379` multi-modality reality-gap evaluation; MCRPG (Sensors 26/1338) — both show the gap is **per-facet**, not one scalar |
| **D-4** | UNECE `ECE-TRANS-WP.29-GRVA-2026-02e.pdf` · regulations.gov `NHTSA-2026-0034-0058` · federalregister.gov 2026-01274 · transportation.gov 2026-01274 · globalpolicywatch 2026/05 | 5 routes | ⛔ **ALL 403 / bot-challenge** (curl `--ssl-no-revoke` + browser UA on the UNECE PDF also 403). **Routes 9–13 failed; not bypassed.** D-4 STANDS |
| **D-13** | "Ashok Elluswamy ICCV 2025 workshop talk transcript "world simulator" …" | 10 | ⛔ **EMPTY for a transcript** (coverage only). The speaker's own X post exists (`x.com/aelluswamy/status/1981644831790379245`) — unfetchable here. D-13 STANDS |
| A1 | "latent world model driving arXiv 2609 planning imagination" | 9 | ⭐ `2609.15781` backdoored latent WM checkpoints (**banked**); `2609.03294` LEAP (→A2); PLAN-S `2606.06014`; taxonomy `2603.09086` |
| A2 | "V-JEPA 2 action-conditioned planning new results September 2026 …" | 9 | nothing newer than V-JEPA 2-AC; value-guided JEPA planning `2601.00844`. ⭐ Served by **LEAP `2609.03294`** (frozen LeWM, gradient planner) |
| A3 | "frozen vision foundation model end-to-end driving planner ablation DINOv3 SigLIP encoder comparison 2026" (the 09-18 pre-committed phrasing) | 9 | ⛔ **E-A3, 4th consecutive driving-term probe without a NEW dedicated benchmark.** Hits = DriveZero (held), FROST-Drive (held), "The Constant Eye" `2602.12563` (appearance robustness, SCAN). ⭐ **DeepSight Table 3 is an A3 result** (DINOv3 target vs VAE codebook, +47.04 DS) |
| A4 | "vision-language-action autonomous driving latency real-time distillation small model 2609 arXiv" | 9 | RT-VLA `2606.14010` (re-find); LatentVLA; XYZ-Drive. ⭐ **A4 DEEP = World Tokens** |
| A5 | "NAVSIM navhard OR Bench2Drive new state of the art September 2026 closed-loop benchmark" | 9 | ⛔ **EMPTY for a new navhard/Bench2Drive SOTA** after DriveZero-Scale. ⭐ DeepSight **86.23 DS / 71.36 % SR** on Bench2Drive base (Think2Drive expert) recorded for the stamp table |
| B1 | "vision language model release September 2026 open weights spatial reasoning Qwen OR Gemma OR InternVL" | 9 | SenseNova-U1.5 (Sept 2026, encoder-free unified) — RELAYED, SCAN only |
| B2 | "Mamba OR linear attention OR gated delta net world model video prediction 2026 arXiv efficient long rollout" | 9 | Gated DeltaNet-2 `2605.22791`; TetherCache `2606.13035` (long AR video, gated recall). ⛔ **E-B2 (3rd): no robotics/driving predictor transfer primary** |
| B3 | "speculative decoding OR KV cache compression world model rollout inference speedup 2026 arXiv" | 9 | VeriCache `2605.17613` (lossless via draft-on-compressed-KV); all LLM text. ⛔ **EMPTY for WM-rollout transfer** |
| B4 | "Muon optimizer vision transformer world model training efficiency 2026 results ablation" | 9 | Muon-in-ViT `2605.24770` (recipe-dependent), CMuon `2608.02502` — same as 09-18; SCAN only |
| B5 | "GRPO reinforcement fine-tuning world model planner driving closed-loop September 2026 arXiv" | 9 | WorldRFT `2512.19133` (AAAI-26; collision −83 % nuScenes), RAD-2 `2604.15308`, DAWN `2605.11550` — SCAN only |
| B6 | "self-improving world model self-play generated curriculum embodied agent September 2026 arXiv" | 9 | `2606.15386` (MDL pressure makes self-play *self-improving* not *self-confirming*); `2606.30639`; WAV `2604.01985` (held) — SCAN only |
| B7 | "flow matching trajectory planner driving one-step mean flow 2026 arXiv multimodal" | 9 | GuideFlow (CVPR 2026, constraint-guided FM + EBM), MeanFuser `2602.20060`, FlowDrive — SCAN only |
| B8 | "video tokenizer latent space world model FSQ semantic tokens driving 2026 arXiv new" | 9 | ⭐ **OneWM-VLA `2605.07931` DEEP**; Delta tokens `2604.04913` (held); token streaming `2605.09886` |
| B9 | "driving dataset curation long-tail scenario selection data pruning end-to-end planner 2026 arXiv" | 9 | trajectory-entropy pruning `2512.19270`; nuReasoning `2605.31572`; WOD-E2E (CVPR 2026) — SCAN only |
| B10 | "retrieval augmented planning memory bank driving scenes embedding test-time 2026 arXiv" | 9 | ⭐ **DriveVLA-M0 DEEP**; `2609.08217` hierarchical memory + tool-grounded reasoning (new, SCAN); PersonaDrive `2606.12616` |
| B11 | "physics-informed neural operator dynamics prior learned simulator long rollout stability 2026 arXiv" | 9 | DiffusionRollout `2602.13616` (uncertainty-aware adaptive rollout), PI-Laplace NO `2602.12706` (causal time-decaying residual weight). ⭐ **B11 DEEP = LGS local full-text re-read (correction)** |
| B12 | "long-horizon world model memory 30 seconds driving streaming state recurrent 2026 arXiv minute-long" | 9 | HorizonDrive `2605.11596` (~1-min driving rollouts, self-corrective), ReWorld `2608.23565`, Matrix-Game 3.0, DecMem. ⭐ **B12 DEEP = DeepSight** |
| B13 | "camera-only 3D occupancy forecasting OR Gaussian splatting BEV self-supervised September 2026 arXiv" | 9 | GEM `2605.17682` (Gaussian evolution for occupancy forecasting + planning), UnsOcc `2606.03581`. ⛔ **EMPTY for Sept-2026** |
| C1 | "Wayve OR Waymo OR Tesla robotaxi announcement September 2026" | 9 | Tesla Cybercab paid service Austin (09-03); Waymo Denver/San Diego/Tampa (14 US cities); Uber×Wayve London (09-03) |
| C2 | "JetPack 7 Jetson Thor release notes September 2026 TensorRT" | 9 | ⛔ **EMPTY for a post-7.2.1 JetPack**; 7.2.1 = TRT 10.16.2 (unchanged since 09-17). MIG tech preview on T5000 (7.2) |
| C3 | "UNECE GRVA September 2026 ADS regulation adopted OR NHTSA AV framework September 2026" | 10 | GRVA adopted draft UNR + GTR on ADS (Jan 2026), WP.29 vote expected June 2026 (RELAYED). ⭐ **NHTSA Audit Query AQ26002 into Cybercab FMVSS self-certification (2026-09-03, RELAYED)** |
| C4 | (via A5) | — | no new leaderboard entry above DriveZero-Scale |
| **MM-Q** | "posted speed limit estimation from dashcam images traffic sign dataset map speed limit label open dataset" | 9 | TT100K (221 sign classes incl. speed limits), Mapillary TSD `1909.04422` (**held on disk, orphan recovered today**) — vision side only |
| **MM-Q** | "nuPlan map lane speed_limit_mps attribute devkit" → **code probe** `motional/nuplan-devkit …/nuplan_map/lane.py` | 1 file | ⭐⭐ **FOUND: `speed_limit_mps` is a per-lane property**, `Optional[float]`; the getter itself guards NaN (`speed_limit == speed_limit`) ⇒ **coverage is partial, fraction unmeasured** |
| **MM-Q** | "Waymo Open Motion Dataset map features lane speed_limit_mph proto" → **code probe** `waymo-open-dataset …/protos/map.proto` | 1 file | ⭐⭐ **FOUND: `LaneCenter.speed_limit_mph` (`optional double`, field 1)** — "The speed limit for this lane." Coverage unmeasured |

## EMPTIES named (11)

E-D-CN (Chinese-major doctrine) · E-D4 (UNECE primary, routes 9–13, all 403) · E-D13 (Elluswamy transcript, 2nd probe) · E-A3 (4th driving-term probe, no new dedicated benchmark) · E-A5 (no navhard/B2D SOTA above DriveZero-Scale) · E-B2 (post-transformer predictor transfer, 3rd) · E-B3 (WM-rollout decoding transfer) · E-B13 (Sept-2026 occupancy, 2nd) · E-C2 (post-7.2.1 JetPack) · E-WB-N (Waabi publishes no scenario count / module breakdown for 99.7 %) · E-OneWM-EB (OneWM-VLA reports no error bars).

⭐ **The MM standing question produced its first POSITIVE result in 9 days of probes** — and it came from reading **schemas in code** (the tool that owns the fact, rule 2), not from searching prose, which had returned only OSM `maxspeed` for a week.
