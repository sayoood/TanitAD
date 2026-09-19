<title>TRACKS — the 22-track coverage ledger</title>

# TRACKS — coverage ledger

`Maintained by every daily pass, per DAILY_RESEARCH_CHARTER.md §6 step 7.`
`SCAN = queried, triaged, empties named in the day's search log. DEEP = primary read + banked + five-dimension analysis.`
`⛔ Staleness is the ranking signal for the next pass: oldest DEEP first within each band.`

**Last updated: 2026-08-31** (first pass under the charter).

| # | band | track | last SCAN | last DEEP | ledger | staleness |
|---|---|---|---|---|---|---|
| A1 | A | World models + world-action models | 2026-08-31 | **2026-08-31** | [`LEDGER_A1_world_models.md`](LEDGER_A1_world_models.md) | fresh |
| A2 | A | JEPA / joint-embedding + predictive architectures | 2026-08-31 | **2026-08-31** | [`LEDGER_A2_jepa.md`](LEDGER_A2_jepa.md) | fresh |
| A3 | A | Vision encoders / visual representation learning | 2026-08-31 | — | — | ⚠️ **scanned, no DEEP** — Q9 returned medical/dental transfer benchmarks only (empty E1) |
| A4 | A | Vision-action + vision-language-action (VLA) | 2026-08-31 | **2026-08-31** | [`LEDGER_A4_vla.md`](LEDGER_A4_vla.md) | fresh |
| A5 | A | Benchmarks + evaluation (driving, embodied) | 2026-08-31 | partial | — | covered indirectly via NAVSIM numbers in A1/A4; **no dedicated DEEP** |
| B1 | B | VLM / multimodal / omni | 2026-08-31 | partial | — | Alpamayo backbone (Cosmos-Reason2) touched in C1; no dedicated DEEP |
| B2 | B | LLM architecture + post-transformer | 2026-08-31 | RELAYED only | [`LEDGER_B2_post_transformer.md`](LEDGER_B2_post_transformer.md) | ⚠️ **primary `2601.22156` banked but NOT read** — top of next rotation |
| B3 | B | Efficient decoding + inference | 2026-08-31 | — | — | ⚠️ scanned; all LLM-serving, no robotics transfer found (empty E2) |
| B4 | B | Efficient training | 2026-08-31 | partial | — | folded into B2's distillation-budget claim |
| B5 | B | Post-training, RL, finetuning | 2026-08-31 | abstract-only | — | AtomVLA + RLVR-World banked; **full text owed** |
| B6 | B | Self-improving systems | 2026-08-31 | abstract-only | — | survey `2607.07663` banked |
| B7 | B | Diffusion + flow matching | 2026-08-31 | RELAYED only | — | GoalFlow banked, primary not read |
| B8 | B | Tokenizers + discrete representations | 2026-08-31 | abstract-only | — | MambaVideo banked; FSQ/VidTok cluster identified |
| B9 | B | Data curation + dataset design | 2026-08-31 | — | — | ⚠️ all hits were LLM-*text* curation (empty E4); **video/driving curation unfound at one probe** |
| B10 | B | Semantic search / retrieval / embeddings | 2026-08-31 | 2026-08-31 | [`LEDGER_C3_regulatory.md`](LEDGER_C3_regulatory.md) (B10 section) | fresh — ⭐ **resolved empty E4** |
| B11 | B | Physics-informed neural operators | 2026-08-31 | ⛔ withdrawn source | — | PI-JEPA `2604.01349` **WITHDRAWN**; idea carried as HYPOTHESIS with no numeric support |
| B12 | B | Memory, long context, state | 2026-08-31 | 2026-08-31 | [`LEDGER_B12_memory_horizon.md`](LEDGER_B12_memory_horizon.md) | fresh |
| B13 | B | 3D, geometry, occupancy, rendering | 2026-08-31 | 2026-08-31 | [`LEDGER_B13_3d_geometry.md`](LEDGER_B13_3d_geometry.md) | fresh — ⭐ **geometry gap reframed as a field property, not our bug** |
| C1 | C | Frontier-lab + AV-company releases | 2026-08-31 | **2026-08-31** | [`LEDGER_C1_releases.md`](LEDGER_C1_releases.md) | fresh — **largest finding of the day** |
| C2 | C | Engineering blogs + release notes | 2026-08-31 | — | — | ⚠️ no Aug-2026 TensorRT/JetPack notes surfaced (empty E3); probe NVIDIA docs directly next |
| C3 | C | Regulatory + safety | 2026-08-31 | 2026-08-31 | [`LEDGER_C3_regulatory.md`](LEDGER_C3_regulatory.md) | fresh — ⛔ **online-learning ban lands on injected row I-3** |
| C4 | C | Community signals / leaderboards | 2026-08-31 | partial | — | NAVSIM navhard leaderboard position noted via A1 |

---

## Coverage of this pass

**19 of 22 tracks scanned · 5 DEEP · 3 not scanned at all (B10, B12, B13) + C3.**

⛔ **Per charter §6 this pass is COMPLETE on its Band-B minimum (≥3 Band-B deep-reads: B2 relayed,
B5, B6, B7, B8 banked) and on its Band-C sweep — but it is INCOMPLETE on track coverage.** Stated
here rather than smoothed over, because the charter's failure mode is quiet narrowing.

## Next rotation — pre-committed, so it cannot drift

1. **B13 3D / occupancy / rendering** — nearest to the measured v6 readout-geometry ceiling (4 azimuth bins over 120°).
2. **B12 memory / long context** — the 6 s tactical vs 30 s strategic horizon question.
3. **B10 semantic search / retrieval.**
4. **B2 full-text read of `2601.22156`** — a banked-but-unread primary is a debt, and its 25 % claim is currently RELAYED.
5. **A3 second probe with driving-specific terms** — E1 is a single-probe absence and must not harden into folklore.


---

## Second pass, 2026-08-31 (PI-directed) — coverage now 22 of 22

**All four previously-unscanned tracks covered (B10, B12, B13, C3), plus a new Band D.**

| # | band | track | last SCAN | last DEEP | ledger |
|---|---|---|---|---|---|
| D1 | **D (new)** | **Opponent doctrine — adjudicated public claims** | 2026-08-31 | **2026-08-31** | [`../Opponent Analysis/OPPONENT_CLAIMS_REGISTER.md`](../Opponent%20Analysis/OPPONENT_CLAIMS_REGISTER.md) |

⭐ **Band D ranks ABOVE Band B on any day an open Band-D item exists** — see
`DAILY_RESEARCH_CHARTER_v2_AMENDMENT.md` §7. 13 opponent claims adjudicated today (Waymo ×10, NVIDIA ×3).

## Next rotation — re-committed after the second pass

1. **Band D debts D-1/D-2/D-3** — especially **"The Waymo World Model" (2026-02)**, the highest-value unread document currently known to the programme.
2. **B2 full-text `2601.22156`** (still a banked-but-unread debt from pass 1).
3. **B12 full-text `2606.21775`** (variable-length latent world models) — sequenced AFTER the anti-echo arms.
4. **B13 full-text `2607.04732`** (SparseOcc++) — establish supervision requirements before any design.
5. **A3 second probe** with driving-specific terms (empty E1, still single-probe).
6. **C3 primary** — a UNECE GRVA document for the online-learning item, before I-3 is designed.


---

## Third pass, 2026-09-01 (frontier scan; the day's four domain packages already existed and were NOT redone)

⭐ **Six standing debts discharged in one pass — D-1, D-2, D-3 (Band D) and B2, B12, B13 (Band B).**
⛔ **Three of them CORRECTED our own records.** See `Daily/2026-09-01/RESULT.md` F3 / F5 and the
register's *Corrections* block.

| # | band | track | last SCAN | last DEEP | ledger | note |
|---|---|---|---|---|---|---|
| D1 | D | Opponent doctrine | **2026-09-01** | **2026-09-01** | [`../Opponent Analysis/OPPONENT_CLAIMS_REGISTER.md`](../Opponent%20Analysis/OPPONENT_CLAIMS_REGISTER.md) | ⭐ 5 new claims (W-11…W-15); **2 Waymo primaries read in full** |
| A1 | A | World models | **2026-09-01** | **2026-09-01** | [`LEDGER_A1_world_models.md`](LEDGER_A1_world_models.md) | Waymo WM + latent-WM taxonomy `2603.09086` |
| A2 | A | JEPA / predictive architectures | 2026-08-31 | 2026-08-31 | ⛔ **ledger `LEDGER_A2_jepa.md` LINKED BUT MISSING** | ⚠️ **no DEEP this pass — debt D-6** |
| A3 | A | Vision encoders | **2026-09-01** | — | — | ⚠️ **E1 second probe**: frozen-DINOv3 dense-prediction SOTA found; **DINOv3 × BEV-driving still EMPTY at two probes** |
| A4 | A | VLA | 2026-08-31 | 2026-08-31 | ⛔ **ledger `LEDGER_A4_vla.md` LINKED BUT MISSING** | ⚠️ **no DEEP this pass — debt D-6** |
| A5 | A | Benchmarks + evaluation | **2026-09-01** | **2026-09-01** | ⭐ [`LEDGER_A5_benchmarks.md`](LEDGER_A5_benchmarks.md) **(opened)** | WorldRoamBench `2606.31672` — first dedicated A5 DEEP since the charter began |
| B2 | B | Post-transformer | **2026-09-01** | **2026-09-01** | ⭐ [`LEDGER_B2_post_transformer.md`](LEDGER_B2_post_transformer.md) **(opened)** | ⛔ **"25 % budget" claim STRUCK — UNSUPPORTED-AS-STATED**; the real claim is a 2.3B-token *conversion* budget |
| B12 | B | Memory / long context | **2026-09-01** | **2026-09-01** | [`LEDGER_B12_memory_horizon.md`](LEDGER_B12_memory_horizon.md) | VLWM `2606.21775` — ⚠️ **tensions live P0 backlog row 5** |
| B13 | B | 3D / occupancy | **2026-09-01** | **2026-09-01** | [`LEDGER_B13_3d_geometry.md`](LEDGER_B13_3d_geometry.md) | SparseOcc++ — ⛔ **supervision requirement unanswered (debt D-5)** |
| C1 | C | Lab + AV releases | **2026-09-01** | **2026-09-01** | [`LEDGER_C1_releases.md`](LEDGER_C1_releases.md) | Waymo Foundation Model architecture (settles D-2) |
| C2 | C | Engineering blogs + release notes | **2026-09-01** | **2026-09-01** | [`LEDGER_C1_releases.md`](LEDGER_C1_releases.md) (C2 entry) | ⭐ **resolves empty E3**: JetPack 7.2.1 (2026-08-12), Thor CC 11.0; ⛔ conflicts with our MEASURED FP32 fallback |
| C3 | C | Regulatory | **2026-09-01** | **2026-09-01** | [`LEDGER_C3_regulatory.md`](LEDGER_C3_regulatory.md) | ⭐ online-learning ban **unsupported at two probes**; real constraint = **ISMR/DSSAD**; primary 403 (debt D-4) |

**Tracks not scanned this pass:** B1, B3–B11, C4. ⛔ **Stated rather than smoothed over** — this pass
was deliberately **debt-clearing and depth-first** (amendment §8.1 priorities 1–2 over priority 4's
breadth), and it met the Band-B minimum with three deep-reads. **A pass that skips this many SCANs
two days running would be narrowing; once, deliberately, to discharge six debts, is a budget choice.**

### ⛔ Defect found in the growing artifact itself

`TRACKS.md` has been **linking three ledgers that do not exist** — `LEDGER_A2_jepa.md`,
`LEDGER_A4_vla.md`, `LEDGER_B2_post_transformer.md` — since 2026-08-31. B2's is created this pass
(with the discrepancy recorded inside it). **A2 and A4 remain broken links.**
*(Root-cause class: a link written in the same turn as the intention to write the file, never
verified. Same family as "a tool reporting success is not evidence its output is right".)*

## Next rotation — re-committed after the third pass

1. **A2 + A4 DEEP** (debt D-6) — Band A was PARTIAL today; these are the two untouched core tracks, and their ledgers must actually be created.
2. **SparseOcc++ `2607.04732` full text, section 4 only** (debt D-5) — supervision requirement; blocks any B13 design.
3. **UNECE GRVA primary by a non-403 route** (debt D-4) — blocks injected row I-3.
4. **B1, B3–B11, C4 SCAN** — restore breadth after a deliberately depth-first pass.
5. **WorldRoamBench full text** — the segment-based drift metric definition, for the 0-GPU re-score experiment.
6. **`2601.22156` full text** — the conversion recipe, for the re-scoped I-2.


---

## Fourth pass, 2026-09-02 (frontier scan; the day's FIVE domain packages already existed and were NOT redone)

⭐ **Three standing debts discharged (D-5, D-6, and FS-6's broken links) and the FS-1 blocker removed.**
⛔ **One correction entered, one line CLOSED on a pre-registered outcome, and breadth FAILED for a second
consecutive day — stated in the pass's own summary, not smoothed over.**

| # | band | track | last SCAN | last DEEP | ledger | note |
|---|---|---|---|---|---|---|
| D1 | D | Opponent doctrine | **2026-09-02** | **2026-09-02** | [`../Opponent Analysis/OPPONENT_CLAIMS_REGISTER.md`](../Opponent%20Analysis/OPPONENT_CLAIMS_REGISTER.md) | ⭐ **Mobileye Compound AI**, 3 claims (M-1…M-3). First Band-D item backed by a **proof**; counter-search flipped it to **CONTESTED** |
| A1 | A | World models | 2026-09-01 | 2026-09-01 | [`LEDGER_A1_world_models.md`](LEDGER_A1_world_models.md) | ⚠️ not scanned this pass |
| A2 | A | JEPA / predictive architectures | **2026-09-02** | **2026-09-02** | ⭐ [`LEDGER_A2_jepa.md`](LEDGER_A2_jepa.md) **(CREATED — FS-6 fixed)** | ⭐⭐⭐ **Delta-JEPA `2606.31232`, full text. LeWM measured action-insensitive on OUR diagnostic; endpoint-concat probes are leak-confounded** |
| A3 | A | Vision encoders | 2026-09-01 | — | — | ⛔ **E1 now at THREE probes, still empty.** Promote to the standing empty-search register |
| A4 | A | VLA | **2026-09-02** | **2026-09-02** | ⭐ [`LEDGER_A4_vla.md`](LEDGER_A4_vla.md) **(CREATED — FS-6 fixed)** | `2512.16760` full text; ⛔ **"sub-50 ms" struck as UNATTRIBUTED** |
| A5 | A | Benchmarks + evaluation | **2026-09-02** | **2026-09-02** | [`LEDGER_A5_benchmarks.md`](LEDGER_A5_benchmarks.md) | ⭐ WorldRoamBench **full text** — drift metric fully specified, **FS-1 UNBLOCKED** |
| B1 | B | VLM / multimodal / omni | 2026-08-31 | partial | — | ⚠️ not scanned |
| B2 | B | Post-transformer | **2026-09-02** | **2026-09-02** | [`LEDGER_B2_post_transformer.md`](LEDGER_B2_post_transformer.md) | HALO **full text**; ⛔ **payoff regime is 128K–1M context — NOT ours.** I-2 downgraded |
| B3–B8 | B | decoding / training / RL / self-improving / diffusion / tokenizers | 2026-08-31 | varies | — | ⚠️ **not scanned** |
| B9 | B | Data curation | **2026-09-02** | — | — | ⭐ **first B9 signal** (MiniWorld `2608.01127` banked; Summer-22B). ⚠️ Open-Sora 70M→10M figure is **RELAYED**, primary unread (debt D-8) |
| B10 | B | Semantic search / retrieval | 2026-08-31 | 2026-08-31 | [`LEDGER_C3_regulatory.md`](LEDGER_C3_regulatory.md) (B10 section) | ⚠️ not scanned |
| B11 | B | Physics-informed operators | 2026-08-31 | ⛔ withdrawn source | — | ⚠️ not scanned |
| B12 | B | Memory / long context | **2026-09-02** | **2026-09-02** | [`LEDGER_B12_memory_horizon.md`](LEDGER_B12_memory_horizon.md) | ⭐⭐ VLWM **full text** — **FiLM named as the action-entangling mechanism**; re-opens P-2 branch (c); corroborates L-1 |
| B13 | B | 3D / occupancy | **2026-09-02** | **2026-09-02** | [`LEDGER_B13_3d_geometry.md`](LEDGER_B13_3d_geometry.md) | ⛔ **D-5 DISCHARGED → LINE CLOSED** (needs occupancy GT + LiDAR). ⭐ Redirect: self-supervised occupancy + our NuRec/gsplat |
| C1 | C | Lab + AV releases | 2026-09-01 | 2026-09-01 | [`LEDGER_C1_releases.md`](LEDGER_C1_releases.md) | ⚠️ no new releases probed this pass |
| C2 | C | Engineering blogs / release notes | 2026-09-01 | 2026-09-01 | [`LEDGER_C1_releases.md`](LEDGER_C1_releases.md) (C2) | ⚠️ not scanned |
| C3 | C | Regulatory | **2026-09-02** | **2026-09-02** | [`LEDGER_C3_regulatory.md`](LEDGER_C3_regulatory.md) | ⛔ **third route to the primary 403 — D-4 ESCALATED TO THE PI.** ⭐ SMS spans *post-deployment* ⇒ I-3 is a safety-case problem, not legality |
| C4 | C | Community signals / leaderboards | **2026-09-02** | **2026-09-02** | ⭐ [`LEDGER_C1_releases.md`](LEDGER_C1_releases.md) (C4 entry, **opened**) | ⭐⭐ **DrivoR 56.3 vs privileged PDM-Closed 56.6** — the efficiency wedge must now beat ~40 M, not 32 B. ⚠️ split-mixing barred (V-5) |

### ⛔ Banked-but-unread — the count FS-5 asked for, first instalment

`MEASURED this pass. FS-5 proposed this column; here is the evidence that it is the Lab's dominant inefficiency.`

**6 of the 8 primaries needed today were ALREADY BANKED.** Yesterday: 5 of 6. On 2026-08-31: 2 of 10.
⛔ **The single most consequential finding of this pass — Delta-JEPA — was banked and unread**; the web
surfaced it as a discovery and `kb_add` answered *"already banked"*.

| track | known banked-but-unread |
|---|---|
| A2 | ⛔ was `2606.31232` (**read this pass**); `2602.03604` EB-JEPA abstract-only |
| B5 / B6 / B7 / B8 | AtomVLA, RLVR-World, `2607.07663`, GoalFlow, MambaVideo — **all abstract-only since 2026-08-31** |
| B9 | `2608.01127` MiniWorld (banked this pass, unread) |
| A1 | `2603.09086` abstract-only |

> ⭐ **A banked primary with no five-dimension analysis is not an asset, it is a debt with a hash — and the
> highest-value item in the programme can sit in it.** Library: **313 entries / 2,136.4 MB.**

### ⛔ Coverage failure, stated plainly

**11 of 22 tracks scanned · 7 DEEP · 12 tracks NOT SCANNED** (A1, A3, B1, B3–B8, B10, B11, C2).

⛔ **This is the SECOND CONSECUTIVE DAY below the breadth mandate.** Yesterday's was a deliberate one-off
debt-clearing choice and said so. **Twice is a pattern**, and the charter's §6 fail-loudly clause applies:
the pass met the Band-B minimum (3 deep-reads) and the Band-C sweep, and **failed track breadth**.
⭐ The cause is structural, not lazy: **Band D + full-text debt-clearing consumes the budget, and both are
ranked above breadth by amendment §8.1.** ⇒ **Escalated to the Master Mind as a charter tension needing a
ruling, not another apology.**

## Next rotation — pre-committed

1. ⛔ **Instrument audit: check every action-decodability probe for the endpoint-concatenation leak** (A2 /
   Delta-JEPA). 0 GPU, and it sits underneath the programme's gating problem.
2. **LeWorldModel `2603.19312` + Sub-JEPA `2605.09241`** — bank and read (debt D-9). We are citing LeWM's
   failure through its critic only.
3. **`1604.06915` full text** (debt D-7) — how far does the sample-complexity separation generalise?
4. ⭐ **B1, B3–B8, B10, B11, C2 SCAN — breadth restoration is now the top rotation item, not the last.**
5. **B9 primary** (debt D-8) — video/driving curation, for backlog row 22.
6. **A3 fourth probe or formal retirement of E1** — three probes is enough to stop guessing.


---

## Fifth pass, 2026-09-05 (frontier scan; 14 domain packages already existed today and were NOT redone)

⛔⛔ **FIRST, THE GAP THIS PASS DID NOT CAUSE AND DOES NOT BACKFILL: 2026-09-03 and 2026-09-04 produced
domain packages but NO FRONTIER SCAN AT ALL.** Two days of Band-D, Band-B and Band-C coverage do not
exist. **Escalated to the Master Mind as a scheduling failure, distinct from GS-7's budget tension.**

⭐ **This pass: the first WAYVE adjudication ever (Band D), two full-text deep reads, THREE ledgers created,
one four-day-old empty RESOLVED, and two corrections to our own records.**

| # | band | track | last SCAN | last DEEP | ledger | note |
|---|---|---|---|---|---|---|
| D1 | D | Opponent doctrine | **2026-09-05** | **2026-09-05** | [`../Opponent Analysis/OPPONENT_CLAIMS_REGISTER.md`](../Opponent%20Analysis/OPPONENT_CLAIMS_REGISTER.md) | ⭐⭐ **Wayve GAIA-4** (2026-08-03), claims **V-1…V-4**. ⛔ The concession — *"no vehicle, pedestrian, or cyclist changes its behavior in response to the AI Driver"* — makes their closed loop **EGO-ONLY against a frozen world**, i.e. **ours**. Our binding open/closed-loop ruling **SURVIVES and GAIA-4 sits on our side of it** |
| A1 | A | World models | **2026-09-05** | 2026-09-01 | [`LEDGER_A1_world_models.md`](LEDGER_A1_world_models.md) | scanned via Band D / C1; no dedicated DEEP |
| A2 | A | JEPA / predictive architectures | **2026-09-05** | **2026-09-05** | [`LEDGER_A2_jepa.md`](LEDGER_A2_jepa.md) | ⭐⭐⭐ **ATM `2606.09028`, FULL TEXT.** Probe input `[z_t, z_{t+1}, Δz]` ⇒ **the diagonal carries the GS-1 endpoint leak; the OFF-DIAGONAL `G_{T→P}` does not.** ρ=0.813 vs 0.498 for prediction loss. ⛔ **AITS is encoder-side — SECOND consecutive action-identifiability objective blocked by our frozen trunk** |
| A3 | A | Vision encoders | **2026-09-05** | **2026-09-05** | ⭐ [`LEDGER_A3_vision_encoders.md`](LEDGER_A3_vision_encoders.md) **(CREATED)** | ⭐⭐ **EMPTY E1 RESOLVED AT THE FOURTH PROBE**, one pass before formal retirement. `2501.08118`: frozen DINOv2 + LSS **+7.4 IoU on half the data/iters**. ⇒ **the suspect moves off the trunk onto the READOUT**; GS-4 gains priority |
| A4 | A | VLA | 2026-09-02 | 2026-09-02 | [`LEDGER_A4_vla.md`](LEDGER_A4_vla.md) | ⚠️ not scanned this pass |
| A5 | A | Benchmarks + evaluation | **2026-09-05** | **2026-09-05** | [`LEDGER_A5_benchmarks.md`](LEDGER_A5_benchmarks.md) | ⛔ **navhard EPDMS does not triangulate.** Three independent tables cap it at **≤45.0** (GuideFlow full text 23.1–45.0 · IDOL 38.0 · RAP-DINO 36.9) against our recorded **55.5 / 56.3**. **Row 32 BARRED** — new debt **D-10** |
| B5 | B | Post-training / RL | **2026-09-05** | **2026-09-05** (abstract-level) | ⭐ [`LEDGER_B5_post_training.md`](LEDGER_B5_post_training.md) **(CREATED)** | ⭐ `2607.08072`: scalar GRPO-style rollout reward **hides cross-dimension degradation**; fix is **event-localised** credit. ⇒ **today's MEASURED RL fan-safety regression is a known field failure mode, not our bug** |
| B7 | B | Diffusion + flow matching | **2026-09-05** | **2026-09-05** | ⭐ [`LEDGER_B7_diffusion_flow.md`](LEDGER_B7_diffusion_flow.md) **(CREATED)** | ⭐⭐⭐ **GuideFlow `2511.18729`, FULL TEXT. The SCORER is worth +15.9 EPDMS; all three generative constraint mechanisms together +4.0.** And **per-step forcing (CVF, +1.4) is the WEAKEST** — *"CF applies once"*. ⇒ REF-C's 201/201 ranking blindness disconnects this family's largest lever |
| C1 | C | Lab + AV releases | **2026-09-05** | **2026-09-05** | [`LEDGER_C1_releases.md`](LEDGER_C1_releases.md) | GAIA-4 (2026-08-03); Wayve+Uber London launch 2026-09-03. ⛔ **`LEADERBOARD.md:1370` superseded** — it still calls GAIA-3 *"offline"* |
| C2 | C | Engineering blogs / release notes | **2026-09-05** | — | [`LEDGER_C1_releases.md`](LEDGER_C1_releases.md) (C2 sub-entry) | ⚠️ **EMPTY E3b at one probe.** Newest NVIDIA dev-blog item 2026-03-12. Alpamayo-1 on DRIVE Thor uses FP8 ViT at *"production-viable latencies"* — **no number, so FS-4 stands unchanged** |
| C4 | C | Community signals / leaderboards | **2026-09-05** | **2026-09-05** | [`LEDGER_A5_benchmarks.md`](LEDGER_A5_benchmarks.md) | ⭐ **NAVSIM navhard Stage 2 IS a 3DGS counterfactual pseudo-closed-loop evaluation** — independent confirmation of the Band-D doctrine, and direct support for backlog row 14 |

### ⛔ Coverage this pass — stated, not smoothed

**10 of 22 tracks scanned · 6 DEEP (2 full text) · 12 tracks NOT SCANNED** (A4, B1, B2, B3, B4, B6, B8,
B9, B10, B11, B12, B13, C3 — and C2 only partially).

⛔ **FOURTH CONSECUTIVE DAY below the breadth mandate**, counting 09-03 and 09-04 as total non-passes.
⚠️ **Band-B minimum MISSED: 2 deep reads (B7 full text, B5 abstract-level), not 3.** B3 was not reached.
**Stated as a miss.**

⛔⛔ **GS-7 RE-ESCALATED, NOT RE-APOLOGISED FOR.** Amendment §8.1 ranks Band D and full-text depth **above**
breadth; §6 mandates breadth. **A pass cannot honour both in one budget, and four days is no longer a
budget choice — it is the charter contradicting itself.** ⇒ **A Master-Mind ruling is required: re-scope §6
into an explicitly budgeted scan-only rotation (N tracks/day on a fixed cycle), or cap depth.**

### ⭐ Banked-but-unread — the FS-5 count, third instalment

**2 of 7 primaries needed today were already banked (29 %).** Trend: 2026-08-31 **20 %** → 09-01 **83 %** →
09-02 **75 %** → **09-05 29 %**. ⭐ **The fall is real and is explained: this pass went to genuinely new
material (Wayve doctrine, a June-2026 diagnostic, a CVPR-2026 planner) rather than to standing debts.**
⛔ **But V-1 still paid twice**: ATM was already held, and the check **corrected debt D-9**, which asserted
two papers were UNBANKED when both are in `library.json`.

**Library: 441 entries / 2,890.2 MB** (was 313 / 2,136.4 MB on 2026-09-02).

## Next rotation — pre-committed, so it cannot drift

1. ⛔ **B3, B1, B2, B4, B6, B8, B9, B10, B11, C3 SCAN — breadth restoration**, and it stays item 1 until a
   Master-Mind ruling on GS-7 changes the mandate. **Four days is enough.**
2. ⛔ **LeWorldModel `2603.19312` full text (debt D-9).** We now cite it through **two** critics (Delta-JEPA
   and ATM) without having read it. **This is the register's own named failure class.**
3. **`2501.08118` §method (A3)** — the lift-head specification, to settle whether P-5's "behind on 4 of 5"
   is a HEAD result or a REPRESENTATION result. Blocks GS-4's priority.
4. ⛔ **Reconcile the navhard scoring basis (new debt D-10)** — three independent tables vs our two numbers.
   Blocks every external comparability claim we hold.
5. **`2606.30807` "Off the Rails"** — bank and read. Its relevance jumped this pass: B7 measured the
   **scorer** as the dominant component, and this is a published attack on scoring heads.
6. **A4, A1 dedicated DEEP** — both Band A, both carried indirectly for two passes.

---

## Sixth pass, 2026-09-09 (LAB-RUN-010) - ⭐ BREADTH RESTORED: 22 of 22 SCANNED

⛔⛔ **FIRST, THE GAP: 2026-09-03, 09-04, 09-06, 09-07 and 09-08 produced NO FRONTIER SCAN.** The
2026-09-05 pass escalated the first two; **09-06 to 09-08 are new and the count is now five days.**
2026-09-07 produced eleven domain packages, so the Lab was working - what is missing is specifically
this artifact's Band-D / Band-B / Band-C coverage. **Re-escalated to the Master Mind as a scheduling
failure, distinct from GS-7's budget tension.**

⭐ **This pass: 22/22 tracks scanned, 6 DEEP (3 FULL TEXT), a new Band-D adjudication (Alpamayo 1.5,
claims A15-1..A15-4), debt D-10 resolved in mechanism, LEDGER_B10 opened, and TWO corrections to our
own records.**

| # | band | track | last SCAN | last DEEP | ledger | note |
|---|---|---|---|---|---|---|
| D1 | D | Opponent doctrine | **2026-09-09** | **2026-09-09** | [`../Opponent Analysis/OPPONENT_CLAIMS_REGISTER.md`](../Opponent%20Analysis/OPPONENT_CLAIMS_REGISTER.md) | ⭐⭐ **NVIDIA Alpamayo 1.5**, claims A15-1..A15-4. Concession A15-2 (*teacher runs in cloud or on-prem*, **24-60 GB VRAM**) = **CONFIRMS-US**. A15-3 (verbose NL CoT) **UNSUPPORTED-AS-STATED** at two independent contradictions |
| A1 | A | World models | **2026-09-09** | **2026-09-09** | [`LEDGER_A1_world_models.md`](LEDGER_A1_world_models.md) | Latent-WAM `2603.24581` **FULL TEXT**. ⛔ **104M AT INFERENCE / 191M in training** - new guideline T-5 |
| A2 | A | JEPA / predictive architectures | **2026-09-09** | **2026-09-09** | [`LEDGER_A2_jepa.md`](LEDGER_A2_jepa.md) | ⭐⭐⭐ **ACPC `2608.12939` FULL TEXT. Post-hoc diagnostics on a FROZEN model - the first item in this family our frozen trunk does NOT block.** SR collapse **0.066** vs healthy **0.967-0.984**. **Corrects this ledger's own over-broad conclusion.** THIRD critic of LeWM - **D-9 hardens** |
| A3 | A | Vision encoders | **2026-09-09** | partial | [`LEDGER_A3_vision_encoders.md`](LEDGER_A3_vision_encoders.md) | scanned via the B13 query (OccFeat, SelfOcc, DVGT `2512.16919`); no dedicated DEEP |
| A4 | A | VLA | **2026-09-09** | **2026-09-09** | [`LEDGER_A4_vla.md`](LEDGER_A4_vla.md) | via the Band-D counter-search: XCoT-VLA `2608.10976`, LCDrive `2512.10226`, runtime monitor `2608.29583`. ⭐ **Independently re-confirms Waymo W-7 from a second stack** |
| A5 | A | Benchmarks + evaluation | **2026-09-09** | **2026-09-09** | [`LEDGER_A5_benchmarks.md`](LEDGER_A5_benchmarks.md) | ⭐⭐⭐ **D-10 MECHANISM RESOLVED: two disjoint EPDMS populations** (navhard 23.1-45.0; "NAVSIM v2" 12k-scenario 84.8-89.3), maintainers discourage self-reported splits. **Provenance of OUR 55.5/56.3 still open; row 32 stays BARRED** |
| B1 | B | VLM / multimodal / omni | **2026-09-09** | - | - | OmniSpatial (ICLR 2026), `2512.19683` spatial-reasoning gap, `2603.06054` lightweight driving-VLM probing |
| B2 | B | Post-transformer | **2026-09-09** | - | [`LEDGER_B2_post_transformer.md`](LEDGER_B2_post_transformer.md) | scanned; SSM-as-neural-operator `2409.03231`. No live decision moved (V-4); I-2 still gated on P-6's benchmark |
| B3 | B | Efficient decoding | **2026-09-09** | - | - | ⛔ **EMPTY at a TERM-VARIED second probe (E-B3).** All hits LLM token-serving. **The one Band-B track with no transfer path in three passes** |
| B4 | B | Efficient training | **2026-09-09** | - | - | scanned via Q9; muP referenced only in AR-scaling context |
| B5 | B | Post-training / RL | **2026-09-09** | **2026-09-09** (abstract-only) | [`LEDGER_B5_post_training.md`](LEDGER_B5_post_training.md) | ⭐ **RLIR `2509.23958`: 5-10 % action-following gain - and BLOCKED FOR US pending the GS-1 endpoint-leak audit**, because its inverse-dynamics reward is exactly that shape |
| B6 | B | Self-improving systems | **2026-09-09** | - | - | Curriculum-RLAIF `2505.20075`; RL post-training demystified `2608.24949` |
| B7 | B | Diffusion + flow matching | **2026-09-09** | - | [`LEDGER_B7_diffusion_flow.md`](LEDGER_B7_diffusion_flow.md) | FlowR2A `2606.24231` (reward-to-action, unifies scoring-dense supervision with anchor generation); WAM-Flow `2512.06112` |
| B8 | B | Tokenizers | **2026-09-09** | - | - | InfoTok `2512.16975`, iFSQ `2601.17124`, VideoFlexTok `2604.12887` |
| B9 | B | Data curation | **2026-09-09** | partial | [`LEDGER_B10_retrieval.md`](LEDGER_B10_retrieval.md) | ⛔ **Kairos `2606.16533` curation section UNREAD AT THREE ROUTES** - claim stays `RELAYED`; **debt D-8 WORSENS**. VidaForge `2609.06652` newly surfaced |
| B10 | B | Semantic search / retrieval | **2026-09-09** | **2026-09-09** | ⭐ [`LEDGER_B10_retrieval.md`](LEDGER_B10_retrieval.md) **(CREATED)** | ⭐⭐⭐ **`2606.28383` FULL TEXT: zero-label clip-complexity = JEPA prediction error, AP 0.512 vs 0.436.** ⛔ **constant-velocity baseline rho=0.314** (kinematic floor) and **no-EMA = 44x collapse at FIXED alpha=0.996** |
| B11 | B | Physics-informed operators | **2026-09-09** | - | - | Chronos `2606.30318` (physics-informed, non-Markovian, long-horizon). ⚠️ PI-JEPA `2604.01349` remains WITHDRAWN |
| B12 | B | Memory / long context | **2026-09-09** | - | [`LEDGER_B12_memory_horizon.md`](LEDGER_B12_memory_horizon.md) | `2606.25136` memory retrieval; `2606.16178` short-term-memory scaling |
| B13 | B | 3D / occupancy | **2026-09-09** | - | [`LEDGER_B13_3d_geometry.md`](LEDGER_B13_3d_geometry.md) | ⭐ **OccFeat / SelfOcc found - the self-supervised occupancy family S-3 redirected us to**, needing no occupancy GT. Feeds GS-4 |
| C1 | C | Lab + AV releases | **2026-09-09** | **2026-09-09** | [`LEDGER_C1_releases.md`](LEDGER_C1_releases.md) | Alpamayo 1.5 (the Band-D item) |
| C2 | C | Engineering blogs / release notes | **2026-09-09** | - | [`LEDGER_C1_releases.md`](LEDGER_C1_releases.md) (C2) | ⚠️ **EMPTY E-C2b, second probe.** Newest JetPack still 7.2.1 (2026-08-12). **FS-4 unchanged for a second pass** |
| C3 | C | Regulatory | **2026-09-09** | - | [`LEDGER_C3_regulatory.md`](LEDGER_C3_regulatory.md) | ⛔ **D-4 routes 4 AND 5 FAILED (five total).** ⭐ **Character changed: route 4 is a DIRECT primary URL that exists and is domain-blocked (403) - a RETRIEVAL-CHANNEL problem, not a discovery one** |
| C4 | C | Community signals | **2026-09-09** | - | [`LEDGER_C1_releases.md`](LEDGER_C1_releases.md) (C4) | Commercial only: Waymo 3 new cities + >500k rides/week; Zoox Houston/San Diego; Tesla Cybercab public riders since 2026-09-04; Mobileye 2027 |

### ⭐ Coverage - and the GS-7 tension answered empirically rather than by ruling

**22 of 22 tracks SCANNED. 6 DEEP, 3 of them FULL TEXT. Band-B minimum MET** (B10, B9, B5).

⭐⭐ **BREADTH IS RESTORED, ending a four-pass failure.** The 2026-09-05 pre-committed rotation item 1
was *"B3, B1, B2, B4, B6, B8, B9, B10, B11, C3 SCAN - breadth restoration, and it stays item 1 until a
Master-Mind ruling on GS-7 changes the mandate. Four days is enough."* **All ten were scanned.**

⭐ **GS-7 now has evidence, not just a complaint.** Breadth and depth coexisted in one budget **because
two of the three full-text reads came from SCAN hits on stale tracks (B10 and A5), not from the debt
list.** ⇒ **Amendment 8.1's depth-first ordering starves breadth only when depth is spent on standing
debts.** **Proposed ruling: keep 8.1's priority order and add one clause - the day's full-text budget is
spent on the best hit AVAILABLE, whether it came from a debt or from a scan.**

### ⛔ Banked-but-unread - the FS-5 count, fourth instalment

**4 of 8 primaries needed today were ALREADY BANKED (50 %).**
Trend: 2026-08-31 **20 %** -> 09-01 **83 %** -> 09-02 **75 %** -> 09-05 **29 %** -> **09-09 50 %**.

⛔⛔ **And the 2026-09-02 Delta-JEPA pattern repeated exactly: `2603.24581` Latent-WAM - the paper that
resolved D-10's mechanism and produced guideline T-5 - was ALREADY IN THE LIBRARY, UNREAD.** So were all
three Band-D counter-evidence papers, which means **the entire contradiction step of today's adjudication
came from papers we already held.**

**Library: 491 entries, 3,198.3 MB** (was 441 / 2,890.2 MB on 2026-09-05).

## Next rotation - pre-committed, so it cannot drift

1. ⛔⛔ **LeWM `2603.19312` full text (debt D-9). THREE critics deep now** (Delta-JEPA, ATM, ACPC). This
   is the register's own named failure class and it is banked and 0 GPU. **It is item 1 and stays there.**
2. ⭐ **Run the SR/IR probe on banked v7 arms with our own controls** (constant-predictor + shuffled-action)
   - 0 GPU, and it sits underneath the v7f freeze.
3. **Kairos `2606.16533` curation section by a FOURTH route** (debt D-8) - try the ar5iv mirror or the
   listing-level HTML, since pdf/abs/html v1 all failed.
4. **A3 and B13 dedicated DEEP** - OccFeat/SelfOcc supervision requirements, for GS-4.
5. **Split-stamp extraction for all six external EPDMS numbers** (D-10 provenance half).
6. ⚠️ **D-4 is NOT a Lab item any more** - five routes, one of them a direct 403'd primary URL. It needs a
   human browser, and it is with the PI.

---

## Seventh pass — 2026-09-10 (LAB-RUN-011)

| # | band | track | last SCAN | last DEEP | ledger | note |
|---|---|---|---|---|---|---|
| D1 | D | Opponent doctrine | **2026-09-10** | **2026-09-10** | [`../Opponent Analysis/OPPONENT_CLAIMS_REGISTER.md`](../Opponent%20Analysis/OPPONENT_CLAIMS_REGISTER.md) | ⭐ **NVIDIA closed-loop post-training** (Ivanovic & Pavone, 2026-05-31) adjudicated 7 steps → **A16-1…A16-3**. ⛔ **Zero numbers, zero ablations in the post**; contested at ≥2 independent sources. New debt **D-12** |
| A1 | A | World models | **2026-09-10** | **2026-09-10** | [`LEDGER_A1_world_models.md`](LEDGER_A1_world_models.md) | ⭐⭐⭐ **Drive-HWM `2609.03572` FULL TEXT.** Hierarchy +0.8 PDMS over its own flat control — ⛔ **NOT params-matched. Row 18 / H1b still has no matched datapoint** |
| A2 | A | JEPA / predictive architectures | **2026-09-10** | **2026-09-10** | [`LEDGER_A2_jepa.md`](LEDGER_A2_jepa.md) | ⭐⭐⭐ **TWO full texts.** `2601.00844` value-guided JEPA (the winner is an **encoder** loss; joint training loses 5/6; `pred EMA` **0.04** on Maze) · `2603.19312` **LeWM (debt D-9)** — **AdaLN per layer = our FiLM family**, and **no action-sensitivity control at all** |
| A3 | A | Vision encoders | **2026-09-10** | — | [`LEDGER_A3_vision_encoders.md`](LEDGER_A3_vision_encoders.md) | ⛔ **E1 EMPTY at a FIFTH probe**, term varied twice. Approaching genuine absence — ⭐ and if so, a positioning asset |
| A4 | A | VLA | **2026-09-10** | — | — | scanned only: `2608.30144` *Rethinking Language's Role in Efficient VLA for AVs* (2026-08-31), DeeAD, FASTer, LinkVLA. ⚠️ **no DEEP this pass** |
| A5 | A | Benchmarks + evaluation | **2026-09-10** | **2026-09-10** | [`LEDGER_A5_benchmarks.md`](LEDGER_A5_benchmarks.md) | ⭐⭐⭐ **D-10 COUNTS RESOLVED from the maintainers' primary** — navhard **450 S1 / 5,462 S2**; navtest ≈12 k. ⛔ **BASIS STILL OPEN: PDM-Closed reads 51.3 AND 56.6 on "navhard".** ⚠️ our own "≤ 45.0 cap" is **superseded** |
| B1 | B | VLM / multimodal / omni | 2026-09-09 | — | — | ⚠️ **not scanned this pass** — swept only through other result sets |
| B2 | B | Post-transformer | 2026-09-09 | — | [`LEDGER_B2_post_transformer.md`](LEDGER_B2_post_transformer.md) | ⚠️ not scanned. I-2 still gated on P-6's benchmark half |
| B3 | B | Efficient decoding | **2026-09-10** | **2026-09-10** | [`LEDGER_A1_world_models.md`](LEDGER_A1_world_models.md) (via Drive-HWM Table III) | ⭐⭐ **B3's transfer path FOUND after four passes of empties — and it is TEMPORAL ABSTRACTION, not token serving.** Cost model `Tavg = Tf + Ts/N`, `Tpeak = Tf + Ts`. ⛔ **Tpeak 107.2 ms misses a 100 ms budget while Tavg 84.8 ms clears it** |
| B4 | B | Efficient training | **2026-09-10** | — | — | ⚠️ **EMPTY E-B4, first probe.** All hits are language-model training (LLMQ, MuonQ, µLO, MONA, BAOC, Megatron-MoE). **Four passes scanned, still no dedicated DEEP.** Second probe must vary the term |
| B5 | B | Post-training / RL | **2026-09-10** | **2026-09-10** | [`LEDGER_B5_post_training.md`](LEDGER_B5_post_training.md) | ⭐ Band-D counter-search: **the diagnosis is confirmed ≥3×, the prescription is contested ≥2×.** Guideline **T-7** |
| B6 | B | Self-improving systems | 2026-09-09 | — | — | ⚠️ not scanned. ⭐ Kairos v3 §5.1 (*rollout–evaluation–refinement*, understanding module as a built-in reward) is a live B6 lead for the next pass |
| B7 | B | Diffusion + flow matching | **2026-09-10** | — | [`LEDGER_B7_diffusion_flow.md`](LEDGER_B7_diffusion_flow.md) | scanned: FlowR2A `2606.24231`, WAM-Flow `2512.06112`, FlowDrive `2509.21961`, GoalFlow PDMS 90.3. FlowR2A remains the standing candidate for FS9-5 |
| B8 | B | Tokenizers | 2026-09-09 | — | — | ⚠️ not scanned |
| B9 | B | Data curation | **2026-09-10** | **2026-09-10** | [`LEDGER_B10_retrieval.md`](LEDGER_B10_retrieval.md) (B9 section) | ⭐⭐⭐ **Kairos `2606.16533` FULL TEXT at the FIFTH route — debt D-8 DISCHARGED.** ⛔ **The claim is a PROPOSAL: *"does not yet compute CID directly"*.** ⛔⭐ **v1 does NOT contain the claim; v3 does — reading v1 would have filed a false REFUTED** |
| B10 | B | Semantic search / retrieval | 2026-09-09 | 2026-09-09 | [`LEDGER_B10_retrieval.md`](LEDGER_B10_retrieval.md) | ⚠️ not scanned this pass |
| B11 | B | Physics-informed operators | 2026-09-09 | — | — | ⚠️ not scanned. **Still no live primary since `2604.01349` was WITHDRAWN** |
| B12 | B | Memory / long context | 2026-09-09 | — | [`LEDGER_B12_memory_horizon.md`](LEDGER_B12_memory_horizon.md) | ⚠️ not scanned. ⭐ Kairos v3 §2.3 (*hybrid multi-scale temporal memory*) is a live B12 lead |
| B13 | B | 3D / occupancy | 2026-09-09 | — | [`LEDGER_B13_3d_geometry.md`](LEDGER_B13_3d_geometry.md) | ⚠️ not scanned. FS9-9 (OccFeat / SelfOcc supervision requirements) still open |
| C1 | C | Lab + AV releases | **2026-09-10** | **2026-09-10** | [`LEDGER_C1_releases.md`](LEDGER_C1_releases.md) | Alpamayo 2 Super commercial release; no new Wayve doctrine post |
| C2 | C | Engineering blogs / release notes | **2026-09-10** | — | [`LEDGER_C1_releases.md`](LEDGER_C1_releases.md) (C2) | ⛔ **EMPTY E-C2c, THIRD probe** — still JetPack 7.2.1 (2026-08-12). ⭐ New: **DOPE ONNX→TRT conversion FAILS on AGX Thor on unsupported layers** — a third-party instance of the D-B1-GATE class |
| C3 | C | Regulatory | — | — | [`LEDGER_C3_regulatory.md`](LEDGER_C3_regulatory.md) | ⛔ **NOT SCANNED — declared.** D-4 at five failed routes, with the PI |
| C4 | C | Community signals | **2026-09-10** | — | [`LEDGER_A5_benchmarks.md`](LEDGER_A5_benchmarks.md) | navhard anchors extracted from the maintainers' primary |

### ⛔ Coverage honesty for this pass

**18 of 22 tracks carry a query or a named sweep. Four do not: B1, B8, B11, B12** (swept only through other result sets), **and C3 was deliberately not probed.** This is the GS-7 tension again — Band D plus four full texts consumed the budget — and it is stated rather than narrowed silently.

⭐ **But FS9-10's proposed clause worked in practice**: three of the four full texts came from that day's own scan hits (Drive-HWM, value-JEPA) or from a debt that a scan hit made cheap to clear (Kairos), rather than from the debt list outranking breadth by construction.

### Banked-but-unread, per track (FS-5 instalment)

| track | banked, unread |
|---|---|
| A2 | `2605.09241` Sub-JEPA — **debt D-9's surviving half** · `2602.03604` EB-JEPA (abstract-only) |
| A4 | `2608.30144` (found today, unbanked and unread) |
| B7 | `2606.24231` FlowR2A, `2512.06112` WAM-Flow (both unread) |
| B9 | `2608.01127` MiniWorld (banked 09-02, still unread) |
| A1 | `2604.03208` *Hierarchical Planning with Latent World Models* (found today, unbanked — a direct row-18 candidate) |


---

## Eighth pass — 2026-09-13 (LAB-RUN-012)

| # | band | track | last SCAN | last DEEP | ledger | note |
|---|---|---|---|---|---|---|
| D1 | D | Opponent doctrine | **2026-09-13** | **2026-09-13** | [`../Opponent Analysis/OPPONENT_CLAIMS_REGISTER.md`](../Opponent%20Analysis/OPPONENT_CLAIMS_REGISTER.md) | Waymo compute post (2026-08-20) → **A17-1…A17-3**; Tesla Cybercab (RELAYED) → **A17-4 CONTESTED**. Concession: *"low-batch regimes we often operate"* |
| A1 | A | World models | **2026-09-13** | **2026-09-13** | [`LEDGER_A1_world_models.md`](LEDGER_A1_world_models.md) | ⭐⭐⭐ **HWM `2604.03208` FULL TEXT — Table 16 capacity control** (flat 98M 35 % vs hierarchy 94M 78 %), frozen encoder. A10-2 branch 1 fires |
| A2 | A | JEPA / predictive architectures | **2026-09-13** | **2026-09-13** | [`LEDGER_A2_jepa.md`](LEDGER_A2_jepa.md) | ⭐⭐ **SG-JEPA `2609.10464` FULL TEXT** — physics as action coordinate; gain located in the encoder (≈12 %) |
| A3 | A | Vision encoders | **2026-09-13** | — | [`LEDGER_A3_vision_encoders.md`](LEDGER_A3_vision_encoders.md) | ⚠️ scan only — **EMPTY for a Sept-2026 encoder release** (probe 1 of a new term). MVV (B1) supplies a DINOv2-vs-VLM datapoint |
| A4 | A | VLA | **2026-09-13** | **2026-09-13** | [`LEDGER_A4_vla.md`](LEDGER_A4_vla.md) | `2608.30144` read — Language Residue L1–L4; **TanitAD is L1** |
| A5 | A | Benchmarks + evaluation | **2026-09-13** | **2026-09-13** | [`LEDGER_A5_benchmarks.md`](LEDGER_A5_benchmarks.md) | ⭐⭐⭐ **D-10 CLOSED** — 51.3 vs 56.6 = `2506.04218` v2 vs **v3**; Stage 1 identical 9/9 |
| B1 | B | VLM / multimodal / omni | **2026-09-13** | **2026-09-13** | [`LEDGER_B1_vlm.md`](LEDGER_B1_vlm.md) *(created)* | MVV `2508.02047` FULL TEXT: signs DINOv2 0.484 vs InternVL-9B 0.467, types not values |
| B2 | B | Post-transformer | **2026-09-13** | — | [`LEDGER_B2_post_transformer.md`](LEDGER_B2_post_transformer.md) | scanned — EMPTY for Sept 2026 |
| B3 | B | Efficient decoding | **2026-09-13** | — | — | scanned — BeaconKV `2609.04971` (LLM-only); real-time AR policies `2606.13355` |
| B4 | B | Efficient training | **2026-09-13** | — | — | ⛔ **EMPTY E-B4, SECOND probe, term varied** (vision/video data-efficient pretraining). Nearest: `2605.19137` (frozen-image-FM video pretraining). ⚠️ candidate for row-39 promotion per FS5-6 — **not promoted** |
| B5 | B | Post-training / RL | **2026-09-13** | — | [`LEDGER_B5_post_training.md`](LEDGER_B5_post_training.md) | scanned — `2609.03225` WM-RL + GRPO multi-style driving |
| B6 | B | Self-improving systems | **2026-09-13** | — | — | scanned — EMPTY for Sept 2026 |
| B7 | B | Diffusion + flow matching | **2026-09-13** | — | [`LEDGER_B7_diffusion_flow.md`](LEDGER_B7_diffusion_flow.md) *(created — the 09-10 link was BROKEN, FS-6 class)* | scanned — `2609.04921` |
| B8 | B | Tokenizers | **2026-09-13** | — | — | scanned — EMPTY for Sept 2026 |
| B9 | B | Data curation | **2026-09-13** | — | [`LEDGER_B10_retrieval.md`](LEDGER_B10_retrieval.md) (B9 section) | scanned — ScenarioCharacterization `2608.16041`, OpenLongTail `2607.09655` |
| B10 | B | Semantic search / retrieval | **2026-09-13** | — | [`LEDGER_B10_retrieval.md`](LEDGER_B10_retrieval.md) | scanned — EMPTY for Sept 2026 |
| B11 | B | Physics-informed operators | **2026-09-13** | **2026-09-13** | [`LEDGER_B11_physics_operators.md`](LEDGER_B11_physics_operators.md) *(created)* | ⭐ **first live primary since the 2604.01349 withdrawal** — SG-JEPA |
| B12 | B | Memory / long context | **2026-09-13** | — | [`LEDGER_B12_memory_horizon.md`](LEDGER_B12_memory_horizon.md) | scanned — 2AM `2609.11308`, MemoryWAM `2606.20562` |
| B13 | B | 3D / occupancy | **2026-09-13** | **2026-09-13** | [`LEDGER_B13_3d_geometry.md`](LEDGER_B13_3d_geometry.md) | ⭐⭐ **OccFeat `2404.14027`** — FS9-9 answered: LiDAR occupancy + DINOv2 features; **our LiDAR BEV GT (A&I, today) supplies it**; encoder-side |
| C1 | C | Lab + AV releases | **2026-09-13** | **2026-09-13** | [`LEDGER_C1_releases.md`](LEDGER_C1_releases.md) | NuRec 26.04 = 1,607 clips; NCore; Waymo 14 cities; Cybercab |
| C2 | C | Engineering blogs / release notes | **2026-09-13** | — | [`LEDGER_C1_releases.md`](LEDGER_C1_releases.md) (C2) | ⛔ **E-C2c EMPTY, FOURTH probe**; ⭐ TensorRT 11.2.1 excludes JetPack |
| C3 | C | Regulatory | **2026-09-13** | — | [`LEDGER_C3_regulatory.md`](LEDGER_C3_regulatory.md) | ⭐ **scanned again after two unscanned passes** — UN ADS Regulation adopted 2026-06-24 (RELAYED); NHTSA comment period → 09-30. ⛔ D-4 routes 6–8 failed |
| C4 | C | Community signals | **2026-09-13** | — | [`LEDGER_A5_benchmarks.md`](LEDGER_A5_benchmarks.md) | navhard 03/2026 snapshot (11 entries); Bench2Drive LB updated 2026-08-28 |

### Coverage honesty

**22 / 22 tracks carry a named query.** DEEP: D1, A1, A2, A4, A5, B1, B11, B13 (+ C1 from HF cards). **A3 scan only.** ⚠️ A17-4 rests on RELAYED sources only.

### Banked-but-unread, per track (FS-5 instalment)

| track | banked, unread |
|---|---|
| A2 | `2605.09241` Sub-JEPA (D-9 half) · `2602.03604` EB-JEPA |
| A5 | `2603.23034` TS-1M (abstract-only, banked today) · `2403.04133` nuPlan (abstract-only) |
| B7 | `2606.24231` FlowR2A, `2512.06112` WAM-Flow |
| B9 | `2608.01127` MiniWorld |
| D | `2305.07147` COLA (abstract-only, banked today) |

**Library: 493 entries, 3,262.1 MB** (was 491 / 3,198.3 MB). ⚠️ Six of today's banks carry **empty title metadata** — content verified by magic bytes and local text extraction.

## Next rotation — pre-committed

1. ⭐⭐ **A3 dedicated DEEP** — two passes scan-only in a row.
2. **`E-BE-S1S2-1`** Stage-1 fingerprint of the external EPDMS rows (0 GPU).
3. **Sub-JEPA `2605.09241`** (D-9's surviving half) and **`1604.06915`** (D-7).
4. **B7 DEEP** — FlowR2A for FS9-5.
5. ⛔ **D-4 is not a Lab item** (eight routes).


---

## Ninth pass — 2026-09-17 (LAB-RUN-014)

| # | band | track | last SCAN | last DEEP | ledger | note |
|---|---|---|---|---|---|---|
| D1 | D | Opponent doctrine | **2026-09-17** | **2026-09-17** | [`../Opponent Analysis/OPPONENT_CLAIMS_REGISTER.md`](../Opponent%20Analysis/OPPONENT_CLAIMS_REGISTER.md) | ⭐⭐ **D-1 DISCHARGED** — the Waymo World Model (2026-02), open 17 days → **W-17-1…4**; NVIDIA Alpamayo 2 Super → **N-17-1…3**. Concessions: *"the longer the simulation, the tougher it is to … maintain stable quality"*, *"purely reconstructive … visual breakdowns"* |
| A1 | A | World models | **2026-09-17** | **2026-09-17** | [`LEDGER_A1_world_models.md`](LEDGER_A1_world_models.md) | ⭐⭐⭐ **Fast-WAM `2603.16666` FULL TEXT** — test-time imagination unnecessary (91.8 vs 91.3/90.6), training-time co-training essential (83.8 without), 190 ms vs 810 ms |
| A2 | A | JEPA / predictive architectures | **2026-09-17** | **2026-09-17** | [`LEDGER_A2_jepa.md`](LEDGER_A2_jepa.md) | ⭐⭐⭐ **WA-JEPA `2608.20974` FULL TEXT** — *"deterministic regression … fundamentally ill-suited"*; flow matching **+1.0 EPDMS** vs masking +0.4. ⛔ 91.7 = **navtest** |
| A3 | A | Vision encoders | **2026-09-17** | ⭐ **2026-09-17** | [`LEDGER_A3_vision_encoders.md`](LEDGER_A3_vision_encoders.md) | ⭐⭐⭐ **DEBT DISCHARGED after two scan-only passes** — FROST-Drive `2601.03460` FULL TEXT, frozen 14B beats fine-tuned 14B; **the ledger's own prescribed term change is what found it** |
| A4 | A | VLA | **2026-09-17** | — | [`LEDGER_A4_vla.md`](LEDGER_A4_vla.md) | ⚠️ **scan only** — covered through Alpamayo 2 Super (34 B VLA, diffusion action decoder); no dedicated VLA primary |
| A5 | A | Benchmarks + evaluation | **2026-09-17** | **2026-09-17** | [`LEDGER_A5_benchmarks.md`](LEDGER_A5_benchmarks.md) | ⭐⭐ **#151 primary read** — EPDMS partition recorded; `E-BE-FIX151-1` fires its second branch. **`2607.07196` L0–L4 ladder**, with the L0-vs-L1/L2 reversal |
| B1 | B | VLM / multimodal / omni | **2026-09-17** | — | [`LEDGER_B1_vlm.md`](LEDGER_B1_vlm.md) | scan through C1 only (Cosmos 3 Super Reasoner, 32 B backbone) |
| B2 | B | Post-transformer | **2026-09-17** | — | [`LEDGER_B2_post_transformer.md`](LEDGER_B2_post_transformer.md) | scanned — Log-Linear Attention `2506.04761` (ICLR 2026), Kalman Linear Attention `2602.10743`, sparse post-training `2512.05865` |
| B3 | B | Efficient decoding | **2026-09-17** | partial | — | ⭐ Fast-WAM's **190 ms vs 810 ms (>4×)** is this track's transfer number, from an A1 paper |
| B4 | B | Efficient training | **2026-09-17** | — | — | ⛔ **EMPTY E-B4, THIRD probe** for a Sept-2026 vision/video item |
| B5 | B | Post-training / RL | **2026-09-17** | — | [`LEDGER_B5_post_training.md`](LEDGER_B5_post_training.md) | scanned — WorldCompass `2602.09022`, diffusion policy optimisation `2605.26282`, VLA-World GRPO, RehearseVLA, WMPO |
| B6 | B | Self-improving systems | **2026-09-17** | — | — | ⛔ **EMPTY E-B6, second probe** |
| B7 | B | Diffusion + flow matching | **2026-09-17** | partial | [`LEDGER_B7_diffusion_flow.md`](LEDGER_B7_diffusion_flow.md) | ⭐ first controlled **number** on this track (WA-JEPA's +1.0), but it arrives via an A2 paper. ⛔ FlowR2A `2606.24231` **still banked-unread — pre-committed rotation item 4, missed twice** |
| B8 | B | Tokenizers | **2026-09-17** | — | — | scanned — iFSQ `2601.17124`, InfoTok `2512.16975`, IDEAL `2606.11096`, FSQ-for-diffusion `2606.09962` |
| B9 | B | Data curation | **2026-09-17** | **2026-09-17** | — | ⭐ KITScenes CLOSED (CC BY-NC 4.0, 4.58 TB, no value field at 3 probes); OSM closed-by-coordinates |
| B10 | B | Semantic search / retrieval | **2026-09-17** | — | [`LEDGER_B10_retrieval.md`](LEDGER_B10_retrieval.md) | ⛔ **EMPTY E-B10, second probe** |
| B11 | B | Physics-informed operators | ⛔ **NOT SCANNED** | — | [`LEDGER_B11_physics_operators.md`](LEDGER_B11_physics_operators.md) | ⛔ **no query today — declared unscanned, not empty.** The one track this pass missed |
| B12 | B | Memory / long context | **2026-09-17** | — | [`LEDGER_B12_memory_horizon.md`](LEDGER_B12_memory_horizon.md) | ⛔ **EMPTY E-B12, second probe** — B-Q5 returned only occupancy/3DGS |
| B13 | B | 3D / occupancy | **2026-09-17** | — | [`LEDGER_B13_3d_geometry.md`](LEDGER_B13_3d_geometry.md) | scanned — GaussianOcc3D `2601.22729`, VG3S `2603.06210`, EnerGS `2604.26238`, `2607.04661` |
| C1 | C | Lab + AV releases | **2026-09-17** | **2026-09-17** | [`LEDGER_C1_releases.md`](LEDGER_C1_releases.md) | ⭐⭐ **OpenMDW-1.1 across the whole Alpamayo lineup**; 34 B = 32 B + 2.3 B; AlpaSim 1.50 ± 0.13 |
| C2 | C | Engineering blogs / release notes | **2026-09-17** | — | [`LEDGER_C1_releases.md`](LEDGER_C1_releases.md) (C2) | **JetPack 7.2.1 = JL 39.2.1 / CUDA 13.2.1 / TRT 10.16.2**; DRIVE OS 7.0.3. ⛔ **E-C2 EMPTY, FIFTH probe** |
| C3 | C | Regulatory | **2026-09-17** | **2026-09-17** | [`LEDGER_C3_regulatory.md`](LEDGER_C3_regulatory.md) | UN GTR adopted at WP.29 23–26 Jun 2026; foot-brake NPRM 06-26, comments closed 07-27, no final rule; ⭐ **today: NHTSA puts the Cybercab under sworn oath** (touches A17-4) |
| C4 | C | Community signals | **2026-09-17** | — | [`LEDGER_A5_benchmarks.md`](LEDGER_A5_benchmarks.md) | ⚠️ scan-through only — Waymo E2E board (FROST-Drive 3rd), HUGSIM 436 scenarios, navtest board |

### ⛔ Coverage honesty

**21 / 22 tracks carry a named query. B11 does not and is declared UNSCANNED.**
**DEEP:** D1 ×2, A1, A2, **A3 (debt discharged)**, A5, B9, C1, C3. **A4 and C4 are scan-only.**
⚠️ **Band-B depth is this pass's weakest column and is not counted generously:** B3 and B7 are served
by *transfer numbers inside A-band full texts*, not by dedicated Band-B primaries. **Counted as ~2 of
the required 3.** Per charter §6 the pass is therefore **INCOMPLETE on two counts** and says so in its
own RESULT.md §4.
**FULL-TEXT reads: 3** (Fast-WAM, WA-JEPA, FROST-Drive) — clears V-2's ≥ 2.
**Library: 503 → 507 entries, 3,438.2 MB.** ⭐ V-1 re-find rate **2 of 5**.

### Banked-but-unread, per track (FS-5 instalment)

| track | banked, unread |
|---|---|
| A2 | ⛔ `2606.31232` **Delta-JEPA — PDF extraction FAILED today**; nearest external relative of row 13 / H-RANK-17 · `2608.07409` UniJEPA (abstract-only) · `2602.03604` EB-JEPA |
| A5 | ⛔ `2607.07196` **full-text debt opened today** (rung definitions read from the abstract only) · `2603.23034` TS-1M · `2403.04133` nuPlan |
| B7 | ⛔ `2606.24231` **FlowR2A — pre-committed rotation item, missed on two consecutive passes** · `2512.06112` WAM-Flow |
| B9 | `2608.01127` MiniWorld |
| D | `2305.07147` COLA |

### Next rotation — pre-committed, so it cannot drift

1. ⛔ **B11** — the one unscanned track; it leads by construction.
2. ⛔ **FlowR2A `2606.24231`** — banked-unread and pre-committed twice. A third miss should be escalated, not re-listed.
3. ⛔ **Delta-JEPA `2606.31232`** via a working extraction path (local pypdf, not the fetch route that failed).
4. **`2607.07196` full text** — extract the operational L1 definition, for backlog row 14 (`E-BE-LADDER-1`).
5. **Band-D debts D-2 / D-3** (Waymo *Demonstrably Safe AI*, Foundation Model post).
6. **A4 dedicated DEEP** — scan-only today.

---

## Tenth pass — 2026-09-18 (LAB-RUN-015)

| # | band | track | last SCAN | last DEEP | ledger | note |
|---|---|---|---|---|---|---|
| D1 | D | Opponent doctrine | **2026-09-18** | **2026-09-18** | [`../Opponent Analysis/OPPONENT_CLAIMS_REGISTER.md`](../Opponent%20Analysis/OPPONENT_CLAIMS_REGISTER.md) | ⭐ **Tesla T-18-1…3** (distillation SUPPORTED/unmeasured; concession: student ships supervised; open-loop-loss claim CONFIRMS-US). ⛔ **D-2/D-3 bookkeeping corrected — already discharged** |
| A1 | A | World models | **2026-09-18** | **2026-09-18** | [`LEDGER_A1_world_models.md`](LEDGER_A1_world_models.md) | ⭐⭐⭐ **DriveZero `2609.06055`** — navhard two-stage **57.1** (above our stamp table) |
| A2 | A | JEPA | **2026-09-18** | ⭐ **2026-09-18 FULL TEXT** | [`LEDGER_A2_jepa.md`](LEDGER_A2_jepa.md) | ⭐ **Delta-JEPA debt discharged** (pypdf); `E-AI-LDAD-0` MEASURED on our trunk |
| A3 | A | Vision encoders | **2026-09-18** | via A1 | [`LEDGER_A3_vision_encoders.md`](LEDGER_A3_vision_encoders.md) | ⛔ E-A3 driving query EMPTY (3rd); served by DriveZero's frozen-VFM ablation (+0.53 PDMS, ViT-S) — **no A3-ledger append today** |
| A4 | A | VLA | **2026-09-18** | ⭐ **2026-09-18** | [`LEDGER_A4_vla.md`](LEDGER_A4_vla.md) | ⭐ **dedicated DEEP (09-17 debt)** — `2605.31041`: open-loop +7.1 % vs closed-loop −14.6 % |
| A5 | A | Benchmarks | **2026-09-18** | ⭐ **2026-09-18 FULL TEXT** | [`LEDGER_A5_benchmarks.md`](LEDGER_A5_benchmarks.md) | ⭐ **`2607.07196` full-text debt discharged** — L1 trivial for 3DGS, L2 binds row 14 |
| B1 | B | VLM | **2026-09-18** | — | [`LEDGER_B1_vlm.md`](LEDGER_B1_vlm.md) | ⛔ E-B1 — no new Sept-2026 VLM release |
| B2 | B | Post-transformer | **2026-09-18** | — | [`LEDGER_B2_post_transformer.md`](LEDGER_B2_post_transformer.md) | ⛔ E-B2 — no video-predictor transfer primary |
| B3 | B | Efficient decoding | **2026-09-18** | partial | — | FLASH speculative VLA (19.1 ms, 3.04×), MeanFlow one-step; FlowR2A's latency table (K=10: 53.2 ms) |
| B4 | B | Efficient training | **2026-09-18** | — | — | ⭐ **empty streak BROKEN** (Muon-in-ViT `2605.24770`, CMuon) — scan only |
| B5 | B | Post-training / RL | **2026-09-18** | ⭐ **2026-09-18** | [`LEDGER_B5_post_training.md`](LEDGER_B5_post_training.md) | DriveZero's DriveRL — 5.70 M privileged PPO teacher distilled into a vision student |
| B6 | B | Self-improving | **2026-09-18** | — | — | ⭐ **empty streak BROKEN** (`2606.12072`, `2604.01985`) — scan only |
| B7 | B | Diffusion / flow | **2026-09-18** | ⭐ **2026-09-18 FULL TEXT** | [`LEDGER_B7_diffusion_flow.md`](LEDGER_B7_diffusion_flow.md) | ⭐⭐ **FlowR2A — missed twice, discharged**; the reward-as-identifier leak |
| B8 | B | Tokenizers | **2026-09-18** | — | — | scan only |
| B9 | B | Data curation | **2026-09-18** | — | — | ActiveAD (CVPR 2026) — scan only; in-corpus side served by `E-DE-SIGN-3` |
| B10 | B | Retrieval | **2026-09-18** | — | [`LEDGER_B10_retrieval.md`](LEDGER_B10_retrieval.md) | ⭐ **empty streak BROKEN** (DriveVLA-M0 47.0 navhard) — scan only |
| B11 | B | Physics operators | ⭐ **2026-09-18** | ⭐ **2026-09-18** | [`LEDGER_B11_physics_operators.md`](LEDGER_B11_physics_operators.md) | ⭐⭐⭐ **09-17's unscanned track leads and delivers**: LGS — flow term +145–155 %; contraction bound |
| B12 | B | Memory / long context | **2026-09-18** | — | [`LEDGER_B12_memory_horizon.md`](LEDGER_B12_memory_horizon.md) | ⭐ **empty streak BROKEN** (DeepSight `2605.10564`) — next rotation #1 |
| B13 | B | 3D / occupancy | **2026-09-18** | — | [`LEDGER_B13_3d_geometry.md`](LEDGER_B13_3d_geometry.md) | ⛔ E-B13 — only 2024-25 items |
| C1 | C | Releases | **2026-09-18** | — | [`LEDGER_C1_releases.md`](LEDGER_C1_releases.md) | Wayve × Uber London (supervised) |
| C2 | C | Release notes | **2026-09-18** | ⭐ **2026-09-18** | [`LEDGER_C1_releases.md`](LEDGER_C1_releases.md) (C2) | ⭐⭐ **TensorRT 11: weak typing + kFP8/kFP4/kINT8 flags REMOVED; JetPack NOT supported** ⇒ backlog row 1 |
| C3 | C | Regulatory | **2026-09-18** | — | [`LEDGER_C3_regulatory.md`](LEDGER_C3_regulatory.md) | NHTSA AV-Framework comments extended to 2026-09-30 |
| C4 | C | Community signals | **2026-09-18** | via A1 | [`LEDGER_A5_benchmarks.md`](LEDGER_A5_benchmarks.md) | navhard stub: DriveZero-Scale, SimWAM `2608.07468` |

### ⛔ Coverage honesty
**22 / 22 tracks carry a named query.** DEEP: D1, A1, A2, A4, A5, **B5, B7, B11** (three dedicated Band-B deep-reads — the bar is met without the transfer-number crutch), C2. **A3 is served only through A1** and gets no ledger append. **Full texts: 3 via local pypdf** (Delta-JEPA, FlowR2A, `2607.07196`) + 2 HTML primaries through the fetch summariser (DriveZero, LGS — flagged in RESULT §4). **10 empties named.** **Library: 507 → 513 entries, 3,504.2 MB.** V-1 re-finds: **4** papers + **1 register-level** (D-2/D-3).

### Next rotation — pre-committed
1. **B12 DEEP** — DeepSight `2605.10564`.
2. **B10 DEEP** — DriveVLA-M0 `2608.10413` (touches I-3).
3. **Local pypdf re-read of DriveZero + LGS** before any of their numbers enters the registry.
4. **D-13** — Elluswamy transcript primary.
5. **A3 driving-specific DEEP** — three consecutive driving-term empties; try "frozen vision foundation model end-to-end planner ablation".
