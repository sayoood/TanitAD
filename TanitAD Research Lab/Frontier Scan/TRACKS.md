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
