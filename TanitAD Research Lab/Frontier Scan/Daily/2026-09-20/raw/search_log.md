# search log — Frontier Scan 2026-09-20 (LAB-RUN-017)

`Charter §2: every one of the 22 tracks is SCANNED every day, and every EMPTY search is recorded.
A query that returned nothing useful is a result, not a gap in the log.`

## Queries, in the order run

| # | band / track | query | hits | outcome |
|---|---|---|---|---|
| 1 | **D1** | `Waymo "Reference Driver" blog 2026 human benchmark safety methodology` | 9 | ⭐ **D-14's document found** — waymo.com/blog/2026/06/reference-driver/ + a *Nature Comms* citation |
| 2 | **D1** | fetch the Waymo ReD blog | read | claims verbatim; **zero numbers, zero limitations stated** |
| 3 | **D1** | fetch `nature.com/articles/s41467-026-73345-0` | ⛔ **303 → idp.nature.com** | **BLOCKED (E-D15)** ⇒ W-20-1 `NOT-ADJUDICABLE` |
| 4 | **D1** | ⭐ mandatory step-4 counter-search: `critique NIEON Waymo human driver benchmark counterfactual model criticism limitations behavioral competence baseline` | 9 | ⭐⭐ **the productive probe** — Waymo's own three NIEON concessions **and** an independent critical piece |
| 5 | **D1** | fetch Yoshida (2026-02-02) | read | Koopman / Reimer / Cummings, all on **context**. ⚠️ RELAYED through one journalist |
| 6 | **D1** | fetch `axios.com/2026/09/16/wayve-robotaxi-av-tesla-waymo` | ⛔ **403** | **BLOCKED (E-WV20)** — a Wayve Band-D item was attempted and not obtained |
| 7 | **D1** | Library: is `1604.06915` (debt **D-7**) banked? | ✅ **yes** | ⛔ it was banked all along; five passes said "not reached" |
| 8 | **D1** | local pypdf full text of `1604.06915` | 4 pp, 13,406 chars | ⭐⭐⭐ **D-7 CLOSED** — c-approximate independence + conjunctive target |
| 9 | **A1** | `arxiv September 2026 world model autonomous driving latent prediction new` | 9 | ⭐⭐⭐ **Drive-HWM `2609.03572`**; also DriveFuture, Latent-WM taxonomy `2603.09086` |
| 10 | **A1** | fetch `arxiv.org/abs/2609.03572` (abstract) | read | hierarchical slow–fast; ablations claimed |
| 11 | **A1** | fetch `arxiv.org/html/2609.03572v1` (tables, via summariser) | read | Table IV/main numbers reported |
| 12 | **A1** | ⭐ **local pypdf re-read of the banked PDF** (FS19-9) | 14 pp, 77,347 chars | ✅ **Tables IV, V, VI, VII verified verbatim**; and the local read yielded **Tables VI and VII, which the summariser never surfaced** — findings 2 and A1's FiLM result exist only because of this step |
| 13 | **A1/A5** | in-PDF search for `navtest` / `navhard` / `ego status` | 0 / 0 / **1** | ⛔ **the split is NOT stamped in the paper.** The v2 table carries an `Ego Status` blind row at **64.0** and TransFuser at **76.7** ⇒ **navtest magnitudes** ⇒ 86.4 excluded from the navhard table |
| 14 | **A2** | `arxiv 2609 September 2026 JEPA joint embedding predictive architecture video representation` | 9 | scan: P-JEPA, BiJEPA, MJEPA, VL-JEPA — none a driving WM. The day's A2 substance came from Drive-HWM Table V instead |
| 15 | **A2** | ⭐ **V-1 re-find** on `2601.00844` before citing it | 10 files | ⛔ **not a new find** — this line's founding citation; cited as such |
| 16 | **A2** | local pypdf re-read of `2601.00844` | 7 pp | *"we **learn representations such that** …"*; **`frozen` appears 0 times**; toy wall (200) / maze (80) only |
| 17 | **A3** | driving-specific vision-encoder terms | — | ⛔ **E-A3, 5th consecutive empty** |
| 18 | **B13** | `arxiv September 2026 occupancy BEV representation ablation target supervision driving 3D geometry` | 9 | ⭐ **DualPathOcc `2609.06370`** |
| 19 | **B13** | fetch `arxiv.org/abs/2609.06370` | read | abstract only — **no ablation on the abstract page**, stated |
| 20 | **B13** | fetch `arxiv.org/html/2609.06370v1` | read | ⭐⭐ **Table 2 component ladder + Table 4 depth-supervision result** (finding 3) |
| 21 | **B5/B6** | `arxiv September 2026 reinforcement fine-tuning world model post-training GRPO robot policy self-improving` | 10 | ⭐⭐ **RISE `2602.11075`**, **WMPO `2511.09515`**; scanned WAM-RL, TACO, RehearseVLA, WorldRFT |
| 22 | **B6** | fetch `arxiv.org/abs/2602.11075` | read | findings 4 + 5; ⚠️ **no matched no-self-improvement baseline in the abstract** |
| 23 | **B5** | fetch `arxiv.org/abs/2511.09515` | read | finding 6 — pixel space **on purpose**; ⚠️ **no numbers, no matched latent baseline** |
| 24 | **B2/B3** | `arxiv September 2026 linear attention state space model video prediction efficient decoding KV cache robotics latency` | 9 + 9 | ⛔ **E-B2 (4th)** and **E-B3 (2nd)** — all LLM-serving / image-SSM; no WM-rollout transfer |
| 25 | **C1** | `Waymo Tesla Wayve NVIDIA autonomous driving announcement September 2026 model release blog` | 9 | Waymo 14 metros / ~4,000 vehicles / ~500 k rides-wk; Wayve London-Uber 09-03 + $1.2 B at $8.6 B; NVIDIA physical AI ≈ $6 B |
| 26 | **C2/C3** | `TensorRT JetPack release notes September 2026 NHTSA UNECE autonomous vehicle rule September 2026` | 10 + 9 | ⭐ **JetPack 7.2.1 = TensorRT 10.16.2** ⇒ Thor stays on the D-B1-GATE line (**2nd probe confirming LR15-10**). UN ADS **GTR** adopted at the **June 2026 WP.29** session (same session as R185 / D-4) |
| 27 | **B9 (standing)** | speed-limit reader acceptance thresholds — `traffic sign recognition ... acceptance threshold 2026` | 9 | ⛔ **E-TSR (new)** — GTSRB accuracy only; **no published anchor for our 0.80 bar** |
| 28 | **B1/B9** | `arxiv 2026 vision language model speed limit sign reading autonomous driving evaluation` | 9 | ⭐⭐ **`2606.08860`** — the only located operating point for the E-DE-SIGN-1 task |
| 29 | **B1** | local pypdf re-read of `2606.08860` (FS19-9) | 7 pp, 38,194 chars | ✅ verified; **39 signs, two distinct values, recall loss at *"nighttime rain or heavy fog"*.** 21/39 and 21/22 reconstruct exactly |
| 30 | **B1** | within `2606.08860`: an abstention scoring convention | — | ⛔ **E-ABSTAIN (new)** |
| 31 | **B10/B13** | `arxiv 2026 frozen visual features place recognition versus metric distance estimation which layer stride` | 10 | ⛔ **E-VPR-GEOM (new)** — VPR optimises retrieval; no located analogue for today's stage × task contrast |
| 32 | **A2/B12** | `arxiv 2026 world model value function learned terminal value latent planning driving beats sampling` | 9 | `2604.14732` World-Value-Action; `2609.03294` LEAP (already FS19-4). Neither is a **frozen**-representation result |
| 33 | **C4** | navhard leaderboard above 57.1 | — | ⛔ none located |

## Tracks scanned with no new item worth a row

**B4** (efficient training), **B7** (diffusion/flow — DiffusionDrive 88.x appears only as a Drive-HWM baseline),
**B8** (tokenizers), **B11** (physics operators), **B12** (memory — covered by A1's K-sweep finding, no dedicated probe).
⚠️ **B12 is the pass's under-served clause and is named as such in RESULT §5**, not smoothed over.
