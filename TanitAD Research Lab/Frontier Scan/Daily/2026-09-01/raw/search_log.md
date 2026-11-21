<title>Search log — 2026-09-01 frontier scan</title>

# SEARCH LOG — 2026-09-01

`Charter §3 / §5.4: EVERY query recorded with its hit count, and EVERY empty search NAMED.`
`Retrieval: WebSearch + WebFetch from the dev box. No spend, no GPU, no pod, no Thor.`

⚠️ **PROCESS DEVIATION, declared:** V-1 says *search the Library BEFORE the web*. This pass ran the
web first and the Library second (via `kb_add.py`'s already-banked report). The deviation is recorded
rather than smoothed over, and it produced a MEASURED finding — see Q-BANK below.

---

## Band D — opponent doctrine (priority 1, amendment §8.1)

| # | query | hits | outcome |
|---|---|---|---|
| Q1 | `Waymo "world model" blog 2026 waymo.com research` | 8 | ⭐ **HIT** — located *The Waymo World Model* (2026-02-06), the D-3 debt's highest-value unread document. FETCHED. |
| Q2 | `arxiv 2505.06113 camera lidar autonomous driving comparison` | 10 + 8 (2 rounds) | ⚠️ **first round EMPTY on the ID** — generic lidar/camera fusion surveys only; the ID resolved only on a second, ID-explicit round. **A single-probe "not found" here would have been a false absence.** |
| Q3 | `Waymo Foundation Model end-to-end architecture 2025 December blog` | 10 | ⭐ **HIT** — located *Demonstrably Safe AI* (2025-12), discharging the rest of D-3 and settling D-2. FETCHED. |
| Q4 | `Genie 3 world model long-horizon consistency drift limitations criticism` | 10 | ⭐ **HIT (counter-search, §7.1 step 4)** — independent confirmation that Genie-3-class long-horizon drift is unsolved; surfaced WorldRoamBench `2606.31672`. |
| Q5 | `generative world model simulation sim2real gap autonomous driving evaluation validity critique 2026` | 10 | ⭐ **HIT (counter-search)** — surfaced the latent-WM taxonomy `2603.09086` and the "action faithfulness" framing. |

## Band A — core

| # | query | hits | outcome |
|---|---|---|---|
| Q6 | `vision encoder driving representation learning DINOv3 frozen backbone BEV 2026 arxiv` | 10 | ⚠️ **PARTIAL / E1 SECOND PROBE.** Frozen-DINOv3 dense-prediction SOTA found (COCO mAP 66.1, ADE20k mIoU 63.0). ⛔ **DINOv3 × BEV-driving specifically: EMPTY at the second probe.** E1 now stands at two probes in two different phrasings — recorded as a genuine gap, not folklore. |
| Q7 | fetch `arxiv.org/abs/2603.09086` | 1 | HIT — A1 taxonomy, abstract-only. |
| Q8 | fetch `arxiv.org/abs/2606.31672` | 1 | HIT — A5 benchmark, abstract-only. |

## Band B — neighbouring disciplines

| # | query | hits | outcome |
|---|---|---|---|
| Q9 | `arxiv 2601.22156 state space model linear attention world model` | 7 | ⛔ **EMPTY ON THE ID** — the ID did not resolve via search; the paper was retrieved only by direct `arxiv.org/abs/` fetch (Q10). Surfaced *Long-Context State-Space Video World Models* as an unbanked I-2-adjacent lead. |
| Q10 | fetch `arxiv.org/abs/2601.22156` | 1 | ⭐ **HIT — B2 debt discharged, and the record was WRONG.** See F3. |
| Q11 | fetch `arxiv.org/abs/2606.21775` | 1 | ⭐ **HIT — B12 debt discharged.** |
| Q12 | fetch `arxiv.org/abs/2607.04732` | 1 | HIT — B13 debt discharged; ⚠️ **supervision requirements NOT in the abstract**, i.e. the exact sub-question the rotation asked remains OPEN. |

## Band C — non-paper sources

| # | query | hits | outcome |
|---|---|---|---|
| Q13 | `NVIDIA TensorRT release notes August 2026 FP8 FP4 Blackwell Jetson Thor JetPack` | 7 + 10 + 10 (3 rounds) | ⭐ **HIT — resolves empty E3 from 2026-08-31.** JetPack 7.2.1 (2026-08-12); Thor at compute capability 11.0; FP4 requires CC ≥ 10.0, FP8 ≥ 8.9. |
| Q14 | `UNECE GRVA online learning machine learning in-vehicle regulation ban 2026` | 8 | ⚠️ **EMPTY ON THE BAN CLAIM** — GRVA's Jan-2026 draft ADS regulation located; **no online-learning prohibition surfaced.** |
| Q15 | fetch `unece.org/transport/documents/2026/04/...grva-proposal...` | 0 | ⛔ **HTTP 403 FORBIDDEN.** The GTR primary text is **UNREAD**. Recorded as a live debt, not as an absence. |
| Q16 | fetch `connectedautomateddriving.eu/blog/unece-grva-adopts-draft-global-regulation...` | 1 | ⚠️ **SECOND PROBE, ALSO EMPTY on the ban** — explicitly states ML / online learning / in-vehicle learning / post-deployment updates are **not mentioned**. Found ISMR + DSSAD instead. |

## Library-before-web check

| # | probe | result |
|---|---|---|
| Q-BANK | `kb_add.py` × 6 primaries | ⭐⭐ **5 of 6 ALREADY BANKED** (only `2606.31672` was new). On 2026-08-31 this fraction was 2 of 10. See F6 — this is the pass's most uncomfortable finding. |

---

## Empty searches named (charter §5.4)

- **E1 (carried, now two-probe):** DINOv3 / frozen-encoder × **BEV driving** — unfound in two differently-phrased probes. Frozen-encoder dense prediction generally is *not* empty and is strong.
- **E5 (new):** UNECE GRVA **online-learning ban** — unfound at two independent probes; primary GTR text 403-blocked and unread.
- **E6 (new):** arXiv **ID-based search** is unreliable — `2505.06113` and `2601.22156` both failed to resolve by ID via search and needed direct `/abs/` fetches. ⇒ **Never conclude a banked ID is missing from a search probe.**
