<title>LEDGER C3 — regulatory and safety standards</title>

# LEDGER C3 — Regulatory · safety standards

⛔ **APPEND-ONLY.** Evidence class here is `PUBLISHED-RELEASE-NOTE` / `RELAYED` (law-firm and trade
summaries), **never** `PUBLISHED`, until the regulatory text itself is read. **No claim below is
sourced to a primary regulatory document — every one is a secondary summary.**

---

## Entry 2026-08-31-01 — ⛔ a regulatory constraint lands on an OPEN programme research direction

⚠️ **RELAYED (UNECE/GRVA coverage, 2026-07).** Reported: *"GRVA discussed AI related definitions,
**the ban of online learning** and a draft resolution with guidance on the use of AI in vehicles."*

⭐⭐ **This lands directly on injected backlog row I-3** — *"Self-supervised closed-loop adaptation:
can T1 rollout disagreement (prediction-vs-observation at deployment) supervise **online encoder
updates** without labels?"*

⇒ **If a ban on in-vehicle online learning is adopted, I-3's DEPLOYMENT path closes** and the
technique survives only as an **offline / fleet-loop** method: collect disagreement in the vehicle,
update off-vehicle, ship a validated model. That is a *different experiment* with a different cost
model — and it is far better to learn this before designing the arm than after.

⛔ **DO NOT ACT ON THIS YET.** It is a **RELAYED report of a discussion**, not an adopted regulation,
and I have not read a single GRVA document. The correct next step is to read the primary
(UNECE GRVA session documents), not to re-scope I-3 on a trade summary. **Recorded as a
constraint-to-verify, not a constraint.**

## Entry 2026-08-31-02 — the regulatory ground moved twice in 2026

| item | date | class |
|---|---|---|
| **UNECE established, for the first time, a global regulatory framework for Automated Driving Systems (ADS)** — beyond L3 ALKS toward driverless; GRVA adopted a supplement to UN R157 | **2026-06** | RELAYED (law-firm summary) |
| **NHTSA/DOT**: *"the most significant recalibration of federal AV oversight in years"* — first-ever **commercial robotaxi exemption** (Zoox: 8 FMVSS, up to **2,500 vehicles/year for two years**), plus the **$5 M three-year "ASCEND" consortium** with SAE ITC to develop **first-ever AV performance standards** | **2026-07-30/31** | RELAYED |
| Nevada approved Tesla / Uber / Waymo robotaxi permits | 2026-08-20 | RELAYED |

⭐ **The ASCEND item is the one to watch for us.** *"First-ever AV performance standards"* and *"a
single national standard for AV safety"* means **a standardised evaluation protocol is being
constructed right now**. Our programme's whole eval doctrine (four metric families, T0/T1 tiers,
episode-cluster bootstrap) is an unusually well-specified position to hold while that settles — and
a standard that hardens without our metric families in it is a long-term comparability risk.

**What this changes for TanitAD:** nothing operationally today. It adds one verification debt (the
online-learning item) and one watch item (ASCEND). ⚠️ **We are a research programme on a camera
corpus, not a type-approval candidate** — the honest scope statement from the Waymo adjudication
(S-4) applies here too, and prevents this track from generating work we do not need.

`Next: read a UNECE GRVA primary for the online-learning item before I-3 is designed.`

---

# LEDGER B10 — Semantic search · retrieval · embeddings

## Entry 2026-08-31-01 — the track's yield is a CURATION tool, not a driving-memory tool

**Scanned 2026-08-31.** Hits: `SafeDriveRAG` (knowledge-graph RAG for driving), `RealGen`
(retrieval-augmented controllable traffic scenarios), *Driving-RAG*, `2604.20598` (self-aware vector
embeddings), `2602.07125` **Reasoning-Augmented Representations for Multimodal Retrieval** (banked).

⭐ **The cross-link is worth more than the track itself:** *"Automated Smart Data Curation via
Embedding-Based Scenario Retrieval"* (Springer) **answers the empty search E4 from the first pass**,
where every data-curation hit was LLM-*text* curation and nothing addressed video/driving corpora.
**Embedding-based scenario retrieval is the driving-corpus curation method we failed to find under
B9's query.** ⇒ **E4 is now a resolved empty, and the lesson is that the miss was a query-framing
error, not an absence** — precisely why single-probe empties may never harden into absence claims.

⚠️ **Retrieval-augmented driving *inference* is NOT currently useful to us**: it presumes a knowledge
base and a semantic index we do not have, and it adds latency to a loop today's measurement shows is
already over budget at 37.8 M. **The value in this track is offline — curation and scenario mining
against our parity-locked corpus.**

`Next: pull the Springer scenario-retrieval chapter; assess against our fixed 2376-episode parity corpus.`


---

## Entry 2026-09-01-01 - the UNECE "online-learning ban" is UNFOUND at two probes; the real constraint is ISMR/DSSAD

**Sources:** (1) `connectedautomateddriving.eu` report on the GRVA session; (2) UNECE press/document
listing. **Evidence class:** PUBLISHED-BLOG. ⛔ **The GRVA primary text returned HTTP 403 and is UNREAD.**

Injected row **I-3** carries a `CONSTRAINT-TO-VERIFY`: GRVA *"reportedly discussed the ban of online
learning"* (RELAYED, trade summary, no primary). Result of two independent probes today:

- GRVA **adopted a draft ADS regulation at its 19-23 January 2026 session**; submitted to WP.29 for
  the **23-26 June 2026** session.
- ⛔ **Neither probe surfaced any online-learning / in-vehicle-learning / post-deployment-update
  prohibition.** The second source states explicitly that such provisions are **not mentioned**.

⚠️ **Status is "UNSUPPORTED AT TWO PROBES", NOT "REFUTED".** The document that would settle it was
403-blocked and has not been read. Absence at two locations is stronger than at one and is still not
proof - `CLAUDE.md` rule 2 cuts both ways.

⭐ **What the probes DID find, which the relayed claim obscured:** the framework requires **In-Service
Monitoring and Reporting (ISMR)** (manufacturers report critical and significant occurrences) and a
**Data Storage System for Automated Driving (DSSAD)**. These do not ban adaptation - they bind
**traceability and reporting of deployed behaviour**, which an online-adapting encoder makes
materially harder to satisfy.

⇒ **I-3's in-vehicle path is NOT closed.** Re-scoped: the binding question is **auditability under
ISMR/DSSAD**, not legality. The GRVA primary remains an open debt before the arm is designed.

`Next in this track: obtain the GRVA/WP.29 document text by a non-403 route (UNECE doc server, or the
WP.29 June-2026 session papers).`
