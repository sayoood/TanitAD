<title>LEDGER A4 — vision-action and vision-language-action models</title>

# LEDGER A4 — Vision-action · vision-language-action (VLA)

⛔ **APPEND-ONLY.**

`⭐ CREATED 2026-09-02. TRACKS.md had linked this file since 2026-08-31 without it existing (backlog FS-6).`
`The 2026-08-31 A4 DEEP is recorded below as INHERITED and marked as such.`

## Our position in this track

TanitAD is **not** a VLA — it is a latent world model with a planner. This track matters for three
reasons: (1) VLAs are the dominant competing paradigm and set the comparison our paper is read against;
(2) the **action-channel question** is live for us (P-2 branch (b): *our "action" is realised motion, not a
command — a genuine command channel has never been tested*), and VLAs are where action representation is
studied hardest; (3) their **latency** literature is the closest external analogue to our MEASURED 100 ms
budget and the P-7 finding that a flat strategic rollout (K=300, ~350-1,015 ms) is undeployable.

---

## Entry 2026-08-31-01 — DriveWorld-VLA (INHERITED from the 2026-08-31 pass)

⚠️ `INHERITED — recorded for continuity at ledger creation; NOT re-verified.` Full-text read on that pass;
one of the two reads that carried its actionable findings (V-2).

---

## Entry 2026-09-02-01 — the driving-VLA survey, and a number that did not survive contact

`PUBLISHED lib 2512.16760 · FULL TEXT READ 2026-09-02 · Hu et al. · arXiv 2025-12-18 · debt D-6 (A4) DISCHARGED`

### ⛔ CORRECTION — the "sub-50 ms" latency requirement is NOT in this primary

A web search summary asserted *"Achieving sub-50ms inference remains an unmet requirement for
safety-critical deployment"* and attributed it to this survey. **The primary contains no latency
requirement at all.** Latency appears once, qualitatively — *"Sparse query methods significantly reduce
inference latency"* — with **no concrete number anywhere in the document**, and there is **no
parameter-count-vs-performance data in any table**.

⇒ **Barred from every TanitAD document.** Its evidence class is not even `RELAYED`; it is **UNATTRIBUTED**.
Had it been carried, a fabricated 50 ms budget would have sat beside our MEASURED 100 ms budget and the
MEASURED ~1,015 ms K=300 figure. *(Third correction in three days whose root cause is identical: the
load-bearing number was the one nobody opened the primary for — cf. D-1's 85.1/92.1 and B2's 25 %.)*

### The taxonomy (what the survey actually gives us)

| paradigm | description | subclasses |
|---|---|---|
| **End-to-End VLA** | perception, reasoning and planning in one model | textual action generators (language heads) · numerical action generators (regression/generative heads) |
| ⭐ **Dual-System VLA** | *"separates slow deliberation (via VLMs) from fast, safety-critical execution (via planners)"* | explicit action guidance (textual rationales) · implicit representation transfer (**structured latent intents**) |

⭐ **"Implicit representations transfer / structured latent intents" is architecturally the nearest
published relative of TanitAD's strategic→tactical latent interface.** Worth tracking by name.

### Stated limitations, and one that argues FOR us

- *"VLA models provide little insight into their decision-making process"* — the interpretability gap.
- *"must simultaneously reason and act in real time, creating challenges for latency and safety"*.
- ⭐⭐ *"the **absence of a dense future-world representation** can restrict long-horizon reasoning and
  planning safety"* — **a VLA survey naming the missing world model as a limitation of VLAs.** This is
  quotable support for the programme's premise, from outside the programme.
- Occupancy models *"rely on costly 3D annotations, which can limit scalability"* — ⭐ independently
  consistent with today's B13/F2 finding that SparseOcc++ needs occupancy GT we cannot produce.

### Evidence class of the whole document

`PUBLISHED` — but it is a **survey**: nothing in it is an ablation, and it makes **no deployment claims**
and names **no deployable architecture**. It may inform positioning; it may not decide a GPU-day.

### Consequence for live rows

| row | movement |
|---|---|
| **P-7** (flat strategic rollout undeployable) | **no external latency budget obtained** — our MEASURED 100 ms stands alone and unchallenged. E-empty recorded. |
| **H1 positioning** | Dual-System VLA is a **third** independent stack with a fast/slow seam (with Waymo W-13 and Mobileye M-1). ⚠️ Convergence of designs is **not** evidence of necessity — FS-3 remains the only route to that. |
| **paper** | the *"absence of a dense future-world representation"* line is a citable external argument for an in-loop world model. |

`Next in this track: RT-VLA 2606.14010 (distillation, claimed 7.9x), VLA-Cache 2502.02175 (claimed 1.7x`
`CUDA speedup) and LinkVLA (claimed 86 % latency cut) are all UNREAD and UNBANKED. ⚠️ All three speedup`
`figures above are RELAYED from search summaries and are barred from use until primaries are read —`
`this ledger's own first entry is a correction of exactly that failure.`
