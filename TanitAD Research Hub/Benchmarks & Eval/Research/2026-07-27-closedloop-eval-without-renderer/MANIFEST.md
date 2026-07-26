# Deliverable manifest — 2026-07-27 closed-loop eval without a renderer

**Agent:** Benchmarks & Eval research (CPU / web only) · **Branch:** `agent/benchmarks-eval-20260721`
**Operating rules honoured:** staged, **never committed, never pushed**, never switched branches.
**Pods:** ⛔ **NONE contacted.** pod1 / pod2 / pod3 / tanitad-eval all untouched. No GPU used. No training load added.
**Parity:** untouched — no episode re-selection, no corpus write.

## Artifacts

| # | artifact | where it lives | exists in >1 place? | status |
|:--:|---|---|:--:|---|
| 1 | `PREREGISTRATION.md` — the four repair-conditions (R1–R4) + adopt-conditions (A1–A3), the C13 self-gate, and four falsifiable predictions, **written and staged before any source was read** | `repo:TanitAD Research Hub/Benchmarks & Eval/Research/2026-07-27-closedloop-eval-without-renderer/PREREGISTRATION.md` | repo only ✅ (git-tracked) | **staged** |
| 2 | `CLOSEDLOOP_EVAL_RESEARCH.md` — headline, pre-registration verdict, the pseudo-simulation finding, the NAVSIM correlation question, the replay-validity literature, per-benchmark table, reactive agents, metrics, **the ranked recommendation**, escalations, self-refutations | `repo:…/2026-07-27-closedloop-eval-without-renderer/CLOSEDLOOP_EVAL_RESEARCH.md` | repo only ✅ | **staged** |
| 3 | `CITATIONS.md` — 16 works + 12 licence documents (code and data as separate fields) + 5 internal primary sources + an explicit "could not verify" list | `repo:…/2026-07-27-closedloop-eval-without-renderer/CITATIONS.md` | repo only ✅ | **staged** |
| 4 | `MANIFEST.md` — this file | `repo:…/2026-07-27-closedloop-eval-without-renderer/MANIFEST.md` | repo only ✅ | **staged** |

**Nothing produced in this task exists only on a pod, only in a worktree, or only in an agent's context.**
No code was written, no experiment was run, no checkpoint was touched. This was a literature and licence study.

## Sources consulted but NOT copied into the repo

Two web fetches produced byte artifacts in the tool-result cache, not in the repo:

| what | where | action |
|---|---|---|
| NAVSIM PDF (arXiv 2406.15349, 3.3 MB) — fetch returned undecodable binary; superseded by the HTML version | `C:\Users\Admin\.claude\projects\…\tool-results\webfetch-1785107076063-at8hbs.pdf` (session cache) | **not staged** — it is a public arXiv PDF, reproducible from the URL in `CITATIONS.md` C2. No loss. |

## ⭐ Integration required — this will NOT happen by itself

Escalated here **and** in `CLOSEDLOOP_EVAL_RESEARCH.md` §9 headline, per operating rule 3
(an integration request written only into a README sat unread for 10 days).

| # | what needs a decision or a cross-stream change | owner |
|:--:|---|---|
| **1** | ⛔ **Waymax licence vs TanitAD's identity.** The licence forbids use *"to train or otherwise develop or improve (directly or indirectly) an artificial intelligence foundation model."* Is a sub-300 M hierarchical latent world model for driving such a model? **Not an agent's call.** | **Sayed** |
| **2** | ⚠️ **`Project Steering/MODEL_REGISTRY.md` must label every closed-loop number `EXTRAPOLATION`** until the Option-1 protocol change lands. Cross-stream; will not happen by itself. | Model-registry agent |
| **3** | ⚠️ **Standing rule proposed:** no external corpus or simulator is adopted until its **licence document** is fetched and its **code and data terms recorded separately**. This study is the **third** licence-from-short-name near-miss (ZOD → nuScenes → Waymax/Bench2Drive). Belongs in `CLAUDE.md` / `AGENT_OPERATING_STANDARD.md`. | **Sayed** / orchestrator |
| **4** | ⚠️ **§7.4 experiment needs an eval-pod slot** — lateral-vs-yaw warp fidelity over `\|dlat\| ∈ {0.5, 1.0, 1.5, 2.0} m`, with the destroyed-observation controls as the dynamic-range scale. CPU / 1 GPU, **touches no training pod**. It decides whether the pseudo-simulation grid is 1-D (heading only) or 2-D. | Benchmarks & Eval |
| **5** | ⚠️ **A map-free metric set** must be designed before Option 1 can produce a score: DAC / Lane Keeping / Driving-Direction / Traffic-Light Compliance are **impossible** on PhysicalAI-AV. Weight **ego progress** heavily (ρ = 0.83); do not lean on collision rate alone (ρ = 0.45); C13-gate every clause for dynamic range. | Benchmarks & Eval |
| **6** | ⚠️ **Retraction-log candidates** — two claims in this study should be checked against `RETRACTION_LOG.md` classes before anyone quotes them onward: (a) NAVSIM v1's correlation coefficient is **graphical, not numeric** (do not quote a number); (b) nuPlan-R's IDM-overestimation figures came from a **search summary, not the paper** — `UNVERIFIED`. | whoever cites them |

## Verification

- `git status --short` for this directory shows 4 added files, all under
  `TanitAD Research Hub/Benchmarks & Eval/Research/2026-07-27-closedloop-eval-without-renderer/`.
- No foreign staged entries were disturbed. Sibling agents' staged work in the index
  (`2026-07-26-situation-classifier`, `2026-07-26-trafficsim-wheelbase`, `2026-07-25-tanitdataset-hf-push`)
  was **left exactly as found** — nothing was committed, so nothing could be swept in.
- `Keys.txt` was never read, printed, or referenced.
