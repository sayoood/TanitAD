<title>PLAN — E-LAB-DATA-0831</title>

# PLAN — E-LAB-DATA-0831

`Executed 2026-08-31. Compute budget ZERO: no GPU, no Thor (k60p30k live), no HF metered call,`
`no download. Cost was reading and two web fetches.`

## Steps as executed

| # | step | status |
|---|---|---|
| 1 | Read the gate's P2(b) corpus claim verbatim and treat it as an **absence claim** to be checked | done |
| 2 | Re-read our own L2D descriptor artifact (`l2d_info.json` = the corpus's `meta/info.json`), not a summary of it | done |
| 3 | Re-verify the licence and the action-space fields at the **official card**, today | done — Apache-2.0; fields confirmed; provenance **absent from the card** |
| 4 | Probe whether the steering layer knows L2D | ⛔ **done twice, and the first probe was WRONG** — see step 5 |
| 5 | ⭐ Re-probe case-sensitively over the whole `Project Steering` dir after the first result looked too clean | done — 9 hits in 3 files; **found the GDPR gate a narrow probe would have hidden**; the finding was rewritten downward as a result |
| 6 | Cross-check the B1 open items (D-B1-100H road-class, D-SPEED-GATE >14 m/s) against native L2D fields | done |
| 7 | Verify the third route (nuScenes CAN) at the citing paper's **full text**, not its abstract | done — `2606.12987` §3 |
| 8 | Write SPEC (criteria first), `raw/L2D_SCHEMA_EVIDENCE.md`, RESULT, COMMS; append KB; stage | done |

## Follow-on work, ranked by decision value / cost

⛔ None authorised by this package. Each needs its own SPEC.

| # | item | owner | cost | why this rank |
|---|---|---|---|---|
| 1 | **`E-DATA-CMDPROV`** — is a bus-sourced steering signal materially decorrelated from the curvature proxy, **stratified by lateral acceleration and steering rate**, with the PhysicalAI self-correlation as the must-read-≈1.0 control | DataFlyWheel | CPU-minutes | it is the gate's own *"cheapest discriminating test"*, it decides whether P2(b) deserves an arm, and it is now unblocked on two independent corpora |
| 2 | **Price the three routes** — which of comma2k19 / L2D / nuScenes-CAN is actually in hand, and under what licence | DataFlyWheel | hours | step 1 above needs exactly one of them; picking wrong costs a download, not an arm |
| 3 | **`turn_signal` as a discrete command channel** — the only non-path-derivable intent signal found, and the shape Codevilla's branched conditioning needs | Data Eng + Arch | design | ⚠️ depends on 1; and on a corpus with an open GDPR gate for frames |
| 4 | **Action-space coverage as a third P2 lever** — expert-only data restricts the state-action space (ReSim); L2D's 13.8 % STUDENT split is a native non-expert source | Arch + Data Eng | design | a *data* answer to an *architecture* problem; unpriced |
| 5 | Road-class / >14 m/s stratum sourcing from L2D natives, as a **separate** corpus with its own parity key | DataFlyWheel | days | fills two B1 open items; ⛔ confounds corpus with regime, so it cannot answer "does high-speed data help" |

## What was deliberately NOT done

- ⛔ **Nothing downloaded.** No L2D bytes, no nuScenes bytes, no HF metered call.
- ⛔ **No probe built.** The correlation script is *specified* in RESULT §6 and left unbuilt —
  it belongs to the DataFlyWheel, and writing it here would be the integration the operating
  standard says to escalate rather than perform.
- ⛔ **No parity touched, no episode re-selected, no corpus mixed.**
- ⛔ The GDPR gate at `LOOP_STATE.md:1025` was **reported, not resolved**. It is not mine.
