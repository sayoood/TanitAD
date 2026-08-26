# Next steps — ordered against the PI's staged plan

**Rewritten** 2026-08-26 after the PI restated the goal: *"find the architecture and
training recipe for v7 and prepare the scaled training"*, and the sequence: **show the
OPERATIVE capability first; in PARALLEL generate the tactical and strategic data (Data
FlyWheel); then train the rest; then show the dominance of the hierarchical reasoning.**

⭐ Rationale and evidence for every choice below: `Project Steering/V7_RECIPE_AND_SCALEUP.md`.
⚠️ This is a **PRIORITY ORDER, NOT A DEPENDENCY CHAIN** — a blocked item never licenses idling.

---

## STAGE A — operative capability (the near-term deliverable)

| # | item | state | blocker |
|---|---|---|---|
| **A1** | **D1 — the crossed cell `postrain30k_freeze`.** Does freezing stop drift *without* wrecking prediction? Pre-registered: **drift < 0.45 AND held-out nrmse ≤ 0.893**, DEGENERATE a live branch | ▶ **RUNNING 24,000/30,000** | — |
| **A2** | **First T1 on a v7 arm, parity corpus, both arms.** ⛔ The programme has NEVER evaluated a v7 arm at T1, so it holds **no** driving claim | 🔶 chain armed on Thor, fires on A1's done-marker | A1 |
| **A3** | **D2 — `--cond-param omega_accel_v`** ([yaw_rate, a_long, v] as measured state, the PI's directive). Implemented, 7 tests green, **never run as an arm**. ⚠️ Tests whether a channel that is *not* a restatement of the realised trajectory behaves differently — the confound E-DEC-57 exposed | ⏸ queued | Thor busy (A1) |
| **A4** | **Distance-keeping** — the half-family carrying 88.7 % of the oracle gap. Engineering **DONE** (`b6e98043a`: the PI's straight-driving gate, `NOT_STRAIGHT` state, 10 tests) | ⛔ **DATA-BLOCKED** | `obstacle.offline` parquet not staged on Thor or dev box → **A5** |
| **A5** | **Stage `obstacle.offline` for the parity clips** | ⏸ not started | **Data FlyWheel** |
| **A6** | **Strategic family** — needs map-derived option sets (`strategic_gt.py` → `taniteval.strategic_optionset`). ⚠️ A route label read off the ego's own future yaw is **NOT** a substitute — it cannot tell whether the map admitted a choice | ⏸ not started | Stage B data |

---

## STAGE B — the data the upper levels need (Data FlyWheel; runs in PARALLEL, not after)

| # | item | why it is not gated on Stage A |
|---|---|---|
| **B1** | **Tactical labels** — manoeuvre classes, selected-vs-executed, goal/anchor selection | Label derivation may use ego, other agents, maps, future poses (PI, 2026-08-03). None of it needs a trained arm |
| **B2** | **Strategic labels** — map-derived option sets, route/goal. ⛔ PhysicalAI-AV has **no map, lane graph, junction annotation, traffic-light feature or route signal** (card, verbatim: *"we do not include open maps data"*), and `egomotion` carries **no lat/lon**, so OSM map-matching on our traces is impossible ⇒ the topology must come from **AlpaSim** (`map.xodr`) or an external corpus | This is the long pole for Stage D and the reason to start it now |
| **B3** | **`obstacle.offline` staging** (= A5) — unblocks distance-keeping immediately | Pure provisioning |

⚠️ **B2 is the critical path to the PI's thesis** and nothing in Stage A shortens it.

---

## STAGE C — train the upper levels

Gated on B1/B2. Recipe carried from `V7_RECIPE_AND_SCALEUP.md` §5.1 plus whatever D1/D2 settle.

---

## STAGE D — hierarchy dominance over the references

| # | item | state |
|---|---|---|
| **D-a** | **REF-C** baseline | ✅ trained — `refc-diffusion-xl-30k`, complete at 29,999, scored |
| **D-b** | **REF-D** | ⛔ **does not exist** — 0 registry rows. **Needs a PI definition of what REF-D is** |
| **D-c** | **Hierarchy-traversing eval** | ⛔ does not exist. `four_families` states it verbatim: *a world-model FIDELITY pass does not traverse the hierarchy* — which is why the strategic family reads UNAVAILABLE |
| **D-d** | **Frozen-DINOv3 WM** as reference/fallback (PI directive) | 🔶 partial — `dinofrozen30k` was pulled from the queue after C156 and should be restored, since DINOv3 currently **beats our trained encoder** on free space (+0.3701 vs +0.2869) and side occupancy (+0.2735 vs +0.1325) |

---

## 0-GPU work, pullable any time (gated ≠ idle)

- The **D1-negative contingency**: if freezing comes back degenerate, specify the
  partial/staged unfreeze and the **content-preserving auxiliary that is NOT
  self-generated** — the single property every one of the ten failed objectives shared
  (E-DEC-7).
- **Ablate what actually distinguishes `splitp30k` from `postrain30k`** with the init held
  fixed. C164 killed the attribution but the **0.47 effect is real and seed-stable
  (~1.5 % run-to-run)**; we currently have the result with no explanation.
- Restore `dinofrozen30k` to the queue (D-d).
- Re-read older §13 cells at t 2–3 against the **measured null (|t| ≈ 2.9)**.
