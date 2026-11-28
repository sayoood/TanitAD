# REFe: plan to reach navtest PDMS ≥ 94.55 (MM, 2026-11-26)

**State.**
- The best proven REFe is still **C1, 92.55 [91.80, 93.26]** (MODEL_REGISTRY 14.6).
- The REFe pod has ended (PI, 2026-11-26).
- SCORER-FT arm P reads **preliminary −2.55 PDMS vs C**: harness CSV, T1 tracked, 12,146 tokens, no CI yet. The card's read lands separately.

Every number below is **MEASURED, T1 (NAVSIM v1.1, LQR-tracked)**, on all 12,146 navtest tokens, from the 64-proposal tables `scorerft_read/propscore/full_{SFTC,SFTT}_s*of2.jsonl` (sha256 below). It is **descriptive**. ⚠️ **Navtest is a SNOOPED population for these numbers: no rule may be fitted on it.** Every lever below is fitted on navtrain held-out logs, or is parameter-free, and is then read once on navtest under a frozen card.

## 1. Where the 1.9 points are

### 1.1 The generator is not the bottleneck; selection is

| (C arm) | PDMS |
|---|---|
| random proposal | 68.25 |
| **scorer pick** | **92.67** |
| **oracle best-of-64** (the selection ceiling of this generator) | **99.35** |

- On 97.8 % of tokens the generator already offers a proposal with PDMS ≥ 0.95.
- The goal needs about **28 % of the 6.67-point selection regret**.
- This matches `D-REFE-HEADATTR-1` (C1: 6.80).

### 1.2 The regret has two terms, and each alone would clear the goal

Bounds below are oracle-informed and not achievable; they size the levers.

| cause (first sub-score in which the pick is worse than the oracle proposal) | tokens | regret | bound if fixed |
|---|---|---|---|
| **safety miss**: the pick fails NC / DAC / TTC / DDC while a passing proposal exists | 919 (7.6 %) | **4.20** (DAC 1.54, TTC 1.43, NC 1.08, DDC 0.15) | **96.88** |
| **progress miss**: the pick is safe but slower | 5,392 | **2.47** | **95.14** |

Of the safety misses:
- **about 1/3 are TRACKER-INDUCED**, carrying 1.31 pts: the pick passes the term as planned and fails only after NAVSIM's LQR re-simulation (DAC 73 tokens, TTC 128, NC 30);
- **about 2/3 are unsafe as planned**, carrying about 2.9 pts: a head-boundary error.

This matches `D-REFE-HEADATTR-1`: the safety heads have AUC 0.90–0.96 on average but are confidently wrong at the boundary.

### 1.3 Why everything tried so far failed: progress was bought with safety

| lever | read | mechanism (same tables) |
|---|---|---|
| S-RULE head reweighting (`D-REFE-SRULE-1`) | Δ −0.17 separated worse | EP −0.67 |
| learned re-rankers over frozen outputs + geometry (`D-REFE-SRERANK-1`) | no signal | switches on 0.2 % of tokens |
| SCORER-FT T (rank loss through the trunk) | −0.90 separated | **EP regret 2.47 → 1.59, safety regret 4.20 → 6.01** |
| SCORER-FT P (pick-level rank loss) | −2.55 preliminary | EP +0.64, **TTC −4.07, DAC −1.71, NC −1.27** |

⇒ The progress head ranks poorly: Spearman 0.41, against 0.83 for plain path length. Every push toward progress therefore selects longer, boundary-hugging proposals that the safety heads cannot reject.

> ⛔ **The binding constraint is SAFETY-BOUNDARY DISCRIMINATION. Progress may only be pushed after that is fixed.**

The teacher shows the same thin margin (`D-REFE-TEACHER-NAVTEST-1`): 0.37 m in turns, against the human's 0.60 m. 87 % of its DAC failures are tracker-induced.

## 2. Levers, ranked by measured effect and cost

| # | lever | targets | bound | cost / compute | blocker |
|---|---|---|---|---|---|
| **L1** | **TRK-SCORE: tracker-consistent selection.** Run NAVSIM's LQR tracker on each of the 64 proposals from the ego state, then score the TRACKED set with the existing heads (`model.py:699` `score_trajectories` takes any trajectory set) and pick on that. The submitted trajectory stays the plan. | tracker-induced safety misses | 1.31 pts | **parameter-free**; ~1 day of engineering + one dev-box navtest seam (~6–8 h, RTX 4060) + read. **No pod.** | none: **start now** |
| **L2** | **SAFE-HEAD: boundary-focused safety-head training, DETACHED from the trunk.** Hinge/focal loss on NC/DAC/TTC near the boundary, on on-policy navtrain proposals with TRACKED harness labels; EP head untouched. This is the card's own `P-detach` direction, with a safety-first target. | unsafe-as-planned misses | 2.9 pts | ~15 L40-h, 1,000 steps on C1 (ESTIMATED), + seam + read | **a GPU** (PI: no pod now) |
| **L3** | **EP-GEOM: path-length EP surrogate, only inside the confidently-safe set** (the set from L1/L2). The confidence threshold is fitted on navtrain held-out. | progress misses | 2.47 pts | CPU on tables, + one seam | after L1/L2 (the T/P lesson) |
| **L4** | **TEACHER margin (standing TF stream, PI 2026-11-11).** Raise the teacher's turn margin from 0.37 m to ≥ 0.60 m, relabel, retrain the C1 recipe. | makes proposals less boundary-hugging | not sized | CPU rollouts + ~30 L40-h | a GPU |
| L5 | seed hygiene | — | — | F_REFe = 0.12 (one draw): effects below ~0.25 are not readable; 21/70 family components separate by seed alone | — |

**Path to the goal (ESTIMATED, not a promise).**
- L1 captures part of 1.31 pts; L2 part of 2.9 pts.
- Together, about 1.9 pts is the goal; L3 then adds on top.
- Each lever runs its own pre-registered card, with both outcomes committed, a replicate where it trains, and the four metric families.

## 3. Execution order
1. **Now (no GPU spend):**
   - L1 card + implementation + dev-box seam.
   - P's card read (running) → H-SCORERFT-2 verdict.
   - L3 design on navtrain tables, if the banked on-policy sets carry per-proposal labels.
2. **PI decision:** one GPU for L2. The cheapest option is a short pod (~15–20 h). The other is a dev-box head-only variant on cached scoring contexts, which needs a feasibility check: memory, and the GPU lock shared with other sessions.
3. **After L1/L2:** L3, then L4 under the TF stream.

## 4. Provenance
- Instruments (this folder): `oracle_decomp.py`, `safety_miss_split.py`; outputs `oracle_decomp_C_T.out`, `safety_miss_split_C_T.out`.
- Table sha256:

  | table | sha256 |
  |---|---|
  | SFTC s0 | 8d436f45… |
  | SFTC s1 | 8a9846c6… |
  | SFTT s0 | 5f06b843… |
  | SFTT s1 | f6bd8ec8… |

  Full values are in `oracle_decomp_C_T.out`'s sibling `TABLES.sha256`.
- Prior diagnoses: `D-REFE-HEADATTR-1`, `D-REFE-SRULE-1`, `D-REFE-SRERANK-1`, `D-REFE-TEACHER-NAVTEST-1`, `D-REFE-TEACHER-E2E1-1`.
