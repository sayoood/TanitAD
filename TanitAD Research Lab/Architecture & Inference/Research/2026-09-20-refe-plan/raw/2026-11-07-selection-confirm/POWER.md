# POWER -- token plan for the fresh-token confirmation of H-SEL-1 (stream YSEL2, 2026-11-07)

**What is being powered:** the FROZEN rule gBG(0.20, 0.05) of `PREREG_SEL.md` (sha256 `e00fa28a...`). The bars are unchanged:
- (i) dPDMS mix lower CI > -0.5 x100;
- (ii) the CRUISE slow-down share drops by at least 10 points, with CI-hi < 0.

**Sources and evidence class.** Every number is MEASURED on banked data:
- C3 `table.npz` (1,058 tokens, TRUE per-proposal PDMS);
- the ego probe `full/results.npz`;
- YSEL `split.json`.

No fresh token was used. Scripts: `selc_power.py` -> `power.json` (normal model), `selc_power_sim.py` -> `power_sim.json` (heavy-tail simulation).

## 1. The fact that shapes the whole plan: a fresh token is never zeroed
- All **529/529** zeroed u400 tokens are already C3 tokens. That means banked PDMS = 0, NC x DAC = 0, asserted by `make_confirm_tokens.py`.
- So any token outside C3 is non-zeroed, and a fresh set can only measure the **random-stratum** effect.
- The prereg therefore reads the full-navtest mix with the zeroed-stratum delta set to its **structural floor 0**: `mix = W_zero*0 + W_random*delta_random`, with W_random = 11617/12146 = 0.9564.
  - This floor is an identity, not an estimate. The banked PDMS there is exactly 0 and PDMS >= 0, so the true zeroed delta is >= 0.
  - This can only make the statistic conservative.
  - P2's held-out zeroed delta, +3.86 x100, is **not** borrowed.
- ⇒ **The effect to power against is NOT P2's +0.47.** That figure includes +0.168 from the zeroed stratum. The relevant effect is the random-stratum delta:

| source (MEASURED, `power.json` `variance_components`) | n tokens / logs | mean d x100 | s2w | s2b |
|---|---|---|---|---|
| held-out random stratum (P2's own) | 269 / 60 | **+0.317** | 60.24 | 0.219 |
| TRAIN random stratum (selected on, optimistic) | 260 / 57 | **-0.225** | 42.21 | 0 (MoM < 0) |
| all C3 random stratum | 529 / 117 | **+0.051** | 51.36 | 0 |

Per-token deltas are **heavy-tailed**: about 85 % are exactly 0, and the minimum is -100 x100 (one pick change into a collision). So the plan is checked by simulation as well as by the normal model.
- On P2's own held-out allocation, the model SE is 0.482, while P2's bootstrap SD reads 0.549.
- The normal model is therefore about 14 % optimistic on 269 tokens. That gap is why the simulation exists.

## 2. Design: a CENSUS of the non-TRAIN fresh pool, not a sample
| fresh pool (non-zeroed, not in C3) | tokens | logs | used? |
|---|---|---|---|
| YSEL held-out logs | 5,580 | 62 | **YES** |
| navtest logs C3 never touched | 299 | 13 | **YES** |
| YSEL TRAIN logs | 5,209 | 61 | **NO** (leakage, section 3) |

- **Population: all 5,877 tokens** (5,580 in held-out logs + 297 in never-in-C3 logs; 5,879 minus the 2 seen in plumbing, see `PREREG_SEL_CONFIRM.md` §0), over 75 logs.
- `tokens_confirm.json`, sha256 `013b67718fbe057aec158dbf926d2abf697004caa5a2bbe75e43e46bd7600451`.
- There are no sampling choices to defend.
- Per-log counts: median 70, maximum 568.

| design (uniform draw from the non-TRAIN pool) | half-width of the mix CI (x100, normal model, held-out variance) | P(lower > -0.5), true random d = 0 | same, true = +0.317 (P2 held-out) |
|---|---|---|---|
| 1,000 tokens | 0.48 | 0.53 | 0.90 |
| 2,000 | 0.36 | 0.78 | 0.99 |
| 3,000 | 0.30 | 0.90 | 1.00 |
| **5,877 (census, chosen)** | **0.24** | **0.98** | **1.00** |

**Heavy-tail simulation of the census** (`power_sim.json`):
- Setup: 400 simulated reads on the actual 75-log allocation. Per-token deltas are resampled from C3's 529 measured random-stratum deltas, plus a log effect N(0, 0.219). Each read uses a 1,000-draw log-cluster bootstrap.
- **Half-width ≈ 0.21 x100.**
- P(lower CI > -0.5):

| assumed true random-stratum effect | P(lower CI > -0.5) |
|---|---|
| 0 | **0.98** |
| +0.051 (pooled) | 1.00 |
| +0.317 (held-out) | 1.00 |

⇒ **Condition (i) is well powered**, even if the rule's true PDMS effect is exactly zero.

**⚠ Condition (ii) is where the risk sits, and the bar is frozen.**
- On NON-ZEROED CRUISE tokens, which are the only kind a fresh token can be, C3 measured slow-down share drops of:

| C3 population (non-zeroed CRUISE) | drop in slow-down share | n | P(condition ii) on about 700 fresh CRUISE tokens |
|---|---|---|---|
| held-out | -39.4 pts | 33 | 1.00 |
| TRAIN | -6.7 pts | 30 | **0.25** |
| pooled | -23.8 pts | 63 | 0.99 |

- If the TRAIN-random figure is the truth, the rule's slow-down effect on the non-zeroed population is too small to clear the 10-point bar. The verdict would then be NOT PROVEN through (ii), not through (i).
- That is a substantive possible finding, not a power artefact, and it is committed in the prereg's outcome table.

## 3. Leakage: what reusing TRAIN or held-out logs would cost, and the bound
- **TRAIN logs (excluded).**
  - δ, ε and the gBG family were chosen on the tokens of these 61 logs.
  - Navtest tokens in one log are frames of the same drive: same route, map area, weather, traffic and driver. Neighbouring tokens' 4 s futures overlap.
  - Tuning optimism therefore transfers through the log. The TRAIN pick was selected on, at a -0.111 TRAIN mix.
  - Excluding the 61 logs makes **tuning leakage zero by construction**. It costs 5,209 tokens, and the census above does not need them.
- **Held-out logs (used).**
  - The rule was never tuned on them, so there is no tuning leak.
  - The residual risk is a **shared-log correlation with P2's read**. This confirmation exists because P2's point was positive (winner's curse). Fresh tokens in the same logs could therefore inherit P2's luck through the log effect.
  - **Bound (MEASURED variance components):**
    - s2b = 0.219 against s2w = 60.24 (x100², held-out random stratum), so ICC = 0.0036.
    - With P2's ~4.3 random-stratum tokens per log, the best linear predictor carries λ = s2b / (s2b + s2w/4.3) ≈ **0.016** of P2's per-log deviation into the fresh logs.
    - Even if P2's whole +0.317 were luck, the inherited bias is ≈ **+0.005 x100**. That is 2 % of the 0.21 half-width.
  - **Direct check (DESCRIPTIVE, pre-declared):** the result is also reported separately on the 13 never-in-C3 logs (297 tokens). They carry no shared-log exposure at all, but they are too few to decide anything (half-width ~±1).
- **The 2 plumbing tokens** whose outcome was seen before the freeze are excluded (`PREREG_SEL_CONFIRM.md` §0).

## 4. Compute plan
⚠ **The 64-proposal table is NOT on the verdict's critical path.** The verdict needs only two harness columns per token: the banked pick and the rule pick. They are the same harness, and C3's G2 showed that a column reproduces the per-token score bit for bit, independent of the token set. The full table follows, as the secondary H-SEL-LANE-1 read and the descriptive families.

| stage | measured basis | estimate (ESTIMATED) |
|---|---|---|
| GPU dump | C3 dump: 2,233 s for 1,058 tokens / 123 logs = **2.11 s/token** (`c3 .../logs/1_dump.log`) | 5,877 tokens ≈ **3.4 h**, in 7 segments of ≤ 1,000 tokens; lock released between segments; plus lock waits |
| smoke scoring | 65 calls. A 2-token call took 25 s in C3 (`refe_c3_u400_smoke_p00`), and 34–56 s per call (3 in parallel) here today | 11–20 min |
| verdict columns | 14 calls (2 arms x 7 chunks). C3: 112–155 s per 1,058-token call at 4 workers | 10–15 min |
| 64-column table | **448 calls** (64 x 7 chunks), 3 workers | **4.6–6.2 h** |

- **Total ≈ 8.5–10.5 h plus GPU lock waits.**
- The **verdict lands at about 4 h plus waits**, well inside the ~12 h envelope.
- The table may run past 12 h if the GPU queue (YAXN, desktop VRAM at the 3,500 MiB bar) adds more than about 1.5 h.
- Why the token list is chunked: W3's harness passes its tokens on the Hydra command line. C3 fitted 1,058, and Windows caps a command line at 32,767 characters. Hence 7 whole-log chunks of ≤ 1,000 tokens each.
