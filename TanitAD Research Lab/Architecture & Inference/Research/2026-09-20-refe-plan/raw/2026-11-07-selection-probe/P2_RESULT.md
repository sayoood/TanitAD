# P2 RESULT -- pre-registered re-pick rule gBG(0.20, 0.05) for REFe u400 (stream YSEL, 2026-11-07)

**Committed verdict: NOT PROVEN** (mechanical, `PREREG_SEL.md` §2, sha256 `e00fa28a7a5d6352b6f5a1782b34966bd75d88ad6b99ce7a614264581db0b7f0`,
frozen 2026-11-07 18:46 +01:00 before any held-out PDMS was computed; `p2_score.py` refuses to run on any other hash).
**Tier T0** open loop. **Held-out:** 62 logs / 505 tokens (236 zeroed, 269 random) / 68 CRUISE, log-disjoint from TRAIN.
**Estimator:** paired log-cluster bootstrap over the 62 held-out logs, 10,000 draws, seed 20260927, 95 % percentile — answers
"another draw of logs" only (one checkpoint, deterministic planner). Artifact: `p2_result.json`. All numbers MEASURED.

## Gates (all PASS before the verdict was read)
G0 pick reproduction 1058/1058 · G0b rule at δ = 0 reproduces 1058/1058 · G0c literal per-token loop of PREREG §1 == vectorised rule
1058/1058, and its **RED arm (guard sign flipped) disagrees on 175/1058** (the check can see a defect) · vectorised == `sel_rules.pick_BG` gated 1058/1058.

## The decision
| condition | bar | measured | holds |
|---|---|---|---|
| (i) PDMS non-inferior (full-navtest mix, x100) | lower CI > −0.5 | **+0.471 [−0.622, +1.512]** | **NO** (lower bound 0.12 short) |
| (ii) CRUISE slow-down share drop | ≥ 10 pts and CI-hi < 0 | **−38.2 pts [−56.9, −22.1]** (0.588 → 0.206) | YES |
| harm shown (REFUTED branch) | point < −0.5 and hi < 0 | no | — |

⇒ **NOT PROVEN.** The rule removes about two-thirds of the planned CRUISE slow-downs with a positive PDMS point estimate. The interval is
too wide to show non-inferiority at −0.5 on 62 logs. That is a **power** failure, not harm: 54 of the 110 changed picks are better by
true PDMS and 2 are worse.

## Every family (held-out; banked → rule; paired Δ with CI)
| family | metric | banked | rule | paired Δ |
|---|---|---|---|---|
| **LONGITUDINAL** | plan v4 − GT v4, CRUISE (m/s) | −1.355 [−2.055, −0.740] | **−0.402 [−0.966, +0.086]** | +0.953 [+0.713, +1.272] |
| | plan v4 − GT v4, ego gate set (n=145) | −1.147 | −0.156 [−0.559, +0.215] | +0.992 [+0.829, +1.196] |
| | slow-down share (v4/v0 < 0.85), CRUISE | 0.588 | 0.206 | −0.382 [−0.569, −0.221] |
| | overspeed share (v4/v0 > 1.15), CRUISE | 0.074 | 0.074 | 0.000 [0, 0] |
| | mean plan v4/v0, CRUISE | 0.817 | 0.930 | — |
| **LATERAL** (what the proposals allow) | ADE to human future, CRUISE (m) | 1.681 | — | **−0.510 [−0.745, −0.315]** |
| | \|y\| at 4 s on changed picks (m) | — | — | +0.494 [+0.317, +0.682] (the rule's proposals sit wider laterally) |
| | DAC failure rate | 0.337 | 0.323 | −0.014 [−0.029, −0.002] |
| | DDC mean | 0.972 | 0.964 | — |
| **safety** | NC fail / TTC fail rate | 0.137 / 0.176 | 0.135 / 0.168 | −0.002 [−0.010, +0.004] / −0.008 [−0.019, +0.002] |
| **PDMS** | mix x100 · random · zeroed | 88.35 · 92.37 · 0.00 | 88.82 · 92.69 · 3.86 | +0.47 [−0.62, +1.51] · +0.32 [−0.84, +1.39] · +3.86 [+1.61, +6.87] |
| **TACTICAL** | — | n/a: the C3 table carries no manoeuvre-decision label and REFe emits no tactical output here (n = 0) | | |
| **STRATEGIC** | — | **PAUSED** (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` item 25) | | |

Secondary rules on the same held-out logs, **DESCRIPTIVE only — none can change the verdict, none is promoted** (they were not the
pre-registered primary; reading several on one held-out set is exactly how a probe manufactures a result):
Sg(1) ΔPDMS −0.23 [−1.30, +1.14], slow −33.8 pts · Bg(0.2) −0.45 [−1.60, +0.84], slow −47.1 pts · uBG(0.2,0.05) −0.36 [−2.02, +1.27],
slow −38.2 pts · L(0.25) +1.03 [+0.67, +1.53], slow −1.5 pts · E(0.5) +0.11 [−0.28, +0.41], slow 0.0 pts.
⚠️ L(0.25) (lane-head temper) is the only arm with a separated PDMS gain, but it does **not** touch the slow-down; it is a post-hoc
held-out observation and needs its own pre-registration and a fresh set before it is quoted as a lever.

## Sim spec (for the sim session's RefePolicy) and the sim token
**Token `41d7b533797c5209`** (TRAIN log — spec check, not evidence): gate ON (v0 5.37, ax0 −0.10); the rule **changes the pick 37 → 7**,
v4/v0 **0.43 → 0.63**, aggregate 0.923 → 0.795, true PDMS 0.911 → 0.954; 40 proposals admitted. **It is a partial fix only:** the four
speed-holding proposals (true PDMS 1.000) are excluded by the band (36–65 % of the pick's aggregate) and by the guard (their predicted
TTC is 0.16–0.66 below the pick's). On this token the slow-down is carried by the TTC/NC heads, which the guard deliberately trusts.

**Exact rule (per planning cycle, after `model(img, ego_vec, goal)` returns `traj` [64, 20, 3] (0.2 s grid, ego frame) and `score` [64, 7]):**
```
p     = sigmoid(score)                                  # order NC, DAC, EP, TTC, C, DDC, lane
agg   = p0*p1*(5*p2 + 5*p3 + 2*p4)/12 * p6              # navsim_v1_lane, planner.py:552-556
b     = argmax(agg)                                     # the banked pick (planner.py:583)
v0    = measured ego speed now (m/s); ax0 = measured longitudinal accel now (m/s^2)
if not (v0 > 5.0 and abs(ax0) < 0.3): return b
v4_k  = ||traj[k,19,:2] - traj[k,18,:2]|| / 0.2          # raw proposal, before any heading repair
s_k   = min(1, (v4_k + 1) / (v0 + 1))
K     = {k : agg_k >= 0.80*agg_b  and  p_k[j] >= p_b[j] - 0.05 for j in (NC=0, DAC=1, TTC=3)}
return argmax_{k in K} (s_k + 1e-6*agg_k)               # b is always in K
```
Reference implementation: `p2_score.py::rule_vec` / `rule_loop` (bit-identical on 1,058 tokens). ⚠️ Given NOT PROVEN, the sim may run it
only as a **labelled experimental arm** beside the banked rule, never as the default — and on the gate's own definition it switches
itself off once the car is below 5 m/s, which is where the sim's crawl ends up.

## Rule Zero — the next levers, ranked by measured effect
1. **Fresh-token confirmation of the FROZEN gBG(0.20, 0.05)** — the failing condition is interval width (point +0.47, lower −0.62).
   Needs u400 64-proposal + logit dumps on navtest tokens outside the C3 1,058 (GPU inference) and harness scoring of only the changed
   picks (~22 % of tokens; CPU, `eval/score_navtest_refe.py`). **BLOCKED: compute — this brief was CPU-only.** ~2–4× the logs ⇒ a
   ~±0.5–0.75 half-width. Unblock: MM/PI authorise the dump.
2. **Speed-covariate recalibration of NC/DAC/EP/TTC** (`p3_recal_screen.py`, TRAIN-only 5-fold log CV): **REFUTED AT SCREEN** —
   slow-down share 0.267 → 0.267 (gated and ungated), ΔPDMS +0.01 / −0.08; fitted speed slopes are small (TTC +0.20, DAC +0.15 per
   unit v4/v0). A global speed term does not capture a within-token bias.
3. **Root cause = the scorer's speed pessimism (P1 Q4; the sim token's TTC/NC).** The durable fix is training-side: speed-balanced
   TTC/NC targets or hard pairs (slow-unsafe vs fast-safe) for the scorer. **BLOCKED: GPU + PI authorisation** (an architecture/training
   change). Selection alone cannot recover the four PDMS-1.000 proposals on the sim token without overriding the TTC head.
