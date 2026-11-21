# COMMS — `2026-08-30-anti-echo-control-precedent`

## Integration status: ⛔ ESCALATED — register row NOT applied by me

`Project Steering/GOALS_AND_CLAIMS.md` moved under this run (modified 09:15:06). Paste-ready
delta below. **Owner to apply: TanitAD Master Mind** (this one also lands in the paper, which
the Master Mind maintains).

---

## Delta — REGISTER the methodological claim, scoped so it survives review

| id | claim | status | evidence |
|---|---|---|---|
| `D-ANTIECHO-PRECEDENT` | ⭐ **Our anti-echo control suite is stricter than published practice — but ONLY on the ACTION SIDE, and the claim must be written that way.** ⛔ **NOT novel:** *"open-loop is not driving performance"* is established and repeatedly re-established — `1809.04843` (2018) · `2306.07962` (2023) · `2505.05638` (2025) · `2605.00066` (2026); claiming it invites a correct reviewer objection. ⭐ **Novel at five probes:** (a) a **HOLD-ACTION floor the model must beat** — NOT FOUND in any driving world model; (b) a **GT-calibrated echo METRIC** (`echo_index` 0.0000 vs GT 0.2113) — the copycat concept is named (`2010.14876`) and fixed architecturally (`2207.09705`) but **no diagnostic metric is published**. (c) **HOLD-V0 is WELL precedented** — NAVSIM ships a `ConstantVelocityAgent` (`2406.15349`) — ⛔ **claim no novelty there**; the claimable part is that we run it on *every* closed-loop number. ⛔ **Both purpose-built action-fidelity benchmarks were verified at source to run NO control arm**: ACT-Bench (`2412.05337`, IEC + ADE/FDE, no frozen/shuffled action, no floor, no ceiling) and WorldLens (`2512.10958`, 24 dimensions incl. Action-Following, no null/adversarial baseline). | **PUBLISHED — supported, scoped** | ⭐ The field's own strongest warrant for our doctrine is someone else's number: **AD-MLP — ego-state-only, no camera, no LiDAR, near-SOTA open-loop — scores DS 18.05 / Success Rate 0.00 % closed-loop** (`2406.03877` Tab. 3, VERIFIED). Presentation template for our echo result: `2312.03031`'s ego-status ablation ladder (UniAD **1.03 m** with no ego status → 0.66 → 0.46). ⭐ `2511.20325` (AD-R1) documents the **dual** defect — WMs hallucinate a safe future when conditioned on an *unsafe* trajectory — narratively, with no metric; it corroborates `H-ARCH-ACTINS`. The 2026 position papers (`2606.15032`) **call for** interventional action fidelity as the decisive evidence class with no experiments — i.e. they ask for the instrument we already built. · `…/2026-08-30-anti-echo-control-precedent/RESULT.md` |

## Two guards this run asks to be made standing rules

1. ⚠️ **ESTIMATOR GUARD on `2605.00066`.** Its headline is **ρ = −0.36, n = 8, p = 0.43**. That
   establishes **ABSENCE of correlation, NOT negative correlation.** Quoting ρ bare — as though
   open-loop *anti-predicts* closed-loop — is precisely the slip our own operating standard
   forbids. **Always carry n and p.** (Its NAVSIM PDMS ρ = 0.90, p = 0.002 comes with ranking
   inversions.)
2. ⛔ **COMPARABILITY GUARD.** No table may place a TanitAD ADE/FDE beside a PDMS / DS / EPDMS
   score. Our T1 numbers are **metres on our own 40-episode PhysicalAI split**; camera-only
   closed-loop SOTA is **SimLingo 85.07 DS / 67.27 % SR** on Bench2Drive (`2503.09594` Tab. 2) —
   different corpus, metric family and simulator regime. Cross-programme comparability requires
   running a community benchmark (the standing TanitEval work item), not a prose bridge.
   ⚠️ **Related scope trap, live:** NAVSIM v2 EPDMS is quoted in the wild as **87.1 / 88.6 /
   36.9** — these **cannot be one quantity** (navhard vs navtest). **Name the split or do not
   quote.**

## Decision asked

**Adopt the two guards above as standing criteria** in the `TanitAD_BenchmarkCriteria` registry
(they are cheap, mechanical, and each prevents a class of error we have already paid for in
another costume), and **write the paper's methodology claim as action-side only.**

## ⚠️ Not banked, not quotable

`2601.01528` (DrivingGen — whether it carries a null control is UNCONFIRMED), `2603.12864`
(reported action-branch-removal ablation — read from a search summary only, and in any case a
*generator* ablation, not a baseline to beat), `2604.22748` (position paper). **PARA-Drive has
no arXiv ID** — CVPR 2024 open-access only; bank via `kb_add.py --local` if ever cited.
