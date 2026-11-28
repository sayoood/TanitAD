# PREREG_SEL -- a pre-registered RE-PICK rule for REFe u400's unjustified CRUISE slow-down (stream YSEL, 2026-11-07)

**Status: FROZEN before any held-out PDMS was computed.** Its sha256 is recorded in `PREREG_SEL.sha256`, and `p2_score.py`
refuses to run unless this file hashes to that value. Tier **T0** (open loop, one step, logged ego history). Zero GPU.
Selection only: the u400 scorer, its 64 proposals and its logits are FIXED (`table.npz` of
`raw/2026-10-11-c3-selection-headroom/`, 1,058 C3 navtest tokens, 123 logs).

## 0. What has been seen at freeze time (stated so nothing is hidden)
- **G0 PASS** (`g0_gate.json`, sha256 `00e6c577...`). The offline re-pick reproduces the banked `pick` on **1,058/1,058** tokens,
  and on the ego probe's A0 logits too. Its three RED arms fail as they must: no-lane 635/1058, logit-sum 334/1058, argmin 0/1058.
  The rule source is `refe/planner.py:551-556` (aggregate), `:583` (`_pick`), `:475` (`infer`).
- **TRAIN logs only:** `split.json` (sha256 `d3b5d57d...`), 61 logs, 553 tokens, 86 CRUISE.
  The split is seeded at 20261107 and stratified by has-CRUISE-token. On TRAIN I computed:
  - the P1 diagnosis, `p1_diag_train.json` (`3de83c29...`);
  - every rule family, in `p2_explore_train*.json`: `98671a17...`, `57c04b0e...`, `fcd2aac0...`, `45b24365...`.
- **Held-out logs:** 62 logs, 505 tokens, 68 CRUISE. **No PDMS, sub-score, or rule outcome has been computed on them.**
  - Prior knowledge of the full set comes from earlier programme reads, already banked: C3's oracle headroom and the ego probe's 41 % CRUISE slow-down share.
  - From those, the held-out baseline slow-down share is derivable at about 40/68. It is a speed descriptor, not a PDMS.
- The token the sim session uses, `41d7b533797c5209`, is in a **TRAIN** log. Its read is therefore a spec check, not held-out evidence.

## 1. The rule (PRIMARY, the only one that decides)

Notation, per token, over the 64 proposals k:

| symbol | definition |
|---|---|
| `p_k` | sigmoid of the 7 u400 logits, order NC, DAC, EP, TTC, C, DDC, lane |
| `agg_k` | `p0*p1*(5*p2 + 5*p3 + 2*p4)/12 * p6`: the banked `navsim_v1_lane` aggregate |
| `b` | `argmax_k agg_k`: the banked pick |
| `v0` | measured ego speed now (`ego_real[:, 6]`) |
| `v4_k` | `|xy_k[19] - xy_k[18]| / 0.2 s`: the one-step chord speed at t = 4 s of raw proposal k (20 x 0.2 s grid) |
| `s_k` | `min(1, (v4_k + 1) / (v0 + 1))`: speed consistency; it never rewards going faster than now |

**Rule `gBG(delta = 0.20, eps = 0.05)`.**

1. **GATE.** If `v0 > 5.0 m/s` and `|ax0| < 0.3 m/s^2` (current measured ego only), apply steps 2 to 4. Otherwise return `b`.
2. **BAND.** `K = { k : agg_k >= (1 - 0.20) * agg_b }`.
3. **GUARD.** Remove from `K` every k that fails `p_k[j] >= p_b[j] - 0.05` for any j in {NC, DAC, TTC}.
4. **RE-PICK.** Return `argmax_{k in K} (s_k + 1e-6 * agg_k)`. That is the most speed-consistent admitted proposal, with ties broken by the higher aggregate. If `K` is empty, return `b`. `b` is always in `K`.

**Admissibility.**
- The rule reads only the model's own outputs (proposals and logits) and the current measured ego speed and acceleration. REFe's `ego_vec` already consumes these.
- It reads no future, no GT and no privileged channel.
- It has no situation-classifier output (the goal-input ruling does not apply).
- The gate copies the ego half of the CRUISE definition. The GT half, `|GT v4 - v0| < 1`, is inadmissible at inference and is NOT used.

**How it was chosen (TRAIN only).**
- I compared four families on TRAIN:
  - **S**: speed factor;
  - **L**: lane temper;
  - **E**: EP weight;
  - **V**: speed veto.
- I also compared their ego-gated and guarded variants, B / Bg / BG / gBG.
- Each one that cut the slow-down share also cost TRAIN PDMS (`p2_explore_train*.json`). The ungated speed factor S(1) cost -2.15 x100.
- The losses come from a few large TTC/DAC drops (`p2_explore_train_why.py`).
- **Primary family:** gBG. P1 (TRAIN) shows:
  - slow picks win on small aggregate margins, 13 % on the scorer's own fast alternative;
  - the margin is carried by the **lane head** (the dominant term on 13/19) and by safety heads whose predictions fall with speed while their truths are flat.
  - A rule that overrides the scorer only inside its own margin, and only where the scorer's own safety heads do not object, is the least invasive form.
- **Selection criterion**, written after viewing the TRAIN gBG table and applied to it mechanically:
  - maximise the TRAIN CRUISE slow-down drop, subject to TRAIN mix-weighted dPDMS >= -0.15 x100;
  - break ties to the smaller delta, then the smaller eps.
  - Result: gBG(0.2, 0.05), with TRAIN dPDMS **-0.111** x100 and slow-down share 0.267 -> **0.105**.
  - The TRAIN numbers are optimistic by construction (selected on), and are NOT evidence.

## 2. The single held-out read
- **Population:** the 62 held-out logs (505 tokens). Scored ONCE by `p2_score.py`.
- **Estimator:** paired log-cluster bootstrap over the held-out logs.
  - `rng = default_rng(20260927)`, `rng.choice(n_logs, (10000, n_logs), replace=True)`, percentile 95 %.
  - Every statistic, and both arms, use the SAME draws.
  - It answers *"another draw of logs"* only. One checkpoint, deterministic planner (ego probe C1). It says nothing about training or inference seed.
- **PDMS (x100)** is the full-navtest MIX: stratum means weighted 529/12146 (zeroed) and 11617/12146 (random), as in C3. The unweighted mean is reported beside it.
- **Slow-down share:** the share of held-out CRUISE tokens (`tokens_cruise.json` ∩ held-out) whose pick has `v4/v0 < 0.85`. This uses the ego probe's definition, unweighted, as in the ego probe.

### Committed outcomes (both written now)
| verdict | condition |
|---|---|
| **ADOPT-for-sim** | (i) dPDMS_mix (rule - banked) **lower CI > -0.5** x100 **AND** (ii) the CRUISE slow-down share drops by **>= 10 percentage points** (point estimate) **with the paired CI upper bound < 0** |
| **REFUTED** | the rule is shown harmful: dPDMS_mix point **< -0.5** AND upper CI **< 0**; **OR** the slow-down share does not fall (point drop **<= 0**) |
| **NOT PROVEN** | anything else |

## 3. Reported with it (DESCRIPTIVE, can never change the verdict)
- **LONGITUDINAL**
  - plan v4 minus GT v4, on held-out CRUISE and on the held-out gate set (gate set: v0 > 5, |ax0| < 0.3, GT valid);
  - mean plan v4/v0 on CRUISE;
  - the slow-down share;
  - **overspeed share**, plan v4/v0 > 1.15 on CRUISE (the failure the rule could create).
- **LATERAL**
  - what the proposals allow: the change in |y| at 4 s and in heading at 4 s between the re-pick and the banked pick;
  - ADE to the human future (8 x 0.5 s), banked vs re-pick;
  - DAC and DDC sub-score rates.
- **TACTICAL:** n/a. The table carries no manoeuvre-decision label and REFe has no tactical output here. Reported with that reason and n = 0.
- **STRATEGIC:** PAUSED (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` item 25).
- **Per-stratum PDMS**, NC/TTC/DAC failure rates, and the pick-changed share.
- **Secondary rules on held-out, DESCRIPTIVE ONLY:** Sg(1), Bg(0.2), L(0.25), E(0.5), and ungated BG(0.2, 0.05).
- **The sim token `41d7b533797c5209`:** does the rule change its pick, and to which index, v4/v0 and PDMS. It is in TRAIN, so this is a spec check.
- The P1 diagnosis re-run on ALL logs and on held-out only, as an appendix computed AFTER the held-out read.

## 4. Gates that must hold before the verdict is read
- **G0** (pick reproduction) re-asserted inside `p2_score.py`.
- **G0b:** the rule with `delta = 0` reproduces the banked pick on 1,058/1,058.
- **G0c (spec conformance):** a plain per-token loop that implements section 1 literally must equal the vectorised rule on 1,058/1,058 tokens.
  - Its **RED arm** is the vectorised rule with the guard sign flipped, `p_k >= p_b + eps`. That arm must DISAGREE with the loop on at least one token, or the check cannot see a defect.
  - Both arms run on all tokens. Only picks and logits are compared, no PDMS.
- The held-out mask is log-disjoint from TRAIN (asserted).
