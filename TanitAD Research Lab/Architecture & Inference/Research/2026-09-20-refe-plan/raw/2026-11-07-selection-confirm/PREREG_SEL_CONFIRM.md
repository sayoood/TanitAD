# PREREG_SEL_CONFIRM -- fresh-token confirmation of H-SEL-1's FROZEN re-pick rule gBG(0.20, 0.05) (stream YSEL2, 2026-11-07)

**Status: FROZEN before any token of the confirmation population was run through the model or scored.**
- Its sha256 is in `PREREG_SEL_CONFIRM.sha256`.
- `selc_score.py` (verdict and table stages) and `selc_analyze.py` (verdict and table reads) refuse to run unless this file hashes to that value.
- They also refuse unless every pinned file matches `CODE_PIN.json`, whose own sha256 is frozen here: `86e7b9fdb5a3479cfefdddc9cee906be1e48f8134666c8346b7b92a79284f3ea`.
- **No retuning:** the rule, its parameters and its bars are those of `PREREG_SEL.md` (sha256 `e00fa28a7a5d6352b6f5a1782b34966bd75d88ad6b99ce7a614264581db0b7f0`), unchanged.
- **Tier T0:** open loop, one step, logged ego history.

## 0. What has been seen at freeze time
- **Everything of YSEL:**
  - P1 on TRAIN, ALL and held-out;
  - the TRAIN rule exploration;
  - P2's single held-out read, NOT PROVEN, with ΔPDMS mix +0.471 [-0.622, +1.512] x100 and slow-down -38.2 pts [-56.9, -22.1] on 62 logs / 505 tokens;
  - P3.
- All of C3's 1,058 tokens, and the ego probe's 1,058 tokens.
- **Power analysis** (`POWER.md`, `power.json`, `power_sim.json`): computed on banked C3 data only.
- **Plumbing (CPU, before this freeze), disclosed in full:**
  - The 6 smoke C3 tokens were dumped on CPU.
  - So were 2 FRESH tokens, `019eaadd84cb56f8` and `335576eae9705bfc` (log `2021.08.30.13.45.25_veh-40_00784_00867`). Their banked and rule harness columns, and a 64-column table, were scored to test the code path.
  - Their outcome was seen: on the 2 tokens, gBG read ΔPDMS +2.43 x100 with 1 changed pick, and the 64-column plumbing table read L(0.25) at +0.69 x100.
  - ⇒ **Both tokens are EXCLUDED** from the population (`make_confirm_tokens.py` `EXCLUDE_SEEN`).
  - The superseded list that still held them is kept as `tokens_confirm.SUPERSEDED_plumbing_seen.json`.
- **No other fresh token has been through the model or the harness.**

## 1. The rule (PRIMARY; the only one that decides H-SEL-1)
- The rule is `PREREG_SEL.md` §1 verbatim: GATE (`v0 > 5.0` and `|ax0| < 0.3`), then BAND (`agg_k >= 0.80*agg_b`), then GUARD (`p_k[j] >= p_b[j] - 0.05` for NC, DAC, TTC), then RE-PICK (`argmax s_k + 1e-6*agg_k`).
- Inputs are only the model's own 64 raw proposals, the 7 scorer logits, and the current measured ego speed and acceleration.
- **Implementation:** `selc_rule.py`.
  - It is a copy of `p2_score.py::rule_vec` and `::rule_loop`, with globals turned into arguments.
  - **Gate G-RULE** (`selc_gates.py`) re-executes p2_score's functions from their source text. Copy and original must agree:
    - vectorised: 1058/1058;
    - loop: 1058/1058;
    - loop == vectorised: 1058/1058;
    - banked pick == C3 pick: 1058/1058;
    - δ = 0 reproduces the banked pick: 1058/1058.
  - Its **RED arm** is the guard sign flipped. It must disagree on at least 1 token. At freeze it disagreed on 175.

## 2. Population: a census, chosen before any of it was run
- **`tokens_confirm.json`, sha256 `013b67718fbe057aec158dbf926d2abf697004caa5a2bbe75e43e46bd7600451`.**
  - **5,877 tokens over 75 logs:** 62 YSEL held-out logs (5,580 tokens) and the 13 navtest logs C3 never touched (297 tokens; 299 before the §0 exclusion).
- The list contains every NAVSIM v1.1 navtest token that meets all four conditions:
  - it is NOT among the 1,058 C3 tokens;
  - it is non-zeroed in the banked u400 csv (this is implied, because all 529 zeroed tokens are C3 tokens; asserted);
  - its log is NOT one of YSEL's 61 TRAIN logs;
  - it is not one of the 2 tokens of §0.
- Smoke list: `tokens_smoke.json`, sha256 `8d510492d4b6fa36670679a9d87522c6ef3beb0427f1b11814679f3a827715fb`. It holds 6 C3 tokens, used for gates only, never for the verdict.
- **Leakage** (bounded in `POWER.md` §3):
  - TRAIN logs are excluded, so tuning leakage is zero by construction.
  - Held-out logs are reused without a tuning leak. Their shared-log correlation with P2's read is bounded at λ ≈ 0.016, which is ≈ +0.005 x100 of inherited bias.
- **Harness chunks:** whole logs, ≤ 1,000 tokens each, 7 chunks. This is only because the token list rides on the Hydra command line. It changes no number, since each token is scored independently, as C3's G2 showed.

## 3. Estimator (unchanged from PREREG_SEL §2, one structural difference stated)
- **Paired log-cluster bootstrap** over the 75 fresh logs:
  - `rng = default_rng(20260927)`, `rng.choice(75, (10000, 75), replace=True)`;
  - ratio estimator (sum over resampled logs / count);
  - 95 % percentile interval.
  - The SAME draws are used for every statistic and for both arms.
  - It answers *"another draw of logs"* only. There is one checkpoint (u400, sha256 `8b4dba48...`) and the planner is deterministic (ego probe C1), so the read says nothing about training seed or inference seed.
- **PDMS (x100), the zero-floor mix:** `100 * (W_zero*0 + W_random*Δ_random)`, with W = 529/12146 and 11617/12146.
  - No fresh token is zeroed, so the zeroed stratum enters at its **structural floor 0**. The banked PDMS there is exactly 0 and PDMS >= 0, so the true zeroed delta is >= 0. The statistic is therefore a lower bound on the full-navtest mix.
  - The unweighted mean is reported beside it.
- **Slow-down share:**
  - Computed over fresh CRUISE tokens, defined as in `refe_ego_probe_20261106/make_cruise_tokens.py`: gt_valid, `v0 > 5`, `|ax0| < 0.3` and `|GT v4 - v0| < 1`. GT is used only to define the label, never at inference.
  - A token counts as slowed when its pick has `v4/v0 < 0.85` (v4 = the one-step 0.2 s chord at t = 4 s of the raw proposal).
  - Unweighted.
- **Implementation:** `selc_analyze.py`, gate G-ANA. The same code, run on P2's 505 held-out C3 tokens, reproduces `p2_result.json`'s primary block **exactly** (max |diff| 0.0) and its verdict.
  - RED arms: swapping the arms flips the sign; banked vs banked reads exactly 0 with CI [0, 0].
  - The CRUISE formula reproduces `tokens_cruise.json` on 1058/1058.

## 4. Committed outcomes (PREREG_SEL §2, verbatim bars, both written now)
| verdict | condition |
|---|---|
| **ADOPT-for-sim** (H-SEL-1 CONFIRMED at T0) | (i) ΔPDMS zero-floor mix **lower CI > -0.5** x100 **AND** (ii) the CRUISE slow-down share drops by **>= 10 percentage points** (point estimate) **with the paired CI upper bound < 0** |
| **REFUTED** | the rule is shown harmful: ΔPDMS point **< -0.5** AND upper CI **< 0**; **OR** the slow-down share does not fall (point drop **<= 0**) |
| **NOT PROVEN** | anything else. This includes the slow clause being unmeasurable (n_CRUISE = 0), which is never a pass and never a refutation. |

**Precedence:** INSTRUMENT FAILURE comes first. If any gate of §6 is red, there is no verdict and no number from the read is quoted.

**What each outcome commits us to (Rule Zero; the next lever is named now):**
- **ADOPT-for-sim:**
  - H-SEL-1 becomes CONFIRMED (T0) in `GOALS_AND_CLAIMS.md`.
  - The sim session runs gBG as a labelled arm next to the banked rule, as the T1 capability test (spec: `P2_RESULT.md`). T0 confirmation is never a T1 claim.
  - The durable fix stays training-side: speed-balanced TTC/NC targets for the scorer (`D-REFE-SPEEDPESS-1`).
- **NOT PROVEN through (ii) with (i) passing:**
  - The rule is safe but does not remove enough slow-downs on the non-zeroed population.
  - Selection inside the scorer's own margin is exhausted as a lever.
  - Next lever: the training-side scorer fix (this needs GPU and PI authorisation, a named blocker).
- **NOT PROVEN through (i):**
  - That would be a power miss at ±0.21. Unexpected per `POWER.md`; the cause will be diagnosed (heavy tail? log effect?) before any re-read.
  - There will be no third read of the same rule without a new prereg.
- **REFUTED:**
  - gBG is dropped.
  - Next lever: the scorer retrain, or H-SEL-LANE-1 (§7) if it confirms.

## 5. Reported with it (DESCRIPTIVE; can never change the verdict). Four families.
- **LONGITUDINAL:**
  - plan v4 - GT v4 on fresh CRUISE and on the ego gate set (gt_valid);
  - the slow-down share;
  - the **overspeed share** (v4/v0 > 1.15 on CRUISE);
  - mean plan v4/v0 on CRUISE.
- **LATERAL** (what the proposals allow):
  - ADE to the human future (8 x 0.5 s) on CRUISE, banked vs rule;
  - Δ|y| at 4 s on changed picks;
  - the DAC failure rate and DDC mean.
- **Safety:** NC and TTC failure rates; the number of changed picks that are better or worse by true PDMS.
- **TACTICAL: n/a.** There is no manoeuvre-decision label on these tokens, and REFe emits no tactical output here. Reported with that reason and n = 0.
- **STRATEGIC: PAUSED** (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` item 25).
- **By log class:** the 62 held-out logs vs the 13 never-in-C3 logs. This is the shared-log / winner's-curse check of `POWER.md` §3.

## 6. Gates that must ALL hold before the verdict is read (any red = INSTRUMENT FAILURE)
1. **G-PIN:**
   - this file's sha256;
   - `CODE_PIN.json` (`selc_*.py`, `make_confirm_tokens.py`, both token lists) and its sha256 above;
   - `PREREG_SEL.md` == `e00fa28a...`.
2. **G-RULE and G-ANA** (§1, §3).
3. **Dump smoke identity** (GPU, under the shared lock). On the 6 smoke C3 tokens:
   - logits bit-identical to C3 `table.npz`;
   - pick equal;
   - executed props8 xy bit-identical (heading |d| ≤ 1e-6);
   - raw traj bit-identical to the ego probe's `traj_all_A0`;
   - ego vector bit-identical to `ego_real`.
   - **RED:** the same comparison against a cyclically shifted reference must fail.
4. **Score smoke gate.** On the 6 smoke tokens, all 64 harness columns must equal C3's table rows **bit for bit**: PDMS and all six sub-scores, 384/384 cells.
   - **RED:** the banked pick's poses shifted by +2.0 m in y must differ on at least 1 token.
5. **G0-fresh:** the dump's own pick equals the offline banked pick (torch float32 argmax of the aggregate) on every fresh token.
6. **Fidelity:**
   - the banked-pick poses equal the banked full-navtest u400 seam to ≤ 1e-3 m;
   - fingerprints are equal;
   - the frame control is ≤ 1 mm on every token.
7. **G2:** the banked column equals the banked full-navtest csv **bit for bit** on tokens with identical poses. There must be at least one such token, and every row must be valid and finite.
8. **Column identity:** on every token where the rule pick equals the banked pick, the two columns are identical.

## 7. SECONDARY pre-registered hypothesis (separate; it can NOT change §4): H-SEL-LANE-1
- **The rule:** lane-head temper L(0.25). Pick `argmax NC*DAC*(5EP + 5TTC + 2C)/12 * p_lane^0.25`, ungated (`sel_rules.pick_L(L, 0.25)`).
- **Its record so far:** P2 read it descriptively and post hoc on held-out, at +1.03 [+0.67, +1.53] mix x100.
- **When it is read:** from the 64-column table, once it exists, with the same estimator and the zero-floor mix.
  - Table gates: the table at the banked pick and at the rule pick each equals its verdict column bit for bit.
- **Committed outcomes:**

| verdict | condition |
|---|---|
| **CONFIRMED (PDMS superiority)** | lower CI > 0 |
| **REFUTED** | point < 0 AND upper CI < 0 |
| **NOT PROVEN** | anything else |

- It is reported with the four families.
- Also descriptive only, from the same table: Sg(1), Bg(0.2), E(0.5), uBG(0.2, 0.05), and the actual / random-of-64 / best-of-64 headroom.

## 8. Compute and where things live
- **Chain:** `selc_chain.py`, launched detached by WMI. It runs `preflight` → `dump_smoke` → `score_smoke` → `dump` (7 GPU segments, lock released between them) → `consolidate` → `verdict` → `analyze` → `table` → `table_read`.
- **Outputs:**
  - `CONFIRM_RESULT.md` and `confirm_result.json`, with marker `markers/selc_verdict.DONE`;
  - `CONFIRM_TABLE_APPENDIX.md`, with marker `markers/selc_full.DONE`;
  - `markers/selc_full.FAILED` on any failure.
- Seams and harness run dirs are written to `F:/TanitAD_selc_20261107/`, which is NTFS, so they stay off the E: drive with its 1 MiB clusters.
- **Estimate:** verdict ≈ 4 h plus lock waits; table ≈ a further 4.6–6.2 h (`POWER.md` §4).
