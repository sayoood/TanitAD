# P2 result: R1 ADOPT, R2 GO for P6 (2026-10-14)

> **Arm:** P2, the joint PDM fine-tune. HF `Sayood/tanitad-refe-p2-joint-pdm`; `model_final` sha256 `ea31db2296210b6a35dbf32ad95f1813edfa259ab330002b3ca7564aac727e51`; 1,209 steps; **one training seed**.
> **Rules:** `eval/PREREG_P2.md` and `eval/PREREG_P2_AMENDMENT_1.md`, both landed before any P2 data existed.
> **Tier:** T1 = NAVSIM v1.1 PDMS. This pseudo-simulates the ego's open-loop plan against logged agents. It is **self-action open loop, not closed loop**, and not a driving-in-simulation claim.
> **Estimator:** paired log-cluster bootstrap over the navtest logs, 10,000 resamples, 95 %, seed 20260927. It answers "would another draw of LOGS say this?" and nothing else.
> **Evidence class:** every number below is **MEASURED (ours)** and comes from the JSON named beside it.

## 1 · The two reads

| read | tokens / logs | vs | Δ PDMS ×100 [95 % CI] | better / worse / tied | outcome |
|---|---|---|---|---|---|
| **R1** | 923 / 43 | deployed system (85.61) | **+3.97 [+2.22, +5.67]**: 85.61 → 89.57 | 400 / 260 / 263 | **ADOPT** (`raw/2026-10-05-sft-stage2/p2_R1/result.json`) |
| **R2** (decides P6) | 12,146 / 136 | SFT-4 arm B u400 (88.25) | **+0.57 [+0.02, +1.18]**: 88.25 → 88.82 | 5,449 / 3,178 / 3,519 | **GO** (`raw/p2_readout/p2_full/r2_result.json`) |
| R2 minus C1 (reported, not gated) | 12,110 / 136 | u400 | +0.57 [+0.01, +1.18] | 5,435 / 3,170 / 3,505 | same |
| R2b (reported, decides nothing) | 12,146 / 136 | deployed (84.46) | +4.36 [+3.67, +5.08] | 6,012 / 3,106 / 3,028 | — |

**R2 against the amendment §4 rule.** The rule, as committed: GO requires **lo > 0 AND no sub-score adverse**, where a sub-score is adverse if its paired interval lies entirely below 0.
- lo = **+0.018**, which is > 0.
- No sub-score is adverse.
- ⇒ **GO.**

**Gates (all `ok`, `gates_ok: true`):**
- the pinned token file: md5 `3a8b3653…`, 12,146 tokens, 136 logs;
- the pinned u400 and deployed baselines;
- the harness: `PASS 12146/12146`, csv means equal to the harness summary at 4 dp;
- the seam: 12,146 rows, 0 misses, `rule == navsim_v1`, settings equal to u400's;
- the frame control: equal to u400's, i.e. the declared 36-token C1 set and nothing more;
- the checkpoint: sha256 matches;
- R2b equals stage 2's own primary bit for bit;
- `stage2_sft.py` unmodified (md5 `831a8d76…`).

## 2 · The six PDMS sub-scores (R2, P2 − u400, ×100)

| sub-score | Δ [95 % CI] | u400 → P2 | adverse? |
|---|---|---|---|
| NC | −0.13 [−0.31, +0.08] | 98.94 → 98.81 | no |
| DAC | +0.30 [−0.23, +0.89] | 96.58 → 96.87 | no |
| **EP** | **+1.14 [+0.61, +1.70]** | 81.20 → 82.34 | no (separated gain) |
| TTC | −0.31 [−0.80, +0.14] | 95.63 → 95.32 | no |
| C | +0.01 [0.00, +0.03] | 99.97 → 99.98 | no |
| DDC | −0.16 [−0.43, +0.13] | 97.66 → 97.51 | no |

- **The gain is ego progress.** EP is the only separated sub-score. NC, TTC and DDC point the other way without separating.
- Zero-PDMS tokens fell from **529 to 509**.
- Distance left to the human row (94.55, `RESULT_NAVTEST_FINAL.md`): **5.73 PDMS**.

## 3 · The four metric families (R2 vs u400; reported, not gating, amendment §5)

| family | result |
|---|---|
| **LONGITUDINAL** | **3 of the 21 longitudinal/lateral components are adverse, all longitudinal** (unpaired interval non-overlap, a conservative reading). Values are u400 → P2: along-track MAE **[0.988, 1.083] → [1.093, 1.180] m**; speed MAE **[0.711, 0.771] → [0.778, 0.836] m/s**; speed RMSE **[1.062, 1.153] → [1.168, 1.251] m/s**. ⇒ P2 tracks the human's speed less closely while making more progress (EP +1.14). This is the same signature as a planner driving faster than the logged human. |
| **LATERAL** | no adverse component |
| **TACTICAL** (no interval; reported, not tested) | lateral decision accuracy 0.9251 → 0.9237 (κ 0.863 → 0.860); longitudinal decision 0.6791 → 0.6748 (κ 0.516 → 0.510); collapsed 5-way manoeuvre 0.7831 → 0.7794 (κ 0.718 → 0.715) |
| **STRATEGIC** | **PAUSED** by the PI (2026-09-19, `PI_DECISION_QUEUE.md` item 25) and **UNAVAILABLE in NAVSIM** (no strategic decision is scored). Never a pass. |

**Against the deployed system (R2b):** 0 of 21 components adverse; every tactical accuracy is higher with P2.

## 4 · What the GO does not mean (amendment §6, verbatim in substance, stated before the data)

1. **Labels.** u400 trained on 146,185 of 174,912 PDM labels; P2 trained on all 174,912. So Δ mixes three differences:
   - complete labels;
   - the un-detached visual context;
   - joint training of the LoRA.

   One arm cannot separate them; SFT-4R can.
2. **One seed each (H-ESTIM-SEED-1).** The interval is blind to training variance. A separated interval from single-seed arms is **necessary, not sufficient**.
   - This rig's training-seed floor is unmeasured.
   - The lower bound here is **+0.018**, the thinnest possible margin. This GO is therefore exactly the case the rule warns about.
3. **u400 is update 400 of 2,335,** a non-final checkpoint stage 1 never selected. P2 reports its final checkpoint, unselected.
4. **Inference is deterministic,** so there is no inference-seed variance on either side.
5. **Six sub-scores gate the GO,** so a NO-GO could have been a false negative. That costs nothing here.

⇒ **GO is a direction decision for the next run, not a lever claim.** It supplies the confidence that P2's recipe direction (complete labels plus joint fine-tuning) does not regress against u400. The **P6 spend is the PI's call**, and a pod needs the PI's explicit go-ahead (amendment §4).

## 5 · Next, per the rule

- **Finalise and land `eval/PREREG_P6.md` before the first P6 step.**
  - Its precondition 1 ("P2 has a verdict") now holds.
  - Of its precondition 2, the PI has decided P6-T (25 epochs), P6-N (command), P6-Y (unwrapped) and P6-D (data). P6-C is a provisioning recommendation: ≥ 24 vCPU, ≥ 64 GB RAM, ≥ 400 GB. ⚠️ The current A40 pod has **11 vCPU**.
- **Order on the pod (PI):** P2 ✅ → RL-S1 (running) → S1 aux drivable fine-tunes (queued; `PREREG_AUX_DRIVABLE_S1.md` A3) → P6.
- **The longitudinal finding is the lever to carry.** P2's progress gain came with worse human speed-tracking. Whether P6's paper recipe keeps EP while recovering speed fidelity is a read to register for P6's families.

## 6 · Files

- `raw/2026-10-05-sft-stage2/p2_full/`: stage 2's `result.json`, `score_status.json`, `score_1.log`, `seam.log`, and the families JSON and logs for P2 and deployed. Seam `.npz` copies are not banked, as for R1.
- `raw/p2_readout/p2_full/r2_result.json`: the decision record.
- `raw/p2_readout/p2_full/arm/`: the per-token table `refe_stage2_p2_full.csv` (md5 `bed5687f…`), the seam `stage2_p2_full.npz` (md5 `49a1f6ec…`) and its `.report.json`, the counts and manifest, and `MD5SUMS.txt`.
- R1: `raw/2026-10-05-sft-stage2/p2_R1/` (landed `0cadf78`).
