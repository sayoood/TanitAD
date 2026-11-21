# VERDICT_OBD (2026-11-11): DETACH-KEEPS at both seeds; BAND SEEDS-DISAGREE, so a third seed per init is due

**Card:** `PREREG_V8F_OBD.md` (frozen, sha256 `2d42cd51`), code of record `cdbff971`.

**Read:** `read/RESULT_OBD.json` / `.md` (`obd_read.py`, B 2,000, 54.8 s, problems 0).

**Tier and estimator:**
* Tier **T1-kin**: self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene, deterministic planner. It is never quoted as driving performance.
* Estimator: paired episode-cluster bootstrap over the 141 eval episodes. It answers *another draw of episodes*.
* *Another training run* is answered only by floor F (OBD-NULL_rep vs OBD-NULL_s0) and floor G (OBD-BAND_rep vs OBD-BAND_s0), one replicate pair each.
* **STRATEGIC: PAUSED** (PI item 25).

**Evidence class:** MEASURED (eight banked runs, dumps in one environment, zero training in the read).

**Reference:** NULL parity fails (252 of 276 tensors; `null_drift/NULL_DRIFT.md`: float drift from the first steps, not a path change). So, by the card's fallback, every row "against DAG-OACC" is read against OBD-NULL of the same seed. OBD-NULL_s1 was trained for that.

## 1. DETACH (OBD-CTL = DAG-OACC's recipe + the PI's detach, against OBD-NULL of the same seed): **DETACH-KEEPS** at both seeds
* **No cell costs beyond floor F at either seed**, in any family.
* **Seed 0:** no cell above floor F. Two cells are separated but sub-floor, both in the improving direction:
  * operative accel −0.0171;
  * tactical curvature −0.0192.
* **Seed 1:** seven gains beyond floor F:
  * operative ADE −0.0053 [−0.0078, −0.0030], speed −0.0088, along −0.0057, accel −0.0330;
  * tactical FDE −0.2595 m [−0.4657, −0.0520], speed −0.0850, cross −0.0416.
* ⚠️ **The seed-1 tactical gains coincide with cells where OBD-NULL_s1 itself drifted from DAG-OACC_s1** (code-drift row, §3: tactical FDE +0.2209, speed +0.0791, cross +0.0347). Netted against DAG-OACC_s1 they nearly cancel: FDE −0.2595 + 0.2209 ≈ −0.04 m. That figure is ESTIMATED, a subtraction of two paired deltas, not a bootstrap.
* ⇒ **The detach is cost-free on this read. That it gains is not shown.**

## 2. BAND (OBD-BAND = OBD-CTL + the band loss, against OBD-CTL of the same seed): **SEEDS-DISAGREE** (s0 BAND-ADDS, s1 BAND-COSTS)
**Four families, per seed** (cells beyond floor G):

| family | seed 0 | seed 1 |
|---|---|---|
| PRIMARY (ADE / FDE) | operative ADE −0.0019 [−0.0030, −0.0009], FDE −0.0058 [−0.0092, −0.0026]: **gain** | operative ADE −0.0014 [−0.0027, −0.0001], FDE −0.0051 [−0.0094, −0.0009]: **gain** |
| LONGITUDINAL (speed, along, accel, headway, time gap) | none beyond floor | none beyond floor. Separated sub-floor, worse: tactical speed +0.0537, along +0.0434; operative accel +0.0062, time gap +0.0058 |
| LATERAL (cross, heading, yaw rate, curvature) | operative cross −0.0013 and tactical cross −0.0398 [−0.0706, −0.0089]: **gain** | operative cross −0.0021: **gain**; tactical heading +1.1491° [+0.5761, +1.8852]: **COST** |
| TACTICAL (T's decisions) | none beyond floor | TAC-LAT −0.0171 [−0.0329, −0.0021]: **COST**; TAC-LON −0.0175 separated sub-floor |
| STRATEGIC | PAUSED (PI item 25) | PAUSED |

* **Consistent at both seeds:** the band loss improves the operative band (ADE, FDE, cross-track), as designed.
* **Not consistent:** the tactical band. Seed 1 pays in tactical heading and in T's lateral decisions.

## 3. The code-drift row is the finding to carry forward (`beside.code_drift_NULL_minus_DAGOACC`)
OBD-NULL and DAG-OACC share code, argv, seed and data; only GPU float order differs. Their difference is **separated on 7 of 24 cells at each seed**:
* **Seed 0:** operative ADE +0.0020, FDE +0.0041, TAC-LAT +0.0104.
* **Seed 1:** operative ADE −0.0015, FDE −0.0058; tactical FDE +0.2209 m, accel +0.2418.

These are the same sizes as the effects in §1 and §2. The band's operative gain (≈ −0.002 m ADE) is no larger than what a same-code rerun moves.

⇒ This is direct, MEASURED support for `H-ESTIM-SEED-1` on the full v8f-mini trainer: a separated episode CI does not separate a lever from a rerun. The single-pair floors F and G are therefore necessary, and with one pair each they are thin.

## 4. Committed next step (card line 48)
* **The card's step:** OBD-CTL and OBD-BAND at a third training seed per init: `--seed 10` (init M2b s0) and `--seed 11` (init M2b s1). That is 4 runs, ESTIMATED about 2.7 h on box 2 (this chain's runs took about 40 min each, MEASURED from the chain log). Then the band question is re-read per the card before any band outcome is quoted.
* **Box-2 slot:** after STOPBAL. Asked of the PI with this verdict.
* **Work item, beside the card:** a deterministic mode for the full trainer (`torch.use_deterministic_algorithms`, a fixed cuBLAS workspace). With it, a NULL-parity control can read 0 again, and replicates measure the seed alone. ESTIMATED: one 38-minute box-2 run to verify.

## 5. Beside reads (no bar)
`out/HEADS_OBD.json` (T's own decisions) and `out/TOPK_OBD.json` (top-3): see §6, appended when they finish.

## 6. Beside reads, appended 13:35 (no bar; `read/out/HEADS_OBD.json`, `read/out/TOPK_OBD.json`; MEASURED)
**T's own decisions over all ticks** (T-LAT-F1, T-LON-F1, goal macro-F1; paired episode-cluster bootstrap):

| pair | T-LAT-F1 | T-LON-F1 | goal macro-F1 | separated rows of 87 |
|---|---|---|---|---|
| OBD-CTL_s0 − OBD-NULL_s0 | −0.0049 [−0.0361, +0.0200] | −0.0134 [−0.0364, +0.0156] | −0.0076 [−0.0212, +0.0168] | 30 |
| OBD-CTL_s1 − OBD-NULL_s1 | −0.0072 [−0.0353, +0.0222] | +0.0121 [−0.0157, +0.0364] | −0.0020 [−0.0442, +0.0153] | 27 |
| OBD-BAND_s0 − OBD-CTL_s0 | +0.0106 [−0.0049, +0.0307] | +0.0119 [−0.0013, +0.0330] | **+0.0141 [+0.0008, +0.0257]** | 12 |
| OBD-BAND_s1 − OBD-CTL_s1 | −0.0155 [−0.0550, +0.0099] | −0.0041 [−0.0250, +0.0210] | −0.0035 [−0.0185, +0.0067] | 6 |
| replicate OBD-NULL_rep − OBD-NULL_s0 | −0.0122 [−0.0424, +0.0127] | +0.0132 [−0.0141, +0.0409] | +0.0005 [−0.0192, +0.0194] | 8 |

Reading of the table:
* No F1 row shows a separated cost.
* The detach moves many constraint rows (27–30 of 87 separated), as expected when O's gradient into T is removed. It moves no decision F1.
* The replicate pair alone separates 8 of 87 rows. That is the same training-run noise as §3, on T's heads.

**Top-3, tactical band (best-of-3 minus top-1, BAND − CTL):** unchanged at both seeds: s0 −0.0031 [−0.0101, +0.0043], s1 +0.0064 [−0.0036, +0.0171].
