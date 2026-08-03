# P9 — the IDM's `long_accel` channel: a fix, its REFUTATION, and the better diagnosis the refutation bought

**Date** 2026-08-03 · **Substrate** dev box (RTX 4060), **0 pod GPU-h** · **Run directory**
`TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-03-idm-derived-accel/`
· **Primary artifact** `results_idm_derived_accel.json` (B=2000) · **log** `run_log.txt`

---

## 0. HEADLINE

1. ⛔ **My own fix is REFUTED — separated-WORSE.** Deriving `long_accel` from a supervised speed
   sequence moves ΔR² **−0.2530 [−0.4831, −0.0997]** (−0.2061 → −0.4591), and costs `yaw_rate`
   **−0.1142 [−0.3635, −0.0688]**. Both outcomes were committed before the run; this is the negative
   one and it is reported as the result.
2. ⭐ **The refutation produced the better diagnosis, and it is the real finding.** On the same run
   the **BASELINE's `long_accel` is NOT SEPARATED from a negative control whose latent↔target link
   was destroyed** (−0.0984 [−0.3087, +0.0179]) — while `speed` (+0.7187 [+0.4827, +0.9830]) and
   `yaw_rate` (+0.2252 [+0.1317, +0.3505]) and ADE (−6.59 m [−9.99, −3.44]) **are** separated.
   ⇒ **`long_accel` carries no recoverable information from the frozen v1 latents at this scale.**
   No reparameterisation of the head can repair that; the lever is the representation or the data.
3. ⛔ **A defect bigger than the one I set out to fix, exposed only because the four families were
   run: TACTICAL is AT OR BELOW CHANCE on every arm.** Lateral BA **0.3293** vs chance 0.3333;
   longitudinal **0.2735** vs 0.3333; mixed **0.1610** vs 0.2000. The lateral confusion never
   predicts `right` **at all** (0 of 35 true right-turns) and answers `straight` for 2,278 of 2,306.
   A scalar ADE of 5.18 m cannot see any of this.
4. ✅ **The instrument discriminates.** The shuffled-latent control lands at chance on every channel
   (speed R² +0.0050, ADE 11.77 m vs 5.18 m) and the separations above fire against it, so a null on
   `long_accel` is a property of the channel, not of the protocol.

---

## 1. THE DEFECT, ARGUED FROM PRIMARY SOURCE

**MEASURED (mine, re-read at HEAD)** — `…/2026-07-27-fleet-sync-idm-steer/raw/idm5_ensemble.json`,
the banked HELD-OUT read of the shipped ensemble:

| channel | `a0_on_these_windows` | seed 0 | seed 1 | per-domain pai / cm |
|---|---:|---:|---:|---|
| speed | **+0.8651** | +0.8586 | +0.8567 | +0.9070 / +0.7590 |
| yaw_rate | **+0.8108** | +0.9055 | +0.9049 | +0.9035 / +0.3308 |
| steer | **+0.7419** | +0.7866 | +0.7857 | +0.7340 / +0.5648 |
| **long_accel** | **−0.2398** | **−0.1862** | **−0.1510** | **−0.2979 / −0.2538** |

**A negative R² means the head predicts `long_accel` worse than the constant training mean.** Three
of four channels work; one is broken, on every seed and in both domains. *(The `IDM_FOUR_FAMILIES.md`
headline quotes the ensemble's −0.0591; every individual read is worse.)*

**Why it looked avoidable — MEASURED (mine),** comma2k19 val cache, 30 episodes / **8,940 windows**
(`scratchpad/probe_idm_defect.py`): the CAN `long_accel` channel is recoverable from the **TRUE**
speed track by a centred difference at **R² 0.9021** (pooled corr 0.9506; per-episode corr median
0.867). The head's window is NON-CAUSAL and spans t−4..t+4, so v(t−1) and v(t+1) are already inside
it. Nothing in the architecture or the loss connected the two: `long_accel` was a fourth independent
linear readout of the centre token (`stack/scripts/idm_head.py`, `IDMHead.forward`).

---

## 2. WHAT I IMPLEMENTED

| file | what |
|---|---|
| `stack/scripts/idm_head.py` | `speed_seq_targets_at`, `derive_long_accel`, `IDMHead(speed_seq=…)` + per-token speed readout, `deployed_scalars`, `predict`, `fit_head` (module-returning; `train_head` stays JSON-safe for `run_idm_proof.py:387`), `head_kw` passthrough, and the `idm_loss` terms. **All off by default — every banked checkpoint and all 12 existing callers keep the legacy path bit-for-bit.** |
| `stack/tests/test_idm_head.py` | 14 tests (was 8): target alignment, the exact centred-difference identity, the deployed-scalar swap, the legacy no-op, the loss-term contracts, and the end-to-end contract with its own negative control. |
| `…/2026-08-03-idm-derived-accel/run_idm_derived_accel.py` | the experiment: encode → episode-disjoint split → 3 arms × 3 seeds → four families + paired episode-cluster bootstrap. |

**A design error caught by measurement, before the real run.** The first version supervised only the
absolute speed sequence and let `long_accel` be derived from it. That is **worse than the direct
readout** — differencing two window positions multiplies the sequence's error by 1/(2·dt) = **5×**
(synthetic contract: derived **0.6134** vs direct **0.8755**). The derived channel must still be
supervised **on its own target**: the physics is a reparameterisation, not a substitute for
supervision. `test_derived_accel_needs_its_own_supervision_not_just_a_speed_sequence` is the
regression for it. With that term the synthetic contract inverts decisively — long_accel
**0.1127 → 0.9830** — and the gain *grows* as the latent gets noisier, which is the direction the
real corpus lies in. **That is exactly why the real corpus had to be run: the synthetic evidence
pointed the wrong way.**

---

## 3. THE PROTOCOL

* **Substrate** 50 content-clean comma2k19 episodes `ep_00040..ep_00089` (the band settled by sha256
  of raw bytes on both hosts, `…/2026-07-27-anchor-settlement/`), encoder
  `v1_speedjerk_ckpt.pt` step 29999, state_dim 2048, k=4/9 frames, stride 2, horizons (5,10,15,20).
* ⛔ **Geometry asserted before scoring** — `frames_u8` is `[T, 9, 256, 256]`, asserted in
  `build_substrate` (flagship v1 is 256 px SQUARE).
* **Split EPISODE-DISJOINT**, 33 train / **17 held-out** episodes → 4,554 / **2,346** windows.
* **Labels** use `heading_repair(v_min=0.5)` verbatim from the prior panel, so the yaw-rate target is
  the one the shipped checkpoint was stamped with.
* **Both arms are trained FROM SCRATCH on the same TRAIN episodes**, 3 seeds, ensembled by mean of
  per-seed predictions (the shipped recipe), 100 epochs.
* ⚠️ **No claim is made here about the shipped checkpoint.** The local comma cache IS its training
  pool (`IDM_FOUR_FAMILIES.md:105`); every contrast below is internal to this run.
* **100 epochs, not 12, because the fix has a precondition.** At 12 epochs speed R² is **−1.66** for
  every arm — the speed channel is below chance, so a fix that derives from it cannot be tested. The
  epoch sweep (12 / 40 / 100 → speed R² −1.82 / +0.30 / +0.71) is in §6. Measuring a longitudinal fix
  in a regime where the longitudinal channel is broken would not have been a test.

---

## 4. THE RESULT — per-scalar R², episode-cluster bootstrap, B=2000

| arm | speed | yaw_rate | steer | **long_accel** |
|---|---:|---:|---:|---:|
| **A — baseline (direct readout)** | +0.7237 [+0.4064, +0.8516] | +0.1552 [−0.0446, +0.1947] | +0.3003 [−0.4615, +0.3065] | **−0.2061 [−0.4254, −0.0598]** |
| **B — derived from speed sequence** | +0.7263 [+0.3762, +0.8602] | +0.0411 [−0.3510, +0.0993] | +0.3079 [−0.6549, +0.3157] | **−0.4591 [−0.8327, −0.2031]** |
| **NEG — latents shuffled** | +0.0050 [−0.3448, +0.0102] | −0.0699 [−0.2569, −0.0398] | −0.0249 [−0.1497, −0.0180] | −0.1077 [−0.2999, −0.0170] |

**Paired episode-cluster bootstrap** (same resampled clusters in each draw), `*` = SEPARATED:

| contrast | speed | yaw_rate | steer | long_accel | ADE@2s |
|---|---|---|---|---|---|
| **B − A (THE FIX)** | +0.0025 | **−0.1142 [−0.3635, −0.0688]\*** | +0.0076 | **−0.2530 [−0.4831, −0.0997]\*** | −0.232 m |
| B − NEG | **+0.7212\*** | +0.1110 | +0.3329 | **−0.3514 [−0.7327, −0.1170]\*** | **−6.822 m\*** |
| **A − NEG** | **+0.7187 [+0.4827, +0.9830]\*** | **+0.2252 [+0.1317, +0.3505]\*** | +0.3252 | **−0.0984 [−0.3087, +0.0179] — NOT separated** | **−6.590 m\*** |

⛔ **Read the last row first.** The baseline's `long_accel` cannot be told apart from a control whose
latents were permuted across windows, while every other channel and ADE separate decisively from that
same control. **The channel is not being predicted at all** — it was never a parameterisation
problem, and the negative R² is the head fitting noise, which is why it *degrades with more training*
(§6). Deriving it makes it worse for the mechanical reason in §2: speed MAE is still **4.035 m/s** at
R² 0.72, and 5× that against a target whose own MAE is **0.452 m/s²** is hopeless.

---

## 5. THE FOUR FAMILIES — reported per family, never pooled (baseline arm, n = 2,346 windows / 17 clusters)

**ADE@2s 5.1804 m [3.9332, 6.6052]** — one row, *not* the result.

**LONGITUDINAL** — `traj_speed_mae 4.5162 m/s` (bias −0.2355), `along_mae 5.1285 m`
(final bias −0.4771), `accel_mae 3.3686 m/s²`, scalar `speed_mae 4.035 m/s` (bias +0.2808).
**Distance-keeping (headway / time-gap / TTC): UNAVAILABLE, n = 0** — comma2k19 ships no object
annotation at all. A **WORK ITEM**, not a pass.

**LATERAL** — `heading_mae 0.0190 rad` (n=8,715), **`curvature_mae 0.00844 1/m`** (n=6,487),
**`yaw_rate_mae 0.02431 rad/s`** (n=6,487), `cross_track_mae 0.300 m`, final 0.6318 m; scalar
`yaw_rate_mae 0.02024 rad/s`, `steer_mae 0.00926`. Cross-track alone would read "lateral is fine";
curvature and yaw-rate are where the trouble is, which is why the family is not cross-track.

**TACTICAL** — ⛔ **at or below chance on every axis and every arm.**

| axis | baseline BA | derived BA | shuffled-latent control | chance |
|---|---:|---:|---:|---:|
| lateral | 0.3293 [0.3216, 1.0000] | 0.3184 | 0.3333 | 0.3333 |
| longitudinal | 0.2735 [0.1396, 0.4182] | 0.2588 | 0.3333 | 0.3333 |
| mixed (5-way) | 0.1610 [0.0880, 0.3559] | 0.1443 | 0.2000 | 0.2000 |

*lateral confusion (gt rows × pred cols)* `[[0,35,0],[0,2278,28],[0,5,0]]` — **`right` is NEVER
predicted** (0 of 35), `left` never predicted correctly (0 of 5), and `straight` absorbs 98.8 %.
Accuracy reads **0.971** because the classes are that imbalanced; balanced accuracy reads 0.3293.
*longitudinal confusion* `[[99,75,94],[1326,335,314],[58,16,29]]` — recall `decelerate` 0.3694,
`cruise` 0.1696, `accelerate` 0.2816. **This is the programme's known "manoeuvre decision is the
weakest link" defect, quantified on the IDM for the first time**, and no ADE number exposes it.
`goal_setting`: **PARTIAL** — selected-vs-executed is reported; anchor selection is not, because the
IDM regresses a single trajectory and has no anchor set.

**STRATEGIC** — **UNAVAILABLE, n = 2,346.** No route/goal label on any IDM substrate: comma2k19 has
none, and PhysicalAI-AV is settled at five probes as carrying no map, lane graph, junction annotation
or route signal, with clip-local egomotion and no GNSS. A **WORK ITEM**, not a pass.

---

## 6. THE SWEEP THAT SET THE OPERATING POINT (MEASURED, mine; 1 seed, same split)

| epochs | A speed R² | A long_accel | B long_accel (derived) | B direct readout | A ADE |
|---:|---:|---:|---:|---:|---:|
| 12 | −1.8170 | −0.1150 | −0.2603 | −0.4558 | 20.815 |
| 40 | +0.2964 | −0.1540 | −0.2304 | −0.4615 | 12.576 |
| 100 | **+0.7111** | **−0.2419** | −0.6419 | −0.5442 | 5.230 |

Speed and ADE improve monotonically with training; **`long_accel` gets monotonically WORSE.** That is
the signature of a head fitting noise on a channel with no signal, and it independently corroborates
§4's not-separated-from-control result.

---

## 7. WHAT I RECOMMEND

1. ⛔ **Do not enable `speed_seq` to fix `long_accel`.** Refuted, separated-worse. The code stays
   because it is off by default, tested, and now carries the negative result in its docstring so
   nobody re-derives it.
2. ⭐ **Stop treating `long_accel` as a head defect.** It is not separated from a destroyed-input
   control. The next admissible experiment is whether ANY head recovers it from these latents — e.g.
   a linear probe on the true window latents — and if not, the lever is the encoder or the corpus.
3. ⭐ **The TACTICAL family is the bigger defect and it is now measured.** Never-predicted turn
   classes at BA 0.3293 vs chance 0.3333 is a larger, more decision-relevant failure than a scalar
   R², and it was invisible in every ADE-only report.
4. ⚠️ **`distance_keeping` is UNAVAILABLE on comma2k19 by construction.** Any longitudinal claim that
   needs headway/TTC must move to a substrate with object tracks — PhysicalAI-AV's `obstacle.offline`
   covers 97.44 % of the corpus and the episode ingest still does not read it. A work item.

---

## 8. EVIDENCE CLASS

| claim | class |
|---|---|
| §1 banked held-out per-channel table | **MEASURED (mine, re-read from `idm5_ensemble.json`)** — not inherited from the summary |
| §1 R² 0.9021 recoverability identity | **MEASURED (mine)** — `scratchpad/probe_idm_defect.py`, 8,940 windows |
| §4 / §5 / §6 every number | **MEASURED (mine)** — `results_idm_derived_accel.json`, `run_log.txt` |
| §2 synthetic 0.6134 / 0.8755 / 0.1127 / 0.9830 | **MEASURED (mine)** — `stack/tests/test_idm_head.py` fixture |
| content-clean band `ep_00040..ep_00089` | **INHERITED** (`…/2026-07-27-anchor-settlement/`), not re-hashed here |
| local comma cache is the shipped head's train pool | **INHERITED** (`IDM_FOUR_FAMILIES.md:105`) — the reason no shipped-checkpoint claim is made |
| no route/map label on either substrate | **INHERITED** (CLAUDE.md five-probe settlement) |
