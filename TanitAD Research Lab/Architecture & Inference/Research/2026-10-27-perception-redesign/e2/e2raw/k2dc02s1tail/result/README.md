# k2dc02s1tail/result -- the K2DC02s1 TAIL read (refcv9 E2), banked by MM 4a3d05 on 2026-11-23

The TAIL-EMA weights (W_tail = (EMA_6000 - 0.81872 * EMA_5000) / 0.18128, the E3-representative read; no step-0 init) of K2DC02s1 = K2DC02's argv
at training seed 1 (1f14f8d81657, 6,000 steps). Pre-registration `../../PREREG_K2DC02S1_2026-11-20.md`; the TAIL outcome decides the dose reading
(sec. 5). Kit `../scripts` + `../devbox` (seed-guard fix cdebf8e1). The EMA read of the same run: `../../../read_K2DC02s1/` (fa37d470).
Parent tail: `../../k2dc02tail/result/`.

## E2 verdict on the tail (`read/E2_VERDICT.md`): **NO-GO -- only clause (d) fails**

| clause | tier | K2DC02s1 tail | K2DC02 tail (parent) |
|---|---|---|---|
| (a) map primary nf_swedge_prec_lv_5_20 > K0t | T0 | PASS 0.4878 vs 0.4665 (s0 = s1) | PASS |
| (b) box AP@2 m not separated worse | T0 | PASS +0.0091 [+0.0038, +0.0146] | PASS |
| (c) S6c sidewalk < K0t | T1 | PASS s0 0.1827 vs 0.1914 | PASS |
| (d) no family row separated worse | T1 | **FAIL s0 + s1** | FAIL |
| (e) W4 preview vs camera-blind | T1 | PASS | PASS |

**The replicated defect is lateral cross-track.** Separated worse vs K0t at BOTH training seeds and BOTH inference seeds (paired episode-cluster
bootstrap, B 2000, 70 DEV episodes):

| row | K2DC02s1 s0 | K2DC02s1 s1 | K2DC02 s0 | K2DC02 s1 |
|---|---|---|---|---|
| LAT_cross_mae_m | +0.0074 [+0.0027, +0.0122] | +0.0059 [+0.0011, +0.0113] | +0.0075 [+0.0024, +0.0126] | +0.0083 [+0.0024, +0.0147] |
| LAT_cross_abs_final_m | +0.0165 [+0.0061, +0.0272] | +0.0129 [+0.0020, +0.0251] | +0.0173 [+0.0064, +0.0283] | +0.0198 [+0.0065, +0.0340] |

Beyond cross-track, one row separates in both training runs but at inference seed 0 only: LON_dk_headway_abs_err_m +0.0416 [+0.0048, +0.0884]
(parent +0.0486 [+0.0004, +0.1201]). The parent's six inference-s1-only rows (LON_accel, LON_dk_headway_min, LON_dk_min_ttc, LAT_heading,
LAT_yaw_rate, TAC_traj_lat) do NOT replicate here. Read every single-seed cell against the ~9.5 % replicate false-positive rate per cell
(H-ESTIM-SEED-1).
* LONGITUDINAL: only LON_dk_headway_abs_err_m (s0, both training seeds). * LATERAL: the two cross-track rows above; heading / curvature / yaw not
separated. * TACTICAL: no row separated (lat / lon / man5 within +-0.0054). * STRATEGIC (`strategic/`, report-only, sanitized; control REF - REF
exactly 0.0): plan route-following K2DC02s1t - K0t -0.0259 [-0.0772, +0.0259] s0, **-0.0815 [-0.1522, -0.0245] s1 separated worse**; the parent tail
shows the same at s1 only (-0.0630 [-0.1345, -0.0041]); gstr -0.0065 [-0.0420, +0.0248] and anchor (s0 +0.0167, s1 -0.0217) not separated.

## Dose reading (`verdicts/dose`, `floor`, `combined`): **DOSE-VOID -> the combined tail reading is VOID (no dose claim)**
* In-training validity VOID: 1 of 13 late probes (step >= 3,000) inside [0.01, 0.03] (PREREG_K2DC02 sec. 2 needs >= 2; H-R9-E2-DOSECTL).
* Pre-registered rows, reported anyway: LAT_cross_mae K2DC02s1t - K2DCt -0.0052 [-0.0128, +0.0010] s0 / -0.0032 [-0.0105, +0.0029] s1 (point
  past the -2.5 mm bar, not separated); jitter not < 0 -> neither SUPPORTED nor REFUTED even before validity. The floor (K2DC02s1 - K2DC02,
  training-run noise) -0.0001 / -0.0024: the dose deltas exceed it at both seeds, which decides nothing on a VOID read.
* Reproduction (sec. 2 item 6): REPRO-PASS, LAT_cross_mae_m pinned K2DCt - re-rolled K0t vs pinned: diff 0.0 at seed 0.
* Validity items (`verdicts/VALIDITY.txt`): seed-1 + HSG build guard passed; identity control IDCTL-PASS-TENSOR; deliberate regression RED;
  tail guard step 6000 / md5 28544db4; pinned comparators 2,643 files sha256-equal to the pin manifest.

## Next lever (PI-approved 2026-11-22 on exactly this outcome)
K2DC02cl, the clamp (`--w-map-hires 3 --map-hires-w-bounds 2.5,3.5`, PREREG_K2DC02CL_2026-11-22, e5ed8b55): it makes the dose ACTUALLY ~0.02
(the replay's mean 0.025-0.026), so the replicated lateral cross-track regression can be read against a valid dose for the first time.

## Provenance
Thor E2TAIL_K2DC02s1 (`thor/`: build, stage A, reproduction roll, stage B status; the first attempt was refused by the whole-file seed guard and
kept on Thor as E2TAIL_K2DC02s1.refused_seedguard_04baf902). Dev box: E2-TAIL K0-slot bank restored from HF (14,426 / 14,426 members), pull
sha256-verified, S6c / box-paired / verdict, then the tail verdicts. `logs/ema_verdicts.log` is the driver's EMA step refusing EXISTS inside the
tail's work root (expected: that step ran separately, banked with read_K2DC02s1). Line endings: 43 files converted CRLF -> LF.
