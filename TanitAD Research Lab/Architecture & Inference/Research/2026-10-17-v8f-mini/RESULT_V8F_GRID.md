# RESULT: PREREG_V8F_GRID (T's scene input at 8×20 tokens per frame instead of 4×10), seed 0

**VERDICT, as committed: FAIL-NO-MECHANISM. STOP at seed 0; seed 1 was not trained.**

* **Evidence class:** MEASURED.
* **Tier:** T1-class open loop (eval-141 grid, 2,399 windows, 141 episodes). The arm is conditioned on its own held upper-layer outputs. The mechanism read uses T0 absmask and GT-oracle evals.
* **Estimator:** paired episode-cluster bootstrap, B = 2,000, over episodes. It answers "another draw of EPISODES" only (`H-ESTIM-SEED-1`). F = |M2b_s0 − M2b_s1| = 0.1207 m (B's own two seeds); 2F = 0.2415 m.
* **Raw:** `raw/2026-10-22-grid/run/grid_read_s0.json` (the committed read) and the rest of `run/`.
* **Base:** B = **M2b**, decided by the committed rule from PREREG_V8F_PGT's own read (FAIL-NO-MECHANISM, `RESULT_V8F_PGT.md`).
* **Lever:** G = B + `--grid pool8x20`.
  * Built: +30,720 parameters exactly, all in the position table (7,214,495 → 7,245,215).
  * Same clips (4,512), steps (19,164) and p_gt schedule (1.0 → 0.2) as B; G gate PASS.
  * The derived sampler chunk fell 48 → 41 (lru 174 → 43); Amendment 1 declares it part of the lever.
* **Run:** box 2. The first chain run refreshed box 2 at tip `a8f2c89` (instruments landed at `4abdf834`).
  * G s0 trained 22:22–03:07 (17,016 s; 1.126 steps/s against B's 1.217, −7.5 %) and was evaluated to 03:11.
  * The first read REFUSED on a lever-record schema difference. Amendment 2 fixed it BEFORE any number was read (`raw/2026-10-22-grid/a2/`).
  * The chain was relaunched at `cf5956f2`: it skipped the finished training and evals and read at 03:29.
  * Boot unchanged; max USB temperature 44 °C; no guard stop.

## 1. The committed reads

| read | B = M2b_s0 | G = M2b-8x20_s0 | Δ (G − B) [95 %] | committed bar | met |
|---|---|---|---|---|---|
| **PRIMARY** ADE6 (m) | 2.175 | 2.099 | **−0.076 [−0.175, +0.016], ns** | < −2F (−0.2415), separated, at both seeds | **no** |
| **M-LAT** T's latcons present-slot RMSE (SD, slot-pooled) | 0.686 | 0.695 | **fall −1.2 %**; MSE Δ +0.016 [−0.004, +0.036], ns | a fall ≥ 12.5 %, separated | **no** |

* The window-weighted RMSE (reported only) agrees: 0.468 → 0.485, fall −3.6 %.
* Pre-registered beside reads:
  * latcons leak Λ 0.268 → 0.231 m (−0.037 [−0.080, +0.007], ns);
  * S5 low-TTC tail 16.98 % → 15.55 % (−1.43 pp [−3.16, +0.28], ns).

## 2. The four families (G − B, paired, same windows)

* **LONGITUDINAL**
  * **Speed MAE −0.052 m/s [−0.098, −0.009], separated, better.**
  * **Accel MAE −0.123 m/s² [−0.163, −0.085], separated, better.**
  * Along-track MAE −0.059 m, ns.
  * Distance keeping (K = 10, lead windows):
    * **min headway +0.099 m [+0.026, +0.180], separated**;
    * **min time-gap +0.016 s [+0.003, +0.032], separated**;
    * min TTC +0.03 s, ns.
* **LATERAL**
  * Heading MAE −0.33° [−0.83, +0.03], ns.
  * Yaw-rate MAE −0.05 °/s, ns.
  * Curvature MAE −0.0051 1/m [−0.0107, +0.0002], ns.
  * Cross-track −0.040 m [−0.089, +0.009], ns.
* **TACTICAL**
  * Per arm, descriptive: lateral macro-F1 0.816 → 0.822; longitudinal macro-F1 0.499 → 0.502.
  * T's present-slot RMSE by group, all slightly worse and ns: latcons 0.686 → 0.695, lonspeed 0.452 → 0.463, stoplead 0.815 → 0.818.
* **STRATEGIC** (un-paused for v8F-mini). Per-arm episode-cluster CIs, not paired; every one inside the other's interval:
  * position error @30 7.02 → 7.05 m, @80 11.21 → 11.46 m;
  * heading error @30 4.20° → 4.18°.

## 3. What the run shows (EXPLORATORY, one training seed; it guides the next lever and decides nothing)

1. **Four times the scene tokens does not make T more accurate.** All three constraint groups are slightly worse, ns. Spatial resolution of the frozen DINOv3 pooled tokens is not what limits T: the committed reading.
2. **G drives a little better longitudinally anyway**, separated on four reads: speed, accel, headway, time-gap.
   * ⚠️ Those four are one family and correlated, so they are one observation, not four.
   * This rig's `separated` has a ~9.5 % false-positive rate per cell with no replicate (`H-ESTIM-SEED-1`).
   * It is below every committed bar.
3. **With the GT code, G and B are identical:** oracle ADE6 0.969 against 0.971, −0.002 [−0.041, +0.036], ns.
   * So whatever G gained, it gained in what T hands O, not in how O reads the scene.
   * T's present-slot constraint RMSE did not improve, so it is not in those slots. That leaves T's decisions, goal or presence predictions, which this read does not separate.
4. **The sampler-chunk change (48 → 41) cannot be separated from the grid here.** Amendment 1 declared it part of the lever.

## 4. Rule Zero: the next levers

* **The committed next: more history frames (plan 13.6),** i.e. temporal context for turn and lane-change timing, under its own PREREG.
  * The pre-registration and instruments are zero-GPU work.
  * The box-2 run needs a PI go.
* **The pattern across the last three arms, MEASURED:**
  * **The arms:**
    * `PREREG_V8F_WC`: T's loss weight; present-slot RMSE −2.4 % over all groups, latcons only −1 %.
    * `PREREG_V8F_PGT`: O's exposure to T's own outputs; T slightly worse.
    * `PREREG_V8F_GRID`: T's input resolution; latcons RMSE +1.2 %.
  * None moved T's latcons accuracy, while the headroom through T's code is still **1.20 m ADE6** (GT-code oracle 0.971 against own code 2.175 m at M2b_s0).
  * Every arm moved ADE6 by less than 0.08 m, against a 0.2415 m bar.
* ⇒ **What T is given is the remaining place to look:**
  * temporal context (history frames, next, zero-GPU to prepare);
  * the encoder itself (LoRA on DINOv3), which is an architecture change, so the PI's AR-2 go.

## 5. Provenance

| artifact | where |
|---|---|
| committed read; base decision; B-side check; cache check; gate selfcheck | `raw/2026-10-22-grid/run/{grid_read_s0.json, base.json, check_inputs.json, cache_check.json, gate_M2b-8x20/selfcheck.json}` |
| G's four evals (npz / json / log) | `raw/2026-10-22-grid/run/eval/eval_M2b-8x20_s0*` |
| G's run (summary, selfcheck, argv, train log) | `raw/2026-10-22-grid/run/runs/M2b_s0_8x20/` |
| box-2 logs: both chain runs, both guard runs, every job log, telemetry | `raw/2026-10-22-grid/run/box2_logs/`, `run/telemetry_v8fgrid.csv` |
| box 2's eval-cache manifest | `raw/2026-10-22-grid/run/manifest_eval.jsonl`. The train manifest (3.05 MB) is not banked: sha256 `661f5d4578a462a0d7d071b96365d3b93f57911dfb85a0a99203efa0eadbfe12`, summarised by `cache_check.json` |
| Amendment 2 (the refusal and its fix) | `raw/2026-10-22-grid/a2/` |
| checkpoint (84 MB, not in git) | dev box `D:/Projects/TanitAD-artifacts/v8f_grid/M2b_s0_8x20/ckpt_last.pt`, sha256 `9562ceb795f3d79b0eb63e9a932f97629026ec5999ada45afc183569f8877d8b`; box 2 `D:/tanitad/runs/v8f_mini_full/M2b_s0_8x20/` |
