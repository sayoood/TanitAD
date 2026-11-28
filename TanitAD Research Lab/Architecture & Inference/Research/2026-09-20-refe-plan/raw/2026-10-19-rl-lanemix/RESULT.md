# RL-2: dosing only the LANE output of the RL scorer (zero training, CPU, RL-val selection tier)

Architecture & Inference agent (Sonnet), 2026-10-18. Package target `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-rl-lanemix/`. CPU only; no GPU, pod, ssh, train, navtest, git add/commit/push, register edit. RL-1's files and `rl/nav/` untouched (read-only use of `rlh.py`, `composites.py`, `mechanism.py`, the banked `composites_rlval.json`).
Pre-registration `raw/PREREG_RL2_LANEDOSE.md`, sha256 `8268ef2b39fedbfe2e28f567aa353d662d0bf5757b45bd3b5bf994a6be0654ba` (`sha256sum -c` OK), written and hashed BEFORE `lanedose.py` / `composites_ext.py` scored a pick against a label. The output JSON carries the PREREG hash and the script hash (`c4220a02...6fc6`, equal to the PREREG's).

## 0. Headline

1. **Formal row for the primary arm (R0 u1000): R2 L-PASS-BASE, and it is a knife-edge pass that does NOT show a lane-dose effect.** On FIT no config cleared the bar (feasible set empty); the mechanical rule took the minimum-deficit config `L0.25` (lane = 0.75 base + 0.25 RL; EP, C from RL; NC, DAC, TTC, DDC from base0). Read ONCE on the HELD half (37 logs, 1,439 sets, vs base0): **PDMS +0.308 [+0.003, +0.609]**, NC -0.139 [-0.346, 0.000], DAC 0.000 [-0.204, +0.207], EP +0.725 [+0.414, +1.035], TTC +0.139 [-0.372, +0.629], DDC -0.139 [-0.320, 0.000] -> PASS-VAL by the letter of RL-1's bar. **Bar S vs own A fails** (TTC -1.04 [-2.15, +0.07], NC -0.66 [-1.23, -0.16]).
2. **Why the pass is not evidence for the hypothesis (all MEASURED, RL-val, from the pre-registered reference rows and a post-verdict sensitivity):**
   * the lane dose adds nothing separable over V1 (dose 0) on HELD: PDMS **+0.023 [-0.150, +0.200]**, TTC -0.069 [-0.483, +0.338];
   * **V1 itself passes the PDMS bar on this HELD half** (+0.285 [+0.018, +0.548]) although it reads GAIN-LOST on the full set (+0.168 [-0.115, +0.402]) — the half is a favourable draw of logs;
   * the PDMS lower bound +0.003 is at the Monte-Carlo resolution of a 10,000-draw percentile: ten other bootstrap seeds on the same HELD sets give lower bounds between **-0.0004 and +0.015** (8 of 10 above 0; `raw/sensitivity_heldseed.json`, post-verdict, not part of the PREREG);
   * the NC upper bound 0.000 is discreteness: NC is worse on exactly **2 HELD sets and better on 0** (the NC delta is -0.139 in all three arms, the same two sets), so "not separated" there means two sets in two logs;
   * FIT-infeasible pick, B itself is **UNDERPOWERED** on HELD (its PDMS lower bound is -0.003).
3. **Consistency arms (same HELD sets, so NOT independent confirmations):** R0d u1000 -> R2 (c* = `L0.25`, PDMS +0.314 [+0.031, +0.601], lower bound stable +0.030..+0.044 over ten seeds; contrast vs V1 +0.060 [-0.025, +0.155]); R0b u500 -> **R4b** (c* = `L0.5`, PDMS +0.342 [-0.066, +0.808], GAIN-LOST-VAL). Bar S fails on HELD in all three.
4. **"The gain and the TTC loss move together at every dose" is NOT what this tier shows, and it cannot show the opposite either.** FIT: COMOVE false on all three arms (R0: G = +0.18/+0.20/+0.10/+0.03 and K = 0.00/+0.10/0.00/-0.13 for doses 0.25/0.5/0.75/1.0). FIT+HELD union (descriptive, post-verdict): R0 dPDMS vs base0 = V1 +0.168, then 0.25: +0.274, 0.5: +0.319, 0.75: +0.317, 1.0: +0.336 (about 63 % of the dose-1 lane gain over V1 is already there at 0.25); TTC vs V1 contrast within +-0.07 at every dose with CIs of about +-0.4. RL-val's resolution on TTC is the problem: B itself reads TTC -0.17 [-0.71, +0.38] here and -0.40 [-0.71, -0.09] on navtest (D-REFE-FREEPOINT-1, INHERITED). The question has to be answered at T1.
5. **Step 2 (extension, identical code): RL-1's V1 and V2 on R0d u1000 both read GAIN-LOST-VAL**, as on R0/R0b. V1: PDMS **+0.17 [-0.12, +0.40]**, EP +0.60 [+0.36, +0.83] (0.35 of B's +1.73), TTC -0.03 [-0.35, +0.26], NC -0.08 [-0.19, 0.00]; V2: PDMS -0.26 [-0.92, +0.37], EP -0.70, TTC +0.60 [-0.03, +1.27]. The reproduction control holds: the same script re-computes RL-1's banked R0 u1000 block (809 numbers) with max abs difference **0.0**.
6. **Committed next arm (row R2): the navtest registration of `L0.25` (R0 u1000 only), written in section 6, NOT built, NOT run** — the one navtest look is spent on V1, so this second look needs the caller/PI. The anchored RL arm stays BLOCKED (A40 runs P6; PI declined extra compute) and becomes the committed arm only if that T1 read fails.

## 1. Evidence class, tier, estimator

Tier: RL-val SELECTION tier (the true NAVSIM v1.1 sub-scores of the argmax pick; 3,000 sets / 75 logs); never T1. Class MEASURED (banked logit dumps + `pdm_val`/`terms_val`; input hashes in the PREREG). Estimator: paired log-cluster bootstrap (`rlh.Boot`, 10,000 resamples, seed 20260927) built on the logs of the half read (FIT 38 logs / 1,561 sets, HELD 37 logs / 1,439 sets, split `default_rng(20261019)`). It answers "another draw of LOGS" only; one training seed per arm (H-ESTIM-SEED-1); the scorer is deterministic, no inference variance. Nothing here is a lever-effect claim.
Controls, all held (`lanedose_rlval.json` `controls`): dose 0 = V1 picks, dose 1.0 = D1 picks, clip tau 0 = V1, tau 1e9 = D1, bit-exact on all three arms; lane-form algebra difference 0.0; split disjoint/complete (38/37 logs); bootstrap identity vs `proxy_eval.boot` max abs 0.0 on full set, FIT and HELD; reproduction of RL-1's banked V1 and D1 rows (full set, all 7 metrics, mean and both CI ends) max abs 0.0; deliberate-regression arm (`L1.0` with NC, DAC, TTC, DDC from another set's base0) dPDMS **-1.41 [-2.19, -0.71]**, the bar fails it. Extension: all-RL = B and no-RL = SRC bit-exact (both SRCs) on R0 and R0d, permuted-SRC regression arm dPDMS -4.27 [-5.15, -3.42] on R0d.
Families at this tier: LONGITUDINAL = EP, TTC; LATERAL = DAC, DDC (reported in the JSON for every row); TACTICAL and lateral kinematics NOT COMPUTABLE (n = 0, no instrument at RL-val); STRATEGIC PAUSED (PI item 25, 2026-09-19).

## 2. Step 3: the lane-dose family, FIT selection and the single HELD read

Search space (7 selectable configs + V1 reference = 8 <= 9): `L0.25, L0.5, L0.75, L1.0` (lane = (1-a) base + a RL) and `C0.6, C1.4, C2.6` (lane = base + clip(RL - base, +-tau)); tau = R0 u1000 quantiles q25/q50/q75 of |lane delta| (0.609, 1.359, 2.625) rounded, computed from logits only before the PREREG (`raw/stage0_taus.json`).

| arm | FIT feasible set | c* (FIT deficit) | HELD label of c* | HELD dPDMS | bar S | UNDERPOWERED | row |
|---|---|---|---|---|---|---|---|
| **R0 u1000** (primary) | empty | `L0.25` (0.109) | PASS-VAL | +0.308 [+0.003, +0.609] | fails | yes | **R2** |
| R0b u500 | empty | `L0.5` (0.182) | GAIN-LOST-VAL | +0.342 [-0.066, +0.808] | fails | no | R4b |
| R0d u1000 | empty | `L0.25` (0.111) | PASS-VAL | +0.314 [+0.031, +0.601] | fails | no | R2 |

FIT table, R0 u1000 (vs base0, x100; 7 configs, lower bound of dPDMS decides): `L0.25` PDMS +0.242 [-0.109, +0.592] (deficit 0.109), `L0.5` +0.263 [-0.217, +0.736], `L0.75` +0.163 [-0.338, +0.636], `L1.0` +0.093 [-0.596, +0.661], `C0.6` +0.164 [-0.348, +0.654], `C1.4` -0.005 [-0.664, +0.576] (BOTH-LOST: DAC upper bound -0.058), `C2.6` +0.117 [-0.565, +0.679]. No config is PASS-VAL on FIT; none carries a separated NC/TTC adverse component except `C1.4` (DAC). FIT is half the data: V1 itself is +0.061 [-0.430, +0.426] and B itself +0.604 [-0.082, +1.262] there.

Post-verdict DESCRIPTIVE dose-response (FIT+HELD union, vs base0, x100; it selects nothing and changes no row; the HELD half is already consumed, and the family endpoints were known to me, PREREG section 0):

| config (R0 u1000) | dPDMS | dTTC | dNC | dEP | label | contrast vs V1: PDMS / TTC / NC |
|---|---|---|---|---|---|---|
| V1 (dose 0) | +0.168 [-0.115, +0.402] | -0.067 | -0.083 | +0.64 | GAIN-LOST | 0 |
| `L0.25` | +0.274 [+0.041, +0.506] | -0.033 [-0.389, +0.301] | -0.083 [-0.225, +0.036] | +0.730 | PASS-VAL | +0.105 / +0.033 / 0.000 |
| `L0.5` | +0.319 [-0.023, +0.667] | -0.033 | -0.033 | +0.858 | GAIN-LOST | +0.150 / +0.033 / +0.050 |
| `L0.75` | +0.317 [-0.043, +0.671] | -0.100 | -0.083 | +0.971 | GAIN-LOST | +0.149 / -0.033 / 0.000 |
| `L1.0` (D1) | +0.336 [-0.092, +0.730] | -0.100 [-0.681, +0.428] | -0.167 [-0.468, +0.080] | +1.023 | GAIN-LOST | +0.167 / -0.033 / -0.083 |
| `C0.6` / `C1.4` / `C2.6` | +0.239 / +0.238 / +0.341 | 0.000 / -0.033 / -0.067 | -0.083 / -0.133 / -0.167 | +0.76 / +0.87 / +1.00 | GAIN-LOST | +0.07 / +0.07 / +0.17 |

Bar S vs own A fails for every config, every dose, at R0 u1000 (TTC -0.83 to -0.93, NC -0.20 to -0.33 on the union): the base0 safety heads set it, exactly as V1's -0.90, so the lane dose cannot move it. On R0b, bar S passes for `L0.5`, `L1.0`, `C1.4`, `C2.6` on the union (R0b's own A holds less TTC over base0).
R0d union: `L0.25`, `L0.75`, `C0.6` read PASS-VAL, `L1.0` GAIN-LOST (+0.384 [-0.050, +0.794]). R0b union: no config passes.

## 3. Step 2: RL-1's V1/V2 on R0d (an extension; identical code; no navtest)

`composites_ext.py` differs from RL-1's `composites.py` only as listed in `raw/composites_ext.diff`. R0d, vs base0, x100, paired log-cluster bootstrap, SRC of V2 = C0-A at the same update (code identical), "own A" = R0d's A:

| state | B itself dPDMS | V1 label, dPDMS | V1 EP, TTC, NC | V2 label, dPDMS |
|---|---|---|---|---|
| u250 | +0.35 [-0.18, +0.84] | BOTH-LOST-VAL, +0.06 [-0.21, +0.30] | +0.69, -0.30 [-0.62, +0.00], NC -0.12 [-0.24, -0.02] | GAIN-LOST-VAL, +0.12 |
| u500 | +0.53 [+0.08, +0.94] | GAIN-LOST-VAL, +0.07 [-0.21, +0.30] | +0.34, +0.03, -0.08 | GAIN-LOST-VAL, -0.16 |
| u750 | +0.70 [+0.31, +1.08] | GAIN-LOST-VAL, +0.18 [-0.07, +0.40] | +0.62, -0.03, -0.08 | GAIN-LOST-VAL, -0.27 |
| **u1000** | **+0.63 [+0.21, +1.03]** | **GAIN-LOST-VAL, +0.17 [-0.12, +0.40]** | **+0.60 [+0.36, +0.83], -0.03 [-0.35, +0.26], -0.08 [-0.19, 0.00]** | **GAIN-LOST-VAL, -0.26 [-0.92, +0.37]** |

At u1000 V1 vs own A: TTC -0.77 [-1.55, -0.03] (bar S fails), V2 TTC -0.13 [-0.50, +0.22] (passes). R0d's B itself is TTC -0.03 [-0.58, +0.53] vs base0 on RL-val (R0: -0.17) but -0.77 [-1.49, -0.06] vs its own A. Descriptive R0d u1000 (vs base0): D1 (`L1.0`) +0.38 [-0.05, +0.79]; D2 B with lane from base0 +0.11 [-0.44, +0.59] (NC -0.20 [-0.35, -0.07] adverse); lane from A +0.27; D3 B with NC / DAC / EP / TTC / C / DDC from base0 each keeps PDMS +0.55..+0.66, only lane from base0 drops it (+0.11): the same carrier as R0. Selection/next-arm text per RL-1 PREREG_B section 4: GAIN-LOST-VAL -> anchored RL arm (BLOCKED, section 6).

## 4. Forecast check (committed in the PREREG, section 7)

Predicted row family R4 (HELD GAIN-LOST) with R1/R2 about 10 %: **wrong** — two of three arms read R2 (a pass by the letter), R0b read R4b. Predicted UNDERPOWERED true: right for R0, not for R0b/R0d. Predicted bar S fails for every config: right on R0/R0d, wrong on R0b's union. Predicted monotone PDMS between V1 and D1 with TTC inside its CI: right.

## 5. What this does and does not establish

* **Does:** the pre-registered rule, run as written, returns a PASS-VAL by RL-1's letter on the primary arm. The family and its controls are sound and reusable (`lanedose.py`). Gain over V1 from lane dosing at RL-val is small (union +0.07..+0.17 PDMS) and no point-estimate cost on TTC or NC is visible (union TTC contrast within +-0.07).
* **Does not:** show that a partial dose splits the gain from the navtest TTC/NC loss. RL-val cannot see that loss (B itself passes its bar here), the HELD contrast vs V1 is not separated, and the pass margin is Monte-Carlo-sized. Nothing is a lever-effect claim; no replicate arm was run (H-ESTIM-SEED-1).
* **Class to log (for RETRACTION_LOG-style learning, not a retraction):** a HELD half is only informative against its own reference; **check the dose-0 / base arm on the same half before reading a pass** (here V1 passes it too).

## 6. Next arm for row R2, and the blockers (Rule Zero)

Committed: **navtest registration of `L0.25` on R0 u1000** — the one experiment that can discriminate, because the loss it is meant to avoid exists only at T1. Written, NOT built, NOT run (brief: stop at the registration):
1. `nt_build.py` (RL-1's, FREEPOINT pipeline unchanged): add `VAR["L0.25"]` = `x = compose(R0B, BASE, (2, 4)); x[..., 6] = 0.75 * BASE[..., 6] + 0.25 * R0B[..., 6]` with the chunk logits `C:/Users/Admin/freept/work/out/chunk_00*.npz` (BASE and R0B; **only R0 u1000 has navtest logits — R0b and R0d cannot be registered without a GPU forward**). Gates (picks only, no label): BASE and R0B picks reproduced 12,146 / 12,146; all-RL = R0B; none-RL = BASE; **the same builder at dose 0 must reproduce RL-1's `points_V1.npz` V1 picks bit-exactly** (the navtest analogue of control 1).
2. `nt_run.py` (CPU, W3's `score_e.py`, own `--out`, marker `DONE_<V>`, RL-1 expected 25-60 min, FREEPOINT 1,335.7 s unloaded per 12,146 tokens; needs 4 GB free RAM).
3. `nt_read.py` (dry known-value test first, as RL-1's handoff step 2): bar dPDMS lower bound > 0 vs BASE and NC, DAC, TTC upper bounds >= 0; lever view vs R0B and C0A; families6.
4. Committed outcome rows: N-PASS -> report a zero-training deployable point (adoption is the PI's); N-GAIN-LOST / N-SAFETY-LOST / N-BOTH -> the anchored RL arm.
**BLOCKED, named:** (a) the second navtest look needs the caller/PI (RL-1's one look is V1); (b) the **anchored RL arm** (RL-1 section 7: lr 0 on the NC, DAC, TTC, DDC and lane slices during EPO; R0's reward, `--epo-lam 1.0`; a G-DVB-gated logit anchor) is ESTIMATED at 3.7 A40-h per arm plus a control and is blocked because the A40 runs P6 and the PI declined extra compute; it unblocks on a PI grant of about 4-8 A40-h or P6 freeing the A40, plus a `launch_gate.py` PASS token for the new flag. (c) Not run because of the 9-config cap: a safety-conditioned lane dose (apply the RL lane delta only where base0's NC and TTC pass probabilities exceed a fixed threshold); needs its own PREREG.

## 7. Proposed register rows (NOT edited; for the Master Mind)

* `D-REFE-RL2-LANEDOSE` (2026-10-18): lane-dose family `L(a)/C(tau)` on R0 u1000, selected on a FIT half by the pre-registered deficit rule (feasible set empty), `L0.25` reads PASS-VAL on the HELD half (dPDMS +0.308 [+0.003, +0.609], NC -0.139 [-0.346, 0.000], TTC +0.139 [-0.372, +0.629]) and fails bar S (TTC -1.04, NC -0.66 vs own A). Contrast vs V1 +0.023 [-0.150, +0.200]; V1 itself passes the same half; lower bound Monte-Carlo range -0.0004..+0.015. MEASURED, RL-val selection tier, one seed per arm; NOT a lever claim. `raw/2026-10-19-rl-lanemix/lanedose_rlval.json`.
* `H-REFE-LANEDOSE-1` (OPEN): a partial lane dose keeps the navtest PDMS gain without the NC/TTC loss. Not supported, not refuted at RL-val (tier blind to that loss). Discriminating test: navtest T1 read of `L0.25` (section 6).
* `D-REFE-RL1-HEADWISE` addendum: V1/V2 on R0d u1000 both GAIN-LOST-VAL (V1 PDMS +0.17 [-0.12, +0.40]); extension; `composites_r0d_ext.json`.
* BACKLOG: "REFE navtest look #2: `L0.25` composite, seam build + ~40 min CPU scoring (needs PI/caller go-ahead)"; "anchored RL arm: BLOCKED on A40 (P6), PI declined extra compute".

## 8. Limits and not-done

* `picks_changed_*` fields in `lanedose_rlval.json` count over all 3,000 sets, not the half named in the block (HELD-only: 566 / 555 picks differ from base0 for R0 / R0d `L0.25`, `sensitivity_heldseed.json`). The ratio-estimator CIs of `retention_of_B` contain NaN ends where B's own delta is near 0 (TTC, NC) and are unusable; PDMS retention on HELD: R0 0.54 [-0.32, +2.52], R0d 0.39 [+0.05, +0.91].
* Three arms read the same HELD logs: agreement among them is not replication. No inference variance exists (deterministic scorer); training variance is untested.
* Re-running needs the `pk/` layout (`eval/`, `refe/`, `raw/2026-10-14-rls1b-attribution/`) beside `rlh.py` and the dump paths in the PREREG (`C:/Users/Admin/rls1b_dl`, `C:/Users/Admin/par18/rl/r0d_dl`).

## 9. Deliverable manifest

All files staged under `C:/Users/Admin/par18/rl2/out/` (the one place; nothing is in git, a pod or a worktree). Repo target prefix `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-rl-lanemix/`; `out/paths.json` lists every `[repo path, source, "raw"]` row.

| artifact | local path (single copy) |
|---|---|
| RESULT.md (this file) | `C:/Users/Admin/par18/rl2/out/RESULT.md` |
| PREREG + hash | `.../out/raw/PREREG_RL2_LANEDOSE.md`, `.../out/raw/PREREG_RL2_LANEDOSE.sha256` |
| lane-dose results (3 arms, FIT, HELD, descriptive, controls) | `.../out/raw/lanedose_rlval.json` |
| R0d extension of V1/V2 | `.../out/raw/composites_r0d_ext.json`, `.../out/raw/composites_ext.diff` |
| outcome-free tau derivation | `.../out/raw/stage0_taus.json` |
| post-verdict seed sensitivity | `.../out/raw/sensitivity_heldseed.json` |
| run logs | `.../out/raw/logs/ext.log`, `.../out/raw/logs/lane.log` |
| code | `.../out/raw/code/{lanedose.py, composites_ext.py, stage0_taus.py, sens.py, rlh.py}` |
