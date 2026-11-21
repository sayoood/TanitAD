# PREREG RL-1 / A: where does RL's TTC loss live? (mechanism read, banked RL-val states, zero training, CPU)

Written 2026-10-18 by the RL-1 research agent, BEFORE any drift, pick-flip, margin or share number of this package was computed.
Registered jointly with its sha256 (file `PREREG_RL1_A_MECHANISM.sha256`). The composites (step 2) get their OWN prereg (`PREREG_RL1_B_COMPOSITES.md`), written after this read.

## 0. Disclosures (what the author already knew when writing this)

* INHERITED, read before writing: `raw/2026-10-14-rls1b-attribution/REPORT.md` section 3 and `attribution.json` Q5 (hybrids H1-H4 at R0 u1000 vs C0-A: H1 = safety from R0-A + progress {EP, C, lane} from R0-B reads TTC -0.80 [-1.51, -0.14], PDMS +0.95; H4 (reverse) TTC -0.23 [-0.77, +0.28]), and `D-REFE-SAFEHEAD-1` A2 (a separate safety head removes 30-49 % of R0's TTC loss, bar -0.5 met 0 of 3). I did NOT see any per-head drift, pick-pair margin decomposition, or any R0b number of this package.
* Prediction committed from that prior: **MECH-PROGRESS** at R0 u1000 (the flip lives mostly in EP/C/lane, not in NC/DAC/TTC). It is a prediction, not a bar.
* R0d is NOT in this read: its score dumps exist only on the pod and on private HF (`Sayood/tanitad-refe-rls1-r0d`, `final/scores_update{250,500,750,1000}_{A,B}.npz`, 8 x 11.5 MB); the dev box holds only its summary JSONs. A download needs the caller's go-ahead; the same script runs on it unchanged (`--arm r0d --dl <dir>`). Named blocker, not a silent narrowing.

## 1. Data, tier, estimator

* Tier: **RL-val selection tier** (the true NAVSIM v1.1 sub-scores of the scorer's argmax pick on 3,000 held-out RL-val sets / 75 logs; never an endpoint, not T1). Evidence class MEASURED (banked live bf16 logits as dumped by the trainer; labels `pdm_val` + `terms_val`, sha256 as `attrib_rls1b.EXPECT`).
* States: R0 u250 / u500 / u750 / u1000 and R0b u250 / u500 (R0b was K3-killed after u500), each arm B (EPO on) vs its OWN arm A (same run, BCE only) and vs `scores_step0_base` (the deployed scorer, identical across arms: verified bit-equal r0 = r0b = c0).
* Dump channels [N, 64, 7]: 0 NC, 1 DAC, 2 EP, 3 TTC, 4 C, 5 DDC (not in the rule), 6 lane. Rule `navsim_v1_lane`: S = sigmoid(NC) sigmoid(DAC) (5 EP + 5 TTC + 2 C)/12 sigmoid(lane); ties -> lowest index.
* Estimator: paired log-cluster bootstrap, 10,000 resamples, 95 % percentile, seed 20260927 (`rlh.Boot`, identity-checked against `proxy_eval.boot`: max abs diff 0.0). It answers "another draw of LOGS" only; one training seed per arm, no replicate, so nothing here is a lever-effect claim (H-ESTIM-SEED-1).

## 2. Quantities (all fixed here)

* **M1 drift, per head h, per state, vs ref in {own A, base0}:** (a) mean logit drift mean(L_B - L_ref); (b) within-scene-centred rms drift sd_w(L_B - L_ref); (c) relative drift (b) / sd_w(L_ref); (d) mean abs probability drift. DESCRIPTIVE (no CI).
* **M1b discrimination drift:** within-scene AUC of NC / DAC / TTC / EP>=0.99 logit against its pass label, B minus ref, paired log-cluster ratio bootstrap.
* **M2 pick sets (ref = own A, and ref = base0):** W_ttc = sets where B's pick fails TTC (label < 1) and ref's pick passes; Z_ttc = the reverse; W_nc, Z_nc likewise for NC. Counts reported.
* **M3 head outputs at the picks, on W_ttc (TTC head) and W_nc (NC head):** p_B and p_ref and p_base0 of the head at B's pick i_B; their differences (B - ref) with paired log-cluster CIs; its sign says OVERCONFIDENT (CI > 0: B gives a failing pick a higher pass probability than ref), NOT-OVERCONFIDENT (CI < 0), UNRESOLVED (spans 0). Also the EP-head lift at i_B and at ref's pick i_ref (B - ref).
* **M4 flip decomposition (the primary quantity):** on W_ttc take the pair (i_B, i_ref); margin m(x) = S(x[i_B]) - S(x[i_ref]). For the 7 channels let a coalition C take its logits from B and the rest from ref; m(none) = m_ref <= 0 (ref prefers its own pick) and m(all) = m_B >= 0. The exact Shapley value phi_h of each channel (all 128 coalitions) sums to m_B - m_ref (**efficiency, checked to 1e-12 on every set**). Group shares over the set: safety = {NC, DAC, TTC}, progress = {EP, C}, other = {lane, DDC}; share_g = sum_sets sum_{h in g} phi_h / sum_sets (m_B - m_ref), a ratio estimator with the log-cluster bootstrap. Per-head shares reported too. Computed on W_ttc (primary), Z_ttc and W_nc (secondary).

## 3. Controls (each must pass or the read is INVALID; reported in `mechanism.json` `controls`)

1. Zero control: B := ref gives all drift exactly 0.0 and |W| = 0.
2. Shapley efficiency to 1e-12 on every set of every state; and exactness of the coalition endpoints (m(none) = ref margin, m(all) = B margin, recomputed independently).
3. Planted: take the real A arrays, raise only the EP logit of one proposal per set so that it overtakes A's pick -> M4 must give share_progress = 1.0 and share_safety = 0.0 (to 1e-9); the same with only the TTC logit lowered on A's pick -> share_safety = 1.0.
4. Identity to banked numbers: pick counts of R0 u1000 vs C0-A (changed 2,329; TTC-worse 59; TTC-better 29; NC-worse 14, `attribution.json` Q1) reproduced exactly; `PDMS(C0A u1000) = 90.05977`, `PDMS(R0B u1000) = 91.22299`; the AUC of NC at R0 u1000 reproduces `attribution.json` Q2.
5. Bootstrap identity: `rlh.Boot` equals `proxy_eval.boot` (done: 0.0).

## 4. Outcomes (committed before the numbers), read at the PRIMARY state R0 u1000 vs own A, M4 on W_ttc

| label | rule | what it means | committed next |
|---|---|---|---|
| **MECH-SAFETY** | share_safety CI lower bound > 0.5 | the TTC flip lives in the safety heads' drift; a head-wise composite with anchored safety heads is premised correctly | register the brief's two composites, SRC = the source whose safety heads M3 shows least drift |
| **MECH-PROGRESS** | share_safety CI upper bound < 0.5 | the flip lives in the progress heads (EP/C) and the lane gate; swapping safety heads cannot remove it | still register two composites (the brief asks), but predicted to FAIL on TTC; name the progress-side lever for the next arm (anchor the EP/C head's lift on TTC-failing plans) |
| **MECH-MIXED** | CI contains 0.5 | both sides carry it | register the composites with the progress side restricted to the head set M4 ranks highest |

Consistency over the other states (R0 u250 / u500 / u750, R0b u250 / u500) is reported per state with the same rule and not pooled. If the primary label differs from at least half of them, the headline says "state-dependent".
The question "is the RL safety head overconfident on those proposals" is answered by M3's sign at R0 u1000, TTC head, ref = own A and ref = base0, separately.

## 5. What this prereg does not do

It evaluates no composite pick against labels (no head-swap TTC delta), opens no navtest, and fits nothing. Everything that selects (which heads, which source) is decided in `PREREG_RL1_B` on RL-val only.
