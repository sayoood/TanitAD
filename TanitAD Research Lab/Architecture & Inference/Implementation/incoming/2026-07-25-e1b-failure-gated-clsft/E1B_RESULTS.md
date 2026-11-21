# E1b — RESULTS: failure-gated closed-loop SFT, paired verdict

**VERDICT: `BOUND`** — the pre-registered primary **fired hard** (junction
corridor-departure@K185 CI-separated LOWER, paired Δ **−0.4270 [−0.6838,
−0.1648]**), but **guardrail (a) regressed CI-separated-worse** (open-loop ADE@2s
**+0.1947 [+0.1415, +0.2522]**). PRE_REGISTRATION §3 commits that combination to
**BOUND/FAILURE, not SUCCESS**, and it is reported as such. Nothing was re-tuned,
re-thresholded or re-selected.

`2026-07-25/26 (Europe/Berlin; pod logs are UTC)` · `tanitad-pod3` (A40) ·
renderer-free imagination/kinematic closed loop (NOT AlpaSim). PI: Sayed.
Eval ran `2026-07-25T22:18:42Z → 23:06:42Z` (rc=0, 2874.9 s).

Evidence-class legend: `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) ·
`INHERITED` · `ESTIMATED` · `HYPOTHESIS`. Every number below is `MEASURED` from
`e1b_eval_result.json` in this directory and was **generated** into these tables,
never re-typed.

---

## 1. The pre-registration, restated BEFORE any number (PRE_REGISTRATION.md §3)

Both outcomes were committed in advance, staged before the fine-tune launched.

**Primary metric.** Junction corridor-departure-rate **@ K=185** (18.5 s) on the
held-out 44 episodes (E1a's exact eval set), **paired episode-cluster bootstrap**
(`taniteval/ci.py`, B=2000, resampling episodes).

- **SUCCESS.** FT junction departure@K185 **CI-separated LOWER** (paired
  Δ `hi < 0`) **AND** the guardrails hold: (a) open-loop **ADE@2s** Δ CI
  **includes 0 or better**; (b) the open-loop **anchor block** unchanged.
- **BOUND / FAILURE (equally publishable).** Not CI-separated, **OR** open-loop
  ADE@2s **regresses CI-separated-worse** — *"the replay branch is designed to
  prevent exactly this, so its failure is informative about the objective's
  tension."*

**What happened: the second clause fired while the first also fired.** The
pre-registration anticipated this exact trade and pre-committed the label.

**Estimator discipline.** Every interval is the episode-cluster bootstrap, paired
for two-arm deltas. `overlapping_holdout_se` is used **nowhere**.

---

## 2. What ran (all `MEASURED`)

| item | value |
|---|---|
| base | `refc-diffusion-base-v21-30k/ckpt.pt`, step **29999**, 128 anchors `[128,4,2]` |
| FT | `/workspace/e1b/refc-base-e1b-clsft/ckpt.pt`, step **3999**, `metrics.json {"done": true}` |
| FT trainable / frozen | **13,732,945** / 90,458,632 — encoder FROZEN (E2a: perception is not the bottleneck) |
| FT hyper-params (as launched) | lr 2e-5, warmup 100, cosine, 4000 steps, cl-batch **16** / replay-batch **16**, `lam_cl = lam_replay = 1.0`, 2 denoise steps |
| mined buffer | **3,537** recoverable pre-failure states from **362** distinct parity-train episodes (400 rolled, 401 windows, 373 departing); \|XTE\| at mining 0.303–1.750 m (mean 1.275) |
| mining source | `physicalai-train-e438721ae894` — the sacred parity corpus |
| eval set | `physicalai-val-heldout-79d4e3d2d4c6` — **44** episodes, E1a's exact set |
| leak guard | mined-episode ids ∩ held-out ids = **0**, re-asserted at train startup (`--assert-disjoint-heldout` refuses to train on overlap) |
| wall | mine 8,288 s + CL-SFT 4,342 s + paired eval 2,875 s |

### 2.1 Built-in control: the base arm REPRODUCES E1a exactly

Both arms were re-rolled in this session (E1a never persisted per-window arrays),
which is what makes the bootstrap paired — and it doubles as a reproduction test.
The base arm returned, on all 44 held-out episodes:

| K | overall CDR | junction CDR | peak \|XTE\| | OOD peak | n |
|---|---|---|---|---|---|
| 20 | **0.0053** | **0.0395** | **0.368** | **1.022** | 967 w |
| 185 | **0.5877** | **0.8414** | **38.944** | **1.266** | 43 w |

Every figure is **bit-identical** to `e1a_horizon_heldout44_K185.json`. The
rollout is deterministic, the additive capture patch changed nothing, and the
open-loop canary came back at **0.4747 [0.4029, 0.5528]** against the registry's
canonical-val REF-C base **0.4728 [0.3835, 0.5699]** (MODEL_REGISTRY §4.3).

### 2.2 The CL-SFT training curve (`ft_run/clsft_train_log.jsonl`, banked)

| step block | cl_cls | cl_traj | **cl_anchor_acc** | rp_loss | rp_traj | rp_anchor_acc |
|---|---|---|---|---|---|---|
| 0–250 | 5.002 | 1.128 | **0.131** | 1.826 | 0.300 | 0.550 |
| 250–1000 | 2.236 | 0.728 | **0.285** | 1.957 | 0.350 | 0.517 |
| 1k–2k | 1.793 | 0.561 | **0.395** | 1.711 | 0.292 | 0.567 |
| 2k–3k | 1.496 | 0.473 | **0.473** | 1.631 | 0.265 | 0.570 |
| 3k–4k | 1.391 | 0.455 | **0.511** | 1.613 | 0.275 | 0.584 |

**This is the mechanism of the BOUND.** The score head learned to rank the
*return* anchor (0.131 → 0.511 on mined failure states) **and the replay loss
FELL on parity-train** (1.83 → 1.61, rp_anchor_acc 0.55 → 0.58). By its own
training-set instrument the forgetting guard looked like it was working — while
**held-out** open-loop ADE rose 41 % and held-out anchor accuracy fell. A
training-set replay loss is not a held-out generalisation claim, and this run is
the counter-example.

---

## 3. PRIMARY VERDICT — junction corridor-departure @ K=185

| arm | rate | n |
|---|---|---|
| base | 0.8414 [0.8144, 0.8667] | 6w/6ep |
| FT (E1b CL-SFT) | 0.4144 [0.1486, 0.6919] | 6w/6ep |
| **paired Δ(FT−base)** | **−0.4270 [−0.6838, −0.1648] SEPARATED** (`p_delta_gt0` = 0.0) | 6w/6ep |

**The primary fired: CI-separated LOWER.** Estimator: paired episode-cluster
bootstrap, B=2000, over the 6 junction episodes. *Read §9 on the n=6 power bound
before quoting this alone — the overall and longitudinal strata below carry 43
and 19 clusters and say the same thing far more strongly.*

---

## 4. Strata

### Closed loop @ K=185 (18.5 s)

| stratum | metric | base | FT | paired Δ(FT−base) | n |
|---|---|---|---|---|---|
| overall | corridor-departure rate | 0.5877 [0.5107, 0.6622] | 0.1603 [0.0855, 0.2493] | −0.4274 [−0.5161, −0.3378] **SEP** | 43w/43ep |
| overall | window departure rate | 0.9302 [0.8605, 1.0000] | 0.4419 [0.3023, 0.5814] | −0.4884 [−0.6512, −0.3023] **SEP** | 43w/43ep |
| overall | peak \|XTE\| (m) | 38.9445 [27.0163, 52.6962] | 3.0415 [2.0496, 4.2845] | −35.9030 [−49.3302, −24.1241] **SEP** | 43w/43ep |
| overall | mean \|XTE\| (m) | 14.3062 [9.8386, 19.2428] | 1.3908 [0.8508, 2.0630] | −12.9153 [−17.6887, −8.5585] **SEP** | 43w/43ep |
| overall | peak \|Δψ\| (deg) | 25.0635 [20.4211, 29.9424] | 13.4647 [9.5194, 17.9761] | −11.5988 [−15.1709, −8.2172] **SEP** | 43w/43ep |
| overall | closed ADE@2s (m) | 0.4957 [0.3749, 0.6434] | 0.6363 [0.4988, 0.8020] | **+0.1406 [+0.0423, +0.2738] SEP (worse)** | 43w/43ep |
| overall | OOD peak ratio | 1.2664 [1.2422, 1.2880] | 1.1339 [1.1060, 1.1625] | −0.1325 [−0.1640, −0.0981] **SEP** | 43w/43ep |
| overall | OOD mean ratio | 1.1583 [1.1365, 1.1796] | 1.0559 [1.0356, 1.0803] | −0.1024 [−0.1261, −0.0793] **SEP** | 43w/43ep |
| overall | frac windows out of envelope | 0.9070 [0.8140, 0.9767] | 0.2558 [0.1395, 0.3953] | −0.6512 [−0.7907, −0.4884] **SEP** | 43w/43ep |
| junction | corridor-departure rate | 0.8414 [0.8144, 0.8667] | 0.4144 [0.1486, 0.6919] | **−0.4270 [−0.6838, −0.1648] SEP** | 6w/6ep |
| junction | window departure rate | 1.0000 [1.0000, 1.0000] | 0.8333 [0.5000, 1.0000] | −0.1667 [−0.5000, +0.0000] not sep | 6w/6ep |
| junction | peak \|XTE\| (m) | 46.2475 [24.4878, 68.7290] | 7.0027 [2.6065, 12.0766] | −39.2447 [−63.6992, −18.5090] **SEP** | 6w/6ep |
| junction | mean \|XTE\| (m) | 21.4716 [12.1178, 31.8233] | 3.4698 [1.0387, 6.3100] | −18.0018 [−29.2782, −9.0686] **SEP** | 6w/6ep |
| junction | peak \|Δψ\| (deg) | 44.7871 [37.9884, 51.7735] | 27.1022 [11.6881, 43.7914] | −17.6848 [−28.3425, −7.1961] **SEP** | 6w/6ep |
| junction | closed ADE@2s (m) | 0.7318 [0.5102, 1.0078] | 0.5961 [0.4789, 0.7149] | −0.1357 [−0.4385, +0.0620] not sep | 6w/6ep |
| junction | OOD peak ratio | 1.2989 [1.2989, 1.2989] | 1.2018 [1.1198, 1.2768] | −0.0971 [−0.1791, −0.0221] **SEP** | 6w/6ep |
| junction | OOD mean ratio | 1.2442 [1.2282, 1.2572] | 1.1305 [1.0529, 1.2135] | −0.1137 [−0.1854, −0.0403] **SEP** | 6w/6ep |
| junction | frac windows out of envelope | 1.0000 [1.0000, 1.0000] | 0.5000 [0.1667, 0.8333] | −0.5000 [−0.8333, −0.1667] **SEP** | 6w/6ep |
| longitudinal | corridor-departure rate | 0.6654 [0.5613, 0.7491] | 0.0990 [0.0137, 0.2174] | **−0.5664 [−0.6845, −0.4267] SEP** | 19w/19ep |
| longitudinal | window departure rate | 1.0000 [1.0000, 1.0000] | 0.3684 [0.1579, 0.5789] | −0.6316 [−0.8421, −0.4211] **SEP** | 19w/19ep |
| longitudinal | peak \|XTE\| (m) | 56.3890 [34.8915, 79.2986] | 2.6024 [1.3844, 4.3745] | −53.7866 [−76.5947, −32.8202] **SEP** | 19w/19ep |
| longitudinal | mean \|XTE\| (m) | 19.6665 [12.0722, 28.1526] | 1.1765 [0.5117, 2.1538] | −18.4900 [−26.7812, −11.1481] **SEP** | 19w/19ep |
| longitudinal | peak \|Δψ\| (deg) | 21.0409 [14.9175, 27.1862] | 7.9166 [4.7181, 12.2345] | −13.1242 [−18.6587, −7.4340] **SEP** | 19w/19ep |
| longitudinal | closed ADE@2s (m) | 0.5165 [0.2917, 0.8080] | 0.7507 [0.4771, 1.0836] | **+0.2343 [+0.0673, +0.4949] SEP (worse)** | 19w/19ep |
| longitudinal | OOD peak ratio | 1.2712 [1.2402, 1.2927] | 1.1065 [1.0721, 1.1449] | −0.1647 [−0.2026, −0.1237] **SEP** | 19w/19ep |
| longitudinal | OOD mean ratio | 1.1673 [1.1359, 1.1948] | 1.0310 [1.0111, 1.0597] | −0.1363 [−0.1671, −0.1028] **SEP** | 19w/19ep |
| longitudinal | frac windows out of envelope | 0.9474 [0.8421, 1.0000] | 0.1579 [0.0000, 0.3158] | −0.7895 [−0.9474, −0.5789] **SEP** | 19w/19ep |

**Read.** The long-horizon corridor-keeping effect is large, separated in **every
stratum**, and *not* a junction-only artefact — the longitudinal stratum improves
most (0.6654 → 0.0990). Peak \|XTE\| collapses from **38.94 m to 3.04 m** overall
and **56.39 m to 2.60 m** longitudinally. This is the biggest closed-loop
movement the program has measured.

**But the honest nuance:** at junctions the *window* departure rate is
**1.0000 → 0.8333, not separated**. The FT still leaves the corridor at some
point in 5 of 6 junction windows. What collapsed is **how long** it stays out and
**how far** it goes — not whether it departs at all.

### Closed loop @ K=20 (2.0 s) — the standing instrument

| stratum | metric | base | FT | paired Δ(FT−base) | n |
|---|---|---|---|---|---|
| overall | corridor-departure rate | 0.0053 [0.0018, 0.0096] | 0.0068 [0.0023, 0.0127] | +0.0015 [−0.0019, +0.0066] not sep | 967w/44ep |
| overall | window departure rate | 0.0352 [0.0124, 0.0663] | 0.0372 [0.0166, 0.0620] | +0.0021 [−0.0165, +0.0238] not sep | 967w/44ep |
| overall | peak \|XTE\| (m) | 0.3683 [0.2867, 0.4591] | 0.5176 [0.4481, 0.5965] | **+0.1493 [+0.0885, +0.2112] SEP (worse)** | 967w/44ep |
| overall | closed ADE@2s (m) | 0.5227 [0.4456, 0.6076] | 0.6238 [0.5505, 0.6991] | **+0.1012 [+0.0619, +0.1451] SEP (worse)** | 967w/44ep |
| junction | corridor-departure rate | 0.0395 [0.0178, 0.0676] | 0.0359 [0.0152, 0.0609] | −0.0036 [−0.0194, +0.0083] not sep | 124w/18ep |
| junction | window departure rate | 0.2661 [0.1182, 0.4386] | 0.2016 [0.1065, 0.3077] | −0.0645 [−0.1532, +0.0081] not sep | 124w/18ep |
| junction | peak \|XTE\| (m) | 1.2287 [1.0061, 1.4792] | 1.0738 [0.8478, 1.3466] | −0.1549 [−0.3006, −0.0302] **SEP (better)** | 124w/18ep |
| junction | closed ADE@2s (m) | 0.9454 [0.7499, 1.2417] | 1.0093 [0.7709, 1.3891] | +0.0639 [+0.0052, +0.1504] **SEP (worse)** | 124w/18ep |
| longitudinal | corridor-departure rate | 0.0000 [0.0000, 0.0000] | 0.0047 [0.0000, 0.0129] | +0.0047 [+0.0000, +0.0129] not sep | 457w/27ep |
| longitudinal | window departure rate | 0.0000 [0.0000, 0.0000] | 0.0241 [0.0000, 0.0581] | +0.0241 [+0.0000, +0.0581] not sep | 457w/27ep |
| longitudinal | peak \|XTE\| (m) | 0.3471 [0.2640, 0.4382] | 0.5564 [0.4653, 0.6540] | **+0.2093 [+0.1118, +0.3092] SEP (worse)** | 457w/27ep |
| longitudinal | closed ADE@2s (m) | 0.4330 [0.3418, 0.5440] | 0.6056 [0.5139, 0.7008] | **+0.1726 [+0.1171, +0.2312] SEP (worse)** | 457w/27ep |

**Stated explicitly, per PRE_REGISTRATION §3's "no hidden trade" clause:** the
2 s instrument sees this fine-tune as **neutral-to-mildly-WORSE** (departure
unchanged, peak \|XTE\| and closed ADE@2s both CI-separated worse). Had E1b been
judged on the standing 2 s harness — as every previous closed-loop decision in
this program was — it would have been rejected. E1a's finding cuts both ways: the
2 s instrument hid a 170× failure, **and** it hides a 4× recovery.

---

## 5. Guardrails

### (a) open-loop ADE@2s — **FAIL** (the BOUND trigger)

| arm | ADE@2s (m) |
|---|---|
| base | 0.4747 [0.4029, 0.5528] (967w/44ep) |
| FT | 0.6693 [0.5773, 0.7687] (967w/44ep) |
| **paired Δ(FT−base)** | **+0.1947 [+0.1415, +0.2522] SEPARATED WORSE** |

+41 % open-loop ADE. The pre-registered pass rule is *"CI includes 0 or better"*;
this CI excludes 0 on the worse side. **Guardrail (a) fails.**

### (b) open-loop anchor block (the REF-C stand-in for a WM canary) — **FAIL**

| metric | base | FT | paired Δ(FT−base) |
|---|---|---|---|
| anchor_acc | 0.6815 [0.6267, 0.7381] | 0.6163 [0.5553, 0.6763] | −0.0651 [−0.0961, −0.0352] **SEP (worse)** |
| anchor_ce | 0.8757 [0.7367, 1.0261] | 1.1637 [0.9858, 1.3563] | +0.2880 [+0.2011, +0.3789] **SEP (worse)** |
| anchor_traj_l1 | 0.1775 [0.1594, 0.1975] | 0.2399 [0.2174, 0.2640] | +0.0624 [+0.0500, +0.0747] **SEP (worse)** |

The CL-SFT re-purposed exactly this head, and exactly this head degraded on
held-out open-loop. The intervention is **specific**, not diffuse damage.

### (c) OOD envelope — **PASS**, and it moved the *right* way

| metric @K185 overall | base | FT | ratio FT/base |
|---|---|---|---|
| OOD peak ratio | 1.2664 | **1.1339** | **0.8954** |
| OOD mean ratio | 1.1583 | 1.0559 | 0.912 |
| frac windows any step out of measured envelope | 0.9070 | **0.2558** | 0.282 |

**The improvement is not confounded by distribution shift.** The FT sits *more*
in-distribution than base (1.134 vs 1.266, both ≤ the ~1.30 band), and the
fraction of windows leaving the MEASURED P1 envelope drops from 91 % to 26 %.
That also **strengthens** the closed-loop numbers: base's K=185 figures were
partly extrapolation (91 % of windows outside the measured envelope, where the
OOD ratio is only a lower bound); the FT's are far more nearly measurement.

### Guardrail summary (machine-emitted)

```json
{
  "a_openloop_ade2s_ok": false,
  "b_anchor_acc_ok": false,
  "b_anchor_traj_l1_ok": false,
  "c_ood_in_band": true,
  "c_ood_ft": 1.1339,
  "c_ood_base": 1.2664,
  "c_ood_ratio_ft_over_base": 0.8954,
  "all_ok": false
}
```

---

## 6. M1 lateral / longitudinal decomposition (`taniteval/lateral.py`)

No ADE is reported without its (lat, lon) split. Both frames emitted; GT identity
check `max|Δ| = 0.0` (the two arms really are on the same windows and the same
ground truth); axis convention **verified** on both blocks.

### Open loop (967 windows, 44 episodes)

| mode | metric | base | FT | paired Δ(FT−base) |
|---|---|---|---|---|
| ego | ADE over knots | 0.4747 [0.4029, 0.5528] | 0.6693 [0.5773, 0.7687] | +0.1947 [+0.1415, +0.2522] **SEP** |
| ego | cross_abs@2s | 0.2494 [0.2093, 0.2899] | 0.4105 [0.3349, 0.4840] | +0.1611 [+0.1126, +0.2097] **SEP** |
| ego | cross_p90@2s | 0.6025 [0.4899, 0.6818] | 0.9970 [0.7423, 1.2033] | +0.3945 [+0.1876, +0.5577] **SEP** |
| ego | along_abs@2s | 0.8974 [0.7476, 1.0628] | 1.0812 [0.8978, 1.2895] | +0.1838 [+0.1090, +0.2698] **SEP** |
| frenet | cross_abs@2s | 0.2343 [0.1981, 0.2690] | 0.3900 [0.3210, 0.4591] | +0.1557 [+0.1089, +0.2051] **SEP** |
| frenet | cross_p90@2s | 0.5777 [0.4714, 0.6665] | 0.9248 [0.7346, 1.0727] | +0.3471 [+0.2147, +0.4742] **SEP** |
| frenet | along_abs@2s | 0.9039 [0.7501, 1.0736] | 1.0900 [0.9021, 1.3009] | +0.1861 [+0.1110, +0.2718] **SEP** |

Squared-error energy share (ego): base **lon 0.9112 / lat 0.0888** → FT
**lon 0.7947 / lat 0.2053**. Frenet: 0.9353/0.0647 → 0.8120/0.1880.

\|XTE\|@2s tail (ego): base mean 0.2494, p90 0.6025, p99 1.5482, max 4.4698,
frac>1.75 m **0.0083** → FT mean 0.4105, p90 0.9970, p99 3.7354, max 6.0653,
frac>1.75 m **0.0434**.

**This is the finding the undecomposed ADE hides.** In absolute metres the
open-loop regression looks longitudinal (+0.184 along vs +0.161 cross), but
*proportionally* the cross-track channel degraded **+65 %** against the
longitudinal **+20 %**, the lateral energy share more than **doubled**, and the
rate of open-loop windows more than 1.75 m off cross-track went up **5.2×**.
**The CL-SFT bought closed-loop lateral behaviour by paying in open-loop lateral
accuracy — the same axis, opposite sign, at the two horizons.**

### Closed loop, the 2 s knots inside the K=185 rollout (43 windows)

| mode | metric | base | FT | paired Δ(FT−base) |
|---|---|---|---|---|
| ego | ADE over knots | 0.4957 [0.3749, 0.6434] | 0.6363 [0.4988, 0.8020] | +0.1406 [+0.0423, +0.2738] **SEP** |
| ego | cross_abs@2s | 0.3763 [0.2482, 0.5316] | 0.5917 [0.4083, 0.8485] | +0.2154 [+0.0038, +0.4815] **SEP** |
| ego | cross_p90@2s | 0.8994 [0.6201, 1.4452] | 1.0955 [0.7189, 1.8257] | +0.1961 [−0.3202, +0.9815] not sep |
| ego | along_abs@2s | 0.8985 [0.6382, 1.2048] | 0.9628 [0.7108, 1.2620] | +0.0643 [−0.0881, +0.1992] not sep |
| frenet | cross_abs@2s | 0.3889 [0.2617, 0.5461] | 0.5816 [0.4024, 0.8323] | +0.1927 [−0.0240, +0.4553] not sep |
| frenet | along_abs@2s | 0.8962 [0.6408, 1.2076] | 0.9816 [0.7313, 1.2759] | +0.0853 [−0.0386, +0.2070] not sep |

Energy share (ego): base lon 0.8341 / lat 0.1659 → FT lon 0.6358 / lat 0.3642.

**Read:** inside the closed loop the 2 s *tracking* error also degrades on the
cross-track axis while the along-track delta is **not separated** — i.e. the
short-horizon cost is lateral too. The FT trades **short-horizon lateral
precision** for **long-horizon lateral containment**. That is a coherent
mechanism, not noise: a score head pushed toward "return to corridor" biases
every short-horizon plan toward the corridor centre, which is wrong at 2 s and
right at 18.5 s.

---

## 7. The baseline this was read against (E1a, `MEASURED`, reproduced above)

`…/2026-07-25-closedloop-horizon-and-shift/e1a_horizon_heldout44_K185.json`.
At K=185 the window set **is** the common set (43 windows, one per surviving
episode), so all-windows and common-start agree.

| stratum | K=20 CDR | K=185 CDR | peak \|XTE\| @K185 | OOD-peak @K185 | n @K185 |
|---|---|---|---|---|---|
| overall | 0.0035 (common) / 0.0053 (all 967 w) | 0.5877 [0.5107, 0.6622] | 38.94 m | 1.2664 | 43 w / 43 ep |
| junction | 0.0250 (common) / 0.0395 (all 124 w) | 0.8414 [0.8144, 0.8667] | 46.25 m | 1.2989 | **6 w / 6 ep** |
| longitudinal | 0.0000 | 0.6654 [0.5613, 0.7491] | 56.39 m | 1.2712 | 19 w / 19 ep |

---

## 8. Method changes to `e1b_eval.py` made for this run (declared, not silent)

The shipped evaluator did not cover four things this verdict requires. Each
change is additive; none touches the pre-registered primary or its estimator.

1. **Open-loop guardrail (a) is now PAIRED.** The shipped version reported two
   single-arm intervals and carried the note *"overlapping CIs ⇒ no CI-separated
   open-loop regression (guardrail a PASS)"*. That inference is **invalid in
   general**: overlapping single-arm intervals do not imply a paired delta that
   includes zero, because the two arms are scored on the same windows and are not
   independent. *Accuracy check on this run, since the claim matters:* here the
   single-arm CIs happen **not** to overlap ([0.4029, 0.5528] vs [0.5773,
   0.7687]), so the old heuristic would also have flagged a problem — the
   substantive defect is different and worse: **the shipped verdict logic never
   consulted the guardrails at all.** It emitted
   `"SUCCESS: … CI-separated LOWER for FT (paired). Check guardrails."` purely
   from the primary, leaving the actual pass/fail to a human reading a note. This
   run would have been written up as a SUCCESS. The verdict string now consumes
   an explicit machine-emitted `GUARDRAIL_SUMMARY` and downgrades to BOUND
   itself.
2. **Guardrail (b) implemented** — open-loop anchor-cls accuracy / CE /
   traj-recon L1 on held-out, i.e. `refc_train.compute_losses`' anchor block
   (lines 268–276) with the GT target: exactly the block the CL-SFT re-purposed.
   PRE_REGISTRATION §3 names this the REF-C stand-in for a world-model canary
   (REF-C is a direct anchored-diffusion planner with no operative imagination
   rollout to canary).
3. **Peak \|XTE\|, peak \|Δψ\|, OOD-envelope ratio and out-of-envelope fraction**
   added per arm and paired.
4. **M1 lateral/longitudinal split** of every reported ADE via
   `taniteval/lateral.py`, both `ego` and `frenet`.

To compute (4) on the closed loop, `e1a_horizon.rollout` must retain the
predicted and GT 2 s knots. **The E1a source at `/workspace/e1a_e2a/` is not
mutated:** `make_capture_rollout.py` builds `/workspace/e1b/e1a_horizon.py` by
exact string replacement and asserts the patch is additive — it printed
**`0 removed lines, +5 added`**, all five being `.append(...)` / `out[...] =`
statements. Independent confirmation: §2.1, the base arm reproduces every E1a
figure bit-for-bit. The evaluator fails loud at import if the unpatched module
wins the `sys.path` race (it did on the first attempt — the working directory is
already `sys.path[0]`, so the later inserts jumped ahead of the capture copy; a
silent fallback would have dropped §6 from this report without a word).

**Estimator provenance.** `taniteval_ci.py` on pod3 is md5
`ef925f06febd20a99f5901491fcf75cb` — byte-identical to the repo's
`taniteval/taniteval/ci.py`. `taniteval_lateral.py` is md5
`897938ae40b6cb2dfa51802f0ec260b9` — byte-identical to the repo's
`taniteval/taniteval/lateral.py`, vendored unmodified. `overlapping_holdout_se`
appears nowhere in the E1b pipeline.

---

## 9. Honest bounds

- **The primary stratum is 6 windows in 6 episodes.** An episode-cluster
  bootstrap over 6 clusters is low-powered by construction. It separated anyway,
  and the overall (43 clusters) and longitudinal (19 clusters) strata separate
  far more strongly in the same direction — but the *headline primary number*
  rests on 6 clusters and must never be quoted without that.
- The closed loop is **map/agent-free** — this measures drift / corridor-keeping,
  not collision or off-road safety. A corridor-keeping win is not a
  certified-safety claim.
- K=185 is the **structural ceiling** on this 190–199-frame corpus (E1a §1.4).
- The P1 OOD envelope was MEASURED only to \|dlat\| ≤ 3.0 m / \|dyaw\| ≤ 12°;
  `np.interp` clamps beyond, so reported OOD is a **lower bound** at long
  horizons. Base had 91 % of K=185 windows outside that envelope (so base's
  38.94 m peak \|XTE\| is partly extrapolation); the FT has 26 %.
- The recovery target is a **kinematic demonstration** (logged corridor in the
  offset ego frame), not a renderer rollout.
- **Not measured here:** whether the open-loop cost is recoverable (replay weight,
  replay-on-held-out-distribution, shorter fine-tune, LoRA-style constraint), and
  whether the closed-loop win survives in AlpaSim/CARLA with agents and a map.

---

## 10. What this licenses next (proposals, not conclusions)

`HYPOTHESIS` in every line below.

1. **The BOUND is a tension, not a dead end — and it is now localised.** The
   primary moved by the largest closed-loop margin the program has measured,
   with OOD moving *favourably*. The blocker is a specific, measured open-loop
   cost on a specific head. That is the cheapest kind of failure to attack.
2. **The replay branch's own loss is not a valid forgetting guard.** It fell
   while held-out open-loop rose. Any successor must gate on **held-out**
   open-loop ADE + anchor block *during* training, not on the replay loss.
   (Root-cause class for `RETRACTION_LOG.md`: *training-set instrument used as a
   generalisation guard* — sibling of "trainer val is not eval output".)
3. **Pre-register a `lam_replay` / early-stop sweep** with both outcomes
   committed: is there a point on the trade-off curve where junction
   departure@K185 stays CI-separated-lower **and** open-loop ADE's CI includes 0?
   The K=20 table says the FT is already only mildly worse at 2 s, so the curve
   may not be steep. **This must be a new pre-registration — it is explicitly
   NOT a re-tune of E1b to convert this BOUND into a SUCCESS.**
4. **E1a's lesson is now two-sided and belongs in the standing protocol.** The
   2 s instrument hid a 170× failure *and* hides a 4× recovery. No closed-loop
   verdict in this program should be issued at K=20 alone again.

---

## 11. Artifact manifest

| artifact | where |
|---|---|
| **raw paired eval JSON (the quotable source)** | `…/2026-07-25-e1b-failure-gated-clsft/e1b_eval_result.json` · pod `/workspace/e1b/e1b_eval_result.json` |
| eval run log | `…/2026-07-25-e1b-failure-gated-clsft/ft_run/e1b_eval.log` · pod `/workspace/e1b/e1b_eval.log` |
| **FT training curve (banked — was pod-only)** | `…/ft_run/clsft_train_log.jsonl` |
| FT metrics / config | `…/ft_run/clsft_metrics.json`, `…/ft_run/clsft_config.json` |
| mined-buffer meta | `…/ft_run/mined_buffer.meta.json` |
| mine + CL-SFT master log | `…/ft_run/e1b_run.log` |
| evaluator (extended, §8) | `…/scripts/e1b_eval.py` · pod `/workspace/e1b/e1b_eval.py` |
| additive-capture builder | `…/scripts/make_capture_rollout.py` · pod `/workspace/e1b/make_capture_rollout.py` (generates `/workspace/e1b/e1a_horizon.py`) |
| eval launcher | `…/scripts/run_e1b_eval.sh` · pod `/workspace/e1b/run_e1b_eval.sh` |
| table generator (every number in §3–§6) | `…/scripts/summarize_e1b.py` |
| **mined failure buffer (banked, 757 KB)** | `…/ft_run/mined_buffer.pt` — md5 `a32cfe9bfea4b1b5c196d3bb7f71fa5f`, verified equal to pod3's copy |
| FT checkpoint (527 MB, **pod only**) | `tanitad-pod3:/workspace/e1b/refc-base-e1b-clsft/ckpt.pt` |

Every code artifact and every small binary E1b produced is now in the repo
working tree. The mined buffer was banked specifically because it is the
expensive artifact to regenerate (**2.3 h** of K=185 rollout over 400
parity-train episodes) and it was living on a single disk.

**Escalation for the orchestrator (integration, not a README request):** the
**FT checkpoint (527 MB) is the one remaining single-disk artifact**. If E1b's
successor (§10.3) is authorised it must be pushed to HF from pod3 (~118 MB/s)
rather than relayed through the dev box (~1 MB/s) — pods cannot SSH each other.
Until then, a pod3 volume event loses it and costs 3.5 h of GPU to rebuild.

**Pod state at hand-off (`MEASURED`):** pod3 GPU **0 MiB / 0 %**, no `e1b`
process remaining, eval exited `rc=0`. pod1 and pod2 were never contacted.

Repo files are written into the working tree and **not** committed or pushed
(Agent Operating Standard: stage, never push — the orchestrator commits).
