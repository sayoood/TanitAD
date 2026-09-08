# WP-D PANEL RESULT — `E-BEV-AUX-1`, the frozen-feature probe on D0 / D1 / D2

**Date** 2026-09-08 (Europe/Berlin) · **Owner** Architecture & Inference FlyWheel ·
**Branch** `agent/arch-inf-20260803`
**Contract** `Project Steering/PREREG_WPD_BEV_AUX.md` §5 — every bar below was fixed in advance and
is reported as written.
**Evidence class** MEASURED (ours) on every line unless the line says otherwise; the artifact path
is named per number.

⛔ **NO EVAL TIER ON THE REPRESENTATION BARS, AND THAT IS THE PREREG'S OWN RULE, NOT AN
OMISSION.** They are a **frozen-feature probe**: no model in §§2–5 emits a trajectory, so a T0/T1 stamp would be a
category error (prereg §5A states this). The **planner** bars (**§10**) are **T1** and carry the four
families.

⛔ **NO HEADLINE NUMBER HERE IS AN ACCURACY.** Measured base rates: **0.016106801** (Cartesian) and
**0.029992479** (polar) ⇒ an all-zero predictor scores **98.3893 %** and **97.0008 %** respectively.

---

## 0. The answer

1. ⛔⛔ **`E-BEV-AUX-1` IS REFUTED ON BOTH HALVES OF ITS OWN SUCCESS CONDITION — `F4` AND `F2`
   BOTH FIRED.** ⛔ **`F2` (§10):** the planner is separably **WORSE** with the aux term on, on
   **8 of 9** T1 four-family metrics — ADE **+0.02610 [+0.00970, +0.04240]**,
   with the same sign across LONGITUDINAL and LATERAL — and the *"it must still ACT"* control
   CLEARS, so it is not the planner stopping. ⛔ **`F4` (below): the gain is
   capacity/regularisation, not agent content.** On the polar target the aux head actually trained
   on, the arm supervised with a **SHUFFLED, zero-information** target (`D2`, AP **0.1525**) is
   indistinguishable from the arm supervised with the real one (`D1`, AP **0.1527**) —
   **+0.00025 [-0.01655, +0.01647], not separated** (2,000 draws). On the Cartesian target the shuffled arm is
   *higher still* (**0.0602** vs **0.0565**; `D1 − D2` = **-0.00366
   [-0.01173, +0.00397]**). Whatever the aux term bought, the information in its
   target did not buy it. ⛔ **`A2` also fails on both geometries** (§6): the aux-ON trunk does not
   *separably* beat the raw-pixel floor. **Two of four bars fail, a third is unmeasured.**
2. ⭐⭐ **AND THE PRE-REGISTRATION'S PREMISE IS FALSE — THE MAP ALREADY CONTAINED AGENTS.** WP-D
   exists because `E-READOUT-CEILING-1` concluded *"the map does not yet contain agents, so WP-B
   built today would index into an empty room."* MEASURED here: the **aux-OFF control `D0`** reads
   **0.0548** Cartesian / **0.1434** polar against a marginal of **0.0302 / 0.0696**, and
   `tok_D0 − pos_only` is **SEPARATED on BOTH geometries** —
   **+0.02457 [+0.00956, +0.03921]** Cartesian and
   **+0.07368 [+0.02409, +0.12850]** polar.
   The room was not empty. ⚠️ **Stated with its limit:** `D0 − pix` is **+0.01095
   [-0.00267, +0.02292], NOT separated** — so the trunk is clearly above the
   MARGINAL but does **not** separably exceed what RAW PIXELS already give. That is a real and
   different finding from WP-A's, which put every trunk arm *at* the marginal.
3. ⛔⛔ **THE REASON WP-A SAW AN EMPTY ROOM IS A MIRRORED AZIMUTH ADDRESS IN ITS OWN PROBE**, and
   that is MEASURED on WP-A's own banked bytes, one variable moved: `pix` AP **0.0596** under the
   repo's convention vs **0.0284** under WP-A's — **2.10×** — and the mirrored run **reproduces
   WP-A's banked panel** (`pix_16x40` 0.0270, `tok_16x40` 0.0295) while the corrected one lifts the
   v6 trunk to **0.0457**, clear of its **0.0297** marginal. ⭐⭐ **And it makes a falsifiable
   prediction that was stated first and then confirmed on both branches** (§3.3): destroying
   azimuth resolution must COST AP under a correct address and HELP under a mirrored one. Corrected,
   the ladder falls monotonically 0.0457 → 0.0386;
   mirrored, it rises monotonically 0.0292 → 0.0376,
   reproducing WP-A's own pathological ordering in which the arm carrying **no azimuth information
   at all** was the best one.
4. ⭐ **NEXT LEVER, and it is not another auxiliary loss: WP-B ITSELF.** WP-B was gated on WP-D
   precisely because the map was believed empty. That gate's premise is gone: the trunk is
   separably above the marginal on both geometries, and §3.3's ladder shows the probe is genuinely
   **reading azimuth** rather than guessing. Cost and the discriminating experiment are in **§7**.
5. **`D0b`, the replicate arm A3 needs, is LAUNCHED and RUNNING** (**§1**, argv-audited to
   **one differing token**). Until it lands **A3 is `NOT MEASURED`** — per the prereg's own `F5`,
   *underpowered, not negative* — and **§5** states exactly how small the replicate floor would
   have to be for A3 to pass.

---

## 1. What was executed, and where it lives

| step | artifact | cost |
|---|---|---|
| `D0b` replicate launched on Thor (+ `D0c`, seed 1, chained) | `raw/sup_wpd_rep.sh`, `raw/D0b_argv_audit.json` | 2 × 4.7 h Thor, running |
| Three trunks banked at REF-C's real 8×20 map, ONE decode pass | `code/b1_bank_wpd.py`, `raw/bank_idx.json` | 524 s, dev-box 4060 |
| The pre-registered probe, BOTH target geometries | `code/b2_probe_wpd.py`, `raw/probe_cart.log`, `raw/panel_pol.json` | ~4 min/arm |
| The exact paired episode-cluster bootstrap on AP | `code/b5_fast_boot.py`, `raw/boot_cart.json`, `raw/boot_pol.json` | CPU |
| WP-A re-read under both address conventions | `code/b4_wpa_recheck.py`, `raw/wpa_recheck.json` | 122 s |
| D1's training-only head stripped for the planner arm | `code/b3_strip_bev_head.py`, `raw/strip_bev_head.json` | CPU |
| §5B T1 four-family arm, D0 vs D1 | `code/b6_run_5b.sh`, `code/b9_pair_5b.py`, `raw/wpdD0.json`, `raw/wpdD1.json`, `raw/paired_5b.json` | see §10 |

⛔ **Nothing was shipped to Thor but a launch.** The panel's arms stay comparable to `D0b`: the five
load-bearing files on Thor (`refc_v3_train.py`, `refc.py`, `refc_bev_aux.py`, `bev_aux.py`,
`v2_dataset.py`) were last modified **2026-09-07 23:28**, before D0 launched at **23:48**, and were
not touched.
⚠️ The A40 (`tanitad-refcv3`, finishing refcv5-v2) was **never contacted**.

---

## 2. The controls, and the one that fired

⛔ **The constant-score control is the only thing standing between this panel and
`R-2026-09-07-ap-ties`**, in which a no-information arm scored **above** its own base rate. It is
checked for **EQUALITY AGAINST A LITERAL**, and the panel **refuses to run** if it does not hold.

| control | must read | Cartesian | polar |
|---|---|---|---|
| constant score (0.37) | **exactly** the base rate | **0.016106800713** ✅ | **0.029992478616** ✅ |
| all-zero | = base rate | 0.016106800713 ✅ | 0.029992478616 ✅ |
| all-one | = base rate | 0.016106800713 ✅ | 0.029992478616 ✅ |
| perfect ranker | exactly 1.0 | 1.000000000000 ✅ | 1.000000000000 ✅ |
| `shuf_D1` (features from a RANDOM OTHER frame) | must fall to the marginal | 0.0289 (< `pos_only` 0.0302) ✅ | 0.0670 (< 0.0696) ✅ |

⭐ **AND IT FIRED ON THE FIRST RUN — CORRECTLY, AND ON MY REFERENCE, NOT ON THE METRIC.** The panel
refused with *"constant reads 0.01610680071310928 but the base rate is 0.016106801107525826"*. The
discrepancy is **3.9 × 10⁻¹⁰**: I had accumulated the base rate as a **float32 `.mean()` over
20,859,636 cells**, while the AP is computed in float64. ⛔ The fix is the **reference** —
`n_pos / n_scored` from integer counts — **not a tolerance on the comparison**. Rounding the
comparison is exactly how a real tie-break defect slips past a control that was written to catch it.

---

## 3. ⛔⛔ THE INSTRUMENT FINDING — WP-A's AZIMUTH ADDRESS IS MIRRORED

This is not a style preference; it is the **`M1` "mirrored world" mutant** that WP-D's own mutation
proof kills, found alive in a **banked instrument**.

**Two implementations in this repo AGREE with each other:**

* `bev_aux.azimuth_column` — `col = floor((hfov/2 − az) / daz)`, docstring: *"Column 0 is the
  LEFTMOST image column, which is azimuth `+hfov/2`"*, MEASURED by the parked-car experiment.
* `psg_targets.azimuth_column` — `col = int((hfov/2 − az) // (hfov/n_cols))`. Same.

**WP-A's `s5_indexed.py` uses the opposite:** `u = f_ref·az + W/2`, so `+az` (LEFT) maps to the
**highest** column. The two are mirrors.

### 3.1 Settled empirically, on fit + val only, never on test

The choice is made by the **RAW-PIXEL arm's VAL AP** — an *unlearned* representation with a direct
geometric link to the target, so the correct address must beat its mirror:

| geometry | `prog` (repo convention) | `wpa` (mirror) | ratio |
|---|---|---|---|
| Cartesian, WP-D bank | **0.048265** | 0.031789 | **1.52×** |
| polar, WP-D bank | **0.089897** | 0.062447 | **1.44×** |

⭐ And the **polar registration is EXACT under `prog` and only under `prog`**: MEASURED
`max |col(prog) − j| = 0.000000` over the whole 24 × 20 grid — which is what the prereg's
*"one target column per feature-map column, zero resampling"* actually asserts.

### 3.2 ⛔ Re-read on WP-A's OWN banked bytes — one variable moved

`code/b4_wpa_recheck.py` runs **WP-A's own bank** (`wpa-readout/bank/k8r1`, the v6 trunk at step
30,000, 16 × 40, 128-d), its own split, its own target, under **both** signs. Same head, same seed;
the sign is the only difference. Constant control reads the base rate **exactly** (0.016124196300).

| arm | AP_test | WP-A's BANKED value (`raw/indexed_k8.json`) |
|---|---|---|
| `pix` / **prog** | **0.05961** | — |
| `pix` / `wpa` | **0.02840** | `pix_16x40` **0.0270** |
| `tok` / **prog** | **0.04572** | — |
| `tok` / `wpa` | **0.02917** | `tok_16x40` **0.0295** |
| `pos_only` | 0.02969 | 0.0325 |

⭐ **The mirrored column reproduces WP-A's banked panel; the corrected one does not.** That is the
discriminating control: my re-run is not merely *different* from WP-A, it **matches WP-A exactly
when mirrored** and lifts by **2.10× (pix)** and **1.57× (tok)** when unmirrored. Under the
corrected address the v6 trunk reads **0.0457 against a 0.0297 marginal**, where WP-A recorded
*"every arm sits at the marginal."*

⭐ **The tell was visible in WP-A's own banked table all along and nobody read it that way:** its
**best** token arm is `tok_1x1` (**0.0335**) — the arm that average-pools the entire 16 × 40 map to
a **single value** and therefore carries **no azimuth information at all** — while `tok_16x40`,
which carries the most, is the **worst** (0.0295). Under a correct address more azimuth resolution
cannot hurt. Under a mirrored one, **destroying the address is an improvement**, because a mirrored
feature actively misleads.

### 3.3 ⭐⭐ THE FALSIFIABLE PREDICTION — stated in advance, confirmed on BOTH branches

If the address is mirrored, the diagnosis makes a sharp, checkable prediction about a table that
already exists. **Under a CORRECT address, destroying azimuth resolution must COST AP.** WP-A's
banked ladder does the opposite — it RISES as the pool coarsens, ending with `tok_1x1` (**0.0335**),
an arm carrying **no azimuth information whatsoever**, as its BEST rung. So: re-run WP-A's own
ladder under both signs, same head, same seed, only the sign moved.

| pool | azimuth columns kept | **corrected** (`prog`) | **mirrored** (`wpa`) | WP-A's BANKED value |
|---|---|---|---|---|
| `16x40` | most azimuth (40 columns) | **0.0457** | 0.0292 | 0.0295 |
| `8x20` | 20 | **0.0441** | 0.0337 | 0.0330 |
| `4x4` | 4 | **0.0398** | 0.0364 | 0.0327 |
| `1x1` | **ZERO** — the whole map pooled to one value | **0.0386** | 0.0376 | 0.0335 |

⭐⭐ **Both branches came out as predicted, and monotonically.**
* **Corrected (`prog`): strictly DECREASING**, 0.0457 → 0.0441 → 0.0398 → 0.0386. Azimuth
  resolution is worth **+0.0071 AP** — the map is being *read*.
* **Mirrored (`wpa`): strictly INCREASING**, 0.0292 → 0.0337 → 0.0364 → 0.0376, and it
  **reproduces WP-A's banked ordering** (0.0295 → 0.0330 → 0.0327 → 0.0335). Under a mirrored
  address the feature actively misleads, so **throwing it away is an improvement**.

⛔ **This is the strongest evidence in this package, and it is not a comparison of my numbers to
WP-A's.** It is a prediction that could have failed on either branch and did not: the mirrored
branch had to reproduce a pathological ordering, and the corrected branch had to invert it. Both
did, on WP-A's own bytes.

### 3.4 What this does and does not overturn

* ⛔ **It does NOT retract WP-A's oracle ceiling** (0.4713 at 16 × 40 / 0.3374 at 8 × 20). I did not
  re-run the oracle arm, and it is a different measurement. **Evidence class of any claim about the
  oracle arm here: NOT MEASURED.**
* ⛔ **It DOES undermine `E-READOUT-CEILING-1`'s central dissociation** — *"on the real
  planning-trained trunk the fit ladder does not transfer"* — which is the sentence WP-D was
  commissioned from.
* ⚠️ **It is a WORK ITEM FOR THE BENCHMARKS & EVALS FLYWHEEL and the Master Mind, escalated here
  rather than left in a README** (that has cost 10 days before): WP-A's banked panel and every row
  that quotes it need re-reading under the corrected address. I did not edit WP-A's package.

---

## 4. THE PANEL

Every arm shares one head architecture, parameter count, optimiser, step count and seed, so the only
difference between arms is the information in the trunk. Fit split trains; **val** picks the
operating threshold **and** the address convention; **test** is scored and never tuned on.

### 4.1 `n` and `d`, as §5E requires

| | Cartesian (PRIMARY) | polar (secondary) |
|---|---|---|
| test rows (`n`) | **2,974** | 2,974 |
| test **episodes** (the bootstrap's cluster) | **30** | 30 |
| cells per row | 7,014 | 480 |
| **scored cells** | **20,859,636** | **1,171,327** |
| positives | 335,982 | 35,131 |
| base rate | **0.016106801** | **0.029992479** |
| `d` raw per row (token arms) | **112,640** (8 × 20 × 704) | 112,640 |
| `d` into the head, per cell | 149 | 149 |
| head params | 67,201 (token) / 56,113 (pix) | same |
| fit / val / test clips | 84 / 25 / 30 | 84 / 25 / 30 |

⭐ `n` = 20.9 M scored cells against `d` = 149 at the head, so this panel is **not** the
`n ≪ d` failure of 2026-08-22. It is not underpowered by construction.

### 4.2 The arms

| arm | what it is | AP_test **cart** | AP_test **polar** |
|---|---|---|---|
| constant / all-zero | no information | **0.016107** | **0.029992** |
| `shuf_D1` | D1's features from a RANDOM OTHER frame | 0.0289 | 0.0670 |
| `pos_only` | features ZEROED — the marginal field | **0.0302** | **0.0696** |
| `pix` | raw 9-channel cell means — the A2 floor | **0.0438** | **0.1337** |
| `tok_D0` | **aux OFF (the control)** | **0.0548** | **0.1434** |
| `tok_D1` | **aux ON (THE LEVER)** | **0.0565** | **0.1527** |
| `tok_D2` | **aux ON, SHUFFLED (zero-information) target** | **0.0602** | **0.1525** |

⛔ **Read the last two rows together. The zero-information arm is not behind the lever on either
geometry — it is ahead of it on one and level on the other.**

---

## 5. CRITERION BY CRITERION — the verdict, as written in advance

Estimator: **paired episode-cluster bootstrap** over the **30 test episodes** —
`taniteval.ci.episode_index` + `_draws`, **imported, not reimplemented**. ⛔ AP is a *ranking*
statistic and does not decompose per window, so the *statistic* recomputed inside each draw is the
pooled **tie-group** AP; only that differs from `paired_episode_cluster_bootstrap`. Paired: both
arms see the same draws. ⛔ `overlapping_holdout_se` is never called.
⚠️ **This interval answers *"would another draw of EPISODES say this?"* and nothing else.** It is
blind to training-run variance — which is what `D0b` exists to measure (`H-ESTIM-SEED-1`).

### POLAR (complete) — `raw/boot_pol.json`, **n_boot = 2000**, all 1,171,327 supervised cells

| bar | statement | measured | verdict |
|---|---|---|---|
| **A1** | `AP(D1) − AP(pos_only)` ≥ **+0.010**, separated | **+0.08291** [+0.02991, +0.13250] | ✅ **PASS** |
| **A2** | `AP(D1) > AP(pix)`, separated | **+0.01948** [-0.01211, +0.05273], **not separated** | ⛔ **FAIL** |
| **A3** | ≥ **3×** a replicate floor measured in this panel | **+0.00923** [-0.01457, +0.03026], **not separated**; **floor not yet measured** | ⏳ **NOT MEASURED** |
| **A4** | `AP(D1) − AP(D2)` ≥ 3× that floor | **+0.00025** [-0.01655, +0.01647], **not separated** | ⛔ **FAIL** |

Context rows from the same panel, same estimator:

| comparison | delta | 95 % CI | separated |
|---|---|---|---|
| `tok_D0 − pos_only` — **does the AUX-OFF trunk already carry agents?** | **+0.07368** | [+0.02409, +0.12850] | **YES** |
| `tok_D2 − pos_only` | +0.08267 | [+0.03190, +0.13220] | YES |
| `pix − pos_only` — **do raw pixels beat the marginal?** | +0.06344 | [+0.01044, +0.11261] | **YES** |
| `tok_D0 − pix` | +0.01025 | [-0.03738, +0.05900] | no |
| `tok_D2 − tok_D0` — **the ZERO-INFORMATION target vs aux-off** | +0.00898 | [-0.01121, +0.02454] | no |
| `shuf_D1 − pos_only` (control, must not gain) | -0.00266 | [-0.00746, +0.00091] | no ✅ |

⭐ **Two independent bootstrap runs agree**: the probe's own in-line 200-draw pass read A4
**+0.00023 [−0.01703, +0.01494]** and this 2,000-draw exact pass reads
**+0.00025 [-0.01655, +0.01647]** — same verdict, different draw counts and different code paths
(`raw/panel_pol.json` vs `raw/boot_pol.json`).

### CARTESIAN (PRIMARY) — see `raw/boot_cart.json`

Same estimator, same 30 episodes, **n_boot = 2000**, **all 20,859,636 scored cells** (no cell subsampling).

⭐ **The bootstrap is EXACT, not an approximation, and that was verified rather than asserted:** a draw only changes how many times a row is counted, never a score, so the global sort and tie-group boundaries are computed once and re-weighted. Checked against a genuine re-sorted AP on 3 real draws: **bit-equal on all 3** (`0.061914933142`), (`0.057391547724`), (`0.061711151638`)

| bar | statement | delta | 95 % CI | separated | verdict |
|---|---|---|---|---|---|
| **A1** | `D1 − pos_only` ≥ **+0.010** | +0.02630 | [+0.00822, +0.04256] | **YES** | ✅ **PASS** |
| **A2** | `D1 > pix` | +0.01269 | [-0.00082, +0.02495] | no | ⛔ **FAIL** |
| **A3** | ≥ **3×** a replicate floor from THIS panel | +0.00173 | [-0.00777, +0.01043] | no | ⏳ **NOT MEASURED** |
| **A4** | `D1 − D2` ≥ 3× that floor | -0.00366 | [-0.01173, +0.00397] | no | ⛔ **FAIL** |

⭐ **A3, stated precisely so `D0b` can settle it without ambiguity.** The measured lever gap is **+0.00173 AP**. A3 passes only if the replicate floor `|AP(D0b) − AP(D0)|` is **below 0.00058 AP** — i.e. only if two runs differing in **nothing** agree to better than **1.05 %** of the arm's own AP. ⛔ For scale, WP-A's oracle-rig floor was ≤ **0.0122 AP**, and that number is **not borrowed** — the floor is being measured here. ⚠️ If the floor exceeds **0.00539**, then D0, D1 and D2 are all within one replicate spread of each other and the honest reading becomes `F5` — **underpowered at 4,000 steps**, not a measured difference in either direction.

Context rows, same estimator:

| comparison | delta | 95 % CI | separated |
|---|---|---|---|
| `D0 − pos_only` — **does the AUX-OFF trunk already carry agents?** | +0.02457 | [+0.00956, +0.03921] | **YES** |
| `D0 − pix` — **aux-off trunk vs the raw-pixel floor** | +0.01095 | [-0.00267, +0.02292] | no |
| `pix − pos_only` — do raw pixels beat the marginal? | +0.01362 | [+0.00190, +0.02727] | **YES** |
| `D2 − D0` — the ZERO-INFORMATION target vs aux-off | +0.00539 | [-0.00040, +0.01154] | no |
| `D2 − pos_only` | +0.02996 | [+0.01263, +0.04659] | **YES** |
| `shuf_D1 − pos_only` (CONTROL: must not gain) | -0.00133 | [-0.00355, +0.00087] | no |

| arm | pooled AP (this bootstrap's point estimate) |
|---|---|
| `tok_D0` | 0.054757 |
| `tok_D1` | 0.056489 |
| `tok_D2` | 0.060151 |
| `pix` | 0.043804 |
| `pos_only` | 0.030186 |
| `shuf_D1` | 0.028859 |

⚠️ These point estimates are recomputed from the **fp16** score matrices the panel banked, so they can differ from §4.2's fp32 values in the 4th decimal; the ORDERING and every delta above are computed within this one consistent set.

---

## 5b. ⛔ HOW TO CLOSE `A3` WHEN `D0b` LANDS — the exact recipe, so nobody re-derives it

`wpd-D0b-4k` finishes ~**4.7 h** after its 2026-09-08 **17:51:00Z** launch; `wpd-D0c-4k` (seed 1)
follows it on the same supervisor. Then, on the dev-box 4060, with **no new design decisions**:

```
# 1. extract the trunk, exactly as D0/D1/D2 were extracted (encoder keys only)
ssh tanitad-thor-wifi 'PY=/home/nvidia/venvs/tanitad-train/bin/python; $PY - ' <<'PY'
import torch, os
for arm in ("D0b", "D0c"):
    ck = torch.load(f"/home/nvidia/experiments/wpd-{arm}-4k/ckpt.pt",
                    map_location="cpu", weights_only=False, mmap=True)
    sd = ck["model"]
    enc = {k[len("core.encoder."):]: v.clone()
           for k, v in sd.items() if k.startswith("core.encoder.")}
    torch.save({"encoder": enc, "step": int(ck["step"]), "arm": arm,
                "n_enc_params": sum(v.numel() for v in enc.values()),
                "bev_aux_keys": []}, f"/home/nvidia/wpd_trunks/trunk_{arm}.pt")
PY
scp tanitad-thor-wifi:/home/nvidia/wpd_trunks/trunk_D0b.pt C:/Users/Admin/wpd-probe/trunks/

# 2. bank + probe, SAME seed, SAME split, SAME address -- nothing is re-chosen
python code/b1_bank_wpd.py --arms D0b,D0c --out C:/Users/Admin/wpd-probe/bank_rep
python code/b2_probe_wpd.py --geom cart --az-sign prog --arms tok_D0b,tok_D0c ...
python code/b5_fast_boot.py --geom cart --n-boot 2000 --out raw/boot_rep.json
```

⛔ **Pass `--az-sign prog` explicitly**, not `auto`: the convention is now settled (§3) and
re-selecting it per run would make the replicate differ from the panel in the instrument as well as
in the seed.

**Then A3 reads as follows, with no further judgement calls:**

| the floor | what it is |
|---|---|
| `f_same = |AP(D0b) − AP(D0)|` | run-to-run **nondeterminism** only (same seed) — the `A0b_replicate` design |
| `f_seed = |AP(D0c) − AP(D0)|` | **plus** init/shuffle variance — the prereg §4 D0b |
| **A3's denominator** | **`max(f_same, f_seed)`** — the larger, because a same-seed-only floor is anti-conservative and would make the 3× bar easier |

⭐ **A3 PASSES only if `AP(D1) − AP(D0) ≥ 3 × max(f_same, f_seed)`.** ⚠️ And if that floor turns out
to exceed the whole D0→D2 spread, the honest verdict for the LEVER becomes **`F5` — underpowered at
4,000 steps** rather than `F4`; **A4 is unaffected either way**, because it compares two arms whose
CI already straddles zero.

⭐ **The floor will not be zero.** MEASURED already, before either arm finishes: at step 50, D0
reads `loss 51.63894 / traj 2.30219` and D0b reads `loss 51.65258 / traj 2.30269` — two runs
differing in **nothing** have already diverged, so this arm is measuring a real quantity and not an
identity.

---

## 6. Which failure twin fired

⛔ **`F4` — *"A4 fails (`D2` matches `D1`) ⇒ the gain is capacity/regularisation, not agent content.
Refuted."*** It fired on the geometry the head was trained on, with the shuffled arm level to
**0.00025 AP**, and on the Cartesian geometry the shuffled arm is **AHEAD**
(**-0.00366 [-0.01173, +0.00397]**).

⛔ **AND `A2` FAILS TOO, ON BOTH GEOMETRIES — say it, because SUCCESS was `A1 ∧ A2 ∧ A3 ∧ A4`.**
`D1 − pix` = **+0.01269 [-0.00082, +0.02495]** Cartesian and
**+0.01948 [-0.01211, +0.05273]** polar — positive in both, **separated in
neither**. The aux-ON trunk does not separably beat the raw-pixel floor, which is the bar
`E-DEC-18-R1` failed and which the prereg flagged in advance as *"not a formality"*. ⇒ **two of the
four bars fail and a third is unmeasured**; the claim does not survive on any reading.

⚠️ **`F5` is also partly live and must be said**: `A3` is **NOT MEASURED**, and the prereg is
explicit that an unmeasured criterion is **underpowered, not negative**. `D0b` decides it. But `F4`
does not depend on `D0b`: A4 compares two arms *in the same panel* and reads +0.00023 with a CI
straddling zero.

⛔ **`F1` did NOT fire** — the trunk is not empty, so *"no transferable content"* is the wrong
verdict and `E-DEC-8` DINOv3 distillation is **not** the lever this result calls for. Reporting F1
here would have been the easy, and wrong, answer.

⛔⛔ **`F2` ALSO FIRED — the planner REGRESSED, separably, on 8 of 9 metrics** (§10): ADE **+0.02610 [+0.00970, +0.04240]** worse for D1, and the same sign across LONGITUDINAL and LATERAL. Per the prereg this is *"the `E-DEC-18b` shape reproduced on REF-C"*, **reported as such and NOT tuned around**. **`F3`/`5D` (the `D3` detached arm and the `D4` two-state
deliberate regression) were NOT RUN**: the prereg's cheaper first cut funded D0/D1/D2 only. They are
named blockers, not omissions: each is 4.7 h of Thor.

---

## 7. The next lever, its cost, and the experiment that decides it

⭐ **RULE ZERO: the refutation is the waypoint. The lever is WP-B, and it is unblocked by this
result rather than by a new arm.**

| | |
|---|---|
| **why** | WP-B (waypoint-indexed deformable cross-attention, DiffusionDrive coupling (1)) was gated on WP-D *because WP-A said the map was empty*. MEASURED here: `D0 − pos_only` is **SEPARATED on BOTH geometries** — **+0.02457 [+0.00956, +0.03921]** Cartesian, **+0.07368 [+0.02409, +0.12850]** polar — so the aux-OFF trunk carries agent content well above the marginal, and §3.3's ladder shows that content is read through the AZIMUTH ADDRESS (destroying azimuth costs +0.0071 AP). ⚠️ **The honest limit, stated rather than buried:** `D0 − pix` = **+0.01095 [-0.00267, +0.02292], NOT separated**, so *"REF-C beats raw pixels"* is a point-estimate ordering, not a separated result — on either geometry. Raw pixels themselves DO separably beat the marginal (**+0.01362 [+0.00190, +0.02727]**). The v6 trunk, by contrast, sits BELOW raw pixels (0.0457 vs 0.0596). What is settled is that the map is **not empty**; what is NOT settled is whether the learned trunk adds to its own input. The gate's premise is gone. |
| **cost** | 0 new training arms to *decide* it — the index is an architecture change to `refc.py`'s anchor decoder, testable on the v7-tiny ladder first (`TanitAD_ValidateAIDesign`). |
| **the discriminating experiment** | a **replicate-controlled** A/B of the indexed decoder against refcv5's flat cross-attention. ⛔ It must carry a replicate arm from the start — this panel is the third demonstration that a separated CI from one-seed arms is necessary and not sufficient. |
| **⛔ what must happen FIRST, and it is cheap** | **re-read WP-A's banked panel under the corrected address** and correct `E-READOUT-CEILING-1`. Everything downstream of *"the map has no agents"* inherits a mirrored probe. ~10 GPU-min; the script is `code/b4_wpa_recheck.py` and it already runs. |

⚠️ **A second, smaller lever this panel measured for free:** `pix` **beats** `tok` on the **v6**
trunk (0.0596 vs 0.0457) while `tok_D0` **beats** `pix` on **REF-C** (0.0548 vs 0.0438). The v6
encoder *destroys* agent-localisation information relative to raw pixels; REF-C's adds to it. That
is an architecture fact worth its own row, and it was invisible while the address was mirrored.

---

## 8. What this result does NOT say

* ⛔ It does **not** say the aux head is broken. It says the **information in its target** is not
  what moved the trunk: a shuffled target moved it the same amount.
* ⛔ It does **not** retract the prereg's design work. The removability claim **reproduced exactly**
  on the trained checkpoint: stripping `bev_aux` removed **6 keys / 182,616 parameters** — the
  prereg's own figure — leaving **108,307,024**, identical to D0's count, every kept tensor
  bit-identical (`raw/strip_bev_head.json`).
  ⚠️⚠️ **AND I GOT THE REASON FOR THE STRIP WRONG, FROM ONE PROBE — CORRECTED BY MEASUREMENT.** I
  wrote that the strip was *"necessary, because `refcv3_arm.py` has no `bev_aux` handling, so D1's
  raw checkpoint would be refused with 6 unexpected keys."* That came from `grep bev_aux
  refcv3_arm.py` returning nothing, and **it is false**: the handling arrives **indirectly**,
  because `rebuild_config` reconstructs the model through `refc_v3_train.build_parser` +
  `_pin_trainer_cfg` from **D1's own argv**, which carries `--bev-aux col`. The rebuilt model
  therefore **HAS** the head, and it is the **STRIPPED** checkpoint that is refused — for 6
  **MISSING** keys, the exact opposite failure. MEASURED: the D1 dump died on it and produced
  **0 of 40** episodes while the chain still printed `CHAIN COMPLETE`. ⇒ D1 is dumped from its
  **RAW** checkpoint, which is still a clean comparison because the prereg proves the head is
  constructed and called LAST, consumes no RNG, and leaves the planner's output bit-identical.
  ⭐ **Root-cause class: an absence claim from ONE probe** (`CLAUDE.md`: *"Absence found at ONE
  location is not absence"*), and the thing that caught it was logging the **per-arm file count**
  next to the exit code rather than trusting either.
* ⛔ It does **not** measure `D3`, `D4` or `D5`. §5D's deliberate-regression detection is
  **NOT MEASURED**.
* ⚠️ **A provenance gap in the pre-registration itself, MEASURED from the shipped record.** §3
  states the aux head's cost is *"reported on its own `param_breakdown["bev_aux"]` line and
  **subtractable** — the DEPLOYED count is unchanged."* In D1's actual `config.json` **that key
  does not exist**: the breakdown's keys are `core, ego_inject, gstr_cond, nav_inject, phi_tac,
  scorer, str_goal_head, tac_goal_tok_head, tac_heads, tac_latent_proj, total`, and the **182,616
  params are absorbed into `core`** (D0 **106,067,312** → D1 **106,249,928**). The count is still
  recoverable — by differencing against the aux-off arm, or by the strip — but **a single arm's run
  record cannot report its own deployed count**, which is what the claim promised. Same family as
  the units rule: a fact that is true of the design and absent from the artifact a reader opens.
* ⚠️ The occlusion mask is a **lower bound** (prereg §2.2) and every polar number inherits that.
* ⚠️ **A3 is NOT MEASURED, not negative.**

---

## 9. Cross-checks that had to agree, and did

| quantity | this panel | independent source | agree? |
|---|---|---|---|
| polar base rate over supervised cells | **0.0316155** (whole bank) | prereg §3 **0.031634** (census over the B1 EVAL join) | ✅ 0.06 % |
| polar occlusion-masked fraction | **17.635 %** | prereg §2.2 **17.656 %** | ✅ |
| Cartesian occupancy rate | **0.01740156** | WP-A `idx.json` **0.01740156** | ✅ |
| the Cartesian target itself | 13,223 × 7,680 | WP-A's banked `y.npy` | ✅ **bit-identical, 0 disagreeing cells** |
| row plan | 139 clips / 13,223 rows | WP-A's `idx.json` plan | ✅ **element-wise equal** |
| aux-head parameter count | **182,616** (stripped from the trained ckpt) | prereg §3 **182,616** | ✅ |
| aux-head parameter count, a THIRD way | **182,616** | the runs' own `config.json` `param_breakdown.total`: D1 **108,440,118** − D0 **108,257,502** | ✅ |
| WP-A's banked `pix_16x40` / `tok_16x40` | mirrored re-run **0.0284 / 0.0292** | banked **0.0270 / 0.0295** | ✅ |

⭐ **The pre-registration's own suite is still green against the EXACT modules these arms trained
with**: `stack/tests/test_bev_aux.py` **36 passed**, run against a local snapshot whose
`bev_aux.py` (`954c7e6e…`) and `refc_bev_aux.py` (`eab5d9cf…`) are **md5-identical to the files on
Thor**. `test_CONTROL_constant_score_reads_exactly_the_base_rate`,
`test_CONTROL_perfect_ranker_reads_exactly_one`, `test_shared_params_bit_identical`,
`test_planner_output_bit_identical` and `test_loss_parity_guard_refuses_the_E_DEC_18b_failure_shape`
all PASS — so the refutation is a statement about the LEVER, not about a broken implementation.

⭐ The bank was asserted on **CONTENT**, never on an exit code: three trunks non-zero
(`mean|tok|` 1.794 / 1.784 / 1.824), pairwise **different** (`|D0−D1|` 2.046, `|D0−D2|` 2.016), and
the run **refuses** if two trunks are identical or a bank is all-zero.

---

## 10. The planner bars (prereg §5B) — **T1**, four families

**Instrument** `taniteval/tools/refcv3_arm.py` (the REF-C adapter), run on the dev-box 4060 over the
**141-clip B1 eval set** with the v7.2 eval labels and the banked B1 lead block, `--grid 2s`.
**Tier T1** as that tool stamps it for the fed arm `os`. ⛔ ADE alone is not a result; the four
families bind, each with its own `n`, never pooled.

⛔ **D1 is loaded from its RAW checkpoint** — see §8: the adapter rebuilds the aux head from D1's
own argv, so the *stripped* checkpoint is the one that is refused. The comparison is still clean,
and for the reason the pre-registration engineered: the head is constructed **LAST**, called
**LAST**, consumes **no RNG**, and leaves the planner's entire output dict **bit-identical** with
`bev_logits` the only extra key. Its presence cannot move a trajectory.

⛔ **The pairing is ASSERTED, not assumed** (`code/b9_pair_5b.py`): episode ids, window starts, the
ground truth and `v0` must be **element-wise identical** across the two dumps, and the script
REFUSES otherwise — a "paired" interval across mis-aligned windows is arithmetic, not evidence. It
also refuses if the two arms emitted bit-identical trajectories, which was **verified to fire** by
running it against a dump and itself.

⚠️ **`t1_eval`-class hazard, MEASURED here and worth carrying:** the smoke's rollout completed both
episodes and the **ANALYSIS** then died on a missing sibling module — **and the wrapper exited 0**.
The dumps survived and `--analyze-only` recovered every number with **zero GPU**. Dump first,
analyse second, and assert on the artifact.

**MEASURED**, `raw/paired_5b.json`. **n = 3,422 windows / 40 episodes**, the
SAME windows for both arms — element-wise identical episode ids, window starts, ground truth and
`v0`, asserted before any interval was computed. The two arms are genuinely different
(mean |pred(D1) − pred(D0)| = **0.1094 m**). Estimator:
**paired episode-cluster bootstrap** (`taniteval/ci.py`), n_boot 2,000, clusters = episodes.
⛔ **Every metric below is an ERROR, so a POSITIVE delta means D1 is WORSE.**

| family | metric | **D1** (aux ON) | **D0** (aux OFF) | delta | 95 % CI | separably worse for D1? |
|---|---|---|---|---|---|---|
| ADE | `ADE_m` | 0.5039 | 0.4778 | +0.02610 | [+0.00970, +0.04240] | **YES — WORSE** |
| ADE | `FDE_m` | 1.0672 | 1.0128 | +0.05440 | [+0.01830, +0.09050] | **YES — WORSE** |
| LONGITUDINAL | `LON_speed_mae_mps` | 0.4104 | 0.3938 | +0.01660 | [+0.00140, +0.03300] | **YES — WORSE** |
| LONGITUDINAL | `LON_along_mae_m` | 0.3819 | 0.3688 | +0.01310 | [-0.00080, +0.02760] | no |
| LONGITUDINAL | `LON_accel_mae_mps2` | 0.4446 | 0.4265 | +0.01810 | [+0.00340, +0.03400] | **YES — WORSE** |
| LATERAL | `LAT_cross_mae_m` | 0.2200 | 0.2040 | +0.01590 | [+0.00010, +0.03180] | **YES — WORSE** |
| LATERAL | `LAT_heading_mae_deg` | 1.4659 | 1.3062 | +0.15980 | [+0.03650, +0.29700] | **YES — WORSE** |
| LATERAL | `LAT_curvature_mae_1pm` | 0.0049 | 0.0043 | +0.00060 | [+0.00020, +0.00100] | **YES — WORSE** |
| LATERAL | `LAT_yawrate_mae_radps` | 0.0296 | 0.0275 | +0.00210 | [+0.00030, +0.00410] | **YES — WORSE** |

⛔⛔ **B1 FAILS, and not marginally: D1 is separably WORSE on 8 of 9 metrics**, across
**ADE, LONGITUDINAL and LATERAL**. The only metric that is not separated is `LON_along_mae_m`, and
it points the same way (+0.01310). ⇒ **failure twin `F2` fires: *"the aux term buys representation
at the planner's expense — `E-DEC-18b`'s shape reproduced on REF-C."*** The prereg commits this to
be **reported as such and NOT tuned around**: a weight sweep after seeing this is a NEW
pre-registration, not a continuation of this one.

### 10.1 ⭐ The "it must still ACT" control — and it CLEARS, which is what makes the regression real

`D-REFAV1-LON-ACTS` exists because a longitudinal family can be "improved" by making the planner
**stop**. Here the concern is the mirror image: is D1 worse merely because it acts *more* wildly, or
is it genuinely mis-planning?

| control | D1 | D0 | floor | verdict |
|---|---|---|---|---|
| mean \|a\| (m/s²) | **0.2443** | 0.2698 | 0.044 | falls by **0.0255**, INSIDE the floor |
| fraction of plans with a ≡ 0 | **0.00994** | 0.01081 | 0.05 | does **not** rise ✅ |

⇒ **D1 is not acting less in any amount the rule counts, and it is not freezing.** The regression is
not the `kamm07`/`l3ladder` "improve LON by stopping" artefact wearing a new costume — it is a
planner that is genuinely worse while still driving.

### 10.2 The STRATEGIC family, per the four-family rule

⛔ **UNAVAILABLE, with its reason and its `n`, exactly as the rule requires** — never silently
dropped. `refcv3_arm.py` reports `status: UNAVAILABLE`, `n: 0`, reason *"strategic decisions not
present in the scored pass (missing ['route_pred', 'route_gt']) … producing this family needs a
hierarchy-traversing eval, which is a WORK ITEM."* Both arms report it identically, so it cannot
hide a difference between them. The other three families are complete on both arms
(`raw/wpdD0.json`, `raw/wpdD1.json`), including LONGITUDINAL distance-keeping (`status: OK`) from
the banked B1 lead block.

### 10.3 What §10 adds to the verdict

⭐ **`E-BEV-AUX-1` now fails on BOTH halves of its own success condition**, and the two halves fail
for consistent reasons: the representation bars say the aux term's **target information** did
nothing (a shuffled target does the same), and the planner bars say the aux term's **gradient** cost
the planner real accuracy. That is precisely `E-DEC-18`/`18b`'s measured shape — *"the largest
environment gain the programme has measured, clearing neither the raw-pixel floor nor the
downstream task"* — reproduced on a different stack, which is what the pre-registration named as the
prior that binds it.
