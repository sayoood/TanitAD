# RESULT — flagship-v1 as a cross-rig / IDM substrate (the own-encoder PIVOT evidence)

**Landed 2026-07-24** on pod3 (A40), `gpu_lock v1-encoder-char` (released). Follows the Branch B
FAIL (`../RESULTS_branchB.md`): Branch B's mixed read of v1 frozen (multirig_val rig-B speed R²
**+0.657** but rig_val **−1.169**) needed resolving before Sayed's pivot call. **Evidence class:
MEASURED (ours + artifact: `results_v1_encoder_char.json`; the comma-yaw control from
`../../2026-07-22-own-dynamics-encoder/results_camcond_multirig.json`).** Converged head
(epochs=50; flagship in-domain speed R² 0.86–0.92 = the C5 convergence/validity check).

**Factual correction (verified by code — `geom_augment` is only in `train_dynamics_encoder.py`):**
flagship-v1 (train_flagship4b, WM recipe) trained **without** geom_augment. So for v1 the aug arm
is a **geometric-robustness probe**, not a distribution match. For Branch B (trained *with* aug)
the aug arm is its **matched condition** → it doubles as closure of the RESULTS_branchB
in-domain-weakness caveat.

---

## PIVOT VERDICT — v1 IS a usable IDM substrate AS-IS → the CHEAP pivot is supported (1 condition, 1 bounded residual)

Per the pre-registered rule (v1 usable AS-IS iff it transfers R²>0.5 on the cross-sets; else the
warm-started-longer variant is needed): the alarming **−1.169 rig-B failure was a HEAD-DIVERSITY
artifact, not an encoder ceiling** — it **does not persist** once the readout head sees ≥2
domains. **Recommendation: build the IDM on frozen v1 + a multi-domain readout head** (the cheap
pivot) for the immediate PhysicalAI fisheye-heterogeneity + speed/trajectory need; reserve the
expensive warm-start-longer for a cross-geometry-class **yaw** gap that is currently **untestable
here** (see residual) — and test that cheaply first.

---

## The numbers (MEASURED, converged head — full IDM readout)

### flagship-v1 frozen — cross-rig transfer is a READOUT-diversity property, not an encoder ceiling

| head trained on | eval | speed R² | yaw R² | steer R² | ADE@2s |
|---|---|---|---|---|---|
| — (in-domain ceiling) | rig-A held-out | **0.86–0.92** | **0.84–0.92** | 0.81–0.85 | 2.6–3.6 |
| **rig-A only** | cross **rig-B** | **−1.169** | −0.208 | +0.03 | 13.86 |
| **rig-A + comma** | cross **rig-B** | **+0.657** [CI 0.36, 0.80] | **+0.504** | +0.363 | 5.74 |
| rig-A only | cross **comma** | +0.313 | (n/a†) | −1.19‡ | 11.57 |
| **rig-A + rig-B** | cross **comma** | **+0.585** | (n/a†) | +0.36‡ | 7.50 |
| rig-A only (AUG eval) | in-domain rig-A **+aug** | **0.790** | 0.906 | 0.797 | 4.22 |
| rig-A only (AUG eval) | cross rig-B **+aug** | −1.438 | −0.216 | +0.08 | 14.53 |

† **comma yaw is UNREADABLE in-domain** (MEASURED control: comma-co-trained head → comma yaw R²
**−0.00003 / −0.00009**, `results_camcond_multirig.json`) — a comma **label** property, NOT a v1
transfer failure. Comma can only test **speed** transfer. ‡ comma steer is confounded
(STEER_RATIO 15.3, pre-registered secondary).

**Five reads:**

1. **v1 frozen is a strong in-domain substrate** — speed 0.86–0.92, yaw 0.84–0.92, steer 0.81–0.85.
2. **The −1.169 rig-B "failure" is a single-domain-head artifact.** A rig-A**+comma** head lifts
   v1's held-out rig-B transfer to speed **+0.657** and yaw **+0.504** — **both >0.5**. The
   encoder latent supports cross-rig transfer; a one-domain readout overfits geometry and does not.
3. **v1's speed latent is geometry-robust across a rig CLASS boundary.** Fisheye-trained v1
   transfers **speed** to rectilinear comma at **+0.585** (2-rig head) — essentially comma's own
   **in-domain** ceiling (**0.592**). Speed generalizes fisheye→rectilinear.
4. **v1 is robust to geometric perturbation** (±5–12 px vertical pitch-proxy shift + matched cam):
   in-domain speed 0.86→**0.79**, yaw 0.92→**0.91**. Small, graceful degradation.
5. **accel R² ≈ 0 everywhere** (in-domain included) — long-accel is not linearly decodable from
   this latent for *any* arm; a general limitation, not v1-specific or cross-rig.

### Branch B — the aug-mismatch caveat is CLOSED (weakness is real, not an augmentation artifact)

| arm | in-domain rig-A speed R² | own-head val rig-B speed R² |
|---|---|---|
| clean frames | +0.039 | +0.242 |
| **aug frames (Branch B's matched training condition)** | +0.050 | +0.200 |

Branch B reads augmented frames (its training distribution) **the same** as clean → its weak
readout is **not** a clean-vs-aug mismatch. The RESULTS_branchB "weak substrate" finding stands
unqualified. **v1 ≫ Branch B on every metric, in-domain and cross-rig.** Build on v1, not Branch B.

---

## Honest bounds / caveats

- **C5 (head-fit convergence).** All numbers at a converged head (flagship in-domain 0.86–0.92
  confirms). The rig-B transfer is unlocked by head **diversity**, which is a readout property —
  reported as such; the encoder latent is the enabler, not the whole story.
- **C6 (comma disqualified for yaw).** The comma yaw≈0 is a comma-label artifact (unreadable
  in-domain), NOT a v1 cross-geometry rotation failure — corrected using the existing MEASURED
  in-domain control. Do not read it as a v1 ceiling.
- **Bounded residual → the only place the EXPENSIVE pivot might still be needed:** cross-geometry-
  class **YAW** transfer (fisheye→rectilinear / varied YouTube rigs) is **UNVERIFIED** — comma
  cannot test it. Speed transfers cross-class; rotation across class is untested. **Cheapest next
  step before any warm-start-longer GPU-days:** a rectilinear (or varied-rig) corpus with
  **readable** yaw, fit a multi-domain head, measure v1's rotation transfer. That single cheap
  test discriminates cheap-vs-expensive pivot for the YouTube-diversity case.
- These are frozen-encoder + multi-domain-head reads; they say v1's *latent* is a usable substrate
  for a small readout, not that v1 end-to-end is deployment-final.

---

## Provenance / reproduce

```
# pod3 (A40), venv python, PYTHONPATH=/workspace/TanitAD/stack:.../scripts
python scripts/run_v1_encoder_char.py --epochs 50
```
Runner: `stack/scripts/run_v1_encoder_char.py` (staged). Reuses the transfer-eval CLEAN latent
cache (`pod3:/workspace/tmp/branchb_eval/`); only the aug arm is re-encoded
(`pod3:/workspace/tmp/v1char/lat_*_aug/`, per-clip dv∈{−12,−8,−5,5,8,12}). Full readout
speed/yaw/accel/steer R²+MAE+ADE. Bootstrap: episode-cluster over eval clips, 2000×. Raw:
`results_v1_encoder_char.json`. Comma-yaw control: `../../2026-07-22-own-dynamics-encoder/
results_camcond_multirig.json` (`in_comma_heldout`).
