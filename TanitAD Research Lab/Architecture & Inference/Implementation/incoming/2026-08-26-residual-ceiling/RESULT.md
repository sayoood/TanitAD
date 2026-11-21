# RESULT — E-DEC-63: the ceiling on predicting Δz beyond drift

**Measured** 2026-08-26 23:40 (Europe/Berlin) · **Evidence class** MEASURED (ours;
dev-box RTX 4060) · **Tier** T0-DIAGNOSTIC — never a driving number · **Arm**
`rdw8p30k` (step 30,000) · 80 clips (`physicalai-val130-heldout`), 7,280 rows, k=4,
Δz PCA band [0:8), K-fold 10 · **Raw** `raw/residual_ceiling.json` · **Probe**
`residual_ceiling.py` (banked here) · Pre-registered: `SPEC.md` + AMENDMENT A1
(recorded before any read).

## The panel

| column | d | r | shuf | marginal over z_t | t | tier |
|---|---|---|---|---|---|---|
| A0 constant | 1 | +0.0000 | +0.0000 | — | — | exact ✅ |
| A1 `z_t` (drift, POSITIVE CTRL) | 2,048 | **+0.6722** | −0.0056 | — | 126.43 | ⭐ **matches banked +0.6718 to 4 decimals** |
| A6 `z_t`+noise3 (DELIB-REGRESS) | 2,051 | +0.6700 | −0.0040 | −0.0021 | −1.47 | INSIDE_NULL ✅ (failed as required) |
| A2 `z_t`+`zhat_k4` (OUR PREDICTOR) | 4,096 | +0.6699 | −0.0053 | **−0.0023** | −1.84 | INSIDE_NULL |
| A3 `z_t`+pixels 32×80 (RAW FLOOR) | 4,608 | +0.6818 | −0.0029 | ⭐ **+0.0096** | **+5.11** | **SURVIVES** |
| A4 `z_t`+tokens4×10×128+pixels (ORACLE) | 9,728 | +0.6682 | +0.0007 | −0.0040 | −2.52 | INSIDE_NULL |

**RIG VALID** — every control read its known value.

## ⛔ The verdict tree hit a cell it had no slot for — stated, not rounded away

The SPEC's §5 outcomes keyed H-A/H-B on **A4 (the oracle)**, and the script's
mechanical verdict printed **H-A** ("residual not predictable — stop optimising the
predictor"). ⛔ **That print is NOT adopted.** The table contradicts it in one cell:
**A3 — raw pixels — SURVIVES at t 5.11**, while A4, whose input is a strict SUPERSET
of A3's (same pixels + 5,120 token dims), reads null. A pre-registered tree that
lacks a cell does not get to default (C160); the cell is:

> ⭐ **A small but decisive component of the post-drift residual IS predictable —
> from raw pixels — and BOTH our predictor (−0.0023) and the encoder's own pooled
> token field fail to carry it.** The A4 null is most plausibly **capacity
> dilution** (the probe's 96-component PCA bottleneck squeezing 9,728 dims), not
> token-emptiness: the same pixel signal that clears at 4,608 dims vanishes when
> 5,120 more dims are added. Dilution is the SPEC's own named caveat (AMENDMENT A1)
> arriving in the data.

## What is and is not established

1. **MEASURED:** the achievable ceiling beyond drift from time-t observation is
   **small in this band — ~+0.01 against drift's +0.6722** — and our predictor sits
   at **0** of it. The "predictor is broken vs at-ceiling" question resolves as:
   *near-ceiling in magnitude, but the ceiling is not zero and what remains is
   pixel-borne*.
2. **MEASURED:** the encoder **discards pixel information that predicts its own
   latent's future change** (if `z_t` carried it, A3's marginal over `z_t` would be
   0). This is direct, held-out evidence for the representational half of the v7
   plan — sharper than E-DEC-7's construction argument.
3. ⛔ **NOT established: what the pixel component IS.** A deflationary explanation
   exists and must be tested before this cell steers any GPU: **photometric drift**
   (auto-exposure / sun angle) is pixel-visible, latent-invisible, and highly
   autocorrelated — it would produce exactly this signature while being *boring*.
4. ⛔ **NOT established: that the token field is empty of it** — the A4 null is
   confounded by dilution (point above).

## The follow-up probe (specified now, runnable next tick, no GPU retraining)

Same rig, four added columns, everything else frozen:
- **F1 dilution control:** `z_t + pixels + noise5120` (width-matched to A4). If the
  +0.0096 dies, A4's null is dilution and the token question reopens; if it
  survives, the token field genuinely lacks the pixel component.
- **F2 photometric control:** `z_t + lum3` (mean luminance + top/bottom-half means
  only, 3 dims). If ~+0.01 reappears, the discovery is exposure dynamics, not scene.
- **F3 tokens alone:** `z_t + tokens` (no pixels), same width question from the
  other side.
- **F4 per-direction split** of A3's marginal (which PCA directions of Δz carry it).

## Consequence for the v7 design (V7_RECIPE_AND_SCALEUP.md §5)

Unchanged in substance, sharpened in evidence: **the scaled run's lever is the
representation, not a bigger predictor** — A2 confirms the predictor is at the
achievable ceiling of the current latent space to within noise, and A3 shows the
latent is missing predictable content that is present in its own input. Whether
that content is scene structure or photometrics decides *which* representational
fix (F2 settles it, minutes of compute).


---

## F1-F4 OUTCOME — measured 2026-08-27 00:50, same rig, same fold seeds

| column | marginal over `z_t` | t | verdict |
|---|---|---|---|
| A3 pixels (re-run) | **+0.0096** | 5.11 | replicates exactly ✅ |
| F1 pixels + noise5120 (width-matched to oracle) | **+0.0131** | **9.10** | SURVIVES |
| F2 lum3 (photometric) | −0.0013 | −1.09 | INSIDE_NULL |
| F3 tokens, no pixels | **−0.0066** | **−3.95** | negative |

**F4** — the pixel marginal is SPREAD across Δz directions (PC0 +0.0216, PC3
+0.0187, PC1 +0.0113, PC5–7 ≈+0.009), not concentrated in one appearance mode.

⭐⭐ **Three of the four open questions close, one against my own hypothesis:**

1. ⛔ **Dilution does NOT explain the oracle null** — width-matched NOISE leaves the
   pixel signal fully intact (F1, t 9.10, stronger than A3). My capacity-dilution
   reading in the main RESULT is **withdrawn as the mechanism**.
2. ⭐ **The mechanism is the TOKENS THEMSELVES: F3 is *negative* (t −3.95).** The
   encoder's pooled token field carries drift-redundant structure that actively
   displaces the pixel signal in the probe's PCA basis — structured redundancy
   captures basis components; isotropic noise does not. That is why tokens+pixels
   (A4) read null while pixels+noise (F1) survives.
3. ⛔ **The photometric deflation is REFUTED at the luminance level** (F2 null) and
   disfavoured by F4's spread. The component behaves like scene structure.
4. ⚠️ **Bound stated:** "absent from the token field" is established for the
   **4×10-pooled** field; the unpooled 640-token field remains untested (compute
   bound, AMENDMENT A1).

⇒ **The E-DEC-63 conclusion sharpens into the v7 design input:** the encoder
discards pixel-level content that predicts its own latent's future, that content is
not photometric drift, and the token representation actively crowds it out. The
representational lever is not merely "where the work is" — it now has a measured,
specific defect to aim at.
