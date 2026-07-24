# RESULT — camera-conditioning ablation (the own-encoder go/no-go)

**Landed 2026-07-23** on pod3 (A40), warm-started from the **md5-verified** flagship-v1
(`b5f07d9e3dd2ca643949bc86832e6585`, step 29999). Both experiments completed;
raw JSON: `results_camcond_rig.json`, `results_camcond_multirig.json`. **Evidence class: MEASURED
(ours + artifact).** Gate (frozen in `PRE_REGISTRATION.md`): cross speed R²>0.9 AND yaw R²>0.9 AND
ADE@2s < 1.5× in-domain.

## The numbers (MEASURED)

| experiment | arm | cross-rig **speed R²** | cross yaw R² | cross ADE@2s (m) | in-dom ADE (m) | ratio | PASS |
|---|---|---|---|---|---|---|---|
| **rig** (rig-A→rig-B) | OFF | **−2.344** | +0.484 | 14.91 | 4.05 | 3.68 | ❌ |
| | **ON** | **−2.253** | +0.480 | 14.61 | 4.05 | 3.61 | ❌ |
| | **ON−OFF Δ** | **+0.091** | −0.004 | −0.30 | | | |
| **multirig** ({rig-A+comma}→rig-B) | OFF | **−2.176** | −0.200 | 14.52 | 3.80 | 3.82 | ❌ |
| | **ON** | **−2.057** | −0.070 | 14.23 | 3.77 | 3.77 | ❌ |
| | **ON−OFF Δ** | **+0.119** | +0.130 | −0.29 | | | |

External baselines (`MEASURED`, prior artifacts): plain light-FT (no conditioning) rig-A→rig-B
**−1.65** (`results_regate.json`), {rig-A+comma}→rig-B **−1.61** (`results_multirig.json`). In-domain
rig-A speed R² ~0.80–0.85 in both arms (the encoder works in-distribution; the failure is purely
cross-rig — consistent with every prior result).

## Verdict — FAIL (reported plainly; not forced)

**GAIA-2 camera conditioning added as a warm-start, last-4-block, light-FT add-on does NOT recover
cross-rig transfer.** Both arms fail the gate decisively (cross-rig speed R² ~−2.1 to −2.3 vs the 0.9
bar; ADE ratio ~3.7 vs the 1.5 bar). Two honest observations:

1. **Directionally, conditioning helps — consistently but marginally.** ON > OFF on cross-rig speed R²
   in BOTH experiments (+0.091, +0.119) AND on cross-rig yaw R² in multirig (−0.200 → −0.070). The
   mechanism is **not refuted** — it moves the needle the right way, every time.
2. **The magnitude is nowhere near enough, and ON does not even beat the plain re-gate light-FT.** A
   +0.1 nudge on a −2.2 collapse is inert for the decision; and ON (−2.06 / −2.25) is *worse than* the
   plain light-FT baseline (−1.61 / −1.65) — the suffix-only conditioning machinery, warm-started onto
   a frozen 8-block prefix that already baked in rig-A geometry, cannot undo the binding.

Mechanistically this is exactly what GAIA-2's own attribution predicts: it credits rig-generalization
to explicit conditioning **at every block**, **learned from-scratch**, **with multi-rig coverage**. Our
cheap probe had none of those three — it conditioned only the trainable suffix (4 of 12 blocks), on a
warm-started encoder whose frozen prefix is already rig-bound, trained on rig-A(-and-comma) only so
rig-B (cy≈753) is a pure extrapolation from cy≈542. The probe therefore refutes the **cheap warm-start
shortcut (Branch A)**, NOT the mechanism.

## Decision (pre-registered): Branch B is the go

> 🟥 **FOLLOW-ON OUTCOME 2026-07-24 — Branch B ALSO FAILED.** The from-scratch, all-block, multi-rig Branch B
> ran to 40k and was refuted on held-out-rig transfer (cross-rig speed R² **−0.667**, weaker than plain
> frozen flagship-v1; `../../2026-07-24-branchb-transfer-eval/RESULTS_branchB.md`; `MODEL_REGISTRY §10`).
> So **both** branches of this go/no-go are now spent — explicit camera-conditioning (cheap or expensive)
> does not close the cross-rig gap at this scale. Next lever = flagship-warm-started variant (Sayed-gated).

- ❌ **Branch A (warm-start + suffix conditioning) is REFUTED** — the mechanism as a light-FT add-on is
  far too weak.
- ✅ **Branch B (from-scratch camera-conditioned video-SSL) is the PRIMARY path**, now with a MEASURED
  pre-registered justification: the conditioning must be **learned from scratch, injected at every
  block, with genuinely multi-rig training** (the GAIA-2 regime), not bolted onto a frozen rig-bound
  trunk. The FAIL-branch escalation — **geometry-as-input** (per-pixel Plücker ray-maps, 2510.02268;
  or PRoPE projective relative positional encoding, 2507.10496) as a stronger, lower-level rig signal
  than global per-block FiLM — is now **on the table** for Branch B.

## Note on this experiment being reported as "dead"

The run was flagged mid-flight as crashed (PID gone, GPU idle, JSONs "not written"). **It had in fact
COMPLETED normally** — the launcher PID exits and the GPU frees on success exactly as on a crash, and
both JSONs + the terminal `ALL_CAMCOND_DONE` marker were present (`camcond.log`). Verified before
acting; no relaunch was needed (re-running a completed experiment would burn GPU-hours for nothing).
Root-cause class: **C2 (absence/termination from a single probe)** — "process gone + GPU 0% + file not
seen" is the SUCCESS signature as much as the failure one; the terminal marker + the output artifact are
the discriminating checks.
