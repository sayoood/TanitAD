# Deliverable manifest — Branch B held-out-rig transfer eval (2026-07-24)

**Stream:** the DECISIVE go/no-go for the own dynamics-encoder line — does the from-scratch,
all-block GAIA-2 camera-conditioned Branch B encoder (step 40k) recover cross-rig transfer?
**Operating rules:** STAGE, NEVER PUSH. Everything below is `git add`-ed into the working tree;
nothing committed, nothing pushed, no branch switch.

## Verdict (one line)

**FAIL, decisively.** Branch B does not recover cross-rig transfer (best cross-rig speed R²
**−0.667**, gate +0.9) and is a **weaker** dynamics substrate than the plain flagship-v1 encoder
(paired dR2 CI excludes 0, Branch B worse, on 3 of 4 arms; the 4th favours Branch B only via
episode leakage). Pre-registered outcome = **"≈ ablation → camera-conditioning insufficient"**,
and stronger: a regression. Full analysis: `RESULTS_branchB.md`.

## Follow-up (same session, coordinator-requested) — the own-encoder PIVOT evidence

`v1-encoder-char/` characterizes **flagship-v1 frozen** as the cross-rig/IDM substrate for Sayed's
pivot call. **Verdict: v1 IS usable AS-IS (the CHEAP pivot) with a multi-domain readout head** —
the alarming rig-B −1.169 was a head-diversity artifact; a rig-A+comma head gives rig-B speed R²
**+0.657** / yaw **+0.504** (both >0.5), and v1 transfers speed cross-geometry-class to comma
(+0.585 ≈ comma's in-domain ceiling 0.592) and is aug-robust. Bounded residual: cross-class **yaw**
transfer is untestable on comma (comma yaw unreadable in-domain — C6). Branch B's aug caveat is
**closed** (weakness is real, not augmentation). See `v1-encoder-char/RESULTS_v1_encoder_char.md`.

## Escalation (integration) — read first

0. **The own-encoder / GAIA-2 camera-conditioning path is NOT validated at 40k/2466-clip scale.**
   Both the cheap warm-start ablation (Branch A, −2.1) and the expensive from-scratch Branch B
   are refuted. The YouTube-scale IDM-pretraining thesis that rests on this encoder is **not
   supported** by this result. Any further encoder-line spend (Plücker/PRoPE, YouTube pretrain)
   should be re-pre-registered against the new evidence.
1. **`MODEL_REGISTRY.md` update needed (do not silently edit — flagged for the registry owner):**
   register `dynenc-branchB` step 40000 with its transfer verdict (FAIL; cross-rig speed R²
   −0.667 best; own-head in-sample rig-B 0.24; weaker than flagship-v1 frozen paired). The
   `BRANCHB_LAUNCH.md` "Branch B is the go" and `LAUNCH_PLAN.md §4` framing now have their
   MEASURED answer and should be annotated FAIL.
2. **`RETRACTION_LOG.md` candidate class:** none retracted (this is a fresh pre-registered
   result, reported plainly per rule 5), BUT it re-demonstrates **C5** — cross-rig R² is highly
   sensitive to head-fit convergence (e10→e50 swings of 1–3.5 pts). The decision-grade read uses
   a **converged** head (flagship in-domain ~0.9 verified) + the paired same-regime contrast.
3. **New reusable code lives in `stack/scripts/run_branchb_transfer.py`** (single source of
   truth, not stranded in this incoming dir).

## Artifacts

| artifact | where it lives | only copy? | notes |
|---|---|---|---|
| Result note (decisive verdict + controls + caveats) | `repo: …/incoming/2026-07-24-branchb-transfer-eval/RESULTS_branchB.md` | yes → staged | the analysis |
| Transfer-eval runner | `repo: stack/scripts/run_branchb_transfer.py` **+** `…/incoming/…/run_branchb_transfer.py` (copy) | no (also pod3:/workspace/tmp) | reuses camcond downstream probe; swaps in Branch B encoder + frozen flagship-v1 paired control |
| Raw JSON — CONVERGED (decision-grade) | `repo: …/incoming/…/results_branchb_transfer_e50_CONVERGED.json` | no (also pod3:/workspace/tmp/branchb_eval) | epochs=50; flagship in-domain 0.86–0.91 = converged |
| Raw JSON — UNDERFIT (the C5 lesson) | `repo: …/incoming/…/results_branchb_transfer_e10_UNDERFIT.json` | no (also pod3) | epochs=10; in-domain negative → shows head-fit sensitivity |
| This manifest | `repo: …/incoming/…/MANIFEST.md` | yes → staged | |

## On pods (NOT the only copy of anything below — all durable)

- **Branch B ckpt** (the model under test): `pod3:/workspace/experiments/dynenc-branchB/ckpt.pt`
  (step 40000, md5 `a0d7e7c1…`) + `history.json` + `milestone_step2000.pt`. Durable MooseFS;
  pod3 dd write test passed (500 MB @ 389 MB/s → space available).
- **Cached latents** (branchB + flagshipv1, all eval clips): `pod3:/workspace/tmp/branchb_eval/`.
- **Logs:** `pod3:/tmp/branchb_transfer.log` (e10), `/tmp/branchb_transfer_e50.log` (e50).

## NOT done

- **P2 — HF backup push of the Branch B ckpt to gated `Sayood/tanitad-dynenc-branchB`:**
  **NOT done.** The push requires staging the HF write-token to pod3 (pod3 has no HF auth
  configured); the credential-movement step was **gated by the safety classifier** and I did not
  work around it. The ckpt is preserved on pod3 + durable MooseFS (verified space), so it is not
  the only copy. **Action for Sayed/user:** authorize the token handling, or push from a box with
  HF auth. (Runner precedent: the 3-arm `push_ckpt.py` flow.)
